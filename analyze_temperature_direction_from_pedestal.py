#!/usr/bin/env python3
"""
analyze_temperature_direction_from_pedestal.py

1 Hz の温度周期について、phase bin ごとの pedestal の移動方向と、
同じ phase bin における laser pulse の IQ ベクトルを比較する。

仮定:
laser pulse は局所的な加熱であり、温度上昇と同じ主な KID 応答方向を与える。

z_ped(b): phase bin b の pedestal median
p(b):     phase bin b の laser pulse vector
v_T(b):   phase bin に沿った pedestal の中央差分

alignment = Re[v_T p*] / (|v_T||p|)

alignment > 0 : phase 増加方向が laser heating と同方向 -> warming candidate
alignment < 0 : 逆方向 -> cooling candidate

この判定は additive offset と、一定の complex rotation/scale に頑健。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DATE = "20260527"
TARGET_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"
NPZ_PATTERN = "wf_*.npz"

N_EVENTS_TO_USE = 1000
EVENTS_PER_TEMP_CYCLE = 50
N_PHASE_BINS = 50         # 50なら各bin=20 events。10にすると見やすく粗くなる。
assert EVENTS_PER_TEMP_CYCLE % N_PHASE_BINS == 0

BASELINE_WINDOW_US = None  # None -> t<0
AMP_WINDOW_US = (0.0, 1.5)

# phase trajectoryの微分はノイズに敏感なので、3または5を推奨
PEDESTAL_SMOOTHING_WIDTH = 3  # odd number only

# 各phaseのpulseの局所ピークを使う。Falseなら全bin共通peak時刻。
USE_LOCAL_PULSE_PEAK = True

# 速度が小さい折返し点などを ambiguous にする条件
MIN_RELATIVE_SPEED = 0.20
MIN_ABS_ALIGNMENT = 0.25

DPI = 300


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / DATA_DATE / f"temperature_direction_{N_PHASE_BINS}bin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_ROOTS = [
    Path("/Volumes/NO NAME/data") / DATA_DATE,
    Path.home() / "Library" / "CloudStorage" / "OneDrive-TheUniversityofTokyo"
    / "東京大学" / "4S" / "kidfit" / DATA_DATE,
    Path.home() / "OneDrive - The University of Tokyo"
    / "東京大学" / "4S" / "kidfit" / DATA_DATE,
    HERE / "data" / DATA_DATE,
]


# =============================================================================
# HELPERS
# =============================================================================

def scalar(x):
    a = np.asarray(x)
    return a.item() if a.size == 1 else x


def time_axis_s(npts, sample_rate_hz, ref_percent):
    return (np.arange(npts, dtype=float) - npts * ref_percent / 100.0) / sample_rate_hz


def normalize(z):
    return z / abs(z) if abs(z) > 0 else np.nan + 1j * np.nan


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def periodic_smooth(z, width):
    z = np.asarray(z, dtype=complex)
    if width <= 1:
        return z.copy()
    if width % 2 == 0:
        raise ValueError("PEDESTAL_SMOOTHING_WIDTH must be odd.")
    h = width // 2
    out = np.zeros_like(z)
    for shift in range(-h, h + 1):
        out += np.roll(z, shift)
    return out / width


def centered_cyclic_diff(z):
    return 0.5 * (np.roll(z, -1) - np.roll(z, +1))


# =============================================================================
# LOADING
# =============================================================================

def find_data_dir():
    print("\n===== input roots =====")
    candidates = []
    for root in INPUT_ROOTS:
        root = Path(root)
        candidate = root / TARGET_DIR_NAME
        n = len(list(candidate.glob(NPZ_PATTERN))) if candidate.is_dir() else 0
        print(root, "exists=", root.is_dir(), "| candidate wf npz =", n)
        if n > 0:
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            f"{TARGET_DIR_NAME} containing {NPZ_PATTERN} was not found."
        )

    if len(candidates) > 1:
        print("WARNING: duplicate folders found; use first valid root.")
    print("selected:", candidates[0])
    return candidates[0]


def load_events(meas_dir):
    files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    b0, b1, used = [], [], []
    t_ref = None
    n = 0

    for path in files:
        if n >= N_EVENTS_TO_USE:
            break

        d = np.load(path)
        required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        if any(k not in d.files for k in required):
            print("skip missing keys:", path.name)
            continue

        c0 = np.asarray(d["ch0"], dtype=float)
        c1 = np.asarray(d["ch1"], dtype=float)
        if c0.ndim == 1:
            c0 = c0[None, :]
        if c1.ndim == 1:
            c1 = c1[None, :]
        if c0.shape != c1.shape:
            print("skip shape mismatch:", path.name)
            continue

        t = time_axis_s(
            int(scalar(d["npts"])),
            float(scalar(d["sample_rate"])),
            float(scalar(d["ref_position"])),
        )
        if t_ref is None:
            t_ref = t
        elif len(t) != len(t_ref) or not np.allclose(t, t_ref):
            print("skip time-axis mismatch:", path.name)
            continue

        ntake = min(len(c0), N_EVENTS_TO_USE - n)
        b0.append(c0[:ntake])
        b1.append(c1[:ntake])
        used.append(path)
        n += ntake
        print(f"load {path.name}: {ntake} events (total={n})")

    if t_ref is None or n == 0:
        raise RuntimeError("No waveform data loaded.")

    if n < N_EVENTS_TO_USE:
        print(f"WARNING: requested {N_EVENTS_TO_USE}; loaded {n}")

    return t_ref, np.vstack(b0), np.vstack(b1), used


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze(t_s, ch0, ch1):
    t_us = t_s * 1e6

    if BASELINE_WINDOW_US is None:
        bmask = t_us < 0.0
    else:
        lo, hi = BASELINE_WINDOW_US
        bmask = (t_us >= lo) & (t_us <= hi)

    amask = (t_us >= AMP_WINDOW_US[0]) & (t_us <= AMP_WINDOW_US[1])

    if bmask.sum() < 3 or amask.sum() < 3:
        raise RuntimeError("Baseline or amplitude window is too short.")

    ped0 = ch0[:, bmask].mean(axis=1)
    ped1 = ch1[:, bmask].mean(axis=1)

    d0 = ch0 - ped0[:, None]
    d1 = ch1 - ped1[:, None]

    total_mean0 = d0.mean(axis=0)
    total_mean1 = d1.mean(axis=0)
    total_a2d = np.hypot(total_mean0, total_mean1)
    amp_indices = np.where(amask)[0]
    global_peak = amp_indices[np.argmax(total_a2d[amask])]

    events_per_bin = EVENTS_PER_TEMP_CYCLE // N_PHASE_BINS
    phase50 = np.arange(len(ch0)) % EVENTS_PER_TEMP_CYCLE
    phasebin = phase50 // events_per_bin

    rows = []
    waveforms = {}

    for b in range(N_PHASE_BINS):
        idx = np.where(phasebin == b)[0]
        if len(idx) == 0:
            continue

        zped_events = ped0[idx] + 1j * ped1[idx]
        rmed, rq25, rq75 = median_iqr(zped_events.real)
        imed, iq25, iq75 = median_iqr(zped_events.imag)

        m0 = d0[idx].mean(axis=0)
        m1 = d1[idx].mean(axis=0)
        ma2d = np.hypot(m0, m1)

        if USE_LOCAL_PULSE_PEAK:
            pidx = amp_indices[np.argmax(ma2d[amask])]
        else:
            pidx = global_peak

        p = complex(m0[pidx], m1[pidx])

        rows.append({
            "phase_bin": b,
            "laser_phase_start": int(b * events_per_bin),
            "laser_phase_end": int((b + 1) * events_per_bin - 1),
            "n_events": len(idx),

            "ped_ch0_median_V": rmed,
            "ped_ch0_q25_V": rq25,
            "ped_ch0_q75_V": rq75,
            "ped_ch1_median_V": imed,
            "ped_ch1_q25_V": iq25,
            "ped_ch1_q75_V": iq75,

            "pulse_peak_time_us": t_us[pidx],
            "pulse_dch0_V": p.real,
            "pulse_dch1_V": p.imag,
            "pulse_length_V": abs(p),
            "pulse_angle_deg": np.degrees(np.angle(p)),
        })

        waveforms[b] = (m0, m1, ma2d)

    df = pd.DataFrame(rows).sort_values("phase_bin").reset_index(drop=True)

    zped = df["ped_ch0_median_V"].to_numpy() + 1j * df["ped_ch1_median_V"].to_numpy()
    zsmooth = periodic_smooth(zped, PEDESTAL_SMOOTHING_WIDTH)
    vtemp = centered_cyclic_diff(zsmooth)

    p = df["pulse_dch0_V"].to_numpy() + 1j * df["pulse_dch1_V"].to_numpy()
    speed = abs(vtemp)
    plen = abs(p)

    dot = np.real(vtemp * np.conj(p))
    alignment = np.full(len(df), np.nan)
    good = (speed > 0) & (plen > 0)
    alignment[good] = dot[good] / (speed[good] * plen[good])

    projection = np.real(vtemp * np.conj(np.array([normalize(q) for q in p])))

    nonzero_speed = speed[speed > 0]
    speed_ref = np.median(nonzero_speed)
    speed_threshold = MIN_RELATIVE_SPEED * speed_ref

    labels = []
    for sp, al, pr in zip(speed, alignment, projection):
        if (not np.isfinite(al)) or sp < speed_threshold or abs(al) < MIN_ABS_ALIGNMENT:
            labels.append("ambiguous / near turning point")
        elif pr > 0:
            labels.append("warming candidate")
        else:
            labels.append("cooling candidate")

    df["ped_smooth_ch0_V"] = zsmooth.real
    df["ped_smooth_ch1_V"] = zsmooth.imag
    df["ped_velocity_ch0_V_per_bin"] = vtemp.real
    df["ped_velocity_ch1_V_per_bin"] = vtemp.imag
    df["ped_speed_V_per_bin"] = speed
    df["laser_alignment_cosine"] = alignment
    df["projection_along_laser_vector_V_per_bin"] = projection
    df["temperature_trend_inferred"] = labels

    meta = {
        "t_us": t_us,
        "waveforms": waveforms,
        "global_peak_time_us": t_us[global_peak],
        "events_per_bin": events_per_bin,
        "speed_threshold": speed_threshold,
    }
    return df, meta


# =============================================================================
# PLOT
# =============================================================================

def arrow(ax, z0, z1, **kwargs):
    ax.annotate(
        "",
        xy=(z1.real, z1.imag),
        xytext=(z0.real, z0.imag),
        arrowprops=dict(arrowstyle="->", **kwargs),
    )

def legend_below(ax, handles=None, labels=None, ncol=2):
    """凡例を subplot 外の下側へ置く。"""
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()

    if len(handles) == 0:
        return

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        fontsize=7,
        frameon=True,
        borderaxespad=0.0,
    )

def plot(df, meta, path):
    b = df["phase_bin"].to_numpy(dtype=int)
    z = df["ped_ch0_median_V"].to_numpy() + 1j * df["ped_ch1_median_V"].to_numpy()
    zs = df["ped_smooth_ch0_V"].to_numpy() + 1j * df["ped_smooth_ch1_V"].to_numpy()
    v = df["ped_velocity_ch0_V_per_bin"].to_numpy() + 1j * df["ped_velocity_ch1_V_per_bin"].to_numpy()
    p = df["pulse_dch0_V"].to_numpy() + 1j * df["pulse_dch1_V"].to_numpy()

    z *= 1e3
    zs *= 1e3
    v *= 1e3
    p *= 1e3

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 13),
        constrained_layout=True,
    )

    fig.set_constrained_layout_pads(
        h_pad=0.25,
        w_pad=0.12,
        hspace=0.22,
        wspace=0.16,
    )

    # IQ path
    ax = axes[0, 0]
    sc = ax.scatter(z.real, z.imag, c=b, s=60, zorder=4, label="pedestal median")
    ax.plot(zs.real, zs.imag, lw=1, alpha=0.6, label="smoothed pedestal")

    span = max(np.ptp(z.real), np.ptp(z.imag), 1e-6)
    for i in range(len(b)):
        j = (i + 1) % len(b)
        arrow(ax, z[i], z[j], lw=0.7, alpha=0.4)

        vu = normalize(v[i])
        pu = normalize(p[i])
        if np.isfinite(vu.real):
            arrow(ax, zs[i], zs[i] + 0.13 * span * vu, lw=1.8, color="black")
        if np.isfinite(pu.real):
            arrow(ax, zs[i], zs[i] + 0.13 * span * pu, lw=1.8, color="red")

        ax.annotate(str(b[i]), (z[i].real, z[i].imag), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)

    fig.colorbar(sc, ax=ax, label="phase bin")
    ax.plot([], [], color="black", label="pedestal velocity, phase increasing")
    ax.plot([], [], color="red", label="laser heating direction")
    ax.set_title("Direction test in raw ADC IQ plane")
    ax.set_xlabel("ch0 pedestal [mV]")
    ax.set_ylabel("ch1 pedestal [mV]")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    legend_below(ax, ncol=2)

    # alignment
    ax = axes[0, 1]
    alignment = df["laser_alignment_cosine"].to_numpy()
    projection = df["projection_along_laser_vector_V_per_bin"].to_numpy() * 1e3
    ax.plot(b, alignment, "o-", label="alignment cosine")
    ax.axhline(0, lw=1)
    ax.axhline(MIN_ABS_ALIGNMENT, ls="--", lw=1, label="classification threshold")
    ax.axhline(-MIN_ABS_ALIGNMENT, ls="--", lw=1)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("temperature phase bin")
    ax.set_ylabel(r"$\cos(v_T,p_{laser})$")
    ax.set_title("Positive means phase progression follows heating direction")
    ax.grid(True)

    axr = ax.twinx()
    axr.plot(b, projection, "s-", alpha=0.75, label="signed projection")
    axr.set_ylabel("projection [mV/bin]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()

    legend_below(
        ax,
        handles=h1 + h2,
        labels=l1 + l2,
        ncol=2,
    )

    # signed direction and speed
    ax = axes[1, 0]
    ax.plot(b, projection, "o-", label="along laser heating direction")
    ax.axhline(0, lw=1)
    ax.set_xlabel("temperature phase bin")
    ax.set_ylabel("signed pedestal displacement [mV/bin]")
    ax.set_title("Positive = warming candidate, negative = cooling candidate")
    ax.grid(True)

    axr = ax.twinx()
    speed = df["ped_speed_V_per_bin"].to_numpy() * 1e3
    axr.plot(b, speed, "s-", alpha=0.7, label="pedestal speed")
    axr.axhline(meta["speed_threshold"] * 1e3, ls="--", lw=1, label="turning threshold")
    axr.set_ylabel("speed [mV/bin]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()

    legend_below(
        ax,
        handles=h1 + h2,
        labels=l1 + l2,
        ncol=2,
    )

    # pulse reference
    ax = axes[1, 1]
    ax.plot(b, df["pulse_angle_deg"], "o-", label="pulse-vector angle")
    ax.set_xlabel("temperature phase bin")
    ax.set_ylabel("angle [deg]")
    ax.set_title("Laser pulse vectors used as local heating references")
    ax.grid(True)
    axr = ax.twinx()
    axr.plot(b, df["pulse_length_V"].to_numpy() * 1e3, "s-", alpha=0.75, label="pulse length")
    axr.set_ylabel("pulse length [mV]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()

    legend_below(
        ax,
        handles=h1 + h2,
        labels=l1 + l2,
        ncol=2,
    )

    fig.suptitle(
        f"Temperature-direction inference: pedestal velocity vs laser pulse\n"
        f"{TARGET_DIR_NAME}; {N_PHASE_BINS} phase bins; "
        f"global pulse peak={meta['global_peak_time_us']:.3f} us",
        fontsize=13,
    )
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.20,
    )
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Output:", OUT_DIR)
    print("target:", TARGET_DIR_NAME)
    print("N_PHASE_BINS:", N_PHASE_BINS)
    print("PEDESTAL_SMOOTHING_WIDTH:", PEDESTAL_SMOOTHING_WIDTH)

    d = find_data_dir()
    t, c0, c1, used = load_events(d)
    df, meta = analyze(t, c0, c1)

    png = OUT_DIR / "temperature_direction_alignment.png"
    csv = OUT_DIR / "temperature_direction_summary.csv"
    info = OUT_DIR / "temperature_direction_run_info.txt"

    plot(df, meta, png)
    df.to_csv(csv, index=False)

    info.write_text(
        "\n".join([
            f"measurement_dir = {d}",
            f"N_EVENTS = {len(c0)}",
            f"N_PHASE_BINS = {N_PHASE_BINS}",
            f"events_per_bin = {meta['events_per_bin']}",
            f"BASELINE_WINDOW_US = {BASELINE_WINDOW_US}",
            f"AMP_WINDOW_US = {AMP_WINDOW_US}",
            f"PEDESTAL_SMOOTHING_WIDTH = {PEDESTAL_SMOOTHING_WIDTH}",
            f"global_peak_time_us = {meta['global_peak_time_us']}",
            f"speed_threshold_V_per_bin = {meta['speed_threshold']}",
            "",
            "used_files:",
            *map(str, used),
        ]),
        encoding="utf-8",
    )

    print("\n===== inferred temperature direction =====")
    for _, r in df.iterrows():
        print(
            f"bin {int(r['phase_bin']):2d}: "
            f"align={r['laser_alignment_cosine']:+.3f}, "
            f"proj={r['projection_along_laser_vector_V_per_bin']*1e3:+.4f} mV/bin, "
            f"{r['temperature_trend_inferred']}"
        )

    print("\nsaved:")
    print(" ", png)
    print(" ", csv)
    print(" ", info)


if __name__ == "__main__":
    main()
