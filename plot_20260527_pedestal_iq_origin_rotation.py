#!/usr/bin/env python3
"""
plot_20260527_pedestal_iq_origin_rotation.py

20260527 / 5.476GHz_z=7.5mm_x=3.4mm の first data の pedestal を、
50 Hz laser / 1 Hz thermal cycle とみなして phase binning する。

仮定:
    z_raw = A exp(i phi0) S21
すなわち、raw IQ には原点中心の一定回転・一定倍率だけがある。

reference phase bin の pedestal を指定角度へ送る固定回転:
    z_rot = z_raw * exp[i(target_angle - arg(z_ref))]
をかけ、IQ trajectory を比較する。

注意:
- 固定回転は IQ 軌跡の向きをそろえるだけで、平行移動はしない。
- offset があれば、これだけでは共振点 (1-d, 0) は決められない。
- radius-normalized panel は、pure origin rotation 成分だけを可視化する。

出力:
  ~/software/kidanalysis/data/20260527/
    iq_temperature_track_originrot_XXbin_shiftYY/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DATE = "20260527"
TARGET_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"
NPZ_PATTERN = "wf_*.npz"

# None: all events in first data. Example: 1000 for 20 temperature cycles.
N_EVENTS_TO_USE = None

# 50 Hz laser / assumed 1 Hz thermal modulation
EVENTS_PER_THERMAL_CYCLE = 50

# Change this freely: 50, 25, 10, 5, 2, 1 ...
N_PHASE_BINS = 50
assert EVENTS_PER_THERMAL_CYCLE % N_PHASE_BINS == 0
EVENTS_PER_PHASE_BIN = EVENTS_PER_THERMAL_CYCLE // N_PHASE_BINS

# Shifts the event-index origin within the 50-event thermal cycle.
PHASE_OFFSET_EVENTS = 0

# False: z = ch0 + i ch1; True: z = ch0 - i ch1.
CONJUGATE_CH1 = False

# Reference bin after PHASE_OFFSET_EVENTS; fixed global rotation sends it here.
REFERENCE_PHASE_BIN = 0
REFERENCE_TARGET_ANGLE_DEG = 0.0  # 0: +I, 180: -I

LABEL_EVERY_N_BINS = 5
DRAW_PHASE_ARROWS = True
DPI = 300


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
OUT_DIR = (
    HERE / "data" / DATA_DATE
    / f"iq_temperature_track_originrot_{N_PHASE_BINS}bin_shift{PHASE_OFFSET_EVENTS:+03d}"
)
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


def make_time_axis_s(npts, sample_rate_hz, ref_position_percent):
    return (
        np.arange(npts, dtype=float)
        - npts * ref_position_percent / 100.0
    ) / sample_rate_hz


def median_iqr(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def circular_mean_angle(theta):
    theta = np.asarray(theta, dtype=float)
    theta = theta[np.isfinite(theta)]
    if len(theta) == 0:
        return np.nan
    return np.angle(np.mean(np.exp(1j * theta)))


def normalized_complex(z):
    return z / abs(z) if abs(z) > 0 else np.nan + 1j * np.nan


def legend_below(ax, handles=None, labels=None, ncol=2):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        fontsize=8,
        frameon=True,
        borderaxespad=0.0,
    )


# =============================================================================
# DISCOVERY / STREAMING PEDESTAL EXTRACTION
# =============================================================================

def find_target_directory():
    print("\n===== input roots =====")
    candidates = []
    for root in INPUT_ROOTS:
        root = Path(root).expanduser()
        candidate = root / TARGET_DIR_NAME
        n_npz = len(list(candidate.glob(NPZ_PATTERN))) if candidate.is_dir() else 0
        print(root, "exists=", root.is_dir(), "| target wf files =", n_npz)
        if n_npz > 0:
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError(f"Cannot find {TARGET_DIR_NAME!r} with {NPZ_PATTERN}.")
    if len(candidates) > 1:
        print("WARNING: duplicate valid folders; first root is used.")
    print("selected:", candidates[0])
    return candidates[0]


def load_pedestals(meas_dir):
    files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    ped0_blocks, ped1_blocks, used_files = [], [], []
    n_loaded = 0
    time_ref = None
    bmask = None

    for path in files:
        if N_EVENTS_TO_USE is not None and n_loaded >= N_EVENTS_TO_USE:
            break

        try:
            data = np.load(path)
        except Exception as exc:
            print("skip unreadable:", path.name, exc)
            continue

        needed = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        if any(k not in data.files for k in needed):
            print("skip missing keys:", path.name)
            continue

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)
        if ch0.ndim == 1:
            ch0 = ch0[None, :]
        if ch1.ndim == 1:
            ch1 = ch1[None, :]
        if ch0.shape != ch1.shape:
            print("skip shape mismatch:", path.name)
            continue

        time_s = make_time_axis_s(
            int(scalar(data["npts"])),
            float(scalar(data["sample_rate"])),
            float(scalar(data["ref_position"])),
        )

        if time_ref is None:
            time_ref = time_s
            bmask = time_s < 0.0
            if bmask.sum() < 3:
                raise RuntimeError("Too few t<0 points for pedestal.")
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time-axis mismatch:", path.name)
            continue

        n_take = len(ch0)
        if N_EVENTS_TO_USE is not None:
            n_take = min(n_take, N_EVENTS_TO_USE - n_loaded)

        ped0_blocks.append(ch0[:n_take, bmask].mean(axis=1))
        ped1_blocks.append(ch1[:n_take, bmask].mean(axis=1))
        used_files.append(path)
        n_loaded += n_take
        print(f"load: {path.name} -> {n_take} events (total={n_loaded})")

    if n_loaded == 0:
        raise RuntimeError("No valid pedestal data loaded.")
    return np.concatenate(ped0_blocks), np.concatenate(ped1_blocks), used_files


# =============================================================================
# PHASE BINNING / FIXED ROTATION
# =============================================================================

def phase_index(n_events):
    event_index = np.arange(n_events, dtype=int)
    phase50 = (event_index + PHASE_OFFSET_EVENTS) % EVENTS_PER_THERMAL_CYCLE
    phase_bin = phase50 // EVENTS_PER_PHASE_BIN
    return event_index, phase50, phase_bin


def summarize(z_raw, z_rot, z_unit):
    _, _, bin_index = phase_index(len(z_raw))
    rows = []

    for b in range(N_PHASE_BINS):
        idx = np.where(bin_index == b)[0]
        if len(idx) == 0:
            continue

        def stats(z):
            re, re25, re75 = median_iqr(z[idx].real)
            im, im25, im75 = median_iqr(z[idx].imag)
            return re, re25, re75, im, im25, im75

        raw = stats(z_raw)
        rot = stats(z_rot)
        unit = stats(z_unit)
        rad, rad25, rad75 = median_iqr(abs(z_raw[idx]))
        phase_mean = circular_mean_angle(np.angle(z_rot[idx]))

        rows.append({
            "phase_bin": b,
            "phase50_start": int(b * EVENTS_PER_PHASE_BIN),
            "phase50_end": int((b + 1) * EVENTS_PER_PHASE_BIN - 1),
            "n_events": len(idx),

            "raw_I_median_V": raw[0],
            "raw_I_q25_V": raw[1],
            "raw_I_q75_V": raw[2],
            "raw_Q_median_V": raw[3],
            "raw_Q_q25_V": raw[4],
            "raw_Q_q75_V": raw[5],

            "rot_I_median_V": rot[0],
            "rot_I_q25_V": rot[1],
            "rot_I_q75_V": rot[2],
            "rot_Q_median_V": rot[3],
            "rot_Q_q25_V": rot[4],
            "rot_Q_q75_V": rot[5],

            "unit_I_median_V": unit[0],
            "unit_I_q25_V": unit[1],
            "unit_I_q75_V": unit[2],
            "unit_Q_median_V": unit[3],
            "unit_Q_q25_V": unit[4],
            "unit_Q_q75_V": unit[5],

            "radius_median_V": rad,
            "radius_q25_V": rad25,
            "radius_q75_V": rad75,
            "relative_phase_mean_deg": np.degrees(phase_mean),
        })
    return pd.DataFrame(rows)


def draw_track(ax, x, y, bins, title, xlabel, ylabel, errors=None):
    if errors is not None:
        xlo, xhi, ylo, yhi = errors
        ax.errorbar(x, y, xerr=[xlo, xhi], yerr=[ylo, yhi],
                    fmt="none", alpha=0.4, zorder=2)

    sc = ax.scatter(x, y, c=bins, s=58, zorder=4, label="phase-bin median")
    ax.plot(x, y, lw=0.9, alpha=0.55, zorder=3)

    if DRAW_PHASE_ARROWS:
        for i in range(len(x)):
            j = (i + 1) % len(x)
            ax.annotate(
                "",
                xy=(x[j], y[j]),
                xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="->", lw=0.75, alpha=0.55),
            )

    for i, b in enumerate(bins):
        if b % LABEL_EVERY_N_BINS == 0:
            ax.annotate(str(b), (x[i], y[i]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    return sc


# =============================================================================
# PLOT
# =============================================================================

def plot_result(df, raw_reference_angle_deg, reference_radius_V, output_path):
    bins = df["phase_bin"].to_numpy(dtype=int)
    ref_idx = int(np.where(bins == REFERENCE_PHASE_BIN)[0][0])

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.0), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.34, w_pad=0.12, hspace=0.28, wspace=0.16)

    # raw
    ax = axes[0, 0]
    x = df["raw_I_median_V"].to_numpy() * 1e3
    y = df["raw_Q_median_V"].to_numpy() * 1e3
    e = (
        x - df["raw_I_q25_V"].to_numpy() * 1e3,
        df["raw_I_q75_V"].to_numpy() * 1e3 - x,
        y - df["raw_Q_q25_V"].to_numpy() * 1e3,
        df["raw_Q_q75_V"].to_numpy() * 1e3 - y,
    )
    sc = draw_track(ax, x, y, bins, "Raw pedestal IQ track",
                    "raw ch0 pedestal [mV]", "raw ch1 pedestal [mV]", e)
    fig.colorbar(sc, ax=ax, label="thermal phase bin")
    legend_below(ax, ncol=1)

    # global fixed rotation
    ax = axes[0, 1]
    x = df["rot_I_median_V"].to_numpy() * 1e3
    y = df["rot_Q_median_V"].to_numpy() * 1e3
    e = (
        x - df["rot_I_q25_V"].to_numpy() * 1e3,
        df["rot_I_q75_V"].to_numpy() * 1e3 - x,
        y - df["rot_Q_q25_V"].to_numpy() * 1e3,
        df["rot_Q_q75_V"].to_numpy() * 1e3 - y,
    )
    sc = draw_track(ax, x, y, bins,
                    "Fixed origin-centered phase correction",
                    "rotated I [mV]", "rotated Q [mV]", e)
    ax.scatter([x[ref_idx]], [y[ref_idx]], marker="*", s=210, zorder=7,
               label=f"reference bin {REFERENCE_PHASE_BIN}")
    ax.axhline(0.0, lw=1.0)
    ax.axvline(0.0, lw=1.0)
    fig.colorbar(sc, ax=ax, label="thermal phase bin")
    legend_below(ax, ncol=2)

    # normalized radius
    ax = axes[1, 0]
    x = df["unit_I_median_V"].to_numpy() * 1e3
    y = df["unit_Q_median_V"].to_numpy() * 1e3
    sc = draw_track(ax, x, y, bins,
                    "Pure-rotation projection: radius normalized",
                    "I after correction + normalization [mV]",
                    "Q after correction + normalization [mV]")
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    rr = reference_radius_V * 1e3
    ax.plot(rr * np.cos(theta), rr * np.sin(theta), lw=1.0,
            alpha=0.8, label=r"$|z|=|z_{\rm ref}|$")
    ax.axhline(0.0, lw=1.0)
    ax.axvline(0.0, lw=1.0)
    fig.colorbar(sc, ax=ax, label="thermal phase bin")
    legend_below(ax, ncol=2)

    # radius and angle
    ax = axes[1, 1]
    radius = df["radius_median_V"].to_numpy() * 1e3
    rlo = radius - df["radius_q25_V"].to_numpy() * 1e3
    rhi = df["radius_q75_V"].to_numpy() * 1e3 - radius
    ax.errorbar(bins, radius, yerr=[rlo, rhi], marker="o", capsize=3,
                label=r"$|z_{\rm ped}|$ median ± IQR")
    ax.axhline(rr, ls="--", lw=1.0, label="reference radius")
    ax.set_title("Diagnostic: origin radius and relative phase")
    ax.set_xlabel("thermal phase bin")
    ax.set_ylabel("origin radius [mV]")
    ax.grid(True)

    phase_deg = np.degrees(np.unwrap(np.radians(df["relative_phase_mean_deg"].to_numpy())))
    phase_deg -= phase_deg[ref_idx] - REFERENCE_TARGET_ANGLE_DEG
    axr = ax.twinx()
    axr.plot(bins, phase_deg, marker="s", label="phase after fixed rotation")
    axr.set_ylabel("relative phase [deg]")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    fig.suptitle(
        "20260527 pedestal track under origin-centered rotation assumption\n"
        f"{TARGET_DIR_NAME}; {N_PHASE_BINS} bins / 50 events; "
        f"phase offset={PHASE_OFFSET_EVENTS:+d}; "
        f"reference raw angle={raw_reference_angle_deg:.2f}°",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Output:", OUT_DIR)
    print("Target:", TARGET_DIR_NAME)
    print("N_EVENTS_TO_USE:", N_EVENTS_TO_USE)
    print("N_PHASE_BINS:", N_PHASE_BINS)
    print("PHASE_OFFSET_EVENTS:", PHASE_OFFSET_EVENTS)
    print("CONJUGATE_CH1:", CONJUGATE_CH1)
    print("REFERENCE_PHASE_BIN:", REFERENCE_PHASE_BIN)
    print("REFERENCE_TARGET_ANGLE_DEG:", REFERENCE_TARGET_ANGLE_DEG)

    if not 0 <= REFERENCE_PHASE_BIN < N_PHASE_BINS:
        raise ValueError("REFERENCE_PHASE_BIN must be within phase-bin range.")

    meas_dir = find_target_directory()
    ped0, ped1, used_files = load_pedestals(meas_dir)

    z_raw = ped0 + (-1j if CONJUGATE_CH1 else 1j) * ped1

    _, _, bin_index = phase_index(len(z_raw))
    idx_ref = np.where(bin_index == REFERENCE_PHASE_BIN)[0]
    z_ref = np.median(z_raw[idx_ref].real) + 1j * np.median(z_raw[idx_ref].imag)

    target = np.radians(REFERENCE_TARGET_ANGLE_DEG)
    rotation = np.exp(1j * (target - np.angle(z_ref)))
    z_rot = rotation * z_raw

    r_ref = abs(z_ref)
    z_unit = np.divide(
        r_ref * z_rot,
        np.abs(z_rot),
        out=np.full(z_rot.shape, np.nan + 1j * np.nan, dtype=complex),
        where=np.abs(z_rot) > 0,
    )

    df = summarize(z_raw, z_rot, z_unit)

    radii = abs(z_raw)
    r_med = np.median(radii)
    r_iqr = np.percentile(radii, 75) - np.percentile(radii, 25)
    frac_iqr = r_iqr / r_med if r_med > 0 else np.nan

    out_png = OUT_DIR / "pedestal_iq_origin_rotation.png"
    out_csv = OUT_DIR / "pedestal_iq_phase_summary.csv"
    out_json = OUT_DIR / "rotation_settings.json"
    out_info = OUT_DIR / "run_info.txt"

    plot_result(df, np.degrees(np.angle(z_ref)), r_ref, out_png)
    df.to_csv(out_csv, index=False)

    settings = {
        "target_dir_name": TARGET_DIR_NAME,
        "n_events_used": int(len(z_raw)),
        "events_per_thermal_cycle": EVENTS_PER_THERMAL_CYCLE,
        "n_phase_bins": N_PHASE_BINS,
        "events_per_phase_bin": EVENTS_PER_PHASE_BIN,
        "phase_offset_events": PHASE_OFFSET_EVENTS,
        "conjugate_ch1": CONJUGATE_CH1,
        "reference_phase_bin": REFERENCE_PHASE_BIN,
        "reference_target_angle_deg": REFERENCE_TARGET_ANGLE_DEG,
        "reference_raw_re_V": float(z_ref.real),
        "reference_raw_im_V": float(z_ref.imag),
        "reference_radius_V": float(r_ref),
        "reference_raw_angle_deg": float(np.degrees(np.angle(z_ref))),
        "fixed_rotation_deg": float(np.degrees(target - np.angle(z_ref))),
        "raw_radius_median_V": float(r_med),
        "raw_radius_IQR_V": float(r_iqr),
        "raw_radius_fractional_IQR": float(frac_iqr),
        "interpretation": (
            "Small fractional radius IQR supports a pure origin-centered rotation "
            "approximation. Large variation means rotation alone is insufficient."
        ),
    }
    out_json.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    out_info.write_text(
        "\n".join([
            f"measurement_dir = {meas_dir}",
            f"n_events_used = {len(z_raw)}",
            f"N_PHASE_BINS = {N_PHASE_BINS}",
            f"EVENTS_PER_PHASE_BIN = {EVENTS_PER_PHASE_BIN}",
            f"PHASE_OFFSET_EVENTS = {PHASE_OFFSET_EVENTS}",
            f"CONJUGATE_CH1 = {CONJUGATE_CH1}",
            f"REFERENCE_PHASE_BIN = {REFERENCE_PHASE_BIN}",
            f"REFERENCE_TARGET_ANGLE_DEG = {REFERENCE_TARGET_ANGLE_DEG}",
            "",
            "used_files:",
            *map(str, used_files),
        ]),
        encoding="utf-8",
    )

    print("\n===== rotation-only diagnostic =====")
    print(f"events used = {len(z_raw)}")
    print(f"reference raw pedestal = {z_ref.real*1e3:+.6f} + i {z_ref.imag*1e3:+.6f} mV")
    print(f"fixed rotation = {np.degrees(target - np.angle(z_ref)):+.4f} deg")
    print(f"raw radius median = {r_med*1e3:.6f} mV")
    print(f"raw radius IQR = {r_iqr*1e3:.6f} mV")
    print(f"raw radius fractional IQR = {frac_iqr:.5f}")
    print("\nsaved:")
    print(" ", out_png)
    print(" ", out_csv)
    print(" ", out_json)
    print(" ", out_info)


if __name__ == "__main__":
    main()
