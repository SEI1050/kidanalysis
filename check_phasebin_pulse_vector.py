#!/usr/bin/env python3
"""
check_phasebin_pulse_vector.py

温度 1 Hz / レーザー 50 Hz のデータを event_index % 50 で折り畳み、
projected amplitude の周期変化が

(A) pulse IQ ベクトルの大きさ |ΔIQ| の変化
(B) pulse IQ ベクトルの回転により固定射影軸から外れる効果

のどちらによるかを確認する。

対象フォルダは TARGET_DIR_NAME と完全一致させるため、
_second / _third / _fourth / _fifth は読み込まない。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# ここだけ基本的に変える
# ============================================================

DATA_DATE = "20260527"

# "cloud" : OneDrive / CloudStorage側
# "local" : kidanalysis/data/20260527 側
# "both"  : 両方を探索（同じ測定を二重に読まない）
INPUT_MODE = "cloud"

# 14:27 first のみ
TARGET_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"
NPZ_PATTERN = "wf_*.npz"

# first の先頭から厳密に 1000 events
N_EVENTS_TO_USE = 1000
N_PHASE_BINS = 50

# pedestal範囲 [us]; Noneなら t<0 全部
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# pulse peak を探す範囲 [us]
AMP_WINDOW_US = (0.0, 1.5)

# 平均波形を描く範囲 [us]
PLOT_TIME_WINDOW_US = (-0.3, 2.0)
WAVEFORM_PLOT_EVERY = 5

DPI = 300


# ============================================================
# path setting: 既存コードと同じ方針
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "phasebin_pulse_vector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

local_data_dir = HERE / "data" / DATA_DATE

cloud_data_candidates = [
    Path.home() / "Library" / "CloudStorage" / "OneDrive-TheUniversityofTokyo"
    / "東京大学" / "4S" / "kidfit" / DATA_DATE,

    Path.home() / "OneDrive - The University of Tokyo"
    / "東京大学" / "4S" / "kidfit" / DATA_DATE,

    Path.home() / "Library" / "CloudStorage" / "OneDrive - The University of Tokyo"
    / "東京大学" / "4S" / "kidfit" / DATA_DATE,
]

EXTRA_INPUT_ROOTS = [
    # Path("/Users/kubokosei/Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit/20260527"),
]


# ============================================================
# utility
# ============================================================

def scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else x


def make_time_axis_s(npts, sample_rate_hz, ref_position_percent):
    return (
        np.arange(npts, dtype=float)
        - npts * ref_position_percent / 100.0
    ) / sample_rate_hz


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def normalized(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.array([1.0, 0.0])


def wrap_angle(theta):
    return np.angle(np.exp(1j * theta))


def p2p_over_mean(x):
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x)
    return (np.nanmax(x) - np.nanmin(x)) / abs(mean) if mean != 0 else np.nan


# ============================================================
# input search / loading
# ============================================================

def collect_input_roots():
    candidates = []

    if INPUT_MODE in ["local", "both"]:
        candidates.append(("local", local_data_dir))
    if INPUT_MODE in ["cloud", "both"]:
        candidates.extend(("cloud", p) for p in cloud_data_candidates)
    candidates.extend(("extra", p) for p in EXTRA_INPUT_ROOTS)

    print()
    print("===== path check =====")
    print("HERE:", HERE)
    print("INPUT_MODE:", INPUT_MODE)

    roots = []
    seen = set()
    for kind, p in candidates:
        p = Path(p).expanduser().resolve(strict=False)
        exists = p.is_dir()

        print(f"[{kind}] {p}")
        print("   exists:", exists)

        if exists and p.as_posix() not in seen:
            roots.append(p)
            seen.add(p.as_posix())

    if not roots:
        raise RuntimeError("入力フォルダが見つかりません。")

    return roots


def find_first_target_dir(roots):
    """
    exact match only.
    TARGET_DIR_NAME_second などは絶対に読まない。
    """
    found = []

    for root in roots:
        candidate = root / TARGET_DIR_NAME
        print("candidate:", candidate, "exists:", candidate.is_dir())
        if candidate.is_dir():
            found.append(candidate)

    if not found:
        raise RuntimeError(
            f"'{TARGET_DIR_NAME}' が見つかりません。フォルダ名とパス設定を確認してください。"
        )

    if len(found) > 1:
        print("WARNING: 同一 first folder が複数 root にあります。最初の一つだけ使います。")

    return found[0]


def load_first_n_events(meas_dir):
    npz_files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    if not npz_files:
        raise RuntimeError(f"{meas_dir} に {NPZ_PATTERN} がありません。")

    print()
    print("===== selected measurement =====")
    print("measurement dir:", meas_dir)
    print("number of npz files:", len(npz_files))

    all_ch0 = []
    all_ch1 = []
    used_files = []
    time_ref = None
    n_loaded = 0

    for f in npz_files:
        if n_loaded >= N_EVENTS_TO_USE:
            break

        try:
            data = np.load(f)
        except Exception as e:
            print("skip load error:", f, e)
            continue

        required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        missing = [k for k in required if k not in data.files]
        if missing:
            print("skip missing keys:", f, missing)
            continue

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)

        if ch0.ndim == 1:
            ch0 = ch0[None, :]
        if ch1.ndim == 1:
            ch1 = ch1[None, :]

        if ch0.shape != ch1.shape:
            print("skip shape mismatch:", f)
            continue

        npts = int(scalar(data["npts"]))
        sr = float(scalar(data["sample_rate"]))
        ref_position = float(scalar(data["ref_position"]))

        if ch0.shape[1] != npts:
            print("skip npts mismatch:", f)
            continue

        time_s = make_time_axis_s(npts, sr, ref_position)
        if time_ref is None:
            time_ref = time_s
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time axis mismatch:", f)
            continue

        n_take = min(len(ch0), N_EVENTS_TO_USE - n_loaded)
        all_ch0.append(ch0[:n_take])
        all_ch1.append(ch1[:n_take])
        used_files.append(f)
        n_loaded += n_take

        print(f"load: {f.name} -> {n_take} events (total={n_loaded})")

    if time_ref is None:
        raise RuntimeError("有効な waveform を読み込めませんでした。")
    if n_loaded != N_EVENTS_TO_USE:
        raise RuntimeError(
            f"{N_EVENTS_TO_USE} events 必要ですが、{n_loaded} events しか読めませんでした。"
        )

    return time_ref, np.vstack(all_ch0), np.vstack(all_ch1), used_files


# ============================================================
# phase-bin pulse vector analysis
# ============================================================

def analyze_phase_bins(time_s, ch0, ch1):
    """
    各イベントで pedestal を引いた後、
    全イベントの平均IQ波形で |ΔIQ| が最大となる時刻を一つ決める。
    その共通時刻で、phase bin ごとの pulse vector を比較する。

    これにより
      A_2D = |<ΔIQ>|
      A_parallel = <ΔIQ> · u_global
      A_perp = <ΔIQ> · v_global
      theta = atan2(<ΔQ>, <ΔI>)
    を同じタイミングで比較する。
    """
    time_us = time_s * 1e6

    if BASELINE_WINDOW_US is None:
        baseline_mask = time_us < 0
    else:
        lo, hi = BASELINE_WINDOW_US
        baseline_mask = (time_us >= lo) & (time_us <= hi)

    amp_mask = (time_us >= AMP_WINDOW_US[0]) & (time_us <= AMP_WINDOW_US[1])

    if baseline_mask.sum() < 3:
        raise ValueError("baseline points が少なすぎます。")
    if amp_mask.sum() < 3:
        raise ValueError("AMP_WINDOW_US 内の sample 点が少なすぎます。")

    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    dch0 = ch0 - ped0[:, None]
    dch1 = ch1 - ped1[:, None]

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)
    mean_r = np.hypot(mean0, mean1)

    amp_indices = np.where(amp_mask)[0]
    idx_peak = amp_indices[np.argmax(mean_r[amp_mask])]
    t_peak_us = time_us[idx_peak]

    u_global = normalized([mean0[idx_peak], mean1[idx_peak]])
    v_global = np.array([-u_global[1], u_global[0]])
    theta_global = np.arctan2(u_global[1], u_global[0])

    # Event-level definitions; compare with previous amp_projected.
    projected_waveform = dch0 * u_global[0] + dch1 * u_global[1]
    two_d_waveform = np.hypot(dch0, dch1)

    amp_proj_event = np.max(projected_waveform[:, amp_mask], axis=1)
    amp_2d_event = np.max(two_d_waveform[:, amp_mask], axis=1)

    phase = np.arange(len(ch0), dtype=int) % N_PHASE_BINS
    rows = []
    bin_waveforms = {}

    for b in range(N_PHASE_BINS):
        event_idx = np.where(phase == b)[0]
        if len(event_idx) == 0:
            continue

        # Event-by-event pulse vectors evaluated at the identical global peak time.
        vectors = np.column_stack([
            dch0[event_idx, idx_peak],
            dch1[event_idx, idx_peak],
        ])
        vec = vectors.mean(axis=0)

        a2d = float(np.linalg.norm(vec))
        apar = float(vec @ u_global)
        aperp = float(vec @ v_global)
        theta = float(np.arctan2(vec[1], vec[0]))
        theta_rel = float(wrap_angle(theta - theta_global))

        u_bin = normalized(vec)
        v_bin = np.array([-u_bin[1], u_bin[0]])

        # First-order errors from event-by-event scatter.
        a2d_sem = sem(vectors @ u_bin)
        apar_sem = sem(vectors @ u_global)
        aperp_sem = sem(vectors @ v_global)
        theta_sem_rad = sem(vectors @ v_bin) / a2d if a2d > 0 else np.nan

        ped = np.array([ped0[event_idx].mean(), ped1[event_idx].mean()])
        tip = ped + vec

        rows.append({
            "phase_bin": b,
            "n_events": len(event_idx),

            "ped0_mean_V": ped[0],
            "ped1_mean_V": ped[1],
            "ped0_sem_V": sem(ped0[event_idx]),
            "ped1_sem_V": sem(ped1[event_idx]),

            "pulse_dch0_at_global_peak_V": vec[0],
            "pulse_dch1_at_global_peak_V": vec[1],
            "pulse_tip_ch0_V": tip[0],
            "pulse_tip_ch1_V": tip[1],

            "amp_2d_meanwave_V": a2d,
            "amp_2d_meanwave_sem_V": a2d_sem,
            "amp_projected_meanwave_V": apar,
            "amp_projected_meanwave_sem_V": apar_sem,
            "amp_perpendicular_meanwave_V": aperp,
            "amp_perpendicular_meanwave_sem_V": aperp_sem,

            "pulse_theta_rad": theta,
            "pulse_theta_rel_rad": theta_rel,
            "pulse_theta_rel_deg": np.degrees(theta_rel),
            "pulse_theta_sem_deg": np.degrees(theta_sem_rad),

            "amp_projected_event_peak_mean_V": np.mean(amp_proj_event[event_idx]),
            "amp_projected_event_peak_sem_V": sem(amp_proj_event[event_idx]),
            "amp_2d_event_peak_mean_V": np.mean(amp_2d_event[event_idx]),
            "amp_2d_event_peak_sem_V": sem(amp_2d_event[event_idx]),
        })

        bin_waveforms[b] = {
            "dch0": dch0[event_idx].mean(axis=0),
            "dch1": dch1[event_idx].mean(axis=0),
            "proj": projected_waveform[event_idx].mean(axis=0),
        }

    df = pd.DataFrame(rows).sort_values("phase_bin").reset_index(drop=True)

    meta = {
        "time_us": time_us,
        "idx_peak": idx_peak,
        "t_peak_us": t_peak_us,
        "u_global": u_global,
        "v_global": v_global,
        "theta_global": theta_global,
        "bin_waveforms": bin_waveforms,
    }
    return df, meta


# ============================================================
# plots
# ============================================================

def draw_phase_arrows(ax, x, y):
    for i in range(len(x)):
        j = (i + 1) % len(x)
        ax.annotate(
            "",
            xy=(x[j], y[j]),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", lw=0.7, alpha=0.55),
        )


def plot_summary(df, meta, output_path):
    bins = df["phase_bin"].to_numpy()

    ped0 = df["ped0_mean_V"].to_numpy() * 1e3
    ped1 = df["ped1_mean_V"].to_numpy() * 1e3
    tip0 = df["pulse_tip_ch0_V"].to_numpy() * 1e3
    tip1 = df["pulse_tip_ch1_V"].to_numpy() * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)

    # IQ trajectory with phase-dependent pulse vectors.
    ax = axes[0, 0]
    sc = ax.scatter(ped0, ped1, c=bins, s=50, zorder=4, label="pedestal")
    ax.scatter(
        tip0, tip1, c=bins, s=35, marker="x",
        alpha=0.9, label=f"pulse tip at t={meta['t_peak_us']:.3f} us",
    )
    draw_phase_arrows(ax, ped0, ped1)

    for i in range(len(bins)):
        ax.annotate(
            "",
            xy=(tip0[i], tip1[i]),
            xytext=(ped0[i], ped1[i]),
            arrowprops=dict(arrowstyle="->", lw=0.9, alpha=0.55),
        )

    for i, b in enumerate(bins):
        if b % 5 == 0:
            ax.annotate(
                str(b), (ped0[i], ped1[i]),
                xytext=(4, 4), textcoords="offset points", fontsize=8,
            )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("phase bin")
    ax.set_xlabel("ch0 [mV]")
    ax.set_ylabel("ch1 [mV]")
    ax.set_title("Pedestal trajectory + phase-binned pulse vectors")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)

    # 2D amplitude vs fixed projection.
    ax = axes[0, 1]
    ax.errorbar(
        bins,
        df["amp_2d_meanwave_V"] * 1e3,
        yerr=df["amp_2d_meanwave_sem_V"] * 1e3,
        marker="o", lw=1.3, capsize=2.5,
        label=r"$|\langle\Delta IQ\rangle|$",
    )
    ax.errorbar(
        bins,
        df["amp_projected_meanwave_V"] * 1e3,
        yerr=df["amp_projected_meanwave_sem_V"] * 1e3,
        marker="s", lw=1.3, capsize=2.5,
        label=r"$\langle\Delta IQ\rangle\cdot \hat{u}_{global}$",
    )
    ax.set_xlabel("phase bin = event index mod 50")
    ax.set_ylabel("pulse amplitude [mV]")
    ax.set_title("Magnitude change or fixed-axis projection effect?")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Rotation relative to global pulse direction.
    ax = axes[1, 0]
    ax.errorbar(
        bins,
        df["pulse_theta_rel_deg"],
        yerr=df["pulse_theta_sem_deg"],
        marker="o", lw=1.3, capsize=2.5,
        label=r"$\theta_{\rm pulse}-\theta_{\rm global}$",
    )
    ax.axhline(0.0, ls="--", lw=1.0, label="global projection direction")
    ax.set_xlabel("phase bin = event index mod 50")
    ax.set_ylabel("pulse-angle shift [deg]")
    ax.set_title("Does the pulse-vector direction rotate?")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Parallel / perpendicular components.
    ax = axes[1, 1]
    ax.errorbar(
        bins,
        df["amp_projected_meanwave_V"] * 1e3,
        yerr=df["amp_projected_meanwave_sem_V"] * 1e3,
        marker="o", lw=1.3, capsize=2.5,
        label=r"parallel: $\Delta IQ\cdot\hat{u}_{global}$",
    )
    ax.errorbar(
        bins,
        df["amp_perpendicular_meanwave_V"] * 1e3,
        yerr=df["amp_perpendicular_meanwave_sem_V"] * 1e3,
        marker="s", lw=1.3, capsize=2.5,
        label=r"perpendicular: $\Delta IQ\cdot\hat{v}_{global}$",
    )
    ax.axhline(0.0, ls="--", lw=1.0)
    ax.set_xlabel("phase bin = event index mod 50")
    ax.set_ylabel("pulse component [mV]")
    ax.set_title("Pulse-vector decomposition in fixed global axes")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{TARGET_DIR_NAME} (first only): phase-bin pulse-vector check\n"
        rf"global peak = {meta['t_peak_us']:.3f} us, "
        rf"$\hat{{u}}=({meta['u_global'][0]:+.3f}, {meta['u_global'][1]:+.3f})$",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_waveforms(df, meta, output_path):
    time_us = meta["time_us"]
    draw_mask = (
        (time_us >= PLOT_TIME_WINDOW_US[0])
        & (time_us <= PLOT_TIME_WINDOW_US[1])
    )

    selected = [
        int(b) for b in df["phase_bin"].to_numpy()
        if int(b) % WAVEFORM_PLOT_EVERY == 0
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True, constrained_layout=True)

    for b in selected:
        wf = meta["bin_waveforms"][b]
        axes[0].plot(time_us[draw_mask], wf["dch0"][draw_mask] * 1e3, label=f"bin {b}")
        axes[1].plot(time_us[draw_mask], wf["dch1"][draw_mask] * 1e3, label=f"bin {b}")
        axes[2].plot(time_us[draw_mask], wf["proj"][draw_mask] * 1e3, label=f"bin {b}")

    axes[0].set_ylabel(r"$\langle\Delta$ch0$\rangle$ [mV]")
    axes[1].set_ylabel(r"$\langle\Delta$ch1$\rangle$ [mV]")
    axes[2].set_ylabel("global projection [mV]")
    axes[2].set_xlabel("time from trigger [us]")

    for ax in axes:
        ax.axvline(meta["t_peak_us"], ls="--", lw=1.0)
        ax.grid(True)

    axes[0].set_title("Phase-binned average waveforms")
    axes[0].legend(
        ncols=2, fontsize=8,
        bbox_to_anchor=(1.02, 1.0), loc="upper left",
    )

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# console interpretation
# ============================================================

def print_summary(df, meta):
    a2d = df["amp_2d_meanwave_V"].to_numpy()
    aproj = df["amp_projected_meanwave_V"].to_numpy()
    theta_deg = df["pulse_theta_rel_deg"].to_numpy()

    a2d_evt = df["amp_2d_event_peak_mean_V"].to_numpy()
    aproj_evt = df["amp_projected_event_peak_mean_V"].to_numpy()

    a2d_p2p = p2p_over_mean(a2d)
    aproj_p2p = p2p_over_mean(aproj)
    theta_span = np.nanmax(theta_deg) - np.nanmin(theta_deg)

    print()
    print("===== phase-bin pulse-vector result =====")
    print(f"global peak time = {meta['t_peak_us']:.6f} us")
    print(
        "global pulse direction "
        f"u = ({meta['u_global'][0]:+.5f}, {meta['u_global'][1]:+.5f})"
    )
    print()
    print("common-peak mean-waveform metrics:")
    print(f"  2D amplitude p2p / mean       = {100*a2d_p2p:.2f} %")
    print(f"  projected amplitude p2p / mean = {100*aproj_p2p:.2f} %")
    print(f"  pulse angle span               = {theta_span:.2f} deg")
    print()
    print("event-level peak metrics:")
    print(f"  2D event-peak p2p / mean       = {100*p2p_over_mean(a2d_evt):.2f} %")
    print(f"  projected event-peak p2p/mean  = {100*p2p_over_mean(aproj_evt):.2f} %")
    print()

    # Heuristic guide only, not a physical model fit.
    if aproj_p2p > 1.5 * a2d_p2p and theta_span > 3.0:
        print("interpretation:")
        print("  projected amplitude changes more than true 2D length,")
        print("  and the pulse vector rotates.")
        print("  -> fixed projection-axis mismatch is an important contribution.")
    elif a2d_p2p > 0.05:
        print("interpretation:")
        print("  true 2D pulse magnitude also changes substantially.")
        print("  -> responsivity / detuning / Q-related variation is present,")
        print("     possibly together with vector rotation.")
    else:
        print("interpretation:")
        print("  2D length is comparatively stable at the common peak.")
        print("  -> inspect angle and perpendicular-component plots for rotation.")


# ============================================================
# main
# ============================================================

def main():
    print("DATA_DATE:", DATA_DATE)
    print("TARGET_DIR_NAME:", TARGET_DIR_NAME)
    print("N_EVENTS_TO_USE:", N_EVENTS_TO_USE)
    print("N_PHASE_BINS:", N_PHASE_BINS)
    print("OUT_DIR:", OUT_DIR)

    roots = collect_input_roots()
    meas_dir = find_first_target_dir(roots)
    time_s, ch0, ch1, used_files = load_first_n_events(meas_dir)

    df, meta = analyze_phase_bins(time_s, ch0, ch1)

    summary_png = OUT_DIR / "phasebin_pulse_vector_summary.png"
    waveforms_png = OUT_DIR / "phasebin_average_IQ_waveforms.png"
    summary_csv = OUT_DIR / "phasebin_pulse_vector_summary.csv"
    info_txt = OUT_DIR / "run_info.txt"

    plot_summary(df, meta, summary_png)
    plot_waveforms(df, meta, waveforms_png)
    df.to_csv(summary_csv, index=False)

    info_txt.write_text(
        "\n".join([
            f"measurement_dir = {meas_dir}",
            f"N_EVENTS_TO_USE = {N_EVENTS_TO_USE}",
            f"N_PHASE_BINS = {N_PHASE_BINS}",
            f"BASELINE_WINDOW_US = {BASELINE_WINDOW_US}",
            f"AMP_WINDOW_US = {AMP_WINDOW_US}",
            f"global_peak_time_us = {meta['t_peak_us']}",
            f"global_direction_ch0 = {meta['u_global'][0]}",
            f"global_direction_ch1 = {meta['u_global'][1]}",
            "",
            "used_files:",
            *[str(p) for p in used_files],
        ]),
        encoding="utf-8",
    )

    print_summary(df, meta)

    print()
    print("saved:")
    print(" ", summary_png)
    print(" ", waveforms_png)
    print(" ", summary_csv)
    print(" ", info_txt)


if __name__ == "__main__":
    main()
