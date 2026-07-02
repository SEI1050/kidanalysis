#!/usr/bin/env python3
"""
plot_20260527_pedestal_iq_pulseq0_left_simple.py

20260527 / 5.476GHz_z=7.5mm_x=3.4mm の first data について、
I=ch1, Q=ch0 とみなし、

- raw pedestal IQ track
- pulse-Q≈0 の phase を左端に持ってきた rotated pedestal IQ track
- ch1 pedestal (= raw I) vs thermal phase
- ch0 pedestal (= raw Q) vs thermal phase

の 4 パネルだけを描く簡易版。

出力:
  ~/software/kidanalysis/data/20260527/
    iq_temperature_track_pulseQ0left_simple_{N_PHASE_BINS}bin_shift{PHASE_OFFSET_EVENTS:+03d}/
      pedestal_iq_simple.png
      pedestal_iq_phase_summary.csv
      pulse_phase_summary.csv
      rotation_settings.json
      run_info.txt
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

N_EVENTS_TO_USE = None

EVENTS_PER_THERMAL_CYCLE = 50
N_PHASE_BINS = 50
assert EVENTS_PER_THERMAL_CYCLE % N_PHASE_BINS == 0
EVENTS_PER_PHASE_BIN = EVENTS_PER_THERMAL_CYCLE // N_PHASE_BINS

PHASE_OFFSET_EVENTS = 0

# I = ch1, Q = ch0
USE_CH1_AS_I_AND_CH0_AS_Q = True

PEAK_SEARCH_TMIN_US = 0.00
PEAK_SEARCH_TMAX_US = 0.40
PEAK_MODE = "global_peak"

REFERENCE_PHASE_BIN = None
REQUIRE_POSITIVE_PULSE_I = True
REFERENCE_TARGET_ANGLE_DEG = 180.0

LABEL_EVERY_N_BINS = 5
DRAW_PHASE_ARROWS = True
DPI = 300


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
OUT_DIR = (
    HERE / "data" / DATA_DATE
    / f"iq_temperature_track_pulseQ0left_simple_{N_PHASE_BINS}bin_shift{PHASE_OFFSET_EVENTS:+03d}"
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


def phase_index(n_events):
    event_index = np.arange(n_events, dtype=int)
    phase50 = (event_index + PHASE_OFFSET_EVENTS) % EVENTS_PER_THERMAL_CYCLE
    phase_bin = phase50 // EVENTS_PER_PHASE_BIN
    return event_index, phase50, phase_bin


# =============================================================================
# DATA LOADING
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
        print("WARNING: duplicate valid folders found; first valid root is used.")
    print("selected:", candidates[0])
    return candidates[0]


def load_waveforms(meas_dir):
    files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No {NPZ_PATTERN} in {meas_dir}")

    ch0_blocks = []
    ch1_blocks = []
    used_files = []
    time_ref = None
    n_loaded = 0

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
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time-axis mismatch:", path.name)
            continue

        n_take = len(ch0)
        if N_EVENTS_TO_USE is not None:
            n_take = min(n_take, N_EVENTS_TO_USE - n_loaded)

        ch0_blocks.append(ch0[:n_take])
        ch1_blocks.append(ch1[:n_take])
        used_files.append(path)
        n_loaded += n_take
        print(f"load: {path.name} -> {n_take} events (total={n_loaded})")

    if n_loaded == 0:
        raise RuntimeError("No valid waveform events loaded.")

    return time_ref, np.vstack(ch0_blocks), np.vstack(ch1_blocks), used_files


# =============================================================================
# ANALYSIS
# =============================================================================

def build_complex_waveforms(ch0, ch1):
    if USE_CH1_AS_I_AND_CH0_AS_Q:
        return ch1 + 1j * ch0
    return ch0 + 1j * ch1


def compute_pedestal_and_pulse(time_s, z):
    baseline_mask = time_s < 0.0
    if baseline_mask.sum() < 3:
        raise RuntimeError("Too few baseline points.")

    z_ped = z[:, baseline_mask].mean(axis=1)
    dz = z - z_ped[:, None]

    t_us = time_s * 1e6
    search_mask = (t_us >= PEAK_SEARCH_TMIN_US) & (t_us <= PEAK_SEARCH_TMAX_US)
    if search_mask.sum() == 0:
        raise RuntimeError("Empty peak-search window.")

    mean_a2d = np.abs(dz).mean(axis=0)
    local_idx = np.where(search_mask)[0]
    i_peak_global = local_idx[np.argmax(mean_a2d[search_mask])]
    return z_ped, dz, i_peak_global


def summarize_by_phase(time_s, z_ped, dz, i_peak_global):
    _, _, phase_bin = phase_index(len(z_ped))
    rows_ped = []
    rows_pulse = []
    t_us = time_s * 1e6

    for b in range(N_PHASE_BINS):
        idx = np.where(phase_bin == b)[0]
        if len(idx) == 0:
            continue

        ped_re, ped_re25, ped_re75 = median_iqr(z_ped[idx].real)
        ped_im, ped_im25, ped_im75 = median_iqr(z_ped[idx].imag)
        ped_r, ped_r25, ped_r75 = median_iqr(np.abs(z_ped[idx]))
        ped_angle = circular_mean_angle(np.angle(z_ped[idx]))

        rows_ped.append({
            "phase_bin": b,
            "phase50_start": int(b * EVENTS_PER_PHASE_BIN),
            "phase50_end": int((b + 1) * EVENTS_PER_PHASE_BIN - 1),
            "n_events": len(idx),
            "ped_I_median_V": ped_re,
            "ped_I_q25_V": ped_re25,
            "ped_I_q75_V": ped_re75,
            "ped_Q_median_V": ped_im,
            "ped_Q_q25_V": ped_im25,
            "ped_Q_q75_V": ped_im75,
            "ped_radius_median_V": ped_r,
            "ped_radius_q25_V": ped_r25,
            "ped_radius_q75_V": ped_r75,
            "ped_angle_mean_deg": np.degrees(ped_angle),
        })

        dz_bin_mean = dz[idx].mean(axis=0)
        a2d_bin = np.abs(dz_bin_mean)

        if PEAK_MODE == "global_peak":
            i_peak = i_peak_global
        elif PEAK_MODE == "per_bin_peak":
            search_mask = (t_us >= PEAK_SEARCH_TMIN_US) & (t_us <= PEAK_SEARCH_TMAX_US)
            local_idx = np.where(search_mask)[0]
            i_peak = local_idx[np.argmax(a2d_bin[search_mask])]
        else:
            raise ValueError("PEAK_MODE must be 'global_peak' or 'per_bin_peak'.")

        pulse_vec = dz_bin_mean[i_peak]
        rows_pulse.append({
            "phase_bin": b,
            "peak_time_us": t_us[i_peak],
            "pulse_I_V": pulse_vec.real,
            "pulse_Q_V": pulse_vec.imag,
            "pulse_A2D_V": abs(pulse_vec),
            "pulse_angle_deg": np.degrees(np.angle(pulse_vec)),
            "abs_pulseQ_over_A": abs(pulse_vec.imag) / abs(pulse_vec) if abs(pulse_vec) > 0 else np.nan,
        })

    return pd.DataFrame(rows_ped), pd.DataFrame(rows_pulse)


def choose_reference_bin(pulse_df):
    if REFERENCE_PHASE_BIN is not None:
        return int(REFERENCE_PHASE_BIN)

    df = pulse_df.copy()
    if REQUIRE_POSITIVE_PULSE_I:
        df_pos = df[df["pulse_I_V"] > 0]
        if len(df_pos) > 0:
            df = df_pos

    df = df.assign(score=np.abs(df["pulse_Q_V"]))
    row = df.sort_values(["score", "abs_pulseQ_over_A", "phase_bin"]).iloc[0]
    return int(row["phase_bin"])


def apply_global_rotation(z_ped, ped_df, pulse_df, ref_bin):
    row = ped_df.loc[ped_df["phase_bin"] == ref_bin].iloc[0]
    z_ref = row["ped_I_median_V"] + 1j * row["ped_Q_median_V"]

    target = np.radians(REFERENCE_TARGET_ANGLE_DEG)
    phi_ref = np.angle(z_ref)
    rot = np.exp(1j * (target - phi_ref))

    ped_rot_df = ped_df.copy()
    zmed = ped_df["ped_I_median_V"].to_numpy() + 1j * ped_df["ped_Q_median_V"].to_numpy()
    zmed_rot = rot * zmed
    ped_rot_df["rot_I_median_V"] = zmed_rot.real
    ped_rot_df["rot_Q_median_V"] = zmed_rot.imag
    ped_rot_df["rot_angle_mean_deg"] = np.degrees(np.angle(zmed_rot))

    pulse_rot_df = pulse_df.copy()
    pvec = pulse_df["pulse_I_V"].to_numpy() + 1j * pulse_df["pulse_Q_V"].to_numpy()
    pvec_rot = rot * pvec
    pulse_rot_df["rot_pulse_I_V"] = pvec_rot.real
    pulse_rot_df["rot_pulse_Q_V"] = pvec_rot.imag

    return z_ref, rot, ped_rot_df, pulse_rot_df


# =============================================================================
# PLOT
# =============================================================================

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
            ax.annotate(str(b), (x[i], y[i]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    return sc


def plot_result(ped_df, ref_bin, output_path):
    """
    Keep only the essentials:
      1) raw pedestal IQ track
      2) globally rotated pedestal IQ track
      3) ch1 pedestal (= I before rotation) vs thermal phase
      4) ch0 pedestal (= Q before rotation) vs thermal phase

    The old pulse-component and radius diagnostics are intentionally omitted.
    """
    bins = ped_df["phase_bin"].to_numpy(dtype=int)
    ref_idx = int(np.where(bins == ref_bin)[0][0])

    fig, axes = plt.subplots(2, 2, figsize=(15.6, 11.8), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.34, w_pad=0.14, hspace=0.28, wspace=0.18)

    # -------------------------------------------------------------------------
    # (1) Raw pedestal IQ
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    raw_I = ped_df["ped_I_median_V"].to_numpy() * 1e3
    raw_Q = ped_df["ped_Q_median_V"].to_numpy() * 1e3
    raw_errors = (
        raw_I - ped_df["ped_I_q25_V"].to_numpy() * 1e3,
        ped_df["ped_I_q75_V"].to_numpy() * 1e3 - raw_I,
        raw_Q - ped_df["ped_Q_q25_V"].to_numpy() * 1e3,
        ped_df["ped_Q_q75_V"].to_numpy() * 1e3 - raw_Q,
    )
    sc = draw_track(
        ax, raw_I, raw_Q, bins,
        "Raw pedestal IQ track",
        "raw I [mV]  (= ch1 pedestal)",
        "raw Q [mV]  (= ch0 pedestal)",
        raw_errors,
    )
    fig.colorbar(sc, ax=ax, label="thermal phase bin")
    legend_below(ax, ncol=1)

    # -------------------------------------------------------------------------
    # (2) Same pedestal track after one fixed global rotation
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    rot_I = ped_df["rot_I_median_V"].to_numpy() * 1e3
    rot_Q = ped_df["rot_Q_median_V"].to_numpy() * 1e3
    sc = draw_track(
        ax, rot_I, rot_Q, bins,
        "Globally rotated pedestal IQ track",
        "rotated I [mV]",
        "rotated Q [mV]",
        errors=None,
    )
    ax.scatter(
        [rot_I[ref_idx]], [rot_Q[ref_idx]],
        marker="*", s=220, zorder=7,
        label=f"pulse-Q≈0 reference bin {ref_bin}",
    )
    ax.axhline(0.0, lw=1.0)
    ax.axvline(0.0, lw=1.0)
    fig.colorbar(sc, ax=ax, label="thermal phase bin")
    legend_below(ax, ncol=2)

    # -------------------------------------------------------------------------
    # (3) ch1 pedestal (raw I) vs thermal phase
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    y = raw_I
    ylo = y - ped_df["ped_I_q25_V"].to_numpy() * 1e3
    yhi = ped_df["ped_I_q75_V"].to_numpy() * 1e3 - y
    ax.errorbar(
        bins, y, yerr=[ylo, yhi],
        marker="o", capsize=3,
        label="median ± IQR",
    )
    ax.axvline(ref_bin, ls="--", lw=1.0, label=f"reference bin {ref_bin}")
    ax.set_title("Pedestal ch1 (= raw I) across the 1 Hz cycle")
    ax.set_xlabel("thermal phase bin (event index mod 50)")
    ax.set_ylabel("ch1 pedestal [mV]")
    ax.grid(True)
    legend_below(ax, ncol=2)

    # -------------------------------------------------------------------------
    # (4) ch0 pedestal (raw Q) vs thermal phase
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    y = raw_Q
    ylo = y - ped_df["ped_Q_q25_V"].to_numpy() * 1e3
    yhi = ped_df["ped_Q_q75_V"].to_numpy() * 1e3 - y
    ax.errorbar(
        bins, y, yerr=[ylo, yhi],
        marker="o", capsize=3,
        label="median ± IQR",
    )
    ax.axvline(ref_bin, ls="--", lw=1.0, label=f"reference bin {ref_bin}")
    ax.set_title("Pedestal ch0 (= raw Q) across the 1 Hz cycle")
    ax.set_xlabel("thermal phase bin (event index mod 50)")
    ax.set_ylabel("ch0 pedestal [mV]")
    ax.grid(True)
    legend_below(ax, ncol=2)

    fig.suptitle(
        "20260527 pedestal IQ and phase dependence\n"
        f"{TARGET_DIR_NAME}; {N_PHASE_BINS} bins / 50 events; "
        f"phase offset={PHASE_OFFSET_EVENTS:+d}; reference bin={ref_bin}",
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
    print("N_PHASE_BINS:", N_PHASE_BINS)
    print("PHASE_OFFSET_EVENTS:", PHASE_OFFSET_EVENTS)

    meas_dir = find_target_directory()
    time_s, ch0, ch1, used_files = load_waveforms(meas_dir)

    z = build_complex_waveforms(ch0, ch1)
    z_ped, dz, i_peak_global = compute_pedestal_and_pulse(time_s, z)
    ped_df, pulse_df = summarize_by_phase(time_s, z_ped, dz, i_peak_global)

    ref_bin = choose_reference_bin(pulse_df)
    z_ref, rot, ped_df, pulse_df = apply_global_rotation(z_ped, ped_df, pulse_df, ref_bin)

    png_path = OUT_DIR / "pedestal_iq_simple.png"
    ped_csv = OUT_DIR / "pedestal_iq_phase_summary.csv"
    pulse_csv = OUT_DIR / "pulse_phase_summary.csv"
    json_path = OUT_DIR / "rotation_settings.json"
    info_path = OUT_DIR / "run_info.txt"

    plot_result(ped_df, ref_bin, png_path)
    ped_df.to_csv(ped_csv, index=False)
    pulse_df.to_csv(pulse_csv, index=False)

    settings = {
        "target_dir_name": TARGET_DIR_NAME,
        "n_events_used": int(len(z_ped)),
        "I_definition": "ch1",
        "Q_definition": "ch0",
        "n_phase_bins": N_PHASE_BINS,
        "events_per_phase_bin": EVENTS_PER_PHASE_BIN,
        "phase_offset_events": PHASE_OFFSET_EVENTS,
        "peak_mode": PEAK_MODE,
        "global_peak_time_us": float(time_s[i_peak_global] * 1e6),
        "reference_phase_bin": int(ref_bin),
        "reference_target_angle_deg": float(REFERENCE_TARGET_ANGLE_DEG),
        "reference_pedestal_raw_I_V": float(z_ref.real),
        "reference_pedestal_raw_Q_V": float(z_ref.imag),
        "reference_pedestal_raw_angle_deg": float(np.degrees(np.angle(z_ref))),
        "fixed_rotation_deg": float(np.degrees(np.angle(rot))),
    }
    json_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    info_lines = [
        f"measurement_dir = {meas_dir}",
        f"n_events_used = {len(z_ped)}",
        f"I_definition = ch1",
        f"Q_definition = ch0",
        f"N_PHASE_BINS = {N_PHASE_BINS}",
        f"EVENTS_PER_PHASE_BIN = {EVENTS_PER_PHASE_BIN}",
        f"PHASE_OFFSET_EVENTS = {PHASE_OFFSET_EVENTS}",
        f"reference_phase_bin = {ref_bin}",
        f"reference pedestal raw = {z_ref.real*1e3:+.6f} + i {z_ref.imag*1e3:+.6f} mV",
        f"reference pedestal raw angle = {np.degrees(np.angle(z_ref)):+.6f} deg",
        f"fixed_rotation_deg = {np.degrees(np.angle(rot)):+.6f}",
        "",
        "used_files:",
        *map(str, used_files),
    ]
    info_path.write_text("\n".join(info_lines), encoding="utf-8")

    print("\nsaved:")
    print(" ", png_path)
    print(" ", ped_csv)
    print(" ", pulse_csv)
    print(" ", json_path)
    print(" ", info_path)


if __name__ == "__main__":
    main()
