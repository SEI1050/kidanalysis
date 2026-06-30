#!/usr/bin/env python3
"""
analyze_20260618_highstat_100k.py

20260618 / 5.443GHz_z=7.5mm_x=3.40mm_100000evants
の高統計（最大 100000 events）waveform を、メモリを使いすぎないよう
npz ファイル単位で streaming 解析する。

高統計データでまず確認すること
================================
1. 平均 pulse waveform
2. event ごとの peak amplitude / pulse area / tau_eff / SNR のヒストグラム
3. pedestal IQ の密度分布と経時変化
4. laser 50 Hz・温度 1 Hz を仮定した event_index mod 50 の phase-folding
5. pulse amplitude と pedestal / temperature phase の相関
6. event index に沿った長時間ドリフト・異常 event

主な出力
========
~/software/kidanalysis/data/20260618/highstat_100k/
  01_highstat_overview.png
  02_highstat_thermal_phase.png
  03_highstat_distributions.png
  highstat_thermal_phase_summary.csv
  highstat_event_metrics.csv       (100000 rows; SAVE_EVENT_CSV=True の場合)
  highstat_event_metrics.npz
  highstat_run_info.txt

注意
====
- 2D pulse amplitude は pedestal を引いてから
      A2D(t) = sqrt(<Δch0(t)>^2 + <Δch1(t)>^2)
  と定義する。
- event ごとの peak は、global mean pulse direction への射影
  A_proj(t) を主な scalar amplitude として使う。
  これにより sqrt(I^2+Q^2) の baseline noise bias を避ける。
- event_index mod 50 による phase-folding は、laser event の欠落が少なく、
  laser 50 Hz / thermal cycle 1 Hz が同期している場合に有効。
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DATE = "20260618"
TARGET_DIR_NAME = "5.443GHz_z=7.5mm_x=3.40mm_100000evants"
NPZ_PATTERN = "wf_*.npz"

# None: all valid events. Target directory name implies 100000 events.
MAX_EVENTS = 100000

# Trigger-relative analysis windows [us]
BASELINE_WINDOW_US = None       # None -> all t < 0
AMP_WINDOW_US = (0.0, 1.50)
INTEGRAL_WINDOW_US = (0.0, 2.00)
PLOT_TIME_WINDOW_US = (-0.30, 2.00)

# Acquisition timing:
# Laser is 2 Hz. If the thermal modulation remains 1 Hz, the laser samples
# two reproducible thermal phases per temperature period: even / odd events.
LASER_HZ = 2.0
THERMAL_HZ = 1.0

EVENTS_PER_TEMP_CYCLE = int(round(LASER_HZ / THERMAL_HZ))
if not np.isclose(EVENTS_PER_TEMP_CYCLE * THERMAL_HZ, LASER_HZ):
    raise ValueError(
        "LASER_HZ / THERMAL_HZ must be an integer for event-index phase folding."
    )

# With 2 Hz laser and 1 Hz thermal modulation, this must be 2.
N_PHASE_BINS = EVENTS_PER_TEMP_CYCLE

# Long-time trend: summarize this many consecutive events per point
TREND_BLOCK_SIZE = 500

# Histogram plotting
HIST_BINS = 160
MAX_HEXBIN_GRIDSIZE = 90

# Outputs
SAVE_EVENT_CSV = True
DPI = 280


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / DATA_DATE / "highstat_100k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# External SSD comes first to avoid accidentally selecting a OneDrive directory
# containing only metadata or incomplete files.
INPUT_ROOTS = [
    Path("/Volumes/NO NAME/data") / DATA_DATE,
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-TheUniversityofTokyo"
    / "東京大学"
    / "4S"
    / "kidfit"
    / DATA_DATE,
    Path.home()
    / "OneDrive - The University of Tokyo"
    / "東京大学"
    / "4S"
    / "kidfit"
    / DATA_DATE,
    HERE / "data" / DATA_DATE,
]


# =============================================================================
# HELPERS
# =============================================================================

def scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else x


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


def robust_sigma(values):
    """Gaussian-equivalent robust sigma from MAD."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def finite_limits(values, qlo=0.002, qhi=0.998, pad_fraction=0.08):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (0.0, 1.0)
    lo, hi = np.quantile(x, [qlo, qhi])
    if hi <= lo:
        width = max(abs(lo), 1.0)
        return lo - 0.1 * width, hi + 0.1 * width
    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def safe_unit(z):
    mag = abs(z)
    return z / mag if mag > 0 else 1.0 + 0.0j


def get_npz_arrays(path):
    """
    Load a waveform file and return ch0, ch1, time_s.
    Return None if invalid.
    """
    try:
        data = np.load(path)
    except Exception as exc:
        print("skip unreadable:", path.name, exc)
        return None

    required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
    missing = [key for key in required if key not in data.files]
    if missing:
        print("skip missing keys:", path.name, missing)
        return None

    ch0 = np.asarray(data["ch0"], dtype=float)
    ch1 = np.asarray(data["ch1"], dtype=float)

    if ch0.ndim == 1:
        ch0 = ch0[None, :]
    if ch1.ndim == 1:
        ch1 = ch1[None, :]

    if ch0.shape != ch1.shape:
        print("skip shape mismatch:", path.name, ch0.shape, ch1.shape)
        return None

    time_s = make_time_axis_s(
        int(scalar(data["npts"])),
        float(scalar(data["sample_rate"])),
        float(scalar(data["ref_position"])),
    )

    if ch0.shape[1] != len(time_s):
        print("skip npts/time mismatch:", path.name)
        return None

    return ch0, ch1, time_s


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def find_measurement_dir():
    print("\n===== input roots =====")
    candidates = []

    for root in INPUT_ROOTS:
        root = Path(root).expanduser()
        candidate = root / TARGET_DIR_NAME
        n_npz = len(list(candidate.glob(NPZ_PATTERN))) if candidate.is_dir() else 0
        print(root, "exists=", root.is_dir(), "| candidate wf files =", n_npz)
        if n_npz > 0:
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            f"Could not find {TARGET_DIR_NAME!r} containing {NPZ_PATTERN}."
        )

    if len(candidates) > 1:
        print("WARNING: duplicate data folders found; use first valid root.")

    selected = candidates[0]
    print("selected:", selected)
    return selected


def list_waveform_files(meas_dir):
    files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No {NPZ_PATTERN} files in {meas_dir}")
    return files


# =============================================================================
# TWO-PASS STREAMING ANALYSIS
# =============================================================================

def setup_time_masks(time_s):
    time_us = time_s * 1e6

    if BASELINE_WINDOW_US is None:
        baseline_mask = time_us < 0.0
    else:
        lo, hi = BASELINE_WINDOW_US
        baseline_mask = (time_us >= lo) & (time_us <= hi)

    amp_mask = (time_us >= AMP_WINDOW_US[0]) & (time_us <= AMP_WINDOW_US[1])
    integral_mask = (
        (time_us >= INTEGRAL_WINDOW_US[0])
        & (time_us <= INTEGRAL_WINDOW_US[1])
    )

    if baseline_mask.sum() < 3:
        raise RuntimeError("Too few baseline samples.")
    if amp_mask.sum() < 3:
        raise RuntimeError("Too few amplitude-window samples.")
    if integral_mask.sum() < 3:
        raise RuntimeError("Too few integral-window samples.")

    return time_us, baseline_mask, amp_mask, integral_mask


def first_pass_global_mean(files):
    """
    Stream through all events once:
      event-wise pedestal subtraction -> global mean dch0/dch1 waveform.

    This determines a stable global pulse direction and peak time for pass 2.
    """
    print("\n===== first pass: global mean pulse =====")

    time_ref = None
    masks = None
    sum_d0 = None
    sum_d1 = None
    n_total = 0
    used_files = []

    for path in files:
        if MAX_EVENTS is not None and n_total >= MAX_EVENTS:
            break

        result = get_npz_arrays(path)
        if result is None:
            continue
        ch0, ch1, time_s = result

        if time_ref is None:
            time_ref = time_s
            time_us, bmask, amask, imask = setup_time_masks(time_s)
            masks = (time_us, bmask, amask, imask)
            sum_d0 = np.zeros(len(time_s), dtype=np.float64)
            sum_d1 = np.zeros(len(time_s), dtype=np.float64)
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip mismatched time axis:", path.name)
            continue

        n_take = len(ch0)
        if MAX_EVENTS is not None:
            n_take = min(n_take, MAX_EVENTS - n_total)

        c0 = ch0[:n_take]
        c1 = ch1[:n_take]
        _, bmask, _, _ = masks

        ped0 = c0[:, bmask].mean(axis=1)
        ped1 = c1[:, bmask].mean(axis=1)

        d0 = c0 - ped0[:, None]
        d1 = c1 - ped1[:, None]

        sum_d0 += d0.sum(axis=0)
        sum_d1 += d1.sum(axis=0)
        n_total += n_take
        used_files.append(path)

        if len(used_files) % 20 == 0 or n_total == MAX_EVENTS:
            print(f"  first pass: {n_total} events")

    if time_ref is None or n_total == 0:
        raise RuntimeError("No valid events loaded in first pass.")

    mean_d0 = sum_d0 / n_total
    mean_d1 = sum_d1 / n_total
    a2d = np.hypot(mean_d0, mean_d1)

    time_us, _, amask, _ = masks
    amp_indices = np.where(amask)[0]
    peak_idx = int(amp_indices[np.argmax(a2d[amask])])
    pulse_vec = complex(mean_d0[peak_idx], mean_d1[peak_idx])
    pulse_unit = safe_unit(pulse_vec)

    print("global mean pulse:")
    print(f"  events = {n_total}")
    print(f"  peak time = {time_us[peak_idx]:.6f} us")
    print(
        "  peak vector = "
        f"{pulse_vec.real*1e3:+.5f} + i {pulse_vec.imag*1e3:+.5f} mV"
    )
    print(f"  global direction = ({pulse_unit.real:+.5f}, {pulse_unit.imag:+.5f})")

    return {
        "time_s": time_ref,
        "time_us": time_us,
        "baseline_mask": masks[1],
        "amp_mask": masks[2],
        "integral_mask": masks[3],
        "mean_dch0": mean_d0,
        "mean_dch1": mean_d1,
        "mean_a2d": a2d,
        "peak_idx": peak_idx,
        "pulse_vec": pulse_vec,
        "pulse_unit": pulse_unit,
        "n_events": n_total,
        "files": used_files,
    }


def second_pass_event_metrics(first):
    """
    Stream a second time and compute event-level metrics.

    Kept metrics per event are small arrays only, so 100000 events remain cheap.
    """
    print("\n===== second pass: event metrics =====")

    files = first["files"]
    time_us = first["time_us"]
    bmask = first["baseline_mask"]
    amask = first["amp_mask"]
    imask = first["integral_mask"]
    peak_idx_global = first["peak_idx"]
    u = first["pulse_unit"]
    v = -1j * u  # perpendicular direction

    amp_indices = np.where(amask)[0]
    event_counter = 0

    metrics = {
        "event_index": [],
        "thermal_phase": [],
        "ped_ch0_V": [],
        "ped_ch1_V": [],
        "peak_proj_V": [],
        "peak_2d_V": [],
        "integral_proj_Vus": [],
        "integral_2d_Vus": [],
        "tau_eff_2d_us": [],
        "peak_time_us": [],
        "baseline_rms_proj_V": [],
        "snr_proj": [],
        "dch0_at_global_peak_V": [],
        "dch1_at_global_peak_V": [],
        "parallel_at_global_peak_V": [],
        "perp_at_global_peak_V": [],
    }

    for path in files:
        if MAX_EVENTS is not None and event_counter >= MAX_EVENTS:
            break

        result = get_npz_arrays(path)
        if result is None:
            continue
        ch0, ch1, time_s = result

        if len(time_s) != len(first["time_s"]) or not np.allclose(time_s, first["time_s"]):
            print("skip mismatched time axis:", path.name)
            continue

        n_take = len(ch0)
        if MAX_EVENTS is not None:
            n_take = min(n_take, MAX_EVENTS - event_counter)

        c0 = ch0[:n_take]
        c1 = ch1[:n_take]

        ped0 = c0[:, bmask].mean(axis=1)
        ped1 = c1[:, bmask].mean(axis=1)
        d0 = c0 - ped0[:, None]
        d1 = c1 - ped1[:, None]

        # Projection along the global pulse direction
        proj = d0 * u.real + d1 * u.imag
        perp = d0 * v.real + d1 * v.imag

        a2d_event = np.hypot(d0, d1)

        # Local peak based on projected pulse, avoiding 2D baseline noise bias
        local_peak_index_in_window = np.argmax(proj[:, amask], axis=1)
        local_peak_indices = amp_indices[local_peak_index_in_window]

        row_index = np.arange(n_take)
        peak_proj = proj[row_index, local_peak_indices]
        peak_2d = a2d_event[row_index, local_peak_indices]

        integral_proj = np.trapezoid(proj[:, imask], x=time_us[imask], axis=1)
        integral_2d = np.trapezoid(a2d_event[:, imask], x=time_us[imask], axis=1)
        tau_eff_2d = np.divide(
            integral_2d,
            peak_2d,
            out=np.full(n_take, np.nan),
            where=peak_2d > 0,
        )

        baseline_rms = np.std(proj[:, bmask], axis=1, ddof=1)
        snr = np.divide(
            peak_proj,
            baseline_rms,
            out=np.full(n_take, np.nan),
            where=baseline_rms > 0,
        )

        event_index = np.arange(event_counter, event_counter + n_take)
        z_global = d0[:, peak_idx_global] + 1j * d1[:, peak_idx_global]

        metrics["event_index"].append(event_index)
        metrics["thermal_phase"].append(event_index % EVENTS_PER_TEMP_CYCLE)
        metrics["ped_ch0_V"].append(ped0)
        metrics["ped_ch1_V"].append(ped1)
        metrics["peak_proj_V"].append(peak_proj)
        metrics["peak_2d_V"].append(peak_2d)
        metrics["integral_proj_Vus"].append(integral_proj)
        metrics["integral_2d_Vus"].append(integral_2d)
        metrics["tau_eff_2d_us"].append(tau_eff_2d)
        metrics["peak_time_us"].append(time_us[local_peak_indices])
        metrics["baseline_rms_proj_V"].append(baseline_rms)
        metrics["snr_proj"].append(snr)
        metrics["dch0_at_global_peak_V"].append(z_global.real)
        metrics["dch1_at_global_peak_V"].append(z_global.imag)
        metrics["parallel_at_global_peak_V"].append(proj[:, peak_idx_global])
        metrics["perp_at_global_peak_V"].append(perp[:, peak_idx_global])

        event_counter += n_take
        if event_counter % 10000 == 0 or event_counter == first["n_events"]:
            print(f"  second pass: {event_counter} events")

    out = {
        name: np.concatenate(parts) if parts else np.array([])
        for name, parts in metrics.items()
    }
    return out


# =============================================================================
# SUMMARIES
# =============================================================================

def thermal_phase_summary(metrics):
    phase = metrics["thermal_phase"]
    rows = []

    for p in range(EVENTS_PER_TEMP_CYCLE):
        idx = phase == p
        row = {
            "thermal_phase": p,
            "n_events": int(idx.sum()),
        }
        for name in [
            "ped_ch0_V",
            "ped_ch1_V",
            "peak_proj_V",
            "peak_2d_V",
            "integral_proj_Vus",
            "tau_eff_2d_us",
            "snr_proj",
            "peak_time_us",
        ]:
            med, q25, q75 = median_iqr(metrics[name][idx])
            row[f"{name}_median"] = med
            row[f"{name}_q25"] = q25
            row[f"{name}_q75"] = q75
        rows.append(row)

    return pd.DataFrame(rows)


def trend_summary(metrics):
    n = len(metrics["event_index"])
    rows = []

    for start in range(0, n, TREND_BLOCK_SIZE):
        stop = min(start + TREND_BLOCK_SIZE, n)
        idx = slice(start, stop)

        row = {
            "event_start": int(start),
            "event_end": int(stop - 1),
            "event_center": 0.5 * (start + stop - 1),
            "n_events": stop - start,
        }
        for name in [
            "ped_ch0_V",
            "ped_ch1_V",
            "peak_proj_V",
            "peak_2d_V",
            "snr_proj",
            "tau_eff_2d_us",
        ]:
            med, q25, q75 = median_iqr(metrics[name][idx])
            row[f"{name}_median"] = med
            row[f"{name}_q25"] = q25
            row[f"{name}_q75"] = q75
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def legend_below(ax, handles=None, labels=None, ncol=2):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=ncol,
        fontsize=8,
        frameon=True,
        borderaxespad=0.0,
    )


def plot_overview(first, metrics, trend, output_path):
    time_us = first["time_us"]
    mask = (
        (time_us >= PLOT_TIME_WINDOW_US[0])
        & (time_us <= PLOT_TIME_WINDOW_US[1])
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.0), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.36, w_pad=0.12, hspace=0.25, wspace=0.16)

    # Mean waveform
    ax = axes[0, 0]
    ax.plot(time_us[mask], first["mean_dch0"][mask] * 1e3, label="mean Δch0")
    ax.plot(time_us[mask], first["mean_dch1"][mask] * 1e3, label="mean Δch1")
    ax.plot(time_us[mask], first["mean_a2d"][mask] * 1e3, label=r"$A_{2D}$")
    ax.axvline(time_us[first["peak_idx"]], ls="--", lw=1.0, label="global peak")
    ax.set_title("Global mean pulse waveform")
    ax.set_xlabel("time from trigger [us]")
    ax.set_ylabel("signal [mV]")
    ax.grid(True)
    legend_below(ax, ncol=2)

    # Long-time amplitude trend
    ax = axes[0, 1]
    x = trend["event_center"].to_numpy()
    y = trend["peak_proj_V_median"].to_numpy() * 1e3
    ylo = y - trend["peak_proj_V_q25"].to_numpy() * 1e3
    yhi = trend["peak_proj_V_q75"].to_numpy() * 1e3 - y
    ax.fill_between(x, y - ylo, y + yhi, alpha=0.25, label="IQR")
    ax.plot(x, y, marker="o", ms=3, label="median peak projection")
    ax.set_title("Long-time trend: pulse amplitude")
    ax.set_xlabel("event index")
    ax.set_ylabel("peak projected amplitude [mV]")
    ax.grid(True)

    axr = ax.twinx()
    axr.plot(
        x,
        trend["ped_ch1_V_median"].to_numpy() * 1e3,
        marker="s",
        ms=3,
        alpha=0.75,
        label="pedestal ch1 median",
    )
    axr.set_ylabel("pedestal ch1 [mV]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    # Pedestal density
    ax = axes[1, 0]
    xped = metrics["ped_ch0_V"] * 1e3
    yped = metrics["ped_ch1_V"] * 1e3
    hb = ax.hexbin(
        xped,
        yped,
        gridsize=MAX_HEXBIN_GRIDSIZE,
        mincnt=1,
        bins="log",
    )
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_title("Pedestal IQ density: all events")
    ax.set_xlabel("pedestal ch0 [mV]")
    ax.set_ylabel("pedestal ch1 [mV]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Peak component density
    ax = axes[1, 1]
    xvec = metrics["dch0_at_global_peak_V"] * 1e3
    yvec = metrics["dch1_at_global_peak_V"] * 1e3
    hb = ax.hexbin(
        xvec,
        yvec,
        gridsize=MAX_HEXBIN_GRIDSIZE,
        mincnt=1,
        bins="log",
    )
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.axhline(0.0, lw=1.0)
    ax.axvline(0.0, lw=1.0)
    ax.set_title("Pulse-vector density at global peak time")
    ax.set_xlabel(r"$\Delta$ch0 [mV]")
    ax.set_ylabel(r"$\Delta$ch1 [mV]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    fig.suptitle(
        f"High-statistics overview: {TARGET_DIR_NAME}\n"
        f"N={len(metrics['event_index'])} events; global pulse peak={time_us[first['peak_idx']]:.3f} us",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def plot_thermal_phase(metrics, phase_df, output_path):
    phase = phase_df["thermal_phase"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.0), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.36, w_pad=0.12, hspace=0.25, wspace=0.16)

    # Pedestal path by phase
    ax = axes[0, 0]
    x = phase_df["ped_ch0_V_median"].to_numpy() * 1e3
    y = phase_df["ped_ch1_V_median"].to_numpy() * 1e3
    sc = ax.scatter(x, y, c=phase, s=40, zorder=4)
    ax.plot(x, y, lw=1.0, alpha=0.5)
    for i, p in enumerate(phase):
        if p % 5 == 0:
            ax.annotate(str(p), (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.colorbar(sc, ax=ax, label="event index mod 50")
    ax.set_title("Pedestal trajectory folded into two thermal phases (even / odd events)")
    ax.set_xlabel("pedestal ch0 [mV]")
    ax.set_ylabel("pedestal ch1 [mV]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Pedestal components by phase
    ax = axes[0, 1]
    for col, label in [
        ("ped_ch0_V", "pedestal ch0"),
        ("ped_ch1_V", "pedestal ch1"),
    ]:
        y = phase_df[f"{col}_median"].to_numpy() * 1e3
        ylo = y - phase_df[f"{col}_q25"].to_numpy() * 1e3
        yhi = phase_df[f"{col}_q75"].to_numpy() * 1e3 - y
        ax.errorbar(phase, y, yerr=[ylo, yhi], marker="o", capsize=2, label=label)
    ax.set_title("Pedestal components across 1 Hz phase")
    ax.set_xlabel("thermal phase = event index mod 2")
    ax.set_ylabel("pedestal [mV]")
    ax.grid(True)
    legend_below(ax, ncol=2)

    # Pulse amplitude phase dependence
    ax = axes[1, 0]
    for col, label in [
        ("peak_proj_V", "projected peak"),
        ("peak_2d_V", "2D peak at projected-peak time"),
    ]:
        y = phase_df[f"{col}_median"].to_numpy() * 1e3
        ylo = y - phase_df[f"{col}_q25"].to_numpy() * 1e3
        yhi = phase_df[f"{col}_q75"].to_numpy() * 1e3 - y
        ax.errorbar(phase, y, yerr=[ylo, yhi], marker="o", capsize=2, label=label)
    ax.set_title("Pulse amplitude across 1 Hz phase")
    ax.set_xlabel("thermal phase = event index mod 2")
    ax.set_ylabel("peak amplitude [mV]")
    ax.grid(True)
    legend_below(ax, ncol=2)

    # SNR + tau
    ax = axes[1, 1]
    y = phase_df["snr_proj_median"].to_numpy()
    ylo = y - phase_df["snr_proj_q25"].to_numpy()
    yhi = phase_df["snr_proj_q75"].to_numpy() - y
    ax.errorbar(phase, y, yerr=[ylo, yhi], marker="o", capsize=2, label="SNR")
    ax.set_title("Pulse quality across 1 Hz phase")
    ax.set_xlabel("thermal phase = event index mod 2")
    ax.set_ylabel("projected SNR")
    ax.grid(True)

    axr = ax.twinx()
    tau = phase_df["tau_eff_2d_us_median"].to_numpy()
    axr.plot(phase, tau, marker="s", label=r"$\tau_{\rm eff,2D}$")
    axr.set_ylabel(r"$\tau_{\rm eff,2D}$ [us]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    fig.suptitle(
        "High-statistics even/odd comparison: 2 Hz laser / 1 Hz thermal cycle assumption",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def plot_distributions(metrics, output_path):
    peak_proj = metrics["peak_proj_V"] * 1e3
    peak_2d = metrics["peak_2d_V"] * 1e3
    snr = metrics["snr_proj"]
    tau = metrics["tau_eff_2d_us"]
    peak_time = metrics["peak_time_us"]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 11.5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.34, w_pad=0.12, hspace=0.24, wspace=0.15)

    # Main amplitude histogram
    ax = axes[0, 0]
    lim = finite_limits(peak_proj)
    ax.hist(peak_proj, bins=HIST_BINS, range=lim, histtype="stepfilled", alpha=0.7)
    med, q25, q75 = median_iqr(peak_proj)
    ax.axvline(med, ls="--", label=f"median={med:.3f} mV")
    ax.axvspan(q25, q75, alpha=0.18, label="IQR")
    ax.set_title("Histogram: projected pulse peak")
    ax.set_xlabel("peak projected amplitude [mV]")
    ax.set_ylabel("events")
    ax.grid(True)
    legend_below(ax, ncol=2)

    # 2D amplitude and SNR
    ax = axes[0, 1]
    lim = finite_limits(peak_2d)
    ax.hist(peak_2d, bins=HIST_BINS, range=lim, histtype="stepfilled", alpha=0.7, label="2D peak")
    ax.set_title("Histogram: 2D pulse peak")
    ax.set_xlabel("2D peak amplitude [mV]")
    ax.set_ylabel("events")
    ax.grid(True)
    legend_below(ax, ncol=1)

    # SNR
    ax = axes[1, 0]
    lim = finite_limits(snr)
    ax.hist(snr, bins=HIST_BINS, range=lim, histtype="stepfilled", alpha=0.7)
    med, q25, q75 = median_iqr(snr)
    ax.axvline(med, ls="--", label=f"median={med:.1f}")
    ax.axvspan(q25, q75, alpha=0.18, label="IQR")
    ax.set_title("Histogram: projected pulse SNR")
    ax.set_xlabel("SNR")
    ax.set_ylabel("events")
    ax.grid(True)
    legend_below(ax, ncol=2)

    # Tau / peak time
    ax = axes[1, 1]
    lim = finite_limits(tau)
    ax.hist(tau, bins=HIST_BINS, range=lim, histtype="stepfilled", alpha=0.65, label=r"$\tau_{\rm eff,2D}$")
    ax.set_title("Histogram: effective pulse width")
    ax.set_xlabel(r"$\tau_{\rm eff,2D}$ [us]")
    ax.set_ylabel("events")
    ax.grid(True)

    axr = ax.twinx()
    lim_t = finite_limits(peak_time)
    axr.hist(
        peak_time,
        bins=HIST_BINS,
        range=lim_t,
        histtype="step",
        lw=1.4,
        label="peak time",
    )
    axr.set_ylabel("events (peak-time histogram)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    fig.suptitle(
        f"High-statistics distributions: N={len(peak_proj)} events",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


# =============================================================================
# OUTPUT
# =============================================================================

def save_event_metrics(metrics, path_npz, path_csv):
    np.savez_compressed(path_npz, **metrics)

    if SAVE_EVENT_CSV:
        df = pd.DataFrame(metrics)
        # mV is much easier to inspect manually in a spreadsheet.
        for col in list(df.columns):
            if col.endswith("_V"):
                df[col.replace("_V", "_mV")] = df[col] * 1e3
            elif col.endswith("_Vus"):
                df[col.replace("_Vus", "_mVus")] = df[col] * 1e3
        df.to_csv(path_csv, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Output:", OUT_DIR)
    print("Target:", TARGET_DIR_NAME)
    print("MAX_EVENTS:", MAX_EVENTS)
    print("N_THERMAL_PHASES:", N_PHASE_BINS)

    meas_dir = find_measurement_dir()
    files = list_waveform_files(meas_dir)
    print("waveform files:", len(files))

    first = first_pass_global_mean(files)
    metrics = second_pass_event_metrics(first)

    phase_df = thermal_phase_summary(metrics)
    trend_df = trend_summary(metrics)

    out_overview = OUT_DIR / "01_highstat_overview.png"
    out_phase = OUT_DIR / "02_highstat_thermal_phase.png"
    out_dist = OUT_DIR / "03_highstat_distributions.png"
    out_phase_csv = OUT_DIR / "highstat_thermal_phase_summary.csv"
    out_trend_csv = OUT_DIR / "highstat_trend_summary.csv"
    out_npz = OUT_DIR / "highstat_event_metrics.npz"
    out_csv = OUT_DIR / "highstat_event_metrics.csv"
    out_info = OUT_DIR / "highstat_run_info.txt"

    plot_overview(first, metrics, trend_df, out_overview)
    plot_thermal_phase(metrics, phase_df, out_phase)
    plot_distributions(metrics, out_dist)

    phase_df.to_csv(out_phase_csv, index=False)
    trend_df.to_csv(out_trend_csv, index=False)
    save_event_metrics(metrics, out_npz, out_csv)

    # Compact console summary
    peak_mV = metrics["peak_proj_V"] * 1e3
    snr = metrics["snr_proj"]
    tau = metrics["tau_eff_2d_us"]
    peak_med, peak_q25, peak_q75 = median_iqr(peak_mV)
    snr_med, snr_q25, snr_q75 = median_iqr(snr)
    tau_med, tau_q25, tau_q75 = median_iqr(tau)

    print("\n===== high-statistics summary =====")
    print(f"N valid events = {len(metrics['event_index'])}")
    print(f"projected peak median/IQR = {peak_med:.4f} [{peak_q25:.4f}, {peak_q75:.4f}] mV")
    print(f"SNR median/IQR            = {snr_med:.2f} [{snr_q25:.2f}, {snr_q75:.2f}]")
    print(f"tau_eff,2D median/IQR     = {tau_med:.4f} [{tau_q25:.4f}, {tau_q75:.4f}] us")
    print(
        "robust amplitude sigma (MAD) = "
        f"{robust_sigma(peak_mV):.4f} mV"
    )

    info_lines = [
        f"measurement_dir = {meas_dir}",
        f"MAX_EVENTS = {MAX_EVENTS}",
        f"n_valid_events = {len(metrics['event_index'])}",
        f"baseline_window_us = {BASELINE_WINDOW_US}",
        f"amp_window_us = {AMP_WINDOW_US}",
        f"integral_window_us = {INTEGRAL_WINDOW_US}",
        f"global_peak_time_us = {first['time_us'][first['peak_idx']]}",
        f"global_pulse_direction_ch0 = {first['pulse_unit'].real}",
        f"global_pulse_direction_ch1 = {first['pulse_unit'].imag}",
        f"laser_hz = {LASER_HZ}",
        f"thermal_hz = {THERMAL_HZ}",
        f"events_per_temp_cycle = {EVENTS_PER_TEMP_CYCLE}",
        f"n_phase_bins = {N_PHASE_BINS}",
        f"trend_block_size = {TREND_BLOCK_SIZE}",
        "",
        "used_files:",
        *[str(p) for p in first["files"]],
    ]
    out_info.write_text("\n".join(info_lines), encoding="utf-8")

    print("\nsaved:")
    for path in [
        out_overview,
        out_phase,
        out_dist,
        out_phase_csv,
        out_trend_csv,
        out_npz,
        out_info,
    ]:
        print(" ", path)
    if SAVE_EVENT_CSV:
        print(" ", out_csv)


if __name__ == "__main__":
    main()
