#!/usr/bin/env python3
"""
analyze_20260618_focal_10um_xscan.py

20260618 の focal 10 um x-scan を解析する。

対象例:
  5.443GHz_z=8.5mm_x=4.36mm_focal
  5.443GHz_z=8.5mm_x=4.37mm_focal
  ...
  5.443GHz_z=8.5mm_x=4.40mm_focal

目的:
  - 10 um 間隔の横方向 scan で、KID 周辺の位置応答に構造が見えるか調べる
  - pedestal を引いた 2D pulse amplitude
        A2D(t) = sqrt(<Δch0(t)>^2 + <Δch1(t)>^2)
    を主な応答量にする
  - peak amplitude, integrated response, tau_eff, peak time, pulse angle を位置ごとに比較する
  - effective Gaussian width を参考値として fit する
    ※これは laser spot / phonon transport / KID geometry の畳み込み幅であり、
      そのまま回路線幅とは解釈しない。

出力:
  ~/software/kidanalysis/data/20260618/focal_10um_xscan/
    focal_10um_xscan_summary.png
    focal_10um_xscan_waveforms.png
    focal_10um_xscan_summary.csv
    focal_10um_xscan_gaussian_fit.json
    run_info.txt
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DATE = "20260618"

# Scan condition from the focal folders
TARGET_FREQ_GHZ = 5.443
TARGET_Z_MM = 8.5
TARGET_X_VALUES_MM = [4.36, 4.37, 4.38, 4.39, 4.40]
REQUIRED_TAG_SUBSTRING = "focal"

FREQ_ATOL_GHZ = 1e-4
POS_ATOL_MM = 1e-6
NPZ_PATTERN = "wf_*.npz"

# None: use all available events. For an exactly equal event count per point,
# set a positive integer such as 1000.
MAX_EVENTS_PER_CONDITION = 1000

# Trigger-relative analysis windows [us]
BASELINE_WINDOW_US = None      # None -> all t < 0
AMP_WINDOW_US = (0.0, 1.5)
INTEGRAL_WINDOW_US = (0.0, 2.0)
PLOT_TIME_WINDOW_US = (-0.30, 2.00)

# Effective Gaussian fit:
# H(x) = baseline + amplitude * exp(-(x-x0)^2/(2 sigma^2))
# It is only a descriptive fit. With five x positions, its width can be unstable.
FIT_GAUSSIAN = True
GAUSSIAN_SIGMA_GRID_UM = np.linspace(2.0, 200.0, 995)
GAUSSIAN_CENTER_GRID_MARGIN_UM = 30.0

DPI = 300


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / DATA_DATE / "focal_10um_xscan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# External SSD first: contains waveform npz in this project setup.
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
# UTILITIES
# =============================================================================

MEASUREMENT_RE = re.compile(
    r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>.+))?$"
)


def scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else x


def make_time_axis_s(npts, sample_rate_hz, ref_position_percent):
    return (
        np.arange(npts, dtype=float)
        - npts * ref_position_percent / 100.0
    ) / sample_rate_hz


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def parse_measurement_dir(path: Path):
    match = MEASUREMENT_RE.match(path.name)
    if match is None:
        return None
    d = match.groupdict()
    return {
        "path": path,
        "freq_ghz": float(d["freq"]),
        "z_mm": float(d["z"]),
        "x_mm": float(d["x"]),
        "tag": d["tag"] or "",
    }


def has_waveform_npz(path: Path):
    return any(path.glob(NPZ_PATTERN))


def is_repeat_tag(tag: str):
    return any(word in tag.lower() for word in ["second", "third", "fourth", "fifth"])


# =============================================================================
# DISCOVERY / LOADING
# =============================================================================

def discover_xscan_measurements():
    """
    Use only focal data, exact target f/z, exact target x points.
    If duplicated folders exist, prefer the first root, normally the external SSD.
    """
    selected_by_x = {}

    print("\n===== input roots =====")
    for root in INPUT_ROOTS:
        root = Path(root).expanduser()
        print(root, "exists=", root.is_dir())
        if not root.is_dir():
            continue

        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue

            info = parse_measurement_dir(child)
            if info is None:
                continue

            tag = info["tag"]
            if REQUIRED_TAG_SUBSTRING.lower() not in tag.lower():
                continue
            if is_repeat_tag(tag):
                continue
            if not np.isclose(info["freq_ghz"], TARGET_FREQ_GHZ, atol=FREQ_ATOL_GHZ, rtol=0):
                continue
            if not np.isclose(info["z_mm"], TARGET_Z_MM, atol=POS_ATOL_MM, rtol=0):
                continue
            if not any(
                np.isclose(info["x_mm"], target_x, atol=POS_ATOL_MM, rtol=0)
                for target_x in TARGET_X_VALUES_MM
            ):
                continue
            if not has_waveform_npz(child):
                continue

            x = info["x_mm"]
            if x not in selected_by_x:
                selected_by_x[x] = info

    missing = [
        x for x in TARGET_X_VALUES_MM
        if x not in selected_by_x
    ]
    if missing:
        raise RuntimeError(
            f"Missing valid focal data for x = {missing} mm.\n"
            "Check folder names, tags, or TARGET_X_VALUES_MM."
        )

    result = [selected_by_x[x] for x in sorted(selected_by_x)]
    print("\nselected datasets:")
    for info in result:
        print(" ", info["path"])
    return result


def load_events(path: Path):
    files = sorted(path.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No {NPZ_PATTERN} in {path}")

    block0, block1, used_files = [], [], []
    time_ref = None
    n_loaded = 0

    for f in files:
        if MAX_EVENTS_PER_CONDITION is not None and n_loaded >= MAX_EVENTS_PER_CONDITION:
            break

        try:
            data = np.load(f)
        except Exception as exc:
            print("skip unreadable:", f.name, exc)
            continue

        required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        if any(key not in data.files for key in required):
            print("skip missing keys:", f.name)
            continue

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)
        if ch0.ndim == 1:
            ch0 = ch0[None, :]
        if ch1.ndim == 1:
            ch1 = ch1[None, :]

        if ch0.shape != ch1.shape:
            print("skip shape mismatch:", f.name)
            continue

        time_s = make_time_axis_s(
            int(scalar(data["npts"])),
            float(scalar(data["sample_rate"])),
            float(scalar(data["ref_position"])),
        )

        if time_ref is None:
            time_ref = time_s
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time-axis mismatch:", f.name)
            continue

        if MAX_EVENTS_PER_CONDITION is None:
            n_take = len(ch0)
        else:
            n_take = min(len(ch0), MAX_EVENTS_PER_CONDITION - n_loaded)

        block0.append(ch0[:n_take])
        block1.append(ch1[:n_take])
        used_files.append(f)
        n_loaded += n_take

        print(f"  load {f.name}: {n_take} events (total={n_loaded})")

    if time_ref is None or n_loaded == 0:
        raise RuntimeError(f"No valid waveform data in {path}")

    return time_ref, np.vstack(block0), np.vstack(block1), used_files


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(time_s, ch0, ch1):
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

    if baseline_mask.sum() < 3 or amp_mask.sum() < 3 or integral_mask.sum() < 3:
        raise RuntimeError("Analysis windows are too short.")

    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    d0 = ch0 - ped0[:, None]
    d1 = ch1 - ped1[:, None]

    # Noise-unbiased 2D mean waveform:
    # take norm after averaging I/Q, not mean of norms.
    mean0 = d0.mean(axis=0)
    mean1 = d1.mean(axis=0)
    a2d = np.hypot(mean0, mean1)

    amp_indices = np.where(amp_mask)[0]
    peak_idx = int(amp_indices[np.argmax(a2d[amp_mask])])

    peak = float(a2d[peak_idx])
    integral = float(np.trapezoid(a2d[integral_mask], time_us[integral_mask]))
    tau_eff = integral / peak if peak > 0 else np.nan

    pulse_vec = complex(mean0[peak_idx], mean1[peak_idx])
    pulse_unit = pulse_vec / abs(pulse_vec) if abs(pulse_vec) > 0 else 1.0 + 0.0j

    # Event-level scalar pulse amplitude for IQR uncertainty.
    # Project each event onto the mean pulse direction, then find its peak.
    projected = d0 * pulse_unit.real + d1 * pulse_unit.imag
    event_peak = np.max(projected[:, amp_mask], axis=1)
    ev_med, ev_q25, ev_q75 = median_iqr(event_peak)

    return {
        "time_us": time_us,
        "ped0": ped0,
        "ped1": ped1,
        "mean_dch0": mean0,
        "mean_dch1": mean1,
        "a2d": a2d,
        "peak_idx": peak_idx,
        "peak_time_us": float(time_us[peak_idx]),
        "peak_a2d_mean_V": peak,
        "integral_a2d_Vus": integral,
        "tau_eff_us": tau_eff,
        "pulse_dch0_V": pulse_vec.real,
        "pulse_dch1_V": pulse_vec.imag,
        "pulse_angle_deg": float(np.degrees(np.angle(pulse_vec))),
        "event_peak_proj_median_V": ev_med,
        "event_peak_proj_q25_V": ev_q25,
        "event_peak_proj_q75_V": ev_q75,
    }


# =============================================================================
# EFFECTIVE GAUSSIAN FIT: no scipy dependency
# =============================================================================

def fit_effective_gaussian(x_um, y):
    """
    Grid search over center and sigma.
    For each (center, sigma), solve y = c + A*g by linear least squares.
    Return a descriptive effective Gaussian fit.

    This width is an optical/phonon/geometrical convolution width, not automatically
    the physical resonator trace width.
    """
    x_um = np.asarray(x_um, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x_um) < 5 or not np.all(np.isfinite(y)):
        return None

    x_center_grid = np.linspace(
        np.min(x_um) - GAUSSIAN_CENTER_GRID_MARGIN_UM,
        np.max(x_um) + GAUSSIAN_CENTER_GRID_MARGIN_UM,
        800,
    )

    best = None
    for center in x_center_grid:
        for sigma in GAUSSIAN_SIGMA_GRID_UM:
            g = np.exp(-0.5 * ((x_um - center) / sigma)**2)
            mat = np.column_stack([np.ones_like(g), g])
            coeff, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
            baseline, amplitude = coeff
            model = baseline + amplitude * g
            rss = float(np.sum((y - model)**2))

            if best is None or rss < best["rss"]:
                best = {
                    "center_um": float(center),
                    "sigma_um": float(sigma),
                    "fwhm_um": float(2.354820045 * sigma),
                    "baseline": float(baseline),
                    "amplitude": float(amplitude),
                    "rss": rss,
                }

    return best


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


def plot_summary(df, fit_result, output_path):
    x_um = df["x_mm"].to_numpy() * 1e3
    peak = df["peak_a2d_mean_mV"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.5), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.35, w_pad=0.12, hspace=0.25, wspace=0.15)

    # (1) Fine spatial profile
    ax = axes[0, 0]
    ev = df["event_peak_proj_median_mV"].to_numpy()
    elo = ev - df["event_peak_proj_q25_mV"].to_numpy()
    ehi = df["event_peak_proj_q75_mV"].to_numpy() - ev
    ax.errorbar(
        x_um, ev,
        yerr=[elo, ehi],
        marker="o",
        capsize=3,
        label="event peak projection: median ± IQR",
    )
    ax.plot(x_um, peak, "s--", label=r"$A_{2D}$ of mean waveform")

    if fit_result is not None:
        xx = np.linspace(np.min(x_um) - 10, np.max(x_um) + 10, 400)
        yy = (
            fit_result["baseline"]
            + fit_result["amplitude"]
            * np.exp(-0.5 * ((xx - fit_result["center_um"]) / fit_result["sigma_um"])**2)
        )
        ax.plot(
            xx, yy,
            lw=1.5,
            label=(
                f"effective Gaussian fit\n"
                f"center={fit_result['center_um']:.1f} um, "
                f"FWHM={fit_result['fwhm_um']:.1f} um"
            ),
        )

    ax.set_title("10 um focal x-scan: pulse amplitude")
    ax.set_xlabel("laser x position [um]")
    ax.set_ylabel("pulse amplitude [mV]")
    ax.grid(True)
    legend_below(ax, ncol=1)

    # (2) Shape / timing
    ax = axes[0, 1]
    ax.plot(x_um, df["tau_eff_us"], "o-", label=r"$\tau_{\rm eff}$")
    ax.set_xlabel("laser x position [um]")
    ax.set_ylabel(r"$\tau_{\rm eff}$ [us]")
    ax.set_title("Pulse effective width")
    ax.grid(True)

    axr = ax.twinx()
    axr.plot(x_um, df["peak_time_us"], "s-", label="peak time")
    axr.set_ylabel("peak time [us]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    # (3) Integrated response and normalized amplitude
    ax = axes[1, 0]
    integral = df["integral_a2d_mVus"].to_numpy()
    ax.plot(x_um, integral, "o-", label=r"$\int A_{2D}\,dt$")
    ax.set_xlabel("laser x position [um]")
    ax.set_ylabel(r"integrated response [mV us]")
    ax.set_title("Integrated response")
    ax.grid(True)

    axr = ax.twinx()
    norm = peak / np.nanmax(peak)
    axr.plot(x_um, norm, "s--", label="normalized A2D peak")
    axr.set_ylabel("normalized peak response")
    axr.set_ylim(-0.05, 1.10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    # (4) Raw component mixture
    ax = axes[1, 1]
    ax.plot(x_um, df["pulse_dch0_mV"], "o-", label="ch0 at A2D peak")
    ax.plot(x_um, df["pulse_dch1_mV"], "s-", label="ch1 at A2D peak")
    ax.axhline(0.0, lw=1.0)
    ax.set_xlabel("laser x position [um]")
    ax.set_ylabel("pulse component [mV]")
    ax.set_title("Pulse-vector components across 10 um scan")
    ax.grid(True)

    axr = ax.twinx()
    axr.plot(x_um, df["pulse_angle_deg"], "^-", label="pulse angle")
    axr.set_ylabel("pulse-vector angle [deg]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    legend_below(ax, h1 + h2, l1 + l2, ncol=2)

    fig.suptitle(
        f"20260618 focal 10 um x-scan: {TARGET_FREQ_GHZ:.3f} GHz, "
        f"z={TARGET_Z_MM:.2f} mm",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def plot_waveforms(df, feature_map, output_path):
    x_values = df["x_mm"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 10.5), sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.20, w_pad=0.10, hspace=0.25, wspace=0.10)

    for x in x_values:
        feat = feature_map[float(x)]
        mask = (
            (feat["time_us"] >= PLOT_TIME_WINDOW_US[0])
            & (feat["time_us"] <= PLOT_TIME_WINDOW_US[1])
        )
        label = f"x={x*1e3:.0f} um"
        axes[0].plot(
            feat["time_us"][mask],
            feat["a2d"][mask] * 1e3,
            label=label,
        )

        peak = feat["peak_a2d_mean_V"]
        normalized = feat["a2d"] / peak if peak > 0 else feat["a2d"]
        axes[1].plot(
            feat["time_us"][mask],
            normalized[mask],
            label=label,
        )

    axes[0].set_title(r"Pedestal-subtracted $A_{2D}$ waveforms")
    axes[0].set_ylabel(r"$A_{2D}$ [mV]")
    axes[0].grid(True)

    axes[1].set_title("Same waveforms normalized by their own peak")
    axes[1].set_xlabel("time from trigger [us]")
    axes[1].set_ylabel(r"$A_{2D}/A_{2D,\rm peak}$")
    axes[1].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(x_values),
        fontsize=8,
        frameon=True,
    )

    fig.suptitle(
        "20260618 focal 10 um x-scan: waveform comparison\n"
        r"$A_{2D}(t)=\sqrt{\langle\Delta ch0\rangle^2+\langle\Delta ch1\rangle^2}$",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Output:", OUT_DIR)
    print("Condition:")
    print(f"  f={TARGET_FREQ_GHZ} GHz, z={TARGET_Z_MM} mm")
    print(f"  x={TARGET_X_VALUES_MM} mm")
    print(f"  max events / point={MAX_EVENTS_PER_CONDITION}")

    measurements = discover_xscan_measurements()

    rows = []
    feature_map = {}
    used_files_by_x = {}

    for info in measurements:
        x = info["x_mm"]
        print(f"\n===== x={x:.3f} mm =====")
        time_s, ch0, ch1, used_files = load_events(info["path"])
        feat = extract_features(time_s, ch0, ch1)

        rows.append({
            "folder": info["path"].name,
            "freq_ghz": info["freq_ghz"],
            "z_mm": info["z_mm"],
            "x_mm": x,
            "x_um": x * 1e3,
            "tag": info["tag"],
            "n_events": len(ch0),

            "peak_time_us": feat["peak_time_us"],
            "peak_a2d_mean_mV": feat["peak_a2d_mean_V"] * 1e3,
            "integral_a2d_mVus": feat["integral_a2d_Vus"] * 1e3,
            "tau_eff_us": feat["tau_eff_us"],

            "pulse_dch0_mV": feat["pulse_dch0_V"] * 1e3,
            "pulse_dch1_mV": feat["pulse_dch1_V"] * 1e3,
            "pulse_angle_deg": feat["pulse_angle_deg"],

            "event_peak_proj_median_mV": feat["event_peak_proj_median_V"] * 1e3,
            "event_peak_proj_q25_mV": feat["event_peak_proj_q25_V"] * 1e3,
            "event_peak_proj_q75_mV": feat["event_peak_proj_q75_V"] * 1e3,
        })

        feature_map[float(x)] = feat
        used_files_by_x[float(x)] = [str(p) for p in used_files]

    df = pd.DataFrame(rows).sort_values("x_mm").reset_index(drop=True)

    fit_result = None
    if FIT_GAUSSIAN:
        fit_result = fit_effective_gaussian(
            df["x_um"].to_numpy(),
            df["peak_a2d_mean_mV"].to_numpy(),
        )

    summary_png = OUT_DIR / "focal_10um_xscan_summary.png"
    waveform_png = OUT_DIR / "focal_10um_xscan_waveforms.png"
    csv_path = OUT_DIR / "focal_10um_xscan_summary.csv"
    fit_path = OUT_DIR / "focal_10um_xscan_gaussian_fit.json"
    info_path = OUT_DIR / "run_info.txt"

    plot_summary(df, fit_result, summary_png)
    plot_waveforms(df, feature_map, waveform_png)
    df.to_csv(csv_path, index=False)

    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fit_type": "effective Gaussian profile",
                "warning": (
                    "This fitted FWHM is an effective lateral response width. "
                    "It includes the convolution of laser spot size, optical alignment, "
                    "phonon transport, and KID geometry. It is not automatically equal "
                    "to the resonator linewidth."
                ),
                "fit_result": fit_result,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    info_lines = [
        f"data_date = {DATA_DATE}",
        f"target_freq_ghz = {TARGET_FREQ_GHZ}",
        f"target_z_mm = {TARGET_Z_MM}",
        f"target_x_values_mm = {TARGET_X_VALUES_MM}",
        f"required_tag_substring = {REQUIRED_TAG_SUBSTRING}",
        f"max_events_per_condition = {MAX_EVENTS_PER_CONDITION}",
        f"baseline_window_us = {BASELINE_WINDOW_US}",
        f"amp_window_us = {AMP_WINDOW_US}",
        f"integral_window_us = {INTEGRAL_WINDOW_US}",
        "",
        "used_files:",
    ]
    for x in sorted(used_files_by_x):
        info_lines.append(f"x={x:.3f} mm")
        info_lines.extend(used_files_by_x[x])

    info_path.write_text("\n".join(info_lines), encoding="utf-8")

    print("\n===== Result summary =====")
    print(df[
        [
            "x_um",
            "peak_a2d_mean_mV",
            "event_peak_proj_median_mV",
            "tau_eff_us",
            "pulse_angle_deg",
        ]
    ].to_string(index=False))

    if fit_result is not None:
        print("\nEffective Gaussian fit (descriptive only):")
        print(
            f"  center = {fit_result['center_um']:.2f} um\n"
            f"  sigma  = {fit_result['sigma_um']:.2f} um\n"
            f"  FWHM   = {fit_result['fwhm_um']:.2f} um"
        )

    print("\nsaved:")
    print(" ", summary_png)
    print(" ", waveform_png)
    print(" ", csv_path)
    print(" ", fit_path)
    print(" ", info_path)


if __name__ == "__main__":
    main()
