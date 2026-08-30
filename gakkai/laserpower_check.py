"""Laser-power scan summary with per-position median/error-band plots.

The script looks for data collected under a root directory such as:
    /Volumes/NO NAME/data/20260825/

and expects one or more subdirectories organized as:
    Filter=1/1/...
    Filter=1/10/...
    ...
    Filter=1/1000/...

Within each filter folder, the data are expected to be in date folders like:
    data_0826_123456/
which contain waveform files such as wf_*.npz.

For every valid x,z position having all target filter values available, the script
computes:
    - amp median with 16–84% band
    - tau_r_eff median with 16–84% band
    - tau_d_eff median with 16–84% band

and saves a PDF in the current working directory.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT_CANDIDATES = [
    Path("/Volumes/NO NAME/data/20260825"),
    Path("/Volumes/NO NAME/data/20260826"),
    Path("/Volumes/NO NAME/data"),
    Path.cwd(),
]
ROOT_DIR = next((candidate for candidate in ROOT_CANDIDATES if candidate.exists()), ROOT_CANDIDATES[0])
TARGET_FILTERS = ["1/1", "1/10", "1/40", "1/100", "1/200", "1/400", "1/1000"]
TARGET_FILTER_VALUES = {
    "1/1": 1.0,
    "1/10": 0.1,
    "1/40": 0.025,
    "1/100": 0.01,
    "1/200": 0.005,
    "1/400": 0.0025,
    "1/1000": 0.001,
}

# The actual dataset layout is not Filter=... folders.
# The file names are date folders like data_0825_154145, and the filter/position
# relation is recovered from the measurement table shown in the image and written
# directly here.
# This is the minimum explicit mapping needed for the 20260825 dataset.
FOLDER_FILTER_POSITION = {
    "data_0825_154145": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/1"},
    "data_0825_154219": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/10"},
    "data_0825_154411": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/40"},
    "data_0825_134832": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/100"},
    "data_0825_154541": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/200"},
    "data_0825_143227": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/400"},
    "data_0825_154628": {"x_mm": 5.10, "z_mm": 6.70, "filter": "1/1000"},

    "data_0825_150934": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/1"},
    "data_0825_151552": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/10"},
    "data_0825_151745": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/40"},
    "data_0825_134208": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/100"},
    "data_0825_151840": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/200"},
    "data_0825_144055": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/400"},
    "data_0825_151928": {"x_mm": 5.10, "z_mm": 6.00, "filter": "1/1000"},

    "data_0825_152048": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/1"},
    "data_0825_152415": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/10"},
    "data_0825_152623": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/40"},
    "data_0825_134447": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/100"},
    "data_0825_152711": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/200"},
    "data_0825_143634": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/400"},
    "data_0825_152800": {"x_mm": 5.10, "z_mm": 6.30, "filter": "1/1000"},

    "data_0825_154756": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/1"},
    "data_0825_154834": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/10"},
    "data_0825_154918": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/40"},
    "data_0825_131518": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/100"},
    "data_0825_155009": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/200"},
    "data_0825_144740": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/400"},
    "data_0825_155057": {"x_mm": 4.60, "z_mm": 6.00, "filter": "1/1000"},
}

BIN_SIZE = 1
THRESHOLD_FRACTION = 0.10


def linear_crossing(time_ns, signal, level, start, stop, direction):
    """Return the linearly interpolated time where signal reaches level."""
    if direction == "up":
        condition = lambda left, right: left < level <= right
        indices = range(start, stop)
    elif direction == "up_from_peak":
        condition = lambda left, right: left < level <= right
        indices = range(start - 1, stop - 1, -1)
    else:
        condition = lambda left, right: left >= level > right
        indices = range(start, stop)

    for index in indices:
        if condition(signal[index], signal[index + 1]):
            left_time = time_ns[index]
            right_time = time_ns[index + 1]
            left_signal = signal[index]
            right_signal = signal[index + 1]
            if right_signal == left_signal:
                return 0.5 * (left_time + right_time)
            return left_time + (level - left_signal) * (
                right_time - left_time
            ) / (right_signal - left_signal)
    return np.nan


def integral_between(time_ns, signal, first_time, last_time):
    """Integrate signal between two times using trapezoids."""
    if not np.isfinite(first_time) or not np.isfinite(last_time):
        return np.nan
    if last_time <= first_time:
        return np.nan

    inside = (time_ns > first_time) & (time_ns < last_time)
    integration_time = np.concatenate(([first_time], time_ns[inside], [last_time]))
    integration_signal = np.interp(integration_time, time_ns, signal)
    return np.trapezoid(integration_signal, integration_time)


def analyze_event(time_ns, ch0_waveform, ch1_waveform, ref_position):
    """Match amp_tau_dr_integ_su.py exactly for tau calculations."""
    pedestal_end = (ref_position - 10.0) / 100.0 * time_ns.size
    pedestal_end = max(1, min(time_ns.size, int(pedestal_end)))
    ped0 = np.mean(ch0_waveform[:pedestal_end])
    ped1 = np.mean(ch1_waveform[:pedestal_end])

    ch0_signal = ch0_waveform - ped0
    ch1_signal = ch1_waveform - ped1
    signal = np.hypot(ch0_signal, ch1_signal)

    peak_index = int(np.argmax(signal))
    amp = signal[peak_index]
    peak_time = time_ns[peak_index]

    level = THRESHOLD_FRACTION * amp
    t10_left = linear_crossing(time_ns, signal, level, peak_index, 0, "up_from_peak")
    t10_right = linear_crossing(time_ns, signal, level, peak_index, signal.size - 1, "down")

    right_integral_end = t10_right if np.isfinite(t10_right) else time_ns[-1]
    left_integral = integral_between(time_ns, signal, t10_left, peak_time)
    right_integral = integral_between(time_ns, signal, peak_time, right_integral_end)

    tau_r_eff = left_integral / amp if np.isfinite(left_integral) else np.nan
    tau_d_eff = right_integral / amp if np.isfinite(right_integral) else np.nan
    fwhm_10 = t10_right - t10_left

    result = {
        "amp": amp,
        "t10_left": t10_left,
        "t_peak": peak_time,
        "t10_right": t10_right,
        "left_integral": left_integral,
        "right_integral": right_integral,
        "tau_r_eff": tau_r_eff,
        "tau_d_eff": tau_d_eff,
        "fwhm_10": fwhm_10,
    }
    return result, signal


def bin_waveforms(ch0, ch1, npts_raw, sample_rate, ref_position):
    """Average BIN_SIZE samples and return binned waveform and time axis."""
    nbin = npts_raw // BIN_SIZE
    usable_npts = nbin * BIN_SIZE
    ch0_binned = ch0[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)
    ch1_binned = ch1[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)

    raw_time = (np.arange(usable_npts) - npts_raw * ref_position / 100.0) / sample_rate
    time_ns = raw_time.reshape(nbin, BIN_SIZE).mean(axis=1) * 1e9
    return ch0_binned, ch1_binned, time_ns


def _normalize_filter_label(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("Filter=", "").replace("filter=", "")
    cleaned = cleaned.replace("_", "/")
    if cleaned.startswith("/"):
        cleaned = cleaned[1:]
    return cleaned


def find_filter_dirs(root: Path) -> Dict[str, Path]:
    """The image-based dataset does not use Filter=... directories.

    Instead, the relationship is recovered from explicit folder-name lists written in
    FOLDER_FILTER_POSITION, and each folder is then searched for wf_*.npz files under
    the root /Volumes/NO NAME/data/20260825 directory.
    """
    labels: Dict[str, Path] = {}
    if not root.exists():
        return labels

    for folder_name, meta in FOLDER_FILTER_POSITION.items():
        folder_path = root / folder_name
        if folder_path.exists() and folder_path.is_dir():
            labels[meta["filter"]] = folder_path
    return labels


def parse_position_from_npz(npz_path: Path):
    """Parse x,z if embedded in an NPZ file; otherwise return None."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    for key in ("x_position", "x", "x_mm", "X", "z_position", "z", "z_mm", "Z"):
        if key in data:
            if key.startswith("x"):
                return float(np.asarray(data[key]).reshape(-1)[0]), None
            if key.startswith("z"):
                return None, float(np.asarray(data[key]).reshape(-1)[0])

    # Some files store x,z as arrays or nested metadata.
    for key in ("position", "pos"):
        if key in data:
            arr = np.asarray(data[key]).reshape(-1)
            if arr.size >= 2:
                return float(arr[0]), float(arr[1])

    return None


def load_event_rows_from_npz(npz_path: Path, *, require_valid_position: bool = False) -> List[dict]:
    """Load event-wise amp/tau values from a waveform NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    if "ch0" not in data or "ch1" not in data:
        return []

    npts_raw = int(data["npts"])
    ref_position = float(data["ref_position"])
    sample_rate = float(data["sample_rate"])
    ch0, ch1, time_ns = bin_waveforms(data["ch0"], data["ch1"], npts_raw, sample_rate, ref_position)

    rows: List[dict] = []
    n_events = min(ch0.shape[0], ch1.shape[0])
    for event_index in range(n_events):
        ch0_event = ch0[event_index]
        ch1_event = ch1[event_index]
        result, signal = analyze_event(time_ns, ch0_event, ch1_event, ref_position)
        amp = result["amp"]
        if np.isfinite(amp) and amp > 0:
            pedestal_end = max(1, min(ch0_event.size, int((ref_position - 10.0) / 100.0 * ch0_event.size)))
            ch0_signal = ch0_event - np.mean(ch0_event[:pedestal_end])
            ch1_signal = ch1_event - np.mean(ch1_event[:pedestal_end])
            result["waveform_ch0_norm"] = ch0_signal / amp
            result["waveform_ch1_norm"] = ch1_signal / amp
            result["waveform_abs_norm"] = signal / amp
        else:
            result["waveform_ch0_norm"] = np.full(ch0_event.shape, np.nan)
            result["waveform_ch1_norm"] = np.full(ch1_event.shape, np.nan)
            result["waveform_abs_norm"] = np.full(signal.shape, np.nan)
        result["time_ns"] = time_ns
        result["event_index"] = event_index
        rows.append(result)
    return rows


def select_filtered_data(root: Path) -> Dict[Tuple[float, float], Dict[str, List[dict]]]:
    """Use the explicit data_0825_* -> filter/position mapping instead of searching for Filter=... folders."""
    selected: Dict[Tuple[float, float], Dict[str, List[dict]]] = {}

    for folder_name, meta in FOLDER_FILTER_POSITION.items():
        folder_path = root / folder_name
        if not folder_path.exists():
            continue

        filter_label = meta["filter"]
        if filter_label not in TARGET_FILTER_VALUES:
            continue

        npz_files = sorted(folder_path.glob("wf_*.npz"))
        if not npz_files:
            continue

        x_pos = float(meta["x_mm"])
        z_pos = float(meta["z_mm"])
        key = (x_pos, z_pos)
        selected.setdefault(key, {})
        selected[key].setdefault(filter_label, [])
        for npz_path in npz_files:
            selected[key][filter_label].extend(load_event_rows_from_npz(npz_path))

    return selected


def summarize_position_stats(position_data: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """Return median and 16–84% percentiles for amp/tau values for each filter."""
    summary: Dict[str, Dict[str, float]] = {}
    for filter_label, rows in position_data.items():
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        for metric in ("amp", "tau_r_eff", "tau_d_eff"):
            values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            median = float(np.median(values))
            p16 = float(np.percentile(values, 16))
            p84 = float(np.percentile(values, 84))
            summary.setdefault(filter_label, {})[metric] = median
            summary[filter_label][f"{metric}_p16"] = p16
            summary[filter_label][f"{metric}_p84"] = p84
    return summary


def _make_metric_page(pdf, position_stats, metric, ylabel, log_scale):
    """Create one spatially arranged page for one metric and power-axis scale."""
    positions = sorted(position_stats)
    x_values = sorted({key[0] for key in positions})
    z_values = sorted({key[1] for key in positions}, reverse=True)
    x_index = {value: index for index, value in enumerate(x_values)}
    z_index = {value: index for index, value in enumerate(z_values)}

    figure_width = max(10.0, 4.2 * len(x_values))
    figure_height = max(7.0, 3.2 * len(z_values))
    fig, axes = plt.subplots(
        len(z_values),
        len(x_values),
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for z_value in z_values:
        for x_value in x_values:
            axis = axes[z_index[z_value], x_index[x_value]]
            stats_by_filter = position_stats.get((x_value, z_value))
            if stats_by_filter is None:
                axis.set_visible(False)
                continue

            labels = [label for label in TARGET_FILTERS if label in stats_by_filter]
            power = np.asarray([TARGET_FILTER_VALUES[label] for label in labels], dtype=float)
            median = np.asarray([stats_by_filter[label][metric] for label in labels], dtype=float)
            lower = np.asarray([
                stats_by_filter[label].get(f"{metric}_p16", stats_by_filter[label][metric])
                for label in labels
            ], dtype=float)
            upper = np.asarray([
                stats_by_filter[label].get(f"{metric}_p84", stats_by_filter[label][metric])
                for label in labels
            ], dtype=float)

            axis.errorbar(
                power,
                median,
                yerr=[median - lower, upper - median],
                fmt="o-",
                capsize=3,
                markersize=4,
                lw=1.2,
            )
            axis.set_title(f"x={x_value:.2f}, z={z_value:.2f} mm", fontsize=9)
            axis.grid(alpha=0.3)
            axis.set_xscale("log" if log_scale else "linear")
            if metric == "amp":
                axis.set_yscale("log")
            axis.set_xlabel("optical power")
            axis.set_ylabel(ylabel)

    scale_name = "log" if log_scale else "linear"
    fig.suptitle(f"-2 dBm: {metric} vs optical power ({scale_name} x-axis)", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _make_position_detail_page(pdf, x_z_key, stats_by_filter):
    """Create the original 3x2 metric/scale page for one x,z position."""
    positions = [label for label in TARGET_FILTERS if label in stats_by_filter]
    power = np.asarray([TARGET_FILTER_VALUES[label] for label in positions], dtype=float)
    metric_specs = [
        ("amp", "amp [mV]"),
        ("tau_r_eff", "tau_r_eff [ns]"),
        ("tau_d_eff", "tau_d_eff [ns]"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False, constrained_layout=True)

    for row_index, (metric, ylabel) in enumerate(metric_specs):
        for column_index, log_scale in enumerate((False, True)):
            axis = axes[row_index, column_index]
            median = np.asarray([stats_by_filter[label][metric] for label in positions], dtype=float)
            lower = np.asarray([
                stats_by_filter[label].get(f"{metric}_p16", stats_by_filter[label][metric])
                for label in positions
            ], dtype=float)
            upper = np.asarray([
                stats_by_filter[label].get(f"{metric}_p84", stats_by_filter[label][metric])
                for label in positions
            ], dtype=float)
            axis.errorbar(
                power,
                median,
                yerr=[median - lower, upper - median],
                fmt="o-",
                capsize=4,
                markersize=5,
                lw=1.5,
            )
            axis.set_xscale("log" if log_scale else "linear")
            # プロットの y 軸を対数スケールに設定するのは、amp メトリックの場合のみです。これをコメントアウトでノーマルに戻せる
            if metric == "amp":
                axis.set_yscale("log")
            ###
            axis.set_title(f"{metric} ({'log' if log_scale else 'linear'} x)")
            axis.set_xlabel("optical power")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)

    x_value, z_value = x_z_key
    fig.suptitle(f"-2 dBm: x={x_value:.2f} mm, z={z_value:.2f} mm", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _make_normalized_waveform_page(pdf, waveform_stats, waveform_key, title):
    """Plot normalized waveform median and 16-84% band at every x,z position."""
    positions = sorted(waveform_stats)
    x_values = sorted({key[0] for key in positions})
    z_values = sorted({key[1] for key in positions}, reverse=True)
    x_index = {value: index for index, value in enumerate(x_values)}
    z_index = {value: index for index, value in enumerate(z_values)}
    fig, axes = plt.subplots(
        len(z_values),
        len(x_values),
        figsize=(max(10.0, 4.2 * len(x_values)), max(7.0, 3.2 * len(z_values))),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(TARGET_FILTERS)))
    color_by_filter = dict(zip(TARGET_FILTERS, colors))

    for (x_value, z_value), filter_data in waveform_stats.items():
        axis = axes[z_index[z_value], x_index[x_value]]
        for filter_label in TARGET_FILTERS:
            event_waveforms = filter_data.get(filter_label, [])
            if not event_waveforms:
                continue
            time_ns = event_waveforms[0]["time_ns"]
            waveforms = np.asarray([event[waveform_key] for event in event_waveforms], dtype=float)
            valid = np.any(np.isfinite(waveforms), axis=1)
            waveforms = waveforms[valid]
            if waveforms.size == 0:
                continue
            median = np.nanmedian(waveforms, axis=0)
            lower = np.nanpercentile(waveforms, 16, axis=0)
            upper = np.nanpercentile(waveforms, 84, axis=0)
            color = color_by_filter[filter_label]
            axis.plot(time_ns, median, color=color, lw=1.4, label=filter_label)
            axis.fill_between(time_ns, lower, upper, color=color, alpha=0.12)
        axis.set_title(f"x={x_value:.2f}, z={z_value:.2f} mm", fontsize=9)
        axis.grid(alpha=0.3)
        axis.set_xlabel("time [ns]")
        axis.set_ylabel("normalized amplitude")

    legend_handles = [
        Line2D([0], [0], color=color_by_filter[label], lw=1.8, label=label)
        for label in TARGET_FILTERS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        title="filter",
        frameon=True,
    )
    fig.suptitle(f"-2 dBm: {title} / amp, median with 16-84% band", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def _correlation_and_slope(amp_values, tau_values):
    """Return Pearson r and least-squares slope for finite event values."""
    valid = np.isfinite(amp_values) & np.isfinite(tau_values)
    amp_values = np.asarray(amp_values[valid], dtype=float)
    tau_values = np.asarray(tau_values[valid], dtype=float)
    if amp_values.size < 2 or np.ptp(amp_values) == 0:
        return np.nan, np.nan
    correlation = float(np.corrcoef(amp_values, tau_values)[0, 1])
    slope = float(np.polyfit(amp_values, tau_values, 1)[0])
    return correlation, slope


def _make_amp_tau_scatter_page(pdf, waveform_stats, tau_key, tau_label):
    """Plot event-wise amp-tau scatter and per-filter correlation summaries."""
    positions = sorted(waveform_stats)
    x_values = sorted({key[0] for key in positions})
    z_values = sorted({key[1] for key in positions}, reverse=True)
    x_index = {value: index for index, value in enumerate(x_values)}
    z_index = {value: index for index, value in enumerate(z_values)}
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(TARGET_FILTERS)))
    color_by_filter = dict(zip(TARGET_FILTERS, colors))
    fig, axes = plt.subplots(
        len(z_values),
        len(x_values),
        figsize=(max(10.0, 4.2 * len(x_values)), max(7.0, 3.2 * len(z_values))),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for (x_value, z_value), filter_data in waveform_stats.items():
        axis = axes[z_index[z_value], x_index[x_value]]
        summary_lines = []
        for filter_label in TARGET_FILTERS:
            rows = filter_data.get(filter_label, [])
            if not rows:
                continue
            amp_values = np.asarray([row["amp"] for row in rows], dtype=float)
            tau_values = np.asarray([row[tau_key] for row in rows], dtype=float)
            valid = np.isfinite(amp_values) & np.isfinite(tau_values)
            if not np.any(valid):
                continue
            color = color_by_filter[filter_label]
            axis.scatter(
                amp_values[valid],
                tau_values[valid],
                s=7,
                alpha=0.35,
                color=color,
                label=filter_label,
            )
            correlation, slope = _correlation_and_slope(amp_values, tau_values)
            summary_lines.append(f"{filter_label}: r={correlation:.2f}, slope={slope:.3g}")

        axis.set_title(f"x={x_value:.2f}, z={z_value:.2f} mm", fontsize=9)
        axis.set_xlabel("amp")
        axis.set_ylabel(tau_label)
        axis.set_xscale("log")
        axis.grid(alpha=0.3)
        if summary_lines:
            axis.text(
                0.03,
                0.97,
                "\n".join(summary_lines),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               color=color_by_filter[label], label=label)
        for label in TARGET_FILTERS
    ]
    fig.legend(handles=legend_handles, loc="upper right", title="filter", frameon=True)
    fig.suptitle(f"-2 dBm: log(amp) vs {tau_label} event scatter", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def create_position_pdf(root: Path, output_path: Path):
    """Create the final PDF with one page per x,z position."""
    selected = select_filtered_data(root)
    filtered_positions = []
    waveform_positions = {}
    for key, filter_map in sorted(selected.items()):
        stats = summarize_position_stats(filter_map)
        if not stats:
            continue
        if all(label in stats for label in TARGET_FILTERS):
            filtered_positions.append((key, stats))
            waveform_positions[key] = {
                label: filter_map.get(label, []) for label in TARGET_FILTERS
            }

    if not filtered_positions:
        raise RuntimeError(
            f"No valid x,z positions with all requested filters under {root}. "
            f"Expected folders like Filter=1/1, 1/10, ..., 1/1000. "
            f"In this environment the existing data root is {root}, but it does not contain that structure."
        )

    position_stats = dict(filtered_positions)
    metric_specs = [
        ("amp", "amp [mV]"),
        ("tau_r_eff", "tau_r_eff [ns]"),
        ("tau_d_eff", "tau_d_eff [ns]"),
    ]
    with PdfPages(output_path) as pdf:
        for metric, ylabel in metric_specs:
            _make_metric_page(pdf, position_stats, metric, ylabel, log_scale=True)
        _make_normalized_waveform_page(pdf, waveform_positions, "waveform_ch0_norm", "ch0")
        _make_normalized_waveform_page(pdf, waveform_positions, "waveform_ch1_norm", "ch1")
        _make_normalized_waveform_page(pdf, waveform_positions, "waveform_abs_norm", "absolute")
        _make_amp_tau_scatter_page(pdf, waveform_positions, "tau_r_eff", "tau_r_eff")
        _make_amp_tau_scatter_page(pdf, waveform_positions, "tau_d_eff", "tau_d_eff")
        for key, stats in filtered_positions:
            _make_position_detail_page(pdf, key, stats)

    print(
        f"saved {output_path} with {8 + len(filtered_positions)} pages "
        f"(3 log-power pages + 3 normalized-waveform pages + 2 amp-tau scatter pages + "
        f"{len(filtered_positions)} position detail pages)"
    )


def main():
    output_path = Path.cwd() / "laser_power_position_summary.pdf"
    root = ROOT_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"Data root not found: {root}. "
            "The dataset is under /Volumes/NO NAME/data/20260825 or /Volumes/NO NAME/data/20260826 and uses data_0825_XXXXXX folders, not Filter=... folders."
        )
    create_position_pdf(root, output_path)


if __name__ == "__main__":
    main()
