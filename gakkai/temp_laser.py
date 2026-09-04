"""Analyze IQ waveforms from .npz files and save per-file + position-summary PDFs.

This script creates the following outputs for each input .npz file:
  - <stem>_amp_tau_eff.csv
  - <stem>_amp_tau_eff.pdf

It also creates a final summary PDF named by default
  temp_laser_summary.pdf
which contains four pages arranged in an x/z position grid:
  1. amp histograms by position
  2. tau_r_eff histograms by position
  3. tau_d_eff histograms by position
  4. amplitude vs tau correlation by position

Examples:
    ./.venv/bin/python temp_laser.py 20260825_combined.npz
    ./.venv/bin/python temp_laser.py --batch
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


BIN_SIZE = 1
THRESHOLD_FRACTION = 0.10
AMP_THRESHOLD = 0.002
TAU_R_EFF_THRESHOLD = 20.0
TAU_D_EFF_THRESHOLD = 20.0

FOLDER_TEMP_POSITION = {
    "data_0825_131115": {"x_mm": 4.60, "z_mm": 6.70, "temp": "4.68K"},
    "data_0825_165155": {"x_mm": 4.60, "z_mm": 6.70, "temp": "5.22K"},
    "data_0825_171703": {"x_mm": 4.60, "z_mm": 6.70, "temp": "5.71K"},
    "data_0825_131249": {"x_mm": 4.60, "z_mm": 6.30, "temp": "4.68K"},
    "data_0825_165101": {"x_mm": 4.60, "z_mm": 6.30, "temp": "5.22K"},
    "data_0825_171758": {"x_mm": 4.60, "z_mm": 6.30, "temp": "5.71K"},
    "data_0825_131518": {"x_mm": 4.60, "z_mm": 6.00, "temp": "4.68K"},
    "data_0825_165000": {"x_mm": 4.60, "z_mm": 6.00, "temp": "5.22K"},
    "data_0825_171843": {"x_mm": 4.60, "z_mm": 6.00, "temp": "5.71K"},
    "data_0825_134832": {"x_mm": 5.10, "z_mm": 6.70, "temp": "4.68K"},
    "data_0825_164423": {"x_mm": 5.10, "z_mm": 6.70, "temp": "5.22K"},
    "data_0825_172226": {"x_mm": 5.10, "z_mm": 6.70, "temp": "5.71K"},
    "data_0825_134710": {"x_mm": 5.10, "z_mm": 6.50, "temp": "4.68K"},
    "data_0825_164603": {"x_mm": 5.10, "z_mm": 6.50, "temp": "5.22K"},
    "data_0825_172148": {"x_mm": 5.10, "z_mm": 6.50, "temp": "5.71K"},
    "data_0825_134447": {"x_mm": 5.10, "z_mm": 6.30, "temp": "4.68K"},
    "data_0825_164654": {"x_mm": 5.10, "z_mm": 6.30, "temp": "5.22K"},
    "data_0825_172106": {"x_mm": 5.10, "z_mm": 6.30, "temp": "5.71K"},
    "data_0825_134328": {"x_mm": 5.10, "z_mm": 6.10, "temp": "4.68K"},
    "data_0825_164746": {"x_mm": 5.10, "z_mm": 6.10, "temp": "5.22K"},
    "data_0825_172027": {"x_mm": 5.10, "z_mm": 6.10, "temp": "5.71K"},
    "data_0825_134208": {"x_mm": 5.10, "z_mm": 6.00, "temp": "4.68K"},
    "data_0825_164855": {"x_mm": 5.10, "z_mm": 6.00, "temp": "5.22K"},
    "data_0825_171939": {"x_mm": 5.10, "z_mm": 6.00, "temp": "5.71K"},
}


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
    """Analyze one event and return its parameters and plotting information."""
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

    left_integral = integral_between(time_ns, signal, t10_left, peak_time)
    right_integral_end = t10_right if np.isfinite(t10_right) else time_ns[-1]
    right_integral = integral_between(time_ns, signal, peak_time, right_integral_end)

    tau_r_eff = left_integral / amp if np.isfinite(left_integral) else np.nan
    tau_d_eff = right_integral / amp if np.isfinite(right_integral) else np.nan
    fwhm_10 = t10_right - t10_left

    result = {
        "ped0": ped0,
        "ped1": ped1,
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
    """Average BIN_SIZE samples and return waveforms and time axis."""
    nbin = npts_raw // BIN_SIZE
    usable_npts = nbin * BIN_SIZE
    ch0_binned = ch0[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)
    ch1_binned = ch1[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)

    raw_time = (np.arange(usable_npts) - npts_raw * ref_position / 100.0) / sample_rate
    time_ns = raw_time.reshape(nbin, BIN_SIZE).mean(axis=1) * 1e9
    return ch0_binned, ch1_binned, time_ns


def process_single_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    npts_raw = int(data["npts"])
    ref_position = float(data["ref_position"])
    sample_rate = float(data["sample_rate"])
    ch0, ch1, time_ns = bin_waveforms(data["ch0"], data["ch1"], npts_raw, sample_rate, ref_position)

    nwf = min(ch0.shape[0], ch1.shape[0])
    results = []
    signals = []

    for event_index in range(nwf):
        result, signal = analyze_event(time_ns, ch0[event_index], ch1[event_index], ref_position)
        result["event"] = event_index
        results.append(result)
        signals.append(signal)

    columns = [
        "event",
        "ped0",
        "ped1",
        "amp",
        "t10_left",
        "t_peak",
        "t10_right",
        "left_integral",
        "right_integral",
        "tau_r_eff",
        "tau_d_eff",
        "fwhm_10",
    ]
    parameters = pd.DataFrame(results)[columns]
    amplitude = pd.to_numeric(parameters["amp"], errors="coerce")
    tau_r_eff = pd.to_numeric(parameters["tau_r_eff"], errors="coerce")
    tau_d_eff = pd.to_numeric(parameters["tau_d_eff"], errors="coerce")
    finite = np.isfinite(amplitude) & np.isfinite(tau_r_eff) & np.isfinite(tau_d_eff)
    selected = finite & (
        (amplitude > AMP_THRESHOLD)
        | ((tau_r_eff > TAU_R_EFF_THRESHOLD) & (tau_d_eff > TAU_D_EFF_THRESHOLD))
    )
    not_selected = finite & ~selected

    basename = os.path.splitext(os.path.basename(npz_path))[0]
    csv_name = npz_path.with_name(basename + "_amp_tau_eff.csv")
    pdf_name = npz_path.with_name(basename + "_amp_tau_eff.pdf")
    parameters.to_csv(csv_name, index=False)

    with PdfPages(pdf_name) as pdf:
        ncol = 4
        nrow = 4

        figure, axes = plt.subplots(nrow, ncol, figsize=(16, 9), sharex=True, sharey=True)
        axes = np.asarray(axes).ravel()
        for event_index in range(min(nrow * ncol, nwf)):
            axis = axes[event_index]
            axis.plot(time_ns, ch0[event_index] * 1e3, color="C0", linewidth=0.8, label="ch0")
            axis.plot(time_ns, ch1[event_index] * 1e3, color="C1", linewidth=0.8, label="ch1")
            axis.set_title(f"event {event_index}")
            axis.grid(alpha=0.3)
            if event_index == 0:
                axis.legend(fontsize=7, loc="best")
        for axis in axes:
            axis.set_xlabel("Time [ns]")
            axis.set_ylabel("raw signal [mV]")
        figure.suptitle("Raw ch0/ch1, pedestal not subtracted")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(nrow, ncol, figsize=(16, 9), sharex=True, sharey=True)
        axes = np.asarray(axes).ravel()
        for event_index in range(min(nrow * ncol, nwf)):
            axis = axes[event_index]
            row = parameters.iloc[event_index]
            axis.plot(time_ns, signals[event_index] * 1e3, color="C0")
            axis.axhline(row["amp"] * 1e2, color="C1", ls="--", alpha=0.7)
            axis.axvline(row["t10_left"], color="C2", ls=":")
            axis.axvline(row["t_peak"], color="k", ls="-")
            axis.axvline(row["t10_right"], color="C3", ls=":")
            axis.set_title(f"event {event_index}\ntau_r={row['tau_r_eff']:.1f}, tau_d={row['tau_d_eff']:.1f}")
            axis.grid(alpha=0.3)
        for axis in axes:
            axis.set_xlabel("Time [ns]")
            axis.set_ylabel("sqrt(ch0^2 + ch1^2) [mV]")
        figure.suptitle("Absolute signal")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        histogram_names = [
            "amp",
            "t10_left",
            "t_peak",
            "t10_right",
            "left_integral",
            "right_integral",
            "tau_r_eff",
            "tau_d_eff",
            "fwhm_10",
        ]

        figure, axes = plt.subplots(3, 3, figsize=(16, 9))
        for axis, name in zip(axes.ravel(), histogram_names):
            values = pd.to_numeric(parameters[name], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                axis.hist(values, bins=100, color="C0")
            axis.set_xlabel(name)
            axis.set_ylabel("counts")
            axis.grid(alpha=0.3)
        figure.suptitle("All events")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 3, figsize=(16, 9))
        for axis, name in zip(axes.ravel(), histogram_names):
            values = pd.to_numeric(parameters[name], errors="coerce")
            for mask, label, color in ((selected, "selected", "C0"), (not_selected, "not selected", "C1")):
                group_values = values[mask].to_numpy(dtype=float)
                group_values = group_values[np.isfinite(group_values)]
                if group_values.size:
                    axis.hist(group_values, bins=100, histtype="step", color=color, linewidth=1.2, label=label)
            axis.set_xlabel(name)
            axis.set_ylabel("counts")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        figure.suptitle(
            f"Thresholded events: amp > {AMP_THRESHOLD} or "
            f"(tau_r_eff > {TAU_R_EFF_THRESHOLD} and tau_d_eff > {TAU_D_EFF_THRESHOLD})"
        )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

    print(f"Processed {npz_path.name}: {len(parameters)} events")
    print(f"Saved {csv_name} and {pdf_name}")
    return csv_name


def find_target_dirs(root_dir: Path):
    """Return only the measurement folders explicitly listed in FOLDER_TEMP_POSITION."""
    if root_dir.is_file():
        return [root_dir.parent] if root_dir.name in FOLDER_TEMP_POSITION else []

    matched = []
    for name in FOLDER_TEMP_POSITION:
        folder = root_dir / name
        if folder.is_dir():
            matched.append(folder)
    return sorted(matched)


def find_npz_files_for_target_root(root_dir: Path):
    """Collect .npz files from only the FOLDER_TEMP_POSITION folders under a root."""
    target_dirs = find_target_dirs(root_dir)
    if not target_dirs:
        return []

    npz_files = []
    for target_dir in target_dirs:
        npz_files.extend(sorted(target_dir.glob("*.npz")))
    return npz_files


def make_position_histogram_page(frame: pd.DataFrame, metric_name: str, title: str):
    z_values = sorted({float(row["z_mm"]) for _, row in frame.iterrows()}, reverse=True)
    x_values = sorted({float(row["x_mm"]) for _, row in frame.iterrows()})

    nrow = len(z_values)
    ncol = len(x_values)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * max(1, ncol), 3.0 * max(1, nrow)), squeeze=False)
    axes = axes.ravel()

    temp_colors = {
        "4.68K": "C0",
        "5.22K": "C1",
        "5.71K": "C2",
    }

    for row_index, z_mm in enumerate(z_values):
        for col_index, x_mm in enumerate(x_values):
            axis = axes[row_index * ncol + col_index]
            subset = frame.loc[(frame["x_mm"] == x_mm) & (frame["z_mm"] == z_mm)]
            values = pd.to_numeric(subset[metric_name], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size:
                for temp, color in temp_colors.items():
                    temp_values = pd.to_numeric(
                        subset.loc[subset["temp"] == temp, metric_name],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    temp_values = temp_values[np.isfinite(temp_values)]
                    if temp_values.size:
                        axis.hist(temp_values, bins=30, histtype="step", color=color, linewidth=1.2, alpha=0.9, label=f"{temp}")

            axis.set_title(f"x={x_mm:.2f}, z={z_mm:.2f}", fontsize=8)
            axis.grid(alpha=0.25)
            axis.set_xlabel(metric_name)
            axis.set_ylabel("counts")

    for axis in axes[nrow * ncol:]:
        pass

    handles, labels = axes[0].get_legend_handles_labels() if axes.size else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_position_correlation_page(frame: pd.DataFrame, metric_name: str, title: str):
    z_values = sorted({float(row["z_mm"]) for _, row in frame.iterrows()}, reverse=True)
    x_values = sorted({float(row["x_mm"]) for _, row in frame.iterrows()})

    nrow = len(z_values)
    ncol = len(x_values)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * max(1, ncol), 3.0 * max(1, nrow)), squeeze=False)
    axes = axes.ravel()

    temp_colors = {
        "4.68K": "C0",
        "5.22K": "C1",
        "5.71K": "C2",
    }

    for row_index, z_mm in enumerate(z_values):
        for col_index, x_mm in enumerate(x_values):
            axis = axes[row_index * ncol + col_index]
            subset = frame.loc[(frame["x_mm"] == x_mm) & (frame["z_mm"] == z_mm)]
            for temp, color in temp_colors.items():
                temp_subset = subset.loc[subset["temp"] == temp]
                amp = pd.to_numeric(temp_subset["amp"], errors="coerce").to_numpy(dtype=float)
                tau = pd.to_numeric(temp_subset[metric_name], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(amp) & np.isfinite(tau)
                if finite.any():
                    axis.scatter(tau[finite], amp[finite], s=12, alpha=0.75, color=color, label=f"{temp}")

            axis.set_title(f"x={x_mm:.2f}, z={z_mm:.2f}", fontsize=8)
            axis.grid(alpha=0.25)
            axis.set_xlabel(f"{metric_name} [ns]")
            axis.set_ylabel("amp")

    for axis in axes[nrow * ncol:]:
        pass

    handles, labels = axes[0].get_legend_handles_labels() if axes.size else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def build_summary_pdf(csv_paths: list[Path], output_pdf: Path):
    frames = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        folder_name = csv_path.parent.name
        meta = FOLDER_TEMP_POSITION.get(folder_name)
        if meta is None:
            stem = csv_path.stem.replace("_amp_tau_eff", "")
            meta = FOLDER_TEMP_POSITION.get(stem)
        if meta is None:
            continue

        frame = pd.read_csv(csv_path)
        required = {"amp", "tau_r_eff", "tau_d_eff"}
        if not required.issubset(frame.columns):
            continue

        frame["x_mm"] = float(meta["x_mm"])
        frame["z_mm"] = float(meta["z_mm"])
        frame["temp"] = meta["temp"]

        amp = pd.to_numeric(frame["amp"], errors="coerce")
        tau_r = pd.to_numeric(frame["tau_r_eff"], errors="coerce")
        tau_d = pd.to_numeric(frame["tau_d_eff"], errors="coerce")
        finite = np.isfinite(amp) & np.isfinite(tau_r) & np.isfinite(tau_d)
        selected = finite & ((amp > AMP_THRESHOLD) | ((tau_r > TAU_R_EFF_THRESHOLD) & (tau_d > TAU_D_EFF_THRESHOLD)))
        frame = frame.loc[selected].copy()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        print(f"No selected position-matched CSV files were found for summary PDF generation; skipped: {output_pdf}")
        return

    combined = pd.concat(frames, ignore_index=True)
    with PdfPages(output_pdf) as pdf:
        for metric, title in [
            ("amp", "Histogram: amp by x/z position (selected only)"),
            ("tau_r_eff", "Histogram: tau_r_eff by x/z position (selected only)"),
            ("tau_d_eff", "Histogram: tau_d_eff by x/z position (selected only)"),
        ]:
            fig = make_position_histogram_page(combined, metric, title)
            pdf.savefig(fig)
            plt.close(fig)

        for metric, title in [
            ("tau_r_eff", "Amplitude vs tau_r_eff correlation by x/z position (selected only, color = temp)"),
            ("tau_d_eff", "Amplitude vs tau_d_eff correlation by x/z position (selected only, color = temp)"),
        ]:
            fig = make_position_correlation_page(combined, metric, title)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved summary PDF: {output_pdf}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze temp-laser IQ waveforms.")
    parser.add_argument("input", nargs="?", help="Root folder containing the target data_0825_* folders, or a single .npz file.")
    parser.add_argument("--batch", action="store_true", help="Process all matching data_0825_* folders under the current directory.")
    parser.add_argument("--summary-name", default="temp_laser_summary.pdf", help="Name of the final x/z-position summary PDF.")
    return parser.parse_args()


def find_npz_files(input_path: str | None, batch: bool) -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    if input_path:
        candidate = Path(input_path).resolve()
        if candidate.is_dir():
            return find_npz_files_for_target_root(candidate)
        if candidate.is_file() and candidate.suffix.lower() == ".npz":
            return [candidate]
        return []

    if batch:
        return find_npz_files_for_target_root(script_dir)

    return find_npz_files_for_target_root(script_dir)


def main():
    args = parse_args()
    npz_paths = find_npz_files(args.input, args.batch)
    if not npz_paths:
        raise SystemExit("No matching data_0825_* folders or .npz files were found under the target path.")

    csv_paths = []
    for npz_path in npz_paths:
        csv_path = process_single_npz(npz_path)
        if csv_path is not None and csv_path.exists():
            csv_paths.append(csv_path)

    build_summary_pdf(csv_paths, Path(__file__).resolve().parent / args.summary_name)


if __name__ == "__main__":
    main()
