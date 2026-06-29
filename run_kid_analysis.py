#!/usr/bin/env python3
"""
One-command KID waveform summary analysis.

This script discovers measurement folders such as
`5.476GHz_z=7.5mm_x=3.4mm`, extracts simple pulse metrics from wf_*.npz
files, and writes summary CSV files and scan plots.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tempfile
import os

# ハードディスク上の任意のパスを dir で指定する
target_dir = "/Volumes/NO NAME/Temp" # Macの例

with tempfile.TemporaryDirectory(dir=target_dir) as tmpdir:
    print(f"一時ディレクトリ: {tmpdir}")
    # この中で処理を行う（withブロックを抜けると自動削除されます）


MEAS_DIR_RE = re.compile(
    r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>.+))?$"
)


@dataclass(frozen=True)
class Measurement:
    path: Path
    freq_ghz: float
    z_mm: float
    x_mm: float
    tag: str
    npz_files: tuple[Path, ...]


def scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.size == 1 else value


def parse_window_us(text: str | None) -> tuple[float | None, float | None] | None:
    if text is None or text.lower() == "none":
        return None

    pieces = [p.strip() for p in text.split(",")]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("window must be 'lo,hi', for example '0,1.5'")

    def parse_piece(piece: str) -> float | None:
        if piece == "" or piece.lower() == "none":
            return None
        return float(piece)

    return parse_piece(pieces[0]), parse_piece(pieces[1])


def mask_from_window(time_s: np.ndarray, window_us, default: np.ndarray) -> np.ndarray:
    if window_us is None:
        return default

    lo_us, hi_us = window_us
    mask = np.ones_like(time_s, dtype=bool)

    if lo_us is not None:
        mask &= time_s >= lo_us * 1e-6
    if hi_us is not None:
        mask &= time_s <= hi_us * 1e-6

    return mask


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1))


def make_time_axis(npts: int, sample_rate_hz: float, ref_position_percent: float) -> np.ndarray:
    return (np.arange(npts, dtype=float) - npts * ref_position_percent / 100.0) / sample_rate_hz


def as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[None, :]
    return array


def default_input_roots(date: str) -> list[Path]:
    home = Path.home()
    return [
        home / "Library" / "CloudStorage" / "OneDrive-TheUniversityofTokyo" / "東京大学" / "4S" / "kidfit" / date,
        home / "OneDrive - The University of Tokyo" / "東京大学" / "4S" / "kidfit" / date,
        Path(__file__).resolve().parent / "data" / date,
    ]


def discover_measurements(roots: list[Path], pattern: str, recursive: bool) -> list[Measurement]:
    measurements: list[Measurement] = []

    for root in roots:
        if not root.is_dir():
            continue

        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue

            match = MEAS_DIR_RE.match(child.name)
            if match is None:
                continue

            files = tuple(sorted(child.rglob(pattern) if recursive else child.glob(pattern)))
            if not files:
                continue

            groups = match.groupdict()
            measurements.append(
                Measurement(
                    path=child,
                    freq_ghz=float(groups["freq"]),
                    z_mm=float(groups["z"]),
                    x_mm=float(groups["x"]),
                    tag=groups["tag"] or "",
                    npz_files=files,
                )
            )

    return sorted(measurements, key=lambda m: (m.freq_ghz, m.z_mm, m.x_mm, m.tag, m.path.name))


def load_measurement(measurement: Measurement, max_events: int | None):
    all_ch0 = []
    all_ch1 = []
    time_ref = None
    daq_rates = []

    remaining = max_events

    for npz_path in measurement.npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as exc:
            print(f"skip load error: {npz_path}: {exc}")
            continue

        required = {"ch0", "ch1", "npts", "sample_rate", "ref_position"}
        if not required.issubset(data.files):
            print(f"skip missing keys: {npz_path}")
            continue

        ch0 = as_2d(data["ch0"])
        ch1 = as_2d(data["ch1"])
        if ch0.shape != ch1.shape:
            print(f"skip shape mismatch: {npz_path}")
            continue

        npts = int(scalar(data["npts"]))
        if ch0.shape[1] != npts:
            print(f"skip npts mismatch: {npz_path}")
            continue

        sample_rate = float(scalar(data["sample_rate"]))
        ref_position = float(scalar(data["ref_position"]))
        time = make_time_axis(npts, sample_rate, ref_position)

        if time_ref is None:
            time_ref = time
        elif len(time) != len(time_ref) or not np.allclose(time, time_ref):
            print(f"skip time axis mismatch: {npz_path}")
            continue

        if "daq_rate" in data.files:
            daq_rates.append(float(scalar(data["daq_rate"])))

        if remaining is not None:
            if remaining <= 0:
                break
            take = min(remaining, ch0.shape[0])
            ch0 = ch0[:take]
            ch1 = ch1[:take]
            remaining -= take

        all_ch0.append(ch0)
        all_ch1.append(ch1)

    if not all_ch0 or time_ref is None:
        return None

    return time_ref, np.vstack(all_ch0), np.vstack(all_ch1), daq_rates


def projected_signal(time_s, ch0, ch1, baseline_mask, amp_mask):
    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    dch0 = ch0 - ped0[:, None]
    dch1 = ch1 - ped1[:, None]

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)
    radius = np.hypot(mean0, mean1)
    amp_indices = np.where(amp_mask)[0]
    peak_idx = amp_indices[np.argmax(radius[amp_mask])]

    direction = np.array([mean0[peak_idx], mean1[peak_idx]], dtype=float)
    norm = np.hypot(direction[0], direction[1])
    if norm == 0:
        direction[:] = [1.0, 0.0]
    else:
        direction /= norm

    signal = dch0 * direction[0] + dch1 * direction[1]
    return signal, direction, peak_idx, ped0, ped1


def analyze_measurement(measurement, args):
    loaded = load_measurement(measurement, args.max_events)
    if loaded is None:
        return None, None

    time_s, ch0, ch1, daq_rates = loaded
    baseline_mask = mask_from_window(time_s, args.baseline_us, time_s < 0)
    amp_mask = mask_from_window(time_s, args.amp_us, time_s >= 0)
    integral_mask = mask_from_window(time_s, args.integral_us, time_s >= 0)

    if baseline_mask.sum() < 2 or amp_mask.sum() < 2 or integral_mask.sum() < 2:
        print(f"skip insufficient time window: {measurement.path.name}")
        return None, None

    signal, direction, peak_idx, ped0, ped1 = projected_signal(time_s, ch0, ch1, baseline_mask, amp_mask)

    peak = signal[:, amp_mask].max(axis=1)
    peak_to_peak = np.ptp(signal[:, amp_mask], axis=1)
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(np.clip(signal[:, integral_mask], 0, None), time_s[integral_mask], axis=1)
    else:
        integral = np.trapz(np.clip(signal[:, integral_mask], 0, None), time_s[integral_mask], axis=1)
    tau_eff = np.divide(integral, peak, out=np.full_like(integral, np.nan), where=np.abs(peak) > args.min_abs_peak)

    peak_mean, peak_std = mean_std(peak)
    p2p_mean, p2p_std = mean_std(peak_to_peak)
    integral_mean, integral_std = mean_std(integral)
    tau_mean, tau_std = mean_std(tau_eff)
    ped0_mean, ped0_std = mean_std(ped0)
    ped1_mean, ped1_std = mean_std(ped1)

    summary = {
        "dir_name": measurement.path.name,
        "path": str(measurement.path),
        "freq_ghz": measurement.freq_ghz,
        "z_mm": measurement.z_mm,
        "x_mm": measurement.x_mm,
        "tag": measurement.tag,
        "n_files": len(measurement.npz_files),
        "n_events": ch0.shape[0],
        "daq_rate_hz_mean": float(np.mean(daq_rates)) if daq_rates else np.nan,
        "projection_ch0": direction[0],
        "projection_ch1": direction[1],
        "peak_time_us": time_s[peak_idx] * 1e6,
        "peak_mean": peak_mean,
        "peak_std": peak_std,
        "peak_sem": sem(peak),
        "peak_to_peak_mean": p2p_mean,
        "peak_to_peak_std": p2p_std,
        "integral_positive_mean": integral_mean,
        "integral_positive_std": integral_std,
        "tau_eff_s_mean": tau_mean,
        "tau_eff_s_std": tau_std,
        "tau_eff_s_sem": sem(tau_eff),
        "tau_eff_us_mean": tau_mean * 1e6 if np.isfinite(tau_mean) else np.nan,
        "tau_eff_us_sem": sem(tau_eff) * 1e6 if np.isfinite(sem(tau_eff)) else np.nan,
        "pedestal_ch0_mean": ped0_mean,
        "pedestal_ch0_std": ped0_std,
        "pedestal_ch1_mean": ped1_mean,
        "pedestal_ch1_std": ped1_std,
    }

    event_rows = pd.DataFrame(
        {
            "dir_name": measurement.path.name,
            "event_index": np.arange(ch0.shape[0]),
            "peak": peak,
            "peak_to_peak": peak_to_peak,
            "integral_positive": integral,
            "tau_eff_s": tau_eff,
            "pedestal_ch0": ped0,
            "pedestal_ch1": ped1,
        }
    )

    return summary, event_rows


def save_plots(summary_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if summary_df.empty:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skip PNG plots")
        return

    for value, ylabel, stem in [
        ("peak_mean", "projected peak", "peak_vs_position"),
        ("integral_positive_mean", "positive integral [signal s]", "integral_vs_position"),
        ("tau_eff_us_mean", "tau_eff [us]", "tau_eff_vs_position"),
    ]:
        for fixed_col, scan_col in [("z_mm", "x_mm"), ("x_mm", "z_mm")]:
            fixed_values = sorted(summary_df[fixed_col].dropna().unique())
            if not fixed_values:
                continue
            fixed = min(fixed_values, key=lambda x: abs(x - 7.5 if fixed_col == "z_mm" else x - 3.4))
            view = summary_df[np.isclose(summary_df[fixed_col], fixed)].copy()
            if view.empty:
                continue

            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            for freq, part in view.groupby("freq_ghz"):
                part = part.sort_values(scan_col)
                yerr_col = value.replace("_mean", "_sem")
                yerr = part[yerr_col] if yerr_col in part.columns else None
                ax.errorbar(part[scan_col], part[value], yerr=yerr, marker="o", capsize=3, label=f"{freq:.3f} GHz")

            ax.set_xlabel(scan_col)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} ({fixed_col}={fixed:g})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}_{scan_col}_fixed_{fixed_col}_{fixed:g}.png", dpi=dpi)
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260527", help="data date, e.g. 20260527")
    parser.add_argument("--input-root", action="append", type=Path, help="input root; can be given multiple times")
    parser.add_argument("--output-dir", type=Path, help="output directory")
    parser.add_argument("--pattern", default="wf_*.npz", help="npz filename pattern")
    parser.add_argument("--recursive", action="store_true", help="search wf files recursively inside each measurement")
    parser.add_argument("--max-events", type=int, default=None, help="limit events per measurement for quick checks")
    parser.add_argument("--freq", type=float, action="append", help="only analyze selected frequency in GHz")
    parser.add_argument("--z", type=float, action="append", help="only analyze selected z position in mm")
    parser.add_argument("--x", type=float, action="append", help="only analyze selected x position in mm")
    parser.add_argument("--baseline-us", type=parse_window_us, default=None, help="baseline window 'lo,hi' in us; default t<0")
    parser.add_argument("--amp-us", type=parse_window_us, default=parse_window_us("0,1.5"), help="peak window in us")
    parser.add_argument("--integral-us", type=parse_window_us, default=parse_window_us("0,1.5"), help="integral window in us")
    parser.add_argument("--min-abs-peak", type=float, default=1e-12, help="minimum |peak| for tau_eff")
    parser.add_argument("--dpi", type=int, default=250, help="plot DPI")
    return parser


def keep_selected(value: float, selected: list[float] | None, atol: float = 1e-6) -> bool:
    if selected is None:
        return True
    return any(np.isclose(value, item, atol=atol, rtol=0) for item in selected)


def main() -> int:
    args = build_arg_parser().parse_args()

    roots = [p.expanduser().resolve(strict=False) for p in args.input_root] if args.input_root else default_input_roots(args.date)
    out_dir = args.output_dir or Path(__file__).resolve().parent / "data" / args.date / "auto_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("input roots:")
    for root in roots:
        print(" ", root, "exists=", root.is_dir())
    print("output:", out_dir)

    measurements = discover_measurements(roots, args.pattern, args.recursive)
    measurements = [
        m
        for m in measurements
        if keep_selected(m.freq_ghz, args.freq, atol=1e-3)
        and keep_selected(m.z_mm, args.z)
        and keep_selected(m.x_mm, args.x)
    ]

    if not measurements:
        raise SystemExit("no measurement folders found")

    summaries = []
    event_frames = []
    for index, measurement in enumerate(measurements, start=1):
        print(f"[{index}/{len(measurements)}] {measurement.path.name}")
        summary, event_rows = analyze_measurement(measurement, args)
        if summary is None:
            continue
        summaries.append(summary)
        event_frames.append(event_rows)

    if not summaries:
        raise SystemExit("no usable measurements")

    summary_df = pd.DataFrame(summaries).sort_values(["freq_ghz", "z_mm", "x_mm", "tag", "dir_name"])
    summary_path = out_dir / "kid_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if event_frames:
        event_path = out_dir / "kid_event_metrics.csv"
        pd.concat(event_frames, ignore_index=True).to_csv(event_path, index=False)

    save_plots(summary_df, out_dir, args.dpi)

    print("wrote:", summary_path)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
