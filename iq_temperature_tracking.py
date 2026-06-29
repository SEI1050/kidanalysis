#!/usr/bin/env python3
"""
Track 1 Hz temperature motion from 50 Hz KID laser waveform data.

Measurement folders are expected to look like
`5.476GHz_z=7.5mm_x=9.4mm`.  For each z/x group, the resonance angular
frequency is defined as the median of the measured angular frequencies.
The I/Q waveform response is baseline-subtracted, shifted to the resonance
point, rotated so Q is minimized and I is positive at resonance, and then
plotted both event-by-event and in 1 second bins.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MEAS_DIR_RE = re.compile(
    r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>.+))?$"
)


@dataclass(frozen=True)
class Run:
    path: Path
    freq_ghz: float
    z_mm: float
    x_mm: float
    tag: str
    npz_files: tuple[Path, ...]


def scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.size == 1 else value


def as_2d(array):
    array = np.asarray(array)
    return array[None, :] if array.ndim == 1 else array


def safe_name(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(",", "_")
    )


def parse_window_us(text):
    if text is None or str(text).lower() == "none":
        return None
    pieces = [p.strip() for p in str(text).split(",")]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("window must be 'lo,hi', e.g. '0,1.0'")

    def parse_piece(piece):
        if piece == "" or piece.lower() == "none":
            return None
        return float(piece)

    return parse_piece(pieces[0]), parse_piece(pieces[1])


def mask_from_window_us(time_s, window_us, default_mask):
    if window_us is None:
        return default_mask
    lo_us, hi_us = window_us
    mask = np.ones_like(time_s, dtype=bool)
    if lo_us is not None:
        mask &= time_s >= lo_us * 1e-6
    if hi_us is not None:
        mask &= time_s <= hi_us * 1e-6
    return mask


def make_time_axis(npts, sample_rate, ref_position):
    return (np.arange(npts, dtype=float) - npts * ref_position / 100.0) / sample_rate


def discover_runs(root, pattern, recursive):
    runs = []
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
        runs.append(
            Run(
                path=child,
                freq_ghz=float(groups["freq"]),
                z_mm=float(groups["z"]),
                x_mm=float(groups["x"]),
                tag=groups["tag"] or "",
                npz_files=files,
            )
        )
    return sorted(runs, key=lambda r: (r.z_mm, r.x_mm, r.tag, r.freq_ghz, r.path.name))


def load_run_features(run, baseline_window_us, signal_window_us, signal_stat, max_events):
    rows = []
    event_offset = 0
    for npz_path in run.npz_files:
        data = np.load(npz_path, allow_pickle=True)
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

        if max_events is not None:
            remaining = max_events - event_offset
            if remaining <= 0:
                break
            ch0 = ch0[:remaining]
            ch1 = ch1[:remaining]

        sample_rate = float(scalar(data["sample_rate"]))
        ref_position = float(scalar(data["ref_position"]))
        time_s = make_time_axis(npts, sample_rate, ref_position)
        baseline_mask = mask_from_window_us(time_s, baseline_window_us, time_s < 0)
        signal_mask = mask_from_window_us(time_s, signal_window_us, time_s >= 0)
        if baseline_mask.sum() < 2 or signal_mask.sum() < 2:
            raise ValueError(f"too few points in baseline or signal window: {npz_path}")

        base_i = ch0[:, baseline_mask].mean(axis=1)
        base_q = ch1[:, baseline_mask].mean(axis=1)
        di = ch0[:, signal_mask] - base_i[:, None]
        dq = ch1[:, signal_mask] - base_q[:, None]

        if signal_stat == "peak":
            peak_idx = np.argmax(np.hypot(di, dq), axis=1)
            event_i = di[np.arange(di.shape[0]), peak_idx]
            event_q = dq[np.arange(dq.shape[0]), peak_idx]
        else:
            event_i = di.mean(axis=1)
            event_q = dq.mean(axis=1)

        if "deltat" in data.files:
            deltat = data["deltat"][: len(event_i)]
            elapsed_s = np.array([dt.total_seconds() for dt in deltat], dtype=float)
        else:
            daq_rate = float(scalar(data["daq_rate"])) if "daq_rate" in data.files else 50.0
            elapsed_s = np.arange(len(event_i), dtype=float) / daq_rate

        for local_idx, (elapsed, i_value, q_value) in enumerate(zip(elapsed_s, event_i, event_q)):
            rows.append(
                {
                    "run_dir": run.path.name,
                    "file": npz_path.name,
                    "event": event_offset + local_idx,
                    "elapsed_s": elapsed,
                    "second": int(math.floor(elapsed)),
                    "freq_ghz": run.freq_ghz,
                    "omega_rad_s": 2.0 * math.pi * run.freq_ghz * 1e9,
                    "z_mm": run.z_mm,
                    "x_mm": run.x_mm,
                    "tag": run.tag,
                    "i_raw": float(i_value),
                    "q_raw": float(q_value),
                }
            )
        event_offset += len(event_i)
    return pd.DataFrame(rows)


def group_key(run, include_tag):
    return (run.z_mm, run.x_mm, run.tag) if include_tag else (run.z_mm, run.x_mm)


def normalize_group(df, resonance_omega):
    freqs = np.sort(df["freq_ghz"].unique())
    resonance_freq = resonance_omega / (2.0 * math.pi * 1e9)
    resonance_freq = float(freqs[np.argmin(np.abs(freqs - resonance_freq))])

    med = df.groupby("freq_ghz", as_index=False)[["i_raw", "q_raw"]].median()
    ref = med.iloc[np.argmin(np.abs(med["freq_ghz"].to_numpy() - resonance_freq))]
    ref_complex = complex(float(ref["i_raw"]), float(ref["q_raw"]))
    centered = df["i_raw"].to_numpy() + 1j * df["q_raw"].to_numpy() - ref_complex

    on_res = centered[df["freq_ghz"].to_numpy() == resonance_freq]
    q_axis = np.nanmedian(on_res) if len(on_res) else np.nanmedian(centered)
    if not np.isfinite(q_axis.real) or abs(q_axis) < 1e-15:
        q_axis = np.nanmedian(centered) if abs(np.nanmedian(centered)) > 0 else 1.0 + 0.0j

    rotated = centered * np.exp(-1j * np.angle(q_axis))
    rotated -= 1j * np.nanmedian(rotated[df["freq_ghz"].to_numpy() == resonance_freq].imag)
    if np.nanmedian(rotated.real) < 0:
        rotated *= -1.0

    scale = np.nanpercentile(np.abs(rotated), 90)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    out = df.copy()
    out["i_norm"] = rotated.real / scale
    out["q_norm"] = rotated.imag / scale
    out["resonance_freq_ghz"] = resonance_freq
    out["resonance_omega_rad_s"] = 2.0 * math.pi * resonance_freq * 1e9
    meta = {
        "resonance_freq_ghz": resonance_freq,
        "resonance_omega_rad_s": 2.0 * math.pi * resonance_freq * 1e9,
        "ref_i_raw": ref_complex.real,
        "ref_q_raw": ref_complex.imag,
        "scale": float(scale),
    }
    return out, meta


def plot_group(df, per_second, title, out_png):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    sc0 = ax[0].scatter(
        df["i_norm"],
        df["q_norm"],
        c=df["elapsed_s"],
        s=7,
        alpha=0.35,
        cmap="viridis",
        linewidths=0,
    )
    ax[0].axhline(0, color="0.55", lw=0.8)
    ax[0].axvline(0, color="0.55", lw=0.8)
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set_xlabel("normalized I")
    ax[0].set_ylabel("normalized Q")
    ax[0].set_title("50 Hz events")
    fig.colorbar(sc0, ax=ax[0], label="elapsed time [s]")

    sc1 = ax[1].scatter(
        per_second["i_norm"],
        per_second["q_norm"],
        c=per_second["second"],
        s=42,
        cmap="plasma",
        edgecolors="black",
        linewidths=0.35,
    )
    ax[1].plot(per_second["i_norm"], per_second["q_norm"], color="0.25", lw=0.8, alpha=0.65)
    for _, row in per_second.iterrows():
        ax[1].annotate(
            str(int(row["second"])),
            (row["i_norm"], row["q_norm"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax[1].axhline(0, color="0.55", lw=0.8)
    ax[1].axvline(0, color="0.55", lw=0.8)
    ax[1].set_aspect("equal", adjustable="datalim")
    ax[1].set_xlabel("normalized I")
    ax[1].set_ylabel("normalized Q")
    ax[1].set_title("1 Hz bins")
    fig.colorbar(sc1, ax=ax[1], label="second")
    fig.suptitle(title)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def analyze(args):
    root = args.root.expanduser().resolve(strict=False)
    out_dir = args.output_dir.expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root, args.pattern, args.recursive)
    if not runs:
        raise SystemExit(f"no measurement folders found under {root}")

    groups = {}
    for run in runs:
        groups.setdefault(group_key(run, args.include_tag), []).append(run)

    all_rows = []
    summary_rows = []
    print(f"input: {root}")
    print(f"runs: {len(runs)}")
    print(f"groups: {len(groups)}")
    print(f"output: {out_dir}")

    for _, group_runs in sorted(groups.items()):
        freqs = np.array([r.freq_ghz for r in group_runs], dtype=float)
        unique_freqs = np.unique(freqs)
        if len(unique_freqs) < args.min_freqs:
            continue

        resonance_omega = float(np.median(2.0 * math.pi * freqs * 1e9))
        frames = [
            load_run_features(
                run,
                args.baseline_window_us,
                args.signal_window_us,
                args.signal_stat,
                args.max_events,
            )
            for run in group_runs
        ]
        group_df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
        if group_df.empty:
            continue

        normalized, meta = normalize_group(group_df, resonance_omega)
        per_second = (
            normalized.groupby("second", as_index=False)
            .agg(
                i_norm=("i_norm", "mean"),
                q_norm=("q_norm", "mean"),
                i_raw=("i_raw", "mean"),
                q_raw=("q_raw", "mean"),
                n_events=("event", "size"),
                elapsed_s=("elapsed_s", "mean"),
                z_mm=("z_mm", "first"),
                x_mm=("x_mm", "first"),
                tag=("tag", "first"),
                resonance_freq_ghz=("resonance_freq_ghz", "first"),
                resonance_omega_rad_s=("resonance_omega_rad_s", "first"),
            )
            .sort_values("second")
        )

        label = f"z={group_runs[0].z_mm:g}mm_x={group_runs[0].x_mm:g}mm"
        if args.include_tag and group_runs[0].tag:
            label += f"_{group_runs[0].tag}"
        stem = safe_name(label)
        normalized.to_csv(out_dir / f"{stem}_events.csv", index=False)
        per_second.to_csv(out_dir / f"{stem}_1Hz_bins.csv", index=False)
        plot_group(
            normalized,
            per_second,
            f"{label}, fres={meta['resonance_freq_ghz']:.6f} GHz, stat={args.signal_stat}",
            out_dir / f"{stem}_iq.png",
        )

        all_rows.append(normalized)
        summary_rows.append(
            {
                "label": label,
                "z_mm": group_runs[0].z_mm,
                "x_mm": group_runs[0].x_mm,
                "tag": group_runs[0].tag,
                "n_runs": len(group_runs),
                "n_unique_freqs": len(unique_freqs),
                "n_events": len(normalized),
                **meta,
            }
        )
        print(f"saved {label}: fres={meta['resonance_freq_ghz']:.6f} GHz, events={len(normalized)}")

    if not summary_rows:
        raise SystemExit("no groups passed the min frequency requirement")
    pd.DataFrame(summary_rows).to_csv(out_dir / "iq_temperature_tracking_summary.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(out_dir / "iq_temperature_tracking_events_all.csv", index=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260527")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="wf_*.npz")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--include-tag", action="store_true")
    parser.add_argument("--min-freqs", type=int, default=3)
    parser.add_argument("--baseline-window-us", type=parse_window_us, default=None)
    parser.add_argument("--signal-window-us", type=parse_window_us, default="0,1.0")
    parser.add_argument("--signal-stat", choices=["peak", "mean"], default="peak")
    parser.add_argument("--max-events", type=int, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.root is None:
        args.root = Path("/Volumes/NO NAME/data") / args.date
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "data" / args.date / "iq_temperature_tracking"
    analyze(args)


if __name__ == "__main__":
    main()
