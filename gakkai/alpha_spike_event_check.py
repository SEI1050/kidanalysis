"""Separate prompt-spike candidates and inspect their raw event waveforms.

The spike score is

    max(signal between -10 and 30 ns) / median(signal between 50 and 150 ns)

where signal is the pedestal-subtracted IQ magnitude.  Events above an
automatic robust upper threshold are labelled ``spike`` rather than rejected.
The complete score table and an event-by-event PDF are written to disk.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kidanalysis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from alpha_common_tau_fit import extract_arrays


def event_signals(ch0, ch1, event, pre_end):
    raw_i = np.asarray(ch0[event], dtype=np.float32)
    raw_q = np.asarray(ch1[event], dtype=np.float32)
    sig_i = raw_i - np.mean(raw_i[:pre_end])
    sig_q = raw_q - np.mean(raw_q[:pre_end])
    magnitude = np.hypot(sig_i, sig_q)
    noise_floor = np.median(magnitude[:pre_end])
    signal = magnitude - noise_floor
    return raw_i, raw_q, sig_i, sig_q, signal, noise_floor


def calculate_scores(ch0, ch1, candidates, time_ns, pre_end):
    prompt = (time_ns >= -10.0) & (time_ns <= 30.0)
    shoulder = (time_ns >= 50.0) & (time_ns <= 150.0)
    rows = []
    for count, row in enumerate(candidates.itertuples(index=False), 1):
        event = int(row.event)
        _, _, _, _, signal, noise_floor = event_signals(ch0, ch1, event, pre_end)
        prompt_max = float(np.max(signal[prompt]))
        shoulder_median = float(np.median(signal[shoulder]))
        # A noise-scale floor prevents a nearly zero denominator from producing
        # an arbitrarily large score while retaining that diagnostic value.
        denominator = max(shoulder_median, 3.0 * noise_floor, 1e-12)
        rows.append({
            "event": event,
            "amp": float(row.amp),
            "tau_d_eff": float(row.tau_d_eff),
            "t_peak": float(row.t_peak),
            "prompt_max": prompt_max,
            "shoulder_median": shoulder_median,
            "pretrigger_noise_floor": float(noise_floor),
            "spike_ratio": prompt_max / denominator,
        })
        if count % 5000 == 0:
            print(f"scored {count} / {len(candidates)} events", flush=True)
    return pd.DataFrame(rows)


def automatic_threshold(scores):
    positive = scores[np.isfinite(scores) & (scores > 0)]
    log_score = np.log(positive)
    center = np.median(log_score)
    robust_sigma = 1.4826 * np.median(np.abs(log_score - center))
    robust_limit = np.exp(center + 5.0 * robust_sigma)
    # Always retain at least the upper 0.5% as a diagnostic group, but do not
    # allow a broad tail to label more than the robust outliers automatically.
    return float(min(robust_limit, np.quantile(positive, 0.995)))


def plot_outputs(pdf_path, table, ch0, ch1, time_ns, pre_end, threshold,
                 max_spike_events, max_control_events, seed):
    rng = np.random.default_rng(seed)
    spike = table[table["class"] == "spike"].sort_values("spike_ratio", ascending=False)
    spike_show = spike.head(max_spike_events)

    # Controls are drawn from the non-spike population in the same amplitude
    # range as the displayed spike events.
    normal = table[table["class"] == "non_spike"]
    if not spike_show.empty:
        lo, hi = spike_show["amp"].min(), spike_show["amp"].max()
        matched = normal[normal["amp"].between(lo, hi)]
        if len(matched) < max_control_events:
            matched = normal
    else:
        matched = normal
    control_ids = rng.choice(matched.index, min(max_control_events, len(matched)), replace=False)
    control_show = matched.loc[control_ids]

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        finite = np.isfinite(table["spike_ratio"]) & (table["spike_ratio"] > 0)
        axes[0].hist(table.loc[finite, "spike_ratio"], bins=np.logspace(-2, 2.5, 140),
                     color="0.35")
        axes[0].axvline(threshold, color="C3", lw=2, label=f"threshold={threshold:.3g}")
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel("prompt spike ratio"); axes[0].set_ylabel("events")
        axes[0].legend(); axes[0].grid(alpha=0.2)
        axes[1].scatter(table["amp"], table["spike_ratio"], s=2, alpha=0.15,
                        c=np.where(table["class"].eq("spike"), "C3", "0.3"))
        axes[1].axhline(threshold, color="C3", lw=1.5)
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("amp"); axes[1].set_ylabel("prompt spike ratio")
        axes[1].grid(alpha=0.2)
        fig.suptitle(f"Spike candidates: {len(spike)} / {len(table)} signal candidates")
        pdf.savefig(fig); plt.close(fig)

        selected = [("SPIKE", row) for row in spike_show.itertuples(index=False)]
        selected += [("CONTROL", row) for row in control_show.itertuples(index=False)]
        for label, row in selected:
            event = int(row.event)
            raw_i, raw_q, sig_i, sig_q, signal, _ = event_signals(ch0, ch1, event, pre_end)
            integral = np.concatenate(([0.0], np.cumsum(
                0.5 * (signal[:-1] + signal[1:]) * np.diff(time_ns))))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            axes[0, 0].plot(time_ns, raw_i, lw=0.8, label="raw I")
            axes[0, 0].plot(time_ns, raw_q, lw=0.8, label="raw Q")
            axes[0, 0].set_title("Raw digitizer channels"); axes[0, 0].legend()
            axes[0, 1].plot(time_ns, sig_i, lw=0.8, label="I - pedestal")
            axes[0, 1].plot(time_ns, sig_q, lw=0.8, label="Q - pedestal")
            axes[0, 1].set_title("Pedestal-subtracted I/Q"); axes[0, 1].legend()
            axes[1, 0].plot(time_ns, signal, color="black", lw=0.8)
            axes[1, 0].axvspan(-10, 30, color="C3", alpha=0.12, label="prompt")
            axes[1, 0].axvspan(50, 150, color="C0", alpha=0.10, label="shoulder")
            axes[1, 0].set_title("|IQ| - pretrigger noise floor"); axes[1, 0].legend()
            axes[1, 1].plot(time_ns, integral, color="C2", lw=0.9)
            axes[1, 1].set_title("Cumulative signed integral")
            for axis in axes.ravel():
                axis.set_xlim(-300, 1600); axis.set_xlabel("time [ns]"); axis.grid(alpha=0.2)
            fig.suptitle(
                f"{label}  event {event}   ratio={row.spike_ratio:.3g}   "
                f"amp={row.amp:.4g}   tau_d={row.tau_d_eff:.1f} ns"
            )
            pdf.savefig(fig); plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=Path("alphaDC_spike_event_check"))
    parser.add_argument("--min-amp", type=float, default=0.002)
    parser.add_argument("--peak-min-ns", type=float, default=-30.0)
    parser.add_argument("--peak-max-ns", type=float, default=500.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-spike-events", type=int, default=40)
    parser.add_argument("--max-control-events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv or args.npz.with_name(args.npz.stem + "_amp_tau_eff.csv")
    frame = pd.read_csv(csv_path)
    candidates = frame[
        np.isfinite(frame["amp"]) & np.isfinite(frame["tau_d_eff"])
        & np.isfinite(frame["t_peak"]) & (frame["amp"] >= args.min_amp)
        & frame["t_peak"].between(args.peak_min_ns, args.peak_max_ns)
        & (frame["tau_d_eff"] > 0)
    ].copy()
    temporary = args.cache_dir is None
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="alpha_spike_check_"))
    try:
        ch0, ch1, npts, sample_rate, ref_position = extract_arrays(args.npz, cache_dir)
        time_ns = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
        pre_end = max(2, int((ref_position - 10.0) / 100.0 * npts))
        table = calculate_scores(ch0, ch1, candidates, time_ns, pre_end)
        threshold = args.threshold if args.threshold is not None else automatic_threshold(
            table["spike_ratio"].to_numpy())
        table["class"] = np.where(table["spike_ratio"] >= threshold, "spike", "non_spike")
        table = table.sort_values("spike_ratio", ascending=False)
        table.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
        plot_outputs(args.output_prefix.with_suffix(".pdf"), table, ch0, ch1, time_ns,
                     pre_end, threshold, args.max_spike_events,
                     args.max_control_events, args.seed)
        print(f"threshold={threshold:.6g}")
        print(table["class"].value_counts().to_string())
        print(f"saved {args.output_prefix.with_suffix('.csv')}")
        print(f"saved {args.output_prefix.with_suffix('.pdf')}")
    finally:
        if temporary and not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
