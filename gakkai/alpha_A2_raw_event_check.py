"""Inspect raw waveforms from the A2_D0 and A2_D1 alpha-event groups."""

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

from alpha_common_tau_fit import extract_arrays, select_groups
from alpha_spike_event_check import event_signals


AMP_THRESHOLD = 0.002
TAU_R_EFF_THRESHOLD = 20.0
TAU_D_EFF_THRESHOLD = 20.0


def add_prompt_ratio(frame, ch0, ch1, time_ns, pre_end):
    prompt = (time_ns >= -10.0) & (time_ns <= 30.0)
    shoulder = (time_ns >= 50.0) & (time_ns <= 150.0)
    ratios = []
    for row in frame.itertuples():
        _, _, _, _, signal, noise_floor = event_signals(ch0, ch1, int(row.event), pre_end)
        denominator = max(float(np.median(signal[shoulder])), 3.0 * noise_floor, 1e-12)
        ratios.append(float(np.max(signal[prompt])) / denominator)
    result = frame.copy()
    result["spike_ratio"] = ratios
    return result


def stratified_examples(group, n_events, seed):
    """Sample evenly across prompt-ratio quantiles instead of only extremes."""
    rng = np.random.default_rng(seed)
    ordered = group.sort_values("spike_ratio")
    chunks = [ordered.loc[index]
              for index in np.array_split(ordered.index.to_numpy(), min(5, len(ordered)))]
    chosen = []
    per_chunk = max(1, int(np.ceil(n_events / len(chunks))))
    for chunk in chunks:
        take = min(per_chunk, len(chunk))
        chosen.extend(rng.choice(chunk.index.to_numpy(), take, replace=False))
    return ordered.loc[chosen[:n_events]].sort_values("spike_ratio")


def make_pdf(path, groups, ch0, ch1, time_ns, pre_end, n_events, seed):
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for name, group in groups.items():
            axes[0].hist(group["spike_ratio"], bins=80, histtype="step", lw=1.4,
                         label=f"{name} (n={len(group)})")
            axes[1].scatter(group["amp"], group["tau_d_eff"], s=4, alpha=0.25,
                            label=name)
        axes[0].set_xlabel("prompt spike ratio"); axes[0].set_ylabel("events")
        axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(alpha=0.2)
        axes[1].set_xlabel("amp"); axes[1].set_ylabel("tau_d_eff [ns]")
        axes[1].legend(); axes[1].grid(alpha=0.2)
        fig.suptitle("A2 groups after amp, tau_r_eff and tau_d_eff noise cuts")
        pdf.savefig(fig); plt.close(fig)

        for group_index, (name, group) in enumerate(groups.items()):
            examples = stratified_examples(group, n_events, seed + group_index)
            for row in examples.itertuples():
                event = int(row.event)
                raw_i, raw_q, sig_i, sig_q, signal, _ = event_signals(
                    ch0, ch1, event, pre_end)
                integral = np.concatenate(([0.0], np.cumsum(
                    0.5 * (signal[:-1] + signal[1:]) * np.diff(time_ns))))
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
                axes[0, 0].plot(time_ns, raw_i, lw=0.75, label="raw I")
                axes[0, 0].plot(time_ns, raw_q, lw=0.75, label="raw Q")
                axes[0, 0].set_title("Raw digitizer channels"); axes[0, 0].legend()
                axes[0, 1].plot(time_ns, sig_i, lw=0.75, label="I - pedestal")
                axes[0, 1].plot(time_ns, sig_q, lw=0.75, label="Q - pedestal")
                axes[0, 1].set_title("Pedestal-subtracted I/Q"); axes[0, 1].legend()
                axes[1, 0].plot(time_ns, signal, color="black", lw=0.8)
                axes[1, 0].axvspan(-10, 30, color="C3", alpha=0.12)
                axes[1, 0].axvspan(50, 150, color="C0", alpha=0.10)
                axes[1, 0].set_title("|IQ| - pretrigger noise floor")
                axes[1, 1].plot(time_ns, integral, color="C2", lw=0.8)
                axes[1, 1].set_title("Cumulative signed integral")
                for axis in axes.ravel():
                    axis.set_xlim(-300, 1600); axis.set_xlabel("time [ns]")
                    axis.grid(alpha=0.2)
                fig.suptitle(
                    f"{name}  event {event}  amp={row.amp:.4g}  "
                    f"tau_r={row.tau_r_eff:.1f} ns  tau_d={row.tau_d_eff:.1f} ns  "
                    f"R_spike={row.spike_ratio:.3g}"
                )
                pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=Path("alphaDC_A2_D0_D1_raw_events"))
    parser.add_argument("--events-per-group", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv or args.npz.with_name(args.npz.stem + "_amp_tau_eff.csv")
    frame = pd.read_csv(csv_path)
    # Reproduce the original A/D group definitions first, then apply the user's
    # established noise rejection so A2_D0/A2_D1 retain their earlier meaning.
    grouped = select_groups(frame, 3, 3, AMP_THRESHOLD, -30.0, 500.0)
    selected = grouped[
        (grouped["amp"] > AMP_THRESHOLD)
        & (grouped["tau_r_eff"] > TAU_R_EFF_THRESHOLD)
        & (grouped["tau_d_eff"] > TAU_D_EFF_THRESHOLD)
        & grouped["group"].isin(["A2_D0", "A2_D1"])
    ].copy()

    temporary = args.cache_dir is None
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="alpha_A2_check_"))
    try:
        ch0, ch1, npts, sample_rate, ref_position = extract_arrays(args.npz, cache_dir)
        time_ns = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
        pre_end = max(2, int((ref_position - 10.0) / 100.0 * npts))
        selected = add_prompt_ratio(selected, ch0, ch1, time_ns, pre_end)
        selected.sort_values(["group", "spike_ratio"]).to_csv(
            args.output_prefix.with_suffix(".csv"), index=False)
        groups = {name: group for name, group in selected.groupby("group", sort=True)}
        make_pdf(args.output_prefix.with_suffix(".pdf"), groups, ch0, ch1, time_ns,
                 pre_end, args.events_per_group, args.seed)
        print(selected.groupby("group").agg(
            n=("event", "size"), amp_median=("amp", "median"),
            tau_r_median=("tau_r_eff", "median"),
            tau_d_median=("tau_d_eff", "median"),
            spike_ratio_median=("spike_ratio", "median"),
        ).to_string())
        print(f"saved {args.output_prefix.with_suffix('.csv')}")
        print(f"saved {args.output_prefix.with_suffix('.pdf')}")
    finally:
        if temporary and not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
