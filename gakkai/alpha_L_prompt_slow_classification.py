"""Classify alpha events by agreement with the laser-L prompt+slow shape.

The prompt/rise/slow time constants are read from the event-wise laser-L fit
and held fixed.  Only one common alpha trigger offset is estimated; the two
non-negative component amplitudes are then fitted independently to every
pedestal-subtracted absolute-IQ waveform.  An accepted event is labelled
L-like when its 0--300 ns NRMSE is <= the requested threshold, otherwise it is
labelled C-like.
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
from scipy.optimize import minimize_scalar

from alpha_common_tau_fit import extract_arrays
from laser_eventwise_L_shared_tau_C_test import basis, event_metrics, vector_nnls


AMP_THRESHOLD = 0.002
TAU_R_EFF_THRESHOLD = 20.0
TAU_D_EFF_THRESHOLD = 20.0


def read_laser_constants(path: Path):
    row = pd.read_csv(path, nrows=1).iloc[0]
    return tuple(float(row[name]) for name in (
        "shared_tau_r_ns", "shared_tau_fast_ns", "shared_tau_slow_ns",
        "shared_t0_ns",
    ))


def make_theta(tau_r, tau_fast, tau_slow, t0):
    if not tau_r < tau_fast < tau_slow:
        raise ValueError("Laser time constants must satisfy tau_r < tau_fast < tau_slow")
    return np.array([
        np.log(tau_r), np.log(tau_fast - tau_r),
        np.log(tau_slow - tau_fast), t0,
    ])


def accepted_events(csv_path: Path):
    frame = pd.read_csv(csv_path)
    for name in ("event", "amp", "tau_r_eff", "tau_d_eff", "t10_left"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    finite = np.isfinite(frame["event"]) & np.isfinite(frame["amp"])
    selected = finite & (
        (frame["amp"] > AMP_THRESHOLD)
        | ((frame["tau_r_eff"] > TAU_R_EFF_THRESHOLD)
           & (frame["tau_d_eff"] > TAU_D_EFF_THRESHOLD))
    )
    result = frame.loc[selected].copy()
    result["event"] = result["event"].astype(int)
    return result.sort_values("event").reset_index(drop=True)


def aligned_signal_block(ch0, ch1, event_ids, centers, full_time,
                         relative_time, pre_end):
    i = np.asarray(ch0[event_ids], dtype=np.float32)
    q = np.asarray(ch1[event_ids], dtype=np.float32)
    # The requested baseline definition: event-wise pre-trigger medians.
    i -= np.median(i[:, :pre_end], axis=1)[:, None]
    q -= np.median(q[:, :pre_end], axis=1)[:, None]
    mag = np.hypot(i, q)
    floor = np.median(mag[:, :pre_end], axis=1)
    noise = np.std(mag[:, :pre_end] - floor[:, None], axis=1)
    signal = mag - floor[:, None]
    dt = float(full_time[1] - full_time[0])
    positions = (centers[:, None] + relative_time[None, :] - full_time[0]) / dt
    left = np.floor(positions).astype(np.int64)
    fraction = positions - left
    left = np.clip(left, 0, signal.shape[1] - 2)
    raw = ((1.0 - fraction) * np.take_along_axis(signal, left, axis=1)
           + fraction * np.take_along_axis(signal, left + 1, axis=1))
    peak = np.max(raw, axis=1)
    normalized = raw / np.maximum(peak[:, None], 1e-15)
    snr = peak / np.maximum(noise, 1e-15)
    return normalized.astype(np.float32), peak, noise, snr


def load_selected_waveforms(ch0, ch1, event_ids, centers, full_time,
                            relative_time, pre_end, chunk_size):
    waves, peaks, noises, snrs = [], [], [], []
    for start in range(0, len(event_ids), chunk_size):
        stop = min(start + chunk_size, len(event_ids))
        values = aligned_signal_block(
            ch0, ch1, event_ids[start:stop], centers[start:stop], full_time,
            relative_time, pre_end)
        for store, value in zip((waves, peaks, noises, snrs), values):
            store.append(value)
        if stop % 5000 == 0 or stop == len(event_ids):
            print(f"prepared {stop} / {len(event_ids)} selected events", flush=True)
    return (np.concatenate(waves), np.concatenate(peaks),
            np.concatenate(noises), np.concatenate(snrs))


def estimate_common_t0(time, training, constants, t0_min, t0_max):
    tau_r, tau_fast, tau_slow = constants[:3]

    def objective(t0):
        design = basis(time, make_theta(tau_r, tau_fast, tau_slow, t0))
        amps = vector_nnls(training, design)
        mse = np.mean((training - amps @ design.T) ** 2, axis=1)
        return float(np.mean(np.minimum(mse, np.quantile(mse, 0.95))))

    result = minimize_scalar(objective, bounds=(t0_min, t0_max), method="bounded",
                             options={"xatol": 0.05})
    return float(result.x), float(result.fun)


def random_ids(group, count, rng):
    if group.empty:
        return np.array([], dtype=int)
    return rng.choice(group.index.to_numpy(), min(count, len(group)), replace=False)


def make_pdf(path, results, time, waves, models, threshold, constants,
             random_per_class, plots_per_page, seed, ch0, ch1, full_time,
             pre_end):
    rng = np.random.default_rng(seed)
    classes = ("L-like", "C-like")
    colors = {"L-like": "C0", "C-like": "C3"}
    tau_r, tau_fast, tau_slow, alpha_t0 = constants

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, height_ratios=(0.7, 1.3))
        ax_text = fig.add_subplot(grid[0, :]); ax_text.axis("off")
        counts = results["class"].value_counts()
        text = (
            "Alpha event-wise test against the laser-L prompt+slow model\n\n"
            f"accepted events: {len(results):,}    "
            f"L-like: {counts.get('L-like', 0):,}    "
            f"C-like: {counts.get('C-like', 0):,}\n"
            f"classification: L-like if early NRMSE <= {threshold:.3f}; "
            f"C-like otherwise\n"
            f"fixed laser-L constants: tau_r={tau_r:.2f} ns, "
            f"tau_fast={tau_fast:.2f} ns, tau_slow={tau_slow:.2f} ns\n"
            f"event alignment: t10_left = 0; model onset relative to t10_left: "
            f"{alpha_t0:.2f} ns"
        )
        ax_text.text(0.02, 0.95, text, va="top", fontsize=12)

        metrics = (("early_nrmse_0_300ns", "early: 0-300 ns"),
                   ("nrmse", "whole fit (t10-relative)"),
                   ("late_nrmse_300_1200ns", "late: 300-1200 ns"))
        for column, (metric, title) in enumerate(metrics):
            axis = fig.add_subplot(grid[1, column])
            values = [results.loc[results["class"] == name, metric] for name in classes]
            axis.boxplot(values, tick_labels=classes, showfliers=False,
                         patch_artist=True,
                         boxprops={"facecolor": "white"})
            if metric == "early_nrmse_0_300ns":
                axis.axhline(threshold, color="black", ls="--", lw=1,
                            label=f"cut = {threshold:.3f}")
                axis.legend(fontsize=8)
            axis.set_title(title); axis.set_ylabel("NRMSE / event peak")
            axis.grid(alpha=0.25)
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        upper = min(0.5, float(results["early_nrmse_0_300ns"].quantile(0.995)))
        bins = np.linspace(0, max(upper, threshold * 1.2), 120)
        for name in classes:
            group = results[results["class"] == name]
            axes[0].hist(group["early_nrmse_0_300ns"], bins=bins, histtype="step",
                         lw=1.5, label=f"{name} (n={len(group):,})",
                         color=colors[name])
            axes[1].scatter(group["early_nrmse_0_300ns"], group["late_nrmse_300_1200ns"],
                            s=3, alpha=0.15, color=colors[name], label=name)
        axes[0].axvline(threshold, color="black", ls="--")
        axes[0].set_xlabel("early NRMSE"); axes[0].set_ylabel("events")
        axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(alpha=0.25)
        axes[1].axvline(threshold, color="black", ls="--")
        axes[1].set_xlabel("early NRMSE"); axes[1].set_ylabel("late NRMSE")
        axes[1].legend(); axes[1].grid(alpha=0.25)
        fig.suptitle("Fit residual distributions and the L-like/C-like boundary")
        pdf.savefig(fig); plt.close(fig)

        for name in classes:
            group = results[results["class"] == name]
            chosen = random_ids(group, random_per_class, rng)
            for page_start in range(0, len(chosen), plots_per_page):
                page_ids = chosen[page_start:page_start + plots_per_page]
                ncols = 3; nrows = int(np.ceil(len(page_ids) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows),
                                         constrained_layout=True, squeeze=False)
                for axis, row_id in zip(axes.ravel(), page_ids):
                    row = results.loc[row_id]
                    event = int(row.event)
                    raw_i = np.asarray(ch0[event], dtype=np.float32)
                    raw_q = np.asarray(ch1[event], dtype=np.float32)
                    sig_i = raw_i - np.median(raw_i[:pre_end])
                    sig_q = raw_q - np.median(raw_q[:pre_end])
                    magnitude = np.hypot(sig_i, sig_q)
                    signal = magnitude - np.median(magnitude[:pre_end])
                    plot_wave = signal / max(float(row.raw_peak), 1e-15)
                    event_t0 = float(row.t10_left) + alpha_t0
                    full_design = basis(full_time, make_theta(
                        tau_r, tau_fast, tau_slow, event_t0))
                    plot_model = np.array([
                        row.A_prompt_normalized, row.A_slow_normalized
                    ]) @ full_design.T
                    axis.plot(full_time, plot_wave, color="0.25", lw=0.7,
                              label="full data")
                    axis.plot(full_time, plot_model, color=colors[name], lw=1.3,
                              label="prompt+slow fit")
                    axis.axvspan(float(row.t10_left), float(row.t10_left) + 300.0,
                                 color="C1", alpha=0.08)
                    axis.set_title(f"event {int(row.event)}  early={row.early_nrmse_0_300ns:.3f}",
                                   fontsize=9)
                    axis.set_xlim(full_time[0], full_time[-1])
                    axis.set_ylim(-0.15, 1.15)
                    axis.set_xlabel("time [ns]"); axis.set_ylabel("normalized |IQ|")
                    axis.grid(alpha=0.2)
                for axis in axes.ravel()[len(page_ids):]: axis.axis("off")
                axes.ravel()[0].legend(fontsize=8)
                fig.suptitle(f"Random {name} alpha events: data and fixed-shape fit")
                pdf.savefig(fig); plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path,
                        default=here / "alphaDC_combined.npz")
    parser.add_argument("--feature-csv", type=Path, default=None)
    parser.add_argument("--laser-events-csv", type=Path,
                        default=here / "laser_eventwise_L_expanded_shared_tau_C_test.csv")
    parser.add_argument("--output-prefix", type=Path,
                        default=here / "alpha_L_prompt_slow_classification")
    parser.add_argument("--early-threshold", type=float, default=0.05)
    parser.add_argument("--fit-start-ns", type=float, default=-30.0)
    parser.add_argument("--fit-stop-ns", type=float, default=1200.0)
    parser.add_argument("--t0-min-ns", type=float, default=-30.0,
                        help="Earliest model onset relative to each event's t10_left")
    parser.add_argument("--t0-max-ns", type=float, default=0.0,
                        help="Latest model onset relative to each event's t10_left")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--t0-training-events", type=int, default=2500)
    parser.add_argument("--random-per-class", type=int, default=24)
    parser.add_argument("--plots-per-page", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    feature_csv = args.feature_csv or args.npz.with_name(args.npz.stem + "_amp_tau_eff.csv")
    selected = accepted_events(feature_csv)
    laser_constants = read_laser_constants(args.laser_events_csv)
    temporary = args.cache_dir is None
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="alpha_l_shape_"))
    rng = np.random.default_rng(args.seed)
    try:
        ch0, ch1, npts, sample_rate, ref_position = extract_arrays(args.npz, cache_dir)
        full_time = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
        pre_end = max(2, int((ref_position - 10.0) / 100.0 * npts))
        dt = float(full_time[1] - full_time[0])
        time = np.arange(args.fit_start_ns,
                         args.fit_stop_ns + 0.5 * dt * args.stride,
                         dt * args.stride)
        # Retain only events whose complete t10-relative fit window is present.
        selected = selected[
            selected["t10_left"].between(
                full_time[0] - args.fit_start_ns,
                full_time[-1] - args.fit_stop_ns)
        ].reset_index(drop=True)
        event_ids = selected["event"].to_numpy(dtype=int)
        centers = selected["t10_left"].to_numpy(dtype=float)
        waves, peaks, noises, snrs = load_selected_waveforms(
            ch0, ch1, event_ids, centers, full_time, time, pre_end,
            args.chunk_size)

        train_ids = rng.choice(len(waves), min(args.t0_training_events, len(waves)),
                               replace=False)
        if args.t0_min_ns >= args.t0_max_ns:
            raise ValueError("--t0-min-ns must be smaller than --t0-max-ns")
        alpha_t0, objective = estimate_common_t0(
            time, waves[train_ids], laser_constants,
            args.t0_min_ns, args.t0_max_ns)
        theta = make_theta(*laser_constants[:3], alpha_t0)
        design = basis(time, theta)
        amplitudes = vector_nnls(waves, design)
        models, metrics = event_metrics(time, waves, design, amplitudes)

        results = selected.copy()
        results["raw_peak"] = peaks; results["pretrigger_rms"] = noises
        results["snr"] = snrs
        results["A_prompt_normalized"] = amplitudes[:, 0]
        results["A_slow_normalized"] = amplitudes[:, 1]
        results["prompt_fraction"] = amplitudes[:, 0] / np.maximum(
            amplitudes.sum(axis=1), 1e-15)
        for name, values in metrics.items(): results[name] = values
        results["class"] = np.where(
            results["early_nrmse_0_300ns"] <= args.early_threshold,
            "L-like", "C-like")
        results["laser_tau_r_ns"] = laser_constants[0]
        results["laser_tau_fast_ns"] = laser_constants[1]
        results["laser_tau_slow_ns"] = laser_constants[2]
        results["alpha_common_t0_ns"] = alpha_t0
        results.to_csv(args.output_prefix.with_suffix(".csv"), index=False)

        make_pdf(args.output_prefix.with_suffix(".pdf"), results, time, waves, models,
                 args.early_threshold,
                 (*laser_constants[:3], alpha_t0), args.random_per_class,
                 args.plots_per_page, args.seed, ch0, ch1, full_time, pre_end)
        print(f"alpha t0={alpha_t0:.3f} ns, trimmed objective={objective:.6g}")
        print(results.groupby("class").agg(
            n=("event", "size"),
            early_nrmse_median=("early_nrmse_0_300ns", "median"),
            whole_nrmse_median=("nrmse", "median"),
            late_nrmse_median=("late_nrmse_300_1200ns", "median"),
        ).to_string())
        print(f"saved {args.output_prefix.with_suffix('.csv')}")
        print(f"saved {args.output_prefix.with_suffix('.pdf')}")
    finally:
        if temporary and not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
