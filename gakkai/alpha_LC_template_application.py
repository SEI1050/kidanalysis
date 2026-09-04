"""Calibrate an aligned laser L/C classifier and apply it to alpha events."""

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
from alpha_L_prompt_slow_classification import accepted_events, load_selected_waveforms
from laser_eventwise_L_shared_tau_C_test import find_file, load_abs_iq
from laser_LC_template_validation import (
    L_RUNS, C_RUNS, split_ids, feature_frame, precision_limit,
)


CLASSES = ("L-like", "C-like", "ambiguous", "Neither")
COLORS = {"L-like": "C0", "C-like": "C3", "ambiguous": "0.45",
          "Neither": "C4"}


def t10_crossings(time, waves):
    centers = np.full(len(waves), np.nan)
    for row, wave in enumerate(waves):
        peak = int(np.argmax(wave)); target = 0.1 * float(wave[peak])
        candidates = np.flatnonzero(wave[:peak + 1] <= target)
        if not len(candidates) or candidates[-1] >= peak:
            continue
        left = int(candidates[-1]); right = left + 1
        denominator = float(wave[right] - wave[left])
        fraction = 0.0 if abs(denominator) < 1e-15 else (
            target - float(wave[left])) / denominator
        centers[row] = time[left] + fraction * (time[right] - time[left])
    return centers


def align_existing(time, waves, centers, relative_time):
    aligned = np.empty((len(waves), len(relative_time)), dtype=np.float32)
    for row, center in enumerate(centers):
        aligned[row] = np.interp(center + relative_time, time, waves[row])
    peak = np.max(aligned, axis=1)
    return aligned / np.maximum(peak[:, None], 1e-15)


def predict(frame, c_limit, l_limit, distance_limit):
    score = frame["score_dC_minus_dL"].to_numpy()
    low, high = sorted((c_limit, l_limit))
    label = np.full(len(frame), "ambiguous", dtype=object)
    label[score < low] = "C-like"
    label[score > high] = "L-like"
    label[np.minimum(frame.d_L_early, frame.d_C_early) > distance_limit] = "Neither"
    result = frame.copy(); result["classification"] = label
    return result, low, high


def laser_model(root, relative_time, snr_threshold, rng, template_fraction,
                calibration_fraction, target_precision):
    template_sets = {"L": [], "C": []}; template_names = {"L": [], "C": []}
    split_waves = {"calibration": [], "test": []}
    split_meta = {"calibration": [], "test": []}
    inputs = []
    for true_class, runs in (("L", L_RUNS), ("C", C_RUNS)):
        for z, timestamp in runs.items():
            try:
                path = find_file(root, timestamp)
            except FileNotFoundError:
                print(f"WARNING: skip missing laser {true_class} z={z:.2f}", flush=True)
                continue
            time, waves, peak, _, snr, accepted = load_abs_iq(
                path, -350.0, 1600.0, 5, snr_threshold)
            ids = np.flatnonzero(accepted)
            centers = t10_crossings(time, waves[ids])
            complete = (np.isfinite(centers)
                        & (centers + relative_time[0] >= time[0])
                        & (centers + relative_time[-1] <= time[-1]))
            ids = ids[complete]; centers = centers[complete]
            aligned = align_existing(time, waves[ids], centers, relative_time)
            local = np.arange(len(aligned))
            train, calibration, test = split_ids(
                local, rng, template_fraction, calibration_fraction)
            template = np.median(aligned[train], axis=0)
            template /= max(float(template.max()), 1e-15)
            template_sets[true_class].append(template)
            template_names[true_class].append(f"{true_class} z={z:.2f}")
            for split, chosen in (("calibration", calibration), ("test", test)):
                split_waves[split].append(aligned[chosen])
                split_meta[split].append(pd.DataFrame({
                    "true_class": true_class, "z_mm": z,
                    "timestamp": timestamp, "event": ids[chosen],
                    "t10_left_ns": centers[chosen], "snr": snr[ids[chosen]],
                }))
            inputs.append({"class": true_class, "z_mm": z, "timestamp": timestamp,
                           "usable": len(ids), "template_n": len(train),
                           "calibration_n": len(calibration), "test_n": len(test)})

    for name in ("L", "C"):
        if len(template_sets[name]) < 2:
            raise RuntimeError(f"Not enough usable {name} laser positions")
    frames = {}; stored_waves = {}
    early = (relative_time >= 0) & (relative_time <= 300)
    for split in ("calibration", "test"):
        stored_waves[split] = np.concatenate(split_waves[split])
        frames[split] = feature_frame(
            stored_waves[split], pd.concat(split_meta[split], ignore_index=True),
            template_sets["L"], template_sets["C"], early)
    calibration = frames["calibration"]
    scores = calibration.score_dC_minus_dL.to_numpy()
    labels = calibration.true_class.to_numpy()
    c_limit = precision_limit(scores, labels, target_precision, "C")
    l_limit = precision_limit(scores, labels, target_precision, "L")
    d_true = np.where(labels == "L", calibration.d_L_early,
                      calibration.d_C_early)
    # The larger class-specific 95th percentile retains at least 95% of each
    # laser class before declaring an event outside both known populations.
    distance_limit = max(
        np.quantile(d_true[labels == "L"], .95),
        np.quantile(d_true[labels == "C"], .95))
    test, low, high = predict(frames["test"], c_limit, l_limit, distance_limit)
    return {
        "templates": template_sets, "template_names": template_names,
        "calibration": calibration, "test": test,
        "test_waves": stored_waves["test"], "low": low, "high": high,
        "distance_limit": float(distance_limit), "inputs": pd.DataFrame(inputs),
    }


def alpha_features(npz_path, feature_csv, relative_time, templates, chunk_size,
                   cache_dir):
    selected = accepted_events(feature_csv)
    ch0, ch1, npts, sample_rate, ref_position = extract_arrays(npz_path, cache_dir)
    full_time = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
    selected = selected[selected.t10_left.between(
        full_time[0] - relative_time[0], full_time[-1] - relative_time[-1]
    )].reset_index(drop=True)
    event_ids = selected.event.to_numpy(dtype=int)
    centers = selected.t10_left.to_numpy(dtype=float)
    pre_end = max(2, int((ref_position - 10.0) / 100.0 * npts))
    waves, peaks, noises, snrs = load_selected_waveforms(
        ch0, ch1, event_ids, centers, full_time, relative_time, pre_end, chunk_size)
    metadata = selected.copy(); metadata["raw_peak_aligned"] = peaks
    metadata["pretrigger_rms"] = noises; metadata["snr"] = snrs
    early = (relative_time >= 0) & (relative_time <= 300)
    frame = feature_frame(waves, metadata, templates["L"], templates["C"], early)
    return frame, waves


def confusion(frame):
    return pd.crosstab(frame.true_class, frame.classification, margins=True)


def make_pdf(path, model, alpha, alpha_waves, relative_time, rng,
             random_per_class):
    test = model["test"]
    with PdfPages(path) as pdf:
        matrix = confusion(test)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].axis("off")
        counts = alpha.classification.value_counts()
        axes[0].text(0.02, .98,
            "Aligned laser validation and alpha application\n\n"
            "All waveforms aligned to t10_left = 0\n"
            f"C-like: score < {model['low']:.4f}\n"
            f"L-like: score > {model['high']:.4f}\n"
            f"Neither: min(dL,dC) > {model['distance_limit']:.4f}\n\n"
            "Laser independent test:\n" + matrix.to_string() + "\n\n"
            f"Alpha usable: {len(alpha):,}\n" + counts.to_string(),
            va="top", family="monospace", fontsize=9)
        order = [x for x in CLASSES if x in matrix.columns]
        shown = matrix.loc[[x for x in ("L", "C") if x in matrix.index], order]
        im = axes[1].imshow(shown.to_numpy(), cmap="Blues")
        axes[1].set_xticks(range(len(order)), order, rotation=25, ha="right")
        axes[1].set_yticks(range(len(shown)), shown.index)
        axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true laser class")
        for i in range(shown.shape[0]):
            for j in range(shown.shape[1]):
                axes[1].text(j, i, f"{shown.iloc[i,j]:,}", ha="center", va="center")
        fig.colorbar(im, ax=axes[1], shrink=.8)
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
        counts = alpha.classification.value_counts().reindex(CLASSES, fill_value=0)
        axes[0].bar(counts.index, counts.values,
                    color=[COLORS[x] for x in counts.index])
        axes[0].tick_params(axis="x", rotation=25); axes[0].set_ylabel("alpha events")
        for name in CLASSES:
            group = alpha[alpha.classification == name]
            axes[1].hist(group.score_dC_minus_dL, bins=100, histtype="step",
                         lw=1.2, color=COLORS[name], label=name)
            axes[2].scatter(group.d_L_early, group.d_C_early, s=3, alpha=.15,
                            color=COLORS[name], label=name)
        axes[1].axvspan(model["low"], model["high"], color=".5", alpha=.15)
        axes[1].set_xlabel("score = d_C - d_L"); axes[1].set_yscale("log")
        axes[2].set_xlabel("d_L early NRMSE"); axes[2].set_ylabel("d_C early NRMSE")
        for axis in axes: axis.grid(alpha=.2)
        axes[1].legend(fontsize=7); axes[2].legend(fontsize=7)
        fig.suptitle("Alpha classification relative to measured laser templates")
        pdf.savefig(fig); plt.close(fig)

        for name in CLASSES:
            group = alpha[alpha.classification == name]
            chosen = rng.choice(group.index.to_numpy(),
                                min(random_per_class, len(group)), replace=False)
            ncols = 3; nrows = max(1, int(np.ceil(len(chosen) / ncols)))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows),
                                     constrained_layout=True, squeeze=False)
            for axis, row_id in zip(axes.ravel(), chosen):
                row = alpha.loc[row_id]
                if row.d_L_early <= row.d_C_early:
                    template = model["templates"]["L"][int(row.nearest_L_template)]
                    nearest = "L"
                else:
                    template = model["templates"]["C"][int(row.nearest_C_template)]
                    nearest = "C"
                axis.plot(relative_time, alpha_waves[row_id], color=".25", lw=.7,
                          label="alpha")
                axis.plot(relative_time, template, color=COLORS[name], lw=1.2,
                          label=f"nearest {nearest} template")
                axis.set_title(f"event {int(row.event)}  dL={row.d_L_early:.3f} "
                               f"dC={row.d_C_early:.3f}", fontsize=8)
                axis.set_xlabel("time from t10_left [ns]"); axis.set_ylabel("normalized |IQ|")
                axis.grid(alpha=.2)
            for axis in axes.ravel()[len(chosen):]: axis.axis("off")
            if len(chosen): axes.ravel()[0].legend(fontsize=7)
            fig.suptitle(f"Random alpha events: {name}")
            pdf.savefig(fig); plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("laser_root", nargs="?", type=Path,
                        default=Path("/Users/kubokosei/Downloads/20260825"))
    parser.add_argument("--alpha-npz", type=Path, default=here / "alphaDC_combined.npz")
    parser.add_argument("--alpha-feature-csv", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path,
                        default=here / "alpha_LC_template_application")
    parser.add_argument("--snr-threshold", type=float, default=8.0)
    parser.add_argument("--target-precision", type=float, default=.95)
    parser.add_argument("--random-per-class", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args(); rng = np.random.default_rng(args.seed)
    relative_time = np.arange(-30.0, 1200.01, 2.0)
    model = laser_model(args.laser_root, relative_time, args.snr_threshold, rng,
                        .4, .3, args.target_precision)
    cache_dir = Path(tempfile.mkdtemp(prefix="alpha_lc_template_"))
    try:
        feature_csv = args.alpha_feature_csv or args.alpha_npz.with_name(
            args.alpha_npz.stem + "_amp_tau_eff.csv")
        alpha, alpha_waves = alpha_features(
            args.alpha_npz, feature_csv, relative_time, model["templates"],
            args.chunk_size, cache_dir)
        alpha, _, _ = predict(alpha, model["low"], model["high"],
                              model["distance_limit"])
        alpha.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
        model["test"].to_csv(args.output_prefix.with_name(
            args.output_prefix.name + "_laser_test.csv"), index=False)
        model["inputs"].to_csv(args.output_prefix.with_name(
            args.output_prefix.name + "_laser_inputs.csv"), index=False)
        make_pdf(args.output_prefix.with_suffix(".pdf"), model, alpha, alpha_waves,
                 relative_time, rng, args.random_per_class)
        print("laser test confusion:")
        print(confusion(model["test"]).to_string())
        print("alpha classification:")
        print(alpha.classification.value_counts().to_string())
        print(f"saved {args.output_prefix.with_suffix('.pdf')}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
