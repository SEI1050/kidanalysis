"""Validate whether laser L and C events are separable from waveform shape.

Each scan position is split into independent template, calibration, and test
subsets.  Position-specific median templates are built only from the template
subset.  For every event, d_L and d_C are the minimum early-window NRMSE to an
L or C template.  Calibration data determine conservative L/C score limits;
events between them are reported as ambiguous rather than forced into a class.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kidanalysis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from laser_eventwise_L_shared_tau_C_test import find_file, load_abs_iq


# Signal-rich x=5.10 L scans.  The low-SNR 5.70, 6.85, and 6.90 scans are not
# suitable for validating a waveform-shape classifier.
L_RUNS = {
    5.90: "134041", 5.95: "134123", 6.00: "134208", 6.05: "134248",
    6.10: "134328", 6.20: "134407", 6.30: "134447", 6.40: "134626",
    6.50: "134710", 6.60: "134748", 6.70: "134832", 6.75: "134913",
    6.80: "134954",
}
C_RUNS = {6.00: "131518", 6.30: "131249", 6.70: "131115", 6.80: "130955"}


def split_ids(ids, rng, template_fraction, calibration_fraction):
    ids = rng.permutation(ids)
    n_template = int(len(ids) * template_fraction)
    n_calibration = int(len(ids) * calibration_fraction)
    return (ids[:n_template], ids[n_template:n_template + n_calibration],
            ids[n_template + n_calibration:])


def distances(waves, templates, mask):
    result = np.full(len(waves), np.inf)
    nearest = np.full(len(waves), -1, dtype=int)
    for index, template in enumerate(templates):
        value = np.sqrt(np.mean((waves[:, mask] - template[None, mask]) ** 2,
                                axis=1))
        replace = value < result
        result[replace] = value[replace]
        nearest[replace] = index
    return result, nearest


def feature_frame(waves, metadata, l_templates, c_templates, early):
    d_l, nearest_l = distances(waves, l_templates, early)
    d_c, nearest_c = distances(waves, c_templates, early)
    frame = metadata.copy().reset_index(drop=True)
    frame["d_L_early"] = d_l
    frame["d_C_early"] = d_c
    frame["score_dC_minus_dL"] = d_c - d_l
    frame["nearest_L_template"] = nearest_l
    frame["nearest_C_template"] = nearest_c
    return frame


def precision_limit(scores, labels, target, side):
    order = np.argsort(scores)
    scores = scores[order]; labels = labels[order]
    if side == "C":
        correct = np.cumsum(labels == "C")
        precision = correct / np.arange(1, len(scores) + 1)
        valid = np.flatnonzero(precision >= target)
        return float(scores[valid[-1]]) if len(valid) else float(scores[0])
    correct = np.cumsum((labels[::-1] == "L"))[::-1]
    count = np.arange(len(scores), 0, -1)
    precision = correct / count
    valid = np.flatnonzero(precision >= target)
    return float(scores[valid[0]]) if len(valid) else float(scores[-1])


def classify(frame, c_limit, l_limit):
    score = frame["score_dC_minus_dL"].to_numpy()
    if c_limit >= l_limit:
        # No clean high-purity gap: keep the overlap explicitly ambiguous.
        low, high = l_limit, c_limit
    else:
        low, high = c_limit, l_limit
    predicted = np.full(len(frame), "ambiguous", dtype=object)
    predicted[score < low] = "C"
    predicted[score > high] = "L"
    result = frame.copy(); result["predicted"] = predicted
    return result, low, high


def confusion_table(frame):
    return pd.crosstab(frame["true_class"], frame["predicted"], margins=True)


def make_pdf(path, calibration, test, templates, time, early, low, high,
             target_precision, rng, examples):
    colors = {"L": "C0", "C": "C3", "ambiguous": "0.45"}
    with PdfPages(path) as pdf:
        matrix = confusion_table(test)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].axis("off")
        axes[0].text(0.02, 0.98,
                     "Laser L/C waveform-template validation\n\n"
                     f"Independent test events: {len(test):,}\n"
                     f"Target calibration precision: {target_precision:.1%}\n"
                     f"C if score < {low:.4f}\n"
                     f"L if score > {high:.4f}\n"
                     "otherwise: ambiguous\n\n"
                     + matrix.to_string(), va="top", family="monospace", fontsize=10)
        shown = matrix.loc[[x for x in ("L", "C") if x in matrix.index],
                           [x for x in ("L", "C", "ambiguous") if x in matrix.columns]]
        image = axes[1].imshow(shown.to_numpy(), cmap="Blues")
        axes[1].set_xticks(range(shown.shape[1]), shown.columns)
        axes[1].set_yticks(range(shown.shape[0]), shown.index)
        axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
        axes[1].set_title("Independent test confusion matrix")
        for i in range(shown.shape[0]):
            for j in range(shown.shape[1]):
                axes[1].text(j, i, f"{shown.iloc[i, j]:,}", ha="center", va="center")
        fig.colorbar(image, ax=axes[1], shrink=0.8)
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for true, color in (("L", "C0"), ("C", "C3")):
            group = test[test.true_class == true]
            axes[0].hist(group.score_dC_minus_dL, bins=100, histtype="step",
                         lw=1.5, label=f"true {true}", color=color)
            axes[1].scatter(group.d_L_early, group.d_C_early, s=4, alpha=.22,
                            label=f"true {true}", color=color)
        axes[0].axvspan(low, high, color="0.5", alpha=.18, label="ambiguous")
        axes[0].axvline(0, color="black", lw=.8)
        axes[0].set_xlabel("score = d_C - d_L"); axes[0].set_ylabel("test events")
        axes[0].legend(); axes[0].grid(alpha=.2)
        bound = max(test.d_L_early.quantile(.995), test.d_C_early.quantile(.995))
        axes[1].plot([0, bound], [0, bound], color="black", lw=.8)
        axes[1].set_xlim(0, bound); axes[1].set_ylim(0, bound)
        axes[1].set_xlabel("d_L early NRMSE"); axes[1].set_ylabel("d_C early NRMSE")
        axes[1].legend(); axes[1].grid(alpha=.2)
        fig.suptitle("Does an event resemble an L or C laser template?")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 constrained_layout=True)
        for axis, (name, values) in zip(axes, templates.items()):
            for label, waveform in values:
                axis.plot(time, waveform, lw=1, alpha=.8, label=label)
            axis.set_ylabel("normalized |IQ|"); axis.set_title(f"{name} templates")
            axis.grid(alpha=.2); axis.legend(ncol=7, fontsize=7)
        axes[-1].set_xlabel("time [ns]"); fig.suptitle("Training-only position templates")
        pdf.savefig(fig); plt.close(fig)

        groups = [("correct L", test[(test.true_class == "L") & (test.predicted == "L")]),
                  ("correct C", test[(test.true_class == "C") & (test.predicted == "C")]),
                  ("ambiguous", test[test.predicted == "ambiguous"]),
                  ("misclassified", test[(test.predicted != "ambiguous")
                                           & (test.predicted != test.true_class)])]
        fig, axes = plt.subplots(4, examples, figsize=(14, 9), constrained_layout=True,
                                 squeeze=False, sharex=True, sharey=True)
        all_waves = test.attrs["waves"]
        for row_index, (name, group) in enumerate(groups):
            ids = rng.choice(group.index.to_numpy(), min(examples, len(group)), replace=False)
            for column, row_id in enumerate(ids):
                axis = axes[row_index, column]; row = test.loc[row_id]
                axis.plot(time, all_waves[row_id], color=colors.get(row.predicted, "0.3"), lw=.7)
                axis.set_title(f"{name}\ntrue {row.true_class}, s={row.score_dC_minus_dL:.3f}",
                               fontsize=7)
                axis.grid(alpha=.15)
            for axis in axes[row_index, len(ids):]: axis.axis("off")
        for axis in axes[-1]: axis.set_xlabel("time [ns]")
        for axis in axes[:, 0]: axis.set_ylabel("normalized |IQ|")
        fig.suptitle("Random independent test waveforms")
        pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("/Volumes/NO NAME/data/20260825"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path(__file__).resolve().parent / "laser_LC_template_validation")
    parser.add_argument("--snr-threshold", type=float, default=8.0)
    parser.add_argument("--template-fraction", type=float, default=.4)
    parser.add_argument("--calibration-fraction", type=float, default=.3)
    parser.add_argument("--target-precision", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--examples-per-group", type=int, default=4)
    parser.add_argument("--require-all-runs", action="store_true",
                        help="Fail instead of skipping run folders without wf_*.npz")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    records = []; split_waves = {"template": [], "calibration": [], "test": []}
    split_meta = {key: [] for key in split_waves}
    template_sets = {"L": [], "C": []}; template_labels = {"L": [], "C": []}
    time = None
    for true_class, runs in (("L", L_RUNS), ("C", C_RUNS)):
        for z, timestamp in runs.items():
            try:
                path = find_file(args.root, timestamp)
            except FileNotFoundError:
                if args.require_all_runs:
                    raise
                print(f"WARNING: skip missing {true_class} z={z:.2f}, "
                      f"data_0825_{timestamp}", flush=True)
                continue
            loaded = load_abs_iq(path, -30, 1200, 5, args.snr_threshold)
            time, waves, peak, noise, snr, accepted = loaded
            ids = np.flatnonzero(accepted)
            train, calibration, test = split_ids(
                ids, rng, args.template_fraction, args.calibration_fraction)
            template = np.median(waves[train], axis=0)
            template /= max(float(template.max()), 1e-15)
            template_sets[true_class].append(template)
            template_labels[true_class].append(f"{true_class} z={z:.2f}")
            for split, chosen in (("template", train), ("calibration", calibration),
                                  ("test", test)):
                split_waves[split].append(waves[chosen])
                split_meta[split].append(pd.DataFrame({
                    "true_class": true_class, "z_mm": z, "timestamp": timestamp,
                    "event": chosen, "raw_peak": peak[chosen], "snr": snr[chosen],
                }))
            records.append({"class": true_class, "z_mm": z, "timestamp": timestamp,
                            "accepted": len(ids), "template_n": len(train),
                            "calibration_n": len(calibration), "test_n": len(test)})

    for name in ("L", "C"):
        if len(template_sets[name]) < 2:
            raise RuntimeError(f"Need at least two usable {name} runs; found "
                               f"{len(template_sets[name])}")

    early = (time >= 0) & (time <= 300)
    frames = {}
    for split in ("calibration", "test"):
        waves = np.concatenate(split_waves[split])
        metadata = pd.concat(split_meta[split], ignore_index=True)
        frames[split] = feature_frame(
            waves, metadata, template_sets["L"], template_sets["C"], early)
        frames[split].attrs["waves"] = waves

    calibration = frames["calibration"]
    scores = calibration.score_dC_minus_dL.to_numpy()
    labels = calibration.true_class.to_numpy()
    c_limit = precision_limit(scores, labels, args.target_precision, "C")
    l_limit = precision_limit(scores, labels, args.target_precision, "L")
    calibration, low, high = classify(calibration, c_limit, l_limit)
    test, _, _ = classify(frames["test"], c_limit, l_limit)
    test.attrs["waves"] = frames["test"].attrs["waves"]

    calibration.to_csv(args.output_prefix.with_name(
        args.output_prefix.name + "_calibration.csv"), index=False)
    test.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_test.csv"),
                index=False)
    pd.DataFrame(records).to_csv(args.output_prefix.with_name(
        args.output_prefix.name + "_inputs.csv"), index=False)
    templates_for_plot = {
        name: list(zip(template_labels[name], template_sets[name])) for name in ("L", "C")}
    make_pdf(args.output_prefix.with_suffix(".pdf"), calibration, test,
             templates_for_plot, time, early, low, high, args.target_precision,
             rng, args.examples_per_group)
    print(f"score limits: C < {low:.6f}, L > {high:.6f}")
    print(confusion_table(test).to_string())
    print(f"saved {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
