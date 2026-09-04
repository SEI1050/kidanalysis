"""Event-wise test of an L-derived prompt+slow model on L and C positions.

No median waveform is fitted.  Each waveform is the pedestal-subtracted IQ
magnitude, |(I-Ipre)+i(Q-Qpre)|, with its pretrigger magnitude floor removed.
Shared time constants are learned from accepted L events, and the two
non-negative amplitudes are solved separately for every event.  The same
frozen L basis is then fitted event-by-event to C.
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
from scipy.optimize import differential_evolution, minimize



L_RUNS = {
    6.8: "134954", 6.7: "134832", 6.5: "134710",
    6.3: "134447", 6.1: "134328", 6.0: "134208",
}
C_RUNS = {6.8: "130955", 6.7: "131115", 6.3: "131249", 6.0: "131518"}


def find_file(root, timestamp):
    files = sorted((root / f"data_0825_{timestamp}").glob("wf_*.npz"))
    if not files:
        raise FileNotFoundError(timestamp)
    return files[0]


def load_abs_iq(path, fit_start, fit_stop, stride, snr_threshold):
    with np.load(path, allow_pickle=False) as data:
        ch0 = np.asarray(data["ch0"], dtype=np.float32)
        ch1 = np.asarray(data["ch1"], dtype=np.float32)
        npts = int(data["npts"]); rate = float(data["sample_rate"])
        ref = float(data["ref_position"])
    time = (np.arange(npts) - npts * ref / 100.0) / rate * 1e9
    pre_end = max(2, int((ref - 10.0) / 100.0 * npts))
    pedestal_i = np.median(ch0[:, :pre_end], axis=1)
    pedestal_q = np.median(ch1[:, :pre_end], axis=1)
    dev_i = ch0 - pedestal_i[:, None]
    dev_q = ch1 - pedestal_q[:, None]
    magnitude = np.hypot(dev_i, dev_q)
    noise_floor = np.median(magnitude[:, :pre_end], axis=1)
    signal = magnitude - noise_floor[:, None]
    noise = np.std(signal[:, :pre_end], axis=1)
    fit_index = np.flatnonzero((time >= fit_start) & (time <= fit_stop))[::stride]
    y = np.asarray(signal[:, fit_index], dtype=np.float64)
    peak = np.max(y, axis=1)
    snr = peak / np.maximum(noise, 1e-15)
    accepted = np.isfinite(snr) & (snr >= snr_threshold) & (peak > 0)
    # Normalize only for shape fitting. The original peak is retained in CSV.
    normalized = y / np.maximum(peak[:, None], 1e-15)
    return time[fit_index], normalized, peak, noise, snr, accepted


def unpack(theta):
    tau_r = np.exp(theta[0])
    tau_f = tau_r + np.exp(theta[1])
    tau_s = tau_f + np.exp(theta[2])
    return tau_r, tau_f, tau_s, theta[3]


def basis(time, theta):
    tr, tf, ts, t0 = unpack(theta)
    u = np.maximum(time - t0, 0.0); gate = time >= t0
    x1 = np.where(gate, np.exp(-u / tf) - np.exp(-u / tr), 0.0)
    x2 = np.where(gate, np.exp(-u / ts) - np.exp(-u / tr), 0.0)
    # Unit-peak normalization makes coefficients interpretable and avoids huge
    # amplitudes when two fitted time constants approach each other.
    x1 /= max(np.max(x1), 1e-15); x2 /= max(np.max(x2), 1e-15)
    return np.column_stack((x1, x2))


def vector_nnls(y, design):
    """Exact non-negative least squares for a two-column design, vectorized."""
    gram = design.T @ design
    b = y @ design
    inv = np.linalg.pinv(gram)
    unconstrained = b @ inv
    a_fast = np.maximum(b[:, 0] / max(gram[0, 0], 1e-30), 0.0)
    a_slow = np.maximum(b[:, 1] / max(gram[1, 1], 1e-30), 0.0)
    candidates = np.stack((
        unconstrained,
        np.column_stack((a_fast, np.zeros_like(a_fast))),
        np.column_stack((np.zeros_like(a_slow), a_slow)),
        np.zeros_like(unconstrained),
    ), axis=1)
    invalid = np.any(candidates[:, 0, :] < 0, axis=1)
    candidates[invalid, 0, :] = 0.0
    y2 = np.sum(y * y, axis=1)[:, None]
    ab = np.einsum("nki,ni->nk", candidates, b)
    aga = np.einsum("nki,ij,nkj->nk", candidates, gram, candidates)
    rss = y2 - 2.0 * ab + aga
    choice = np.argmin(rss, axis=1)
    return candidates[np.arange(len(y)), choice]


def fit_constants(time, y, seed):
    def objective(theta):
        design = basis(time, theta)
        amps = vector_nnls(y, design)
        residual = y - amps @ design.T
        event_mse = np.mean(residual * residual, axis=1)
        # A 5% trim prevents a handful of pickup/noise events defining the basis.
        cutoff = np.quantile(event_mse, 0.95)
        return float(np.mean(np.minimum(event_mse, cutoff)))

    bounds = [
        (np.log(1.0), np.log(250.0)),
        (np.log(0.5), np.log(1500.0)),
        (np.log(10.0), np.log(5000.0)),
        (-30.0, 120.0),
    ]
    global_result = differential_evolution(
        objective, bounds, seed=seed, popsize=8, maxiter=45, polish=False,
        updating="immediate", workers=1,
    )
    local = minimize(objective, global_result.x, method="Powell", bounds=bounds,
                     options={"maxiter": 500, "xtol": 1e-4, "ftol": 1e-9})
    return local.x, float(local.fun)


def event_metrics(time, y, design, amps):
    model = amps @ design.T
    residual = y - model
    early = (time >= 0) & (time <= 300)
    late = (time > 300) & (time <= 1200)
    data_peak_index = np.argmax(y, axis=1)
    rows = np.arange(len(y))
    return model, {
        "nrmse": np.sqrt(np.mean(residual ** 2, axis=1)),
        "early_nrmse_0_300ns": np.sqrt(np.mean(residual[:, early] ** 2, axis=1)),
        "late_nrmse_300_1200ns": np.sqrt(np.mean(residual[:, late] ** 2, axis=1)),
        "max_abs_residual_over_peak": np.max(np.abs(residual), axis=1),
        "residual_at_data_peak_over_peak": residual[rows, data_peak_index],
        "r2": 1.0 - np.sum(residual ** 2, axis=1)
        / np.maximum(np.sum((y - np.mean(y, axis=1, keepdims=True)) ** 2, axis=1), 1e-30),
    }


def summary_table(events):
    good = events[events["accepted"]].copy()
    metrics = ["nrmse", "early_nrmse_0_300ns", "late_nrmse_300_1200ns",
               "max_abs_residual_over_peak", "residual_at_data_peak_over_peak", "r2"]
    rows = []
    for (region, z), group in good.groupby(["region", "z_mm"], sort=True):
        row = {"region": region, "z_mm": z, "n_total": int(
            len(events[(events.region == region) & (events.z_mm == z)])),
               "n_accepted": len(group)}
        for metric in metrics:
            row[f"{metric}_median"] = group[metric].median()
            row[f"{metric}_q16"] = group[metric].quantile(.16)
            row[f"{metric}_q84"] = group[metric].quantile(.84)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("/Volumes/NO NAME/data/20260825"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("laser_eventwise_L_shared_tau_C_test"))
    parser.add_argument("--snr-threshold", type=float, default=0.0)
    parser.add_argument("--fit-start-ns", type=float, default=-30.0)
    parser.add_argument("--fit-stop-ns", type=float, default=1200.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--train-per-position", type=int, default=300)
    parser.add_argument("--examples-per-position", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    datasets = {}
    input_rows = []
    for region, runs, x in (("L", L_RUNS, 5.1), ("C", C_RUNS, 4.6)):
        for z, timestamp in runs.items():
            path = find_file(args.root, timestamp)
            loaded = load_abs_iq(path, args.fit_start_ns, args.fit_stop_ns,
                                 args.stride, args.snr_threshold)
            datasets[(region, z)] = loaded
            input_rows.append({"region": region, "x_mm": x, "z_mm": z,
                               "timestamp": timestamp, "file": str(path),
                               "n_events": len(loaded[1]),
                               "n_accepted": int(np.sum(loaded[5]))})
            print(f"loaded {region} z={z}: {np.sum(loaded[5])}/{len(loaded[5])} accepted")
    pd.DataFrame(input_rows).to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_inputs.csv"), index=False)

    time = next(iter(datasets.values()))[0]
    training = []
    for key in sorted(k for k in datasets if k[0] == "L"):
        y, accepted = datasets[key][1], datasets[key][5]
        ids = np.flatnonzero(accepted)
        if len(ids) > args.train_per_position:
            ids = rng.choice(ids, args.train_per_position, replace=False)
        training.append(y[ids])
    train_y = np.concatenate(training)
    theta, objective = fit_constants(time, train_y, args.seed)
    constants = unpack(theta); design = basis(time, theta)
    print("L constants:", constants, "objective:", objective, flush=True)

    event_frames = []
    example_store = {}
    for (region, z), loaded in datasets.items():
        _, y, peak, noise, snr, accepted = loaded
        amps = vector_nnls(y, design)
        model, values = event_metrics(time, y, design, amps)
        frame = pd.DataFrame({
            "region": region, "x_mm": 5.1 if region == "L" else 4.6,
            "z_mm": z, "event": np.arange(len(y)), "accepted": accepted,
            "raw_peak": peak, "pretrigger_rms": noise, "snr": snr,
            "A_fast_normalized": amps[:, 0], "A_slow_normalized": amps[:, 1],
            "fast_fraction": amps[:, 0] / np.maximum(np.sum(amps, axis=1), 1e-30),
            **values,
        })
        event_frames.append(frame)
        good_ids = np.flatnonzero(accepted)
        order = good_ids[np.argsort(frame.loc[good_ids, "nrmse"].to_numpy())]
        quantile_index = np.linspace(0, max(len(order) - 1, 0),
                                     args.examples_per_position).astype(int)
        example_store[(region, z)] = (y, model, order[quantile_index])

    events = pd.concat(event_frames, ignore_index=True)
    for name, value in zip(("shared_tau_r_ns", "shared_tau_fast_ns",
                            "shared_tau_slow_ns", "shared_t0_ns"), constants):
        events[name] = value
    events.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    summary = summary_table(events)
    summary.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
                   index=False)

    good = events[events.accepted]
    l_error = good[good.region == "L"].nrmse
    c_error = good[good.region == "C"].nrmse
    l95 = float(l_error.quantile(.95))
    comparison = pd.DataFrame([{
        "L_n": len(l_error), "C_n": len(c_error),
        "L_nrmse_median": l_error.median(), "C_nrmse_median": c_error.median(),
        "C_over_L_nrmse_median": c_error.median() / l_error.median(),
        "L_early_nrmse_median": good[good.region == "L"].early_nrmse_0_300ns.median(),
        "C_early_nrmse_median": good[good.region == "C"].early_nrmse_0_300ns.median(),
        "L_nrmse_95pct": l95,
        "C_fraction_above_L_95pct": float(np.mean(c_error > l95)),
    }])
    comparison.to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_comparison.csv"), index=False)

    with PdfPages(args.output_prefix.with_suffix(".pdf")) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for axis, metric, title in zip(
            axes,
            ("nrmse", "early_nrmse_0_300ns", "late_nrmse_300_1200ns"),
            ("whole fit", "early: 0-300 ns", "late: 300-1200 ns"),
        ):
            labels, series = [], []
            for region in ("L", "C"):
                for z in sorted(good[good.region == region].z_mm.unique(), reverse=True):
                    # Keep the full scan coordinate: one decimal place made
                    # 6.75 and 6.85 both appear as 6.8 on the expanded scan.
                    labels.append(f"{region} {z:.2f}")
                    series.append(good[(good.region == region) & (good.z_mm == z)][metric])
            axis.boxplot(series, tick_labels=labels, showfliers=False)
            axis.set_title(title); axis.set_ylabel("NRMSE / event peak")
            axis.set_xlabel("region and z position [mm]")
            axis.grid(alpha=.25)
            axis.tick_params(axis="x", labelsize=7, labelrotation=45)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
        fig.suptitle(f"Event-wise L model: tau_r={constants[0]:.1f}, "
                     f"tau_f={constants[1]:.1f}, tau_s={constants[2]:.1f} ns")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        bins = np.linspace(0, min(.5, good.nrmse.quantile(.995)), 120)
        axes[0].hist(l_error, bins=bins, density=True, histtype="step", lw=1.5, label="L")
        axes[0].hist(c_error, bins=bins, density=True, histtype="step", lw=1.5, label="C")
        axes[0].axvline(l95, color="C0", ls="--", label="L 95%")
        axes[0].set_xlabel("event NRMSE"); axes[0].set_ylabel("density")
        axes[0].legend(); axes[0].grid(alpha=.25)
        for region, color in (("L", "C0"), ("C", "C3")):
            subset = good[good.region == region]
            axes[1].scatter(subset.nrmse, subset.early_nrmse_0_300ns,
                            s=3, alpha=.15, label=region, color=color)
        axes[1].set_xlabel("whole NRMSE"); axes[1].set_ylabel("early NRMSE")
        axes[1].legend(); axes[1].grid(alpha=.25)
        pdf.savefig(fig); plt.close(fig)

        for key in sorted(example_store, key=lambda k: (k[0], -k[1])):
            region, z = key; y, model, ids = example_store[key]
            fig, axes = plt.subplots(len(ids), 2, figsize=(12, 2.5 * len(ids)),
                                     constrained_layout=True, squeeze=False)
            for row, event in enumerate(ids):
                axes[row, 0].plot(time, y[event], color="black", lw=.7, label="event")
                axes[row, 0].plot(time, model[event], color="C0" if region == "L" else "C3",
                                  lw=1.2, label="L-derived fit")
                axes[row, 0].set_title(f"{region} z={z:.1f}, event {event}")
                axes[row, 0].legend(fontsize=7); axes[row, 0].grid(alpha=.2)
                axes[row, 1].plot(time, y[event] - model[event], color="C3", lw=.7)
                axes[row, 1].axhline(0, color="black", lw=.6)
                axes[row, 1].set_title("residual / event peak"); axes[row, 1].grid(alpha=.2)
            for axis in axes.ravel(): axis.set_xlabel("time [ns]")
            fig.suptitle(f"Individual events across fit-error quantiles: {region}, z={z:.1f}")
            pdf.savefig(fig); plt.close(fig)

    print(summary[["region", "z_mm", "n_accepted", "nrmse_median",
                   "early_nrmse_0_300ns_median", "late_nrmse_300_1200ns_median",
                   "r2_median"]].to_string(index=False))
    print(comparison.to_string(index=False))
    print(f"saved {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
