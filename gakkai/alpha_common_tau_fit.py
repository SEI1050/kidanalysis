"""Fit alpha pulse families with shared rise/fast/slow time constants.

The alpha events are divided into amplitude x decay-shape quantile groups.
For each group a pedestal-subtracted |IQ| median waveform is constructed, then
all medians are fitted simultaneously with

    s_g(t) = Af_g [exp(-u/tau_f) - exp(-u/tau_r)]
           + As_g [exp(-u/tau_s) - exp(-u/tau_r)],  u=t-t0 >= 0.

The time constants and t0 are shared, while Af and As vary by group.  A second
model fits separate time constants to every group.  AIC/BIC compare whether the
extra group-dependent time constants are warranted.

The large NPZ is extracted to a temporary directory and read using mmap, so the
two 1.3 GB waveform arrays do not need to reside in RAM simultaneously.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kidanalysis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def extract_arrays(npz_path: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = ("ch0.npy", "ch1.npy", "npts.npy", "sample_rate.npy", "ref_position.npy")
    with zipfile.ZipFile(npz_path) as archive:
        for name in required:
            target = cache_dir / name
            if not target.exists() or target.stat().st_size != archive.getinfo(name).file_size:
                print(f"extracting {name} ...", flush=True)
                with archive.open(name) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    return (
        np.load(cache_dir / "ch0.npy", mmap_mode="r"),
        np.load(cache_dir / "ch1.npy", mmap_mode="r"),
        int(np.load(cache_dir / "npts.npy")),
        float(np.load(cache_dir / "sample_rate.npy")),
        float(np.load(cache_dir / "ref_position.npy")),
    )


def select_groups(frame: pd.DataFrame, amp_bins: int, shape_bins: int,
                  min_amp: float, peak_min: float, peak_max: float):
    valid = frame.copy()
    for column in ("amp", "tau_d_eff", "t_peak"):
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid[
        np.isfinite(valid["amp"])
        & np.isfinite(valid["tau_d_eff"])
        & np.isfinite(valid["t_peak"])
        & (valid["amp"] >= min_amp)
        & valid["t_peak"].between(peak_min, peak_max)
        & (valid["tau_d_eff"] > 0)
    ].copy()
    if valid.empty:
        raise RuntimeError("No events survive the signal and peak-time cuts")
    valid["amp_group"] = pd.qcut(valid["amp"], amp_bins, labels=False, duplicates="drop")
    # Shape quantiles are evaluated inside each amplitude range, reducing the
    # otherwise strong correlation between pulse height and decay estimator.
    valid["shape_group"] = valid.groupby("amp_group", observed=True)["tau_d_eff"].transform(
        lambda values: pd.qcut(values, shape_bins, labels=False, duplicates="drop")
    )
    valid["group"] = (
        "A" + valid["amp_group"].astype(int).astype(str)
        + "_D" + valid["shape_group"].astype(int).astype(str)
    )
    return valid


def group_medians(ch0, ch1, frame: pd.DataFrame, time_ns: np.ndarray,
                  ref_position: float, max_events: int, seed: int):
    rng = np.random.default_rng(seed)
    pre_end = max(2, min(time_ns.size, int((ref_position - 10.0) / 100.0 * time_ns.size)))
    medians = {}
    summaries = []
    for group_name, group in frame.groupby("group", sort=True):
        indices = group["event"].astype(int).to_numpy()
        if indices.size > max_events:
            indices = rng.choice(indices, max_events, replace=False)
        waves = np.empty((indices.size, time_ns.size), dtype=np.float32)
        for row_index, event_index in enumerate(indices):
            i_wave = np.asarray(ch0[event_index], dtype=np.float32)
            q_wave = np.asarray(ch1[event_index], dtype=np.float32)
            i_signal = i_wave - np.mean(i_wave[:pre_end])
            q_signal = q_wave - np.mean(q_wave[:pre_end])
            magnitude = np.hypot(i_signal, q_signal)
            waves[row_index] = magnitude - np.median(magnitude[:pre_end])
        median = np.median(waves, axis=0)
        medians[group_name] = median
        summaries.append({
            "group": group_name,
            "n_available": len(group),
            "n_median": len(indices),
            "amp_median": float(group["amp"].median()),
            "tau_d_eff_median": float(group["tau_d_eff"].median()),
            "t_peak_median": float(group["t_peak"].median()),
        })
        print(f"median {group_name}: {len(indices)} / {len(group)} events", flush=True)
    return medians, pd.DataFrame(summaries)


def unpack_shared(parameters, n_groups):
    tau_r = np.exp(parameters[0])
    tau_f = tau_r + np.exp(parameters[1])
    tau_s = tau_f + np.exp(parameters[2])
    t0 = parameters[3]
    amplitudes = np.exp(parameters[4:].reshape(n_groups, 2))
    return tau_r, tau_f, tau_s, t0, amplitudes


def pulse_model(time_ns, tau_r, tau_f, tau_s, t0, af, a_s):
    u = np.maximum(time_ns - t0, 0.0)
    gate = time_ns >= t0
    model = (
        af * (np.exp(-u / tau_f) - np.exp(-u / tau_r))
        + a_s * (np.exp(-u / tau_s) - np.exp(-u / tau_r))
    )
    return np.where(gate, model, 0.0)


def fit_shared(time_fit, waveforms):
    n_groups = waveforms.shape[0]
    peaks = np.maximum(np.max(waveforms, axis=1), 1e-9)
    initial_amp = np.column_stack((0.6 * peaks, 0.4 * peaks))
    initial = np.concatenate((
        np.log([8.0, 80.0 - 8.0, 700.0 - 80.0]),
        [0.0],
        np.log(np.maximum(initial_amp, 1e-12)).ravel(),
    ))

    def residual(parameters):
        tau_r, tau_f, tau_s, t0, amplitudes = unpack_shared(parameters, n_groups)
        model = np.asarray([
            pulse_model(time_fit, tau_r, tau_f, tau_s, t0, af, a_s)
            for af, a_s in amplitudes
        ])
        return ((model - waveforms) / peaks[:, None]).ravel()

    lower = np.concatenate((np.log([0.2, 0.2, 0.2]), [-100.0], np.full(2 * n_groups, -30.0)))
    upper = np.concatenate((np.log([500.0, 5000.0, 20000.0]), [100.0], np.full(2 * n_groups, 5.0)))
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=5000)
    return result, unpack_shared(result.x, n_groups)


def fit_individual(time_fit, waveform):
    peak = max(float(np.max(waveform)), 1e-9)
    initial = np.concatenate((
        np.log([8.0, 72.0, 620.0]), [0.0], np.log([0.6 * peak, 0.4 * peak])
    ))

    def residual(parameters):
        tau_r, tau_f, tau_s, t0, amplitudes = unpack_shared(parameters, 1)
        model = pulse_model(time_fit, tau_r, tau_f, tau_s, t0, *amplitudes[0])
        return (model - waveform) / peak

    lower = np.concatenate((np.log([0.2, 0.2, 0.2]), [-100.0], [-30.0, -30.0]))
    upper = np.concatenate((np.log([500.0, 5000.0, 20000.0]), [100.0], [5.0, 5.0]))
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=3000)
    return result, unpack_shared(result.x, 1)


def information_criteria(residual, n_parameters):
    n = residual.size
    rss = float(np.sum(residual ** 2))
    variance = max(rss / n, np.finfo(float).tiny)
    aic = n * np.log(variance) + 2 * n_parameters
    bic = n * np.log(variance) + n_parameters * np.log(n)
    return rss, aic, bic


def create_outputs(prefix: Path, time_fit, names, waveforms, summaries,
                   shared_result, shared_fit, individual_fits):
    tau_r, tau_f, tau_s, t0, amplitudes = shared_fit
    shared_models = np.asarray([
        pulse_model(time_fit, tau_r, tau_f, tau_s, t0, af, a_s)
        for af, a_s in amplitudes
    ])
    shared_rss, shared_aic, shared_bic = information_criteria(
        shared_result.fun, shared_result.x.size
    )
    individual_models = []
    individual_residuals = []
    rows = []
    for index, (name, result, fitted) in enumerate(zip(names, *zip(*individual_fits))):
        i_tau_r, i_tau_f, i_tau_s, i_t0, i_amp = fitted
        model = pulse_model(time_fit, i_tau_r, i_tau_f, i_tau_s, i_t0, *i_amp[0])
        individual_models.append(model)
        individual_residuals.append(result.fun)
        rows.append({
            "group": name,
            "shared_Af": amplitudes[index, 0],
            "shared_As": amplitudes[index, 1],
            "shared_Af_fraction": amplitudes[index, 0] / amplitudes[index].sum(),
            "free_tau_r_ns": i_tau_r,
            "free_tau_f_ns": i_tau_f,
            "free_tau_s_ns": i_tau_s,
            "free_t0_ns": i_t0,
            "free_Af": i_amp[0, 0],
            "free_As": i_amp[0, 1],
        })
    free_residual = np.concatenate(individual_residuals)
    free_parameter_count = sum(result.x.size for result, _ in individual_fits)
    free_rss, free_aic, free_bic = information_criteria(free_residual, free_parameter_count)

    fit_frame = summaries.merge(pd.DataFrame(rows), on="group")
    fit_frame["shared_tau_r_ns"] = tau_r
    fit_frame["shared_tau_f_ns"] = tau_f
    fit_frame["shared_tau_s_ns"] = tau_s
    fit_frame["shared_t0_ns"] = t0
    fit_frame.to_csv(prefix.with_name(prefix.name + "_groups.csv"), index=False)
    comparison = pd.DataFrame([
        {"model": "shared_taus", "rss": shared_rss, "aic": shared_aic, "bic": shared_bic,
         "n_parameters": shared_result.x.size},
        {"model": "free_taus", "rss": free_rss, "aic": free_aic, "bic": free_bic,
         "n_parameters": free_parameter_count},
    ])
    comparison.to_csv(prefix.with_name(prefix.name + "_model_comparison.csv"), index=False)

    with PdfPages(prefix.with_suffix(".pdf")) as pdf:
        ncols = 3
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False,
                                 constrained_layout=True)
        for axis, name, data, model in zip(axes.ravel(), names, waveforms, shared_models):
            axis.plot(time_fit, data, color="black", lw=1, label="median")
            axis.plot(time_fit, model, color="C3", lw=1.5, label="shared taus")
            axis.set_title(name); axis.grid(alpha=0.25); axis.legend(fontsize=8)
            axis.set_xlabel("time [ns]"); axis.set_ylabel("|IQ| - noise floor")
        for axis in axes.ravel()[len(names):]: axis.set_visible(False)
        fig.suptitle(f"Shared fit: tau_r={tau_r:.2f}, tau_f={tau_f:.2f}, tau_s={tau_s:.2f} ns")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows), squeeze=False,
                                 constrained_layout=True)
        for axis, name, data, model in zip(axes.ravel(), names, waveforms, shared_models):
            scale = max(float(np.max(data)), 1e-12)
            axis.plot(time_fit, (data - model) / scale, color="C0", lw=1)
            axis.axhline(0, color="black", lw=0.7)
            axis.set_title(name); axis.grid(alpha=0.25)
            axis.set_xlabel("time [ns]"); axis.set_ylabel("residual / peak")
        for axis in axes.ravel()[len(names):]: axis.set_visible(False)
        fig.suptitle("Residuals of shared-time-constant model")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        axes[0].scatter(fit_frame["amp_median"], fit_frame["shared_Af_fraction"],
                        c=fit_frame["tau_d_eff_median"], cmap="viridis", s=70)
        for row in fit_frame.itertuples():
            axes[0].annotate(row.group, (row.amp_median, row.shared_Af_fraction), fontsize=7)
        axes[0].set_xscale("log"); axes[0].set_xlabel("median amplitude")
        axes[0].set_ylabel("Af / (Af + As)"); axes[0].grid(alpha=0.25)
        axes[1].bar(comparison["model"], comparison["bic"] - comparison["bic"].min())
        axes[1].set_ylabel("Delta BIC"); axes[1].set_title("Lower is preferred")
        pdf.savefig(fig); plt.close(fig)
    print(comparison.to_string(index=False))
    print(f"saved {prefix.with_suffix('.pdf')}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=Path("alphaDC_common_tau_fit"))
    parser.add_argument("--amp-bins", type=int, default=3)
    parser.add_argument("--shape-bins", type=int, default=3)
    parser.add_argument("--max-events-per-group", type=int, default=500)
    parser.add_argument("--min-amp", type=float, default=0.002)
    parser.add_argument("--peak-min-ns", type=float, default=-30.0)
    parser.add_argument("--peak-max-ns", type=float, default=500.0)
    parser.add_argument("--fit-start-ns", type=float, default=-50.0)
    parser.add_argument("--fit-stop-ns", type=float, default=1600.0)
    parser.add_argument("--fit-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv or args.npz.with_name(args.npz.stem + "_amp_tau_eff.csv")
    frame = pd.read_csv(csv_path)
    selected = select_groups(
        frame, args.amp_bins, args.shape_bins, args.min_amp,
        args.peak_min_ns, args.peak_max_ns,
    )
    temporary = args.cache_dir is None
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="alpha_common_tau_"))
    try:
        ch0, ch1, npts, sample_rate, ref_position = extract_arrays(args.npz, cache_dir)
        if len(frame) != ch0.shape[0]:
            raise RuntimeError(f"CSV rows ({len(frame)}) != waveform events ({ch0.shape[0]})")
        time_ns = (
            np.arange(npts, dtype=float) - npts * ref_position / 100.0
        ) / sample_rate * 1e9
        medians, summaries = group_medians(
            ch0, ch1, selected, time_ns, ref_position,
            args.max_events_per_group, args.seed,
        )
        fit_mask = (time_ns >= args.fit_start_ns) & (time_ns <= args.fit_stop_ns)
        fit_indices = np.flatnonzero(fit_mask)[::args.fit_stride]
        time_fit = time_ns[fit_indices]
        names = sorted(medians)
        waveforms = np.asarray([medians[name][fit_indices] for name in names], dtype=float)
        shared_result, shared_fit = fit_shared(time_fit, waveforms)
        individual_fits = [fit_individual(time_fit, waveform) for waveform in waveforms]
        create_outputs(
            args.output_prefix, time_fit, names, waveforms, summaries,
            shared_result, shared_fit, individual_fits,
        )
    finally:
        if temporary and not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
