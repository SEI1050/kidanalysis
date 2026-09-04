"""Learn a two-component shared-time-constant model on L laser positions.

The L-position median projected waveforms are fitted jointly with shared
tau_r, tau_fast, tau_slow and t0, while the two non-negative amplitudes vary
with position.  The fitted L time constants are then frozen and applied to C.
Leave-one-L-position-out fits provide an honest reference error for an unseen
L position.
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
from scipy.optimize import least_squares, nnls


# Third timestamp is the -2 dBm run in Loc_wave.py.
L_RUNS = {
    6.8: "134954", 6.7: "134832", 6.5: "134710",
    6.3: "134447", 6.1: "134328", 6.0: "134208",
}
C_RUNS = {6.8: "130955", 6.7: "131115", 6.3: "131249", 6.0: "131518"}


def find_file(root: Path, timestamp: str) -> Path:
    matches = sorted((root / f"data_0825_{timestamp}").glob("wf_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No wf_*.npz in data_0825_{timestamp}")
    return matches[0]


def median_projected_wave(path: Path):
    with np.load(path, allow_pickle=False) as data:
        ch0 = np.asarray(data["ch0"], dtype=np.float32)
        ch1 = np.asarray(data["ch1"], dtype=np.float32)
        npts = int(data["npts"])
        sample_rate = float(data["sample_rate"])
        ref_position = float(data["ref_position"])
    time_ns = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
    pre_end = max(2, int((ref_position - 10.0) / 100.0 * npts))
    dev_i = ch0 - np.mean(ch0[:, :pre_end], axis=1, keepdims=True)
    dev_q = ch1 - np.mean(ch1[:, :pre_end], axis=1, keepdims=True)
    mean_i, mean_q = np.mean(dev_i, axis=0), np.mean(dev_q, axis=0)
    peak = int(np.argmax(np.hypot(mean_i, mean_q)))
    norm = float(np.hypot(mean_i[peak], mean_q[peak]))
    direction = np.array([mean_i[peak], mean_q[peak]]) / max(norm, 1e-15)
    projected = dev_i * direction[0] + dev_q * direction[1]
    wave = np.median(projected, axis=0)
    wave -= np.median(wave[:pre_end])
    return time_ns, wave, ch0.shape[0]


def pulse_components(time_ns, tau_r, tau_f, tau_s, t0):
    u = np.maximum(time_ns - t0, 0.0)
    gate = time_ns >= t0
    fast = np.where(gate, np.exp(-u / tau_f) - np.exp(-u / tau_r), 0.0)
    slow = np.where(gate, np.exp(-u / tau_s) - np.exp(-u / tau_r), 0.0)
    return np.column_stack((fast, slow))


def unpack(parameters, n_wave):
    tau_r = np.exp(parameters[0])
    tau_f = tau_r + np.exp(parameters[1])
    tau_s = tau_f + np.exp(parameters[2])
    t0 = parameters[3]
    amplitudes = np.exp(parameters[4:].reshape(n_wave, 2))
    return tau_r, tau_f, tau_s, t0, amplitudes


def fit_shared(time_ns, waves):
    peaks = np.maximum(np.max(waves, axis=1), 1e-12)
    n_wave = len(waves)
    initial_amp = np.column_stack((0.6 * peaks, 0.4 * peaks))
    initial = np.r_[np.log([8.0, 72.0, 620.0]), 0.0,
                    np.log(np.maximum(initial_amp, 1e-12)).ravel()]

    def residual(parameters):
        tr, tf, ts, t0, amps = unpack(parameters, n_wave)
        design = pulse_components(time_ns, tr, tf, ts, t0)
        models = amps @ design.T
        return ((models - waves) / peaks[:, None]).ravel()

    lower = np.r_[np.log([0.2, 0.2, 0.2]), -100.0, np.full(2 * n_wave, -30.0)]
    upper = np.r_[np.log([500.0, 5000.0, 20000.0]), 100.0, np.full(2 * n_wave, 5.0)]
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=7000)
    return result, unpack(result.x, n_wave)


def fixed_fit(time_ns, wave, constants):
    tr, tf, ts, t0 = constants
    design = pulse_components(time_ns, tr, tf, ts, t0)
    amps, _ = nnls(design, wave)
    return amps, design @ amps


def metrics(time_ns, wave, model):
    peak = max(float(np.max(wave)), 1e-12)
    residual = wave - model
    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((wave - np.mean(wave)) ** 2))
    early = (time_ns >= 0.0) & (time_ns <= 300.0)
    late = (time_ns > 300.0) & (time_ns <= 1600.0)
    data_peak_index = int(np.argmax(wave))
    return {
        "nrmse": float(np.sqrt(np.mean(residual ** 2)) / peak),
        "early_nrmse_0_300ns": float(np.sqrt(np.mean(residual[early] ** 2)) / peak),
        "late_nrmse_300_1600ns": float(np.sqrt(np.mean(residual[late] ** 2)) / peak),
        "max_abs_residual_over_peak": float(np.max(np.abs(residual)) / peak),
        "residual_at_data_peak_over_peak": float(residual[data_peak_index] / peak),
        "r2": 1.0 - rss / max(tss, 1e-30),
        "rss": rss,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("/Volumes/NO NAME/data/20260825"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("laser_L_shared_tau_C_test"))
    parser.add_argument("--fit-start-ns", type=float, default=-50.0)
    parser.add_argument("--fit-stop-ns", type=float, default=1600.0)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    records, raw = [], {}
    for region, runs, x in (("L", L_RUNS, 5.1), ("C", C_RUNS, 4.6)):
        for z, timestamp in runs.items():
            path = find_file(args.root, timestamp)
            time_ns, wave, n_events = median_projected_wave(path)
            raw[(region, z)] = (time_ns, wave)
            records.append({"region": region, "x_mm": x, "z_mm": z,
                            "timestamp": timestamp, "file": str(path),
                            "n_events": n_events})
            print(f"loaded {region} z={z:.1f}: {n_events} events", flush=True)

    reference_time = next(iter(raw.values()))[0]
    mask = (reference_time >= args.fit_start_ns) & (reference_time <= args.fit_stop_ns)
    index = np.flatnonzero(mask)[::args.stride]
    fit_time = reference_time[index]
    l_keys = sorted((key for key in raw if key[0] == "L"), key=lambda key: -key[1])
    c_keys = sorted((key for key in raw if key[0] == "C"), key=lambda key: -key[1])
    l_waves = np.asarray([raw[key][1][index] for key in l_keys])
    c_waves = np.asarray([raw[key][1][index] for key in c_keys])

    result, fitted = fit_shared(fit_time, l_waves)
    tr, tf, ts, t0, l_amplitudes = fitted
    constants = (tr, tf, ts, t0)
    design = pulse_components(fit_time, *constants)
    l_models = l_amplitudes @ design.T

    rows = []
    for key, wave, model, amps in zip(l_keys, l_waves, l_models, l_amplitudes):
        rows.append({"region": "L", "z_mm": key[1], "fit_kind": "all_L_training",
                     "A_fast": amps[0], "A_slow": amps[1],
                     "fast_fraction": amps[0] / max(amps.sum(), 1e-30),
                     **metrics(fit_time, wave, model)})

    # Honest reference: predict each held-out L shape using constants learned
    # from all other L positions, fitting only its two amplitudes.
    loo_errors = {}
    for held in range(len(l_keys)):
        train = np.delete(l_waves, held, axis=0)
        _, loo_fit = fit_shared(fit_time, train)
        loo_constants = loo_fit[:4]
        amps, model = fixed_fit(fit_time, l_waves[held], loo_constants)
        item = metrics(fit_time, l_waves[held], model)
        loo_errors[l_keys[held]] = (model, item)
        rows.append({"region": "L", "z_mm": l_keys[held][1],
                     "fit_kind": "L_leave_one_out", "A_fast": amps[0],
                     "A_slow": amps[1],
                     "fast_fraction": amps[0] / max(amps.sum(), 1e-30), **item})

    c_models = []
    for key, wave in zip(c_keys, c_waves):
        amps, model = fixed_fit(fit_time, wave, constants)
        c_models.append(model)
        rows.append({"region": "C", "z_mm": key[1], "fit_kind": "fixed_L_taus",
                     "A_fast": amps[0], "A_slow": amps[1],
                     "fast_fraction": amps[0] / max(amps.sum(), 1e-30),
                     **metrics(fit_time, wave, model)})

    frame = pd.DataFrame(rows)
    l_reference = frame.query("fit_kind == 'L_leave_one_out'")["nrmse"]
    median_ref = float(l_reference.median())
    mad_ref = float(np.median(np.abs(l_reference - median_ref)))
    scale_ref = max(1.4826 * mad_ref, 1e-12)
    frame["nrmse_over_L_LOO_median"] = frame["nrmse"] / median_ref
    frame["nrmse_robust_z_vs_L_LOO"] = (frame["nrmse"] - median_ref) / scale_ref
    frame["shared_tau_r_ns"] = tr; frame["shared_tau_fast_ns"] = tf
    frame["shared_tau_slow_ns"] = ts; frame["shared_t0_ns"] = t0
    frame.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    pd.DataFrame(records).to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_inputs.csv"), index=False)

    with PdfPages(args.output_prefix.with_suffix(".pdf")) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for axis, key, wave, model in zip(axes.ravel(), l_keys, l_waves, l_models):
            axis.plot(fit_time, wave, color="black", lw=1, label="L median")
            axis.plot(fit_time, model, color="C0", lw=1.5, label="shared taus")
            value = metrics(fit_time, wave, model)["nrmse"]
            axis.set_title(f"L x=5.1, z={key[1]:.1f} mm; NRMSE={value:.3f}")
            axis.grid(alpha=.25); axis.legend(fontsize=8)
        for axis in axes.ravel(): axis.set_xlabel("time [ns]")
        fig.suptitle(f"L shared fit: tau_r={tr:.1f}, tau_f={tf:.1f}, tau_s={ts:.1f} ns")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, key, wave, model in zip(axes.ravel(), c_keys, c_waves, c_models):
            item = metrics(fit_time, wave, model)
            axis.plot(fit_time, wave, color="black", lw=1, label="C median")
            axis.plot(fit_time, model, color="C3", lw=1.5, label="L taus fixed")
            axis.set_title(f"C x=4.6, z={key[1]:.1f}; NRMSE={item['nrmse']:.3f}")
            axis.grid(alpha=.25); axis.legend(fontsize=8); axis.set_xlabel("time [ns]")
        fig.suptitle("Extrapolation of L-derived model to C")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, key, wave, model in zip(axes.ravel(), c_keys, c_waves, c_models):
            peak = max(np.max(wave), 1e-12)
            axis.plot(fit_time, (wave - model) / peak, color="C3", lw=1)
            axis.axhline(0, color="black", lw=.7)
            axis.set_title(f"C z={key[1]:.1f} residual / peak")
            axis.grid(alpha=.25); axis.set_xlabel("time [ns]")
        fig.suptitle("Structured residuals: C data minus L-derived model")
        pdf.savefig(fig); plt.close(fig)

        fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        loo = frame.query("fit_kind == 'L_leave_one_out'").sort_values("z_mm")
        cfit = frame.query("fit_kind == 'fixed_L_taus'").sort_values("z_mm")
        axis.scatter([f"L {z:.1f}" for z in loo.z_mm], loo.nrmse, s=65, label="unseen L")
        axis.scatter([f"C {z:.1f}" for z in cfit.z_mm], cfit.nrmse, s=65, label="C with L model")
        axis.axhline(median_ref, color="C0", ls="--", label="L LOO median")
        axis.set_ylabel("NRMSE / waveform peak"); axis.grid(alpha=.25); axis.legend()
        axis.tick_params(axis="x", rotation=35)
        pdf.savefig(fig); plt.close(fig)

    print(f"shared L taus: tau_r={tr:.5g}, tau_f={tf:.5g}, tau_s={ts:.5g}, t0={t0:.5g} ns")
    print(frame[["region", "z_mm", "fit_kind", "nrmse", "r2",
                 "nrmse_over_L_LOO_median", "nrmse_robust_z_vs_L_LOO"]].to_string(index=False))
    print(f"saved {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
