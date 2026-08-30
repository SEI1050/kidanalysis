"""Geometric IQ calibration for one IQ scan and many fixed-frequency waveforms.

The correction is applied in this order:
    1. cancel the frequency-dependent phase delay (tau)
    2. rotate by alpha
    3. normalize by a
    4. rotate around (1, 0) by phi

The script saves one set of step-by-step figures for the IQ scan and one
waveform figure/result for every NPZ in WAVEFORM_DIR.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

IQ_SCAN = Path("/Volumes/NO NAME/data/20260714/iq_scan_f_reso_5.267GHz.npz")
WAVEFORM_DIR = Path("/Volumes/NO NAME/data/20260714/5.267GHz_trig_ch0_2.0mV")
OUTPUT_DIR = Path("/Users/kubokosei/Documents/kid/kidanalysis/gakkai/iq_correct_output")
READOUT_FREQUENCY_HZ = 5.267e9

Q_SIGN = 1.0
N_EDGE_POINTS = 15
CIRCLE_HALF_WIDTH_POINTS = 15
MAX_EVENTS_TO_PLOT = 200
PLOT_SAMPLE_STRIDE = 5

# Event groups to draw. Use None for an open lower or upper bound.
# Each group creates one Step 5 PNG and one corresponding PDF page.
EVENT_CONDITION_SETS = [
	{
		"name": "all",
		"amp_min": 0.0175,
		"amp_max": 0.025,
		"tau_r_eff_min": 0,
		"tau_r_eff_max": 15,
	},
    {
		"name": "all",
		"amp_min": 0,
		"amp_max": 0.0050,
		"tau_r_eff_min": 40,
		"tau_r_eff_max": 100,
	},
]

# The pulse analysis used for event selection follows plot_eventwave.py.
EVENT_THRESHOLD_FRACTION = 0.10


@dataclass
class Calibration:
	"""Parameters determined from the IQ scan."""

	tau_s: float
	alpha_rad: float
	amplitude_a: float
	phi_rad: float
	circle_center: complex
	circle_radius: float
	resonance_index: int
	resonance_frequency_hz: float
	point_p: complex


def frequency_to_hz(frequency: np.ndarray) -> np.ndarray:
	"""Convert a frequency column written in GHz, MHz, or Hz to Hz."""
	frequency = np.asarray(frequency, dtype=float)
	typical = float(np.nanmedian(np.abs(frequency)))
	if typical < 100.0:
		return frequency * 1e9
	if typical < 1e7:
		return frequency * 1e6
	return frequency


def load_iq_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
	with np.load(path, allow_pickle=False) as data:
		if "dd" not in data:
			raise KeyError(f"'dd' was not found in {path}")
		dd = np.asarray(data["dd"], dtype=float)
	if dd.ndim != 2 or dd.shape[1] < 3:
		raise ValueError(f"dd must have shape (N, >=3), got {dd.shape}")
	frequency_hz = frequency_to_hz(dd[:, 0])
	z_scan = dd[:, 1] + 1j * Q_SIGN * dd[:, 2]
	order = np.argsort(frequency_hz)
	return frequency_hz[order], z_scan[order]


def find_key(keys: list[str], candidates: tuple[str, ...]) -> str:
	by_lower = {key.lower(): key for key in keys}
	for candidate in candidates:
		if candidate.lower() in by_lower:
			return by_lower[candidate.lower()]
	raise KeyError(f"Could not find {candidates}; keys={keys}")


def load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
	# Some DAQ files contain object metadata, so allow_pickle=True is required.
	with np.load(path, allow_pickle=True) as data:
		keys = list(data.keys())
		ch0 = np.asarray(data[find_key(keys, ("ch0", "channel0", "i"))], dtype=float)
		ch1 = np.asarray(data[find_key(keys, ("ch1", "channel1", "q"))], dtype=float)
		metadata = {}
		for key in ("npts", "sample_rate", "ref_position", "daq_rate"):
			if key in data:
				value = np.asarray(data[key])
				if value.dtype != object and value.size <= 100:
					metadata[key] = value
	if ch0.shape != ch1.shape:
		raise ValueError(f"ch0 and ch1 shapes differ: {ch0.shape}, {ch1.shape}")
	if ch0.ndim == 1:
		ch0, ch1 = ch0[None, :], ch1[None, :]
	if ch0.ndim != 2:
		raise ValueError(f"waveform must be 1D or 2D, got {ch0.shape}")
	if ch0.shape[0] > ch0.shape[1] and ch0.shape[1] <= 2000:
		ch0, ch1 = ch0.T, ch1.T
	return ch0, ch1, metadata


def fit_tau(frequency_hz: np.ndarray, z_scan: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
	"""Fit angle(z) = b - 2*pi*tau*f using only both scan edges."""
	n = frequency_hz.size
	if n < 2 * N_EDGE_POINTS:
		raise ValueError("The IQ scan does not contain enough points for the edge fit")
	low = np.unwrap(np.angle(z_scan[:N_EDGE_POINTS]))
	high = np.unwrap(np.angle(z_scan[-N_EDGE_POINTS:]))
	f_edge = np.r_[frequency_hz[:N_EDGE_POINTS], frequency_hz[-N_EDGE_POINTS:]]
	f_ref = np.mean(f_edge)
	x = f_edge - f_ref
	best = None
	for branch in range(-5, 6):
		phase = np.r_[low, high + 2.0 * np.pi * branch]
		slope, intercept = np.linalg.lstsq(np.c_[x, np.ones_like(x)], phase, rcond=None)[0]
		residual = phase - (slope * x + intercept)
		result = (float(np.sum(residual ** 2)), float(slope), phase)
		if best is None or result[0] < best[0]:
			best = result
	assert best is not None
	tau_s = -best[1] / (2.0 * np.pi)
	phase_fit = best[1] * (frequency_hz - f_ref) + np.mean(best[2])
	return tau_s, phase_fit, best[2]


def fit_circle(z: np.ndarray) -> tuple[complex, float, float]:
	x, y = np.real(z), np.imag(z)
	d, e, f0 = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], -(x ** 2 + y ** 2), rcond=None)[0]
	center = complex(-d / 2.0, -e / 2.0)
	radius = float(np.sqrt(max(center.real ** 2 + center.imag ** 2 - f0, 0.0)))
	rms = float(np.sqrt(np.mean((np.abs(z - center) - radius) ** 2)))
	return center, radius, rms


def calculate_calibration(frequency_hz: np.ndarray, z_scan: np.ndarray) -> tuple[Calibration, dict]:
	tau_s, phase_fit, edge_phase = fit_tau(frequency_hz, z_scan)
	z_tau = z_scan * np.exp(1j * 2.0 * np.pi * tau_s * frequency_hz)
	resonance_index = int(np.argmin(np.abs(frequency_hz - READOUT_FREQUENCY_HZ)))
	i0 = max(0, resonance_index - CIRCLE_HALF_WIDTH_POINTS)
	i1 = min(frequency_hz.size, resonance_index + CIRCLE_HALF_WIDTH_POINTS + 1)
	center, radius, radial_rms = fit_circle(z_tau[i0:i1])
	point_p = 2.0 * center - z_tau[resonance_index]
	alpha = float(np.angle(point_p))
	point_after_alpha = point_p * np.exp(-1j * alpha)
	a = float(np.abs(point_after_alpha))
	if a == 0.0:
		raise ZeroDivisionError("The calculated amplitude normalization a is zero")
	center_after_a = center * np.exp(-1j * alpha) / a
	phi = float(np.angle(1.0 - center_after_a))
	return Calibration(
		tau_s=tau_s,
		alpha_rad=alpha,
		amplitude_a=a,
		phi_rad=phi,
		circle_center=center,
		circle_radius=radius,
		resonance_index=resonance_index,
		resonance_frequency_hz=float(frequency_hz[resonance_index]),
		point_p=point_p,
	), {
		"phase_fit": phase_fit,
		"edge_phase": edge_phase,
		"fit_start": i0,
		"fit_stop": i1,
		"radial_rms": radial_rms,
	}


def apply_correction(z: np.ndarray, frequency_hz: float | np.ndarray, calibration: Calibration) -> np.ndarray:
	z_tau = z * np.exp(1j * 2.0 * np.pi * calibration.tau_s * np.asarray(frequency_hz))
	z_alpha = z_tau * np.exp(-1j * calibration.alpha_rad)
	z_a = z_alpha / calibration.amplitude_a
	return 1.0 + (z_a - 1.0) * np.exp(-1j * calibration.phi_rad)


def plot_iq(ax, z: np.ndarray, title: str, extra: list[tuple[complex, str, str]] = []):
	ax.plot(z.real, z.imag, "o-", ms=3, lw=0.8, color="0.45")
	ax.scatter(z.real, z.imag, c=np.arange(z.size), cmap="viridis", s=18)
	for point, marker, label in extra:
		ax.scatter(point.real, point.imag, marker=marker, s=100, label=label)
	ax.set_title(title)
	ax.set_xlabel("I")
	ax.set_ylabel("Q")
	ax.grid(alpha=0.3)
	ax.set_aspect("equal", adjustable="box")
	if extra:
		ax.legend(fontsize=8, loc="best")


def plot_circle(ax, center: complex, radius: float, label: str = "fitted circle"):
	theta = np.linspace(0.0, 2.0 * np.pi, 400)
	circle = center + radius * np.exp(1j * theta)
	ax.plot(circle.real, circle.imag, color="tab:orange", lw=1.8, label=label)


def linear_crossing(time_ns: np.ndarray, signal: np.ndarray, level: float, start: int, stop: int, direction: str) -> float:
	"""Return the linearly interpolated crossing time for one pulse."""
	if direction == "up_from_peak":
		indices = range(start - 1, stop - 1, -1)
		condition = lambda left, right: left < level <= right
	else:
		indices = range(start, stop)
		condition = lambda left, right: left >= level > right

	for index in indices:
		if condition(signal[index], signal[index + 1]):
			if signal[index + 1] == signal[index]:
				return 0.5 * (time_ns[index] + time_ns[index + 1])
			return time_ns[index] + (level - signal[index]) * (
				time_ns[index + 1] - time_ns[index]
			) / (signal[index + 1] - signal[index])
	return np.nan


def integral_between(time_ns: np.ndarray, signal: np.ndarray, first_time: float, last_time: float) -> float:
	"""Integrate a signal between two interpolated times."""
	if not np.isfinite(first_time) or not np.isfinite(last_time) or last_time <= first_time:
		return np.nan
	inside = (time_ns > first_time) & (time_ns < last_time)
	integration_time = np.concatenate(([first_time], time_ns[inside], [last_time]))
	integration_signal = np.interp(integration_time, time_ns, signal)
	return float(np.trapezoid(integration_signal, integration_time))


def analyze_event_parameters(ch0_waveform: np.ndarray, ch1_waveform: np.ndarray, metadata: dict) -> dict[str, float]:
	"""Calculate amp and tau_r_eff using the plot_eventwave.py definition."""
	npts = int(np.asarray(metadata["npts"]).item())
	ref_position = float(np.asarray(metadata["ref_position"]).item())
	sample_rate = float(np.asarray(metadata["sample_rate"]).item())
	time_ns = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
	pedestal_end = max(1, min(npts, int((ref_position - 10.0) / 100.0 * npts)))
	signal = np.hypot(
		ch0_waveform[:npts] - np.mean(ch0_waveform[:pedestal_end]),
		ch1_waveform[:npts] - np.mean(ch1_waveform[:pedestal_end]),
	)
	peak_index = int(np.argmax(signal))
	amp = float(signal[peak_index])
	level = EVENT_THRESHOLD_FRACTION * amp
	t10_left = linear_crossing(time_ns, signal, level, peak_index, 0, "up_from_peak")
	left_integral = integral_between(time_ns, signal, t10_left, time_ns[peak_index])
	tau_r_eff = left_integral / amp if np.isfinite(left_integral) and amp != 0.0 else np.nan
	return {"amp": amp, "tau_r_eff": float(tau_r_eff)}


def condition_mask(parameters: list[dict[str, float]], condition: dict) -> np.ndarray:
	"""Return events satisfying all configured amp/tau inequalities."""
	amp = np.asarray([item["amp"] for item in parameters], dtype=float)
	tau = np.asarray([item["tau_r_eff"] for item in parameters], dtype=float)
	mask = np.isfinite(amp) & np.isfinite(tau)
	if condition["amp_min"] is not None:
		mask &= amp >= condition["amp_min"]
	if condition["amp_max"] is not None:
		mask &= amp <= condition["amp_max"]
	if condition["tau_r_eff_min"] is not None:
		mask &= tau >= condition["tau_r_eff_min"]
	if condition["tau_r_eff_max"] is not None:
		mask &= tau <= condition["tau_r_eff_max"]
	return mask


def save_scan_figures(frequency_hz: np.ndarray, z_raw: np.ndarray, calibration: Calibration, details: dict, pdf: PdfPages):
	tau = calibration.tau_s
	z_tau = apply_correction(z_raw, frequency_hz, Calibration(tau, 0, 1, 0, 0j, 1, 0, 0, 0j))
	z_alpha = z_tau * np.exp(-1j * calibration.alpha_rad)
	z_a = z_alpha / calibration.amplitude_a
	z_final = apply_correction(z_raw, frequency_hz, calibration)

	fig, ax = plt.subplots(figsize=(7, 5))
	ax.plot(frequency_hz / 1e9, np.unwrap(np.angle(z_raw)), "o-", label="raw phase")
	ax.plot(frequency_hz / 1e9, details["phase_fit"], label="edge fit")
	ax.set(xlabel="frequency [GHz]", ylabel="phase [rad]", title=f"Step 1: tau fit ({tau * 1e9:.5f} ns)")
	ax.grid(alpha=0.3); ax.legend()
	fig.tight_layout(); pdf.savefig(fig); fig.savefig(OUTPUT_DIR / "calibration_step1_tau_fit.png", dpi=180); plt.close(fig)

	fig, ax = plt.subplots(figsize=(7, 5))
	plot_iq(ax, z_tau, "Step 2: after tau correction", [(calibration.circle_center, "X", "circle center"), (calibration.point_p, "*", "P")])
	plot_circle(ax, calibration.circle_center, calibration.circle_radius)
	ax.legend(fontsize=8, loc="best")
	fig.tight_layout(); pdf.savefig(fig); fig.savefig(OUTPUT_DIR / "calibration_step2_circle_fit.png", dpi=180); plt.close(fig)

	fig, ax = plt.subplots(1, 3, figsize=(16, 5))
	plot_iq(ax[0], z_tau, "Before alpha")
	plot_iq(ax[1], z_alpha, f"Step 3: alpha = {calibration.alpha_rad:.5f} rad", [(calibration.point_p * np.exp(-1j * calibration.alpha_rad), "*", "P -> x axis")])
	plot_circle(ax[1], calibration.circle_center * np.exp(-1j * calibration.alpha_rad), calibration.circle_radius)
	plot_iq(ax[2], z_a, f"Step 3: a = {calibration.amplitude_a:.6g}", [(1 + 0j, "+", "(1, 0)")])
	center_a = calibration.circle_center * np.exp(-1j * calibration.alpha_rad) / calibration.amplitude_a
	plot_circle(ax[2], center_a, calibration.circle_radius / calibration.amplitude_a)
	fig.tight_layout(); pdf.savefig(fig); fig.savefig(OUTPUT_DIR / "calibration_step3_alpha_a_phi.png", dpi=180); plt.close(fig)

	fig, ax = plt.subplots(1, 2, figsize=(12, 5))
	plot_iq(ax[0], z_final, f"Step 4: final scan (phi = {calibration.phi_rad:.5f} rad)", [(1 + 0j, "+", "(1, 0)")])
	center_final = 1.0 + (center_a - 1.0) * np.exp(-1j * calibration.phi_rad)
	plot_circle(ax[0], center_final, calibration.circle_radius / calibration.amplitude_a)
	ax[0].legend(fontsize=8, loc="best")
	ax[1].plot(frequency_hz / 1e9, np.abs(z_final), "o-", label="final amplitude")
	ax[1].plot(frequency_hz / 1e9, np.unwrap(np.angle(z_final)), "o-", label="final phase")
	ax[1].set(xlabel="frequency [GHz]", title="Final amplitude and phase"); ax[1].grid(alpha=0.3); ax[1].legend()
	fig.tight_layout(); pdf.savefig(fig); fig.savefig(OUTPUT_DIR / "calibration_step4_final_scan_amp_phase.png", dpi=180); plt.close(fig)

	return z_final


def format_bound(value: float | None, positive: bool) -> str:
	"""Format an open or closed condition bound for figure titles."""
	if value is None:
		return "-inf" if positive else "+inf"
	return f"{value:g}"


def condition_description(condition: dict) -> str:
	return (
		f"{condition['name']}: "
		f"{format_bound(condition['amp_min'], True)} <= amp <= "
		f"{format_bound(condition['amp_max'], False)}, "
		f"{format_bound(condition['tau_r_eff_min'], True)} <= tau_r_eff <= "
		f"{format_bound(condition['tau_r_eff_max'], False)} ns"
	)


def save_waveform_result(path: Path, calibration: Calibration, pdf: PdfPages):
	ch0, ch1, metadata = load_waveform(path)
	for key in ("npts", "sample_rate", "ref_position"):
		if key not in metadata:
			raise KeyError(f"{key!r} is required in waveform file {path}")
	npts = min(int(np.asarray(metadata["npts"]).item()), ch0.shape[1])
	ref_position = float(np.asarray(metadata["ref_position"]).item())
	sample_rate = float(np.asarray(metadata["sample_rate"]).item())
	time_ns = (np.arange(npts) - npts * ref_position / 100.0) / sample_rate * 1e9
	z_raw = ch0 + 1j * Q_SIGN * ch1
	z_tau = apply_correction(z_raw, READOUT_FREQUENCY_HZ, Calibration(calibration.tau_s, 0, 1, 0, 0j, 1, 0, 0, 0j))
	z_alpha_a = z_tau * np.exp(-1j * calibration.alpha_rad) / calibration.amplitude_a
	z_final = apply_correction(z_raw, READOUT_FREQUENCY_HZ, calibration)

	parameters = [
		analyze_event_parameters(ch0[event], ch1[event], metadata)
		for event in range(z_raw.shape[0])
	]
	parameter_arrays = {
		"amp": np.asarray([item["amp"] for item in parameters]),
		"tau_r_eff": np.asarray([item["tau_r_eff"] for item in parameters]),
	}
	parameter_path = OUTPUT_DIR / f"{path.stem}_event_parameters.npz"
	np.savez_compressed(parameter_path, event=np.arange(z_raw.shape[0]), **parameter_arrays)

	for condition in EVENT_CONDITION_SETS:
		event_mask = condition_mask(parameters, condition)
		matching_events = np.flatnonzero(event_mask)
		if matching_events.size > MAX_EVENTS_TO_PLOT:
			matching_events = np.linspace(
				matching_events[0],
				matching_events[-1],
				MAX_EVENTS_TO_PLOT,
				dtype=int,
			)

		fig, ax = plt.subplots(2, 2, figsize=(14, 11))
		for axis, values, title in zip(
			ax.ravel(),
			(z_raw, z_tau, z_alpha_a, z_final),
			("Raw", "After tau", "After alpha + a", "Final corrected"),
		):
			sample_indices = np.arange(0, npts, max(1, PLOT_SAMPLE_STRIDE))
			norm = Normalize(vmin=time_ns[sample_indices[0]], vmax=time_ns[sample_indices[-1]])
			for event in matching_events:
				trajectory = values[event, sample_indices]
				segments = np.stack(
					[
						 np.column_stack((trajectory[:-1].real, trajectory[:-1].imag)),
						 np.column_stack((trajectory[1:].real, trajectory[1:].imag)),
					],
					axis=1,
				)
				line_collection = LineCollection(
					segments,
					cmap="plasma",
					norm=norm,
					linewidths=0.9,
					alpha=0.45,
				)
				line_collection.set_array(time_ns[sample_indices[:-1]])
				axis.add_collection(line_collection)
			axis.set_title(title)
			axis.set_xlabel("I")
			axis.set_ylabel("Q")
			axis.grid(alpha=0.3)
			axis.set_aspect("equal", adjustable="box")
		fig.suptitle(
			f"Step 5: {path.name}\n"
			f"{condition_description(condition)}\n"
			f"matching events: {event_mask.sum()}/{z_raw.shape[0]} "
			f"(drawn: {matching_events.size})"
		)
		fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.88))
		cax = fig.add_axes((0.89, 0.22, 0.025, 0.56))
		scalar_mappable = ScalarMappable(norm=norm, cmap="plasma")
		scalar_mappable.set_array(time_ns[sample_indices])
		colorbar = fig.colorbar(scalar_mappable, cax=cax)
		colorbar.set_label("time [ns]", rotation=90, labelpad=10)
		pdf.savefig(fig)
		condition_name = str(condition["name"])
		fig.savefig(
			OUTPUT_DIR / f"{path.stem}_calibration_step5_waveform_before_after_{condition_name}.png",
			dpi=180,
		)
		plt.close(fig)

		# Add a slide-friendly page containing only the final corrected IQ plot.
		final_fig, final_axis = plt.subplots(figsize=(9, 8))
		for event in matching_events:
			trajectory = z_final[event, sample_indices]
			segments = np.stack(
				[
					 np.column_stack((trajectory[:-1].real, trajectory[:-1].imag)),
					 np.column_stack((trajectory[1:].real, trajectory[1:].imag)),
				],
				axis=1,
			)
			line_collection = LineCollection(
				segments,
				cmap="plasma",
				norm=norm,
				linewidths=1.1,
				alpha=0.5,
			)
			line_collection.set_array(time_ns[sample_indices[:-1]])
			final_axis.add_collection(line_collection)
		final_axis.autoscale_view()
		final_axis.set_title(
			f"Final corrected IQ\n"
			f"{path.name}, {condition_name}\n"
			f"matching events: {event_mask.sum()}/{z_raw.shape[0]}"
		)
		final_axis.set_xlabel("I_final")
		final_axis.set_ylabel("Q_final")
		final_axis.grid(alpha=0.3)
		final_axis.set_aspect("equal", adjustable="box")
		final_fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.92))
		final_cax = final_fig.add_axes((0.88, 0.2, 0.035, 0.6))
		final_mappable = ScalarMappable(norm=norm, cmap="plasma")
		final_mappable.set_array(time_ns[sample_indices])
		final_colorbar = final_fig.colorbar(final_mappable, cax=final_cax)
		final_colorbar.set_label("time [ns]", rotation=90, labelpad=10)
		pdf.savefig(final_fig)
		plt.close(final_fig)

	output_path = OUTPUT_DIR / f"{path.stem}_corrected.npz"
	np.savez_compressed(output_path, ch0_corrected=z_final.real, ch1_corrected=Q_SIGN * z_final.imag, **metadata, tau_s=calibration.tau_s, alpha_rad=calibration.alpha_rad, amplitude_a=calibration.amplitude_a, phi_rad=calibration.phi_rad, readout_frequency_hz=READOUT_FREQUENCY_HZ)
	print(f"[saved] {output_path}")


def main():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	frequency_hz, z_scan_raw = load_iq_scan(IQ_SCAN)
	calibration, details = calculate_calibration(frequency_hz, z_scan_raw)

	parameters = asdict(calibration)
	for key, value in list(parameters.items()):
		if isinstance(value, complex):
			parameters[key] = {"real": value.real, "imag": value.imag}
	parameters["tau_ns"] = calibration.tau_s * 1e9
	parameters["radial_rms"] = details["radial_rms"]
	(OUTPUT_DIR / "calibration_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

	with PdfPages(OUTPUT_DIR / "iqscan_geometric_calibration.pdf") as pdf:
		z_scan_final = save_scan_figures(frequency_hz, z_scan_raw, calibration, details, pdf)
		waveform_files = sorted(WAVEFORM_DIR.glob("*.npz"))
		if not waveform_files:
			raise FileNotFoundError(f"No .npz files found in {WAVEFORM_DIR}")
		for waveform_file in waveform_files:
			save_waveform_result(waveform_file, calibration, pdf)

	print(f"[saved] {OUTPUT_DIR / 'iqscan_geometric_calibration.pdf'}")
	print(f"tau={calibration.tau_s * 1e9:.6f} ns, alpha={calibration.alpha_rad:.6f} rad, a={calibration.amplitude_a:.6g}, phi={calibration.phi_rad:.6f} rad")
	print(f"scan points={frequency_hz.size}, waveform files={len(waveform_files)}")


if __name__ == "__main__":
	main()
