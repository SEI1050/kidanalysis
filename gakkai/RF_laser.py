"""Measure pulse amplitude and effective response times from IQ waveforms.

Usage:
	python3 amp_tau_dr_FWHM.py input.npz
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# Change this value when you want to average neighboring samples.
BIN_SIZE = 1

# The 10% level is searched after the peak and before the peak.
THRESHOLD_FRACTION = 0.10


def linear_crossing(time_ns, signal, level, start, stop, direction):
	"""Return the linearly interpolated time where signal reaches level."""
	if direction == "up":
		condition = lambda left, right: left < level <= right
		indices = range(start, stop)
	elif direction == "up_from_peak":
		condition = lambda left, right: left < level <= right
		indices = range(start - 1, stop - 1, -1)
	else:
		condition = lambda left, right: left >= level > right
		indices = range(start, stop)

	for index in indices:
		if condition(signal[index], signal[index + 1]):
			left_time = time_ns[index]
			right_time = time_ns[index + 1]
			left_signal = signal[index]
			right_signal = signal[index + 1]

			if right_signal == left_signal:
				return 0.5 * (left_time + right_time)

			return left_time + (level - left_signal) * (
				right_time - left_time
			) / (right_signal - left_signal)

	return np.nan


def integral_between(time_ns, signal, first_time, last_time):
	"""Integrate signal between two times using trapezoids."""
	if not np.isfinite(first_time) or not np.isfinite(last_time):
		return np.nan

	if last_time <= first_time:
		return np.nan

	inside = (time_ns > first_time) & (time_ns < last_time)
	integration_time = np.concatenate((
		[first_time],
		time_ns[inside],
		[last_time],
	))
	integration_signal = np.interp(
		integration_time,
		time_ns,
		signal,
	)

	return np.trapezoid(integration_signal, integration_time)


def analyze_event(time_ns, ch0_waveform, ch1_waveform, ref_position):
	"""Analyze one event and return its parameters and plotting information."""
	# Use the pre-trigger region to estimate each channel's pedestal.
	pedestal_end = (ref_position-10) / 100 * time_ns.size
	pedestal_end = max(1, min(time_ns.size, int(pedestal_end)))
	ped0 = np.mean(ch0_waveform[:pedestal_end])
	ped1 = np.mean(ch1_waveform[:pedestal_end])

	ch0_signal = ch0_waveform - ped0
	ch1_signal = ch1_waveform - ped1

	# The square-root of the sum of squares is always non-negative.
	signal = np.hypot(ch0_signal, ch1_signal)
	peak_index = int(np.argmax(signal))
	amp = signal[peak_index]
	peak_time = time_ns[peak_index]

	# if not np.isfinite(amp) or amp <= 0 or peak_index == 0:
	# 	return {
	# 		"ped0": ped0,
	# 		"ped1": ped1,
	# 		"amp": np.nan,
	# 		"t10_left": np.nan,
	# 		"t_peak": peak_time,
	# 		"t10_right": np.nan,
	# 		"left_integral": np.nan,
	# 		"right_integral": np.nan,
	# 		"tau_r_eff": np.nan,
	# 		"tau_d_eff": np.nan,
	# 		"fwhm_10": np.nan,
	# 	}, signal

	level = THRESHOLD_FRACTION * amp
	t10_left = linear_crossing(
		time_ns,
		signal,
		level,
		peak_index,
		0,
		"up_from_peak",
	)
	t10_right = linear_crossing(
		time_ns,
		signal,
		level,
		peak_index,
		signal.size - 1,
		"down",
	)
	# If the decay is still above 10% at the record end, use the
	# available truncated waveform instead of discarding the integral.
	right_integral_end = t10_right if np.isfinite(t10_right) else time_ns[-1]

	left_integral = integral_between(
		time_ns,
		signal,
		t10_left,
		peak_time,
	)
	right_integral = integral_between(
		time_ns,
		signal,
		peak_time,
		right_integral_end,
	)

	# Integral [mV ns] / amplitude [mV] = ns.
	tau_r_eff = left_integral / amp if np.isfinite(left_integral) else np.nan
	tau_d_eff = right_integral / amp if np.isfinite(right_integral) else np.nan
	fwhm_10 = t10_right - t10_left

	result = {
		"ped0": ped0,
		"ped1": ped1,
		"amp": amp,
		"t10_left": t10_left,
		"t_peak": peak_time,
		"t10_right": t10_right,
		"left_integral": left_integral,
		"right_integral": right_integral,
		"tau_r_eff": tau_r_eff,
		"tau_d_eff": tau_d_eff,
		"fwhm_10": fwhm_10,
	}
	return result, signal


def bin_waveforms(ch0, ch1, npts_raw, sample_rate, ref_position):
	"""Average BIN_SIZE samples and return waveforms and their time axis."""
	nbin = npts_raw // BIN_SIZE
	usable_npts = nbin * BIN_SIZE

	ch0_binned = ch0[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)
	ch1_binned = ch1[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)

	raw_time = (
		np.arange(usable_npts) - npts_raw * ref_position / 100
	) / sample_rate
	time_ns = raw_time.reshape(nbin, BIN_SIZE).mean(axis=1) * 1e9

	return ch0_binned, ch1_binned, time_ns


def process_single_file(filename):
	if len(sys.argv) < 2:
		raise SystemExit("Usage: python3 amp_tau_dr_FWHM.py input.npz")

	filename = sys.argv[1]
	data = np.load(filename, allow_pickle=True)

	npts_raw = int(data["npts"])
	ref_position = float(data["ref_position"])
	sample_rate = float(data["sample_rate"])
	ch0, ch1, time_ns = bin_waveforms(
		data["ch0"],
		data["ch1"],
		npts_raw,
		sample_rate,
		ref_position,
	)

	nwf = min(ch0.shape[0], ch1.shape[0])
	results = []
	signals = []

	for event_index in range(nwf):
		result, signal = analyze_event(
			time_ns,
			ch0[event_index],
			ch1[event_index],
			ref_position,
		)
		result["event"] = event_index
		results.append(result)
		signals.append(signal)

	columns = [
		"event",
		"ped0",
		"ped1",
		"amp",
		"t10_left",
		"t_peak",
		"t10_right",
		"left_integral",
		"right_integral",
		"tau_r_eff",
		"tau_d_eff",
		"fwhm_10",
	]
	parameters = pd.DataFrame(results)[columns]

	basename = os.path.splitext(os.path.basename(filename))[0]
	csvname = basename + "_amp_tau_eff.csv"
	pdfname = basename + "_amp_tau_eff.pdf"
	parameters.to_csv(csvname, index=False)

	print("number of points before binning =", npts_raw)
	print("number of events =", nwf)
	print("ref_position =", ref_position,"ped_ana_use =", (ref_position-10)/100*npts_raw)
	print(parameters)
	print("saved:", csvname)

	# Keep the same 4 x 4 event layout as waveform_fit.py.
	with PdfPages(pdfname) as pdf:
		ncol = 4
		nrow = 4
		figure, axes = plt.subplots(
			nrow,
			ncol,
			figsize=(16, 9),
			sharex=True,
			sharey=True,
		)
		axes = np.asarray(axes).ravel()
		for event_index in range(min(nrow * ncol, nwf)):
			axis = axes[event_index]
			row = parameters.iloc[event_index]
			axis.plot(time_ns, signals[event_index] * 1e3, color="C0")
			axis.axhline(row["amp"] * 1e2, color="C1", ls="--", alpha=0.7)
			axis.axvline(row["t10_left"], color="C2", ls=":")
			axis.axvline(row["t_peak"], color="k", ls="-")
			axis.axvline(row["t10_right"], color="C3", ls=":")
			axis.set_title(f"event {event_index}")
			axis.grid(alpha=0.3)

		for axis in axes:
			axis.set_xlabel("Time [ns]")
			axis.set_ylabel("sqrt(ch0^2 + ch1^2) [mV]")
		figure.tight_layout()
		pdf.savefig(figure)
		plt.close(figure)

		# Histograms of the measured quantities.
		histogram_names = [
			"amp",
			"t10_left",
			"t_peak",
			"t10_right",
			"left_integral",
			"right_integral",
			"tau_r_eff",
			"tau_d_eff",
			"fwhm_10",
		]
		figure, axes = plt.subplots(3, 3, figsize=(16, 9))
		for axis, name in zip(axes.ravel(), histogram_names):
			values = parameters[name].to_numpy(dtype=float)
			values = values[np.isfinite(values)]
			if values.size:
				axis.hist(values, bins=100, color="C0")
			axis.set_xlabel(name)
			axis.set_ylabel("counts")
			axis.grid(alpha=0.3)
		figure.tight_layout()
		pdf.savefig(figure)
		plt.close(figure)

	print("saved:", pdfname)


RF_POSITION_FILES = {
	(6.9, 4.6): ("145306", "145342", "130830"),
	(6.9, 5.1): ("142907", "142946", "135121"),
	(6.7, 4.6): ("145048", "145123", "131115"),
	(6.7, 5.1): ("143116", "143153", "134832"),
	(6.5, 5.1): ("143316", "143356", "134710"),
	(6.3, 4.3): ("141842", "141921", "132358"),
	(6.3, 4.6): ("144835", "144913", "131249"),
	(6.3, 5.1): ("143521", "143557", "134447"),
	(6.3, 5.6): ("142610", "142644", "132809"),
	(6.1, 5.1): ("143735", "143810", "133545"),
	(6.0, 4.3): ("142128", "142207", "132441"),
	(6.0, 4.6): ("144624", "144702", "131920"),
	(6.0, 5.1): ("143938", "144012", "134208"),
	(6.0, 5.6): ("142406", "142440", "132725"),
	(5.7, 4.6): ("144402", "144439", "131703"),
	(5.7, 5.1): ("144148", "144224", "133955"),
}

RF_LABELS = ("0 dBm", "-5 dBm", "-2 dBm")
RF_COLORS = ("#70fa70", "#101070", "#1688ff")


def load_rf_result(filename):
	"""Load one RF run and return parameters plus mean IQ waves."""
	data = np.load(filename, allow_pickle=True)
	npts_raw = int(data["npts"])
	ref_position = float(data["ref_position"])
	sample_rate = float(data["sample_rate"])
	ch0, ch1, time_ns = bin_waveforms(
		data["ch0"], data["ch1"], npts_raw, sample_rate, ref_position
	)
	nwf = min(ch0.shape[0], ch1.shape[0])
	dev0 = np.empty_like(ch0[:nwf], dtype=float)
	dev1 = np.empty_like(ch1[:nwf], dtype=float)
	results = []
	for event_index in range(nwf):
		pedestal_end = int((ref_position - 10) / 100 * time_ns.size)
		pedestal_end = max(1, min(time_ns.size, pedestal_end))
		dev0[event_index] = ch0[event_index] - np.mean(ch0[event_index, :pedestal_end])
		dev1[event_index] = ch1[event_index] - np.mean(ch1[event_index, :pedestal_end])
		result, _ = analyze_event(time_ns, ch0[event_index], ch1[event_index], ref_position)
		result["event"] = event_index
		results.append(result)

	mean_i = np.mean(dev0, axis=0)
	mean_q = np.mean(dev1, axis=0)
	peak_index = int(np.argmax(np.hypot(mean_i, mean_q)))
	direction_norm = np.hypot(mean_i[peak_index], mean_q[peak_index])
	if direction_norm == 0:
		direction = (1.0, 0.0)
	else:
		direction = (mean_i[peak_index] / direction_norm, mean_q[peak_index] / direction_norm)
	mean_proj = mean_i * direction[0] + mean_q * direction[1]
	return {
		"parameters": pd.DataFrame(results),
		"time_ns": time_ns,
		"mean_i": mean_i,
		"mean_q": mean_q,
		"mean_proj": mean_proj,
	}


def normalize_wave(wave):
	"""Normalize a mean wave to its largest positive excursion."""
	peak = np.max(wave)
	if not np.isfinite(peak) or peak <= 0:
		peak = np.max(np.abs(wave))
	return wave / peak if np.isfinite(peak) and peak > 0 else wave


def process_position_rf_comparison(root):
	"""Create RF-dependent waveform and histogram pages for each position."""
	output = root / "RF_laser_position_comparison.pdf"
	histogram_names = [
		"amp", "t10_left", "t_peak", "t10_right", "left_integral",
		"right_integral", "tau_r_eff", "tau_d_eff", "fwhm_10",
	]
	position_groups = {}
	with PdfPages(output) as pdf:
		for (z_position, x_position), timestamps in RF_POSITION_FILES.items():
			groups = []
			for timestamp, label, color in zip(timestamps, RF_LABELS, RF_COLORS):
				matches = sorted((root / f"data_0825_{timestamp}").glob("wf_*.npz"))
				if not matches:
					print(f"[skip] missing data_0825_{timestamp}")
					continue
				result = load_rf_result(matches[0])
				result.update({"label": label, "color": color})
				groups.append(result)
			if not groups:
				continue
			if len(groups) != len(RF_LABELS):
				print(
					f"[skip] incomplete RF set at z={z_position:.1f}, "
					f"x={x_position:.1f}"
				)
				continue

			figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
			for axis, wave_name, axis_label in zip(
				axes, ("mean_i", "mean_q", "mean_proj"), ("I (ch0)", "Q (ch1)", "Proj")
			):
				for group in groups:
					axis.plot(
						group["time_ns"] / 1000,
						normalize_wave(group[wave_name]),
						color=group["color"], label=group["label"], lw=1.8,
					)
				axis.set_title(axis_label)
				axis.set_xlabel("time [us]")
				axis.set_ylabel("normalized voltage")
				axis.set_xlim(-0.35, 1.65)
				axis.grid(alpha=0.3)
				axis.legend(fontsize=8)
			figure.suptitle(f"RF dependence: z={z_position:.1f}, x={x_position:.1f} mm")
			figure.tight_layout()
			pdf.savefig(figure)
			plt.close(figure)

			figure, axes = plt.subplots(3, 3, figsize=(16, 9))
			for axis, name in zip(axes.ravel(), histogram_names):
				all_values = np.concatenate([
					group["parameters"][name].to_numpy(dtype=float) for group in groups
				])
				all_values = all_values[np.isfinite(all_values)]
				if all_values.size:
					bins = np.linspace(all_values.min(), all_values.max(), 101)
					if bins[0] == bins[-1]:
						bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, 101)
					for group in groups:
						values = group["parameters"][name].to_numpy(dtype=float)
						values = values[np.isfinite(values)]
						if values.size:
							axis.hist(
								values, bins=bins, histtype="step", fill=False,
								color=group["color"], linewidth=1.4, label=group["label"],
							)
				axis.set_xlabel(name)
				axis.set_ylabel("counts")
				axis.grid(alpha=0.3)
			axes[0, 0].legend(fontsize=8)
			figure.suptitle(f"RF dependence histograms: z={z_position:.1f}, x={x_position:.1f} mm")
			figure.tight_layout()
			pdf.savefig(figure)
			plt.close(figure)

			position_groups[(z_position, x_position)] = groups

		# Arrange the requested positions spatially so each histogram can be
		# compared without searching through separate position pages.
		requested_z = (6.7, 6.5, 6.3, 6.1, 6.0)
		requested_x = (4.6, 5.1)
		for name in histogram_names:
			figure, axes = plt.subplots(
				len(requested_z), len(requested_x), figsize=(12, 18), squeeze=False
			)
			for row_index, z_position in enumerate(requested_z):
				for column_index, x_position in enumerate(requested_x):
					axis = axes[row_index, column_index]
					groups = position_groups.get((z_position, x_position), [])
					if groups:
						all_values = np.concatenate([
							group["parameters"][name].to_numpy(dtype=float)
							for group in groups
						])
						all_values = all_values[np.isfinite(all_values)]
						if all_values.size:
							bins = np.linspace(all_values.min(), all_values.max(), 51)
							if bins[0] == bins[-1]:
								bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, 51)
							for group in groups:
								values = group["parameters"][name].to_numpy(dtype=float)
								values = values[np.isfinite(values)]
								axis.hist(
									values, bins=bins, histtype="step", fill=False,
									color=group["color"], linewidth=1.2,
									label=group["label"],
								)
					axis.set_title(f"x={x_position:.1f}, z={z_position:.1f} mm")
					axis.grid(alpha=0.3)
					if column_index == 0:
						axis.set_ylabel("counts")
					if row_index == len(requested_z) - 1:
						axis.set_xlabel(name)
					if row_index == 0 and column_index == 0:
						axis.legend(fontsize=7)
			figure.suptitle(f"Position dependence of {name}")
			figure.tight_layout()
			pdf.savefig(figure)
			plt.close(figure)

		# Arrange the normalized mean I, Q, and projected waveforms spatially.
		for wave_name, wave_label in (
			("mean_i", "I (ch0)"),
			("mean_q", "Q (ch1)"),
			("mean_proj", "Proj"),
		):
			figure, axes = plt.subplots(
				len(requested_z), len(requested_x), figsize=(12, 15),
				squeeze=False, sharex=True, sharey=True,
			)
			for row_index, z_position in enumerate(requested_z):
				for column_index, x_position in enumerate(requested_x):
					axis = axes[row_index, column_index]
					groups = position_groups.get((z_position, x_position), [])
					for group in groups:
						axis.plot(
							group["time_ns"] / 1000,
							normalize_wave(group[wave_name]),
							color=group["color"], linewidth=1.2,
							label=group["label"],
						)
					axis.set_title(f"x={x_position:.1f}, z={z_position:.1f} mm")
					axis.set_xlim(-0.35, 1.65)
					axis.set_ylim(-0.25, 1.1)
					axis.grid(alpha=0.3)
					if column_index == 0:
						axis.set_ylabel("normalized voltage")
					if row_index == len(requested_z) - 1:
						axis.set_xlabel("time [us]")
					if row_index == 0 and column_index == 0:
						axis.legend(fontsize=7)
			figure.suptitle(f"Position dependence of normalized {wave_label} mean waveform")
			figure.tight_layout()
			pdf.savefig(figure)
			plt.close(figure)
	print("saved:", output)


def main():
	if len(sys.argv) < 2:
		raise SystemExit("Usage: python3 RF_laser.py input.npz or 20260825_directory")
	input_path = os.path.abspath(sys.argv[1])
	if os.path.isdir(input_path):
		from pathlib import Path
		process_position_rf_comparison(Path(input_path))
	else:
		process_single_file(input_path)


if __name__ == "__main__":
	main()


