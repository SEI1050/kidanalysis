"""Measure pulse amplitude and effective response times from IQ waveforms.

Usage:
	python3 amp_tau_dr_FWHM.py input.npz
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# Change this value when you want to average neighboring samples.
BIN_SIZE = 1

# Number of randomly selected events to draw in the PDF.
MAX_EVENTS_TO_PLOT = 100
RANDOM_SEED = None

# The 10% level is searched after the peak and before the peak.
THRESHOLD_FRACTION = 0.10

# Default input files. Edit this list directly when you want to fix the data path in code.
# If command-line arguments are provided, they take precedence over this list.
DEFAULT_INPUT_PATHS = [
    "/Volumes/NO NAME/data/20260828/data_0828_145003/data_0828_145003_combined.npz",
    "./alphaDC_combined.npz",
]


def channel_colors(base_color):
    """Use fixed raw-waveform colors for ch0/ch1 so later inputs do not collapse onto the same color."""
    # Raw-channel colors should be stable across inputs; only the comparison overlays use input-specific colors.
    return "#1f77b4", "#ff7f0e"


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
		t10_right,
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


def load_input_dataset(filename):
	"""Load a dataset and return event arrays needed for plotting."""
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
	all_signals = []
	for event_index in range(nwf):
		result, signal = analyze_event(
			time_ns,
			ch0[event_index],
			ch1[event_index],
			ref_position,
		)
		result["event"] = event_index
		results.append(result)
		all_signals.append(signal)

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
	select_count = min(MAX_EVENTS_TO_PLOT, nwf)
	rng = np.random.default_rng(RANDOM_SEED)
	selected_events = rng.choice(nwf, size=select_count, replace=False)
	return {
		"filename": filename,
		"label": os.path.splitext(os.path.basename(filename))[0],
		"time_ns": time_ns,
		"ch0": ch0,
		"ch1": ch1,
		"nwf": nwf,
		"parameters": parameters,
		"selected_events": selected_events,
		"signals": all_signals,
		"ref_position": ref_position,
		"npts_raw": npts_raw,
	}


def main():
	if len(sys.argv) > 1:
		filenames = [Path(arg).expanduser() for arg in sys.argv[1:]]
	else:
		filenames = [Path(path).expanduser() for path in DEFAULT_INPUT_PATHS]

	filenames = [path for path in filenames if path.exists()]
	if not filenames:
		raise SystemExit(
			"No input NPZ files were found. "
			"Edit DEFAULT_INPUT_PATHS in this script or pass explicit paths on the command line."
		)

	save_dir = Path.cwd()
	if len(filenames) == 1:
		output_path = save_dir / f"{filenames[0].stem}_event_waveforms.pdf"
	else:
		output_path = save_dir / "multi_input_event_waveforms.pdf"

	datasets = [load_input_dataset(str(path)) for path in filenames]
	colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
	if not colors:
		colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

	print("input files:", [str(path) for path in filenames])
	for dataset in datasets:
		print("file:", dataset["filename"])
		print("number of points before binning =", dataset["npts_raw"])
		print("number of events =", dataset["nwf"])
		print("ref_position =", dataset["ref_position"], "ped_ana_use =", (dataset["ref_position"] - 10) / 100 * dataset["npts_raw"])
		print("randomly selected events =", len(dataset["selected_events"]))

	with PdfPages(str(output_path)) as pdf:
		for index, dataset in enumerate(datasets):
			label = dataset["label"]
			base_color = colors[index % len(colors)]
			ch0_color, ch1_color = channel_colors(base_color)
			time_ns = dataset["time_ns"]
			ch0 = dataset["ch0"]
			ch1 = dataset["ch1"]
			parameters = dataset["parameters"]
			selected_events = dataset["selected_events"]
			ncol = 4
			nrow = 4
			for page_start in range(0, len(selected_events), nrow * ncol):
				page_events = selected_events[page_start:page_start + nrow * ncol]
				figure, axes = plt.subplots(
					nrow,
					ncol,
					figsize=(16, 9),
					sharex=True,
					sharey=True,
				)
				axes = np.asarray(axes).ravel()
				for axis, event_index in zip(axes, page_events):
					row = parameters.iloc[event_index]
					axis.plot(time_ns, ch0[event_index] * 1e3, color=ch0_color, linewidth=0.8, label="ch0")
					axis.plot(time_ns, ch1[event_index] * 1e3, color=ch1_color, linewidth=0.8, label="ch1")
					axis.set_title(
						f"{label}: event {event_index}\n"
						f"amp={row['amp'] * 1e3:.4g} mV, "
						f"tau_r={row['tau_r_eff']:.4g} ns, "
						f"tau_d={row['tau_d_eff']:.4g} ns",
						fontsize=9,
					)
					axis.grid(alpha=0.3)
					axis.legend(fontsize=7, loc="best")
				for axis in axes:
					axis.set_xlabel("Time [ns]")
					axis.set_ylabel("raw signal [mV]")
				for axis in axes[len(page_events):]:
					axis.axis("off")
				figure.suptitle(f"{label}: raw ch0/ch1 waveforms")
				figure.tight_layout()
				pdf.savefig(figure)
				plt.close(figure)

			if dataset["signals"]:
				# The band is computed from all events in the dataset, not only the randomly selected subset.
				signal_stack = np.asarray(dataset["signals"], dtype=float)
				median_signal = np.median(signal_stack, axis=0)
				lower_band = np.percentile(signal_stack, 16, axis=0)
				upper_band = np.percentile(signal_stack, 84, axis=0)
				figure, axis = plt.subplots(figsize=(10, 5))
				axis.plot(time_ns, median_signal * 1e3, color=ch0_color, linewidth=2.0, label=f"{label}: median")
				axis.fill_between(time_ns, lower_band * 1e3, upper_band * 1e3, color=ch0_color, alpha=0.25, label=f"{label}: 16-84%")
				axis.set_xlabel("Time [ns]")
				axis.set_ylabel("sqrt(ch0^2 + ch1^2) [mV]")
				axis.set_title(f"{label}: median waveform + 16-84% band")
				axis.grid(alpha=0.3)
				axis.legend(loc="best")
				figure.tight_layout()
				pdf.savefig(figure)
				plt.close(figure)

		# Final comparison pages for ch0, ch1, and absolute waveform.
		for channel_name, channel_key in [("ch0", "ch0"), ("ch1", "ch1")]:
			figure, axis = plt.subplots(figsize=(10, 5))
			for index, dataset in enumerate(datasets):
				label = dataset["label"]
				color = colors[index % len(colors)]
				waveforms = dataset[channel_key]
				stack = np.asarray(waveforms, dtype=float)
				median_signal = np.median(stack, axis=0)
				lower_band = np.percentile(stack, 16, axis=0)
				upper_band = np.percentile(stack, 84, axis=0)
				axis.plot(dataset["time_ns"], median_signal * 1e3, color=color, linewidth=2.0, label=f"{label}: median")
				axis.fill_between(dataset["time_ns"], lower_band * 1e3, upper_band * 1e3, color=color, alpha=0.2, label=f"{label}: 16-84%")
			axis.set_xlabel("Time [ns]")
			axis.set_ylabel(f"{channel_name} signal [mV]")
			axis.set_title(f"Comparison: median waveform + 16-84% band ({channel_name})")
			axis.grid(alpha=0.3)
			axis.legend(loc="best", fontsize=8)
			figure.tight_layout()
			pdf.savefig(figure)
			plt.close(figure)

		figure, axis = plt.subplots(figsize=(10, 5))
		for index, dataset in enumerate(datasets):
			label = dataset["label"]
			color = colors[index % len(colors)]
			signal_stack = np.asarray(dataset["signals"], dtype=float)
			median_signal = np.median(signal_stack, axis=0)
			lower_band = np.percentile(signal_stack, 16, axis=0)
			upper_band = np.percentile(signal_stack, 84, axis=0)
			axis.plot(dataset["time_ns"], median_signal * 1e3, color=color, linewidth=2.0, label=f"{label}: median")
			axis.fill_between(dataset["time_ns"], lower_band * 1e3, upper_band * 1e3, color=color, alpha=0.2, label=f"{label}: 16-84%")
		axis.set_xlabel("Time [ns]")
		axis.set_ylabel("sqrt(ch0^2 + ch1^2) [mV]")
		axis.set_title("Comparison: median waveform + 16-84% band (absolute)")
		axis.grid(alpha=0.3)
		axis.legend(loc="best", fontsize=8)
		figure.tight_layout()
		pdf.savefig(figure)
		plt.close(figure)

	print("saved:", output_path)


if __name__ == "__main__":
	main()


