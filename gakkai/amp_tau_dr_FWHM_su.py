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


def main():
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


if __name__ == "__main__":
	main()


