"""Measure pulse amplitude and effective response times from IQ waveforms.

Usage:
	python3 amp_tau_dr_FWHM.py input.npz
"""

import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# Change this value when you want to average neighboring samples.
BIN_SIZE = 1

# Number of randomly selected events to draw for the channel waveforms.
MAX_EVENTS_TO_PLOT = 100
RANDOM_SEED = None

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


def combine_npz_files(folder, output_dir=None):
	"""Combine all compatible NPZ waveform files in a folder."""
	folder = Path(folder)
	output_dir = Path(output_dir) if output_dir else folder
	output_dir.mkdir(parents=True, exist_ok=True)
	combined_path = output_dir / f"{folder.name}_combined.npz"
    # selected_date_dirs = [
	# 	"data_0825_125646",
	# 	"data_0825_125855",
	# 	"data_0825_130029",
	# 	"data_0825_130112",
	# 	"data_0825_130152",
    # ]
	# Restrict the search to only the date folders explicitly listed below.
	# This avoids scanning unrelated folders in the same parent directory.
	# selected_date_dirs = [
	# 	"data_0825_125646",
	# 	"data_0825_125855",
	# 	"data_0825_130029",
	# 	"data_0825_130112",
	# 	"data_0825_130152",
	# 	"data_0825_130234",
	# 	"data_0825_130319",
	# 	"data_0825_130522",
	# 	"data_0825_130614",
	# 	"data_0825_130702",
	# 	"data_0825_130830",
	# 	"data_0825_130912",
	# 	"data_0825_130955",
	# 	"data_0825_131115",
	# 	"data_0825_131200",
	# 	"data_0825_131249",
	# 	"data_0825_131334",
	# 	"data_0825_131433",
	# 	"data_0825_131518",
	# 	"data_0825_131603",
	# 	"data_0825_131703",
	# 	"data_0825_131920",
	# 	"data_0825_132005",
	# 	"data_0825_132057",
	# 	"data_0825_132223",
	# 	"data_0825_132306",
	# 	"data_0825_132358",
	# 	"data_0825_132441",
	# 	"data_0825_132530",
	# 	"data_0825_132639",
	# 	"data_0825_132725",
	# 	"data_0825_132809",
	# 	"data_0825_132904",
	# 	"data_0825_132943",
	# 	"data_0825_133057",
	# 	"data_0825_133143",
	# 	"data_0825_133234",
	# 	"data_0825_133314",
	# 	"data_0825_133401",
	# 	"data_0825_133500",
	# 	"data_0825_133545",
	# 	"data_0825_133635",
	# 	"data_0825_133718",
	# 	"data_0825_133805",
	# 	"data_0825_133849",
	# 	"data_0825_133955",
	# 	"data_0825_134041",
	# 	"data_0825_134123",
	# 	"data_0825_134208",
	# 	"data_0825_134248",
	# 	"data_0825_134328",
	# 	"data_0825_134407",
	# 	"data_0825_134447",
	# 	"data_0825_134626",
	# 	"data_0825_134710",
	# 	"data_0825_134748",
	# 	"data_0825_134832",
	# 	"data_0825_134913",
	# 	"data_0825_134954",
	# 	"data_0825_135036",
	# 	"data_0825_135121",
	# 	"data_0825_135222",
	# 	"data_0825_135306",
	# 	"data_0825_135354",
	# 	"data_0825_135440",
	# 	"data_0825_135532",
    # ]
		# "data_0825_135620",
		# "data_0825_141842",
		# "data_0825_141921",
		# "data_0825_142013",
		# "data_0825_142128",
		# "data_0825_142207",
		# "data_0825_142254",
		# "data_0825_142406",
		# "data_0825_142440",
		# "data_0825_142514",
		# "data_0825_142610",
		# "data_0825_142644",
		# "data_0825_142719",
		# "data_0825_142907",
		# "data_0825_142946",
		# "data_0825_143024",
		# "data_0825_143116",
		# "data_0825_143153",
		# "data_0825_143227",
		# "data_0825_143316",
		# "data_0825_143356",
		# "data_0825_143432",
		# "data_0825_143521",
		# "data_0825_143557",
		# "data_0825_143634",
		# "data_0825_143735",
		# "data_0825_143810",
		# "data_0825_143848",
		# "data_0825_143938",
		# "data_0825_144012",
		# "data_0825_144055",
		# "data_0825_144148",
		# "data_0825_144224",
		# "data_0825_144259",
		# "data_0825_144402",
		# "data_0825_144439",
		# "data_0825_144518",
		# "data_0825_144624",
		# "data_0825_144702",
		# "data_0825_144740",
		# "data_0825_144835",
		# "data_0825_144913",
		# "data_0825_144948",
		# "data_0825_145048",
		# "data_0825_145123",
		# "data_0825_145208",
		# "data_0825_145306",
		# "data_0825_145342",
		# "data_0825_145429",
		# "data_0825_150934",
		# "data_0825_151012",
		# "data_0825_151055",
		# "data_0825_151433",
		# "data_0825_151552",
		# "data_0825_151656",
		# "data_0825_151745",
		# "data_0825_151840",
		# "data_0825_151928",
		# "data_0825_152048",
		# "data_0825_152134",
		# "data_0825_152219",
		# "data_0825_152330",
		# "data_0825_152415",
		# "data_0825_152521",
		# "data_0825_152623",
		# "data_0825_152711",
		# "data_0825_152800",
		# "data_0825_154145",
		# "data_0825_154219",
		# "data_0825_154411",
		# "data_0825_154541",
		# "data_0825_154628",
		# "data_0825_154756",
		# "data_0825_154834",
		# "data_0825_154918",
		# "data_0825_155009",
		# "data_0825_155057",
		# "data_0825_160123",
		# "data_0825_160303",
		# "data_0825_160504",
		# "data_0825_160550",
		# "data_0825_160653",
		# "data_0825_161934",
		# "data_0825_162017",
		# "data_0825_162102",
		# "data_0825_164423",
		# "data_0825_164603",
		# "data_0825_164654",
		# "data_0825_164746",
		# "data_0825_164855",
		# "data_0825_165000",
		# "data_0825_165101",
		# "data_0825_165155",
		# "data_0825_171703",
		# "data_0825_171758",
		# "data_0825_171843",
		# "data_0825_171939",
		# "data_0825_172027",
		# "data_0825_172106",
		# "data_0825_172148",
		# "data_0825_172226",
	
	# candidate_dirs = [
	# 	folder / date_dir
	# 	for date_dir in selected_date_dirs
	# 	if (folder / date_dir).is_dir()
	# ]
	# files = sorted(
	# 	path
	# 	for data_dir in candidate_dirs
	# 	for path in data_dir.rglob("*.npz")
	# 	if not path.name.startswith("._")
	# 	if not path.name.endswith("_combined.npz")
	# )

	# Full recursive option (leave enabled only when you want all folders):
	files = sorted(
		path
		for path in folder.rglob("*.npz")
		if not path.name.startswith("._")
		if not path.name.endswith("_combined.npz")
	)

	if not files:
		raise FileNotFoundError(f"No .npz files found in {folder}")

	ch0_parts = []
	ch1_parts = []
	metadata = None
	for path in files:
		with np.load(path, allow_pickle=True) as data:
			required_keys = {"npts", "sample_rate", "ref_position", "ch0", "ch1"}
			missing_keys = required_keys.difference(data.files)
			if missing_keys:
				raise ValueError(
					f"{path} is missing NPZ keys: {sorted(missing_keys)}"
				)

			npts = int(data["npts"])
			sample_rate = float(data["sample_rate"])
			ref_position = float(data["ref_position"])
			ch0 = np.asarray(data["ch0"])
			ch1 = np.asarray(data["ch1"])
			if ch0.ndim != 2 or ch1.ndim != 2 or ch0.shape != ch1.shape:
				raise ValueError(
					f"{path} must have matching 2D ch0/ch1 arrays; "
					f"got {ch0.shape} and {ch1.shape}"
				)
			if ch0.shape[1] < npts:
				raise ValueError(
					f"{path} has {ch0.shape[1]} samples, fewer than npts={npts}"
				)

			current_metadata = (npts, sample_rate, ref_position)
			if metadata is None:
				metadata = current_metadata
			elif not (
				current_metadata[0] == metadata[0]
				and np.isclose(current_metadata[1:], metadata[1:]).all()
			):
				raise ValueError(
					f"Incompatible metadata in {path}: {current_metadata}; "
					f"expected {metadata}"
				)

		ch0_parts.append(ch0[:, :npts])
		ch1_parts.append(ch1[:, :npts])

	npts, sample_rate, ref_position = metadata
	total_bytes = sum(
		ch0_part.nbytes + ch1_part.nbytes
		for ch0_part, ch1_part in zip(ch0_parts, ch1_parts)
	)
	if shutil.disk_usage(output_dir).free < total_bytes:
		if output_dir != Path.cwd():
			fallback_dir = Path.cwd()
			if shutil.disk_usage(fallback_dir).free < total_bytes:
				raise OSError(
					f"Not enough disk space for {total_bytes / 1024**3:.2f} GB "
					f"of combined data in {output_dir} or {fallback_dir}"
				)
			output_dir = fallback_dir
			combined_path = output_dir / f"{folder.name}_combined.npz"
		print(f"not enough space in {folder}; using {output_dir}")

	np.savez(
		combined_path,
		npts=npts,
		sample_rate=sample_rate,
		ref_position=ref_position,
		ch0=np.concatenate(ch0_parts, axis=0),
		ch1=np.concatenate(ch1_parts, axis=0),
	)
	print(f"combined {len(files)} files -> {combined_path}")
	return combined_path


def main():
	if len(sys.argv) < 2:
		raise SystemExit(
		"Usage: python3 ex_check.py input.npz|folder [output_dir]"
	)

	input_path = Path(sys.argv[1])
	if input_path.is_dir():
		output_dir = sys.argv[2] if len(sys.argv) >= 3 else None
		filename = combine_npz_files(input_path, output_dir)
	else:
		filename = input_path
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
	output_dir = Path(filename).parent
	csvname = output_dir / (basename + "_amp_tau_eff.csv")
	pdfname = output_dir / (basename + "_amp_tau_eff.pdf")
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

		# Draw pedestal-subtracted ch0 and ch1 for random events.
		select_count = min(MAX_EVENTS_TO_PLOT, nwf)
		rng = np.random.default_rng(RANDOM_SEED)
		selected_events = rng.choice(nwf, size=select_count, replace=False)
		for page_start in range(0, select_count, nrow * ncol):
			page_events = selected_events[page_start:page_start + nrow * ncol]
			figure, axes = plt.subplots(
				nrow, ncol, figsize=(16, 9), sharex=True, sharey=True,
			)
			axes = np.asarray(axes).ravel()
			for axis, event_index in zip(axes, page_events):
				row = parameters.iloc[event_index]
				axis.plot(
					time_ns, (ch0[event_index] - row["ped0"]) * 1e3,
					color="C0", linewidth=0.8, label="ch0",
				)
				axis.plot(
					time_ns, (ch1[event_index] - row["ped1"]) * 1e3,
					color="C1", linewidth=0.8, label="ch1",
				)
				axis.set_title(
					f"event {event_index}\n"
					f"amp={row['amp'] * 1e3:.4g} mV, "
					f"tau_r={row['tau_r_eff']:.4g} ns, "
					f"tau_d={row['tau_d_eff']:.4g} ns",
					fontsize=9,
				)
				axis.grid(alpha=0.3)
				axis.legend(fontsize=7, loc="best")
			for axis in axes:
				axis.set_xlabel("Time [ns]")
				axis.set_ylabel("pedestal-subtracted [mV]")
			for axis in axes[len(page_events):]:
				axis.axis("off")
			figure.suptitle(
				f"Random {select_count} events: ch0 and ch1",
				fontsize=12,
			)
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


