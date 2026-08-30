"""Measure pulse amplitude and effective response times from IQ waveforms.

Usage:
	python ex_temp.py temperature_folder_1 temperature_folder_2
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# Change this value when you want to average neighboring samples.
BIN_SIZE = 1

# The 10% level is searched after the peak and before the peak.
THRESHOLD_FRACTION = 0.10


class ProgressBar:
	"""Render a compact progress bar without requiring an extra package."""

	def __init__(self, total, label):
		self.total = max(0, int(total))
		self.label = label
		self.current = 0
		self.update(0)

	def update(self, current=None):
		if current is None:
			self.current += 1
		else:
			self.current = current
		self.current = min(self.current, self.total)
		fraction = self.current / self.total if self.total else 1.0
		width = 28
		filled = int(width * fraction)
		bar = "#" * filled + "." * (width - filled)
		message = (
			f"\r{self.label}: [{bar}] {fraction:6.1%} "
			f"({self.current}/{self.total})"
		)
		sys.stdout.write(message)
		sys.stdout.flush()
		if self.current >= self.total:
			sys.stdout.write("\n")


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
	files = sorted(
		path
		for path in folder.glob("*.npz")
		if not path.name.startswith("._")
		if path.name != f"{folder.name}_combined.npz"
	)
	if not files:
		raise FileNotFoundError(f"No .npz files found in {folder}")

	ch0_parts = []
	ch1_parts = []
	metadata = None
	progress = ProgressBar(len(files), f"Combining {folder.name}")
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
		progress.update()

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


def load_folder(folder):
	"""Load the existing combined file, or create it when absent."""
	folder = Path(folder)
	combined_files = sorted(
		path for path in folder.glob("*_combined.npz")
		if not path.name.startswith("._")
	)
	filename = combined_files[0] if combined_files else combine_npz_files(folder)
	with np.load(filename, allow_pickle=True) as data:
		npts_raw = int(data["npts"])
		ref_position = float(data["ref_position"])
		sample_rate = float(data["sample_rate"])
		ch0, ch1, time_ns = bin_waveforms(
			data["ch0"], data["ch1"], npts_raw, sample_rate, ref_position
		)

	nwf = min(ch0.shape[0], ch1.shape[0])
	results = []
	signals = []
	progress = ProgressBar(nwf, f"Analyzing {folder.name}")
	for event_index in range(nwf):
		result, signal = analyze_event(
			time_ns, ch0[event_index], ch1[event_index], ref_position
		)
		result["event"] = event_index
		results.append(result)
		signals.append(signal)
		progress.update()

	return {
		"label": folder.name,
		"filename": filename,
		"time_ns": time_ns,
		"ch0": ch0[:nwf],
		"ch1": ch1[:nwf],
		"signals": np.asarray(signals),
		"parameters": pd.DataFrame(results),
		"npts_raw": npts_raw,
	}


HISTOGRAM_NAMES = [
	"amp", "t10_left", "t_peak", "t10_right", "left_integral",
	"right_integral", "tau_r_eff", "tau_d_eff", "fwhm_10",
]


def parse_args():
	parser = argparse.ArgumentParser(
		description="Compare waveform measurements from multiple temperature folders."
	)
	parser.add_argument("folders", nargs="+", type=Path)
	parser.add_argument(
		"-o", "--output", type=Path,
		help="PDF output path (default: temperature_comparison.pdf)",
	)
	parser.add_argument("--seed", type=int, default=None)
	return parser.parse_args()


def main():
	args = parse_args()
	groups = []
	group_progress = ProgressBar(len(args.folders), "Loading folders")
	for folder in args.folders:
		groups.append(load_folder(folder))
		group_progress.update()
	pdfname = args.output or args.folders[0] / "temperature_comparison.pdf"
	palette = plt.get_cmap("tab10").colors
	colors = [palette[index % len(palette)] for index in range(len(groups))]
	rng = np.random.default_rng(args.seed)

	with PdfPages(pdfname) as pdf:
		# Shared bins make the temperature distributions directly comparable.
		figure, axes = plt.subplots(3, 3, figsize=(16, 9))
		for axis, name in zip(axes.ravel(), HISTOGRAM_NAMES):
			all_values = [
				group["parameters"][name].to_numpy(dtype=float)
				for group in groups
				if name in group["parameters"]
			]
			all_values = np.concatenate(all_values) if all_values else np.array([])
			all_values = all_values[np.isfinite(all_values)]
			if all_values.size:
				bins = np.linspace(all_values.min(), all_values.max(), 101)
				if bins[0] == bins[-1]:
					bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, 101)
				for group, color in zip(groups, colors):
					values = group["parameters"][name].to_numpy(dtype=float)
					values = values[np.isfinite(values)]
					if values.size:
						axis.hist(
							values,
							bins=bins,
							weights=np.full(values.size, 1 / values.size),
							histtype="step",
							fill=False,
							color=color,
							linewidth=1.4,
							label=group["label"],
						)
			axis.set_xlabel(name)
			axis.set_ylabel("relative frequency")
			axis.grid(alpha=0.3)
		axes[0, 0].legend(fontsize=8)
		figure.suptitle("Temperature comparison: measurement histograms")
		figure.tight_layout()
		pdf.savefig(figure)
		plt.close(figure)

		# Draw 50 randomly selected raw ch0/ch1 events for each temperature.
		for group, color in zip(groups, colors):
			count = min(50, group["signals"].shape[0])
			selected = rng.choice(group["signals"].shape[0], count, replace=False)
			page_progress = ProgressBar(
				int(np.ceil(count / 16)),
				f"PDF raw waveforms {group['label']}"
			)
			for page_start in range(0, count, 16):
				figure, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True, sharey=True)
				for axis, event_index in zip(axes.ravel(), selected[page_start:page_start + 16]):
					axis.plot(group["time_ns"], group["ch0"][event_index] * 1e3,
						color="C0", linewidth=0.7, label="ch0")
					axis.plot(group["time_ns"], group["ch1"][event_index] * 1e3,
						color="C1", linewidth=0.7, label="ch1")
					axis.set_title(f"event {event_index}", fontsize=8)
					axis.grid(alpha=0.3)
				for axis in axes.ravel():
					axis.set_xlabel("Time [ns]")
					axis.set_ylabel("raw IQ [mV]")
				figure.suptitle(f"{group['label']}: random raw waveforms ({count} events)")
				figure.tight_layout()
				pdf.savefig(figure)
				plt.close(figure)
				page_progress.update()

		# Mean pulse shapes overlaid by temperature.
		figure, axis = plt.subplots(figsize=(12, 7))
		for group, color in zip(groups, colors):
			axis.plot(group["time_ns"], group["signals"].mean(axis=0) * 1e3,
				color=color, linewidth=2, label=group["label"])
		axis.set_xlabel("Time [ns]")
		axis.set_ylabel("sqrt(ch0^2 + ch1^2) [mV]")
		axis.set_title("Mean waveform by temperature")
		axis.grid(alpha=0.3)
		axis.legend()
		figure.tight_layout()
		pdf.savefig(figure)
		plt.close(figure)

		# Overlay 500 randomly selected pulse shapes per temperature.
		figure, axis = plt.subplots(figsize=(12, 7))
		counts = [min(500, group["signals"].shape[0]) for group in groups]
		overlay_progress = ProgressBar(
			sum(counts), "PDF overlay waveforms"
		)
		for group, color in zip(groups, colors):
			count = min(500, group["signals"].shape[0])
			selected = rng.choice(group["signals"].shape[0], count, replace=False)
			for event_index in selected:
				axis.plot(group["time_ns"], group["signals"][event_index] * 1e3,
					color=color, alpha=0.025, linewidth=0.35)
				overlay_progress.update()
			axis.plot([], [], color=color, linewidth=2, label=group["label"])
		axis.set_xlabel("Time [ns]")
		axis.set_ylabel("sqrt(ch0^2 + ch1^2) [mV]")
		axis.set_title("Random waveforms overlaid by temperature (up to 500 each)")
		axis.grid(alpha=0.3)
		axis.legend()
		figure.tight_layout()
		pdf.savefig(figure)
		plt.close(figure)

	print(f"Loaded {len(groups)} folders")
	print(f"Saved: {pdfname}")


if __name__ == "__main__":
	main()


