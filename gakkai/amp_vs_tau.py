"""Create 2D histograms of amplitude and effective time constants."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


# The CSV file is in the same directory as this script.
INPUT_FILE = Path(__file__).with_name(
	"alphaDC_combined_amp_tau_eff.csv"
)

# The output files will also be saved in the same directory.
OUTPUT_FILE = INPUT_FILE.with_name(
	INPUT_FILE.stem + "_2d_hist.png"
)
LOG_OUTPUT_FILE = INPUT_FILE.with_name(
	INPUT_FILE.stem + "_2d_hist_log.png"
)

# An event is selected when its amplitude exceeds AMP_THRESHOLD, or when
# both effective time constants exceed their respective thresholds.
AMP_THRESHOLD = 0.002
AMP_UPPER_THRESHOLD = 0.1
TAU_R_THRESHOLD = 20.0
TAU_R_UPPER_THRESHOLD = 300.0
TAU_D_THRESHOLD = 20.0
TAU_D_UPPER_THRESHOLD = 2000.0



def plot_histograms(output_path, *, log_counts=False):
	"""Draw the 2D amplitude-vs-tau histograms.

	If log_counts is True, use a logarithmic color scale to better show sparse
	regions at large tau values.
	"""
	# Read the CSV file into a pandas DataFrame.
	data = pd.read_csv(INPUT_FILE)

	# Convert the three columns to numbers.
	# Invalid values become NaN, which can be removed below.
	amplitude = pd.to_numeric(data["amp"], errors="coerce")
	tau_r_eff = pd.to_numeric(data["tau_r_eff"], errors="coerce")
	tau_d_eff = pd.to_numeric(data["tau_d_eff"], errors="coerce")

	finite = (
		np.isfinite(amplitude)
		& np.isfinite(tau_r_eff)
		& np.isfinite(tau_d_eff)
	)
	amp_valid = finite & (amplitude > 0)
	selected = amp_valid & (
		(amplitude > AMP_THRESHOLD)
		| (
			(tau_r_eff > TAU_R_THRESHOLD)
			& (tau_d_eff > TAU_D_THRESHOLD)
		)
	)
	selected = selected & (amplitude <= AMP_UPPER_THRESHOLD)
	selected = selected & (tau_r_eff <= TAU_R_UPPER_THRESHOLD)
	selected = selected & (tau_d_eff <= TAU_D_UPPER_THRESHOLD)
	not_selected = amp_valid & ~selected

	if not finite.any():
		raise ValueError("No finite amp/tau values were found in the CSV.")

	# Separate the amplitude-tau correlations by the common event criterion.
	figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
	for row, mask, group_name in (
		(0, selected, "selected"),
		(1, not_selected, "not selected"),
	):
		for axis, tau, tau_name in (
			(axes[row, 0], tau_r_eff, "tau_r_eff"),
			(axes[row, 1], tau_d_eff, "tau_d_eff"),
		):
			mask_plot = (
				mask
				& (amplitude > 0)
				& (amplitude <= AMP_UPPER_THRESHOLD)
				& (tau <= TAU_R_UPPER_THRESHOLD if tau_name == "tau_r_eff" else tau <= TAU_D_UPPER_THRESHOLD)
			)
			if mask_plot.any():
				if log_counts:
					histogram = axis.hist2d(
						amplitude[mask_plot], tau[mask_plot], bins=200, cmap="viridis",
						norm=LogNorm(vmin=1, vmax=max(2, int(np.nanmax(np.histogram2d(
							amplitude[mask_plot], tau[mask_plot], bins=200,
						)[0]
						))))
					)
					figure.colorbar(histogram[3], ax=axis, label="log10(counts)")
				else:
					histogram = axis.hist2d(
						amplitude[mask_plot], tau[mask_plot], bins=200, cmap="viridis",
					)
					figure.colorbar(histogram[3], ax=axis, label="counts")
			axis.set_title(f"{group_name}: amp vs {tau_name} (n={mask_plot.sum()})")
			axis.set_xlabel("amp")
			axis.set_ylabel(tau_name)
			axis.set_xscale("log")
			axis.set_xlim(left=AMP_THRESHOLD * 0.5, right=AMP_UPPER_THRESHOLD)
			axis.grid(alpha=0.25)

	if log_counts:
		figure.suptitle("Amplitude vs tau (log count scale)")
	else:
		figure.suptitle("Amplitude vs tau")
	figure.savefig(output_path, dpi=150)
	plt.close(figure)

	return data, finite, selected, not_selected


def main():
	data, finite, selected, not_selected = plot_histograms(OUTPUT_FILE, log_counts=False)
	plot_histograms(LOG_OUTPUT_FILE, log_counts=True)

	print(f"Read {len(data)} rows from: {INPUT_FILE}")
	print(f"Finite events: {finite.sum()}")
	print(f"Selected events: {selected.sum()}")
	print(f"Not selected events: {not_selected.sum()}")
	print(
		"Thresholds: "
		f"amp > {AMP_THRESHOLD}, "
		f"amp <= {AMP_UPPER_THRESHOLD}, "
		f"tau_r_eff > {TAU_R_THRESHOLD}, "
		f"tau_r_eff <= {TAU_R_UPPER_THRESHOLD}, "
		f"tau_d_eff > {TAU_D_THRESHOLD}, "
		f"tau_d_eff <= {TAU_D_UPPER_THRESHOLD}"
	)
	print(f"Saved: {OUTPUT_FILE}")
	print(f"Saved: {LOG_OUTPUT_FILE}")


if __name__ == "__main__":
	main()
