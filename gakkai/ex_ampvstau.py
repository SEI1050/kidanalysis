"""Create 2D histograms of amplitude and effective time constants."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# The CSV file is in the same directory as this script.
INPUT_FILE = Path(__file__).with_name(
	"data_0827_145027_combined_amp_tau_eff.csv"
)

# The output file will also be saved in the same directory.
OUTPUT_FILE = INPUT_FILE.with_name(
	INPUT_FILE.stem + "_2d_hist.png"
)

AMP_MIN = 0.01
TAU_R_EFF_MIN = 15.0
TAU_D_EFF_MIN = 50.0


def main():
	# Read the CSV file into a pandas DataFrame.
	data = pd.read_csv(INPUT_FILE)

	# Convert the three columns to numbers.
	# Invalid values become NaN, which can be removed below.
	amplitude = pd.to_numeric(data["amp"], errors="coerce")
	tau_r_eff = pd.to_numeric(data["tau_r_eff"], errors="coerce")
	tau_d_eff = pd.to_numeric(data["tau_d_eff"], errors="coerce")

	# Keep only rows where all three values are finite.
	valid_for_rise = (
		np.isfinite(amplitude)
		& np.isfinite(tau_r_eff)
		& ~(
			(amplitude < AMP_MIN)
			& (tau_r_eff < TAU_R_EFF_MIN)
		)
	)
	valid_for_decay = (
		np.isfinite(amplitude)
		& np.isfinite(tau_d_eff)
		& ~(
			(amplitude < AMP_MIN)
			& (tau_d_eff < TAU_D_EFF_MIN)
		)
	)

	if not valid_for_rise.any() and not valid_for_decay.any():
		raise ValueError("No finite amp/tau values were found in the CSV.")

	# Make two plots: amplitude vs tau_r_eff and amplitude vs tau_d_eff.
	figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

	rise_histogram = axes[0].hist2d(
		tau_r_eff[valid_for_rise],
		amplitude[valid_for_rise],
		bins=50,
		cmap="viridis",
	)
	axes[0].set_title("Amplitude vs tau_r_eff")
	axes[0].set_xlabel("tau_r_eff")
	axes[0].set_ylabel("amp")
	axes[0].grid(alpha=0.25)
	figure.colorbar(rise_histogram[3], ax=axes[0], label="counts")

	decay_histogram = axes[1].hist2d(
		tau_d_eff[valid_for_decay],
		amplitude[valid_for_decay],
		bins=50,
		cmap="viridis",
	)
	axes[1].set_title("Amplitude vs tau_d_eff")
	axes[1].set_xlabel("tau_d_eff")
	axes[1].set_ylabel("amp")
	axes[1].grid(alpha=0.25)
	figure.colorbar(decay_histogram[3], ax=axes[1], label="counts")

	figure.savefig(OUTPUT_FILE, dpi=150)
	plt.close(figure)

	print(f"Read {len(data)} rows from: {INPUT_FILE}")
	print(f"Valid rows for tau_r_eff: {valid_for_rise.sum()}")
	print(f"Valid rows for tau_d_eff: {valid_for_decay.sum()}")
	print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
