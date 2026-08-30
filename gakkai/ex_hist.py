"""Create the measurement histograms from an amp/tau CSV file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# python ex_hist.py data_0827_145027_combined_amp_tau_eff.csv \
#   --min-amp 0.01 \
#   --min-tau_r_eff 25 \
#   --min-tau_d_eff 100

HISTOGRAM_NAMES = [
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


def parse_args():
	parser = argparse.ArgumentParser(
		description="Create the same 3x3 histograms as ex_check.py."
	)
	parser.add_argument(
		"csv_file",
		type=Path,
		help="CSV file containing the measured quantities",
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Output image path (default: <csv stem>_hist.png)",
	)
	parser.add_argument(
		"--min-value",
		type=float,
		default=None,
		help="Common lower threshold for every histogram",
	)
	for name in HISTOGRAM_NAMES:
		parser.add_argument(
			f"--min-{name}",
			type=float,
			default=None,
			help=f"Lower threshold for {name}",
		)
	return parser.parse_args()


def main():
	args = parse_args()
	output_file = args.output or args.csv_file.with_name(
		args.csv_file.stem + "_hist.png"
	)

	parameters = pd.read_csv(args.csv_file)
	missing_names = [
		name for name in HISTOGRAM_NAMES if name not in parameters
	]
	if missing_names:
		names = ", ".join(missing_names)
		raise ValueError(f"CSV is missing required columns: {names}")

	figure, axes = plt.subplots(3, 3, figsize=(16, 9))
	for axis, name in zip(axes.ravel(), HISTOGRAM_NAMES):
		values = pd.to_numeric(parameters[name], errors="coerce")
		values = values.to_numpy(dtype=float)
		values = values[np.isfinite(values)]

		minimum = getattr(args, f"min_{name}")
		if minimum is None:
			minimum = args.min_value
		if minimum is not None:
			values = values[values >= minimum]

		if values.size:
			axis.hist(values, bins=100, color="C0")
		axis.set_xlabel(name)
		axis.set_ylabel("counts")
		axis.grid(alpha=0.3)

	figure.tight_layout()
	figure.savefig(output_file, dpi=150)
	plt.close(figure)

	print(f"Read {len(parameters)} rows from: {args.csv_file}")
	print(f"Saved: {output_file}")


if __name__ == "__main__":
	main()
