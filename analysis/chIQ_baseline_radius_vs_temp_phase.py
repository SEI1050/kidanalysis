#!/usr/bin/env python3
"""Plot baseline radius sqrt(ch0**2 + ch1**2) against 1-Hz temperature phase.

The file is written for the 2026-05-27 KID waveform data.  Temperature phase is
constructed from the event order and the laser repetition rate, because this
NPZ format has no per-event timestamps.  Therefore phase=0 is the first event
in the file; only relative phase within the 1-Hz cycle is meaningful unless a
separate temperature trigger is available.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================
INPUT_PATH = Path(
    "/Volumes/NO NAME/data/20260527/"
    "5.476GHz_z=7.5mm_x=3.4mm/wf_260527_142822_49.73Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260527/"
    "baseline_radius_temp_phase_5p476GHz_z7p5mm_x3p4mm"
)

# The filename records 49.73 Hz.  Change this only when the acquisition rate
# is known more accurately from the measurement log.
LASER_REPETITION_HZ = 49.73
TEMPERATURE_FREQUENCY_HZ = 1.0
PHASE_OFFSET_CYCLES = 0.0  # Set after a temperature-trigger measurement, if available.

# Number of phase bins used for the mean, median, and standard error.
N_PHASE_BINS = 50

# Baseline window: use this many samples immediately before the pulse.
# A guard interval excludes samples directly adjacent to the pulse onset.
BASELINE_WINDOW_SAMPLES = 1000
PRETRIGGER_GUARD_SAMPLES = 200

# Fallback pulse position when ``ref_position`` is absent from the NPZ file.
FALLBACK_PULSE_POSITION_FRACTION = 0.50


# =============================================================================
# DATA HANDLING
# =============================================================================
def find_array(npz: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> np.ndarray:
    """Return one array using case-insensitive exact key matching."""
    lower_to_original = {key.lower(): key for key in npz.files}
    for candidate in candidates:
        key = lower_to_original.get(candidate.lower())
        if key is not None:
            return np.asarray(npz[key])

    raise KeyError(
        "Could not find any of "
        f"{candidates}. Available keys are: {', '.join(npz.files)}"
    )


def as_event_by_sample(array: np.ndarray, name: str) -> np.ndarray:
    """Return waveform data with shape (n_events, n_samples)."""
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}.")

    # For this dataset n_samples is normally 5000 and n_events is normally 500.
    # Put the larger dimension on the sample axis.
    if array.shape[0] > array.shape[1]:
        array = array.T
    return array


def infer_pulse_index(npz: np.lib.npyio.NpzFile, n_samples: int) -> int:
    """Read ref_position when present; otherwise use the configured fallback."""
    lower_to_original = {key.lower(): key for key in npz.files}
    for candidate in ("ref_position", "refposition", "reference_position"):
        key = lower_to_original.get(candidate)
        if key is None:
            continue

        value = float(np.nanmedian(np.asarray(npz[key], dtype=float)))
        # Accept either a fraction (0--1) or percent (0--100).
        fraction = value if 0.0 < value <= 1.0 else value / 100.0
        if 0.05 <= fraction <= 0.95:
            return int(round(fraction * n_samples))

    return int(round(FALLBACK_PULSE_POSITION_FRACTION * n_samples))


def baseline_radius(ch0: np.ndarray, ch1: np.ndarray, pulse_index: int) -> np.ndarray:
    """Return one baseline radius value per event."""
    baseline_stop = pulse_index - PRETRIGGER_GUARD_SAMPLES
    baseline_start = max(0, baseline_stop - BASELINE_WINDOW_SAMPLES)

    if baseline_stop - baseline_start < 10:
        raise ValueError(
            "The inferred baseline window is too short. Adjust "
            "BASELINE_WINDOW_SAMPLES, PRETRIGGER_GUARD_SAMPLES, or the pulse position."
        )

    radius = np.hypot(ch0[:, baseline_start:baseline_stop], ch1[:, baseline_start:baseline_stop])
    return np.nanmedian(radius, axis=1)


def phase_bin_statistics(phase: np.ndarray, value: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Calculate binned mean, median, sample count, and standard error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.clip(np.digitize(phase, edges, right=False) - 1, 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=int)
    means = np.full(n_bins, np.nan)
    medians = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)

    for i in range(n_bins):
        values = value[bin_index == i]
        values = values[np.isfinite(values)]
        counts[i] = values.size
        if values.size == 0:
            continue

        means[i] = np.mean(values)
        medians[i] = np.median(values)
        if values.size >= 2:
            sems[i] = np.std(values, ddof=1) / np.sqrt(values.size)

    return {
        "edges": edges,
        "centers": centers,
        "counts": counts,
        "mean": means,
        "median": medians,
        "sem": sems,
    }


def extrema_rows(stats: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    """Return min/max phase points for both binned mean and binned median."""
    rows: list[dict[str, float | int | str]] = []
    for metric in ("mean", "median"):
        values = stats[metric]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue

        valid_indices = np.flatnonzero(valid)
        min_index = valid_indices[np.argmin(values[valid])]
        max_index = valid_indices[np.argmax(values[valid])]
        for kind, index in (("minimum", min_index), ("maximum", max_index)):
            rows.append(
                {
                    "metric": metric,
                    "extremum": kind,
                    "phase_cycles": float(stats["centers"][index]),
                    "phase_degrees": float(360.0 * stats["centers"][index]),
                    "value": float(values[index]),
                    "sem": float(stats["sem"][index]),
                    "n_events": int(stats["counts"][index]),
                    "phase_bin": int(index),
                }
            )
    return rows


# =============================================================================
# OUTPUT
# =============================================================================
def save_csvs(output_dir: Path, stats: dict[str, np.ndarray], extrema: list[dict[str, float | int | str]]) -> None:
    binned_path = output_dir / "baseline_radius_phase_binned.csv"
    with binned_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase_bin",
                "phase_left_cycles",
                "phase_center_cycles",
                "phase_center_degrees",
                "n_events",
                "baseline_radius_median",
                "baseline_radius_mean",
                "baseline_radius_sem",
            ]
        )
        for i in range(stats["centers"].size):
            writer.writerow(
                [
                    i,
                    stats["edges"][i],
                    stats["centers"][i],
                    360.0 * stats["centers"][i],
                    stats["counts"][i],
                    stats["median"][i],
                    stats["mean"][i],
                    stats["sem"][i],
                ]
            )

    extrema_path = output_dir / "baseline_radius_phase_extrema.csv"
    fields = ["metric", "extremum", "phase_cycles", "phase_degrees", "value", "sem", "n_events", "phase_bin"]
    with extrema_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(extrema)


def make_plot(
    output_dir: Path,
    phase: np.ndarray,
    radius: np.ndarray,
    stats: dict[str, np.ndarray],
    extrema: list[dict[str, float | int | str]],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    # Raw per-event baseline-radius values.
    ax.scatter(
        phase,
        radius,
        s=10,
        alpha=0.28,
        rasterized=True,
        label="raw event baseline median",
    )

    # Median and mean of the event-level baseline-radius values in each phase bin.
    ax.plot(
        stats["centers"],
        stats["median"],
        "o-",
        ms=4,
        lw=1.4,
        label="binned median",
    )
    ax.errorbar(
        stats["centers"],
        stats["mean"],
        yerr=stats["sem"],
        fmt="s-",
        ms=3.5,
        lw=1.2,
        capsize=2.5,
        label="binned mean ± SEM",
    )

    # Mark extrema for both binned estimators.  The annotation gives a phase
    # in cycles, where 0 and 1 correspond to the same temperature phase.
    marker_for = {
        ("mean", "minimum"): "v",
        ("mean", "maximum"): "^",
        ("median", "minimum"): "<",
        ("median", "maximum"): ">",
    }
    for row in extrema:
        metric = str(row["metric"])
        kind = str(row["extremum"])
        x = float(row["phase_cycles"])
        y = float(row["value"])
        ax.scatter(
            [x],
            [y],
            s=70,
            marker=marker_for[(metric, kind)],
            zorder=6,
            label=f"{metric} {kind}: {x:.3f} cycle",
        )

    ax.set(
        xlabel="1-Hz temperature phase (cycles; arbitrary zero)",
        ylabel=r"baseline median of $\sqrt{\mathrm{ch0}^2+\mathrm{ch1}^2}$",
        title="Baseline radius versus temperature phase",
        xlim=(0.0, 1.0),
    )
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8.5, ncol=2)

    fig.savefig(output_dir / "baseline_radius_vs_temperature_phase.png", dpi=220)
    fig.savefig(output_dir / "baseline_radius_vs_temperature_phase.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(INPUT_PATH, allow_pickle=False) as npz:
        ch0 = as_event_by_sample(find_array(npz, ("ch0", "channel0", "i", "I")), "ch0")
        ch1 = as_event_by_sample(find_array(npz, ("ch1", "channel1", "q", "Q")), "ch1")
        if ch0.shape != ch1.shape:
            raise ValueError(f"ch0 shape {ch0.shape} and ch1 shape {ch1.shape} do not match.")
        pulse_index = infer_pulse_index(npz, ch0.shape[1])

    radius = baseline_radius(ch0, ch1, pulse_index)
    event_index = np.arange(radius.size)
    time_s = event_index / LASER_REPETITION_HZ
    phase = (TEMPERATURE_FREQUENCY_HZ * time_s + PHASE_OFFSET_CYCLES) % 1.0

    stats = phase_bin_statistics(phase, radius, N_PHASE_BINS)
    extrema = extrema_rows(stats)

    save_csvs(OUTPUT_DIR, stats, extrema)
    make_plot(OUTPUT_DIR, phase, radius, stats, extrema)

    baseline_stop = pulse_index - PRETRIGGER_GUARD_SAMPLES
    baseline_start = max(0, baseline_stop - BASELINE_WINDOW_SAMPLES)
    print(f"Input: {INPUT_PATH}")
    print(f"Waveform shape: {ch0.shape} (events, samples)")
    print(f"Baseline window: samples [{baseline_start}:{baseline_stop})")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Extrema (phase is in 1-Hz-cycle units):")
    for row in extrema:
        print(
            f"  {row['metric']:6s} {row['extremum']:7s}: "
            f"phase={row['phase_cycles']:.4f}, value={row['value']:.8g}, "
            f"SEM={row['sem']:.3g}, N={row['n_events']}"
        )


if __name__ == "__main__":
    main()
