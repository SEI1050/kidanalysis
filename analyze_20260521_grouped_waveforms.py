from __future__ import annotations

"""
Grouped waveform overlay analysis for the 2026-05-21 / 2026-05-22 Americium KID data.

This script extends `analyze_20260521_americium.py` by classifying events in each run by:
  1) whether the projected waveform crosses the baseline after the pulse, and
  2) the IQ distance between baseline and peak (used here as an alpha-energy proxy).

It then saves overlay plots for each class.

Place this file in:
    /Users/kubokosei/software/kidanalysis/analyze_20260521_grouped_waveforms.py

Run:
    cd /Users/kubokosei/software/kidanalysis
    python analyze_20260521_grouped_waveforms.py
"""

from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_20260521_americium import (
    INPUT_DIR,
    OUTPUT_DIR,
    RUN_INFO,
    discover_npz_files,
    group_waveform_files,
    ensure_dir,
    sanitize_name,
    baseline_slice,
    search_slice,
    load_waveform_npz,
    first_pass,
    finite_stats,
    EVENT_PEAK_HALF_WIDTH_SAMPLES,
    PEAK_AVG_HALF_WIDTH_SAMPLES,
)


# =============================================================================
# SETTINGS
# =============================================================================

# Baseline-crossing判定
BASE_CROSS_SIGMA = 3.0
BASE_CROSS_MIN_CONSECUTIVE = 8
BASE_CROSS_POST_OFFSET_SAMPLES = 10
BASE_CROSS_POST_WINDOW_SAMPLES = 2500

# IQ距離(dist_iq)によるenergy bin数
ENERGY_N_BINS = 3
ENERGY_BIN_NAMES = ["low", "mid", "high"]

# 波形重ね書き用（peak合わせ）
ALIGN_PRE_NS = 400.0
ALIGN_POST_NS = 1600.0
MAX_OVERLAY_WAVEFORMS_PER_CLASS = 80
MAX_PLOT_EVENTS_PER_CLASS = 80

# 1 runあたりの分類図出力をスキップしない最小イベント数
MIN_EVENTS_TO_PLOT_CLASS = 5


# =============================================================================
# HELPERS
# =============================================================================

def has_consecutive_true(mask: np.ndarray, min_run: int) -> bool:
    """Return True if `mask` contains at least `min_run` consecutive True values."""
    if min_run <= 1:
        return bool(np.any(mask))
    run = 0
    for v in mask:
        if v:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def make_energy_bins(dist_iq: np.ndarray, n_bins: int = ENERGY_N_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Make per-run quantile bins for IQ distance, returning (edges, bin_index)."""
    x = np.asarray(dist_iq, dtype=float)
    idx = np.full(x.shape, -1, dtype=int)
    finite = np.isfinite(x)
    xf = x[finite]

    if xf.size == 0:
        return np.array([np.nan, np.nan]), idx

    if xf.size < n_bins:
        # Too few events: make fewer effective bins by linear split.
        xmin = float(np.nanmin(xf))
        xmax = float(np.nanmax(xf))
        if xmax <= xmin:
            idx[finite] = 0
            return np.array([xmin, xmax]), idx
        edges = np.linspace(xmin, xmax, n_bins + 1)
    else:
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(xf, q)
        # If duplicated edges appear, fall back to linear spacing.
        if np.unique(edges).size < 2:
            idx[finite] = 0
            return np.array([float(edges[0]), float(edges[-1])]), idx
        if np.unique(edges).size < len(edges):
            xmin = float(np.nanmin(xf))
            xmax = float(np.nanmax(xf))
            if xmax <= xmin:
                idx[finite] = 0
                return np.array([xmin, xmax]), idx
            edges = np.linspace(xmin, xmax, n_bins + 1)

    inner = edges[1:-1]
    idx[finite] = np.digitize(xf, inner, right=False)
    idx[idx < 0] = 0
    idx[idx >= n_bins] = n_bins - 1
    return np.asarray(edges, dtype=float), idx


def reservoir_append(store: list[np.ndarray], arr: np.ndarray, max_keep: int, seen_count: int, rng: np.random.Generator) -> None:
    """Reservoir sampling for storing a bounded number of example waveforms."""
    if len(store) < max_keep:
        store.append(arr.copy())
        return
    j = int(rng.integers(0, seen_count + 1))
    if j < max_keep:
        store[j] = arr.copy()


def align_waveform(y: np.ndarray, peak_idx: int, pre_samples: int, post_samples: int) -> np.ndarray:
    """Return a fixed-length peak-aligned waveform, padded with NaN when needed."""
    out = np.full(pre_samples + post_samples + 1, np.nan, dtype=float)
    src0 = max(0, peak_idx - pre_samples)
    src1 = min(len(y), peak_idx + post_samples + 1)
    dst0 = pre_samples - (peak_idx - src0)
    dst1 = dst0 + (src1 - src0)
    out[dst0:dst1] = y[src0:src1]
    return out


def nanmean_std_from_sums(sum_y: np.ndarray, sumsq_y: np.ndarray, count_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(sum_y.shape, np.nan, dtype=float)
    std = np.full(sum_y.shape, np.nan, dtype=float)
    good = count_y > 0
    mean[good] = sum_y[good] / count_y[good]
    good2 = count_y > 1
    var = np.full(sum_y.shape, np.nan, dtype=float)
    var[good2] = (sumsq_y[good2] - (sum_y[good2] ** 2) / count_y[good2]) / (count_y[good2] - 1)
    var[good2] = np.maximum(var[good2], 0.0)
    std[good2] = np.sqrt(var[good2])
    std[good & ~good2] = 0.0
    return mean, std


# =============================================================================
# EVENT METRICS + CLASSIFICATION
# =============================================================================

def compute_event_metrics_and_classes(paths: list[Path], fp) -> dict[str, np.ndarray]:
    """
    Recompute per-event metrics and classify each event.

    Classification:
      cross_flag = 1 if the projected waveform goes below baseline by > BASE_CROSS_SIGMA * noise
                   for at least BASE_CROSS_MIN_CONSECUTIVE consecutive samples after the pulse.
      energy_bin  = tercile bin of dist_iq = sqrt((peak0-base0)^2 + (peak1-base1)^2).
    """
    n_samples = fp.n_samples
    time_ns = fp.time_ns
    base_sl = baseline_slice(n_samples)
    s_sl = search_slice(n_samples)

    pk0 = max(s_sl.start, fp.peak_idx - EVENT_PEAK_HALF_WIDTH_SAMPLES)
    pk1 = min(s_sl.stop, fp.peak_idx + EVENT_PEAK_HALF_WIDTH_SAMPLES + 1)
    event_search = slice(pk0, pk1)

    rows: dict[str, list[float]] = {
        "base0": [], "base1": [],
        "peak0": [], "peak1": [],
        "d0": [], "d1": [],
        "dist_iq": [],
        "amp_proj": [],
        "noise_proj": [],
        "snr_proj": [],
        "t_peak_ns": [],
        "peak_idx": [],
        "post_min_proj": [],
        "cross_flag": [],
    }

    for path in paths:
        wf = load_waveform_npz(path)
        if wf is None:
            continue
        ch0 = wf.ch0[:, :n_samples]
        ch1 = wf.ch1[:, :n_samples]
        if ch0.shape[1] < n_samples or ch1.shape[1] < n_samples:
            continue

        base0 = np.median(ch0[:, base_sl], axis=1)
        base1 = np.median(ch1[:, base_sl], axis=1)
        y0 = ch0 - base0[:, None]
        y1 = ch1 - base1[:, None]
        proj = y0 * fp.response_u0 + y1 * fp.response_u1

        local = proj[:, event_search]
        local_arg = np.argmax(local, axis=1)
        amp_proj = local[np.arange(local.shape[0]), local_arg]
        event_peak_idx = event_search.start + local_arg

        peak0 = np.empty(ch0.shape[0], dtype=float)
        peak1 = np.empty(ch1.shape[0], dtype=float)
        dist_iq = np.empty(ch0.shape[0], dtype=float)
        noise = np.std(proj[:, base_sl], axis=1, ddof=1)
        post_min = np.empty(ch0.shape[0], dtype=float)
        cross = np.zeros(ch0.shape[0], dtype=int)

        for i, pi in enumerate(event_peak_idx):
            a = max(0, pi - PEAK_AVG_HALF_WIDTH_SAMPLES)
            b = min(n_samples, pi + PEAK_AVG_HALF_WIDTH_SAMPLES + 1)
            peak0[i] = np.mean(ch0[i, a:b])
            peak1[i] = np.mean(ch1[i, a:b])
            d0 = peak0[i] - base0[i]
            d1 = peak1[i] - base1[i]
            dist_iq[i] = math.hypot(d0, d1)

            p0 = min(n_samples, pi + BASE_CROSS_POST_OFFSET_SAMPLES)
            p1 = min(n_samples, pi + BASE_CROSS_POST_WINDOW_SAMPLES)
            post = proj[i, p0:p1]
            if post.size == 0:
                post_min[i] = np.nan
                cross[i] = 0
            else:
                post_min[i] = float(np.min(post))
                thr = -BASE_CROSS_SIGMA * max(float(noise[i]), 1e-15)
                cross[i] = int(has_consecutive_true(post < thr, BASE_CROSS_MIN_CONSECUTIVE))

        with np.errstate(divide="ignore", invalid="ignore"):
            snr = amp_proj / noise

        rows["base0"].extend(base0.tolist())
        rows["base1"].extend(base1.tolist())
        rows["peak0"].extend(peak0.tolist())
        rows["peak1"].extend(peak1.tolist())
        rows["d0"].extend((peak0 - base0).tolist())
        rows["d1"].extend((peak1 - base1).tolist())
        rows["dist_iq"].extend(dist_iq.tolist())
        rows["amp_proj"].extend(amp_proj.tolist())
        rows["noise_proj"].extend(noise.tolist())
        rows["snr_proj"].extend(snr.tolist())
        rows["t_peak_ns"].extend(time_ns[event_peak_idx].tolist())
        rows["peak_idx"].extend(event_peak_idx.astype(float).tolist())
        rows["post_min_proj"].extend(post_min.tolist())
        rows["cross_flag"].extend(cross.astype(float).tolist())

    out: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    out["n_events"] = np.asarray([len(rows["amp_proj"])], dtype=float)

    edges, energy_bin = make_energy_bins(out["dist_iq"], n_bins=ENERGY_N_BINS)
    out["energy_bin"] = energy_bin.astype(float)
    out["energy_edges"] = edges.astype(float)

    cross_label = np.where(out["cross_flag"].astype(int) == 1, "cross", "no_cross")
    energy_label = np.full(energy_bin.shape, "unknown", dtype=object)
    for i, name in enumerate(ENERGY_BIN_NAMES[:ENERGY_N_BINS]):
        energy_label[energy_bin == i] = name

    out["cross_label"] = cross_label
    out["energy_label"] = energy_label
    out["combined_label"] = np.asarray([
        f"{c}_{e}" if e != "unknown" else c
        for c, e in zip(cross_label, energy_label)
    ], dtype=object)
    return out


# =============================================================================
# OVERLAY BUILDING
# =============================================================================

def build_overlay_stats(paths: list[Path], fp, metrics: dict[str, np.ndarray], label_key: str) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    n_samples = fp.n_samples
    base_sl = baseline_slice(n_samples)
    s_sl = search_slice(n_samples)
    pk0 = max(s_sl.start, fp.peak_idx - EVENT_PEAK_HALF_WIDTH_SAMPLES)
    pk1 = min(s_sl.stop, fp.peak_idx + EVENT_PEAK_HALF_WIDTH_SAMPLES + 1)
    event_search = slice(pk0, pk1)

    labels = np.asarray(metrics[label_key])
    pre_samples = int(round(ALIGN_PRE_NS * 1e-9 * fp.sample_rate_hz))
    post_samples = int(round(ALIGN_POST_NS * 1e-9 * fp.sample_rate_hz))
    time_aligned_us = (np.arange(pre_samples + post_samples + 1) - pre_samples) / fp.sample_rate_hz * 1e6

    unique_labels = [str(x) for x in np.unique(labels)]
    out: dict[str, dict[str, object]] = {}
    for label in unique_labels:
        out[label] = {
            "count": 0,
            "sum": np.zeros_like(time_aligned_us, dtype=float),
            "sumsq": np.zeros_like(time_aligned_us, dtype=float),
            "n_per_sample": np.zeros_like(time_aligned_us, dtype=float),
            "samples": [],
            "event_amp": [],
            "event_dist": [],
            "event_snr": [],
        }

    rng = np.random.default_rng(20260521)
    global_idx = 0

    for path in paths:
        wf = load_waveform_npz(path)
        if wf is None:
            continue
        ch0 = wf.ch0[:, :n_samples]
        ch1 = wf.ch1[:, :n_samples]
        if ch0.shape[1] < n_samples or ch1.shape[1] < n_samples:
            continue

        base0 = np.median(ch0[:, base_sl], axis=1)
        base1 = np.median(ch1[:, base_sl], axis=1)
        y0 = ch0 - base0[:, None]
        y1 = ch1 - base1[:, None]
        proj = y0 * fp.response_u0 + y1 * fp.response_u1

        local = proj[:, event_search]
        local_arg = np.argmax(local, axis=1)
        event_peak_idx = event_search.start + local_arg

        n_evt = ch0.shape[0]
        for i in range(n_evt):
            if global_idx >= len(labels):
                break
            label = str(labels[global_idx])
            stat = out[label]
            aligned = align_waveform(proj[i], int(event_peak_idx[i]), pre_samples, post_samples)
            valid = np.isfinite(aligned)
            stat["sum"][valid] += aligned[valid]
            stat["sumsq"][valid] += aligned[valid] ** 2
            stat["n_per_sample"][valid] += 1.0
            stat["count"] += 1
            stat["event_amp"].append(float(metrics["amp_proj"][global_idx]))
            stat["event_dist"].append(float(metrics["dist_iq"][global_idx]))
            stat["event_snr"].append(float(metrics["snr_proj"][global_idx]))
            reservoir_append(stat["samples"], aligned, MAX_OVERLAY_WAVEFORMS_PER_CLASS, int(stat["count"]), rng)
            global_idx += 1

    return time_aligned_us, out


# =============================================================================
# PLOTTING
# =============================================================================

def plot_overlay_panels(
    out_path: Path,
    time_us: np.ndarray,
    stats: dict[str, dict[str, object]],
    group_order: list[str],
    title: str,
) -> None:
    nonempty = [g for g in group_order if g in stats and int(stats[g]["count"]) >= MIN_EVENTS_TO_PLOT_CLASS]
    if not nonempty:
        return

    n = len(nonempty)
    ncols = 1 if n <= 3 else 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, group in zip(axes_flat, nonempty):
        stat = stats[group]
        samples = list(stat["samples"])
        for y in samples[:MAX_PLOT_EVENTS_PER_CLASS]:
            ax.plot(time_us, y, alpha=0.12, linewidth=0.8)

        mean_y, std_y = nanmean_std_from_sums(stat["sum"], stat["sumsq"], stat["n_per_sample"])
        ax.plot(time_us, mean_y, linewidth=2.2, label="mean projected waveform")
        finite_mean = np.isfinite(mean_y) & np.isfinite(std_y)
        if np.any(finite_mean):
            ax.fill_between(time_us[finite_mean], mean_y[finite_mean] - std_y[finite_mean], mean_y[finite_mean] + std_y[finite_mean], alpha=0.18)

        amp_stat = finite_stats(np.asarray(stat["event_amp"], dtype=float))
        dist_stat = finite_stats(np.asarray(stat["event_dist"], dtype=float))
        snr_stat = finite_stats(np.asarray(stat["event_snr"], dtype=float))
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(group)
        ax.set_xlabel("time from event peak [µs]")
        ax.set_ylabel("projected amplitude [raw unit]")
        ax.text(
            0.98, 0.95,
            "\n".join([
                f"N = {int(stat['count'])}",
                f"dist_iq med = {dist_stat['median']:.4g}",
                f"amp_proj med = {amp_stat['median']:.4g}",
                f"SNR med = {snr_stat['median']:.4g}",
            ]),
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=9,
        )
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_classification_csv(run_dir: Path, run_name: str, metrics: dict[str, np.ndarray]) -> Path:
    out = run_dir / f"{sanitize_name(run_name)}_event_classification.csv"
    keys = [
        "base0", "base1", "peak0", "peak1", "d0", "d1",
        "dist_iq", "amp_proj", "noise_proj", "snr_proj",
        "t_peak_ns", "peak_idx", "post_min_proj", "cross_flag", "energy_bin",
        "cross_label", "energy_label", "combined_label",
    ]
    n = len(metrics["amp_proj"])
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            row = []
            for k in keys:
                v = metrics[k][i]
                row.append(v)
            w.writerow(row)
    return out


def save_group_summary_csv(run_dir: Path, run_name: str, stats: dict[str, dict[str, object]], kind: str) -> Path:
    out = run_dir / f"{sanitize_name(run_name)}_{kind}_group_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "count", "dist_iq_median", "amp_proj_median", "snr_median"])
        for g, s in stats.items():
            amp = finite_stats(np.asarray(s["event_amp"], dtype=float))
            dist = finite_stats(np.asarray(s["event_dist"], dtype=float))
            snr = finite_stats(np.asarray(s["event_snr"], dtype=float))
            w.writerow([g, int(s["count"]), dist["median"], amp["median"], snr["median"]])
    return out


# =============================================================================
# MAIN
# =============================================================================

def analyze_one_run(run_name: str, paths: list[Path]) -> dict[str, object] | None:
    info = RUN_INFO.get(run_name, {})
    label = str(info.get("label", run_name))
    print("\n" + "=" * 90)
    print(f"RUN: {run_name}")
    print(f"LABEL: {label}")
    print(f"FILES: {len(paths)}")
    print("=" * 90)

    fp = first_pass(paths)
    if fp is None:
        print("[skip] no compatible waveform files")
        return None

    metrics = compute_event_metrics_and_classes(paths, fp)
    n_evt = len(metrics["amp_proj"])
    if n_evt == 0:
        print("[skip] zero events after loading")
        return None

    run_dir = OUTPUT_DIR / "runs" / sanitize_name(run_name)
    ensure_dir(run_dir)
    class_csv = save_classification_csv(run_dir, run_name, metrics)

    # Plot 1: split only by baseline crossing.
    cross_order = ["no_cross", "cross"]
    time_us, cross_stats = build_overlay_stats(paths, fp, metrics, label_key="cross_label")
    plot_overlay_panels(
        run_dir / f"{sanitize_name(run_name)}_overlay_projected_by_crossing.png",
        time_us,
        cross_stats,
        cross_order,
        title=f"Projected waveform overlays by baseline crossing: {label}",
    )
    save_group_summary_csv(run_dir, run_name, cross_stats, kind="crossing")

    # Plot 2: split only by IQ-distance (energy proxy).
    energy_order = ENERGY_BIN_NAMES[:ENERGY_N_BINS]
    time_us2, energy_stats = build_overlay_stats(paths, fp, metrics, label_key="energy_label")
    plot_overlay_panels(
        run_dir / f"{sanitize_name(run_name)}_overlay_projected_by_energy.png",
        time_us2,
        energy_stats,
        energy_order,
        title=f"Projected waveform overlays by IQ-distance bin: {label}",
    )
    save_group_summary_csv(run_dir, run_name, energy_stats, kind="energy")

    # Plot 3: combined classification.
    comb_order = [f"{c}_{e}" for c in cross_order for e in energy_order]
    time_us3, comb_stats = build_overlay_stats(paths, fp, metrics, label_key="combined_label")
    plot_overlay_panels(
        run_dir / f"{sanitize_name(run_name)}_overlay_projected_by_crossing_and_energy.png",
        time_us3,
        comb_stats,
        comb_order,
        title=f"Projected waveform overlays by crossing × IQ-distance: {label}",
    )
    save_group_summary_csv(run_dir, run_name, comb_stats, kind="combined")

    cross_arr = metrics["cross_flag"].astype(int)
    frac_cross = float(np.mean(cross_arr == 1)) if cross_arr.size > 0 else np.nan
    dist = finite_stats(np.asarray(metrics["dist_iq"], dtype=float))
    amp = finite_stats(np.asarray(metrics["amp_proj"], dtype=float))

    print(f"[saved] event classification CSV: {class_csv}")
    print(f"[summary] N={n_evt}, cross_fraction={frac_cross:.3f}, dist_iq_med={dist['median']:.5g}, amp_proj_med={amp['median']:.5g}")

    return {
        "run": run_name,
        "label": label,
        "n_events": n_evt,
        "cross_fraction": frac_cross,
        "dist_iq_median": dist["median"],
        "amp_proj_median": amp["median"],
        "classification_csv": str(class_csv),
        "overlay_crossing": str(run_dir / f"{sanitize_name(run_name)}_overlay_projected_by_crossing.png"),
        "overlay_energy": str(run_dir / f"{sanitize_name(run_name)}_overlay_projected_by_energy.png"),
        "overlay_combined": str(run_dir / f"{sanitize_name(run_name)}_overlay_projected_by_crossing_and_energy.png"),
    }


def save_run_level_summary(rows: list[dict[str, object]]) -> Path:
    out = OUTPUT_DIR / "grouped_waveform_overlay_summary.csv"
    keys = [
        "run", "label", "n_events", "cross_fraction", "dist_iq_median", "amp_proj_median",
        "classification_csv", "overlay_crossing", "overlay_energy", "overlay_combined",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    return out


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print(f"[input]  {INPUT_DIR}")
    print(f"[output] {OUTPUT_DIR}")

    files = discover_npz_files(INPUT_DIR)
    groups = group_waveform_files(INPUT_DIR, files)
    print(f"[discover] waveform run groups: {len(groups)}")

    rows: list[dict[str, object]] = []
    for run_name, paths in groups.items():
        result = analyze_one_run(run_name, paths)
        if result is not None:
            rows.append(result)

    summary_path = save_run_level_summary(rows)
    print("\nDONE")
    print(f"summary: {summary_path}")
    print("Main outputs are saved into each run directory, for example:")
    print(f"  {OUTPUT_DIR / 'runs' / 'data_0522_135424'}")
    print("Recommended first look:")
    print(f"  {OUTPUT_DIR / 'runs' / 'data_0522_135424' / 'data_0522_135424_overlay_projected_by_crossing_and_energy.png'}")


if __name__ == "__main__":
    main()
