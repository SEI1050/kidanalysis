from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


# =============================================================================
# SETTINGS
# =============================================================================
WAVEFORM_FILE = Path(
    "/Volumes/NO NAME/data/20260709/5.501GHz_z=8.0mm_x=4.4mm_first/"
    "wf_260709_175104_49.78Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "iqscan0709_temp_phase_overlay"
)

OUTPUT_FILE = OUTPUT_DIR / "pedestal_and_all_raw_tracks.png"

# pedestal を求める区間
BASELINE_WIDTH = 500
BASELINE_GAP = 100

# track の表示範囲（trigger基準）
TRACK_START_OFFSET = -200
TRACK_END_OFFSET = 2500

# raw track を間引くかどうか
TRACK_DECIMATION = 2

# ref_position が読めないときの fallback
DEFAULT_TRIGGER_FRACTION = 0.25


# =============================================================================
# LOAD
# =============================================================================
def find_channel_key(npz: np.lib.npyio.NpzFile, channel_name: str) -> str:
    exact_candidates = (
        channel_name,
        channel_name.lower(),
        channel_name.upper(),
        f"data_{channel_name}",
        f"waveform_{channel_name}",
    )

    for candidate in exact_candidates:
        if candidate in npz.files:
            arr = np.asarray(npz[candidate])
            if arr.ndim == 2:
                return candidate

    for key in npz.files:
        if channel_name.lower() in key.lower():
            arr = np.asarray(npz[key])
            if arr.ndim == 2:
                return key

    raise KeyError(
        f"{channel_name} に対応する2次元配列が見つかりません。\n"
        f"npz keys: {npz.files}"
    )


def orient_event_sample(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"2次元配列ではありません: shape={array.shape}")

    if array.shape[1] == 5000:
        return array

    if array.shape[0] == 5000:
        return array.T

    if array.shape[0] > array.shape[1]:
        return array.T

    return array


def read_trigger_index(npz: np.lib.npyio.NpzFile, n_samples: int) -> int:
    trigger_keys = (
        "ref_position",
        "reference_position",
        "trigger_position",
        "trigger_index",
    )

    for key in trigger_keys:
        if key not in npz.files:
            continue

        value_array = np.asarray(npz[key]).squeeze()
        if value_array.size != 1:
            continue

        value = float(value_array)

        if 0.0 <= value <= 1.0:
            trigger_index = round(value * n_samples)
        elif 1.0 < value <= 100.0:
            trigger_index = round(value / 100.0 * n_samples)
        else:
            trigger_index = round(value)

        return int(np.clip(trigger_index, 0, n_samples - 1))

    return int(DEFAULT_TRIGGER_FRACTION * n_samples)


def load_waveform(filepath: Path) -> tuple[np.ndarray, np.ndarray, int]:
    if not filepath.exists():
        raise FileNotFoundError(f"Waveform file not found:\n{filepath}")

    with np.load(filepath, allow_pickle=False) as npz:
        print("=" * 80)
        print(f"WAVEFORM FILE: {filepath}")
        print(f"npz keys: {npz.files}")

        ch0_key = find_channel_key(npz, "ch0")
        ch1_key = find_channel_key(npz, "ch1")

        ch0 = orient_event_sample(npz[ch0_key])
        ch1 = orient_event_sample(npz[ch1_key])

        if ch0.shape != ch1.shape:
            raise ValueError(
                "ch0とch1のshapeが一致しません。\n"
                f"ch0: {ch0.shape}\n"
                f"ch1: {ch1.shape}"
            )

        trigger_index = read_trigger_index(npz, ch0.shape[1])

    print(f"ch0 key       : {ch0_key}")
    print(f"ch1 key       : {ch1_key}")
    print(f"waveform shape: {ch0.shape}")
    print(f"trigger index : {trigger_index}")
    print("=" * 80)

    return ch0, ch1, trigger_index


# =============================================================================
# ANALYSIS
# =============================================================================
def calculate_ranges(trigger_index: int, n_samples: int) -> tuple[int, int, int, int]:
    baseline_end = max(1, trigger_index - BASELINE_GAP)
    baseline_start = max(0, baseline_end - BASELINE_WIDTH)

    if baseline_end <= baseline_start:
        baseline_start = 0
        baseline_end = min(BASELINE_WIDTH, n_samples)

    track_start = max(0, trigger_index + TRACK_START_OFFSET)
    track_end = min(n_samples, trigger_index + TRACK_END_OFFSET)

    if track_end <= track_start:
        track_start = 0
        track_end = n_samples

    return baseline_start, baseline_end, track_start, track_end


def calculate_pedestal(ch0: np.ndarray, ch1: np.ndarray, baseline_start: int, baseline_end: int):
    pedestal_ch0 = np.median(ch0[:, baseline_start:baseline_end], axis=1)
    pedestal_ch1 = np.median(ch1[:, baseline_start:baseline_end], axis=1)

    pedestal_center_ch0 = float(np.median(pedestal_ch0))
    pedestal_center_ch1 = float(np.median(pedestal_ch1))

    return pedestal_ch0, pedestal_ch1, pedestal_center_ch0, pedestal_center_ch1


def calculate_median_track(ch0: np.ndarray, ch1: np.ndarray, track_start: int, track_end: int):
    track_indices = np.arange(track_start, track_end, TRACK_DECIMATION)

    median_waveform_ch0 = np.median(ch0, axis=0)
    median_waveform_ch1 = np.median(ch1, axis=0)

    track_ch0 = median_waveform_ch0[track_indices]
    track_ch1 = median_waveform_ch1[track_indices]

    return track_indices, track_ch0, track_ch1


# =============================================================================
# PLOT
# =============================================================================
def make_plot(ch0: np.ndarray, ch1: np.ndarray, trigger_index: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_events, n_samples = ch0.shape

    baseline_start, baseline_end, track_start, track_end = calculate_ranges(
        trigger_index, n_samples
    )

    pedestal_ch0, pedestal_ch1, pedestal_center_ch0, pedestal_center_ch1 = calculate_pedestal(
        ch0, ch1, baseline_start, baseline_end
    )

    track_indices, median_track_ch0, median_track_ch1 = calculate_median_track(
        ch0, ch1, track_start, track_end
    )

    print(f"[baseline] {baseline_start}:{baseline_end}")
    print(f"[track]    {track_start}:{track_end} (decimation={TRACK_DECIMATION})")

    fig, ax = plt.subplots(figsize=(9, 8))

    # -------------------------------------------------------------------------
    # all raw tracks (each event)
    # -------------------------------------------------------------------------
    for i in range(n_events):
        event_track_ch0 = ch0[i, track_start:track_end:TRACK_DECIMATION]
        event_track_ch1 = ch1[i, track_start:track_end:TRACK_DECIMATION]

        ax.plot(
            event_track_ch0,
            event_track_ch1,
            linewidth=0.5,
            alpha=0.06,
            zorder=1,
            color="gray",
        )

    # 凡例用ダミー
    ax.plot([], [], linewidth=1.2, color="gray", alpha=0.6, label="All raw laser tracks")

    # -------------------------------------------------------------------------
    # pedestal (each event)
    # -------------------------------------------------------------------------
    ax.scatter(
        pedestal_ch0,
        pedestal_ch1,
        s=10,
        alpha=0.35,
        zorder=2,
        label="Pedestal (each event)",
    )

    ax.scatter(
        pedestal_center_ch0,
        pedestal_center_ch1,
        marker="X",
        s=130,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Pedestal median",
    )

    # -------------------------------------------------------------------------
    # median track with time gradient
    # -------------------------------------------------------------------------
    points = np.array([median_track_ch0, median_track_ch1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    time_values = np.linspace(0, 1, len(segments))

    lc = LineCollection(
        segments,
        array=time_values,
        cmap="viridis",
        linewidth=2.8,
        alpha=1.0,
        zorder=4,
    )
    ax.add_collection(lc)

    ax.scatter(
        median_track_ch0[0],
        median_track_ch1[0],
        marker="o",
        s=55,
        edgecolor="black",
        linewidth=0.7,
        zorder=6,
        label="Median track start",
    )

    ax.scatter(
        median_track_ch0[-1],
        median_track_ch1[-1],
        marker="s",
        s=45,
        edgecolor="black",
        linewidth=0.7,
        zorder=6,
        label="Median track end",
    )

    # pedestal 中心から最遠点
    dist2 = (
        (median_track_ch0 - pedestal_center_ch0) ** 2
        + (median_track_ch1 - pedestal_center_ch1) ** 2
    )
    far_idx = int(np.argmax(dist2))
    far_ch0 = float(median_track_ch0[far_idx])
    far_ch1 = float(median_track_ch1[far_idx])

    ax.scatter(
        far_ch0,
        far_ch1,
        marker="*",
        s=180,
        edgecolor="black",
        linewidth=0.8,
        zorder=7,
        label="Maximum displacement",
    )

    ax.annotate(
        "",
        xy=(far_ch0, far_ch1),
        xytext=(pedestal_center_ch0, pedestal_center_ch1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4},
        zorder=6,
    )

    # -------------------------------------------------------------------------
    # colorbar
    # -------------------------------------------------------------------------
    cbar = fig.colorbar(
        lc,
        ax=ax,
        pad=0.02,
        fraction=0.045,
    )
    cbar.set_label("Laser track time")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["start", "end"])

    # -------------------------------------------------------------------------
    # layout
    # -------------------------------------------------------------------------
    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.set_title(
        "Pedestal and all raw laser tracks in IQ plane\n"
        f"{WAVEFORM_FILE.parent.name}"
    )

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout(rect=[0.0, 0.12, 0.88, 1.0])

    fig.savefig(
        OUTPUT_FILE,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[saved] {OUTPUT_FILE}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    ch0, ch1, trigger_index = load_waveform(WAVEFORM_FILE)
    make_plot(ch0, ch1, trigger_index)


if __name__ == "__main__":
    main()