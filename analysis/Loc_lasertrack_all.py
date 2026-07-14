from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection


# =============================================================================
# SETTINGS
# =============================================================================
BASE_DIR = Path("/Volumes/NO NAME/data/20260709")

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "all_folders_pedestal_track_pdf"
)
OUTPUT_PDF = OUTPUT_DIR / "pedestal_and_laser_tracks_all_folders.pdf"

# pedestal 計算区間
BASELINE_WIDTH = 500
BASELINE_GAP = 100

# track 描画範囲（trigger基準）
TRACK_START_OFFSET = -200
TRACK_END_OFFSET = 2500

# track の間引き
TRACK_DECIMATION = 2

# ref_position が見つからない時
DEFAULT_TRIGGER_FRACTION = 0.25

# raw track の見た目
RAW_TRACK_COLOR = "gray"
RAW_TRACK_ALPHA = 0.05
RAW_TRACK_LINEWIDTH = 0.45

# pedestal 点
PEDESTAL_ALPHA = 0.35
PEDESTAL_SIZE = 10


# =============================================================================
# LOAD HELPERS
# =============================================================================
def find_channel_key(npz: np.lib.npyio.NpzFile, channel_name: str) -> str:
    candidates = (
        channel_name,
        channel_name.lower(),
        channel_name.upper(),
        f"data_{channel_name}",
        f"waveform_{channel_name}",
    )

    for key in candidates:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.ndim == 2:
                return key

    for key in npz.files:
        if channel_name.lower() in key.lower():
            arr = np.asarray(npz[key])
            if arr.ndim == 2:
                return key

    raise KeyError(
        f"{channel_name} に対応する2次元配列が見つかりません。"
        f" npz keys = {npz.files}"
    )


def orient_event_sample(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"2次元配列ではありません: shape={arr.shape}")

    if arr.shape[1] == 5000:
        return arr
    if arr.shape[0] == 5000:
        return arr.T

    if arr.shape[0] > arr.shape[1]:
        return arr.T

    return arr


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

        value_arr = np.asarray(npz[key]).squeeze()
        if value_arr.size != 1:
            continue

        value = float(value_arr)

        if 0.0 <= value <= 1.0:
            trigger_index = round(value * n_samples)
        elif 1.0 < value <= 100.0:
            trigger_index = round(value / 100.0 * n_samples)
        else:
            trigger_index = round(value)

        return int(np.clip(trigger_index, 0, n_samples - 1))

    return int(DEFAULT_TRIGGER_FRACTION * n_samples)


def load_single_waveform_npz(filepath: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(filepath, allow_pickle=False) as npz:
        ch0_key = find_channel_key(npz, "ch0")
        ch1_key = find_channel_key(npz, "ch1")

        ch0 = orient_event_sample(npz[ch0_key])
        ch1 = orient_event_sample(npz[ch1_key])

        if ch0.shape != ch1.shape:
            raise ValueError(
                f"ch0/ch1 shape mismatch in {filepath}\n"
                f"ch0: {ch0.shape}, ch1: {ch1.shape}"
            )

        trigger_index = read_trigger_index(npz, ch0.shape[1])

    return ch0, ch1, trigger_index


def load_folder_waveforms(folder: Path) -> tuple[np.ndarray, np.ndarray, int, list[Path]]:
    """
    フォルダ内の全 npz を読み込み、イベント方向に結合する。
    trigger_index は最初のファイルのものを採用。
    """

    npz_files = sorted(folder.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"npz file not found in {folder}")

    ch0_list = []
    ch1_list = []
    trigger_list = []
    sample_shapes = []

    for filepath in npz_files:
        try:
            ch0, ch1, trigger_index = load_single_waveform_npz(filepath)
        except Exception as error:
            print(f"[skip] {filepath.name}: {error}")
            continue

        ch0_list.append(ch0)
        ch1_list.append(ch1)
        trigger_list.append(trigger_index)
        sample_shapes.append(ch0.shape[1])

    if len(ch0_list) == 0:
        raise RuntimeError(f"有効な波形npzがありません: {folder}")

    # サンプル数が一致しているものだけ使う
    common_n_samples = max(set(sample_shapes), key=sample_shapes.count)

    filtered_ch0 = []
    filtered_ch1 = []
    filtered_trigger = []

    for ch0, ch1, trig in zip(ch0_list, ch1_list, trigger_list):
        if ch0.shape[1] != common_n_samples:
            print(
                f"[skip due to sample mismatch] folder={folder.name}, "
                f"shape={ch0.shape}"
            )
            continue
        filtered_ch0.append(ch0)
        filtered_ch1.append(ch1)
        filtered_trigger.append(trig)

    ch0_all = np.concatenate(filtered_ch0, axis=0)
    ch1_all = np.concatenate(filtered_ch1, axis=0)

    # trigger は中央値
    trigger_index = int(round(np.median(filtered_trigger)))

    return ch0_all, ch1_all, trigger_index, npz_files


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


def calculate_pedestal(
    ch0: np.ndarray,
    ch1: np.ndarray,
    baseline_start: int,
    baseline_end: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    pedestal_ch0 = np.median(ch0[:, baseline_start:baseline_end], axis=1)
    pedestal_ch1 = np.median(ch1[:, baseline_start:baseline_end], axis=1)

    pedestal_center_ch0 = float(np.median(pedestal_ch0))
    pedestal_center_ch1 = float(np.median(pedestal_ch1))

    return pedestal_ch0, pedestal_ch1, pedestal_center_ch0, pedestal_center_ch1


def calculate_median_track(
    ch0: np.ndarray,
    ch1: np.ndarray,
    track_start: int,
    track_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_indices = np.arange(track_start, track_end, TRACK_DECIMATION)

    median_waveform_ch0 = np.median(ch0, axis=0)
    median_waveform_ch1 = np.median(ch1, axis=0)

    median_track_ch0 = median_waveform_ch0[track_indices]
    median_track_ch1 = median_waveform_ch1[track_indices]

    return track_indices, median_track_ch0, median_track_ch1


# =============================================================================
# PLOT
# =============================================================================
def plot_folder_page(
    pdf: PdfPages,
    folder: Path,
    ch0: np.ndarray,
    ch1: np.ndarray,
    trigger_index: int,
    npz_files: list[Path],
) -> None:
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

    fig, ax = plt.subplots(figsize=(9.5, 8))

    # -------------------------------------------------------------------------
    # all raw tracks
    # -------------------------------------------------------------------------
    for i in range(n_events):
        event_track_ch0 = ch0[i, track_start:track_end:TRACK_DECIMATION]
        event_track_ch1 = ch1[i, track_start:track_end:TRACK_DECIMATION]

        ax.plot(
            event_track_ch0,
            event_track_ch1,
            color=RAW_TRACK_COLOR,
            alpha=RAW_TRACK_ALPHA,
            linewidth=RAW_TRACK_LINEWIDTH,
            zorder=1,
        )

    ax.plot([], [], color=RAW_TRACK_COLOR, alpha=0.6, linewidth=1.0, label="All raw laser tracks")

    # -------------------------------------------------------------------------
    # pedestal
    # -------------------------------------------------------------------------
    ax.scatter(
        pedestal_ch0,
        pedestal_ch1,
        s=PEDESTAL_SIZE,
        alpha=PEDESTAL_ALPHA,
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
    # median track with gradient
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

    # -------------------------------------------------------------------------
    # farthest point from pedestal median
    # -------------------------------------------------------------------------
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
    title = (
        f"{folder.name}\n"
        f"events={n_events}, npz_files={len(npz_files)}, trigger={trigger_index}, "
        f"baseline={baseline_start}:{baseline_end}, track={track_start}:{track_end}"
    )
    ax.set_title(title, fontsize=11)

    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout(rect=[0.0, 0.10, 0.88, 1.0])

    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subfolders = [p for p in sorted(BASE_DIR.iterdir()) if p.is_dir()]

    if len(subfolders) == 0:
        raise FileNotFoundError(f"サブフォルダが見つかりません: {BASE_DIR}")

    print(f"[info] found {len(subfolders)} folders")

    success_count = 0

    with PdfPages(OUTPUT_PDF) as pdf:
        for folder in subfolders:
            print("=" * 100)
            print(f"[folder] {folder}")

            try:
                ch0, ch1, trigger_index, npz_files = load_folder_waveforms(folder)

                print(f"  used npz files : {len(npz_files)}")
                print(f"  total events   : {ch0.shape[0]}")
                print(f"  samples/event  : {ch0.shape[1]}")
                print(f"  trigger index  : {trigger_index}")

                plot_folder_page(
                    pdf=pdf,
                    folder=folder,
                    ch0=ch0,
                    ch1=ch1,
                    trigger_index=trigger_index,
                    npz_files=npz_files,
                )
                success_count += 1

            except Exception as error:
                print(f"[skip folder] {folder.name}: {error}")

    print("=" * 100)
    print(f"[done] pages saved: {success_count}")
    print(f"[saved] {OUTPUT_PDF}")


if __name__ == "__main__":
    main()