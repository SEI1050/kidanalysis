from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


# =============================================================================
# SETTINGS
# =============================================================================
IQSCAN_DIR = Path("/Volumes/NO NAME/data/20260709")

WAVEFORM_FILE = Path(
    "/Volumes/NO NAME/data/20260709/"
    "5.501GHz_z=8.0mm_x=4.4mm_first/"
    "wf_260709_175104_49.78Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "iqscan0709_temp_phase_overlay"
)

OUTPUT_FILE = OUTPUT_DIR / "iqscan_pedestal_laser_track.png"


# pedestal を求める区間
# trigger位置の直前 BASELINE_WIDTH サンプルを使用する
BASELINE_WIDTH = 500
BASELINE_GAP = 100

# レーザートラックとして表示する範囲
# trigger位置を基準にしたサンプル数
TRACK_START_OFFSET = -200
TRACK_END_OFFSET = 2500

# 軌跡が重い場合の間引き
TRACK_DECIMATION = 2

# ref_position が読み取れない場合の trigger位置
DEFAULT_TRIGGER_FRACTION = 0.25


# =============================================================================
# LOAD WAVEFORM
# =============================================================================
def find_channel_key(
    npz: np.lib.npyio.NpzFile,
    channel_name: str,
) -> str:
    """ch0/ch1に対応する2次元配列のキーを探す。"""

    exact_candidates = (
        channel_name,
        channel_name.lower(),
        channel_name.upper(),
        f"data_{channel_name}",
        f"waveform_{channel_name}",
    )

    for candidate in exact_candidates:
        if candidate in npz.files:
            array = np.asarray(npz[candidate])

            if array.ndim == 2:
                return candidate

    for key in npz.files:
        if channel_name.lower() not in key.lower():
            continue

        array = np.asarray(npz[key])

        if array.ndim == 2:
            return key

    raise KeyError(
        f"{channel_name} に対応する2次元配列が見つかりません。\n"
        f"npz keys: {npz.files}"
    )


def orient_event_sample(array: np.ndarray) -> np.ndarray:
    """
    配列を shape=(イベント数, サンプル数) にそろえる。
    今回は通常 (1000, 5000) を想定。
    """

    array = np.asarray(array, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"2次元配列ではありません: shape={array.shape}")

    # 5000サンプルの軸が見つかる場合
    if array.shape[1] == 5000:
        return array

    if array.shape[0] == 5000:
        return array.T

    # 一般にはサンプル数の方がイベント数より大きいと仮定
    if array.shape[0] > array.shape[1]:
        return array.T

    return array


def read_trigger_index(
    npz: np.lib.npyio.NpzFile,
    n_samples: int,
) -> int:
    """
    ref_positionをサンプル番号へ変換する。

    対応する値:
        0～1   : 全波形に対する割合
        1～100 : パーセント
        100超  : サンプル番号
    """

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


def load_waveform(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """波形npzからch0、ch1、trigger位置を読み込む。"""

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
# LOAD IQ SCANS
# =============================================================================
def extract_temperature(filename: str) -> float | None:
    """iq_3.62K.npzなどのファイル名から温度を取得する。"""

    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*[Kk]",
        filename,
    )

    if match is None:
        return None

    return float(match.group(1))


def load_iq_scans(
    directory: Path,
    excluded_file: Path,
) -> list[dict[str, object]]:
    """
    指定ディレクトリ以下から、dd[:,0:3]を含むnpzをすべて探す。

    dd[:, 0] : frequency
    dd[:, 1] : ch0
    dd[:, 2] : ch1
    """

    if not directory.exists():
        raise FileNotFoundError(f"IQ scan directory not found:\n{directory}")

    scans: list[dict[str, object]] = []

    for filepath in sorted(directory.rglob("*.npz")):
        if filepath.resolve() == excluded_file.resolve():
            continue

        try:
            with np.load(filepath, allow_pickle=False) as npz:
                if "dd" not in npz.files:
                    continue

                dd = np.asarray(npz["dd"], dtype=float)

        except Exception as error:
            print(f"[skip] {filepath}: {error}")
            continue

        if dd.ndim != 2 or dd.shape[1] < 3:
            print(f"[skip] invalid dd shape: {filepath}, shape={dd.shape}")
            continue

        temperature = extract_temperature(filepath.name)

        scans.append(
            {
                "filepath": filepath,
                "temperature": temperature,
                "frequency": dd[:, 0],
                "ch0": dd[:, 1],
                "ch1": dd[:, 2],
            }
        )

    scans.sort(
        key=lambda scan: (
            scan["temperature"] is None,
            scan["temperature"] if scan["temperature"] is not None else 0.0,
            str(scan["filepath"]),
        )
    )

    print(f"[IQ scan] found {len(scans)} file(s)")

    for scan in scans:
        print(
            f"  {scan['filepath']} "
            f"shape=({len(scan['frequency'])}, 3)"
        )

    return scans


# =============================================================================
# ANALYSIS
# =============================================================================
def calculate_pedestal_and_track(
    ch0: np.ndarray,
    ch1: np.ndarray,
    trigger_index: int,
) -> dict[str, np.ndarray | float | int]:
    """pedestal点と全イベント中央値のレーザー軌跡を求める。"""

    n_events, n_samples = ch0.shape

    baseline_end = max(1, trigger_index - BASELINE_GAP)
    baseline_start = max(0, baseline_end - BASELINE_WIDTH)

    if baseline_end <= baseline_start:
        baseline_start = 0
        baseline_end = min(BASELINE_WIDTH, n_samples)

    # イベントごとのpedestal
    pedestal_ch0 = np.median(
        ch0[:, baseline_start:baseline_end],
        axis=1,
    )
    pedestal_ch1 = np.median(
        ch1[:, baseline_start:baseline_end],
        axis=1,
    )

    # pedestal分布全体の代表値
    pedestal_center_ch0 = float(np.median(pedestal_ch0))
    pedestal_center_ch1 = float(np.median(pedestal_ch1))

    # イベント方向に中央値を取り、代表的なレーザー波形にする
    median_waveform_ch0 = np.median(ch0, axis=0)
    median_waveform_ch1 = np.median(ch1, axis=0)

    track_start = max(0, trigger_index + TRACK_START_OFFSET)
    track_end = min(n_samples, trigger_index + TRACK_END_OFFSET)

    if track_end <= track_start:
        track_start = 0
        track_end = n_samples

    track_indices = np.arange(
        track_start,
        track_end,
        TRACK_DECIMATION,
    )

    track_ch0 = median_waveform_ch0[track_indices]
    track_ch1 = median_waveform_ch1[track_indices]

    # pedestal中心から最も遠いトラック上の点
    distance_squared = (
        (track_ch0 - pedestal_center_ch0) ** 2
        + (track_ch1 - pedestal_center_ch1) ** 2
    )

    peak_local_index = int(np.argmax(distance_squared))
    peak_sample_index = int(track_indices[peak_local_index])

    peak_ch0 = float(track_ch0[peak_local_index])
    peak_ch1 = float(track_ch1[peak_local_index])

    print(
        f"[baseline] samples: "
        f"{baseline_start}:{baseline_end}"
    )
    print(
        f"[pedestal median] "
        f"ch0={pedestal_center_ch0:.6g}, "
        f"ch1={pedestal_center_ch1:.6g}"
    )
    print(
        f"[laser farthest point] "
        f"sample={peak_sample_index}, "
        f"ch0={peak_ch0:.6g}, "
        f"ch1={peak_ch1:.6g}"
    )

    return {
        "pedestal_ch0": pedestal_ch0,
        "pedestal_ch1": pedestal_ch1,
        "pedestal_center_ch0": pedestal_center_ch0,
        "pedestal_center_ch1": pedestal_center_ch1,
        "track_ch0": track_ch0,
        "track_ch1": track_ch1,
        "track_indices": track_indices,
        "peak_sample_index": peak_sample_index,
        "peak_ch0": peak_ch0,
        "peak_ch1": peak_ch1,
        "track_start_ch0": float(track_ch0[0]),
        "track_start_ch1": float(track_ch1[0]),
        "track_end_ch0": float(track_ch0[-1]),
        "track_end_ch1": float(track_ch1[-1]),
    }


# =============================================================================
# PLOT
# =============================================================================
def make_plot(
    scans: list[dict[str, object]],
    waveform_result: dict[str, np.ndarray | float | int],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # -------------------------------------------------------------------------
    # IQ scans
    # -------------------------------------------------------------------------
    for scan in scans:
        scan_ch0 = np.asarray(scan["ch0"])
        scan_ch1 = np.asarray(scan["ch1"])
        frequency = np.asarray(scan["frequency"])
        temperature = scan["temperature"]
        filepath = scan["filepath"]

        if temperature is not None:
            label = f"IQ scan: {temperature:.3f} K"
        else:
            label = f"IQ scan: {Path(filepath).stem}"

        line, = ax.plot(
            scan_ch0,
            scan_ch1,
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            alpha=0.85,
            label=label,
            zorder=2,
        )

        # scan開始周波数
        ax.scatter(
            scan_ch0[0],
            scan_ch1[0],
            marker="s",
            s=35,
            color=line.get_color(),
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        ax.annotate(
            f"{frequency[0] / 1e9:.6f} GHz",
            xy=(scan_ch0[0], scan_ch1[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=line.get_color(),
        )

    # -------------------------------------------------------------------------
    # Pedestal distribution
    # -------------------------------------------------------------------------
    pedestal_ch0 = np.asarray(waveform_result["pedestal_ch0"])
    pedestal_ch1 = np.asarray(waveform_result["pedestal_ch1"])

    pedestal_center_ch0 = float(
        waveform_result["pedestal_center_ch0"]
    )
    pedestal_center_ch1 = float(
        waveform_result["pedestal_center_ch1"]
    )

    ax.scatter(
        pedestal_ch0,
        pedestal_ch1,
        s=12,
        alpha=0.25,
        label="Pedestal: each event",
        zorder=4,
    )

    ax.scatter(
        pedestal_center_ch0,
        pedestal_center_ch1,
        marker="X",
        s=130,
        edgecolor="black",
        linewidth=0.8,
        label="Pedestal median",
        zorder=7,
    )

    # -------------------------------------------------------------------------
    # Median laser track
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Median laser track (time-gradient)
    # -------------------------------------------------------------------------
    track_ch0 = np.asarray(waveform_result["track_ch0"])
    track_ch1 = np.asarray(waveform_result["track_ch1"])
    track_indices = np.asarray(waveform_result["track_indices"])

    # 線分ごとに色をつけるための準備
    points = np.array([track_ch0, track_ch1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 時系列に沿った値（0 -> 1）
    time_values = np.linspace(0, 1, len(segments))

    lc = LineCollection(
        segments,
        array=time_values,
        cmap="viridis",   # シンプルで見やすい
        linewidth=2.5,
        alpha=0.95,
        zorder=5,
    )

    ax.add_collection(lc)

    # 見やすさのため、始点と終点だけ明示
    ax.scatter(
        track_ch0[0],
        track_ch1[0],
        marker="o",
        s=55,
        edgecolor="black",
        linewidth=0.7,
        label="Track start",
        zorder=6,
    )

    ax.scatter(
        track_ch0[-1],
        track_ch1[-1],
        marker="s",
        s=45,
        edgecolor="black",
        linewidth=0.7,
        label="Track end",
        zorder=6,
    )

    # カラーバー（時系列）
    cbar = fig.colorbar(
        lc,
        ax=ax,
        pad=0.02,
        fraction=0.045,
    )

    cbar.set_label("Laser track time")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["start", "end"])

    ax.scatter(
        float(waveform_result["track_start_ch0"]),
        float(waveform_result["track_start_ch1"]),
        marker="o",
        s=55,
        edgecolor="black",
        linewidth=0.7,
        label="Track start",
        zorder=6,
    )

    peak_ch0 = float(waveform_result["peak_ch0"])
    peak_ch1 = float(waveform_result["peak_ch1"])

    ax.scatter(
        peak_ch0,
        peak_ch1,
        marker="*",
        s=180,
        edgecolor="black",
        linewidth=0.8,
        label="Maximum displacement",
        zorder=8,
    )

    # pedestal中心から最大変位点へのベクトル
    ax.annotate(
        "",
        xy=(peak_ch0, peak_ch1),
        xytext=(pedestal_center_ch0, pedestal_center_ch1),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
        },
        zorder=6,
    )

    # -------------------------------------------------------------------------
    # Figure settings
    # -------------------------------------------------------------------------
    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")

    ax.set_title(
        "IQ scan, pedestal and median laser-signal track\n"
        f"{WAVEFORM_FILE.parent.name}"
    )

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

        # 凡例はプロットの下側へ配置
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=8,
        frameon=True,
    )

    # 下側の凡例スペースを確保
    fig.subplots_adjust(
        right=0.88,
        bottom=0.22,
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

    scans = load_iq_scans(
        directory=IQSCAN_DIR,
        excluded_file=WAVEFORM_FILE,
    )

    if len(scans) == 0:
        print(
            "[warning] ddを含むIQ scanファイルが見つかりません。\n"
            "pedestalとlaser trackのみをプロットします。"
        )

    waveform_result = calculate_pedestal_and_track(
        ch0=ch0,
        ch1=ch1,
        trigger_index=trigger_index,
    )

    make_plot(
        scans=scans,
        waveform_result=waveform_result,
    )


if __name__ == "__main__":
    main()