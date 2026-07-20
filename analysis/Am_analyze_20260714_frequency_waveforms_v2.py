from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


# ============================================================
# 1. ユーザー設定
# ============================================================

DATA_ROOT = Path("/Volumes/NO NAME/data/20260714")

# 解析したい周波数を手入力する。
# 例: "4.463GHz", "5.161GHz", "5.267GHz"
TARGET_FREQUENCY = "4.463GHz"

# 波形を何サンプルずつ平均して1 binにするか。
# 1ならbin化なし。
SAMPLE_BIN = 1

# IQ trackで使用するサンプルのstride。
# 5なら、0, 5, 10, 15, ... 番目を使用する。
TRACK_STRIDE = 5

# 各イベントの冒頭何割をpedestal計算に使用するか。
PEDESTAL_FRACTION = 0.10

# 個別表示する冒頭イベント数。
FIRST_N_EVENTS = 25

# 全イベント図に描画するイベント数。
# Noneなら全イベント。試運転時は100などに変更可能。
MAX_EVENTS_TO_PLOT = None

# 全イベント重ね書きの線設定。
OVERLAY_ALPHA = 0.08
OVERLAY_LINEWIDTH = 0.45

# 外れ値を除いてIQ表示範囲を決めたい場合は、
# 例: (0.1, 99.9)
# 全点を含める場合はNone。
IQ_LIMIT_PERCENTILES = None

OUTPUT_DPI = 200

# Trueなら、各データフォルダの既存PNGを解析前に削除する。
# 以前の 02_first25_iq_tracks.png が残って見える問題を防ぐ。
CLEAR_OLD_PNGS = True

OUTPUT_ROOT = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260714"
) / (
    f"{TARGET_FREQUENCY}"
    f"_bin{SAMPLE_BIN}"
    f"_trackstride{TRACK_STRIDE}"
)


# ============================================================
# 2. データ読み込み
# ============================================================

def read_optional_scalar(npz_data, key):
    """npz内のスカラー値を読み込む。存在しなければNone。"""
    if key not in npz_data:
        return None

    value = np.asarray(npz_data[key]).squeeze()

    if value.size != 1:
        return None

    return float(value)


def standardize_event_sample_shape(array, npts=None):
    """
    配列を (イベント数, サンプル数) にそろえる。
    """
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)

    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.ndim != 2:
        raise ValueError(
            "ch0/ch1は1次元または2次元を想定しています。"
            f" shape={array.shape}"
        )

    if npts is not None:
        npts = int(npts)

        if array.shape[1] == npts:
            return array

        if array.shape[0] == npts:
            return array.T

    # 通常はサンプル数の方がイベント数より多い。
    if array.shape[0] > array.shape[1]:
        array = array.T

    return array


def load_waveforms_from_folder(folder):
    """
    フォルダ以下の全npzからch0/ch1を読み込み、
    イベント方向に連結する。
    """
    npz_files = sorted(folder.rglob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"npzファイルがありません: {folder}"
        )

    ch0_list = []
    ch1_list = []
    sample_rates = []
    used_files = []

    for npz_file in npz_files:
        try:
            with np.load(npz_file, allow_pickle=False) as data:
                if "ch0" not in data or "ch1" not in data:
                    print(
                        "[skip] ch0/ch1がないため除外: "
                        f"{npz_file.name} keys={list(data.keys())}"
                    )
                    continue

                npts = read_optional_scalar(data, "npts")
                sample_rate = read_optional_scalar(
                    data,
                    "sample_rate",
                )

                ch0 = standardize_event_sample_shape(
                    data["ch0"],
                    npts=npts,
                )
                ch1 = standardize_event_sample_shape(
                    data["ch1"],
                    npts=npts,
                )

                if ch0.shape != ch1.shape:
                    raise ValueError(
                        "ch0とch1のshapeが異なります: "
                        f"ch0={ch0.shape}, ch1={ch1.shape}"
                    )

                ch0_list.append(ch0)
                ch1_list.append(ch1)
                used_files.append(npz_file)

                if sample_rate is not None:
                    sample_rates.append(sample_rate)

        except Exception as exc:
            print(f"[skip] 読み込み失敗: {npz_file}")
            print(
                f"       {type(exc).__name__}: {exc}"
            )

    if not ch0_list:
        raise RuntimeError(
            f"読み込めるch0/ch1データがありません: {folder}"
        )

    sample_lengths = [
        array.shape[1]
        for array in ch0_list
    ]
    min_samples = min(sample_lengths)

    if len(set(sample_lengths)) != 1:
        print(
            "[warning] サンプル数がファイル間で異なるため、"
            f"{min_samples} samplesに切りそろえます。"
        )

    ch0 = np.concatenate(
        [
            array[:, :min_samples]
            for array in ch0_list
        ],
        axis=0,
    )
    ch1 = np.concatenate(
        [
            array[:, :min_samples]
            for array in ch1_list
        ],
        axis=0,
    )

    sample_rate = None

    if sample_rates:
        sample_rate = sample_rates[0]

        for sr in sample_rates[1:]:
            if not np.isclose(
                sr,
                sample_rate,
                rtol=1.0e-9,
                atol=0.0,
            ):
                print(
                    "[warning] sample_rateがファイル間で異なります。"
                    f"最初の値 {sample_rate:g} Hz を使います。"
                )
                break

    return ch0, ch1, sample_rate, used_files


# ============================================================
# 3. 前処理
# ============================================================

def bin_waveforms(array, bin_size):
    """
    連続するbin_sizeサンプルを平均して1点にする。
    末尾の余りは切り捨てる。
    """
    if bin_size < 1:
        raise ValueError(
            "SAMPLE_BINは1以上にしてください。"
        )

    if bin_size == 1:
        return array

    n_events, n_samples = array.shape
    n_usable = (
        n_samples // bin_size
    ) * bin_size

    if n_usable == 0:
        raise ValueError(
            f"SAMPLE_BIN={bin_size} が"
            f"サンプル数={n_samples}より大きいです。"
        )

    trimmed = array[:, :n_usable]

    return trimmed.reshape(
        n_events,
        n_usable // bin_size,
        bin_size,
    ).mean(axis=2)


def stride_samples(array, stride):
    """サンプル軸をstrideで間引く。"""
    if stride < 1:
        raise ValueError(
            "TRACK_STRIDEは1以上にしてください。"
        )

    return array[:, ::stride]


def make_horizontal_axis(
    n_binned_samples,
    sample_rate,
    bin_size,
):
    """
    sample_rateがあれば時間[µs]、
    なければ元データ換算のsample indexを返す。
    """
    raw_sample_position = (
        np.arange(
            n_binned_samples,
            dtype=float,
        ) * bin_size
        + 0.5 * (bin_size - 1)
    )

    if (
        sample_rate is not None
        and sample_rate > 0
    ):
        time_us = (
            raw_sample_position
            / sample_rate
            * 1.0e6
        )
        return time_us, "Time [µs]"

    return (
        raw_sample_position,
        "Original sample index",
    )


def choose_events(array):
    """全イベント図で使用するイベントを選ぶ。"""
    if MAX_EVENTS_TO_PLOT is None:
        return array

    n_events = min(
        MAX_EVENTS_TO_PLOT,
        array.shape[0],
    )
    return array[:n_events]


def calculate_pedestal_subtracted_amplitude(
    ch0,
    ch1,
):
    """
    各イベントの冒頭PEDESTAL_FRACTIONを平均して、
    ch0とch1それぞれのpedestalとする。

    amplitude
      = sqrt(
          (ch0 - pedestal_ch0)^2
          + (ch1 - pedestal_ch1)^2
        )
    """
    n_samples = ch0.shape[1]

    n_pedestal = max(
        1,
        int(
            np.ceil(
                n_samples
                * PEDESTAL_FRACTION
            )
        ),
    )

    pedestal_ch0 = np.mean(
        ch0[:, :n_pedestal],
        axis=1,
        keepdims=True,
    )
    pedestal_ch1 = np.mean(
        ch1[:, :n_pedestal],
        axis=1,
        keepdims=True,
    )

    amplitude = np.hypot(
        ch0 - pedestal_ch0,
        ch1 - pedestal_ch1,
    )

    return (
        amplitude,
        pedestal_ch0,
        pedestal_ch1,
        n_pedestal,
    )


# ============================================================
# 4. プロット補助
# ============================================================

def finite_range(
    values,
    percentiles=None,
):
    """有限値だけから表示範囲を求める。"""
    values = np.asarray(values)
    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return -1.0, 1.0

    if percentiles is None:
        lower = float(
            np.min(finite_values)
        )
        upper = float(
            np.max(finite_values)
        )
    else:
        lower, upper = np.percentile(
            finite_values,
            percentiles,
        )
        lower = float(lower)
        upper = float(upper)

    if lower == upper:
        width = max(
            abs(lower) * 0.05,
            1.0e-12,
        )
        return (
            lower - width,
            upper + width,
        )

    padding = 0.04 * (
        upper - lower
    )

    return (
        lower - padding,
        upper + padding,
    )


def set_common_iq_limits(
    ax,
    ch0,
    ch1,
):
    """IQ図の表示範囲を設定する。"""
    x_min, x_max = finite_range(
        ch0,
        percentiles=IQ_LIMIT_PERCENTILES,
    )
    y_min, y_max = finite_range(
        ch1,
        percentiles=IQ_LIMIT_PERCENTILES,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def add_many_lines(
    ax,
    x,
    y,
    *,
    linewidth=0.45,
    alpha=0.08,
    color="tab:blue",
    chunk_size=100,
):
    """
    多数の線をLineCollectionで描画する。

    x, yがともに2次元:
        各行を1本のXY trackとして描画。

    xが1次元、yが2次元:
        共通x軸の波形を描画。
    """
    y = np.asarray(y)

    if y.ndim != 2:
        raise ValueError(
            "yは2次元を想定しています。"
            f" shape={y.shape}"
        )

    if np.ndim(x) == 1:
        x = np.asarray(x)

        if x.size != y.shape[1]:
            raise ValueError(
                "xとyのサンプル数が一致しません: "
                f"x={x.size}, y={y.shape}"
            )

        x_is_1d = True

    elif np.shape(x) == np.shape(y):
        x = np.asarray(x)
        x_is_1d = False

    else:
        raise ValueError(
            "xとyのshapeが一致しません: "
            f"x={np.shape(x)}, y={np.shape(y)}"
        )

    n_lines = y.shape[0]

    for start in range(
        0,
        n_lines,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_lines,
        )

        y_chunk = y[start:stop]

        if x_is_1d:
            x_chunk = np.broadcast_to(
                x,
                y_chunk.shape,
            )
        else:
            x_chunk = x[start:stop]

        segments = np.stack(
            (x_chunk, y_chunk),
            axis=2,
        ).astype(
            np.float32,
            copy=False,
        )

        collection = LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            rasterized=True,
        )
        ax.add_collection(collection)


def add_gradient_track(
    ax,
    x_track,
    y_track,
    *,
    cmap="viridis",
    linewidth=2.8,
):
    """
    1本のIQ trackを時系列グラデーションで描画する。
    """
    x_track = np.asarray(
        x_track,
        dtype=float,
    )
    y_track = np.asarray(
        y_track,
        dtype=float,
    )

    if (
        x_track.ndim != 1
        or y_track.ndim != 1
    ):
        raise ValueError(
            "x_trackとy_trackは"
            "1次元配列にしてください。"
        )

    if x_track.size != y_track.size:
        raise ValueError(
            "x_trackとy_trackの長さが"
            "一致しません。"
        )

    if x_track.size < 2:
        raise ValueError(
            "trackには2点以上必要です。"
        )

    points = np.column_stack(
        [x_track, y_track]
    )
    segments = np.stack(
        [points[:-1], points[1:]],
        axis=1,
    )

    # 0が冒頭、1が末尾。
    time_fraction = np.linspace(
        0.0,
        1.0,
        len(segments),
    )

    collection = LineCollection(
        segments,
        cmap=cmap,
        linewidths=linewidth,
        alpha=1.0,
    )
    collection.set_array(time_fraction)
    collection.set_clim(0.0, 1.0)

    ax.add_collection(collection)

    colormap = plt.get_cmap(cmap)

    ax.scatter(
        x_track[0],
        y_track[0],
        marker="o",
        s=42,
        color=colormap(0.0),
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="median-track start",
    )
    ax.scatter(
        x_track[-1],
        y_track[-1],
        marker="s",
        s=42,
        color=colormap(1.0),
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="median-track end",
    )

    return collection


def finish_figure(
    fig,
    output_file,
):
    """図を保存して閉じる。"""
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[saved] {output_file}")



def clear_old_output_pngs(output_directory):
    """
    以前の実行で作ったPNGを削除する。

    特に旧版の
        02_first25_iq_tracks.png
    が残り続けるのを防ぐ。
    """
    if not CLEAR_OLD_PNGS:
        return

    if not output_directory.exists():
        return

    for png_file in output_directory.glob("*.png"):
        png_file.unlink()
        print(f"[removed old] {png_file}")


# ============================================================
# 5. 各種プロット
# ============================================================

def plot_all_raw_iq_tracks(
    ch0,
    ch1,
    title,
    output_file,
):
    """
    全イベントのraw IQ trackをstrideで間引いて重ね書きする。

    さらに、各strideサンプル位置で
    ch0とch1のイベント中央値を求め、
    中央値trackを時系列グラデーションで描画する。
    """
    ch0_plot = choose_events(ch0)
    ch1_plot = choose_events(ch1)

    ch0_stride = stride_samples(
        ch0_plot,
        TRACK_STRIDE,
    )
    ch1_stride = stride_samples(
        ch1_plot,
        TRACK_STRIDE,
    )

    median_ch0 = np.median(
        ch0_stride,
        axis=0,
    )
    median_ch1 = np.median(
        ch1_stride,
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(8.8, 8.0)
    )

    add_many_lines(
        ax,
        ch0_stride,
        ch1_stride,
        linewidth=OVERLAY_LINEWIDTH,
        alpha=OVERLAY_ALPHA,
        color="tab:blue",
    )

    gradient_collection = add_gradient_track(
        ax,
        median_ch0,
        median_ch1,
        cmap="viridis",
        linewidth=2.8,
    )

    set_common_iq_limits(
        ax,
        ch0_stride,
        ch1_stride,
    )

    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.set_title(title)
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.grid(alpha=0.25)

    colorbar = fig.colorbar(
        gradient_collection,
        ax=ax,
        pad=0.02,
        fraction=0.046,
    )
    colorbar.set_label(
        "Median-track time order "
        "(0: start, 1: end)"
    )

    ax.text(
        0.02,
        0.98,
        f"events shown: {ch0_plot.shape[0]}\n"
        f"track stride: {TRACK_STRIDE}\n"
        "thick line: sample-wise median track",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.82,
            edgecolor="none",
        ),
    )

    ax.legend(loc="best")

    finish_figure(
        fig,
        output_file,
    )


def plot_first_waveform_grid(
    x_axis,
    waveform,
    x_label,
    channel_name,
    title,
    output_file,
):
    """
    ch0またはch1の冒頭FIRST_N_EVENTS波形を
    5列の一覧として表示する。
    """
    n_events = min(
        FIRST_N_EVENTS,
        waveform.shape[0],
    )

    if n_events == 0:
        return

    waveform_first = waveform[:n_events]

    n_columns = 5
    n_rows = int(
        np.ceil(
            n_events / n_columns
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            3.4 * n_columns,
            2.7 * n_rows,
        ),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    x_min, x_max = finite_range(
        x_axis
    )
    y_min, y_max = finite_range(
        waveform_first
    )

    for event_index, ax in enumerate(
        axes.flat
    ):
        if event_index >= n_events:
            ax.axis("off")
            continue

        ax.plot(
            x_axis,
            waveform_first[event_index],
            linewidth=0.9,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(
            f"event {event_index}",
            fontsize=10,
        )
        ax.grid(alpha=0.20)

        row = (
            event_index // n_columns
        )
        column = (
            event_index % n_columns
        )

        if row == n_rows - 1:
            ax.set_xlabel(x_label)

        if column == 0:
            ax.set_ylabel(channel_name)

    fig.suptitle(
        f"{title}\n"
        f"first {n_events} events: {channel_name} vs time",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    finish_figure(
        fig,
        output_file,
    )


def plot_all_waveforms(
    x_axis,
    waveform,
    x_label,
    channel_name,
    title,
    output_file,
):
    """全イベントのch0またはch1波形を重ね書きする。"""
    waveform_plot = choose_events(
        waveform
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    add_many_lines(
        ax,
        x_axis,
        waveform_plot,
        linewidth=OVERLAY_LINEWIDTH,
        alpha=OVERLAY_ALPHA,
    )

    ax.plot(
        x_axis,
        np.median(
            waveform_plot,
            axis=0,
        ),
        color="black",
        linewidth=2.0,
        label="median waveform",
    )

    x_min, x_max = finite_range(
        x_axis
    )
    y_min, y_max = finite_range(
        waveform_plot
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(x_label)
    ax.set_ylabel(channel_name)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax.text(
        0.02,
        0.98,
        f"events shown: {waveform_plot.shape[0]}\n"
        f"bin size: {SAMPLE_BIN}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
        ),
    )

    finish_figure(
        fig,
        output_file,
    )


def plot_pedestal_subtracted_amplitude(
    x_axis,
    amplitude,
    x_label,
    n_pedestal_samples,
    title,
    output_file,
):
    """
    pedestal除去後の振幅波形を全イベント重ね書きする。
    """
    amplitude_plot = choose_events(
        amplitude
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    add_many_lines(
        ax,
        x_axis,
        amplitude_plot,
        linewidth=OVERLAY_LINEWIDTH,
        alpha=OVERLAY_ALPHA,
    )

    ax.plot(
        x_axis,
        np.median(
            amplitude_plot,
            axis=0,
        ),
        color="black",
        linewidth=2.0,
        label="median amplitude",
    )

    x_min, x_max = finite_range(
        x_axis
    )
    y_min, y_max = finite_range(
        amplitude_plot
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(x_label)
    ax.set_ylabel(
        r"$\sqrt{(ch0-ped0)^2+(ch1-ped1)^2}$"
    )
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax.text(
        0.02,
        0.98,
        f"events shown: {amplitude_plot.shape[0]}\n"
        f"pedestal: first {PEDESTAL_FRACTION:.0%}"
        f" = {n_pedestal_samples} binned samples\n"
        f"bin size: {SAMPLE_BIN}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="none",
        ),
    )

    finish_figure(
        fig,
        output_file,
    )


# ============================================================
# 6. 1フォルダを解析
# ============================================================

def analyze_dataset_folder(folder):
    print()
    print("=" * 72)
    print(f"Analyzing: {folder}")
    print("=" * 72)

    (
        ch0_raw,
        ch1_raw,
        sample_rate,
        used_files,
    ) = load_waveforms_from_folder(
        folder
    )

    print(f"files          : {len(used_files)}")
    print(f"events         : {ch0_raw.shape[0]}")
    print(f"raw samples    : {ch0_raw.shape[1]}")
    print(f"sample rate    : {sample_rate}")

    ch0 = bin_waveforms(
        ch0_raw,
        SAMPLE_BIN,
    )
    ch1 = bin_waveforms(
        ch1_raw,
        SAMPLE_BIN,
    )

    print(f"binned samples : {ch0.shape[1]}")

    x_axis, x_label = make_horizontal_axis(
        n_binned_samples=ch0.shape[1],
        sample_rate=sample_rate,
        bin_size=SAMPLE_BIN,
    )

    dataset_output = (
        OUTPUT_ROOT / folder.name
    )
    dataset_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 旧版のfirst25 IQ図など、同じ出力先に残ったPNGを削除。
    clear_old_output_pngs(dataset_output)

    # pedestalフォルダとtrigフォルダの両方で作成。
    plot_all_raw_iq_tracks(
        ch0,
        ch1,
        title=(
            f"{folder.name}\n"
            "all raw IQ tracks "
            "+ median gradient track"
        ),
        output_file=(
            dataset_output
            / "01_raw_iq_tracks_all.png"
        ),
    )

    # *_pedestalはraw trackだけ。
    if folder.name.endswith(
        "_pedestal"
    ):
        return

    if "_trig_" not in folder.name:
        print(
            "[skip] pedestalでもtrigでもないため、"
            f"raw IQ trackだけ作成: {folder.name}"
        )
        return

    # 修正版:
    # 冒頭25イベントはIQ平面ではなく、
    # ch0波形とch1波形をそれぞれ一覧表示する。
    plot_first_waveform_grid(
        x_axis,
        ch0,
        x_label=x_label,
        channel_name="ch0",
        title=folder.name,
        output_file=(
            dataset_output
            / "02_first25_ch0_waveforms.png"
        ),
    )

    plot_first_waveform_grid(
        x_axis,
        ch1,
        x_label=x_label,
        channel_name="ch1",
        title=folder.name,
        output_file=(
            dataset_output
            / "03_first25_ch1_waveforms.png"
        ),
    )

    plot_all_waveforms(
        x_axis,
        ch0,
        x_label=x_label,
        channel_name="ch0",
        title=(
            f"{folder.name}\n"
            "all ch0 waveforms"
        ),
        output_file=(
            dataset_output
            / "04_ch0_waveforms_all.png"
        ),
    )

    plot_all_waveforms(
        x_axis,
        ch1,
        x_label=x_label,
        channel_name="ch1",
        title=(
            f"{folder.name}\n"
            "all ch1 waveforms"
        ),
        output_file=(
            dataset_output
            / "05_ch1_waveforms_all.png"
        ),
    )

    (
        amplitude,
        _,
        _,
        n_pedestal_samples,
    ) = calculate_pedestal_subtracted_amplitude(
        ch0,
        ch1,
    )

    plot_pedestal_subtracted_amplitude(
        x_axis,
        amplitude,
        x_label=x_label,
        n_pedestal_samples=n_pedestal_samples,
        title=(
            f"{folder.name}\n"
            "pedestal-subtracted amplitude waveforms"
        ),
        output_file=(
            dataset_output
            / "06_pedestal_subtracted_amplitude_all.png"
        ),
    )


# ============================================================
# 7. main
# ============================================================

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOTが存在しません: {DATA_ROOT}"
        )

    if SAMPLE_BIN < 1:
        raise ValueError(
            "SAMPLE_BINは1以上にしてください。"
        )

    if TRACK_STRIDE < 1:
        raise ValueError(
            "TRACK_STRIDEは1以上にしてください。"
        )

    if not (
        0.0
        < PEDESTAL_FRACTION
        <= 1.0
    ):
        raise ValueError(
            "PEDESTAL_FRACTIONは"
            "0より大きく1以下にしてください。"
        )

    target_folders = sorted(
        [
            path
            for path in DATA_ROOT.iterdir()
            if (
                path.is_dir()
                and path.name.startswith(
                    f"{TARGET_FREQUENCY}_"
                )
                and (
                    path.name.endswith(
                        "_pedestal"
                    )
                    or "_trig_" in path.name
                )
            )
        ],
        key=lambda path: (
            0
            if path.name.endswith(
                "_pedestal"
            )
            else 1,
            path.name,
        ),
    )

    if not target_folders:
        available_folders = sorted(
            path.name
            for path in DATA_ROOT.iterdir()
            if path.is_dir()
        )

        print("利用可能なフォルダ:")

        for name in available_folders:
            print(f"  - {name}")

        raise FileNotFoundError(
            "\n"
            f"TARGET_FREQUENCY={TARGET_FREQUENCY!r}"
            " に一致するpedestal/trigフォルダが"
            "見つかりません。"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"DATA_ROOT        : {DATA_ROOT}")
    print(f"TARGET_FREQUENCY : {TARGET_FREQUENCY}")
    print(f"SAMPLE_BIN       : {SAMPLE_BIN}")
    print(f"TRACK_STRIDE     : {TRACK_STRIDE}")
    print(f"OUTPUT_ROOT      : {OUTPUT_ROOT}")
    print("Target folders:")

    for folder in target_folders:
        print(f"  - {folder.name}")

    failed_folders = []

    for folder in target_folders:
        try:
            analyze_dataset_folder(
                folder
            )

        except Exception as exc:
            failed_folders.append(
                (folder, exc)
            )

            print()
            print(
                f"[error] {folder.name} の"
                "解析に失敗しました。"
            )
            print(
                f"        {type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 72)
    print("Finished")
    print("=" * 72)
    print(f"Output: {OUTPUT_ROOT}")

    if failed_folders:
        print("失敗したフォルダ:")

        for folder, exc in failed_folders:
            print(
                f"  - {folder.name}: "
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
