from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

DATA_PATH = Path(
    "/Volumes/NO NAME/data/20260527/"
    "5.476GHz_z=7.5mm_x=3.4mm/"
    "wf_260527_142822_49.73Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260527/"
    "iq_phase_baseline_peak_vectors"
)

# レーザーは 50 Hz と仮定
LASER_REPETITION_HZ = 50.0

# 温度変化は 1 Hz と仮定し、1 周期を 10 分割
TEMP_PERIOD_S = 1.0
N_TEMP_PHASE_BINS = 10

# 最初の何割を baseline とするか
BASELINE_FRACTION = 0.20

# 平均波形から peak 時刻を自動検出するとき、
# waveform 冒頭の誤検出を避けるため探索を開始する位置
PULSE_SEARCH_START_FRACTION = 0.20

# ADC のサンプリングレート。peak 周辺の中央値を取る幅に使う。
SAMPLE_RATE_HZ = 2.5e9
PEAK_HALF_WINDOW_NS = 10.0

# phase bin 9 → 0 の周期を閉じるか
CLOSE_TEMP_CYCLE = True

# zoom 図で表示する「規格化した peak 方向ベクトル」の長さ
# baseline 点群の広がりに対する比率
ZOOM_NORMALIZED_PULSE_ARROW_FRACTION = 0.16

SAVE_DPI = 250


# =============================================================================
# UTILS
# =============================================================================

def find_channel_key(npz: np.lib.npyio.NpzFile, target: str) -> str:
    """npz 内から ch0 / ch1 に対応するキーを探す。"""
    keys = list(npz.keys())
    lower_to_original = {key.lower(): key for key in keys}

    candidates = [
        target,
        target.lower(),
        target.upper(),
        f"{target}_data",
        f"data_{target}",
        f"{target}_waveform",
        f"waveform_{target}",
    ]

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    for key in keys:
        if target.lower() in key.lower():
            return key

    raise KeyError(
        f"'{target}' に対応する配列を見つけられませんでした。\n"
        f"npz keys: {keys}"
    )


def ensure_event_by_sample(array: np.ndarray, name: str) -> np.ndarray:
    """配列を (n_event, n_sample) 形式へそろえる。"""
    array = np.asarray(array, dtype=float)

    if array.ndim != 2:
        raise ValueError(
            f"{name} の次元が想定外です: shape={array.shape}\n"
            "2 次元の waveform 配列を想定しています。"
        )

    # 通常は n_sample = 5000 程度で n_event より大きい。
    # (n_sample, n_event) で入っていた場合は転置する。
    if array.shape[0] > array.shape[1]:
        array = array.T

    return array


def add_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    vector: np.ndarray,
    *,
    color,
    width: float,
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    """データ座標系で長さを保った矢印を描く。"""
    ax.quiver(
        start[0],
        start[1],
        vector[0],
        vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=width,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.5,
        alpha=alpha,
        zorder=zorder,
    )


def padded_limits(values: np.ndarray, pad_fraction: float = 0.14) -> tuple[float, float]:
    """データ範囲に少し余白をつけた axis limit を返す。"""
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = vmax - vmin

    if not np.isfinite(span) or span <= 0:
        span = max(abs(vmin), 1.0) * 0.05

    pad = span * pad_fraction
    return vmin - pad, vmax + pad


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def load_baseline_and_peak() -> dict[str, np.ndarray | int]:
    """各イベントの baseline IQ と peak IQ を作る。"""
    with np.load(DATA_PATH, allow_pickle=False) as npz:
        ch0_key = find_channel_key(npz, "ch0")
        ch1_key = find_channel_key(npz, "ch1")

        ch0 = ensure_event_by_sample(npz[ch0_key], ch0_key)
        ch1 = ensure_event_by_sample(npz[ch1_key], ch1_key)

    n_event = min(ch0.shape[0], ch1.shape[0])
    n_sample = min(ch0.shape[1], ch1.shape[1])

    ch0 = ch0[:n_event, :n_sample]
    ch1 = ch1[:n_event, :n_sample]

    baseline_end = max(10, int(round(n_sample * BASELINE_FRACTION)))
    baseline_end = min(baseline_end, n_sample - 2)

    # 各イベントの baseline 中央値
    baseline_ch0 = np.nanmedian(ch0[:, :baseline_end], axis=1)
    baseline_ch1 = np.nanmedian(ch1[:, :baseline_end], axis=1)

    baseline_iq = np.column_stack([baseline_ch0, baseline_ch1])

    # 全イベントの中央値波形から、IQ 平面上で最も baseline から離れる時刻を peak とする
    mean_ch0 = np.nanmedian(ch0, axis=0)
    mean_ch1 = np.nanmedian(ch1, axis=0)

    reference_baseline_ch0 = np.nanmedian(mean_ch0[:baseline_end])
    reference_baseline_ch1 = np.nanmedian(mean_ch1[:baseline_end])

    distance_from_baseline = np.hypot(
        mean_ch0 - reference_baseline_ch0,
        mean_ch1 - reference_baseline_ch1,
    )

    search_start = max(
        baseline_end,
        int(round(n_sample * PULSE_SEARCH_START_FRACTION)),
    )

    peak_index = search_start + int(
        np.nanargmax(distance_from_baseline[search_start:])
    )

    peak_half_window = max(
        1,
        int(round(PEAK_HALF_WINDOW_NS * 1e-9 * SAMPLE_RATE_HZ)),
    )

    peak_left = max(0, peak_index - peak_half_window)
    peak_right = min(n_sample, peak_index + peak_half_window + 1)

    # 同じ peak 時刻近傍について、イベントごとに中央値を取る
    peak_ch0 = np.nanmedian(ch0[:, peak_left:peak_right], axis=1)
    peak_ch1 = np.nanmedian(ch1[:, peak_left:peak_right], axis=1)

    peak_iq = np.column_stack([peak_ch0, peak_ch1])

    print(f"[load] ch0 key: {ch0_key}")
    print(f"[load] ch1 key: {ch1_key}")
    print(f"[load] events: {n_event}")
    print(f"[load] samples/event: {n_sample}")
    print(f"[peak] index: {peak_index}")
    print(f"[peak] window: {peak_left}:{peak_right}")

    return {
        "baseline_iq": baseline_iq,
        "peak_iq": peak_iq,
        "n_event": n_event,
        "peak_index": peak_index,
    }


def make_temperature_phase(n_event: int) -> tuple[np.ndarray, np.ndarray]:
    """
    event 0 を温度 phase = 0 として、
    50 Hz のレーザーイベント列から 1 Hz 温度 phase を作る。
    """
    event_time_s = np.arange(n_event, dtype=float) / LASER_REPETITION_HZ
    temp_phase = np.mod(event_time_s, TEMP_PERIOD_S) / TEMP_PERIOD_S

    phase_bin = np.floor(temp_phase * N_TEMP_PHASE_BINS).astype(int)
    phase_bin = np.clip(phase_bin, 0, N_TEMP_PHASE_BINS - 1)

    return temp_phase, phase_bin


def calculate_bin_medians(
    baseline_iq: np.ndarray,
    peak_iq: np.ndarray,
    phase_bin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """温度 phase bin ごとの baseline / peak IQ 中央値を作る。"""
    baseline_median = np.full((N_TEMP_PHASE_BINS, 2), np.nan)
    peak_median = np.full((N_TEMP_PHASE_BINS, 2), np.nan)
    count = np.zeros(N_TEMP_PHASE_BINS, dtype=int)

    for ibin in range(N_TEMP_PHASE_BINS):
        mask = phase_bin == ibin
        count[ibin] = int(np.sum(mask))

        if count[ibin] == 0:
            continue

        baseline_median[ibin] = np.nanmedian(baseline_iq[mask], axis=0)
        peak_median[ibin] = np.nanmedian(peak_iq[mask], axis=0)

    return baseline_median, peak_median, count


# =============================================================================
# PLOTTING
# =============================================================================

def draw_full_panel(
    ax: plt.Axes,
    baseline_iq: np.ndarray,
    peak_iq: np.ndarray,
    phase_bin: np.ndarray,
    baseline_median: np.ndarray,
    peak_median: np.ndarray,
    cmap,
    norm,
) -> None:
    """baseline + peak の全体図を描く。"""

    phase_color_value = phase_bin + 0.5

    ax.scatter(
        baseline_iq[:, 0],
        baseline_iq[:, 1],
        c=phase_color_value,
        cmap=cmap,
        norm=norm,
        s=14,
        alpha=0.50,
        marker="o",
        linewidths=0,
        zorder=2,
    )

    ax.scatter(
        peak_iq[:, 0],
        peak_iq[:, 1],
        c=phase_color_value,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.52,
        marker="^",
        linewidths=0,
        zorder=2,
    )

    for ibin in range(N_TEMP_PHASE_BINS):
        jbin = ibin + 1

        if jbin >= N_TEMP_PHASE_BINS:
            if not CLOSE_TEMP_CYCLE:
                continue
            jbin = 0

        b0 = baseline_median[ibin]
        b1 = baseline_median[jbin]
        p0 = peak_median[ibin]
        p1 = peak_median[jbin]

        if not (
            np.all(np.isfinite(b0))
            and np.all(np.isfinite(b1))
            and np.all(np.isfinite(p0))
            and np.all(np.isfinite(p1))
        ):
            continue

        pair_color = cmap(ibin)

        # 温度 phase による baseline の変化ベクトル
        add_arrow(
            ax,
            start=b0,
            vector=b1 - b0,
            color=pair_color,
            width=0.0050,
            alpha=0.96,
            zorder=8,
        )

        # 隣接 2 phase bin の中点における base -> peak ベクトル
        base_mid = 0.5 * (b0 + b1)
        peak_mid = 0.5 * (p0 + p1)

        add_arrow(
            ax,
            start=base_mid,
            vector=peak_mid - base_mid,
            color=pair_color,
            width=0.0035,
            alpha=0.92,
            zorder=7,
        )

        # 各 phase bin の中央値を小さめの黒縁で表示
        ax.scatter(
            b0[0],
            b0[1],
            s=38,
            facecolor=pair_color,
            edgecolor="black",
            linewidth=0.55,
            zorder=10,
        )

    ax.set_title("Full IQ plane: baseline and peak raw data")
    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.grid(alpha=0.24)
    ax.set_aspect("equal", adjustable="datalim")


def draw_zoom_panel(
    ax: plt.Axes,
    baseline_iq: np.ndarray,
    phase_bin: np.ndarray,
    baseline_median: np.ndarray,
    peak_median: np.ndarray,
    cmap,
    norm,
) -> None:
    """
    baseline の温度位相依存だけを見やすくした zoom 図。
    peak 点は描かず、peak 方向は規格化ベクトルだけを描く。
    """
    phase_color_value = phase_bin + 0.5

    ax.scatter(
        baseline_iq[:, 0],
        baseline_iq[:, 1],
        c=phase_color_value,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.62,
        marker="o",
        linewidths=0,
        zorder=2,
    )

    base_span_ch0 = (
        np.nanmax(baseline_iq[:, 0])
        - np.nanmin(baseline_iq[:, 0])
    )

    base_span_ch1 = (
        np.nanmax(baseline_iq[:, 1])
        - np.nanmin(baseline_iq[:, 1])
    )

    base_span = max(base_span_ch0, base_span_ch1)

    if not np.isfinite(base_span) or base_span <= 0:
        base_span = 1.0

    normalized_pulse_arrow_length = (
        ZOOM_NORMALIZED_PULSE_ARROW_FRACTION * base_span
    )

    for ibin in range(N_TEMP_PHASE_BINS):
        jbin = ibin + 1

        if jbin >= N_TEMP_PHASE_BINS:
            if not CLOSE_TEMP_CYCLE:
                continue
            jbin = 0

        b0 = baseline_median[ibin]
        b1 = baseline_median[jbin]
        p0 = peak_median[ibin]
        p1 = peak_median[jbin]

        if not (
            np.all(np.isfinite(b0))
            and np.all(np.isfinite(b1))
            and np.all(np.isfinite(p0))
            and np.all(np.isfinite(p1))
        ):
            continue

        pair_color = cmap(ibin)

        # baseline の実際の移動ベクトル
        add_arrow(
            ax,
            start=b0,
            vector=b1 - b0,
            color=pair_color,
            width=0.0060,
            alpha=0.98,
            zorder=8,
        )

        # 中点での pulse 応答方向
        base_mid = 0.5 * (b0 + b1)
        peak_mid = 0.5 * (p0 + p1)
        pulse_vector = peak_mid - base_mid
        pulse_norm = float(np.linalg.norm(pulse_vector))

        if pulse_norm > 0:
            unit_pulse_vector = (
                pulse_vector / pulse_norm * normalized_pulse_arrow_length
            )

            add_arrow(
                ax,
                start=base_mid,
                vector=unit_pulse_vector,
                color=pair_color,
                width=0.0035,
                alpha=0.92,
                zorder=7,
            )

        ax.scatter(
            b0[0],
            b0[1],
            s=44,
            facecolor=pair_color,
            edgecolor="black",
            linewidth=0.60,
            zorder=10,
        )

    xlim = padded_limits(baseline_iq[:, 0], pad_fraction=0.18)
    ylim = padded_limits(baseline_iq[:, 1], pad_fraction=0.18)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title("Zoom: baseline motion with normalized pulse directions")
    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.grid(alpha=0.24)
    ax.set_aspect("equal", adjustable="box")


def add_legend(ax: plt.Axes, zoom: bool) -> None:
    """図の説明用 legend。"""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=7,
            label="baseline raw data",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=7,
            label="peak raw data",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            label=r"baseline shift: $B_i \rightarrow B_{i+1}$",
        ),
    ]

    if zoom:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.5,
                linestyle="--",
                label="normalized base-to-peak direction",
            )
        )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.5,
                linestyle="--",
                label="base-to-peak vector at phase-pair midpoint",
            )
        )

    if zoom:
        handles = [handles[0], handles[2], handles[3]]

    ax.legend(
        handles=handles,
        loc="best",
        fontsize=8,
        frameon=True,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = load_baseline_and_peak()

    baseline_iq = result["baseline_iq"]
    peak_iq = result["peak_iq"]
    n_event = int(result["n_event"])

    temp_phase, phase_bin = make_temperature_phase(n_event)

    baseline_median, peak_median, count = calculate_bin_medians(
        baseline_iq,
        peak_iq,
        phase_bin,
    )

    print("\n[phase-bin event count]")
    for ibin, n in enumerate(count):
        p0 = ibin / N_TEMP_PHASE_BINS
        p1 = (ibin + 1) / N_TEMP_PHASE_BINS
        print(f"  bin {ibin:02d}: phase {p0:.1f}-{p1:.1f}, N={n}")

    cmap = plt.get_cmap("hsv", N_TEMP_PHASE_BINS)
    boundaries = np.arange(N_TEMP_PHASE_BINS + 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    phase_tick_positions = np.arange(N_TEMP_PHASE_BINS) + 0.5
    phase_tick_labels = [
        f"{i / N_TEMP_PHASE_BINS:.1f}–{(i + 1) / N_TEMP_PHASE_BINS:.1f}"
        for i in range(N_TEMP_PHASE_BINS)
    ]

    # -------------------------------------------------------------------------
    # Combined figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7.2),
        constrained_layout=True,
    )

    draw_full_panel(
        axes[0],
        baseline_iq,
        peak_iq,
        phase_bin,
        baseline_median,
        peak_median,
        cmap,
        norm,
    )
    add_legend(axes[0], zoom=False)

    draw_zoom_panel(
        axes[1],
        baseline_iq,
        phase_bin,
        baseline_median,
        peak_median,
        cmap,
        norm,
    )
    add_legend(axes[1], zoom=True)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    colorbar = fig.colorbar(
        mappable,
        ax=axes,
        ticks=phase_tick_positions,
        shrink=0.86,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(phase_tick_labels)
    colorbar.set_label("temperature phase in 1 Hz cycle")

    fig.suptitle(
        "5.476 GHz, z = 7.5 mm, x = 3.4 mm: "
        "baseline / peak IQ versus temperature phase",
        fontsize=13,
    )

    combined_path = OUTPUT_DIR / "iq_baseline_peak_phase_full_and_zoom.png"
    fig.savefig(combined_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Full only
    # -------------------------------------------------------------------------
    fig_full, ax_full = plt.subplots(figsize=(8.5, 7.5))

    draw_full_panel(
        ax_full,
        baseline_iq,
        peak_iq,
        phase_bin,
        baseline_median,
        peak_median,
        cmap,
        norm,
    )
    add_legend(ax_full, zoom=False)

    colorbar_full = fig_full.colorbar(
        mappable,
        ax=ax_full,
        ticks=phase_tick_positions,
        shrink=0.86,
        pad=0.02,
    )
    colorbar_full.ax.set_yticklabels(phase_tick_labels)
    colorbar_full.set_label("temperature phase in 1 Hz cycle")

    full_path = OUTPUT_DIR / "iq_baseline_peak_phase_full.png"
    fig_full.savefig(full_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig_full)

    # -------------------------------------------------------------------------
    # Zoom only
    # -------------------------------------------------------------------------
    fig_zoom, ax_zoom = plt.subplots(figsize=(8.5, 7.5))

    draw_zoom_panel(
        ax_zoom,
        baseline_iq,
        phase_bin,
        baseline_median,
        peak_median,
        cmap,
        norm,
    )
    add_legend(ax_zoom, zoom=True)

    colorbar_zoom = fig_zoom.colorbar(
        mappable,
        ax=ax_zoom,
        ticks=phase_tick_positions,
        shrink=0.86,
        pad=0.02,
    )
    colorbar_zoom.ax.set_yticklabels(phase_tick_labels)
    colorbar_zoom.set_label("temperature phase in 1 Hz cycle")

    zoom_path = OUTPUT_DIR / "iq_baseline_phase_zoom.png"
    fig_zoom.savefig(zoom_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig_zoom)

    print("\n[saved]")
    print(f"  {combined_path}")
    print(f"  {full_path}")
    print(f"  {zoom_path}")


if __name__ == "__main__":
    main()