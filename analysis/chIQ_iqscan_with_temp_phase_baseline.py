from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

TARGET_FREQUENCY_HZ = 5.501e9
# =============================================================================
# SETTINGS
# =============================================================================
IQSCAN_DIR = Path("/Volumes/NO NAME/data/20260709")
# WAVEFORM_FILE = Path(
#     "/Volumes/NO NAME/data/20260527/5.476GHz_z=7.5mm_x=3.4mm/"
#     "wf_260527_142822_49.73Hz.npz"
# )

# OUTPUT_DIR = Path(
#     "/Users/kubokosei/software/kidanalysis/analysis/data/20260527/"
#     "iqscan0703_temp_phase_overlay"
# )
WAVEFORM_FILE = Path(
    "/Volumes/NO NAME/data/20260709/5.501GHz_z=8.0mm_x=4.4mm_first/"
    "wf_260709_175104_49.78Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "iqscan0709_temp_phase_overlay"
)
# iqscan0703 内のファイル名が iq_5.80K.npz, iq_6.30K.npz であることを想定。
# SCAN_TEMPERATURES_K = (3.62, 3.60)
SCAN_TEMPERATURES_K = (3.62, )

# 波形ファイル名に合わせて 49.73 Hz を初期値にする。
EVENT_RATE_HZ = 49.78
TEMP_MODULATION_HZ = 1.0
N_PHASE_BINS = 50

# phase=0 の定義をずらしたいときに使う（単位：温度周期）。
PHASE_OFFSET_CYCLES = 0.0

# 各イベントの baseline を求める pre-trigger 区間。
BASELINE_SLICE = slice(0, 1000)


# =============================================================================
# LOADERS
# =============================================================================
def find_iqscan_file(temperature_k: float) -> Path:
    """指定温度の iq scan npz を探す。"""
    exact = IQSCAN_DIR / f"iq_{temperature_k:.2f}K.npz"
    if exact.exists():
        return exact

    candidates = sorted(IQSCAN_DIR.glob(f"*{temperature_k:.2f}K*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"{IQSCAN_DIR} に {temperature_k:.2f} K の iq scan が見つかりません。"
        )
    raise RuntimeError(
        f"{temperature_k:.2f} K に一致する iq scan が複数見つかりました:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_iqscan(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dd[:, 0]=frequency, dd[:, 1]=ch0, dd[:, 2]=ch1 を読む。"""
    with np.load(path, allow_pickle=False) as npz:
        if "dd" not in npz:
            raise KeyError(f"{path} に 'dd' キーがありません。keys={list(npz.keys())}")
        dd = np.asarray(npz["dd"], dtype=float)

    if dd.ndim != 2 or dd.shape[1] < 3:
        raise ValueError(f"{path}: dd の shape が想定外です: {dd.shape}")

    return dd[:, 0], dd[:, 1], dd[:, 2]


def waveform_array_events_by_samples(x: np.ndarray, name: str) -> np.ndarray:
    """波形を (n_events, n_samples) にそろえる。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} は2次元配列である必要があります。shape={x.shape}")

    # 通常は (events, samples) = (1000 or 4000, 5000)。
    # 逆なら自動で転置する。
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x


def load_waveform_baselines(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """ch0/ch1 の各イベント baseline 中央値を返す。"""
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        if "ch0" not in npz or "ch1" not in npz:
            raise KeyError(
                f"{path} に ch0/ch1 がありません。keys={keys}\n"
                "波形ファイルのキー名が異なる場合は load_waveform_baselines を修正してください。"
            )
        ch0 = waveform_array_events_by_samples(npz["ch0"], "ch0")
        ch1 = waveform_array_events_by_samples(npz["ch1"], "ch1")

    if ch0.shape != ch1.shape:
        raise ValueError(f"ch0/ch1 の shape が一致しません: {ch0.shape}, {ch1.shape}")
    if BASELINE_SLICE.stop is not None and BASELINE_SLICE.stop > ch0.shape[1]:
        raise ValueError(
            f"BASELINE_SLICE={BASELINE_SLICE} が波形長 {ch0.shape[1]} を超えています。"
        )

    baseline_ch0 = np.median(ch0[:, BASELINE_SLICE], axis=1)
    baseline_ch1 = np.median(ch1[:, BASELINE_SLICE], axis=1)
    return baseline_ch0, baseline_ch1


# =============================================================================
# TEMPERATURE-PHASE BINNING
# =============================================================================
def temperature_phase_bin_medians(
    baseline_ch0: np.ndarray,
    baseline_ch1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """温度 1 Hz 周期を 50 分割し、各 bin の IQ baseline 中央値を求める。"""
    n_events = len(baseline_ch0)
    if n_events != len(baseline_ch1):
        raise ValueError("baseline_ch0 と baseline_ch1 のイベント数が違います。")

    event_index = np.arange(n_events)

    # 各イベントの温度周期内位相 [0, 1)
    phase_cycles = np.mod(
        event_index * TEMP_MODULATION_HZ / EVENT_RATE_HZ + PHASE_OFFSET_CYCLES,
        1.0,
    )
    phase_bin = np.floor(phase_cycles * N_PHASE_BINS).astype(int)
    phase_bin = np.clip(phase_bin, 0, N_PHASE_BINS - 1)

    centers_cycles = (np.arange(N_PHASE_BINS) + 0.5) / N_PHASE_BINS
    median_ch0 = np.full(N_PHASE_BINS, np.nan)
    median_ch1 = np.full(N_PHASE_BINS, np.nan)
    counts = np.zeros(N_PHASE_BINS, dtype=int)

    for ibin in range(N_PHASE_BINS):
        mask = phase_bin == ibin
        counts[ibin] = np.count_nonzero(mask)
        if counts[ibin] > 0:
            median_ch0[ibin] = np.median(baseline_ch0[mask])
            median_ch1[ibin] = np.median(baseline_ch1[mask])

    return centers_cycles, median_ch0, median_ch1, counts


# =============================================================================
# PLOT
# =============================================================================
def plot_overlay(
    baseline_ch0: np.ndarray,
    baseline_ch1: np.ndarray,
    phase_cycles: np.ndarray,
    med_ch0: np.ndarray,
    med_ch1: np.ndarray,
    *,
    swap_iqscan_axes: bool,
    output_png: Path,
) -> None:
    """
    swap_iqscan_axes=False:
        iq scan      -> (x, y) = (ch0, ch1)
        waveform med -> (x, y) = (ch0, ch1)

    swap_iqscan_axes=True:
        iq scan      -> (x, y) = (ch1, ch0)   # ここだけ入れ替え
        waveform med -> (x, y) = (ch0, ch1)   # これはそのまま
    """
    fig, ax = plt.subplots(figsize=(8.8, 8.0), constrained_layout=True)

    scan_styles = {
        5.80: dict(marker="o", ms=3.2, lw=1.0),
        6.30: dict(marker="s", ms=3.0, lw=1.0),
    }


    for temperature_k in SCAN_TEMPERATURES_K:
        scan_path = find_iqscan_file(temperature_k)
        freq_hz, ch0, ch1 = load_iqscan(scan_path)
        style = scan_styles.get(temperature_k, dict(marker="o", ms=3.0, lw=1.0))

        if swap_iqscan_axes:
            x_scan = ch1
            y_scan = ch0
            scan_label = f"iq scan {temperature_k:.2f} K (swapped)"
        else:
            x_scan = ch0
            y_scan = ch1
            scan_label = f"iq scan {temperature_k:.2f} K"

        # iq scan 全体
        ax.plot(
            x_scan,
            y_scan,
            "-",
            lw=style["lw"],
            alpha=0.75,
            label=scan_label,
        )
        ax.plot(
            x_scan,
            y_scan,
            linestyle="None",
            marker=style["marker"],
            ms=style["ms"],
            alpha=0.85,
        )

        # -------------------------------------------------------------
        # scan 開始点と 5.476 GHz 最近傍点
        # -------------------------------------------------------------
        start_idx = 0
        near_idx = int(np.argmin(np.abs(freq_hz - TARGET_FREQUENCY_HZ)))

        start_freq_ghz = freq_hz[start_idx] / 1e9
        fin_freq_ghz = freq_hz[-1] / 1e9
        near_freq_ghz = freq_hz[near_idx] / 1e9
        delta_freq_khz = (freq_hz[near_idx] - TARGET_FREQUENCY_HZ) / 1e3

        # scan 開始周波数の点
        ax.scatter(
            x_scan[start_idx],
            y_scan[start_idx],
            marker=">",
            s=95,
            facecolor="limegreen",
            edgecolor="k",
            linewidth=0.8,
            zorder=7,
            label=(
                f"{temperature_k:.2f} K start: "
                f"{start_freq_ghz:.6f} GHz"
            ),
        )
        ax.scatter(
            x_scan[-1],
            y_scan[-1],
            marker="<",
            s=95,
            facecolor="crimson",
            edgecolor="k",
            linewidth=0.8,
            zorder=7,
            label=(
                f"{temperature_k:.2f} K end: "
                f"{fin_freq_ghz:.6f} GHz"
            ),
        )

        # 5.476 GHz に最も近い scan 点
        ax.scatter(
            x_scan[near_idx],
            y_scan[near_idx],
            marker="*",
            s=150,
            facecolor="gold",
            edgecolor="k",
            linewidth=0.8,
            zorder=8,
            label=(
                f"{temperature_k:.2f} K nearest 5.501 GHz: "
                f"{near_freq_ghz:.6f} GHz "
                f"({delta_freq_khz:+.1f} kHz)"
            ),
        )

        print(
            f"[iq scan] {scan_path.name}: N={len(freq_hz)}, "
            f"start={start_freq_ghz:.6f} GHz, "
            f"end={fin_freq_ghz:.6f} GHz, "
            f"nearest 5.501 GHz={near_freq_ghz:.6f} GHz "
            f"({delta_freq_khz:+.1f} kHz), "
            f"range={freq_hz.min()/1e9:.6f}--{freq_hz.max()/1e9:.6f} GHz, "
            f"swap_iqscan_axes={swap_iqscan_axes}"
        )

    # 5.476 GHz waveform baseline median は反転しない
    valid = np.isfinite(med_ch0) & np.isfinite(med_ch1)

    loop_ch0 = np.r_[med_ch0[valid], med_ch0[valid][0]]
    loop_ch1 = np.r_[med_ch1[valid], med_ch1[valid][0]]

    ax.plot(
        loop_ch0,
        loop_ch1,
        "-",
        color="k",
        lw=1.2,
        alpha=0.50,
        zorder=4,
        label="5.476 GHz waveform baseline median (50 temp-phase bins)",
    )

    norm = Normalize(vmin=0.0, vmax=1.0)
    points = ax.scatter(
        med_ch0[valid],
        med_ch1[valid],
        c=phase_cycles[valid],
        cmap="viridis",
        norm=norm,
        s=34,
        edgecolors="k",
        linewidths=0.35,
        zorder=5,
    )

    # phase ~ 0 の点
    phase0_idx = int(np.nanargmin(np.abs(phase_cycles - 0.0)))
    ax.scatter(
        med_ch0[phase0_idx],
        med_ch1[phase0_idx],
        marker="*",
        s=150,
        facecolor="none",
        edgecolor="crimson",
        linewidth=1.2,
        zorder=6,
        label="temp phase 0",
    )

    cbar = fig.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label("temperature phase within 1 Hz cycle [cycle]")
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

    if swap_iqscan_axes:
        ax.set_xlabel("x-axis: iq scan ch1 / waveform ch0 [raw]")
        ax.set_ylabel("y-axis: iq scan ch0 / waveform ch1 [raw]")
        ax.set_title(
            "IQ scans with swapped axes (x=ch1, y=ch0)\n"
            "overlaid with waveform baseline medians at 5.476 GHz (not swapped)"
        )
    else:
        ax.set_xlabel("ch0 [raw]")
        ax.set_ylabel("ch1 [raw]")
        ax.set_title(
            "IQ scans (5.80 K, 6.30 K) and waveform baseline medians by temperature phase"
        )

    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)

    fig.savefig(output_png, dpi=240)
    plt.close(fig)
    print(f"[saved] {output_png}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 波形データの phase-bin baseline median
    baseline_ch0, baseline_ch1 = load_waveform_baselines(WAVEFORM_FILE)
    phase_cycles, med_ch0, med_ch1, counts = temperature_phase_bin_medians(
        baseline_ch0, baseline_ch1
    )

    # --- 元の図 ---
    output_png_normal = OUTPUT_DIR / "iqscan_5p80K_6p30K_with_50_temp_phase_baselines.png"
    plot_overlay(
        baseline_ch0,
        baseline_ch1,
        phase_cycles,
        med_ch0,
        med_ch1,
        swap_iqscan_axes=False,
        output_png=output_png_normal,
    )

    # --- 追加図: iq scan だけ ch0/ch1 を入れ替え ---
    output_png_swapped = OUTPUT_DIR / "iqscan_5p80K_6p30K_swapped_iqscan_only_with_50_temp_phase_baselines.png"
    plot_overlay(
        baseline_ch0,
        baseline_ch1,
        phase_cycles,
        med_ch0,
        med_ch1,
        swap_iqscan_axes=True,
        output_png=output_png_swapped,
    )

    # CSV は共通なので 1 回だけ保存
    output_csv = OUTPUT_DIR / "waveform_baseline_median_by_50_temp_phase_bins.csv"
    header = "phase_center_cycle,median_ch0,median_ch1,n_events"
    np.savetxt(
        output_csv,
        np.column_stack([phase_cycles, med_ch0, med_ch1, counts]),
        delimiter=",",
        header=header,
        comments="",
    )

    print(f"[waveform] {WAVEFORM_FILE.name}: N events={len(baseline_ch0)}")
    print(f"[baseline] samples used per event: {BASELINE_SLICE.start}:{BASELINE_SLICE.stop}")
    print(f"[phase] event rate={EVENT_RATE_HZ:.4f} Hz, bins={N_PHASE_BINS}")
    print(f"[saved] {output_csv}")


if __name__ == "__main__":
    main()