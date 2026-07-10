
from pathlib import Path
import csv
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# 設定
# ============================================================
BASE_DIR = Path("./data/20260527")
INPUT_DIR = Path("/Volumes/NO NAME/data/20260527")

TARGET_Z = 7.5
TARGET_X = 3.4
TARGET_FREQS_GHZ = [5.451, 5.461, 5.476, 5.491, 5.501]

N_PRE = 500
SAMPLE_RATE_HZ = 2.5e9

# 20260527: laser = 50 Hz, pedestal/temperature modulation ~= 1 Hz
EVENT_RATE_HZ = 50.0
TEMP_FREQ_HINT_HZ = 1.0
TEMP_SEARCH_HALF_WIDTH_HZ = 0.20
N_TEMP_FREQ_GRID = 801

# 表示するパルスの時間範囲
PLOT_TMIN_NS = 400.0
PLOT_TMAX_NS = 1500.0

# phase bin は「同じ温度サイクル内の位相」で選ぶ。
# 16 bins -> 幅約22.5 deg; 1000 eventsなら1 binあたり概ね50-60 events。
N_PHASE_BINS = 16
PHASE_BIN_WIDTH_FACTOR = 0.90  # 1.0なら隙間なく分割、<1で少し狭くする

# pedestal bin は、pedestal IQ 平面で局所的に近いイベントだけを選ぶ。
N_PEDESTAL_BINS = 12
PEDESTAL_EVENTS_PER_BIN = None  # None: N_event // N_PEDESTAL_BINS を使う

MIN_EVENTS_PER_BIN = 20
FLIP_CH0 = False
SHOW_FIGURES = False

OUT_ROOT = BASE_DIR / "rf_iq_trajectory_binned"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 読み込み
# ============================================================
FOLDER_RE = re.compile(
    r"(?P<freq>\d+(?:\.\d+)?)GHz_z=(?P<z>\d+(?:\.\d+)?)mm_x=(?P<x>\d+(?:\.\d+)?)mm"
)


def find_channel_keys(keys):
    candidates0 = ["ch0", "channel0", "wave0", "data0", "I", "i"]
    candidates1 = ["ch1", "channel1", "wave1", "data1", "Q", "q"]

    key0 = next((k for k in candidates0 if k in keys), None)
    key1 = next((k for k in candidates1 if k in keys), None)
    if key0 is None or key1 is None:
        raise KeyError(f"ch0/ch1 key not found. keys={keys}")
    return key0, key1


def ensure_event_sample_shape(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Unexpected waveform shape: {a.shape}")

    # 通常は (event, sample)。もし (sample, event) なら転置する。
    return a.T if a.shape[0] > a.shape[1] else a


def resolve_folder(input_dir, freq_ghz, z_mm, x_mm, freq_tol_ghz=0.003):
    """完全一致を優先し、5.490/5.491 GHzのような表記差も救済する。"""
    exact = input_dir / f"{freq_ghz:.3f}GHz_z={z_mm:.1f}mm_x={x_mm:.1f}mm"
    if exact.is_dir():
        return exact

    candidates = []
    for p in input_dir.iterdir():
        if not p.is_dir():
            continue
        m = FOLDER_RE.match(p.name)
        if m is None:
            continue

        f = float(m.group("freq"))
        z = float(m.group("z"))
        x = float(m.group("x"))

        if abs(f - freq_ghz) <= freq_tol_ghz and np.isclose(z, z_mm) and np.isclose(x, x_mm):
            # 「second」は無印のデータが存在すれば後回しにする。
            penalty_second = 1 if "second" in p.name.lower() else 0
            candidates.append((penalty_second, abs(f - freq_ghz), p))

    if not candidates:
        raise FileNotFoundError(
            f"Folder not found for {freq_ghz:.3f} GHz, z={z_mm:.1f}, x={x_mm:.1f}"
        )

    candidates.sort(key=lambda t: (t[0], t[1], t[2].name))
    return candidates[0][2]


def load_all_waveforms(folder):
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files: {folder}")

    ch0_list, ch1_list = [], []
    for fp in files:
        with np.load(fp, allow_pickle=True) as d:
            key0, key1 = find_channel_keys(list(d.keys()))
            ch0 = ensure_event_sample_shape(d[key0])
            ch1 = ensure_event_sample_shape(d[key1])

        if ch0.shape != ch1.shape:
            raise ValueError(f"Shape mismatch in {fp.name}: {ch0.shape}, {ch1.shape}")

        ch0_list.append(ch0)
        ch1_list.append(ch1)

    return np.concatenate(ch0_list, axis=0), np.concatenate(ch1_list, axis=0)


# ============================================================
# pedestal から温度位相を推定
# ============================================================
def estimate_temperature_phase(ped0, ped1):
    """
    pedestal IQ の主成分を 1 Hz 近傍の sinusoid で fit し、
    「pedestal主成分が最大の瞬間」を phase=0 にした位相を返す。

    注意:
    これは温度計で校正された絶対温度位相ではなく、
    pedestal の1 Hz周期から得る相対位相。
    """
    ped = np.column_stack([ped0, ped1])

    center = np.mean(ped, axis=0)
    scale = np.std(ped, axis=0, ddof=1)
    scale[scale < np.finfo(float).eps] = 1.0
    ped_z = (ped - center) / scale

    # pedestal変動の主軸（PCA第1主成分）
    _, _, vh = np.linalg.svd(ped_z, full_matrices=False)
    pc = ped_z @ vh.T
    trace = pc[:, 0] - np.mean(pc[:, 0])

    t_s = np.arange(len(trace)) / EVENT_RATE_HZ

    # 0.8--1.2 Hz で周期を微調整
    f_grid = np.linspace(
        TEMP_FREQ_HINT_HZ - TEMP_SEARCH_HALF_WIDTH_HZ,
        TEMP_FREQ_HINT_HZ + TEMP_SEARCH_HALF_WIDTH_HZ,
        N_TEMP_FREQ_GRID,
    )
    score = np.empty_like(f_grid)
    for i, f in enumerate(f_grid):
        score[i] = np.abs(np.sum(trace * np.exp(-2j * np.pi * f * t_s)))

    f_est = f_grid[np.argmax(score)]
    omega = 2.0 * np.pi * f_est

    # trace = a cos(wt) + b sin(wt) + c
    design = np.column_stack([
        np.cos(omega * t_s),
        np.sin(omega * t_s),
        np.ones_like(t_s),
    ])
    a, b, c = np.linalg.lstsq(design, trace, rcond=None)[0]
    fit_trace = design @ np.array([a, b, c])

    ss_res = np.sum((trace - fit_trace) ** 2)
    ss_tot = np.sum((trace - np.mean(trace)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # a cos(wt)+b sin(wt)=A cos(wt-delta)
    # wt=delta で主成分が最大 -> phase=0
    delta = np.arctan2(b, a)
    phase_rad = np.mod(omega * t_s - delta, 2.0 * np.pi)

    return {
        "phase_rad": phase_rad,
        "f_est_hz": f_est,
        "fit_r2": r2,
        "ped_z": ped_z,
        "pc1": pc[:, 0],
        "pc1_fit": fit_trace,
        "time_event_s": t_s,
    }


def circular_distance(a, b):
    return np.angle(np.exp(1j * (a - b)))


def make_phase_bins(phase_rad, n_bins):
    centers = np.arange(n_bins) * 2.0 * np.pi / n_bins
    half_width = np.pi / n_bins * PHASE_BIN_WIDTH_FACTOR

    masks, labels = [], []
    for i, center in enumerate(centers):
        mask = np.abs(circular_distance(phase_rad, center)) <= half_width
        masks.append(mask)
        labels.append(
            f"phase {np.degrees(center):.1f}° "
            f"(±{np.degrees(half_width):.1f}°)"
        )
    return masks, labels, centers


def make_local_pedestal_bins(ped_z, pc1, n_bins, n_keep=None):
    """
    IQ pedestal 平面上で局所的に近いイベントを選ぶ。
    pc1 の等分位点を「代表位置」にして、そこから2次元距離が近いイベントのみを使う。
    """
    n_event = len(pc1)
    if n_keep is None:
        n_keep = max(MIN_EVENTS_PER_BIN, n_event // n_bins)
    n_keep = min(n_keep, n_event)

    quantiles = (np.arange(n_bins) + 0.5) / n_bins
    targets = np.quantile(pc1, quantiles)

    masks, labels, centers = [], [], []
    for i, target in enumerate(targets):
        i_center = np.argmin(np.abs(pc1 - target))
        center = ped_z[i_center]
        d2 = np.sum((ped_z - center) ** 2, axis=1)

        keep_idx = np.argpartition(d2, n_keep - 1)[:n_keep]
        mask = np.zeros(n_event, dtype=bool)
        mask[keep_idx] = True

        masks.append(mask)
        labels.append(f"local pedestal {i + 1}/{n_bins}")
        centers.append(center)

    return masks, labels, np.asarray(centers)


# ============================================================
# IQ 軌跡の作成
# ============================================================
def add_colored_trajectory(ax, x, y, t_ns, norm, cmap="viridis", lw=2.0):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(t_ns[:-1])
    ax.add_collection(lc)
    return lc


def make_bin_result(ch0, ch1, ped0_each, ped1_each, event_mask, time_ns):
    n_selected = int(np.sum(event_mask))
    if n_selected < MIN_EVENTS_PER_BIN:
        return None

    # 重要: pedestal は各イベントごとに引く。
    # これにより、bin内で残った局所的なパルス形状だけを平均する。
    mean_dev0 = np.mean(ch0[event_mask] - ped0_each[event_mask, None], axis=0)
    mean_dev1 = np.mean(ch1[event_mask] - ped1_each[event_mask, None], axis=0)

    plot_mask = (time_ns >= PLOT_TMIN_NS) & (time_ns <= PLOT_TMAX_NS)
    peak_search_mask = (time_ns >= N_PRE / SAMPLE_RATE_HZ * 1e9) & (time_ns <= PLOT_TMAX_NS)

    dist = np.hypot(mean_dev0, mean_dev1)
    candidate = np.where(peak_search_mask)[0]
    peak_idx = candidate[np.argmax(dist[candidate])]

    # 直線からの外れの簡単な指標:
    # peak方向に垂直な成分の最大値 / 最大IQ変位
    peak_vec = np.array([mean_dev0[peak_idx], mean_dev1[peak_idx]])
    peak_amp = np.linalg.norm(peak_vec)
    if peak_amp > 0:
        u = peak_vec / peak_amp
        perp = -u[1] * mean_dev0[plot_mask] + u[0] * mean_dev1[plot_mask]
        curvature_ratio = np.max(np.abs(perp)) / peak_amp
    else:
        curvature_ratio = np.nan

    return {
        "n_event": n_selected,
        "mean_dev0": mean_dev0,
        "mean_dev1": mean_dev1,
        "plot_mask": plot_mask,
        "peak_idx": peak_idx,
        "peak_time_ns": time_ns[peak_idx],
        "peak_dev0": mean_dev0[peak_idx],
        "peak_dev1": mean_dev1[peak_idx],
        "peak_amp": peak_amp,
        "curvature_ratio": curvature_ratio,
    }


def make_condition(freq_ghz):
    folder = resolve_folder(INPUT_DIR, freq_ghz, TARGET_Z, TARGET_X)
    print(f"[load] {folder.name}")

    ch0, ch1 = load_all_waveforms(folder)
    if FLIP_CH0:
        ch0 = -ch0

    n_event, n_sample = ch0.shape
    time_ns = np.arange(n_sample) / SAMPLE_RATE_HZ * 1e9

    ped0_each = np.mean(ch0[:, :N_PRE], axis=1)
    ped1_each = np.mean(ch1[:, :N_PRE], axis=1)

    phase_info = estimate_temperature_phase(ped0_each, ped1_each)

    return {
        "freq": freq_ghz,
        "folder": folder,
        "n_event": n_event,
        "time_ns": time_ns,
        "ped0_each": ped0_each,
        "ped1_each": ped1_each,
        "phase_info": phase_info,
        "waveforms": (ch0, ch1),
    }


def create_binned_results(cond, mode):
    ch0, ch1 = cond["waveforms"]
    phase_info = cond["phase_info"]

    if mode == "phase":
        masks, labels, centers = make_phase_bins(
            phase_info["phase_rad"], N_PHASE_BINS
        )
    elif mode == "pedestal":
        masks, labels, centers = make_local_pedestal_bins(
            phase_info["ped_z"],
            phase_info["pc1"],
            N_PEDESTAL_BINS,
            PEDESTAL_EVENTS_PER_BIN,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = [
        make_bin_result(
            ch0,
            ch1,
            cond["ped0_each"],
            cond["ped1_each"],
            event_mask,
            cond["time_ns"],
        )
        for event_mask in masks
    ]

    return {
        "mode": mode,
        "masks": masks,
        "labels": labels,
        "centers": centers,
        "results": results,
    }


# ============================================================
# 描画
# ============================================================
def symmetric_limits(bin_results):
    values = []
    for r in bin_results:
        if r is None:
            continue
        m = r["plot_mask"]
        values.append(np.abs(r["mean_dev0"][m]))
        values.append(np.abs(r["mean_dev1"][m]))

    if not values:
        return (-1.0, 1.0)

    extent = max(np.max(v) for v in values)
    extent = max(extent, np.finfo(float).eps)
    return (-1.12 * extent, 1.12 * extent)


def draw_one_trajectory(ax, result, time_ns, norm, xylim, title):
    if result is None:
        ax.text(0.5, 0.5, "too few events", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    m = result["plot_mask"]
    add_colored_trajectory(
        ax,
        result["mean_dev0"][m],
        result["mean_dev1"][m],
        time_ns[m],
        norm=norm,
    )

    ax.scatter(
        0.0,
        0.0,
        s=55,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        marker="o",
        zorder=5,
    )
    ax.scatter(
        result["peak_dev0"],
        result["peak_dev1"],
        s=85,
        color="black",
        marker="*",
        zorder=6,
    )

    ax.axhline(0.0, color="black", lw=0.6, alpha=0.35)
    ax.axvline(0.0, color="black", lw=0.6, alpha=0.35)
    ax.set_xlim(*xylim)
    ax.set_ylim(*xylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=9)


def save_pedestal_diagnostic(cond):
    out_dir = OUT_ROOT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = cond["phase_info"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sc = axes[0].scatter(
        cond["ped0_each"],
        cond["ped1_each"],
        c=np.degrees(info["phase_rad"]),
        cmap="twilight",
        s=10,
        alpha=0.8,
    )
    axes[0].set_xlabel("pedestal ch0")
    axes[0].set_ylabel("pedestal ch1")
    axes[0].set_title("Pedestal IQ colored by estimated phase")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[0], label="relative pedestal phase [deg]")

    axes[1].plot(info["time_event_s"], info["pc1"], ".", ms=2.5, alpha=0.65, label="pedestal PC1")
    axes[1].plot(info["time_event_s"], info["pc1_fit"], lw=1.8, label="1 Hz fit")
    axes[1].set_xlabel("event time inferred from 50 Hz laser [s]")
    axes[1].set_ylabel("standardized pedestal PC1")
    axes[1].set_title(
        f"f_est = {info['f_est_hz']:.4f} Hz, fit $R^2$ = {info['fit_r2']:.3f}"
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"{cond['freq']:.3f} GHz: pedestal phase diagnostic", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"pedestal_phase_{cond['freq']:.3f}GHz.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(cond, binned, norm):
    mode = binned["mode"]
    out_dir = OUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(binned["results"])
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.0 * nrow), squeeze=False)
    axes_flat = axes.ravel()

    xylim = symmetric_limits(binned["results"])
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue

        r = binned["results"][i]
        if r is None:
            subtitle = f"{binned['labels'][i]}\nN < {MIN_EVENTS_PER_BIN}"
        else:
            subtitle = (
                f"{binned['labels'][i]}\n"
                f"N={r['n_event']}, C={r['curvature_ratio']:.3f}"
            )

        draw_one_trajectory(ax, r, cond["time_ns"], norm, xylim, subtitle)
        ax.set_xlabel(r"$\Delta$ ch0")
        ax.set_ylabel(r"$\Delta$ ch1")

    fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes_flat[:n],
        label="time [ns]",
        shrink=0.84,
    )
    fig.suptitle(
        f"{cond['freq']:.3f} GHz — pedestal-subtracted IQ trajectories\n"
        f"{mode} binning; z={TARGET_Z:.1f} mm, x={TARGET_X:.1f} mm",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(out_dir / f"{cond['freq']:.3f}GHz_contact_sheet.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / f"{cond['freq']:.3f}GHz_contact_sheet.pdf", bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def save_cross_frequency_pages(conditions, all_binned, mode, norm):
    out_dir = OUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    n_bin = len(all_binned[0]["results"])
    pdf_path = out_dir / f"iq_trajectory_{mode}_all_frequencies.pdf"

    with PdfPages(pdf_path) as pdf:
        for i_bin in range(n_bin):
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            axes_flat = axes.ravel()

            for ax, cond, binned in zip(axes_flat, conditions, all_binned):
                r = binned["results"][i_bin]
                xylim = symmetric_limits(binned["results"])

                if r is None:
                    subtitle = f"{cond['freq']:.3f} GHz\nN < {MIN_EVENTS_PER_BIN}"
                else:
                    subtitle = (
                        f"{cond['freq']:.3f} GHz\n"
                        f"N={r['n_event']}, C={r['curvature_ratio']:.3f}"
                    )

                draw_one_trajectory(ax, r, cond["time_ns"], norm, xylim, subtitle)
                ax.set_xlabel(r"$\Delta$ ch0")
                ax.set_ylabel(r"$\Delta$ ch1")

            for ax in axes_flat[len(conditions):]:
                ax.axis("off")

            label = all_binned[0]["labels"][i_bin]
            fig.colorbar(
                ScalarMappable(norm=norm, cmap="viridis"),
                ax=axes_flat[:len(conditions)],
                label="time [ns]",
                shrink=0.82,
            )
            fig.suptitle(
                f"{mode} bin: {label}\n"
                f"Pedestal-subtracted IQ trajectories, z={TARGET_Z:.1f} mm, x={TARGET_X:.1f} mm",
                fontsize=15,
            )
            fig.tight_layout(rect=[0, 0, 0.92, 0.94])

            png_path = out_dir / f"{mode}_bin_{i_bin:02d}_all_frequencies.png"
            fig.savefig(png_path, dpi=250, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")

            if SHOW_FIGURES:
                plt.show()
            plt.close(fig)


# ============================================================
# 実行
# ============================================================
MODES = ("phase", "pedestal")

# すべての周波数で同じ時間色スケール
norm = Normalize(vmin=PLOT_TMIN_NS, vmax=PLOT_TMAX_NS)

conditions = []
summary_rows = []

for freq in TARGET_FREQS_GHZ:
    try:
        cond = make_condition(freq)
    except FileNotFoundError as exc:
        print(f"[skip] {exc}")
        continue

    save_pedestal_diagnostic(cond)
    cond["binned"] = {}

    # 大きな波形配列を保持したまま次の周波数を読まないよう、
    # この周波数について全bin平均をここで作り切る。
    for mode in MODES:
        binned = create_binned_results(cond, mode)
        cond["binned"][mode] = binned
        save_contact_sheet(cond, binned, norm)

        for i, r in enumerate(binned["results"]):
            if r is None:
                continue
            summary_rows.append({
                "mode": mode,
                "freq_GHz": cond["freq"],
                "bin_index": i,
                "bin_label": binned["labels"][i],
                "N_event": r["n_event"],
                "f_temp_est_Hz": cond["phase_info"]["f_est_hz"],
                "phase_fit_R2": cond["phase_info"]["fit_r2"],
                "peak_time_ns": r["peak_time_ns"],
                "peak_dch0": r["peak_dev0"],
                "peak_dch1": r["peak_dev1"],
                "peak_amp": r["peak_amp"],
                "curvature_ratio": r["curvature_ratio"],
            })

    # 4000 events x 5000 samples の配列を直ちに解放する。
    del cond["waveforms"]
    conditions.append(cond)

if not conditions:
    raise RuntimeError("対象フォルダを一つも読み込めませんでした。")

for mode in MODES:
    all_binned = [cond["binned"][mode] for cond in conditions]
    save_cross_frequency_pages(conditions, all_binned, mode, norm)

summary_path = OUT_ROOT / "iq_trajectory_binned_summary.csv"
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nSaved:")
print(f"  {OUT_ROOT}")
print(f"  {summary_path}")
