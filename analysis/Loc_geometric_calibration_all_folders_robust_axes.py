from __future__ import annotations

"""
20260709 の 5.501GHz 系 waveform フォルダすべてに、
iq_3.62K.npz から求めた geometric calibration

    z_final = 1 + (((z_raw * exp(i 2pi tau f)) * exp(-i alpha)) / a - 1) * exp(-i phi)

を適用し、各フォルダごとの最終補正後 IQ track をまとめてプロットする。

出力
----
- corrected_iq_tracks_all_folders.pdf   : 複数ページ PDF（9 枚/ページ）
- corrected_iq_tracks_page_01.png など : 各ページ PNG
- calibration_summary.json              : 補正パラメータ
"""

from dataclasses import dataclass
import json
import math
import re
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

DATA_ROOT = Path("/Volumes/NO NAME/data/20260709")
IQ_SCAN = DATA_ROOT / "iq_3.62K.npz"

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "all_folders_corrected_iqtracks"
)

READOUT_FREQUENCY_HZ = 5.501e9
Q_SIGN = +1

# tau fit 用
N_EDGE_POINTS = 5

# 円 fit 用
CIRCLE_HALF_WIDTH_POINTS = 15

# 描画設定
MAX_EVENTS_TO_PLOT = 120
PLOT_SAMPLE_STRIDE = 5
PEDESTAL_FRACTION = 0.10

# 1ページあたり 3x3 = 9 枚。18フォルダ前後なら約2ページ。
NROWS = 3
NCOLS = 3

# 軸範囲の決め方
# "per_folder": 各フォルダを個別に robust zoom（形を見比べやすい）
# "per_page"  : 同じページ内では共通軸
# "global"    : 全ページ・全フォルダで共通軸（絶対スケール比較向け）
AXIS_MODE = "per_folder"

# 灰色イベント群のうち、この百分位範囲を軸決定に使用する。
# median track と pedestal は百分位に関係なく必ず全体を含める。
AXIS_PERCENTILES = (0.5, 99.5)
AXIS_PADDING_FRACTION = 0.08

# IQ 平面の x, y の縮尺を同じにするため、表示領域も正方形にする。
FORCE_SQUARE_LIMITS = True

# 極端に小さいトラックでも軸幅が潰れないための最小幅
MIN_AXIS_SPAN = 0.02

# 対象フォルダ
TARGET_DIR_PATTERN = "5.501GHz_*"

# フォルダ内で使う waveform npz の候補
WAVEFORM_GLOB = "wf_*.npz"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class TauFitResult:
    tau_s: float
    intercept_b_rad: float
    slope_rad_per_hz: float
    f_ref_hz: float
    rmse_edge_rad: float
    r_squared_edge: float
    high_edge_branch_shift: int


@dataclass(frozen=True)
class CircleFitResult:
    center: complex
    radius: float
    radial_rms: float
    resonance_index: int
    resonance_frequency_hz: float
    resonance_point: complex


@dataclass(frozen=True)
class GeometricCalibration:
    alpha_rad: float
    amplitude_a: float
    phi_rad: float
    point_p: complex
    circle_center_after_tau: complex
    circle_radius_after_tau: float


@dataclass(frozen=True)
class FolderResult:
    folder_name: str
    waveform_path: Path
    z_final: np.ndarray
    pedestal: complex
    median_track: np.ndarray
    title: str


# =============================================================================
# BASIC LOADERS
# =============================================================================

def convert_frequency_to_hz(frequency: np.ndarray) -> np.ndarray:
    f = np.asarray(frequency, dtype=float)
    typical = float(np.nanmedian(np.abs(f)))
    if typical < 100.0:
        print("[frequency] IQ scan frequency interpreted as GHz")
        return f * 1e9
    if typical < 1e7:
        print("[frequency] IQ scan frequency interpreted as MHz")
        return f * 1e6
    print("[frequency] IQ scan frequency interpreted as Hz")
    return f


def load_iq_scan(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"IQ scan file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        if "dd" not in npz:
            raise KeyError(f"'dd' not found in {path.name}. keys={list(npz.keys())}")
        dd = np.asarray(npz["dd"], dtype=float)

    if dd.ndim != 2 or dd.shape[1] < 3:
        raise ValueError(f"dd shape must be (N, >=3), got {dd.shape}")

    frequency_hz = convert_frequency_to_hz(dd[:, 0])
    ch0 = dd[:, 1]
    ch1 = dd[:, 2]

    order = np.argsort(frequency_hz)
    return frequency_hz[order], ch0[order], ch1[order]


def find_key_case_insensitive(keys: list[str], candidates: tuple[str, ...]) -> str:
    lower_to_original = {key.lower(): key for key in keys}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    raise KeyError(f"Could not find any of {candidates}. keys={keys}")


def load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Waveform file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        ch0_key = find_key_case_insensitive(keys, ("ch0", "channel0", "channel_0", "i"))
        ch1_key = find_key_case_insensitive(keys, ("ch1", "channel1", "channel_1", "q"))

        ch0 = np.asarray(npz[ch0_key], dtype=float)
        ch1 = np.asarray(npz[ch1_key], dtype=float)

    if ch0.shape != ch1.shape:
        raise ValueError(f"ch0 shape {ch0.shape} and ch1 shape {ch1.shape} differ")

    if ch0.ndim == 1:
        ch0 = ch0[np.newaxis, :]
        ch1 = ch1[np.newaxis, :]

    if ch0.ndim != 2:
        raise ValueError(f"waveform must be 2D, got {ch0.shape}")

    # 必要なら (events, samples) に直す
    if ch0.shape[0] > ch0.shape[1] and ch0.shape[1] <= 2000:
        print(f"[waveform] transposing {path.name} to (events, samples)")
        ch0 = ch0.T
        ch1 = ch1.T

    return ch0, ch1


# =============================================================================
# CALIBRATION FITS
# =============================================================================

def make_edge_mask(n_points: int, n_edge_points: int) -> np.ndarray:
    if n_points < 2 * n_edge_points:
        raise ValueError(
            f"Too few scan points ({n_points}) for n_edge_points={n_edge_points}"
        )
    mask = np.zeros(n_points, dtype=bool)
    mask[:n_edge_points] = True
    mask[-n_edge_points:] = True
    return mask


def fit_tau_from_phase(
    frequency_hz: np.ndarray,
    z_scan: np.ndarray,
    n_edge_points: int,
) -> TauFitResult:
    """
    scan 両端だけを使って
        phi_env(f) = b - 2*pi*tau*f
    を fit する。
    """
    edge_mask = make_edge_mask(frequency_hz.size, n_edge_points)

    raw_phase = np.angle(z_scan)
    low_phase = np.unwrap(raw_phase[:n_edge_points])
    high_phase_base = np.unwrap(raw_phase[-n_edge_points:])

    f_edge = np.concatenate([frequency_hz[:n_edge_points], frequency_hz[-n_edge_points:]])
    f_ref = float(np.mean(f_edge))
    x_edge = f_edge - f_ref

    best = None
    for k in range(-5, 6):
        phase_edge = np.concatenate([low_phase, high_phase_base + 2.0 * np.pi * k])

        design = np.column_stack([x_edge, np.ones_like(x_edge)])
        slope, intercept = np.linalg.lstsq(design, phase_edge, rcond=None)[0]
        prediction = slope * x_edge + intercept
        residual = phase_edge - prediction
        rss = float(np.sum(residual**2))

        if best is None or rss < best["rss"]:
            best = {
                "k": k,
                "slope": float(slope),
                "intercept": float(intercept),
                "residual": residual,
                "rss": rss,
                "phase_edge": phase_edge,
            }

    assert best is not None

    slope = float(best["slope"])
    intercept = float(best["intercept"])
    tau = -slope / (2.0 * np.pi)

    residual_edge = np.asarray(best["residual"], dtype=float)
    rmse = float(np.sqrt(np.mean(residual_edge**2)))
    phase_edge = np.asarray(best["phase_edge"], dtype=float)
    ss_tot = float(np.sum((phase_edge - np.mean(phase_edge)) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1.0 - float(best["rss"]) / ss_tot

    # phi = slope*(f-f_ref)+intercept = b - 2*pi*tau*f
    intercept_b = intercept + 2.0 * np.pi * tau * f_ref

    return TauFitResult(
        tau_s=tau,
        intercept_b_rad=float(intercept_b),
        slope_rad_per_hz=slope,
        f_ref_hz=f_ref,
        rmse_edge_rad=rmse,
        r_squared_edge=float(r_squared),
        high_edge_branch_shift=int(best["k"]),
    )


def apply_tau_correction(
    z: np.ndarray,
    frequency_hz: np.ndarray | float,
    tau_s: float,
) -> np.ndarray:
    return z * np.exp(1j * 2.0 * np.pi * tau_s * np.asarray(frequency_hz))


def resonance_window_mask(z: np.ndarray, half_width_points: int) -> tuple[np.ndarray, int]:
    resonance_index = int(np.argmin(np.abs(z)))
    n = z.size
    i0 = max(0, resonance_index - half_width_points)
    i1 = min(n, resonance_index + half_width_points + 1)
    mask = np.zeros(n, dtype=bool)
    mask[i0:i1] = True
    return mask, resonance_index


def algebraic_circle_fit(z: np.ndarray) -> tuple[complex, float, float]:
    x = np.real(z)
    y = np.imag(z)

    design = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    d, e, f0 = np.linalg.lstsq(design, rhs, rcond=None)[0]

    xc = -d / 2.0
    yc = -e / 2.0
    radius_sq = xc**2 + yc**2 - f0
    radius = float(np.sqrt(max(radius_sq, 0.0)))

    center = complex(xc, yc)
    radial_residual = np.abs(z - center) - radius
    radial_rms = float(np.sqrt(np.mean(radial_residual**2)))
    return center, radius, radial_rms


def fit_circle_near_resonance(
    frequency_hz: np.ndarray,
    z_tau: np.ndarray,
    half_width_points: int,
) -> CircleFitResult:
    fit_mask, resonance_index = resonance_window_mask(z_tau, half_width_points)
    center, radius, radial_rms = algebraic_circle_fit(z_tau[fit_mask])

    return CircleFitResult(
        center=center,
        radius=radius,
        radial_rms=radial_rms,
        resonance_index=resonance_index,
        resonance_frequency_hz=float(frequency_hz[resonance_index]),
        resonance_point=complex(z_tau[resonance_index]),
    )


def compute_geometric_calibration(circle_fit: CircleFitResult) -> GeometricCalibration:
    c = circle_fit.center
    z_res = circle_fit.resonance_point

    # 円中心を挟んだ共振点の反対側の点
    P = 2.0 * c - z_res

    # P を x軸上へ
    alpha = float(np.angle(P))
    c_alpha = c * np.exp(-1j * alpha)
    P_alpha = P * np.exp(-1j * alpha)

    # P -> (1,0)
    a = float(np.abs(P_alpha))
    if a == 0:
        raise ZeroDivisionError("Computed amplitude a is zero")

    c_a = c_alpha / a

    # 1 を固定点として回し、円中心を x軸へ
    phi = float(np.angle(1.0 - c_a))

    return GeometricCalibration(
        alpha_rad=alpha,
        amplitude_a=a,
        phi_rad=phi,
        point_p=P,
        circle_center_after_tau=c,
        circle_radius_after_tau=circle_fit.radius,
    )


def apply_full_calibration(
    z: np.ndarray,
    frequency_hz: np.ndarray | float,
    tau_s: float,
    alpha_rad: float,
    amplitude_a: float,
    phi_rad: float,
) -> np.ndarray:
    z1 = apply_tau_correction(z, frequency_hz, tau_s)
    z2 = z1 * np.exp(-1j * alpha_rad)
    z3 = z2 / amplitude_a
    z4 = 1.0 + (z3 - 1.0) * np.exp(-1j * phi_rad)
    return z4


# =============================================================================
# FOLDER DISCOVERY / SORT
# =============================================================================

_FOLDER_RE = re.compile(
    r"^(?P<freq>[\d.]+)GHz_z=(?P<z>[\d.]+)mm_x=(?P<x>[\d.]+)mm(?:_(?P<tag>.+))?$"
)


def parse_folder_info(name: str) -> tuple[float, float, str, str]:
    m = _FOLDER_RE.match(name)
    if not m:
        return (999.0, 999.0, name, name)

    z = float(m.group("z"))
    x = float(m.group("x"))
    tag = m.group("tag") or ""
    return (z, x, tag, name)


def friendly_title(name: str) -> str:
    m = _FOLDER_RE.match(name)
    if not m:
        return name

    z = m.group("z")
    x = m.group("x")
    tag = m.group("tag")
    title = f"z={z} mm, x={x} mm"
    if tag:
        title += f"\n({tag})"
    return title


def discover_waveform_files(root: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []

    for folder in sorted(root.glob(TARGET_DIR_PATTERN)):
        if not folder.is_dir():
            continue

        wf_files = sorted(folder.glob(WAVEFORM_GLOB))
        if not wf_files:
            print(f"[skip] no waveform npz found in {folder.name}")
            continue

        if len(wf_files) > 1:
            print(f"[info] {folder.name}: multiple waveform files found, use first -> {wf_files[0].name}")

        pairs.append((folder.name, wf_files[0]))

    pairs.sort(key=lambda item: parse_folder_info(item[0]))
    return pairs


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def choose_event_indices(n_events: int, max_events: int | None) -> np.ndarray:
    if max_events is None or max_events >= n_events:
        return np.arange(n_events)
    return np.unique(np.linspace(0, n_events - 1, max_events, dtype=int))


def compute_pedestal_and_median_track(z_wave: np.ndarray) -> tuple[complex, np.ndarray]:
    baseline_stop = max(1, int(round(z_wave.shape[1] * PEDESTAL_FRACTION)))
    baseline = z_wave[:, :baseline_stop]

    pedestal = (
        np.median(np.real(baseline))
        + 1j * np.median(np.imag(baseline))
    )

    i_median = np.median(np.real(z_wave), axis=0)
    q_median = np.median(np.imag(z_wave), axis=0)
    median_track = i_median + 1j * q_median

    return pedestal, median_track


def robust_limits(
    selected_results: list[FolderResult],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    選択されたフォルダ群から robust な表示範囲を作る。

    灰色イベント群:
        AXIS_PERCENTILES の範囲だけを採用し、少数の外れ値は切る。

    中央値トラック・pedestal:
        形の主要部分なので、百分位に関係なく全点を必ず含める。

    selected_results が1要素なら各フォルダ個別の軸、
    1ページ分ならページ共通軸、全結果なら全体共通軸になる。
    """
    cloud_x: list[np.ndarray] = []
    cloud_y: list[np.ndarray] = []
    important_x: list[np.ndarray] = []
    important_y: list[np.ndarray] = []

    for result in selected_results:
        z = result.z_final

        # 軸決定用には描画イベントより少し少ないイベントで十分。
        event_idx = choose_event_indices(
            z.shape[0],
            min(MAX_EVENTS_TO_PLOT, 80),
        )
        sample_idx = np.arange(
            0,
            z.shape[1],
            max(1, PLOT_SAMPLE_STRIDE * 2),
        )
        sampled = z[np.ix_(event_idx, sample_idx)]

        x_event = np.real(sampled).ravel()
        y_event = np.imag(sampled).ravel()
        finite = np.isfinite(x_event) & np.isfinite(y_event)

        if np.any(finite):
            cloud_x.append(x_event[finite])
            cloud_y.append(y_event[finite])

        # 中央値トラックは全点を確実に表示する。
        median = result.median_track
        finite_median = np.isfinite(np.real(median)) & np.isfinite(np.imag(median))
        if np.any(finite_median):
            important_x.append(np.real(median[finite_median]))
            important_y.append(np.imag(median[finite_median]))

        if np.isfinite(result.pedestal.real) and np.isfinite(result.pedestal.imag):
            important_x.append(np.array([result.pedestal.real]))
            important_y.append(np.array([result.pedestal.imag]))

    if not cloud_x and not important_x:
        return (-1.0, 1.0), (-1.0, 1.0)

    low_pct, high_pct = AXIS_PERCENTILES

    if cloud_x:
        x_cloud = np.concatenate(cloud_x)
        y_cloud = np.concatenate(cloud_y)
        xlo, xhi = np.nanpercentile(x_cloud, [low_pct, high_pct])
        ylo, yhi = np.nanpercentile(y_cloud, [low_pct, high_pct])
    else:
        xlo = xhi = 0.0
        ylo = yhi = 0.0

    # median track と pedestal は必ず軸内に含める。
    if important_x:
        x_important = np.concatenate(important_x)
        y_important = np.concatenate(important_y)
        xlo = min(float(xlo), float(np.nanmin(x_important)))
        xhi = max(float(xhi), float(np.nanmax(x_important)))
        ylo = min(float(ylo), float(np.nanmin(y_important)))
        yhi = max(float(yhi), float(np.nanmax(y_important)))

    xspan = max(float(xhi - xlo), MIN_AXIS_SPAN)
    yspan = max(float(yhi - ylo), MIN_AXIS_SPAN)

    xcenter = 0.5 * (xlo + xhi)
    ycenter = 0.5 * (ylo + yhi)

    if FORCE_SQUARE_LIMITS:
        span = max(xspan, yspan)
        half = 0.5 * span * (1.0 + 2.0 * AXIS_PADDING_FRACTION)
        return (
            (xcenter - half, xcenter + half),
            (ycenter - half, ycenter + half),
        )

    xpad = AXIS_PADDING_FRACTION * xspan
    ypad = AXIS_PADDING_FRACTION * yspan
    return (
        (xlo - xpad, xhi + xpad),
        (ylo - ypad, yhi + ypad),
    )


def plot_folder_track(
    ax: plt.Axes,
    result: FolderResult,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    z_wave = result.z_final

    event_idx = choose_event_indices(z_wave.shape[0], MAX_EVENTS_TO_PLOT)
    sample_idx = np.arange(0, z_wave.shape[1], max(1, PLOT_SAMPLE_STRIDE))
    z_plot = z_wave[np.ix_(event_idx, sample_idx)]

    # 灰色の個別イベント
    for trace in z_plot:
        ax.plot(
            np.real(trace),
            np.imag(trace),
            color="0.65",
            linewidth=0.35,
            alpha=0.15,
            zorder=1,
        )

    # 中央値トラック（時間順グラデ）
    median_trace = result.median_track[sample_idx]
    points = np.column_stack([np.real(median_trace), np.imag(median_trace)])
    if len(points) >= 2:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap="plasma",
            linewidths=2.0,
            zorder=3,
        )
        lc.set_array(np.arange(segments.shape[0]))
        ax.add_collection(lc)

    # pedestal
    ax.scatter(
        result.pedestal.real,
        result.pedestal.imag,
        marker="X",
        s=38,
        color="black",
        zorder=4,
    )

    # 中央値トラックの始点終点
    ax.scatter(
        np.real(median_trace[0]),
        np.imag(median_trace[0]),
        s=18,
        marker="o",
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.0,
        zorder=5,
    )
    ax.scatter(
        np.real(median_trace[-1]),
        np.imag(median_trace[-1]),
        s=18,
        marker="s",
        facecolors="none",
        edgecolors="tab:green",
        linewidths=1.0,
        zorder=5,
    )

    ax.set_title(result.title, fontsize=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def save_page(
    fig: plt.Figure,
    pdf: PdfPages,
    output_png: Path,
) -> None:
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_png}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) calibration parameters from IQ scan
    # ------------------------------------------------------------
    frequency_hz, scan_ch0, scan_ch1 = load_iq_scan(IQ_SCAN)
    z_scan_raw = scan_ch0 + 1j * Q_SIGN * scan_ch1

    tau_fit = fit_tau_from_phase(frequency_hz, z_scan_raw, N_EDGE_POINTS)
    z_scan_tau = apply_tau_correction(z_scan_raw, frequency_hz, tau_fit.tau_s)
    circle_fit = fit_circle_near_resonance(
        frequency_hz, z_scan_tau, CIRCLE_HALF_WIDTH_POINTS
    )
    geom = compute_geometric_calibration(circle_fit)

    print("=" * 80)
    print("GLOBAL CALIBRATION")
    print("=" * 80)
    print(f"tau   = {tau_fit.tau_s:.12g} s ({tau_fit.tau_s * 1e9:.9f} ns)")
    print(f"alpha = {geom.alpha_rad:.12g} rad")
    print(f"a     = {geom.amplitude_a:.12g}")
    print(f"phi   = {geom.phi_rad:.12g} rad")
    print()

    # ------------------------------------------------------------
    # 2) discover folders and apply correction
    # ------------------------------------------------------------
    folder_pairs = discover_waveform_files(DATA_ROOT)
    if not folder_pairs:
        raise RuntimeError("No target waveform folders were found.")

    results: list[FolderResult] = []
    for folder_name, waveform_path in folder_pairs:
        print(f"[load] {folder_name}")
        ch0, ch1 = load_waveform(waveform_path)
        z_raw = ch0 + 1j * Q_SIGN * ch1

        z_final = apply_full_calibration(
            z=z_raw,
            frequency_hz=READOUT_FREQUENCY_HZ,
            tau_s=tau_fit.tau_s,
            alpha_rad=geom.alpha_rad,
            amplitude_a=geom.amplitude_a,
            phi_rad=geom.phi_rad,
        )

        pedestal, median_track = compute_pedestal_and_median_track(z_final)

        results.append(
            FolderResult(
                folder_name=folder_name,
                waveform_path=waveform_path,
                z_final=z_final,
                pedestal=pedestal,
                median_track=median_track,
                title=friendly_title(folder_name),
            )
        )

    # global モード用の軸。per_page / per_folder では後で再計算する。
    global_xlim, global_ylim = robust_limits(results)

    # ------------------------------------------------------------
    # 3) summary json
    # ------------------------------------------------------------
    summary = {
        "iq_scan": str(IQ_SCAN),
        "data_root": str(DATA_ROOT),
        "readout_frequency_hz": READOUT_FREQUENCY_HZ,
        "q_sign": Q_SIGN,
        "n_edge_points": N_EDGE_POINTS,
        "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
        "plot": {
            "max_events_to_plot": MAX_EVENTS_TO_PLOT,
            "plot_sample_stride": PLOT_SAMPLE_STRIDE,
            "pedestal_fraction": PEDESTAL_FRACTION,
            "nrows": NROWS,
            "ncols": NCOLS,
            "axis_mode": AXIS_MODE,
            "axis_percentiles": list(AXIS_PERCENTILES),
            "axis_padding_fraction": AXIS_PADDING_FRACTION,
            "force_square_limits": FORCE_SQUARE_LIMITS,
        },
        "calibration": {
            "tau_s": tau_fit.tau_s,
            "tau_ns": tau_fit.tau_s * 1e9,
            "intercept_b_rad": tau_fit.intercept_b_rad,
            "phase_fit_rmse_rad": tau_fit.rmse_edge_rad,
            "phase_fit_r_squared": tau_fit.r_squared_edge,
            "alpha_rad": geom.alpha_rad,
            "amplitude_a": geom.amplitude_a,
            "phi_rad": geom.phi_rad,
            "circle_center_real": circle_fit.center.real,
            "circle_center_imag": circle_fit.center.imag,
            "circle_radius": circle_fit.radius,
            "circle_radial_rms": circle_fit.radial_rms,
            "resonance_frequency_hz": circle_fit.resonance_frequency_hz,
        },
        "folders": [
            {
                "folder_name": r.folder_name,
                "waveform_path": str(r.waveform_path),
                "pedestal_real": r.pedestal.real,
                "pedestal_imag": r.pedestal.imag,
            }
            for r in results
        ],
        "global_xlim": [global_xlim[0], global_xlim[1]],
        "global_ylim": [global_ylim[0], global_ylim[1]],
    }

    summary_path = OUTPUT_DIR / "calibration_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {summary_path}")

    # ------------------------------------------------------------
    # 4) multipage plot
    # ------------------------------------------------------------
    per_page = NROWS * NCOLS
    n_pages = math.ceil(len(results) / per_page)

    pdf_path = OUTPUT_DIR / "corrected_iq_tracks_all_folders.pdf"
    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * per_page
            stop = min((page + 1) * per_page, len(results))
            page_results = results[start:stop]

            fig, axes = plt.subplots(
                NROWS, NCOLS,
                figsize=(16, 14),
                constrained_layout=True,
            )
            axes = np.array(axes).reshape(-1)

            if AXIS_MODE == "global":
                page_xlim, page_ylim = global_xlim, global_ylim
            elif AXIS_MODE == "per_page":
                page_xlim, page_ylim = robust_limits(page_results)
            elif AXIS_MODE == "per_folder":
                page_xlim = page_ylim = None
            else:
                raise ValueError(
                    f"Unknown AXIS_MODE={AXIS_MODE!r}. "
                    "Use 'per_folder', 'per_page', or 'global'."
                )

            for ax, result in zip(axes, page_results):
                if AXIS_MODE == "per_folder":
                    xlim, ylim = robust_limits([result])
                else:
                    assert page_xlim is not None and page_ylim is not None
                    xlim, ylim = page_xlim, page_ylim

                plot_folder_track(
                    ax,
                    result,
                    xlim=xlim,
                    ylim=ylim,
                )

            for ax in axes[len(page_results):]:
                ax.axis("off")

            fig.suptitle(
                "Final corrected IQ tracks for all waveform folders\n"
                "Thin gray: sampled events | Thick plasma line: median IQ track "
                "(purple early → yellow late) | Black X: median pedestal of first 10%\n"
                f"Axis mode: {AXIS_MODE}",
                fontsize=15,
            )

            # ページ下の共通ラベル
            fig.supxlabel("I_final", fontsize=12)
            fig.supylabel("Q_final", fontsize=12)

            png_path = OUTPUT_DIR / f"corrected_iq_tracks_page_{page+1:02d}.png"
            save_page(fig, pdf, png_path)

    print(f"[saved] {pdf_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
