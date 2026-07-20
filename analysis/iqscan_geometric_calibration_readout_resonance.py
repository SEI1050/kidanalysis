from __future__ import annotations

"""
IQ scan から幾何学的に calibration を行い、固定周波数 waveform にも同じ補正をかける。

手順
----
(1) IQ scan の位相 angle(z) を scan 両端だけ使って
        phi_env(f) = b - 2*pi*tau*f
    で最小二乗 fit し、tau を求める。

(2) tau を打ち消す:
        z_tau(f) = z_raw(f) * exp(+i 2*pi tau f)

(3) READOUT_FREQUENCY_HZ に最も近い IQ-scan 点を共振点として選び、
    その点の近傍の z_tau を円 fit して、中心 c と半径 r を求める。

(4) 上で選んだ readout 周波数の点 z_res に対して、円中心を挟んだ反対側の点
        P = 2c - z_res
    を定義する。

(5) P が x 軸上に乗るように、回転角 alpha を解析的に求める:
        alpha = arg(P)
        z_alpha = z_tau * exp(-i alpha)

(6) P が (1, 0) に乗るように、振幅 a を解析的に求める:
        a = |P * exp(-i alpha)| = |P|
        z_norm = z_alpha / a

(7) 1 を固定点にして円を回し、円中心が x 軸に乗るように
    phi を解析的に求める:
        c_norm = c * exp(-i alpha) / a
        phi = arg(1 - c_norm)
        z_final = 1 + (z_norm - 1) * exp(-i phi)

固定周波数 waveform (f = READOUT_FREQUENCY_HZ) に対しては
上の (2), (5), (6), (7) と同じ補正を順に適用する。

出力
----
- calibration_step1_tau_fit.png
- calibration_step2_circle_fit.png
- calibration_step3_alpha_a_phi.png
- waveform_before_after.png
- iqscan_geometric_calibration.pdf
- calibration_parameters.json
- corrected_waveform.npz
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection


# =============================================================================
# USER SETTINGS
# =============================================================================

IQ_SCAN = Path("/Volumes/NO NAME/data/20260709/iq_3.62K.npz")

WAVEFORM_FILE = Path(
    "/Volumes/NO NAME/data/20260709/5.501GHz_z=8.0mm_x=4.4mm_first/"
    "wf_260709_175104_49.78Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "iqscan_geometric_calibration_3.62K_5.501GHz"
)

READOUT_FREQUENCY_HZ = 5.501e9

# z = ch0 + i * Q_SIGN * ch1
Q_SIGN = +1

# tau fit に使う scan 両端の点数
N_EDGE_POINTS = 15

# 円 fit に使う共振点近傍の片側点数
CIRCLE_HALF_WIDTH_POINTS = 15

# waveform 可視化設定
MAX_EVENTS_TO_PLOT: int | None = 200
PLOT_SAMPLE_STRIDE = 5

SAVE_CORRECTED_NPZ = True


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class TauFitResult:
    tau_s: float
    intercept_b_rad: float
    slope_rad_per_hz: float
    f_ref_hz: float
    phase_fit_all_rad: np.ndarray
    edge_mask: np.ndarray
    edge_phase_used_rad: np.ndarray
    residual_edge_rad: np.ndarray
    rmse_edge_rad: float
    r_squared_edge: float
    high_edge_branch_shift: int


@dataclass(frozen=True)
class CircleFitResult:
    center: complex
    radius: float
    radial_rms: float
    fit_mask: np.ndarray
    resonance_index: int
    resonance_point: complex


@dataclass(frozen=True)
class GeometricCalibration:
    point_p: complex
    alpha_rad: float
    amplitude_a: float
    phi_rad: float
    center_after_tau: complex
    center_after_alpha: complex
    center_after_a: complex
    center_final: complex
    p_after_alpha: complex
    p_after_a: complex
    p_final: complex


# =============================================================================
# LOADERS
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


def load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Waveform file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())

        ch0_key = find_key_case_insensitive(keys, ("ch0", "channel0", "channel_0", "i"))
        ch1_key = find_key_case_insensitive(keys, ("ch1", "channel1", "channel_1", "q"))

        ch0 = np.asarray(npz[ch0_key], dtype=float)
        ch1 = np.asarray(npz[ch1_key], dtype=float)

        metadata: dict[str, np.ndarray] = {}
        for key in ("npts", "sample_rate", "ref_position", "daq_rate"):
            if key not in keys:
                continue
            try:
                value = np.asarray(npz[key])
            except ValueError:
                continue
            if value.dtype != object and value.size <= 100:
                metadata[key] = value

    if ch0.shape != ch1.shape:
        raise ValueError(f"ch0 shape {ch0.shape} and ch1 shape {ch1.shape} differ")

    if ch0.ndim == 1:
        ch0 = ch0[np.newaxis, :]
        ch1 = ch1[np.newaxis, :]

    if ch0.ndim != 2:
        raise ValueError(f"waveform must be 2D, got {ch0.shape}")

    if ch0.shape[0] > ch0.shape[1] and ch0.shape[1] <= 2000:
        print("[waveform] transposing to (events, samples)")
        ch0 = ch0.T
        ch1 = ch1.T

    return ch0, ch1, metadata


# =============================================================================
# FITS
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
    scan 両端を使って
        phi_env(f) = b - 2*pi*tau*f
    を fit する。

    数値安定性のため内部的には
        phi = slope * (f - f_ref) + intercept
    で fit し、tau = -slope / (2*pi) に戻す。
    """
    n = frequency_hz.size
    edge_mask = make_edge_mask(n, n_edge_points)

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
                "phase_edge": phase_edge,
                "prediction": prediction,
                "residual": residual,
                "rss": rss,
            }

    assert best is not None

    slope = float(best["slope"])
    intercept = float(best["intercept"])
    tau = -slope / (2.0 * np.pi)

    phase_fit_all = slope * (frequency_hz - f_ref) + intercept

    residual_edge = np.asarray(best["residual"], dtype=float)
    rmse = float(np.sqrt(np.mean(residual_edge**2)))
    phase_edge = np.asarray(best["phase_edge"], dtype=float)
    ss_tot = float(np.sum((phase_edge - np.mean(phase_edge)) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1.0 - float(best["rss"]) / ss_tot

    # phi = slope*(f-f_ref) + intercept = b - 2*pi*tau*f
    # => b = intercept + 2*pi*tau*f_ref
    intercept_b = intercept + 2.0 * np.pi * tau * f_ref

    return TauFitResult(
        tau_s=tau,
        intercept_b_rad=float(intercept_b),
        slope_rad_per_hz=slope,
        f_ref_hz=f_ref,
        phase_fit_all_rad=phase_fit_all,
        edge_mask=edge_mask,
        edge_phase_used_rad=phase_edge,
        residual_edge_rad=residual_edge,
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


def resonance_window_mask(
    frequency_hz: np.ndarray,
    readout_frequency_hz: float,
    half_width_points: int,
) -> tuple[np.ndarray, int]:
    """readout 周波数に最も近い scan 点を中心に円 fit 範囲を作る。"""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    resonance_index = int(np.argmin(np.abs(frequency_hz - readout_frequency_hz)))

    n = frequency_hz.size
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
    readout_frequency_hz: float,
    half_width_points: int,
) -> CircleFitResult:
    fit_mask, resonance_index = resonance_window_mask(
        frequency_hz=frequency_hz,
        readout_frequency_hz=readout_frequency_hz,
        half_width_points=half_width_points,
    )
    center, radius, radial_rms = algebraic_circle_fit(z_tau[fit_mask])

    # |z| 最小点ではなく、readout 周波数に最も近い scan 点を z_res とする。
    resonance_point = complex(z_tau[resonance_index])
    return CircleFitResult(
        center=center,
        radius=radius,
        radial_rms=radial_rms,
        fit_mask=fit_mask,
        resonance_index=resonance_index,
        resonance_point=resonance_point,
    )


def compute_geometric_calibration(
    circle_result: CircleFitResult,
) -> GeometricCalibration:
    c = circle_result.center
    z_res = circle_result.resonance_point

    # 円中心を挟んで共振点の反対側
    P = 2.0 * c - z_res

    alpha = float(np.angle(P))
    rot_alpha = np.exp(-1j * alpha)

    c_alpha = c * rot_alpha
    P_alpha = P * rot_alpha

    a = float(np.abs(P_alpha))
    if a == 0:
        raise ZeroDivisionError("Computed amplitude a is zero")

    c_a = c_alpha / a
    P_a = P_alpha / a

    # 1 を固定点として回し、中心を x 軸に乗せる
    # c_final = 1 + (c_a - 1) * exp(-i phi)
    # これの虚部が 0 になるように
    phi = float(np.angle(1.0 - c_a))
    rot_phi = np.exp(-1j * phi)

    c_final = 1.0 + (c_a - 1.0) * rot_phi
    P_final = 1.0 + (P_a - 1.0) * rot_phi

    return GeometricCalibration(
        point_p=P,
        alpha_rad=alpha,
        amplitude_a=a,
        phi_rad=phi,
        center_after_tau=c,
        center_after_alpha=c_alpha,
        center_after_a=c_a,
        center_final=c_final,
        p_after_alpha=P_alpha,
        p_after_a=P_a,
        p_final=P_final,
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
# PLOTTING HELPERS
# =============================================================================

def align_phase_to_reference(raw_phase: np.ndarray, reference_phase: np.ndarray) -> np.ndarray:
    return raw_phase + 2.0 * np.pi * np.round((reference_phase - raw_phase) / (2.0 * np.pi))


def add_frequency_colored_iq(
    ax: plt.Axes,
    z: np.ndarray,
    frequency_hz: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    extra_points: list[tuple[complex, str, str, int]] | None = None,
) -> None:
    f_ghz = frequency_hz / 1e9
    ax.plot(np.real(z), np.imag(z), color="0.5", linewidth=0.9, alpha=0.65)
    sc = ax.scatter(np.real(z), np.imag(z), c=f_ghz, s=28, cmap="viridis", zorder=2)

    ax.scatter(
        np.real(z[0]), np.imag(z[0]),
        s=70, marker="o", facecolors="none", edgecolors="tab:blue", linewidths=1.6,
        label=f"start {f_ghz[0]:.6f} GHz", zorder=3
    )
    ax.scatter(
        np.real(z[-1]), np.imag(z[-1]),
        s=70, marker="s", facecolors="none", edgecolors="tab:green", linewidths=1.6,
        label=f"end {f_ghz[-1]:.6f} GHz", zorder=3
    )

    i_rf = int(np.argmin(np.abs(frequency_hz - READOUT_FREQUENCY_HZ)))
    ax.scatter(
        np.real(z[i_rf]), np.imag(z[i_rf]),
        s=120, marker="*", color="tab:red",
        label=f"nearest to {READOUT_FREQUENCY_HZ/1e9:.6f} GHz", zorder=4
    )

    if extra_points:
        for point, marker, label, size in extra_points:
            ax.scatter(point.real, point.imag, marker=marker, s=size, label=label, zorder=5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(sc, ax=ax, label="frequency [GHz]")


def add_circle(ax: plt.Axes, center: complex, radius: float, label: str = "circle fit") -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    circle = center + radius * np.exp(1j * theta)
    ax.plot(circle.real, circle.imag, color="tab:orange", linewidth=2.0, label=label, zorder=4)
    ax.scatter(center.real, center.imag, marker="X", s=80, color="tab:red", label="circle center", zorder=5)


def choose_event_indices(n_events: int, max_events: int | None) -> np.ndarray:
    if max_events is None or max_events >= n_events:
        return np.arange(n_events)
    return np.unique(np.linspace(0, n_events - 1, max_events, dtype=int))


def add_iq_waveform_trajectories(
    ax: plt.Axes,
    z_wave: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    event_idx = choose_event_indices(z_wave.shape[0], MAX_EVENTS_TO_PLOT)
    sample_idx = np.arange(0, z_wave.shape[1], max(1, PLOT_SAMPLE_STRIDE))
    z_plot = z_wave[np.ix_(event_idx, sample_idx)]

    segments = np.stack([np.real(z_plot), np.imag(z_plot)], axis=-1)
    lc = LineCollection(segments, colors="0.55", linewidths=0.35, alpha=0.16, rasterized=True, zorder=1)
    ax.add_collection(lc)

    median_trace = np.median(z_wave, axis=0)[sample_idx]
    points = np.column_stack([np.real(median_trace), np.imag(median_trace)])
    seg = np.stack([points[:-1], points[1:]], axis=1)

    grad = LineCollection(seg, cmap="plasma", linewidths=2.6, zorder=3)
    grad.set_array(np.arange(seg.shape[0]))
    ax.add_collection(grad)

    baseline_stop = max(1, z_wave.shape[1] // 10)
    pedestal = complex(np.median(z_wave[:, :baseline_stop]))
    ax.scatter(
        pedestal.real, pedestal.imag,
        marker="X", s=90, color="black",
        label="median pedestal (first 10%)", zorder=4
    )

    all_x = np.real(z_plot).ravel()
    all_y = np.imag(z_plot).ravel()
    finite = np.isfinite(all_x) & np.isfinite(all_y)
    if np.any(finite):
        xlim = np.nanpercentile(all_x[finite], [0.2, 99.8])
        ylim = np.nanpercentile(all_y[finite], [0.2, 99.8])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{title}\n{event_idx.size}/{z_wave.shape[0]} events, sample stride={PLOT_SAMPLE_STRIDE}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(grad, ax=ax, label="sample order of median track")


def save_figure(fig: plt.Figure, png_path: Path, pdf: PdfPages) -> None:
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frequency_hz, ch0_scan, ch1_scan = load_iq_scan(IQ_SCAN)
    wave_ch0, wave_ch1, waveform_metadata = load_waveform(WAVEFORM_FILE)

    z_scan_raw = ch0_scan + 1j * Q_SIGN * ch1_scan
    z_wave_raw = wave_ch0 + 1j * Q_SIGN * wave_ch1

    tau_fit = fit_tau_from_phase(frequency_hz, z_scan_raw, N_EDGE_POINTS)
    z_scan_tau = apply_tau_correction(z_scan_raw, frequency_hz, tau_fit.tau_s)

    circle_fit = fit_circle_near_resonance(
        frequency_hz=frequency_hz,
        z_tau=z_scan_tau,
        readout_frequency_hz=READOUT_FREQUENCY_HZ,
        half_width_points=CIRCLE_HALF_WIDTH_POINTS,
    )
    geom = compute_geometric_calibration(circle_fit)

    selected_resonance_frequency_hz = float(
        frequency_hz[circle_fit.resonance_index]
    )
    readout_frequency_error_hz = (
        selected_resonance_frequency_hz - READOUT_FREQUENCY_HZ
    )
    print(
        "[resonance] use scan point nearest to readout: "
        f"requested={READOUT_FREQUENCY_HZ/1e9:.9f} GHz, "
        f"selected={selected_resonance_frequency_hz/1e9:.9f} GHz, "
        f"delta={readout_frequency_error_hz/1e3:+.3f} kHz"
    )

    # P is used only to determine alpha and a.
    # The resulting transformations are applied to every IQ-scan point, not only P.
    z_scan_alpha = z_scan_tau * np.exp(-1j * geom.alpha_rad)
    z_scan_a = z_scan_alpha / geom.amplitude_a
    z_scan_final = 1.0 + (z_scan_a - 1.0) * np.exp(-1j * geom.phi_rad)

    z_wave_final = apply_full_calibration(
        z=z_wave_raw,
        frequency_hz=READOUT_FREQUENCY_HZ,
        tau_s=tau_fit.tau_s,
        alpha_rad=geom.alpha_rad,
        amplitude_a=geom.amplitude_a,
        phi_rad=geom.phi_rad,
    )

    # 途中段階の waveform も可視化用に
    z_wave_tau = apply_tau_correction(z_wave_raw, READOUT_FREQUENCY_HZ, tau_fit.tau_s)
    z_wave_alpha = z_wave_tau * np.exp(-1j * geom.alpha_rad)
    z_wave_a = z_wave_alpha / geom.amplitude_a

    pdf_path = OUTPUT_DIR / "iqscan_geometric_calibration.pdf"
    f_ghz = frequency_hz / 1e9

    with PdfPages(pdf_path) as pdf:
        # ---------------------------------------------------------------------
        # Figure 1: tau fit
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
        fig.suptitle(
            "Step 1: phase fit for tau\n"
            r"$\phi_{\rm env}(f)=b-2\pi\tau f$ (use only scan edges)",
            fontsize=16,
        )

        add_frequency_colored_iq(
            axes[0, 0],
            z_scan_raw,
            frequency_hz,
            title="Raw IQ scan",
            xlabel="ch0",
            ylabel=f"{'+' if Q_SIGN > 0 else '-'} ch1",
        )

        axes[0, 1].plot(f_ghz, np.abs(z_scan_raw), "o-", ms=4, label="|z_raw|")
        axes[0, 1].scatter(
            f_ghz[tau_fit.edge_mask],
            np.abs(z_scan_raw[tau_fit.edge_mask]),
            s=55, marker="s", facecolors="none", edgecolors="tab:orange",
            label="points used in phase fit",
        )
        axes[0, 1].set_title("Magnitude of raw IQ scan")
        axes[0, 1].set_xlabel("frequency [GHz]")
        axes[0, 1].set_ylabel("|z_raw|")
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend(fontsize=8)

        phase_for_plot = align_phase_to_reference(np.angle(z_scan_raw), tau_fit.phase_fit_all_rad)
        axes[1, 0].plot(f_ghz, phase_for_plot, "o-", ms=4, label="angle(z_raw)")
        axes[1, 0].plot(f_ghz, tau_fit.phase_fit_all_rad, linewidth=2.0, color="tab:orange", label="fit")
        edge_indices = np.flatnonzero(tau_fit.edge_mask)
        axes[1, 0].scatter(
            f_ghz[edge_indices],
            tau_fit.edge_phase_used_rad,
            s=60, marker="s", facecolors="none", edgecolors="tab:red", linewidths=1.6,
            label="fit points",
            zorder=4,
        )
        axes[1, 0].axvline(READOUT_FREQUENCY_HZ / 1e9, color="0.3", linestyle=":", label="waveform RF")
        axes[1, 0].set_title("Least-squares fit for phase")
        axes[1, 0].set_xlabel("frequency [GHz]")
        axes[1, 0].set_ylabel("phase [rad]")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].axhline(0.0, color="0.3", linewidth=1.0)
        axes[1, 1].scatter(
            f_ghz[edge_indices],
            tau_fit.residual_edge_rad,
            s=42,
            label="edge residual",
        )
        axes[1, 1].set_title(
            "Residual of phase fit\n"
            f"RMSE={tau_fit.rmse_edge_rad:.4g} rad, R^2={tau_fit.r_squared_edge:.6f}"
        )
        axes[1, 1].set_xlabel("frequency [GHz]")
        axes[1, 1].set_ylabel("residual [rad]")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend(fontsize=8)

        text = (
            f"tau = {tau_fit.tau_s*1e9:.6f} ns\n"
            f"b   = {tau_fit.intercept_b_rad:.6f} rad\n"
            f"slope = {tau_fit.slope_rad_per_hz:.6e} rad/Hz\n"
            f"2pi branch shift (high edge) = {tau_fit.high_edge_branch_shift}"
        )
        axes[1, 1].text(
            0.03, 0.97, text, transform=axes[1, 1].transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
        )

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, OUTPUT_DIR / "calibration_step1_tau_fit.png", pdf)

        # ---------------------------------------------------------------------
        # Figure 2: circle fit after tau correction
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3))
        fig.suptitle(
            "Step 2: cancel tau, then circle-fit near resonance",
            fontsize=16,
        )

        add_frequency_colored_iq(
            axes[0],
            z_scan_tau,
            frequency_hz,
            title="After tau correction",
            xlabel="I_tau",
            ylabel="Q_tau",
            extra_points=[
                (circle_fit.center, "X", "circle center", 85),
                (circle_fit.resonance_point, "o", "readout resonance point", 70),
                (geom.point_p, "*", "P = 2c - z_res", 130),
            ],
        )
        add_circle(axes[0], circle_fit.center, circle_fit.radius)

        axes[1].plot(f_ghz, np.abs(z_scan_tau), "o-", ms=4, label="|z_tau|")
        axes[1].scatter(
            f_ghz[circle_fit.fit_mask],
            np.abs(z_scan_tau[circle_fit.fit_mask]),
            s=55, marker="s", facecolors="none", edgecolors="tab:orange",
            label="circle-fit region",
        )
        axes[1].axvline(
            selected_resonance_frequency_hz / 1e9,
            color="tab:red",
            linestyle=":",
            label="readout resonance point",
        )
        axes[1].set_title("Magnitude after tau correction")
        axes[1].set_xlabel("frequency [GHz]")
        axes[1].set_ylabel("|z_tau|")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)

        phase_tau = np.unwrap(np.angle(z_scan_tau))
        axes[2].plot(f_ghz, phase_tau, "o-", ms=4, label="angle(z_tau)")
        axes[2].scatter(
            f_ghz[circle_fit.fit_mask],
            phase_tau[circle_fit.fit_mask],
            s=50, marker="s", facecolors="none", edgecolors="tab:orange",
            label="circle-fit region",
        )
        axes[2].set_title(
            "Phase after tau correction\n"
            f"circle radial RMS = {circle_fit.radial_rms:.4g}"
        )
        axes[2].set_xlabel("frequency [GHz]")
        axes[2].set_ylabel("phase [rad]")
        axes[2].grid(alpha=0.3)
        axes[2].legend(fontsize=8)

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_figure(fig, OUTPUT_DIR / "calibration_step2_circle_fit.png", pdf)

        # ---------------------------------------------------------------------
        # Figure 3: alpha, a, phi
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
        fig.suptitle(
            "Step 3: geometric normalization by alpha, a, phi",
            fontsize=16,
        )

        add_frequency_colored_iq(
            axes[0, 0],
            z_scan_tau,
            frequency_hz,
            title="Before alpha correction",
            xlabel="I_tau",
            ylabel="Q_tau",
            extra_points=[
                (geom.point_p, "*", "P", 130),
                (circle_fit.center, "X", "center", 85),
            ],
        )
        add_circle(axes[0, 0], circle_fit.center, circle_fit.radius)
        axes[0, 0].legend(fontsize=8, loc="best")

        add_frequency_colored_iq(
            axes[0, 1],
            z_scan_alpha,
            frequency_hz,
            title=f"After alpha correction (alpha = {geom.alpha_rad:.6f} rad)",
            xlabel="I_alpha",
            ylabel="Q_alpha",
            extra_points=[
                (geom.p_after_alpha, "*", "P after alpha", 130),
                (geom.center_after_alpha, "X", "center after alpha", 85),
            ],
        )

        # alpha は原点まわりの剛体回転なので、円中心も同じ角度だけ回り、
        # 半径は変わらない。P から求めた alpha は円全体に適用される。
        add_circle(
            axes[0, 1],
            geom.center_after_alpha,
            circle_fit.radius,
            label="transformed fitted circle",
        )
        axes[0, 1].legend(fontsize=8, loc="best")

        add_frequency_colored_iq(
            axes[1, 0],
            z_scan_a,
            frequency_hz,
            title=f"After amplitude normalization (a = {geom.amplitude_a:.6g})",
            xlabel="I_norm",
            ylabel="Q_norm",
            extra_points=[
                (geom.p_after_a, "*", "P -> (1,0)", 130),
                (geom.center_after_a, "X", "center after a", 85),
            ],
        )
        # a で全複素平面を割るため、円中心と半径もともに 1/a 倍される。
        radius_after_a = circle_fit.radius / geom.amplitude_a
        add_circle(
            axes[1, 0],
            geom.center_after_a,
            radius_after_a,
            label="transformed fitted circle",
        )
        axes[1, 0].axhline(0.0, color="0.3", linewidth=1.0)
        axes[1, 0].axvline(1.0, color="0.3", linewidth=1.0, linestyle=":")
        axes[1, 0].scatter(1.0, 0.0, marker="+", s=120, color="black", label="(1,0)", zorder=5)
        axes[1, 0].legend(fontsize=8, loc="best")

        add_frequency_colored_iq(
            axes[1, 1],
            z_scan_final,
            frequency_hz,
            title=f"Final correction (phi = {geom.phi_rad:.6f} rad)",
            xlabel="I_final",
            ylabel="Q_final",
            extra_points=[
                (geom.center_final, "X", "center final", 85),
                (geom.p_final, "*", "P final", 130),
            ],
        )
        # phi は (1,0) を固定点とする剛体回転。
        # したがって円中心は同じ写像で動き、半径は radius_after_a のまま。
        add_circle(
            axes[1, 1],
            geom.center_final,
            radius_after_a,
            label="transformed fitted circle",
        )
        axes[1, 1].axhline(0.0, color="0.3", linewidth=1.0)
        axes[1, 1].axvline(1.0, color="0.3", linewidth=1.0, linestyle=":")
        axes[1, 1].scatter(1.0, 0.0, marker="+", s=120, color="black", label="(1,0)", zorder=5)
        axes[1, 1].legend(fontsize=8, loc="best")

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, OUTPUT_DIR / "calibration_step3_alpha_a_phi.png", pdf)

        # ---------------------------------------------------------------------
        # Figure 4: waveform before/after
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0))
        fig.suptitle(
            "Waveform IQ trajectories before / after calibration",
            fontsize=16,
        )

        add_iq_waveform_trajectories(
            axes[0, 0], z_wave_raw,
            title="Raw waveform IQ",
            xlabel="ch0",
            ylabel=f"{'+' if Q_SIGN > 0 else '-'} ch1",
        )
        add_iq_waveform_trajectories(
            axes[0, 1], z_wave_tau,
            title="After tau correction",
            xlabel="I_tau",
            ylabel="Q_tau",
        )
        add_iq_waveform_trajectories(
            axes[1, 0], z_wave_a,
            title="After alpha + a correction",
            xlabel="I_norm",
            ylabel="Q_norm",
        )
        add_iq_waveform_trajectories(
            axes[1, 1], z_wave_final,
            title="Final corrected waveform IQ",
            xlabel="I_final",
            ylabel="Q_final",
        )

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(fig, OUTPUT_DIR / "waveform_before_after.png", pdf)

    print(f"[saved] {pdf_path}")

    params = {
        "iq_scan_file": str(IQ_SCAN),
        "waveform_file": str(WAVEFORM_FILE),
        "readout_frequency_hz": READOUT_FREQUENCY_HZ,
        "q_sign": Q_SIGN,
        "n_edge_points": N_EDGE_POINTS,
        "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
        "tau_fit": {
            "tau_s": tau_fit.tau_s,
            "tau_ns": tau_fit.tau_s * 1e9,
            "intercept_b_rad": tau_fit.intercept_b_rad,
            "slope_rad_per_hz": tau_fit.slope_rad_per_hz,
            "f_ref_hz": tau_fit.f_ref_hz,
            "rmse_edge_rad": tau_fit.rmse_edge_rad,
            "r_squared_edge": tau_fit.r_squared_edge,
            "high_edge_branch_shift": tau_fit.high_edge_branch_shift,
        },
        "circle_fit": {
            "center_real": circle_fit.center.real,
            "center_imag": circle_fit.center.imag,
            "radius": circle_fit.radius,
            "radial_rms": circle_fit.radial_rms,
            "resonance_selection": "nearest IQ-scan point to READOUT_FREQUENCY_HZ",
            "requested_readout_frequency_hz": READOUT_FREQUENCY_HZ,
            "resonance_index": circle_fit.resonance_index,
            "resonance_frequency_hz": selected_resonance_frequency_hz,
            "readout_frequency_error_hz": readout_frequency_error_hz,
            "resonance_point_real": circle_fit.resonance_point.real,
            "resonance_point_imag": circle_fit.resonance_point.imag,
        },
        "geometric_calibration": {
            "point_p_real": geom.point_p.real,
            "point_p_imag": geom.point_p.imag,
            "alpha_rad": geom.alpha_rad,
            "amplitude_a": geom.amplitude_a,
            "phi_rad": geom.phi_rad,
            "center_after_tau_real": geom.center_after_tau.real,
            "center_after_tau_imag": geom.center_after_tau.imag,
            "center_after_a_real": geom.center_after_a.real,
            "center_after_a_imag": geom.center_after_a.imag,
            "center_final_real": geom.center_final.real,
            "center_final_imag": geom.center_final.imag,
        },
        "final_transform": {
            "description": (
                "z_final = 1 + (((z_raw * exp(i 2pi tau f)) * exp(-i alpha)) / a - 1) * exp(-i phi)"
            )
        },
    }

    param_path = OUTPUT_DIR / "calibration_parameters.json"
    param_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {param_path}")

    if SAVE_CORRECTED_NPZ:
        corrected_path = OUTPUT_DIR / "corrected_waveform.npz"
        np.savez_compressed(
            corrected_path,
            ch0_corrected=np.real(z_wave_final),
            ch1_corrected=Q_SIGN * np.imag(z_wave_final),
            tau_s=tau_fit.tau_s,
            alpha_rad=geom.alpha_rad,
            amplitude_a=geom.amplitude_a,
            phi_rad=geom.phi_rad,
            readout_frequency_hz=READOUT_FREQUENCY_HZ,
            original_ch0_shape=np.array(wave_ch0.shape),
            **{f"original_metadata__{k}": v for k, v in waveform_metadata.items()},
        )
        print(f"[saved] {corrected_path}")

    print()
    print("=" * 78)
    print("GEOMETRIC CALIBRATION RESULT")
    print("=" * 78)
    print(f"tau   = {tau_fit.tau_s:.12g} s ({tau_fit.tau_s*1e9:.9f} ns)")
    print(f"b     = {tau_fit.intercept_b_rad:.12g} rad")
    print(f"alpha = {geom.alpha_rad:.12g} rad")
    print(f"a     = {geom.amplitude_a:.12g}")
    print(f"phi   = {geom.phi_rad:.12g} rad")
    print(
        "resonance frequency used = "
        f"{selected_resonance_frequency_hz:.12g} Hz "
        f"(nearest scan point to readout, delta={readout_frequency_error_hz:+.6g} Hz)"
    )
    print(f"circle center after tau = {circle_fit.center.real:.12g}{circle_fit.center.imag:+.12g}j")
    print(f"P                    = {geom.point_p.real:.12g}{geom.point_p.imag:+.12g}j")
    print(
        "final transform: z_final = 1 + (((z_raw * exp(i 2pi tau f)) * exp(-i alpha)) / a - 1) * exp(-i phi)"
    )


if __name__ == "__main__":
    main()
