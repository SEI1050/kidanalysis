from __future__ import annotations

"""
IQ scan から配線・位相シフタ等の environment 項

    C_env(f) = a * exp[i(alpha - 2*pi*tau*f)]

を推定し、固定周波数で取得した waveform に

    z_corrected = z_raw / C_env(f_readout)

を適用する。

出力
------
1. calibration_process.png
   IQ scan、振幅、位相直線 fit、fit 残差
2. corrected_iqscan.png
   補正後 IQ scan と円 fit
3. waveform_before_after.png
   waveform の補正前後 IQ 軌跡
4. iq_environment_calibration.pdf
   上記の図をまとめた PDF
5. calibration_parameters.json
   tau, a, 補正因子など
6. corrected_waveform.npz
   補正後 ch0/ch1（SAVE_CORRECTED_NPZ=True の場合）
"""

from dataclasses import dataclass
import json
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

IQ_SCAN = Path(
    "/Volumes/NO NAME/data/20260709/iq_3.62K.npz"
)

WAVEFORM_FILE = Path(
    "/Volumes/NO NAME/data/20260709/5.501GHz_z=8.0mm_x=4.4mm_first/"
    "wf_260709_175104_49.78Hz.npz"
)

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "iq_environment_calibration_3.62K_5.501GHz"
)

# waveform を取得した RF 周波数
READOUT_FREQUENCY_HZ = 5.501e9

# IQ の定義。通常は z = ch0 + i*ch1 なので +1。
# 補正後の回転方向がおかしい場合のみ -1 を試す。
Q_SIGN = +1

# 位相直線 fit に使う scan 両端の点数。
# 51 点 scan なら 6～10 点程度が目安。
N_EDGE_POINTS = 20

# 補正後の円 fit に使う共振点付近の片側点数。
# 例: 15 なら共振点を中心に最大 31 点。
CIRCLE_HALF_WIDTH_POINTS = 15

# True:
#   a*exp(i phi) を丸ごと割るため、補正後 waveform は無次元。
# False:
#   exp(i phi) だけを割り、振幅の単位を保つ。
APPLY_AMPLITUDE_NORMALIZATION = True

# waveform の全イベントを描くと重い場合は 200 などにする。
# None なら全イベント。
MAX_EVENTS_TO_PLOT: int | None = None

# IQ 軌跡を描画するときのサンプル間引き。
# 5000 sample なら 5 で十分滑らか。
PLOT_SAMPLE_STRIDE = 5

SAVE_CORRECTED_NPZ = True


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class EnvironmentFit:
    amplitude: float
    tau_s: float
    slope_rad_per_hz: float
    f_ref_hz: float
    phi_ref_rad: float
    alpha_unwrapped_rad: float
    alpha_wrapped_rad: float
    phase_fit_rad: np.ndarray
    edge_mask: np.ndarray
    edge_phase_rad: np.ndarray
    rmse_rad: float
    r_squared: float
    high_edge_branch_shift: int

    def phase_at(self, frequency_hz: float | np.ndarray) -> np.ndarray:
        frequency_hz = np.asarray(frequency_hz, dtype=float)
        return self.phi_ref_rad + self.slope_rad_per_hz * (
            frequency_hz - self.f_ref_hz
        )

    def factor_at(
        self,
        frequency_hz: float,
        normalize_amplitude: bool = True,
    ) -> complex:
        amplitude = self.amplitude if normalize_amplitude else 1.0
        return complex(
            amplitude * np.exp(1j * float(self.phase_at(frequency_hz)))
        )


# =============================================================================
# LOADERS
# =============================================================================

def load_iq_scan(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dd[:,0]=frequency, dd[:,1]=ch0, dd[:,2]=ch1 を読む。"""
    if not path.exists():
        raise FileNotFoundError(f"IQ scan file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        if "dd" not in npz:
            raise KeyError(
                f"'dd' is not in {path.name}. Available keys: {list(npz.keys())}"
            )
        dd = np.asarray(npz["dd"], dtype=float)

    if dd.ndim != 2 or dd.shape[1] < 3:
        raise ValueError(
            f"dd must have shape (N, >=3), but got {dd.shape}"
        )

    frequency_hz = convert_frequency_to_hz(dd[:, 0])
    ch0 = dd[:, 1]
    ch1 = dd[:, 2]

    order = np.argsort(frequency_hz)
    return frequency_hz[order], ch0[order], ch1[order]


def convert_frequency_to_hz(frequency: np.ndarray) -> np.ndarray:
    """
    周波数軸が Hz/GHz/MHz のどれで保存されていても概ね判定する。

    5.5       -> GHz と解釈
    5500      -> MHz と解釈
    5.5e9     -> Hz と解釈
    """
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


def find_key_case_insensitive(
    keys: list[str],
    candidates: tuple[str, ...],
) -> str:
    lower_to_original = {key.lower(): key for key in keys}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise KeyError(
        f"None of keys {candidates} were found. Available keys: {keys}"
    )


def load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """waveform npz から ch0/ch1 を読む。"""
    if not path.exists():
        raise FileNotFoundError(f"Waveform file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())

        ch0_key = find_key_case_insensitive(
            keys, ("ch0", "channel0", "channel_0", "i")
        )
        ch1_key = find_key_case_insensitive(
            keys, ("ch1", "channel1", "channel_1", "q")
        )

        ch0 = np.asarray(npz[ch0_key], dtype=float)
        ch1 = np.asarray(npz[ch1_key], dtype=float)

        # 小さいメタデータだけ保持する。
        # 数値スカラーのメタデータだけ保持する。
        # deltat などの object 配列は allow_pickle=False では読めないため、
        # 必要なキーだけを明示的に読む。
        metadata: dict[str, np.ndarray] = {}

        metadata_keys = (
            "npts",
            "sample_rate",
            "ref_position",
            "daq_rate",
        )

        for key in metadata_keys:
            if key not in keys:
                continue

            try:
                value = np.asarray(npz[key])
            except ValueError as exc:
                print(f"[waveform] skip metadata {key!r}: {exc}")
                continue

            if value.dtype != object and value.size <= 100:
                metadata[key] = value

    if ch0.shape != ch1.shape:
        raise ValueError(
            f"ch0 and ch1 shapes differ: {ch0.shape} vs {ch1.shape}"
        )

    if ch0.ndim == 1:
        ch0 = ch0[np.newaxis, :]
        ch1 = ch1[np.newaxis, :]

    if ch0.ndim != 2:
        raise ValueError(
            f"Expected waveform shape (events, samples), got {ch0.shape}"
        )

    # 通常は events < samples。逆なら転置する。
    if ch0.shape[0] > ch0.shape[1] and ch0.shape[1] <= 2000:
        print("[waveform] Transposing arrays to (events, samples)")
        ch0 = ch0.T
        ch1 = ch1.T

    return ch0, ch1, metadata


# =============================================================================
# CALIBRATION
# =============================================================================

def make_edge_mask(n_points: int, n_edge: int) -> np.ndarray:
    if n_points < 6:
        raise ValueError("At least 6 IQ scan points are required.")

    n_edge = min(n_edge, max(2, n_points // 3))
    mask = np.zeros(n_points, dtype=bool)
    mask[:n_edge] = True
    mask[-n_edge:] = True
    return mask


def fit_environment_phase(
    frequency_hz: np.ndarray,
    z_scan: np.ndarray,
    n_edge_points: int,
) -> EnvironmentFit:
    """
    scan 両端を off-resonance とみなし、

        phi_env(f) = alpha - 2*pi*tau*f

    を最小二乗 fit する。

    数値安定性のため実際には

        phi_env(f) = phi_ref + slope*(f-f_ref)

    を fit し、slope = -2*pi*tau とする。

    共振部をまたいだ phase unwrap の 2*pi 分岐問題を避けるため、
    low edge と high edge を個別に unwrap し、high edge に加える
    2*pi*k を探索する。
    """
    n = frequency_hz.size
    edge_mask = make_edge_mask(n, n_edge_points)
    n_edge = int(edge_mask.sum() // 2)

    raw_angle = np.angle(z_scan)
    low_phase = np.unwrap(raw_angle[:n_edge])
    high_phase_base = np.unwrap(raw_angle[-n_edge:])

    f_edge = np.concatenate(
        [frequency_hz[:n_edge], frequency_hz[-n_edge:]]
    )
    f_ref_hz = float(np.mean(f_edge))
    x_edge = f_edge - f_ref_hz

    best: dict[str, object] | None = None

    # 共振をまたいだ後の high-frequency 側に 2*pi*k を加え、
    # 両端が最もよく一本の直線に乗る分岐を選ぶ。
    for k in range(-5, 6):
        phase_edge = np.concatenate(
            [low_phase, high_phase_base + 2.0 * np.pi * k]
        )

        design = np.column_stack([x_edge, np.ones_like(x_edge)])
        slope, intercept = np.linalg.lstsq(
            design, phase_edge, rcond=None
        )[0]
        prediction = slope * x_edge + intercept
        residual = phase_edge - prediction
        rss = float(np.sum(residual**2))

        if best is None or rss < float(best["rss"]):
            best = {
                "k": k,
                "phase_edge": phase_edge,
                "slope": float(slope),
                "intercept": float(intercept),
                "prediction": prediction,
                "residual": residual,
                "rss": rss,
            }

    if best is None:
        raise RuntimeError("Environment phase fit failed.")

    slope = float(best["slope"])
    phi_ref = float(best["intercept"])
    phase_edge = np.asarray(best["phase_edge"], dtype=float)
    residual = np.asarray(best["residual"], dtype=float)

    tau_s = -slope / (2.0 * np.pi)

    # phi = phi_ref - 2*pi*tau*(f-f_ref)
    #     = (phi_ref + 2*pi*tau*f_ref) - 2*pi*tau*f
    alpha_unwrapped = phi_ref + 2.0 * np.pi * tau_s * f_ref_hz
    alpha_wrapped = float(np.angle(np.exp(1j * alpha_unwrapped)))

    phase_fit = phi_ref + slope * (frequency_hz - f_ref_hz)

    amplitude = float(np.median(np.abs(z_scan[edge_mask])))

    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_tot = float(np.sum((phase_edge - np.mean(phase_edge)) ** 2))
    r_squared = 1.0 - float(best["rss"]) / ss_tot if ss_tot > 0 else np.nan

    return EnvironmentFit(
        amplitude=amplitude,
        tau_s=tau_s,
        slope_rad_per_hz=slope,
        f_ref_hz=f_ref_hz,
        phi_ref_rad=phi_ref,
        alpha_unwrapped_rad=float(alpha_unwrapped),
        alpha_wrapped_rad=alpha_wrapped,
        phase_fit_rad=phase_fit,
        edge_mask=edge_mask,
        edge_phase_rad=phase_edge,
        rmse_rad=rmse,
        r_squared=r_squared,
        high_edge_branch_shift=int(best["k"]),
    )


def align_phase_to_reference(
    raw_phase_rad: np.ndarray,
    reference_phase_rad: np.ndarray,
) -> np.ndarray:
    """
    各点に 2*pi*n を加え、reference に最も近い枝へ移す。
    プロット用であり、fit 自体には使わない。
    """
    return raw_phase_rad + 2.0 * np.pi * np.round(
        (reference_phase_rad - raw_phase_rad) / (2.0 * np.pi)
    )


def correct_complex_data(
    z: np.ndarray,
    factor: complex,
) -> np.ndarray:
    if abs(factor) == 0:
        raise ZeroDivisionError("Calibration factor is zero.")
    return z / factor


# =============================================================================
# CIRCLE FIT
# =============================================================================

def algebraic_circle_fit(z: np.ndarray) -> tuple[complex, float, float]:
    """
    x^2+y^2+D*x+E*y+F=0 の代数的最小二乗円 fit。
    戻り値: center, radius, radial_rms
    """
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


def resonance_window_mask(
    z_scan_corrected: np.ndarray,
    half_width_points: int,
) -> np.ndarray:
    n = z_scan_corrected.size
    resonance_index = int(np.argmin(np.abs(z_scan_corrected)))

    start = max(0, resonance_index - half_width_points)
    stop = min(n, resonance_index + half_width_points + 1)

    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return mask


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def add_frequency_colored_iq(
    ax: plt.Axes,
    z: np.ndarray,
    frequency_hz: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    frequency_ghz = frequency_hz / 1e9

    ax.plot(
        np.real(z),
        np.imag(z),
        linewidth=1.0,
        alpha=0.65,
        color="0.45",
        zorder=1,
    )
    scatter = ax.scatter(
        np.real(z),
        np.imag(z),
        c=frequency_ghz,
        s=28,
        cmap="viridis",
        zorder=2,
    )
    ax.scatter(
        np.real(z[0]),
        np.imag(z[0]),
        s=75,
        marker="o",
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.7,
        label=f"start {frequency_ghz[0]:.6f} GHz",
        zorder=3,
    )
    ax.scatter(
        np.real(z[-1]),
        np.imag(z[-1]),
        s=75,
        marker="s",
        facecolors="none",
        edgecolors="tab:green",
        linewidths=1.7,
        label=f"end {frequency_ghz[-1]:.6f} GHz",
        zorder=3,
    )

    nearest = int(np.argmin(np.abs(frequency_hz - READOUT_FREQUENCY_HZ)))
    ax.scatter(
        np.real(z[nearest]),
        np.imag(z[nearest]),
        marker="*",
        s=130,
        color="tab:red",
        label=f"nearest to {READOUT_FREQUENCY_HZ/1e9:.6f} GHz",
        zorder=4,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(scatter, ax=ax, label="frequency [GHz]")


def choose_event_indices(n_events: int, max_events: int | None) -> np.ndarray:
    if max_events is None or max_events >= n_events:
        return np.arange(n_events)

    return np.unique(
        np.linspace(0, n_events - 1, max_events, dtype=int)
    )


def add_iq_waveform_trajectories(
    ax: plt.Axes,
    z_wave: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    event_indices = choose_event_indices(
        z_wave.shape[0], MAX_EVENTS_TO_PLOT
    )
    sample_indices = np.arange(
        0, z_wave.shape[1], max(1, PLOT_SAMPLE_STRIDE)
    )

    z_plot = z_wave[np.ix_(event_indices, sample_indices)]

    segments = np.stack(
        [np.real(z_plot), np.imag(z_plot)],
        axis=-1,
    )
    event_collection = LineCollection(
        segments,
        colors="0.55",
        linewidths=0.35,
        alpha=0.16,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(event_collection)

    median_trace = np.median(z_wave, axis=0)[sample_indices]
    points = np.column_stack(
        [np.real(median_trace), np.imag(median_trace)]
    )
    median_segments = np.stack([points[:-1], points[1:]], axis=1)

    time_color = np.arange(median_segments.shape[0])
    gradient = LineCollection(
        median_segments,
        cmap="plasma",
        linewidths=2.6,
        zorder=3,
    )
    gradient.set_array(time_color)
    ax.add_collection(gradient)

    # median pedestal の目安として最初の 10% の中央値を表示。
    baseline_stop = max(1, z_wave.shape[1] // 10)
    pedestal = complex(np.median(z_wave[:, :baseline_stop]))
    ax.scatter(
        pedestal.real,
        pedestal.imag,
        marker="X",
        s=85,
        color="black",
        label="median pedestal (first 10%)",
        zorder=4,
    )

    all_x = np.real(z_plot).ravel()
    all_y = np.imag(z_plot).ravel()
    finite = np.isfinite(all_x) & np.isfinite(all_y)
    if np.any(finite):
        ax.set_xlim(np.nanpercentile(all_x[finite], [0.2, 99.8]))
        ax.set_ylim(np.nanpercentile(all_y[finite], [0.2, 99.8]))

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{title}\n"
        f"{event_indices.size}/{z_wave.shape[0]} events, "
        f"sample stride={PLOT_SAMPLE_STRIDE}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(gradient, ax=ax, label="sample order of median track")


def save_figure(
    fig: plt.Figure,
    png_path: Path,
    pdf: PdfPages,
) -> None:
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frequency_hz, scan_ch0, scan_ch1 = load_iq_scan(IQ_SCAN)
    wave_ch0, wave_ch1, waveform_metadata = load_waveform(WAVEFORM_FILE)

    z_scan_raw = scan_ch0 + 1j * Q_SIGN * scan_ch1
    z_wave_raw = wave_ch0 + 1j * Q_SIGN * wave_ch1

    fit = fit_environment_phase(
        frequency_hz=frequency_hz,
        z_scan=z_scan_raw,
        n_edge_points=N_EDGE_POINTS,
    )

    # IQ scan は各周波数ごとの因子で補正。
    scan_amplitude = (
        fit.amplitude if APPLY_AMPLITUDE_NORMALIZATION else 1.0
    )
    scan_factor = scan_amplitude * np.exp(1j * fit.phase_fit_rad)
    z_scan_corrected = z_scan_raw / scan_factor

    # waveform は固定 readout frequency なので単一の複素因子で補正。
    waveform_factor = fit.factor_at(
        READOUT_FREQUENCY_HZ,
        normalize_amplitude=APPLY_AMPLITUDE_NORMALIZATION,
    )
    z_wave_corrected = correct_complex_data(
        z_wave_raw,
        waveform_factor,
    )

    if not (
        frequency_hz.min()
        <= READOUT_FREQUENCY_HZ
        <= frequency_hz.max()
    ):
        print(
            "[warning] READOUT_FREQUENCY_HZ is outside the IQ scan range; "
            "the phase factor is extrapolated."
        )

    circle_mask = resonance_window_mask(
        z_scan_corrected,
        CIRCLE_HALF_WIDTH_POINTS,
    )
    circle_center, circle_radius, circle_rms = algebraic_circle_fit(
        z_scan_corrected[circle_mask]
    )

    pdf_path = OUTPUT_DIR / "iq_environment_calibration.pdf"

    with PdfPages(pdf_path) as pdf:
        # ---------------------------------------------------------------------
        # Figure 1: calibration process
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
        fig.suptitle(
            "Environment calibration from IQ scan\n"
            r"$C_{\rm env}(f)=a\exp[i(\alpha-2\pi\tau f)]$",
            fontsize=16,
        )

        add_frequency_colored_iq(
            axes[0, 0],
            z_scan_raw,
            frequency_hz,
            title="Raw IQ scan",
            xlabel="ch0",
            ylabel=f"{'+' if Q_SIGN > 0 else '-'} ch1 imaginary axis",
        )

        f_ghz = frequency_hz / 1e9
        axes[0, 1].plot(
            f_ghz,
            np.abs(z_scan_raw),
            "o-",
            ms=4,
            label=r"$|z_{\rm scan}|$",
        )
        axes[0, 1].scatter(
            f_ghz[fit.edge_mask],
            np.abs(z_scan_raw[fit.edge_mask]),
            s=55,
            marker="s",
            facecolors="none",
            edgecolors="tab:orange",
            label="points used for a",
            zorder=3,
        )
        axes[0, 1].axhline(
            fit.amplitude,
            color="tab:red",
            linestyle="--",
            label=f"a = {fit.amplitude:.6g}",
        )
        axes[0, 1].set_title("Magnitude normalization")
        axes[0, 1].set_xlabel("frequency [GHz]")
        axes[0, 1].set_ylabel("|IQ|")
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend(fontsize=9)

        phase_for_plot = align_phase_to_reference(
            np.angle(z_scan_raw),
            fit.phase_fit_rad,
        )
        axes[1, 0].plot(
            f_ghz,
            phase_for_plot,
            "o-",
            ms=4,
            label="scan phase (branch aligned)",
        )
        axes[1, 0].plot(
            f_ghz,
            fit.phase_fit_rad,
            linewidth=2.0,
            color="tab:orange",
            label=r"fit: $\phi_{\rm env}=\alpha-2\pi\tau f$",
        )

        edge_indices = np.flatnonzero(fit.edge_mask)
        axes[1, 0].scatter(
            f_ghz[edge_indices],
            fit.edge_phase_rad,
            s=65,
            marker="s",
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.5,
            label="fit points (scan edges)",
            zorder=4,
        )
        axes[1, 0].axvline(
            READOUT_FREQUENCY_HZ / 1e9,
            color="0.25",
            linestyle=":",
            label="waveform RF",
        )
        axes[1, 0].set_title("Least-squares phase fit")
        axes[1, 0].set_xlabel("frequency [GHz]")
        axes[1, 0].set_ylabel("phase [rad]")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend(fontsize=8)

        edge_prediction = (
            fit.phi_ref_rad
            + fit.slope_rad_per_hz
            * (frequency_hz[edge_indices] - fit.f_ref_hz)
        )
        residual = fit.edge_phase_rad - edge_prediction
        axes[1, 1].axhline(0.0, color="0.3", linewidth=1.0)
        axes[1, 1].scatter(
            f_ghz[edge_indices],
            residual,
            s=42,
        )
        axes[1, 1].set_title(
            "Phase-fit residual\n"
            f"RMSE={fit.rmse_rad:.4g} rad, "
            f"$R^2$={fit.r_squared:.6f}"
        )
        axes[1, 1].set_xlabel("frequency [GHz]")
        axes[1, 1].set_ylabel("residual [rad]")
        axes[1, 1].grid(alpha=0.3)

        factor_phase = float(np.angle(waveform_factor))
        text = (
            rf"$\tau={fit.tau_s*1e9:.6f}\ {{\rm ns}}$" "\n"
            rf"$a={fit.amplitude:.8g}$" "\n"
            rf"$\phi_{{env}}(f_{{RF}})={factor_phase:.6f}\ {{\rm rad}}$" "\n"
            rf"$C_{{env}}(f_{{RF}})="
            f"{waveform_factor.real:.6g}"
            f"{waveform_factor.imag:+.6g}i$"
        )
        axes[1, 1].text(
            0.03,
            0.97,
            text,
            transform=axes[1, 1].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(
            fig,
            OUTPUT_DIR / "calibration_process.png",
            pdf,
        )

        # ---------------------------------------------------------------------
        # Figure 2: corrected IQ scan
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3))
        fig.suptitle(
            "IQ scan after cancelling the environment factor",
            fontsize=16,
        )

        add_frequency_colored_iq(
            axes[0],
            z_scan_corrected,
            frequency_hz,
            title="Corrected IQ scan",
            xlabel="corrected I",
            ylabel="corrected Q",
        )

        theta = np.linspace(0, 2.0 * np.pi, 500)
        circle = circle_center + circle_radius * np.exp(1j * theta)
        axes[0].plot(
            circle.real,
            circle.imag,
            color="tab:orange",
            linewidth=2.0,
            label="circle fit",
        )
        axes[0].scatter(
            [circle_center.real],
            [circle_center.imag],
            marker="X",
            s=80,
            color="tab:red",
            label="circle center",
            zorder=5,
        )
        axes[0].legend(fontsize=8, loc="best")

        axes[1].plot(
            f_ghz,
            np.abs(z_scan_corrected),
            "o-",
            ms=4,
            label="corrected magnitude",
        )
        axes[1].axvline(
            READOUT_FREQUENCY_HZ / 1e9,
            color="0.3",
            linestyle=":",
            label="waveform RF",
        )
        axes[1].set_title("Corrected magnitude")
        axes[1].set_xlabel("frequency [GHz]")
        axes[1].set_ylabel("magnitude")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)

        corrected_phase = np.unwrap(np.angle(z_scan_corrected))
        axes[2].plot(
            f_ghz,
            corrected_phase,
            "o-",
            ms=4,
            label="corrected phase",
        )
        axes[2].scatter(
            f_ghz[circle_mask],
            corrected_phase[circle_mask],
            s=45,
            marker="s",
            facecolors="none",
            edgecolors="tab:orange",
            label="circle-fit frequency region",
        )
        axes[2].set_title(
            "Corrected resonator phase\n"
            f"circle radial RMS={circle_rms:.4g}"
        )
        axes[2].set_xlabel("frequency [GHz]")
        axes[2].set_ylabel("phase [rad]")
        axes[2].grid(alpha=0.3)
        axes[2].legend(fontsize=8)

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_figure(
            fig,
            OUTPUT_DIR / "corrected_iqscan.png",
            pdf,
        )

        # ---------------------------------------------------------------------
        # Figure 3: waveform before / after
        # ---------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
        fig.suptitle(
            "Waveform IQ trajectories before and after calibration\n"
            rf"$z_{{corr}}=z_{{raw}}/C_{{env}}"
            f"({READOUT_FREQUENCY_HZ/1e9:.6f} GHz)",
            fontsize=15,
        )

        add_iq_waveform_trajectories(
            axes[0],
            z_wave_raw,
            title="Raw waveform IQ",
            xlabel="ch0",
            ylabel=f"{'+' if Q_SIGN > 0 else '-'} ch1",
        )

        corrected_unit = (
            "normalized I" if APPLY_AMPLITUDE_NORMALIZATION
            else "phase-corrected ch0"
        )
        corrected_qunit = (
            "normalized Q" if APPLY_AMPLITUDE_NORMALIZATION
            else "phase-corrected ch1"
        )
        add_iq_waveform_trajectories(
            axes[1],
            z_wave_corrected,
            title="Corrected waveform IQ",
            xlabel=corrected_unit,
            ylabel=corrected_qunit,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(
            fig,
            OUTPUT_DIR / "waveform_before_after.png",
            pdf,
        )

    print(f"[saved] {pdf_path}")

    # -------------------------------------------------------------------------
    # Save parameters
    # -------------------------------------------------------------------------
    parameter_dict = {
        "iq_scan_file": str(IQ_SCAN),
        "waveform_file": str(WAVEFORM_FILE),
        "readout_frequency_hz": READOUT_FREQUENCY_HZ,
        "q_sign": Q_SIGN,
        "n_edge_points_each_side": int(fit.edge_mask.sum() // 2),
        "apply_amplitude_normalization": APPLY_AMPLITUDE_NORMALIZATION,
        "model": "C_env(f) = a * exp(i * (alpha - 2*pi*tau*f))",
        "amplitude_a": fit.amplitude,
        "tau_s": fit.tau_s,
        "tau_ns": fit.tau_s * 1e9,
        "slope_rad_per_hz": fit.slope_rad_per_hz,
        "f_ref_hz": fit.f_ref_hz,
        "phi_ref_rad": fit.phi_ref_rad,
        "alpha_unwrapped_rad": fit.alpha_unwrapped_rad,
        "alpha_wrapped_rad": fit.alpha_wrapped_rad,
        "phase_fit_rmse_rad": fit.rmse_rad,
        "phase_fit_r_squared": fit.r_squared,
        "high_edge_branch_shift_2pi_multiple": (
            fit.high_edge_branch_shift
        ),
        "waveform_factor_real": waveform_factor.real,
        "waveform_factor_imag": waveform_factor.imag,
        "waveform_factor_abs": abs(waveform_factor),
        "waveform_factor_phase_rad": float(np.angle(waveform_factor)),
        "circle_center_real": circle_center.real,
        "circle_center_imag": circle_center.imag,
        "circle_radius": circle_radius,
        "circle_radial_rms": circle_rms,
    }

    parameter_path = OUTPUT_DIR / "calibration_parameters.json"
    parameter_path.write_text(
        json.dumps(parameter_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {parameter_path}")

    # -------------------------------------------------------------------------
    # Save corrected waveform
    # -------------------------------------------------------------------------
    if SAVE_CORRECTED_NPZ:
        # z = ch0 + i*Q_SIGN*ch1 と定義したので、
        # 保存時は ch1_corrected = Q_SIGN * Im(z_corrected)。
        ch0_corrected = np.real(z_wave_corrected)
        ch1_corrected = Q_SIGN * np.imag(z_wave_corrected)

        corrected_path = OUTPUT_DIR / "corrected_waveform.npz"

        save_data: dict[str, np.ndarray | float | int | str] = {
            "ch0_corrected": ch0_corrected,
            "ch1_corrected": ch1_corrected,
            "readout_frequency_hz": READOUT_FREQUENCY_HZ,
            "environment_amplitude": fit.amplitude,
            "environment_tau_s": fit.tau_s,
            "environment_phase_at_readout_rad": float(
                fit.phase_at(READOUT_FREQUENCY_HZ)
            ),
            "environment_factor_real": waveform_factor.real,
            "environment_factor_imag": waveform_factor.imag,
            "q_sign": Q_SIGN,
        }
        for key, value in waveform_metadata.items():
            save_data[f"original_metadata__{key}"] = value

        np.savez_compressed(corrected_path, **save_data)
        print(f"[saved] {corrected_path}")

    print()
    print("=" * 78)
    print("CALIBRATION RESULT")
    print("=" * 78)
    print(f"a                 = {fit.amplitude:.12g}")
    print(f"tau               = {fit.tau_s:.12g} s")
    print(f"tau               = {fit.tau_s * 1e9:.9f} ns")
    print(f"alpha (wrapped)   = {fit.alpha_wrapped_rad:.9f} rad")
    print(f"phase-fit RMSE    = {fit.rmse_rad:.9g} rad")
    print(f"phase-fit R^2     = {fit.r_squared:.9f}")
    print(
        "C_env(f_readout)  = "
        f"{waveform_factor.real:.12g}"
        f"{waveform_factor.imag:+.12g}j"
    )
    print(
        "waveform correction: "
        "z_corrected = z_raw / C_env(f_readout)"
    )


if __name__ == "__main__":
    main()
