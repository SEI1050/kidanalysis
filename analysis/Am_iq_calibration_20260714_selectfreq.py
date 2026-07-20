from __future__ import annotations

"""
2026-07-14 データ用
------------------
1つの選択周波数だけを解析するバッチコード。

主な仕様
--------
- 最初の USER SETTINGS で解析する周波数 SELECT_FREQUENCY_GHZ を選ぶ。
- waveform の重ね書き・pedestal を引いた振幅波形の重ね書きは、
  pedestal フォルダではなく trig 系フォルダに対して作る。
- IQ track は 2 種類作る。
    (A) pedestal フォルダの track
    (B) trig 系フォルダの track
- trig 系 track は sample を stride=5 で間引き、
  各 sample の中央値 track を時系列グラデーション付きで描く。
- raw データと IQ calibration 後データの両方について、同様の図を作る。
- calibration は、選択した周波数と同じセットの IQ scan だけを使う。
- tau fit の両端点数は 5 点固定。
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import traceback
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_ROOT = Path("/Volumes/NO NAME/data/20260714")
OUTPUT_ROOT = DATA_ROOT / "_analysis_iqcal_selectfreq"

# ここで解析したい周波数を選ぶ。
# 例: 4.463, 5.161, 5.267
SELECT_FREQUENCY_GHZ = 4.463

# z = ch0 + i * Q_SIGN * ch1
Q_SIGN = +1

# tau fit に使う scan 両端の点数（指定どおり 5 点）
N_EDGE_POINTS = 5

# 円 fit に使う共振点近傍の片側点数
CIRCLE_HALF_WIDTH_POINTS = 15

# pedestal は各イベント先頭 10% の平均
PEDESTAL_FRACTION = 0.10

# calibration set と waveform 周波数の許容差
MAX_CALIBRATION_MATCH_HZ = 5.0e6

# 図に描くイベント数。None なら全イベント
MAX_EVENTS_TO_PLOT: int | None = None

# trigger 系 track 専用設定
TRIGGER_TRACK_SAMPLE_STRIDE = 5
TRIGGER_TRACK_MEDIAN_CMAP = "plasma"

# pedestal 系 track / waveform 用サンプル stride
PEDESTAL_TRACK_SAMPLE_STRIDE = 1
WAVEFORM_SAMPLE_STRIDE = 1

PNG_DPI = 200
SAVE_CORRECTED_NPZ = True
SAVE_CALIBRATION_DIAGNOSTICS = True


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
    center_final: complex
    p_final: complex


@dataclass(frozen=True)
class IQScanData:
    path: Path
    frequency_hz: np.ndarray
    ch0: np.ndarray
    ch1: np.ndarray


@dataclass(frozen=True)
class CalibrationSet:
    name: str
    target_frequency_hz: float
    scan_path: Path
    frequency_hz: np.ndarray
    z_scan_raw: np.ndarray
    tau_fit: TauFitResult
    circle_fit: CircleFitResult
    geometry: GeometricCalibration
    z_scan_final: np.ndarray
    final_circle_center: complex
    final_circle_radius: float


@dataclass
class WaveformRecord:
    path: Path
    ch0_key: str
    ch1_key: str
    ch0: np.ndarray
    ch1: np.ndarray
    metadata: dict[str, np.ndarray]
    original_was_1d: bool
    original_was_transposed: bool


@dataclass
class DatasetGroup:
    name: str
    folder_path: Path
    kind: str                # "pedestal" or "trigger"
    frequency_hz: float
    records: list[WaveformRecord]
    ch0: np.ndarray
    ch1: np.ndarray


# =============================================================================
# HELPERS
# =============================================================================

def sanitize_name(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z._+\-]+", "_", text)
    return text.strip("_") or "dataset"


def find_key_case_insensitive(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    mapping = {key.lower(): key for key in keys}
    for candidate in candidates:
        found = mapping.get(candidate.lower())
        if found is not None:
            return found
    return None


def scalar_from_array(value: np.ndarray) -> float | None:
    array = np.asarray(value)
    if array.size != 1 or array.dtype == object:
        return None
    try:
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(result):
        return None
    return result


def convert_frequency_scalar_to_hz(value: float) -> float:
    value = float(value)
    absolute = abs(value)
    if absolute < 100.0:
        return value * 1e9
    if absolute < 1e7:
        return value * 1e6
    return value


def convert_frequency_array_to_hz(frequency: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    finite = frequency[np.isfinite(frequency)]
    if finite.size == 0:
        raise ValueError("frequency array has no finite values")
    typical = float(np.nanmedian(np.abs(finite)))
    if typical < 100.0:
        return frequency * 1e9
    if typical < 1e7:
        return frequency * 1e6
    return frequency


def parse_ghz_from_text(text: str) -> list[float]:
    pattern = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*GHz", re.IGNORECASE)
    return [float(m.group(1)) * 1e9 for m in pattern.finditer(text)]


def choose_event_indices(n_events: int) -> np.ndarray:
    if MAX_EVENTS_TO_PLOT is None or MAX_EVENTS_TO_PLOT >= n_events:
        return np.arange(n_events)
    return np.unique(np.linspace(0, n_events - 1, MAX_EVENTS_TO_PLOT, dtype=int))


def robust_limits(values: np.ndarray, lower: float = 0.2, upper: float = 99.8) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(values, [lower, upper])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi <= lo:
        center = 0.5 * (lo + hi)
        scale = max(abs(center), 1.0) * 1e-6
        return center - scale, center + scale
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def frequency_matches_selected(value_hz: float, selected_hz: float, tol_hz: float = 5.0e6) -> bool:
    return abs(float(value_hz) - float(selected_hz)) <= tol_hz


# =============================================================================
# IQ SCAN DISCOVERY / CALIBRATION
# =============================================================================

def load_iq_scan(path: Path) -> IQScanData:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        if "dd" in npz:
            dd = np.asarray(npz["dd"], dtype=float)
            if dd.ndim != 2 or dd.shape[1] < 3:
                raise ValueError(f"{path.name}: dd shape must be (N,>=3), got {dd.shape}")
            frequency = dd[:, 0]
            ch0 = dd[:, 1]
            ch1 = dd[:, 2]
        else:
            frequency_key = find_key_case_insensitive(
                keys,
                ("frequency", "frequencies", "frequency_hz", "freq", "freq_hz", "f", "f_hz"),
            )
            ch0_key = find_key_case_insensitive(keys, ("ch0", "channel0", "channel_0", "i", "I"))
            ch1_key = find_key_case_insensitive(keys, ("ch1", "channel1", "channel_1", "q", "Q"))
            if frequency_key is None or ch0_key is None or ch1_key is None:
                raise KeyError(f"{path.name}: IQ scan arrays not found. keys={keys}")
            frequency = np.asarray(npz[frequency_key], dtype=float).squeeze()
            ch0 = np.asarray(npz[ch0_key], dtype=float).squeeze()
            ch1 = np.asarray(npz[ch1_key], dtype=float).squeeze()

    frequency_hz = convert_frequency_array_to_hz(frequency)
    ch0 = np.asarray(ch0, dtype=float).reshape(-1)
    ch1 = np.asarray(ch1, dtype=float).reshape(-1)

    finite = np.isfinite(frequency_hz) & np.isfinite(ch0) & np.isfinite(ch1)
    frequency_hz = frequency_hz[finite]
    ch0 = ch0[finite]
    ch1 = ch1[finite]

    order = np.argsort(frequency_hz)
    frequency_hz = frequency_hz[order]
    ch0 = ch0[order]
    ch1 = ch1[order]

    if frequency_hz.size < 2 * N_EDGE_POINTS + 3:
        raise ValueError(f"{path.name}: too few scan points ({frequency_hz.size})")

    return IQScanData(path=path, frequency_hz=frequency_hz, ch0=ch0, ch1=ch1)


def discover_iq_scans(root: Path) -> list[IQScanData]:
    scans: list[IQScanData] = []
    for path in sorted(root.glob("iq_scan*.npz")):
        if "f_reso" in path.name.lower():
            continue
        try:
            scans.append(load_iq_scan(path))
        except Exception as exc:
            print(f"[IQ scan skip] {path.name}: {exc}")
    if not scans:
        raise FileNotFoundError("No usable iq_scan*.npz was found")
    return scans


def extract_frequency_from_resonance_npz(path: Path) -> float | None:
    preferred_keys = (
        "f_reso", "f_resonance", "resonance_frequency", "resonance_frequency_hz",
        "fr", "fr_hz", "frequency", "frequency_hz", "f0", "f0_hz",
    )
    try:
        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.keys())
            preferred = find_key_case_insensitive(keys, preferred_keys)
            if preferred is not None:
                scalar = scalar_from_array(npz[preferred])
                if scalar is not None:
                    return convert_frequency_scalar_to_hz(scalar)
            for key in keys:
                low = key.lower()
                if not any(token in low for token in ("freq", "reso", "fr", "f0")):
                    continue
                scalar = scalar_from_array(npz[key])
                if scalar is None:
                    continue
                converted = convert_frequency_scalar_to_hz(scalar)
                if 1e8 <= abs(converted) <= 1e11:
                    return converted
    except Exception as exc:
        print(f"[resonance warning] {path.name}: {exc}")

    found = parse_ghz_from_text(path.stem)
    return found[0] if found else None


def discover_resonance_targets(root: Path) -> list[float]:
    targets: list[float] = []
    for path in sorted(root.glob("iq_scan_f_reso*.npz")):
        frequency_hz = extract_frequency_from_resonance_npz(path)
        if frequency_hz is not None:
            targets.append(float(frequency_hz))
    targets = sorted(targets)

    unique: list[float] = []
    for freq in targets:
        if not unique or abs(freq - unique[-1]) >= 1e3:
            unique.append(freq)
    return unique


def choose_scan_for_target(scans: list[IQScanData], target_hz: float) -> IQScanData:
    containing = [
        scan for scan in scans
        if scan.frequency_hz.min() <= target_hz <= scan.frequency_hz.max()
    ]
    if containing:
        return min(containing, key=lambda scan: float(np.ptp(scan.frequency_hz)))
    return min(scans, key=lambda scan: float(np.min(np.abs(scan.frequency_hz - target_hz))))


def select_scan_segment(
    scan: IQScanData,
    target_hz: float,
    all_targets_for_scan: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = sorted(all_targets_for_scan)
    index = int(np.argmin(np.abs(np.asarray(targets) - target_hz)))

    lower = -np.inf if index == 0 else 0.5 * (targets[index - 1] + targets[index])
    upper = np.inf if index == len(targets) - 1 else 0.5 * (targets[index] + targets[index + 1])

    mask = (scan.frequency_hz >= lower) & (scan.frequency_hz < upper)
    minimum_points = max(2 * N_EDGE_POINTS + 3, 2 * CIRCLE_HALF_WIDTH_POINTS + 3)
    if np.count_nonzero(mask) < minimum_points:
        count = min(minimum_points, scan.frequency_hz.size)
        nearest = np.argsort(np.abs(scan.frequency_hz - target_hz))[:count]
        nearest = np.sort(nearest)
        mask = np.zeros(scan.frequency_hz.size, dtype=bool)
        mask[nearest] = True

    frequency_hz = scan.frequency_hz[mask]
    ch0 = scan.ch0[mask]
    ch1 = scan.ch1[mask]
    order = np.argsort(frequency_hz)
    return frequency_hz[order], ch0[order], ch1[order]


def make_edge_mask(n_points: int, n_edge_points: int) -> np.ndarray:
    if n_points < 2 * n_edge_points:
        raise ValueError(f"Too few scan points ({n_points})")
    mask = np.zeros(n_points, dtype=bool)
    mask[:n_edge_points] = True
    mask[-n_edge_points:] = True
    return mask


def fit_tau_from_phase(frequency_hz: np.ndarray, z_scan: np.ndarray, n_edge_points: int) -> TauFitResult:
    edge_mask = make_edge_mask(frequency_hz.size, n_edge_points)
    raw_phase = np.angle(z_scan)
    low_phase = np.unwrap(raw_phase[:n_edge_points])
    high_phase_base = np.unwrap(raw_phase[-n_edge_points:])

    f_edge = np.concatenate([frequency_hz[:n_edge_points], frequency_hz[-n_edge_points:]])
    f_ref = float(np.mean(f_edge))
    x_edge = f_edge - f_ref

    best: dict[str, Any] | None = None
    for branch_shift in range(-30, 31):
        phase_edge = np.concatenate([low_phase, high_phase_base + 2.0 * np.pi * branch_shift])
        design = np.column_stack([x_edge, np.ones_like(x_edge)])
        slope, intercept = np.linalg.lstsq(design, phase_edge, rcond=None)[0]
        prediction = slope * x_edge + intercept
        residual = phase_edge - prediction
        rss = float(np.sum(residual**2))
        if best is None or rss < best["rss"]:
            best = {
                "branch_shift": branch_shift,
                "slope": float(slope),
                "intercept": float(intercept),
                "phase_edge": phase_edge,
                "residual": residual,
                "rss": rss,
            }

    if best is None:
        raise RuntimeError("tau fit failed")

    slope = float(best["slope"])
    intercept = float(best["intercept"])
    tau_s = -slope / (2.0 * np.pi)
    phase_fit_all = slope * (frequency_hz - f_ref) + intercept
    residual_edge = np.asarray(best["residual"], dtype=float)
    phase_edge = np.asarray(best["phase_edge"], dtype=float)
    rmse = float(np.sqrt(np.mean(residual_edge**2)))
    ss_tot = float(np.sum((phase_edge - np.mean(phase_edge)) ** 2))
    r_squared = np.nan if ss_tot == 0.0 else 1.0 - float(best["rss"]) / ss_tot
    intercept_b = intercept + 2.0 * np.pi * tau_s * f_ref

    return TauFitResult(
        tau_s=tau_s,
        intercept_b_rad=float(intercept_b),
        slope_rad_per_hz=slope,
        f_ref_hz=f_ref,
        phase_fit_all_rad=phase_fit_all,
        edge_mask=edge_mask,
        edge_phase_used_rad=phase_edge,
        residual_edge_rad=residual_edge,
        rmse_edge_rad=rmse,
        r_squared_edge=float(r_squared),
        high_edge_branch_shift=int(best["branch_shift"]),
    )


def apply_tau_correction(z: np.ndarray, frequency_hz: np.ndarray | float, tau_s: float) -> np.ndarray:
    return z * np.exp(1j * 2.0 * np.pi * tau_s * np.asarray(frequency_hz, dtype=float))


def algebraic_circle_fit(z: np.ndarray) -> tuple[complex, float, float]:
    z = np.asarray(z, dtype=complex).reshape(-1)
    if z.size < 3:
        raise ValueError("circle fit needs at least 3 points")
    x = np.real(z)
    y = np.imag(z)
    design = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    d, e, f0 = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = complex(-d / 2.0, -e / 2.0)
    radius_sq = center.real**2 + center.imag**2 - f0
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        raise ValueError(f"invalid fitted radius^2={radius_sq}")
    radius = float(np.sqrt(radius_sq))
    radial_residual = np.abs(z - center) - radius
    radial_rms = float(np.sqrt(np.mean(radial_residual**2)))
    return center, radius, radial_rms


def fit_circle_near_resonance(frequency_hz: np.ndarray, z_tau: np.ndarray, target_frequency_hz: float) -> CircleFitResult:
    resonance_index = int(np.argmin(np.abs(frequency_hz - target_frequency_hz)))
    max_half_width = max(1, (frequency_hz.size - 3) // 2)
    half_width = min(CIRCLE_HALF_WIDTH_POINTS, max_half_width)

    i0 = max(0, resonance_index - half_width)
    i1 = min(frequency_hz.size, resonance_index + half_width + 1)
    desired = min(frequency_hz.size, 2 * half_width + 1)
    if i1 - i0 < desired:
        if i0 == 0:
            i1 = desired
        elif i1 == frequency_hz.size:
            i0 = frequency_hz.size - desired

    fit_mask = np.zeros(frequency_hz.size, dtype=bool)
    fit_mask[i0:i1] = True
    center, radius, radial_rms = algebraic_circle_fit(z_tau[fit_mask])
    resonance_point = complex(z_tau[resonance_index])
    return CircleFitResult(
        center=center,
        radius=radius,
        radial_rms=radial_rms,
        fit_mask=fit_mask,
        resonance_index=resonance_index,
        resonance_point=resonance_point,
    )


def compute_geometric_calibration(circle_fit: CircleFitResult) -> GeometricCalibration:
    center = circle_fit.center
    resonance_point = circle_fit.resonance_point

    point_p = 2.0 * center - resonance_point
    alpha = float(np.angle(point_p))
    rot_alpha = np.exp(-1j * alpha)

    center_alpha = center * rot_alpha
    p_alpha = point_p * rot_alpha
    amplitude_a = float(np.abs(p_alpha))
    if not np.isfinite(amplitude_a) or amplitude_a <= 0.0:
        raise ValueError(f"invalid amplitude normalization a={amplitude_a}")

    center_normalized = center_alpha / amplitude_a
    p_normalized = p_alpha / amplitude_a

    phi = float(np.angle(1.0 - center_normalized))
    rot_phi = np.exp(-1j * phi)
    center_final = 1.0 + (center_normalized - 1.0) * rot_phi
    p_final = 1.0 + (p_normalized - 1.0) * rot_phi

    return GeometricCalibration(
        point_p=point_p,
        alpha_rad=alpha,
        amplitude_a=amplitude_a,
        phi_rad=phi,
        center_after_tau=center,
        center_final=center_final,
        p_final=p_final,
    )


def apply_full_calibration(z: np.ndarray, frequency_hz: np.ndarray | float, calibration: CalibrationSet) -> np.ndarray:
    z_tau = apply_tau_correction(z, frequency_hz, calibration.tau_fit.tau_s)
    z_alpha = z_tau * np.exp(-1j * calibration.geometry.alpha_rad)
    z_norm = z_alpha / calibration.geometry.amplitude_a
    return 1.0 + (z_norm - 1.0) * np.exp(-1j * calibration.geometry.phi_rad)


def build_selected_calibration_set(scans: list[IQScanData], all_targets_hz: list[float], selected_frequency_hz: float) -> CalibrationSet:
    if all_targets_hz:
        target_hz = min(all_targets_hz, key=lambda f: abs(f - selected_frequency_hz))
        if abs(target_hz - selected_frequency_hz) > MAX_CALIBRATION_MATCH_HZ:
            raise ValueError(
                f"No resonance target near selected frequency {selected_frequency_hz/1e9:.9f} GHz"
            )
    else:
        target_hz = float(selected_frequency_hz)

    scan = choose_scan_for_target(scans, target_hz)
    scan_for_target = {freq: choose_scan_for_target(scans, freq) for freq in all_targets_hz} if all_targets_hz else {target_hz: scan}
    targets_for_same_scan = [freq for freq, assigned in scan_for_target.items() if assigned.path == scan.path]
    if not targets_for_same_scan:
        targets_for_same_scan = [target_hz]

    frequency_hz, ch0, ch1 = select_scan_segment(scan, target_hz, targets_for_same_scan)
    z_raw = ch0 + 1j * Q_SIGN * ch1

    tau_fit = fit_tau_from_phase(frequency_hz, z_raw, N_EDGE_POINTS)
    z_tau = apply_tau_correction(z_raw, frequency_hz, tau_fit.tau_s)
    circle_fit = fit_circle_near_resonance(frequency_hz, z_tau, target_hz)
    geometry = compute_geometric_calibration(circle_fit)
    z_alpha = z_tau * np.exp(-1j * geometry.alpha_rad)
    z_norm = z_alpha / geometry.amplitude_a
    z_final = 1.0 + (z_norm - 1.0) * np.exp(-1j * geometry.phi_rad)
    final_radius = circle_fit.radius / geometry.amplitude_a

    calibration = CalibrationSet(
        name=f"{target_hz/1e9:.6f}GHz",
        target_frequency_hz=float(target_hz),
        scan_path=scan.path,
        frequency_hz=frequency_hz,
        z_scan_raw=z_raw,
        tau_fit=tau_fit,
        circle_fit=circle_fit,
        geometry=geometry,
        z_scan_final=z_final,
        final_circle_center=geometry.center_final,
        final_circle_radius=float(final_radius),
    )

    print(
        f"[selected calibration] target={target_hz/1e9:.9f} GHz, "
        f"tau={tau_fit.tau_s*1e9:.6f} ns, a={geometry.amplitude_a:.6g}"
    )
    return calibration


# =============================================================================
# WAVEFORM LOADING / DATASET DISCOVERY
# =============================================================================

def normalize_waveform_orientation(ch0: np.ndarray, ch1: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    ch0 = np.asarray(ch0, dtype=float)
    ch1 = np.asarray(ch1, dtype=float)
    if ch0.shape != ch1.shape:
        raise ValueError(f"ch0 shape {ch0.shape} != ch1 shape {ch1.shape}")

    original_was_1d = ch0.ndim == 1
    original_was_transposed = False
    if original_was_1d:
        ch0 = ch0[np.newaxis, :]
        ch1 = ch1[np.newaxis, :]
    if ch0.ndim != 2:
        raise ValueError(f"waveform must be 1D or 2D, got {ch0.shape}")
    if ch0.shape[0] > ch0.shape[1] and ch0.shape[1] <= 2000:
        ch0 = ch0.T
        ch1 = ch1.T
        original_was_transposed = True
    return ch0, ch1, original_was_1d, original_was_transposed


def load_waveform_record(path: Path) -> WaveformRecord:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        ch0_key = find_key_case_insensitive(keys, ("ch0", "channel0", "channel_0", "i"))
        ch1_key = find_key_case_insensitive(keys, ("ch1", "channel1", "channel_1", "q"))
        if ch0_key is None or ch1_key is None:
            raise KeyError(f"ch0/ch1 not found. keys={keys}")
        ch0 = np.asarray(npz[ch0_key], dtype=float)
        ch1 = np.asarray(npz[ch1_key], dtype=float)

        metadata: dict[str, np.ndarray] = {}
        for key in keys:
            if key in (ch0_key, ch1_key):
                continue
            try:
                value = np.asarray(npz[key])
            except Exception:
                continue
            if value.dtype != object and value.size <= 100:
                metadata[key] = value

    ch0, ch1, was_1d, was_transposed = normalize_waveform_orientation(ch0, ch1)
    return WaveformRecord(
        path=path,
        ch0_key=ch0_key,
        ch1_key=ch1_key,
        ch0=ch0,
        ch1=ch1,
        metadata=metadata,
        original_was_1d=was_1d,
        original_was_transposed=was_transposed,
    )


def discover_waveform_records(folder: Path) -> list[WaveformRecord]:
    records: list[WaveformRecord] = []
    for path in sorted(folder.rglob("*.npz")):
        low = path.name.lower()
        if low.startswith("iq_scan") or low.endswith("_iqcal.npz"):
            continue
        if OUTPUT_ROOT in path.parents:
            continue
        try:
            records.append(load_waveform_record(path))
        except Exception:
            continue
    return records


def extract_frequency_from_folder_name(folder: Path) -> float | None:
    found = parse_ghz_from_text(folder.name)
    return found[0] if found else None


def classify_folder_kind(folder: Path) -> str:
    return "pedestal" if "pedestal" in folder.name.lower() else "trigger"


def discover_selected_dataset_groups(root: Path, selected_frequency_hz: float) -> list[DatasetGroup]:
    groups: list[DatasetGroup] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder == OUTPUT_ROOT:
            continue
        folder_frequency_hz = extract_frequency_from_folder_name(folder)
        if folder_frequency_hz is None:
            continue
        if not frequency_matches_selected(folder_frequency_hz, selected_frequency_hz, MAX_CALIBRATION_MATCH_HZ):
            continue

        records = discover_waveform_records(folder)
        if not records:
            continue

        kind = classify_folder_kind(folder)
        by_n_samples: dict[int, list[WaveformRecord]] = {}
        for record in records:
            by_n_samples.setdefault(record.ch0.shape[1], []).append(record)

        for n_samples, same_length_records in sorted(by_n_samples.items()):
            ch0 = np.concatenate([record.ch0 for record in same_length_records], axis=0)
            ch1 = np.concatenate([record.ch1 for record in same_length_records], axis=0)
            suffix = "" if len(by_n_samples) == 1 else f"_npts{n_samples}"
            groups.append(
                DatasetGroup(
                    name=f"{folder.name}{suffix}",
                    folder_path=folder,
                    kind=kind,
                    frequency_hz=float(folder_frequency_hz),
                    records=same_length_records,
                    ch0=ch0,
                    ch1=ch1,
                )
            )
            print(
                f"[dataset] {folder.name}{suffix}: kind={kind}, "
                f"events={ch0.shape[0]}, samples={ch0.shape[1]}, "
                f"freq={folder_frequency_hz/1e9:.9f} GHz"
            )

    if not groups:
        raise FileNotFoundError(
            f"No top-level folders matching {selected_frequency_hz/1e9:.9f} GHz were found"
        )
    return groups


# =============================================================================
# PEDESTAL / TIME AXIS
# =============================================================================

def pedestal_stop_index(n_samples: int) -> int:
    return max(1, min(n_samples, int(np.ceil(PEDESTAL_FRACTION * n_samples))))


def subtract_iq_pedestal(ch0: np.ndarray, ch1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stop = pedestal_stop_index(ch0.shape[1])
    pedestal_ch0 = np.mean(ch0[:, :stop], axis=1)
    pedestal_ch1 = np.mean(ch1[:, :stop], axis=1)
    ch0_sub = ch0 - pedestal_ch0[:, np.newaxis]
    ch1_sub = ch1 - pedestal_ch1[:, np.newaxis]
    return ch0_sub, ch1_sub, pedestal_ch0, pedestal_ch1


def pedestal_subtracted_magnitude(ch0: np.ndarray, ch1: np.ndarray) -> np.ndarray:
    ch0_sub, ch1_sub, _, _ = subtract_iq_pedestal(ch0, ch1)
    return np.sqrt(ch0_sub**2 + ch1_sub**2)


def get_sample_rate_hz(records: list[WaveformRecord]) -> float | None:
    candidate_keys = ("sample_rate", "sample_rate_hz", "sampling_rate", "fs", "daq_rate")
    for record in records:
        key = find_key_case_insensitive(record.metadata.keys(), candidate_keys)
        if key is None:
            continue
        scalar = scalar_from_array(record.metadata[key])
        if scalar is not None and scalar > 0.0:
            return float(scalar)
    return None


def make_x_axis(n_samples: int, sample_rate_hz: float | None) -> tuple[np.ndarray, str]:
    if sample_rate_hz is None:
        return np.arange(n_samples, dtype=float), "sample index"
    duration_s = n_samples / sample_rate_hz
    if duration_s < 1e-3:
        return np.arange(n_samples) / sample_rate_hz * 1e6, "time [µs]"
    if duration_s < 1.0:
        return np.arange(n_samples) / sample_rate_hz * 1e3, "time [ms]"
    return np.arange(n_samples) / sample_rate_hz, "time [s]"


# =============================================================================
# PLOTTING
# =============================================================================

def add_event_trajectories(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sample_stride: int, linewidth: float = 0.35, alpha: float = 0.16) -> np.ndarray:
    event_indices = choose_event_indices(x.shape[0])
    sample_indices = np.arange(0, x.shape[1], max(1, sample_stride))
    x_plot = x[np.ix_(event_indices, sample_indices)]
    y_plot = y[np.ix_(event_indices, sample_indices)]
    segments = np.stack([x_plot, y_plot], axis=-1)
    collection = LineCollection(segments, linewidths=linewidth, alpha=alpha, colors="0.55", rasterized=True)
    ax.add_collection(collection)
    return event_indices


def add_gradient_median_track(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sample_stride: int, cmap: str) -> None:
    sample_indices = np.arange(0, x.shape[1], max(1, sample_stride))
    x_med = np.median(x, axis=0)[sample_indices]
    y_med = np.median(y, axis=0)[sample_indices]
    points = np.column_stack([x_med, y_med])
    if points.shape[0] >= 2:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        grad = LineCollection(segments, cmap=cmap, linewidths=2.8, zorder=4)
        grad.set_array(np.arange(segments.shape[0]))
        ax.add_collection(grad)
        plt.colorbar(grad, ax=ax, label="sample order of median track")
    else:
        ax.plot(x_med, y_med, linewidth=2.8)


def compute_median_pedestal_complex(ch0: np.ndarray, ch1: np.ndarray) -> complex:
    stop = pedestal_stop_index(ch0.shape[1])
    return complex(
        np.median(np.mean(ch0[:, :stop], axis=1)),
        np.median(Q_SIGN * np.mean(ch1[:, :stop], axis=1)),
    )


def plot_pedestal_track(ch0: np.ndarray, ch1: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    event_indices = add_event_trajectories(ax, ch0, Q_SIGN * ch1, sample_stride=PEDESTAL_TRACK_SAMPLE_STRIDE)
    sample_indices = np.arange(0, ch0.shape[1], max(1, PEDESTAL_TRACK_SAMPLE_STRIDE))
    ax.plot(np.median(ch0, axis=0)[sample_indices], np.median(Q_SIGN * ch1, axis=0)[sample_indices], linewidth=2.0, label="median track")

    pedestal = compute_median_pedestal_complex(ch0, ch1)
    ax.scatter([pedestal.real], [pedestal.imag], marker="x", s=80, linewidths=2.0, label="median pedestal (first 10%)", zorder=5)

    xlim = robust_limits(ch0[event_indices, ::max(1, PEDESTAL_TRACK_SAMPLE_STRIDE)])
    ylim = robust_limits((Q_SIGN * ch1)[event_indices, ::max(1, PEDESTAL_TRACK_SAMPLE_STRIDE)])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ch0")
    ax.set_ylabel(f"{'+' if Q_SIGN > 0 else '-'}ch1")
    ax.set_title(f"{title}\npedestal track, all events")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_trigger_track(ch0: np.ndarray, ch1: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    event_indices = add_event_trajectories(ax, ch0, Q_SIGN * ch1, sample_stride=TRIGGER_TRACK_SAMPLE_STRIDE)
    add_gradient_median_track(ax, ch0, Q_SIGN * ch1, sample_stride=TRIGGER_TRACK_SAMPLE_STRIDE, cmap=TRIGGER_TRACK_MEDIAN_CMAP)

    pedestal = compute_median_pedestal_complex(ch0, ch1)
    ax.scatter([pedestal.real], [pedestal.imag], marker="x", s=85, linewidths=2.0, color="black", label="median pedestal (first 10%)", zorder=6)

    xlim = robust_limits(ch0[event_indices, ::max(1, TRIGGER_TRACK_SAMPLE_STRIDE)])
    ylim = robust_limits((Q_SIGN * ch1)[event_indices, ::max(1, TRIGGER_TRACK_SAMPLE_STRIDE)])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ch0")
    ax.set_ylabel(f"{'+' if Q_SIGN > 0 else '-'}ch1")
    ax.set_title(f"{title}\ntrigger track, all events, sample stride={TRIGGER_TRACK_SAMPLE_STRIDE}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def add_waveform_lines(ax: plt.Axes, x_axis: np.ndarray, waveforms: np.ndarray, ylabel: str) -> np.ndarray:
    event_indices = choose_event_indices(waveforms.shape[0])
    sample_indices = np.arange(0, waveforms.shape[1], max(1, WAVEFORM_SAMPLE_STRIDE))
    x_plot = x_axis[sample_indices]
    y_plot = waveforms[np.ix_(event_indices, sample_indices)]
    segments = np.stack([np.broadcast_to(x_plot, y_plot.shape), y_plot], axis=-1)
    collection = LineCollection(segments, linewidths=0.35, alpha=0.16, colors="0.55", rasterized=True)
    ax.add_collection(collection)
    median_waveform = np.median(waveforms, axis=0)[sample_indices]
    ax.plot(x_plot, median_waveform, linewidth=2.0, label="median waveform")
    ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))
    ax.set_ylim(*robust_limits(y_plot))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return event_indices


def plot_ch0_ch1_waveforms(ch0: np.ndarray, ch1: np.ndarray, x_axis: np.ndarray, x_label: str, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    event_indices = add_waveform_lines(axes[0], x_axis, ch0, "ch0")
    add_waveform_lines(axes[1], x_axis, ch1, "ch1")
    axes[1].set_xlabel(x_label)
    fig.suptitle(f"{title}\nall waveforms: {event_indices.size}/{ch0.shape[0]} events", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_pedestal_subtracted_magnitude(ch0: np.ndarray, ch1: np.ndarray, x_axis: np.ndarray, x_label: str, title: str) -> plt.Figure:
    magnitude = pedestal_subtracted_magnitude(ch0, ch1)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    event_indices = add_waveform_lines(ax, x_axis, magnitude, r"$\sqrt{(ch0-ped0)^2 + (ch1-ped1)^2}$")
    ax.set_xlabel(x_label)
    ax.set_title(
        f"{title}\npedestal = event-wise mean of first {PEDESTAL_FRACTION:.0%}; "
        f"{event_indices.size}/{ch0.shape[0]} events"
    )
    fig.tight_layout()
    return fig


def save_plot_page(fig: plt.Figure, png_path: Path, pdf: PdfPages) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


def plot_dataset_report(dataset: DatasetGroup, corrected_ch0: np.ndarray | None, corrected_ch1: np.ndarray | None, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = get_sample_rate_hz(dataset.records)
    x_axis, x_label = make_x_axis(dataset.ch0.shape[1], sample_rate_hz)

    report_path = output_dir / "report.pdf"
    with PdfPages(report_path) as pdf:
        if dataset.kind == "pedestal":
            fig = plot_pedestal_track(dataset.ch0, dataset.ch1, title=f"{dataset.name}: RAW")
            save_plot_page(fig, output_dir / "raw_track_pedestal.png", pdf)
            if corrected_ch0 is not None and corrected_ch1 is not None:
                fig = plot_pedestal_track(corrected_ch0, corrected_ch1, title=f"{dataset.name}: CALIBRATED")
                save_plot_page(fig, output_dir / "corrected_track_pedestal.png", pdf)
        else:
            fig = plot_trigger_track(dataset.ch0, dataset.ch1, title=f"{dataset.name}: RAW")
            save_plot_page(fig, output_dir / "raw_track_trigger.png", pdf)
            fig = plot_ch0_ch1_waveforms(dataset.ch0, dataset.ch1, x_axis, x_label, title=f"{dataset.name}: RAW ch0 / ch1")
            save_plot_page(fig, output_dir / "raw_waveforms_ch0_ch1.png", pdf)
            fig = plot_pedestal_subtracted_magnitude(dataset.ch0, dataset.ch1, x_axis, x_label, title=f"{dataset.name}: RAW pedestal-subtracted magnitude")
            save_plot_page(fig, output_dir / "raw_waveforms_pedestal_subtracted_magnitude.png", pdf)

            if corrected_ch0 is not None and corrected_ch1 is not None:
                fig = plot_trigger_track(corrected_ch0, corrected_ch1, title=f"{dataset.name}: CALIBRATED")
                save_plot_page(fig, output_dir / "corrected_track_trigger.png", pdf)
                fig = plot_ch0_ch1_waveforms(corrected_ch0, corrected_ch1, x_axis, x_label, title=f"{dataset.name}: CALIBRATED ch0 / ch1")
                save_plot_page(fig, output_dir / "corrected_waveforms_ch0_ch1.png", pdf)
                fig = plot_pedestal_subtracted_magnitude(corrected_ch0, corrected_ch1, x_axis, x_label, title=f"{dataset.name}: CALIBRATED pedestal-subtracted magnitude")
                save_plot_page(fig, output_dir / "corrected_waveforms_pedestal_subtracted_magnitude.png", pdf)

    print(f"[saved] {report_path}")


# =============================================================================
# CALIBRATION DIAGNOSTICS
# =============================================================================

def align_phase_to_reference(raw_phase: np.ndarray, reference_phase: np.ndarray) -> np.ndarray:
    return raw_phase + 2.0 * np.pi * np.round((reference_phase - raw_phase) / (2.0 * np.pi))


def add_circle(ax: plt.Axes, center: complex, radius: float, label: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    circle = center + radius * np.exp(1j * theta)
    ax.plot(circle.real, circle.imag, linewidth=1.6, label=label)
    ax.scatter([center.real], [center.imag], marker="x", s=70, label="center")


def save_calibration_diagnostics(calibration: CalibrationSet, output_root: Path) -> None:
    output_dir = output_root / sanitize_name(calibration.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_ghz = calibration.frequency_hz / 1e9
    tau_fit = calibration.tau_fit
    circle_fit = calibration.circle_fit
    geometry = calibration.geometry
    z_tau = apply_tau_correction(calibration.z_scan_raw, calibration.frequency_hz, tau_fit.tau_s)

    pdf_path = output_dir / "calibration_diagnostics.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
        axes[0, 0].plot(calibration.z_scan_raw.real, calibration.z_scan_raw.imag, marker="o", markersize=3)
        axes[0, 0].set_title("Raw IQ scan segment")
        axes[0, 0].set_xlabel("ch0")
        axes[0, 0].set_ylabel(f"{'+' if Q_SIGN > 0 else '-'}ch1")
        axes[0, 0].set_aspect("equal", adjustable="box")
        axes[0, 0].grid(alpha=0.3)

        phase_plot = align_phase_to_reference(np.angle(calibration.z_scan_raw), tau_fit.phase_fit_all_rad)
        axes[0, 1].plot(frequency_ghz, phase_plot, "o-", ms=3, label="phase")
        axes[0, 1].plot(frequency_ghz, tau_fit.phase_fit_all_rad, linewidth=2.0, label="edge fit")
        edge_indices = np.flatnonzero(tau_fit.edge_mask)
        axes[0, 1].scatter(
            frequency_ghz[edge_indices],
            tau_fit.edge_phase_used_rad,
            marker="s",
            facecolors="none",
            s=55,
            label=f"used: {N_EDGE_POINTS}+{N_EDGE_POINTS} points",
        )
        axes[0, 1].set_title(f"tau fit: {tau_fit.tau_s*1e9:.6f} ns")
        axes[0, 1].set_xlabel("frequency [GHz]")
        axes[0, 1].set_ylabel("phase [rad]")
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(z_tau.real, z_tau.imag, "o-", ms=3, label="after tau")
        axes[1, 0].scatter(
            z_tau[circle_fit.fit_mask].real,
            z_tau[circle_fit.fit_mask].imag,
            marker="s",
            facecolors="none",
            s=45,
            label="circle-fit points",
        )
        add_circle(axes[1, 0], circle_fit.center, circle_fit.radius, "circle fit")
        axes[1, 0].set_title("Circle fit after tau correction")
        axes[1, 0].set_xlabel("I_tau")
        axes[1, 0].set_ylabel("Q_tau")
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend(fontsize=7)

        axes[1, 1].plot(calibration.z_scan_final.real, calibration.z_scan_final.imag, "o-", ms=3, label="final corrected scan")
        add_circle(axes[1, 1], calibration.final_circle_center, calibration.final_circle_radius, "final circle")
        axes[1, 1].scatter([1.0], [0.0], marker="+", s=100, label="(1,0)")
        axes[1, 1].set_title(
            f"Final calibration\nalpha={geometry.alpha_rad:.6f}, a={geometry.amplitude_a:.6g}, phi={geometry.phi_rad:.6f}"
        )
        axes[1, 1].set_xlabel("I_final")
        axes[1, 1].set_ylabel("Q_final")
        axes[1, 1].set_aspect("equal", adjustable="box")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend(fontsize=7)

        fig.suptitle(f"Calibration set {calibration.target_frequency_hz/1e9:.9f} GHz\nsource scan: {calibration.scan_path.name}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / "calibration_diagnostics.png", dpi=PNG_DPI, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    params = {
        "name": calibration.name,
        "scan_file": str(calibration.scan_path),
        "target_frequency_hz": calibration.target_frequency_hz,
        "q_sign": Q_SIGN,
        "n_edge_points_each_side": N_EDGE_POINTS,
        "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
        "tau_s": calibration.tau_fit.tau_s,
        "alpha_rad": calibration.geometry.alpha_rad,
        "amplitude_a": calibration.geometry.amplitude_a,
        "phi_rad": calibration.geometry.phi_rad,
    }
    write_json(output_dir / "calibration_parameters.json", params)


# =============================================================================
# CORRECTED NPZ OUTPUT
# =============================================================================

def restore_original_orientation(array: np.ndarray, record: WaveformRecord) -> np.ndarray:
    restored = np.asarray(array)
    if record.original_was_transposed:
        restored = restored.T
    if record.original_was_1d:
        restored = restored.reshape(-1)
    return restored


def save_corrected_record(record: WaveformRecord, calibration: CalibrationSet, readout_frequency_hz: float, output_dir: Path) -> Path:
    z_raw = record.ch0 + 1j * Q_SIGN * record.ch1
    z_corrected = apply_full_calibration(z_raw, readout_frequency_hz, calibration)
    corrected_ch0 = np.real(z_corrected)
    corrected_ch1 = np.imag(z_corrected) / Q_SIGN

    payload: dict[str, Any] = {}
    with np.load(record.path, allow_pickle=False) as npz:
        for key in npz.keys():
            if key in (record.ch0_key, record.ch1_key):
                continue
            try:
                value = np.asarray(npz[key])
            except Exception:
                continue
            if value.dtype == object:
                continue
            payload[key] = value

    payload[record.ch0_key] = restore_original_orientation(corrected_ch0, record)
    payload[record.ch1_key] = restore_original_orientation(corrected_ch1, record)
    payload.update(
        {
            "iqcal_source_file": np.array(str(record.path)),
            "iqcal_q_sign": np.array(Q_SIGN),
            "iqcal_waveform_readout_frequency_hz": np.array(readout_frequency_hz),
            "iqcal_target_frequency_hz": np.array(calibration.target_frequency_hz),
            "iqcal_tau_s": np.array(calibration.tau_fit.tau_s),
            "iqcal_alpha_rad": np.array(calibration.geometry.alpha_rad),
            "iqcal_amplitude_a": np.array(calibration.geometry.amplitude_a),
            "iqcal_phi_rad": np.array(calibration.geometry.phi_rad),
        }
    )

    output_file = output_dir / "corrected_npz" / f"{record.path.stem}_iqcal.npz"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **payload)
    print(f"[saved corrected npz] {output_file}")
    return output_file


# =============================================================================
# PROCESSING
# =============================================================================

def process_dataset(dataset: DatasetGroup, calibration: CalibrationSet, output_root: Path) -> dict[str, Any]:
    dataset_output = output_root / sanitize_name(dataset.name)
    dataset_output.mkdir(parents=True, exist_ok=True)

    corrected_ch0: np.ndarray | None = None
    corrected_ch1: np.ndarray | None = None
    corrected_files: list[str] = []

    difference_hz = abs(dataset.frequency_hz - calibration.target_frequency_hz)
    if difference_hz <= MAX_CALIBRATION_MATCH_HZ:
        z_raw = dataset.ch0 + 1j * Q_SIGN * dataset.ch1
        z_corrected = apply_full_calibration(z_raw, dataset.frequency_hz, calibration)
        corrected_ch0 = np.real(z_corrected)
        corrected_ch1 = np.imag(z_corrected) / Q_SIGN
        if SAVE_CORRECTED_NPZ:
            for record in dataset.records:
                corrected_files.append(str(save_corrected_record(record, calibration, dataset.frequency_hz, dataset_output)))
        status = "calibrated"
        message = f"matched to calibration {calibration.target_frequency_hz/1e9:.9f} GHz"
    else:
        status = "raw_only"
        message = f"frequency mismatch too large: {difference_hz/1e6:.3f} MHz"

    plot_dataset_report(dataset, corrected_ch0, corrected_ch1, dataset_output)

    summary = {
        "dataset_name": dataset.name,
        "folder_path": str(dataset.folder_path),
        "kind": dataset.kind,
        "n_files": len(dataset.records),
        "n_events": int(dataset.ch0.shape[0]),
        "n_samples": int(dataset.ch0.shape[1]),
        "frequency_hz": dataset.frequency_hz,
        "status": status,
        "message": message,
        "corrected_files": corrected_files,
    }
    write_json(dataset_output / "dataset_summary.json", summary)
    print(f"[dataset done] {dataset.name}: {status}")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    selected_frequency_hz = float(SELECT_FREQUENCY_GHZ) * 1e9
    selected_output_root = OUTPUT_ROOT / f"{SELECT_FREQUENCY_GHZ:.6f}GHz"
    selected_output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("2026-07-14 selected-frequency IQ waveform analysis and calibration")
    print("=" * 88)
    print(f"DATA_ROOT                : {DATA_ROOT}")
    print(f"OUTPUT_ROOT              : {selected_output_root}")
    print(f"SELECT_FREQUENCY_GHZ     : {SELECT_FREQUENCY_GHZ}")
    print(f"N_EDGE_POINTS            : {N_EDGE_POINTS} (each edge)")
    print(f"TRIGGER_TRACK_STRIDE     : {TRIGGER_TRACK_SAMPLE_STRIDE}")
    print(f"PEDESTAL_FRACTION        : {PEDESTAL_FRACTION:.0%}")
    print()

    datasets = discover_selected_dataset_groups(DATA_ROOT, selected_frequency_hz)
    iq_scans = discover_iq_scans(DATA_ROOT)
    resonance_targets = discover_resonance_targets(DATA_ROOT)
    calibration = build_selected_calibration_set(iq_scans, resonance_targets, selected_frequency_hz)

    if SAVE_CALIBRATION_DIAGNOSTICS:
        save_calibration_diagnostics(calibration, selected_output_root / "calibration")

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for dataset in datasets:
        print()
        print("-" * 88)
        print(f"Processing: {dataset.name}")
        print("-" * 88)
        try:
            summaries.append(process_dataset(dataset, calibration, selected_output_root / "datasets"))
        except Exception as exc:
            traceback.print_exc()
            failures.append({"dataset_name": dataset.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[dataset failed] {dataset.name}: {exc}")

    batch_summary = {
        "data_root": str(DATA_ROOT),
        "output_root": str(selected_output_root),
        "selected_frequency_ghz": SELECT_FREQUENCY_GHZ,
        "selected_frequency_hz": selected_frequency_hz,
        "settings": {
            "q_sign": Q_SIGN,
            "n_edge_points_each_side": N_EDGE_POINTS,
            "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
            "pedestal_fraction": PEDESTAL_FRACTION,
            "max_calibration_match_hz": MAX_CALIBRATION_MATCH_HZ,
            "trigger_track_sample_stride": TRIGGER_TRACK_SAMPLE_STRIDE,
            "pedestal_track_sample_stride": PEDESTAL_TRACK_SAMPLE_STRIDE,
            "waveform_sample_stride": WAVEFORM_SAMPLE_STRIDE,
        },
        "calibration": {
            "target_frequency_hz": calibration.target_frequency_hz,
            "scan_file": str(calibration.scan_path),
            "tau_s": calibration.tau_fit.tau_s,
            "alpha_rad": calibration.geometry.alpha_rad,
            "amplitude_a": calibration.geometry.amplitude_a,
            "phi_rad": calibration.geometry.phi_rad,
        },
        "datasets": summaries,
        "failures": failures,
    }
    write_json(selected_output_root / "batch_summary.json", batch_summary)

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"successful datasets: {len(summaries)}")
    print(f"failed datasets    : {len(failures)}")
    print(f"results            : {selected_output_root}")


if __name__ == "__main__":
    main()
