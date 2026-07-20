from __future__ import annotations

"""
2026-07-14 の waveform データを一括解析し、IQ scan から求めた幾何学的
calibration を、対応する読み出し周波数のデータだけに適用する。

各データセットについて作る図
------------------------------
Raw:
  1. ch0 を横軸、ch1 を縦軸にした全イベントの IQ track 重ね書き
  2. ch0 / ch1 の全 waveform 重ね書き
  3. 各イベントの冒頭 10 % から ch0, ch1 pedestal を求め、
     sqrt((ch0-ped0)^2 + (ch1-ped1)^2) を全イベント重ね書き

Corrected:
  Raw と同じ 3 種類の図

さらに、元の npz を変更せず、補正後 ch0/ch1 を入れた *_iqcal.npz を保存する。

IQ calibration
--------------
  z = ch0 + i * Q_SIGN * ch1

  1. IQ scan の両端 N_EDGE_POINTS 点ずつで
         phase = b - 2*pi*tau*f
     を最小二乗 fit
  2. tau 補正
  3. 対象共振周波数近傍を円 fit
  4. P = 2c - z_res から alpha と振幅 a を決定
  5. (1, 0) を固定した回転 phi で円中心を実軸上へ移す

重要:
  - N_EDGE_POINTS = 5
  - waveform と calibration set は読み出し周波数で最近傍照合し、
    MAX_CALIBRATION_MATCH_HZ を超えて離れる組み合わせには補正を掛けない。
  - 5.161 GHz のようにフォルダ名が丸められていても、
    iq_scan_f_reso_5.1617GHz.npz のような同一共振器セットに照合できる。
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

# 外付けディスク内に解析結果をまとめる。
# 別の場所へ保存したい場合はこの 1 行だけ変更する。
OUTPUT_ROOT = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260714/"
    "check_track_wave_normal_calibration"
)

# z = ch0 + i * Q_SIGN * ch1
Q_SIGN = +1

# 指定どおり、tau fit に使う IQ scan 両端の点数は 5 点ずつ。
N_EDGE_POINTS = 5

# 円 fit に使う、共振点の片側の点数。
CIRCLE_HALF_WIDTH_POINTS = 3

# waveform の先頭何割を pedestal とするか。
PEDESTAL_FRACTION = 0.10

# 「同じ周波数セット」と判断する最大周波数差。
# 例: 5.161 GHz と 5.1617 GHz の差 0.7 MHz は許容される。
MAX_CALIBRATION_MATCH_HZ = 5.0e6

# 描画対象イベント。None なら全イベントを描画する。
MAX_EVENTS_TO_PLOT: int | None = None

# 描画時の sample 間引き。1 なら全 sample を描画する。
PLOT_SAMPLE_STRIDE = 1

# PNG 保存解像度。
PNG_DPI = 200

# 補正済み npz を保存する。
SAVE_CORRECTED_NPZ = True

# calibration diagnostics を保存する。
SAVE_CALIBRATION_DIAGNOSTICS = True

# フォルダ名にも npz metadata にも読み出し周波数が無い場合の手動指定。
# 例:
# READOUT_FREQUENCY_OVERRIDES_GHZ = {
#     "data_0714_173939": 4.463,
#     "data_0714_174109": 5.161,
# }
READOUT_FREQUENCY_OVERRIDES_GHZ: dict[str, float] = {}

# True の場合、周波数を検出できない data_* フォルダ等は raw 図だけ作り、
# calibration はスキップする。誤った周波数セットとの変換は行わない。
ALLOW_RAW_ONLY_WHEN_FREQUENCY_UNKNOWN = True


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
    ch0: np.ndarray          # shape = (events, samples)
    ch1: np.ndarray          # shape = (events, samples)
    metadata: dict[str, np.ndarray]
    original_was_1d: bool
    original_was_transposed: bool


@dataclass
class DatasetGroup:
    name: str
    base_dir: Path
    records: list[WaveformRecord]
    ch0: np.ndarray
    ch1: np.ndarray
    readout_frequency_hz: float | None
    frequency_source: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def sanitize_name(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z._+\-]+", "_", text)
    return text.strip("_") or "dataset"


def find_key_case_insensitive(
    keys: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    lower_to_original = {key.lower(): key for key in keys}
    for candidate in candidates:
        found = lower_to_original.get(candidate.lower())
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
        return value * 1e9       # GHz
    if absolute < 1e7:
        return value * 1e6       # MHz
    return value                 # Hz


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
    return [float(match.group(1)) * 1e9 for match in pattern.finditer(text)]


def choose_event_indices(n_events: int) -> np.ndarray:
    if MAX_EVENTS_TO_PLOT is None or MAX_EVENTS_TO_PLOT >= n_events:
        return np.arange(n_events)
    return np.unique(
        np.linspace(0, n_events - 1, MAX_EVENTS_TO_PLOT, dtype=int)
    )


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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# IQ SCAN DISCOVERY / LOADING
# =============================================================================

def load_iq_scan(path: Path) -> IQScanData:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())

        if "dd" in npz:
            dd = np.asarray(npz["dd"], dtype=float)
            if dd.ndim != 2 or dd.shape[1] < 3:
                raise ValueError(f"{path.name}: dd shape must be (N, >=3), got {dd.shape}")
            frequency = dd[:, 0]
            ch0 = dd[:, 1]
            ch1 = dd[:, 2]
        else:
            frequency_key = find_key_case_insensitive(
                keys,
                (
                    "frequency", "frequencies", "frequency_hz", "freq", "freq_hz",
                    "f", "f_hz", "rf_frequency", "readout_frequency",
                ),
            )
            ch0_key = find_key_case_insensitive(
                keys,
                ("ch0", "channel0", "channel_0", "i", "I"),
            )
            ch1_key = find_key_case_insensitive(
                keys,
                ("ch1", "channel1", "channel_1", "q", "Q"),
            )
            if frequency_key is None or ch0_key is None or ch1_key is None:
                raise KeyError(
                    f"{path.name}: IQ scan arrays not found. keys={keys}"
                )
            frequency = np.asarray(npz[frequency_key], dtype=float).squeeze()
            ch0 = np.asarray(npz[ch0_key], dtype=float).squeeze()
            ch1 = np.asarray(npz[ch1_key], dtype=float).squeeze()

    frequency_hz = convert_frequency_array_to_hz(frequency)
    ch0 = np.asarray(ch0, dtype=float).reshape(-1)
    ch1 = np.asarray(ch1, dtype=float).reshape(-1)

    if not (frequency_hz.size == ch0.size == ch1.size):
        raise ValueError(
            f"{path.name}: size mismatch: f={frequency_hz.size}, "
            f"ch0={ch0.size}, ch1={ch1.size}"
        )
    if frequency_hz.size < 2 * N_EDGE_POINTS + 3:
        raise ValueError(
            f"{path.name}: too few IQ scan points ({frequency_hz.size})"
        )

    finite = np.isfinite(frequency_hz) & np.isfinite(ch0) & np.isfinite(ch1)
    frequency_hz = frequency_hz[finite]
    ch0 = ch0[finite]
    ch1 = ch1[finite]

    order = np.argsort(frequency_hz)
    return IQScanData(
        path=path,
        frequency_hz=frequency_hz[order],
        ch0=ch0[order],
        ch1=ch1[order],
    )


def discover_iq_scans(root: Path) -> list[IQScanData]:
    candidates = sorted(root.glob("iq_scan*.npz"))
    scans: list[IQScanData] = []

    for path in candidates:
        if "f_reso" in path.name.lower():
            continue
        try:
            scan = load_iq_scan(path)
        except Exception as exc:
            print(f"[IQ scan skip] {path.name}: {exc}")
            continue
        scans.append(scan)
        print(
            f"[IQ scan] {path.name}: {scan.frequency_hz.size} points, "
            f"{scan.frequency_hz.min()/1e9:.9f}--"
            f"{scan.frequency_hz.max()/1e9:.9f} GHz"
        )

    if not scans:
        raise FileNotFoundError(
            f"No usable IQ scan was found under {root}. "
            "Expected iq_scan*.npz containing dd[:,0:3] or frequency/ch0/ch1 arrays."
        )
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

            # key 名に frequency / reso / fr / f0 が入る scalar を探す。
            for key in keys:
                low = key.lower()
                if not any(token in low for token in ("freq", "reso", "fr", "f0")):
                    continue
                scalar = scalar_from_array(npz[key])
                if scalar is not None:
                    converted = convert_frequency_scalar_to_hz(scalar)
                    if 1e8 <= abs(converted) <= 1e11:
                        return converted
    except Exception as exc:
        print(f"[resonance file warning] {path.name}: {exc}")

    frequencies = parse_ghz_from_text(path.stem)
    return frequencies[0] if frequencies else None


def discover_resonance_targets(root: Path) -> list[float]:
    targets: list[float] = []
    for path in sorted(root.glob("iq_scan_f_reso*.npz")):
        frequency_hz = extract_frequency_from_resonance_npz(path)
        if frequency_hz is None:
            print(f"[resonance skip] frequency not found: {path.name}")
            continue
        targets.append(float(frequency_hz))
        print(f"[resonance target] {path.name} -> {frequency_hz/1e9:.9f} GHz")

    # 重複除去。1 kHz 未満の差は同じものとみなす。
    unique: list[float] = []
    for frequency_hz in sorted(targets):
        if not unique or abs(frequency_hz - unique[-1]) >= 1e3:
            unique.append(frequency_hz)
    return unique


def choose_scan_for_target(scans: list[IQScanData], target_hz: float) -> IQScanData:
    containing = [
        scan for scan in scans
        if scan.frequency_hz.min() <= target_hz <= scan.frequency_hz.max()
    ]
    if containing:
        # 複数の scan が対象を含む場合、最も狭い scan を優先する。
        return min(
            containing,
            key=lambda scan: float(np.ptp(scan.frequency_hz)),
        )

    # 範囲外でも最近傍点が最も近い scan を選び、後段で距離を確認する。
    return min(
        scans,
        key=lambda scan: float(np.min(np.abs(scan.frequency_hz - target_hz))),
    )


def select_scan_segment(
    scan: IQScanData,
    target_hz: float,
    all_targets_for_scan: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1 つの広い IQ scan に複数共振器が含まれる場合、隣接する target 周波数の
    中点で scan を分割する。これにより、tau fit の両端 5 点が同じ共振器セット
    の両端から選ばれる。
    """
    targets = sorted(all_targets_for_scan)
    index = int(np.argmin(np.abs(np.asarray(targets) - target_hz)))

    lower = -np.inf if index == 0 else 0.5 * (targets[index - 1] + targets[index])
    upper = np.inf if index == len(targets) - 1 else 0.5 * (targets[index] + targets[index + 1])

    mask = (scan.frequency_hz >= lower) & (scan.frequency_hz < upper)
    minimum_points = max(2 * N_EDGE_POINTS + 3, 2 * CIRCLE_HALF_WIDTH_POINTS + 3)

    if np.count_nonzero(mask) < minimum_points:
        # 分割結果が短すぎるときは target に近い点を minimum_points 個まで使う。
        count = min(minimum_points, scan.frequency_hz.size)
        nearest_indices = np.argsort(np.abs(scan.frequency_hz - target_hz))[:count]
        nearest_indices = np.sort(nearest_indices)
        mask = np.zeros(scan.frequency_hz.size, dtype=bool)
        mask[nearest_indices] = True

    frequency_hz = scan.frequency_hz[mask]
    ch0 = scan.ch0[mask]
    ch1 = scan.ch1[mask]

    if frequency_hz.size < 2 * N_EDGE_POINTS + 3:
        raise ValueError(
            f"Too few points for target {target_hz/1e9:.9f} GHz: "
            f"{frequency_hz.size}"
        )

    order = np.argsort(frequency_hz)
    return frequency_hz[order], ch0[order], ch1[order]


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
    edge_mask = make_edge_mask(frequency_hz.size, n_edge_points)

    raw_phase = np.angle(z_scan)
    low_phase = np.unwrap(raw_phase[:n_edge_points])
    high_phase_base = np.unwrap(raw_phase[-n_edge_points:])

    f_edge = np.concatenate(
        [frequency_hz[:n_edge_points], frequency_hz[-n_edge_points:]]
    )
    f_ref = float(np.mean(f_edge))
    x_edge = f_edge - f_ref

    best: dict[str, Any] | None = None
    # 左右端の unwrap branch を合わせる。scan span が大きい場合を考え、広めに探索。
    for branch_shift in range(-30, 31):
        phase_edge = np.concatenate(
            [low_phase, high_phase_base + 2.0 * np.pi * branch_shift]
        )
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

    # phase = slope*(f-f_ref)+intercept = b - 2*pi*tau*f
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


def apply_tau_correction(
    z: np.ndarray,
    frequency_hz: np.ndarray | float,
    tau_s: float,
) -> np.ndarray:
    return z * np.exp(
        1j * 2.0 * np.pi * tau_s * np.asarray(frequency_hz, dtype=float)
    )


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


def fit_circle_near_resonance(
    frequency_hz: np.ndarray,
    z_tau: np.ndarray,
    target_frequency_hz: float,
) -> CircleFitResult:
    resonance_index = int(np.argmin(np.abs(frequency_hz - target_frequency_hz)))
    max_half_width = max(1, (frequency_hz.size - 3) // 2)
    half_width = min(CIRCLE_HALF_WIDTH_POINTS, max_half_width)

    i0 = max(0, resonance_index - half_width)
    i1 = min(frequency_hz.size, resonance_index + half_width + 1)

    # target が端に寄っていても、可能なら点数を確保する。
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


def compute_geometric_calibration(
    circle_fit: CircleFitResult,
) -> GeometricCalibration:
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


def apply_full_calibration(
    z: np.ndarray,
    frequency_hz: np.ndarray | float,
    calibration: CalibrationSet,
) -> np.ndarray:
    z_tau = apply_tau_correction(z, frequency_hz, calibration.tau_fit.tau_s)
    z_alpha = z_tau * np.exp(-1j * calibration.geometry.alpha_rad)
    z_norm = z_alpha / calibration.geometry.amplitude_a
    return 1.0 + (z_norm - 1.0) * np.exp(-1j * calibration.geometry.phi_rad)


def build_calibration_sets(
    scans: list[IQScanData],
    target_frequencies_hz: list[float],
) -> list[CalibrationSet]:
    if not target_frequencies_hz:
        raise ValueError(
            "No resonance target was found. Expected iq_scan_f_reso_*.npz files."
        )

    scan_for_target: dict[float, IQScanData] = {
        target: choose_scan_for_target(scans, target)
        for target in target_frequencies_hz
    }

    calibration_sets: list[CalibrationSet] = []

    for target_hz in sorted(target_frequencies_hz):
        scan = scan_for_target[target_hz]
        targets_for_same_scan = [
            frequency_hz
            for frequency_hz, assigned_scan in scan_for_target.items()
            if assigned_scan.path == scan.path
        ]

        frequency_hz, ch0, ch1 = select_scan_segment(
            scan=scan,
            target_hz=target_hz,
            all_targets_for_scan=targets_for_same_scan,
        )
        z_raw = ch0 + 1j * Q_SIGN * ch1

        tau_fit = fit_tau_from_phase(
            frequency_hz=frequency_hz,
            z_scan=z_raw,
            n_edge_points=N_EDGE_POINTS,
        )
        z_tau = apply_tau_correction(z_raw, frequency_hz, tau_fit.tau_s)
        circle_fit = fit_circle_near_resonance(
            frequency_hz=frequency_hz,
            z_tau=z_tau,
            target_frequency_hz=target_hz,
        )
        geometry = compute_geometric_calibration(circle_fit)

        # scan 全点に同じ補正を掛ける。
        z_alpha = z_tau * np.exp(-1j * geometry.alpha_rad)
        z_norm = z_alpha / geometry.amplitude_a
        z_final = 1.0 + (z_norm - 1.0) * np.exp(-1j * geometry.phi_rad)

        final_radius = circle_fit.radius / geometry.amplitude_a
        name = f"{target_hz/1e9:.6f}GHz"
        calibration = CalibrationSet(
            name=name,
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
        calibration_sets.append(calibration)

        selected_hz = frequency_hz[circle_fit.resonance_index]
        print(
            f"[calibration] target={target_hz/1e9:.9f} GHz, "
            f"scan point={selected_hz/1e9:.9f} GHz, "
            f"tau={tau_fit.tau_s*1e9:.6f} ns, "
            f"a={geometry.amplitude_a:.6g}"
        )

    return calibration_sets


# =============================================================================
# WAVEFORM LOADING / FREQUENCY DETECTION
# =============================================================================

def normalize_waveform_orientation(
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
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

    # 通常は (events, samples)。第 0 軸が明らかに samples の場合だけ転置。
    if ch0.shape[0] > ch0.shape[1] and ch0.shape[1] <= 2000:
        ch0 = ch0.T
        ch1 = ch1.T
        original_was_transposed = True

    return ch0, ch1, original_was_1d, original_was_transposed


def load_waveform_record(path: Path) -> WaveformRecord:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        ch0_key = find_key_case_insensitive(
            keys,
            ("ch0", "channel0", "channel_0", "i"),
        )
        ch1_key = find_key_case_insensitive(
            keys,
            ("ch1", "channel1", "channel_1", "q"),
        )
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
        lower_name = path.name.lower()
        if lower_name.startswith("iq_scan"):
            continue
        if lower_name.endswith("_iqcal.npz"):
            continue
        if OUTPUT_ROOT in path.parents:
            continue

        try:
            record = load_waveform_record(path)
        except Exception:
            continue
        records.append(record)

    return records


def detect_frequency_from_metadata(records: list[WaveformRecord]) -> tuple[float | None, str]:
    preferred_exact = (
        "readout_frequency_hz", "readout_frequency", "readout_freq_hz",
        "readout_freq", "rf_frequency_hz", "rf_frequency", "rf_freq_hz",
        "rf_freq", "frequency_hz", "frequency", "tone_frequency_hz",
        "tone_frequency", "tone_freq_hz", "tone_freq",
    )

    for record in records:
        keys = list(record.metadata.keys())
        preferred_key = find_key_case_insensitive(keys, preferred_exact)
        if preferred_key is not None:
            scalar = scalar_from_array(record.metadata[preferred_key])
            if scalar is not None:
                converted = convert_frequency_scalar_to_hz(scalar)
                if 1e8 <= abs(converted) <= 1e11:
                    return float(converted), f"metadata:{record.path.name}:{preferred_key}"

    # 一般の key 探索。ただし sample rate、laser、trigger などは除外。
    excluded_tokens = (
        "sample", "daq", "laser", "trigger", "trig", "rate", "clock",
        "modulation", "pulse",
    )
    for record in records:
        for key, value in record.metadata.items():
            low = key.lower()
            if not any(token in low for token in ("freq", "tone", "readout", "rf")):
                continue
            if any(token in low for token in excluded_tokens):
                continue
            scalar = scalar_from_array(value)
            if scalar is None:
                continue
            converted = convert_frequency_scalar_to_hz(scalar)
            if 1e8 <= abs(converted) <= 1e11:
                return float(converted), f"metadata:{record.path.name}:{key}"

    return None, "not found in metadata"


def detect_frequency_from_text_files(folder: Path) -> tuple[float | None, str]:
    for extension in ("*.txt", "*.log", "*.json", "*.md", "*.csv"):
        for path in sorted(folder.glob(extension)):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            frequencies = parse_ghz_from_text(text)
            if frequencies:
                return frequencies[0], f"text:{path.name}"
    return None, "not found in text files"


def detect_readout_frequency(
    dataset_dir: Path,
    records: list[WaveformRecord],
) -> tuple[float | None, str]:
    # 0. ユーザーが明示した手動対応を最優先する。
    if dataset_dir.name in READOUT_FREQUENCY_OVERRIDES_GHZ:
        frequency_ghz = READOUT_FREQUENCY_OVERRIDES_GHZ[dataset_dir.name]
        return float(frequency_ghz) * 1e9, f"manual override:{dataset_dir.name}"

    # 1. npz metadata を優先する。
    frequency_hz, source = detect_frequency_from_metadata(records)
    if frequency_hz is not None:
        return frequency_hz, source

    # 2. データセットフォルダ名、親フォルダ名、ファイル名から GHz を読む。
    candidate_texts = [dataset_dir.name]
    candidate_texts.extend(record.path.name for record in records)
    for text in candidate_texts:
        frequencies = parse_ghz_from_text(text)
        if frequencies:
            return frequencies[0], f"name:{text}"

    # 3. 同じフォルダのログ類を確認する。
    frequency_hz, source = detect_frequency_from_text_files(dataset_dir)
    if frequency_hz is not None:
        return frequency_hz, source

    return None, "frequency not detected"


def discover_dataset_groups(root: Path) -> list[DatasetGroup]:
    groups: list[DatasetGroup] = []

    # スクリーンショットにある各トップレベルフォルダを 1 データセット単位とする。
    candidate_dirs = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and path != OUTPUT_ROOT
    ]

    for dataset_dir in candidate_dirs:
        records = discover_waveform_records(dataset_dir)
        if not records:
            continue

        # sample 数が異なるファイルは同一図に重ねられないため分ける。
        by_n_samples: dict[int, list[WaveformRecord]] = {}
        for record in records:
            by_n_samples.setdefault(record.ch0.shape[1], []).append(record)

        frequency_hz, source = detect_readout_frequency(dataset_dir, records)

        for n_samples, same_length_records in sorted(by_n_samples.items()):
            ch0 = np.concatenate([record.ch0 for record in same_length_records], axis=0)
            ch1 = np.concatenate([record.ch1 for record in same_length_records], axis=0)
            suffix = "" if len(by_n_samples) == 1 else f"_npts{n_samples}"
            name = f"{dataset_dir.name}{suffix}"

            groups.append(
                DatasetGroup(
                    name=name,
                    base_dir=dataset_dir,
                    records=same_length_records,
                    ch0=ch0,
                    ch1=ch1,
                    readout_frequency_hz=frequency_hz,
                    frequency_source=source,
                )
            )

            frequency_text = (
                "unknown"
                if frequency_hz is None
                else f"{frequency_hz/1e9:.9f} GHz"
            )
            print(
                f"[dataset] {name}: files={len(same_length_records)}, "
                f"events={ch0.shape[0]}, samples={ch0.shape[1]}, "
                f"readout={frequency_text} ({source})"
            )

    # root 直下に waveform npz がある場合も処理する。
    root_records: list[WaveformRecord] = []
    for path in sorted(root.glob("*.npz")):
        if path.name.lower().startswith("iq_scan"):
            continue
        try:
            root_records.append(load_waveform_record(path))
        except Exception:
            continue

    for record in root_records:
        frequency_hz, source = detect_readout_frequency(root, [record])
        groups.append(
            DatasetGroup(
                name=record.path.stem,
                base_dir=root,
                records=[record],
                ch0=record.ch0,
                ch1=record.ch1,
                readout_frequency_hz=frequency_hz,
                frequency_source=source,
            )
        )

    if not groups:
        raise FileNotFoundError(f"No waveform npz containing ch0/ch1 was found under {root}")
    return groups


def match_calibration_set(
    readout_frequency_hz: float,
    calibration_sets: list[CalibrationSet],
) -> tuple[CalibrationSet | None, float]:
    calibration = min(
        calibration_sets,
        key=lambda item: abs(item.target_frequency_hz - readout_frequency_hz),
    )
    difference_hz = abs(calibration.target_frequency_hz - readout_frequency_hz)
    if difference_hz > MAX_CALIBRATION_MATCH_HZ:
        return None, float(difference_hz)
    return calibration, float(difference_hz)


# =============================================================================
# PEDESTAL / TIME AXIS
# =============================================================================

def pedestal_stop_index(n_samples: int) -> int:
    return max(1, min(n_samples, int(np.ceil(PEDESTAL_FRACTION * n_samples))))


def subtract_iq_pedestal(
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def add_event_trajectories(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    linewidth: float = 0.35,
    alpha: float = 0.16,
) -> np.ndarray:
    event_indices = choose_event_indices(x.shape[0])
    sample_indices = np.arange(0, x.shape[1], max(1, PLOT_SAMPLE_STRIDE))

    x_plot = x[np.ix_(event_indices, sample_indices)]
    y_plot = y[np.ix_(event_indices, sample_indices)]
    segments = np.stack([x_plot, y_plot], axis=-1)

    collection = LineCollection(
        segments,
        linewidths=linewidth,
        alpha=alpha,
        rasterized=True,
    )
    ax.add_collection(collection)
    return event_indices


def plot_iq_tracks(
    ch0: np.ndarray,
    ch1: np.ndarray,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    event_indices = add_event_trajectories(ax, ch0, Q_SIGN * ch1)

    sample_indices = np.arange(0, ch0.shape[1], max(1, PLOT_SAMPLE_STRIDE))
    median_ch0 = np.median(ch0, axis=0)[sample_indices]
    median_ch1 = Q_SIGN * np.median(ch1, axis=0)[sample_indices]
    ax.plot(median_ch0, median_ch1, linewidth=2.0, label="median track")

    stop = pedestal_stop_index(ch0.shape[1])
    pedestal = np.median(
        np.mean(ch0[:, :stop], axis=1)
        + 1j * Q_SIGN * np.mean(ch1[:, :stop], axis=1)
    )
    ax.scatter(
        [pedestal.real],
        [pedestal.imag],
        marker="x",
        s=80,
        linewidths=2.0,
        label="median pedestal (first 10%)",
        zorder=5,
    )

    xlim = robust_limits(ch0[event_indices, ::max(1, PLOT_SAMPLE_STRIDE)])
    ylim = robust_limits(Q_SIGN * ch1[event_indices, ::max(1, PLOT_SAMPLE_STRIDE)])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ch0")
    ax.set_ylabel(f"{'+' if Q_SIGN > 0 else '-'}ch1")
    ax.set_title(
        f"{title}\nall IQ tracks: {event_indices.size}/{ch0.shape[0]} events, "
        f"sample stride={PLOT_SAMPLE_STRIDE}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def add_waveform_lines(
    ax: plt.Axes,
    x_axis: np.ndarray,
    waveforms: np.ndarray,
    ylabel: str,
) -> np.ndarray:
    event_indices = choose_event_indices(waveforms.shape[0])
    sample_indices = np.arange(0, waveforms.shape[1], max(1, PLOT_SAMPLE_STRIDE))
    x_plot = x_axis[sample_indices]
    y_plot = waveforms[np.ix_(event_indices, sample_indices)]

    segments = np.stack(
        [
            np.broadcast_to(x_plot, y_plot.shape),
            y_plot,
        ],
        axis=-1,
    )
    collection = LineCollection(
        segments,
        linewidths=0.35,
        alpha=0.16,
        rasterized=True,
    )
    ax.add_collection(collection)

    median_waveform = np.median(waveforms, axis=0)[sample_indices]
    ax.plot(x_plot, median_waveform, linewidth=2.0, label="median waveform")

    ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))
    ax.set_ylim(*robust_limits(y_plot))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return event_indices


def plot_ch0_ch1_waveforms(
    ch0: np.ndarray,
    ch1: np.ndarray,
    x_axis: np.ndarray,
    x_label: str,
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    event_indices = add_waveform_lines(axes[0], x_axis, ch0, "ch0")
    add_waveform_lines(axes[1], x_axis, ch1, "ch1")
    axes[1].set_xlabel(x_label)
    fig.suptitle(
        f"{title}\nall waveforms: {event_indices.size}/{ch0.shape[0]} events, "
        f"sample stride={PLOT_SAMPLE_STRIDE}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_pedestal_subtracted_magnitude(
    ch0: np.ndarray,
    ch1: np.ndarray,
    x_axis: np.ndarray,
    x_label: str,
    title: str,
) -> plt.Figure:
    magnitude = pedestal_subtracted_magnitude(ch0, ch1)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    event_indices = add_waveform_lines(
        ax,
        x_axis,
        magnitude,
        r"$\sqrt{(ch0-ped0)^2 + (ch1-ped1)^2}$",
    )
    ax.set_xlabel(x_label)
    ax.set_title(
        f"{title}\npedestal = event-wise mean of first {PEDESTAL_FRACTION:.0%}; "
        f"{event_indices.size}/{ch0.shape[0]} events"
    )
    fig.tight_layout()
    return fig


def save_plot_page(
    fig: plt.Figure,
    png_path: Path,
    pdf: PdfPages,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


def plot_dataset_raw_and_corrected(
    dataset: DatasetGroup,
    corrected_ch0: np.ndarray | None,
    corrected_ch1: np.ndarray | None,
    calibration: CalibrationSet | None,
    calibration_difference_hz: float | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = get_sample_rate_hz(dataset.records)
    x_axis, x_label = make_x_axis(dataset.ch0.shape[1], sample_rate_hz)

    report_path = output_dir / "waveform_raw_and_corrected.pdf"
    with PdfPages(report_path) as pdf:
        fig = plot_iq_tracks(
            dataset.ch0,
            dataset.ch1,
            title=f"{dataset.name}: RAW IQ",
        )
        save_plot_page(fig, output_dir / "raw_01_iq_tracks.png", pdf)

        fig = plot_ch0_ch1_waveforms(
            dataset.ch0,
            dataset.ch1,
            x_axis,
            x_label,
            title=f"{dataset.name}: RAW ch0 / ch1",
        )
        save_plot_page(fig, output_dir / "raw_02_ch0_ch1_waveforms.png", pdf)

        fig = plot_pedestal_subtracted_magnitude(
            dataset.ch0,
            dataset.ch1,
            x_axis,
            x_label,
            title=f"{dataset.name}: RAW pedestal-subtracted IQ magnitude",
        )
        save_plot_page(fig, output_dir / "raw_03_pedestal_subtracted_magnitude.png", pdf)

        if corrected_ch0 is not None and corrected_ch1 is not None and calibration is not None:
            match_text = (
                f"readout={dataset.readout_frequency_hz/1e9:.9f} GHz, "
                f"calibration={calibration.target_frequency_hz/1e9:.9f} GHz, "
                f"difference={calibration_difference_hz/1e6:.3f} MHz"
            )

            fig = plot_iq_tracks(
                corrected_ch0,
                corrected_ch1,
                title=f"{dataset.name}: CALIBRATED IQ\n{match_text}",
            )
            save_plot_page(fig, output_dir / "corrected_01_iq_tracks.png", pdf)

            fig = plot_ch0_ch1_waveforms(
                corrected_ch0,
                corrected_ch1,
                x_axis,
                x_label,
                title=f"{dataset.name}: CALIBRATED ch0 / ch1\n{match_text}",
            )
            save_plot_page(fig, output_dir / "corrected_02_ch0_ch1_waveforms.png", pdf)

            fig = plot_pedestal_subtracted_magnitude(
                corrected_ch0,
                corrected_ch1,
                x_axis,
                x_label,
                title=f"{dataset.name}: CALIBRATED pedestal-subtracted IQ magnitude\n{match_text}",
            )
            save_plot_page(
                fig,
                output_dir / "corrected_03_pedestal_subtracted_magnitude.png",
                pdf,
            )
        else:
            fig, ax = plt.subplots(figsize=(11.0, 4.0))
            ax.axis("off")
            ax.text(
                0.02,
                0.90,
                "Calibration was not applied.",
                transform=ax.transAxes,
                fontsize=16,
                va="top",
            )
            ax.text(
                0.02,
                0.72,
                f"frequency detection: {dataset.frequency_source}\n"
                f"readout frequency: {dataset.readout_frequency_hz}\n"
                f"matching tolerance: {MAX_CALIBRATION_MATCH_HZ/1e6:.3f} MHz",
                transform=ax.transAxes,
                fontsize=11,
                va="top",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[saved] {report_path}")


# =============================================================================
# CALIBRATION DIAGNOSTICS
# =============================================================================

def align_phase_to_reference(raw_phase: np.ndarray, reference_phase: np.ndarray) -> np.ndarray:
    return raw_phase + 2.0 * np.pi * np.round(
        (reference_phase - raw_phase) / (2.0 * np.pi)
    )


def add_circle(ax: plt.Axes, center: complex, radius: float, label: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    circle = center + radius * np.exp(1j * theta)
    ax.plot(circle.real, circle.imag, linewidth=1.6, label=label)
    ax.scatter([center.real], [center.imag], marker="x", s=70, label="center")


def save_calibration_diagnostics(
    calibration: CalibrationSet,
    output_root: Path,
) -> None:
    output_dir = output_root / sanitize_name(calibration.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequency_ghz = calibration.frequency_hz / 1e9
    tau_fit = calibration.tau_fit
    circle_fit = calibration.circle_fit
    geometry = calibration.geometry
    z_tau = apply_tau_correction(
        calibration.z_scan_raw,
        calibration.frequency_hz,
        tau_fit.tau_s,
    )

    pdf_path = output_dir / "calibration_diagnostics.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
        axes[0, 0].plot(
            calibration.z_scan_raw.real,
            calibration.z_scan_raw.imag,
            marker="o",
            markersize=3,
        )
        axes[0, 0].set_title("Raw IQ scan segment")
        axes[0, 0].set_xlabel("ch0")
        axes[0, 0].set_ylabel(f"{'+' if Q_SIGN > 0 else '-'}ch1")
        axes[0, 0].set_aspect("equal", adjustable="box")
        axes[0, 0].grid(alpha=0.3)

        phase_plot = align_phase_to_reference(
            np.angle(calibration.z_scan_raw),
            tau_fit.phase_fit_all_rad,
        )
        axes[0, 1].plot(frequency_ghz, phase_plot, "o-", ms=3, label="phase")
        axes[0, 1].plot(
            frequency_ghz,
            tau_fit.phase_fit_all_rad,
            linewidth=2.0,
            label="edge fit",
        )
        edge_indices = np.flatnonzero(tau_fit.edge_mask)
        axes[0, 1].scatter(
            frequency_ghz[edge_indices],
            tau_fit.edge_phase_used_rad,
            marker="s",
            facecolors="none",
            s=55,
            label=f"used: {N_EDGE_POINTS}+{N_EDGE_POINTS} points",
        )
        axes[0, 1].set_title(
            f"tau fit: {tau_fit.tau_s*1e9:.6f} ns\n"
            f"RMSE={tau_fit.rmse_edge_rad:.4g} rad, "
            f"R²={tau_fit.r_squared_edge:.6f}"
        )
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
        axes[1, 0].scatter(
            [circle_fit.resonance_point.real],
            [circle_fit.resonance_point.imag],
            marker="*",
            s=110,
            label="resonance point",
        )
        axes[1, 0].scatter(
            [geometry.point_p.real],
            [geometry.point_p.imag],
            marker="+",
            s=110,
            label="P = 2c - z_res",
        )
        axes[1, 0].set_title(
            f"Circle fit after tau correction\nradial RMS={circle_fit.radial_rms:.4g}"
        )
        axes[1, 0].set_xlabel("I_tau")
        axes[1, 0].set_ylabel("Q_tau")
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend(fontsize=7)

        axes[1, 1].plot(
            calibration.z_scan_final.real,
            calibration.z_scan_final.imag,
            "o-",
            ms=3,
            label="final corrected scan",
        )
        add_circle(
            axes[1, 1],
            calibration.final_circle_center,
            calibration.final_circle_radius,
            "final circle",
        )
        axes[1, 1].scatter([1.0], [0.0], marker="+", s=100, label="(1,0)")
        axes[1, 1].set_title(
            f"Final calibration\nalpha={geometry.alpha_rad:.6f}, "
            f"a={geometry.amplitude_a:.6g}, phi={geometry.phi_rad:.6f}"
        )
        axes[1, 1].set_xlabel("I_final")
        axes[1, 1].set_ylabel("Q_final")
        axes[1, 1].set_aspect("equal", adjustable="box")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend(fontsize=7)

        fig.suptitle(
            f"Calibration set {calibration.target_frequency_hz/1e9:.9f} GHz\n"
            f"source scan: {calibration.scan_path.name}",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / "calibration_diagnostics.png", dpi=PNG_DPI, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    params = {
        "name": calibration.name,
        "scan_file": str(calibration.scan_path),
        "target_frequency_hz": calibration.target_frequency_hz,
        "scan_frequency_min_hz": float(calibration.frequency_hz.min()),
        "scan_frequency_max_hz": float(calibration.frequency_hz.max()),
        "scan_point_count": int(calibration.frequency_hz.size),
        "q_sign": Q_SIGN,
        "n_edge_points_each_side": N_EDGE_POINTS,
        "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
        "tau_fit": {
            "tau_s": calibration.tau_fit.tau_s,
            "tau_ns": calibration.tau_fit.tau_s * 1e9,
            "intercept_b_rad": calibration.tau_fit.intercept_b_rad,
            "slope_rad_per_hz": calibration.tau_fit.slope_rad_per_hz,
            "rmse_edge_rad": calibration.tau_fit.rmse_edge_rad,
            "r_squared_edge": calibration.tau_fit.r_squared_edge,
            "high_edge_branch_shift": calibration.tau_fit.high_edge_branch_shift,
        },
        "circle_fit": {
            "center_real": calibration.circle_fit.center.real,
            "center_imag": calibration.circle_fit.center.imag,
            "radius": calibration.circle_fit.radius,
            "radial_rms": calibration.circle_fit.radial_rms,
            "resonance_index": calibration.circle_fit.resonance_index,
            "selected_frequency_hz": float(
                calibration.frequency_hz[calibration.circle_fit.resonance_index]
            ),
        },
        "geometry": {
            "point_p_real": calibration.geometry.point_p.real,
            "point_p_imag": calibration.geometry.point_p.imag,
            "alpha_rad": calibration.geometry.alpha_rad,
            "amplitude_a": calibration.geometry.amplitude_a,
            "phi_rad": calibration.geometry.phi_rad,
            "final_circle_center_real": calibration.final_circle_center.real,
            "final_circle_center_imag": calibration.final_circle_center.imag,
            "final_circle_radius": calibration.final_circle_radius,
        },
        "transform": (
            "z_final = 1 + (((z_raw * exp(i*2*pi*tau*f)) "
            "* exp(-i*alpha)) / a - 1) * exp(-i*phi)"
        ),
    }
    write_json(output_dir / "calibration_parameters.json", params)
    print(f"[saved] {pdf_path}")


# =============================================================================
# CORRECTED NPZ OUTPUT
# =============================================================================

def restore_original_orientation(
    array: np.ndarray,
    record: WaveformRecord,
) -> np.ndarray:
    restored = np.asarray(array)
    if record.original_was_transposed:
        restored = restored.T
    if record.original_was_1d:
        restored = restored.reshape(-1)
    return restored


def save_corrected_record(
    record: WaveformRecord,
    calibration: CalibrationSet,
    readout_frequency_hz: float,
    output_dir: Path,
) -> Path:
    z_raw = record.ch0 + 1j * Q_SIGN * record.ch1
    z_corrected = apply_full_calibration(
        z=z_raw,
        frequency_hz=readout_frequency_hz,
        calibration=calibration,
    )
    corrected_ch0 = np.real(z_corrected)
    corrected_ch1 = np.imag(z_corrected) / Q_SIGN

    output_payload: dict[str, Any] = {}
    with np.load(record.path, allow_pickle=False) as npz:
        for key in npz.keys():
            if key in (record.ch0_key, record.ch1_key):
                continue
            try:
                value = np.asarray(npz[key])
            except Exception:
                continue
            if value.dtype == object:
                print(f"[npz metadata skip] object array: {record.path.name}:{key}")
                continue
            output_payload[key] = value

    # 元データと同じ channel key 名で補正データを保存する。
    output_payload[record.ch0_key] = restore_original_orientation(corrected_ch0, record)
    output_payload[record.ch1_key] = restore_original_orientation(corrected_ch1, record)

    output_payload.update(
        {
            "iqcal_source_file": np.array(str(record.path)),
            "iqcal_q_sign": np.array(Q_SIGN),
            "iqcal_waveform_readout_frequency_hz": np.array(readout_frequency_hz),
            "iqcal_target_frequency_hz": np.array(calibration.target_frequency_hz),
            "iqcal_tau_s": np.array(calibration.tau_fit.tau_s),
            "iqcal_alpha_rad": np.array(calibration.geometry.alpha_rad),
            "iqcal_amplitude_a": np.array(calibration.geometry.amplitude_a),
            "iqcal_phi_rad": np.array(calibration.geometry.phi_rad),
            "iqcal_n_edge_points_each_side": np.array(N_EDGE_POINTS),
            "iqcal_pedestal_fraction_for_plots": np.array(PEDESTAL_FRACTION),
        }
    )

    try:
        relative = record.path.relative_to(record.path.parents[1])
    except ValueError:
        relative = Path(record.path.name)

    relative_parent = relative.parent
    output_file = (
        output_dir
        / "corrected_npz"
        / relative_parent
        / f"{record.path.stem}_iqcal.npz"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **output_payload)
    print(f"[saved corrected npz] {output_file}")
    return output_file


# =============================================================================
# DATASET PROCESSING
# =============================================================================

def process_dataset(
    dataset: DatasetGroup,
    calibration_sets: list[CalibrationSet],
) -> dict[str, Any]:
    dataset_output = OUTPUT_ROOT / "datasets" / sanitize_name(dataset.name)
    dataset_output.mkdir(parents=True, exist_ok=True)

    corrected_ch0: np.ndarray | None = None
    corrected_ch1: np.ndarray | None = None
    matched_calibration: CalibrationSet | None = None
    frequency_difference_hz: float | None = None
    corrected_files: list[str] = []
    status = "raw_only"
    message = ""

    if dataset.readout_frequency_hz is None:
        message = "readout frequency could not be detected"
        if not ALLOW_RAW_ONLY_WHEN_FREQUENCY_UNKNOWN:
            raise ValueError(f"{dataset.name}: {message}")
    else:
        matched_calibration, frequency_difference_hz = match_calibration_set(
            dataset.readout_frequency_hz,
            calibration_sets,
        )
        if matched_calibration is None:
            message = (
                "nearest calibration frequency is too far: "
                f"{frequency_difference_hz/1e6:.3f} MHz > "
                f"{MAX_CALIBRATION_MATCH_HZ/1e6:.3f} MHz"
            )
        else:
            z_raw = dataset.ch0 + 1j * Q_SIGN * dataset.ch1
            z_corrected = apply_full_calibration(
                z=z_raw,
                frequency_hz=dataset.readout_frequency_hz,
                calibration=matched_calibration,
            )
            corrected_ch0 = np.real(z_corrected)
            corrected_ch1 = np.imag(z_corrected) / Q_SIGN
            status = "calibrated"
            message = (
                f"matched to {matched_calibration.target_frequency_hz/1e9:.9f} GHz "
                f"(difference {frequency_difference_hz/1e6:.3f} MHz)"
            )

            if SAVE_CORRECTED_NPZ:
                for record in dataset.records:
                    output_file = save_corrected_record(
                        record=record,
                        calibration=matched_calibration,
                        readout_frequency_hz=dataset.readout_frequency_hz,
                        output_dir=dataset_output,
                    )
                    corrected_files.append(str(output_file))

    plot_dataset_raw_and_corrected(
        dataset=dataset,
        corrected_ch0=corrected_ch0,
        corrected_ch1=corrected_ch1,
        calibration=matched_calibration,
        calibration_difference_hz=frequency_difference_hz,
        output_dir=dataset_output,
    )

    summary = {
        "dataset_name": dataset.name,
        "base_dir": str(dataset.base_dir),
        "input_files": [str(record.path) for record in dataset.records],
        "n_files": len(dataset.records),
        "n_events": int(dataset.ch0.shape[0]),
        "n_samples": int(dataset.ch0.shape[1]),
        "readout_frequency_hz": dataset.readout_frequency_hz,
        "frequency_source": dataset.frequency_source,
        "status": status,
        "message": message,
        "matched_calibration_frequency_hz": (
            None if matched_calibration is None
            else matched_calibration.target_frequency_hz
        ),
        "frequency_difference_hz": frequency_difference_hz,
        "pedestal_fraction": PEDESTAL_FRACTION,
        "pedestal_definition": (
            "event-wise mean of first 10% separately for ch0 and ch1; "
            "magnitude=sqrt((ch0-ped0)^2+(ch1-ped1)^2)"
        ),
        "corrected_files": corrected_files,
    }
    write_json(dataset_output / "dataset_summary.json", summary)
    print(f"[dataset done] {dataset.name}: {status}; {message}")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("2026-07-14 batch IQ waveform analysis and calibration")
    print("=" * 88)
    print(f"DATA_ROOT       : {DATA_ROOT}")
    print(f"OUTPUT_ROOT     : {OUTPUT_ROOT}")
    print(f"N_EDGE_POINTS   : {N_EDGE_POINTS} points on each edge")
    print(f"PEDESTAL        : first {PEDESTAL_FRACTION:.0%}, event by event")
    print(f"MATCH TOLERANCE : {MAX_CALIBRATION_MATCH_HZ/1e6:.3f} MHz")
    print()

    datasets = discover_dataset_groups(DATA_ROOT)
    iq_scans = discover_iq_scans(DATA_ROOT)
    resonance_targets = discover_resonance_targets(DATA_ROOT)

    # resonance file が無い場合のみ、波形側で検出できた周波数を target 候補にする。
    if not resonance_targets:
        resonance_targets = sorted(
            {
                float(dataset.readout_frequency_hz)
                for dataset in datasets
                if dataset.readout_frequency_hz is not None
            }
        )
        print("[fallback] resonance targets were inferred from waveform readout frequencies")

    calibration_sets = build_calibration_sets(
        scans=iq_scans,
        target_frequencies_hz=resonance_targets,
    )

    if SAVE_CALIBRATION_DIAGNOSTICS:
        calibration_root = OUTPUT_ROOT / "calibration_sets"
        for calibration in calibration_sets:
            save_calibration_diagnostics(calibration, calibration_root)

    all_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for dataset in datasets:
        print()
        print("-" * 88)
        print(f"Processing: {dataset.name}")
        print("-" * 88)
        try:
            summary = process_dataset(dataset, calibration_sets)
            all_summaries.append(summary)
        except Exception as exc:
            traceback.print_exc()
            failures.append(
                {
                    "dataset_name": dataset.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[dataset failed] {dataset.name}: {exc}")

    batch_summary = {
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "settings": {
            "q_sign": Q_SIGN,
            "n_edge_points_each_side": N_EDGE_POINTS,
            "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
            "pedestal_fraction": PEDESTAL_FRACTION,
            "max_calibration_match_hz": MAX_CALIBRATION_MATCH_HZ,
            "max_events_to_plot": MAX_EVENTS_TO_PLOT,
            "plot_sample_stride": PLOT_SAMPLE_STRIDE,
        },
        "calibration_sets": [
            {
                "name": calibration.name,
                "target_frequency_hz": calibration.target_frequency_hz,
                "scan_file": str(calibration.scan_path),
                "tau_s": calibration.tau_fit.tau_s,
                "alpha_rad": calibration.geometry.alpha_rad,
                "amplitude_a": calibration.geometry.amplitude_a,
                "phi_rad": calibration.geometry.phi_rad,
            }
            for calibration in calibration_sets
        ],
        "datasets": all_summaries,
        "failures": failures,
    }
    write_json(OUTPUT_ROOT / "batch_summary.json", batch_summary)

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"successful datasets: {len(all_summaries)}")
    print(f"failed datasets    : {len(failures)}")
    print(f"results            : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
