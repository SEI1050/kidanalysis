from __future__ import annotations

"""
2026-07-09 の waveform 群に対して、iq_3.62K.npz から求めた
IQ 幾何学 calibration を適用し、pedestal 選別後の
sqrt(ch0_sub^2 + ch1_sub^2) の波形を x-scan / z-scan として描く。

処理の要点
-----------
1. iq_3.62K.npz から tau, alpha, a, phi を求める。
2. 各 waveform の ch0 + i * ch1 に同じ calibration を適用する。
3. 各イベントの先頭 10% サンプルから pedestal を求める。
4. ch0, ch1 それぞれについて
       half_width = (ped.max() - ped.min()) / laser_rate_hz / 2
   とし、ped median ± half_width に入るイベントのみ採用する。
   採用条件は ch0 と ch1 の両方を満たすこと。
5. 採用イベントごとに自分自身の pedestal を差し引き、
       magnitude(t) = sqrt(ch0_sub(t)^2 + ch1_sub(t)^2)
   を作る。
6. xscan: z=8.0 mm 固定で x ごとに、
   zscan: x=4.4 mm 固定で z ごとに、
   位置ごとに全採用イベントをまとめて
   - median waveform
   - mean waveform
   - mean の標準誤差 SEM = std/sqrt(N)
   を描く。
"""

from dataclasses import dataclass
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = Path("/Volumes/NO NAME/data/20260709")
IQ_SCAN = BASE_DIR / "iq_3.62K.npz"

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/"
    "iq_calibrated_waveform_scans"
)

READOUT_FREQUENCY_HZ = 5.501e9
Q_SIGN = +1

# IQ calibration settings
N_EDGE_POINTS = 15
CIRCLE_HALF_WIDTH_POINTS = 15

# Pedestal settings
BASELINE_FRACTION = 0.10
PEDESTAL_REDUCER = "mean"  # "mean" or "median"

# Scan definitions
XSCAN_FIXED_Z_MM = 8.0
ZSCAN_FIXED_X_MM = 4.4
POSITION_ATOL_MM = 1e-9

# 見やすさのため、通常強度と異なる可能性が高いフォルダはデフォルトで除外。
# 必要なら空タプル () に変更してください。
EXCLUDE_FOLDER_KEYWORDS: tuple[str, ...] = ("normal", "one-tenth")

# 平均波形の誤差バーを何サンプルおきに描くか
ERRORBAR_EVERY = 50

# 時間軸単位
TIME_UNIT = "us"  # "s", "ms", "us", "ns", or "sample"


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
    resonance_point: complex


@dataclass(frozen=True)
class GeometricCalibration:
    alpha_rad: float
    amplitude_a: float
    phi_rad: float
    point_p: complex
    center_final: complex


@dataclass(frozen=True)
class RunInfo:
    folder: Path
    waveform_file: Path
    readout_frequency_hz: float
    z_mm: float
    x_mm: float
    suffix: str


@dataclass
class ProcessedRun:
    folder_name: str
    waveform_name: str
    suffix: str
    x_mm: float
    z_mm: float
    laser_rate_hz: float
    n_events_total: int
    n_events_selected: int
    ped0_median: float
    ped1_median: float
    ped0_half_width: float
    ped1_half_width: float
    magnitude: np.ndarray          # shape = (n_selected, n_samples)
    time_axis: np.ndarray          # shape = (n_samples,)
    time_unit_label: str


@dataclass(frozen=True)
class GroupedWaveform:
    scan_axis: str
    position_value_mm: float
    fixed_value_mm: float
    n_runs: int
    n_events: int
    waveform_names: tuple[str, ...]
    folder_names: tuple[str, ...]
    time_axis: np.ndarray
    time_unit_label: str
    median_waveform: np.ndarray
    mean_waveform: np.ndarray
    sem_waveform: np.ndarray


# =============================================================================
# LOADERS / PARSERS
# =============================================================================


FOLDER_PATTERN = re.compile(
    r"^(?P<freq>[0-9]+(?:\.[0-9]+)?)GHz"
    r"_z=(?P<z>[+-]?[0-9]+(?:\.[0-9]+)?)mm"
    r"_x=(?P<x>[+-]?[0-9]+(?:\.[0-9]+)?)mm"
    r"(?P<suffix>.*)$"
)

LASER_RATE_PATTERN = re.compile(r"_(?P<hz>[0-9]+(?:\.[0-9]+)?)Hz\.npz$", re.IGNORECASE)


def convert_frequency_to_hz(frequency: np.ndarray) -> np.ndarray:
    f = np.asarray(frequency, dtype=float)
    typical = float(np.nanmedian(np.abs(f)))
    if typical < 100.0:
        return f * 1e9
    if typical < 1e7:
        return f * 1e6
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
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
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
            if key in keys:
                value = np.asarray(npz[key])
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
        ch0 = ch0.T
        ch1 = ch1.T

    return ch0, ch1, metadata


def parse_laser_rate_hz(path: Path, metadata: dict[str, np.ndarray]) -> float:
    match = LASER_RATE_PATTERN.search(path.name)
    if match:
        rate = float(match.group("hz"))
        if rate > 0:
            return rate

    if "daq_rate" in metadata:
        rate = float(np.asarray(metadata["daq_rate"]).squeeze())
        if np.isfinite(rate) and rate > 0:
            return rate

    raise ValueError(f"Could not obtain laser rate from filename or metadata: {path}")


def build_time_axis(n_samples: int, metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    if TIME_UNIT == "sample" or "sample_rate" not in metadata:
        return np.arange(n_samples, dtype=float), "sample index"

    sample_rate_hz = float(np.asarray(metadata["sample_rate"]).squeeze())
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return np.arange(n_samples, dtype=float), "sample index"

    time_s = np.arange(n_samples, dtype=float) / sample_rate_hz

    if TIME_UNIT == "s":
        return time_s, "time [s]"
    if TIME_UNIT == "ms":
        return time_s * 1e3, "time [ms]"
    if TIME_UNIT == "us":
        return time_s * 1e6, "time [µs]"
    if TIME_UNIT == "ns":
        return time_s * 1e9, "time [ns]"

    raise ValueError(f"Unknown TIME_UNIT: {TIME_UNIT}")


def discover_runs(base_dir: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        match = FOLDER_PATTERN.match(folder.name)
        if not match:
            continue

        if any(keyword in folder.name for keyword in EXCLUDE_FOLDER_KEYWORDS):
            print(f"[skip by keyword] {folder.name}")
            continue

        readout_frequency_hz = float(match.group("freq")) * 1e9
        z_mm = float(match.group("z"))
        x_mm = float(match.group("x"))
        suffix = match.group("suffix").lstrip("_")

        waveform_files = sorted(folder.glob("wf_*Hz.npz"))
        if not waveform_files:
            print(f"[skip: no waveform] {folder}")
            continue

        for waveform_file in waveform_files:
            runs.append(
                RunInfo(
                    folder=folder,
                    waveform_file=waveform_file,
                    readout_frequency_hz=readout_frequency_hz,
                    z_mm=z_mm,
                    x_mm=x_mm,
                    suffix=suffix,
                )
            )

    return runs


# =============================================================================
# IQ CALIBRATION
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
    n = frequency_hz.size
    edge_mask = make_edge_mask(n, n_edge_points)

    raw_phase = np.angle(z_scan)
    low_phase = np.unwrap(raw_phase[:n_edge_points])
    high_phase_base = np.unwrap(raw_phase[-n_edge_points:])

    f_edge = np.concatenate([frequency_hz[:n_edge_points], frequency_hz[-n_edge_points:]])
    f_ref = float(np.mean(f_edge))
    x_edge = f_edge - f_ref

    best: dict[str, object] | None = None
    for branch_shift in range(-5, 6):
        phase_edge = np.concatenate([low_phase, high_phase_base + 2.0 * np.pi * branch_shift])
        design = np.column_stack([x_edge, np.ones_like(x_edge)])
        slope, intercept = np.linalg.lstsq(design, phase_edge, rcond=None)[0]
        residual = phase_edge - (slope * x_edge + intercept)
        rss = float(np.sum(residual**2))
        if best is None or rss < float(best["rss"]):
            best = {
                "branch_shift": branch_shift,
                "slope": float(slope),
                "intercept": float(intercept),
                "phase_edge": phase_edge,
                "residual": residual,
                "rss": rss,
            }

    assert best is not None
    slope = float(best["slope"])
    intercept = float(best["intercept"])
    tau_s = -slope / (2.0 * np.pi)
    intercept_b_rad = intercept + 2.0 * np.pi * tau_s * f_ref

    residual = np.asarray(best["residual"], dtype=float)
    phase_edge = np.asarray(best["phase_edge"], dtype=float)
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_tot = float(np.sum((phase_edge - np.mean(phase_edge)) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1.0 - float(best["rss"]) / ss_tot

    return TauFitResult(
        tau_s=tau_s,
        intercept_b_rad=float(intercept_b_rad),
        slope_rad_per_hz=slope,
        f_ref_hz=f_ref,
        rmse_edge_rad=rmse,
        r_squared_edge=float(r_squared),
        high_edge_branch_shift=int(best["branch_shift"]),
    )


def apply_tau_correction(
    z: np.ndarray,
    frequency_hz: np.ndarray | float,
    tau_s: float,
) -> np.ndarray:
    return z * np.exp(1j * 2.0 * np.pi * tau_s * np.asarray(frequency_hz))


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
    radial_rms = float(np.sqrt(np.mean((np.abs(z - center) - radius) ** 2)))
    return center, radius, radial_rms


def fit_circle_near_resonance(
    frequency_hz: np.ndarray,
    z_tau: np.ndarray,
    readout_frequency_hz: float,
    half_width_points: int,
) -> CircleFitResult:
    resonance_index = int(np.argmin(np.abs(frequency_hz - readout_frequency_hz)))
    i0 = max(0, resonance_index - half_width_points)
    i1 = min(frequency_hz.size, resonance_index + half_width_points + 1)

    center, radius, radial_rms = algebraic_circle_fit(z_tau[i0:i1])
    resonance_point = complex(z_tau[resonance_index])
    return CircleFitResult(
        center=center,
        radius=radius,
        radial_rms=radial_rms,
        resonance_index=resonance_index,
        resonance_point=resonance_point,
    )


def compute_geometric_calibration(circle_result: CircleFitResult) -> GeometricCalibration:
    center = circle_result.center
    z_res = circle_result.resonance_point

    point_p = 2.0 * center - z_res
    alpha_rad = float(np.angle(point_p))
    rot_alpha = np.exp(-1j * alpha_rad)

    center_alpha = center * rot_alpha
    point_p_alpha = point_p * rot_alpha
    amplitude_a = float(np.abs(point_p_alpha))
    if not np.isfinite(amplitude_a) or amplitude_a == 0:
        raise ZeroDivisionError("Computed calibration amplitude a is invalid")

    center_a = center_alpha / amplitude_a
    phi_rad = float(np.angle(1.0 - center_a))
    center_final = 1.0 + (center_a - 1.0) * np.exp(-1j * phi_rad)

    return GeometricCalibration(
        alpha_rad=alpha_rad,
        amplitude_a=amplitude_a,
        phi_rad=phi_rad,
        point_p=point_p,
        center_final=center_final,
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
    return 1.0 + (z3 - 1.0) * np.exp(-1j * phi_rad)


def build_calibration() -> tuple[TauFitResult, CircleFitResult, GeometricCalibration]:
    frequency_hz, ch0_scan, ch1_scan = load_iq_scan(IQ_SCAN)
    z_scan_raw = ch0_scan + 1j * Q_SIGN * ch1_scan

    tau_fit = fit_tau_from_phase(
        frequency_hz=frequency_hz,
        z_scan=z_scan_raw,
        n_edge_points=N_EDGE_POINTS,
    )
    z_scan_tau = apply_tau_correction(z_scan_raw, frequency_hz, tau_fit.tau_s)

    circle_fit = fit_circle_near_resonance(
        frequency_hz=frequency_hz,
        z_tau=z_scan_tau,
        readout_frequency_hz=READOUT_FREQUENCY_HZ,
        half_width_points=CIRCLE_HALF_WIDTH_POINTS,
    )
    geom = compute_geometric_calibration(circle_fit)
    return tau_fit, circle_fit, geom


# =============================================================================
# PEDESTAL SELECTION / PROCESSING
# =============================================================================


def calculate_pedestal(channel: np.ndarray, baseline_stop: int) -> np.ndarray:
    baseline = channel[:, :baseline_stop]
    if PEDESTAL_REDUCER == "mean":
        return np.nanmean(baseline, axis=1)
    if PEDESTAL_REDUCER == "median":
        return np.nanmedian(baseline, axis=1)
    raise ValueError(f"Unknown PEDESTAL_REDUCER: {PEDESTAL_REDUCER}")


def process_run(
    run: RunInfo,
    tau_fit: TauFitResult,
    geom: GeometricCalibration,
) -> ProcessedRun | None:
    raw_ch0, raw_ch1, metadata = load_waveform(run.waveform_file)
    laser_rate_hz = parse_laser_rate_hz(run.waveform_file, metadata)

    z_raw = raw_ch0 + 1j * Q_SIGN * raw_ch1
    z_corrected = apply_full_calibration(
        z=z_raw,
        frequency_hz=run.readout_frequency_hz,
        tau_s=tau_fit.tau_s,
        alpha_rad=geom.alpha_rad,
        amplitude_a=geom.amplitude_a,
        phi_rad=geom.phi_rad,
    )

    corrected_ch0 = np.real(z_corrected)
    corrected_ch1 = Q_SIGN * np.imag(z_corrected)

    n_events, n_samples = corrected_ch0.shape
    baseline_stop = max(1, int(np.ceil(BASELINE_FRACTION * n_samples)))
    time_axis, time_unit_label = build_time_axis(n_samples, metadata)

    ped0 = calculate_pedestal(corrected_ch0, baseline_stop)
    ped1 = calculate_pedestal(corrected_ch1, baseline_stop)

    finite_ped = np.isfinite(ped0) & np.isfinite(ped1)
    if not np.any(finite_ped):
        print(f"[skip: no finite pedestal] {run.waveform_file}")
        return None

    ped0_valid = ped0[finite_ped]
    ped1_valid = ped1[finite_ped]

    ped0_median = float(np.median(ped0_valid))
    ped1_median = float(np.median(ped1_valid))
    ped0_half_width = float((np.max(ped0_valid) - np.min(ped0_valid)) / laser_rate_hz )
    ped1_half_width = float((np.max(ped1_valid) - np.min(ped1_valid)) / laser_rate_hz )

    selected = (
        finite_ped
        & (np.abs(ped0 - ped0_median) <= ped0_half_width)
        & (np.abs(ped1 - ped1_median) <= ped1_half_width)
    )
    n_selected = int(np.count_nonzero(selected))
    if n_selected == 0:
        print(
            f"[skip: 0 selected] {run.waveform_file.name} | "
            f"ch0 half={ped0_half_width:.6g}, ch1 half={ped1_half_width:.6g}"
        )
        return None

    ch0_sub = corrected_ch0[selected] - ped0[selected, np.newaxis]
    ch1_sub = corrected_ch1[selected] - ped1[selected, np.newaxis]
    magnitude = np.hypot(ch0_sub, ch1_sub)

    print(
        f"[ok] {run.folder.name}/{run.waveform_file.name} | "
        f"selected={n_selected}/{n_events} ({n_selected/n_events:.1%})"
    )

    return ProcessedRun(
        folder_name=run.folder.name,
        waveform_name=run.waveform_file.name,
        suffix=run.suffix,
        x_mm=run.x_mm,
        z_mm=run.z_mm,
        laser_rate_hz=laser_rate_hz,
        n_events_total=n_events,
        n_events_selected=n_selected,
        ped0_median=ped0_median,
        ped1_median=ped1_median,
        ped0_half_width=ped0_half_width,
        ped1_half_width=ped1_half_width,
        magnitude=magnitude,
        time_axis=time_axis,
        time_unit_label=time_unit_label,
    )


# =============================================================================
# GROUPING / STATISTICS
# =============================================================================


def _check_compatible_time_axes(runs: list[ProcessedRun]) -> tuple[np.ndarray, str]:
    first = runs[0]
    time_axis = np.asarray(first.time_axis, dtype=float)
    unit_label = first.time_unit_label
    for run in runs[1:]:
        if run.time_unit_label != unit_label:
            raise ValueError("Time-axis unit labels are inconsistent across runs")
        if run.time_axis.shape != time_axis.shape:
            raise ValueError("Waveform lengths are inconsistent across runs")
        if not np.allclose(run.time_axis, time_axis, rtol=0.0, atol=0.0):
            raise ValueError("Time axes are inconsistent across runs")
    return time_axis, unit_label


def build_grouped_waveforms(
    processed_runs: list[ProcessedRun],
    scan_axis: str,
    fixed_value_mm: float,
) -> list[GroupedWaveform]:
    if scan_axis == "x":
        selected_runs = [
            run for run in processed_runs
            if np.isclose(run.z_mm, fixed_value_mm, atol=POSITION_ATOL_MM, rtol=0.0)
        ]
        key_func = lambda run: run.x_mm
    elif scan_axis == "z":
        selected_runs = [
            run for run in processed_runs
            if np.isclose(run.x_mm, fixed_value_mm, atol=POSITION_ATOL_MM, rtol=0.0)
        ]
        key_func = lambda run: run.z_mm
    else:
        raise ValueError(f"Unknown scan_axis: {scan_axis}")

    if not selected_runs:
        return []

    positions = sorted({float(key_func(run)) for run in selected_runs})
    grouped: list[GroupedWaveform] = []

    for position in positions:
        runs_here = [run for run in selected_runs if np.isclose(key_func(run), position, atol=POSITION_ATOL_MM, rtol=0.0)]
        if not runs_here:
            continue

        time_axis, time_unit_label = _check_compatible_time_axes(runs_here)
        magnitude = np.concatenate([run.magnitude for run in runs_here], axis=0)
        n_events = int(magnitude.shape[0])

        median_waveform = np.nanmedian(magnitude, axis=0)
        mean_waveform = np.nanmean(magnitude, axis=0)

        if n_events >= 2:
            sem_waveform = np.nanstd(magnitude, axis=0, ddof=1) / np.sqrt(n_events)
        else:
            sem_waveform = np.zeros_like(mean_waveform)

        grouped.append(
            GroupedWaveform(
                scan_axis=scan_axis,
                position_value_mm=float(position),
                fixed_value_mm=float(fixed_value_mm),
                n_runs=len(runs_here),
                n_events=n_events,
                waveform_names=tuple(run.waveform_name for run in runs_here),
                folder_names=tuple(run.folder_name for run in runs_here),
                time_axis=time_axis,
                time_unit_label=time_unit_label,
                median_waveform=median_waveform,
                mean_waveform=mean_waveform,
                sem_waveform=sem_waveform,
            )
        )

    return grouped


# =============================================================================
# OUTPUT / PLOTTING
# =============================================================================


def save_run_summary_csv(processed_runs: list[ProcessedRun], path: Path) -> None:
    fieldnames = [
        "folder_name",
        "waveform_name",
        "suffix",
        "x_mm",
        "z_mm",
        "laser_rate_hz",
        "n_events_total",
        "n_events_selected",
        "selected_fraction",
        "ped0_median",
        "ped1_median",
        "ped0_half_width",
        "ped1_half_width",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in processed_runs:
            writer.writerow(
                {
                    "folder_name": run.folder_name,
                    "waveform_name": run.waveform_name,
                    "suffix": run.suffix,
                    "x_mm": run.x_mm,
                    "z_mm": run.z_mm,
                    "laser_rate_hz": run.laser_rate_hz,
                    "n_events_total": run.n_events_total,
                    "n_events_selected": run.n_events_selected,
                    "selected_fraction": run.n_events_selected / run.n_events_total,
                    "ped0_median": run.ped0_median,
                    "ped1_median": run.ped1_median,
                    "ped0_half_width": run.ped0_half_width,
                    "ped1_half_width": run.ped1_half_width,
                }
            )
    print(f"[saved] {path}")


def save_group_summary_csv(grouped: list[GroupedWaveform], path: Path) -> None:
    fieldnames = [
        "scan_axis",
        "position_value_mm",
        "fixed_value_mm",
        "n_runs",
        "n_events",
        "folder_names",
        "waveform_names",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for group in grouped:
            writer.writerow(
                {
                    "scan_axis": group.scan_axis,
                    "position_value_mm": group.position_value_mm,
                    "fixed_value_mm": group.fixed_value_mm,
                    "n_runs": group.n_runs,
                    "n_events": group.n_events,
                    "folder_names": " | ".join(group.folder_names),
                    "waveform_names": " | ".join(group.waveform_names),
                }
            )
    print(f"[saved] {path}")


def _legend_label(group: GroupedWaveform) -> str:
    axis_name = "x" if group.scan_axis == "x" else "z"
    return f"{axis_name}={group.position_value_mm:g} mm, n={group.n_events}, runs={group.n_runs}"


def plot_waveform_scan(
    grouped: list[GroupedWaveform],
    scan_axis: str,
    png_path: Path,
    pdf: PdfPages,
) -> None:
    if not grouped:
        print(f"[skip plot: no grouped data] {scan_axis}")
        return

    grouped = sorted(grouped, key=lambda g: g.position_value_mm)
    fixed_value_mm = grouped[0].fixed_value_mm
    time_axis = grouped[0].time_axis
    time_unit_label = grouped[0].time_unit_label

    if scan_axis == "x":
        title = f"IQ-calibrated waveform x-scan (fixed z = {fixed_value_mm:g} mm)"
    elif scan_axis == "z":
        title = f"IQ-calibrated waveform z-scan (fixed x = {fixed_value_mm:g} mm)"
    else:
        raise ValueError(f"Unknown scan_axis: {scan_axis}")

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 10.0), sharex=True)
    fig.suptitle(title, fontsize=15)

    ax_median, ax_mean = axes

    for group in grouped:
        label = _legend_label(group)
        ax_median.plot(group.time_axis, group.median_waveform, label=label)
        ax_mean.errorbar(
            group.time_axis,
            group.mean_waveform,
            yerr=group.sem_waveform,
            fmt="-",
            linewidth=1.2,
            capsize=2.0,
            errorevery=ERRORBAR_EVERY,
            label=label,
        )

    ax_median.set_title("Median waveform of sqrt(ch0_sub^2 + ch1_sub^2)")
    ax_median.set_ylabel("median magnitude")
    ax_median.grid(alpha=0.3)
    ax_median.legend(fontsize=8, loc="best", title="laser position")

    ax_mean.set_title("Mean waveform of sqrt(ch0_sub^2 + ch1_sub^2) with SEM error bars")
    ax_mean.set_xlabel(time_unit_label)
    ax_mean.set_ylabel("mean magnitude")
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(fontsize=8, loc="best", title="laser position")

    summary_text = (
        f"pedestal = first {BASELINE_FRACTION:.0%}; "
        f"selection width = (max-min)/laser_rate/2; "
        f"mean error bar = SEM"
    )
    ax_mean.text(
        0.01,
        0.02,
        summary_text,
        transform=ax_mean.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


def save_calibration_json(
    tau_fit: TauFitResult,
    circle_fit: CircleFitResult,
    geom: GeometricCalibration,
    path: Path,
) -> None:
    payload = {
        "iq_scan": str(IQ_SCAN),
        "readout_frequency_hz": READOUT_FREQUENCY_HZ,
        "q_sign": Q_SIGN,
        "tau_fit": {
            "tau_s": tau_fit.tau_s,
            "tau_ns": tau_fit.tau_s * 1e9,
            "intercept_b_rad": tau_fit.intercept_b_rad,
            "slope_rad_per_hz": tau_fit.slope_rad_per_hz,
            "rmse_edge_rad": tau_fit.rmse_edge_rad,
            "r_squared_edge": tau_fit.r_squared_edge,
            "high_edge_branch_shift": tau_fit.high_edge_branch_shift,
        },
        "circle_fit": {
            "center_real": circle_fit.center.real,
            "center_imag": circle_fit.center.imag,
            "radius": circle_fit.radius,
            "radial_rms": circle_fit.radial_rms,
            "resonance_index": circle_fit.resonance_index,
        },
        "geometric_calibration": {
            "alpha_rad": geom.alpha_rad,
            "amplitude_a": geom.amplitude_a,
            "phi_rad": geom.phi_rad,
            "point_p_real": geom.point_p.real,
            "point_p_imag": geom.point_p.imag,
            "center_final_real": geom.center_final.real,
            "center_final_imag": geom.center_final.imag,
        },
        "analysis_settings": {
            "baseline_fraction": BASELINE_FRACTION,
            "pedestal_reducer": PEDESTAL_REDUCER,
            "xscan_fixed_z_mm": XSCAN_FIXED_Z_MM,
            "zscan_fixed_x_mm": ZSCAN_FIXED_X_MM,
            "exclude_folder_keywords": list(EXCLUDE_FOLDER_KEYWORDS),
            "time_unit": TIME_UNIT,
            "errorbar_every": ERRORBAR_EVERY,
            "mean_error_definition": "SEM = std(ddof=1) / sqrt(N)",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILD IQ CALIBRATION")
    print("=" * 80)
    tau_fit, circle_fit, geom = build_calibration()
    print(f"tau   = {tau_fit.tau_s * 1e9:.9f} ns")
    print(f"alpha = {geom.alpha_rad:.12g} rad")
    print(f"a     = {geom.amplitude_a:.12g}")
    print(f"phi   = {geom.phi_rad:.12g} rad")

    save_calibration_json(
        tau_fit=tau_fit,
        circle_fit=circle_fit,
        geom=geom,
        path=OUTPUT_DIR / "calibration_and_analysis_settings.json",
    )

    print()
    print("=" * 80)
    print("DISCOVER AND PROCESS WAVEFORMS")
    print("=" * 80)
    runs = discover_runs(BASE_DIR)
    if not runs:
        raise RuntimeError(f"No waveform runs found under {BASE_DIR}")
    print(f"found {len(runs)} waveform file(s)")

    processed_runs: list[ProcessedRun] = []
    failures: list[str] = []

    for index, run in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] {run.folder.name}/{run.waveform_file.name}")
        try:
            result = process_run(run, tau_fit=tau_fit, geom=geom)
        except Exception as error:
            message = f"{run.waveform_file}: {type(error).__name__}: {error}"
            failures.append(message)
            print(f"[ERROR] {message}")
            continue
        if result is not None:
            processed_runs.append(result)

    if not processed_runs:
        raise RuntimeError("No run produced a valid result. Check pedestal selection widths.")

    save_run_summary_csv(processed_runs, OUTPUT_DIR / "run_summary.csv")

    grouped_x = build_grouped_waveforms(
        processed_runs=processed_runs,
        scan_axis="x",
        fixed_value_mm=XSCAN_FIXED_Z_MM,
    )
    grouped_z = build_grouped_waveforms(
        processed_runs=processed_runs,
        scan_axis="z",
        fixed_value_mm=ZSCAN_FIXED_X_MM,
    )

    save_group_summary_csv(grouped_x, OUTPUT_DIR / "xscan_group_summary.csv")
    save_group_summary_csv(grouped_z, OUTPUT_DIR / "zscan_group_summary.csv")

    pdf_path = OUTPUT_DIR / "waveform_xscan_zscan.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_waveform_scan(
            grouped=grouped_x,
            scan_axis="x",
            png_path=OUTPUT_DIR / f"xscan_waveforms_z{XSCAN_FIXED_Z_MM:g}mm.png",
            pdf=pdf,
        )
        plot_waveform_scan(
            grouped=grouped_z,
            scan_axis="z",
            png_path=OUTPUT_DIR / f"zscan_waveforms_x{ZSCAN_FIXED_X_MM:g}mm.png",
            pdf=pdf,
        )
    print(f"[saved] {pdf_path}")

    if failures:
        failure_path = OUTPUT_DIR / "failed_runs.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[saved] {failure_path}")
        print(f"WARNING: {len(failures)} run(s) failed")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"valid processed runs: {len(processed_runs)}")
    print(f"xscan groups: {len(grouped_x)}")
    print(f"zscan groups: {len(grouped_z)}")
    print(f"output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
