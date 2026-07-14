from __future__ import annotations

"""
Analysis script for the 2026-05-21 / 2026-05-22 Americium KID waveform data.

Input  : /Volumes/NO NAME/data/20260521/
Output : /Users/kubokosei/software/kidanalysis/data/20260521/americium_analysis/

The script is intentionally robust against slightly different npz key names.
It handles two types of npz files:
  1. iq_scan files with key dd, where
        dd[:, 0] = frequency
        dd[:, 1] = ch0
        dd[:, 2] = ch1
  2. oscilloscope waveform files containing ch0/ch1-like arrays.

Run:
    cd /Users/kubokosei/software/kidanalysis
    python analyze_20260521_americium.py
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_DIR = Path("/Volumes/NO NAME/data/20260521")
OUTPUT_DIR = Path("/Users/kubokosei/software/kidanalysis/data/20260521/americium_analysis")

# 20260521/20260527 waveform files in this experiment are usually 2.5 GS/s.
# If sample_rate exists inside npz, it will be used instead.
DEFAULT_SAMPLE_RATE_HZ = 2.5e9

# Baseline is evaluated using the first part of the trace.
BASELINE_FRACTION = 0.20
BASELINE_MAX_SAMPLES = 1000
BASELINE_MIN_SAMPLES = 50

# Pulse search region. Leave broad because trigger position can differ by file.
SEARCH_START_FRACTION = 0.05
SEARCH_END_FRACTION = 0.95

# For per-event amplitude, search around the mean pulse peak.
EVENT_PEAK_HALF_WIDTH_SAMPLES = 80

# For event peak IQ point, average a small window around the event peak.
PEAK_AVG_HALF_WIDTH_SAMPLES = 5

# Event scatter can become huge for 40k events. The saved scatter plot samples at most this many.
MAX_SCATTER_POINTS = 6000

# Downsample displayed mean waveforms if traces are long.
MAX_WAVEFORM_PLOT_POINTS = 2500

# Use these to flip signs if later you decide ch0/ch1 sign convention should be changed.
CH0_SIGN = 1.0
CH1_SIGN = 1.0

# Skip folders named "bin" by default. Set True if you want to analyze files inside bin/ too.
ANALYZE_BIN_DIR = False

# Frequencies to annotate in IQ scan plots.
IQSCAN_REFERENCE_FREQ_GHZ = [5.476, 5.485, 5.491, 5.496, 5.504]


# =============================================================================
# RUN LOG METADATA
# =============================================================================

RUN_INFO: dict[str, dict[str, object]] = {
    # 5/21 afternoon
    "data_0521_135103": {
        "label": "13:51 Am, ch1 tuned, 5.476GHz -8dBm",
        "freq_ghz": 5.476,
        "rf_dbm": -8.0,
        "trigger_mv": -2.0,
        "temperature_k": None,
        "group": "early_on_res_candidate",
        "note": "Log says CH1 signal was made visible by pipe-length adjustment. This run may have been measured with DC coupling.",
    },
    "data_0521_144519": {
        "label": "14:45 Am, AC, 5.476GHz -5dBm",
        "freq_ghz": 5.476,
        "rf_dbm": -5.0,
        "trigger_mv": -3.0,
        "temperature_k": 5.6,
        "group": "on_res_candidate",
        "note": "Retaken after switching to AC coupling.",
    },
    "data_0521_154018": {
        "label": "15:40 50ohm control, 5.476GHz 0dBm",
        "freq_ghz": 5.476,
        "rf_dbm": 0.0,
        "trigger_mv": -3.0,
        "temperature_k": None,
        "group": "50ohm_control",
        "note": "50 ohm termination / control run. Low statistics in log.",
    },
    "data_0521_164551": {
        "label": "16:45 Off-res?, 5.496GHz 0dBm",
        "freq_ghz": 5.496,
        "rf_dbm": 0.0,
        "trigger_mv": -7.0,
        "temperature_k": None,
        "group": "off_res_candidate",
        "note": "Off-resonance triggerable run; noise contamination noted.",
    },
    "data_0521_171612": {
        "label": "17:16 extra run, metadata not fixed",
        "freq_ghz": None,
        "rf_dbm": None,
        "trigger_mv": None,
        "temperature_k": None,
        "group": "unknown",
        "note": "Folder appears in screenshot but detailed log was not specified. Edit RUN_INFO if needed.",
    },
    "data_0521_184401": {
        "label": "18:44 extra run, metadata not fixed",
        "freq_ghz": None,
        "rf_dbm": None,
        "trigger_mv": None,
        "temperature_k": None,
        "group": "unknown",
        "note": "Folder appears in screenshot but detailed log was not specified. Edit RUN_INFO if needed.",
    },
    "data_0521_190308": {
        "label": "19:03 pre/high-stat?, 5.473GHz?",
        "freq_ghz": 5.473,
        "rf_dbm": -8.0,
        "trigger_mv": -1.6,
        "temperature_k": 5.5,
        "group": "high_stat_on_res_uncertain",
        "note": "Likely related to the high-stat alpha run; exact mapping should be checked from file count/timestamps.",
    },
    "data_0521_190724": {
        "label": "19:07 high-stat Am, 5.473GHz -8dBm",
        "freq_ghz": 5.473,
        "rf_dbm": -8.0,
        "trigger_mv": -1.6,
        "temperature_k": 5.5,
        "group": "high_stat_on_res_uncertain",
        "note": "Log says this may or may not have been true on resonance; later note says true fr was 5.485GHz.",
    },
    # 5/22 morning/afternoon folders are under the same 20260521 root in the screenshot.
    "data_0522_100846": {
        "label": "10:08 setup/check, 5.504GHz off-res?",
        "freq_ghz": 5.504,
        "rf_dbm": -4.0,
        "trigger_mv": -1.3,
        "temperature_k": 5.5,
        "group": "off_res_0522",
        "note": "The log has an off-resonance 5.504GHz run around 10:16. Check exact folder mapping.",
    },
    "data_0522_101551": {
        "label": "10:15 Off-res, 5.504GHz -4dBm",
        "freq_ghz": 5.504,
        "rf_dbm": -4.0,
        "trigger_mv": -1.3,
        "temperature_k": 5.5,
        "group": "off_res_0522",
        "note": "Off-resonance with vertical offset -30mV in log.",
    },
    "data_0522_132228": {
        "label": "13:22 setup/check, true fr nearby 5.485GHz",
        "freq_ghz": 5.485,
        "rf_dbm": -8.0,
        "trigger_mv": 1.4,
        "temperature_k": 5.5,
        "group": "on_res_0522",
        "note": "Folder appears before the logged 13:40/13:54 on-res runs. Check exact mapping if needed.",
    },
    "data_0522_132707": {
        "label": "13:27 on-res candidate, 5.485GHz -8dBm",
        "freq_ghz": 5.485,
        "rf_dbm": -8.0,
        "trigger_mv": 1.4,
        "temperature_k": 5.5,
        "group": "on_res_0522",
        "note": "True resonance was noted as 5.485GHz. This may correspond to the 1500-event run.",
    },
    "data_0522_135424": {
        "label": "13:54 high-stat true on-res, 5.485GHz -8dBm",
        "freq_ghz": 5.485,
        "rf_dbm": -8.0,
        "trigger_mv": 1.4,
        "temperature_k": 5.5,
        "group": "high_stat_on_res_0522",
        "note": "High-stat true on-resonance retake in the log.",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

@dataclass
class WaveformFile:
    path: Path
    ch0: np.ndarray
    ch1: np.ndarray
    sample_rate_hz: float


@dataclass
class FirstPassResult:
    n_events: int
    n_samples: int
    sample_rate_hz: float
    time_ns: np.ndarray
    mean0: np.ndarray
    mean1: np.ndarray
    mean_r: np.ndarray
    peak_idx: int
    response_u0: float
    response_u1: float
    response_norm: float


def safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)


def npz_keys(path: Path) -> list[str]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return list(z.keys())
    except Exception:
        return []


def has_iqscan(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "dd" not in z:
                return False
            dd = np.asarray(z["dd"])
            return dd.ndim == 2 and dd.shape[1] >= 3
    except Exception:
        return False


def find_sample_rate(z: np.lib.npyio.NpzFile) -> float:
    candidates = [
        "sample_rate", "samplerate", "sampling_rate", "fs", "rate",
        "sample_rate_hz", "sampling_rate_hz",
    ]
    for key in candidates:
        if key in z:
            try:
                value = np.asarray(z[key]).squeeze()
                value = float(value)
                if value > 0:
                    return value
            except Exception:
                pass
    return DEFAULT_SAMPLE_RATE_HZ


def get_array_by_candidates(z: np.lib.npyio.NpzFile, candidates: list[str]) -> np.ndarray | None:
    lower_to_key = {k.lower(): k for k in z.keys()}
    for cand in candidates:
        if cand in z:
            arr = np.asarray(z[cand])
            if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
                return arr
        low = cand.lower()
        if low in lower_to_key:
            arr = np.asarray(z[lower_to_key[low]])
            if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
                return arr
    return None


def as_event_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim > 2:
        # Flatten all leading dimensions as event index.
        arr = arr.reshape((-1, arr.shape[-1]))
    return arr.astype(np.float64, copy=False)


def split_combined_channel_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        return None

    # Common cases:
    #   (2, n_events, n_samples)
    #   (n_events, 2, n_samples)
    #   (n_events, n_samples, 2)
    if arr.shape[0] == 2:
        return as_event_matrix(arr[0]), as_event_matrix(arr[1])
    if arr.ndim >= 3 and arr.shape[1] == 2:
        return as_event_matrix(arr[:, 0, ...]), as_event_matrix(arr[:, 1, ...])
    if arr.shape[-1] == 2:
        return as_event_matrix(np.moveaxis(arr[..., 0], -1, -1)), as_event_matrix(np.moveaxis(arr[..., 1], -1, -1))
    return None


def load_waveform_npz(path: Path) -> WaveformFile | None:
    """Return ch0/ch1 arrays if this npz looks like a waveform file."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if "dd" in z:
                # This is an iq_scan file, not an oscilloscope waveform file.
                return None

            sample_rate_hz = find_sample_rate(z)

            ch0_candidates = [
                "ch0", "CH0", "Ch0", "channel0", "Channel0", "wave0", "wf0", "y0",
                "data_ch0", "ch0_waveform", "ch0_wf", "trace0",
            ]
            ch1_candidates = [
                "ch1", "CH1", "Ch1", "channel1", "Channel1", "wave1", "wf1", "y1",
                "data_ch1", "ch1_waveform", "ch1_wf", "trace1",
            ]
            ch0 = get_array_by_candidates(z, ch0_candidates)
            ch1 = get_array_by_candidates(z, ch1_candidates)

            if ch0 is not None and ch1 is not None:
                ch0 = as_event_matrix(ch0) * CH0_SIGN
                ch1 = as_event_matrix(ch1) * CH1_SIGN
            else:
                # Fallback: look for exactly two numeric arrays with compatible shapes.
                skip_words = ("time", "timestamp", "sample", "rate", "freq", "dd", "ref", "position")
                arrays: list[tuple[str, np.ndarray]] = []
                for key in z.keys():
                    if any(w in key.lower() for w in skip_words):
                        continue
                    arr = np.asarray(z[key])
                    if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 1 and arr.size > 100:
                        arrays.append((key, arr))

                # Try a combined channel array first.
                for _, arr in arrays:
                    split = split_combined_channel_array(arr)
                    if split is not None:
                        ch0, ch1 = split
                        ch0 = ch0 * CH0_SIGN
                        ch1 = ch1 * CH1_SIGN
                        break

                if ch0 is None or ch1 is None:
                    # Try two similarly shaped arrays.
                    matrices = [(k, as_event_matrix(a)) for k, a in arrays]
                    for i in range(len(matrices)):
                        for j in range(i + 1, len(matrices)):
                            a = matrices[i][1]
                            b = matrices[j][1]
                            if a.shape == b.shape:
                                ch0 = a * CH0_SIGN
                                ch1 = b * CH1_SIGN
                                warnings.warn(
                                    f"Using fallback channel keys for {path.name}: "
                                    f"{matrices[i][0]} -> ch0, {matrices[j][0]} -> ch1"
                                )
                                break
                        if ch0 is not None and ch1 is not None:
                            break

            if ch0 is None or ch1 is None:
                return None

            n = min(ch0.shape[0], ch1.shape[0])
            m = min(ch0.shape[1], ch1.shape[1])
            if n <= 0 or m <= 10:
                return None
            ch0 = np.asarray(ch0[:n, :m], dtype=np.float64)
            ch1 = np.asarray(ch1[:n, :m], dtype=np.float64)
            return WaveformFile(path=path, ch0=ch0, ch1=ch1, sample_rate_hz=sample_rate_hz)
    except Exception as e:
        warnings.warn(f"Could not load {path}: {e}")
        return None


def discover_npz_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    files = sorted(root.rglob("*.npz"))
    if not ANALYZE_BIN_DIR:
        files = [p for p in files if "bin" not in p.relative_to(root).parts]
    return files


def group_waveform_files(root: Path, files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        if has_iqscan(path):
            continue
        rel = path.relative_to(root)
        # Use the first data_* folder as run name if present.
        run_name = None
        for part in rel.parts:
            if part.startswith("data_"):
                run_name = part
                break
        if run_name is None:
            run_name = rel.parts[0] if len(rel.parts) > 1 else "root"
        groups.setdefault(run_name, []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def baseline_slice(n_samples: int) -> slice:
    n_base = int(n_samples * BASELINE_FRACTION)
    n_base = max(BASELINE_MIN_SAMPLES, min(BASELINE_MAX_SAMPLES, n_base))
    n_base = min(n_base, max(1, n_samples // 2))
    return slice(0, n_base)


def search_slice(n_samples: int) -> slice:
    i0 = int(n_samples * SEARCH_START_FRACTION)
    i1 = int(n_samples * SEARCH_END_FRACTION)
    i0 = max(0, min(n_samples - 2, i0))
    i1 = max(i0 + 1, min(n_samples, i1))
    return slice(i0, i1)


def load_first_compatible_file(paths: list[Path]) -> WaveformFile | None:
    for path in paths:
        wf = load_waveform_npz(path)
        if wf is not None:
            return wf
    return None


def first_pass(paths: list[Path]) -> FirstPassResult | None:
    first = load_first_compatible_file(paths)
    if first is None:
        return None

    n_samples = first.ch0.shape[1]
    sample_rate_hz = first.sample_rate_hz
    base_sl = baseline_slice(n_samples)

    sum0 = np.zeros(n_samples, dtype=np.float64)
    sum1 = np.zeros(n_samples, dtype=np.float64)
    n_events = 0

    for path in paths:
        wf = load_waveform_npz(path)
        if wf is None:
            continue
        m = min(n_samples, wf.ch0.shape[1], wf.ch1.shape[1])
        if m != n_samples:
            # Keep the common initial length. This branch is rare.
            if m < n_samples:
                sum0 = sum0[:m]
                sum1 = sum1[:m]
                n_samples = m
                base_sl = baseline_slice(n_samples)
            ch0 = wf.ch0[:, :n_samples]
            ch1 = wf.ch1[:, :n_samples]
        else:
            ch0 = wf.ch0
            ch1 = wf.ch1

        ped0 = np.median(ch0[:, base_sl], axis=1)
        ped1 = np.median(ch1[:, base_sl], axis=1)
        sum0 += np.sum(ch0 - ped0[:, None], axis=0)
        sum1 += np.sum(ch1 - ped1[:, None], axis=0)
        n_events += ch0.shape[0]

    if n_events == 0:
        return None

    mean0 = sum0 / n_events
    mean1 = sum1 / n_events
    mean_r = np.sqrt(mean0**2 + mean1**2)
    s_sl = search_slice(n_samples)
    local_peak = int(np.argmax(mean_r[s_sl]))
    peak_idx = s_sl.start + local_peak

    vec0 = float(mean0[peak_idx])
    vec1 = float(mean1[peak_idx])
    norm = float(math.hypot(vec0, vec1))
    if norm <= 0 or not np.isfinite(norm):
        response_u0, response_u1 = 1.0, 0.0
        norm = 0.0
    else:
        response_u0, response_u1 = vec0 / norm, vec1 / norm

    time_ns = np.arange(n_samples, dtype=np.float64) / sample_rate_hz * 1e9
    return FirstPassResult(
        n_events=n_events,
        n_samples=n_samples,
        sample_rate_hz=sample_rate_hz,
        time_ns=time_ns,
        mean0=mean0,
        mean1=mean1,
        mean_r=mean_r,
        peak_idx=peak_idx,
        response_u0=response_u0,
        response_u1=response_u1,
        response_norm=norm,
    )


def second_pass(paths: list[Path], fp: FirstPassResult) -> dict[str, np.ndarray | float | int]:
    n_samples = fp.n_samples
    base_sl = baseline_slice(n_samples)
    s_sl = search_slice(n_samples)

    pk0 = max(s_sl.start, fp.peak_idx - EVENT_PEAK_HALF_WIDTH_SAMPLES)
    pk1 = min(s_sl.stop, fp.peak_idx + EVENT_PEAK_HALF_WIDTH_SAMPLES + 1)
    event_search = slice(pk0, pk1)

    # Broad integral window around the pulse. The clipping makes the area robust to sign/noise.
    int0 = max(s_sl.start, fp.peak_idx - 3 * EVENT_PEAK_HALF_WIDTH_SAMPLES)
    int1 = min(s_sl.stop, fp.peak_idx + 8 * EVENT_PEAK_HALF_WIDTH_SAMPLES + 1)
    integral_sl = slice(int0, int1)

    rows: dict[str, list[float]] = {
        "base0": [], "base1": [],
        "peak0": [], "peak1": [],
        "d0": [], "d1": [],
        "amp_proj": [], "amp_ch0_signed": [], "amp_ch1_signed": [],
        "noise_proj": [], "snr_proj": [],
        "t_peak_ns": [], "area_proj": [], "tau_eff_ns": [],
    }

    # Signs for single-channel pulse heights are derived from the average waveform at the vector peak.
    sign0 = 1.0 if fp.mean0[fp.peak_idx] >= 0 else -1.0
    sign1 = 1.0 if fp.mean1[fp.peak_idx] >= 0 else -1.0

    for path in paths:
        wf = load_waveform_npz(path)
        if wf is None:
            continue
        ch0 = wf.ch0[:, :n_samples]
        ch1 = wf.ch1[:, :n_samples]
        if ch0.shape[1] < n_samples or ch1.shape[1] < n_samples:
            continue

        base0 = np.median(ch0[:, base_sl], axis=1)
        base1 = np.median(ch1[:, base_sl], axis=1)
        y0 = ch0 - base0[:, None]
        y1 = ch1 - base1[:, None]
        proj = y0 * fp.response_u0 + y1 * fp.response_u1

        local = proj[:, event_search]
        local_arg = np.argmax(local, axis=1)
        amp_proj = local[np.arange(local.shape[0]), local_arg]
        event_peak_idx = event_search.start + local_arg

        peak0 = np.empty(ch0.shape[0], dtype=np.float64)
        peak1 = np.empty(ch1.shape[0], dtype=np.float64)
        amp_ch0 = np.empty(ch0.shape[0], dtype=np.float64)
        amp_ch1 = np.empty(ch1.shape[0], dtype=np.float64)
        for i, pi in enumerate(event_peak_idx):
            a = max(0, pi - PEAK_AVG_HALF_WIDTH_SAMPLES)
            b = min(n_samples, pi + PEAK_AVG_HALF_WIDTH_SAMPLES + 1)
            peak0[i] = np.mean(ch0[i, a:b])
            peak1[i] = np.mean(ch1[i, a:b])
            amp_ch0[i] = sign0 * np.mean(y0[i, a:b])
            amp_ch1[i] = sign1 * np.mean(y1[i, a:b])

        noise = np.std(proj[:, base_sl], axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = amp_proj / noise

        # Positive clipped projected area and effective time constant area/height.
        proj_int = np.clip(proj[:, integral_sl], 0.0, None)
        area = np.trapezoid(proj_int, x=fp.time_ns[integral_sl], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_eff = area / amp_proj

        rows["base0"].extend(base0.tolist())
        rows["base1"].extend(base1.tolist())
        rows["peak0"].extend(peak0.tolist())
        rows["peak1"].extend(peak1.tolist())
        rows["d0"].extend((peak0 - base0).tolist())
        rows["d1"].extend((peak1 - base1).tolist())
        rows["amp_proj"].extend(amp_proj.tolist())
        rows["amp_ch0_signed"].extend(amp_ch0.tolist())
        rows["amp_ch1_signed"].extend(amp_ch1.tolist())
        rows["noise_proj"].extend(noise.tolist())
        rows["snr_proj"].extend(snr.tolist())
        rows["t_peak_ns"].extend(fp.time_ns[event_peak_idx].tolist())
        rows["area_proj"].extend(area.tolist())
        rows["tau_eff_ns"].extend(tau_eff.tolist())

    out: dict[str, np.ndarray | float | int] = {}
    for k, v in rows.items():
        out[k] = np.asarray(v, dtype=np.float64)
    out["n_events"] = len(rows["amp_proj"])
    return out


def finite_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "sem": np.nan, "n": 0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
        "sem": float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size >= 2 else 0.0,
        "n": int(x.size),
    }


def downsample_xy(x: np.ndarray, ys: list[np.ndarray], max_points: int) -> tuple[np.ndarray, list[np.ndarray]]:
    n = len(x)
    if n <= max_points:
        return x, ys
    step = int(math.ceil(n / max_points))
    return x[::step], [y[::step] for y in ys]


def save_run_event_csv(run_dir: Path, run_name: str, metrics: dict[str, np.ndarray | float | int]) -> Path:
    path = run_dir / f"{sanitize_name(run_name)}_event_metrics.csv"
    keys = [
        "base0", "base1", "peak0", "peak1", "d0", "d1",
        "amp_proj", "amp_ch0_signed", "amp_ch1_signed",
        "noise_proj", "snr_proj", "t_peak_ns", "area_proj", "tau_eff_ns",
    ]
    n = int(metrics.get("n_events", 0))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([float(np.asarray(metrics[k])[i]) for k in keys])
    return path


def plot_run(run_dir: Path, run_name: str, fp: FirstPassResult, metrics: dict[str, np.ndarray | float | int]) -> dict[str, Path]:
    ensure_dir(run_dir)
    label = str(RUN_INFO.get(run_name, {}).get("label", run_name))
    paths: dict[str, Path] = {}

    # 1. Mean ch0/ch1 and projected waveform.
    proj_mean = fp.mean0 * fp.response_u0 + fp.mean1 * fp.response_u1
    x, [m0, m1, mp] = downsample_xy(fp.time_ns, [fp.mean0, fp.mean1, proj_mean], MAX_WAVEFORM_PLOT_POINTS)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, m0, label="mean ch0 - pedestal")
    ax.plot(x, m1, label="mean ch1 - pedestal")
    ax.plot(x, mp, label="projected mean response", linewidth=2.0)
    ax.axvline(fp.time_ns[fp.peak_idx], linestyle="--", linewidth=1.0, label="mean vector peak")
    ax.set_title(f"Mean waveform: {label}")
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("amplitude [raw unit]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = run_dir / f"{sanitize_name(run_name)}_mean_waveform.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths["mean_waveform"] = out

    # 2. Projected amplitude histogram.
    amp = np.asarray(metrics["amp_proj"])
    snr = np.asarray(metrics["snr_proj"])
    tau = np.asarray(metrics["tau_eff_ns"])
    fig, ax = plt.subplots(figsize=(8, 5))
    amp_f = amp[np.isfinite(amp)]
    if amp_f.size > 0:
        bins = min(120, max(20, int(np.sqrt(amp_f.size))))
        ax.hist(amp_f, bins=bins, histtype="step")
    st = finite_stats(amp)
    ax.set_title(f"Projected pulse-height histogram: {label}")
    ax.set_xlabel("projected pulse height [raw unit]")
    ax.set_ylabel("events")
    ax.text(
        0.98, 0.95,
        f"N = {st['n']}\nmedian = {st['median']:.4g}\nmean = {st['mean']:.4g}\nSEM = {st['sem']:.3g}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = run_dir / f"{sanitize_name(run_name)}_amp_hist.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths["amp_hist"] = out

    # 3. IQ scatter: baseline and peak, plus response direction.
    base0 = np.asarray(metrics["base0"])
    base1 = np.asarray(metrics["base1"])
    peak0 = np.asarray(metrics["peak0"])
    peak1 = np.asarray(metrics["peak1"])
    n = len(base0)
    if n > 0:
        rng = np.random.default_rng(20260521)
        if n > MAX_SCATTER_POINTS:
            idx = rng.choice(n, size=MAX_SCATTER_POINTS, replace=False)
        else:
            idx = np.arange(n)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(base0[idx], base1[idx], s=5, alpha=0.35, label="baseline")
        ax.scatter(peak0[idx], peak1[idx], s=5, alpha=0.35, label="peak")
        b0_med = np.nanmedian(base0)
        b1_med = np.nanmedian(base1)
        p0_med = np.nanmedian(peak0)
        p1_med = np.nanmedian(peak1)
        ax.plot([b0_med, p0_med], [b1_med, p1_med], marker="o", linewidth=2.0, label="median response")
        ax.set_title(f"IQ baseline/peak scatter: {label}")
        ax.set_xlabel("ch0")
        ax.set_ylabel("ch1")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = run_dir / f"{sanitize_name(run_name)}_iq_scatter.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        paths["iq_scatter"] = out

    # 4. SNR and tau_eff histograms.
    fig, ax = plt.subplots(figsize=(8, 5))
    snr_f = snr[np.isfinite(snr)]
    if snr_f.size > 0:
        lo, hi = np.nanpercentile(snr_f, [1, 99]) if snr_f.size > 20 else (np.nanmin(snr_f), np.nanmax(snr_f))
        snr_clip = snr_f[(snr_f >= lo) & (snr_f <= hi)]
        ax.hist(snr_clip, bins=min(100, max(20, int(np.sqrt(snr_clip.size)))), histtype="step")
    st_snr = finite_stats(snr)
    ax.set_title(f"Projected SNR histogram: {label}")
    ax.set_xlabel("projected pulse height / baseline RMS")
    ax.set_ylabel("events")
    ax.text(
        0.98, 0.95,
        f"median = {st_snr['median']:.4g}\nmean = {st_snr['mean']:.4g}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = run_dir / f"{sanitize_name(run_name)}_snr_hist.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths["snr_hist"] = out

    fig, ax = plt.subplots(figsize=(8, 5))
    tau_f = tau[np.isfinite(tau)]
    if tau_f.size > 0:
        lo, hi = np.nanpercentile(tau_f, [1, 99]) if tau_f.size > 20 else (np.nanmin(tau_f), np.nanmax(tau_f))
        tau_clip = tau_f[(tau_f >= lo) & (tau_f <= hi)]
        ax.hist(tau_clip, bins=min(100, max(20, int(np.sqrt(tau_clip.size)))), histtype="step")
    st_tau = finite_stats(tau)
    ax.set_title(f"Projected tau_eff histogram: {label}")
    ax.set_xlabel("tau_eff = positive area / height [ns]")
    ax.set_ylabel("events")
    ax.text(
        0.98, 0.95,
        f"median = {st_tau['median']:.4g} ns\nmean = {st_tau['mean']:.4g} ns",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = run_dir / f"{sanitize_name(run_name)}_tau_eff_hist.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    paths["tau_eff_hist"] = out

    return paths


def analyze_waveform_run(run_name: str, paths: list[Path], out_root: Path) -> dict[str, object] | None:
    print(f"\n================================================================================")
    print(f"RUN: {run_name}")
    print(f"files: {len(paths)}")
    print(f"================================================================================")
    for p in paths[:5]:
        print(f"  {p.name}: keys={npz_keys(p)}")
    if len(paths) > 5:
        print(f"  ... {len(paths) - 5} more files")

    fp = first_pass(paths)
    if fp is None:
        print(f"[skip] no compatible waveform ch0/ch1 arrays found in {run_name}")
        return None

    print(f"[first pass] events={fp.n_events}, samples={fp.n_samples}, fs={fp.sample_rate_hz:.6g} Hz")
    print(
        f"[response] peak={fp.time_ns[fp.peak_idx]:.3f} ns, "
        f"u=({fp.response_u0:.4g}, {fp.response_u1:.4g}), "
        f"norm={fp.response_norm:.4g}"
    )

    metrics = second_pass(paths, fp)
    run_dir = out_root / "runs" / sanitize_name(run_name)
    ensure_dir(run_dir)
    event_csv = save_run_event_csv(run_dir, run_name, metrics)
    plot_paths = plot_run(run_dir, run_name, fp, metrics)

    amp = finite_stats(np.asarray(metrics["amp_proj"]))
    snr = finite_stats(np.asarray(metrics["snr_proj"]))
    tau = finite_stats(np.asarray(metrics["tau_eff_ns"]))
    base0 = finite_stats(np.asarray(metrics["base0"]))
    base1 = finite_stats(np.asarray(metrics["base1"]))
    d0 = finite_stats(np.asarray(metrics["d0"]))
    d1 = finite_stats(np.asarray(metrics["d1"]))

    info = RUN_INFO.get(run_name, {})
    summary: dict[str, object] = {
        "run": run_name,
        "label": info.get("label", run_name),
        "group": info.get("group", "unknown"),
        "freq_ghz": info.get("freq_ghz"),
        "rf_dbm": info.get("rf_dbm"),
        "trigger_mv": info.get("trigger_mv"),
        "temperature_k": info.get("temperature_k"),
        "note": info.get("note", ""),
        "n_files": len(paths),
        "n_events": int(metrics["n_events"]),
        "n_samples": fp.n_samples,
        "sample_rate_hz": fp.sample_rate_hz,
        "mean_peak_time_ns": float(fp.time_ns[fp.peak_idx]),
        "response_u0": fp.response_u0,
        "response_u1": fp.response_u1,
        "response_norm_mean": fp.response_norm,
        "base0_median": base0["median"],
        "base1_median": base1["median"],
        "d0_median": d0["median"],
        "d1_median": d1["median"],
        "amp_proj_median": amp["median"],
        "amp_proj_mean": amp["mean"],
        "amp_proj_sem": amp["sem"],
        "snr_proj_median": snr["median"],
        "tau_eff_ns_median": tau["median"],
        "tau_eff_ns_sem": tau["sem"],
        "event_csv": str(event_csv),
        "plot_mean_waveform": str(plot_paths.get("mean_waveform", "")),
        "plot_amp_hist": str(plot_paths.get("amp_hist", "")),
        "plot_iq_scatter": str(plot_paths.get("iq_scatter", "")),
    }
    print(
        f"[summary] amp_med={summary['amp_proj_median']:.5g}, "
        f"SNR_med={summary['snr_proj_median']:.5g}, "
        f"tau_eff_med={summary['tau_eff_ns_median']:.5g} ns"
    )
    return summary


def save_summary_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    keys = [
        "run", "label", "group", "freq_ghz", "rf_dbm", "trigger_mv", "temperature_k", "note",
        "n_files", "n_events", "n_samples", "sample_rate_hz", "mean_peak_time_ns",
        "response_u0", "response_u1", "response_norm_mean",
        "base0_median", "base1_median", "d0_median", "d1_median",
        "amp_proj_median", "amp_proj_mean", "amp_proj_sem", "snr_proj_median",
        "tau_eff_ns_median", "tau_eff_ns_sem",
        "event_csv", "plot_mean_waveform", "plot_amp_hist", "plot_iq_scatter",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def plot_overall_summary(rows: list[dict[str, object]], out_root: Path) -> None:
    if not rows:
        return

    # Sort by run name/time.
    rows = sorted(rows, key=lambda r: str(r.get("run", "")))
    runs = [str(r["run"]) for r in rows]
    labels = [str(r.get("label", r["run"])) for r in rows]
    x = np.arange(len(rows))

    def arr(key: str) -> np.ndarray:
        return np.asarray([safe_float(r.get(key)) if safe_float(r.get(key)) is not None else np.nan for r in rows], dtype=float)

    amp = arr("amp_proj_median")
    amp_sem = arr("amp_proj_sem")
    snr = arr("snr_proj_median")
    tau = arr("tau_eff_ns_median")
    freq = arr("freq_ghz")

    # 1. Run summary: projected amplitude, SNR, tau_eff.
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(rows)), 5))
    ax.errorbar(x, amp, yerr=amp_sem, fmt="o", capsize=3, label="projected pulse height median ± SEM")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.set_ylabel("projected pulse height [raw unit]")
    ax.set_title("20260521/0522 Americium response: projected pulse height")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "summary_projected_amplitude_by_run.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(rows)), 5))
    ax.plot(x, snr, marker="o", linestyle="none", label="median projected SNR")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.set_ylabel("median SNR")
    ax.set_title("20260521/0522 Americium response: SNR by run")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "summary_snr_by_run.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(rows)), 5))
    ax.plot(x, tau, marker="o", linestyle="none", label="median tau_eff")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.set_ylabel("tau_eff [ns]")
    ax.set_title("20260521/0522 Americium response: effective decay width by run")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "summary_tau_eff_by_run.png", dpi=220)
    plt.close(fig)

    # 2. Frequency dependence for runs with known frequency.
    m = np.isfinite(freq) & np.isfinite(amp)
    if np.count_nonzero(m) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i in np.where(m)[0]:
            ax.errorbar(freq[i], amp[i], yerr=amp_sem[i] if np.isfinite(amp_sem[i]) else None, fmt="o", capsize=3)
            ax.annotate(runs[i], (freq[i], amp[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("RF frequency [GHz]")
        ax.set_ylabel("projected pulse height [raw unit]")
        ax.set_title("Pulse height vs RF frequency")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_root / "summary_amplitude_vs_rf_frequency.png", dpi=220)
        plt.close(fig)

    # 3. Median baseline/response vectors in IQ plane.
    base0 = arr("base0_median")
    base1 = arr("base1_median")
    d0 = arr("d0_median")
    d1 = arr("d1_median")
    m = np.isfinite(base0) & np.isfinite(base1) & np.isfinite(d0) & np.isfinite(d1)
    if np.count_nonzero(m) > 0:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(base0[m], base1[m], s=35, label="median baseline")
        for i in np.where(m)[0]:
            ax.arrow(base0[i], base1[i], d0[i], d1[i], length_includes_head=True, head_width=0.03 * np.nanmax(np.abs(d0[m] + 1e-30)), alpha=0.8)
            ax.annotate(runs[i], (base0[i], base1[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("ch0")
        ax.set_ylabel("ch1")
        ax.set_title("Median baseline and alpha response vector in IQ plane")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_root / "summary_iq_baseline_response_vectors.png", dpi=220)
        plt.close(fig)

    # 4. Save a human-readable labels table as JSON too.
    with (out_root / "summary_labels.json").open("w") as f:
        json.dump({r["run"]: {"label": r.get("label"), "note": r.get("note")} for r in rows}, f, ensure_ascii=False, indent=2)


def analyze_iq_scan_files(root: Path, files: list[Path], out_root: Path) -> list[Path]:
    iq_files = [p for p in files if has_iqscan(p)]
    if not iq_files:
        print("\n[iq_scan] no dd-based iq_scan npz files found")
        return []

    out_dir = out_root / "iq_scan"
    ensure_dir(out_dir)
    saved: list[Path] = []

    csv_path = out_dir / "iq_scan_summary.csv"
    with csv_path.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "file", "n_points", "freq_min", "freq_max",
            "ch0_first", "ch1_first", "ch0_last", "ch1_last",
        ])

        for path in iq_files:
            with np.load(path, allow_pickle=False) as z:
                dd = np.asarray(z["dd"], dtype=np.float64)
            freq = dd[:, 0]
            ch0 = dd[:, 1]
            ch1 = dd[:, 2]
            rel_name = str(path.relative_to(root))
            name = sanitize_name(path.stem)

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(ch0, ch1, c=freq, s=20)
            ax.plot(ch0, ch1, linewidth=0.8, alpha=0.6)
            ax.scatter(ch0[0], ch1[0], marker="o", s=70, label=f"start {freq[0]:.9g}")
            ax.scatter(ch0[-1], ch1[-1], marker="s", s=70, label=f"end {freq[-1]:.9g}")

            # Annotate nearest points to reference frequencies.
            for ref_ghz in IQSCAN_REFERENCE_FREQ_GHZ:
                ref_hz = ref_ghz * 1e9
                # If frequency axis looks already GHz, use GHz directly.
                target = ref_ghz if np.nanmedian(np.abs(freq)) < 1e6 else ref_hz
                idx = int(np.nanargmin(np.abs(freq - target)))
                ax.scatter(ch0[idx], ch1[idx], marker="x", s=80)
                ax.annotate(
                    f"near {ref_ghz:.3f}GHz\n{freq[idx]:.9g}",
                    (ch0[idx], ch1[idx]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7,
                )

            ax.set_xlabel("ch0")
            ax.set_ylabel("ch1")
            ax.set_title(f"IQ scan: {rel_name}")
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("frequency")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
            fig.tight_layout()
            out = out_dir / f"{name}_iq_scan.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            plt.close(fig)
            saved.append(out)

            w.writerow([
                rel_name, len(freq), float(np.nanmin(freq)), float(np.nanmax(freq)),
                float(ch0[0]), float(ch1[0]), float(ch0[-1]), float(ch1[-1]),
            ])

    saved.append(csv_path)
    print(f"\n[iq_scan] analyzed {len(iq_files)} file(s). Output: {out_dir}")
    return saved


def write_readme(out_root: Path, rows: list[dict[str, object]]) -> None:
    text = """# 20260521 Americium KID analysis

This directory was generated by `analyze_20260521_americium.py`.

## Main output files

- `run_summary.csv`  
  One row per run. Compare projected pulse height, SNR, tau_eff, baseline, and response vector.

- `summary_projected_amplitude_by_run.png`  
  Projected alpha-response pulse height for each run.

- `summary_snr_by_run.png`  
  Median SNR for each run.

- `summary_tau_eff_by_run.png`  
  Effective pulse width tau_eff = positive projected area / pulse height.

- `summary_amplitude_vs_rf_frequency.png`  
  Pulse height versus RF frequency for runs with metadata.

- `summary_iq_baseline_response_vectors.png`  
  Median pedestal point and alpha-response vector in the ch0-ch1 plane.

- `runs/<run_name>/...`  
  Per-run mean waveform, pulse-height histogram, SNR histogram, tau_eff histogram, IQ scatter, and event-level CSV.

- `iq_scan/...`  
  IQ scan plots for any npz file with `dd[:,0] = frequency`, `dd[:,1] = ch0`, `dd[:,2] = ch1`.

## Interpretation guide

The safest comparison quantity is usually `amp_proj_median`, not raw ch0 or ch1 alone.  
For each run, the script finds the mean response vector in the IQ plane and projects every event onto that direction.  
This removes the arbitrary ch0/ch1 rotation caused by mixer/pipe phase settings.

Use 50 ohm and off-resonance runs as controls.  
If they show a comparable projected pulse-height distribution or SNR to the on-resonance runs, the triggers are likely dominated by pickup/noise rather than KID resonator response.

The log says the true resonance was later identified as 5.485 GHz, so the 5.473/5.476 GHz runs should be treated as on-resonance candidates, not guaranteed true on-resonance data.
"""
    with (out_root / "README.md").open("w") as f:
        f.write(text)

    if rows:
        # Also write a compact ranking by median projected amplitude.
        valid = []
        for r in rows:
            amp = safe_float(r.get("amp_proj_median"))
            snr = safe_float(r.get("snr_proj_median"))
            if amp is not None and np.isfinite(amp):
                valid.append((amp, snr, r))
        valid.sort(reverse=True, key=lambda t: t[0])
        with (out_root / "quick_ranking_by_projected_amplitude.txt").open("w") as f:
            f.write("Projected pulse-height ranking\n")
            f.write("================================\n")
            for amp, snr, r in valid:
                f.write(f"{r['run']}: amp_med={amp:.6g}, SNR_med={snr:.6g} | {r.get('label', '')}\n")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print(f"[input]  {INPUT_DIR}")
    print(f"[output] {OUTPUT_DIR}")

    files = discover_npz_files(INPUT_DIR)
    print(f"[discover] npz files: {len(files)}")
    if len(files) == 0:
        print("No .npz files were found. Check INPUT_DIR.")
        return

    # Save raw file list and keys for debugging.
    with (OUTPUT_DIR / "npz_file_inventory.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "kind", "keys"])
        for p in files:
            kind = "iq_scan_dd" if has_iqscan(p) else "waveform_or_unknown"
            w.writerow([str(p.relative_to(INPUT_DIR)), kind, ";".join(npz_keys(p))])

    analyze_iq_scan_files(INPUT_DIR, files, OUTPUT_DIR)

    groups = group_waveform_files(INPUT_DIR, files)
    print(f"[discover] waveform run groups: {len(groups)}")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} npz file(s)")

    rows: list[dict[str, object]] = []
    for run_name, paths in groups.items():
        summary = analyze_waveform_run(run_name, paths, OUTPUT_DIR)
        if summary is not None:
            rows.append(summary)

    save_summary_csv(rows, OUTPUT_DIR / "run_summary.csv")
    plot_overall_summary(rows, OUTPUT_DIR)
    write_readme(OUTPUT_DIR, rows)

    print("\nDONE")
    print(f"Summary CSV: {OUTPUT_DIR / 'run_summary.csv'}")
    print(f"Plots      : {OUTPUT_DIR}")
    print("Start by opening:")
    print(f"  {OUTPUT_DIR / 'summary_projected_amplitude_by_run.png'}")
    print(f"  {OUTPUT_DIR / 'summary_iq_baseline_response_vectors.png'}")
    print(f"  {OUTPUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
