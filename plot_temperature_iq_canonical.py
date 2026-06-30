#!/usr/bin/env python3
"""
plot_temperature_iq_canonical.py

20260527 の 5.467 GHz 付近で取得した waveform data について、

  50 Hz laser / 1 Hz temperature cycle
      -> 1 周期 50 event を 10 phase-bin (各5 event) にまとめる
      -> 各 bin の pedestal IQ の median と IQR を描く

さらに、同じ ADC / mixer / phase-shifter 設定で取得した raw IQ S21 sweep を使って

  raw IQ
    -> cable-delay correction (optional)
    -> circle center subtraction
    -> circle-radius normalization
    -> resonance point is set to (1-d, 0)
    -> laser heating direction is set to +Q

という canonical S21 座標への補正を行う。

重要:
  * 「共振点を (1-d, 0)」へ置くには、waveform だけでは不十分。
    同じ readout chain で測定した raw IQ S21 sweep が必要。
  * SWEEP_FILE は waveform と同じ ADC ch0/ch1 座標で取得した sweep を指定する。
    VNA の normalized S21 だけでは絶対的な ADC IQ 座標への変換はできない。
  * sweep の cable delay が残る場合は SWEEP_CABLE_DELAY_NS を設定する。

対応 sweep format:
  CSV:
    frequency column: freq_ghz / frequency_ghz / f_ghz / freq / frequency / f
    IQ columns: ch0,ch1  OR  I,Q  OR  real,imag  OR  re,im
  NPZ:
    same key candidates as above

出力:
  data/20260527/temperature_iq_canonical/
      temperature_phase_10bin_iq_canonical.png
      temperature_phase_10bin_summary.csv
      iq_normalization_transform.json
      run_info.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Main settings: usually only edit this block
# ============================================================

DATA_DATE = "20260527"

# "cloud" : OneDrive / CloudStorage side
# "local" : kidanalysis/data/20260527
# "both"  : search both, then use the first exact match only
# ============================================================
# Main settings
# ============================================================

INPUT_MODE = "cloud"

# データ取得時の readout tone が書かれた、実在するフォルダ名
TARGET_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"


# waveform を取った固定 readout tone
F_TONE_GHZ = 5.476

# S21 sweep 上で「共振点」とみなして左端 (1-d, 0) に送る周波数
FR_REFERENCE_GHZ = 5.467

TARGET_FREQ_GHZ = 5.467
TARGET_Z_MM = 7.5
TARGET_X_MM = 3.4

FREQ_ATOL_GHZ = 1e-3
POS_ATOL_MM = 1e-6

NPZ_PATTERN = "wf_*.npz"

# Use only the first N events from the first measurement.
# 1000 events -> 20 temperature cycles -> 100 events per 10-bin group.
N_EVENTS_TO_USE = 1000

# laser = 50 Hz, temperature modulation = 1 Hz
EVENTS_PER_TEMP_CYCLE = 50
N_TEMP_PHASE_BINS = 10
assert EVENTS_PER_TEMP_CYCLE % N_TEMP_PHASE_BINS == 0
EVENTS_PER_BIN_PER_CYCLE = EVENTS_PER_TEMP_CYCLE // N_TEMP_PHASE_BINS

# pedestal range [us]. None means all t < 0.
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# Used only to find the mean pulse vector that fixes the sign of Q.
# At resonance, a heating pulse is defined to point toward +Q.
AMP_WINDOW_US = (0.0, 1.5)

# ------------------------------------------------------------
# Raw IQ S21 sweep calibration
# ------------------------------------------------------------

# REQUIRED for canonical coordinate:
# Set this to an S21 sweep measured through the same ADC/mixer/phase-shifter
# chain as the waveform data.
#
# Example:
# SWEEP_FILE = (
#     Path.home()
#     / "Library/CloudStorage/OneDrive-TheUniversityofTokyo"
#     / "東京大学/4S/kidfit/20260527"
#     / "raw_iq_sweep_5p467GHz.csv"
# )
SWEEP_FILE = None

# If SWEEP_FILE is None:
#   The script makes a phase-aligned *relative IQ* plot only.
#   It does NOT claim that the resonance point is exactly (1-d, 0).
#
# If SWEEP_FILE is supplied:
#   The script circle-fits the raw sweep and makes canonical S21 coordinates.
REQUIRE_SWEEP_FOR_CANONICAL = False

# Reference resonance frequency of the raw sweep [GHz].
# It is used to choose the sweep point sent to (1-d, 0).
FR_REFERENCE_GHZ = 5.467

# Fixed readout tone of the waveform data [GHz].
F_TONE_GHZ = 5.467

# Optional cable delay correction.
# For f in GHz and tau in ns, exp(+2*pi*i*f*tau) is dimensionless.
# Leave None if the raw IQ sweep is already delay corrected or the sweep is
# sufficiently narrow that its residual delay rotation is negligible.
SWEEP_CABLE_DELAY_NS = None

# d = Ql / |Qc|, which sets the canonical circle:
#   center = 1 - d/2
#   radius = d/2
#   resonance point = 1-d
#
# Update these using the notch fit nearest the target condition if available.
QL_FOR_D = 452.6852
QC_FOR_D = 488.4716
D_NOTCH = QL_FOR_D / abs(QC_FOR_D)

# circle fit frequency range [GHz]; None uses all sweep points.
# Restrict this if the sweep contains other resonators / glitches.
SWEEP_FIT_RANGE_GHZ = None
# SWEEP_FIT_RANGE_GHZ = (5.44, 5.50)

DPI = 300


# ============================================================
# Paths: same policy as the existing kidanalysis scripts
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "temperature_iq_canonical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

local_data_dir = HERE / "data" / DATA_DATE

cloud_data_candidates = [
    Path("/Volumes/NO NAME/data/20260527"),
]

EXTRA_INPUT_ROOTS = [
    Path("/Volumes/NO NAME/data/20260527"),
]


# ============================================================
# Utilities
# ============================================================

def scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else x


def make_time_axis_s(npts, sample_rate_hz, ref_position_percent):
    return (
        np.arange(npts, dtype=float)
        - npts * ref_position_percent / 100.0
    ) / sample_rate_hz


def normalized_complex(z):
    mag = abs(z)
    return z / mag if mag > 0 else 1.0 + 0.0j


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def circular_unwrap_deg(theta_rad):
    return np.degrees(np.unwrap(np.asarray(theta_rad, dtype=float)))


# ============================================================
# Input discovery
# ============================================================

def collect_input_roots():
    candidates = []

    if INPUT_MODE in ["local", "both"]:
        candidates.append(("local", local_data_dir))

    if INPUT_MODE in ["cloud", "both"]:
        candidates.extend(("cloud", p) for p in cloud_data_candidates)

    candidates.extend(("extra", p) for p in EXTRA_INPUT_ROOTS)

    print()
    print("===== path check =====")
    print("HERE:", HERE)
    print("INPUT_MODE:", INPUT_MODE)

    roots = []
    seen = set()

    for kind, p in candidates:
        p = Path(p).expanduser().resolve(strict=False)
        exists = p.is_dir()
        print(f"[{kind}] {p}")
        print("   exists:", exists)

        if exists and p.as_posix() not in seen:
            roots.append(p)
            seen.add(p.as_posix())

    if not roots:
        raise RuntimeError("入力フォルダが見つかりません。")

    return roots


def parse_measurement_dir_name(name):
    """
    expected:
      5.467GHz_z=7.5mm_x=3.4mm
      5.467GHz_z=7.5mm_x=3.4mm_second
    """
    import re

    pat = re.compile(
        r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
        r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
        r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
        r"(?:_(?P<tag>.+))?$"
    )
    m = pat.match(name)
    if m is None:
        return None

    d = m.groupdict()
    return {
        "freq_ghz": float(d["freq"]),
        "z_mm": float(d["z"]),
        "x_mm": float(d["x"]),
        "tag": d["tag"] or "",
    }


def find_target_first_dir(roots):
    """
    Exact first data only:
      - if TARGET_DIR_NAME set: exact folder name match
      - otherwise: tag == "" and target f/z/x match
    """
    candidates = []

    for root in roots:
        if TARGET_DIR_NAME is not None:
            p = root / TARGET_DIR_NAME
            print("candidate:", p, "exists:", p.is_dir())
            if p.is_dir():
                candidates.append(p)
            continue

        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue

            info = parse_measurement_dir_name(p.name)
            if info is None:
                continue

            is_target = (
                info["tag"] == ""
                and np.isclose(info["freq_ghz"], TARGET_FREQ_GHZ, atol=FREQ_ATOL_GHZ, rtol=0)
                and np.isclose(info["z_mm"], TARGET_Z_MM, atol=POS_ATOL_MM, rtol=0)
                and np.isclose(info["x_mm"], TARGET_X_MM, atol=POS_ATOL_MM, rtol=0)
            )
            if is_target:
                candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "対象の first measurement directory が見つかりません。\n"
            f"TARGET_DIR_NAME={TARGET_DIR_NAME}\n"
            f"TARGET_FREQ_GHZ={TARGET_FREQ_GHZ}, "
            f"TARGET_Z_MM={TARGET_Z_MM}, TARGET_X_MM={TARGET_X_MM}\n"
            "フォルダ名または target settings を確認してください。"
        )

    # 同名フォルダが複数あっても、実際に waveform npz を持つ方を優先する
    candidates_with_npz = [
        p for p in candidates
        if any(p.glob(NPZ_PATTERN))
    ]

    if not candidates_with_npz:
        print("候補フォルダは見つかりましたが、どれにも", NPZ_PATTERN, "がありません。")
        for p in candidates:
            print("  candidate:", p)
        raise RuntimeError(
            f"対象フォルダは見つかりましたが、{NPZ_PATTERN} がありません。"
        )

    if len(candidates_with_npz) > 1:
        print("WARNING: waveform npz を持つ同名フォルダが複数あります。")
        print("         最初のものを使います。")

    chosen = candidates_with_npz[0]
    print("selected measurement dir:", chosen)
    return chosen


def load_waveforms(meas_dir):
    npz_files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    if not npz_files:
        raise RuntimeError(f"{meas_dir} に {NPZ_PATTERN} がありません。")

    ch0_blocks = []
    ch1_blocks = []
    used_files = []
    time_ref = None
    n_loaded = 0

    for f in npz_files:
        if n_loaded >= N_EVENTS_TO_USE:
            break

        try:
            data = np.load(f)
        except Exception as exc:
            print("skip load error:", f, exc)
            continue

        required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        missing = [k for k in required if k not in data.files]
        if missing:
            print("skip missing keys:", f, missing)
            continue

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)

        if ch0.ndim == 1:
            ch0 = ch0[None, :]
        if ch1.ndim == 1:
            ch1 = ch1[None, :]

        if ch0.shape != ch1.shape:
            print("skip shape mismatch:", f)
            continue

        npts = int(scalar(data["npts"]))
        sr = float(scalar(data["sample_rate"]))
        ref_pos = float(scalar(data["ref_position"]))

        if ch0.shape[1] != npts:
            print("skip npts mismatch:", f)
            continue

        time_s = make_time_axis_s(npts, sr, ref_pos)
        if time_ref is None:
            time_ref = time_s
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time-axis mismatch:", f)
            continue

        n_take = min(len(ch0), N_EVENTS_TO_USE - n_loaded)
        ch0_blocks.append(ch0[:n_take])
        ch1_blocks.append(ch1[:n_take])
        used_files.append(f)
        n_loaded += n_take
        print(f"load: {f.name} -> {n_take} events (total={n_loaded})")

    if time_ref is None or n_loaded == 0:
        raise RuntimeError("有効な waveform を読み込めませんでした。")

    if n_loaded != N_EVENTS_TO_USE:
        raise RuntimeError(
            f"N_EVENTS_TO_USE={N_EVENTS_TO_USE} に対して {n_loaded} events しか読めませんでした。"
        )

    return time_ref, np.vstack(ch0_blocks), np.vstack(ch1_blocks), used_files


# ============================================================
# Event pedestal / pulse reference
# ============================================================

def event_pedestal_and_pulse_reference(time_s, ch0, ch1):
    """
    Returns:
      z_ped_raw per event
      p_ref_raw: global mean pulse vector at global 2D peak
      t_peak_us
    """
    time_us = time_s * 1e6

    if BASELINE_WINDOW_US is None:
        baseline_mask = time_us < 0
    else:
        lo, hi = BASELINE_WINDOW_US
        baseline_mask = (time_us >= lo) & (time_us <= hi)

    amp_mask = (time_us >= AMP_WINDOW_US[0]) & (time_us <= AMP_WINDOW_US[1])

    if baseline_mask.sum() < 3:
        raise ValueError("baseline points are too few.")
    if amp_mask.sum() < 3:
        raise ValueError("AMP_WINDOW_US points are too few.")

    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)
    z_ped_raw = ped0 + 1j * ped1

    dch0 = ch0 - ped0[:, None]
    dch1 = ch1 - ped1[:, None]

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)
    mean_abs = np.hypot(mean0, mean1)

    candidates = np.where(amp_mask)[0]
    idx_peak = candidates[np.argmax(mean_abs[amp_mask])]
    p_ref_raw = mean0[idx_peak] + 1j * mean1[idx_peak]

    return {
        "z_ped_raw": z_ped_raw,
        "p_ref_raw": p_ref_raw,
        "t_peak_us": float(time_us[idx_peak]),
        "ped0_raw": ped0,
        "ped1_raw": ped1,
    }


# ============================================================
# Raw sweep loader / circle fit
# ============================================================

def find_named_item(names, candidates):
    lower_map = {str(name).lower(): name for name in names}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def convert_frequency_to_ghz(freq):
    freq = np.asarray(freq, dtype=float)
    median = np.nanmedian(np.abs(freq))
    if median > 1e7:  # Hz
        return freq / 1e9
    if median > 1e4:  # MHz
        return freq / 1e3
    return freq


def load_raw_iq_sweep(path):
    """
    Reads a raw IQ sweep, preferably from the same ADC coordinate system as waveform data.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SWEEP_FILE not found: {path}")

    freq_candidates = [
        "freq_ghz", "frequency_ghz", "f_ghz",
        "freq", "frequency", "f",
    ]
    re_candidates = [
        "ch0", "i", "real", "re", "s21_re", "s21_real", "I",
    ]
    im_candidates = [
        "ch1", "q", "imag", "im", "s21_im", "s21_imag", "Q",
    ]

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        fcol = find_named_item(df.columns, freq_candidates)
        rcol = find_named_item(df.columns, re_candidates)
        icol = find_named_item(df.columns, im_candidates)

        if fcol is None or rcol is None or icol is None:
            raise KeyError(
                f"CSV columns cannot be recognized.\n"
                f"found: {list(df.columns)}\n"
                "need frequency + two IQ columns."
            )

        f = df[fcol].to_numpy(dtype=float)
        z = df[rcol].to_numpy(dtype=float) + 1j * df[icol].to_numpy(dtype=float)

    elif suffix == ".npz":
        data = np.load(path)
        keys = list(data.files)

        fkey = find_named_item(keys, freq_candidates)
        rkey = find_named_item(keys, re_candidates)
        ikey = find_named_item(keys, im_candidates)

        if fkey is None or rkey is None or ikey is None:
            raise KeyError(
                f"NPZ keys cannot be recognized.\n"
                f"found: {keys}\n"
                "need frequency + two IQ arrays."
            )

        f = np.asarray(data[fkey], dtype=float).ravel()
        z = (
            np.asarray(data[rkey], dtype=float).ravel()
            + 1j * np.asarray(data[ikey], dtype=float).ravel()
        )
    else:
        raise ValueError("SWEEP_FILE supports only .csv or .npz in this script.")

    f_ghz = convert_frequency_to_ghz(f)

    good = np.isfinite(f_ghz) & np.isfinite(z.real) & np.isfinite(z.imag)
    f_ghz = f_ghz[good]
    z = z[good]

    order = np.argsort(f_ghz)
    return f_ghz[order], z[order]


def remove_cable_delay(z, f_ghz, tau_ns):
    """
    Correct e^{-2πifτ} by multiplying e^{+2πifτ}.
    """
    if tau_ns is None:
        return z
    return z * np.exp(+2j * np.pi * np.asarray(f_ghz) * float(tau_ns))


def algebraic_circle_fit(z):
    """
    Fit x^2+y^2 + A x + B y + C = 0.
    This is adequate after cable-delay correction over one resonator.
    """
    x = np.asarray(z.real, dtype=float)
    y = np.asarray(z.imag, dtype=float)

    mat = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    a, b, c = np.linalg.lstsq(mat, rhs, rcond=None)[0]

    center = -0.5 * a + 1j * (-0.5 * b)
    radius_sq = (a*a + b*b) / 4.0 - c
    if radius_sq <= 0:
        raise RuntimeError("circle fit failed: non-positive radius^2.")
    radius = float(np.sqrt(radius_sq))

    residual = np.abs(z - center) - radius
    return center, radius, residual


# ============================================================
# Canonical transform
# ============================================================

class IQTransform:
    """
    z_delay = z_raw * exp(+2π i f_tone tau)
    u        = (z_delay - center) / radius
    u        = orientation * u
    maybe conjugate u
    Scanon   = (1-d/2) + (d/2)*u

    At resonance:
      Scanon = 1-d + 0i
    At the opposite off-resonant point:
      Scanon = 1 + 0i
    """

    def __init__(
        self,
        center,
        radius,
        orientation,
        use_conjugate,
        d_notch,
        tau_ns,
    ):
        self.center = complex(center)
        self.radius = float(radius)
        self.orientation = complex(orientation)
        self.use_conjugate = bool(use_conjugate)
        self.d_notch = float(d_notch)
        self.tau_ns = tau_ns

    def delay_correct_fixed_tone(self, z):
        if self.tau_ns is None:
            return z
        return z * np.exp(+2j * np.pi * F_TONE_GHZ * float(self.tau_ns))

    def unit_circle(self, z_raw):
        z = self.delay_correct_fixed_tone(z_raw)
        u = self.orientation * (z - self.center) / self.radius
        if self.use_conjugate:
            u = np.conj(u)
        return u

    def canonical_s21(self, z_raw):
        u = self.unit_circle(z_raw)
        return (1.0 - self.d_notch / 2.0) + (self.d_notch / 2.0) * u


def build_canonical_transform(f_sweep, z_sweep_raw, p_ref_raw):
    """
    Fit raw sweep circle, send the reference resonance point to -1 on unit circle,
    then choose the reflection so the mean heating-pulse direction is +Q.
    """
    z_sweep = remove_cable_delay(z_sweep_raw, f_sweep, SWEEP_CABLE_DELAY_NS)

    mask = np.ones(len(f_sweep), dtype=bool)
    if SWEEP_FIT_RANGE_GHZ is not None:
        lo, hi = SWEEP_FIT_RANGE_GHZ
        mask = (f_sweep >= lo) & (f_sweep <= hi)

    if mask.sum() < 8:
        raise RuntimeError("circle-fit sweep points are too few.")

    center, radius, residual = algebraic_circle_fit(z_sweep[mask])

    i_res = int(np.argmin(np.abs(f_sweep - FR_REFERENCE_GHZ)))
    u_res = (z_sweep[i_res] - center) / radius

    # Send the resonance point to -1 (left side of the unit circle).
    orientation = -1.0 / normalized_complex(u_res)

    # Apply the same fixed-tone delay correction to the pulse vector.
    if SWEEP_CABLE_DELAY_NS is None:
        p_delay = p_ref_raw
    else:
        p_delay = p_ref_raw * np.exp(
            +2j * np.pi * F_TONE_GHZ * float(SWEEP_CABLE_DELAY_NS)
        )

    p_unit = orientation * p_delay / radius

    # The remaining ambiguity is complex conjugation.
    # Choose it such that a heating pulse (fr down) points to +Q at resonance.
    use_conjugate = p_unit.imag < 0
    if use_conjugate:
        p_unit = np.conj(p_unit)

    transform = IQTransform(
        center=center,
        radius=radius,
        orientation=orientation,
        use_conjugate=use_conjugate,
        d_notch=D_NOTCH,
        tau_ns=SWEEP_CABLE_DELAY_NS,
    )

    return transform, {
        "f_sweep_ghz": f_sweep,
        "z_sweep_raw": z_sweep_raw,
        "z_sweep_delay_corrected": z_sweep,
        "circle_fit_mask": mask,
        "circle_center": center,
        "circle_radius": radius,
        "circle_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "resonance_index": i_res,
        "resonance_frequency_ghz": float(f_sweep[i_res]),
        "p_unit_after_orientation": p_unit,
    }


def build_relative_phase_transform(z_ped_raw, p_ref_raw):
    """
    Fallback when no raw sweep is available.
    It only moves the median pedestal to zero and rotates the mean pulse direction
    to +Q. It does NOT create an absolute canonical S21 coordinate.
    """
    z_origin = np.median(z_ped_raw.real) + 1j * np.median(z_ped_raw.imag)
    rot = 1j / normalized_complex(p_ref_raw)

    def apply(z):
        return rot * (z - z_origin)

    return apply, {
        "relative_origin": z_origin,
        "relative_rotation": rot,
    }


# ============================================================
# 10-bin phase summary
# ============================================================

def make_phase_summary(z_raw, z_plot, n_events):
    """
    temperature phase group:
       event % 50 = 0..49
       temp_bin = floor((event % 50)/5) = 0..9
    """
    event_index = np.arange(n_events, dtype=int)
    laser_phase_50 = event_index % EVENTS_PER_TEMP_CYCLE
    temp_bin_10 = laser_phase_50 // EVENTS_PER_BIN_PER_CYCLE

    rows = []
    for b in range(N_TEMP_PHASE_BINS):
        idx = np.where(temp_bin_10 == b)[0]

        raw_re_med, raw_re_q25, raw_re_q75 = median_iqr(z_raw[idx].real)
        raw_im_med, raw_im_q25, raw_im_q75 = median_iqr(z_raw[idx].imag)

        re_med, re_q25, re_q75 = median_iqr(z_plot[idx].real)
        im_med, im_q25, im_q75 = median_iqr(z_plot[idx].imag)

        z_med = re_med + 1j * im_med
        rows.append({
            "temp_phase_bin_10": b,
            "laser_phase_start_50": int(b * EVENTS_PER_BIN_PER_CYCLE),
            "laser_phase_end_50": int((b + 1) * EVENTS_PER_BIN_PER_CYCLE - 1),
            "n_events": len(idx),

            "raw_re_median_V": raw_re_med,
            "raw_re_q25_V": raw_re_q25,
            "raw_re_q75_V": raw_re_q75,
            "raw_im_median_V": raw_im_med,
            "raw_im_q25_V": raw_im_q25,
            "raw_im_q75_V": raw_im_q75,

            "plot_I_median": re_med,
            "plot_I_q25": re_q25,
            "plot_I_q75": re_q75,
            "plot_Q_median": im_med,
            "plot_Q_q25": im_q25,
            "plot_Q_q75": im_q75,
            "plot_abs_median": abs(z_med),
        })

    df = pd.DataFrame(rows)

    # angle around canonical circle center, when canonical mode is used.
    circle_center_canon = 1.0 - D_NOTCH / 2.0
    df["circle_angle_deg"] = circular_unwrap_deg(
        np.angle(
            (df["plot_I_median"].to_numpy() - circle_center_canon)
            + 1j * df["plot_Q_median"].to_numpy()
        )
    )
    return df


# ============================================================
# Plot
# ============================================================

def draw_time_arrows(ax, x, y):
    for i in range(len(x)):
        j = (i + 1) % len(x)
        ax.annotate(
            "",
            xy=(x[j], y[j]),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.70),
        )


def plot_results(summary_df, mode, transform_info, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0), constrained_layout=True)

    bins = summary_df["temp_phase_bin_10"].to_numpy(dtype=int)

    # --------------------------------------------------------
    # raw 10-bin median IQ
    # --------------------------------------------------------
    ax = axes[0, 0]

    x = summary_df["raw_re_median_V"].to_numpy() * 1e3
    y = summary_df["raw_im_median_V"].to_numpy() * 1e3
    xlo = x - summary_df["raw_re_q25_V"].to_numpy() * 1e3
    xhi = summary_df["raw_re_q75_V"].to_numpy() * 1e3 - x
    ylo = y - summary_df["raw_im_q25_V"].to_numpy() * 1e3
    yhi = summary_df["raw_im_q75_V"].to_numpy() * 1e3 - y

    ax.errorbar(x, y, xerr=[xlo, xhi], yerr=[ylo, yhi], fmt="none", alpha=0.55)
    sc_raw = ax.scatter(x, y, c=bins, s=72, zorder=4)
    draw_time_arrows(ax, x, y)

    for i, b in enumerate(bins):
        ax.annotate(str(b), (x[i], y[i]), xytext=(5, 5), textcoords="offset points")

    fig.colorbar(sc_raw, ax=ax, label="10-bin temperature phase")
    ax.set_xlabel("raw ch0 pedestal [mV]")
    ax.set_ylabel("raw ch1 pedestal [mV]")
    ax.set_title("Raw pedestal IQ: 10-bin median with IQR")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # --------------------------------------------------------
    # canonical / relative normalized IQ
    # --------------------------------------------------------
    ax = axes[0, 1]

    xp = summary_df["plot_I_median"].to_numpy()
    yp = summary_df["plot_Q_median"].to_numpy()
    xplo = xp - summary_df["plot_I_q25"].to_numpy()
    xphi = summary_df["plot_I_q75"].to_numpy() - xp
    yplo = yp - summary_df["plot_Q_q25"].to_numpy()
    yphi = summary_df["plot_Q_q75"].to_numpy() - yp

    if mode == "canonical":
        theta = np.linspace(0.0, 2.0 * np.pi, 800)
        c = 1.0 - D_NOTCH / 2.0
        r = D_NOTCH / 2.0
        circle = c + r * np.exp(1j * theta)
        ax.plot(circle.real, circle.imag, lw=1.5, label="canonical notch circle")

        z_sweep_canon = transform_info["z_sweep_canonical"]
        ax.plot(
            z_sweep_canon.real,
            z_sweep_canon.imag,
            lw=1.0,
            alpha=0.55,
            label="raw IQ sweep after normalization",
        )

        ax.scatter(
            [1.0 - D_NOTCH, 1.0],
            [0.0, 0.0],
            marker="x",
            s=80,
            label=r"resonance $(1-d,0)$ / off-resonance $(1,0)$",
        )

        x_res = 1.0 - D_NOTCH
        arrow_len = 0.22 * D_NOTCH
        ax.annotate(
            r"heating / $f_r\downarrow$",
            xy=(x_res, arrow_len),
            xytext=(x_res, 0.0),
            arrowprops=dict(arrowstyle="->", lw=1.7),
            fontsize=9,
        )

        title = "Canonical S21 IQ: resonance is exactly (1-d, 0)"
        ax.set_xlabel(r"canonical $I=\mathrm{Re}(S_{21})$")
        ax.set_ylabel(r"canonical $Q=\mathrm{Im}(S_{21})$")
    else:
        ax.axhline(0.0, lw=1.0)
        ax.axvline(0.0, lw=1.0)
        title = "Relative phase-aligned IQ (no raw sweep calibration)"
        ax.set_xlabel("phase-aligned relative I")
        ax.set_ylabel("phase-aligned relative Q")

    ax.errorbar(xp, yp, xerr=[xplo, xphi], yerr=[yplo, yphi], fmt="none", alpha=0.55)
    sc = ax.scatter(xp, yp, c=bins, s=72, zorder=4, label="10-bin median pedestal")
    draw_time_arrows(ax, xp, yp)

    for i, b in enumerate(bins):
        ax.annotate(str(b), (xp[i], yp[i]), xytext=(5, 5), textcoords="offset points")

    fig.colorbar(sc, ax=ax, label="10-bin temperature phase")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")

    # --------------------------------------------------------
    # Canonical I/Q vs temperature phase
    # --------------------------------------------------------
    ax = axes[1, 0]
    ax.errorbar(
        bins,
        xp,
        yerr=[xplo, xphi],
        marker="o",
        capsize=3,
        label="I median ± IQR",
    )
    ax.errorbar(
        bins,
        yp,
        yerr=[yplo, yphi],
        marker="s",
        capsize=3,
        label="Q median ± IQR",
    )
    ax.set_xlabel("temperature phase bin (5 laser events / bin)")
    ax.set_ylabel("canonical coordinate" if mode == "canonical" else "relative coordinate")
    ax.set_title("Median IQ components across the 1 Hz cycle")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------------
    # phase angle / count
    # --------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(
        bins,
        summary_df["circle_angle_deg"],
        marker="o",
        label="angle around canonical-circle center",
    )
    ax.set_xlabel("temperature phase bin (5 laser events / bin)")
    ax.set_ylabel("circle angle [deg]")
    ax.grid(True)
    ax.set_title("Progress along the IQ trajectory")

    axr = ax.twinx()
    axr.bar(
        bins,
        summary_df["n_events"],
        width=0.55,
        alpha=0.25,
        label="events/bin",
    )
    axr.set_ylabel("events/bin")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    fig.suptitle(
        f"{DATA_DATE}, target f={TARGET_FREQ_GHZ:.3f} GHz, "
        f"z={TARGET_Z_MM:.1f} mm, x={TARGET_X_MM:.1f} mm\n"
        f"50 laser events / temperature cycle -> {N_TEMP_PHASE_BINS} median IQ bins",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("DATA_DATE:", DATA_DATE)
    print("target:", TARGET_DIR_NAME or (
        f"{TARGET_FREQ_GHZ:.3f}GHz, z={TARGET_Z_MM:.1f}mm, x={TARGET_X_MM:.1f}mm"
    ))
    print("N_EVENTS_TO_USE:", N_EVENTS_TO_USE)
    print("N_TEMP_PHASE_BINS:", N_TEMP_PHASE_BINS)
    print("D_NOTCH:", D_NOTCH)
    print("OUT_DIR:", OUT_DIR)

    roots = collect_input_roots()
    meas_dir = find_target_first_dir(roots)
    time_s, ch0, ch1, used_files = load_waveforms(meas_dir)

    info = event_pedestal_and_pulse_reference(time_s, ch0, ch1)
    z_ped_raw = info["z_ped_raw"]
    p_ref_raw = info["p_ref_raw"]

    print()
    print("global pulse reference:")
    print(f"  peak time = {info['t_peak_us']:.6f} us")
    print(f"  vector raw = {p_ref_raw.real*1e3:+.5f} + i {p_ref_raw.imag*1e3:+.5f} mV")

    transform_record = {}
    transform_info = {}

    if SWEEP_FILE is None:
        if REQUIRE_SWEEP_FOR_CANONICAL:
            raise RuntimeError(
                "SWEEP_FILE is required for canonical coordinates. "
                "Set it to a raw ADC IQ sweep obtained with the same readout chain."
            )

        mode = "relative"
        apply_relative, relative_info = build_relative_phase_transform(z_ped_raw, p_ref_raw)
        z_plot = apply_relative(z_ped_raw)

        transform_record = {
            "mode": "relative_phase_align_only",
            "warning": (
                "No raw IQ S21 sweep supplied. This plot is phase-aligned relative IQ only; "
                "it does not place resonance exactly at (1-d, 0)."
            ),
            "relative_origin_re_V": float(relative_info["relative_origin"].real),
            "relative_origin_im_V": float(relative_info["relative_origin"].imag),
            "relative_rotation_re": float(relative_info["relative_rotation"].real),
            "relative_rotation_im": float(relative_info["relative_rotation"].imag),
        }

        print()
        print("WARNING: SWEEP_FILE=None")
        print("  -> phase-aligned relative IQ is plotted.")
        print("  -> canonical S21 circle / exact (1-d,0) placement is NOT claimed.")

    else:
        mode = "canonical"
        f_sweep, z_sweep_raw = load_raw_iq_sweep(SWEEP_FILE)
        transform, cal = build_canonical_transform(f_sweep, z_sweep_raw, p_ref_raw)

        z_plot = transform.canonical_s21(z_ped_raw)
        z_sweep_canon = transform.canonical_s21(z_sweep_raw)

        transform_info = {
            "z_sweep_canonical": z_sweep_canon,
        }

        transform_record = {
            "mode": "canonical_s21",
            "sweep_file": str(Path(SWEEP_FILE).expanduser()),
            "delay_ns": SWEEP_CABLE_DELAY_NS,
            "circle_center_re_raw": float(cal["circle_center"].real),
            "circle_center_im_raw": float(cal["circle_center"].imag),
            "circle_radius_raw": float(cal["circle_radius"]),
            "circle_fit_residual_rms_raw": float(cal["circle_residual_rms"]),
            "resonance_frequency_used_ghz": float(cal["resonance_frequency_ghz"]),
            "orientation_re": float(transform.orientation.real),
            "orientation_im": float(transform.orientation.imag),
            "complex_conjugated": bool(transform.use_conjugate),
            "d_notch": float(D_NOTCH),
            "canonical_circle_center": float(1.0 - D_NOTCH / 2.0),
            "canonical_circle_radius": float(D_NOTCH / 2.0),
            "resonance_point": [float(1.0 - D_NOTCH), 0.0],
            "off_resonance_point": [1.0, 0.0],
            "reference_pulse_after_orientation_re": float(cal["p_unit_after_orientation"].real),
            "reference_pulse_after_orientation_im": float(cal["p_unit_after_orientation"].imag),
        }

        print()
        print("canonical calibration:")
        print("  circle center raw =", cal["circle_center"])
        print("  circle radius raw =", cal["circle_radius"])
        print("  resonance sweep point used =", cal["resonance_frequency_ghz"], "GHz")
        print("  conjugated =", transform.use_conjugate)
        print("  convention: heating / fr down is +Q at resonance")

    summary_df = make_phase_summary(
        z_raw=z_ped_raw,
        z_plot=z_plot,
        n_events=len(z_ped_raw),
    )
    summary_df["normalization_mode"] = mode

    png_path = OUT_DIR / "temperature_phase_10bin_iq_canonical.png"
    csv_path = OUT_DIR / "temperature_phase_10bin_summary.csv"
    transform_path = OUT_DIR / "iq_normalization_transform.json"
    info_path = OUT_DIR / "run_info.txt"

    plot_results(
        summary_df=summary_df,
        mode=mode,
        transform_info=transform_info,
        output_path=png_path,
    )

    summary_df.to_csv(csv_path, index=False)
    transform_path.write_text(
        json.dumps(transform_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    info_path.write_text(
        "\n".join([
            f"measurement_dir = {meas_dir}",
            f"N_EVENTS_TO_USE = {N_EVENTS_TO_USE}",
            f"EVENTS_PER_TEMP_CYCLE = {EVENTS_PER_TEMP_CYCLE}",
            f"N_TEMP_PHASE_BINS = {N_TEMP_PHASE_BINS}",
            f"events_per_bin_per_cycle = {EVENTS_PER_BIN_PER_CYCLE}",
            f"BASELINE_WINDOW_US = {BASELINE_WINDOW_US}",
            f"AMP_WINDOW_US = {AMP_WINDOW_US}",
            f"pulse_reference_peak_time_us = {info['t_peak_us']}",
            f"pulse_reference_raw = {p_ref_raw}",
            "",
            "used waveform files:",
            *[str(p) for p in used_files],
        ]),
        encoding="utf-8",
    )

    print()
    print("saved:")
    print(" ", png_path)
    print(" ", csv_path)
    print(" ", transform_path)
    print(" ", info_path)


if __name__ == "__main__":
    main()
