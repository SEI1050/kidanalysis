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
- 生データを IQ calibration で変換し、変換後データについて 01 / 05 相当の図を作る。
- calibration は iq_scan_f_reso_*.npz を直接読み、選択周波数と同じファイルだけを使う。
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


# ここで解析したい周波数を選ぶ。
# 例: 4.463, 5.161, 5.267
SELECT_FREQUENCY_GHZ = 5.267

# z = ch0 + i * Q_SIGN * ch1
Q_SIGN = +1

# tau fit に使う scan 両端の点数（指定どおり 5 点）
N_EDGE_POINTS = 5

# 円 fit に使う共振点近傍の片側点数
CIRCLE_HALF_WIDTH_POINTS = 3

# pedestal は各イベント先頭 10% の平均
PEDESTAL_FRACTION = 0.10

# calibration set と waveform 周波数の許容差
MAX_CALIBRATION_MATCH_HZ = 5.0e6

# IQ calibrationには、必ずこの名前形式の局所scanを使う。
IQ_RESONANCE_SCAN_GLOB = "iq_scan_f_reso_*.npz"

# 自動選択ではなくファイル名を明示したい場合に指定する。
# NoneならSELECT_FREQUENCY_GHZに最も近い
# iq_scan_f_reso_*.npzを自動選択する。
# 例:
# IQ_RESONANCE_SCAN_FILE_OVERRIDE = "iq_scan_f_reso_4.463GHz.npz"
IQ_RESONANCE_SCAN_FILE_OVERRIDE: str | None = None

# 図に描くイベント数。None なら全イベント
MAX_EVENTS_TO_PLOT: int | None = None

# trigger 系 track 専用設定
TRIGGER_TRACK_SAMPLE_STRIDE = 5
TRIGGER_TRACK_MEDIAN_CMAP = "plasma"

# pedestal 系 track / waveform 用サンプル stride
PEDESTAL_TRACK_SAMPLE_STRIDE = 5
WAVEFORM_SAMPLE_STRIDE = 5

# 図を作るときだけ、連続する何サンプルを平均して1 binにするか。
# 補正済みnpzはbin化せず、元のサンプル数のまま保存する。
# 1ならbin化なし。
PLOT_SAMPLE_BIN = 20

# フォルダ名が丸め値で、実際の読み出し周波数を手入力したい場合に使う。
# metadata内に読み出し周波数があれば、それを優先する。
# 例:
# READOUT_FREQUENCY_OVERRIDES_GHZ = {
#     "5.161GHz_trig_ch1_-1mV": 5.1617,
# }
READOUT_FREQUENCY_OVERRIDES_GHZ: dict[str, float] = {}

# Trueなら、各dataset出力先に残っている古いPNGを削除してから作る。
CLEAR_OLD_PNGS = True

PNG_DPI = 200
SAVE_CORRECTED_NPZ = True
SAVE_CALIBRATION_DIAGNOSTICS = True

OUTPUT_ROOT = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260714"
) / ( f"iqcal"
    f"_{SELECT_FREQUENCY_GHZ}"
    f"_bin{PLOT_SAMPLE_BIN}"
    f"_trackstride{TRIGGER_TRACK_SAMPLE_STRIDE}"
)
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

def _numeric_1d_array(value: np.ndarray) -> np.ndarray | None:
    """
    npz内の値を1次元数値配列として取り出す。
    scalar、object配列、空配列は除外する。
    """
    array = np.asarray(value)

    if array.dtype == object or array.size <= 1:
        return None

    array = np.squeeze(array)

    if array.ndim != 1:
        return None

    try:
        result = np.asarray(array, dtype=float)
    except (TypeError, ValueError):
        return None

    return result


def _complex_iq_array(value: np.ndarray) -> np.ndarray | None:
    """
    IQを1次元complex配列として取り出す。

    対応形式
    --------
    - complexの1次元配列
    - shape=(N, 2): [:,0]がI、[:,1]がQ
    - shape=(2, N): [0]がI、[1]がQ
    """
    array = np.asarray(value)

    if array.dtype == object or array.size <= 1:
        return None

    array = np.squeeze(array)

    if np.iscomplexobj(array) and array.ndim == 1:
        return np.asarray(array, dtype=complex)

    if array.ndim == 2:
        try:
            array_float = np.asarray(array, dtype=float)
        except (TypeError, ValueError):
            return None

        if array_float.shape[1] == 2:
            return (
                array_float[:, 0]
                + 1j * array_float[:, 1]
            )

        if array_float.shape[0] == 2:
            return (
                array_float[0]
                + 1j * array_float[1]
            )

    return None


def _find_frequency_vector(
    npz: np.lib.npyio.NpzFile,
    expected_length: int | None = None,
) -> tuple[str, np.ndarray]:
    """
    scan周波数のベクトルを探す。

    f_resoなどのscalarは使わず、必ず複数点の配列を選ぶ。
    """
    preferred_keys = (
        "frequency_hz",
        "frequencies_hz",
        "frequency",
        "frequencies",
        "freq_hz",
        "freq",
        "f_hz",
        "f",
        "x",
    )

    keys = list(npz.keys())
    ordered_keys: list[str] = []

    for candidate in preferred_keys:
        found = find_key_case_insensitive(
            keys,
            (candidate,),
        )
        if found is not None and found not in ordered_keys:
            ordered_keys.append(found)

    for key in keys:
        low = key.lower()
        if (
            key not in ordered_keys
            and any(
                token in low
                for token in (
                    "frequency",
                    "freq",
                )
            )
            and "reso" not in low
        ):
            ordered_keys.append(key)

    candidates: list[tuple[str, np.ndarray]] = []

    for key in ordered_keys:
        array = _numeric_1d_array(npz[key])

        if array is None:
            continue

        if (
            expected_length is not None
            and array.size != expected_length
        ):
            continue

        candidates.append((key, array))

    if not candidates:
        raise KeyError(
            "frequency vector was not found. "
            f"keys={keys}, expected_length={expected_length}"
        )

    # 周波数らしい範囲・変化量を優先する。
    def score(item: tuple[str, np.ndarray]) -> tuple[int, float]:
        _, array = item
        converted = convert_frequency_array_to_hz(array)
        typical = float(
            np.nanmedian(
                np.abs(converted)
            )
        )
        frequency_like = int(
            1.0e8 <= typical <= 1.0e11
        )
        span = float(
            np.nanmax(converted)
            - np.nanmin(converted)
        )
        return frequency_like, span

    return max(candidates, key=score)


def _find_iq_vectors(
    npz: np.lib.npyio.NpzFile,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """
    iq_scan_f_reso_*.npzからI/Qを取り出す。

    優先順位
    --------
    1. ch0/ch1, I/Qの別配列
    2. iq, s21, iq0等のcomplexまたは2列配列

    実験時表示で使われることの多い"iq"を"iq0"より優先する。
    """
    keys = list(npz.keys())

    i_key = find_key_case_insensitive(
        keys,
        (
            "ch0",
            "channel0",
            "channel_0",
            "i",
            "real",
            "re",
        ),
    )
    q_key = find_key_case_insensitive(
        keys,
        (
            "ch1",
            "channel1",
            "channel_1",
            "q",
            "imag",
            "im",
        ),
    )

    if i_key is not None and q_key is not None:
        ch0 = _numeric_1d_array(npz[i_key])
        ch1 = _numeric_1d_array(npz[q_key])

        if (
            ch0 is not None
            and ch1 is not None
            and ch0.size == ch1.size
        ):
            return i_key, q_key, ch0, ch1

    complex_key_order = (
        "iq",
        "s21",
        "complex_iq",
        "iq_complex",
        "s21_complex",
        "iq0",
    )

    ordered_keys: list[str] = []

    for candidate in complex_key_order:
        found = find_key_case_insensitive(
            keys,
            (candidate,),
        )
        if found is not None and found not in ordered_keys:
            ordered_keys.append(found)

    for key in keys:
        low = key.lower()
        if (
            key not in ordered_keys
            and any(
                token in low
                for token in (
                    "iq",
                    "s21",
                    "complex",
                )
            )
            and "ratio" not in low
        ):
            ordered_keys.append(key)

    for key in ordered_keys:
        z = _complex_iq_array(npz[key])

        if z is None:
            continue

        return (
            f"Re({key})",
            f"Im({key})",
            np.real(z),
            np.imag(z),
        )

    raise KeyError(
        "IQ arrays were not found in resonance scan. "
        f"keys={keys}"
    )


def _make_scan_monotonic(
    frequency_hz: np.ndarray,
    ch0: np.ndarray,
    ch1: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    局所scanの取得順を保ったまま、周波数を昇順にそろえる。

    単調増加ならそのまま。
    単調減少なら配列全体を反転する。
    上下に折り返すscanは黙ってsortせず、エラーにする。
    """
    difference = np.diff(frequency_hz)

    increasing = np.all(
        difference >= 0.0
    )
    decreasing = np.all(
        difference <= 0.0
    )

    if increasing:
        return frequency_hz, ch0, ch1

    if decreasing:
        return (
            frequency_hz[::-1],
            ch0[::-1],
            ch1[::-1],
        )

    raise ValueError(
        f"{path.name}: scan frequency is not monotonic. "
        "The code will not sort and mix separate sweeps automatically."
    )


def load_iq_scan(path: Path) -> IQScanData:
    """
    iq_scan_f_reso_*.npzを局所IQ scanとして直接読み込む。
    """
    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        keys = list(npz.keys())

        if "dd" in npz:
            dd = np.asarray(
                npz["dd"],
                dtype=float,
            )

            if (
                dd.ndim != 2
                or dd.shape[1] < 3
            ):
                raise ValueError(
                    f"{path.name}: dd shape must be "
                    f"(N,>=3), got {dd.shape}"
                )

            frequency = dd[:, 0]
            ch0 = dd[:, 1]
            ch1 = dd[:, 2]
            frequency_key = "dd[:,0]"
            i_key = "dd[:,1]"
            q_key = "dd[:,2]"

        else:
            (
                i_key,
                q_key,
                ch0,
                ch1,
            ) = _find_iq_vectors(npz)

            (
                frequency_key,
                frequency,
            ) = _find_frequency_vector(
                npz,
                expected_length=ch0.size,
            )

    frequency_hz = convert_frequency_array_to_hz(
        frequency
    )
    ch0 = np.asarray(
        ch0,
        dtype=float,
    ).reshape(-1)
    ch1 = np.asarray(
        ch1,
        dtype=float,
    ).reshape(-1)

    if not (
        frequency_hz.size
        == ch0.size
        == ch1.size
    ):
        raise ValueError(
            f"{path.name}: length mismatch: "
            f"frequency={frequency_hz.size}, "
            f"ch0={ch0.size}, ch1={ch1.size}"
        )

    finite = (
        np.isfinite(frequency_hz)
        & np.isfinite(ch0)
        & np.isfinite(ch1)
    )

    frequency_hz = frequency_hz[finite]
    ch0 = ch0[finite]
    ch1 = ch1[finite]

    (
        frequency_hz,
        ch0,
        ch1,
    ) = _make_scan_monotonic(
        frequency_hz,
        ch0,
        ch1,
        path,
    )

    if (
        frequency_hz.size
        < 2 * N_EDGE_POINTS + 3
    ):
        raise ValueError(
            f"{path.name}: too few scan points "
            f"({frequency_hz.size})"
        )

    print(
        "[resonance scan loaded] "
        f"file={path.name}, "
        f"keys=({frequency_key}, {i_key}, {q_key}), "
        f"points={frequency_hz.size}, "
        f"range={frequency_hz.min()/1e9:.9f}"
        f"–{frequency_hz.max()/1e9:.9f} GHz"
    )

    return IQScanData(
        path=path,
        frequency_hz=frequency_hz,
        ch0=ch0,
        ch1=ch1,
    )


def resonance_set_frequency_from_file(
    path: Path,
) -> float:
    """
    resonance scanが属する周波数setを取得する。

    ファイル内のf_reso等のscalarを優先し、
    無ければファイル名のGHz値を使う。
    """
    preferred_keys = (
        "f_reso",
        "f_resonance",
        "resonance_frequency",
        "resonance_frequency_hz",
        "fr",
        "fr_hz",
        "f0",
        "f0_hz",
    )

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as npz:
            key = find_key_case_insensitive(
                npz.keys(),
                preferred_keys,
            )

            if key is not None:
                scalar = scalar_from_array(
                    npz[key]
                )

                if scalar is not None:
                    return convert_frequency_scalar_to_hz(
                        scalar
                    )

    except Exception as exc:
        print(
            f"[resonance-frequency warning] "
            f"{path.name}: {exc}"
        )

    frequencies = parse_ghz_from_text(
        path.stem
    )

    if not frequencies:
        raise ValueError(
            f"{path.name}: frequency was not found "
            "in metadata or filename"
        )

    return float(frequencies[-1])


def discover_resonance_scan_paths(
    root: Path,
) -> list[Path]:
    paths = sorted(
        root.glob(
            IQ_RESONANCE_SCAN_GLOB
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No files matched "
            f"{root / IQ_RESONANCE_SCAN_GLOB}"
        )

    return paths


def select_resonance_scan_path(
    root: Path,
    selected_frequency_hz: float,
) -> tuple[Path, float]:
    """
    SELECT_FREQUENCY_GHZと同じsetの
    iq_scan_f_reso_*.npzを1つだけ選ぶ。
    """
    if (
        IQ_RESONANCE_SCAN_FILE_OVERRIDE
        is not None
    ):
        path = (
            root
            / IQ_RESONANCE_SCAN_FILE_OVERRIDE
        )

        if not path.exists():
            raise FileNotFoundError(
                f"IQ resonance scan override "
                f"does not exist: {path}"
            )

        set_frequency_hz = (
            resonance_set_frequency_from_file(
                path
            )
        )

    else:
        candidates: list[
            tuple[float, Path, float]
        ] = []

        for path in discover_resonance_scan_paths(
            root
        ):
            try:
                set_frequency_hz = (
                    resonance_set_frequency_from_file(
                        path
                    )
                )
            except Exception as exc:
                print(
                    f"[resonance scan skip] "
                    f"{path.name}: {exc}"
                )
                continue

            candidates.append(
                (
                    abs(
                        set_frequency_hz
                        - selected_frequency_hz
                    ),
                    path,
                    set_frequency_hz,
                )
            )

        if not candidates:
            raise FileNotFoundError(
                "No usable iq_scan_f_reso_*.npz "
                "was found"
            )

        (
            difference_hz,
            path,
            set_frequency_hz,
        ) = min(
            candidates,
            key=lambda item: item[0],
        )

        if (
            difference_hz
            > MAX_CALIBRATION_MATCH_HZ
        ):
            candidate_text = ", ".join(
                f"{item[1].name}: "
                f"{item[2]/1e9:.9f} GHz"
                for item in sorted(
                    candidates,
                    key=lambda item: item[2],
                )
            )

            raise ValueError(
                "No iq_scan_f_reso file matches "
                f"{selected_frequency_hz/1e9:.9f} GHz "
                f"within "
                f"{MAX_CALIBRATION_MATCH_HZ/1e6:.3f} MHz. "
                f"Candidates: {candidate_text}"
            )

    difference_hz = abs(
        set_frequency_hz
        - selected_frequency_hz
    )

    if (
        difference_hz
        > MAX_CALIBRATION_MATCH_HZ
    ):
        raise ValueError(
            "Selected resonance scan is not "
            "the same frequency set: "
            f"selected={selected_frequency_hz/1e9:.9f} GHz, "
            f"file={path.name}, "
            f"set={set_frequency_hz/1e9:.9f} GHz, "
            f"difference={difference_hz/1e6:.3f} MHz"
        )

    print(
        "[selected resonance scan] "
        f"{path.name}, "
        f"set frequency="
        f"{set_frequency_hz/1e9:.9f} GHz, "
        f"difference from selection="
        f"{difference_hz/1e3:.3f} kHz"
    )

    return path, float(
        set_frequency_hz
    )


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


def build_selected_calibration_set(
    root: Path,
    selected_frequency_hz: float,
) -> CalibrationSet:
    """
    選択周波数に対応するiq_scan_f_reso_*.npzを
    そのまま局所scanとして使用してcalibrationを作る。

    広帯域のiq_scan_*.npzは一切使用しない。
    周波数領域の自動切り出しもしない。
    """
    (
        scan_path,
        target_hz,
    ) = select_resonance_scan_path(
        root,
        selected_frequency_hz,
    )

    scan = load_iq_scan(
        scan_path
    )

    if not (
        scan.frequency_hz.min()
        <= target_hz
        <= scan.frequency_hz.max()
    ):
        nearest_hz = float(
            scan.frequency_hz[
                np.argmin(
                    np.abs(
                        scan.frequency_hz
                        - target_hz
                    )
                )
            ]
        )

        raise ValueError(
            f"{scan_path.name}: resonance set frequency "
            f"{target_hz/1e9:.9f} GHz is outside "
            f"the scan range "
            f"{scan.frequency_hz.min()/1e9:.9f}–"
            f"{scan.frequency_hz.max()/1e9:.9f} GHz. "
            f"Nearest scan point is "
            f"{nearest_hz/1e9:.9f} GHz."
        )

    frequency_hz = scan.frequency_hz
    ch0 = scan.ch0
    ch1 = scan.ch1
    z_raw = (
        ch0
        + 1j * Q_SIGN * ch1
    )

    tau_fit = fit_tau_from_phase(
        frequency_hz,
        z_raw,
        N_EDGE_POINTS,
    )

    z_tau = apply_tau_correction(
        z_raw,
        frequency_hz,
        tau_fit.tau_s,
    )

    circle_fit = fit_circle_near_resonance(
        frequency_hz,
        z_tau,
        target_hz,
    )

    geometry = compute_geometric_calibration(
        circle_fit
    )

    z_alpha = (
        z_tau
        * np.exp(
            -1j * geometry.alpha_rad
        )
    )
    z_norm = (
        z_alpha
        / geometry.amplitude_a
    )
    z_final = (
        1.0
        + (z_norm - 1.0)
        * np.exp(
            -1j * geometry.phi_rad
        )
    )

    final_radius = (
        circle_fit.radius
        / geometry.amplitude_a
    )

    calibration = CalibrationSet(
        name=(
            f"{target_hz/1e9:.6f}GHz"
        ),
        target_frequency_hz=float(
            target_hz
        ),
        scan_path=scan.path,
        frequency_hz=frequency_hz,
        z_scan_raw=z_raw,
        tau_fit=tau_fit,
        circle_fit=circle_fit,
        geometry=geometry,
        z_scan_final=z_final,
        final_circle_center=(
            geometry.center_final
        ),
        final_circle_radius=float(
            final_radius
        ),
    )

    print(
        "[calibration built from resonance scan] "
        f"file={scan.path.name}, "
        f"target={target_hz/1e9:.9f} GHz, "
        f"scan range="
        f"{frequency_hz.min()/1e9:.9f}–"
        f"{frequency_hz.max()/1e9:.9f} GHz, "
        f"tau={tau_fit.tau_s*1e9:.6f} ns, "
        f"a={geometry.amplitude_a:.6g}"
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



def bin_waveforms_for_plot(array: np.ndarray, bin_size: int) -> np.ndarray:
    """
    図を作るときだけサンプル方向をbin化する。

    連続するbin_sizeサンプルの平均を1点とする。
    補正済みnpzの保存データには、このbin化を適用しない。
    """
    array = np.asarray(array, dtype=float)

    if bin_size < 1:
        raise ValueError("PLOT_SAMPLE_BIN must be >= 1")

    if bin_size == 1:
        return array

    n_events, n_samples = array.shape
    n_usable = (n_samples // bin_size) * bin_size

    if n_usable == 0:
        raise ValueError(
            f"PLOT_SAMPLE_BIN={bin_size} is larger than n_samples={n_samples}"
        )

    return array[:, :n_usable].reshape(
        n_events,
        n_usable // bin_size,
        bin_size,
    ).mean(axis=2)


def readout_frequency_from_metadata(records: list[WaveformRecord]) -> float | None:
    """
    waveform npz のmetadataから実際の読み出し周波数を探す。

    genericな frequency はレーザー周波数等の可能性もあるため、
    最後にcalibration targetとの近さを確認してから使用する。
    """
    candidate_keys = (
        "readout_frequency_hz",
        "readout_frequency",
        "rf_frequency_hz",
        "rf_frequency",
        "tone_frequency_hz",
        "tone_frequency",
        "carrier_frequency_hz",
        "carrier_frequency",
        "frequency_hz",
        "freq_hz",
    )

    values_hz: list[float] = []

    for record in records:
        key = find_key_case_insensitive(
            record.metadata.keys(),
            candidate_keys,
        )
        if key is None:
            continue

        scalar = scalar_from_array(record.metadata[key])
        if scalar is None:
            continue

        frequency_hz = convert_frequency_scalar_to_hz(scalar)

        if 1.0e8 <= abs(frequency_hz) <= 1.0e11:
            values_hz.append(float(frequency_hz))

    if not values_hz:
        return None

    return float(np.median(values_hz))


def resolve_dataset_readout_frequency_hz(
    dataset: DatasetGroup,
    calibration: CalibrationSet,
) -> tuple[float, str]:
    """
    calibrationをwaveformへ適用するときの読み出し周波数を決める。

    優先順位
    --------
    1. READOUT_FREQUENCY_OVERRIDES_GHZ
    2. waveform npz metadata
    3. 同じsetとして対応付けた f_reso（calibration target）

    どの場合もcalibration targetからMAX_CALIBRATION_MATCH_HZ以内であることを確認する。
    """
    override_ghz = READOUT_FREQUENCY_OVERRIDES_GHZ.get(dataset.folder_path.name)

    if override_ghz is None:
        override_ghz = READOUT_FREQUENCY_OVERRIDES_GHZ.get(dataset.name)

    if override_ghz is not None:
        frequency_hz = float(override_ghz) * 1.0e9
        source = "manual override"
    else:
        metadata_frequency_hz = readout_frequency_from_metadata(dataset.records)

        if (
            metadata_frequency_hz is not None
            and abs(
                metadata_frequency_hz
                - calibration.target_frequency_hz
            )
            <= MAX_CALIBRATION_MATCH_HZ
        ):
            frequency_hz = metadata_frequency_hz
            source = "waveform metadata"
        else:
            # フォルダ名が 5.161 GHz、f_resoが 5.1617 GHz のように
            # 丸められている場合は、同じsetの正確な f_reso を使用する。
            frequency_hz = calibration.target_frequency_hz
            source = "matched iqscan/f_reso set"

    difference_hz = abs(
        frequency_hz
        - calibration.target_frequency_hz
    )

    if difference_hz > MAX_CALIBRATION_MATCH_HZ:
        raise ValueError(
            "waveform readout frequency and IQ calibration set do not match: "
            f"waveform={frequency_hz/1e9:.9f} GHz, "
            f"calibration={calibration.target_frequency_hz/1e9:.9f} GHz, "
            f"difference={difference_hz/1e6:.3f} MHz"
        )

    return float(frequency_hz), source


def clear_old_pngs(output_dir: Path) -> None:
    if not CLEAR_OLD_PNGS or not output_dir.exists():
        return

    for path in output_dir.glob("*.png"):
        path.unlink()
        print(f"[removed old] {path}")


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


def plot_trigger_track(
    ch0: np.ndarray,
    ch1: np.ndarray,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 7.6))

    event_indices = add_event_trajectories(
        ax,
        ch0,
        Q_SIGN * ch1,
        sample_stride=TRIGGER_TRACK_SAMPLE_STRIDE,
    )

    add_gradient_median_track(
        ax,
        ch0,
        Q_SIGN * ch1,
        sample_stride=TRIGGER_TRACK_SAMPLE_STRIDE,
        cmap=TRIGGER_TRACK_MEDIAN_CMAP,
    )

    pedestal = compute_median_pedestal_complex(
        ch0,
        ch1,
    )

    ax.scatter(
        [pedestal.real],
        [pedestal.imag],
        marker="x",
        s=85,
        linewidths=2.0,
        color="black",
        label="median pedestal (first 10%)",
        zorder=6,
    )

    # ============================================================
    # 01のtrack図だけ、描画した全グレーtrackが入る軸範囲にする
    # ============================================================
    sample_stride = max(
        1,
        TRIGGER_TRACK_SAMPLE_STRIDE,
    )

    x_plot = ch0[
        event_indices,
        ::sample_stride,
    ]

    y_plot = (Q_SIGN * ch1)[
        event_indices,
        ::sample_stride,
    ]

    finite_x = x_plot[
        np.isfinite(x_plot)
    ]

    finite_y = y_plot[
        np.isfinite(y_plot)
    ]

    if finite_x.size > 0:
        x_min = float(np.min(finite_x))
        x_max = float(np.max(finite_x))

        if x_max > x_min:
            x_pad = 0.03 * (x_max - x_min)
        else:
            x_pad = max(
                abs(x_min),
                1.0,
            ) * 1e-6

        ax.set_xlim(
            x_min - x_pad,
            x_max + x_pad,
        )

    if finite_y.size > 0:
        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))

        if y_max > y_min:
            y_pad = 0.03 * (y_max - y_min)
        else:
            y_pad = max(
                abs(y_min),
                1.0,
            ) * 1e-6

        ax.set_ylim(
            y_min - y_pad,
            y_max + y_pad,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("ch0")
    ax.set_ylabel(
        f"{'+' if Q_SIGN > 0 else '-'}ch1"
    )

    ax.set_title(
        f"{title}\n"
        f"trigger track, all events, "
        f"sample stride={TRIGGER_TRACK_SAMPLE_STRIDE}"
    )

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

    # ------------------------------------------------------------
    # 最後のプロットだけ、グレーの全波形が入るように ylim を上書き
    # （他の処理は変えない）
    # ------------------------------------------------------------
    sample_indices = np.arange(0, magnitude.shape[1], max(1, WAVEFORM_SAMPLE_STRIDE))
    y_plot = magnitude[np.ix_(event_indices, sample_indices)]

    finite_y = y_plot[np.isfinite(y_plot)]
    if finite_y.size > 0:
        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))

        if y_max <= y_min:
            pad = max(abs(y_min), 1.0) * 1e-6
        else:
            pad = 0.04 * (y_max - y_min)

        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)

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


def plot_calibrated_01_and_05(
    dataset: DatasetGroup,
    corrected_ch0: np.ndarray,
    corrected_ch1: np.ndarray,
    output_dir: Path,
    calibration: CalibrationSet,
) -> None:
    """
    IQ calibration後のデータについて、以前の01と05相当を作る。

    01:
        ch0を横軸、ch1を縦軸にした全イベントIQ track。
        trigger系ではsample stride=5で間引き、
        各sample位置のイベント中央値trackを時系列グラデーション表示。

    05:
        各イベントの冒頭10%からpedestalを求めて引いた
        sqrt(ch0^2 + ch1^2) の全イベント重ね書き。
        trigger系だけ作成する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_old_pngs(output_dir)

    ch0_plot = bin_waveforms_for_plot(
        corrected_ch0,
        PLOT_SAMPLE_BIN,
    )
    ch1_plot = bin_waveforms_for_plot(
        corrected_ch1,
        PLOT_SAMPLE_BIN,
    )

    sample_rate_hz = get_sample_rate_hz(dataset.records)

    if sample_rate_hz is not None:
        effective_sample_rate_hz = (
            sample_rate_hz
            / PLOT_SAMPLE_BIN
        )
    else:
        effective_sample_rate_hz = None

    x_axis, x_label = make_x_axis(
        ch0_plot.shape[1],
        effective_sample_rate_hz,
    )

    report_path = output_dir / "iqcal_01_05_report.pdf"

    with PdfPages(report_path) as pdf:
        if dataset.kind == "pedestal":
            fig = plot_pedestal_track(
                ch0_plot,
                ch1_plot,
                title=f"{dataset.name}: IQ CALIBRATED",
            )
        else:
            fig = plot_trigger_track(
                ch0_plot,
                ch1_plot,
                title=f"{dataset.name}: IQ CALIBRATED",
            )

        # 01のIQ track図だけに、IQ補正後の共振円を重ねる。
        # 軸範囲など、既存のtrack描画処理は変更しない。
        track_ax = fig.axes[0]
        theta = np.linspace(0.0, 2.0 * np.pi, 500)
        resonance_circle = (
            calibration.final_circle_center
            + calibration.final_circle_radius
            * np.exp(1j * theta)
        )
        track_ax.plot(
            resonance_circle.real,
            resonance_circle.imag,
            linewidth=1.8,
            linestyle="--",
            label="fitted resonance circle",
            zorder=3,
        )
        track_ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        save_plot_page(
            fig,
            output_dir / "01_iqcal_iq_tracks_all.png",
            pdf,
        )

        if dataset.kind == "trigger":
            fig = plot_pedestal_subtracted_magnitude(
                ch0_plot,
                ch1_plot,
                x_axis,
                x_label,
                title=(
                    f"{dataset.name}: IQ CALIBRATED "
                    "pedestal-subtracted magnitude"
                ),
            )
            save_plot_page(
                fig,
                output_dir
                / "05_iqcal_pedestal_subtracted_amplitude_all.png",
                pdf,
            )

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


def add_frequency_sweep_endpoints(
    ax: plt.Axes,
    z_scan: np.ndarray,
    frequency_hz: np.ndarray,
) -> None:
    """
    IQ周波数スイープの先頭点と末尾点を表示する。
    calibration計算には影響せず、diagnostics図への描画だけを追加する。
    """
    z_scan = np.asarray(z_scan, dtype=complex).reshape(-1)
    frequency_hz = np.asarray(frequency_hz, dtype=float).reshape(-1)

    if z_scan.size == 0 or z_scan.size != frequency_hz.size:
        return

    endpoint_specs = (
        (0, "sweep start", "^", "tab:green", (8, 8)),
        (-1, "sweep end", "v", "tab:red", (8, -16)),
    )

    for index, label, marker, color, offset in endpoint_specs:
        point = z_scan[index]
        frequency_ghz = frequency_hz[index] / 1e9

        ax.scatter(
            [point.real],
            [point.imag],
            marker=marker,
            s=105,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=10,
            label=f"{label}: {frequency_ghz:.9f} GHz",
        )
        ax.annotate(
            label,
            xy=(point.real, point.imag),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=color,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 0.9,
            },
            zorder=11,
        )


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
        add_frequency_sweep_endpoints(
            axes[0, 0],
            calibration.z_scan_raw,
            calibration.frequency_hz,
        )
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
        add_frequency_sweep_endpoints(
            axes[1, 0],
            z_tau,
            calibration.frequency_hz,
        )
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
        add_frequency_sweep_endpoints(
            axes[1, 1],
            calibration.z_scan_final,
            calibration.frequency_hz,
        )
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
            "iq_resonance_scan_glob": IQ_RESONANCE_SCAN_GLOB,
            "iq_resonance_scan_file_override": IQ_RESONANCE_SCAN_FILE_OVERRIDE,
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

def process_dataset(
    dataset: DatasetGroup,
    calibration: CalibrationSet,
    output_root: Path,
) -> dict[str, Any]:
    dataset_output = (
        output_root
        / sanitize_name(dataset.name)
    )
    dataset_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # まずfolderの周波数setがcalibration setと一致しているか確認。
    set_difference_hz = abs(
        dataset.frequency_hz
        - calibration.target_frequency_hz
    )

    if set_difference_hz > MAX_CALIBRATION_MATCH_HZ:
        raise ValueError(
            "dataset and IQ calibration are not the same frequency set: "
            f"dataset folder={dataset.frequency_hz/1e9:.9f} GHz, "
            f"calibration={calibration.target_frequency_hz/1e9:.9f} GHz, "
            f"difference={set_difference_hz/1e6:.3f} MHz"
        )

    (
        readout_frequency_hz,
        readout_frequency_source,
    ) = resolve_dataset_readout_frequency_hz(
        dataset,
        calibration,
    )

    print(
        "[frequency match] "
        f"dataset={dataset.name}, "
        f"folder set={dataset.frequency_hz/1e9:.9f} GHz, "
        f"calibration target={calibration.target_frequency_hz/1e9:.9f} GHz, "
        f"applied waveform frequency={readout_frequency_hz/1e9:.9f} GHz "
        f"({readout_frequency_source})"
    )

    z_raw = (
        dataset.ch0
        + 1j * Q_SIGN * dataset.ch1
    )

    z_corrected = apply_full_calibration(
        z_raw,
        readout_frequency_hz,
        calibration,
    )

    corrected_ch0 = np.real(
        z_corrected
    )
    corrected_ch1 = (
        np.imag(z_corrected)
        / Q_SIGN
    )

    corrected_files: list[str] = []

    if SAVE_CORRECTED_NPZ:
        for record in dataset.records:
            corrected_files.append(
                str(
                    save_corrected_record(
                        record,
                        calibration,
                        readout_frequency_hz,
                        dataset_output,
                    )
                )
            )

    plot_calibrated_01_and_05(
        dataset,
        corrected_ch0,
        corrected_ch1,
        dataset_output,
        calibration,
    )

    summary = {
        "dataset_name": dataset.name,
        "folder_path": str(dataset.folder_path),
        "kind": dataset.kind,
        "n_files": len(dataset.records),
        "n_events": int(dataset.ch0.shape[0]),
        "n_samples": int(dataset.ch0.shape[1]),
        "folder_frequency_hz": dataset.frequency_hz,
        "calibration_target_frequency_hz": (
            calibration.target_frequency_hz
        ),
        "applied_waveform_frequency_hz": (
            readout_frequency_hz
        ),
        "applied_waveform_frequency_source": (
            readout_frequency_source
        ),
        "frequency_set_difference_hz": (
            set_difference_hz
        ),
        "status": "calibrated",
        "corrected_files": corrected_files,
    }

    write_json(
        dataset_output
        / "dataset_summary.json",
        summary,
    )

    print(
        f"[dataset done] {dataset.name}: calibrated"
    )

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
    print("2026-07-14 selected-frequency IQ calibration: corrected 01 / 05 plots")
    print("=" * 88)
    print(f"DATA_ROOT                : {DATA_ROOT}")
    print(f"OUTPUT_ROOT              : {selected_output_root}")
    print(f"SELECT_FREQUENCY_GHZ     : {SELECT_FREQUENCY_GHZ}")
    print(f"N_EDGE_POINTS            : {N_EDGE_POINTS} (each edge)")
    print(f"IQ_SCAN_GLOB             : {IQ_RESONANCE_SCAN_GLOB}")
    print(f"IQ_SCAN_OVERRIDE         : {IQ_RESONANCE_SCAN_FILE_OVERRIDE}")
    print(f"TRIGGER_TRACK_STRIDE     : {TRIGGER_TRACK_SAMPLE_STRIDE}")
    print(f"PEDESTAL_FRACTION        : {PEDESTAL_FRACTION:.0%}")
    print(f"PLOT_SAMPLE_BIN          : {PLOT_SAMPLE_BIN}")
    print()

    datasets = discover_selected_dataset_groups(
        DATA_ROOT,
        selected_frequency_hz,
    )

    # IQ calibrationには必ず
    # iq_scan_f_reso_*.npzを直接使用する。
    calibration = build_selected_calibration_set(
        DATA_ROOT,
        selected_frequency_hz,
    )

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
            "iq_resonance_scan_glob": IQ_RESONANCE_SCAN_GLOB,
            "iq_resonance_scan_file_override": IQ_RESONANCE_SCAN_FILE_OVERRIDE,
            "circle_half_width_points": CIRCLE_HALF_WIDTH_POINTS,
            "pedestal_fraction": PEDESTAL_FRACTION,
            "max_calibration_match_hz": MAX_CALIBRATION_MATCH_HZ,
            "trigger_track_sample_stride": TRIGGER_TRACK_SAMPLE_STRIDE,
            "pedestal_track_sample_stride": PEDESTAL_TRACK_SAMPLE_STRIDE,
            "waveform_sample_stride": WAVEFORM_SAMPLE_STRIDE,
            "plot_sample_bin": PLOT_SAMPLE_BIN,
            "readout_frequency_overrides_ghz": READOUT_FREQUENCY_OVERRIDES_GHZ,
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
