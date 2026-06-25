#!/usr/bin/env python3
"""
14:27 の first 測定だけを使って、50 Hz laser / 1 Hz temperature-cycle で
phase-fold した pedestal 軌跡と notch S21 model を比較する。

対象:
    20260527 / 5.476GHz_z=7.5mm_x=3.4mm
    -> フォルダ名が完全一致する first のみを読む
    -> _second, _third, _fourth, _fifth は絶対に読まない

実行前に主に確認・変更する場所:
  1. DATA_DATE, INPUT_MODE
  2. FIT_JSON_PATH または FIT_PARAMETERS
  3. 必要なら FR_SCAN_HALF_WIDTH_MHZ
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
# ここだけ基本的に変える
# ============================================================

DATA_DATE = "20260527"

# "cloud" : OneDrive / CloudStorage側
# "local" : kidanalysis/data/20260527 側
# "both"  : 両方を探索する（同じ測定を二重に読まないよう最初に見つかったものだけ使う）
INPUT_MODE = "cloud"

# 14:27 first のみ。
# "5.476GHz_z=7.5mm_x=3.4mm_second" 等は使わない。
TARGET_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"
NPZ_PATTERN = "wf_*.npz"

# 測定ログ: 1000 events, laser 50 Hz, temperature modulation 1 Hz
N_EVENTS_TO_USE = 1000
N_PHASE_BINS = 50

# baseline: None なら全 t < 0 を使う
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# readout tone / f_r simulation
F_TONE_GHZ = 5.476
FR_SCAN_HALF_WIDTH_MHZ = 5.0
N_FR_SCAN = 4001

# notch fit parameter input.
# JSON を使う場合はそのパスを入れる。
# JSON に a, alpha, tau_ns (or tau), Ql, Qc, phi, fr_ghz (or fr) が必要。
FIT_JSON_PATH = None
# 例:
# FIT_JSON_PATH = (
#     Path.home()
#     / "Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit"
#     / DATA_DATE
#     / "s21_fit_5.476GHz.json"
# )

# FIT_JSON_PATH = None のときはこちらを編集する。
# 重要: ch0/ch1 waveform と同じ複素座標・同じ単位で fit した値を入れる。
# offset_re, offset_im は必要に応じて使用する。
FIT_PARAMETERS = {
    "a": 1.0,
    "alpha": 0.0,
    "tau_ns": 0.0,
    "Ql": 452.6852,
    "Qc": 488.4716,
    "phi": 0.0,          # 要取得
    "fr_ghz": 5.479238,
    "offset_re": 0.0,
    "offset_im": 0.0,
}

# 物理的にモデルと waveform の絶対スケールが違う場合でも、
# 向き・handedness を壊さずに「表示だけ」重ねるための設定。
# "auto_positive_real": 正の実数スケール + 平行移動のみを自動適用。
#                       回転・共役変換はしない。
# "none": モデルをそのまま表示する。
OVERLAY_SCALE_MODE = "auto_positive_real"

DPI = 250


# ============================================================
# パス設定: 既存の解析スクリプトと同じ方針
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "temperature_iq_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

local_data_dir = HERE / "data" / DATA_DATE

cloud_data_candidates = [
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-TheUniversityofTokyo"
    / "東京大学"
    / "4S"
    / "kidfit"
    / DATA_DATE,

    Path.home()
    / "OneDrive - The University of Tokyo"
    / "東京大学"
    / "4S"
    / "kidfit"
    / DATA_DATE,

    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive - The University of Tokyo"
    / "東京大学"
    / "4S"
    / "kidfit"
    / DATA_DATE,
]

# 必要なら手動で追加
EXTRA_INPUT_ROOTS = [
    # Path("/Users/kubokosei/Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit/20260527"),
]


# ============================================================
# utility
# ============================================================

def scalar(x):
    """npz scalar / 1-element array を Python scalar にする。"""
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else x


def make_time_axis_s(npts: int, sample_rate_hz: float, ref_position_percent: float) -> np.ndarray:
    return (
        np.arange(npts, dtype=float) - npts * ref_position_percent / 100.0
    ) / sample_rate_hz


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def normalize(z: complex) -> complex:
    mag = abs(z)
    return z / mag if mag > 0 else np.nan + 1j * np.nan


# ============================================================
# input path / data loading
# ============================================================

def collect_input_roots() -> list[Path]:
    candidates: list[tuple[str, Path]] = []

    if INPUT_MODE in ["local", "both"]:
        candidates.append(("local", local_data_dir))

    if INPUT_MODE in ["cloud", "both"]:
        for p in cloud_data_candidates:
            candidates.append(("cloud", p))

    for p in EXTRA_INPUT_ROOTS:
        candidates.append(("extra", p))

    print()
    print("===== path check =====")
    print("HERE:", HERE)
    print("INPUT_MODE:", INPUT_MODE)

    roots: list[Path] = []
    seen: set[str] = set()

    for kind, path in candidates:
        path = Path(path).expanduser().resolve(strict=False)
        print(f"[{kind}] {path}")
        print("   exists:", path.is_dir())

        if path.is_dir() and path.as_posix() not in seen:
            roots.append(path)
            seen.add(path.as_posix())

    if not roots:
        raise RuntimeError("入力フォルダが見つかりません。cloud_data_candidates を確認してください。")

    return roots


def find_first_measurement_dir(roots: list[Path]) -> Path:
    """
    TARGET_DIR_NAME が完全一致するフォルダだけを返す。
    _second 以降を混ぜないため rglob / prefix match はしない。
    """
    found: list[Path] = []

    for root in roots:
        candidate = root / TARGET_DIR_NAME
        print("candidate:", candidate, "exists:", candidate.is_dir())
        if candidate.is_dir():
            found.append(candidate)

    if not found:
        raise RuntimeError(
            f"'{TARGET_DIR_NAME}' が見つかりません。\n"
            "first のフォルダ名が本当にこの名前か、パス設定を確認してください。"
        )

    if len(found) > 1:
        print("\nWARNING: 同名フォルダが複数の root に見つかりました。")
        print("最初のものだけ使います（重複ロードを防ぐため）。")

    chosen = found[0]

    # 念のため first の中に repeat らしい名前が入っていないか表示。
    assert chosen.name == TARGET_DIR_NAME
    assert not any(
        chosen.name.endswith("_" + tag)
        for tag in ["second", "third", "fourth", "fifth"]
    )

    return chosen


def load_first_1000_events(meas_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    npz_files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)

    if not npz_files:
        raise RuntimeError(f"{meas_dir} に {NPZ_PATTERN} がありません。")

    print()
    print("===== selected measurement =====")
    print("measurement dir:", meas_dir)
    print("number of npz files:", len(npz_files))

    ch0_blocks: list[np.ndarray] = []
    ch1_blocks: list[np.ndarray] = []
    used_files: list[Path] = []
    time_ref: np.ndarray | None = None
    n_loaded = 0

    for path in npz_files:
        if n_loaded >= N_EVENTS_TO_USE:
            break

        try:
            data = np.load(path)
        except Exception as e:
            print("skip load error:", path, e)
            continue

        required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        missing = [k for k in required if k not in data.files]
        if missing:
            print("skip missing keys:", path, missing)
            continue

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)

        if ch0.ndim == 1:
            ch0 = ch0[None, :]
        if ch1.ndim == 1:
            ch1 = ch1[None, :]

        if ch0.shape != ch1.shape:
            print("skip shape mismatch:", path, ch0.shape, ch1.shape)
            continue

        npts = int(scalar(data["npts"]))
        sample_rate_hz = float(scalar(data["sample_rate"]))
        ref_position_percent = float(scalar(data["ref_position"]))

        if ch0.shape[1] != npts:
            print("skip npts mismatch:", path, ch0.shape[1], npts)
            continue

        time_s = make_time_axis_s(npts, sample_rate_hz, ref_position_percent)
        if time_ref is None:
            time_ref = time_s
        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time axis mismatch:", path)
            continue

        n_take = min(len(ch0), N_EVENTS_TO_USE - n_loaded)
        ch0_blocks.append(ch0[:n_take])
        ch1_blocks.append(ch1[:n_take])
        used_files.append(path)
        n_loaded += n_take

        print(f"load: {path.name}  -> {n_take} events (total={n_loaded})")

    if time_ref is None:
        raise RuntimeError("有効な waveform が読み込めませんでした。")

    if n_loaded != N_EVENTS_TO_USE:
        raise RuntimeError(
            f"{N_EVENTS_TO_USE} events を要求しましたが、{n_loaded} events しか読めませんでした。"
        )

    return time_ref, np.vstack(ch0_blocks), np.vstack(ch1_blocks), used_files


# ============================================================
# pedestal folding
# ============================================================

def compute_phase_folded_pedestal(
    time_s: np.ndarray,
    ch0: np.ndarray,
    ch1: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    time_us = time_s * 1e6

    if BASELINE_WINDOW_US is None:
        baseline_mask = time_us < 0
    else:
        lo, hi = BASELINE_WINDOW_US
        baseline_mask = (time_us >= lo) & (time_us <= hi)

    if baseline_mask.sum() < 3:
        raise ValueError("baseline points が少なすぎます。BASELINE_WINDOW_US を確認してください。")

    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    event_index = np.arange(len(ped0), dtype=int)
    raw_df = pd.DataFrame(
        {
            "event_index": event_index,
            "phase_bin": event_index % N_PHASE_BINS,
            "ped0_V": ped0,
            "ped1_V": ped1,
        }
    )

    g = raw_df.groupby("phase_bin", sort=True)
    phase_df = g.agg(
        n_events=("event_index", "size"),
        ped0_mean_V=("ped0_V", "mean"),
        ped0_std_V=("ped0_V", "std"),
        ped1_mean_V=("ped1_V", "mean"),
        ped1_std_V=("ped1_V", "std"),
    ).reset_index()

    phase_df["ped0_sem_V"] = phase_df["ped0_std_V"] / np.sqrt(phase_df["n_events"])
    phase_df["ped1_sem_V"] = phase_df["ped1_std_V"] / np.sqrt(phase_df["n_events"])

    return phase_df, ped0, ped1


# ============================================================
# notch model
# ============================================================

def load_fit_parameters() -> dict[str, float]:
    if FIT_JSON_PATH is None:
        par = dict(FIT_PARAMETERS)
    else:
        path = Path(FIT_JSON_PATH).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FIT_JSON_PATH が見つかりません: {path}")
        with path.open() as f:
            par = json.load(f)

    required = ["a", "alpha", "Ql", "Qc", "phi"]
    missing = [k for k in required if k not in par]
    if missing:
        raise KeyError(f"notch parameter に必要な key がありません: {missing}")

    if "fr_ghz" not in par:
        if "fr" not in par:
            raise KeyError("notch parameter に 'fr_ghz' または 'fr' が必要です。")
        par["fr_ghz"] = par["fr"]

    if "tau_ns" not in par:
        par["tau_ns"] = par.get("tau", 0.0)

    return {k: float(v) for k, v in par.items()}


def s21_notch(par: dict[str, float], f_ghz: np.ndarray) -> np.ndarray:
    """
    s21_notch_func.py と同じ notch model。

    f: GHz, tau: ns とすると exp(-2*pi*i*f*tau) は無次元。
    """
    f_ghz = np.asarray(f_ghz, dtype=float)

    env = (
        par["a"]
        * np.exp(1j * par["alpha"])
        * np.exp(-2j * np.pi * f_ghz * par["tau_ns"])
    )
    res = (
        par["Ql"] / abs(par["Qc"]) * np.exp(1j * par["phi"])
    ) / (
        1.0 + 2j * par["Ql"] * (f_ghz / par["fr_ghz"] - 1.0)
    )

    offset = complex(par.get("offset_re", 0.0), par.get("offset_im", 0.0))
    return offset + env * (1.0 - res)


def model_fr_curve(par: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fr0 = par["fr_ghz"]
    fr_grid = fr0 + np.linspace(
        -FR_SCAN_HALF_WIDTH_MHZ,
        FR_SCAN_HALF_WIDTH_MHZ,
        N_FR_SCAN,
    ) * 1e-3

    z_grid = np.empty(len(fr_grid), dtype=complex)
    for i, fr in enumerate(fr_grid):
        p = par.copy()
        p["fr_ghz"] = fr
        z_grid[i] = s21_notch(p, np.array([F_TONE_GHZ]))[0]

    return fr_grid, z_grid, s21_notch(par, np.array([F_TONE_GHZ]))[0]


def positive_real_overlay_transform(
    z_data: np.ndarray,
    z_model: np.ndarray,
) -> tuple[float, complex]:
    """
    z' = scale * z_model + offset
    scale > 0 の実数、offset は複素数。

    向き・回転方向・handedness を不変に保った表示用の整列だけ行う。
    """
    if OVERLAY_SCALE_MODE == "none":
        return 1.0, 0.0 + 0.0j

    if OVERLAY_SCALE_MODE != "auto_positive_real":
        raise ValueError("OVERLAY_SCALE_MODE は 'auto_positive_real' か 'none' にしてください。")

    # 方向を曲げない正の実数の倍率を、IQ上の span の比で決める。
    dspan = max(np.ptp(z_data.real), np.ptp(z_data.imag), 1e-15)
    mspan = max(np.ptp(z_model.real), np.ptp(z_model.imag), 1e-15)
    scale = dspan / mspan

    offset = np.mean(z_data) - scale * np.mean(z_model)
    return scale, offset


def nearest_fr_on_curve(
    z_data: np.ndarray,
    fr_grid: np.ndarray,
    z_model_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distance = abs(z_data[:, None] - z_model_grid[None, :])
    j = np.argmin(distance, axis=1)
    return fr_grid[j], distance[np.arange(len(z_data)), j]


# ============================================================
# compare z = ch0 + i ch1 and z = ch0 - i ch1
# ============================================================

def evaluate_convention(
    phase_df: pd.DataFrame,
    par: dict[str, float],
    sign_q: int,
) -> dict:
    """
    sign_q=+1: z=ch0 + i ch1
    sign_q=-1: z=ch0 - i ch1
    """
    z_data = (
        phase_df["ped0_mean_V"].to_numpy()
        + 1j * sign_q * phase_df["ped1_mean_V"].to_numpy()
    )

    fr_grid, z_model_raw, z_ref_raw = model_fr_curve(par)
    scale, offset = positive_real_overlay_transform(z_data, z_model_raw)
    z_model = scale * z_model_raw + offset
    z_ref = scale * z_ref_raw + offset

    fr_est, mismatch = nearest_fr_on_curve(z_data, fr_grid, z_model)

    # T↑ => f_r↓ を仮定したとき
    hot_idx = int(np.argmin(fr_est))
    cold_idx = int(np.argmax(fr_est))

    # f_r を 1 MHz 下げたときの局所モデル変位
    p_down = par.copy()
    p_down["fr_ghz"] -= 1e-3
    z_fr_down = scale * s21_notch(p_down, np.array([F_TONE_GHZ]))[0] + offset
    v_fr_down = z_fr_down - z_ref

    # cold -> hot のデータ変位
    v_data_cold_to_hot = z_data[hot_idx] - z_data[cold_idx]
    alignment = np.real(
        normalize(v_data_cold_to_hot) * np.conj(normalize(v_fr_down))
    )

    return {
        "sign_q": sign_q,
        "label": "ch0 + i ch1" if sign_q == 1 else "ch0 - i ch1",
        "z_data": z_data,
        "fr_grid": fr_grid,
        "z_model": z_model,
        "z_ref": z_ref,
        "scale": scale,
        "offset": offset,
        "fr_est": fr_est,
        "mismatch": mismatch,
        "hot_idx": hot_idx,
        "cold_idx": cold_idx,
        "v_fr_down": v_fr_down,
        "alignment": alignment,
        "mean_mismatch": float(np.mean(mismatch)),
    }


# ============================================================
# plot / output
# ============================================================

def plot_result(
    phase_df: pd.DataFrame,
    result: dict,
    par: dict[str, float],
    out_png: Path,
) -> None:
    z_data = result["z_data"]
    z_model = result["z_model"]
    z_ref = result["z_ref"]
    fr_grid = result["fr_grid"]
    fr_est = result["fr_est"]
    mismatch = result["mismatch"]
    hot_idx = result["hot_idx"]
    cold_idx = result["cold_idx"]
    bins = phase_df["phase_bin"].to_numpy(dtype=int)

    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 11.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    ax = axes[0]

    # f_r scan curve: low -> high orientation
    ax.plot(
        z_model.real * 1e3,
        z_model.imag * 1e3,
        lw=2.0,
        label=rf"model at $f_{{tone}}={F_TONE_GHZ:.6f}$ GHz; $f_r$ scan",
    )

    # add arrow in the model curve showing fr decreasing
    k_center = len(z_model) // 2
    k_low = max(0, k_center - len(z_model) // 8)
    ax.annotate(
        r"$f_r \downarrow$",
        xy=(z_model[k_low].real * 1e3, z_model[k_low].imag * 1e3),
        xytext=(z_model[k_center].real * 1e3, z_model[k_center].imag * 1e3),
        arrowprops=dict(arrowstyle="->", lw=1.7),
        fontsize=10,
    )

    sc = ax.scatter(
        z_data.real * 1e3,
        z_data.imag * 1e3,
        c=bins,
        s=45,
        zorder=5,
        label="phase-folded pedestal",
    )
    fig.colorbar(sc, ax=ax, label="phase bin")

    # phase progression arrows
    for i in range(len(z_data)):
        j = (i + 1) % len(z_data)
        ax.annotate(
            "",
            xy=(z_data[j].real * 1e3, z_data[j].imag * 1e3),
            xytext=(z_data[i].real * 1e3, z_data[i].imag * 1e3),
            arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.55),
        )

    for i, b in enumerate(bins):
        if b % 5 == 0:
            ax.annotate(
                str(b),
                (z_data[i].real * 1e3, z_data[i].imag * 1e3),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    ax.scatter(
        z_data[cold_idx].real * 1e3,
        z_data[cold_idx].imag * 1e3,
        s=100,
        marker="s",
        zorder=7,
        label=rf"cold candidate: bin {bins[cold_idx]} ($f_r$ max)",
    )
    ax.scatter(
        z_data[hot_idx].real * 1e3,
        z_data[hot_idx].imag * 1e3,
        s=190,
        marker="*",
        zorder=7,
        label=rf"hot candidate: bin {bins[hot_idx]} ($f_r$ min)",
    )

    # local model vector: f_r down
    v = result["v_fr_down"]
    data_span = max(np.ptp(z_data.real), np.ptp(z_data.imag), 1e-15)
    v_plot = 0.28 * data_span * normalize(v)
    ax.annotate(
        "",
        xy=((z_ref + v_plot).real * 1e3, (z_ref + v_plot).imag * 1e3),
        xytext=(z_ref.real * 1e3, z_ref.imag * 1e3),
        arrowprops=dict(arrowstyle="->", lw=2.0),
    )
    ax.annotate(
        r"model local direction: $f_r \downarrow$",
        ((z_ref + 1.1 * v_plot).real * 1e3, (z_ref + 1.1 * v_plot).imag * 1e3),
        fontsize=9,
    )

    ax.set_xlabel(f"Re[{result['label']}] [mV]")
    ax.set_ylabel(f"Im[{result['label']}] [mV]")
    ax.set_title(
        "14:27 first only — pedestal trajectory vs notch model\n"
        + result["label"]
        + rf",  alignment(cold$\to$hot, $f_r\downarrow$) = {result['alignment']:.3f}"
    )
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")

    ax2 = axes[1]
    ax2.plot(
        bins,
        (fr_est - par["fr_ghz"]) * 1e3,
        "o-",
        label=r"nearest model $f_r$",
    )
    ax2.set_xlabel("phase bin = event index mod 50")
    ax2.set_ylabel(r"estimated $\Delta f_r$ [MHz]")
    ax2.grid(True)

    axr = ax2.twinx()
    axr.plot(
        bins,
        mismatch * 1e3,
        "s--",
        alpha=0.75,
        label=r"distance to $f_r$-only curve",
    )
    axr.set_ylabel("distance [mV]")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def main() -> None:
    print("DATA_DATE:", DATA_DATE)
    print("TARGET_DIR_NAME:", TARGET_DIR_NAME)
    print("N_EVENTS_TO_USE:", N_EVENTS_TO_USE)
    print("N_PHASE_BINS:", N_PHASE_BINS)
    print("OUT_DIR:", OUT_DIR)

    roots = collect_input_roots()
    measurement_dir = find_first_measurement_dir(roots)
    time_s, ch0, ch1, used_files = load_first_1000_events(measurement_dir)

    phase_df, ped0, ped1 = compute_phase_folded_pedestal(time_s, ch0, ch1)
    par = load_fit_parameters()

    results = [
        evaluate_convention(phase_df, par, sign_q=+1),
        evaluate_convention(phase_df, par, sign_q=-1),
    ]

    # model-fit coordinateが正しい前提で、f_r-only curve への平均距離が小さい方を採択
    best = min(results, key=lambda r: r["mean_mismatch"])

    print()
    print("===== convention comparison =====")
    for r in results:
        print(
            f"{r['label']:14s}"
            f" mean mismatch = {r['mean_mismatch']*1e3:.6g} mV,"
            f" alignment = {r['alignment']:.4f}"
        )

    print()
    print("selected convention:", best["label"])
    print(
        "interpretation: T↑ => f_r↓ を仮定した cold->hot vector と"
        " model f_r↓ vector の alignment"
    )
    print(f"alignment = {best['alignment']:.4f}")
    print("  +1 に近い: 温度ドリフトは主に f_r↓ と整合")
    print("   0 に近い: Q / gain / phase drift 等の寄与が大きい")
    print("  -1 に近い: 温度方向が逆、または model IQ coordinate の不整合")

    phase_out = phase_df.copy()
    phase_out["fr_est_ghz"] = best["fr_est"]
    phase_out["fr_est_shift_mhz"] = (best["fr_est"] - par["fr_ghz"]) * 1e3
    phase_out["fr_only_distance_V"] = best["mismatch"]
    phase_out["complex_convention"] = best["label"]
    phase_out["temperature_assumption"] = "T_up_implies_fr_down"
    phase_out["temperature_label"] = "intermediate"
    phase_out.loc[best["hot_idx"], "temperature_label"] = "hot_candidate"
    phase_out.loc[best["cold_idx"], "temperature_label"] = "cold_candidate"

    raw_out = OUT_DIR / "pedestal_per_event_first_1000.csv"
    pd.DataFrame(
        {
            "event_index": np.arange(N_EVENTS_TO_USE),
            "phase_bin": np.arange(N_EVENTS_TO_USE) % N_PHASE_BINS,
            "ped0_V": ped0,
            "ped1_V": ped1,
        }
    ).to_csv(raw_out, index=False)

    phase_out_path = OUT_DIR / "phase_folded_pedestal_with_fr_estimate.csv"
    phase_out.to_csv(phase_out_path, index=False)

    png_path = OUT_DIR / "pedestal_temperature_direction_vs_s21.png"
    plot_result(phase_df, best, par, png_path)

    info_path = OUT_DIR / "run_info.txt"
    info_lines = [
        f"measurement_dir = {measurement_dir}",
        f"files_used = {len(used_files)}",
        *[str(p) for p in used_files],
        "",
        f"complex_convention = {best['label']}",
        f"mean_model_mismatch_V = {best['mean_mismatch']}",
        f"alignment_cold_to_hot_vs_fr_down = {best['alignment']}",
        f"overlay_scale_mode = {OVERLAY_SCALE_MODE}",
        f"positive_real_scale = {best['scale']}",
        f"translation_offset = {best['offset']}",
    ]
    info_path.write_text("\n".join(info_lines), encoding="utf-8")

    print()
    print("saved:")
    print(" ", png_path)
    print(" ", raw_out)
    print(" ", phase_out_path)
    print(" ", info_path)
    print("\ndone")


if __name__ == "__main__":
    main()
