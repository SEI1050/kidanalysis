#!/usr/bin/env python3
"""
9 K で規格化した温度別 S21 sweep と、
20260527 / 5.476GHz_z=7.5mm_x=3.4mm の baseline magnitude を比較する。

作る図:
  1. 温度ごとの normalized S21 notch を絶対周波数軸で重ねた図
  2. f_tone = 5.476 GHz における |S21(T)/S21(9K)| の温度依存
  3. baseline magnitude の温度周期位相依存
  4. 温度波形の位相が分かっている場合だけ、baseline と sweep 予測の直接比較

注意:
  - s2p の第 6, 7 列（Python index 5, 6）が
    S21[dB], phase[deg] である前提。
  - raw の mV と S21 の絶対値は比べない。
    比較するのは規格化後の相対変化。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

# 固定読出し周波数
F_TONE_GHZ = 5.476

# このスクリプト自身の場所
HERE = Path(__file__).resolve().parent

# .s2p の置き場所
# 例:
#   your_project/
#   ├── compare_s21_sweeps_to_baseline.py
#   └── data/
#       ├── 1_3.5K.s2p
#       ├── ...
#       └── 7_9.0K.s2p
S2P_DIR = HERE / "data"

S2P_FILES = {
    3.50: "1_3.5K.s2p",
    4.74: "2_4.74K.s2p",
    5.56: "3_5.56K.s2p",
    6.10: "4_6.10K.s2p",
    6.68: "5_6.68K.s2p",
    7.41: "6_7.41K.s2p",
    9.00: "7_9.0K.s2p",
}

NORMAL_T_K = 9.00

# あなたの s2p 読み込みコードに合わせる
S21_DB_COLUMN = 5
S21_PHASE_DEG_COLUMN = 6

# baseline スクリプトが作った CSV
BASELINE_SUMMARY_CSV = (
    HERE
    / "data"
    / "20260527"
    / "baseline_vs_temperature_phase"
    / "baseline_phase_summary.csv"
)

OUT_DIR = (
    HERE
    / "data"
    / "20260527"
    / "s21_sweep_baseline_compare"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 250


# -----------------------------------------------------------------------------
# 任意設定:
#
# 実際の温度変調が例えば 5.75--5.82 K だと分かっているなら設定する。
# 温度波形の位相も分かる場合だけ、baseline と S21 予測を同じ横軸に重ねる。
#
# 温度最大が起きる時刻を 0--1 s で入れる。
# 例: 温度最大が phase=0.74 s なら TEMP_MAX_PHASE_S = 0.74
# -----------------------------------------------------------------------------

TEMP_MEAN_K = 5.84
TEMP_AMPLITUDE_K = 0.04
TEMP_MAX_PHASE_S = 0.13

# 例:
# TEMP_MEAN_K = 5.785
# TEMP_AMPLITUDE_K = 0.035
# TEMP_MAX_PHASE_S = 0.74


# =============================================================================
# HELPERS
# =============================================================================

def to_ghz(freq: np.ndarray) -> np.ndarray:
    """
    周波数列を GHz に揃える。

    例:
      5.476       -> GHz のまま
      5.476e9     -> Hz とみなして GHz に変換
      5476        -> MHz とみなして GHz に変換
    """
    freq = np.asarray(freq, dtype=float)
    scale = np.nanmedian(np.abs(freq))

    if scale > 1e8:
        return freq / 1e9

    if scale > 1e3:
        return freq / 1e3

    return freq


def load_s21_from_s2p(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    s2p の S21[dB], phase[deg] を複素 S21 に変換する。
    """
    data = np.loadtxt(path, comments=["!", "#"])

    if data.ndim != 2:
        raise ValueError(f"読み込めない形式です: {path}")

    required_col = max(S21_DB_COLUMN, S21_PHASE_DEG_COLUMN)

    if data.shape[1] <= required_col:
        raise ValueError(
            f"{path.name}: column が足りません。"
            f"必要 index={required_col}, 実際={data.shape[1]}"
        )

    freq_ghz = to_ghz(data[:, 0])

    s21_db = data[:, S21_DB_COLUMN]
    s21_phase_deg = data[:, S21_PHASE_DEG_COLUMN]

    amplitude = 10.0 ** (s21_db / 20.0)
    phase_rad = np.deg2rad(s21_phase_deg)

    s21 = amplitude * np.exp(1j * phase_rad)

    order = np.argsort(freq_ghz)

    return freq_ghz[order], s21[order]


def interp_complex(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray | float,
) -> np.ndarray:
    """
    複素配列を実部・虚部に分けて線形補間する。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=complex)
    x_new = np.asarray(x_new, dtype=float)

    if np.min(x_new) < np.min(x) or np.max(x_new) > np.max(x):
        raise ValueError(
            "補間位置が S21 sweep の周波数範囲外です。"
            f"requested={x_new.min():.6f}--{x_new.max():.6f} GHz, "
            f"available={x.min():.6f}--{x.max():.6f} GHz"
        )

    return (
        np.interp(x_new, x, y.real)
        + 1j * np.interp(x_new, x, y.imag)
    )


def normalize_by_9k(
    f_ghz: np.ndarray,
    s21: np.ndarray,
    f_9k_ghz: np.ndarray,
    s21_9k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    S21(T) / S21(9K) を作る。
    周波数グリッドが微妙に違っても 9 K 側を補間する。
    """
    lo = max(np.min(f_ghz), np.min(f_9k_ghz))
    hi = min(np.max(f_ghz), np.max(f_9k_ghz))

    mask = (f_ghz >= lo) & (f_ghz <= hi)

    f_use = f_ghz[mask]
    s21_use = s21[mask]

    s21_9k_interp = interp_complex(
        f_9k_ghz,
        s21_9k,
        f_use,
    )

    if np.any(np.abs(s21_9k_interp) < 1e-15):
        raise RuntimeError("9 K S21 にほぼゼロの点があり、除算できません。")

    return f_use, s21_use / s21_9k_interp


def db_from_magnitude(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=float)
    return 20.0 * np.log10(np.maximum(mag, 1e-15))


def load_baseline_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "baseline summary CSV が見つかりません。\n"
            f"expected: {path}\n"
            "先に baseline のスクリプトを実行してください。"
        )

    df = pd.read_csv(path)

    required = [
        "temperature_phase_center_s",
        "baseline_mag_median_V",
        "baseline_mag_q25_V",
        "baseline_mag_q75_V",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"baseline CSV に必要な列がありません: {missing}"
        )

    return df.sort_values("temperature_phase_center_s").reset_index(drop=True)


# =============================================================================
# PLOTS
# =============================================================================

def plot_notches(
    normalized: dict[float, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for temp_k in sorted(normalized):
        f_ghz, s21_norm = normalized[temp_k]

        ax.plot(
            f_ghz,
            db_from_magnitude(np.abs(s21_norm)),
            lw=1.6,
            label=f"{temp_k:.2f} K",
        )

    ax.axvline(
        F_TONE_GHZ,
        color="black",
        lw=1.6,
        ls="--",
        label=rf"$f_{{\rm tone}}={F_TONE_GHZ:.3f}$ GHz",
    )

    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"$20\log_{10}|S_{21}(T)/S_{21}(9\,\mathrm{{K}})|$ [dB]")
    ax.set_title("Temperature-dependent normalized S21 notches")
    ax.grid(True)
    ax.legend(ncols=2, fontsize=9)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tone_vs_temperature(
    temperatures: np.ndarray,
    tone_magnitude: np.ndarray,
    output_path: Path,
) -> None:
    reference_index = np.argmin(np.abs(temperatures - NORMAL_T_K))
    ref_mag = tone_magnitude[reference_index]

    relative = tone_magnitude / ref_mag

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        temperatures,
        db_from_magnitude(tone_magnitude),
        "o-",
        lw=1.8,
    )
    axes[0].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[0].set_ylabel(
        rf"$20\log_{{10}}|S_{{21}}({F_TONE_GHZ:.3f}\,\mathrm{{GHz}},T)"
        r"/S_{21}(9\,{\rm K})|$ [dB]"
    )
    axes[0].set_title("Normalized S21 magnitude at the fixed readout frequency")
    axes[0].grid(True)

    axes[1].plot(
        temperatures,
        relative,
        "o-",
        lw=1.8,
    )
    axes[1].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[1].set_xlabel("temperature [K]")
    axes[1].set_ylabel(r"$|S_{21}(T)| / |S_{21}(9\,{\rm K})|$")
    axes[1].grid(True)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_baseline(
    baseline_df: pd.DataFrame,
    output_path: Path,
) -> None:
    phase_s = baseline_df["temperature_phase_center_s"].to_numpy()

    mag_mV = baseline_df["baseline_mag_median_V"].to_numpy() * 1e3
    q25_mV = baseline_df["baseline_mag_q25_V"].to_numpy() * 1e3
    q75_mV = baseline_df["baseline_mag_q75_V"].to_numpy() * 1e3

    mag_ref = np.median(mag_mV)
    relative = mag_mV / mag_ref
    relative_q25 = q25_mV / mag_ref
    relative_q75 = q75_mV / mag_ref

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].errorbar(
        phase_s,
        mag_mV,
        yerr=[mag_mV - q25_mV, q75_mV - mag_mV],
        fmt="o-",
        capsize=3,
        lw=1.5,
    )
    axes[0].set_ylabel(r"$\sqrt{ch0^2+ch1^2}$ baseline [mV]")
    axes[0].set_title("Baseline magnitude folded into the 1 Hz temperature cycle")
    axes[0].grid(True)

    axes[1].errorbar(
        phase_s,
        relative,
        yerr=[relative - relative_q25, relative_q75 - relative],
        fmt="o-",
        capsize=3,
        lw=1.5,
    )
    axes[1].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[1].set_xlabel("temperature-cycle phase [s]")
    axes[1].set_ylabel("baseline / cycle median")
    axes[1].grid(True)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_direct_comparison_if_possible(
    temperatures: np.ndarray,
    tone_magnitude: np.ndarray,
    baseline_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    温度の位相情報を設定した場合だけ、
    baseline と sweep 側の予測を同じ phase 軸で比較する。
    """
    if (
        TEMP_MEAN_K is None
        or TEMP_AMPLITUDE_K is None
        or TEMP_MAX_PHASE_S is None
    ):
        return False

    phase_s = baseline_df["temperature_phase_center_s"].to_numpy()
    baseline_mV = baseline_df["baseline_mag_median_V"].to_numpy() * 1e3

    # T(phi) = Tmean + dT cos[2pi(phi - phi_max)]
    temp_dynamic = (
        TEMP_MEAN_K
        + TEMP_AMPLITUDE_K
        * np.cos(2.0 * np.pi * (phase_s - TEMP_MAX_PHASE_S))
    )

    sweep_mag_dynamic = np.interp(
        temp_dynamic,
        temperatures,
        tone_magnitude,
    )

    # readout power や gain の違いを消すため、各系列を中央値で規格化
    baseline_rel = baseline_mV / np.median(baseline_mV)
    sweep_rel = sweep_mag_dynamic / np.median(sweep_mag_dynamic)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        phase_s,
        temp_dynamic,
        "o-",
        lw=1.5,
        label="assumed temperature waveform",
    )
    axes[0].set_ylabel("temperature [K]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        phase_s,
        baseline_rel,
        "o-",
        lw=1.8,
        label=r"data: $\sqrt{ch0^2+ch1^2}$/median",
    )

    axes[1].plot(
        phase_s,
        sweep_rel,
        "s--",
        lw=1.6,
        label=(
            rf"sweep prediction at {F_TONE_GHZ:.3f} GHz "
            r"($S_{21}(T)/S_{21}(9\,{\rm K})$)"
        ),
    )

    axes[1].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[1].set_xlabel("temperature-cycle phase [s]")
    axes[1].set_ylabel("relative magnitude")
    axes[1].grid(True)
    axes[1].legend()

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("===== load S21 sweeps =====")

    loaded: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for temp_k, filename in S2P_FILES.items():
        path = S2P_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"S2P file not found: {path}\n"
                "S2P_DIR または S2P_FILES を確認してください。"
            )

        f_ghz, s21 = load_s21_from_s2p(path)
        loaded[temp_k] = (f_ghz, s21)

        print(
            f"{temp_k:>5.2f} K : {path.name}"
            f"  ({f_ghz.min():.6f}--{f_ghz.max():.6f} GHz)"
        )

    if NORMAL_T_K not in loaded:
        raise RuntimeError(f"{NORMAL_T_K} K の reference sweep がありません。")

    f_9k, s21_9k = loaded[NORMAL_T_K]

    normalized: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for temp_k, (f_ghz, s21) in loaded.items():
        f_use, s21_norm = normalize_by_9k(
            f_ghz=f_ghz,
            s21=s21,
            f_9k_ghz=f_9k,
            s21_9k=s21_9k,
        )
        normalized[temp_k] = (f_use, s21_norm)

    # -------------------------------------------------------------------------
    # fixed-tone value
    # -------------------------------------------------------------------------
    temperatures = np.array(sorted(normalized), dtype=float)

    tone_values = []

    for temp_k in temperatures:
        f_ghz, s21_norm = normalized[temp_k]

        z_tone = interp_complex(
            f_ghz,
            s21_norm,
            F_TONE_GHZ,
        )

        tone_values.append(np.abs(z_tone))

    tone_magnitude = np.asarray(tone_values, dtype=float)

    print("\n===== fixed-tone S21 =====")
    for temp_k, mag in zip(temperatures, tone_magnitude):
        print(
            f"T={temp_k:5.2f} K : "
            f"|S21(T)/S21(9K)| = {mag:.8f}, "
            f"{db_from_magnitude(mag):+.4f} dB"
        )

    # -------------------------------------------------------------------------
    # baseline summary
    # -------------------------------------------------------------------------
    baseline_df = load_baseline_summary(BASELINE_SUMMARY_CSV)

    # -------------------------------------------------------------------------
    # plots
    # -------------------------------------------------------------------------
    notch_png = OUT_DIR / "normalized_s21_notches_absolute_frequency.png"
    tone_png = OUT_DIR / "normalized_s21_at_fixed_tone_vs_temperature.png"
    baseline_png = OUT_DIR / "baseline_magnitude_vs_temperature_phase.png"
    direct_png = OUT_DIR / "baseline_vs_s21_prediction_same_phase.png"

    plot_notches(normalized, notch_png)
    plot_tone_vs_temperature(temperatures, tone_magnitude, tone_png)
    plot_baseline(baseline_df, baseline_png)

    direct_created = plot_direct_comparison_if_possible(
        temperatures=temperatures,
        tone_magnitude=tone_magnitude,
        baseline_df=baseline_df,
        output_path=direct_png,
    )

    # -------------------------------------------------------------------------
    # summary CSV
    # -------------------------------------------------------------------------
    out_csv = OUT_DIR / "fixed_tone_s21_vs_temperature.csv"

    pd.DataFrame({
        "temperature_K": temperatures,
        "f_tone_GHz": F_TONE_GHZ,
        "abs_S21_over_S21_9K": tone_magnitude,
        "relative_dB": db_from_magnitude(tone_magnitude),
    }).to_csv(out_csv, index=False)

    baseline_mag_mV = (
        baseline_df["baseline_mag_median_V"].to_numpy() * 1e3
    )

    baseline_p2p_percent = (
        (np.max(baseline_mag_mV) - np.min(baseline_mag_mV))
        / np.mean(baseline_mag_mV)
        * 100.0
    )

    print("\n===== baseline magnitude =====")
    print(f"min = {np.min(baseline_mag_mV):.5f} mV")
    print(f"max = {np.max(baseline_mag_mV):.5f} mV")
    print(f"peak-to-peak / mean = {baseline_p2p_percent:.3f} %")

    print("\n===== output =====")
    print(notch_png)
    print(tone_png)
    print(baseline_png)
    print(out_csv)
    z = (
        summary_df["baseline_ch0_median_V"].to_numpy()
        + 1j * summary_df["baseline_ch1_median_V"].to_numpy()
    )

    phase_s = summary_df["temperature_phase_center_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        z.real * 1e3,
        z.imag * 1e3,
        c=phase_s,
        s=60,
    )

    ax.plot(z.real * 1e3, z.imag * 1e3, "-")
    ax.set_xlabel("ch0 baseline [mV]")
    ax.set_ylabel("ch1 baseline [mV]")
    ax.set_title("Baseline IQ trajectory during one temperature cycle")
    ax.axis("equal")
    ax.grid(True)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("temperature-cycle phase [s]")

    plt.show()

    if direct_created:
        print(direct_png)
    else:
        print(
            "\nDirect phase-by-phase comparison was skipped.\n"
            "Set TEMP_MEAN_K, TEMP_AMPLITUDE_K, "
            "and TEMP_MAX_PHASE_S to enable it."
        )



if __name__ == "__main__":
    main()