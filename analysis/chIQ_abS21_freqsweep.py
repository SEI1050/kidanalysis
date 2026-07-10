from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

HERE = Path(__file__).resolve().parent

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

OUT_DIR = HERE / "data" / "20260527" / "s21_sweep_baseline_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "s21_ratio_all_temperatures_over_9K_magnitude.png"

DPI = 250
POINT_SIZE = 5


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_s21_s2p(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s2p から周波数 [Hz]、S21振幅 [dB]、S21位相 [deg] を読む。"""
    data = np.loadtxt(path, comments=["!", "#"])

    freq_hz = data[:, 0]
    amp_db = data[:, 5]
    phase_deg = data[:, 6]

    return freq_hz, amp_db, phase_deg


def calc_normalized_s21(
    amp_db: np.ndarray,
    phase_deg: np.ndarray,
    amp_normal_db: np.ndarray,
    phase_normal_deg: np.ndarray,
) -> np.ndarray:
    """
    S21(T) / S21(normal) を複素数として返す。
    振幅列が dB であることを仮定する。
    """
    amp_ratio = 10 ** ((amp_db - amp_normal_db) / 20.0)
    phase_diff_deg = phase_deg - phase_normal_deg

    return amp_ratio * np.exp(1j * np.deg2rad(phase_diff_deg))


# =============================================================================
# LOAD NORMAL STATE
# =============================================================================

normal_path = S2P_DIR / S2P_FILES[NORMAL_T_K]
freq_normal_hz, amp_normal_db, phase_normal_deg = load_s21_s2p(normal_path)

freq_ghz = freq_normal_hz / 1e9


# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(9.5, 5.8))

for temp_k, filename in S2P_FILES.items():
    path = S2P_DIR / filename
    freq_hz, amp_db, phase_deg = load_s21_s2p(path)

    if len(freq_hz) != len(freq_normal_hz) or not np.allclose(
        freq_hz, freq_normal_hz
    ):
        raise ValueError(
            f"周波数点が 9.0 K データと一致しません:\n{path}"
        )

    s21_normalized = calc_normalized_s21(
        amp_db,
        phase_deg,
        amp_normal_db,
        phase_normal_deg,
    )

    ax.scatter(
        freq_ghz,
        np.abs(s21_normalized),
        s=POINT_SIZE,
        alpha=0.75,
        label=f"{temp_k:.2f} K",
    )

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$|S_{21}(T) / S_{21}(9.0\,\mathrm{K})|$")
ax.set_title(r"$|S_{21}(T)/S_{21}(9K)|$ magnitude versus readout frequency")
ax.grid(True, alpha=0.3)

# 凡例を図の右側に配置して、データと重ならないようにする
ax.legend(
    title="Temperature",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0,
    markerscale=1.4,
)

fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"saved: {OUT_FILE}")