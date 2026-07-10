from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_DIR = Path("/Volumes/NO NAME/data/iqscan0703")

OUTPUT_DIR = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260527/"
    "iqscan0703_iq_raw"
)

POINT_SIZE = 22
LINE_WIDTH = 1.0
ALPHA = 0.8

# dd の列番号
COL_FREQUENCY = 0
COL_CH0 = 1
COL_CH1 = 2


# =============================================================================
# FUNCTIONS
# =============================================================================

def temperature_label_from_filename(path: Path) -> str:
    """iq_4.76K.npz -> 4.76 K, iq_wide_4.80K.npz -> wide 4.80 K"""
    return path.stem.replace("iq_", "").replace("_", " ")


def natural_sort_key(path: Path) -> list[object]:
    """ファイル名中の温度を数値順に並べる。"""
    parts = re.split(r"(\d+\.\d+|\d+)", path.name)

    key: list[object] = []
    for part in parts:
        try:
            key.append(float(part))
        except ValueError:
            key.append(part.lower())

    return key


def load_iqscan(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    dd[:, 0] = frequency
    dd[:, 1] = ch0
    dd[:, 2] = ch1
    """
    with np.load(path, allow_pickle=False) as npz:
        if "dd" not in npz.files:
            raise KeyError(
                f"{path.name}: 'dd' が見つかりません。"
                f"利用可能なキー: {npz.files}"
            )

        dd = np.asarray(npz["dd"], dtype=float)

    if dd.ndim != 2 or dd.shape[1] < 3:
        raise ValueError(
            f"{path.name}: dd の shape が想定外です: {dd.shape}\n"
            "期待: (n_points, 3) 以上"
        )

    frequency = dd[:, COL_FREQUENCY]
    ch0 = dd[:, COL_CH0]
    ch1 = dd[:, COL_CH1]

    return frequency, ch0, ch1


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("iq_*.npz"), key=natural_sort_key)

    if not files:
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {INPUT_DIR / 'iq_*.npz'}"
        )

    print(f"[input]  {INPUT_DIR}")
    print(f"[output] {OUTPUT_DIR}")
    print(f"[files]  {len(files)} file(s)")

    fig, ax = plt.subplots(figsize=(9, 8))

    summary_rows: list[tuple[str, int, float, float, float, float]] = []

    for path in files:
        frequency, ch0, ch1 = load_iqscan(path)
        label = temperature_label_from_filename(path)

        # 周波数掃引の順番を見やすくするため、線も重ねる
        ax.plot(
            ch0,
            ch1,
            "-",
            lw=LINE_WIDTH,
            alpha=0.6,
        )

        ax.scatter(
            ch0,
            ch1,
            s=POINT_SIZE,
            alpha=ALPHA,
            label=f"{label} (N={len(ch0)})",
        )

        print(f"\n[load] {path.name}")
        print(f"  shape          : ({len(ch0)}, 3)")
        print(f"  frequency range: {np.min(frequency):.8g} -- {np.max(frequency):.8g}")
        print(f"  ch0 range      : {np.min(ch0):.8g} -- {np.max(ch0):.8g}")
        print(f"  ch1 range      : {np.min(ch1):.8g} -- {np.max(ch1):.8g}")
        print(f"  first row      : {frequency[0]:.8g}, {ch0[0]:.8g}, {ch1[0]:.8g}")

        summary_rows.append(
            (
                label,
                len(ch0),
                float(np.min(frequency)),
                float(np.max(frequency)),
                float(np.mean(ch0)),
                float(np.mean(ch1)),
            )
        )

    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.set_title("IQ scan: raw ch0 vs ch1")

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax.legend(
        title="Temperature",
        fontsize=8,
        title_fontsize=9,
        loc="best",
    )

    fig.tight_layout()

    output_png = OUTPUT_DIR / "iqscan0703_raw_ch0_vs_ch1_all_temperatures.png"
    fig.savefig(output_png, dpi=250)
    plt.close(fig)

    summary_csv = OUTPUT_DIR / "iqscan0703_summary.csv"

    with summary_csv.open("w", encoding="utf-8") as f:
        f.write(
            "dataset,n_points,"
            "frequency_min,frequency_max,"
            "ch0_mean,ch1_mean\n"
        )

        for row in summary_rows:
            f.write(
                f"{row[0]},{row[1]},"
                f"{row[2]:.12g},{row[3]:.12g},"
                f"{row[4]:.12g},{row[5]:.12g}\n"
            )

    print(f"\n[saved] {output_png}")
    print(f"[saved] {summary_csv}")


if __name__ == "__main__":
    main()