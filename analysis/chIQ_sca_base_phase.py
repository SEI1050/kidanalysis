#!/usr/bin/env python3
"""
20260527 / 5.476GHz_z=7.5mm_x=3.4mm の first data について、

- ch0 baseline
- ch1 baseline

を、50 Hz laser / 1 Hz temperature modulation として
温度周期内の位相に対して描画する。

baseline:
    各 waveform の t < 0 の平均値

出力:
    data/20260527/baseline_vs_temperature_phase/
      baseline_vs_temperature_phase.png
      baseline_phase_summary.csv
      baseline_event_by_event.csv
      run_info.txt
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

DATA_DATE = "20260527"
TARGET_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"
NPZ_PATTERN = "wf_*.npz"

# first data の先頭から使うイベント数
# 1000 events = 温度周期 20 回分（50 Hz laser / 1 Hz temperature の場合）
N_EVENTS_TO_USE = 1000

# 温度変調とレーザーの設定
LASER_RATE_HZ = 50.0
TEMPERATURE_RATE_HZ = 1.0

EVENTS_PER_TEMPERATURE_CYCLE = int(round(LASER_RATE_HZ / TEMPERATURE_RATE_HZ))
assert np.isclose(
    EVENTS_PER_TEMPERATURE_CYCLE,
    LASER_RATE_HZ / TEMPERATURE_RATE_HZ,
), "LASER_RATE_HZ / TEMPERATURE_RATE_HZ must be an integer."

# 50 にすると、温度周期をレーザー1発ごとに分ける
# 10 にすると、5イベントずつ平均されて見やすくなる
N_PHASE_BINS = 50
assert EVENTS_PER_TEMPERATURE_CYCLE % N_PHASE_BINS == 0
EVENTS_PER_PHASE_BIN = EVENTS_PER_TEMPERATURE_CYCLE // N_PHASE_BINS

# 温度周期の開始位置をずらしたい場合に変更する
# 例: 温度最大位置を横軸中央に移したい場合など
PHASE_OFFSET_EVENTS = 0

# baseline 領域 [us]
# None なら t < 0 をすべて使う
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

DPI = 300


# =============================================================================
# PATHS
# =============================================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "baseline_vs_temperature_phase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_ROOTS = [
    Path("/Volumes/NO NAME/data") / DATA_DATE,
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
    HERE / "data" / DATA_DATE,
]


# =============================================================================
# HELPERS
# =============================================================================

def scalar(x):
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else x


def make_time_axis_s(npts, sample_rate_hz, ref_position_percent):
    return (
        np.arange(npts, dtype=float)
        - npts * ref_position_percent / 100.0
    ) / sample_rate_hz


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    return (
        np.median(values),
        np.percentile(values, 25),
        np.percentile(values, 75),
    )


def find_target_directory():
    candidates = []

    print("\n===== input roots =====")
    for root in INPUT_ROOTS:
        root = Path(root).expanduser()
        target = root / TARGET_DIR_NAME

        n_npz = len(list(target.glob(NPZ_PATTERN))) if target.is_dir() else 0

        print(root)
        print("  exists:", root.is_dir())
        print("  target:", target)
        print("  wf files:", n_npz)

        if n_npz > 0:
            candidates.append(target)

    if not candidates:
        raise RuntimeError(
            f"{TARGET_DIR_NAME!r} を含む入力フォルダが見つかりません。"
        )

    if len(candidates) > 1:
        print("\nWARNING: 同じ測定フォルダが複数見つかりました。最初のものを使います。")

    print("\nselected:", candidates[0])
    return candidates[0]


def load_baselines(meas_dir):
    files = sorted(meas_dir.glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)

    baseline_ch0_blocks = []
    baseline_ch1_blocks = []
    used_files = []

    n_loaded = 0
    time_ref = None
    baseline_mask = None

    for path in files:
        if N_EVENTS_TO_USE is not None and n_loaded >= N_EVENTS_TO_USE:
            break

        try:
            data = np.load(path)
        except Exception as exc:
            print("skip unreadable:", path.name, exc)
            continue

        required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        missing = [key for key in required if key not in data.files]

        if missing:
            print("skip missing keys:", path.name, missing)
            continue

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)

        if ch0.ndim == 1:
            ch0 = ch0[None, :]
        if ch1.ndim == 1:
            ch1 = ch1[None, :]

        if ch0.shape != ch1.shape:
            print("skip shape mismatch:", path.name)
            continue

        npts = int(scalar(data["npts"]))
        sample_rate_hz = float(scalar(data["sample_rate"]))
        ref_position_percent = float(scalar(data["ref_position"]))

        time_s = make_time_axis_s(
            npts,
            sample_rate_hz,
            ref_position_percent,
        )

        if time_ref is None:
            time_ref = time_s
            time_us = time_s * 1e6

            if BASELINE_WINDOW_US is None:
                baseline_mask = time_us < 0.0
            else:
                lo_us, hi_us = BASELINE_WINDOW_US
                baseline_mask = (time_us >= lo_us) & (time_us <= hi_us)

            if baseline_mask.sum() < 3:
                raise RuntimeError("baseline に使える点が少なすぎます。")

        elif len(time_s) != len(time_ref) or not np.allclose(time_s, time_ref):
            print("skip time-axis mismatch:", path.name)
            continue

        n_take = len(ch0)
        if N_EVENTS_TO_USE is not None:
            n_take = min(n_take, N_EVENTS_TO_USE - n_loaded)

        baseline_ch0_blocks.append(
            ch0[:n_take, baseline_mask].mean(axis=1)
        )
        baseline_ch1_blocks.append(
            ch1[:n_take, baseline_mask].mean(axis=1)
        )

        used_files.append(path)
        n_loaded += n_take

        print(f"load: {path.name} -> {n_take} events (total={n_loaded})")

    if n_loaded == 0:
        raise RuntimeError("有効なイベントを読み込めませんでした。")

    return (
        np.concatenate(baseline_ch0_blocks),
        np.concatenate(baseline_ch1_blocks),
        used_files,
    )


def make_event_table(baseline_ch0, baseline_ch1):
    event_index = np.arange(len(baseline_ch0), dtype=int)

    laser_phase = (
        event_index + PHASE_OFFSET_EVENTS
    ) % EVENTS_PER_TEMPERATURE_CYCLE

    phase_bin = laser_phase // EVENTS_PER_PHASE_BIN

    thermal_phase_s = laser_phase / LASER_RATE_HZ
    thermal_phase_deg = 360.0 * laser_phase / EVENTS_PER_TEMPERATURE_CYCLE

    baseline_mag = np.sqrt(baseline_ch0**2 + baseline_ch1**2)

    return pd.DataFrame({
        "event_index": event_index,
        "laser_phase_in_temperature_cycle": laser_phase,
        "temperature_phase_bin": phase_bin,
        "temperature_phase_s": thermal_phase_s,
        "temperature_phase_deg": thermal_phase_deg,
        "baseline_ch0_V": baseline_ch0,
        "baseline_ch1_V": baseline_ch1,
        "baseline_mag_V": baseline_mag,
        "baseline_ch0_mV": baseline_ch0 * 1e3,
        "baseline_ch1_mV": baseline_ch1 * 1e3,
        "baseline_mag_mV": baseline_mag * 1e3,
    })


def make_phase_summary(event_df):
    rows = []

    for phase_bin in range(N_PHASE_BINS):
        df = event_df[event_df["temperature_phase_bin"] == phase_bin]

        if len(df) == 0:
            continue

        ch0_med, ch0_q25, ch0_q75 = median_iqr(df["baseline_ch0_V"])
        ch1_med, ch1_q25, ch1_q75 = median_iqr(df["baseline_ch1_V"])
        mag_med, mag_q25, mag_q75 = median_iqr(df["baseline_mag_V"])

        phase_start_event = phase_bin * EVENTS_PER_PHASE_BIN
        phase_end_event = (phase_bin + 1) * EVENTS_PER_PHASE_BIN - 1

        phase_center_event = 0.5 * (phase_start_event + phase_end_event)
        phase_center_s = phase_center_event / LASER_RATE_HZ
        phase_center_deg = (
            360.0 * phase_center_event / EVENTS_PER_TEMPERATURE_CYCLE
        )

        rows.append({
            "temperature_phase_bin": phase_bin,
            "laser_phase_start": phase_start_event,
            "laser_phase_end": phase_end_event,
            "temperature_phase_center_s": phase_center_s,
            "temperature_phase_center_deg": phase_center_deg,
            "n_events": len(df),

            "baseline_ch0_median_V": ch0_med,
            "baseline_ch0_q25_V": ch0_q25,
            "baseline_ch0_q75_V": ch0_q75,

            "baseline_ch1_median_V": ch1_med,
            "baseline_ch1_q25_V": ch1_q25,
            "baseline_ch1_q75_V": ch1_q75,

            "baseline_mag_median_V": mag_med,
            "baseline_mag_q25_V": mag_q25,
            "baseline_mag_q75_V": mag_q75,
        })

    return pd.DataFrame(rows)


def plot_baselines(event_df, summary_df, output_path):
    x_raw = event_df["temperature_phase_s"].to_numpy()
    x = summary_df["temperature_phase_center_s"].to_numpy()

    ch0 = summary_df["baseline_ch0_median_V"].to_numpy() * 1e3
    ch0_q25 = summary_df["baseline_ch0_q25_V"].to_numpy() * 1e3
    ch0_q75 = summary_df["baseline_ch0_q75_V"].to_numpy() * 1e3

    ch1 = summary_df["baseline_ch1_median_V"].to_numpy() * 1e3
    ch1_q25 = summary_df["baseline_ch1_q25_V"].to_numpy() * 1e3
    ch1_q75 = summary_df["baseline_ch1_q75_V"].to_numpy() * 1e3

    mag = summary_df["baseline_mag_median_V"].to_numpy() * 1e3
    mag_q25 = summary_df["baseline_mag_q25_V"].to_numpy() * 1e3
    mag_q75 = summary_df["baseline_mag_q75_V"].to_numpy() * 1e3

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 11),
        sharex=True,
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            "ch0",
            event_df["baseline_ch0_mV"].to_numpy(),
            ch0,
            ch0_q25,
            ch0_q75,
            "ch0 baseline [mV]",
            "ch0 baseline across the 1 Hz temperature cycle",
        ),
        (
            axes[1],
            "ch1",
            event_df["baseline_ch1_mV"].to_numpy(),
            ch1,
            ch1_q25,
            ch1_q75,
            "ch1 baseline [mV]",
            "ch1 baseline across the 1 Hz temperature cycle",
        ),
        (
            axes[2],
            "mag",
            event_df["baseline_mag_mV"].to_numpy(),
            mag,
            mag_q25,
            mag_q75,
            r"$\sqrt{ch0^2 + ch1^2}$ baseline [mV]",
            r"$\sqrt{ch0^2 + ch1^2}$ baseline across the 1 Hz temperature cycle",
        ),
    ]

    for ax, label, y_raw, y_med, y_q25, y_q75, ylabel, title in panels:
        ax.scatter(
            x_raw,
            y_raw,
            s=12,
            alpha=0.20,
            label="individual events",
        )

        ax.errorbar(
            x,
            y_med,
            yerr=[y_med - y_q25, y_q75 - y_med],
            fmt="o-",
            ms=4,
            capsize=3,
            lw=1.4,
            label="median ± IQR",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    axes[2].set_xlabel(
        "temperature-cycle phase [s] "
        f"(0–{1 / TEMPERATURE_RATE_HZ:.1f} s, folded)"
    )

    fig.suptitle(
        f"{DATA_DATE} / {TARGET_DIR_NAME}\n"
        f"{LASER_RATE_HZ:g} Hz laser, {TEMPERATURE_RATE_HZ:g} Hz temperature modulation",
        fontsize=13,
    )

    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)



def main():
    meas_dir = find_target_directory()

    baseline_ch0, baseline_ch1, used_files = load_baselines(meas_dir)

    event_df = make_event_table(baseline_ch0, baseline_ch1)
    summary_df = make_phase_summary(event_df)

    event_csv = OUT_DIR / "baseline_event_by_event.csv"
    summary_csv = OUT_DIR / "baseline_phase_summary.csv"
    figure_path = OUT_DIR / "baseline_vs_temperature_phase.png"
    info_path = OUT_DIR / "run_info.txt"

    event_df.to_csv(event_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_baselines(event_df, summary_df, figure_path)

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"measurement_dir: {meas_dir}\n")
        f.write(f"n_events: {len(event_df)}\n")
        f.write(f"laser_rate_hz: {LASER_RATE_HZ}\n")
        f.write(f"temperature_rate_hz: {TEMPERATURE_RATE_HZ}\n")
        f.write(f"events_per_temperature_cycle: {EVENTS_PER_TEMPERATURE_CYCLE}\n")
        f.write(f"n_phase_bins: {N_PHASE_BINS}\n")
        f.write(f"phase_offset_events: {PHASE_OFFSET_EVENTS}\n")
        f.write(f"baseline_window_us: {BASELINE_WINDOW_US}\n")
        f.write("\nused_files:\n")
        for path in used_files:
            f.write(f"{path}\n")

    print("\n===== output =====")
    print("figure :", figure_path)
    print("summary:", summary_csv)
    print("events :", event_csv)
    print("info   :", info_path)


if __name__ == "__main__":
    main()