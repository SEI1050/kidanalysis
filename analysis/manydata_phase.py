from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


#ファイルの読み込み
DATA_DIR = Path(
    "/Volumes/NO NAME/data/20260618/"
    "5.443GHz_z=7.5mm_x=3.40mm_100000evants"
)

LASER_HZ = 2.0
AMP_WINDOW_US = (0.0, 1.5)

#関数の準備

def read_peak_from_one_npz(path):
    with np.load(path) as data:
        ch0 = data["ch0"]
        ch1 = data["ch1"]

        npts = int(data["npts"].item())
        sample_rate_hz = float(data["sample_rate"].item())
        ref_position = int(data["ref_position"].item())

    time_us = (np.arange(npts)-npts*ref_position/100) / sample_rate_hz * 1e6
    baseline_mask = (time_us < AMP_WINDOW_US[0]) | (time_us > AMP_WINDOW_US[1])

    pulse_mask = (time_us >= AMP_WINDOW_US[0]) & (time_us <= AMP_WINDOW_US[1])
    ped0 = np.mean(ch0[:, baseline_mask], axis=1)
    ped1 = np.mean(ch1[:, baseline_mask], axis=1)

    delta0 = ch0 - ped0[:, None]
    delta1 = ch1 - ped1[:, None]

    amplitude_2d = np.hypot(delta0[:, pulse_mask], delta1[:, pulse_mask])
    peak_2d_V = np.max(amplitude_2d, axis=1)

    return peak_2d_V



#データの読み込み
    
files = sorted(DATA_DIR.glob("*.npz"))

if len(files) == 0:
    raise ValueError(f"No npz files found in {DATA_DIR}")

all_peaks = []

for i, path in enumerate(files):
    peaks = read_peak_from_one_npz(path)
    all_peaks.append(peaks)

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(files)} files")

peak_mV = np.concatenate(all_peaks) *1e3

events_index = np.arange(len(peak_mV))

time_s = events_index / LASER_HZ

phase = events_index % LASER_HZ

print("number of events:", len(peak_mV))


def block_median(x, y, block_size):
    x_result = []
    y_result = []

    for i in range(0, len(x), block_size):
        x_block = x[i:i + block_size]
        y_block = y[i:i + block_size]

        if len(x_block) == 0:
            continue

        x_result.append(np.median(x_block))
        y_result.append(np.median(y_block))
    return np.array(x_result), np.array(y_result)

PLOT_EVERY = 10
BLOCK_SIZE_PER_PHASE = 100

phase_colors = ["tab:blue", "tab:orange"]
phase_labels = ["phase 0: even events", "phase 1: odd events"]

fig, ax = plt.subplots(figsize=(13, 6))

for p in [0, 1]:
    mask = phase == p

    # 生データは間引いて薄く描く
    ax.scatter(
        time_s[mask][::PLOT_EVERY],
        peak_mV[mask][::PLOT_EVERY],
        s=4,
        alpha=0.15,
        color=phase_colors[p],
    )

    # 同じ phase の event だけでブロック中央値を取る
    time_line, peak_line = block_median(
        time_s[mask],
        peak_mV[mask],
        BLOCK_SIZE_PER_PHASE,
    )

    ax.plot(
        time_line,
        peak_line,
        "-o",
        ms=3,
        lw=1.5,
        color=phase_colors[p],
        label=phase_labels[p],
    )

ax.set_title("2 Hz pulse-amplitude trend from raw npz files")
ax.set_xlabel("estimated elapsed time [s]")
ax.set_ylabel("2D peak amplitude [mV]")
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig("02hz_phase_trend_from_npz.png", dpi=200)
plt.show()
