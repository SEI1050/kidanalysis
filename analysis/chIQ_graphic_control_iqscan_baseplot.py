from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # インタラクティブ用

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, Button, CheckButtons


# =============================================================================
# SETTINGS
# =============================================================================
IQSCAN_DIR = Path("/Volumes/NO NAME/data/iqscan0703")
WAVEFORM_FILE = Path(
    "/Volumes/NO NAME/data/20260527/5.476GHz_z=7.5mm_x=3.4mm/"
    "wf_260527_142822_49.73Hz.npz"
)

TARGET_TEMPERATURE_K = 5.80

EVENT_RATE_HZ = 49.73
TEMP_MODULATION_HZ = 1.0
N_PHASE_BINS = 50
PHASE_OFFSET_CYCLES = 0.0
BASELINE_SLICE = slice(0, 1000)


# =============================================================================
# LOADERS
# =============================================================================
def find_iqscan_file(temperature_k: float) -> Path:
    exact = IQSCAN_DIR / f"iq_{temperature_k:.2f}K.npz"
    if exact.exists():
        return exact

    candidates = sorted(IQSCAN_DIR.glob(f"*{temperature_k:.2f}K*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"{IQSCAN_DIR} に {temperature_k:.2f} K の iq scan が見つかりません。"
        )
    raise RuntimeError(
        f"{temperature_k:.2f} K に一致する iq scan が複数見つかりました:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_iqscan(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        if "dd" not in npz:
            raise KeyError(f"{path} に 'dd' キーがありません。keys={list(npz.keys())}")
        dd = np.asarray(npz["dd"], dtype=float)

    if dd.ndim != 2 or dd.shape[1] < 3:
        raise ValueError(f"{path}: dd の shape が想定外です: {dd.shape}")

    return dd[:, 0], dd[:, 1], dd[:, 2]


def waveform_array_events_by_samples(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} は2次元配列である必要があります。shape={x.shape}")

    # 通常は (samples, events) or (events, samples) のどちらか
    # samples=5000, events=1000 or 4000 を想定
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x


def load_waveform_baselines(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        if "ch0" not in npz or "ch1" not in npz:
            raise KeyError(
                f"{path} に ch0/ch1 がありません。keys={keys}\n"
                "必要ならキー名を修正してください。"
            )

        ch0 = waveform_array_events_by_samples(npz["ch0"], "ch0")
        ch1 = waveform_array_events_by_samples(npz["ch1"], "ch1")

    if ch0.shape != ch1.shape:
        raise ValueError(f"ch0/ch1 の shape が一致しません: {ch0.shape}, {ch1.shape}")

    baseline_ch0 = np.median(ch0[:, BASELINE_SLICE], axis=1)
    baseline_ch1 = np.median(ch1[:, BASELINE_SLICE], axis=1)
    return baseline_ch0, baseline_ch1


# =============================================================================
# PHASE BINNING
# =============================================================================
def temperature_phase_bin_medians(
    baseline_ch0: np.ndarray,
    baseline_ch1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_events = len(baseline_ch0)
    if n_events != len(baseline_ch1):
        raise ValueError("baseline_ch0 と baseline_ch1 のイベント数が違います。")

    event_index = np.arange(n_events)
    phase_cycles = np.mod(
        event_index * TEMP_MODULATION_HZ / EVENT_RATE_HZ + PHASE_OFFSET_CYCLES,
        1.0,
    )
    phase_bin = np.floor(phase_cycles * N_PHASE_BINS).astype(int)
    phase_bin = np.clip(phase_bin, 0, N_PHASE_BINS - 1)

    centers_cycles = (np.arange(N_PHASE_BINS) + 0.5) / N_PHASE_BINS
    median_ch0 = np.full(N_PHASE_BINS, np.nan)
    median_ch1 = np.full(N_PHASE_BINS, np.nan)

    for ibin in range(N_PHASE_BINS):
        mask = phase_bin == ibin
        if np.any(mask):
            median_ch0[ibin] = np.median(baseline_ch0[mask])
            median_ch1[ibin] = np.median(baseline_ch1[mask])

    return centers_cycles, median_ch0, median_ch1


# =============================================================================
# GEOMETRIC TRANSFORM
# =============================================================================
def rotate_points(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr


def reflect_y_eq_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return y.copy(), x.copy()


def transform_points(
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    do_reflect: bool,
) -> tuple[np.ndarray, np.ndarray]:
    # まず回転、その後に y=x 反転
    xt, yt = rotate_points(x, y, angle_deg)
    if do_reflect:
        xt, yt = reflect_y_eq_x(xt, yt)
    return xt, yt


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    # --- load iq scan ---
    iqscan_path = find_iqscan_file(TARGET_TEMPERATURE_K)
    freq_hz, scan_ch0, scan_ch1 = load_iqscan(iqscan_path)

    # --- load waveform baseline medians ---
    baseline_ch0, baseline_ch1 = load_waveform_baselines(WAVEFORM_FILE)
    phase_cycles, med_ch0, med_ch1 = temperature_phase_bin_medians(
        baseline_ch0, baseline_ch1
    )

    valid = np.isfinite(med_ch0) & np.isfinite(med_ch1)
    med_x = med_ch0[valid]
    med_y = med_ch1[valid]
    med_phase = phase_cycles[valid]

    loop_x = np.r_[med_x, med_x[0]]
    loop_y = np.r_[med_y, med_y[0]]

    # --- figure layout ---
    fig = plt.figure(figsize=(10.5, 9.0))
    ax = fig.add_axes([0.10, 0.22, 0.72, 0.70])  # main plot
    ax_angle = fig.add_axes([0.10, 0.11, 0.58, 0.04])  # slider
    ax_check = fig.add_axes([0.84, 0.55, 0.12, 0.10])  # checkbox
    ax_reset = fig.add_axes([0.84, 0.45, 0.12, 0.06])  # reset button

    # --- fixed reference: waveform baseline medians ---
    norm = Normalize(vmin=0.0, vmax=1.0)
    ax.plot(
        loop_x, loop_y,
        "-",
        color="k",
        lw=1.2,
        alpha=0.55,
        label="5.476 GHz waveform baseline median"
    )
    sc_phase = ax.scatter(
        med_x, med_y,
        c=med_phase,
        cmap="viridis",
        norm=norm,
        s=38,
        edgecolors="k",
        linewidths=0.35,
        zorder=4,
        label="temp-phase binned medians"
    )

    phase0_idx = int(np.nanargmin(np.abs(med_phase - 0.0)))
    ax.scatter(
        med_x[phase0_idx], med_y[phase0_idx],
        marker="*",
        s=170,
        facecolor="none",
        edgecolor="crimson",
        linewidth=1.3,
        zorder=5,
        label="temp phase 0"
    )

    cbar = fig.colorbar(sc_phase, ax=ax, pad=0.02)
    cbar.set_label("temperature phase within 1 Hz cycle [cycle]")
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])

    # --- original iq scan (fixed, gray) ---
    ax.plot(
        scan_ch0, scan_ch1,
        "--",
        color="0.65",
        lw=1.2,
        alpha=0.9,
        label=f"original iq scan {TARGET_TEMPERATURE_K:.2f} K"
    )
    ax.plot(
        scan_ch0, scan_ch1,
        linestyle="None",
        marker="o",
        ms=4.0,
        color="0.65",
        alpha=0.8,
    )

    # --- transformed iq scan (updated interactively) ---
    init_angle = 0.0
    init_reflect = False
    x_t, y_t = transform_points(scan_ch0, scan_ch1, init_angle, init_reflect)

    line_trans, = ax.plot(
        x_t, y_t,
        "-",
        color="tab:blue",
        lw=1.8,
        alpha=0.95,
        label="transformed iq scan"
    )
    pts_trans, = ax.plot(
        x_t, y_t,
        linestyle="None",
        marker="o",
        ms=4.2,
        color="tab:blue",
        alpha=0.95,
    )

    # 点番号を先頭だけ少し見せてもよい
    start_marker = ax.scatter(
        [x_t[0]], [y_t[0]],
        marker="D",
        s=52,
        color="tab:red",
        zorder=6,
        label="scan start point"
    )

    # 原点
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)

    ax.set_xlabel("ch0 [raw]")
    ax.set_ylabel("ch1 [raw]")
    ax.set_title(
        f"Interactive rotation / y=x reflection of iq scan at T={TARGET_TEMPERATURE_K:.2f} K"
    )
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)

    # 軸範囲は元のscanとbaseline両方を含むように少し余裕をもたせる
    all_x = np.r_[scan_ch0, med_x]
    all_y = np.r_[scan_ch1, med_y]
    xmax = np.nanmax(np.abs(all_x)) * 1.25
    ymax = np.nanmax(np.abs(all_y)) * 1.25
    lim = max(xmax, ymax)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # --- widgets ---
    slider_angle = Slider(
        ax=ax_angle,
        label="rotation angle [deg]",
        valmin=-180.0,
        valmax=180.0,
        valinit=init_angle,
        valstep=1.0,
    )

    check = CheckButtons(
        ax=ax_check,
        labels=["reflect y=x"],
        actives=[init_reflect],
    )

    btn_reset = Button(ax_reset, "Reset")

    info_text = fig.text(
        0.84, 0.36,
        "transform order:\n1) rotate\n2) reflect y=x",
        fontsize=10,
        va="top"
    )

    state_text = fig.text(
        0.84, 0.28,
        "",
        fontsize=10,
        va="top"
    )

    def update_state_text() -> None:
        angle = slider_angle.val
        reflect = check.get_status()[0]
        state_text.set_text(
            f"angle = {angle:.1f} deg\nreflect = {reflect}"
        )

    def update_plot(_=None) -> None:
        angle = slider_angle.val
        reflect = check.get_status()[0]

        xt, yt = transform_points(scan_ch0, scan_ch1, angle, reflect)

        line_trans.set_data(xt, yt)
        pts_trans.set_data(xt, yt)
        start_marker.set_offsets(np.array([[xt[0], yt[0]]]))

        update_state_text()
        fig.canvas.draw_idle()

    def reset(_event) -> None:
        slider_angle.reset()
        current = check.get_status()[0]
        if current:
            check.set_active(0)  # ON -> OFF
        update_plot()

    slider_angle.on_changed(update_plot)
    check.on_clicked(update_plot)
    btn_reset.on_clicked(reset)

    update_state_text()
    plt.show()


if __name__ == "__main__":
    main()