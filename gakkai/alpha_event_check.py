"""Real-time random-event amp/tau check for pedestal-subtracted IQ pulses.

Usage:
    python3 alpha_event_check.py input.npz
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd


BIN_SIZE = 1
RANDOM_EVENT_COUNT = 5
REFRESH_INTERVAL_MS = 2000
RANDOM_SEED = None


def bin_waveforms(ch0, ch1, npts_raw, sample_rate, ref_position):
    """Average neighboring samples and return binned waveforms and the time axis."""
    nbin = npts_raw // BIN_SIZE
    usable_npts = nbin * BIN_SIZE

    ch0_binned = ch0[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)
    ch1_binned = ch1[:, :usable_npts].reshape(-1, nbin, BIN_SIZE).mean(axis=2)

    raw_time = (
        np.arange(usable_npts) - npts_raw * ref_position / 100.0
    ) / sample_rate
    time_ns = raw_time.reshape(nbin, BIN_SIZE).mean(axis=1) * 1e9
    return ch0_binned, ch1_binned, time_ns


def integrate_waveform(time_ns, signal):
    """Return the cumulative trapezoidal integral of a waveform."""
    if signal.size < 2:
        return np.zeros_like(signal, dtype=float)

    dt = np.diff(time_ns)
    if np.any(~np.isfinite(dt)):
        return np.full_like(signal, np.nan, dtype=float)

    integral = np.concatenate((
        [0.0],
        np.cumsum(0.5 * (signal[:-1] + signal[1:]) * dt),
    ))
    return integral


def interpolate_crossing(time_ns, signal, level):
    """Interpolate the time at which signal reaches the requested level."""
    if signal.size < 2:
        return np.nan

    for index in range(signal.size - 1):
        left_value = signal[index]
        right_value = signal[index + 1]
        if (left_value - level) * (right_value - level) <= 0:
            left_time = time_ns[index]
            right_time = time_ns[index + 1]
            if abs(right_value - left_value) < 1e-12:
                return 0.5 * (left_time + right_time)
            return left_time + (level - left_value) * (right_time - left_time) / (right_value - left_value)
    return np.nan


def analyze_event(time_ns, ch0_waveform, ch1_waveform, ref_position):
    """Compute amplitude and rise time from the pedestal-subtracted absolute waveform."""
    pedestal_end = max(1, min(time_ns.size, int((ref_position - 10) / 100.0 * time_ns.size)))
    ped0 = np.mean(ch0_waveform[:pedestal_end])
    ped1 = np.mean(ch1_waveform[:pedestal_end])

    ch0_signal = ch0_waveform - ped0
    ch1_signal = ch1_waveform - ped1
    abs_signal = np.hypot(ch0_signal, ch1_signal)

    peak_index = int(np.argmax(abs_signal))
    amp = float(abs_signal[peak_index])
    peak_time = float(time_ns[peak_index])

    integral_signal = integrate_waveform(time_ns, abs_signal)
    final_value = float(integral_signal[-1]) if integral_signal.size else 0.0
    if not np.isfinite(final_value) or final_value <= 0:
        tau = np.nan
        t10 = np.nan
        t90 = np.nan
    else:
        level10 = 0.10 * final_value
        level90 = 0.90 * final_value
        t10 = interpolate_crossing(time_ns, integral_signal, level10)
        t90 = interpolate_crossing(time_ns, integral_signal, level90)
        tau = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan

    result = {
        "amp": amp,
        "tau": tau,
        "peak_index": peak_index,
        "peak_time": peak_time,
        "t10": t10,
        "t90": t90,
        "ped0": ped0,
        "ped1": ped1,
    }
    return result, abs_signal, integral_signal


def load_waveform_table(filename):
    """Load the dataset, compute event metrics, and return a dataframe and arrays."""
    data = np.load(filename, allow_pickle=True)
    npts_raw = int(data["npts"])
    ref_position = float(data["ref_position"])
    sample_rate = float(data["sample_rate"])

    ch0, ch1, time_ns = bin_waveforms(
        data["ch0"],
        data["ch1"],
        npts_raw,
        sample_rate,
        ref_position,
    )

    nwf = min(ch0.shape[0], ch1.shape[0])
    rows = []
    abs_waveforms = []
    integrated_waveforms = []
    for event_index in range(nwf):
        result, abs_signal, integral_signal = analyze_event(
            time_ns,
            ch0[event_index],
            ch1[event_index],
            ref_position,
        )
        result["event"] = event_index
        rows.append(result)
        abs_waveforms.append(abs_signal)
        integrated_waveforms.append(integral_signal)

    parameters = pd.DataFrame(rows)
    return {
        "filename": filename,
        "time_ns": time_ns,
        "ch0": ch0,
        "ch1": ch1,
        "abs_waveforms": abs_waveforms,
        "integrated_waveforms": integrated_waveforms,
        "parameters": parameters,
        "nwf": nwf,
    }


def _plot_single_waveform_panel(axis, time_ns, signal, event_index, color, ylabel, title):
    axis.plot(time_ns, signal, color=color, linewidth=1.4)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("Time [ns]")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)


def draw_random_dashboard(fig, ax_abs, ax_int, ax_corr, dataset, selected_index, colorbar=None):
    """Draw a dashboard with one standard amp-tau scatter plot and highlighted random points."""
    time_ns = dataset["time_ns"]
    parameters = dataset["parameters"]
    abs_waveforms = dataset["abs_waveforms"]
    integrated_waveforms = dataset["integrated_waveforms"]

    event_ids = np.asarray(selected_index, dtype=int)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, event_ids.size)))

    for axis in (ax_abs, ax_int):
        axis.clear()

    for event_id, color in zip(event_ids, colors):
        abs_signal = abs_waveforms[event_id]
        integral_signal = integrated_waveforms[event_id]
        event_label = f"event {event_id}"
        ax_abs.plot(time_ns, abs_signal * 1e3, color=color, linewidth=1.3, label=event_label)
        ax_int.plot(time_ns, integral_signal * 1e3, color=color, linewidth=1.3, label=event_label)

    ax_abs.set_title("Absolute waveform (pedestal-subtracted)")
    ax_abs.set_xlabel("Time [ns]")
    ax_abs.set_ylabel("|IQ| [mV]")
    ax_abs.grid(alpha=0.3)
    ax_abs.legend(loc="upper right", fontsize=8)

    ax_int.set_title("Integrated absolute waveform")
    ax_int.set_xlabel("Time [ns]")
    ax_int.set_ylabel("Integral [mV·ns]")
    ax_int.grid(alpha=0.3)

    amp = parameters["amp"].to_numpy(dtype=float)
    tau = parameters["tau"].to_numpy(dtype=float)
    finite = np.isfinite(amp) & np.isfinite(tau)

    ax_corr.clear()
    if np.any(finite):
        hist = ax_corr.hist2d(
            amp[finite],
            tau[finite],
            bins=(240, 80),
            cmap="Blues",
            norm=matplotlib.colors.LogNorm(),
        )
        ax_corr.scatter(
            amp[event_ids],
            tau[event_ids],
            c=colors,
            edgecolors="black",
            linewidths=0.9,
            s=80,
            zorder=3,
        )
        for event_id, color in zip(event_ids, colors):
            ax_corr.annotate(
                f"{event_id}",
                (amp[event_id], tau[event_id]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
                weight="bold",
            )
        ax_corr.set_xscale("log")
        ax_corr.set_title("amp vs tau")
        ax_corr.set_xlabel("amp [a.u.] (log scale)")
        ax_corr.set_ylabel("tau [ns]")
        ax_corr.grid(alpha=0.3)

        if colorbar is not None:
            colorbar.update_normal(hist[3])
            colorbar.set_label("counts")
        else:
            colorbar = fig.colorbar(hist[3], ax=ax_corr, label="counts")
    else:
        ax_corr.text(0.5, 0.5, "No valid events", ha="center", va="center", transform=ax_corr.transAxes)
        ax_corr.set_title("amp vs tau")
        ax_corr.set_xlabel("amp [a.u.] (log scale)")
        ax_corr.set_ylabel("tau [ns]")
        if colorbar is not None:
            colorbar.remove()
            colorbar = None

    fig.suptitle("Random event check: amp and tau from pedestal-subtracted absolute waveform")
    return colorbar


def randomize_events(rng, dataset):
    """Return a new random selection of 5 events."""
    return rng.choice(dataset["nwf"], size=min(RANDOM_EVENT_COUNT, dataset["nwf"]), replace=False)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 alpha_event_check.py input.npz")

    filename = sys.argv[1]
    dataset = load_waveform_table(filename)
    parameters = dataset["parameters"]
    parameters["amp"] = pd.to_numeric(parameters["amp"], errors="coerce")
    parameters["tau"] = pd.to_numeric(parameters["tau"], errors="coerce")

    if dataset["nwf"] == 0:
        raise SystemExit(f"No events found in {filename}")

    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": [1.8, 1.8, 1.4]},
    )
    selected = randomize_events(rng, dataset)
    colorbar = None
    colorbar = draw_random_dashboard(fig, axes[0], axes[1], axes[2], dataset, selected, colorbar)

    button_ax = fig.add_axes([0.62, 0.88, 0.22, 0.06])
    random_button = Button(button_ax, "Randomize 5 events")

    def update_random(_event):
        nonlocal colorbar
        selected[:] = randomize_events(rng, dataset)
        colorbar = draw_random_dashboard(fig, axes[0], axes[1], axes[2], dataset, selected, colorbar)
        fig.canvas.draw_idle()

    random_button.on_clicked(update_random)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


