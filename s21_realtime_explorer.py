#!/usr/bin/env python3
"""
s21_realtime_explorer.py

Matplotlib slider GUI for exploring a notch-type KID resonator in real time.

Run from a normal terminal:
    python s21_realtime_explorer.py

Controls
--------
f_r center [GHz] : reference resonance frequency
Δf_r [MHz]       : instantaneous resonance shift. For the usual KID behavior,
                    temperature increase corresponds to Δf_r < 0.
f_tone [GHz]     : fixed readout frequency
f_r span [MHz]   : extent of the fixed-tone trajectory shown in the IQ panel
Ql, Qc, phi      : resonator parameters
a, alpha, tau    : environmental gain / phase / cable-delay parameters

The IQ panel shows:
  - the frequency-sweep S21 locus at the current f_r
  - the locus followed by a fixed f_tone while f_r is varied
  - the current fixed-tone point
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


# ============================================================
# Initial values: 5.56 K fit as a convenient starting point.
# Units: frequency = GHz, tau = ns
# ============================================================

INITIAL = {
    "fr_center": 5.479238,
    "df_mhz": 0.0,
    "f_tone": 5.476000,
    "fr_span_mhz": 5.0,
    "Ql": 452.6852,
    "Qc": 488.4716,
    "phi": 0.0,
    "a": 1.0,
    "alpha": 0.0,
    "tau_ns": 0.0,
}

FREQ_MIN_GHZ = 5.445
FREQ_MAX_GHZ = 5.510
N_FREQ = 2500
SAVE_JSON = Path("s21_realtime_parameters.json")


def s21_notch(par: dict[str, float], f_ghz: np.ndarray) -> np.ndarray:
    """Notch S21 model. Frequency in GHz; tau in ns."""
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
    return env * (1.0 - res)


def qi_from_ql_qc(ql: float, qc: float) -> float:
    """Return Qi from 1/Ql = 1/Qi + 1/Qc."""
    denom = 1.0 / ql - 1.0 / abs(qc)
    return 1.0 / denom if denom > 0 else np.nan


class S21Explorer:
    def __init__(self) -> None:
        self.f = np.linspace(FREQ_MIN_GHZ, FREQ_MAX_GHZ, N_FREQ)
        self.playing = False
        self.play_start_time = None

        self.fig = plt.figure(figsize=(15, 9))
        gs = self.fig.add_gridspec(
            2, 3,
            left=0.06, right=0.97, top=0.91, bottom=0.34,
            wspace=0.34, hspace=0.42,
        )

        self.ax_iq = self.fig.add_subplot(gs[:, 0])
        self.ax_mag = self.fig.add_subplot(gs[0, 1:])
        self.ax_phase = self.fig.add_subplot(gs[1, 1:])

        # Slider axes: two columns, five rows
        slider_left, slider_right = 0.08, 0.57
        slider_w, slider_h = 0.32, 0.025
        y0, dy = 0.27, 0.047

        axes = {}
        for col, x in enumerate([slider_left, slider_right]):
            for row in range(5):
                axes[(col, row)] = self.fig.add_axes(
                    [x, y0 - row * dy, slider_w, slider_h]
                )

        self.s_fr = Slider(
            axes[(0, 0)], r"$f_{r,\mathrm{center}}$ [GHz]",
            5.450, 5.505, valinit=INITIAL["fr_center"], valstep=1e-6
        )
        self.s_df = Slider(
            axes[(0, 1)], r"$\Delta f_r$ [MHz]",
            -15.0, 15.0, valinit=INITIAL["df_mhz"], valstep=0.001
        )
        self.s_tone = Slider(
            axes[(0, 2)], r"$f_{\mathrm{tone}}$ [GHz]",
            5.450, 5.505, valinit=INITIAL["f_tone"], valstep=1e-6
        )
        self.s_span = Slider(
            axes[(0, 3)], r"$f_r$ path half-span [MHz]",
            0.05, 20.0, valinit=INITIAL["fr_span_mhz"], valstep=0.01
        )
        self.s_ql = Slider(
            axes[(0, 4)], r"$Q_l$",
            50.0, 2000.0, valinit=INITIAL["Ql"], valstep=0.1
        )

        self.s_qc = Slider(
            axes[(1, 0)], r"$|Q_c|$",
            50.0, 4000.0, valinit=INITIAL["Qc"], valstep=0.1
        )
        self.s_phi = Slider(
            axes[(1, 1)], r"$\phi$ [rad]",
            -np.pi, np.pi, valinit=INITIAL["phi"], valstep=0.001
        )
        self.s_a = Slider(
            axes[(1, 2)], r"$a$",
            0.05, 2.0, valinit=INITIAL["a"], valstep=0.001
        )
        self.s_alpha = Slider(
            axes[(1, 3)], r"$\alpha$ [rad]",
            -np.pi, np.pi, valinit=INITIAL["alpha"], valstep=0.001
        )
        self.s_tau = Slider(
            axes[(1, 4)], r"$\tau$ [ns]",
            -1000.0, 1000.0, valinit=INITIAL["tau_ns"], valstep=0.1
        )

        self.sliders = [
            self.s_fr, self.s_df, self.s_tone, self.s_span, self.s_ql,
            self.s_qc, self.s_phi, self.s_a, self.s_alpha, self.s_tau,
        ]
        for slider in self.sliders:
            slider.on_changed(self.update)

        self.ax_reset = self.fig.add_axes([0.78, 0.05, 0.09, 0.045])
        self.ax_play = self.fig.add_axes([0.66, 0.05, 0.09, 0.045])
        self.ax_save = self.fig.add_axes([0.89, 0.05, 0.08, 0.045])

        self.b_reset = Button(self.ax_reset, "Reset")
        self.b_play = Button(self.ax_play, "Play")
        self.b_save = Button(self.ax_save, "Save JSON")
        self.b_reset.on_clicked(self.reset)
        self.b_play.on_clicked(self.toggle_play)
        self.b_save.on_clicked(self.save_json)

        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.animate)

        self.draw_initial_artists()
        self.update(None)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def get_parameters(self) -> dict[str, float]:
        return {
            "fr_ghz": self.s_fr.val + self.s_df.val * 1e-3,
            "Ql": self.s_ql.val,
            "Qc": self.s_qc.val,
            "phi": self.s_phi.val,
            "a": self.s_a.val,
            "alpha": self.s_alpha.val,
            "tau_ns": self.s_tau.val,
        }

    def draw_initial_artists(self) -> None:
        (self.line_iq,) = self.ax_iq.plot(
            [], [], "-", label=r"frequency sweep $S_{21}(f)$"
        )
        (self.line_path,) = self.ax_iq.plot(
            [], [], "--", label=r"fixed $f_{\rm tone}$; scan $f_r$"
        )
        (self.point_tone,) = self.ax_iq.plot(
            [], [], "o", label=r"current fixed-tone point"
        )
        (self.point_center,) = self.ax_iq.plot(
            [], [], "s", label=r"$\Delta f_r=0$"
        )
        self.text_iq = self.ax_iq.text(
            0.03, 0.97, "",
            transform=self.ax_iq.transAxes,
            va="top", fontsize=9,
            bbox={"boxstyle": "round", "alpha": 0.85},
        )

        self.ax_iq.set_xlabel(r"Re[$S_{21}$]")
        self.ax_iq.set_ylabel(r"Im[$S_{21}$]")
        self.ax_iq.set_title("Complex S21 / IQ plane")
        self.ax_iq.grid(True)
        self.ax_iq.legend(fontsize=8, loc="best")
        self.ax_iq.set_aspect("equal", adjustable="box")

        (self.line_mag,) = self.ax_mag.plot([], [], "-")
        self.vline_tone_mag = self.ax_mag.axvline(
            0.0, ls="--", label=r"$f_{\rm tone}$"
        )
        self.vline_fr_mag = self.ax_mag.axvline(
            0.0, ls=":", label=r"current $f_r$"
        )
        self.ax_mag.set_xlabel("frequency [GHz]")
        self.ax_mag.set_ylabel(r"$|S_{21}|$")
        self.ax_mag.set_title("Magnitude")
        self.ax_mag.grid(True)
        self.ax_mag.legend(fontsize=8)

        (self.line_phase,) = self.ax_phase.plot([], [], "-")
        self.vline_tone_phase = self.ax_phase.axvline(
            0.0, ls="--", label=r"$f_{\rm tone}$"
        )
        self.vline_fr_phase = self.ax_phase.axvline(
            0.0, ls=":", label=r"current $f_r$"
        )
        self.ax_phase.set_xlabel("frequency [GHz]")
        self.ax_phase.set_ylabel(r"arg($S_{21}$) [rad]")
        self.ax_phase.set_title("Phase")
        self.ax_phase.grid(True)
        self.ax_phase.legend(fontsize=8)

    def update(self, _event) -> None:
        par = self.get_parameters()
        z = s21_notch(par, self.f)

        f_tone = self.s_tone.val
        z_tone = s21_notch(par, np.array([f_tone]))[0]

        # Fixed-tone trajectory while fr is varied around the center.
        fr_center = self.s_fr.val
        span_ghz = self.s_span.val * 1e-3
        fr_path = fr_center + np.linspace(-span_ghz, span_ghz, 700)

        z_path = np.empty(len(fr_path), dtype=complex)
        for i, fr in enumerate(fr_path):
            temp_par = par.copy()
            temp_par["fr_ghz"] = fr
            z_path[i] = s21_notch(temp_par, np.array([f_tone]))[0]

        center_par = par.copy()
        center_par["fr_ghz"] = fr_center
        z_center = s21_notch(center_par, np.array([f_tone]))[0]

        self.line_iq.set_data(z.real, z.imag)
        self.line_path.set_data(z_path.real, z_path.imag)
        self.point_tone.set_data([z_tone.real], [z_tone.imag])
        self.point_center.set_data([z_center.real], [z_center.imag])

        self.line_mag.set_data(self.f, np.abs(z))
        self.line_phase.set_data(self.f, np.unwrap(np.angle(z)))

        for line in [self.vline_tone_mag, self.vline_tone_phase]:
            line.set_xdata([f_tone, f_tone])
        for line in [self.vline_fr_mag, self.vline_fr_phase]:
            line.set_xdata([par["fr_ghz"], par["fr_ghz"]])

        self.ax_mag.relim()
        self.ax_mag.autoscale_view(scalex=False, scaley=True)
        self.ax_phase.relim()
        self.ax_phase.autoscale_view(scalex=False, scaley=True)

        all_z = np.concatenate([z, z_path, np.array([z_tone, z_center])])
        xmin, xmax = np.min(all_z.real), np.max(all_z.real)
        ymin, ymax = np.min(all_z.imag), np.max(all_z.imag)
        dx = max(xmax - xmin, 1e-9)
        dy = max(ymax - ymin, 1e-9)
        pad = 0.12 * max(dx, dy)
        self.ax_iq.set_xlim(xmin - pad, xmax + pad)
        self.ax_iq.set_ylim(ymin - pad, ymax + pad)

        qi = qi_from_ql_qc(par["Ql"], par["Qc"])
        detuning_mhz = (f_tone - par["fr_ghz"]) * 1e3
        qi_text = f"{qi:.1f}" if np.isfinite(qi) else "not physical"

        self.text_iq.set_text(
            "\n".join(
                [
                    rf"$f_r$ = {par['fr_ghz']:.6f} GHz",
                    rf"$f_{{tone}}-f_r$ = {detuning_mhz:+.3f} MHz",
                    rf"$Q_i$ = {qi_text}",
                    r"$T\uparrow$ usually means $\Delta f_r<0$",
                ]
            )
        )

        self.fig.suptitle(
            "Real-time notch S21 explorer — change parameters and watch the IQ trajectory",
            fontsize=13,
        )
        self.fig.canvas.draw_idle()

    def reset(self, _event) -> None:
        for slider in self.sliders:
            slider.reset()

    def toggle_play(self, _event) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_start_time = time.perf_counter()
            self.b_play.label.set_text("Stop")
            self.timer.start()
        else:
            self.b_play.label.set_text("Play")
            self.timer.stop()

    def animate(self) -> None:
        if not self.playing:
            return

        # Oscillation of fr around its center.
        # During the negative half-cycle fr decreases: usual direction of T increase.
        elapsed = time.perf_counter() - self.play_start_time
        span = self.s_span.val
        self.s_df.set_val(-span * np.sin(2.0 * np.pi * elapsed / 2.0))

    def save_json(self, _event) -> None:
        par = self.get_parameters()
        payload = {
            "fr_ghz": par["fr_ghz"],
            "fr_center_ghz": self.s_fr.val,
            "delta_fr_mhz": self.s_df.val,
            "f_tone_ghz": self.s_tone.val,
            "fr_path_half_span_mhz": self.s_span.val,
            "Ql": par["Ql"],
            "Qc": par["Qc"],
            "Qi": qi_from_ql_qc(par["Ql"], par["Qc"]),
            "phi": par["phi"],
            "a": par["a"],
            "alpha": par["alpha"],
            "tau_ns": par["tau_ns"],
        }
        SAVE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved: {SAVE_JSON.resolve()}")

    def on_close(self, _event) -> None:
        self.timer.stop()


if __name__ == "__main__":
    explorer = S21Explorer()
    plt.show()
