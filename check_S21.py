import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# f(z) = 1 - d / (1 + i z)
# ============================================================
def f_of_z(z, d):
    return 1 - d / (1 + 1j * z)


# z を掃引する範囲
Z_RANGE = 5.0
z_array = np.linspace(-Z_RANGE, Z_RANGE, 2000)

# 初期値
d_re0 = 0.8
d_im0 = 0.0
z0 = 0.0

# 初期計算
d0 = d_re0 + 1j * d_im0
f_array0 = f_of_z(z_array, d0)
f_selected0 = f_of_z(z0, d0)

# ============================================================
# 図の作成
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(bottom=0.28)

ax_left, ax_iq = axes

# 左：Re, Im, |f|
line_re, = ax_left.plot(z_array, f_array0.real, label="Re[f(z)]")
line_im, = ax_left.plot(z_array, f_array0.imag, label="Im[f(z)]")
line_abs, = ax_left.plot(z_array, np.abs(f_array0), "--", label="|f(z)|")

vline_z = ax_left.axvline(z0, linestyle=":")
point_re, = ax_left.plot([z0], [f_selected0.real], "o")
point_im, = ax_left.plot([z0], [f_selected0.imag], "o")

ax_left.set_xlabel("z")
ax_left.set_ylabel("f(z)")
ax_left.set_title(r"$f(z)=1-\frac{d}{1+i z}$")
ax_left.grid(True)
ax_left.legend()

# 右：IQ 平面
line_iq, = ax_iq.plot(f_array0.real, f_array0.imag, label="IQ trajectory")
point_iq, = ax_iq.plot(
    [f_selected0.real],
    [f_selected0.imag],
    "o",
    label=f"z = {z0:.2f}",
)

ax_iq.axhline(0, linewidth=0.8)
ax_iq.axvline(0, linewidth=0.8)
ax_iq.set_xlabel("I = Re[f(z)]")
ax_iq.set_ylabel("Q = Im[f(z)]")
ax_iq.set_title("IQ plane")
ax_iq.grid(True)
ax_iq.set_aspect("equal", adjustable="box")
ax_iq.legend()

# 表示用テキスト
info_text = fig.text(
    0.50,
    0.03,
    "",
    ha="center",
    fontsize=11,
)

# ============================================================
# スライダー
# ============================================================
ax_d_re = fig.add_axes([0.15, 0.17, 0.70, 0.03])
ax_d_im = fig.add_axes([0.15, 0.12, 0.70, 0.03])
ax_z = fig.add_axes([0.15, 0.07, 0.70, 0.03])

slider_d_re = Slider(ax_d_re, "Re[d]", -3.0, 3.0, valinit=d_re0, valstep=0.01)
slider_d_im = Slider(ax_d_im, "Im[d]", -3.0, 3.0, valinit=d_im0, valstep=0.01)
slider_z = Slider(ax_z, "z", -Z_RANGE, Z_RANGE, valinit=z0, valstep=0.01)


def update(_):
    d_re = slider_d_re.val
    d_im = slider_d_im.val
    z_selected = slider_z.val

    d = d_re + 1j * d_im
    f_array = f_of_z(z_array, d)
    f_selected = f_of_z(z_selected, d)

    # 左図を更新
    line_re.set_ydata(f_array.real)
    line_im.set_ydata(f_array.imag)
    line_abs.set_ydata(np.abs(f_array))

    vline_z.set_xdata([z_selected, z_selected])
    point_re.set_data([z_selected], [f_selected.real])
    point_im.set_data([z_selected], [f_selected.imag])

    # IQ 図を更新
    line_iq.set_data(f_array.real, f_array.imag)
    point_iq.set_data([f_selected.real], [f_selected.imag])
    point_iq.set_label(f"z = {z_selected:.2f}")

    # 見やすい範囲に自動調整
    ax_left.relim()
    ax_left.autoscale_view()

    ax_iq.relim()
    ax_iq.autoscale_view()
    ax_iq.set_aspect("equal", adjustable="box")

    # 凡例と数値
    ax_iq.legend()

    info_text.set_text(
        f"d = {d_re:+.3f} {d_im:+.3f} i    |    "
        f"z = {z_selected:+.3f}    |    "
        f"f(z) = {f_selected.real:+.5f} {f_selected.imag:+.5f} i    |    "
        f"|f(z)| = {abs(f_selected):.5f}"
    )

    fig.canvas.draw_idle()


slider_d_re.on_changed(update)
slider_d_im.on_changed(update)
slider_z.on_changed(update)

update(None)
plt.show()