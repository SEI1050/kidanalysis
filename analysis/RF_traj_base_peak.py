from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 設定
# ============================================================
BASE_DIR = Path("./data/20260527")
INPUT_DIR = Path("/Volumes/NO NAME/data/20260527")

TARGET_Z = 7.5
TARGET_X = 3.4
TARGET_FREQS_GHZ = [5.451, 5.461, 5.476, 5.491, 5.501]

N_PRE = 500
SAMPLE_RATE_HZ = 2.5e9

# 軌跡として表示する時間範囲 [ns]
# パルスが約470 ns付近から始まっているので、少し前から表示
PLOT_TMIN_NS = 400
PLOT_TMAX_NS = 1500

# ch0の符号を反転している解析系なら True
FLIP_CH0 = False

OUT_DIR = BASE_DIR / "rf_iq_trajectory"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 前のコードと同じ読み込み関数
# ============================================================
def folder_name(freq_ghz, z_mm, x_mm):
    return f"{freq_ghz:.3f}GHz_z={z_mm:.1f}mm_x={x_mm:.1f}mm"


def find_channel_keys(keys):
    candidates0 = ["ch0", "channel0", "wave0", "data0", "I", "i"]
    candidates1 = ["ch1", "channel1", "wave1", "data1", "Q", "q"]

    key0 = next((k for k in candidates0 if k in keys), None)
    key1 = next((k for k in candidates1 if k in keys), None)

    if key0 is None or key1 is None:
        raise KeyError(f"ch0/ch1 key not found. keys={keys}")

    return key0, key1


def ensure_event_sample_shape(a):
    a = np.asarray(a)

    if a.ndim == 1:
        return a[None, :]

    if a.ndim != 2:
        raise ValueError(f"Unexpected shape: {a.shape}")

    if a.shape[0] > a.shape[1]:
        return a.T

    return a


def load_all_waveforms(folder):
    files = sorted(folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No npz files: {folder}")

    ch0_list = []
    ch1_list = []

    for fp in files:
        with np.load(fp, allow_pickle=True) as d:
            key0, key1 = find_channel_keys(list(d.keys()))

            ch0 = ensure_event_sample_shape(d[key0])
            ch1 = ensure_event_sample_shape(d[key1])

            if ch0.shape != ch1.shape:
                raise ValueError(
                    f"shape mismatch in {fp.name}: {ch0.shape}, {ch1.shape}"
                )

            ch0_list.append(ch0)
            ch1_list.append(ch1)

    return (
        np.concatenate(ch0_list, axis=0),
        np.concatenate(ch1_list, axis=0),
    )


# ============================================================
# 時刻色付きの線を描く関数
# ============================================================
def add_colored_trajectory(ax, x, y, t_ns, norm, cmap="viridis", lw=2.5):
    """
    x, y: IQ軌跡
    t_ns: 各点の時刻
    """
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=lw,
    )
    lc.set_array(t_ns[:-1])

    ax.add_collection(lc)
    ax.autoscale_view()

    return lc


# ============================================================
# 各周波数について平均IQ軌跡を作る
# ============================================================
results = []

for freq in TARGET_FREQS_GHZ:
    folder = INPUT_DIR / folder_name(freq, TARGET_Z, TARGET_X)

    if not folder.is_dir():
        print(f"[skip] folder not found: {folder}")
        continue

    ch0, ch1 = load_all_waveforms(folder)

    if FLIP_CH0:
        ch0 = -ch0

    n_event, n_sample = ch0.shape
    time_ns = np.arange(n_sample) / SAMPLE_RATE_HZ * 1e9

    # 各イベントのpedestal
    ped0_each = np.mean(ch0[:, :N_PRE], axis=1)
    ped1_each = np.mean(ch1[:, :N_PRE], axis=1)

    # 周波数ごとの平均base点
    base0 = np.mean(ped0_each)
    base1 = np.mean(ped1_each)

    # 各イベントごとにbaselineを引く
    dev0 = ch0 - ped0_each[:, None]
    dev1 = ch1 - ped1_each[:, None]

    # 平均IQ応答
    mean_dev0 = np.mean(dev0, axis=0)
    mean_dev1 = np.mean(dev1, axis=0)

    # raw IQ座標に戻した平均軌跡
    mean_ch0 = base0 + mean_dev0
    mean_ch1 = base1 + mean_dev1

    # IQ距離最大点
    iq_dist = np.hypot(mean_dev0, mean_dev1)
    peak_index = N_PRE + np.argmax(iq_dist[N_PRE:])

    # 表示区間
    mask = (time_ns >= PLOT_TMIN_NS) & (time_ns <= PLOT_TMAX_NS)

    results.append(
        {
            "freq": freq,
            "time_ns": time_ns,
            "mask": mask,
            "base0": base0,
            "base1": base1,
            "mean_ch0": mean_ch0,
            "mean_ch1": mean_ch1,
            "mean_dev0": mean_dev0,
            "mean_dev1": mean_dev1,
            "peak_index": peak_index,
            "peak0": mean_ch0[peak_index],
            "peak1": mean_ch1[peak_index],
            "peak_dev0": mean_dev0[peak_index],
            "peak_dev1": mean_dev1[peak_index],
            "peak_time_ns": time_ns[peak_index],
        }
    )

if not results:
    raise RuntimeError("対象データを読み込めませんでした。")


# 色の基準は全パネル共通
norm = Normalize(vmin=PLOT_TMIN_NS, vmax=PLOT_TMAX_NS)
sm = ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])


# ============================================================
# 1. 各RF周波数のraw IQ軌跡を別パネルで描く
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for ax, r in zip(axes, results):
    m = r["mask"]

    add_colored_trajectory(
        ax,
        r["mean_ch0"][m],
        r["mean_ch1"][m],
        r["time_ns"][m],
        norm=norm,
    )

    # base点
    ax.scatter(
        r["base0"],
        r["base1"],
        s=90,
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        marker="o",
        zorder=5,
        label="base",
    )

    # peak点
    ax.scatter(
        r["peak0"],
        r["peak1"],
        s=120,
        color="black",
        marker="*",
        zorder=6,
        label="peak",
    )

    # 開始・終了時刻の位置を小さく表示
    first_idx = np.where(m)[0][0]
    last_idx = np.where(m)[0][-1]

    ax.scatter(
        r["mean_ch0"][first_idx],
        r["mean_ch1"][first_idx],
        s=30,
        color="black",
        marker="s",
        zorder=5,
    )

    ax.scatter(
        r["mean_ch0"][last_idx],
        r["mean_ch1"][last_idx],
        s=30,
        color="black",
        marker="x",
        zorder=5,
    )

    ax.set_title(
        f"{r['freq']:.3f} GHz\npeak: {r['peak_time_ns']:.1f} ns"
    )
    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

# 余った6枚目のパネルを消す
for ax in axes[len(results):]:
    ax.axis("off")

fig.colorbar(
    sm,
    ax=axes[:len(results)],
    label="Time [ns]",
    shrink=0.82,
)

fig.suptitle(
    f"Average IQ pulse trajectories\n"
    f"z = {TARGET_Z:.1f} mm, x = {TARGET_X:.1f} mm",
    fontsize=16,
)

fig.tight_layout(rect=[0, 0, 0.92, 0.95])

fig.savefig(
    OUT_DIR / "iq_trajectory_by_frequency.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "iq_trajectory_by_frequency.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 2. pedestalを原点にそろえたIQ軌跡を一枚に重ねる
# ============================================================
fig, ax = plt.subplots(figsize=(9, 8))

for r in results:
    m = r["mask"]

    add_colored_trajectory(
        ax,
        r["mean_dev0"][m],
        r["mean_dev1"][m],
        r["time_ns"][m],
        norm=norm,
        lw=2.0,
    )

    # peak位置
    ax.scatter(
        r["peak_dev0"],
        r["peak_dev1"],
        s=80,
        marker="*",
        color="black",
        zorder=5,
    )

    ax.annotate(
        f"{r['freq']:.3f} GHz",
        xy=(r["peak_dev0"], r["peak_dev1"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
    )

# pedestalを原点として表示
ax.scatter(
    0,
    0,
    s=100,
    facecolors="white",
    edgecolors="black",
    linewidths=1.5,
    marker="o",
    zorder=6,
    label="pedestal-subtracted base",
)

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.axvline(0, color="black", lw=0.8, alpha=0.5)

ax.set_xlabel(r"$\Delta$ ch0")
ax.set_ylabel(r"$\Delta$ ch1")
ax.set_title(
    "Average IQ trajectories with pedestal aligned at origin\n"
    f"z = {TARGET_Z:.1f} mm, x = {TARGET_X:.1f} mm"
)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")
ax.legend()

fig.colorbar(sm, ax=ax, label="Time [ns]")
fig.tight_layout()

fig.savefig(
    OUT_DIR / "iq_trajectory_pedestal_aligned.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "iq_trajectory_pedestal_aligned.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 3. 数値一覧
# ============================================================
print("\n--- IQ trajectory summary ---")
for r in results:
    print(
        f"{r['freq']:.3f} GHz | "
        f"peak time = {r['peak_time_ns']:.1f} ns | "
        f"base = ({r['base0']:.6g}, {r['base1']:.6g}) | "
        f"peak-base = ({r['peak_dev0']:.6g}, {r['peak_dev1']:.6g})"
    )

print(f"\nSaved to: {OUT_DIR}")