from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================
INPUT_DIR = Path("/Volumes/NO NAME/data/20260527")
BASE_DIR = Path("./data/20260527")

TARGET_Z = 7.5
TARGET_X = 3.4
TARGET_FREQS_GHZ = [5.451, 5.461, 5.476, 5.491, 5.501]

# pedestal を取る波形先頭部のサンプル数
# パルスがこの範囲に入らない値にする
N_PRE = 500

# ch0 を反転したIQ座標系を使う場合だけ True
FLIP_CH0 = False

# 2.5 GS/s の場合。CSV出力中の peak_time_ns 用
SAMPLE_RATE_HZ = 2.5e9

OUT_DIR = BASE_DIR / "rf_iq_base_peak"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# データ読み込み
# ============================================================
def folder_name(freq_ghz, z_mm, x_mm):
    """無印フォルダ名を完全一致で作る。"""
    return f"{freq_ghz:.3f}GHz_z={z_mm:.1f}mm_x={x_mm:.1f}mm"


def find_channel_keys(keys):
    """npz 内の ch0 / ch1 キーを探す。"""
    candidates0 = ["ch0", "channel0", "wave0", "data0", "I", "i"]
    candidates1 = ["ch1", "channel1", "wave1", "data1", "Q", "q"]

    key0 = next((k for k in candidates0 if k in keys), None)
    key1 = next((k for k in candidates1 if k in keys), None)

    if key0 is None or key1 is None:
        raise KeyError(
            "ch0 / ch1 のキーを特定できません。\n"
            f"利用可能なキー: {keys}"
        )

    return key0, key1


def ensure_event_sample_shape(a):
    """
    shape を (N_event, N_sample) にそろえる。
    通常は N_event << N_sample を想定。
    """
    a = np.asarray(a)

    if a.ndim == 1:
        return a[None, :]

    if a.ndim != 2:
        raise ValueError(f"想定外の配列次元: shape={a.shape}")

    if a.shape[0] > a.shape[1]:
        return a.T

    return a


def load_all_waveforms(folder):
    """フォルダ内の全 npz を読み、ch0/ch1 を結合する。"""
    files = sorted(folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"npz が見つかりません: {folder}")

    ch0_list = []
    ch1_list = []

    for fp in files:
        with np.load(fp, allow_pickle=True) as d:
            key0, key1 = find_channel_keys(list(d.keys()))

            ch0 = ensure_event_sample_shape(d[key0])
            ch1 = ensure_event_sample_shape(d[key1])

            if ch0.shape != ch1.shape:
                raise ValueError(
                    f"ch0/ch1 のshapeが一致しません: {fp.name}\n"
                    f"ch0={ch0.shape}, ch1={ch1.shape}"
                )

            ch0_list.append(ch0)
            ch1_list.append(ch1)

    return (
        np.concatenate(ch0_list, axis=0),
        np.concatenate(ch1_list, axis=0),
    )


# ============================================================
# 各周波数の base 点・peak 点を求める
# ============================================================
results = []

for freq in TARGET_FREQS_GHZ:
    folder = INPUT_DIR / folder_name(freq, TARGET_Z, TARGET_X)

    if not folder.is_dir():
        print(f"[skip] フォルダがありません: {folder}")
        continue

    ch0, ch1 = load_all_waveforms(folder)

    if FLIP_CH0:
        ch0 = -ch0

    if ch0.shape[1] <= N_PRE:
        raise ValueError(
            f"{folder.name}: N_PRE={N_PRE} が波形長 {ch0.shape[1]} 以上です。"
        )

    # 各イベントの pedestal
    ped0_each = np.mean(ch0[:, :N_PRE], axis=1)
    ped1_each = np.mean(ch1[:, :N_PRE], axis=1)

    # 周波数ごとの base 点
    base0 = np.mean(ped0_each)
    base1 = np.mean(ped1_each)

    # pedestal をイベントごとに引いて平均波形を作る
    mean_dev0 = np.mean(ch0 - ped0_each[:, None], axis=0)
    mean_dev1 = np.mean(ch1 - ped1_each[:, None], axis=0)

    # 同じ時刻の ch0/ch1 からIQ距離を定義
    # ch0 と ch1 の別々の最大点を取らない点が重要
    iq_distance = np.hypot(mean_dev0, mean_dev1)

    # pedestal領域を除外して、base点から最も離れた時刻をpeakとする
    peak_index = N_PRE + np.argmax(iq_distance[N_PRE:])

    peak0 = base0 + mean_dev0[peak_index]
    peak1 = base1 + mean_dev1[peak_index]

    results.append(
        {
            "freq_GHz": freq,
            "N_event": len(ch0),
            "base_ch0": base0,
            "base_ch1": base1,
            "peak_ch0": peak0,
            "peak_ch1": peak1,
            "delta_ch0": peak0 - base0,
            "delta_ch1": peak1 - base1,
            "delta_iq": np.hypot(peak0 - base0, peak1 - base1),
            "peak_index": int(peak_index),
            "peak_time_ns": peak_index / SAMPLE_RATE_HZ * 1e9,
        }
    )

results.sort(key=lambda r: r["freq_GHz"])

if not results:
    raise RuntimeError("対象データを一つも読み込めませんでした。")


# ============================================================
# IQ平面プロット
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

for i, r in enumerate(results):
    # 自動色を使い、同じ周波数のbase/peak/線を同色にする
    line, = ax.plot(
        [r["base_ch0"], r["peak_ch0"]],
        [r["base_ch1"], r["peak_ch1"]],
        "-",
        lw=1.8,
    )
    c = line.get_color()

    # base: 白抜き丸
    ax.scatter(
        r["base_ch0"],
        r["base_ch1"],
        s=80,
        marker="o",
        facecolors="white",
        edgecolors=c,
        linewidths=2.0,
        zorder=3,
        label="base (pedestal)" if i == 0 else None,
    )

    # peak: 塗りつぶし三角
    ax.scatter(
        r["peak_ch0"],
        r["peak_ch1"],
        s=90,
        marker="^",
        color=c,
        zorder=4,
        label="peak" if i == 0 else None,
    )

    # 周波数ラベルはpeak側へ表示
    ax.annotate(
        f"{r['freq_GHz']:.3f} GHz",
        xy=(r["peak_ch0"], r["peak_ch1"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color=c,
    )

ax.set_xlabel("ch0")
ax.set_ylabel("ch1")
ax.set_title(
    f"IQ base and pulse peak points\n"
    f"z = {TARGET_Z:.1f} mm, x = {TARGET_X:.1f} mm"
)

ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="best")
fig.tight_layout()

png_path = OUT_DIR / "iq_base_peak_z7p5mm_x3p4mm.png"
pdf_path = OUT_DIR / "iq_base_peak_z7p5mm_x3p4mm.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 数値もCSVに保存
# ============================================================
csv_path = OUT_DIR / "iq_base_peak_z7p5mm_x3p4mm.csv"

fieldnames = [
    "freq_GHz",
    "N_event",
    "base_ch0",
    "base_ch1",
    "peak_ch0",
    "peak_ch1",
    "delta_ch0",
    "delta_ch1",
    "delta_iq",
    "peak_index",
    "peak_time_ns",
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n保存完了:")
print(f"  {png_path}")
print(f"  {pdf_path}")
print(f"  {csv_path}")

print("\n--- summary ---")
for r in results:
    print(
        f"{r['freq_GHz']:.3f} GHz : "
        f"base=({r['base_ch0']:.6g}, {r['base_ch1']:.6g}), "
        f"peak=({r['peak_ch0']:.6g}, {r['peak_ch1']:.6g}), "
        f"|ΔIQ|={r['delta_iq']:.6g}"
    )