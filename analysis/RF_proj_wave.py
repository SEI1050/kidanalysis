from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================
BASE_DIR = Path("./data/20260527")
INPUT_DIR = Path("/Volumes/NO NAME/data/20260527")

TARGET_Z = 7.5
TARGET_X = 3.4
TARGET_FREQS_GHZ = [5.451, 5.461, 5.476, 5.491, 5.501]

# pedestal 算出に使う先頭サンプル数
N_PRE = 500

# peak探索領域
# パルスが出る時刻範囲が分かっているなら、適宜狭める。
PEAK_SEARCH_START = N_PRE
PEAK_SEARCH_STOP = None  # None なら波形末尾まで使う

# 2.5 GS/s の場合
SAMPLE_RATE_HZ = 2.5e9

# 必要なら ch0 を反転
FLIP_CH0 = False

OUT_DIR = BASE_DIR / "rf_iq_projection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 読み込み関数
# ============================================================
def folder_name(freq_ghz, z_mm, x_mm):
    # _second, _third, _fourth を除いた無印フォルダだけを完全一致で指定
    return f"{freq_ghz:.3f}GHz_z={z_mm:.1f}mm_x={x_mm:.1f}mm"


def find_channel_keys(keys):
    candidates0 = ["ch0", "channel0", "wave0", "data0", "I", "i"]
    candidates1 = ["ch1", "channel1", "wave1", "data1", "Q", "q"]

    key0 = next((k for k in candidates0 if k in keys), None)
    key1 = next((k for k in candidates1 if k in keys), None)

    if key0 is None or key1 is None:
        raise KeyError(
            "ch0/ch1 キーが見つかりません。\n"
            f"利用可能なキー: {keys}"
        )

    return key0, key1


def ensure_event_sample_shape(a):
    """
    shape を (N_event, N_sample) にそろえる。
    """
    a = np.asarray(a)

    if a.ndim == 1:
        return a[None, :]

    if a.ndim != 2:
        raise ValueError(f"想定外の次元です: {a.shape}")

    # 通常は N_event << N_sample と仮定
    if a.shape[0] > a.shape[1]:
        return a.T

    return a


def load_all_waveforms(folder):
    npz_files = sorted(folder.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"npzがありません: {folder}")

    ch0_list = []
    ch1_list = []

    for fp in npz_files:
        with np.load(fp, allow_pickle=True) as d:
            key0, key1 = find_channel_keys(list(d.keys()))

            ch0 = ensure_event_sample_shape(d[key0])
            ch1 = ensure_event_sample_shape(d[key1])

            if ch0.shape != ch1.shape:
                raise ValueError(
                    f"ch0/ch1 shape mismatch: {fp.name}\n"
                    f"ch0={ch0.shape}, ch1={ch1.shape}"
                )

            ch0_list.append(ch0)
            ch1_list.append(ch1)

    return (
        np.concatenate(ch0_list, axis=0),
        np.concatenate(ch1_list, axis=0),
    )


# ============================================================
# 周波数ごとに応答方向を求め、射影波形を作る
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

    if n_sample <= N_PRE:
        raise ValueError(
            f"{folder.name}: N_PRE={N_PRE} が波形長 {n_sample} 以上です。"
        )

    # --------------------------------------------------------
    # イベントごとの pedestal を引く
    # 温度変動などによるbase揺れをなるべく除去する
    # --------------------------------------------------------
    ped0_each = np.mean(ch0[:, :N_PRE], axis=1)
    ped1_each = np.mean(ch1[:, :N_PRE], axis=1)

    dev0 = ch0 - ped0_each[:, None]
    dev1 = ch1 - ped1_each[:, None]

    # 周波数ごとの平均IQ波形
    mean_dev0 = np.mean(dev0, axis=0)
    mean_dev1 = np.mean(dev1, axis=0)

    # --------------------------------------------------------
    # 平均IQ波形において base から最も遠い点を peak とする
    # --------------------------------------------------------
    iq_distance = np.hypot(mean_dev0, mean_dev1)

    search_stop = n_sample if PEAK_SEARCH_STOP is None else PEAK_SEARCH_STOP
    if not (0 <= PEAK_SEARCH_START < search_stop <= n_sample):
        raise ValueError(
            f"peak search range is invalid: "
            f"{PEAK_SEARCH_START}:{search_stop}, n_sample={n_sample}"
        )

    peak_index_local = np.argmax(
        iq_distance[PEAK_SEARCH_START:search_stop]
    )
    peak_index = PEAK_SEARCH_START + peak_index_local

    delta0_peak = mean_dev0[peak_index]
    delta1_peak = mean_dev1[peak_index]

    delta_iq_peak = np.hypot(delta0_peak, delta1_peak)

    if delta_iq_peak == 0:
        raise RuntimeError(
            f"{freq:.3f} GHz: peak displacement is zero. "
            "N_PRE または peak探索範囲を確認してください。"
        )

    # --------------------------------------------------------
    # 応答ベクトル方向 u と直交方向 v
    # u: パルス方向
    # v: パルスに垂直な方向
    # --------------------------------------------------------
    u0 = delta0_peak / delta_iq_peak
    u1 = delta1_peak / delta_iq_peak

    v0 = -u1
    v1 = u0

    # 各イベントを応答方向・直交方向へ射影
    proj_parallel_evt = dev0 * u0 + dev1 * u1
    proj_perp_evt = dev0 * v0 + dev1 * v1

    # 平均と標準誤差
    proj_parallel_mean = np.mean(proj_parallel_evt, axis=0)
    proj_perp_mean = np.mean(proj_perp_evt, axis=0)

    proj_parallel_sem = (
        np.std(proj_parallel_evt, axis=0, ddof=1) / np.sqrt(n_event)
    )
    proj_perp_sem = (
        np.std(proj_perp_evt, axis=0, ddof=1) / np.sqrt(n_event)
    )

    # 射影波形上のピーク
    peak_proj = proj_parallel_mean[peak_index]

    results.append(
        {
            "freq_GHz": freq,
            "N_event": n_event,
            "n_sample": n_sample,
            "peak_index": peak_index,
            "peak_time_ns": peak_index / SAMPLE_RATE_HZ * 1e9,
            "u_ch0": u0,
            "u_ch1": u1,
            "v_ch0": v0,
            "v_ch1": v1,
            "delta_ch0_peak": delta0_peak,
            "delta_ch1_peak": delta1_peak,
            "delta_iq_peak": delta_iq_peak,
            "peak_proj": peak_proj,
            "parallel_mean": proj_parallel_mean,
            "parallel_sem": proj_parallel_sem,
            "perp_mean": proj_perp_mean,
            "perp_sem": proj_perp_sem,
        }
    )

results.sort(key=lambda d: d["freq_GHz"])

if not results:
    raise RuntimeError("対象の無印データを読み込めませんでした。")

n_sample = results[0]["n_sample"]
time_ns = np.arange(n_sample) / SAMPLE_RATE_HZ * 1e9


# ============================================================
# 1. 応答ベクトル方向へ射影した平均波形
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for r in results:
    ax.plot(
        time_ns,
        r["parallel_mean"],
        lw=1.8,
        label=f"{r['freq_GHz']:.3f} GHz",
    )

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Projected amplitude along response vector")
ax.set_title(
    "Pulse waveforms projected onto each RF response direction\n"
    f"z = {TARGET_Z:.1f} mm, x = {TARGET_X:.1f} mm"
)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    OUT_DIR / "projected_waveforms_absolute.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "projected_waveforms_absolute.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 2. 最大値で規格化した射影波形
#    → rise / fall の形だけ比較する
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for r in results:
    y = r["parallel_mean"].copy()
    peak = r["peak_proj"]

    if peak != 0:
        y = y / peak

    ax.plot(
        time_ns,
        y,
        lw=1.8,
        label=f"{r['freq_GHz']:.3f} GHz",
    )

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Normalized projected amplitude")
ax.set_title(
    "Normalized projected pulse waveforms\n"
    f"z = {TARGET_Z:.1f} mm, x = {TARGET_X:.1f} mm"
)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    OUT_DIR / "projected_waveforms_normalized.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "projected_waveforms_normalized.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 3. 応答に垂直な成分
#    → 周波数シフトだけなら小さいことが期待される
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for r in results:
    ax.plot(
        time_ns,
        r["perp_mean"],
        lw=1.5,
        label=f"{r['freq_GHz']:.3f} GHz",
    )

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Projected amplitude perpendicular to response")
ax.set_title(
    "Waveforms projected perpendicular to response direction\n"
    f"z = {TARGET_Z:.1f} mm, x = {TARGET_X:.1f} mm"
)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    OUT_DIR / "projected_waveforms_perpendicular.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "projected_waveforms_perpendicular.pdf",
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 4. 応答方向・ピーク値のCSV保存
# ============================================================
csv_path = OUT_DIR / "projection_summary.csv"

fieldnames = [
    "freq_GHz",
    "N_event",
    "peak_index",
    "peak_time_ns",
    "u_ch0",
    "u_ch1",
    "v_ch0",
    "v_ch1",
    "delta_ch0_peak",
    "delta_ch1_peak",
    "delta_iq_peak",
    "peak_proj",
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in results:
        row = {key: r[key] for key in fieldnames}
        writer.writerow(row)

print("\n保存先:")
print(OUT_DIR)

print("\n--- response-vector summary ---")
for r in results:
    print(
        f"{r['freq_GHz']:.3f} GHz | "
        f"u=({r['u_ch0']:+.4f}, {r['u_ch1']:+.4f}) | "
        f"|ΔIQ|={r['delta_iq_peak']:.6g} | "
        f"peak t={r['peak_time_ns']:.2f} ns"
    )