from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
BASE_DIR = Path("/Volumes/NO NAME/data/20260527")  # ←適宜変更
TARGET_Z = 7.5
TARGET_X = 3.4
TARGET_FREQS = [5.451, 5.461, 5.476, 5.491, 5.501]  # GHz

# サンプリング周波数 [Hz]（既知ならそのまま）
SAMPLE_RATE = 2.5e9  # 2.5 GS/s
DT_NS = 1e9 / SAMPLE_RATE

# =========================
# 補助関数
# =========================
def exact_folder_name(freq, z, x):
    return f"{freq:.3f}GHz_z={z:.1f}mm_x={x:.1f}mm"

def find_target_folders(base_dir, freqs, z, x):
    folders = []
    for f in freqs:
        name = exact_folder_name(f, z, x)
        p = base_dir / name
        if p.exists() and p.is_dir():
            folders.append((f, p))
        else:
            print(f"[WARN] folder not found: {p}")
    return folders

def guess_waveform_keys(npz_keys):
    """
    よくあるキー名候補から ch0, ch1 を推定
    """
    key_candidates0 = ["ch0", "wave0", "data0", "channel0", "I", "i"]
    key_candidates1 = ["ch1", "wave1", "data1", "channel1", "Q", "q"]

    key0 = None
    key1 = None

    for k in key_candidates0:
        if k in npz_keys:
            key0 = k
            break
    for k in key_candidates1:
        if k in npz_keys:
            key1 = k
            break

    return key0, key1

def load_all_waveforms_from_folder(folder):
    """
    folder 内の .npz を全部読んで ch0/ch1 を結合して返す
    戻り値:
        ch0_all: shape (N_event, N_sample)
        ch1_all: shape (N_event, N_sample)
    """
    npz_files = sorted(folder.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    ch0_list = []
    ch1_list = []

    for fp in npz_files:
        with np.load(fp, allow_pickle=True) as d:
            keys = list(d.keys())
            key0, key1 = guess_waveform_keys(keys)

            if key0 is None or key1 is None:
                raise KeyError(
                    f"Could not identify ch0/ch1 keys in {fp.name}. keys={keys}"
                )

            ch0 = np.asarray(d[key0])
            ch1 = np.asarray(d[key1])

            # shape を (N_event, N_sample) にそろえる
            if ch0.ndim == 1:
                ch0 = ch0[None, :]
            if ch1.ndim == 1:
                ch1 = ch1[None, :]

            # 万一 (N_sample, N_event) なら転置
            if ch0.shape[0] > ch0.shape[1]:
                # 通常イベント数 << サンプル数 なので、
                # 行数が列数より大きければ転置を疑う
                ch0 = ch0.T
            if ch1.shape[0] > ch1.shape[1]:
                ch1 = ch1.T

            ch0_list.append(ch0)
            ch1_list.append(ch1)

    ch0_all = np.concatenate(ch0_list, axis=0)
    ch1_all = np.concatenate(ch1_list, axis=0)

    return ch0_all, ch1_all

def baseline_subtract(wf, n_pre=300):
    """
    先頭 n_pre サンプルの平均を引く
    wf: (N_event, N_sample) or (N_sample,)
    """
    wf = np.asarray(wf)
    if wf.ndim == 1:
        ped = np.mean(wf[:n_pre])
        return wf - ped
    else:
        ped = np.mean(wf[:, :n_pre], axis=1, keepdims=True)
        return wf - ped

def make_time_axis_ns(n_sample, dt_ns):
    return np.arange(n_sample) * dt_ns

# =========================
# 読み込み
# =========================
folders = find_target_folders(BASE_DIR, TARGET_FREQS, TARGET_Z, TARGET_X)

data = {}
for freq, folder in folders:
    ch0_all, ch1_all = load_all_waveforms_from_folder(folder)

    # pedestal subtraction
    ch0_bs = baseline_subtract(ch0_all, n_pre=300)
    ch1_bs = baseline_subtract(ch1_all, n_pre=300)

    mean0 = np.mean(ch0_bs, axis=0)
    mean1 = np.mean(ch1_bs, axis=0)
    sem0 = np.std(ch0_bs, axis=0, ddof=1) / np.sqrt(ch0_bs.shape[0])
    sem1 = np.std(ch1_bs, axis=0, ddof=1) / np.sqrt(ch1_bs.shape[0])

    data[freq] = {
        "N": ch0_bs.shape[0],
        "mean0": mean0,
        "mean1": mean1,
        "sem0": sem0,
        "sem1": sem1,
    }

# time axis
if len(data) == 0:
    raise RuntimeError("No data loaded.")

first_freq = sorted(data.keys())[0]
n_sample = len(data[first_freq]["mean0"])
t_ns = make_time_axis_ns(n_sample, DT_NS)

# =========================
# プロット1: 生の平均波形（ch0, ch1）
# =========================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for freq in sorted(data.keys()):
    d = data[freq]
    axes[0].plot(t_ns, d["mean0"], label=f"{freq:.3f} GHz (N={d['N']})")
    axes[1].plot(t_ns, d["mean1"], label=f"{freq:.3f} GHz (N={d['N']})")

axes[0].set_title(f"Average waveform at z={TARGET_Z} mm, x={TARGET_X} mm")
axes[0].set_ylabel("ch0 (baseline-subtracted)")
axes[1].set_ylabel("ch1 (baseline-subtracted)")
axes[1].set_xlabel("Time [ns]")

axes[0].grid(True, alpha=0.3)
axes[1].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# =========================
# プロット2: 正規化波形（形だけ比較）
# =========================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for freq in sorted(data.keys()):
    d = data[freq]

    m0 = d["mean0"].copy()
    m1 = d["mean1"].copy()

    if np.max(np.abs(m0)) > 0:
        m0 = m0 / np.max(np.abs(m0))
    if np.max(np.abs(m1)) > 0:
        m1 = m1 / np.max(np.abs(m1))

    axes[0].plot(t_ns, m0, label=f"{freq:.3f} GHz")
    axes[1].plot(t_ns, m1, label=f"{freq:.3f} GHz")

axes[0].set_title("Normalized average waveform")
axes[0].set_ylabel("ch0 / max|ch0|")
axes[1].set_ylabel("ch1 / max|ch1|")
axes[1].set_xlabel("Time [ns]")

axes[0].grid(True, alpha=0.3)
axes[1].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# =========================
# プロット3: 周波数ごとの peak 値比較
# =========================
freqs = []
peak0 = []
peak1 = []

for freq in sorted(data.keys()):
    d = data[freq]
    freqs.append(freq)
    # 符号付きpeakを見たいなら max/min 両方確認して大きい方を取る
    p0 = d["mean0"][np.argmax(np.abs(d["mean0"]))]
    p1 = d["mean1"][np.argmax(np.abs(d["mean1"]))]
    peak0.append(p0)
    peak1.append(p1)

plt.figure(figsize=(8, 5))
plt.plot(freqs, peak0, "o-", label="ch0 peak")
plt.plot(freqs, peak1, "o-", label="ch1 peak")
plt.xlabel("Input RF frequency [GHz]")
plt.ylabel("Signed peak of average waveform")
plt.title(f"Peak vs RF frequency at z={TARGET_Z} mm, x={TARGET_X} mm")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()