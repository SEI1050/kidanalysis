from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 基本設定
# ============================================================

DATA_DATE = "20260618"

# 解析したい周波数・位置
# Noneなら全データを解析する
TARGET_FREQ_GHZ = 5.443
TARGET_Z_MM = 7.5
TARGET_X_MM = 3.4

# 位置や周波数の一致判定
FREQ_ATOL_GHZ = 1e-3
POS_ATOL_MM = 1e-6

# レーザー周波数
LASER_HZ = 2

# 温度周期が1 Hzなら1秒を50分割
N_PHASE_BINS = 2

# 入力
# "cloud": OneDrive / CloudStorage側
# "local": KIDANALYSIS/data/20260527側
# "both" : 両方
INPUT_MODE = "local"

NPZ_PATTERN = "wf_*.npz"
RECURSIVE_SEARCH = False

# baseline範囲 [us]
# Noneなら t < 0 全部
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# ampを取る時間範囲 [us]
PULSE_WINDOW_US = (0, None)
# PULSE_WINDOW_US = (0, 1.0)

# ampの定義
# "projected": IQ射影後のpulse height
# "abs": sqrt(dch0^2 + dch1^2) の最大値
# "ch0": ch0の最大値
# "ch1": ch1の最大値
AMP_MODE = "projected"

# 各phase binで平均と中央値を両方出す
PLOT_MEAN = True
PLOT_MEDIAN = True

DPI = 300


# ============================================================
# パス
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

# 解析結果の保存先：スクリプト側の data/ 以下
OUT_DIR = HERE / "data" / DATA_DATE / "phase2_amp_ped"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力データ：外付けSSD /Volumes/NO NAME/data/20260618
local_data_dir = Path("/Volumes/NO NAME/data") / DATA_DATE

if not local_data_dir.exists():
    raise FileNotFoundError(
        f"ローカルデータフォルダが見つかりません: {local_data_dir}"
    )

print(f"Input data directory: {local_data_dir}")
print(f"Output directory:     {OUT_DIR}")

cloud_data_candidates = [
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

    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive - The University of Tokyo"
    / "東京大学"
    / "4S"
    / "kidfit"
    / DATA_DATE,
]

EXTRA_INPUT_ROOTS = [
    # 必要なら手動で追加
    # Path("/Users/kubokosei/Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit/20260527"),
]


def collect_input_roots():
    roots = []
    candidates = []

    if INPUT_MODE in ["local", "both"]:
        candidates.append(("local", local_data_dir))

    if INPUT_MODE in ["cloud", "both"]:
        for p in cloud_data_candidates:
            candidates.append(("cloud", p))

    for p in EXTRA_INPUT_ROOTS:
        candidates.append(("extra", p))

    print()
    print("===== path check =====")
    print("HERE:", HERE)
    print("INPUT_MODE:", INPUT_MODE)

    seen = set()

    for kind, p in candidates:
        p = Path(p).expanduser().resolve(strict=False)
        exists = p.is_dir()

        print(f"[{kind}] {p}")
        print("   exists:", exists)

        if exists:
            key = p.as_posix()
            if key not in seen:
                roots.append(p)
                seen.add(key)

    if len(roots) == 0:
        raise RuntimeError("入力フォルダが見つかりません。")

    return roots


INPUT_ROOTS = collect_input_roots()


# ============================================================
# フォルダ名読み取り
# ============================================================

MEAS_DIR_PATTERN = re.compile(
    r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>.+))?$"
)


def parse_meas_dir_name(name):
    m = MEAS_DIR_PATTERN.match(name)

    if m is None:
        return None

    d = m.groupdict()

    return {
        "freq_ghz": float(d["freq"]),
        "z_mm": float(d["z"]),
        "x_mm": float(d["x"]),
        "tag": d["tag"] or "",
    }


def safe_name(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
    )


def close(a, b, atol=1e-6):
    return np.isclose(a, b, atol=atol, rtol=0)


def build_runs():
    runs = []

    for root in INPUT_ROOTS:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue

            info = parse_meas_dir_name(d.name)

            if info is None:
                continue

            if TARGET_FREQ_GHZ is not None:
                if not np.isclose(info["freq_ghz"], TARGET_FREQ_GHZ, atol=FREQ_ATOL_GHZ, rtol=0):
                    continue

            if TARGET_Z_MM is not None:
                if not close(info["z_mm"], TARGET_Z_MM, atol=POS_ATOL_MM):
                    continue

            if TARGET_X_MM is not None:
                if not close(info["x_mm"], TARGET_X_MM, atol=POS_ATOL_MM):
                    continue

            if RECURSIVE_SEARCH:
                npz_files = sorted(d.rglob(NPZ_PATTERN))
            else:
                npz_files = sorted(d.glob(NPZ_PATTERN))

            if len(npz_files) == 0:
                continue

            runs.append({
                "dir": d.name,
                "path": d,
                "freq_ghz": info["freq_ghz"],
                "z_mm": info["z_mm"],
                "x_mm": info["x_mm"],
                "tag": info["tag"],
                "npz_files": npz_files,
            })

    runs = sorted(
        runs,
        key=lambda r: (
            r["freq_ghz"],
            r["z_mm"],
            r["x_mm"],
            r["tag"],
            r["dir"],
        ),
    )

    return runs


# ============================================================
# npz utility
# ============================================================

def scalar(x):
    arr = np.asarray(x)
    if arr.size == 1:
        return arr.item()
    return x


def as_2d_waveform(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x


def make_time_axis(npts, sample_rate, ref_position):
    return (np.arange(npts) - npts * ref_position / 100.0) / sample_rate


def make_mask_us(time, window_us, default_mask=None):
    if window_us is None:
        if default_mask is None:
            return np.ones_like(time, dtype=bool)
        return default_mask

    lo, hi = window_us
    mask = np.ones_like(time, dtype=bool)

    if lo is not None:
        mask &= time >= lo * 1e-6

    if hi is not None:
        mask &= time <= hi * 1e-6

    return mask


def find_event_time_array(data, n_events, npts):
    """
    npz内にeventごとの絶対/相対時刻があれば拾う。
    なければ None。

    波形時間軸らしき length=npts の配列は除外する。
    """
    time_key_candidates = [
        "event_time",
        "event_times",
        "timestamp",
        "timestamps",
        "trigger_time",
        "trigger_times",
        "unix_time",
        "unix_times",
        "time_stamp",
        "time_stamps",
        "t_event",
        "t_events",
    ]

    for key in time_key_candidates:
        if key in data.files:
            arr = np.asarray(data[key]).squeeze()

            if arr.ndim == 1 and len(arr) == n_events:
                return arr.astype(float), key

    # それ以外にも n_events 長の時刻っぽいキーを探す
    for key in data.files:
        low = key.lower()

        if "time" not in low and "stamp" not in low:
            continue

        arr = np.asarray(data[key]).squeeze()

        if arr.ndim == 1 and len(arr) == n_events:
            return arr.astype(float), key

    return None, None


def load_all_events_from_runs(runs):
    """
    対象runの全npzを読み、イベントを時系列に並べる。
    event_timeがあれば使う。
    なければファイル順 + イベント順で仮のevent_indexを使う。
    """
    all_rows = []
    all_ch0 = []
    all_ch1 = []
    time_ref = None
    global_event_index = 0

    printed_keys = False
    event_time_key_used = None
    has_event_time = False

    for run in runs:
        for f in sorted(run["npz_files"], key=lambda p: p.stat().st_mtime):
            data = np.load(f)

            if not printed_keys:
                print()
                print("===== first npz keys =====")
                print("file:", f)
                print("keys:", data.files)
                printed_keys = True

            sample_rate = float(scalar(data["sample_rate"]))
            npts = int(scalar(data["npts"]))
            ref_position = float(scalar(data["ref_position"]))

            time = make_time_axis(npts, sample_rate, ref_position)

            ch0 = as_2d_waveform(data["ch0"])
            ch1 = as_2d_waveform(data["ch1"])

            if ch0.shape != ch1.shape:
                print("skip shape mismatch:", f)
                continue

            n_events = ch0.shape[0]

            if ch0.shape[1] != npts:
                print("skip npts mismatch:", f)
                continue

            if time_ref is None:
                time_ref = time
            else:
                if len(time) != len(time_ref) or not np.allclose(time, time_ref):
                    print("skip time axis mismatch:", f)
                    continue

            event_times, key = find_event_time_array(data, n_events, npts)

            if event_times is not None:
                has_event_time = True
                event_time_key_used = key

            for i in range(n_events):
                row = {
                    "global_event_index": global_event_index,
                    "file": f.as_posix(),
                    "file_name": f.name,
                    "event_in_file": i,
                    "freq_ghz": run["freq_ghz"],
                    "z_mm": run["z_mm"],
                    "x_mm": run["x_mm"],
                    "tag": run["tag"],
                    "run_dir": run["dir"],
                }

                if event_times is not None:
                    row["event_time"] = event_times[i]
                else:
                    row["event_time"] = np.nan

                all_rows.append(row)
                global_event_index += 1

            all_ch0.append(ch0)
            all_ch1.append(ch1)

    if len(all_ch0) == 0:
        raise RuntimeError("有効な波形が読み込めませんでした。")

    ch0_all = np.vstack(all_ch0)
    ch1_all = np.vstack(all_ch1)
    event_df = pd.DataFrame(all_rows)

    print()
    print("loaded events:", len(event_df))
    print("ch0 shape:", ch0_all.shape)
    print("event_time found:", has_event_time)
    print("event_time key:", event_time_key_used)

    return time_ref, ch0_all, ch1_all, event_df, has_event_time


# ============================================================
# direct amp / ped calculation
# ============================================================

def analyze_direct_amp_ped(time, ch0, ch1, event_df, has_event_time):
    baseline_default = time < 0
    baseline_mask = make_mask_us(
        time,
        BASELINE_WINDOW_US,
        default_mask=baseline_default,
    )

    pulse_default = time >= 0
    pulse_mask = make_mask_us(
        time,
        PULSE_WINDOW_US,
        default_mask=pulse_default,
    )

    if baseline_mask.sum() < 2:
        raise ValueError("baseline points too few")

    if pulse_mask.sum() < 2:
        raise ValueError("pulse points too few")

    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    dch0 = ch0 - ped0[:, None]
    dch1 = ch1 - ped1[:, None]

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)

    r_mean = np.sqrt(mean0**2 + mean1**2)
    pulse_indices = np.where(pulse_mask)[0]
    idx_peak = pulse_indices[np.argmax(r_mean[pulse_mask])]

    v0 = mean0[idx_peak]
    v1 = mean1[idx_peak]
    norm = np.sqrt(v0**2 + v1**2)

    if norm == 0:
        u0, u1 = 1.0, 0.0
    else:
        u0, u1 = v0 / norm, v1 / norm

    proj = dch0 * u0 + dch1 * u1
    mean_proj = proj.mean(axis=0)

    sign = np.sign(mean_proj[idx_peak])
    if sign == 0:
        sign = 1.0

    proj *= sign

    abs_iq = np.sqrt(dch0**2 + dch1**2)

    if AMP_MODE == "projected":
        amp = np.max(proj[:, pulse_mask], axis=1)

    elif AMP_MODE == "abs":
        amp = np.max(abs_iq[:, pulse_mask], axis=1)

    elif AMP_MODE == "ch0":
        amp = np.max(dch0[:, pulse_mask], axis=1)

    elif AMP_MODE == "ch1":
        amp = np.max(dch1[:, pulse_mask], axis=1)

    else:
        raise ValueError("AMP_MODE must be projected, abs, ch0, or ch1")

    # phase bin
    if has_event_time and event_df["event_time"].notna().all():
        # event_time の小数秒を使って 1秒を50分割
        t = event_df["event_time"].to_numpy(dtype=float)

        # UNIX timeでも相対時刻でも fracは同じ
        frac_sec = t - np.floor(t)
        phase_bin = np.floor(frac_sec * N_PHASE_BINS).astype(int)
        phase_bin = np.clip(phase_bin, 0, N_PHASE_BINS - 1)

        phase_method = "event_time_fractional_second"

    else:
        # 時刻がない場合はイベント順を50で割った余り
        idx = event_df["global_event_index"].to_numpy(dtype=int)
        phase_bin = idx % N_PHASE_BINS
        phase_method = "event_index_mod_50"

    out = event_df.copy()
    out["phase_bin"] = phase_bin
    out["ped0"] = ped0
    out["ped1"] = ped1
    out["ped_abs"] = np.sqrt(ped0**2 + ped1**2)
    out["amp"] = amp
    out["amp_mode"] = AMP_MODE
    out["baseline_noise_proj"] = proj[:, baseline_mask].std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["snr_direct"] = out["amp"] / out["baseline_noise_proj"]

    print()
    print("phase method:", phase_method)
    print("signal direction:", u0, u1)
    print("idx_peak:", idx_peak, "t_peak_us:", time[idx_peak] * 1e6)

    return out, {
        "phase_method": phase_method,
        "signal_dir_ch0": u0,
        "signal_dir_ch1": u1,
        "idx_peak": idx_peak,
        "t_peak_us": time[idx_peak] * 1e6,
    }


def summarize_by_phase(event_metrics):
    rows = []

    for b in range(N_PHASE_BINS):
        g = event_metrics[event_metrics["phase_bin"] == b]

        row = {
            "phase_bin": b,
            "n_events": len(g),
        }

        for col in ["amp", "ped0", "ped1", "ped_abs", "snr_direct"]:
            if len(g) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_sem"] = np.nan
            else:
                v = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = v.mean()
                row[f"{col}_median"] = v.median()
                row[f"{col}_std"] = v.std(ddof=1)
                row[f"{col}_sem"] = v.std(ddof=1) / np.sqrt(v.notna().sum())

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# plot
# ============================================================

def plot_phase_summary(phase_df, info_text, out_png):
    x = phase_df["phase_bin"]

    fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # amp
    if PLOT_MEAN:
        ax[0].errorbar(
            x,
            phase_df["amp_mean"],
            yerr=phase_df["amp_sem"],
            marker="o",
            lw=1.5,
            capsize=2,
            label="mean",
        )

    if PLOT_MEDIAN:
        ax[0].plot(
            x,
            phase_df["amp_median"],
            marker="s",
            lw=1.5,
            label="median",
        )

    ax[0].set_ylabel(f"amp [{AMP_MODE}]")
    ax[0].grid(True)
    ax[0].legend()

    # ped0
    if PLOT_MEAN:
        ax[1].errorbar(
            x,
            phase_df["ped0_mean"],
            yerr=phase_df["ped0_sem"],
            marker="o",
            lw=1.5,
            capsize=2,
            label="mean",
        )

    if PLOT_MEDIAN:
        ax[1].plot(
            x,
            phase_df["ped0_median"],
            marker="s",
            lw=1.5,
            label="median",
        )

    ax[1].set_ylabel("ped0 [V]")
    ax[1].grid(True)

    # ped1
    if PLOT_MEAN:
        ax[2].errorbar(
            x,
            phase_df["ped1_mean"],
            yerr=phase_df["ped1_sem"],
            marker="o",
            lw=1.5,
            capsize=2,
            label="mean",
        )

    if PLOT_MEDIAN:
        ax[2].plot(
            x,
            phase_df["ped1_median"],
            marker="s",
            lw=1.5,
            label="median",
        )

    ax[2].set_ylabel("ped1 [V]")
    ax[2].grid(True)

    # n events
    ax[3].bar(
        x,
        phase_df["n_events"],
        width=0.8,
    )
    ax[3].set_ylabel("events/bin")
    ax[3].set_xlabel("laser phase bin in 1 s cycle")
    ax[3].grid(True)

    fig.suptitle(
        f"50-bin phase dependence of direct amp and pedestal\n{info_text}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_png)


# ============================================================
# main
# ============================================================

runs = build_runs()

print()
print("found runs:", len(runs))

for r in runs:
    print(
        f"{r['dir']}  "
        f"freq={r['freq_ghz']:.3f}GHz  "
        f"z={r['z_mm']:.1f}mm  "
        f"x={r['x_mm']:.1f}mm  "
        f"tag={r['tag']}  "
        f"npz={len(r['npz_files'])}"
    )

if len(runs) == 0:
    raise RuntimeError("対象runが見つかりません。設定を確認してください。")

time, ch0, ch1, event_df, has_event_time = load_all_events_from_runs(runs)

event_metrics, meta = analyze_direct_amp_ped(
    time,
    ch0,
    ch1,
    event_df,
    has_event_time,
)

phase_df = summarize_by_phase(event_metrics)

freq_text = "allfreq" if TARGET_FREQ_GHZ is None else f"{TARGET_FREQ_GHZ:.3f}GHz"
z_text = "allz" if TARGET_Z_MM is None else f"z{TARGET_Z_MM:.1f}mm"
x_text = "allx" if TARGET_X_MM is None else f"x{TARGET_X_MM:.1f}mm"

base = safe_name(
    f"{DATA_DATE}_{freq_text}_{z_text}_{x_text}_phase{N_PHASE_BINS}_{AMP_MODE}"
)

event_csv = OUT_DIR / f"events_{base}.csv"
phase_csv = OUT_DIR / f"phase_summary_{base}.csv"
png_path = OUT_DIR / f"phase_summary_{base}.png"

event_metrics.to_csv(event_csv, index=False)
phase_df.to_csv(phase_csv, index=False)

print("saved:", event_csv)
print("saved:", phase_csv)

info_text = (
    f"date={DATA_DATE}, "
    f"freq={freq_text}, {z_text}, {x_text}, "
    f"amp={AMP_MODE}, "
    f"phase={meta['phase_method']}"
)

plot_phase_summary(
    phase_df,
    info_text=info_text,
    out_png=png_path,
)

print()
print("done")
print("outputs saved in:", OUT_DIR)