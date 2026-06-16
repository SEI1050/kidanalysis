from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# ここだけ基本的に変える
# ============================================================

DATA_DATE = "20260527"

# 解析対象
# Noneにすると全対象を読む
TARGET_FREQ_GHZ = 5.476
TARGET_Z_MM = 7.5
TARGET_X_MM = 3.4

FREQ_ATOL_GHZ = 1e-3
POS_ATOL_MM = 1e-6

# 1秒を50分割
N_PHASE_BINS = 50

# 入力
# "cloud" : OneDrive / CloudStorage側の npz
# "local" : KIDANALYSIS/data/20260527側のnpz
# "both"  : 両方
INPUT_MODE = "cloud"

NPZ_PATTERN = "wf_*.npz"
RECURSIVE_SEARCH = False

# baseline範囲 [us]
# Noneなら t < 0 全部
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# amp H を探す範囲 [us]
AMP_WINDOW_US = (0, 1.5)

# integral A を計算する範囲 [us]
INTEGRAL_WINDOW_US = (0, 1.5)

# ampの定義
# "projected": IQ射影後の波形
# "abs": sqrt(dch0^2 + dch1^2)
# "ch0": ch0
# "ch1": ch1
AMP_MODE = "projected"

# 積分の定義
# "signed": そのまま積分
# "positive": 正の部分だけ積分
# "abs": 絶対値を積分
INTEGRAL_MODE = "signed"

# 散布図の表示点数を制限
# 相関・fitは全点で計算する
MAX_SCATTER_POINTS = 20000

DPI = 300


# ============================================================
# パス設定
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "integral_tau_eff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

local_data_dir = HERE / "data" / DATA_DATE

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

# 必要なら手動で追加
EXTRA_INPUT_ROOTS = [
    # Path("/Users/kubokosei/Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit/20260527"),
]


def safe_name(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(":", "")
    )


def close(a, b, atol=POS_ATOL_MM):
    return np.isclose(a, b, atol=atol, rtol=0)


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
# 測定フォルダ読み取り
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
                if not np.isclose(
                    info["freq_ghz"],
                    TARGET_FREQ_GHZ,
                    atol=FREQ_ATOL_GHZ,
                    rtol=0,
                ):
                    continue

            if TARGET_Z_MM is not None:
                if not close(info["z_mm"], TARGET_Z_MM):
                    continue

            if TARGET_X_MM is not None:
                if not close(info["x_mm"], TARGET_X_MM):
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
    npz内にイベントごとの時刻があれば拾う。
    なければ None。
    """
    candidates = [
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

    for key in candidates:
        if key in data.files:
            arr = np.asarray(data[key]).squeeze()

            if arr.ndim == 1 and len(arr) == n_events:
                return arr.astype(float), key

    for key in data.files:
        low = key.lower()

        if "time" not in low and "stamp" not in low:
            continue

        arr = np.asarray(data[key]).squeeze()

        if arr.ndim == 1 and len(arr) == n_events:
            return arr.astype(float), key

    return None, None


def load_all_events_from_runs(runs):
    all_rows = []
    all_ch0 = []
    all_ch1 = []
    time_ref = None
    global_event_index = 0

    printed_keys = False
    has_event_time = False
    event_time_key_used = None

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

            if ch0.shape[1] != npts:
                print("skip npts mismatch:", f)
                continue

            if time_ref is None:
                time_ref = time
            else:
                if len(time) != len(time_ref) or not np.allclose(time, time_ref):
                    print("skip time axis mismatch:", f)
                    continue

            n_events = ch0.shape[0]
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
# direct integral / amp
# ============================================================

def compute_phase_bin(event_df, has_event_time):
    if has_event_time and event_df["event_time"].notna().all():
        t = event_df["event_time"].to_numpy(dtype=float)

        frac_sec = t - np.floor(t)
        phase_bin = np.floor(frac_sec * N_PHASE_BINS).astype(int)
        phase_bin = np.clip(phase_bin, 0, N_PHASE_BINS - 1)

        phase_method = "event_time_fractional_second"

    else:
        idx = event_df["global_event_index"].to_numpy(dtype=int)
        phase_bin = idx % N_PHASE_BINS
        phase_method = "event_index_mod_50"

    return phase_bin, phase_method


def analyze_integral_amp(time, ch0, ch1, event_df, has_event_time):
    baseline_default = time < 0

    baseline_mask = make_mask_us(
        time,
        BASELINE_WINDOW_US,
        default_mask=baseline_default,
    )

    amp_mask = make_mask_us(
        time,
        AMP_WINDOW_US,
        default_mask=time >= 0,
    )

    integral_mask = make_mask_us(
        time,
        INTEGRAL_WINDOW_US,
        default_mask=time >= 0,
    )

    if baseline_mask.sum() < 2:
        raise ValueError("baseline points too few")

    if amp_mask.sum() < 2:
        raise ValueError("amp points too few")

    if integral_mask.sum() < 2:
        raise ValueError("integral points too few")

    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    dch0 = ch0 - ped0[:, None]
    dch1 = ch1 - ped1[:, None]

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)

    # projected方向は平均IQのピーク方向
    r_mean = np.sqrt(mean0**2 + mean1**2)
    amp_indices = np.where(amp_mask)[0]
    idx_peak = amp_indices[np.argmax(r_mean[amp_mask])]

    v0 = mean0[idx_peak]
    v1 = mean1[idx_peak]
    norm = np.sqrt(v0**2 + v1**2)

    if norm == 0:
        u0, u1 = 1.0, 0.0
    else:
        u0, u1 = v0 / norm, v1 / norm

    proj = dch0 * u0 + dch1 * u1
    mean_proj = proj.mean(axis=0)

    # peakが正になるように符号をそろえる
    sign = np.sign(mean_proj[idx_peak])

    if sign == 0:
        sign = 1.0

    proj *= sign

    abs_iq = np.sqrt(dch0**2 + dch1**2)

    if AMP_MODE == "projected":
        signal = proj

    elif AMP_MODE == "abs":
        signal = abs_iq

    elif AMP_MODE == "ch0":
        signal = dch0

    elif AMP_MODE == "ch1":
        signal = dch1

    else:
        raise ValueError("AMP_MODE must be projected, abs, ch0, or ch1")

    # H = peak amplitude
    H = np.max(signal[:, amp_mask], axis=1)

    # A = integral
    sig_int = signal[:, integral_mask]
    t_int = time[integral_mask]

    if INTEGRAL_MODE == "signed":
        sig_for_int = sig_int

    elif INTEGRAL_MODE == "positive":
        sig_for_int = np.maximum(sig_int, 0)

    elif INTEGRAL_MODE == "abs":
        sig_for_int = np.abs(sig_int)

    else:
        raise ValueError("INTEGRAL_MODE must be signed, positive, or abs")

    A = np.trapezoid(sig_for_int, t_int, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau_eff_s = A / H
        inv_tau_eff = H / A

    phase_bin, phase_method = compute_phase_bin(event_df, has_event_time)

    out = event_df.copy()
    out["phase_bin"] = phase_bin

    out["ped0"] = ped0
    out["ped1"] = ped1
    out["ped_abs"] = np.sqrt(ped0**2 + ped1**2)

    out["amp_H"] = H
    out["integral_A_Vs"] = A
    out["integral_A_Vus"] = A * 1e6
    out["tau_eff_s"] = tau_eff_s
    out["tau_eff_us"] = tau_eff_s * 1e6
    out["inv_tau_eff_1_per_s"] = inv_tau_eff
    out["inv_tau_eff_1_per_us"] = inv_tau_eff / 1e6

    out["amp_mode"] = AMP_MODE
    out["integral_mode"] = INTEGRAL_MODE

    print()
    print("phase method:", phase_method)
    print("signal direction:", u0, u1)
    print("idx_peak:", idx_peak, "t_peak_us:", time[idx_peak] * 1e6)
    print("amp window us:", AMP_WINDOW_US)
    print("integral window us:", INTEGRAL_WINDOW_US)

    meta = {
        "phase_method": phase_method,
        "signal_dir_ch0": u0,
        "signal_dir_ch1": u1,
        "idx_peak": idx_peak,
        "t_peak_us": time[idx_peak] * 1e6,
    }

    return out, meta


def summarize_by_phase(metrics):
    rows = []

    for b in range(N_PHASE_BINS):
        g = metrics[metrics["phase_bin"] == b]

        row = {
            "phase_bin": b,
            "n_events": len(g),
        }

        for col in [
            "amp_H",
            "integral_A_Vus",
            "tau_eff_us",
            "inv_tau_eff_1_per_us",
            "ped0",
            "ped1",
            "ped_abs",
        ]:
            if len(g) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_sem"] = np.nan
            else:
                v = pd.to_numeric(g[col], errors="coerce")
                n = v.notna().sum()

                row[f"{col}_mean"] = v.mean()
                row[f"{col}_median"] = v.median()
                row[f"{col}_std"] = v.std(ddof=1)
                row[f"{col}_sem"] = v.std(ddof=1) / np.sqrt(n) if n > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# plot
# ============================================================

def thin_for_scatter(df):
    if len(df) <= MAX_SCATTER_POINTS:
        return df

    return df.sample(MAX_SCATTER_POINTS, random_state=0)


def linear_fit(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    # slope, intercept = np.polyfit(x[mask], y[mask], 1)
    slope = np.sum(x * y) / np.sum(x**2)
    intercept = 0.0

    r = np.corrcoef(x[mask], y[mask])[0, 1]

    return slope, intercept, r


def make_info_text():
    freq_text = "allfreq" if TARGET_FREQ_GHZ is None else f"{TARGET_FREQ_GHZ:.3f}GHz"
    z_text = "allz" if TARGET_Z_MM is None else f"z={TARGET_Z_MM:.1f}mm"
    x_text = "allx" if TARGET_X_MM is None else f"x={TARGET_X_MM:.1f}mm"

    return (
        f"date={DATA_DATE}, f={freq_text}, {z_text}, {x_text}, "
        f"amp={AMP_MODE}, integral={INTEGRAL_MODE}, "
        f"int window={INTEGRAL_WINDOW_US} us"
    )


def plot_integral_amp_summary(metrics, phase_df, out_png):
    df = metrics.copy()

    # 有効値だけ
    valid = (
        np.isfinite(df["amp_H"])
        & np.isfinite(df["integral_A_Vus"])
        & np.isfinite(df["tau_eff_us"])
    )
    df = df[valid].copy()

    if len(df) == 0:
        print("no valid metrics to plot")
        return

    df_plot = thin_for_scatter(df)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # --------------------------------------------------------
    # 1. integral A vs amp H
    # --------------------------------------------------------
    ax0 = ax[0, 0]

    sc = ax0.scatter(
        df_plot["amp_H"],
        df_plot["integral_A_Vus"],
        c=df_plot["phase_bin"],
        s=8,
        alpha=0.35,
    )

    slope, intercept, r = linear_fit(
        df["amp_H"].to_numpy(dtype=float),
        df["integral_A_Vus"].to_numpy(dtype=float),
    )

    if np.isfinite(slope):
        xmin = np.nanpercentile(df["amp_H"], 1)
        xmax = np.nanpercentile(df["amp_H"], 99)
        xx = np.linspace(xmin, xmax, 200)
        yy = slope * xx + intercept

        ax0.plot(
            xx,
            yy,
            color="black",
            lw=2,
            label=f"linear fit, slope={slope:.3g} us"
        )

    phase_mean = (
        df.groupby("phase_bin")[["amp_H", "integral_A_Vus"]]
        .mean()
        .reset_index()
    )

    ax0.plot(
        phase_mean["amp_H"],
        phase_mean["integral_A_Vus"],
        marker="o",
        lw=1.5,
        color="red",
        label="phase mean"
    )

    ax0.set_xlabel("amp H [V]")
    ax0.set_ylabel("integral A [V us]")
    ax0.set_title(f"Integral A vs amp H\nr = {r:.3f}")
    ax0.grid(True)
    ax0.legend(fontsize=8)

    # --------------------------------------------------------
    # 2. tau_eff = A/H vs phase bin
    # --------------------------------------------------------
    ax1 = ax[0, 1]

    ax1.errorbar(
        phase_df["phase_bin"],
        phase_df["tau_eff_us_mean"],
        yerr=phase_df["tau_eff_us_sem"],
        marker="o",
        lw=1.5,
        capsize=2,
        label="mean ± SEM",
    )

    ax1.plot(
        phase_df["phase_bin"],
        phase_df["tau_eff_us_median"],
        marker="s",
        lw=1.5,
        label="median",
    )

    ax1.set_xlabel("phase bin")
    ax1.set_ylabel(r"$\tau_{\rm eff}=A/H$ [$\mu$s]")
    ax1.set_title(r"Effective width $\tau_{\rm eff}$ vs phase bin")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # --------------------------------------------------------
    # 3. amp and integral vs phase bin
    # --------------------------------------------------------
    ax2 = ax[1, 0]

    ax2.errorbar(
        phase_df["phase_bin"],
        phase_df["amp_H_mean"],
        yerr=phase_df["amp_H_sem"],
        marker="o",
        lw=1.5,
        capsize=2,
        label="amp H mean ± SEM",
    )

    ax2.set_xlabel("phase bin")
    ax2.set_ylabel("amp H [V]")
    ax2.grid(True)

    ax2b = ax2.twinx()
    ax2b.errorbar(
        phase_df["phase_bin"],
        phase_df["integral_A_Vus_mean"],
        yerr=phase_df["integral_A_Vus_sem"],
        marker="s",
        lw=1.5,
        capsize=2,
        label="integral A mean ± SEM",
    )

    ax2b.set_ylabel("integral A [V us]")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax2.set_title("amp H and integral A vs phase bin")

    # --------------------------------------------------------
    # 4. H/A vs phase bin
    # --------------------------------------------------------
    ax3 = ax[1, 1]

    ax3.errorbar(
        phase_df["phase_bin"],
        phase_df["inv_tau_eff_1_per_us_mean"],
        yerr=phase_df["inv_tau_eff_1_per_us_sem"],
        marker="o",
        lw=1.5,
        capsize=2,
        label="mean ± SEM",
    )

    ax3.plot(
        phase_df["phase_bin"],
        phase_df["inv_tau_eff_1_per_us_median"],
        marker="s",
        lw=1.5,
        label="median",
    )

    ax3.set_xlabel("phase bin")
    ax3.set_ylabel(r"$H/A$ [$1/\mu$s]")
    ax3.set_title(r"Inverse effective width $H/A$ vs phase bin")
    ax3.grid(True)
    ax3.legend(fontsize=8)

    # colorbar
    fig.tight_layout(rect=[0, 0, 0.90, 0.94])
    cax = fig.add_axes([0.92, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("phase bin")

    fig.suptitle(
        f"Integral-based time-shape check\n{make_info_text()}",
        fontsize=14,
    )

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_png)


# ============================================================
# main
# ============================================================

print()
print("DATA_DATE:", DATA_DATE)
print("OUT_DIR:", OUT_DIR)

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

metrics, meta = analyze_integral_amp(
    time,
    ch0,
    ch1,
    event_df,
    has_event_time,
)

phase_df = summarize_by_phase(metrics)

freq_text = "allfreq" if TARGET_FREQ_GHZ is None else f"{TARGET_FREQ_GHZ:.3f}GHz"
z_text = "allz" if TARGET_Z_MM is None else f"z{TARGET_Z_MM:.1f}mm"
x_text = "allx" if TARGET_X_MM is None else f"x{TARGET_X_MM:.1f}mm"

base = safe_name(
    f"{DATA_DATE}_{freq_text}_{z_text}_{x_text}_"
    f"phase{N_PHASE_BINS}_{AMP_MODE}_{INTEGRAL_MODE}"
)

event_csv = OUT_DIR / f"events_integral_{base}.csv"
phase_csv = OUT_DIR / f"phase_integral_summary_{base}.csv"
png_path = OUT_DIR / f"integral_tau_eff_{base}.png"

metrics.to_csv(event_csv, index=False)
phase_df.to_csv(phase_csv, index=False)

print("saved:", event_csv)
print("saved:", phase_csv)

plot_integral_amp_summary(
    metrics,
    phase_df,
    out_png=png_path,
)

print()
print("done")
print("outputs saved in:", OUT_DIR)