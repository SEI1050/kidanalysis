from pathlib import Path
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

# 入力
# "cloud" : OneDrive / CloudStorage側
# "local" : kidanalysis/data/20260527 側
# "both"  : 両方
INPUT_MODE = "cloud"

NPZ_PATTERN = "wf_*.npz"
RECURSIVE_SEARCH = False

# repeat測定を同じ freq,z,x としてまとめるか
# 例: 5.451GHz_z=7.5mm_x=3.4mm と
#     5.451GHz_z=7.5mm_x=3.4mm_fourth をまとめる
GROUP_REPEATS_BY_POSITION = True

# 解析する信号
# "projected": IQ射影波形
# "abs"      : sqrt(dch0^2 + dch1^2)
# "ch0"      : ch0
# "ch1"      : ch1
SIGNAL_MODE = "projected"

# baseline範囲 [us]
# Noneなら t < 0 を全部使う
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# H=maxを探す範囲 [us]
AMP_WINDOW_US = (0, 1.5)

# A=積分を計算する範囲 [us]
INTEGRAL_WINDOW_US = (0, 1.5)

# 積分の定義
# "signed"   : そのまま積分
# "positive" : 正の部分だけ積分
# "abs"      : 絶対値を積分
INTEGRAL_MODE = "positive"

# scan plotの固定位置
XSCAN_FIXED_Z_MM = 7.5       # tau_eff vs x, z固定
ZSCAN_FIXED_X_MM = 3.4       # tau_eff vs z, x固定

FREQSCAN_FIXED_X_MM = 3.4    # tau_eff vs frequency
FREQSCAN_FIXED_Z_MM = 7.5

# Noneなら全周波数を描く
PLOT_FREQ_GHZ_LIST = None
# PLOT_FREQ_GHZ_LIST = [5.451, 5.476, 5.501]

# 位置・周波数の一致判定
FREQ_ATOL_GHZ = 1e-3
POS_ATOL_MM = 1e-6

# 小さすぎるampを除外
MIN_ABS_H = 1e-12

DPI = 300


# ============================================================
# パス設定
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "tau_eff_scan"
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

# 必要なら手動追加
EXTRA_INPUT_ROOTS = [
    # Path("/Users/kubokosei/Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit/20260527"),
]


# ============================================================
# utility
# ============================================================

def safe_name(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(":", "")
    )


def scalar(x):
    arr = np.asarray(x)
    if arr.size == 1:
        return arr.item()
    return x


def close_pos(a, b):
    return np.isclose(a, b, atol=POS_ATOL_MM, rtol=0)


def close_freq(a, b):
    return np.isclose(a, b, atol=FREQ_ATOL_GHZ, rtol=0)


def integrate_trapezoid(y, x, axis=1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


def mean_std_sem_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "n": 0,
        }

    return {
        "mean": np.mean(x),
        "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
        "sem": sem(x),
        "median": np.median(x),
        "n": len(x),
    }


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


def as_2d_waveform(x):
    x = np.asarray(x)

    if x.ndim == 1:
        x = x[None, :]

    return x


def make_time_axis(npts, sample_rate, ref_position):
    return (np.arange(npts) - npts * ref_position / 100.0) / sample_rate


# ============================================================
# 入力フォルダ探索
# ============================================================

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
# 測定フォルダ名 parse
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


def discover_measurements():
    measurements = []

    for root in INPUT_ROOTS:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue

            info = parse_meas_dir_name(d.name)

            if info is None:
                continue

            if RECURSIVE_SEARCH:
                npz_files = sorted(d.rglob(NPZ_PATTERN))
            else:
                npz_files = sorted(d.glob(NPZ_PATTERN))

            if len(npz_files) == 0:
                continue

            item = {
                "root": root,
                "dir_name": d.name,
                "path": d,
                "freq_ghz": info["freq_ghz"],
                "z_mm": info["z_mm"],
                "x_mm": info["x_mm"],
                "tag": info["tag"],
                "npz_files": npz_files,
            }

            measurements.append(item)

    measurements = sorted(
        measurements,
        key=lambda r: (
            r["freq_ghz"],
            r["z_mm"],
            r["x_mm"],
            r["tag"],
            r["dir_name"],
        ),
    )

    return measurements


def group_measurements(measurements):
    groups = {}

    for m in measurements:
        if GROUP_REPEATS_BY_POSITION:
            key = (
                round(m["freq_ghz"], 6),
                round(m["z_mm"], 6),
                round(m["x_mm"], 6),
            )
        else:
            key = (
                round(m["freq_ghz"], 6),
                round(m["z_mm"], 6),
                round(m["x_mm"], 6),
                m["tag"],
                m["dir_name"],
            )

        if key not in groups:
            groups[key] = {
                "freq_ghz": m["freq_ghz"],
                "z_mm": m["z_mm"],
                "x_mm": m["x_mm"],
                "tags": [],
                "dir_names": [],
                "paths": [],
                "npz_files": [],
            }

        groups[key]["tags"].append(m["tag"])
        groups[key]["dir_names"].append(m["dir_name"])
        groups[key]["paths"].append(m["path"])
        groups[key]["npz_files"].extend(m["npz_files"])

    out = list(groups.values())

    out = sorted(
        out,
        key=lambda g: (
            g["freq_ghz"],
            g["z_mm"],
            g["x_mm"],
            ",".join(g["dir_names"]),
        ),
    )

    return out


# ============================================================
# groupごとの tau_eff 計算
# ============================================================

def load_waveforms_for_group(group):
    all_ch0 = []
    all_ch1 = []
    time_ref = None

    used_files = 0

    npz_files = sorted(group["npz_files"], key=lambda p: p.stat().st_mtime)

    for f in npz_files:
        try:
            data = np.load(f)
        except Exception as e:
            print("skip load error:", f, e)
            continue

        needed = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
        if not all(k in data.files for k in needed):
            print("skip missing keys:", f, data.files)
            continue

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

        all_ch0.append(ch0)
        all_ch1.append(ch1)
        used_files += 1

    if len(all_ch0) == 0:
        return None, None, None, 0

    ch0_all = np.vstack(all_ch0)
    ch1_all = np.vstack(all_ch1)

    return time_ref, ch0_all, ch1_all, used_files


def compute_tau_eff_for_group(group):
    time, ch0, ch1, used_files = load_waveforms_for_group(group)

    if time is None:
        return None

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

    # pedestal
    ped0 = ch0[:, baseline_mask].mean(axis=1)
    ped1 = ch1[:, baseline_mask].mean(axis=1)

    dch0 = ch0 - ped0[:, None]
    dch1 = ch1 - ped1[:, None]

    # --------------------------------------------------------
    # signal作成
    # --------------------------------------------------------
    if SIGNAL_MODE == "projected":
        mean0 = dch0.mean(axis=0)
        mean1 = dch1.mean(axis=0)

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

        signal = dch0 * u0 + dch1 * u1

        mean_signal = signal.mean(axis=0)
        sign = np.sign(mean_signal[idx_peak])
        if sign == 0:
            sign = 1.0

        signal *= sign

        direction_ch0 = u0
        direction_ch1 = u1
        t_peak_us = time[idx_peak] * 1e6

    elif SIGNAL_MODE == "abs":
        signal = np.sqrt(dch0**2 + dch1**2)

        mean_signal = signal.mean(axis=0)
        amp_indices = np.where(amp_mask)[0]
        idx_peak = amp_indices[np.argmax(mean_signal[amp_mask])]

        direction_ch0 = np.nan
        direction_ch1 = np.nan
        t_peak_us = time[idx_peak] * 1e6

    elif SIGNAL_MODE == "ch0":
        signal = dch0

        mean_signal = signal.mean(axis=0)
        amp_indices = np.where(amp_mask)[0]
        idx_peak = amp_indices[np.argmax(np.abs(mean_signal[amp_mask]))]

        sign = np.sign(mean_signal[idx_peak])
        if sign == 0:
            sign = 1.0

        signal *= sign

        direction_ch0 = 1.0
        direction_ch1 = 0.0
        t_peak_us = time[idx_peak] * 1e6

    elif SIGNAL_MODE == "ch1":
        signal = dch1

        mean_signal = signal.mean(axis=0)
        amp_indices = np.where(amp_mask)[0]
        idx_peak = amp_indices[np.argmax(np.abs(mean_signal[amp_mask]))]

        sign = np.sign(mean_signal[idx_peak])
        if sign == 0:
            sign = 1.0

        signal *= sign

        direction_ch0 = 0.0
        direction_ch1 = 1.0
        t_peak_us = time[idx_peak] * 1e6

    else:
        raise ValueError("SIGNAL_MODE must be projected, abs, ch0, or ch1")

    # --------------------------------------------------------
    # H, A, tau_eff
    # --------------------------------------------------------
    H = np.max(signal[:, amp_mask], axis=1)

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

    A = integrate_trapezoid(sig_for_int, t_int, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau_eff_s = A / H
        inv_tau_eff = H / A

    tau_eff_us = tau_eff_s * 1e6
    A_Vus = A * 1e6
    inv_tau_eff_1_per_us = inv_tau_eff / 1e6

    valid = (
        np.isfinite(H)
        & np.isfinite(A_Vus)
        & np.isfinite(tau_eff_us)
        & (np.abs(H) > MIN_ABS_H)
    )

    H_v = H[valid]
    A_v = A_Vus[valid]
    tau_v = tau_eff_us[valid]
    inv_tau_v = inv_tau_eff_1_per_us[valid]

    # A vs H の原点通過傾き
    if len(H_v) >= 2 and np.sum(H_v**2) > 0:
        slope_origin_us = np.sum(H_v * A_v) / np.sum(H_v**2)
    else:
        slope_origin_us = np.nan

    if len(H_v) >= 3:
        corr_A_H = np.corrcoef(H_v, A_v)[0, 1]
    else:
        corr_A_H = np.nan

    H_stat = mean_std_sem_median(H_v)
    A_stat = mean_std_sem_median(A_v)
    tau_stat = mean_std_sem_median(tau_v)
    inv_tau_stat = mean_std_sem_median(inv_tau_v)
    ped0_stat = mean_std_sem_median(ped0)
    ped1_stat = mean_std_sem_median(ped1)

    row = {
        "freq_ghz": group["freq_ghz"],
        "z_mm": group["z_mm"],
        "x_mm": group["x_mm"],
        "tags": ",".join([str(t) for t in group["tags"] if str(t) != ""]),
        "dir_names": ";".join(group["dir_names"]),
        "n_files": used_files,
        "n_events_total": len(H),
        "n_events_valid": int(valid.sum()),

        "signal_mode": SIGNAL_MODE,
        "integral_mode": INTEGRAL_MODE,
        "amp_window_us": str(AMP_WINDOW_US),
        "integral_window_us": str(INTEGRAL_WINDOW_US),
        "baseline_window_us": str(BASELINE_WINDOW_US),

        "direction_ch0": direction_ch0,
        "direction_ch1": direction_ch1,
        "t_peak_us": t_peak_us,

        "amp_H_mean": H_stat["mean"],
        "amp_H_std": H_stat["std"],
        "amp_H_sem": H_stat["sem"],
        "amp_H_median": H_stat["median"],

        "integral_A_Vus_mean": A_stat["mean"],
        "integral_A_Vus_std": A_stat["std"],
        "integral_A_Vus_sem": A_stat["sem"],
        "integral_A_Vus_median": A_stat["median"],

        "tau_eff_us_mean": tau_stat["mean"],
        "tau_eff_us_std": tau_stat["std"],
        "tau_eff_us_sem": tau_stat["sem"],
        "tau_eff_us_median": tau_stat["median"],

        "inv_tau_eff_1_per_us_mean": inv_tau_stat["mean"],
        "inv_tau_eff_1_per_us_std": inv_tau_stat["std"],
        "inv_tau_eff_1_per_us_sem": inv_tau_stat["sem"],
        "inv_tau_eff_1_per_us_median": inv_tau_stat["median"],

        "ped0_mean": ped0_stat["mean"],
        "ped0_std": ped0_stat["std"],
        "ped0_sem": ped0_stat["sem"],

        "ped1_mean": ped1_stat["mean"],
        "ped1_std": ped1_stat["std"],
        "ped1_sem": ped1_stat["sem"],

        "A_vs_H_corr": corr_A_H,
        "A_vs_H_slope_origin_us": slope_origin_us,
    }

    return row


# ============================================================
# plot
# ============================================================

def maybe_filter_freqs(df):
    if PLOT_FREQ_GHZ_LIST is None:
        return df.copy()

    mask = np.zeros(len(df), dtype=bool)

    for f in PLOT_FREQ_GHZ_LIST:
        mask |= np.isclose(
            df["freq_ghz"].to_numpy(dtype=float),
            float(f),
            atol=FREQ_ATOL_GHZ,
            rtol=0,
        )

    return df[mask].copy()


def plot_xscan(summary_df):
    df = maybe_filter_freqs(summary_df)
    df = df[np.isclose(df["z_mm"], XSCAN_FIXED_Z_MM, atol=POS_ATOL_MM, rtol=0)]
    df = df.sort_values(["freq_ghz", "x_mm"])

    if len(df) == 0:
        print("skip xscan plot: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for freq, g in df.groupby("freq_ghz"):
        g = g.sort_values("x_mm")

        ax.errorbar(
            g["x_mm"],
            g["tau_eff_us_mean"],
            yerr=g["tau_eff_us_sem"],
            marker="o",
            lw=1.5,
            capsize=3,
            label=f"{freq:.3f} GHz",
        )

    ax.set_xlabel("x position [mm]")
    ax.set_ylabel(r"$\tau_{\rm eff}=A/H$ [$\mu$s]")
    ax.set_title(
        rf"$\tau_{{\rm eff}}$ x scan, z={XSCAN_FIXED_Z_MM:.1f} mm"
        "\n"
        f"signal={SIGNAL_MODE}, integral={INTEGRAL_MODE}, "
        f"window={INTEGRAL_WINDOW_US} us"
    )
    ax.grid(True)

    ax.legend(
        fontsize=8,
        title="frequency",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out = OUT_DIR / safe_name(
        f"tau_eff_xscan_z{XSCAN_FIXED_Z_MM:.1f}mm_{SIGNAL_MODE}_{INTEGRAL_MODE}.png"
    )
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)


def plot_zscan(summary_df):
    df = maybe_filter_freqs(summary_df)
    df = df[np.isclose(df["x_mm"], ZSCAN_FIXED_X_MM, atol=POS_ATOL_MM, rtol=0)]
    df = df.sort_values(["freq_ghz", "z_mm"])

    if len(df) == 0:
        print("skip zscan plot: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for freq, g in df.groupby("freq_ghz"):
        g = g.sort_values("z_mm")

        ax.errorbar(
            g["z_mm"],
            g["tau_eff_us_mean"],
            yerr=g["tau_eff_us_sem"],
            marker="o",
            lw=1.5,
            capsize=3,
            label=f"{freq:.3f} GHz",
        )

    ax.set_xlabel("z position [mm]")
    ax.set_ylabel(r"$\tau_{\rm eff}=A/H$ [$\mu$s]")
    ax.set_title(
        rf"$\tau_{{\rm eff}}$ z scan, x={ZSCAN_FIXED_X_MM:.1f} mm"
        "\n"
        f"signal={SIGNAL_MODE}, integral={INTEGRAL_MODE}, "
        f"window={INTEGRAL_WINDOW_US} us"
    )
    ax.grid(True)

    ax.legend(
        fontsize=8,
        title="frequency",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out = OUT_DIR / safe_name(
        f"tau_eff_zscan_x{ZSCAN_FIXED_X_MM:.1f}mm_{SIGNAL_MODE}_{INTEGRAL_MODE}.png"
    )
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)


def plot_freqscan(summary_df):
    df = summary_df.copy()

    df = df[
        np.isclose(df["x_mm"], FREQSCAN_FIXED_X_MM, atol=POS_ATOL_MM, rtol=0)
        & np.isclose(df["z_mm"], FREQSCAN_FIXED_Z_MM, atol=POS_ATOL_MM, rtol=0)
    ]

    df = maybe_filter_freqs(df)
    df = df.sort_values("freq_ghz")

    if len(df) == 0:
        print("skip freqscan plot: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        df["freq_ghz"],
        df["tau_eff_us_mean"],
        yerr=df["tau_eff_us_sem"],
        marker="o",
        lw=1.5,
        capsize=3,
        label=rf"x={FREQSCAN_FIXED_X_MM:.1f} mm, z={FREQSCAN_FIXED_Z_MM:.1f} mm",
    )

    ax.set_xlabel("readout frequency [GHz]")
    ax.set_ylabel(r"$\tau_{\rm eff}=A/H$ [$\mu$s]")
    ax.set_title(
        rf"$\tau_{{\rm eff}}$ frequency scan"
        "\n"
        f"signal={SIGNAL_MODE}, integral={INTEGRAL_MODE}, "
        f"window={INTEGRAL_WINDOW_US} us"
    )
    ax.grid(True)
    ax.legend(fontsize=9)

    fig.tight_layout()

    out = OUT_DIR / safe_name(
        f"tau_eff_freqscan_x{FREQSCAN_FIXED_X_MM:.1f}mm_z{FREQSCAN_FIXED_Z_MM:.1f}mm_{SIGNAL_MODE}_{INTEGRAL_MODE}.png"
    )
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)


def plot_all_quantity_xscan(summary_df):
    """
    tau_effだけでなく、ampとintegralも一緒に確認する補助図。
    """
    df = maybe_filter_freqs(summary_df)
    df = df[np.isclose(df["z_mm"], XSCAN_FIXED_Z_MM, atol=POS_ATOL_MM, rtol=0)]
    df = df.sort_values(["freq_ghz", "x_mm"])

    if len(df) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    quantities = [
        ("amp_H_mean", "amp_H_sem", "amp H [V]"),
        ("integral_A_Vus_mean", "integral_A_Vus_sem", "integral A [V us]"),
        ("tau_eff_us_mean", "tau_eff_us_sem", r"$\tau_{\rm eff}$ [$\mu$s]"),
    ]

    for ax, (col, err, ylabel) in zip(axes, quantities):
        for freq, g in df.groupby("freq_ghz"):
            g = g.sort_values("x_mm")

            ax.errorbar(
                g["x_mm"],
                g[col],
                yerr=g[err],
                marker="o",
                lw=1.5,
                capsize=3,
                label=f"{freq:.3f} GHz",
            )

        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[-1].set_xlabel("x position [mm]")

    axes[0].set_title(
        f"amp, integral, tau_eff x scan, z={XSCAN_FIXED_Z_MM:.1f} mm"
    )

    axes[0].legend(
        fontsize=8,
        title="frequency",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out = OUT_DIR / safe_name(
        f"amp_integral_tau_eff_xscan_z{XSCAN_FIXED_Z_MM:.1f}mm_{SIGNAL_MODE}_{INTEGRAL_MODE}.png"
    )
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)


def plot_all_quantity_zscan(summary_df):
    """
    tau_effだけでなく、ampとintegralも一緒に確認する補助図。
    """
    df = maybe_filter_freqs(summary_df)
    df = df[np.isclose(df["x_mm"], ZSCAN_FIXED_X_MM, atol=POS_ATOL_MM, rtol=0)]
    df = df.sort_values(["freq_ghz", "z_mm"])

    if len(df) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    quantities = [
        ("amp_H_mean", "amp_H_sem", "amp H [V]"),
        ("integral_A_Vus_mean", "integral_A_Vus_sem", "integral A [V us]"),
        ("tau_eff_us_mean", "tau_eff_us_sem", r"$\tau_{\rm eff}$ [$\mu$s]"),
    ]

    for ax, (col, err, ylabel) in zip(axes, quantities):
        for freq, g in df.groupby("freq_ghz"):
            g = g.sort_values("z_mm")

            ax.errorbar(
                g["z_mm"],
                g[col],
                yerr=g[err],
                marker="o",
                lw=1.5,
                capsize=3,
                label=f"{freq:.3f} GHz",
            )

        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[-1].set_xlabel("z position [mm]")

    axes[0].set_title(
        f"amp, integral, tau_eff z scan, x={ZSCAN_FIXED_X_MM:.1f} mm"
    )

    axes[0].legend(
        fontsize=8,
        title="frequency",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out = OUT_DIR / safe_name(
        f"amp_integral_tau_eff_zscan_x{ZSCAN_FIXED_X_MM:.1f}mm_{SIGNAL_MODE}_{INTEGRAL_MODE}.png"
    )
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)


# ============================================================
# main
# ============================================================

print()
print("DATA_DATE:", DATA_DATE)
print("OUT_DIR:", OUT_DIR)
print("SIGNAL_MODE:", SIGNAL_MODE)
print("INTEGRAL_MODE:", INTEGRAL_MODE)
print("AMP_WINDOW_US:", AMP_WINDOW_US)
print("INTEGRAL_WINDOW_US:", INTEGRAL_WINDOW_US)

measurements = discover_measurements()
groups = group_measurements(measurements)

print()
print("found measurement dirs:", len(measurements))
print("found groups:", len(groups))

rows = []

for i, g in enumerate(groups, start=1):
    print()
    print(
        f"[{i}/{len(groups)}] "
        f"f={g['freq_ghz']:.3f} GHz, "
        f"z={g['z_mm']:.1f} mm, "
        f"x={g['x_mm']:.1f} mm, "
        f"dirs={len(g['dir_names'])}, "
        f"npz={len(g['npz_files'])}"
    )

    try:
        row = compute_tau_eff_for_group(g)
    except Exception as e:
        print("  failed:", e)
        row = None

    if row is not None:
        rows.append(row)

        print(
            "  tau_eff = "
            f"{row['tau_eff_us_mean']:.5g} ± {row['tau_eff_us_sem']:.2g} us, "
            f"n={row['n_events_valid']}"
        )

if len(rows) == 0:
    raise RuntimeError("有効な解析結果がありません。")

summary_df = pd.DataFrame(rows)

summary_df = summary_df.sort_values(["freq_ghz", "z_mm", "x_mm"])

summary_csv = OUT_DIR / safe_name(
    f"tau_eff_scan_summary_{DATA_DATE}_{SIGNAL_MODE}_{INTEGRAL_MODE}.csv"
)

summary_df.to_csv(summary_csv, index=False)
print()
print("saved:", summary_csv)

plot_xscan(summary_df)
plot_zscan(summary_df)
plot_freqscan(summary_df)

plot_all_quantity_xscan(summary_df)
plot_all_quantity_zscan(summary_df)

print()
print("done")
print("outputs saved in:", OUT_DIR)