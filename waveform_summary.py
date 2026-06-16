from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

# ======================
# 図を画面表示しない保存専用backend
# ======================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================
# ここだけ基本的に変える
# ======================
DATA_DATE = "20260527"

# 対象にしたい周波数だけ指定したい場合
# None なら全部
TARGET_FREQ_GHZ = None
# TARGET_FREQ_GHZ = 5.451

# 対象にしたい z, x だけ指定したい場合
# None なら全部
TARGET_Z_MM = None
TARGET_X_MM = None

# 既存のファイル名生成との互換用
base_dir = DATA_DATE


# ======================
# 設定
# ======================
NPZ_PATTERN = "wf_*.npz"

# 測定フォルダの直下だけ探すなら False
# さらに深い場所まで探したいなら True
RECURSIVE_SEARCH = False

# 重ね書きする最大イベント数
MAX_OVERLAY = 100

# baseline に使う時間範囲 [us]
# None にすると t < 0 全部を baseline に使う
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-200, -20)

# pulse を探す時間範囲 [us]
PULSE_WINDOW_US = (0, None)
# PULSE_WINDOW_US = (0, 500)

# 図の保存・表示
SAVE_FIG = True

# 実行中に図を表示しない
SHOW_FIG = False

# 比較図で重ねる最大run数
MAX_COMPARE_RUNS = 60


# ======================
# ローカル側
# この py ファイルがあるフォルダ
# ======================
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()


# ======================
# 出力先 ローカル側
# 例:
# KIDANALYSIS/data/20260527/trigger_summary/
# ======================
OUT_DIR = HERE / "data" / DATA_DATE / "trigger_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# OneDrive側のデータフォルダ候補
# ======================
onedrive_candidates = [
    Path.home()
    / "OneDrive - The University of Tokyo"
    / "東京大学"
    / "4S"
    / "kidfit",

    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-TheUniversityofTokyo"
    / "東京大学"
    / "4S"
    / "kidfit",
]

ROOT_DIR = None

for p in onedrive_candidates:
    candidate = p / DATA_DATE
    if candidate.is_dir():
        ROOT_DIR = candidate
        break

if ROOT_DIR is None:
    print("ERROR: データフォルダが見つかりません。候補は以下です。")
    for p in onedrive_candidates:
        print("  ", p / DATA_DATE)
    sys.exit(1)


# ======================
# 測定フォルダ名の読み取り
# 例:
# 5.451GHz_z=7.5mm_x=5.4mm
# 5.451GHz_z=7.5mm_x=3.4mm_second
# ======================
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


def build_runs_from_folder(root_dir):
    """
    日付フォルダ直下から
    5.451GHz_z=7.5mm_x=5.4mm
    みたいな測定フォルダを自動検出する。
    """
    runs = []

    for d in sorted(root_dir.iterdir()):
        if not d.is_dir():
            continue

        info = parse_meas_dir_name(d.name)

        if info is None:
            continue

        if TARGET_FREQ_GHZ is not None and not np.isclose(
            info["freq_ghz"],
            TARGET_FREQ_GHZ,
        ):
            continue

        if TARGET_Z_MM is not None and not np.isclose(
            info["z_mm"],
            TARGET_Z_MM,
        ):
            continue

        if TARGET_X_MM is not None and not np.isclose(
            info["x_mm"],
            TARGET_X_MM,
        ):
            continue

        if RECURSIVE_SEARCH:
            npz_files = sorted(d.rglob(NPZ_PATTERN))
        else:
            npz_files = sorted(d.glob(NPZ_PATTERN))

        if len(npz_files) == 0:
            print("skip no npz:", d)
            continue

        tag_part = f"_{info['tag']}" if info["tag"] else ""

        label = (
            f"{info['freq_ghz']:.3f} GHz "
            f"z={info['z_mm']:.1f} mm "
            f"x={info['x_mm']:.1f} mm"
            f"{tag_part}"
        )

        runs.append({
            "dir": d.name,
            "path": d,
            "freq_ghz": info["freq_ghz"],
            "z_mm": info["z_mm"],
            "x_mm": info["x_mm"],
            "tag": info["tag"],
            "label": label,
            "n_npz": len(npz_files),
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


def safe_name(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
    )


# ======================
# utility
# ======================
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


def find_npz_files(run_path):
    if RECURSIVE_SEARCH:
        files = list(run_path.rglob(NPZ_PATTERN))
    else:
        files = list(run_path.glob(NPZ_PATTERN))

    files = sorted(files, key=lambda p: p.stat().st_mtime)

    return files


def load_run_npz(run_path):
    """
    1つの測定フォルダから wf_*.npz を全部読んで、
    time, ch0_all, ch1_all を返す。

    ch0_all, ch1_all shape = (n_events, npts)
    """
    files = find_npz_files(run_path)

    if len(files) == 0:
        raise FileNotFoundError(f"npz file not found: {run_path / NPZ_PATTERN}")

    all_ch0 = []
    all_ch1 = []
    time_ref = None
    meta_rows = []

    for f in files:
        data = np.load(f)

        sample_rate = float(scalar(data["sample_rate"]))
        npts = int(scalar(data["npts"]))
        ref_position = float(scalar(data["ref_position"]))

        time = make_time_axis(npts, sample_rate, ref_position)

        ch0 = as_2d_waveform(data["ch0"])
        ch1 = as_2d_waveform(data["ch1"])

        if ch0.shape != ch1.shape:
            print(
                f"skip shape mismatch: {f.name}, "
                f"ch0={ch0.shape}, ch1={ch1.shape}"
            )
            continue

        if ch0.shape[1] != npts:
            print(
                f"skip npts mismatch: {f.name}, "
                f"npts={npts}, ch0.shape={ch0.shape}"
            )
            continue

        if time_ref is None:
            time_ref = time
        else:
            if len(time) != len(time_ref):
                print(f"skip time length mismatch: {f.name}")
                continue

            if not np.allclose(time, time_ref):
                print(f"skip time axis mismatch: {f.name}")
                continue

        all_ch0.append(ch0)
        all_ch1.append(ch1)

        meta_rows.append({
            "file": f.name,
            "nwaveform": ch0.shape[0],
            "npts": npts,
            "sample_rate": sample_rate,
            "ref_position": ref_position,
        })

    if len(all_ch0) == 0:
        raise ValueError(f"no valid waveform in {run_path}")

    ch0_all = np.vstack(all_ch0)
    ch1_all = np.vstack(all_ch1)

    meta = pd.DataFrame(meta_rows)

    return time_ref, ch0_all, ch1_all, meta


def analyze_waveforms(time, ch0, ch1):
    n_events, npts = ch0.shape

    baseline_default = time < 0

    baseline_mask = make_mask_us(
        time,
        BASELINE_WINDOW_US,
        default_mask=baseline_default,
    )

    if baseline_mask.sum() < 2:
        raise ValueError(
            "baseline points too few. "
            "BASELINE_WINDOW_US を確認してください。"
        )

    base0 = ch0[:, baseline_mask].mean(axis=1, keepdims=True)
    base1 = ch1[:, baseline_mask].mean(axis=1, keepdims=True)

    dch0 = ch0 - base0
    dch1 = ch1 - base1

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)

    std0 = dch0.std(axis=0, ddof=1) if n_events > 1 else np.zeros(npts)
    std1 = dch1.std(axis=0, ddof=1) if n_events > 1 else np.zeros(npts)

    sem0 = std0 / np.sqrt(n_events)
    sem1 = std1 / np.sqrt(n_events)

    pulse_default = time >= 0

    pulse_mask = make_mask_us(
        time,
        PULSE_WINDOW_US,
        default_mask=pulse_default,
    )

    if pulse_mask.sum() < 2:
        raise ValueError(
            "pulse points too few. "
            "PULSE_WINDOW_US を確認してください。"
        )

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
    std_proj = proj.std(axis=0, ddof=1) if n_events > 1 else np.zeros(npts)
    sem_proj = std_proj / np.sqrt(n_events)

    sign = np.sign(mean_proj[idx_peak])

    if sign == 0:
        sign = 1.0

    pulse_height = sign * np.max(sign * proj[:, pulse_mask], axis=1)

    baseline_noise_each = proj[:, baseline_mask].std(axis=1, ddof=1)
    baseline_noise_median = np.median(baseline_noise_each)

    snr_each = pulse_height / baseline_noise_each
    snr_median = np.median(snr_each)

    result = {
        "dch0": dch0,
        "dch1": dch1,
        "mean0": mean0,
        "mean1": mean1,
        "std0": std0,
        "std1": std1,
        "sem0": sem0,
        "sem1": sem1,
        "proj": proj,
        "mean_proj": mean_proj,
        "std_proj": std_proj,
        "sem_proj": sem_proj,
        "pulse_height": pulse_height,
        "baseline_noise_each": baseline_noise_each,
        "snr_each": snr_each,
        "baseline_mask": baseline_mask,
        "pulse_mask": pulse_mask,
        "idx_peak": idx_peak,
        "signal_direction": (u0, u1),
        "baseline_noise_median": baseline_noise_median,
        "snr_median": snr_median,
    }

    return result


def plot_run_summary(time, result, label, outpath=None):
    """
    1つの測定フォルダ分のまとめplot。
    波形重ね書き、平均波形、IQ軌跡、射影波形、pulse height histogram を描く。
    """
    time_us = time * 1e6

    dch0 = result["dch0"]
    dch1 = result["dch1"]

    mean0 = result["mean0"]
    mean1 = result["mean1"]

    sem0 = result["sem0"]
    sem1 = result["sem1"]

    mean_proj = result["mean_proj"]
    sem_proj = result["sem_proj"]

    pulse_height = result["pulse_height"]
    idx_peak = result["idx_peak"]

    n_events = dch0.shape[0]
    n_overlay = min(n_events, MAX_OVERLAY)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # ----------------------
    # ch0 overlay
    # ----------------------
    for i in range(n_overlay):
        ax[0, 0].plot(time_us, dch0[i], alpha=0.15, lw=0.8)

    ax[0, 0].plot(time_us, mean0, color="black", lw=2.5, label="mean")
    ax[0, 0].axvline(0, ls="--", color="gray")
    ax[0, 0].set_title("ch0 baseline-subtracted")
    ax[0, 0].set_xlabel(r"Time [$\mu$s]")
    ax[0, 0].set_ylabel("ch0 [V]")
    ax[0, 0].legend()

    # ----------------------
    # ch1 overlay
    # ----------------------
    for i in range(n_overlay):
        ax[1, 0].plot(time_us, dch1[i], alpha=0.15, lw=0.8)

    ax[1, 0].plot(time_us, mean1, color="black", lw=2.5, label="mean")
    ax[1, 0].axvline(0, ls="--", color="gray")
    ax[1, 0].set_title("ch1 baseline-subtracted")
    ax[1, 0].set_xlabel(r"Time [$\mu$s]")
    ax[1, 0].set_ylabel("ch1 [V]")
    ax[1, 0].legend()

    # ----------------------
    # average ch0/ch1
    # ----------------------
    ax[0, 1].plot(time_us, mean0, label="mean ch0")
    ax[0, 1].fill_between(
        time_us,
        mean0 - sem0,
        mean0 + sem0,
        alpha=0.25,
    )

    ax[0, 1].plot(time_us, mean1, label="mean ch1")
    ax[0, 1].fill_between(
        time_us,
        mean1 - sem1,
        mean1 + sem1,
        alpha=0.25,
    )

    ax[0, 1].axvline(0, ls="--", color="gray")
    ax[0, 1].axvline(time_us[idx_peak], ls=":", color="red", label="peak")
    ax[0, 1].set_title("average waveform")
    ax[0, 1].set_xlabel(r"Time [$\mu$s]")
    ax[0, 1].set_ylabel("Voltage [V]")
    ax[0, 1].legend()

    # ----------------------
    # mean IQ trajectory
    # ----------------------
    # mean IQ trajectory
    ax[1, 1].plot(
        mean0,
        mean1,
        marker=".",
        ms=3,
        lw=1.5,
        color="C0",
        alpha=0.75,
        label="trajectory",
        zorder=1,
    )

    # start marker
    ax[1, 1].scatter(
        mean0[0],
        mean1[0],
        s=140,
        marker="*",
        color="limegreen",
        edgecolors="black",
        linewidths=1.2,
        label="start",
        zorder=5,
    )

    # peak marker
    ax[1, 1].scatter(
        mean0[idx_peak],
        mean1[idx_peak],
        s=120,
        marker="o",
        color="orange",
        edgecolors="black",
        linewidths=1.2,
        label="peak",
        zorder=6,
    )

    ax[1, 1].set_title("mean IQ trajectory")
    ax[1, 1].set_xlabel("ch0")
    ax[1, 1].set_ylabel("ch1")
    ax[1, 1].axis("equal")
    ax[1, 1].legend()

    # ----------------------
    # projected pulse
    # ----------------------
    ax[0, 2].plot(time_us, mean_proj, label="projected mean")
    ax[0, 2].fill_between(
        time_us,
        mean_proj - sem_proj,
        mean_proj + sem_proj,
        alpha=0.25,
        label="SEM",
    )

    ax[0, 2].axvline(0, ls="--", color="gray")
    ax[0, 2].axvline(time_us[idx_peak], ls=":", color="red", label="peak")
    ax[0, 2].set_title("projected pulse")
    ax[0, 2].set_xlabel(r"Time [$\mu$s]")
    ax[0, 2].set_ylabel("projected signal [V]")
    ax[0, 2].legend()

    # ----------------------
    # pulse height histogram
    # ----------------------
    ax[1, 2].hist(pulse_height, bins=50, histtype="step")
    ax[1, 2].set_title("pulse height histogram")
    ax[1, 2].set_xlabel("pulse height [V]")
    ax[1, 2].set_ylabel("counts")

    for a in ax.ravel():
        a.grid(True)

    fig.suptitle(f"{label}  n={n_events}", fontsize=16)
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300)
        print("saved:", outpath)

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison_all(all_results, out_dir, base_dir):
    """
    全データの平均波形・射影波形をまとめて比較。
    run数が多すぎる場合は MAX_COMPARE_RUNS まで。
    """
    items = list(all_results.items())

    if len(items) == 0:
        return

    if len(items) > MAX_COMPARE_RUNS:
        print(
            f"comparison plot: too many runs, "
            f"use first {MAX_COMPARE_RUNS}/{len(items)}"
        )
        items = items[:MAX_COMPARE_RUNS]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    for label, item in items:
        time = item["time"]
        time_us = time * 1e6
        result = item["result"]

        ax[0, 0].plot(time_us, result["mean0"], label=label)
        ax[0, 1].plot(time_us, result["mean1"], label=label)
        ax[1, 0].plot(time_us, result["mean_proj"], label=label)
        ax[1, 1].hist(
            result["pulse_height"],
            bins=50,
            histtype="step",
            density=False,
            label=label,
        )

    ax[0, 0].set_title("mean ch0")
    ax[0, 0].set_xlabel(r"Time [$\mu$s]")
    ax[0, 0].set_ylabel("ch0 [V]")

    ax[0, 1].set_title("mean ch1")
    ax[0, 1].set_xlabel(r"Time [$\mu$s]")
    ax[0, 1].set_ylabel("ch1 [V]")

    ax[1, 0].set_title("mean projected pulse")
    ax[1, 0].set_xlabel(r"Time [$\mu$s]")
    ax[1, 0].set_ylabel("projected signal [V]")

    ax[1, 1].set_title("pulse height comparison")
    ax[1, 1].set_xlabel("pulse height [V]")
    ax[1, 1].set_ylabel("counts")

    for a in ax.ravel():
        if "Time" in a.get_xlabel():
            a.axvline(0, ls="--", color="gray")

        a.grid(True)
        a.legend(fontsize=6)

    fig.suptitle(f"comparison: {base_dir}", fontsize=16)
    fig.tight_layout()

    outpath = out_dir / f"comparison_all_{base_dir}.png"

    if SAVE_FIG:
        fig.savefig(outpath, dpi=300)
        print("saved:", outpath)

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


def plot_scan_summary(summary_df, out_dir, base_dir):
    """
    周波数・位置依存性を見るための簡単な summary plot。
    pulse_height_median と snr_median を
    z scan / x scan ごとに描く。
    """
    if len(summary_df) == 0:
        return

    # ======================
    # 同じ freq, x で z を振っている scan
    # ======================
    for (freq, x), g in summary_df.groupby(["freq_ghz", "x_mm"]):
        if g["z_mm"].nunique() < 2:
            continue

        g = g.sort_values("z_mm")

        fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

        ax[0].plot(g["z_mm"], g["pulse_height_median"], marker="o")
        ax[0].set_ylabel("median pulse height [V]")
        ax[0].grid(True)

        ax[1].plot(g["z_mm"], g["snr_median"], marker="o")
        ax[1].set_xlabel("z [mm]")
        ax[1].set_ylabel("median SNR")
        ax[1].grid(True)

        fig.suptitle(f"z scan: {freq:.3f} GHz, x={x:.1f} mm")
        fig.tight_layout()

        outpath = out_dir / f"scan_z_{base_dir}_{freq:.3f}GHz_x{x:.1f}mm.png"

        if SAVE_FIG:
            fig.savefig(outpath, dpi=300)
            print("saved:", outpath)

        if SHOW_FIG:
            plt.show()
        else:
            plt.close(fig)

    # ======================
    # 同じ freq, z で x を振っている scan
    # ======================
    for (freq, z), g in summary_df.groupby(["freq_ghz", "z_mm"]):
        if g["x_mm"].nunique() < 2:
            continue

        g = g.sort_values("x_mm")

        fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

        ax[0].plot(g["x_mm"], g["pulse_height_median"], marker="o")
        ax[0].set_ylabel("median pulse height [V]")
        ax[0].grid(True)

        ax[1].plot(g["x_mm"], g["snr_median"], marker="o")
        ax[1].set_xlabel("x [mm]")
        ax[1].set_ylabel("median SNR")
        ax[1].grid(True)

        fig.suptitle(f"x scan: {freq:.3f} GHz, z={z:.1f} mm")
        fig.tight_layout()

        outpath = out_dir / f"scan_x_{base_dir}_{freq:.3f}GHz_z{z:.1f}mm.png"

        if SAVE_FIG:
            fig.savefig(outpath, dpi=300)
            print("saved:", outpath)

        if SHOW_FIG:
            plt.show()
        else:
            plt.close(fig)


# ======================
# main
# ======================
if not ROOT_DIR.exists():
    raise FileNotFoundError(f"日付フォルダが見つかりません: {ROOT_DIR}")

runs = build_runs_from_folder(ROOT_DIR)

print("DATA_DATE      :", DATA_DATE)
print("input ROOT_DIR :", ROOT_DIR)
print("output OUT_DIR :", OUT_DIR)
print("found runs     :", len(runs))

for run in runs:
    print(
        f"{run['dir']}  "
        f"freq={run['freq_ghz']:.3f}GHz  "
        f"z={run['z_mm']:.1f}mm  "
        f"x={run['x_mm']:.1f}mm  "
        f"tag={run['tag']}  "
        f"npz={run['n_npz']}"
    )

if len(runs) == 0:
    raise RuntimeError(
        "測定フォルダが見つかりませんでした。"
        "フォルダ名の形式を確認してください。"
    )

all_results = {}
summary_rows = []


# ======================
# 各測定フォルダごとの解析
# ======================
for run in runs:
    run_path = run["path"]
    label = run["label"]

    print()
    print("===== load:", label, "=====")

    try:
        time, ch0, ch1, meta = load_run_npz(run_path)
    except Exception as e:
        print("LOAD ERROR skip:", run_path, e)
        continue

    print("n_events =", ch0.shape[0])
    print("npts     =", ch0.shape[1])
    print("files    =", len(meta))

    try:
        result = analyze_waveforms(time, ch0, ch1)
    except Exception as e:
        print("ANALYZE ERROR skip:", run_path, e)
        continue

    all_results[label] = {
        "run": run,
        "time": time,
        "ch0": ch0,
        "ch1": ch1,
        "meta": meta,
        "result": result,
    }

    u0, u1 = result["signal_direction"]

    summary_rows.append({
        "run_dir": run["dir"],
        "freq_ghz": run["freq_ghz"],
        "z_mm": run["z_mm"],
        "x_mm": run["x_mm"],
        "tag": run["tag"],
        "label": label,
        "n_npz": run["n_npz"],
        "n_events": ch0.shape[0],
        "npts": ch0.shape[1],
        "signal_dir_ch0": u0,
        "signal_dir_ch1": u1,
        "t_peak_us": time[result["idx_peak"]] * 1e6,
        "pulse_height_mean": np.mean(result["pulse_height"]),
        "pulse_height_median": np.median(result["pulse_height"]),
        "pulse_height_std": np.std(result["pulse_height"], ddof=1),
        "baseline_noise_median": result["baseline_noise_median"],
        "snr_median": result["snr_median"],
    })

    # ======================
    # 各測定フォルダごとの summary plot
    # ======================
    outpath = OUT_DIR / f"summary_{base_dir}_{safe_name(run['dir'])}.png"

    plot_run_summary(
        time,
        result,
        label=label,
        outpath=outpath if SAVE_FIG else None,
    )

    # ======================
    # 各測定フォルダの meta も保存
    # ======================
    meta_path = OUT_DIR / f"meta_{base_dir}_{safe_name(run['dir'])}.csv"
    meta.to_csv(meta_path, index=False)
    print("saved:", meta_path)


# ======================
# summary table
# ======================
summary_df = pd.DataFrame(summary_rows)

print()
print("===== summary =====")
print(summary_df)

if len(summary_df) > 0:
    summary_csv = OUT_DIR / f"summary_{base_dir}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("saved:", summary_csv)

    # 全データ比較
    plot_comparison_all(all_results, OUT_DIR, base_dir)

    # x scan / z scan summary
    plot_scan_summary(summary_df, OUT_DIR, base_dir)

print()
print("done")
print("input ROOT_DIR :", ROOT_DIR)
print("output OUT_DIR :", OUT_DIR)