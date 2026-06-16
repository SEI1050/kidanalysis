from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

# ======================
# 画面表示せず保存だけ
# ======================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# ここだけ基本的に変える
# ============================================================

DATA_DATE = "20260527"

# 右4枚 overlay を作る対象周波数 [GHz]
TARGET_FREQ_GHZ = 5.501
FREQ_ATOL_GHZ = 1e-3

# ----------------------
# x依存 right4 overlay
# z=7.5固定でxを5点くらい比較
# x=3.4は含めたい
# ----------------------
X_SCAN_FIXED_Z_MM = 7.5
X_SCAN_CENTER_X_MM = 3.4
X_SCAN_N_POS = 5

# Noneなら中心に近い位置から自動選択
# 明示したい場合：
# X_SCAN_POSITIONS_MM = [1.4, 2.4, 3.4, 4.4, 5.4]
X_SCAN_POSITIONS_MM = None

# ----------------------
# z依存 right4 overlay
# x=3.4固定でzを5点くらい比較
# ----------------------
Z_SCAN_FIXED_X_MM = 3.4
Z_SCAN_CENTER_Z_MM = 7.5
Z_SCAN_N_POS = 5

# Noneなら中心に近い位置から自動選択
# 明示したい場合：
# Z_SCAN_POSITIONS_MM = [5.5, 6.5, 7.5, 8.5, 9.5]
Z_SCAN_POSITIONS_MM = None

# 同じ位置に _second, _third など複数runがある場合にまとめる
GROUP_REPEATS_BY_POSITION = True


# ============================================================
# 周波数ごとの scan overlay 設定
# ============================================================

# Noneなら見つかった全周波数を使う
# 特定の周波数だけ使いたい場合：
# FREQ_SCAN_GHZ_LIST = [5.451, 5.501, 5.551]
FREQ_SCAN_GHZ_LIST = None
FREQ_SCAN_ATOL_GHZ = 1e-3

# x scan overlay:
# z固定で、x依存を周波数ごとに同じ図へ重ねる
MULTIFREQ_XSCAN_FIXED_Z_MM = 7.5

# Noneなら、そのzに存在する全xを使う
# 5点くらいに絞りたいなら：
# MULTIFREQ_XSCAN_POSITIONS_MM = [1.4, 2.4, 3.4, 4.4, 5.4]
MULTIFREQ_XSCAN_POSITIONS_MM = None

# z scan overlay:
# x固定で、z依存を周波数ごとに同じ図へ重ねる
MULTIFREQ_ZSCAN_FIXED_X_MM = 3.4

# Noneなら、そのxに存在する全zを使う
# MULTIFREQ_ZSCAN_POSITIONS_MM = [5.5, 6.5, 7.5, 8.5, 9.5]
MULTIFREQ_ZSCAN_POSITIONS_MM = None

MULTIFREQ_GROUP_REPEATS = True


# ============================================================
# 解析設定
# ============================================================

NPZ_PATTERN = "wf_*.npz"

# 測定フォルダ直下だけ探すなら False
# さらに深い場所まで探すなら True
RECURSIVE_SEARCH = False

# baseline に使う時間範囲 [us]
# None にすると t < 0 全部を使う
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# pulse を探す時間範囲 [us]
PULSE_WINDOW_US = (0, None)
# PULSE_WINDOW_US = (0, 0.8)

HIST_BINS = 50
DPI = 300


# ============================================================
# ローカル側とOneDrive側
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "locationdepen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ============================================================
# 測定フォルダ名の読み取り
# 例:
# 5.501GHz_z=7.5mm_x=3.4mm
# 5.501GHz_z=7.5mm_x=3.4mm_second
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


def unique_sorted(values):
    return sorted(set(round(float(v), 6) for v in values))


def select_nearest_positions(values, center, n):
    """
    centerに近い順にn個選ぶ。
    centerが存在すれば自然に含まれる。
    最後は昇順に並べる。
    """
    values = unique_sorted(values)

    if len(values) == 0:
        return []

    chosen = sorted(values, key=lambda v: abs(v - center))[:n]
    chosen = sorted(chosen)

    return chosen


def choose_one_run_prefer_untagged(runs):
    """
    同じ位置に複数runがある場合、
    tagなしを優先して1つ選ぶ。
    """
    runs = sorted(runs, key=lambda r: (r["tag"] != "", r["tag"], r["dir"]))
    return runs[0]


def build_runs_from_folder(root_dir, target_freq_ghz=None, freq_atol_ghz=1e-3):
    runs = []

    for d in sorted(root_dir.iterdir()):
        if not d.is_dir():
            continue

        info = parse_meas_dir_name(d.name)

        if info is None:
            continue

        if target_freq_ghz is not None:
            if not np.isclose(
                info["freq_ghz"],
                target_freq_ghz,
                atol=freq_atol_ghz,
                rtol=0,
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


# ============================================================
# waveform utility
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


def find_npz_files(run_path):
    if RECURSIVE_SEARCH:
        files = list(run_path.rglob(NPZ_PATTERN))
    else:
        files = list(run_path.glob(NPZ_PATTERN))

    files = sorted(files, key=lambda p: p.stat().st_mtime)

    return files


def load_group_npz(group_runs):
    """
    複数の測定フォルダをまとめて読み込む。
    同じ位置の _second, _third などをまとめるために使う。
    """
    all_ch0 = []
    all_ch1 = []
    time_ref = None
    meta_rows = []

    for run in group_runs:
        files = find_npz_files(run["path"])

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
                "run_dir": run["dir"],
                "file": f.name,
                "nwaveform": ch0.shape[0],
                "npts": npts,
                "sample_rate": sample_rate,
                "ref_position": ref_position,
                "freq_ghz": run["freq_ghz"],
                "z_mm": run["z_mm"],
                "x_mm": run["x_mm"],
                "tag": run["tag"],
            })

    if len(all_ch0) == 0:
        dirs = [r["dir"] for r in group_runs]
        raise ValueError(f"no valid waveform in group: {dirs}")

    ch0_all = np.vstack(all_ch0)
    ch1_all = np.vstack(all_ch1)

    meta = pd.DataFrame(meta_rows)

    return time_ref, ch0_all, ch1_all, meta


def analyze_waveforms(time, ch0, ch1):
    """
    baseline subtraction
    mean waveform
    IQ direction projection
    pulse height
    SNR
    """
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

    # 平均IQの大きさが最大になる時刻をpeakにする
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

    # IQ射影
    proj = dch0 * u0 + dch1 * u1

    mean_proj = proj.mean(axis=0)
    std_proj = proj.std(axis=0, ddof=1) if n_events > 1 else np.zeros(npts)
    sem_proj = std_proj / np.sqrt(n_events)

    sign = np.sign(mean_proj[idx_peak])

    if sign == 0:
        sign = 1.0

    # 各イベントのpulse height
    pulse_height = sign * np.max(sign * proj[:, pulse_mask], axis=1)

    baseline_noise_each = proj[:, baseline_mask].std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_each = pulse_height / baseline_noise_each

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
        "baseline_noise_median": np.nanmedian(baseline_noise_each),
        "snr_median": np.nanmedian(snr_each),
    }

    return result


# ============================================================
# 右4枚 overlay 用
# ============================================================

def make_position_results(runs, mode):
    """
    mode = "xscan":
        z固定, xを変える

    mode = "zscan":
        x固定, zを変える
    """
    if mode == "xscan":
        fixed_z = X_SCAN_FIXED_Z_MM

        candidates = [
            r for r in runs
            if close(r["z_mm"], fixed_z)
        ]

        available = unique_sorted([r["x_mm"] for r in candidates])

        if X_SCAN_POSITIONS_MM is None:
            selected_positions = select_nearest_positions(
                available,
                X_SCAN_CENTER_X_MM,
                X_SCAN_N_POS,
            )
        else:
            selected_positions = X_SCAN_POSITIONS_MM

        group_key = "x_mm"
        fixed_text = f"{TARGET_FREQ_GHZ:.3f} GHz, z={fixed_z:.1f} mm"
        title_text = f"x dependence at z={fixed_z:.1f} mm"

    elif mode == "zscan":
        fixed_x = Z_SCAN_FIXED_X_MM

        candidates = [
            r for r in runs
            if close(r["x_mm"], fixed_x)
        ]

        available = unique_sorted([r["z_mm"] for r in candidates])

        if Z_SCAN_POSITIONS_MM is None:
            selected_positions = select_nearest_positions(
                available,
                Z_SCAN_CENTER_Z_MM,
                Z_SCAN_N_POS,
            )
        else:
            selected_positions = Z_SCAN_POSITIONS_MM

        group_key = "z_mm"
        fixed_text = f"{TARGET_FREQ_GHZ:.3f} GHz, x={fixed_x:.1f} mm"
        title_text = f"z dependence at x={fixed_x:.1f} mm"

    else:
        raise ValueError("mode must be 'xscan' or 'zscan'")

    print()
    print("mode:", mode)
    print("available positions:", available)
    print("selected positions :", selected_positions)

    results = []

    for pos in selected_positions:
        group = [
            r for r in candidates
            if close(r[group_key], pos)
        ]

        if len(group) == 0:
            print("skip no run at", group_key, pos)
            continue

        if GROUP_REPEATS_BY_POSITION:
            group_runs = group
        else:
            group_runs = [choose_one_run_prefer_untagged(group)]

        print()
        print(f"===== {mode} {group_key}={pos:.1f} =====")
        print("use runs:")
        for r in group_runs:
            print("  ", r["dir"])

        try:
            time, ch0, ch1, meta = load_group_npz(group_runs)
            result = analyze_waveforms(time, ch0, ch1)
        except Exception as e:
            print("ERROR skip:", pos, e)
            continue

        if mode == "xscan":
            label = f"x={pos:.1f} mm"
        else:
            label = f"z={pos:.1f} mm"

        results.append({
            "position": pos,
            "label": label,
            "group_runs": group_runs,
            "time": time,
            "ch0": ch0,
            "ch1": ch1,
            "meta": meta,
            "result": result,
            "n_events": ch0.shape[0],
        })

    return results, fixed_text, title_text


def common_hist_bins(results, bins=50):
    vals = []

    for item in results:
        ph = item["result"]["pulse_height"]
        ph = ph[np.isfinite(ph)]
        vals.append(ph)

    if len(vals) == 0:
        return bins

    vals = np.concatenate(vals)

    if len(vals) == 0:
        return bins

    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return bins

    return np.linspace(vmin, vmax, bins + 1)


def plot_right4_overlay(results, fixed_text, title_text, outpath):
    """
    右側4枚を位置ごとに重ねる：
    average waveform
    projected pulse
    mean IQ trajectory
    pulse height histogram
    """
    if len(results) == 0:
        print("no results to plot:", title_text)
        return

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax_avg = ax[0, 0]
    ax_proj = ax[0, 1]
    ax_iq = ax[1, 0]
    ax_hist = ax[1, 1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hist_bins = common_hist_bins(results, bins=HIST_BINS)

    for i, item in enumerate(results):
        color = colors[i % len(colors)]

        time = item["time"]
        time_us = time * 1e6

        res = item["result"]

        mean0 = res["mean0"]
        mean1 = res["mean1"]
        mean_proj = res["mean_proj"]
        pulse_height = res["pulse_height"]
        idx_peak = res["idx_peak"]

        label = f"{item['label']}  n={item['n_events']}"

        # average waveform
        ax_avg.plot(
            time_us,
            mean0,
            color=color,
            ls="-",
            lw=1.8,
            label=f"{item['label']} ch0",
        )

        ax_avg.plot(
            time_us,
            mean1,
            color=color,
            ls="--",
            lw=1.8,
            label=f"{item['label']} ch1",
        )

        # projected pulse
        ax_proj.plot(
            time_us,
            mean_proj,
            color=color,
            lw=2.0,
            label=label,
        )

        ax_proj.scatter(
            time_us[idx_peak],
            mean_proj[idx_peak],
            color=color,
            edgecolors="black",
            s=45,
            zorder=5,
        )

        # mean IQ trajectory
        ax_iq.plot(
            mean0,
            mean1,
            color=color,
            lw=1.8,
            marker=".",
            ms=2.5,
            label=label,
        )

        ax_iq.scatter(
            mean0[0],
            mean1[0],
            color=color,
            marker="*",
            edgecolors="black",
            s=80,
            zorder=5,
        )

        ax_iq.scatter(
            mean0[idx_peak],
            mean1[idx_peak],
            color=color,
            marker="o",
            edgecolors="black",
            s=55,
            zorder=6,
        )

        # pulse height histogram
        ax_hist.hist(
            pulse_height,
            bins=hist_bins,
            histtype="step",
            lw=1.8,
            color=color,
            label=label,
        )

    ax_avg.axvline(0, ls="--", color="gray", lw=1.2)
    ax_avg.set_title("average waveform")
    ax_avg.set_xlabel(r"Time [$\mu$s]")
    ax_avg.set_ylabel("Voltage [V]")
    ax_avg.grid(True)
    ax_avg.legend(fontsize=7, ncols=2)

    ax_proj.axvline(0, ls="--", color="gray", lw=1.2)
    ax_proj.set_title("projected pulse")
    ax_proj.set_xlabel(r"Time [$\mu$s]")
    ax_proj.set_ylabel("projected signal [V]")
    ax_proj.grid(True)
    ax_proj.legend(fontsize=8)

    ax_iq.set_title("mean IQ trajectory")
    ax_iq.set_xlabel("ch0")
    ax_iq.set_ylabel("ch1")
    ax_iq.axis("equal")
    ax_iq.grid(True)
    ax_iq.legend(fontsize=8)

    ax_hist.set_title("pulse height histogram")
    ax_hist.set_xlabel("pulse height [V]")
    ax_hist.set_ylabel("counts")
    ax_hist.grid(True)
    ax_hist.legend(fontsize=8)

    fig.suptitle(f"{fixed_text}  {title_text}", fontsize=16)
    fig.tight_layout()

    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)

    print("saved:", outpath)


def results_to_summary_df(results, mode, freq_ghz):
    rows = []

    for item in results:
        res = item["result"]
        u0, u1 = res["signal_direction"]

        row = {
            "mode": mode,
            "freq_ghz": freq_ghz,
            "position": item["position"],
            "label": item["label"],
            "n_events": item["n_events"],
            "n_runs": len(item["group_runs"]),
            "run_dirs": ";".join([r["dir"] for r in item["group_runs"]]),
            "signal_dir_ch0": u0,
            "signal_dir_ch1": u1,
            "t_peak_us": item["time"][res["idx_peak"]] * 1e6,
            "pulse_height_mean": np.nanmean(res["pulse_height"]),
            "pulse_height_median": np.nanmedian(res["pulse_height"]),
            "pulse_height_std": np.nanstd(res["pulse_height"], ddof=1),
            "baseline_noise_median": res["baseline_noise_median"],
            "snr_median": res["snr_median"],
        }

        if mode == "xscan":
            row["x_mm"] = item["position"]
            row["z_mm"] = X_SCAN_FIXED_Z_MM
        else:
            row["z_mm"] = item["position"]
            row["x_mm"] = Z_SCAN_FIXED_X_MM

        rows.append(row)

    return pd.DataFrame(rows)


def save_results_summary(results, mode, freq_ghz, outpath):
    df = results_to_summary_df(results, mode, freq_ghz)
    df.to_csv(outpath, index=False)
    print("saved:", outpath)
    return df


def plot_singlefreq_scan(summary_df, mode, outpath):
    """
    1つの周波数に対する x scan / z scan。
    上段: median pulse height
    下段: median SNR
    """
    if len(summary_df) == 0:
        print("no summary data:", mode)
        return

    if mode == "xscan":
        xcol = "x_mm"
        xlabel = "x [mm]"
        title = f"x scan: {TARGET_FREQ_GHZ:.3f} GHz, z={X_SCAN_FIXED_Z_MM:.1f} mm"
    elif mode == "zscan":
        xcol = "z_mm"
        xlabel = "z [mm]"
        title = f"z scan: {TARGET_FREQ_GHZ:.3f} GHz, x={Z_SCAN_FIXED_X_MM:.1f} mm"
    else:
        raise ValueError("mode must be 'xscan' or 'zscan'")

    g = summary_df.sort_values(xcol)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(
        g[xcol],
        g["pulse_height_median"],
        marker="o",
        lw=2,
    )
    ax[0].set_ylabel("median pulse height [V]")
    ax[0].grid(True)

    ax[1].plot(
        g[xcol],
        g["snr_median"],
        marker="o",
        lw=2,
    )
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("median SNR")
    ax[1].grid(True)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)

    print("saved:", outpath)


# ============================================================
# 周波数ごとの scan overlay 用
# ============================================================

def get_frequency_list_for_multifreq_scan(runs):
    freqs = unique_sorted([r["freq_ghz"] for r in runs])

    if FREQ_SCAN_GHZ_LIST is None:
        return freqs

    selected = []

    for f0 in FREQ_SCAN_GHZ_LIST:
        matched = [
            f for f in freqs
            if np.isclose(f, f0, atol=FREQ_SCAN_ATOL_GHZ, rtol=0)
        ]

        if len(matched) == 0:
            print(f"WARNING: requested freq not found: {f0:.3f} GHz")
            continue

        selected.append(matched[0])

    return unique_sorted(selected)


def make_multifreq_scan_summary(runs, mode):
    """
    mode = "xscan":
        z固定でx依存を周波数ごとに作る

    mode = "zscan":
        x固定でz依存を周波数ごとに作る
    """
    rows = []
    freqs = get_frequency_list_for_multifreq_scan(runs)

    print()
    print("===== multifrequency scan summary =====")
    print("mode:", mode)
    print("freqs:", freqs)

    for freq in freqs:
        if mode == "xscan":
            fixed_z = MULTIFREQ_XSCAN_FIXED_Z_MM

            candidates = [
                r for r in runs
                if np.isclose(r["freq_ghz"], freq, atol=FREQ_SCAN_ATOL_GHZ, rtol=0)
                and close(r["z_mm"], fixed_z)
            ]

            available_positions = unique_sorted([r["x_mm"] for r in candidates])

            if MULTIFREQ_XSCAN_POSITIONS_MM is None:
                selected_positions = available_positions
            else:
                selected_positions = MULTIFREQ_XSCAN_POSITIONS_MM

            position_key = "x_mm"

        elif mode == "zscan":
            fixed_x = MULTIFREQ_ZSCAN_FIXED_X_MM

            candidates = [
                r for r in runs
                if np.isclose(r["freq_ghz"], freq, atol=FREQ_SCAN_ATOL_GHZ, rtol=0)
                and close(r["x_mm"], fixed_x)
            ]

            available_positions = unique_sorted([r["z_mm"] for r in candidates])

            if MULTIFREQ_ZSCAN_POSITIONS_MM is None:
                selected_positions = available_positions
            else:
                selected_positions = MULTIFREQ_ZSCAN_POSITIONS_MM

            position_key = "z_mm"

        else:
            raise ValueError("mode must be 'xscan' or 'zscan'")

        print()
        print(f"--- freq={freq:.3f} GHz ---")
        print("available positions:", available_positions)
        print("selected positions :", selected_positions)

        for pos in selected_positions:
            group = [
                r for r in candidates
                if close(r[position_key], pos)
            ]

            if len(group) == 0:
                print(f"skip no data: freq={freq:.3f}, {position_key}={pos:.1f}")
                continue

            if MULTIFREQ_GROUP_REPEATS:
                group_runs = group
            else:
                group_runs = [choose_one_run_prefer_untagged(group)]

            print(
                f"analyze freq={freq:.3f} GHz, "
                f"{position_key}={pos:.1f}, "
                f"runs={len(group_runs)}"
            )

            try:
                time, ch0, ch1, meta = load_group_npz(group_runs)
                result = analyze_waveforms(time, ch0, ch1)
            except Exception as e:
                print("ERROR skip:", e)
                continue

            u0, u1 = result["signal_direction"]

            row = {
                "mode": mode,
                "freq_ghz": freq,
                "position_mm": pos,
                "n_events": ch0.shape[0],
                "n_runs": len(group_runs),
                "run_dirs": ";".join([r["dir"] for r in group_runs]),
                "signal_dir_ch0": u0,
                "signal_dir_ch1": u1,
                "t_peak_us": time[result["idx_peak"]] * 1e6,
                "pulse_height_mean": np.nanmean(result["pulse_height"]),
                "pulse_height_median": np.nanmedian(result["pulse_height"]),
                "pulse_height_std": np.nanstd(result["pulse_height"], ddof=1),
                "baseline_noise_median": result["baseline_noise_median"],
                "snr_median": result["snr_median"],
            }

            if mode == "xscan":
                row["x_mm"] = pos
                row["z_mm"] = MULTIFREQ_XSCAN_FIXED_Z_MM
            else:
                row["z_mm"] = pos
                row["x_mm"] = MULTIFREQ_ZSCAN_FIXED_X_MM

            rows.append(row)

    return pd.DataFrame(rows)


def plot_multifreq_scan_overlay(scan_df, mode, outpath):
    """
    周波数ごとの x scan / z scan を同じ図に重ねる。
    上段: median pulse height
    下段: median SNR
    """
    if len(scan_df) == 0:
        print("no scan data to plot:", mode)
        return

    if mode == "xscan":
        xcol = "x_mm"
        xlabel = "x [mm]"
        title = f"x scan overlay by frequency: z={MULTIFREQ_XSCAN_FIXED_Z_MM:.1f} mm"

    elif mode == "zscan":
        xcol = "z_mm"
        xlabel = "z [mm]"
        title = f"z scan overlay by frequency: x={MULTIFREQ_ZSCAN_FIXED_X_MM:.1f} mm"

    else:
        raise ValueError("mode must be 'xscan' or 'zscan'")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    freqs = sorted(scan_df["freq_ghz"].dropna().unique())

    for freq in freqs:
        g = scan_df[np.isclose(scan_df["freq_ghz"], freq)]
        g = g.sort_values(xcol)

        if len(g) == 0:
            continue

        label = f"{freq:.3f} GHz"

        ax[0].plot(
            g[xcol],
            g["pulse_height_median"],
            marker="o",
            lw=2,
            label=label,
        )

        ax[1].plot(
            g[xcol],
            g["snr_median"],
            marker="o",
            lw=2,
            label=label,
        )

    ax[0].set_ylabel("median pulse height [V]")
    ax[0].grid(True)
    ax[0].legend(fontsize=8)

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("median SNR")
    ax[1].grid(True)
    ax[1].legend(fontsize=8)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)

    print("saved:", outpath)


# ============================================================
# main
# ============================================================

print("DATA_DATE       :", DATA_DATE)
print("TARGET_FREQ_GHZ :", TARGET_FREQ_GHZ)
print("input ROOT_DIR  :", ROOT_DIR)
print("output OUT_DIR  :", OUT_DIR)

# 全周波数run
all_runs = build_runs_from_folder(ROOT_DIR, target_freq_ghz=None)

# 右4枚overlay用の特定周波数run
runs = build_runs_from_folder(
    ROOT_DIR,
    target_freq_ghz=TARGET_FREQ_GHZ,
    freq_atol_ghz=FREQ_ATOL_GHZ,
)

print("found all runs          :", len(all_runs))
print("found target freq runs  :", len(runs))

if len(all_runs) == 0:
    raise RuntimeError(
        "測定フォルダが見つかりませんでした。"
        "フォルダ名やDATA_DATEを確認してください。"
    )

if len(runs) == 0:
    print(
        f"WARNING: TARGET_FREQ_GHZ={TARGET_FREQ_GHZ:.3f} GHz のrunが見つかりません。"
        "right4 overlay と singlefreq scan は作られません。"
    )

for r in runs:
    print(
        f"{r['dir']}  "
        f"freq={r['freq_ghz']:.3f}GHz  "
        f"z={r['z_mm']:.1f}mm  "
        f"x={r['x_mm']:.1f}mm  "
        f"tag={r['tag']}  "
        f"npz={r['n_npz']}"
    )


# ============================================================
# 1. right4 x依存: z=7.5固定、xを変える
# ============================================================

if len(runs) > 0:
    x_results, x_fixed_text, x_title_text = make_position_results(runs, mode="xscan")

    x_png = OUT_DIR / (
        f"right4_overlay_xscan_"
        f"{safe_name(f'{TARGET_FREQ_GHZ:.3f}GHz')}_"
        f"z{safe_name(f'{X_SCAN_FIXED_Z_MM:.1f}mm')}.png"
    )

    plot_right4_overlay(
        x_results,
        fixed_text=x_fixed_text,
        title_text=x_title_text,
        outpath=x_png,
    )

    x_csv = OUT_DIR / (
        f"right4_overlay_xscan_summary_"
        f"{safe_name(f'{TARGET_FREQ_GHZ:.3f}GHz')}_"
        f"z{safe_name(f'{X_SCAN_FIXED_Z_MM:.1f}mm')}.csv"
    )

    x_df = save_results_summary(
        x_results,
        mode="xscan",
        freq_ghz=TARGET_FREQ_GHZ,
        outpath=x_csv,
    )

    x_single_png = OUT_DIR / (
        f"singlefreq_xscan_"
        f"{safe_name(f'{TARGET_FREQ_GHZ:.3f}GHz')}_"
        f"z{safe_name(f'{X_SCAN_FIXED_Z_MM:.1f}mm')}.png"
    )

    plot_singlefreq_scan(
        x_df,
        mode="xscan",
        outpath=x_single_png,
    )


# ============================================================
# 2. right4 z依存: x=3.4固定、zを変える
# ============================================================

if len(runs) > 0:
    z_results, z_fixed_text, z_title_text = make_position_results(runs, mode="zscan")

    z_png = OUT_DIR / (
        f"right4_overlay_zscan_"
        f"{safe_name(f'{TARGET_FREQ_GHZ:.3f}GHz')}_"
        f"x{safe_name(f'{Z_SCAN_FIXED_X_MM:.1f}mm')}.png"
    )

    plot_right4_overlay(
        z_results,
        fixed_text=z_fixed_text,
        title_text=z_title_text,
        outpath=z_png,
    )

    z_csv = OUT_DIR / (
        f"right4_overlay_zscan_summary_"
        f"{safe_name(f'{TARGET_FREQ_GHZ:.3f}GHz')}_"
        f"x{safe_name(f'{Z_SCAN_FIXED_X_MM:.1f}mm')}.csv"
    )

    z_df = save_results_summary(
        z_results,
        mode="zscan",
        freq_ghz=TARGET_FREQ_GHZ,
        outpath=z_csv,
    )

    z_single_png = OUT_DIR / (
        f"singlefreq_zscan_"
        f"{safe_name(f'{TARGET_FREQ_GHZ:.3f}GHz')}_"
        f"x{safe_name(f'{Z_SCAN_FIXED_X_MM:.1f}mm')}.png"
    )

    plot_singlefreq_scan(
        z_df,
        mode="zscan",
        outpath=z_single_png,
    )


# ============================================================
# 3. multifrequency x scan overlay
# z固定で、x scan を周波数ごとに重ねる
# ============================================================

mf_x_df = make_multifreq_scan_summary(all_runs, mode="xscan")

mf_x_csv = OUT_DIR / (
    f"multifreq_xscan_summary_"
    f"z{safe_name(f'{MULTIFREQ_XSCAN_FIXED_Z_MM:.1f}mm')}.csv"
)

mf_x_df.to_csv(mf_x_csv, index=False)
print("saved:", mf_x_csv)

mf_x_png = OUT_DIR / (
    f"multifreq_xscan_overlay_"
    f"z{safe_name(f'{MULTIFREQ_XSCAN_FIXED_Z_MM:.1f}mm')}.png"
)

plot_multifreq_scan_overlay(
    mf_x_df,
    mode="xscan",
    outpath=mf_x_png,
)


# ============================================================
# 4. multifrequency z scan overlay
# x固定で、z scan を周波数ごとに重ねる
# ============================================================

mf_z_df = make_multifreq_scan_summary(all_runs, mode="zscan")

mf_z_csv = OUT_DIR / (
    f"multifreq_zscan_summary_"
    f"x{safe_name(f'{MULTIFREQ_ZSCAN_FIXED_X_MM:.1f}mm')}.csv"
)

mf_z_df.to_csv(mf_z_csv, index=False)
print("saved:", mf_z_csv)

mf_z_png = OUT_DIR / (
    f"multifreq_zscan_overlay_"
    f"x{safe_name(f'{MULTIFREQ_ZSCAN_FIXED_X_MM:.1f}mm')}.png"
)

plot_multifreq_scan_overlay(
    mf_z_df,
    mode="zscan",
    outpath=mf_z_png,
)


print()
print("done")
print("outputs saved in:", OUT_DIR)