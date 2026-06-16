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

# Noneなら全周波数
# 特定周波数だけなら [5.451, 5.501] など
TARGET_FREQ_GHZ_LIST = None
# TARGET_FREQ_GHZ_LIST = [5.451]

FREQ_ATOL_GHZ = 1e-3

# x scan: z固定でx依存を見る
XSCAN_FIXED_Z_MM = 7.5

# z scan: x固定でz依存を見る
ZSCAN_FIXED_X_MM = 3.4

# 同じ freq, z, x に _second, _third などがある場合、
# Trueなら全部まとめる
GROUP_REPEATS_BY_POSITION = True

# 入力モード
# "cloud" : OneDrive / CloudStorage側
# "local" : KIDANALYSIS/data/20260527側
# "both"  : 両方
INPUT_MODE = "cloud"

# npz探索
NPZ_PATTERN = "wf_*.npz"
RECURSIVE_SEARCH = False

# baseline範囲 [us]
# Noneなら t < 0 全部
BASELINE_WINDOW_US = None
# BASELINE_WINDOW_US = (-0.30, -0.05)

# pulse peakを探す範囲 [us]
PULSE_WINDOW_US = (0, None)
# PULSE_WINDOW_US = (0, 1.0)

# 図の時間表示範囲 [us]
XLIM_US = (-0.3, 1.6)

# SEM帯を描くか
PLOT_SEM = False

# peakで規格化した重ね書きも保存するか
SAVE_NORMALIZED = True

DPI = 300


# ============================================================
# パス設定
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

OUT_DIR = HERE / "data" / DATA_DATE / "projected_waveform_scan"
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

# 必要なら手動パスをここへ追加
EXTRA_INPUT_ROOTS = [
    # Path("/Users/kubokosei/Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit/20260527"),
]


def expand_and_resolve_path(p):
    return Path(p).expanduser().resolve(strict=False)


def collect_input_roots():
    roots = []

    print()
    print("===== path check =====")
    print("HERE:", HERE)
    print("INPUT_MODE:", INPUT_MODE)
    print()

    candidates = []

    if INPUT_MODE in ["local", "both"]:
        candidates.append(("local", local_data_dir))

    if INPUT_MODE in ["cloud", "both"]:
        for p in cloud_data_candidates:
            candidates.append(("cloud", p))

    for p in EXTRA_INPUT_ROOTS:
        candidates.append(("extra", p))

    seen = set()

    for kind, p in candidates:
        p = expand_and_resolve_path(p)
        exists = p.is_dir()

        print(f"[{kind}] {p}")
        print("   exists:", exists)

        if exists:
            key = p.as_posix()
            if key not in seen:
                roots.append(p)
                seen.add(key)

    print()
    print("selected input roots:")
    for r in roots:
        print("  ", r)

    if len(roots) == 0:
        print("ERROR: 入力フォルダが見つかりません。")
        sys.exit(1)

    return roots


INPUT_ROOTS = collect_input_roots()


# ============================================================
# 測定フォルダ名読み取り
# 例:
# 5.451GHz_z=7.5mm_x=3.4mm
# 5.451GHz_z=7.5mm_x=3.4mm_second
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


def select_freqs(all_freqs):
    freqs = unique_sorted(all_freqs)

    if TARGET_FREQ_GHZ_LIST is None:
        return freqs

    selected = []

    for f0 in TARGET_FREQ_GHZ_LIST:
        matched = [
            f for f in freqs
            if np.isclose(f, f0, atol=FREQ_ATOL_GHZ, rtol=0)
        ]

        if len(matched) == 0:
            print(f"WARNING: requested frequency not found: {f0:.3f} GHz")
            continue

        selected.append(matched[0])

    return unique_sorted(selected)


def build_runs():
    runs = []

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


def analyze_projected_waveform(time, ch0, ch1):
    n_events, npts = ch0.shape

    baseline_default = time < 0
    baseline_mask = make_mask_us(
        time,
        BASELINE_WINDOW_US,
        default_mask=baseline_default,
    )

    if baseline_mask.sum() < 2:
        raise ValueError("baseline points too few")

    base0 = ch0[:, baseline_mask].mean(axis=1, keepdims=True)
    base1 = ch1[:, baseline_mask].mean(axis=1, keepdims=True)

    dch0 = ch0 - base0
    dch1 = ch1 - base1

    mean0 = dch0.mean(axis=0)
    mean1 = dch1.mean(axis=0)

    pulse_default = time >= 0
    pulse_mask = make_mask_us(
        time,
        PULSE_WINDOW_US,
        default_mask=pulse_default,
    )

    if pulse_mask.sum() < 2:
        raise ValueError("pulse points too few")

    # 平均IQの大きさ最大をpeakとする
    r_mean = np.sqrt(mean0**2 + mean1**2)
    pulse_indices = np.where(pulse_mask)[0]
    idx_peak = pulse_indices[np.argmax(r_mean[pulse_mask])]

    # peak方向を射影方向にする
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

    # peakが正になるように符号をそろえる
    sign = np.sign(mean_proj[idx_peak])
    if sign == 0:
        sign = 1.0

    proj *= sign
    mean_proj *= sign

    pulse_height = np.max(proj[:, pulse_mask], axis=1)
    baseline_noise_each = proj[:, baseline_mask].std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_each = pulse_height / baseline_noise_each

    return {
        "dch0": dch0,
        "dch1": dch1,
        "mean0": mean0,
        "mean1": mean1,
        "proj": proj,
        "mean_proj": mean_proj,
        "sem_proj": sem_proj,
        "idx_peak": idx_peak,
        "signal_direction": (u0, u1),
        "pulse_height": pulse_height,
        "baseline_noise_each": baseline_noise_each,
        "snr_each": snr_each,
        "pulse_height_median": np.nanmedian(pulse_height),
        "snr_median": np.nanmedian(snr_each),
    }


# ============================================================
# scan result
# ============================================================

def choose_one_run_prefer_untagged(runs):
    runs = sorted(runs, key=lambda r: (r["tag"] != "", r["tag"], r["dir"]))
    return runs[0]


def make_scan_results(runs, freq, mode):
    if mode == "xscan":
        candidates = [
            r for r in runs
            if np.isclose(r["freq_ghz"], freq, atol=FREQ_ATOL_GHZ, rtol=0)
            and close(r["z_mm"], XSCAN_FIXED_Z_MM)
        ]

        pos_col = "x_mm"
        fixed_text = f"f={freq:.3f} GHz, z={XSCAN_FIXED_Z_MM:.1f} mm"
        title = f"x scan projected waveform overlay"

    elif mode == "zscan":
        candidates = [
            r for r in runs
            if np.isclose(r["freq_ghz"], freq, atol=FREQ_ATOL_GHZ, rtol=0)
            and close(r["x_mm"], ZSCAN_FIXED_X_MM)
        ]

        pos_col = "z_mm"
        fixed_text = f"f={freq:.3f} GHz, x={ZSCAN_FIXED_X_MM:.1f} mm"
        title = f"z scan projected waveform overlay"

    else:
        raise ValueError("mode must be xscan or zscan")

    positions = unique_sorted([r[pos_col] for r in candidates])

    print()
    print(f"make_scan_results: freq={freq:.3f}, mode={mode}")
    print("positions:", positions)

    results = []

    for pos in positions:
        group = [
            r for r in candidates
            if close(r[pos_col], pos)
        ]

        if len(group) == 0:
            continue

        if GROUP_REPEATS_BY_POSITION:
            group_runs = group
        else:
            group_runs = [choose_one_run_prefer_untagged(group)]

        print(f"  {pos:.1f} mm: runs={len(group_runs)}")

        try:
            time, ch0, ch1, meta = load_group_npz(group_runs)
            result = analyze_projected_waveform(time, ch0, ch1)
        except Exception as e:
            print("  ERROR skip:", pos, e)
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
            "n_events": ch0.shape[0],
            "meta": meta,
            "result": result,
        })

    return results, fixed_text, title


def save_scan_summary(results, freq, mode, out_csv):
    rows = []

    for item in results:
        res = item["result"]
        u0, u1 = res["signal_direction"]

        row = {
            "mode": mode,
            "freq_ghz": freq,
            "position_mm": item["position"],
            "label": item["label"],
            "n_events": item["n_events"],
            "n_runs": len(item["group_runs"]),
            "run_dirs": ";".join([r["dir"] for r in item["group_runs"]]),
            "signal_dir_ch0": u0,
            "signal_dir_ch1": u1,
            "t_peak_us": item["time"][res["idx_peak"]] * 1e6,
            "peak_projected_mean": res["mean_proj"][res["idx_peak"]],
            "pulse_height_median": res["pulse_height_median"],
            "snr_median": res["snr_median"],
        }

        if mode == "xscan":
            row["x_mm"] = item["position"]
            row["z_mm"] = XSCAN_FIXED_Z_MM
        else:
            row["z_mm"] = item["position"]
            row["x_mm"] = ZSCAN_FIXED_X_MM

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)


# ============================================================
# plot
# ============================================================

def plot_projected_overlay(results, freq, mode, fixed_text, title, out_png, normalize=False):
    if len(results) == 0:
        print("no results to plot:", freq, mode)
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, item in enumerate(results):
        color = colors[i % len(colors)]

        time_us = item["time"] * 1e6
        mean_proj = item["result"]["mean_proj"].copy()
        sem_proj = item["result"]["sem_proj"].copy()
        idx_peak = item["result"]["idx_peak"]

        peak = mean_proj[idx_peak]

        if normalize:
            if peak != 0 and np.isfinite(peak):
                mean_proj = mean_proj / peak
                sem_proj = sem_proj / abs(peak)

        label = f"{item['label']} n={item['n_events']}"

        ax.plot(
            time_us,
            mean_proj,
            color=color,
            lw=2.0,
            label=label,
        )

        if PLOT_SEM:
            ax.fill_between(
                time_us,
                mean_proj - sem_proj,
                mean_proj + sem_proj,
                color=color,
                alpha=0.15,
            )

        ax.scatter(
            time_us[idx_peak],
            mean_proj[idx_peak],
            color=color,
            edgecolors="black",
            s=45,
            zorder=5,
        )

    ax.axvline(0, ls="--", color="gray", lw=1.2)
    ax.grid(True)

    ax.set_xlim(*XLIM_US)
    ax.set_xlabel(r"Time [$\mu$s]")

    if normalize:
        ax.set_ylabel("normalized projected signal")
        norm_text = "normalized by peak"
    else:
        ax.set_ylabel("projected signal [V]")
        norm_text = "raw amplitude"

    ax.set_title(f"{title}\n{fixed_text}, {norm_text}")

    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
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

print("found runs:", len(runs))

if len(runs) == 0:
    raise RuntimeError("測定フォルダまたはwf_*.npzが見つかりませんでした。")

freqs = select_freqs([r["freq_ghz"] for r in runs])

print("frequencies:")
for f in freqs:
    print(f"  {f:.3f} GHz")

for freq in freqs:
    print()
    print("============================================================")
    print(f"frequency = {freq:.3f} GHz")
    print("============================================================")

    # ----------------------
    # x scan
    # ----------------------
    x_results, x_fixed_text, x_title = make_scan_results(
        runs,
        freq=freq,
        mode="xscan",
    )

    x_csv = OUT_DIR / (
        f"projected_xscan_summary_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"z{safe_name(f'{XSCAN_FIXED_Z_MM:.1f}mm')}.csv"
    )

    save_scan_summary(
        x_results,
        freq=freq,
        mode="xscan",
        out_csv=x_csv,
    )

    x_png = OUT_DIR / (
        f"projected_xscan_overlay_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"z{safe_name(f'{XSCAN_FIXED_Z_MM:.1f}mm')}.png"
    )

    plot_projected_overlay(
        x_results,
        freq=freq,
        mode="xscan",
        fixed_text=x_fixed_text,
        title=x_title,
        out_png=x_png,
        normalize=False,
    )

    if SAVE_NORMALIZED:
        x_norm_png = OUT_DIR / (
            f"projected_xscan_overlay_normalized_"
            f"{safe_name(f'{freq:.3f}GHz')}_"
            f"z{safe_name(f'{XSCAN_FIXED_Z_MM:.1f}mm')}.png"
        )

        plot_projected_overlay(
            x_results,
            freq=freq,
            mode="xscan",
            fixed_text=x_fixed_text,
            title=x_title,
            out_png=x_norm_png,
            normalize=True,
        )

    # ----------------------
    # z scan
    # ----------------------
    z_results, z_fixed_text, z_title = make_scan_results(
        runs,
        freq=freq,
        mode="zscan",
    )

    z_csv = OUT_DIR / (
        f"projected_zscan_summary_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"x{safe_name(f'{ZSCAN_FIXED_X_MM:.1f}mm')}.csv"
    )

    save_scan_summary(
        z_results,
        freq=freq,
        mode="zscan",
        out_csv=z_csv,
    )

    z_png = OUT_DIR / (
        f"projected_zscan_overlay_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"x{safe_name(f'{ZSCAN_FIXED_X_MM:.1f}mm')}.png"
    )

    plot_projected_overlay(
        z_results,
        freq=freq,
        mode="zscan",
        fixed_text=z_fixed_text,
        title=z_title,
        out_png=z_png,
        normalize=False,
    )

    if SAVE_NORMALIZED:
        z_norm_png = OUT_DIR / (
            f"projected_zscan_overlay_normalized_"
            f"{safe_name(f'{freq:.3f}GHz')}_"
            f"x{safe_name(f'{ZSCAN_FIXED_X_MM:.1f}mm')}.png"
        )

        plot_projected_overlay(
            z_results,
            freq=freq,
            mode="zscan",
            fixed_text=z_fixed_text,
            title=z_title,
            out_png=z_norm_png,
            normalize=True,
        )

print()
print("done")
print("outputs saved in:", OUT_DIR)