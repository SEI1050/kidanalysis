from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

DATA_DATE = "20260527"

# phase binを何分割にするか
N_PHASE_BINS_NEW = 5

# 読み込むevents CSV
# Noneなら phase50_amp_ped/events_*.csv を全部処理
EVENT_CSV = None
# EVENT_CSV = "/Users/kubokosei/software/kidanalysis/data/20260527/phase50_amp_ped/events_20260527_5p451GHz_z7p5mm_x3p4mm_phase50_projected.csv"

# histogram設定
HIST_BINS = 60

# 外れ値を除いてbin範囲を決める
USE_QUANTILE_RANGE = True
Q_LOW = 1
Q_HIGH = 99

# 縦軸をイベント数にする
DENSITY = False

# 表示する列
PED_COLUMN = "ped1"

# amp列は自動探索する
# phase50_amp_ped.py由来なら "amp"
# integral_tau_eff.py由来なら "amp_H"
AMP_COLUMN_CANDIDATES = ["amp", "amp1", "amp_H"]

DPI = 300


# ============================================================
# path
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

IN_DIR = HERE / "data" / DATA_DATE / "phase50_amp_ped"
OUT_DIR = HERE / "data" / DATA_DATE / "phasebin10_hist"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def find_event_csvs():
    if EVENT_CSV is not None:
        p = Path(EVENT_CSV).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"EVENT_CSV not found: {p}")
        return [p]

    files = sorted(IN_DIR.glob("events_*.csv"))

    if len(files) == 0:
        raise RuntimeError(
            f"events_*.csv が見つかりません。\n"
            f"searched: {IN_DIR}\n"
            f"先に plot_phase50_amp_ped.py を実行してください。"
        )

    return files


def detect_amp_column(df):
    for c in AMP_COLUMN_CANDIDATES:
        if c in df.columns:
            return c

    raise RuntimeError(
        f"amp列が見つかりません。\n"
        f"探した列: {AMP_COLUMN_CANDIDATES}\n"
        f"実際の列: {list(df.columns)}"
    )


def make_phase_bin10(df):
    """
    phase binをN_PHASE_BINS_NEWに作り直す。

    優先順：
    1. event_time があるなら event_timeの小数秒から作る
    2. 既存 phase_bin があるなら、それを粗くまとめる
    3. global_event_index があるなら mod N_PHASE_BINS_NEW
    4. 行番号で mod N_PHASE_BINS_NEW
    """

    if "event_time" in df.columns:
        t = pd.to_numeric(df["event_time"], errors="coerce").to_numpy(dtype=float)

        if np.isfinite(t).sum() == len(t):
            frac = t - np.floor(t)
            phase_bin_new = np.floor(frac * N_PHASE_BINS_NEW).astype(int)
            phase_bin_new = np.clip(phase_bin_new, 0, N_PHASE_BINS_NEW - 1)
            method = "event_time_fractional_second"
            return phase_bin_new, method

    if "phase_bin" in df.columns:
        old = pd.to_numeric(df["phase_bin"], errors="coerce").to_numpy(dtype=float)
        old_valid = old[np.isfinite(old)]

        if len(old_valid) > 0:
            old_min = int(np.nanmin(old_valid))
            old_max = int(np.nanmax(old_valid))

            # 例: 0..49 を 0..9 に粗くする
            old_nbins = old_max - old_min + 1

            phase_bin_new = np.floor(
                (old - old_min) / old_nbins * N_PHASE_BINS_NEW
            ).astype(int)

            phase_bin_new = np.clip(phase_bin_new, 0, N_PHASE_BINS_NEW - 1)
            method = f"rebinned_from_existing_phase_bin_{old_nbins}_to_{N_PHASE_BINS_NEW}"
            return phase_bin_new, method

    if "global_event_index" in df.columns:
        idx = pd.to_numeric(df["global_event_index"], errors="coerce").to_numpy(dtype=float)
        idx = np.nan_to_num(idx, nan=0).astype(int)

        phase_bin_new = idx % N_PHASE_BINS_NEW
        method = "global_event_index_mod_new_bins"
        return phase_bin_new, method

    idx = np.arange(len(df))
    phase_bin_new = idx % N_PHASE_BINS_NEW
    method = "row_index_mod_new_bins"
    return phase_bin_new, method


def make_common_bins(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise RuntimeError("histogram用の有効値がありません。")

    if USE_QUANTILE_RANGE:
        lo, hi = np.nanpercentile(values, [Q_LOW, Q_HIGH])
    else:
        lo, hi = np.nanmin(values), np.nanmax(values)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(values), np.nanmax(values)

    if lo == hi:
        eps = abs(lo) * 1e-6 + 1e-12
        lo -= eps
        hi += eps

    return np.linspace(lo, hi, HIST_BINS + 1)


def make_info_text(df, path):
    parts = []

    for key in ["freq_ghz", "z_mm", "x_mm", "amp_mode"]:
        if key in df.columns:
            vals = df[key].dropna().unique()
            if len(vals) == 1:
                parts.append(f"{key}={vals[0]}")

    if len(parts) == 0:
        parts.append(path.stem)

    return ", ".join(parts)


def plot_overlay_hist(df, value_col, out_png, title):
    if value_col not in df.columns:
        print(f"skip: {value_col} not in columns")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    values_all = df[value_col].to_numpy(dtype=float)
    bins = make_common_bins(values_all)

    fig, ax = plt.subplots(figsize=(10, 6))

    for b in range(N_PHASE_BINS_NEW):
        g = df[df["phase_bin_new"] == b]
        values = g[value_col].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.7,
            density=DENSITY,
            label=f"phase {b}  n={len(values)}",
        )

    ax.set_xlabel(value_col)
    ax.set_ylabel("events" if not DENSITY else "density")
    ax.set_title(title)
    ax.grid(True)

    ax.legend(
        fontsize=8,
        title=f"{N_PHASE_BINS_NEW} phase bins",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_png)


def plot_one_csv(path):
    df = pd.read_csv(path)

    if PED_COLUMN not in df.columns:
        raise RuntimeError(
            f"{PED_COLUMN} が見つかりません。\n"
            f"columns: {list(df.columns)}"
        )

    amp_col = detect_amp_column(df)

    phase_bin_new, method = make_phase_bin10(df)
    df["phase_bin_new"] = phase_bin_new

    print()
    print("processing:", path)
    print("phase method:", method)
    print("amp column:", amp_col)
    print("events:", len(df))

    info = make_info_text(df, path)

    out_base = safe_name(
        path.stem.replace("events_", f"phase{N_PHASE_BINS_NEW}_hist_")
    )

    out_ped_png = OUT_DIR / f"{out_base}_{PED_COLUMN}.png"
    out_amp_png = OUT_DIR / f"{out_base}_{amp_col}.png"
    out_csv = OUT_DIR / f"{out_base}_events_with_phase{N_PHASE_BINS_NEW}.csv"

    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    plot_overlay_hist(
        df,
        PED_COLUMN,
        out_ped_png,
        title=f"{PED_COLUMN} histogram by phase bin\n{info}",
    )

    plot_overlay_hist(
        df,
        amp_col,
        out_amp_png,
        title=f"{amp_col} histogram by phase bin\n{info}",
    )


# ============================================================
# main
# ============================================================

print("DATA_DATE:", DATA_DATE)
print("IN_DIR :", IN_DIR)
print("OUT_DIR:", OUT_DIR)
print("N_PHASE_BINS_NEW:", N_PHASE_BINS_NEW)

csv_files = find_event_csvs()
print("found csv:", len(csv_files))

for path in csv_files:
    plot_one_csv(path)

print()
print("done")
print("outputs saved in:", OUT_DIR)