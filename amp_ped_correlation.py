from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# ここだけ基本的に変える
# ============================================================

DATA_DATE = "20260527"

# Noneなら phase50_amp_ped 内の events_*.csv を全部処理
# 1つだけ指定したい場合はパスを書く
EVENT_CSV = None
# EVENT_CSV = "/Users/kubokosei/software/kidanalysis/data/20260527/phase50_amp_ped/events_20260527_5p451GHz_z7p5mm_x3p4mm_phase50_projected.csv"

# 散布図の点が多すぎる場合、表示だけ間引く
# 解析の相関係数は全点で計算する
MAX_SCATTER_POINTS = 20000

# phase binごとの平均点も重ねる
PLOT_PHASE_MEAN = True

# pedを横軸、ampを縦軸にする
X_COLUMNS = ["ped0", "ped1", "ped_abs"]

# amp列
Y_COLUMN = "amp"

DPI = 300


# ============================================================
# パス設定
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

IN_DIR = HERE / "data" / DATA_DATE / "phase50_amp_ped"
OUT_DIR = HERE / "data" / DATA_DATE / "amp_ped_correlation"
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
            f"events_*.csv が見つかりませんでした。\n"
            f"先に plot_phase50_amp_ped.py を実行してください。\n"
            f"searched: {IN_DIR}"
        )

    return files


def pearson_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]


def spearman_corr(x, y):
    s = pd.Series(x)
    t = pd.Series(y)
    df = pd.DataFrame({"x": s, "y": t}).dropna()

    if len(df) < 3:
        return np.nan

    return df["x"].rank().corr(df["y"].rank())


def linear_fit(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan

    a, b = np.polyfit(x[mask], y[mask], 1)
    return a, b


def thin_for_scatter(df):
    if len(df) <= MAX_SCATTER_POINTS:
        return df

    return df.sample(MAX_SCATTER_POINTS, random_state=0)


def make_info_text(df, path):
    parts = []

    for key in ["freq_ghz", "z_mm", "x_mm", "amp_mode"]:
        if key in df.columns:
            vals = df[key].dropna().unique()
            if len(vals) == 1:
                parts.append(f"{key}={vals[0]}")

    parts.append(path.stem)

    return ", ".join(parts)


def plot_one_events_csv(path):
    df = pd.read_csv(path)

    required = [Y_COLUMN, "phase_bin"] + X_COLUMNS
    missing = [c for c in required if c not in df.columns]

    if missing:
        print("skip missing columns:", path)
        print("missing:", missing)
        return None

    # 数値化
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_clean = df.dropna(subset=[Y_COLUMN] + X_COLUMNS).copy()

    if len(df_clean) == 0:
        print("skip no valid rows:", path)
        return None

    df_plot = thin_for_scatter(df_clean)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    summary_rows = []

    # ------------------------------------------------------------
    # amp vs ped0, ped1, ped_abs
    # ------------------------------------------------------------
    for i, xcol in enumerate(X_COLUMNS):
        ax = axes[i]

        x_all = df_clean[xcol].to_numpy(dtype=float)
        y_all = df_clean[Y_COLUMN].to_numpy(dtype=float)

        r_p = pearson_corr(x_all, y_all)
        r_s = spearman_corr(x_all, y_all)
        slope, intercept = linear_fit(x_all, y_all)

        summary_rows.append({
            "source_csv": path.as_posix(),
            "x": xcol,
            "y": Y_COLUMN,
            "n": len(df_clean),
            "pearson_r": r_p,
            "spearman_r": r_s,
            "linear_slope": slope,
            "linear_intercept": intercept,
        })

        sc = ax.scatter(
            df_plot[xcol],
            df_plot[Y_COLUMN],
            c=df_plot["phase_bin"],
            s=8,
            alpha=0.35,
        )

        if np.isfinite(slope) and np.isfinite(intercept):
            xmin = np.nanpercentile(x_all, 1)
            xmax = np.nanpercentile(x_all, 99)
            xx = np.linspace(xmin, xmax, 200)
            yy = slope * xx + intercept

            ax.plot(
                xx,
                yy,
                color="black",
                lw=2,
                label=f"linear fit"
            )

        if PLOT_PHASE_MEAN:
            phase_mean = (
                df_clean
                .groupby("phase_bin")[[xcol, Y_COLUMN]]
                .mean()
                .reset_index()
            )

            ax.plot(
                phase_mean[xcol],
                phase_mean[Y_COLUMN],
                marker="o",
                ms=5,
                lw=1.5,
                color="red",
                label="phase mean"
            )

        ax.set_xlabel(f"{xcol} [V]")
        ax.set_ylabel(Y_COLUMN)
        ax.set_title(
            f"{Y_COLUMN} vs {xcol}\n"
            f"Pearson r={r_p:.3f}, Spearman r={r_s:.3f}"
        )
        ax.grid(True)
        ax.legend(fontsize=8)

    # ------------------------------------------------------------
    # amp vs phase bin
    # ------------------------------------------------------------
    ax = axes[3]

    phase_summary = (
        df_clean
        .groupby("phase_bin")
        .agg(
            n=(Y_COLUMN, "count"),
            amp_mean=(Y_COLUMN, "mean"),
            amp_median=(Y_COLUMN, "median"),
            amp_std=(Y_COLUMN, "std"),
        )
        .reset_index()
    )

    phase_summary["amp_sem"] = phase_summary["amp_std"] / np.sqrt(phase_summary["n"])

    ax.errorbar(
        phase_summary["phase_bin"],
        phase_summary["amp_mean"],
        yerr=phase_summary["amp_sem"],
        marker="o",
        lw=1.5,
        capsize=2,
        label="mean ± SEM",
    )

    ax.plot(
        phase_summary["phase_bin"],
        phase_summary["amp_median"],
        marker="s",
        lw=1.5,
        label="median",
    )

    ax.set_xlabel("phase bin")
    ax.set_ylabel(Y_COLUMN)
    ax.set_title(f"{Y_COLUMN} vs phase bin")
    ax.grid(True)
    ax.legend(fontsize=8)

    info_text = make_info_text(df_clean, path)

    fig.suptitle(
        f"Event-by-event amp-ped correlation\n{info_text}",
        fontsize=14,
    )

    # 右側にcolorbar用の余白を確保
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])

    # colorbar専用の軸を右外に作る
    cax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("phase bin")

    out_base = safe_name(path.stem.replace("events_", "amp_ped_corr_"))

    out_png = OUT_DIR / f"{out_base}.png"
    out_summary = OUT_DIR / f"{out_base}_summary.csv"
    out_phase = OUT_DIR / f"{out_base}_phase_summary.csv"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    phase_summary.to_csv(out_phase, index=False)

    print("saved:", out_png)
    print("saved:", out_summary)
    print("saved:", out_phase)

    return out_png


# ============================================================
# main
# ============================================================

print("DATA_DATE:", DATA_DATE)
print("IN_DIR :", IN_DIR)
print("OUT_DIR:", OUT_DIR)

csv_files = find_event_csvs()

print("found events csv:", len(csv_files))

for path in csv_files:
    print()
    print("processing:", path)
    plot_one_events_csv(path)

print()
print("done")
print("outputs saved in:", OUT_DIR)