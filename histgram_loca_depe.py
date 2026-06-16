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

# 現在クラウド側には rebin1 と rebin25 がある。
# rebin20 を使いたい場合は先に rebin20 のfit csvを作る必要あり。
REBIN = 20

# x依存を見るときに固定する z
XSCAN_FIXED_Z_MM = 7.5

# z依存を見るときに固定する x
ZSCAN_FIXED_X_MM = 3.4

# Noneなら全周波数を使う
# 特定の周波数だけ使いたい場合:
# TARGET_FREQ_GHZ_LIST = [5.451, 5.501]
TARGET_FREQ_GHZ_LIST = None

FREQ_ATOL_GHZ = 1e-3
POS_ATOL_MM = 1e-6

# 同じ freq, z, x に _second, _third などがある場合、
# Trueなら全部まとめて1つの位置データとして扱う
GROUP_REPEATS_BY_POSITION = True

# fit_status列があるCSVの場合だけ、fit成功イベントで絞る
USE_FIT_STATUS_CUT = True
GOOD_FIT_STATUS = 1

# ヒストグラムbin数
HIST_BINS = 50

# ============================================================
# ヒストグラム正規化
# ============================================================

# Trueなら、各位置のヒストグラムをイベント数で割る。
# n=1000でもn=4000でも、分布の形を比較できる。
NORMALIZE_HIST = True

# 外れ値で横軸が潰れるのを防ぐ
# 1%〜99%範囲でbinを作る
USE_QUANTILE_BINS = True
BIN_Q_LOW = 1
BIN_Q_HIGH = 99

DPI = 300

# 入力モード
# "cloud" : OneDrive / CloudStorage 側だけ探す
# "local" : KIDANALYSIS/data/20260527 だけ探す
# "both"  : 両方探す
INPUT_MODE = "local"


# ============================================================
# ローカル側とクラウド側のパス設定
# ============================================================

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

# 出力は必ずローカル側
OUT_DIR = HERE / "data" / DATA_DATE / f"locationdepen_hist_rebin{REBIN}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ローカルにfit結果csvを保存している場合の候補
local_data_dir = HERE / "data" / DATA_DATE

# OneDrive / CloudStorage 側の候補
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

# 必要ならVS Codeで Copy Path したパスをここに追加
EXTRA_INPUT_ROOTS = [
    # 例:
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
        print()
        print("ERROR: 入力フォルダが見つかりません。")
        print("OneDriveが同期されているか、またはパス表記が違う可能性があります。")
        print("VS Codeで対象フォルダを右クリックして Copy Path し、EXTRA_INPUT_ROOTS に追加してください。")
        sys.exit(1)

    return roots


input_roots = collect_input_roots()


# ============================================================
# 測定情報の読み取り
# 例:
# 5.451GHz_z=7.5mm_x=3.4mm
# 5.451GHz_z=7.5mm_x=3.4mm_second
# 5.451GHz_z=7.5mm_x=3.4mm__wf_..._fitres_rebin25.csv
# ============================================================

MEAS_PATTERN = re.compile(
    r"(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))?"
)


def parse_measurement_info(path: Path):
    """
    csvの親フォルダ名またはcsvファイル名から
    freq, z, x, tag を取り出す。
    """
    candidates = [
        path.parent.name,
        path.name,
        path.as_posix(),
    ]

    for s in candidates:
        m = MEAS_PATTERN.search(s)

        if m is not None:
            d = m.groupdict()

            return {
                "freq_ghz": float(d["freq"]),
                "z_mm": float(d["z"]),
                "x_mm": float(d["x"]),
                "tag": d["tag"] or "",
            }

    return None


def safe_name(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
    )


def close(a, b, atol=POS_ATOL_MM):
    return np.isclose(a, b, atol=atol, rtol=0)


def unique_sorted(values):
    return sorted(set(round(float(v), 6) for v in values))


# ============================================================
# CSV探索
# ============================================================

def find_fitres_csvs():
    """
    rebin指定のfitres csvを探す。
    例:
      *_fitres_rebin1.csv
      *_fitres_rebin25.csv
    """
    pattern = f"*fitres_rebin{REBIN}.csv"

    files = []

    print()
    print("===== csv search =====")
    print("pattern:", pattern)

    for root in input_roots:
        print("search root:", root)
        found = sorted(root.rglob(pattern))
        print("  found:", len(found))
        files.extend(found)

    files = sorted(set(files))

    print("total unique csv:", len(files))

    if len(files) == 0:
        print()
        print("WARNING: csvが見つかりませんでした。")
        print("確認ポイント:")
        print(f"  1. REBIN = {REBIN} で合っているか")
        print(f"  2. ファイル名が '*fitres_rebin{REBIN}.csv' になっているか")
        print("  3. OneDriveがローカルに同期されているか")
        print("  4. INPUT_MODE が cloud/local/both のどれになっているか")
        print()
        print("参考: 近いfitresファイルも探します。")

        for root in input_roots:
            nearby = sorted(root.rglob("*fitres_rebin*.csv"))
            print("nearby root:", root)
            print("  nearby count:", len(nearby))

            for f in nearby[:40]:
                print("   ", f)

            if len(nearby) > 40:
                print("   ...")

    rows = []

    for f in files:
        info = parse_measurement_info(f)

        if info is None:
            print("skip parse failed:", f)
            continue

        rows.append({
            "csv_path": f,
            "freq_ghz": info["freq_ghz"],
            "z_mm": info["z_mm"],
            "x_mm": info["x_mm"],
            "tag": info["tag"],
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df = df.sort_values(["freq_ghz", "z_mm", "x_mm", "tag", "csv_path"])

    return df


def select_freqs(df):
    freqs = unique_sorted(df["freq_ghz"].values)

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


# ============================================================
# CSV読み込み・列判定
# ============================================================

def get_fit_columns(df):
    """
    CSVの形式に応じて、ヒストグラムに使うfit列を決める。

    対応形式:
    1. IQ射影fit版:
       iq_t0, iq_k, iq_tau, iq_rise, iq_ped

    2. ch0/ch1個別fit版:
       ch0_t0, ch0_k, ch0_tau, ch0_rise, ch0_ped,
       ch1_t0, ch1_k, ch1_tau, ch1_rise, ch1_ped,
       absk

    3. 旧形式:
       t0, k, tau, rise, ped
    """

    iq_cols = [
        "iq_t0", "iq_k", "iq_tau", "iq_rise", "iq_ped",
    ]

    ch0_cols = [
        "ch0_t0", "ch0_k", "ch0_tau", "ch0_rise", "ch0_ped",
    ]

    ch1_cols = [
        "ch1_t0", "ch1_k", "ch1_tau", "ch1_rise", "ch1_ped",
    ]

    old_cols = [
        "t0", "k", "tau", "rise", "ped",
    ]

    if all(c in df.columns for c in iq_cols):
        cols = iq_cols.copy()
        if "absk" in df.columns:
            cols.append("absk")
        return cols

    if all(c in df.columns for c in ch0_cols + ch1_cols):
        cols = ch0_cols + ch1_cols
        if "absk" in df.columns:
            cols.append("absk")
        return cols

    if all(c in df.columns for c in ch0_cols):
        cols = ch0_cols.copy()
        if "absk" in df.columns:
            cols.append("absk")
        return cols

    if all(c in df.columns for c in ch1_cols):
        cols = ch1_cols.copy()
        if "absk" in df.columns:
            cols.append("absk")
        return cols

    if all(c in df.columns for c in old_cols):
        cols = old_cols.copy()
        if "absk" in df.columns:
            cols.append("absk")
        return cols

    candidates = [
        "iq_t0", "iq_k", "iq_tau", "iq_rise", "iq_ped",
        "ch0_t0", "ch0_k", "ch0_tau", "ch0_rise", "ch0_ped",
        "ch1_t0", "ch1_k", "ch1_tau", "ch1_rise", "ch1_ped",
        "t0", "k", "tau", "rise", "ped",
        "absk",
    ]

    cols = [c for c in candidates if c in df.columns]

    if len(cols) == 0:
        raise ValueError("fit parameter columns not found")

    return cols


def column_label(col):
    labels = {
        "iq_t0": r"IQ $t_0$ [$\mu$s]",
        "iq_k": "IQ k [mV]",
        "iq_tau": r"IQ $\tau$ [$\mu$s]",
        "iq_rise": r"IQ rise [$\mu$s]",
        "iq_ped": "IQ ped [mV]",

        "ch0_t0": r"ch0 $t_0$",
        "ch0_k": "ch0 k",
        "ch0_tau": r"ch0 $\tau$",
        "ch0_rise": "ch0 rise",
        "ch0_ped": "ch0 ped",

        "ch1_t0": r"ch1 $t_0$",
        "ch1_k": "ch1 k",
        "ch1_tau": r"ch1 $\tau$",
        "ch1_rise": "ch1 rise",
        "ch1_ped": "ch1 ped",

        "t0": r"$t_0$",
        "k": "k",
        "tau": r"$\tau$",
        "rise": "rise",
        "ped": "ped",

        "absk": "absk",
    }

    return labels.get(col, col)


def read_and_filter_csv(path: Path):
    df = pd.read_csv(path)

    if USE_FIT_STATUS_CUT and "fit_status" in df.columns:
        df = df[df["fit_status"] == GOOD_FIT_STATUS].copy()

    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() > 0:
            df[c] = converted

    return df


def load_group_csvs(group_rows):
    """
    同じ位置の複数csvをまとめて読む。
    """
    dfs = []

    for _, row in group_rows.iterrows():
        path = Path(row["csv_path"])

        try:
            df = read_and_filter_csv(path)
        except Exception as e:
            print("skip csv read error:", path, e)
            continue

        df["source_csv"] = path.as_posix()
        df["freq_ghz"] = row["freq_ghz"]
        df["z_mm"] = row["z_mm"]
        df["x_mm"] = row["x_mm"]
        df["tag"] = row["tag"]

        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# ヒストグラム補助
# ============================================================

def common_bins(data_list, col, bins=50):
    vals = []

    for df in data_list:
        if col not in df.columns:
            continue

        v = pd.to_numeric(df[col], errors="coerce").dropna().values

        if len(v) > 0:
            vals.append(v)

    if len(vals) == 0:
        return bins

    vals = np.concatenate(vals)

    if len(vals) == 0:
        return bins

    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return bins

    if USE_QUANTILE_BINS:
        vmin, vmax = np.nanpercentile(vals, [BIN_Q_LOW, BIN_Q_HIGH])
    else:
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return bins

    return np.linspace(vmin, vmax, bins + 1)


def choose_one_csv_prefer_untagged(g):
    """
    GROUP_REPEATS_BY_POSITION=False の場合に使う。
    tagなしを優先して1csvだけ選ぶ。
    """
    gg = g.copy()
    gg["tag_priority"] = gg["tag"].apply(lambda x: 0 if str(x) == "" else 1)
    gg = gg.sort_values(["tag_priority", "tag", "csv_path"])
    return gg.head(1).drop(columns=["tag_priority"])


# ============================================================
# scan group 作成
# ============================================================

def make_scan_groups(csv_df, freq, mode):
    """
    mode = xscan:
        freq固定, z=7.5固定, xごと

    mode = zscan:
        freq固定, x=3.4固定, zごと
    """
    df = csv_df[
        np.isclose(csv_df["freq_ghz"], freq, atol=FREQ_ATOL_GHZ, rtol=0)
    ].copy()

    if mode == "xscan":
        df = df[df["z_mm"].apply(lambda v: close(v, XSCAN_FIXED_Z_MM))]
        pos_col = "x_mm"

    elif mode == "zscan":
        df = df[df["x_mm"].apply(lambda v: close(v, ZSCAN_FIXED_X_MM))]
        pos_col = "z_mm"

    else:
        raise ValueError("mode must be 'xscan' or 'zscan'")

    positions = unique_sorted(df[pos_col].values)

    groups = []

    print()
    print(f"make_scan_groups: freq={freq:.3f}, mode={mode}")
    print("positions:", positions)

    for pos in positions:
        g = df[df[pos_col].apply(lambda v: close(v, pos))].copy()

        if len(g) == 0:
            continue

        if not GROUP_REPEATS_BY_POSITION:
            g = choose_one_csv_prefer_untagged(g)

        data = load_group_csvs(g)

        if len(data) == 0:
            print("skip empty data:", pos)
            continue

        label = f"{pos:.1f} mm"

        print(f"  {label}: csv={len(g)}, events={len(data)}")

        groups.append({
            "position": pos,
            "label": label,
            "csv_rows": g,
            "data": data,
        })

    return groups


def summarize_groups(groups, freq, mode):
    rows = []

    for item in groups:
        df = item["data"]

        try:
            cols = get_fit_columns(df)
        except Exception:
            cols = []

        row = {
            "mode": mode,
            "freq_ghz": freq,
            "omega_rad_s": 2.0 * np.pi * freq * 1e9,
            "position_mm": item["position"],
            "label": item["label"],
            "n_events": len(df),
            "n_csv": len(item["csv_rows"]),
            "csv_paths": ";".join([Path(p).as_posix() for p in item["csv_rows"]["csv_path"]]),
        }

        if mode == "xscan":
            row["x_mm"] = item["position"]
            row["z_mm"] = XSCAN_FIXED_Z_MM
        else:
            row["z_mm"] = item["position"]
            row["x_mm"] = ZSCAN_FIXED_X_MM

        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            row[f"{c}_mean"] = v.mean()
            row[f"{c}_median"] = v.median()
            row[f"{c}_std"] = v.std(ddof=1)
            row[f"{c}_n"] = v.notna().sum()

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# plot: xscan/zscanをそれぞれ1枚のPNGにまとめる
# ============================================================
def plot_hist_overlay(groups, freq, mode, out_png):
    """
    xscan / zscan を 1枚のPNGにまとめて保存する。
    各fit parameterをsubplotで並べ、位置ごとのhistを重ねる。
    legendは図の右外に置く。

    NORMALIZE_HIST=True のとき、
    各位置のヒストグラムをイベント数で割って
    fraction / bin として描画する。
    """
    if len(groups) == 0:
        print("no groups for", freq, mode)
        return

    fit_cols = None

    for item in groups:
        try:
            fit_cols = get_fit_columns(item["data"])
            break
        except Exception:
            continue

    if fit_cols is None or len(fit_cols) == 0:
        print("no fit columns:", freq, mode)
        return

    preferred_order = [
        "iq_t0", "iq_k", "iq_tau", "iq_rise", "iq_ped",

        "ch0_t0", "ch0_k", "ch0_tau", "ch0_rise", "ch0_ped",
        "ch1_t0", "ch1_k", "ch1_tau", "ch1_rise", "ch1_ped",

        "t0", "k", "tau", "rise", "ped",

        "absk",
    ]

    ordered = [c for c in preferred_order if c in fit_cols]
    extra = [c for c in fit_cols if c not in ordered]
    fit_cols = ordered + extra

    # 11個列がある場合、4列配置が見やすい
    if len(fit_cols) <= 6:
        ncols = 3
    else:
        ncols = 4

    nrows = int(np.ceil(len(fit_cols) / ncols))

    data_list = [item["data"] for item in groups]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    legend_handles = []
    legend_labels = []

    for ax, col in zip(axes_flat, fit_cols):
        bins = common_bins(data_list, col, bins=HIST_BINS)

        for i, item in enumerate(groups):
            df = item["data"]
            color = colors[i % len(colors)]

            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors="coerce").dropna().values
            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            if NORMALIZE_HIST:
                # 各位置ごとに総和が1になるように正規化
                weights = np.ones_like(values, dtype=float) / len(values)
                y_label = "fraction / bin"
            else:
                weights = None
                y_label = "counts"

            h = ax.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                lw=1.7,
                color=color,
                label=f"{item['label']} n={len(values)}",
            )

            # 共通legendは最初のsubplotからだけ回収
            if col == fit_cols[0]:
                legend_handles.append(h[2][0])
                legend_labels.append(f"{item['label']} n={len(values)}")

        ax.set_xlabel(column_label(col))
        ax.set_ylabel(y_label)
        ax.set_title(col)
        ax.grid(True)

    for ax in axes_flat[len(fit_cols):]:
        ax.axis("off")

    if mode == "xscan":
        suptitle = (
            f"x scan histogram overlay\n"
            f"f = {freq:.3f} GHz,  z = {XSCAN_FIXED_Z_MM:.1f} mm,  rebin = {REBIN}"
        )
    else:
        suptitle = (
            f"z scan histogram overlay\n"
            f"f = {freq:.3f} GHz,  x = {ZSCAN_FIXED_X_MM:.1f} mm,  rebin = {REBIN}"
        )

    fig.suptitle(suptitle, fontsize=14, y=0.985)

    if len(legend_handles) > 0:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            fontsize=8,
            frameon=True,
            ncol=1,
        )

    # 右側にlegend用スペースを空ける
    fig.tight_layout(rect=[0, 0, 0.82, 0.93])

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print("saved:", out_png)

    plt.close(fig)

# ============================================================
# main
# ============================================================

print()
print("DATA_DATE:", DATA_DATE)
print("REBIN:", REBIN)
print("OUT_DIR:", OUT_DIR)

csv_df = find_fitres_csvs()

print()
print("found parsed csv:", len(csv_df))

if len(csv_df) == 0:
    raise RuntimeError(f"*_fitres_rebin{REBIN}.csv が見つかりませんでした。")

csv_list_path = OUT_DIR / f"found_fitres_rebin{REBIN}.csv"
csv_df.to_csv(csv_list_path, index=False)
print("saved:", csv_list_path)

freqs = select_freqs(csv_df)

print()
print("frequencies:")
for f in freqs:
    print(f"  f={f:.3f} GHz")

for freq in freqs:
    print()
    print("============================================================")
    print(f"frequency = {freq:.3f} GHz")
    print("============================================================")

    # --------------------------------------------------------
    # x scan hist
    # --------------------------------------------------------
    x_groups = make_scan_groups(csv_df, freq, mode="xscan")

    x_summary = summarize_groups(x_groups, freq, mode="xscan")
    x_summary_path = OUT_DIR / (
        f"hist_xscan_summary_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"z{safe_name(f'{XSCAN_FIXED_Z_MM:.1f}mm')}_"
        f"rebin{REBIN}.csv"
    )
    x_summary.to_csv(x_summary_path, index=False)
    print("saved:", x_summary_path)

    x_overlay_png = OUT_DIR / (
        f"hist_xscan_overlay_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"z{safe_name(f'{XSCAN_FIXED_Z_MM:.1f}mm')}_"
        f"rebin{REBIN}.png"
    )

    plot_hist_overlay(
        x_groups,
        freq=freq,
        mode="xscan",
        out_png=x_overlay_png,
    )

    # --------------------------------------------------------
    # z scan hist
    # --------------------------------------------------------
    z_groups = make_scan_groups(csv_df, freq, mode="zscan")

    z_summary = summarize_groups(z_groups, freq, mode="zscan")
    z_summary_path = OUT_DIR / (
        f"hist_zscan_summary_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"x{safe_name(f'{ZSCAN_FIXED_X_MM:.1f}mm')}_"
        f"rebin{REBIN}.csv"
    )
    z_summary.to_csv(z_summary_path, index=False)
    print("saved:", z_summary_path)

    z_overlay_png = OUT_DIR / (
        f"hist_zscan_overlay_"
        f"{safe_name(f'{freq:.3f}GHz')}_"
        f"x{safe_name(f'{ZSCAN_FIXED_X_MM:.1f}mm')}_"
        f"rebin{REBIN}.png"
    )

    plot_hist_overlay(
        z_groups,
        freq=freq,
        mode="zscan",
        out_png=z_overlay_png,
    )

print()
print("done")
print("outputs saved in:", OUT_DIR)