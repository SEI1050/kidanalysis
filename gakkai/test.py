import os
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Settings
# ============================================================

filename = sys.argv[1]

BIN_SIZE = 1

# pedestal は t < PED_END_NS の領域から計算
PED_END_NS = -50.0

# peak を探す時間領域
PEAK_SEARCH_MIN_NS = -300.0
PEAK_SEARCH_MAX_NS = 800.0

# t0 の定義
ONSET_NSIGMA = 3.0
ONSET_FRAC = 0.05
ONSET_CONSEC = 10

# pulse end の定義
END_NSIGMA = 3.0
END_FRAC = 0.02
END_CONSEC = 20


FEATURES = [
    "H",
    "rise1090",
    "fall9010",
    "AL",
    "AR",
    "AoverH",
]


# ============================================================
# Helper functions
# ============================================================

def linear_crossing(t1, y1, t2, y2, level):
    """
    y=level の交点を2点間で線形補間する
    """
    if y2 == y1:
        return 0.5 * (t1 + t2)

    return t1 + (level - y1) * (t2 - t1) / (y2 - y1)


def find_up_crossing(t, y, level, i_start, i_stop):
    """
    i_start -> i_stop の方向に見て、
    level を下から上へ横切る最初の時刻
    """
    i_start = max(0, i_start)
    i_stop = min(len(y) - 1, i_stop)

    for i in range(i_start, i_stop):
        if y[i] < level <= y[i + 1]:
            return linear_crossing(
                t[i], y[i],
                t[i + 1], y[i + 1],
                level
            )

    return np.nan


def find_down_crossing(t, y, level, i_start, i_stop):
    """
    i_start -> i_stop の方向に見て、
    level を上から下へ横切る最初の時刻
    """
    i_start = max(0, i_start)
    i_stop = min(len(y) - 1, i_stop)

    for i in range(i_start, i_stop):
        if y[i] >= level > y[i + 1]:
            return linear_crossing(
                t[i], y[i],
                t[i + 1], y[i + 1],
                level
            )

    return np.nan


def first_consecutive_true(mask, nconsec):
    """
    True が nconsec 点連続する最初の index を返す
    """
    if len(mask) < nconsec:
        return None

    kernel = np.ones(nconsec, dtype=int)
    count = np.convolve(mask.astype(int), kernel, mode="valid")

    found = np.where(count >= nconsec)[0]

    if len(found) == 0:
        return None

    return int(found[0])


# ============================================================
# Main waveform analysis
# ============================================================

def analyze_waveform(t_ns, waveform):
    """
    1 waveform から

      H
      rise1090
      fall9010
      AL
      AR
      AoverH

    を求める。

    単位:
      H          : mV
      rise/fall  : ns
      AL, AR     : mV ns
      AoverH     : ns
    """

    # --------------------------------------------------------
    # 1. pedestal
    # --------------------------------------------------------

    ped_mask = t_ns < PED_END_NS

    # pre-trigger が十分取れない場合のfallback
    if np.count_nonzero(ped_mask) < 20:
        n_ped = max(20, int(0.05 * len(waveform)))
        ped_values = waveform[:n_ped]
    else:
        ped_values = waveform[ped_mask]

    ped = np.mean(ped_values)
    sigma_ped = np.std(ped_values)

    # baseline subtraction
    dev = waveform - ped


    # --------------------------------------------------------
    # 2. peak search window
    # --------------------------------------------------------

    peak_mask = (
        (t_ns >= PEAK_SEARCH_MIN_NS)
        & (t_ns <= PEAK_SEARCH_MAX_NS)
    )

    peak_indices = np.where(peak_mask)[0]

    if len(peak_indices) == 0:
        peak_indices = np.arange(len(waveform))

    # 正方向と負方向、どちらの excursion が大きいか
    local_dev = dev[peak_indices]

    max_positive = np.max(local_dev)
    max_negative = np.min(local_dev)

    if abs(max_positive) >= abs(max_negative):
        pol = +1.0
    else:
        pol = -1.0

    # pulse が正になるように向きをそろえる
    s = pol * dev

    # mV に変換
    s_mV = s * 1e3

    # peak
    ipeak_local = np.argmax(s[peak_indices])
    ipeak = peak_indices[ipeak_local]

    H = s_mV[ipeak]
    tpeak = t_ns[ipeak]


    if not np.isfinite(H) or H <= 0:
        result = {key: np.nan for key in FEATURES}

        diagnostic = {
            "ped": ped,
            "sigma_ped": sigma_ped,
            "pol": pol,
            "s_mV": s_mV,
            "ipeak": ipeak,
            "tpeak": tpeak,
            "t0": np.nan,
            "tend": np.nan,
            "t10L": np.nan,
            "t90L": np.nan,
            "t90R": np.nan,
            "t10R": np.nan,
        }

        return result, diagnostic


    # --------------------------------------------------------
    # 3. t0
    # --------------------------------------------------------

    H_V = H * 1e-3

    onset_threshold = max(
        ONSET_NSIGMA * sigma_ped,
        ONSET_FRAC * H_V
    )

    # peak search window の開始から peak まで
    istart = peak_indices[0]

    onset_mask = s[istart:ipeak + 1] > onset_threshold

    i0_relative = first_consecutive_true(
        onset_mask,
        ONSET_CONSEC
    )

    if i0_relative is None:
        i0 = istart
    else:
        i0 = istart + i0_relative

    t0 = t_ns[i0]


    # --------------------------------------------------------
    # 4. pulse end
    # --------------------------------------------------------

    end_threshold = max(
        END_NSIGMA * sigma_ped,
        END_FRAC * H_V
    )

    # peak 後、baseline に戻ったところ
    end_mask = np.abs(s[ipeak:]) < end_threshold

    iend_relative = first_consecutive_true(
        end_mask,
        END_CONSEC
    )

    if iend_relative is None:
        iend = len(s) - 1
    else:
        iend = ipeak + iend_relative

    # 最低でも peak の次の点までは確保
    iend = max(iend, ipeak + 1)
    iend = min(iend, len(s) - 1)

    tend = t_ns[iend]


    # --------------------------------------------------------
    # 5. 10%-90% rise
    # --------------------------------------------------------

    level10 = 0.10 * H
    level90 = 0.90 * H

    t10L = find_up_crossing(
        t_ns,
        s_mV,
        level10,
        istart,
        ipeak
    )

    t90L = find_up_crossing(
        t_ns,
        s_mV,
        level90,
        istart,
        ipeak
    )

    if np.isfinite(t10L) and np.isfinite(t90L):
        rise1090 = t90L - t10L
    else:
        rise1090 = np.nan


    # --------------------------------------------------------
    # 6. 90%-10% fall
    # --------------------------------------------------------

    t90R = find_down_crossing(
        t_ns,
        s_mV,
        level90,
        ipeak,
        len(s_mV) - 1
    )

    t10R = find_down_crossing(
        t_ns,
        s_mV,
        level10,
        ipeak,
        len(s_mV) - 1
    )

    if np.isfinite(t90R) and np.isfinite(t10R):
        fall9010 = t10R - t90R
    else:
        fall9010 = np.nan


    # --------------------------------------------------------
    # 7. left / right integral
    # --------------------------------------------------------

    # t は ns、signal は mV
    # -> integral の単位は mV ns

    if ipeak > i0:
        AL = np.trapezoid(
            s_mV[i0:ipeak + 1],
            t_ns[i0:ipeak + 1]
        )
    else:
        AL = 0.0

    if iend > ipeak:
        AR = np.trapezoid(
            s_mV[ipeak:iend + 1],
            t_ns[ipeak:iend + 1]
        )
    else:
        AR = 0.0

    A = AL + AR

    if H > 0:
        AoverH = A / H
    else:
        AoverH = np.nan


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "H": H,
        "rise1090": rise1090,
        "fall9010": fall9010,
        "AL": AL,
        "AR": AR,
        "AoverH": AoverH,
    }

    # plot/debug用
    diagnostic = {
        "ped": ped,
        "sigma_ped": sigma_ped,
        "pol": pol,
        "s_mV": s_mV,

        "ipeak": ipeak,
        "tpeak": tpeak,

        "i0": i0,
        "t0": t0,

        "iend": iend,
        "tend": tend,

        "t10L": t10L,
        "t90L": t90L,
        "t90R": t90R,
        "t10R": t10R,
    }

    return result, diagnostic


# ============================================================
# Load data
# ============================================================

data = np.load(filename, allow_pickle=True)

print("number of points =", data["npts"])
print(
    "number of waveforms =",
    data["ch0"].shape[0],
    data["ch1"].shape[0]
)

sample_rate = data["sample_rate"]
npts_raw = data["npts"]
ref_position = data["ref_position"]

nwf = data["ch1"].shape[0]

nbin = npts_raw // BIN_SIZE
usable_npts = nbin * BIN_SIZE

ch0 = (
    data["ch0"][:, :usable_npts]
    .reshape(nwf, nbin, BIN_SIZE)
    .mean(axis=2)
)

ch1 = (
    data["ch1"][:, :usable_npts]
    .reshape(nwf, nbin, BIN_SIZE)
    .mean(axis=2)
)

npts = nbin

tbin_raw = (
    np.arange(usable_npts)
    - npts_raw * ref_position / 100
) / sample_rate

tbin = (
    tbin_raw
    .reshape(nbin, BIN_SIZE)
    .mean(axis=1)
)

# ns
t_ns = tbin * 1e9

print("sample_rate =", sample_rate)
print("npts =", npts)
print("ref_position =", ref_position)


# ============================================================
# Analyze all events
# ============================================================

all_results = []
all_diagnostics = []

for idx in range(nwf):

    result0, diag0 = analyze_waveform(
        t_ns,
        ch0[idx]
    )

    result1, diag1 = analyze_waveform(
        t_ns,
        ch1[idx]
    )

    row = {}

    for key in FEATURES:
        row[f"ch0_{key}"] = result0[key]
        row[f"ch1_{key}"] = result1[key]

    all_results.append(row)
    all_diagnostics.append([diag0, diag1])


df_shape = pd.DataFrame(all_results)


# optional:
# ch0/ch1 が I/Q なら peak amplitude の二乗和も見られる
df_shape["H_IQ"] = np.hypot(
    df_shape["ch0_H"],
    df_shape["ch1_H"]
)

print(df_shape)


# ============================================================
# Save CSV
# ============================================================

basename = os.path.splitext(
    os.path.basename(filename)
)[0]

csvname = basename + "_shape_parameters.csv"

df_shape.to_csv(
    csvname,
    index=False
)

print("saved:", csvname)


# ============================================================
# Diagnostic PDF
# ============================================================

pdfname = basename + "_shape_diagnostic.pdf"

with PdfPages(pdfname) as pdf:

    # --------------------------------------------------------
    # waveform plots
    # --------------------------------------------------------
        # ========================================================
    # 2D scatter analysis
    #
    #   1. T_rise vs A_L/H
    #   2. A_R/H vs A_L/H
    #
    # df_shape の index を event number として使用する
    # ========================================================

    def robust_outlier_mask(x, y, threshold=4.0):
        """
        x, y の中央値と MAD を使って2次元外れ値を検出する。

        robust distance:
            sqrt(zx^2 + zy^2)

        が threshold を超えたイベントを外れ値とする。
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_med = np.nanmedian(x)
        y_med = np.nanmedian(y)

        x_mad = np.nanmedian(np.abs(x - x_med))
        y_mad = np.nanmedian(np.abs(y - y_med))

        # MAD = 0 の場合にゼロ除算を避ける
        if not np.isfinite(x_mad) or x_mad == 0:
            x_mad = np.nanstd(x)

        if not np.isfinite(y_mad) or y_mad == 0:
            y_mad = np.nanstd(y)

        if not np.isfinite(x_mad) or x_mad == 0:
            x_mad = 1.0

        if not np.isfinite(y_mad) or y_mad == 0:
            y_mad = 1.0

        # MADを標準偏差相当の尺度へ変換
        zx = 0.6745 * (x - x_med) / x_mad
        zy = 0.6745 * (y - y_med) / y_mad

        robust_distance = np.sqrt(zx**2 + zy**2)

        return robust_distance > threshold, robust_distance


    def calculate_correlations(x, y):
        """
        Pearson相関係数とSpearman順位相関係数を計算する。
        計算できない場合はnanを返す。
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3:
            return np.nan, np.nan, np.nan, np.nan

        # 全イベントで同じ値の場合、相関係数は定義できない
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan, np.nan, np.nan, np.nan

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        return pearson_r, pearson_p, spearman_r, spearman_p


    def draw_scatter_with_outliers(
        ax,
        x,
        y,
        event_numbers,
        xlabel,
        ylabel,
        title,
        color,
        annotate_outliers=True,
    ):
        """
        scatter plot、相関係数、外れ値イベント番号を描画する。
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        event_numbers = np.asarray(event_numbers)

        # nan, infを除外
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x_valid = x[valid]
        y_valid = y[valid]
        event_valid = event_numbers[valid]

        if len(x_valid) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid events",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid()

            return []

        # robustな2次元外れ値検出
        outlier_mask, robust_distance = robust_outlier_mask(
            x_valid,
            y_valid,
            threshold=4.0,
        )

        normal_mask = ~outlier_mask

        # 通常イベント
        ax.scatter(
            x_valid[normal_mask],
            y_valid[normal_mask],
            s=22,
            alpha=0.55,
            color=color,
            edgecolors="none",
            label="normal",
        )

        # 外れ値
        if np.any(outlier_mask):
            ax.scatter(
                x_valid[outlier_mask],
                y_valid[outlier_mask],
                s=55,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=1.4,
                label="outlier",
                zorder=3,
            )

        # 外れ値のイベント番号を表示
        if annotate_outliers:
            for xv, yv, event in zip(
                x_valid[outlier_mask],
                y_valid[outlier_mask],
                event_valid[outlier_mask],
            ):
                ax.annotate(
                    str(event),
                    xy=(xv, yv),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color="red",
                )

        (
            pearson_r,
            pearson_p,
            spearman_r,
            spearman_p,
        ) = calculate_correlations(
            x_valid,
            y_valid,
        )

        correlation_text = (
            f"N = {len(x_valid)}\n"
            f"Pearson r = {pearson_r:.3f}"
            f"  (p={pearson_p:.2g})\n"
            f"Spearman rho = {spearman_r:.3f}"
            f"  (p={spearman_p:.2g})\n"
            f"outliers = {np.count_nonzero(outlier_mask)}"
        )

        ax.text(
            0.03,
            0.97,
            correlation_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "gray",
            },
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        return event_valid[outlier_mask].tolist()


    # --------------------------------------------------------
    # ch0 / ch1 のscatter plot
    # --------------------------------------------------------

    event_numbers = df_shape.index.to_numpy()

    scatter_outlier_rows = []

    for ich in range(2):

        H = df_shape[f"ch{ich}_H"].to_numpy(dtype=float)
        AL = df_shape[f"ch{ich}_AL"].to_numpy(dtype=float)
        AR = df_shape[f"ch{ich}_AR"].to_numpy(dtype=float)
        rise = df_shape[
            f"ch{ich}_rise1090"
        ].to_numpy(dtype=float)

        # H <= 0 または非有限値では比を定義しない
        valid_H = np.isfinite(H) & (H > 0)

        AL_over_H = np.full(
            len(df_shape),
            np.nan,
            dtype=float,
        )

        AR_over_H = np.full(
            len(df_shape),
            np.nan,
            dtype=float,
        )

        AL_over_H[valid_H] = (
            AL[valid_H] / H[valid_H]
        )

        AR_over_H[valid_H] = (
            AR[valid_H] / H[valid_H]
        )

        # 後の解析・CSV保存にも使えるようdf_shapeへ追加
        df_shape[f"ch{ich}_ALoverH"] = AL_over_H
        df_shape[f"ch{ich}_ARoverH"] = AR_over_H

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 6),
        )

        # ----------------------------------------------------
        # 1. T_rise vs A_L/H
        # ----------------------------------------------------

        outliers_rise_left = draw_scatter_with_outliers(
            ax=axes[0],
            x=AL_over_H,
            y=rise,
            event_numbers=event_numbers,
            xlabel=r"$A_L/H$ [ns]",
            ylabel=r"$T_{\mathrm{rise},10-90}$ [ns]",
            title=(
                f"Ch{ich}: "
                r"$T_{\mathrm{rise}}$ vs $A_L/H$"
            ),
            color=f"C{ich}",
            annotate_outliers=True,
        )

        # ----------------------------------------------------
        # 2. A_R/H vs A_L/H
        # ----------------------------------------------------

        outliers_area = draw_scatter_with_outliers(
            ax=axes[1],
            x=AL_over_H,
            y=AR_over_H,
            event_numbers=event_numbers,
            xlabel=r"$A_L/H$ [ns]",
            ylabel=r"$A_R/H$ [ns]",
            title=(
                f"Ch{ich}: "
                r"$A_R/H$ vs $A_L/H$"
            ),
            color=f"C{ich}",
            annotate_outliers=True,
        )

        fig.suptitle(
            f"Ch{ich} model-independent shape correlations"
        )

        fig.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

        # 外れ値イベントを表形式で保存
        for event in outliers_rise_left:
            scatter_outlier_rows.append({
                "channel": ich,
                "event": int(event),
                "plot": "rise1090_vs_ALoverH",
                "H_mV": df_shape.loc[event, f"ch{ich}_H"],
                "rise1090_ns": df_shape.loc[
                    event,
                    f"ch{ich}_rise1090",
                ],
                "ALoverH_ns": df_shape.loc[
                    event,
                    f"ch{ich}_ALoverH",
                ],
                "ARoverH_ns": df_shape.loc[
                    event,
                    f"ch{ich}_ARoverH",
                ],
            })

        for event in outliers_area:
            scatter_outlier_rows.append({
                "channel": ich,
                "event": int(event),
                "plot": "ARoverH_vs_ALoverH",
                "H_mV": df_shape.loc[event, f"ch{ich}_H"],
                "rise1090_ns": df_shape.loc[
                    event,
                    f"ch{ich}_rise1090",
                ],
                "ALoverH_ns": df_shape.loc[
                    event,
                    f"ch{ich}_ALoverH",
                ],
                "ARoverH_ns": df_shape.loc[
                    event,
                    f"ch{ich}_ARoverH",
                ],
            })


    # --------------------------------------------------------
    # scatterで抽出された外れ値イベントの波形確認
    # --------------------------------------------------------

    df_scatter_outliers = pd.DataFrame(
        scatter_outlier_rows
    )

    if not df_scatter_outliers.empty:

        # 同じイベントが両scatterで外れ値になることがあるため重複除去
        outlier_pairs = (
            df_scatter_outliers[
                ["channel", "event"]
            ]
            .drop_duplicates()
            .sort_values(["channel", "event"])
        )

        pairs = list(
            outlier_pairs.itertuples(
                index=False,
                name=None,
            )
        )

        plots_per_page = 12

        for page_start in range(
            0,
            len(pairs),
            plots_per_page,
        ):

            page_pairs = pairs[
                page_start:page_start + plots_per_page
            ]

            fig, axes = plt.subplots(
                3,
                4,
                figsize=(16, 10),
                sharex=True,
            )

            axes = axes.flatten()

            for ax, (ich, event) in zip(
                axes,
                page_pairs,
            ):

                diag = all_diagnostics[event][ich]
                y = diag["s_mV"]

                ax.plot(
                    t_ns,
                    y,
                    color=f"C{ich}",
                    linewidth=1.2,
                )

                ax.axhline(
                    0,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                )

                ax.plot(
                    diag["tpeak"],
                    y[diag["ipeak"]],
                    "o",
                    color="red",
                    markersize=4,
                )

                if np.isfinite(diag["t10L"]):
                    ax.axvline(
                        diag["t10L"],
                        color="green",
                        linestyle=":",
                        linewidth=0.8,
                    )

                if np.isfinite(diag["t90L"]):
                    ax.axvline(
                        diag["t90L"],
                        color="green",
                        linestyle=":",
                        linewidth=0.8,
                    )

                ax.set_title(
                    f"Ch{ich}, event {event}\n"
                    f"rise={df_shape.loc[event, f'ch{ich}_rise1090']:.2f} ns, "
                    f"AL/H={df_shape.loc[event, f'ch{ich}_ALoverH']:.2f} ns\n"
                    f"AR/H={df_shape.loc[event, f'ch{ich}_ARoverH']:.2f} ns",
                    fontsize=9,
                )

                ax.grid(alpha=0.4)

            # 使用していないsubplotを消す
            for ax in axes[len(page_pairs):]:
                ax.axis("off")

            for ax in axes[-4:]:
                if ax.axison:
                    ax.set_xlabel("Time [ns]")

            fig.suptitle(
                "Waveforms of scatter-plot outliers"
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    for ich, channel in enumerate([ch0, ch1]):

        fig, axes = plt.subplots(
            4,
            4,
            figsize=(16, 10),
            sharex=True
        )

        axes = axes.flatten()

        for idx in range(min(16, nwf)):

            ax = axes[idx]

            diag = all_diagnostics[idx][ich]

            y = diag["s_mV"]

            ax.plot(
                t_ns,
                y,
                alpha=0.7
            )

            # baseline
            ax.axhline(
                0,
                linestyle="--",
                linewidth=1
            )

            # peak
            ax.plot(
                diag["tpeak"],
                y[diag["ipeak"]],
                "o"
            )

            # t0
            if np.isfinite(diag["t0"]):
                ax.axvline(
                    diag["t0"],
                    linestyle=":"
                )

            # 10/90 rise
            if np.isfinite(diag["t10L"]):
                ax.plot(
                    diag["t10L"],
                    0.10 * y[diag["ipeak"]],
                    "o"
                )

            if np.isfinite(diag["t90L"]):
                ax.plot(
                    diag["t90L"],
                    0.90 * y[diag["ipeak"]],
                    "o"
                )

            # 90/10 fall
            if np.isfinite(diag["t90R"]):
                ax.plot(
                    diag["t90R"],
                    0.90 * y[diag["ipeak"]],
                    "o"
                )

            if np.isfinite(diag["t10R"]):
                ax.plot(
                    diag["t10R"],
                    0.10 * y[diag["ipeak"]],
                    "o"
                )

            ax.grid()

            ax.set_title(
                f"event {idx}"
            )

        for ax in axes[-4:]:
            ax.set_xlabel("Time [ns]")

        for i in range(0, 16, 4):
            axes[i].set_ylabel("Signal [mV]")

        fig.suptitle(
            f"Ch{ich} model-independent waveform analysis"
        )

        fig.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)


    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    for ich in range(2):

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 8)
        )

        axes = axes.flatten()

        labels = {
            "H": "H [mV]",
            "rise1090": "10-90 rise [ns]",
            "fall9010": "90-10 fall [ns]",
            "AL": "Left area [mV ns]",
            "AR": "Right area [mV ns]",
            "AoverH": "A/H [ns]",
        }

        for iax, key in enumerate(FEATURES):

            values = df_shape[
                f"ch{ich}_{key}"
            ].values

            values = values[
                np.isfinite(values)
            ]

            axes[iax].hist(
                values,
                bins=50
            )

            axes[iax].set_xlabel(
                labels[key]
            )

            axes[iax].set_ylabel(
                "Counts"
            )

            axes[iax].grid()

        fig.suptitle(
            f"Ch{ich} shape parameters"
        )

        fig.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

# AL/H、AR/Hを追加したdf_shapeを保存し直す
df_shape.to_csv(
    csvname,
    index=False,
)

# scatter外れ値一覧
scatter_outlier_csv = (
    basename + "_scatter_outliers.csv"
)

if not df_scatter_outliers.empty:
    df_scatter_outliers.to_csv(
        scatter_outlier_csv,
        index=False,
    )

    print(
        "saved:",
        scatter_outlier_csv,
    )
else:
    print("No scatter outliers were found.")
print("saved:", pdfname)