from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# 設定
# ============================================================

ROOT_DIR = Path("/Volumes/NO NAME/data/20260709")

OUTPUT_PDF = Path(
    "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/view_all_waveform/"
    "all_folders_waveform_overlay_base_subtracted_9perpage.pdf"
)

BASE_FRACTION = 0.10   # 冒頭10%を baseline 算出に使う
PANELS_PER_PAGE = 9
NROWS = 3
NCOLS = 3

DEFAULT_SAMPLE_RATE = 2.5e9

# True にすると、1フォルダ内に wf_*.npz が複数あった場合に最初の1個だけ使う
USE_ONLY_FIRST_WF_IN_EACH_FOLDER = True


# ============================================================
# ユーティリティ
# ============================================================

def find_waveform_files(root_dir: Path):
    """
    各フォルダ内の wf_*.npz を探す。
    """
    waveform_files = []

    for subdir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        wf_list = sorted(subdir.glob("wf_*.npz"))
        if not wf_list:
            continue

        if USE_ONLY_FIRST_WF_IN_EACH_FOLDER:
            waveform_files.append(wf_list[0])
            if len(wf_list) > 1:
                print(f"[info] {subdir.name}: wf_*.npz が複数あります。先頭のみ使用 -> {wf_list[0].name}")
        else:
            waveform_files.extend(wf_list)

    return waveform_files


def ensure_2d_event_sample(ch0: np.ndarray, ch1: np.ndarray):
    """
    配列を (イベント数, サンプル数) にそろえる。
    """
    if ch0.ndim == 1:
        ch0 = ch0[np.newaxis, :]
        ch1 = ch1[np.newaxis, :]

    # (サンプル数, イベント数) なら転置
    if ch0.shape[0] > ch0.shape[1]:
        ch0 = ch0.T
        ch1 = ch1.T

    return ch0, ch1


def subtract_baseline(arr: np.ndarray, base_fraction: float):
    """
    各イベントごとに、最初の base_fraction 分の平均を baseline として引く。
    arr shape = (n_events, n_samples)
    """
    n_events, n_samples = arr.shape
    n_base = max(1, int(n_samples * base_fraction))

    baseline = np.mean(arr[:, :n_base], axis=1, keepdims=True)
    arr_sub = arr - baseline

    return arr_sub, baseline.squeeze()


def load_waveform(npz_path: Path, base_fraction: float):
    """
    1つの npz を読み込み、baseline subtraction まで行う。
    """
    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.keys())
        print(f"[load] {npz_path}")
        print(f"       keys = {keys}")

        ch0 = np.asarray(data["ch0"], dtype=float)
        ch1 = np.asarray(data["ch1"], dtype=float)

        if "sample_rate" in data:
            sample_rate = float(np.asarray(data["sample_rate"]).squeeze())
        else:
            sample_rate = DEFAULT_SAMPLE_RATE

    ch0, ch1 = ensure_2d_event_sample(ch0, ch1)
    n_events, n_samples = ch0.shape

    time_ns = np.arange(n_samples) / sample_rate * 1e9

    # baseline subtraction
    ch0_sub, ch0_base = subtract_baseline(ch0, base_fraction)
    ch1_sub, ch1_base = subtract_baseline(ch1, base_fraction)

    mean_ch0 = np.mean(ch0_sub, axis=0)
    mean_ch1 = np.mean(ch1_sub, axis=0)

    return {
        "folder_name": npz_path.parent.name,
        "file_name": npz_path.name,
        "path": npz_path,
        "sample_rate": sample_rate,
        "time_ns": time_ns,
        "n_events": n_events,
        "n_samples": n_samples,
        "ch0": ch0_sub,
        "ch1": ch1_sub,
        "mean_ch0": mean_ch0,
        "mean_ch1": mean_ch1,
        "ch0_base": ch0_base,
        "ch1_base": ch1_base,
    }


def robust_ylim(arr: np.ndarray, mean_waveform: np.ndarray):
    """
    外れ値に引っ張られにくい y 軸範囲を作る。
    """
    values = np.concatenate([arr.ravel(), mean_waveform.ravel()])

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0

    y_low, y_high = np.percentile(finite_values, [0.5, 99.5])

    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
        y_low = np.min(finite_values)
        y_high = np.max(finite_values)

    if y_low == y_high:
        y_low -= 1.0
        y_high += 1.0

    pad = 0.08 * (y_high - y_low)
    return y_low - pad, y_high + pad


def draw_page(pdf, dataset_chunk, channel_key: str, page_index: int, total_pages: int):
    """
    1ページ（3x3=9枚）描く
    channel_key = "ch0" or "ch1"
    """
    fig, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=(14, 10),
        constrained_layout=True,
    )
    axes = axes.ravel()

    mean_key = f"mean_{channel_key}"

    for ax, ds in zip(axes, dataset_chunk):
        x = ds["time_ns"]
        y_all = ds[channel_key]
        y_mean = ds[mean_key]

        # 全イベント
        for i in range(ds["n_events"]):
            ax.plot(
                x,
                y_all[i],
                color="gray",
                linewidth=0.35,
                alpha=0.08,
            )

        # 平均波形
        ax.plot(
            x,
            y_mean,
            color="black",
            linewidth=1.5,
        )

        y0, y1 = robust_ylim(y_all, y_mean)
        ax.set_ylim(y0, y1)

        ax.set_title(
            f"{ds['folder_name']}\nN={ds['n_events']}",
            fontsize=9
        )
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    # 余った枠は消す
    for ax in axes[len(dataset_chunk):]:
        ax.axis("off")

    # 軸ラベル（簡易）
    for idx, ax in enumerate(axes):
        row = idx // NCOLS
        col = idx % NCOLS

        if row == NROWS - 1:
            ax.set_xlabel("Time [ns]", fontsize=9)
        if col == 0:
            ax.set_ylabel(channel_key, fontsize=9)

    fig.suptitle(
        f"{channel_key}  |  all-event overlay + mean waveform\n"
        f"Baseline subtraction: mean of first {BASE_FRACTION*100:.0f}% samples of each event\n"
        f"gray = all events, black = mean   |   page {page_index}/{total_pages}",
        fontsize=13
    )

    pdf.savefig(fig)
    plt.close(fig)


# ============================================================
# メイン処理
# ============================================================

def main():
    waveform_files = find_waveform_files(ROOT_DIR)

    if not waveform_files:
        raise FileNotFoundError(f"wf_*.npz が見つかりませんでした: {ROOT_DIR}")

    print(f"[info] 対象ファイル数: {len(waveform_files)}")

    datasets = []
    for wf_path in waveform_files:
        try:
            ds = load_waveform(wf_path, BASE_FRACTION)
            datasets.append(ds)
        except Exception as e:
            print(f"[warning] 読み込み失敗: {wf_path}")
            print(f"          {e}")

    if not datasets:
        raise RuntimeError("読み込み可能なデータがありませんでした。")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(datasets) + PANELS_PER_PAGE - 1) // PANELS_PER_PAGE
    total_pages = n_chunks * 2   # ch0 と ch1

    page_counter = 1

    with PdfPages(OUTPUT_PDF) as pdf:
        # ch0 pages
        for i in range(0, len(datasets), PANELS_PER_PAGE):
            chunk = datasets[i:i + PANELS_PER_PAGE]
            draw_page(
                pdf=pdf,
                dataset_chunk=chunk,
                channel_key="ch0",
                page_index=page_counter,
                total_pages=total_pages,
            )
            page_counter += 1

        # ch1 pages
        for i in range(0, len(datasets), PANELS_PER_PAGE):
            chunk = datasets[i:i + PANELS_PER_PAGE]
            draw_page(
                pdf=pdf,
                dataset_chunk=chunk,
                channel_key="ch1",
                page_index=page_counter,
                total_pages=total_pages,
            )
            page_counter += 1

    print(f"[saved] {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np


# # ============================================================
# # 設定
# # ============================================================

# WAVEFORM_FILE = Path(
#     "/Volumes/NO NAME/data/20260709/"
#     "5.501GHz_z=8.0mm_x=4.4mm_first/"
#     "wf_260709_175104_49.78Hz.npz"
# )

# OUTPUT_FILE = Path(
#     "/Users/kubokosei/software/kidanalysis/analysis/data/20260709/view_all_waveform/"
#     "all_events_and_mean_waveform.png"
# )


# # ============================================================
# # データを読み込む
# # ============================================================

# with np.load(WAVEFORM_FILE, allow_pickle=False) as data:
#     print("npz keys:", list(data.keys()))

#     ch0 = np.asarray(data["ch0"], dtype=float)
#     ch1 = np.asarray(data["ch1"], dtype=float)

#     if "sample_rate" in data:
#         sample_rate = float(
#             np.asarray(data["sample_rate"]).squeeze()
#         )
#     else:
#         sample_rate = 2.5e9


# # ============================================================
# # 配列を (イベント数, サンプル数) にそろえる
# # ============================================================

# if ch0.ndim == 1:
#     ch0 = ch0[np.newaxis, :]
#     ch1 = ch1[np.newaxis, :]

# # (サンプル数, イベント数) だった場合は転置
# if ch0.shape[0] > ch0.shape[1]:
#     ch0 = ch0.T
#     ch1 = ch1.T


# n_events, n_samples = ch0.shape

# print("イベント数:", n_events)
# print("サンプル数:", n_samples)
# print("sample rate:", sample_rate)


# # ============================================================
# # 時間軸
# # ============================================================

# time_ns = (
#     np.arange(n_samples)
#     / sample_rate
#     * 1e9
# )


# # ============================================================
# # 全イベントの平均波形
# # ============================================================

# mean_ch0 = np.mean(ch0, axis=0)
# mean_ch1 = np.mean(ch1, axis=0)


# # ============================================================
# # プロット
# # ============================================================

# fig, axes = plt.subplots(
#     2,
#     1,
#     figsize=(12, 8),
#     sharex=True,
# )


# # ------------------------------------------------------------
# # ch0
# # ------------------------------------------------------------

# for event_index in range(n_events):
#     axes[0].plot(
#         time_ns,
#         ch0[event_index],
#         color="gray",
#         linewidth=0.4,
#         alpha=0.08,
#     )

# axes[0].plot(
#     time_ns,
#     mean_ch0,
#     color="black",
#     linewidth=2.5,
#     label="Mean waveform",
# )

# axes[0].set_title(
#     f"All waveform events: {WAVEFORM_FILE.parent.name}\n"
#     f"{n_events} events"
# )
# axes[0].set_ylabel("ch0")
# axes[0].grid(alpha=0.3)
# axes[0].legend()


# # ------------------------------------------------------------
# # ch1
# # ------------------------------------------------------------

# for event_index in range(n_events):
#     axes[1].plot(
#         time_ns,
#         ch1[event_index],
#         color="gray",
#         linewidth=0.4,
#         alpha=0.08,
#     )

# axes[1].plot(
#     time_ns,
#     mean_ch1,
#     color="black",
#     linewidth=2.5,
#     label="Mean waveform",
# )

# axes[1].set_xlabel("Time [ns]")
# axes[1].set_ylabel("ch1")
# axes[1].grid(alpha=0.3)
# axes[1].legend()


# # ============================================================
# # 保存
# # ============================================================

# fig.tight_layout()

# OUTPUT_FILE.parent.mkdir(
#     parents=True,
#     exist_ok=True,
# )

# fig.savefig(
#     OUTPUT_FILE,
#     dpi=200,
#     bbox_inches="tight",
# )

# print("saved:", OUTPUT_FILE)

# plt.show()