from __future__ import annotations

"""
Realtime raw waveform picker/viewer for 20260521 Americium KID data.

目的:
  - GUIで data_0521_..., data_0522_... などのフォルダを選ぶ
  - 生波形を 25 event くらい重ね書きで確認する
  - 冒頭から25個、末尾から25個、ランダム25個、任意startから25個を選べる

置き場所:
  /Users/kubokosei/software/kidanalysis/view_raw_waveforms_realtime.py

実行:
  cd /Users/kubokosei/software/kidanalysis
  python view_raw_waveforms_realtime.py

注意:
  GUIなのでローカルMacのターミナルから実行してください。
"""

from pathlib import Path
import math
import re
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import matplotlib
# GUI表示用。Macで問題があればコメントアウトして、matplotlibの自動backendに任せてください。
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# =============================================================================
# SETTINGS
# =============================================================================

ROOT_DIR = Path("/Volumes/NO NAME/data/20260709")

DEFAULT_SAMPLE_RATE_HZ = 2.5e9
DEFAULT_N_EVENTS = 25

# ch0/ch1の符号を後から変えたい場合だけ変更
CH0_SIGN = 1.0
CH1_SIGN = 1.0

# 表示が重すぎる場合に間引く。5000 sample程度ならそのまま出る。
MAX_PLOT_POINTS = 5000

# binフォルダを候補に含めるか
INCLUDE_BIN_DIR = False


# =============================================================================
# NPZ LOADER
# =============================================================================

def find_sample_rate(z: np.lib.npyio.NpzFile) -> float:
    candidates = [
        "sample_rate", "samplerate", "sampling_rate", "fs", "rate",
        "sample_rate_hz", "sampling_rate_hz",
    ]
    for key in candidates:
        if key in z:
            try:
                value = float(np.asarray(z[key]).squeeze())
                if value > 0:
                    return value
            except Exception:
                pass
    return DEFAULT_SAMPLE_RATE_HZ


def get_array_by_candidates(z: np.lib.npyio.NpzFile, candidates: list[str]) -> np.ndarray | None:
    lower_to_key = {k.lower(): k for k in z.keys()}
    for cand in candidates:
        if cand in z:
            arr = np.asarray(z[cand])
            if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
                return arr
        low = cand.lower()
        if low in lower_to_key:
            arr = np.asarray(z[lower_to_key[low]])
            if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
                return arr
    return None


def as_event_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Return shape = (n_events, n_samples).

    1Dなら1イベント扱い。
    3D以上なら最後の軸をsample軸として、手前をevent軸に潰す。
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim > 2:
        arr = arr.reshape((-1, arr.shape[-1]))
    return arr.astype(np.float64, copy=False)


def split_combined_channel_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    ch0/ch1が一つの配列にまとまっている場合のfallback。
    よくある形:
      (2, n_events, n_samples)
      (n_events, 2, n_samples)
      (n_events, n_samples, 2)
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        return None

    if arr.shape[0] == 2:
        return as_event_matrix(arr[0]), as_event_matrix(arr[1])

    if arr.ndim >= 3 and arr.shape[1] == 2:
        return as_event_matrix(arr[:, 0, ...]), as_event_matrix(arr[:, 1, ...])

    if arr.shape[-1] == 2:
        return as_event_matrix(arr[..., 0]), as_event_matrix(arr[..., 1])

    return None


def load_waveform_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    npzから ch0, ch1 を読む。
    戻り値:
      ch0: shape (n_events, n_samples)
      ch1: shape (n_events, n_samples)
      sample_rate_hz
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            # iq_scanのddは波形ではないので除外
            if "dd" in z:
                return None

            sample_rate_hz = find_sample_rate(z)

            ch0_candidates = [
                "ch0", "CH0", "Ch0", "channel0", "Channel0",
                "wave0", "wf0", "y0", "data_ch0",
                "ch0_waveform", "ch0_wf", "trace0",
            ]
            ch1_candidates = [
                "ch1", "CH1", "Ch1", "channel1", "Channel1",
                "wave1", "wf1", "y1", "data_ch1",
                "ch1_waveform", "ch1_wf", "trace1",
            ]

            ch0 = get_array_by_candidates(z, ch0_candidates)
            ch1 = get_array_by_candidates(z, ch1_candidates)

            if ch0 is not None and ch1 is not None:
                ch0 = as_event_matrix(ch0) * CH0_SIGN
                ch1 = as_event_matrix(ch1) * CH1_SIGN
            else:
                # fallback 1: 一つの配列に2chが入っている場合
                skip_words = ("time", "timestamp", "sample", "rate", "freq", "dd", "ref", "position")
                arrays: list[tuple[str, np.ndarray]] = []
                for key in z.keys():
                    if any(w in key.lower() for w in skip_words):
                        continue
                    arr = np.asarray(z[key])
                    if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 1 and arr.size > 100:
                        arrays.append((key, arr))

                for _, arr in arrays:
                    split = split_combined_channel_array(arr)
                    if split is not None:
                        ch0, ch1 = split
                        ch0 = ch0 * CH0_SIGN
                        ch1 = ch1 * CH1_SIGN
                        break

                # fallback 2: 同じshapeの数値配列2つをch0/ch1とみなす
                if ch0 is None or ch1 is None:
                    matrices = [(k, as_event_matrix(a)) for k, a in arrays]
                    for i in range(len(matrices)):
                        for j in range(i + 1, len(matrices)):
                            a = matrices[i][1]
                            b = matrices[j][1]
                            if a.shape == b.shape:
                                print(f"[fallback] {path.name}: {matrices[i][0]} -> ch0, {matrices[j][0]} -> ch1")
                                ch0 = a * CH0_SIGN
                                ch1 = b * CH1_SIGN
                                break
                        if ch0 is not None and ch1 is not None:
                            break

            if ch0 is None or ch1 is None:
                return None

            n_events = min(ch0.shape[0], ch1.shape[0])
            n_samples = min(ch0.shape[1], ch1.shape[1])
            if n_events <= 0 or n_samples <= 10:
                return None

            return ch0[:n_events, :n_samples], ch1[:n_events, :n_samples], sample_rate_hz

    except Exception as e:
        print(f"[load error] {path}: {e}")
        return None


def waveform_npz_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.rglob("*.npz"))
    if not INCLUDE_BIN_DIR:
        files = [p for p in files if "bin" not in p.relative_to(run_dir).parts]
    return files


def discover_run_dirs(root: Path) -> list[Path]:
    """
    ROOT_DIR直下の data_* フォルダを候補にする。
    もし直下に無ければ、再帰的に data_* を探す。
    """
    if not root.exists():
        return []

    # direct = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("data_")])
    direct = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("5.501")])
    if direct:
        return direct

    found = sorted([p for p in root.rglob("data_*") if p.is_dir()])
    return found


def count_events_in_file(path: Path) -> tuple[int, int, float] | None:
    loaded = load_waveform_npz(path)
    if loaded is None:
        return None
    ch0, ch1, fs = loaded
    return ch0.shape[0], ch0.shape[1], fs


def build_file_index(run_dir: Path) -> list[dict[str, object]]:
    """
    runフォルダ内のnpzファイルごとに、
    global event index の範囲を作る。
    """
    index: list[dict[str, object]] = []
    start = 0
    for path in waveform_npz_files(run_dir):
        info = count_events_in_file(path)
        if info is None:
            continue
        n_events, n_samples, fs = info
        index.append({
            "path": path,
            "start": start,
            "stop": start + n_events,
            "n_events": n_events,
            "n_samples": n_samples,
            "sample_rate_hz": fs,
        })
        start += n_events
    return index


def select_event_indices(total_events: int, mode: str, n_pick: int, start_index: int, seed: int) -> np.ndarray:
    if total_events <= 0:
        return np.array([], dtype=int)

    n_pick = max(1, min(int(n_pick), total_events))

    if mode == "冒頭から":
        return np.arange(n_pick, dtype=int)

    if mode == "末尾から":
        return np.arange(total_events - n_pick, total_events, dtype=int)

    if mode == "ランダム":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total_events, size=n_pick, replace=False)).astype(int)

    if mode == "任意startから":
        s = max(0, min(int(start_index), total_events - 1))
        e = min(total_events, s + n_pick)
        return np.arange(s, e, dtype=int)

    return np.arange(n_pick, dtype=int)


def load_selected_events(file_index: list[dict[str, object]], event_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    global event indexを指定して、そのイベントのch0/ch1だけ取り出す。
    戻り値:
      ch0_sel, ch1_sel, sample_rate_hz, actual_global_indices
    """
    if len(event_indices) == 0:
        return np.empty((0, 0)), np.empty((0, 0)), DEFAULT_SAMPLE_RATE_HZ, np.array([], dtype=int)

    selected_ch0: list[np.ndarray] = []
    selected_ch1: list[np.ndarray] = []
    selected_global: list[int] = []
    sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ
    min_samples: int | None = None

    for item in file_index:
        path = item["path"]
        start = int(item["start"])
        stop = int(item["stop"])
        m = (event_indices >= start) & (event_indices < stop)
        if not np.any(m):
            continue

        local_indices = event_indices[m] - start
        loaded = load_waveform_npz(Path(path))
        if loaded is None:
            continue

        ch0, ch1, fs = loaded
        sample_rate_hz = fs

        for gidx, lidx in zip(event_indices[m], local_indices):
            if 0 <= lidx < ch0.shape[0]:
                selected_ch0.append(ch0[int(lidx)])
                selected_ch1.append(ch1[int(lidx)])
                selected_global.append(int(gidx))
                min_samples = ch0.shape[1] if min_samples is None else min(min_samples, ch0.shape[1])

    if not selected_ch0 or min_samples is None:
        return np.empty((0, 0)), np.empty((0, 0)), sample_rate_hz, np.array([], dtype=int)

    ch0_out = np.asarray([x[:min_samples] for x in selected_ch0], dtype=float)
    ch1_out = np.asarray([x[:min_samples] for x in selected_ch1], dtype=float)
    return ch0_out, ch1_out, sample_rate_hz, np.asarray(selected_global, dtype=int)


def baseline_subtract(y: np.ndarray, frac: float = 0.20) -> np.ndarray:
    if y.size == 0:
        return y
    n_base = max(10, min(1000, int(y.shape[1] * frac)))
    ped = np.median(y[:, :n_base], axis=1)
    return y - ped[:, None]


def downsample_time_and_waveforms(time_us: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y.shape[1] <= MAX_PLOT_POINTS:
        return time_us, y
    step = int(math.ceil(y.shape[1] / MAX_PLOT_POINTS))
    return time_us[::step], y[:, ::step]


# =============================================================================
# GUI
# =============================================================================

class RawWaveformViewer:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("20260521 raw waveform viewer")
        self.master.geometry("1200x800")

        self.root_dir_var = tk.StringVar(value=str(ROOT_DIR))
        self.folder_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="冒頭から")
        self.channel_var = tk.StringVar(value="ch0+ch1")
        self.baseline_var = tk.BooleanVar(value=False)
        self.n_events_var = tk.StringVar(value=str(DEFAULT_N_EVENTS))
        self.start_index_var = tk.StringVar(value="0")
        self.seed_var = tk.StringVar(value="20260521")

        self.run_dirs: list[Path] = []
        self.file_index: list[dict[str, object]] = []

        self._build_ui()
        self.refresh_folders()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.master, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Root").grid(row=0, column=0, sticky="w")
        root_entry = ttk.Entry(top, textvariable=self.root_dir_var, width=55)
        root_entry.grid(row=0, column=1, columnspan=4, sticky="we", padx=4)

        refresh_btn = ttk.Button(top, text="フォルダ再読込", command=self.refresh_folders)
        refresh_btn.grid(row=0, column=5, sticky="we", padx=4)

        ttk.Label(top, text="run folder").grid(row=1, column=0, sticky="w")
        self.folder_combo = ttk.Combobox(top, textvariable=self.folder_var, width=45, state="readonly")
        self.folder_combo.grid(row=1, column=1, columnspan=2, sticky="we", padx=4)
        self.folder_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_folder_changed())

        ttk.Label(top, text="選び方").grid(row=1, column=3, sticky="e")
        mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["冒頭から", "末尾から", "ランダム", "任意startから"],
            width=14,
            state="readonly",
        )
        mode_combo.grid(row=1, column=4, sticky="we", padx=4)

        plot_btn = ttk.Button(top, text="表示更新", command=self.update_plot)
        plot_btn.grid(row=1, column=5, sticky="we", padx=4)

        ttk.Label(top, text="N events").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.n_events_var, width=8).grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(top, text="start index").grid(row=2, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.start_index_var, width=10).grid(row=2, column=3, sticky="w", padx=4)

        ttk.Label(top, text="channel").grid(row=2, column=4, sticky="e")
        channel_combo = ttk.Combobox(
            top,
            textvariable=self.channel_var,
            values=["ch0+ch1", "ch0", "ch1", "sqrt(ch0^2+ch1^2)"],
            width=18,
            state="readonly",
        )
        channel_combo.grid(row=2, column=5, sticky="we", padx=4)

        ttk.Checkbutton(top, text="baselineを引いて表示", variable=self.baseline_var).grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Label(top, text="random seed").grid(row=3, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.seed_var, width=10).grid(row=3, column=3, sticky="w", padx=4)

        self.info_label = ttk.Label(top, text="info: -")
        self.info_label.grid(row=3, column=4, columnspan=2, sticky="w")

        for c in range(6):
            top.columnconfigure(c, weight=1)

        # Matplotlib area
        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def refresh_folders(self) -> None:
        root = Path(self.root_dir_var.get()).expanduser()
        self.run_dirs = discover_run_dirs(root)

        names = [p.name for p in self.run_dirs]
        self.folder_combo["values"] = names

        if names:
            if self.folder_var.get() not in names:
                self.folder_var.set(names[0])
            self.on_folder_changed()
        else:
            self.folder_var.set("")
            self.info_label.config(text=f"info: data_* folder not found under {root}")
            self.ax.clear()
            self.ax.set_title("No data_* folders found")
            self.canvas.draw_idle()

    def on_folder_changed(self) -> None:
        name = self.folder_var.get()
        run_dir = self._current_run_dir()
        if run_dir is None:
            return

        self.info_label.config(text=f"loading index for {name} ...")
        self.master.update_idletasks()

        self.file_index = build_file_index(run_dir)
        total = self.total_events()
        n_files = len(self.file_index)

        if total == 0:
            self.info_label.config(text=f"info: {name}: no waveform npz found")
        else:
            first = self.file_index[0]
            self.info_label.config(
                text=(
                    f"info: {name}: files={n_files}, events={total}, "
                    f"samples≈{first['n_samples']}, fs≈{float(first['sample_rate_hz']):.4g} Hz"
                )
            )
            self.update_plot()

    def _current_run_dir(self) -> Path | None:
        name = self.folder_var.get()
        if not name:
            return None
        for p in self.run_dirs:
            if p.name == name:
                return p
        return None

    def total_events(self) -> int:
        if not self.file_index:
            return 0
        return int(self.file_index[-1]["stop"])

    def update_plot(self) -> None:
        try:
            n_pick = int(self.n_events_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "N events は整数にしてください")
            return

        try:
            start_index = int(self.start_index_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "start index は整数にしてください")
            return

        try:
            seed = int(self.seed_var.get())
        except ValueError:
            seed = 20260521

        total = self.total_events()
        if total <= 0:
            self.ax.clear()
            self.ax.set_title("No waveform events found")
            self.canvas.draw_idle()
            return

        event_indices = select_event_indices(
            total_events=total,
            mode=self.mode_var.get(),
            n_pick=n_pick,
            start_index=start_index,
            seed=seed,
        )

        ch0, ch1, fs, actual_indices = load_selected_events(self.file_index, event_indices)
        if ch0.size == 0:
            self.ax.clear()
            self.ax.set_title("Could not load selected events")
            self.canvas.draw_idle()
            return

        if self.baseline_var.get():
            ch0_plot = baseline_subtract(ch0)
            ch1_plot = baseline_subtract(ch1)
            y_label_suffix = "baseline-subtracted"
        else:
            ch0_plot = ch0
            ch1_plot = ch1
            y_label_suffix = "raw"

        n_samples = ch0_plot.shape[1]
        time_us = np.arange(n_samples, dtype=float) / fs * 1e6

        channel = self.channel_var.get()
        self.ax.clear()

        if channel == "ch0":
            t, y = downsample_time_and_waveforms(time_us, ch0_plot)
            for i in range(y.shape[0]):
                self.ax.plot(t, y[i], alpha=0.45, linewidth=1.0, label=f"ev {actual_indices[i]}" if i < 5 else None)
            self.ax.set_ylabel(f"ch0 [{y_label_suffix}]")

        elif channel == "ch1":
            t, y = downsample_time_and_waveforms(time_us, ch1_plot)
            for i in range(y.shape[0]):
                self.ax.plot(t, y[i], alpha=0.45, linewidth=1.0, label=f"ev {actual_indices[i]}" if i < 5 else None)
            self.ax.set_ylabel(f"ch1 [{y_label_suffix}]")

        elif channel == "sqrt(ch0^2+ch1^2)":
            amp = np.sqrt(ch0_plot**2 + ch1_plot**2)
            t, y = downsample_time_and_waveforms(time_us, amp)
            for i in range(y.shape[0]):
                self.ax.plot(t, y[i], alpha=0.45, linewidth=1.0, label=f"ev {actual_indices[i]}" if i < 5 else None)
            self.ax.set_ylabel(f"sqrt(ch0^2+ch1^2) [{y_label_suffix}]")

        else:
            # ch0 + ch1を同じ図に重ねる。ch0実線、ch1破線。
            t, y0 = downsample_time_and_waveforms(time_us, ch0_plot)
            _, y1 = downsample_time_and_waveforms(time_us, ch1_plot)
            for i in range(y0.shape[0]):
                self.ax.plot(t, y0[i], alpha=0.38, linewidth=1.0)
                self.ax.plot(t, y1[i], alpha=0.38, linewidth=1.0, linestyle="--")
            self.ax.plot([], [], color="black", linestyle="-", label="ch0")
            self.ax.plot([], [], color="black", linestyle="--", label="ch1")
            self.ax.set_ylabel(f"amplitude [{y_label_suffix}]")

        run_name = self.folder_var.get()
        mode = self.mode_var.get()
        idx_text = f"{actual_indices[0]}–{actual_indices[-1]}" if len(actual_indices) > 0 else "-"
        self.ax.set_title(
            f"{run_name} | {mode} {len(actual_indices)} events | global event index {idx_text}"
        )
        self.ax.set_xlabel("time [µs]")
        self.ax.grid(True, alpha=0.3)

        if len(actual_indices) <= 5 or channel == "ch0+ch1":
            self.ax.legend(loc="best", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    app = RawWaveformViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
