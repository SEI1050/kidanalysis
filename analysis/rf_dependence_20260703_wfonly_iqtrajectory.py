"""
20260703 / T = 4.76 K
レーザー照射パルスの入力 RF 周波数依存性を解析する。

入力は、測定日のフォルダ直下にある条件フォルダと、その中の
`wf_*.npz` のみを使う。

  /Volumes/NO NAME/data/
  ├── 20260703/
  │   ├── 4.76K_5.4653GHz/
  │   │   └── wf_260703_....npz
  │   ├── 4.76K_5.4853GHz/
  │   └── ...
  └── analysis/
      └── 20260703_rf_dependence_4.76K/  # 出力先

この版では IQ scan ファイルを前提にしない。
したがって、S21 sweep / resonance circle を重ねる解析は行わず、
wf_*.npz に含まれる ch0=I, ch1=Q の波形だけから以下を作る。

  * 各 RF の pedestal と laser response vector の IQ 平面図
  * pedestal-subtracted ch0/ch1 平均波形
  * 応答方向への射影波形と直交成分
  * peak, IQ amplitude, response angle, tau_eff, SNR, peak time の RF 依存

実行例:
  python analyze_rf_dependence_20260703_wfonly.py

外付け SSD 名や保存場所が違う場合:
  python analyze_rf_dependence_20260703_wfonly.py \
      --data-root "/Volumes/NO NAME/data"

指定温度だけを変更したい場合:
  python analyze_rf_dependence_20260703_wfonly.py --temp 5.80
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def trapezoid_integral(y, x):
    """NumPy 1.x / 2.x の両方で動く台形積分。"""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x)
    return np.trapz(y, x=x)


# ============================================================
# 基本設定
# ============================================================
DEFAULT_DATA_ROOT = Path("/Volumes/NO NAME/data")
DEFAULT_RUN_NAME = "20260703"
DEFAULT_TEMP_K = 4.76

# 波形データの仕様
SAMPLE_RATE_HZ = 2.5e9
N_PRE = 500                      # 各 event で pedestal を取るサンプル数
PEAK_TMIN_NS = 250.0             # peak 探索窓
PEAK_TMAX_NS = 1500.0
PLOT_TMIN_NS = 0.0
PLOT_TMAX_NS = 1800.0
INTEGRAL_TMIN_NS = 250.0         # tau_eff の積分窓
INTEGRAL_TMAX_NS = 1800.0

# pedestal の 1 Hz 揺れが強い場合にも、平均が壊れないように
# event ごとの pre-trigger pedestal を引いてから平均する。
# True にすると pedestal PCA の中心 80% の event だけで平均する。
USE_CENTRAL_PEDESTAL_EVENTS = False
CENTRAL_PEDESTAL_FRACTION = 0.80

# 今回は phase shifter 調整後の定義通り、ch0=I、ch1=Q として使う。
FLIP_CH0 = False
FLIP_CH1 = False

# main() 内で実際の出力先に差し替える
# --out-dir を指定しない通常実行時の保存先
# 既存の解析結果と混ざる点に注意。必要なら --out-dir で個別フォルダを指定できる。
DEFAULT_OUT_DIR = Path("/Users/kubokosei/software/kidanalysis/analysis/data")
OUT_DIR = DEFAULT_OUT_DIR


# ============================================================
# 入出力: 20260703 のフォルダ構成を自動認識
# ============================================================
CONDITION_RE = re.compile(
    r"^(?P<temp>\d+(?:\.\d+)?)K_(?P<freq>\d+(?:\.\d+)?)GHz(?:.*)?$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="20260703 の KID レーザー応答を RF 周波数ごとに比較する。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help='20260512, 20260703, analysis が並ぶ親フォルダ（default: "/Volumes/NO NAME/data"）',
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="測定日のフォルダ名（default: 20260703）",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMP_K,
        help="解析する温度 [K]（default: 4.76）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力先を明示指定する場合のパス",
    )
    parser.add_argument(
        "--only-freq",
        type=float,
        nargs="*",
        default=None,
        metavar="GHz",
        help="特定周波数だけ解析する場合。例: --only-freq 5.4853 5.4953",
    )
    return parser.parse_args()


def find_channel_keys(keys: list[str]) -> tuple[str, str]:
    """npz の ch0/ch1 に相当する key を探す。"""
    lower_to_original = {k.lower(): k for k in keys}
    candidates0 = ["ch0", "channel0", "wave0", "data0", "i"]
    candidates1 = ["ch1", "channel1", "wave1", "data1", "q"]

    key0 = next((lower_to_original[k] for k in candidates0 if k in lower_to_original), None)
    key1 = next((lower_to_original[k] for k in candidates1 if k in lower_to_original), None)
    if key0 is None or key1 is None:
        raise KeyError(
            "ch0/ch1 (または I/Q) の key が見つかりません。"
            f" npz keys = {keys}"
        )
    return key0, key1


def ensure_event_sample_shape(a: np.ndarray) -> np.ndarray:
    """配列を (event, sample) に揃える。"""
    a = np.asarray(a)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Unexpected waveform array shape: {a.shape}")
    # 多くの場合、sample 数（例: 5000）> event 数（例: 500）。
    return a.T if a.shape[0] > a.shape[1] else a


def discover_conditions(
    run_dir: Path,
    temp_k: float,
    only_freq: list[float] | None = None,
    temp_tol_k: float = 0.02,
) -> list[tuple[float, Path]]:
    """
    run_dir 直下の '4.76K_5.4953GHz' 形式フォルダを全て検出する。
    data_0703_... など、条件名に一致しないフォルダは自動的に無視される。
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"測定日フォルダがありません: {run_dir}")

    found: list[tuple[float, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = CONDITION_RE.match(p.name)
        if m is None:
            continue

        this_temp = float(m.group("temp"))
        this_freq = float(m.group("freq"))
        if abs(this_temp - temp_k) > temp_tol_k:
            continue
        if only_freq is not None and not any(np.isclose(this_freq, x, atol=5e-4) for x in only_freq):
            continue
        found.append((this_freq, p))

    found.sort(key=lambda x: x[0])
    if not found:
        labels = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"{temp_k:.2f} K の 'xxK_xxGHz' フォルダが見つかりません。\n"
            f"run_dir = {run_dir}\n"
            f"見つかったフォルダ = {labels}"
        )
    return found


def find_waveform_files(folder: Path) -> list[Path]:
    """
    各条件フォルダ直下の波形ファイルを探す。

    まず Finder に見えている wf_*.npz を優先する。
    それがなければ、I/Q (ch0/ch1) を持つ直下の npz を候補にする。
    解析結果などを誤って混ぜないよう、サブフォルダは探索しない。
    """
    preferred = sorted(folder.glob("wf_*.npz"))
    if preferred:
        return preferred

    candidates: list[Path] = []
    for fp in sorted(folder.glob("*.npz")):
        name = fp.name.lower()
        try:
            with np.load(fp, allow_pickle=True) as d:
                find_channel_keys(list(d.keys()))
        except (KeyError, OSError, ValueError):
            continue
        candidates.append(fp)
    return candidates


def load_all_waveforms(folder: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """条件フォルダ内の波形 npz を読み、(event, sample) で連結する。"""
    files = find_waveform_files(folder)
    if not files:
        raise FileNotFoundError(
            f"波形 npz が見つかりません: {folder}\n"
            "wf_*.npz が存在するか、ch0/ch1 key を含む npz かを確認してください。"
        )

    ch0_list: list[np.ndarray] = []
    ch1_list: list[np.ndarray] = []
    for fp in files:
        with np.load(fp, allow_pickle=True) as d:
            key0, key1 = find_channel_keys(list(d.keys()))
            ch0 = ensure_event_sample_shape(d[key0])
            ch1 = ensure_event_sample_shape(d[key1])

        if ch0.shape != ch1.shape:
            raise ValueError(f"ch0/ch1 shape mismatch: {fp}\n{ch0.shape} vs {ch1.shape}")
        ch0_list.append(ch0)
        ch1_list.append(ch1)

    return np.concatenate(ch0_list, axis=0), np.concatenate(ch1_list, axis=0), files


# ============================================================
# 解析
# ============================================================
@dataclass
class Result:
    freq_ghz: float
    folder: Path
    waveform_files: list[Path]
    n_event: int
    time_ns: np.ndarray
    ped0_each: np.ndarray
    ped1_each: np.ndarray
    ped0_mean: float
    ped1_mean: float
    mean_d0: np.ndarray
    mean_d1: np.ndarray
    sem_d0: np.ndarray
    sem_d1: np.ndarray
    mean_parallel: np.ndarray
    mean_perpendicular: np.ndarray
    peak_idx: int
    peak_time_ns: float
    peak_d0: float
    peak_d1: float
    peak_amp: float
    response_angle_deg: float
    tau_eff_ns: float
    noise_sigma_event: float
    snr_event: float
    n_used: int


def select_events_from_pedestal(ped0: np.ndarray, ped1: np.ndarray) -> np.ndarray:
    """pedestal IQ の主成分から中心付近の event を選ぶ（任意）。"""
    n = len(ped0)
    if not USE_CENTRAL_PEDESTAL_EVENTS:
        return np.ones(n, dtype=bool)

    ped = np.column_stack([ped0, ped1])
    ped_z = (ped - np.mean(ped, axis=0)) / np.maximum(np.std(ped, axis=0, ddof=1), 1e-30)
    _, _, vh = np.linalg.svd(ped_z, full_matrices=False)
    pc1 = ped_z @ vh[0]
    lo = np.quantile(pc1, (1.0 - CENTRAL_PEDESTAL_FRACTION) / 2.0)
    hi = np.quantile(pc1, 1.0 - (1.0 - CENTRAL_PEDESTAL_FRACTION) / 2.0)
    return (pc1 >= lo) & (pc1 <= hi)


def analyze_one_frequency(
    freq_ghz: float,
    folder: Path,
) -> Result:
    ch0, ch1, waveform_files = load_all_waveforms(folder)
    print(f"[load] {folder.name}: {len(waveform_files)} waveform file(s), {ch0.shape[0]} events")

    if FLIP_CH0:
        ch0 = -ch0
    if FLIP_CH1:
        ch1 = -ch1

    n_event, n_sample = ch0.shape
    time_ns = np.arange(n_sample) / SAMPLE_RATE_HZ * 1e9
    if N_PRE >= n_sample:
        raise ValueError(f"N_PRE={N_PRE} >= waveform length={n_sample}")

    # event ごとの pedestal を引く。1 Hz baseline 揺れを平均波形に混ぜない。
    ped0_each = np.mean(ch0[:, :N_PRE], axis=1)
    ped1_each = np.mean(ch1[:, :N_PRE], axis=1)
    use = select_events_from_pedestal(ped0_each, ped1_each)

    d0 = ch0[use] - ped0_each[use, None]
    d1 = ch1[use] - ped1_each[use, None]
    n_used = d0.shape[0]
    mean_d0, mean_d1 = np.mean(d0, axis=0), np.mean(d1, axis=0)
    sem_d0 = np.std(d0, axis=0, ddof=1) / np.sqrt(n_used)
    sem_d1 = np.std(d1, axis=0, ddof=1) / np.sqrt(n_used)

    peak_window = (time_ns >= PEAK_TMIN_NS) & (time_ns <= PEAK_TMAX_NS)
    if not np.any(peak_window):
        raise ValueError("peak search window が波形時間範囲にありません。")

    amp_trace = np.hypot(mean_d0, mean_d1)
    candidates = np.flatnonzero(peak_window)
    peak_idx = int(candidates[np.argmax(amp_trace[candidates])])
    peak_vec = np.array([mean_d0[peak_idx], mean_d1[peak_idx]])
    peak_amp = float(np.linalg.norm(peak_vec))
    if peak_amp == 0:
        raise RuntimeError(f"{freq_ghz:.4f} GHz: IQ response peak is zero.")

    # 各 RF の応答ベクトルを +parallel 方向に取る。
    u_parallel = peak_vec / peak_amp
    u_perp = np.array([-u_parallel[1], u_parallel[0]])
    mean_parallel = u_parallel[0] * mean_d0 + u_parallel[1] * mean_d1
    mean_perpendicular = u_perp[0] * mean_d0 + u_perp[1] * mean_d1

    integral_mask = (time_ns >= INTEGRAL_TMIN_NS) & (time_ns <= INTEGRAL_TMAX_NS)
    area = trapezoid_integral(
        np.clip(mean_parallel[integral_mask], 0.0, None),
        time_ns[integral_mask],
    )
    tau_eff_ns = float(area / peak_amp)

    pre_parallel = u_parallel[0] * d0[:, :N_PRE] + u_parallel[1] * d1[:, :N_PRE]
    noise_sigma_event = float(np.median(np.std(pre_parallel, axis=1, ddof=1)))
    snr_event = float(peak_amp / noise_sigma_event) if noise_sigma_event > 0 else np.nan

    return Result(
        freq_ghz=freq_ghz,
        folder=folder,
        waveform_files=waveform_files,
        n_event=n_event,
        time_ns=time_ns,
        ped0_each=ped0_each,
        ped1_each=ped1_each,
        ped0_mean=float(np.mean(ped0_each[use])),
        ped1_mean=float(np.mean(ped1_each[use])),
        mean_d0=mean_d0,
        mean_d1=mean_d1,
        sem_d0=sem_d0,
        sem_d1=sem_d1,
        mean_parallel=mean_parallel,
        mean_perpendicular=mean_perpendicular,
        peak_idx=peak_idx,
        peak_time_ns=float(time_ns[peak_idx]),
        peak_d0=float(peak_vec[0]),
        peak_d1=float(peak_vec[1]),
        peak_amp=peak_amp,
        response_angle_deg=float(np.degrees(np.arctan2(peak_vec[1], peak_vec[0]))),
        tau_eff_ns=tau_eff_ns,
        noise_sigma_event=noise_sigma_event,
        snr_event=snr_event,
        n_used=n_used,
    )


# ============================================================
# 描画
# ============================================================
def set_common_time_axis(ax: plt.Axes) -> None:
    ax.set_xlim(PLOT_TMIN_NS, PLOT_TMAX_NS)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time from waveform start [ns]")


def plot_iq_pedestal_and_vectors(
    results: list[Result],
    temp_k: float,
) -> None:
    """wf_*.npz から得た動作点とレーザー応答を IQ 平面に描く。"""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    for r in results:
        x0, y0 = r.ped0_mean, r.ped1_mean
        dx, dy = r.peak_d0, r.peak_d1
        ax.scatter(x0, y0, s=58, marker="o", edgecolor="black", linewidth=0.8, zorder=5)
        ax.scatter(x0 + dx, y0 + dy, s=75, marker="*", zorder=6)
        ax.annotate(
            "",
            xy=(x0 + dx, y0 + dy),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.4),
            zorder=4,
        )
        ax.annotate(
            f"{r.freq_ghz:.4f}",
            xy=(x0, y0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("ch0 = I")
    ax.set_ylabel("ch1 = Q")
    ax.set_title(f"{temp_k:.2f} K: pedestal positions and laser response vectors")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_iq_pedestal_and_response_vectors.png", dpi=250)
    plt.close(fig)


def plot_mean_ch0_ch1(results: list[Result], temp_k: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharex=True)
    for r in results:
        m = (r.time_ns >= PLOT_TMIN_NS) & (r.time_ns <= PLOT_TMAX_NS)
        label = f"{r.freq_ghz:.4f} GHz"
        axes[0].plot(r.time_ns[m], r.mean_d0[m], lw=1.7, label=label)
        axes[1].plot(r.time_ns[m], r.mean_d1[m], lw=1.7, label=label)

    axes[0].set_ylabel(r"$\langle\Delta I\rangle$ = $\langle\Delta$ch0$\rangle$")
    axes[1].set_ylabel(r"$\langle\Delta Q\rangle$ = $\langle\Delta$ch1$\rangle$")
    for ax, title in zip(axes, ["ch0 / I", "ch1 / Q"]):
        set_common_time_axis(ax)
        ax.set_title(title)
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.4)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{temp_k:.2f} K: pedestal-subtracted mean laser waveforms")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_mean_ch0_ch1_waveforms.png", dpi=250)
    plt.close(fig)


def plot_projected_waveforms(results: list[Result], temp_k: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharex=True)
    for r in results:
        m = (r.time_ns >= PLOT_TMIN_NS) & (r.time_ns <= PLOT_TMAX_NS)
        label = f"{r.freq_ghz:.4f} GHz"
        axes[0].plot(r.time_ns[m], r.mean_parallel[m], lw=1.7, label=label)
        axes[1].plot(r.time_ns[m], r.mean_perpendicular[m], lw=1.5, label=label)

    axes[0].set_ylabel(r"$\Delta S_\parallel$")
    axes[1].set_ylabel(r"$\Delta S_\perp$")
    for ax, title in zip(axes, ["Along each RF response vector", "Perpendicular to response vector"]):
        set_common_time_axis(ax)
        ax.set_title(title)
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.4)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{temp_k:.2f} K: coordinate-independent waveform comparison")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_signal_axis_projected_waveforms.png", dpi=250)
    plt.close(fig)


def plot_pedestal_iq(results: list[Result], temp_k: float) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    for r in results:
        ax.scatter(r.ped0_each, r.ped1_each, s=7, alpha=0.22, label=f"{r.freq_ghz:.4f} GHz")
    ax.set_xlabel("pre-trigger pedestal ch0 = I")
    ax.set_ylabel("pre-trigger pedestal ch1 = Q")
    ax.set_title(f"{temp_k:.2f} K: pedestal distribution for each readout RF")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_pedestal_iq_by_frequency.png", dpi=250)
    plt.close(fig)



def _add_direction_arrow(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """IQ 軌跡の時間進行方向を矢印で示す。"""
    if len(x) < 4:
        return
    i0 = max(0, int(0.30 * (len(x) - 1)))
    i1 = min(len(x) - 1, i0 + max(2, int(0.08 * len(x))))
    ax.annotate(
        "",
        xy=(x[i1], y[i1]),
        xytext=(x[i0], y[i0]),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        zorder=8,
    )


def plot_iq_trajectory(results: list[Result], temp_k: float) -> None:
    """
    波形の時系列を IQ 平面上の軌跡として描く。

    左: 実際の pedestal 位置を含む I-Q 平面。
    右: 各周波数の pedestal を原点にそろえた ΔI-ΔQ 平面。
    丸が開始 pedestal、星が IQ amplitude 最大の時刻、矢印が時間進行方向。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0))

    for r in results:
        m = (r.time_ns >= PLOT_TMIN_NS) & (r.time_ns <= PLOT_TMAX_NS)
        label = f"{r.freq_ghz:.4f} GHz"

        # 絶対 IQ 軌跡: 各 RF の pedestal 位置からの実際の動き
        i_abs = r.ped0_mean + r.mean_d0[m]
        q_abs = r.ped1_mean + r.mean_d1[m]
        line = axes[0].plot(i_abs, q_abs, lw=1.7, label=label)[0]
        color = line.get_color()
        axes[0].scatter(
            r.ped0_mean, r.ped1_mean,
            s=42, marker="o", color=color, edgecolor="black", linewidth=0.55, zorder=9,
        )
        axes[0].scatter(
            r.ped0_mean + r.peak_d0, r.ped1_mean + r.peak_d1,
            s=76, marker="*", color=color, edgecolor="black", linewidth=0.45, zorder=10,
        )
        _add_direction_arrow(axes[0], i_abs, q_abs)

        # pedestal を引いた軌跡: 波形形状・向きだけを比較する IQ 平面
        di = r.mean_d0[m]
        dq = r.mean_d1[m]
        axes[1].plot(di, dq, lw=1.7, color=color, label=label)
        axes[1].scatter(
            0.0, 0.0,
            s=42, marker="o", color=color, edgecolor="black", linewidth=0.55, zorder=9,
        )
        axes[1].scatter(
            r.peak_d0, r.peak_d1,
            s=76, marker="*", color=color, edgecolor="black", linewidth=0.45, zorder=10,
        )
        _add_direction_arrow(axes[1], di, dq)

    axes[0].set_xlabel("ch0 = I")
    axes[0].set_ylabel("ch1 = Q")
    axes[0].set_title("Absolute IQ trajectory")
    axes[0].legend(fontsize=8, loc="best")

    axes[1].set_xlabel(r"$\Delta$ch0 = $\Delta I$")
    axes[1].set_ylabel(r"$\Delta$ch1 = $\Delta Q$")
    axes[1].set_title("Pedestal-subtracted IQ trajectory")

    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.65, alpha=0.30)
        ax.axvline(0.0, color="black", lw=0.65, alpha=0.30)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        f"{temp_k:.2f} K: time evolution of mean laser response in the IQ plane",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_iq_trajectory.png", dpi=250)
    plt.close(fig)

def plot_summary(results: list[Result], temp_k: float) -> None:
    f = np.array([r.freq_ghz for r in results])
    d0 = np.array([r.peak_d0 for r in results])
    d1 = np.array([r.peak_d1 for r in results])
    amp = np.array([r.peak_amp for r in results])
    angle = np.array([r.response_angle_deg for r in results])
    tau_eff = np.array([r.tau_eff_ns for r in results])
    snr = np.array([r.snr_event for r in results])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    axes = axes.ravel()
    axes[0].plot(f, d0, "o-", label=r"$\Delta I_\mathrm{peak}$")
    axes[0].plot(f, d1, "o-", label=r"$\Delta Q_\mathrm{peak}$")
    axes[0].axhline(0.0, color="black", lw=0.7, alpha=0.4)
    axes[0].set_ylabel("Peak response [arb. unit]")
    axes[0].set_title("Raw ch0/ch1 peak")
    axes[0].legend()

    axes[1].plot(f, amp, "o-")
    axes[1].set_ylabel(r"$|\Delta S_\mathrm{peak}|$")
    axes[1].set_title("IQ response magnitude")

    axes[2].plot(f, np.unwrap(np.radians(angle), discont=np.pi) * 180.0 / np.pi, "o-")
    axes[2].set_ylabel(r"Response angle $\theta_\mathrm{resp}$ [deg]")
    axes[2].set_title(r"$\tan^{-1}(\Delta Q/\Delta I)$")

    axes[3].plot(f, tau_eff, "o-")
    axes[3].set_ylabel(r"$\tau_\mathrm{eff}$ [ns]")
    axes[3].set_title(r"Effective width: $\int\Delta S_\parallel dt / \Delta S_\mathrm{peak}$")

    axes[4].plot(f, snr, "o-")
    axes[4].set_ylabel("Approx. event SNR")
    axes[4].set_title("Peak amplitude / pre-trigger noise")

    peak_time = np.array([r.peak_time_ns for r in results])
    axes[5].plot(f, peak_time, "o-")
    axes[5].set_ylabel("Peak time [ns]")
    axes[5].set_title("Pulse timing")

    for ax in axes:
        if ax.axison:
            ax.set_xlabel("Readout RF [GHz]")
            ax.grid(True, alpha=0.25)

    fig.suptitle(f"{temp_k:.2f} K: RF-frequency dependence of laser response", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_rf_dependence_summary.png", dpi=250)
    plt.close(fig)


def save_summary(results: list[Result], run_dir: Path, temp_k: float) -> None:
    rows: list[dict[str, float | int | str]] = []
    for r in results:
        rows.append({
            "freq_GHz": r.freq_ghz,
            "condition_folder": str(r.folder),
            "waveform_files": "; ".join(str(p) for p in r.waveform_files),
            "N_event_total": r.n_event,
            "N_event_used": r.n_used,
            "pedestal_ch0_mean": r.ped0_mean,
            "pedestal_ch1_mean": r.ped1_mean,
            "peak_time_ns": r.peak_time_ns,
            "peak_dch0": r.peak_d0,
            "peak_dch1": r.peak_d1,
            "peak_amp_IQ": r.peak_amp,
            "response_angle_deg": r.response_angle_deg,
            "tau_eff_ns": r.tau_eff_ns,
            "noise_sigma_event": r.noise_sigma_event,
            "approx_event_snr": r.snr_event,
        })

    with (OUT_DIR / "rf_dependence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 入力対応表。何を読んだかを後から確かめられる。
    with (OUT_DIR / "input_manifest.txt").open("w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"temperature_K: {temp_k:.4f}\n")
        for r in results:
            f.write(f"{r.freq_ghz:.4f} GHz\n")
            f.write(f"  folder: {r.folder}\n")
            for wf in r.waveform_files:
                f.write(f"  waveform: {wf}\n")

    np.savez_compressed(
        OUT_DIR / "mean_waveforms.npz",
        time_ns=results[0].time_ns,
        freq_GHz=np.array([r.freq_ghz for r in results]),
        mean_dch0=np.stack([r.mean_d0 for r in results]),
        mean_dch1=np.stack([r.mean_d1 for r in results]),
        mean_parallel=np.stack([r.mean_parallel for r in results]),
        mean_perpendicular=np.stack([r.mean_perpendicular for r in results]),
    )


# ============================================================
# 実行
# ============================================================
def main() -> None:
    global OUT_DIR
    args = parse_args()

    run_dir = args.data_root / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"run_dir がありません: {run_dir}\n"
            "Finder の 20260703 フォルダまでのパスに合わせて --data-root または --run-name を指定してください。"
        )

    # 通常実行では Mac 内の analysis/data へ直接保存する。
    # 条件ごとに分けたい場合だけ --out-dir を指定する。
    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUT_DIR
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions = discover_conditions(run_dir, args.temp, args.only_freq)
    print(f"[run] {run_dir}")
    print("[conditions]")
    for freq, folder in conditions:
        print(f"  {freq:.4f} GHz  <-  {folder.name}")

    results = [
        analyze_one_frequency(freq, folder)
        for freq, folder in conditions
    ]

    plot_iq_pedestal_and_vectors(results, args.temp)
    plot_mean_ch0_ch1(results, args.temp)
    plot_projected_waveforms(results, args.temp)
    plot_pedestal_iq(results, args.temp)
    plot_iq_trajectory(results, args.temp)
    plot_summary(results, args.temp)
    save_summary(results, run_dir, args.temp)

    print("\nSaved files:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
