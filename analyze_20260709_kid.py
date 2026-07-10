from __future__ import annotations

"""
Analyze 20260709 KID waveform position scan + IQ calibration.

Expected data layout
--------------------
/Volumes/NO NAME/data/20260709/
    5.501GHz_z=7.3mm_x=4.4mm/
        *.npz  # oscilloscope waveform npz files
    5.501GHz_z=8.0mm_x=4.4mm_first/
        *.npz
    ...
    iq_scan_202607091840.npz  # contains dd[:,0]=freq, dd[:,1]=ch0, dd[:,2]=ch1

Outputs
-------
/Users/kubokosei/software/kidanalysis/analysis/data/20260709/analysis_5501GHz/
    summary_by_run.csv
    summary_by_position_1over40.csv
    *.png

Run
---
python /Users/kubokosei/software/kidanalysis/analysis/analyze_20260709_kid.py
"""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DATA_ROOT = Path("/Volumes/NO NAME/data/20260709")
OUT_DIR = Path("/Users/kubokosei/software/kidanalysis/data/20260709/analysis_5501GHz")

READOUT_FREQ_GHZ = 5.501
READOUT_FREQ_HZ = READOUT_FREQ_GHZ * 1e9

# Oscilloscope sample rate. If an npz has sample_rate, that value is used instead.
DEFAULT_SAMPLE_RATE_HZ = 2.5e9

# Baseline: normally the pulse begins after the pre-trigger region.
# If ref_position exists in the npz, the code uses it.
# Otherwise it uses the first BASELINE_FRACTION of the waveform.
BASELINE_FRACTION = 0.20

# For speed, keep all events but downsample plots only.
MAX_WAVEFORMS_TO_PLOT = 250

# Suffix/light-power interpretation from your log.
# Original state: 1/(4*10) = 1/40.
SUFFIX_TO_LIGHT = {
    "first": "1/40",
    "second": "1/40",
    "third": "1/40",
    "one-tenth": "1/10",
    "normal": "1/1",
    "": "1/40",
}

# Central point used for reproducibility/light-power comparison.
CENTRAL_Z_MM = 8.0
CENTRAL_X_MM = 4.4


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x: object, default: float = np.nan) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return float(arr.ravel()[0])
    except Exception:
        return default


def normalize_waveform_array(a: np.ndarray) -> np.ndarray:
    """
    Return waveform array with shape (events, samples).

    Many oscilloscope files are already (events, samples). If the array is
    transposed, this attempts to fix it.
    """
    a = np.asarray(a, dtype=float)

    if a.ndim == 1:
        return a.reshape(1, -1)

    if a.ndim > 2:
        # Collapse all leading dimensions into events and keep last axis as samples.
        a = a.reshape(-1, a.shape[-1])

    # Heuristic: samples are usually much larger than event count.
    # If shape is (samples, events), transpose.
    if a.shape[0] > a.shape[1] and a.shape[1] <= 10000:
        a = a.T

    return a


def pick_key(keys: list[str], candidates: list[str]) -> str | None:
    lower = {k.lower(): k for k in keys}

    # exact first
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    # contains next
    for k in keys:
        kl = k.lower()
        for c in candidates:
            if c.lower() in kl:
                return k

    return None


def load_ch0_ch1_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Load ch0/ch1 arrays from one oscilloscope npz file.

    Expected keys often include ch0/ch1, but this function also handles
    arr_0/arr_1-like files as a fallback.
    """
    with np.load(path, allow_pickle=True) as npz:
        keys = list(npz.keys())

        k0 = pick_key(keys, ["ch0", "channel0", "chan0", "I", "i_data"])
        k1 = pick_key(keys, ["ch1", "channel1", "chan1", "Q", "q_data"])

        if k0 is None or k1 is None:
            # Fallback: use the two largest numeric arrays that are not dd/frequency-like.
            numeric_arrays: list[tuple[int, str, np.ndarray]] = []
            for k in keys:
                if k.lower() in {"dd", "freq", "frequency", "f"}:
                    continue
                arr = np.asarray(npz[k])
                if np.issubdtype(arr.dtype, np.number) and arr.size > 100:
                    numeric_arrays.append((arr.size, k, arr))
            numeric_arrays.sort(reverse=True, key=lambda t: t[0])
            if len(numeric_arrays) < 2:
                raise KeyError(
                    f"Could not find ch0/ch1 in {path}. npz keys = {keys}"
                )
            k0 = numeric_arrays[0][1]
            k1 = numeric_arrays[1][1]

        ch0 = normalize_waveform_array(np.asarray(npz[k0], dtype=float))
        ch1 = normalize_waveform_array(np.asarray(npz[k1], dtype=float))

        if ch0.shape != ch1.shape:
            n_events = min(ch0.shape[0], ch1.shape[0])
            n_samples = min(ch0.shape[1], ch1.shape[1])
            ch0 = ch0[:n_events, :n_samples]
            ch1 = ch1[:n_events, :n_samples]

        meta: dict[str, object] = {
            "keys": keys,
            "ch0_key": k0,
            "ch1_key": k1,
            "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
            "ref_position": None,
        }

        k_sr = pick_key(keys, ["sample_rate", "sample_rate_hz", "fs", "sampling_rate"])
        if k_sr is not None:
            meta["sample_rate_hz"] = safe_float(npz[k_sr], DEFAULT_SAMPLE_RATE_HZ)

        k_ref = pick_key(keys, ["ref_position", "reference_position", "trigger_position"])
        if k_ref is not None:
            meta["ref_position"] = safe_float(npz[k_ref], np.nan)

    return ch0, ch1, meta


def find_iq_scan_files() -> list[Path]:
    return sorted(DATA_ROOT.glob("iq_scan*.npz"))


def load_iq_scan(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as npz:
        if "dd" not in npz:
            raise KeyError(f"{path} has no 'dd'. keys={list(npz.keys())}")
        dd = np.asarray(npz["dd"], dtype=float)

    if dd.ndim != 2 or dd.shape[1] < 3:
        raise ValueError(f"{path}: dd must have shape (N, >=3), got {dd.shape}")

    freq = dd[:, 0]
    # Auto-detect Hz vs GHz-ish.
    freq_hz = freq.copy()
    if np.nanmedian(np.abs(freq_hz)) < 1e7:
        freq_hz = freq_hz * 1e9

    df = pd.DataFrame({
        "freq_hz": freq_hz,
        "freq_ghz": freq_hz / 1e9,
        "ch0": dd[:, 1],
        "ch1": dd[:, 2],
    })
    df["amp"] = np.sqrt(df["ch0"] ** 2 + df["ch1"] ** 2)
    df["phase_rad"] = np.unwrap(np.arctan2(df["ch1"], df["ch0"]))
    return df


@dataclass
class FolderMeta:
    folder: Path
    freq_ghz: float
    z_mm: float
    x_mm: float
    suffix: str
    light: str


def parse_folder_meta(folder: Path) -> FolderMeta | None:
    """
    Parse folder names like:
        5.501GHz_z=8.0mm_x=4.4mm_first
        5.501GHz_z=8.0mm_x=4.4mm_one-tenth
        5.501GHz_z=7.5mm_x=4.0mm
    """
    name = folder.name
    pat = (
        r"(?P<freq>[0-9.]+)GHz"
        r"_z=(?P<z>[0-9.]+)mm"
        r"_x=(?P<x>[0-9.]+)mm"
        r"(?:_(?P<suffix>.+))?$"
    )
    m = re.match(pat, name)
    if m is None:
        return None

    suffix = m.group("suffix") or ""
    light = SUFFIX_TO_LIGHT.get(suffix, "unknown")
    return FolderMeta(
        folder=folder,
        freq_ghz=float(m.group("freq")),
        z_mm=float(m.group("z")),
        x_mm=float(m.group("x")),
        suffix=suffix,
        light=light,
    )


def discover_waveform_folders() -> list[FolderMeta]:
    folders: list[FolderMeta] = []
    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir():
            continue
        meta = parse_folder_meta(p)
        if meta is None:
            continue
        if not list(p.glob("*.npz")):
            warnings.warn(f"No npz files in {p}")
            continue
        folders.append(meta)
    return folders


def load_folder_waveforms(folder: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found in {folder}")

    ch0_list: list[np.ndarray] = []
    ch1_list: list[np.ndarray] = []
    sample_rates: list[float] = []
    ref_positions: list[float] = []
    key_records: list[str] = []

    for f in files:
        ch0, ch1, meta = load_ch0_ch1_from_npz(f)
        ch0_list.append(ch0)
        ch1_list.append(ch1)
        sample_rates.append(float(meta.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ)))
        ref_positions.append(safe_float(meta.get("ref_position"), np.nan))
        key_records.append(f"{f.name}: {meta.get('ch0_key')}/{meta.get('ch1_key')}")

    n_samples = min(a.shape[1] for a in ch0_list + ch1_list)
    ch0_all = np.concatenate([a[:, :n_samples] for a in ch0_list], axis=0)
    ch1_all = np.concatenate([a[:, :n_samples] for a in ch1_list], axis=0)

    meta_out: dict[str, object] = {
        "n_files": len(files),
        "files": [f.name for f in files],
        "sample_rate_hz": float(np.nanmedian(sample_rates)) if sample_rates else DEFAULT_SAMPLE_RATE_HZ,
        "ref_position": float(np.nanmedian(ref_positions)) if np.isfinite(ref_positions).any() else np.nan,
        "key_records": "; ".join(key_records),
    }
    return ch0_all, ch1_all, meta_out


def choose_baseline_slice(n_samples: int, ref_position: float) -> slice:
    """
    Select baseline region.

    ref_position in many scope npz files is a percentage. If present, use
    0 -> about 80% of the trigger index. Otherwise use first 20%.
    """
    if np.isfinite(ref_position):
        if 0 < ref_position < 100:
            trigger_idx = int(n_samples * ref_position / 100.0)
        else:
            trigger_idx = int(ref_position)

        baseline_end = max(10, int(trigger_idx * 0.8))
        baseline_end = min(baseline_end, n_samples // 2)
        return slice(0, baseline_end)

    return slice(0, max(10, int(BASELINE_FRACTION * n_samples)))


def estimate_pulse_window(mean_r: np.ndarray, baseline_sl: slice) -> tuple[int, slice]:
    n = mean_r.size
    bstop = baseline_sl.stop or int(BASELINE_FRACTION * n)

    search_start = min(max(bstop + 5, int(0.05 * n)), n - 1)
    search_stop = n

    if search_start >= search_stop:
        search_start = 0

    peak_idx = int(search_start + np.nanargmax(mean_r[search_start:search_stop]))

    # Event-by-event amplitude window around the common peak.
    half_width = max(20, int(0.03 * n))
    lo = max(search_start, peak_idx - half_width)
    hi = min(n, peak_idx + half_width)
    return peak_idx, slice(lo, hi)


def analyze_one_folder(fm: FolderMeta) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    ch0, ch1, meta = load_folder_waveforms(fm.folder)
    n_events, n_samples = ch0.shape
    sample_rate_hz = float(meta["sample_rate_hz"])
    time_us = np.arange(n_samples) / sample_rate_hz * 1e6

    baseline_sl = choose_baseline_slice(n_samples, safe_float(meta.get("ref_position"), np.nan))

    ped0_evt = np.nanmedian(ch0[:, baseline_sl], axis=1)
    ped1_evt = np.nanmedian(ch1[:, baseline_sl], axis=1)
    ped0 = float(np.nanmedian(ped0_evt))
    ped1 = float(np.nanmedian(ped1_evt))

    d0 = ch0 - ped0_evt[:, None]
    d1 = ch1 - ped1_evt[:, None]

    mean_d0 = np.nanmean(d0, axis=0)
    mean_d1 = np.nanmean(d1, axis=0)
    mean_r = np.sqrt(mean_d0**2 + mean_d1**2)

    peak_idx, amp_window = estimate_pulse_window(mean_r, baseline_sl)
    peak_vec = np.array([mean_d0[peak_idx], mean_d1[peak_idx]], dtype=float)
    peak_norm = float(np.linalg.norm(peak_vec))

    if not np.isfinite(peak_norm) or peak_norm == 0:
        u = np.array([1.0, 0.0])
    else:
        u = peak_vec / peak_norm

    proj = d0 * u[0] + d1 * u[1]
    mean_proj = np.nanmean(proj, axis=0)

    # Keep the projected pulse positive.
    if mean_proj[peak_idx] < 0:
        u = -u
        proj = -proj
        mean_proj = -mean_proj

    amp_evt = np.nanmax(proj[:, amp_window], axis=1)
    amp_median = float(np.nanmedian(amp_evt))
    amp_mean = float(np.nanmean(amp_evt))
    amp_std = float(np.nanstd(amp_evt, ddof=1)) if n_events > 1 else np.nan
    amp_se = amp_std / np.sqrt(n_events) if n_events > 1 else np.nan

    # Integrated projected pulse area and simple tau_eff = area / height.
    # Use only positive part in the common pulse window.
    ywin = np.clip(proj[:, amp_window], 0.0, None)
    dt_us = 1.0 / sample_rate_hz * 1e6
    area_evt = np.nansum(ywin, axis=1) * dt_us
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_eff_us_evt = area_evt / amp_evt
    tau_eff_us = float(np.nanmedian(tau_eff_us_evt))

    row: dict[str, object] = {
        "folder": fm.folder.name,
        "freq_ghz": fm.freq_ghz,
        "z_mm": fm.z_mm,
        "x_mm": fm.x_mm,
        "suffix": fm.suffix,
        "light": fm.light,
        "n_files": int(meta["n_files"]),
        "n_events": int(n_events),
        "n_samples": int(n_samples),
        "sample_rate_hz": sample_rate_hz,
        "baseline_start": int(baseline_sl.start or 0),
        "baseline_stop": int(baseline_sl.stop or 0),
        "peak_idx": peak_idx,
        "peak_time_us": float(time_us[peak_idx]),
        "amp_window_start": int(amp_window.start or 0),
        "amp_window_stop": int(amp_window.stop or n_samples),
        "ped_ch0_median": ped0,
        "ped_ch1_median": ped1,
        "ped_radius": float(np.sqrt(ped0**2 + ped1**2)),
        "u_ch0": float(u[0]),
        "u_ch1": float(u[1]),
        "amp_proj_median": amp_median,
        "amp_proj_mean": amp_mean,
        "amp_proj_std": amp_std,
        "amp_proj_se": amp_se,
        "tau_eff_us_median": tau_eff_us,
        "key_records": meta["key_records"],
    }

    arrays = {
        "time_us": time_us,
        "mean_d0": mean_d0,
        "mean_d1": mean_d1,
        "mean_proj": mean_proj,
        "amp_evt": amp_evt,
        "ped0_evt": ped0_evt,
        "ped1_evt": ped1_evt,
    }

    return row, arrays


def weighted_mean_and_se(values: np.ndarray, ses: np.ndarray | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    values = values[ok]
    if values.size == 0:
        return np.nan, np.nan

    if ses is None:
        if values.size == 1:
            return float(values[0]), np.nan
        return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))

    ses = np.asarray(ses, dtype=float)[ok]
    ok2 = np.isfinite(ses) & (ses > 0)
    if np.count_nonzero(ok2) == 0:
        if values.size == 1:
            return float(values[0]), np.nan
        return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))

    w = 1.0 / ses[ok2] ** 2
    mean = float(np.sum(w * values[ok2]) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return mean, se


# =============================================================================
# PLOTS
# =============================================================================

def plot_iq_calibrations(iq_files: list[Path]) -> None:
    if not iq_files:
        print("[warn] No iq_scan*.npz files found.")
        return

    for path in iq_files:
        try:
            df = load_iq_scan(path)
        except Exception as e:
            print(f"[warn] failed to load IQ scan {path}: {e}")
            continue

        idx = int(np.nanargmin(np.abs(df["freq_hz"].to_numpy() - READOUT_FREQ_HZ)))
        near = df.iloc[idx]

        # IQ trajectory
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        sc = ax.scatter(df["ch0"], df["ch1"], c=df["freq_ghz"], s=30)
        ax.plot(df["ch0"], df["ch1"], lw=0.8, alpha=0.5)
        ax.scatter([near["ch0"]], [near["ch1"]], marker="*", s=220, label=f"nearest {near['freq_ghz']:.6f} GHz")
        ax.set_xlabel("ch0")
        ax.set_ylabel("ch1")
        ax.set_title(f"IQ calibration: {path.name}")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("frequency [GHz]")
        fig.tight_layout()
        out = OUT_DIR / f"iq_calibration_iq_{path.stem}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)

        # Frequency response
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.plot(df["freq_ghz"], df["amp"], marker="o", ms=3, lw=1.0, label=r"$\sqrt{ch0^2+ch1^2}$")
        ax.axvline(READOUT_FREQ_GHZ, ls="--", lw=1.0, label=f"readout {READOUT_FREQ_GHZ:.3f} GHz")
        ax.scatter([near["freq_ghz"]], [near["amp"]], marker="*", s=150)
        ax.set_xlabel("frequency [GHz]")
        ax.set_ylabel("IQ radius")
        ax.set_title(f"IQ scan amplitude: {path.name}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        out = OUT_DIR / f"iq_calibration_amp_{path.stem}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)

        df.to_csv(OUT_DIR / f"iq_calibration_{path.stem}.csv", index=False)


def aggregate_original_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average repeated runs at the same x,z in the original 1/40 light state.
    """
    d = df[df["light"] == "1/40"].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for (z, x), g in d.groupby(["z_mm", "x_mm"], sort=True):
        amp_mean, amp_se = weighted_mean_and_se(
            g["amp_proj_median"].to_numpy(),
            g["amp_proj_se"].to_numpy(),
        )
        tau_mean, tau_se = weighted_mean_and_se(g["tau_eff_us_median"].to_numpy(), None)
        rows.append({
            "z_mm": z,
            "x_mm": x,
            "n_runs": len(g),
            "folders": ", ".join(g["folder"].astype(str)),
            "amp_proj": amp_mean,
            "amp_proj_se": amp_se,
            "tau_eff_us": tau_mean,
            "tau_eff_us_se": tau_se,
            "ped_ch0": float(g["ped_ch0_median"].mean()),
            "ped_ch1": float(g["ped_ch1_median"].mean()),
        })
    return pd.DataFrame(rows)


def plot_position_map(pos: pd.DataFrame) -> None:
    if pos.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sc = ax.scatter(pos["x_mm"], pos["z_mm"], c=pos["amp_proj"], s=220)
    for _, r in pos.iterrows():
        ax.text(r["x_mm"], r["z_mm"], f"{r['amp_proj']:.3g}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Position map: projected pulse amplitude, light=1/40")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("projected amplitude")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "position_map_amp_1over40.png", dpi=220)
    plt.close(fig)


def plot_x_z_scans(pos: pd.DataFrame) -> None:
    if pos.empty:
        return

    # x scan at z=8.0
    dx = pos[np.isclose(pos["z_mm"], CENTRAL_Z_MM)].sort_values("x_mm")
    if not dx.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.errorbar(dx["x_mm"], dx["amp_proj"], yerr=dx["amp_proj_se"], fmt="o", capsize=3)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("projected amplitude")
        ax.set_title(f"x scan at z={CENTRAL_Z_MM:g} mm, light=1/40")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "x_scan_amp_z8p0_1over40.png", dpi=220)
        plt.close(fig)

    # z scan at x=4.4
    dz = pos[np.isclose(pos["x_mm"], CENTRAL_X_MM)].sort_values("z_mm")
    if not dz.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.errorbar(dz["z_mm"], dz["amp_proj"], yerr=dz["amp_proj_se"], fmt="o", capsize=3)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("projected amplitude")
        ax.set_title(f"z scan at x={CENTRAL_X_MM:g} mm, light=1/40")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "z_scan_amp_x4p4_1over40.png", dpi=220)
        plt.close(fig)

    # tau_eff maps/scans
    if "tau_eff_us" in pos.columns:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        sc = ax.scatter(pos["x_mm"], pos["z_mm"], c=pos["tau_eff_us"], s=220)
        for _, r in pos.iterrows():
            ax.text(r["x_mm"], r["z_mm"], f"{r['tau_eff_us']:.3g}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(r"Position map: $\tau_{\rm eff}$, light=1/40")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\tau_{\rm eff}$ [$\mu$s]")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "position_map_tau_eff_1over40.png", dpi=220)
        plt.close(fig)


def plot_pedestal_iq(summary: pd.DataFrame, iq_files: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # Overlay latest IQ calibration if available
    if iq_files:
        try:
            iqdf = load_iq_scan(iq_files[-1])
            ax.plot(iqdf["ch0"], iqdf["ch1"], lw=1.0, alpha=0.4, label=f"IQ scan {iq_files[-1].name}")
            idx = int(np.nanargmin(np.abs(iqdf["freq_hz"].to_numpy() - READOUT_FREQ_HZ)))
            near = iqdf.iloc[idx]
            ax.scatter([near["ch0"]], [near["ch1"]], marker="*", s=180, label=f"nearest {near['freq_ghz']:.6f} GHz")
        except Exception as e:
            print(f"[warn] Could not overlay IQ scan: {e}")

    d = summary[summary["light"] == "1/40"].copy()
    if not d.empty:
        sc = ax.scatter(d["ped_ch0_median"], d["ped_ch1_median"], c=d["z_mm"], s=55, label="pedestal 1/40")
        for _, r in d.iterrows():
            label = f"z{r['z_mm']:g},x{r['x_mm']:g}"
            if str(r["suffix"]):
                label += f" {r['suffix']}"
            ax.text(r["ped_ch0_median"], r["ped_ch1_median"], label, fontsize=7)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("z [mm]")

    ax.set_xlabel("ch0")
    ax.set_ylabel("ch1")
    ax.set_title("Pedestal points on IQ plane")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pedestal_iq_position_scan.png", dpi=220)
    plt.close(fig)


def plot_light_comparison(summary: pd.DataFrame) -> None:
    d = summary[
        np.isclose(summary["z_mm"], CENTRAL_Z_MM)
        & np.isclose(summary["x_mm"], CENTRAL_X_MM)
    ].copy()
    if d.empty:
        return

    # Ordered light levels. Keep repeated 1/40 runs separately in one plot.
    light_order = {"1/40": 0, "1/10": 1, "1/1": 2, "unknown": 9}
    d["light_order"] = d["light"].map(light_order).fillna(9)
    d = d.sort_values(["light_order", "suffix", "folder"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    labels = [
        f"{r.light}\n{r.suffix if str(r.suffix) else 'scan'}"
        for r in d.itertuples()
    ]
    x = np.arange(len(d))
    ax.errorbar(x, d["amp_proj_median"], yerr=d["amp_proj_se"], fmt="o", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("projected amplitude")
    ax.set_title(f"Light-power / repeat comparison at z={CENTRAL_Z_MM:g} mm, x={CENTRAL_X_MM:g} mm")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "light_power_and_repeat_comparison_center.png", dpi=220)
    plt.close(fig)

    # Aggregate by light
    rows = []
    for light, g in d.groupby("light", sort=False):
        mean, se = weighted_mean_and_se(g["amp_proj_median"].to_numpy(), g["amp_proj_se"].to_numpy())
        rows.append({"light": light, "amp_proj": mean, "amp_proj_se": se, "n_runs": len(g)})
    agg = pd.DataFrame(rows)
    agg.to_csv(OUT_DIR / "summary_light_power_center.csv", index=False)

    if not agg.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        x = np.arange(len(agg))
        ax.errorbar(x, agg["amp_proj"], yerr=agg["amp_proj_se"], fmt="o", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(agg["light"])
        ax.set_xlabel("relative light after filters")
        ax.set_ylabel("projected amplitude")
        ax.set_title("Aggregated light-power comparison")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "light_power_aggregated_center.png", dpi=220)
        plt.close(fig)


def plot_mean_projected_waveforms(summary: pd.DataFrame, arrays_by_folder: dict[str, dict[str, np.ndarray]]) -> None:
    # Central light/repeat comparison
    d = summary[
        np.isclose(summary["z_mm"], CENTRAL_Z_MM)
        & np.isclose(summary["x_mm"], CENTRAL_X_MM)
    ].copy()
    if not d.empty:
        light_order = {"1/40": 0, "1/10": 1, "1/1": 2, "unknown": 9}
        d["light_order"] = d["light"].map(light_order).fillna(9)
        d = d.sort_values(["light_order", "suffix", "folder"])

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for _, r in d.iterrows():
            arr = arrays_by_folder.get(r["folder"])
            if arr is None:
                continue
            label = f"{r['light']} {r['suffix']}".strip()
            ax.plot(arr["time_us"], arr["mean_proj"], lw=1.1, label=label)
        ax.set_xlabel("time [us]")
        ax.set_ylabel("mean projected waveform")
        ax.set_title(f"Mean projected waveforms at z={CENTRAL_Z_MM:g} mm, x={CENTRAL_X_MM:g} mm")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "mean_projected_waveforms_center.png", dpi=220)
        plt.close(fig)

    # Original 1/40 position scan, all mean projected waveforms
    d = summary[summary["light"] == "1/40"].sort_values(["z_mm", "x_mm", "suffix"])
    if not d.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        for _, r in d.iterrows():
            arr = arrays_by_folder.get(r["folder"])
            if arr is None:
                continue
            label = f"z{r['z_mm']:g} x{r['x_mm']:g}"
            if str(r["suffix"]):
                label += f" {r['suffix']}"
            ax.plot(arr["time_us"], arr["mean_proj"], lw=0.9, label=label)
        ax.set_xlabel("time [us]")
        ax.set_ylabel("mean projected waveform")
        ax.set_title("Mean projected waveforms: position scan, light=1/40")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "mean_projected_waveforms_position_scan_1over40.png", dpi=220)
        plt.close(fig)


def plot_pedestal_time_variation(summary: pd.DataFrame, arrays_by_folder: dict[str, dict[str, np.ndarray]]) -> None:
    """
    Diagnostic plot: pedestal event cloud for central repeats/light-power runs.
    Useful for checking 1 Hz temperature-induced baseline motion.
    """
    d = summary[
        np.isclose(summary["z_mm"], CENTRAL_Z_MM)
        & np.isclose(summary["x_mm"], CENTRAL_X_MM)
    ].copy()
    if d.empty:
        return

    light_order = {"1/40": 0, "1/10": 1, "1/1": 2, "unknown": 9}
    d["light_order"] = d["light"].map(light_order).fillna(9)
    d = d.sort_values(["light_order", "suffix", "folder"])

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for _, r in d.iterrows():
        arr = arrays_by_folder.get(r["folder"])
        if arr is None:
            continue
        ped0 = arr["ped0_evt"]
        ped1 = arr["ped1_evt"]
        step = max(1, len(ped0) // MAX_WAVEFORMS_TO_PLOT)
        label = f"{r['light']} {r['suffix']}".strip()
        ax.scatter(ped0[::step], ped1[::step], s=8, alpha=0.35, label=label)
    ax.set_xlabel("event pedestal ch0")
    ax.set_ylabel("event pedestal ch1")
    ax.set_title("Event-by-event pedestal cloud at center")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pedestal_event_cloud_center.png", dpi=220)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_out_dir()

    print("=" * 90)
    print("20260709 KID analysis")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"OUT_DIR   = {OUT_DIR}")
    print("=" * 90)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    iq_files = find_iq_scan_files()
    print("[IQ scan files]")
    for f in iq_files:
        print(f"  - {f.name}")
    plot_iq_calibrations(iq_files)

    folders = discover_waveform_folders()
    print("\n[waveform folders]")
    for fm in folders:
        print(f"  - {fm.folder.name:45s} light={fm.light}")

    if not folders:
        raise RuntimeError(f"No waveform folders found under {DATA_ROOT}")

    rows: list[dict[str, object]] = []
    arrays_by_folder: dict[str, dict[str, np.ndarray]] = {}

    for i, fm in enumerate(folders, start=1):
        print(f"\n[{i}/{len(folders)}] analyzing {fm.folder.name}")
        try:
            row, arrays = analyze_one_folder(fm)
        except Exception as e:
            print(f"[error] failed: {fm.folder.name}: {e}")
            continue
        rows.append(row)
        arrays_by_folder[fm.folder.name] = arrays
        print(
            f"    events={row['n_events']}, amp_med={row['amp_proj_median']:.6g}, "
            f"tau_eff={row['tau_eff_us_median']:.6g} us, peak={row['peak_time_us']:.4g} us"
        )

    if not rows:
        raise RuntimeError("All folder analyses failed.")

    summary = pd.DataFrame(rows).sort_values(["light", "z_mm", "x_mm", "suffix", "folder"])
    summary.to_csv(OUT_DIR / "summary_by_run.csv", index=False)
    print(f"\n[saved] {OUT_DIR / 'summary_by_run.csv'}")

    pos = aggregate_original_positions(summary)
    if not pos.empty:
        pos = pos.sort_values(["z_mm", "x_mm"])
        pos.to_csv(OUT_DIR / "summary_by_position_1over40.csv", index=False)
        print(f"[saved] {OUT_DIR / 'summary_by_position_1over40.csv'}")

    plot_position_map(pos)
    plot_x_z_scans(pos)
    plot_pedestal_iq(summary, iq_files)
    plot_light_comparison(summary)
    plot_mean_projected_waveforms(summary, arrays_by_folder)
    plot_pedestal_time_variation(summary, arrays_by_folder)

    print("\n[done] Output files:")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
