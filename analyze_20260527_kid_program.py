#!/usr/bin/env python3
"""
Unified KID analysis for 20260527

Task 1: 1 Hz pedestal IQ motion vs S21 sweep
Task 2: center illumination, RF-frequency dependence
Task 3: position scan at the RF tone with the largest 2D pulse response

Raw ch0/ch1 are treated as ADC coordinates. They are called physical I/Q only
when RAW_IQ_SWEEP_FILE is supplied and the canonical sweep calibration succeeds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG: edit here
# =============================================================================
DATA_DATE = "20260527"
NPZ_PATTERN = "wf_*.npz"

# Prefer the SSD because it contains waveform npz files.
INPUT_ROOTS = [
    Path("/Volumes/NO NAME/data") / DATA_DATE,
    Path.home() / "Library/CloudStorage/OneDrive-TheUniversityofTokyo/東京大学/4S/kidfit" / DATA_DATE,
    Path.home() / "OneDrive - The University of Tokyo/東京大学/4S/kidfit" / DATA_DATE,
    Path(__file__).resolve().parent / "data" / DATA_DATE,
]

SSD_ROOT = Path("/Volumes/NO NAME/data") / DATA_DATE
OUT_DIR = Path.home() / "software" / "kidanalysis" / "data" / DATA_DATE / "analysis_kid_program"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use first-only folders. _second, _third, ... are excluded.
FIRST_ONLY = True

# Waveform settings
N_EVENTS_PER_CONDITION = 1000
BASELINE_WINDOW_US = None          # None means all t < 0
AMP_WINDOW_US = (0.0, 1.5)
INTEGRAL_WINDOW_US = (0.0, 2.0)

# Temperature modulation: 50 Hz laser / 1 Hz temperature
EVENTS_PER_TEMP_CYCLE = 50
N_TEMP_BINS = 10
EVENTS_PER_TEMP_BIN = EVENTS_PER_TEMP_CYCLE // N_TEMP_BINS
assert EVENTS_PER_TEMP_CYCLE % N_TEMP_BINS == 0

# Task 1 reference waveform data
REFERENCE_DIR_NAME = "5.476GHz_z=7.5mm_x=3.4mm"
F_TONE_REFERENCE_GHZ = 5.476
FR_REFERENCE_GHZ = 5.467

# Canonical S21 calibration from a RAW ADC IQ sweep measured with the same
# mixer / phase shifter / ADC setup as waveform data.
# Examples supported: CSV or NPZ with frequency + ch0/ch1 (or I/Q) columns.
# Set None to run in relative-IQ mode only.
RAW_IQ_SWEEP_FILE = None
SWEEP_CABLE_DELAY_NS = None       # optional; set only if known
SWEEP_FIT_RANGE_GHZ = None        # e.g. (5.44, 5.50)

# d = Ql/|Qc|. Update with the closest relevant notch fit.
QL_FOR_D = 452.6852
QC_FOR_D = 488.4716
D_NOTCH = QL_FOR_D / abs(QC_FOR_D)

# Task 2 center illumination location and RF range
CENTER_Z_MM = 7.5
CENTER_X_MM = 3.4
CENTER_FREQ_RANGE_GHZ = (5.44, 5.52)

# Task 3 choice:
# "auto_peak_2d" selects the Task-2 center frequency with maximum median A2D.
# Or set e.g. POSITION_TONE_GHZ = 5.476.
POSITION_TONE_GHZ: str | float = "auto_peak_2d"

# For position comparisons, default uses all events.
# "reference_resonance_phase" is valid only if all directories started at
# the same temperature-cycle phase.
POSITION_TEMP_SELECTION = "all_events"

DPI = 260

# =============================================================================
# Utility
# =============================================================================
MEAS_RE = re.compile(
    r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>.+))?$"
)


def scalar(x):
    a = np.asarray(x)
    return a.item() if a.size == 1 else x


def time_axis_s(npts, fs_hz, ref_percent):
    return (np.arange(npts, dtype=float) - npts * ref_percent / 100.0) / fs_hz


def med_iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def normz(z):
    return z / abs(z) if abs(z) > 0 else 1.0 + 0.0j


def close(a, b, atol=1e-6):
    return np.isclose(a, b, atol=atol, rtol=0.0)


def parse_dir(path: Path):
    m = MEAS_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "path": path,
        "folder": path.name,
        "freq_ghz": float(d["freq"]),
        "z_mm": float(d["z"]),
        "x_mm": float(d["x"]),
        "tag": d["tag"] or "",
    }


def has_npz(path: Path):
    return any(path.glob(NPZ_PATTERN))


# =============================================================================
# Discovery and waveform loading
# =============================================================================
def discover_measurements():
    print("\n===== input roots =====")
    out = {}
    for root in INPUT_ROOTS:
        root = Path(root).expanduser()
        print(root, "exists=", root.is_dir())
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            info = parse_dir(p)
            if info is None:
                continue
            if FIRST_ONLY and info["tag"]:
                continue
            if not has_npz(p):
                continue
            # External SSD root comes first; duplicate names are ignored later.
            out.setdefault(p.name, info)
    ms = sorted(out.values(), key=lambda q: (q["freq_ghz"], q["z_mm"], q["x_mm"]))
    if not ms:
        raise RuntimeError(f"No first measurement directories containing {NPZ_PATTERN} were found.")
    print("valid first directories:", len(ms))
    return ms


def load_events(info, nmax=N_EVENTS_PER_CONDITION):
    files = sorted(info["path"].glob(NPZ_PATTERN), key=lambda p: p.stat().st_mtime)
    ch0_list, ch1_list, used = [], [], []
    nload, tref = 0, None
    for p in files:
        if nload >= nmax:
            break
        try:
            d = np.load(p)
            required = ["ch0", "ch1", "npts", "sample_rate", "ref_position"]
            if any(k not in d.files for k in required):
                print("skip keys:", p.name)
                continue
            c0 = np.asarray(d["ch0"], float)
            c1 = np.asarray(d["ch1"], float)
            if c0.ndim == 1: c0 = c0[None, :]
            if c1.ndim == 1: c1 = c1[None, :]
            if c0.shape != c1.shape:
                print("skip shape:", p.name)
                continue
            npts = int(scalar(d["npts"]))
            fs = float(scalar(d["sample_rate"]))
            rp = float(scalar(d["ref_position"]))
            if c0.shape[1] != npts:
                print("skip npts:", p.name)
                continue
            t = time_axis_s(npts, fs, rp)
            if tref is None:
                tref = t
            elif len(t) != len(tref) or not np.allclose(t, tref):
                print("skip time mismatch:", p.name)
                continue
            ntake = min(len(c0), nmax - nload)
            ch0_list.append(c0[:ntake]); ch1_list.append(c1[:ntake])
            used.append(p); nload += ntake
        except Exception as e:
            print("skip read error:", p, e)
    if tref is None or nload == 0:
        raise RuntimeError(f"Could not read waveform data from {info['path']}")
    if nload < nmax:
        print(f"WARNING {info['folder']}: loaded {nload}/{nmax} events")
    return tref, np.vstack(ch0_list), np.vstack(ch1_list), used


# =============================================================================
# Pedestal-subtracted waveform features
# =============================================================================
def features(t_s, ch0, ch1, select_events=None):
    t_us = t_s * 1e6
    bmask = t_us < 0 if BASELINE_WINDOW_US is None else ((t_us >= BASELINE_WINDOW_US[0]) & (t_us <= BASELINE_WINDOW_US[1]))
    amask = (t_us >= AMP_WINDOW_US[0]) & (t_us <= AMP_WINDOW_US[1])
    imask = (t_us >= INTEGRAL_WINDOW_US[0]) & (t_us <= INTEGRAL_WINDOW_US[1])
    if bmask.sum() < 3 or amask.sum() < 3 or imask.sum() < 3:
        raise RuntimeError("baseline / amplitude / integral window has too few samples")

    p0_all = ch0[:, bmask].mean(axis=1)
    p1_all = ch1[:, bmask].mean(axis=1)
    if select_events is None:
        select_events = np.arange(len(ch0))
    select_events = np.asarray(select_events, int)
    if len(select_events) == 0:
        raise RuntimeError("no selected events")

    p0, p1 = p0_all[select_events], p1_all[select_events]
    d0 = ch0[select_events] - p0[:, None]
    d1 = ch1[select_events] - p1[:, None]
    m0, m1 = d0.mean(axis=0), d1.mean(axis=0)
    a2d = np.hypot(m0, m1)  # avoids noise bias of mean(hypot(d0,d1))

    cand = np.where(amask)[0]
    ipk = int(cand[np.argmax(a2d[amask])])
    h = float(a2d[ipk])
    integ = float(np.trapezoid(a2d[imask], t_us[imask]))

    # Event-level peak distributions for robust median / IQR error bars.
    a2d_event = np.hypot(d0, d1)
    hp_event = np.max(a2d_event[:, amask], axis=1)
    hm, h25, h75 = med_iqr(hp_event)

    vec = complex(m0[ipk], m1[ipk])
    return {
        "n_events": len(select_events), "t_us": t_us,
        "ped0": p0, "ped1": p1, "zped": p0 + 1j * p1,
        "m0": m0, "m1": m1, "a2d": a2d,
        "ipk": ipk, "tpk_us": float(t_us[ipk]),
        "h_mean": h, "integral": integ, "tau_eff": integ / h if h > 0 else np.nan,
        "vec": vec, "angle_raw_deg": float(np.degrees(np.angle(vec))),
        "h_event_med": hm, "h_event_q25": h25, "h_event_q75": h75,
        "ch0pk": float(m0[ipk]), "ch1pk": float(m1[ipk]),
    }


# =============================================================================
# Raw ADC IQ sweep -> canonical S21 transform
# =============================================================================
def col(names, options):
    low = {str(n).lower(): n for n in names}
    for o in options:
        if o.lower() in low:
            return low[o.lower()]
    return None


def to_ghz(f):
    f = np.asarray(f, float)
    med = np.nanmedian(abs(f))
    if med > 1e7: return f / 1e9
    if med > 1e4: return f / 1e3
    return f


def load_sweep(path):
    path = Path(path).expanduser()
    fn = ["freq_ghz", "frequency_ghz", "f_ghz", "freq", "frequency", "f"]
    rn = ["ch0", "i", "I", "real", "re", "s21_re", "s21_real"]
    qn = ["ch1", "q", "Q", "imag", "im", "s21_im", "s21_imag"]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        fc, rc, qc = col(df.columns, fn), col(df.columns, rn), col(df.columns, qn)
        if None in (fc, rc, qc):
            raise KeyError(f"Need frequency + two IQ columns. Found: {list(df.columns)}")
        f = df[fc].to_numpy(float); z = df[rc].to_numpy(float) + 1j * df[qc].to_numpy(float)
    elif path.suffix.lower() == ".npz":
        d = np.load(path)
        fc, rc, qc = col(d.files, fn), col(d.files, rn), col(d.files, qn)
        if None in (fc, rc, qc):
            raise KeyError(f"Need frequency + two IQ keys. Found: {list(d.files)}")
        f = np.asarray(d[fc], float).ravel(); z = np.asarray(d[rc], float).ravel() + 1j * np.asarray(d[qc], float).ravel()
    else:
        raise ValueError("RAW_IQ_SWEEP_FILE must be .csv or .npz")
    f = to_ghz(f)
    good = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag)
    order = np.argsort(f[good])
    return f[good][order], z[good][order]


def delay(z, f, tau_ns):
    z = np.asarray(z, complex)
    return z if tau_ns is None else z * np.exp(2j * np.pi * np.asarray(f, float) * tau_ns)


def circle_fit(z):
    x, y = z.real, z.imag
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x*x + y*y)
    aa, bb, cc = np.linalg.lstsq(A, b, rcond=None)[0]
    center = -aa/2 + 1j*(-bb/2)
    r2 = (aa*aa + bb*bb)/4 - cc
    if r2 <= 0: raise RuntimeError("circle fit failed")
    r = float(np.sqrt(r2))
    residual = abs(z-center) - r
    return center, r, residual


def make_transform(fsw, zraw, reference_pulse):
    zdel = delay(zraw, fsw, SWEEP_CABLE_DELAY_NS)
    mask = np.ones(len(fsw), bool)
    if SWEEP_FIT_RANGE_GHZ is not None:
        lo, hi = SWEEP_FIT_RANGE_GHZ
        mask = (fsw >= lo) & (fsw <= hi)
    center, radius, res = circle_fit(zdel[mask])
    ires = int(np.argmin(abs(fsw - FR_REFERENCE_GHZ)))
    ures = (zdel[ires] - center) / radius
    rot = -1 / normz(ures)  # resonance -> unit-circle left point (-1,0)
    p = rot * delay(reference_pulse, F_TONE_REFERENCE_GHZ, SWEEP_CABLE_DELAY_NS) / radius
    conjugate = bool(p.imag < 0)  # choose heating pulse as +Q

    def apply(z, f):
        u = rot * (delay(z, f, SWEEP_CABLE_DELAY_NS) - center) / radius
        if conjugate: u = np.conj(u)
        return (1-D_NOTCH/2) + (D_NOTCH/2)*u

    def apply_vec(dz, f):
        u = rot * delay(dz, f, SWEEP_CABLE_DELAY_NS) / radius
        if conjugate: u = np.conj(u)
        return (D_NOTCH/2)*u

    zcan = apply(zraw, fsw)
    meta = {
        "center": center, "radius": radius, "res_rms": float(np.sqrt(np.mean(res[mask]**2))),
        "ires": ires, "f_res_used": float(fsw[ires]), "rot": rot, "conjugate": conjugate,
        "fsw": fsw, "zraw": zraw, "zcan": zcan,
    }
    return apply, apply_vec, meta


# =============================================================================
# Task 1
# =============================================================================
def temp_summary(feat, apply=None):
    n = feat["n_events"]
    idx = np.arange(n)
    bins = (idx % EVENTS_PER_TEMP_CYCLE) // EVENTS_PER_TEMP_BIN
    zraw = feat["zped"]
    if apply is None:
        origin = np.median(zraw.real) + 1j*np.median(zraw.imag)
        rot = 1j / normz(feat["vec"])
        zplot = rot * (zraw-origin)
        mode = "relative"
    else:
        zplot = apply(zraw, F_TONE_REFERENCE_GHZ)
        mode = "canonical"

    rows = []
    for b in range(N_TEMP_BINS):
        ii = np.where(bins == b)[0]
        a,b1,c = med_iqr(zraw[ii].real); d,e,f = med_iqr(zraw[ii].imag)
        g,h,i = med_iqr(zplot[ii].real); j,k,l = med_iqr(zplot[ii].imag)
        rows.append({
            "temp_phase_bin": b, "n_events": len(ii),
            "raw_ch0_median_V": a, "raw_ch0_q25_V": b1, "raw_ch0_q75_V": c,
            "raw_ch1_median_V": d, "raw_ch1_q25_V": e, "raw_ch1_q75_V": f,
            "I_median": g, "I_q25": h, "I_q75": i,
            "Q_median": j, "Q_q25": k, "Q_q75": l,
        })
    df = pd.DataFrame(rows)
    if mode == "canonical":
        z = df.I_median.to_numpy()+1j*df.Q_median.to_numpy()
        df["distance_to_resonance"] = abs(z - (1-D_NOTCH))
    return df, mode


def arrows(ax, x, y):
    for ii in range(len(x)):
        jj = (ii+1) % len(x)
        ax.annotate("", xy=(x[jj],y[jj]), xytext=(x[ii],y[ii]), arrowprops=dict(arrowstyle="->", alpha=.65))


def plot_task1(df, mode, sweep_meta, out):
    fig, axa = plt.subplots(2,2, figsize=(14,10), constrained_layout=True)
    b = df.temp_phase_bin.to_numpy()
    # raw
    ax = axa[0,0]; x=df.raw_ch0_median_V.to_numpy()*1e3; y=df.raw_ch1_median_V.to_numpy()*1e3
    ax.scatter(x,y,c=b,s=65); arrows(ax,x,y)
    for q in range(len(b)): ax.annotate(str(b[q]),(x[q],y[q]),xytext=(4,4),textcoords="offset points")
    ax.set(title="Raw ADC pedestal IQ (10-bin median)",xlabel="ch0 [mV]",ylabel="ch1 [mV]"); ax.grid(); ax.set_aspect("equal","box")
    # canonical/relative
    ax=axa[0,1]; x=df.I_median.to_numpy(); y=df.Q_median.to_numpy()
    if mode=="canonical":
        th=np.linspace(0,2*np.pi,800); c=1-D_NOTCH/2; r=D_NOTCH/2
        z=c+r*np.exp(1j*th); ax.plot(z.real,z.imag,label="canonical notch circle")
        zs=sweep_meta["zcan"]; ax.plot(zs.real,zs.imag,alpha=.65,label="raw ADC sweep calibrated")
        ax.scatter([1-D_NOTCH,1],[0,0],marker="x",s=95,label=r"resonance $(1-d,0)$ / off-res. $(1,0)$")
        ib=int(df.distance_to_resonance.idxmin()); ax.scatter([x[ib]],[y[ib]],marker="*",s=220,label=f"closest resonance bin {ib}")
        title="Canonical S21 pedestal IQ"
        xl=r"I = Re($S_{21}$)"; yl=r"Q = Im($S_{21}$)"
    else:
        ax.axhline(0); ax.axvline(0); title="Relative phase-aligned IQ (no raw sweep)"; xl="relative I"; yl="relative Q"
    ax.scatter(x,y,c=b,s=65,zorder=3); arrows(ax,x,y)
    for q in range(len(b)): ax.annotate(str(b[q]),(x[q],y[q]),xytext=(4,4),textcoords="offset points")
    ax.set(title=title,xlabel=xl,ylabel=yl); ax.grid(); ax.set_aspect("equal","box"); ax.legend(fontsize=8)
    # components
    ax=axa[1,0]
    ax.plot(b,x,marker="o",label="I median"); ax.plot(b,y,marker="s",label="Q median")
    ax.set(title="Pedestal components across 1 Hz cycle",xlabel="temperature phase bin",ylabel="canonical S21" if mode=="canonical" else "relative IQ"); ax.grid(); ax.legend()
    # resonance distance or relative phase
    ax=axa[1,1]
    if mode=="canonical":
        ax.plot(b,df.distance_to_resonance,marker="o"); ax.set(title="Distance to canonical resonance point",ylabel=r"$|S_{21}-(1-d)|$")
    else:
        ax.plot(b,np.unwrap(np.angle(x+1j*y))*180/np.pi,marker="o"); ax.set(title="Relative trajectory angle",ylabel="angle [deg]")
    ax.set_xlabel("temperature phase bin"); ax.grid()
    fig.suptitle(f"Task 1: 1 Hz pedestal motion; readout tone = {F_TONE_REFERENCE_GHZ:.6f} GHz",fontsize=14)
    fig.savefig(out,dpi=DPI); plt.close(fig)


# =============================================================================
# Tasks 2/3 batch analysis
# =============================================================================
def analyze_one(info, apply_vec=None, phase_select=None):
    t,c0,c1,used=load_events(info)
    sel=None
    if phase_select is not None:
        ev=np.arange(len(c0)); sel=np.where(((ev%EVENTS_PER_TEMP_CYCLE)//EVENTS_PER_TEMP_BIN)==phase_select)[0]
    ft=features(t,c0,c1,sel)
    row={
        **{k:info[k] for k in ["folder","freq_ghz","z_mm","x_mm","tag"]},
        "n_events_used":ft["n_events"], "peak_time_us":ft["tpk_us"],
        "peak_a2d_mean_mV":ft["h_mean"]*1e3,
        "peak_a2d_median_mV":ft["h_event_med"]*1e3,
        "peak_a2d_q25_mV":ft["h_event_q25"]*1e3,
        "peak_a2d_q75_mV":ft["h_event_q75"]*1e3,
        "integral_a2d_mVus":ft["integral"]*1e3, "tau_eff_us":ft["tau_eff"],
        "ch0_at_peak_mV":ft["ch0pk"]*1e3, "ch1_at_peak_mV":ft["ch1pk"]*1e3,
        "raw_pulse_angle_deg":ft["angle_raw_deg"],
    }
    if apply_vec is not None:
        dz=apply_vec(ft["vec"],info["freq_ghz"])
        row.update({"canonical_dI":dz.real,"canonical_dQ":dz.imag,"canonical_angle_deg":float(np.degrees(np.angle(dz))),"canonical_length":abs(dz)})
    return row,ft,used


def plot_task2(df, fmap, out1, out2):
    df=df.sort_values("freq_ghz").reset_index(drop=True); f=df.freq_ghz.to_numpy()
    fig,axs=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for ax,key,title,ylab in [
        (axs[0,0],"m0","raw ch0 waveform",r"$\langle\Delta$ch0$\rangle$ [mV]"),
        (axs[0,1],"m1","raw ch1 waveform",r"$\langle\Delta$ch1$\rangle$ [mV]"),
        (axs[1,0],"a2d",r"2D waveform $A_{2D}$",r"$A_{2D}$ [mV]"),
    ]:
        for _,r in df.iterrows():
            ft=fmap[r.folder]; mask=(ft["t_us"]>=-.3)&(ft["t_us"]<=2.0)
            ax.plot(ft["t_us"][mask],ft[key][mask]*1e3,label=f"{r.freq_ghz:.3f} GHz")
        ax.set(title=title,xlabel="time [us]",ylabel=ylab); ax.grid(); ax.legend(fontsize=7,ncols=2)
    ax=axs[1,1]; y=df.peak_a2d_median_mV.to_numpy(); lo=y-df.peak_a2d_q25_mV.to_numpy(); hi=df.peak_a2d_q75_mV.to_numpy()-y
    ax.errorbar(f,y,yerr=[lo,hi],marker="o",capsize=3,label=r"event $A_{2D,peak}$ median ± IQR")
    ax.set(xlabel="readout tone [GHz]",ylabel=r"$A_{2D,peak}$ [mV]"); ax.grid()
    axr=ax.twinx(); axr.plot(f,df.tau_eff_us,marker="s",label=r"$\tau_{eff}$"); axr.set_ylabel(r"$\tau_{eff}$ [us]")
    h,l=ax.get_legend_handles_labels(); h2,l2=axr.get_legend_handles_labels(); ax.legend(h+h2,l+l2,fontsize=8)
    fig.suptitle("Task 2: center illumination — RF frequency dependence",fontsize=14); fig.savefig(out1,dpi=DPI); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axs[0].plot(f,df.ch0_at_peak_mV,marker="o",label="raw ch0"); axs[0].plot(f,df.ch1_at_peak_mV,marker="s",label="raw ch1"); axs[0].axhline(0); axs[0].grid(); axs[0].legend(); axs[0].set(title="Raw pulse components at A2D peak",xlabel="readout tone [GHz]",ylabel="component [mV]")
    axs[1].plot(f,df.raw_pulse_angle_deg,marker="o",label="raw pulse-vector angle"); axs[1].axhline(90,ls="--"); axs[1].axhline(-90,ls="--"); axs[1].grid(); axs[1].legend(); axs[1].set(title="Pulse-vector rotation",xlabel="readout tone [GHz]",ylabel="angle [deg]")
    fig.savefig(out2,dpi=DPI); plt.close(fig)


def plot_task3(df,fmap,out1,out2):
    xs=df[np.isclose(df.z_mm,CENTER_Z_MM)].sort_values("x_mm")
    zs=df[np.isclose(df.x_mm,CENTER_X_MM)].sort_values("z_mm")
    fig,axs=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    series=[(xs,"x_mm",f"x scan (z={CENTER_Z_MM:g} mm)"),(zs,"z_mm",f"z scan (x={CENTER_X_MM:g} mm)")]
    for dd,var,label in series:
        if len(dd)==0: continue
        y=dd.peak_a2d_median_mV.to_numpy(); lo=y-dd.peak_a2d_q25_mV.to_numpy(); hi=dd.peak_a2d_q75_mV.to_numpy()-y
        axs[0,0].errorbar(dd[var],y,yerr=[lo,hi],marker="o",capsize=3,label=label)
        axs[0,1].plot(dd[var],dd.tau_eff_us,marker="o",label=label)
        axs[1,0].plot(dd[var],dd.integral_a2d_mVus,marker="o",label=label)
        axs[1,1].plot(dd[var],dd.raw_pulse_angle_deg,marker="o",label=label)
    labels=[("Position dependence of A2D peak","laser position [mm]",r"$A_{2D,peak}$ [mV]"),("Position dependence of tau_eff","laser position [mm]",r"$\tau_{eff}$ [us]"),("Position dependence of integrated response","laser position [mm]",r"$\int A_{2D}dt$ [mV us]"),("Position dependence of ch0/ch1 mixture","laser position [mm]","raw pulse angle [deg]")]
    for ax,(title,xl,yl) in zip(axs.ravel(),labels): ax.set(title=title,xlabel=xl,ylabel=yl); ax.grid(); ax.legend()
    fig.suptitle(r"Task 3: position scan; $A_{2D}=\sqrt{<Δch0>^2+<Δch1>^2}$",fontsize=13); fig.savefig(out1,dpi=DPI); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ax,dd,var,title in [(axs[0],xs,"x_mm",f"x scan at z={CENTER_Z_MM:g} mm"),(axs[1],zs,"z_mm",f"z scan at x={CENTER_X_MM:g} mm")]:
        for _,r in dd.iterrows():
            ft=fmap[r.folder]; mask=(ft["t_us"]>=-.3)&(ft["t_us"]<=2.0)
            ax.plot(ft["t_us"][mask],ft["a2d"][mask]*1e3,label=f"{var[0]}={r[var]:.1f} mm")
        ax.set(title=title,xlabel="time [us]",ylabel=r"$A_{2D}$ [mV]"); ax.grid(); ax.legend(fontsize=7,ncols=2)
    fig.suptitle("Task 3: pedestal-subtracted 2D pulse shapes",fontsize=13); fig.savefig(out2,dpi=DPI); plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    print("Output:",OUT_DIR)
    ms=discover_measurements()
    ref=[m for m in ms if m["folder"]==REFERENCE_DIR_NAME]
    if not ref: raise RuntimeError(f"Reference directory not found: {REFERENCE_DIR_NAME}")
    ref=ref[0]
    print("\nTask 1 reference:",ref["path"])
    t,c0,c1,files=load_events(ref)
    rf=features(t,c0,c1)
    print("reference raw pulse vector [mV] =",rf["vec"]*1e3,"peak time [us] =",rf["tpk_us"])

    apply=apply_vec=None; smeta=None
    if RAW_IQ_SWEEP_FILE is not None:
        fsw,zsw=load_sweep(RAW_IQ_SWEEP_FILE)
        apply,apply_vec,smeta=make_transform(fsw,zsw,rf["vec"])
        print("canonical calibration: center=",smeta["center"],"radius=",smeta["radius"],"fres used=",smeta["f_res_used"],"conjugate=",smeta["conjugate"])
    else:
        print("RAW_IQ_SWEEP_FILE=None -> Task 1 uses relative IQ only; no (1-d,0) claim.")

    phase_df,mode=temp_summary(rf,apply)
    if mode=="canonical":
        resonance_phase=int(phase_df.distance_to_resonance.idxmin())
        # map to nearest sweep point as a f_r-only diagnostic, not a fit
        zph=phase_df.I_median.to_numpy()+1j*phase_df.Q_median.to_numpy(); zs=smeta["zcan"]; fs=smeta["fsw"]
        ii=np.argmin(abs(zph[:,None]-zs[None,:]),axis=1); feq=fs[ii]; frest=F_TONE_REFERENCE_GHZ*FR_REFERENCE_GHZ/feq
        phase_df["nearest_sweep_freq_ghz"]=feq; phase_df["fr_estimate_fronly_ghz"]=frest; phase_df["fr_only_distance"]=abs(zph-zs[ii])
        print("closest canonical resonance phase bin:",resonance_phase)
    else: resonance_phase=None
    phase_df.to_csv(OUT_DIR/"01_temperature_phase_summary.csv",index=False)
    plot_task1(phase_df,mode,smeta,OUT_DIR/"01_temperature_pedestal_vs_s21.png")

    # Task 2
    centers=[m for m in ms if close(m["z_mm"],CENTER_Z_MM) and close(m["x_mm"],CENTER_X_MM) and CENTER_FREQ_RANGE_GHZ[0]<=m["freq_ghz"]<=CENTER_FREQ_RANGE_GHZ[1]]
    if not centers: raise RuntimeError("No center RF-scan folders found")
    print("\nTask 2 center RF scan:")
    rows=[]; fmap={}; usedmap={}
    for m in centers:
        print(" ",m["folder"])
        row,ft,used=analyze_one(m,apply_vec)
        rows.append(row); fmap[m["folder"]]=ft; usedmap[m["folder"]]=[str(x) for x in used]
    cdf=pd.DataFrame(rows).sort_values("freq_ghz").reset_index(drop=True)
    cdf.to_csv(OUT_DIR/"02_center_frequency_scan_summary.csv",index=False)
    plot_task2(cdf,fmap,OUT_DIR/"02_center_frequency_scan_waveforms_metrics.png",OUT_DIR/"02_center_frequency_scan_vectors.png")

    if POSITION_TONE_GHZ=="auto_peak_2d":
        ptone=float(cdf.loc[cdf.peak_a2d_median_mV.idxmax(),"freq_ghz"]); pnote="auto_peak_2d"
    else: ptone=float(POSITION_TONE_GHZ); pnote="configured"
    print("\nTask 3 selected position tone:",ptone,"GHz (",pnote,")")
    pms=[m for m in ms if close(m["freq_ghz"],ptone,atol=1e-3)]
    if not pms: raise RuntimeError("No position scan folders at selected tone")
    phase_sel=resonance_phase if (POSITION_TEMP_SELECTION=="reference_resonance_phase" and resonance_phase is not None) else None
    print("Task 3 phase selection:",phase_sel if phase_sel is not None else "all events")
    rows=[]; pfmap={}; pused={}
    for m in pms:
        print(" ",m["folder"])
        row,ft,used=analyze_one(m,apply_vec,phase_sel)
        rows.append(row); pfmap[m["folder"]]=ft; pused[m["folder"]]=[str(x) for x in used]
    pdf=pd.DataFrame(rows).sort_values(["z_mm","x_mm"]).reset_index(drop=True)
    pdf.to_csv(OUT_DIR/"03_position_scan_summary.csv",index=False)
    plot_task3(pdf,pfmap,OUT_DIR/"03_position_scan_metrics.png",OUT_DIR/"03_position_scan_a2d_waveforms.png")

    meta={
        "mode":mode,"raw_iq_sweep_file":str(RAW_IQ_SWEEP_FILE) if RAW_IQ_SWEEP_FILE is not None else None,
        "d_notch":D_NOTCH,"reference_dir":str(ref["path"]),"f_tone_reference_ghz":F_TONE_REFERENCE_GHZ,
        "fr_reference_ghz":FR_REFERENCE_GHZ,"resonance_phase_bin":resonance_phase,
        "position_tone_ghz":ptone,"position_tone_selection":pnote,"position_temp_selection":POSITION_TEMP_SELECTION,
        "reference_files":[str(x) for x in files],"center_used_files":usedmap,"position_used_files":pused,
    }
    if smeta is not None:
        meta.update({"circle_center_re":smeta["center"].real,"circle_center_im":smeta["center"].imag,"circle_radius":smeta["radius"],"circle_residual_rms":smeta["res_rms"],"sweep_resonance_used_ghz":smeta["f_res_used"],"conjugated":smeta["conjugate"]})
    (OUT_DIR/"run_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nSaved:")
    for p in sorted(OUT_DIR.iterdir()): print(" ",p)

if __name__=="__main__":
    main()
