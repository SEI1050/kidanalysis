import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from pathlib import Path
import re

# ======================
# arguments
# ======================
if len(sys.argv) < 2:
    print("usage: python Rez_waveform_fit.py file.npz [rebin] [out_dir]")
    sys.exit()

filename = Path(sys.argv[1]).expanduser().resolve()

if len(sys.argv) >= 3:
    rebin = int(sys.argv[2])
else:
    rebin = 1

# 第3引数があれば、そこにpdf/csvを保存する
# なければ従来通り npz と同じフォルダに保存
if len(sys.argv) >= 4:
    out_dir = Path(sys.argv[3]).expanduser().resolve()
else:
    out_dir = filename.parent

out_dir.mkdir(parents=True, exist_ok=True)

print("filename =", filename)
print("rebin =", rebin)
print("out_dir =", out_dir)

# ======================
# fit function
# ======================
def funcdex(X_, tau_, rise_):
    tmax = np.log(rise_ / tau_) / (1 / tau_ - 1 / rise_)
    return np.where(
        X_ >= 0,
        (np.exp(-X_ / tau_) - np.exp(-X_ / rise_))
        / np.abs(np.exp(-tmax / tau_) - np.exp(-tmax / rise_)),
        0,
    )

def funcfit(X_, t0_, k_, tau_, rise_, ped_):
    return k_ * funcdex(X_ - t0_, tau_, rise_) + ped_

def rebin_array(arr, rebin):
    if rebin <= 1:
        return arr

    n = len(arr)
    n_use = n // rebin * rebin

    return arr[:n_use].reshape(-1, rebin).mean(axis=1)
    
def time_mask(t_ns, tmin=None, tmax=None):
    mask = np.ones_like(t_ns, dtype=bool)

    if tmin is not None:
        mask &= t_ns >= tmin

    if tmax is not None:
        mask &= t_ns <= tmax

    return mask


def get_baseline_mask(t_ns, ref_position,
                      base_tmin=None, base_tmax=-100):
    """
    基本は t < base_tmax の領域をbaselineにする。
    十分な点数がなければ、ref_positionより前を使う。
    """

    mask = time_mask(t_ns, base_tmin, base_tmax)

    if np.count_nonzero(mask) < 5:
        nbase = int(len(t_ns) * ref_position / 100)
        nbase = max(nbase, 5)

        mask = np.zeros_like(t_ns, dtype=bool)
        mask[:nbase] = True

    return mask


def make_delta_z(wave0, wave1, t_ns, ref_position,
                 base_tmin=None, base_tmax=-100):
    """
    ch0, ch1から baseline を引いた複素波形 Δz を作る。
    """

    base_mask = get_baseline_mask(
        t_ns,
        ref_position,
        base_tmin=base_tmin,
        base_tmax=base_tmax,
    )

    ped0 = wave0[base_mask].mean()
    ped1 = wave1[base_mask].mean()

    dz = (wave0 - ped0) + 1j * (wave1 - ped1)

    return dz, ped0, ped1, base_mask


def estimate_event_direction(dz, t_ns, base_mask,
                             search_tmin=-50, search_tmax=400,
                             late_tmin=1000, late_tmax=1600):
    """
    1イベントごとに、IQ平面上の最大変位方向を推定する。
    ただし、この方向はテンプレート作成用に使う。
    """

    search_mask = time_mask(t_ns, search_tmin, search_tmax)

    if np.count_nonzero(search_mask) < 3:
        search_mask = np.ones_like(t_ns, dtype=bool)

    search_indices = np.where(search_mask)[0]

    idx_local = np.argmax(np.abs(dz[search_mask]))
    idx_peak = search_indices[idx_local]

    z_peak = dz[idx_peak]
    amp = np.abs(z_peak)

    if amp > 0 and np.isfinite(amp):
        u = z_peak / amp
    else:
        u = np.nan + 1j * np.nan

    # baseline noise
    noise_rms = np.sqrt(
        np.var(dz[base_mask].real) + np.var(dz[base_mask].imag)
    )

    if not np.isfinite(noise_rms) or noise_rms <= 0:
        noise_rms = 1e-30

    snr = amp / noise_rms

    # late noise
    late_mask = time_mask(t_ns, late_tmin, late_tmax)

    if np.count_nonzero(late_mask) >= 5:
        late_rms = np.sqrt(
            np.var(dz[late_mask].real) + np.var(dz[late_mask].imag)
        )
    else:
        late_rms = noise_rms

    late_rms_ratio = late_rms / noise_rms

    return {
        "u": u,
        "idx_peak": idx_peak,
        "t_peak_ns": t_ns[idx_peak],
        "amp": amp,
        "noise_rms": noise_rms,
        "snr": snr,
        "late_rms": late_rms,
        "late_rms_ratio": late_rms_ratio,
    }


def estimate_template_direction(data, t_ns, ref_position, rebin,
                                min_snr=5.0,
                                max_late_rms_ratio=5.0):
    """
    全イベントから平均的なIQ信号方向 u_template を作る。
    """

    rows = []

    nwf = data["ch1"].shape[0]

    for idx in range(nwf):
        wave0 = rebin_array(data["ch0"][idx], rebin)
        wave1 = rebin_array(data["ch1"][idx], rebin)

        dz, ped0, ped1, base_mask = make_delta_z(
            wave0,
            wave1,
            t_ns,
            ref_position,
            base_tmin=BASE_TMIN_NS,
            base_tmax=BASE_TMAX_NS,
        )

        info = estimate_event_direction(
            dz,
            t_ns,
            base_mask,
            search_tmin=SEARCH_TMIN_NS,
            search_tmax=SEARCH_TMAX_NS,
            late_tmin=LATE_TMIN_NS,
            late_tmax=LATE_TMAX_NS,
        )

        rows.append({
            "idx": idx,
            "u_re": np.real(info["u"]),
            "u_im": np.imag(info["u"]),
            "amp": info["amp"],
            "snr": info["snr"],
            "late_rms_ratio": info["late_rms_ratio"],
            "idx_peak": info["idx_peak"],
            "t_peak_ns": info["t_peak_ns"],
        })

    df_dir = pd.DataFrame(rows)

    # まず基本的なquality cut
    good = (
        np.isfinite(df_dir["u_re"])
        & np.isfinite(df_dir["u_im"])
        & np.isfinite(df_dir["amp"])
        & (df_dir["amp"] > 0)
        & (df_dir["snr"] >= min_snr)
        & (df_dir["late_rms_ratio"] <= max_late_rms_ratio)
    )

    # 極端に小さい/大きいイベントをテンプレート作成から外す
    if good.sum() >= 10:
        amp_good = df_dir.loc[good, "amp"]

        amp_low = amp_good.quantile(0.10)
        amp_high = amp_good.quantile(0.95)

        good &= (
            (df_dir["amp"] >= amp_low)
            & (df_dir["amp"] <= amp_high)
        )

    df_dir["template_good"] = good

    if good.sum() == 0:
        print("WARNING: no good events for template direction. Use u_template = 1+0j")
        u_template = 1.0 + 0.0j
        coherence = 0.0
        return u_template, coherence, df_dir

    u_each = df_dir.loc[good, "u_re"].values + 1j * df_dir.loc[good, "u_im"].values
    weights = df_dir.loc[good, "amp"].values

    # ======================
    # sign alignment
    # ======================
    # 最初は振幅最大のイベントを基準方向にする
    iref = np.argmax(weights)
    u_ref = u_each[iref]

    if np.abs(u_ref) == 0 or not np.isfinite(np.abs(u_ref)):
        u_ref = 1.0 + 0.0j
    else:
        u_ref = u_ref / np.abs(u_ref)

    # u と -u を同じ軸として扱うため、基準方向と逆向きなら反転する
    dot = np.real(u_each * np.conj(u_ref))
    flip = dot < 0

    u_each_aligned = u_each.copy()
    u_each_aligned[flip] *= -1

    # 反転後に平均
    u_mean = np.sum(weights * u_each_aligned)

    if np.abs(u_mean) == 0 or not np.isfinite(np.abs(u_mean)):
        print("WARNING: template direction average failed. Use u_template = 1+0j")
        u_template = 1.0 + 0.0j
        coherence = 0.0
    else:
        u_template = u_mean / np.abs(u_mean)
        coherence = np.abs(u_mean) / np.sum(weights)

    # 参考用に、反転されたイベントを記録
    df_dir.loc[good, "direction_flipped"] = flip
    df_dir.loc[~good, "direction_flipped"] = False

    return u_template, coherence, df_dir


def make_iq_projected_wave(wave0, wave1, t_ns, ref_position,
                           u_template=None):
    """
    u_template が与えられた場合は、その共通方向へ射影する。
    u_template が None の場合は、イベントごとの最大変位方向を使う。
    """

    dz, ped0, ped1, base_mask = make_delta_z(
        wave0,
        wave1,
        t_ns,
        ref_position,
        base_tmin=BASE_TMIN_NS,
        base_tmax=BASE_TMAX_NS,
    )

    info = estimate_event_direction(
        dz,
        t_ns,
        base_mask,
        search_tmin=SEARCH_TMIN_NS,
        search_tmax=SEARCH_TMAX_NS,
        late_tmin=LATE_TMIN_NS,
        late_tmax=LATE_TMAX_NS,
    )

    if u_template is None:
        u = info["u"]
    else:
        u = u_template

    if not np.isfinite(np.real(u)) or not np.isfinite(np.imag(u)) or np.abs(u) == 0:
        u = 1.0 + 0.0j

    u = u / np.abs(u)

    wave_iq = np.real(dz * np.conj(u))
    ped_iq = wave_iq[base_mask].mean()
    
    search_mask = time_mask(t_ns, SEARCH_TMIN_NS, SEARCH_TMAX_NS)

    if np.count_nonzero(search_mask) > 0:
        k_pos = wave_iq[search_mask].max() - ped_iq
        k_neg = ped_iq - wave_iq[search_mask].min()

        if k_neg > k_pos:
            wave_iq *= -1
            u *= -1
            ped_iq *= -1


    # template方向と逆向きに出たイベントは、fitしやすいようにflag用情報だけ残す
    search_mask = time_mask(t_ns, SEARCH_TMIN_NS, SEARCH_TMAX_NS)

    if np.count_nonzero(search_mask) > 0:
        k_guess = wave_iq[search_mask].max() - ped_iq
        k_neg = ped_iq - wave_iq[search_mask].min()
    else:
        k_guess = wave_iq.max() - ped_iq
        k_neg = ped_iq - wave_iq.min()

    return wave_iq, ped_iq, ped0, ped1, u, info, k_guess, k_neg

# ======================
# load data
# ======================
data = np.load(filename, allow_pickle=True)

sample_rate = data["sample_rate"]
npts = data["npts"]
ref_position = data["ref_position"]

print("number of points =", npts)
print("number of waveforms =", data["ch0"].shape[0], data["ch1"].shape[0])
print(sample_rate, npts, ref_position)

tbin = (np.arange(npts) - npts * ref_position / 100) / sample_rate
tbin_fit = rebin_array(tbin, rebin)

nwf = data["ch1"].shape[0]
# ======================
# IQ projection settings
# ======================
BASE_TMIN_NS = None
BASE_TMAX_NS = -100

SEARCH_TMIN_NS = -50
SEARCH_TMAX_NS = 400

FIT_TMIN_NS = -50
FIT_TMAX_NS = 800

LATE_TMIN_NS = 1000
LATE_TMAX_NS = 1600

MIN_TEMPLATE_SNR = 5.0
MAX_LATE_RMS_RATIO = 5.0

USE_TEMPLATE_DIRECTION = True

# ======================
# fit
# ======================
lpopt = []
lmeta = []

t_ns = tbin_fit * 1e9

# ----------------------
# make template direction
# ----------------------
if USE_TEMPLATE_DIRECTION:
    u_template, template_coherence, df_dir = estimate_template_direction(
        data,
        t_ns,
        ref_position,
        rebin,
        min_snr=MIN_TEMPLATE_SNR,
        max_late_rms_ratio=MAX_LATE_RMS_RATIO,
    )
else:
    u_template = None
    template_coherence = np.nan
    df_dir = None

print("u_template =", u_template)
print("template coherence =", template_coherence)

if df_dir is not None:
    print("number of template events =", int(df_dir["template_good"].sum()), "/", len(df_dir))

for idx in range(nwf):
    # waveform rebin
    wave0 = rebin_array(data["ch0"][idx], rebin)
    wave1 = rebin_array(data["ch1"][idx], rebin)

    # ----------------------
    # IQ projection
    # ----------------------
    wave_iq, ped_iq, ped0, ped1, u_used, info, k_guess, k_neg = make_iq_projected_wave(
        wave0,
        wave1,
        t_ns,
        ref_position,
        u_template=u_template,
    )

    # ----------------------
    # fit projected waveform
    # ----------------------
    fit_mask = time_mask(t_ns, FIT_TMIN_NS, FIT_TMAX_NS)

    if np.count_nonzero(fit_mask) < 5:
        fit_mask = np.ones_like(t_ns, dtype=bool)

    # 正方向パルスとしてfitする
    k_iq = k_guess

    if not np.isfinite(k_iq) or k_iq <= 0:
        k_iq = np.abs(wave_iq[fit_mask] - ped_iq).max()

    if not np.isfinite(k_iq) or k_iq <= 0:
        k_iq = 1e-12

    p0 = [0, k_iq, 500, 50, ped_iq]

    bounds = (
        [0, 0.02 * k_iq, 1, 1, -np.inf],
        [500, 10.0 * k_iq, 5000, 500, np.inf],
    )

    try:
        popt, pcov = curve_fit(
            funcfit,
            t_ns[fit_mask],
            wave_iq[fit_mask],
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        fit_status = 1
    except Exception:
        popt = [0, 0, 0, 0, 0]
        fit_status = 0

    lpopt.append(popt)

    if df_dir is not None:
        template_good = bool(df_dir.loc[idx, "template_good"])
    else:
        template_good = False

    # IQ射影に使った情報も保存
    lmeta.append([
        ped0,
        ped1,
        np.real(u_used),
        np.imag(u_used),
        info["idx_peak"],
        info["t_peak_ns"],
        info["amp"],
        info["snr"],
        info["late_rms_ratio"],
        k_guess,
        k_neg,
        fit_status,
        template_good,
        np.real(u_template) if u_template is not None else np.nan,
        np.imag(u_template) if u_template is not None else np.nan,
        template_coherence,
    ])

# ======================
# dataframe
# ======================
lfitpar = ["t0", "k", "tau", "rise", "ped"]

lpopt = np.array(lpopt)
lmeta = np.array(lmeta, dtype=object)

df_fit = pd.DataFrame(
    lpopt,
    columns=[f"iq_{itag}" for itag in lfitpar],
)

df_meta = pd.DataFrame(
    lmeta,
    columns=[
        "ch0_ped_raw",
        "ch1_ped_raw",
        "iq_u_re",
        "iq_u_im",
        "iq_peak_idx",
        "iq_peak_t_ns",
        "iq_peak_amp",
        "iq_peak_snr",
        "iq_late_rms_ratio",
        "iq_k_guess_pos",
        "iq_k_guess_neg",
        "fit_status",
        "template_good",
        "template_u_re",
        "template_u_im",
        "template_coherence",
    ],
)

# 数値列をfloatに変換
for col in df_meta.columns:
    if col not in ["template_good"]:
        df_meta[col] = pd.to_numeric(df_meta[col], errors="coerce")

df_meta["fit_status"] = df_meta["fit_status"].astype(int)
df_meta["template_good"] = df_meta["template_good"].astype(bool)

# 単位変換
# fit時点では t0, tau, rise は ns, k, ped は V
df_fit["iq_t0"] *= 1e-3      # ns -> us
df_fit["iq_k"] *= 1e3        # V -> mV
df_fit["iq_tau"] *= 1e-3     # ns -> us
df_fit["iq_rise"] *= 1e-3    # ns -> us
df_fit["iq_ped"] *= 1e3      # V -> mV

df_meta["ch0_ped_raw"] *= 1e3
df_meta["ch1_ped_raw"] *= 1e3
df_meta["iq_peak_amp"] *= 1e3
df_meta["iq_k_guess_pos"] *= 1e3
df_meta["iq_k_guess_neg"] *= 1e3

df_meta = df_meta.rename(columns={
    "ch0_ped_raw": "ch0_ped_mV",
    "ch1_ped_raw": "ch1_ped_mV",
    "iq_peak_amp": "iq_peak_amp_mV",
    "iq_k_guess_pos": "iq_k_guess_pos_mV",
    "iq_k_guess_neg": "iq_k_guess_neg_mV",
})

df_meta["iq_peak_t_us"] = df_meta["iq_peak_t_ns"] * 1e-3

df_fit = pd.concat([df_fit, df_meta], axis=1)

# 従来の absk 相当
df_fit["absk"] = df_fit["iq_k"]

print(df_fit)

# ======================
# output names
# ======================
def safe_name(s):
    return re.sub(r"[^\w.\-=]+", "_", s)

rebin_tag = f"_rebin{rebin}"

# 親フォルダ名も入れて、同名ファイルの上書きを防ぐ
# 例:
# 5.451GHz_z=7.5mm_x=3.4mm__wf_20260527_172140_49.88Hz_fit_rebin20.pdf
meas_name = safe_name(filename.parent.name)
base_name = f"{meas_name}__{filename.stem}"

resname = out_dir / f"{base_name}_fitres{rebin_tag}.csv"
pdfname = out_dir / f"{base_name}_fit{rebin_tag}.pdf"

# ======================
# pdf
# ======================
pdf1 = PdfPages(pdfname)

ncol = 4
nrow = 4

fig, ax = plt.subplots(figsize=(16, 9), ncols=ncol, nrows=nrow, sharex=True)
ax = ax.flatten()

ch0_range = (data["ch0"].min(), data["ch0"].max())
ch1_range = (data["ch1"].min(), data["ch1"].max())

for idx in range(min(ncol * nrow, nwf)):
    wave0 = rebin_array(data["ch0"][idx], rebin)
    wave1 = rebin_array(data["ch1"][idx], rebin)

    wave_iq, ped_iq, ped0, ped1, u, info, k_guess, k_neg = make_iq_projected_wave(
    wave0,
    wave1,
    t_ns,
    ref_position,
    u_template=u_template,
)

    ax[idx].plot(t_ns, wave_iq, "-", label="IQ projected", c="C2", alpha=0.8)
    ax[idx].plot(t_ns, funcfit(t_ns, *lpopt[idx]), "-", c="k", label="fit")

    ax[idx].axhline(ped_iq, color="gray", ls="--", lw=0.8)
    ax[idx].axvline(info["t_peak_ns"], color="gray", ls=":", lw=0.8)

    ax[idx].axvspan(FIT_TMIN_NS, FIT_TMAX_NS, color="gray", alpha=0.10)

    ax[idx].grid()
    ax[idx].set_title(f"idx={idx}")

for ic in range(ncol):
    ax[ncol * (nrow - 1) + ic].set_xlabel("Time [ns]")

for ir in range(nrow):
    ax[ir * ncol].set_ylabel("IQ projected [V]")

fig.tight_layout()
fig.savefig(pdf1, format="pdf")
plt.close(fig)

# histogram
# histogram
fig, ax = plt.subplots(figsize=(16, 9), nrows=1, ncols=len(lfitpar), sharex=False)

for idx, icol in enumerate(lfitpar):
    col = f"iq_{icol}"

    ax[idx].hist(df_fit[col], bins=50, color="C2")
    ax[idx].set_xlabel(icol)
    ax[idx].grid()

fig.tight_layout()
fig.savefig(pdf1, format="pdf")
plt.close(fig)
# template direction check
if df_dir is not None:
    fig, ax = plt.subplots(figsize=(8, 8))

    u_all = df_dir["u_re"].values + 1j * df_dir["u_im"].values

    ax.scatter(
        df_dir["u_re"],
        df_dir["u_im"],
        s=10,
        alpha=0.3,
        label="all events",
    )

    good = df_dir["template_good"].values

    ax.scatter(
        df_dir.loc[good, "u_re"],
        df_dir.loc[good, "u_im"],
        s=20,
        alpha=0.8,
        label="template events",
    )

    ax.arrow(
        0,
        0,
        np.real(u_template),
        np.imag(u_template),
        width=0.01,
        length_includes_head=True,
        label="u_template",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(u)")
    ax.set_ylabel("Im(u)")
    ax.grid()
    ax.legend()
    ax.set_title(f"Template direction, coherence={template_coherence:.3f}")

    fig.tight_layout()
    fig.savefig(pdf1, format="pdf")
    plt.close(fig)

pdf1.close()

# ======================
# save csv
# ======================
df_fit.to_csv(resname, index=False)

print("saved:", resname)
print("saved:", pdfname)