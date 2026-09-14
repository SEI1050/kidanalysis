import os
import re
import sys
import time
import glob
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit  
import quadpy

import ds_style

VERBOSE = 1


## python3 gakkai/parameter_ckeck.py --gui data.npz

def _first_or_nan(idx_arr):
    """First element of a np.where(...)[0] result, or nan if no crossing was found."""
    return idx_arr[0] if len(idx_arr) > 0 else np.nan

def _first_or_endpoint(idx_arr, endpoint_idx):
    """First crossing index, or the waveform endpoint when no crossing exists."""
    return idx_arr[0] if len(idx_arr) > 0 else endpoint_idx

def _frac_to_float(s):
    """Parse a label like '1', '1/2', '1/400' into a float (1.0, 0.5, 0.0025, ...)."""
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)

def _parse_wf_timestamp(fn_):
    """Parse the 'YYMMDD_HHMMSS' timestamp embedded in a wf_*.npz filename
    (e.g. 'wf_260827_190229_0.60Hz.npz' -> datetime(2026,8,27,19,2,29)).

    Note a run directory named e.g. 'Aug27th' can still contain files whose
    embedded date has rolled over past midnight into the next day (the
    filename's own date always wins over the directory name).

    Returns None if fn_'s basename doesn't match the expected pattern.
    """
    m = re.search(r'wf_(\d{6})_(\d{6})_', os.path.basename(fn_))
    if m is None:
        return None
    return datetime.datetime.strptime(m.group(1) + m.group(2), '%y%m%d%H%M%S')

def _to_datetime(t):
    """Coerce a 'YYMMDD_HHMMSS' string (matching the wf_*.npz filename
    timestamp format, e.g. '260827_190000') or a datetime.datetime into a
    datetime.datetime. None passes through unchanged (unbounded)."""
    if t is None or isinstance(t, datetime.datetime):
        return t
    return datetime.datetime.strptime(t, '%y%m%d_%H%M%S')

def _rebin_waveforms(arr2d, rebin_factor):
    """Rebin a (nwf, npts) waveform array along the time axis by averaging
    groups of `rebin_factor` consecutive samples.

    rebin_factor=1 returns `arr2d` unchanged. If npts is not a multiple of
    rebin_factor, the trailing leftover samples are dropped. Works for
    complex arrays too (e.g. the rotated I/Q waveform).
    """
    rebin_factor = int(rebin_factor)
    if rebin_factor <= 1:
        return arr2d
    nwf, npts = arr2d.shape
    npts_used = (npts // rebin_factor) * rebin_factor
    return arr2d[:, :npts_used].reshape(nwf, npts_used // rebin_factor, rebin_factor).mean(axis=2)

def main():
    return main_realtime_check_gui()


DEFAULT_DISPLAY_PARAMS = [
    'proj_max',
    'proj_max_t',
    'proj_half_max_left',
    'proj_half_max_right',
    'proj_integ',
]

DISPLAY_PARAMETER_CHOICES = [
    'ch0_ped', 'ch0_peak_max', 'ch0_peak_max_t',
    'ch0_half_max_left', 'ch0_half_max_right',
    'ch0_ninth_max_left', 'ch0_ninth_max_right',
    'ch0_e_max_left', 'ch0_e_max_right',
    'ch0_integ', 'ch0_integ_left', 'ch0_integ_right',
    'ch1_ped', 'ch1_peak_max', 'ch1_peak_max_t',
    'ch1_half_max_left', 'ch1_half_max_right',
    'ch1_ninth_max_left', 'ch1_ninth_max_right',
    'ch1_e_max_left', 'ch1_e_max_right',
    'ch1_integ', 'ch1_integ_left', 'ch1_integ_right',
    'proj_max', 'proj_max_t', 'proj_theta',
    'proj_half_max_left', 'proj_half_max_right',
    'proj_ninth_max_left', 'proj_ninth_max_right',
    'proj_e_max_left', 'proj_e_max_right',
    'proj_integ', 'proj_integ_left', 'proj_integ_right',
]


def _format_display_parameter(name, value):
    """Format one parameter for the annotation above an event waveform."""
    if not np.isfinite(value):
        return f'{name}=nan'
    return f'{name}={value:.4g}'


def main_realtime_check_gui(fn_=None):
    """Open a GUI for selecting the input file, parameters, and random seed."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title('Raw event parameter check')
    root.geometry('570x560')

    file_var = tk.StringVar(value=fn_ or '')
    seed_var = tk.StringVar(value='')
    rebin_var = tk.StringVar(value='1')
    alpha_var = tk.BooleanVar(value=False)

    file_frame = tk.Frame(root)
    file_frame.pack(fill='x', padx=12, pady=(12, 6))
    tk.Label(file_frame, text='NPZ file').pack(side='left')
    tk.Entry(file_frame, textvariable=file_var).pack(side='left', fill='x', expand=True, padx=8)

    def choose_file():
        selected_file = filedialog.askopenfilename(
            title='Select waveform data',
            filetypes=[('NPZ files', '*.npz'), ('All files', '*.*')],
        )
        if selected_file:
            file_var.set(selected_file)

    tk.Button(file_frame, text='Browse...', command=choose_file).pack(side='right')

    settings_frame = tk.Frame(root)
    settings_frame.pack(fill='x', padx=12, pady=6)
    tk.Label(settings_frame, text='Seed (blank = random)').pack(side='left')
    tk.Entry(settings_frame, textvariable=seed_var, width=12).pack(side='left', padx=8)
    tk.Label(settings_frame, text='Rebin').pack(side='left', padx=(18, 4))
    tk.Entry(settings_frame, textvariable=rebin_var, width=6).pack(side='left')
    tk.Checkbutton(settings_frame, text='flagAlpha', variable=alpha_var).pack(side='left', padx=12)

    tk.Label(root, text='Display parameters (Ctrl-click / Command-click for multiple selection)').pack(
        anchor='w', padx=12, pady=(12, 4)
    )
    parameter_list = tk.Listbox(root, selectmode=tk.MULTIPLE, height=20, exportselection=False)
    parameter_list.pack(fill='both', expand=True, padx=12)
    for parameter_name in DISPLAY_PARAMETER_CHOICES:
        parameter_list.insert(tk.END, parameter_name)
    for parameter_name in DEFAULT_DISPLAY_PARAMS:
        parameter_list.selection_set(DISPLAY_PARAMETER_CHOICES.index(parameter_name))

    def run_check():
        selected_file = file_var.get().strip()
        if not selected_file:
            messagebox.showerror('Missing file', 'Please select an NPZ file.')
            return
        selected_params = [parameter_list.get(index) for index in parameter_list.curselection()]
        if not selected_params:
            messagebox.showerror('Missing parameters', 'Select at least one parameter.')
            return
        try:
            rebin_factor = int(rebin_var.get())
            seed = int(seed_var.get()) if seed_var.get().strip() else None
            if rebin_factor < 1:
                raise ValueError('Rebin must be at least 1.')
        except ValueError as error:
            messagebox.showerror('Invalid setting', str(error))
            return
        root.destroy()
        main_realtime_check(
            selected_file,
            display_params=selected_params,
            seed=seed,
            rebin_factor=rebin_factor,
            flagAlpha=alpha_var.get(),
        )

    tk.Button(root, text='Run check', command=run_check).pack(pady=12)
    root.mainloop()


def main_realtime_check(fn_, n_events=10, display_params=None, seed=None,
                        rebin_factor=1, flagAlpha=False, output='raw_event_check.png'):
    """Display randomly selected raw events and their calculated parameters."""
    dfres, data_corr, vc_proj, tbin, sample_rate, nwf = analyze_waveforms(
        fn_, rebin_factor=rebin_factor, flagAlpha=flagAlpha
    )
    if display_params is None:
        display_params = DEFAULT_DISPLAY_PARAMS
    display_params = list(display_params)
    missing = [name for name in display_params if name not in dfres.columns]
    if missing:
        available = ', '.join(dfres.columns)
        raise ValueError(f'Unknown parameter(s): {missing}. Available: {available}')
    if not display_params:
        raise ValueError('display_params must contain at least one parameter')

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(nwf, size=min(n_events, nwf), replace=False))
    ncols = 5
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows=2 * nrows, ncols=ncols, figsize=(18, 5.2 * nrows),
        squeeze=False, gridspec_kw={'height_ratios': [1, 1] * nrows},
    )

    for plot_idx, event_idx in enumerate(selected):
        row = (plot_idx // ncols) * 2
        col = plot_idx % ncols
        waveform_ax = axes[row, col]
        iq_ax = axes[row + 1, col]

        waveform_ax.plot(tbin, data_corr['ch0'][event_idx] * 1e3, label='I')
        waveform_ax.plot(tbin, data_corr['ch1'][event_idx] * 1e3, label='Q')
        waveform_ax.plot(tbin, vc_proj[event_idx] * 1e3, label='Proj')
        parameter_text = '\n'.join(
            _format_display_parameter(name, float(dfres.iloc[event_idx][name]))
            for name in display_params
        )
        waveform_ax.text(
            0.02, 0.97, parameter_text, transform=waveform_ax.transAxes,
            va='top', ha='left', fontsize=8,
            bbox={'facecolor': 'white', 'alpha': 0.78, 'edgecolor': '0.7', 'pad': 3},
        )
        waveform_ax.set_title(f'Event #{event_idx}')
        waveform_ax.set_ylabel('voltage [mV]')
        waveform_ax.grid(alpha=0.35)
        waveform_ax.legend(fontsize='small', loc='upper right')

        iq_ax.scatter(
            data_corr['ch0'][event_idx] * 1e3,
            data_corr['ch1'][event_idx] * 1e3,
            c=tbin, cmap='viridis', s=4,
        )
        iq_ax.set_xlabel('I [mV]')
        iq_ax.set_ylabel('Q [mV]')
        iq_ax.grid(alpha=0.35)

    for plot_idx in range(len(selected), nrows * ncols):
        row = (plot_idx // ncols) * 2
        col = plot_idx % ncols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')

    fig.suptitle(f'Random raw-event check: {os.path.basename(fn_)}', y=0.995)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=160)
        print(f'saved {output}')
    plt.show()
    return dfres, selected


def analyze_waveforms(fn_, rebin_factor=1, flagAlpha=False):
    """Run the per-waveform analysis for one .npz file.

    rebin_factor -- integer; peak_max/peak_max_t/half_max_*/ninth_max_* (both
    the per-channel ch0_*/ch1_* and the rotated-projection proj_*) are measured
    on a copy of the waveform rebinned by this factor (averaging groups of
    `rebin_factor` consecutive samples), to suppress sample-to-sample noise
    before locating the peak and its half/ninth-max crossings. rebin_factor=1
    measures on the raw (un-rebinned) waveform. This does not affect ped or
    the *_integ columns, which are always computed on the raw waveform, nor
    data_corr/vc_proj (returned at full resolution, for plotting).

    Returns (dfres, data_corr, vc_proj, tbin, sample_rate, nwf):
      dfres      -- one row per waveform, with per-channel ('ch0_*'/'ch1_*') and
                    rotated-projection ('proj_*') columns. peak_max_t and the
                    half_max_*/ninth_max_* crossings are in nanoseconds,
                    relative to the trigger position (same t=0 as `tbin`) --
                    not raw bin numbers.
      data_corr  -- dict of {'ch0':..., 'ch1':...} phase-corrected I/Q arrays.
      vc_proj    -- real-valued waveform after rotating onto the peak's phase.
      tbin       -- time axis [us].
      sample_rate, nwf -- as read from the file.
    """
    data = np.load(fn_,allow_pickle=True)
    for ikey in data.keys():
        print(ikey)

    sample_rate = data['sample_rate']
    npts = data['npts']
    ref_position = data['ref_position']
    nwf = data['ch1'].shape[0]
    tbin = (np.arange(npts) - npts*ref_position/100)/sample_rate
    tbin *= 1e6

    if 1<=VERBOSE:
        print('number of points =',data['npts'])
        print('number of waveforms =',data['ch0'].shape[0],data['ch1'].shape[0])
        print(ref_position)
        print(sample_rate)

    ##### waveform analysis
    if flagAlpha:
        pedbin = [0, int(100e-9*sample_rate)]
    else:
        pedbin = [0,int(npts*ref_position/100)]
    bin1000 = pedbin[1] + int(1000e-9*sample_rate)

    # rebinned time axis used only for peak/half-max/ninth-max measurements
    rebin_factor = int(rebin_factor)
    sample_rate_meas = sample_rate / rebin_factor
    pedbin_meas1 = pedbin[1] // rebin_factor
    tbin_meas = (np.arange(npts//rebin_factor) - (npts//rebin_factor)*ref_position/100.)/(sample_rate/rebin_factor)
    tbin_meas *= 1e6

    v0 = data['ch0']
    v1 = data['ch1']
    vc = v0 + 1j*v1
    # vc = np.exp(-1j*np.pi/2)*vc

    data_corr = {'ch0': np.real(vc), 'ch1': np.imag(vc)}

    dres = {}
    for ich in range(2):
        # ped/integ are always measured on the raw (un-rebinned) waveform
        ped = data_corr[f'ch{ich}'][:,pedbin[0]:pedbin[1]].mean(axis=1)
        wf_corr = data_corr[f'ch{ich}'] - np.reshape(ped,(-1,1))
        print(data_corr[f'ch{ich}'].shape,wf_corr.shape)

        # peak_max/peak_max_t/half_max_*/ninth_max_* are measured on a
        # rebinned copy of wf_corr (rebin_factor=1 -> same as wf_corr)
        wf_meas = _rebin_waveforms(wf_corr, rebin_factor)
        peak_max = np.max(wf_meas,axis=1)
        peak_max_t = np.array([np.argmax(wf_meas[idx]) for idx in range(nwf)])
        # peak_min = wf_meas[:,pedbin_meas1:].min(axis=1)
        # peak_min_t = np.array([np.argmin(wf_meas[idx][pedbin_meas1:]) for idx in range(nwf)])

        # integ = data[f'ch{ich}'][:,pedbin[1]:bin1000].sum(axis=1) - ped*(bin1000-pedbin[1])
        integ = wf_meas[:,pedbin[1]//rebin_factor:bin1000//rebin_factor].sum(axis=1)
        integ_left = np.array([wf_meas[idx, pedbin[1]//rebin_factor:peak_max_t[idx]].sum() for idx in range(nwf)])
        integ_right = np.array([wf_meas[idx, peak_max_t[idx]:peak_max_t[idx]+int(1000e-9*sample_rate_meas)].sum() for idx in range(nwf)])

        half_max_left = [peak_max_t[idx] - _first_or_nan(np.where(wf_meas[idx][:peak_max_t[idx]+1] >= peak_max[idx]/2)[0]) for idx in range(nwf)]
        # first bin below peak_max/2 after the peak (falling edge crossing)
        half_max_right = [_first_or_endpoint(
            np.where(wf_meas[idx][peak_max_t[idx]:] < peak_max[idx]/2)[0],
            len(wf_meas[idx]) - peak_max_t[idx] - 1,
        ) for idx in range(nwf)]
        half_max_right_censored = [
            not np.any(wf_meas[idx][peak_max_t[idx]:] < peak_max[idx]/2)
            for idx in range(nwf)
        ]
        ninth_max_left = [peak_max_t[idx] - _first_or_nan(np.where(wf_meas[idx][:peak_max_t[idx]+1] >= peak_max[idx]/9)[0]) for idx in range(nwf)]
        # first bin below peak_max/9 after the peak (falling edge crossing)
        ninth_max_right = [_first_or_nan(np.where(wf_meas[idx][peak_max_t[idx]:] < peak_max[idx]/9)[0]) for idx in range(nwf)]
        e_max_left = [peak_max_t[idx] - _first_or_nan(np.where(wf_meas[idx][:peak_max_t[idx]+1] >= peak_max[idx]/np.e)[0]) for idx in range(nwf)]
        # first bin below peak_max/9 after the peak (falling edge crossing)
        e_max_right = [_first_or_nan(np.where(wf_meas[idx][peak_max_t[idx]:] < peak_max[idx]/np.e)[0]) for idx in range(nwf)]

        dres[f'ch{ich}_ped'] = ped
        dres[f'ch{ich}_peak_max'] = peak_max
        # times in nanoseconds relative to the trigger position, not bin numbers
        dres[f'ch{ich}_peak_max_t'] = tbin_meas[peak_max_t]
        dres[f'ch{ich}_half_max_left'] = np.array(half_max_left, dtype=float) / sample_rate_meas * 1e6
        dres[f'ch{ich}_half_max_right'] = np.array(half_max_right, dtype=float) / sample_rate_meas * 1e6
        dres[f'ch{ich}_half_max_right_censored'] = half_max_right_censored
        dres[f'ch{ich}_ninth_max_left'] = np.array(ninth_max_left, dtype=float) / sample_rate_meas * 1e6
        dres[f'ch{ich}_ninth_max_right'] = np.array(ninth_max_right, dtype=float) / sample_rate_meas * 1e6
        dres[f'ch{ich}_e_max_left'] = np.array(e_max_left, dtype=float) / sample_rate_meas * 1e6
        dres[f'ch{ich}_e_max_right'] = np.array(e_max_right, dtype=float) / sample_rate_meas * 1e6
        dres[f'ch{ich}_integ'] = integ
        dres[f'ch{ich}_integ_left'] = integ_left
        dres[f'ch{ich}_integ_right'] = integ_right

    # vc_corr/vc_proj stay at full resolution (used for plotting and for
    # proj_integ, which -- like ch{0,1}_integ -- is always measured on the
    # raw waveform). Only proj_max/proj_max_t/half_max_*/ninth_max_* are
    # measured on a rebinned copy.
    vc_corr = vc - np.reshape(dres['ch0_ped'],(-1,1)) - 1j*np.reshape(dres['ch1_ped'],(-1,1))
    vc_corr_meas = _rebin_waveforms(vc_corr, rebin_factor)
    proj_max = np.max(np.abs(vc_corr_meas),axis=1)
    proj_max_t = np.array([np.argmax(np.abs(vc_corr_meas[idx])) for idx in range(nwf)])
    proj_theta = np.angle(vc_corr_meas[np.arange(nwf), proj_max_t])
    vc_proj = np.real(vc_corr * np.exp(-1j*proj_theta.reshape(-1,1)))
    vc_proj_meas = np.real(vc_corr_meas * np.exp(-1j*proj_theta.reshape(-1,1)))
    proj_half_max_left = [proj_max_t[idx] - _first_or_nan(np.where(vc_proj_meas[idx][:proj_max_t[idx]+1] >= proj_max[idx]/2)[0]) for idx in range(nwf)]
    # first bin below proj_max/2 after the peak (falling edge crossing)
    proj_half_max_right = [_first_or_endpoint(
        np.where(vc_proj_meas[idx][proj_max_t[idx]:] < proj_max[idx]/2)[0],
        len(vc_proj_meas[idx]) - proj_max_t[idx] - 1,
    ) for idx in range(nwf)]
    proj_half_max_right_censored = [
        not np.any(vc_proj_meas[idx][proj_max_t[idx]:] < proj_max[idx]/2)
        for idx in range(nwf)
    ]
    proj_nineth_max_left = [proj_max_t[idx] - _first_or_nan(np.where(vc_proj_meas[idx][:proj_max_t[idx]+1] >= proj_max[idx]/9)[0]) for idx in range(nwf)]
    proj_e_max_left = [proj_max_t[idx] - _first_or_nan(np.where(vc_proj_meas[idx][:proj_max_t[idx]+1] >= proj_max[idx]/np.e)[0]) for idx in range(nwf)]
    # first bin below proj_max/9 after the peak (falling edge crossing)
    proj_nineth_max_right = [_first_or_nan(np.where(vc_proj_meas[idx][proj_max_t[idx]:] < proj_max[idx]/9)[0]) for idx in range(nwf)]
    proj_e_max_right = [_first_or_nan(np.where(vc_proj_meas[idx][proj_max_t[idx]:] < proj_max[idx]/np.e)[0]) for idx in range(nwf)]
    proj_integ = np.array([np.sum(vc_proj_meas[idx][pedbin[1]//rebin_factor:bin1000//rebin_factor]) for idx in range(nwf)])
    proj_integ_left = np.array([np.sum(vc_proj_meas[idx][pedbin[1]//rebin_factor:proj_max_t[idx]]) for idx in range(nwf)])
    proj_integ_right = np.array([np.sum(vc_proj_meas[idx][proj_max_t[idx]:proj_max_t[idx]+int(1000e-9*sample_rate_meas)]) for idx in range(nwf)])

    dres['proj_max'] = proj_max
    dres['proj_max_t'] = tbin_meas[proj_max_t]
    dres['proj_theta'] = proj_theta
    dres['proj_half_max_left'] = (np.array(proj_half_max_left, dtype=float)) / sample_rate_meas * 1e6
    dres['proj_half_max_right'] = (np.array(proj_half_max_right, dtype=float)) / sample_rate_meas * 1e6
    dres['proj_half_max_right_censored'] = proj_half_max_right_censored
    dres['proj_ninth_max_left'] = (np.array(proj_nineth_max_left, dtype=float)) / sample_rate_meas * 1e6
    dres['proj_ninth_max_right'] = (np.array(proj_nineth_max_right, dtype=float)) / sample_rate_meas * 1e6
    dres['proj_e_max_left'] = (np.array(proj_e_max_left, dtype=float)) / sample_rate_meas * 1e6
    dres['proj_e_max_right'] = (np.array(proj_e_max_right, dtype=float)) / sample_rate_meas * 1e6
    dres['proj_integ'] = proj_integ
    dres['proj_integ_left'] = proj_integ_left
    dres['proj_integ_right'] = proj_integ_right

    dfres = pd.DataFrame(dres)
    return dfres, data_corr, vc_proj, tbin, sample_rate, nwf


def main_singlefile(fn_, rebin_factor=1, flagAlpha=False):
    dfres, data_corr, vc_proj, tbin, sample_rate, nwf = analyze_waveforms(fn_, rebin_factor=rebin_factor, flagAlpha=flagAlpha)
    print(dfres)

    ##### Make average waveform
    ch0_avg = np.mean(data_corr['ch0']-dfres['ch0_ped'].values.reshape(-1,1),axis=0)
    ch1_avg = np.mean(data_corr['ch1']-dfres['ch1_ped'].values.reshape(-1,1),axis=0)
    vc_avg = np.mean(vc_proj,axis=0)

    fig_scale = 1.0
    ##### Plot in IQ plane
    fig,axs = plt.subplots(figsize=(14/fig_scale,12/fig_scale),ncols=4, nrows=4,sharex=True,sharey=True)
    nax = len(axs.flatten())
    rb = 5
    vt = tbin.reshape(-1,rb).mean(axis=1)
    for idx in range(min(nax,nwf)):
        irow = idx//4
        icol = idx%4
        if idx>=nwf:
            axs[irow,icol].axis('off')
            continue

        v0 = data_corr['ch0'][idx].reshape(-1,rb).mean(axis=1)
        v1 = data_corr['ch1'][idx].reshape(-1,rb).mean(axis=1)
        vc = v0 + 1j*v1
        # vc = np.exp(-1j*np.pi/2)*vc

        ax = axs[irow,icol]
        im = ax.scatter(np.real(vc)*1e3, np.imag(vc)*1e3, c=vt, cmap='viridis')
        ax.plot(dfres['ch0_ped'][idx]*1e3, dfres['ch1_ped'][idx]*1e3, 'x', ms=10, color='r', label='ped')
        ax.plot((dfres['proj_max'][idx]*np.cos(dfres['proj_theta'][idx])+dfres['ch0_ped'][idx])*1e3, (dfres['proj_max'][idx]*np.sin(dfres['proj_theta'][idx])+dfres['ch1_ped'][idx])*1e3, '*', ms=10, color='r', label='proj max')
        if irow==3:
            ax.set_xlabel('I [mV]')
        if icol==0:
            ax.set_ylabel('Q [mV]')
        cbar = fig.colorbar(im, ax=ax)
        if icol==3:
            cbar.set_label('time [$\mu$s]')
        ax.set_title(f'Event #{idx}')
        ax.grid()

    fig.tight_layout()
    fig.savefig('pc1.png')


    #### Plot 1D waveforms
    fig2,axs2 = plt.subplots(figsize=(14/fig_scale,12/fig_scale),ncols=4, nrows=4,sharex=True,sharey=True)
    nax = len(axs2.flatten())
    for idx in range(min(nax,nwf)):
        irow = idx//4
        icol = idx%4
        if idx>=nwf:
            axs2[irow,icol].axis('off')
            continue

        ax = axs2[irow,icol]
        ax.plot(tbin, data_corr['ch0'][idx]*1e3, label='I')
        ax.plot(tbin, data_corr['ch1'][idx]*1e3, label='Q')
        ax.plot(tbin, vc_proj[idx]*1e3, label='Proj')

        if irow==3:
            ax.set_xlabel('time [$\mu$s]')
        if icol==0:
            ax.set_ylabel('voltage [mV]')
        ax.set_title(f'Event #{idx}')
        ax.grid()
        ax.legend(fontsize='small')

    fig2.tight_layout()
    fig2.savefig('pc2.png')

    ##### Histograms
    fig3,axs3 = plt.subplots(figsize=(14/fig_scale,12/fig_scale),ncols=3, nrows=3)
    ax = axs3[0,0]
    ax.hist(dfres['proj_max']*1e3, bins=50)
    ax.set_xlabel('proj max [mV]')
    ax.set_ylabel('counts')
    ax = axs3[0,1]
    ax.hist(dfres['proj_integ']*1e3, bins=50)
    ax.set_xlabel('proj integ [mV]')
    ax.set_ylabel('counts')
    ax = axs3[0,2]
    ax.hist2d(dfres['proj_max'], dfres['proj_integ'], bins=50)
    ax.set_xlabel('proj max [V]')
    ax.set_ylabel('proj integ [V]')
    ax = axs3[1,0]
    ax.hist(dfres['proj_half_max_right']-dfres['proj_half_max_left'], bins=50)
    ax.set_xlabel('proj FWHM [$\mu$s]')
    ax.set_ylabel('counts')
    ax = axs3[1,1]
    ax.hist(dfres['proj_theta'], bins=50)
    ax.set_xlabel('proj theta [rad]') 
    ax.set_ylabel('counts')
    ax = axs3[1,2]
    ax.hist2d(dfres['proj_integ'], dfres['proj_theta'], bins=50)
    ax.set_xlabel('proj integ [V]')
    ax.set_ylabel('proj theta [rad]')
    ax = axs3[2,0]
    ax.hist((dfres['proj_half_max_left']), bins=50, alpha=0.5, label='Rise')
    ax.hist((dfres['proj_half_max_right']), bins=50, alpha=0.5, label='Fall')
    ax.set_xlabel('proj tau [$\mu$s]')
    ax.set_ylabel('counts')
    ax.legend(fontsize='small')
    ax = axs3[2,1]
    ax.hist2d(dfres['proj_integ'], (dfres['proj_half_max_left']), bins=50)
    ax.set_xlabel('proj integ [V]')
    ax.set_ylabel('proj rise [$\mu$s]')
    ax = axs3[2,2]
    # proj_half_max_right can be NaN (falling edge never dropped below half-max
    # within the recorded window) -- hist2d's autorange can't handle that, so
    # drop those waveforms here (unlike ax.hist(), which already tolerates NaN)
    proj_fall = (dfres['proj_half_max_right'])
    finite = np.isfinite(proj_fall)
    if (~finite).sum() > 0:
        print(f'WARNING: {(~finite).sum()} waveform(s) had no proj_half_max_right crossing -- excluded from the fall-time hist2d')
    ax.hist2d(dfres['proj_integ'][finite], proj_fall[finite], bins=50)
    ax.set_xlabel('proj integ [V]')
    ax.set_ylabel('proj fall [$\mu$s]')

    for iax in axs3.flatten():
        iax.grid() 

    fig3.tight_layout()
    fig3.savefig('pc3.png')

    ##### Average waveform
    fig4,axs4 = plt.subplots(figsize=(14/fig_scale,12/fig_scale),ncols=1, nrows=1)
    ax = axs4
    ax.plot(tbin, ch0_avg*1e3, label='I')
    ax.plot(tbin, ch1_avg*1e3, label='Q')
    ax.plot(tbin, vc_avg*1e3, label='Proj')
    ax.set_xlabel('time [$\mu$s]')
    ax.set_ylabel('voltage [mV]')
    ax.grid()
    ax.legend(fontsize='small')
    fig4.tight_layout()
    fig4.savefig('pc4.png')

    plt.show()



def main_compare_distance(patterns_labels=None, rebin_factor=1):
    """Compare proj_* waveform parameters and average pulse shapes across several runs.

    patterns_labels -- list of (glob_pattern, label) pairs. Each pattern is globbed
    (relative to this script's directory) to find the matching .npz file(s); all
    waveforms across the matched file(s) are pooled under that label.
    rebin_factor -- passed through to analyze_waveforms() for the peak/half-max/
    ninth-max measurement (see its docstring).

    Returns (dgroups, davg, dtbin):
      dgroups -- {label: dfgroup} per-waveform parameter DataFrame (as before).
      davg    -- {label: {'ch0':..., 'ch1':..., 'proj':...}} average waveform per channel.
      dtbin   -- {label: tbin} time axis [us] used for that label's average waveforms.
    """
    if patterns_labels is None:
        # patterns_labels = [ # Laser-power scan at (x,z) = (5.10, 6.00) mm
        #     ('Aug31st/wf_260831_16450?_*.npz', '1/10'),
        #     ('Aug31st/wf_260831_16454?_*.npz', '1/20'),
        #     ('Aug31st/wf_260831_16462?_*.npz', '1/40'),
        #     ('Aug31st/wf_260831_15575?_*.npz', '1/100'),
        #     ('Aug31st/wf_260831_16471?_*.npz', '1/200'),
        #     ('Aug31st/wf_260831_16475?_*.npz', '1/400'),
        #     ('Aug31st/wf_260831_16483?_*.npz', '1/1000'),
        # ]
        # patterns_labels = [ # RF-power scan at (x,z) = (4.00, 6.30) mm
        #     ('Aug31st/wf_260831_15575?_*.npz', '+1 dB'),
        #     ('Aug31st/wf_260831_16062?_*.npz', '-2 dB'),
        #     ('Aug31st/wf_260831_16070?_*.npz', '-5 dB'),
        #     ('Aug31st/wf_260831_16274?_*.npz', '-2 dB (direct)'),
        # ]
        # patterns_labels = [ # RF-power scan at (x,z) = (4.00, 6.00) mm
        #     ('Aug31st/wf_260831_16151?_*.npz', '+1 dB'),
        #     ('Aug31st/wf_260831_15502?_*.npz', '-2 dB'),
        #     ('Aug31st/wf_260831_16154?_*.npz', '-5 dB'),
        #     ('Aug31st/wf_260831_16312?_*.npz', '-2 dB (direct)'),
        # ]
        # patterns_labels = [ # z-scan at x = 4.0 mm
        #     ('Aug31st/wf_260831_15554?_*.npz', '6.90 mm'),
        #     ('Aug31st/wf_260831_15562?_*.npz', '6.70 mm'),
        #     ('Aug31st/wf_260831_15570?_*.npz', '6.50 mm'),
        #     ('Aug31st/wf_260831_15575?_*.npz', '6.30 mm'),
        #     ('Aug31st/wf_260831_15413?_*.npz', '6.15 mm'),
        #     ('Aug31st/wf_260831_15421?_*.npz', '6.00 mm'),
        #     ('Aug31st/wf_260831_15425?_*.npz', '5.85 mm'),
        #     ('Aug31st/wf_260831_15434?_*.npz', '5.70 mm'),
        #     ('Aug31st/wf_260831_16212?_*.npz', '5.50 mm'),
        # ]
        # patterns_labels = [ # z-scan at x = 3.5 mm
        #     ('Aug31st/wf_260831_15474?_*.npz', '6.90 mm'),
        #     ('Aug31st/wf_260831_15482?_*.npz', '6.70 mm'),
        #     ('Aug31st/wf_260831_15490?_*.npz', '6.50 mm'),
        #     ('Aug31st/wf_260831_15494?_*.npz', '6.30 mm'),
        #     ('Aug31st/wf_260831_15502?_*.npz', '6.00 mm'),
        #     ('Aug31st/wf_260831_15514?_*.npz', '5.70 mm'),
        #     ('Aug31st/wf_260831_16203?_*.npz', '5.50 mm'),
        # ]
        # patterns_labels = [ # RF Freq scan at (x,z) = (4.00, 6.30) mm
        #     ('Sep1st/wf_260901_14322?_*.npz', '-40 MHz'),
        #     ('Sep1st/wf_260901_14320?_*.npz', '-30 MHz'),
        #     ('Sep1st/wf_260901_14313?_*.npz', '-20 MHz'),
        #     ('Sep1st/wf_260901_14310?_*.npz', '-10 MHz'),
        #     ('Sep1st/wf_260901_14284?_*.npz', '0 MHz'),
        #     ('Sep1st/wf_260901_14291?_*.npz', '+10 MHz'),
        #     ('Sep1st/wf_260901_14294?_*.npz', '+20 MHz'),
        #     ('Sep1st/wf_260901_14300?_*.npz', '+30 MHz'),
        #     ('Sep1st/wf_260901_14303?_*.npz', '+40 MHz'),
        # ]
        # patterns_labels = [ # RF Freq scan at (x,z) = (4.00, 6.00) mm
        #     ('Sep1st/wf_260901_14271?_*.npz', '-40 MHz'),
        #     ('Sep1st/wf_260901_14264?_*.npz', '-30 MHz'),
        #     ('Sep1st/wf_260901_14261?_*.npz', '-20 MHz'),
        #     ('Sep1st/wf_260901_14255?_*.npz', '-10 MHz'),
        #     ('Sep1st/wf_260901_14232?_*.npz', '0 MHz'),
        #     ('Sep1st/wf_260901_14235?_*.npz', '+10 MHz'),
        #     ('Sep1st/wf_260901_14242?_*.npz', '+20 MHz'),
        #     ('Sep1st/wf_260901_14245?_*.npz', '+30 MHz'),
        #     ('Sep1st/wf_260901_14252?_*.npz', '+40 MHz'),
        # ]
        # patterns_labels = [ # RF Freq scan at (x,z) = (3.50, 6.00) mm
        #     ('Sep1st/wf_260901_14592?_*.npz', '-40 MHz'),
        #     ('Sep1st/wf_260901_14590?_*.npz', '-30 MHz'),
        #     ('Sep1st/wf_260901_14583?_*.npz', '-20 MHz'),
        #     ('Sep1st/wf_260901_14581?_*.npz', '-10 MHz'),
        #     ('Sep1st/wf_260901_14560?_*.npz', '0 MHz'),
        #     ('Sep1st/wf_260901_14562?_*.npz', '+10 MHz'),
        #     ('Sep1st/wf_260901_14565?_*.npz', '+20 MHz'),
        #     ('Sep1st/wf_260901_14571?_*.npz', '+30 MHz'),
        #     ('Sep1st/wf_260901_14574?_*.npz', '+40 MHz'),
        # ]
        # patterns_labels = [ # RF Freq scan at (x,z) = (3.50, 6.00) mm
        #     ('Aug31st/wf_260831_17292?_*.npz', '5.299 GHz'),
        #     ('Aug31st/wf_260831_15502?_*.npz', '5.269 GHz'),
        #     ('Aug31st/wf_260831_17300?_*.npz', '5.239 GHz'),
        # ]
        # patterns_labels = [ # Temp scan at (x,z) = (4.00, 6.30) mm
        #     ('Sep1st/wf_260901_14284?_*.npz', '3.90 K'),
        #     ('Sep1st/wf_260901_15450?_*.npz', '4.20 K'),
        #     ('Sep1st/wf_260901_16103?_*.npz', '4.55 K'),
        #     ('Sep1st/wf_260901_16461?_*.npz', '4.90 K'),
        #     ('Sep1st/wf_260901_17042?_*.npz', '5.30 K'),
        #     ('Sep1st/wf_260901_17300?_*.npz', '5.95 K'),
        # ]
        # patterns_labels = [ # Temp scan at (x,z) = (4.00, 6.00) mm
        #     ('Sep1st/wf_260901_14232?_*.npz', '3.90 K'),
        #     ('Sep1st/wf_260901_15462?_*.npz', '4.20 K'),
        #     ('Sep1st/wf_260901_16082?_*.npz', '4.55 K'),
        #     ('Sep1st/wf_260901_16445?_*.npz', '4.90 K'),
        #     ('Sep1st/wf_260901_17055?_*.npz', '5.30 K'),
        #     ('Sep1st/wf_260901_17284?_*.npz', '5.95 K'),
        # ]
        patterns_labels = [ # Temp scan at (x,z) = (3.50, 6.00) mm
            ('Sep1st/wf_260901_14560?_*.npz', '3.90 K'),
            ('Sep1st/wf_260901_15351?_*.npz', '4.20 K'),
            ('Sep1st/wf_260901_16164?_*.npz', '4.55 K'),
            ('Sep1st/wf_260901_16431?_*.npz', '4.90 K'),
            ('Sep1st/wf_260901_17072?_*.npz', '5.30 K'),
            ('Sep1st/wf_260901_17271?_*.npz', '5.95 K'),
        ]


    params = {
        'proj_max_t': 'proj max t [ns]',
        'proj_rise': 'proj max t - proj half max left [ns]',
        'proj_fall': 'proj half max right - proj max t [ns]',
        'proj_integ': 'proj integral',
    }

    basedir = os.path.dirname(os.path.abspath(__file__))
    dgroups = {}
    dtbin = {}
    dsample_rate = {}
    davg = {}
    for pattern, label in patterns_labels:
        fns = sorted(glob.glob(os.path.join(basedir, pattern)))
        if len(fns) == 0:
            print(f'WARNING: no files matched pattern {pattern!r} -- skipping "{label}"')
            continue
        if len(fns) > 1:
            print(f'WARNING: pattern {pattern!r} matched {len(fns)} files -- pooling all of them for "{label}"')

        dfs = []
        ch0_corr_list = []
        ch1_corr_list = []
        vc_proj_list = []
        for fn_ in fns:
            print(f'analyzing {fn_} ({label}) ...')
            dfres, data_corr, vc_proj, tbin, sample_rate, nwf = analyze_waveforms(fn_, rebin_factor=rebin_factor)
            dfs.append(dfres)
            ch0_corr_list.append(data_corr['ch0'] - dfres['ch0_ped'].values.reshape(-1,1))
            ch1_corr_list.append(data_corr['ch1'] - dfres['ch1_ped'].values.reshape(-1,1))
            vc_proj_list.append(vc_proj)
        dfgroup = pd.concat(dfs, ignore_index=True)
        dfgroup['proj_rise'] = dfgroup['proj_half_max_left']
        dfgroup['proj_fall'] = dfgroup['proj_half_max_right']
        dgroups[label] = dfgroup
        dtbin[label] = tbin
        dsample_rate[label] = sample_rate
        # average over all pooled waveforms (equivalent to per-file average weighted by nwf)
        davg[label] = {
            'ch0': np.concatenate(ch0_corr_list, axis=0).mean(axis=0),
            'ch1': np.concatenate(ch1_corr_list, axis=0).mean(axis=0),
            'proj': np.concatenate(vc_proj_list, axis=0).mean(axis=0),
        }

    fig,axs = plt.subplots(figsize=(12,10),ncols=2, nrows=2)
    for idx,(label,dfgroup) in enumerate(dgroups.items()):
        tbin = dtbin[label]
        sample_rate = dsample_rate[label]

        # proj_max_t is already a time [ns] relative to the trigger (t=0 of
        # tbin), not a bin index -- convert ns -> us directly instead of
        # indexing into tbin.
        binmaxt = np.linspace(tbin.min(),tbin.max(),100)
        ax = axs[0,0]
        ax.hist(dfgroup['proj_max_t'].to_numpy(), bins=binmaxt, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)

        bininteg = np.arange(0,50.1,0.2)
        ax = axs[0,1]
        ax.hist((dfgroup['proj_integ']), bins=bininteg, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)

        bintau = np.linspace(0,0.8,100)
        ax = axs[1,0]
        ax.hist(dfgroup['proj_rise'].to_numpy(), bins=bintau, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)
        ax = axs[1,1]
        ax.hist(dfgroup['proj_fall'].to_numpy(), bins=bintau, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)

    axs[0,0].set_xlabel('proj max t [$\mu$s]')
    axs[0,1].set_xlabel('proj integ [V * 0.4 ns]')
    axs[1,0].set_xlabel('proj rise [$\mu$s]')
    axs[1,1].set_xlabel('proj fall [$\mu$s]')
    for iax in axs.flatten():
        iax.set_ylabel('counts')
        iax.grid()
        iax.legend(fontsize='x-small', ncol=2)

    fig.tight_layout()
    fig.savefig('pc1.png')

    ##### Average waveform comparison
    fig5,axs5 = plt.subplots(figsize=(18,5),ncols=3, nrows=1,sharex=True,sharey=True)
    for idx,(label,avg) in enumerate(davg.items()):
        tbin = dtbin[label]
        axs5[0].plot(tbin, avg['ch0']*1e3, c=mpl.cm.jet(idx/len(davg)), label=label)
        axs5[1].plot(tbin, avg['ch1']*1e3, c=mpl.cm.jet(idx/len(davg)), label=label)
        axs5[2].plot(tbin, avg['proj']*1e3, c=mpl.cm.jet(idx/len(davg)), label=label)

    axs5[0].set_title('I (ch0)')
    axs5[1].set_title('Q (ch1)')
    axs5[2].set_title('Proj')
    axs5[0].set_ylabel('voltage [mV]')
    for iax in axs5:
        iax.set_xlabel('time [$\mu$s]')
        iax.grid()
        iax.legend(fontsize='x-small', ncol=2)

    fig5.tight_layout()
    fig5.savefig('pc5.png')

    ymax = axs5[0].get_ylim()[1]
    for iax in axs5:
        iax.set_yscale('log')
        iax.set_ylim(1e-3, ymax)
    fig5.savefig('pc5_log.png')
        

    ##### Average waveform comparison
    fig6,axs6 = plt.subplots(figsize=(18,5),ncols=3, nrows=1,sharex=True,sharey=True)
    for idx,(label,avg) in enumerate(davg.items()):
        tbin = dtbin[label]
        axs6[0].plot(tbin, avg['ch0']/np.max(avg['ch0']), c=mpl.cm.jet(idx/len(davg)), label=label)
        axs6[1].plot(tbin, avg['ch1']/np.max(avg['ch1']), c=mpl.cm.jet(idx/len(davg)), label=label)
        axs6[2].plot(tbin, avg['proj']/np.max(avg['proj']), c=mpl.cm.jet(idx/len(davg)), label=label)

    axs6[0].set_title('I (ch0)')
    axs6[1].set_title('Q (ch1)')
    axs6[2].set_title('Proj')
    axs6[0].set_ylabel('voltage [normalized]')
    for iax in axs6:
        iax.set_xlabel('time [$\mu$s]')
        iax.grid()
        iax.legend(fontsize='x-small', ncol=2)

    fig6.tight_layout()
    fig6.savefig('pc6.png')

    ##### Check integrated voltage vs parameter
    # fig7,axs7 = plt.subplots(figsize=(9,6),sharex=True,sharey=True)
    # ax = axs7
    # xvals = np.array([_frac_to_float(label) for label in dgroups.keys()])
    # print('fig7\t',xvals,dgroups.keys())
    # ax.plot(xvals, np.array([dfres['ch0_integ'].mean() for dfres in dgroups.values()]), 'o-', label='I (ch0)')
    # ax.plot(xvals, np.array([dfres['ch1_integ'].mean() for dfres in dgroups.values()]), 's-', label='Q (ch1)')
    # ax.plot(xvals, np.array([dfres['proj_integ'].mean() for dfres in dgroups.values()]), '^-', label='Proj')

    # ax.set_xlabel('Parameter')
    # ax.set_ylabel('integrated voltage [mV * 0.4 ns]')
    # ax.grid()
    # ax.legend(fontsize='x-small', ncol=1)

    # fig7.tight_layout()
    # fig7.savefig('pc7.png')

    plt.show()

    return dgroups, davg, dtbin


def main_process_timerange(dirname='Aug27th', t_start=None, t_end=None, rebin_factor=1):
    """Run analyze_waveforms() on every wf_*.npz file in `dirname` whose
    embedded timestamp falls within [t_start, t_end], selecting files by when
    they were taken rather than by matching a glob pattern per labeled group.

    dirname -- directory to scan for wf_*.npz files, relative to this
        script's directory (e.g. 'Aug27th') or an absolute path. Note the
        directory name is just a label -- files inside it are matched by
        their own embedded timestamp, which can roll over past midnight
        into the next calendar day.
    t_start, t_end -- inclusive bounds on the file's embedded timestamp.
        Each is either a 'YYMMDD_HHMMSS' string (same format as in the
        filename, e.g. '260827_190000'), a datetime.datetime, or None for
        unbounded on that side.
    rebin_factor -- passed through to analyze_waveforms().

    Returns list of (fn_, dfres) for the files actually processed, in time
    order.
    """
    t_start = _to_datetime(t_start)
    t_end = _to_datetime(t_end)

    basedir = os.path.dirname(os.path.abspath(__file__))
    scandir = dirname if os.path.isabs(dirname) else os.path.join(basedir, dirname)

    fns = sorted(glob.glob(os.path.join(scandir, 'wf_*.npz')))
    if len(fns) == 0:
        print(f'WARNING: no wf_*.npz files found in {scandir!r}')
        return []

    selected = []
    for fn_ in fns:
        ts = _parse_wf_timestamp(fn_)
        if ts is None:
            print(f'WARNING: could not parse a timestamp from {fn_!r} -- skipping')
            continue
        if t_start is not None and ts < t_start:
            continue
        if t_end is not None and ts > t_end:
            continue
        selected.append((fn_, ts))
    selected.sort(key=lambda x: x[1])

    print(f'{len(selected)}/{len(fns)} files in {scandir!r} fall within the requested time range'
          f' ({t_start} to {t_end})')

    results = []
    for fn_, ts in selected:
        print(f'analyzing {fn_} (taken {ts}) ...')
        dfres, data_corr, vc_proj, tbin, sample_rate, nwf = analyze_waveforms(fn_, rebin_factor=rebin_factor, flagAlpha=True)
        results.append((fn_, dfres))

    return results


if __name__ == "__main__":
    # optional trailing integer arg = rebin_factor for peak/half-max/ninth-max
    # measurement (see analyze_waveforms()); defaults to 1 (no rebinning)
    if len(sys.argv)>1 and sys.argv[1] == '--gui':
        initial_file = sys.argv[2] if len(sys.argv) > 2 else None
        main_realtime_check_gui(initial_file)
    elif len(sys.argv)>1 and sys.argv[1] == 'compare':
        rebin_factor = int(sys.argv[2]) if len(sys.argv)>2 else 1
        main_compare_distance(rebin_factor=rebin_factor)
    elif len(sys.argv)>1 and sys.argv[1] == 'timerange':
        # python ana_forJPS.py timerange [dirname] [t_start] [t_end] [rebin_factor]
        # dirname defaults to 'Aug27th'; t_start/t_end are 'YYMMDD_HHMMSS'
        # (matching the wf_*.npz filename timestamp) or 'None' for unbounded.
        # e.g.: python ana_forJPS.py timerange Aug27th 260827_190000 260827_230000
        dirname = sys.argv[2] if len(sys.argv)>2 else 'Aug27th'
        t_start = sys.argv[3] if len(sys.argv)>3 and sys.argv[3]!='None' else None
        t_end = sys.argv[4] if len(sys.argv)>4 and sys.argv[4]!='None' else None
        rebin_factor = int(sys.argv[5]) if len(sys.argv)>5 else 1
        main_process_timerange(dirname, t_start=t_start, t_end=t_end, rebin_factor=rebin_factor)
    elif len(sys.argv)>1 and '.npz' in sys.argv[1]:
        # python parameter_ckeck.py data.npz [rebin] [flagAlpha] [params] [seed]
        rebin_factor = int(sys.argv[2]) if len(sys.argv)>2 else 1
        flagAlpha = bool(int(sys.argv[3])) if len(sys.argv)>3 else False
        display_params = (
            sys.argv[4].split(',') if len(sys.argv) > 4 else DEFAULT_DISPLAY_PARAMS
        )
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
        main_realtime_check(
            sys.argv[1], rebin_factor=rebin_factor, flagAlpha=flagAlpha,
            display_params=display_params, seed=seed,
        )
    else:
        main()
