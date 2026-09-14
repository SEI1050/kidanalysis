import os
import re
import sys
import time
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit  
import quadpy

import ds_style

VERBOSE = 1

def _right_one_over_e_point(waveform):
    """最大点より右側で、初めて max/e を下回る点を返す。"""
    proj_max_t = int(np.argmax(waveform))
    proj_max = waveform[proj_max_t]

    below = np.flatnonzero(
        waveform[proj_max_t + 1:] < proj_max / np.e
    )
    if below.size == 0:
        return None

    point_idx = proj_max_t + 1 + int(below[0])
    return point_idx, waveform[point_idx]

def _first_or_nan(idx_arr):
    """First element of a np.where(...)[0] result, or nan if no crossing was found."""
    return idx_arr[0] if len(idx_arr) > 0 else np.nan

def _last_or_nan(idx_arr):
    """Last element of a np.where(...)[0] result, or nan if no crossing was found."""
    return idx_arr[-1] if len(idx_arr) > 0 else np.nan

def _frac_to_float(s):
    """Parse a label like '1', '1/2', '1/400' into a float (1.0, 0.5, 0.0025, ...)."""
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)

def main():
    return 1


def analyze_waveforms(fn_):
    """Run the per-waveform analysis for one .npz file.

    Returns (dfres, data_corr, vc_proj, tbin, sample_rate, nwf):
      dfres      -- one row per waveform, with per-channel ('ch0_*'/'ch1_*') and
                    rotated-projection ('proj_*') columns.
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
    pedbin = [0,int(npts*ref_position/100)]
    bin1000 = pedbin[1] + int(1000e-9*sample_rate)

    v0 = data['ch0']
    v1 = data['ch1']
    vc = v0 + 1j*v1
    # vc = np.exp(-1j*np.pi/2)*vc

    data_corr = {'ch0': np.real(vc), 'ch1': np.imag(vc)}

    dres = {}
    for ich in range(2):
        ped = data_corr[f'ch{ich}'][:,pedbin[0]:pedbin[1]].mean(axis=1)
        peak_max = data_corr[f'ch{ich}'][:,pedbin[1]:].max(axis=1)
        peak_max_t = np.array([np.argmax(data_corr[f'ch{ich}'][idx][pedbin[1]:]) for idx in range(nwf)])
        # peak_min = data_corr[f'ch{ich}'][:,pedbin[1]:].min(axis=1)
        # peak_min_t = np.array([np.argmin(data_corr[f'ch{ich}'][idx][pedbin[1]:]) for idx in range(nwf)])

        wf_corr = data_corr[f'ch{ich}'] - np.reshape(ped,(-1,1))
        print(data_corr[f'ch{ich}'].shape,wf_corr.shape)
        # integ = data[f'ch{ich}'][:,pedbin[1]:bin1000].sum(axis=1) - ped*(bin1000-pedbin[1])
        integ = wf_corr[:,pedbin[1]:bin1000].sum(axis=1)
        peak_max_t_abs = pedbin[1] + peak_max_t
        half_max_left = [_first_or_nan(np.where(wf_corr[idx][pedbin[1]:peak_max_t_abs[idx]+1] >= peak_max[idx]/2)[0]) for idx in range(nwf)]
        half_max_right = [peak_max_t[idx] + _last_or_nan(np.where(wf_corr[idx][peak_max_t_abs[idx]:bin1000] >= peak_max[idx]/2)[0]) for idx in range(nwf)]
        ninth_max_left = [_first_or_nan(np.where(wf_corr[idx][pedbin[1]:peak_max_t_abs[idx]+1] >= peak_max[idx]/9)[0]) for idx in range(nwf)]
        ninth_max_right = [peak_max_t[idx] + _last_or_nan(np.where(wf_corr[idx][peak_max_t_abs[idx]:bin1000] >= peak_max[idx]/9)[0]) for idx in range(nwf)]

        dres[f'ch{ich}_ped'] = ped
        dres[f'ch{ich}_peak_max'] = peak_max
        dres[f'ch{ich}_peak_max_t'] = peak_max_t
        dres[f'ch{ich}_half_max_left'] = half_max_left
        dres[f'ch{ich}_half_max_right'] = half_max_right
        dres[f'ch{ich}_ninth_max_left'] = ninth_max_left
        dres[f'ch{ich}_ninth_max_right'] = ninth_max_right
        dres[f'ch{ich}_integ'] = integ

    vc_corr = vc - np.reshape(dres['ch0_ped'],(-1,1)) - 1j*np.reshape(dres['ch1_ped'],(-1,1))
    proj_max = np.max(np.abs(vc_corr),axis=1)
    proj_max_t = np.array([np.argmax(np.abs(vc_corr[idx])) for idx in range(nwf)])
    proj_theta = np.angle(vc_corr[np.arange(nwf), proj_max_t])
    vc_proj = vc_corr * np.exp(-1j*proj_theta.reshape(-1,1))
    vc_proj = np.real(vc_proj)
    proj_half_max_left = [_first_or_nan(np.where(vc_proj[idx][:proj_max_t[idx]+1] >= proj_max[idx]/2)[0]) for idx in range(nwf)]
    proj_half_max_right = [proj_max_t[idx] + _last_or_nan(np.where(vc_proj[idx][proj_max_t[idx]:] >= proj_max[idx]/2)[0]) for idx in range(nwf)]
    proj_nineth_max_left = [_first_or_nan(np.where(vc_proj[idx][:proj_max_t[idx]+1] >= proj_max[idx]/9)[0]) for idx in range(nwf)]
    proj_nineth_max_right = [proj_max_t[idx] + _last_or_nan(np.where(vc_proj[idx][proj_max_t[idx]:] >= proj_max[idx]/9)[0]) for idx in range(nwf)]  
    proj_integ = np.array([np.sum(vc_proj[idx][pedbin[1]:bin1000]) for idx in range(nwf)])

    dres['proj_max'] = proj_max
    dres['proj_max_t'] = proj_max_t
    dres['proj_theta'] = proj_theta
    dres['proj_half_max_left'] = proj_half_max_left
    dres['proj_half_max_right'] = proj_half_max_right
    dres['proj_ninth_max_left'] = proj_nineth_max_left
    dres['proj_ninth_max_right'] = proj_nineth_max_right
    dres['proj_integ'] = proj_integ

    dfres = pd.DataFrame(dres)
    return dfres, data_corr, vc_proj, tbin, sample_rate, nwf


def main_singlefile(fn_):
    dfres, data_corr, vc_proj, tbin, sample_rate, nwf = analyze_waveforms(fn_)
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
    ax.set_xlabel('proj FWHM [samples]')
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
    ax.hist((dfres['proj_max_t']-dfres['proj_half_max_left'])/sample_rate*1e6, bins=50, alpha=0.5, label='Rise')
    ax.hist((dfres['proj_half_max_right']-dfres['proj_max_t'])/sample_rate*1e6, bins=50, alpha=0.5, label='Fall')
    ax.set_xlabel('proj tau [$\mu$s]')
    ax.set_ylabel('counts')
    ax.legend(fontsize='small')
    ax = axs3[2,1]
    ax.hist2d(dfres['proj_integ'], (dfres['proj_max_t']-dfres['proj_half_max_left'])/sample_rate*1e6, bins=50)
    ax.set_xlabel('proj integ [V]')
    ax.set_ylabel('proj rise [$\mu$s]')
    ax = axs3[2,2]
    ax.hist2d(dfres['proj_integ'], (dfres['proj_half_max_right']-dfres['proj_max_t'])/sample_rate*1e6, bins=50)
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

    dfres.to_csv(fn_[:-4]+'_ana.csv',index=False)


def main_compare_distance(patterns_labels=None):
    """Compare proj_* waveform parameters and average pulse shapes across several runs.

    patterns_labels -- list of (glob_pattern, label) pairs. Each pattern is globbed
    (relative to this script's directory) to find the matching .npz file(s); all
    waveforms across the matched file(s) are pooled under that label.

    Returns (dgroups, davg, dtbin):
      dgroups -- {label: dfgroup} per-waveform parameter DataFrame (as before).
      davg    -- {label: {'ch0':..., 'ch1':..., 'proj':...}} average waveform per channel.
      dtbin   -- {label: tbin} time axis [us] used for that label's average waveforms.
    """
    if patterns_labels is None:
        # patterns_labels = [ # z-scan at x = 5.25 mm
        #     ('Aug25th/wf_260825_13305?_*.npz', '6.90 mm'),
        #     ('Aug25th/wf_260825_13314?_*.npz', '6.80 mm'),
        #     ('Aug25th/wf_260825_13323?_*.npz', '6.75 mm'),
        #     ('Aug25th/wf_260825_13331?_*.npz', '6.70 mm'),
        #     ('Aug25th/wf_260825_13340?_*.npz', '6.50 mm'),
        #     ('Aug25th/wf_260825_13350?_*.npz', '6.30 mm'),
        #     ('Aug25th/wf_260825_13354?_*.npz', '6.10 mm'),
        #     ('Aug25th/wf_260825_13363?_*.npz', '6.05 mm'),
        #     ('Aug25th/wf_260825_13371?_*.npz', '6.00 mm'),
        #     ('Aug25th/wf_260825_13380?_*.npz', '5.95 mm'),
        #     ('Aug25th/wf_260825_13385?_*.npz', '5.70 mm'),
        # ]
        patterns_labels = [ # z-scan at x = 4.00 mm
            ('/Volumes/NO NAME/20260831/data_0831_155539/wf_260831_15554?_*.npz', '6.90 mm'),
            # ('Aug25th/wf_260825_13512?_*.npz', '6.85 mm'),
            # ('Aug25th/wf_260825_13495?_*.npz', '6.80 mm'),
            # ('Aug25th/wf_260825_13491?_*.npz', '6.75 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_155621/wf_260831_15562?_*.npz', '6.70 mm'),
            # ('Aug25th/wf_260825_13475?_*.npz', '6.60 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_155700/wf_260831_15570?_*.npz', '6.50 mm'),
            # ('Aug25th/wf_260825_13462?_*.npz', '6.40 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_153857/wf_260831_15385?_*.npz', '6.30 mm'),
            # ('Aug25th/wf_260825_13440?_*.npz', '6.20 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_154132/wf_260831_15413?_*.npz', '6.15 mm'),
            # ('Aug25th/wf_260825_13425?_*.npz', '6.05 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_154214/wf_260831_15421?_*.npz', '6.00 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_154255/wf_260831_15425?_*.npz', '5.85 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_154341/wf_260831_15434?_*.npz', '5.70 mm'),
            ('/Volumes/NO NAME/20260831/data_0831_162119/wf_260831_16212?_*.npz', '5.50 mm'),
        ]
        # patterns_labels = [ # z-scan at x = 4.60 mm
        #     ('Aug25th/wf_260825_13083?_*.npz', '6.90 mm'),
        #     ('Aug25th/wf_260825_13091?_*.npz', '6.85 mm'),
        #     ('Aug25th/wf_260825_13095?_*.npz', '6.80 mm'),
        #     ('Aug25th/wf_260825_13103?_*.npz', '6.75 mm'),
        #     ('Aug25th/wf_260825_13111?_*.npz', '6.70 mm'),
        #     ('Aug25th/wf_260825_13120?_*.npz', '6.50 mm'),
        #     ('Aug25th/wf_260825_13125?_*.npz', '6.30 mm'),
        #     ('Aug25th/wf_260825_13133?_*.npz', '6.10 mm'),
        #     ('Aug25th/wf_260825_13143?_*.npz', '6.05 mm'),
        #     ('Aug25th/wf_260825_13151?_*.npz', '6.00 mm'),
        #     ('Aug25th/wf_260825_13160?_*.npz', '5.95 mm'),
        #     ('Aug25th/wf_260825_13170?_*.npz', '5.70 mm'),
        # ]
        # patterns_labels = [ # x-scan at z = 6.30 mm
        #     ('Aug25th/wf_260825_13240?_*.npz', '4.30 mm'),
        #     ('Aug25th/wf_260825_13200?_*.npz', '4.50 mm'),
        #     ('Aug25th/wf_260825_13125?_*.npz', '4.60 mm'),
        #     ('Aug25th/wf_260825_13061?_*.npz', '4.75 mm'),
        #     ('Aug25th/wf_260825_13562?_*.npz', '4.95 mm'),
        #     ('Aug25th/wf_260825_13444?_*.npz', '5.10 mm'),
        #     ('Aug25th/wf_260825_13350?_*.npz', '5.25 mm'),
        #     ('Aug25th/wf_260825_13281?_*.npz', '5.60 mm'),
        # ]
        # patterns_labels = [ # z-scan at x = 5.10 mm, 0 dB
        #     ('Aug25th/wf_260825_14290?_*.npz', '6.90 mm'),
        #     ('Aug25th/wf_260825_14311?_*.npz', '6.70 mm'),
        #     ('Aug25th/wf_260825_14331?_*.npz', '6.50 mm'),
        #     ('Aug25th/wf_260825_14352?_*.npz', '6.30 mm'),
        #     ('Aug25th/wf_260825_14373?_*.npz', '6.10 mm'),
        #     ('Aug25th/wf_260825_14393?_*.npz', '6.00 mm'),
        #     ('Aug25th/wf_260825_14414?_*.npz', '5.70 mm'),
        # ]
        # patterns_labels = [ # z-scan at x = 5.10 mm, -5 dB
        #     ('Aug25th/wf_260825_14294?_*.npz', '6.90 mm'),
        #     ('Aug25th/wf_260825_14315?_*.npz', '6.70 mm'),
        #     ('Aug25th/wf_260825_14335?_*.npz', '6.50 mm'),
        #     ('Aug25th/wf_260825_14355?_*.npz', '6.30 mm'),
        #     ('Aug25th/wf_260825_14381?_*.npz', '6.10 mm'),
        #     ('Aug25th/wf_260825_14401?_*.npz', '6.00 mm'),
        #     ('Aug25th/wf_260825_14422?_*.npz', '5.70 mm'),
        # ]
        # patterns_labels = [ # z-scan at x = 5.10 mm, Filter 1/400
        #     ('Aug25th/wf_260825_14302?_*.npz', '6.90 mm'),
        #     ('Aug25th/wf_260825_14322?_*.npz', '6.70 mm'),
        #     ('Aug25th/wf_260825_14343?_*.npz', '6.50 mm'),
        #     ('Aug25th/wf_260825_14363?_*.npz', '6.30 mm'),
        #     ('Aug25th/wf_260825_14385?_*.npz', '6.10 mm'),
        #     ('Aug25th/wf_260825_14405?_*.npz', '6.00 mm'),
        #     ('Aug25th/wf_260825_14430?_*.npz', '5.70 mm'),
        # ]
        # patterns_labels = [ # RF-power scan at (x,z) = (5.10, 6.70) mm
        #     ('Aug25th/wf_260825_14315?_*.npz', '-5 dB'),
        #     ('Aug25th/wf_260825_13483?_*.npz', '-2 dB'),
        #     ('Aug25th/wf_260825_14311?_*.npz', '0 dB'),
        #     ('Aug25th/wf_260825_16065?_*.npz', '-2 dB\n(direct)'),
        # ]
        # patterns_labels = [ # RF-power scan at (x,z) = (5.10, 6.30) mm
        #     ('Aug25th/wf_260825_14363?_*.npz', '-5 dB'),
        #     ('Aug25th/wf_260825_13444?_*.npz', '-2 dB'),
        #     ('Aug25th/wf_260825_14352?_*.npz', '0 dB'),
        #     ('Aug25th/wf_260825_16055?_*.npz', '-2 dB\n(direct)'),
        # ]
        # patterns_labels = [ # RF-power scan at (x,z) = (4.60, 6.00) mm
        #             ('Aug25th/wf_260825_14220?_*.npz', '-5 dB'),
        #             ('Aug25th/wf_260825_13143?_*.npz', '-2 dB'),
        #             ('Aug25th/wf_260825_14213?_*.npz', '0 dB'),
        #             ('Aug25th/wf_260825_16030?_*.npz', '-2 dB\n(direct)'),
        #         ]
        # patterns_labels = [ # Laser-power scan at (x,z) = (5.10, 6.70) mm
        #     ('Aug25th/wf_260825_15414?_*.npz', '1'),
        #     ('Aug25th/wf_260825_15422?_*.npz', '1/10'),
        #     ('Aug25th/wf_260825_15441?_*.npz', '1/40'),
        #     ('Aug25th/wf_260825_13483?_*.npz', '1/100'),
        #     ('Aug25th/wf_260825_15454?_*.npz', '1/200'),
        #     ('Aug25th/wf_260825_14322?_*.npz', '1/400'),
        #     ('Aug25th/wf_260825_15462?_*.npz', '1/1000'),
        # ]
        # patterns_labels = [ # Laser-power scan at (x,z) = (5.10, 6.30) mm
        #     ('Aug25th/wf_260825_15205?_*.npz', '1'),
        #     ('Aug25th/wf_260825_15213?_*.npz', '1/2'),
        #     ('Aug25th/wf_260825_15222?_*.npz', '1/4'),
        #     ('Aug25th/wf_260825_15233?_*.npz', '1/8'),
        #     ('Aug25th/wf_260825_15241?_*.npz', '1/10'),
        #     ('Aug25th/wf_260825_15252?_*.npz', '1/20'),
        #     ('Aug25th/wf_260825_15262?_*.npz', '1/40'),
        #     ('Aug25th/wf_260825_13444?_*.npz', '1/100'),
        #     ('Aug25th/wf_260825_15271?_*.npz', '1/200'),
        #     ('Aug25th/wf_260825_14363?_*.npz', '1/400'),
        #     ('Aug25th/wf_260825_15280?_*.npz', '1/1000'),
        # ]
        # patterns_labels = [ # Laser-power scan at (x,z) = (5.10, 6.00) mm
        #     ('Aug25th/wf_260825_15093?_*.npz', '1'),
        #     ('Aug25th/wf_260825_15101?_*.npz', '1/2'),
        #     ('Aug25th/wf_260825_15105?_*.npz', '1/4'),
        #     ('Aug25th/wf_260825_15143?_*.npz', '1/8'),
        #     ('Aug25th/wf_260825_15155?_*.npz', '1/10'),
        #     ('Aug25th/wf_260825_15165?_*.npz', '1/20'),
        #     ('Aug25th/wf_260825_15174?_*.npz', '1/40'),
        #     ('Aug25th/wf_260825_13421?_*.npz', '1/100'),
        #     ('Aug25th/wf_260825_15184?_*.npz', '1/200'),
        #     ('Aug25th/wf_260825_14401?_*.npz', '1/400'),
        #     ('Aug25th/wf_260825_15193?_*.npz', '1/1000'),
        # ]
        # patterns_labels = [ # RF-power scan at (x,z) = (5.10, 6.70) mm
        #             ('Aug25th/wf_260825_17222?_*.npz', '5.71 K'),
        #             ('Aug25th/wf_260825_16442?_*.npz', '5.22 K'),
        #             ('Aug25th/wf_260825_13483?_*.npz', '4.60 K'),
        #         ]
        # patterns_labels = [ # RF-power scan at (x,z) = (5.10, 6.30) mm
        #             ('Aug25th/wf_260825_17210?_*.npz', '5.71 K'),
        #             ('Aug25th/wf_260825_16465?_*.npz', '5.22 K'),
        #             ('Aug25th/wf_260825_13444?_*.npz', '4.60 K'),
        #         ]
        patterns_labels = [ # z-scan at x = 3.65 mm
                            ('/Volumes/NO NAME/20260831/data_0831_154637/wf_260831_15463?_*.npz', '6.70 mm'),
                            ('/Volumes/NO NAME/20260831/data_0831_154548/wf_260831_15454?_*.npz', '6.30 mm'),
                            ('/Volumes/NO NAME/20260831/data_0831_154505/wf_260831_15450?_*.npz', '6.00 mm')
                        ]
        # patterns_labels = [ # z-scan at x = 3.5 mm
        #             ('/Volumes/NO NAME/20260831/data_0831_154744/wf_260831_15474?_*.npz', '6.90 mm'),
        #             ('/Volumes/NO NAME/20260831/data_0831_154822/wf_260831_15482?_*.npz', '6.70 mm'),
        #             ('/Volumes/NO NAME/20260831/data_0831_154900/wf_260831_15490?_*.npz', '6.50 mm'),
        #             ('/Volumes/NO NAME/20260831/data_0831_154941/wf_260831_15494?_*.npz', '6.30 mm'),
        #             ('/Volumes/NO NAME/20260831/data_0831_155018/wf_260831_15502?_*.npz', '6.00 mm'),
        #             ('/Volumes/NO NAME/20260831/data_0831_155146/wf_260831_15514?_*.npz', '5.70 mm'),
        #             ('/Volumes/NO NAME/20260831/data_0831_162034/wf_260831_16203?_*.npz', '5.50 mm')
        #         ]


    params = {
        'proj_max_t': 'proj max t [samples]',
        'proj_rise': 'proj max t - proj half max left [samples]',
        'proj_fall': 'proj half max right - proj max t [samples]',
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
            dfres, data_corr, vc_proj, tbin, sample_rate, nwf = analyze_waveforms(fn_)
            dfres.to_csv(fn_[:-4]+'_ana.csv',index=False)
            dfs.append(dfres)
            ch0_corr_list.append(data_corr['ch0'] - dfres['ch0_ped'].values.reshape(-1,1))
            ch1_corr_list.append(data_corr['ch1'] - dfres['ch1_ped'].values.reshape(-1,1))
            vc_proj_list.append(vc_proj)
        dfgroup = pd.concat(dfs, ignore_index=True)
        dfgroup['proj_rise'] = dfgroup['proj_max_t'] - dfgroup['proj_half_max_left']
        dfgroup['proj_fall'] = dfgroup['proj_half_max_right'] - dfgroup['proj_max_t']
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

        binmaxt = np.linspace(tbin.min(),tbin.max(),100)
        ax = axs[0,0]
        ax.hist(tbin[dfgroup['proj_max_t'].to_numpy()], bins=binmaxt, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)

        bininteg = np.arange(0,50.1,0.2)
        ax = axs[0,1]
        ax.hist((dfgroup['proj_integ']), bins=bininteg, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)

        bintau = np.linspace(0,0.8,100)
        ax = axs[1,0]
        ax.hist(dfgroup['proj_rise'].to_numpy()/sample_rate*1e6, bins=bintau, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)
        ax = axs[1,1]
        ax.hist(dfgroup['proj_fall'].to_numpy()/sample_rate*1e6, bins=bintau, histtype='step', lw=1.5, color=mpl.cm.jet(idx/len(davg)), label=label)

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
    for idx, (label, avg) in enumerate(davg.items()):
        tbin = dtbin[label]
        color = mpl.cm.jet(idx / len(davg))

        axs5[0].plot(tbin, avg['ch0'] * 1e3, c=color, label=label)
        axs5[1].plot(tbin, avg['ch1'] * 1e3, c=color, label=label)
        axs5[2].plot(tbin, avg['proj'] * 1e3, c=color, label=label)

        one_over_e = _right_one_over_e_point(avg['proj'])
        if one_over_e is not None:
            point_idx, point_value = one_over_e
            axs5[2].plot(
                tbin[point_idx],
                point_value * 1e3,
                marker='o',
                ms=7,
                mfc='none',
                mec=color,
                mew=1.5,
            )
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

    ##### Average waveform comparison
    fig6,axs6 = plt.subplots(figsize=(18,5),ncols=3, nrows=1,sharex=True,sharey=True)
    for idx,(label,avg) in enumerate(davg.items()):
        tbin = dtbin[label]
        axs6[0].plot(tbin, avg['ch0']/np.max(avg['ch0']), c=mpl.cm.jet(idx/len(davg)), label=label)
        axs6[1].plot(tbin, avg['ch1']/np.max(avg['ch1']), c=mpl.cm.jet(idx/len(davg)), label=label)
        axs6[2].plot(tbin, avg['proj']/np.max(avg['proj']), c=mpl.cm.jet(idx/len(davg)), label=label)
        one_over_e = _right_one_over_e_point(avg['proj'])
        if one_over_e is not None:
            point_idx, point_value = one_over_e
            axs6[2].plot(
                tbin[point_idx],
                point_value / np.max(avg['proj']),
                marker='o',
                ms=7,
                mfc='none',
                mec=color,
                mew=1.5,
    )

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


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == 'compare':
        main_compare_distance()
    elif len(sys.argv)>1 and '.npz' in sys.argv[1]:
        main_singlefile(sys.argv[1])
    else:
        main()
