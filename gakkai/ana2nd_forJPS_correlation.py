import os
from posixpath import dirname
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


def _binned_quantiles(x, y, bins):
    centers = 0.5 * (bins[:-1] + bins[1:])
    q16 = np.full(len(centers), np.nan)
    q50 = np.full(len(centers), np.nan)
    q84 = np.full(len(centers), np.nan)
    count = np.zeros(len(centers), dtype=int)
    which = np.digitize(x, bins) - 1
    for i in range(len(centers)):
        values = y[(which == i) & np.isfinite(y)]
        count[i] = len(values)
        if len(values) >= 10:
            q16[i], q50[i], q84[i] = np.percentile(values, [16, 50, 84])
    return centers, q16, q50, q84, count


def make_noise_quality_report(csv_or_npz, output=None):
    """Create diagnostics separating estimator/noise effects from pulse structure.

    Run ana_forJPS.py first: this routine consumes its companion *_ana.csv.
    It deliberately calls the crossing observables "effective widths", not
    physical exponential time constants.
    """
    csv_fn = csv_or_npz[:-4] + '_ana.csv' if csv_or_npz.endswith('.npz') else csv_or_npz
    if output is None:
        output = os.path.splitext(csv_fn)[0] + '_noise_quality.pdf'
    df = pd.read_csv(csv_fn)
    required = {'proj_max', 'proj_integ_Vus', 'proj_snr', 'proj_spike_flag',
                'proj_e_max_left_interp', 'proj_e_max_right_interp',
                'proj_e_max_left_err', 'proj_e_max_right_err'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError('CSV lacks new quality columns: ' + ', '.join(sorted(missing)) +
                           '. Re-run ana_forJPS.py on the NPZ first.')

    rise = df['proj_e_max_left_interp'].to_numpy(float)
    fall = df['proj_e_max_right_interp'].to_numpy(float)
    rise_err = df['proj_e_max_left_err'].to_numpy(float)
    fall_err = df['proj_e_max_right_err'].to_numpy(float)
    finite = np.isfinite(rise) & np.isfinite(fall) & (rise > 0) & (fall > 0)
    clean = finite & ~df['proj_spike_flag'].astype(bool).to_numpy()
    relative20 = clean & (rise_err/rise < 0.20) & (fall_err/fall < 0.20)

    with PdfPages(output) as pdf, mpl.rc_context({
            'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
            'xtick.labelsize': 9, 'ytick.labelsize': 9,
            'legend.fontsize': 9, 'figure.titlesize': 13}):
        fig, axs = plt.subplots(2, 2, figsize=(14, 11))
        selections = [('all finite', finite), ('ringing removed', clean),
                      ('relative error <20%', relative20)]
        for ax, (label, mask) in zip(axs.flat[:3], selections):
            h = ax.hist2d(fall[mask], rise[mask], bins=80,
                          norm=mpl.colors.LogNorm(), cmap='viridis')
            fig.colorbar(h[3], ax=ax, label='events / bin')
            ax.set_title(f'{label}: N={mask.sum()}')
            ax.set_xlabel('peak-to-1/e fall width [us]')
            ax.set_ylabel('peak-to-1/e rise width [us]')
        ax = axs.flat[3]
        ax.hist(df.loc[finite, 'proj_snr'], bins=100, histtype='step', label='all')
        ax.hist(df.loc[clean, 'proj_snr'], bins=100, histtype='step', label='ringing removed')
        ax.set_xlabel('projected peak / pedestal MAD sigma')
        ax.set_ylabel('events')
        ax.set_yscale('log'); ax.legend(); ax.grid()
        fig.subplots_adjust(left=.09, right=.96, bottom=.08, top=.95,
                            hspace=.38, wspace=.32)
        pdf.savefig(fig); plt.close(fig)

        pairs = [('proj_max', 'projected peak [V]'),
                 ('proj_integ_Vus', 'projected integral [V us]')]
        for xcol, xlabel in pairs:
            x = df[xcol].to_numpy(float)
            fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
            for ax, y, ylabel in zip(axs, (rise, fall),
                                     ('peak-to-1/e rise width [us]',
                                      'peak-to-1/e fall width [us]')):
                mask = relative20 & np.isfinite(x)
                ax.scatter(x[mask], y[mask], s=2, alpha=0.10, color='0.25')
                if mask.sum() >= 20:
                    bins = np.quantile(x[mask], np.linspace(0, 1, 21))
                    bins = np.unique(bins)
                    if len(bins) > 2:
                        c, q16, q50, q84, n = _binned_quantiles(x[mask], y[mask], bins)
                        good = n >= 10
                        ax.plot(c[good], q50[good], color='tab:red', label='bin median')
                        ax.fill_between(c[good], q16[good], q84[good], color='tab:red',
                                        alpha=0.25, label='16-84%')
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(); ax.legend()
            fig.suptitle('Ringing removed; crossing relative errors <20%')
            fig.tight_layout(rect=(0, 0, 1, .95)); pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].scatter(df['proj_ped_mad_sigma']*1e3, df['proj_max']*1e3,
                       s=2, alpha=.15)
        axs[0].set(xlabel='pedestal MAD sigma [mV]', ylabel='peak [mV]')
        axs[1].scatter(df['proj_ped_mad_sigma']*1e3, rise, s=2, alpha=.15)
        axs[1].set(xlabel='pedestal MAD sigma [mV]', ylabel='rise width [us]')
        axs[2].scatter(df['proj_ped_mad_sigma']*1e3, fall, s=2, alpha=.15)
        axs[2].set(xlabel='pedestal MAD sigma [mV]', ylabel='fall width [us]')
        for ax in axs: ax.grid()
        fig.suptitle('Direct check for dependence on pedestal noise width')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f'wrote {output}')
    print(f'all={finite.sum()}, ringing_removed={clean.sum()}, relative_error_lt20={relative20.sum()}')
    return output


def _peek_npz_meta(fn_):
    """Read only sample_rate/npts/ref_position from a waveform .npz file.

    These are small scalars stored as separate members of the .npz (zip)
    archive, so accessing them does not require decompressing the much
    larger ch0/ch1 waveform arrays -- this is cheap even though it opens
    the original .npz rather than the companion csv.
    """
    with np.load(fn_, allow_pickle=True) as data:
        sample_rate = float(data['sample_rate'])
        npts = int(data['npts'])
        ref_position = float(data['ref_position'])
    tbin = (np.arange(npts) - npts * ref_position / 100) / sample_rate
    tbin *= 1e6
    return sample_rate, tbin


def load_group_from_csv(pattern, basedir):
    """Load the per-waveform parameter csv(s) matching `pattern` for one group.

    `pattern` is the same .npz glob pattern used in ana_forJPS.py's
    main_compare_distance(); for each matched <name>.npz we read the
    companion <name>_ana.csv written by that function (analyze_waveforms'
    dfres saved via dfres.to_csv(...)) instead of redoing the waveform
    analysis. All matched files are pooled into one DataFrame.

    Returns (dfgroup, sample_rate, tbin), or (None, None, None) if no
    companion csv was found for this pattern.
    """
    fns = sorted(glob.glob(os.path.join(basedir, pattern)))
    if len(fns) == 0:
        print(f'WARNING: no files matched pattern {pattern!r}')
        return None, None, None

    dfs = []
    sample_rate = None
    tbin = None
    for fn_ in fns:
        csv_fn = fn_[:-4] + '_ana.csv'
        if not os.path.exists(csv_fn):
            print(f'WARNING: {csv_fn!r} not found -- run ana_forJPS.py (compare mode) first -- skipping')
            continue

        dfs.append(pd.read_csv(csv_fn))

        if sample_rate is None:
            # only need this once per group; assumed constant across pooled files
            sample_rate, tbin = _peek_npz_meta(fn_)

    if len(dfs) == 0:
        return None, None, None

    dfgroup = pd.concat(dfs, ignore_index=True)
    return dfgroup, sample_rate, tbin


def main_compare_from_csv_laser(patterns_labels=None):
    """Compare proj_* waveform parameters across several runs, reading them
    back from the *_ana.csv files written by ana_forJPS.py instead of
    reprocessing the raw waveforms.

    patterns_labels -- list of (glob_pattern, label) pairs, same format as
    ana_forJPS.py's main_compare_distance(). Each pattern is globbed
    (relative to this script's directory) for .npz files; the companion
    '<name>_ana.csv' of each match is read and pooled under that label.

    Returns dgroups -- {label: dfgroup} per-waveform parameter DataFrame,
    with 'proj_rise'/'proj_fall' columns added.
    """
    if patterns_labels is None:
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
        patterns_labels = [ # Temp scan at (x,z) = (4.00, 6.30) mm
            ('Sep1st/wf_260901_14284?_*.npz', '3.90 K'),
            ('Sep1st/wf_260901_15450?_*.npz', '4.20 K'),
            ('Sep1st/wf_260901_16103?_*.npz', '4.55 K'),
            ('Sep1st/wf_260901_16461?_*.npz', '4.90 K'),
            ('Sep1st/wf_260901_17042?_*.npz', '5.30 K'),
            ('Sep1st/wf_260901_17300?_*.npz', '5.95 K'),
        ]
        # patterns_labels = [ # Temp scan at (x,z) = (4.00, 6.00) mm
        #     ('Sep1st/wf_260901_14232?_*.npz', '3.90 K'),
        #     ('Sep1st/wf_260901_15462?_*.npz', '4.20 K'),
        #     ('Sep1st/wf_260901_16082?_*.npz', '4.55 K'),
        #     ('Sep1st/wf_260901_16445?_*.npz', '4.90 K'),
        #     ('Sep1st/wf_260901_17055?_*.npz', '5.30 K'),
        #     ('Sep1st/wf_260901_17284?_*.npz', '5.95 K'),
        # ]
        # patterns_labels = [ # Temp scan at (x,z) = (3.50, 6.00) mm
        #     ('Sep1st/wf_260901_14560?_*.npz', '3.90 K'),
        #     ('Sep1st/wf_260901_15351?_*.npz', '4.20 K'),
        #     ('Sep1st/wf_260901_16164?_*.npz', '4.55 K'),
        #     ('Sep1st/wf_260901_16431?_*.npz', '4.90 K'),
        #     ('Sep1st/wf_260901_17072?_*.npz', '5.30 K'),
        #     ('Sep1st/wf_260901_17271?_*.npz', '5.95 K'),
        # ]
        # patterns_labels = [ # RF Freq scan at (x,z) = (4.00, 6.30) mm
        #     ('Sep1st/wf_260901_14322?_*.npz', '-40 MHz'),
        #     ('Sep1st/wf_260901_14315?_*.npz', '-30 MHz'),
        #     ('Sep1st/wf_260901_14313?_*.npz', '-20 MHz'),
        #     ('Sep1st/wf_260901_14310?_*.npz', '-10 MHz'),
        #     ('Sep1st/wf_260901_14284?_*.npz', '0 MHz'),
        #     ('Sep1st/wf_260901_14291?_*.npz', '+10 MHz'),
        #     ('Sep1st/wf_260901_14294?_*.npz', '+20 MHz'),
        #     ('Sep1st/wf_260901_14300?_*.npz', '+30 MHz'),
        #     ('Sep1st/wf_260901_14303?_*.npz', '+40 MHz'),
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

    basedir = os.path.dirname(os.path.abspath(__file__))
    dgroups = {}
    dtbin = {}
    dsample_rate = {}
    for pattern, label in patterns_labels:
        dfgroup, sample_rate, tbin = load_group_from_csv(pattern, basedir)
        if dfgroup is None:
            print(f'WARNING: nothing to load for pattern {pattern!r} -- skipping "{label}"')
            continue
        dgroups[label] = dfgroup
        dtbin[label] = tbin
        dsample_rate[label] = sample_rate

    fig, axs = plt.subplots(figsize=(15, 18), ncols=3, nrows=5)
    binped = np.linspace(-25, 25, 101)
    bintmax = np.arange(0, 0.2, 0.002)
    bintaur = np.arange(0, 0.451, 0.004)
    bintaud = np.arange(0, 1.51, 0.01)
    binq = np.linspace(0, 16.1, 101)
    binpeak = np.linspace(0, 40.1, 101)

    for jdx, (label, dfgroup) in enumerate(dgroups.items()):
        tbin = dtbin[label]
        sample_rate = dsample_rate[label]

        for idx,ich in enumerate(['ch0','ch1']):
            ax = axs[0, idx]
            hy,_ = np.histogram(dfgroup[f'{ich}_ped'].to_numpy()*1e3, bins=binped)
            ax.hist(binped[:-1], bins=binped, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

            ax = axs[1, idx]
            hy,_ = np.histogram(dfgroup[f'{ich}_peak_max_t'].to_numpy(), bins=bintmax)
            ax.hist(bintmax[:-1], bins=bintmax, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

            ax = axs[2, idx]
            hy,_ = np.histogram(dfgroup[f'{ich}_half_max_left'].to_numpy(), bins=bintaur)
            ax.hist(bintaur[:-1], bins=bintaur, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

            ax = axs[3, idx]
            hy,_ = np.histogram(dfgroup[f'{ich}_half_max_right'].to_numpy(), bins=bintaud)
            ax.hist(bintaud[:-1], bins=bintaud, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

            ax = axs[4, idx]
            hy,_ = np.histogram(dfgroup[f'{ich}_integ'].to_numpy(), bins=binq)
            ax.hist(binq[:-1], bins=binq, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

        ax = axs[1, 2]
        hy,_ = np.histogram(dfgroup['proj_max_t'].to_numpy(), bins=bintmax)
        ax.hist(bintmax[:-1], bins=bintmax, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

        ax = axs[2, 2]
        hy,_ = np.histogram(dfgroup['proj_half_max_left'].to_numpy(), bins=bintaur)
        ax.hist(bintaur[:-1], bins=bintaur, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

        ax = axs[3, 2]
        hy,_ = np.histogram(dfgroup[f'proj_half_max_right'].to_numpy(), bins=bintaud)
        ax.hist(bintaud[:-1], bins=bintaud, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

        ax = axs[4, 2]
        hy,_ = np.histogram(dfgroup[f'proj_integ'].to_numpy(), bins=binq)
        ax.hist(binq[:-1], bins=binq, weights=hy, histtype='step', lw=1.5, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

    for ich in range(2):
        axs[0, ich].set_xlabel(f'ch{ich} Pedestal [mV]')
        axs[1, ich].set_xlabel(f'ch{ich} Peak Time [$\mu$s]')
        axs[2, ich].set_xlabel(f'ch{ich} Half Time (Rise) [$\mu$s]')
        axs[3, ich].set_xlabel(f'ch{ich} Half Time (Decay) [$\mu$s]')
        axs[4, ich].set_xlabel(f'ch{ich} Integral')
    axs[1, 2].set_xlabel(f'proj Peak Time [$\mu$s]')
    axs[2, 2].set_xlabel(f'proj Half Time (Rise) [$\mu$s]')
    axs[3, 2].set_xlabel(f'proj Half Time (Decay) [$\mu$s]')
    axs[4, 2].set_xlabel(f'proj Integral')

    for ax in axs.flatten():
        ax.grid()
        ax.legend(ncols=2,fontsize='x-small')

    fig.tight_layout()
    fig.savefig('pc1_fromcsv.png')


    fig2,axs2 = plt.subplots(figsize=(13,10), ncols=3, nrows=3, sharex=True, sharey=True)
    fig3,axs3 = plt.subplots(figsize=(13,10), ncols=3, nrows=3, sharex=True, sharey=True)

    for jdx, (label,dfgroup) in enumerate(dgroups.items()):
        ax2 = axs2[jdx//3,jdx%3]
        ax3 = axs3[jdx//3,jdx%3]

        H,_,_ = np.histogram2d(dfgroup['proj_integ'],dfgroup['proj_half_max_left'],bins=(binq,bintaur))
        im = ax2.pcolormesh(*np.meshgrid(binq,bintaur),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet')
        # ax2.set_xlabel('proj Integral')
        # ax2.set_ylabel('proj Half Time (Rise) [$\mu$s]')
        ax2.set_title(label)
        ax2.grid()
        fig2.colorbar(im,ax=ax2)

        H,_,_ = np.histogram2d(dfgroup['proj_integ'],dfgroup['proj_half_max_right'],bins=(binq,bintaud))
        im = ax3.pcolormesh(*np.meshgrid(binq,bintaud),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet')
        # ax3.set_xlabel('proj Integral')
        # ax3.set_ylabel('proj Half Time (Decay) [$\mu$s]')
        ax3.set_title(label)
        ax3.grid()
        fig3.colorbar(im,ax=ax3)

    axs2[-1,1].set_xlabel('proj Integral')
    axs2[1,0].set_ylabel('proj Half Time (Rise) [$\mu$s]')
    axs3[-1,1].set_xlabel('proj Integral')
    axs3[1,0].set_ylabel('proj Half Time (Decay) [$\mu$s]')

    fig2.tight_layout()
    fig3.tight_layout()
    fig2.savefig('pc2_fromcsv.png')
    fig3.savefig('pc3_fromcsv.png')


    fig4,axs4 = plt.subplots(figsize=(16,10), ncols=3, nrows=2)
    for jdx, (label,dfgroup) in enumerate(dgroups.items()):
        ax = axs4[0,0]
        ax.plot(dfgroup['proj_half_max_left'], dfgroup['proj_e_max_right'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
        ax = axs4[0,1]
        ax.plot(dfgroup['proj_integ'], dfgroup['proj_e_max_left'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
        ax = axs4[0,2]
        ax.plot(dfgroup['proj_integ'], dfgroup['proj_e_max_right'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

        ax = axs4[1,0]
        ax.plot(dfgroup['ch1_ped']*1e3, dfgroup['proj_integ'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
        ax = axs4[1,1]
        ax.plot(dfgroup['ch1_ped']*1e3, dfgroup['proj_e_max_left'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
        ax = axs4[1,2]
        ax.plot(dfgroup['ch1_ped']*1e3, dfgroup['proj_e_max_right'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

    axs4[0,0].set_xlabel(r'$\tau_{r}$ [$\mu$s]')
    axs4[0,0].set_ylabel(r'$\tau_{d}$ [$\mu$s]')
    axs4[0,0].set_xlim(bintaur[0], bintaur[-1])
    axs4[0,0].set_ylim(bintaud[0], bintaud[-1])
    axs4[0,1].set_xlabel('proj Integral')
    axs4[0,1].set_ylabel(r'$\tau_{r}$ [$\mu$s]')
    axs4[0,1].set_xlim(binq[0], binq[-1])
    axs4[0,1].set_ylim(bintaur[0], bintaur[-1])
    axs4[0,2].set_xlabel('proj Integral')
    axs4[0,2].set_ylabel(r'$\tau_{d}$ [$\mu$s]')
    axs4[0,2].set_xlim(binq[0], binq[-1])
    axs4[0,2].set_ylim(bintaud[0], bintaud[-1])

    axs4[1,0].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,0].set_ylabel('proj Integral')
    axs4[1,0].set_xlim(binped[0], binped[-1])
    axs4[1,0].set_ylim(binq[0], binq[-1])
    axs4[1,1].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,1].set_ylabel(r'$\tau_{r}$ [$\mu$s]')
    axs4[1,1].set_xlim(binped[0], binped[-1])
    axs4[1,1].set_ylim(bintaur[0], bintaur[-1])
    axs4[1,2].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,2].set_ylabel(r'$\tau_{d}$ [$\mu$s]')
    axs4[1,2].set_xlim(binped[0], binped[-1])
    axs4[1,2].set_ylim(bintaud[0], bintaud[-1])

    for ax in axs4.flatten():
        ax.grid()
        # the scatter points use ms=1, alpha=0.2 for density; without overriding
        # the legend swatches they inherit the same tiny/faint marker and are
        # basically invisible, so boost size + opacity for the legend only
        leg = ax.legend(ncols=2, fontsize='x-small', markerscale=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    fig4.tight_layout()
    fig4.savefig('pc4_fromcsv.png')

    # fig5,axs5 = plt.subplots(figsize=(6,5))
    # ax = axs5
    # for jdx, (label,dfgroup) in enumerate(dgroups.items()):
    #     ax.plot(dfgroup['proj_half_max_left'], dfgroup['proj_max_t'], 'o', ms=1, alpha=0.2, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
    # ax.set_xlabel('proj Half Time (Rise) [$\mu$s]')
    # ax.set_ylabel('proj Peak Time [$\mu$s]')
    # ax.grid()
    # leg = ax.legend(ncols=2, fontsize='x-small', markerscale=10)
    # for lh in leg.legend_handles:
    #     lh.set_alpha(1)
    # fig5.tight_layout()
    # fig5.savefig('pc5_fromcsv.png')
    fig5, axs5 = plt.subplots(figsize=(16,10), ncols=3, nrows=2)
    for idx,(itau,ibin) in enumerate(zip(['proj_e_max_left','proj_e_max_right'],[bintaur,bintaud])):
        for jdx, (label,dfgroup) in enumerate(dgroups.items()):
            ax = axs5[idx,0]
            ax.plot(dfgroup['proj_integ'], dfgroup[itau], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
            ax = axs5[idx,1]
            ax.plot(dfgroup['proj_max']*1e3, dfgroup[itau], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
            ax = axs5[idx,2]
            ax.plot(dfgroup['ch0_peak_max']*1e3, dfgroup[itau], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
        ax = axs5[idx,0]
        ax.set_xlim(binq[0], binq[-1])
        ax.set_ylim(ibin[0], ibin[-1])
        ax.set_xlabel('Proj Integral')
        ax.set_ylabel(r'$\tau_{r}$ [$\mu$s]' if idx==0 else r'$\tau_{d}$ [$\mu$s]')
        ax = axs5[idx,1]
        ax.set_xlim(binpeak[0], binpeak[-1])
        ax.set_ylim(ibin[0], ibin[-1])
        ax.set_xlabel('Proj Peak Amplitude [mV]')
        ax.set_ylabel(r'$\tau_{r}$ [$\mu$s]' if idx==0 else r'$\tau_{d}$ [$\mu$s]')
        ax = axs5[idx,2]
        ax.set_xlim(binpeak[0], binpeak[-1])
        ax.set_ylim(ibin[0], ibin[-1])
        ax.set_xlabel('ch0 Peak Amplitude [mV]')
        ax.set_ylabel(r'$\tau_{r}$ [$\mu$s]' if idx==0 else r'$\tau_{d}$ [$\mu$s]')
    for ax in axs5.flatten():
        ax.grid()
        leg = ax.legend(ncols=2, fontsize='x-small', markerscale=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    fig5.tight_layout()
    fig5.savefig('pc5_fromcsv.png')


    fig6,axs6 = plt.subplots(figsize=(16,7),ncols=2, nrows=1)
    for jdx, (label,dfgroup) in enumerate(dgroups.items()):
        ax = axs6[0]
        ax.plot(dfgroup['proj_integ'], dfgroup['proj_max']*1e3, 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
        ax = axs6[1]
        ax.plot(dfgroup['proj_e_max_right'], dfgroup['proj_e_max_left'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
    axs6[0].set_xlabel('proj Integral')
    axs6[0].set_ylabel('proj Peak Amplitude [mV]')
    axs6[1].set_xlabel(r'$\tau_{d}$ [$\mu$s]')
    axs6[1].set_ylabel(r'$\tau_{r}$ [$\mu$s]')
    axs6[0].set_xlim(binq[0], binq[-1])
    axs6[0].set_ylim(binpeak[0], binpeak[-1])
    axs6[1].set_xlim(bintaud[0], bintaud[-1])
    axs6[1].set_ylim(bintaur[0], bintaur[-1])


    # fig6,axs6 = plt.subplots(figsize=(16,10),ncols=3, nrows=2)
    # for jdx, (label,dfgroup) in enumerate(dgroups.items()):
    #     ax = axs6[0,0]
    #     ax.plot(dfgroup['proj_integ_left'], dfgroup['proj_e_max_left'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
    #     ax = axs6[0,1]
    #     ax.plot(dfgroup['proj_integ_right'], dfgroup['proj_e_max_right'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
    #     ax = axs6[1,0]
    #     ax.plot(dfgroup['proj_integ_left'], dfgroup['ch1_ped']*1e3, 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)
    #     ax = axs6[1,1]
    #     ax.plot(dfgroup['proj_integ_right'], dfgroup['ch1_ped']*1e3, 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

    #     ax = axs6[0,2]
    #     ax.plot(dfgroup['proj_integ_right'], dfgroup['proj_integ_left'], 'o', ms=1, alpha=0.8, color=mpl.cm.jet(jdx / len(dgroups)), label=label)

    # axs6[0,0].set_xlabel('proj Integral (Left)')
    # axs6[0,0].set_ylabel('proj Half Time (Rise) [$\mu$s]')
    # axs6[0,1].set_xlabel('proj Integral (Right)')
    # axs6[0,1].set_ylabel('proj Half Time (Decay) [$\mu$s]')
    # axs6[1,0].set_xlabel('proj Integral (Left)')
    # axs6[1,0].set_ylabel('ch1 Pedestal [mV]')
    # axs6[1,1].set_xlabel('proj Integral (Right)')
    # axs6[1,1].set_ylabel('ch1 Pedestal [mV]')
    # axs6[0,2].set_xlabel('proj Integral (Right)')
    # axs6[0,2].set_ylabel('proj Integral (Left)')

    for ax in axs6.flatten():
        ax.grid()
        leg = ax.legend(ncols=2, fontsize='x-small', markerscale=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    fig6.tight_layout()
    fig6.savefig('pc6_fromcsv.png')

    plt.show()

    return dgroups


def _to_datetime(t):
    """Coerce a 'YYMMDD_HHMMSS' string (matching the wf_*.npz filename
    timestamp format, e.g. '260827_190000') or a datetime.datetime into a
    datetime.datetime. None passes through unchanged (unbounded)."""
    if t is None or isinstance(t, datetime.datetime):
        return t
    return datetime.datetime.strptime(t, '%y%m%d_%H%M%S')


def main_compare_from_csv_alpha(dirname='Aug27th', t_start=None, t_end=None):
    if t_start is None:
        t_start = '260827_190000'
    if t_end is None:
        t_end = '260828_130100'

    t_start = _to_datetime(t_start)
    t_end = _to_datetime(t_end)

    basedir = os.path.dirname(os.path.abspath(__file__))
    scandir = dirname if os.path.isabs(dirname) else os.path.join(basedir, dirname)

    fns = sorted(glob.glob(os.path.join(scandir, 'wf_*.npz')))
 
    basedir = os.path.dirname(os.path.abspath(__file__))
    dgroups = {}
    dtbin = {}
    dsample_rate = {}
    for fn_ in fns:
        dfgroup, sample_rate, tbin = load_group_from_csv(fn_, basedir)
        if dfgroup is None:
            print(f'WARNING: nothing to load for file {fn_!r} -- skipping')
            continue
        dgroups[fn_] = dfgroup
        dtbin[fn_] = tbin
        dsample_rate[fn_] = sample_rate

    dfall = pd.concat(dgroups.values(), ignore_index=True)

    # cut = f'ch0_peak_max>0.5e-3 and proj_integ>1 and ch0_peak_max_t>-0.3'
    # cut = f'ch1_peak_max>0.5e-3'
    # dfall = dfall.query(cut)

    print(dfall)
    print(len(dfall))
    dfall['ch0_ped'] = dfall['ch0_ped']*1e3
    dfall['ch1_ped'] = dfall['ch1_ped']*1e3
    dfall['ch0_peak_max'] = dfall['ch0_peak_max']*1e3
    dfall['ch1_peak_max'] = dfall['ch1_peak_max']*1e3
    dfall['proj_max'] = dfall['proj_max']*1e3

    dfall['corr_integ'] = dfall['proj_integ'] + 0.06*dfall['ch1_ped'] 
    dfall['corr_max'] = dfall['proj_max'] + 0.065*dfall['ch1_ped'] 


    binped = np.linspace(-25, 25, 101)
    bintmax = np.arange(-0.1, 0.3, 0.004)
    bintaur = np.arange(0, 0.451, 0.004)
    bintaud = np.arange(0, 1.51, 0.01)
    binq = np.linspace(0,16.1,101)
    binpeak = np.linspace(0, 80.1, 101)

    fig,axs = plt.subplots(figsize=(16,10), ncols=3, nrows=2)
    axs = axs.flatten()
    vals = ['ped', 'peak_max_t', 'peak_max', 'half_max_left', 'half_max_right', 'integ']
    vals_proj = ['', 'proj_max_t', 'proj_max', 'proj_half_max_left', 'proj_half_max_right', 'proj_integ']
    bins = ['binped', 'bintmax', 'binpeak', 'bintaur', 'bintaud', 'binq']


    for idx,vals in enumerate(vals):
        ax = axs[idx]
        for ich in range(2):
            hy,_ = np.histogram(dfall[f'ch{ich}_{vals}'].to_numpy(), bins=locals()[bins[idx]])
            ax.hist(locals()[bins[idx]][:-1], bins=locals()[bins[idx]], weights=hy, histtype='step', lw=1.5, label=f'ch{ich}')
        if vals_proj[idx] != '':
            hy,_ = np.histogram(dfall[vals_proj[idx]].to_numpy(), bins=locals()[bins[idx]])
            ax.hist(locals()[bins[idx]][:-1], bins=locals()[bins[idx]], weights=hy, histtype='step', lw=1.5, label=f'proj')
    axs[0].set_xlabel('Pedestal [mV]')
    axs[1].set_xlabel('Peak Time [$\mu$s]')
    axs[2].set_xlabel('Peak Amplitude [mV]')
    axs[3].set_xlabel('Half Time (Rise) [$\mu$s]')
    axs[4].set_xlabel('Half Time (Decay) [$\mu$s]')
    axs[5].set_xlabel('Integral')
    for ax in axs:
        ax.grid()
        ax.legend(ncols=2,fontsize='x-small')

    fig.tight_layout()
    fig.savefig('pc1_fromcsv_alpha.png')


    fig4, axs4 = plt.subplots(figsize=(20,10), ncols=4, nrows=2)
    ax = axs4[0,0]
    H,_,_ = np.histogram2d(dfall['proj_half_max_left'], dfall['proj_half_max_right'], bins=(bintaur,bintaud))
    im = ax.pcolormesh(*np.meshgrid(bintaur,bintaud),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax)
    ax = axs4[0,1]
    H,_,_ = np.histogram2d(dfall['proj_integ'], dfall['proj_half_max_left'], bins=(binq,bintaur))
    im = ax.pcolormesh(*np.meshgrid(binq,bintaur),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax) 
    ax = axs4[0,2]
    H,_,_ = np.histogram2d(dfall['proj_integ'], dfall['proj_half_max_right'], bins=(binq,bintaud))
    im = ax.pcolormesh(*np.meshgrid(binq,bintaud),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax)
    ax = axs4[0,3]
    H,_,_ = np.histogram2d(dfall['proj_integ'], dfall['proj_max'], bins=(binq,binpeak))
    im = ax.pcolormesh(*np.meshgrid(binq,binpeak),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax) 
    ax = axs4[1,0]
    H,_,_ = np.histogram2d(dfall['ch1_ped'], dfall['proj_integ'], bins=(binped,binq))
    im = ax.pcolormesh(*np.meshgrid(binped,binq),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax) 
    ax = axs4[1,1]
    H,_,_ = np.histogram2d(dfall['ch1_ped'], dfall['proj_half_max_left'], bins=(binped,bintaur))
    im = ax.pcolormesh(*np.meshgrid(binped,bintaur),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax)
    ax = axs4[1,2]
    H,_,_ = np.histogram2d(dfall['ch1_ped'], dfall['proj_half_max_right'], bins=(binped,bintaud))
    im = ax.pcolormesh(*np.meshgrid(binped,bintaud),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax)
    ax = axs4[1,3]
    H,_,_ = np.histogram2d(dfall['ch1_ped'], dfall['proj_max'], bins=(binped,binpeak))
    im = ax.pcolormesh(*np.meshgrid(binped,binpeak),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig4.colorbar(im,ax=ax)

    axs4[0,0].set_xlabel('proj Half Time (Rise) [$\mu$s]')
    axs4[0,0].set_ylabel('proj Half Time (Decay) [$\mu$s]')
    axs4[0,0].set_xlim(bintaur[0], bintaur[-1])
    axs4[0,0].set_ylim(bintaud[0], bintaud[-1])
    axs4[0,1].set_xlabel('proj Integral')
    axs4[0,1].set_ylabel('proj Half Time (Rise) [$\mu$s]')
    axs4[0,1].set_xlim(binq[0], binq[-1])
    axs4[0,1].set_ylim(bintaur[0], bintaur[-1])
    axs4[0,2].set_xlabel('proj Integral')
    axs4[0,2].set_ylabel('proj Half Time (Decay) [$\mu$s]')
    axs4[0,2].set_xlim(binq[0], binq[-1])
    axs4[0,2].set_ylim(bintaud[0], bintaud[-1])
    axs4[0,3].set_xlabel('proj Integral')
    axs4[0,3].set_ylabel('proj Peak Amplitude [mV]')
    axs4[0,3].set_xlim(binq[0], binq[-1])
    axs4[0,3].set_ylim(binpeak[0], binpeak[-1])

    axs4[1,0].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,0].set_ylabel('proj Integral')
    axs4[1,0].set_xlim(binped[0], binped[-1])
    axs4[1,0].set_ylim(binq[0], binq[-1])
    axs4[1,1].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,1].set_ylabel('proj Half Time (Rise) [$\mu$s]')
    axs4[1,1].set_xlim(binped[0], binped[-1])
    axs4[1,1].set_ylim(bintaur[0], bintaur[-1])
    axs4[1,2].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,2].set_ylabel('proj Half Time (Decay) [$\mu$s]')
    axs4[1,2].set_xlim(binped[0], binped[-1])
    axs4[1,2].set_ylim(bintaud[0], bintaud[-1])
    axs4[1,3].set_xlabel('ch1 Pedestal [mV]')
    axs4[1,3].set_ylabel('proj Peak Amplitude [mV]')
    axs4[1,3].set_xlim(binped[0], binped[-1])
    axs4[1,3].set_ylim(binpeak[0], binpeak[-1])
    for ax in axs4.flatten():
        ax.grid()
    fig4.tight_layout()
    fig4.savefig('pc4_fromcsv_alpha.png')


    fig5, axs5 = plt.subplots(figsize=(16,10), ncols=3, nrows=2)
    for idx,(itau,ibin) in enumerate(zip(['proj_e_max_left','proj_e_max_right'],[bintaur,bintaud])):
        ax = axs5[idx,0]
        H,_,_ = np.histogram2d(dfall['proj_integ'], dfall[itau], bins=(binq,ibin))
        im = ax.pcolormesh(*np.meshgrid(binq,ibin),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
        fig5.colorbar(im,ax=ax)
        ax.set_xlim(binq[0], binq[-1])
        ax.set_ylim(ibin[0], ibin[-1])
        ax.set_xlabel('Proj Integral')
        ax.set_ylabel(r'$\tau_{r}$ [$\mu$s]' if idx==0 else r'$\tau_{d}$ [$\mu$s]')
        ax = axs5[idx,1]
        H,_,_ = np.histogram2d(dfall['proj_max'], dfall[itau], bins=(binpeak,ibin))
        im = ax.pcolormesh(*np.meshgrid(binpeak,ibin),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
        fig5.colorbar(im,ax=ax)
        ax.set_xlim(binpeak[0], binpeak[-1])
        ax.set_ylim(ibin[0], ibin[-1])
        ax.set_xlabel('Proj Peak Amplitude [mV]')
        ax.set_ylabel(r'$\tau_{r}$ [$\mu$s]' if idx==0 else r'$\tau_{d}$ [$\mu$s]')
        ax = axs5[idx,2]
        H,_,_ = np.histogram2d(dfall['ch0_peak_max'], dfall[itau], bins=(binpeak,ibin))
        im = ax.pcolormesh(*np.meshgrid(binpeak,ibin),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
        fig5.colorbar(im,ax=ax)
        ax.set_xlim(binpeak[0], binpeak[-1])
        ax.set_ylim(ibin[0], ibin[-1])
        ax.set_xlabel('ch0 peak Amplitude [mV]')
        ax.set_ylabel(r'$\tau_{r}$ [$\mu$s]' if idx==0 else r'$\tau_{d}$ [$\mu$s]')
    for ax in axs5.flatten():
        ax.grid()
    fig5.tight_layout()
    fig5.savefig('pc5_fromcsv_alpha.png')


    # fig6, axs6 = plt.subplots(figsize=(15,8), ncols=3, nrows=2)
    # ax = axs6[0,0]
    # H,_,_ = np.histogram2d(dfall['ch1_ped'], dfall['corr_integ'], bins=(binped,binq))
    # im = ax.pcolormesh(*np.meshgrid(binped,binq),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    # fig6.colorbar(im,ax=ax) 
    # ax = axs6[0,1]
    # H,_,_ = np.histogram2d(dfall['proj_integ'], dfall['corr_integ'], bins=(binq,binq))
    # im = ax.pcolormesh(*np.meshgrid(binq,binq),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    # fig6.colorbar(im,ax=ax)
    # ax = axs6[0,2]
    # hy,_ = np.histogram(dfall['proj_integ'], bins=binq)
    # ax.hist(binq[:-1], bins=binq, weights=hy, histtype='stepfilled', lw=1.5, alpha=0.65, label=f'Raw')
    # hy,_ = np.histogram(dfall['corr_integ'], bins=binq)
    # ax.hist(binq[:-1], bins=binq, weights=hy, histtype='stepfilled', lw=1.5, alpha=0.65, label=f'Corrected')
    # ax.legend(ncols=2,fontsize='x-small')   
    # ax = axs6[1,0]
    # H,_,_ = np.histogram2d(dfall['ch1_ped'], dfall['corr_max'], bins=(binped,binpeak))
    # im = ax.pcolormesh(*np.meshgrid(binped,binpeak),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    # fig6.colorbar(im,ax=ax) 
    # ax = axs6[1,1]
    # H,_,_ = np.histogram2d(dfall['proj_max'], dfall['corr_max'], bins=(binpeak,binpeak))
    # im = ax.pcolormesh(*np.meshgrid(binpeak,binpeak),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    # fig6.colorbar(im,ax=ax)
    # ax = axs6[1,2]
    # hy,_ = np.histogram(dfall['proj_max'], bins=binpeak)
    # ax.hist(binpeak[:-1], bins=binpeak, weights=hy, histtype='stepfilled', lw=1.5, alpha=0.65, label=f'Raw')
    # hy,_ = np.histogram(dfall['corr_max'], bins=binpeak)
    # ax.hist(binpeak[:-1], bins=binpeak, weights=hy, histtype='stepfilled', lw=1.5, alpha=0.65, label=f'Corrected')
    # ax.legend(ncols=2,fontsize='x-small')   

    # axs6[0,0].set_xlabel('ch1 Pedestal [mV]')
    # axs6[0,0].set_ylabel('Integral (Corrected)')
    # axs6[0,0].set_xlim(binped[0], binped[-1])
    # axs6[0,0].set_ylim(binq[0], binq[-1])
    # axs6[0,1].set_xlabel('Integral (Raw)')
    # axs6[0,1].set_ylabel('Integral (Corrected)')
    # axs6[0,1].set_xlim(binq[0], binq[-1])
    # axs6[0,1].set_ylim(binq[0], binq[-1])
    # axs6[0,2].set_xlabel('Integral')
    # axs6[0,2].set_xlim(binq[0], binq[-1])
    # axs6[1,0].set_xlabel('ch1 Pedestal [mV]')
    # axs6[1,0].set_ylabel('Peak Amplitude (Corrected)')
    # axs6[1,0].set_xlim(binped[0], binped[-1])
    # axs6[1,0].set_ylim(binpeak[0], binpeak[-1])
    # axs6[1,1].set_xlabel('Peak Amplitude (Raw)')
    # axs6[1,1].set_ylabel('Peak Amplitude (Corrected)')
    # axs6[1,1].set_xlim(binpeak[0], binpeak[-1])
    # axs6[1,1].set_ylim(binpeak[0], binpeak[-1])
    # axs6[1,2].set_xlabel('Peak Amplitude')
    # axs6[1,2].set_xlim(binpeak[0], binpeak[-1])

    fig6, axs6 = plt.subplots(figsize=(16,7), ncols=2, nrows=1)
    ax = axs6[0]
    H,_,_ = np.histogram2d(dfall['proj_integ'], dfall['proj_max'], bins=(binq,binpeak))
    im = ax.pcolormesh(*np.meshgrid(binq,binpeak),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig6.colorbar(im,ax=ax)
    ax = axs6[1]
    H,_,_ = np.histogram2d(dfall['proj_e_max_right'], dfall['proj_e_max_left'], bins=(bintaud,bintaur))
    im = ax.pcolormesh(*np.meshgrid(bintaud,bintaur),np.ma.masked_where(H.T==0, H.T),shading='auto',cmap='jet',norm=mpl.colors.LogNorm())
    fig6.colorbar(im,ax=ax)
    axs6[0].set_xlabel('proj Integral')
    axs6[0].set_ylabel('proj Peak Amplitude [mV]')
    axs6[0].set_xlim(binq[0], binq[-1])
    axs6[0].set_ylim(binpeak[0], binpeak[-1])
    axs6[1].set_xlabel(r'$\tau_{d}$ [$\mu$s]')
    axs6[1].set_ylabel(r'$\tau_{r}$ [$\mu$s]')
    axs6[1].set_xlim(bintaud[0], bintaud[-1])
    axs6[1].set_ylim(bintaur[0], bintaur[-1])
    for ax in axs6.flatten():
        ax.grid()
    fig6.tight_layout()
    fig6.savefig('pc6_fromcsv_alpha.png')
    
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'Laser':
        main_compare_from_csv_laser()
    elif len(sys.argv) > 1 and sys.argv[1] == 'Alpha':
        main_compare_from_csv_alpha()
    elif len(sys.argv) > 2 and sys.argv[1] == 'Quality':
        make_noise_quality_report(sys.argv[2])
 
