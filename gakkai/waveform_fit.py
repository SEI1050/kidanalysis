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

filename = sys.argv[1]
BIN_SIZE = 1

def funcdex(X_, tau_d, tau_r):
    tmax = np.log(tau_r/tau_d)/(1/tau_d-1/tau_r)
    return np.where(X_>=0,(np.exp(-X_/tau_d)-np.exp(-X_/tau_r))/np.abs(np.exp(-tmax/tau_d)-np.exp(-tmax/tau_r)),0)

def funcfit(X_, t0_, k_, tau_d, tau_r, ped_):
    return k_*funcdex(X_-t0_,tau_d,tau_r) + ped_


data = np.load(filename,allow_pickle=True)
print('number of points =',data['npts'])
print('number of waveforms =',data['ch0'].shape[0],data['ch1'].shape[0])

sample_rate = data['sample_rate']
npts_raw = data['npts']
ref_position = data['ref_position']
nwf = data['ch1'].shape[0]
nbin = npts_raw // BIN_SIZE
usable_npts = nbin * BIN_SIZE
ch0 = data['ch0'][:, :usable_npts].reshape(nwf, nbin, BIN_SIZE).mean(axis=2)
ch1 = data['ch1'][:, :usable_npts].reshape(nwf, nbin, BIN_SIZE).mean(axis=2)
npts = nbin
tbin_raw = (np.arange(usable_npts) - npts_raw*ref_position/100)/sample_rate
tbin = tbin_raw.reshape(nbin, BIN_SIZE).mean(axis=1)
probe_index = min(npts - 1, int((500 + npts_raw*ref_position/100)/BIN_SIZE))
deltat = (np.array(data['deltat'],dtype='timedelta64[ms]')/np.timedelta64(1,'ms')).astype(int)

print(sample_rate,npts,ref_position)
print(tbin)

lpopt = []
for idx in range(nwf):
    ## ch0
    ped0 = ch0[idx][:(int)(npts*(ref_position+10)/100)].mean()
    pol = np.sign(ch0[idx][probe_index] - ped0)
    k0 = ch0[idx].max()-ped0 if pol>0 else ch0[idx].min()-ped0
    p0 = [0, k0, 500, 50, ped0]
    bounds = ([-300, 0.1*k0, 0, 0, -np.inf],[500, 5*k0, 2000, 500, np.inf]) if pol>0 else ([-300, 5*k0, 0, 0, -np.inf],[500, 0.1*k0, 2000, 500, np.inf])
    # print('ch0',pol,p0, bounds)
    try:
        popt0,pcov0 = curve_fit(funcfit, tbin*1e9, ch0[idx], p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        popt0 = [0,0,0,0,0]
        pcov0 = [0 for _ in range(5) for _ in range(5)]
    # print(popt0)

    ## ch1
    ped0 = ch1[idx][:(int)(npts*(ref_position-5)/100)].mean()
    pol = np.sign(ch1[idx][probe_index] - ped0)
    k0 = ch1[idx].max()-ped0 if pol>0 else ch1[idx].min()-ped0
    p0 = [0, k0, 500, 50, ped0]
    bounds = ([-300, 0.1*k0, 0, 0, -np.inf],[500, 5*k0, 2000, 500, np.inf]) if pol>0 else ([-300, 5*k0, 0, 0, -np.inf],[500, 0.1*k0, 2000, 500, np.inf])
    # print('ch1',pol,p0, bounds)
    try:
        popt1,pcov1 = curve_fit(funcfit, tbin*1e9, ch1[idx], p0=p0, bounds=bounds, maxfev=10000)
        # popt1,pcov1 = curve_fit(funcfit, tbin*1e9, data['ch1'][idx], p0=[0,data['ch1'][idx].min(),500,20,lped[idx]], maxfev=10000, bounds=([-np.inf,-np.inf,0,0,-np.inf],[np.inf,np.inf,1000,200,np.inf]))
    except RuntimeError:
        popt1 = [0,0,0,0,0]
        pcov1 = [0 for _ in range(5) for _ in range(5)]
    # print(popt1)
    lpopt.append([popt0,popt1])

lfitpar = ['t0','k','tau','rise','ped']
lpopt = np.array(lpopt)
print(lpopt.shape)
df_fit = pd.DataFrame(lpopt.reshape(lpopt.shape[0],-1), columns = [f'ch{ich}_{itag}' for ich in range(2) for itag in lfitpar])

for ich in range(2):
    df_fit[f'ch{ich}_t0'] = df_fit[f'ch{ich}_t0']*1e-3
    df_fit[f'ch{ich}_k'] = df_fit[f'ch{ich}_k']*1e3
    df_fit[f'ch{ich}_tau'] = df_fit[f'ch{ich}_tau']*1e-3
    df_fit[f'ch{ich}_rise'] = df_fit[f'ch{ich}_rise']*1e-3
    df_fit[f'ch{ich}_ped'] = df_fit[f'ch{ich}_ped']*1e3
df_fit['absk'] = np.hypot(df_fit['ch0_k'].values,df_fit['ch1_k'].values)
print(df_fit)


pdf1 = PdfPages('pc1.pdf')

ncol = 4
nrow = 4 
ch0_range = (ch0.min(),ch0.max())
ch1_range = (ch1.min(),ch1.max())
fig,ax = plt.subplots(figsize=(16,9),ncols=ncol,nrows=nrow,sharex=True,sharey=True)
ax = ax.flatten()
for idx in range(min(ncol*nrow,nwf)):
    ax[idx].plot(tbin*1e9,ch0[idx],'-',label=f'Ch0',c='C0',alpha=0.7)
    ax[idx].plot(tbin*1e9,funcfit(tbin*1e9,*lpopt[idx][0]),'-',c='k')
    ax_r = ax[idx].twinx()
    ax_r.plot(tbin*1e9,ch1[idx],'-',label=f'Ch1',c='C1',alpha=0.7)
    ax_r.plot(tbin*1e9,funcfit(tbin*1e9,*lpopt[idx][1]),'-',c='k')
    ax[idx].grid()
    # ax[idx].legend()
    ax[idx].tick_params(axis='y',colors='C0')
    ax_r.tick_params(colors='C1')
    ax[idx].set_ylim(ch0_range)
    ax_r.set_ylim(ch1_range)
    if idx%nrow!=ncol-1:
        ax_r.tick_params(labelright=False)
    
for ic in range(ncol):
    ax[ncol*(nrow-1)+ic].set_xlabel('Time [ns]')
for ir in range(nrow):
    ax[ir*ncol].set_ylabel('Voltage [V]')

fig.tight_layout()
fig.savefig(pdf1,format='pdf')


fig,ax = plt.subplots(figsize=(16,9),nrows=2,ncols=len(popt0),sharex=False)
for idx,icol in enumerate(lfitpar):
    bins = np.linspace(min(df_fit[f'ch0_{icol}'].min(),df_fit[f'ch1_{icol}'].min()),max(df_fit[f'ch0_{icol}'].max(),df_fit[f'ch1_{icol}'].max()),50)
    for ich in range(2):
        ax[ich,idx].hist(df_fit[f'ch{ich}_{icol}'],bins=bins,color=f'C{ich}')
    ax[-1,idx].set_xlabel(icol)
for iax in ax.flatten():
    iax.grid()
fig.tight_layout()
fig.savefig(pdf1,format='pdf')

pdf1.close()

# resname = filename_.replace('.npz','_fitres.csv')
# df_fit.to_csv(resname)

