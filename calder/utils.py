import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle 
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as tick
import numpy as np
import pandas as pd
import math
import scipy
import os
from tqdm import tqdm
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import constants as const
from astropy.timeseries import LombScargle as ls

colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39", 
              "#ce5761", "#c68c45", '#b5b246', '#d77fcc', '#7362cf', '#ce443f', '#3fc1bf', '#cda735',
              '#a1b055']

def year_to_jd(year):
    jd_epoch = 2449718.5 - (2.458 * 10 **6)
    year_epoch = 1995
    days_in_year = 365.25
    return (year-year_epoch)*days_in_year + jd_epoch-2450000

def jd_to_year(jd):
    jd_epoch = 2449718.5 - (2.458 * 10 **6)
    year_epoch = 1995
    days_in_year = 365.25
    return year_epoch + (jd - jd_epoch) / days_in_year

def read_lc_dat(asassn_id, path):

    if os.path.exists(f"{path}/{asassn_id}.dat"):
        file = os.path.join(path, f"{asassn_id}.dat")
        # column names
        columns = ["JD", 
                   "mag", 
                   "error", 
                   "good_bad", 
                   "camera#", 
                   "v_g_band", 
                   "saturated", 
                   "cam_field"]

        # read fwf
        df = pd.read_fwf(
            file,
            header=None,
            names=columns
        )
    
        # split the "cam_field" column into two
        df[["camera_name", "field"]] = df["cam_field"].str.split("/", expand=True)

        # drop the old combined column
        df = df.drop(columns=["cam_field"])

        # enforce dtypes
        df = df.astype({
            "JD": "float64",
            "mag": "float64",
            "error": "float64",
            "good_bad": "int64",
            "camera#": "int64",
            "v_g_band": "int64",
            "saturated": "int64",
            "camera_name": "string",
            "field": "string"
        })

        df_g = df.loc[df["v_g_band"] == 0].reset_index(drop=True)    
        df_v = df.loc[df["v_g_band"] == 1].reset_index(drop=True)

        if df_v.empty:
            print(f"[warn] {asassn_id}: no V band rows")
        if df_g.empty:
            print(f"[warn] {asassn_id}: no g band rows")
        
    else:
        print(f"[error] {asassn_id}: file not found in {path}")
        df_g = pd.DataFrame()
        df_v = pd.DataFrame()
                 
    return df_g, df_v

def naive_peak_search(df, prominence=0.17, distance=25, height=0.3, width=2):

    mag = df["mag"]
    jd = df['JD']

    meanmag = sum(mag) / len(mag)
    df_mag_avg = [i - meanmag for i in mag]

    peaks = scipy.signal.find_peaks(df_mag_avg,
                                    prominence=prominence,
                                    distance=distance, 
                                    height=height, 
                                    width=width) 
    
    peak = peaks[0]
    prop = peaks[1]

    length = len(peak)
    peak = [int(i) for i in peak]
    peak = pd.Series(peak)

    return peak, meanmag, length

def custom_id(ra_val,dec_val):
    c = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree, frame='icrs')
    ra_num = c.ra.hms
    dec_num = c.dec.dms

    if int(dec_num[0]) < 0:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$-$'+str(int(c.dec.dms[0])*(-1)).rjust(2,'0')+str(int(c.dec.dms[1])*(-1)).rjust(2,'0')+str(int(round(c.dec.dms[2])*(-1))).rjust(2,'0')
    else:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$+$'+str(int(c.dec.dms[0])).rjust(2,'0')+str(int(c.dec.dms[1])).rjust(2,'0')+str(int(round(c.dec.dms[2]))).rjust(2,'0')

    return cust_id


def plot_multiband(dfv, dfg, ra, dec, peak_option=False):
    cust_id = custom_id(ra,dec)
    peak, meanmag, length = naive_peak_search(dfg)

    fig, ax = pl.subplots(1, 1, figsize=(8, 4))

    gcams = dfg["camera#"]
    gcamtype = np.unique(gcams)
    gcamnum = len(gcamtype)

    vcams = dfv["camera#"]
    vcamtype = np.unique(vcams)
    vcamnum = len(vcamtype)

    if max(dfg.Mag) < max(dfv.Mag):
        Max_mag = max(dfg.Mag)+0.2
    else:
        Max_mag = max(dfv.Mag)+0.2

    if min(dfg.Mag) < min(dfv.Mag):
        Min_mag = min(dfg.Mag)-0.4
    else:
        Min_mag = min(dfv.Mag)-0.4

    if peak_option == False:

        for i in range(0,gcamnum):
            gcam = dfg.loc[dfg["camera#"] == gcamtype[i]].reset_index(drop=True)
            gcamjd = gcam["JD"].astype(float) - (2.458 * 10 ** 6)
            gcammag = gcam["mag"].astype(float)
            ax.scatter(gcamjd, gcammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(0,vcamnum):
            vcam = dfv.loc[dfv["camera#"] == vcamtype[i]].reset_index(drop=True)
            vcamjd = vcam["JD"].astype(float) - (2.458 * 10 ** 6)
            vcammag = vcam["mag"].astype(float)
            ax.scatter(vcamjd, vcammag, color=colors[i], alpha=0.6, marker='.')

        ax.set_xlim((min(dfv.JD)-(2.458 * 10 ** 6)-500),(max(dfg.JD)-(2.458 * 10 ** 6)+150))
        ax.set_ylim(Min_mag,Max_mag)
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('V & g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=True, bottom=True, pad=-15, labelsize=12)
        ax.tick_params(axis='y', direction='in', right=False, pad=-35, labelsize=12)
        ax.tick_params('both', length=6, width=1.5, which='major')
        ax.tick_params('both', direction='in', length=4, width=1, which='minor')
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))
        secax = ax.secondary_xaxis('top', functions=(jd_to_year,year_to_jd))
        secax.xaxis.set_tick_params(direction='in', labelsize=12, pad=-18, length=6, width=1.5) 
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

    if peak_option == True:
        print('The mean g magnitude:', meanmag)
        print('The number of detected peaks:', length)

        for i in range(0,camnum):
            gcam = dfg.loc[dfg["camera#"] == gcamtype[i]].reset_index(drop=True)
            gcamjd = gcam["JD"].astype(float) - (2.458 * 10 ** 6)
            gcammag = gcam["mag"].astype(float)
            ax.scatter(gcamjd, gcammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(0,vcamnum):
            vcam = dfv.loc[dfv["camera#"] == vcamtype[i]].reset_index(drop=True)
            vcamjd = vcam["JD"].astype(float) - (2.458 * 10 ** 6)
            vcammag = vcam["mag"].astype(float)
            ax.scatter(vcamjd, vcammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(len(peak)-1):
            ax.vlines((dfg.JD[peak[i]] - (2.458 * 10**6)), (min(dfg["mag"])-0.1), (max(dfg["mag"])+0.1), "k", alpha=0.4)

        ax.set_xlim((min(dfv.JD)-(2.458 * 10 ** 6)-300),(max(df.JD)-(2.458 * 10 ** 6)+150))
        ax.set_ylim(Min_mag,Max_mag)
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=True, bottom=True, pad=-15, labelsize=12)
        ax.tick_params(axis='y', direction='in', right=False, pad=-35, labelsize=12)
        ax.tick_params('both', length=6, width=1.5, which='major')
        ax.tick_params('both', direction='in', length=4, width=1, which='minor')
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))
        secax = ax.secondary_xaxis('top', functions=(jd_to_year,year_to_jd))
        secax.xaxis.set_tick_params(direction='in', labelsize=12, pad=-18, length=6, width=1.5) 
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

def plot_light_curve(df, ra, dec, peak_option=False):
    cust_id = custom_id(ra,dec)
    peak, meanmag, length = naive_peak_search(df)

    fig, ax = pl.subplots(1, 1, figsize=(8, 4))

    cams = df["camera#"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    if peak_option == False:

        for i in range(0,camnum):
            cam = df.loc[df["camera#"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        ax.set_xlim((min(df.JD)-(2.458 * 10 ** 6)-300),(max(df.JD)-(2.458 * 10 ** 6)+150))
        ax.set_ylim((min(df["mag"])-0.1),(max(df["mag"])+0.1))
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=True, bottom=True, pad=-15, labelsize=12)
        ax.tick_params(axis='y', direction='in', right=False, pad=-35, labelsize=12)
        ax.tick_params('both', length=6, width=1.5, which='major')
        ax.tick_params('both', direction='in', length=4, width=1, which='minor')
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))
        secax = ax.secondary_xaxis('top', functions=(jd_to_year,year_to_jd))
        secax.xaxis.set_tick_params(direction='in', labelsize=12, pad=-18, length=6, width=1.5) 
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

    if peak_option == True:
        print('The mean magnitude:', meanmag)
        print('The number of detected peaks:', length)

        for i in range(0,camnum):
            cam = df.loc[df["camera#"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(len(peak)-1):
            ax.vlines((df.JD[peak[i]] - (2.458 * 10**6)), (min(df["mag"])-0.1), (max(df["mag"])+0.1), "k", alpha=0.4)

        ax.set_xlim((min(df.JD)-(2.458 * 10 ** 6)-300),(max(df.JD)-(2.458 * 10 ** 6)+150))
        ax.set_ylim((min(df["mag"])-0.1),(max(df["mag"])+0.1))
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=True, bottom=True, pad=-15, labelsize=12)
        ax.tick_params(axis='y', direction='in', right=False, pad=-35, labelsize=12)
        ax.tick_params('both', length=6, width=1.5, which='major')
        ax.tick_params('both', direction='in', length=4, width=1, which='minor')
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))
        secax = ax.secondary_xaxis('top', functions=(jd_to_year,year_to_jd))
        secax.xaxis.set_tick_params(direction='in', labelsize=12, pad=-18, length=6, width=1.5) 
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

def plotparams(ax, labelsize=15):

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax

def plot_zoom(df, ra, dec, zoom_range=[-300,3000], peak_option=False):

    cust_id = custom_id(ra,dec)
    peak, meanmag, length = naive_peak_search(df)

    fig, ax = pl.subplots(1, 1, figsize=(10, 4))
    ax = plotparams(ax)

    cams = df["camera#"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    if peak_option == False:

        for i in range(0,camnum):
            cam = df.loc[df["camera#"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        ax.set_xlim(zoom_range[0],zoom_range[1])
        ax.set_ylim((min(df["mag"])-0.1),(max(df["mag"])+0.1))
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()

    if peak_option == True:

        for i in range(0,camnum):
            cam = df.loc[df["camera#"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(len(peak)-1):
            ax.vlines((df.JD[peak[i]] - (2.458 * 10**6)), (min(df["mag"])-0.1), (max(df["mag"])+0.1), "k", alpha=0.4)

        ax.set_xlim(zoom_range[0],zoom_range[1])
        ax.set_ylim((min(df["mag"])-0.1),(max(df["mag"])+0.1))
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()

