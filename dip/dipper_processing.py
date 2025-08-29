import matplotlib.pyplot as plt
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


def plotparams(ax, labelsize=15):
    '''
    Basic plot params

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax

def read_lightcurve_dat(asas_sn_id, guide = 'known_dipper_lightcurves/'):
    """
    Input: 
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        dfv: This is the dataframe for the V-band data of the star
        dfg: This is the dataframe for the g-band data of the star
    
    This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto 
    a data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the
    Photo filter column data has a value of one, its row is sorted into the data frame corresponding to the V-band. If the Photo
    filter column data has a value of zero, it gets sorted into the data frame corresponding to the g-band.
    """
    fname = os.path.join(guide, str(asas_sn_id)+'.dat')

    dfv = pd.DataFrame()
    dfg = pd.DataFrame()

    fdata = pd.read_fwf(fname, header=None)
    fdata.columns = ["JD", "Mag", "Mag_err", "Quality", "Cam_number", "Phot_filter", "Camera"] #These are the columns of data

    dfv = fdata.loc[fdata["Phot_filter"] == 1].reset_index(drop=True) #This sorts the data into the V-band
    dfg = fdata.loc[fdata["Phot_filter"] == 0].reset_index(drop=True) #This sorts the data into the g-band

    dfv['Mag'].astype(float)
    dfg['Mag'].astype(float)

    dfv['JD'].astype(float)
    dfg['JD'].astype(float)

    return dfv, dfg


def read_lightcurve_csv(asas_sn_id, guide = 'known_dipper_lightcurves/'):
    """
    Input: 
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        dfv: This is the dataframe for the V-band data of the star
        dfg: This is the dataframe for the g-band data of the star
    
    This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto 
    a data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the
    Photo filter column data has a value of one, its row is sorted into the data frame corresponding to the V-band. If the Photo
    filter column data has a value of zero, it gets sorted into the data frame corresponding to the g-band.
    """
    fname = os.path.join(guide, str(asas_sn_id)+'.csv')

    df = pd.read_csv(fname)

    df['Mag'] = pd.to_numeric(df['mag'],errors='coerce')
    df = df.dropna()
    
    df['Mag'].astype(float)
    df['JD'] = df.HJD.astype(float)

    dfg = df.loc[df['Filter'] == 'g'].reset_index(drop=True)
    dfv = df.loc[df['Filter'] == 'V'].reset_index(drop=True)

    return dfv, dfg

# This function finds the peaks 
def find_peak(df, prominence=0.17, distance=25, height=0.3, width=2):
	'''
	Inputs:
		df: dataframe of the data, requires columns of 'Mag' and 'JD'
		prominence: same parameter of scipy.signal.find_peaks()
		distance: same parameter of scipy.signal.find_peaks()
		height: same parameter of scipy.signal.find_peaks()
		width: same parameter of scipy.signal.find_peaks()

	Outputs:
		peak: a series of the peaks found
		meanmag: the average magnitude of the light curve
		length: the number of peaks found

	Description:
	'''
	df['Mag'] = [float(i) for i in df['Mag']]
	df['JD'] = [float(i) for i in df['JD']]
	mag = df['Mag']
	jd = df['JD']

	meanmag = sum(mag) / len(mag)
	df_mag_avg = [i - meanmag for i in mag]
	
    peaks = scipy.signal.find_peaks(df_mag_avg, prominence=prominence, distance=distance, height=height, width=width) 
	
    peak = peaks[0]
	prop = peaks[1]
	
    length = len(peak)
	
    peak = [int(i) for i in peak]
	peak = pd.Series(peak)
	
    return peak, meanmag, length	
# End of the find_peak

# This function creates a custom id using the position of the star
def custom_id(ra_val,dec_val):
    c = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree, frame='icrs')
    ra_num = c.ra.hms
    dec_num = c.dec.dms

    if int(dec_num[0]) < 0:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$-$'+str(int(c.dec.dms[0])*(-1)).rjust(2,'0')+str(int(c.dec.dms[1])*(-1)).rjust(2,'0')+str(int(round(c.dec.dms[2])*(-1))).rjust(2,'0')
    else:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$+$'+str(int(c.dec.dms[0])).rjust(2,'0')+str(int(c.dec.dms[1])).rjust(2,'0')+str(int(round(c.dec.dms[2]))).rjust(2,'0')

    return cust_id
# End of custom_id

#
def year_to_jd(year):
    jd_epoch = 2449718.5 - (2.458 * 10 **6)
    year_epoch = 1995
    days_in_year = 365.25
    return (year-year_epoch)*days_in_year + jd_epoch-2450000
#

#
def jd_to_year(jd):
    jd_epoch = 2449718.5 - (2.458 * 10 **6)
    year_epoch = 1995
    days_in_year = 365.25
    return year_epoch + (jd - jd_epoch) / days_in_year
#

# This function plots the light curve
def plot_light_curve(df, ra, dec, peak_option=False):
    '''
    '''
    cust_id = custom_id(ra,dec)
    peak, meanmag, length = find_peak(df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    cams = df["Camera"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    if peak_option == False:

        for i in range(0,camnum):
            cam = df.loc[df["Camera"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["Mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        ax.set_xlim((min(df.JD)-(2.458 * 10 ** 6)-300),(max(df.JD)-(2.458 * 10 ** 6)+150))
        ax.set_ylim((min(df['Mag'])-0.1),(max(df['Mag'])+0.1))
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
            cam = df.loc[df["Camera"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["Mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(len(peak)-1):
            ax.vlines((df.JD[peak[i]] - (2.458 * 10**6)), (min(df['Mag'])-0.1), (max(df['Mag'])+0.1), "k", alpha=0.4)

        ax.set_xlim((min(df.JD)-(2.458 * 10 ** 6)-300),(max(df.JD)-(2.458 * 10 ** 6)+150))
        ax.set_ylim((min(df['Mag'])-0.1),(max(df['Mag'])+0.1))
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
# End of plot_light_curve

#
def plot_zoom(df, ra, dec, zoom_range=[-300,3000], peak_option=False):
    '''
    '''
    cust_id = custom_id(ra,dec)
    peak, meanmag, length = find_peak(df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax = plotparams(ax)

    cams = df["Camera"]
    camtype = np.unique(cams)
    camnum = len(camtype)

    if peak_option == False:

        for i in range(0,camnum):
            cam = df.loc[df["Camera"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["Mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        ax.set_xlim(zoom_range[0],zoom_range[1])
        ax.set_ylim((min(df['Mag'])-0.1),(max(df['Mag'])+0.1))
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()

    if peak_option == True:

        for i in range(0,camnum):
            cam = df.loc[df["Camera"] == camtype[i]].reset_index(drop=True)
            camjd = cam["JD"].astype(float) - (2.458 * 10 ** 6)
            cammag = cam["Mag"].astype(float)
            ax.scatter(camjd, cammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(len(peak)-1):
            ax.vlines((df.JD[peak[i]] - (2.458 * 10**6)), (min(df['Mag'])-0.1), (max(df['Mag'])+0.1), "k", alpha=0.4)

        ax.set_xlim(zoom_range[0],zoom_range[1])
        ax.set_ylim((min(df['Mag'])-0.1),(max(df['Mag'])+0.1))
        ax.set_xlabel('Julian Date $- 2458000$ [d]', fontsize=15)
        ax.set_ylabel('g [mag]', fontsize=15)
        ax.set_title(cust_id, y=1.03, fontsize=20)
        ax.invert_yaxis()
        ax.minorticks_on()
#

#
def plot_multiband(dfv, dfg, ra, dec, peak_option=False):
    '''
    '''
    cust_id = custom_id(ra,dec)
    peak, meanmag, length = find_peak(dfg)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    gcams = dfg["Camera"]
    gcamtype = np.unique(gcams)
    gcamnum = len(gcamtype)

    vcams = dfv["Camera"]
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
            gcam = dfg.loc[dfg["Camera"] == gcamtype[i]].reset_index(drop=True)
            gcamjd = gcam["JD"].astype(float) - (2.458 * 10 ** 6)
            gcammag = gcam["Mag"].astype(float)
            ax.scatter(gcamjd, gcammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(0,vcamnum):
            vcam = dfv.loc[dfv["Camera"] == vcamtype[i]].reset_index(drop=True)
            vcamjd = vcam["JD"].astype(float) - (2.458 * 10 ** 6)
            vcammag = vcam["Mag"].astype(float)
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
            gcam = dfg.loc[dfg["Camera"] == gcamtype[i]].reset_index(drop=True)
            gcamjd = gcam["JD"].astype(float) - (2.458 * 10 ** 6)
            gcammag = gcam["Mag"].astype(float)
            ax.scatter(gcamjd, gcammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(0,vcamnum):
            vcam = dfv.loc[dfv["Camera"] == vcamtype[i]].reset_index(drop=True)
            vcamjd = vcam["JD"].astype(float) - (2.458 * 10 ** 6)
            vcammag = vcam["Mag"].astype(float)
            ax.scatter(vcamjd, vcammag, color=colors[i], alpha=0.6, marker='.')

        for i in range(len(peak)-1):
            ax.vlines((dfg.JD[peak[i]] - (2.458 * 10**6)), (min(dfg['Mag'])-0.1), (max(dfg['Mag'])+0.1), "k", alpha=0.4)

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
#

#
#def peak_params(df):
#