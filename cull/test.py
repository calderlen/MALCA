
import numpy as np
import pandas as pd
import scipy
import math
import os
import re
import time

from tqdm import tqdm

from astropy.timeseries import LombScargle as ls
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery import Gaia

import glob

lc_12_12_5 = asassn_dir + '/12_12.5'
lc_12_5_13 = asassn_dir + '/12.5_13'
lc_13_13_5 = asassn_dir + '/13_13.5'
lc_13_5_14 = asassn_dir + '/13.5_14'
lc_14_5_15 = asassn_dir + '/14.5_15'

# Find all files that match
files_12_12_5 = [f for f in glob.glob(os.path.join(lc_12_12_5, "lc*_cal")) if os.path.isfile(f)]
num_12_12_5 = len(files_12_12_5)
directories_12_12_5 = [str(i) for i in range(num_12_12_5)]

files_12_5_13 = [f for f in glob.glob(os.path.join(lc_12_5_13, "lc*_cal")) if os.path.isfile(f)]
num_12_5_13 = len(files_12_5_13)
directories_12_5_13 = [str(i) for i in range(num_12_5_13)]

files_13_13_5 = [f for f in glob.glob(os.path.join(lc_13_13_5, "lc*_cal")) if os.path.isfile(f)]
num_13_13_5 = len(files_13_13_5)
directories_13_13_5 = [str(i) for i in range(num_13_13_5)]

files_13_5_14 = [f for f in glob.glob(os.path.join(lc_13_5_14, "lc*_cal")) if os.path.isfile(f)]
num_13_5_14 = len(files_13_5_14)
directories_13_5_14 = [str(i) for i in range(num_13_5_14)]

files_14_5_15 = [f for f in glob.glob(os.path.join(lc_14_5_15, "lc*_cal")) if os.path.isfile(f)]
num_14_5_15 = len(files_14_5_15)
directories_14_5_15 = [str(i) for i in range(num_14_5_15)]


for x in range num_12_12_5:
    directory = lc_12_12_5 + '/lc' + str(x) + '_cal'
        for filename in os.listdir(directory):
            if filename.endswith('.dat'):
                path = directory + '/' + filename
                target = [filename.split('.')[0]]
                Target = [int(i) for i in target]
                Target = Target[0]

                i = Target

                if i in Targets:
                    print(str(x) + " / " + str(30) + " - " + str(Target))
                    
                    df1 = pd.read_csv(path, sep="\s+", names=column_names)
                    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)

                    ra = np.where(ID['asas_sn_id'] == Target) #lccal2

                    df = pd.read_table(path, sep="\s+", names=column_names)
                    df.drop(df[df['good/bad'] < 1].index, inplace = True)

                    # Why is this filtered out?
                    if Target == 17181160895:
                        df.drop(df[df['good/bad'] < 1].index, inplace = True)
                        df.drop(df[df['JD'] < 2.458e+06].index, inplace = True)

                    RA = ID['ra_deg'].iloc[ra]

                    #pstarr mag
                    #p_mag = ID['pstarrs_g_mag'].iloc[ra]
                    #p_mag = np.array(p_mag)
                    #p_mag = float(p_mag)

                    #median of overall lc
                    #lcMED = np.median(df.mag)
                    #lcMED_err = mad_std(df.mag)
        
                    #dispersion of overall lc (range)
                    #lcDISP = np.ptp(df.mag)

                    # Why are these filtered out?

                    #if Target == 77310656806:
                    #        continue
                    #    elif Target == 51539980676:
                    #        continue
                    #    elif Target == 395137591432:
                    #        continue
                    #    elif Target == 180388896780:
                    #        continue
                    #    elif Target == 68719657585:
                    #        continue
                    #    elif Target == 51539954617:
                    #        continue
                    #    elif Target == 137440390128:
                    #        continue
                    #    elif Target == 412316992073:
                    #        continue
                    #    elif Target == 558346685740:
                    #        continue
                    #    elif Target == 386548333169:
                    #        continue
                    #    elif Target == 266288989631:
                    #        continue

                    #    time = df['JD']
                    #    mag = df['mag']



def make_id(ra_val,dec_val):
    """
    CHANGES: ENCODE GMAG INTO THIS ID!!!
    """

    c = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree, frame='icrs')
    ra_num = c.ra.hms
    dec_num = c.dec.dms

    sign = '+' if c.dec.dms[0] >= 0 else '-'
    deg = abs(int(c.dec.dms[0]))
    arcmin = abs(int(c.dec.dms[1]))
    arcsec = abs(int(round(c.dec.dms[2])))

    cust_id = (
        'J'
        + str(int(c.ra.hms[0])).rjust(2, '0')
        + str(int(c.ra.hms[1])).rjust(2, '0')
        + str(int(round(c.ra.hms[2]))).rjust(2, '0')
        + sign
        + str(deg).rjust(2, '0')
        + str(arcmin).rjust(2, '0')
        + str(arcsec).rjust(2, '0')
    )

    return id

def naive_dip_detection(df, prominence=0.17, distance=25, height=0.3, width=2):
    # code adapted from Brayden JoHantgen's code

	df['Mag'] = [float(i) for i in df['Mag']]
	df['JD'] = [float(i) for i in df['JD']]

    mag = df['Mag']
    jd = df['JD']

    mag_mean = sum(mag)/len(mag)
    df_mag_avg = [i - mag_mean for i in mag]
    
    peaks = scipy.signal.find_peaks(df_mag_avg, prominence=prominence, distance=distance, height=height, width=width) 

    peak = peaks[0]
	prop = peaks[1]
	
    length = len(peak)
	
    peak = [int(i) for i in peak]
	peak = pd.Series(peak)

    return peak, mag_mean, length