
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

def read_lightcurve(asassn_id, path):
    # code adapted from Brayden JoHantgen's code

    # different processing for .dat and .csv files

    if os.path.exists(f"{path}/{asassn_id}.dat"):

        fname = os.path.join(path, f"{asassn_id}.dat")

        df_v = pd.DataFrame()
        df_g = pd.DataFrame()

        fdata = pd.read_fwf(fname, header=None)
        fdata.columns = ["JD", "Mag", "Mag_err", "Quality", "Cam_number", "Phot_filter", "Camera"]

        df_v = fdata.loc[fdata["Phot_filter"] == 1].reset_index(drop=True)
        df_g = fdata.loc[fdata["Phot_filter"] == 0].reset_index(drop=True)      

        df_v['Mag'].astype(float)
        df_v['JD'].astype(float)

        df_g['Mag'].astype(float)
        df_g['JD'].astype(float)

    elif os.path.exists(f"{path}/{asassn_id}.csv"):

        fname = os.path.join(path, f"{asassn_id}.csv")

        df = pd.read_csv(fname)

        df['Mag'] = pd.to_numeric(df['Mag'], errors='coerce')
        df = df.dropna()

        df['Mag'].astype(float)
        df['JD'] = df.HJD.astype(float)

        df_g = df.loc[df["Filter"] == 'g'].reset_index(drop=True)
        df_v = df.loc[df["Filter"] == 'V'].reset_index(drop=True)

    return df_v, df_g
