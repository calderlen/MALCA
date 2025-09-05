
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
files_12_12_5 = glob.glob(lc_12_12_5+"/lc*_cal")
num = len(files_12_12_5)
directories_12_12_5 = [str(i) for i in range(num)]

files_12_5_13 = glob.glob(lc_12_5_13+"/lc*_cal")
num = len(files_12_5_13)
directories_12_5_13 = [str(i) for i in range(num)]

files_13_13_5 = glob.glob(lc_13_13_53+"/lc*_cal")
num = len(files_13_13_5)
directories_13_13_5  = [str(i) for i in range(num)]

files_13_5_14 = glob.glob(lc_13_5_14+"/lc*_cal")
num = len(files_13_5_14)
directories_13_5_14 = [str(i) for i in range(num)]

files_14_5_15 = glob.glob(lc_14_5_15+"/lc*_cal")
num = len(files_14_5_15)
directories_14_5_15 = [str(i) for i in range(num)]


for x in directories_12_12_5:
    startTime = time.time()
    print('Starting 12-12.5 ' + x + ' directory')

    ID = pd.read_table('12-12.5/index' + x + '.csv', sep=r'\,|\t', engine='python')
    directory = '12-12.5/lc' + x + '_cal/'

    for filename in os.listdir(directory):
            path = directory + filename
            target = [filename.split('.')[0]]
            Target = [int(i) for i in target]
            Target = Target[0]






test_index = lc_12_12_5 + 'index0.csv'

df_asassn_idx = pd.read_csv(test_index)


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
