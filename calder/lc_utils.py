import pandas as pd
import os
import re
from glob import glob
from tqdm import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord

colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39", 
              "#ce5761", "#c68c45", '#b5b246', '#d77fcc', '#7362cf', '#ce443f', '#3fc1bf', '#cda735',
              '#a1b055']

def year_to_jd(year):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    jd_epoch = 2449718.5 - (2.458 * 10 **6)
    year_epoch = 1995
    days_in_year = 365.25
    return (year-year_epoch)*days_in_year + jd_epoch-2450000

def jd_to_year(jd):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
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

        #if df_v.empty:
        #    print(f"[warn] {asassn_id}: no V band rows")
        #if df_g.empty:
        #    print(f"[warn] {asassn_id}: no g band rows")
        
    else:
        print(f"[error] {asassn_id}: file not found in {path}")
        df_g = pd.DataFrame()
        df_v = pd.DataFrame()
                 
    return df_g, df_v


def match_index_to_lc(
    index_path: str = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path:    str = "/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins:   list = ['12_12.5','12.5_13','13_13.5','13.5_14','14.5_15'],
    id_column:  str = "asas_sn_id",
):
    """
    Generator function that iterates over index*_masked.csv files in lcsv2_masked/<mag_bin>/, find corresponding lc<num>_cal/ directories in lcsv2/<mag_bin>/, and yield one record per asas_sn_id with whether its .dat file exists. Outputs a dict
    """

    idx_pattern = re.compile(r"index(\d+)_masked\.csv$", re.IGNORECASE)

    for mag_bin in tqdm(mag_bins, desc="Bins", unit="bin"):
        idx_paths = sorted(glob(os.path.join(index_path, mag_bin, "index*_masked.csv")))
        for idx_csv in tqdm(idx_paths, desc=f"{mag_bin} index CSVs", leave=False):

            # Assume pattern matches; will raise if not.
            idx_num = int(idx_pattern.search(os.path.basename(idx_csv)).group(1))

            lc_dir = os.path.join(lc_path, mag_bin, f"lc{idx_num}_cal")

            ids = (
                pd.read_csv(idx_csv, dtype={id_column: "string"})[id_column]
                .dropna()
                .astype(str)
                .unique()
            )

            for asn in ids:
                dat_path = os.path.join(lc_dir, f"{asn}.dat")
                found = os.path.exists(dat_path)
                yield {
                    "mag_bin":      mag_bin,
                    "index_num":    idx_num,
                    "index_csv":    idx_csv,
                    "lc_dir":       lc_dir,
                    "asas_sn_id":   asn,
                    "dat_path":     dat_path if found else None,
                    "found":        found,
                }


def custom_id(ra_val,dec_val):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    c = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree, frame='icrs')
    ra_num = c.ra.hms
    dec_num = c.dec.dms

    if int(dec_num[0]) < 0:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$-$'+str(int(c.dec.dms[0])*(-1)).rjust(2,'0')+str(int(c.dec.dms[1])*(-1)).rjust(2,'0')+str(int(round(c.dec.dms[2])*(-1))).rjust(2,'0')
    else:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$+$'+str(int(c.dec.dms[0])).rjust(2,'0')+str(int(c.dec.dms[1])).rjust(2,'0')+str(int(round(c.dec.dms[2]))).rjust(2,'0')

    return cust_id


def plotparams(ax, labelsize=15):
    """
    ADAPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax



def divide_cameras()
    """
    ADAPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    pass