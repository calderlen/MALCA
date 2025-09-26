import numpy as np
import pandas as pd
from baseline import per_camera_baseline
from utils import read_lc_dat, naive_peak_search, match_index_to_lc
from tqdm import tqdm
import os

import json, tempfile

lc_dir_masked = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked"

MAG_BINS = ['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15']

# masked index CSVs live here (for index*_masked.csv)
masked_bins = [f"{lc_dir_masked}/{b}" for b in MAG_BINS]

lc_12_12_5 = lc_dir_masked + '/12_12.5'
lc_12_5_13 = lc_dir_masked + '/12.5_13'
lc_13_13_5 = lc_dir_masked + '/13_13.5'
lc_13_5_14 = lc_dir_masked + '/13.5_14'
lc_14_14_5 = lc_dir_masked + '/14_14.5'
lc_14_5_15 = lc_dir_masked + '/14.5_15'

asassn_columns=["JD",
                "mag",
                'error', 
                'good_bad', #1=good, 0 =bad
                'camera#', 
                'v_g_band', #1=V, 0=g
                'saturated',
                'cam_field']
  
asassn_raw_columns = ['cam#',
                'median',
                '1siglow', 
               '1sighigh', 
               '90percentlow',
               '90percenthigh']

asassn_index_columns = ['asassn_id',
                        'ra_deg',
                        'dec_deg',
                        'refcat_id',
                        'gaia_id', 
                        'hip_id',
                        'tyc_id',
                        'tmass_id',
                        'sdss_id',
                        'allwise_id',
                        'tic_id',
                        'plx',
                        'plx_d',
                        'pm_ra',
                        'pm_ra_d',
                        'pm_dec',
                        'pm_dec_d',
                        'gaia_mag',
                        'gaia_mag_d',
                        'gaia_b_mag',
                        'gaia_b_mag_d',
                        'gaia_r_mag',
                        'gaia_r_mag_d',
                        'gaia_eff_temp',
                        'gaia_g_extinc',
                        'gaia_var',
                        'sfd_g_extinc',
                        'rp_00_1',
                        'rp_01',
                        'rp_10',
                        'pstarrs_g_mag',
                        'pstarrs_g_mag_d',
                        'pstarrs_g_mag_chi',
                        'pstarrs_g_mag_contrib',
                        'pstarrs_r_mag',
                        'pstarrs_r_mag_d',
                        'pstarrs_r_mag_chi',
                        'pstarrs_r_mag_contrib',
                        'pstarrs_i_mag',
                        'pstarrs_i_mag_d',
                        'pstarrs_i_mag_chi',
                        'pstarrs_i_mag_contrib',
                        'pstarrs_z_mag',
                        'pstarrs_z_mag_d',
                        'pstarrs_z_mag_chi',
                        'pstarrs_z_mag_contrib',
                        'nstat']


def clean_lc(df):
    
    mask = np.ones(len(df), dtype=bool)

    #if "good_bad" in df.columns:
    #    mask &= (df["good_bad"] == 1) #NOT DOING THIS FOR DIPPERS RIGHT NOW UNTIL YOU RECOVER SAME CANDIDATES AS BRAYDEN WITH DIPS ENABLED!
    if "saturated" in df.columns:
        mask &= (df["saturated"] == 0)
    
    # required fields
    mask &= df["JD"].notna() & df["mag"].notna()
    if "error" in df.columns:
        mask &= df["error"].notna() & (df["error"] > 0.) & (df["error"] < 1.)
    df = df.loc[mask].cop

    df = df.sort_values("JD").reset_index(drop=True)
    return df


def dip_finder(
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'],
    id_column="asas_sn_id",
    # naive_peak_search params
    prominence=0.17, distance=25, height=0.3, width=2,
    ):

    #collect matches
    matches = list(match_index_to_lc(index_path=index_path,
                                     lc_path=lc_path,
                                     mag_bins=mag_bins,
                                     id_column=id_column))
    df_matches = pd.DataFrame(matches)

    # only rows with a found .dat
    work = df_matches[df_matches["found"]].copy()

    rows = []
    for _, r in tqdm(work.iterrows(), total=len(work), desc="Peak search", unit="lc"):
        asn = r["asas_sn_id"]
        lc_dir = r["lc_dir"]                      # you return lc_dir in match_index_to_lc
        dfg, dfv = read_lc_dat(asn, lc_dir)       # your reader, returns (g, V)

        # g band
        n_peaks_g = mean_g = None
        peaks_g_idx = peaks_g_jd = None
        if not dfg.empty:
            peaks_g, mean_g, n_peaks_g = naive_peak_search(
                dfg, prominence=prominence, distance=distance, height=height, width=width
            )
            # convert indices to JD
            peaks_g_idx = peaks_g.tolist()
            peaks_g_jd  = dfg["JD"].values[peaks_g_idx].tolist()

        # V band
        n_peaks_v = mean_v = None
        peaks_v_idx = peaks_v_jd = None
        if not dfv.empty:
            peaks_v, mean_v, n_peaks_v = naive_peak_search(
                dfv, prominence=prominence, distance=distance, height=height, width=width
            )
            peaks_v_idx = peaks_v.tolist()
            peaks_v_jd  = dfv["JD"].values[peaks_v_idx].tolist()

        rows.append({
            "asas_sn_id": asn,
            "mag_bin":    r["mag_bin"],
            "index_num":  r["index_num"],
            "index_csv":  r["index_csv"],
            "lc_dir":     lc_dir,
            "dat_path":   r["dat_path"],
            # g band
            "g_n_peaks":  n_peaks_g,
            "g_mean_mag": mean_g,
            "g_peaks_idx": peaks_g_idx,
            "g_peaks_jd":  peaks_g_jd,
            # V band
            "v_n_peaks":  n_peaks_v,
            "v_mean_mag": mean_v,
            "v_peaks_idx": peaks_v_idx,
            "v_peaks_jd":  peaks_v_jd,
            # LC meta
            "jd_first": float((dfg["JD"].iloc[0] if not dfg.empty else (dfv["JD"].iloc[0] if not dfv.empty else np.nan))) if (not dfg.empty or not dfv.empty) else np.nan,
            "jd_last":  float((dfg["JD"].iloc[-1] if not dfg.empty else (dfv["JD"].iloc[-1] if not dfv.empty else np.nan))) if (not dfg.empty or not dfv.empty) else np.nan,
            "n_rows_g":  int(len(dfg)) if not dfg.empty else 0,
            "n_rows_v":  int(len(dfv)) if not dfv.empty else 0,
        })

    return pd.DataFrame(rows)



# dip finder that does only one bin at a time

def dip_finder_by_bin(
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=('12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'),
    id_column="asas_sn_id",
    out_dir="./peak_results",
    out_format="parquet",           # "parquet" or "csv"
    **peak_kwargs                   # forwarded to naive_peak_search
):
    os.makedirs(out_dir, exist_ok=True)

    for b in tqdm(mag_bins, desc="Bins", unit="bin"):
        # If we already did this bin, skip
        out_path = os.path.join(out_dir, f"peaks_{b.replace('.','_')}.{out_format}")
        if os.path.exists(out_path):
            continue

        # restrict matching to this bin (saves time + memory)
        dfm = pd.DataFrame(list(match_index_to_lc(
            index_path=index_path, lc_path=lc_path, mag_bins=[b], id_column=id_column
        )))
        work = dfm[dfm["found"]].copy()
        rows = []

        for _, r in tqdm(work.iterrows(), total=len(work), desc=f"{b} peak search", leave=False):
            asn = r["asas_sn_id"]
            dfg, dfv = read_lc_dat(asn, r["lc_dir"])

            # g
            g = {"n": None, "mean": None, "idx": None, "jd": None}
            if not dfg.empty:
                peaks_g, mean_g, n_g = naive_peak_search(dfg, **peak_kwargs)
                g = {
                    "n": n_g,
                    "mean": mean_g,
                    "idx": peaks_g.tolist(),
                    "jd":  dfg["JD"].values[peaks_g].tolist(),
                }

            # V
            v = {"n": None, "mean": None, "idx": None, "jd": None}
            if not dfv.empty:
                peaks_v, mean_v, n_v = naive_peak_search(dfv, **peak_kwargs)
                v = {
                    "n": n_v,
                    "mean": mean_v,
                    "idx": peaks_v.tolist(),
                    "jd":  dfv["JD"].values[peaks_v].tolist(),
                }

            rows.append({
                "mag_bin": b,
                "asas_sn_id": asn,
                "index_num": r["index_num"],
                "index_csv": r["index_csv"],
                "lc_dir": r["lc_dir"],
                "dat_path": r["dat_path"],
                "g_n_peaks": g["n"],
                "g_mean_mag": g["mean"],
                "g_peaks_idx": g["idx"],
                "g_peaks_jd":  g["jd"],
                "v_n_peaks": v["n"],
                "v_mean_mag": v["mean"],
                "v_peaks_idx": v["idx"],
                "v_peaks_jd":  v["jd"],
            })

        df_out = pd.DataFrame(rows)
        if out_format == "parquet":
            df_out.to_parquet(out_path, index=False)
        else:
            df_out.to_csv(out_path, index=False)

