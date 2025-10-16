import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from utils import read_lc_dat, naive_peak_search, match_index_to_lc

from lc_metrics import run_metrics_pcb, is_dip_dominated

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
  
asassn_raw_columns = [
                'cam#',
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

# naive dip finder that works one bin at a time

def naive_dip_finder(
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=('12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'),
    id_column="asas_sn_id",
    out_dir="./peak_results",
    out_format="parquet",           # "parquet" to save space or "csv" for convenience; picking parquet for now so as to not stress servers
    **peak_kwargs                   # forwarded to naive_peak_search
):
    os.makedirs(out_dir, exist_ok=True)

    for b in tqdm(mag_bins, desc="Bins", unit="bin"):

        # if we already did this bin, skip
        out_path = os.path.join(out_dir, f"peaks_{b.replace('.','_')}.{out_format}")
        if os.path.exists(out_path):
            continue

        # restrict matching to single bin
        dfm = pd.DataFrame(list(match_index_to_lc(
            index_path=index_path, lc_path=lc_path, mag_bins=[b], id_column=id_column
        )))
        work = dfm[dfm["found"]].copy()
        rows = []

        for _, r in tqdm(work.iterrows(), total=len(work), desc=f"{b} peak search", leave=False):
            asn = r["asas_sn_id"]
            df_g, df_v = read_lc_dat(asn, r["lc_dir"])

            jd_first = np.nan
            jd_last = np.nan
            
            # g
            g = {"n": None, "mean": None, "idx": None, "jd": None}
            if not df_g.empty:
                peaks_g, mean_g, n_g = naive_peak_search(df_g, **peak_kwargs)
                g = {
                    "n": n_g,
                    "mean": mean_g,
                    "idx": peaks_g.tolist(),
                    "jd":  df_g["JD"].values[peaks_g].tolist(),
                }
                jd_first = float(df_g["JD"].iloc[0])
                jd_last = float(df_g["JD"].iloc[-1])
            # V
            v = {"n": None, "mean": None, "idx": None, "jd": None}
            if not df_v.empty:
                peaks_v, mean_v, n_v = naive_peak_search(df_v, **peak_kwargs)
                v = {
                    "n": n_v,
                    "mean": mean_v,
                    "idx": peaks_v.tolist(),
                    "jd":  df_v["JD"].values[peaks_v].tolist(),
                }
                jd_first = float(df_v["JD"].iloc[0])
                jd_last = float(df_v["JD"].iloc[-1])

            n_rows_g = int(len(df_g)) if not df_g.empty else 0
            n_rows_v = int(len(df_v)) if not df_v.empty else 0

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
                "g_peaks_jd": g["jd"],
                "v_n_peaks": v["n"],
                "v_mean_mag": v["mean"],
                "v_peaks_idx": v["idx"],
                "v_peaks_jd": v["jd"],
                "jd_first": jd_first,
                "jd_last": jd_last,
                "n_rows_g": n_rows_g,
                "n_rows_v": n_rows_v,
            })

        df_out = pd.DataFrame(rows)
        if out_format == "parquet":
            df_out.to_parquet(out_path, index=False)
        else:
            df_out.to_csv(out_path, index=False)


def dip_finder(
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=('12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'),
    id_column="asas_sn_id",
    out_dir="./peak_results",
    out_format="parquet",
    **pcb_kwargs,
):
    os.makedirs(out_dir, exist_ok=True)

    metric_fields = (
        "n_dip_runs",
        "n_jump_runs",
        "n_dip_points",
        "n_jump_points",
        "most_recent_dip",
        "most_recent_jump",
        "max_depth",
        "max_height",
        "max_dip_duration",
        "max_jump_duration",
        "dip_fraction",
        "jump_fraction",
    )

    def empty_metrics(prefix):
        vals = {
            "n_dip_runs": 0,
            "n_jump_runs": 0,
            "n_dip_points": 0,
            "n_jump_points": 0,
            "most_recent_dip": np.nan,
            "most_recent_jump": np.nan,
            "max_depth": np.nan,
            "max_height": np.nan,
            "max_dip_duration": np.nan,
            "max_jump_duration": np.nan,
            "dip_fraction": np.nan,
            "jump_fraction": np.nan,
        }
        out = {f"{prefix}_{k}": vals[k] for k in metric_fields}
        out[f"{prefix}_is_dip_dominated"] = False
        return out

    for b in tqdm(mag_bins, desc="Bins", unit="bin"):
        out_path = os.path.join(out_dir, f"peaks_{b.replace('.','_')}.{out_format}")
        if os.path.exists(out_path):
            continue

        dfm = pd.DataFrame(
            list(
                match_index_to_lc(
                    index_path=index_path,
                    lc_path=lc_path,
                    mag_bins=[b],
                    id_column=id_column,
                )
            )
        )
        work = dfm[dfm["found"]].copy()
        rows = []

        for _, r in tqdm(work.iterrows(), total=len(work), desc=f"{b} peak search", leave=False):
            asn = r["asas_sn_id"]
            dfg, dfv = read_lc_dat(asn, r["lc_dir"])

            jd_first = np.nan
            jd_last = np.nan

            if not dfg.empty:
                peaks_g, mean_g, n_g = naive_peak_search(dfg, **pcb_kwargs)
                jd_first = float(dfg["JD"].iloc[0])
                jd_last = float(dfg["JD"].iloc[-1])
                g_stats = run_metrics_pcb(dfg, **pcb_kwargs)
                g_metrics = {f"g_{k}": v for k, v in g_stats.items()}
                g_metrics["g_is_dip_dominated"] = bool(is_dip_dominated(g_stats))
            else:
                peaks_g, mean_g, n_g = (pd.Series(dtype=int), np.nan, 0)
                g_metrics = empty_metrics("g")

            if not dfv.empty:
                peaks_v, mean_v, n_v = naive_peak_search(dfv, **pcb_kwargs)
                if np.isnan(jd_first):
                    jd_first = float(dfv["JD"].iloc[0])
                if np.isnan(jd_last):
                    jd_last = float(dfv["JD"].iloc[-1])
                v_stats = run_metrics_pcb(dfv, **pcb_kwargs)
                v_metrics = {f"v_{k}": v for k, v in v_stats.items()}
                v_metrics["v_is_dip_dominated"] = bool(is_dip_dominated(v_stats))
            else:
                peaks_v, mean_v, n_v = (pd.Series(dtype=int), np.nan, 0)
                v_metrics = empty_metrics("v")

            row = {
                "mag_bin": b,
                "asas_sn_id": asn,
                "index_num": r["index_num"],
                "index_csv": r["index_csv"],
                "lc_dir": r["lc_dir"],
                "dat_path": r["dat_path"],
                "g_n_peaks": n_g,
                "g_mean_mag": mean_g,
                "g_peaks_idx": peaks_g.tolist(),
                "g_peaks_jd": dfg["JD"].values[peaks_g].tolist() if not dfg.empty else [],
                "v_n_peaks": n_v,
                "v_mean_mag": mean_v,
                "v_peaks_idx": peaks_v.tolist(),
                "v_peaks_jd": dfv["JD"].values[peaks_v].tolist() if not dfv.empty else [],
                "jd_first": jd_first,
                "jd_last": jd_last,
                "n_rows_g": int(len(dfg)) if not dfg.empty else 0,
                "n_rows_v": int(len(dfv)) if not dfv.empty else 0,
            }
            row.update(g_metrics)
            row.update(v_metrics)
            rows.append(row)

        df_out = pd.DataFrame(rows)
        if out_format == "parquet":
            df_out.to_parquet(out_path, index=False)
        else:
            df_out.to_csv(out_path, index=False)


            