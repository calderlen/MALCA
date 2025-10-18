import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from lc_utils import read_lc_dat, match_index_to_lc
from df_utils import naive_peak_search
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
    df = df.loc[mask].copy()

    df = df.sort_values("JD").reset_index(drop=True)
    return df

METRIC_FIELDS = (
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
        out = {f"{prefix}_{k}": vals[k] for k in METRIC_FIELDS}
        out[f"{prefix}_is_dip_dominated"] = False
        return out

# naive dip finder that works one bin at a time

def _process_record(record: dict, pcb_kwargs: dict):
    """
    Worker function to process a single ASAS-SN target. Returns a dict row.
    """
    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat(asn, record["lc_dir"])

    # basic cleaning to drop non-finite/invalid points and sort by time
    if not dfg.empty:
        dfg = clean_lc(dfg)
    if not dfv.empty:
        dfv = clean_lc(dfv)

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
        "mag_bin": record["mag_bin"],
        "asas_sn_id": asn,
        "index_num": record["index_num"],
        "index_csv": record["index_csv"],
        "lc_dir": record["lc_dir"],
        "dat_path": record["dat_path"],
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
    return row

def naive_dip_finder(
    # dip finder still uses naive peak search and is thus still naive
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=('12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'),
    id_column="asas_sn_id",
    out_dir="./peak_results",
    out_format="csv",
    n_workers=None,
    chunk_size=100000,
    max_inflight=None,
    **pcb_kwargs,
):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")

    # determine worker defaults
    if n_workers is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_workers = max(1, min(48, cpu - 2))
    if max_inflight is None:
        max_inflight = max(4, n_workers * 4)

    for b in tqdm(mag_bins, desc="Bins", unit="bin"):
        suffix = f"_{timestamp}" if timestamp else ""
        out_path = os.path.join(out_dir, f"peaks_{b.replace('.','_')}{suffix}.{out_format}")
        if os.path.exists(out_path):
            continue

        rows_buffer = []
        if out_format == "csv":
            header_written = False
        else:
            header_written = None  # unused for parquet

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            pending = set()
            pbar = tqdm(desc=f"{b} peak search", unit="obj", leave=False)
            scheduled = 0

            def flush_if_needed():
                if out_format == "csv" and len(rows_buffer) >= chunk_size:
                    mode = "a" if os.path.exists(out_path) else "w"
                    header = not os.path.exists(out_path)
                    pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
                    rows_buffer.clear()

            def drain_some(all_pending):
                done_now = 0
                for fut in list(all_pending):
                    if fut.done():
                        row = fut.result()
                        rows_buffer.append(row)
                        all_pending.remove(fut)
                        done_now += 1
                        flush_if_needed()
                return done_now

            for rec in match_index_to_lc(
                index_path=index_path,
                lc_path=lc_path,
                mag_bins=[b],
                id_column=id_column,
            ):
                if not rec.get("found", False):
                    continue
                pending.add(ex.submit(_process_record, rec, pcb_kwargs))
                scheduled += 1
                # grow total dynamically for ETA
                pbar.total = scheduled
                pbar.refresh()
                if len(pending) >= max_inflight:
                    done = drain_some(pending)
                    pbar.update(done)

            # Drain remaining
            while pending:
                done = drain_some(pending)
                pbar.update(done)

            pbar.close()

        # Final flush
        if rows_buffer:
            if out_format == "parquet":
                pd.DataFrame(rows_buffer).to_parquet(out_path, index=False)
            else:
                mode = "a" if os.path.exists(out_path) else "w"
                header = not os.path.exists(out_path)
                pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
                rows_buffer.clear()

    return None



def plot_multiband(dfv, dfg, ra, dec, peak_option=False):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
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
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
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



def plot_zoom(df, ra, dec, zoom_range=[-300,3000], peak_option=False):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
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
