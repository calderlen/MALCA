import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from lc_baseline import per_camera_baseline
from lc_utils import read_lc_dat, read_lc_raw, match_index_to_lc
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
def process_record_naive(record: dict, pcb_kwargs: dict):
    """
    Worker function to process a single ASAS-SN target. Returns a dict row.
    """
    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    if not dfg.empty:
        dfg = clean_lc(dfg)
        dfg_pcb = (
            per_camera_baseline(dfg, **pcb_kwargs)
            .sort_values("JD")
            .reset_index(drop=True)
        )
        peaks_g, mean_g, n_g = naive_peak_search(dfg_pcb)
        jd_first = float(dfg_pcb["JD"].iloc[0])
        jd_last = float(dfg_pcb["JD"].iloc[-1])
        g_stats = run_metrics_pcb(dfg, **pcb_kwargs)
        g_metrics = {f"g_{k}": v for k, v in g_stats.items()}
        g_metrics["g_is_dip_dominated"] = bool(is_dip_dominated(g_stats))
        peak_idx_g = peaks_g.to_numpy(dtype=int) if n_g > 0 else np.array([], dtype=int)
        g_peaks_idx = peak_idx_g.tolist()
        g_peaks_jd = dfg_pcb["JD"].values[peak_idx_g].tolist() if peak_idx_g.size else []
    else:
        dfg = pd.DataFrame()
        peaks_g, mean_g, n_g = (pd.Series(dtype=int), np.nan, 0)
        g_metrics = empty_metrics("g")
        g_peaks_idx = []
        g_peaks_jd = []

    if not dfv.empty:
        dfv = clean_lc(dfv)
        dfv_pcb = (
            per_camera_baseline(dfv, **pcb_kwargs)
            .sort_values("JD")
            .reset_index(drop=True)
        )
        peaks_v, mean_v, n_v = naive_peak_search(dfv_pcb)
        if np.isnan(jd_first):
            jd_first = float(dfv_pcb["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(dfv_pcb["JD"].iloc[-1])
        v_stats = run_metrics_pcb(dfv, **pcb_kwargs)
        v_metrics = {f"v_{k}": v for k, v in v_stats.items()}
        v_metrics["v_is_dip_dominated"] = bool(is_dip_dominated(v_stats))
        peak_idx_v = peaks_v.to_numpy(dtype=int) if n_v > 0 else np.array([], dtype=int)
        v_peaks_idx = peak_idx_v.tolist()
        v_peaks_jd = dfv_pcb["JD"].values[peak_idx_v].tolist() if peak_idx_v.size else []
    else:
        dfv = pd.DataFrame()
        peaks_v, mean_v, n_v = (pd.Series(dtype=int), np.nan, 0)
        v_metrics = empty_metrics("v")
        v_peaks_idx = []
        v_peaks_jd = []

    row = {
        "mag_bin": record["mag_bin"],
        "asas_sn_id": asn,
        "index_num": record["index_num"],
        "index_csv": record["index_csv"],
        "lc_dir": record["lc_dir"],
        "dat_path": record["dat_path"],
        "raw_path": os.path.join(record["lc_dir"], f"{asn}.raw"),
        "g_n_peaks": n_g,
        "g_mean_mag": mean_g,
        "g_peaks_idx": g_peaks_idx,
        "g_peaks_jd": g_peaks_jd,
        "v_n_peaks": n_v,
        "v_mean_mag": mean_v,
        "v_peaks_idx": v_peaks_idx,
        "v_peaks_jd": v_peaks_jd,
        "jd_first": jd_first,
        "jd_last": jd_last,
        "n_rows_g": int(len(dfg)) if not dfg.empty else 0,
        "n_rows_v": int(len(dfv)) if not dfv.empty else 0,
    }
    row.update(g_metrics)
    row.update(v_metrics)

    if not raw_df.empty:
        raw_median_min = float(raw_df["median"].min())
        raw_median_max = float(raw_df["median"].max())
        row["raw_median_min"] = raw_median_min
        row["raw_median_min_camera"] = int(raw_df.loc[raw_df["median"].idxmin(), "camera#"])
        row["raw_median_max"] = raw_median_max
        row["raw_median_max_camera"] = int(raw_df.loc[raw_df["median"].idxmax(), "camera#"])
        row["raw_median_range"] = raw_median_max - raw_median_min
    else:
        row["raw_median_min"] = np.nan
        row["raw_median_min_camera"] = np.nan
        row["raw_median_max"] = np.nan
        row["raw_median_max_camera"] = np.nan
        row["raw_median_range"] = np.nan

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
    chunk_size=250000,
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
                pending.add(ex.submit(process_record_naive, rec, pcb_kwargs))
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