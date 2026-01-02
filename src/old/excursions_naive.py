from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from baseline import per_camera_trend_baseline, per_camera_median_baseline
from lc_utils import read_lc_dat2, read_lc_raw, match_index_to_lc
from df_utils import peak_search_residual_baseline, clean_lc, empty_metrics
from lc_metrics import run_metrics, is_dip_dominated

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from astropy.stats import biweight_location, biweight_scale

lc_dir_masked = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked"

MAG_BINS = ['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15']

def lc_proc_naive(
    record: dict,
    baseline_kwargs: dict,
    *,
    baseline_func=per_camera_trend_baseline,
    metrics_baseline_func=None,
    metrics_dip_threshold=0.3,
):
    """
    
    """
    baseline_kwargs = dict(baseline_kwargs)

    metrics_baseline = metrics_baseline_func or baseline_func

    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat2(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    if not dfg.empty:
        dfg = clean_lc(dfg)
        dfg_baseline = (
            baseline_func(dfg, **baseline_kwargs)
            .reset_index(drop=True)
        )
        peaks_g, mean_g, n_g = peak_search_residual_baseline(dfg_baseline)
        jd_first = float(dfg_baseline["JD"].iloc[0])
        jd_last = float(dfg_baseline["JD"].iloc[-1])
        g_stats = run_metrics(
            dfg,
            baseline_func=metrics_baseline,
            dip_threshold=metrics_dip_threshold,
            **baseline_kwargs,
        )
        g_metrics = {f"g_{k}": v for k, v in g_stats.items()}
        g_metrics["g_is_dip_dominated"] = bool(is_dip_dominated(g_stats))
        peak_idx_g = peaks_g.to_numpy(dtype=int) if n_g > 0 else np.array([], dtype=int)
        g_peaks_idx = peak_idx_g.tolist()
        g_peaks_jd = dfg_baseline["JD"].values[peak_idx_g].tolist() if peak_idx_g.size else []
    else:
        dfg = pd.DataFrame()
        peaks_g, mean_g, n_g = (pd.Series(dtype=int), np.nan, 0)
        g_metrics = empty_metrics("g")
        g_peaks_idx = []
        g_peaks_jd = []

    if not dfv.empty:
        dfv = clean_lc(dfv)
        dfv_baseline = (
            baseline_func(dfv, **baseline_kwargs)
            .reset_index(drop=True)
        )
        peaks_v, mean_v, n_v = peak_search_residual_baseline(dfv_baseline)
        if np.isnan(jd_first):
            jd_first = float(dfv_baseline["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(dfv_baseline["JD"].iloc[-1])
        v_stats = run_metrics(
            dfv,
            baseline_func=metrics_baseline,
            dip_threshold=metrics_dip_threshold,
            **baseline_kwargs,
        )
        v_metrics = {f"v_{k}": v for k, v in v_stats.items()}
        v_metrics["v_is_dip_dominated"] = bool(is_dip_dominated(v_stats))
        peak_idx_v = peaks_v.to_numpy(dtype=int) if n_v > 0 else np.array([], dtype=int)
        v_peaks_idx = peak_idx_v.tolist()
        v_peaks_jd = dfv_baseline["JD"].values[peak_idx_v].tolist() if peak_idx_v.size else []
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
        
                                                             
                                                                                      
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite_scatter = scatter_vals[np.isfinite(scatter_vals)]
        row["raw_robust_scatter"] = float(np.nanmedian(finite_scatter)) if finite_scatter.size > 0 else np.nan
    else:
        row["raw_median_min"] = np.nan
        row["raw_median_min_camera"] = np.nan
        row["raw_median_max"] = np.nan
        row["raw_median_max_camera"] = np.nan
        row["raw_median_range"] = np.nan
        row["raw_robust_scatter"] = np.nan

    return row


                                               
def dip_finder_naive(
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=('12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'),
    id_column="asas_sn_id",
    out_dir="./peak_results",
    out_format="csv",
    n_workers=None,
    chunk_size=250000,
    max_inflight=None,
    baseline_func=per_camera_median_baseline,
    metrics_baseline_func=None,
    metrics_dip_threshold=0.3,
    target_ids_by_bin: dict[str, set[str]] | None = None,
    records_by_bin: dict[str, list[dict[str, object]]] | None = None,
    return_rows: bool = False,
    **baseline_kwargs,
):
    """
    run the naive dip search across one or more magnitude bins.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")

                               
    if n_workers is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_workers = max(1, min(48, cpu - 2))
    if max_inflight is None:
        max_inflight = max(4, n_workers * 4)

    target_ids_norm: dict[str, set[str]] | None = None
    if target_ids_by_bin:
        target_ids_norm = {
            str(bin_key): {str(asas_id) for asas_id in ids if asas_id is not None}
            for bin_key, ids in target_ids_by_bin.items()
            if ids
        }

    record_map: dict[str, list[dict[str, object]]] | None = None
    if records_by_bin:
        record_map = {}
        for bin_key, records in records_by_bin.items():
            norm_key = str(bin_key)
            if not records:
                continue
            record_map[norm_key] = []
            for rec in records:
                if rec is None:
                    continue
                norm = dict(rec)
                norm["mag_bin"] = str(norm.get("mag_bin", norm_key))
                norm["asas_sn_id"] = str(norm.get("asas_sn_id"))
                if not norm["asas_sn_id"]:
                    continue
                norm.setdefault("lc_dir", "")
                norm.setdefault("index_num", None)
                norm.setdefault("index_csv", None)
                norm.setdefault("dat_path", os.path.join(norm["lc_dir"], f"{norm['asas_sn_id']}.dat"))
                norm.setdefault("found", True)
                record_map[norm_key].append(norm)

    if record_map:
        bins_iter = [b for b in mag_bins if b in record_map]
        for extra in record_map.keys():
            if extra not in bins_iter:
                bins_iter.append(extra)
    else:
        bins_iter = [b for b in mag_bins if not target_ids_norm or b in target_ids_norm]

    collected_rows: list[dict] | None = [] if return_rows else None

    for b in tqdm(bins_iter, desc="Bins", unit="bin"):
        suffix = f"_{timestamp}" if timestamp else ""
        out_path = os.path.join(out_dir, f"peaks_{b.replace('.','_')}{suffix}.{out_format}")
        if os.path.exists(out_path):
            continue

        rows_buffer = []
        if out_format == "csv":
            header_written = False
        else:
            header_written = None                      

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            pending = set()
            pbar = tqdm(desc=f"{b} peak search", unit="obj", leave=False)
            scheduled = 0

                                                                 
            def flush_if_needed():
                """
                
                """
                if out_format == "csv" and len(rows_buffer) >= chunk_size:
                    mode = "a" if os.path.exists(out_path) else "w"
                    header = not os.path.exists(out_path)
                    pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
                    rows_buffer.clear()


            def drain_some(all_pending):
                """
                
                """
                done_now = 0
                for fut in list(all_pending):
                    if fut.done():
                        row = fut.result()
                        rows_buffer.append(row)
                        if collected_rows is not None:
                            collected_rows.append(row)
                        all_pending.remove(fut)
                        done_now += 1
                        flush_if_needed()
                return done_now

            if record_map and b in record_map:
                record_iter = record_map[b]
            else:
                record_iter = match_index_to_lc(
                    index_path=index_path,
                    lc_path=lc_path,
                    mag_bins=[b],
                    id_column=id_column,
                )

            for rec in record_iter:
                if target_ids_norm is not None:
                    allowed = target_ids_norm.get(str(rec.get("mag_bin", b)))
                    if allowed is None or str(rec.get("asas_sn_id")) not in allowed:
                        continue
                if not rec.get("found", False):
                    continue
                pending.add(
                    ex.submit(
                        lc_proc_naive,
                        rec,
                        baseline_kwargs,
                        baseline_func=baseline_func,
                        metrics_baseline_func=metrics_baseline_func,
                        metrics_dip_threshold=metrics_dip_threshold,
                    )
                )
                scheduled += 1
                                                
                pbar.total = scheduled
                pbar.refresh()
                if len(pending) >= max_inflight:
                    done = drain_some(pending)
                    pbar.update(done)

                             
            while pending:
                done = drain_some(pending)
                pbar.update(done)

            pbar.close()

                     
        if rows_buffer:
            if out_format == "parquet":
                pd.DataFrame(rows_buffer).to_parquet(out_path, index=False)
            else:
                mode = "a" if os.path.exists(out_path) else "w"
                header = not os.path.exists(out_path)
                pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
                rows_buffer.clear()

    if collected_rows is not None:
        return pd.DataFrame(collected_rows)
    return None
