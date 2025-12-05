from __future__ import annotations

import argparse  # <--- Added this
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from lc_baseline import per_camera_trend_baseline, per_camera_median_baseline, per_camera_mean_baseline, global_mean_baseline
from lc_utils import read_lc_dat, read_lc_raw, match_index_to_lc
from df_utils import peak_search_baseline_residual, peak_search_biweight_delta
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from astropy.stats import biweight_location, biweight_scale
from lc_metrics import run_metrics, is_dip_dominated

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

def process_record_naive(
    record: dict,
    baseline_kwargs: dict,
    *,
    baseline_func=per_camera_trend_baseline,
    metrics_baseline_func=None,
    metrics_dip_threshold=0.3,
):
    """
    worker function that processes a single lc
    """
    baseline_kwargs = dict(baseline_kwargs)

    metrics_baseline = metrics_baseline_func or baseline_func

    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    if not dfg.empty:
        dfg = clean_lc(dfg)
        dfg_baseline = (
            baseline_func(dfg, **baseline_kwargs)
            .sort_values("JD")
            .reset_index(drop=True)
        )
        peaks_g, mean_g, n_g = peak_search_baseline_residual(dfg_baseline)
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
            .sort_values("JD")
            .reset_index(drop=True)
        )
        peaks_v, mean_v, n_v = peak_search_baseline_residual(dfv_baseline)
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
        
        # --- ADDED OPTIMIZATION FROM PREVIOUS DISCUSSION ---
        # Calculating raw scatter here to avoid I/O bottlenecks during filtering later
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


def _biweight_gaussian_metrics(delta, jd, err, peak_idx, sigma_threshold):
    """
    Fit simple Gaussians to biweight-delta peaks and compute a heuristic score.
    """
    out = {
        "total_score": 0.0,
        "best_score": 0.0,
        "scores": [],
        "fwhm": [],
        "delta_peak": [],
        "n_det": [],
        "chi2_red": [],
    }

    if len(peak_idx) == 0 or delta.size == 0:
        return out

    delta = np.asarray(delta, float)
    jd = np.asarray(jd, float)
    err = np.asarray(err, float) if err is not None else np.full_like(delta, np.nan)

    def gauss(t, amp, mu, sig):
        return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)

    scores = []
    fwhm_list = []
    delta_peak_list = []
    n_det_list = []
    chi2_list = []

    for p in peak_idx:
        p = int(p)
        delta_peak = float(delta[p]) if 0 <= p < len(delta) else 0.0

        # find window where delta returns to <= 0.5 sigma_threshold on both sides
        left = p
        while left > 0 and delta[left] > 0.5 * sigma_threshold:
            left -= 1
        right = p
        nmax = len(delta) - 1
        while right < nmax and delta[right] > 0.5 * sigma_threshold:
            right += 1

        window = slice(left, right + 1)
        t_win = jd[window]
        d_win = delta[window]
        n_det = len(d_win)
        if n_det < 2:
            # fallback, cannot fit
            fwhm = 0.0
            chi2_red = np.inf
            score = 0.0
        else:
            amp0 = delta_peak
            mu0 = float(jd[p])
            sig0 = max((t_win.max() - t_win.min()) / 6.0, 0.5)
            try:
                popt, _ = curve_fit(
                    gauss,
                    t_win,
                    d_win,
                    p0=[amp0, mu0, sig0],
                    maxfev=2000,
                )
                amp, mu, sig = popt
                model = gauss(t_win, amp, mu, sig)
                resid = d_win - model
                dof = max(n_det - 3, 1)
                chi2 = float(np.nansum(resid**2))
                chi2_red = chi2 / dof
                fwhm = float(2.3548 * abs(sig))
                # paper score term: (delta/2) * FWHM * N_det * (1 / chi2_red)
                score = float(
                    (max(delta_peak, 0.0) / 2.0)
                    * max(fwhm, 0.0)
                    * max(n_det, 1)
                    / max(chi2_red, 1e-6)
                )
            except Exception:
                fwhm = 0.0
                chi2_red = np.inf
                score = 0.0

        scores.append(score)
        fwhm_list.append(fwhm)
        delta_peak_list.append(delta_peak)
        n_det_list.append(n_det)
        chi2_list.append(chi2_red)

    N = len(scores)
    if N > 0:
        denom = float(np.log(N + 1) ** N)
        denom = denom if np.isfinite(denom) and denom > 0 else 1.0
        total_score = float(np.nansum(scores) / denom)
        best_score = float(np.nanmax(scores))
    else:
        total_score = 0.0
        best_score = 0.0

    out.update(
        {
            "total_score": total_score,
            "best_score": best_score,
            "scores": scores,
            "fwhm": fwhm_list,
            "delta_peak": delta_peak_list,
            "n_det": n_det_list,
            "chi2_red": chi2_list,
        }
    )
    return out


def _compute_biweight_delta(df, *, mag_col="mag", t_col="JD", err_col="error", biweight_c=6.0, eps=1e-6):
    """
    Compute biweight delta series (global location/scale) for a light curve.
    """
    mag = np.asarray(df[mag_col], float) if mag_col in df.columns else np.array([], float)
    jd = np.asarray(df[t_col], float) if t_col in df.columns else np.array([], float)
    if err_col in df.columns:
        err = np.asarray(df[err_col], float)
    else:
        err = np.full_like(mag, np.nan, dtype=float)

    finite_m = np.isfinite(mag)
    R = float(biweight_location(mag[finite_m], c=biweight_c)) if finite_m.any() else np.nan
    S = float(biweight_scale(mag[finite_m], c=biweight_c)) if finite_m.any() else 0.0
    if not np.isfinite(S) or S < 0:
        S = 0.0

    err2 = np.where(np.isfinite(err), err**2, 0.0)
    denom = np.sqrt(err2 + S**2)
    denom = np.where(denom > 0, denom, eps)
    delta = (mag - R) / denom
    return delta, jd, R


def _compute_biweight_delta_peaks(df, *, mag_col="mag", t_col="JD", err_col="error", biweight_c=6.0, eps=1e-6):
    """
    Peak-oriented version of biweight delta where brightenings are positive.

    Uses (R - mag) so that microlensing-like peaks (smaller magnitudes) produce
    positive delta values suitable for peak finding.
    """
    mag = np.asarray(df[mag_col], float) if mag_col in df.columns else np.array([], float)
    jd = np.asarray(df[t_col], float) if t_col in df.columns else np.array([], float)
    if err_col in df.columns:
        err = np.asarray(df[err_col], float)
    else:
        err = np.full_like(mag, np.nan, dtype=float)

    finite_m = np.isfinite(mag)
    R = float(biweight_location(mag[finite_m], c=biweight_c)) if finite_m.any() else np.nan
    S = float(biweight_scale(mag[finite_m], c=biweight_c)) if finite_m.any() else 0.0
    if not np.isfinite(S) or S < 0:
        S = 0.0

    err2 = np.where(np.isfinite(err), err**2, 0.0)
    denom = np.sqrt(err2 + S**2)
    denom = np.where(denom > 0, denom, eps)
    delta = (R - mag) / denom
    return delta, jd, R


def process_record_biweight(
    record: dict,
    baseline_kwargs: dict,
    *,
    baseline_func=per_camera_trend_baseline,
    metrics_baseline_func=None,
    metrics_dip_threshold=0.3,
    peak_kwargs: dict | None = None,
):
    """
    Worker that uses the robust biweight-delta peak searcher.
    """
    baseline_kwargs = dict(baseline_kwargs)
    peak_kwargs = dict(peak_kwargs or {})
    sigma_threshold = float(peak_kwargs.get("sigma_threshold", 3.0))
    metrics_baseline = metrics_baseline_func or baseline_func

    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    if not dfg.empty and len(dfg) >= 50:
        dfg = clean_lc(dfg)
        # baseline not required for biweight delta, but keep sort/reset for consistency
        dfg = dfg.sort_values("JD").reset_index(drop=True)
        delta_g, jd_g, R_g = _compute_biweight_delta(dfg, **peak_kwargs)
        peaks_g, _ = find_peaks(
            np.nan_to_num(delta_g, nan=0.0, posinf=0.0, neginf=0.0),
            height=sigma_threshold,
            distance=int(peak_kwargs.get("distance", 50)),
        )
        n_g = len(peaks_g)
        g_fit = _biweight_gaussian_metrics(
            delta_g, jd_g, dfg["error"].to_numpy(float) if "error" in dfg.columns else None, peaks_g, sigma_threshold
        )
        jd_first = float(dfg["JD"].iloc[0])
        jd_last = float(dfg["JD"].iloc[-1])
        g_stats = run_metrics(
            dfg,
            baseline_func=metrics_baseline,
            dip_threshold=metrics_dip_threshold,
            **baseline_kwargs,
        )
        g_metrics = {f"g_{k}": v for k, v in g_stats.items()}
        g_metrics["g_is_dip_dominated"] = bool(is_dip_dominated(g_stats))
        peak_idx_g = np.asarray(peaks_g, dtype=int) if n_g > 0 else np.array([], dtype=int)
        g_peaks_idx = peak_idx_g.tolist()
        g_peaks_jd = dfg["JD"].values[peak_idx_g].tolist() if peak_idx_g.size else []
    else:
        dfg = pd.DataFrame()
        delta_g = np.array([], float)
        jd_g = np.array([], float)
        peaks_g, R_g, n_g = (pd.Series(dtype=int), np.nan, 0)
        g_metrics = empty_metrics("g")
        g_peaks_idx = []
        g_peaks_jd = []
        g_fit = {
            "total_score": 0.0,
            "best_score": 0.0,
            "scores": [],
            "fwhm": [],
            "delta_peak": [],
            "n_det": [],
            "chi2_red": [],
        }

    if not dfv.empty and len(dfv) >= 50:
        dfv = clean_lc(dfv)
        dfv = dfv.sort_values("JD").reset_index(drop=True)
        delta_v, jd_v, R_v = _compute_biweight_delta(dfv, **peak_kwargs)
        peaks_v, _ = find_peaks(
            np.nan_to_num(delta_v, nan=0.0, posinf=0.0, neginf=0.0),
            height=sigma_threshold,
            distance=int(peak_kwargs.get("distance", 50)),
        )
        n_v = len(peaks_v)
        v_fit = _biweight_gaussian_metrics(
            delta_v, jd_v, dfv["error"].to_numpy(float) if "error" in dfv.columns else None, peaks_v, sigma_threshold
        )
        if np.isnan(jd_first):
            jd_first = float(dfv["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(dfv["JD"].iloc[-1])
        v_stats = run_metrics(
            dfv,
            baseline_func=metrics_baseline,
            dip_threshold=metrics_dip_threshold,
            **baseline_kwargs,
        )
        v_metrics = {f"v_{k}": v for k, v in v_stats.items()}
        v_metrics["v_is_dip_dominated"] = bool(is_dip_dominated(v_stats))
        peak_idx_v = np.asarray(peaks_v, dtype=int) if n_v > 0 else np.array([], dtype=int)
        v_peaks_idx = peak_idx_v.tolist()
        v_peaks_jd = dfv["JD"].values[peak_idx_v].tolist() if peak_idx_v.size else []
    else:
        dfv = pd.DataFrame()
        peaks_v, R_v, n_v = (pd.Series(dtype=int), np.nan, 0)
        v_metrics = empty_metrics("v")
        v_peaks_idx = []
        v_peaks_jd = []
        v_fit = {
            "total_score": 0.0,
            "best_score": 0.0,
            "scores": [],
            "fwhm": [],
            "delta_peak": [],
            "n_det": [],
            "chi2_red": [],
        }

    row = {
        "mag_bin": record["mag_bin"],
        "asas_sn_id": asn,
        "index_num": record["index_num"],
        "index_csv": record["index_csv"],
        "lc_dir": record["lc_dir"],
        "dat_path": record["dat_path"],
        "raw_path": os.path.join(record["lc_dir"], f"{asn}.raw"),
        "g_n_peaks": n_g,
        "g_biweight_R": R_g,
        "g_peaks_idx": g_peaks_idx,
        "g_peaks_jd": g_peaks_jd,
        "g_total_score": g_fit["total_score"],
        "g_best_score": g_fit["best_score"],
        "g_scores": g_fit["scores"],
        "g_fwhm": g_fit["fwhm"],
        "g_delta_peak": g_fit["delta_peak"],
        "g_n_det": g_fit["n_det"],
        "g_chi2_red": g_fit["chi2_red"],
        "v_n_peaks": n_v,
        "v_biweight_R": R_v,
        "v_peaks_idx": v_peaks_idx,
        "v_peaks_jd": v_peaks_jd,
        "v_total_score": v_fit["total_score"],
        "v_best_score": v_fit["best_score"],
        "v_scores": v_fit["scores"],
        "v_fwhm": v_fit["fwhm"],
        "v_delta_peak": v_fit["delta_peak"],
        "v_n_det": v_fit["n_det"],
        "v_chi2_red": v_fit["chi2_red"],
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


# naive dip finder that works one bin at a time
def naive_dip_finder(
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
    """Run the naive dip search across one or more magnitude bins.

    Args:
        target_ids_by_bin: optional mapping of ``mag_bin`` to set of ASAS-SN IDs to check if some subset of candidates are reproduced.
        return_rows: when True, return a DataFrame of all processed rows instead of just writing peak CSV/Parquet files.
    """
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
            header_written = None  # unused for parquet

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            pending = set()
            pbar = tqdm(desc=f"{b} peak search", unit="obj", leave=False)
            scheduled = 0

            # flushes chunk_size amount of data when it's reached
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
                        process_record_naive,
                        rec,
                        baseline_kwargs,
                        baseline_func=baseline_func,
                        metrics_baseline_func=metrics_baseline_func,
                        metrics_dip_threshold=metrics_dip_threshold,
                    )
                )
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

    if collected_rows is not None:
        return pd.DataFrame(collected_rows)
    return None


def biweight_dip_finder(
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=('12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'),
    id_column="asas_sn_id",
    out_dir="./peak_results_biweight",
    out_format="csv",
    n_workers=None,
    chunk_size=250000,
    max_inflight=None,
    baseline_func=per_camera_median_baseline,
    metrics_baseline_func=None,
    metrics_dip_threshold=0.3,
    peak_kwargs: dict | None = None,
    target_ids_by_bin: dict[str, set[str]] | None = None,
    records_by_bin: dict[str, list[dict[str, object]]] | None = None,
    return_rows: bool = False,
    **baseline_kwargs,
):
    """
    Dip finder that uses the robust biweight-delta peak searcher. Mirrors naive_dip_finder but swaps in peak_search_biweight_delta. This means that no baseline is required.
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

    for b in tqdm(bins_iter, desc="Bins (biweight)", unit="bin"):
        suffix = f"_{timestamp}" if timestamp else ""
        out_path = os.path.join(out_dir, f"peaks_biweight_{b.replace('.','_')}{suffix}.{out_format}")
        if os.path.exists(out_path):
            continue

        rows_buffer = []
        header_written = False if out_format == "csv" else None

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            pending = set()
            pbar = tqdm(desc=f"{b} biweight peak search", unit="obj", leave=False)
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
                        process_record_biweight,
                        rec,
                        baseline_kwargs,
                        baseline_func=baseline_func,
                        metrics_baseline_func=metrics_baseline_func,
                        metrics_dip_threshold=metrics_dip_threshold,
                        peak_kwargs=peak_kwargs,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run naive_dip_finder across bins.")
    parser.add_argument(
        "--mag-bin",
        dest="mag_bins",
        action="append",
        choices=MAG_BINS,
        help="Specify bins to run; omit to process all.",
    )
    parser.add_argument("--out-dir", default="./results_peaks")
    parser.add_argument("--format", choices=("parquet", "csv"), default="csv")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=10,
        help="Parallel processes",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250000,
        help="Rows per CSV flush",
    )
    parser.add_argument(
        "--metrics-baseline",
        dest="metrics_baseline",
        default=None,
        help="Baseline function import path (e.g. 'lc_baseline:per_camera_trend_baseline')",
    )
    parser.add_argument(
        "--metrics-dip-threshold",
        dest="metrics_dip_threshold",
        type=float,
        default=0.3,
        help="Dip threshold used by run_metrics / run_metrics_pcb.",
    )

    args = parser.parse_args()
    bins = args.mag_bins or MAG_BINS

    metrics_baseline_func = None
    if args.metrics_baseline:
        try:
            module_path, func_name = args.metrics_baseline.split(":")
            module = __import__(module_path, fromlist=[func_name])
            metrics_baseline_func = getattr(module, func_name)
            print(f"[CLI] Loaded metrics_baseline_func: {func_name} from {module_path}")
        except Exception as e:
            print(f"[CLI] Error loading --metrics-baseline '{args.metrics_baseline}': {e}")
            exit(1)

    naive_dip_finder(
        mag_bins=bins,
        out_dir=args.out_dir,
        out_format=args.format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        metrics_baseline_func=metrics_baseline_func,
        metrics_dip_threshold=args.metrics_dip_threshold,
    )