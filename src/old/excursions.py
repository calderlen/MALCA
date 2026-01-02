from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from lc_utils import read_lc_dat2, read_lc_raw, match_index_to_lc
from old.lc_metrics import run_metrics, is_dip_dominated
from baseline import per_camera_median_baseline
from df_utils import clean_lc, empty_metrics

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from astropy.stats import biweight_location, biweight_scale

warnings.filterwarnings("ignore", message=".*Covariance of the parameters could not be estimated.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")

lc_dir_masked = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked"

MAG_BINS = ['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15']

                                                     
masked_bins = [f"{lc_dir_masked}/{b}" for b in MAG_BINS]

lc_12_12_5 = lc_dir_masked + '/12_12.5'
lc_12_5_13 = lc_dir_masked + '/12.5_13'
lc_13_13_5 = lc_dir_masked + '/13_13.5'
lc_13_5_14 = lc_dir_masked + '/13.5_14'
lc_14_14_5 = lc_dir_masked + '/14_14.5'
lc_14_5_15 = lc_dir_masked + '/14.5_15'

METRIC_NAMES = (
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

def gaussian(t, amp, mu, sig):
    """
    
    """
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)

def mag_to_delta(
    df,
    *,
    mag_col="mag",
    t_col="JD",
    err_col="error",
    biweight_c=6.0,
    eps=1e-6,
    sign: float = 1.0,
    baseline_func=None,
    baseline_kwargs: dict | None = None,
):
    """
    Compute delta relative to a robust baseline.

    - Default: global biweight location/scale on the full light curve.
    - If baseline_func is provided (e.g., from baseline.py), it is called as
      baseline_func(df, **baseline_kwargs). A "baseline" column is expected in the
      returned DataFrame; residuals are mag - baseline. A robust scale on residuals
      is then used in the denominator.
    - sign=+1 for dimmings (mag - baseline), sign=-1 for brightenings.
    """
    baseline_kwargs = baseline_kwargs or {}

    mag = np.asarray(df[mag_col], float) if mag_col in df.columns else np.array([], float)
    jd = np.asarray(df[t_col], float) if t_col in df.columns else np.array([], float)
    if err_col in df.columns:
        err = np.asarray(df[err_col], float)
    else:
        err = np.full_like(mag, np.nan, dtype=float)

    if baseline_func is None:
        finite_m = np.isfinite(mag)
        R = float(biweight_location(mag[finite_m], c=biweight_c)) if finite_m.any() else np.nan
        S = float(biweight_scale(mag[finite_m], c=biweight_c)) if finite_m.any() else 0.0
        resid = mag - R
    else:
        df_base = baseline_func(df, **baseline_kwargs)
        baseline_vals = (
            np.asarray(df_base["baseline"], float)
            if "baseline" in df_base.columns
            else np.asarray(df_base[mag_col], float)
        )
        resid = mag - baseline_vals
        finite_r = np.isfinite(resid)
        R = float(np.nanmedian(baseline_vals)) if np.isfinite(baseline_vals).any() else np.nan
        S = float(biweight_scale(resid[finite_r], c=biweight_c)) if finite_r.any() else 0.0

    if not np.isfinite(S) or S < 0:
        S = 0.0

    err2 = np.where(np.isfinite(err), err**2, 0.0)
    denom = np.sqrt(err2 + S**2)
    denom = np.where(denom > 0, denom, eps)
    delta = sign * resid / denom
    return delta, jd, R

def score_dips_gaussian(delta, jd, err, peak_idx, sigma_threshold):
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

    scores = []
    fwhm_list = []
    delta_peak_list = []
    n_det_list = []
    chi2_list = []

    for p in peak_idx:
        p = int(p)
        delta_peak = float(delta[p]) if 0 <= p < len(delta) else 0.0

                                                                                 
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
                                  
            fwhm = 0.0
            chi2_red = np.inf
            score = 0.0
        else:
            amp0 = delta_peak
            mu0 = float(jd[p])
            sig0 = max((t_win.max() - t_win.min()) / 6.0, 0.5)
            try:
                popt, _ = curve_fit(
                    gaussian,
                    t_win,
                    d_win,
                    p0=[amp0, mu0, sig0],
                    maxfev=min(2000, len(t_win) * 50),                               
                )
                amp, mu, sig = popt
                model = gaussian(t_win, amp, mu, sig)
                resid = d_win - model
                dof = max(n_det - 3, 1)
                chi2 = float(np.nansum(resid**2))
                chi2_red = chi2 / dof
                fwhm = float(2.3548 * abs(sig))
                                                                             
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
    
def paczynski(t, amp, t0, tE):
    """
    
    """
    tE = np.maximum(tE, 1e-6)
    return amp / np.sqrt(1.0 + ((t - t0) / tE) ** 2)

def score_peaks_paczynski(delta, jd, err, peak_idx, sigma_threshold):
    """
    Fit Paczynski-like microlensing curves to biweight-delta peaks and compute a heuristic score.

    Model: amp / sqrt(1 + ((t - t0) / tE)^2)
    FWHM for this model is 2 * sqrt(3) * tE.
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

    scores = []
    fwhm_list = []
    delta_peak_list = []
    n_det_list = []
    chi2_list = []

    for p in peak_idx:
        p = int(p)
        delta_peak = float(delta[p]) if 0 <= p < len(delta) else 0.0

                                                                                 
                                                           
        max_window_size = 500
        left = p
        left_steps = 0
        while left > 0 and delta[left] > 0.5 * sigma_threshold and left_steps < max_window_size:
            left -= 1
            left_steps += 1
        right = p
        nmax = len(delta) - 1
        right_steps = 0
        while right < nmax and delta[right] > 0.5 * sigma_threshold and right_steps < max_window_size:
            right += 1
            right_steps += 1

        window = slice(left, right + 1)
        t_win = jd[window]
        d_win = delta[window]
        n_det = len(d_win)
        if n_det < 3:
                                  
            fwhm = 0.0
            chi2_red = np.inf
            score = 0.0
        else:
            amp0 = max(delta_peak, 0.0)
            t0_0 = float(jd[p])
            window_span = max(t_win.max() - t_win.min(), 1e-3)
            tE0 = max(window_span / (2.0 * np.sqrt(3.0)), 0.5)
            half_width = window_span / 2.0
            try:
                popt, _ = curve_fit(
                    paczynski,
                    t_win,
                    d_win,
                    p0=[amp0, t0_0, tE0],
                    bounds=(
                        [0.0, t_win.min() - half_width, 1e-3],
                        [np.inf, t_win.max() + half_width, np.inf],
                    ),
                    maxfev=min(4000, len(t_win) * 100),                               
                )
                amp, t0_fit, tE = popt
                tE = abs(tE)
                model = paczynski(t_win, amp, t0_fit, tE)
                resid = d_win - model
                dof = max(n_det - 3, 1)
                chi2 = float(np.nansum(resid**2))
                chi2_red = chi2 / dof
                fwhm = float(2.0 * np.sqrt(3.0) * tE)
                                                                             
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

def lc_band_proc(
    df_band,
    *,
    mode: str,
    peak_kwargs: dict,
    sigma_threshold: float,
    metrics_dip_threshold: float,
    baseline_func=None,
    baseline_kwargs: dict | None = None,
):
    """
    Analyze a single band: clean, delta-transform, peak find, fit/score, metrics.
    """
    if df_band.empty or len(df_band) < 30:
        return {
            "n": 0,
            "R": np.nan,
            "peaks_idx": [],
            "peaks_jd": [],
            "fit": {
                "total_score": 0.0,
                "best_score": 0.0,
                "scores": [],
                "fwhm": [],
                "delta_peak": [],
                "n_det": [],
                "chi2_red": [],
            },
            "metrics": empty_metrics("x"),
        }

    df_band = clean_lc(df_band).sort_values("JD").reset_index(drop=True)
    if mode == "peaks":
        delta, jd, R = mag_to_delta(
            df_band,
            mag_col=peak_kwargs.get("mag_col", "mag"),
            t_col=peak_kwargs.get("t_col", "JD"),
            err_col=peak_kwargs.get("err_col", "error"),
            biweight_c=peak_kwargs.get("biweight_c", 6.0),
            eps=peak_kwargs.get("eps", 1e-6),
            sign=-1.0,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
        )
        fit = score_peaks_paczynski(
            delta,
            jd,
            df_band["error"].to_numpy(float) if "error" in df_band.columns else None,
            find_peaks(
                np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
                height=sigma_threshold,
                distance=int(peak_kwargs.get("distance", 50)),
            )[0],
            sigma_threshold,
        )
        peaks_idx = np.asarray(
            find_peaks(
                np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
                height=sigma_threshold,
                distance=int(peak_kwargs.get("distance", 50)),
            )[0],
            dtype=int,
        )
        metrics = empty_metrics("x")
    else:
        delta, jd, R = mag_to_delta(
            df_band,
            mag_col=peak_kwargs.get("mag_col", "mag"),
            t_col=peak_kwargs.get("t_col", "JD"),
            err_col=peak_kwargs.get("err_col", "error"),
            biweight_c=peak_kwargs.get("biweight_c", 6.0),
            eps=peak_kwargs.get("eps", 1e-6),
            sign=1.0,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
        )
        peaks_idx, _ = find_peaks(
            np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
            height=sigma_threshold,
            distance=int(peak_kwargs.get("distance", 50)),
        )
        fit = score_dips_gaussian(
            delta,
            jd,
            df_band["error"].to_numpy(float) if "error" in df_band.columns else None,
            peaks_idx,
            sigma_threshold,
        )
        stats = run_metrics(
            df_band,
            baseline_func=baseline_func,
            dip_threshold=metrics_dip_threshold,
            **(baseline_kwargs or {}),
        )
        metrics = {f"x_{k}": v for k, v in stats.items()}
        metrics["x_is_dip_dominated"] = bool(is_dip_dominated(stats))

    return {
        "n": len(peaks_idx),
        "R": R,
        "peaks_idx": peaks_idx.tolist(),
        "peaks_jd": df_band["JD"].values[peaks_idx].tolist() if len(peaks_idx) else [],
        "fit": fit,
        "metrics": metrics,
    }


def prefix_metrics(prefix: str, metrics_dict: dict | None):
    """
    Rename metrics keys from x_ -> <prefix>_ and ensure a filled dict when missing.
    """
    if not metrics_dict:
        return empty_metrics(prefix)
    if "x_is_dip_dominated" in metrics_dict:
        out = {}
        for k, v in metrics_dict.items():
            if k.startswith("x_"):
                out[f"{prefix}_{k[2:]}"] = v
            else:
                out[f"{prefix}_{k}"] = v
        return out
    return empty_metrics(prefix)

def lc_proc(
    record: dict,
    *,
    mode: str = "dips",
    peak_kwargs: dict | None = None,
    baseline_func=None,
    baseline_kwargs: dict | None = None,
    ) -> dict:
    """
    processes a single light curve

    mode="dips": use (mag-R) biweight delta, Gaussian fits, and dip metrics.
    mode="peaks": use (R-mag) biweight delta, Paczynski fits, no dip metrics.
    """
    peak_kwargs = dict(peak_kwargs or {})
    baseline_func = baseline_func or per_camera_median_baseline
    sigma_threshold = float(peak_kwargs.get("sigma_threshold", 3.0))
    metrics_dip_threshold = sigma_threshold

    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat2(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    g_res = lc_band_proc(
        dfg,
        mode=mode,
        peak_kwargs=peak_kwargs,
        sigma_threshold=sigma_threshold,
        metrics_dip_threshold=metrics_dip_threshold,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )
    v_res = lc_band_proc(
        dfv,
        mode=mode,
        peak_kwargs=peak_kwargs,
        sigma_threshold=sigma_threshold,
        metrics_dip_threshold=metrics_dip_threshold,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )

    if not dfg.empty:
        jd_first = float(dfg["JD"].iloc[0])
        jd_last = float(dfg["JD"].iloc[-1])
    if not dfv.empty:
        if np.isnan(jd_first):
            jd_first = float(dfv["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(dfv["JD"].iloc[-1])

                          
    g_metrics = prefix_metrics("g", g_res["metrics"])
    v_metrics = prefix_metrics("v", v_res["metrics"])

    row = {
        "mag_bin": record["mag_bin"],
        "asas_sn_id": asn,
        "index_num": record["index_num"],
        "index_csv": record["index_csv"],
        "lc_dir": record["lc_dir"],
        "dat_path": record["dat_path"],
        "raw_path": os.path.join(record["lc_dir"], f"{asn}.raw"),
        "g_n_peaks": g_res["n"],
        "g_biweight_R": g_res["R"],
        "g_peaks_idx": g_res["peaks_idx"],
        "g_peaks_jd": g_res["peaks_jd"],
        "g_total_score": g_res["fit"]["total_score"],
        "g_best_score": g_res["fit"]["best_score"],
        "g_scores": g_res["fit"]["scores"],
        "g_fwhm": g_res["fit"]["fwhm"],
        "g_delta_peak": g_res["fit"]["delta_peak"],
        "g_n_det": g_res["fit"]["n_det"],
        "g_chi2_red": g_res["fit"]["chi2_red"],
        "v_n_peaks": v_res["n"],
        "v_biweight_R": v_res["R"],
        "v_peaks_idx": v_res["peaks_idx"],
        "v_peaks_jd": v_res["peaks_jd"],
        "v_total_score": v_res["fit"]["total_score"],
        "v_best_score": v_res["fit"]["best_score"],
        "v_scores": v_res["fit"]["scores"],
        "v_fwhm": v_res["fit"]["fwhm"],
        "v_delta_peak": v_res["fit"]["delta_peak"],
        "v_n_det": v_res["fit"]["n_det"],
        "v_chi2_red": v_res["fit"]["chi2_red"],
        "jd_first": jd_first,
        "jd_last": jd_last,
        "n_rows_g": int(len(dfg)) if not dfg.empty else 0,
        "n_rows_v": int(len(dfv)) if not dfv.empty else 0,
    }
    row.update(g_metrics)
    row.update(v_metrics)

    if not raw_df.empty:
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite_scatter = scatter_vals[np.isfinite(scatter_vals)]
        row["raw_robust_scatter"] = float(np.nanmedian(finite_scatter)) if finite_scatter.size > 0 else np.nan
    else:
        row["raw_robust_scatter"] = np.nan

    return row

def excursion_finder(
    *,
    mode="dips",
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=MAG_BINS,
    id_column="asas_sn_id",
    out_dir=None,
    out_format="csv",
    n_workers=None,
    chunk_size=250000,
    max_inflight=None,
    peak_kwargs: dict | None = None,
    baseline_func=None,
    baseline_kwargs: dict | None = None,
    target_ids_by_bin: dict[str, set[str]] | None = None,
    records_by_bin: dict[str, list[dict[str, object]]] | None = None,
    return_rows: bool = False,
    ) -> pd.DataFrame | None:
    """
    Combined excursion finder (dips or peaks) using efficient process pool.
    """

    def _normalize_target_ids(target_ids_by_bin: dict[str, set[str]] | None):
        """
        
        """
        if not target_ids_by_bin:
            return None
        return {
            str(bin_key): {str(asas_id) for asas_id in ids if asas_id is not None}
            for bin_key, ids in target_ids_by_bin.items()
            if ids
        }

    def _build_record_map(records_by_bin: dict[str, list[dict[str, object]]] | None):
        """
        
        """
        if not records_by_bin:
            return None
        record_map: dict[str, list[dict[str, object]]] = {}
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
        return record_map

    def _iter_records_for_bin(
        bin_key: str,
        record_map: dict[str, list[dict[str, object]]] | None,
        target_ids_norm: dict[str, set[str]] | None,
    ):
        """
        
        """
        if record_map and bin_key in record_map:
            record_iter = record_map[bin_key]
        else:
            record_iter = match_index_to_lc(
                index_path=index_path,
                lc_path=lc_path,
                mag_bins=[bin_key],
                id_column=id_column,
            )

        for rec in record_iter:
            if target_ids_norm is not None:
                allowed = target_ids_norm.get(str(rec.get("mag_bin", bin_key)))
                if allowed is None or str(rec.get("asas_sn_id")) not in allowed:
                    continue
            if not rec.get("found", False):
                continue
            yield rec

    def _flush_rows(rows_buffer, *, force: bool = False):
        """
        
        """
        if not rows_buffer:
            return
        if len(rows_buffer) < chunk_size and not force:
            return
        try:
            if out_format == "parquet":
                pd.DataFrame(rows_buffer).to_parquet(out_path, index=False)
            else:
                mode = "a" if os.path.exists(out_path) else "w"
                header = not os.path.exists(out_path)
                pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
            rows_buffer.clear()
        except Exception as e:
            print(f"ERROR: Failed to flush rows to {out_path}: {e}", flush=True)
            raise

    def _process_bin(
        bin_key: str,
        record_iter,
        ex,
        rows_buffer,
        collected_rows_list,
    ):
        """
        
        """
        pending = set()
        scheduled = 0
        pbar = tqdm(desc=f"{bin_key}", unit="obj", leave=False)

        def drain_some(all_pending):
            """
            
            """
            done_now = 0
            done = [fut for fut in all_pending if fut.done()]
            for fut in done:
                try:
                    row = fut.result()
                    rows_buffer.append(row)
                    if collected_rows_list is not None:
                        collected_rows_list.append(row)
                    done_now += 1
                    _flush_rows(rows_buffer)
                except Exception as e:
                    print(f"WARNING: Task failed: {type(e).__name__}: {e}", flush=True)
                finally:
                    all_pending.remove(fut)
            return done_now

        for rec in record_iter:
            pending.add(
                ex.submit(
                    lc_proc,
                    rec,
                    mode=mode,
                    peak_kwargs=peak_kwargs,
                    baseline_func=baseline_func,
                    baseline_kwargs=baseline_kwargs,
                )
            )
            scheduled += 1
            pbar.total = scheduled
            pbar.refresh()
            if len(pending) >= max_inflight:
                done_n = drain_some(pending)
                pbar.update(done_n)

        while pending:
            done_n = drain_some(pending)
            pbar.update(done_n)

        pbar.close()
        _flush_rows(rows_buffer, force=True)

    peak_kwargs = dict(peak_kwargs or {})
    baseline_func = baseline_func or per_camera_median_baseline
    if out_dir is None:
        out_dir = f"./results_{mode}"
    
    os.makedirs(out_dir, exist_ok=True)
    
                   
    if n_workers is None:
        try:
            cpu = os.cpu_count() or 1
        except Exception:
            cpu = 1
        n_workers = max(1, min(48, cpu - 2))
    
    if max_inflight is None:
        max_inflight = max(4, n_workers * 4)

                 
    out_path = os.path.join(out_dir, f"excursions_{mode}.{out_format}")
    
                                                            
    target_ids_norm = _normalize_target_ids(target_ids_by_bin)

    record_map = _build_record_map(records_by_bin)

                               
    if record_map:
        bins_iter = [b for b in mag_bins if b in record_map]
        for extra in record_map.keys():
            if extra not in bins_iter:
                bins_iter.append(extra)
    else:
        bins_iter = [b for b in mag_bins if not target_ids_norm or b in target_ids_norm]

    collected_rows_list = [] if return_rows else None
    rows_buffer: list[dict] = []
    
                  
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for b in tqdm(bins_iter, desc=f"Bins ({mode})", unit="bin"):
            record_iter = _iter_records_for_bin(b, record_map, target_ids_norm)
            _process_bin(b, record_iter, ex, rows_buffer, collected_rows_list)

    if collected_rows_list is not None:
        return pd.DataFrame(collected_rows_list)
    return None

__all__ = ["MAG_BINS", "lc_proc", "excursion_finder"]

               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dip or microlensing finder across bins.")
    parser.add_argument("--mode", choices=("dips", "peaks"), default="dips", help="Select dips (biweight delta + Gaussian fits) or peaks (microlensing Paczynski fits).")
    parser.add_argument("--mag-bin", dest="mag_bins", action="append", choices=MAG_BINS, help="Specify bins to run; omit to process all.",)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--format", choices=("parquet", "csv"), default="csv")
    parser.add_argument("--n-workers", type=int, default=10, help="Parallel processes",)
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per CSV flush",)
    args = parser.parse_args()
    bins = args.mag_bins or MAG_BINS

    excursion_finder(
        mode=args.mode,
        mag_bins=bins,
        out_dir=args.out_dir,
        out_format=args.format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )
