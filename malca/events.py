import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
import glob
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # optional; only needed for parquet outputs
    pa = None
    pq = None

warnings.filterwarnings("ignore", message=".*Covariance of the parameters could not be estimated.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in.*", category=RuntimeWarning)

from malca.utils import read_lc_dat2, read_lc_csv, clean_lc, gaussian, paczynski_kernel, fred, skew_gaussian, filter_bad_cameras
from malca.baseline import (
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_trend_baseline,
)
from malca.score import compute_event_score
from malca.stats import log_gaussian, robust_median_dt_days, bic

from numba import njit


MAG_BINS = ['12_12.5', '12.5_13', '13_13.5', '13.5_14', '14_14.5', '14.5_15']

DEFAULT_BASELINE_KWARGS = dict(
    S0=0.0005,
    w0=0.0031415926535897933,
    q=0.7,
    jitter=0.006,
    sigma_floor=None,
    add_sigma_eff_col=True,
)

def logit_spaced_grid(p_min=1e-4, p_max=1.0 - 1e-4, n=12):
    """
    probability grid that is uniform in logit space, with minimum of 1e-12 and maximum of 1-1e-12
    """
    p_min = float(np.clip(p_min, 1e-12, 1 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1 - 1e-12))
    q_min = np.log(p_min / (1.0 - p_min))
    q_max = np.log(p_max / (1.0 - p_max))
    q = np.linspace(q_min, q_max, int(n))
    return 1.0 / (1.0 + np.exp(-q))


def uniform_p_grid(p_min=0.9, p_max=1.0 - 1e-6, n=36):
    """
    Uniform grid in p for approximating int(dp) with a uniform prior P(p)=const.
    """
    p_min = float(np.clip(p_min, 1e-12, 1.0 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1.0 - 1e-12))
    if not (p_min < p_max):
        raise ValueError(f"Require p_min < p_max, got {p_min=} {p_max=}")
    return np.linspace(p_min, p_max, int(n), dtype=float)



def default_mag_grid(baseline_mag: float, mags: np.ndarray, kind: str, n=12):
    """
    
    """
    mags_finite = mags[np.isfinite(mags)]
    if len(mags_finite) == 0:
        raise ValueError("No finite magnitude values for grid construction")
    lo, hi = np.nanpercentile(mags, [5, 95])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        med = np.nanmedian(mags)
        lo, hi = med - 0.5, med + 0.5
    spread = max(hi - lo, 0.05)

    if kind == "dip":
        start = baseline_mag + 0.02
        stop = max(baseline_mag + 0.02, hi + 0.5 * spread)
    elif kind == "jump":
        start = min(baseline_mag - 0.02, lo - 0.5 * spread)
        stop = baseline_mag - 0.02
    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    if start == stop:
        stop = start + (0.1 if kind == "dip" else -0.1)

    return np.linspace(start, stop, int(n))


def compute_symmetry_score(
    jd: np.ndarray,
    flux: np.ndarray,
    center_idx: int,
    start_idx: int,
    end_idx: int,
    baseline_flux: float | None = None,
) -> float:
    """
    Compute light-curve symmetry score (Tzanidakis+2025 Eq. 5).
    
    Compares integral areas from start-to-center vs center-to-end.
    Uses normalized flux (baseline = 1).
    
    Parameters
    ----------
    jd : np.ndarray
        Julian dates
    flux : np.ndarray
        Normalized flux values (baseline ~1, dip < 1)
    center_idx : int
        Index of dip center (minimum flux)
    start_idx : int
        Index of dip start (ingress begins)
    end_idx : int
        Index of dip end (egress completes)
    baseline_flux : float, optional
        Baseline flux level. If None, uses biweight mean of flux.
    
    Returns
    -------
    float
        Symmetry score:
        - Near 0: symmetric dip
        - Positive: longer ingress (slower rise to center)
        - Negative: longer egress (slower recovery from center)
    """
    jd = np.asarray(jd, float)
    flux = np.asarray(flux, float)
    
    # Validate indices
    if not (0 <= start_idx < center_idx < end_idx < len(jd)):
        return np.nan
    
    # Baseline estimation if not provided
    if baseline_flux is None:
        from astropy.stats import biweight_location
        baseline_flux = biweight_location(flux, ignore_nan=True)
    
    # Extract segments
    t_sc = jd[start_idx:center_idx + 1]
    f_sc = flux[start_idx:center_idx + 1]
    
    t_ce = jd[center_idx:end_idx + 1]
    f_ce = flux[center_idx:end_idx + 1]
    
    if len(t_sc) < 2 or len(t_ce) < 2:
        return np.nan
    
    # Compute integrals: I = integral of (baseline - flux) dt
    # Paper Eq. 4: I = sum over points of (baseline - flux_n) * dt
    # We use trapezoidal integration for better accuracy
    deviation_sc = baseline_flux - f_sc
    deviation_ce = baseline_flux - f_ce
    
    I_sc = np.trapezoid(deviation_sc, t_sc)
    I_ce = np.trapezoid(deviation_ce, t_ce)
    
    # Paper Eq. 5: symmetry_score = (I_sc + I_ce) / sqrt(I_sc^2 + I_ce^2)
    denominator = np.sqrt(I_sc**2 + I_ce**2)
    
    if denominator < 1e-10:
        return 0.0  # No significant dip
    
    return float((I_sc - I_ce) / denominator)


def classify_run_morphology(jd, mag, err, run_idx, kind="dip"):
    """
    fits gaussian vs paczynski vs noise kernel to an individual run, returns a dict with kernel chosen and params
    """
    pad = 5
    start_i = int(max(0, run_idx[0] - pad))
    end_i = int(min(len(jd), run_idx[-1] + pad + 1))
    
    t_seg = jd[start_i:end_i]
    y_seg = mag[start_i:end_i]
    e_seg = err[start_i:end_i]
    
    if len(t_seg) < 4:
        return {
            "morphology": "none", "bic": np.nan, "delta_bic_null": 0.0, "params": {}
        }

    baseline_guess = np.nanmedian(y_seg)
    abs_diff = np.abs(y_seg - baseline_guess)
    peak_local_idx = np.argmax(abs_diff)
    
    t0_guess = t_seg[peak_local_idx]
    amp_guess = y_seg[peak_local_idx] - baseline_guess
    sigma_guess = max((t_seg[-1] - t_seg[0]) / 4.0, 0.01)
    
    resid_null = y_seg - baseline_guess
    bic_null = bic(resid_null, e_seg, 1)
    
    best_bic = bic_null
    best_model = "noise"
    best_params = {}

    try:
        popt_g, _ = curve_fit(
            gaussian, t_seg, y_seg, 
            p0=[amp_guess, t0_guess, sigma_guess, baseline_guess],
            sigma=e_seg, maxfev=2000
        )
        resid_g = y_seg - gaussian(t_seg, *popt_g)
        bic_g = bic(resid_g, e_seg, 4)
        
        is_valid = (popt_g[0] > 0) if kind == "dip" else (popt_g[0] < 0)
        
        if is_valid and bic_g < (best_bic - 10):
            best_bic = bic_g
            best_model = "gaussian"
            best_params = {
                "amp": popt_g[0], "t0": popt_g[1], 
                "sigma": popt_g[2], "baseline": popt_g[3]
            }
    except Exception:
        pass

    # Try skew_gaussian for dips (asymmetric profiles)
    if kind == "dip":
        try:
            # Start with no skew (alpha=0), let optimizer find it
            popt_sg, _ = curve_fit(
                skew_gaussian, t_seg, y_seg,
                p0=[amp_guess, t0_guess, sigma_guess, baseline_guess, 0.0],
                sigma=e_seg, maxfev=3000,
                bounds=(
                    [-np.inf, t_seg[0], 1e-5, -np.inf, -10],  # lower bounds
                    [np.inf, t_seg[-1], np.inf, np.inf, 10]   # upper bounds
                )
            )
            resid_sg = y_seg - skew_gaussian(t_seg, *popt_sg)
            bic_sg = bic(resid_sg, e_seg, 5)  # 5 parameters

            is_valid_sg = (popt_sg[0] > 0)  # positive amp for dips

            # Only accept if BIC is significantly better (delta > 10)
            if is_valid_sg and bic_sg < (best_bic - 10):
                best_bic = bic_sg
                best_model = "skew_gaussian"
                best_params = {
                    "amp": popt_sg[0], "t0": popt_sg[1],
                    "sigma": popt_sg[2], "baseline": popt_sg[3],
                    "alpha": popt_sg[4]
                }
        except Exception:
            pass

    if kind == "jump":
        try:
            amp_p_guess = -abs(amp_guess)

            popt_p, _ = curve_fit(
                paczynski_kernel, t_seg, y_seg,
                p0=[amp_p_guess, t0_guess, sigma_guess, baseline_guess],
                sigma=e_seg, maxfev=2000
            )
            resid_p = y_seg - paczynski_kernel(t_seg, *popt_p)
            bic_p = bic(resid_p, e_seg, 4)

            is_valid_p = (popt_p[0] < 0)

            if is_valid_p and bic_p < (best_bic - 10):
                best_bic = bic_p
                best_model = "paczynski"
                best_params = {
                    "amp": popt_p[0], "t0": popt_p[1],
                    "tE": popt_p[2], "baseline": popt_p[3]
                }
        except Exception:
            pass

        # Try FRED for jumps (flares)
        try:
            # guess tau as small
            tau_guess = 0.05 
            
            popt_f, _ = curve_fit(
                fred, t_seg, y_seg,
                p0=[amp_guess, t0_guess, tau_guess, baseline_guess],
                sigma=e_seg, maxfev=2000
            )
            # For flares (jumps), amp should be negative (brightening in magnitudes)
            is_valid_f = (popt_f[0] < 0) 
            
            if is_valid_f:
                resid_f = y_seg - fred(t_seg, *popt_f)
                bic_f = bic(resid_f, e_seg, 4) # 4 params: amp, t0, tau, baseline
                
                if bic_f < (best_bic - 10):
                    best_bic = bic_f
                    best_model = "fred"
                    best_params = {
                        "amp": popt_f[0], "t0": popt_f[1],
                        "tau": popt_f[2], "baseline": popt_f[3]
                    }
        except Exception:
            pass

    return {
        "morphology": best_model,
        "bic": float(best_bic),
        "delta_bic_null": float(bic_null - best_bic),
        "params": best_params
    }



def build_runs(
    trig_idx: np.ndarray,
    jd: np.ndarray,
    *,
    allow_gap_points: int = 1,
    max_gap_days: float | None = None,
):
    """
    build runs from clustered triggered points
    """
    jd = np.asarray(jd, float)
    trig_idx = np.asarray(trig_idx, dtype=int)
    trig_idx = trig_idx[(trig_idx >= 0) & (trig_idx < jd.size)]
    if trig_idx.size == 0:
        return []

    trig_idx = np.unique(trig_idx)
    trig_idx.sort()

    if max_gap_days is None:
        # Use 99.73th percentile (3-sigma) of gaps between subsequent data points
        dt = np.diff(jd)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            max_gap_days = float(np.nanpercentile(dt, 99.73))
            # Ensure a reasonable minimum
            max_gap_days = max(max_gap_days, 1.0)
        else:
            max_gap_days = 5.0
    max_gap_days = float(max_gap_days)

    max_index_step = int(allow_gap_points) + 1

    runs = []
    cur = [int(trig_idx[0])]
    for k in range(1, trig_idx.size):
        i_prev = cur[-1]
        i = int(trig_idx[k])

        idx_step = i - i_prev
        dt = jd[i] - jd[i_prev]

        if (idx_step <= max_index_step) and np.isfinite(dt) and (dt <= max_gap_days):
            cur.append(i)
        else:
            runs.append(np.asarray(cur, dtype=int))
            cur = [i]
    runs.append(np.asarray(cur, dtype=int))
    return runs


def filter_runs(
    runs,
    jd: np.ndarray,
    score_vec: np.ndarray,
    *,
    min_points: int = 2,
    min_duration_days: float | None = None,
    per_point_threshold: float | None = None,
    cam_vec: np.ndarray | None = None,
):
    """
    filter runs by minimum points, minimum duration, per_point_threshold; returns dict of kept runs' starting and ending indices/JDs, number of points
    """
    jd = np.asarray(jd, float)
    score_vec = np.asarray(score_vec, float)

    cad = robust_median_dt_days(jd)
    if min_duration_days is None:
        if np.isfinite(cad):
            min_duration_days = max(2.0 * cad, 2.0)
        else:
            min_duration_days = 2.0
    min_duration_days = float(min_duration_days)

    kept = []
    summaries = []

    for r in runs:
        r = np.asarray(r, dtype=int)
        if r.size == 0:
            continue

        n = int(r.size)
        dur = float(jd[r[-1]] - jd[r[0]]) if n >= 2 else 0.0
        vals = score_vec[r]
        run_max = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
        run_sum = float(np.nansum(vals)) if np.isfinite(vals).any() else np.nan
        run_n_cameras = None
        if cam_vec is not None:
            cams = np.asarray(cam_vec[r])
            if cams.size:
                cams = cams[~pd.isna(cams)]
            run_n_cameras = int(np.unique(cams.astype(str)).size) if cams.size else 0

        ok = True
        if n < int(min_points):
            ok = False
        if dur < min_duration_days:
            ok = False
        if (per_point_threshold is not None) and (not (np.isfinite(run_max) and run_max >= float(per_point_threshold))):
            ok = False

        summaries.append(
            dict(
                start_idx=int(r[0]),
                end_idx=int(r[-1]),
                n_points=n,
                start_jd=float(jd[r[0]]),
                end_jd=float(jd[r[-1]]),
                duration_days=dur,
                run_max=run_max,
                run_sum=run_sum,
                run_n_cameras=run_n_cameras,
                kept=bool(ok),
            )
        )

        if ok:
            kept.append(r)

    return kept, summaries


def summarize_kept_runs(
    kept_runs,
    jd: np.ndarray,
    score_vec: np.ndarray,
    cam_vec: np.ndarray | None = None,
):
    """
    
    """
    jd = np.asarray(jd, float)
    score_vec = np.asarray(score_vec, float)

    if not kept_runs:
        return dict(
            n_runs=0,
            max_run_points=0,
            max_run_duration=np.nan,
            max_run_sum=np.nan,
            max_run_max=np.nan,
            max_run_cameras=0,
        )

    max_pts = 0
    max_dur = -np.inf
    max_sum = -np.inf
    max_max = -np.inf
    max_cams = 0

    for r in kept_runs:
        r = np.asarray(r, int)
        max_pts = max(max_pts, int(r.size))
        if r.size >= 2:
            max_dur = max(max_dur, float(jd[r[-1]] - jd[r[0]]))
        else:
            max_dur = max(max_dur, 0.0)

        vals = score_vec[r]
        if np.isfinite(vals).any():
            max_sum = max(max_sum, float(np.nansum(vals)))
            max_max = max(max_max, float(np.nanmax(vals)))
        if cam_vec is not None:
            cams = np.asarray(cam_vec[r])
            if cams.size:
                cams = cams[~pd.isna(cams)]
            run_n_cameras = int(np.unique(cams.astype(str)).size) if cams.size else 0
            max_cams = max(max_cams, run_n_cameras)

    return dict(
        n_runs=int(len(kept_runs)),
        max_run_points=int(max_pts),
        max_run_duration=float(max_dur) if np.isfinite(max_dur) else np.nan,
        max_run_sum=float(max_sum) if np.isfinite(max_sum) else np.nan,
        max_run_max=float(max_max) if np.isfinite(max_max) else np.nan,
        max_run_cameras=int(max_cams),
    )


@njit(fastmath=True, cache=True)
def compute_global_loglik_numba(log_Pb_grid, log_Pf_grid, log_p, log_1mp):
    """
    numba kernel for global loglikelihood computation
    """
    M, N = log_Pb_grid.shape
    P = log_p.shape[0]
    loglik = np.zeros((M, P), dtype=log_Pb_grid.dtype) 

    for m in range(M):
        for p in range(P):
            lp = log_p[p]
            l1mp = log_1mp[p]
            acc = 0.0
            
            for i in range(N):
                val_b = log_Pb_grid[m, i] + lp
                val_f = log_Pf_grid[m, i] + l1mp

                if val_b > val_f:
                    mix = val_b + np.log1p(np.exp(val_f - val_b))
                else:
                    mix = val_f + np.log1p(np.exp(val_b - val_f))
                
                acc += mix
            
            loglik[m, p] = acc
            
    return loglik

@njit(fastmath=True, cache=True)
def fast_loo_event_prob_numba(loglik, log_p, log_1mp, log_Pb_grid, log_Pf_grid, is_faint):
    """ 
    numba kernel for leave-one-out event probability computation 
    """
    M, P = loglik.shape
    _, N = log_Pb_grid.shape

    event_prob = np.zeros(N, dtype=np.float64)

    for i in range(N):
        max_b = -np.inf
        sum_b = 0.0

        max_f = -np.inf
        sum_f = 0.0

        for m in range(M):
            val_Pb = log_Pb_grid[m, i]
            val_Pf = log_Pf_grid[m, i]

            for p in range(P):
                t1 = log_p[p] + val_Pb
                t2 = log_1mp[p] + val_Pf

                if t1 > t2:
                    mix = t1 + np.log1p(np.exp(t2 - t1))
                else:
                    mix = t2 + np.log1p(np.exp(t1 - t2))

                ll_excl = loglik[m, p] - mix

                val_b = ll_excl + t1
                val_f = ll_excl + t2

                if val_b > max_b:
                    sum_b = sum_b * np.exp(max_b - val_b) + 1.0
                    max_b = val_b
                else:
                    sum_b += np.exp(val_b - max_b)

                if val_f > max_f:
                    sum_f = sum_f * np.exp(max_f - val_f) + 1.0
                    max_f = val_f
                else:
                    sum_f += np.exp(val_f - max_f)

        log_bright = max_b + np.log(sum_b)
        log_faint = max_f + np.log(sum_f)

        if log_bright > log_faint:
            log_norm = log_bright + np.log1p(np.exp(log_faint - log_bright))
        else:
            log_norm = log_faint + np.log1p(np.exp(log_bright - log_faint))
            
        if is_faint:
            event_prob[i] = np.exp(log_faint - log_norm)
        else:
            event_prob[i] = np.exp(log_bright - log_norm)
            
    return event_prob



def bayesian_event_significance(
    df: pd.DataFrame,
    *,
    kind: str = "dip",
    mag_col: str = "mag",
    err_col: str = "error",

    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,
    df_base: pd.DataFrame | None = None,

    use_sigma_eff: bool = True,
    require_sigma_eff: bool = False,

    p_min: float | None = None,
    p_max: float | None = None,
    p_points: int = 12,
    mag_grid: np.ndarray | None = None,
    mag_points: int = 12,

    trigger_mode: str = "posterior_prob", # posterior probability or logbf
    logbf_threshold: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 2,
    run_allow_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,

    compute_event_prob: bool = True,
):
    """
    Returns a dict including:
      - log_bf_local (N,)
      - event_probability (N,) if compute_event_prob
      - event_indices (after run gating)
      - significant (after run gating)
      - run diagnostics
      - global bayes_factor
    """
    # Only clean if df_base was not pre-computed; if df_base is provided,
    # the caller already cleaned df and df_base must match it.
    if df_base is None:
        df = clean_lc(df)
    cam_vec = df["camera#"].to_numpy() if "camera#" in df.columns else None
    jd = np.asarray(df["JD"], float)
    mags = np.asarray(df[mag_col], float)

    mags_finite = np.isfinite(mags).sum()
    mags_total = len(mags)
    if mags_finite == 0:
        raise ValueError(
            f"All magnitudes are NaN/inf after reading: "
            f"total={mags_total}, finite={mags_finite}, "
            f"NaN={np.isnan(mags).sum()}, inf={np.isinf(mags).sum()}"
        )

    errs = np.asarray(df[err_col], float)
    
    errs_finite = np.isfinite(errs).sum()
    errs_positive = (errs > 0).sum() if errs_finite > 0 else 0
    if errs_finite == 0:
        raise ValueError(
            f"All errors are NaN/inf: "
            f"total={len(errs)}, finite={errs_finite}, "
            f"NaN={np.isnan(errs).sum()}, inf={np.isinf(errs).sum()}"
        )
    if errs_positive == 0:
        raise ValueError(
            f"All errors are non-positive: "
            f"total={len(errs)}, finite={errs_finite}, positive={errs_positive}, "
            f"min={np.nanmin(errs) if errs_finite > 0 else 'N/A'}"
        )

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    used_sigma_eff = False

    if df_base is None and baseline_func is not None:
        df_base = baseline_func(df, **baseline_kwargs)

    if df_base is None:
        if not np.isfinite(mags).any():
            raise ValueError("All magnitude values are NaN/inf")
        baseline_mags = np.full_like(mags, np.nanmedian(mags))
        baseline_sources = np.full(len(mags), "global_median", dtype=object)
    else:
        if "baseline" in df_base.columns:
            baseline_mags = np.asarray(df_base["baseline"], float)
        else:
            baseline_mags = np.asarray(df_base[mag_col], float)
        if "baseline_source" in df_base.columns:
            baseline_sources = np.asarray(df_base["baseline_source"], dtype=object)
        else:
            baseline_sources = np.full(len(df_base), "unknown", dtype=object)

        if use_sigma_eff and ("sigma_eff" in df_base.columns):
            errs_new = np.asarray(df_base["sigma_eff"], float)
            errs_new_finite = np.isfinite(errs_new).sum()
            errs_new_positive = (errs_new > 0).sum() if errs_new_finite > 0 else 0
            if errs_new_finite == 0:
                raise ValueError(
                    f"Baseline returned all NaN/inf sigma_eff: "
                    f"total={len(errs_new)}, finite={errs_new_finite}, "
                    f"NaN={np.isnan(errs_new).sum()}, inf={np.isinf(errs_new).sum()}"
                )
            if errs_new_positive == 0:
                raise ValueError(
                    f"Baseline returned all non-positive sigma_eff: "
                    f"total={len(errs_new)}, finite={errs_new_finite}, positive={errs_new_positive}, "
                    f"min={np.nanmin(errs_new) if errs_new_finite > 0 else 'N/A'}"
                )
            errs = errs_new
            used_sigma_eff = True
        elif require_sigma_eff:
            raise RuntimeError("require_sigma_eff=True but baseline did not return 'sigma_eff'")

    baseline_finite = np.isfinite(baseline_mags).sum()
    if baseline_finite == 0:
        raise ValueError(
            f"Baseline function returned all NaN/inf values: "
            f"total={len(baseline_mags)}, finite={baseline_finite}, "
            f"NaN={np.isnan(baseline_mags).sum()}, inf={np.isinf(baseline_mags).sum()}, "
            f"baseline_func={baseline_func.__name__ if baseline_func else 'None'}"
        )
    
    errs_finite_final = np.isfinite(errs).sum()
    errs_positive_final = (errs > 0).sum() if errs_finite_final > 0 else 0
    if errs_finite_final == 0:
        raise ValueError(
            f"All errors are NaN/inf after baseline: "
            f"total={len(errs)}, finite={errs_finite_final}, "
            f"NaN={np.isnan(errs).sum()}, inf={np.isinf(errs).sum()}"
        )
    if errs_positive_final == 0:
        raise ValueError(
            f"All errors are non-positive after baseline: "
            f"total={len(errs)}, finite={errs_finite_final}, positive={errs_positive_final}, "
            f"min={np.nanmin(errs) if errs_finite_final > 0 else 'N/A'}"
        )
    
    total_points = len(mags)
    valid_mask = (
        np.isfinite(mags)
        & np.isfinite(errs)
        & (errs > 0)
        & np.isfinite(baseline_mags)
    )
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError(
            "No valid points after baseline/error filtering: "
            f"total={total_points}, finite_mags={np.isfinite(mags).sum()}, "
            f"finite_errs={np.isfinite(errs).sum()}, positive_errs={(errs > 0).sum()}, "
            f"finite_baseline={np.isfinite(baseline_mags).sum()}"
        )
    if n_valid < total_points:
        mags = mags[valid_mask]
        errs = errs[valid_mask]
        baseline_mags = baseline_mags[valid_mask]
        baseline_sources = baseline_sources[valid_mask]
        jd = jd[valid_mask]
        if cam_vec is not None:
            cam_vec = cam_vec[valid_mask]

    baseline_mag = float(np.nanmedian(baseline_mags))

    if p_min is None and p_max is None:
        if kind == "dip":
            p_min, p_max = 0.5, 1.0 - 1e-4
        elif kind == "jump":
            p_min, p_max = 1e-4, 0.5
        else:
            raise ValueError("kind must be 'dip' or 'jump'")

    p_grid = uniform_p_grid(p_min=p_min, p_max=p_max, n=p_points)

    if mag_grid is None:
        mag_grid = default_mag_grid(baseline_mag, mags, kind, n=mag_points)
    else:
        mag_grid = np.asarray(mag_grid, float)

    M = int(len(mag_grid))
    N = int(len(mags))

    if kind == "dip":
        log_Pb_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pb_grid = np.broadcast_to(log_Pb_vec, (M, N))
        log_Pf_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)
        event_component = "faint"

    elif kind == "jump":
        log_Pb_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)
        log_Pf_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pf_grid = np.broadcast_to(log_Pf_vec, (M, N))
        event_component = "bright"
        
        if not np.isfinite(log_Pf_vec).any():
            raise ValueError("All baseline likelihood values are NaN/inf")
        if not np.isfinite(log_Pb_grid).any():
            raise ValueError("All event likelihood values are NaN/inf")

    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    valid_points = (np.isfinite(log_Pb_grid).any(axis=0)) | (np.isfinite(log_Pf_grid).any(axis=0))
    n_valid_points = int(valid_points.sum())
    total_points = log_Pb_grid.shape[1]
    if n_valid_points == 0:
        raise ValueError(
            "No valid likelihood contributions after baseline: "
            f"total={total_points}, baseline_finite={np.isfinite(log_Pb_grid).sum()}, "
            f"event_finite={np.isfinite(log_Pf_grid).sum()}"
        )
    if n_valid_points < total_points:
        mags = mags[valid_points]
        errs = errs[valid_points]
        baseline_mags = baseline_mags[valid_points]
        baseline_sources = baseline_sources[valid_points]
        jd = jd[valid_points]
        if cam_vec is not None:
            cam_vec = cam_vec[valid_points]
        log_Pb_grid = log_Pb_grid[:, valid_points]
        log_Pf_grid = log_Pf_grid[:, valid_points]
        if kind == "dip":
            log_Pb_vec = log_Pb_vec[valid_points]
        else:
            log_Pf_vec = log_Pf_vec[valid_points]
        N = n_valid_points

    if kind == "dip":
        loglik_baseline_only = float(np.sum(log_Pb_vec))
        log_px_baseline = log_Pb_vec
        log_px_event = logsumexp(log_Pf_grid, axis=0) - np.log(M)
    else:
        loglik_baseline_only = float(np.sum(log_Pf_vec))
        log_px_baseline = log_Pf_vec
        log_px_event = logsumexp(log_Pb_grid, axis=0) - np.log(M)

    log_bf_local = log_px_event - log_px_baseline

    max_log_bf_local = float(np.nanmax(log_bf_local)) if np.isfinite(log_bf_local).any() else np.nan

    log_p = np.log(p_grid)
    log_1mp = np.log1p(-p_grid)

    loglik = compute_global_loglik_numba(
        np.ascontiguousarray(log_Pb_grid), 
        np.ascontiguousarray(log_Pf_grid), 
        log_p, 
        log_1mp
    )

    loglik_finite = np.isfinite(loglik).sum()
    loglik_total = loglik.size
    loglik_inf_neg = np.isinf(loglik) & (loglik < 0)
    loglik_inf_neg_count = loglik_inf_neg.sum()
    
    if loglik_finite == 0:
        if loglik_inf_neg_count == loglik_total:
            raise ValueError(
                f"All loglik values are -inf (all inputs were invalid): "
                f"total={loglik_total}, finite={loglik_finite}, -inf={loglik_inf_neg_count}, "
                f"This indicates all data points or baseline values were invalid."
            )
        else:
            raise ValueError(
                f"All loglik values are NaN/inf before normalization: "
                f"total={loglik_total}, finite={loglik_finite}, "
                f"NaN={np.isnan(loglik).sum()}, -inf={loglik_inf_neg_count}, +inf={np.isinf(loglik).sum() - loglik_inf_neg_count}"
            )
    
    loglik_sum = logsumexp(loglik)
    if not np.isfinite(loglik_sum):
        raise ValueError(
            f"logsumexp(loglik) is NaN/inf: "
            f"loglik_sum={loglik_sum}, loglik_finite={loglik_finite}/{loglik_total}, "
            f"loglik_min={np.nanmin(loglik) if loglik_finite > 0 else 'N/A'}, "
            f"loglik_max={np.nanmax(loglik) if loglik_finite > 0 else 'N/A'}"
        )
    
    log_post_norm = loglik - loglik_sum
    
    log_post_finite = np.isfinite(log_post_norm).sum()
    if log_post_finite == 0:
        raise ValueError(
            f"All log_posterior values are NaN/inf after normalization: "
            f"total={log_post_norm.size}, finite={log_post_finite}, "
            f"loglik_finite={loglik_finite}/{loglik_total}, loglik_sum={loglik_sum}, "
            f"loglik_range=[{np.nanmin(loglik) if loglik_finite > 0 else 'N/A'}, {np.nanmax(loglik) if loglik_finite > 0 else 'N/A'}]"
        )
    
    best_m_idx, best_p_idx = np.unravel_index(np.nanargmax(log_post_norm), log_post_norm.shape)
    best_mag_event = float(mag_grid[int(best_m_idx)])
    best_p = float(p_grid[int(best_p_idx)])

    K = loglik.size
    log_evidence_mixture = float(logsumexp(loglik) - np.log(K))
    bayes_factor = float(log_evidence_mixture - loglik_baseline_only)

    if compute_event_prob:
        event_prob = fast_loo_event_prob_numba(
                loglik,
                log_p,
                log_1mp,
                log_Pb_grid,
                log_Pf_grid,
                (event_component == "faint")
            )
    else:
        event_prob = None

        
    if trigger_mode == "logbf":
        per_point_thr = float(logbf_threshold)
        score_vec = np.asarray(log_bf_local, float)
        raw_idx = np.nonzero(np.isfinite(score_vec) & (score_vec >= per_point_thr))[0]
        trigger_threshold_used = per_point_thr
        trigger_value_max = max_log_bf_local

    elif trigger_mode == "posterior_prob":
        if event_prob is None:
            raise RuntimeError("trigger_mode='posterior_prob' requires compute_event_prob=True")

        thr_prob = significance_threshold / 100.0 if significance_threshold > 1.0 else float(significance_threshold)
        score_vec = np.asarray(event_prob, float)
        raw_idx = np.nonzero(np.isfinite(score_vec) & (score_vec >= thr_prob))[0]
        trigger_threshold_used = thr_prob
        trigger_value_max = float(np.nanmax(score_vec)) if score_vec.size else np.nan

    else:
        raise ValueError("trigger_mode must be 'logbf' or 'posterior_prob'")

    kept_runs = []
    run_summaries = []

    if raw_idx.size == 0:
        event_indices = np.array([], dtype=int)
        significant = False
        run_stats = summarize_kept_runs([], jd, score_vec, cam_vec=cam_vec)
    else:
        runs = build_runs(
            raw_idx,
            jd,
            allow_gap_points=int(run_allow_gap_points),
            max_gap_days=run_max_gap_days,
        )

        kept_runs, initial_summaries = filter_runs(
            runs,
            jd,
            score_vec,
            min_points=int(run_min_points),
            min_duration_days=run_min_duration_days,
            per_point_threshold=trigger_threshold_used,
            cam_vec=cam_vec,
        )

        final_summaries = []
        for i, r in enumerate(kept_runs):
            summary = initial_summaries[i]
            morph_res = classify_run_morphology(jd, mags, errs, r, kind=kind)
            summary.update(morph_res)
            
            # Compute symmetry score for dips (Tzanidakis+2025 Eq. 5)
            if kind == "dip" and len(r) >= 3:
                # Convert magnitudes to flux (f = 10^((baseline - m) / 2.5))
                baseline_mag_run = morph_res.get("params", {}).get("baseline", baseline_mag)
                flux = np.power(10.0, (baseline_mag_run - mags) / 2.5)
                
                # Find dip center (minimum flux = maximum magnitude)
                center_local = np.argmax(mags[r])
                center_idx = r[center_local]
                start_idx = r[0]
                end_idx = r[-1]
                
                sym_score = compute_symmetry_score(
                    jd, flux, center_idx, start_idx, end_idx, baseline_flux=1.0
                )
                summary["symmetry_score"] = sym_score
            else:
                summary["symmetry_score"] = np.nan
            
            final_summaries.append(summary)
        
        run_summaries = final_summaries

        if kept_runs:
            event_indices = np.unique(np.concatenate(kept_runs)).astype(int)
            significant = True
        else:
            event_indices = np.array([], dtype=int)
            significant = False

        run_stats = summarize_kept_runs(kept_runs, jd, score_vec, cam_vec=cam_vec)

    return dict(
        kind=str(kind),
        baseline_mag=float(baseline_mag),
        best_mag_event=float(best_mag_event),
        best_p=float(best_p),

        log_bf_local=log_bf_local,
        max_log_bf_local=float(max_log_bf_local) if np.isfinite(max_log_bf_local) else np.nan,
        event_probability=event_prob,
        used_sigma_eff=bool(used_sigma_eff),

        trigger_mode=str(trigger_mode),
        trigger_threshold=float(trigger_threshold_used),
        trigger_max=float(trigger_value_max) if np.isfinite(trigger_value_max) else np.nan,
        event_indices=event_indices,
        significant=bool(significant),

        run_summaries=run_summaries,
        **run_stats,

        bayes_factor=float(bayes_factor),
        log_evidence_mixture=float(log_evidence_mixture),
        log_evidence_baseline=float(loglik_baseline_only),
        baseline_source=",".join(sorted({str(x) for x in baseline_sources if isinstance(x, (str, bytes)) and len(str(x)) > 0})) or "unknown",

        p_grid=p_grid,
        mag_grid=mag_grid,

        df_base=df_base,
    )


def run_bayesian_significance(
    df: pd.DataFrame,
    *,
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,

    p_points: int = 12,
    p_min_dip: float | None = None,
    p_max_dip: float | None = None,
    p_min_jump: float | None = None,
    p_max_jump: float | None = None,
    mag_grid_dip: np.ndarray | None = None,
    mag_grid_jump: np.ndarray | None = None,
    mag_points: int = 12,

    trigger_mode: str = "posterior_prob", # posterior probability or logbf
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 2,
    run_allow_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,

    use_sigma_eff: bool = True,
    require_sigma_eff: bool = True,

    compute_event_prob: bool = True,
):
    """
    compute baseline one then reuse it for dip and jump scoring
    """
    df = clean_lc(df)

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    df_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None

    dip = bayesian_event_significance(
        df,
        kind="dip",
        baseline_func=None,
        baseline_kwargs=baseline_kwargs,
        df_base=df_base,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        p_min=p_min_dip,
        p_max=p_max_dip,
        p_points=p_points,
        mag_grid=mag_grid_dip,
        mag_points=mag_points,
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_dip,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        compute_event_prob=compute_event_prob,
    )

    jump = bayesian_event_significance(
        df,
        kind="jump",
        baseline_func=None,
        baseline_kwargs=baseline_kwargs,
        df_base=df_base,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        p_min=p_min_jump,
        p_max=p_max_jump,
        p_points=p_points,
        mag_grid=mag_grid_jump,
        mag_points=mag_points,
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        compute_event_prob=compute_event_prob,
    )

    return dict(dip=dip, jump=jump, df_base=df_base)



def process_one(
    path: str,
    *,
    trigger_mode: str,
    logbf_threshold_dip: float,
    logbf_threshold_jump: float,
    significance_threshold: float,
    p_points: int,
    p_min_dip: float | None,
    p_max_dip: float | None,
    p_min_jump: float | None,
    p_max_jump: float | None,
    mag_points: int,
    mag_min_dip: float | None = None,
    mag_max_dip: float | None = None,
    mag_min_jump: float | None = None,
    mag_max_jump: float | None = None,

    run_min_points: int,
    run_allow_gap_points: int,
    run_max_gap_days: float | None,
    run_min_duration_days: float | None,

    baseline_tag: str,
    baseline_kwargs: dict | None = None,
    use_sigma_eff: bool,
    require_sigma_eff: bool,

    compute_event_prob: bool,
    excluded_cameras: str | None = None,
    auto_filter_bad_cameras: bool = False,
    bad_camera_scatter_ratio: float = 2.5,
):
    """
    
    """
    import os
    try:
        # script mode: sys.path[0] points to the malca/ directory with plot/plot.py
        from malca.plot import read_skypatrol_csv  # type: ignore
    except ImportError:
        try:
            # package-style
            from malca.plot import read_skypatrol_csv  # type: ignore
        except ImportError:
            # fallback if a flat plot.py is on the path
            from plot import read_skypatrol_csv  # type: ignore
    path = str(path)
    
    if os.path.isfile(path) and (path.endswith('.csv') or path.endswith('-light-curves.csv')):
        try:
            result = read_skypatrol_csv(path)
            if isinstance(result, tuple):
                if len(result) == 2:
                    df = pd.concat([result[0], result[1]], ignore_index=True) if not (result[0].empty and result[1].empty) else pd.DataFrame()
                else:
                    raise ValueError(f"read_skypatrol_csv returned unexpected tuple length: {len(result)}")
            else:
                df = result
        except ValueError as e:
            raise ValueError(f"Error reading {path}: {e}")
    elif os.path.isfile(path) and path.endswith('.csv'):
        dir_path = os.path.dirname(path) or '.'
        basename = os.path.basename(path)
        asassn_id = basename.replace('.csv', '')
        try:
            dfg, dfv = read_lc_csv(asassn_id, dir_path)
            df = pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error reading .csv file {path}: {e}")
    elif os.path.isfile(path) and path.endswith('.dat2'):
        dir_path = os.path.dirname(path) or '.'
        basename = os.path.basename(path)
        asassn_id = basename.replace('.dat2', '')
        try:
            dfg, dfv = read_lc_dat2(asassn_id, dir_path, excluded_cameras=excluded_cameras)
            df = pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error reading .dat2 file {path}: {e}")
    else:
        import glob
        csv_files = sorted(glob.glob(os.path.join(path, '*-light-curves.csv')))
        if csv_files:
            df = read_skypatrol_csv(csv_files[0])
        else:
            raise ValueError(f"Cannot read light curve from path (not a CSV file and no CSVs found in directory): {path}")

    valid_mask = (
        np.isfinite(df["JD"]) &
        np.isfinite(df["mag"]) &
        np.isfinite(df["error"]) &
        (df["error"] > 0) &
        (df["error"] < 10)
    )

    df = df[valid_mask].copy()

    n_points = len(df)
    if n_points < 10:
        raise ValueError(f"Insufficient valid data points ({n_points} < 10) in {path}")
    
    # Auto-filter bad cameras if enabled
    bad_cameras_filtered = set()
    if auto_filter_bad_cameras and "camera#" in df.columns:
        df, bad_cameras_filtered = filter_bad_cameras(
            df,
            lc_path=path,
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )
        n_points = len(df)
        if n_points < 10:
            raise ValueError(f"Insufficient valid data points after bad camera filtering ({n_points} < 10) in {path}")
    
    baseline_func_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "trend": per_camera_trend_baseline,
    }
    baseline_func = baseline_func_map.get(baseline_tag, per_camera_gp_baseline)

    # Build mag grids from min/max/points if bounds are provided
    mag_grid_dip = None
    mag_grid_jump = None
    if mag_min_dip is not None and mag_max_dip is not None:
        mag_grid_dip = np.linspace(mag_min_dip, mag_max_dip, mag_points)
    if mag_min_jump is not None and mag_max_jump is not None:
        mag_grid_jump = np.linspace(mag_min_jump, mag_max_jump, mag_points)

    res = run_bayesian_significance(
        df,
        trigger_mode=trigger_mode,
        logbf_threshold_dip=logbf_threshold_dip,
        logbf_threshold_jump=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        p_points=p_points,
        p_min_dip=p_min_dip,
        p_max_dip=p_max_dip,
        p_min_jump=p_min_jump,
        p_max_jump=p_max_jump,
        mag_points=mag_points,
        mag_grid_dip=mag_grid_dip,
        mag_grid_jump=mag_grid_jump,

        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,

        compute_event_prob=compute_event_prob,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )

    dip = res["dip"]
    jump = res["jump"]

    jd_arr = np.asarray(df["JD"], float)
    jd_first = float(np.nanmin(jd_arr)) if jd_arr.size else np.nan
    jd_last = float(np.nanmax(jd_arr)) if jd_arr.size else np.nan
    cadence_median_days = float(robust_median_dt_days(jd_arr))

    def max_event_prob(ev):
        ep = ev.get("event_probability")
        if ep is None or (isinstance(ep, float) and not np.isfinite(ep)):
            return np.nan
        ep = np.asarray(ep, float)
        return float(np.nanmax(ep)) if ep.size else np.nan

    def get_best_morph_info(run_list):
        """
        Extract morphology info and symmetry score from the best run.
        """
        if not run_list:
            return "none", 0.0, 0.0, np.nan
        best = sorted(run_list, key=lambda x: x['run_max'], reverse=True)[0]
        
        morph = best.get('morphology', 'none')
        delta_bic = best.get('delta_bic_null', 0.0)
        symmetry = best.get('symmetry_score', np.nan)
        
        params = best.get('params', {})
        if morph == 'gaussian':
            main_param = params.get('sigma', np.nan)
        elif morph == 'paczynski':
            main_param = params.get('tE', np.nan)
        else:
            main_param = np.nan
            
        return morph, float(delta_bic), float(main_param), float(symmetry)

    dip_morph, dip_dbic, dip_param, dip_symmetry = get_best_morph_info(dip["run_summaries"])
    jump_morph, jump_dbic, jump_param, _ = get_best_morph_info(jump["run_summaries"])

    cams = df["camera#"].dropna() if "camera#" in df.columns else pd.Series([], dtype=str)

    unique_cams = np.unique(cams.astype(str)) if len(cams) > 0 else np.array([], dtype=str)
    n_cameras = int(unique_cams.size)
    cam_counts = cams.value_counts() if len(cams) > 0 else pd.Series([], dtype=int)
    camera_min_points = int(cam_counts.min()) if len(cam_counts) else 0
    camera_max_points = int(cam_counts.max()) if len(cam_counts) else 0
    camera_ids = ",".join(unique_cams) if len(unique_cams) > 0 else ""

    dipper_score = 0.0
    dipper_n_dips = 0
    dipper_n_valid_dips = 0
    if bool(dip["significant"]):
        # Use GP baseline for scoring (compute residuals)
        df_base = res.get("df_base")
        if df_base is not None and "baseline" in df_base.columns:
            baseline_mags = df_base["baseline"].to_numpy()
        else:
            baseline_mags = None
        score, events = compute_event_score(df, event_type='dip', baseline_mags=baseline_mags)
        dipper_score = float(score)
        dipper_n_dips = int(len(events))
        dipper_n_valid_dips = int(sum(1 for e in events if e.valid))

    return dict(
        path=str(path),

        dip_significant=bool(dip["significant"]),
        jump_significant=bool(jump["significant"]),

        n_points=int(n_points),
        jd_first=jd_first,
        jd_last=jd_last,
        cadence_median_days=cadence_median_days,

        dip_best_morph=str(dip_morph),
        dip_best_delta_bic=float(dip_dbic),
        dip_best_width_param=float(dip_param),
        dip_symmetry_score=float(dip_symmetry),
        
        jump_best_morph=str(jump_morph),
        jump_best_delta_bic=float(jump_dbic),
        jump_best_width_param=float(jump_param),

        dip_count=int(len(dip["event_indices"])),
        jump_count=int(len(jump["event_indices"])),

        dip_run_count=int(dip.get("n_runs", 0)),
        jump_run_count=int(jump.get("n_runs", 0)),

        dip_max_run_points=int(dip.get("max_run_points", 0)),
        jump_max_run_points=int(jump.get("max_run_points", 0)),
        dip_max_run_duration=float(dip.get("max_run_duration", np.nan)),
        jump_max_run_duration=float(jump.get("max_run_duration", np.nan)),
        dip_max_run_sum=float(dip.get("max_run_sum", np.nan)),
        jump_max_run_sum=float(jump.get("max_run_sum", np.nan)),
        dip_max_run_max=float(dip.get("max_run_max", np.nan)),
        jump_max_run_max=float(jump.get("max_run_max", np.nan)),
        dip_max_run_cameras=int(dip.get("max_run_cameras", 0)),
        jump_max_run_cameras=int(jump.get("max_run_cameras", 0)),

        dip_max_log_bf_local=float(dip.get("max_log_bf_local", np.nan)),
        jump_max_log_bf_local=float(jump.get("max_log_bf_local", np.nan)),

        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),

        baseline_mag=float(dip.get("baseline_mag", jump.get("baseline_mag", np.nan))),
        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),
        dip_best_mag_event=float(dip.get("best_mag_event", np.nan)),
        jump_best_mag_event=float(jump.get("best_mag_event", np.nan)),
        dip_trigger_max=float(dip.get("trigger_max", np.nan)),
        jump_trigger_max=float(jump.get("trigger_max", np.nan)),
        dip_max_event_prob=max_event_prob(dip),
        jump_max_event_prob=max_event_prob(jump),

        n_cameras=int(n_cameras),
        camera_ids=str(camera_ids),
        camera_min_points=int(camera_min_points),
        camera_max_points=int(camera_max_points),

        dipper_score=float(dipper_score),
        dipper_n_dips=int(dipper_n_dips),
        dipper_n_valid_dips=int(dipper_n_valid_dips),

        used_sigma_eff=bool(dip.get("used_sigma_eff", False) and jump.get("used_sigma_eff", False)),
        baseline_source=str(dip.get("baseline_source", jump.get("baseline_source", "unknown"))),
        trigger_mode=str(trigger_mode),
        dip_trigger_threshold=float(dip.get("trigger_threshold", np.nan)),
        jump_trigger_threshold=float(jump.get("trigger_threshold", np.nan)),
        bad_cameras_filtered=",".join(str(c) for c in sorted(bad_cameras_filtered)) if bad_cameras_filtered else "",
    )


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian event scoring on light curves in parallel.")
    parser.add_argument("--input", dest="input_patterns", nargs="*", default=None, help="Paths or globs to light-curve files (repeatable).")
    parser.add_argument("inputs", nargs="*", help="Legacy positional light-curve paths or globs (optional if using --input/--mag-bin).")
    parser.add_argument("--mag-bin", dest="mag_bins", action="append", choices=MAG_BINS, help="Process all light curves in this magnitude bin (choices: 12_12.5, 12.5_13, 13_13.5, 13.5_14, 14_14.5, 14.5_15).")
    parser.add_argument("--lc-path", type=str, default="/data/poohbah/1/assassin/rowan.90/lcsv2", help="Base path to light curve directories")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes")
    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"], help="Triggering mode: logbf = per-point log Bayes factor threshold; posterior_prob = posterior probability threshold (requires event probs).")
    parser.add_argument("--logbf-threshold-dip", type=float, default=5.0, help="Per-point dip trigger")
    parser.add_argument("--logbf-threshold-jump", type=float, default=5.0, help="Per-point jump trigger")
    parser.add_argument("--significance-threshold", type=float, default=99.99997, help="Only used if --trigger-mode posterior_prob")
    parser.add_argument("--p-points", type=int, default=12, help="Number of points in the logit-spaced p grid")
    parser.add_argument("--mag-points", type=int, default=12, help="Number of points in the magnitude grid")
    parser.add_argument("--run-min-points", type=int, default=2, help="Min triggered points in a run")
    parser.add_argument("--run-allow-gap-points", type=int, default=1, help="Allow up to this many missing indices inside a run")
    parser.add_argument("--run-max-gap-days", type=float, default=None, help="Break runs if JD gap exceeds this")
    parser.add_argument("--run-min-duration-days", type=float, default=0.0, help="Require run duration >= this (default: 0.0 = disabled)")
    parser.add_argument("--no-event-prob", action="store_true", help="Skip LOO event responsibilities")
    parser.add_argument("--p-min-dip", type=float, default=None, help="Minimum dip fraction for p-grid (overrides default)")
    parser.add_argument("--p-max-dip", type=float, default=None, help="Maximum dip fraction for p-grid (overrides default)")
    parser.add_argument("--p-min-jump", type=float, default=None, help="Minimum jump fraction for p-grid (overrides default)")
    parser.add_argument("--p-max-jump", type=float, default=None, help="Maximum jump fraction for p-grid (overrides default)")
    parser.add_argument("--baseline-func", type=str, default="gp", choices=["gp", "gp_masked", "trend"], help="Baseline function to use")
    # Baseline kwargs (GP kernel parameters)
    parser.add_argument("--baseline-s0", type=float, default=0.0005, help="GP kernel S0 parameter (default: 0.0005)")
    parser.add_argument("--baseline-w0", type=float, default=0.0031415926535897933, help="GP kernel w0 parameter (default: pi/1000)")
    parser.add_argument("--baseline-q", type=float, default=0.7, help="GP kernel Q parameter (default: 0.7)")
    parser.add_argument("--baseline-jitter", type=float, default=0.006, help="GP jitter term (default: 0.006)")
    parser.add_argument("--baseline-sigma-floor", type=float, default=None, help="Minimum sigma floor (default: None)")
    # Magnitude grid bounds (override auto-detection)
    parser.add_argument("--mag-min-dip", type=float, default=None, help="Min magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-max-dip", type=float, default=None, help="Max magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-min-jump", type=float, default=None, help="Min magnitude for jump grid (overrides auto)")
    parser.add_argument("--mag-max-jump", type=float, default=None, help="Max magnitude for jump grid (overrides auto)")
    parser.add_argument("--no-sigma-eff", action="store_true", help="Do not replace errors with sigma_eff from baseline")
    parser.add_argument("--allow-missing-sigma-eff", action="store_true", help="Do not error if baseline omits sigma_eff (sets require_sigma_eff=False)")
    # Bad camera filtering
    parser.add_argument("--no-filter-bad-cameras", dest="filter_bad_cameras", action="store_false", help="Disable auto-filtering of cameras with anomalously high scatter (enabled by default)")
    parser.add_argument("--bad-camera-scatter-ratio", type=float, default=2.5, help="Scatter ratio threshold for bad camera filtering (default: 2.5)")
    parser.add_argument("--min-mag-offset", type=float, default=0.1, help="Apply signal amplitude filter: require |event_mag - baseline_mag| > threshold (e.g., 0.05)")
    parser.add_argument("--output", type=str, default=None, help="Output path for results (suffix adjusted per format).")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Optional CSV with 'path' and extra metadata columns to attach to results.")
    parser.add_argument("--output-format", type=str, default="csv", choices=["csv", "parquet", "parquet_chunk"], help="Output format for results.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Write results in chunks of this many rows.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (default: quiet).")
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(filter_bad_cameras=True)

    args = parser.parse_args()
    if args.trigger_mode == "posterior_prob" and args.no_event_prob:
        raise SystemExit("posterior_prob triggering requires event_prob; remove --no-event-prob")

    compute_event_prob = (not args.no_event_prob)
    use_sigma_eff = not args.no_sigma_eff
    require_sigma_eff = use_sigma_eff and (not args.allow_missing_sigma_eff)
    baseline_tag = args.baseline_func

    output_format = args.output_format.lower()
    quiet = not args.verbose

    def default_output_dir() -> Path:
        base_dir = Path("/home/lenhart.106/code/malca/output")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_dir / "runs" / timestamp / "results"

    if not args.output:
        out_dir = default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / "lc_events_results.csv")

    metadata_by_path = None
    if args.metadata_csv:
        meta_df = pd.read_csv(args.metadata_csv)
        if "path" not in meta_df.columns:
            raise SystemExit("metadata-csv must include a 'path' column")
        meta_df["path"] = meta_df["path"].astype(str)
        metadata_by_path = meta_df.set_index("path").to_dict(orient="index")

    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    def ensure_suffix(path: Path | None, fmt: str) -> Path | None:
        if path is None:
            return None
        suffix_map = {"csv": ".csv", "parquet": ".parquet", "parquet_chunk": None}
        ext = suffix_map.get(fmt)
        if ext and path.suffix.lower() != ext:
            return path.with_suffix(ext)
        return path

    def collect_processed_from_output(path: Path | None, fmt: str) -> set[str]:
        if path is None or (not path.exists()):
            return set()
        try:
            if fmt == "csv":
                df_existing = pd.read_csv(path, usecols=["path"])
            elif fmt == "parquet":
                if pq is None:
                    raise ImportError("pyarrow is required for parquet outputs")
                table = pq.read_table(path, columns=["path"])
                df_existing = table.to_pandas()
            elif fmt == "parquet_chunk":
                if pq is None:
                    raise ImportError("pyarrow is required for parquet outputs")
                import pyarrow.dataset as ds
                dataset = ds.dataset(path, format="parquet")
                table = dataset.to_table(columns=["path"])
                df_existing = table.to_pandas()
            else:
                return set()
            if "path" in df_existing.columns:
                return set(df_existing["path"].astype(str))
        except Exception as e:
            log(f"Warning: could not read existing output {path} to skip duplicates: {e}")
        return set()

    def clear_existing_output(path: Path | None, fmt: str) -> None:
        if path is None or (not path.exists()):
            return
        try:
            if fmt == "parquet_chunk" and path.is_dir():
                removed_any = False
                for child in path.glob("chunk_*.parquet*"):
                    child.unlink()
                    removed_any = True
                if removed_any:
                    log(f"Overwriting existing output chunks in {path}")
            else:
                path.unlink()
                log(f"Overwriting existing output file: {path}")
        except Exception as e:
            log(f"Warning: Could not remove existing output {path} ({e}). Will append.")

    # checkpoint
    base_output_path = ensure_suffix(Path(args.output).expanduser() if args.output else None, output_format)
    if args.mag_bins and base_output_path is not None:
        # pick the bin name if only one was given; otherwise use the "multi" tag
        bin_tag = args.mag_bins[0] if len(args.mag_bins) == 1 else "multi"
        base_output_path = base_output_path.with_name(f"{base_output_path.stem}_{bin_tag}{base_output_path.suffix}")

    if base_output_path:
        checkpoint_log = base_output_path.with_name(f"{base_output_path.stem}_PROCESSED.txt")
    else:
        checkpoint_log = None

    processed_files = set()
    if checkpoint_log and checkpoint_log.exists() and args.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            log(f"Overwriting checkpoint log: {checkpoint_log}")
        except Exception as e:
            log(f"Warning: Could not overwrite checkpoint file ({e}). Continuing without resume.")

    if args.overwrite:
        clear_existing_output(base_output_path, output_format)

    if checkpoint_log and checkpoint_log.exists() and not args.overwrite:
        log("--- RESUME DETECTED ---")
        log(f"Reading processed files from: {checkpoint_log}")
        try:
            with open(checkpoint_log, "r") as f:
                processed_files = set(line.strip() for line in f)
            log(f"Found {len(processed_files)} previously processed files.")
        except Exception as e:
            log(f"Warning: Could not read checkpoint file ({e}). Starting fresh.")

    # existing output (avoid duplicates if checkpoint was out-of-sync)
    if not args.overwrite:
        processed_files |= collect_processed_from_output(base_output_path, output_format)

    input_patterns: list[str] = []
    if args.input_patterns:
        input_patterns.extend(args.input_patterns)
    if args.inputs:
        input_patterns.extend(args.inputs)

    expanded_inputs = []
    if args.mag_bins:
        lc_path = args.lc_path
        for mag_bin in args.mag_bins:
            mag_bin_dir = os.path.join(lc_path, mag_bin)
            lc_dirs = sorted(glob.glob(os.path.join(mag_bin_dir, "lc*_cal")))
            for lc_dir in lc_dirs:
                csv_files = sorted(glob.glob(os.path.join(lc_dir, "*.csv")))
                dat2_files = sorted(glob.glob(os.path.join(lc_dir, "*.dat2")))
                if csv_files: expanded_inputs.extend(csv_files)
                elif dat2_files: expanded_inputs.extend(dat2_files)
    
    for pattern in input_patterns:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = glob.glob(pattern)
            if matches:
                expanded_inputs.extend(sorted(matches))
            else:
                log(f"Warning: glob pattern '{pattern}' matched no files")
        else: expanded_inputs.append(pattern)
    
    seen = set()
    expanded_inputs = [x for x in expanded_inputs if not (x in seen or seen.add(x))]
    
    if not expanded_inputs: raise SystemExit("No input files found.")
    
    # --- CHECKPOINT FILTERING ---
    original_count = len(expanded_inputs)
    expanded_inputs = [x for x in expanded_inputs if str(x) not in processed_files]
    log(f"Processing {len(expanded_inputs)} light curve file(s) (Filtered from {original_count})...")
    
    if len(expanded_inputs) == 0:
        log("All files have been processed according to checkpoint! Exiting.")
        return

    results = []
    errors = []
    
    if args.chunk_size is None:
        if len(expanded_inputs) < 10000: chunk_size = 500
        elif len(expanded_inputs) < 100000: chunk_size = 1000
        else: chunk_size = 5000
        log(f"Auto-selected chunk size: {chunk_size}")
    elif args.chunk_size > 0:
        chunk_size = args.chunk_size
    else:
        chunk_size = None

    if output_format == "csv" and chunk_size is not None and args.chunk_size == 10000:
        # default to per-LC append for the line-oriented CSV mode
        chunk_size = 1

    total_written = 0
    total_dip_sig = 0
    total_jump_sig = 0
    total_any_sig = 0

    class CsvWriter:
        def __init__(self, path: Path):
            self.path = Path(path)
            self.columns = None
            if self.path.exists() and self.path.stat().st_size > 0:
                try:
                    self.columns = pd.read_csv(self.path, nrows=0).columns.tolist()
                except Exception:
                    self.columns = None

        def write_chunk(self, chunk_results):
            if not chunk_results:
                return
            df_chunk = pd.DataFrame(chunk_results)
            if self.columns is None:
                self.columns = list(df_chunk.columns)
            df_chunk = df_chunk.reindex(columns=self.columns)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = not self.path.exists() or self.path.stat().st_size == 0
            df_chunk.to_csv(self.path, mode="a", header=header, index=False)

        def close(self):
            return

    class ParquetChunkWriter:
        def __init__(self, path: Path):
            if pa is None or pq is None:
                raise ImportError("pyarrow is required for parquet outputs")
            self.path = Path(path)
            self.append = self.path.exists() and self.path.stat().st_size > 0

        def write_chunk(self, chunk_results):
            if not chunk_results:
                return
            df_chunk = pd.DataFrame(chunk_results)
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, self.path, compression="brotli", append=self.append)
            self.append = True

        def close(self):
            return

    class ParquetDatasetWriter:
        def __init__(self, path: Path):
            if pa is None or pq is None:
                raise ImportError("pyarrow is required for parquet outputs")
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            existing = sorted(self.path.glob("chunk_*.parquet"))
            if existing:
                try:
                    last = existing[-1].stem.split("_")[-1]
                    self.counter = int(last) + 1
                except Exception:
                    self.counter = len(existing)
            else:
                self.counter = 0

        def write_chunk(self, chunk_results):
            if not chunk_results:
                return
            df_chunk = pd.DataFrame(chunk_results)
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            tmp_path = self.path / f"chunk_{self.counter:06d}.parquet.tmp"
            final_path = self.path / f"chunk_{self.counter:06d}.parquet"
            pq.write_table(table, tmp_path, compression="brotli")
            os.replace(tmp_path, final_path)
            self.counter += 1

        def close(self):
            return

    def make_writer(path: Path | None, fmt: str):
        if path is None:
            return None
        if fmt == "csv":
            return CsvWriter(path)
        elif fmt == "parquet":
            return ParquetChunkWriter(path)
        elif fmt == "parquet_chunk":
            return ParquetDatasetWriter(path)
        else:
            raise ValueError(f"Unknown output format: {fmt}")

    output_path = base_output_path
    writer = make_writer(output_path, output_format)
    if output_path:
        args.output = str(output_path)

    def count_significant(rows: list[dict]) -> tuple[int, int, int]:
        dip = 0
        jump = 0
        any_sig = 0
        for row in rows:
            dip_sig = bool(row.get("dip_significant"))
            jump_sig = bool(row.get("jump_significant"))
            if dip_sig:
                dip += 1
            if jump_sig:
                jump += 1
            if dip_sig or jump_sig:
                any_sig += 1
        return dip, jump, any_sig

    def write_chunk(chunk_results, is_final=False):
        if not chunk_results: 
            return
        nonlocal total_written, total_dip_sig, total_jump_sig, total_any_sig, writer
        
        # Tag signal amplitude failures if requested
        if args.min_mag_offset is not None and args.min_mag_offset > 0:
            df_chunk = pd.DataFrame(chunk_results)
            dip_diff = np.abs(df_chunk["dip_best_mag_event"] - df_chunk["baseline_mag"])
            jump_diff = np.abs(df_chunk["jump_best_mag_event"] - df_chunk["baseline_mag"])
            passed = (dip_diff > args.min_mag_offset) | (jump_diff > args.min_mag_offset)
            df_chunk["failed_signal_amplitude"] = ~passed
            n_failed = int((~passed).sum())
            if n_failed > 0:
                log(f"Signal amplitude filter: {n_failed}/{len(df_chunk)} failed")
            chunk_results = df_chunk.to_dict('records')
        
        if writer is not None:
            writer.write_chunk(chunk_results)

        if checkpoint_log:
            try:
                checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_log, "a") as f:
                    for row in chunk_results:
                        f.write(str(row['path']) + "\n")
            except Exception as e:
                log(f"WARNING: Could not update checkpoint log: {e}")

        chunk_dip, chunk_jump, chunk_any = count_significant(chunk_results)
        total_dip_sig += chunk_dip
        total_jump_sig += chunk_jump
        total_any_sig += chunk_any
        total_written += len(chunk_results)
        if is_final:
            if writer is not None:
                writer.close()
            if args.output:
                log(
                    f"Wrote {total_written} total rows to {args.output} "
                    f"(dip_sig={total_dip_sig}, jump_sig={total_jump_sig}, any_sig={total_any_sig})"
                )
        else:
            log(
                f"Wrote chunk: {len(chunk_results)} rows (total: {total_written}) "
                f"(dip_sig={total_dip_sig}, jump_sig={total_jump_sig}, any_sig={total_any_sig})"
            )

    # Build baseline_kwargs from CLI args
    baseline_kwargs = dict(
        S0=args.baseline_s0,
        w0=args.baseline_w0,
        q=args.baseline_q,
        jitter=args.baseline_jitter,
        sigma_floor=args.baseline_sigma_floor,
        add_sigma_eff_col=True,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for path in expanded_inputs:
            # Extract excluded_cameras from metadata if available
            path_excluded = None
            if metadata_by_path:
                meta = metadata_by_path.get(str(path))
                if meta and "excluded_cameras" in meta:
                    path_excluded = meta.get("excluded_cameras")
                    if pd.isna(path_excluded) or path_excluded == "":
                        path_excluded = None

            fut = ex.submit(
                process_one, path, trigger_mode=args.trigger_mode, logbf_threshold_dip=args.logbf_threshold_dip,
                logbf_threshold_jump=args.logbf_threshold_jump, significance_threshold=args.significance_threshold,
                p_points=args.p_points, p_min_dip=args.p_min_dip, p_max_dip=args.p_max_dip,
                p_min_jump=args.p_min_jump, p_max_jump=args.p_max_jump, mag_points=args.mag_points,
                mag_min_dip=args.mag_min_dip, mag_max_dip=args.mag_max_dip,
                mag_min_jump=args.mag_min_jump, mag_max_jump=args.mag_max_jump,
                run_min_points=args.run_min_points, run_allow_gap_points=args.run_allow_gap_points,
                run_max_gap_days=args.run_max_gap_days, run_min_duration_days=args.run_min_duration_days,
                baseline_tag=baseline_tag, baseline_kwargs=baseline_kwargs,
                use_sigma_eff=use_sigma_eff, require_sigma_eff=require_sigma_eff,
                compute_event_prob=compute_event_prob,
                excluded_cameras=path_excluded,
                auto_filter_bad_cameras=args.filter_bad_cameras,
                bad_camera_scatter_ratio=args.bad_camera_scatter_ratio,
            )
            futs[fut] = path


        for fut in tqdm(as_completed(futs), total=len(futs), desc="LCs", unit="lc", disable=quiet):
            path = futs[fut]
            try:
                result = fut.result()
                if metadata_by_path:
                    meta = metadata_by_path.get(str(path))
                    if meta:
                        result.update(meta)
                results.append(result)
                if chunk_size and len(results) >= chunk_size:
                    write_chunk(results)
                    results = []
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                errors.append(dict(path=str(path), error=repr(e), traceback=tb_str))
                print(f"ERROR processing {path}: {e}", flush=True)
                if "too many values to unpack" in str(e): print(f"Full traceback:\n{tb_str}", flush=True)

    if results:
        write_chunk(results, is_final=True)
    elif args.output and total_written == 0:
        pass
    else:
        if not quiet:
            for row in results:
                print(f"{row['path']}\tmode={row['trigger_mode']}\tdip_sig={row['dip_significant']} jump_sig={row['jump_significant']}")

    if errors:
        print(f"Completed with {len(errors)} failures.", flush=True)


if __name__ == "__main__":
    main()
