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
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", message=".*Covariance of the parameters could not be estimated.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in.*", category=RuntimeWarning)

from malca.utils import (
    read_lc_dat2,
    read_lc_csv,
    read_skypatrol_csv,
    clean_lc,
    gaussian,
    paczynski_kernel,
    fred,
    skew_gaussian,
    filter_bad_cameras,
    log as _log,
)
from malca.baseline import (
    global_median_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.score import compute_event_score
from malca.stats import log_gaussian, median_dt, bic
from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION, OUTPUT_FORMAT, EVENTS_OUTPUT_CHUNK_SIZE
from malca.config.config_paths import LCV2_ROOT
from malca.config.config_pipeline import (
    WORKERS, TRIGGER_MODE, P_POINTS, MAG_POINTS,
    LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP, SIGNIFICANCE_THRESHOLD,
    MIN_MAG_OFFSET, RUN_MIN_POINTS, RUN_MAX_GAP_POINTS,
    BASELINE_FUNC, BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
)
from malca.config.config_filters import BAD_CAMERA_SCATTER_RATIO_THRESHOLD

from numba import njit, prange


MAG_BINS = ['12_12.5', '12.5_13', '13_13.5', '13.5_14', '14_14.5', '14.5_15']

EventKind: TypeAlias = Literal["dip", "jump"]

DEFAULT_BASELINE_KWARGS = dict(
    S0=BASELINE_S0,
    w0=BASELINE_W0,
    q=BASELINE_Q,
    jitter=BASELINE_JITTER,
    sigma_floor=None,
    add_sigma_eff_col=True,
)

def sigmoid_spaced_p_grid(p_min=1e-4, p_max=1.0 - 1e-4, n=12):
    """
    Probability grid that is uniform in logit/sigmoid space.

    This corresponds to placing equal spacing in log-odds:
      q = log(p / (1 - p))
    then mapping back through the sigmoid.
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



def default_mag_grid(
    baseline_mag: float,
    mags: np.ndarray,
    kind: EventKind,  # "dip" or "jump"
    n: int = 12,
):
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
        if kind == "dip":
            stop = start + 0.1
        else:
            stop = start - 0.1

    return np.linspace(start, stop, int(n))


def compute_symmetry_score(
    jd: np.ndarray,  # times (JD)
    resid: np.ndarray,  # mag - baseline (positive in dips)
    center_idx: int,  # run center index
    start_idx: int,  # run start index
    end_idx: int,  # run end index
) -> float:
    """Tzanidakis+2025 Eq. 5 symmetry score (ingress vs egress area)."""
    jd = np.asarray(jd, float)
    resid = np.asarray(resid, float)

    if not (0 <= start_idx < center_idx < end_idx < len(jd)):
        return np.nan

    # ingress segment [start..center], egress segment [center..end]
    t_ingress = jd[start_idx:center_idx + 1]
    resid_ingress = resid[start_idx:center_idx + 1]

    t_egress = jd[center_idx:end_idx + 1]
    resid_egress = resid[center_idx:end_idx + 1]

    I_ingress = np.trapz(resid_ingress, t_ingress)
    I_egress = np.trapz(resid_egress, t_egress)

    denominator = np.sqrt(I_ingress**2 + I_egress**2)
    if denominator < 1e-10:
        return 0.0

    return float((I_ingress - I_egress) / denominator)


def classify_run_morphology(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    run_idx: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    kind: EventKind = "dip",
):
    """
    Fits gaussian / skew_gaussian / paczynski / fred / noise to a padded run segment.
    *baseline* – full-length baseline array (use baseline[slice] as baseline_guess).
    """
    pad = 5
    start_i = int(max(0, run_idx[0] - pad))
    end_i = int(min(len(jd), run_idx[-1] + pad + 1))

    t_padded = jd[start_i:end_i]
    mag_padded = mag[start_i:end_i]
    err_padded = err[start_i:end_i]

    # Use sliced GP baseline as guess when available; fall back to nanmedian
    if baseline is not None:
        baseline_guess = float(np.nanmedian(baseline[start_i:end_i]))
    else:
        baseline_guess = float(np.nanmedian(mag_padded))

    abs_diff = np.abs(mag_padded - baseline_guess)
    peak_local_idx = np.argmax(abs_diff)

    t0_guess = t_padded[peak_local_idx]
    amp_guess = mag_padded[peak_local_idx] - baseline_guess
    sigma_guess = max((t_padded[-1] - t_padded[0]) / 4.0, 0.01)

    resid_null = mag_padded - baseline_guess
    bic_null = bic(resid_null, err_padded, 1)

    best_bic = bic_null
    best_model = "noise"
    best_params = {}

    if kind == "dip":
        try:
            popt_g, _ = curve_fit(
                gaussian, t_padded, mag_padded,
                p0=[amp_guess, t0_guess, sigma_guess, baseline_guess],
                sigma=err_padded, maxfev=2000
            )
            resid_g = mag_padded - gaussian(t_padded, *popt_g)
            bic_g = bic(resid_g, err_padded, 4)

            if (popt_g[0] > 0) and bic_g < (best_bic - 10):
                best_bic = bic_g
                best_model = "gaussian"
                best_params = {
                    "amp": popt_g[0], "t0": popt_g[1],
                    "sigma": popt_g[2], "baseline": popt_g[3]
                }
        except Exception:
            pass

    # skew_gaussian for dips (asymmetric profiles)
    if kind == "dip":
        try:
            popt_sg, _ = curve_fit(
                skew_gaussian, t_padded, mag_padded,
                p0=[amp_guess, t0_guess, sigma_guess, baseline_guess, 0.0],
                sigma=err_padded, maxfev=3000,
                bounds=(
                    [-np.inf, t_padded[0], 1e-5, -np.inf, -10],
                    [np.inf, t_padded[-1], np.inf, np.inf, 10]
                )
            )
            resid_sg = mag_padded - skew_gaussian(t_padded, *popt_sg)
            bic_sg = bic(resid_sg, err_padded, 5)

            if (popt_sg[0] > 0) and bic_sg < (best_bic - 10):
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
            popt_p, _ = curve_fit(
                paczynski_kernel, t_padded, mag_padded,
                p0=[-abs(amp_guess), t0_guess, sigma_guess, baseline_guess],
                sigma=err_padded, maxfev=2000
            )
            resid_p = mag_padded - paczynski_kernel(t_padded, *popt_p)
            bic_p = bic(resid_p, err_padded, 4)

            if (popt_p[0] < 0) and bic_p < (best_bic - 10):
                best_bic = bic_p
                best_model = "paczynski"
                best_params = {
                    "amp": popt_p[0], "t0": popt_p[1],
                    "tE": popt_p[2], "baseline": popt_p[3]
                }
        except Exception:
            pass

        try:
            popt_f, _ = curve_fit(
                fred, t_padded, mag_padded,
                p0=[amp_guess, t0_guess, 0.05, baseline_guess],
                sigma=err_padded, maxfev=2000
            )
            if popt_f[0] < 0:
                resid_f = mag_padded - fred(t_padded, *popt_f)
                bic_f = bic(resid_f, err_padded, 4)

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
    max_gap_points: int = 1,
    max_gap_days: float | None = None,
):
    """Build runs from clustered triggered points."""
    jd = np.asarray(jd, float)
    trig_idx = np.asarray(trig_idx, dtype=int)
    trig_idx = trig_idx[(trig_idx >= 0) & (trig_idx < jd.size)]
    if trig_idx.size == 0:
        return []

    trig_idx = np.unique(trig_idx)
    trig_idx.sort()

    if max_gap_days is None:
        # 99.73th percentile (3-sigma) of gaps between sorted data points
        dt = np.diff(np.sort(jd))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            max_gap_days = float(np.nanpercentile(dt, 99.73))
        else:
            max_gap_days = 5.0
    max_gap_days = float(max_gap_days)

    max_index_step = int(max_gap_points) + 1

    runs = []
    current_run = [int(trig_idx[0])]
    for k in range(1, trig_idx.size):
        i_prev = current_run[-1]
        i = int(trig_idx[k])

        idx_step = i - i_prev
        dt = jd[i] - jd[i_prev]

        if (idx_step <= max_index_step) and np.isfinite(dt) and (dt <= max_gap_days):
            current_run.append(i)
        else:
            runs.append(np.asarray(current_run, dtype=int))
            current_run = [i]
    runs.append(np.asarray(current_run, dtype=int))
    return runs


def filter_runs(
    runs,
    jd: np.ndarray,
    point_significance: np.ndarray,
    *,
    min_points: int = 2,
    min_duration_days: float | None = None,
    per_point_threshold: float | None = None,
    cam_vec: np.ndarray | None = None,
):
    """Filter runs by minimum points, duration, and per-point threshold."""
    jd = np.asarray(jd, float)
    point_significance = np.asarray(point_significance, float)

    cad = median_dt(jd)
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
        vals = point_significance[r]
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
    point_significance: np.ndarray,
    cam_vec: np.ndarray | None = None,
):
    jd = np.asarray(jd, float)
    point_significance = np.asarray(point_significance, float)

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

        vals = point_significance[r]
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


def compute_recurrence_stats(run_summaries: list[dict]) -> dict:
    """Compute inter-event recurrence statistics from run summaries.

    Parameters
    ----------
    run_summaries : list[dict]
        Each dict must have ``start_jd``, ``end_jd``, ``duration_days``,
        and ``run_max`` (peak significance amplitude).

    Returns
    -------
    dict
        is_single_event, inter_event_spacing_median, inter_event_spacing_std,
        amplitude_consistency, duration_consistency.
    """
    empty = dict(
        is_single_event=True,
        inter_event_spacing_median=np.nan,
        inter_event_spacing_std=np.nan,
        amplitude_consistency=np.nan,
        duration_consistency=np.nan,
    )
    if not run_summaries or len(run_summaries) < 1:
        return empty

    if len(run_summaries) == 1:
        return empty

    # Sort by start_jd to ensure chronological order
    sorted_runs = sorted(run_summaries, key=lambda s: s.get("start_jd", 0.0))

    # Inter-event spacing: gap between end of one event and start of the next
    spacings = []
    for i in range(1, len(sorted_runs)):
        prev_end = sorted_runs[i - 1].get("end_jd", np.nan)
        cur_start = sorted_runs[i].get("start_jd", np.nan)
        if np.isfinite(prev_end) and np.isfinite(cur_start):
            spacings.append(cur_start - prev_end)

    spacings = np.asarray(spacings, float)
    spacing_median = float(np.nanmedian(spacings)) if spacings.size else np.nan
    spacing_std = float(np.nanstd(spacings, ddof=1)) if spacings.size >= 2 else np.nan

    # Amplitude consistency: coefficient of variation of run_max across runs
    amps = np.asarray([s.get("run_max", np.nan) for s in sorted_runs], float)
    amps = amps[np.isfinite(amps)]
    if amps.size >= 2 and np.mean(amps) != 0:
        amplitude_consistency = float(np.std(amps, ddof=1) / np.mean(amps))
    else:
        amplitude_consistency = np.nan

    # Duration consistency: coefficient of variation of duration_days
    durs = np.asarray([s.get("duration_days", np.nan) for s in sorted_runs], float)
    durs = durs[np.isfinite(durs)]
    if durs.size >= 2 and np.mean(durs) != 0:
        duration_consistency = float(np.std(durs, ddof=1) / np.mean(durs))
    else:
        duration_consistency = np.nan

    return dict(
        is_single_event=False,
        inter_event_spacing_median=spacing_median,
        inter_event_spacing_std=spacing_std,
        amplitude_consistency=amplitude_consistency,
        duration_consistency=duration_consistency,
    )


@njit(fastmath=True, cache=True, parallel=True)
def marginal_loglikelihood_grid(log_Pb, log_Pf, log_p, log_1_minus_p):
    """Marginal log-likelihood over the (mag_grid × p_grid) posterior grid."""
    M, N = log_Pb.shape  # shape: M x N
    P = log_p.shape[0]
    loglik = np.zeros((M, P), dtype=log_Pb.dtype)

    for m in prange(M):
        for p in range(P):
            lp = log_p[p]
            l1mp = log_1_minus_p[p]
            acc = 0.0

            for n in range(N):
                val_b = log_Pb[m, n] + lp
                val_f = log_Pf[m, n] + l1mp

                if val_b > val_f:
                    mix = val_b + np.log1p(np.exp(val_f - val_b))
                else:
                    mix = val_f + np.log1p(np.exp(val_b - val_f))

                acc += mix

            loglik[m, p] = acc

    return loglik

@njit(fastmath=True, cache=True, parallel=True)
def loo_event_probabilities(loglik, log_p, log_1_minus_p, log_Pb, log_Pf, is_faint):
    """Leave-one-out posterior event-probability for every data point."""
    M, P = loglik.shape
    _, N = log_Pb.shape  # shape: M x N

    event_prob = np.zeros(N, dtype=np.float64)

    for n in prange(N):
        max_b = -np.inf
        sum_b = 0.0

        max_f = -np.inf
        sum_f = 0.0

        for m in range(M):
            val_Pb = log_Pb[m, n]
            val_Pf = log_Pf[m, n]

            for p in range(P):
                t1 = log_p[p] + val_Pb
                t2 = log_1_minus_p[p] + val_Pf

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
            event_prob[n] = np.exp(log_faint - log_norm)
        else:
            event_prob[n] = np.exp(log_bright - log_norm)

    return event_prob



def score_events_bayesian(
    df: pd.DataFrame,
    *,
    kind: EventKind = "dip",
    mag_col: str = "mag",
    err_col: str = "error",

    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,
    df_base: pd.DataFrame | None = None,

    p_min: float | None = None,
    p_max: float | None = None,
    p_points: int = 12,
    mag_grid: np.ndarray | None = None,
    mag_points: int = 12,

    trigger_mode: str = "posterior_prob",
    logbf_threshold: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 2,
    max_gap_points: int = 1,
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

        # sigma_eff is mandatory — every baseline must produce it
        if "sigma_eff" not in df_base.columns:
            raise RuntimeError("Baseline did not return 'sigma_eff'. All baselines must produce sigma_eff.")
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
    log_1_minus_p = np.log1p(-p_grid)

    loglik = marginal_loglikelihood_grid(
        np.ascontiguousarray(log_Pb_grid),
        np.ascontiguousarray(log_Pf_grid),
        log_p,
        log_1_minus_p
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
        event_prob = loo_event_probabilities(
                loglik,
                log_p,
                log_1_minus_p,
                log_Pb_grid,
                log_Pf_grid,
                (event_component == "faint")
            )
    else:
        event_prob = None

    if trigger_mode == "logbf":
        per_point_thr = float(logbf_threshold)
        point_significance = np.asarray(log_bf_local, float)
        raw_idx = np.nonzero(np.isfinite(point_significance) & (point_significance >= per_point_thr))[0]
        trigger_threshold_used = per_point_thr
        trigger_value_max = max_log_bf_local

    elif trigger_mode == "posterior_prob":
        if event_prob is None:
            raise RuntimeError("trigger_mode='posterior_prob' requires compute_event_prob=True")

        thr_prob = significance_threshold / 100.0 if significance_threshold > 1.0 else float(significance_threshold)
        point_significance = np.asarray(event_prob, float)
        raw_idx = np.nonzero(np.isfinite(point_significance) & (point_significance >= thr_prob))[0]
        trigger_threshold_used = thr_prob
        trigger_value_max = float(np.nanmax(point_significance)) if point_significance.size else np.nan

    else:
        raise ValueError("trigger_mode must be 'logbf' or 'posterior_prob'")

    kept_runs = []
    run_summaries = []

    # Pull baseline array for morphology classification
    baseline_arr = np.asarray(df_base["baseline"], float) if (df_base is not None and "baseline" in df_base.columns) else None

    if raw_idx.size == 0:
        event_indices = np.array([], dtype=int)
        significant = False
        run_stats = summarize_kept_runs([], jd, point_significance, cam_vec=cam_vec)
    else:
        runs = build_runs(
            raw_idx,
            jd,
            max_gap_points=int(max_gap_points),
            max_gap_days=run_max_gap_days,
        )

        kept_runs, initial_summaries = filter_runs(
            runs,
            jd,
            point_significance,
            min_points=int(run_min_points),
            min_duration_days=run_min_duration_days,
            per_point_threshold=trigger_threshold_used,
            cam_vec=cam_vec,
        )

        final_summaries = []
        for i, r in enumerate(kept_runs):
            summary = initial_summaries[i]
            morph_res = classify_run_morphology(jd, mags, errs, r, baseline=baseline_arr, kind=kind)
            summary.update(morph_res)
            
            # Symmetry score for dips (Tzanidakis+2025 Eq. 5), computed on residuals
            if kind == "dip" and len(r) >= 3:
                resid = mags - baseline_mags
                center_idx = int(r[np.argmax(resid[r])])
                start_idx = int(r[0])
                end_idx = int(r[-1])
                summary["symmetry_score"] = compute_symmetry_score(jd, resid, center_idx, start_idx, end_idx)
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

        run_stats = summarize_kept_runs(kept_runs, jd, point_significance, cam_vec=cam_vec)

    return dict(
        kind=str(kind),
        baseline_mag=float(baseline_mag),
        best_mag_event=float(best_mag_event),
        best_p=float(best_p),

        log_bf_local=log_bf_local,
        max_log_bf_local=float(max_log_bf_local) if np.isfinite(max_log_bf_local) else np.nan,
        event_probability=event_prob,

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


def score_lightcurve(
    df: pd.DataFrame,
    *,
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,

    p_points: int = 12,
    mag_points: int = 12,
    trigger_mode: str = "posterior_prob",
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 2,
    max_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,

    compute_event_prob: bool = True,

    p_min_dip: float | None = None,
    p_max_dip: float | None = None,
    p_min_jump: float | None = None,
    p_max_jump: float | None = None,
    mag_grid_dip: np.ndarray | None = None,
    mag_grid_jump: np.ndarray | None = None,
):
    """Compute baseline once, then score dips and jumps via kind_configs loop."""
    df = clean_lc(df)

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    df_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None

    kind_configs = {
        "dip": dict(
            p_min=p_min_dip, p_max=p_max_dip,
            mag_grid=mag_grid_dip, logbf_threshold=logbf_threshold_dip,
        ),
        "jump": dict(
            p_min=p_min_jump, p_max=p_max_jump,
            mag_grid=mag_grid_jump, logbf_threshold=logbf_threshold_jump,
        ),
    }

    results = {}
    for kind, cfg in kind_configs.items():
        results[kind] = score_events_bayesian(
            df, kind=kind,
            baseline_func=None, baseline_kwargs=baseline_kwargs, df_base=df_base,
            p_min=cfg["p_min"], p_max=cfg["p_max"],
            p_points=p_points, mag_grid=cfg["mag_grid"], mag_points=mag_points,
            trigger_mode=trigger_mode, logbf_threshold=cfg["logbf_threshold"],
            significance_threshold=significance_threshold,
            run_min_points=run_min_points, max_gap_points=max_gap_points,
            run_max_gap_days=run_max_gap_days,
            run_min_duration_days=run_min_duration_days,
            compute_event_prob=compute_event_prob,
        )

    return dict(dip=results["dip"], jump=results["jump"], df_base=df_base)



def process_lightcurve(
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
    max_gap_points: int,
    run_max_gap_days: float | None,
    run_min_duration_days: float | None,

    baseline_tag: str,
    baseline_kwargs: dict | None = None,

    compute_event_prob: bool,
    excluded_cameras: str | None = None,
    auto_filter_bad_cameras: bool = False,
    bad_camera_scatter_ratio: float = 2.5,
):
    path = str(path)

    if os.path.isfile(path) and path.endswith('.csv'):
        df = read_skypatrol_csv(path)
    elif os.path.isfile(path) and path.endswith('.dat2'):
        dir_path = os.path.dirname(path) or '.'
        asassn_id = os.path.basename(path).replace('.dat2', '')
        dfg, dfv = read_lc_dat2(asassn_id, dir_path, excluded_cameras=excluded_cameras)
        df = pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()
    else:
        raise ValueError(f"Cannot read light curve from path: {path}")

    valid_mask = (
        np.isfinite(df["JD"]) &
        np.isfinite(df["mag"]) &
        np.isfinite(df["error"]) &
        (df["error"] > 0) &
        (df["error"] < 10)
    )
    df = df[valid_mask].copy()

    # Auto-filter bad cameras if enabled
    bad_cameras_filtered = set()
    if auto_filter_bad_cameras and "camera#" in df.columns:
        df, bad_cameras_filtered = filter_bad_cameras(
            df,
            lc_path=path,
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )

    n_points = len(df)

    baseline_func_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
    }
    baseline_func = baseline_func_map.get(baseline_tag, per_camera_gp_baseline)

    # Build mag grids from min/max/points if bounds are provided
    mag_grid_dip = None
    mag_grid_jump = None
    if mag_min_dip is not None and mag_max_dip is not None:
        mag_grid_dip = np.linspace(mag_min_dip, mag_max_dip, mag_points)
    if mag_min_jump is not None and mag_max_jump is not None:
        mag_grid_jump = np.linspace(mag_min_jump, mag_max_jump, mag_points)

    res = score_lightcurve(
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
        max_gap_points=max_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,

        compute_event_prob=compute_event_prob,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )

    dip = res["dip"]
    jump = res["jump"]

    jd_arr = np.asarray(df["JD"], float)
    jd_first = float(np.nanmin(jd_arr)) if jd_arr.size else np.nan
    jd_last = float(np.nanmax(jd_arr)) if jd_arr.size else np.nan
    cadence_median_days = float(median_dt(jd_arr))

    def max_event_prob(ev):
        ep = ev.get("event_probability")
        if ep is None or (isinstance(ep, float) and not np.isfinite(ep)):
            return np.nan
        ep = np.asarray(ep, float)
        return float(np.nanmax(ep)) if ep.size else np.nan

    def get_best_morph_info(run_list):
        """Extract morphology info, full params, and symmetry from the best run.

        Returns
        -------
        dict with keys: morph, delta_bic, width_param, symmetry,
                        amp, t0, alpha, tau.
        """
        empty = dict(
            morph="none", delta_bic=0.0, width_param=np.nan, symmetry=np.nan,
            amp=np.nan, t0=np.nan, alpha=np.nan, tau=np.nan,
        )
        if not run_list:
            return empty
        best_run = max(run_list, key=lambda x: x['run_max'])

        morph = best_run.get('morphology', 'none')
        delta_bic = best_run.get('delta_bic_null', 0.0)
        symmetry = best_run.get('symmetry_score', np.nan)

        params = best_run.get('params', {})

        # Main width parameter (backward-compatible)
        if morph == 'gaussian':
            width_param = params.get('sigma', np.nan)
        elif morph == 'skew_gaussian':
            width_param = params.get('sigma', np.nan)
        elif morph == 'paczynski':
            width_param = params.get('tE', np.nan)
        elif morph == 'fred':
            width_param = params.get('tau', np.nan)
        else:
            width_param = np.nan

        amp = params.get('amp', np.nan)
        t0 = params.get('t0', np.nan)
        alpha = params.get('alpha', np.nan)      # skew_gaussian only
        tau = params.get('tau', np.nan)           # fred only

        return dict(
            morph=str(morph),
            delta_bic=float(delta_bic),
            width_param=float(width_param) if np.isfinite(width_param) else np.nan,
            symmetry=float(symmetry),
            amp=float(amp) if np.isfinite(amp) else np.nan,
            t0=float(t0) if np.isfinite(t0) else np.nan,
            alpha=float(alpha) if np.isfinite(alpha) else np.nan,
            tau=float(tau) if np.isfinite(tau) else np.nan,
        )

    dip_mi = get_best_morph_info(dip["run_summaries"])
    jump_mi = get_best_morph_info(jump["run_summaries"])

    dip_recurrence = compute_recurrence_stats(dip["run_summaries"])
    jump_recurrence = compute_recurrence_stats(jump["run_summaries"])

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
        # Use computed baseline for scoring
        df_base = res.get("df_base")
        if df_base is not None and "baseline" in df_base.columns:
            baseline_mags = df_base["baseline"].to_numpy()
        else:
            baseline_mags = None
        score, events = compute_event_score(df, event_type='dip', baseline_mags=baseline_mags)
        dipper_score = float(score)
        dipper_n_dips = int(len(events))
        dipper_n_valid_dips = int(sum(1 for e in events if e.valid))

    jumper_score = 0.0
    jumper_n_jumps = 0
    jumper_n_valid_jumps = 0
    if bool(jump["significant"]):
        df_base = res.get("df_base")
        if df_base is not None and "baseline" in df_base.columns:
            baseline_mags = df_base["baseline"].to_numpy()
        else:
            baseline_mags = None
        score, events = compute_event_score(df, event_type='jump', baseline_mags=baseline_mags)
        jumper_score = float(score)
        jumper_n_jumps = int(len(events))
        jumper_n_valid_jumps = int(sum(1 for e in events if e.valid))

    return dict(
        path=str(path),

        dip_significant=bool(dip["significant"]),
        jump_significant=bool(jump["significant"]),

        n_points=int(n_points),
        jd_first=jd_first,
        jd_last=jd_last,
        cadence_median_days=cadence_median_days,

        dip_best_morph=str(dip_mi["morph"]),
        dip_best_delta_bic=float(dip_mi["delta_bic"]),
        dip_best_width_param=float(dip_mi["width_param"]),
        dip_symmetry_score=float(dip_mi["symmetry"]),
        dip_best_amp=float(dip_mi["amp"]),
        dip_best_t0=float(dip_mi["t0"]),
        dip_best_alpha=float(dip_mi["alpha"]),
        dip_best_tau=float(dip_mi["tau"]),

        jump_best_morph=str(jump_mi["morph"]),
        jump_best_delta_bic=float(jump_mi["delta_bic"]),
        jump_best_width_param=float(jump_mi["width_param"]),
        jump_best_amp=float(jump_mi["amp"]),
        jump_best_t0=float(jump_mi["t0"]),
        jump_best_alpha=float(jump_mi["alpha"]),
        jump_best_tau=float(jump_mi["tau"]),

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

        jumper_score=float(jumper_score),
        jumper_n_jumps=int(jumper_n_jumps),
        jumper_n_valid_jumps=int(jumper_n_valid_jumps),

        baseline_source=str(dip.get("baseline_source", jump.get("baseline_source", "unknown"))),
        trigger_mode=str(trigger_mode),
        dip_trigger_threshold=float(dip.get("trigger_threshold", np.nan)),
        jump_trigger_threshold=float(jump.get("trigger_threshold", np.nan)),
        bad_cameras_filtered=",".join(str(c) for c in sorted(bad_cameras_filtered)) if bad_cameras_filtered else "",

        # Recurrence statistics
        dip_is_single_event=bool(dip_recurrence["is_single_event"]),
        dip_inter_event_spacing_median=float(dip_recurrence["inter_event_spacing_median"]),
        dip_inter_event_spacing_std=float(dip_recurrence["inter_event_spacing_std"]),
        dip_amplitude_consistency=float(dip_recurrence["amplitude_consistency"]),
        dip_duration_consistency=float(dip_recurrence["duration_consistency"]),

        jump_is_single_event=bool(jump_recurrence["is_single_event"]),
        jump_inter_event_spacing_median=float(jump_recurrence["inter_event_spacing_median"]),
        jump_inter_event_spacing_std=float(jump_recurrence["inter_event_spacing_std"]),
        jump_amplitude_consistency=float(jump_recurrence["amplitude_consistency"]),
        jump_duration_consistency=float(jump_recurrence["duration_consistency"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian event scoring on light curves in parallel.")
    parser.add_argument("--input", dest="input_patterns", nargs="*", default=None, help="Paths or globs to light-curve files (repeatable).")
    parser.add_argument("--mag-bin", dest="mag_bins", action="append", choices=MAG_BINS, help="Process all light curves in this magnitude bin (choices: 12_12.5, 12.5_13, 13_13.5, 13.5_14, 14_14.5, 14.5_15).")
    parser.add_argument("--lc-path", type=str, default=str(LCV2_ROOT), help="Base path to light curve directories")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Number of worker processes")
    parser.add_argument("--trigger-mode", type=str, default=TRIGGER_MODE, choices=["logbf", "posterior_prob"], help="Triggering mode: logbf = per-point log Bayes factor threshold; posterior_prob = posterior probability threshold (requires event probs).")
    parser.add_argument("--logbf-threshold-dip", type=float, default=LOGBF_THRESHOLD_DIP, help="Per-point dip trigger")
    parser.add_argument("--logbf-threshold-jump", type=float, default=LOGBF_THRESHOLD_JUMP, help="Per-point jump trigger")
    parser.add_argument("--significance-threshold", type=float, default=SIGNIFICANCE_THRESHOLD, help="Only used if --trigger-mode posterior_prob")
    parser.add_argument("--p-points", type=int, default=P_POINTS, help="Number of points in the p grid")
    parser.add_argument("--mag-points", type=int, default=MAG_POINTS, help="Number of points in the magnitude grid")
    parser.add_argument("--run-min-points", type=int, default=RUN_MIN_POINTS, help="Min triggered points in a run")
    parser.add_argument("--run-max-gap-points", type=int, default=RUN_MAX_GAP_POINTS, help="Allow up to this many missing indices inside a run")
    parser.add_argument("--run-max-gap-days", type=float, default=None, help="Break runs if JD gap exceeds this")
    parser.add_argument("--run-min-duration-days", type=float, default=0.0, help="Require run duration >= this (default: 0.0 = disabled)")
    parser.add_argument("--no-event-prob", action="store_true", help="Skip LOO event responsibilities")
    parser.add_argument("--p-min-dip", type=float, default=None, help="Minimum dip fraction for p-grid (overrides default)")
    parser.add_argument("--p-max-dip", type=float, default=None, help="Maximum dip fraction for p-grid (overrides default)")
    parser.add_argument("--p-min-jump", type=float, default=None, help="Minimum jump fraction for p-grid (overrides default)")
    parser.add_argument("--p-max-jump", type=float, default=None, help="Maximum jump fraction for p-grid (overrides default)")
    parser.add_argument(
        "--baseline-func",
        type=str,
        default=BASELINE_FUNC,
        choices=["gp", "gp_masked", "global_median", "per_camera_median"],
        help="Baseline function to use",
    )
    # Baseline kwargs (GP kernel parameters)
    parser.add_argument("--baseline-s0", type=float, default=BASELINE_S0, help="GP kernel S0 parameter (default: 0.0005)")
    parser.add_argument("--baseline-w0", type=float, default=BASELINE_W0, help="GP kernel w0 parameter (default: pi/1000)")
    parser.add_argument("--baseline-q", type=float, default=BASELINE_Q, help="GP kernel Q parameter (default: 0.7)")
    parser.add_argument("--baseline-jitter", type=float, default=BASELINE_JITTER, help="GP jitter term (default: 0.006)")
    parser.add_argument("--baseline-sigma-floor", type=float, default=None, help="Minimum sigma floor (default: None)")
    # Magnitude grid bounds (override auto-detection)
    parser.add_argument("--mag-min-dip", type=float, default=None, help="Min magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-max-dip", type=float, default=None, help="Max magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-min-jump", type=float, default=None, help="Min magnitude for jump grid (overrides auto)")
    parser.add_argument("--mag-max-jump", type=float, default=None, help="Max magnitude for jump grid (overrides auto)")
    # Bad camera filtering
    parser.add_argument("--no-filter-bad-cameras", dest="filter_bad_cameras", action="store_false", help="Disable auto-filtering of cameras with anomalously high scatter (enabled by default)")
    parser.add_argument("--bad-camera-scatter-ratio", type=float, default=BAD_CAMERA_SCATTER_RATIO_THRESHOLD, help="Scatter ratio threshold for bad camera filtering (default: 2.5)")
    parser.add_argument("--min-mag-offset", type=float, default=MIN_MAG_OFFSET, help="Apply signal amplitude filter: require |event_mag - baseline_mag| > threshold (e.g., 0.05)")
    parser.add_argument("--output", type=str, default=None, help="Output path for results (suffix adjusted per format).")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Optional CSV with 'path' and extra metadata columns to attach to results.")
    parser.add_argument("--output-format", type=str, default=OUTPUT_FORMAT, choices=["csv", "parquet", "parquet_chunk"], help="Output format for results.")
    parser.add_argument("--chunk-size", type=int, default=EVENTS_OUTPUT_CHUNK_SIZE, help="Write results in chunks of this many rows.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (default: quiet).")
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(filter_bad_cameras=True)

    args = parser.parse_args()
    if args.trigger_mode == "posterior_prob" and args.no_event_prob:
        raise SystemExit("posterior_prob triggering requires event_prob; remove --no-event-prob")

    compute_event_prob = (not args.no_event_prob)
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
        args.output = str(out_dir / "lc_events_results.parquet")

    metadata_by_path = None
    if args.metadata_csv:
        meta_df = pd.read_csv(args.metadata_csv)
        if "path" not in meta_df.columns:
            raise SystemExit("metadata-csv must include a 'path' column")
        meta_df["path"] = meta_df["path"].astype(str)
        metadata_by_path = meta_df.set_index("path").to_dict(orient="index")

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
                table = pq.read_table(path, columns=["path"])
                df_existing = table.to_pandas()
            elif fmt == "parquet_chunk":
                import pyarrow.dataset as ds
                dataset = ds.dataset(path, format="parquet")
                table = dataset.to_table(columns=["path"])
                df_existing = table.to_pandas()
            else:
                return set()
            if "path" in df_existing.columns:
                return set(df_existing["path"].astype(str))
        except Exception as e:
            _log(f"Warning: could not read existing output {path} to skip duplicates: {e}", quiet)
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
                    _log(f"Overwriting existing output chunks in {path}", quiet)
            else:
                path.unlink()
                _log(f"Overwriting existing output file: {path}", quiet)
        except Exception as e:
            _log(f"Warning: Could not remove existing output {path} ({e}). Will append.", quiet)

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
            _log(f"Overwriting checkpoint log: {checkpoint_log}", quiet)
        except Exception as e:
            _log(f"Warning: Could not overwrite checkpoint file ({e}). Continuing without resume.", quiet)

    if args.overwrite:
        clear_existing_output(base_output_path, output_format)

    if checkpoint_log and checkpoint_log.exists() and not args.overwrite:
        _log("--- RESUME DETECTED ---", quiet)
        _log(f"Reading processed files from: {checkpoint_log}", quiet)
        try:
            with open(checkpoint_log, "r") as f:
                processed_files = set(line.strip() for line in f)
            _log(f"Found {len(processed_files)} previously processed files.", quiet)
        except Exception as e:
            _log(f"Warning: Could not read checkpoint file ({e}). Starting fresh.", quiet)

    # existing output (avoid duplicates if checkpoint was out-of-sync)
    if not args.overwrite:
        processed_files |= collect_processed_from_output(base_output_path, output_format)

    input_patterns: list[str] = []
    if args.input_patterns:
        input_patterns.extend(args.input_patterns)

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
                _log(f"Warning: glob pattern '{pattern}' matched no files", quiet)
        else: expanded_inputs.append(pattern)
    
    seen = set()
    expanded_inputs = [x for x in expanded_inputs if not (x in seen or seen.add(x))]
    
    if not expanded_inputs: raise SystemExit("No input files found.")
    
    # --- CHECKPOINT FILTERING ---
    original_count = len(expanded_inputs)
    expanded_inputs = [x for x in expanded_inputs if str(x) not in processed_files]
    _log(f"Processing {len(expanded_inputs)} light curve file(s) (Filtered from {original_count})...", quiet)
    
    if len(expanded_inputs) == 0:
        _log("All files have been processed according to checkpoint! Exiting.", quiet)
        return

    results = []
    errors = []
    
    if args.chunk_size is None:
        if len(expanded_inputs) < 10000: chunk_size = 500
        elif len(expanded_inputs) < 100000: chunk_size = 1000
        else: chunk_size = 5000
        _log(f"Auto-selected chunk size: {chunk_size}", quiet)
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
            self.path = Path(path)
            self.append = self.path.exists() and self.path.stat().st_size > 0
            self.schema_invalidated = False

        def write_chunk(self, chunk_results):
            if not chunk_results:
                return
            df_chunk = pd.DataFrame(chunk_results)
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.append:
                existing = pq.read_table(self.path)
                if existing.schema.equals(table.schema):
                    table = pa.concat_tables([existing, table])
                else:
                    _log(f"Schema mismatch in {self.path}; discarding old checkpoint data.", False)
                    self.schema_invalidated = True
            pq.write_table(table, self.path, compression=PARQUET_OUTPUT_COMPRESSION)
            self.append = True

        def close(self):
            return

    class ParquetDatasetWriter:
        def __init__(self, path: Path):
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
            pq.write_table(table, tmp_path, compression=PARQUET_OUTPUT_COMPRESSION)
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
                _log(f"Signal amplitude filter: {n_failed}/{len(df_chunk)} failed", quiet)
            chunk_results = df_chunk.to_dict('records')
        
        if writer is not None:
            writer.write_chunk(chunk_results)

        if checkpoint_log:
            checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(writer, 'schema_invalidated') and writer.schema_invalidated:
                mode = "w"
                writer.schema_invalidated = False
            else:
                mode = "a"
            with open(checkpoint_log, mode) as f:
                for row in chunk_results:
                    f.write(str(row['path']) + "\n")

        chunk_dip, chunk_jump, chunk_any = count_significant(chunk_results)
        total_dip_sig += chunk_dip
        total_jump_sig += chunk_jump
        total_any_sig += chunk_any
        total_written += len(chunk_results)
        if is_final:
            if writer is not None:
                writer.close()
            if args.output:
                _log(
                    f"Wrote {total_written} total rows to {args.output} "
                    f"(dip_sig={total_dip_sig}, jump_sig={total_jump_sig}, any_sig={total_any_sig})"
                , quiet)
        else:
            _log(
                f"Wrote chunk: {len(chunk_results)} rows (total: {total_written}) "
                f"(dip_sig={total_dip_sig}, jump_sig={total_jump_sig}, any_sig={total_any_sig})"
            , quiet)

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
                process_lightcurve, path, trigger_mode=args.trigger_mode, logbf_threshold_dip=args.logbf_threshold_dip,
                logbf_threshold_jump=args.logbf_threshold_jump, significance_threshold=args.significance_threshold,
                p_points=args.p_points, p_min_dip=args.p_min_dip, p_max_dip=args.p_max_dip,
                p_min_jump=args.p_min_jump, p_max_jump=args.p_max_jump, mag_points=args.mag_points,
                mag_min_dip=args.mag_min_dip, mag_max_dip=args.mag_max_dip,
                mag_min_jump=args.mag_min_jump, mag_max_jump=args.mag_max_jump,
                run_min_points=args.run_min_points, max_gap_points=args.run_max_gap_points,
                run_max_gap_days=args.run_max_gap_days, run_min_duration_days=args.run_min_duration_days,
                baseline_tag=baseline_tag, baseline_kwargs=baseline_kwargs,
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
