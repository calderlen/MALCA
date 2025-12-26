# Thread caps (set before numpy/scipy import when possible)
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from tqdm import tqdm

try:
    from lc_utils import read_lc_dat2
    from lc_baseline import per_camera_gp_baseline
except ImportError:
    from calder.lc_utils import read_lc_dat2
    from calder.lc_baseline import per_camera_gp_baseline


# ----------------------------
# Defaults / constants
# ----------------------------

DEFAULT_BASELINE_KWARGS = dict(
    S0=0.0005,
    w0=0.0031415926535897933,
    q=0.7,
    jitter=0.006,
    sigma_floor=None,        # None => estimate per camera
    add_sigma_eff_col=True,  # ensure sigma_eff exists
)


# ----------------------------
# Small utilities
# ----------------------------

def clean_lc(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)

    if "saturated" in df.columns:
        mask &= (df["saturated"] == 0)

    mask &= df["JD"].notna() & df["mag"].notna()
    if "error" in df.columns:
        mask &= df["error"].notna() & (df["error"] > 0.0) & (df["error"] < 1.0)

    df = df.loc[mask]
    df = df.sort_values("JD").reset_index(drop=True)
    return df


def log_gaussian(x, mu, sigma):
    """
    Physics convention (Gaussian log-likelihood):
        ln p = -1/2 * ((x-mu)/sigma)^2 - ln(sigma) - 1/2 ln(2pi)
    """
    x = np.asarray(x, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    sigma = np.clip(sigma, 1e-12, np.inf)
    z = (x - mu) / sigma
    return -0.5 * z**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def logit_spaced_grid(p_min=1e-4, p_max=1.0 - 1e-4, n=80):
    p_min = float(np.clip(p_min, 1e-12, 1 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1 - 1e-12))
    q_min = np.log(p_min / (1.0 - p_min))
    q_max = np.log(p_max / (1.0 - p_max))
    q = np.linspace(q_min, q_max, int(n))
    return 1.0 / (1.0 + np.exp(-q))


def default_mag_grid(baseline_mag: float, mags: np.ndarray, kind: str, n=60):
    lo, hi = np.nanpercentile(mags, [5, 95])
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


def robust_median_dt_days(jd: np.ndarray) -> float:
    jd = np.asarray(jd, float)
    if jd.size < 2:
        return np.nan
    dt = np.diff(jd)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return float(np.nanmedian(dt))


# ----------------------------
# Morphology Utilities (Hybrid)
# ----------------------------

def model_gaussian(t, amp, t0, sigma, baseline):
    return baseline + amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

def model_paczynski(t, amp, t0, tE, baseline):
    # tE must be positive
    tE = np.maximum(np.abs(tE), 1e-5) 
    # Paczynski strictly brightens relative to baseline
    # Note: amp logic handled in caller (jump vs dip)
    return baseline + amp / np.sqrt(1.0 + ((t - t0) / tE) ** 2)

def calc_bic(resid, err, n_params):
    """
    Bayesian Information Criterion: chi2 + k * ln(n)
    Lower is better.
    """
    # Clip error to avoid division by zero
    err = np.clip(err, 1e-9, np.inf)
    # Robust chi2 to ignore extreme outliers during shape fitting
    chi2 = np.nansum((resid / err) ** 2)
    n_points = len(resid)
    if n_points == 0:
        return np.inf
    return chi2 + n_params * np.log(n_points)

def classify_run_morphology(jd, mag, err, run_idx, kind="dip"):
    """
    Fits Gaussian vs Paczynski vs Flat (Noise) to a specific run.
    Returns dictionary with best model name and parameters.
    """
    # 1. Extract and Pad Data (add context around the run)
    pad = 5 # points on either side
    start_i = int(max(0, run_idx[0] - pad))
    end_i = int(min(len(jd), run_idx[-1] + pad + 1))
    
    t_seg = jd[start_i:end_i]
    y_seg = mag[start_i:end_i]
    e_seg = err[start_i:end_i]
    
    if len(t_seg) < 4:
        return {
            "morphology": "none", "bic": np.nan, "delta_bic_null": 0.0, "params": {}
        }

    # 2. Initial Guesses
    baseline_guess = np.nanmedian(y_seg)
    # Find peak relative to baseline
    abs_diff = np.abs(y_seg - baseline_guess)
    peak_local_idx = np.argmax(abs_diff)
    
    t0_guess = t_seg[peak_local_idx]
    amp_guess = y_seg[peak_local_idx] - baseline_guess
    sigma_guess = max((t_seg[-1] - t_seg[0]) / 4.0, 0.01)
    
    # 3. Fit Baseline (Null Hypothesis)
    # k=1 (baseline level)
    resid_null = y_seg - baseline_guess
    bic_null = calc_bic(resid_null, e_seg, 1)
    
    best_bic = bic_null
    best_model = "noise"
    best_params = {}

    # 4. Fit Gaussian (The "Dip" or "Flare" hypothesis)
    try:
        popt_g, _ = curve_fit(
            model_gaussian, t_seg, y_seg, 
            p0=[amp_guess, t0_guess, sigma_guess, baseline_guess],
            sigma=e_seg, maxfev=2000
        )
        resid_g = y_seg - model_gaussian(t_seg, *popt_g)
        bic_g = calc_bic(resid_g, e_seg, 4) # k=4 params
        
        # Enforce polarity
        # If kind="dip", we expect positive amplitude in magnitude space (fainter)
        # If kind="jump", we expect negative amplitude in magnitude space (brighter)
        is_valid = (popt_g[0] > 0) if kind == "dip" else (popt_g[0] < 0)
        
        # Require strong evidence (delta BIC > 10) to prefer over noise
        if is_valid and bic_g < (best_bic - 10):
            best_bic = bic_g
            best_model = "gaussian"
            best_params = {
                "amp": popt_g[0], "t0": popt_g[1], 
                "sigma": popt_g[2], "baseline": popt_g[3]
            }
    except Exception:
        pass

    # 5. Fit Paczynski (The "Microlensing" hypothesis)
    # Usually only relevant for jumps (brightening), but code supports checking both
    if kind == "jump":
        try:
            # Paczynski strictly brightens (negative amp in mag)
            # Ensure initial guess is negative
            amp_p_guess = -abs(amp_guess) 
            
            popt_p, _ = curve_fit(
                model_paczynski, t_seg, y_seg,
                p0=[amp_p_guess, t0_guess, sigma_guess, baseline_guess],
                sigma=e_seg, maxfev=2000
            )
            resid_p = y_seg - model_paczynski(t_seg, *popt_p)
            bic_p = calc_bic(resid_p, e_seg, 4)

            # Valid if amp is negative (brightening)
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

    return {
        "morphology": best_model,
        "bic": float(best_bic),
        "delta_bic_null": float(bic_null - best_bic), # Positive means better than noise
        "params": best_params
    }


# ----------------------------
# Clustering Utilities
# ----------------------------

def cluster_triggered_indices(
    trig_idx: np.ndarray,
    jd: np.ndarray,
    *,
    allow_gap_points: int = 1,
    max_gap_days: float | None = None,
):
    """
    Build runs from triggered indices in time order.
    """
    jd = np.asarray(jd, float)
    trig_idx = np.asarray(trig_idx, dtype=int)
    trig_idx = trig_idx[(trig_idx >= 0) & (trig_idx < jd.size)]
    if trig_idx.size == 0:
        return []

    trig_idx = np.unique(trig_idx)
    trig_idx.sort()

    cad = robust_median_dt_days(jd)
    if max_gap_days is None:
        if np.isfinite(cad):
            max_gap_days = max(5.0 * cad, 5.0)  # avoid merging seasonal gaps
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


def gate_runs(
    runs,
    jd: np.ndarray,
    score_vec: np.ndarray,
    *,
    min_points: int = 3,
    min_duration_days: float | None = None,
    per_point_threshold: float | None = None,
    sum_threshold: float | None = None,
):
    """
    Gate runs by:
      - min_points
      - min_duration_days (cadence-adaptive if None)
      - optional per_point_threshold (run_max >= threshold)
      - optional sum_threshold (run_sum >= threshold)

    Returns: (kept_runs, run_summaries)
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

        ok = True
        if n < int(min_points):
            ok = False
        if dur < min_duration_days:
            ok = False
        if (per_point_threshold is not None) and (not (np.isfinite(run_max) and run_max >= float(per_point_threshold))):
            ok = False
        if (sum_threshold is not None) and (not (np.isfinite(run_sum) and run_sum >= float(sum_threshold))):
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
                kept=bool(ok),
            )
        )

        if ok:
            kept.append(r)

    return kept, summaries


def summarize_kept_runs(kept_runs, jd: np.ndarray, score_vec: np.ndarray):
    jd = np.asarray(jd, float)
    score_vec = np.asarray(score_vec, float)

    if not kept_runs:
        return dict(
            n_runs=0,
            max_run_points=0,
            max_run_duration=np.nan,
            max_run_sum=np.nan,
            max_run_max=np.nan,
        )

    max_pts = 0
    max_dur = -np.inf
    max_sum = -np.inf
    max_max = -np.inf

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

    return dict(
        n_runs=int(len(kept_runs)),
        max_run_points=int(max_pts),
        max_run_duration=float(max_dur) if np.isfinite(max_dur) else np.nan,
        max_run_sum=float(max_sum) if np.isfinite(max_sum) else np.nan,
        max_run_max=float(max_max) if np.isfinite(max_max) else np.nan,
    )


# ----------------------------
# Core Bayesian scoring
# ----------------------------

def bayesian_excursion_significance(
    df: pd.DataFrame,
    *,
    kind: str = "dip",            # "dip" or "jump"
    mag_col: str = "mag",
    err_col: str = "error",

    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,
    df_base: pd.DataFrame | None = None,  # pass precomputed baseline to avoid refit

    use_sigma_eff: bool = True,
    require_sigma_eff: bool = False,

    p_min: float | None = None,
    p_max: float | None = None,
    p_points: int = 80,
    mag_grid: np.ndarray | None = None,

    # per-point trigger mode
    trigger_mode: str = "logbf",  # "logbf" or "posterior_prob"
    logbf_threshold: float = 5.0,
    significance_threshold: float = 99.99997,  # used if posterior_prob

    # run-based triggers (1,2,3)
    run_min_points: int = 3,
    run_allow_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,
    run_sum_threshold: float | None = None,
    run_sum_multiplier: float = 2.5,

    # diagnostics
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
    df = clean_lc(df)
    jd = np.asarray(df["JD"], float)
    mags = np.asarray(df[mag_col], float)

    # raw errors as fallback
    if err_col in df.columns:
        errs = np.asarray(df[err_col], float)
    else:
        errs = np.full_like(mags, 0.05)

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    used_sigma_eff = False

    # baseline / sigma_eff
    if df_base is None and baseline_func is not None:
        df_base = baseline_func(df, **baseline_kwargs)

    if df_base is None:
        baseline_mags = np.full_like(mags, np.nanmedian(mags))
    else:
        if "baseline" in df_base.columns:
            baseline_mags = np.asarray(df_base["baseline"], float)
        else:
            baseline_mags = np.asarray(df_base[mag_col], float)

        if use_sigma_eff and ("sigma_eff" in df_base.columns):
            errs = np.asarray(df_base["sigma_eff"], float)
            used_sigma_eff = True
        elif require_sigma_eff:
            raise RuntimeError("require_sigma_eff=True but baseline did not return 'sigma_eff'")

    baseline_mag = float(np.nanmedian(baseline_mags))

    # choose p-grid bounds if not specified
    if p_min is None and p_max is None:
        if kind == "dip":
            # p ~ baseline fraction (dominant), excursions rare
            p_min, p_max = 0.5, 1.0 - 1e-4
        elif kind == "jump":
            # bright excursions rare
            p_min, p_max = 1e-4, 0.5
        else:
            raise ValueError("kind must be 'dip' or 'jump'")

    p_grid = logit_spaced_grid(p_min=p_min, p_max=p_max, n=p_points)

    if mag_grid is None:
        mag_grid = default_mag_grid(baseline_mag, mags, kind, n=60)
    else:
        mag_grid = np.asarray(mag_grid, float)

    M = int(len(mag_grid))
    N = int(len(mags))

    # Likelihood pieces
    if kind == "dip":
        # baseline fixed at baseline_mags; excursion is fainter grid
        log_Pb_vec = log_gaussian(mags, baseline_mags, errs)                # (N,)
        log_Pb_grid = np.broadcast_to(log_Pb_vec, (M, N))                 # (M,N)
        log_Pf_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)  # (M,N)
        excursion_component = "faint"
        loglik_baseline_only = float(np.sum(log_Pb_vec))

        # local per-point logBF: ln p(x|event) - ln p(x|baseline), event mag uniform prior
        log_px_baseline = log_Pb_vec
        log_px_event = logsumexp(log_Pf_grid, axis=0) - np.log(M)
        log_bf_local = log_px_event - log_px_baseline

    elif kind == "jump":
        # excursion is brighter grid; baseline fixed at baseline_mags
        log_Pb_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)  # (M,N)
        log_Pf_vec = log_gaussian(mags, baseline_mags, errs)                # (N,)
        log_Pf_grid = np.broadcast_to(log_Pf_vec, (M, N))                    # (M,N)
        excursion_component = "bright"
        loglik_baseline_only = float(np.sum(log_Pf_vec))

        log_px_baseline = log_Pf_vec
        log_px_event = logsumexp(log_Pb_grid, axis=0) - np.log(M)
        log_bf_local = log_px_event - log_px_baseline

    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    max_log_bf_local = float(np.nanmax(log_bf_local)) if np.isfinite(log_bf_local).any() else np.nan

    # Mixture model evidence over (mag_event, p)
    log_p = np.log(p_grid)         # (P,)
    log_1mp = np.log1p(-p_grid)    # (P,)

    # (M,P,N)
    log_Pb_weighted = log_p[None, :, None] + log_Pb_grid[:, None, :]
    log_Pf_weighted = log_1mp[None, :, None] + log_Pf_grid[:, None, :]

    log_mix = np.logaddexp(log_Pb_weighted, log_Pf_weighted)  # (M,P,N)
    loglik = np.sum(log_mix, axis=2)                            # (M,P)

    # posterior mode on the grid (uniform priors)
    log_post_norm = loglik - logsumexp(loglik)
    best_m_idx, best_p_idx = np.unravel_index(np.nanargmax(log_post_norm), log_post_norm.shape)
    best_mag_event = float(mag_grid[int(best_m_idx)])
    best_p = float(p_grid[int(best_p_idx)])

    # global evidence (uniform over grid points)
    K = loglik.size
    log_evidence_mixture = float(logsumexp(loglik) - np.log(K))
    bayes_factor = float(log_evidence_mixture - loglik_baseline_only)

    # LOO responsibilities (optional; expensive)
    if compute_event_prob:
        event_prob = np.zeros(N, dtype=float)
        for j in range(N):
            loglik_excl = loglik - log_mix[:, :, j]  # (M,P)

            bright_num = loglik_excl + log_p[None, :] + log_Pb_grid[:, j][:, None]
            faint_num = loglik_excl + log_1mp[None, :] + log_Pf_grid[:, j][:, None]

            log_bright = logsumexp(bright_num)
            log_faint = logsumexp(faint_num)
            log_norm = logsumexp(np.array([log_bright, log_faint]))

            bright_prob = float(np.exp(log_bright - log_norm))
            faint_prob = float(np.exp(log_faint - log_norm))

            if excursion_component == "faint":
                event_prob[j] = faint_prob
            else:
                event_prob[j] = bright_prob
    else:
        event_prob = None

    # -----------------
    # Triggering: point -> runs -> gated runs
    # -----------------
    if trigger_mode == "logbf":
        per_point_thr = float(logbf_threshold)
        score_vec = np.asarray(log_bf_local, float)
        raw_idx = np.nonzero(np.isfinite(score_vec) & (score_vec >= per_point_thr))[0]
        trigger_threshold_used = per_point_thr
        trigger_value_max = max_log_bf_local

        # default run sum gate in logBF units
        if run_sum_threshold is None:
            run_sum_threshold_eff = float(run_sum_multiplier) * per_point_thr
        else:
            run_sum_threshold_eff = float(run_sum_threshold)

    elif trigger_mode == "posterior_prob":
        if event_prob is None:
            raise RuntimeError("trigger_mode='posterior_prob' requires compute_event_prob=True")

        thr_prob = significance_threshold / 100.0 if significance_threshold > 1.0 else float(significance_threshold)
        score_vec = np.asarray(event_prob, float)
        raw_idx = np.nonzero(np.isfinite(score_vec) & (score_vec >= thr_prob))[0]
        trigger_threshold_used = thr_prob
        trigger_value_max = float(np.nanmax(score_vec)) if score_vec.size else np.nan

        # default run sum gate in probability units
        if run_sum_threshold is None:
            run_sum_threshold_eff = float(run_min_points) * thr_prob
        else:
            run_sum_threshold_eff = float(run_sum_threshold)

    else:
        raise ValueError("trigger_mode must be 'logbf' or 'posterior_prob'")

    # build/gate runs
    kept_runs = []
    run_summaries = []

    if raw_idx.size == 0:
        event_indices = np.array([], dtype=int)
        significant = False
        run_stats = summarize_kept_runs([], jd, score_vec)
    else:
        runs = cluster_triggered_indices(
            raw_idx,
            jd,
            allow_gap_points=int(run_allow_gap_points),
            max_gap_days=run_max_gap_days,
        )

        kept_runs, initial_summaries = gate_runs(
            runs,
            jd,
            score_vec,
            min_points=int(run_min_points),
            min_duration_days=run_min_duration_days,
            per_point_threshold=trigger_threshold_used,
            sum_threshold=run_sum_threshold_eff,
        )

        # --- MORPHOLOGY CLASSIFICATION BLOCK ---
        final_summaries = []
        for i, r in enumerate(kept_runs):
            # Get the basic summary from gate_runs
            summary = initial_summaries[i]
            # Run morphology classification on this specific run
            morph_res = classify_run_morphology(jd, mags, errs, r, kind=kind)
            # Merge results into summary
            summary.update(morph_res)
            final_summaries.append(summary)
        
        run_summaries = final_summaries
        # ---------------------------------------

        if kept_runs:
            event_indices = np.unique(np.concatenate(kept_runs)).astype(int)
            significant = True
        else:
            event_indices = np.array([], dtype=int)
            significant = False

        run_stats = summarize_kept_runs(kept_runs, jd, score_vec)

    return dict(
        kind=str(kind),
        baseline_mag=float(baseline_mag),
        best_mag_event=float(best_mag_event),
        best_p=float(best_p),

        # diagnostics / scoring
        log_bf_local=log_bf_local,
        max_log_bf_local=float(max_log_bf_local) if np.isfinite(max_log_bf_local) else np.nan,
        event_probability=event_prob,
        used_sigma_eff=bool(used_sigma_eff),

        # trigger outputs (run-gated)
        trigger_mode=str(trigger_mode),
        trigger_threshold=float(trigger_threshold_used),
        trigger_max=float(trigger_value_max) if np.isfinite(trigger_value_max) else np.nan,
        event_indices=event_indices,
        significant=bool(significant),

        # run outputs
        run_sum_threshold=float(run_sum_threshold_eff),
        run_summaries=run_summaries,   # list[dict] (includes morphology now)
        **run_stats,

        # global model comparison
        bayes_factor=float(bayes_factor),
        log_evidence_mixture=float(log_evidence_mixture),
        log_evidence_baseline=float(loglik_baseline_only),

        # grids (debug)
        p_grid=p_grid,
        mag_grid=mag_grid,
    )


def run_bayesian_significance(
    df: pd.DataFrame,
    *,
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,

    # grids
    p_points: int = 80,
    p_min_dip: float | None = None,
    p_max_dip: float | None = None,
    p_min_jump: float | None = None,
    p_max_jump: float | None = None,
    mag_grid_dip: np.ndarray | None = None,
    mag_grid_jump: np.ndarray | None = None,

    # triggering
    trigger_mode: str = "logbf",
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    significance_threshold: float = 99.99997,

    # run-based triggers
    run_min_points: int = 3,
    run_allow_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,
    run_sum_threshold: float | None = None,
    run_sum_multiplier: float = 2.5,

    # noise
    use_sigma_eff: bool = True,
    require_sigma_eff: bool = True,

    # diagnostics
    compute_event_prob: bool = True,
):
    """
    Compute baseline ONCE, then reuse it for dip & jump scoring.
    """
    df = clean_lc(df)

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    df_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None

    dip = bayesian_excursion_significance(
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
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_dip,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,
        compute_event_prob=compute_event_prob,
    )

    jump = bayesian_excursion_significance(
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
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,
        compute_event_prob=compute_event_prob,
    )

    return dict(dip=dip, jump=jump)


# ----------------------------
# Multiprocessing wrapper
# ----------------------------

def _process_one(
    path: str,
    *,
    trigger_mode: str,
    logbf_threshold_dip: float,
    logbf_threshold_jump: float,
    significance_threshold: float,
    p_points: int,

    run_min_points: int,
    run_allow_gap_points: int,
    run_max_gap_days: float | None,
    run_min_duration_days: float | None,
    run_sum_threshold: float | None,
    run_sum_multiplier: float,

    compute_event_prob: bool,
):
    df = read_lc_dat2(path)

    res = run_bayesian_significance(
        df,
        trigger_mode=trigger_mode,
        logbf_threshold_dip=logbf_threshold_dip,
        logbf_threshold_jump=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        p_points=p_points,

        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,

        compute_event_prob=compute_event_prob,
        use_sigma_eff=True,
        require_sigma_eff=True,
    )

    dip = res["dip"]
    jump = res["jump"]

    # Helper to extract best morphology info
    def get_best_morph_info(run_list):
        if not run_list:
            return "none", 0.0, 0.0, 0.0
        # sort by run_sum descending (strongest detection first)
        best = sorted(run_list, key=lambda x: x['run_sum'], reverse=True)[0]
        
        morph = best.get('morphology', 'none')
        delta_bic = best.get('delta_bic_null', 0.0)
        
        # safely get params
        params = best.get('params', {})
        # approximate "best_param" based on morphology
        if morph == 'gaussian':
            # return sigma
            main_param = params.get('sigma', np.nan)
        elif morph == 'paczynski':
            # return tE
            main_param = params.get('tE', np.nan)
        else:
            main_param = np.nan
            
        return morph, float(delta_bic), float(main_param)

    dip_morph, dip_dbic, dip_param = get_best_morph_info(dip["run_summaries"])
    jump_morph, jump_dbic, jump_param = get_best_morph_info(jump["run_summaries"])

    return dict(
        path=str(path),

        dip_significant=bool(dip["significant"]),
        jump_significant=bool(jump["significant"]),

        # Morphology Results
        dip_best_morph=str(dip_morph),
        dip_best_delta_bic=float(dip_dbic),
        dip_best_width_param=float(dip_param),
        
        jump_best_morph=str(jump_morph),
        jump_best_delta_bic=float(jump_dbic),
        jump_best_width_param=float(jump_param),

        # counts (post run-gating)
        dip_count=int(len(dip["event_indices"])),
        jump_count=int(len(jump["event_indices"])),

        # run counts (post run-gating)
        dip_run_count=int(dip.get("n_runs", 0)),
        jump_run_count=int(jump.get("n_runs", 0)),

        # run summary maxima
        dip_max_run_points=int(dip.get("max_run_points", 0)),
        jump_max_run_points=int(jump.get("max_run_points", 0)),
        dip_max_run_duration=float(dip.get("max_run_duration", np.nan)),
        jump_max_run_duration=float(jump.get("max_run_duration", np.nan)),
        dip_max_run_sum=float(dip.get("max_run_sum", np.nan)),
        jump_max_run_sum=float(jump.get("max_run_sum", np.nan)),
        dip_max_run_max=float(dip.get("max_run_max", np.nan)),
        jump_max_run_max=float(jump.get("max_run_max", np.nan)),

        # local BF diagnostics
        dip_max_log_bf_local=float(dip.get("max_log_bf_local", np.nan)),
        jump_max_log_bf_local=float(jump.get("max_log_bf_local", np.nan)),

        # global BF diagnostics
        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),

        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),

        used_sigma_eff=bool(dip.get("used_sigma_eff", False) and jump.get("used_sigma_eff", False)),
        trigger_mode=str(trigger_mode),
        dip_trigger_threshold=float(dip.get("trigger_threshold", np.nan)),
        jump_trigger_threshold=float(jump.get("trigger_threshold", np.nan)),
        dip_run_sum_threshold=float(dip.get("run_sum_threshold", np.nan)),
        jump_run_sum_threshold=float(jump.get("run_sum_threshold", np.nan)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian excursion scoring on light curves in parallel."
    )
    parser.add_argument("inputs", nargs="+", help="Paths to light-curve files")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (mp.cpu_count() or 1) - 20),
        help="Number of worker processes (default: cpu_count-20)",
    )

    parser.add_argument(
        "--trigger-mode",
        type=str,
        default="logbf",
        choices=["logbf", "posterior_prob"],
        help="Triggering mode: 'logbf' (default) or 'posterior_prob' (slow)",
    )
    parser.add_argument(
        "--logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Per-point dip trigger: log BF_j >= this (default 5.0 ~ BF>150)",
    )
    parser.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Per-point jump trigger: log BF_j >= this (default 5.0 ~ BF>150)",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=99.99997,
        help="Only used if --trigger-mode posterior_prob (default ~ 5-sigma)",
    )
    parser.add_argument(
        "--p-points",
        type=int,
        default=80,
        help="Number of points in the logit-spaced p grid (default 80)",
    )

    # Run-based triggers (1,2,3)
    parser.add_argument(
        "--run-min-points",
        type=int,
        default=3,
        help="Min triggered points in a run (default 3)",
    )
    parser.add_argument(
        "--run-allow-gap-points",
        type=int,
        default=1,
        help="Allow up to this many missing indices inside a run (default 1)",
    )
    parser.add_argument(
        "--run-max-gap-days",
        type=float,
        default=None,
        help="Break runs if JD gap exceeds this (default cadence-adaptive)",
    )
    parser.add_argument(
        "--run-min-duration-days",
        type=float,
        default=None,
        help="Require run duration >= this (default cadence-adaptive)",
    )
    parser.add_argument(
        "--run-sum-threshold",
        type=float,
        default=None,
        help="Require run sum-score >= this (default derived from per-point threshold)",
    )
    parser.add_argument(
        "--run-sum-multiplier",
        type=float,
        default=2.5,
        help="If --run-sum-threshold is None: sum_thr = multiplier * per_point_thr (logbf mode)",
    )

    parser.add_argument(
        "--no-event-prob",
        action="store_true",
        help="Skip LOO event responsibilities (faster). Required unless using posterior_prob triggering.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path; if omitted, print to stdout.",
    )

    args = parser.parse_args()

    if args.trigger_mode == "posterior_prob" and args.no_event_prob:
        raise SystemExit("posterior_prob triggering requires event_prob; remove --no-event-prob")

    compute_event_prob = (not args.no_event_prob)

    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _process_one,
                path,
                trigger_mode=args.trigger_mode,
                logbf_threshold_dip=args.logbf_threshold_dip,
                logbf_threshold_jump=args.logbf_threshold_jump,
                significance_threshold=args.significance_threshold,
                p_points=args.p_points,
                run_min_points=args.run_min_points,
                run_allow_gap_points=args.run_allow_gap_points,
                run_max_gap_days=args.run_max_gap_days,
                run_min_duration_days=args.run_min_duration_days,
                run_sum_threshold=args.run_sum_threshold,
                run_sum_multiplier=args.run_sum_multiplier,
                compute_event_prob=compute_event_prob,
            ): path
            for path in args.inputs
        }

        for fut in tqdm(as_completed(futs), total=len(futs), desc="LCs", unit="lc"):
            path = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(dict(path=str(path), error=repr(e)))
                print(f"ERROR processing {path}: {e}", flush=True)

    if args.output:
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.output, index=False)
        print(f"Wrote {len(results)} rows to {args.output}")
    else:
        for row in results:
            print(
                f"{row['path']}\t"
                f"mode={row['trigger_mode']}\t"
                f"dip_sig={row['dip_significant']} ({row['dip_best_morph']}) dip_dBIC={row['dip_best_delta_bic']:.1f}\t"
                f"jump_sig={row['jump_significant']} ({row['jump_best_morph']}) jump_dBIC={row['jump_best_delta_bic']:.1f}"
            )

    if errors:
        print(f"Completed with {len(errors)} failures.", flush=True)


if __name__ == "__main__":
    main()