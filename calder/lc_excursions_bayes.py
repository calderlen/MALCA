import numpy as np
import pandas as pd
from scipy.special import logsumexp

from lc_utils import read_lc_dat, read_lc_raw, match_index_to_lc

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
    df = df.loc[mask]

    df = df.sort_values("JD").reset_index(drop=True)
    return df

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
        out = {f"{prefix}_{k}": vals[k] for k in METRIC_NAMES}
        out[f"{prefix}_is_dip_dominated"] = False
        return out


def log_gaussian(mags, mean, sigma):
    mags = np.asarray(mags, float)
    mean = np.asarray(mean, float)
    sigma = np.asarray(sigma, float)

    return -((mags - mean) ** 2) / (sigma**2)


def default_mag_grid(baseline, mags, kind, n=50):

    lo, hi = np.nanpercentile(mags, [5, 95])
    spread = max(hi - lo, 0.05)

    if kind == "dip":
        start = baseline + 0.02
        stop = max(baseline + 0.02, hi + 0.5 * spread)
    elif kind == "jump":
        start = min(baseline - 0.02, lo - 0.5 * spread)
        stop = baseline - 0.02

    if start == stop:
        stop = start + 0.1 if kind == "dip" else start - 0.1

    return np.linspace(start, stop, n)


def bayesian_excursion_significance(
    df,
    *,
    kind="dip",
    mag_col="mag",
    err_col="error",
    baseline_func=None,
    baseline_kwargs=None,
    use_per_point_baseline=True,
    p_grid=None,
    mag_grid=None,
    significance_threshold=0.99,
    eps=1e-6,
):
    """
    kind:
        "dip"  -> excursion component is faint; baseline is bright median.
        "jump" -> excursion component is bright; baseline is median (faint).

    Returns a dict with per-point excursion probabilities, best-fit grid values,
    and a Bayes factor vs. the baseline-only (no excursion) model.
    """
    if df is None or len(df) == 0:
        return {
            "kind": kind,
            "event_probability": np.array([]),
            "event_indices": np.array([], dtype=int),
            "significant": False,
            "baseline_mag": np.nan,
            "best_mag_event": np.nan,
            "best_p": np.nan,
            "bayes_factor": -np.inf,
            "log_evidence_mixture": -np.inf,
            "log_evidence_baseline": -np.inf,
            "p_grid": np.asarray([]),
            "mag_grid": np.asarray([]),
        }

    df = clean_lc(df)
    baseline_kwargs = baseline_kwargs or {}
    mags = np.asarray(df[mag_col], float)
    errs = (
        np.asarray(df[err_col], float)
        if err_col in df.columns
        else np.full_like(mags, np.nan)
    )
    if not np.isfinite(errs).any():
        errs = np.full_like(mags, 0.05)
    errs = np.where(np.isfinite(errs) & (errs > eps), errs, np.nanmedian(errs))

    if baseline_func is None:
        baseline_vals = np.full_like(mags, np.nanmedian(mags))
    else:
        df_base = baseline_func(df, **baseline_kwargs)
        if "baseline" in df_base.columns:
            baseline_vals = np.asarray(df_base["baseline"], float)
        else:
            baseline_vals = np.asarray(df_base[mag_col], float)
    baseline_mag = float(np.nanmedian(baseline_vals))
    if not use_per_point_baseline or not np.isfinite(baseline_vals).any():
        baseline_vals = np.full_like(mags, baseline_mag)

    if p_grid is None:
        p_grid = np.linspace(0.01, 0.9, 60)
    else:
        p_grid = np.asarray(p_grid, float)

    if mag_grid is None:
        mag_grid = default_mag_grid(baseline_mag, mags, kind, n=60)
    else:
        mag_grid = np.asarray(mag_grid, float)

    M = len(mag_grid)
    P = len(p_grid)
    N = len(mags)

    # Component means
    if kind == "dip":
        # Pb: baseline (bright, fixed at median), Pf: excursion (faint, grid)
        log_pb_vec = log_gaussian(mags, baseline_vals, errs, eps=eps)
        log_pb_grid = np.broadcast_to(log_pb_vec, (M, N))
        log_pf_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs, eps=eps)
        excursion_component = "faint"
        loglik_baseline_only = np.sum(log_pb_vec)
    else:
        # Pb: excursion (bright, grid), Pf: baseline (faint, fixed at median)
        log_pb_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs, eps=eps)
        log_pf_vec = log_gaussian(mags, baseline_vals, errs, eps=eps)
        log_pf_grid = np.broadcast_to(log_pf_vec, (M, N))
        excursion_component = "bright"
        loglik_baseline_only = np.sum(log_pf_vec)

    log_p = np.log(p_grid)
    log_1mp = np.log1p(-p_grid)

    # log_mix shape: (M, P, N)
    log_mix = np.logaddexp(
        log_p[None, :, None] + log_pb_grid[:, None, :],
        log_1mp[None, :, None] + log_pf_grid[:, None, :],
    )
    loglik = np.sum(log_mix, axis=2)  # (M, P)

    # Posterior on the grid (uniform priors)
    log_post_norm = loglik - logsumexp(loglik)
    best_m_idx, best_p_idx = np.unravel_index(np.nanargmax(log_post_norm), log_post_norm.shape)
    best_mag_event = float(mag_grid[best_m_idx])
    best_p = float(p_grid[best_p_idx])

    K = loglik.size
    log_evidence_mixture = logsumexp(loglik) - np.log(K)
    bayes_factor = log_evidence_mixture - loglik_baseline_only

    # Per-point excursion probability via marginalization over p, mag_grid
    event_prob = np.zeros(N, dtype=float)
    log_mix_den = logsumexp(loglik)  # shared denominator (evidence)

    for j in range(N):
        loglik_noj = loglik - log_mix[:, :, j]
        bright_num = loglik_noj + log_p[None, :] + log_pb_grid[:, None, j]
        faint_num = loglik_noj + log_1mp[None, :] + log_pf_grid[:, None, j]

        log_bright = logsumexp(bright_num)
        log_faint = logsumexp(faint_num)
        log_norm = logsumexp(np.array([log_bright, log_faint]))

        bright_prob = np.exp(log_bright - log_norm)
        faint_prob = np.exp(log_faint - log_norm)
        event_prob[j] = faint_prob if excursion_component == "faint" else bright_prob

    event_indices = np.nonzero(event_prob >= significance_threshold)[0]
    significant = event_prob.size > 0 and float(np.nanmax(event_prob)) >= significance_threshold

    return {
        "kind": kind,
        "baseline_mag": baseline_mag,
        "best_mag_event": best_mag_event,
        "best_p": best_p,
        "event_probability": event_prob,
        "event_indices": event_indices,
        "significant": significant,
        "significance_threshold": significance_threshold,
        "bayes_factor": bayes_factor,
        "log_evidence_mixture": log_evidence_mixture,
        "log_evidence_baseline": loglik_baseline_only,
        "p_grid": p_grid,
        "mag_grid": mag_grid,
    }


def run_bayesian_significance(
    df,
    *,
    mag_col="mag",
    err_col="error",
    baseline_func=None,
    baseline_kwargs=None,
    use_per_point_baseline=True,
    p_grid=None,
    mag_grid_dip=None,
    mag_grid_jump=None,
    significance_threshold=0.99,
    eps=1e-6,
):
    """
    Convenience wrapper to evaluate both dips and jumps with the same inputs
    used by lc_excursions.py.
    """
    dip = bayesian_excursion_significance(
        df,
        kind="dip",
        mag_col=mag_col,
        err_col=err_col,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        use_per_point_baseline=use_per_point_baseline,
        p_grid=p_grid,
        mag_grid=mag_grid_dip,
        significance_threshold=significance_threshold,
        eps=eps,
    )
    jump = bayesian_excursion_significance(
        df,
        kind="jump",
        mag_col=mag_col,
        err_col=err_col,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        use_per_point_baseline=use_per_point_baseline,
        p_grid=p_grid,
        mag_grid=mag_grid_jump,
        significance_threshold=significance_threshold,
        eps=eps,
    )
    return {"dip": dip, "jump": jump}