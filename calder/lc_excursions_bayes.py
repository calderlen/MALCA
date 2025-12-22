import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm

from lc_utils import read_lc_dat2, read_lc_raw, match_index_to_lc

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


def logit_spaced_grid(p_min=1e-4, p_max=1 - 1e-4, n=80):

    p_min = float(np.clip(p_min, 1e-12, 1 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1 - 1e-12))

    q_min = np.log(p_min / (1.0 - p_min))
    q_max = np.log(p_max / (1.0 - p_max))
    q = np.linspace(q_min, q_max, int(n))
    return 1.0 / (1.0 + np.exp(-q))


def bayesian_excursion_significance(
    df,
    *,
    kind="dip",
    mag_col="mag",
    err_col="error",
    baseline_func=None,
    baseline_kwargs=None,
    p_min=None,
    p_max=None,
    p_points=80,
    mag_grid=None,
    significance_threshold=0.99,
):
    """
    kind: "dip" or "jump"

    returns dict with (1) per-point excursion probabilities, (2) best-fit grid values, (3) a Bayes factor vs. the baseline-only (no excursion) model.
    """

    df = clean_lc(df)

    baseline_kwargs = baseline_kwargs or {}

    mags = np.asarray(df[mag_col], float)

    if err_col in df.columns:
        errs = np.asarray(df[err_col], float)
    else:
        errs = np.full_like(mags, 0.05)

    if baseline_func is None:
        baseline_mags = np.full_like(mags, np.nanmedian(mags))

    else:
        df_base = baseline_func(df, **baseline_kwargs)
        if "baseline" in df_base.columns:
            baseline_mags = np.asarray(df_base["baseline"], float)
        else:
            baseline_mags = np.asarray(df_base[mag_col], float)
            
    baseline_mag = float(np.nanmedian(baseline_mags))

    if p_min is None and p_max is None:
        if kind == "dip":
            p_min, p_max = 0.5, 1 - 1e-4  # rare dips => 1 - p small
        elif kind == "jump":
            p_min, p_max = 1e-4, 0.5      # rare jumps => p small

    p_grid = logit_spaced_grid(p_min=p_min, p_max=p_max, n=p_points)

    if mag_grid is None:
        mag_grid = default_mag_grid(baseline_mag, mags, kind, n=60)
    else:
        mag_grid = np.asarray(mag_grid, float)

    mag_grid_len = len(mag_grid)
    p_grid_len = len(p_grid)
    mag_num = len(mags)

    # creating log-likelihoods of baseline-only (null) model to later compare to the mixture model and calculate Bayes factor
    if kind == "dip":
        # Pb: baseline (bright, fixed at median), Pf: excursion (faint, grid)
        log_Pb_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pb_grid = np.broadcast_to(log_Pb_vec, (mag_grid_len, mag_num))
        log_Pf_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)
        excursion_component = "faint"
        loglik_baseline_only = np.sum(log_Pb_vec)
    elif kind == "jump":
        # Pb: excursion (bright, grid), Pf: baseline (faint, fixed at median)
        log_Pb_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)
        log_Pf_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pf_grid = np.broadcast_to(log_Pf_vec, (mag_grid_len, mag_num))
        excursion_component = "bright"
        loglik_baseline_only = np.sum(log_Pf_vec)

    log_p = np.log(p_grid)
    log_1mp = np.log1p(-p_grid)

    log_Pb_weighted = log_p[None, :, None] + log_Pb_grid[:, None, :]
    log_Pf_weighted = log_1mp[None, :, None] + log_Pf_grid[:, None, :]

    # log_mix shape: (p_grid_len, mag_grid_len, mag_num)
    log_mix = np.logaddexp(log_Pb_weighted, log_Pf_weighted)
    loglik = np.sum(log_mix, axis=2)  # (mag_grid_len, p_grid_len)

    # Posterior on the grid (uniform priors)
    log_post_norm = loglik - logsumexp(loglik)
    best_m_idx, best_p_idx = np.unravel_index(np.nanargmax(log_post_norm), log_post_norm.shape)
    best_mag_event = float(mag_grid[best_m_idx])
    best_p = float(p_grid[best_p_idx])

    K = loglik.size
    log_evidence_mixture = logsumexp(loglik) - np.log(K)
    bayes_factor = log_evidence_mixture - loglik_baseline_only

    # Per-point excursion probability via marginalization over p, mag_grid
    event_prob = np.zeros(mag_num, dtype=float)
    log_mix_denom = logsumexp(loglik)  # shared denominator (evidence)

    for j in range(mag_num):
        loglik_excluding_j = loglik - log_mix[:, :, j] # leave one out
        bright_num = loglik_excluding_j + log_p[None, :] + log_Pb_grid[:, None, j]
        faint_num = loglik_excluding_j + log_1mp[None, :] + log_Pf_grid[:, None, j]

        log_bright = logsumexp(bright_num)
        log_faint = logsumexp(faint_num)
        log_norm = logsumexp(np.array([log_bright, log_faint]))

        bright_prob = np.exp(log_bright - log_norm)
        faint_prob = np.exp(log_faint - log_norm)
        if excursion_component == "faint":
            event_prob[j] = faint_prob
        elif excursion_component == "bright":
            event_prob[j] = bright_prob

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
    p_min_dip=None,
    p_max_dip=None,
    p_min_jump=None,
    p_max_jump=None,
    p_points=80,
    mag_grid_dip=None,
    mag_grid_jump=None,
    significance_threshold=0.9545, #2-sigma
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
        p_min=p_min_dip,
        p_max=p_max_dip,
        p_points=p_points,
        mag_grid=mag_grid_dip,
        significance_threshold=significance_threshold,
    )
    jump = bayesian_excursion_significance(
        df,
        kind="jump",
        mag_col=mag_col,
        err_col=err_col,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        p_min=p_min_jump,
        p_max=p_max_jump,
        p_points=p_points,
        mag_grid=mag_grid_jump,
        significance_threshold=significance_threshold,
    )
    return {"dip": dip, "jump": jump}


# ----------------------------
# CLI / multiprocessing runner
# ----------------------------

# Keep BLAS/OMP single-threaded per worker to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _process_one(path, *, significance_threshold=0.9545, p_points=80):
    df = read_lc_dat2(path)
    res = run_bayesian_significance(
        df,
        significance_threshold=significance_threshold,
        p_points=p_points,
    )
    dip = res["dip"]
    jump = res["jump"]
    return {
        "path": path,
        "dip_significant": bool(dip["significant"]),
        "jump_significant": bool(jump["significant"]),
        "dip_bayes_factor": float(dip["bayes_factor"]),
        "jump_bayes_factor": float(jump["bayes_factor"]),
        "dip_best_p": float(dip["best_p"]),
        "jump_best_p": float(jump["best_p"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian excursion significance on light curves in parallel."
    )
    parser.add_argument("inputs", nargs="+", help="Paths to light-curve files")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 20),
        help="Number of worker processes (default: cpu_count-20)",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.9545,
        help="Per-point significance threshold (default: 0.9545 ~ 2-sigma)",
    )
    parser.add_argument(
        "--p-points",
        type=int,
        default=80,
        help="Number of points in the logit-spaced p grid (default: 80)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path to write results; if omitted, print to stdout.",
    )
    args = parser.parse_args()

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _process_one,
                path,
                significance_threshold=args.significance_threshold,
                p_points=args.p_points,
            ): path
            for path in args.inputs
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LCs", unit="lc"):
            path = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append({"path": path, "error": repr(e)})
                print(f"ERROR processing {path}: {e}", flush=True)

    if args.output:
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.output, index=False)
        print(f"Wrote {len(results)} rows to {args.output}")
    else:
        for row in results:
            print(
                f"{row['path']}\t"
                f"dip_sig={row['dip_significant']}\tjump_sig={row['jump_significant']}\t"
                f"dip_bf={row['dip_bayes_factor']:.3f}\tjump_bf={row['jump_bayes_factor']:.3f}"
            )

    if errors:
        print(f"Completed with {len(errors)} failures.", flush=True)

if __name__ == "__main__":
    main()