"""
LTV NEOWISE Light Curve Extraction — Optimized for Scale.

Implements NEOWISE IR light curve extraction from Hwang & Zakamska (2020):
- Query IRSA TAP for NEOWISE single-exposure photometry
- Combine closely spaced points into epochs
- Fit W1 and W1-W2 color evolution with linear/quadratic functions

Optimized for scale with:
- Parallel IRSA TAP queries via ThreadPoolExecutor
- Chunked processing for memory efficiency
- Progress bars throughout
- Rate limiting to avoid API throttling

Note: NEOWISE bulk data is 42TB, so we use IRSA TAP queries.
For ~36K filtered candidates, parallel queries complete in ~1-2 hours.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from astroquery.ipac.irsa import Irsa


# NEOWISE epoch grouping: combine points within this many days
EPOCH_COMBINE_DAYS = 7.0

# Minimum SNR for valid W1/W2 measurements
MIN_SNR = 3.0

# Rate limiting: seconds between requests per worker
RATE_LIMIT_SECONDS = 0.1


# =============================================================================
# IRSA TAP QUERY (single source)
# =============================================================================

def query_neowise_lc(
    ra: float,
    dec: float,
    *,
    match_radius_arcsec: float = 3.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query IRSA TAP for NEOWISE single-exposure photometry.
    
    Uses the neowiser_p1bs_psd table (NEOWISE Reactivation Single Exposure Source Table).
    
    Returns DataFrame with columns: mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro
    """
    try:
        query = f"""
        SELECT 
            mjd,
            w1mpro, w1sigmpro, w1snr,
            w2mpro, w2sigmpro, w2snr,
            qual_frame,
            cc_flags
        FROM neowiser_p1bs_psd
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {match_radius_arcsec / 3600.0})
        ) = 1
        ORDER BY mjd ASC
        """

        result = Irsa.query_tap(query)

        if result is None or len(result) == 0:
            return pd.DataFrame()

        df = result.to_pandas()

        if "qual_frame" in df.columns:
            df = df[df["qual_frame"].isin([0, 1])]

        if "cc_flags" in df.columns:
            df = df[~df["cc_flags"].str.contains("[^0]", regex=True, na=False)]

        if "w1snr" in df.columns:
            df = df[df["w1snr"] >= MIN_SNR]
        if "w2snr" in df.columns:
            df = df[df["w2snr"] >= MIN_SNR]

        return df.reset_index(drop=True)

    except Exception as e:
        if verbose:
            print(f"NEOWISE query error for ({ra}, {dec}): {e}")
        return pd.DataFrame()


# =============================================================================
# EPOCH COMBINATION
# =============================================================================

def combine_epochs(
    lc: pd.DataFrame,
    *,
    epoch_days: float = EPOCH_COMBINE_DAYS,
) -> pd.DataFrame:
    """
    Combine closely spaced NEOWISE measurements into epochs.
    
    Following Hwang & Zakamska (2020): group points within epoch_days and
    compute weighted mean magnitudes.
    """
    if lc.empty or "mjd" not in lc.columns:
        return pd.DataFrame()
    
    mjd = lc["mjd"].values
    w1 = lc["w1mpro"].values
    w1sig = lc["w1sigmpro"].values
    w2 = lc["w2mpro"].values
    w2sig = lc["w2sigmpro"].values
    
    # Assign epoch groups
    epoch_ids = np.zeros(len(mjd), dtype=int)
    current_epoch = 0
    epoch_ids[0] = 0
    
    for i in range(1, len(mjd)):
        if mjd[i] - mjd[epoch_ids == current_epoch].mean() > epoch_days:
            current_epoch += 1
        epoch_ids[i] = current_epoch
    
    # Compute weighted means per epoch
    epochs = []
    for e in np.unique(epoch_ids):
        mask = epoch_ids == e
        
        # Weighted mean for W1
        w1_weights = 1.0 / (w1sig[mask] ** 2 + 1e-10)
        w1_mean = np.sum(w1[mask] * w1_weights) / np.sum(w1_weights)
        w1_err = 1.0 / np.sqrt(np.sum(w1_weights))
        
        # Weighted mean for W2
        w2_weights = 1.0 / (w2sig[mask] ** 2 + 1e-10)
        w2_mean = np.sum(w2[mask] * w2_weights) / np.sum(w2_weights)
        w2_err = 1.0 / np.sqrt(np.sum(w2_weights))
        
        epochs.append({
            "mjd": np.mean(mjd[mask]),
            "w1mpro": w1_mean,
            "w1err": w1_err,
            "w2mpro": w2_mean,
            "w2err": w2_err,
            "w1_w2": w1_mean - w2_mean,
            "n_points": mask.sum(),
        })
    
    return pd.DataFrame(epochs)


# =============================================================================
# TREND FITTING
# =============================================================================

def fit_neowise_trends(
    lc: pd.DataFrame,
) -> dict:
    """
    Fit linear and quadratic trends to W1 and W1-W2 color.
    
    Returns dict with trend metrics.
    """
    result = {
        "w1_slope": np.nan,
        "w1_quad_coeff": np.nan,
        "w1_w2_slope": np.nan,
        "w1_w2_quad_coeff": np.nan,
        "w1_w2_median": np.nan,
        "neowise_n_epochs": 0,
    }
    
    if lc.empty or len(lc) < 3:
        return result
    
    result["neowise_n_epochs"] = len(lc)
    
    # Convert MJD to years (relative to first epoch)
    mjd = lc["mjd"].values
    t_years = (mjd - mjd.min()) / 365.25
    
    # Fit W1
    w1 = lc["w1mpro"].values
    try:
        lin_fit = np.polyfit(t_years, w1, 1)
        result["w1_slope"] = float(lin_fit[0])
        
        if len(lc) >= 4:
            quad_fit = np.polyfit(t_years, w1, 2)
            result["w1_quad_coeff"] = float(quad_fit[0])
    except Exception:
        pass
    
    # Fit W1-W2 color
    w1_w2 = lc["w1_w2"].values
    result["w1_w2_median"] = float(np.median(w1_w2))
    
    try:
        lin_fit = np.polyfit(t_years, w1_w2, 1)
        result["w1_w2_slope"] = float(lin_fit[0])
        
        if len(lc) >= 4:
            quad_fit = np.polyfit(t_years, w1_w2, 2)
            result["w1_w2_quad_coeff"] = float(quad_fit[0])
    except Exception:
        pass
    
    return result


# =============================================================================
# PARALLEL EXTRACTION
# =============================================================================

def _extract_one_source(
    ra: float,
    dec: float,
    idx: int,
    *,
    match_radius_arcsec: float = 3.0,
    epoch_days: float = EPOCH_COMBINE_DAYS,
) -> dict | None:
    """Extract NEOWISE trends for a single source."""
    try:
        # Rate limit
        time.sleep(RATE_LIMIT_SECONDS)
        
        lc_raw = query_neowise_lc(ra, dec, match_radius_arcsec=match_radius_arcsec)
        
        if lc_raw.empty:
            return None
        
        lc = combine_epochs(lc_raw, epoch_days=epoch_days)
        
        if lc.empty:
            return None
        
        trends = fit_neowise_trends(lc)
        trends["_idx"] = idx
        
        return trends
    except Exception:
        return None


def extract_neowise_trends(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = 3.0,
    epoch_days: float = EPOCH_COMBINE_DAYS,
    n_workers: int = 8,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extract NEOWISE light curves and fit trends for all sources.
    
    Uses parallel processing for efficiency.
    
    Adds columns:
    - w1_slope: Linear slope of W1 (mag/yr)
    - w1_quad_coeff: Quadratic coefficient of W1
    - w1_w2_slope: Linear slope of W1-W2 color (mag/yr)
    - w1_w2_quad_coeff: Quadratic coefficient of W1-W2
    - w1_w2_median: Median W1-W2 color
    - neowise_n_epochs: Number of NEOWISE epochs
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for NEOWISE extraction")
        return df
    
    df = df.copy()
    
    # Initialize new columns
    new_cols = [
        "w1_slope", "w1_quad_coeff",
        "w1_w2_slope", "w1_w2_quad_coeff", "w1_w2_median",
        "neowise_n_epochs"
    ]
    for col in new_cols:
        df[col] = np.nan
    df["neowise_n_epochs"] = 0
    
    valid_mask = df[ra_column].notna() & df[dec_column].notna()
    
    if not valid_mask.any():
        return df
    
    # Prepare tasks
    tasks = [
        (df.loc[idx, ra_column], df.loc[idx, dec_column], idx)
        for idx in df.index[valid_mask]
    ]
    
    if verbose:
        print(f"[extract_neowise_trends] Querying {len(tasks)} sources...")
        print(f"  Workers: {n_workers}, Rate limit: {RATE_LIMIT_SECONDS}s/request")
        estimated_time = len(tasks) * RATE_LIMIT_SECONDS / n_workers
        print(f"  Estimated time: {estimated_time/60:.1f} minutes")
    
    # Process in parallel
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _extract_one_source,
                ra, dec, idx,
                match_radius_arcsec=match_radius_arcsec,
                epoch_days=epoch_days,
            ): idx
            for ra, dec, idx in tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="NEOWISE", disable=not verbose):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Merge results back
    for r in results:
        idx = r["_idx"]
        if idx in df.index:
            for col in new_cols:
                if col in r:
                    df.loc[idx, col] = r[col]
    
    if verbose:
        n_with_data = (df["neowise_n_epochs"] > 0).sum()
        print(f"[extract_neowise_trends] {n_with_data}/{len(df)} sources have NEOWISE data")
    
    return df
