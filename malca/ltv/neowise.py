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
import io
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm.auto import tqdm
from astroquery.ipac.irsa import Irsa

from malca.config.config_ltv import (
    NEOWISE_EPOCH_COMBINE_DAYS,
    NEOWISE_MIN_SNR,
    NEOWISE_RATE_LIMIT_SECONDS,
    NEOWISE_MATCH_RADIUS_ARCSEC,
    LTV_WORKERS,
)


# NEOWISE epoch grouping: combine points within this many days
EPOCH_COMBINE_DAYS = NEOWISE_EPOCH_COMBINE_DAYS

# Minimum SNR for valid W1/W2 measurements
MIN_SNR = NEOWISE_MIN_SNR

# Rate limiting: seconds between requests per worker
RATE_LIMIT_SECONDS = NEOWISE_RATE_LIMIT_SECONDS



# =============================================================================
# IRSA TAP QUERY (bulk)
# =============================================================================

def query_neowise_lc_bulk(
    df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    id_col: str,
    *,
    match_radius_arcsec: float = NEOWISE_MATCH_RADIUS_ARCSEC,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query IRSA TAP for NEOWISE single-exposure photometry using bulk table upload.
    
    Uploads a table of coordinates and performs a spatial join on the server.
    Returns DataFrame with columns: mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, and the input id_col.
    """
    if df.empty:
        return pd.DataFrame()

    # Create astropy table for upload
    t = Table(
        [df[id_col].values, df[ra_col].values, df[dec_col].values], 
        names=('target_id', 'ra', 'dec'), 
        dtype=(int, float, float)
    )
    
    # Write to memory buffer in IPAC format
    f_str = io.StringIO()
    t.write(f_str, format='ipac')
    f_bytes = io.BytesIO(f_str.getvalue().encode('utf-8'))
    
    query = f"""
    SELECT 
        db.mjd AS mjd, db.w1mpro AS w1mpro, db.w1sigmpro AS w1sigmpro, db.w1snr AS w1snr, 
        db.w2mpro AS w2mpro, db.w2sigmpro AS w2sigmpro, db.w2snr AS w2snr, 
        db.qual_frame AS qual_frame, db.cc_flags AS cc_flags, 
        my_table.target_id AS target_id
    FROM neowiser_p1bs_psd AS db, TAP_UPLOAD.my_table AS my_table
    WHERE CONTAINS(POINT(db.ra, db.dec), CIRCLE(my_table.ra, my_table.dec, {match_radius_arcsec / 3600.0})) = 1
    """

    files = {'table.tbl': f_bytes}
    data = {
        'UPLOAD': 'my_table,param:table.tbl',
        'FORMAT': 'VOTABLE',
        'QUERY': query
    }

    try:
        if verbose:
            print(f"  Sending TAP query for {len(df)} targets...")
        
        response = requests.post('https://irsa.ipac.caltech.edu/TAP/sync', files=files, data=data, timeout=600)
        
        if response.status_code == 200:
            if b"ERROR" in response.content:
                 if verbose:
                     print(f"NEOWISE query error: {response.content.decode('utf-8')[:200]}")
                 return pd.DataFrame()
            else:
                 try:
                     result_table = Table.read(io.BytesIO(response.content), format='votable')
                     res_df = result_table.to_pandas()
                     if len(res_df.columns) == 10:
                         res_df.columns = [
                             'mjd', 'w1mpro', 'w1sigmpro', 'w1snr',
                             'w2mpro', 'w2sigmpro', 'w2snr',
                             'qual_frame', 'cc_flags', 'target_id'
                         ]
                     
                     # Filter bad data (same logic as before)
                     if "qual_frame" in res_df.columns:
                         res_df = res_df[res_df["qual_frame"].isin([0, 1])]
                     if "cc_flags" in res_df.columns:
                         # Need to handle bytes in astropy votable pandas conversion
                         if len(res_df) > 0 and res_df["cc_flags"].dtype == object and isinstance(res_df["cc_flags"].iloc[0], bytes):
                             res_df["cc_flags"] = res_df["cc_flags"].str.decode("utf-8")
                         res_df = res_df[~res_df["cc_flags"].str.contains("[^0]", regex=True, na=False)]
                     if "w1snr" in res_df.columns:
                         res_df = res_df[res_df["w1snr"] >= MIN_SNR]
                     if "w2snr" in res_df.columns:
                         res_df = res_df[res_df["w2snr"] >= MIN_SNR]
                         
                     # Rename target_id back to original id_col
                     res_df = res_df.rename(columns={"target_id": id_col})
                     return res_df.reset_index(drop=True)
                     
                 except Exception as e:
                     if verbose:
                         print(f"NEOWISE table parse error: {e}")
                     return pd.DataFrame()
        else:
            if verbose:
                print(f"NEOWISE query HTTP {response.status_code}: {response.content.decode('utf-8')[:200]}")
            return pd.DataFrame()
            
    except Exception as e:
        if verbose:
            print(f"NEOWISE request error: {e}")
        return pd.DataFrame()


def query_neowise_lc(
    ra: float,
    dec: float,
    *,
    match_radius_arcsec: float = NEOWISE_MATCH_RADIUS_ARCSEC,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query IRSA TAP for NEOWISE single-exposure photometry.
    Returns DataFrame with columns: mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro
    """
    df = pd.DataFrame({"ra": [ra], "dec": [dec], "id": [0]})
    res = query_neowise_lc_bulk(df, "ra", "dec", "id", match_radius_arcsec=match_radius_arcsec, verbose=verbose)
    if not res.empty and "id" in res.columns:
        res = res.drop(columns=["id"])
    return res


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
    match_radius_arcsec: float = NEOWISE_MATCH_RADIUS_ARCSEC,
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
    match_radius_arcsec: float = NEOWISE_MATCH_RADIUS_ARCSEC,
    epoch_days: float = EPOCH_COMBINE_DAYS,
    n_workers: int = 1,  # Kept for compatibility but not used in bulk mode
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extract NEOWISE light curves and fit trends for all sources.
    
    Uses bulk table upload queries for efficiency.
    
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
        
    df_valid = df[valid_mask].copy()
    df_valid["_temp_id"] = df_valid.index.values
    
    chunk_size = 500
    chunks = [df_valid.iloc[i:i + chunk_size] for i in range(0, len(df_valid), chunk_size)]
    
    if verbose:
        print(f"[extract_neowise_trends] Querying {len(df_valid)} sources in {len(chunks)} bulk chunks of {chunk_size}...")
        
    for i, chunk in enumerate(tqdm(chunks, desc="NEOWISE Bulk", disable=not verbose)):
        if verbose:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
        # Bulk query
        raw_lcs = query_neowise_lc_bulk(
            chunk, ra_column, dec_column, "_temp_id",
            match_radius_arcsec=match_radius_arcsec,
            verbose=verbose
        )
        
        if raw_lcs.empty:
            continue
            
        # Group by target and process
        for target_id, lc_raw in raw_lcs.groupby("_temp_id"):
            lc = combine_epochs(lc_raw, epoch_days=epoch_days)
            if lc.empty:
                continue
                
            trends = fit_neowise_trends(lc)
            for col in new_cols:
                if col in trends:
                    df.loc[target_id, col] = trends[col]
                    
    if verbose:
        n_with_data = (df["neowise_n_epochs"] > 0).sum()
        print(f"[extract_neowise_trends] {n_with_data}/{len(df)} sources have NEOWISE data")
    
    return df