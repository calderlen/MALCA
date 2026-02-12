"""
LTV False Positive Filtering — Optimized for Scale.

Implements filtering steps from the paper to remove false positives:
- Slope/Δg threshold filtering (vectorized, instant)
- South pole artifact removal (vectorized, instant)
- High proper motion removal (batch Gaia TAP)
- Bright star artifact removal (batch Gaia TAP)

Optimized for 17M+ sources with:
- Batch TAP queries (upload source tables)
- Parallel processing via ThreadPoolExecutor
- Progress bars throughout
- Chunked processing for memory efficiency
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from malca.config.config_ltv import (
    LTV_MIN_SLOPE,
    LTV_MIN_DIFF,
    LTV_MIN_DEC,
    LTV_MAX_PM,
    LTV_MAX_REDUCED_CHI2,
    LTV_MAX_SINGLE_JUMP_FRACTION,
    LTV_MAX_EB_PERIOD_DAYS,
    LTV_MIN_LS_POWER,
    LTV_MAX_LS_FAP,
    LTV_MAX_CROWDING_COUNT,
    LTV_MATCH_RADIUS_ARCSEC,
    LTV_CHUNK_SIZE,
    LTV_GAIA_CHUNK_SIZE,
    LTV_CROSSMATCH_CHUNK_SIZE,
    LTV_BRIGHT_STAR_COEFF,
    LTV_BRIGHT_STAR_MAG_OFFSET,
    LTV_BRIGHT_STAR_BASE_ARCSEC,
    LTV_BRIGHT_STAR_MAX_ARCSEC,
    LTV_BRIGHT_MAG_LIMIT,
    LTV_MIN_OVERLAP_DAYS,
    LTV_MIN_OVERLAP_FRACTION,
    LTV_CROWDING_SEARCH_RADIUS_ARCSEC,
)


# =============================================================================
# LOGGING UTILITY
# =============================================================================

def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None = None,
) -> None:
    """Log rejected candidates to a CSV file."""
    if log_csv is None:
        return
    
    if "ASAS-SN ID" in df_before.columns:
        id_col = "ASAS-SN ID"
    else:
        id_col = df_before.columns[0]
    
    before_ids = set(df_before[id_col].astype(str))
    after_ids = set(df_after[id_col].astype(str))
    rejected_ids = before_ids - after_ids
    
    if not rejected_ids:
        return
    
    rejected = df_before[df_before[id_col].astype(str).isin(rejected_ids)].copy()
    rejected["rejection_reason"] = filter_name
    
    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    header = not log_path.exists() or log_path.stat().st_size == 0
    rejected.to_csv(log_path, mode="a", header=header, index=False)


# =============================================================================
# VECTORIZED THRESHOLD FILTERS (instant, no API calls)
# =============================================================================

def filter_slope_threshold(
    df: pd.DataFrame,
    *,
    min_slope: float = LTV_MIN_SLOPE,
    slope_column: str = "Slope",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep sources with |Slope| > min_slope (mag/yr).
    Vectorized — runs instantly on any size.
    """
    n0 = len(df)
    
    if slope_column not in df.columns:
        if verbose:
            print(f"Warning: '{slope_column}' column not found, skipping filter")
        return df
    
    mask = np.abs(df[slope_column].values) > min_slope
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_slope_threshold] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_slope_threshold", log_csv)
    return df_out


def filter_max_diff_threshold(
    df: pd.DataFrame,
    *,
    min_diff: float = LTV_MIN_DIFF,
    diff_column: str = "max diff",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep sources with |max diff| > min_diff (Δg in magnitudes).
    Vectorized — runs instantly on any size.
    """
    n0 = len(df)
    
    if diff_column not in df.columns:
        if verbose:
            print(f"Warning: '{diff_column}' column not found, skipping filter")
        return df
    
    mask = np.abs(df[diff_column].values) > min_diff
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_max_diff_threshold] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_max_diff_threshold", log_csv)
    return df_out


def filter_south_pole(
    df: pd.DataFrame,
    *,
    min_dec: float = LTV_MIN_DEC,
    dec_column: str = "dec_deg",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources near celestial south pole (dec < min_dec).
    Vectorized — runs instantly on any size.
    """
    n0 = len(df)
    
    if dec_column not in df.columns:
        if verbose:
            print(f"Warning: '{dec_column}' column not found, skipping filter")
        return df
    
    mask = df[dec_column].values >= min_dec
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_south_pole] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_south_pole", log_csv)
    return df_out


def filter_photometric_scatter(
    df: pd.DataFrame,
    *,
    max_reduced_chi2: float = LTV_MAX_REDUCED_CHI2,
    slope_column: str = "Slope",
    dispersion_column: str = "Dispersion",
    median_err_column: str = "Median_err",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources where high scatter suggests noise rather than real trend.
    
    Uses reduced χ² = dispersion² / expected_variance to identify sources
    where the observed scatter is inconsistent with a linear trend.
    
    Vectorized — runs instantly on any size.
    """
    n0 = len(df)
    
    # Need dispersion and error columns
    if dispersion_column not in df.columns:
        if verbose:
            print(f"Warning: '{dispersion_column}' not found, skipping scatter filter")
        return df
    
    # Compute reduced chi-squared proxy
    # High dispersion relative to expected noise = bad fit = likely noise artifact
    dispersion = df[dispersion_column].values
    
    if median_err_column in df.columns:
        err = df[median_err_column].values
        # chi2 ~ (dispersion / error)^2
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = (dispersion / np.maximum(err, 0.01)) ** 2
    else:
        # Fallback: use dispersion directly with typical error
        chi2 = (dispersion / 0.02) ** 2
    
    # If slope is small AND chi2 is high → noise artifact
    if slope_column in df.columns:
        slope = np.abs(df[slope_column].values)
        # Only reject if slope is marginal (< 0.05) AND scatter is high
        mask = ~((chi2 > max_reduced_chi2) & (slope < 0.05))
    else:
        mask = chi2 <= max_reduced_chi2
    
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_photometric_scatter] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_photometric_scatter", log_csv)
    return df_out


def filter_transient_contamination(
    df: pd.DataFrame,
    *,
    max_single_jump_fraction: float = LTV_MAX_SINGLE_JUMP_FRACTION,
    min_seasons: int = 3,
    coeff1_column: str = "coeff1",
    coeff2_column: str = "coeff2",
    max_diff_column: str = "max diff",
    slope_column: str = "Slope",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources where variability is driven by a single outlier jump.
    
    A real LTV source should show gradual change across multiple seasons.
    If >60% of total Δg comes from one season transition, it's likely
    a transient contamination (nova, flare, bad epoch).
    
    Vectorized — runs instantly on any size.
    """
    n0 = len(df)
    
    if max_diff_column not in df.columns or slope_column not in df.columns:
        if verbose:
            print("Warning: Required columns not found, skipping transient filter")
        return df
    
    max_diff = np.abs(df[max_diff_column].values)
    slope = np.abs(df[slope_column].values)
    
    # Approximate total change over baseline (assume ~5 year baseline)
    total_change = slope * 5.0
    
    # Fraction of total change in single jump
    with np.errstate(divide='ignore', invalid='ignore'):
        single_jump_fraction = max_diff / np.maximum(total_change, 0.01)
    
    # Keep sources where max single jump is < threshold of total change
    # OR where the total change is large enough to be robust
    mask = (single_jump_fraction < max_single_jump_fraction) | (total_change > 0.5)
    
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_transient_contamination] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_transient_contamination", log_csv)
    return df_out


def filter_eclipsing_binary_signature(
    df: pd.DataFrame,
    *,
    max_eb_period_days: float = LTV_MAX_EB_PERIOD_DAYS,
    min_ls_power: float = LTV_MIN_LS_POWER,
    max_ls_fap: float = LTV_MAX_LS_FAP,
    ls_period_column: str = "ls_period",
    ls_power_column: str = "ls_power",
    ls_fap_column: str = "ls_fap",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove likely eclipsing binaries misclassified as LTV.
    
    EBs with periods <100 days can create artificial long-term trends
    when seasonal medians sample different eclipse phases.
    
    Uses Lomb-Scargle periodogram results (already computed in core.py).
    Vectorized — runs instantly on any size.
    """
    n0 = len(df)
    
    if ls_period_column not in df.columns:
        if verbose:
            print(f"Warning: '{ls_period_column}' not found, skipping EB filter")
        return df
    
    period = df[ls_period_column].values
    power = df[ls_power_column].values if ls_power_column in df.columns else np.zeros(len(df))
    fap = df[ls_fap_column].values if ls_fap_column in df.columns else np.ones(len(df))
    
    # EB signature: short period + high power + low FAP
    is_eb = (period < max_eb_period_days) & (power > min_ls_power) & (fap < max_ls_fap)
    
    mask = ~is_eb
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_eclipsing_binary_signature] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_eclipsing_binary_signature", log_csv)
    return df_out


# =============================================================================
# BATCH GAIA TAP QUERIES
# =============================================================================

def _batch_gaia_cone_query(
    coords_df: pd.DataFrame,
    *,
    select_cols: str,
    extra_where: str = "",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Batch Gaia TAP query using table upload for efficient cone search.

    Uses pyvo async job with uploaded coordinate table for server-side crossmatch.
    """
    import pyvo
    from astropy.table import Table
    from malca.config.config_paths import GAIA_AIP_TAP_URL

    if coords_df.empty:
        return pd.DataFrame()

    tap = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)
    results = []
    chunks = [coords_df.iloc[i:i+chunk_size] for i in range(0, len(coords_df), chunk_size)]

    def process_chunk(chunk_df):
        """Process a single chunk via TAP upload."""
        try:
            upload_table = Table.from_pandas(chunk_df[["_idx", "ra", "dec"]])

            query = f"""
            SELECT
                u._idx as _idx,
                g.source_id,
                {select_cols},
                DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', u.ra, u.dec)) * 3600.0 as sep_arcsec
            FROM TAP_UPLOAD.upload_table AS u
            JOIN gaiadr3.gaia_source AS g
            ON 1=CONTAINS(
                POINT('ICRS', g.ra, g.dec),
                CIRCLE('ICRS', u.ra, u.dec, {match_radius_arcsec / 3600.0})
            )
            {extra_where}
            """

            result = tap.run_async(query, uploads={"upload_table": upload_table})
            return result.to_table().to_pandas() if result else pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"Gaia batch query error: {e}")
            return pd.DataFrame()

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Gaia batch query", disable=not verbose):
            result = future.result()
            if not result.empty:
                results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def query_gaia_proper_motions_batch(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Batch query Gaia DR3 for proper motions.
    
    Uses TAP upload for efficient server-side crossmatch.
    Returns df with added columns: gaia_pmra, gaia_pmdec, gaia_pm_total
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for Gaia PM query")
        return df
    
    df = df.copy()
    df["gaia_pmra"] = np.nan
    df["gaia_pmdec"] = np.nan
    df["gaia_pm_total"] = np.nan
    
    valid_mask = df[ra_column].notna() & df[dec_column].notna()
    
    if not valid_mask.any():
        return df
    
    # Prepare upload table
    coords_df = pd.DataFrame({
        "_idx": df.index[valid_mask],
        "ra": df.loc[valid_mask, ra_column].values,
        "dec": df.loc[valid_mask, dec_column].values,
    })
    
    if verbose:
        print(f"Querying Gaia for {len(coords_df)} sources...")
    
    result = _batch_gaia_cone_query(
        coords_df,
        select_cols="g.pmra, g.pmdec",
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
    )
    
    if result.empty:
        return df
    
    # Keep only closest match per source
    result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
    
    # Merge back
    for _, row in result.iterrows():
        idx = int(row["_idx"])
        if idx in df.index:
            df.loc[idx, "gaia_pmra"] = row["pmra"] if pd.notna(row["pmra"]) else np.nan
            df.loc[idx, "gaia_pmdec"] = row["pmdec"] if pd.notna(row["pmdec"]) else np.nan
            if pd.notna(row["pmra"]) and pd.notna(row["pmdec"]):
                df.loc[idx, "gaia_pm_total"] = np.sqrt(row["pmra"]**2 + row["pmdec"]**2)
    
    if verbose:
        n_matched = df["gaia_pm_total"].notna().sum()
        print(f"[query_gaia_proper_motions_batch] Matched {n_matched}/{len(df)}")
    
    return df


def filter_high_proper_motion(
    df: pd.DataFrame,
    *,
    max_pm: float = LTV_MAX_PM,
    pm_column: str = "gaia_pm_total",
    query_gaia: bool = True,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources with proper motion > max_pm (mas/yr).
    Uses batch Gaia TAP queries for efficiency.
    """
    n0 = len(df)
    
    # Query Gaia if needed
    if pm_column not in df.columns and query_gaia:
        df = query_gaia_proper_motions_batch(
            df,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
        )
    
    if pm_column not in df.columns:
        if verbose:
            print(f"Warning: '{pm_column}' column not found, skipping filter")
        return df
    
    # Keep sources with PM <= threshold (or NaN PM = keep by default)
    pm_values = df[pm_column].values
    mask = (pm_values <= max_pm) | np.isnan(pm_values)
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_high_proper_motion] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_high_proper_motion", log_csv)
    return df_out


# =============================================================================
# BRIGHT STAR ARTIFACT FILTER (batch)
# =============================================================================

def query_nearby_bright_stars_batch(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    max_search_radius_deg: float = 1.0,
    bright_mag_limit: float = LTV_BRIGHT_MAG_LIMIT,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Batch query Gaia DR3 for nearby bright stars (g < bright_mag_limit).
    Uses TAP upload for efficient server-side crossmatch.
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for bright star query")
        return df
    
    df = df.copy()
    df["nearby_bright_dist_arcsec"] = np.nan
    df["nearby_bright_mag"] = np.nan
    
    valid_mask = df[ra_column].notna() & df[dec_column].notna()
    
    if not valid_mask.any():
        return df
    
    coords_df = pd.DataFrame({
        "_idx": df.index[valid_mask],
        "ra": df.loc[valid_mask, ra_column].values,
        "dec": df.loc[valid_mask, dec_column].values,
    })
    
    if verbose:
        print(f"Querying bright stars near {len(coords_df)} sources...")
    
    result = _batch_gaia_cone_query(
        coords_df,
        select_cols="g.phot_g_mean_mag",
        extra_where=f"WHERE g.phot_g_mean_mag < {bright_mag_limit}",
        match_radius_arcsec=max_search_radius_deg * 3600.0,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
    )
    
    if result.empty:
        return df
    
    # Keep only closest bright star per source
    result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
    
    for _, row in result.iterrows():
        idx = int(row["_idx"])
        if idx in df.index:
            df.loc[idx, "nearby_bright_dist_arcsec"] = row["sep_arcsec"]
            df.loc[idx, "nearby_bright_mag"] = row["phot_g_mean_mag"]
    
    if verbose:
        n_with_bright = df["nearby_bright_dist_arcsec"].notna().sum()
        print(f"[query_nearby_bright_stars_batch] {n_with_bright}/{len(df)} have nearby bright stars")
    
    return df


def bright_star_distance_curve(mag: np.ndarray) -> np.ndarray:
    """
    Empirical curve defining minimum acceptable distance from bright stars.

    Paper formula (PK25 / continued search):
      R > 0.2 * exp(12.5 - g) + 70, capped at 3600 arcsec.
    """
    mag = np.asarray(mag, dtype=float)
    dist = LTV_BRIGHT_STAR_COEFF * np.exp(LTV_BRIGHT_STAR_MAG_OFFSET - mag) + LTV_BRIGHT_STAR_BASE_ARCSEC
    return np.clip(dist, 0.0, LTV_BRIGHT_STAR_MAX_ARCSEC)


def filter_vg_overlap(
    df: pd.DataFrame,
    *,
    min_overlap_days: float = LTV_MIN_OVERLAP_DAYS,
    min_overlap_fraction: float = LTV_MIN_OVERLAP_FRACTION,
    has_v_col: str = "vg_has_v",
    overlap_days_col: str = "vg_overlap_days",
    overlap_frac_col: str = "vg_overlap_fraction",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove candidates with insufficient V/g temporal overlap.

    Keeps sources with:
      - no V-band data (no overlap requirement), OR
      - overlap_days >= min_overlap_days AND overlap_fraction >= min_overlap_fraction
    """
    n0 = len(df)

    if has_v_col not in df.columns or overlap_days_col not in df.columns or overlap_frac_col not in df.columns:
        if verbose:
            print("Warning: V/g overlap columns not found, skipping filter")
        return df

    has_v = df[has_v_col].fillna(False).astype(bool).values
    overlap_days = df[overlap_days_col].astype(float).values
    overlap_frac = df[overlap_frac_col].astype(float).values

    ok_overlap = (overlap_days >= min_overlap_days) & (overlap_frac >= min_overlap_fraction)
    mask = (~has_v) | ok_overlap

    df_out = df[mask].reset_index(drop=True)

    if verbose:
        print(f"[filter_vg_overlap] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")

    log_rejections(df, df_out, "filter_vg_overlap", log_csv)
    return df_out


def filter_bright_star_artifacts(
    df: pd.DataFrame,
    *,
    query_gaia: bool = True,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources that may be artifacts from nearby bright stars.
    Uses batch Gaia TAP queries for efficiency.
    """
    n0 = len(df)
    
    if "nearby_bright_dist_arcsec" not in df.columns and query_gaia:
        df = query_nearby_bright_stars_batch(
            df,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
        )
    
    if "nearby_bright_dist_arcsec" not in df.columns:
        if verbose:
            print("Warning: 'nearby_bright_dist_arcsec' not found, skipping filter")
        return df
    
    dist = df["nearby_bright_dist_arcsec"].values
    mag = df["nearby_bright_mag"].values
    min_dist = bright_star_distance_curve(mag)
    
    mask = (dist >= min_dist) | np.isnan(dist)
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_bright_star_artifacts] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_bright_star_artifacts", log_csv)
    return df_out


# =============================================================================
# CROWDING FILTER (batch Gaia TAP)
# =============================================================================

def query_crowding_batch(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    search_radius_arcsec: float = LTV_CROWDING_SEARCH_RADIUS_ARCSEC,
    target_mag_column: str = "Pstarss gmag",
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Batch query Gaia DR3 for source density (crowding metric).
    
    Counts number of Gaia sources within search_radius that are
    brighter than target + 3 mag (potential blending contaminants).
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for crowding query")
        return df
    
    df = df.copy()
    df["crowding_count"] = 0
    df["crowding_bright_count"] = 0
    
    valid_mask = df[ra_column].notna() & df[dec_column].notna()
    
    if not valid_mask.any():
        return df
    
    coords_df = pd.DataFrame({
        "_idx": df.index[valid_mask],
        "ra": df.loc[valid_mask, ra_column].values,
        "dec": df.loc[valid_mask, dec_column].values,
    })
    
    if target_mag_column in df.columns:
        coords_df["target_mag"] = df.loc[valid_mask, target_mag_column].values
    else:
        coords_df["target_mag"] = 14.0  # default
    
    if verbose:
        print(f"Querying crowding for {len(coords_df)} sources...")
    
    # Query all sources within radius
    result = _batch_gaia_cone_query(
        coords_df,
        select_cols="g.phot_g_mean_mag",
        match_radius_arcsec=search_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
    )
    
    if result.empty:
        return df
    
    # Count sources per target
    for idx in coords_df["_idx"].unique():
        matches = result[result["_idx"] == idx]
        if len(matches) > 0:
            df.loc[idx, "crowding_count"] = len(matches) - 1  # exclude self
            
            # Count bright contaminants
            target_mag = coords_df.loc[coords_df["_idx"] == idx, "target_mag"].values[0]
            bright_matches = matches[matches["phot_g_mean_mag"] < target_mag + 3]
            df.loc[idx, "crowding_bright_count"] = max(0, len(bright_matches) - 1)
    
    if verbose:
        mean_crowd = df["crowding_count"].mean()
        print(f"[query_crowding_batch] Mean crowding: {mean_crowd:.1f} sources within {search_radius_arcsec}\"")
    
    return df


def filter_crowding(
    df: pd.DataFrame,
    *,
    max_crowding_count: int = LTV_MAX_CROWDING_COUNT,
    query_gaia: bool = True,
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources in crowded fields where blending may cause artifacts.
    
    Uses batch Gaia TAP queries to count sources within 30 arcsec.
    """
    n0 = len(df)
    
    if "crowding_count" not in df.columns and query_gaia:
        df = query_crowding_batch(
            df,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
        )
    
    if "crowding_count" not in df.columns:
        if verbose:
            print("Warning: 'crowding_count' not found, skipping filter")
        return df
    
    mask = df["crowding_count"].values <= max_crowding_count
    df_out = df[mask].reset_index(drop=True)
    
    if verbose:
        print(f"[filter_crowding] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")
    
    log_rejections(df, df_out, "filter_crowding", log_csv)
    return df_out


# =============================================================================
# COMBINED FILTER PIPELINE
# =============================================================================

def apply_all_filters(
    df: pd.DataFrame,
    *,
    # Basic thresholds
    min_slope: float = LTV_MIN_SLOPE,
    min_diff: float = LTV_MIN_DIFF,
    min_dec: float = LTV_MIN_DEC,
    max_pm: float = LTV_MAX_PM,
    # Enhanced filter thresholds
    max_reduced_chi2: float = LTV_MAX_REDUCED_CHI2,
    max_single_jump_fraction: float = LTV_MAX_SINGLE_JUMP_FRACTION,
    max_eb_period_days: float = LTV_MAX_EB_PERIOD_DAYS,
    max_crowding_count: int = LTV_MAX_CROWDING_COUNT,
    # Options
    run_enhanced_filters: bool = True,
    query_gaia: bool = True,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply all paper filters in sequence.
    
    IMPORTANT: Vectorized filters run FIRST to reduce data size before
    expensive Gaia queries. This is critical for performance.
    
    Order:
    1. Slope threshold (vectorized, instant)
    2. Max diff threshold (vectorized, instant)
    3. South pole (vectorized, instant)
    4. Photometric scatter (vectorized) — NEW
    5. Transient contamination (vectorized) — NEW
    6. Eclipsing binary signature (vectorized) — NEW
    7. Bright star artifacts (batch Gaia TAP)
    8. High proper motion (batch Gaia TAP)
    9. Crowding (batch Gaia TAP) — NEW
    """
    n0 = len(df)
    
    if verbose:
        print(f"Starting with {n0} sources")
        print("Phase 1: Vectorized filters (instant)...")
    
    # Vectorized filters first — instant, reduces data size
    df = filter_slope_threshold(df, min_slope=min_slope, verbose=verbose, log_csv=log_csv)
    df = filter_max_diff_threshold(df, min_diff=min_diff, verbose=verbose, log_csv=log_csv)
    df = filter_south_pole(df, min_dec=min_dec, verbose=verbose, log_csv=log_csv)
    
    # Enhanced vectorized filters
    if run_enhanced_filters:
        if verbose:
            print("\nPhase 1b: Enhanced vectorized filters...")
        
        df = filter_photometric_scatter(
            df,
            max_reduced_chi2=max_reduced_chi2,
            verbose=verbose,
            log_csv=log_csv,
        )
        df = filter_vg_overlap(
            df,
            verbose=verbose,
            log_csv=log_csv,
        )
        df = filter_transient_contamination(
            df,
            max_single_jump_fraction=max_single_jump_fraction,
            verbose=verbose,
            log_csv=log_csv,
        )
        df = filter_eclipsing_binary_signature(
            df,
            max_eb_period_days=max_eb_period_days,
            verbose=verbose,
            log_csv=log_csv,
        )
    
    if verbose:
        print(f"\nAfter vectorized filters: {len(df)} sources ({len(df)/n0*100:.2f}% remaining)")
        print("Phase 2: Gaia TAP queries (batch)...")
    
    # Gaia queries only on reduced dataset
    df = filter_bright_star_artifacts(
        df,
        query_gaia=query_gaia,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        log_csv=log_csv,
    )
    df = filter_high_proper_motion(
        df,
        max_pm=max_pm,
        query_gaia=query_gaia,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        log_csv=log_csv,
    )
    
    # Enhanced crowding filter
    if run_enhanced_filters:
        df = filter_crowding(
            df,
            max_crowding_count=max_crowding_count,
            query_gaia=query_gaia,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
            log_csv=log_csv,
        )
    
    if verbose:
        print(f"\n[apply_all_filters] TOTAL: {n0} → {len(df)} ({len(df)/n0*100:.2f}% remaining)")
    
    return df
