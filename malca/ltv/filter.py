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
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config import (
    LTV_MIN_SLOPE,
    LTV_MIN_DIFF,
    LTV_MIN_DEC,
    LTV_MAX_PM,
    LTV_MAX_REDUCED_CHI2,
    LTV_MAX_CROWDING_COUNT,
    LTV_MATCH_RADIUS_ARCSEC,
    LTV_CHUNK_SIZE,
    LTV_GAIA_CHUNK_SIZE,
    LTV_CROSSMATCH_CHUNK_SIZE,
    LTV_WORKERS,
    LTV_CROWDING_SEARCH_RADIUS_ARCSEC,
    LTV_MAX_REFCAT_OFFSET,
    LTV_APERTURE_RADIUS_ARCSEC,
    LTV_ASASSN_BASELINE_YEARS,
    LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    LTV_NEIGHBOR_MIN_PM_MAS_YR,
)
from malca.utils import log_rejections, batch_gaia_cone_query


_SELF_MATCH_FALLBACK_SEP_ARCSEC = 1.0


# =============================================================================
# VECTORIZED THRESHOLD FILTERS (instant, no API calls)
# =============================================================================

def filter_slope_threshold(
    df: pd.DataFrame,
    *,
    min_slope: float = LTV_MIN_SLOPE,
    slope_column: str = "ltv_slope",
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
    diff_column: str = "ltv_max_diff",
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
    dec_column: str = "dec",
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
    slope_column: str = "ltv_slope",
    dispersion_column: str = "ltv_dispersion",
    median_err_column: str = "ltv_median_err",
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


def query_gaia_proper_motions_batch(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra",
    dec_column: str = "dec",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Batch query Gaia DR3 for proper motions.
    
    Uses TAP upload for efficient server-side crossmatch.
    Returns df with added columns: pmra, pmdec, pm_total, source_id,
    gaia_id, gaia_sep_arcsec.
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for Gaia PM query")
        return df
    
    df = df.copy()
    df["source_id"] = pd.Series(index=df.index, dtype="Int64")
    df["gaia_sep_arcsec"] = np.nan
    df["pmra"] = np.nan
    df["pmdec"] = np.nan
    df["pm_total"] = np.nan
    
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
    
    result = batch_gaia_cone_query(
        coords_df,
        select_cols="g.pmra, g.pmdec",
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        raise_on_failed_chunk=True,
    )
    
    if result.empty:
        return df
    
    # Keep only closest match per source
    result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
    
    # Merge back
    for _, row in result.iterrows():
        idx = int(row["_idx"])
        if idx in df.index:
            if pd.notna(row.get("source_id", np.nan)):
                df.loc[idx, "source_id"] = int(row["source_id"])
                if "gaia_id" not in df.columns:
                    df["gaia_id"] = pd.Series(index=df.index, dtype="object")
                df.loc[idx, "gaia_id"] = str(int(row["source_id"]))
            if pd.notna(row.get("sep_arcsec", np.nan)):
                df.loc[idx, "gaia_sep_arcsec"] = float(row["sep_arcsec"])
            df.loc[idx, "pmra"] = row["pmra"] if pd.notna(row["pmra"]) else np.nan
            df.loc[idx, "pmdec"] = row["pmdec"] if pd.notna(row["pmdec"]) else np.nan
            if pd.notna(row["pmra"]) and pd.notna(row["pmdec"]):
                df.loc[idx, "pm_total"] = np.sqrt(row["pmra"]**2 + row["pmdec"]**2)
    
    if verbose:
        n_matched = df["pm_total"].notna().sum()
        print(f"[query_gaia_proper_motions_batch] Matched {n_matched}/{len(df)}")
    
    return df


def filter_high_proper_motion(
    df: pd.DataFrame,
    *,
    max_pm: float = LTV_MAX_PM,
    pm_column: str = "pm_total",
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
# CROWDING FILTER (batch Gaia TAP)
# =============================================================================

def query_crowding_batch(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra",
    dec_column: str = "dec",
    search_radius_arcsec: float = LTV_CROWDING_SEARCH_RADIUS_ARCSEC,
    target_mag_column: str = "baseline_mag",
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
    result = batch_gaia_cone_query(
        coords_df,
        select_cols="g.phot_g_mean_mag",
        match_radius_arcsec=search_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        raise_on_failed_chunk=True,
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
    
    Uses batch Gaia TAP queries to count sources within the configured
    crowding radius.
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
# REFCAT MAGNITUDE OFFSET FILTER (vectorized)
# =============================================================================

def filter_refcat_offset(
    df: pd.DataFrame,
    *,
    max_refcat_offset: float = LTV_MAX_REFCAT_OFFSET,
    asas_mag_col: str = "ltv_median",
    refcat_mag_col: str = "baseline_mag",
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources where gASAS is more than max_refcat_offset mag brighter than gREFCAT.

    ASAS-SN uses a ~25" aperture; if a nearby bright star is blended in, the
    ASAS-SN median will be anomalously bright relative to the REFCAT/PanSTARRS
    single-star magnitude.  Keep condition: gASAS − gREFCAT ≥ −max_refcat_offset
    (i.e. ASAS-SN is not more than 1.5 mag brighter than expected).

    Sources without a REFCAT match (NaN diff) are kept.
    Vectorized — instant on any size.
    """
    n0 = len(df)

    if asas_mag_col not in df.columns or refcat_mag_col not in df.columns:
        if verbose:
            print(
                f"Warning: '{asas_mag_col}' or '{refcat_mag_col}' not found, "
                "skipping REFCAT offset filter"
            )
        return df

    asas_mag = df[asas_mag_col].values.astype(float)
    refcat_mag = df[refcat_mag_col].values.astype(float)
    diff = asas_mag - refcat_mag

    mask = (diff >= -max_refcat_offset) | np.isnan(diff)
    df_out = df[mask].reset_index(drop=True)

    if verbose:
        print(f"[filter_refcat_offset] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")

    log_rejections(df, df_out, "filter_refcat_offset", log_csv)
    return df_out


# =============================================================================
# NEIGHBOR HIGH-PM STAR FILTER (batch Gaia TAP)
# =============================================================================

def query_neighbor_high_pm_batch(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra",
    dec_column: str = "dec",
    target_mag_col: str = "baseline_mag",
    search_radius_arcsec: float = LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    aperture_radius_arcsec: float = LTV_APERTURE_RADIUS_ARCSEC,
    flux_ratio_limit: float = LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    min_pm_mas_yr: float = LTV_NEIGHBOR_MIN_PM_MAS_YR,
    t_start_year: float = -4.0,
    t_end_year: float = 6.0,
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Flag targets contaminated by a high-PM Gaia neighbor during the ASAS-SN baseline.

    For each target, queries all Gaia DR3 neighbors within search_radius_arcsec
    (large enough to capture stars that may walk into the aperture over the
    baseline).  For every neighbor with a Gaia proper-motion measurement:

      1. Skip if the neighbor's flux ratio F_nb/F_target < flux_ratio_limit
         (too faint to matter; corresponds to >5 mag fainter).
      2. Propagate the neighbor's position analytically over [t_start_year,
         t_end_year] relative to the Gaia DR3 reference epoch (J2016.0).
      3. If the minimum separation drops below aperture_radius_arcsec, mark
         the target as contaminated.

    Adds column: neighbor_pm_contam (bool).
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for neighbor PM query")
        df = df.copy()
        df["neighbor_pm_contam"] = False
        return df

    df = df.copy()
    df["neighbor_pm_contam"] = False

    valid_mask = df[ra_column].notna() & df[dec_column].notna()
    if not valid_mask.any():
        return df

    coords_df = pd.DataFrame({
        "_idx": df.index[valid_mask],
        "ra": df.loc[valid_mask, ra_column].values,
        "dec": df.loc[valid_mask, dec_column].values,
    })

    if verbose:
        print(f"Querying Gaia neighbors for {len(coords_df)} sources...")

    result = batch_gaia_cone_query(
        coords_df,
        select_cols="g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag",
        match_radius_arcsec=search_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        raise_on_failed_chunk=True,
    )

    if result.empty:
        return df

    mag_col = target_mag_col if target_mag_col in df.columns else None
    n_contam = 0

    for idx, neighbors in result.groupby("_idx"):
        if idx not in df.index:
            continue

        target_ra = float(df.loc[idx, ra_column])
        target_dec = float(df.loc[idx, dec_column])
        target_mag = float(df.loc[idx, mag_col]) if mag_col else 14.0
        cos_dec = np.cos(np.radians(target_dec))
        target_gaia_source_id = df.loc[idx, "source_id"] if "source_id" in df.columns else pd.NA

        for _, nb in neighbors.iterrows():
            nb_source_id = nb.get("source_id", np.nan)
            nb_sep_arcsec = nb.get("sep_arcsec", np.nan)

            if pd.notna(target_gaia_source_id) and pd.notna(nb_source_id):
                if int(nb_source_id) == int(target_gaia_source_id):
                    continue
            elif pd.notna(nb_sep_arcsec) and float(nb_sep_arcsec) <= _SELF_MATCH_FALLBACK_SEP_ARCSEC:
                continue

            nb_pmra = nb.get("pmra", np.nan)
            nb_pmdec = nb.get("pmdec", np.nan)
            nb_mag = nb.get("phot_g_mean_mag", np.nan)

            if pd.isna(nb_pmra) or pd.isna(nb_pmdec):
                continue

            # Skip neighbors with total PM below threshold (Gaia pmra/pmdec in mas/yr)
            pm_total_mas_yr = np.sqrt(float(nb_pmra) ** 2 + float(nb_pmdec) ** 2)
            if pm_total_mas_yr < min_pm_mas_yr:
                continue

            # Skip neighbors too faint to matter
            if not (pd.isna(nb_mag) or pd.isna(target_mag)):
                flux_ratio = 10.0 ** (-0.4 * (float(nb_mag) - float(target_mag)))
                if flux_ratio < flux_ratio_limit:
                    continue

            # Initial offset (arcsec) and PM (arcsec/yr)
            dx0 = (float(nb["ra"]) - target_ra) * cos_dec * 3600.0
            dy0 = (float(nb["dec"]) - target_dec) * 3600.0
            a = float(nb_pmra) / 1000.0
            b = float(nb_pmdec) / 1000.0

            # Minimum separation over [t_start_year, t_end_year]
            denom = a * a + b * b
            if denom < 1e-12:
                min_sep = np.sqrt(dx0 * dx0 + dy0 * dy0)
            else:
                t_closest = -(a * dx0 + b * dy0) / denom
                t_clamped = float(np.clip(t_closest, t_start_year, t_end_year))
                seps = []
                for t in (t_start_year, t_clamped, t_end_year):
                    seps.append(np.sqrt((dx0 + a * t) ** 2 + (dy0 + b * t) ** 2))
                min_sep = min(seps)

            if min_sep < aperture_radius_arcsec:
                df.loc[idx, "neighbor_pm_contam"] = True
                n_contam += 1
                break  # one contaminating neighbor is enough

    if verbose:
        print(
            f"[query_neighbor_high_pm_batch] "
            f"{n_contam} targets flagged as contaminated by a passing high-PM neighbor"
        )

    return df


def filter_neighbor_high_pm(
    df: pd.DataFrame,
    *,
    search_radius_arcsec: float = LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    aperture_radius_arcsec: float = LTV_APERTURE_RADIUS_ARCSEC,
    flux_ratio_limit: float = LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    min_pm_mas_yr: float = LTV_NEIGHBOR_MIN_PM_MAS_YR,
    query_gaia: bool = True,
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove targets contaminated by a passing high-PM Gaia neighbor.

    Uses batch Gaia TAP queries plus an analytic trajectory check (paper §2).
    If 'neighbor_pm_contam' is already present in df it is used directly.
    """
    n0 = len(df)

    if "neighbor_pm_contam" not in df.columns and query_gaia:
        df = query_neighbor_high_pm_batch(
            df,
            search_radius_arcsec=search_radius_arcsec,
            aperture_radius_arcsec=aperture_radius_arcsec,
            flux_ratio_limit=flux_ratio_limit,
            min_pm_mas_yr=min_pm_mas_yr,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
        )

    if "neighbor_pm_contam" not in df.columns:
        if verbose:
            print("Warning: 'neighbor_pm_contam' not found, skipping filter")
        return df

    mask = ~df["neighbor_pm_contam"].fillna(False).astype(bool)
    df_out = df[mask].reset_index(drop=True)

    if verbose:
        print(f"[filter_neighbor_high_pm] {n0} → {len(df_out)} (removed {n0 - len(df_out)})")

    log_rejections(df, df_out, "filter_neighbor_high_pm", log_csv)
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
    # Paper filter thresholds
    max_refcat_offset: float = LTV_MAX_REFCAT_OFFSET,
    # Enhanced filter thresholds
    max_reduced_chi2: float = LTV_MAX_REDUCED_CHI2,
    max_crowding_count: int = LTV_MAX_CROWDING_COUNT,
    # Neighbor PM filter
    aperture_radius_arcsec: float = LTV_APERTURE_RADIUS_ARCSEC,
    neighbor_flux_ratio_limit: float = LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    neighbor_search_radius_arcsec: float = LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    neighbor_min_pm_mas_yr: float = LTV_NEIGHBOR_MIN_PM_MAS_YR,
    # Options
    run_enhanced_filters: bool = True,
    run_neighbor_pm_filter: bool = True,
    query_gaia: bool = True,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    log_csv: str | Path | None = None,
    return_rejected: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all paper filters in sequence.

    IMPORTANT: Vectorized filters run FIRST to reduce data size before
    expensive Gaia queries. This is critical for performance.

    Order:
    1. Slope threshold (vectorized)
    2. Max diff threshold (vectorized)
    3. South pole (vectorized)
    4. REFCAT magnitude offset (vectorized) — paper Fig. 1
    5. Photometric scatter (vectorized)
    6. Bright star artifacts (batch Gaia TAP)
    7. High proper motion (batch Gaia TAP)
    8. Neighbor high-PM contamination (batch Gaia TAP) — paper §2
    9. Crowding (batch Gaia TAP)
    """
    n0 = len(df)

    if verbose:
        print(f"Starting with {n0} sources")
        print("Phase 1: Vectorized filters (instant)...")

    # Vectorized filters first — instant, reduces data size
    df = filter_slope_threshold(df, min_slope=min_slope, verbose=verbose, log_csv=log_csv)
    df = filter_max_diff_threshold(df, min_diff=min_diff, verbose=verbose, log_csv=log_csv)

    rejected_list = []

    def run_filter(func, df_in, name, **kwargs):
        df_out = func(df_in, **kwargs)
        if return_rejected:
            id_col = None
            for candidate in ["candidate_id", "asas_sn_id", "lc_path", "source_id"]:
                if candidate in df_in.columns:
                    id_col = candidate
                    break
            if id_col is None:
                id_col = df_in.columns[0]
            
            before_ids = set(df_in[id_col].astype(str))
            after_ids = set(df_out[id_col].astype(str))
            rejected_ids = before_ids - after_ids
            
            if rejected_ids:
                rejected = df_in[df_in[id_col].astype(str).isin(rejected_ids)].copy()
                rejected["filter_reason"] = name
                rejected_list.append(rejected)
        return df_out

    df = run_filter(filter_south_pole, df, "south_pole", min_dec=min_dec, verbose=verbose, log_csv=log_csv)
    df = run_filter(filter_refcat_offset, df, "refcat_offset", max_refcat_offset=max_refcat_offset, verbose=verbose, log_csv=log_csv)

    # Enhanced vectorized filters
    if run_enhanced_filters:
        if verbose:
            print("\nPhase 1b: Enhanced vectorized filters...")

        df = run_filter(filter_photometric_scatter, df, "photometric_scatter", max_reduced_chi2=max_reduced_chi2, verbose=verbose, log_csv=log_csv)

    if verbose:
        print(f"\nAfter vectorized filters: {len(df)} sources ({len(df)/n0*100:.2f}% remaining)")
        print("Phase 2: Gaia TAP queries (batch)...")

    # Gaia queries only on reduced dataset
    df = run_filter(filter_high_proper_motion, df, "high_proper_motion", max_pm=max_pm, query_gaia=query_gaia, chunk_size=chunk_size, n_workers=n_workers, verbose=verbose, log_csv=log_csv)

    # Neighbor high-PM contamination filter (paper §2)
    if run_neighbor_pm_filter:
        df = run_filter(filter_neighbor_high_pm, df, "neighbor_high_pm", search_radius_arcsec=neighbor_search_radius_arcsec, aperture_radius_arcsec=aperture_radius_arcsec, flux_ratio_limit=neighbor_flux_ratio_limit, min_pm_mas_yr=neighbor_min_pm_mas_yr, query_gaia=query_gaia, chunk_size=chunk_size, n_workers=n_workers, verbose=verbose, log_csv=log_csv)

    # Enhanced crowding filter
    if run_enhanced_filters:
        df = run_filter(filter_crowding, df, "crowding", max_crowding_count=max_crowding_count, query_gaia=query_gaia, chunk_size=chunk_size, n_workers=n_workers, verbose=verbose, log_csv=log_csv)

    if verbose:
        print(f"\n[apply_all_filters] TOTAL: {n0} → {len(df)} ({len(df)/n0*100:.2f}% remaining)")

    if return_rejected:
        if rejected_list:
            df_rejected = pd.concat(rejected_list, ignore_index=True)
        else:
            df_rejected = pd.DataFrame(columns=df.columns.tolist() + ["filter_reason"])
        return df, df_rejected

    return df


LTV_AUDIT_FAILED_COLUMNS = {
    "slope": "ltv_failed_slope",
    "max_diff": "ltv_failed_max_diff",
    "dec": "ltv_failed_dec",
    "refcat_offset": "ltv_failed_refcat_offset",
    "photometric_scatter": "ltv_failed_photometric_scatter",
    "high_pm": "ltv_failed_high_pm",
    "neighbor_high_pm": "ltv_failed_neighbor_high_pm",
    "crowding": "ltv_failed_crowding",
}


def _false_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index, dtype=bool)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _merge_audit_annotations(
    audit: pd.DataFrame,
    annotated: pd.DataFrame,
    audit_id_col: str,
) -> pd.DataFrame:
    if annotated.empty or audit_id_col not in annotated.columns:
        return audit

    updates = annotated.drop_duplicates(subset=[audit_id_col], keep="first").set_index(audit_id_col)
    ids = audit[audit_id_col]
    matched = ids.isin(updates.index)
    if not bool(matched.any()):
        return audit

    for col in updates.columns:
        if col == audit_id_col:
            continue
        if col not in audit.columns:
            audit[col] = pd.NA
        audit.loc[matched, col] = ids.loc[matched].map(updates[col]).to_numpy()
    return audit


def _first_filter_reason(row: pd.Series) -> str | None:
    for reason, col in LTV_AUDIT_FAILED_COLUMNS.items():
        if bool(row.get(col, False)):
            return reason
    return None


def apply_all_filters_audit(
    df: pd.DataFrame,
    *,
    min_slope: float = LTV_MIN_SLOPE,
    min_diff: float = LTV_MIN_DIFF,
    min_dec: float = LTV_MIN_DEC,
    max_pm: float = LTV_MAX_PM,
    max_refcat_offset: float = LTV_MAX_REFCAT_OFFSET,
    max_reduced_chi2: float = LTV_MAX_REDUCED_CHI2,
    max_crowding_count: int = LTV_MAX_CROWDING_COUNT,
    aperture_radius_arcsec: float = LTV_APERTURE_RADIUS_ARCSEC,
    neighbor_flux_ratio_limit: float = LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    neighbor_search_radius_arcsec: float = LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    neighbor_min_pm_mas_yr: float = LTV_NEIGHBOR_MIN_PM_MAS_YR,
    run_enhanced_filters: bool = True,
    run_neighbor_pm_filter: bool = True,
    query_gaia: bool = True,
    chunk_size: int = LTV_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    return_passers: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Apply LTV filters with STV-style audit semantics.

    Unlike ``apply_all_filters``, this preserves every input row and annotates
    failures with ``ltv_failed_*`` plus ``failed_any``. Expensive Gaia-backed
    checks run only on rows still passing earlier filters, then their annotation
    columns are merged back into the full audit table.
    """
    audit = df.copy().reset_index(drop=True)
    audit_id_col = "_ltv_audit_id"
    audit[audit_id_col] = np.arange(len(audit), dtype=np.int64)

    for col in LTV_AUDIT_FAILED_COLUMNS.values():
        audit[col] = False

    passing = pd.Series(True, index=audit.index, dtype=bool)

    def mark_failures(label: str, fail_mask: pd.Series) -> None:
        nonlocal passing, audit
        fail_col = LTV_AUDIT_FAILED_COLUMNS[label]
        fail_mask = fail_mask.reindex(audit.index, fill_value=False).fillna(False).astype(bool)
        fail_mask &= passing
        audit[fail_col] = fail_mask
        passing &= ~fail_mask
        if verbose:
            print(f"[ltv audit filter:{label}] failed {int(fail_mask.sum())}/{len(audit)}")

    required = ("ltv_slope", "ltv_max_diff", "dec", "ltv_median", "baseline_mag")
    missing = [col for col in required if col not in audit.columns]
    if missing:
        raise ValueError(f"LTV candidate products missing canonical filter columns: {', '.join(missing)}")

    mark_failures("slope", _numeric_series(audit, "ltv_slope").abs() <= float(min_slope))
    mark_failures("max_diff", _numeric_series(audit, "ltv_max_diff").abs() <= float(min_diff))
    mark_failures("dec", _numeric_series(audit, "dec") < float(min_dec))

    diff = _numeric_series(audit, "ltv_median") - _numeric_series(audit, "baseline_mag")
    mark_failures("refcat_offset", (diff < -float(max_refcat_offset)) & diff.notna())

    if run_enhanced_filters and "ltv_dispersion" in audit.columns:
        dispersion = _numeric_series(audit, "ltv_dispersion")
        if "ltv_median_err" in audit.columns:
            err = _numeric_series(audit, "ltv_median_err").clip(lower=0.01)
            chi2 = (dispersion / err) ** 2
        else:
            chi2 = (dispersion / 0.02) ** 2
        slope = _numeric_series(audit, "ltv_slope").abs()
        fail = (chi2 > float(max_reduced_chi2)) & (slope < 0.05)
        mark_failures("photometric_scatter", fail.fillna(False))
    else:
        mark_failures("photometric_scatter", _false_mask(audit))
        if verbose and run_enhanced_filters:
            print("Warning: 'ltv_dispersion' not found, skipping photometric scatter filter")

    # Gaia-backed filters run on current passers only.
    eligible = passing.copy()
    if bool(eligible.any()) and query_gaia:
        subset = audit.loc[eligible].copy()
        if "pm_total" not in subset.columns:
            subset = query_gaia_proper_motions_batch(
                subset,
                chunk_size=chunk_size,
                n_workers=n_workers,
                verbose=verbose,
            )
        audit = _merge_audit_annotations(audit, subset, audit_id_col)
    if "pm_total" in audit.columns:
        mark_failures("high_pm", _numeric_series(audit, "pm_total") > float(max_pm))
    else:
        mark_failures("high_pm", _false_mask(audit))
        if verbose and query_gaia:
            print("Warning: 'pm_total' column not found, skipping high-PM filter")

    if run_neighbor_pm_filter:
        eligible = passing.copy()
        if bool(eligible.any()) and query_gaia:
            subset = audit.loc[eligible].copy()
            if "neighbor_pm_contam" not in subset.columns:
                subset = query_neighbor_high_pm_batch(
                    subset,
                    search_radius_arcsec=neighbor_search_radius_arcsec,
                    aperture_radius_arcsec=aperture_radius_arcsec,
                    flux_ratio_limit=neighbor_flux_ratio_limit,
                    min_pm_mas_yr=neighbor_min_pm_mas_yr,
                    chunk_size=chunk_size,
                    n_workers=n_workers,
                    verbose=verbose,
                )
            audit = _merge_audit_annotations(audit, subset, audit_id_col)
        if "neighbor_pm_contam" in audit.columns:
            mark_failures("neighbor_high_pm", audit["neighbor_pm_contam"].fillna(False).astype(bool))
        else:
            mark_failures("neighbor_high_pm", _false_mask(audit))
    else:
        mark_failures("neighbor_high_pm", _false_mask(audit))

    if run_enhanced_filters:
        eligible = passing.copy()
        if bool(eligible.any()) and query_gaia:
            subset = audit.loc[eligible].copy()
            if "crowding_count" not in subset.columns:
                subset = query_crowding_batch(
                    subset,
                    chunk_size=chunk_size,
                    n_workers=n_workers,
                    verbose=verbose,
                )
            audit = _merge_audit_annotations(audit, subset, audit_id_col)
        if "crowding_count" in audit.columns:
            mark_failures("crowding", _numeric_series(audit, "crowding_count") > int(max_crowding_count))
        else:
            mark_failures("crowding", _false_mask(audit))
    else:
        mark_failures("crowding", _false_mask(audit))

    fail_cols = list(LTV_AUDIT_FAILED_COLUMNS.values())
    audit["failed_any"] = audit[fail_cols].fillna(False).astype(bool).any(axis=1)
    audit["filter_reason"] = audit.apply(_first_filter_reason, axis=1)
    audit = audit.drop(columns=[audit_id_col])

    if verbose:
        n_pass = int((~audit["failed_any"]).sum())
        pct = (n_pass / len(audit) * 100.0) if len(audit) else 0.0
        print(f"[ltv audit filter] passed {n_pass}/{len(audit)} ({pct:.2f}%)")

    if return_passers:
        return audit, audit.loc[~audit["failed_any"]].reset_index(drop=True).copy()
    return audit
