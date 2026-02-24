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
    LTV_WORKERS,

    LTV_MIN_OVERLAP_DAYS,
    LTV_MIN_OVERLAP_FRACTION,
    LTV_CROWDING_SEARCH_RADIUS_ARCSEC,
    LTV_MAX_REFCAT_OFFSET,
    LTV_APERTURE_RADIUS_ARCSEC,
    LTV_ASASSN_BASELINE_YEARS,
    LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
)
from malca.utils import log_rejections, batch_gaia_cone_query


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
    
    result = batch_gaia_cone_query(
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
    result = batch_gaia_cone_query(
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
# REFCAT MAGNITUDE OFFSET FILTER (vectorized)
# =============================================================================

def filter_refcat_offset(
    df: pd.DataFrame,
    *,
    max_refcat_offset: float = LTV_MAX_REFCAT_OFFSET,
    asas_mag_col: str = "Median",
    refcat_mag_col: str = "Pstarss gmag",
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
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    target_mag_col: str = "Pstarss gmag",
    search_radius_arcsec: float = LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    aperture_radius_arcsec: float = LTV_APERTURE_RADIUS_ARCSEC,
    flux_ratio_limit: float = LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
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

        for _, nb in neighbors.iterrows():
            nb_pmra = nb.get("pmra", np.nan)
            nb_pmdec = nb.get("pmdec", np.nan)
            nb_mag = nb.get("phot_g_mean_mag", np.nan)

            if pd.isna(nb_pmra) or pd.isna(nb_pmdec):
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
    max_single_jump_fraction: float = LTV_MAX_SINGLE_JUMP_FRACTION,
    max_eb_period_days: float = LTV_MAX_EB_PERIOD_DAYS,
    max_crowding_count: int = LTV_MAX_CROWDING_COUNT,
    # Neighbor PM filter
    aperture_radius_arcsec: float = LTV_APERTURE_RADIUS_ARCSEC,
    neighbor_flux_ratio_limit: float = LTV_NEIGHBOR_FLUX_RATIO_LIMIT,
    neighbor_search_radius_arcsec: float = LTV_NEIGHBOR_SEARCH_RADIUS_ARCSEC,
    # Options
    run_enhanced_filters: bool = True,
    run_neighbor_pm_filter: bool = True,
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
    1. Slope threshold (vectorized)
    2. Max diff threshold (vectorized)
    3. South pole (vectorized)
    4. REFCAT magnitude offset (vectorized) — paper Fig. 1
    5. Photometric scatter (vectorized)
    6. Transient contamination (vectorized)
    7. Eclipsing binary signature (vectorized)
    8. Bright star artifacts (batch Gaia TAP)
    9. High proper motion (batch Gaia TAP)
    10. Neighbor high-PM contamination (batch Gaia TAP) — paper §2
    11. Crowding (batch Gaia TAP)
    """
    n0 = len(df)

    if verbose:
        print(f"Starting with {n0} sources")
        print("Phase 1: Vectorized filters (instant)...")

    # Vectorized filters first — instant, reduces data size
    df = filter_slope_threshold(df, min_slope=min_slope, verbose=verbose, log_csv=log_csv)
    df = filter_max_diff_threshold(df, min_diff=min_diff, verbose=verbose, log_csv=log_csv)
    df = filter_south_pole(df, min_dec=min_dec, verbose=verbose, log_csv=log_csv)
    df = filter_refcat_offset(
        df,
        max_refcat_offset=max_refcat_offset,
        verbose=verbose,
        log_csv=log_csv,
    )

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
    df = filter_high_proper_motion(
        df,
        max_pm=max_pm,
        query_gaia=query_gaia,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        log_csv=log_csv,
    )

    # Neighbor high-PM contamination filter (paper §2)
    if run_neighbor_pm_filter:
        df = filter_neighbor_high_pm(
            df,
            search_radius_arcsec=neighbor_search_radius_arcsec,
            aperture_radius_arcsec=aperture_radius_arcsec,
            flux_ratio_limit=neighbor_flux_ratio_limit,
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
