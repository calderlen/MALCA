"""
Dipper Classification Module

Implements classification scenarios from Tzanidakis et al. (2025):
- Eclipsing Binary (EB) rejection
- Cataclysmic Variable (CV) rejection  
- Starspot rejection
- YSO classification (Koenig & Leisawitz 2014)
- Circumstellar material estimation
- Disk occultation probability

Usage:
    malca classify --input output/events.parquet --output output/classified.parquet
"""
from pathlib import Path
import argparse
import json
import os

from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from tqdm import tqdm
import astropy.units as u
import numpy as np
import pandas as pd

from malca.config import (
    SOLAR_MASS_KG, SOLAR_RADIUS_M, AU_M, DAY_S,
    GRAVITATIONAL_CONSTANT_SI, EARTH_MASS_KG,
    EB_SHORT_DIP_DAYS, EB_LONG_DIP_DAYS, EB_SHORT_P, EB_LONG_P, EB_VERY_LONG_P,
    EB_PERIODIC_BONUS, EB_SYMMETRIC_BONUS, EB_BINARY_BONUS, EB_ASYMMETRY_THRESHOLD,
    CV_BP_RP_THRESHOLD, CV_G_ABS_THRESHOLD, CV_BASE_P,
    CV_HA_EW_THRESHOLD, CV_HA_BONUS, CV_KNOWN_P,
    STARSPOT_SMALL_AMP, STARSPOT_MEDIUM_AMP, STARSPOT_SMALL_P,
    STARSPOT_MEDIUM_P, STARSPOT_LARGE_P, STARSPOT_ROTATION_PERIOD_DAYS,
    YSO_CLASS_I_W1W2, YSO_CLASS_II_W1W2_MIN, YSO_CLASS_II_HK,
    YSO_DUST_CORRECTION_HK, YSO_DUST_CORRECTION_W1W2,
    DISK_BASE_P, DISK_LARGE_A_AU, DISK_VERY_LARGE_A_AU,
    DISK_LARGE_A_P, DISK_VERY_LARGE_A_P, DISK_LARGE_HILL_BONUS,
    DISK_NO_IR_EXCESS_BONUS, DISK_P_CAP,
    CLASSIFY_EB_THRESHOLD, CLASSIFY_CV_THRESHOLD, CLASSIFY_STARSPOT_THRESHOLD,
    CLASSIFY_DISK_THRESHOLD, CLASSIFY_MS_EB_REJECTION, CLASSIFY_MS_CV_REJECTION,
    CLASSIFY_IPHAS_RADIUS_ARCSEC, CLASSIFY_PS1_RADIUS_ARCSEC,
)
from malca.products.candidates import select_passing_candidates_if_present
from malca.io.table_io import read_feature_table, write_feature_table
from malca.config import VIZIER_TAP_URL
from malca.core.utils import batch_tap_crossmatch


CLASSIFIER_VERSION = "heuristic-v2"

# These aliases are deliberately ordered.  The first finite value wins on each
# row, and the selected column is recorded in the output so a score can be
# traced back to the actual pipeline field that supplied it.
CLASSIFIER_INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "duration_days": ("event_duration_days", "timescale_days", "duration", "duration_days"),
    "depth_mag": ("event_depth_mag", "dip_depth_mag"),
    "depth_fraction": ("max_depth",),
    "mass_solar": ("mass50", "mass_gspphot", "stellar_mass"),
    "radius_solar": ("radius", "radius_gspphot", "stellar_radius"),
    "g_mag": ("phot_g_mean_mag", "gaia_g"),
    "bp_rp": ("bp_rp",),
    "distance_pc": ("distance_gspphot", "distance_pc"),
    "h_mag": ("tmass_h", "Hmag", "h_m"),
    "k_mag": ("tmass_k", "Kmag", "k_m", "ks_m"),
    "w1_mag": ("w1", "W1mag", "w1mpro"),
    "w2_mag": ("w2", "W2mag", "w2mpro"),
}


def _coalesce_numeric(df: pd.DataFrame, aliases: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series("", index=df.index, dtype=object)
    for col in aliases:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        take = values.isna() & candidate.notna() & np.isfinite(candidate)
        values.loc[take] = candidate.loc[take]
        source.loc[take] = col
    return values, source


def _record_input_source(df: pd.DataFrame, key: str, source: pd.Series) -> None:
    df[f"classification_{key}_source"] = source.fillna("").astype(str)


def _append_note(df: pd.DataFrame, mask: pd.Series, column: str, note: str) -> None:
    if bool(mask.any()):
        df.loc[mask, column] = df.loc[mask, column].fillna("").astype(str) + note


def _finite_score(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").where(lambda value: np.isfinite(value))


def _coerce_bool_evidence(series: pd.Series) -> pd.Series:
    """Parse catalog/pipeline flags without treating the string ``False`` as true."""
    out = pd.Series(False, index=series.index, dtype=bool)
    text = series.astype("string").str.strip().str.lower()
    out.loc[text.isin({"true", "t", "yes", "y", "1"})] = True
    numeric = pd.to_numeric(series, errors="coerce")
    out.loc[numeric.notna()] = numeric.loc[numeric.notna()].ne(0)
    return out






# =============================================================================
# ECLIPSING BINARY REJECTION
# =============================================================================

def check_eb_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test for eclipsing binary contamination.
    
    Checks:
    1. Light curve asymmetry (EBs are symmetric)
    2. Dip duration vs Keplerian expectations
    3. Transit probability at required separation
    
    Returns df with columns: P_eb, eb_notes
    """
    df = df.copy()
    df['P_eb'] = np.nan
    df['eb_notes'] = ''
    duration_days, duration_source = _coalesce_numeric(
        df, CLASSIFIER_INPUT_ALIASES["duration_days"]
    )
    _record_input_source(df, "duration", duration_source)
    valid_duration = duration_days.notna() & (duration_days > 0)
    df["eb_classifier_status"] = np.where(valid_duration, "ok", "missing_duration")
    _append_note(df, ~valid_duration, "eb_notes", "No valid duration data; ")
    
    # For EB with semimajor axis ~1.8 AU (to explain single eclipse in 2.5yr baseline)
    # Eclipse duration ~ 1.5 days for tangential velocity of 21 km/s
    # If observed duration >> 1.5 days, EB is unlikely
    
    # Expected EB duration for a = 1.8 AU
    # Retain the physically motivated scale in the documentation, but do not
    # silently invent a stellar radius just to manufacture a score.
    
    # Dips lasting weeks-months require 10-10000 AU separations
    # Transit probability at such separations: 10^-4 to 10^-7
    
    # Simple heuristic: if duration > EB_SHORT_DIP_DAYS, unlikely to be EB
    long_dip = valid_duration & (duration_days > EB_SHORT_DIP_DAYS)
    very_long_dip = valid_duration & (duration_days > EB_LONG_DIP_DAYS)

    # Assign probabilities
    df.loc[valid_duration & ~long_dip, 'P_eb'] = EB_SHORT_P  # Short dips could be EBs
    df.loc[long_dip, 'P_eb'] = EB_LONG_P  # Long dips unlikely EBs
    df.loc[very_long_dip, 'P_eb'] = EB_VERY_LONG_P  # Very long dips very unlikely EBs
    
    # Check for periodicity if available
    if 'is_periodic' in df.columns:
        periodic = _coerce_bool_evidence(df['is_periodic'])
        base = df.loc[periodic, 'P_eb'].fillna(0.0)
        df.loc[periodic, 'P_eb'] = np.minimum(base + EB_PERIODIC_BONUS, 1.0)
        _append_note(df, periodic, 'eb_notes', 'Periodic; ')
        df.loc[periodic & ~valid_duration, "eb_classifier_status"] = "partial_periodicity_only"
    
    # Check for symmetry if available
    if 'asymmetry' in df.columns:
        asymmetry = pd.to_numeric(df['asymmetry'], errors='coerce')
        symmetric = asymmetry.notna() & (np.abs(asymmetry) < EB_ASYMMETRY_THRESHOLD)
        base = df.loc[symmetric, 'P_eb'].fillna(0.0)
        df.loc[symmetric, 'P_eb'] = np.minimum(base + EB_SYMMETRIC_BONUS, 1.0)
        _append_note(df, symmetric, 'eb_notes', 'Symmetric; ')
        df.loc[symmetric & ~valid_duration, "eb_classifier_status"] = "partial_symmetry_only"
    
    # Gaia binary flag
    if 'non_single_star' in df.columns:
        binary = pd.to_numeric(df['non_single_star'], errors='coerce').fillna(0) > 0
        base = df.loc[binary, 'P_eb'].fillna(0.0)
        df.loc[binary, 'P_eb'] = np.minimum(base + EB_BINARY_BONUS, 1.0)
        _append_note(df, binary, 'eb_notes', 'Gaia binary; ')
        df.loc[binary & ~valid_duration, "eb_classifier_status"] = "partial_binary_only"
    
    return df


# =============================================================================
# EXTERNAL CATALOG QUERIES (IPHAS, PS1)
# =============================================================================

def _float_or_nan(value: object) -> float:
    if value is None or np.ma.is_masked(value):
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _row_first_numeric(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row.index:
            value = _float_or_nan(row.get(name))
            if np.isfinite(value):
                return value
    return np.nan


def query_iphas_by_coords(df: pd.DataFrame, radius_arcsec: float = CLASSIFY_IPHAS_RADIUS_ARCSEC) -> pd.DataFrame:
    """
    Query IPHAS DR2 for Hα photometry using batch VizieR TAP upload.
    
    IPHAS covers Northern Galactic Plane: -5° < b < 5°, 30° < l < 215°
    Returns r, i, Hα magnitudes and r-Hα color
    """



    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: No ra/dec for IPHAS query")
        return df
    
    df = df.copy()
    df['iphas_r'] = np.nan
    df['iphas_i'] = np.nan
    df['iphas_ha'] = np.nan
    df['r_ha'] = np.nan
    df['r_i'] = np.nan
    df['ha_ew'] = np.nan
    df['iphas_sep_arcsec'] = np.nan
    df['iphas_match_status'] = 'not_queried'

    valid = df['ra'].notna() & df['dec'].notna()
    df.loc[valid, 'iphas_match_status'] = 'no_match'
    df.loc[~valid, 'iphas_match_status'] = 'missing_coordinates'
    if not valid.any():
        return df

    print(f"Querying IPHAS for {int(valid.sum())} sources via TAP...")

    coords_df = pd.DataFrame({
        "_idx": df.index[valid],
        "ra": df.loc[valid, "ra"].values,
        "dec": df.loc[valid, "dec"].values,
    })

    result = batch_tap_crossmatch(
        coords_df,
        tap_url=VIZIER_TAP_URL,
        catalog_table='"II/321/iphas2"',
        select_cols='c.r, c.i, c.ha, c.rmi, c.rmha',
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        match_radius_arcsec=radius_arcsec,
        chunk_size=1000,
        n_workers=4,
        verbose=True,
        desc="IPHAS TAP",
    )

    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                r = _row_first_numeric(row, "r", "rmag")
                i = _row_first_numeric(row, "i", "imag")
                ha = _row_first_numeric(row, "ha", "Ha", "Hamag")
                r_i = _row_first_numeric(row, "rmi", "r-i")
                r_ha = _row_first_numeric(row, "rmha", "r-ha", "r-Ha")
                if not np.isfinite(r_i) and np.isfinite(r) and np.isfinite(i):
                    r_i = r - i
                if not np.isfinite(r_ha) and np.isfinite(r) and np.isfinite(ha):
                    r_ha = r - ha
                df.loc[idx, 'iphas_r'] = r
                df.loc[idx, 'iphas_i'] = i
                df.loc[idx, 'iphas_ha'] = ha
                df.loc[idx, 'r_i'] = r_i
                df.loc[idx, 'r_ha'] = r_ha
                df.loc[idx, 'iphas_sep_arcsec'] = _row_first_numeric(row, 'sep_arcsec')
                df.loc[idx, 'iphas_match_status'] = 'matched'

    n_found = df['iphas_r'].notna().sum()
    print(f"Found IPHAS photometry for {n_found}/{len(df)} sources")
    
    return df


def query_ps1_by_coords(df: pd.DataFrame, radius_arcsec: float = CLASSIFY_PS1_RADIUS_ARCSEC) -> pd.DataFrame:
    """
    Query Pan-STARRS1 for grizy photometry using MAST.
    
    PS1 covers 3π sky (dec > -30°)
    Returns g, r, i, z, y PSF magnitudes
    """
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: No ra/dec for PS1 query")
        return df
    
    df = df.copy()
    for band in ['g', 'r', 'i', 'z', 'y']:
        df[f'ps1_{band}'] = np.nan
    df['ps1_sep_arcsec'] = np.nan
    df['ps1_match_status'] = 'not_queried'

    print(f"Querying PS1 for {len(df)} sources...")
    
    for idx in tqdm(df.index, desc="PS1"):
        ra, dec = df.loc[idx, 'ra'], df.loc[idx, 'dec']
        if pd.isna(ra) or pd.isna(dec):
            df.loc[idx, 'ps1_match_status'] = 'missing_coordinates'
            continue
        df.loc[idx, 'ps1_match_status'] = 'no_match'
            
        try:
            result = Catalogs.query_region(
                f"{ra} {dec}",
                radius=radius_arcsec/3600,  # degrees
                catalog="Panstarrs",
                table="mean"
            )
            
            if result is not None and len(result) > 0:
                target = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
                ra_col = next((name for name in ('raMean', 'raStack', 'ra') if name in result.colnames), None)
                dec_col = next((name for name in ('decMean', 'decStack', 'dec') if name in result.colnames), None)
                if ra_col and dec_col:
                    catalog_coords = SkyCoord(
                        np.asarray(result[ra_col], dtype=float) * u.deg,
                        np.asarray(result[dec_col], dtype=float) * u.deg,
                    )
                    separations = target.separation(catalog_coords).arcsec
                    best = int(np.nanargmin(separations))
                    sep_arcsec = float(separations[best])
                else:
                    best = 0
                    sep_arcsec = np.nan
                row = result[best]
                for band in ['g', 'r', 'i', 'z', 'y']:
                    col = f'{band}MeanPSFMag'
                    if col in row.colnames:
                        df.loc[idx, f'ps1_{band}'] = row[col]
                df.loc[idx, 'ps1_sep_arcsec'] = sep_arcsec
                df.loc[idx, 'ps1_match_status'] = 'matched'
        except Exception:
            df.loc[idx, 'ps1_match_status'] = 'error'
            continue
    
    n_found = df['ps1_g'].notna().sum()
    print(f"Found PS1 photometry for {n_found}/{len(df)} sources")
    
    return df


# =============================================================================
# CATACLYSMIC VARIABLE REJECTION
# =============================================================================

def check_cv_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test for CV contamination using color-color locus.
    
    Checks:
    1. PS1 grizy colors vs main sequence locus
    2. Hα excess (if IPHAS data available)
    3. Gaia CMD position (CVs between MS and WD cooling sequence)
    
    Returns df with columns: P_cv, cv_notes
    """
    df = df.copy()
    df['P_cv'] = np.nan
    df['cv_notes'] = ''
    g_mag, g_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["g_mag"])
    bp_rp, color_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["bp_rp"])
    distance_pc, distance_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["distance_pc"])
    _record_input_source(df, "g_mag", g_source)
    _record_input_source(df, "bp_rp", color_source)
    _record_input_source(df, "distance", distance_source)
    valid = g_mag.notna() & bp_rp.notna() & distance_pc.notna() & (distance_pc > 0)
    df['cv_classifier_status'] = np.where(valid, 'ok', 'missing_cmd_inputs')

    if valid.any():
        dist_pc = distance_pc.loc[valid]
        G_abs = g_mag.loc[valid] - 5 * np.log10(dist_pc / 10)
        bp_rp_valid = bp_rp.loc[valid]
        
        # CV region: blue (BP-RP < CV_BP_RP_THRESHOLD) and faint (G_abs > CV_G_ABS_THRESHOLD)
        cv_like = (bp_rp_valid < CV_BP_RP_THRESHOLD) & (G_abs > CV_G_ABS_THRESHOLD)
        df.loc[valid, 'P_cv'] = np.where(cv_like, CV_BASE_P, 0.01)

        # cv_like is indexed only over the valid-distance subset; use its index
        # to avoid boolean mask shape mismatches when some rows are invalid.
        cv_like_idx = cv_like[cv_like].index
        _append_note(df, df.index.to_series().isin(cv_like_idx), 'cv_notes', 'Blue+faint in CMD; ')
    
    # Check Hα if IPHAS data available
    if 'ha_ew' in df.columns:
        ha_ew = pd.to_numeric(df['ha_ew'], errors='coerce')
        ha_measured = ha_ew.notna()
        ha_excess = ha_measured & (ha_ew > CV_HA_EW_THRESHOLD)  # Å
        df.loc[ha_measured & df['P_cv'].isna(), 'P_cv'] = 0.0
        df.loc[ha_excess, 'P_cv'] = np.minimum(df.loc[ha_excess, 'P_cv'].fillna(0.0) + CV_HA_BONUS, 1.0)
        _append_note(df, ha_excess, 'cv_notes', 'Hα excess; ')
        df.loc[ha_measured & ~valid, 'cv_classifier_status'] = 'partial_halpha_only'
    
    # Check for known CV catalogs
    if 'is_known_cv' in df.columns:
        known = _coerce_bool_evidence(df['is_known_cv'])
        df.loc[known, 'P_cv'] = CV_KNOWN_P
        _append_note(df, known, 'cv_notes', 'Known CV; ')
        df.loc[known, 'cv_classifier_status'] = 'known_catalog_cv'
    
    return df


# =============================================================================
# STARSPOT REJECTION
# =============================================================================

def check_starspot_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test for starspot-induced variability.
    
    Checks:
    1. Amplitude (starspots cause ~few % variations, not >0.1 mag)
    2. Timescale (starspots modulate on rotation periods: hours-days)
    
    Returns df with columns: P_starspot, starspot_notes
    """
    df = df.copy()
    df['P_starspot'] = np.nan
    df['starspot_notes'] = ''
    depth, depth_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["depth_mag"])
    fractional_depth, fractional_source = _coalesce_numeric(
        df, CLASSIFIER_INPUT_ALIASES["depth_fraction"]
    )
    use_fraction = depth.isna() & fractional_depth.notna()
    # Convert a fractional flux decrement to its equivalent magnitude depth.
    valid_fraction = use_fraction & (fractional_depth > 0) & (fractional_depth < 1)
    depth.loc[valid_fraction] = -2.5 * np.log10(1.0 - fractional_depth.loc[valid_fraction])
    depth_source.loc[valid_fraction] = fractional_source.loc[valid_fraction] + ':fraction_to_mag'
    _record_input_source(df, "depth", depth_source)
    valid_depth = depth.notna() & (depth >= 0)
    df['starspot_classifier_status'] = np.where(valid_depth, 'ok', 'missing_depth')
    _append_note(df, ~valid_depth, 'starspot_notes', 'No valid depth data; ')
    
    # Starspots typically cause <0.05 mag variations
    # Dips >0.1 mag are unlikely to be starspots
    
    small_amp = valid_depth & (depth < STARSPOT_SMALL_AMP)
    medium_amp = valid_depth & (depth >= STARSPOT_SMALL_AMP) & (depth < STARSPOT_MEDIUM_AMP)
    large_amp = valid_depth & (depth >= STARSPOT_MEDIUM_AMP)

    df.loc[small_amp, 'P_starspot'] = STARSPOT_SMALL_P
    df.loc[small_amp, 'starspot_notes'] += 'Small amplitude; '

    df.loc[medium_amp, 'P_starspot'] = STARSPOT_MEDIUM_P
    df.loc[large_amp, 'P_starspot'] = STARSPOT_LARGE_P
    
    # Timescale check
    duration, duration_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["duration_days"])
    if duration.notna().any():
        # Starspot rotation periods are typically hours to ~STARSPOT_ROTATION_PERIOD_DAYS days
        short_timescale = duration.notna() & (duration > 0) & (duration < STARSPOT_ROTATION_PERIOD_DAYS)
        df.loc[short_timescale, 'P_starspot'] = np.minimum(
            df.loc[short_timescale, 'P_starspot'].fillna(0.0) + STARSPOT_MEDIUM_P, 1.0
        )
        df.loc[short_timescale & ~valid_depth, 'starspot_classifier_status'] = 'partial_timescale_only'
    
    return df


# =============================================================================
# YSO CLASSIFICATION
# =============================================================================

def classify_yso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify YSO candidates using 2MASS-WISE IR colors.
    Following Koenig & Leisawitz (2014).
    
    Returns df with columns: yso_class, H_K, w1_w2
    """
    df = df.copy()
    
    H, h_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["h_mag"])
    K, k_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["k_mag"])
    W1, w1_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["w1_mag"])
    W2, w2_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["w2_mag"])
    for key, source in (
        ("h_mag", h_source), ("k_mag", k_source),
        ("w1_mag", w1_source), ("w2_mag", w2_source),
    ):
        _record_input_source(df, key, source)
        
    hk_color = H - K
    w1w2_color = W1 - W2
    
    # Dust correction if available
    df['yso_extinction_status'] = 'not_available'
    if 'A_v_3d' in df.columns:
        av = pd.to_numeric(df['A_v_3d'], errors='coerce')
        valid_av = av.notna() & np.isfinite(av) & (av >= 0)
        hk_color.loc[valid_av] = hk_color.loc[valid_av] - (YSO_DUST_CORRECTION_HK * av.loc[valid_av])
        w1w2_color.loc[valid_av] = w1w2_color.loc[valid_av] - (YSO_DUST_CORRECTION_W1W2 * av.loc[valid_av])
        df.loc[valid_av, 'yso_extinction_status'] = np.where(
            av.loc[valid_av] > 0, 'corrected', 'measured_zero'
        )
        df.loc[av.notna() & ~valid_av, 'yso_extinction_status'] = 'invalid'
    
    df['H_K'] = hk_color 
    df['w1_w2'] = w1w2_color
    valid_colors = hk_color.notna() & w1w2_color.notna()
    df['yso_classifier_status'] = np.where(valid_colors, 'ok', 'missing_ir_bands')
    
    # Classification
    class_i = valid_colors & (df['w1_w2'] >= YSO_CLASS_I_W1W2)
    class_ii = (
        valid_colors
        & (df['w1_w2'] > YSO_CLASS_II_W1W2_MIN)
        & (df['w1_w2'] < YSO_CLASS_I_W1W2)
        & (df['H_K'] >= YSO_CLASS_II_HK)
    )
    trans = (
        valid_colors
        & (df['w1_w2'] > YSO_CLASS_II_W1W2_MIN)
        & (df['w1_w2'] < YSO_CLASS_I_W1W2)
        & (df['H_K'] < YSO_CLASS_II_HK)
    )
    ms = valid_colors & (df['w1_w2'] <= YSO_CLASS_II_W1W2_MIN)
    
    df['yso_class'] = 'unknown'
    df.loc[class_i, 'yso_class'] = 'Class I'
    df.loc[class_ii, 'yso_class'] = 'Class II'
    df.loc[trans, 'yso_class'] = 'Transition Disk'
    df.loc[ms, 'yso_class'] = 'Main Sequence'
    
    return df


# =============================================================================
# CIRCUMSTELLAR MATERIAL ESTIMATION
# =============================================================================

def estimate_semimajor_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate upper limit on semimajor axis of occulting material.
    
    Based on Tzanidakis et al. (2025) Eq. 11:
    a_circ ∝ M*^(1/2) * (S + R*)^(1/2) * Δt
    
    Assumes:
    - Circular equatorial transit
    - Opaque occulter
    - Occulter mass << stellar mass
    
    Returns df with columns: a_circ_au, transit_prob, hill_radius_rsun
    """
    df = df.copy()
    df['a_circ_au'] = np.nan
    df['transit_prob'] = np.nan
    df['hill_radius_rsun'] = np.nan
    df['semimajor_status'] = 'missing_inputs'
    
    # Get dip depth
    depth_mag, depth_mag_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["depth_mag"])
    depth_fraction, depth_fraction_source = _coalesce_numeric(
        df, CLASSIFIER_INPUT_ALIASES["depth_fraction"]
    )
    tau = pd.Series(np.nan, index=df.index, dtype=float)
    valid_mag = depth_mag.notna() & (depth_mag >= 0)
    tau.loc[valid_mag] = 1 - 10 ** (-0.4 * depth_mag.loc[valid_mag])
    use_fraction = tau.isna() & depth_fraction.notna() & (depth_fraction > 0) & (depth_fraction < 1)
    tau.loc[use_fraction] = depth_fraction.loc[use_fraction]
    depth_source = depth_mag_source.copy()
    depth_source.loc[use_fraction] = depth_fraction_source.loc[use_fraction] + ':fraction'
    _record_input_source(df, "semimajor_depth", depth_source)
    
    # Get duration
    dt_days, duration_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["duration_days"])
    _record_input_source(df, "semimajor_duration", duration_source)
    
    # Stellar mass (default 1 M_sun)
    M_star, mass_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["mass_solar"])
    assumed_mass = M_star.isna()
    M_star = M_star.fillna(1.0)
    mass_source.loc[assumed_mass] = 'assumed_solar'
    _record_input_source(df, "mass", mass_source)
    
    # Stellar radius (default 1 R_sun)
    R_star, radius_source = _coalesce_numeric(df, CLASSIFIER_INPUT_ALIASES["radius_solar"])
    assumed_radius = R_star.isna()
    R_star = R_star.fillna(1.0)
    radius_source.loc[assumed_radius] = 'assumed_solar'
    _record_input_source(df, "radius", radius_source)

    valid = (
        tau.notna() & (tau > 0) & (tau < 1)
        & dt_days.notna() & (dt_days > 0)
        & np.isfinite(M_star) & (M_star > 0)
        & np.isfinite(R_star) & (R_star > 0)
    )
    df.loc[valid, 'semimajor_status'] = 'ok'
    df.loc[valid & (assumed_mass | assumed_radius), 'semimajor_status'] = 'assumed_stellar_properties'
    
    # Occulter size estimate (assume S ~ R* * sqrt(tau))
    S = R_star * np.sqrt(tau)
    
    # Semimajor axis (simplified Keplerian)
    # v = 2π * a / P, transit duration ~ 2(S+R*)/v
    # Solving: a ~ [M* * (S+R*) * dt]^(1/2) in appropriate units
    
    # Using Kepler's 3rd law: P = 2π * sqrt(a^3 / GM)
    # Transit duration ~ 2(S+R*) * P / (2π * a) = (S+R*) * sqrt(a / GM)
    # Solving for a: a = (GM * dt^2) / (S+R*)^2
    
    M_kg = M_star * SOLAR_MASS_KG
    R_m = R_star * SOLAR_RADIUS_M
    S_m = S * SOLAR_RADIUS_M
    dt_s = dt_days * DAY_S
    
    # a = GM * dt^2 / (S+R)^2
    a_m = (GRAVITATIONAL_CONSTANT_SI * M_kg * dt_s**2) / ((S_m + R_m)**2)
    a_au = a_m / AU_M
    
    df.loc[valid, 'a_circ_au'] = a_au.loc[valid]
    
    # Transit probability: P ~ (R* + R_occulter) / a
    transit_probability = ((R_m + S_m) / a_m).clip(lower=0.0, upper=1.0)
    df.loc[valid, 'transit_prob'] = transit_probability.loc[valid]
    
    # Hill radius for 1 Earth mass at estimated a
    # R_H = a * (M_planet / 3*M_star)^(1/3)
    r_hill_m = a_m * (EARTH_MASS_KG / (3 * M_kg))**(1/3)
    hill_radius = r_hill_m / SOLAR_RADIUS_M
    df.loc[valid, 'hill_radius_rsun'] = hill_radius.loc[valid]
    
    return df


# =============================================================================
# DISK OCCULTATION PROBABILITY
# =============================================================================

def estimate_disk_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate probability of disk occultation scenario.
    
    Favorable conditions:
    - Large semimajor axis (>2 AU)
    - Sufficient Hill radius for disk
    - No hot disk detected in WISE
    
    Returns df with column: P_disk
    """
    df = df.copy()
    df['P_disk'] = DISK_BASE_P  # Base probability
    df['disk_classifier_status'] = 'prior_only'

    # Check semimajor axis
    if 'a_circ_au' in df.columns:
        a_circ = pd.to_numeric(df['a_circ_au'], errors='coerce')
        valid_a = a_circ.notna() & (a_circ > 0)
        large_a = valid_a & (a_circ > DISK_LARGE_A_AU)
        very_large_a = valid_a & (a_circ > DISK_VERY_LARGE_A_AU)
        df.loc[valid_a, 'disk_classifier_status'] = 'ok'

        df.loc[large_a, 'P_disk'] = DISK_LARGE_A_P
        df.loc[very_large_a, 'P_disk'] = DISK_VERY_LARGE_A_P

    # Check Hill radius
    if 'hill_radius_rsun' in df.columns:
        hill = pd.to_numeric(df['hill_radius_rsun'], errors='coerce')
        large_hill = hill.notna() & (hill > 10)
        df.loc[large_hill, 'P_disk'] += DISK_LARGE_HILL_BONUS

    # Check WISE upper limits (no hot disk)
    # If W3/W4 not detected, consistent with cool/no disk
    if 'w1_w2' in df.columns:
        w1_w2 = pd.to_numeric(df['w1_w2'], errors='coerce')
        no_ir_excess = w1_w2.notna() & (w1_w2 < YSO_CLASS_II_W1W2_MIN)
        df.loc[no_ir_excess, 'P_disk'] += DISK_NO_IR_EXCESS_BONUS

    # Cap at DISK_P_CAP (never fully certain without RV confirmation)
    df['P_disk'] = df['P_disk'].clip(upper=DISK_P_CAP)
    
    return df


# =============================================================================
# MASTER CLASSIFICATION
# =============================================================================

def compute_all_classifications(
    df: pd.DataFrame,
    *,
    run_eb: bool = True,
    run_cv: bool = True,
    run_starspot: bool = True,
) -> pd.DataFrame:
    """
    Run all classifiers and compute final classification.
    
    Returns df with:
    - P_eb, P_cv, P_starspot, P_disk
    - yso_class
    - a_circ_au, transit_prob, hill_radius_rsun
    - final_class (most likely classification)
    """
    df = df.copy()
    print("Running EB contamination check...")
    if run_eb:
        df = check_eb_contamination(df)
    else:
        df['P_eb'] = np.nan
        df['eb_notes'] = 'Disabled; '
        df['eb_classifier_status'] = 'disabled'
    
    print("Running CV contamination check...")
    if run_cv:
        df = check_cv_contamination(df)
    else:
        df['P_cv'] = np.nan
        df['cv_notes'] = 'Disabled; '
        df['cv_classifier_status'] = 'disabled'
    
    print("Running starspot check...")
    if run_starspot:
        df = check_starspot_contamination(df)
    else:
        df['P_starspot'] = np.nan
        df['starspot_notes'] = 'Disabled; '
        df['starspot_classifier_status'] = 'disabled'
    
    print("Running YSO classification...")
    df = classify_yso(df)
    
    print("Estimating semimajor axis...")
    df = estimate_semimajor_axis(df)
    
    print("Estimating disk probability...")
    df = estimate_disk_probability(df)
    
    # Compute final classification
    # Priority: known classes > high-probability contaminants > disk/circumstellar
    
    df['final_class'] = 'Unknown Dipper'
    df['classification_score'] = np.nan
    df['classification_score_kind'] = 'uncalibrated_heuristic'
    df['classification_version'] = CLASSIFIER_VERSION

    score_frame = pd.DataFrame(
        {
            'Likely EB': _finite_score(df['P_eb']),
            'Likely CV': _finite_score(df['P_cv']),
            'Likely Starspot': _finite_score(df['P_starspot']),
        },
        index=df.index,
    )
    thresholds = pd.Series(
        {
            'Likely EB': CLASSIFY_EB_THRESHOLD,
            'Likely CV': CLASSIFY_CV_THRESHOLD,
            'Likely Starspot': CLASSIFY_STARSPOT_THRESHOLD,
        }
    )
    eligible = score_frame.gt(thresholds, axis='columns')
    winning_scores = score_frame.where(eligible).max(axis=1, skipna=True)
    has_contaminant = eligible.any(axis=1)
    winning_labels = pd.Series(pd.NA, index=df.index, dtype="string")
    if has_contaminant.any():
        winning_labels.loc[has_contaminant] = (
            score_frame.loc[has_contaminant]
            .where(eligible.loc[has_contaminant])
            .idxmax(axis=1)
            .astype("string")
        )
    df.loc[has_contaminant, 'final_class'] = winning_labels.loc[has_contaminant]
    df.loc[has_contaminant, 'classification_score'] = winning_scores.loc[has_contaminant]

    # YSO classes are assigned only when no contaminant score crossed its
    # threshold.  This prevents iteration order from overwriting a stronger,
    # contradictory contaminant score.
    yso_classes = ['Class I', 'Class II', 'Transition Disk']
    for yc in yso_classes:
        mask = (~has_contaminant) & (df['yso_class'] == yc)
        df.loc[mask, 'final_class'] = f'YSO ({yc})'

    # Disk candidates
    disk_cand = (df['P_disk'] > CLASSIFY_DISK_THRESHOLD) & (df['final_class'] == 'Unknown Dipper')
    df.loc[disk_cand, 'final_class'] = 'Disk Occultation Candidate'
    df.loc[disk_cand, 'classification_score'] = df.loc[disk_cand, 'P_disk']

    # Main sequence dippers
    eb_rejected = df['P_eb'].notna() & (df['P_eb'] < CLASSIFY_MS_EB_REJECTION)
    cv_rejected = df['P_cv'].notna() & (df['P_cv'] < CLASSIFY_MS_CV_REJECTION)
    ms_dipper = (
        (df['final_class'] == 'Unknown Dipper')
        & (df['yso_class'] == 'Main Sequence')
        & eb_rejected
        & cv_rejected
    )
    df.loc[ms_dipper, 'final_class'] = 'Main Sequence Dipper'

    component_status_columns = [
        'eb_classifier_status', 'cv_classifier_status', 'starspot_classifier_status',
        'yso_classifier_status', 'semimajor_status', 'disk_classifier_status',
    ]
    component_statuses = df[component_status_columns].fillna('unknown').astype(str)
    df['classification_status'] = 'ok'
    any_error = component_statuses.apply(lambda row: any(value == 'error' for value in row), axis=1)
    any_missing = component_statuses.apply(
        lambda row: any(value.startswith('missing') or value.startswith('partial') for value in row),
        axis=1,
    )
    all_disabled_or_missing = component_statuses.apply(
        lambda row: all(value == 'disabled' or value.startswith('missing') for value in row),
        axis=1,
    )
    df.loc[any_missing, 'classification_status'] = 'partial_inputs'
    df.loc[all_disabled_or_missing, 'classification_status'] = 'insufficient_inputs'
    df.loc[any_error, 'classification_status'] = 'error'
    df['classification_scores_json'] = [
        json.dumps(
            {
                'P_eb': None if pd.isna(row.P_eb) else float(row.P_eb),
                'P_cv': None if pd.isna(row.P_cv) else float(row.P_cv),
                'P_starspot': None if pd.isna(row.P_starspot) else float(row.P_starspot),
                'P_disk': None if pd.isna(row.P_disk) else float(row.P_disk),
            },
            sort_keys=True,
            separators=(',', ':'),
        )
        for row in df[['P_eb', 'P_cv', 'P_starspot', 'P_disk']].itertuples(index=False)
    ]
    source_columns = sorted(col for col in df.columns if col.startswith('classification_') and col.endswith('_source'))
    df['classification_input_map_json'] = [
        json.dumps(
            {col.removeprefix('classification_').removesuffix('_source'): str(value or '') for col, value in zip(source_columns, row)},
            sort_keys=True,
            separators=(',', ':'),
        )
        for row in df[source_columns].itertuples(index=False, name=None)
    ]
    
    print("Classification complete.")
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Classify dipper candidates (Tzanidakis+ 2025)")
    parser.add_argument("--input", type=Path, required=True, help="Input events Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output classified Parquet")
    parser.add_argument("--skip-eb", action="store_true", help="Skip EB check")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV check")
    parser.add_argument("--skip-starspot", action="store_true", help="Skip starspot check")
    parser.add_argument("--enable-iphas", dest="iphas", action="store_true", help="Query IPHAS for H-alpha photometry (slow)")
    parser.add_argument("--enable-ps1", dest="ps1", action="store_true", help="Query PS1 for grizy photometry (slow)")
    parser.add_argument("--all-candidates", action="store_true", help="Classify all input rows instead of only failed_any=False passers")
    
    args = parser.parse_args()
    
    # Load
    print(f"Loading {args.input}...")
    df = read_feature_table(args.input)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
    
    print(f"Loaded {len(df)} events")
    
    # Optional external queries
    if args.iphas:
        print("\n=== Querying IPHAS ===")
        df = query_iphas_by_coords(df)
    
    if args.ps1:
        print("\n=== Querying PS1 ===")
        df = query_ps1_by_coords(df)
    
    # Classify
    print("\n=== Running Classification ===")
    df = compute_all_classifications(
        df,
        run_eb=not args.skip_eb,
        run_cv=not args.skip_cv,
        run_starspot=not args.skip_starspot,
    )
    
    # Summary
    print("\n=== Classification Summary ===")
    print(df['final_class'].value_counts())
    
    # Save
    print(f"\nSaving to {args.output}...")
    write_feature_table(df, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
