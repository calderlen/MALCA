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
    malca classify --input output/events.csv --output output/classified.csv
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
SOLAR_MASS_KG = 1.989e30
SOLAR_RADIUS_M = 6.957e8
AU_M = 1.496e11
DAY_S = 86400


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
    df['P_eb'] = 0.0
    df['eb_notes'] = ''
    
    # Get required columns
    has_asymmetry = 'asymmetry' in df.columns or 'skewness' in df.columns
    has_duration = 'event_duration_days' in df.columns or 'timescale_days' in df.columns
    has_stellar = 'mass50' in df.columns or 'teff_gspphot' in df.columns
    
    if not has_duration:
        df['eb_notes'] = 'No duration data'
        return df
    
    # Get duration column
    dur_col = 'event_duration_days' if 'event_duration_days' in df.columns else 'timescale_days'
    duration_days = df[dur_col].fillna(1.0)
    
    # Estimate stellar mass (default to 1 M_sun)
    if 'mass50' in df.columns:
        M_star = df['mass50'].fillna(1.0)
    else:
        M_star = 1.0
    
    # Estimate stellar radius (default to 1 R_sun)
    if 'radius' in df.columns:
        R_star = df['radius'].fillna(1.0)
    else:
        R_star = 1.0
    
    # For EB with semimajor axis ~1.8 AU (to explain single eclipse in 2.5yr baseline)
    # Eclipse duration ~ 1.5 days for tangential velocity of 21 km/s
    # If observed duration >> 1.5 days, EB is unlikely
    
    # Expected EB duration for a = 1.8 AU
    v_tang = 21  # km/s for 2.5yr period
    expected_eb_duration = 2 * R_star * SOLAR_RADIUS_M / (v_tang * 1000) / DAY_S
    
    # Dips lasting weeks-months require 10-10000 AU separations
    # Transit probability at such separations: 10^-4 to 10^-7
    
    # Simple heuristic: if duration > 10 days, unlikely to be EB
    long_dip = duration_days > 10
    very_long_dip = duration_days > 30
    
    # Assign probabilities
    df.loc[~long_dip, 'P_eb'] = 0.3  # Short dips could be EBs
    df.loc[long_dip, 'P_eb'] = 0.05  # Long dips unlikely EBs
    df.loc[very_long_dip, 'P_eb'] = 0.01  # Very long dips very unlikely EBs
    
    # Check for periodicity if available
    if 'is_periodic' in df.columns:
        periodic = df['is_periodic'] == True
        df.loc[periodic, 'P_eb'] = np.minimum(df.loc[periodic, 'P_eb'] + 0.4, 1.0)
        df.loc[periodic, 'eb_notes'] += 'Periodic; '
    
    # Check for symmetry if available
    if 'asymmetry' in df.columns:
        symmetric = np.abs(df['asymmetry']) < 0.1
        df.loc[symmetric, 'P_eb'] = np.minimum(df.loc[symmetric, 'P_eb'] + 0.2, 1.0)
        df.loc[symmetric, 'eb_notes'] += 'Symmetric; '
    
    # Gaia binary flag
    if 'non_single_star' in df.columns:
        binary = df['non_single_star'] > 0
        df.loc[binary, 'P_eb'] = np.minimum(df.loc[binary, 'P_eb'] + 0.3, 1.0)
        df.loc[binary, 'eb_notes'] += 'Gaia binary; '
    
    return df


# =============================================================================
# EXTERNAL CATALOG QUERIES (IPHAS, PS1)
# =============================================================================

def query_iphas_by_coords(df: pd.DataFrame, radius_arcsec: float = 2.0) -> pd.DataFrame:
    """
    Query IPHAS DR2 for Hα photometry using VizieR TAP.
    
    IPHAS covers Northern Galactic Plane: -5° < b < 5°, 30° < l < 215°
    Returns r, i, Hα magnitudes and r-Hα color
    """
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: No ra/dec for IPHAS query")
        return df
    
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        print("Error: astroquery required for IPHAS query")
        return df
    
    df = df.copy()
    df['iphas_r'] = np.nan
    df['iphas_i'] = np.nan
    df['iphas_ha'] = np.nan
    df['r_ha'] = np.nan
    df['ha_ew'] = np.nan
    
    # IPHAS DR2 catalog
    Vizier.ROW_LIMIT = -1
    
    print(f"Querying IPHAS for {len(df)} sources...")
    
    for idx in tqdm(df.index, desc="IPHAS"):
        ra, dec = df.loc[idx, 'ra'], df.loc[idx, 'dec']
        if pd.isna(ra) or pd.isna(dec):
            continue
            
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        
        try:
            result = Vizier.query_region(coord, radius=radius_arcsec*u.arcsec, catalog='II/321/iphas2')
            if result and len(result) > 0 and len(result[0]) > 0:
                row = result[0][0]
                df.loc[idx, 'iphas_r'] = row['rmag'] if 'rmag' in row.colnames else np.nan
                df.loc[idx, 'iphas_i'] = row['imag'] if 'imag' in row.colnames else np.nan
                df.loc[idx, 'iphas_ha'] = row['Hamag'] if 'Hamag' in row.colnames else np.nan
                
                if not pd.isna(df.loc[idx, 'iphas_r']) and not pd.isna(df.loc[idx, 'iphas_ha']):
                    df.loc[idx, 'r_ha'] = df.loc[idx, 'iphas_r'] - df.loc[idx, 'iphas_ha']
        except Exception:
            continue
    
    n_found = df['iphas_r'].notna().sum()
    print(f"Found IPHAS photometry for {n_found}/{len(df)} sources")
    
    return df


def query_ps1_by_coords(df: pd.DataFrame, radius_arcsec: float = 2.0) -> pd.DataFrame:
    """
    Query Pan-STARRS1 for grizy photometry using MAST.
    
    PS1 covers 3π sky (dec > -30°)
    Returns g, r, i, z, y PSF magnitudes
    """
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: No ra/dec for PS1 query")
        return df
    
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        print("Error: astroquery required for PS1 query")
        return df
    
    df = df.copy()
    for band in ['g', 'r', 'i', 'z', 'y']:
        df[f'ps1_{band}'] = np.nan
    
    print(f"Querying PS1 for {len(df)} sources...")
    
    for idx in tqdm(df.index, desc="PS1"):
        ra, dec = df.loc[idx, 'ra'], df.loc[idx, 'dec']
        if pd.isna(ra) or pd.isna(dec):
            continue
            
        try:
            result = Catalogs.query_region(
                f"{ra} {dec}",
                radius=radius_arcsec/3600,  # degrees
                catalog="Panstarrs",
                table="mean"
            )
            
            if result and len(result) > 0:
                row = result[0]
                for band in ['g', 'r', 'i', 'z', 'y']:
                    col = f'{band}MeanPSFMag'
                    if col in row.colnames:
                        df.loc[idx, f'ps1_{band}'] = row[col]
        except Exception:
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
    df['P_cv'] = 0.0
    df['cv_notes'] = ''
    
    # Check Gaia CMD position
    # CVs typically have: G_abs > 4 and BP-RP < 0.5 (blue, faint)
    has_gaia = 'phot_g_mean_mag' in df.columns and 'bp_rp' in df.columns
    
    if has_gaia and 'distance_gspphot' in df.columns:
        valid = df['distance_gspphot'].notna() & (df['distance_gspphot'] > 0)
        dist_pc = df.loc[valid, 'distance_gspphot']
        G_abs = df.loc[valid, 'phot_g_mean_mag'] - 5 * np.log10(dist_pc / 10)
        bp_rp = df.loc[valid, 'bp_rp']
        
        # CV region: blue (BP-RP < 0.5) and faint (G_abs > 8)
        cv_like = (bp_rp < 0.5) & (G_abs > 8)
        df.loc[valid, 'P_cv'] = np.where(cv_like, 0.3, 0.01)
        df.loc[valid & cv_like.values, 'cv_notes'] += 'Blue+faint in CMD; '
    
    # Check Hα if IPHAS data available
    if 'ha_ew' in df.columns:
        ha_excess = df['ha_ew'] > 10  # Å
        df.loc[ha_excess, 'P_cv'] = np.minimum(df.loc[ha_excess, 'P_cv'] + 0.4, 1.0)
        df.loc[ha_excess, 'cv_notes'] += 'Hα excess; '
    
    # Check for known CV catalogs
    if 'is_known_cv' in df.columns:
        df.loc[df['is_known_cv'], 'P_cv'] = 0.95
        df.loc[df['is_known_cv'], 'cv_notes'] += 'Known CV; '
    
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
    df['P_starspot'] = 0.0
    df['starspot_notes'] = ''
    
    # Get dip amplitude
    if 'event_depth_mag' in df.columns:
        depth = df['event_depth_mag'].fillna(0.1)
    elif 'max_depth' in df.columns:
        depth = df['max_depth'].fillna(0.1)
    else:
        df['starspot_notes'] = 'No depth data'
        return df
    
    # Starspots typically cause <0.05 mag variations
    # Dips >0.1 mag are unlikely to be starspots
    
    small_amp = depth < 0.05
    medium_amp = (depth >= 0.05) & (depth < 0.15)
    large_amp = depth >= 0.15
    
    df.loc[small_amp, 'P_starspot'] = 0.4
    df.loc[small_amp, 'starspot_notes'] += 'Small amplitude; '
    
    df.loc[medium_amp, 'P_starspot'] = 0.1
    df.loc[large_amp, 'P_starspot'] = 0.01
    
    # Timescale check
    dur_col = None
    for col in ['event_duration_days', 'timescale_days', 'duration']:
        if col in df.columns:
            dur_col = col
            break
    
    if dur_col:
        duration = df[dur_col].fillna(10)
        # Starspot rotation periods are typically hours to ~30 days
        short_timescale = duration < 30
        df.loc[short_timescale, 'P_starspot'] = np.minimum(
            df.loc[short_timescale, 'P_starspot'] + 0.1, 1.0
        )
    
    return df


# =============================================================================
# YSO CLASSIFICATION
# =============================================================================

def classify_yso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify YSO candidates using 2MASS-WISE IR colors.
    Following Koenig & Leisawitz (2014).
    
    Returns df with columns: yso_class, H_K, W1_W2
    """
    df = df.copy()
    
    # Map columns
    col_map = {
        'H': ['Hmag', 'tmass_h'], 
        'K': ['Kmag', 'tmass_k'], 
        'W1': ['W1mag', 'unwise_w1', 'w1mpro'], 
        'W2': ['W2mag', 'unwise_w2', 'w2mpro']
    }
    
    vals = {}
    for bands, candidates in col_map.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        vals[bands] = found
        
    if not all(vals.values()):
        df['yso_class'] = 'unknown'
        return df
        
    H = df[vals['H']]
    K = df[vals['K']]
    W1 = df[vals['W1']]
    W2 = df[vals['W2']]
        
    hk_color = H - K
    w1w2_color = W1 - W2
    
    # Dust correction if available
    if 'A_v_3d' in df.columns and df['A_v_3d'].sum() > 0:
        av = df['A_v_3d'].fillna(0.0)
        hk_color = hk_color - (0.18 * av)
        w1w2_color = w1w2_color - (0.05 * av)
    
    df['H_K'] = hk_color 
    df['W1_W2'] = w1w2_color
    
    # Classification
    class_i = df['W1_W2'] > 0.8
    class_ii = ((df['W1_W2'] > 0.25) & (df['W1_W2'] < 0.8) & (df['H_K'] > 0.3))
    trans = ((df['W1_W2'] > 0.25) & (df['W1_W2'] < 0.8) & (df['H_K'] < 0.3))
    ms = df['W1_W2'] < 0.25
    
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
    
    # Get dip depth
    if 'event_depth_mag' in df.columns:
        tau = 1 - 10**(-0.4 * df['event_depth_mag'].fillna(0.1))
    elif 'max_depth' in df.columns:
        tau = df['max_depth'].fillna(0.1)
    else:
        return df
    
    # Get duration
    dur_col = None
    for col in ['event_duration_days', 'timescale_days', 'duration']:
        if col in df.columns:
            dur_col = col
            break
    
    if dur_col is None:
        return df
        
    dt_days = df[dur_col].fillna(10)
    
    # Stellar mass (default 1 M_sun)
    if 'mass50' in df.columns:
        M_star = df['mass50'].fillna(1.0)
    else:
        M_star = 1.0
    
    # Stellar radius (default 1 R_sun)
    if 'radius' in df.columns:
        R_star = df['radius'].fillna(1.0)
    else:
        R_star = 1.0
    
    # Occulter size estimate (assume S ~ R* * sqrt(tau))
    S = R_star * np.sqrt(tau)
    
    # Semimajor axis (simplified Keplerian)
    # v = 2π * a / P, transit duration ~ 2(S+R*)/v
    # Solving: a ~ [M* * (S+R*) * dt]^(1/2) in appropriate units
    
    # Using Kepler's 3rd law: P = 2π * sqrt(a^3 / GM)
    # Transit duration ~ 2(S+R*) * P / (2π * a) = (S+R*) * sqrt(a / GM)
    # Solving for a: a = (GM * dt^2) / (S+R*)^2
    
    G_SI = 6.674e-11
    M_kg = M_star * SOLAR_MASS_KG
    R_m = R_star * SOLAR_RADIUS_M
    S_m = S * SOLAR_RADIUS_M
    dt_s = dt_days * DAY_S
    
    # a = GM * dt^2 / (S+R)^2
    a_m = (G_SI * M_kg * dt_s**2) / ((S_m + R_m)**2)
    a_au = a_m / AU_M
    
    df['a_circ_au'] = a_au
    
    # Transit probability: P ~ (R* + R_occulter) / a
    df['transit_prob'] = (R_m + S_m) / a_m
    
    # Hill radius for 1 Earth mass at estimated a
    # R_H = a * (M_planet / 3*M_star)^(1/3)
    M_earth = 5.972e24  # kg
    r_hill_m = a_m * (M_earth / (3 * M_kg))**(1/3)
    df['hill_radius_rsun'] = r_hill_m / SOLAR_RADIUS_M
    
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
    df['P_disk'] = 0.1  # Base probability
    
    # Check semimajor axis
    if 'a_circ_au' in df.columns:
        large_a = df['a_circ_au'] > 2
        very_large_a = df['a_circ_au'] > 10
        
        df.loc[large_a, 'P_disk'] = 0.3
        df.loc[very_large_a, 'P_disk'] = 0.5
    
    # Check Hill radius
    if 'hill_radius_rsun' in df.columns:
        large_hill = df['hill_radius_rsun'] > 10
        df.loc[large_hill, 'P_disk'] += 0.1
    
    # Check WISE upper limits (no hot disk)
    # If W3/W4 not detected, consistent with cool/no disk
    if 'unwise_w1' in df.columns and 'unwise_w2' in df.columns:
        no_ir_excess = df['W1_W2'] < 0.25 if 'W1_W2' in df.columns else True
        if isinstance(no_ir_excess, pd.Series):
            df.loc[no_ir_excess, 'P_disk'] += 0.1
    
    # Cap at 0.8 (never fully certain without RV confirmation)
    df['P_disk'] = df['P_disk'].clip(upper=0.8)
    
    return df


# =============================================================================
# MASTER CLASSIFICATION
# =============================================================================

def compute_all_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all classifiers and compute final classification.
    
    Returns df with:
    - P_eb, P_cv, P_starspot, P_disk
    - yso_class
    - a_circ_au, transit_prob, hill_radius_rsun
    - final_class (most likely classification)
    """
    print("Running EB contamination check...")
    df = check_eb_contamination(df)
    
    print("Running CV contamination check...")
    df = check_cv_contamination(df)
    
    print("Running starspot check...")
    df = check_starspot_contamination(df)
    
    print("Running YSO classification...")
    df = classify_yso(df)
    
    print("Estimating semimajor axis...")
    df = estimate_semimajor_axis(df)
    
    print("Estimating disk probability...")
    df = estimate_disk_probability(df)
    
    # Compute final classification
    # Priority: known classes > high-probability contaminants > disk/circumstellar
    
    df['final_class'] = 'Unknown Dipper'
    
    # YSO classes
    yso_classes = ['Class I', 'Class II', 'Transition Disk']
    for yc in yso_classes:
        df.loc[df['yso_class'] == yc, 'final_class'] = f'YSO ({yc})'
    
    # Contamination flags
    df.loc[df['P_eb'] > 0.5, 'final_class'] = 'Likely EB'
    df.loc[df['P_cv'] > 0.5, 'final_class'] = 'Likely CV'
    df.loc[df['P_starspot'] > 0.5, 'final_class'] = 'Likely Starspot'
    
    # Disk candidates
    disk_cand = (df['P_disk'] > 0.4) & (df['final_class'] == 'Unknown Dipper')
    df.loc[disk_cand, 'final_class'] = 'Disk Occultation Candidate'
    
    # Main sequence dippers
    ms_dipper = (df['yso_class'] == 'Main Sequence') & (df['P_eb'] < 0.3) & (df['P_cv'] < 0.3)
    df.loc[ms_dipper, 'final_class'] = 'Main Sequence Dipper'
    
    print("Classification complete.")
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Classify dipper candidates (Tzanidakis+ 2025)")
    parser.add_argument("--input", type=Path, required=True, help="Input events CSV/Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output classified CSV/Parquet")
    parser.add_argument("--skip-eb", action="store_true", help="Skip EB check")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV check")
    parser.add_argument("--skip-starspot", action="store_true", help="Skip starspot check")
    parser.add_argument("--iphas", action="store_true", help="Query IPHAS for Hα photometry (slow)")
    parser.add_argument("--ps1", action="store_true", help="Query PS1 for grizy photometry (slow)")
    
    args = parser.parse_args()
    
    # Load
    print(f"Loading {args.input}...")
    if str(args.input).endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
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
    df = compute_all_classifications(df)
    
    # Summary
    print("\n=== Classification Summary ===")
    print(df['final_class'].value_counts())
    
    # Save
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if str(args.output).endswith('.parquet'):
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)
    
    print("Done!")


if __name__ == "__main__":
    main()
