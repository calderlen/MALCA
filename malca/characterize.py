"""
Multi-wavelength characterization for ASAS-SN dipper candidates.

This module consolidates:
- Gaia DR3 querying (astrometry, astrophysics, 2MASS/WISE photometry)
- StarHorse local catalog join (stellar ages, masses)
- 3D dust extinction via dustmaps3d (Wang et al. 2025)
- YSO classification (Koenig & Leisawitz 2014)
- Galactic population classification

Usage:
    malca characterize --input output/events.csv --output output/characterized.csv --dust
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.units as u
import pyvo
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.xmatch import XMatch
import banyan_sigma as banyan_sigma_pkg
from dustmaps3d import dustmaps3d

# Suppress astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

from malca.config.config_characterize import (
    GAIA_CHUNK_SIZE, STARHORSE_TAP_CHUNK_SIZE,
    IPHAS_MAX_SEP_ARCSEC, CLUSTER_MAX_SEP_ARCSEC, UNWISE_MAX_SEP_ARCSEC,
    UNWISE_TIMEOUT_SECONDS, UNWISE_FRACFLUX_MIN, UNWISE_QF_MIN,
    UNWISE_VARIABILITY_ZSCORE, UNWISE_EXPECTED_SCATTER_BASE,
    UNWISE_EXPECTED_SCATTER_SLOPE, UNWISE_EXPECTED_SCATTER_MAG_REF,
    SFR_MAX_DIST_KPC, SFR_DIST_TOLERANCE_FRACTION, SFR_CATALOG,
    BANYAN_MIN_ASSOC_PROB, IPHAS_HA_EXCESS_THRESHOLD,
)
from malca.config.config_paths import (
    VSX_CROSSMATCH_PATH, STARHORSE_DEFAULT_PATH, STARHORSE_TAP_URL,
    UNTIMELY_API_URL, DEFAULT_CACHE_DIR, GAIA_CACHE_FILE,
)


# =============================================================================
# GAIA DR3 QUERYING
# =============================================================================

def query_gaia_by_ids(source_ids: list[str | int], chunk_size: int = GAIA_CHUNK_SIZE, cache_file: str | None = None) -> pd.DataFrame:
    """
    Query Gaia DR3 for a list of Source IDs.
    
    Retrieves astrometry, astrophysics, and 2MASS/WISE photometry via ADQL joins.
    """
    cached_df = pd.DataFrame()
    ids_to_query = [str(x) for x in source_ids if str(x).isdigit()]
    
    if cache_file and Path(cache_file).exists():
        print(f"Loading Gaia cache from {cache_file}...")
        cached_df = pd.read_parquet(cache_file)
        if "source_id" in cached_df.columns:
            cached_df["source_id"] = cached_df["source_id"].astype(str)
            processed_ids = set(cached_df["source_id"])
            ids_to_query = [x for x in ids_to_query if x not in processed_ids]
            if ids_to_query:
                print(f"Use {len(processed_ids)} cached sources. Querying {len(ids_to_query)} new sources.")
            else:
                return cached_df
    
    if not ids_to_query:
        return cached_df

    results = []
    
    for i in tqdm(range(0, len(ids_to_query), chunk_size), desc="Querying Gaia DR3"):
        chunk_ids = ids_to_query[i : i + chunk_size]
        ids_str = ",".join(chunk_ids)
        
        query = f"""
        SELECT
            g.source_id,
            g.ra, g.dec,
            g.parallax, g.parallax_error, g.ruwe,
            g.pmra, g.pmdec,
            g.phot_g_mean_mag, g.bp_rp,
            g.teff_gspphot, g.logg_gspphot, g.mh_gspphot,
            g.distance_gspphot, g.ag_gspphot,
            
            xm_tm.original_ext_source_id AS tmass_id,
            tm.j_m AS tmass_j, tm.h_m AS tmass_h, tm.ks_m AS tmass_k,
            tm.j_msigcom AS tmass_j_err, tm.h_msigcom AS tmass_h_err, tm.ks_msigcom AS tmass_k_err,
            
            xm_aw.original_ext_source_id AS allwise_id,
            aw.w1mpro AS unwise_w1, aw.w2mpro AS unwise_w2,
            aw.w1sigmpro AS unwise_w1_err, aw.w2sigmpro AS unwise_w2_err
            
        FROM gaiadr3.gaia_source AS g
        
        LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm_tm
            ON g.source_id = xm_tm.source_id
        LEFT JOIN external.tmass_psc AS tm
            ON xm_tm.original_ext_source_id = tm.designation
            
        LEFT JOIN gaiadr3.allwise_best_neighbour AS xm_aw
            ON g.source_id = xm_aw.source_id
        LEFT JOIN external.allwise AS aw
            ON xm_aw.original_ext_source_id = aw.designation
            
        WHERE g.source_id IN ({ids_str})
        """
        
        try:
            job = Gaia.launch_job_async(query)
            chunk_df = job.get_results().to_pandas()
            results.append(chunk_df)
        except Exception as e:
            print(f"Error querying Gaia chunk {i}: {e}")
            continue
            
    new_results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    if not new_results.empty and "source_id" in new_results.columns:
        new_results["source_id"] = new_results["source_id"].astype(str)
        
    full_df = pd.concat([cached_df, new_results], ignore_index=True) if not new_results.empty else cached_df
    
    if cache_file and not new_results.empty:
        full_df.to_parquet(cache_file, index=False, compression="snappy")
        
    return full_df


# =============================================================================
# STARHORSE LOCAL CATALOG
# =============================================================================

def query_starhorse_by_ids(source_ids: list[str | int], starhorse_file: str | Path | None = None, use_tap: bool = True) -> pd.DataFrame:
    """
    Retrieve StarHorse 2021 stellar parameters (Anders et al.).
    
    **Recommended**: TAP queries (default, use_tap=True)
    - Queries gaia.aip.de TAP service remotely
    - No large download required
    - Returns age, mass, distance, extinction
    
    **Alternative**: Local catalog join (use_tap=False)
    - Requires downloading ~100GB catalog from https://cdsarc.cds.unistra.fr/viz-bin/cat/I/354
    - Faster for repeated queries on same dataset
    """
    if use_tap:
        # TAP query via pyvo
        # Convert IDs to strings
        valid_ids = [str(x) for x in source_ids if str(x).isdigit()]
        if not valid_ids:
            return pd.DataFrame()
            
        print(f"Querying StarHorse via TAP for {len(valid_ids)} sources...")
        
        # Query in chunks (TAP has query length limits)
        chunk_size = STARHORSE_TAP_CHUNK_SIZE
        results = []
        
        for i in tqdm(range(0, len(valid_ids), chunk_size), desc="StarHorse TAP"):
            chunk_ids = valid_ids[i:i+chunk_size]
            ids_str = ",".join(chunk_ids)
            
            query = f"""
            SELECT 
                source_id,
                teff50, logg50, met50,
                dist50, dist16, dist84,
                av50, av16, av84,
                mass50, mass16, mass84,
                age50, age16, age84
            FROM gaiaedr3_contrib.starhorse
            WHERE source_id IN ({ids_str})
            """
            
            try:
                tap_service = pyvo.dal.TAPService(STARHORSE_TAP_URL)
                result = tap_service.search(query)
                chunk_df = result.to_table().to_pandas()
                results.append(chunk_df)
            except Exception as e:
                print(f"TAP query error for chunk {i}: {e}")
                continue
        
        if not results:
            print("Warning: No StarHorse results from TAP queries.")
            return pd.DataFrame()
            
        sh_df = pd.concat(results, ignore_index=True)
        sh_df['source_id'] = sh_df['source_id'].astype(str)
        
        print(f"Retrieved {len(sh_df)} StarHorse entries via TAP.")
        return sh_df
        
    else:
        # Local catalog join (original implementation)
        if starhorse_file is None:
            starhorse_file = os.environ.get('STARHORSE_PATH', STARHORSE_DEFAULT_PATH)
            
        starhorse_path = Path(starhorse_file)
        
        if not starhorse_path.exists():
            print(f"Warning: StarHorse catalog not found at {starhorse_path}")
            print("Tip: Use use_tap=True to query remotely instead of downloading 100GB catalog.")
            return pd.DataFrame()
            
        print(f"Loading StarHorse catalog from {starhorse_path}...")
        
        try:
            if str(starhorse_path).endswith('.parquet'):
                sh_df = pd.read_parquet(starhorse_path)
            elif str(starhorse_path).endswith('.fits') or str(starhorse_path).endswith('.fits.gz'):
                sh_df = Table.read(starhorse_path).to_pandas()
            else:
                sh_df = pd.read_csv(starhorse_path)
        except Exception as e:
            print(f"Error loading StarHorse: {e}")
            return pd.DataFrame()
            
        # Standardize column name
        if 'Source' in sh_df.columns:
            sh_df = sh_df.rename(columns={'Source': 'source_id'})
        elif 'EDR3Name' in sh_df.columns:
            sh_df = sh_df.rename(columns={'EDR3Name': 'source_id'})
            
        if 'source_id' not in sh_df.columns:
            print("Warning: Could not find source_id column in StarHorse catalog.")
            return pd.DataFrame()
            
        sh_df['source_id'] = sh_df['source_id'].astype(str)
        
        # Filter to requested IDs
        valid_ids = set(str(x) for x in source_ids if str(x).isdigit())
        sh_filtered = sh_df[sh_df['source_id'].isin(valid_ids)]
        
        print(f"Found {len(sh_filtered)}/{len(valid_ids)} sources in StarHorse catalog.")
        
        return sh_filtered


# =============================================================================
# 3D DUST EXTINCTION (dustmaps3d - Wang et al. 2025)
# =============================================================================

def get_dust_extinction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query 3D dust extinction using dustmaps3d (Wang et al. 2025).
    All-sky coverage, ~350MB data, fast queries.
    """
    if df.empty:
        return df
        
    df = df.copy()
    df['A_v_3d'] = 0.0
    df['ebv_3d'] = np.nan
    
    if 'ra' in df.columns and 'dec' in df.columns:
        ra_col = 'ra'
        dec_col = 'dec'
    elif 'ra_deg' in df.columns and 'dec_deg' in df.columns:
        ra_col = 'ra_deg'
        dec_col = 'dec_deg'
    else:
        print("Warning: Missing ra/dec columns for dust query.")
        return df
        
    # Distance (in pc from Gaia, need kpc for dustmaps3d)
    if 'distance_gspphot' in df.columns:
        dist_pc = df['distance_gspphot'].values
    elif 'gaia_parallax' in df.columns:
        plx = df['gaia_parallax'].values
        valid_plx = (np.isfinite(plx)) & (plx > 0)
        dist_pc = np.full(len(df), np.nan)
        dist_pc[valid_plx] = 1000.0 / plx[valid_plx]
    elif 'parallax' in df.columns:
        plx = df['parallax'].values
        valid_plx = (np.isfinite(plx)) & (plx > 0)
        dist_pc = np.full(len(df), np.nan)
        dist_pc[valid_plx] = 1000.0 / plx[valid_plx]
    else:
        print("Warning: No distance info for dust query.")
        return df
    
    dist_kpc = dist_pc / 1000.0
    valid_mask = (np.isfinite(df[ra_col])) & (np.isfinite(df[dec_col])) & (np.isfinite(dist_kpc)) & (dist_kpc > 0)
    
    if not valid_mask.any():
        return df
    
    # Convert RA/Dec to Galactic l, b
    coords = SkyCoord(ra=df.loc[valid_mask, ra_col].values * u.deg, 
                      dec=df.loc[valid_mask, dec_col].values * u.deg, 
                      frame='icrs')
    galactic = coords.galactic
    
    l = galactic.l.deg
    b = galactic.b.deg
    d = dist_kpc[valid_mask]
    
    try:
        print(f"Querying dustmaps3d for {valid_mask.sum()} sources...")
        ebv, dust_density, sigma, max_dist = dustmaps3d(l, b, d)
        
        A_V = 3.1 * ebv
        
        df.loc[valid_mask, 'ebv_3d'] = ebv
        df.loc[valid_mask, 'A_v_3d'] = A_V
        df.loc[valid_mask, 'dust_sigma'] = sigma
        df.loc[valid_mask, 'dust_max_dist_kpc'] = max_dist
        
        print(f"Dust query complete. Mean A_V = {A_V[np.isfinite(A_V)].mean():.3f}")
        
    except Exception as e:
        print(f"Error querying dustmaps3d: {e}")
        
    df['A_v_3d'] = df['A_v_3d'].fillna(0.0)
    
    return df


# =============================================================================
# YSO CLASSIFICATION (Koenig & Leisawitz 2014)
# =============================================================================

def classify_yso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify YSO candidates using 2MASS-WISE color-color diagram.
    Supports dust-corrected colors if A_v_3d is present.
    """
    df = df.copy()
    
    # Map columns (support multiple naming conventions)
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
    
    # Dust Correction
    if 'A_v_3d' in df.columns and df['A_v_3d'].sum() > 0:
        av = df['A_v_3d'].fillna(0.0)
        hk_color = hk_color - (0.18 * av)
        w1w2_color = w1w2_color - (0.05 * av)
        df['H_K_dered'] = hk_color
        df['W1_W2_dered'] = w1w2_color
    
    df['H_K'] = hk_color 
    df['W1_W2'] = w1w2_color
    
    # Classification criteria
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
# GALACTIC POPULATION CLASSIFICATION
# =============================================================================

def classify_galactic_population(df: pd.DataFrame) -> pd.DataFrame:
    """Classify stars into thin/thick disk based on age (StarHorse) or metallicity (Gaia)."""
    if df.empty:
        return df
    df = df.copy()
    
    # Use StarHorse age if available
    if 'age50' in df.columns and 'met50' in df.columns:
        low_alpha = (df['age50'] < 8) & (df['met50'] > -0.5)
        high_alpha = (df['age50'] > 8) | (df['met50'] < -0.5)
        df['population'] = 'unknown'
        df.loc[low_alpha, 'population'] = 'thin_disk'
        df.loc[high_alpha, 'population'] = 'thick_disk'
    elif 'mh_gspphot' in df.columns:
        # Fallback to Gaia metallicity
        thin = df['mh_gspphot'] > -0.4
        thick = df['mh_gspphot'] <= -0.4
        df['population'] = 'unknown'
        df.loc[thin, 'population'] = 'thin_disk_candidate'
        df.loc[thick, 'population'] = 'thick_disk_candidate'
        
    return df


# =============================================================================
# BANYAN Σ MEMBERSHIP (Gagné+2018)
# =============================================================================

def query_banyan_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query BANYAN Σ for young stellar association membership probability.
    
    Requires: ra, dec, pmra, pmdec, parallax columns.
    Optional: radial_velocity for better constraints.
    
    Returns dataframe with banyan_field_prob and banyan_best_assoc columns.
    """
    if df.empty:
        return df
    df = df.copy()
    
    # Check for required columns
    required = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: BANYAN Σ requires columns: {missing}")
        df['banyan_field_prob'] = np.nan
        df['banyan_best_assoc'] = ''
        return df
    
    field_probs = []
    best_assocs = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BANYAN Σ"):
        try:
            ra = float(row['ra'])
            dec = float(row['dec'])
            pmra = float(row['pmra'])
            pmdec = float(row['pmdec'])
            plx = float(row['parallax'])
            
            if not all(np.isfinite([ra, dec, pmra, pmdec, plx])):
                field_probs.append(np.nan)
                best_assocs.append('')
                continue
            
            # Optional RV
            rv = row.get('radial_velocity', np.nan)
            rv = float(rv) if np.isfinite(rv) else None
            
            if not hasattr(banyan_sigma_pkg, "banyan_sigma"):
                raise RuntimeError("banyan_sigma package does not expose banyan_sigma()")

            result = banyan_sigma_pkg.banyan_sigma(
                ra=ra, dec=dec,
                pmra=pmra, pmdec=pmdec,
                plx=plx, rv=rv
            )
            
            field_probs.append(float(result.get('field', np.nan)))
            
            # Best association (highest non-field probability)
            assoc_probs = {k: v for k, v in result.items() if k != 'field'}
            if assoc_probs:
                best = max(assoc_probs, key=assoc_probs.get)
                best_assocs.append(best if assoc_probs[best] > BANYAN_MIN_ASSOC_PROB else '')
            else:
                best_assocs.append('')
                
        except Exception as e:
            field_probs.append(np.nan)
            best_assocs.append('')
    
    df['banyan_field_prob'] = field_probs
    df['banyan_best_assoc'] = best_assocs
    
    print(f"BANYAN Σ: {(np.array(field_probs) < 0.99).sum()} sources with < 99% field probability")
    return df


# =============================================================================
# IPHAS Hα CROSSMATCH (Barentsen+2014)
# =============================================================================

def crossmatch_iphas(df: pd.DataFrame, max_sep_arcsec: float = IPHAS_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to IPHAS DR2 for Hα emission detection.
    
    Uses CDS XMatch for efficient batch crossmatching. Returns (r-Hα) and (r-i) colors.
    """
    if df.empty:
        return df
    df = df.copy()
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: IPHAS crossmatch requires ra, dec columns")
        df['iphas_r_ha'] = np.nan
        df['iphas_r_i'] = np.nan
        df['iphas_ha_excess'] = False
        return df
    
    # Initialize output columns
    df['iphas_r_ha'] = np.nan
    df['iphas_r_i'] = np.nan
    df['iphas_ha_excess'] = False
    
    # Prepare source table with unique index for matching back
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return df
    
    # Create astropy table for XMatch
    source_table = Table()
    source_table['_idx'] = np.where(valid_mask)[0]
    source_table['ra'] = df.loc[valid_mask, 'ra'].values
    source_table['dec'] = df.loc[valid_mask, 'dec'].values
    
    print(f"Running IPHAS XMatch for {len(source_table)} sources...")
    
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:II/321/iphas2',
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RAJ2000', colDec2='DEJ2000'
        )
        
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            
            # For sources with multiple matches, keep closest
            if 'angDist' in result_df.columns:
                result_df = result_df.sort_values('angDist').drop_duplicates(subset='_idx', keep='first')
            else:
                result_df = result_df.drop_duplicates(subset='_idx', keep='first')
            
            # Compute colors
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                r_mag = float(row.get('r', np.nan))
                ha_mag = float(row.get('Ha', np.nan))
                i_mag = float(row.get('i', np.nan))
                
                if np.isfinite(r_mag) and np.isfinite(ha_mag):
                    r_ha = r_mag - ha_mag
                    df.at[df.index[idx], 'iphas_r_ha'] = r_ha
                    df.at[df.index[idx], 'iphas_ha_excess'] = r_ha > IPHAS_HA_EXCESS_THRESHOLD
                
                if np.isfinite(r_mag) and np.isfinite(i_mag):
                    df.at[df.index[idx], 'iphas_r_i'] = r_mag - i_mag
            
            matched = len(result_df)
            ha_excess_count = (df['iphas_ha_excess'] == True).sum()
            print(f"IPHAS: {matched}/{len(df)} matched, {ha_excess_count} with Hα excess")
        else:
            print("IPHAS: No matches found")
            
    except Exception as e:
        print(f"IPHAS XMatch error: {e}")
    
    return df


# =============================================================================
# STAR-FORMING REGION PROXIMITY (Prisinzano+2022)
# =============================================================================

def check_sfr_proximity(df: pd.DataFrame, max_dist_kpc: float = SFR_MAX_DIST_KPC) -> pd.DataFrame:
    """
    Check proximity to known star-forming regions.
    
    Uses Prisinzano+2022 Table 1 coordinates and distances.
    Vectorized implementation for efficient processing of large catalogs.
    """
    if df.empty:
        return df
    df = df.copy()
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: SFR proximity check requires ra, dec columns")
        df['near_sfr'] = False
        df['sfr_name'] = ''
        return df
    
    # Get distance column
    if 'distance_gspphot' in df.columns:
        dist_pc = df['distance_gspphot'].values.astype(float)
    elif 'parallax' in df.columns:
        plx = df['parallax'].values.astype(float)
        dist_pc = np.where((plx > 0) & np.isfinite(plx), 1000.0 / plx, np.nan)
    else:
        dist_pc = np.full(len(df), np.nan)
    
    # Initialize output arrays
    near_sfr = np.zeros(len(df), dtype=bool)
    sfr_names = np.full(len(df), '', dtype=object)
    
    # Build source coordinates once (vectorized)
    valid_coords = df['ra'].notna() & df['dec'].notna()
    if not valid_coords.any():
        df['near_sfr'] = near_sfr
        df['sfr_name'] = sfr_names
        return df
    
    source_coords = SkyCoord(
        ra=df.loc[valid_coords, 'ra'].values * u.deg,
        dec=df.loc[valid_coords, 'dec'].values * u.deg
    )
    valid_indices = np.where(valid_coords)[0]
    valid_dist_pc = dist_pc[valid_coords]
    
    print(f"Checking SFR proximity for {len(source_coords)} sources...")
    
    # Check each SFR with vectorized separation
    for sfr in SFR_CATALOG:
        sfr_coord = SkyCoord(ra=sfr['ra'] * u.deg, dec=sfr['dec'] * u.deg)
        
        # Vectorized angular separation
        seps = source_coords.separation(sfr_coord).deg
        
        # Sources within angular radius
        within_radius = seps < sfr['radius_deg']
        
        if not within_radius.any():
            continue
        
        # Check distance consistency for sources within radius
        for local_idx in np.where(within_radius)[0]:
            global_idx = valid_indices[local_idx]
            
            # Skip if already matched to a closer SFR
            if near_sfr[global_idx]:
                continue
            
            d = valid_dist_pc[local_idx]
            if np.isfinite(d):
                # Distance within 50% of SFR distance
                if abs(d - sfr['dist_pc']) / sfr['dist_pc'] < SFR_DIST_TOLERANCE_FRACTION:
                    near_sfr[global_idx] = True
                    sfr_names[global_idx] = sfr['name']
            else:
                # No distance info, flag based on position only
                near_sfr[global_idx] = True
                sfr_names[global_idx] = sfr['name'] + ' (pos only)'
    
    df['near_sfr'] = near_sfr
    df['sfr_name'] = sfr_names
    
    print(f"SFR proximity: {near_sfr.sum()}/{len(df)} sources near star-forming regions")
    return df


# =============================================================================
# OPEN CLUSTER MEMBERSHIP (Cantat-Gaudin+2020)
# =============================================================================

def crossmatch_open_clusters(df: pd.DataFrame, max_sep_arcsec: float = CLUSTER_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to Cantat-Gaudin+2020 open cluster catalog.
    
    Catalog contains 1867 clusters with ages and distances.
    Uses CDS XMatch for efficient batch crossmatching.
    """
    if df.empty:
        return df
    df = df.copy()
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: Open cluster crossmatch requires ra, dec columns")
        df['cluster_name'] = ''
        df['cluster_age_myr'] = np.nan
        df['cluster_dist_pc'] = np.nan
        return df
    
    # Initialize output columns
    df['cluster_name'] = ''
    df['cluster_age_myr'] = np.nan
    df['cluster_dist_pc'] = np.nan
    
    # Prepare source table with unique index for matching back
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return df
    
    # Create astropy table for XMatch
    source_table = Table()
    source_table['_idx'] = np.where(valid_mask)[0]
    source_table['ra'] = df.loc[valid_mask, 'ra'].values
    source_table['dec'] = df.loc[valid_mask, 'dec'].values
    
    print(f"Running open cluster XMatch for {len(source_table)} sources...")
    
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:J/A+A/640/A1/members',  # Cantat-Gaudin+2020 members
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RAJ2000', colDec2='DEJ2000'
        )
        
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            
            # For sources with multiple matches, keep closest
            if 'angDist' in result_df.columns:
                result_df = result_df.sort_values('angDist').drop_duplicates(subset='_idx', keep='first')
            else:
                result_df = result_df.drop_duplicates(subset='_idx', keep='first')
            
            # Assign cluster properties
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                df.at[df.index[idx], 'cluster_name'] = str(row.get('Cluster', ''))
                
                age_val = row.get('Age', np.nan)
                if age_val is not None:
                    df.at[df.index[idx], 'cluster_age_myr'] = float(age_val)
                
                dist_val = row.get('Dist', np.nan)
                if dist_val is not None:
                    df.at[df.index[idx], 'cluster_dist_pc'] = float(dist_val)
            
            matched = len(result_df)
            print(f"Cluster crossmatch: {matched}/{len(df)} sources in open clusters")
        else:
            print("Cluster crossmatch: No matches found")
            
    except Exception as e:
        print(f"Cluster XMatch error: {e}")
    
    return df


# =============================================================================
# unWISE/unTimely IR VARIABILITY
# =============================================================================

def query_unwise_variability(df: pd.DataFrame, max_sep_arcsec: float = UNWISE_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Query unWISE/unTimely for mid-IR variability (Meisner+2023).
    
    Computes W1 and W2 variability z-scores from time-domain photometry.
    """
    if df.empty:
        return df
    df = df.copy()
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: unWISE query requires ra, dec columns")
        df['unwise_w1_zscore'] = np.nan
        df['unwise_w2_zscore'] = np.nan
        df['unwise_w1_var'] = False
        return df
    
    w1_zscores = []
    w2_zscores = []
    w1_var = []
    
    # unTimely API endpoint
    base_url = UNTIMELY_API_URL
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="unWISE variability"):
        try:
            # Query unTimely catalog
            params = {
                'catalog': 'untimely',
                'spatial': 'cone',
                'radius': max_sep_arcsec,
                'radunits': 'arcsec',
                'objstr': f"{row['ra']} {row['dec']}",
                'outfmt': '1',  # JSON
            }
            
            resp = requests.get(base_url, params=params, timeout=UNWISE_TIMEOUT_SECONDS)
            
            if resp.status_code == 200:
                data = resp.json()
                
                if data and len(data) > 0:
                    # Filter by quality (paper criteria)
                    good = [d for d in data
                            if d.get('fracflux', 0) > UNWISE_FRACFLUX_MIN
                            and d.get('qf', 0) > UNWISE_QF_MIN]
                    
                    if good:
                        w1_mags = [d['w1mpro'] for d in good if 'w1mpro' in d]
                        w2_mags = [d['w2mpro'] for d in good if 'w2mpro' in d]
                        
                        if len(w1_mags) >= 3:
                            w1_std = np.std(w1_mags)
                            w1_med = np.median(w1_mags)
                            # Estimate expected scatter from magnitude
                            expected_scatter = UNWISE_EXPECTED_SCATTER_BASE + UNWISE_EXPECTED_SCATTER_SLOPE * max(0, w1_med - UNWISE_EXPECTED_SCATTER_MAG_REF)
                            z = w1_std / expected_scatter
                            w1_zscores.append(z)
                            w1_var.append(z > UNWISE_VARIABILITY_ZSCORE)
                        else:
                            w1_zscores.append(np.nan)
                            w1_var.append(False)
                        
                        if len(w2_mags) >= 3:
                            w2_std = np.std(w2_mags)
                            w2_med = np.median(w2_mags)
                            expected_scatter = UNWISE_EXPECTED_SCATTER_BASE + UNWISE_EXPECTED_SCATTER_SLOPE * max(0, w2_med - UNWISE_EXPECTED_SCATTER_MAG_REF)
                            w2_zscores.append(w2_std / expected_scatter)
                        else:
                            w2_zscores.append(np.nan)
                    else:
                        w1_zscores.append(np.nan)
                        w2_zscores.append(np.nan)
                        w1_var.append(False)
                else:
                    w1_zscores.append(np.nan)
                    w2_zscores.append(np.nan)
                    w1_var.append(False)
            else:
                w1_zscores.append(np.nan)
                w2_zscores.append(np.nan)
                w1_var.append(False)
                
        except Exception:
            w1_zscores.append(np.nan)
            w2_zscores.append(np.nan)
            w1_var.append(False)
    
    df['unwise_w1_zscore'] = w1_zscores
    df['unwise_w2_zscore'] = w2_zscores
    df['unwise_w1_var'] = w1_var
    
    n_var = sum(w1_var)
    print(f"unWISE: {n_var}/{len(df)} sources with W1 variability z-score > 3")
    return df


def _set_module_state(
    df: pd.DataFrame,
    module: str,
    status: str,
    error: str = "",
) -> pd.DataFrame:
    """Annotate module execution status on all rows."""
    out = df.copy()
    out[f"char_status_{module}"] = status
    out[f"char_error_{module}"] = error
    return out


def _run_optional_module(
    df: pd.DataFrame,
    *,
    module: str,
    enabled: bool,
    description: str,
    func,
    **kwargs,
) -> pd.DataFrame:
    """Run optional characterize module in fail-open mode."""
    if not enabled:
        return _set_module_state(df, module, "skipped", "")

    print(description)
    try:
        out = func(df, **kwargs)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{module} did not return a pandas DataFrame")
        return _set_module_state(out, module, "ok", "")
    except Exception as e:
        msg = str(e)
        print(f"Warning: characterize module '{module}' failed: {msg}")
        return _set_module_state(df, module, "error", msg)


def characterize_candidates_df(
    df: pd.DataFrame,
    *,
    crossmatch: Path = VSX_CROSSMATCH_PATH,
    chunk_size: int = GAIA_CHUNK_SIZE,
    cache: Path = GAIA_CACHE_FILE,
    dust: bool = False,
    starhorse: str | None = None,
    run_banyan: bool = True,
    run_iphas: bool = True,
    run_sfr: bool = True,
    run_clusters: bool = True,
    run_unwise: bool = True,
) -> pd.DataFrame:
    """Characterize candidates and return an enriched dataframe."""
    if "asas_sn_id" not in df.columns:
        print("Warning: characterize skipped: missing 'asas_sn_id' column")
        return df

    df_in = df.copy()
    xmatch_path = crossmatch.expanduser()

    if not xmatch_path.exists():
        print(f"Warning: Crossmatch file {xmatch_path} not found")
        if "gaia_id" in df_in.columns:
            print("Proceeding with existing gaia_id column")
            df_merged = df_in
        else:
            return df_in
    else:
        print(f"Loading crossmatch file {xmatch_path}...")
        xmatch_cols = ["asas_sn_id", "gaia_id", "tmass_id", "allwise_id"]
        try:
            header = pd.read_csv(xmatch_path, nrows=0).columns
            use_cols = ["asas_sn_id"] + [c for c in xmatch_cols if c in header and c != "asas_sn_id"]
            df_xmatch = pd.read_csv(xmatch_path, usecols=use_cols, dtype=str)
            df_in["asas_sn_id"] = df_in["asas_sn_id"].astype(str)
            df_xmatch["asas_sn_id"] = df_xmatch["asas_sn_id"].astype(str)
            df_merged = df_in.merge(df_xmatch, on="asas_sn_id", how="left")
            print(f"Merged {len(df_merged)} rows")
        except Exception as e:
            print(f"Warning: characterize crossmatch read failed: {e}")
            return df_in

    if "gaia_id" not in df_merged.columns:
        print("Warning: characterize skipped Gaia query: gaia_id not present")
        return df_merged

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)

    missing_gaia = df_merged["gaia_id"].isna().sum()
    print(f"Found Gaia IDs for {len(df_merged) - missing_gaia}/{len(df_merged)} sources")
    gaia_ids = df_merged["gaia_id"].dropna().unique().tolist()

    if not gaia_ids:
        print("Warning: characterize found no Gaia IDs")
        return df_merged

    print(f"Querying Gaia DR3 for {len(gaia_ids)} sources...")
    gaia_df = query_gaia_by_ids(
        gaia_ids,
        chunk_size=chunk_size,
        cache_file=str(cache) if cache else None,
    )
    if gaia_df.empty:
        print("Warning: characterize Gaia query returned no rows")
        return df_merged

    df_merged["gaia_id"] = df_merged["gaia_id"].astype(str)
    gaia_df["source_id"] = gaia_df["source_id"].astype(str)

    print("Merging Gaia results...")
    df_char = df_merged.merge(gaia_df, left_on="gaia_id", right_on="source_id", how="left", suffixes=("", "_gaia"))

    # Compute Galactic coordinates from RA/Dec
    _ra_col = "ra" if "ra" in df_char.columns else ("ra_deg" if "ra_deg" in df_char.columns else None)
    _dec_col = "dec" if "dec" in df_char.columns else ("dec_deg" if "dec_deg" in df_char.columns else None)
    if _ra_col and _dec_col:
        _gc_mask = np.isfinite(df_char[_ra_col].astype(float)) & np.isfinite(df_char[_dec_col].astype(float))
        if _gc_mask.any():
            _gc_coords = SkyCoord(
                ra=df_char.loc[_gc_mask, _ra_col].values * u.deg,
                dec=df_char.loc[_gc_mask, _dec_col].values * u.deg,
                frame="icrs",
            )
            df_char.loc[_gc_mask, "gal_l"] = _gc_coords.galactic.l.deg
            df_char.loc[_gc_mask, "gal_b"] = _gc_coords.galactic.b.deg
        if "gal_l" not in df_char.columns:
            df_char["gal_l"] = np.nan
            df_char["gal_b"] = np.nan
    else:
        df_char["gal_l"] = np.nan
        df_char["gal_b"] = np.nan

    print("Classifying Galactic populations...")
    df_char = classify_galactic_population(df_char)
    df_char = _set_module_state(df_char, "population", "ok", "")

    if starhorse:
        print("Loading StarHorse catalog for ages...")
        try:
            use_tap_query = not Path(starhorse).exists() if starhorse != "tap" else True
            sh_df = query_starhorse_by_ids(
                gaia_ids,
                starhorse_file=starhorse if not use_tap_query else None,
                use_tap=use_tap_query,
            )
            if not sh_df.empty:
                df_char = df_char.merge(sh_df, on="source_id", how="left", suffixes=("", "_sh"))
                if "age50" in df_char.columns:
                    df_char = classify_galactic_population(df_char)
            df_char = _set_module_state(df_char, "starhorse", "ok", "")
        except Exception as e:
            msg = str(e)
            print(f"Warning: characterize module 'starhorse' failed: {msg}")
            df_char = _set_module_state(df_char, "starhorse", "error", msg)
    else:
        df_char = _set_module_state(df_char, "starhorse", "skipped", "")

    df_char = _run_optional_module(
        df_char,
        module="dust",
        enabled=dust,
        description="Computing 3D dust extinction (dustmaps3d)...",
        func=get_dust_extinction,
    )

    if "tmass_j" in df_char.columns:
        print("Classifying YSOs...")
        try:
            df_char = classify_yso(df_char)
            df_char = _set_module_state(df_char, "yso", "ok", "")
        except Exception as e:
            msg = str(e)
            print(f"Warning: characterize module 'yso' failed: {msg}")
            df_char = _set_module_state(df_char, "yso", "error", msg)
    else:
        print("Warning: IR photometry columns not found for YSO classification")
        df_char = _set_module_state(df_char, "yso", "skipped", "")

    df_char = _run_optional_module(
        df_char,
        module="banyan",
        enabled=run_banyan,
        description="Running BANYAN Σ membership checks...",
        func=query_banyan_sigma,
    )
    df_char = _run_optional_module(
        df_char,
        module="iphas",
        enabled=run_iphas,
        description="Running IPHAS H-alpha crossmatch...",
        func=crossmatch_iphas,
    )
    df_char = _run_optional_module(
        df_char,
        module="sfr",
        enabled=run_sfr,
        description="Checking star-forming region proximity...",
        func=check_sfr_proximity,
    )
    df_char = _run_optional_module(
        df_char,
        module="clusters",
        enabled=run_clusters,
        description="Running open cluster crossmatch...",
        func=crossmatch_open_clusters,
    )
    df_char = _run_optional_module(
        df_char,
        module="unwise",
        enabled=run_unwise,
        description="Querying unWISE/unTimely variability...",
        func=query_unwise_variability,
    )

    return df_char


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-wavelength characterization for dipper candidates")
    parser.add_argument("--input", type=Path, required=True, help="Input events CSV/Parquet (must have asas_sn_id)")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet")
    parser.add_argument("--crossmatch", type=Path,
                        default=VSX_CROSSMATCH_PATH,
                        help="Path to ASAS-SN x VSX crossmatch CSV (must contain asas_sn_id and gaia_id)")
    parser.add_argument("--chunk-size", type=int, default=GAIA_CHUNK_SIZE, help="Gaia query chunk size")
    parser.add_argument("--cache", type=Path, default=GAIA_CACHE_FILE, help="Cache file for Gaia queries")
    parser.add_argument("--dust", action="store_true", help="Enable dustmaps3d 3D extinction query")
    parser.add_argument("--starhorse", type=str, default=None, help="StarHorse stellar ages/masses: 'tap' for remote TAP query (recommended), or path to local catalog file")
    parser.add_argument("--no-characterize-banyan", dest="characterize_banyan", action="store_false", help="Disable BANYAN Sigma enrichment")
    parser.add_argument("--no-characterize-iphas", dest="characterize_iphas", action="store_false", help="Disable IPHAS enrichment")
    parser.add_argument("--no-characterize-sfr", dest="characterize_sfr", action="store_false", help="Disable star-forming-region enrichment")
    parser.add_argument("--no-characterize-clusters", dest="characterize_clusters", action="store_false", help="Disable open-cluster enrichment")
    parser.add_argument("--no-characterize-unwise", dest="characterize_unwise", action="store_false", help="Disable unWISE variability enrichment")
    parser.set_defaults(
        characterize_banyan=True,
        characterize_iphas=True,
        characterize_sfr=True,
        characterize_clusters=True,
        characterize_unwise=True,
    )
    
    args = parser.parse_args()
    
    # Load input
    print(f"Loading {args.input}...")
    if str(args.input).endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
        
    df_char = characterize_candidates_df(
        df,
        crossmatch=args.crossmatch,
        chunk_size=args.chunk_size,
        cache=args.cache,
        dust=args.dust,
        starhorse=args.starhorse,
        run_banyan=args.characterize_banyan,
        run_iphas=args.characterize_iphas,
        run_sfr=args.characterize_sfr,
        run_clusters=args.characterize_clusters,
        run_unwise=args.characterize_unwise,
    )
    
    # Save results
    print("Saving results...")
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(output_path).endswith(".parquet"):
        df_char.to_parquet(output_path, index=False, compression="zstd")
    else:
        df_char.to_csv(output_path, index=False)
        
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
