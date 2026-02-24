"""
LTV Catalog Crossmatches — Optimized for Scale.

Implements catalog crossmatches from the paper:
- LOCAL CATALOG: Pre-matched ASAS-SN × VSX file (Gaia DR3 + VSX, no API)
- Gaia Alerts (transient alerts) — API
- MILLIQUAS (AGN catalog) — API
- SIMBAD (classifications) — API

Optimized for scale with:
- Local catalog for Gaia/VSX (eliminates 2 API query types)
- Batch TAP queries with table upload
- Parallel processing via ThreadPoolExecutor
- Chunked processing for memory efficiency
- Progress bars throughout

NOTE: These should run AFTER filtering to reduce the dataset from 17M to ~36K.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from tqdm.auto import tqdm

from malca.config.config_ltv import (
    LTV_MATCH_RADIUS_ARCSEC,
    LTV_WORKERS,
    LTV_CROSSMATCH_CHUNK_SIZE,
    VIZIER_TAP_URL,
    SIMBAD_TAP_URL,
)
from malca.config.config_paths import VSX_CROSSMATCH_PATH


# =============================================================================
# LOCAL CATALOG (Gaia DR3 + VSX — no API queries needed)
# =============================================================================

DEFAULT_CATALOG_PATH = VSX_CROSSMATCH_PATH

GAIA_COLUMN_MAP = {
    "gaia_id": "gaia_source_id",
    "plx": "gaia_parallax",
    "pm_ra": "gaia_pmra",
    "pm_dec": "gaia_pmdec",
    "gaia_mag": "gaia_phot_g_mean_mag",
    "gaia_b_mag": "gaia_bp_mag",
    "gaia_r_mag": "gaia_rp_mag",
    "gaia_eff_temp": "gaia_teff",
}

VSX_COLUMN_MAP = {
    "id_vsx": "vsx_oid",
    "name": "vsx_name",
    "class": "vsx_type",
    "period": "vsx_period",
    "mag_max": "vsx_mag_max",
    "mag_min": "vsx_mag_min",
    "spectral_type": "vsx_spectral_type",
}

_cached_catalog = None


def load_local_catalog(
    path: str | Path | None = None,
    *,
    cache: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load pre-matched ASAS-SN × VSX catalog (~99K sources with Gaia/VSX data)."""
    global _cached_catalog
    
    if cache and _cached_catalog is not None:
        return _cached_catalog
    
    if path is None:
        path = DEFAULT_CATALOG_PATH
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Local catalog not found: {path}")
    
    if verbose:
        print(f"Loading local catalog from {path}...")
    
    df = pd.read_csv(path)
    
    rename_map = {**GAIA_COLUMN_MAP, **VSX_COLUMN_MAP}
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    if "gaia_source_id" in df.columns:
        df["gaia_source_id"] = df["gaia_source_id"].astype(str)
    
    if verbose:
        print(f"Loaded {len(df):,} sources with Gaia/VSX data")
    
    if cache:
        _cached_catalog = df
    
    return df


def merge_local_catalog(
    df: pd.DataFrame,
    *,
    catalog_path: str | Path | None = None,
    id_column: str = "ASAS-SN ID",
    catalog_id_column: str = "asas_sn_id",
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge local catalog by ASAS-SN ID (fast ID-based join)."""
    if id_column not in df.columns:
        if verbose:
            print(f"Warning: '{id_column}' not found, skipping local catalog merge")
        return df
    
    catalog = load_local_catalog(catalog_path, verbose=verbose)
    
    if catalog_id_column not in catalog.columns:
        if verbose:
            print(f"Warning: '{catalog_id_column}' not found in catalog")
        return df
    
    merge_cols = [catalog_id_column]
    merge_cols.extend([v for v in GAIA_COLUMN_MAP.values() if v in catalog.columns])
    merge_cols.extend([v for v in VSX_COLUMN_MAP.values() if v in catalog.columns])
    merge_cols = list(set(merge_cols))
    
    catalog_subset = catalog[merge_cols].copy()
    catalog_subset = catalog_subset.rename(columns={catalog_id_column: id_column})
    
    df = df.copy()
    df[id_column] = df[id_column].astype(str)
    catalog_subset[id_column] = catalog_subset[id_column].astype(str)
    
    n_before = len(df)
    df = df.merge(catalog_subset, on=id_column, how="left", suffixes=("", "_local"))
    
    if verbose:
        n_matched = df["gaia_source_id"].notna().sum() if "gaia_source_id" in df.columns else 0
        print(f"[merge_local_catalog] Matched {n_matched}/{n_before} from local catalog")
    
    return df


def crossmatch_from_local(
    df: pd.DataFrame,
    *,
    catalog_path: str | Path | None = None,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    verbose: bool = False,
) -> pd.DataFrame:
    """Crossmatch by RA/Dec against local catalog (coordinate-based)."""
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec not found, skipping local catalog crossmatch")
        return df
    
    catalog = load_local_catalog(catalog_path, verbose=verbose)
    
    if "ra_deg" not in catalog.columns or "dec_deg" not in catalog.columns:
        if verbose:
            print("Warning: RA/Dec not found in catalog")
        return df
    
    if verbose:
        print(f"[crossmatch_from_local] Matching {len(df)} sources...")
    
    df_coords = SkyCoord(ra=df[ra_column].values * u.deg, dec=df[dec_column].values * u.deg, frame="icrs")
    catalog_coords = SkyCoord(ra=catalog["ra_deg"].values * u.deg, dec=catalog["dec_deg"].values * u.deg, frame="icrs")
    
    idx, sep, _ = match_coordinates_sky(df_coords, catalog_coords)
    matched = sep.arcsec <= match_radius_arcsec
    
    df = df.copy()
    for col in GAIA_COLUMN_MAP.values():
        if col in catalog.columns:
            df[col] = np.nan
    for col in VSX_COLUMN_MAP.values():
        if col in catalog.columns:
            df[col] = None if col in ["vsx_name", "vsx_type", "vsx_spectral_type"] else np.nan
    df["local_catalog_sep_arcsec"] = np.nan
    
    for i, (cat_idx, is_matched) in enumerate(zip(idx, matched)):
        if is_matched:
            for col in list(GAIA_COLUMN_MAP.values()) + list(VSX_COLUMN_MAP.values()):
                if col in catalog.columns:
                    df.iloc[i, df.columns.get_loc(col)] = catalog.iloc[cat_idx][col]
            df.iloc[i, df.columns.get_loc("local_catalog_sep_arcsec")] = sep[i].arcsec
    
    if verbose:
        print(f"[crossmatch_from_local] Matched {matched.sum()}/{len(df)} sources")
    
    return df


def clear_catalog_cache():
    """Clear the cached local catalog from memory."""
    global _cached_catalog
    _cached_catalog = None


# =============================================================================
# BATCH TAP QUERY UTILITIES
# =============================================================================

def _batch_tap_crossmatch(
    coords_df: pd.DataFrame,
    *,
    tap_service: str,
    catalog_table: str,
    select_cols: str,
    ra_col: str = "RAJ2000",
    dec_col: str = "DEJ2000",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
    desc: str = "TAP crossmatch",
) -> pd.DataFrame:
    """
    Generic batch TAP crossmatch using coordinate upload.
    
    For catalogs that support TAP uploads, this is much faster than row-by-row.
    """
    from astroquery.utils.tap.core import TapPlus
    
    if coords_df.empty:
        return pd.DataFrame()
    
    results = []
    chunks = [coords_df.iloc[i:i+chunk_size] for i in range(0, len(coords_df), chunk_size)]
    
    def process_chunk(chunk_df):
        try:
            tap = TapPlus(url=tap_service)
            upload_table = Table.from_pandas(chunk_df[["_idx", "ra", "dec"]])
            
            query = f"""
            SELECT 
                u._idx as _idx,
                {select_cols},
                DISTANCE(POINT('ICRS', c.{ra_col}, c.{dec_col}), POINT('ICRS', u.ra, u.dec)) * 3600.0 as sep_arcsec
            FROM TAP_UPLOAD.upload_table AS u
            JOIN {catalog_table} AS c
            ON 1=CONTAINS(
                POINT('ICRS', c.{ra_col}, c.{dec_col}),
                CIRCLE('ICRS', u.ra, u.dec, {match_radius_arcsec / 3600.0})
            )
            """
            
            job = tap.launch_job_async(
                query,
                upload_resource=upload_table,
                upload_table_name="upload_table",
                verbose=False,
            )
            result = job.get_results()
            return result.to_pandas() if result else pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"TAP query error: {e}")
            return pd.DataFrame()
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, disable=not verbose):
            result = future.result()
            if not result.empty:
                results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)


def crossmatch_tap_catalog(
    df: pd.DataFrame,
    *,
    tap_service: str,
    catalog_table: str,
    select_cols: str,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    ra_col: str = "RAJ2000",
    dec_col: str = "DEJ2000",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    col_prefix: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generic TAP crossmatch helper.

    Use this for VizieR/TAP catalogs (ATLAS, ZTF, WISE, etc.) once you have
    the TAP service URL and table name.
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for TAP crossmatch")
        return df

    coords_df = pd.DataFrame({
        "_idx": df.index,
        "ra": df[ra_column].values,
        "dec": df[dec_column].values,
    })

    result = _batch_tap_crossmatch(
        coords_df,
        tap_service=tap_service,
        catalog_table=catalog_table,
        select_cols=select_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        desc=f"TAP crossmatch: {catalog_table}",
    )

    if result.empty:
        return df

    # Keep closest match per source
    result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")

    # Apply optional prefix to avoid column collisions
    if col_prefix:
        rename_map = {}
        for col in result.columns:
            if col not in ["_idx", "sep_arcsec"]:
                rename_map[col] = f"{col_prefix}{col}"
        if rename_map:
            result = result.rename(columns=rename_map)

    df_out = df.copy()
    df_out["_idx"] = df_out.index
    df_out = df_out.merge(result, on="_idx", how="left")
    df_out = df_out.drop(columns=["_idx"])

    return df_out





# =============================================================================
# MILLIQUAS CROSSMATCH (parallel VizieR)
# =============================================================================

def crossmatch_milliquas(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Crossmatch to MILLIQUAS v8 via batch VizieR TAP upload.

    Adds columns: milliquas_name, milliquas_type, milliquas_z, milliquas_sep_arcsec
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for MILLIQUAS crossmatch")
        return df

    df = df.copy()
    df["milliquas_name"] = None
    df["milliquas_type"] = None
    df["milliquas_z"] = np.nan
    df["milliquas_sep_arcsec"] = np.nan

    coords_df = pd.DataFrame({
        "_idx": df.index,
        "ra": df[ra_column].values,
        "dec": df[dec_column].values,
    })

    if verbose:
        print(f"[crossmatch_milliquas] Querying {len(df)} sources via TAP...")

    result = _batch_tap_crossmatch(
        coords_df,
        tap_service=VIZIER_TAP_URL,
        catalog_table='"VII/294/milliqua"',
        select_cols='c."Name", c."Type", c.z',
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        desc="MILLIQUAS TAP",
    )

    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "milliquas_name"] = row.get("Name")
                df.loc[idx, "milliquas_type"] = row.get("Type")
                df.loc[idx, "milliquas_z"] = float(row["z"]) if pd.notna(row.get("z")) else np.nan
                df.loc[idx, "milliquas_sep_arcsec"] = row["sep_arcsec"]

    if verbose:
        n_matched = df["milliquas_name"].notna().sum()
        print(f"[crossmatch_milliquas] Matched {n_matched}/{len(df)}")

    return df


# =============================================================================
# GAIA ALERTS CROSSMATCH (parallel VizieR)
# =============================================================================

def crossmatch_gaia_alerts(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Crossmatch to Gaia Alerts via batch VizieR TAP upload.

    Adds columns: gaia_alert_name, gaia_alert_class, gaia_alert_sep_arcsec
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for Gaia Alerts crossmatch")
        return df

    df = df.copy()
    df["gaia_alert_name"] = None
    df["gaia_alert_class"] = None
    df["gaia_alert_sep_arcsec"] = np.nan

    coords_df = pd.DataFrame({
        "_idx": df.index,
        "ra": df[ra_column].values,
        "dec": df[dec_column].values,
    })

    if verbose:
        print(f"[crossmatch_gaia_alerts] Querying {len(df)} sources via TAP...")

    result = _batch_tap_crossmatch(
        coords_df,
        tap_service=VIZIER_TAP_URL,
        catalog_table='"I/358/vari"',
        select_cols='c."Name", c."Class"',
        ra_col="RA_ICRS",
        dec_col="DE_ICRS",
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        desc="Gaia Alerts TAP",
    )

    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "gaia_alert_name"] = row.get("Name")
                df.loc[idx, "gaia_alert_class"] = row.get("Class")
                df.loc[idx, "gaia_alert_sep_arcsec"] = row["sep_arcsec"]

    if verbose:
        n_matched = df["gaia_alert_name"].notna().sum()
        print(f"[crossmatch_gaia_alerts] Matched {n_matched}/{len(df)}")

    return df


# =============================================================================
# SIMBAD CLASSIFICATION (parallel)
# =============================================================================

def query_simbad_classification(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    chunk_size: int = LTV_CROSSMATCH_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query SIMBAD for classifications via batch TAP upload.

    Adds columns: simbad_main_id, simbad_otype, simbad_sp_type, simbad_sep_arcsec
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for SIMBAD query")
        return df

    df = df.copy()
    df["simbad_main_id"] = None
    df["simbad_otype"] = None
    df["simbad_sp_type"] = None
    df["simbad_sep_arcsec"] = np.nan

    coords_df = pd.DataFrame({
        "_idx": df.index,
        "ra": df[ra_column].values,
        "dec": df[dec_column].values,
    })

    if verbose:
        print(f"[query_simbad_classification] Querying {len(df)} sources via TAP...")

    result = _batch_tap_crossmatch(
        coords_df,
        tap_service=SIMBAD_TAP_URL,
        catalog_table="basic",
        select_cols="c.main_id, c.otype, c.sp_type",
        ra_col="ra",
        dec_col="dec",
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        desc="SIMBAD TAP",
    )

    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "simbad_main_id"] = row.get("main_id")
                df.loc[idx, "simbad_otype"] = row.get("otype")
                df.loc[idx, "simbad_sp_type"] = row.get("sp_type")
                df.loc[idx, "simbad_sep_arcsec"] = row["sep_arcsec"]

    if verbose:
        n_matched = df["simbad_main_id"].notna().sum()
        print(f"[query_simbad_classification] Matched {n_matched}/{len(df)}")

    return df


# =============================================================================
# COMBINED CROSSMATCH
# =============================================================================

def crossmatch_all_catalogs(
    df: pd.DataFrame,
    *,
    # Local catalog options (eliminates Gaia DR3 + VSX API queries)
    use_local_catalog: bool = True,
    local_catalog_path: str | None = None,
    # API query options (only for data NOT in local catalog)
    include_gaia_dr3: bool = True,
    include_gaia_alerts: bool = True,
    include_vsx: bool = True,
    include_milliquas: bool = True,
    include_simbad: bool = True,
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run all catalog crossmatches with optimized processing.
    
    Optimizations:
    - LOCAL CATALOG: Uses pre-matched ASAS-SN × VSX file for Gaia DR3 & VSX
      data, ELIMINATING those API queries entirely (~99K sources available)
    - Parallel processing via ThreadPoolExecutor
    
    Data sources:
    - Gaia DR3: LOCAL (from pre-matched catalog)
    - VSX: LOCAL (from pre-matched catalog)
    - Gaia Alerts: API (not in local catalog)
    - MILLIQUAS: API (not in local catalog)
    - SIMBAD: API (not in local catalog)
    
    NOTE: This should run AFTER filtering to reduce dataset size.
    """
    if verbose:
        print(f"[crossmatch_all_catalogs] Processing {len(df)} sources")
    
    # =========================================================================
    # LOCAL CATALOG (Gaia DR3 + VSX — no API queries needed)
    # =========================================================================
    if use_local_catalog and (include_gaia_dr3 or include_vsx):
        try:
            from malca.ltv.local_catalog import merge_local_catalog, crossmatch_from_local
            
            # Try ID-based merge first (faster)
            if "ASAS-SN ID" in df.columns:
                df = merge_local_catalog(df, catalog_path=local_catalog_path, verbose=verbose)
            else:
                # Fall back to coordinate crossmatch
                df = crossmatch_from_local(
                    df,
                    catalog_path=local_catalog_path,
                    match_radius_arcsec=match_radius_arcsec,
                    verbose=verbose,
                )
            
            if verbose:
                n_gaia = df["gaia_source_id"].notna().sum() if "gaia_source_id" in df.columns else 0
                n_vsx = df["vsx_name"].notna().sum() if "vsx_name" in df.columns else 0
                print(f"  Local catalog: {n_gaia} Gaia, {n_vsx} VSX matches (no API queries)")
            
            # Mark these as done so we don't query APIs
            include_gaia_dr3 = False
            include_vsx = False
            
        except (ImportError, FileNotFoundError) as e:
            if verbose:
                print(f"  Local catalog not available ({e}), falling back to API queries")
    
    # =========================================================================
    # API QUERIES (only for data NOT in local catalog)
    # =========================================================================
    
    # Determine which API queries are still needed
    api_queries_needed = []
    if include_gaia_dr3:
        api_queries_needed.append("Gaia DR3")
    if include_gaia_alerts:
        api_queries_needed.append("Gaia Alerts")
    if include_vsx:
        api_queries_needed.append("VSX")
    if include_milliquas:
        api_queries_needed.append("MILLIQUAS")
    if include_simbad:
        api_queries_needed.append("SIMBAD")
    
    if verbose and api_queries_needed:
        print(f"  API queries needed: {', '.join(api_queries_needed)}")
    
    # Process API queries on the full (already-filtered) dataset
    if include_gaia_alerts:
        df = crossmatch_gaia_alerts(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=False)

    if include_milliquas:
        df = crossmatch_milliquas(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=False)

    if include_simbad:
        df = query_simbad_classification(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=False)
    
    if verbose:
        print(f"[crossmatch_all_catalogs] Complete")
        if "gaia_source_id" in df.columns:
            print(f"  Gaia DR3: {df['gaia_source_id'].notna().sum()}/{len(df)} matched")
        if "vsx_name" in df.columns:
            print(f"  VSX: {df['vsx_name'].notna().sum()}/{len(df)} matched")
        if "milliquas_name" in df.columns:
            print(f"  MILLIQUAS: {df['milliquas_name'].notna().sum()}/{len(df)} matched")
        if "simbad_main_id" in df.columns:
            print(f"  SIMBAD: {df['simbad_main_id'].notna().sum()}/{len(df)} matched")
    
    return df

