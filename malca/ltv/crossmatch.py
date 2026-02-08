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
    LTV_HEALPIX_NSIDE,
    LTV_WORKERS,
    LTV_CROSSMATCH_CHUNK_SIZE,
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


def _parallel_query(
    df: pd.DataFrame,
    query_func,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    n_workers: int = LTV_WORKERS,
    desc: str = "Query",
    verbose: bool = False,
) -> list[dict]:
    """
    Run queries in parallel using ThreadPoolExecutor.
    
    query_func should take (ra, dec, idx) and return a dict or None.
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        return []
    
    valid_mask = df[ra_column].notna() & df[dec_column].notna()
    tasks = [
        (df.loc[idx, ra_column], df.loc[idx, dec_column], idx)
        for idx in df.index[valid_mask]
    ]
    
    results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(query_func, ra, dec, idx): idx for ra, dec, idx in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, disable=not verbose):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


# =============================================================================
# MILLIQUAS CROSSMATCH (parallel VizieR)
# =============================================================================

def crossmatch_milliquas(
    df: pd.DataFrame,
    *,
    ra_column: str = "ra_deg",
    dec_column: str = "dec_deg",
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Crossmatch to MILLIQUAS v8 via parallel VizieR queries.
    
    Adds columns: milliquas_name, milliquas_type, milliquas_z, milliquas_sep_arcsec
    """
    from astroquery.vizier import Vizier
    
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for MILLIQUAS crossmatch")
        return df
    
    df = df.copy()
    df["milliquas_name"] = None
    df["milliquas_type"] = None
    df["milliquas_z"] = np.nan
    df["milliquas_sep_arcsec"] = np.nan
    
    vizier = Vizier(columns=["Name", "Type", "z", "RAJ2000", "DEJ2000"], row_limit=1)
    
    def query_one(ra, dec, idx):
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            result = vizier.query_region(coord, radius=match_radius_arcsec * u.arcsec, catalog="VII/294/milliqua")
            
            if result and len(result) > 0 and len(result[0]) > 0:
                row = result[0][0]
                result_coord = SkyCoord(
                    ra=float(row["RAJ2000"]) * u.deg,
                    dec=float(row["DEJ2000"]) * u.deg,
                    frame="icrs"
                )
                return {
                    "_idx": idx,
                    "milliquas_name": str(row["Name"]) if "Name" in row.colnames else None,
                    "milliquas_type": str(row["Type"]) if "Type" in row.colnames else None,
                    "milliquas_z": float(row["z"]) if "z" in row.colnames and row["z"] else np.nan,
                    "milliquas_sep_arcsec": coord.separation(result_coord).arcsec,
                }
        except Exception:
            pass
        return None
    
    if verbose:
        print(f"[crossmatch_milliquas] Querying {len(df)} sources...")
    
    results = _parallel_query(df, query_one, ra_column=ra_column, dec_column=dec_column,
                              n_workers=n_workers, desc="MILLIQUAS", verbose=verbose)
    
    for r in results:
        idx = r["_idx"]
        if idx in df.index:
            for col in ["milliquas_name", "milliquas_type", "milliquas_z", "milliquas_sep_arcsec"]:
                df.loc[idx, col] = r[col]
    
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
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Crossmatch to Gaia Alerts via parallel VizieR queries.
    
    Adds columns: gaia_alert_name, gaia_alert_class, gaia_alert_sep_arcsec
    """
    from astroquery.vizier import Vizier
    
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for Gaia Alerts crossmatch")
        return df
    
    df = df.copy()
    df["gaia_alert_name"] = None
    df["gaia_alert_class"] = None
    df["gaia_alert_sep_arcsec"] = np.nan
    
    vizier = Vizier(columns=["*"], row_limit=1)
    
    def query_one(ra, dec, idx):
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            result = vizier.query_region(coord, radius=match_radius_arcsec * u.arcsec, catalog="I/358/vari")
            
            if result and len(result) > 0 and len(result[0]) > 0:
                row = result[0][0]
                result_coord = SkyCoord(
                    ra=float(row["RA_ICRS"]) * u.deg,
                    dec=float(row["DE_ICRS"]) * u.deg,
                    frame="icrs"
                )
                return {
                    "_idx": idx,
                    "gaia_alert_name": str(row["Name"]) if "Name" in row.colnames else None,
                    "gaia_alert_class": str(row["Class"]) if "Class" in row.colnames else None,
                    "gaia_alert_sep_arcsec": coord.separation(result_coord).arcsec,
                }
        except Exception:
            pass
        return None
    
    if verbose:
        print(f"[crossmatch_gaia_alerts] Querying {len(df)} sources...")
    
    results = _parallel_query(df, query_one, ra_column=ra_column, dec_column=dec_column,
                              n_workers=n_workers, desc="Gaia Alerts", verbose=verbose)
    
    for r in results:
        idx = r["_idx"]
        if idx in df.index:
            for col in ["gaia_alert_name", "gaia_alert_class", "gaia_alert_sep_arcsec"]:
                df.loc[idx, col] = r[col]
    
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
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Query SIMBAD for classifications via parallel queries.
    
    Adds columns: simbad_main_id, simbad_otype, simbad_sp_type, simbad_sep_arcsec
    """
    from astroquery.simbad import Simbad
    
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for SIMBAD query")
        return df
    
    df = df.copy()
    df["simbad_main_id"] = None
    df["simbad_otype"] = None
    df["simbad_sp_type"] = None
    df["simbad_sep_arcsec"] = np.nan
    
    simbad = Simbad()
    simbad.add_votable_fields("otype", "sp")
    
    def query_one(ra, dec, idx):
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            result = simbad.query_region(coord, radius=match_radius_arcsec * u.arcsec)
            
            if result is not None and len(result) > 0:
                row = result[0]
                sep = np.nan
                if "RA" in result.colnames and "DEC" in result.colnames:
                    result_coord = SkyCoord(result["RA"][0], result["DEC"][0], unit=(u.hourangle, u.deg), frame="icrs")
                    sep = coord.separation(result_coord).arcsec
                
                return {
                    "_idx": idx,
                    "simbad_main_id": str(row["MAIN_ID"]) if "MAIN_ID" in result.colnames else None,
                    "simbad_otype": str(row["OTYPE"]) if "OTYPE" in result.colnames else None,
                    "simbad_sp_type": str(row["SP_TYPE"]) if "SP_TYPE" in result.colnames else None,
                    "simbad_sep_arcsec": sep,
                }
        except Exception:
            pass
        return None
    
    if verbose:
        print(f"[query_simbad_classification] Querying {len(df)} sources...")
    
    results = _parallel_query(df, query_one, ra_column=ra_column, dec_column=dec_column,
                              n_workers=n_workers, desc="SIMBAD", verbose=verbose)
    
    for r in results:
        idx = r["_idx"]
        if idx in df.index:
            for col in ["simbad_main_id", "simbad_otype", "simbad_sp_type", "simbad_sep_arcsec"]:
                df.loc[idx, col] = r[col]
    
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
    use_healpix: bool = True,
    healpix_nside: int = LTV_HEALPIX_NSIDE,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run all catalog crossmatches with optimized processing.
    
    Optimizations:
    - LOCAL CATALOG: Uses pre-matched ASAS-SN × VSX file for Gaia DR3 & VSX
      data, ELIMINATING those API queries entirely (~99K sources available)
    - HEALPix spatial chunking for remaining API queries
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
    
    # HEALPix partitioning for API queries
    if use_healpix and api_queries_needed:
        from malca.ltv.optim import healpix_partition
        partitions = healpix_partition(df, nside=healpix_nside)
        
        if verbose:
            print(f"  Partitioned into {len(partitions)} HEALPix pixels for API queries")
    else:
        partitions = {0: df}
    
    # Process each partition (API queries only)
    results = []
    for pix, partition in tqdm(partitions.items(), desc="HEALPix chunks", disable=not verbose or len(partitions) == 1):
        if include_gaia_alerts:
            partition = crossmatch_gaia_alerts(partition, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=False)
        
        if include_milliquas:
            partition = crossmatch_milliquas(partition, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=False)
        
        if include_simbad:
            partition = query_simbad_classification(partition, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=False)
        
        results.append(partition)
    
    df = pd.concat(results, ignore_index=True)
    
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

