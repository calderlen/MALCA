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

from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
import numpy as np
import pandas as pd
import pyvo
from astroquery.simbad import Simbad
from astroquery.xmatch import XMatch
from astropy.table import Table

from malca.config.config_ltv import (
    LTV_MATCH_RADIUS_ARCSEC,
    LTV_WORKERS,
    LTV_CROSSMATCH_CHUNK_SIZE,
    VIZIER_TAP_URL,
    SIMBAD_TAP_URL,
)
from malca.config.config_paths import VSX_CROSSMATCH_PATH
from malca.utils import batch_tap_crossmatch as shared_batch_tap_crossmatch
from malca.filter import (
    fetch_chen2020_ztf_periodic,
    fetch_ogle_periodic_catalog,
    _match_period_catalog,
)








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
    
    for col in catalog_subset.columns:
        if col == id_column:
            continue
        local_col = f"{col}_local"
        if local_col in df.columns:
            df[col] = df[col].fillna(df[local_col])
            df = df.drop(columns=[local_col])
    
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
        if col in catalog.columns and col not in df.columns:
            df[col] = np.nan
    for col in VSX_COLUMN_MAP.values():
        if col in catalog.columns and col not in df.columns:
            df[col] = None if col in ["vsx_name", "vsx_type", "vsx_spectral_type"] else np.nan
    
    if "local_catalog_sep_arcsec" not in df.columns:
        df["local_catalog_sep_arcsec"] = np.nan
    
    for i, (cat_idx, is_matched) in enumerate(zip(idx, matched)):
        if is_matched:
            for col in list(GAIA_COLUMN_MAP.values()) + list(VSX_COLUMN_MAP.values()):
                if col in catalog.columns:
                    # Only overwrite if current value is null/nan to avoid destroying API data
                    current_val = df.iloc[i, df.columns.get_loc(col)]
                    if pd.isna(current_val):
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

    Delegates to the shared TAP helper so async result retries and
    sync-subchunk fallback stay consistent across the codebase.
    """
    return shared_batch_tap_crossmatch(
        coords_df,
        tap_url=tap_service,
        catalog_table=catalog_table,
        select_cols=select_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        match_radius_arcsec=match_radius_arcsec,
        chunk_size=chunk_size,
        n_workers=n_workers,
        verbose=verbose,
        desc=desc,
    )


def _batch_xmatch(
    coords_df: pd.DataFrame,
    *,
    cat2: str,
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    verbose: bool = False,
    desc: str = "XMatch",
) -> pd.DataFrame:
    """
    Highly optimized batch crossmatch via CDS XMatch (supports VizieR and SIMBAD).
    Replaces slow TAP batches while preventing query hangs.
    """
    if coords_df.empty:
        return pd.DataFrame()

    table1 = Table.from_pandas(coords_df[["_idx", "ra", "dec"]])
    
    if verbose:
        print(f"  {desc} running via CDS XMatch for {len(coords_df)} sources...")

    try:
        result_table = XMatch.query(
            cat1=table1,
            cat2=cat2,
            max_distance=match_radius_arcsec * u.arcsec,
            colRA1="ra",
            colDec1="dec",
            timeout=300,
        )
        if result_table is None or len(result_table) == 0:
            return pd.DataFrame()
            
        df_res = result_table.to_pandas()
        if "angDist" in df_res.columns:
            df_res["sep_arcsec"] = df_res["angDist"]
        return df_res
    except Exception as e:
        if verbose:
            print(f"  {desc} failed: {e}")
        return pd.DataFrame()


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
        print(f"[crossmatch_milliquas] Querying {len(df)} sources via XMatch...")

    result = _batch_xmatch(
        coords_df,
        cat2="vizier:VII/294/catalog",
        match_radius_arcsec=match_radius_arcsec,
        verbose=verbose,
        desc="MILLIQUAS XMatch",
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
        print(f"[crossmatch_gaia_alerts] Querying {len(df)} sources via XMatch...")

    result = _batch_xmatch(
        coords_df,
        cat2="vizier:I/358/vclassre",
        match_radius_arcsec=match_radius_arcsec,
        verbose=verbose,
        desc="Gaia Alerts XMatch",
    )

    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                # In I/358/vclassre the target key is Source
                df.loc[idx, "gaia_alert_name"] = str(row.get("Source"))
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
        print(f"[query_simbad_classification] Querying {len(df)} sources via XMatch...")

    result = _batch_xmatch(
        coords_df,
        cat2="simbad",
        match_radius_arcsec=match_radius_arcsec,
        verbose=verbose,
        desc="SIMBAD XMatch",
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
# 2MASS & SYDNEY LTV CROSSMATCH
# =============================================================================


def _sydney_2mass_to_ra_dec(clean_2mass: str) -> tuple[float | None, float | None]:
    """
    Parse a cleaned Sydney-style 2MASS ID (e.g. 06400303+1800009) to (ra_deg, dec_deg).
    Format JHHMMSSs±DDMMSSs: RA 8 chars (hh mm ss.ss), Dec sign + 7 chars (dd mm ss.s).
    Returns (None, None) if unparseable (e.g. \\ldots, RW Aur, ASASSN-V J...).
    """
    s = (clean_2mass or "").strip()
    if len(s) < 15 or s[8] not in ("+", "-"):
        return None, None
    try:
        ra_str = s[:8]
        dec_str = s[8:]
        ra_h = int(ra_str[0:2]) + int(ra_str[2:4]) / 60.0 + (int(ra_str[4:6]) + int(ra_str[6:8]) / 100.0) / 3600.0
        ra_deg = 15.0 * ra_h
        sign = 1 if dec_str[0] == "+" else -1
        dec_d = int(dec_str[1:3]) + int(dec_str[3:5]) / 60.0 + (int(dec_str[5:7]) + int(dec_str[7:8]) / 10.0) / 3600.0
        dec_deg = sign * dec_d
        return ra_deg, dec_deg
    except (ValueError, IndexError):
        return None, None


def crossmatch_2mass(
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
    Crossmatch to 2MASS Point Source Catalog via batch VizieR TAP upload.

    Adds columns: 2MASS_ID, 2MASS_sep_arcsec
    """
    if ra_column not in df.columns or dec_column not in df.columns:
        if verbose:
            print("Warning: RA/Dec columns not found for 2MASS crossmatch")
        return df

    df = df.copy()
    df["2MASS_ID"] = None
    df["2MASS_sep_arcsec"] = np.nan

    coords_df = pd.DataFrame({
        "_idx": df.index,
        "ra": df[ra_column].values,
        "dec": df[dec_column].values,
    })

    if verbose:
        print(f"[crossmatch_2mass] Querying {len(df)} sources via XMatch...")

    result = _batch_xmatch(
        coords_df,
        cat2="vizier:II/246/out",
        match_radius_arcsec=match_radius_arcsec,
        verbose=verbose,
        desc="2MASS XMatch",
    )

    if not result.empty:
        result = result.sort_values("sep_arcsec").drop_duplicates(subset="_idx", keep="first")
        for _, row in result.iterrows():
            idx = int(row["_idx"])
            if idx in df.index:
                df.loc[idx, "2MASS_ID"] = row.get("2MASS")
                df.loc[idx, "2MASS_sep_arcsec"] = row["sep_arcsec"]

    if verbose:
        n_matched = df["2MASS_ID"].notna().sum()
        print(f"[crossmatch_2mass] Matched {n_matched}/{len(df)}")

    return df


def crossmatch_sydney_ltv(
    df: pd.DataFrame,
    sydney_csv_path: str | Path,
    *,
    sydney_coord_fallback_arcsec: float = 3.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Merge with SydneyLTVs.csv using the 2MASS_ID column, then fill remaining
    matches by coordinate fallback for sources without a 2MASS_ID (or whose
    2MASS_ID did not match the Sydney list).

    Must be run AFTER crossmatch_2mass.
    Prefixes all added columns with 'sydney_'.
    """
    if "2MASS_ID" not in df.columns:
        if verbose:
            print("Warning: '2MASS_ID' column not found. Run crossmatch_2mass first.")
        return df

    sydney_csv_path = Path(sydney_csv_path)
    if not sydney_csv_path.exists():
        if verbose:
            print(f"Warning: SydneyLTVs.csv not found at {sydney_csv_path}")
        return df

    if verbose:
        print(f"[crossmatch_sydney_ltv] Merging with {sydney_csv_path.name}...")

    sydney_df = pd.read_csv(sydney_csv_path)

    # Prefix columns to avoid collision, except the join key 2MASS
    rename_map = {col: f"sydney_{col}" for col in sydney_df.columns if col != "2MASS"}
    sydney_df = sydney_df.rename(columns=rename_map)

    # Strip off the "2MJ", "$+$", "$-$", etc from the Sydney file so it matches vizier pure coordinate strings
    sydney_df["2MASS_clean"] = sydney_df["2MASS"].astype(str).str.replace(r'2MJ', '', regex=False)
    sydney_df["2MASS_clean"] = sydney_df["2MASS_clean"].str.replace(r'\$\+\$', '+', regex=True)
    sydney_df["2MASS_clean"] = sydney_df["2MASS_clean"].str.replace(r'\$-\$', '-', regex=True)
    sydney_df["2MASS_clean"] = sydney_df["2MASS_clean"].str.strip()

    # Build Sydney coordinate catalog (parseable 2MASS IDs -> ra_deg, dec_deg) for fallback
    ra_dec = [_sydney_2mass_to_ra_dec(s) for s in sydney_df["2MASS_clean"]]
    sydney_df["_ra_deg"] = [r[0] for r in ra_dec]
    sydney_df["_dec_deg"] = [r[1] for r in ra_dec]
    sydney_coord_mask = sydney_df["_ra_deg"].notna()
    sydney_coord_catalog = sydney_df.loc[sydney_coord_mask].copy()
    sydney_coord_catalog = sydney_coord_catalog.drop(columns=["2MASS_clean"], errors="ignore")

    # Strip whitespace from 2MASS_ID for safe joining, handling NaNs
    df = df.copy()
    merge_mask = df["2MASS_ID"].notna()
    df.loc[merge_mask, "2MASS_ID_clean"] = df.loc[merge_mask, "2MASS_ID"].astype(str).str.strip()

    # Merge on 2MASS ID
    n_before = len(df)
    df = df.merge(
        sydney_df,
        left_on="2MASS_ID_clean",
        right_on="2MASS_clean",
        how="left"
    )

    # Clean up merge keys
    df = df.drop(columns=["2MASS_ID_clean", "2MASS_clean"], errors="ignore")
    df = df.drop(columns=["sydney_2MASS"], errors="ignore")
    n_matched_by_id = df["sydney_Class"].notna().sum() if "sydney_Class" in df.columns else 0

    # Coordinate fallback: rows still missing Sydney data
    if (
        sydney_coord_catalog is not None
        and not sydney_coord_catalog.empty
        and "ra_deg" in df.columns
        and "dec_deg" in df.columns
    ):
        miss_mask = (
            (df["sydney_Class"].isna() if "sydney_Class" in df.columns else pd.Series(True, index=df.index))
            & df["ra_deg"].notna()
            & df["dec_deg"].notna()
        )
        if miss_mask.any():
            cand_coords = SkyCoord(
                ra=df.loc[miss_mask, "ra_deg"].values * u.deg,
                dec=df.loc[miss_mask, "dec_deg"].values * u.deg,
            )
            cat_coords = SkyCoord(
                ra=sydney_coord_catalog["_ra_deg"].values * u.deg,
                dec=sydney_coord_catalog["_dec_deg"].values * u.deg,
            )
            idx_cat, sep, _ = cand_coords.match_to_catalog_sky(cat_coords)
            sep_arcsec = sep.arcsec
            df_miss_indices = df.index[miss_mask].tolist()
            coord_fill_cols = [c for c in sydney_coord_catalog.columns if c.startswith("sydney_")]
            for i, df_idx in enumerate(df_miss_indices):
                if sep_arcsec[i] < sydney_coord_fallback_arcsec:
                    cat_row = sydney_coord_catalog.iloc[int(idx_cat[i])]
                    for col in coord_fill_cols:
                        if col in df.columns:
                            df.loc[df_idx, col] = cat_row[col]
                    if "sydney_match_by_coord" not in df.columns:
                        df["sydney_match_by_coord"] = False
                    df.loc[df_idx, "sydney_match_by_coord"] = True

    n_matched_total = df["sydney_Class"].notna().sum() if "sydney_Class" in df.columns else 0
    n_matched_coord = n_matched_total - n_matched_by_id

    if verbose:
        print(f"[crossmatch_sydney_ltv] Merged {n_matched_total}/{n_before} from Sydney list", end="")
        if n_matched_coord > 0:
            print(f" ({n_matched_by_id} from 2MASS ID, {n_matched_coord} from coordinate fallback)")
        else:
            print()

    return df


# =============================================================================
# ZTF AND OGLE PERIODIC (reuse filter.py fetch + match)
# =============================================================================

def crossmatch_ztf_periodic(
    df: pd.DataFrame,
    *,
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    verbose: bool = False,
) -> pd.DataFrame:
    """Crossmatch to ZTF periodic variables (Chen+2020). Adds period_ztf_periodic_* columns."""
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if verbose:
            print("  [ZTF periodic] Skipping: ra_deg/dec_deg not found")
        return df

    added_gaia_id = False
    if "gaia_source_id" in df.columns and "gaia_id" not in df.columns:
        df = df.assign(gaia_id=df["gaia_source_id"])
        added_gaia_id = True

    try:
        ztf_cat = fetch_chen2020_ztf_periodic(show_tqdm=verbose)
    except Exception as e:
        if verbose:
            print(f"  [ZTF periodic] Fetch failed: {e}")
        if added_gaia_id:
            df = df.drop(columns=["gaia_id"])
        return df

    ztf_match = _match_period_catalog(
        df,
        ztf_cat,
        source_label="ztf_periodic",
        max_sep_arcsec=match_radius_arcsec,
        period_col="period",
        class_col="var_type",
        gaia_col="gaia_id",
        show_tqdm=verbose,
    )
    df = pd.concat([df, ztf_match], axis=1)
    if added_gaia_id:
        df = df.drop(columns=["gaia_id"])
    return df


def crossmatch_ogle_periodic(
    df: pd.DataFrame,
    *,
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    verbose: bool = False,
) -> pd.DataFrame:
    """Crossmatch to OGLE periodic variables (II/213). Adds period_ogle_* columns."""
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if verbose:
            print("  [OGLE periodic] Skipping: ra_deg/dec_deg not found")
        return df

    try:
        ogle_cat = fetch_ogle_periodic_catalog(show_tqdm=verbose)
    except Exception as e:
        if verbose:
            print(f"  [OGLE periodic] Fetch failed: {e}")
        return df

    ogle_match = _match_period_catalog(
        df,
        ogle_cat,
        source_label="ogle",
        max_sep_arcsec=match_radius_arcsec,
        period_col="period",
        class_col="var_type",
        gaia_col="gaia_id",
        show_tqdm=verbose,
    )
    df = pd.concat([df, ogle_match], axis=1)
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
    include_sydney_ltv: bool = True,
    include_ztf_periodic: bool = True,
    include_ogle_periodic: bool = True,
    sydney_csv_path: str | Path | None = None,
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
    if include_sydney_ltv:
        api_queries_needed.append("2MASS (for Sydney LTV)")
    if include_ztf_periodic:
        api_queries_needed.append("ZTF periodic")
    if include_ogle_periodic:
        api_queries_needed.append("OGLE periodic")

    if verbose and api_queries_needed:
        print(f"  API queries needed: {', '.join(api_queries_needed)}")
    
    # Process API queries on the full (already-filtered) dataset
    if include_gaia_alerts:
        df = crossmatch_gaia_alerts(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=verbose)

    if include_milliquas:
        df = crossmatch_milliquas(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=verbose)

    if include_simbad:
        df = query_simbad_classification(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=verbose)

    if include_sydney_ltv:
        df = crossmatch_2mass(df, match_radius_arcsec=match_radius_arcsec, n_workers=n_workers, verbose=verbose)
        if sydney_csv_path:
            df = crossmatch_sydney_ltv(df, sydney_csv_path=sydney_csv_path, verbose=verbose)

    if include_ztf_periodic:
        df = crossmatch_ztf_periodic(
            df,
            match_radius_arcsec=match_radius_arcsec,
            verbose=verbose,
        )
    if include_ogle_periodic:
        df = crossmatch_ogle_periodic(
            df,
            match_radius_arcsec=match_radius_arcsec,
            verbose=verbose,
        )

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
        if "2MASS_ID" in df.columns:
            print(f"  2MASS: {df['2MASS_ID'].notna().sum()}/{len(df)} matched")
        if "sydney_Class" in df.columns:
            print(f"  Sydney LTV: {df['sydney_Class'].notna().sum()}/{len(df)} matched")
        if "period_ztf_periodic_match" in df.columns:
            n_ztf = df["period_ztf_periodic_match"].sum()
            print(f"  ZTF periodic: {int(n_ztf)}/{len(df)} matched")
        if "period_ogle_match" in df.columns:
            n_ogle = df["period_ogle_match"].sum()
            print(f"  OGLE periodic: {int(n_ogle)}/{len(df)} matched")

    return df
