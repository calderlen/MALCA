"""
Multi-wavelength characterization for ASAS-SN dipper candidates.

This module consolidates:
- Gaia DR3 querying (astrometry, astrophysics, 2MASS/WISE photometry)
- StarHorse local catalog join (stellar ages, masses)
- 3D dust extinction via dustmaps3d (Wang et al. 2025)
- YSO classification (Koenig & Leisawitz 2014)
- Galactic population classification

Usage:
    malca characterize --input output/events.parquet --output output/characterized.parquet --dust
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import os
import time
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astroquery.ipac.irsa import Irsa
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from dustmaps3d import dustmaps3d
from tqdm import tqdm
import astropy.units as u
import banyan_sigma as banyan_sigma_pkg
import numpy as np
import pandas as pd
import pyvo

from malca.config import (
    GAIA_CHUNK_SIZE, STARHORSE_TAP_CHUNK_SIZE,
    IPHAS_MAX_SEP_ARCSEC, CLUSTER_MAX_SEP_ARCSEC, UNWISE_MAX_SEP_ARCSEC,
    UNWISE_VARIABILITY_ZSCORE, UNWISE_EXPECTED_SCATTER_BASE,
    UNWISE_EXPECTED_SCATTER_SLOPE, UNWISE_EXPECTED_SCATTER_MAG_REF,
    UNWISE_WORKERS, UNWISE_CHECKPOINT_EVERY, UNWISE_MAX_RETRIES,
    SFR_MAX_DIST_KPC, SFR_DIST_TOLERANCE_FRACTION, SFR_CATALOG,
    BANYAN_MIN_ASSOC_PROB, IPHAS_HA_EXCESS_THRESHOLD,
    GALEX_MAX_SEP_ARCSEC, APASS_MAX_SEP_ARCSEC, ALLWISE_MAX_SEP_ARCSEC,
    TMASS_MAX_SEP_ARCSEC,
)
from malca.config import (
    YSO_CLASS_I_W1W2,
    YSO_CLASS_II_W1W2_MIN,
    YSO_CLASS_II_HK,
    YSO_DUST_CORRECTION_HK,
    YSO_DUST_CORRECTION_W1W2,
)
from malca.config import PARQUET_CACHE_COMPRESSION
from malca.config import PARQUET_OUTPUT_COMPRESSION
from malca.config import (
    VSX_CROSSMATCH_PATH, STARHORSE_DEFAULT_PATH, STARHORSE_TAP_URL,
    DEFAULT_CACHE_DIR, LEGACY_DEFAULT_CACHE_DIR, GAIA_CACHE_FILE,
    GAIA_LOCAL_CATALOG, LEGACY_GAIA_CACHE_FILE, LEGACY_GAIA_LOCAL_CATALOG,
)
from malca.candidates import select_passing_candidates_if_present
from malca.feature_layers import to_layer_first_frame
from malca.gaia_ids import normalize_gaia_source_ids, parse_gaia_source_id
from malca.neowise_filters import filter_neowise_single_exposure_lc
from malca.table_io import read_feature_table, read_parquet_table, write_feature_table
from malca.vsx.metadata import normalize_asas_sn_ids, normalize_vsx_match_columns, select_best_vsx_matches



# Suppress astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)



CATALOG_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser()
LEGACY_CATALOG_CACHE_DIR = LEGACY_DEFAULT_CACHE_DIR.expanduser()
STARHORSE_TAP_CACHE_FILE = CATALOG_CACHE_DIR / "starhorse" / "starhorse_tap_cache.parquet"
LEGACY_STARHORSE_TAP_CACHE_FILE = LEGACY_CATALOG_CACHE_DIR / "starhorse_tap_cache.parquet"
OPEN_CLUSTER_META_CACHE_FILE = CATALOG_CACHE_DIR / "open_clusters" / "cantat_gaudin2020_table1.parquet"
LEGACY_OPEN_CLUSTER_META_CACHE_FILE = LEGACY_CATALOG_CACHE_DIR / "cantat_gaudin2020_table1.parquet"
CHARACTERIZE_CACHE_DIR = CATALOG_CACHE_DIR / "characterize"
UNWISE_CHECKPOINT_BASENAME = "unwise_variability_CHECKPOINT.parquet"

WISE_LEGACY_COLUMN_RENAMES = {
    "allwise_w3": "w3",
    "allwise_w3_err": "w3_err",
    "allwise_w4": "w4",
    "allwise_w4_err": "w4_err",
}
WISE_COLOR_PAIRS = (
    ("w1", "w2"),
    ("w1", "w3"),
    ("w1", "w4"),
    ("w2", "w3"),
    ("w2", "w4"),
    ("w3", "w4"),
)
WISE_COLOR_COLUMNS = [f"{left}_{right}" for left, right in WISE_COLOR_PAIRS]

CHARACTERIZE_CACHE_META_COLUMNS = {"_cache_key", "_cache_status", "_cache_updated_at"}
ALLWISE_CACHE_COLUMNS = ["w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err", *WISE_COLOR_COLUMNS]
TMASS_CACHE_COLUMNS = ["tmass_j", "tmass_j_err", "tmass_h", "tmass_h_err", "tmass_k", "tmass_k_err"]
APASS_CACHE_COLUMNS = [
    "apass_v", "apass_v_err", "apass_b", "apass_b_err",
    "apass_g", "apass_g_err", "apass_r", "apass_r_err", "apass_i", "apass_i_err",
]
GALEX_CACHE_COLUMNS = ["galex_fuv", "galex_fuv_err", "galex_nuv", "galex_nuv_err"]
IPHAS_CACHE_COLUMNS = ["iphas_r_ha", "iphas_r_i", "iphas_ha_excess"]
VPHAS_CACHE_COLUMNS = ["vphas_ha_mag", "vphas_r_ha", "vphas_r_i", "vphas_ha_excess"]
OPEN_CLUSTER_CACHE_COLUMNS = ["cluster_name", "cluster_age_myr", "cluster_dist_pc"]
DUST_BASE_CACHE_COLUMNS = ["A_v_3d", "ebv_3d", "dust_sigma", "dust_max_dist_kpc"]
DUST_DERED_SOURCE_COLUMNS = [
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "tmass_j", "tmass_h", "tmass_k", "w1", "w2",
    "apass_b", "apass_v", "apass_g", "apass_r", "apass_i",
    "galex_fuv", "galex_nuv", "baseline_mag", "g", "r", "i",
]


def _parquet_column_names(path: Path) -> list[str] | None:
    """Return Parquet column names without materializing row data."""
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return None


def _canonicalize_wise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse legacy provenance-prefixed WISE magnitude columns to w1..w4."""
    if df.empty:
        return df
    out = df.copy()
    for old_col, new_col in WISE_LEGACY_COLUMN_RENAMES.items():
        if old_col not in out.columns:
            continue
        if new_col not in out.columns:
            out = out.rename(columns={old_col: new_col})
            continue
        old_values = pd.to_numeric(out[old_col], errors="coerce")
        new_values = pd.to_numeric(out[new_col], errors="coerce")
        missing = new_values.isna() & old_values.notna()
        out.loc[missing, new_col] = out.loc[missing, old_col]
        out = out.drop(columns=[old_col])
    return out


def _add_wise_color_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all pairwise WISE color columns available from w1..w4."""
    out = _canonicalize_wise_columns(df)
    for left, right in WISE_COLOR_PAIRS:
        color_col = f"{left}_{right}"
        if left in out.columns and right in out.columns:
            out[color_col] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
    return out


def _characterize_cache_path(module: str) -> Path:
    return CHARACTERIZE_CACHE_DIR / f"{module}.parquet"


def _row_distance_token(row: pd.Series) -> str:
    dist = np.nan
    if "distance_gspphot" in row.index:
        dist = pd.to_numeric(row.get("distance_gspphot"), errors="coerce")
    if not np.isfinite(dist) and "parallax" in row.index:
        plx = pd.to_numeric(row.get("parallax"), errors="coerce")
        if np.isfinite(plx) and plx > 0:
            dist = 1000.0 / plx
    return f"{float(dist):.3f}" if np.isfinite(dist) else "nan"


def _characterize_cache_key(row: pd.Series, module: str) -> str | None:
    for id_col in ("source_id", "gaia_id"):
        if id_col in row.index:
            sid = parse_gaia_source_id(row.get(id_col))
            if sid:
                if module == "dust":
                    return f"gaia:{sid}:dist_pc:{_row_distance_token(row)}"
                return f"gaia:{sid}"

    ra_col = "ra" if "ra" in row.index else ("ra_deg" if "ra_deg" in row.index else None)
    dec_col = "dec" if "dec" in row.index else ("dec_deg" if "dec_deg" in row.index else None)
    if not ra_col or not dec_col:
        return None
    try:
        ra = float(row.get(ra_col))
        dec = float(row.get(dec_col))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None
    if module == "dust":
        return f"coord:{ra:.7f}:{dec:.7f}:dist_pc:{_row_distance_token(row)}"
    return f"coord:{ra:.7f}:{dec:.7f}"


def _characterize_cache_keys(df: pd.DataFrame, module: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    return df.apply(lambda row: _characterize_cache_key(row, module), axis=1)


def _read_characterize_cache(module: str) -> pd.DataFrame:
    path = _characterize_cache_path(module).expanduser()
    if not path.exists():
        return pd.DataFrame()
    try:
        cache = pd.read_parquet(path)
    except Exception as exc:
        print(f"Warning: could not read characterize cache {path}: {exc}")
        return pd.DataFrame()
    if "_cache_key" not in cache.columns:
        return pd.DataFrame()
    cache = cache.copy()
    cache["_cache_key"] = cache["_cache_key"].astype(str)
    return cache.drop_duplicates(subset=["_cache_key"], keep="last")


def _write_characterize_cache(module: str, rows: pd.DataFrame, key_cols: list[str]) -> None:
    if rows.empty:
        return
    path = _characterize_cache_path(module).expanduser()
    try:
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
        combined = combined.drop_duplicates(subset=["_cache_key"], keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False, compression=PARQUET_CACHE_COMPRESSION)
    except Exception as exc:
        print(f"Warning: could not write characterize cache {path}: {exc}")


def _dust_cache_columns(frame: pd.DataFrame) -> list[str]:
    cols = list(DUST_BASE_CACHE_COLUMNS)
    cols.extend(f"{col}_dered" for col in DUST_DERED_SOURCE_COLUMNS if col in frame.columns)
    return cols


def _run_cached_characterization_module(
    df: pd.DataFrame,
    *,
    module: str,
    func,
    output_columns: list[str],
    **kwargs,
) -> pd.DataFrame:
    """Run a characterization lookup using durable per-source catalog cache."""
    if df.empty:
        return df

    out = df.copy()
    keys = _characterize_cache_keys(out, module)
    cache = _read_characterize_cache(module)
    cache_columns = [c for c in cache.columns if c not in CHARACTERIZE_CACHE_META_COLUMNS]
    cache_hit_mask = pd.Series(False, index=out.index)

    if not cache.empty and cache_columns:
        lookup = cache.set_index("_cache_key")
        cache_hit_mask = keys.notna() & keys.astype(str).isin(lookup.index)
        if cache_hit_mask.any():
            for col in cache_columns:
                if col not in out.columns:
                    out[col] = pd.NA
                values = keys.astype(str).map(lookup[col])
                out.loc[cache_hit_mask, col] = values.loc[cache_hit_mask]

    cacheable_mask = keys.notna()
    run_mask = ~cache_hit_mask
    run_df = out.loc[run_mask].copy()
    if not cache.empty or cache_hit_mask.any():
        print(f"{module} cache hit: {int(cache_hit_mask.sum())}/{int(cacheable_mask.sum())}")
    if run_df.empty:
        return _add_wise_color_columns(out) if module == "allwise" else out

    result = func(run_df, **kwargs)
    if result is None:
        result = run_df
    result = result.copy()

    for col in output_columns:
        if col not in result.columns:
            result[col] = pd.NA
        if col not in out.columns:
            out[col] = pd.NA
        out.loc[result.index, col] = result[col]

    result_keys = keys.loc[result.index]
    cache_rows = result.loc[result_keys.notna(), output_columns].copy()
    if not cache_rows.empty:
        cache_rows.insert(0, "_cache_key", result_keys.loc[result_keys.notna()].astype(str).values)
        cache_rows["_cache_status"] = "ok"
        cache_rows["_cache_updated_at"] = pd.Timestamp.utcnow().isoformat()
        _write_characterize_cache(module, cache_rows, output_columns)

    return _add_wise_color_columns(out) if module == "allwise" else out


# =============================================================================
# GAIA DR3 QUERYING
# =============================================================================


def _normalize_source_ids(source_ids: list[str | int]) -> list[str]:
    """Normalize mixed-type source IDs to digit strings."""
    return normalize_gaia_source_ids(source_ids)

def query_gaia_by_ids(source_ids: list[str | int], chunk_size: int = GAIA_CHUNK_SIZE, cache_file: str | None = None) -> pd.DataFrame:
    """
    Look up Gaia DR3 data from a local Parquet catalog.

    The catalog is produced by ``malca gaia-fetch``.  This function reads it
    and returns the subset matching *source_ids* — no network call is made.

    Lookup order for the catalog file:
    1. *cache_file* argument (legacy name kept for API compat)
    2. ``GAIA_LOCAL_CATALOG`` default path
    """
    attempted_paths: list[Path] = []
    gaia_df: pd.DataFrame | None = None
    catalog_path: Path | None = None

    for candidate in [
        cache_file,
        GAIA_LOCAL_CATALOG,
        LEGACY_GAIA_LOCAL_CATALOG,
        LEGACY_GAIA_CACHE_FILE,
    ]:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists() or path in attempted_paths:
            continue
        attempted_paths.append(path)

        print(f"Loading local Gaia catalog from {path}...")
        try:
            candidate_df = pd.read_parquet(path)
        except Exception as e:
            print(f"Warning: ignoring unreadable Gaia catalog at {path}: {e}")
            continue

        if "source_id" not in candidate_df.columns or candidate_df.empty:
            print(f"Warning: ignoring invalid Gaia catalog at {path}")
            continue

        catalog_path = path
        gaia_df = candidate_df
        break

    if catalog_path is None or gaia_df is None:
        raise FileNotFoundError(
            "Local Gaia catalog not found or invalid. Run:\n"
            "  malca gaia-fetch --input <your_candidates.parquet>\n"
            "to download Gaia DR3 data before characterization."
        )

    gaia_df["source_id"] = gaia_df["source_id"].map(parse_gaia_source_id)
    gaia_df = gaia_df.dropna(subset=["source_id"])
    requested = _normalize_source_ids(source_ids)
    result = gaia_df[gaia_df["source_id"].isin(requested)].copy()
    result = _add_wise_color_columns(result)

    print(f"Matched {len(result)}/{len(requested)} requested Gaia IDs from local catalog.")
    return result


# =============================================================================
# STARHORSE LOCAL CATALOG
# =============================================================================

STARHORSE_TAP_TABLE_CANDIDATES = (
    "gaiaedr3_contrib.starhorse",
    "gaiaedr3_contrib.starhorse_1_1",
    "gaiadr2_contrib.starhorse",
    "gaiadr2_contrib.starhorse_v05",
)

STARHORSE_PREFERRED_COLUMNS = (
    "source_id",
    "teff16",
    "teff50",
    "teff84",
    "logg16",
    "logg50",
    "logg84",
    "met16",
    "met50",
    "met84",
    "dist05",
    "dist16",
    "dist50",
    "dist84",
    "dist95",
    "av05",
    "av16",
    "av50",
    "av84",
    "av95",
    "mass16",
    "mass50",
    "mass84",
    "age16",
    "age50",
    "age84",
    "ag50",
    "abp50",
    "arp50",
    "xgal",
    "ygal",
    "zgal",
    "rgal",
    "fidelity",
    "bp_rp_excess_corr",
    "sh_photoflag",
    "sh_outflag",
)


def _load_starhorse_cache(cache_path: Path) -> pd.DataFrame:
    """Load StarHorse TAP cache parquet if present."""
    read_path = cache_path
    if not read_path.exists() and cache_path == STARHORSE_TAP_CACHE_FILE.expanduser():
        read_path = LEGACY_STARHORSE_TAP_CACHE_FILE.expanduser()
    if not read_path.exists():
        return pd.DataFrame()
    try:
        df_cache = pd.read_parquet(read_path)
    except Exception:
        return pd.DataFrame()

    if "source_id" not in df_cache.columns:
        return pd.DataFrame()

    df_cache = df_cache.copy()
    df_cache["source_id"] = df_cache["source_id"].astype(str)
    return df_cache.drop_duplicates(subset=["source_id"], keep="last")


def _save_starhorse_cache(df_cache: pd.DataFrame, cache_path: Path) -> None:
    """Persist StarHorse TAP cache parquet."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_cache.to_parquet(cache_path, index=False, compression=PARQUET_CACHE_COMPRESSION)


def _resolve_starhorse_tap_table(tap_service: pyvo.dal.TAPService) -> tuple[str, list[str]]:
    """Return a StarHorse TAP table and supported column list."""
    for table_name in STARHORSE_TAP_TABLE_CANDIDATES:
        try:
            query = (
                "SELECT column_name FROM TAP_SCHEMA.columns "
                f"WHERE table_name = '{table_name}' "
                "ORDER BY column_index"
            )
            cols_tab = tap_service.search(query=query).to_table()
        except Exception:
            continue

        if "column_name" not in cols_tab.colnames:
            continue

        available_cols = {str(v) for v in cols_tab["column_name"]}
        selected_cols = [c for c in STARHORSE_PREFERRED_COLUMNS if c in available_cols]
        if "source_id" in selected_cols:
            return table_name, selected_cols

    raise RuntimeError("No compatible StarHorse TAP table found")


def query_starhorse_by_ids(
    source_ids: list[str | int],
    starhorse_file: str | Path | None = None,
    use_tap: bool = True,
    cache_file: str | Path | None = None,
) -> pd.DataFrame:
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
        valid_ids = _normalize_source_ids(source_ids)
        if not valid_ids:
            return pd.DataFrame()

        print(f"Querying StarHorse via TAP for {len(valid_ids)} sources...")

        cache_path = Path(cache_file).expanduser() if cache_file else STARHORSE_TAP_CACHE_FILE.expanduser()
        cache_df = _load_starhorse_cache(cache_path)
        cached_ids = set(cache_df["source_id"].astype(str)) if not cache_df.empty else set()

        missing_ids = [sid for sid in valid_ids if sid not in cached_ids]
        if missing_ids:
            print(f"StarHorse cache hit: {len(valid_ids) - len(missing_ids)}/{len(valid_ids)}")
        else:
            print("StarHorse cache hit: all requested IDs")

        new_rows: list[pd.DataFrame] = []
        chunk_size = STARHORSE_TAP_CHUNK_SIZE

        if missing_ids:
            tap_service = pyvo.dal.TAPService(STARHORSE_TAP_URL)
            table_name, select_cols = _resolve_starhorse_tap_table(tap_service)
            select_cols_sql = ", ".join(select_cols)

            for i in tqdm(range(0, len(missing_ids), chunk_size), desc="StarHorse TAP"):
                chunk_ids = missing_ids[i : i + chunk_size]
                ids_str = ",".join(chunk_ids)

                query = (
                    f"SELECT {select_cols_sql} "
                    f"FROM {table_name} "
                    f"WHERE source_id IN ({ids_str})"
                )

                try:
                    chunk_df = tap_service.search(query=query).to_table().to_pandas()
                    if not chunk_df.empty:
                        chunk_df["source_id"] = chunk_df["source_id"].astype(str)
                        new_rows.append(chunk_df)
                except Exception as e:
                    print(f"TAP query error for chunk {i}: {e}")
                    continue

        new_df = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
        if not new_df.empty:
            cache_df = pd.concat([cache_df, new_df], ignore_index=True) if not cache_df.empty else new_df
            cache_df = cache_df.drop_duplicates(subset=["source_id"], keep="last")
            _save_starhorse_cache(cache_df, cache_path)
            print(f"Updated StarHorse cache: {len(cache_df)} rows at {cache_path}")

        if cache_df.empty:
            print("Warning: No StarHorse results from TAP queries.")
            return pd.DataFrame()

        out_df = cache_df[cache_df["source_id"].isin(valid_ids)].copy()
        print(f"Retrieved {len(out_df)}/{len(valid_ids)} StarHorse entries via cache+TAP.")
        return out_df

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
            if str(starhorse_path).endswith('.fits') or str(starhorse_path).endswith('.fits.gz'):
                sh_df = Table.read(starhorse_path).to_pandas()
            else:
                sh_df = pd.read_parquet(starhorse_path)
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
        valid_ids = _normalize_source_ids(source_ids)
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

        # dustmaps3d returns pandas Series indexed by healpix cell, which can
        # contain duplicate labels. Convert to numpy arrays before assignment
        # to avoid pandas index alignment/reindex errors.
        ebv_arr = np.asarray(ebv, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        max_dist_arr = np.asarray(max_dist, dtype=float)

        n_valid = int(valid_mask.sum())
        if len(ebv_arr) != n_valid:
            raise ValueError(f"dustmaps3d returned {len(ebv_arr)} rows for {n_valid} inputs")

        A_V = 3.1 * ebv_arr
        valid_mask_arr = np.asarray(valid_mask, dtype=bool)

        df.loc[valid_mask_arr, 'ebv_3d'] = ebv_arr
        df.loc[valid_mask_arr, 'A_v_3d'] = A_V
        df.loc[valid_mask_arr, 'dust_sigma'] = sigma_arr
        df.loc[valid_mask_arr, 'dust_max_dist_kpc'] = max_dist_arr

        finite_av = A_V[np.isfinite(A_V)]
        if finite_av.size > 0:
            print(f"Dust query complete. Mean A_V = {finite_av.mean():.3f}")
        else:
            print("Dust query complete. No finite A_V values returned.")
        
    except Exception as e:
        print(f"Error querying dustmaps3d: {e}")
        raise

    df['A_v_3d'] = df['A_v_3d'].fillna(0.0)
    
    # Compute dereddened magnitudes for active pipeline bands if they exist
    extinction_coeffs = {
        'phot_g_mean_mag': 0.789,
        'phot_bp_mean_mag': 1.002,
        'phot_rp_mean_mag': 0.589,
        'tmass_j': 0.282,
        'tmass_h': 0.175,
        'tmass_k': 0.112,
        'w1': 0.061,
        'w2': 0.047,
        'apass_b': 1.321,
        'apass_v': 1.000,
        'apass_g': 1.199,
        'apass_r': 0.858,
        'apass_i': 0.639,
        'galex_fuv': 2.61,  # typical values
        'galex_nuv': 2.76,  
        'baseline_mag': 1.199,  # Default to g-band for ASAS-SN if unspecified
        'g': 1.199,
        'r': 0.858,
        'i': 0.639,
    }

    av_col = df['A_v_3d']
    for col, coeff in extinction_coeffs.items():
        if col in df.columns:
            # For each magnitude column present, compute dereddened
            df[f'{col}_dered'] = df[col] - (coeff * av_col)
            
    return df


# =============================================================================
# YSO CLASSIFICATION (Koenig & Leisawitz 2014)
# =============================================================================

def classify_yso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify YSO candidates using 2MASS-WISE color-color diagram.
    Supports dust-corrected colors if A_v_3d is present.
    """
    df = _add_wise_color_columns(df)
    
    # Map columns (support current and common external naming conventions)
    col_map = {
        'H': ['Hmag', 'tmass_h'], 
        'K': ['Kmag', 'tmass_k'], 
        'W1': ['w1', 'W1mag', 'w1mpro'], 
        'W2': ['w2', 'W2mag', 'w2mpro']
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
        hk_color = hk_color - (YSO_DUST_CORRECTION_HK * av)
        w1w2_color = w1w2_color - (YSO_DUST_CORRECTION_W1W2 * av)
        df['H_K_dered'] = hk_color
        df['w1_w2_dered'] = w1w2_color
    
    df['H_K'] = hk_color 
    df['w1_w2'] = w1w2_color
    
    # Classification criteria
    class_i = df['w1_w2'] > YSO_CLASS_I_W1W2
    class_ii = ((df['w1_w2'] > YSO_CLASS_II_W1W2_MIN) & (df['w1_w2'] < YSO_CLASS_I_W1W2) & (df['H_K'] > YSO_CLASS_II_HK))
    trans = ((df['w1_w2'] > YSO_CLASS_II_W1W2_MIN) & (df['w1_w2'] < YSO_CLASS_I_W1W2) & (df['H_K'] < YSO_CLASS_II_HK))
    ms = df['w1_w2'] < YSO_CLASS_II_W1W2_MIN
    
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
        # Not fatal, return what we have

    return df


# =============================================================================
# VPHAS+ Hα CROSSMATCH (Drew+2016, 2025)
# =============================================================================

def crossmatch_vphas(df: pd.DataFrame, max_sep_arcsec: float = IPHAS_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to VPHAS+ DR2 (II/341/vphasp) for Hα emission detection in the Southern Galactic Plane.
    
    Uses CDS XMatch for efficient batch crossmatching. Returns (r-Hα) and (r-i) colors.
    """
    if df.empty:
        return df
    df = df.copy()
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: VPHAS+ crossmatch requires ra, dec columns")
        df['vphas_ha_mag'] = np.nan
        df['vphas_r_ha'] = np.nan
        df['vphas_r_i'] = np.nan
        df['vphas_ha_excess'] = False
        return df
    
    # Initialize output columns
    df['vphas_ha_mag'] = np.nan
    df['vphas_r_ha'] = np.nan
    df['vphas_r_i'] = np.nan
    df['vphas_ha_excess'] = False
    
    # Prepare source table with unique index for matching back
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return df
    
    # Create astropy table for XMatch
    source_table = Table()
    source_table['_idx'] = np.where(valid_mask)[0]
    source_table['ra'] = df.loc[valid_mask, 'ra'].values
    source_table['dec'] = df.loc[valid_mask, 'dec'].values
    
    print(f"Running VPHAS+ XMatch for {len(source_table)} sources...")
    
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:II/341/vphasp',
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
                r_mag = float(row.get('rmag', np.nan))
                ha_mag = float(row.get('Hamag', np.nan))
                i_mag = float(row.get('imag', np.nan))
                
                if np.isfinite(ha_mag):
                    df.at[df.index[idx], 'vphas_ha_mag'] = ha_mag
                
                if np.isfinite(r_mag) and np.isfinite(ha_mag):
                    r_ha = r_mag - ha_mag
                    df.at[df.index[idx], 'vphas_r_ha'] = r_ha
                    df.at[df.index[idx], 'vphas_ha_excess'] = r_ha > IPHAS_HA_EXCESS_THRESHOLD
                
                if np.isfinite(r_mag) and np.isfinite(i_mag):
                    df.at[df.index[idx], 'vphas_r_i'] = r_mag - i_mag
            
            matched = len(result_df)
            ha_excess_count = (df['vphas_ha_excess'] == True).sum()
            print(f"VPHAS+: {matched}/{len(df)} matched, {ha_excess_count} with Hα excess")
        else:
            print("VPHAS+: No matches found")
            
    except Exception as e:
        print(f"VPHAS+ XMatch error: {e}")
        # Not fatal, return what we have

    return df


# =============================================================================
# APASS (Optical), GALEX (UV), and AllWISE (Mid-IR) CROSSMATCH
# =============================================================================

def crossmatch_apass(df: pd.DataFrame, max_sep_arcsec: float = APASS_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to APASS DR9 (Vizier II/336/apass9).
    Returns B, V, g', r', i' magnitudes.
    """
    if df.empty:
        return df
    df = df.copy()
    
    # Initialize output columns
    for col in ["apass_v", "apass_v_err", "apass_b", "apass_b_err", 
                "apass_g", "apass_g_err", "apass_r", "apass_r_err", "apass_i", "apass_i_err"]:
        df[col] = np.nan
        
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return df
        
    source_table = Table()
    source_table['_idx'] = np.where(valid_mask)[0]
    source_table['ra'] = df.loc[valid_mask, 'ra'].values
    source_table['dec'] = df.loc[valid_mask, 'dec'].values
    
    print(f"Running APASS DR9 XMatch for {len(source_table)} sources...")
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:II/336/apass9',
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RAJ2000', colDec2='DEJ2000'
        )
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            if 'angDist' in result_df.columns:
                result_df = result_df.sort_values('angDist').drop_duplicates(subset='_idx', keep='first')
            else:
                result_df = result_df.drop_duplicates(subset='_idx', keep='first')
                
            # Map Vizier columns to internal names
            # APASS DR9 cols: Vmag, e_Vmag, Bmag, e_Bmag, g_mag, e_g_mag, r_mag, e_r_mag, i_mag, e_i_mag
            col_map = {
                "Vmag": "apass_v", "e_Vmag": "apass_v_err",
                "Bmag": "apass_b", "e_Bmag": "apass_b_err",
                "g_mag": "apass_g", "e_g_mag": "apass_g_err",
                "r_mag": "apass_r", "e_r_mag": "apass_r_err",
                "i_mag": "apass_i", "e_i_mag": "apass_i_err"
            }
            
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                for viz_col, my_col in col_map.items():
                    val = row.get(viz_col, np.nan)
                    if pd.notna(val):
                        df.at[df.index[idx], my_col] = float(val)
            
            print(f"APASS: {len(result_df)} matches found")
    except Exception as e:
        print(f"APASS XMatch error: {e}")
        
    return df


def crossmatch_galex(df: pd.DataFrame, max_sep_arcsec: float = GALEX_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to GALEX AIS (Vizier II/312/ais).
    Returns FUV and NUV magnitudes.
    """
    if df.empty:
        return df
    df = df.copy()
    
    # Initialize output columns
    for col in ["galex_fuv", "galex_fuv_err", "galex_nuv", "galex_nuv_err"]:
        df[col] = np.nan
        
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return df
        
    source_table = Table()
    source_table['_idx'] = np.where(valid_mask)[0]
    source_table['ra'] = df.loc[valid_mask, 'ra'].values
    source_table['dec'] = df.loc[valid_mask, 'dec'].values
    
    print(f"Running GALEX AIS XMatch for {len(source_table)} sources...")
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:II/312/ais',
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RAJ2000', colDec2='DEJ2000'
        )
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            if 'angDist' in result_df.columns:
                result_df = result_df.sort_values('angDist').drop_duplicates(subset='_idx', keep='first')
            else:
                result_df = result_df.drop_duplicates(subset='_idx', keep='first')
                
            col_map = {
                "FUVmag": "galex_fuv", "e_FUVmag": "galex_fuv_err",
                "NUVmag": "galex_nuv", "e_NUVmag": "galex_nuv_err"
            }
            
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                for viz_col, my_col in col_map.items():
                    val = row.get(viz_col, np.nan)
                    if pd.notna(val):
                        df.at[df.index[idx], my_col] = float(val)
            
            print(f"GALEX: {len(result_df)} matches found")
    except Exception as e:
        print(f"GALEX XMatch error: {e}")
        
    return df


def crossmatch_allwise(df: pd.DataFrame, max_sep_arcsec: float = ALLWISE_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to AllWISE (Vizier II/328/allwise) for W1-W4 magnitudes.
    """
    if df.empty:
        return df
    df = _canonicalize_wise_columns(df)
    
    # Initialize output columns
    for col in ["w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err"]:
        if col not in df.columns:
            df[col] = np.nan
        
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return _add_wise_color_columns(df)
        
    source_table = Table()
    source_table['_idx'] = np.where(valid_mask)[0]
    source_table['ra'] = df.loc[valid_mask, 'ra'].values
    source_table['dec'] = df.loc[valid_mask, 'dec'].values
    
    print(f"Running AllWISE XMatch (W1-W4) for {len(source_table)} sources...")
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:II/328/allwise',
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RAJ2000', colDec2='DEJ2000'
        )
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            if 'angDist' in result_df.columns:
                result_df = result_df.sort_values('angDist').drop_duplicates(subset='_idx', keep='first')
            else:
                result_df = result_df.drop_duplicates(subset='_idx', keep='first')
                
            col_map = {
                "W1mag": "w1", "e_W1mag": "w1_err",
                "W2mag": "w2", "e_W2mag": "w2_err",
                "W3mag": "w3", "e_W3mag": "w3_err",
                "W4mag": "w4", "e_W4mag": "w4_err",
            }
            
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                for viz_col, my_col in col_map.items():
                    val = row.get(viz_col, np.nan)
                    if pd.notna(val):
                        df.at[df.index[idx], my_col] = float(val)
            
            print(f"AllWISE: {len(result_df)} matches found")
    except Exception as e:
        print(f"AllWISE XMatch error: {e}")
        
    return _add_wise_color_columns(df)


def crossmatch_2mass(df: pd.DataFrame, max_sep_arcsec: float = TMASS_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """
    Crossmatch to 2MASS PSC (Vizier II/246/out) for J, H, Ks magnitudes.
    Used when Gaia merge is skipped (e.g. fetch path) so YSO classification can run.
    """
    if df.empty:
        return df
    df = df.copy()

    for col in ["tmass_j", "tmass_j_err", "tmass_h", "tmass_h_err", "tmass_k", "tmass_k_err"]:
        if col not in df.columns:
            df[col] = np.nan

    valid_mask = df["ra"].notna() & df["dec"].notna()
    if not valid_mask.any():
        return df

    source_table = Table()
    source_table["_idx"] = np.where(valid_mask)[0]
    source_table["ra"] = df.loc[valid_mask, "ra"].values
    source_table["dec"] = df.loc[valid_mask, "dec"].values

    print("Running 2MASS XMatch for J/H/Ks photometry...")
    try:
        result = XMatch.query(
            cat1=source_table,
            cat2="vizier:II/246/out",
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1="ra", colDec1="dec",
            colRA2="RAJ2000", colDec2="DEJ2000",
        )
        if result is not None and len(result) > 0:
            result_df = result.to_pandas()
            if "angDist" in result_df.columns:
                result_df = result_df.sort_values("angDist").drop_duplicates(subset="_idx", keep="first")
            else:
                result_df = result_df.drop_duplicates(subset="_idx", keep="first")

            col_map = {
                "Jmag": "tmass_j", "e_Jmag": "tmass_j_err",
                "Hmag": "tmass_h", "e_Hmag": "tmass_h_err",
                "Kmag": "tmass_k", "e_Kmag": "tmass_k_err",
            }
            for viz_col, my_col in col_map.items():
                if viz_col not in result_df.columns:
                    continue
                for _, row in result_df.iterrows():
                    idx = int(row["_idx"])
                    val = row.get(viz_col, np.nan)
                    if pd.notna(val):
                        df.at[df.index[idx], my_col] = float(val)

            print(f"2MASS: {len(result_df)} matches found")
    except Exception as e:
        print(f"2MASS XMatch error: {e}")

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

def _load_open_cluster_metadata(cache_file: Path | None = None) -> pd.DataFrame:
    """Load Cantat-Gaudin+2020 cluster metadata (age, distance) with local cache."""
    cache_path = Path(cache_file).expanduser() if cache_file else OPEN_CLUSTER_META_CACHE_FILE.expanduser()
    read_path = cache_path
    if not read_path.exists() and cache_path == OPEN_CLUSTER_META_CACHE_FILE.expanduser():
        read_path = LEGACY_OPEN_CLUSTER_META_CACHE_FILE.expanduser()

    if read_path.exists():
        try:
            cached = pd.read_parquet(read_path)
            if {"cluster_name", "cluster_age_myr", "cluster_dist_pc"}.issubset(cached.columns):
                return cached
        except Exception:
            pass

    try:
        viz = Vizier(columns=["Cluster", "AgeNN", "DistPc"], row_limit=-1)
        tables = viz.get_catalogs("J/A+A/640/A1/table1")
        if not tables:
            raise RuntimeError("No tables returned for J/A+A/640/A1/table1")

        tab = tables[0].to_pandas()
        out = pd.DataFrame()
        out["cluster_name"] = tab.get("Cluster", pd.Series(dtype=str)).astype(str)

        age_log = pd.to_numeric(tab.get("AgeNN", np.nan), errors="coerce")
        out["cluster_age_myr"] = np.where(np.isfinite(age_log), np.power(10.0, age_log - 6.0), np.nan)
        out["cluster_dist_pc"] = pd.to_numeric(tab.get("DistPc", np.nan), errors="coerce")

        out = out.replace({"cluster_name": {"nan": ""}})
        out = out[out["cluster_name"].astype(str).str.len() > 0]
        out = out.drop_duplicates(subset=["cluster_name"], keep="first")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False, compression=PARQUET_CACHE_COMPRESSION)
        return out
    except Exception as e:
        print(f"Warning: failed loading open-cluster metadata: {e}")
        return pd.DataFrame(columns=["cluster_name", "cluster_age_myr", "cluster_dist_pc"])


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
        cluster_meta = _load_open_cluster_metadata()
        age_map = dict(zip(cluster_meta["cluster_name"], cluster_meta["cluster_age_myr"])) if not cluster_meta.empty else {}
        dist_map = dict(zip(cluster_meta["cluster_name"], cluster_meta["cluster_dist_pc"])) if not cluster_meta.empty else {}

        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:J/A+A/640/A1/nodup',
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RA_ICRS', colDec2='DE_ICRS'
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
                cluster_name = str(row.get('Cluster', ''))
                df.at[df.index[idx], 'cluster_name'] = cluster_name

                age_val = age_map.get(cluster_name, np.nan)
                dist_val = dist_map.get(cluster_name, np.nan)
                if pd.notna(age_val):
                    df.at[df.index[idx], 'cluster_age_myr'] = float(age_val)
                if pd.notna(dist_val):
                    df.at[df.index[idx], 'cluster_dist_pc'] = float(dist_val)
            
            matched = len(result_df)
            print(f"Cluster crossmatch: {matched}/{len(df)} sources in open clusters")
        else:
            print("Cluster crossmatch: No matches found")
            
    except Exception as e:
        print(f"Cluster XMatch error: {e}")
        raise

    return df


# =============================================================================
# unWISE/unTimely IR VARIABILITY
# =============================================================================

def _unwise_empty_result(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "unwise_w1_zscore": np.nan,
        "unwise_w2_zscore": np.nan,
        "unwise_w1_var": False,
    }


def _query_unwise_single(
    candidate_id: str,
    ra: float,
    dec: float,
    *,
    max_sep_arcsec: float,
    max_retries: int,
) -> dict[str, object]:
    """Query NEOWISE single-exposure photometry for one source and compute variability."""
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return _unwise_empty_result(candidate_id)

    query = f"""
    SELECT
        mjd,
        w1mpro, w1snr,
        w2mpro, w2snr,
        qual_frame,
        qi_fact,
        cc_flags
    FROM neowiser_p1bs_psd
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {max_sep_arcsec / 3600.0})
    ) = 1
    ORDER BY mjd ASC
    """

    for attempt in range(1, max_retries + 1):
        try:
            result = Irsa.query_tap(query)
            table = result.to_table()
            if table is None or len(table) == 0:
                return _unwise_empty_result(candidate_id)

            df_lc = table.to_pandas()
            df_lc = filter_neowise_single_exposure_lc(df_lc)

            if df_lc.empty:
                return _unwise_empty_result(candidate_id)

            w1_mags = pd.to_numeric(df_lc.get("w1mpro"), errors="coerce").dropna().to_numpy()
            w2_mags = pd.to_numeric(df_lc.get("w2mpro"), errors="coerce").dropna().to_numpy()

            out = _unwise_empty_result(candidate_id)

            if len(w1_mags) >= 3:
                w1_std = float(np.std(w1_mags))
                w1_med = float(np.median(w1_mags))
                expected_scatter = UNWISE_EXPECTED_SCATTER_BASE + UNWISE_EXPECTED_SCATTER_SLOPE * max(
                    0.0, w1_med - UNWISE_EXPECTED_SCATTER_MAG_REF
                )
                if expected_scatter > 0:
                    w1_z = w1_std / expected_scatter
                    out["unwise_w1_zscore"] = w1_z
                    out["unwise_w1_var"] = bool(w1_z > UNWISE_VARIABILITY_ZSCORE)

            if len(w2_mags) >= 3:
                w2_std = float(np.std(w2_mags))
                w2_med = float(np.median(w2_mags))
                expected_scatter = UNWISE_EXPECTED_SCATTER_BASE + UNWISE_EXPECTED_SCATTER_SLOPE * max(
                    0.0, w2_med - UNWISE_EXPECTED_SCATTER_MAG_REF
                )
                if expected_scatter > 0:
                    out["unwise_w2_zscore"] = w2_std / expected_scatter

            return out
        except Exception:
            if attempt >= max_retries:
                return _unwise_empty_result(candidate_id)
            time.sleep(float(2 ** (attempt - 1)))

    return _unwise_empty_result(candidate_id)


def _merge_unwise_checkpoint(existing: pd.DataFrame, fresh_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Merge incremental unWISE rows into checkpoint dataframe."""
    if not fresh_rows:
        return existing

    fresh_df = pd.DataFrame(fresh_rows)
    if existing.empty:
        merged = fresh_df
    else:
        merged = pd.concat([existing, fresh_df], ignore_index=True)

    merged = merged.drop_duplicates(subset=["candidate_id"], keep="last")
    return merged


def query_unwise_variability(
    df: pd.DataFrame,
    max_sep_arcsec: float = UNWISE_MAX_SEP_ARCSEC,
    *,
    workers: int = UNWISE_WORKERS,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = UNWISE_CHECKPOINT_EVERY,
    max_retries: int = UNWISE_MAX_RETRIES,
) -> pd.DataFrame:
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
    
    id_col = "candidate_id" if "candidate_id" in df.columns else "asas_sn_id"
    if id_col not in df.columns:
        id_col = "candidate_id"
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)

    coords = df[[id_col, "ra", "dec"]].dropna(subset=["ra", "dec"]).copy()
    if coords.empty:
        df['unwise_w1_zscore'] = np.nan
        df['unwise_w2_zscore'] = np.nan
        df['unwise_w1_var'] = False
        return df

    coords = coords.drop_duplicates(subset=[id_col], keep="first")
    coords = coords.rename(columns={id_col: "candidate_id"})

    ckpt_df = pd.DataFrame(columns=["candidate_id", "unwise_w1_zscore", "unwise_w2_zscore", "unwise_w1_var"])
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            loaded = pd.read_parquet(checkpoint_path)
            required = {"candidate_id", "unwise_w1_zscore", "unwise_w2_zscore", "unwise_w1_var"}
            if required.issubset(loaded.columns):
                ckpt_df = loaded[list(required)].copy()
                ckpt_df["candidate_id"] = ckpt_df["candidate_id"].astype(str)
                ckpt_df = ckpt_df.drop_duplicates(subset=["candidate_id"], keep="last")
                print(f"[unwise] Loaded checkpoint: {len(ckpt_df)} candidates")
        except Exception:
            ckpt_df = pd.DataFrame(columns=["candidate_id", "unwise_w1_zscore", "unwise_w2_zscore", "unwise_w1_var"])

    done_ids = set(ckpt_df["candidate_id"]) if not ckpt_df.empty else set()
    coords_todo = coords[~coords["candidate_id"].isin(done_ids)] if done_ids else coords

    if not coords_todo.empty:
        pending_rows: list[dict[str, object]] = []
        workers_n = max(1, int(workers))

        if workers_n == 1:
            progress = tqdm(coords_todo.itertuples(index=False), total=len(coords_todo), desc="unWISE variability")
            for n_done, row in enumerate(progress, start=1):
                try:
                    pending_rows.append(_query_unwise_single(
                        str(row.candidate_id), float(row.ra), float(row.dec),
                        max_sep_arcsec=max_sep_arcsec, max_retries=max_retries,
                    ))
                except Exception:
                    pending_rows.append(_unwise_empty_result(str(row.candidate_id)))

                if checkpoint_path and (n_done % max(1, int(checkpoint_every)) == 0):
                    ckpt_df = _merge_unwise_checkpoint(ckpt_df, pending_rows)
                    pending_rows = []
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    ckpt_df.to_parquet(checkpoint_path, index=False, compression=PARQUET_CACHE_COMPRESSION)
        else:
            with ThreadPoolExecutor(max_workers=workers_n) as executor:
                futures = {
                    executor.submit(
                        _query_unwise_single,
                        str(row.candidate_id),
                        float(row.ra),
                        float(row.dec),
                        max_sep_arcsec=max_sep_arcsec,
                        max_retries=max_retries,
                    ): str(row.candidate_id)
                    for row in coords_todo.itertuples(index=False)
                }

                progress = tqdm(as_completed(futures), total=len(futures), desc="unWISE variability")
                for n_done, fut in enumerate(progress, start=1):
                    candidate_id = futures[fut]
                    try:
                        pending_rows.append(fut.result())
                    except Exception:
                        pending_rows.append(_unwise_empty_result(candidate_id))

                    if checkpoint_path and (n_done % max(1, int(checkpoint_every)) == 0):
                        ckpt_df = _merge_unwise_checkpoint(ckpt_df, pending_rows)
                        pending_rows = []
                        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                        ckpt_df.to_parquet(checkpoint_path, index=False, compression=PARQUET_CACHE_COMPRESSION)

        if pending_rows:
            ckpt_df = _merge_unwise_checkpoint(ckpt_df, pending_rows)
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                ckpt_df.to_parquet(checkpoint_path, index=False, compression=PARQUET_CACHE_COMPRESSION)
    elif done_ids:
        print(f"[unwise] All {len(coords)} candidates already in checkpoint, skipping queries")

    if ckpt_df.empty:
        df['unwise_w1_zscore'] = np.nan
        df['unwise_w2_zscore'] = np.nan
        df['unwise_w1_var'] = False
        return df

    ckpt_idx = ckpt_df.set_index("candidate_id")
    key = df[id_col].astype(str)
    df['unwise_w1_zscore'] = key.map(ckpt_idx['unwise_w1_zscore'])
    df['unwise_w2_zscore'] = key.map(ckpt_idx['unwise_w2_zscore'])
    df['unwise_w1_var'] = key.map(ckpt_idx['unwise_w1_var']).fillna(False).astype(bool)

    n_var = int(df['unwise_w1_var'].fillna(False).sum())
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


def _save_char_checkpoint(df: pd.DataFrame, path: Path) -> None:
    """Save characterization checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=PARQUET_CACHE_COMPRESSION)


def _module_completed(df: pd.DataFrame, module: str) -> bool:
    """Check if a characterization module already ran successfully."""
    col = f"char_status_{module}"
    if col not in df.columns:
        return False
    vals = df[col].dropna().unique()
    return len(vals) > 0 and all(v in ("ok", "skipped") for v in vals)


def characterize_candidates_df(
    df: pd.DataFrame,
    *,
    crossmatch: Path = VSX_CROSSMATCH_PATH,
    chunk_size: int = GAIA_CHUNK_SIZE,
    cache: Path = GAIA_CACHE_FILE,
    dust: bool = False,
    starhorse: str | None = None,
    starhorse_cache: Path | None = None,
    run_banyan: bool = True,
    run_iphas: bool = True,
    run_vphas: bool = True,
    run_sfr: bool = True,
    run_clusters: bool = True,
    run_unwise: bool = True,
    run_apass: bool = True,
    run_galex: bool = True,
    run_allwise: bool = True,
    unwise_workers: int = UNWISE_WORKERS,
    unwise_checkpoint_every: int = UNWISE_CHECKPOINT_EVERY,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Characterize candidates and return an enriched dataframe."""

    # Fallback for missing ra/dec but present ra_gaia/dec_gaia
    if "ra" not in df.columns and "ra_gaia" in df.columns:
        df["ra"] = df["ra_gaia"]
    if "dec" not in df.columns and "dec_gaia" in df.columns:
        df["dec"] = df["dec_gaia"]
    def _has_finite_values(frame: pd.DataFrame, *columns: str) -> bool:
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return True
        return False

    def _needs_gaia_enrichment(frame: pd.DataFrame) -> bool:
        has_g_mag = _has_finite_values(frame, "phot_g_mean_mag")
        has_color = _has_finite_values(frame, "bp_rp") or (
            _has_finite_values(frame, "phot_bp_mean_mag")
            and _has_finite_values(frame, "phot_rp_mean_mag")
        )
        has_distance = _has_finite_values(frame, "distance_gspphot", "parallax")
        has_motion = _has_finite_values(frame, "pmra") and _has_finite_values(frame, "pmdec")
        return not (has_g_mag and has_color and has_distance and has_motion)

    def _merge_missing_gaia_columns(frame: pd.DataFrame) -> pd.DataFrame:
        if "source_id" not in frame.columns:
            return frame

        gaia_ids = _normalize_source_ids(frame["source_id"].dropna().tolist())
        if not gaia_ids:
            return frame

        print(f"Querying Gaia DR3 for {len(gaia_ids)} sources...")
        try:
            gaia_df = query_gaia_by_ids(
                gaia_ids,
                chunk_size=chunk_size,
                cache_file=str(cache) if cache else None,
            )
        except Exception as e:
            print(f"Warning: characterize Gaia query failed: {e}")
            return frame

        if gaia_df.empty:
            print("Warning: characterize Gaia query returned no rows")
            return frame

        gaia_df = gaia_df.copy()
        gaia_df["source_id"] = gaia_df["source_id"].astype(str)
        lookup = gaia_df.drop_duplicates(subset=["source_id"], keep="last").set_index("source_id")

        out = frame.copy()
        source_ids = out["source_id"].map(parse_gaia_source_id)
        for column in lookup.columns:
            values = source_ids.map(lookup[column])
            if column in out.columns:
                base = out[column]
                if pd.api.types.is_object_dtype(base) or pd.api.types.is_string_dtype(base):
                    base_str = base.astype(str).str.strip().str.lower()
                    missing = base.isna() | base_str.isin({"", "nan", "none", "<na>"})
                    out.loc[missing, column] = values.loc[missing]
                else:
                    out[column] = base.combine_first(values)
            else:
                out[column] = values
        return _add_wise_color_columns(out)

    # If source_id + coordinates already present (e.g. from SkyPatrol fetch),
    # we can skip the crossmatch step and use source_id directly for Gaia enrichment.
    _has_gaia_already = (
        "source_id" in df.columns
        and "ra" in df.columns
        and "dec" in df.columns
    )
    if not _has_gaia_already and "asas_sn_id" not in df.columns:
        print("Warning: characterize skipped: missing 'asas_sn_id' (and no source_id+coords)")
        return df

    # Load checkpoint if available
    df_char = None
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            df_char = pd.read_parquet(checkpoint_path)
            df_char = _add_wise_color_columns(df_char)
            completed = [m for m in ["population", "starhorse", "dust", "yso",
                                      "banyan", "iphas", "vphas", "sfr", "clusters", "unwise",
                                      "apass", "galex", "allwise"]
                         if _module_completed(df_char, m)]
            print(f"Loaded characterization checkpoint ({len(df_char)} rows)")
            if completed:
                print(f"  Modules already completed: {', '.join(completed)}")
        except Exception as e:
            print(f"Warning: could not load characterization checkpoint: {e}")
            df_char = None

    # If source_id + coords already present, skip the crossmatch+Gaia block
    if df_char is None and _has_gaia_already:
        print("Gaia identifiers already present (source_id + coords), skipping crossmatch")
        df_char = _add_wise_color_columns(df)
        if "source_id" in df_char.columns:
            df_char["source_id"] = df_char["source_id"].astype(str)
        if _needs_gaia_enrichment(df_char):
            print("Gaia photometry/astrometry incomplete; fetching Gaia catalog rows")
            df_char = _merge_missing_gaia_columns(df_char)
            df_char = _add_wise_color_columns(df_char)
        _ra_col = "ra"
        _dec_col = "dec"
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
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    # Run Gaia merge + galactic coords if not already done (checkpoint has source_id)
    if df_char is None or "source_id" not in df_char.columns:
        df_in = _add_wise_color_columns(df)

        if "gaia_id" in df_in.columns:
            print("Using existing gaia_id column; skipping characterize crossmatch")
            df_merged = df_in
        else:
            xmatch_path = crossmatch.expanduser()

            if not xmatch_path.exists():
                print(f"Warning: Crossmatch file {xmatch_path} not found. Skipping VSX enrichment.")
                df_merged = df_in
            
            if xmatch_path.exists():
                print(f"Loading crossmatch file {xmatch_path}...")
            try:
                header = _parquet_column_names(xmatch_path)
                if header is None:
                    df_xmatch = read_parquet_table(xmatch_path).astype(str)
                    header = list(df_xmatch.columns)
                else:
                    df_xmatch = None
                id_col = next((c for c in ("asas_sn_id", "ASAS-SN ID", "asassn_id") if c in header), None)
                if id_col is None:
                    raise ValueError(f"crossmatch missing ASAS-SN ID column; found columns: {header[:15]}")

                requested = {
                    id_col,
                    "gaia_id",
                    "tmass_id",
                    "allwise_id",
                    "vsx_class",
                    "vsx_sep_arcsec",
                    "vsx_period",
                    "class",
                    "sep_arcsec",
                    "period",
                }
                use_cols = [c for c in header if c in requested]
                if df_xmatch is None:
                    df_xmatch = read_parquet_table(xmatch_path, columns=use_cols).astype(str)
                else:
                    df_xmatch = df_xmatch[use_cols].astype(str)
                if id_col != "asas_sn_id":
                    df_xmatch = df_xmatch.rename(columns={id_col: "asas_sn_id"})
                df_xmatch = normalize_vsx_match_columns(df_xmatch)

                df_in = df_in.copy()
                df_in["asas_sn_id"] = normalize_asas_sn_ids(df_in["asas_sn_id"])
                df_xmatch = select_best_vsx_matches(df_xmatch, id_column="asas_sn_id")

                overlap_cols = [c for c in df_xmatch.columns if c != "asas_sn_id" and c in df_in.columns]
                if overlap_cols:
                    df_xmatch = df_xmatch.rename(columns={c: f"{c}_xmatch" for c in overlap_cols})

                df_merged = df_in.merge(df_xmatch, on="asas_sn_id", how="left")

                for col in overlap_cols:
                    xcol = f"{col}_xmatch"
                    if col in {"vsx_sep_arcsec", "vsx_period"}:
                        base_num = pd.to_numeric(df_merged[col], errors="coerce")
                        fill_num = pd.to_numeric(df_merged[xcol], errors="coerce")
                        df_merged[col] = base_num.combine_first(fill_num)
                    else:
                        base = df_merged[col]
                        base_str = base.astype(str).str.strip().str.lower()
                        missing = base.isna() | base_str.isin({"", "nan", "none", "<na>"})
                        df_merged.loc[missing, col] = df_merged.loc[missing, xcol]
                    df_merged = df_merged.drop(columns=[xcol])

                if "vsx_sep_arcsec" in df_merged.columns:
                    df_merged["vsx_sep_arcsec"] = pd.to_numeric(df_merged["vsx_sep_arcsec"], errors="coerce")
                if "vsx_period" in df_merged.columns:
                    df_merged["vsx_period"] = pd.to_numeric(df_merged["vsx_period"], errors="coerce")
                print(f"Merged {len(df_merged)} rows")
            except Exception as e:
                print(f"Warning: characterize crossmatch read failed: {e}")
                df_merged = df_in

        if "gaia_id" not in df_merged.columns:
            print("Warning: characterize skipped Gaia query: gaia_id not present")
            df_char = df_merged.copy()
        else:
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)

            df_merged["gaia_id"] = df_merged["gaia_id"].map(parse_gaia_source_id)
            missing_gaia = df_merged["gaia_id"].isna().sum()
            print(f"Found Gaia IDs for {len(df_merged) - missing_gaia}/{len(df_merged)} sources")
            gaia_ids = df_merged["gaia_id"].dropna().unique().tolist()

            if not gaia_ids:
                print("Warning: characterize found no Gaia IDs")
                df_char = df_merged.copy()
            else:
                print(f"Querying Gaia DR3 for {len(gaia_ids)} sources...")
                gaia_df = query_gaia_by_ids(
                    gaia_ids,
                    chunk_size=chunk_size,
                    cache_file=str(cache) if cache else None,
                )
                if gaia_df.empty:
                    print("Warning: characterize Gaia query returned no rows")
                    df_char = df_merged.copy()
                else:
                    gaia_df["source_id"] = gaia_df["source_id"].map(parse_gaia_source_id)
                    gaia_df = gaia_df.dropna(subset=["source_id"])

                    print("Merging Gaia results...")
                    df_char = df_merged.merge(gaia_df, left_on="gaia_id", right_on="source_id", how="left", suffixes=("", "_gaia"))

        df_char = _add_wise_color_columns(df_char)

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

        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "population"):
        print("Classifying Galactic populations...")
        df_char = classify_galactic_population(df_char)
        df_char = _set_module_state(df_char, "population", "ok", "")

    if not _module_completed(df_char, "starhorse"):
        if starhorse:
            print("Loading StarHorse catalog for ages...")
            gaia_ids = df_char["gaia_id"].dropna().unique().tolist() if "gaia_id" in df_char.columns else []
            try:
                use_tap_query = not Path(starhorse).exists() if starhorse != "tap" else True
                sh_df = query_starhorse_by_ids(
                    gaia_ids,
                    starhorse_file=starhorse if not use_tap_query else None,
                    use_tap=use_tap_query,
                    cache_file=starhorse_cache,
                )
                if not sh_df.empty:
                    df_char["source_id"] = df_char["source_id"].astype(str)
                    sh_df["source_id"] = sh_df["source_id"].astype(str)
                    df_char = df_char.merge(sh_df, on="source_id", how="left", suffixes=("", "_sh"))
                    if "age50" in df_char.columns:
                        df_char = classify_galactic_population(df_char)
                    df_char = _set_module_state(df_char, "starhorse", "ok", "")
                else:
                    df_char = _set_module_state(df_char, "starhorse", "skipped", "no StarHorse TAP rows returned")
            except Exception as e:
                msg = str(e)
                print(f"Warning: characterize module 'starhorse' failed: {msg}")
                df_char = _set_module_state(df_char, "starhorse", "error", msg)
        else:
            df_char = _set_module_state(df_char, "starhorse", "skipped", "")
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "dust"):
        df_char = _run_optional_module(
            df_char,
            module="dust",
            enabled=dust,
            description="Computing 3D dust extinction (dustmaps3d)...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="dust",
                func=get_dust_extinction,
                output_columns=_dust_cache_columns(frame),
            ),
        )
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if dust and "A_v_3d" in df_char.columns:
        from malca.ltv.cmd import compute_cmd_features

        df_char = compute_cmd_features(df_char)
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "allwise"):
        df_char = _run_optional_module(
            df_char, module="allwise", enabled=run_allwise,
            description="Running AllWISE (W1-W4) crossmatch...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="allwise",
                func=crossmatch_allwise,
                output_columns=ALLWISE_CACHE_COLUMNS,
            )
        )
        if checkpoint_path: _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "yso"):
        if ("tmass_j" not in df_char.columns or df_char["tmass_j"].isna().all()) and "ra" in df_char.columns and "dec" in df_char.columns and df_char["ra"].notna().any():
            df_char = _run_cached_characterization_module(
                df_char,
                module="2mass",
                func=crossmatch_2mass,
                output_columns=TMASS_CACHE_COLUMNS,
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
            df_char = _set_module_state(df_char, "yso", "skipped", "missing IR photometry")
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    # Run new crossmatches
    if not _module_completed(df_char, "apass"):
        df_char = _run_optional_module(
            df_char, module="apass", enabled=run_apass,
            description="Running APASS DR9 crossmatch...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="apass",
                func=crossmatch_apass,
                output_columns=APASS_CACHE_COLUMNS,
            )
        )
        if checkpoint_path: _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "galex"):
        df_char = _run_optional_module(
            df_char, module="galex", enabled=run_galex,
            description="Running GALEX AIS crossmatch...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="galex",
                func=crossmatch_galex,
                output_columns=GALEX_CACHE_COLUMNS,
            )
        )
        if checkpoint_path: _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "banyan"):
        df_char = _run_optional_module(
            df_char,
            module="banyan",
            enabled=run_banyan,
            description="Running BANYAN Σ membership checks...",
            func=query_banyan_sigma,
        )
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "iphas"):
        df_char = _run_optional_module(
            df_char,
            module="iphas",
            enabled=run_iphas,
            description="Running IPHAS H-alpha crossmatch...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="iphas",
                func=crossmatch_iphas,
                output_columns=IPHAS_CACHE_COLUMNS,
            ),
        )
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "vphas"):
        df_char = _run_optional_module(
            df_char,
            module="vphas",
            enabled=run_vphas,
            description="Running VPHAS+ H-alpha crossmatch...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="vphas",
                func=crossmatch_vphas,
                output_columns=VPHAS_CACHE_COLUMNS,
            ),
        )
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "sfr"):
        df_char = _run_optional_module(
            df_char,
            module="sfr",
            enabled=run_sfr,
            description="Checking star-forming region proximity...",
            func=check_sfr_proximity,
        )

    if not _module_completed(df_char, "clusters"):
        df_char = _run_optional_module(
            df_char,
            module="clusters",
            enabled=run_clusters,
            description="Running open cluster crossmatch...",
            func=lambda frame: _run_cached_characterization_module(
                frame,
                module="open_clusters",
                func=crossmatch_open_clusters,
                output_columns=OPEN_CLUSTER_CACHE_COLUMNS,
            ),
        )
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    if not _module_completed(df_char, "unwise"):
        unwise_checkpoint = None
        if checkpoint_path:
            unwise_checkpoint = Path(checkpoint_path).with_name(UNWISE_CHECKPOINT_BASENAME)

        df_char = _run_optional_module(
            df_char,
            module="unwise",
            enabled=run_unwise,
            description="Querying unWISE/unTimely variability...",
            func=query_unwise_variability,
            workers=unwise_workers,
            checkpoint_path=unwise_checkpoint,
            checkpoint_every=unwise_checkpoint_every,
        )
        if checkpoint_path:
            _save_char_checkpoint(df_char, checkpoint_path)

    df_char = to_layer_first_frame(df_char)

    # Clean up checkpoint on success
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return df_char


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-wavelength characterization for dipper candidates")
    parser.add_argument("--input", type=Path, required=True, help="Input events Parquet (must have asas_sn_id)")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet")
    parser.add_argument("--vsx-crossmatch", type=Path,
                        default=VSX_CROSSMATCH_PATH,
                        help="Path to ASAS-SN x VSX crossmatch Parquet (must contain asas_sn_id and gaia_id)")
    parser.add_argument("--chunk-size", type=int, default=GAIA_CHUNK_SIZE, help="Gaia query chunk size")
    parser.add_argument("--cache", type=Path, default=GAIA_CACHE_FILE, help="Cache file for Gaia queries")
    parser.add_argument("--enable-dust", dest="dust", action="store_true", help="Enable dustmaps3d 3D extinction query")
    parser.add_argument("--starhorse", type=str, default=None, help="StarHorse stellar ages/masses: 'tap' for remote TAP query (recommended), or path to local catalog file")
    parser.add_argument("--starhorse-cache", type=Path, default=STARHORSE_TAP_CACHE_FILE, help=f"StarHorse TAP cache parquet path (default: {STARHORSE_TAP_CACHE_FILE})")
    parser.add_argument("--unwise-workers", type=int, default=UNWISE_WORKERS, help="Parallel workers for unWISE variability queries")
    parser.add_argument("--unwise-checkpoint-every", type=int, default=UNWISE_CHECKPOINT_EVERY, help="Persist unWISE checkpoint every N completed sources")
    parser.add_argument("--no-characterize-banyan", dest="characterize_banyan", action="store_false", help="Disable BANYAN Sigma enrichment")
    parser.add_argument("--no-characterize-iphas", dest="characterize_iphas", action="store_false", help="Disable IPHAS enrichment")
    parser.add_argument("--no-characterize-vphas", dest="characterize_vphas", action="store_false", help="Disable VPHAS+ enrichment")
    parser.add_argument("--no-characterize-sfr", dest="characterize_sfr", action="store_false", help="Disable star-forming-region enrichment")
    parser.add_argument("--no-characterize-clusters", dest="characterize_clusters", action="store_false", help="Disable open-cluster enrichment")
    parser.add_argument("--characterize-unwise", dest="characterize_unwise", action="store_true", help="Enable unWISE/unTimely variability enrichment (default: disabled)")
    parser.add_argument("--no-characterize-unwise", dest="characterize_unwise", action="store_false", help="Disable unWISE variability enrichment")
    parser.add_argument("--all-candidates", action="store_true", help="Characterize all input rows instead of only failed_any=False passers")
    parser.set_defaults(
        characterize_banyan=True,
        characterize_iphas=True,
        characterize_vphas=True,
        characterize_sfr=True,
        characterize_clusters=True,
        characterize_unwise=False,
    )
    
    args = parser.parse_args()
    
    # Load input
    print(f"Loading {args.input}...")
    df = read_feature_table(args.input)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
        
    df_char = characterize_candidates_df(
        df,
        crossmatch=args.vsx_crossmatch,
        chunk_size=args.chunk_size,
        cache=args.cache,
        dust=args.dust,
        starhorse=args.starhorse,
        starhorse_cache=args.starhorse_cache,
        run_banyan=args.characterize_banyan,
        run_iphas=args.characterize_iphas,
        run_vphas=args.characterize_vphas,
        run_sfr=args.characterize_sfr,
        run_clusters=args.characterize_clusters,
        run_unwise=args.characterize_unwise,
        unwise_workers=args.unwise_workers,
        unwise_checkpoint_every=args.unwise_checkpoint_every,
    )
    
    # Save results
    print("Saving results...")
    output_path = args.output.expanduser()
    write_feature_table(df_char, output_path)
        
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
