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
import json
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
from malca.products.candidates import select_passing_candidates_if_present
from malca.products.feature_layers import to_layer_first_frame
from malca.catalogs.gaia_fetch import GAIA_BANYAN_REQUIRED_COLUMNS, gaia_banyan_input_mask
from malca.catalogs.gaia_ids import canonicalize_gaia_ids_in_frame, normalize_gaia_source_ids, parse_gaia_source_id
from malca.catalogs.neowise_filters import filter_neowise_single_exposure_lc
from malca.io.table_io import read_feature_table, read_parquet_table, write_feature_table
from malca.vsx.metadata import normalize_asas_sn_ids, normalize_vsx_match_columns, select_best_vsx_matches
from malca.enrichment.banyan import BANYAN_OUTPUT_COLUMNS, compute_banyan_membership



# Suppress astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)



CATALOG_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser()
LEGACY_CATALOG_CACHE_DIR = LEGACY_DEFAULT_CACHE_DIR.expanduser()
STARHORSE_TAP_CACHE_FILE = CATALOG_CACHE_DIR / "starhorse" / "starhorse_tap_cache.parquet"
LEGACY_STARHORSE_TAP_CACHE_FILE = LEGACY_CATALOG_CACHE_DIR / "starhorse_tap_cache.parquet"
OPEN_CLUSTER_META_CACHE_FILE = CATALOG_CACHE_DIR / "open_clusters" / "cantat_gaudin2020_table1.parquet"
LEGACY_OPEN_CLUSTER_META_CACHE_FILE = LEGACY_CATALOG_CACHE_DIR / "cantat_gaudin2020_table1.parquet"
CHARACTERIZE_CACHE_DIR = CATALOG_CACHE_DIR / "characterize"
CHARACTERIZE_CACHE_VERSION = "catalog-match-v2"
CHARACTERIZE_STATUS_VERSION = "2"
CHARACTERIZE_NEGATIVE_CACHE_MAX_AGE_DAYS = 7.0
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
ALLWISE_QUALITY_COLUMNS = [
    "allwise_id",
    "allwise_sep_arcsec",
    "allwise_ph_qual",
    "allwise_cc_flags",
    "allwise_ext_flg",
    "allwise_nb",
    "allwise_na",
    "allwise_var_flg",
    *[f"allwise_w{band}_{suffix}" for band in range(1, 5) for suffix in ("snr", "rchi2", "sat", "ndet", "nframe")],
]
ALLWISE_TEXT_QUALITY_COLUMNS = {
    "allwise_id", "allwise_ph_qual", "allwise_cc_flags", "allwise_var_flg",
}

CHARACTERIZE_CACHE_META_COLUMNS = {
    "_cache_key", "_cache_status", "_cache_updated_at", "_cache_version"
}
ALLWISE_CACHE_COLUMNS = [
    "w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err",
    *WISE_COLOR_COLUMNS,
    *ALLWISE_QUALITY_COLUMNS,
]
TMASS_CACHE_COLUMNS = ["tmass_j", "tmass_j_err", "tmass_h", "tmass_h_err", "tmass_k", "tmass_k_err"]
APASS_CACHE_COLUMNS = [
    "apass_v", "apass_v_err", "apass_b", "apass_b_err",
    "apass_g", "apass_g_err", "apass_r", "apass_r_err", "apass_i", "apass_i_err",
]
GALEX_CACHE_COLUMNS = ["galex_fuv", "galex_fuv_err", "galex_nuv", "galex_nuv_err"]
IPHAS_CACHE_COLUMNS = [
    "iphas_r_mag", "iphas_r_err",
    "iphas_i_mag", "iphas_i_err",
    "iphas_ha_mag", "iphas_ha_err",
    "iphas_r_i", "iphas_r_i_err",
    "iphas_r_ha", "iphas_r_ha_err",
    "iphas_sep_arcsec", "iphas_source_catalog",
    "iphas_ha_excess",
]
VPHAS_CACHE_COLUMNS = [
    "vphas_r_mag", "vphas_r_err",
    "vphas_i_mag", "vphas_i_err",
    "vphas_ha_mag", "vphas_ha_err",
    "vphas_r_i", "vphas_r_i_err",
    "vphas_r_ha", "vphas_r_ha_err",
    "vphas_sep_arcsec", "vphas_source_catalog",
    "vphas_ha_excess",
]
OPEN_CLUSTER_CACHE_COLUMNS = ["cluster_name", "cluster_age_myr", "cluster_dist_pc"]
DUST_BASE_CACHE_COLUMNS = [
    "A_v_3d", "ebv_3d", "dust_sigma", "dust_max_dist_kpc",
    "dust_status", "dust_distance_source", "dust_distance_pc",
]
DUST_DERED_SOURCE_COLUMNS = [
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "tmass_j", "tmass_h", "tmass_k", "w1", "w2",
    "apass_b", "apass_v", "apass_g", "apass_r", "apass_i",
    "galex_fuv", "galex_nuv", "baseline_mag", "g", "r", "i",
]

MODULE_COMPLETION_COLUMNS = {
    "allwise": ALLWISE_CACHE_COLUMNS,
    "tmass": TMASS_CACHE_COLUMNS,
    "apass": APASS_CACHE_COLUMNS,
    "galex": GALEX_CACHE_COLUMNS,
    "iphas": IPHAS_CACHE_COLUMNS,
    "vphas": VPHAS_CACHE_COLUMNS,
    "clusters": OPEN_CLUSTER_CACHE_COLUMNS,
    "banyan": list(BANYAN_OUTPUT_COLUMNS),
}


def _parquet_column_names(path: Path) -> list[str] | None:
    """Return Parquet column names without materializing row data."""
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return None


def _float_or_nan(value: object) -> float:
    """Coerce a scalar catalog value to float, preserving missing values."""
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
    """Return the first finite numeric value among a row's possible column aliases."""
    for name in names:
        if name in row.index:
            value = _float_or_nan(row.get(name))
            if np.isfinite(value):
                return value
    return np.nan


def _row_sep_arcsec(row: pd.Series) -> float:
    return _row_first_numeric(row, "angDist", "sep_arcsec", "_r")


def _quadrature_error(left: float, right: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float(np.hypot(left, right))
    return np.nan


def _ensure_output_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col.endswith("_source_catalog"):
            df[col] = ""
        elif col.endswith("_ha_excess"):
            df[col] = False
        else:
            df[col] = np.nan
    return df


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
                    return f"{CHARACTERIZE_CACHE_VERSION}:gaia:{sid}:dist_pc:{_row_distance_token(row)}"
                return f"{CHARACTERIZE_CACHE_VERSION}:gaia:{sid}"

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
        return f"{CHARACTERIZE_CACHE_VERSION}:coord:{ra:.7f}:{dec:.7f}:dist_pc:{_row_distance_token(row)}"
    return f"{CHARACTERIZE_CACHE_VERSION}:coord:{ra:.7f}:{dec:.7f}"


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
    if "_cache_version" in cache.columns:
        cache = cache[cache["_cache_version"].fillna("").astype(str) == CHARACTERIZE_CACHE_VERSION]
    if "_cache_status" in cache.columns and "_cache_updated_at" in cache.columns:
        status = cache["_cache_status"].fillna("").astype(str).str.lower()
        updated = pd.to_datetime(cache["_cache_updated_at"], errors="coerce", utc=True)
        age_days = (pd.Timestamp.now(tz="UTC") - updated).dt.total_seconds() / 86400.0
        stale_negative = status.isin({"miss", "no_data"}) & (
            updated.isna() | (age_days > CHARACTERIZE_NEGATIVE_CACHE_MAX_AGE_DAYS)
        )
        cache = cache.loc[~stale_negative]
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


def _characterize_value_present(value: object) -> bool:
    if value is None or np.ma.is_masked(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "nan", "none", "<na>", "--"}
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


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
    if not cache.empty:
        missing_cache_columns = [col for col in output_columns if col not in cache.columns]
        if missing_cache_columns:
            print(
                f"{module} cache schema stale; rerunning lookup "
                f"for missing columns: {', '.join(missing_cache_columns[:5])}"
            )
            cache = pd.DataFrame()
    cache_columns = [c for c in cache.columns if c not in CHARACTERIZE_CACHE_META_COLUMNS]
    cache_hit_mask = pd.Series(False, index=out.index)
    cache_status_col = f"char_cache_status_{module}"
    out[cache_status_col] = "uncacheable" if keys.isna().all() else "miss"

    if not cache.empty and cache_columns:
        lookup = cache.set_index("_cache_key")
        cache_hit_mask = keys.notna() & keys.astype(str).isin(lookup.index)
        if cache_hit_mask.any():
            for col in cache_columns:
                if col not in out.columns:
                    out[col] = pd.NA
                values = keys.astype(str).map(lookup[col])
                out.loc[cache_hit_mask, col] = values.loc[cache_hit_mask]
            cached_status = keys.astype(str).map(lookup.get("_cache_status", pd.Series(dtype=object)))
            out.loc[cache_hit_mask, cache_status_col] = (
                "cached_" + cached_status.loc[cache_hit_mask].fillna("hit").astype(str)
            )
    out.loc[keys.isna(), cache_status_col] = "uncacheable"

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

    evidence_columns = (
        [col for col in ("A_v_3d", "ebv_3d") if col in result.columns]
        if module == "dust"
        else [
            col for col in output_columns
            if not col.endswith(("_status", "_source"))
        ]
    )
    has_data = result[evidence_columns].apply(
        lambda row: any(_characterize_value_present(value) for value in row), axis=1
    ) if evidence_columns else pd.Series(False, index=result.index)
    result_cache_status = pd.Series("miss", index=result.index, dtype=object)
    result_cache_status.loc[has_data] = "hit"
    out.loc[result.index, cache_status_col] = np.where(has_data, "fetched", "fetched_no_data")

    result_keys = keys.loc[result.index]
    cache_rows = result.loc[result_keys.notna(), output_columns].copy()
    if not cache_rows.empty:
        cache_rows.insert(0, "_cache_key", result_keys.loc[result_keys.notna()].astype(str).values)
        cache_rows["_cache_status"] = result_cache_status.loc[result_keys.notna()].values
        cache_rows["_cache_updated_at"] = pd.Timestamp.utcnow().isoformat()
        cache_rows["_cache_version"] = CHARACTERIZE_CACHE_VERSION
        _write_characterize_cache(module, cache_rows, output_columns)

    return _add_wise_color_columns(out) if module == "allwise" else out


# =============================================================================
# GAIA DR3 QUERYING
# =============================================================================


def _normalize_source_ids(source_ids: list[str | int]) -> list[str]:
    """Normalize mixed-type source IDs to digit strings."""
    return normalize_gaia_source_ids(source_ids)


GAIA_ENRICHMENT_REQUIRED_COLUMNS = tuple(dict.fromkeys((
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "parallax",
    "parallax_error",
    *GAIA_BANYAN_REQUIRED_COLUMNS,
)))


def gaia_identifier_series(frame: pd.DataFrame) -> pd.Series:
    """Return a normalized Gaia DR3 key, preferring the explicit ``gaia_id``."""
    identifiers = pd.Series(pd.NA, index=frame.index, dtype=object)
    for column in ("gaia_id", "source_id"):
        if column not in frame.columns:
            continue
        parsed = frame[column].map(parse_gaia_source_id)
        identifiers = identifiers.combine_first(parsed)
    return identifiers


def gaia_enrichment_needed_mask(frame: pd.DataFrame) -> pd.Series:
    """Return Gaia-keyed rows missing at least one current enrichment field."""
    identifiers = gaia_identifier_series(frame)
    incomplete = pd.Series(False, index=frame.index, dtype=bool)
    for column in GAIA_ENRICHMENT_REQUIRED_COLUMNS:
        if column not in frame.columns:
            incomplete |= True
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        incomplete |= values.isna() | ~np.isfinite(values)
        if column in {"pmra_error", "pmdec_error"}:
            incomplete |= values <= 0
        elif column == "ra":
            incomplete |= ~values.between(0.0, 360.0, inclusive="left")
        elif column == "dec":
            incomplete |= ~values.between(-90.0, 90.0, inclusive="both")
    return identifiers.notna() & incomplete


def merge_gaia_catalog_rows(frame: pd.DataFrame, gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Fill row-level Gaia gaps from a local catalog using canonical Gaia IDs."""
    out = frame.copy()
    identifiers = gaia_identifier_series(out)
    if "gaia_id" not in out.columns:
        out["gaia_id"] = identifiers
    else:
        out["gaia_id"] = out["gaia_id"].map(parse_gaia_source_id).combine_first(identifiers)
    if "source_id" not in out.columns:
        out["source_id"] = identifiers

    lookup_frame = gaia_df.copy()
    if "source_id" not in lookup_frame.columns:
        return out
    lookup_frame["source_id"] = lookup_frame["source_id"].map(parse_gaia_source_id)
    lookup_frame = lookup_frame.dropna(subset=["source_id"])
    lookup = lookup_frame.drop_duplicates(subset=["source_id"], keep="last").set_index("source_id")
    matched = identifiers.notna() & identifiers.isin(lookup.index)

    for column in lookup.columns:
        values = identifiers.map(lookup[column])
        if column not in out.columns:
            out[column] = values
            continue
        base = out[column]
        if pd.api.types.is_object_dtype(base) or pd.api.types.is_string_dtype(base):
            text = base.astype(str).str.strip().str.lower()
            missing = base.isna() | text.isin({"", "nan", "none", "<na>"})
            out.loc[missing, column] = values.loc[missing]
        else:
            out[column] = base.combine_first(values)

    complete = gaia_banyan_input_mask(out)
    status = pd.Series("missing_gaia_id", index=out.index, dtype=object)
    status.loc[identifiers.notna()] = "not_in_cache"
    status.loc[matched] = "partial"
    status.loc[matched & complete] = "complete"
    status.loc[~matched & complete] = "existing_complete"
    out["gaia_enrichment_status"] = status
    out["gaia_enrichment_source"] = np.where(matched, "gaia_dr3_cache", "")
    out["gaia_astrometry_complete"] = complete.astype(bool)
    out["gaia_banyan_input_complete"] = complete.astype(bool)
    out["gaia_enrichment_updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    def _missing_fields(row: pd.Series) -> str:
        missing: list[str] = []
        for column in GAIA_BANYAN_REQUIRED_COLUMNS:
            value = pd.to_numeric(
                pd.Series([row.get(column, np.nan)]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(value):
                missing.append(column)
            elif column in {"pmra_error", "pmdec_error"} and value <= 0:
                missing.append(column)
            elif column == "ra" and not 0.0 <= value < 360.0:
                missing.append(column)
            elif column == "dec" and not -90.0 <= value <= 90.0:
                missing.append(column)
        return json.dumps(missing, separators=(",", ":"))

    out["gaia_missing_fields_json"] = out.apply(_missing_fields, axis=1)
    return _add_wise_color_columns(out)

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

                for attempt in range(1, 4):
                    try:
                        chunk_df = tap_service.search(query=query).to_table().to_pandas()
                        if not chunk_df.empty:
                            chunk_df["source_id"] = chunk_df["source_id"].astype(str)
                            new_rows.append(chunk_df)
                        break
                    except Exception as e:
                        if attempt >= 3:
                            print(f"TAP query error for chunk {i}: {e}")
                            break
                        delay = min(5.0 * attempt, 30.0)
                        print(f"TAP query error for chunk {i} attempt {attempt}/3: {e}; retrying in {delay:.0f}s")
                        time.sleep(delay)

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
    df['A_v_3d'] = np.nan
    df['ebv_3d'] = np.nan
    df['dust_sigma'] = np.nan
    df['dust_max_dist_kpc'] = np.nan
    df['dust_status'] = 'not_queried'
    df['dust_distance_source'] = ''
    df['dust_distance_pc'] = np.nan
    
    if 'ra' in df.columns and 'dec' in df.columns:
        ra_col = 'ra'
        dec_col = 'dec'
    elif 'ra_deg' in df.columns and 'dec_deg' in df.columns:
        ra_col = 'ra_deg'
        dec_col = 'dec_deg'
    else:
        print("Warning: Missing ra/dec columns for dust query.")
        df['dust_status'] = 'missing_coordinates'
        return df
        
    # Distance (in pc from Gaia, need kpc for dustmaps3d)
    dist_pc = np.full(len(df), np.nan)
    
    if 'gaia_parallax' in df.columns:
        plx = pd.to_numeric(df['gaia_parallax'], errors='coerce').to_numpy(dtype=float)
        valid_plx = (np.isfinite(plx)) & (plx > 0)
        dist_pc[valid_plx] = 1000.0 / plx[valid_plx]
        df.loc[valid_plx, 'dust_distance_source'] = 'gaia_parallax'
    elif 'parallax' in df.columns:
        plx = pd.to_numeric(df['parallax'], errors='coerce').to_numpy(dtype=float)
        valid_plx = (np.isfinite(plx)) & (plx > 0)
        dist_pc[valid_plx] = 1000.0 / plx[valid_plx]
        df.loc[valid_plx, 'dust_distance_source'] = 'parallax'
        
    if 'distance_gspphot' in df.columns:
        gsp = pd.to_numeric(df['distance_gspphot'], errors='coerce').to_numpy(dtype=float)
        valid_gsp = np.isfinite(gsp) & (gsp > 0)
        dist_pc[valid_gsp] = gsp[valid_gsp]
        df.loc[valid_gsp, 'dust_distance_source'] = 'distance_gspphot'

    df['dust_distance_pc'] = dist_pc
        
    if np.isnan(dist_pc).all():
        print("Warning: No distance info for dust query.")
        df['dust_status'] = 'missing_distance'
        return df
    
    dist_kpc = dist_pc / 1000.0
    valid_mask = (np.isfinite(df[ra_col])) & (np.isfinite(df[dec_col])) & (np.isfinite(dist_kpc)) & (dist_kpc > 0)
    
    if not valid_mask.any():
        df.loc[~np.isfinite(dist_kpc) | (dist_kpc <= 0), 'dust_status'] = 'missing_distance'
        coordinate_valid = np.isfinite(pd.to_numeric(df[ra_col], errors='coerce')) & np.isfinite(pd.to_numeric(df[dec_col], errors='coerce'))
        df.loc[~coordinate_valid, 'dust_status'] = 'missing_coordinates'
        return df
    df.loc[valid_mask, 'dust_status'] = 'query_pending'
    df.loc[~valid_mask & (~np.isfinite(dist_kpc) | (dist_kpc <= 0)), 'dust_status'] = 'missing_distance'
    
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
        finite_result = np.isfinite(ebv_arr) & np.isfinite(A_V)
        valid_indices = df.index[valid_mask_arr]
        df.loc[valid_indices[finite_result], 'dust_status'] = 'ok'
        df.loc[valid_indices[~finite_result], 'dust_status'] = 'no_data'

        finite_av = A_V[np.isfinite(A_V)]
        if finite_av.size > 0:
            print(f"Dust query complete. Mean A_V = {finite_av.mean():.3f}")
        else:
            print("Dust query complete. No finite A_V values returned.")
        
    except Exception as e:
        print(f"Error querying dustmaps3d: {e}")
        raise

    df['A_v_3d'] = pd.to_numeric(df['A_v_3d'], errors="coerce")
    
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
    
    def coalesce(candidates: list[str], label: str) -> pd.Series:
        values = pd.Series(np.nan, index=df.index, dtype=float)
        source = pd.Series("", index=df.index, dtype=object)
        for column in candidates:
            if column not in df.columns:
                continue
            candidate = pd.to_numeric(df[column], errors="coerce")
            take = values.isna() & candidate.notna()
            values.loc[take] = candidate.loc[take]
            source.loc[take] = column
        df[f"yso_{label}_source"] = source
        return values

    H = coalesce(col_map['H'], 'h')
    K = coalesce(col_map['K'], 'k')
    W1 = coalesce(col_map['W1'], 'w1')
    W2 = coalesce(col_map['W2'], 'w2')
        
    hk_color = H - K
    w1w2_color = W1 - W2
    
    # Dust Correction
    df['yso_extinction_status'] = 'not_available'
    if 'A_v_3d' in df.columns:
        av = pd.to_numeric(df['A_v_3d'], errors="coerce")
        valid_av = av.notna() & np.isfinite(av) & (av >= 0)
        hk_color.loc[valid_av] = hk_color.loc[valid_av] - (YSO_DUST_CORRECTION_HK * av.loc[valid_av])
        w1w2_color.loc[valid_av] = w1w2_color.loc[valid_av] - (YSO_DUST_CORRECTION_W1W2 * av.loc[valid_av])
        df['H_K_dered'] = np.nan
        df['w1_w2_dered'] = np.nan
        df.loc[valid_av, 'H_K_dered'] = hk_color.loc[valid_av]
        df.loc[valid_av, 'w1_w2_dered'] = w1w2_color.loc[valid_av]
        df.loc[valid_av, 'yso_extinction_status'] = np.where(av.loc[valid_av] > 0, 'corrected', 'measured_zero')
        df.loc[av.notna() & ~valid_av, 'yso_extinction_status'] = 'invalid'
    
    df['H_K'] = hk_color 
    df['w1_w2'] = w1w2_color
    valid_colors = hk_color.notna() & w1w2_color.notna()
    df['yso_input_status'] = np.where(valid_colors, 'ok', 'missing_ir_bands')
    
    # Classification criteria
    class_i = valid_colors & (df['w1_w2'] >= YSO_CLASS_I_W1W2)
    class_ii = valid_colors & (df['w1_w2'] > YSO_CLASS_II_W1W2_MIN) & (df['w1_w2'] < YSO_CLASS_I_W1W2) & (df['H_K'] >= YSO_CLASS_II_HK)
    trans = valid_colors & (df['w1_w2'] > YSO_CLASS_II_W1W2_MIN) & (df['w1_w2'] < YSO_CLASS_I_W1W2) & (df['H_K'] < YSO_CLASS_II_HK)
    ms = valid_colors & (df['w1_w2'] <= YSO_CLASS_II_W1W2_MIN)
    
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
    """Evaluate BANYAN Σ through MALCA's versioned, schema-stable adapter."""
    out = compute_banyan_membership(
        df,
        association_threshold=BANYAN_MIN_ASSOC_PROB,
    )
    if out.empty:
        return out
    status_counts = out["banyan_status"].value_counts(dropna=False).to_dict()
    print(f"BANYAN Σ statuses: {status_counts}")
    return out


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
    df = _ensure_output_columns(df, IPHAS_CACHE_COLUMNS)
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: IPHAS crossmatch requires ra, dec columns")
        return df

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
                out_idx = df.index[idx]
                r_mag = _row_first_numeric(row, "r", "rmag", "r_mag")
                i_mag = _row_first_numeric(row, "i", "imag", "i_mag")
                ha_mag = _row_first_numeric(row, "ha", "Ha", "Hamag", "Ha_mag")
                r_err = _row_first_numeric(row, "rErr", "e_r", "e_rmag", "r_err")
                i_err = _row_first_numeric(row, "iErr", "e_i", "e_imag", "i_err")
                ha_err = _row_first_numeric(row, "haErr", "e_ha", "e_Ha", "e_Hamag", "ha_err")
                r_i = _row_first_numeric(row, "rmi", "r-i", "r_i")
                r_ha = _row_first_numeric(row, "rmha", "r-ha", "r-Ha", "r_Ha", "r_ha")
                r_i_err = _row_first_numeric(row, "e_rmi", "e_r-i", "e_r_i", "r_i_err")
                r_ha_err = _row_first_numeric(row, "e_rmha", "e_r-ha", "e_r-Ha", "e_r_ha", "r_ha_err")
                
                if not np.isfinite(r_i) and np.isfinite(r_mag) and np.isfinite(i_mag):
                    r_i = r_mag - i_mag
                if not np.isfinite(r_ha) and np.isfinite(r_mag) and np.isfinite(ha_mag):
                    r_ha = r_mag - ha_mag
                if not np.isfinite(r_i_err):
                    r_i_err = _quadrature_error(r_err, i_err)
                if not np.isfinite(r_ha_err):
                    r_ha_err = _quadrature_error(r_err, ha_err)

                df.at[out_idx, 'iphas_r_mag'] = r_mag
                df.at[out_idx, 'iphas_i_mag'] = i_mag
                df.at[out_idx, 'iphas_ha_mag'] = ha_mag
                df.at[out_idx, 'iphas_r_err'] = r_err
                df.at[out_idx, 'iphas_i_err'] = i_err
                df.at[out_idx, 'iphas_ha_err'] = ha_err
                df.at[out_idx, 'iphas_r_i'] = r_i
                df.at[out_idx, 'iphas_r_i_err'] = r_i_err
                df.at[out_idx, 'iphas_r_ha'] = r_ha
                df.at[out_idx, 'iphas_r_ha_err'] = r_ha_err
                df.at[out_idx, 'iphas_sep_arcsec'] = _row_sep_arcsec(row)
                df.at[out_idx, 'iphas_source_catalog'] = "II/321/iphas2"
                
                if np.isfinite(r_ha):
                    df.at[out_idx, 'iphas_ha_excess'] = r_ha > IPHAS_HA_EXCESS_THRESHOLD
            
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
    Crossmatch to VPHAS+ for Hα emission detection in the Southern Galactic Plane.
    
    Uses VPHAS+ DR3 first, then falls back to DR2 for rows without complete
    Hα colors. Returns (r-Hα) and (r-i) colors plus raw r/i/Hα photometry.
    """
    if df.empty:
        return df
    df = df.copy()
    df = _ensure_output_columns(df, VPHAS_CACHE_COLUMNS)
    
    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: VPHAS+ crossmatch requires ra, dec columns")
        return df

    # Prepare source table with unique index for matching back
    valid_mask = df['ra'].notna() & df['dec'].notna()
    if not valid_mask.any():
        return df

    valid_positions = np.where(valid_mask)[0]

    def _source_table_for_positions(positions: np.ndarray) -> Table:
        tab = Table()
        tab['_idx'] = positions
        tab['ra'] = df.iloc[positions]['ra'].values
        tab['dec'] = df.iloc[positions]['dec'].values
        return tab

    def _apply_matches(result, catalog: str, schema: str) -> int:
        if result is None or len(result) == 0:
            return 0
        result_df = result.to_pandas()
        if result_df.empty:
            return 0
        if 'angDist' in result_df.columns:
            result_df = result_df.sort_values('angDist').drop_duplicates(subset='_idx', keep='first')
        else:
            result_df = result_df.drop_duplicates(subset='_idx', keep='first')

        for _, row in result_df.iterrows():
            idx = int(row['_idx'])
            out_idx = df.index[idx]
            if schema == "dr3":
                r_mag = _row_first_numeric(row, "rap3", "rmag", "r_mag")
                i_mag = _row_first_numeric(row, "iap3", "imag", "i_mag")
                ha_mag = _row_first_numeric(row, "Haap3", "Hamag", "Ha_mag")
                r_err = _row_first_numeric(row, "e_rap3", "e_rmag", "r_err")
                i_err = _row_first_numeric(row, "e_iap3", "e_imag", "i_err")
                ha_err = _row_first_numeric(row, "e_Haap3", "e_Hamag", "ha_err")
                r_i = _row_first_numeric(row, "r-ipnt", "r-i", "r_i")
                r_ha = _row_first_numeric(row, "r-Hapnt", "r-ha", "r_Ha", "r_ha")
                r_i_err = _row_first_numeric(row, "e_r-ipnt", "e_r-i", "e_r_i", "r_i_err")
                r_ha_err = _row_first_numeric(row, "e_r-Hapnt", "e_r-ha", "e_r_ha", "r_ha_err")
            else:
                r_mag = _row_first_numeric(row, "rmag", "r_mag", "rap3")
                i_mag = _row_first_numeric(row, "imag", "i_mag", "iap3")
                ha_mag = _row_first_numeric(row, "Hamag", "Ha_mag", "Haap3")
                r_err = _row_first_numeric(row, "e_rmag", "e_rap3", "r_err")
                i_err = _row_first_numeric(row, "e_imag", "e_iap3", "i_err")
                ha_err = _row_first_numeric(row, "e_Hamag", "e_Haap3", "ha_err")
                r_i = _row_first_numeric(row, "r-i", "r-ipnt", "r_i")
                r_ha = _row_first_numeric(row, "r-ha", "r-Hapnt", "r_Ha", "r_ha")
                r_i_err = _row_first_numeric(row, "e_r-i", "e_r-ipnt", "e_r_i", "r_i_err")
                r_ha_err = _row_first_numeric(row, "e_r-ha", "e_r-Hapnt", "e_r_ha", "r_ha_err")

            if not np.isfinite(r_i) and np.isfinite(r_mag) and np.isfinite(i_mag):
                r_i = r_mag - i_mag
            if not np.isfinite(r_ha) and np.isfinite(r_mag) and np.isfinite(ha_mag):
                r_ha = r_mag - ha_mag
            if not np.isfinite(r_i_err):
                r_i_err = _quadrature_error(r_err, i_err)
            if not np.isfinite(r_ha_err):
                r_ha_err = _quadrature_error(r_err, ha_err)

            df.at[out_idx, 'vphas_r_mag'] = r_mag
            df.at[out_idx, 'vphas_i_mag'] = i_mag
            df.at[out_idx, 'vphas_ha_mag'] = ha_mag
            df.at[out_idx, 'vphas_r_err'] = r_err
            df.at[out_idx, 'vphas_i_err'] = i_err
            df.at[out_idx, 'vphas_ha_err'] = ha_err
            df.at[out_idx, 'vphas_r_i'] = r_i
            df.at[out_idx, 'vphas_r_i_err'] = r_i_err
            df.at[out_idx, 'vphas_r_ha'] = r_ha
            df.at[out_idx, 'vphas_r_ha_err'] = r_ha_err
            df.at[out_idx, 'vphas_sep_arcsec'] = _row_sep_arcsec(row)
            df.at[out_idx, 'vphas_source_catalog'] = catalog
            if np.isfinite(r_ha):
                df.at[out_idx, 'vphas_ha_excess'] = r_ha > IPHAS_HA_EXCESS_THRESHOLD
        return len(result_df)

    print(f"Running VPHAS+ DR3 XMatch for {len(valid_positions)} sources...")
    try:
        source_table = _source_table_for_positions(valid_positions)
        result = XMatch.query(
            cat1=source_table,
            cat2='vizier:II/386/vphasplus32',
            max_distance=max_sep_arcsec * u.arcsec,
            colRA1='ra', colDec1='dec',
            colRA2='RAJ2000', colDec2='DEJ2000'
        )
        _apply_matches(result, "II/386/vphasplus32", "dr3")
    except Exception as e:
        print(f"VPHAS+ DR3 XMatch error: {e}")

    complete = pd.to_numeric(df['vphas_r_ha'], errors="coerce").notna() & pd.to_numeric(df['vphas_r_i'], errors="coerce").notna()
    fallback_positions = np.array([pos for pos in valid_positions if not bool(complete.iloc[pos])], dtype=int)
    if len(fallback_positions) > 0:
        print(f"Running VPHAS+ DR2 fallback XMatch for {len(fallback_positions)} sources...")
        try:
            source_table = _source_table_for_positions(fallback_positions)
            result = XMatch.query(
                cat1=source_table,
                cat2='vizier:II/341/vphasp',
                max_distance=max_sep_arcsec * u.arcsec,
                colRA1='ra', colDec1='dec',
                colRA2='RAJ2000', colDec2='DEJ2000'
            )
            _apply_matches(result, "II/341/vphasp", "dr2")
        except Exception as e:
            print(f"VPHAS+ DR2 XMatch error: {e}")

    matched_rows = int(df['vphas_source_catalog'].astype(str).str.len().gt(0).sum())
    if matched_rows:
        ha_excess_count = (df['vphas_ha_excess'] == True).sum()
        complete_count = int((pd.to_numeric(df['vphas_r_ha'], errors="coerce").notna() & pd.to_numeric(df['vphas_r_i'], errors="coerce").notna()).sum())
        print(f"VPHAS+: {matched_rows}/{len(df)} matched, {complete_count} complete color pairs, {ha_excess_count} with Hα excess")
    else:
        print("VPHAS+: No matches found")

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
                
            # Map Vizier columns to internal names. The Sloan-like APASS
            # columns are named g'mag/r'mag/i'mag in VizieR.
            col_map = {
                ("Vmag",): "apass_v", ("e_Vmag",): "apass_v_err",
                ("Bmag",): "apass_b", ("e_Bmag",): "apass_b_err",
                ("g'mag", "g_mag", "gmag"): "apass_g",
                ("e_g'mag", "e_g_mag", "e_gmag"): "apass_g_err",
                ("r'mag", "r_mag", "rmag"): "apass_r",
                ("e_r'mag", "e_r_mag", "e_rmag"): "apass_r_err",
                ("i'mag", "i_mag", "imag"): "apass_i",
                ("e_i'mag", "e_i_mag", "e_imag"): "apass_i_err",
            }
            
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                for viz_cols, my_col in col_map.items():
                    val = _row_first_numeric(row, *viz_cols)
                    if np.isfinite(val):
                        df.at[df.index[idx], my_col] = val
            
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
                ("FUV", "FUVmag"): "galex_fuv",
                ("e_FUV", "e_FUVmag"): "galex_fuv_err",
                ("NUV", "NUVmag"): "galex_nuv",
                ("e_NUV", "e_NUVmag"): "galex_nuv_err",
            }
            
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                for viz_cols, my_col in col_map.items():
                    val = _row_first_numeric(row, *viz_cols)
                    if np.isfinite(val):
                        df.at[df.index[idx], my_col] = val
            
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
    for col in ["w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err", *ALLWISE_QUALITY_COLUMNS]:
        if col not in df.columns:
            if col in ALLWISE_TEXT_QUALITY_COLUMNS:
                df[col] = pd.Series("", index=df.index, dtype=object)
            else:
                df[col] = np.nan
        elif col in ALLWISE_TEXT_QUALITY_COLUMNS:
            df[col] = df[col].astype(object)
        
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
            numeric_quality_map = {
                "angDist": "allwise_sep_arcsec",
                "ex": "allwise_ext_flg",
                "nb": "allwise_nb",
                "na": "allwise_na",
                **{f"snr{band}": f"allwise_w{band}_snr" for band in range(1, 5)},
                **{f"chi2W{band}": f"allwise_w{band}_rchi2" for band in range(1, 5)},
                **{f"sat{band}": f"allwise_w{band}_sat" for band in range(1, 5)},
                **{f"nW{band}": f"allwise_w{band}_ndet" for band in range(1, 5)},
                **{f"mW{band}": f"allwise_w{band}_nframe" for band in range(1, 5)},
            }
            text_quality_map = {
                "AllWISE": "allwise_id",
                "qph": "allwise_ph_qual",
                "ccf": "allwise_cc_flags",
                "var": "allwise_var_flg",
            }
            
            for _, row in result_df.iterrows():
                idx = int(row['_idx'])
                for viz_col, my_col in col_map.items():
                    val = row.get(viz_col, np.nan)
                    if pd.notna(val):
                        df.at[df.index[idx], my_col] = float(val)
                for viz_col, my_col in numeric_quality_map.items():
                    val = _row_first_numeric(row, viz_col)
                    if np.isfinite(val):
                        df.at[df.index[idx], my_col] = float(val)
                for viz_col, my_col in text_quality_map.items():
                    val = row.get(viz_col, None)
                    if val is not None and not np.ma.is_masked(val) and str(val).strip() not in {"", "nan", "--"}:
                        df.at[df.index[idx], my_col] = str(val).strip()
            
            print(f"AllWISE: {len(result_df)} matches found")
    except Exception as e:
        print(f"AllWISE XMatch error: {e}")
        
    return _add_wise_color_columns(df)


def query_allwise_vizier(df: pd.DataFrame, max_sep_arcsec: float = ALLWISE_MAX_SEP_ARCSEC) -> pd.DataFrame:
    """Query AllWISE directly through VizieR for a small coordinate batch.

    This is the reliable fallback for targeted work when the CDS XMatch upload
    endpoint returns an invalid/non-VOTable response.  VizieR's ``_q`` column
    preserves the association with each input coordinate.
    """

    if df.empty:
        return df
    df = _canonicalize_wise_columns(df)
    for col in ["w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err", *ALLWISE_QUALITY_COLUMNS]:
        if col not in df.columns:
            if col in ALLWISE_TEXT_QUALITY_COLUMNS:
                df[col] = pd.Series("", index=df.index, dtype=object)
            else:
                df[col] = np.nan
        elif col in ALLWISE_TEXT_QUALITY_COLUMNS:
            df[col] = df[col].astype(object)

    valid_mask = df["ra"].notna() & df["dec"].notna()
    if not valid_mask.any():
        return _add_wise_color_columns(df)
    valid_indices = df.index[valid_mask].tolist()
    coordinates = SkyCoord(
        pd.to_numeric(df.loc[valid_indices, "ra"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df.loc[valid_indices, "dec"], errors="coerce").to_numpy(dtype=float),
        unit="deg",
    )
    columns = [
        "AllWISE", "RAJ2000", "DEJ2000",
        "W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag", "e_W3mag", "W4mag", "e_W4mag",
        "qph", "ccf", "ex", "var", "nb", "na",
        *[f"snr{band}" for band in range(1, 5)],
        *[f"chi2W{band}" for band in range(1, 5)],
        *[f"sat{band}" for band in range(1, 5)],
        *[f"nW{band}" for band in range(1, 5)],
        *[f"mW{band}" for band in range(1, 5)],
        "+_r",
    ]
    print(f"Running direct AllWISE VizieR query for {len(valid_indices)} sources...")
    try:
        tables = Vizier(columns=columns, row_limit=-1, timeout=60).query_region(
            coordinates,
            radius=max_sep_arcsec * u.arcsec,
            catalog="II/328/allwise",
        )
        if not tables:
            return _add_wise_color_columns(df)
        result_df = tables[0].to_pandas()
        if result_df.empty or "_q" not in result_df:
            return _add_wise_color_columns(df)
        distance_col = "_r" if "_r" in result_df else None
        if distance_col:
            result_df = result_df.sort_values(distance_col)
        result_df = result_df.drop_duplicates(subset="_q", keep="first")
        photometry_map = {
            "W1mag": "w1", "e_W1mag": "w1_err", "W2mag": "w2", "e_W2mag": "w2_err",
            "W3mag": "w3", "e_W3mag": "w3_err", "W4mag": "w4", "e_W4mag": "w4_err",
        }
        numeric_quality_map = {
            "_r": "allwise_sep_arcsec", "ex": "allwise_ext_flg", "nb": "allwise_nb", "na": "allwise_na",
            **{f"snr{band}": f"allwise_w{band}_snr" for band in range(1, 5)},
            **{f"chi2W{band}": f"allwise_w{band}_rchi2" for band in range(1, 5)},
            **{f"sat{band}": f"allwise_w{band}_sat" for band in range(1, 5)},
            **{f"nW{band}": f"allwise_w{band}_ndet" for band in range(1, 5)},
            **{f"mW{band}": f"allwise_w{band}_nframe" for band in range(1, 5)},
        }
        text_quality_map = {
            "AllWISE": "allwise_id", "qph": "allwise_ph_qual",
            "ccf": "allwise_cc_flags", "var": "allwise_var_flg",
        }
        matched = 0
        for _, row in result_df.iterrows():
            query_index = int(row["_q"]) - 1
            if query_index < 0 or query_index >= len(valid_indices):
                continue
            output_index = valid_indices[query_index]
            for viz_col, output_col in photometry_map.items():
                value = _row_first_numeric(row, viz_col)
                if np.isfinite(value):
                    df.at[output_index, output_col] = float(value)
            for viz_col, output_col in numeric_quality_map.items():
                value = _row_first_numeric(row, viz_col)
                if np.isfinite(value):
                    df.at[output_index, output_col] = float(value)
            for viz_col, output_col in text_quality_map.items():
                value = row.get(viz_col, None)
                if value is not None and not np.ma.is_masked(value) and str(value).strip() not in {"", "nan", "--"}:
                    df.at[output_index, output_col] = str(value).strip()
            matched += 1
        print(f"Direct AllWISE VizieR: {matched}/{len(valid_indices)} matched")
    except Exception as exc:
        print(f"Direct AllWISE VizieR error: {exc}")
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
    df['unwise_w1_var'] = key.map(ckpt_idx['unwise_w1_var']).astype("boolean").fillna(False).astype(bool)

    n_var = int(df['unwise_w1_var'].sum())
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
    out[f"char_updated_at_{module}"] = pd.Timestamp.utcnow().isoformat()
    out["characterization_status_version"] = CHARACTERIZE_STATUS_VERSION
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
        return _set_module_state(df, module, "disabled", "")

    print(description)
    try:
        out = func(df, **kwargs)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{module} did not return a pandas DataFrame")
        cache_columns = [
            col for col in out.columns
            if col.startswith("char_cache_status_") and col not in df.columns
        ]
        if cache_columns:
            cache_values = out[cache_columns].fillna("").astype(str)
            any_data = cache_values.apply(
                lambda row: any(value in {"fetched", "cached_hit"} for value in row), axis=1
            )
            any_no_data = cache_values.apply(
                lambda row: any(value in {"fetched_no_data", "cached_miss", "cached_no_data"} for value in row), axis=1
            )
            out = _set_module_state(out, module, "ok", "")
            out.loc[~any_data & any_no_data, f"char_status_{module}"] = "no_data"
            return out
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
    # ``skipped`` is a legacy terminal state written by older checkpoints when
    # an optional query was intentionally omitted.  Keep accepting it so those
    # checkpoints remain resumable, while the new ``disabled`` state is
    # deliberately non-terminal and will not masquerade as completed science.
    if len(vals) == 0 or any(v not in ("ok", "no_data", "not_applicable", "skipped") for v in vals):
        return False
    if "ok" in set(vals):
        required = MODULE_COMPLETION_COLUMNS.get(module, [])
        missing = [out_col for out_col in required if out_col not in df.columns]
        if missing:
            return False
        if module == "banyan":
            status = df["banyan_status"].fillna("").astype(str)
            eligible = gaia_banyan_input_mask(df)
            if (eligible & status.ne("ok")).any():
                return False
    return True


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

    df = df.copy()
    # Fallback for missing ra/dec but present ra_gaia/dec_gaia
    if "ra" not in df.columns and "ra_gaia" in df.columns:
        df["ra"] = df["ra_gaia"]
    if "dec" not in df.columns and "dec_gaia" in df.columns:
        df["dec"] = df["dec_gaia"]
    if "gaia_id" in df.columns:
        df = canonicalize_gaia_ids_in_frame(
            df,
            gaia_cache_path=cache,
            chunk_size=chunk_size,
            warn=True,
        )

    def _merge_missing_gaia_columns(frame: pd.DataFrame) -> pd.DataFrame:
        identifiers = gaia_identifier_series(frame)
        gaia_ids = _normalize_source_ids(identifiers.dropna().tolist())
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
        return merge_gaia_catalog_rows(frame, gaia_df)

    # If source_id + coordinates already present (e.g. from SkyPatrol fetch),
    # we can skip the crossmatch step and use source_id directly for Gaia enrichment.
    _has_gaia_already = (
        gaia_identifier_series(df).notna().any()
        and "ra" in df.columns
        and "dec" in df.columns
    )
    if not _has_gaia_already and "asas_sn_id" not in df.columns:
        print("Warning: characterize skipped: missing 'asas_sn_id' (and no source_id+coords)")
        return df

    # Load checkpoint if available
    df_char = None
    loaded_checkpoint = False
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            df_char = pd.read_parquet(checkpoint_path)
            loaded_checkpoint = True
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
        if "source_id" not in df_char.columns:
            df_char["source_id"] = gaia_identifier_series(df_char)
        if gaia_enrichment_needed_mask(df_char).any():
            missing_count = int(gaia_enrichment_needed_mask(df_char).sum())
            print(f"Gaia photometry/astrometry incomplete for {missing_count} row(s); loading Gaia catalog rows")
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

    if loaded_checkpoint and df_char is not None and gaia_enrichment_needed_mask(df_char).any():
        missing_count = int(gaia_enrichment_needed_mask(df_char).sum())
        print(f"Checkpoint Gaia enrichment incomplete for {missing_count} row(s); loading Gaia catalog rows")
        df_char = _merge_missing_gaia_columns(df_char)
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

                    print("Merging Gaia results by canonical Gaia DR3 ID...")
                    df_char = merge_gaia_catalog_rows(df_merged, gaia_df)

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
                    df_char = _set_module_state(df_char, "starhorse", "no_data", "no StarHorse rows returned")
            except Exception as e:
                msg = str(e)
                print(f"Warning: characterize module 'starhorse' failed: {msg}")
                df_char = _set_module_state(df_char, "starhorse", "error", msg)
        else:
            df_char = _set_module_state(df_char, "starhorse", "disabled", "")
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
                if "yso_input_status" in df_char.columns:
                    missing_yso = df_char["yso_input_status"].astype(str) != "ok"
                    df_char.loc[missing_yso, "char_status_yso"] = "no_data"
                    df_char.loc[missing_yso, "char_error_yso"] = "missing IR photometry"
            except Exception as e:
                msg = str(e)
                print(f"Warning: characterize module 'yso' failed: {msg}")
                df_char = _set_module_state(df_char, "yso", "error", msg)
        else:
            print("Warning: IR photometry columns not found for YSO classification")
            df_char = _set_module_state(df_char, "yso", "no_data", "missing IR photometry")
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

    module_status_columns = sorted(
        col for col in df_char.columns if col.startswith("char_status_")
    )
    if module_status_columns:
        module_status = df_char[module_status_columns].fillna("unknown").astype(str)
        any_error = module_status.eq("error").any(axis=1)
        any_ok = module_status.eq("ok").any(axis=1)
        any_no_data = module_status.eq("no_data").any(axis=1)
        all_disabled = module_status.eq("disabled").all(axis=1)
        df_char["characterization_status"] = "ok"
        df_char.loc[any_no_data, "characterization_status"] = "partial"
        df_char.loc[~any_ok & any_no_data, "characterization_status"] = "no_data"
        df_char.loc[all_disabled, "characterization_status"] = "disabled"
        df_char.loc[any_error, "characterization_status"] = "error"
        df_char["characterization_modules_json"] = [
            json.dumps(
                {
                    col.removeprefix("char_status_"): str(value)
                    for col, value in zip(module_status_columns, row)
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in module_status.itertuples(index=False, name=None)
        ]
    else:
        df_char["characterization_status"] = "not_run"
        df_char["characterization_modules_json"] = "{}"
    df_char["characterization_status_version"] = CHARACTERIZE_STATUS_VERSION

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
