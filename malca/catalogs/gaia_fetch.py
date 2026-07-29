"""
Bulk-download Gaia DR3 data via the AIP TAP mirror.

Reads a candidate Parquet file (filter output), extracts Gaia source IDs
via the VSX crossmatch, and downloads astrometry + photometry in chunks.
Saves results as a local Parquet catalog for offline use by characterize.py.

Usage:
    malca gaia-fetch --input output/runs/20260101/results/lc_events_filtered.parquet
    malca gaia-fetch --input candidates.parquet --output my_gaia_catalog.parquet
    malca gaia-fetch --input candidates.parquet --vsx-crossmatch input/vsx/my_crossmatch.parquet
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import time

from astropy.table import Table
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyvo

from malca.config import GAIA_CHUNK_SIZE
from malca.config import (
    GAIA_AIP_TAP_URL,
    GAIA_LOCAL_CATALOG,
    LEGACY_GAIA_CACHE_FILE,
    LEGACY_GAIA_LOCAL_CATALOG,
    VSX_CROSSMATCH_PATH,
)
from malca.products.candidates import select_passing_candidates_if_present
from malca.products.feature_layers import with_feature_columns
from malca.catalogs.gaia_ids import canonicalize_gaia_ids, normalize_gaia_source_ids
from malca.io.table_io import (
    is_layer_first_table,
    read_feature_table,
    read_parquet_table,
)





# Gaia ADQL query executed via TAP async upload chunks.
_GAIA_QUERY_TEMPLATE = """
SELECT
    g.source_id,
    g.ra, g.dec, g.ref_epoch,
    g.parallax, g.parallax_error, g.parallax_over_error, g.ruwe,
    g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
    g.parallax_pmra_corr, g.parallax_pmdec_corr, g.pmra_pmdec_corr,
    g.astrometric_params_solved,
    g.astrometric_excess_noise, g.astrometric_excess_noise_sig,
    g.astrometric_n_good_obs_al, g.astrometric_sigma5d_max,
    g.visibility_periods_used,
    g.ipd_frac_multi_peak, g.ipd_frac_odd_win,
    g.ipd_gof_harmonic_amplitude,
    g.duplicated_source,
    g.radial_velocity, g.radial_velocity_error,
    g.rv_amplitude_robust,
    g.rv_nb_transits, g.rv_chisq_pvalue, g.rv_renormalised_gof,
    g.rv_time_duration, g.rv_method_used, g.grvs_mag,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
    g.phot_bp_rp_excess_factor,
    g.phot_bp_n_obs, g.phot_rp_n_obs,
    g.phot_bp_n_blended_transits, g.phot_rp_n_blended_transits,
    g.phot_bp_n_contaminated_transits, g.phot_rp_n_contaminated_transits,
    g.non_single_star, g.phot_variable_flag,
    g.has_epoch_photometry, g.has_epoch_rv, g.has_rvs,
    g.teff_gspphot, g.logg_gspphot, g.mh_gspphot,
    g.distance_gspphot, g.ag_gspphot,

    xm_tm.original_ext_source_id AS tmass_id,
    xm_aw.original_ext_source_id AS allwise_id,
    aw.w1mpro AS w1,
    aw.w1sigmpro AS w1_err,
    aw.w2mpro AS w2,
    aw.w2sigmpro AS w2_err,
    aw.w3mpro AS w3,
    aw.w3sigmpro AS w3_err,
    aw.w4mpro AS w4,
    aw.w4sigmpro AS w4_err

FROM TAP_UPLOAD.upload_table AS u
JOIN gaiadr3.gaia_source AS g
    ON g.source_id = u.source_id

LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm_tm
    ON g.source_id = xm_tm.source_id

LEFT JOIN gaiadr3.allwise_best_neighbour AS xm_aw
    ON g.source_id = xm_aw.source_id

LEFT JOIN catalogs.allwise AS aw
    ON xm_aw.allwise_oid = aw.allwise_oid
"""

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
GAIA_FETCH_SCHEMA_VERSION = "4"

GAIA_REQUIRED_COLUMNS = ("source_id",)
GAIA_EXPECTED_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "ruwe",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
    "astrometric_params_solved",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "astrometric_n_good_obs_al",
    "astrometric_sigma5d_max",
    "visibility_periods_used",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "ipd_gof_harmonic_amplitude",
    "duplicated_source",
    "radial_velocity",
    "radial_velocity_error",
    "rv_amplitude_robust",
    "rv_nb_transits",
    "rv_chisq_pvalue",
    "rv_renormalised_gof",
    "rv_time_duration",
    "rv_method_used",
    "grvs_mag",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "phot_bp_rp_excess_factor",
    "phot_bp_n_obs",
    "phot_rp_n_obs",
    "phot_bp_n_blended_transits",
    "phot_rp_n_blended_transits",
    "phot_bp_n_contaminated_transits",
    "phot_rp_n_contaminated_transits",
    "non_single_star",
    "phot_variable_flag",
    "has_epoch_photometry",
    "has_epoch_rv",
    "has_rvs",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "distance_gspphot",
    "ag_gspphot",
    "tmass_id",
    "tmass_j",
    "tmass_h",
    "tmass_k",
    "tmass_j_err",
    "tmass_h_err",
    "tmass_k_err",
    "allwise_id",
    "w1",
    "w1_err",
    "w2",
    "w2_err",
    "w3",
    "w3_err",
    "w4",
    "w4_err",
    "gaia_fetch_schema_version",
    "gaia_fetch_updated_at",
)
GAIA_STRING_COLUMNS = {
    "source_id",
    "tmass_id",
    "allwise_id",
    "phot_variable_flag",
    "gaia_fetch_schema_version",
    "gaia_fetch_updated_at",
}
WISE_FETCH_COLUMNS = {"w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err"}
GAIA_CURRENT_FETCH_COLUMNS = WISE_FETCH_COLUMNS | {
    "parallax_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "pmra_error",
    "pmdec_error",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
    "radial_velocity_error",
    "ref_epoch",
    "astrometric_params_solved",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "astrometric_n_good_obs_al",
    "astrometric_sigma5d_max",
    "visibility_periods_used",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "ipd_gof_harmonic_amplitude",
    "duplicated_source",
    "rv_nb_transits",
    "rv_chisq_pvalue",
    "rv_renormalised_gof",
    "rv_time_duration",
    "rv_method_used",
    "grvs_mag",
    "phot_bp_rp_excess_factor",
    "phot_bp_n_obs",
    "phot_rp_n_obs",
    "phot_bp_n_blended_transits",
    "phot_rp_n_blended_transits",
    "phot_bp_n_contaminated_transits",
    "phot_rp_n_contaminated_transits",
    "non_single_star",
    "phot_variable_flag",
    "has_epoch_photometry",
    "has_epoch_rv",
    "has_rvs",
}
GAIA_BANYAN_REQUIRED_COLUMNS = (
    "ra",
    "dec",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
)


def _has_required_gaia_columns(df: pd.DataFrame | None) -> bool:
    return df is not None and all(col in df.columns for col in GAIA_REQUIRED_COLUMNS)


def _has_current_gaia_fetch_schema(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    columns = {str(col).lower() for col in df.columns}
    return GAIA_CURRENT_FETCH_COLUMNS.issubset(columns)


def _current_schema_row_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows known to have been queried with the current Gaia schema."""
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    if "gaia_fetch_schema_version" in df.columns:
        version = df["gaia_fetch_schema_version"].fillna("").astype(str)
        marked = version.eq(GAIA_FETCH_SCHEMA_VERSION)
        if marked.any():
            return marked
    # Compatibility for complete catalogs written before explicit versioning.
    return pd.Series(_has_current_gaia_fetch_schema(df), index=df.index, dtype=bool)


def gaia_banyan_input_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows with finite, physical minimum inputs for BANYAN Sigma."""
    if df.empty or any(column not in df.columns for column in GAIA_BANYAN_REQUIRED_COLUMNS):
        return pd.Series(False, index=df.index, dtype=bool)
    numeric = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in GAIA_BANYAN_REQUIRED_COLUMNS
    }
    mask = pd.Series(True, index=df.index, dtype=bool)
    for values in numeric.values():
        mask &= values.notna() & np.isfinite(values)
    mask &= numeric["ra"].between(0.0, 360.0, inclusive="left")
    mask &= numeric["dec"].between(-90.0, 90.0, inclusive="both")
    mask &= numeric["pmra_error"] > 0
    mask &= numeric["pmdec_error"] > 0
    return mask


def _ensure_gaia_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Gaia cache schema for downstream consumers."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    legacy_wise_columns = {
        "unwise_w1": "w1",
        "unwise_w1_err": "w1_err",
        "unwise_w2": "w2",
        "unwise_w2_err": "w2_err",
        "allwise_w3": "w3",
        "allwise_w3_err": "w3_err",
        "allwise_w4": "w4",
        "allwise_w4_err": "w4_err",
    }
    for old_col, new_col in legacy_wise_columns.items():
        if old_col in out.columns and new_col not in out.columns:
            out = out.rename(columns={old_col: new_col})

    if not _has_required_gaia_columns(out):
        return out

    for col in GAIA_EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA if col in GAIA_STRING_COLUMNS else np.nan

    return out.loc[:, list(GAIA_EXPECTED_COLUMNS)]


def _mark_current_fetch_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_gaia_schema(df)
    if out.empty:
        return out
    out["gaia_fetch_schema_version"] = GAIA_FETCH_SCHEMA_VERSION
    out["gaia_fetch_updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return out


def _atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = output_path.stat().st_mode & 0o777 if output_path.exists() else 0o644
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp.parquet", dir=output_path.parent
    )
    os.close(fd)
    try:
        df.to_parquet(temp_name, index=False, compression="snappy")
        check = pd.read_parquet(temp_name)
        if len(check) != len(df) or not _has_required_gaia_columns(check):
            raise RuntimeError("Atomic Gaia cache validation failed")
        os.chmod(temp_name, output_mode)
        os.replace(temp_name, output_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _write_fetch_report(output_path: Path, payload: dict[str, object]) -> Path:
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_mode = report_path.stat().st_mode & 0o777 if report_path.exists() else 0o644
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, report_mode)
        os.replace(temp_name, report_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return report_path


def _normalize_gaia_ids(values: list[object]) -> list[str]:
    """Normalize mixed-type Gaia IDs to digit strings."""
    return normalize_gaia_source_ids(values)


def _parquet_column_names(path: Path) -> list[str] | None:
    """Return Parquet column names without reading row data."""
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return None


def _read_candidate_id_columns(
    input_path: Path,
    *,
    only_passers: bool,
) -> pd.DataFrame:
    """Read only columns needed to derive Gaia IDs from a candidate table."""
    columns = _parquet_column_names(input_path)
    if columns is None:
        df = read_feature_table(input_path)
        if only_passers:
            df = select_passing_candidates_if_present(df, label="rows", printer=print)
        return df

    if is_layer_first_table(input_path):
        df = with_feature_columns(read_feature_table(input_path), ["gaia_id", "source_id", "failed_any"])
        if only_passers:
            df = select_passing_candidates_if_present(df, label="rows", printer=print)
        if "gaia_id" in df.columns and df["gaia_id"].notna().any():
            return df
        if "source_id" in df.columns and df["source_id"].notna().any():
            df["gaia_id"] = df["source_id"]
            return df
        if "asas_sn_id" in df.columns:
            return df
        raise ValueError(
            f"Input file {input_path} has neither 'gaia_id' nor 'asas_sn_id' column."
        )

    if "gaia_id" in columns:
        wanted = ["gaia_id"]
    elif "asas_sn_id" in columns:
        wanted = ["asas_sn_id"]
    else:
        raise ValueError(
            f"Input file {input_path} has neither 'gaia_id' nor 'asas_sn_id' column."
        )

    raise ValueError(f"Input file is not layer-first: {input_path}. Run 'malca migrate' before Gaia fetch.")


def _extract_gaia_ids(
    input_path: Path,
    crossmatch_path: Path,
    *,
    only_passers: bool = False,
) -> list[str]:
    """Read candidates and merge with VSX crossmatch to get Gaia source IDs."""
    print(f"Loading candidates from {input_path}...")
    df = _read_candidate_id_columns(input_path, only_passers=only_passers)

    if "gaia_id" in df.columns:
        # Already has gaia_id (e.g. from a previous merge)
        raw_gaia_ids = df["gaia_id"].dropna().tolist()
        gaia_ids = _normalize_gaia_ids(raw_gaia_ids)
        if gaia_ids:
            print(f"Found {len(raw_gaia_ids)} Gaia IDs directly in input file; normalized to {len(gaia_ids)} unique valid IDs.")
            return gaia_ids
        if "asas_sn_id" not in df.columns:
            print(f"Found {len(raw_gaia_ids)} Gaia IDs directly in input file, but none were valid.")
        else:
            df = df.drop(columns=["gaia_id"])

    if "asas_sn_id" not in df.columns:
        raise ValueError(
            f"Input file {input_path} has neither 'gaia_id' nor 'asas_sn_id' column."
        )

    xmatch_path = crossmatch_path.expanduser()
    if not xmatch_path.exists():
        raise FileNotFoundError(
            f"VSX crossmatch file not found: {xmatch_path}. "
            "Provide --vsx-crossmatch or ensure the default path exists."
        )

    print(f"Loading crossmatch file {xmatch_path}...")
    xmatch_cols = ["asas_sn_id", "gaia_id", "tmass_id", "allwise_id"]
    header = _parquet_column_names(xmatch_path)
    if header is None:
        xmatch = read_parquet_table(xmatch_path)
        header = xmatch.columns
    else:
        xmatch = None
    use_cols = ["asas_sn_id"] + [
        c for c in xmatch_cols if c in header and c != "asas_sn_id"
    ]
    if xmatch is None:
        xmatch = read_parquet_table(xmatch_path, columns=use_cols)
    df_xmatch = xmatch[use_cols].astype(str)

    df["asas_sn_id"] = df["asas_sn_id"].astype(str)
    df_xmatch["asas_sn_id"] = df_xmatch["asas_sn_id"].astype(str)
    df_merged = df.merge(df_xmatch, on="asas_sn_id", how="left")

    if "gaia_id" not in df_merged.columns:
        raise ValueError("Crossmatch file does not contain 'gaia_id' column.")

    gaia_ids = _normalize_gaia_ids(df_merged["gaia_id"].dropna().tolist())
    print(f"Extracted {len(gaia_ids)} unique Gaia IDs from {len(df_merged)} candidates.")
    return gaia_ids


def _checkpoint_dir_for_output(output_path: Path) -> Path:
    """Return directory used for durable chunk checkpoints."""
    return output_path.parent / f"{output_path.stem}.chunks"


def _chunk_key(chunk_ids: list[str]) -> str:
    """Stable key for a chunk based on exact ID membership and order."""
    payload = ",".join(chunk_ids).encode("ascii")
    hasher = hashlib.sha1()
    hasher.update(payload)
    return hasher.hexdigest()[:16]


def _chunk_part_path(checkpoint_dir: Path, key: str) -> Path:
    return checkpoint_dir / f"{key}.parquet"


def _chunk_is_checkpointed(checkpoint_dir: Path, key: str) -> bool:
    path = _chunk_part_path(checkpoint_dir, key)
    if not path.exists():
        return False
    try:
        part = pd.read_parquet(path)
    except Exception:
        return False
    return _has_current_gaia_fetch_schema(part)


def _load_checkpoint_parts(checkpoint_dir: Path) -> pd.DataFrame:
    """Load all persisted chunk parquet parts from checkpoint directory."""
    part_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not part_files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in part_files:
        part = pd.read_parquet(path)
        if _has_current_gaia_fetch_schema(part):
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_checkpointed_ids(checkpoint_dir: Path) -> set[str]:
    """Load Gaia IDs known to be completed from checkpoint parts."""
    completed: set[str] = set()

    parts_df = _load_checkpoint_parts(checkpoint_dir)
    if (not parts_df.empty) and ("source_id" in parts_df.columns):
        completed.update(_normalize_gaia_ids(parts_df["source_id"].dropna().tolist()))

    return completed


def _chunk_upload_table(chunk_ids: list[str]) -> Table:
    """Build TAP upload table for one chunk of Gaia source IDs."""
    return Table({"source_id": [int(x) for x in chunk_ids]})


def _fetch_chunk(tap_service: pyvo.dal.TAPService, chunk_ids: list[str]) -> pd.DataFrame | None:
    """Query a single chunk of Gaia IDs with retry logic (async TAP upload)."""
    query = _GAIA_QUERY_TEMPLATE

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            upload_table = _chunk_upload_table(chunk_ids)
            result = tap_service.run_async(query, uploads={"upload_table": upload_table})
            chunk_df = result.to_table().to_pandas()
            chunk_df = _mark_current_fetch_rows(chunk_df)
            if not _has_required_gaia_columns(chunk_df):
                raise RuntimeError("Gaia TAP response missing required 'source_id' column")
            return chunk_df
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** attempt
                print(f"  Chunk query failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Chunk query failed after {MAX_RETRIES} attempts: {e}")
                return None


def fetch_gaia_catalog(
    gaia_ids: list[str],
    output_path: Path,
    chunk_size: int = GAIA_CHUNK_SIZE,
    *,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """
    Download Gaia DR3 data for given source IDs, with incremental caching.

    If ``output_path`` already exists, requested IDs already queried with the
    current schema are skipped. Legacy rows are preserved, while requested
    legacy rows are refreshed so newly required astrometric columns are filled.
    Returns the full catalog (cached + newly fetched).
    """
    output_path = Path(output_path)
    requested_ids = _normalize_gaia_ids(gaia_ids)
    gaia_ids = list(requested_ids)
    if requested_ids:
        mapping = canonicalize_gaia_ids(
            requested_ids,
            gaia_cache_path=output_path,
            chunk_size=chunk_size,
            warn=True,
        )
        if not mapping.empty:
            translated = int(mapping["gaia_id_mapping_status"].eq("dr2_translated").sum())
            gaia_ids = _normalize_gaia_ids(mapping["source_id"].dropna().tolist())
            if translated:
                print(f"Translated {translated} Gaia DR2 ID(s) to DR3 before Gaia fetch.")
    requested_ids = list(gaia_ids)
    cached_df = pd.DataFrame()
    current_cached_ids: set[str] = set()

    checkpoint_dir = _checkpoint_dir_for_output(output_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load existing catalog for incremental fetch.  New writes always go to
    # output_path, but default cache reads can fall back to pre-unification
    # cache locations once.
    read_candidates = [output_path]
    if output_path == GAIA_LOCAL_CATALOG:
        read_candidates.extend([LEGACY_GAIA_LOCAL_CATALOG, LEGACY_GAIA_CACHE_FILE])

    existing_read_path = next((path for path in read_candidates if path.exists()), None)
    if existing_read_path is not None:
        print(f"Loading existing Gaia catalog from {existing_read_path}...")
        try:
            cached_candidate = pd.read_parquet(existing_read_path)
        except Exception as e:
            print(f"  Warning: could not read existing Gaia cache at {existing_read_path}: {e}")
        else:
            current_mask = _current_schema_row_mask(cached_candidate)
            cached_candidate = _ensure_gaia_schema(cached_candidate)
            if _has_required_gaia_columns(cached_candidate) and not cached_candidate.empty:
                cached_df = cached_candidate
                cached_df["source_id"] = cached_df["source_id"].map(
                    lambda value: _normalize_gaia_ids([value])[0]
                    if _normalize_gaia_ids([value]) else pd.NA
                )
                cached_df = cached_df.dropna(subset=["source_id"])
                current_cached_ids = set(
                    cached_df.loc[current_mask.reindex(cached_df.index, fill_value=False), "source_id"]
                    .dropna()
                    .astype(str)
                )
            else:
                print(f"  Warning: ignoring invalid existing Gaia cache at {existing_read_path}")

    checkpointed_ids = _load_checkpointed_ids(checkpoint_dir)
    completed_ids = current_cached_ids | checkpointed_ids
    gaia_ids = [gaia_id for gaia_id in requested_ids if gaia_id not in completed_ids]
    if cached_df.empty:
        print(f"  No valid existing Gaia rows; {len(gaia_ids)} IDs require fetching.")
    else:
        existing_ids = set(cached_df["source_id"].dropna().astype(str))
        legacy_requested = set(requested_ids) & existing_ids - current_cached_ids
        print(
            f"  {len(existing_ids)} IDs cached; {len(current_cached_ids)} current-schema; "
            f"{len(legacy_requested)} requested legacy row(s) require refresh; "
            f"{len(gaia_ids)} requested ID(s) remain."
        )

    n_chunks_done = 0
    n_chunks_failed = 0
    n_chunks_empty = 0
    n_rows_written = 0
    if gaia_ids:
        print(f"Fetching Gaia DR3 data for {len(gaia_ids)} sources from {GAIA_AIP_TAP_URL}...")
        tap_service = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)
        for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia DR3 TAP"):
            chunk_ids = gaia_ids[i : i + chunk_size]
            key = _chunk_key(chunk_ids)

            if _chunk_is_checkpointed(checkpoint_dir, key):
                n_chunks_done += 1
                continue

            chunk_df = _fetch_chunk(tap_service, chunk_ids)
            if chunk_df is None:
                n_chunks_failed += 1
                continue

            if not chunk_df.empty:
                chunk_df.to_parquet(
                    _chunk_part_path(checkpoint_dir, key), index=False, compression="snappy"
                )
                n_rows_written += len(chunk_df)
            else:
                n_chunks_empty += 1
            n_chunks_done += 1

    if n_chunks_failed and not allow_partial:
        raise RuntimeError(
            f"Gaia fetch left {n_chunks_failed} failed chunk(s); checkpoints were preserved, "
            "but the canonical cache was not replaced. Retry the command or pass "
            "allow_partial=True only for diagnostic work."
        )

    checkpoint_df = _load_checkpoint_parts(checkpoint_dir)
    new_df = _ensure_gaia_schema(checkpoint_df) if not checkpoint_df.empty else checkpoint_df.copy()

    if not new_df.empty and _has_required_gaia_columns(new_df):
        new_df["source_id"] = new_df["source_id"].astype(str)

    # Merge with cached data
    if not cached_df.empty and not new_df.empty:
        full_df = pd.concat([cached_df, new_df], ignore_index=True)
    elif not new_df.empty:
        full_df = new_df
    else:
        full_df = cached_df

    # Deduplicate on source_id
    if not full_df.empty:
        full_df = _ensure_gaia_schema(full_df)

    if _has_required_gaia_columns(full_df):
        full_df = full_df.drop_duplicates(subset="source_id", keep="last")

    if full_df.empty or not _has_required_gaia_columns(full_df):
        raise RuntimeError(
            "Gaia fetch produced no valid rows; leaving the previous cache untouched."
        )

    # Save atomically only after the merged cache can be read back.
    _atomic_write_parquet(full_df, output_path)
    print(f"Saved {len(full_df)} Gaia rows to {output_path}")
    print(
        "  "
        f"(chunks checkpointed: {n_chunks_done}, chunks failed this run: {n_chunks_failed}, "
        f"chunks empty this run: {n_chunks_empty}, rows written to chunk parts this run: {n_rows_written})"
    )

    fetched = len(new_df) if not new_df.empty else 0
    print(f"  ({fetched} newly fetched, {len(full_df) - fetched} from cache)")

    returned_ids = set(full_df["source_id"].dropna().astype(str))
    requested_returned = returned_ids & set(requested_ids)
    requested_rows = full_df[full_df["source_id"].astype(str).isin(requested_returned)]
    report = {
        "schema_version": GAIA_FETCH_SCHEMA_VERSION,
        "output_path": str(output_path),
        "requested_unique": len(requested_ids),
        "already_current": len(set(requested_ids) & current_cached_ids),
        "refresh_requested": len(gaia_ids),
        "requested_returned": len(requested_returned),
        "requested_not_returned": sorted(set(requested_ids) - requested_returned),
        "banyan_input_complete": int(gaia_banyan_input_mask(requested_rows).sum()),
        "cache_rows_before": int(len(cached_df)),
        "cache_rows_after": int(len(full_df)),
        "chunks_completed": n_chunks_done,
        "chunks_failed": n_chunks_failed,
        "chunks_empty": n_chunks_empty,
        "rows_written_to_checkpoints": n_rows_written,
    }
    report_path = _write_fetch_report(output_path, report)
    print(f"Saved Gaia fetch report to {report_path}")

    return full_df


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download Gaia DR3 data for MALCA candidates via AIP TAP mirror"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Candidate Parquet file (filter output with asas_sn_id or gaia_id column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GAIA_LOCAL_CATALOG,
        help=f"Output Parquet path for local Gaia catalog (default: {GAIA_LOCAL_CATALOG})",
    )
    parser.add_argument(
        "--vsx-crossmatch",
        type=Path,
        default=VSX_CROSSMATCH_PATH,
        help="Path to ASAS-SN x VSX crossmatch Parquet (must contain asas_sn_id and gaia_id)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=GAIA_CHUNK_SIZE,
        help=f"Number of Gaia IDs per TAP query (default: {GAIA_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Fetch Gaia rows for all input candidates instead of only failed_any=False passers.",
    )

    args = parser.parse_args()

    # Extract Gaia IDs from input + crossmatch
    gaia_ids = _extract_gaia_ids(
        args.input,
        args.vsx_crossmatch,
        only_passers=not getattr(args, "all_candidates", False),
    )

    if not gaia_ids:
        print("No Gaia IDs found. Nothing to fetch.")
        return

    # Fetch and save
    fetch_gaia_catalog(
        gaia_ids,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
