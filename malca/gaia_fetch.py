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
from malca.candidates import select_passing_candidates_if_present
from malca.gaia_ids import normalize_gaia_source_ids
from malca.table_io import read_parquet_table





# Gaia ADQL query executed via TAP async upload chunks.
_GAIA_QUERY_TEMPLATE = """
SELECT
    g.source_id,
    g.ra, g.dec,
    g.parallax, g.parallax_error, g.ruwe,
    g.pmra, g.pmdec,
    g.radial_velocity,
    g.rv_amplitude_robust,
    g.phot_g_mean_mag, g.bp_rp,
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

GAIA_REQUIRED_COLUMNS = ("source_id",)
GAIA_EXPECTED_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "ruwe",
    "pmra",
    "pmdec",
    "radial_velocity",
    "rv_amplitude_robust",
    "phot_g_mean_mag",
    "bp_rp",
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
)
GAIA_STRING_COLUMNS = {"source_id", "tmass_id", "allwise_id"}
WISE_FETCH_COLUMNS = {"w1", "w1_err", "w2", "w2_err", "w3", "w3_err", "w4", "w4_err"}


def _has_required_gaia_columns(df: pd.DataFrame | None) -> bool:
    return df is not None and all(col in df.columns for col in GAIA_REQUIRED_COLUMNS)


def _has_current_wise_fetch_schema(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    columns = {str(col).lower() for col in df.columns}
    return WISE_FETCH_COLUMNS.issubset(columns)


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


def _normalize_gaia_ids(values: list[object]) -> list[str]:
    """Normalize mixed-type Gaia IDs to digit strings."""
    return normalize_gaia_source_ids(values)


def _extract_gaia_ids(
    input_path: Path,
    crossmatch_path: Path,
    *,
    only_passers: bool = False,
) -> list[str]:
    """Read candidates and merge with VSX crossmatch to get Gaia source IDs."""
    print(f"Loading candidates from {input_path}...")
    df = read_parquet_table(input_path)

    if only_passers:
        df = select_passing_candidates_if_present(df, label="rows", printer=print)

    if "gaia_id" in df.columns:
        # Already has gaia_id (e.g. from a previous merge)
        raw_gaia_ids = df["gaia_id"].dropna().tolist()
        gaia_ids = _normalize_gaia_ids(raw_gaia_ids)
        print(f"Found {len(raw_gaia_ids)} Gaia IDs directly in input file; normalized to {len(gaia_ids)} unique valid IDs.")
        return gaia_ids

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
    xmatch = read_parquet_table(xmatch_path)
    header = xmatch.columns
    use_cols = ["asas_sn_id"] + [
        c for c in xmatch_cols if c in header and c != "asas_sn_id"
    ]
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
    return _has_current_wise_fetch_schema(part)


def _load_checkpoint_parts(checkpoint_dir: Path) -> pd.DataFrame:
    """Load all persisted chunk parquet parts from checkpoint directory."""
    part_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not part_files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in part_files:
        part = pd.read_parquet(path)
        if _has_current_wise_fetch_schema(part):
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
            chunk_df = _ensure_gaia_schema(chunk_df)
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
) -> pd.DataFrame:
    """
    Download Gaia DR3 data for given source IDs, with incremental caching.

    If output_path already exists, IDs present in it are skipped.
    Returns the full catalog (cached + newly fetched).
    """
    output_path = Path(output_path)
    gaia_ids = _normalize_gaia_ids(gaia_ids)
    cached_df = pd.DataFrame()

    checkpoint_dir = _checkpoint_dir_for_output(output_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpointed_ids = _load_checkpointed_ids(checkpoint_dir)
    if checkpointed_ids:
        before = len(gaia_ids)
        gaia_ids = [g for g in gaia_ids if g not in checkpointed_ids]
        print(
            f"  {len(checkpointed_ids)} IDs already checkpointed, "
            f"{len(gaia_ids)} IDs remain after checkpoint resume filtering."
        )

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
            cache_has_current_wise = _has_current_wise_fetch_schema(cached_candidate)
            cached_candidate = _ensure_gaia_schema(cached_candidate)
            if _has_required_gaia_columns(cached_candidate) and cache_has_current_wise and not cached_candidate.empty:
                cached_df = cached_candidate
                cached_df["source_id"] = cached_df["source_id"].astype(str)
                existing_ids = set(cached_df["source_id"])
                before = len(gaia_ids)
                gaia_ids = [g for g in gaia_ids if g not in existing_ids]
                print(f"  {len(existing_ids)} IDs already cached, {len(gaia_ids)} new IDs to fetch.")
                if not gaia_ids:
                    print("All IDs already present in local catalog. Nothing to fetch.")
                    if existing_read_path != output_path:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cached_df.to_parquet(output_path, index=False, compression="snappy")
                        print(f"Migrated Gaia cache to {output_path}")
                    return cached_df
            else:
                print(f"  Warning: ignoring stale or invalid existing Gaia cache at {existing_read_path}")

    if not gaia_ids:
        checkpoint_df = _load_checkpoint_parts(checkpoint_dir)
        if not checkpoint_df.empty:
            checkpoint_df = _ensure_gaia_schema(checkpoint_df)
            if _has_required_gaia_columns(checkpoint_df):
                checkpoint_df["source_id"] = checkpoint_df["source_id"].astype(str)
                if not cached_df.empty:
                    checkpoint_df = pd.concat([cached_df, checkpoint_df], ignore_index=True)
                checkpoint_df = checkpoint_df.drop_duplicates(subset="source_id", keep="last")
                print("No Gaia IDs to fetch; returning checkpointed Gaia rows.")
                return checkpoint_df

        if not cached_df.empty:
            print("No Gaia IDs to fetch.")
            return cached_df

        raise RuntimeError(
            "No Gaia IDs remain to fetch, but no valid Gaia cache rows are available. "
            "Remove stale checkpoint markers and retry."
        )

    print(f"Fetching Gaia DR3 data for {len(gaia_ids)} sources from {GAIA_AIP_TAP_URL}...")
    tap_service = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)

    n_chunks_done = 0
    n_chunks_failed = 0
    n_chunks_empty = 0
    n_rows_written = 0
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
            chunk_df.to_parquet(_chunk_part_path(checkpoint_dir, key), index=False, compression="snappy")
            n_rows_written += len(chunk_df)
        else:
            n_chunks_empty += 1
        n_chunks_done += 1

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

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(output_path, index=False, compression="snappy")
    print(f"Saved {len(full_df)} Gaia rows to {output_path}")
    print(
        "  "
        f"(chunks checkpointed: {n_chunks_done}, chunks failed this run: {n_chunks_failed}, "
        f"chunks empty this run: {n_chunks_empty}, rows written to chunk parts this run: {n_rows_written})"
    )

    fetched = len(new_df) if not new_df.empty else 0
    print(f"  ({fetched} newly fetched, {len(full_df) - fetched} from cache)")

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
