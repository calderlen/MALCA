"""
Bulk-download Gaia DR3 data via the AIP TAP mirror.

Reads a candidate Parquet file (post-filter output), extracts Gaia source IDs
via the VSX crossmatch, and downloads astrometry + photometry in chunks.
Saves results as a local Parquet catalog for offline use by characterize.py.

Usage:
    malca gaia-fetch --input output/runs/20260101/results/lc_events_filtered.parquet
    malca gaia-fetch --input candidates.parquet --output my_gaia_catalog.parquet
    malca gaia-fetch --input candidates.parquet --crossmatch input/vsx/my_crossmatch.csv
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyvo
from astropy.table import Table
from tqdm import tqdm

from malca.config.config_characterize import GAIA_CHUNK_SIZE
from malca.config.config_paths import (
    GAIA_AIP_TAP_URL,
    GAIA_LOCAL_CATALOG,
    VSX_CROSSMATCH_PATH,
)

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
    tm.j_m AS tmass_j, tm.h_m AS tmass_h, tm.ks_m AS tmass_k,
    tm.j_msigcom AS tmass_j_err, tm.h_msigcom AS tmass_h_err, tm.ks_msigcom AS tmass_k_err,

    xm_aw.original_ext_source_id AS allwise_id,
    aw.w1mpro AS unwise_w1, aw.w2mpro AS unwise_w2,
    aw.w1mpro_error AS unwise_w1_err, aw.w2mpro_error AS unwise_w2_err

FROM TAP_UPLOAD.upload_table AS u
JOIN gaiadr3.gaia_source AS g
    ON g.source_id = u.source_id

LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm_tm
    ON g.source_id = xm_tm.source_id
LEFT JOIN gaiadr1.tmass_original_valid AS tm
    ON xm_tm.original_ext_source_id = tm.designation

LEFT JOIN gaiadr3.allwise_best_neighbour AS xm_aw
    ON g.source_id = xm_aw.source_id
LEFT JOIN gaiadr1.allwise_original_valid AS aw
    ON xm_aw.original_ext_source_id = aw.designation
"""

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


def _normalize_gaia_ids(values: list[object]) -> list[str]:
    """Normalize mixed-type Gaia IDs to digit strings."""
    normalized: list[str] = []
    seen: set[str] = set()

    for value in values:
        if bool(pd.isna(value)):
            continue

        s = str(value).strip()
        if not s:
            continue

        gid: str | None = s if s.isdigit() else None

        if gid is None:
            continue
        if gid not in seen:
            seen.add(gid)
            normalized.append(gid)

    return normalized


def _extract_gaia_ids(
    input_path: Path,
    crossmatch_path: Path,
) -> list[str]:
    """Read candidates and merge with VSX crossmatch to get Gaia source IDs."""
    print(f"Loading candidates from {input_path}...")
    if str(input_path).endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

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
            "Provide --crossmatch or ensure the default path exists."
        )

    print(f"Loading crossmatch file {xmatch_path}...")
    xmatch_cols = ["asas_sn_id", "gaia_id", "tmass_id", "allwise_id"]
    header = pd.read_csv(xmatch_path, nrows=0).columns
    use_cols = ["asas_sn_id"] + [
        c for c in xmatch_cols if c in header and c != "asas_sn_id"
    ]
    use_cols_set = set(use_cols)
    df_xmatch = pd.read_csv(xmatch_path, usecols=lambda c: c in use_cols_set, dtype=str)

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


def _done_marker_path(checkpoint_dir: Path, key: str) -> Path:
    return checkpoint_dir / f"{key}.done"


def _chunk_part_path(checkpoint_dir: Path, key: str) -> Path:
    return checkpoint_dir / f"{key}.parquet"


def _chunk_is_checkpointed(checkpoint_dir: Path, key: str) -> bool:
    return _done_marker_path(checkpoint_dir, key).exists() or _chunk_part_path(checkpoint_dir, key).exists()


def _write_done_marker_with_ids(checkpoint_dir: Path, key: str, row_count: int, chunk_ids: list[str]) -> None:
    """Write done marker including queried IDs for chunk-size-agnostic resume."""
    marker = _done_marker_path(checkpoint_dir, key)
    marker.write_text(
        json.dumps({"row_count": int(row_count), "ids": chunk_ids}),
        encoding="utf-8",
    )


def _load_checkpoint_parts(checkpoint_dir: Path) -> pd.DataFrame:
    """Load all persisted chunk parquet parts from checkpoint directory."""
    part_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not part_files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)


def _load_checkpointed_ids(checkpoint_dir: Path) -> set[str]:
    """Load Gaia IDs known to be completed from checkpoint parts and markers."""
    completed: set[str] = set()

    parts_df = _load_checkpoint_parts(checkpoint_dir)
    if (not parts_df.empty) and ("source_id" in parts_df.columns):
        completed.update(parts_df["source_id"].dropna().astype(str).tolist())

    for marker in checkpoint_dir.glob("*.done"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            ids = payload.get("ids") if isinstance(payload, dict) else None
            if isinstance(ids, list):
                completed.update(str(x) for x in ids if str(x))
        except Exception:
            continue

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
            # Normalize column names to lowercase
            chunk_df.columns = [c.lower() for c in chunk_df.columns]
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

    # Load existing catalog for incremental fetch
    if output_path.exists():
        print(f"Loading existing Gaia catalog from {output_path}...")
        cached_df = pd.read_parquet(output_path)
        if "source_id" in cached_df.columns:
            cached_df["source_id"] = cached_df["source_id"].astype(str)
            existing_ids = set(cached_df["source_id"])
            before = len(gaia_ids)
            gaia_ids = [g for g in gaia_ids if g not in existing_ids]
            print(f"  {len(existing_ids)} IDs already cached, {len(gaia_ids)} new IDs to fetch.")
            if not gaia_ids:
                print("All IDs already present in local catalog. Nothing to fetch.")
                return cached_df

    if not gaia_ids:
        print("No Gaia IDs to fetch.")
        return cached_df

    print(f"Fetching Gaia DR3 data for {len(gaia_ids)} sources from {GAIA_AIP_TAP_URL}...")
    tap_service = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)

    n_chunks_done = 0
    n_chunks_failed = 0
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
        _write_done_marker_with_ids(checkpoint_dir, key, row_count=len(chunk_df), chunk_ids=chunk_ids)
        n_chunks_done += 1

    checkpoint_df = _load_checkpoint_parts(checkpoint_dir)
    new_df = checkpoint_df.copy()

    if not new_df.empty and "source_id" in new_df.columns:
        new_df["source_id"] = new_df["source_id"].astype(str)

    # Merge with cached data
    if not cached_df.empty and not new_df.empty:
        full_df = pd.concat([cached_df, new_df], ignore_index=True)
    elif not new_df.empty:
        full_df = new_df
    else:
        full_df = cached_df

    # Deduplicate on source_id
    if "source_id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset="source_id", keep="last")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(output_path, index=False, compression="snappy")
    print(f"Saved {len(full_df)} Gaia rows to {output_path}")
    print(
        "  "
        f"(chunks checkpointed: {n_chunks_done}, chunks failed this run: {n_chunks_failed}, "
        f"rows written to chunk parts this run: {n_rows_written})"
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
        help="Candidate Parquet/CSV file (post-filter output with asas_sn_id or gaia_id column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GAIA_LOCAL_CATALOG,
        help=f"Output Parquet path for local Gaia catalog (default: {GAIA_LOCAL_CATALOG})",
    )
    parser.add_argument(
        "--crossmatch",
        type=Path,
        default=VSX_CROSSMATCH_PATH,
        help="Path to ASAS-SN x VSX crossmatch CSV (must contain asas_sn_id and gaia_id)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=GAIA_CHUNK_SIZE,
        help=f"Number of Gaia IDs per TAP query (default: {GAIA_CHUNK_SIZE})",
    )

    args = parser.parse_args()

    # Extract Gaia IDs from input + crossmatch
    gaia_ids = _extract_gaia_ids(args.input, args.crossmatch)

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
