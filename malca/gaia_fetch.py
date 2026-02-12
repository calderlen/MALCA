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
import time
from pathlib import Path

import pandas as pd
import pyvo
from tqdm import tqdm

from malca.config.config_characterize import GAIA_CHUNK_SIZE
from malca.config.config_paths import (
    GAIA_AIP_TAP_URL,
    GAIA_LOCAL_CATALOG,
    VSX_CROSSMATCH_PATH,
)

# Same ADQL query used by the old query_gaia_by_ids() in characterize.py
# ESA Gaia archive hosts gaiadr3.gaia_source, crossmatch tables, and the
# external photometry catalogs (gaiadr1.tmass_original_valid, gaiadr1.allwise_original_valid).
_GAIA_QUERY_TEMPLATE = """
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
    aw.w1mpro_error AS unwise_w1_err, aw.w2mpro_error AS unwise_w2_err

FROM gaiadr3.gaia_source AS g

LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm_tm
    ON g.source_id = xm_tm.source_id
LEFT JOIN gaiadr1.tmass_original_valid AS tm
    ON xm_tm.original_ext_source_id = tm.designation

LEFT JOIN gaiadr3.allwise_best_neighbour AS xm_aw
    ON g.source_id = xm_aw.source_id
LEFT JOIN gaiadr1.allwise_original_valid AS aw
    ON xm_aw.original_ext_source_id = aw.designation

WHERE g.source_id IN ({ids_str})
"""

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


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
        gaia_ids = df["gaia_id"].dropna().astype(str).unique().tolist()
        print(f"Found {len(gaia_ids)} Gaia IDs directly in input file.")
        return [g for g in gaia_ids if g.isdigit()]

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
    df_xmatch = pd.read_csv(xmatch_path, usecols=use_cols, dtype=str)

    df["asas_sn_id"] = df["asas_sn_id"].astype(str)
    df_xmatch["asas_sn_id"] = df_xmatch["asas_sn_id"].astype(str)
    df_merged = df.merge(df_xmatch, on="asas_sn_id", how="left")

    if "gaia_id" not in df_merged.columns:
        raise ValueError("Crossmatch file does not contain 'gaia_id' column.")

    gaia_ids = df_merged["gaia_id"].dropna().astype(str).unique().tolist()
    gaia_ids = [g for g in gaia_ids if g.isdigit()]
    print(f"Extracted {len(gaia_ids)} unique Gaia IDs from {len(df_merged)} candidates.")
    return gaia_ids


def _fetch_chunk(tap_service: pyvo.dal.TAPService, chunk_ids: list[str]) -> pd.DataFrame:
    """Query a single chunk of Gaia IDs with retry logic."""
    ids_str = ",".join(chunk_ids)
    query = _GAIA_QUERY_TEMPLATE.format(ids_str=ids_str)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = tap_service.search(query)
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
                return pd.DataFrame()


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

    results = []
    for i in tqdm(range(0, len(gaia_ids), chunk_size), desc="Gaia DR3 TAP"):
        chunk_ids = gaia_ids[i : i + chunk_size]
        chunk_df = _fetch_chunk(tap_service, chunk_ids)
        if not chunk_df.empty:
            results.append(chunk_df)

    new_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

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
