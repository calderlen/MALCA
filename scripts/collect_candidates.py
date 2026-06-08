#!/usr/bin/env python3
"""Collect queue candidates across mag bins with ASAS-SN index data.

Scans run directories for enriched/filtered parquets, extracts asas_sn_id
from path stems, joins against the ASAS-SN index parquet, and writes a
single concatenated output parquet.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from malca.config import DEFAULT_OUTPUT_DIR
from malca.table_io import read_feature_table, write_feature_table


def find_best_parquet(run_dir: Path, mag_bin: str) -> Path | None:
    """Return the most-enriched upstream parquet in a run directory.

    Prefers enriched > filtered.
    """
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        return None

    for prefix in ("lc_events_enriched", "lc_events_filtered"):
        # Try with mag bin suffix first, then without
        for name in (f"{prefix}_{mag_bin}.parquet", f"{prefix}.parquet"):
            p = results_dir / name
            if p.exists():
                return p
    return None


def parse_mag_bin(run_dir_name: str) -> str | None:
    """Extract mag bin string from run directory name.

    e.g. 'output_bundle_12_12.5' -> '12_12.5'
    """
    prefix = "output_bundle_"
    if run_dir_name.startswith(prefix):
        suffix = run_dir_name[len(prefix):]
        for marker in ("_home_bundle_", "_bundle_"):
            if marker in suffix:
                return suffix.split(marker, 1)[0]
        return suffix
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect queue candidates with ASAS-SN index data"
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="Path to ASAS-SN index parquet",
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_OUTPUT_DIR / "runs",
        type=Path,
        help=f"Parent directory containing run dirs (default: {DEFAULT_OUTPUT_DIR / 'runs'})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR / "all_candidates.parquet",
        type=Path,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_DIR / 'all_candidates.parquet'})",
    )
    parser.add_argument(
        "--mag-bins",
        nargs="*",
        help="Optional filter to specific mag bins (e.g. 12_12.5 13_13.5)",
    )
    parser.add_argument(
        "--index-cols",
        nargs="*",
        help="Optional subset of index columns to join (default: all)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.index.exists():
        print(f"Error: index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)
    if not args.runs_dir.is_dir():
        print(f"Error: runs directory not found: {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover run directories
    run_dirs = sorted(
        d for d in args.runs_dir.iterdir()
        if d.is_dir() and parse_mag_bin(d.name) is not None
    )
    if not run_dirs:
        print(f"Error: no run directories found in {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Optionally filter mag bins
    if args.mag_bins:
        allowed = set(args.mag_bins)
        run_dirs = [d for d in run_dirs if parse_mag_bin(d.name) in allowed]
        if not run_dirs:
            print(f"Error: no run directories match mag bins: {args.mag_bins}", file=sys.stderr)
            sys.exit(1)

    # Collect candidate dataframes
    dfs = []
    for run_dir in run_dirs:
        mag_bin = parse_mag_bin(run_dir.name)
        pq_path = find_best_parquet(run_dir, mag_bin)
        if pq_path is None:
            print(f"  Skipping {run_dir.name}: no enriched/filtered parquet found")
            continue

        df = read_feature_table(pq_path)
        source = "enriched" if "enriched" in pq_path.name else "filtered"
        print(f"  {run_dir.name}: {len(df)} rows from {source}")

        # Filter to queue candidates
        if "failed_any" in df.columns:
            n_before = len(df)
            df = df[df["failed_any"] == False].copy()  # noqa: E712
            if len(df) < n_before:
                print(f"    Filtered {n_before} -> {len(df)} (failed_any == False)")

        # Extract asas_sn_id from path stem
        if "path" in df.columns:
            df["asas_sn_id"] = df["path"].apply(lambda p: Path(p).stem).astype(str)
        elif "asas_sn_id" not in df.columns:
            print(f"    Warning: no 'path' or 'asas_sn_id' column, skipping")
            continue

        df["mag_bin"] = mag_bin
        dfs.append(df)

    if not dfs:
        print("Error: no candidate data collected", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} candidates across {len(dfs)} mag bins")

    # Read index and join — only load the rows we need to avoid OOM on
    # the full 17M-row index file.
    combined["asas_sn_id"] = combined["asas_sn_id"].astype(str)
    needed_ids = set(combined["asas_sn_id"])
    print(f"Reading index: {args.index} (filtering to {len(needed_ids)} candidate IDs)")

    # Determine which columns to read from the index
    index_schema = pq.read_schema(args.index)
    all_index_cols = index_schema.names
    if args.index_cols:
        join_cols = [c for c in args.index_cols if c in all_index_cols]
        missing = set(args.index_cols) - set(join_cols)
        if missing:
            print(f"  Warning: requested index columns not found: {missing}")
    else:
        join_cols = [c for c in all_index_cols if c != "asas_sn_id"]

    read_cols = ["asas_sn_id"] + join_cols

    # Read only the needed columns, then filter to matching IDs
    index_df = pd.read_parquet(args.index, columns=read_cols)
    index_df["asas_sn_id"] = index_df["asas_sn_id"].astype(str)
    index_df = index_df[index_df["asas_sn_id"].isin(needed_ids)]
    print(f"  Index filtered: {len(index_df)} rows match candidates")

    # Drop any pre-existing index columns to avoid _x/_y suffixes
    existing = [c for c in join_cols if c in combined.columns]
    if existing:
        combined = combined.drop(columns=existing)

    # Preserve gaia_id as exact digit strings
    if "gaia_id" in join_cols and "gaia_id" in index_df.columns:
        gaia_series = pd.to_numeric(index_df["gaia_id"], errors="coerce")
        index_df["gaia_id"] = gaia_series.astype("Int64").astype(str)
        index_df.loc[gaia_series.isna(), "gaia_id"] = pd.NA

    index_df = index_df.drop_duplicates(subset=["asas_sn_id"])
    combined = combined.merge(index_df, on="asas_sn_id", how="left")
    print(f"Joined {len(join_cols)} index columns")

    # Report join stats
    if "ra_deg" in combined.columns:
        matched = combined["ra_deg"].notna().sum()
        print(f"  Coordinate match: {matched}/{len(combined)} ({100*matched/len(combined):.1f}%)")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_feature_table(combined, args.output)
    print(f"\nWrote {len(combined)} candidates to {args.output}")


if __name__ == "__main__":
    main()
