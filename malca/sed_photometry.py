"""CLI for building normalized SED photometry tables."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import pandas as pd

from malca.review.sed import (
    ALL_CATALOG_SOURCES,
    DEFAULT_PIPELINE_SED_SOURCES,
    SED_COLUMNS,
    fetch_sed_photometry,
    resolve_sed_sources,
    upsert_sed_rows,
)
from malca.review.store import db_connect
from malca.table_io import read_parquet_table, write_parquet_table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca sed-photometry",
        description="Fetch and normalize broadband SED photometry for review candidates.",
    )
    parser.add_argument("input", type=Path, help="Input candidate table (.parquet, .csv, or .txt)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output SED Parquet path (default: <input>_sed_photometry.parquet)",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help="Optional review SQLite DB to upsert SED rows into",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="default",
        help=(
            "Comma-separated source keys, 'default', 'far-ir', or 'all'. "
            f"Default: {', '.join(DEFAULT_PIPELINE_SED_SOURCES)}. "
            f"Available: {', '.join(ALL_CATALOG_SOURCES)}"
        ),
    )
    parser.add_argument(
        "--payload-only",
        action="store_true",
        help="Only normalize photometry already present in the candidate table; do not make network catalog queries.",
    )
    return parser


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sed_photometry.parquet")


def _ensure_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    if "candidate_id" in df.columns:
        return df
    if "asas_sn_id" not in df.columns:
        return df
    out = df.copy()
    out["candidate_id"] = out["asas_sn_id"].astype(str)
    return out


def _read_candidate_table(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(input_path)
    return read_parquet_table(input_path)


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser()
    output_path = (args.output or _default_output_path(input_path)).expanduser()

    df = _read_candidate_table(input_path)
    df = _ensure_candidate_id(df)
    requested_sources = "payload" if args.payload_only else args.sources
    print(f"Loaded {len(df)} candidates from {input_path}")
    print(f"SED sources: {', '.join(resolve_sed_sources(requested_sources))}")

    rows = fetch_sed_photometry(df, sources=requested_sources)
    for col in SED_COLUMNS:
        if col not in rows.columns:
            rows[col] = None
    rows = rows[SED_COLUMNS]
    write_parquet_table(rows, output_path)
    print(f"Saved {len(rows)} SED rows to {output_path}")

    if not rows.empty and "source" in rows.columns:
        counts = rows.groupby("source", dropna=False).size().sort_index()
        print("\nRows by source:")
        for source, count in counts.items():
            print(f"  {source}: {count}")

    if args.review_db:
        review_db = args.review_db.expanduser()
        with closing(db_connect(review_db)) as conn:
            updated = upsert_sed_rows(conn, rows)
        print(f"\nUpserted {updated} SED rows into {review_db}")

    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
