#!/usr/bin/env python3
"""Export one or more Parquet tables to CSV for sharing/readability."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from malca.table_io import read_feature_table


def _parse_columns(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    columns: list[str] = []
    for value in values:
        columns.extend(part.strip() for part in value.split(",") if part.strip())
    return columns or None


def _iter_parquet_inputs(paths: list[Path]) -> list[tuple[Path, Path | None]]:
    jobs: list[tuple[Path, Path | None]] = []
    for input_path in paths:
        path = input_path.expanduser()
        if path.is_dir():
            jobs.extend((item, path) for item in sorted(path.rglob("*.parquet")))
        elif path.is_file() and path.suffix.lower() == ".parquet":
            jobs.append((path, None))
        elif path.exists():
            raise ValueError(f"Not a .parquet file or directory: {path}")
        else:
            raise FileNotFoundError(f"Input not found: {path}")
    return jobs


def _output_path(input_path: Path, root: Path | None, output_dir: Path | None) -> Path:
    if output_dir is None:
        return input_path.with_suffix(".csv")
    if root is None:
        return output_dir / input_path.with_suffix(".csv").name
    return output_dir / input_path.relative_to(root).with_suffix(".csv")


def convert_one(
    input_path: Path,
    output_path: Path,
    *,
    columns: list[str] | None,
    limit: int | None,
    force: bool,
) -> int:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists, use --force to overwrite: {output_path}")
    df = read_feature_table(input_path, columns=columns)
    if limit is not None:
        df = df.head(max(int(limit), 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Parquet files to CSV for sharing/readability.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Parquet file(s) or directories to convert recursively.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write CSV files under this directory instead of adjacent to inputs.")
    parser.add_argument("--columns", nargs="+", default=None, help="Column names to export, as a comma-separated list or space-separated values.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per output CSV.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files.")
    args = parser.parse_args()

    columns = _parse_columns(args.columns)
    output_dir = args.output_dir.expanduser() if args.output_dir else None
    jobs = _iter_parquet_inputs(args.inputs)
    if not jobs:
        raise SystemExit("No parquet files found.")

    for input_path, root in jobs:
        out = _output_path(input_path, root, output_dir)
        rows = convert_one(input_path, out, columns=columns, limit=args.limit, force=bool(args.force))
        print(f"{input_path} -> {out} ({rows} rows)")


if __name__ == "__main__":
    main()
