"""Build standalone light-curve ZIPs from existing LTV candidate tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from malca.config import LTV_MIN_SLOPE, LTV_MIN_DIFF
from malca.products.feature_layers import with_feature_columns
from malca.ltv.filter import filter_slope_threshold, filter_max_diff_threshold
from malca.products.run_bundle import BundleFileCollection, collect_candidate_lightcurve_files, export_run_bundle
from malca.io.table_io import read_feature_table


def _load_ltv_table(path: Path) -> pd.DataFrame:
    return read_feature_table(path)


def _discover_input_files(input_path: Path, pattern: str | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if pattern is not None:
        files = sorted(p for p in input_path.glob(pattern) if p.is_file())
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in directory {input_path}"
            )
        return files

    files = sorted(p for p in input_path.glob("*_pipeline.parquet") if p.is_file())
    if not files:
        files = sorted(
            p for p in input_path.glob("LTvar*.parquet")
            if p.is_file() and (not p.name.endswith("_bundle.parquet"))
        )
    if not files:
        raise FileNotFoundError(
            "No LTV parquet files found in directory "
            f"{input_path}. Tried '*_pipeline.parquet' then 'LTvar*.parquet'."
        )
    return files


def _collect_lightcurve_paths(
    input_files: list[Path],
    *,
    min_slope: float,
    min_diff: float,
    verbose: bool,
) -> tuple[BundleFileCollection, int, int]:
    frames: list[pd.DataFrame] = []
    total_rows = 0
    total_passing_rows = 0

    for file_path in input_files:
        df = _load_ltv_table(file_path)
        n0 = len(df)
        total_rows += n0
        if verbose:
            print(f"Loaded {n0} rows from {file_path}")

        df = with_feature_columns(df, ["ltv_slope", "ltv_max_diff", "failed_any"])
        df = filter_slope_threshold(df, min_slope=min_slope, verbose=verbose)
        df = filter_max_diff_threshold(df, min_diff=min_diff, verbose=verbose)
        total_passing_rows += len(df)

        if "lc_path" not in df.columns:
            raise KeyError(
                f"Input file must contain an 'lc_path' column (missing in {file_path})"
            )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    collection = collect_candidate_lightcurve_files(
        combined,
        path_cols=("lc_path",),
        arc_prefix="lightcurves",
    )
    return collection, total_rows, total_passing_rows


def export_ltv_bundle(
    input_path: str | Path,
    output_zip: str | Path,
    *,
    pattern: str | None = None,
    min_slope: float = LTV_MIN_SLOPE,
    min_diff: float = LTV_MIN_DIFF,
    verbose: bool = False,
) -> Path:
    """Bundle passing LTV light-curve files from one file or an LTV output directory."""
    input_path = Path(input_path).expanduser()
    output_zip = Path(output_zip).expanduser()

    input_files = _discover_input_files(input_path, pattern=pattern)
    collection, total_rows, total_passing_rows = _collect_lightcurve_paths(
        input_files,
        min_slope=min_slope,
        min_diff=min_diff,
        verbose=verbose,
    )
    if verbose:
        print(
            f"{collection.added} unique light curve files after filtering "
            f"({total_rows} -> {total_passing_rows} rows across {len(input_files)} file(s))"
        )

    export_run_bundle(
        output_zip,
        input_path if input_path.is_dir() else input_path.parent,
        external_files=collection.files,
        description="LTV light-curve",
    )
    print(
        f"Wrote {collection.added} files to {output_zip}"
        + (f" ({collection.missing} missing)" if collection.missing else "")
    )
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone light-curve ZIP from existing LTV candidate tables. "
            "Use 'malca ltv-pipeline --full-bundle' for a run-integrated bundle."
        ),
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to an LTV output Parquet or a directory containing multiple such files",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern for directory input (default: '*_pipeline.parquet', then 'LTvar*.parquet')",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output zip path (default: <input_stem>_bundle.zip)",
    )
    parser.add_argument("--min-slope", type=float, default=LTV_MIN_SLOPE, help="Minimum |ltv_slope| threshold (mag/yr)")
    parser.add_argument("--min-diff", type=float, default=LTV_MIN_DIFF, help="Minimum |ltv_max_diff| threshold (mag)")
    parser.add_argument("--extension", "-e", type=str, default=None, help="Light curve file extension (e.g., dat, dat2, dat3). Default: dat3 (from config)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if args.output:
        output_zip = Path(args.output).expanduser()
    else:
        output_zip = input_path.with_name(f"{input_path.stem}_bundle.zip")

    export_ltv_bundle(
        input_path,
        output_zip,
        pattern=args.pattern,
        min_slope=args.min_slope,
        min_diff=args.min_diff,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
