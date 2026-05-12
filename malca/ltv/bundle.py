"""Bundle light curve files for LTV candidates passing slope/diff filters."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from malca.config import LTV_MIN_SLOPE, LTV_MIN_DIFF
from malca.ltv.filter import filter_slope_threshold, filter_max_diff_threshold
from malca.table_io import read_parquet_table


def _load_ltv_table(path: Path) -> pd.DataFrame:
    return read_parquet_table(path)


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
) -> tuple[list[Path], int, int]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    total_rows = 0
    total_passing_rows = 0

    for file_path in input_files:
        df = _load_ltv_table(file_path)
        n0 = len(df)
        total_rows += n0
        if verbose:
            print(f"Loaded {n0} rows from {file_path}")

        df = filter_slope_threshold(df, min_slope=min_slope, verbose=verbose)
        df = filter_max_diff_threshold(df, min_diff=min_diff, verbose=verbose)
        total_passing_rows += len(df)

        if "lc_path" not in df.columns:
            raise KeyError(
                f"Input file must contain an 'lc_path' column (missing in {file_path})"
            )

        for lc_path in df["lc_path"].dropna().astype(str).unique():
            resolved = Path(lc_path).expanduser()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_paths.append(resolved)

    return unique_paths, total_rows, total_passing_rows


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
    paths, total_rows, total_passing_rows = _collect_lightcurve_paths(
        input_files,
        min_slope=min_slope,
        min_diff=min_diff,
        verbose=verbose,
    )
    if verbose:
        print(
            f"{len(paths)} unique light curve files after filtering "
            f"({total_rows} -> {total_passing_rows} rows across {len(input_files)} file(s))"
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    missing = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for lc_path in paths:
            p = Path(lc_path)
            if not p.exists():
                missing += 1
                if verbose:
                    print(f"  warning: missing {p}")
                continue
            zf.write(p, arcname=f"lightcurves/{p.name}")
            added += 1

    print(f"Wrote {added} files to {output_zip}" + (f" ({missing} missing)" if missing else ""))
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle light curve files for LTV candidates passing slope/diff filters.",
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
    parser.add_argument("--min-slope", type=float, default=LTV_MIN_SLOPE, help="Minimum |Slope| threshold (mag/yr)")
    parser.add_argument("--min-diff", type=float, default=LTV_MIN_DIFF, help="Minimum |max diff| threshold (mag)")
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
