"""Bundle .dat2 light curve files for LTV candidates passing slope/diff filters."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from malca.config.config_ltv import LTV_MIN_SLOPE, LTV_MIN_DIFF
from malca.ltv.filter import filter_slope_threshold, filter_max_diff_threshold


def export_ltv_bundle(
    input_path: str | Path,
    output_zip: str | Path,
    *,
    min_slope: float = LTV_MIN_SLOPE,
    min_diff: float = LTV_MIN_DIFF,
    verbose: bool = False,
) -> Path:
    """Load LTV core parquet, apply slope+diff filters, zip passing .dat2 files."""
    input_path = Path(input_path).expanduser()
    output_zip = Path(output_zip).expanduser()

    df = pd.read_parquet(input_path)
    n0 = len(df)
    if verbose:
        print(f"Loaded {n0} rows from {input_path}")

    df = filter_slope_threshold(df, min_slope=min_slope, verbose=verbose)
    df = filter_max_diff_threshold(df, min_diff=min_diff, verbose=verbose)

    if "lc_path" not in df.columns:
        raise KeyError("Input parquet must contain an 'lc_path' column (produced by ltv-core)")

    paths = df["lc_path"].dropna().unique()
    if verbose:
        print(f"{len(paths)} unique light curve files after filtering ({n0} → {len(df)} rows)")

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
        description="Bundle .dat2 light curve files for LTV candidates passing slope/diff filters.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to LTV core output parquet (e.g. output/ltv/LTvar12_12.5.parquet)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output zip path (default: <input_stem>_bundle.zip)",
    )
    parser.add_argument("--min-slope", type=float, default=LTV_MIN_SLOPE, help="Minimum |Slope| threshold (mag/yr)")
    parser.add_argument("--min-diff", type=float, default=LTV_MIN_DIFF, help="Minimum |max diff| threshold (mag)")
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
        min_slope=args.min_slope,
        min_diff=args.min_diff,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
