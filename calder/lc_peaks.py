from __future__ import annotations

import argparse

from lc_excursions import MAG_BINS, lc_proc, microlensing_peak_finder

__all__ = ["MAG_BINS", "lc_proc", "microlensing_peak_finder"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run microlensing peak finder across bins.")
    parser.add_argument(
        "--mag-bin",
        dest="mag_bins",
        action="append",
        choices=MAG_BINS,
        help="Specify bins to run; omit to process all.",
    )
    parser.add_argument("--out-dir", default="./results_microlensing")
    parser.add_argument("--format", choices=("parquet", "csv"), default="csv")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel processes")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per CSV flush")
    args = parser.parse_args()
    bins = args.mag_bins or MAG_BINS
    microlensing_peak_finder(
        mag_bins=bins,
        out_dir=args.out_dir,
        out_format=args.format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )
