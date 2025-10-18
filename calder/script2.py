import argparse
from pathlib import Path

from lc_dips import naive_dip_finder, MAG_BINS

def main(mag_bins, **kwargs):
    naive_dip_finder(mag_bins=mag_bins, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run naive_dip_finder across bins.")
    parser.add_argument(
        "--mag-bin",
        dest="mag_bins",
        action="append",
        choices=MAG_BINS,
        help="Specify bins to run; omit to process all.",
    )
    parser.add_argument("--out-dir", default="./peak_results")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel workers (processes). Default: min(32, CPU-2)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Rows per CSV flush")
    args = parser.parse_args()
    bins = args.mag_bins or MAG_BINS
    main(
        bins,
        out_dir=args.out_dir,
        out_format=args.format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )
