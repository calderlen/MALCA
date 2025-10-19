#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse

from df_process_naive import filter_csv


def main() -> int:
    p = argparse.ArgumentParser(description="Run filter_csv and save results to a timestamped CSV.")
    p.add_argument("csv_path", type=Path, help="Input peaks CSV (or stem; module will pick latest match).")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("results_filtered"),
                   help="Directory to write the timestamped CSV (default: results_filtered)")
    p.add_argument("--band", default="either", choices=["g", "v", "both", "either"])
    p.add_argument("--asassn-csv", default="results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv")
    p.add_argument("--vsx-csv", default="results_crossmatch/vsx_cleaned_20250926_1557.csv")
    p.add_argument("--min-dip-fraction", type=float, default=0.66)
    p.add_argument("--min-cameras", type=int, default=2)
    p.add_argument("--max-power", type=float, default=0.5)
    p.add_argument("--min-period", type=float, default=None)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--match-radius-arcsec", type=float, default=3.0)
    p.add_argument("--n-helpers", type=int, default=60)
    p.add_argument("--skip-dip-dom", action="store_true", default=False)
    p.add_argument("--skip-multi-camera", action="store_true", default=False)
    p.add_argument("--skip-periodic", action="store_true", default=False)
    p.add_argument("--chunk-size", type=int, default=None)

    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.csv_path.stem
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = out_dir / f"{stem}_filtered_{ts}.csv"

    # Run the pipeline — returns a DataFrame AND writes to out_csv_path
    df = filter_csv(
        csv_path=args.csv_path,
        out_csv_path=out_csv_path,
        band=args.band,
        asassn_csv=args.asassn_csv,
        vsx_csv=args.vsx_csv,
        min_dip_fraction=args.min_dip_fraction,
        min_cameras=args.min_cameras,
        max_power=args.max_power,
        min_period=args.min_period,
        max_period=args.max_period,
        min_time_span=200.0,       
        min_points_per_day=0.05,   
        min_sigma=3.0,             
        match_radius_arcsec=args.match_radius_arcsec,
        n_helpers=args.n_helpers,
        skip_dip_dom=args.skip_dip_dom,
        skip_multi_camera=args.skip_multi_camera,
        skip_periodic=args.skip_periodic,
        skip_sparse=True,          
        skip_sigma=True,           
        chunk_size=args.chunk_size,
        tqdm_position_base=0,
    )


    print(f"Filtered rows: {len(df)}")
    print(f"Wrote: {out_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
