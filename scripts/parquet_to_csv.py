#!/usr/bin/env python3
"""Extract asas_sn_id, ra_deg, dec_deg, path from a parquet into CSV."""

import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract columns from parquet to CSV")
    parser.add_argument("input", type=Path, help="Input parquet file")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV (default: input with .csv extension)")
    parser.add_argument("--all", action="store_true", help="Output all columns instead of just the summary columns")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".csv")
    
    # Read the parquet file
    df = pd.read_parquet(args.input)
    
    # Ensure asas_sn_id exists if path is available
    if "asas_sn_id" not in df.columns and "path" in df.columns:
        df["asas_sn_id"] = df["path"].apply(lambda p: Path(p).stem).astype(str)
        
    if not args.all:
        desired_cols = ["path", "asas_sn_id", "mag_bin", "ra_deg", "dec_deg"]
        # Only select columns that actually exist in the dataframe
        cols = [c for c in desired_cols if c in df.columns]
        df = df[cols]
        
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")

if __name__ == "__main__":
    main()
