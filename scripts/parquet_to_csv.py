#!/usr/bin/env python3
"""Extract asas_sn_id, ra_deg, dec_deg, path from a parquet into CSV."""

import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract columns from parquet to CSV")
    parser.add_argument("input", type=Path, help="Input parquet file")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV (default: input with .csv extension)")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".csv")
    cols = ["path", "asas_sn_id", "mag_bin", "ra_deg", "dec_deg"]
    df = pd.read_parquet(args.input, columns=cols)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")

if __name__ == "__main__":
    main()
