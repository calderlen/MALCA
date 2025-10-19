from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import re
import fnmatch
import pandas as pd

from df_process_naive import filter_csv


BIN_RE = re.compile(
    r"^peaks_(?P<low>\d+(?:_\d)?)_(?P<high>\d+(?:_\d)?)_(?P<ts>\d{8}_\d{6}[+-]\d{4})\.csv$"
)

def parse_bin_key(p: Path) -> tuple[str, str, str] | None:
    """Return (low, high, ts) strings from filename like peaks_12_5_13_20251018_155817-0400.csv"""
    m = BIN_RE.match(p.name)
    if not m:
        return None
    return (m.group("low"), m.group("high"), m.group("ts"))

def latest_per_bin(files: list[Path]) -> list[Path]:
    """Keep the newest timestamped file per (low, high) bin."""
    best: dict[tuple[str, str], tuple[str, Path]] = {}
    for f in files:
        parsed = parse_bin_key(f)
        if not parsed:
            continue
        low, high, ts = parsed
        key = (low, high)
        prev = best.get(key)
        if prev is None or ts > prev[0]:  # lex compare works with YYYYmmdd_HHMMSS±ZZZZ
            best[key] = (ts, f)
    return [f for (_, f) in best.values()]

def gather_files(
    directory: Path,
    files: list[str] | None,
    includes: list[str] | None,
    excludes: list[str] | None,
    keep_latest: bool,
) -> list[Path]:
    # start with explicit files/globs or default peaks_*.csv
    candidates: set[Path] = set()
    if files:
        for pat in files:
            candidates.update(directory.glob(pat))
    else:
        candidates.update(directory.glob("peaks_*.csv"))

    # include filters (apply on basename)
    if includes:
        filtered = set()
        for f in candidates:
            if any(fnmatch.fnmatch(f.name, pat) for pat in includes):
                filtered.add(f)
        candidates = filtered

    # exclude filters (apply on basename)
    if excludes:
        candidates = {
            f for f in candidates if not any(fnmatch.fnmatch(f.name, pat) for pat in excludes)
        }

    # keep only files that look like peaks_* if we’re deduping per bin
    files_list = sorted(candidates)
    if keep_latest:
        files_list = latest_per_bin(files_list)

    return files_list

def run_one(
    file_path: Path,
    args,
    ts: str,
    *,
    out_dir: Path | None = None,
    out_path_override: Path | None = None,
) -> pd.DataFrame:
    """Run filter_csv for a single input CSV and write a timestamped output CSV."""
    if out_path_override is not None:
        out_csv_path = out_path_override
    else:
        target_dir = out_dir if out_dir is not None else file_path.parent
        out_csv_path = target_dir / f"{file_path.stem}_filtered_{ts}.csv"

    df = filter_csv(
        csv_path=file_path,
        out_csv_path=out_csv_path,
        band=args.band,
        asassn_csv=args.asassn_csv,
        vsx_csv=args.vsx_csv,
        min_dip_fraction=args.min_dip_fraction,
        min_cameras=args.min_cameras,
        max_power=args.max_power,
        min_period=args.min_period,
        max_period=args.max_period,
        min_time_span=200.0,       # defaults preserved even if filters are disabled
        min_points_per_day=0.05,
        min_sigma=3.0,
        match_radius_arcsec=args.match_radius_arcsec,
        apply_bns=args.apply_bns,
        apply_vsx_class=args.apply_vsx_class,
        apply_dip_dom=args.apply_dip_dom,
        apply_multi_camera=args.apply_multi_camera,
        apply_periodic=args.apply_periodic,
        apply_sparse=args.apply_sparse,
        apply_sigma=args.apply_sigma,
        seed_workers=args.seed_workers,
        tqdm_position_base=0,
    )

    print(f"[OK] {file_path} → {out_csv_path} ({len(df)} rows)")
    return df

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run filter_csv on one peaks CSV or on selected peaks_*.csv in a directory; outputs mirror source."
    )
    p.add_argument("csv_path", type=Path,
                   help="Input peaks CSV path OR a directory containing peaks_*.csv files.")
    # selection controls (only used when csv_path is a directory)
    p.add_argument("--files", nargs="+", default=None,
                   help="Specific file globs within the directory (e.g., 'peaks_12_5_13_*.csv').")
    p.add_argument("--include", action="append", default=None,
                   help="Additional basename globs to include; can be given multiple times.")
    p.add_argument("--exclude", action="append", default=None,
                   help="Basename globs to exclude; can be given multiple times.")
    p.add_argument("--latest-per-bin", action="store_true",
                   help="If multiple files exist per mag bin, keep only the latest (by timestamp in filename).")
    p.add_argument("--no-combined", action="store_true",
                   help="Do not write a combined CSV when processing a directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the files that would be processed and exit.")

    # pipeline knobs
    p.add_argument("--band", default="either", choices=["g", "v", "both", "either"])
    p.add_argument("--asassn-csv", default="results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv")
    p.add_argument("--vsx-csv", default="results_crossmatch/vsx_cleaned_20250926_1557.csv")
    p.add_argument("--min-dip-fraction", type=float, default=0.66)
    p.add_argument("--min-cameras", type=int, default=2)
    p.add_argument("--max-power", type=float, default=0.5)
    p.add_argument("--min-period", type=float, default=None)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--match-radius-arcsec", type=float, default=3.0)
    p.add_argument("--seed-workers", type=int, default=1,
                   help="Worker processes for the initial candidates_with_peaks_naive stage (default: 1).")
    p.add_argument("--no-bns", dest="apply_bns", action="store_false", default=True,
                   help="Disable the bright-nearby-star (catalog) join.")
    p.add_argument("--no-vsx-class", dest="apply_vsx_class", action="store_false", default=True,
                   help="Disable VSX variability class enrichment.")
    p.add_argument("--no-dip-dom", dest="apply_dip_dom", action="store_false", default=False,
                   help="Disable the dip-dominated fraction filter.")
    p.add_argument("--no-multi-camera", dest="apply_multi_camera", action="store_false", default=True,
                   help="Disable the multi-camera filter.")
    p.add_argument("--no-periodic", dest="apply_periodic", action="store_false", default=False,
                   help="Disable the periodicity filter.")
    p.add_argument("--no-sparse", dest="apply_sparse", action="store_false", default=False,
                   help="Disable the sparse light-curve filter.")
    p.add_argument("--no-sigma", dest="apply_sigma", action="store_false", default=False,
                   help="Disable the sigma-based depth filter.")
    p.add_argument("--output", type=Path, default=None,
                   help="Destination CSV path when processing a single file.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to place output CSVs (per-file and combined).")

    args = p.parse_args()
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")

    in_path = args.csv_path
    output_override = args.output.expanduser() if args.output is not None else None
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else None

    if output_override and in_path.is_dir():
        p.error("--output is only valid when processing a single input file.")
    if output_dir and output_dir.exists() and not output_dir.is_dir():
        p.error("--output-dir must be a directory.")

    # Case 1: directory — select subset, mirror outputs next to each input
    if in_path.is_dir():
        files = gather_files(
            directory=in_path,
            files=args.files,
            includes=args.include,
            excludes=args.exclude,
            keep_latest=args.latest_per_bin,
        )
        if not files:
            p.error(f"No matching files in {in_path}")

        if args.dry_run:
            print("Files that would be processed:")
            for f in files:
                print("  ", f)
            return 0

        dfs: list[pd.DataFrame] = []
        for f in files:
            dfs.append(run_one(f, args, ts, out_dir=output_dir))

        if not args.no_combined:
            combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            combined_dir = output_dir if output_dir is not None else in_path
            combined_csv = combined_dir / f"peaks_all_filtered_{ts}.csv"
            combined_dir.mkdir(parents=True, exist_ok=True)
            combined.to_csv(combined_csv, index=False)
            print(f"[COMBINED] {combined_csv} ({len(combined)} rows)")
        return 0

    # Case 2: single file (or a stem your module resolves) — mirror output next to it
    if in_path.is_file() or not in_path.exists():
        df = run_one(
            in_path,
            args,
            ts,
            out_dir=output_dir,
            out_path_override=output_override,
        )
        print(f"Filtered rows: {len(df)}")
        return 0

    p.error(f"{in_path} is neither a file nor a directory")


if __name__ == "__main__":
    raise SystemExit(main())
