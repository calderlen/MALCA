from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION
from malca.config.config_paths import LCV2_ROOT, DEFAULT_OUTPUT_DIR
from malca.config.config_pipeline import WORKERS, MAG_BINS

IDX_PATTERN = re.compile(r"index(\d+)\.csv$", re.IGNORECASE)


def _process_index_file(csv_path: Path, mag_bin: str, lc_root: Path, id_column: str, file_ext: str) -> list[dict[str, object]]:
    """Read one index CSV and return records for all IDs."""
    records: list[dict[str, object]] = []
    match = IDX_PATTERN.search(csv_path.name)
    if not match:
        return records
    idx_num = int(match.group(1))
    lc_dir = lc_root / mag_bin / f"lc{idx_num}_cal"
    try:
        ids = (
            pd.read_csv(
                csv_path,
                usecols=[id_column],
                dtype={id_column: "string"},
            )[id_column]
            .dropna()
            .astype(str)
            .unique()
        )
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError):
        return records

    for source_id in ids:
        lc_path = lc_dir / f"{source_id}.{file_ext}"
        file_exists = lc_path.exists()
        records.append({
            "source_id": source_id,
            "mag_bin": mag_bin,
            "index_num": idx_num,
            "index_csv": str(csv_path),
            "lc_dir": str(lc_dir),
            "lc_dir_exists": lc_dir.exists(),
            "dat_path": str(lc_path),
            "dat_exists": file_exists,
        })
    return records


def iter_light_curve_entries(
    index_root: Path,
    lc_root: Path,
    mag_bins: Sequence[str],
    *,
    id_column: str = "asas_sn_id",
    file_ext: str | None = None,
    show_progress: bool = True,
    n_workers: int = 1,
) -> Iterable[dict[str, object]]:
    """
    Yield dictionaries that describe each light-curve entry in index files.
    """
    from malca.config.config_io import LIGHT_CURVE_FILE_EXTENSION
    
    if file_ext is None:
        file_ext = LIGHT_CURVE_FILE_EXTENSION
    
    for mag_bin in tqdm(mag_bins, desc="mag bins", disable=not show_progress):
        idx_dir = index_root / mag_bin
        if not idx_dir.exists():
            tqdm.write(f"[warn] missing index dir for {mag_bin}: {idx_dir}")
            continue
        csv_paths = sorted(idx_dir.glob("index*.csv"))
        if not csv_paths:
            tqdm.write(f"[warn] no index CSVs found in {idx_dir}")
            continue

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_process_index_file, csv_path, mag_bin, lc_root, id_column, file_ext): csv_path
                    for csv_path in csv_paths
                }
                pbar = tqdm(total=len(futures), desc=f"{mag_bin} index CSVs", leave=False, disable=not show_progress)
                for fut in as_completed(futures):
                    for rec in fut.result():
                        yield rec
                    if pbar:
                        pbar.update(1)
                if pbar:
                    pbar.close()
        else:
            for csv_path in tqdm(
                csv_paths,
                desc=f"{mag_bin} index CSVs",
                leave=False,
                disable=not show_progress,
            ):
                for rec in _process_index_file(csv_path, mag_bin, lc_root, id_column, file_ext):
                    yield rec


def build_manifest(
    index_root: Path,
    lc_root: Path,
    *,
    mag_bins: Sequence[str],
    id_column: str,
    file_ext: str | None = None,
    show_progress: bool = True,
    n_workers: int = 1,
) -> pd.DataFrame:
    seen: dict[str, dict[str, object]] = {}
    duplicates = 0
    for record in iter_light_curve_entries(
        index_root,
        lc_root,
        mag_bins,
        id_column=id_column,
        file_ext=file_ext,
        show_progress=show_progress,
        n_workers=n_workers,
    ):
        source_id = record["source_id"]
        if source_id in seen:
            duplicates += 1
            continue
        seen[source_id] = record

    if duplicates:
        tqdm.write(f"[warn] skipped {duplicates} duplicate source_id entries")

    if not seen:
        return pd.DataFrame(
            columns=[
                "source_id",
                "mag_bin",
                "index_num",
                "index_csv",
                "lc_dir",
                "lc_dir_exists",
                "dat_path",
                "dat_exists",
            ]
        )

    df = pd.DataFrame(seen.values())
    df = df.sort_values(["mag_bin", "source_id"]).reset_index(drop=True)
    return df


def iter_source_records(
    index_root: Path,
    lc_root: Path,
    mag_bins: Sequence[str],
    *,
    id_column: str = "asas_sn_id",
    file_ext: str | None = None,
    show_progress: bool = True,
    n_workers: int = 1,
) -> Iterable[dict[str, object]]:
    """Compatibility wrapper for older name; use iter_light_curve_entries."""
    yield from iter_light_curve_entries(
        index_root,
        lc_root,
        mag_bins,
        id_column=id_column,
        file_ext=file_ext,
        show_progress=show_progress,
        n_workers=n_workers,
    )


def build_manifest_dataframe(
    index_root: Path,
    lc_root: Path,
    *,
    mag_bins: Sequence[str],
    id_column: str,
    file_ext: str | None = None,
    show_progress: bool = True,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Compatibility wrapper for older name; use build_manifest."""
    return build_manifest(
        index_root,
        lc_root,
        mag_bins=mag_bins,
        id_column=id_column,
        file_ext=file_ext,
        show_progress=show_progress,
        n_workers=n_workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manifest that maps ASAS-SN IDs to their index CSV and light-curve directory."
        )
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=LCV2_ROOT,
        help="Root directory that contains <mag_bin>/index*.csv files.",
    )
    parser.add_argument(
        "--lc-root",
        type=Path,
        default=LCV2_ROOT,
        help="Root directory that contains <mag_bin>/lc*_cal/ light-curve folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "lc_manifest.parquet",
        help="Output Parquet file path. Default: %(default)s",
    )
    parser.add_argument(
        "--mag-bin",
        action="append",
        dest="mag_bins",
        help="Limit processing to specific mag bins. Repeat for multiple bins.",
    )
    parser.add_argument(
        "--id-column",
        default="asas_sn_id",
        help="Column name to read from index CSVs. Default: %(default)s",
    )
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        default=None,
        help="Light curve file extension (e.g., dat, dat2, dat3). Default: dat3 (from config)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Parallel workers to read index CSVs (default: 10, uses ProcessPoolExecutor).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mag_bins = args.mag_bins if args.mag_bins else MAG_BINS
    df = build_manifest(
        index_root=args.index_root.expanduser(),
        lc_root=args.lc_root.expanduser(),
        mag_bins=mag_bins,
        id_column=args.id_column,
        file_ext=args.extension,
        show_progress=not args.no_progress,
        n_workers=max(1, args.workers),
    )

    out_path = args.out.expanduser()
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    print(f"Wrote {len(df):,} entries to {out_path}")


if __name__ == "__main__":
    main()
