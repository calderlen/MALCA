from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from malca.config import PARQUET_OUTPUT_COMPRESSION
from malca.config import (
    DEFAULT_OUTPUT_DIR,
    MALCA_LCV2_ROOT_ENV,
    require_lcv2_root,
)
from malca.config import WORKERS, MAG_BINS

IDX_PATTERN = re.compile(r"index(\d+)\.csv$", re.IGNORECASE)
MANIFEST_COLUMNS = [
    "source_id",
    "mag_bin",
    "index_num",
    "index_csv",
    "lc_dir",
    "lc_dir_exists",
    "dat_path",
    "dat_exists",
]


def _normalize_file_ext(file_ext: str | None) -> str:
    from malca.config import LIGHT_CURVE_FILE_EXTENSION

    ext = file_ext or LIGHT_CURVE_FILE_EXTENSION
    return str(ext).lstrip(".")


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _sort_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_manifest()
    return df.sort_values(["mag_bin", "source_id"], na_position="last").reset_index(drop=True)


def _load_index_metadata(index_file: Path, id_column: str) -> pd.DataFrame:
    index_file = Path(index_file).expanduser()
    if index_file.suffix.lower() != ".parquet":
        raise ValueError(f"Flat-directory metadata must be a Parquet file: {index_file}")
    df = pd.read_parquet(index_file)

    if id_column not in df.columns:
        raise ValueError(f"Index file {index_file} is missing required column {id_column!r}")

    keep_cols = [id_column]
    keep_cols.extend(col for col in ("mag_bin", "index_num", "index_csv") if col in df.columns)
    meta = df.loc[:, keep_cols].copy()
    meta[id_column] = meta[id_column].astype("string")
    meta = meta.dropna(subset=[id_column]).copy()
    meta[id_column] = meta[id_column].astype(str)
    if "mag_bin" in meta.columns:
        meta["mag_bin"] = meta["mag_bin"].astype("string")
        meta.loc[meta["mag_bin"].isna(), "mag_bin"] = pd.NA
    return meta.drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)


def _iter_flat_light_curve_entries(
    flat_lc_dir: Path,
    *,
    mag_bins: Sequence[str],
    id_column: str,
    file_ext: str | None = None,
    index_file: Path | None = None,
    show_progress: bool = True,
) -> Iterable[dict[str, object]]:
    file_ext = _normalize_file_ext(file_ext)
    flat_lc_dir = Path(flat_lc_dir).expanduser()

    if not flat_lc_dir.exists():
        raise FileNotFoundError(f"Flat light-curve directory not found: {flat_lc_dir}")
    if not flat_lc_dir.is_dir():
        raise NotADirectoryError(f"Flat light-curve path is not a directory: {flat_lc_dir}")

    metadata_by_id: dict[str, dict[str, object]] = {}
    if index_file is not None:
        meta_df = _load_index_metadata(index_file, id_column)
        metadata_by_id = meta_df.set_index(id_column).to_dict(orient="index")

    requested_mag_bins = [str(mb) for mb in mag_bins]
    requested_mag_bin_set = set(requested_mag_bins)
    default_mag_bin = requested_mag_bins[0] if len(requested_mag_bins) == 1 else None

    lc_paths = sorted(flat_lc_dir.glob(f"*.{file_ext}"))
    if not lc_paths and show_progress:
        tqdm.write(f"[warn] no *.{file_ext} light curves found in {flat_lc_dir}")

    unresolved_mag_bin = 0
    for lc_path in tqdm(lc_paths, desc="flat light curves", disable=not show_progress):
        source_id = lc_path.stem
        metadata = metadata_by_id.get(source_id, {})

        mag_bin_value = metadata.get("mag_bin")
        if mag_bin_value is None:
            mag_bin = default_mag_bin
        elif pd.isna(mag_bin_value):
            mag_bin = None
        else:
            mag_bin = str(mag_bin_value)
        if mag_bin is not None and requested_mag_bin_set and mag_bin not in requested_mag_bin_set:
            continue
        if mag_bin is None:
            unresolved_mag_bin += 1
        index_csv_value = metadata.get("index_csv")
        index_csv = None if pd.isna(index_csv_value) else index_csv_value
        if index_csv is None and index_file is not None:
            index_csv = index_file

        yield {
            "source_id": source_id,
            "mag_bin": mag_bin,
            "index_num": metadata.get("index_num"),
            "index_csv": str(index_csv) if index_csv is not None else None,
            "lc_dir": str(flat_lc_dir),
            "lc_dir_exists": True,
            "dat_path": str(lc_path),
            "dat_exists": True,
        }

    if unresolved_mag_bin and show_progress:
        tqdm.write(
            f"[warn] {unresolved_mag_bin} flat light curves lacked mag_bin metadata; "
            "downstream mag-bin filters will treat them as unparsed."
        )


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
    file_ext = _normalize_file_ext(file_ext)
    
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
    index_root: Path | None,
    lc_root: Path | None,
    *,
    mag_bins: Sequence[str],
    id_column: str,
    file_ext: str | None = None,
    show_progress: bool = True,
    n_workers: int = 1,
    flat_lc_dir: Path | None = None,
    index_file: Path | None = None,
) -> pd.DataFrame:
    seen: dict[str, dict[str, object]] = {}
    duplicates = 0

    if flat_lc_dir is not None:
        record_iter = _iter_flat_light_curve_entries(
            flat_lc_dir,
            mag_bins=mag_bins,
            id_column=id_column,
            file_ext=file_ext,
            index_file=index_file,
            show_progress=show_progress,
        )
    else:
        if index_root is None or lc_root is None:
            raise ValueError("index_root and lc_root are required unless flat_lc_dir is provided")
        record_iter = iter_light_curve_entries(
            index_root,
            lc_root,
            mag_bins,
            id_column=id_column,
            file_ext=file_ext,
            show_progress=show_progress,
            n_workers=n_workers,
        )

    for record in record_iter:
        source_id = record["source_id"]
        if source_id in seen:
            duplicates += 1
            continue
        seen[source_id] = record

    if duplicates:
        tqdm.write(f"[warn] skipped {duplicates} duplicate source_id entries")

    if not seen:
        return _empty_manifest()

    df = pd.DataFrame(seen.values())
    return _sort_manifest(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manifest that maps ASAS-SN IDs to their indexed or flat light-curve locations."
        )
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help=(
            "Root directory that contains <mag_bin>/index*.csv files. "
            f"Defaults to ${MALCA_LCV2_ROOT_ENV}; ignored when --flat-lc-dir is used."
        ),
    )
    parser.add_argument(
        "--lc-root",
        type=Path,
        default=None,
        help=(
            "Root directory that contains <mag_bin>/lc*_cal/ light-curve folders. "
            f"Defaults to ${MALCA_LCV2_ROOT_ENV}; ignored when --flat-lc-dir is used."
        ),
    )
    parser.add_argument(
        "--flat-lc-dir",
        type=Path,
        default=None,
        help="Flat directory of <source_id>.<extension> files, such as bundle_assets/lightcurves.",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=None,
        help="Optional Parquet metadata file for flat directories. Used to recover per-source mag_bin/index metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "lc_manifest.parquet",
        help="Output Parquet file path. Default: %(default)s",
    )
    parser.add_argument(
        "--mag-bin",
        nargs="+",
        dest="mag_bins",
        help="Limit processing to specific mag bins.",
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
        help="Disable progress bars.",
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
    index_root = None if args.flat_lc_dir else require_lcv2_root(args.index_root)
    lc_root = None if args.flat_lc_dir else require_lcv2_root(args.lc_root)
    df = build_manifest(
        index_root=index_root,
        lc_root=lc_root,
        mag_bins=mag_bins,
        id_column=args.id_column,
        file_ext=args.extension,
        show_progress=not args.no_progress,
        n_workers=max(1, args.workers),
        flat_lc_dir=args.flat_lc_dir.expanduser() if args.flat_lc_dir else None,
        index_file=args.index_file.expanduser() if args.index_file else None,
    )

    out_path = args.output.expanduser()
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    print(f"Wrote {len(df):,} entries to {out_path}")


if __name__ == "__main__":
    main()
