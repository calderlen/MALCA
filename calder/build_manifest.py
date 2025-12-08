from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm import tqdm

from lc_excursions import MAG_BINS, lc_dir_masked

IDX_PATTERN = re.compile(r"index(\d+)_masked\.csv$", re.IGNORECASE)


def _iter_source_records(
    index_root: Path,
    lc_root: Path,
    mag_bins: Sequence[str],
    *,
    id_column: str = "asas_sn_id",
    show_progress: bool = True,
) -> Iterable[dict[str, object]]:
    """
    Yield dictionaries that describe each source found in the masked index files.
    """
    for mag_bin in tqdm(mag_bins, desc="mag bins", disable=not show_progress):
        idx_dir = index_root / mag_bin
        if not idx_dir.exists():
            tqdm.write(f"[warn] missing index dir for {mag_bin}: {idx_dir}")
            continue
        csv_paths = sorted(idx_dir.glob("index*_masked.csv"))
        if not csv_paths:
            tqdm.write(f"[warn] no index CSVs found in {idx_dir}")
            continue

        for csv_path in tqdm(
            csv_paths,
            desc=f"{mag_bin} index CSVs",
            leave=False,
            disable=not show_progress,
        ):
            match = IDX_PATTERN.search(csv_path.name)
            if not match:
                tqdm.write(f"[warn] skipping unrecognized file name: {csv_path}")
                continue
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
            except FileNotFoundError:
                tqdm.write(f"[warn] index CSV not found: {csv_path}")
                continue
            except pd.errors.EmptyDataError:
                continue

            for source_id in ids:
                dat_path = lc_dir / f"{source_id}.dat"
                yield {
                    "source_id": source_id,
                    "mag_bin": mag_bin,
                    "index_num": idx_num,
                    "index_csv": str(csv_path),
                    "lc_dir": str(lc_dir),
                    "lc_dir_exists": lc_dir.exists(),
                    "dat_path": str(dat_path),
                    "dat_exists": dat_path.exists(),
                }


def build_manifest_dataframe(
    index_root: Path,
    lc_root: Path,
    *,
    mag_bins: Sequence[str],
    id_column: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    seen: dict[str, dict[str, object]] = {}
    duplicates = 0
    for record in _iter_source_records(
        index_root,
        lc_root,
        mag_bins,
        id_column=id_column,
        show_progress=show_progress,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manifest that maps ASAS-SN IDs to their index CSV and light-curve directory."
        )
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path(lc_dir_masked),
        help="Root directory that contains <mag_bin>/index*_masked.csv files.",
    )
    parser.add_argument(
        "--lc-root",
        type=Path,
        default=Path("/data/poohbah/1/assassin/rowan.90/lcsv2"),
        help="Root directory that contains <mag_bin>/lc*_cal/ light-curve folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./lc_manifest.csv"),
        help="Output file path (.csv or .parquet). Default: %(default)s",
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
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
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
    df = build_manifest_dataframe(
        index_root=args.index_root.expanduser(),
        lc_root=args.lc_root.expanduser(),
        mag_bins=mag_bins,
        id_column=args.id_column,
        show_progress=not args.no_progress,
    )

    out_path = args.out.expanduser()
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        raise SystemExit(f"Unsupported output format for {out_path}; use .csv or .parquet")

    print(f"Wrote {len(df):,} entries to {out_path}")


if __name__ == "__main__":
    main()
