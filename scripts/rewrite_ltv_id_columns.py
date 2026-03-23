#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION


LEGACY_ID_COLUMN = "ASAS-SN ID"
STANDARD_ID_COLUMN = "asas_sn_id"
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq"}


def _candidate_files(root: Path) -> list[Path]:
    root = root.expanduser()
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected file or directory: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _normalize_id_column(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if LEGACY_ID_COLUMN not in df.columns:
        return df, False

    out = df.copy()
    if STANDARD_ID_COLUMN in out.columns:
        legacy = out[LEGACY_ID_COLUMN].astype("string").fillna("")
        standard = out[STANDARD_ID_COLUMN].astype("string").fillna("")
        if not legacy.equals(standard):
            raise ValueError(
                f"Both '{LEGACY_ID_COLUMN}' and '{STANDARD_ID_COLUMN}' exist but differ"
            )
        out = out.drop(columns=[LEGACY_ID_COLUMN])
    else:
        out = out.rename(columns={LEGACY_ID_COLUMN: STANDARD_ID_COLUMN})
    return out, True


def _rewrite_file(path: Path, *, write: bool) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    out, changed = _normalize_id_column(df)
    if not changed or (not write):
        return changed

    if suffix == ".csv":
        tmp_path = path.with_name(f"{path.name}.tmp.csv")
        out.to_csv(tmp_path, index=False)
    else:
        tmp_path = path.with_name(f"{path.name}.tmp.parquet")
        out.to_parquet(tmp_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    tmp_path.replace(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite legacy LTV output files from 'ASAS-SN ID' to 'asas_sn_id'.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="LTV file or directory to scan recursively",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply changes in place. Without this flag, run as a dry run.",
    )
    args = parser.parse_args()

    files = _candidate_files(args.path)
    changed_paths: list[Path] = []
    for path in files:
        changed = _rewrite_file(path, write=args.write)
        if changed:
            changed_paths.append(path)
            action = "rewrote" if args.write else "would rewrite"
            print(f"{action}: {path}")

    print(
        f"Scanned {len(files)} file(s); "
        f"{'rewrote' if args.write else 'would rewrite'} {len(changed_paths)} file(s)."
    )


if __name__ == "__main__":
    main()
