#!/usr/bin/env python3
"""Copy microlensing fit PDFs for rows not visually flagged bad/probably_bad.

Defaults:
  - results Parquet: latest output/microlensing/microlensing_results_*.parquet
  - source PDFs: output/microlensing/fit_pdfs
  - destination: output/microlensing/fit_pdfs_not_flagged
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from malca.config import DEFAULT_OUTPUT_DIR
from malca.io.table_io import read_feature_table


def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Could not find repo root (missing pyproject.toml).")


def _default_results_parquet(output_root: Path) -> Path:
    files = sorted(output_root.glob("microlensing_results_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No microlensing_results_*.parquet under {output_root}")
    return files[-1]


def _norm_id(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


def _pdf_id_from_name(pdf_path: Path) -> str:
    # Name format from scripts/microlensing.py: "{chi}_{tE}_{id}.pdf"
    stem = pdf_path.stem
    if "_" not in stem:
        return ""
    return stem.rsplit("_", 1)[-1].strip()


def main(argv: list[str] | None = None) -> int:
    repo_root = _find_repo_root(Path.cwd().resolve())
    out_root = (repo_root / DEFAULT_OUTPUT_DIR / "microlensing").resolve()

    ap = argparse.ArgumentParser(
        description="Copy fit PDFs for candidates not flagged bad/probably_bad by visual inspection.",
    )
    ap.add_argument(
        "--results-parquet",
        type=Path,
        default=None,
        help="Path to microlensing results Parquet (default: newest output/microlensing/microlensing_results_*.parquet).",
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=out_root / "fit_pdfs",
        help="Directory containing source fit PDFs.",
    )
    ap.add_argument(
        "--dest-dir",
        type=Path,
        default=out_root / "fit_pdfs_not_flagged",
        help="Destination directory for copied PDFs.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying files.")
    args = ap.parse_args(argv)

    results_parquet = (
        args.results_parquet.expanduser().resolve()
        if args.results_parquet is not None
        else _default_results_parquet(out_root)
    )
    source_dir = args.source_dir.expanduser().resolve()
    dest_dir = args.dest_dir.expanduser().resolve()

    if not results_parquet.is_file():
        raise FileNotFoundError(f"Results Parquet not found: {results_parquet}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source PDF directory not found: {source_dir}")

    df = read_feature_table(results_parquet)
    if df.empty:
        print(f"No rows in {results_parquet}")
        return 0

    if "visual_inspection_subjective_flag" not in df.columns:
        print(
            "WARNING: visual_inspection_subjective_flag column not found; "
            "treating all rows as not flagged.",
            file=sys.stderr,
        )
        flag = pd.Series([""] * len(df), index=df.index)
    else:
        flag = df["visual_inspection_subjective_flag"].fillna("").astype(str).str.strip().str.lower()

    keep_mask = ~flag.isin({"bad", "probably_bad"})
    df_keep = df.loc[keep_mask].copy()

    if "asas_sn_id" in df_keep.columns:
        ids_keep = {_norm_id(v) for v in df_keep["asas_sn_id"]}
    else:
        ids_keep = set()
    if "candidate_id" in df_keep.columns:
        ids_keep |= {_norm_id(v) for v in df_keep["candidate_id"]}
    ids_keep.discard("")

    pdfs = list(source_dir.glob("*.pdf"))
    by_id: dict[str, list[Path]] = {}
    for p in pdfs:
        pid = _norm_id(_pdf_id_from_name(p))
        if not pid:
            continue
        by_id.setdefault(pid, []).append(p)

    to_copy: list[Path] = []
    missing: list[str] = []
    for cid in sorted(ids_keep):
        matches = by_id.get(cid, [])
        if matches:
            to_copy.extend(matches)
        else:
            missing.append(cid)

    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in to_copy:
            shutil.copy2(src, dest_dir / src.name)

    print(f"Results Parquet: {results_parquet}")
    print(f"Source PDFs: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Rows total: {len(df)}")
    print(f"Rows kept (not bad/probably_bad): {len(df_keep)}")
    print(f"Unique kept IDs: {len(ids_keep)}")
    print(f"PDFs matched: {len(to_copy)}")
    print(f"IDs with no PDF found: {len(missing)}")
    if missing:
        print("First 15 missing IDs:", ", ".join(missing[:15]))
    if args.dry_run:
        print("Dry run: no files copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
