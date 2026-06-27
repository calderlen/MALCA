#!/usr/bin/env python3
"""Export candidates from a run after removing IDs from an existing candidate CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from malca.config import DEFAULT_OUTPUT_DIR
from malca.products.feature_layers import with_feature_columns
from malca.io.table_io import read_feature_table


DEFAULT_COLUMNS = ("lc_path", "asas_sn_id", "mag_bin", "ra", "dec")
DEFAULT_KEY_COLUMNS = ("asas_sn_id", "candidate_id", "source_id", "gaia_id", "id")
MAG_BIN_RE = re.compile(r"/(\d+(?:\.\d+)?_\d+(?:\.\d+)?)(?:/|$)")
INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix == ".parquet" or path.is_dir():
        return read_feature_table(path)
    raise ValueError(f"Unsupported input type: {path}")


def _choose_key(df: pd.DataFrame, key: str | None) -> str:
    if key:
        if key not in df.columns:
            raise ValueError(f"Missing key column {key!r}; columns are: {', '.join(map(str, df.columns))}")
        return key
    for column in DEFAULT_KEY_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not auto-detect a candidate ID column. "
        f"Expected one of: {', '.join(DEFAULT_KEY_COLUMNS)}. Pass --key."
    )


def _normalize_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    match = INTEGER_FLOAT_RE.match(text)
    if match:
        text = match.group(1)
    return text.casefold()


def _id_from_path(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).stem


def _mag_bin_from_path(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    match = MAG_BIN_RE.search(text)
    return match.group(1) if match else ""


def _ensure_asas_sn_id(df: pd.DataFrame, key: str) -> pd.DataFrame:
    out = df.copy()
    if "asas_sn_id" not in out.columns:
        if key in out.columns:
            out["asas_sn_id"] = out[key].astype(str)
        elif "lc_path" in out.columns:
            out["asas_sn_id"] = out["lc_path"].map(_id_from_path)
    if "asas_sn_id" in out.columns:
        out["asas_sn_id"] = out["asas_sn_id"].astype(str).str.strip()
    return out


def _ensure_mag_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mag_bin" not in out.columns:
        if "lc_path" in out.columns:
            out["mag_bin"] = out["lc_path"].map(_mag_bin_from_path)
        else:
            out["mag_bin"] = ""
    return out


def _merge_coordinates_from_characterized(
    candidates: pd.DataFrame,
    characterized_path: Path | None,
    *,
    key: str,
) -> pd.DataFrame:
    if characterized_path is None or not characterized_path.exists():
        return candidates
    candidates = with_feature_columns(candidates, ["ra", "dec"])
    if {"ra", "dec"}.issubset(candidates.columns) and candidates[["ra", "dec"]].notna().any().any():
        return candidates
    candidates = candidates.drop(columns=[c for c in ("ra", "dec") if c in candidates.columns])

    characterized = with_feature_columns(_read_table(characterized_path), ["ra", "dec"])
    char_key = key if key in characterized.columns else _choose_key(characterized, None)
    coord_cols = [c for c in ("ra", "dec") if c in characterized.columns]
    if not coord_cols:
        return candidates

    coords = characterized[[char_key, *coord_cols]].drop_duplicates(subset=[char_key]).copy()
    coords = coords.rename(columns={char_key: key})
    return candidates.merge(coords, on=key, how="left")


def extract_new_candidates(
    run_candidates_path: Path,
    existing_candidates_path: Path,
    output_csv: Path,
    *,
    key: str | None = None,
    characterized_path: Path | None = None,
    all_columns: bool = False,
    force: bool = False,
) -> dict[str, object]:
    if output_csv.exists() and not force:
        raise FileExistsError(f"Output exists, use --force to overwrite: {output_csv}")

    run_df = _read_table(run_candidates_path)
    existing_df = _read_table(existing_candidates_path)
    run_key = _choose_key(run_df, key)
    existing_key = _choose_key(existing_df, key if key in existing_df.columns else None)

    run_df = _ensure_asas_sn_id(run_df, run_key)
    run_key = "asas_sn_id" if "asas_sn_id" in run_df.columns else run_key
    existing_ids = {
        normalized
        for normalized in existing_df[existing_key].map(_normalize_id)
        if normalized
    }

    run_ids = run_df[run_key].map(_normalize_id)
    keep_mask = ~run_ids.isin(existing_ids)
    new_df = run_df.loc[keep_mask].copy()
    new_df = _ensure_asas_sn_id(new_df, run_key)
    new_df = _ensure_mag_bin(new_df)
    new_df = _merge_coordinates_from_characterized(new_df, characterized_path, key=run_key)

    if not all_columns:
        for column in DEFAULT_COLUMNS:
            if column not in new_df.columns:
                new_df[column] = ""
        new_df = new_df.loc[:, list(DEFAULT_COLUMNS)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False)

    return {
        "run_rows": len(run_df),
        "existing_rows": len(existing_df),
        "existing_unique_ids": len(existing_ids),
        "removed_rows": int((~keep_mask).sum()),
        "output_rows": len(new_df),
        "run_key": run_key,
        "existing_key": existing_key,
        "output_csv": output_csv,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export new run candidates after subtracting already-known candidate IDs.",
    )
    parser.add_argument(
        "--run-candidates",
        type=Path,
        default=Path("output/runs/may13/results/lc_events_filtered_all.parquet"),
        help="Run candidate table, CSV/Parquet/directory. Default: output/runs/may13/results/lc_events_filtered_all.parquet",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=Path("output/candidates_12_15_combined.csv"),
        help="Existing candidate CSV/table to subtract. Default: output/candidates_12_15_combined.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "new_candidates_12_15.csv",
        help=f"Output CSV. Default: {DEFAULT_OUTPUT_DIR / 'new_candidates_12_15.csv'}",
    )
    parser.add_argument("--key", default=None, help="Candidate ID column. Defaults to auto-detect.")
    parser.add_argument(
        "--characterized",
        type=Path,
        default=Path("output/runs/may13/results/lc_events_characterized.parquet"),
        help="Optional characterized table used to fill ra_deg/dec_deg when available.",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Write all run columns instead of compact path/asas_sn_id/mag_bin/ra_deg/dec_deg.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output CSV if it exists.")
    args = parser.parse_args(argv)

    characterized = args.characterized.expanduser() if args.characterized else None
    try:
        summary = extract_new_candidates(
            args.run_candidates.expanduser(),
            args.existing.expanduser(),
            args.output.expanduser(),
            key=args.key,
            characterized_path=characterized,
            all_columns=bool(args.all_columns),
            force=bool(args.force),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Run candidates: {args.run_candidates}")
    print(f"Existing candidates: {args.existing}")
    print(f"Output CSV: {summary['output_csv']}")
    print(f"Key columns: {summary['run_key']} vs {summary['existing_key']}")
    print(f"Rows in run table: {summary['run_rows']}")
    print(f"Rows in existing table: {summary['existing_rows']}")
    print(f"Unique existing IDs: {summary['existing_unique_ids']}")
    print(f"Rows removed: {summary['removed_rows']}")
    print(f"Rows written: {summary['output_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
