#!/usr/bin/env python3
"""Write candidates from one CSV after removing IDs found in another CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


DEFAULT_KEY_COLUMNS = (
    "candidate_id",
    "asas_sn_id",
    "asassn_id",
    "source_id",
    "gaia_id",
    "object_id",
    "id",
    "name",
)

INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _choose_key(left: pd.DataFrame, right: pd.DataFrame, requested: str | None) -> tuple[str, str]:
    if requested is not None:
        return requested, requested

    common = set(left.columns) & set(right.columns)
    for column in DEFAULT_KEY_COLUMNS:
        if column in common:
            return column, column

    if len(common) == 1:
        column = next(iter(common))
        return column, column

    if not common:
        raise ValueError(
            "No shared columns found. Pass --left-key and --right-key to choose the candidate ID columns."
        )

    common_list = ", ".join(sorted(common))
    defaults = ", ".join(DEFAULT_KEY_COLUMNS)
    raise ValueError(
        "Could not auto-detect candidate ID column. "
        f"Shared columns: {common_list}. Expected one of: {defaults}. "
        "Pass --key or --left-key/--right-key."
    )


def _normalize_id(value: object, *, case_sensitive: bool) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""

    match = INTEGER_FLOAT_RE.match(text)
    if match:
        text = match.group(1)

    if not case_sensitive:
        text = text.casefold()
    return text


def subtract_candidates(
    left_csv: Path,
    right_csv: Path,
    output_csv: Path,
    *,
    key: str | None = None,
    left_key: str | None = None,
    right_key: str | None = None,
    case_sensitive: bool = False,
    force: bool = False,
) -> dict[str, object]:
    if output_csv.exists() and not force:
        raise FileExistsError(f"Output exists, use --force to overwrite: {output_csv}")

    left = _read_csv(left_csv)
    right = _read_csv(right_csv)

    if key and (left_key or right_key):
        raise ValueError("Use either --key or --left-key/--right-key, not both.")
    if (left_key is None) != (right_key is None):
        raise ValueError("--left-key and --right-key must be provided together.")
    if left_key is None or right_key is None:
        left_key, right_key = _choose_key(left, right, key)

    missing = [f"{left_csv}: {left_key}" if left_key not in left.columns else ""]
    missing.append(f"{right_csv}: {right_key}" if right_key not in right.columns else "")
    missing = [item for item in missing if item]
    if missing:
        raise ValueError("Missing key column(s): " + "; ".join(missing))

    existing_ids = {
        normalized
        for normalized in right[right_key].map(
            lambda value: _normalize_id(value, case_sensitive=case_sensitive)
        )
        if normalized
    }

    left_ids = left[left_key].map(lambda value: _normalize_id(value, case_sensitive=case_sensitive))
    keep_mask = ~left_ids.isin(existing_ids)
    output = left.loc[keep_mask].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "existing_ids": len(existing_ids),
        "removed_rows": int((~keep_mask).sum()),
        "output_rows": len(output),
        "left_key": left_key,
        "right_key": right_key,
        "output_csv": output_csv,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CSV from candidate_list.csv minus candidates already present "
            "in existing_candidates.csv."
        )
    )
    parser.add_argument("candidate_csv", type=Path, help="CSV to filter.")
    parser.add_argument("existing_csv", type=Path, help="CSV containing candidates to remove.")
    parser.add_argument("output_csv", type=Path, help="Output CSV for candidates not in existing_csv.")
    parser.add_argument(
        "--key",
        default=None,
        help="Candidate ID column name used in both CSVs. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--left-key",
        default=None,
        help="Candidate ID column in candidate_csv when column names differ.",
    )
    parser.add_argument(
        "--right-key",
        default=None,
        help="Candidate ID column in existing_csv when column names differ.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match candidate IDs with case-sensitive comparison. Default is case-insensitive.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output_csv if it already exists.")
    args = parser.parse_args(argv)

    try:
        summary = subtract_candidates(
            args.candidate_csv.expanduser(),
            args.existing_csv.expanduser(),
            args.output_csv.expanduser(),
            key=args.key,
            left_key=args.left_key,
            right_key=args.right_key,
            case_sensitive=bool(args.case_sensitive),
            force=bool(args.force),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Candidate CSV: {args.candidate_csv}")
    print(f"Existing CSV: {args.existing_csv}")
    print(f"Output CSV: {summary['output_csv']}")
    print(f"Key columns: {summary['left_key']} vs {summary['right_key']}")
    print(f"Rows in candidate CSV: {summary['left_rows']}")
    print(f"Rows in existing CSV: {summary['right_rows']}")
    print(f"Unique existing IDs: {summary['existing_ids']}")
    print(f"Rows removed: {summary['removed_rows']}")
    print(f"Rows written: {summary['output_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
