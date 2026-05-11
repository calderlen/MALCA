"""Per-field cotrending scaffolding for ASAS-SN light curves.

This module intentionally lives outside the discovery pipeline.  It provides
shared table loading, field-key validation, and grouping helpers for future
CBV/cotrending experiments without changing event detection behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


FIELD_KEY_COLUMN = "asassn_field_key"
PATH_COLUMN_CANDIDATES: tuple[str, ...] = ("dat_path", "path", "lc_path")


def load_candidate_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV/Parquet candidate or manifest table."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported input file type. Use CSV or Parquet.")


def resolve_path_column(
    df: pd.DataFrame,
    *,
    preferred: str | None = None,
    candidates: Iterable[str] = PATH_COLUMN_CANDIDATES,
) -> str:
    """Return the light-curve path column to use for cotrending inputs."""
    ordered = [preferred] if preferred else []
    ordered.extend(candidates)
    for col in ordered:
        if col and col in df.columns:
            return str(col)
    tried = ", ".join(str(col) for col in ordered if col)
    raise ValueError(f"Could not find a light-curve path column. Tried: {tried}")


def require_field_key(df: pd.DataFrame, *, field_col: str = FIELD_KEY_COLUMN) -> pd.DataFrame:
    """Return rows with a non-empty field key, or raise if the column is absent."""
    if field_col not in df.columns:
        raise ValueError(
            f"Input table is missing {field_col!r}; run the field-key propagation pipeline first."
        )
    out = df.copy()
    field_values = out[field_col].astype("string").fillna("").str.strip()
    return out.loc[field_values != ""].assign(**{field_col: field_values[field_values != ""]})


def group_lightcurves_by_field(
    df: pd.DataFrame,
    *,
    field_col: str = FIELD_KEY_COLUMN,
    path_col: str | None = None,
) -> dict[str, list[str]]:
    """Group light-curve paths by propagated ASAS-SN field key."""
    usable = require_field_key(df, field_col=field_col)
    resolved_path_col = resolve_path_column(usable, preferred=path_col)
    grouped: dict[str, list[str]] = {}
    for field_key, group in usable.groupby(field_col, observed=True):
        paths = (
            group[resolved_path_col]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        grouped[str(field_key)] = [path for path in paths if path]
    return grouped


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect ASAS-SN field-key groups for cotrending work")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet table with field keys")
    parser.add_argument("--field-col", default=FIELD_KEY_COLUMN, help="Field key column")
    parser.add_argument("--path-col", default=None, help="Optional LC path column override")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON summary output")
    args = parser.parse_args()

    df = load_candidate_table(args.input)
    groups = group_lightcurves_by_field(df, field_col=args.field_col, path_col=args.path_col)
    summary = {
        "n_fields": len(groups),
        "n_lightcurves": int(sum(len(paths) for paths in groups.values())),
        "fields": {field: len(paths) for field, paths in sorted(groups.items())},
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
