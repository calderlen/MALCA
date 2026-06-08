from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


TRUTHY_FAILED_ANY_VALUES = {"1", "true", "t", "yes", "y"}


def passing_candidates_mask(df: pd.DataFrame, *, failed_col: str = "failed_any") -> pd.Series:
    """Return True for rows that have not failed candidate filtering."""
    if failed_col not in df.columns:
        if failed_col == "failed_any" and "derived_stats" in df.columns:
            from malca.feature_layers import feature_value_series

            failed = feature_value_series(df, "derived_stats.failed_any", default=False)
        else:
            return pd.Series(True, index=df.index, dtype=bool)
    else:
        failed = df[failed_col]

    if pd.api.types.is_bool_dtype(failed):
        return ~failed.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(failed):
        return failed.fillna(0).astype(float) == 0.0

    lowered = failed.fillna("").astype(str).str.strip().str.lower()
    return ~lowered.isin(TRUTHY_FAILED_ANY_VALUES)


def select_passing_candidates(df: pd.DataFrame, *, failed_col: str = "failed_any") -> pd.DataFrame:
    """Return a copy of candidates with failed_any absent or false-like."""
    return df.loc[passing_candidates_mask(df, failed_col=failed_col)].copy()


def select_passing_candidates_if_present(
    df: pd.DataFrame,
    *,
    label: str = "candidates",
    printer: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Filter to passers when failed_any exists, logging the reduction if requested."""
    if "failed_any" not in df.columns and "derived_stats" not in df.columns:
        return df.copy()

    before = len(df)
    out = select_passing_candidates(df)
    if printer is not None:
        printer(f"Using passers only (failed_any=False): {len(out)}/{before} {label}")
    return out


def _normalize_candidate_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        if text.lower() in {"nan", "none", "<na>"}:
            return ""
    except Exception:
        pass
    return text


def _candidate_ids_from_source_columns(
    df: pd.DataFrame,
    *,
    source_cols: tuple[str, ...],
) -> pd.Series:
    ids = pd.Series("", index=df.index, dtype="object")
    for column in source_cols:
        if column not in df.columns:
            continue

        missing = ids.eq("")
        if not bool(missing.any()):
            break

        if column in {"path", "dat_path", "lc_path", "local_lightcurve_path"}:
            values = df.loc[missing, column].map(
                lambda value: Path(str(value)).stem if _normalize_candidate_token(value) else ""
            )
        else:
            values = df.loc[missing, column]
        ids.loc[missing] = values.map(_normalize_candidate_token)
    return ids


def ensure_candidate_id(
    df: pd.DataFrame,
    prefix: str | None = None,
    source_cols: tuple[str, ...] = ("candidate_id", "asas_sn_id", "source_id", "path"),
) -> pd.DataFrame:
    """Return a copy with a populated candidate_id column.

    When ``prefix`` is supplied, unprefixed IDs are normalized to
    ``<prefix>_<id>`` while already-prefixed values are left unchanged.
    """
    out = df.copy()
    ids = _candidate_ids_from_source_columns(out, source_cols=source_cols)
    if prefix:
        clean_prefix = str(prefix).strip().rstrip("_")
        prefix_text = f"{clean_prefix}_"
        ids = ids.map(lambda value: value if not value or value.startswith(prefix_text) else f"{prefix_text}{value}")
    out["candidate_id"] = ids
    return out


def merge_candidate_columns(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    value_cols: list[str] | tuple[str, ...],
    key_col: str = "candidate_id",
) -> pd.DataFrame:
    """Merge selected candidate columns from ``extra`` into ``base`` by candidate ID."""
    if extra.empty:
        return base.copy()
    if key_col not in base.columns or key_col not in extra.columns:
        return base.copy()

    cols = [col for col in value_cols if col in extra.columns]
    if not cols:
        return base.copy()

    left = base.copy()
    right = extra[[key_col, *cols]].copy()
    left[key_col] = left[key_col].astype(str)
    right[key_col] = right[key_col].astype(str)
    right = right.drop_duplicates(subset=[key_col], keep="last")

    merged = left.merge(right, on=key_col, how="left", suffixes=("", "_candidate_new"))
    for col in cols:
        new_col = f"{col}_candidate_new"
        if new_col not in merged.columns:
            continue
        if col in left.columns:
            merged[col] = merged[new_col].where(merged[new_col].notna(), merged[col])
        else:
            merged[col] = merged[new_col]
        merged = merged.drop(columns=[new_col])
    return merged
