from __future__ import annotations

from collections.abc import Callable

import pandas as pd


TRUTHY_FAILED_ANY_VALUES = {"1", "true", "t", "yes", "y"}


def passing_candidates_mask(df: pd.DataFrame, *, failed_col: str = "failed_any") -> pd.Series:
    """Return True for rows that have not failed candidate filtering."""
    if failed_col not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)

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
    if "failed_any" not in df.columns:
        return df.copy()

    before = len(df)
    out = select_passing_candidates(df)
    if printer is not None:
        printer(f"Using passers only (failed_any=False): {len(out)}/{before} {label}")
    return out
