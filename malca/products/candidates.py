from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


TRUTHY_FAILED_ANY_VALUES = {"1", "true", "t", "yes", "y"}
FALSEY_FAILED_ANY_VALUES = {"0", "false", "f", "no", "n"}


class CandidateSelectionError(ValueError):
    """Raised when a science/pass selection cannot be made unambiguously."""


def coerce_strict_bool_series(series: pd.Series, *, field_name: str) -> pd.Series:
    """Return Boolean failure flags, rejecting null or ambiguous values.

    Candidate selection is a science boundary.  Treating an absent, null, or
    misspelled failure value as ``False`` silently promotes an unevaluated row
    into the passing sample, so coercion here is intentionally strict.
    """
    if pd.api.types.is_bool_dtype(series):
        invalid = series.isna()
        flags = series.astype("boolean")
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])
        flags = numeric.eq(1).astype("boolean")
    else:
        lowered = series.astype("string").str.strip().str.lower()
        valid = lowered.isin(TRUTHY_FAILED_ANY_VALUES | FALSEY_FAILED_ANY_VALUES)
        invalid = lowered.isna() | ~valid
        flags = lowered.isin(TRUTHY_FAILED_ANY_VALUES).astype("boolean")

    if bool(invalid.any()):
        examples = [str(value) for value in series.loc[invalid].head(5).tolist()]
        raise CandidateSelectionError(
            f"Cannot select passing candidates: {field_name!r} contains "
            f"{int(invalid.sum())} null/invalid value(s); examples={examples}"
        )
    return flags.astype(bool)


def passing_candidates_mask(
    df: pd.DataFrame,
    *,
    failed_col: str = "failed_any",
    require_failed_col: bool = True,
) -> pd.Series:
    """Return True only for rows explicitly evaluated as passing.

    ``require_failed_col=False`` is reserved for callers that are deliberately
    operating before filtering.  Science, review, and publication consumers
    should use the fail-closed default.
    """
    if failed_col not in df.columns:
        if failed_col == "failed_any" and "derived_stats" in df.columns:
            from malca.products.feature_layers import feature_value_series

            failed = feature_value_series(df, "derived_stats.failed_any", default=pd.NA)
        else:
            if require_failed_col:
                raise CandidateSelectionError(
                    f"Cannot select passing candidates: required {failed_col!r} field is missing"
                )
            return pd.Series(True, index=df.index, dtype=bool)
    else:
        failed = df[failed_col]

    return ~coerce_strict_bool_series(failed, field_name=failed_col)


def select_passing_candidates(
    df: pd.DataFrame,
    *,
    failed_col: str = "failed_any",
    require_failed_col: bool = True,
) -> pd.DataFrame:
    """Return candidates explicitly marked as passing."""
    return df.loc[
        passing_candidates_mask(
            df,
            failed_col=failed_col,
            require_failed_col=require_failed_col,
        )
    ].copy()


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
    out = select_passing_candidates(df, require_failed_col=True)
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


def validate_candidate_ids(
    df: pd.DataFrame,
    *,
    key_col: str = "candidate_id",
    require_unique: bool = True,
) -> pd.Series:
    """Validate and return canonical string candidate identifiers."""
    if key_col not in df.columns:
        raise ValueError(f"Required candidate identity column is missing: {key_col}")
    ids = df[key_col].map(_normalize_candidate_token).astype("string")
    missing = ids.isna() | ids.eq("")
    if bool(missing.any()):
        raise ValueError(f"{key_col} contains {int(missing.sum())} blank/null identifier(s)")
    if require_unique:
        duplicated = ids.duplicated(keep=False)
        if bool(duplicated.any()):
            examples = ids.loc[duplicated].drop_duplicates().head(5).tolist()
            raise ValueError(
                f"{key_col} contains {int(duplicated.sum())} duplicate row(s); examples={examples}"
            )
    return ids.astype(str)


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
    left[key_col] = validate_candidate_ids(left, key_col=key_col, require_unique=True)
    right[key_col] = validate_candidate_ids(right, key_col=key_col, require_unique=True)

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
