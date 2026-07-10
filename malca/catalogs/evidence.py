"""Normalize raw catalog context into canonical review evidence fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from malca.products.feature_layers import feature_value_series, is_layer_first_frame, layer_path_for_column
from malca.review.filter_schema import is_known_variable_type_value


_MISSING_TEXT = {"", "nan", "none", "null", "<na>"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TEXT
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.map(_is_missing).astype(bool)


def _series_for_column(df: pd.DataFrame, column: str) -> pd.Series:
    base = pd.Series(pd.NA, index=df.index, dtype="object")
    if column in df.columns:
        base = df[column]
    if is_layer_first_frame(df):
        path = layer_path_for_column(column)
        if path is not None:
            layered = feature_value_series(df, path)
            missing = _missing_mask(base)
            if missing.any():
                base = base.copy()
                base.loc[missing] = layered.loc[missing]
    return base


def _ensure_flat_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        df[column] = _series_for_column(df, column)


def _fill_blank(df: pd.DataFrame, column: str, values: pd.Series) -> None:
    _ensure_flat_column(df, column)
    aligned = values.reindex(df.index)
    fill = _missing_mask(df[column]) & ~_missing_mask(aligned)
    if fill.any():
        df.loc[fill, column] = aligned.loc[fill]


def _first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    lowered = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        found = lowered.get(name.lower())
        if found is not None:
            return found
    return None


def vsx_evidence_from_neighbors(
    neighbors_long: pd.DataFrame | None,
    *,
    max_sep_arcsec: float = 3.0,
) -> pd.DataFrame:
    """Return closest definite VSX evidence derived from neighbor rows."""
    if neighbors_long is None or neighbors_long.empty:
        return pd.DataFrame(columns=["candidate_id", "vsx_class", "vsx_period", "vsx_sep_arcsec"])

    if "candidate_id" not in neighbors_long.columns:
        return pd.DataFrame(columns=["candidate_id", "vsx_class", "vsx_period", "vsx_sep_arcsec"])

    catalog_col = _first_existing_column(neighbors_long, ("catalog", "source_catalog"))
    if catalog_col is not None:
        vsx_mask = neighbors_long[catalog_col].astype(str).str.contains("vsx", case=False, na=False)
        rows = neighbors_long.loc[vsx_mask].copy()
    else:
        rows = neighbors_long.copy()
    if rows.empty:
        return pd.DataFrame(columns=["candidate_id", "vsx_class", "vsx_period", "vsx_sep_arcsec"])

    sep_col = _first_existing_column(rows, ("sep_arcsec", "vsx_sep_arcsec", "separation_arcsec"))
    type_col = _first_existing_column(rows, ("Type", "type", "vsx_type", "vsx_class", "class"))
    if sep_col is None or type_col is None:
        return pd.DataFrame(columns=["candidate_id", "vsx_class", "vsx_period", "vsx_sep_arcsec"])

    period_col = _first_existing_column(rows, ("Period", "period", "vsx_period"))
    out = pd.DataFrame(
        {
            "candidate_id": rows["candidate_id"].astype(str),
            "vsx_class": rows[type_col].map(lambda value: "" if _is_missing(value) else str(value).strip()),
            "vsx_sep_arcsec": pd.to_numeric(rows[sep_col], errors="coerce"),
        },
        index=rows.index,
    )
    out["vsx_period"] = (
        pd.to_numeric(rows[period_col], errors="coerce")
        if period_col is not None
        else pd.Series(np.nan, index=rows.index)
    )

    definite = out["vsx_class"].map(lambda value: is_known_variable_type_value("vsx_class", value)).astype(bool)
    close = out["vsx_sep_arcsec"].le(float(max_sep_arcsec))
    out = out.loc[definite & close].dropna(subset=["vsx_sep_arcsec"]).copy()
    if out.empty:
        return pd.DataFrame(columns=["candidate_id", "vsx_class", "vsx_period", "vsx_sep_arcsec"])

    out = out.sort_values(["candidate_id", "vsx_sep_arcsec"], kind="mergesort")
    return out.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)


def normalize_catalog_evidence_record(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize one flat payload/record into canonical catalog evidence fields."""
    if not isinstance(record, Mapping):
        return {}
    out = dict(record)
    if _is_missing(out.get("vsx_class")) and not _is_missing(out.get("vsx_type")):
        out["vsx_class"] = out.get("vsx_type")
    return out


def normalize_catalog_evidence(
    df: pd.DataFrame,
    *,
    neighbors_long: pd.DataFrame | None = None,
    vsx_max_sep_arcsec: float = 3.0,
) -> pd.DataFrame:
    """Fill canonical catalog-evidence fields from known synonyms/context.

    Existing canonical values always win. Neighbor-derived VSX classes are
    promoted only for close, definite known-variable VSX types.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    out = df.copy()
    if out.empty:
        for column in ("vsx_class", "vsx_period", "vsx_sep_arcsec"):
            _ensure_flat_column(out, column)
        return out

    _fill_blank(out, "vsx_class", _series_for_column(out, "vsx_type"))

    evidence = vsx_evidence_from_neighbors(neighbors_long, max_sep_arcsec=vsx_max_sep_arcsec)
    if evidence.empty or "candidate_id" not in out.columns:
        return out

    merged = out.merge(
        evidence,
        on="candidate_id",
        how="left",
        suffixes=("", "_neighbor_vsx"),
    )
    for column in ("vsx_class", "vsx_period", "vsx_sep_arcsec"):
        neighbor_col = f"{column}_neighbor_vsx"
        if neighbor_col in merged.columns:
            _fill_blank(merged, column, merged[neighbor_col])
    drop_cols = [col for col in merged.columns if col.endswith("_neighbor_vsx")]
    return merged.drop(columns=drop_cols)
