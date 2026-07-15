"""Normalize raw catalog context into canonical review evidence fields."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

import numpy as np
import pandas as pd

from malca.config import NEIGHBOR_RADIUS_ARCSEC
from malca.products.feature_layers import feature_value_series, is_layer_first_frame, layer_path_for_column
from malca.review.filter_schema import is_dipper_contaminant_type_value, is_known_variable_type_value


_MISSING_TEXT = {"", "nan", "none", "null", "<na>"}
DEFAULT_CATALOG_NEIGHBOR_QUERY_RADIUS_ARCSEC = 30.0
DEFAULT_REVIEW_VETTING_RADIUS_ARCSEC = 15.0
MAX_REVIEW_VETTING_RADIUS_ARCSEC = 30.0
CATALOG_NEIGHBOR_OUTPUT_SUBDIR = "vetting_catalog_neighbors"
CATALOG_NEIGHBOR_FILENAME = "catalog_neighbors.parquet"
CATALOG_NEIGHBOR_COLUMNS = (
    "candidate_id",
    "catalog",
    "object_id",
    "object_name",
    "class_value",
    "sep_arcsec",
    "period_days",
    "rank",
    "is_known_variable",
    "is_dipper_contaminant",
    "query_radius_arcsec",
    "raw_json",
)
CATALOG_NEIGHBOR_CLASS_COLUMNS = {
    "vsx": "vsx_class",
    "simbad": "simbad_otype",
    "asassn_variables": "asassn_var_type",
    "asassn": "asassn_var_type",
    "ztf_periodic_variables": "ztf_var_type",
    "ztf": "ztf_var_type",
    "tns": "tns_type",
    "microlensing_catalogs": "microlens_catalog",
    "microlens": "microlens_catalog",
}
_VSX_EVIDENCE_COLUMNS = ["candidate_id", "vsx_class", "vsx_period", "vsx_sep_arcsec"]
_NEARBY_VSX_DIPPER_COLUMNS = [
    "candidate_id",
    "nearby_vsx_dipper_contaminant",
    "nearby_vsx_dipper_class",
    "nearby_vsx_dipper_sep_arcsec",
    "nearby_vsx_dipper_period",
]


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


def _empty_columns(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _clean_neighbor_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _neighbor_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _neighbor_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out


def _neighbor_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
    return bool(value)


def _neighbor_raw_json(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value))


def _catalog_neighbor_class_column(catalog: Any, class_column: str | None = None) -> str:
    if class_column:
        return str(class_column)
    key = _clean_neighbor_text(catalog).lower()
    return CATALOG_NEIGHBOR_CLASS_COLUMNS.get(key, key)


def _catalog_neighbor_forces_known(catalog: Any, object_name: Any, class_value: Any) -> bool:
    key = _clean_neighbor_text(catalog).lower()
    has_match = bool(_clean_neighbor_text(object_name) or _clean_neighbor_text(class_value))
    return has_match and key in {"tns", "microlensing_catalogs", "microlens"}


def catalog_neighbor_record(
    *,
    candidate_id: Any,
    catalog: str,
    sep_arcsec: Any,
    object_id: Any = "",
    object_name: Any = "",
    class_value: Any = "",
    class_column: str | None = None,
    period_days: Any = None,
    rank: Any = None,
    is_known_variable: Any = None,
    is_dipper_contaminant: Any = None,
    query_radius_arcsec: Any = DEFAULT_CATALOG_NEIGHBOR_QUERY_RADIUS_ARCSEC,
    raw: Any = None,
) -> dict[str, Any]:
    """Return one normalized long-form catalog-neighbor evidence record."""
    class_value_text = _clean_neighbor_text(class_value)
    catalog_text = _clean_neighbor_text(catalog)
    class_col = _catalog_neighbor_class_column(catalog_text, class_column)

    known = _neighbor_bool(is_known_variable)
    if known is None:
        known = (
            _catalog_neighbor_forces_known(catalog_text, object_name, class_value_text)
            or is_known_variable_type_value(class_col, class_value_text)
        )
    dipper = _neighbor_bool(is_dipper_contaminant)
    if dipper is None:
        dipper = is_dipper_contaminant_type_value(class_col, class_value_text)

    return {
        "candidate_id": _clean_neighbor_text(candidate_id),
        "catalog": catalog_text,
        "object_id": _clean_neighbor_text(object_id),
        "object_name": _clean_neighbor_text(object_name),
        "class_value": class_value_text,
        "sep_arcsec": _neighbor_float(sep_arcsec),
        "period_days": _neighbor_float(period_days),
        "rank": _neighbor_int(rank),
        "is_known_variable": bool(known),
        "is_dipper_contaminant": bool(dipper),
        "query_radius_arcsec": _neighbor_float(query_radius_arcsec),
        "raw_json": _neighbor_raw_json(raw),
    }


def normalize_catalog_neighbor_frame(rows: pd.DataFrame | list[Mapping[str, Any]] | None) -> pd.DataFrame:
    """Normalize long-form catalog-neighbor rows to the review/Parquet schema."""
    if rows is None:
        return pd.DataFrame(columns=CATALOG_NEIGHBOR_COLUMNS)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=CATALOG_NEIGHBOR_COLUMNS)

    for column in CATALOG_NEIGHBOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    records = [
        catalog_neighbor_record(
            candidate_id=row.get("candidate_id"),
            catalog=row.get("catalog"),
            object_id=row.get("object_id"),
            object_name=row.get("object_name"),
            class_value=row.get("class_value"),
            sep_arcsec=row.get("sep_arcsec"),
            period_days=row.get("period_days"),
            rank=row.get("rank"),
            is_known_variable=row.get("is_known_variable"),
            is_dipper_contaminant=row.get("is_dipper_contaminant"),
            query_radius_arcsec=row.get("query_radius_arcsec"),
            raw=row.get("raw_json"),
        )
        for _, row in frame.iterrows()
    ]
    out = pd.DataFrame(records, columns=CATALOG_NEIGHBOR_COLUMNS)
    out = out.loc[
        out["candidate_id"].astype(str).str.strip().ne("")
        & out["catalog"].astype(str).str.strip().ne("")
        & pd.to_numeric(out["sep_arcsec"], errors="coerce").notna()
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=CATALOG_NEIGHBOR_COLUMNS)

    out["sep_arcsec"] = pd.to_numeric(out["sep_arcsec"], errors="coerce")
    out["period_days"] = pd.to_numeric(out["period_days"], errors="coerce")
    out["query_radius_arcsec"] = pd.to_numeric(out["query_radius_arcsec"], errors="coerce")
    out["is_known_variable"] = out["is_known_variable"].map(bool)
    out["is_dipper_contaminant"] = out["is_dipper_contaminant"].map(bool)
    out = out.sort_values(["candidate_id", "catalog", "sep_arcsec"], kind="mergesort")

    computed_rank = out.groupby(["candidate_id", "catalog"]).cumcount() + 1
    missing_rank = pd.to_numeric(out["rank"], errors="coerce").isna()
    out.loc[missing_rank, "rank"] = computed_rank.loc[missing_rank]
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    return out.loc[:, CATALOG_NEIGHBOR_COLUMNS].reset_index(drop=True)


def _vsx_neighbor_rows(
    neighbors_long: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str | None, str | None, str | None]:
    if neighbors_long is None or neighbors_long.empty:
        return pd.DataFrame(), None, None, None

    if "candidate_id" not in neighbors_long.columns:
        return pd.DataFrame(), None, None, None

    catalog_col = _first_existing_column(neighbors_long, ("catalog", "source_catalog"))
    if catalog_col is not None:
        vsx_mask = neighbors_long[catalog_col].astype(str).str.contains("vsx", case=False, na=False)
        rows = neighbors_long.loc[vsx_mask].copy()
    else:
        rows = neighbors_long.copy()
    if rows.empty:
        return pd.DataFrame(), None, None, None

    sep_col = _first_existing_column(rows, ("sep_arcsec", "vsx_sep_arcsec", "separation_arcsec"))
    type_col = _first_existing_column(rows, ("Type", "type", "vsx_type", "vsx_class", "class"))
    if sep_col is None or type_col is None:
        return pd.DataFrame(), None, None, None

    period_col = _first_existing_column(rows, ("Period", "period", "vsx_period"))
    return rows, sep_col, type_col, period_col


def vsx_evidence_from_neighbors(
    neighbors_long: pd.DataFrame | None,
    *,
    max_sep_arcsec: float = 3.0,
) -> pd.DataFrame:
    """Return closest definite VSX identity evidence derived from neighbor rows."""
    rows, sep_col, type_col, period_col = _vsx_neighbor_rows(neighbors_long)
    if rows.empty or sep_col is None or type_col is None:
        return _empty_columns(_VSX_EVIDENCE_COLUMNS)

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
        return _empty_columns(_VSX_EVIDENCE_COLUMNS)

    out = out.sort_values(["candidate_id", "vsx_sep_arcsec"], kind="mergesort")
    return out.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)


def nearby_vsx_dipper_contaminants_from_neighbors(
    neighbors_long: pd.DataFrame | None,
    *,
    max_sep_arcsec: float = NEIGHBOR_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Return closest nearby VSX dipper-contaminant evidence from neighbor rows."""
    rows, sep_col, type_col, period_col = _vsx_neighbor_rows(neighbors_long)
    if rows.empty or sep_col is None or type_col is None:
        return _empty_columns(_NEARBY_VSX_DIPPER_COLUMNS)

    out = pd.DataFrame(
        {
            "candidate_id": rows["candidate_id"].astype(str),
            "nearby_vsx_dipper_class": rows[type_col].map(
                lambda value: "" if _is_missing(value) else str(value).strip()
            ),
            "nearby_vsx_dipper_sep_arcsec": pd.to_numeric(rows[sep_col], errors="coerce"),
        },
        index=rows.index,
    )
    out["nearby_vsx_dipper_period"] = (
        pd.to_numeric(rows[period_col], errors="coerce")
        if period_col is not None
        else pd.Series(np.nan, index=rows.index)
    )

    dipper_mimic = out["nearby_vsx_dipper_class"].map(
        lambda value: is_dipper_contaminant_type_value("vsx_class", value)
    ).astype(bool)
    close = out["nearby_vsx_dipper_sep_arcsec"].le(float(max_sep_arcsec))
    out = out.loc[dipper_mimic & close].dropna(subset=["nearby_vsx_dipper_sep_arcsec"]).copy()
    if out.empty:
        return _empty_columns(_NEARBY_VSX_DIPPER_COLUMNS)

    out["nearby_vsx_dipper_contaminant"] = True
    out = out.sort_values(["candidate_id", "nearby_vsx_dipper_sep_arcsec"], kind="mergesort")
    out = out.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)
    return out[_NEARBY_VSX_DIPPER_COLUMNS]


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
    nearby_vsx_max_sep_arcsec: float = NEIGHBOR_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Fill canonical catalog-evidence fields from known synonyms/context.

    Existing canonical values always win. Neighbor-derived VSX classes are
    promoted only for close, definite known-variable VSX types.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    out = df.copy()
    if out.empty:
        for column in (
            "vsx_class",
            "vsx_period",
            "vsx_sep_arcsec",
            "nearby_vsx_dipper_contaminant",
            "nearby_vsx_dipper_class",
            "nearby_vsx_dipper_sep_arcsec",
            "nearby_vsx_dipper_period",
        ):
            _ensure_flat_column(out, column)
        return out

    _fill_blank(out, "vsx_class", _series_for_column(out, "vsx_type"))

    evidence = vsx_evidence_from_neighbors(neighbors_long, max_sep_arcsec=vsx_max_sep_arcsec)
    nearby_dipper = nearby_vsx_dipper_contaminants_from_neighbors(
        neighbors_long,
        max_sep_arcsec=nearby_vsx_max_sep_arcsec,
    )
    if "candidate_id" not in out.columns:
        return out

    merged = out
    if not evidence.empty:
        merged = merged.merge(
            evidence,
            on="candidate_id",
            how="left",
            suffixes=("", "_neighbor_vsx"),
        )
        for column in ("vsx_class", "vsx_period", "vsx_sep_arcsec"):
            neighbor_col = f"{column}_neighbor_vsx"
            if neighbor_col in merged.columns:
                _fill_blank(merged, column, merged[neighbor_col])

    if not nearby_dipper.empty:
        merged = merged.merge(
            nearby_dipper,
            on="candidate_id",
            how="left",
            suffixes=("", "_nearby_vsx_dipper"),
        )
        for column in (
            "nearby_vsx_dipper_contaminant",
            "nearby_vsx_dipper_class",
            "nearby_vsx_dipper_sep_arcsec",
            "nearby_vsx_dipper_period",
        ):
            neighbor_col = f"{column}_nearby_vsx_dipper"
            if neighbor_col in merged.columns:
                _fill_blank(merged, column, merged[neighbor_col])

    drop_cols = [
        col
        for col in merged.columns
        if col.endswith("_neighbor_vsx") or col.endswith("_nearby_vsx_dipper")
    ]
    return merged.drop(columns=drop_cols)
