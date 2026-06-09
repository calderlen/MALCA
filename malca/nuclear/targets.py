from __future__ import annotations

import json
from pathlib import Path
import hashlib
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


ID_ALIASES: tuple[str, ...] = (
    "candidate_id",
    "asas_sn_id",
    "asassn_id",
    "source_id",
    "gaia_id",
    "tde_id",
    "clagn_id",
    "name",
    "object_id",
)

RA_ALIASES: tuple[str, ...] = ("ra", "ra_deg", "ra_gaia", "RA", "RAJ2000", "RA_ICRS")
DEC_ALIASES: tuple[str, ...] = ("dec", "dec_deg", "dec_gaia", "DEC", "DEJ2000", "DE_ICRS")
NESTED_COORD_COLUMNS: tuple[str, ...] = ("external_stats", "payload_json")
NESTED_COORD_GROUPS: tuple[str, ...] = ("external_stats", "lc_stats", "derived_stats")


def _first_existing_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        found = lower_lookup.get(alias.lower())
        if found is not None:
            return found
    return None


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _mapping_value(mapping: Mapping[str, object], aliases: tuple[str, ...]) -> object:
    lower_lookup = {str(key).lower(): key for key in mapping.keys()}
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
        found = lower_lookup.get(alias.lower())
        if found is not None:
            return mapping[found]
    return np.nan


def _is_missing_coord_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _nested_coord_value(value: object, aliases: tuple[str, ...]) -> object:
    mapping = _as_mapping(value)
    if mapping is None:
        return np.nan

    direct = _mapping_value(mapping, aliases)
    if not _is_missing_coord_value(direct):
        return direct

    for group in NESTED_COORD_GROUPS:
        nested = _as_mapping(mapping.get(group))
        if nested is None:
            continue
        nested_value = _mapping_value(nested, aliases)
        if not _is_missing_coord_value(nested_value):
            return nested_value
    return np.nan


def _coord_series_from_nested(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    series: pd.Series | None = None
    for col in NESTED_COORD_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col].map(lambda value: _nested_coord_value(value, aliases)), errors="coerce")
        series = values if series is None else series.combine_first(values)
    return series


def _coord_series(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    col = _first_existing_column(df, aliases)
    if col is not None:
        values = pd.to_numeric(df[col], errors="coerce")
    else:
        values = pd.Series(np.nan, index=df.index, dtype=float)

    nested = _coord_series_from_nested(df, aliases)
    if nested is not None:
        values = values.combine_first(nested)
    return values


def _clean_identifier(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _coord_identifier(ra: object, dec: object, index: object) -> str:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except Exception:
        ra_f = math.nan
        dec_f = math.nan
    if math.isfinite(ra_f) and math.isfinite(dec_f):
        digest = hashlib.sha1(f"{ra_f:.7f}:{dec_f:.7f}".encode("ascii")).hexdigest()[:10]
        return f"NUC-{digest}"
    return f"NUC-{index}"


def _candidate_ids(df: pd.DataFrame, prefix: str) -> pd.Series:
    for col in ID_ALIASES:
        if col not in df.columns:
            continue
        values = df[col].map(_clean_identifier)
        if values.ne("").any():
            return values

    path_col = _first_existing_column(df, ("path", "lc_path", "local_lc_path"))
    if path_col is not None:
        values = df[path_col].map(lambda value: Path(str(value)).stem if _clean_identifier(value) else "")
        if values.ne("").any():
            return values

    ra_col = _first_existing_column(df, RA_ALIASES)
    dec_col = _first_existing_column(df, DEC_ALIASES)
    if ra_col is not None and dec_col is not None:
        return pd.Series(
            [
                _coord_identifier(row[ra_col], row[dec_col], idx)
                for idx, row in df[[ra_col, dec_col]].iterrows()
            ],
            index=df.index,
            dtype=object,
        )
    return pd.Series([f"{prefix}-{idx}" for idx in range(len(df))], index=df.index, dtype=object)


def normalize_nuclear_targets(
    df: pd.DataFrame,
    *,
    id_prefix: str = "NUC",
    default_timescale: str = "nuclear",
) -> pd.DataFrame:
    """Return a copy of *df* with canonical nuclear target columns.

    The normalizer is deliberately permissive because the AGN, TDE, CLAGN, and
    LTV-derived notebooks use slightly different coordinate and identifier
    names.  It preserves all input columns and adds/standardizes the fields
    consumed by the nuclear context pipeline.
    """
    out = df.copy()
    if out.empty:
        if "candidate_id" not in out.columns:
            out["candidate_id"] = pd.Series(dtype=object)
        if "ra" not in out.columns:
            out["ra"] = pd.Series(dtype=float)
        if "dec" not in out.columns:
            out["dec"] = pd.Series(dtype=float)
        if "ra_deg" not in out.columns:
            out["ra_deg"] = pd.Series(dtype=float)
        if "dec_deg" not in out.columns:
            out["dec_deg"] = pd.Series(dtype=float)
        return out

    if "candidate_id" not in out.columns:
        out["candidate_id"] = _candidate_ids(out, id_prefix)
    else:
        ids = out["candidate_id"].map(_clean_identifier)
        missing = ids.eq("")
        if missing.any():
            fallback = _candidate_ids(out.loc[missing], id_prefix)
            ids.loc[missing] = fallback
        out["candidate_id"] = ids
    out["candidate_id"] = out["candidate_id"].astype(str)

    out["ra"] = _coord_series(out, RA_ALIASES)
    out["ra_deg"] = out["ra"]
    out["dec"] = _coord_series(out, DEC_ALIASES)
    out["dec_deg"] = out["dec"]

    if "timescale" not in out.columns:
        out["timescale"] = default_timescale
    if "source_label" not in out.columns:
        source_col = _first_existing_column(out, ("catalog_source", "source_catalog", "source_key"))
        out["source_label"] = out[source_col].astype(str) if source_col else ""

    return out
