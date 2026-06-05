from __future__ import annotations

from pathlib import Path
import hashlib
import math

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

RA_ALIASES: tuple[str, ...] = ("ra", "ra_deg", "RA", "RAJ2000", "RA_ICRS")
DEC_ALIASES: tuple[str, ...] = ("dec", "dec_deg", "DEC", "DEJ2000", "DE_ICRS")


def _first_existing_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        found = lower_lookup.get(alias.lower())
        if found is not None:
            return found
    return None


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

    ra_col = _first_existing_column(out, RA_ALIASES)
    dec_col = _first_existing_column(out, DEC_ALIASES)
    if ra_col is not None:
        out["ra"] = pd.to_numeric(out[ra_col], errors="coerce")
        out["ra_deg"] = out["ra"]
    else:
        out["ra"] = np.nan
        out["ra_deg"] = np.nan
    if dec_col is not None:
        out["dec"] = pd.to_numeric(out[dec_col], errors="coerce")
        out["dec_deg"] = out["dec"]
    else:
        out["dec"] = np.nan
        out["dec_deg"] = np.nan

    if "timescale" not in out.columns:
        out["timescale"] = default_timescale
    if "source_label" not in out.columns:
        source_col = _first_existing_column(out, ("catalog_source", "source_catalog", "source_key"))
        out["source_label"] = out[source_col].astype(str) if source_col else ""

    return out
