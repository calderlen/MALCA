"""Helpers for normalizing VSX catalog and crossmatch metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_asas_sn_ids(series: pd.Series) -> pd.Series:
    """Normalize ASAS-SN IDs for string-keyed joins."""
    s = series.astype(str).str.strip()
    s = s.replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA, "": pd.NA})
    num = pd.to_numeric(s, errors="coerce")
    integral = num.notna() & np.isfinite(num) & (num % 1 == 0)
    if integral.any():
        s.loc[integral] = num.loc[integral].astype("Int64").astype(str)
    return s


def normalize_vsx_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with VSX fields using review-standard column names."""
    out = df.copy()

    rename_map: dict[str, str] = {}
    if "ASAS-SN ID" in out.columns and "asas_sn_id" not in out.columns:
        rename_map["ASAS-SN ID"] = "asas_sn_id"
    if "asassn_id" in out.columns and "asas_sn_id" not in out.columns:
        rename_map["asassn_id"] = "asas_sn_id"
    if "class" in out.columns and "vsx_class" not in out.columns:
        rename_map["class"] = "vsx_class"
    if "sep_arcsec" in out.columns and "vsx_sep_arcsec" not in out.columns:
        rename_map["sep_arcsec"] = "vsx_sep_arcsec"
    if "period" in out.columns and "vsx_period" not in out.columns:
        rename_map["period"] = "vsx_period"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "asas_sn_id" in out.columns:
        out["asas_sn_id"] = normalize_asas_sn_ids(out["asas_sn_id"])
    if "vsx_sep_arcsec" in out.columns:
        out["vsx_sep_arcsec"] = pd.to_numeric(out["vsx_sep_arcsec"], errors="coerce")
    if "vsx_period" in out.columns:
        out["vsx_period"] = pd.to_numeric(out["vsx_period"], errors="coerce")
    return out


def select_best_vsx_matches(
    df: pd.DataFrame,
    *,
    id_column: str = "asas_sn_id",
) -> pd.DataFrame:
    """Deduplicate VSX matches, keeping the nearest match when separations exist."""
    if id_column not in df.columns:
        return df.copy()
    out = df.copy()
    if "vsx_sep_arcsec" in out.columns:
        out = out.sort_values("vsx_sep_arcsec", na_position="last")
    return out.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
