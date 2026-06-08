"""Derived feature helpers for MALCA candidate tables.

Raw light-curve measurements live in :mod:`malca.stats`.  This module derives
secondary quantities from those raw measurements and from catalog metadata, so
the provenance of fitted/measured columns stays separate from algebraic ratios
and color-magnitude combinations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
import pandas as pd


DERIVED_FEATURE_COLUMNS: tuple[str, ...] = (
    "derived_harmonics_r32",
    "derived_harmonics_r42",
    "derived_harmonics_r43",
    "derived_harmonics_a4_a2",
    "derived_harmonics_b4_b2",
    "derived_bp_rp",
    "derived_j_k",
    "derived_mrp",
    "derived_mks",
    "derived_wrp",
    "derived_wjk",
)


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom
    valid = pd.Series(np.isfinite(out), index=out.index) & (denom != 0)
    return out.where(valid, np.nan)


def _numeric_column(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _distance_pc(df: pd.DataFrame) -> pd.Series:
    dist = pd.Series(np.nan, index=df.index, dtype=float)
    for name in (
        "bj_r_med_photogeo",
        "bj_r_med_geo",
        "distance_gspphot",
        "distance_pc",
        "dist",
    ):
        values = _numeric_column(df, (name,)).where(lambda s: s > 0, np.nan)
        dist = dist.combine_first(values)

    parallax = _numeric_column(df, ("parallax", "gaia_parallax"))
    parallax_dist = (1000.0 / parallax).where(parallax > 0, np.nan)
    return dist.combine_first(parallax_dist)


def append_derived_features(df: pd.DataFrame, *, overwrite: bool = False) -> pd.DataFrame:
    """Append deterministic derived features to a candidate table.

    The function accepts both raw ``compute_stats`` keys and flattened
    ``stats_*`` column names.  Existing derived columns are preserved unless
    ``overwrite`` is true.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    idx = out.index
    derived: dict[str, pd.Series] = {}

    a2 = _numeric_column(out, ("stats_harmonics_mag_2", "harmonics_mag_2"))
    a3 = _numeric_column(out, ("stats_harmonics_mag_3", "harmonics_mag_3"))
    a4 = _numeric_column(out, ("stats_harmonics_mag_4", "harmonics_mag_4"))
    derived["derived_harmonics_r32"] = _safe_ratio(a3, a2)
    derived["derived_harmonics_r42"] = _safe_ratio(a4, a2)
    derived["derived_harmonics_r43"] = _safe_ratio(a4, a3)

    sin_a2 = _numeric_column(out, ("stats_harmonics_a2", "harmonics_a2"))
    sin_a4 = _numeric_column(out, ("stats_harmonics_a4", "harmonics_a4"))
    cos_b2 = _numeric_column(out, ("stats_harmonics_b2", "harmonics_b2"))
    cos_b4 = _numeric_column(out, ("stats_harmonics_b4", "harmonics_b4"))
    derived["derived_harmonics_a4_a2"] = _safe_ratio(sin_a4, sin_a2)
    derived["derived_harmonics_b4_b2"] = _safe_ratio(cos_b4, cos_b2)

    bp = _numeric_column(out, ("phot_bp_mean_mag", "bp_mag", "gaia_bp_mag", "BP"))
    rp = _numeric_column(out, ("phot_rp_mean_mag", "rp_mag", "gaia_rp_mag", "RP"))
    bp_rp = _numeric_column(out, ("bp_rp", "BP_RP", "gaia_bp_rp"))
    bp_rp = bp_rp.combine_first(bp - rp)
    derived["derived_bp_rp"] = bp_rp

    jmag = _numeric_column(out, ("tmass_j", "j_mag", "Jmag", "J"))
    kmag = _numeric_column(out, ("tmass_k", "k_mag", "Kmag", "Ks", "K"))
    j_k = _numeric_column(out, ("j_k", "J_K", "J-K", "tmass_j_k"))
    j_k = j_k.combine_first(jmag - kmag)
    derived["derived_j_k"] = j_k

    dist_pc = _distance_pc(out)
    dist_mod = 5.0 * np.log10(dist_pc) - 5.0
    dist_mod = dist_mod.where(np.isfinite(dist_mod), np.nan)
    mrp = rp - dist_mod
    mks = kmag - dist_mod
    derived["derived_mrp"] = mrp
    derived["derived_mks"] = mks
    derived["derived_wrp"] = mrp - 1.3 * bp_rp
    derived["derived_wjk"] = mks - 0.686 * j_k

    for col in DERIVED_FEATURE_COLUMNS:
        values = derived.get(col, pd.Series(np.nan, index=idx, dtype=float))
        values = pd.to_numeric(values, errors="coerce")
        if overwrite or col not in out.columns:
            out[col] = values
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").combine_first(values)

    return out


def compute_derived_feature_row(row: Mapping[str, Any]) -> dict[str, float]:
    """Return derived features for one row-like mapping."""
    if not isinstance(row, Mapping):
        return {col: np.nan for col in DERIVED_FEATURE_COLUMNS}
    frame = append_derived_features(pd.DataFrame([dict(row)]), overwrite=True)
    return {
        col: float(frame.iloc[0][col]) if pd.notna(frame.iloc[0][col]) else np.nan
        for col in DERIVED_FEATURE_COLUMNS
    }
