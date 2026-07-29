"""
Dust-driven variability flags for LTV candidates.

Implements paper-style flags using optical slope vs. NEOWISE W1-W2 trends:
- Mid-IR excess: W1-W2 > 0.3 mag
- Dust forming: optical slope > +threshold AND W1-W2 slope > +threshold
  (fainter + redder)
- Dust clearing: optical slope < -threshold AND W1-W2 slope < -threshold
  (brighter + bluer)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from malca.config import (
    LTV_DUST_OPTICAL_SLOPE_THRESH,
    LTV_DUST_COLOR_SLOPE_THRESH,
    LTV_DUST_COLOR_EXCESS_THRESH,
)


def apply_dust_flags(
    df: pd.DataFrame,
    *,
    optical_slope_col: str = "ltv_slope",
    w1_w2_median_col: str = "ltv_neowise_w1_w2_median",
    w1_w2_slope_col: str = "ltv_neowise_w1_w2_slope",
    optical_slope_thresh: float = LTV_DUST_OPTICAL_SLOPE_THRESH,
    color_slope_thresh: float = LTV_DUST_COLOR_SLOPE_THRESH,
    color_excess_thresh: float = LTV_DUST_COLOR_EXCESS_THRESH,
) -> pd.DataFrame:
    """
    Add dust-driven variability flags to a dataframe.

    Adds columns:
    - ltv_dust_excess: bool (W1-W2 median > threshold)
    - ltv_dust_trend_class: {"redder+fainter","bluer+brighter",None}
    - ltv_dust_trend_flag: bool (either dust_forming or dust_clearing)
    - ltv_dust_candidate: bool (dust_excess OR dust_trend_flag)
    """
    df = df.copy()
    df["ltv_dust_version"] = "slope-color-v2"
    df["ltv_dust_optical_slope_threshold"] = float(optical_slope_thresh)
    df["ltv_dust_color_slope_threshold"] = float(color_slope_thresh)
    df["ltv_dust_color_excess_threshold"] = float(color_excess_thresh)
    df["ltv_dust_provenance_json"] = json.dumps(
        {
            "optical_slope_col": optical_slope_col,
            "w1_w2_median_col": w1_w2_median_col,
            "w1_w2_slope_col": w1_w2_slope_col,
            "slope_unit": "mag_per_year",
            "color": "W1-W2_mag",
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    required = (optical_slope_col, w1_w2_median_col, w1_w2_slope_col)
    missing_columns = [col for col in required if col not in df.columns]
    opt_slope = pd.to_numeric(df.get(optical_slope_col), errors="coerce") if optical_slope_col in df else pd.Series(np.nan, index=df.index)
    color_med = pd.to_numeric(df.get(w1_w2_median_col), errors="coerce") if w1_w2_median_col in df else pd.Series(np.nan, index=df.index)
    color_slope = pd.to_numeric(df.get(w1_w2_slope_col), errors="coerce") if w1_w2_slope_col in df else pd.Series(np.nan, index=df.index)
    valid_optical = opt_slope.notna() & np.isfinite(opt_slope)
    valid_color_median = color_med.notna() & np.isfinite(color_med)
    valid_color_slope = color_slope.notna() & np.isfinite(color_slope)

    dust_excess = pd.Series(pd.NA, index=df.index, dtype="boolean")
    dust_excess.loc[valid_color_median] = (
        color_med.loc[valid_color_median] > float(color_excess_thresh)
    )

    valid_trend = valid_optical & valid_color_slope
    dust_forming = pd.Series(False, index=df.index, dtype="boolean")
    dust_clearing = pd.Series(False, index=df.index, dtype="boolean")
    dust_forming.loc[valid_trend] = (
        (opt_slope.loc[valid_trend] >= float(optical_slope_thresh))
        & (color_slope.loc[valid_trend] >= float(color_slope_thresh))
    )
    dust_clearing.loc[valid_trend] = (
        (opt_slope.loc[valid_trend] <= -float(optical_slope_thresh))
        & (color_slope.loc[valid_trend] <= -float(color_slope_thresh))
    )
    trend_flag = pd.Series(pd.NA, index=df.index, dtype="boolean")
    trend_flag.loc[valid_trend] = (
        dust_forming.loc[valid_trend] | dust_clearing.loc[valid_trend]
    )

    trend_class = pd.Series(pd.NA, index=df.index, dtype="string")
    trend_class.loc[dust_forming.fillna(False)] = "redder+fainter"
    trend_class.loc[dust_clearing.fillna(False)] = "bluer+brighter"

    df["ltv_dust_excess"] = dust_excess
    df["ltv_dust_excess_status"] = np.where(valid_color_median, "ok", "missing_color_median")
    df["ltv_dust_trend_class"] = trend_class
    df["ltv_dust_trend_flag"] = trend_flag
    df["ltv_dust_trend_status"] = np.where(valid_trend, "ok", "missing_slope_inputs")
    df["ltv_dust_candidate"] = dust_excess | trend_flag
    all_valid = valid_optical & valid_color_median & valid_color_slope
    any_valid = valid_optical | valid_color_median | valid_color_slope
    df["ltv_dust_status"] = np.where(all_valid, "ok", np.where(any_valid, "partial", "missing_inputs"))
    if missing_columns:
        df["ltv_dust_status"] = "missing_columns:" + ",".join(missing_columns)

    return df
