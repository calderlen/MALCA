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
    optical_slope_col: str = "Slope",
    w1_w2_median_col: str = "w1_w2_median",
    w1_w2_slope_col: str = "w1_w2_slope",
    optical_slope_thresh: float = LTV_DUST_OPTICAL_SLOPE_THRESH,
    color_slope_thresh: float = LTV_DUST_COLOR_SLOPE_THRESH,
    color_excess_thresh: float = LTV_DUST_COLOR_EXCESS_THRESH,
) -> pd.DataFrame:
    """
    Add dust-driven variability flags to a dataframe.

    Adds columns:
    - dust_excess: bool (W1-W2 median > threshold)
    - dust_trend_class: {"redder+fainter","bluer+brighter",None}
    - dust_trend_flag: bool (either dust_forming or dust_clearing)
    - dust_candidate: bool (dust_excess OR dust_trend_flag)
    """
    if df.empty:
        return df

    if optical_slope_col not in df.columns or w1_w2_median_col not in df.columns or w1_w2_slope_col not in df.columns:
        return df

    df = df.copy()

    opt_slope = df[optical_slope_col].astype(float)
    color_med = df[w1_w2_median_col].astype(float)
    color_slope = df[w1_w2_slope_col].astype(float)

    dust_excess = color_med > color_excess_thresh
    dust_forming = (opt_slope >= optical_slope_thresh) & (color_slope >= color_slope_thresh)
    dust_clearing = (opt_slope <= -optical_slope_thresh) & (color_slope <= -color_slope_thresh)

    trend_class = np.where(dust_forming, "redder+fainter",
                   np.where(dust_clearing, "bluer+brighter", None))

    df["dust_excess"] = dust_excess
    df["dust_trend_class"] = trend_class
    df["dust_trend_flag"] = dust_forming | dust_clearing
    df["dust_candidate"] = df["dust_excess"] | df["dust_trend_flag"]

    return df
