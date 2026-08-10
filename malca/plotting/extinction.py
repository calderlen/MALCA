"""Foreground-extinction helpers for publication color-color diagrams."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from malca.extinction import mid_ir_av_coefficient


# Scalar A_lambda / A_V values for the project's R_V=3.1 catalog-color
# convention.  W3/W4 use the versioned nominal-wavelength G23 policy; keep all
# values aligned with ``Bandpass.av_coeff`` in ``malca.review.sed``.
IR_AV_COEFFICIENTS: dict[str, float] = {
    "tmass_j": 0.282,
    "tmass_h": 0.175,
    "tmass_k": 0.112,
    "w1": 0.061,
    "w2": 0.047,
    "w3": float(mid_ir_av_coefficient("AllWISE", "W3")),
    "w4": float(mid_ir_av_coefficient("AllWISE", "W4")),
}


def add_dereddened_ir_magnitudes(
    frame: pd.DataFrame,
    *,
    av_col: str = "A_v_3d",
    magnitude_columns: Iterable[str] = IR_AV_COEFFICIENTS,
    suffix: str = "_0",
) -> pd.DataFrame:
    """Add foreground-extinction-corrected 2MASS/AllWISE magnitudes.

    ``A_v_3d`` is the line-of-sight foreground correction.  A missing or
    negative value is treated as zero so a source remains plottable, matching
    the existing CMD and paper-figure convention.  This does not correct for
    source-local/circumstellar extinction.
    """
    out = frame.copy()
    if av_col in out.columns:
        av = pd.to_numeric(out[av_col], errors="coerce")
    else:
        av = pd.Series(np.nan, index=out.index, dtype=float)
    av = av.where(av >= 0.0, 0.0).fillna(0.0)

    for column in magnitude_columns:
        if column not in out.columns:
            continue
        coefficient = IR_AV_COEFFICIENTS[column]
        out[f"{column}{suffix}"] = pd.to_numeric(out[column], errors="coerce") - coefficient * av
    return out


def dereddened_color(
    frame: pd.DataFrame,
    left_band: str,
    right_band: str,
    *,
    suffix: str = "_0",
) -> pd.Series:
    """Return a color from columns written by :func:`add_dereddened_ir_magnitudes`."""
    left_col = f"{left_band}{suffix}"
    right_col = f"{right_band}{suffix}"
    if left_col not in frame.columns or right_col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[left_col], errors="coerce") - pd.to_numeric(frame[right_col], errors="coerce")
