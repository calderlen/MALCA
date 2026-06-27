"""Shared quality filtering for NEOWISE single-exposure photometry."""

from __future__ import annotations

import pandas as pd

from malca.config import NEOWISE_MIN_SNR, NEOWISE_QUAL_FRAME_MIN


def _cc_flags_series(cc_col: pd.Series) -> pd.Series:
    if cc_col.empty:
        return cc_col.astype(str)
    if cc_col.dtype == object and isinstance(cc_col.iloc[0], bytes):
        return cc_col.str.decode("utf-8")
    return cc_col.astype(str)


def filter_neowise_single_exposure_lc(
    lc: pd.DataFrame,
    *,
    min_snr: float = NEOWISE_MIN_SNR,
    qual_frame_min: float = NEOWISE_QUAL_FRAME_MIN,
) -> pd.DataFrame:
    """Apply standard NEOWISE-R single-exposure quality cuts.

    NEOWISE ``qual_frame`` is a frameset score on the 0/5/10 scale (higher is
    better). Older code incorrectly used ``qual_frame.isin([0, 1])``, which
    kept only the poorest frames and dropped good 5/10 data.

    ``qi_fact`` is not filtered here: it is a binary/ordinal image-shape flag
    and ``>= 0.9`` incorrectly rejected valid epochs where ``qi_fact`` is 0.0
    or 0.5 despite high SNR.
    """
    if lc is None or lc.empty:
        return pd.DataFrame() if lc is None else lc.iloc[0:0].copy()

    out = lc.copy()

    if "qual_frame" in out.columns:
        qual = pd.to_numeric(out["qual_frame"], errors="coerce")
        out = out[qual >= float(qual_frame_min)]

    if "cc_flags" in out.columns and not out.empty:
        cc = _cc_flags_series(out["cc_flags"])
        out = out[~cc.str.contains("[^0]", regex=True, na=False)]

    if "w1snr" in out.columns:
        out = out[pd.to_numeric(out["w1snr"], errors="coerce") >= float(min_snr)]
    if "w2snr" in out.columns:
        out = out[pd.to_numeric(out["w2snr"], errors="coerce") >= float(min_snr)]

    return out.reset_index(drop=True)
