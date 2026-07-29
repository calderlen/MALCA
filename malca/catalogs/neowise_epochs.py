"""Shared robust epoch aggregation for NEOWISE single-exposure photometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from malca.config import NEOWISE_EPOCH_COMBINE_DAYS


_BANDS = ("w1", "w2")


def _robust_epoch_band(
    frame: pd.DataFrame,
    *,
    mag_col: str,
    err_col: str,
) -> dict[str, float | int]:
    mag = (
        pd.to_numeric(frame[mag_col], errors="coerce").to_numpy(dtype=float)
        if mag_col in frame.columns
        else np.full(len(frame), np.nan, dtype=float)
    )
    good = np.isfinite(mag)
    values = mag[good]
    n_values = int(values.size)
    if n_values == 0:
        return {
            "median": np.nan,
            "error": np.nan,
            "scatter": np.nan,
            "n_points": 0,
        }

    median = float(np.median(values))
    scatter = (
        float(1.4826 * np.median(np.abs(values - median)))
        if n_values >= 2
        else np.nan
    )
    scatter_sem = scatter / np.sqrt(n_values) if np.isfinite(scatter) else np.nan

    formal_sem = np.nan
    if err_col in frame.columns:
        errors = pd.to_numeric(frame[err_col], errors="coerce").to_numpy(dtype=float)[good]
        valid_errors = np.isfinite(errors) & (errors > 0)
        if valid_errors.any():
            formal_sem = float(
                np.sqrt(np.mean(np.square(errors[valid_errors]))) / np.sqrt(valid_errors.sum())
            )

    finite_terms = [value for value in (formal_sem, scatter_sem) if np.isfinite(value)]
    epoch_error = float(max(finite_terms)) if finite_terms else np.nan
    return {
        "median": median,
        "error": epoch_error,
        "scatter": scatter,
        "n_points": n_values,
    }


def combine_neowise_epochs(
    light_curve: pd.DataFrame,
    *,
    epoch_days: float = NEOWISE_EPOCH_COMBINE_DAYS,
) -> pd.DataFrame:
    """Combine nearby NEOWISE exposures into robust visit-level measurements.

    Consecutive measurements separated by no more than ``epoch_days`` belong
    to one visit. Each band is summarized independently with its median. The
    plotted error is the larger of the formal error on the visit measurement
    and the robust within-visit scatter divided by ``sqrt(N)``.

    The returned frame retains the canonical ``w?mpro``/``w?sigmpro`` names
    used by review plots and the historical ``w?err`` aliases used by the LTV
    trend code.
    """
    columns = [
        "mjd",
        "w1mpro",
        "w1sigmpro",
        "w1err",
        "w1_scatter",
        "w1_n_points",
        "w2mpro",
        "w2sigmpro",
        "w2err",
        "w2_scatter",
        "w2_n_points",
        "w1_w2",
        "w1_w2_err",
        "n_points",
        "neowise_epoch_binned",
    ]
    if light_curve is None or light_curve.empty or "mjd" not in light_curve.columns:
        return pd.DataFrame(columns=columns)

    if (
        "neowise_epoch_binned" in light_curve.columns
        and light_curve["neowise_epoch_binned"].fillna(False).astype(bool).all()
    ):
        out = light_curve.copy()
        return out.sort_values("mjd", kind="stable").reset_index(drop=True)

    frame = light_curve.copy()
    frame["mjd"] = pd.to_numeric(frame["mjd"], errors="coerce")
    frame = frame[np.isfinite(frame["mjd"])].sort_values("mjd", kind="stable").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    gap_days = float(epoch_days)
    if not np.isfinite(gap_days) or gap_days <= 0:
        raise ValueError("epoch_days must be a positive finite number")
    gaps = frame["mjd"].diff()
    frame["_neowise_epoch_id"] = gaps.gt(gap_days).fillna(False).cumsum()

    rows: list[dict[str, float | int | bool]] = []
    for _epoch_id, epoch in frame.groupby("_neowise_epoch_id", sort=True):
        row: dict[str, float | int | bool] = {
            "mjd": float(np.median(epoch["mjd"].to_numpy(dtype=float))),
            "n_points": int(len(epoch)),
            "neowise_epoch_binned": True,
        }
        for band in _BANDS:
            result = _robust_epoch_band(
                epoch,
                mag_col=f"{band}mpro",
                err_col=f"{band}sigmpro",
            )
            row[f"{band}mpro"] = float(result["median"])
            row[f"{band}sigmpro"] = float(result["error"])
            row[f"{band}err"] = float(result["error"])
            row[f"{band}_scatter"] = float(result["scatter"])
            row[f"{band}_n_points"] = int(result["n_points"])

        if not (np.isfinite(row["w1mpro"]) or np.isfinite(row["w2mpro"])):
            continue
        row["w1_w2"] = (
            float(row["w1mpro"] - row["w2mpro"])
            if np.isfinite(row["w1mpro"]) and np.isfinite(row["w2mpro"])
            else np.nan
        )
        row["w1_w2_err"] = (
            float(np.hypot(row["w1sigmpro"], row["w2sigmpro"]))
            if np.isfinite(row["w1sigmpro"]) and np.isfinite(row["w2sigmpro"])
            else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
