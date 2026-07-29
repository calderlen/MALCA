"""Event-spacing period estimates from significant dip/jump run epochs.

Complementary to periodogram methods: when a light curve shows a handful of
well-separated dips, the pairwise spacings (and their approximate GCD) are
often a cleaner estimate of the recurrence period than a short-window PDM/CE
search that locks onto high-order harmonics.

Methods
-------
median_dt
    Median consecutive inter-event spacing. Used for exactly 2 events, or as
    a fallback when the GCD estimate is unstable.
gcd_dt
    Approximate GCD of all pairwise differences (Buccheri-style). Robust to
    missed events when ``n_events >= 3``.
single_event_prior
    One event only: no period, but ``baseline/2`` is returned as a soft prior
    for long-period LS refinement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from malca.config import (
    EVENT_PERIOD_MIN_EVENTS_HIGH,
    EVENT_PERIOD_REFINE_FRACTION,
    EVENT_PERIOD_REL_STD_HIGH,
    LONG_PERIOD_BASELINE_FRACTION,
)


def _finite_positive(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _approx_gcd(
    diffs: np.ndarray,
    *,
    rel_tol: float = 0.08,
    max_nearest_multiple: float = 32.0,
) -> float | None:
    """Approximate GCD of positive diffs via iterative remainder reduction.

    Diffs that are near-integer multiples of a common base survive; outliers
    from timing jitter or missed-event aliases are absorbed by the relative
    tolerance. Returns ``None`` when no stable base emerges.
    """
    vals = np.sort(_finite_positive(diffs))
    if vals.size == 0:
        return None
    if vals.size == 1:
        return float(vals[0])

    base = float(vals[0])
    for value in vals[1:]:
        v = float(value)
        while v > rel_tol * base and base > 0:
            if v < base:
                base, v = v, base
            # Closest integer multiple remainder.
            n = max(1, int(round(v / base)))
            rem = abs(v - n * base)
            if rem <= rel_tol * base:
                # Accept this as consistent with current base.
                # Re-estimate base as weighted mean of implied periods.
                implied = v / n
                base = 0.5 * (base + implied)
                break
            v = rem
        else:
            # No common base with this value; keep going with current base.
            continue
    if not np.isfinite(base) or base <= 0:
        return None

    # With irregular floating-point spacings, repeated Euclidean remainders
    # can collapse to machine epsilon.  Such a value trivially makes every
    # spacing look like an integer multiple and previously produced false
    # "high-confidence" periods near zero.  A useful missed-event GCD must
    # explain at least the shortest observed spacing without requiring an
    # implausibly enormous cycle count.
    nearest_multiple = float(np.min(vals) / base)
    if (
        not np.isfinite(nearest_multiple)
        or nearest_multiple > float(max_nearest_multiple)
    ):
        return None
    return float(base)


def _pairwise_diffs(epochs: np.ndarray) -> np.ndarray:
    diffs: list[float] = []
    for i in range(epochs.size):
        for j in range(i + 1, epochs.size):
            dt = float(epochs[j] - epochs[i])
            if dt > 0:
                diffs.append(dt)
    return np.asarray(diffs, dtype=float)


def _consecutive_diffs(epochs: np.ndarray) -> np.ndarray:
    if epochs.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(epochs).astype(float)


def event_based_period(
    run_epochs: Sequence[float],
    run_depths: Sequence[float] | None = None,
    *,
    baseline_days: float | None,
    min_events: int = 2,
    max_baseline_fraction: float = LONG_PERIOD_BASELINE_FRACTION,
) -> dict[str, Any]:
    """Estimate a recurrence period from significant run center epochs.

    Parameters
    ----------
    run_epochs:
        Center JDs of significant dip/jump runs (sorted not required).
    run_depths:
        Optional peak significance per run. Currently unused for the period
        estimate but retained for future depth-weighted variants and logged
        in the result for provenance.
    baseline_days:
        Light-curve time span. Caps the accepted period and supplies the
        single-event prior (``baseline / 2``).
    min_events:
        Minimum number of events required for a period (not a prior).
    max_baseline_fraction:
        Reject periods longer than ``fraction * baseline``.

    Returns
    -------
    dict with keys:
        event_period_days, event_period_std, event_period_n_events,
        event_period_method, event_period_is_high_confidence,
        event_period_rel_std, event_period_prior_days, event_period_status.
    """
    empty: dict[str, Any] = {
        "event_period_days": float("nan"),
        "event_period_std": float("nan"),
        "event_period_n_events": 0,
        "event_period_method": "none",
        "event_period_is_high_confidence": False,
        "event_period_rel_std": float("nan"),
        "event_period_prior_days": float("nan"),
        "event_period_status": "no_events",
        "event_period_depths": [],
    }

    epochs = np.asarray(list(run_epochs), dtype=float)
    epochs = epochs[np.isfinite(epochs)]
    if epochs.size == 0:
        return empty
    epochs = np.sort(np.unique(epochs))
    n_events = int(epochs.size)

    depths: list[float] = []
    if run_depths is not None:
        depth_arr = np.asarray(list(run_depths), dtype=float)
        if depth_arr.size == epochs.size:
            # Align depths to the unique/sorted epochs by re-indexing the
            # original (pre-unique) finite list; for typical inputs depths
            # are already 1:1 with unique centers.
            depths = [float(v) if np.isfinite(v) else float("nan") for v in depth_arr]

    baseline = None
    try:
        baseline = float(baseline_days) if baseline_days is not None else None
    except (TypeError, ValueError):
        baseline = None
    if baseline is not None and (not np.isfinite(baseline) or baseline <= 0):
        baseline = None

    max_period = (
        float(max_baseline_fraction) * baseline
        if baseline is not None
        else float("inf")
    )
    prior = float(baseline / 2.0) if baseline is not None else float("nan")

    result = dict(empty)
    result["event_period_n_events"] = n_events
    result["event_period_depths"] = depths
    result["event_period_prior_days"] = prior

    if n_events < int(min_events):
        if n_events == 1 and np.isfinite(prior) and prior > 0:
            result["event_period_method"] = "single_event_prior"
            result["event_period_status"] = "prior_only"
            result["event_period_days"] = float(prior)
            return result
        result["event_period_status"] = "too_few_events"
        return result

    consecutive = _consecutive_diffs(epochs)
    consecutive = consecutive[np.isfinite(consecutive) & (consecutive > 0)]
    pairwise = _pairwise_diffs(epochs)

    method = "median_dt"
    period = float("nan")
    period_std = float("nan")

    if n_events >= 3 and pairwise.size:
        gcd = _approx_gcd(pairwise)
        if gcd is not None and gcd > 0 and gcd <= max_period:
            # Express each pairwise Δt as nearest integer multiple of gcd;
            # the residual scatter is the period uncertainty.
            multiples = np.maximum(1, np.round(pairwise / gcd))
            implied = pairwise / multiples
            period = float(np.median(implied))
            period_std = float(np.std(implied)) if implied.size > 1 else 0.0
            method = "gcd_dt"

    if not np.isfinite(period) or period <= 0:
        if consecutive.size == 0:
            result["event_period_status"] = "no_positive_diffs"
            return result
        period = float(np.median(consecutive))
        period_std = float(np.std(consecutive)) if consecutive.size > 1 else 0.0
        method = "median_dt"

    if period > max_period:
        result["event_period_status"] = "exceeds_baseline_fraction"
        result["event_period_method"] = method
        result["event_period_days"] = float(period)
        result["event_period_std"] = float(period_std)
        return result

    rel_std = float(period_std / period) if period > 0 and np.isfinite(period_std) else float("nan")
    high_conf = bool(
        n_events >= int(EVENT_PERIOD_MIN_EVENTS_HIGH)
        and np.isfinite(rel_std)
        and rel_std <= float(EVENT_PERIOD_REL_STD_HIGH)
        and method == "gcd_dt"
    )

    result.update(
        {
            "event_period_days": float(period),
            "event_period_std": float(period_std) if np.isfinite(period_std) else float("nan"),
            "event_period_method": method,
            "event_period_is_high_confidence": high_conf,
            "event_period_rel_std": rel_std,
            "event_period_status": "ok",
        }
    )
    return result


def refine_event_period(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    event_period_days: float,
    *,
    fraction: float = EVENT_PERIOD_REFINE_FRACTION,
    n_periods: int = 2000,
    method: str = "ce",
) -> dict[str, Any]:
    """Fine-grid PDM/CE refine around ``event_period_days``.

    Searches ``[P*(1-fraction), P*(1+fraction)]`` and returns the refined peak.
    Used as an optional follow-up when event spacing supplies a prior but we
    want a photometry-backed period for consensus.
    """
    from malca.core.stats import compute_ce_stats, compute_pdm_stats

    period = float(event_period_days) if np.isfinite(event_period_days) else float("nan")
    empty = {
        "refined_period_days": float("nan"),
        "refined_method": str(method),
        "refine_min_period_days": float("nan"),
        "refine_max_period_days": float("nan"),
        "refine_status": "unavailable",
    }
    if not np.isfinite(period) or period <= 0:
        empty["refine_status"] = "invalid_period"
        return empty

    frac = float(fraction) if np.isfinite(fraction) and fraction > 0 else 0.2
    min_p = period * (1.0 - frac)
    max_p = period * (1.0 + frac)
    if max_p <= min_p:
        max_p = min_p * 1.2

    jd_arr = np.asarray(jd, dtype=float)
    mag_arr = np.asarray(mag, dtype=float)
    err_arr = (
        np.asarray(err, dtype=float)
        if err is not None
        else np.ones_like(mag_arr)
    )

    try:
        if str(method).lower() == "pdm":
            result = compute_pdm_stats(
                jd_arr,
                mag_arr,
                err_arr,
                min_period=min_p,
                max_period=max_p,
                n_periods=int(n_periods),
                n_bootstrap=0,
                refine=True,
            )
            refined = float(result.get("pdm_period", float("nan")))
        else:
            result = compute_ce_stats(
                jd_arr,
                mag_arr,
                err_arr,
                min_period=min_p,
                max_period=max_p,
                n_periods=int(n_periods),
                n_bootstrap=0,
                refine=True,
            )
            refined = float(result.get("ce_period", float("nan")))
    except Exception as exc:  # pragma: no cover - defensive
        empty["refine_status"] = f"error:{exc}"
        empty["refine_min_period_days"] = float(min_p)
        empty["refine_max_period_days"] = float(max_p)
        return empty

    return {
        "refined_period_days": refined,
        "refined_method": str(method).lower(),
        "refine_min_period_days": float(min_p),
        "refine_max_period_days": float(max_p),
        "refine_status": "ok" if np.isfinite(refined) and refined > 0 else "no_peak",
        "refine_raw": result,
    }


__all__ = [
    "event_based_period",
    "refine_event_period",
]
