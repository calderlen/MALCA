"""Baseline-adaptive period search bounds.

Historically MALCA hard-coded period bounds:

* Pre-periodicity gate:  0.2 - 100 d
* Post-filter PDM / CE:  1   - 100 d
* Post-filter LS:        0.1 - 365 d
* Review UI:             0.1 - 2000 d (default TUI PDM window)
* LTV LS:                10  - 1000 d

Those bounds were fine for eclipsing binaries and short-period pulsators, but
they exclude long-recurrence dippers (e.g. AA Tau candidates with two dips
across several thousand days). This module returns bounds tailored to each
candidate's baseline, cadence, and processing stage. Every long-period signal
we care about lives at ``P <= 0.55 * baseline`` (i.e. at least ~1.8 cycles
observed), so the upper bound scales with baseline.

Design goals:

- One place computes bounds; every stage calls the same helper.
- Bounds never widen past what the data supports: ``max_period`` is capped by
  the observed baseline, and ``min_period`` is bounded by cadence.
- Deterministic and pure: same inputs -> same bounds.
- Robust against missing/invalid metadata (NaN baseline etc.) by falling back
  to the classic constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math

import numpy as np

from malca.config import (
    ADAPTIVE_BOUNDS_ENABLED,
    LONG_PERIOD_ABSOLUTE_CAP_DAYS,
    LONG_PERIOD_BASELINE_FRACTION,
    LONG_PERIOD_MIN_DAYS,
    PDM_MAX_PERIOD,
    PDM_MIN_PERIOD,
    PERIOD_BOUNDS_MAX_BASELINE_FRACTION,
    PERIOD_BOUNDS_MIN_CADENCE_MULTIPLIER,
    POST_FILTER_LEGACY_MAX_PERIOD,
    POST_FILTER_LEGACY_MIN_PERIOD,
    PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_PERIOD,
    REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS,
    REVIEW_PERIOD_MIN_DAYS,
)

STAGE_PREGATE = "pregate"
STAGE_POSTFILTER = "postfilter"
STAGE_REVIEW = "review"
STAGE_LONG = "long"
_KNOWN_STAGES: frozenset[str] = frozenset(
    {STAGE_PREGATE, STAGE_POSTFILTER, STAGE_REVIEW, STAGE_LONG}
)


@dataclass(frozen=True)
class PeriodBounds:
    """Resolved period search bounds for a given stage."""

    min_period_days: float
    max_period_days: float
    stage: str
    baseline_days: float | None
    cadence_median_days: float | None
    n_points: int | None
    reason: str

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.min_period_days), float(self.max_period_days))


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fallback_bounds(stage: str) -> tuple[float, float, str]:
    if stage == STAGE_PREGATE:
        return (
            float(PRE_PERIODICITY_MIN_PERIOD),
            float(PRE_PERIODICITY_MAX_PERIOD),
            "fallback:pregate_constants",
        )
    if stage == STAGE_POSTFILTER:
        # Prefer historical call-site defaults (1–100 d) when adaptive is off so
        # regression fixtures stay byte-identical to the pre-adaptive pipeline.
        if not ADAPTIVE_BOUNDS_ENABLED:
            return (
                float(POST_FILTER_LEGACY_MIN_PERIOD),
                float(POST_FILTER_LEGACY_MAX_PERIOD),
                "fallback:postfilter_legacy",
            )
        return (
            float(PDM_MIN_PERIOD),
            float(PDM_MAX_PERIOD),
            "fallback:postfilter_constants",
        )
    if stage == STAGE_REVIEW:
        return (
            float(REVIEW_PERIOD_MIN_DAYS),
            float(REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS),
            "fallback:review_constants",
        )
    if stage == STAGE_LONG:
        return (
            float(LONG_PERIOD_MIN_DAYS),
            float(LONG_PERIOD_ABSOLUTE_CAP_DAYS),
            "fallback:long_constants",
        )
    raise ValueError(f"Unknown stage: {stage!r}")


def _stage_absolute_max(stage: str) -> float:
    if stage == STAGE_PREGATE:
        return float(PRE_PERIODICITY_MAX_PERIOD)
    if stage == STAGE_POSTFILTER:
        return float(PDM_MAX_PERIOD)
    if stage == STAGE_REVIEW:
        return float(REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS)
    if stage == STAGE_LONG:
        return float(LONG_PERIOD_ABSOLUTE_CAP_DAYS)
    raise ValueError(f"Unknown stage: {stage!r}")


def _stage_min_floor(stage: str) -> float:
    if stage == STAGE_PREGATE:
        return float(PRE_PERIODICITY_MIN_PERIOD)
    if stage == STAGE_POSTFILTER:
        return float(PDM_MIN_PERIOD)
    if stage == STAGE_REVIEW:
        return float(REVIEW_PERIOD_MIN_DAYS)
    if stage == STAGE_LONG:
        return float(LONG_PERIOD_MIN_DAYS)
    raise ValueError(f"Unknown stage: {stage!r}")


def adaptive_period_bounds(
    *,
    baseline_days: float | None,
    stage: str,
    cadence_median_days: float | None = None,
    n_points: int | None = None,
    user_min_period: float | None = None,
    user_max_period: float | None = None,
) -> PeriodBounds:
    """Return ``PeriodBounds`` scaled to the candidate's baseline and cadence.

    Parameters
    ----------
    baseline_days:
        Time span ``JD_max - JD_min`` for the light curve. If ``None`` or
        non-finite, the helper returns the classic stage constants.
    stage:
        One of ``"pregate"``, ``"postfilter"``, ``"review"``, ``"long"``.
    cadence_median_days:
        Median inter-observation spacing. Used to raise ``min_period`` so we
        never search below the Nyquist-ish limit (``PERIOD_BOUNDS_MIN_CADENCE_MULTIPLIER``).
    n_points:
        Number of finite observations. Used to bail out to fallback if too small.
    user_min_period, user_max_period:
        Explicit overrides (e.g. from the review UI). If both provided, the
        helper honors them and only clamps to be strictly positive with min<max.
    """
    if stage not in _KNOWN_STAGES:
        raise ValueError(f"Unknown stage: {stage!r}")

    # Feature flag: when adaptive bounds are disabled, always return the classic
    # stage constants (unless the caller supplied an explicit user override).
    if not ADAPTIVE_BOUNDS_ENABLED and user_min_period is None and user_max_period is None:
        fb_min, fb_max, reason = _fallback_bounds(stage)
        return PeriodBounds(
            min_period_days=fb_min,
            max_period_days=fb_max,
            stage=stage,
            baseline_days=_finite(baseline_days),
            cadence_median_days=_finite(cadence_median_days),
            n_points=int(n_points) if n_points is not None else None,
            reason=f"disabled:{reason}",
        )

    if user_min_period is not None or user_max_period is not None:
        floor_min = _stage_min_floor(stage)
        floor_max = _stage_absolute_max(stage)
        u_min = _finite(user_min_period) or floor_min
        u_max = _finite(user_max_period) or floor_max
        u_min = max(u_min, 1e-6)
        if u_max <= u_min:
            u_max = u_min * 10.0
        return PeriodBounds(
            min_period_days=float(u_min),
            max_period_days=float(u_max),
            stage=stage,
            baseline_days=_finite(baseline_days),
            cadence_median_days=_finite(cadence_median_days),
            n_points=int(n_points) if n_points is not None else None,
            reason="user_override",
        )

    b = _finite(baseline_days)
    if b is None or b <= 0:
        fb_min, fb_max, reason = _fallback_bounds(stage)
        return PeriodBounds(
            min_period_days=fb_min,
            max_period_days=fb_max,
            stage=stage,
            baseline_days=_finite(baseline_days),
            cadence_median_days=_finite(cadence_median_days),
            n_points=int(n_points) if n_points is not None else None,
            reason=reason,
        )

    if stage == STAGE_LONG:
        max_from_baseline = float(LONG_PERIOD_BASELINE_FRACTION) * b
    else:
        max_from_baseline = float(PERIOD_BOUNDS_MAX_BASELINE_FRACTION) * b

    absolute_cap = _stage_absolute_max(stage)
    max_period = min(max_from_baseline, absolute_cap)

    min_floor = _stage_min_floor(stage)
    cadence = _finite(cadence_median_days)
    if cadence is not None and cadence > 0:
        min_from_cadence = float(PERIOD_BOUNDS_MIN_CADENCE_MULTIPLIER) * cadence
        min_period = max(min_floor, min_from_cadence)
    else:
        min_period = min_floor

    if stage == STAGE_LONG:
        min_period = max(min_period, float(LONG_PERIOD_MIN_DAYS))

    if max_period <= min_period:
        max_period = min_period * 10.0

    reason_parts = [f"stage={stage}", f"baseline={b:.1f}d"]
    if cadence is not None and cadence > 0:
        reason_parts.append(f"cadence={cadence:.3f}d")
    if n_points is not None:
        reason_parts.append(f"n_points={int(n_points)}")
    reason = "adaptive:" + ",".join(reason_parts)

    return PeriodBounds(
        min_period_days=float(min_period),
        max_period_days=float(max_period),
        stage=stage,
        baseline_days=b,
        cadence_median_days=cadence,
        n_points=int(n_points) if n_points is not None else None,
        reason=reason,
    )


def bounds_from_jd(
    jd: Iterable[float],
    *,
    stage: str,
    user_min_period: float | None = None,
    user_max_period: float | None = None,
) -> PeriodBounds:
    """Convenience wrapper: derive baseline/cadence/n_points from a JD array."""
    arr = np.asarray(list(jd), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return adaptive_period_bounds(
            baseline_days=None,
            stage=stage,
            cadence_median_days=None,
            n_points=int(arr.size),
            user_min_period=user_min_period,
            user_max_period=user_max_period,
        )
    arr_sorted = np.sort(arr)
    baseline = float(arr_sorted[-1] - arr_sorted[0])
    diffs = np.diff(arr_sorted)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    cadence = float(np.median(diffs)) if diffs.size else None
    return adaptive_period_bounds(
        baseline_days=baseline,
        stage=stage,
        cadence_median_days=cadence,
        n_points=int(arr.size),
        user_min_period=user_min_period,
        user_max_period=user_max_period,
    )


__all__ = [
    "STAGE_LONG",
    "STAGE_POSTFILTER",
    "STAGE_PREGATE",
    "STAGE_REVIEW",
    "PeriodBounds",
    "adaptive_period_bounds",
    "bounds_from_jd",
]
