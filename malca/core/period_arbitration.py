"""Shared helpers for native period harmonic arbitration."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from malca.config import LS_ALIAS_PERIODS, LS_ALIAS_TOLERANCE


NATIVE_PERIOD_DOWNWARD_HARMONIC_FACTORS: tuple[float, ...] = (
    1.0,
    0.5,
    1.0 / 3.0,
    0.25,
)
NATIVE_PERIOD_WITH_MULTIPLES_FACTORS: tuple[float, ...] = (
    1.0,
    0.5,
    1.0 / 3.0,
    0.25,
    2.0,
    3.0,
    4.0,
)
NATIVE_PERIOD_HARMONIC_FACTORS: tuple[float, ...] = NATIVE_PERIOD_DOWNWARD_HARMONIC_FACTORS
NATIVE_PERIOD_MIN_REL_IMPROVEMENT = 0.02
NATIVE_PERIOD_UPWARD_MIN_REL_IMPROVEMENT = 0.08


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def period_alias_matches(
    period: object,
    *,
    alias_periods: Iterable[float] = LS_ALIAS_PERIODS,
    alias_tolerance: float = LS_ALIAS_TOLERANCE,
) -> list[float]:
    period_value = finite_float(period)
    if period_value is None or period_value <= 0:
        return []
    return [
        float(alias_period)
        for alias_period in alias_periods
        if np.isfinite(alias_period)
        and abs(float(period_value) - float(alias_period)) <= float(alias_tolerance)
    ]


def native_harmonic_period_candidates(
    base_period: object,
    *,
    min_period: float,
    max_period: float,
    harmonic_factors: tuple[float, ...] = NATIVE_PERIOD_HARMONIC_FACTORS,
) -> list[dict[str, object]]:
    base = finite_float(base_period)
    min_p = finite_float(min_period)
    max_p = finite_float(max_period)
    if base is None or base <= 0 or min_p is None or max_p is None:
        return []
    if max_p < min_p:
        min_p, max_p = max_p, min_p

    candidates: list[dict[str, object]] = []
    for factor_raw in harmonic_factors:
        factor = finite_float(factor_raw)
        if factor is None or factor <= 0:
            continue
        period = float(base) * float(factor)
        if period <= 0 or period < min_p or period > max_p:
            continue
        if any(
            abs(period - float(candidate["period"]))
            <= 1e-10 * max(1.0, abs(period), abs(float(candidate["period"])))
            for candidate in candidates
        ):
            continue
        aliases = period_alias_matches(period)
        candidates.append(
            {
                "factor": float(factor),
                "divisor": float(1.0 / factor),
                "period": float(period),
                "upward_multiple_flag": bool(float(factor) > 1.0),
                "alias_flag": bool(aliases),
                "alias_matches": aliases,
            }
        )
    return candidates


def choose_native_harmonic_candidate(
    candidates: list[dict[str, object]],
    *,
    objective_key: str = "selection_objective",
    min_rel_improvement: float = NATIVE_PERIOD_MIN_REL_IMPROVEMENT,
    upward_min_rel_improvement: float = NATIVE_PERIOD_UPWARD_MIN_REL_IMPROVEMENT,
    base_factor: float = 1.0,
    sort_key: Callable[[dict[str, object]], object] | None = None,
) -> dict[str, object] | None:
    finite = [
        candidate
        for candidate in candidates
        if finite_float(candidate.get("period")) is not None
        and finite_float(candidate.get(objective_key, candidate.get("objective"))) is not None
    ]
    if not finite:
        return None

    if sort_key is None:
        selected = min(
            finite,
            key=lambda candidate: float(candidate.get(objective_key, candidate.get("objective", np.inf))),
        )
    else:
        selected = min(finite, key=sort_key)

    base = next(
        (
            candidate
            for candidate in finite
            if abs(float(candidate.get("factor", np.nan)) - float(base_factor)) < 1e-12
        ),
        None,
    )
    if base is None or selected is base:
        return selected

    base_obj = finite_float(base.get(objective_key, base.get("objective")))
    selected_obj = finite_float(selected.get(objective_key, selected.get("objective")))
    if base_obj is None or selected_obj is None:
        return selected
    improvement = (float(base_obj) - float(selected_obj)) / max(abs(float(base_obj)), 1e-9)
    selected_factor = finite_float(selected.get("factor"))
    selected_is_upward = (
        selected_factor is not None
        and selected_factor > float(base_factor)
        and abs(selected_factor - float(base_factor)) > 1e-12
    )
    required_improvement = (
        float(upward_min_rel_improvement)
        if selected_is_upward
        else float(min_rel_improvement)
    )
    selected["rel_improvement_vs_base"] = float(improvement)
    selected["required_rel_improvement"] = required_improvement
    selected["upward_multiple_flag"] = bool(selected_is_upward)
    if not np.isfinite(improvement) or improvement < required_improvement:
        base["rel_improvement_vs_base"] = 0.0
        base["required_rel_improvement"] = required_improvement
        return base
    return selected
