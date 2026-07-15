from __future__ import annotations

import numpy as np
import pytest

from malca.core.period_arbitration import (
    NATIVE_PERIOD_DOWNWARD_HARMONIC_FACTORS,
    NATIVE_PERIOD_HARMONIC_FACTORS,
    NATIVE_PERIOD_WITH_MULTIPLES_FACTORS,
    choose_native_harmonic_candidate,
    native_harmonic_period_candidates,
    period_alias_matches,
)


def test_native_harmonic_candidates_are_downward_only() -> None:
    candidates = native_harmonic_period_candidates(8.0, min_period=0.1, max_period=20.0)

    assert NATIVE_PERIOD_HARMONIC_FACTORS == NATIVE_PERIOD_DOWNWARD_HARMONIC_FACTORS
    assert [candidate["period"] for candidate in candidates] == pytest.approx([8.0, 4.0, 8.0 / 3.0, 2.0])
    assert all(float(candidate["period"]) <= 8.0 for candidate in candidates)


def test_native_harmonic_candidates_can_include_period_multiples() -> None:
    candidates = native_harmonic_period_candidates(
        8.0,
        min_period=0.1,
        max_period=40.0,
        harmonic_factors=NATIVE_PERIOD_WITH_MULTIPLES_FACTORS,
    )

    assert [candidate["period"] for candidate in candidates] == pytest.approx(
        [8.0, 4.0, 8.0 / 3.0, 2.0, 16.0, 24.0, 32.0]
    )
    assert [candidate["upward_multiple_flag"] for candidate in candidates] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_native_harmonic_candidate_flags_known_aliases() -> None:
    candidates = native_harmonic_period_candidates(2.0, min_period=0.1, max_period=5.0)
    one_day = next(candidate for candidate in candidates if np.isclose(float(candidate["period"]), 1.0))

    assert one_day["alias_flag"] is True
    assert period_alias_matches(1.0)


def test_choose_native_harmonic_keeps_base_without_material_gain() -> None:
    candidates = [
        {"period": 8.0, "factor": 1.0, "selection_objective": 1.0},
        {"period": 4.0, "factor": 0.5, "selection_objective": 0.99},
    ]

    selected = choose_native_harmonic_candidate(candidates, min_rel_improvement=0.02)

    assert selected is candidates[0]


def test_choose_native_harmonic_accepts_shorter_material_gain() -> None:
    candidates = [
        {"period": 8.0, "factor": 1.0, "selection_objective": 1.0},
        {"period": 4.0, "factor": 0.5, "selection_objective": 0.80},
    ]

    selected = choose_native_harmonic_candidate(candidates, min_rel_improvement=0.02)

    assert selected is candidates[1]


def test_choose_native_harmonic_rejects_small_upward_gain() -> None:
    candidates = [
        {"period": 8.0, "factor": 1.0, "selection_objective": 1.0},
        {"period": 16.0, "factor": 2.0, "selection_objective": 0.95},
    ]

    selected = choose_native_harmonic_candidate(
        candidates,
        min_rel_improvement=0.02,
        upward_min_rel_improvement=0.08,
    )

    assert selected is candidates[0]


def test_choose_native_harmonic_accepts_large_upward_gain() -> None:
    candidates = [
        {"period": 8.0, "factor": 1.0, "selection_objective": 1.0},
        {"period": 16.0, "factor": 2.0, "selection_objective": 0.88},
    ]

    selected = choose_native_harmonic_candidate(
        candidates,
        min_rel_improvement=0.02,
        upward_min_rel_improvement=0.08,
    )

    assert selected is candidates[1]
    assert selected["upward_multiple_flag"] is True
    assert selected["required_rel_improvement"] == pytest.approx(0.08)
