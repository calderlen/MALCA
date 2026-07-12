from __future__ import annotations

import numpy as np
import pytest

from malca.core.period_arbitration import (
    NATIVE_PERIOD_HARMONIC_FACTORS,
    choose_native_harmonic_candidate,
    native_harmonic_period_candidates,
    period_alias_matches,
)


def test_native_harmonic_candidates_are_downward_only() -> None:
    candidates = native_harmonic_period_candidates(8.0, min_period=0.1, max_period=20.0)

    assert NATIVE_PERIOD_HARMONIC_FACTORS == (1.0, 0.5, 1.0 / 3.0, 0.25)
    assert [candidate["period"] for candidate in candidates] == pytest.approx([8.0, 4.0, 8.0 / 3.0, 2.0])
    assert all(float(candidate["period"]) <= 8.0 for candidate in candidates)


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
