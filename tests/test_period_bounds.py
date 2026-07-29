"""Tests for baseline-adaptive period bounds."""
from __future__ import annotations

import numpy as np
import pytest

from malca.config import (
    LONG_PERIOD_ABSOLUTE_CAP_DAYS,
    LONG_PERIOD_BASELINE_FRACTION,
    LONG_PERIOD_MIN_DAYS,
    PDM_MAX_PERIOD,
    PDM_MIN_PERIOD,
    PERIOD_BOUNDS_MAX_BASELINE_FRACTION,
    PERIOD_BOUNDS_MIN_CADENCE_MULTIPLIER,
    PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_PERIOD,
    REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS,
    REVIEW_PERIOD_MIN_DAYS,
)
from malca.core.period_bounds import (
    STAGE_LONG,
    STAGE_POSTFILTER,
    STAGE_PREGATE,
    STAGE_REVIEW,
    PeriodBounds,
    adaptive_period_bounds,
    bounds_from_jd,
)


def test_unknown_stage_raises() -> None:
    with pytest.raises(ValueError):
        adaptive_period_bounds(baseline_days=1000, stage="bogus")


def test_missing_baseline_falls_back_to_stage_constants() -> None:
    for stage, min_expected, max_expected in [
        (STAGE_PREGATE, PRE_PERIODICITY_MIN_PERIOD, PRE_PERIODICITY_MAX_PERIOD),
        (STAGE_POSTFILTER, PDM_MIN_PERIOD, PDM_MAX_PERIOD),
        (STAGE_REVIEW, REVIEW_PERIOD_MIN_DAYS, REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS),
        (STAGE_LONG, LONG_PERIOD_MIN_DAYS, LONG_PERIOD_ABSOLUTE_CAP_DAYS),
    ]:
        bounds = adaptive_period_bounds(baseline_days=None, stage=stage)
        assert bounds.min_period_days == pytest.approx(min_expected)
        assert bounds.max_period_days == pytest.approx(max_expected)
        assert bounds.reason.startswith("fallback:")

    nan_bounds = adaptive_period_bounds(baseline_days=float("nan"), stage=STAGE_POSTFILTER)
    assert nan_bounds.min_period_days == pytest.approx(PDM_MIN_PERIOD)
    assert nan_bounds.max_period_days == pytest.approx(PDM_MAX_PERIOD)


def test_max_period_scales_with_baseline_for_short_period_stages() -> None:
    baseline = 3653.0
    for stage, absolute_cap in [
        (STAGE_PREGATE, PRE_PERIODICITY_MAX_PERIOD),
        (STAGE_POSTFILTER, PDM_MAX_PERIOD),
        (STAGE_REVIEW, REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS),
    ]:
        bounds = adaptive_period_bounds(baseline_days=baseline, stage=stage)
        expected = min(PERIOD_BOUNDS_MAX_BASELINE_FRACTION * baseline, absolute_cap)
        assert bounds.max_period_days == pytest.approx(expected)
        assert bounds.reason.startswith("adaptive:")


def test_long_stage_uses_larger_baseline_fraction() -> None:
    baseline = 3653.0
    bounds = adaptive_period_bounds(baseline_days=baseline, stage=STAGE_LONG)
    expected_max = min(LONG_PERIOD_BASELINE_FRACTION * baseline, LONG_PERIOD_ABSOLUTE_CAP_DAYS)
    assert bounds.max_period_days == pytest.approx(expected_max)
    assert bounds.min_period_days >= LONG_PERIOD_MIN_DAYS


def test_long_stage_covers_target_2011_day_period() -> None:
    """Regression test for the AA-Tau-like candidate that motivated this module."""
    bounds = adaptive_period_bounds(baseline_days=3653.0, stage=STAGE_LONG)
    assert bounds.min_period_days < 2011.0 < bounds.max_period_days


def test_short_baseline_still_produces_valid_bounds() -> None:
    """Very short baselines must not collapse min>=max."""
    bounds = adaptive_period_bounds(baseline_days=0.5, stage=STAGE_POSTFILTER)
    assert bounds.min_period_days > 0
    assert bounds.max_period_days > bounds.min_period_days


def test_cadence_raises_min_period_floor() -> None:
    cadence = 30.0
    bounds = adaptive_period_bounds(
        baseline_days=5000.0,
        stage=STAGE_POSTFILTER,
        cadence_median_days=cadence,
    )
    assert bounds.min_period_days == pytest.approx(PERIOD_BOUNDS_MIN_CADENCE_MULTIPLIER * cadence)


def test_cadence_never_lowers_min_below_stage_floor() -> None:
    """Sub-cadence min is bounded by the stage's absolute floor."""
    bounds = adaptive_period_bounds(
        baseline_days=1000.0,
        stage=STAGE_POSTFILTER,
        cadence_median_days=0.001,
    )
    assert bounds.min_period_days >= PDM_MIN_PERIOD


def test_absolute_cap_is_respected_when_baseline_is_huge() -> None:
    bounds = adaptive_period_bounds(baseline_days=1_000_000.0, stage=STAGE_LONG)
    assert bounds.max_period_days == pytest.approx(LONG_PERIOD_ABSOLUTE_CAP_DAYS)


def test_user_override_wins_and_reports_reason() -> None:
    bounds = adaptive_period_bounds(
        baseline_days=1000.0,
        stage=STAGE_REVIEW,
        user_min_period=1.5,
        user_max_period=2500.0,
    )
    assert bounds.min_period_days == pytest.approx(1.5)
    assert bounds.max_period_days == pytest.approx(2500.0)
    assert bounds.reason == "user_override"


def test_user_override_only_min_falls_back_to_stage_cap() -> None:
    bounds = adaptive_period_bounds(
        baseline_days=None,
        stage=STAGE_REVIEW,
        user_min_period=5.0,
    )
    assert bounds.min_period_days == pytest.approx(5.0)
    assert bounds.max_period_days == pytest.approx(REVIEW_PERIOD_MAX_ABSOLUTE_CAP_DAYS)


def test_user_override_min_greater_than_max_is_normalized() -> None:
    bounds = adaptive_period_bounds(
        baseline_days=1000.0,
        stage=STAGE_REVIEW,
        user_min_period=10.0,
        user_max_period=5.0,
    )
    assert bounds.min_period_days == pytest.approx(10.0)
    assert bounds.max_period_days > bounds.min_period_days


def test_bounds_from_jd_derives_baseline_and_cadence() -> None:
    rng = np.random.default_rng(0)
    jd = np.sort(rng.uniform(0.0, 2000.0, size=500))
    bounds = bounds_from_jd(jd, stage=STAGE_LONG)
    expected_baseline = float(jd[-1] - jd[0])
    expected_max = min(
        LONG_PERIOD_BASELINE_FRACTION * expected_baseline,
        LONG_PERIOD_ABSOLUTE_CAP_DAYS,
    )
    assert bounds.baseline_days == pytest.approx(expected_baseline)
    assert bounds.max_period_days == pytest.approx(expected_max)
    assert bounds.n_points == jd.size


def test_bounds_from_jd_handles_too_few_points() -> None:
    bounds = bounds_from_jd([1.0], stage=STAGE_POSTFILTER)
    assert bounds.reason.startswith("fallback:")


def test_bounds_from_jd_ignores_nonfinite_entries() -> None:
    jd = np.array([1.0, 2.0, float("nan"), 5.0, 10.0])
    bounds = bounds_from_jd(jd, stage=STAGE_POSTFILTER)
    assert bounds.baseline_days == pytest.approx(9.0)


def test_bounds_is_dataclass_and_returns_tuple() -> None:
    bounds = adaptive_period_bounds(baseline_days=1000.0, stage=STAGE_POSTFILTER)
    assert isinstance(bounds, PeriodBounds)
    assert bounds.as_tuple() == (bounds.min_period_days, bounds.max_period_days)


def test_adaptive_bounds_disabled_returns_legacy_postfilter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: flag off must match pre-adaptive post-filter bounds."""
    import malca.core.period_bounds as period_bounds_mod
    from malca.config import POST_FILTER_LEGACY_MAX_PERIOD, POST_FILTER_LEGACY_MIN_PERIOD

    monkeypatch.setattr(period_bounds_mod, "ADAPTIVE_BOUNDS_ENABLED", False)
    bounds = adaptive_period_bounds(baseline_days=3653.0, stage=STAGE_POSTFILTER)
    assert bounds.min_period_days == pytest.approx(POST_FILTER_LEGACY_MIN_PERIOD)
    assert bounds.max_period_days == pytest.approx(POST_FILTER_LEGACY_MAX_PERIOD)
    assert bounds.reason.startswith("disabled:")


def test_adaptive_bounds_disabled_returns_pregate_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    import malca.core.period_bounds as period_bounds_mod

    monkeypatch.setattr(period_bounds_mod, "ADAPTIVE_BOUNDS_ENABLED", False)
    bounds = adaptive_period_bounds(baseline_days=3653.0, stage=STAGE_PREGATE)
    assert bounds.min_period_days == pytest.approx(PRE_PERIODICITY_MIN_PERIOD)
    assert bounds.max_period_days == pytest.approx(PRE_PERIODICITY_MAX_PERIOD)
