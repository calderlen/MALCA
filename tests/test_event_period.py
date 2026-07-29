"""Tests for event-based period estimation."""
from __future__ import annotations

import numpy as np
import pytest

from malca.stv.event_period import event_based_period, refine_event_period


def test_two_events_median_dt() -> None:
    result = event_based_period([1000.0, 3011.0], baseline_days=4000.0)
    assert result["event_period_method"] == "median_dt"
    assert result["event_period_days"] == pytest.approx(2011.0)
    assert result["event_period_n_events"] == 2
    assert result["event_period_is_high_confidence"] is False


def test_three_events_gcd_high_confidence() -> None:
    # Missed one cycle between first and second: Δt = 2P, P, P
    epochs = [0.0, 4000.0, 6000.0, 8000.0]
    result = event_based_period(epochs, baseline_days=9000.0)
    assert result["event_period_method"] == "gcd_dt"
    assert result["event_period_days"] == pytest.approx(2000.0, rel=0.05)
    assert result["event_period_n_events"] == 4
    assert result["event_period_is_high_confidence"] is True


def test_irregular_event_spacings_do_not_collapse_gcd_to_machine_epsilon() -> None:
    epochs = [
        2458175.58018,
        2458806.77861,
        2459349.52179,
        2459353.52384,
        2459548.73135,
        2459962.78874,
        2460586.54877,
        2461039.60187,
    ]

    result = event_based_period(epochs, baseline_days=3100.0)

    assert result["event_period_method"] == "median_dt"
    assert result["event_period_days"] == pytest.approx(
        np.median(np.diff(np.asarray(epochs)))
    )
    assert result["event_period_days"] > 0.1
    assert result["event_period_is_high_confidence"] is False


def test_single_event_prior() -> None:
    result = event_based_period([2458000.0], baseline_days=3653.0)
    assert result["event_period_method"] == "single_event_prior"
    assert result["event_period_days"] == pytest.approx(3653.0 / 2.0)
    assert result["event_period_status"] == "prior_only"


def test_refine_event_period_around_injected_signal() -> None:
    rng = np.random.default_rng(0)
    true_p = 12.5
    jd = np.sort(rng.uniform(0.0, 200.0, size=300))
    mag = 0.2 * np.sin(2.0 * np.pi * jd / true_p) + rng.normal(0.0, 0.02, size=jd.size)
    err = np.full_like(mag, 0.02)
    out = refine_event_period(jd, mag, err, true_p, fraction=0.2, method="ce")
    assert out["refine_status"] == "ok"
    assert abs(out["refined_period_days"] - true_p) / true_p < 0.05
