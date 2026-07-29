"""Tests for the dip-run epoch helper (persistence + lightweight detection)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from malca.core.event_epochs import (
    DIP_RUN_EPOCHS_SCHEMA_VERSION,
    DipRunEpoch,
    detect_dip_epochs_lightweight,
    dip_center_jds,
    parse_run_epochs_json,
    serialize_run_summaries,
)


# ---------------------------------------------------------------------------
# serialize / parse
# ---------------------------------------------------------------------------

def test_serialize_empty_summaries_returns_none() -> None:
    assert serialize_run_summaries(None) is None
    assert serialize_run_summaries([]) is None
    assert serialize_run_summaries([{"kept": False}], kept_only=True) is None


def test_serialize_and_parse_round_trip() -> None:
    summaries = [
        {"start_jd": 100.0, "end_jd": 120.0, "run_max": 5.5, "n_points": 3, "duration_days": 20.0, "kept": True},
        {"start_jd": 500.0, "end_jd": 510.0, "run_max": 4.2, "n_points": 2, "duration_days": 10.0, "kept": True},
        {"start_jd": 800.0, "end_jd": 802.0, "run_max": 3.0, "n_points": 2, "kept": False},
    ]
    blob = serialize_run_summaries(summaries)
    assert blob is not None
    payload = json.loads(blob)
    assert payload["schema"] == DIP_RUN_EPOCHS_SCHEMA_VERSION

    epochs = parse_run_epochs_json(blob)
    assert [e.start_jd for e in epochs] == [100.0, 500.0]
    assert [e.center_jd for e in epochs] == [110.0, 505.0]
    assert epochs[0].peak_significance == pytest.approx(5.5)
    assert epochs[1].n_points == 2


def test_parse_handles_bad_input() -> None:
    assert parse_run_epochs_json(None) == []
    assert parse_run_epochs_json("") == []
    assert parse_run_epochs_json("not json") == []
    assert parse_run_epochs_json("[]") == []
    assert parse_run_epochs_json("[{}]") == []


def test_parse_accepts_bare_list_payload() -> None:
    """Backfill entries without the schema wrapper still parse."""
    epochs = parse_run_epochs_json(
        json.dumps([{"start_jd": 10.0, "end_jd": 20.0, "center_jd": 15.0}])
    )
    assert len(epochs) == 1
    assert epochs[0].center_jd == pytest.approx(15.0)


def test_dip_center_jds_are_sorted_and_finite() -> None:
    epochs = [
        DipRunEpoch(start_jd=100.0, end_jd=200.0, center_jd=150.0, peak_significance=1, n_points=2, duration_days=100),
        DipRunEpoch(start_jd=50.0, end_jd=60.0, center_jd=55.0, peak_significance=1, n_points=2, duration_days=10),
        DipRunEpoch(start_jd=300.0, end_jd=310.0, center_jd=float("nan"), peak_significance=1, n_points=2, duration_days=10),
    ]
    assert dip_center_jds(epochs) == [55.0, 150.0]


# ---------------------------------------------------------------------------
# detect_dip_epochs_lightweight
# ---------------------------------------------------------------------------

def _make_light_curve_with_two_dips(
    *,
    baseline_days: float = 3653.0,
    n_points: int = 1200,
    dip_centers: tuple[float, ...] = (1000.0, 3011.0),
    dip_width_days: float = 25.0,
    dip_depth_mag: float = 0.7,
    noise_mag: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(0.0, baseline_days, size=n_points))
    mag = np.full(jd.size, 15.0)
    for center in dip_centers:
        mag += dip_depth_mag * np.exp(-0.5 * ((jd - center) / (dip_width_days / 2.355)) ** 2)
    mag += rng.normal(0.0, noise_mag, size=jd.size)
    err = np.full_like(mag, noise_mag)
    return jd, mag, err


def test_lightweight_detects_both_dips() -> None:
    jd, mag, err = _make_light_curve_with_two_dips()
    epochs = detect_dip_epochs_lightweight(jd, mag, err)
    centers = [e.center_jd for e in epochs]
    assert len(epochs) >= 2
    # Expect both dips within ~50 d of their true centers
    assert any(abs(c - 1000.0) < 50.0 for c in centers)
    assert any(abs(c - 3011.0) < 50.0 for c in centers)


def test_lightweight_returns_empty_for_pure_noise() -> None:
    rng = np.random.default_rng(0)
    jd = np.sort(rng.uniform(0.0, 3000.0, size=1000))
    mag = rng.normal(15.0, 0.02, size=jd.size)
    err = np.full_like(mag, 0.02)
    epochs = detect_dip_epochs_lightweight(jd, mag, err, n_sigma=5.0, min_depth_mag=0.2)
    assert epochs == []


def test_lightweight_handles_insufficient_points() -> None:
    jd = np.array([0.0, 1.0, 2.0])
    mag = np.array([15.0, 15.0, 15.0])
    err = np.array([0.02, 0.02, 0.02])
    assert detect_dip_epochs_lightweight(jd, mag, err) == []


def test_lightweight_reports_reasonable_depth() -> None:
    """Detected runs must have a peak significance above the trigger threshold."""
    jd, mag, err = _make_light_curve_with_two_dips(dip_depth_mag=0.5, noise_mag=0.02)
    epochs = detect_dip_epochs_lightweight(jd, mag, err, n_sigma=4.0)
    assert all(np.isfinite(e.peak_significance) for e in epochs)
    assert all(e.peak_significance >= 4.0 for e in epochs)


def test_lightweight_detects_2000_day_recurrence() -> None:
    """Regression test that motivated this whole module."""
    jd, mag, err = _make_light_curve_with_two_dips(
        baseline_days=3653.0,
        dip_centers=(1000.0, 3011.0),  # Δt = 2011 d
        n_points=1500,
    )
    epochs = detect_dip_epochs_lightweight(jd, mag, err)
    centers = sorted(e.center_jd for e in epochs)
    assert len(centers) >= 2
    # Δt between the two brightest events must be near 2011 d
    deltas = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    assert any(abs(d - 2011.0) < 60.0 for d in deltas)
