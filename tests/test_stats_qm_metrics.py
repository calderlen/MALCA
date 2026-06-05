from __future__ import annotations

import numpy as np

from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.stats_merge import merge_stats_summary_into_payload
from malca.review.store import _COL_NAMES
from malca.stats import flux_asymmetry_metric, quasi_periodicity_metric


def test_flux_asymmetry_metric_is_near_zero_for_symmetric_variability() -> None:
    mag = np.linspace(-1.0, 1.0, 100)

    assert abs(flux_asymmetry_metric(mag)) < 1e-12


def test_flux_asymmetry_metric_tracks_dips_and_bursts_in_magnitudes() -> None:
    dipper = np.concatenate([np.zeros(90), np.ones(10)])
    burster = np.concatenate([np.zeros(90), -np.ones(10)])

    assert flux_asymmetry_metric(dipper) > 0.25
    assert flux_asymmetry_metric(burster) < -0.25


def test_flux_asymmetry_metric_requires_scatter_and_enough_points() -> None:
    assert np.isnan(flux_asymmetry_metric(np.ones(10)))
    assert np.isnan(flux_asymmetry_metric(np.arange(9.0)))


def test_quasi_periodicity_metric_is_small_for_clean_periodic_signal() -> None:
    rng = np.random.default_rng(123)
    period = 2.75
    time = np.sort(rng.uniform(0.0, 80.0, 240))
    err = np.full_like(time, 0.02)
    mag = 13.0 + 0.35 * np.sin(2.0 * np.pi * time / period) + rng.normal(0.0, 0.005, size=time.size)

    q_metric = quasi_periodicity_metric(mag, time, err, period)

    assert np.isfinite(q_metric)
    assert q_metric < 0.05


def test_quasi_periodicity_metric_is_large_for_aperiodic_signal() -> None:
    rng = np.random.default_rng(456)
    time = np.sort(rng.uniform(0.0, 80.0, 240))
    err = np.full_like(time, 0.02)
    mag = 13.0 + rng.normal(0.0, 0.35, size=time.size)

    q_metric = quasi_periodicity_metric(mag, time, err, 2.75)

    assert np.isfinite(q_metric)
    assert q_metric > 0.5


def test_quasi_periodicity_metric_returns_nan_for_invalid_period() -> None:
    time = np.arange(20.0)
    mag = np.sin(time)
    err = np.full_like(time, 0.02)

    assert np.isnan(quasi_periodicity_metric(mag, time, err, np.nan))
    assert np.isnan(quasi_periodicity_metric(mag, time, err, 0.0))


def test_qm_stats_merge_and_review_schema_entries() -> None:
    payload: dict[str, float] = {}
    merge_stats_summary_into_payload(
        payload,
        {
            "variability_quasi_periodicity_q": 0.12,
            "variability_flux_asymmetry_m": 0.34,
        },
    )

    assert payload["stats_variability_quasi_periodicity_q"] == 0.12
    assert payload["stats_variability_flux_asymmetry_m"] == 0.34
    assert "stats_variability_quasi_periodicity_q" in _COL_NAMES
    assert "stats_variability_flux_asymmetry_m" in _COL_NAMES

    filter_columns = {entry[1] for _group, entries in SIDEBAR_GROUPS for entry in entries}
    assert "stats_variability_quasi_periodicity_q" in filter_columns
    assert "stats_variability_flux_asymmetry_m" in filter_columns
