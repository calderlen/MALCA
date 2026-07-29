from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from malca.stv.dimming_window import (
    DEFAULT_DIMMING_WINDOW_CONFIG,
    DIMMING_WINDOW_METHOD_VERSION,
    DimmingWindowConfig,
    dimming_complex_zoom_bounds,
    select_dimming_complex_window,
)


def _synthetic_event(*, ongoing_right: bool = False):
    times = np.arange(41, dtype=float) + 0.2
    residual = np.zeros_like(times)
    residual[10:14] = 0.04
    residual[14 : (len(times) if ongoing_right else 27)] = 0.12
    if not ongoing_right:
        residual[27:31] = 0.04
    sigma = np.full_like(times, 0.008)
    return times, residual, sigma


def test_dimming_window_thresholds_are_explicit_and_immutable() -> None:
    config = DEFAULT_DIMMING_WINDOW_CONFIG

    assert isinstance(config, DimmingWindowConfig)
    assert config.recovery_window_sizes == (5, 4, 3)
    assert config.recovery_min_compatible_epochs == 3
    assert config.peak_seed_required_strong_epochs == 2
    with pytest.raises(FrozenInstanceError):
        config.recovery_min_compatible_epochs = 2  # type: ignore[misc]
    assert replace(config, recovery_min_compatible_epochs=4).recovery_min_compatible_epochs == 4


def test_select_completed_dimming_complex_returns_structured_boundaries() -> None:
    times, residual, sigma = _synthetic_event()

    window, diagnostics = select_dimming_complex_window(times, residual, sigma)

    assert window.status == "baseline_bounded"
    assert window.censoring_status == "recovery_bounded"
    assert window.is_lower_limit is False
    assert window.left_boundary_type == "recovery"
    assert window.right_boundary_type == "recovery"
    assert window.start_jd < times[10]
    assert window.end_jd > times[30]
    assert window.peak_depth_mag == pytest.approx(0.12)
    assert diagnostics["recovery_mask"].dtype == np.bool_
    assert diagnostics["event_polarity"] == "dimming"

    metrics = window.to_metrics(times)
    assert metrics["dimming_window_method_version"] == DIMMING_WINDOW_METHOD_VERSION
    assert metrics["dimming_complex_duration_lower_days"] == pytest.approx(
        window.duration_days
    )
    assert metrics["dimming_complex_duration_upper_days"] == pytest.approx(
        window.duration_days
    )


def test_select_ongoing_dimming_complex_is_a_right_censored_lower_limit() -> None:
    times, residual, sigma = _synthetic_event(ongoing_right=True)

    window, _ = select_dimming_complex_window(times, residual, sigma)

    assert window.status == "ongoing_right_censored"
    assert window.censoring_status == "right_censored"
    assert window.is_lower_limit is True
    assert window.left_boundary_type == "recovery"
    assert window.right_boundary_type == "data_edge"
    assert window.end_jd == pytest.approx(times[-1])
    assert np.isnan(window.to_metrics(times)["dimming_complex_duration_upper_days"])


def test_select_brightening_complex_uses_negated_residuals() -> None:
    times, dimming_residual, sigma = _synthetic_event()
    dimming_window, _ = select_dimming_complex_window(
        times,
        dimming_residual,
        sigma,
    )

    window, diagnostics = select_dimming_complex_window(
        times,
        -dimming_residual,
        sigma,
        polarity="brightening",
    )

    assert window.status == "baseline_bounded"
    assert window.peak_depth_mag == pytest.approx(0.12)
    assert window.peak_jd == pytest.approx(dimming_window.peak_jd)
    assert window.start_jd == pytest.approx(dimming_window.start_jd)
    assert window.end_jd == pytest.approx(dimming_window.end_jd)
    assert diagnostics["event_polarity"] == "brightening"


def test_select_rejects_unknown_event_polarity() -> None:
    times, residual, sigma = _synthetic_event()

    with pytest.raises(ValueError, match="unknown event polarity"):
        select_dimming_complex_window(
            times,
            residual,
            sigma,
            polarity="sideways",
        )


def test_select_rejects_an_unanchored_all_dim_light_curve() -> None:
    times = np.arange(30, dtype=float)
    residual = np.full_like(times, 0.12)
    sigma = np.full_like(times, 0.008)

    with pytest.raises(RuntimeError, match="no recovery-anchored"):
        select_dimming_complex_window(times, residual, sigma)


def test_dimming_complex_zoom_bounds_match_atlas_padding_rule() -> None:
    times = np.arange(100.0, 121.0)

    zoom_start, zoom_end = dimming_complex_zoom_bounds(
        times,
        start_jd=105.0,
        end_jd=115.0,
        peak_jd=110.0,
        cadence_days=1.0,
    )

    # The selected complex gets a two-cadence margin, clipped only by data.
    assert (zoom_start, zoom_end) == pytest.approx((103.0, 117.0))
