from __future__ import annotations

import numpy as np
import pytest

from malca.ltv.core import (
    compute_bayesian_block_features,
    compute_binned_structure_function_features,
    compute_lowess_features,
    compute_rolling_smooth_features,
    compute_theil_sen_trend,
    compute_trend_metrics,
    compute_variogram_features,
)


def test_compute_trend_metrics_linear_trend_has_mag_scale_diff() -> None:
    indexes = [1.0, 2.0, 3.0]
    meds = [14.0, 14.1, 14.2]

    lin_slope, quad_slope, coeff1, coeff2, max_diff = compute_trend_metrics(indexes, meds)

    assert lin_slope == pytest.approx(0.1, abs=1e-10)
    assert quad_slope == pytest.approx(0.0, abs=1e-10)
    assert coeff1 == pytest.approx(0.1, abs=1e-10)
    assert coeff2 == pytest.approx(13.9, abs=1e-10)
    assert max_diff == pytest.approx(0.2, abs=1e-10)


def test_compute_trend_metrics_quadratic_uses_true_vertex_for_max_diff() -> None:
    indexes = [1.0, 2.0, 3.0]
    meds = [11.0, 10.0, 11.0]

    lin_slope, quad_slope, coeff1, coeff2, max_diff = compute_trend_metrics(indexes, meds)

    assert lin_slope == pytest.approx(0.0, abs=1e-10)
    assert quad_slope == pytest.approx(1.0, abs=1e-10)
    assert coeff1 == pytest.approx(-4.0, abs=1e-10)
    assert coeff2 == pytest.approx(14.0, abs=1e-10)
    assert max_diff == pytest.approx(1.0, abs=1e-10)


def test_rolling_smooth_amplitude_downweights_single_outlier() -> None:
    jd = np.linspace(0.0, 1000.0, 101)
    mag = 14.0 + 0.001 * jd
    mag[50] += 4.0

    features = compute_rolling_smooth_features(jd, mag)

    assert features["smooth_n_points"] == 101
    assert features["smooth_p95_p5"] < np.ptp(mag) / 2.0
    assert features["smooth_p95_p5"] == pytest.approx(0.8, abs=0.05)


def test_multi_window_rolling_smooth_returns_finite_features() -> None:
    jd = np.linspace(0.0, 3000.0, 301)
    mag = 14.0 + 0.0004 * jd + 0.1 * np.sin(2.0 * np.pi * jd / 400.0)

    features = compute_rolling_smooth_features(jd, mag)

    for label in ("100d", "300d", "1000d"):
        assert np.isfinite(features[f"smooth_{label}_p95_p5"])
        assert np.isfinite(features[f"smooth_{label}_smooth_var"])
        assert np.isfinite(features[f"smooth_{label}_resid_var"])
        assert np.isfinite(features[f"smooth_{label}_long_short_var_ratio"])
        assert features[f"smooth_{label}_n_points"] > 0


def test_rolling_smooth_windows_capture_different_timescales() -> None:
    jd = np.linspace(0.0, 2000.0, 401)
    mag = 14.0 + 0.001 * jd + 0.25 * np.sin(2.0 * np.pi * jd / 180.0)

    features = compute_rolling_smooth_features(jd, mag)

    assert abs(features["smooth_100d_p95_p5"] - features["smooth_1000d_p95_p5"]) > 0.05


def test_rolling_smooth_aliases_match_300_day_window() -> None:
    jd = np.linspace(0.0, 1000.0, 101)
    mag = 14.0 + 0.001 * jd

    features = compute_rolling_smooth_features(jd, mag)

    assert features["smooth_p95_p5"] == features["smooth_300d_p95_p5"]
    assert features["smooth_var"] == features["smooth_300d_smooth_var"]
    assert features["resid_var"] == features["smooth_300d_resid_var"]
    assert features["long_short_var_ratio"] == features["smooth_300d_long_short_var_ratio"]
    assert features["smooth_n_points"] == features["smooth_300d_n_points"]


def test_long_short_variance_ratio_separates_slow_trend_from_scatter() -> None:
    jd = np.linspace(0.0, 1000.0, 101)
    slow_mag = 14.0 + 0.001 * jd
    scatter_mag = 14.0 + 0.1 * np.where(np.arange(jd.size) % 2 == 0, 1.0, -1.0)

    slow = compute_rolling_smooth_features(jd, slow_mag)
    scatter = compute_rolling_smooth_features(jd, scatter_mag)

    assert slow["long_short_var_ratio"] > 10.0
    assert scatter["long_short_var_ratio"] < 1.0


def test_theil_sen_slope_is_robust_to_season_outlier() -> None:
    t_years = np.arange(8, dtype=float)
    meds = 14.0 + 0.2 * t_years
    meds[3] += 5.0

    features = compute_theil_sen_trend(t_years, meds)

    assert features["theil_sen_slope_mag_per_year"] == pytest.approx(0.2, abs=1e-10)


def test_bayesian_blocks_flags_clear_seasonal_step() -> None:
    t_years = np.arange(8, dtype=float)
    meds = np.array([14.00, 14.02, 13.99, 14.01, 15.00, 15.02, 14.98, 15.01])
    meds_err = np.full_like(meds, 0.02)

    features = compute_bayesian_block_features(t_years, meds, meds_err)

    assert features["bb_n_change_points"] >= 1
    assert features["bb_largest_jump_mag"] == pytest.approx(1.0, abs=0.05)


def test_lowess_features_are_finite_for_curved_seasonal_medians() -> None:
    t_years = np.arange(7, dtype=float)
    meds = 14.0 + 0.08 * (t_years - 3.0) ** 2

    features = compute_lowess_features(t_years, meds)

    assert np.isfinite(features["lowess_p95_p5"])
    assert np.isfinite(features["lowess_resid_std"])
    assert np.isfinite(features["lowess_max_abs_resid"])


def test_variogram_long_short_ratio_increases_for_slow_trend() -> None:
    t_years = np.arange(8, dtype=float)
    meds = 14.0 + 0.2 * t_years

    features = compute_variogram_features(t_years, meds)

    assert features["variogram_long_short_ratio"] > 10.0
    assert features["variogram_slope"] > 0.0


def test_binned_structure_function_long_lag_power_increases_for_slow_trend() -> None:
    jd = np.arange(0.0, 3600.0 + 30.0, 30.0)
    mag = 14.0 + 0.0002 * jd

    features = compute_binned_structure_function_features(jd, mag)

    assert features["binned_sf_n_bins"] == jd.size
    assert features["binned_sf_1000d_30d_ratio"] > 100.0
    assert features["binned_sf_3000d_30d_ratio"] > features["binned_sf_1000d_30d_ratio"]
    assert features["binned_sf_slope"] > 0.0


def test_binned_structure_function_returns_nan_ratios_without_short_lag_denominator() -> None:
    jd = np.array([0.0, 200.0, 400.0, 600.0, 800.0])
    mag = np.array([14.0, 14.1, 14.2, 14.3, 14.4])

    features = compute_binned_structure_function_features(jd, mag)

    assert np.isnan(features["binned_sf_30d_mag2"])
    assert np.isnan(features["binned_sf_300d_30d_ratio"])
    assert np.isnan(features["binned_sf_1000d_30d_ratio"])
    assert np.isnan(features["binned_sf_3000d_30d_ratio"])
