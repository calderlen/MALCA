from __future__ import annotations

import numpy as np
import pandas as pd

from malca.plotting.dipper_wavelet import (
    WWZConfig,
    analyze_wavelet,
    prepare_relative_flux,
    weighted_wavelet_z,
)


def test_prepare_relative_flux_normalizes_passbands_before_nightly_binning() -> None:
    lightcurve = pd.DataFrame(
        {
            "jd": [2459000.1, 2459000.2, 2459001.1, 2459002.1, 2459003.1],
            "band": ["V", "V", "V", "g", "g"],
            "mag": [14.0, 14.2, 14.1, 15.0, 15.2],
            "mag_err": [0.02] * 5,
        }
    )

    prepared = prepare_relative_flux(lightcurve)

    assert len(prepared) == 4
    assert np.all(np.isfinite(prepared["relative_flux"]))
    assert np.isclose(np.median(prepared["relative_flux"]), 1.0)
    assert prepared["n_exposures"].tolist() == [2, 1, 1, 1]


def test_weighted_wavelet_z_recovers_irregular_synthetic_scale() -> None:
    rng = np.random.default_rng(20260723)
    times = np.sort(rng.uniform(0.0, 420.0, 700))
    times = times[np.mod(times, 160.0) < 115.0]
    true_period = 23.0
    values = np.sin(2.0 * np.pi * times / true_period)
    values += rng.normal(0.0, 0.30, size=times.size)
    scales = np.geomspace(4.0, 100.0, 56)
    centers = np.linspace(times.min(), times.max(), 72)

    power, effective_points = weighted_wavelet_z(
        times,
        values,
        centers,
        scales,
    )
    valid_counts = np.sum(np.isfinite(power), axis=1)
    global_power = np.divide(
        np.nansum(power, axis=1),
        valid_counts,
        out=np.full(scales.size, np.nan),
        where=valid_counts > 0,
    )
    recovered_scale = float(scales[np.nanargmax(global_power)])

    assert power.shape == (len(scales), len(centers))
    assert effective_points.shape == power.shape
    assert np.isfinite(power).any()
    assert 18.0 < recovered_scale < 28.0


def test_analyze_wavelet_uses_span_adaptive_upper_scale() -> None:
    times = np.linspace(0.0, 2000.0, 401)
    prepared = pd.DataFrame(
        {
            "jd": 2450000.0 + times,
            "relative_flux": 1.0 + 0.03 * np.sin(2.0 * np.pi * times / 220.0),
        }
    )
    result = analyze_wavelet(
        prepared,
        config=WWZConfig(
            max_scale_days=None,
            max_scale_fraction_of_span=0.50,
            n_scales=24,
            n_time_bins=48,
        ),
    )

    assert np.isclose(result.max_scale_analyzed_days, 1000.0)
    assert result.max_scale_analyzed_days > 500.0
    assert 0.0 <= result.dominant_scale_edge_reliable_fraction <= 1.0
