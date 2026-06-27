from __future__ import annotations

import numpy as np

from malca.products.feature_layers import feature_mapping_get
from malca.review.stats_merge import merge_stats_summary_into_payload
from malca.core.stats import fit_fourier_decomposition


def _wrap_2pi(angle: float) -> float:
    return float(np.mod(angle, 2.0 * np.pi))


def test_fit_fourier_decomposition_recovers_classical_features() -> None:
    rng = np.random.default_rng(12345)
    period = 2.75
    a0 = 12.3
    amps = {1: 0.35, 2: 0.14, 3: 0.07}
    phases = {1: 0.8, 2: 2.1, 3: 4.7}

    time = np.sort(rng.uniform(0.0, 80.0, 500))
    phase = np.mod((time - time.min()) / period, 1.0)
    model = np.full_like(time, a0, dtype=float)
    for k, amp in amps.items():
        model = model + amp * np.cos(2.0 * np.pi * k * phase + phases[k])

    err = np.full_like(time, 0.02, dtype=float)
    mag = model + rng.normal(0.0, 0.003, size=time.size)

    result = fit_fourier_decomposition(mag, time, period, err=err, max_harmonics=5)

    assert int(result["harmonics_order"]) == 3
    assert result["harmonics_period"] == period
    np.testing.assert_allclose(result["harmonics_a0"], a0, atol=0.02)
    np.testing.assert_allclose(result["harmonics_mag_1"], amps[1], atol=0.02)
    np.testing.assert_allclose(result["harmonics_mag_2"], amps[2], atol=0.02)
    np.testing.assert_allclose(result["harmonics_mag_3"], amps[3], atol=0.02)
    np.testing.assert_allclose(result["harmonics_r21"], amps[2] / amps[1], atol=0.03)
    np.testing.assert_allclose(result["harmonics_r31"], amps[3] / amps[1], atol=0.03)
    np.testing.assert_allclose(result["harmonics_phase_2"], _wrap_2pi(phases[2] - 2.0 * phases[1]), atol=0.08)
    np.testing.assert_allclose(result["harmonics_phase_3"], _wrap_2pi(phases[3] - 3.0 * phases[1]), atol=0.08)
    assert result["harmonics_model_amplitude"] > 0.6
    assert result["harmonics_reduced_chi2"] < 1.0


def test_fit_fourier_decomposition_returns_nan_without_valid_period() -> None:
    result = fit_fourier_decomposition([1.0, 1.1, 0.9], [0.0, 1.0, 2.0], np.nan)

    assert np.isnan(result["harmonics_order"])
    assert np.isnan(result["harmonics_mag_1"])
    assert np.isnan(result["harmonics_r21"])


def test_merge_stats_summary_maps_fourier_fields() -> None:
    payload: dict[str, float] = {}
    merge_stats_summary_into_payload(
        payload,
        {
            "harmonics_order": 3,
            "harmonics_period": 2.75,
            "harmonics_a0": 12.3,
            "harmonics_model_amplitude": 0.72,
            "harmonics_reduced_chi2": 0.91,
            "harmonics_mag_1": 0.35,
            "harmonics_r21": 0.4,
            "harmonics_phase_2": 0.5,
        },
    )

    assert feature_mapping_get(payload, "stats_harmonics_order") == 3
    assert feature_mapping_get(payload, "stats_harmonics_period") == 2.75
    assert feature_mapping_get(payload, "stats_harmonics_a0") == 12.3
    assert feature_mapping_get(payload, "stats_harmonics_model_amplitude") == 0.72
    assert feature_mapping_get(payload, "stats_harmonics_reduced_chi2") == 0.91
    assert feature_mapping_get(payload, "stats_harmonics_mag_1") == 0.35
    assert feature_mapping_get(payload, "stats_harmonics_r21") == 0.4
    assert feature_mapping_get(payload, "stats_harmonics_phase_2") == 0.5
