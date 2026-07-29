from __future__ import annotations

import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

if "iar.IARModel" not in sys.modules:
    fake_iar_pkg = types.ModuleType("iar")
    fake_iar_model = types.ModuleType("iar.IARModel")

    def _dummy_iar(*args, **kwargs):
        _ = args, kwargs
        return float("nan")

    fake_iar_model.IARphikalman = _dummy_iar
    fake_iar_pkg.IARModel = fake_iar_model
    sys.modules["iar"] = fake_iar_pkg
    sys.modules["iar.IARModel"] = fake_iar_model

from malca.config import (
    STETSON_REWEIGHT_A,
    STETSON_REWEIGHT_B,
    STETSON_REWEIGHT_MAX_ITERS,
    STETSON_REWEIGHT_MIN_ITERS,
    STETSON_REWEIGHT_RTOL,
)
from malca.products.feature_layers import feature_mapping_get
from malca.review.stats_merge import merge_stats_summary_into_payload
from malca.core.stats import (
    compute_sokolovsky_peak_to_peak_summary,
    inverse_von_neumann_ratio,
    paper_iqr,
    paper_stetson_indices,
    roms_statistic,
    sokolovsky_peak_to_peak_variability,
    three_sigma_clipped_mean_mag,
    weighted_std,
)


def _reference_stetson_robust_mean(mag: np.ndarray, err: np.ndarray) -> float:
    weights = 1.0 / np.square(err)
    mu = float(np.sum(weights * mag) / np.sum(weights))
    scale = math.sqrt(mag.size / (mag.size - 1.0))

    for idx in range(STETSON_REWEIGHT_MAX_ITERS):
        resid = np.abs(scale * (mag - mu) / err)
        factors = 1.0 / (1.0 + np.power(resid / STETSON_REWEIGHT_A, STETSON_REWEIGHT_B))
        weights = np.maximum(weights * factors, np.finfo(float).tiny)
        new_mu = float(np.sum(weights * mag) / np.sum(weights))
        if idx + 1 >= STETSON_REWEIGHT_MIN_ITERS and math.isclose(
            new_mu,
            mu,
            rel_tol=STETSON_REWEIGHT_RTOL,
            abs_tol=STETSON_REWEIGHT_RTOL,
        ):
            mu = new_mu
            break
        mu = new_mu

    return mu


def test_paper_iqr_uses_median_of_halves_definition() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert paper_iqr(values) == 3.0


def test_weighted_std_matches_inverse_variance_formula() -> None:
    mag = np.array([10.0, 11.0, 13.0])
    err = np.array([1.0, 2.0, 0.5])
    w = 1.0 / np.square(err)
    w_mean = np.sum(w * mag) / np.sum(w)
    expected = math.sqrt((np.sum(w) / (np.sum(w) ** 2 - np.sum(np.square(w)))) * np.sum(w * np.square(mag - w_mean)))

    np.testing.assert_allclose(weighted_std(mag, err), expected)


def test_inverse_von_neumann_ratio_is_variance_over_mean_square_successive_difference() -> None:
    mag = np.array([0.0, 1.0, 2.0, 3.0])
    expected = np.var(mag, ddof=1) / np.mean(np.square(np.diff(mag)))

    np.testing.assert_allclose(inverse_von_neumann_ratio(mag), expected)


def test_roms_matches_paper_definition() -> None:
    mag = np.array([10.0, 11.0, 13.0])
    err = np.array([1.0, 2.0, 1.0])
    expected = (abs(10.0 - 11.0) / 1.0 + abs(11.0 - 11.0) / 2.0 + abs(13.0 - 11.0) / 1.0) / 2.0

    np.testing.assert_allclose(roms_statistic(mag, err), expected)


def test_sokolovsky_peak_to_peak_variability_matches_definition() -> None:
    mag = np.array([10.0, 11.0, 13.0, np.nan])
    err = np.array([0.1, 0.3, 0.5, 0.2])
    faint_limit = 13.0 - 0.5
    bright_limit = 10.0 + 0.1
    expected = (faint_limit - bright_limit) / (faint_limit + bright_limit)

    np.testing.assert_allclose(sokolovsky_peak_to_peak_variability(mag, err), expected)


def test_sokolovsky_peak_to_peak_variability_preserves_overlap_and_handles_missing() -> None:
    overlap = sokolovsky_peak_to_peak_variability(
        np.array([10.0, 10.05]), np.array([0.1, 0.1])
    )

    assert overlap < 0
    assert np.isnan(sokolovsky_peak_to_peak_variability([10.0], [0.1]))


def test_three_sigma_clipped_mean_mag_matches_stats_policy() -> None:
    assert three_sigma_clipped_mean_mag([10.0, 10.1, 10.2, 30.0]) == pytest.approx(10.1)


def test_sokolovsky_summary_uses_available_single_band_when_only_one_exists() -> None:
    frame = pd.DataFrame(
        {
            "JD": [1.0, 2.0, 3.0, 4.0],
            "mag": [10.0, 11.0, 12.0, 13.0],
            "error": [0.1, 0.2, 0.3, 0.1],
            "good_bad": [1, 1, 1, 0],
            "camera#": [1, 1, 1, 1],
            "v_g_band": [1, 1, 1, 1],
            "saturated": [0, 0, 0, 0],
            "camera_name": ["cam"] * 4,
            "field": ["field"] * 4,
        }
    )

    v_frame, v_summary = compute_sokolovsky_peak_to_peak_summary("source", ".", input_frame=frame)

    assert len(v_frame) == 3
    assert v_summary["sokolovsky_v_status"] == "ok"
    assert v_summary["variability_sokolovsky_v"] == pytest.approx(1.6 / 21.8)
    assert v_summary["sokolovsky_v_band"] == "V_only_no_g_reference"


def test_combined_sokolovsky_summary_offsets_full_v_median_to_g_before_computing_v() -> None:
    frame = pd.DataFrame(
        {
            "JD": [1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.1, 3.1, 4.1, 5.1],
            "mag": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8],
            "error": [0.1] * 10,
            "good_bad": [1] * 10,
            "camera#": [1] * 10,
            "v_g_band": [0] * 5 + [1] * 5,
            "saturated": [0] * 10,
            "camera_name": ["cam"] * 10,
            "field": ["field"] * 10,
        }
    )

    combined_frame, summary = compute_sokolovsky_peak_to_peak_summary(
        "source", ".", input_frame=frame
    )

    assert len(combined_frame) == 10
    assert summary["sokolovsky_v_band"] == "g+V_v_full_median_to_g_full_median"
    assert summary["sokolovsky_v_status"] == "ok"
    assert summary["variability_sokolovsky_v"] == pytest.approx(0.6 / 20.8)
    assert summary["sokolovsky_v_v_minus_g_median_offset_mag"] == pytest.approx(1.0)


def test_paper_stetson_indices_match_manual_single_band_formulae() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    mag = np.array([10.0, 10.2, 10.1, 11.6])
    err = np.full_like(mag, 0.1)

    result = paper_stetson_indices(time, mag, err, dtmax_days=2.0)

    n = mag.size
    mean_mag = _reference_stetson_robust_mean(mag, err)
    delta = np.sqrt(n / (n - 1.0)) * (mag - mean_mag) / err
    pair_terms = [((mag[0] - mean_mag) / err[0]) * ((mag[1] - mean_mag) / err[1]), ((mag[2] - mean_mag) / err[2]) * ((mag[3] - mean_mag) / err[3])]
    expected_i = math.sqrt(1.0 / (2.0 * 1.0)) * sum(pair_terms)
    p_plain = np.array([delta[0] * delta[1], delta[2] * delta[3]])
    expected_j = float(np.mean(np.sign(p_plain) * np.sqrt(np.abs(p_plain))))
    expected_k = float(np.mean(np.abs(delta)) / math.sqrt(np.mean(np.square(delta))))
    expected_l = math.sqrt(math.pi / 2.0) * expected_j * expected_k
    p_time = np.array([delta[0] * delta[1], delta[1] * delta[2], delta[2] * delta[3]])
    weights = np.exp(-np.diff(time) / np.median(np.diff(time)))
    expected_j_time = float(np.sum(weights * np.sign(p_time) * np.sqrt(np.abs(p_time))) / np.sum(weights))
    expected_l_time = math.sqrt(math.pi / 2.0) * expected_j_time * expected_k

    np.testing.assert_allclose(result["stetson_I"], expected_i)
    np.testing.assert_allclose(result["stetson_J"], expected_j)
    np.testing.assert_allclose(result["stetson_K"], expected_k)
    np.testing.assert_allclose(result["stetson_L"], expected_l)
    np.testing.assert_allclose(result["stetson_J_time"], expected_j_time)
    np.testing.assert_allclose(result["stetson_L_time"], expected_l_time)


def test_merge_stats_summary_maps_new_paper_stats() -> None:
    payload: dict[str, float] = {}

    merge_stats_summary_into_payload(
        payload,
        {
            "photometry_weighted_std_mag": 0.21,
            "variability_von_neumann_ratio": 1.8,
            "variability_roms": 1.3,
            "variability_sokolovsky_v": 0.024,
            "variability_stetson_L": 2.4,
            "variability_stetson_J_time": 1.7,
            "variability_stetson_L_time": 2.1,
        },
    )

    assert feature_mapping_get(payload, "stats_photometry_weighted_std_mag") == 0.21
    assert feature_mapping_get(payload, "stats_variability_von_neumann_ratio") == 1.8
    assert feature_mapping_get(payload, "stats_variability_roms") == 1.3
    assert feature_mapping_get(payload, "stats_variability_sokolovsky_v") == 0.024
    assert feature_mapping_get(payload, "stats_variability_stetson_L") == 2.4
    assert feature_mapping_get(payload, "stats_variability_stetson_J_time") == 1.7
    assert feature_mapping_get(payload, "stats_variability_stetson_L_time") == 2.1
