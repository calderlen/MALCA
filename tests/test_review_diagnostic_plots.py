from __future__ import annotations

import numpy as np
import pytest

from malca.review.diagnostic_plots import (
    build_atlas_range_figure,
    build_autocorr_memory_figure,
    build_catalog_support_figure,
    build_classifier_plane_figure,
    build_cluster_astrometry_figure,
    build_dip_repeatability_figure,
    build_gaia_epoch_figure,
    build_harmonic_quality_figure,
    build_ltv_trend_figure,
    build_neowise_range_figure,
    build_neowise_trend_figure,
    build_periodicity_plane_figure,
    build_recurrence_regularity_figure,
    build_score_balance_figure,
    build_shape_impulsiveness_figure,
    build_stetson_scatter_figure,
    build_variability_strength_figure,
    build_ztf_range_figure,
)


def test_build_periodicity_plane_figure_uses_candidate_and_background() -> None:
    fig = build_periodicity_plane_figure(
        {
            "periodicity_score": 0.82,
            "phase_quality_score": 0.74,
        },
        "black",
        background={
            "metric_periodicity_score": np.array([0.15, 0.55, 0.91]),
            "metric_phase_quality_score": np.array([0.22, 0.48, 0.87]),
        },
    )

    assert fig is not None
    assert fig.to_plotly_json()["layout"]["title"]["text"] == "Periodicity Plane"
    assert len(fig.data) == 2
    assert float(fig.data[-1].x[0]) == 0.82
    assert float(fig.data[-1].y[0]) == 0.74


def test_build_score_balance_figure_uses_candidate_and_background() -> None:
    fig = build_score_balance_figure(
        {
            "dipper_score": 6.4,
            "jumper_score": 1.8,
        },
        "black",
        background={
            "metric_dipper_score": np.array([1.0, 2.5, 4.2]),
            "metric_jumper_score": np.array([0.8, 2.0, 5.4]),
        },
    )

    assert fig is not None
    assert fig.to_plotly_json()["layout"]["title"]["text"] == "Morphology Scores"
    assert len(fig.data) == 2
    assert float(fig.data[-1].x[0]) == 6.4
    assert float(fig.data[-1].y[0]) == 1.8


@pytest.mark.parametrize(
    ("builder", "payload", "background", "title"),
    [
        (
            build_catalog_support_figure,
            {"period_n_sources": 3.0, "dip_run_count": 2.0},
            {"plane_catalog_support_x": np.array([1.0, 2.0]), "plane_catalog_support_y": np.array([0.0, 3.0])},
            "Catalog Support vs Dip Recurrence",
        ),
        (
            build_recurrence_regularity_figure,
            {"dip_inter_event_spacing_median": 12.0, "dip_inter_event_spacing_std": 1.5},
            {"plane_recurrence_regularity_x": np.array([5.0, 20.0]), "plane_recurrence_regularity_y": np.array([0.8, 3.0])},
            "Recurrence Regularity",
        ),
        (
            build_dip_repeatability_figure,
            {"dip_amplitude_consistency": 0.8, "dip_duration_consistency": 0.7},
            {"plane_dip_repeatability_x": np.array([0.3, 0.9]), "plane_dip_repeatability_y": np.array([0.2, 0.6])},
            "Dip Repeatability",
        ),
        (
            build_variability_strength_figure,
            {"stats_photometry_robust_sigma_mag": 0.12, "dipper_score": 6.1},
            {"plane_var_strength_x": np.array([0.03, 0.2]), "plane_var_strength_y": np.array([2.0, 7.0])},
            "Dipper Score vs Scatter",
        ),
        (
            build_stetson_scatter_figure,
            {"stats_photometry_robust_sigma_mag": 0.11, "stats_variability_stetson_J": 4.5},
            {"plane_stetson_x": np.array([0.02, 0.3]), "plane_stetson_y": np.array([0.5, 6.0])},
            "Scatter vs Stetson J",
        ),
        (
            build_shape_impulsiveness_figure,
            {"stats_skew": 1.4, "stats_max_slope": 0.7},
            {"plane_shape_x": np.array([-0.5, 2.0]), "plane_shape_y": np.array([0.1, 1.5])},
            "Shape and Impulsiveness",
        ),
        (
            build_harmonic_quality_figure,
            {"stats_harmonics_model_amplitude": 0.25, "stats_harmonics_reduced_chi2": 1.8},
            {"plane_harmonic_x": np.array([0.05, 0.6]), "plane_harmonic_y": np.array([1.1, 4.2])},
            "Harmonic Fit Quality",
        ),
        (
            build_autocorr_memory_figure,
            {"stats_variability_lag1_autocorr": 0.62, "stats_autocor_length": 18.0},
            {"plane_autocorr_x": np.array([-0.2, 0.8]), "plane_autocorr_y": np.array([4.0, 30.0])},
            "Autocorrelation Memory",
        ),
        (
            build_cluster_astrometry_figure,
            {"pm_cluster_offset_sigma": 2.1, "ruwe": 1.08},
            {"plane_cluster_x": np.array([0.8, 4.5]), "plane_cluster_y": np.array([1.0, 1.9])},
            "Cluster Astrometry",
        ),
        (
            build_classifier_plane_figure,
            {"P_disk": 0.84, "P_eb": 0.08},
            {"plane_classifier_x": np.array([0.1, 0.6]), "plane_classifier_y": np.array([0.7, 0.2])},
            "Classifier Plane",
        ),
        (
            build_atlas_range_figure,
            {"atlas_cyan_range": 0.35, "atlas_orange_range": 0.21},
            {"plane_atlas_x": np.array([0.08, 0.5]), "plane_atlas_y": np.array([0.07, 0.4])},
            "ATLAS Range Plane",
        ),
        (
            build_ztf_range_figure,
            {"ztf_lc_g_range": 0.44, "ztf_lc_r_range": 0.31},
            {"plane_ztf_x": np.array([0.05, 0.6]), "plane_ztf_y": np.array([0.04, 0.45])},
            "ZTF Range Plane",
        ),
        (
            build_neowise_range_figure,
            {"neowise_w1_range": 0.27, "neowise_w2_range": 0.19},
            {"plane_neowise_range_x": np.array([0.03, 0.4]), "plane_neowise_range_y": np.array([0.02, 0.3])},
            "NEOWISE Range Plane",
        ),
        (
            build_gaia_epoch_figure,
            {"gaia_epoch_n_obs": 74.0, "gaia_epoch_g_range": 0.12},
            {"plane_gaia_epoch_x": np.array([20.0, 120.0]), "plane_gaia_epoch_y": np.array([0.03, 0.25])},
            "Gaia Epoch Coverage",
        ),
        (
            build_ltv_trend_figure,
            {"ltv_slope": -0.18, "ltv_dispersion": 0.42},
            {"plane_ltv_x": np.array([-0.4, 0.2]), "plane_ltv_y": np.array([0.08, 0.9])},
            "LTV Trend vs Dispersion",
        ),
        (
            build_neowise_trend_figure,
            {"ltv_neowise_w1_slope": 0.05, "ltv_neowise_w1_w2_slope": -0.02},
            {"plane_neowise_trend_x": np.array([-0.1, 0.08]), "plane_neowise_trend_y": np.array([-0.05, 0.04])},
            "NEOWISE Trend Plane",
        ),
    ],
)
def test_additional_diagnostic_builders_return_expected_titles(builder, payload, background, title) -> None:
    fig = builder(payload, "black", background=background)

    assert fig is not None
    assert fig.to_plotly_json()["layout"]["title"]["text"] == title
    assert len(fig.data) == 2
