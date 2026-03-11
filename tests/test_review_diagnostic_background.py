from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.review.store import db_connect, get_diagnostic_background, import_candidates


def test_get_diagnostic_background_includes_metric_planes(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "BG-1",
                        "periodicity_score": 0.7,
                        "phase_quality_score": 0.8,
                        "dipper_score": 5.5,
                        "jumper_score": 1.2,
                        "period_n_sources": 3.0,
                        "dip_run_count": 2.0,
                        "dip_inter_event_spacing_median": 12.0,
                        "dip_inter_event_spacing_std": 1.5,
                        "dip_amplitude_consistency": 0.8,
                        "dip_duration_consistency": 0.7,
                        "stats_photometry_robust_sigma_mag": 0.11,
                        "stats_variability_stetson_J": 4.0,
                        "stats_skew": 1.2,
                        "stats_max_slope": 0.6,
                        "stats_harmonics_model_amplitude": 0.2,
                        "stats_harmonics_reduced_chi2": 1.8,
                        "stats_variability_lag1_autocorr": 0.6,
                        "stats_autocor_length": 15.0,
                        "pm_cluster_offset_sigma": 2.0,
                        "ruwe": 1.1,
                        "P_disk": 0.9,
                        "P_eb": 0.05,
                        "atlas_cyan_range": 0.3,
                        "atlas_orange_range": 0.25,
                        "ztf_lc_g_range": 0.4,
                        "ztf_lc_r_range": 0.35,
                        "neowise_w1_range": 0.2,
                        "neowise_w2_range": 0.18,
                        "gaia_epoch_n_obs": 80.0,
                        "gaia_epoch_g_range": 0.12,
                        "ltv_slope": -0.15,
                        "ltv_dispersion": 0.5,
                        "ltv_neowise_w1_slope": 0.04,
                        "ltv_neowise_w1_w2_slope": -0.01,
                    }
                ]
            ),
            source_path="test://diagnostic-background",
            characterize_before_import=False,
            vet_before_import=False,
        )

        background = get_diagnostic_background(conn)

    assert float(background["metric_periodicity_score"][0]) == 0.7
    assert float(background["metric_jumper_score"][0]) == 1.2
    assert float(background["plane_catalog_support_x"][0]) == 3.0
    assert float(background["plane_var_strength_y"][0]) == 5.5
    assert float(background["plane_cluster_y"][0]) == 1.1
    assert float(background["plane_neowise_trend_x"][0]) == 0.04
