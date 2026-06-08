"""Helpers for mapping `malca.stats.compute_stats()` output into review payloads.

This is split out from `malca.review.pipeline` so that lightweight callers
(`malca.review.fetch`, CLI helpers, etc.) can enrich payloads without importing
optional/heavy dependencies from other pipeline stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from malca.derived_stats import DERIVED_FEATURE_COLUMNS, append_derived_features
from malca.feature_layers import to_layer_first_mapping


def _is_missing_value(value: object) -> bool:
    """Return True when a payload value should be treated as absent."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def merge_stats_summary_into_payload(payload: dict, summary: dict) -> None:
    """Map compute_stats() output into the review app's expected stats schema."""
    if not isinstance(summary, dict):
        return

    scalar_map = {
        "file_points_total": "stats_file_points_total",
        "file_points_kept_after_filter": "stats_file_points_kept_after_filter",
        "jd_start": "stats_jd_start",
        "jd_end": "stats_jd_end",
        "time_span_days": "stats_time_span_days",
        "n_unique_nights": "stats_n_unique_nights",
        "duty_cycle_fraction": "stats_duty_cycle_fraction",
        "cadence_mean_dt_days": "stats_cadence_mean_dt_days",
        "cadence_median_dt_days": "stats_cadence_median_dt_days",
        "cadence_p05_dt_days": "stats_cadence_p05_dt_days",
        "cadence_p95_dt_days": "stats_cadence_p95_dt_days",
        "photometry_mean_mag": "stats_photometry_mean_mag",
        "photometry_median_mag": "stats_photometry_median_mag",
        "photometry_weighted_mean_mag": "stats_photometry_weighted_mean_mag",
        "photometry_weighted_mean_sem": "stats_photometry_weighted_mean_sem",
        "photometry_weighted_std_mag": "stats_photometry_weighted_std_mag",
        "photometry_std_mag": "stats_photometry_std_mag",
        "photometry_robust_sigma_mag": "stats_photometry_robust_sigma_mag",
        "photometry_IQR_mag": "stats_photometry_IQR_mag",
        "photometry_p05_mag": "stats_photometry_p05_mag",
        "photometry_p16_mag": "stats_photometry_p16_mag",
        "photometry_p84_mag": "stats_photometry_p84_mag",
        "photometry_p95_mag": "stats_photometry_p95_mag",
        "clipped_mean_mag_3sigma_about_median": "stats_clipped_mean_mag_3sigma_about_median",
        "clipped_std_mag_3sigma_about_median": "stats_clipped_std_mag_3sigma_about_median",
        "n_outliers_removed_robust_3sigma": "stats_n_outliers_removed_robust_3sigma",
        "variability_reduced_chi2_vs_constant": "stats_variability_reduced_chi2_vs_constant",
        "variability_von_neumann_ratio": "stats_variability_von_neumann_ratio",
        "variability_roms": "stats_variability_roms",
        "variability_lag1_autocorr": "stats_variability_lag1_autocorr",
        "variability_stetson_I": "stats_variability_stetson_I",
        "variability_stetson_J": "stats_variability_stetson_J",
        "variability_stetson_K": "stats_variability_stetson_K",
        "variability_stetson_L": "stats_variability_stetson_L",
        "variability_stetson_J_time": "stats_variability_stetson_J_time",
        "variability_stetson_L_time": "stats_variability_stetson_L_time",
        "variability_flux_asymmetry_m": "stats_variability_flux_asymmetry_m",
        "variability_quasi_periodicity_q": "stats_variability_quasi_periodicity_q",
        "variability_string_length_resid_total": "stats_variability_string_length_resid_total",
        "variability_string_length_resid_mean_step": "stats_variability_string_length_resid_mean_step",
        "variability_string_length_resid_n_steps": "stats_variability_string_length_resid_n_steps",
        "variability_lomb_scargle_best_period_days": "stats_variability_lomb_scargle_best_period_days",
        "variability_lomb_scargle_peak_power": "stats_variability_lomb_scargle_peak_power",
        "variability_lomb_scargle_fap": "stats_variability_lomb_scargle_fap",
        "trend_slope_mag_per_day": "stats_trend_slope_mag_per_day",
        "trend_slope_mag_per_year": "stats_trend_slope_mag_per_year",
        "trend_r2": "stats_trend_r2",
        # ALeRCE-style features
        "amplitude": "stats_amplitude",
        "beyond_1_std": "stats_beyond_1_std",
        "con": "stats_con",
        "delta_mag_fid": "stats_delta_mag_fid",
        "intrinsic_sigma_mag": "stats_intrinsic_sigma_mag",
        "first_mag": "stats_first_mag",
        "gskew": "stats_gskew",
        "max_slope": "stats_max_slope",
        "meanvariance": "stats_meanvariance",
        "median_abs_dev": "stats_median_abs_dev",
        "median_brp": "stats_median_brp",
        "percent_amplitude": "stats_percent_amplitude",
        "ahl_ratio": "stats_ahl_ratio",
        "q31": "stats_q31",
        "skew": "stats_skew",
        "small_kurtosis": "stats_small_kurtosis",
        "constancy_p_value": "stats_constancy_p_value",
        "anderson_darling": "stats_anderson_darling",
        "pair_slope_trend": "stats_pair_slope_trend",
        "rcs": "stats_rcs",
        "autocor_length": "stats_autocor_length",
        "sf_ml_amplitude": "stats_sf_ml_amplitude",
        "sf_ml_gamma": "stats_sf_ml_gamma",
        # period-dependent features
        "harmonics_order": "stats_harmonics_order",
        "harmonics_period": "stats_harmonics_period",
        "harmonics_a0": "stats_harmonics_a0",
        "harmonics_model_amplitude": "stats_harmonics_model_amplitude",
        "harmonics_reduced_chi2": "stats_harmonics_reduced_chi2",
        "harmonics_mag_1": "stats_harmonics_mag_1",
        "harmonics_mag_2": "stats_harmonics_mag_2",
        "harmonics_mag_3": "stats_harmonics_mag_3",
        "harmonics_mag_4": "stats_harmonics_mag_4",
        "harmonics_mag_5": "stats_harmonics_mag_5",
        "harmonics_mag_6": "stats_harmonics_mag_6",
        "harmonics_mag_7": "stats_harmonics_mag_7",
        "harmonics_r21": "stats_harmonics_r21",
        "harmonics_r31": "stats_harmonics_r31",
        "harmonics_r41": "stats_harmonics_r41",
        "harmonics_r51": "stats_harmonics_r51",
        "harmonics_r61": "stats_harmonics_r61",
        "harmonics_r71": "stats_harmonics_r71",
        "harmonics_phase_2": "stats_harmonics_phase_2",
        "harmonics_phase_3": "stats_harmonics_phase_3",
        "harmonics_phase_4": "stats_harmonics_phase_4",
        "harmonics_phase_5": "stats_harmonics_phase_5",
        "harmonics_phase_6": "stats_harmonics_phase_6",
        "harmonics_phase_7": "stats_harmonics_phase_7",
        "harmonics_mse": "stats_harmonics_mse",
        "psi_cs": "stats_psi_cs",
        "psi_eta": "stats_psi_eta",
        "lafler_kinman_t_time": "stats_lafler_kinman_t_time",
        "lafler_kinman_t_phase": "stats_lafler_kinman_t_phase",
        "lafler_kinman_delta": "stats_lafler_kinman_delta",
        "eb_rminima": "stats_eb_rminima",
        "eb_primary_min_depth": "stats_eb_primary_min_depth",
        "eb_secondary_min_depth": "stats_eb_secondary_min_depth",
        # stochastic model features
        "gp_drw_sigma": "stats_gp_drw_sigma",
        "gp_drw_tau": "stats_gp_drw_tau",
        "iar_phi": "stats_iar_phi",
        "mhps_high": "stats_mhps_high",
        "mhps_low": "stats_mhps_low",
        "mhps_non_zero": "stats_mhps_non_zero",
        "mhps_pn_flag": "stats_mhps_pn_flag",
        "mhps_ratio": "stats_mhps_ratio",
        "camera_loo_corr_min": "stats_camera_loo_corr_min",
        "camera_loo_corr_median": "stats_camera_loo_corr_median",
        "camera_loo_rms_max": "stats_camera_loo_rms_max",
    }
    scalar_map.update({f"harmonics_a{k}": f"stats_harmonics_a{k}" for k in range(1, 8)})
    scalar_map.update({f"harmonics_b{k}": f"stats_harmonics_b{k}" for k in range(1, 8)})
    scalar_map.update({
        f"window_alias_period_{k}": f"stats_window_alias_period_{k}"
        for k in range(1, 6)
    })
    scalar_map.update({
        f"window_alias_power_{k}": f"stats_window_alias_power_{k}"
        for k in range(1, 6)
    })

    for source_key, target_key in scalar_map.items():
        value = summary.get(source_key)
        if _is_missing_value(value):
            continue
        if isinstance(value, (pd.DataFrame, pd.Series, dict, list, tuple, set)):
            continue
        payload[target_key] = value

    derived_frame = append_derived_features(pd.DataFrame([payload]))
    if not derived_frame.empty:
        derived_row = derived_frame.iloc[0]
        for col in DERIVED_FEATURE_COLUMNS:
            value = derived_row.get(col)
            if _is_missing_value(value):
                continue
            payload[col] = value
    err_stats = summary.get("error_and_snr_stats")
    if isinstance(err_stats, dict):
        for subkey, value in err_stats.items():
            if _is_missing_value(value):
                continue
            if isinstance(value, (pd.DataFrame, pd.Series, dict, list, tuple, set)):
                continue
            payload[f"stats_error_and_snr_stats_{subkey}"] = value

    # Populate the core summary fields the review UI uses for the Stats chip.
    kept_points = summary.get("file_points_kept_after_filter")
    if not _is_missing_value(kept_points):
        payload["n_points"] = kept_points

    cadence_median = summary.get("cadence_median_dt_days")
    if not _is_missing_value(cadence_median):
        payload["cadence_median_days"] = cadence_median

    if _is_missing_value(payload.get("baseline_mag")):
        baseline_mag = summary.get("photometry_median_mag")
        if not _is_missing_value(baseline_mag):
            payload["baseline_mag"] = baseline_mag

    by_camera = summary.get("by_camera")
    if isinstance(by_camera, pd.DataFrame) and not by_camera.empty:
        payload["n_cameras"] = int(len(by_camera))

    layered = to_layer_first_mapping(payload, layer_values_as_json=False, run_derived=False)
    payload.clear()
    payload.update(layered)
