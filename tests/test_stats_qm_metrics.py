from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.core import stats as stats_mod
from malca.core.stats import flux_asymmetry_metric, phase_template_quasi_periodicity, quasi_periodicity_metric
from malca.products.feature_layers import feature_mapping_get
from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.stats_merge import merge_stats_summary_into_payload
from malca.review.store import _COL_NAMES


def _stats_lightcurve(period: float = 2.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    jd = np.sort(rng.uniform(2458000.0, 2458080.0, 800))
    mag = 13.0 + 0.35 * np.sin(2.0 * np.pi * (jd - jd.min()) / period) + rng.normal(0.0, 0.005, size=jd.size)
    df_g = pd.DataFrame(
        {
            "JD": jd,
            "mag": mag,
            "error": np.full(jd.size, 0.02),
            "good_bad": np.ones(jd.size, dtype=int),
            "camera#": np.ones(jd.size, dtype=int),
            "v_g_band": np.zeros(jd.size, dtype=int),
            "saturated": np.zeros(jd.size, dtype=int),
            "camera_name": ["cam-a"] * jd.size,
            "field": ["field-a"] * jd.size,
        }
    )
    return df_g, pd.DataFrame()


def _patch_compute_stats_loaders(monkeypatch: pytest.MonkeyPatch, period: float = 2.75) -> None:
    df_g, df_v = _stats_lightcurve(period)
    monkeypatch.setattr(stats_mod, "read_lc_csv", lambda *_args, **_kwargs: (df_g.copy(), df_v.copy()))
    monkeypatch.setattr(stats_mod, "read_skypatrol_lc_csv", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(stats_mod, "read_lc_dat2", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(stats_mod, "fit_drw", lambda *_args, **_kwargs: (np.nan, np.nan))
    monkeypatch.setattr(stats_mod, "iar_phi_fit", lambda *_args, **_kwargs: np.nan)
    monkeypatch.setattr(
        stats_mod,
        "mhps",
        lambda *_args, **_kwargs: {
            "mhps_high": np.nan,
            "mhps_low": np.nan,
            "mhps_non_zero": np.nan,
            "mhps_pn_flag": False,
            "mhps_ratio": np.nan,
        },
    )


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
    time = np.sort(rng.uniform(0.0, 80.0, 1000))
    err = np.full_like(time, 0.02)
    mag = 13.0 + 0.35 * np.sin(2.0 * np.pi * time / period) + rng.normal(0.0, 0.005, size=time.size)

    q_metric = quasi_periodicity_metric(mag, time, err, period)

    assert np.isfinite(q_metric)
    assert q_metric < 0.05


def test_quasi_periodicity_metric_is_large_for_aperiodic_signal() -> None:
    rng = np.random.default_rng(456)
    time = np.sort(rng.uniform(0.0, 80.0, 1000))
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


def test_phase_template_q_flags_invalid_period_with_diagnostics() -> None:
    time = np.arange(20.0)
    mag = np.sin(time)
    err = np.full_like(time, 0.02)

    result = phase_template_quasi_periodicity(mag, time, err, np.nan)

    assert np.isnan(result["q"])
    assert result["method"] == "phase_template_med500m2"
    assert result["n_bins"] == 500
    assert result["smooth_window_bins"] == 1
    assert result["status"] == "invalid_period"


def test_phase_template_q_is_low_for_coherent_narrow_dip_train() -> None:
    rng = np.random.default_rng(789)
    period = 2.75
    time = np.sort(rng.uniform(0.0, 120.0, 800))
    err = np.full_like(time, 0.02)
    phase = np.mod((time - time.min()) / period, 1.0)
    in_dip = np.abs(((phase - 0.22 + 0.5) % 1.0) - 0.5) < 0.035
    mag = 13.0 + rng.normal(0.0, 0.025, size=time.size)
    mag[in_dip] += 0.55 + rng.normal(0.0, 0.04, size=int(in_dip.sum()))

    result = phase_template_quasi_periodicity(mag, time, err, period)

    assert result["status"] == "ok"
    assert result["n_bins"] == 500
    assert result["smooth_window_bins"] == 1
    assert result["populated_bins"] >= 50
    assert result["bin_coverage"] >= 0.10
    assert result["q"] < 0.2
    assert result["scatter_ratio"] < 0.35
    assert result["template_amplitude"] > 0.4


def test_phase_template_q_stays_high_for_wrong_period() -> None:
    rng = np.random.default_rng(789)
    period = 2.75
    time = np.sort(rng.uniform(0.0, 120.0, 800))
    err = np.full_like(time, 0.02)
    phase = np.mod((time - time.min()) / period, 1.0)
    in_dip = np.abs(((phase - 0.22 + 0.5) % 1.0) - 0.5) < 0.035
    mag = 13.0 + rng.normal(0.0, 0.025, size=time.size)
    mag[in_dip] += 0.55 + rng.normal(0.0, 0.04, size=int(in_dip.sum()))

    result = phase_template_quasi_periodicity(mag, time, err, period * 1.37)

    assert result["status"] == "ok"
    assert result["q"] > 0.8
    assert result["scatter_ratio"] > 0.85


def test_phase_template_q_fails_safely_with_sparse_phase_coverage() -> None:
    period = 2.75
    phase = np.repeat([0.05, 0.20, 0.40, 0.60], 5)
    time = phase * period
    err = np.full_like(time, 0.02)
    mag = 13.0 + np.sin(2.0 * np.pi * phase)

    result = phase_template_quasi_periodicity(mag, time, err, period)

    assert np.isnan(result["q"])
    assert result["status"] == "insufficient_phase_coverage"
    assert result["populated_bins"] < 50
    assert result["bin_coverage"] < 0.10


def test_phase_template_q_accepts_legacy_noise_subtracted_configuration() -> None:
    rng = np.random.default_rng(123)
    period = 2.75
    time = np.sort(rng.uniform(0.0, 80.0, 1000))
    err = np.full_like(time, 0.02)
    mag = 13.0 + 0.35 * np.sin(2.0 * np.pi * time / period) + rng.normal(0.0, 0.005, size=time.size)

    result = phase_template_quasi_periodicity(
        mag,
        time,
        err,
        period,
        n_phase_bins=50,
        min_bin_points=3,
        smooth_window_bins=3,
        min_bin_coverage=0.25,
        noise_subtract=True,
    )

    assert result["status"] == "ok"
    assert result["n_bins"] == 50
    assert result["smooth_window_bins"] == 3
    assert result["q"] < 0.05


def test_phase_template_q_plain_variance_ignores_photometric_noise_floor() -> None:
    rng = np.random.default_rng(246)
    period = 2.75
    time = np.sort(rng.uniform(0.0, 80.0, 1000))
    err = np.full_like(time, 0.10)
    mag = 13.0 + 0.01 * np.sin(2.0 * np.pi * time / period) + rng.normal(0.0, 0.002, size=time.size)

    plain = phase_template_quasi_periodicity(mag, time, err, period)
    noise_subtracted = phase_template_quasi_periodicity(
        mag,
        time,
        err,
        period,
        n_phase_bins=50,
        min_bin_points=3,
        smooth_window_bins=3,
        min_bin_coverage=0.25,
        noise_subtract=True,
    )

    assert plain["status"] == "ok"
    assert np.isfinite(plain["q"])
    assert noise_subtracted["status"] == "low_intrinsic_variance"


def test_compute_stats_q_uses_explicit_feature_period_without_lomb_scargle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    period = 2.75
    _patch_compute_stats_loaders(monkeypatch, period=period)

    _df, summary = stats_mod.compute_stats(
        "123",
        "/tmp",
        compute_ls=False,
        feature_period_days=period,
        feature_period_source="pdm_corrected_period",
    )

    assert np.isfinite(summary["variability_quasi_periodicity_q"])
    assert summary["variability_quasi_periodicity_q"] < 0.05
    assert summary["variability_quasi_periodicity_method"] == "phase_template_med500m2"
    assert summary["variability_quasi_periodicity_n_bins"] == 500
    assert summary["variability_quasi_periodicity_smooth_window_bins"] == 1
    assert summary["variability_quasi_periodicity_status"] == "ok"
    assert summary["variability_periodic_feature_period_days"] == pytest.approx(period)
    assert summary["variability_periodic_feature_period_source"] == "pdm_corrected_period"
    assert np.isnan(summary["variability_lomb_scargle_best_period_days"])


def test_compute_stats_q_stays_nan_without_explicit_feature_period_and_no_ls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compute_stats_loaders(monkeypatch, period=2.75)

    _df, summary = stats_mod.compute_stats("123", "/tmp", compute_ls=False)

    assert np.isnan(summary["variability_quasi_periodicity_q"])
    assert summary["variability_quasi_periodicity_status"] == "invalid_period"
    assert np.isnan(summary["variability_periodic_feature_period_days"])
    assert summary["variability_periodic_feature_period_source"] == ""
    assert np.isnan(summary["variability_lomb_scargle_best_period_days"])


def test_compute_quasi_periodicity_summary_promotes_best_q_period_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(987)
    period = 2.75
    jd = np.sort(rng.uniform(2458000.0, 2458120.0, 1200))
    phase = np.mod((jd - jd.min()) / period, 1.0)
    in_dip = np.abs(((phase - 0.22 + 0.5) % 1.0) - 0.5) < 0.035
    mag = 13.0 + rng.normal(0.0, 0.025, size=jd.size)
    mag[in_dip] += 0.55 + rng.normal(0.0, 0.04, size=int(in_dip.sum()))
    df_g = pd.DataFrame(
        {
            "JD": jd,
            "mag": mag,
            "error": np.full(jd.size, 0.02),
            "good_bad": np.ones(jd.size, dtype=int),
            "camera#": np.ones(jd.size, dtype=int),
            "v_g_band": np.zeros(jd.size, dtype=int),
            "saturated": np.zeros(jd.size, dtype=int),
            "camera_name": ["cam-a"] * jd.size,
            "field": ["field-a"] * jd.size,
        }
    )
    monkeypatch.setattr(stats_mod, "read_lc_csv", lambda *_args, **_kwargs: (df_g.copy(), pd.DataFrame()))
    monkeypatch.setattr(stats_mod, "read_skypatrol_lc_csv", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(stats_mod, "read_lc_dat2", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))

    summary = stats_mod.compute_quasi_periodicity_summary(
        "123",
        "/tmp",
        feature_period_days=period / 2.0,
        feature_period_source="periodicity_period",
    )

    assert summary["variability_quasi_periodicity_status"] == "ok"
    assert summary["variability_quasi_periodicity_q"] < 0.1
    assert summary["variability_periodic_feature_period_days"] == pytest.approx(period)
    assert summary["variability_periodic_feature_period_source"] == "periodicity_period:q_factor_2"


def test_compute_stats_q_uses_all_bands_after_camera_band_offset_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(321)
    period = 2.75
    jd_g = np.sort(rng.uniform(2458000.0, 2458060.0, 220))
    jd_v = np.sort(rng.uniform(2458000.0, 2458060.0, 220))

    def make_frame(jd: np.ndarray, *, band: int, camera: str, offset: float) -> pd.DataFrame:
        phase = np.mod((jd - min(jd_g.min(), jd_v.min())) / period, 1.0)
        mag = 13.0 + offset + 0.25 * np.sin(2.0 * np.pi * phase) + rng.normal(0.0, 0.006, size=jd.size)
        return pd.DataFrame(
            {
                "JD": jd,
                "mag": mag,
                "error": np.full(jd.size, 0.02),
                "good_bad": np.ones(jd.size, dtype=int),
                "camera#": np.ones(jd.size, dtype=int),
                "v_g_band": np.full(jd.size, band, dtype=int),
                "saturated": np.zeros(jd.size, dtype=int),
                "camera_name": [camera] * jd.size,
                "field": ["field-a"] * jd.size,
            }
        )

    df_g = make_frame(jd_g, band=0, camera="cam-g", offset=0.0)
    df_v = make_frame(jd_v, band=1, camera="cam-v", offset=1.7)
    monkeypatch.setattr(stats_mod, "read_lc_csv", lambda *_args, **_kwargs: (df_g.copy(), df_v.copy()))
    monkeypatch.setattr(stats_mod, "read_skypatrol_lc_csv", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(stats_mod, "read_lc_dat2", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(stats_mod, "fit_drw", lambda *_args, **_kwargs: (np.nan, np.nan))
    monkeypatch.setattr(stats_mod, "iar_phi_fit", lambda *_args, **_kwargs: np.nan)
    monkeypatch.setattr(
        stats_mod,
        "mhps",
        lambda *_args, **_kwargs: {
            "mhps_high": np.nan,
            "mhps_low": np.nan,
            "mhps_non_zero": np.nan,
            "mhps_pn_flag": False,
            "mhps_ratio": np.nan,
        },
    )

    _df, summary = stats_mod.compute_stats(
        "123",
        "/tmp",
        compute_ls=False,
        feature_period_days=period,
        feature_period_source="ce_corrected_period",
    )

    assert summary["variability_quasi_periodicity_status"] == "ok"
    assert summary["variability_quasi_periodicity_n_points"] == 440
    assert summary["variability_quasi_periodicity_q"] < 0.05


def test_qm_stats_merge_and_review_schema_entries() -> None:
    payload: dict[str, float] = {}
    merge_stats_summary_into_payload(
        payload,
        {
            "variability_quasi_periodicity_q": 0.12,
            "variability_quasi_periodicity_method": "phase_template_med500m2",
            "variability_quasi_periodicity_n_points": 120,
            "variability_quasi_periodicity_n_bins": 500,
            "variability_quasi_periodicity_populated_bins": 44,
            "variability_quasi_periodicity_bin_coverage": 0.88,
            "variability_quasi_periodicity_smooth_window_bins": 1,
            "variability_quasi_periodicity_template_amplitude": 0.42,
            "variability_quasi_periodicity_raw_scatter": 0.31,
            "variability_quasi_periodicity_resid_scatter": 0.12,
            "variability_quasi_periodicity_scatter_ratio": 0.39,
            "variability_quasi_periodicity_status": "ok",
            "variability_flux_asymmetry_m": 0.34,
            "variability_periodic_feature_period_days": 2.75,
            "variability_periodic_feature_period_source": "ce_corrected_period",
        },
    )

    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_q") == 0.12
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_method") == "phase_template_med500m2"
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_n_points") == 120
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_n_bins") == 500
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_populated_bins") == 44
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_bin_coverage") == 0.88
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_smooth_window_bins") == 1
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_template_amplitude") == 0.42
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_raw_scatter") == 0.31
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_resid_scatter") == 0.12
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_scatter_ratio") == 0.39
    assert feature_mapping_get(payload, "stats_variability_quasi_periodicity_status") == "ok"
    assert feature_mapping_get(payload, "stats_variability_flux_asymmetry_m") == 0.34
    assert feature_mapping_get(payload, "stats_variability_periodic_feature_period_days") == 2.75
    assert feature_mapping_get(payload, "stats_variability_periodic_feature_period_source") == "ce_corrected_period"
    assert "stats_variability_quasi_periodicity_q" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_method" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_n_points" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_n_bins" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_populated_bins" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_bin_coverage" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_smooth_window_bins" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_template_amplitude" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_raw_scatter" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_resid_scatter" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_scatter_ratio" in _COL_NAMES
    assert "stats_variability_quasi_periodicity_status" in _COL_NAMES
    assert "stats_variability_flux_asymmetry_m" in _COL_NAMES
    assert "stats_variability_periodic_feature_period_days" in _COL_NAMES
    assert "stats_variability_periodic_feature_period_source" in _COL_NAMES

    filter_columns = {entry[1] for _group, entries in SIDEBAR_GROUPS for entry in entries}
    assert "stats_variability_quasi_periodicity_q" in filter_columns
    assert "stats_variability_quasi_periodicity_method" in filter_columns
    assert "stats_variability_quasi_periodicity_n_points" in filter_columns
    assert "stats_variability_quasi_periodicity_n_bins" in filter_columns
    assert "stats_variability_quasi_periodicity_populated_bins" in filter_columns
    assert "stats_variability_quasi_periodicity_bin_coverage" in filter_columns
    assert "stats_variability_quasi_periodicity_smooth_window_bins" in filter_columns
    assert "stats_variability_quasi_periodicity_template_amplitude" in filter_columns
    assert "stats_variability_quasi_periodicity_raw_scatter" in filter_columns
    assert "stats_variability_quasi_periodicity_resid_scatter" in filter_columns
    assert "stats_variability_quasi_periodicity_scatter_ratio" in filter_columns
    assert "stats_variability_quasi_periodicity_status" in filter_columns
    assert "stats_variability_flux_asymmetry_m" in filter_columns
    assert "stats_variability_periodic_feature_period_days" in filter_columns
    assert "stats_variability_periodic_feature_period_source" in filter_columns
