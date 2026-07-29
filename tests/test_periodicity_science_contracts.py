from __future__ import annotations

import numpy as np
import pandas as pd

from malca.core import stats as stats_mod
from malca.core.period_arbitration import period_alias_matches
from malca.core.periodogram import ce_find_period
from malca.stv import filter as post_filter
from malca.stv.periodicity_gate import prepare_periodicity_lightcurve


def test_signal_amplitude_uses_delta_mag_not_absolute_baseline() -> None:
    frame = pd.DataFrame(
        {
            "lc_path": ["small", "large"],
            "baseline_mag": [13.0, 18.0],
            "dip_significant": [True, True],
            "jump_significant": [False, False],
            "dip_best_delta_mag": [0.05, 0.15],
            "jump_best_delta_mag": [-0.04, -0.02],
        }
    )

    out = post_filter.filter_signal_amplitude(frame, min_mag_offset=0.1)

    assert out["lc_path"].tolist() == ["large"]


def test_signal_amplitude_treats_legacy_best_mag_as_residual_offset() -> None:
    frame = pd.DataFrame(
        {
            "lc_path": ["small", "large"],
            "baseline_mag": [13.0, 18.0],
            "dip_significant": [True, True],
            "jump_significant": [False, False],
            "dip_best_mag_event": [0.05, 0.15],
            "jump_best_mag_event": [-0.04, -0.02],
        }
    )

    out = post_filter.filter_signal_amplitude(frame, min_mag_offset=0.1)

    assert out["lc_path"].tolist() == ["large"]


def test_shared_periodicity_cleaning_removes_saturated_and_bad_error_rows() -> None:
    frame = pd.DataFrame(
        {
            "JD": [1.0, 2.0, 3.0, 4.0],
            "mag": [13.0, 13.1, 13.2, 13.3],
            "error": [0.02, 0.02, -1.0, 99.0],
            "saturated": [0, 1, 0, 0],
            "camera#": [1, 1, 2, 2],
        }
    )

    clean = prepare_periodicity_lightcurve(
        frame,
        max_error_absolute=1.0,
        max_error_sigma=10.0,
    )

    assert clean["JD"].tolist() == [1.0]


def test_block_bootstrap_is_deterministic_and_finite_sample_corrected() -> None:
    jd = np.arange(60.0)
    mag = np.sin(2.0 * np.pi * jd / 7.0)

    def metric_finder(times, values, **_kwargs):
        periods = np.asarray([2.0, 3.0])
        metric = np.asarray([np.var(values[::2]), np.var(values[1::2])])
        return float(periods[int(np.argmin(metric))]), periods, metric

    first = stats_mod._bootstrap_min_metric(
        metric_finder,
        jd,
        mag,
        n_bootstrap=12,
        min_period=1.0,
        max_period=5.0,
        n_periods=2,
        random_state=123,
        block_days=3.0,
    )
    second = stats_mod._bootstrap_min_metric(
        metric_finder,
        jd,
        mag,
        n_bootstrap=12,
        min_period=1.0,
        max_period=5.0,
        n_periods=2,
        random_state=123,
        block_days=3.0,
    )

    assert np.array_equal(first, second)
    finite = first[np.isfinite(first)]
    p_value = (np.count_nonzero(finite <= np.min(finite)) + 1) / (finite.size + 1)
    assert p_value >= 1.0 / (finite.size + 1)


def test_lomb_scargle_bootstrap_returns_deterministic_nonzero_p_value() -> None:
    jd = np.linspace(0.0, 90.0, 90)
    mag = 13.0 + 0.2 * np.sin(2.0 * np.pi * jd / 4.7)
    err = np.full(jd.size, 0.03)

    first = stats_mod.bootstrap_lomb_scargle(
        jd,
        mag,
        err,
        n_bootstrap=8,
        min_frequency=1.0 / 10.0,
        max_frequency=1.0,
        random_state=7,
    )
    second = stats_mod.bootstrap_lomb_scargle(
        jd,
        mag,
        err,
        n_bootstrap=8,
        min_frequency=1.0 / 10.0,
        max_frequency=1.0,
        random_state=7,
    )

    assert first["ls_status"] == "ok"
    assert first["ls_bootstrap_sig"] == second["ls_bootstrap_sig"]
    assert first["ls_bootstrap_sig"] >= 1.0 / 9.0


def test_ce_grid_is_uniform_in_frequency() -> None:
    jd = np.linspace(0.0, 20.0, 80)
    mag = np.sin(2.0 * np.pi * jd / 2.3)

    _best, periods, _metric = ce_find_period(
        jd,
        mag,
        min_period=0.5,
        max_period=20.0,
        n_periods=101,
    )

    frequency_steps = np.diff(1.0 / periods)
    assert np.allclose(frequency_steps, frequency_steps[0])


def test_alias_matching_uses_frequency_resolution_when_span_is_known() -> None:
    # A 0.05-day offset around one day is resolved by a 1000-day baseline,
    # even though it fell inside the old fixed 0.1-day window.
    assert period_alias_matches(1.05)
    assert not period_alias_matches(1.05, time_span_days=1000.0)


def test_disabled_postfilter_does_not_retain_stale_failure() -> None:
    frame = pd.DataFrame(
        {
            "lc_path": ["one.dat3"],
            "failed_score": [True],
            "failed_signal_amplitude": [False],
            "failed_any": [True],
        }
    )

    out = post_filter.apply_filters(
        frame,
        apply_evidence_strength=False,
        apply_significant_detection=False,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=False,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
    )

    assert not bool(out.loc[0, "failed_score"])
    assert not bool(out.loc[0, "failed_any"])


def test_catalog_consensus_is_periodic_without_fake_bootstrap_p_value() -> None:
    frame = pd.DataFrame(
        {
            "lc_path": ["catalog-only.dat3"],
            "catalog_match": [True],
            "period_consensus_agree": [True],
            "catalog_period": [3.25],
            "period_primary_source": ["vsx"],
        }
    )

    out = post_filter.validate_periodicity(
        frame,
        n_bootstrap=100,
        skip_if_consensus=True,
        flag_only=True,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "periodic_flag"])
    assert bool(out.loc[0, "periodicity_is_significant"])
    assert np.isnan(out.loc[0, "periodicity_bootstrap_sig"])
    assert out.loc[0, "periodicity_evidence_source"] == "catalog_consensus"
    assert out.loc[0, "periodicity_rejection_reason"] == "catalog_consensus"


def test_band_alignment_refuses_nonoverlapping_epochs() -> None:
    frame = pd.DataFrame(
        {
            "JD": np.concatenate([np.arange(20.0), np.arange(100.0, 120.0)]),
            "mag": np.concatenate([np.full(20, 13.0), np.full(20, 14.0)]),
            "v_g_band": np.concatenate([np.zeros(20), np.ones(20)]),
        }
    )

    aligned, offset, status = stats_mod._align_v_to_g_with_overlap_policy(frame)

    assert aligned.equals(frame)
    assert np.isnan(offset)
    assert status == "not_aligned_no_temporal_overlap"


def test_compute_stats_empty_input_returns_explicit_status() -> None:
    empty = pd.DataFrame(columns=stats_mod._LC_COLUMNS)

    result_frame, summary = stats_mod.compute_stats(
        "empty",
        ".",
        input_frame=empty,
        compute_ls=False,
    )

    assert result_frame.empty
    assert summary["compute_status"] == "insufficient_data"
    assert summary["compute_error"]


def test_q_reports_out_of_fold_evaluation() -> None:
    time = np.linspace(0.0, 100.0, 1000)
    period = 3.7
    mag = 13.0 + 0.2 * np.sin(2.0 * np.pi * time / period)
    err = np.full(time.size, 0.02)

    result = stats_mod.phase_template_quasi_periodicity(
        mag,
        time,
        err,
        period,
        n_phase_bins=50,
        min_bin_points=2,
        min_bin_coverage=0.5,
    )

    assert result["status"] == "ok"
    assert result["evaluation"] == stats_mod.Q_TEMPLATE_EVALUATION
    assert result["n_folds"] >= 2


def test_fourier_coefficients_above_bic_order_are_null() -> None:
    rng = np.random.default_rng(123)
    time = np.sort(rng.uniform(0.0, 80.0, 500))
    period = 2.4
    phase = np.mod(time / period, 1.0)
    mag = 13.0 + 0.3 * np.cos(2.0 * np.pi * phase)
    mag += rng.normal(0.0, 0.01, size=time.size)

    result = stats_mod.fit_fourier_decomposition(
        mag,
        time,
        period,
        err=np.full(time.size, 0.02),
        max_harmonics=7,
    )

    order = int(result["harmonics_order"])
    for harmonic in range(order + 1, 8):
        assert np.isnan(result[f"harmonics_a{harmonic}"])
        assert np.isnan(result[f"harmonics_b{harmonic}"])


def test_drw_fit_uses_positive_ou_rms_and_timescale() -> None:
    rng = np.random.default_rng(44)
    time = np.sort(rng.uniform(0.0, 120.0, 160))
    tau_true = 12.0
    rms_true = 0.2
    values = np.zeros(time.size)
    for idx in range(1, time.size):
        decay = np.exp(-(time[idx] - time[idx - 1]) / tau_true)
        values[idx] = (
            decay * values[idx - 1]
            + rms_true * np.sqrt(max(1.0 - decay**2, 0.0)) * rng.normal()
        )

    rms_fit, tau_fit = stats_mod.fit_drw(
        time,
        values,
        np.full(time.size, 0.03),
    )

    assert np.isfinite(rms_fit) and rms_fit > 0
    assert np.isfinite(tau_fit) and tau_fit > 0
