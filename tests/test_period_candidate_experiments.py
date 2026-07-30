from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.evaluation.period_candidate_methods import (
    _bls_period_groups,
    CandidateScore,
    FIXED_METHODS,
    GLOBAL_METHODS,
    PeriodCandidate,
    PeriodCandidateMethodsConfig,
    event_epoch_detection_diagnostics,
    expand_harmonic_candidates,
    extract_frequency_separated_extrema,
    refine_scored_candidates,
    robust_event_comb_candidates,
    run_global_period_searches,
    run_period_candidate_suite,
    score_candidate_bank,
    score_fixed_period,
)
from malca.evaluation.period_hierarchical_ranker import (
    HierarchicalArbitratorConfig,
    add_harmonic_resolver_features,
    cluster_harmonic_families,
    fit_hierarchical_period_arbitrator,
    score_hierarchical_period_arbitrator,
)
from malca.evaluation.period_candidate_ranker import (
    RankerConfig,
    assign_grouped_splits,
    assign_solution_status,
    default_baseline_feature_specs,
    fit_candidate_ranker,
    label_candidates,
    load_candidate_ranker_artifact,
    method_ablation_summary,
    save_candidate_ranker_artifact,
    score_candidate_ranker,
    select_trial_solutions,
    summarize_recovery,
    sweep_solution_thresholds,
    tune_solution_thresholds,
    validate_feature_allowlist,
)


def _periodic_curve(
    *,
    period_days: float = 37.0,
    n_points: int = 260,
    seed: int = 7,
    morphology: str = "dip",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.sort(
        np.concatenate(([0.0, 4000.0], rng.uniform(0.0, 4000.0, n_points - 2)))
    )
    phase = np.mod(time / period_days, 1.0)
    if morphology == "sine":
        signal = 0.20 * np.sin(2.0 * np.pi * phase)
    else:
        distance = np.abs((phase - 0.22 + 0.5) % 1.0 - 0.5)
        signal = 0.45 * np.exp(-0.5 * np.square(distance / 0.035))
    error = np.full(time.size, 0.02)
    magnitude = 13.0 + signal + rng.normal(0.0, error)
    return time, magnitude, error


def test_requested_global_and_fixed_methods_are_explicit() -> None:
    assert {
        "ls_short",
        "ls_long",
        "multiharmonic_ls_long_2",
        "multiharmonic_ls_long_3",
        "pdm",
        "ce",
        "event_comb",
        "bls_coarse",
        "bls_adaptive",
        "multiharmonic_ls_2",
        "multiharmonic_ls_3",
        "multiharmonic_aov_2",
        "multiharmonic_aov_3",
        "lafler_kinman",
        "supersmoother",
    } == set(GLOBAL_METHODS)
    assert {
        "ls",
        "pdm",
        "ce",
        "bls",
        "multiharmonic_fourier",
        "lafler_kinman",
        "supersmoother",
        "heldout_template",
        "odd_even",
        "event_coherence",
        "alias_evidence",
        "seasonal_stability",
        "null_model_comparison",
    } == set(FIXED_METHODS)
    config = PeriodCandidateMethodsConfig()
    assert not hasattr(config, "n_bootstrap")
    assert config.max_scored_candidates == 128
    assert config.max_refined_candidates == 64


def test_frequency_peak_extraction_separates_physical_families() -> None:
    frequency = np.linspace(0.05, 0.30, 5001)
    metric = (
        np.exp(-0.5 * np.square((frequency - 0.1000) / 0.00035))
        + 0.8 * np.exp(-0.5 * np.square((frequency - 0.1010) / 0.00035))
        + 0.9 * np.exp(-0.5 * np.square((frequency - 0.2200) / 0.00050))
    )
    candidates = extract_frequency_separated_extrema(
        1.0 / frequency,
        metric,
        top_k=3,
        maximize=True,
        baseline_days=4000.0,
        min_frequency_separation=0.003,
        method="test",
    )
    recovered_frequency = np.asarray([item.frequency_per_day for item in candidates])
    assert len(candidates) == 2
    assert np.min(np.abs(recovered_frequency - 0.1000)) < 0.001
    assert np.min(np.abs(recovered_frequency - 0.2200)) < 0.001


def test_harmonic_expansion_contains_full_requested_family() -> None:
    seed = PeriodCandidate(
        method="ls_short",
        period_days=12.0,
        frequency_per_day=1.0 / 12.0,
        raw_score=0.9,
        objective="maximize",
        rank=1,
        normalized_score=1.0,
        contributing_methods=("ls_short",),
    )
    expanded = expand_harmonic_candidates(
        [seed],
        baseline_days=4000.0,
        config=PeriodCandidateMethodsConfig(),
    )
    periods = np.asarray(sorted(item.period_days for item in expanded))
    assert np.allclose(periods, [3.0, 4.0, 6.0, 12.0, 24.0, 36.0, 48.0])


def test_general_search_grid_resolves_the_4000_day_rayleigh_scale() -> None:
    time, magnitude, error = _periodic_curve(
        period_days=37.0,
        n_points=80,
        seed=19,
    )
    config = PeriodCandidateMethodsConfig(
        enabled_global_methods=("ce",),
        short_min_period_days=1.0,
        short_max_period_days=100.0,
        ce_n_frequency=100,
        general_min_samples_per_rayleigh=1.0,
        general_max_frequency_points=10_000,
    )
    result = run_global_period_searches(
        time,
        magnitude,
        error,
        config=config,
    )["ce"]
    expected_floor = int(np.ceil((1.0 - 0.01) * 4000.0)) + 1
    assert len(result.period_grid_days) >= expected_floor
    assert result.metadata["resolved_n_frequency"] >= expected_floor
    assert result.metadata["actual_samples_per_rayleigh"] >= 1.0


def test_long_ls_reserves_candidates_by_cycle_band_and_harmonic_order() -> None:
    time, magnitude, error = _periodic_curve(
        period_days=875.0,
        n_points=260,
        seed=193,
        morphology="sine",
    )
    config = PeriodCandidateMethodsConfig(
        enabled_global_methods=(
            "ls_long",
            "multiharmonic_ls_long_2",
        ),
        long_min_period_days=100.0,
        long_max_baseline_fraction=0.60,
        long_cycle_band_edges=(1.5, 3.0, 8.0, 40.0),
        long_top_k_per_cycle_band=2,
        ls_max_frequency_points=2_000,
    )
    results = run_global_period_searches(
        time,
        magnitude,
        error,
        config=config,
    )
    for method in config.enabled_global_methods:
        result = results[method]
        assert result.status == "ok"
        assert result.metadata["search_strategy"] == (
            "cycle_banded_frequency_quota"
        )
        assert result.metadata["n_cycle_bands"] >= 3
        assert len({candidate.search_band for candidate in result.candidates}) >= 2
    assert any(
        abs(candidate.period_days - 875.0) / 875.0 < 0.08
        for candidate in results["ls_long"].candidates
    )


def test_adaptive_bls_resolves_a_narrow_coarse_seed() -> None:
    period = 19.7
    time, magnitude, error = _periodic_curve(
        period_days=period,
        n_points=420,
        seed=914,
        morphology="dip",
    )
    config = PeriodCandidateMethodsConfig(
        enabled_global_methods=("bls_adaptive",),
        short_min_period_days=5.0,
        short_max_period_days=40.0,
        bls_n_frequency=240,
        bls_adaptive_seed_top_k=8,
        bls_adaptive_top_k=5,
        bls_adaptive_refine_max_frequency_points_per_seed=513,
    )
    result = run_global_period_searches(
        time,
        magnitude,
        error,
        config=config,
    )["bls_adaptive"]
    assert result.status == "ok"
    assert result.metadata["search_strategy"] == (
        "coarse_to_duty_cycle_resolved"
    )
    assert result.metadata["n_refined_frequency_points"] > config.bls_n_frequency
    assert any(
        abs(candidate.period_days - period) / period < 0.02
        for candidate in result.candidates
    )


def test_flat_pdm_and_ce_do_not_create_false_consensus() -> None:
    time = np.linspace(0.0, 4000.0, 100)
    magnitude = np.ones(time.size)
    error = np.full(time.size, 0.02)
    config = PeriodCandidateMethodsConfig(
        enabled_global_methods=("pdm", "ce"),
        pdm_n_frequency=64,
        ce_n_frequency=64,
        general_min_samples_per_rayleigh=0.0,
    )
    results = run_global_period_searches(
        time, magnitude, error, config=config
    )
    assert results["pdm"].status == "flat_metric"
    assert results["ce"].status == "flat_metric"
    assert not results["pdm"].candidates
    assert not results["ce"].candidates


def test_event_epoch_diagnostics_are_one_to_one_and_report_timing_bias() -> None:
    injected = np.asarray([10.0, 20.0, 30.0, 40.0])
    detected = np.asarray([10.1, 19.8, 19.9, 40.2, 55.0])
    result = event_epoch_detection_diagnostics(
        detected,
        injected,
        match_tolerance_days=0.5,
    )
    assert result["event_detection_matched_count"] == 3
    assert result["event_detection_precision"] == pytest.approx(3.0 / 5.0)
    assert result["event_detection_recall"] == pytest.approx(3.0 / 4.0)
    assert np.isfinite(result["event_detection_epoch_bias_days"])


def test_local_refinements_do_not_inflate_independent_support() -> None:
    for global_method, local_method in (
        ("ls_short", "local_ls_refinement"),
        ("bls_coarse", "local_bls_refinement"),
    ):
        score = CandidateScore(
            period_days=10.0,
            frequency_per_day=0.1,
            features={"ls_power": 0.8},
            contributing_methods=(global_method, local_method),
            proposal_method=global_method,
        )
        record = score.to_record()
        assert record["proposal_contributing_method_count"] == 2
        assert record["proposal_independent_method_family_count"] == 1


def test_dense_event_comb_is_capped_and_keeps_the_fundamental() -> None:
    rng = np.random.default_rng(12)
    period = 3.7
    epochs = 50.0 + np.arange(1081) * period
    epochs = epochs + rng.normal(0.0, 0.004 * period, epochs.size)
    config = PeriodCandidateMethodsConfig(
        top_k_event=12,
        event_max_epochs=48,
        event_max_pair_lags=6,
        event_max_seed_hypotheses=400,
    )
    result = robust_event_comb_candidates(
        epochs,
        min_period_days=0.2,
        max_period_days=100.0,
        baseline_days=4000.0,
        config=config,
    )
    assert result.status == "ok"
    assert result.metadata["epochs_truncated"] is True
    assert result.metadata["n_evaluated_seed_hypotheses"] <= 400
    assert any(
        abs(candidate.period_days - period) / period < 0.02
        for candidate in result.candidates
    )


def test_event_comb_rescores_missing_and_outlier_events_on_full_epoch_set() -> None:
    rng = np.random.default_rng(413)
    period = 11.3
    genuine = 25.0 + np.arange(320) * period
    genuine = genuine[rng.random(genuine.size) < 0.72]
    genuine = genuine + rng.normal(0.0, 0.01 * period, genuine.size)
    outliers = rng.uniform(25.0, 25.0 + 319 * period, 24)
    epochs = np.sort(np.concatenate((genuine, outliers)))
    result = robust_event_comb_candidates(
        epochs,
        min_period_days=0.2,
        max_period_days=100.0,
        baseline_days=4000.0,
        config=PeriodCandidateMethodsConfig(
            top_k_event=12,
            event_max_epochs=48,
            event_max_pair_lags=8,
            event_max_seed_hypotheses=500,
            event_min_inlier_fraction=0.60,
        ),
    )
    assert result.status == "ok"
    assert result.metadata["rescored_on_all_events"] is True
    assert result.metadata["n_events_used"] == len(epochs)
    assert any(
        abs(candidate.period_days - period) / period < 0.02
        for candidate in result.candidates
    )


def test_fixed_bls_uses_magnitude_sign_and_locally_refines() -> None:
    time, magnitude, error = _periodic_curve(period_days=37.0, n_points=520)
    config = PeriodCandidateMethodsConfig(
        enabled_fixed_methods=(
            "ls",
            "bls",
            "multiharmonic_fourier",
            "lafler_kinman",
            "heldout_template",
            "odd_even",
            "alias_evidence",
        ),
        fixed_ls_refine_n_frequency=41,
        fixed_bls_refine_n_frequency=21,
    )
    features = score_fixed_period(
        time,
        magnitude,
        36.8,
        error,
        config=config,
    )
    assert features["bls_status"] == "ok"
    assert features["bls_depth"] > 0.0
    assert np.isnan(features["bls_refined_period_days"])
    assert features["bls_exact_power"] == features["bls_power"]
    assert abs(features["ls_local_best_period_days"] - 37.0) / 37.0 < 0.03
    assert features["alias_status"] == "ok"
    candidate = PeriodCandidate(
        method="bls_coarse",
        period_days=36.8,
        frequency_per_day=1.0 / 36.8,
        raw_score=features["bls_power"],
        objective="maximize",
        rank=1,
        normalized_score=1.0,
        contributing_methods=("bls_coarse",),
    )
    score = score_candidate_bank(
        time, magnitude, [candidate], error, config=config
    )
    refinements = refine_scored_candidates(
        [candidate],
        score,
        baseline_days=4000.0,
        time=time,
        mag=magnitude,
        err=error,
        config=config,
    )
    bls_refinement = next(
        item for item in refinements if item.method == "local_bls_refinement"
    )
    assert abs(bls_refinement.period_days - 37.0) / 37.0 < 0.03


def test_narrow_bls_refinement_resolves_half_old_grid_cell() -> None:
    rng = np.random.default_rng(941)
    true_period = 37.0
    duty_cycle = 0.01
    time = np.sort(
        np.concatenate(([0.0, 4000.0], rng.uniform(0.0, 4000.0, 2998)))
    )
    phase = np.mod(time / true_period, 1.0)
    error = np.full(time.size, 0.01)
    magnitude = rng.normal(0.0, error)
    magnitude[phase < duty_cycle] += 0.30
    old_grid_half_cell = (
        0.5 * 4.0 / 30.0 / 4000.0
    )
    seed_frequency = 1.0 / true_period + old_grid_half_cell
    seed_period = 1.0 / seed_frequency
    candidate = PeriodCandidate(
        method="bls_coarse",
        period_days=seed_period,
        frequency_per_day=seed_frequency,
        raw_score=1.0,
        objective="maximize",
        rank=1,
        normalized_score=1.0,
        contributing_methods=("bls_coarse",),
    )
    config = PeriodCandidateMethodsConfig(
        enabled_fixed_methods=("bls",),
        local_refinement_sources=("bls",),
        fixed_bls_max_refinement_seeds=1,
    )
    scores = score_candidate_bank(
        time, magnitude, [candidate], error, config=config
    )
    refinements = refine_scored_candidates(
        [candidate],
        scores,
        baseline_days=4000.0,
        time=time,
        mag=magnitude,
        err=error,
        config=config,
    )
    refined = next(
        item for item in refinements if item.method == "local_bls_refinement"
    )
    frequency_error_rayleigh = (
        abs(refined.frequency_per_day - 1.0 / true_period) * 4000.0
    )
    assert frequency_error_rayleigh < duty_cycle / 2.0


def test_bls_period_groups_preserve_duration_fraction_scale() -> None:
    frequency = np.linspace(0.01, 1.0, 2000)
    periods = 1.0 / frequency
    groups = _bls_period_groups(
        periods,
        n_groups=12,
        max_period_ratio=1.15,
    )
    assert len(groups) > 12
    for group in groups:
        group_periods = periods[group]
        assert (
            float(np.max(group_periods) / np.min(group_periods))
            <= 1.15 * (1.0 + 1.0e-12)
        )


def test_event_features_separate_fixed_and_refined_periods_and_unique_epochs() -> None:
    time, magnitude, error = _periodic_curve(
        period_days=10.0,
        n_points=180,
        seed=28,
    )
    config = PeriodCandidateMethodsConfig(
        enabled_fixed_methods=("event_coherence",),
    )
    duplicate = score_fixed_period(
        time,
        magnitude,
        10.0,
        error,
        event_epochs=(100.0, 100.0),
        config=config,
    )
    assert duplicate["event_status"] == "insufficient_events"
    assert duplicate["event_n_events"] == 1
    assert np.isnan(duplicate["event_score"])

    epochs = 50.0 + np.arange(30) * 10.0
    features = score_fixed_period(
        time,
        magnitude,
        9.8,
        error,
        event_epochs=epochs,
        config=config,
    )
    assert features["event_exact_period_days"] == 9.8
    assert features["event_evidence_enabled"] == 0
    assert np.isnan(features["event_score"])
    assert np.isfinite(features["event_exact_score"])
    assert abs(features["event_local_refined_period_days"] - 10.0) < 1.0e-8
    assert features["event_refined_period_days"] == features[
        "event_local_refined_period_days"
    ]
    coherent = score_fixed_period(
        time,
        magnitude,
        10.0,
        error,
        event_epochs=epochs,
        config=config,
    )
    assert coherent["event_evidence_enabled"] == 1
    assert coherent["event_score"] == coherent["event_exact_score"]


def test_event_features_are_gated_and_seasonal_null_evidence_is_materialized() -> None:
    time, magnitude, error = _periodic_curve(
        period_days=31.0,
        n_points=360,
        seed=281,
        morphology="sine",
    )
    config = PeriodCandidateMethodsConfig(
        enabled_fixed_methods=(
            "event_coherence",
            "seasonal_stability",
            "null_model_comparison",
        ),
    )
    two_event = score_fixed_period(
        time,
        magnitude,
        31.0,
        error,
        event_epochs=(50.0, 81.0),
        config=config,
    )
    assert two_event["event_status"] == "insufficient_coherent_events"
    assert two_event["event_evidence_enabled"] == 0
    assert np.isnan(two_event["event_score"])
    assert np.isfinite(two_event["event_raw_score"])

    many_event = score_fixed_period(
        time,
        magnitude,
        31.0,
        error,
        event_epochs=50.0 + np.arange(20) * 31.0,
        config=config,
    )
    assert many_event["event_status"] == "ok"
    assert many_event["event_evidence_enabled"] == 1
    assert many_event["stability_status"] == "ok"
    assert many_event["stability_valid_segments"] >= 2
    assert many_event["null_model_status"] == "ok"
    assert np.isfinite(many_event["null_periodic_vs_linear_delta_bic"])


def test_odd_even_requires_multiple_cycles_per_parity() -> None:
    time = np.linspace(0.0, 4000.0, 120)
    magnitude = np.sin(2.0 * np.pi * time / 3000.0)
    error = np.full(time.size, 0.02)
    features = score_fixed_period(
        time,
        magnitude,
        3000.0,
        error,
        config=PeriodCandidateMethodsConfig(
            enabled_fixed_methods=("odd_even",)
        ),
    )
    assert features["odd_even_status"] == "insufficient_cycles"
    assert np.isnan(features["odd_even_shape_rms"])


def test_long_period_refinement_stays_inside_candidate_bounds() -> None:
    time = np.linspace(0.0, 4000.0, 180)
    magnitude = 13.0 + 0.15 * np.square((time - 2000.0) / 2000.0)
    error = np.full(time.size, 0.02)
    candidate = PeriodCandidate(
        method="ls_long",
        period_days=3000.0,
        frequency_per_day=1.0 / 3000.0,
        raw_score=0.5,
        objective="maximize",
        rank=1,
        normalized_score=1.0,
        contributing_methods=("ls_long",),
    )
    config = PeriodCandidateMethodsConfig(
        enabled_fixed_methods=("ls",),
        local_refinement_sources=("ls",),
        fixed_ls_refine_n_frequency=31,
    )
    scores = score_candidate_bank(time, magnitude, [candidate], error, config=config)
    refinements = refine_scored_candidates(
        [candidate],
        scores,
        baseline_days=4000.0,
        config=config,
    )
    assert refinements
    assert all(0.1 <= item.period_days <= 3200.0 for item in refinements)


def test_supersmoother_explained_fraction_uses_magnitude_residuals() -> None:
    pytest.importorskip("supersmoother")
    time, magnitude, error = _periodic_curve(
        period_days=19.0,
        n_points=260,
        seed=44,
        morphology="sine",
    )
    features = score_fixed_period(
        time,
        magnitude,
        19.0,
        error,
        config=PeriodCandidateMethodsConfig(
            enabled_fixed_methods=("supersmoother",),
        ),
    )
    assert features["supersmoother_status"] == "ok"
    assert features["supersmoother_cv_mse"] < 0.1
    assert features["supersmoother_cv_standardized_mse"] > features[
        "supersmoother_cv_mse"
    ]
    assert features["supersmoother_explained_fraction"] > -5.0


def test_full_candidate_suite_caps_scoring_and_materializes_refinements() -> None:
    period = 23.4
    time, magnitude, error = _periodic_curve(
        period_days=period,
        n_points=220,
        seed=18,
        morphology="sine",
    )
    epochs = np.arange(40.0, 4000.0, period)
    config = PeriodCandidateMethodsConfig(
        short_min_period_days=1.0,
        short_max_period_days=80.0,
        long_min_period_days=10.0,
        long_max_baseline_fraction=0.30,
        ls_max_frequency_points=600,
        pdm_n_frequency=180,
        ce_n_frequency=180,
        bls_n_frequency=100,
        bls_adaptive_seed_top_k=2,
        bls_adaptive_top_k=2,
        bls_adaptive_refine_max_frequency_points_per_seed=129,
        multiharmonic_n_frequency=180,
        aov_n_frequency=180,
        lafler_kinman_n_frequency=180,
        supersmoother_n_frequency=24,
        fixed_ls_refine_n_frequency=21,
        fixed_bls_refine_n_frequency=9,
        top_k_ls=2,
        top_k_general=2,
        top_k_event=3,
        max_scored_candidates=8,
        max_refined_candidates=8,
        event_max_seed_hypotheses=200,
        general_min_samples_per_rayleigh=0.0,
    )
    result = run_period_candidate_suite(
        time,
        magnitude,
        error,
        event_epochs=epochs,
        config=config,
    )
    assert result.status == "ok"
    assert set(result.search_results) == set(GLOBAL_METHODS)
    assert all(
        search.status == "ok"
        or (
            method == "supersmoother"
            and search.status == "unavailable"
        )
        for method, search in result.search_results.items()
    )
    assert len(result.candidate_scores) == 8
    assert sum(candidate.was_scored for candidate in result.expanded_candidates) == 8
    scored_candidates = [
        candidate for candidate in result.expanded_candidates if candidate.was_scored
    ]
    refinements = refine_scored_candidates(
        scored_candidates,
        result.candidate_scores,
        baseline_days=4000.0,
        time=time,
        mag=magnitude,
        err=error,
        config=config,
    )
    assert refinements
    assert all(candidate.method.startswith("local_") for candidate in refinements)
    assert any(
        abs(candidate.period_days - period) / period < 0.03
        for candidate in (*result.expanded_candidates, *refinements)
    )
    feature_row = result.candidate_records()[0]
    assert feature_row["was_scored"] == 1
    assert "proposal_contributing_method_count" in feature_row
    assert "proposal_independent_method_family_count" in feature_row
    assert "ls_local_best_period_days" in feature_row
    assert "bls_refined_period_days" in feature_row


def _synthetic_ranker_frame(n_groups: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    rows: list[dict[str, object]] = []
    for group_index in range(n_groups):
        is_signal = group_index % 3 != 0
        truth = 5.0 + 0.4 * group_index if is_signal else np.nan
        base_id = f"base-{group_index:04d}"
        view_id = f"{base_id}::none"
        if is_signal:
            periods = (truth, 0.5 * truth, 1.7 * truth, 2.0 * truth)
        else:
            periods = tuple(rng.uniform(2.0, 100.0, size=4))
        for candidate_index, candidate_period in enumerate(periods):
            strong = bool(is_signal and candidate_index == 0)
            rows.append(
                {
                    "base_trial_id": base_id,
                    "base_view_id": view_id,
                    "candidate_id": f"{view_id}:{candidate_index}",
                    "period_days": candidate_period,
                    "true_period_days": truth,
                    "proposal_method": "ls_short" if candidate_index < 2 else "pdm",
                    "proposal_rank": candidate_index + 1,
                    "proposal_normalized_score": (
                        0.92 if strong else 0.15 + 0.08 * rng.random()
                    ),
                    "ls_power": 0.95 if strong else 0.10 + 0.10 * rng.random(),
                    "pdm_theta": 0.12 if strong else 0.75 + 0.10 * rng.random(),
                    "ce_entropy": 0.18 if strong else 0.70 + 0.12 * rng.random(),
                    "event_score": np.nan,
                }
            )
    return label_candidates(pd.DataFrame(rows))


def test_harmonic_family_clustering_and_pairwise_resolver_features() -> None:
    frame = pd.DataFrame(
        {
            "base_trial_id": ["base"] * 4,
            "base_view_id": ["base::none"] * 4,
            "candidate_id": ["half", "exact", "double", "other"],
            "period_days": [5.0, 10.0, 20.0, 17.0],
            "baseline_days": [4000.0] * 4,
            "ls_power": [0.3, 0.9, 0.4, 0.2],
        }
    )
    clustered = cluster_harmonic_families(frame)
    family_by_candidate = clustered.set_index("candidate_id")[
        "harmonic_family_id"
    ]
    assert family_by_candidate["half"] == family_by_candidate["exact"]
    assert family_by_candidate["exact"] == family_by_candidate["double"]
    assert family_by_candidate["other"] != family_by_candidate["exact"]

    resolver, generated = add_harmonic_resolver_features(
        clustered,
        candidate_feature_columns=("period_days", "ls_power"),
    )
    exact = resolver.loc[resolver["candidate_id"].eq("exact")].iloc[0]
    assert exact["support_resolver_has_half"] == 1
    assert exact["support_resolver_has_double"] == 1
    assert exact["support_resolver_ls_power_minus_half"] == pytest.approx(0.6)
    assert "support_resolver_ls_power_minus_double" in generated


def test_hierarchical_arbitrator_fits_three_disjoint_models() -> None:
    pytest.importorskip("lightgbm")
    frame = _synthetic_ranker_frame(n_groups=90)
    frame["baseline_days"] = 4000.0
    frame["baseline_cycles"] = (
        frame["baseline_days"] / frame["period_days"]
    )
    frame["split"] = np.where(
        frame["base_trial_id"].str[-4:].astype(int) < 60,
        "train",
        "calibration",
    )
    frame["event_mode"] = "detected"
    frame["morphology"] = np.where(
        frame["truth_is_periodic"],
        "sinusoid",
        np.where(
            frame["base_trial_id"].str[-4:].astype(int) % 2,
            "seasonal_trend",
            "red_noise",
        ),
    )
    frame["nuisance_mode"] = "none"
    train = frame.loc[frame["split"].eq("train")].copy()
    calibration = frame.loc[frame["split"].eq("calibration")].copy()
    features = (
        "period_days",
        "baseline_cycles",
        "proposal_rank",
        "proposal_normalized_score",
        "ls_power",
        "pdm_theta",
        "ce_entropy",
    )
    config = HierarchicalArbitratorConfig(
        n_estimators=45,
        learning_rate=0.08,
        min_child_samples=2,
        calibration_method="binned",
        n_jobs=1,
    )
    artifact = fit_hierarchical_period_arbitrator(
        train,
        calibration,
        candidate_feature_columns=features,
        config=config,
    )
    assert artifact.family_ranker.model_kind == "ranker"
    assert artifact.resolver_ranker.model_kind == "ranker"
    assert artifact.acceptance_model.model_kind == "classifier"
    assert artifact.metadata["calibration_partition_role"] == "calibration_only"
    assert artifact.acceptance_model.config.sample_weight_col == (
        "training_sample_weight"
    )

    scored = score_hierarchical_period_arbitrator(
        calibration,
        artifact,
    )
    assert len(scored.solutions) == calibration["base_view_id"].nunique()
    assert scored.solutions["base_view_id"].is_unique
    assert {
        "acceptance_probability",
        "family_probability",
        "exact_probability",
        "harmonic_ambiguity_probability",
        "solution_status",
    }.issubset(scored.solutions)
    periodic = scored.solutions.loc[
        scored.solutions["truth_is_periodic"].astype(bool)
    ]
    assert float(periodic["is_harmonic_family"].mean()) >= 0.8


def test_rayleigh_labels_are_principal_and_legacy_relative_labels_are_retained() -> None:
    candidates = pd.DataFrame(
        {
            "period_days": [1.05, 2300.0, 2050.0],
            "true_period_days": [1.0, 2000.0, 2000.0],
        }
    )
    labeled = label_candidates(
        candidates,
        tolerance=0.10,
        exact_tolerance=0.10,
        baseline_days=4000.0,
        rayleigh_tolerance=1.0,
    )
    assert labeled["is_exact_relative"].tolist() == [True, False, True]
    assert labeled["is_exact_resolution_consistent"].tolist() == [
        False,
        True,
        True,
    ]
    assert labeled["is_exact"].tolist() == [False, False, True]
    assert labeled["is_wrong_harmonic"].equals(labeled["is_harmonic_only"])
    assert labeled["truth_match_criterion"].eq(
        "rayleigh_frequency_and_relative_period"
    ).all()
    assert labeled.loc[0, "truth_exact_rayleigh_error"] > 100.0
    assert labeled.loc[1, "truth_exact_rayleigh_error"] < 1.0


def test_default_baseline_has_explicit_scientific_directions() -> None:
    frame = pd.DataFrame(
        {
            "period_days": [10.0],
            "proposal_rank": [1],
            "proposal_normalized_score": [0.8],
            "ls_power": [0.7],
            "fourier_2_bic": [12.0],
            "pdm_theta": [0.2],
            "event_rms_oc_days": [0.1],
            "bls_transit_time": [123.0],
        }
    )
    specs = {spec.column: spec for spec in default_baseline_feature_specs(frame)}
    assert set(specs) == {
        "proposal_normalized_score",
        "ls_power",
        "pdm_theta",
        "event_rms_oc_days",
    }
    assert specs["ls_power"].higher_is_better is True
    assert specs["pdm_theta"].higher_is_better is False
    assert specs["event_rms_oc_days"].higher_is_better is False


def test_ranker_is_group_safe_calibrated_and_threshold_constrained(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lightgbm")
    frame = _synthetic_ranker_frame()
    base_assignment = assign_grouped_splits(
        frame[["base_trial_id"]].drop_duplicates(),
        group_col="base_trial_id",
        seed=91,
        train_fraction=0.65,
        validation_fraction=0.35,
        test_fraction=0.0,
    )
    frame = frame.merge(base_assignment, on="base_trial_id", validate="many_to_one")
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("validation")].copy()
    assert set(train["base_trial_id"]).isdisjoint(validation["base_trial_id"])

    config = RankerConfig(
        group_col="base_view_id",
        feature_columns=(
            "period_days",
            "proposal_method",
            "proposal_rank",
            "proposal_normalized_score",
            "ls_power",
            "pdm_theta",
            "ce_entropy",
            "event_score",
        ),
        categorical_features=("proposal_method",),
        n_estimators=35,
        min_child_samples=2,
        calibration_method="binned",
        min_calibration_samples=8,
        n_jobs=1,
    )
    artifact = fit_candidate_ranker(train, validation, config=config)
    artifact_path = tmp_path / "candidate_ranker.joblib"
    save_candidate_ranker_artifact(artifact, artifact_path)
    loaded_artifact = load_candidate_ranker_artifact(artifact_path)
    assert (
        loaded_artifact.metadata["artifact_fingerprint"]
        == artifact.metadata["artifact_fingerprint"]
    )
    scored = score_candidate_ranker(validation, artifact)
    loaded_scored = score_candidate_ranker(validation, loaded_artifact)
    assert np.allclose(
        scored["ranker_score_raw"],
        loaded_scored["ranker_score_raw"],
    )
    untuned = select_trial_solutions(scored, artifact, group_col="base_view_id")
    tuned = tune_solution_thresholds(
        untuned,
        artifact,
        group_col="base_view_id",
        target_null_accepted_rate=0.05,
        target_harmonic_only_rate=0.05,
        minimum_exact_recovery_rate=0.80,
        minimum_family_recovery_rate=0.95,
        require_feasible=True,
    )
    solutions = select_trial_solutions(
        scored,
        artifact,
        group_col="base_view_id",
        status_thresholds=tuned,
    )
    summary = summarize_recovery(solutions, group_col="base_view_id").iloc[0]
    assert tuned.constraints_satisfied
    assert summary["null_accepted_rate"] <= 0.05
    assert summary["harmonic_only_rate"] <= 0.05
    assert summary["exact_recovery_rate"] > 0.80
    assert summary["family_recovery_rate"] >= 0.95
    assert isinstance(artifact.metadata["calibrators"]["candidate"], dict)
    assert "method" in artifact.metadata["calibrators"]["candidate"]
    expected = validation[
        ["base_view_id", "truth_is_periodic"]
    ].drop_duplicates()
    expected = pd.concat(
        (
            expected,
            pd.DataFrame(
                {
                    "base_view_id": ["candidate-less"],
                    "truth_is_periodic": [True],
                }
            ),
        ),
        ignore_index=True,
    )
    complete_solutions = select_trial_solutions(
        scored,
        artifact,
        group_col="base_view_id",
        status_thresholds=tuned,
        expected_trials=expected,
    )
    missing_solution = complete_solutions.loc[
        complete_solutions["base_view_id"].eq("candidate-less")
    ].iloc[0]
    assert bool(missing_solution["no_candidate"])
    assert missing_solution["solution_status"] == "abstain"
    assert len(complete_solutions) == len(expected)
    assert set(solutions["solution_status"]).issubset(
        {
            "secure_fundamental",
            "secure_family_harmonic_ambiguous",
            "tentative",
            "abstain",
        }
    )


def test_ranker_statuses_leakage_guard_and_json_method_ablation() -> None:
    status = assign_solution_status(
        [0.95, 0.95, 0.50, 0.10],
        [0.95, 0.20, 0.20, 0.10],
        [0.95, 0.95, 0.40, 0.10],
    )
    assert status.tolist() == [
        "secure_fundamental",
        "secure_family_harmonic_ambiguous",
        "tentative",
        "abstain",
    ]
    frame = pd.DataFrame(
        {
            "base_view_id": ["a", "b"],
            "true_period_days": [10.0, 11.0],
            "ls_power": [0.9, 0.8],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_feature_allowlist(
            frame,
            ("ls_power", "true_period_days"),
            group_col="base_view_id",
        )
    for leaked_name in (
        "is_signal",
        "sampling_density",
        "relative_period_error",
        "rayleigh_frequency_error",
        "bls_exact_transit_time",
        "event_exact_period_days",
        "event_local_refined_period_days",
    ):
        leaked = frame.assign(**{leaked_name: [1.0, 2.0]})
        with pytest.raises(ValueError, match="leakage"):
            validate_feature_allowlist(
                leaked,
                ("ls_power", leaked_name),
                group_col="base_view_id",
            )

    candidates = pd.DataFrame(
        {
            "base_view_id": ["a", "a", "b"],
            "truth_is_periodic": [True, True, True],
            "is_exact": [True, False, True],
            "is_harmonic_family": [True, True, True],
            "contributing_methods": [
                '["ls_short", "pdm"]',
                '["ce"]',
                '["pdm"]',
            ],
        }
    )
    ablation = method_ablation_summary(
        candidates,
        method_col="contributing_methods",
        group_col="base_view_id",
    )
    assert {"without_ls_short", "without_pdm", "without_ce"}.issubset(
        set(ablation["ablation"])
    )


def test_threshold_constraints_can_require_wilson_confidence_bounds() -> None:
    def perfect_solutions(n_periodic: int, n_null: int) -> pd.DataFrame:
        periodic = np.r_[
            np.ones(n_periodic, dtype=bool),
            np.zeros(n_null, dtype=bool),
        ]
        return pd.DataFrame(
            {
                "base_trial_id": [
                    f"trial-{index}" for index in range(len(periodic))
                ],
                "truth_is_periodic": periodic,
                "is_exact": periodic,
                "is_harmonic_family": periodic,
                "acceptance_probability": np.where(periodic, 0.99, 0.01),
                "exact_probability": np.where(periodic, 0.99, 0.01),
                "family_probability": np.where(periodic, 0.99, 0.01),
            }
        )

    common = {
        "secure_acceptance_grid": (0.5,),
        "secure_exact_grid": (0.5,),
        "secure_family_grid": (0.5,),
        "target_null_accepted_rate": 0.05,
        "target_harmonic_only_rate": 0.05,
        "minimum_exact_recovery_rate": 0.95,
        "minimum_family_recovery_rate": 0.95,
        "minimum_periodic_coverage": 0.95,
        "use_confidence_bounds_for_constraints": True,
        "constraint_confidence_level": 0.95,
    }
    small = sweep_solution_thresholds(perfect_solutions(20, 20), **common)
    assert not bool(small.iloc[0]["constraints_satisfied"])
    assert small.iloc[0]["null_accepted_rate"] == 0.0
    assert small.iloc[0]["null_accepted_ci_high"] > 0.05

    large = sweep_solution_thresholds(perfect_solutions(100, 100), **common)
    assert bool(large.iloc[0]["constraints_satisfied"])
    assert large.iloc[0]["exact_recovery_ci_low"] > 0.95
    assert large.iloc[0]["null_accepted_ci_high"] < 0.05


def test_period_injection_notebook_code_cells_compile() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "malca"
        / "notebooks"
        / "evaluation"
        / "period_arbitration_injection_recovery.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")
    joined_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "MM_BASELINE_DAYS = 4000.0" in joined_source
    assert "MM_WORKERS = 8" in joined_source
    assert "MALCA_MM_RUN" in joined_source
    assert "dict(score.to_record())" in joined_source
    assert "difference = comparison_values - none_values" in joined_source
    assert "error[baseline_anchor] = median_error" in joined_source
    assert "baseline_anchor_error_v1" in joined_source
    assert "original_checkpoint_parts_remain_immutable" in joined_source
    assert "unexpected_pending" in joined_source
    assert "_notebook_code_sha256" in joined_source
    assert "_legacy_commit_ledger_sha256" in joined_source
    assert "Foreign checkpoint source fingerprint" in joined_source
    assert "Recovery commit exceeds the declared retry scope" in joined_source
    assert "recovery_selected_views" in joined_source
