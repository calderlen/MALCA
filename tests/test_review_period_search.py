from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.review import period_search


def _band_dfs() -> dict[int, pd.DataFrame]:
    jd = np.linspace(2458000.0, 2458100.0, 40)
    resid = np.sin(np.linspace(0.0, 4.0 * np.pi, 40))
    return {
        0: pd.DataFrame({"JD": jd, "resid": resid}),
        1: pd.DataFrame({"JD": jd + 0.1, "resid": resid * 0.8}),
    }


def test_pipeline_recompute_applies_window_to_every_consensus_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "JD": np.linspace(2458000.0, 2461700.0, 60),
            "mag": np.linspace(12.5, 12.7, 60),
            "error": np.full(60, 0.02),
        }
    )
    consensus_calls: list[dict] = []

    monkeypatch.setattr(
        period_search,
        "_prepare_pipeline_periodicity_frame",
        lambda *args, **kwargs: frame,
    )
    monkeypatch.setattr(
        "malca.core.stats.compute_pdm_stats",
        lambda *args, **kwargs: {"pdm_period": 4.25},
    )
    monkeypatch.setattr(
        "malca.core.stats.compute_ce_stats",
        lambda *args, **kwargs: {"ce_period": 4.25},
    )
    monkeypatch.setattr(
        "malca.stv.event_period.event_based_period",
        lambda *args, **kwargs: {"event_period_days": 358.689},
    )

    def fake_consensus(*args, **kwargs):
        consensus_calls.append(kwargs)
        return {
            "period_consensus_days": 4.25,
            "period_method": "pdm+ce",
            "period_confidence": "tentative",
            "long_ls_period_days": 4.25,
            "long_ls_is_significant": False,
        }

    monkeypatch.setattr(
        "malca.core.period_pipeline.compute_period_consensus_for_lc",
        fake_consensus,
    )

    result, _message = period_search.run_pipeline_period_search_for_payload(
        {"dip_run_epochs_json": ""},
        plot_dir=None,
        min_period=0.1,
        max_period=10.0,
    )

    assert result is not None
    assert result["best_period"] == pytest.approx(4.25)
    assert consensus_calls[0]["min_period_days"] == pytest.approx(0.1)
    assert consensus_calls[0]["max_period_days"] == pytest.approx(10.0)
    assert consensus_calls[0]["long_ls_kwargs"] == {
        "n_bootstrap": 0,
        "min_period_days": pytest.approx(0.1),
        "max_period_days": pytest.approx(10.0),
    }


def test_pipeline_recompute_exposes_weak_in_window_pdm_for_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "JD": np.linspace(2458000.0, 2461700.0, 60),
            "mag": np.linspace(12.5, 12.7, 60),
            "error": np.full(60, 0.02),
        }
    )
    monkeypatch.setattr(
        period_search,
        "_prepare_pipeline_periodicity_frame",
        lambda *args, **kwargs: frame,
    )
    monkeypatch.setattr(
        "malca.core.stats.compute_pdm_stats",
        lambda *args, **kwargs: {
            "pdm_period": 3.00099,
            "pdm_snr": 16.8,
            "pdm_min_theta": 0.84,
        },
    )
    monkeypatch.setattr(
        "malca.core.stats.compute_ce_stats",
        lambda *args, **kwargs: {
            "ce_period": 0.49926,
            "ce_snr": 5.8,
            "ce_min_entropy": 1.66,
        },
    )
    monkeypatch.setattr(
        "malca.core.period_pipeline.compute_period_consensus_for_lc",
        lambda *args, **kwargs: {
            "period_consensus_days": np.nan,
            "period_method": "none",
            "period_confidence": "none",
            "long_ls_period_days": 358.689,
            "long_ls_is_significant": False,
        },
    )

    result, message = period_search.run_pipeline_period_search_for_payload(
        {},
        plot_dir=None,
        min_period=0.1,
        max_period=10.0,
    )

    assert result is not None
    assert result["best_period"] == pytest.approx(3.00099)
    assert result["period_method"] == "pdm_review_candidate"
    assert "below consensus significance gates" in message


def test_weak_review_candidate_never_escapes_selected_window() -> None:
    period, method = period_search._bounded_review_period_candidate(
        {"pdm_period": 358.689},
        {"ce_period": 0.49926},
        min_period=0.1,
        max_period=10.0,
    )

    assert period == pytest.approx(0.49926)
    assert method == "ce"


def test_harmonic_check_candidate_periods_returns_base_and_divisors() -> None:
    candidates = period_search._harmonic_check_candidate_periods(8.0, min_period=0.1, max_period=10.0)

    assert [candidate["divisor"] for candidate in candidates] == [1.0, 2.0, 3.0, 4.0]
    assert [candidate["period"] for candidate in candidates] == pytest.approx([8.0, 4.0, 8.0 / 3.0, 2.0])


def test_harmonic_check_candidate_periods_ignores_out_of_bounds_divisors() -> None:
    candidates = period_search._harmonic_check_candidate_periods(8.0, min_period=3.0, max_period=6.0)

    assert [candidate["divisor"] for candidate in candidates] == [2.0]
    assert [candidate["period"] for candidate in candidates] == pytest.approx([4.0])


def test_harmonic_check_prefers_shortest_comparable_period(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {
        8.0: 1.0,
        4.0: 0.99,
        8.0 / 3.0: 0.98,
        2.0: 0.99,
    }

    def fake_score(_band_resid, period, *, alias_penalty=0.0, event_epochs=None):
        score = scores[float(period)]
        return {
            "objective": score,
            "raw_objective": score,
            "scatter_ratio": score,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(period_search, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, diag = period_search.check_stored_period_harmonics(
        _band_dfs(),
        8.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert selected_period == pytest.approx(2.0)
    assert factor == pytest.approx(0.25)
    assert diag["harmonic_divisor"] == pytest.approx(4.0)


def test_harmonic_check_keeps_base_when_it_is_clearly_better(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {
        8.0: 1.0,
        4.0: 1.04,
        8.0 / 3.0: 1.04,
        2.0: 1.04,
    }

    def fake_score(_band_resid, period, *, alias_penalty=0.0, event_epochs=None):
        score = scores[float(period)]
        return {
            "objective": score,
            "raw_objective": score,
            "scatter_ratio": score,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(period_search, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, diag = period_search.check_stored_period_harmonics(
        _band_dfs(),
        8.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert selected_period == pytest.approx(8.0)
    assert factor == pytest.approx(1.0)
    assert diag["harmonic_divisor"] == pytest.approx(1.0)


def test_harmonic_check_returns_nan_when_no_scores_are_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_score(_band_resid, period, *, alias_penalty=0.0, event_epochs=None):
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "lag_phase": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(period_search, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, diag = period_search.check_stored_period_harmonics(
        _band_dfs(),
        8.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert np.isnan(selected_period)
    assert factor == pytest.approx(1.0)
    assert diag["candidates"]


def test_harmonic_selection_keeps_cadence_alias_flag_out_of_ranking() -> None:
    selected = period_search._select_harmonic_check_candidate(
        [
            {"period": 4.0, "selection_objective": 1.0, "alias_flag": False, "alias_matches": []},
            {"period": 1.0, "selection_objective": 1.0, "alias_flag": True, "alias_matches": [1.0]},
        ]
    )

    assert selected is not None
    assert selected["period"] == pytest.approx(1.0)
    assert selected["alias_flag"] is True
    assert selected["alias_matches"] == [1.0]


def test_find_period_arbitration_evaluates_multiples_but_keeps_base_without_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated_periods: list[float] = []

    def fake_score(_band_resid, period, *, alias_penalty=0.2):
        evaluated_periods.append(float(period))
        return {
            "objective": 1.0,
            "raw_objective": 1.0,
            "scatter_ratio": 1.0,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(period_search, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _diag = period_search.arbitrate_harmonic_period(
        _band_dfs(),
        2.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert selected_period == pytest.approx(2.0)
    assert factor == pytest.approx(1.0)
    assert evaluated_periods == pytest.approx([2.0, 1.0, 2.0 / 3.0, 0.5, 4.0, 6.0, 8.0])
    assert any(period > 2.0 for period in evaluated_periods)


def test_find_period_arbitration_rejects_small_upward_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(_band_resid, period, *, alias_penalty=0.2):
        if float(period) > 2.0:
            score = 0.95
        elif np.isclose(float(period), 2.0):
            score = 1.0
        else:
            score = 1.05
        return {
            "objective": score,
            "raw_objective": score,
            "scatter_ratio": score,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(period_search, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, diag = period_search.arbitrate_harmonic_period(
        _band_dfs(),
        2.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert selected_period == pytest.approx(2.0)
    assert factor == pytest.approx(1.0)
    assert diag["base_objective"] == pytest.approx(1.0)


def test_find_period_arbitration_can_promote_to_clear_longer_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(_band_resid, period, *, alias_penalty=0.2):
        if np.isclose(float(period), 4.0):
            score = 0.1
        elif np.isclose(float(period), 2.0):
            score = 1.0
        else:
            score = 1.05
        return {
            "objective": score,
            "raw_objective": score,
            "scatter_ratio": score,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(period_search, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, diag = period_search.arbitrate_harmonic_period(
        _band_dfs(),
        2.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert selected_period == pytest.approx(4.0)
    assert factor == pytest.approx(2.0)
    assert diag["upward_multiple_flag"] is True


def test_review_stored_period_ignores_raw_periodogram_outputs() -> None:
    payload = {
        "candidate_id": "cand-1",
        "periodicity_period": 1.5,
        "lsp_period": 2.0,
        "pdm_period": 3.0,
        "ce_period": 4.0,
        "stats_variability_lomb_scargle_best_period_days": 5818.14746,
        "vsx_period": 9.0,
    }

    assert period_search.has_external_period(payload) is False
    assert period_search.resolve_stored_review_period(payload) == (None, "")


def test_review_stored_period_keeps_authoritative_payload_periods() -> None:
    assert period_search.has_external_period({"period_consensus_days": 8.0}) is True
    assert period_search.resolve_stored_review_period({"period_consensus_days": 8.0}) == (
        8.0,
        "period_consensus_days",
    )
    assert period_search.resolve_stored_review_period({"pre_periodicity_selected_period": 6.0}) == (
        6.0,
        "pre_periodicity_selected_period",
    )


def test_review_stored_period_prefers_long_ls_when_short_stored() -> None:
    payload = {
        "phase_period_days": 6.0,
        "long_ls_period_days": 1857.0,
        "long_ls_is_significant": True,
    }
    period, source = period_search.resolve_stored_review_period(payload)
    assert period == pytest.approx(1857.0)
    assert source == "long_ls_period_days"


def test_adaptive_review_period_bounds_scales_with_baseline() -> None:
    lo, hi = period_search.adaptive_review_period_bounds({"stats_time_span_days": 4000.0})
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(2000.0)


def test_harmonic_check_keeps_long_ls_fundamental_with_sparse_coverage() -> None:
    payload = {
        "long_ls_period_days": 2222.29,
        "long_ls_is_significant": True,
        "stats_time_span_days": 3704.0,
    }
    period, source = period_search.resolve_stored_review_period(payload)
    assert period == pytest.approx(2222.29)
    assert source == "long_ls_period_days"
    assert period_search._keep_long_ls_fundamental_for_review(payload, period, source)

    result, summary = period_search.run_harmonic_check_for_payload(
        payload,
        plot_dir=None,
        min_period=0.1,
        max_period=1852.0,
    )
    assert result is not None
    assert result["best_period"] == pytest.approx(2222.29)
    assert result["harmonic_divisor"] == 1.0
    assert "kept P=" in summary
    assert "2222" in summary
