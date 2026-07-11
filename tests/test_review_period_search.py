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

    def fake_score(_band_resid, period, *, alias_penalty=0.0):
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

    def fake_score(_band_resid, period, *, alias_penalty=0.0):
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
    def fake_score(_band_resid, period, *, alias_penalty=0.0):
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


def test_find_period_arbitration_only_evaluates_base_and_shorter_harmonics(
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
    assert evaluated_periods == pytest.approx([2.0, 1.0, 2.0 / 3.0, 0.5])
    assert all(period <= 2.0 for period in evaluated_periods)


def test_find_period_arbitration_cannot_promote_to_longer_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_score(_band_resid, period, *, alias_penalty=0.2):
        if float(period) > 2.0:
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

    assert selected_period == pytest.approx(2.0)
    assert factor == pytest.approx(1.0)
    assert diag["base_objective"] == pytest.approx(1.0)


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
