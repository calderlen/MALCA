from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.evaluation.period_cost_accuracy import (
    StoredPeriodStrategy,
    add_catalog_reference,
    evaluate_stored_strategies,
    mark_pareto_frontier,
    period_match_arrays,
    project_runtime_days,
    stratified_runtime_sample,
)


def test_catalog_reference_requires_clean_agreement() -> None:
    frame = pd.DataFrame(
        {
            "period_consensus_days": [10.0, 20.0, 30.0, 40.0],
            "period_n_sources": [1, 2, 2, 0],
            "period_consensus_agree": [1, 1, 0, 1],
            "period_conflict_flag": [0, 1, 0, 0],
        }
    )
    result = add_catalog_reference(frame)
    assert result["catalog_reference_available"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert result.loc[0, "catalog_reference_tier"] == "clean_single_catalog"


def test_period_match_arrays_distinguishes_exact_and_harmonic() -> None:
    matched = period_match_arrays(
        [10.1, 5.0, 40.0, np.nan],
        [10.0, 10.0, 10.0, 10.0],
        tolerance=0.05,
    )
    assert matched["is_exact"].tolist() == [True, False, False, False]
    assert matched["is_harmonic_family"].tolist() == [True, True, True, False]
    assert matched["nearest_harmonic_factor"][1] == pytest.approx(0.5)
    assert matched["nearest_harmonic_factor"][2] == pytest.approx(4.0)


def test_catalog_strategy_is_not_scored_against_itself() -> None:
    frame = pd.DataFrame(
        {
            "catalog_reference_period": [10.0, 20.0, np.nan],
            "computed": [10.0, 10.0, 30.0],
        }
    )
    strategies = (
        StoredPeriodStrategy(
            "catalog",
            "Catalog",
            ("catalog_reference_period",),
            "catalog",
            False,
            "coverage only",
        ),
        StoredPeriodStrategy(
            "computed",
            "Computed",
            ("computed",),
            "single_method",
            True,
            "independent",
        ),
    )
    result = evaluate_stored_strategies(frame, strategies=strategies)
    catalog = result.set_index("strategy").loc["catalog"]
    computed = result.set_index("strategy").loc["computed"]
    assert np.isnan(catalog["exact_agreement_conditional"])
    assert computed["exact_agreement_conditional"] == pytest.approx(0.5)
    assert computed["family_agreement_conditional"] == pytest.approx(1.0)


def test_stratified_runtime_sample_is_deterministic_and_weighted() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"source-{index}" for index in range(30)],
            "lc_path": [f"/tmp/source-{index}.dat3" for index in range(30)],
            "median_mag": np.linspace(12.0, 15.0, 30),
            "n_points": np.arange(30) + 100,
            "catalog_reference_available": [index % 2 == 0 for index in range(30)],
        }
    )
    first = stratified_runtime_sample(frame, 12, seed=7)
    second = stratified_runtime_sample(frame, 12, seed=7)
    assert first["candidate_id"].tolist() == second["candidate_id"].tolist()
    assert len(first) == 12
    assert first["sample_weight"].gt(0).all()


def test_runtime_projection_and_pareto_frontier() -> None:
    days = project_runtime_days(
        n_sources=17_000_000,
        seconds_per_source=29.5,
        workers_per_machine=60,
        machines=6,
        parallel_efficiency=1.0,
    )
    assert days == pytest.approx(16.1233, rel=1.0e-4)

    summary = pd.DataFrame(
        {
            "strategy": ["cheap", "dominated", "accurate"],
            "seconds": [1.0, 2.0, 3.0],
            "accuracy": [0.8, 0.7, 0.9],
        }
    )
    marked = mark_pareto_frontier(
        summary,
        cost_col="seconds",
        accuracy_col="accuracy",
    )
    assert marked.set_index("strategy")["is_pareto"].to_dict() == {
        "cheap": True,
        "dominated": False,
        "accurate": True,
    }
