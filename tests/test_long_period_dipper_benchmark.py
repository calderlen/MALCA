"""Smoke tests for the long-period dipper injection benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.evaluation.long_period_dipper_injection_benchmark import (
    LongPeriodDipperBenchmarkConfig,
    build_trial_design,
    inject_long_period_dips,
    summarize_results,
)


def test_build_trial_design_respects_cycle_bins() -> None:
    controls = pd.DataFrame(
        {
            "source_id": ["a"],
            "source_path": ["/tmp/a.dat3"],
            "jd_span": [3653.0],
        }
    )
    config = LongPeriodDipperBenchmarkConfig(
        n_trials_per_setting=1,
        baseline_cycles=(2.0,),
        event_counts=(2,),
    )
    design = build_trial_design(controls, config)
    assert len(design) == 1
    assert design.loc[0, "true_period_days"] == pytest.approx(3653.0 / 2.0)


def test_inject_long_period_dips_adds_epochs() -> None:
    rng = np.random.default_rng(0)
    jd = np.linspace(2458000.0, 2461653.0, 400)
    df = pd.DataFrame({"JD": jd, "mag": np.zeros_like(jd), "error": np.full_like(jd, 0.02)})
    out, centers = inject_long_period_dips(
        df,
        true_period_days=1800.0,
        event_count=2,
        amplitude=0.5,
        duration=40.0,
        phase0=0.2,
        rng=rng,
    )
    assert len(centers) >= 2
    assert bool(np.any(np.abs(out["mag"].to_numpy() - df["mag"].to_numpy()) > 1e-6))


def test_summarize_results_reports_recall() -> None:
    results = pd.DataFrame(
        {
            "status": ["ok", "ok", "ok", "ok"],
            "baseline_cycles": [2.0, 2.0, 3.0, 3.0],
            "requested_event_count": [2, 2, 2, 2],
            "pdm_match": [False, False, False, False],
            "long_ls_match": [True, False, True, True],
            "consensus_match": [True, True, False, True],
        }
    )
    summary = summarize_results(results)
    assert not summary.empty
    long_ls = summary[summary["method"] == "long_ls"]
    assert float(long_ls.loc[long_ls["baseline_cycles"] == 2.0, "recall"].iloc[0]) == pytest.approx(0.5)
