from __future__ import annotations

import pandas as pd
import pytest

from malca.evaluation.periodicity_gate_injection_benchmark import (
    _pipeline_post_filter_input,
    generate_trial_lightcurve,
    summarize_results,
)


def test_injection_records_sampled_support_instead_of_theoretical_events() -> None:
    base = pd.DataFrame(
        {
            "JD": [0.0, 1.0, 2.0, 3.0],
            "mag": [14.0] * 4,
            "error": [0.03] * 4,
            "camera#": [1] * 4,
        }
    )
    trial = {
        "trial_id": 1,
        "trial_seed": 12,
        "class_name": "single_non_recurrent_dip",
        "target_dip": True,
        "amplitude": 0.5,
        "duration": 1.0,
        "t0": 100.0,
        "depth_scatter": 0.0,
    }

    _injected, meta = generate_trial_lightcurve(base, trial)

    assert meta["injected_event_count"] == 1
    assert meta["injected_sampled_event_count"] == 0
    assert meta["injected_support_points"] == 0
    assert meta["injection_observable"] is False


def test_summary_separates_errors_observability_overlap_and_false_positives() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": [1, 2, 3],
            "class_name": ["dip", "dip", "control"],
            "target_dip": [True, True, False],
            "target_gate_label": ["non_periodic", "non_periodic", "non_periodic"],
            "injection_observable": [True, False, False],
            "standard_status": ["ok", "error", "ok"],
            "phase_folded_status": ["error", "error", "error"],
            "bifurcated_status": ["ok", "error", "ok"],
            "standard_detected": [True, False, True],
            "phase_folded_detected": [False, False, False],
            "bifurcated_detected": [True, False, True],
            "standard_dip_injected_overlap": [True, False, False],
            "phase_folded_dip_injected_overlap": [False, False, False],
            "bifurcated_injected_overlap": [True, False, False],
            "pre_periodic_flag": [False, False, False],
            "period_usable": [False, False, False],
        }
    )

    summary = summarize_results(frame)
    row = summary.loc[
        summary["scope"].eq("all") & summary["pipeline"].eq("standard_only")
    ].iloc[0]

    assert row["evaluated_n"] == 2
    assert row["error_n"] == 1
    assert row["target_n"] == 2
    assert row["evaluated_target_n"] == 1
    assert row["observable_target_n"] == 1
    assert row["target_recovered_n"] == 1
    assert row["target_recovery"] == pytest.approx(1.0)
    assert row["end_to_end_target_recovery"] == pytest.approx(0.5)
    assert row["false_positive_rate"] == pytest.approx(1.0)
    assert row["precision_by_trial"] == pytest.approx(0.5)


def test_post_filter_bridge_uses_the_canonical_lightcurve_path_column() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": [1],
            "class_name": ["control"],
            "source_id": ["source-1"],
            "source_path": ["source.dat2"],
            "target_dip": [False],
            "target_gate_label": ["non_periodic"],
            "standard_detected": [False],
            "standard_status": ["ok"],
            "standard_dip_injected_overlap": [False],
            "pre_periodic_flag": [False],
        }
    )

    out = _pipeline_post_filter_input(frame, "standard_only")

    assert out.loc[0, "lc_path"] == "synthetic://standard_only/trial_1"
    assert "path" not in out.columns
