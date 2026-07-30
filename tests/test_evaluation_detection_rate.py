from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.evaluation import detection_rate


def _branch(*, significant: bool, delta: float) -> dict[str, object]:
    return {
        "significant": significant,
        "bayes_factor": 10.0,
        "best_p": 0.2,
        "baseline_mag": 14.0,
        "best_mag_event": delta,
        "best_delta_mag": delta,
    }


def test_detection_amplitude_uses_residual_delta_and_inclusive_threshold() -> None:
    result = detection_rate._extract_detection_result(
        _branch(significant=True, delta=0.1),
        _branch(significant=False, delta=99.0),
        min_mag_offset=0.1,
    )

    assert result["detected"] is True
    assert result["dip_significant"] is True
    assert result["jump_significant"] is False


def test_trial_errors_are_unknown_decisions_not_nondetections(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_load(_candidate_id: str, _path) -> pd.DataFrame:
        raise OSError("broken")

    monkeypatch.setattr(detection_rate, "_load_lc", fail_load)
    result = detection_rate.run_detection_rate_trial(
        0,
        np.array(["a"]),
        np.array(["/tmp/a.dat3"]),
        {},
        seed=7,
    )

    assert result["trial_status"] == "error"
    assert pd.isna(result["detected"])
    assert "load_error" in result["error"]


def test_detection_summary_separates_errors_ineligible_and_nondetections() -> None:
    frame = pd.DataFrame(
        {
            "trial_index": range(5),
            "asas_sn_id": ["a", "b", "c", "d", "e"],
            "trial_status": ["ok", "ok", "ok", "error", "ineligible_magnitude"],
            "detected": pd.Series([True, False, False, pd.NA, pd.NA], dtype="boolean"),
        }
    )

    summary = detection_rate.compute_detection_summary(frame)

    assert summary["successful_trials"] == 3
    assert summary["failed_trials"] == 1
    assert summary["ineligible_trials"] == 1
    assert summary["detections"] == 1
    assert summary["detection_rate"] == pytest.approx(1 / 3)
    assert summary["end_to_end_detection_yield"] == pytest.approx(1 / 5)
    assert summary["detection_rate_ci95_low"] < summary["detection_rate_ci95_high"]


def test_control_sample_rejects_duplicate_identity() -> None:
    manifest = pd.DataFrame(
        {
            "candidate_id": ["same", "same"],
            "path": ["a.dat3", "b.dat3"],
            "n_points": [100, 100],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        detection_rate.select_control_sample(manifest)

