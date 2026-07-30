from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malca.evaluation import attrition, audit, false_positive, reproduce
from malca.io.table_io import write_feature_table, write_parquet_table


def test_attrition_booleans_are_fail_closed_and_retention_has_raw_denominator() -> None:
    parsed = attrition._to_bool(pd.Series([True, "false", "yes", "banana", None, 2]))
    assert parsed.tolist()[:3] == [True, False, True]
    assert parsed.iloc[3:].isna().all()

    pre = pd.DataFrame({"candidate_id": ["a", "b", "c"]})
    post = pd.DataFrame({"candidate_id": ["b", "c"]})
    result = attrition.retention(pre, post)
    assert result["retention_numerator"] == 2
    assert result["retention_denominator"] == 3
    assert result["retention_frac"] == pytest.approx(2 / 3)
    assert result["retention_ci95_low"] < result["retention_ci95_high"]

    flags = attrition.band_flags(
        pd.DataFrame(
            {
                "dip_significant": pd.Series([pd.NA, True, False], dtype="boolean"),
                "jump_significant": pd.Series([pd.NA, pd.NA, False], dtype="boolean"),
            }
        )
    )
    assert pd.isna(flags.loc[0, "either_det"])
    assert flags.loc[1, "either_det"]
    assert not flags.loc[2, "either_det"]


def test_attrition_rejects_duplicate_candidate_identity() -> None:
    duplicated = pd.DataFrame({"candidate_id": ["same", "same"]})
    with pytest.raises(ValueError, match="Duplicate candidate identities"):
        attrition.retention(duplicated, pd.DataFrame())


def test_audit_reports_unknown_booleans_provenance_and_rejects_duplicates(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    filtered_path = tmp_path / "filtered.parquet"
    write_feature_table(
        pd.DataFrame({"candidate_id": ["a", "b"], "failed_any": [False, "not-a-boolean"]}),
        raw_path,
    )
    write_feature_table(
        pd.DataFrame({"candidate_id": ["b"], "failed_any": [True]}),
        filtered_path,
    )

    report = audit.compare_results(raw_path, filtered_path, key="candidate_id")
    assert report["raw"]["failed_any_false"] == 1
    assert report["raw"]["failed_any_unknown"] == 1
    assert report["raw_vs_filtered"]["retention_numerator"] == 1
    assert report["raw_vs_filtered"]["retention_denominator"] == 2
    assert len(report["provenance"]["run_fingerprint"]) == 64

    duplicate_path = tmp_path / "duplicate.parquet"
    write_feature_table(pd.DataFrame({"candidate_id": ["a", "a"]}), duplicate_path)
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        audit.compare_results(duplicate_path, filtered_path, key="candidate_id")


def test_false_positive_summary_does_not_count_errors_as_nondetections() -> None:
    trials = pd.DataFrame(
        {
            "family": ["camera_offset"] * 4,
            "trial_status": ["ok", "ok", "error", "ineligible_empty_light_curve"],
            "detected": pd.Series([True, False, pd.NA, pd.NA], dtype="boolean"),
        }
    )
    summary = false_positive.compute_false_positive_summary(
        trials,
        families=["camera_offset"],
        n_trials_per_family=4,
    ).iloc[0]

    assert summary["n_false_positive"] == 1
    assert summary["rate_denominator"] == 2
    assert summary["nondetections"] == 1
    assert summary["error_trials"] == 1
    assert summary["ineligible_trials"] == 1
    assert summary["false_positive_rate"] == pytest.approx(0.5)
    assert summary["false_positive_rate_ci95_low"] < summary["false_positive_rate_ci95_high"]


def test_false_positive_scoring_is_per_band_and_applies_amplitude_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[pd.DataFrame] = []

    def fake_score(frame: pd.DataFrame, **_kwargs):
        calls.append(frame.copy())
        band = int(frame["v_g_band"].iloc[0])
        delta = 0.05 if band == 0 else 0.2
        return {
            "dip": {"significant": True, "best_delta_mag": delta, "trigger_max": 0.9},
            "jump": {"significant": False, "best_delta_mag": 0.0, "trigger_max": 0.1},
        }

    monkeypatch.setattr(false_positive, "score_lightcurve", fake_score)
    result = false_positive._default_detection_func(
        pd.DataFrame(
            {
                "JD": [1.0, 2.0, 1.5, 2.5],
                "mag": [14.0, 14.1, 13.0, 13.2],
                "v_g_band": [0, 0, 1, 1],
            }
        ),
        {"min_mag_offset": 0.1},
    )

    assert len(calls) == 2
    assert all(frame["v_g_band"].nunique() == 1 for frame in calls)
    assert result["g_dip_significant"] is False
    assert result["v_dip_significant"] is True
    assert result["detected"] is True


def test_false_positive_trials_are_deterministic_and_stale_resume_is_invalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    for candidate_id in ("a", "b"):
        path = tmp_path / f"{candidate_id}.dat3"
        path.write_text("input", encoding="utf-8")
        paths.append(str(path))
    manifest = pd.DataFrame({"candidate_id": ["a", "b"], "path": paths})
    calls: list[tuple] = []

    def fake_trial(*task):
        calls.append(task)
        detected = bool(task[5].get("force_detected", False))
        return {
            "family": task[0],
            "trial": task[4],
            "trial_index": task[4],
            "trial_id": task[8],
            "candidate_id": task[1],
            "asas_sn_id": task[1],
            "input_path": task[9],
            "base_seed": task[7],
            "trial_seed": task[6],
            "input_fingerprint": task[10],
            "config_fingerprint": task[11],
            "run_fingerprint": task[12],
            "trial_status": "ok",
            "detected": detected,
            "dip_significant": detected,
            "jump_significant": False,
            "dip_trigger_max": 1.0,
            "jump_trigger_max": 0.0,
            "error": None,
        }

    monkeypatch.setattr(false_positive, "_run_single_injection", fake_trial)
    out_dir = tmp_path / "out"
    first = false_positive.run_false_positive_benchmark(
        manifest,
        out_dir=out_dir,
        families=["camera_offset"],
        n_trials_per_family=4,
        detection_kwargs={"force_detected": False},
        seed=42,
    )
    first_identity = first[["candidate_id", "trial_seed"]].to_dict("records")
    assert len(calls) == 4

    # A compatible resume runs no trials.
    calls.clear()
    resumed = false_positive.run_false_positive_benchmark(
        manifest,
        out_dir=out_dir,
        families=["camera_offset"],
        n_trials_per_family=4,
        detection_kwargs={"force_detected": False},
        seed=42,
    )
    assert calls == []
    assert resumed[["candidate_id", "trial_seed"]].to_dict("records") == first_identity

    # A science-config change changes the run fingerprint and recomputes all trials.
    refreshed = false_positive.run_false_positive_benchmark(
        manifest,
        out_dir=out_dir,
        families=["camera_offset"],
        n_trials_per_family=4,
        detection_kwargs={"force_detected": True},
        seed=42,
    )
    assert len(calls) == 4
    assert refreshed["detected"].all()
    assert refreshed["run_fingerprint"].nunique() == 1
    assert refreshed["run_fingerprint"].iloc[0] != first["run_fingerprint"].iloc[0]


def test_reproduction_missing_input_is_unknown_not_nondetection(tmp_path: Path) -> None:
    missing = tmp_path / "missing.dat2"
    report = reproduce.build_reproduction_report(
        candidates=[
            {
                "candidate_id": "candidate-a",
                "source_id": "source-a",
                "mag_bin": "13_13.5",
                "path": str(missing),
            }
        ],
        out_dir=None,
        min_mag_offset=0.0,
    )

    assert len(report) == 1
    assert report.loc[0, "candidate_id"] == "candidate-a"
    assert report.loc[0, "trial_status"] == "input_missing"
    assert pd.isna(report.loc[0, "detected"])
    assert len(report.loc[0, "run_fingerprint"]) == 64


def test_reproduction_amplitude_gate_is_band_and_branch_specific() -> None:
    frame = pd.DataFrame(
        {
            "trial_status": ["ok", "ok"],
            "rejection_reason": [None, None],
            "g_rejection_reason": [None, None],
            "v_rejection_reason": [None, None],
            "g_bayes_dip_significant": [True, True],
            "v_bayes_dip_significant": [True, False],
            "g_bayes_jump_significant": [False, False],
            "v_bayes_jump_significant": [False, True],
            "g_dip_best_mag_event": [0.05, 0.05],
            "v_dip_best_mag_event": [0.20, 0.00],
            "g_jump_best_mag_event": [0.00, 0.00],
            "v_jump_best_mag_event": [0.00, 0.20],
            # These deliberately differ by band: the amplitude gate must not
            # construct a hybrid baseline/event measurement from them.
            "g_baseline_mag": [14.0, 14.0],
            "v_baseline_mag": [12.0, 12.0],
        }
    )

    out = reproduce._apply_reproduction_signal_amplitude(frame, min_mag_offset=0.1)

    assert out["g_bayes_dip_significant"].tolist() == [False, False]
    assert out["v_bayes_dip_significant"].tolist() == [True, False]
    assert out["v_bayes_jump_significant"].tolist() == [False, True]
    assert out["dip_signal_amplitude_pass"].tolist() == [True, False]
    assert out["signal_amplitude_pass"].tolist() == [True, True]
    assert pd.isna(out.loc[0, "rejection_reason"])
    assert out.loc[1, "rejection_reason"] == "signal_amplitude"


def test_reproduction_success_uses_one_exact_light_curve_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    light_curve = tmp_path / "source-a.dat3"
    light_curve.write_text(
        "1 14.0 0.05 1 1 0 0 cam/field\n"
        "2 14.4 0.05 1 1 0 0 cam/field\n"
        "3 14.5 0.05 1 2 0 0 cam/field\n"
        "4 14.0 0.05 1 2 0 0 cam/field\n",
        encoding="utf-8",
    )
    other_light_curve = tmp_path / "other" / "source-a.dat3"
    other_light_curve.parent.mkdir()
    other_light_curve.write_text(light_curve.read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path = tmp_path / "manifest.parquet"
    write_parquet_table(
        pd.DataFrame(
            {
                "source_id": ["source-a"],
                "mag_bin": ["13_13.5"],
                "lc_dir": [str(other_light_curve.parent)],
                "dat_path": [str(other_light_curve)],
                "dat_exists": [True],
            }
        ),
        manifest_path,
    )

    def fake_score(frame, **_kwargs):
        return {
            "dip": {
                "significant": True,
                "event_indices": [1, 2],
                "run_summaries": [
                    {
                        "morphology": "gaussian",
                        "params": {"t0": 2.5, "amplitude": 0.5, "sigma": 0.5},
                        "n_points": 2,
                    }
                ],
                "n_runs": 1,
                "n_dips": 2,
                "bayes_factor": 20.0,
                "max_event_prob": 0.99,
                "max_log_bf_local": 5.0,
                "trigger_max": 0.99,
            },
            "jump": {
                "significant": False,
                "event_indices": [],
                "run_summaries": [],
                "n_runs": 0,
                "n_jumps": 0,
                "bayes_factor": 1.0,
                "max_event_prob": 0.1,
                "max_log_bf_local": 0.0,
                "trigger_max": 0.1,
            },
        }

    monkeypatch.setattr(reproduce, "score_lightcurve", fake_score)
    monkeypatch.setattr(reproduce, "compute_event_score", lambda *_args, **_kwargs: (1.0, []))
    report = reproduce.build_reproduction_report(
        candidates=[
            {
                "candidate_id": "candidate-a",
                "source_id": "source-a",
                "mag_bin": "13_13.5",
                "path": str(light_curve),
            }
        ],
        out_dir=None,
        min_mag_offset=0.0,
        baseline_func="global_median",
        manifest_path=manifest_path,
        skip_tags=True,
    )

    assert report.loc[0, "trial_status"] == "ok"
    assert report.loc[0, "detected"]
    assert report.loc[0, "dat_path"] == str(light_curve)
    assert report.loc[0, "path"] == str(light_curve)
    assert "dat_path_det" not in report.columns
    assert report.loc[0, "input_source"] == "candidate_path"
    assert report["input_record_fingerprint"].nunique() == 1


def test_reproduction_rejects_duplicate_candidate_identity(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "candidate_id": ["same", "same"],
            "source_id": ["source-a", "source-b"],
            "path": ["a.dat2", "b.dat2"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        reproduce.load_candidates_df(path)
