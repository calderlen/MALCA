from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.july1_tuned_calibrated_four_class_training import (
    SearchSpec,
    _save_head,
    fit_tuned_calibrated_head,
    parent_search_spec,
    recurrence_search_spec,
    score_tuned_calibrated_head,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_idx, label in enumerate(("dimming", "junk", "other")):
        for idx in range(15):
            rows.append(
                {
                    "candidate_id": f"{label}-{idx}",
                    "target": label,
                    "signal": float(class_idx * 5 + idx / 20),
                    "scatter": float((idx * 7 + class_idx) % 11),
                    "band": "g" if idx % 2 else "g+V",
                }
            )
    return pd.DataFrame(rows)


def _small_search() -> SearchSpec:
    return SearchSpec(
        n_iter=2,
        cv_folds=3,
        random_state=17,
        parameter_distributions={
            "n_estimators": [15, 25],
            "learning_rate": [0.05],
            "num_leaves": [3, 5],
            "min_child_samples": [2],
            "colsample_bytree": [0.8],
            "subsample": [0.8],
            "reg_alpha": [0.0],
            "reg_lambda": [0.0, 0.1],
            "class_weight": [None, "balanced"],
        },
    )


def test_search_defaults_target_probability_quality_and_sparse_recurrence() -> None:
    parent = parent_search_spec()
    recurrence = recurrence_search_spec()

    assert parent.n_iter == 80
    assert parent.cv_repeats == 1
    assert parent.refit_metric == "neg_log_loss"
    assert recurrence.n_iter == 60
    assert recurrence.cv_repeats == 3
    assert recurrence.refit_metric == "neg_log_loss"
    assert recurrence.parameter_distributions["n_estimators"] == [
        100,
        150,
        250,
        400,
    ]
    assert recurrence.parameter_distributions["learning_rate"] == [
        0.01,
        0.03,
        0.05,
    ]
    assert recurrence.parameter_distributions["num_leaves"] == [3, 5, 7]
    assert recurrence.parameter_distributions["min_child_samples"] == [
        16,
        24,
        32,
        40,
    ]
    assert recurrence.parameter_distributions["class_weight"] == [None]


def test_standalone_tuned_head_uses_locked_test_and_calibrated_scores() -> None:
    frame = _training_frame()
    result = fit_tuned_calibrated_head(
        frame,
        target_column="target",
        requested_features=["signal", "scatter", "band"],
        population=frame,
        search_spec=_small_search(),
    )

    assert set(result.split_assignments["split"]) == {"development", "test"}
    assert len(result.test_predictions) == 9
    assert len(result.search_trials) == 2
    assert result.search_trials.iloc[0]["rank_test_neg_log_loss"] == 1
    assert result.raw_model.get_params()["subsample_freq"] == 1
    probability_columns = [
        column
        for column in result.all_candidate_scores
        if column.startswith("prob_")
    ]
    raw_columns = [
        column
        for column in result.all_candidate_scores
        if column.startswith("raw_score_")
    ]
    assert len(probability_columns) == 3
    assert len(raw_columns) == 3
    assert np.allclose(
        result.all_candidate_scores[probability_columns].sum(axis=1), 1.0
    )
    assert result.calibrated_test_metrics["n_eval"] == 9
    assert "log_loss" in result.calibrated_test_metrics
    assert "brier_macro" in result.calibrated_test_metrics


def test_standalone_tuned_head_bundle_roundtrip(tmp_path: Path) -> None:
    frame = _training_frame()
    result = fit_tuned_calibrated_head(
        frame,
        target_column="target",
        requested_features=["signal", "scatter", "band"],
        population=frame,
        search_spec=_small_search(),
    )
    _save_head(result, tmp_path)

    scored = score_tuned_calibrated_head(
        tmp_path / "model.joblib", frame.head(6)
    )

    assert len(scored) == 6
    assert set(scored["y_pred"]).issubset(set(result.label_classes))
    probability_columns = [
        column for column in scored if column.startswith("prob_")
    ]
    assert np.allclose(scored[probability_columns].sum(axis=1), 1.0)
    assert (tmp_path / "evaluation_model.joblib").is_file()
    assert (tmp_path / "search_trials.csv").is_file()
    assert (tmp_path / "raw_calibration_by_bin.csv").is_file()
    assert (tmp_path / "calibrated_calibration_by_bin.csv").is_file()
