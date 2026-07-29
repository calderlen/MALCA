from __future__ import annotations

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.bad_photometry import (
    choose_tabular_feature_columns,
)
from malca.meta_analysis.ml.feature_policy import (
    MODEL_FEATURE_EXCLUSION_COLUMNS,
    restore_legacy_excluded_model_features,
)
from malca.meta_analysis.ml.review_lightgbm import (
    DEFAULT_DROP_COLUMNS,
    TrainingConfig,
    train_target_model,
)


EXPECTED_MODEL_FEATURE_EXCLUSIONS = {
    "stats_jd_start",
    "stats_jd_end",
    "stats_trend_slope_mag_per_year",
    "stats_median_abs_dev",
    "stats_variability_quasi_periodicity_populated_bins",
    "wise_w3_missing",
    "wise_w4_missing",
}


def _training_frame() -> pd.DataFrame:
    labels = ["negative"] * 8 + ["positive"] * 8
    frame = pd.DataFrame(
        {
            "candidate_id": [f"candidate_{index}" for index in range(16)],
            "target": labels,
            "retained_signal": np.arange(16, dtype=float),
        }
    )
    for index, column in enumerate(EXPECTED_MODEL_FEATURE_EXCLUSIONS):
        frame[column] = np.arange(16, dtype=float) + index
    return frame


def test_shared_policy_contains_exact_requested_exclusions() -> None:
    assert MODEL_FEATURE_EXCLUSION_COLUMNS == EXPECTED_MODEL_FEATURE_EXCLUSIONS
    assert MODEL_FEATURE_EXCLUSION_COLUMNS.issubset(DEFAULT_DROP_COLUMNS)


def test_generic_review_model_excludes_shared_policy_columns() -> None:
    result = train_target_model(
        _training_frame(),
        "target",
        config=TrainingConfig(
            random_state=7,
            cv_folds=2,
            n_estimators=8,
            num_leaves=3,
            min_child_samples=1,
            min_class_count=2,
            n_jobs=1,
        ),
        drop_columns={"candidate_id"},
    )

    assert result.feature_columns == ["retained_signal"]


def test_bad_photometry_tabular_model_excludes_shared_policy_columns() -> None:
    feature_columns, _categorical_columns = choose_tabular_feature_columns(
        _training_frame().rename(columns={"target": "bad_photometry"})
    )

    assert "retained_signal" in feature_columns
    assert MODEL_FEATURE_EXCLUSION_COLUMNS.isdisjoint(feature_columns)


def test_legacy_model_scoring_restores_excluded_columns_without_changing_policy() -> None:
    source = pd.DataFrame(
        {
            "candidate_id": ["first", "second"],
            "retained_signal": [1.0, 2.0],
            "stats_median_abs_dev": [0.1, 0.2],
            "stats_trend_slope_mag_per_year": [0.3, 0.4],
        }
    )
    model_input = source[["candidate_id", "retained_signal"]].copy()

    restored, columns = restore_legacy_excluded_model_features(
        model_input,
        source,
        ["retained_signal", "stats_median_abs_dev", "stats_trend_slope_mag_per_year"],
    )

    assert columns == (
        "stats_median_abs_dev",
        "stats_trend_slope_mag_per_year",
    )
    assert restored["stats_median_abs_dev"].tolist() == [0.1, 0.2]
    assert restored["stats_trend_slope_mag_per_year"].tolist() == [0.3, 0.4]
    assert MODEL_FEATURE_EXCLUSION_COLUMNS.isdisjoint(model_input.columns)
