from __future__ import annotations

import pandas as pd

from malca.ml.features import ML_FEATURE_COLUMNS, infer_ml_feature_columns, select_ml_features


def test_ml_feature_columns_include_new_fourier_fields() -> None:
    expected = {
        "stats_harmonics_order",
        "stats_harmonics_period",
        "stats_harmonics_a0",
        "stats_harmonics_model_amplitude",
        "stats_harmonics_reduced_chi2",
        "stats_harmonics_r21",
        "stats_harmonics_r31",
        "stats_harmonics_r41",
        "stats_harmonics_r51",
        "stats_harmonics_r61",
        "stats_harmonics_r71",
    }
    assert expected.issubset(set(ML_FEATURE_COLUMNS))


def test_select_ml_features_keeps_new_fourier_fields() -> None:
    df = pd.DataFrame(
        {
            "stats_harmonics_order": [3],
            "stats_harmonics_period": [2.75],
            "stats_harmonics_a0": [12.3],
            "stats_harmonics_model_amplitude": [0.72],
            "stats_harmonics_reduced_chi2": [0.91],
            "stats_harmonics_r21": [0.4],
            "stats_harmonics_phase_2": [0.5],
            "dip_best_morph": ["gaussian"],
        }
    )

    cols = infer_ml_feature_columns(df)
    assert "stats_harmonics_order" in cols
    assert "stats_harmonics_r21" in cols

    X = select_ml_features(df)
    assert float(X.loc[0, "stats_harmonics_order"]) == 3.0
    assert float(X.loc[0, "stats_harmonics_period"]) == 2.75
    assert float(X.loc[0, "stats_harmonics_r21"]) == 0.4
    assert "dip_best_morph" in X.columns
