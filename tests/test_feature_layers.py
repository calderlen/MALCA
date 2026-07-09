from __future__ import annotations

import warnings

import pandas as pd

from malca.products.feature_layers import expand_feature_layers, with_feature_columns


def _wide_layer_first_frame(n_features: int = 140) -> pd.DataFrame:
    lc_stats = {f"stats_feature_{idx}": idx for idx in range(n_features)}
    return pd.DataFrame(
        {
            "candidate_id": ["stv_1"],
            "timescale": ["stv"],
            "lc_stats": [lc_stats],
            "external_stats": [{}],
            "derived_stats": [{}],
        }
    )


def test_expand_feature_layers_batches_wide_layer_columns_without_fragmentation_warning() -> None:
    df = _wide_layer_first_frame()

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        out = expand_feature_layers(df)

    assert out.loc[0, "stats_feature_0"] == 0
    assert out.loc[0, "stats_feature_139"] == 139


def test_with_feature_columns_batches_wide_layer_columns_without_fragmentation_warning() -> None:
    df = _wide_layer_first_frame()
    columns = [f"stats_feature_{idx}" for idx in range(140)]

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        out = with_feature_columns(df, columns)

    assert out.loc[0, "stats_feature_0"] == 0
    assert out.loc[0, "stats_feature_139"] == 139
