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


def test_layer_expansion_fills_null_flat_values_without_overwriting_sql_values() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["stv_1", "stv_2"],
            "timescale": ["stv", "stv"],
            "stats_amplitude": [None, 9.0],
            "lc_stats": [{"stats_amplitude": 1.5}, {"stats_amplitude": 2.5}],
            "external_stats": [{}, {}],
            "derived_stats": [{}, {}],
        }
    )

    expanded = expand_feature_layers(df)
    selected = with_feature_columns(df, ["stats_amplitude"])

    assert expanded["stats_amplitude"].tolist() == [1.5, 9.0]
    assert selected["stats_amplitude"].tolist() == [1.5, 9.0]


def test_layer_coalescing_handles_null_only_flat_dtype_without_future_warning() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["stv_1", "stv_2"],
            "timescale": ["stv", "stv"],
            # A null-only SQL column is commonly inferred as float64 even when
            # the canonical layer values for this field are strings.
            "quality_status": pd.Series([float("nan"), float("nan")], dtype="float64"),
            "lc_stats": [{"quality_status": "ok"}, {"quality_status": "failed"}],
            "external_stats": [{}, {}],
            "derived_stats": [{}, {}],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        expanded = expand_feature_layers(df)
        selected = with_feature_columns(df, ["quality_status"])

    assert expanded["quality_status"].tolist() == ["ok", "failed"]
    assert selected["quality_status"].tolist() == ["ok", "failed"]
