from __future__ import annotations

import pandas as pd
import pytest

from malca.feature_layers import to_layer_first_frame
from malca.product_schema import (
    ProductSchemaError,
    add_ltv_identity,
    add_stv_identity,
    assert_ltv_product_schema,
    assert_stv_product_schema,
)


def test_add_stv_identity_uses_lc_path_and_prefix() -> None:
    out = add_stv_identity(pd.DataFrame({"lc_path": ["/tmp/ASASSN-1.dat2"]}))

    assert out["candidate_id"].tolist() == ["stv_ASASSN-1"]
    assert out["timescale"].tolist() == ["stv"]


def test_add_ltv_identity_uses_asas_sn_id_and_prefix() -> None:
    out = add_ltv_identity(pd.DataFrame({"asas_sn_id": ["123"], "lc_path": ["/tmp/123.dat2"]}))

    assert out["candidate_id"].tolist() == ["ltv_123"]
    assert out["timescale"].tolist() == ["ltv"]


def test_stv_schema_rejects_legacy_path_column() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["stv_a"],
            "timescale": ["stv"],
            "lc_path": ["a.dat2"],
            "path": ["a.dat2"],
        }
    )

    with pytest.raises(ProductSchemaError, match="forbidden=path"):
        assert_stv_product_schema(df, stage="events")


def test_ltv_schema_rejects_raw_core_columns() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["ltv_1"],
            "timescale": ["ltv"],
            "asas_sn_id": ["1"],
            "lc_path": ["1.dat2"],
            "ra": [1.0],
            "dec": [2.0],
            "Slope": [0.1],
        }
    )

    with pytest.raises(ProductSchemaError, match="forbidden=Slope"):
        assert_ltv_product_schema(df, stage="filtered")


def test_ltv_schema_rejects_wrong_timescale_value() -> None:
    df = to_layer_first_frame(pd.DataFrame(
        {
            "candidate_id": ["ltv_1"],
            "timescale": ["stv"],
            "asas_sn_id": ["1"],
            "lc_path": ["1.dat2"],
            "ra": [1.0],
            "dec": [2.0],
        }
    ))

    with pytest.raises(ProductSchemaError, match="Unexpected timescale"):
        assert_ltv_product_schema(df, stage="pipeline")


def test_ltv_schema_accepts_layer_first_required_features() -> None:
    flat = pd.DataFrame(
        {
            "candidate_id": ["ltv_1"],
            "timescale": ["ltv"],
            "asas_sn_id": ["1"],
            "lc_path": ["1.dat2"],
            "ra": [1.0],
            "dec": [2.0],
            "ltv_slope": [0.2],
        }
    )
    layered = to_layer_first_frame(flat)

    assert "ra" not in layered.columns
    assert "dec" not in layered.columns
    assert_ltv_product_schema(layered, stage="layered")
