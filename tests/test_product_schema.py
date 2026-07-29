from __future__ import annotations

import pandas as pd
import pytest

from malca.products.feature_layers import (
    feature_layer_for_column,
    parse_layer_value,
    to_layer_first_frame,
)
from malca.products.product_schema import (
    ProductSchemaError,
    STV_EVENT_REQUIRED_COLUMNS,
    _assert_required_identity_values,
    add_ltv_identity,
    add_stv_identity,
    assert_ltv_product_schema,
    assert_stv_product_schema,
)


def _canonical_event_row(
    *,
    candidate_id: str = "stv_event_1",
    tag_stats_status: str = "ok",
    tag_stats_error: object = "",
    raw_n_points: object = 20,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "timescale": "stv",
        "asas_sn_id": candidate_id.removeprefix("stv_"),
        "lc_path": f"/tmp/{candidate_id}.dat2",
        "event_schema_version": 2,
        "event_score_version": 2,
        "tag_stats_status": tag_stats_status,
        "tag_stats_error": tag_stats_error,
        "tag_stats_version": 2,
        "raw_n_points": raw_n_points,
        "clean_n_points": 18,
        "raw_n_cameras": 2,
        "raw_camera_ids": "1,2",
        "raw_asassn_fields": "field-a,field-b",
        "raw_camera_names": "cam-a,cam-b",
        "baseline_cross_band_calibrated": False,
        "baseline_cross_band_details": "{}",
        "dip_best_delta_mag": 0.25,
        "jump_best_delta_mag": -0.10,
    }


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


def test_schema_rejects_blank_or_duplicate_candidate_ids() -> None:
    blank = to_layer_first_frame(pd.DataFrame({
        "candidate_id": [""], "timescale": ["stv"], "lc_path": ["a.dat3"],
    }))
    with pytest.raises(ProductSchemaError, match="blank/null"):
        assert_stv_product_schema(blank, stage="events")

    duplicate = to_layer_first_frame(pd.DataFrame({
        "candidate_id": ["stv_a", "stv_a"],
        "timescale": ["stv", "stv"],
        "lc_path": ["a.dat3", "b.dat3"],
    }))
    with pytest.raises(ProductSchemaError, match="duplicate"):
        assert_stv_product_schema(duplicate, stage="events")


def test_schema_rejects_invalid_layer_json() -> None:
    layered = to_layer_first_frame(pd.DataFrame({
        "candidate_id": ["stv_a"], "timescale": ["stv"], "lc_path": ["a.dat3"],
    }))
    layered.loc[0, "lc_stats"] = "not-json"
    with pytest.raises(ProductSchemaError, match="invalid JSON"):
        assert_stv_product_schema(layered, stage="events")


def test_event_schema_preserves_explicit_success_error_and_accepts_canonical_row() -> None:
    layered = to_layer_first_frame(pd.DataFrame([_canonical_event_row()]))
    error_layer = feature_layer_for_column("tag_stats_error")

    assert error_layer is not None
    assert parse_layer_value(layered.loc[0, error_layer])["tag_stats_error"] == ""
    assert_stv_product_schema(
        layered,
        stage="events",
        required=STV_EVENT_REQUIRED_COLUMNS,
    )


def test_event_schema_accepts_mixed_success_and_error_accounting_rows() -> None:
    layered = to_layer_first_frame(pd.DataFrame([
        _canonical_event_row(candidate_id="stv_ok"),
        _canonical_event_row(
            candidate_id="stv_error",
            tag_stats_status="error",
            tag_stats_error="FileNotFoundError: missing.dat2",
        ),
    ]))

    assert_stv_product_schema(
        layered,
        stage="events",
        required=STV_EVENT_REQUIRED_COLUMNS,
    )


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("ok", "unexpected diagnostic"),
        ("ok", None),
        ("error", ""),
        ("error", None),
    ],
)
def test_event_schema_rejects_inconsistent_tag_stats_error(
    status: str,
    error: object,
) -> None:
    layered = to_layer_first_frame(pd.DataFrame([
        _canonical_event_row(tag_stats_status=status, tag_stats_error=error),
    ]))

    with pytest.raises(ProductSchemaError, match="tag_stats_error"):
        assert_stv_product_schema(
            layered,
            stage="events",
            required=STV_EVENT_REQUIRED_COLUMNS,
        )


def test_event_schema_rejects_null_required_value_inside_feature_layer() -> None:
    layered = to_layer_first_frame(pd.DataFrame([_canonical_event_row()]))
    stats = parse_layer_value(layered.loc[0, "lc_stats"])
    stats["raw_n_points"] = None
    layered.at[0, "lc_stats"] = stats

    with pytest.raises(ProductSchemaError, match="raw_n_points.*missing"):
        assert_stv_product_schema(
            layered,
            stage="events",
            required=STV_EVENT_REQUIRED_COLUMNS,
        )


def test_required_value_validation_matches_flat_and_layered_forms() -> None:
    flat = pd.DataFrame([_canonical_event_row(raw_n_points=None)])
    layered = to_layer_first_frame(flat, include_missing=True)

    for frame in (flat, layered):
        with pytest.raises(ProductSchemaError, match="raw_n_points.*missing"):
            _assert_required_identity_values(
                frame,
                STV_EVENT_REQUIRED_COLUMNS,
                timescale="stv",
                stage="events",
            )
