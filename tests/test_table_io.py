from __future__ import annotations

import pandas as pd

from malca.table_io import (
    is_layer_first_table,
    read_feature_table,
    read_parquet_table,
    read_passing_feature_table,
    read_passing_parquet_table,
    write_feature_table,
    write_parquet_table,
)


def test_write_parquet_table_chunked_roundtrip(tmp_path) -> None:
    path = tmp_path / "chunked.parquet"
    df = pd.DataFrame(
        {
            "path": [f"/tmp/{idx}.dat2" for idx in range(5)],
            "score": [0.1, 0.2, None, 0.4, 0.5],
            "failed_any": [False, True, False, True, False],
            "label": ["a", "b", "", "d", "e"],
            "late_object": [None, None, "first-value", None, "second-value"],
        }
    )

    write_parquet_table(df, path, chunk_rows=2)

    out = read_parquet_table(path)
    pd.testing.assert_frame_equal(out, df)


def test_read_passing_parquet_table_filters_failed_any(tmp_path) -> None:
    path = tmp_path / "filtered.parquet"
    df = pd.DataFrame(
        {
            "path": ["pass-a.dat2", "fail.dat2", "pass-b.dat2"],
            "failed_any": [False, True, False],
            "score": [1.0, 2.0, 3.0],
        }
    )
    write_parquet_table(df, path, chunk_rows=2)

    out = read_passing_parquet_table(path, columns=["path"])

    assert out.columns.tolist() == ["path"]
    assert out["path"].tolist() == ["pass-a.dat2", "pass-b.dat2"]


def test_feature_table_layer_first_roundtrip_and_projection(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    df = pd.DataFrame(
        {
            "candidate_id": ["stv_a", "stv_b"],
            "timescale": ["stv", "stv"],
            "lc_path": ["a.dat2", "b.dat2"],
            "dipper_score": [7.0, 2.0],
            "phot_g_mean_mag": [14.2, 15.1],
            "failed_any": [False, True],
        }
    )

    write_feature_table(df, path)

    physical = read_parquet_table(path)
    assert is_layer_first_table(path)
    assert "dipper_score" not in physical.columns
    assert "phot_g_mean_mag" not in physical.columns
    assert "failed_any" not in physical.columns
    assert {"lc_stats", "external_stats", "derived_stats"}.issubset(physical.columns)

    projected = read_feature_table(
        path,
        columns=["candidate_id", "lc_stats.dipper_score", "external_stats.phot_g_mean_mag"],
    )

    assert projected.columns.tolist() == [
        "candidate_id",
        "lc_stats.dipper_score",
        "external_stats.phot_g_mean_mag",
    ]
    assert projected["lc_stats.dipper_score"].tolist() == [7.0, 2.0]
    assert projected["external_stats.phot_g_mean_mag"].tolist() == [14.2, 15.1]


def test_read_passing_feature_table_filters_layer_failed_any(tmp_path) -> None:
    path = tmp_path / "layered_filtered.parquet"
    df = pd.DataFrame(
        {
            "candidate_id": ["stv_pass", "stv_fail"],
            "timescale": ["stv", "stv"],
            "lc_path": ["pass.dat2", "fail.dat2"],
            "failed_any": [False, True],
            "dipper_score": [8.0, 1.0],
        }
    )
    write_feature_table(df, path)

    out = read_passing_feature_table(path, columns=["candidate_id", "lc_stats.dipper_score"])

    assert out.columns.tolist() == ["candidate_id", "lc_stats.dipper_score"]
    assert out["candidate_id"].tolist() == ["stv_pass"]
    assert out["lc_stats.dipper_score"].tolist() == [8.0]


def test_write_feature_table_chunked_layer_conversion(tmp_path) -> None:
    path = tmp_path / "layered_chunked.parquet"
    n_rows = 7
    df = pd.DataFrame(
        {
            "candidate_id": [f"ltv_{idx}" for idx in range(n_rows)],
            "timescale": ["ltv"] * n_rows,
            "lc_path": [f"{idx}.dat2" for idx in range(n_rows)],
            "ra": [float(idx) for idx in range(n_rows)],
            "dec": [float(idx) for idx in range(n_rows)],
            "ltv_slope": [0.1 * idx for idx in range(n_rows)],
            "failed_any": [idx % 2 == 0 for idx in range(n_rows)],
        }
    )

    write_feature_table(df, path, layer_chunk_rows=2)

    projected = read_feature_table(
        path,
        columns=["candidate_id", "lc_stats.ltv_slope", "derived_stats.failed_any"],
    )
    assert len(projected) == n_rows
    assert projected["lc_stats.ltv_slope"].tolist() == [0.1 * idx for idx in range(n_rows)]
    assert projected["derived_stats.failed_any"].tolist() == [idx % 2 == 0 for idx in range(n_rows)]


def test_read_feature_table_flat_file_fails(tmp_path) -> None:
    path = tmp_path / "flat.parquet"
    df = pd.DataFrame({"candidate_id": ["a"], "dipper_score": [1.5]})
    write_parquet_table(df, path)

    import pytest

    with pytest.raises(ValueError, match="not layer-first"):
        read_feature_table(path)
