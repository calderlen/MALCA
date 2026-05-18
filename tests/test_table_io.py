from __future__ import annotations

import pandas as pd

from malca.table_io import read_parquet_table, read_passing_parquet_table, write_parquet_table


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
