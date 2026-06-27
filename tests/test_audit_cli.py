from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.evaluation.audit import compare_results
from malca.io.table_io import write_feature_table


def test_compare_results_reports_filtered_only_and_raw_missing_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw.parquet"
    filtered = tmp_path / "filtered.parquet"
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_a", "stv_b"],
                "timescale": ["stv", "stv"],
                "lc_path": ["a.dat2", "b.dat2"],
                "failed_any": [0, 0],
            }
        ),
        raw,
    )
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_b", "stv_c"],
                "timescale": ["stv", "stv"],
                "lc_path": ["b.dat2", "c.dat2"],
                "failed_any": [0, 1],
            }
        ),
        filtered,
    )

    report = compare_results(raw, filtered, key="lc_path", sample=5)

    assert report["raw"]["rows"] == 2
    assert report["filtered"]["failed_any_true"] == 1
    assert report["raw_vs_filtered"]["left_only_sample"] == ["a.dat2"]
    assert report["raw_vs_filtered"]["right_only_sample"] == ["c.dat2"]
