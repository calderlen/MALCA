from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.audit import compare_results


def test_compare_results_reports_filtered_only_and_raw_missing_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw.parquet"
    filtered = tmp_path / "filtered.parquet"
    pd.DataFrame({"path": ["a.dat2", "b.dat2"], "failed_any": [0, 0]}).to_parquet(raw, index=False)
    pd.DataFrame({"path": ["b.dat2", "c.dat2"], "failed_any": [0, 1]}).to_parquet(filtered, index=False)

    report = compare_results(raw, filtered, key="path", sample=5)

    assert report["raw"]["rows"] == 2
    assert report["filtered"]["failed_any_true"] == 1
    assert report["raw_vs_filtered"]["left_only_sample"] == ["a.dat2"]
    assert report["raw_vs_filtered"]["right_only_sample"] == ["c.dat2"]
