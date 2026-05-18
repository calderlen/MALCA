from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.audit import baseline_compare_commands, compare_results


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


def test_baseline_compare_constructs_global_smoke_then_per_camera_full(tmp_path: Path) -> None:
    candidates = tmp_path / "lc_events_collect_candidates_14_14.5.csv"
    pd.DataFrame(
        {
            "path": [f"/data/{idx}.dat2" for idx in range(4)],
            "asas_sn_id": [str(idx) for idx in range(4)],
            "mag_bin": ["14_14.5"] * 4,
            "ra_deg": [0.0] * 4,
            "dec_deg": [0.0] * 4,
        }
    ).to_csv(candidates, index=False)

    report = baseline_compare_commands(candidates, output_root=tmp_path / "audit", smoke_count=2, workers=3)

    smoke_csv = Path(report["smoke_candidates"])
    assert smoke_csv.exists()
    assert len(pd.read_csv(smoke_csv)) == 2
    assert "--baseline-func global_median" in report["smoke_command_text"]
    assert "--baseline-func per_camera_median" in report["full_command_text"]
    assert "--workers 3" in report["full_command_text"]
