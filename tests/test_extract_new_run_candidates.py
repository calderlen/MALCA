from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_new_run_candidates.py"
SPEC = importlib.util.spec_from_file_location("extract_new_run_candidates", SCRIPT_PATH)
assert SPEC is not None
extract_new_run_candidates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(extract_new_run_candidates)


def test_extract_new_candidates_subtracts_existing_and_writes_compact_csv(tmp_path: Path) -> None:
    run_candidates = tmp_path / "lc_events_filtered_all.parquet"
    existing = tmp_path / "candidates_12_15_combined.csv"
    characterized = tmp_path / "lc_events_characterized.parquet"
    output = tmp_path / "new_candidates_12_15.csv"

    pd.DataFrame(
        {
            "path": [
                "/data/lcsv2/12_12.5/lc1_cal/100.dat2",
                "/data/lcsv2/14_14.5/lc2_cal/200.dat2",
                "/data/lcsv2/14.5_15/lc3_cal/300.dat2",
            ],
            "asas_sn_id": ["100", "200", "300"],
            "dip_bayes_factor": [1.0, 2.0, 3.0],
        }
    ).to_parquet(run_candidates, index=False)
    pd.DataFrame({"asas_sn_id": ["100.0", "300"]}).to_csv(existing, index=False)
    pd.DataFrame({"asas_sn_id": ["200"], "ra_deg": [12.3], "dec_deg": [-4.5]}).to_parquet(
        characterized,
        index=False,
    )

    summary = extract_new_run_candidates.extract_new_candidates(
        run_candidates,
        existing,
        output,
        characterized_path=characterized,
    )

    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert result.to_dict(orient="records") == [
        {
            "path": "/data/lcsv2/14_14.5/lc2_cal/200.dat2",
            "asas_sn_id": "200",
            "mag_bin": "14_14.5",
            "ra_deg": "12.3",
            "dec_deg": "-4.5",
        }
    ]
    assert summary["run_rows"] == 3
    assert summary["removed_rows"] == 2
    assert summary["output_rows"] == 1


def test_extract_new_candidates_can_write_all_columns(tmp_path: Path) -> None:
    run_candidates = tmp_path / "lc_events_filtered_all.parquet"
    existing = tmp_path / "existing.csv"
    output = tmp_path / "new.csv"

    pd.DataFrame(
        {
            "path": ["/data/lcsv2/13_13.5/lc1_cal/111.dat2"],
            "asas_sn_id": ["111"],
            "score": [9.5],
        }
    ).to_parquet(run_candidates, index=False)
    pd.DataFrame({"asas_sn_id": ["222"]}).to_csv(existing, index=False)

    extract_new_run_candidates.extract_new_candidates(
        run_candidates,
        existing,
        output,
        all_columns=True,
    )

    result = pd.read_csv(output)
    assert "score" in result.columns
    assert result.loc[0, "mag_bin"] == "13_13.5"
