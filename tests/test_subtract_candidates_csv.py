from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "subtract_candidates_csv.py"
SPEC = importlib.util.spec_from_file_location("subtract_candidates_csv", SCRIPT_PATH)
assert SPEC is not None
subtract_candidates_csv = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subtract_candidates_csv)


def test_subtract_candidates_auto_detects_candidate_id(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    existing = tmp_path / "existing.csv"
    output = tmp_path / "new.csv"

    pd.DataFrame(
        {
            "candidate_id": ["A-1", "B-2", "123.0", "keep_blank"],
            "score": ["10", "20", "30", "40"],
        }
    ).to_csv(candidates, index=False)
    pd.DataFrame({"candidate_id": ["a-1", "123"], "flag": ["seen", "seen"]}).to_csv(
        existing,
        index=False,
    )

    summary = subtract_candidates_csv.subtract_candidates(candidates, existing, output)

    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert result["candidate_id"].tolist() == ["B-2", "keep_blank"]
    assert summary["removed_rows"] == 2
    assert summary["output_rows"] == 2


def test_subtract_candidates_accepts_different_key_columns(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    existing = tmp_path / "existing.csv"
    output = tmp_path / "new.csv"

    pd.DataFrame({"candidate_id": ["1", "2", "3"], "label": ["one", "two", "three"]}).to_csv(
        candidates,
        index=False,
    )
    pd.DataFrame({"asas_sn_id": ["2"]}).to_csv(existing, index=False)

    summary = subtract_candidates_csv.subtract_candidates(
        candidates,
        existing,
        output,
        left_key="candidate_id",
        right_key="asas_sn_id",
    )

    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert result["candidate_id"].tolist() == ["1", "3"]
    assert summary["left_key"] == "candidate_id"
    assert summary["right_key"] == "asas_sn_id"
