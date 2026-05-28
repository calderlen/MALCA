from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.candidates import ensure_candidate_id, merge_candidate_columns, passing_candidates_mask


def test_ensure_candidate_id_infers_from_asas_sn_id_with_prefix() -> None:
    df = pd.DataFrame({"asas_sn_id": ["1001", "1002"]})

    out = ensure_candidate_id(df, prefix="ltv")

    assert out["candidate_id"].tolist() == ["ltv_1001", "ltv_1002"]


def test_ensure_candidate_id_infers_from_source_id_and_path(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "source_id": ["", None],
            "path": [tmp_path / "unused.dat2", tmp_path / "ABC-123.dat2"],
        }
    )

    out = ensure_candidate_id(df, prefix="stv")

    assert out["candidate_id"].tolist() == ["stv_unused", "stv_ABC-123"]


def test_ensure_candidate_id_does_not_double_prefix() -> None:
    df = pd.DataFrame({"candidate_id": ["stv_1001", "1002"]})

    out = ensure_candidate_id(df, prefix="stv")

    assert out["candidate_id"].tolist() == ["stv_1001", "stv_1002"]


def test_passing_candidates_mask_coerces_common_failed_any_types() -> None:
    df = pd.DataFrame({"failed_any": [False, True, 0, 1, "false", "true", "yes", "", None]})

    assert passing_candidates_mask(df).tolist() == [True, False, True, False, True, False, False, True, True]


def test_merge_candidate_columns_updates_selected_values() -> None:
    base = pd.DataFrame({"candidate_id": ["stv_1", "stv_2"], "score": [1.0, 2.0]})
    extra = pd.DataFrame({"candidate_id": ["stv_1", "stv_2"], "score": [None, 4.0], "label": ["a", "b"]})

    out = merge_candidate_columns(base, extra, ["score", "label"])

    assert out["score"].tolist() == [1.0, 4.0]
    assert out["label"].tolist() == ["a", "b"]
