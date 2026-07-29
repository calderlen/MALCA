from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malca.products.candidates import (
    CandidateSelectionError,
    ensure_candidate_id,
    merge_candidate_columns,
    passing_candidates_mask,
    validate_candidate_ids,
)


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
    df = pd.DataFrame({"failed_any": [False, True, 0, 1, "false", "true", "yes", "no"]})

    assert passing_candidates_mask(df).tolist() == [True, False, True, False, True, False, False, True]


@pytest.mark.parametrize("value", [None, "", "unknown", 2])
def test_passing_candidates_mask_rejects_unknown_state(value: object) -> None:
    with pytest.raises(CandidateSelectionError):
        passing_candidates_mask(pd.DataFrame({"failed_any": [value]}))


def test_passing_candidates_mask_requires_filter_decision() -> None:
    with pytest.raises(CandidateSelectionError, match="required 'failed_any'"):
        passing_candidates_mask(pd.DataFrame({"candidate_id": ["stv_a"]}))


def test_validate_candidate_ids_rejects_blank_and_duplicate_values() -> None:
    with pytest.raises(ValueError, match="blank/null"):
        validate_candidate_ids(pd.DataFrame({"candidate_id": ["stv_a", None]}))
    with pytest.raises(ValueError, match="duplicate"):
        validate_candidate_ids(pd.DataFrame({"candidate_id": ["stv_a", "stv_a"]}))


def test_merge_candidate_columns_updates_selected_values() -> None:
    base = pd.DataFrame({"candidate_id": ["stv_1", "stv_2"], "score": [1.0, 2.0]})
    extra = pd.DataFrame({"candidate_id": ["stv_1", "stv_2"], "score": [None, 4.0], "label": ["a", "b"]})

    out = merge_candidate_columns(base, extra, ["score", "label"])

    assert out["score"].tolist() == [1.0, 4.0]
    assert out["label"].tolist() == ["a", "b"]


def test_merge_candidate_columns_rejects_ambiguous_duplicate_keys() -> None:
    base = pd.DataFrame({"candidate_id": ["stv_1"], "score": [1.0]})
    extra = pd.DataFrame(
        {"candidate_id": ["stv_1", "stv_1"], "score": [2.0, 3.0]}
    )

    with pytest.raises(ValueError, match="duplicate"):
        merge_candidate_columns(base, extra, ["score"])
