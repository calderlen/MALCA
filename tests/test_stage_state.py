from __future__ import annotations

import pytest

from malca.products.stage_state import (
    StageResult,
    assert_reusable_stage_state,
    build_stage_fingerprint,
    read_stage_state,
    write_stage_state,
)


def test_stage_result_requires_balanced_terminal_accounting() -> None:
    with pytest.raises(ValueError, match="balance"):
        StageResult(stage="events", status="partial", expected=3, succeeded=1, failed=1)
    with pytest.raises(ValueError, match="successful"):
        StageResult(stage="events", status="success", expected=1, failed=1)


def test_stage_state_roundtrip_and_fingerprint_invalidation(tmp_path) -> None:
    lc = tmp_path / "A.dat3"
    lc.write_text("1 13 0.1\n")
    fingerprint = build_stage_fingerprint(
        stage="events",
        stage_version="2",
        candidate_ids=["A"],
        input_paths=[lc],
        settings={"threshold": 0.1},
    )
    state_path = tmp_path / "events.stage.json"
    write_stage_state(
        state_path,
        fingerprint=fingerprint,
        result=StageResult(stage="events", status="success", expected=1, succeeded=1),
        outputs=[lc],
    )

    state = read_stage_state(state_path)
    assert_reusable_stage_state(state, fingerprint=fingerprint, require_complete=True)

    changed = build_stage_fingerprint(
        stage="events",
        stage_version="2",
        candidate_ids=["A"],
        input_paths=[lc],
        settings={"threshold": 0.2},
    )
    with pytest.raises(ValueError, match="does not match"):
        assert_reusable_stage_state(state, fingerprint=changed)


def test_candidate_fingerprint_rejects_duplicates(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_stage_fingerprint(
            stage="tag",
            stage_version="2",
            candidate_ids=["A", "A"],
        )


def test_stage_reuse_rejects_modified_output(tmp_path) -> None:
    input_path = tmp_path / "input.dat2"
    output_path = tmp_path / "output.parquet"
    state_path = tmp_path / "stage.json"
    input_path.write_text("input", encoding="utf-8")
    output_path.write_text("first", encoding="utf-8")
    fingerprint = build_stage_fingerprint(
        stage="test",
        stage_version="1",
        candidate_ids=["A"],
        input_paths=[input_path],
    )
    write_stage_state(
        state_path,
        fingerprint=fingerprint,
        result=StageResult(stage="test", status="success", expected=1, succeeded=1),
        outputs=[output_path],
    )
    assert_reusable_stage_state(
        read_stage_state(state_path),
        fingerprint=fingerprint,
        require_complete=True,
    )

    output_path.write_text("second", encoding="utf-8")
    with pytest.raises(ValueError, match="output no longer matches"):
        assert_reusable_stage_state(
            read_stage_state(state_path),
            fingerprint=fingerprint,
            require_complete=True,
        )
