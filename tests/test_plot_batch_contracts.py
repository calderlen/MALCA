from pathlib import Path

import pandas as pd
import pytest

from malca.review.plot_batch import (
    _as_bool,
    _candidate_filename_token,
    _resolve_filtered_result,
    load_passing_candidates,
)


def test_plot_selection_rejects_duplicate_lightcurve_paths() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["stv_a", "stv_b"],
            "lc_path": ["same.dat2", "same.dat2"],
            "failed_any": [False, False],
        }
    )

    with pytest.raises(ValueError, match="duplicate light-curve paths"):
        load_passing_candidates(frame)


def test_plot_boolean_parser_does_not_treat_nan_as_true() -> None:
    assert _as_bool(float("nan")) is False
    assert _as_bool("False") is False
    assert _as_bool("true") is True


def test_candidate_filename_token_avoids_sanitization_collision() -> None:
    assert _candidate_filename_token("stv_a/b") != _candidate_filename_token("stv_a_b")
    assert _candidate_filename_token("stv_a_b") == "stv_a_b"


def test_filtered_result_resolution_is_fail_closed_when_ambiguous(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "lc_events_filtered_13_13.5.parquet").touch()
    (results / "lc_events_filtered_14_14.5.parquet").touch()

    with pytest.raises(ValueError, match="Multiple filtered products"):
        _resolve_filtered_result(results)

    canonical = results / "lc_events_filtered.parquet"
    canonical.touch()
    assert _resolve_filtered_result(results) == canonical
