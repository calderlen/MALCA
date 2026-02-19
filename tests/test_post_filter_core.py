from __future__ import annotations

import pandas as pd
import pytest

post_filter = pytest.importorskip("malca.post_filter")


def test_filter_run_robustness_respects_max_run_count() -> None:
    df = pd.DataFrame(
        {
            "path": ["a.csv", "b.csv"],
            "dip_run_count": [2, 5],
            "jump_run_count": [0, 0],
            "dip_max_run_points": [4, 4],
            "jump_max_run_points": [0, 0],
            "dip_max_run_cameras": [2, 2],
            "jump_max_run_cameras": [0, 0],
        }
    )

    out = post_filter.filter_run_robustness(
        df,
        min_run_count=1,
        max_run_count=3,
        min_run_points=2,
        min_run_cameras=2,
    )

    assert list(out["path"]) == ["a.csv"]


def test_filter_score_branch_specific_thresholds() -> None:
    df = pd.DataFrame(
        {
            "path": ["dip.csv", "jump.csv", "none.csv"],
            "dipper_score": [1.2, -1.0, -2.0],
            "jumper_score": [-1.0, 0.8, -2.0],
        }
    )

    out = post_filter.filter_score(df, min_dip_score=0.5, min_jump_score=0.5)
    assert set(out["path"]) == {"dip.csv", "jump.csv"}


def test_filter_significant_detection_explicit_gate() -> None:
    df = pd.DataFrame(
        {
            "path": ["dip_ok.csv", "no_peak.csv", "flag_false.csv", "jump_ok.csv"],
            "dip_significant": [True, True, False, False],
            "jump_significant": [False, False, False, True],
            "dip_count": [1, 0, 1, 0],
            "jump_count": [0, 0, 0, 1],
            "dip_run_count": [1, 1, 1, 0],
            "jump_run_count": [0, 0, 0, 1],
        }
    )

    out = post_filter.filter_significant_detection(
        df,
        require_significant_flag=True,
        min_peak_count=1,
        min_run_count=1,
    )

    assert set(out["path"]) == {"dip_ok.csv", "jump_ok.csv"}


def test_apply_post_filters_tags_significant_detection_failures() -> None:
    df = pd.DataFrame(
        {
            "path": ["pass.csv", "fail.csv"],
            "dip_significant": [True, False],
            "jump_significant": [False, False],
            "dip_count": [1, 0],
            "jump_count": [0, 0],
            "dip_run_count": [1, 0],
            "jump_run_count": [0, 0],
        }
    )

    out = post_filter.apply_post_filters(
        df,
        apply_evidence_strength=False,
        apply_significant_detection=True,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=False,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
        verbose=False,
    )

    assert "failed_significant_detection" in out.columns
    assert out.set_index("path").loc["pass.csv", "failed_significant_detection"] == 0
    assert out.set_index("path").loc["fail.csv", "failed_significant_detection"] == 1
    assert out.set_index("path").loc["pass.csv", "failed_any"] == 0
    assert out.set_index("path").loc["fail.csv", "failed_any"] == 1
