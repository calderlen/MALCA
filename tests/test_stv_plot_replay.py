from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from malca.core.baseline import global_median_baseline, phase_template_baseline
from malca.stv import plot
from malca.stv.plot import (
    _build_replay_score_kwargs,
    _match_detection_result_row,
    _prepare_results_mode_input,
    _resolve_replay_baseline,
    compare_detection_replay,
    load_detection_results,
    lookup_source_metadata,
)


def test_compare_detection_replay_uses_delta_magnitude() -> None:
    stored = {
        "dip_significant": True,
        "jump_significant": False,
        "dip_best_delta_mag": 0.25,
        "jump_best_delta_mag": -0.05,
    }
    replay = {
        "dip": {"significant": True, "best_delta_mag": 0.25},
        "jump": {"significant": False, "best_delta_mag": -0.05},
    }

    assert compare_detection_replay(stored, replay) == []


def test_compare_detection_replay_flags_science_mismatch() -> None:
    stored = {
        "dip_significant": True,
        "jump_significant": False,
        "dip_best_delta_mag": 0.25,
        "jump_best_delta_mag": -0.05,
    }
    replay = {
        "dip": {"significant": False, "best_delta_mag": 0.10},
        "jump": {"significant": False, "best_delta_mag": -0.05},
    }

    mismatches = compare_detection_replay(stored, replay)

    assert mismatches == ["dip_significant", "dip_best_delta_mag"]


def test_compare_detection_replay_flags_one_sided_missing_values() -> None:
    stored = {
        "dip_significant": True,
        "jump_significant": False,
        "dip_best_delta_mag": 0.25,
        "jump_best_delta_mag": None,
    }
    replay = {
        "dip": {"significant": True, "best_delta_mag": None},
        "jump": {"significant": False, "best_delta_mag": -0.05},
    }

    assert compare_detection_replay(stored, replay) == [
        "dip_best_delta_mag_replay_unavailable",
        "jump_best_delta_mag_stored_unavailable",
    ]


def test_stored_result_matching_prefers_exact_path_over_shared_identifier() -> None:
    target = Path("/data/reduction-a/source.dat")
    stored = pd.DataFrame(
        [
            {
                "lc_path": "/data/reduction-b/source.dat",
                "asas_sn_id": "source",
                "dip_best_delta_mag": 0.9,
            },
            {
                "lc_path": str(target),
                "asas_sn_id": "different-id",
                "dip_best_delta_mag": 0.2,
            },
        ]
    )

    row = _match_detection_result_row(stored, target, asas_sn_id="source")

    assert row is not None
    assert row["lc_path"] == str(target)
    assert row["dip_best_delta_mag"] == 0.2


def test_canonical_lc_path_supports_metadata_lookup_without_legacy_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    table_path = tmp_path / "results.parquet"
    table_path.touch()
    source = pd.DataFrame(
        [
            {
                "lc_path": "/other/source.dat",
                "asas_sn_id": "source",
                "Category": "wrong",
            },
            {
                "lc_path": "/exact/source.dat",
                "asas_sn_id": "different-id",
                "Category": "right",
            },
        ]
    )
    monkeypatch.setattr(plot, "read_feature_table", lambda _path: source.copy())

    loaded = load_detection_results(table_path)
    monkeypatch.setattr(plot, "load_detection_results", lambda _path: loaded)
    metadata = lookup_source_metadata(
        asassn_id="source",
        dat_path="/exact/source.dat",
        csv_path=table_path,
    )

    assert loaded["Match_ID"].tolist() == ["source", "source"]
    assert metadata is not None
    assert metadata["dat_path"] == "/exact/source.dat"
    assert metadata["source_id"] == "different-id"
    assert metadata["category"] == "right"


def test_periodic_result_row_selects_phase_template_and_its_period() -> None:
    baseline_func, baseline_name, kwargs = _resolve_replay_baseline(
        global_median_baseline,
        "global_median",
        {"period_days": 99.0},
        {
            "baseline_source": "phase_template",
            "pre_periodicity_selected_period": 6.25,
        },
    )

    assert baseline_func is phase_template_baseline
    assert baseline_name == "phase_template"
    assert kwargs["period_days"] == 6.25


def test_replay_score_kwargs_restore_mag_grids_and_event_probability_setting() -> None:
    kwargs = _build_replay_score_kwargs(
        {
            "mag_points": 4,
            "mag_min_dip": 0.1,
            "mag_max_dip": 0.4,
            "mag_min_jump": -0.8,
            "mag_max_jump": -0.2,
            "no_event_prob": True,
        },
        logbf_threshold_dip=5.0,
        logbf_threshold_jump=6.0,
        filter_bad_cameras=True,
        bad_camera_scatter_ratio=2.5,
    )

    np.testing.assert_allclose(kwargs["mag_grid_dip"], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(kwargs["mag_grid_jump"], [-0.8, -0.6, -0.4, -0.2])
    assert kwargs["compute_event_prob"] is False


def test_prepare_results_mode_input_applies_path_column_and_significance(
    monkeypatch,
) -> None:
    source = pd.DataFrame(
        {
            "candidate_id": ["one", "two", "three"],
            "custom_lc": ["one.dat", "two.dat", "three.dat"],
            "lc_path": ["stale-1.dat", "stale-2.dat", "stale-3.dat"],
            "dip_significant": [False, "true", False],
            "jump_significant": [False, False, True],
        }
    )
    monkeypatch.setattr(plot, "read_feature_table", lambda _path: source.copy())

    prepared = _prepare_results_mode_input(
        Path("results.parquet"),
        path_col="custom_lc",
        only_significant=True,
    )

    assert prepared["candidate_id"].tolist() == ["two", "three"]
    assert prepared["lc_path"].tolist() == ["two.dat", "three.dat"]


def test_results_mode_writes_plot_log_before_return(
    tmp_path: Path,
    monkeypatch,
) -> None:
    detect_run = tmp_path / "run"
    detect_run.mkdir()
    (detect_run / "run_params.json").write_text(
        json.dumps({"baseline_func": "global_median"}),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.parquet"
    captured: dict[str, object] = {}
    prepared = pd.DataFrame({"candidate_id": ["one"], "lc_path": ["one.dat"]})

    def fake_prepare(results_path_arg, *, path_col, only_significant):
        captured["results_path"] = results_path_arg
        captured["path_col"] = path_col
        captured["only_significant"] = only_significant
        return prepared

    def fake_plot(plot_input, _out_dir, **_kwargs):
        captured["plot_input"] = plot_input
        return {"total_selected": 1, "plotted": 1, "failed": 0}

    monkeypatch.setattr(plot, "_prepare_results_mode_input", fake_prepare)
    monkeypatch.setattr(plot, "plot_passing_candidates", fake_plot)
    monkeypatch.setattr(
        plot.sys,
        "argv",
        [
            "stv-plot",
            "--detect-run",
            str(detect_run),
            "--results",
            str(results_path),
            "--path-col",
            "custom_lc",
            "--only-significant",
            "--workers",
            "1",
        ],
    )

    plot.main()

    assert captured["path_col"] == "custom_lc"
    assert captured["only_significant"] is True
    assert captured["plot_input"] is prepared
    log = json.loads((detect_run / "plot_log.json").read_text(encoding="utf-8"))
    assert log["plot_params"]["path_col"] == "custom_lc"
    assert log["plot_params"]["only_significant"] is True
    assert log["results"]["total_plots"] == 1


def test_phase_folded_plot_remains_independent_of_event_replay_arguments(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "JD": np.arange(8, dtype=float),
            "mag": [14.0, 14.1, 14.0, 14.2, 14.0, 14.1, 14.0, 14.2],
            "error": [0.02] * 8,
            "good_bad": [1] * 8,
            "saturated": [0] * 8,
            "camera#": [1] * 8,
            "v_g_band": [0, 1] * 4,
        }
    )
    monkeypatch.setattr(
        plot,
        "load_lightcurve_df",
        lambda *_args, **_kwargs: (frame.copy(), set()),
    )

    filtered = plot.plot_phase_folded_lightcurve(
        Path("source-with-hyphen.dat2"),
        period_days=2.0,
        show=False,
        return_filtered_cameras=True,
    )

    assert filtered == set()
