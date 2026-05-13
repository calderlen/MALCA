from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("astroquery")
pytest.importorskip("dustmaps3d")
pytest.importorskip("banyan_sigma")

from malca.detect import (
    _branch_events_attempted_this_run,
    _build_filter_kwargs,
    _build_home_external_validation_cmd,
    _select_passing_candidates,
    _should_skip_filter_stage,
    main as detect_main,
)


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        skip_evidence_strength=False,
        min_bayes_factor=10.0,
        allow_infinite_local_bf=False,
        skip_significant_detection=False,
        significant_no_require_flag=False,
        significant_min_peak_count=1,
        significant_min_run_count=1,
        skip_run_robustness=False,
        min_run_count=1,
        max_run_count=None,
        filter_min_run_points=2,
        filter_min_run_cameras=2,
        apply_morphology=False,
        dip_morphology="gaussian",
        jump_morphology="paczynski",
        min_delta_bic=10.0,
        apply_score_filter=True,
        min_score=0.0,
        min_dip_score=None,
        min_jump_score=None,
        apply_periodicity_validation=False,
        periodicity_n_bootstrap=1000,
        periodicity_significance=0.01,
        periodicity_pdm_method="plavchan",
        periodicity_no_exclude_aliases=False,
        periodicity_reject=False,
        periodicity_all_candidates=False,
        periodicity_workers=4,
        periodicity_checkpoint_dir=None,
        skip_gaia_ruwe_validation=False,
        gaia_max_ruwe=1.4,
        gaia_reject=False,
        skip_gaia_pm_validation=False,
        gaia_max_pm=100.0,
        gaia_pm_reject=False,
        skip_periodic_catalog_validation=False,
        periodic_catalog_max_sep=3.0,
        periodic_catalog_reject=False,
        phase_plot_max_sig=0.01,
        phase_plot_min_power=0.3,
        phase_plot_allow_alias=False,
        verbose=False,
    )


def test_build_filter_kwargs_defaults_match_pipeline_behavior() -> None:
    kwargs = _build_filter_kwargs(_base_args())

    assert kwargs["apply_evidence_strength"] is True
    assert kwargs["apply_significant_detection"] is True
    assert kwargs["significant_require_flag"] is True
    assert kwargs["significant_min_peak_count"] == 1
    assert kwargs["significant_min_run_count"] == 1
    assert kwargs["apply_run_robustness"] is True
    assert kwargs["max_run_count"] is None
    assert kwargs["apply_score"] is True
    assert kwargs["min_dip_score"] is None
    assert kwargs["min_jump_score"] is None
    assert kwargs["apply_gaia_ruwe_validation"] is True
    assert kwargs["apply_gaia_pm_validation"] is True
    assert kwargs["apply_periodic_catalog_validation"] is True

    assert kwargs["apply_morphology"] is False
    assert kwargs["apply_periodicity_validation"] is False
    assert kwargs["periodicity_pdm_method"] == "plavchan"

    assert kwargs["gaia_flag_only"] is True
    assert kwargs["gaia_max_pm"] == 100.0
    assert kwargs["gaia_pm_flag_only"] is True
    assert kwargs["periodic_catalog_flag_only"] is True
    assert kwargs["periodicity_flag_only"] is True
    assert kwargs["periodicity_all_candidates"] is False

    assert kwargs["phase_plot_max_sig"] == 0.01
    assert kwargs["phase_plot_min_power"] == 0.3
    assert kwargs["phase_plot_allow_alias"] is False


def test_build_filter_kwargs_respects_cli_overrides() -> None:
    args = _base_args()
    args.apply_score_filter = False
    args.skip_significant_detection = True
    args.significant_no_require_flag = True
    args.significant_min_peak_count = 3
    args.significant_min_run_count = 2
    args.apply_morphology = True
    args.dip_morphology = "paczynski"
    args.jump_morphology = "gaussian"
    args.min_delta_bic = 7.5
    args.max_run_count = 4
    args.min_dip_score = 1.2
    args.min_jump_score = 0.4
    args.apply_periodicity_validation = True
    args.periodicity_n_bootstrap = 250
    args.periodicity_significance = 0.02
    args.periodicity_pdm_method = "classic"
    args.periodicity_no_exclude_aliases = True
    args.periodicity_reject = True
    args.periodicity_all_candidates = True
    args.periodicity_workers = 2
    args.periodicity_checkpoint_dir = Path("output/checkpoints")
    args.phase_plot_max_sig = 0.05
    args.phase_plot_min_power = 0.5
    args.phase_plot_allow_alias = True
    args.skip_gaia_ruwe_validation = True
    args.gaia_reject = True
    args.skip_gaia_pm_validation = True
    args.gaia_max_pm = 50.0
    args.gaia_pm_reject = True
    args.skip_periodic_catalog_validation = True
    args.periodic_catalog_reject = True

    kwargs = _build_filter_kwargs(args)

    assert kwargs["apply_score"] is False
    assert kwargs["apply_significant_detection"] is False
    assert kwargs["significant_require_flag"] is False
    assert kwargs["significant_min_peak_count"] == 3
    assert kwargs["significant_min_run_count"] == 2
    assert kwargs["max_run_count"] == 4
    assert kwargs["apply_morphology"] is True
    assert kwargs["dip_morphology"] == "paczynski"
    assert kwargs["jump_morphology"] == "gaussian"
    assert kwargs["min_delta_bic"] == 7.5
    assert kwargs["min_dip_score"] == 1.2
    assert kwargs["min_jump_score"] == 0.4

    assert kwargs["apply_periodicity_validation"] is True
    assert kwargs["periodicity_n_bootstrap"] == 250
    assert kwargs["periodicity_significance"] == 0.02
    assert kwargs["periodicity_pdm_method"] == "classic"
    assert kwargs["periodicity_exclude_aliases"] is False
    assert kwargs["periodicity_flag_only"] is False
    assert kwargs["periodicity_all_candidates"] is True
    assert kwargs["periodicity_workers"] == 2
    assert kwargs["periodicity_checkpoint_dir"] == Path("output/checkpoints")

    assert kwargs["phase_plot_max_sig"] == 0.05
    assert kwargs["phase_plot_min_power"] == 0.5
    assert kwargs["phase_plot_allow_alias"] is True

    assert kwargs["apply_gaia_ruwe_validation"] is False
    assert kwargs["gaia_flag_only"] is False
    assert kwargs["apply_gaia_pm_validation"] is False
    assert kwargs["gaia_max_pm"] == 50.0
    assert kwargs["gaia_pm_flag_only"] is False

    assert kwargs["apply_periodic_catalog_validation"] is False
    assert kwargs["periodic_catalog_flag_only"] is False


def test_select_passing_candidates_filters_truthy_failed_any_values() -> None:
    df = pd.DataFrame(
        {
            "path": ["a", "b", "c", "d"],
            "failed_any": [False, True, "yes", 0],
        }
    )

    out = _select_passing_candidates(df)

    assert out["path"].tolist() == ["a", "d"]


def test_filter_stage_skip_requires_existing_output_and_no_new_event_attempts(tmp_path: Path) -> None:
    output = tmp_path / "lc_events_filtered_13_13.5.parquet"
    output.write_bytes(b"exists")
    stats = {
        "stochastic": {"attempted_this_run": 0},
        "periodic": {"attempted_this_run": 0},
    }

    assert _branch_events_attempted_this_run(stats) == 0
    assert _should_skip_filter_stage(
        output_path=output,
        overwrite=False,
        branch_detection_stats=stats,
    )


def test_filter_stage_does_not_skip_when_events_attempted(tmp_path: Path) -> None:
    output = tmp_path / "lc_events_filtered_13_13.5.parquet"
    output.write_bytes(b"exists")
    stats = {
        "stochastic": {"attempted_this_run": 0},
        "periodic": {"attempted_this_run": 3},
    }

    assert _branch_events_attempted_this_run(stats) == 3
    assert not _should_skip_filter_stage(
        output_path=output,
        overwrite=False,
        branch_detection_stats=stats,
    )


def test_filter_stage_does_not_skip_without_stats_or_when_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "lc_events_filtered_13_13.5.parquet"
    output.write_bytes(b"exists")

    assert not _should_skip_filter_stage(
        output_path=output,
        overwrite=False,
        branch_detection_stats=None,
    )
    assert not _should_skip_filter_stage(
        output_path=output,
        overwrite=True,
        branch_detection_stats={"stochastic": {"attempted_this_run": 0}},
    )


def test_build_home_external_validation_cmd_forwards_periodicity_options() -> None:
    args = _base_args()
    args.apply_periodicity_validation = True
    args.periodicity_n_bootstrap = 250
    args.periodicity_significance = 0.02
    args.periodicity_pdm_method = "classic"
    args.periodicity_no_exclude_aliases = True
    args.periodicity_reject = True
    args.periodicity_all_candidates = True
    args.periodicity_workers = 2
    args.periodicity_checkpoint_dir = Path("output/checkpoints")
    args.phase_plot_max_sig = 0.05
    args.phase_plot_min_power = 0.5
    args.phase_plot_allow_alias = True
    args.verbose = True

    cmd = _build_home_external_validation_cmd(
        args,
        post_filter_output=Path("results/lc_events_filtered.parquet"),
        index_file=Path("input/index.parquet"),
    )

    assert "--apply-periodicity-validation" in cmd
    assert "--periodicity-n-bootstrap" in cmd
    assert "250" in cmd
    assert "--periodicity-significance" in cmd
    assert "0.02" in cmd
    assert "--periodicity-pdm-method" in cmd
    assert "classic" in cmd
    assert "--periodicity-no-exclude-aliases" in cmd
    assert "--periodicity-reject" in cmd
    assert "--periodicity-all-candidates" in cmd
    assert "--workers" in cmd
    assert "2" in cmd
    assert "--checkpoint-dir" in cmd
    assert "output/checkpoints" in cmd
    assert "--phase-plot-max-sig" in cmd
    assert "0.05" in cmd
    assert "--phase-plot-min-power" in cmd
    assert "0.5" in cmd
    assert "--phase-plot-allow-alias" in cmd
    assert "--verbose" in cmd


def test_pipeline_event_subprocesses_always_use_parquet_chunk(tmp_path: Path, monkeypatch) -> None:
    mag_bin = "13_13.5"
    source_id = "ASASSN-TEST-001"
    lcsv2 = tmp_path / "lcsv2"
    index_dir = lcsv2 / mag_bin
    lc_dir = lcsv2 / mag_bin / "lc1_cal"
    index_dir.mkdir(parents=True)
    lc_dir.mkdir(parents=True)
    (index_dir / "index1.csv").write_text(f"asas_sn_id\n{source_id}\n", encoding="ascii")
    (lc_dir / f"{source_id}.dat2").write_text(
        "1 13.0 0.01 0 1 0 0 cam/field\n",
        encoding="ascii",
    )

    out_dir = tmp_path / "run"
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline": {
                    "extension": "dat3",
                    "trigger_mode": "posterior_prob",
                    "baseline_func": "gp",
                    "skip_sparse": True,
                    "skip_multi_camera": True,
                    "skip_mag_range": True,
                    "skip_vsx": True,
                    "skip_camera_median": True,
                    "run_filter": False,
                    "export_bundle_enabled": False,
                    "review_sync_enabled": False,
                }
            }
        ),
        encoding="ascii",
    )

    captured_cmds: list[list[str]] = []
    captured_paths: list[str] = []
    export_path = tmp_path / "custom_bundle.zip"

    def fake_run(cmd: list[str], check: bool = False):
        captured_cmds.append(list(cmd))
        input_file = Path(cmd[cmd.index("--input-file") + 1])
        output_dir = Path(cmd[cmd.index("--output") + 1])
        paths = [line.strip() for line in input_file.read_text(encoding="ascii").splitlines() if line.strip()]
        captured_paths.extend(paths)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "path": paths,
                "dip_significant": [False] * len(paths),
                "jump_significant": [False] * len(paths),
            }
        ).to_parquet(output_dir / "chunk_000000.parquet", index=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("malca.detect.subprocess.run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca",
            "--mag-bin",
            mag_bin,
            "--index-root",
            str(lcsv2),
            "--lc-root",
            str(lcsv2),
            "--output-dir",
            str(out_dir),
            "--stage",
            "cluster",
            "--extension",
            "dat2",
            "--trigger-mode",
            "logbf",
            "--baseline-func",
            "per_camera_median",
            "--export-bundle",
            str(export_path),
            "--config",
            str(config_path),
            "--workers",
            "1",
            "--overwrite",
        ],
    )

    detect_main()

    assert captured_cmds
    for cmd in captured_cmds:
        assert cmd[cmd.index("--output-format") + 1] == "parquet_chunk"
        assert cmd[cmd.index("--trigger-mode") + 1] == "logbf"
        assert cmd[cmd.index("--baseline-func") + 1] == "per_camera_median"
    assert captured_paths
    assert all(Path(path).suffix == ".dat2" for path in captured_paths)
    assert export_path.exists()

    branch_chunk = (
        out_dir
        / "results"
        / "_branch_events"
        / f"lc_events_stochastic_branch_{mag_bin}"
        / "chunk_000000.parquet"
    )
    canonical_chunk = (
        out_dir
        / "results"
        / f"lc_events_results_{mag_bin}"
        / "chunk_000000.parquet"
    )
    assert branch_chunk.exists()
    assert canonical_chunk.exists()
