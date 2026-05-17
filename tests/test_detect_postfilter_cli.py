from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("astroquery")
pytest.importorskip("dustmaps3d")
pytest.importorskip("banyan_sigma")

from malca.detect import (
    PIPELINE_STAGE_CHOICES,
    _branch_events_attempted_this_run,
    _build_filter_kwargs,
    _build_home_external_validation_cmd,
    _candidate_result_priority,
    _collect_bundle_lightcurve_files,
    _copy_single_tagged_table_output,
    _first_existing_candidate_result,
    _run_external_lcs_enrichment,
    _run_multi_survey_features_enrichment,
    _select_passing_candidates,
    _should_skip_filter_stage,
    _stage_defaults_to_extended_enrichment,
    _stage_runs_downstream,
    _stage_runs_upstream,
    main as detect_main,
)
from malca.config import EVENTS_OUTPUT_CHUNK_SIZE


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


def test_full_extended_stage_semantics() -> None:
    assert "full-extended" in PIPELINE_STAGE_CHOICES
    assert _stage_runs_upstream("full-extended")
    assert _stage_runs_downstream("full-extended")
    assert _stage_defaults_to_extended_enrichment("full-extended")
    assert not _stage_defaults_to_extended_enrichment("full")


def test_candidate_result_priority_prefers_extended_outputs(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for name in [
        "lc_events_classified.parquet",
        "lc_events_vetted.parquet",
        "lc_events_external_lcs.parquet",
        "lc_events_multi_survey_features.parquet",
    ]:
        (results_dir / name).write_text("x", encoding="ascii")

    priority = _candidate_result_priority(results_dir)

    assert priority[0].name == "lc_events_multi_survey_features.parquet"
    assert priority[1].name == "lc_events_external_lcs.parquet"
    assert _first_existing_candidate_result(results_dir).name == "lc_events_multi_survey_features.parquet"
    assert _first_existing_candidate_result(results_dir, include_extended=False).name == "lc_events_vetted.parquet"


def test_bundle_lightcurves_reads_only_passing_candidates(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pass_lc = tmp_path / "pass.dat2"
    fail_lc = tmp_path / "fail.dat2"
    pass_lc.write_text("1 13.0 0.1\n", encoding="ascii")
    fail_lc.write_text("1 14.0 0.1\n", encoding="ascii")
    pd.DataFrame(
        {
            "path": [str(pass_lc), str(fail_lc)],
            "failed_any": [False, True],
        }
    ).to_parquet(results_dir / "lc_events_filtered.parquet", index=False)

    files = _collect_bundle_lightcurve_files(tmp_path)

    assert [path.name for path, _arcname in files] == ["pass.dat2"]


def test_single_tagged_table_merge_copies_without_reading(tmp_path: Path) -> None:
    source = tmp_path / "lc_events_filtered_all.parquet"
    merged = tmp_path / "lc_events_filtered.parquet"
    source.write_bytes(b"not actually parquet")

    assert _copy_single_tagged_table_output([source], merged)
    assert merged.read_bytes() == b"not actually parquet"


def test_extended_enrichment_helpers_use_passers_and_safe_defaults(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured["external_ids"] = df["candidate_id"].tolist()
        captured["external_kwargs"] = kwargs
        out = df.copy()
        out["ztf_lc_n_det"] = 3
        return out

    def fake_compute_multi_survey_features(df: pd.DataFrame, *, external_lc_dir: Path) -> pd.DataFrame:
        captured["multi_ids"] = df["candidate_id"].tolist()
        captured["multi_external_lc_dir"] = external_lc_dir
        out = df.copy()
        out["ms_feature_status"] = "ok"
        return out

    monkeypatch.setattr("malca.vetting.fetch_external_lcs", fake_fetch_external_lcs)
    monkeypatch.setattr("malca.multi_survey_features.compute_multi_survey_features", fake_compute_multi_survey_features)

    df = pd.DataFrame({
        "candidate_id": ["C1", "C2"],
        "failed_any": [False, True],
        "ra": [10.0, 11.0],
        "dec": [20.0, 21.0],
    })
    results_dir = tmp_path / "results"

    external_path, external_dir, external_df = _run_external_lcs_enrichment(
        df,
        results_dir=results_dir,
        atlas=False,
        atlas_token=None,
        workers=7,
        refresh_cache=True,
    )
    multi_path, multi_df = _run_multi_survey_features_enrichment(
        external_df,
        results_dir=results_dir,
        external_lc_dir=external_dir,
    )

    kwargs = captured["external_kwargs"]
    assert captured["external_ids"] == ["C1"]
    assert kwargs["run_atlas"] is False
    assert kwargs["run_ztf"] is True
    assert kwargs["run_gaia_epoch"] is True
    assert kwargs["run_tess"] is True
    assert kwargs["run_neowise"] is True
    assert kwargs["run_ps1"] is True
    assert kwargs["run_crts"] is True
    assert kwargs["run_kepler"] is False
    assert kwargs["run_aavso"] is False
    assert kwargs["workers"] == 7
    assert kwargs["refresh_cache"] is True
    assert captured["multi_ids"] == ["C1"]
    assert captured["multi_external_lc_dir"] == external_dir
    assert external_path.exists()
    assert multi_path.exists()
    assert external_df["ztf_lc_n_det"].iloc[0] == 3
    assert multi_df["ms_feature_status"].iloc[0] == "ok"


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
    assert kwargs["external_validations_passers_only"] is True

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
    args.external_validations_passers_only = False

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
    assert kwargs["external_validations_passers_only"] is False


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
    assert "--external-validations-passers-only" in cmd
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
    stale_manifest = out_dir / "manifests" / f"lc_manifest_{mag_bin}.parquet"
    stale_filtered = out_dir / "tags" / f"lc_filtered_{mag_bin}.parquet"
    stale_manifest.parent.mkdir(parents=True)
    stale_filtered.parent.mkdir(parents=True)
    pd.DataFrame(
        columns=[
            "source_id",
            "mag_bin",
            "index_num",
            "index_csv",
            "lc_dir",
            "lc_dir_exists",
            "dat_path",
            "dat_exists",
        ]
    ).to_parquet(stale_manifest, index=False)
    pd.DataFrame(columns=["source_id", "path", "mag_bin"]).to_parquet(stale_filtered, index=False)

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
        assert int(cmd[cmd.index("--chunk-size") + 1]) == EVENTS_OUTPUT_CHUNK_SIZE
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


def test_pipeline_home_accepts_import_bundle_cli(tmp_path: Path, monkeypatch) -> None:
    bundle_zip = tmp_path / "cluster_bundle.zip"
    bundle_src = tmp_path / "bundle_src"
    filtered = bundle_src / "results" / "lc_events_filtered.parquet"
    filtered.parent.mkdir(parents=True)
    pd.DataFrame(columns=["path", "failed_any"]).to_parquet(filtered, index=False)
    (bundle_src / "run_params.json").write_text(
        json.dumps({"mag_bin": ["13_13.5"]}),
        encoding="ascii",
    )

    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.write(bundle_src / "run_params.json", "run_params.json")
        zf.write(filtered, "results/lc_events_filtered.parquet")

    config_path = tmp_path / "home_config.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline": {
                    "run_filter": False,
                    "run_characterize": False,
                    "run_dust": False,
                    "run_classify": False,
                    "run_neighbor_enrich": False,
                    "run_spectra_enrich": False,
                    "run_vetting": False,
                    "export_bundle_enabled": False,
                    "review_sync_enabled": False,
                }
            }
        ),
        encoding="ascii",
    )

    out_dir = tmp_path / "home_run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca",
            "--stage",
            "home",
            "--import-bundle",
            str(bundle_zip),
            "--output-dir",
            str(out_dir),
            "--config",
            str(config_path),
            "--overwrite",
        ],
    )

    detect_main()

    assert (out_dir / "run_params.json").exists()
    assert (out_dir / "results" / "lc_events_filtered.parquet").exists()
