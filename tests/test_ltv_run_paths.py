from __future__ import annotations

import argparse
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from malca.audit import ltv_status
from malca.ltv import pipeline as ltv_pipeline


def _write_review_db(path: Path, *, candidate_rows: int = 2, review_rows: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("create table candidates (candidate_id text)")
        conn.execute("create table reviews (candidate_id text, status text, event_class text)")
        conn.executemany(
            "insert into candidates(candidate_id) values (?)",
            [(f"ltv_{idx}",) for idx in range(candidate_rows)],
        )
        conn.executemany(
            "insert into reviews(candidate_id, status, event_class) values (?, ?, ?)",
            [(f"ltv_{idx}", "reviewed", "other") for idx in range(review_rows)],
        )
        conn.commit()


def _core_rows(tmp_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asas_sn_id": ["123", "456"],
            "candidate_id": ["ltv_123", "ltv_456"],
            "timescale": ["ltv", "ltv"],
            "ra": [1.0, 2.0],
            "dec": [0.0, 0.0],
            "ltv_slope": [0.4, 0.01],
            "ltv_max_diff": [0.6, 0.6],
            "ltv_median": [13.0, 13.0],
            "baseline_mag": [13.1, 13.1],
            "ltv_dispersion": [0.02, 0.02],
            "ltv_median_err": [0.02, 0.02],
            "pm_total": [0.0, 0.0],
            "neighbor_pm_contam": [False, False],
            "crowding_count": [0, 0],
            "lc_path": [str(tmp_path / "123.dat2"), str(tmp_path / "456.dat2")],
        }
    )


def test_ltv_pipeline_full_writes_audit_metadata_and_ingests_passers(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv_run"
    captured: dict[str, object] = {}

    def fake_run_core(args: argparse.Namespace, mag_bin: str, run_dir_arg: Path) -> Path:
        path = ltv_pipeline.ltv_core_output_path(mag_bin, run_dir_arg)
        path.parent.mkdir(parents=True, exist_ok=True)
        _core_rows(tmp_path).to_parquet(path, index=False)
        return path

    def fake_run_full_pipeline(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.copy()

    def fake_ingest(review_db: Path, df: pd.DataFrame, **kwargs: object) -> tuple[int, int]:
        captured["review_db"] = review_db
        captured["rows"] = len(df)
        captured["run_characterize"] = kwargs["run_characterize"]
        captured["source_path"] = kwargs["source_path"]
        return len(df), len(df)

    monkeypatch.setattr(ltv_pipeline, "_run_core_if_needed", fake_run_core)
    monkeypatch.setattr(ltv_pipeline, "run_full_pipeline", fake_run_full_pipeline)
    monkeypatch.setattr("malca.ltv.review.ingest_ltv_results", fake_ingest)

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        [
            "--mag-bin", "13_13.5",
            "--run-dir", str(run_dir),
            "--no-export-bundle",
            "--no-review-sync",
            "--skip-stats",
        ]
    )

    summary = ltv_pipeline.run_ltv_pipeline_cli(args)

    filtered = pd.read_parquet(run_dir / "results" / "LTvar13-13.5_filtered.parquet")
    enriched = pd.read_parquet(run_dir / "results" / "LTvar13-13.5_pipeline.parquet")

    assert len(filtered) == 2
    assert filtered["failed_any"].tolist() == [False, True]
    assert filtered["ltv_failed_slope"].tolist() == [False, True]
    assert len(enriched) == 1
    assert enriched.loc[0, "ltv_class"] == "ltv_candidate"
    assert (run_dir / "run_params.json").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "run.log").exists()
    assert captured["rows"] == 1
    assert captured["run_characterize"] is False
    assert captured["source_path"] == run_dir
    assert summary["per_bin"]["13_13.5"]["passing_rows"] == 1


def test_ltv_pipeline_cluster_skips_downstream_ingest(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv_cluster"

    def fake_run_core(args: argparse.Namespace, mag_bin: str, run_dir_arg: Path) -> Path:
        path = ltv_pipeline.ltv_core_output_path(mag_bin, run_dir_arg)
        path.parent.mkdir(parents=True, exist_ok=True)
        _core_rows(tmp_path).to_parquet(path, index=False)
        return path

    monkeypatch.setattr(ltv_pipeline, "_run_core_if_needed", fake_run_core)

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        ["--stage", "cluster", "--mag-bin", "13_13.5", "--run-dir", str(run_dir), "--no-export-bundle"]
    )
    summary = ltv_pipeline.run_ltv_pipeline_cli(args)

    assert (run_dir / "results" / "LTvar13-13.5_filtered.parquet").exists()
    assert not (run_dir / "results" / "LTvar13-13.5_pipeline.parquet").exists()
    assert summary["review"] is None


def _patch_ltv_pipeline_no_network(monkeypatch, tmp_path: Path, calls: dict[str, int] | None = None) -> None:
    calls = calls if calls is not None else {}

    def fake_run_core(args: argparse.Namespace, mag_bin: str, run_dir_arg: Path) -> Path:
        path = ltv_pipeline.ltv_core_output_path(mag_bin, run_dir_arg)
        path.parent.mkdir(parents=True, exist_ok=True)
        _core_rows(tmp_path).to_parquet(path, index=False)
        return path

    def fake_run_full_pipeline(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.copy()

    def fake_ingest(_review_db: Path, df: pd.DataFrame, **_kwargs: object) -> tuple[int, int]:
        return len(df), len(df)

    def fake_external(args: argparse.Namespace, mag_bin: str, run_dir: Path, candidates: pd.DataFrame):
        calls["external"] = calls.get("external", 0) + 1
        out = candidates.copy()
        out["ztf_lc_n_det"] = 2
        path = ltv_pipeline.ltv_external_lcs_output_path(mag_bin, run_dir)
        external_dir = run_dir / "results" / "external_lcs"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        return path, external_dir, out

    def fake_multi(args: argparse.Namespace, mag_bin: str, run_dir: Path, candidates: pd.DataFrame, *, external_lc_dir: Path):
        calls["multi"] = calls.get("multi", 0) + 1
        out = candidates.copy()
        out["ltv_ms_feature_status"] = "ok"
        out["ltv_ms_ztf_n_points"] = 2
        path = ltv_pipeline.ltv_multi_survey_output_path(mag_bin, run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        return path, out

    monkeypatch.setattr(ltv_pipeline, "_run_core_if_needed", fake_run_core)
    monkeypatch.setattr(ltv_pipeline, "run_full_pipeline", fake_run_full_pipeline)
    monkeypatch.setattr(ltv_pipeline, "_write_ltv_external_lcs", fake_external)
    monkeypatch.setattr(ltv_pipeline, "_write_ltv_multi_survey_features", fake_multi)
    monkeypatch.setattr("malca.ltv.review.ingest_ltv_results", fake_ingest)


def test_ltv_stage_full_extended_runs_extended_products(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}
    _patch_ltv_pipeline_no_network(monkeypatch, tmp_path, calls)
    run_dir = tmp_path / "runs" / "ltv_full_extended"

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        [
            "--stage", "full-extended",
            "--mag-bin", "13_13.5",
            "--run-dir", str(run_dir),
            "--no-export-bundle",
            "--no-review-sync",
            "--skip-stats",
        ]
    )
    summary = ltv_pipeline.run_ltv_pipeline_cli(args)
    enriched = pd.read_parquet(run_dir / "results" / "LTvar13-13.5_pipeline.parquet")

    assert calls == {"external": 1, "multi": 1}
    assert enriched.loc[0, "ztf_lc_n_det"] == 2
    assert enriched.loc[0, "ltv_ms_feature_status"] == "ok"
    assert summary["per_bin"]["13_13.5"]["external_lcs_rows"] == 1
    assert summary["per_bin"]["13_13.5"]["ltv_multi_survey_rows"] == 1


def test_ltv_stage_full_extended_opt_in_and_out(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}
    _patch_ltv_pipeline_no_network(monkeypatch, tmp_path, calls)

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        [
            "--stage", "full",
            "--mag-bin", "13_13.5",
            "--run-dir", str(tmp_path / "runs" / "full_default"),
            "--no-export-bundle",
            "--no-review-sync",
            "--skip-stats",
        ]
    )
    ltv_pipeline.run_ltv_pipeline_cli(args)
    assert calls == {}

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        [
            "--stage", "full",
            "--run-external-lcs",
            "--run-multi-survey-features",
            "--mag-bin", "13_13.5",
            "--run-dir", str(tmp_path / "runs" / "full_opt_in"),
            "--no-export-bundle",
            "--no-review-sync",
            "--skip-stats",
        ]
    )
    ltv_pipeline.run_ltv_pipeline_cli(args)
    assert calls == {"external": 1, "multi": 1}

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        [
            "--stage", "full-extended",
            "--no-external-lcs",
            "--no-multi-survey-features",
            "--mag-bin", "13_13.5",
            "--run-dir", str(tmp_path / "runs" / "extended_opt_out"),
            "--no-export-bundle",
            "--no-review-sync",
            "--skip-stats",
        ]
    )
    ltv_pipeline.run_ltv_pipeline_cli(args)
    assert calls == {"external": 1, "multi": 1}


def test_ltv_stage_cluster_skips_extended_flags(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}
    _patch_ltv_pipeline_no_network(monkeypatch, tmp_path, calls)

    args = ltv_pipeline.add_ltv_pipeline_args(argparse.ArgumentParser()).parse_args(
        [
            "--stage", "cluster",
            "--run-external-lcs",
            "--run-multi-survey-features",
            "--mag-bin", "13_13.5",
            "--run-dir", str(tmp_path / "runs" / "cluster_extended"),
            "--no-export-bundle",
        ]
    )
    ltv_pipeline.run_ltv_pipeline_cli(args)
    assert calls == {}


def test_ltv_full_bundle_includes_candidate_lightcurves(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv_bundle"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    lc_path = tmp_path / "123.dat2"
    lc_path.write_text("hjd,mag\n1,13.2\n", encoding="ascii")
    pd.DataFrame(
        {
            "asas_sn_id": ["123"],
            "candidate_id": ["ltv_123"],
            "timescale": ["ltv"],
            "lc_path": [str(lc_path)],
            "ra": [1.0],
            "dec": [2.0],
            "failed_any": [False],
        }
    ).to_parquet(results_dir / "LTvar13-13.5_pipeline.parquet", index=False)
    pd.DataFrame({"candidate_id": ["ltv_123"], "ztf_lc_n_det": [2]}).to_parquet(
        results_dir / "LTvar13-13.5_external_lcs.parquet",
        index=False,
    )
    pd.DataFrame({"candidate_id": ["ltv_123"], "ltv_ms_feature_status": ["ok"]}).to_parquet(
        results_dir / "LTvar13-13.5_ltv_multi_survey.parquet",
        index=False,
    )
    external_dir = results_dir / "external_lcs"
    external_dir.mkdir()
    pd.DataFrame({"jd": [2450000.0], "mag": [13.0]}).to_parquet(
        external_dir / "ztf_lc_ltv_123.parquet",
        index=False,
    )

    bundle_path = tmp_path / "ltv_bundle.zip"
    names = ltv_pipeline._export_ltv_run_bundle(run_dir, bundle_path, full_bundle=True)

    assert "bundle_assets/lightcurves/123.dat2" in names
    assert "results/LTvar13-13.5_external_lcs.parquet" in names
    assert "results/LTvar13-13.5_ltv_multi_survey.parquet" in names
    assert "results/external_lcs/ztf_lc_ltv_123.parquet" in names
    with zipfile.ZipFile(bundle_path) as zf:
        assert "bundle_assets/lightcurves/123.dat2" in zf.namelist()
        assert "results/external_lcs/ztf_lc_ltv_123.parquet" in zf.namelist()


def test_ltv_overwrite_clears_core_chunks_and_checkpoint(tmp_path: Path) -> None:
    output_path = tmp_path / "results" / "LTvar13-13.5.parquet"
    output_path.mkdir(parents=True)
    chunk = output_path / "chunk_000000.parquet"
    chunk.write_text("stale", encoding="ascii")
    checkpoint = output_path.with_name("LTvar13-13.5_PROCESSED.txt")
    checkpoint.write_text("old.dat2\n", encoding="ascii")

    ltv_pipeline._clear_ltv_core_outputs(output_path)

    assert not chunk.exists()
    assert not checkpoint.exists()


def test_ltv_status_discovers_run_style_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("output") / "runs" / "ltv_march18"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame({"asas_sn_id": ["123"], "Slope": [0.4]}).to_parquet(
        results_dir / "LTvar13-13.5_pipeline.parquet",
        index=False,
    )
    _write_review_db(run_dir / "review" / "review.db", candidate_rows=2, review_rows=1)

    report = ltv_status()

    assert report["output_dir"] == str(results_dir)
    assert report["review_db"]["path"] == str(run_dir / "review" / "review.db")
    assert report["review_db"]["candidate_rows"] == 2
    assert report["review_db"]["review_rows"] == 1
    assert [record["name"] for record in report["bins"]] == ["LTvar13-13.5_pipeline.parquet"]


def test_ltv_status_accepts_explicit_legacy_paths(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "output" / "ltv" / "ltv"
    legacy_dir.mkdir(parents=True)
    legacy_db = legacy_dir / "ltv_candidates.db"
    pd.DataFrame({"asas_sn_id": ["123"], "Slope": [0.4]}).to_parquet(
        legacy_dir / "LTvar13-13.5_pipeline.parquet",
        index=False,
    )
    _write_review_db(legacy_db, candidate_rows=3, review_rows=2)

    report = ltv_status(legacy_dir)

    assert report["output_dir"] == str(legacy_dir)
    assert report["review_db"]["path"] == str(legacy_db)
    assert report["review_db"]["candidate_rows"] == 3
    assert report["review_db"]["review_rows"] == 2
