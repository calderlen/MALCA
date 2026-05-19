from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

import malca.__main__ as cli
from malca.audit import ltv_status
from malca.ltv import core as ltv_core
from malca.ltv import pipeline as ltv_pipeline
from malca.ltv import review as ltv_review


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


def test_ltv_core_run_dir_sets_default_output(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["malca", "--mag-bin", "13_13.5", "--run-dir", str(run_dir)],
    )

    configs, run_all = ltv_core.parse_args()

    assert run_all is False
    assert configs[0].output == run_dir / "results" / "LTvar13-13.5.parquet"


def test_ltv_build_run_dir_sets_default_input_and_output(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    input_path = results_dir / "LTvar13-13.5.parquet"
    pd.DataFrame({"asas_sn_id": ["123"], "Slope": [0.4], "max diff": [0.6]}).to_parquet(input_path, index=False)

    def fake_run_full_pipeline(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.copy()

    monkeypatch.setattr(ltv_pipeline, "run_full_pipeline", fake_run_full_pipeline)
    args = ltv_pipeline.add_pipeline_args(argparse.ArgumentParser()).parse_args(
        ["--mag-bin", "13_13.5", "--run-dir", str(run_dir)]
    )

    ltv_pipeline.run_pipeline_cli(args)

    assert Path(args.input) == input_path
    assert Path(args.output) == results_dir / "LTvar13-13.5_pipeline.parquet"
    assert Path(args.output).exists()


def test_ltv_pipeline_defaults_review_db_to_run_dir(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv"
    captured: dict[str, object] = {}
    real_import_module = cli.importlib.import_module

    class FakePipelineModule:
        @staticmethod
        def add_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            parser.add_argument("--mag-bin")
            parser.add_argument("--run-dir", default="output/runs/ltv")
            parser.add_argument("--workers", type=int, default=1)
            parser.add_argument("--verbose", action="store_true")
            return parser

        @staticmethod
        def run_pipeline_cli(args: argparse.Namespace) -> pd.DataFrame:
            captured["pipeline_review_db"] = args.review_db
            captured["pipeline_run_dir"] = args.run_dir
            return pd.DataFrame({"asas_sn_id": ["123"]})

    class FakeReviewModule:
        @staticmethod
        def ingest_ltv_results(review_db: str, df: pd.DataFrame, **kwargs: object) -> tuple[int, int]:
            captured["review_db"] = review_db
            captured["source_path"] = kwargs["source_path"]
            captured["rows"] = len(df)
            return len(df), len(df)

    def fake_import_module(name: str):
        if name == "malca.ltv.pipeline":
            return FakePipelineModule
        if name == "malca.ltv.review":
            return FakeReviewModule
        return real_import_module(name)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sys, "argv", ["malca", "ltv-pipeline", "--mag-bin", "13_13.5", "--run-dir", str(run_dir)])

    assert cli.main() == 0
    assert captured["pipeline_review_db"] == str(run_dir / "review" / "review.db")
    assert captured["review_db"] == str(run_dir / "review" / "review.db")
    assert captured["source_path"] == run_dir
    assert captured["rows"] == 1


def test_ltv_ingest_defaults_input_and_review_db_to_run_dir(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame({"asas_sn_id": ["123"], "Slope": [0.4], "max diff": [0.6]}).to_parquet(
        results_dir / "LTvar13-13.5_pipeline.parquet",
        index=False,
    )

    captured: dict[str, object] = {}

    def fake_ingest_ltv_results(review_db: str, df: pd.DataFrame, **kwargs: object) -> tuple[int, int]:
        captured["review_db"] = review_db
        captured["source_path"] = kwargs["source_path"]
        captured["rows"] = len(df)
        return len(df), len(df)

    monkeypatch.setattr(ltv_review, "ingest_ltv_results", fake_ingest_ltv_results)
    monkeypatch.setattr(
        sys,
        "argv",
        ["malca ltv-ingest", "--run-dir", str(run_dir), "--skip-characterize", "--skip-stats"],
    )

    ltv_review.main()

    assert captured["review_db"] == str(run_dir / "review" / "review.db")
    assert captured["source_path"] == run_dir
    assert captured["rows"] == 1


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
