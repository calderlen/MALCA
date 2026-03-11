from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import plotly.graph_objects as go

from malca.review.mini_viewer import (
    clickable_figure_html,
    infer_plot_dir_from_source,
    infer_source_kind,
    load_review_db,
    load_source_data,
)


def test_infer_source_kind() -> None:
    assert infer_source_kind("/tmp/review.db") == "db"
    assert infer_source_kind("/tmp/candidates.parquet") == "parquet"
    assert infer_source_kind("/tmp/candidates.csv") == "csv"


def test_infer_plot_dir_from_source_for_run_local_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    plot_dir = run_dir / "plots"
    review_dir = run_dir / "review"
    results_dir = run_dir / "results"
    plot_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    db_path = review_dir / "review.db"
    db_path.touch()
    parquet_path = results_dir / "lc_events_vetted.parquet"
    parquet_path.touch()

    assert infer_plot_dir_from_source(db_path) == plot_dir.resolve()
    assert infer_plot_dir_from_source(parquet_path) == plot_dir.resolve()


def test_load_review_db_flattens_payload_json(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)"
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            (
                "CAND-1",
                "/tmp/run",
                json.dumps({"candidate_id": "CAND-1", "asas_sn_id": "A-1", "dipper_score": 12.5}),
            ),
        )
        conn.commit()

    df = load_review_db(db_path)

    assert len(df) == 1
    assert df.loc[0, "candidate_id"] == "CAND-1"
    assert df.loc[0, "asas_sn_id"] == "A-1"
    assert df.loc[0, "dipper_score"] == 12.5


def test_load_source_data_builds_id_lookup(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)"
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            (
                "CAND-2",
                "/tmp/run",
                json.dumps({"candidate_id": "CAND-2", "asas_sn_id": "ASAS-2", "gaia_id": "1234"}),
            ),
        )
        conn.commit()

    source = load_source_data(db_path)

    assert source.lookup["CAND-2"] == 0
    assert source.lookup["ASAS-2"] == 0
    assert source.lookup["1234"] == 0
    assert source.default_candidate_id == "CAND-2"


def test_clickable_figure_html_embeds_viewer_link() -> None:
    fig = go.Figure(data=[go.Scatter(x=[1], y=[2], customdata=[["CAND-3"]])])
    html = clickable_figure_html(fig, viewer_url="http://127.0.0.1:8061")

    assert "plotly_click" in html
    assert "candidate_id=" in html
    assert "malca-mini-viewer" in html
