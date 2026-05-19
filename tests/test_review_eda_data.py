from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from malca.review.eda_data import (
    add_eda_columns,
    available_metric_columns,
    infer_plot_dir_from_source,
    infer_source_kind,
    load_candidate_source,
    load_review_db,
)


def _write_review_db(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)")
        for row in rows:
            conn.execute(
                "INSERT INTO candidates VALUES (?, ?, ?)",
                (
                    row["candidate_id"],
                    row.get("source_path", ""),
                    json.dumps(row),
                ),
            )
        conn.commit()


def test_add_eda_columns_builds_proxy_fields(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_a" / "review" / "review.db"
    _write_review_db(
        db,
        [
            {
                "candidate_id": "C1",
                "asas_sn_id": "C1",
                "catalog_match": True,
                "period_consensus_agree": True,
                "period_n_sources": 3,
                "dip_run_count": 4,
                "dipper_n_valid_dips": 12,
                "vetting_likely_known": True,
                "dipper_score": 10,
            },
            {
                "candidate_id": "C2",
                "asas_sn_id": "C2",
                "catalog_match": False,
                "period_consensus_agree": False,
                "period_n_sources": 0,
                "dip_run_count": 1,
                "dip_is_single_event": True,
                "dipper_n_valid_dips": 8,
                "vetting_likely_known": False,
                "dipper_score": 15,
            },
        ],
    )

    frame = add_eda_columns(load_review_db(db))

    row1 = frame.loc[frame["candidate_id"] == "C1"].iloc[0]
    row2 = frame.loc[frame["candidate_id"] == "C2"].iloc[0]
    assert bool(row1["known_periodic_catalog"]) is True
    assert bool(row1["strong_catalog_period"]) is True
    assert bool(row2["proxy_oneoff_dipper"]) is True
    assert frame.attrs["default_target_col"] == "proxy_oneoff_dipper"


def test_add_eda_columns_fills_catalog_type_aliases(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_alias" / "review" / "review.db"
    _write_review_db(
        db,
        [
            {
                "candidate_id": "A1",
                "period_asassn_var_class": "EA",
                "period_ztf_periodic_class": "EW",
                "asassn_var_type": "",
                "ztf_var_type": "",
            },
            {
                "candidate_id": "A2",
                "period_asassn_var_class": "",
                "period_ztf_periodic_class": "",
                "asassn_var_type": "DSCT",
                "ztf_var_type": "RR",
            },
        ],
    )

    frame = add_eda_columns(load_review_db(db))

    row1 = frame.loc[frame["candidate_id"] == "A1"].iloc[0]
    row2 = frame.loc[frame["candidate_id"] == "A2"].iloc[0]

    assert row1["asassn_var_type"] == "EA"
    assert row1["ztf_var_type"] == "EW"
    assert row2["asassn_var_type"] == "DSCT"
    assert row2["ztf_var_type"] == "RR"


def test_load_review_db_merges_review_columns(tmp_path: Path) -> None:
    db = tmp_path / "review.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)")
        conn.execute(
            """
            CREATE TABLE reviews (
                candidate_id TEXT,
                interest_score INTEGER,
                event_class TEXT,
                review_pass INTEGER,
                notes TEXT,
                status TEXT,
                workflow_status TEXT,
                morphology_primary TEXT,
                physical_primary TEXT,
                classification_confidence TEXT,
                known_object_id TEXT,
                taxonomy_version INTEGER,
                reviewer TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            ("C4", "/tmp/run", json.dumps({"candidate_id": "C4", "dipper_score": 8.0})),
        )
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "C4",
                3,
                "dipper",
                2,
                "note",
                "reviewed",
                "reviewed",
                "dimming_event",
                "young_stellar_object_or_pms",
                "secure",
                "VSX J0004",
                1,
                "tester",
                "2026-03-10T00:00:00Z",
            ),
        )
        conn.commit()

    df = load_review_db(db)

    assert df.loc[0, "interest_score"] == 3
    assert df.loc[0, "event_class"] == "dipper"
    assert df.loc[0, "status"] == "reviewed"
    assert df.loc[0, "morphology_primary"] == "dimming_event"
    assert df.loc[0, "physical_primary"] == "young_stellar_object_or_pms"
    assert df.loc[0, "classification_confidence"] == "secure"
    assert df.loc[0, "known_object_id"] == "VSX J0004"


def test_infer_source_kind() -> None:
    assert infer_source_kind("/tmp/review.db") == "db"
    assert infer_source_kind("/tmp/candidates.parquet") == "parquet"
    assert infer_source_kind("/tmp/candidates.csv") == "csv"
    with pytest.raises(ValueError):
        infer_source_kind("/tmp/candidates.txt")


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


def test_load_candidate_source_reads_review_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    _write_review_db(
        db_path,
        [{"candidate_id": "CAND-2", "asas_sn_id": "ASAS-2", "gaia_id": "1234"}],
    )

    frame = load_candidate_source(db_path)

    assert frame.loc[0, "candidate_id"] == "CAND-2"
    assert frame.loc[0, "asas_sn_id"] == "ASAS-2"
    assert frame.loc[0, "gaia_id"] == "1234"


def test_available_metric_columns_prefers_review_eda_fields() -> None:
    frame = pd.DataFrame(
        {
            "period_n_sources": [1],
            "dip_run_count": [2],
            "custom_numeric": [3.0],
            "candidate_key": ["C1"],
        }
    )

    metrics = available_metric_columns(frame)

    assert metrics[:2] == ["period_n_sources", "dip_run_count"]
    assert "custom_numeric" in metrics
    assert "candidate_key" not in metrics
