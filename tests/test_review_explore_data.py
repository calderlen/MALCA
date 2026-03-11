from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from malca.review.explore_data import (
    add_eda_columns,
    find_candidate_key,
    get_candidate_record_by_key,
    load_review_db,
    load_combined_source_data,
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


def test_load_combined_source_data_adds_candidate_keys(tmp_path: Path) -> None:
    db1 = tmp_path / "output" / "runs" / "run_a" / "review" / "review.db"
    db2 = tmp_path / "output" / "runs" / "run_b" / "review" / "review.db"
    _write_review_db(db1, [{"candidate_id": "A1", "asas_sn_id": "A1", "dipper_score": 12.0}])
    _write_review_db(db2, [{"candidate_id": "A1", "asas_sn_id": "B1", "dipper_score": 9.0}])

    combined = load_combined_source_data(sources=[db1, db2])

    assert len(combined.df) == 2
    assert combined.df["candidate_key"].nunique() == 2
    assert find_candidate_key(combined, "A1") is not None


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

    combined = load_combined_source_data(sources=[db])
    frame = add_eda_columns(combined.df)

    row1 = frame.loc[frame["candidate_id"] == "C1"].iloc[0]
    row2 = frame.loc[frame["candidate_id"] == "C2"].iloc[0]
    assert bool(row1["known_periodic_catalog"]) is True
    assert bool(row1["strong_catalog_period"]) is True
    assert bool(row2["proxy_oneoff_dipper"]) is True
    assert frame.attrs["default_target_col"] == "proxy_oneoff_dipper"


def test_get_candidate_record_by_key_round_trips(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_a" / "review" / "review.db"
    _write_review_db(db, [{"candidate_id": "C3", "asas_sn_id": "C3", "dipper_score": 4.0}])
    combined = load_combined_source_data(sources=[db])
    key = str(combined.df.iloc[0]["candidate_key"])

    record = get_candidate_record_by_key(combined, key)

    assert record is not None
    assert record["candidate_id"] == "C3"


def test_load_review_db_merges_review_columns(tmp_path: Path) -> None:
    db = tmp_path / "review.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)")
        conn.execute(
            "CREATE TABLE reviews (candidate_id TEXT, interest_score INTEGER, event_class TEXT, review_pass INTEGER, notes TEXT, status TEXT, reviewer TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            ("C4", "/tmp/run", json.dumps({"candidate_id": "C4", "dipper_score": 8.0})),
        )
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("C4", 3, "dipper", 2, "note", "reviewed", "tester", "2026-03-10T00:00:00Z"),
        )
        conn.commit()

    df = load_review_db(db)

    assert df.loc[0, "interest_score"] == 3
    assert df.loc[0, "event_class"] == "dipper"
    assert df.loc[0, "status"] == "reviewed"
