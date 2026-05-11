from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from malca.review.taxonomy import migrate_legacy_review_db
from malca.review.store import db_connect, get_review, save_review, upsert_candidates_frame


def test_save_review_round_trips_taxonomy_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(conn, pd.DataFrame([{"candidate_id": "TAX1"}]))
        save_review(
            conn,
            candidate_id="TAX1",
            interest_score=4,
            review_pass=1,
            notes="taxonomy",
            workflow_status="reviewed",
            disposition="keep",
            morphology_primary="dimming_event",
            morphology_secondary="big_dipper",
            physical_family="young_stellar_object_or_pms",
            physical_subclass="yso_dipper",
            classification_confidence="likely",
            priority_tags=["priority_dipper", "priority_followup"],
            evidence_flags=["stable_baseline"],
            model_tags=["bic_prefers_dip"],
            reviewer="tester",
        )
        review = get_review(conn, "TAX1")

    assert review["workflow_status"] == "reviewed"
    assert review["event_class"] == "dipper"
    assert review["morphology_primary"] == "dimming_event"
    assert review["morphology_secondary"] == "big_dipper"
    assert review["physical_family"] == "young_stellar_object_or_pms"
    assert review["physical_subclass"] == "yso_dipper"
    assert review["priority_tags"] == ["priority_dipper", "priority_followup"]
    assert review["evidence_flags"] == ["stable_baseline"]
    assert review["model_tags"] == ["bic_prefers_dip"]


def test_migrate_legacy_review_db_preserves_old_review_payload(tmp_path: Path) -> None:
    old_db = tmp_path / "legacy.db"
    new_db = tmp_path / "taxonomy.db"
    with sqlite3.connect(old_db) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, source_path TEXT, payload_json TEXT NOT NULL, imported_at TEXT NOT NULL)")
        conn.execute("INSERT INTO candidates VALUES (?, ?, ?, ?)", ("C1", "/run", json.dumps({"candidate_id": "C1"}), "2026-03-10T00:00:00+00:00"))
        conn.execute(
            """
            CREATE TABLE reviews (
                candidate_id TEXT PRIMARY KEY,
                interest_score INTEGER,
                event_class TEXT,
                review_pass INTEGER,
                notes TEXT,
                status TEXT,
                reviewer TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("C1", 4, "dipper", 2, "legacy note", "reviewed", "legacy", "2026-03-12T00:00:00+00:00"),
        )
        conn.execute("CREATE TABLE review_history (candidate_id TEXT, event_type TEXT, payload_json TEXT, reviewer TEXT, created_at TEXT)")
        conn.execute("INSERT INTO review_history VALUES (?, ?, ?, ?, ?)", ("C1", "save", "{}", "legacy", "2026-03-12T00:00:00+00:00"))
        conn.execute("CREATE TABLE app_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        conn.execute("INSERT INTO app_state VALUES (?, ?, ?)", ("k", "v", "2026-03-12T00:00:00+00:00"))
        conn.commit()

    result = migrate_legacy_review_db(old_db, new_db)

    assert result["candidates"] == 1
    assert result["reviews"] == 1
    with db_connect(new_db) as conn:
        review = get_review(conn, "C1")

    assert review["workflow_status"] == "reviewed"
    assert review["morphology_primary"] == "dimming_event"
    assert review["priority_tags"] == ["priority_dipper"]
    legacy = json.loads(review["legacy_review_json"])
    assert legacy["event_class"] == "dipper"
    assert legacy["notes"] == "legacy note"
