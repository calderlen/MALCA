from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from malca.review.taxonomy import legacy_review_to_taxonomy, migrate_legacy_review_db
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
            physical_primary="young_stellar_object_or_pms",
            physical_secondary="yso_dipper",
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
    assert review["physical_primary"] == "young_stellar_object_or_pms"
    assert review["physical_secondary"] == "yso_dipper"
    assert review["priority_tags"] == ["priority_dipper", "priority_followup"]
    assert review["evidence_flags"] == ["stable_baseline"]
    assert review["model_tags"] == ["bic_prefers_dip"]


def test_init_db_renames_legacy_physical_taxonomy_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_physical.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE reviews (
                candidate_id TEXT PRIMARY KEY,
                interest_score INTEGER,
                event_class TEXT DEFAULT 'unclassified',
                review_pass INTEGER,
                notes TEXT,
                status TEXT,
                reviewer TEXT,
                workflow_status TEXT,
                disposition TEXT,
                morphology_primary TEXT,
                morphology_secondary TEXT,
                morphology_polarity TEXT,
                morphology_recurrence TEXT,
                baseline_behavior TEXT,
                physical_family TEXT,
                physical_subclass TEXT,
                classification_confidence TEXT,
                priority_tags_json TEXT,
                evidence_flags_json TEXT,
                model_tags_json TEXT,
                duplicate_of TEXT,
                known_object_id TEXT,
                known_object_source TEXT,
                taxonomy_version INTEGER,
                legacy_review_json TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO reviews (
                candidate_id, interest_score, event_class, review_pass, notes, status,
                reviewer, workflow_status, morphology_primary, physical_family,
                physical_subclass, priority_tags_json, taxonomy_version,
                legacy_review_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "C1",
                4,
                "dipper",
                1,
                "",
                "reviewed",
                "tester",
                "reviewed",
                "dimming_event",
                "young_stellar_object_or_pms",
                "yso_dipper",
                "[]",
                1,
                "{}",
                "2026-03-12T00:00:00+00:00",
            ),
        )
        conn.commit()

    with db_connect(db_path) as conn:
        review = get_review(conn, "C1")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()}

    assert review["physical_primary"] == "young_stellar_object_or_pms"
    assert review["physical_secondary"] == "yso_dipper"
    assert "physical_primary" in columns
    assert "physical_secondary" in columns
    assert "physical_family" not in columns
    assert "physical_subclass" not in columns


@pytest.mark.parametrize(
    ("event_class", "expected"),
    [
        (
            "dipper",
            {
                "morphology_primary": "dimming_event",
                "morphology_secondary": None,
                "physical_primary": None,
                "physical_secondary": None,
                "priority_tags": ["priority_dipper"],
            },
        ),
        (
            "microlensing",
            {
                "morphology_primary": "brightening_event",
                "morphology_secondary": "possible_microlensing_event",
                "physical_primary": "microlensing",
                "physical_secondary": "generic_microlensing_candidate",
                "priority_tags": [],
            },
        ),
        (
            "flare",
            {
                "morphology_primary": "brightening_event",
                "morphology_secondary": "possible_flare",
                "physical_primary": "flare_star_or_magnetically_active_star",
                "physical_secondary": None,
                "priority_tags": [],
            },
        ),
        (
            "ltv",
            {
                "morphology_primary": "long_term_trend",
                "morphology_secondary": None,
                "physical_primary": None,
                "physical_secondary": None,
                "priority_tags": [],
            },
        ),
        (
            "instrumental",
            {
                "morphology_primary": "artifact_or_bad_photometry",
                "morphology_secondary": None,
                "physical_primary": "false_positive_or_contaminant",
                "physical_secondary": None,
                "priority_tags": [],
            },
        ),
        (
            "unknown_interesting",
            {
                "morphology_primary": "unclear",
                "morphology_secondary": None,
                "physical_primary": "unknown",
                "physical_secondary": None,
                "priority_tags": [],
            },
        ),
        (
            "other",
            {
                "morphology_primary": None,
                "morphology_secondary": None,
                "physical_primary": None,
                "physical_secondary": None,
                "priority_tags": [],
            },
        ),
    ],
)
def test_legacy_review_to_taxonomy_label_only_mapping(event_class: str, expected: dict[str, object]) -> None:
    mapped = legacy_review_to_taxonomy(
        {
            "candidate_id": "C1",
            "event_class": event_class,
            "status": "reviewed",
            "notes": "",
        }
    )

    assert mapped["workflow_status"] == "reviewed"
    assert mapped["disposition"] == "keep"
    for key, value in expected.items():
        assert mapped[key] == value

    legacy = json.loads(mapped["legacy_review_json"])
    assert legacy["event_class"] == event_class


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
    assert result["reviews_mapped"] == 1
    assert result["reviews_unmapped"] == 0
    assert result["mapped_review_label_counts"] == {"dipper": 1}
    assert result["unmapped_review_label_counts"] == {}
    with db_connect(new_db) as conn:
        review = get_review(conn, "C1")
        history_count = conn.execute("SELECT COUNT(*) FROM review_history").fetchone()[0]
        app_state_value = conn.execute("SELECT value FROM app_state WHERE key='k'").fetchone()[0]

    assert review["status"] == "reviewed"
    assert review["workflow_status"] == "reviewed"
    assert review["review_pass"] == 2
    assert review["reviewer"] == "legacy"
    assert review["updated_at"] == "2026-03-12T00:00:00+00:00"
    assert review["morphology_primary"] == "dimming_event"
    assert review["priority_tags"] == ["priority_dipper"]
    assert history_count == 1
    assert app_state_value == "v"
    legacy = json.loads(review["legacy_review_json"])
    assert legacy["event_class"] == "dipper"
    assert legacy["notes"] == "legacy note"


def test_migrate_legacy_review_db_reports_unmapped_labels(tmp_path: Path) -> None:
    old_db = tmp_path / "legacy.db"
    new_db = tmp_path / "taxonomy.db"
    with sqlite3.connect(old_db) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, source_path TEXT, payload_json TEXT NOT NULL, imported_at TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO candidates VALUES (?, ?, ?, ?)",
            [
                ("C1", "/run", json.dumps({"candidate_id": "C1"}), "2026-03-10T00:00:00+00:00"),
                ("C2", "/run", json.dumps({"candidate_id": "C2"}), "2026-03-10T00:00:00+00:00"),
            ],
        )
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
        conn.executemany(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("C1", 4, "microlensing", 1, "", "reviewed", "legacy", "2026-03-12T00:00:00+00:00"),
                ("C2", 2, "other", 1, "", "reviewed", "legacy", "2026-03-12T00:00:00+00:00"),
            ],
        )
        conn.commit()

    result = migrate_legacy_review_db(old_db, new_db)

    assert result["reviews"] == 2
    assert result["reviews_mapped"] == 1
    assert result["reviews_unmapped"] == 1
    assert result["mapped_review_label_counts"] == {"microlensing": 1}
    assert result["unmapped_review_label_counts"] == {"other": 1}
    with db_connect(new_db) as conn:
        microlensing = get_review(conn, "C1")
        other = get_review(conn, "C2")

    assert microlensing["morphology_secondary"] == "possible_microlensing_event"
    assert microlensing["physical_secondary"] == "generic_microlensing_candidate"
    assert other["morphology_primary"] is None
    assert other["physical_primary"] is None
