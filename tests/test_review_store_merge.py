from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from malca.review.explore_data import load_review_db
from malca.review.store import db_connect, export_review_subset_bundle, merge_review_databases, upsert_candidates_frame


def _seed_review_db(path: Path, candidates: list[dict[str, object]], reviews: list[dict[str, object]]) -> None:
    with db_connect(path) as conn:
        upsert_candidates_frame(conn, pd.DataFrame(candidates))
        for review in reviews:
            conn.execute(
                """
                INSERT INTO reviews (candidate_id, interest_score, event_class, review_pass, notes, status, reviewer, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review["candidate_id"],
                    review.get("interest_score"),
                    review.get("event_class", "unclassified"),
                    review.get("review_pass", 1),
                    review.get("notes", ""),
                    review.get("status", "unreviewed"),
                    review.get("reviewer", ""),
                    review["updated_at"],
                ),
            )
            conn.execute(
                "INSERT INTO review_history (candidate_id, event_type, payload_json, reviewer, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    review["candidate_id"],
                    "save",
                    json.dumps({"notes": review.get("notes", "")}),
                    review.get("reviewer", ""),
                    review["updated_at"],
                ),
            )
        conn.commit()


def test_merge_review_databases_prefers_newer_updated_at(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    target_db = tmp_path / "target.db"

    _seed_review_db(
        source_db,
        [
            {"candidate_id": "C1", "source_path": "/src", "asas_sn_id": "C1", "dipper_score": 12.0},
            {"candidate_id": "C2", "source_path": "/src", "asas_sn_id": "C2", "dipper_score": 8.0},
        ],
        [
            {"candidate_id": "C1", "interest_score": 4, "event_class": "dipper", "review_pass": 2, "notes": "newer", "status": "reviewed", "reviewer": "src", "updated_at": "2026-03-12T12:00:00+00:00"},
            {"candidate_id": "C2", "interest_score": 2, "event_class": "flare", "review_pass": 1, "notes": "inserted", "status": "reviewed", "reviewer": "src", "updated_at": "2026-03-12T11:00:00+00:00"},
        ],
    )
    _seed_review_db(
        target_db,
        [{"candidate_id": "C1", "source_path": "/target", "asas_sn_id": "C1", "dipper_score": 4.0}],
        [{"candidate_id": "C1", "interest_score": 1, "event_class": "other", "review_pass": 1, "notes": "older", "status": "reviewed", "reviewer": "target", "updated_at": "2026-03-11T10:00:00+00:00"}],
    )

    result = merge_review_databases(source_db, target_db)
    merged = load_review_db(target_db)

    row_c1 = merged.loc[merged["candidate_id"] == "C1"].iloc[0]
    row_c2 = merged.loc[merged["candidate_id"] == "C2"].iloc[0]

    assert result["reviews_updated"] == 1
    assert result["reviews_inserted"] == 1
    assert result["candidates_inserted"] == 1
    assert row_c1["notes"] == "newer"
    assert row_c2["event_class"] == "flare"


def test_export_review_subset_bundle_writes_review_ready_bundle(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    _seed_review_db(
        source_db,
        [{"candidate_id": "C3", "source_path": "/src", "asas_sn_id": "C3", "dipper_score": 6.0, "lc_path": "/tmp/C3.dat2"}],
        [{"candidate_id": "C3", "interest_score": 3, "event_class": "dipper", "review_pass": 1, "notes": "kept", "status": "reviewed", "reviewer": "src", "updated_at": "2026-03-12T09:00:00+00:00"}],
    )

    export_df = load_review_db(source_db)
    export_df["source_file"] = str(source_db)
    bundle_dir = tmp_path / "bundle"

    result = export_review_subset_bundle(
        bundle_dir,
        export_df,
        selection_meta={"plot": {"x_metric": "period_n_sources", "y_metric": "dipper_score"}, "candidate_count": 1},
    )

    assert (bundle_dir / "review.db").exists()
    assert (bundle_dir / "selection_candidates.parquet").exists()
    assert (bundle_dir / "selection_meta.json").exists()
    bundled = load_review_db(bundle_dir / "review.db")
    assert bundled.loc[0, "candidate_id"] == "C3"
    assert bundled.loc[0, "notes"] == "kept"

    with sqlite3.connect(bundle_dir / "review.db") as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key='explorer_selection_meta'").fetchone()
    assert row is not None
    assert json.loads(row[0])["candidate_count"] == 1
    assert Path(result["review_db"]) == bundle_dir / "review.db"
