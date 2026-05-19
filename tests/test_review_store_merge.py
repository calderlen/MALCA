from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from malca.review.eda_data import load_review_db
from malca.review.store import (
    db_connect,
    get_candidate_payload,
    import_candidates,
    merge_candidate_results,
    merge_review_databases,
    merge_vetting_results,
    upsert_candidates_frame,
)


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


def test_import_candidates_preserves_extended_enrichment_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    df = pd.DataFrame([
        {
            "candidate_id": "C1",
            "asas_sn_id": "C1",
            "failed_any": False,
            "simbad_main_id": "Example Star",
            "ztf_lc_n_det": 12,
            "neowise_n_epochs": 4,
            "ms_feature_status": "ok",
            "ms_event_type": "dip",
        }
    ])

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            df,
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        payload = get_candidate_payload(conn, "C1")

    assert payload["simbad_main_id"] == "Example Star"
    assert payload["ztf_lc_n_det"] == 12
    assert payload["neowise_n_epochs"] == 4
    assert payload["ms_feature_status"] == "ok"
    assert payload["ms_event_type"] == "dip"


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


def test_merge_vetting_results_updates_gaia_variable_sql_column(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "GAIA-VAR",
                        "asas_sn_id": "GAIA-VAR",
                        "gaia_var_flag": False,
                    }
                ]
            ),
        )
        updated = merge_vetting_results(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "GAIA-VAR",
                        "gaia_var_flag": True,
                        "gaia_var_class": "",
                    }
                ]
            ),
        )
        payload = get_candidate_payload(conn, "GAIA-VAR")

    assert updated == 1
    assert payload["gaia_var_flag"] is True


def test_upsert_candidates_frame_derives_known_from_gaia_class(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "GAIA-CLASSIFIED",
                        "asas_sn_id": "GAIA-CLASSIFIED",
                        "gaia_var_flag": True,
                        "gaia_var_class": "LPV",
                        "vetting_likely_known": False,
                    }
                ]
            ),
        )
        payload = get_candidate_payload(conn, "GAIA-CLASSIFIED")

    assert payload["vetting_likely_known"] is True
    assert payload["gaia_var_class"] == "LPV"


def test_merge_candidate_results_updates_candidate_fields_without_touching_reviews(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    _seed_review_db(
        review_db,
        [
            {
                "candidate_id": "ltv_123",
                "source_path": "/src",
                "asas_sn_id": "123",
                "periodicity_score": 0.1,
                "failed_periodicity": 0,
            }
        ],
        [
            {
                "candidate_id": "ltv_123",
                "interest_score": 4,
                "event_class": "dipper",
                "review_pass": 2,
                "notes": "keep me",
                "status": "reviewed",
                "reviewer": "src",
                "updated_at": "2026-03-12T12:00:00+00:00",
            }
        ],
    )

    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "ltv_123",
                "periodicity_score": 3.7,
                "failed_periodicity": 1,
                "periodicity_is_significant": True,
                "periodic_flag": True,
                "pdm_snr": 6.2,
                "ce_snr": 7.4,
            }
        ]
    )

    with db_connect(review_db) as conn:
        updated = merge_candidate_results(conn, candidate_df)
        merged_row = conn.execute(
            "SELECT periodicity_score, failed_periodicity, periodic_flag, pdm_snr, ce_snr FROM candidates WHERE candidate_id='ltv_123'"
        ).fetchone()
        review_row = conn.execute(
            "SELECT notes, review_pass FROM reviews WHERE candidate_id='ltv_123'"
        ).fetchone()

    assert updated == 1
    assert float(merged_row[0]) == 3.7
    assert int(merged_row[1]) == 1
    assert int(merged_row[2]) == 1
    assert float(merged_row[3]) == 6.2
    assert float(merged_row[4]) == 7.4
    assert review_row == ("keep me", 2)
