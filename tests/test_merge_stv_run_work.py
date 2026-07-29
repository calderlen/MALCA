from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from malca.external_lc_manifest import read_external_lc_manifest, upsert_external_lc_manifest_entry
from malca.review.store import db_connect, upsert_candidates_frame


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "merge_stv_run_work.py"
SPEC = importlib.util.spec_from_file_location("merge_stv_run_work", SCRIPT_PATH)
assert SPEC is not None
merge_stv_run_work = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_stv_run_work
assert SPEC.loader is not None
SPEC.loader.exec_module(merge_stv_run_work)


def _seed_review_db(
    path: Path,
    candidates: list[dict[str, object]],
    reviews: list[dict[str, object]] | None = None,
) -> None:
    with db_connect(path) as conn:
        upsert_candidates_frame(conn, pd.DataFrame(candidates))
        for review in reviews or []:
            conn.execute(
                """
                INSERT INTO reviews (
                    candidate_id, classification_confidence, event_class, review_pass, notes,
                    status, workflow_status, reviewer, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review["candidate_id"],
                    review.get("classification_confidence", 1),
                    review.get("event_class", "unclassified"),
                    review.get("review_pass", 1),
                    review.get("notes", ""),
                    review.get("status", "reviewed"),
                    review.get("workflow_status", review.get("status", "reviewed")),
                    review.get("reviewer", "tester"),
                    review.get("updated_at", "2026-01-01T00:00:00+00:00"),
                ),
            )
            conn.execute(
                """
                INSERT INTO review_history (candidate_id, event_type, payload_json, reviewer, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    review["candidate_id"],
                    "save",
                    '{"notes": "%s"}' % review.get("notes", ""),
                    review.get("reviewer", "tester"),
                    review.get("updated_at", "2026-01-01T00:00:00+00:00"),
                ),
            )
        conn.commit()


def test_build_merge_context_maps_bare_asas_ids_to_stv_ids(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    source_db = source_run / "review" / "review.db"
    target_db = target_run / "review" / "review.db"

    _seed_review_db(
        source_db,
        [
            {"candidate_id": "100", "asas_sn_id": "100"},
            {"candidate_id": "200", "asas_sn_id": "200"},
        ],
    )
    _seed_review_db(
        target_db,
        [
            {"candidate_id": "stv_100", "asas_sn_id": "100"},
            {"candidate_id": "stv_300", "asas_sn_id": "300"},
        ],
    )

    ctx = merge_stv_run_work.build_merge_context(source_run, source_db, target_run, target_db)

    assert ctx.overlap_asas == {"100"}
    assert ctx.source_candidate_to_target == {"100": "stv_100"}
    assert ctx.source_cache_id_to_target["100"] == "stv_100"
    assert not ctx.ambiguous_overlap_asas


def test_merge_reviews_remaps_ids_and_preserves_newer_target_review(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    source_db = source_run / "review" / "review.db"
    target_db = target_run / "review" / "review.db"

    _seed_review_db(
        source_db,
        [
            {"candidate_id": "100", "asas_sn_id": "100"},
            {"candidate_id": "200", "asas_sn_id": "200"},
            {"candidate_id": "300", "asas_sn_id": "300"},
            {"candidate_id": "400", "asas_sn_id": "400"},
        ],
        [
            {"candidate_id": "100", "notes": "insert me", "updated_at": "2026-03-12T10:00:00+00:00"},
            {"candidate_id": "200", "notes": "older source", "updated_at": "2026-03-12T10:00:00+00:00"},
            {"candidate_id": "300", "notes": "replace unreviewed", "updated_at": "2026-03-12T10:00:00+00:00"},
            {"candidate_id": "400", "notes": "not overlapping", "updated_at": "2026-03-12T10:00:00+00:00"},
        ],
    )
    _seed_review_db(
        target_db,
        [
            {"candidate_id": "stv_100", "asas_sn_id": "100"},
            {"candidate_id": "stv_200", "asas_sn_id": "200"},
            {"candidate_id": "stv_300", "asas_sn_id": "300"},
        ],
        [
            {"candidate_id": "stv_200", "notes": "newer target", "updated_at": "2026-03-13T10:00:00+00:00"},
            {
                "candidate_id": "stv_300",
                "notes": "target unreviewed",
                "status": "unreviewed",
                "workflow_status": "unreviewed",
                "updated_at": "2026-03-13T10:00:00+00:00",
            },
        ],
    )

    ctx = merge_stv_run_work.build_merge_context(source_run, source_db, target_run, target_db)
    dry = merge_stv_run_work.merge_reviews(ctx, apply=False)
    applied = merge_stv_run_work.merge_reviews(ctx, apply=True)

    assert dry.reviews_inserted == 1
    assert dry.reviews_updated == 1
    assert dry.reviews_skipped == 1
    assert applied.reviews_inserted == 1
    assert applied.reviews_updated == 1
    assert applied.reviews_skipped == 1

    with sqlite3.connect(target_db) as conn:
        reviews = pd.read_sql_query("SELECT candidate_id, notes, workflow_status FROM reviews", conn)
        history_ids = {
            row[0]
            for row in conn.execute("SELECT DISTINCT candidate_id FROM review_history").fetchall()
        }

    notes = dict(zip(reviews["candidate_id"], reviews["notes"]))
    assert notes["stv_100"] == "insert me"
    assert notes["stv_200"] == "newer target"
    assert notes["stv_300"] == "replace unreviewed"
    assert "100" not in notes
    assert history_ids == {"stv_100", "stv_200", "stv_300"}


def test_merge_external_cache_remaps_files_manifest_and_status(tmp_path: Path) -> None:
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    source_db = source_run / "review" / "review.db"
    target_db = target_run / "review" / "review.db"
    source_results = source_run / "results"
    target_external = target_run / "results" / "external_lcs"

    _seed_review_db(source_db, [{"candidate_id": "100", "asas_sn_id": "100"}])
    _seed_review_db(target_db, [{"candidate_id": "stv_100", "asas_sn_id": "100"}])
    source_results.mkdir(parents=True)
    lc_path = source_results / "tess_lc_100.parquet"
    pd.DataFrame({"time": [1.0, 2.0], "flux": [10.0, 11.0], "sector": [1, 1]}).to_parquet(lc_path, index=False)
    upsert_external_lc_manifest_entry(
        source_results,
        candidate_id="100",
        source="tess",
        file_prefix="tess",
        path=lc_path,
    )
    pd.DataFrame(
        [
            {
                "module": "TESS LCs",
                "candidate_id": "100",
                "cache_key": "tess-key",
                "status": "fetched",
                "updated_unix": 1.0,
                "tess_n_sectors": 1,
                "tess_total_points": 2,
                "tess_flux_range": 1.0,
            }
        ]
    ).to_parquet(source_results / "_external_lc_status.parquet", index=False)

    ctx = merge_stv_run_work.build_merge_context(source_run, source_db, target_run, target_db)
    dry = merge_stv_run_work.merge_external_cache(ctx, apply=False)
    applied = merge_stv_run_work.merge_external_cache(ctx, apply=True)

    assert dry.files_to_copy == 1
    assert dry.status_rows_to_insert == 1
    assert applied.files_copied == 1
    assert applied.status_rows_inserted == 1

    dest = target_external / "tess_lc_stv_100.parquet"
    assert dest.exists()
    manifest = read_external_lc_manifest(target_external)
    assert manifest[["candidate_id", "source", "path_relative"]].to_dict("records") == [
        {"candidate_id": "stv_100", "source": "tess", "path_relative": "tess_lc_stv_100.parquet"}
    ]
    status = pd.read_parquet(target_external / "_external_lc_status.parquet")
    assert status["candidate_id"].tolist() == ["stv_100"]
    assert status["module"].tolist() == ["TESS LCs"]
