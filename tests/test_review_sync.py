from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from malca.review import sync
from malca.review.store import db_connect, get_candidate_payload, get_review, upsert_candidates_frame


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(path: Path, assets: list[dict[str, object]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "assets": assets or []}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _seed_candidate_and_review(
    db_path: Path,
    candidate: dict[str, object],
    review: dict[str, object] | None = None,
    *,
    payload_extra: dict[str, object] | None = None,
) -> None:
    with db_connect(db_path) as conn:
        upsert_candidates_frame(conn, pd.DataFrame([candidate]))
        if payload_extra is not None:
            payload = dict(candidate)
            payload.update(payload_extra)
            conn.execute(
                "UPDATE candidates SET payload_json = ? WHERE candidate_id = ?",
                (json.dumps(payload, sort_keys=True), str(candidate["candidate_id"])),
            )
        if review is not None:
            conn.execute(
                """
                INSERT INTO reviews (
                    candidate_id, interest_score, event_class, review_pass, notes, status, workflow_status,
                    morphology_primary, morphology_secondary, physical_family, priority_tags_json,
                    taxonomy_version, legacy_review_json, reviewer, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review["candidate_id"],
                    review.get("interest_score"),
                    review.get("event_class", "unclassified"),
                    review.get("review_pass", 1),
                    review.get("notes", ""),
                    review.get("status", "unreviewed"),
                    review.get("workflow_status", review.get("status", "unreviewed")),
                    review.get("morphology_primary"),
                    review.get("morphology_secondary"),
                    review.get("physical_family"),
                    json.dumps(review.get("priority_tags", []), sort_keys=True),
                    review.get("taxonomy_version", 1),
                    json.dumps(review.get("legacy_review_json", {}), sort_keys=True),
                    review.get("reviewer", ""),
                    review["updated_at"],
                ),
            )
        conn.commit()


def test_review_sync_export_import_roundtrip_and_manifest_hashes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    db_path = run_dir / "review" / "review.db"
    results_path = run_dir / "results" / "lc_events_vetted.parquet"
    lc_path = run_dir / "bundle_assets" / "lightcurves" / "C1.dat2"
    raw_path = run_dir / "bundle_assets" / "lightcurves" / "C1.raw2"
    plot_path = run_dir / "plots" / "C1_candidate.png"
    phase_path = run_dir / "plots" / "C1_phase.png"
    _write_text(results_path, "placeholder\n")
    _write_text(lc_path, "lightcurve\n")
    _write_text(raw_path, "raw stats\n")
    _write_text(plot_path, "plot\n")
    _write_text(phase_path, "phase\n")

    _seed_candidate_and_review(
        db_path,
        {
            "candidate_id": "C1",
            "source_path": str(results_path),
            "asas_sn_id": "C1",
            "lc_path": str(lc_path),
            "dipper_score": 6.0,
            "phase_period_days": 2.5,
        },
        {
            "candidate_id": "C1",
            "interest_score": 4,
            "event_class": "dipper",
            "review_pass": 2,
            "notes": "line one,\nline two",
            "status": "reviewed",
            "morphology_primary": "dimming_event",
            "priority_tags": ["priority_dipper"],
            "reviewer": "tester",
            "updated_at": "2026-03-12T09:00:00+00:00",
        },
        payload_extra={
            "path": str(lc_path),
            "custom_nested": {"values": [1, None, {"ok": True}]},
            "phot_g_mean_mag": 14.2,
        },
    )

    out_dir = tmp_path / "reviews"
    result = sync.export_review_bundle(db_path, out_dir, hash_assets=True)

    assert result["candidates_exported"] == 1
    assert result["reviews_exported"] == 1
    candidates_text = (out_dir / "candidates.jsonl").read_text(encoding="utf-8")
    reviews_text = (out_dir / "reviews.jsonl").read_text(encoding="utf-8")
    manifest_text = (out_dir / "assets_manifest.json").read_text(encoding="utf-8")

    sync.export_review_bundle(db_path, out_dir, hash_assets=True)
    assert (out_dir / "candidates.jsonl").read_text(encoding="utf-8") == candidates_text
    assert (out_dir / "reviews.jsonl").read_text(encoding="utf-8") == reviews_text
    assert (out_dir / "assets_manifest.json").read_text(encoding="utf-8") == manifest_text

    candidate_record = json.loads(candidates_text.splitlines()[0])
    assert candidate_record["payload"]["custom_nested"]["values"][2]["ok"] is True
    assert candidate_record["payload"]["phot_g_mean_mag"] == 14.2

    review_record = json.loads(reviews_text.splitlines()[0])
    assert review_record["notes"] == "line one,\nline two"

    manifest = json.loads(manifest_text)
    manifest_entry = manifest["assets"][0]
    assert manifest_entry["candidate_id"] == "C1"
    by_kind = {asset["kind"]: asset for asset in manifest_entry["assets"]}
    assert by_kind["lightcurve"]["resolved_path"] == "bundle_assets/lightcurves/C1.dat2"
    assert by_kind["lightcurve"]["size_bytes"] == len("lightcurve\n")
    assert len(by_kind["lightcurve"]["sha256"]) == 64
    assert by_kind["raw_stats"]["resolved_path"] == "bundle_assets/lightcurves/C1.raw2"
    assert by_kind["plot"]["resolved_path"] == "plots/C1_candidate.png"
    assert by_kind["phase_plot"]["resolved_path"] == "plots/C1_phase.png"

    rebuilt_db = tmp_path / "rebuilt.db"
    import_result = sync.import_review_bundle(out_dir, db_path=rebuilt_db, replace=True)
    assert import_result["candidates_written"] == 1
    assert import_result["reviews_inserted"] == 1

    with db_connect(rebuilt_db) as conn:
        payload = get_candidate_payload(conn, "C1")
        review = get_review(conn, "C1")
    assert payload["custom_nested"]["values"][2]["ok"] is True
    assert payload["dipper_score"] == 6.0
    assert review["notes"] == "line one,\nline two"
    assert review["event_class"] == "dipper"


def test_review_sync_import_merge_keeps_newer_target_review(tmp_path: Path) -> None:
    in_dir = tmp_path / "reviews"
    _write_jsonl(
        in_dir / "candidates.jsonl",
        [
            {"schema_version": 1, "candidate_id": "C1", "asas_sn_id": "C1", "payload": {}},
            {"schema_version": 1, "candidate_id": "C2", "asas_sn_id": "C2", "payload": {}},
        ],
    )
    _write_jsonl(
        in_dir / "reviews.jsonl",
        [
            {
                "schema_version": 1,
                "candidate_id": "C1",
                "interest_score": 2,
                "event_class": "flare",
                "review_pass": 1,
                "notes": "source older",
                "status": "reviewed",
                "reviewer": "src",
                "updated_at": "2026-03-12T09:00:00+00:00",
            },
            {
                "schema_version": 1,
                "candidate_id": "C2",
                "interest_score": 3,
                "event_class": "dipper",
                "review_pass": 1,
                "notes": "source inserted",
                "status": "reviewed",
                "reviewer": "src",
                "updated_at": "2026-03-12T10:00:00+00:00",
            },
        ],
    )
    _write_manifest(in_dir / "assets_manifest.json")

    db_path = tmp_path / "target.db"
    _seed_candidate_and_review(
        db_path,
        {"candidate_id": "C1", "asas_sn_id": "C1"},
        {
            "candidate_id": "C1",
            "interest_score": 4,
            "event_class": "other",
            "review_pass": 2,
            "notes": "target newer",
            "status": "reviewed",
            "reviewer": "target",
            "updated_at": "2026-03-13T09:00:00+00:00",
        },
    )

    result = sync.import_review_bundle(in_dir, db_path=db_path)

    assert result["reviews_skipped"] == 1
    assert result["reviews_inserted"] == 1
    with db_connect(db_path) as conn:
        assert get_review(conn, "C1")["notes"] == "target newer"
        assert get_review(conn, "C2")["notes"] == "source inserted"


def test_review_sync_import_replace_drops_stale_rows(tmp_path: Path) -> None:
    in_dir = tmp_path / "reviews"
    _write_jsonl(
        in_dir / "candidates.jsonl",
        [{"schema_version": 1, "candidate_id": "C1", "asas_sn_id": "C1", "payload": {}}],
    )
    _write_jsonl(
        in_dir / "reviews.jsonl",
        [
            {
                "schema_version": 1,
                "candidate_id": "C1",
                "interest_score": 3,
                "event_class": "dipper",
                "review_pass": 1,
                "notes": "rebuilt",
                "status": "reviewed",
                "reviewer": "src",
                "updated_at": "2026-03-12T10:00:00+00:00",
            }
        ],
    )
    _write_manifest(in_dir / "assets_manifest.json")

    db_path = tmp_path / "target.db"
    _seed_candidate_and_review(
        db_path,
        {"candidate_id": "OLD", "asas_sn_id": "OLD"},
        {
            "candidate_id": "OLD",
            "interest_score": 1,
            "event_class": "other",
            "review_pass": 1,
            "notes": "stale",
            "status": "reviewed",
            "reviewer": "target",
            "updated_at": "2026-03-11T09:00:00+00:00",
        },
    )

    result = sync.import_review_bundle(in_dir, db_path=db_path, replace=True)

    assert result["replace"] is True
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT candidate_id FROM candidates ORDER BY candidate_id").fetchall()
    assert rows == [("C1",)]


def test_review_sync_cli_export_import_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _seed_candidate_and_review(
        db_path,
        {"candidate_id": "CLI1", "asas_sn_id": "CLI1", "dipper_score": 5.0},
        {
            "candidate_id": "CLI1",
            "interest_score": 2,
            "event_class": "dipper",
            "review_pass": 1,
            "notes": "cli",
            "status": "reviewed",
            "reviewer": "tester",
            "updated_at": "2026-03-12T10:00:00+00:00",
        },
    )

    out_dir = tmp_path / "reviews"
    assert sync.main(["export", "--review-db", str(db_path), "--output-dir", str(out_dir)]) == 0

    imported_db = tmp_path / "imported.db"
    assert sync.main(["import", "--review-db", str(imported_db), "--input-dir", str(out_dir), "--replace"]) == 0
    with db_connect(imported_db) as conn:
        assert get_review(conn, "CLI1")["notes"] == "cli"


def test_auto_export_review_bundle_is_best_effort(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _seed_candidate_and_review(
        db_path,
        {"candidate_id": "AUTO1", "asas_sn_id": "AUTO1"},
        {
            "candidate_id": "AUTO1",
            "interest_score": 2,
            "event_class": "dipper",
            "review_pass": 1,
            "notes": "auto",
            "status": "reviewed",
            "reviewer": "tester",
            "updated_at": "2026-03-12T10:00:00+00:00",
        },
    )

    messages: list[str] = []
    out_dir = tmp_path / "reviews"
    result = sync.auto_export_review_bundle(db_path, out_dir, logger=messages.append)

    assert result["ok"] is True
    assert (out_dir / "candidates.jsonl").exists()
    assert (out_dir / "reviews.jsonl").exists()
    assert (out_dir / "assets_manifest.json").exists()
    assert messages and "Exported review Git bundle" in messages[-1]

    bad_out_dir = tmp_path / "not_a_directory"
    bad_out_dir.write_text("file blocks mkdir", encoding="utf-8")
    failed = sync.auto_export_review_bundle(db_path, bad_out_dir, logger=messages.append)

    assert failed["ok"] is False
    assert "error" in failed
    assert messages and "failed" in messages[-1]
