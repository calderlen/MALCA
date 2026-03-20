from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.review import refresh as review_refresh
from malca.review.store import db_connect, get_candidate_payload, import_candidates, save_review


def test_refresh_review_stats_from_run_replaces_stats_only(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    results_dir = run_dir / "results"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    results_dir.mkdir(parents=True)
    bundle_dir.mkdir(parents=True)

    candidate_source = results_dir / "lc_events_filtered.parquet"
    pd.DataFrame([{"candidate_id": "REFRESH-1"}]).to_parquet(candidate_source, index=False)

    lc_path = bundle_dir / "REFRESH-1.dat2"
    lc_path.write_text("dummy", encoding="ascii")

    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "REFRESH-1",
                        "asas_sn_id": "REFRESH-1",
                        "path": "/missing/original/REFRESH-1.dat2",
                        "lc_path": "/missing/original/REFRESH-1.dat2",
                        "parallax": 7.1,
                        "stats_file_points_total": 999.0,
                        "stats_photometry_mean_mag": 99.0,
                        "stats_legacy_extra": 123.0,
                    }
                ]
            ),
            source_path=str(candidate_source),
            characterize_before_import=False,
            vet_before_import=False,
        )

    def fake_compute_stats(_candidate_id: str, _parent: str, *, compute_ls: bool = True, file_ext: str | None = None):
        assert compute_ls is True
        assert file_ext == "dat2"
        return pd.DataFrame(), {
            "file_points_total": 12.0,
            "file_points_kept_after_filter": 11.0,
            "cadence_median_dt_days": 2.5,
            "photometry_mean_mag": 14.2,
            "photometry_median_mag": 14.1,
        }

    monkeypatch.setattr(review_refresh, "compute_stats", fake_compute_stats)

    result = review_refresh.refresh_review_stats_from_run(run_dir, db_path)

    assert result["refreshed"] == 1
    assert result["unresolved"] == []
    assert result["failed"] == []

    with db_connect(db_path) as conn:
        payload = get_candidate_payload(conn, "REFRESH-1")
        row = conn.execute(
            "SELECT lc_path, stats_file_points_total, stats_photometry_mean_mag FROM candidates WHERE candidate_id = ?",
            ("REFRESH-1",),
        ).fetchone()

    assert payload["parallax"] == 7.1
    assert payload["lc_path"] == str(lc_path)
    assert payload["stats_file_points_total"] == 12.0
    assert payload["stats_photometry_mean_mag"] == 14.2
    assert payload["n_points"] == 11.0
    assert payload["cadence_median_days"] == 2.5
    assert payload["baseline_mag"] == 14.1
    assert "stats_legacy_extra" not in payload
    assert row == (str(lc_path), 12.0, 14.2)


def test_rebuild_review_db_drops_obsolete_candidate_columns(tmp_path: Path) -> None:
    old_db = tmp_path / "old_review.db"
    new_db = tmp_path / "new_review.db"

    with db_connect(old_db) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "REFRESH-2", "asas_sn_id": "REFRESH-2"}]),
            source_path="test://refresh",
            characterize_before_import=False,
            vet_before_import=False,
        )
        save_review(
            conn,
            candidate_id="REFRESH-2",
            interest_score=3,
            review_pass=1,
            notes="keep",
            status="reviewed",
            reviewer="tester",
        )
        conn.execute("ALTER TABLE candidates ADD COLUMN stats_obsolete REAL")
        conn.execute("UPDATE candidates SET stats_obsolete = 42.0 WHERE candidate_id = ?", ("REFRESH-2",))
        conn.commit()

    rebuilt = review_refresh.rebuild_review_db(old_db, new_db)

    assert rebuilt["candidates"] == 1
    assert rebuilt["reviews"] == 1
    assert rebuilt["review_history"] == 1

    with db_connect(new_db) as conn:
        candidate_cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        review_row = conn.execute(
            "SELECT interest_score, status, reviewer FROM reviews WHERE candidate_id = ?",
            ("REFRESH-2",),
        ).fetchone()

    assert "stats_obsolete" not in candidate_cols
    assert review_row == (3, "reviewed", "tester")


def test_refresh_review_stats_from_ltv_scope_matches_db_by_asas_sn_id(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "ltv_run"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)

    candidate_source = run_dir / "LTvar12-12.5_pipeline.parquet"
    pd.DataFrame([{"asas_sn_id": "123"}]).to_parquet(candidate_source, index=False)

    lc_path = bundle_dir / "123.dat2"
    lc_path.write_text("dummy", encoding="ascii")

    db_path = tmp_path / "ltv_review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "ltv_123",
                        "asas_sn_id": "123",
                        "path": "/missing/original/123.dat2",
                        "lc_path": "/missing/original/123.dat2",
                    }
                ]
            ),
            source_path=str(candidate_source),
            characterize_before_import=False,
            vet_before_import=False,
        )

    def fake_compute_stats(_candidate_id: str, _parent: str, *, compute_ls: bool = True, file_ext: str | None = None):
        assert file_ext == "dat2"
        return pd.DataFrame(), {
            "file_points_total": 10.0,
            "file_points_kept_after_filter": 9.0,
            "cadence_median_dt_days": 1.5,
            "photometry_mean_mag": 13.7,
            "photometry_median_mag": 13.6,
        }

    monkeypatch.setattr(review_refresh, "compute_stats", fake_compute_stats)

    result = review_refresh.refresh_review_stats_from_run(
        run_dir,
        db_path,
        candidate_source=candidate_source,
    )

    assert result["scoped_candidates"] == 1
    assert result["matched_db_rows"] == 1
    assert result["refreshed"] == 1
    assert result["missing_from_db"] == []

    with db_connect(db_path) as conn:
        payload = get_candidate_payload(conn, "ltv_123")

    assert payload["lc_path"] == str(lc_path)


def test_get_candidate_payload_aliases_legacy_ltv_pm_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "ltv_review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "ltv_pm",
                        "asas_sn_id": "999",
                        "gaia_pmra": 6.0,
                        "gaia_pmdec": 8.0,
                        "gaia_pm_total": 10.0,
                    }
                ]
            ),
            source_path="test://ltv",
            characterize_before_import=False,
            vet_before_import=False,
        )
        payload = get_candidate_payload(conn, "ltv_pm")

    assert payload["pmra"] == 6.0
    assert payload["pmdec"] == 8.0
    assert payload["pm_total"] == 10.0
    assert payload["high_pm_flag"] is False

    with db_connect(db_path) as conn:
        row = conn.execute(
            "SELECT pmra, pmdec, pm_total, high_pm_flag FROM candidates WHERE candidate_id = ?",
            ("ltv_pm",),
        ).fetchone()

    assert row == (6.0, 8.0, 10.0, 0)
