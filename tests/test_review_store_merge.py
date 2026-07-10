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


def test_import_candidates_preserves_halpha_photometry_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    df = pd.DataFrame([
        {
            "candidate_id": "HA-1",
            "asas_sn_id": "HA-1",
            "iphas_r_mag": 12.20,
            "iphas_i_mag": 11.87,
            "iphas_ha_mag": 12.04,
            "iphas_r_i": 0.33,
            "iphas_r_ha": 0.16,
            "iphas_sep_arcsec": 0.4,
            "iphas_source_catalog": "II/321/iphas2",
            "vphas_r_mag": 16.0,
            "vphas_i_mag": 15.3,
            "vphas_ha_mag": 15.5,
            "vphas_r_i": 0.7,
            "vphas_r_ha": 0.5,
            "vphas_sep_arcsec": 0.7,
            "vphas_source_catalog": "II/341/vphasp",
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
        payload = get_candidate_payload(conn, "HA-1")

    assert payload["iphas_r_mag"] == 12.20
    assert payload["iphas_i_mag"] == 11.87
    assert payload["iphas_ha_mag"] == 12.04
    assert payload["iphas_r_i"] == 0.33
    assert payload["iphas_r_ha"] == 0.16
    assert payload["iphas_source_catalog"] == "II/321/iphas2"
    assert payload["vphas_r_mag"] == 16.0
    assert payload["vphas_i_mag"] == 15.3
    assert payload["vphas_ha_mag"] == 15.5
    assert payload["vphas_r_i"] == 0.7
    assert payload["vphas_r_ha"] == 0.5
    assert payload["vphas_source_catalog"] == "II/341/vphasp"


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


def test_upsert_candidates_frame_derives_known_new_and_unset_vetting_states(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "KNOWN",
                        "asas_sn_id": "KNOWN",
                        "gaia_var_class": "LPV",
                    },
                    {
                        "candidate_id": "LIKELY-NEW",
                        "asas_sn_id": "LIKELY-NEW",
                        "char_status_yso": "ok",
                        "yso_class": "Main Sequence",
                    },
                    {
                        "candidate_id": "RAW",
                        "asas_sn_id": "RAW",
                    },
                ]
            ),
        )
        payload_known = get_candidate_payload(conn, "KNOWN")
        payload_new = get_candidate_payload(conn, "LIKELY-NEW")
        payload_raw = get_candidate_payload(conn, "RAW")
        sql_rows = {
            cid: likely_known
            for cid, likely_known in conn.execute(
                "SELECT candidate_id, vetting_likely_known FROM candidates ORDER BY candidate_id"
            ).fetchall()
        }

    assert payload_known["vetting_likely_known"] is True
    assert payload_new["vetting_likely_known"] is False
    assert "vetting_likely_known" not in payload_raw
    assert sql_rows == {"KNOWN": 1, "LIKELY-NEW": 0, "RAW": None}


def test_upsert_candidates_frame_derives_known_from_definite_vsx_class(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "VSX-EA",
                        "asas_sn_id": "VSX-EA",
                        "vsx_class": "EA",
                        "vetting_likely_known": False,
                    }
                ]
            ),
        )
        payload = get_candidate_payload(conn, "VSX-EA")

    assert payload["vetting_likely_known"] is True
    assert payload["vsx_class"] == "EA"


def test_upsert_candidates_frame_does_not_derive_known_from_generic_simbad_type(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "SIMBAD-IR",
                        "asas_sn_id": "SIMBAD-IR",
                        "simbad_main_id": "IR Source",
                        "simbad_otype": "IR",
                        "vetting_likely_known": False,
                    }
                ]
            ),
        )
        payload = get_candidate_payload(conn, "SIMBAD-IR")

    assert payload["vetting_likely_known"] is False
    assert payload["simbad_otype"] == "IR"


def test_upsert_candidates_frame_derives_known_from_variable_simbad_type(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "SIMBAD-RR",
                        "asas_sn_id": "SIMBAD-RR",
                        "simbad_otype": "RR*",
                        "vetting_likely_known": False,
                    }
                ]
            ),
        )
        payload = get_candidate_payload(conn, "SIMBAD-RR")

    assert payload["vetting_likely_known"] is True
    assert payload["simbad_otype"] == "RR*"


def test_merge_candidate_results_derives_known_from_backfilled_vsx_class(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "VSX-BACKFILL",
                        "asas_sn_id": "300647863051",
                        "vetting_likely_known": False,
                    }
                ]
            ),
        )
        updated = merge_candidate_results(
            conn,
            pd.DataFrame(
                [
                    {
                        "asas_sn_id": "300647863051",
                        "vsx_class": "EA",
                        "vsx_period": 1.7292,
                    }
                ]
            ),
            id_column="asas_sn_id",
        )
        payload = get_candidate_payload(conn, "VSX-BACKFILL")

    assert updated == 1
    assert payload["vetting_likely_known"] is True
    assert payload["vsx_class"] == "EA"


def test_get_candidate_payload_derives_known_from_existing_vsx_class_when_summary_is_stale(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "STALE-VSX",
                        "asas_sn_id": "STALE-VSX",
                        "vsx_class": "EA",
                        "vetting_likely_known": True,
                    }
                ]
            ),
        )
        stale_payload = {
            "candidate_id": "STALE-VSX",
            "asas_sn_id": "STALE-VSX",
            "vsx_class": "EA",
            "vetting_likely_known": False,
        }
        conn.execute(
            "UPDATE candidates SET vetting_likely_known=0, payload_json=? WHERE candidate_id=?",
            (json.dumps(stale_payload), "STALE-VSX"),
        )
        conn.commit()

        payload = get_candidate_payload(conn, "STALE-VSX")

    assert payload["vetting_likely_known"] is True
    assert payload["vsx_class"] == "EA"


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
