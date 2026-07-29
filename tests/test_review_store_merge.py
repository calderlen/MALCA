from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from malca.review.eda_data import load_review_db
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    get_candidate_payload,
    import_candidates,
    merge_candidate_results,
    merge_dipper_probability_scores,
    merge_dipper_recurrence_ml_scores,
    merge_eight_class_probability_scores,
    merge_hierarchical_ml_scores,
    merge_nine_class_probability_scores,
    refresh_dipper_recurrence_classifications,
    review_content_signature,
    save_app_state,
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
                INSERT INTO reviews (candidate_id, classification_confidence, event_class, review_pass, notes, status, reviewer, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review["candidate_id"],
                    review.get("classification_confidence"),
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
            {"candidate_id": "C1", "classification_confidence": 4, "event_class": "dipper", "review_pass": 2, "notes": "newer", "status": "reviewed", "reviewer": "src", "updated_at": "2026-03-12T12:00:00+00:00"},
            {"candidate_id": "C2", "classification_confidence": 2, "event_class": "flare", "review_pass": 1, "notes": "inserted", "status": "reviewed", "reviewer": "src", "updated_at": "2026-03-12T11:00:00+00:00"},
        ],
    )
    _seed_review_db(
        target_db,
        [{"candidate_id": "C1", "source_path": "/target", "asas_sn_id": "C1", "dipper_score": 4.0}],
        [{"candidate_id": "C1", "classification_confidence": 1, "event_class": "other", "review_pass": 1, "notes": "older", "status": "reviewed", "reviewer": "target", "updated_at": "2026-03-11T10:00:00+00:00"}],
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
                "classification_confidence": 4,
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


def test_merge_candidate_results_null_is_sparse_unless_explicitly_cleared(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "dipper_score": 4.0}]),
        )
        merge_candidate_results(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "dipper_score": None}]),
        )
        assert get_candidate_payload(conn, "C1")["dipper_score"] == 4.0

        merge_candidate_results(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "dipper_score": None}]),
            clear_columns={"dipper_score"},
        )
        assert get_candidate_payload(conn, "C1").get("dipper_score") in (None, "")


def test_merge_dipper_probability_scores_updates_review_candidates(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"candidate_id": "C1", "prob_dipper_like": 0.91},
            {"candidate_id": "C2", "prob_dipper_like": 1.5},
            {"candidate_id": "OTHER", "prob_dipper_like": 0.2},
        ]
    ).to_parquet(scores_path, index=False)

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1"}, {"candidate_id": "C2"}]),
        )
        updated = merge_dipper_probability_scores(conn, scores_path)

        content_signature = review_content_signature(review_db)
        unchanged = merge_dipper_probability_scores(conn, scores_path)

        payload1 = get_candidate_payload(conn, "C1")
        payload2 = get_candidate_payload(conn, "C2")

    assert updated == 2
    assert unchanged == 0
    assert review_content_signature(review_db) == content_signature
    assert payload1["prob_dipper_like"] == 0.91
    assert payload2["prob_dipper_like"] == 1.0


def test_refresh_dipper_recurrence_classifications_is_deterministic(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "recurrent", "dip_run_count": 2},
                    {"candidate_id": "one_run", "dip_run_count": 1},
                    {"candidate_id": "flagged_single", "dip_is_single_event": True},
                    {"candidate_id": "unknown", "dip_run_count": 0},
                ]
            ),
        )
        assert refresh_dipper_recurrence_classifications(conn) == 4
        assert refresh_dipper_recurrence_classifications(conn) == 0
        rows = conn.execute(
            """
            SELECT candidate_id, dipper_recurrence_class, dipper_recurrence_evidence
            FROM candidates ORDER BY candidate_id
            """
        ).fetchall()

    assert rows == [
        ("flagged_single", "non_recurrent", "pipeline_single_event_flag"),
        ("one_run", "non_recurrent", "one_detected_dip_run"),
        ("recurrent", "recurrent", "two_or_more_detected_dip_runs"),
        ("unknown", "unknown", "insufficient_dip_recurrence_evidence"),
    ]


def test_merge_dipper_recurrence_ml_scores_updates_conditional_and_parent_scores(
    tmp_path: Path,
) -> None:
    review_db = tmp_path / "review.db"
    scores_path = tmp_path / "recurrence_scores.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "prob_recurrent_given_dipper": 0.8,
                "prob_recurrent_dipper_binary": 0.4,
                "predicted_dipper_recurrence": "recurrent",
            },
            {
                "candidate_id": "C2",
                "prob_recurrent_given_dipper": 0.3,
                "prob_recurrent_dipper_binary": 0.2,
                "predicted_dipper_recurrence": "non_recurrent",
            },
        ]
    ).to_parquet(scores_path, index=False)

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1"}, {"candidate_id": "C2"}]),
        )
        assert merge_dipper_recurrence_ml_scores(conn, scores_path) == 2
        assert merge_dipper_recurrence_ml_scores(conn, scores_path) == 0
        rows = conn.execute(
            """
            SELECT candidate_id, prob_recurrent_given_dipper,
                   prob_recurrent_dipper_binary, predicted_dipper_recurrence
            FROM candidates ORDER BY candidate_id
            """
        ).fetchall()

    assert rows == [
        ("C1", 0.8, 0.4, "recurrent"),
        ("C2", 0.3, 0.2, "non_recurrent"),
    ]


def test_merge_nine_class_probability_scores_requires_full_matching_coverage(
    tmp_path: Path,
) -> None:
    review_db = tmp_path / "review.db"
    scores_path = tmp_path / "nine_class_scores.parquet"
    rows = [
        {
            "candidate_id": "C1",
            "prob_artifact_or_bad_photometry": 0.01,
            "prob_brightening_event": 0.01,
            "prob_dipper": 0.81,
            "prob_eclipsing_binary_like": 0.02,
            "prob_long_period_variable": 0.01,
            "prob_long_term_variable": 0.03,
            "prob_microlensing": 0.04,
            "prob_nonvariable_or_low_snr": 0.01,
            "prob_quasi_periodic": 0.09,
        },
        {
            "candidate_id": "C2",
            "prob_artifact_or_bad_photometry": 0.01,
            "prob_brightening_event": 0.02,
            "prob_dipper": 0.12,
            "prob_eclipsing_binary_like": 0.71,
            "prob_long_period_variable": 0.01,
            "prob_long_term_variable": 0.03,
            "prob_microlensing": 0.04,
            "prob_nonvariable_or_low_snr": 0.01,
            "prob_quasi_periodic": 0.08,
        },
    ]
    pd.DataFrame(rows).to_parquet(scores_path, index=False)

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1"}, {"candidate_id": "C2"}]),
        )
        updated = merge_nine_class_probability_scores(conn, scores_path)
        unchanged = merge_nine_class_probability_scores(conn, scores_path)
        payload = get_candidate_payload(conn, "C1")

    assert updated == 2
    assert unchanged == 0
    assert payload["prob_artifact_or_bad_photometry"] == pytest.approx(0.01)
    assert payload["prob_dipper"] == pytest.approx(0.81)
    assert payload["prob_eclipsing_binary_like"] == pytest.approx(0.02)
    assert payload["prob_long_period_variable"] == pytest.approx(0.01)
    assert payload["prob_nonvariable_or_low_snr"] == pytest.approx(0.01)
    assert payload["prob_quasi_periodic"] == pytest.approx(0.09)


def test_merge_eight_class_probability_scores_requires_full_matching_coverage(
    tmp_path: Path,
) -> None:
    review_db = tmp_path / "review.db"
    scores_path = tmp_path / "eight_class_scores.parquet"
    rows = [
        {
            "candidate_id": "C1",
            "prob_artifact_or_nonvariable": 0.03,
            "prob_brightening_event": 0.01,
            "prob_dipper": 0.81,
            "prob_eclipsing_binary_like": 0.02,
            "prob_long_period_variable": 0.01,
            "prob_long_term_variable": 0.03,
            "prob_microlensing": 0.04,
            "prob_quasi_periodic": 0.05,
        },
        {
            "candidate_id": "C2",
            "prob_artifact_or_nonvariable": 0.08,
            "prob_brightening_event": 0.02,
            "prob_dipper": 0.10,
            "prob_eclipsing_binary_like": 0.70,
            "prob_long_period_variable": 0.01,
            "prob_long_term_variable": 0.03,
            "prob_microlensing": 0.02,
            "prob_quasi_periodic": 0.04,
        },
    ]
    pd.DataFrame(rows).to_parquet(scores_path, index=False)

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1"}, {"candidate_id": "C2"}]),
        )
        updated = merge_eight_class_probability_scores(conn, scores_path)
        unchanged = merge_eight_class_probability_scores(conn, scores_path)
        payload = get_candidate_payload(conn, "C1")

    assert updated == 2
    assert unchanged == 0
    assert payload["prob_artifact_or_nonvariable"] == pytest.approx(0.03)
    assert payload["prob_dipper"] == pytest.approx(0.81)
    assert payload["prob_eclipsing_binary_like"] == pytest.approx(0.02)
    assert payload["prob_long_period_variable"] == pytest.approx(0.01)
    assert payload["prob_quasi_periodic"] == pytest.approx(0.05)


def test_merge_hierarchical_ml_scores_updates_probabilities_and_predictions(
    tmp_path: Path,
) -> None:
    review_db = tmp_path / "review.db"
    scores_path = tmp_path / "hierarchical_scores.parquet"
    from malca.review.store import (
        HIERARCHICAL_ML_PREDICTION_COLUMNS,
        HIERARCHICAL_ML_PROBABILITY_COLUMNS,
    )

    rows = []
    for candidate_id, usable in (("C1", 0.9), ("C2", 0.2)):
        row: dict[str, object] = {"candidate_id": candidate_id}
        for index, column in enumerate(HIERARCHICAL_ML_PROBABILITY_COLUMNS):
            row[column] = min(1.0, usable / (index + 1))
        row.update(
            {
                "predicted_hierarchy_gate": (
                    "usable_astrophysical_variable"
                    if usable > 0.5
                    else "artifact_or_nonvariable"
                ),
                "predicted_primary_morphology": "dipper_dimming",
                "predicted_hierarchical_class": (
                    "dipper_dimming"
                    if usable > 0.5
                    else "artifact_or_nonvariable"
                ),
                "predicted_quasi_periodic": "not_quasi_periodic",
                "predicted_microlensing_like": "not_microlensing_like",
                "predicted_long_timescale_subtype": "long_term_variable",
                "predicted_dipper_recurrence": "non_recurrent",
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(scores_path, index=False)

    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1"}, {"candidate_id": "C2"}]),
        )
        assert merge_hierarchical_ml_scores(conn, scores_path) == 2
        assert merge_hierarchical_ml_scores(conn, scores_path) == 0
        payload = get_candidate_payload(conn, "C1")

    assert payload["prob_usable_astrophysical_variable"] == pytest.approx(0.45)
    assert payload["predicted_hierarchy_gate"] == "usable_astrophysical_variable"
    assert payload["predicted_hierarchical_class"] == "dipper_dimming"
    assert set(HIERARCHICAL_ML_PREDICTION_COLUMNS).issubset(payload)


def test_review_content_signature_ignores_app_state_but_tracks_candidate_changes(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(conn, pd.DataFrame([{"candidate_id": "C1"}]))
    before = review_content_signature(review_db)

    with db_connect(review_db) as conn:
        save_app_state(conn, "gui-only", "expanded")
    assert review_content_signature(review_db) == before

    with db_connect(review_db) as conn:
        conn.execute("UPDATE candidates SET dipper_score = ? WHERE candidate_id = ?", (7.0, "C1"))
        conn.commit()
    assert review_content_signature(review_db) != before


def test_current_review_schema_startup_check_performs_no_write(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        review_schema_before = conn.execute(
            "SELECT value, updated_at FROM app_state WHERE key = 'review_db_schema_version'"
        ).fetchone()
        sed_schema_before = conn.execute(
            "SELECT value, updated_at FROM sed_storage_meta WHERE key = 'schema_version'"
        ).fetchone()

    assert ensure_review_db_schema(review_db) is False

    with db_connect(review_db) as conn:
        review_schema_after = conn.execute(
            "SELECT value, updated_at FROM app_state WHERE key = 'review_db_schema_version'"
        ).fetchone()
        sed_schema_after = conn.execute(
            "SELECT value, updated_at FROM sed_storage_meta WHERE key = 'schema_version'"
        ).fetchone()
    assert review_schema_after == review_schema_before
    assert sed_schema_after == sed_schema_before


def test_review_schema_migrates_to_one_numeric_confidence_column(
    tmp_path: Path,
) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": f"C{score}"} for score in range(1, 5)]),
        )
        conn.execute("ALTER TABLE reviews ADD COLUMN interest_score INTEGER")
        for score in range(1, 5):
            conn.execute(
                """
                INSERT INTO reviews (
                    candidate_id, interest_score, event_class, review_pass,
                    notes, status, reviewer, classification_confidence, updated_at
                ) VALUES (?, ?, 'other', 1, '', 'reviewed', 'tester', NULL, ?)
                """,
                (f"C{score}", score, f"2026-07-21T00:00:0{score}+00:00"),
            )
        conn.execute(
            "UPDATE app_state SET value = '5' WHERE key = 'review_db_schema_version'"
        )
        conn.commit()

    assert ensure_review_db_schema(review_db) is True

    with db_connect(review_db) as conn:
        rows = conn.execute(
            """
            SELECT classification_confidence
            FROM reviews ORDER BY classification_confidence
            """
        ).fetchall()
        columns = {
            row[1]: row[2] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
        }

    assert rows == [(1,), (2,), (3,), (4,)]
    assert columns["classification_confidence"] == "INTEGER"
    assert "interest_score" not in columns


def test_review_schema_v5_clears_only_synthetic_lsp_aliases(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "fake",
                        "periodicity_period": 4.0,
                        "periodicity_bootstrap_sig": 0.01,
                        "periodicity_alias_flag": False,
                        "periodicity_is_significant": True,
                        "lsp_power": None,
                        "lsp_period": 4.0,
                        "lsp_bootstrap_sig": 0.01,
                        "lsp_is_alias": False,
                        "lsp_is_significant": True,
                    },
                    {
                        "candidate_id": "real",
                        "periodicity_period": 4.0,
                        "periodicity_bootstrap_sig": 0.01,
                        "periodicity_alias_flag": False,
                        "periodicity_is_significant": True,
                        "lsp_power": 0.8,
                        "lsp_period": 7.0,
                        "lsp_bootstrap_sig": 0.02,
                        "lsp_is_alias": False,
                        "lsp_is_significant": False,
                    },
                ]
            ),
        )
        conn.execute(
            "UPDATE app_state SET value = '4' WHERE key = 'review_db_schema_version'"
        )
        conn.commit()

    assert ensure_review_db_schema(review_db) is True

    with db_connect(review_db) as conn:
        fake = conn.execute(
            """
            SELECT lsp_power, lsp_period, lsp_bootstrap_sig,
                   lsp_is_alias, lsp_is_significant
            FROM candidates WHERE candidate_id = 'fake'
            """
        ).fetchone()
        real = conn.execute(
            """
            SELECT lsp_power, lsp_period, lsp_bootstrap_sig,
                   lsp_is_alias, lsp_is_significant
            FROM candidates WHERE candidate_id = 'real'
            """
        ).fetchone()

    assert fake == (None, None, None, None, None)
    assert real == (0.8, 7.0, 0.02, 0, 0)


def test_review_schema_has_candidate_first_catalog_neighbor_indexes(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(catalog_neighbors)").fetchall()
        }
    assert "idx_catalog_neighbors_candidate_known_sep" in indexes
    assert "idx_catalog_neighbors_candidate_dipper_sep" in indexes


def test_merge_candidate_results_does_not_cross_identifier_namespaces(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "ALIAS", "asas_sn_id": "OTHER"},
                    {"candidate_id": "C2", "asas_sn_id": "ALIAS"},
                ]
            ),
        )
        assert merge_candidate_results(
            conn,
            pd.DataFrame([{"candidate_id": "OTHER", "dipper_score": 7.0}]),
            id_column="candidate_id",
        ) == 0
        assert merge_candidate_results(
            conn,
            pd.DataFrame([{"asas_sn_id": "ALIAS", "dipper_score": 8.0}]),
            id_column="asas_sn_id",
        ) == 1

        assert get_candidate_payload(conn, "C2")["dipper_score"] == 8.0
        assert "dipper_score" not in get_candidate_payload(conn, "ALIAS")


def test_sparse_merge_does_not_downgrade_known_status_from_partial_catalog_context(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [{"candidate_id": "KNOWN", "gaia_var_class": "LPV", "vetting_likely_known": True}]
            ),
        )
        merge_candidate_results(
            conn,
            pd.DataFrame([{"candidate_id": "KNOWN", "char_status_yso": "ok", "yso_class": "Main Sequence"}]),
        )

        assert get_candidate_payload(conn, "KNOWN")["vetting_likely_known"] is True


def test_clearing_one_stats_field_does_not_erase_the_stats_family(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "stats_amplitude": 1.0, "stats_skew": 2.0}]),
        )
        merge_candidate_results(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "stats_amplitude": None}]),
            clear_columns={"stats_amplitude"},
        )
        payload = get_candidate_payload(conn, "C1")

        assert payload.get("stats_amplitude") in (None, "")
        assert payload["stats_skew"] == 2.0


def test_merge_candidate_results_refuses_to_clear_identity_fields(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(conn, pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "A1"}]))

        with pytest.raises(ValueError, match="identity/provenance"):
            merge_candidate_results(
                conn,
                pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": None}]),
                clear_columns={"asas_sn_id"},
            )
