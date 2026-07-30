from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from malca.catalogs import gaia_fetch
from malca.catalogs.gaia_banyan_backfill import load_review_cohort
from malca.enrichment.banyan import compute_banyan_membership
from malca.enrichment.characterize import (
    _module_completed,
    gaia_enrichment_needed_mask,
    merge_gaia_catalog_rows,
)
from malca.review.store import init_db


def _complete_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "gaia_id": "1",
        "source_id": "1",
        "ra": 10.0,
        "dec": 20.0,
        "phot_g_mean_mag": 12.0,
        "phot_bp_mean_mag": 12.5,
        "phot_rp_mean_mag": 11.5,
        "parallax": 10.0,
        "parallax_error": 0.2,
        "pmra": 5.0,
        "pmra_error": 0.1,
        "pmdec": -3.0,
        "pmdec_error": 0.1,
        "radial_velocity": 15.0,
        "radial_velocity_error": 1.0,
    }
    row.update(overrides)
    return row


def test_banyan_adapter_uses_real_api_shape_and_records_provenance() -> None:
    calls: list[dict[str, object]] = []

    def fake_membership(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                ("ALL", "FIELD"): [0.8],
                ("ALL", "FIELD_MS"): [0.1],
                ("ALL", "BETA_PIC"): [0.1],
                ("YA_PROB", "Global"): [0.1],
                ("BEST_YA", "Global"): ["BETA_PIC"],
            }
        )

    out = compute_banyan_membership(
        pd.DataFrame([_complete_row()]),
        association_threshold=0.05,
        membership_func=fake_membership,
    )

    assert len(calls) == 1
    assert calls[0]["use_plx"] is True
    assert calls[0]["use_rv"] is True
    assert out.loc[0, "banyan_status"] == "ok"
    assert out.loc[0, "banyan_input_mode"] == "pm+plx+rv"
    assert np.isclose(out.loc[0, "banyan_field_prob"], 0.9)
    assert out.loc[0, "banyan_best_assoc"] == "BETA_PIC"
    assert np.isclose(out.loc[0, "banyan_best_assoc_prob"], 0.1)
    assert out.loc[0, "banyan_adapter_version"]
    assert out.loc[0, "banyan_version"]


def test_banyan_adapter_explains_missing_pm_errors_without_calling_package() -> None:
    called = False

    def fail_if_called(**_kwargs: object) -> pd.DataFrame:
        nonlocal called
        called = True
        raise AssertionError("ineligible row must not call BANYAN")

    row = _complete_row(pmra_error=np.nan)
    out = compute_banyan_membership(
        pd.DataFrame([row]),
        membership_func=fail_if_called,
    )

    assert not called
    assert out.loc[0, "banyan_status"] == "missing_proper_motion_error"
    assert pd.isna(out.loc[0, "banyan_field_prob"])


def test_banyan_adapter_falls_back_to_pm_only_for_nonphysical_parallax() -> None:
    calls: list[dict[str, object]] = []

    def fake_membership(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                ("ALL", "FIELD"): [1.0],
                ("YA_PROB", "Global"): [0.0],
                ("BEST_YA", "Global"): ["FIELD"],
            }
        )

    row = _complete_row(parallax=-1.0, radial_velocity=np.nan, radial_velocity_error=np.nan)
    out = compute_banyan_membership(pd.DataFrame([row]), membership_func=fake_membership)

    assert len(calls) == 1
    assert "plx" not in calls[0]
    assert "rv" not in calls[0]
    assert out.loc[0, "banyan_input_mode"] == "pm"
    assert out.loc[0, "banyan_status"] == "ok"


def test_gaia_completeness_is_row_specific_and_merge_prefers_gaia_id() -> None:
    frame = pd.DataFrame(
        [
            _complete_row(source_id="999", pmra_error=np.nan, pmdec_error=np.nan),
            _complete_row(gaia_id="2", source_id="2"),
        ]
    )
    needed = gaia_enrichment_needed_mask(frame)
    assert needed.tolist() == [True, False]

    gaia_row = pd.DataFrame(
        [
            {
                "source_id": "1",
                "pmra_error": 0.2,
                "pmdec_error": 0.3,
                "gaia_fetch_schema_version": "2",
            }
        ]
    )
    out = merge_gaia_catalog_rows(frame, gaia_row)

    assert np.isclose(out.loc[0, "pmra_error"], 0.2)
    assert np.isclose(out.loc[0, "pmdec_error"], 0.3)
    assert bool(out.loc[0, "gaia_banyan_input_complete"])
    assert out.loc[0, "gaia_enrichment_status"] == "complete"
    assert out.loc[1, "gaia_enrichment_status"] == "existing_complete"


def test_banyan_checkpoint_reruns_when_new_gaia_inputs_become_eligible() -> None:
    old = compute_banyan_membership(
        pd.DataFrame([_complete_row(pmra_error=np.nan, pmdec_error=np.nan)]),
        membership_func=lambda **_kwargs: pd.DataFrame(),
    )
    old["char_status_banyan"] = "ok"
    assert _module_completed(old, "banyan")

    old["pmra_error"] = 0.1
    old["pmdec_error"] = 0.1
    assert not _module_completed(old, "banyan")


def test_gaia_fetch_refreshes_requested_legacy_rows_and_preserves_other_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = tmp_path / "gaia.parquet"
    pd.DataFrame(
        [
            {"source_id": "1", "pmra": 1.0},
            {"source_id": "2", "pmra": 2.0},
        ]
    ).to_parquet(cache, index=False)
    monkeypatch.setattr(
        gaia_fetch,
        "canonicalize_gaia_ids",
        lambda values, **_kwargs: pd.DataFrame(
            {"source_id": list(values), "gaia_id_mapping_status": ["dr3"] * len(values)}
        ),
    )
    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda _url: object())

    def fake_fetch(_service: object, chunk_ids: list[str]) -> pd.DataFrame:
        return gaia_fetch._mark_current_fetch_rows(
            pd.DataFrame(
                [
                    {
                        "source_id": source_id,
                        "ra": 10.0,
                        "dec": 20.0,
                        "pmra": 1.0,
                        "pmra_error": 0.1,
                        "pmdec": 2.0,
                        "pmdec_error": 0.1,
                    }
                    for source_id in chunk_ids
                ]
            )
        )

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", fake_fetch)
    out = gaia_fetch.fetch_gaia_catalog(["1"], output_path=cache)

    assert set(out["source_id"].dropna().astype(str)) == {"1", "2"}
    refreshed = out[out["source_id"].astype(str).eq("1")].iloc[0]
    preserved = out[out["source_id"].astype(str).eq("2")].iloc[0]
    assert refreshed["gaia_fetch_schema_version"] == gaia_fetch.GAIA_FETCH_SCHEMA_VERSION
    assert np.isclose(refreshed["pmra_error"], 0.1)
    assert pd.isna(preserved["gaia_fetch_schema_version"])


def test_reviewed_dipper_cohort_selection(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        init_db(conn)
        conn.execute(
            """
            INSERT INTO candidates(candidate_id, payload_json, imported_at)
            VALUES (?, ?, ?), (?, ?, ?)
            """,
            (
                "dip", '{"gaia_id":"1"}', "2026-01-01T00:00:00Z",
                "other", '{"gaia_id":"2"}', "2026-01-01T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO reviews(
                candidate_id, status, workflow_status, morphology_primary, updated_at
            )
            VALUES ('dip', 'reviewed', 'reviewed', 'dimming_event', '2026-01-01T00:00:00Z'),
                   ('other', 'reviewed', 'reviewed', 'periodic', '2026-01-01T00:00:00Z')
            """
        )

    cohort = load_review_cohort(db_path)
    assert cohort["candidate_id"].tolist() == ["dip"]
    assert str(cohort.loc[0, "gaia_id"]) == "1"
