from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    load_app_state,
    save_app_state,
    save_review,
    upsert_candidates_frame,
)
from malca.review.tui_service import (
    CandidateRecord,
    EmptyQueueError,
    NumericRange,
    QueueFilterSpec,
    ReviewRepository,
    TUI_FILTER_STATE_KEY,
)
from malca.review.tui_controller import ReviewDraft


@dataclass
class _Draft:
    morphology_primary: str | None
    morphology_secondaries: tuple[str, ...]
    confidence: int | None
    needs_followup: bool
    physical_primary: str | None = None


@pytest.fixture
def review_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "review.db"
    ensure_review_db_schema(db_path)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "asas_sn_id": "101",
                "lc_path": "/lightcurves/c1.dat3",
                "display_hint": "first",
                "dip_significant": True,
                "jump_significant": False,
                "periodic_flag": False,
                "vetting_likely_known": False,
                "high_ruwe_flag": False,
                "prob_dipper_dimming": 0.9,
                "prob_recurrent_given_dipper": 0.2,
                "prob_single_given_dipper": 0.8,
                "prob_recurrent_dipper_hierarchical": 0.18,
                "prob_single_dipper_hierarchical": 0.72,
                "prob_long_term_variable_hierarchical": 0.72,
                "prob_long_term_variable_given_long_timescale": 0.8,
                "predicted_hierarchy_gate": "usable_astrophysical_variable",
                "predicted_primary_morphology": "long_timescale_variable",
                "predicted_hierarchical_class": "long_timescale_variable",
                "predicted_long_timescale_subtype": "long_term_variable",
                "dipper_score": 8.0,
                "jumper_score": 1.0,
                "stats_variability_quasi_periodicity_q": 0.2,
                "stats_variability_flux_asymmetry_m": 0.3,
                "phot_g_mean_mag": 13.0,
                "periodicity_period": 2.0,
                "vsx_class": "EA",
                "gaia_var_class": "ECL",
                "asassn_var_type": "YSO",
                "simbad_otype": "Y*O",
                "yso_class": "Class II",
            },
            {
                "candidate_id": "c2",
                "asas_sn_id": "202",
                "lc_path": "/lightcurves/c2.dat3",
                "display_hint": "second",
                "dip_significant": False,
                "jump_significant": True,
                "periodic_flag": True,
                "vetting_likely_known": True,
                "high_ruwe_flag": True,
                "prob_dipper_dimming": 0.2,
                "prob_recurrent_given_dipper": 0.9,
                "prob_single_given_dipper": 0.1,
                "prob_recurrent_dipper_hierarchical": 0.18,
                "prob_single_dipper_hierarchical": 0.02,
                "prob_long_term_variable_hierarchical": 0.05,
                "prob_long_term_variable_given_long_timescale": 0.1,
                "predicted_hierarchy_gate": "artifact_or_nonvariable",
                "predicted_primary_morphology": "eb_geometric_periodic",
                "predicted_hierarchical_class": "artifact_or_nonvariable",
                "predicted_long_timescale_subtype": "not_applicable",
                "dipper_score": 1.0,
                "jumper_score": 9.0,
                "stats_variability_quasi_periodicity_q": 0.1,
                "stats_variability_flux_asymmetry_m": -0.2,
                "phot_g_mean_mag": 14.0,
                "periodicity_period": 4.0,
                "vsx_class": "RRAB",
                "gaia_var_class": "RR",
                "asassn_var_type": "RRAB",
                "simbad_otype": "EB*",
                "yso_class": "unknown",
            },
            {
                "candidate_id": "c3",
                "asas_sn_id": "303",
                "lc_path": "/lightcurves/c3.dat3",
                "display_hint": "third",
                "dip_significant": True,
                "jump_significant": True,
                "periodic_flag": False,
                "high_ruwe_flag": False,
                "prob_dipper_dimming": 0.7,
                "prob_recurrent_given_dipper": 0.6,
                "prob_single_given_dipper": 0.4,
                "prob_recurrent_dipper_hierarchical": 0.42,
                "prob_single_dipper_hierarchical": 0.28,
                "prob_long_term_variable_hierarchical": 0.2,
                "prob_long_term_variable_given_long_timescale": 0.3,
                "predicted_hierarchy_gate": "usable_astrophysical_variable",
                "predicted_primary_morphology": "dipper_dimming",
                "predicted_hierarchical_class": "dipper_dimming",
                "predicted_long_timescale_subtype": "not_applicable",
                "dipper_score": 6.0,
                "jumper_score": 5.0,
                "stats_variability_quasi_periodicity_q": 0.7,
                "stats_variability_flux_asymmetry_m": 0.6,
                "phot_g_mean_mag": 15.0,
                "periodicity_period": 8.0,
                "vsx_class": "EA",
                "gaia_var_class": "ECL",
                "asassn_var_type": "VAR",
                "simbad_otype": "Y*O",
                "yso_class": "Class I",
            },
        ]
    )
    with closing(db_connect(db_path)) as conn:
        upsert_candidates_frame(
            conn,
            candidates,
            default_source_path="/bundle/candidates.parquet",
        )
    return db_path


def _draft(
    primary: str = "dimming_event",
    secondaries: tuple[str, ...] = ("single_dip",),
    confidence: int = 3,
    needs_followup: bool = False,
    physical_primary: str | None = None,
) -> _Draft:
    return _Draft(
        primary,
        secondaries,
        confidence,
        needs_followup,
        physical_primary,
    )


def test_queue_is_a_stable_snapshot_and_load_returns_canonical_record(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db)

    assert repository.candidate_ids == ("c1", "c2", "c3")
    assert repository.start_index == 0
    record, review = repository.load("c1")
    assert isinstance(record, CandidateRecord)
    assert record.candidate_id == "c1"
    assert record.asas_sn_id == "101"
    assert record.lc_path == Path("/lightcurves/c1.dat3")
    assert record.source_path == "/bundle/candidates.parquet"
    assert record.payload["display_hint"] == "first"
    assert review["workflow_status"] == "unreviewed"

    refreshed = repository.save(
        "c1", _draft(), increment_pass=False, event_type="tui_save"
    )

    assert refreshed["disposition"] == "keep"
    assert repository.candidate_ids == ("c1", "c2", "c3")
    assert ReviewRepository(review_db).candidate_ids == ("c2", "c3")


def test_resume_uses_last_candidate_but_explicit_candidate_wins(review_db: Path) -> None:
    repository = ReviewRepository(review_db)
    repository.persist_last_candidate("c2")

    resumed = ReviewRepository(review_db)
    assert resumed.start_index == 1

    explicit = ReviewRepository(review_db, candidate_query="c3")
    assert explicit.candidate_ids == ("c1", "c2", "c3")
    assert explicit.start_index == 2


def test_save_preserves_fields_not_owned_by_the_tui(review_db: Path) -> None:
    legacy_json = json.dumps({"legacy": "classification"}, sort_keys=True)
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c1",
            event_class="legacy-dipper",
            review_pass=2,
            notes="browser note must survive",
            workflow_status="unreviewed",
            disposition="ambiguous",
            morphology_primary="dimming_event",
            morphology_secondary="single_dip",
            morphology_secondary_json=["single_dip", "broad_dip"],
            morphology_polarity="fading",
            morphology_recurrence="recurrent",
            baseline_behavior="variable",
            physical_primary="microlensing",
            physical_secondary="point_lens_candidate",
            classification_confidence=2,
            priority_tags=["urgent"],
            evidence_flags=["catalog_support"],
            model_tags=["model-a"],
            duplicate_of="c0",
            known_object_id="KNOWN-1",
            known_object_source="SIMBAD",
            legacy_review_json=legacy_json,
            reviewer="browser-user",
            event_type="seed",
        )

    repository = ReviewRepository(review_db, reviewer="tui-user")
    refreshed = repository.save(
        "c1",
        _draft(
            primary="brightening_event",
            secondaries=("single_brightening",),
            confidence=4,
            needs_followup=True,
            physical_primary="microlensing",
        ),
        increment_pass=False,
        event_type="tui_save",
    )

    assert refreshed["notes"] == "browser note must survive"
    assert refreshed["review_pass"] == 2
    assert refreshed["workflow_status"] == "needs_followup"
    assert refreshed["reviewer"] == "tui-user"
    assert refreshed["disposition"] == "ambiguous"
    assert refreshed["morphology_primary"] == "brightening_event"
    assert refreshed["morphology_secondary_list"] == ["single_brightening"]
    assert refreshed["classification_confidence"] == 4
    assert refreshed["morphology_polarity"] == "fading"
    assert refreshed["morphology_recurrence"] == "recurrent"
    assert refreshed["baseline_behavior"] == "variable"
    assert refreshed["physical_primary"] == "microlensing"
    assert refreshed["physical_secondary"] == "point_lens_candidate"
    # event_class is a legacy compatibility projection.  Physical families
    # intentionally take precedence there while the edited morphology remains
    # independently persisted.
    assert refreshed["event_class"] == "microlensing"
    assert refreshed["priority_tags"] == ["urgent"]
    assert refreshed["evidence_flags"] == ["catalog_support"]
    assert refreshed["model_tags"] == ["model-a"]
    assert refreshed["duplicate_of"] == "c0"
    assert refreshed["known_object_id"] == "KNOWN-1"
    assert refreshed["known_object_source"] == "SIMBAD"
    assert refreshed["legacy_review_json"] == legacy_json


def test_save_updates_broad_physical_label_and_clears_incompatible_subtype(
    review_db: Path,
) -> None:
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c1",
            review_pass=1,
            notes="",
            workflow_status="unreviewed",
            morphology_primary="brightening_event",
            physical_primary="microlensing",
            physical_secondary="point_lens_candidate",
            classification_confidence=3,
            event_type="seed",
        )

    refreshed = ReviewRepository(review_db).save(
        "c1",
        _draft(
            primary="brightening_event",
            secondaries=("single_brightening",),
            confidence=3,
            physical_primary="false_positive_or_contaminant",
        ),
        increment_pass=False,
        event_type="tui_save",
    )

    assert refreshed["physical_primary"] == "false_positive_or_contaminant"
    assert refreshed["physical_secondary"] is None


def test_save_persists_tui_owned_physical_subtype_and_notes(review_db: Path) -> None:
    draft = ReviewDraft(
        morphology_primary="periodic",
        morphology_secondaries=["sinusoidal"],
        physical_primary="pulsating_variable",
        physical_secondary="rr_lyrae",
        confidence=4,
        notes="RR Lyrae candidate; verify colors",
    )

    refreshed = ReviewRepository(review_db).save(
        "c1",
        draft,
        increment_pass=False,
        event_type="tui_save",
    )

    assert refreshed["physical_primary"] == "pulsating_variable"
    assert refreshed["physical_secondary"] == "rr_lyrae"
    assert refreshed["notes"] == "RR Lyrae candidate; verify colors"


def test_save_replaces_and_clears_morphology_secondaries(review_db: Path) -> None:
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c1",
            review_pass=1,
            notes="",
            workflow_status="unreviewed",
            morphology_primary="dimming_event",
            morphology_secondary="single_dip",
            morphology_secondary_json=["single_dip", "broad_dip"],
            classification_confidence=2,
            event_type="seed",
        )

    repository = ReviewRepository(review_db)
    replaced = repository.save(
        "c1",
        _draft("periodic", ("sinusoidal",), 3),
        increment_pass=False,
        event_type="tui_save",
    )
    assert replaced["morphology_primary"] == "periodic"
    assert replaced["morphology_secondary_list"] == ["sinusoidal"]

    cleared = repository.save(
        "c1",
        _draft("periodic", (), 3),
        increment_pass=False,
        event_type="tui_save",
    )
    assert cleared["morphology_secondary"] is None
    assert cleared["morphology_secondary_list"] == []
    assert cleared["morphology_secondary_json"] == "[]"


def test_save_increments_pass_only_when_requested_and_writes_history(
    review_db: Path,
) -> None:
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c1",
            review_pass=2,
            notes="",
            workflow_status="unreviewed",
            morphology_primary="dimming_event",
            classification_confidence=2,
            event_type="seed",
        )

    repository = ReviewRepository(review_db, reviewer="terminal-reviewer")
    unchanged = repository.save(
        "c1", _draft(), increment_pass=False, event_type="tui_save"
    )
    incremented = repository.save(
        "c1", _draft(), increment_pass=True, event_type="tui_done"
    )

    assert unchanged["review_pass"] == 2
    assert incremented["review_pass"] == 3
    with closing(db_connect(review_db)) as conn:
        row = conn.execute(
            """
            SELECT event_type, reviewer, payload_json
            FROM review_history
            WHERE candidate_id = 'c1'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    assert row[0] == "tui_done"
    assert row[1] == "terminal-reviewer"
    payload = json.loads(row[2])
    assert payload["review_pass"] == 3
    assert payload["workflow_status"] == "reviewed"


def test_explicit_candidate_accepts_id_or_asas_sn_id_outside_active_queue(
    review_db: Path,
) -> None:
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c2",
            review_pass=1,
            notes="",
            workflow_status="reviewed",
            morphology_primary="periodic",
            classification_confidence=4,
            event_type="seed",
        )

    by_asas_sn_id = ReviewRepository(review_db, candidate_query="202")
    assert by_asas_sn_id.candidate_ids == ("c2",)
    assert by_asas_sn_id.start_index == 0

    by_candidate_id = ReviewRepository(review_db, candidate_query="c3")
    assert by_candidate_id.candidate_ids == ("c1", "c3")
    assert by_candidate_id.start_index == 1


def test_in_session_search_preserves_matching_queue_and_uses_global_override(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)

    index, outside = repository.search_candidate("303")
    assert (index, outside) == (2, False)
    assert repository.candidate_ids == ("c1", "c2", "c3")

    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c2",
            review_pass=1,
            notes="",
            workflow_status="reviewed",
            morphology_primary="periodic",
            classification_confidence=4,
            event_type="seed",
        )
    unreviewed = ReviewRepository(review_db, restore_filter_state=False)
    index, outside = unreviewed.search_candidate("202")
    assert (index, outside) == (0, True)
    assert unreviewed.candidate_ids == ("c2",)
    assert unreviewed.search_override is True

    index, outside = unreviewed.search_candidate("c2")
    assert (index, outside) == (0, True)
    assert unreviewed.candidate_ids == ("c2",)
    assert unreviewed.search_override is True

    index, outside = unreviewed.search_candidate("c1")
    assert (index, outside) == (0, False)
    assert unreviewed.candidate_ids == ("c1", "c3")
    assert unreviewed.search_override is False


def test_failed_in_session_search_does_not_change_snapshot(review_db: Path) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)
    original = repository.candidate_ids

    with pytest.raises(ValueError, match="No candidate matches"):
        repository.search_candidate("missing")

    assert repository.candidate_ids == original
    assert repository.search_override is False


def test_filter_spec_translates_curated_controls_to_canonical_queue_filters() -> None:
    spec = QueueFilterSpec(
        queue_state="follow-up",
        signal_lane="mixed",
        show_external_lightcurves=False,
        external_lightcurve_sources=("ps1", "ztf"),
        required_external_photometry_sources=("neowise",),
        excluded_external_photometry_sources=("ps1",),
        known_objects="exclude",
        high_ruwe="only",
        high_pm="only",
        exclude_known_neighbors=True,
        exclude_dipper_contaminants=True,
        exclude_failed=True,
        neighbor_radius_arcsec=12.5,
        morphology_primary=("dimming_event", "periodic"),
        physical_primary=("microlensing",),
        excluded_vsx_types=("EA",),
        excluded_gaia_var_types=("ECL",),
        excluded_asassn_var_types=("RRAB",),
        excluded_simbad_types=("EB*",),
        excluded_yso_classes=("Class I",),
        confidence=NumericRange(2, 4),
        prob_dipper_dimming=NumericRange(0.5, None),
        prob_eb_geometric_periodic=NumericRange(None, 0.8),
        prob_long_term_variable_hierarchical=NumericRange(0.4, None),
        prob_long_term_variable_given_long_timescale=NumericRange(0.75, None),
        predicted_hierarchical_class="long_timescale_variable",
        dipper_score=NumericRange(3, 9),
        jumper_score=NumericRange(None, 5),
        q=NumericRange(0.1, 0.8),
        m=NumericRange(-0.5, 0.5),
        g_magnitude=NumericRange(None, 16),
        period_days=NumericRange(0.1, 10),
        sort_by="pdip",
        sort_desc=True,
        categorical_logic="any",
    )

    filters = spec.to_query_filters()
    assert filters["workflow_status_exact"] == "needs_followup"
    assert filters["dip_significant_mode"] == "True"
    assert filters["jump_significant_mode"] == "True"
    assert filters["vetting_likely_known_mode"] == "False"
    assert filters["high_ruwe_flag_mode"] == "True"
    assert filters["high_pm_flag_mode"] == "True"
    assert filters["exclude_known_catalog_neighbors"] is True
    assert filters["exclude_dipper_catalog_neighbors"] is True
    assert filters["catalog_neighbor_radius_arcsec"] == 12.5
    assert filters["require_failed_any_false"] is True
    assert filters["exclude_morphology_primary"] == ["dimming_event", "periodic"]
    assert filters["exclude_physical_primary"] == ["microlensing"]
    assert filters["exclude_predicted_hierarchical_class"] == [
        "long_timescale_variable"
    ]
    assert filters["catalog_type_exclusions"] == {
        "vsx_class": ["EA"],
        "gaia_var_class": ["ECL"],
        "asassn_var_type": ["RRAB"],
        "simbad_otype": ["EB*"],
        "yso_class": ["Class I"],
    }
    assert filters["select_filter_mode"] == "include"
    assert filters["select_filter_logic"] == "or"
    assert filters["min_prob_dipper_dimming"] == 0.5
    assert filters["max_prob_eb_geometric_periodic"] == 0.8
    assert filters["min_prob_long_term_variable_hierarchical"] == 0.4
    assert (
        filters["min_prob_long_term_variable_given_long_timescale"] == 0.75
    )
    assert filters["min_classification_confidence"] == 2.0
    assert filters["max_classification_confidence"] == 4.0
    assert filters["min_stats_variability_quasi_periodicity_q"] == 0.1
    assert filters["max_stats_variability_flux_asymmetry_m"] == 0.5
    assert filters["max_phot_g_mean_mag"] == 16.0
    assert filters["max_periodicity_period"] == 10.0
    assert filters["sort_cols"] == ["prob_dipper_dimming"]
    assert filters["sort_desc"] is True
    assert "show_external_lightcurves" not in filters
    assert "external_lightcurve_sources" not in filters

    round_trip = QueueFilterSpec.from_dict(spec.to_dict())
    assert round_trip == spec
    assert round_trip.show_external_lightcurves is False
    assert round_trip.external_lightcurve_sources == ("ztf", "ps1")
    assert "follow-up" in spec.summary()
    assert "VSX−1" in spec.summary()
    assert "Gaia−1" in spec.summary()
    assert "ASAS-SN−1" in spec.summary()
    assert "SIMBAD−1" in spec.summary()
    assert "YSO−1" in spec.summary()
    assert "high-PM" in spec.summary()
    assert "phot+NEOWISE W1/W2" in spec.summary()
    assert "phot−Pan-STARRS1" in spec.summary()
    assert "MLclass:long_timescale_variable" in spec.summary()
    assert "Pdip\N{GREATER-THAN OR EQUAL TO}0.5" in spec.summary()
    assert spec.summary().endswith("Pdip\N{DOWNWARDS ARROW}")


def test_external_photometry_availability_is_anded_with_ml_probability(
    review_db: Path,
) -> None:
    results_root = review_db.parent / "results"
    results_root.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "source": "neowise",
                "file_prefix": "neowise",
                "path": "/external/neowise_c1.parquet",
            },
            {
                "candidate_id": "c2",
                "source": "neowise",
                "file_prefix": "neowise",
                "path": "/external/neowise_c2.parquet",
            },
            {
                "candidate_id": "c3",
                "source": "ztf",
                "file_prefix": "ztf",
                "path": "/external/ztf_c3.parquet",
            },
        ]
    ).to_parquet(results_root / "external_lc_manifest.parquet", index=False)

    repository = ReviewRepository(review_db, restore_filter_state=False)
    assert repository.external_photometry_counts()["neowise"] == 2
    assert repository.preview_filter_count(
        QueueFilterSpec(
            queue_state="all",
            prob_dipper_dimming=NumericRange(minimum=0.5),
            required_external_photometry_sources=("neowise",),
        )
    ) == 1
    assert repository.preview_filter_count(
        QueueFilterSpec(
            queue_state="all",
            prob_dipper_dimming=NumericRange(minimum=0.5),
            excluded_external_photometry_sources=("neowise",),
        )
    ) == 1
    assert repository.preview_filter_count(
        QueueFilterSpec(
            queue_state="all",
            required_external_photometry_sources=("neowise", "ztf"),
        )
    ) == 0


def test_filter_spec_migrates_persisted_binary_dipper_state_to_hierarchical() -> None:
    spec = QueueFilterSpec.from_dict(
        {
            "prob_dipper_like": {"minimum": 0.75, "maximum": None},
            "prob_long_term_variable": {
                "minimum": 0.6,
                "maximum": None,
            },
            "sort_by": "prob_dipper_like",
            "sort_desc": True,
        }
    )

    assert spec.prob_dipper_dimming == NumericRange(0.75, None)
    assert spec.prob_long_term_variable_hierarchical == NumericRange(0.6, None)
    assert spec.sort_by == "prob_dipper_dimming"
    assert spec.to_query_filters()["min_prob_dipper_dimming"] == 0.75


def test_filter_spec_normalizes_external_sources_and_keeps_asassn_as_base() -> None:
    spec = QueueFilterSpec(
        external_lightcurve_sources=(
            "ps1",
            "asassn",
            "unknown",
            "ztf",
            "ps1",
        )
    )

    assert spec.external_lightcurve_sources == ("ztf", "ps1")
    assert "asassn" not in spec.external_lightcurve_sources
    assert QueueFilterSpec.from_dict(spec.to_dict()) == spec


def test_hierarchical_global_conditional_and_predicted_filters_query_live_db(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)
    spec = QueueFilterSpec(
        queue_state="all",
        prob_long_term_variable_hierarchical=NumericRange(0.5, None),
        prob_long_term_variable_given_long_timescale=NumericRange(0.75, None),
        predicted_hierarchical_class="long_timescale_variable",
        sort_by="prob_long_term_variable_given_long_timescale",
        sort_desc=True,
    )

    assert repository.preview_filter_count(spec) == 1
    repository.apply_filters(spec)
    assert repository.candidate_ids == ("c1",)

    single_spec = QueueFilterSpec(
        queue_state="all",
        prob_single_dipper_hierarchical=NumericRange(0.5, None),
        prob_single_given_dipper=NumericRange(0.75, None),
        sort_by="prob_single_dipper_hierarchical",
        sort_desc=True,
    )
    assert repository.preview_filter_count(single_spec) == 1
    repository.apply_filters(single_spec)
    assert repository.candidate_ids == ("c1",)


def test_high_pm_filter_defaults_to_exclude_and_keeps_missing_measurements(
    review_db: Path,
) -> None:
    with closing(db_connect(review_db)) as conn:
        conn.execute(
            "UPDATE candidates SET high_pm_flag = CASE candidate_id "
            "WHEN 'c1' THEN 0 WHEN 'c2' THEN 1 ELSE NULL END"
        )
        conn.commit()

    repository = ReviewRepository(review_db, restore_filter_state=False)
    default_spec = QueueFilterSpec(queue_state="all")

    assert default_spec.high_pm == "exclude"
    assert default_spec.to_query_filters()["high_pm_flag_mode"] == "False"
    assert repository.preview_filter_count(default_spec) == 2
    assert repository.preview_filter_count(
        QueueFilterSpec(queue_state="all", high_pm="only")
    ) == 1
    assert repository.preview_filter_count(
        QueueFilterSpec(queue_state="all", high_pm="any")
    ) == 3


def test_catalog_type_inventory_is_campaign_local_and_descriptive(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)

    stats = repository.catalog_type_stats()
    by_key = {(stat.catalog, stat.value): stat for stat in stats}

    assert by_key[("vsx", "EA")].count == 2
    assert by_key[("vsx", "EA")].total_candidates == 3
    assert by_key[("vsx", "EA")].fraction == pytest.approx(2 / 3)
    assert by_key[("vsx", "EA")].known_variable is True
    assert by_key[("vsx", "EA")].dipper_contaminant is True
    assert by_key[("vsx", "EA")].description
    assert by_key[("gaia", "ECL")].count == 2
    assert by_key[("asassn", "YSO")].count == 1
    assert by_key[("simbad", "Y*O")].count == 2
    assert by_key[("simbad", "Y*O")].description
    assert by_key[("yso", "Class II")].count == 1
    assert by_key[("yso", "unknown")].count == 1


def test_catalog_type_exclusions_filter_and_persist_per_review_db(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)
    spec = QueueFilterSpec(
        queue_state="all",
        excluded_simbad_types=("Y*O",),
    )

    assert repository.preview_filter_count(spec) == 1
    repository.apply_filters(spec)

    assert repository.candidate_ids == ("c2",)
    assert repository.load_filter_spec() == spec
    assert ReviewRepository(review_db).filter_spec == spec


def test_queue_state_remains_anded_when_taxonomy_filters_use_any_logic(
    review_db: Path,
) -> None:
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c2",
            review_pass=1,
            notes="",
            workflow_status="needs_followup",
            morphology_primary="periodic",
            classification_confidence=3,
            event_type="seed",
        )
        save_review(
            conn,
            candidate_id="c3",
            review_pass=1,
            notes="",
            workflow_status="reviewed",
            morphology_primary="dimming_event",
            physical_primary="microlensing",
            classification_confidence=3,
            event_type="seed",
        )

    repository = ReviewRepository(review_db, restore_filter_state=False)
    repository.apply_filters(
        QueueFilterSpec(
            queue_state="reviewed",
            morphology_primary=("periodic",),
            physical_primary=("microlensing",),
            categorical_logic="any",
        )
    )

    assert repository.candidate_ids == ("c3",)


def test_apply_filters_creates_stable_snapshot_and_retains_matching_anchor(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)
    spec = QueueFilterSpec(
        queue_state="all",
        prob_dipper_dimming=NumericRange(0.5, None),
        sort_by="prob_dipper_dimming",
        sort_desc=True,
    )

    assert repository.preview_filter_count(spec) == 2
    index = repository.apply_filters(spec, anchor_candidate_id="c3")

    assert repository.candidate_ids == ("c1", "c3")
    assert index == 1
    assert repository.start_index == 1
    assert repository.filter_spec == spec

    repository.save(
        "c3", _draft(), increment_pass=False, event_type="tui_save"
    )
    # Saving never re-runs the active query underneath the reviewer.
    assert repository.candidate_ids == ("c1", "c3")


def test_empty_filter_result_keeps_snapshot_and_persisted_filter(
    review_db: Path,
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)
    active = QueueFilterSpec(
        queue_state="all", signal_lane="periodic", sort_by="period_days"
    )
    repository.apply_filters(active)
    assert repository.candidate_ids == ("c2",)
    persisted_before = repository.load_filter_spec()

    with pytest.raises(EmptyQueueError, match="current queue was kept"):
        repository.apply_filters(
            QueueFilterSpec(
                queue_state="all",
                prob_dipper_dimming=NumericRange(2.0, None),
            )
        )

    assert repository.candidate_ids == ("c2",)
    assert repository.start_index == 0
    assert repository.filter_spec == active
    assert repository.load_filter_spec() == persisted_before == active


def test_filter_query_error_keeps_existing_snapshot(
    review_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = ReviewRepository(review_db, restore_filter_state=False)
    previous_ids = repository.candidate_ids
    previous_spec = repository.filter_spec

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated query failure")

    monkeypatch.setattr("malca.review.tui_service.query_queue", _explode)
    with pytest.raises(RuntimeError, match="simulated query failure"):
        repository.apply_filters(QueueFilterSpec(queue_state="all"))

    assert repository.candidate_ids == previous_ids
    assert repository.filter_spec == previous_spec


def test_tui_filter_state_persists_separately_and_restores(review_db: Path) -> None:
    browser_key = "dash_queue_filter_ui_state_v1"
    browser_value = '{"browser":"untouched"}'
    with closing(db_connect(review_db)) as conn:
        save_app_state(conn, browser_key, browser_value)

    repository = ReviewRepository(review_db, restore_filter_state=False)
    spec = QueueFilterSpec(
        queue_state="all",
        signal_lane="periodic",
        show_external_lightcurves=False,
        external_lightcurve_sources=("gaia_epoch", "ps1"),
        sort_by="period_days",
        sort_desc=True,
    )
    repository.apply_filters(spec)

    restored = ReviewRepository(review_db)
    assert restored.filter_spec == spec
    assert restored.candidate_ids == ("c2",)
    with closing(db_connect(review_db)) as conn:
        assert load_app_state(conn, browser_key) == browser_value
        assert load_app_state(conn, TUI_FILTER_STATE_KEY) != browser_value


def test_review_taxonomy_and_confidence_filters_use_canonical_review_columns(
    review_db: Path,
) -> None:
    with closing(db_connect(review_db)) as conn:
        save_review(
            conn,
            candidate_id="c2",
            review_pass=1,
            notes="",
            workflow_status="reviewed",
            morphology_primary="periodic",
            physical_primary="eclipsing_or_geometric_binary",
            classification_confidence=4,
            event_type="seed",
        )

    repository = ReviewRepository(review_db, restore_filter_state=False)
    spec = QueueFilterSpec(
        queue_state="reviewed",
        morphology_primary=("periodic",),
        physical_primary=("eclipsing_or_geometric_binary",),
        confidence=NumericRange(3, 4),
    )
    repository.apply_filters(spec)

    assert repository.candidate_ids == ("c2",)
