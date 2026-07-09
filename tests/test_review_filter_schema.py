from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.products.feature_layers import ALL_FEATURE_LAYER_COLUMNS
from malca.review.filter_schema import (
    REVIEW_FILTER_COLUMN_TYPES,
    REVIEW_TAXONOMY_FILTER_COLUMNS,
    SIDEBAR_GROUPS,
    is_dipper_contaminant_type_value,
    is_definite_known_type_value,
    is_known_variable_type_value,
)
from malca.review.store import (
    _CANDIDATE_COLUMNS,
    db_connect,
    get_distinct_values,
    get_numeric_bounds,
    query_queue,
    save_review,
    upsert_candidates_frame,
)


def test_sidebar_schema_is_complete_and_explicit_for_db_columns() -> None:
    sidebar_cols = [col for _name, items in SIDEBAR_GROUPS for _kind, col in items]
    candidate_col_types = {col: extract_type for col, _sql_type, extract_type in _CANDIDATE_COLUMNS}
    candidate_cols = set(candidate_col_types) - set(ALL_FEATURE_LAYER_COLUMNS)
    allowed_cols = candidate_cols | set(REVIEW_FILTER_COLUMN_TYPES)
    filter_to_store_type = {"bool": "bool", "num": "float", "text": "text", "select": "select"}

    assert len(sidebar_cols) == len(set(sidebar_cols))
    assert candidate_cols.issubset(set(sidebar_cols))
    assert set(sidebar_cols).issubset(allowed_cols)
    assert [
        (kind, col, candidate_col_types[col])
        for _name, items in SIDEBAR_GROUPS
        for kind, col in items
        if col in candidate_col_types and filter_to_store_type[kind] != candidate_col_types[col]
    ] == []


def test_sidebar_schema_includes_review_taxonomy_filters() -> None:
    groups = {name: items for name, items in SIDEBAR_GROUPS}

    assert groups["Review Taxonomy"] == list(REVIEW_TAXONOMY_FILTER_COLUMNS)
    review_cols = {col for _kind, col in groups["Review Taxonomy"]}
    assert {
        "workflow_status",
        "interest_score",
        "review_pass",
        "disposition",
        "morphology_primary",
        "morphology_secondary",
        "morphology_polarity",
        "morphology_recurrence",
        "baseline_behavior",
        "physical_primary",
        "physical_secondary",
        "classification_confidence",
        "duplicate_of",
        "known_object_id",
        "known_object_source",
        "taxonomy_version",
    }.issubset(review_cols)


def test_queue_filters_review_taxonomy_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "C1"},
                    {"candidate_id": "C2"},
                    {"candidate_id": "C3"},
                ]
            ),
        )
        save_review(
            conn,
            candidate_id="C1",
            interest_score=4,
            review_pass=2,
            notes="one",
            workflow_status="reviewed",
            disposition="keep",
            morphology_primary="dimming_event",
            physical_primary="young_stellar_object_or_pms",
            classification_confidence="secure",
            known_object_id="VSX J0001",
            known_object_source="VSX",
            reviewer="alice",
        )
        save_review(
            conn,
            candidate_id="C2",
            interest_score=2,
            review_pass=1,
            notes="two",
            workflow_status="needs_followup",
            disposition="ambiguous",
            morphology_primary="brightening_event",
            physical_primary="microlensing",
            classification_confidence="possible",
            reviewer="bob",
        )

        assert get_distinct_values(conn, "workflow_status") == [
            "needs_followup",
            "reviewed",
            "unreviewed",
        ]
        assert get_distinct_values(conn, "morphology_primary") == [
            "brightening_event",
            "dimming_event",
        ]
        assert get_distinct_values(conn, "known_object_id") == ["VSX J0001"]

        bounds = get_numeric_bounds(
            conn,
            columns=["interest_score", "review_pass", "taxonomy_version"],
        )
        assert bounds["interest_score"] == {"min": 2.0, "max": 4.0}
        assert bounds["review_pass"] == {"min": 1.0, "max": 2.0}
        assert bounds["taxonomy_version"] == {"min": 1.0, "max": 1.0}

        reviewed_ids = query_queue(
            conn,
            filters={
                "select_filter_mode": "include",
                "exclude_morphology_primary": ["dimming_event"],
            },
            ids_only=True,
        )["candidate_id"].tolist()
        assert reviewed_ids == ["C1"]

        unreviewed_ids = query_queue(
            conn,
            filters={"select_filter_mode": "include", "exclude_workflow_status": ["unreviewed"]},
            ids_only=True,
        )["candidate_id"].tolist()
        assert unreviewed_ids == ["C3"]

        reviewer_ids = query_queue(conn, filters={"reviewer": "bob"}, ids_only=True)[
            "candidate_id"
        ].tolist()
        assert reviewer_ids == ["C2"]

        score_ids = query_queue(conn, filters={"min_interest_score": 3}, ids_only=True)[
            "candidate_id"
        ].tolist()
        assert score_ids == ["C1"]


def test_microlens_false_filter_keeps_unset_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "C1"},
                    {"candidate_id": "C2", "microlens_match": False},
                    {"candidate_id": "C3", "microlens_match": True},
                ]
            ),
        )

        ids = query_queue(
            conn,
            filters={"microlens_match_mode": "False"},
            ids_only=True,
        )["candidate_id"].tolist()

    assert ids == ["C1", "C2"]


def test_vetting_likely_known_false_filter_keeps_unset_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "C1"},
                    {"candidate_id": "C2", "vetting_likely_known": False},
                    {"candidate_id": "C3", "vetting_likely_known": True},
                ]
            ),
        )

        ids = query_queue(
            conn,
            filters={"vetting_likely_known_mode": "False"},
            ids_only=True,
        )["candidate_id"].tolist()

    assert ids == ["C1", "C2"]


def test_vsx_known_type_detection_tokenizes_certain_composites() -> None:
    assert is_definite_known_type_value("vsx_class", "GCAS") is True
    assert is_definite_known_type_value("vsx_class", "BE") is True
    assert is_definite_known_type_value("vsx_class", "EA") is True
    assert is_definite_known_type_value("vsx_class", "DSCT") is True
    assert is_definite_known_type_value("vsx_class", "ACEP|CEP") is True
    assert is_definite_known_type_value("vsx_class", "BE|GCAS|SDOR|WR") is True
    assert is_definite_known_type_value("vsx_class", "EA:") is False
    assert is_definite_known_type_value("vsx_class", "EA?") is False
    assert is_definite_known_type_value("vsx_class", "EA/SD:") is False
    assert is_definite_known_type_value("vsx_class", "BE:") is False
    assert is_definite_known_type_value("vsx_class", "BY+Microlens:") is False
    assert is_definite_known_type_value("vsx_class", "DSCT:+VAR") is False
    assert is_definite_known_type_value("vsx_class", "VAR") is False
    assert is_definite_known_type_value("vsx_class", "MISC") is False
    assert is_definite_known_type_value("vsx_class", "None") is False
    assert is_definite_known_type_value("vsx_class", None) is False
    assert is_definite_known_type_value("vsx_class", "nan") is False
    assert is_definite_known_type_value("vsx_class", "*") is False


def test_asassn_uncertainty_and_generic_classes_stay_visible() -> None:
    assert is_definite_known_type_value("asassn_var_type", "ROT") is True
    assert is_definite_known_type_value("asassn_var_type", "ROT:") is False
    assert is_definite_known_type_value("asassn_var_type", "GCAS:") is False
    assert is_definite_known_type_value("asassn_var_type", "VAR") is False


def test_simbad_known_variable_policy_excludes_only_definite_variables() -> None:
    for code in ["EB*", "RR*", "V*", "LP*", "Mi*", "Or*"]:
        assert is_known_variable_type_value("simbad_otype", code) is True
        assert is_definite_known_type_value("simbad_otype", code) is True

    for code in ["Y*O", "TT*", "Be*", "Em*", "IR", "Rad", "X", "FIR", "*", "**"]:
        assert is_known_variable_type_value("simbad_otype", code) is False
        assert is_definite_known_type_value("simbad_otype", code) is False


def test_simbad_question_mark_types_stay_visible() -> None:
    for code in ["EB?", "RR?", "LP?", "Y*?", "AB?", "Be?", "HS?", "S*?", "s?r"]:
        assert is_known_variable_type_value("simbad_otype", code) is False
        assert is_dipper_contaminant_type_value("simbad_otype", code) is False


def test_simbad_dipper_contaminant_policy_is_target_safe() -> None:
    for code in ["EB*", "SB*", "Y*O", "TT*", "Be*", "Ae*", "Or*", "RC*", "CV*", "No*", "HXB", "ev", "Em*"]:
        assert is_dipper_contaminant_type_value("simbad_otype", code) is True

    for code in [
        "Y*?",
        "TT?",
        "LP*",
        "Mi*",
        "RR*",
        "AB*",
        "C*",
        "OH*",
        "pA*",
        "RG*",
        "Be?",
        "IR",
        "Rad",
        "X",
        "FIR",
        "*",
        "**",
    ]:
        assert is_dipper_contaminant_type_value("simbad_otype", code) is False


def test_tns_definite_types_and_candidate_text() -> None:
    assert is_definite_known_type_value("tns_type", "SN Ia") is True
    assert is_definite_known_type_value("tns_type", "SN II") is True
    assert is_definite_known_type_value("tns_type", "Nova") is True
    assert is_definite_known_type_value("tns_type", "CV") is True
    assert is_definite_known_type_value("tns_type", "TDE") is True
    assert is_definite_known_type_value("tns_type", "CV candidate") is False
    assert is_definite_known_type_value("tns_type", "SN?") is False
    assert is_definite_known_type_value("tns_type", "Fading blue star") is False
    assert is_definite_known_type_value("tns_type", "star with a deep dimming event") is False


def test_dipper_contaminant_policy_is_source_specific_and_uncertainty_safe() -> None:
    assert is_dipper_contaminant_type_value("vsx_class", "EA") is True
    assert is_dipper_contaminant_type_value("vsx_class", "EA/SD") is True
    assert is_dipper_contaminant_type_value("vsx_class", "BE|GCAS|SDOR|WR") is True
    assert is_dipper_contaminant_type_value("vsx_class", "UXOR") is True
    assert is_dipper_contaminant_type_value("vsx_class", "YSO/DIP") is True
    assert is_dipper_contaminant_type_value("vsx_class", "DSCT") is False
    assert is_dipper_contaminant_type_value("vsx_class", "ACEP|CEP") is False
    assert is_dipper_contaminant_type_value("vsx_class", "EA:") is False
    assert is_dipper_contaminant_type_value("vsx_class", "UXOR:") is False

    assert is_dipper_contaminant_type_value("asassn_var_type", "EA") is True
    assert is_dipper_contaminant_type_value("asassn_var_type", "YSO") is True
    assert is_dipper_contaminant_type_value("asassn_var_type", "GCAS") is True
    assert is_dipper_contaminant_type_value("asassn_var_type", "RRAB") is False
    assert is_dipper_contaminant_type_value("asassn_var_type", "GCAS:") is False

    assert is_dipper_contaminant_type_value("gaia_var_class", "ECL") is True
    assert is_dipper_contaminant_type_value("gaia_var_class", "YSO") is True
    assert is_dipper_contaminant_type_value("gaia_var_class", "BE|GCAS|SDOR|WR") is True
    assert is_dipper_contaminant_type_value("gaia_var_class", "RR") is False
    assert is_dipper_contaminant_type_value("gaia_var_class", "DSCT|GDOR|SXPHE") is False

    assert is_dipper_contaminant_type_value("ztf_var_type", "EA") is True
    assert is_dipper_contaminant_type_value("ztf_var_type", "EW") is True
    assert is_dipper_contaminant_type_value("ztf_var_type", "RSCVN") is False

    assert is_dipper_contaminant_type_value("alerce_lc_class", "YSO") is True
    assert is_dipper_contaminant_type_value("alerce_lc_class", "EA") is True
    assert is_dipper_contaminant_type_value("alerce_lc_class", "CV/Nova") is True
    assert is_dipper_contaminant_type_value("alerce_lc_class", "Periodic") is False

    assert is_dipper_contaminant_type_value("tns_type", "CV") is True
    assert is_dipper_contaminant_type_value("tns_type", "Nova") is True
    assert is_dipper_contaminant_type_value("tns_type", "SN Ia") is False
    assert is_dipper_contaminant_type_value("tns_type", "CV candidate") is False

    assert is_dipper_contaminant_type_value("yso_class", "Class II") is True
    assert is_dipper_contaminant_type_value("yso_class", "T Tauri") is True
    assert is_dipper_contaminant_type_value("yso_class", "Main Sequence") is False
    assert is_dipper_contaminant_type_value("yso_class", "unknown") is False
    assert is_dipper_contaminant_type_value("yso_class", "YSO?") is False


def test_classifier_sources_are_definite_when_non_empty() -> None:
    assert is_definite_known_type_value("gaia_var_class", "ECL") is True
    assert is_definite_known_type_value("ztf_var_type", "EA") is True
    assert is_definite_known_type_value("alerce_lc_class", "Periodic") is True
    assert is_definite_known_type_value("gaia_var_class", "") is False
