from __future__ import annotations

import json

import pandas as pd
import pytest

from malca.review.paper_candidates import (
    PUBLICATION_COHORT_VERSION,
    PAPER_FEATURE_GROUPS,
    assign_review_bucket,
    audit_flattened_vs_layers,
    build_publication_cohort,
    feature_missingness_by_bucket,
    flatten_layer_payload,
    mwu_separability_table,
    paper_feature_columns,
)


def test_paper_feature_columns_include_expected_prefixes():
    cols = paper_feature_columns()
    assert any(col.startswith("stats_") for col in cols)
    assert any(col.startswith("ms_") for col in cols)
    assert any(col.startswith("ltv_ms_") for col in cols)
    assert any(col.startswith("camera_name_") for col in cols)
    assert "asassn_field_key" in cols


def test_paper_feature_groups_has_stats_and_multisurvey_sections():
    assert "LC Photometric Scatter" in PAPER_FEATURE_GROUPS
    assert "Multi-Survey ZTF" in PAPER_FEATURE_GROUPS
    assert "LTV Multi-Survey" in PAPER_FEATURE_GROUPS
    assert "Camera / Field" in PAPER_FEATURE_GROUPS
    assert PAPER_FEATURE_GROUPS["LC Photometric Scatter"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"event_class": "dipper"}, "Dipper"),
        ({"event_class": "ltv"}, "LTV"),
        ({"event_class": "microlensing"}, "Microlensing"),
        (
            {"event_class": "periodic", "morphology_secondary": "detached_binary_like"},
            "Eclipsing binary",
        ),
        ({"event_class": "other", "physical_primary": "unknown"}, "Unknown"),
    ],
)
def test_assign_review_bucket(row, expected):
    assert assign_review_bucket(row) == expected


def test_generic_periodic_review_is_not_assumed_to_be_an_eclipsing_binary():
    assert assign_review_bucket({"event_class": "periodic"}) is None


def test_flatten_layer_payload_fills_missing_flat_columns():
    payload = {
        "candidate_id": "1",
        "stats_amplitude": None,
        "lc_stats": json.dumps({"stats_amplitude": 0.12, "dipper_score": 3.4}),
        "external_stats": json.dumps({"ms_asassn_delta_g": -0.02}),
    }
    flat = flatten_layer_payload(payload)
    assert flat["stats_amplitude"] == 0.12
    assert flat["dipper_score"] == 3.4
    assert flat["ms_asassn_delta_g"] == -0.02


def test_audit_flattened_vs_layers_reports_json_only_keys():
    df = pd.DataFrame(
        [
            {
                "candidate_id": "123",
                "stats_amplitude": 0.1,
                "lc_stats": json.dumps({"stats_amplitude": 0.1, "dip_best_log_bf": 4.2}),
                "external_stats": "{}",
                "derived_stats": "{}",
            }
        ]
    )
    audit = audit_flattened_vs_layers(df)
    assert not audit.empty
    assert set(audit["issue"]) <= {"json_only", "flat_missing"}
    assert (audit["key"] == "dip_best_log_bf").any()


def test_audit_flattened_vs_layers_reports_conflicting_values():
    df = pd.DataFrame(
        [
            {
                "candidate_id": "123",
                "stats_amplitude": 0.5,
                "lc_stats": json.dumps({"stats_amplitude": 0.1}),
                "external_stats": "{}",
                "derived_stats": "{}",
            }
        ]
    )

    audit = audit_flattened_vs_layers(df)

    assert audit.loc[0, "issue"] == "value_mismatch"


def test_audit_flattened_vs_layers_handles_duplicate_dataframe_index():
    df = pd.DataFrame(
        [
            {"candidate_id": "one", "stats_amplitude": 0.1, "lc_stats": {"stats_amplitude": 0.1}},
            {"candidate_id": "two", "stats_amplitude": 0.2, "lc_stats": {"stats_amplitude": 0.4}},
        ],
        index=[5, 5],
    )

    audit = audit_flattened_vs_layers(df)

    assert audit[["candidate_id", "issue"]].to_dict("records") == [
        {"candidate_id": "two", "issue": "value_mismatch"}
    ]


def test_feature_missingness_does_not_count_object_nulls_as_strings():
    df = pd.DataFrame(
        {
            "review_bucket": ["Dipper", "Dipper", "Dipper"],
            "feature": [None, "", "value"],
        }
    )

    result = feature_missingness_by_bucket(df, groups={"group": ["feature"]})

    assert result.loc[0, "non_null"] == 1


def test_publication_cohort_handles_duplicate_dataframe_index_without_series_truth_errors():
    frame = pd.DataFrame(
        [
            {"candidate_id": "one", "event_class": "dipper", "workflow_status": "reviewed", "disposition": "keep"},
            {"candidate_id": "two", "event_class": "ltv", "workflow_status": "reviewed", "disposition": "keep"},
        ],
        index=[7, 7],
    )

    cohort = build_publication_cohort(frame)

    assert cohort["publication_selected"].tolist() == [True, True]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("dipper", "Dipper"),
        ("Dipper", "Dipper"),
        ("ltv", "LTV"),
    ],
)
def test_resolve_bucket_label(label, expected):
    df = pd.DataFrame(
        {
            "event_class": ["dipper", "ltv"],
            "review_bucket": ["Dipper", "LTV"],
        }
    )
    from malca.review.paper_candidates import resolve_bucket_label

    assert resolve_bucket_label(df, label) == expected


def test_publication_cohort_records_every_exclusion_reason():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "keep",
                "event_class": "dipper",
                "workflow_status": "reviewed",
                "disposition": "keep",
                "classification_confidence": 4,
            },
            {
                "candidate_id": "followup",
                "event_class": "dipper",
                "workflow_status": "needs_followup",
                "disposition": "ambiguous",
                "classification_confidence": 2,
            },
            {
                "candidate_id": "duplicate",
                "event_class": "ltv",
                "workflow_status": "reviewed",
                "disposition": "keep",
                "duplicate_of": "keep",
                "classification_confidence": 3,
            },
        ]
    )

    cohort = build_publication_cohort(frame, require_confident_classification=True)

    assert cohort["publication_cohort_version"].eq(PUBLICATION_COHORT_VERSION).all()
    assert cohort.loc[cohort["candidate_id"].eq("keep"), "publication_selected"].item()
    followup_reason = cohort.loc[
        cohort["candidate_id"].eq("followup"), "publication_exclusion_reason"
    ].item()
    assert "workflow_not_reviewed" in followup_reason
    assert "disposition_not_keep" in followup_reason
    assert "classification_confidence_below_3" in followup_reason
    assert "marked_duplicate" in cohort.loc[
        cohort["candidate_id"].eq("duplicate"), "publication_exclusion_reason"
    ].item()


def test_mwu_table_reports_effect_size_sample_size_and_fdr():
    frame = pd.DataFrame(
        {
            "review_bucket": ["Dipper"] * 4 + ["LTV"] * 4,
            "feature_a": [5, 6, 7, 8, 1, 2, 3, 4],
            "feature_b": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )

    result = mwu_separability_table(
        frame,
        ["feature_a", "feature_b"],
        reference_group="Dipper",
        compare_groups=["LTV"],
    )

    assert set(["reference_n", "compare_n", "rank_biserial", "mwu_fdr_qvalue"]) <= set(result)
    assert result["reference_n"].eq(4).all()
    assert result["compare_n"].eq(4).all()
    assert result["mwu_fdr_qvalue"].ge(result["mwu_pvalue"] - 1e-12).all()


@pytest.mark.slow
def test_load_reviewed_cohort_smoke():
    from malca.review.paper_candidates import REVIEW_DB, load_reviewed_cohort

    if not REVIEW_DB.exists():
        pytest.skip("review DB not present")
    df = load_reviewed_cohort(buckets=["Dipper", "LTV", "Microlensing"])
    assert not df.empty
    assert "review_bucket" in df.columns
    assert set(df["review_bucket"].dropna().unique()).issubset({"Dipper", "LTV", "Microlensing"})
