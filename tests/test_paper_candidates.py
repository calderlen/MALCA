from __future__ import annotations

import json

import pandas as pd
import pytest

from malca.review.paper_candidates import (
    PAPER_FEATURE_GROUPS,
    assign_review_bucket,
    audit_flattened_vs_layers,
    flatten_layer_payload,
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
        ({"event_class": "unknown_interesting"}, "Interesting"),
        (
            {"event_class": "periodic", "morphology_secondary": "detached_binary_like"},
            "Eclipsing binary",
        ),
        ({"event_class": "other", "physical_primary": "unknown"}, "Unknown"),
    ],
)
def test_assign_review_bucket(row, expected):
    assert assign_review_bucket(row) == expected


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


@pytest.mark.slow
def test_load_reviewed_cohort_smoke():
    from malca.review.paper_candidates import REVIEW_DB, load_reviewed_cohort

    if not REVIEW_DB.exists():
        pytest.skip("review DB not present")
    df = load_reviewed_cohort(buckets=["Dipper", "LTV", "Microlensing"])
    assert not df.empty
    assert "review_bucket" in df.columns
    assert set(df["review_bucket"].dropna().unique()).issubset({"Dipper", "LTV", "Microlensing"})
