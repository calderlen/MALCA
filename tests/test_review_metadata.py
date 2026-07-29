from __future__ import annotations

from malca.review.metadata import (
    REVIEW_METADATA_FIELDS,
    bracket_unit_label,
    extract_review_metadata_grouped,
    extract_review_metadata_feature_rows,
    has_catalog_vetting_context,
    markdown_literal_unit_label,
)


def test_unit_labels_use_visible_literal_brackets() -> None:
    assert bracket_unit_label("Period (d)") == "Period [d]"
    assert bracket_unit_label("Amplitude (mag)") == "Amplitude [mag]"


def test_markdown_unit_labels_escape_brackets_without_changing_visible_text() -> None:
    assert markdown_literal_unit_label("Period (d)") == r"Period \[d\]"
    assert markdown_literal_unit_label("Amplitude (mag)") == r"Amplitude \[mag\]"


def test_feature_rows_preserve_all_metadata_keys() -> None:
    payload = {key: 1 for _label, key in REVIEW_METADATA_FIELDS}

    rows = extract_review_metadata_feature_rows(payload)
    returned_keys = {row["key"] for row in rows}

    assert returned_keys == {key for _label, key in REVIEW_METADATA_FIELDS}


def test_feature_rows_keep_duplicate_labels_disambiguated_by_section_and_key() -> None:
    rows = extract_review_metadata_feature_rows({
        "dip_significant": True,
        "jump_significant": False,
    })

    significant_rows = [row for row in rows if row["label"].lower() == "significant"]

    assert {row["key"] for row in significant_rows} == {"dip_significant", "jump_significant"}
    assert {row["section"] for row in significant_rows} == {"Dip Evidence", "Jump Evidence"}


def test_feature_rows_respect_rounding_flag() -> None:
    rounded = extract_review_metadata_feature_rows({"dipper_score": 18.61234}, round_sigfigs=True)
    raw = extract_review_metadata_feature_rows({"dipper_score": 18.61234}, round_sigfigs=False)

    assert rounded[0]["value"] == "18.6"
    assert raw[0]["value"] == "18.61234"


def test_feature_rows_omit_empty_values() -> None:
    rows = extract_review_metadata_feature_rows({
        "asas_sn_id": "",
        "gaia_id": None,
        "dipper_score": 0,
    })

    assert {row["key"] for row in rows} == {"dipper_score"}


def test_browser_metadata_exposes_hierarchical_rejection_score() -> None:
    grouped = dict(
        extract_review_metadata_grouped(
            {
                "prob_hierarchical_artifact_or_nonvariable": 0.8,
            }
        )
    )

    assert grouped["Event Scoring"] == [
        ("ML P(reject)", "0.8"),
    ]


def test_browser_metadata_exposes_both_dipper_recurrence_probabilities() -> None:
    grouped = dict(
        extract_review_metadata_grouped(
            {
                "prob_recurrent_given_dipper": 0.3,
                "prob_single_given_dipper": 0.7,
                "prob_recurrent_dipper_hierarchical": 0.15,
                "prob_single_dipper_hierarchical": 0.35,
            }
        )
    )

    assert grouped["Dip Recurrence"] == [
        ("ML P(recurrent | dipper)", "0.3"),
        ("ML P(single | dipper)", "0.7"),
        ("ML P(recurrent dipper), hierarchy", "0.15"),
        ("ML P(single dipper), hierarchy", "0.35"),
    ]


def test_sfr_evidence_counts_as_completed_vetting_context() -> None:
    assert has_catalog_vetting_context(
        {
            "sfr_membership_class": "environmental_candidate",
            "sfr_environment_consistent": True,
            "banyan_sfr_prob": 0.2,
        }
    )
