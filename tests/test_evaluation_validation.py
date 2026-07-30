from __future__ import annotations

import pandas as pd
import pytest

from malca.evaluation.validation import validate_detections


def test_validation_does_not_call_unlabelled_results_false_positives() -> None:
    results = pd.DataFrame(
        {
            "source_id": ["positive", "negative", "outside"],
            "dip_significant": [True, True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "source_id": ["positive", "missed", "negative"],
            "expected_detected": [True, True, False],
        }
    )

    metrics = validate_detections(results, labels)

    assert metrics["true_positives"] == ["positive"]
    assert metrics["false_negatives"] == ["missed"]
    assert metrics["false_positives"] == ["negative"]
    assert metrics["unlabeled_detections"] == ["outside"]
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall_ci95_low"] < metrics["recall"] < metrics["recall_ci95_high"]


def test_positive_only_reference_does_not_claim_precision() -> None:
    results = pd.DataFrame({"source_id": ["p", "other"], "dip_significant": [True, True]})
    positives = pd.DataFrame({"source_id": ["p"]})

    metrics = validate_detections(results, positives)

    assert metrics["recall"] == 1.0
    assert metrics["precision"] is None
    assert metrics["label_scope"] == "positive_reference_only"


def test_validation_rejects_malformed_significance_and_duplicate_labels() -> None:
    labels = pd.DataFrame({"source_id": ["p"], "expected_detected": [True]})
    with pytest.raises(ValueError, match="null/invalid"):
        validate_detections(
            pd.DataFrame({"source_id": ["p"], "dip_significant": ["perhaps"]}),
            labels,
        )

    duplicate_labels = pd.concat([labels, labels], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_detections(
            pd.DataFrame({"source_id": ["p"], "dip_significant": [True]}),
            duplicate_labels,
        )


def test_coordinate_matching_is_explicit_and_uses_tolerance() -> None:
    results = pd.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [5.0, 0.0],
            "dip_significant": [True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "source_id": ["near", "far"],
            "ra": [10.0 + 1.0 / 3600.0, 30.0],
            "dec": [5.0, 0.0],
            "expected_detected": [True, True],
        }
    )

    metrics = validate_detections(
        results,
        labels,
        match_mode="coordinates",
        match_tolerance_arcsec=2.0,
    )

    assert metrics["true_positives"] == ["near"]
    assert metrics["false_negatives"] == ["far"]
    assert metrics["n_unlabeled_detections"] == 1

