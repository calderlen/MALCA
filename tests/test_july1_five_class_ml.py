from __future__ import annotations

import pandas as pd

from malca.meta_analysis.ml.july1_five_class_training import (
    FIVE_CLASS_ORDER,
    TARGET_COLUMN,
    construct_five_class_target,
    five_class_training_config,
)


def _row(
    candidate_id: str,
    *,
    event_class: str,
    primary: str,
    tags: list[str] | None = None,
    physical_primary: str = "",
) -> dict[str, object]:
    tags = tags or []
    return {
        "candidate_id": candidate_id,
        "status": "reviewed",
        "workflow_status": "reviewed",
        "event_class": event_class,
        "morphology_primary": primary,
        "morphology_secondary": tags[0] if tags else "",
        "morphology_secondary_json": str(tags).replace("'", '"'),
        "physical_primary": physical_primary,
    }


def test_five_class_target_reuses_established_review_rules() -> None:
    table = pd.DataFrame(
        [
            _row(
                "recurrent",
                event_class="dipper",
                primary="dimming_event",
                tags=["recurrent_dips"],
            ),
            _row(
                "non_recurrent",
                event_class="dipper",
                primary="dimming_event",
                tags=["single_dip"],
            ),
            _row(
                "eb_tag",
                event_class="periodic",
                primary="periodic",
                tags=["detached_binary_like"],
            ),
            _row(
                "eb_physical",
                event_class="periodic",
                primary="periodic",
                physical_primary="eclipsing_or_geometric_binary",
            ),
            _row(
                "artifact",
                event_class="instrumental",
                primary="artifact_or_bad_photometry",
            ),
            _row(
                "nonvariable",
                event_class="nonvariable_or_low_snr",
                primary="nonvariable_or_low_snr",
            ),
            _row(
                "other",
                event_class="brightening_event",
                primary="brightening_event",
            ),
        ]
    )

    _target, audit = construct_five_class_target(table)
    labels = dict(zip(table["candidate_id"], table[TARGET_COLUMN]))

    assert labels == {
        "recurrent": "recurrent_dimming_event",
        "non_recurrent": "non_recurrent_dimming_event",
        "eb_tag": "eclipsing_binary",
        "eb_physical": "eclipsing_binary",
        "artifact": "junk",
        "nonvariable": "junk",
        "other": "other",
    }
    assert audit["class_counts"] == {
        "recurrent_dimming_event": 1,
        "non_recurrent_dimming_event": 1,
        "eclipsing_binary": 2,
        "junk": 2,
        "other": 1,
    }
    assert tuple(audit["class_order"]) == FIVE_CLASS_ORDER
    assert audit["n_overlap_rows"] == 0


def test_ambiguous_dimming_and_unclassified_reviews_are_excluded() -> None:
    table = pd.DataFrame(
        [
            _row(
                "missing_recurrence",
                event_class="dipper",
                primary="dimming_event",
            ),
            _row(
                "conflicting_recurrence",
                event_class="dipper",
                primary="dimming_event",
                tags=["single_dip", "recurrent_dips"],
            ),
            _row(
                "unclassified",
                event_class="unclassified",
                primary="",
            ),
        ]
    )

    construct_five_class_target(table)

    assert table[TARGET_COLUMN].isna().all()
    source = dict(zip(table["candidate_id"], table["five_class_label_source"]))
    assert source["missing_recurrence"] == (
        "excluded_dimming_without_unambiguous_recurrence"
    )
    assert source["conflicting_recurrence"] == (
        "excluded_conflicting_human_recurrence_tags"
    )
    assert source["unclassified"] == (
        "excluded_blank_or_unclassified_event_class"
    )


def test_five_class_config_matches_eight_class_lightgbm_regime() -> None:
    config = five_class_training_config()

    assert config.n_estimators == 2500
    assert config.learning_rate == 0.03
    assert config.num_leaves == 23
    assert config.min_child_samples == 10
    assert config.class_weight == "balanced"
    assert config.early_stopping_rounds == 100
    assert config.calibration_method == "none"
