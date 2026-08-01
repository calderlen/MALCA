from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from malca.meta_analysis.ml.july1_four_class_hierarchical_training import (
    PARENT_TARGET,
    build_hierarchy_targets,
    compose_hierarchical_scores,
    construct_four_class_parent_target,
)


def _row(
    candidate_id: str,
    *,
    event_class: str,
    primary: str,
    tags: list[str] | None = None,
    physical_primary: str = "",
    reviewed: bool = True,
) -> dict[str, object]:
    tags = tags or []
    status = "reviewed" if reviewed else "unreviewed"
    return {
        "candidate_id": candidate_id,
        "status": status,
        "workflow_status": status,
        "event_class": event_class,
        "morphology_primary": primary,
        "morphology_secondary": tags[0] if tags else "",
        "morphology_secondary_json": str(tags).replace("'", '"'),
        "physical_primary": physical_primary,
    }


def test_parent_keeps_all_clear_dimmers_regardless_of_subtype_label() -> None:
    table = pd.DataFrame(
        [
            _row(
                "recurrent",
                event_class="dipper",
                primary="dimming_event",
                tags=["recurrent_dips"],
            ),
            _row(
                "single",
                event_class="dipper",
                primary="dimming_event",
                tags=["single_dip"],
            ),
            _row(
                "ambiguous",
                event_class="dipper",
                primary="dimming_event",
                tags=["broad_dip"],
            ),
            _row(
                "eb",
                event_class="periodic",
                primary="periodic",
                tags=["detached_binary_like"],
            ),
            _row(
                "junk",
                event_class="instrumental",
                primary="artifact_or_bad_photometry",
            ),
            _row(
                "other",
                event_class="brightening_event",
                primary="brightening_event",
            ),
        ]
    )

    parent_target, recurrence_target, audit = build_hierarchy_targets(table)
    labels = table.set_index("candidate_id")

    assert labels.loc["recurrent", parent_target] == "dimming_event"
    assert labels.loc["single", parent_target] == "dimming_event"
    assert labels.loc["ambiguous", parent_target] == "dimming_event"
    assert (
        labels.loc["recurrent", recurrence_target]
        == "recurrent_given_dipper"
    )
    assert (
        labels.loc["single", recurrence_target]
        == "non_recurrent_given_dipper"
    )
    assert pd.isna(labels.loc["ambiguous", recurrence_target])
    assert audit["n_parent_dimming_without_subtype_label"] == 1


def test_parent_reuses_eb_junk_other_and_excludes_unclassified() -> None:
    table = pd.DataFrame(
        [
            _row(
                "eb",
                event_class="periodic",
                primary="periodic",
                tags=["eclipsing_like"],
            ),
            _row(
                "junk",
                event_class="nonvariable_or_low_snr",
                primary="nonvariable_or_low_snr",
            ),
            _row(
                "other",
                event_class="quasi_periodic",
                primary="quasi_periodic",
            ),
            _row(
                "unclassified",
                event_class="unclassified",
                primary="",
            ),
        ]
    )

    _, audit = construct_four_class_parent_target(table)

    assert table[PARENT_TARGET].tolist()[:3] == [
        "eclipsing_binary",
        "junk",
        "other",
    ]
    assert pd.isna(table.loc[3, PARENT_TARGET])
    assert audit["n_reviewed_excluded_blank_or_unclassified"] == 1


def test_parent_source_overlap_raises() -> None:
    table = pd.DataFrame(
        [
            _row(
                "dimming_eb",
                event_class="dipper",
                primary="dimming_event",
                tags=["recurrent_dips", "eclipsing_like"],
            )
        ]
    )

    with pytest.raises(ValueError, match="parent labels overlap"):
        construct_four_class_parent_target(table)


def test_composed_leaf_scores_partition_parent_probability() -> None:
    ids = ["a", "b"]
    parent_result = SimpleNamespace(
        label_classes=[
            "dimming_event",
            "eclipsing_binary",
            "junk",
            "other",
        ],
        probability_columns=[
            "prob_dimming_event",
            "prob_eclipsing_binary",
            "prob_junk",
            "prob_other",
        ],
    )
    parent_predictions = pd.DataFrame(
        {
            "candidate_id": ids,
            "y_pred": ["dimming_event", "junk"],
            "prediction_confidence": [0.6, 0.7],
            "prob_dimming_event": [0.6, 0.1],
            "prob_eclipsing_binary": [0.1, 0.1],
            "prob_junk": [0.1, 0.7],
            "prob_other": [0.2, 0.1],
        }
    )
    recurrence_result = SimpleNamespace(
        label_classes=[
            "non_recurrent_given_dipper",
            "recurrent_given_dipper",
        ],
        probability_columns=[
            "prob_non_recurrent_given_dipper",
            "prob_recurrent_given_dipper",
        ],
    )
    recurrence_predictions = pd.DataFrame(
        {
            "candidate_id": ids,
            "y_pred": [
                "recurrent_given_dipper",
                "non_recurrent_given_dipper",
            ],
            "prediction_confidence": [0.75, 0.8],
            "prob_non_recurrent_given_dipper": [0.25, 0.8],
            "prob_recurrent_given_dipper": [0.75, 0.2],
        }
    )

    scores = compose_hierarchical_scores(
        parent_result,
        parent_predictions,
        recurrence_result,
        recurrence_predictions,
    )
    leaf_columns = [
        "prob_recurrent_dimming_event",
        "prob_non_recurrent_dimming_event",
        "prob_eclipsing_binary",
        "prob_junk",
        "prob_other",
    ]

    assert scores.loc[0, "prob_recurrent_dimming_event"] == pytest.approx(
        0.45
    )
    assert scores.loc[0, "prob_non_recurrent_dimming_event"] == pytest.approx(
        0.15
    )
    assert scores[leaf_columns].sum(axis=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert scores["predicted_hierarchical_leaf"].tolist() == [
        "recurrent_dimming_event",
        "junk",
    ]
    assert scores["predicted_dimming_subclass"].tolist() == [
        "recurrent_dimming_event",
        "not_applicable",
    ]
