from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.july1_dimming_hierarchy_training import (
    FINAL_SCORE_COLUMNS,
    PARENT_CLASS_ORDER,
    PARENT_TARGET,
    compose_dimming_hierarchy_scores,
    construct_four_class_parent_target,
    validate_dimming_hierarchy_scores,
)
from malca.meta_analysis.ml.july1_review_training import (
    _dipper_recurrence_labels,
)


def _row(
    candidate_id: str,
    *,
    event_class: str,
    primary: str,
    tags: list[str] | None = None,
    physical_primary: str = "",
) -> dict[str, object]:
    secondary_tags = tags or []
    return {
        "candidate_id": candidate_id,
        "status": "reviewed",
        "workflow_status": "reviewed",
        "event_class": event_class,
        "morphology_primary": primary,
        "morphology_secondary": (
            secondary_tags[0] if secondary_tags else ""
        ),
        "morphology_secondary_json": secondary_tags,
        "physical_primary": physical_primary,
    }


def test_parent_includes_all_dimmers_and_recurrence_head_is_conditional() -> None:
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
                "recurrence_unlabeled",
                event_class="dipper",
                primary="dimming_event",
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
            _row(
                "unclassified",
                event_class="unclassified",
                primary="",
            ),
        ]
    )

    _target, parent_audit = construct_four_class_parent_target(table)
    recurrence_target, _positive, recurrence_audit = (
        _dipper_recurrence_labels(table)
    )
    labels = table.set_index("candidate_id")

    assert labels.loc["recurrent", PARENT_TARGET] == "dimming_event"
    assert labels.loc["non_recurrent", PARENT_TARGET] == "dimming_event"
    assert (
        labels.loc["recurrence_unlabeled", PARENT_TARGET]
        == "dimming_event"
    )
    assert pd.isna(
        labels.loc["recurrence_unlabeled", recurrence_target]
    )
    assert labels.loc["eb", PARENT_TARGET] == "eclipsing_binary"
    assert labels.loc["junk", PARENT_TARGET] == "junk"
    assert labels.loc["other", PARENT_TARGET] == "other"
    assert pd.isna(labels.loc["unclassified", PARENT_TARGET])
    assert parent_audit["class_counts"] == {
        "dimming_event": 3,
        "eclipsing_binary": 1,
        "junk": 1,
        "other": 1,
    }
    assert tuple(parent_audit["class_order"]) == PARENT_CLASS_ORDER
    assert recurrence_audit["n_trainable"] == 2


def test_composed_hierarchy_scores_preserve_probability_identities() -> None:
    table = pd.DataFrame(
        {
            "candidate_id": ["dimmer", "junk"],
            "status": ["unreviewed", "unreviewed"],
            "workflow_status": ["unreviewed", "unreviewed"],
            "event_class": ["unclassified", "unclassified"],
            PARENT_TARGET: [pd.NA, pd.NA],
            "human_dipper_recurrence_label": [pd.NA, pd.NA],
        }
    )
    parent_result = SimpleNamespace(
        label_classes=list(PARENT_CLASS_ORDER),
        probability_columns=[
            "prob_dimming",
            "prob_eb",
            "prob_junk",
            "prob_other",
        ],
        target_column=PARENT_TARGET,
    )
    parent_predictions = pd.DataFrame(
        {
            "candidate_id": ["dimmer", "junk"],
            "y_pred": ["dimming_event", "junk"],
            "prob_dimming": [0.8, 0.1],
            "prob_eb": [0.1, 0.1],
            "prob_junk": [0.05, 0.7],
            "prob_other": [0.05, 0.1],
        }
    )
    recurrence_result = SimpleNamespace(
        label_classes=[
            "non_recurrent_given_dipper",
            "recurrent_given_dipper",
        ],
        probability_columns=["prob_non_recurrent", "prob_recurrent"],
        target_column="human_dipper_recurrence_label",
    )
    recurrence_predictions = pd.DataFrame(
        {
            "candidate_id": ["dimmer", "junk"],
            "y_pred": [
                "recurrent_given_dipper",
                "non_recurrent_given_dipper",
            ],
            "prob_non_recurrent": [0.25, 0.8],
            "prob_recurrent": [0.75, 0.2],
        }
    )

    scores = compose_dimming_hierarchy_scores(
        table,
        parent_result=parent_result,
        parent_predictions=parent_predictions,
        recurrence_result=recurrence_result,
        recurrence_predictions=recurrence_predictions,
    )
    report = validate_dimming_hierarchy_scores(
        scores, candidate_ids={"dimmer", "junk"}
    )

    assert np.isclose(
        scores.loc[
            0, "score_hierarchical_recurrent_dimming_event"
        ],
        0.8 * 0.75,
    )
    assert (
        scores.loc[0, "predicted_hierarchical_class"]
        == "recurrent_dimming_event"
    )
    assert scores.loc[1, "predicted_hierarchical_class"] == "junk"
    assert scores.loc[1, "predicted_dimming_subclass"] == "not_applicable"
    assert np.allclose(scores[list(FINAL_SCORE_COLUMNS)].sum(axis=1), 1.0)
    assert report["max_identity_error"] < 1e-12
