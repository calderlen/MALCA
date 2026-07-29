from __future__ import annotations

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.candidate_features import (
    NEXT_ITERATION_CONTEXT_FEATURES,
    RECOVERY_BOUNDED_EVENT_FEATURES,
)
from malca.meta_analysis.ml.feature_policy import (
    STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
)
from malca.meta_analysis.ml.july1_review_training import (
    ADDITIONAL_LC_FEATURES,
    EIGHT_CLASS_CONTEXT_FEATURES,
    _eight_class_features,
    _eight_class_labels,
)


def test_eight_class_labels_collapse_rejection_classes_and_split_lpv_from_ltv() -> None:
    rows: list[dict[str, object]] = []

    def add_rows(
        prefix: str,
        *,
        event_class: str,
        morphology_primary: str,
        physical_primary: str = "",
    ) -> None:
        for index in range(5):
            rows.append(
                {
                    "candidate_id": f"{prefix}_{index}",
                    "status": "reviewed",
                    "workflow_status": "reviewed",
                    "event_class": event_class,
                    "morphology_primary": morphology_primary,
                    "morphology_secondary": "",
                    "morphology_secondary_json": "[]",
                    "physical_primary": physical_primary,
                }
            )

    add_rows("dipper", event_class="dipper", morphology_primary="dimming_event")
    add_rows(
        "eb",
        event_class="periodic",
        morphology_primary="periodic",
        physical_primary="eclipsing_or_geometric_binary",
    )
    add_rows("ltv", event_class="ltv", morphology_primary="long_term_trend")
    add_rows("lpv", event_class="ltv", morphology_primary="long_period_variability")
    add_rows("micro", event_class="microlensing", morphology_primary="brightening_event")
    add_rows("qp", event_class="quasi_periodic", morphology_primary="quasi_periodic")
    add_rows("bright", event_class="brightening_event", morphology_primary="brightening_event")
    add_rows(
        "artifact",
        event_class="instrumental",
        morphology_primary="artifact_or_bad_photometry",
    )
    add_rows(
        "nonvariable",
        event_class="nonvariable_or_low_snr",
        morphology_primary="nonvariable_or_low_snr",
    )

    table = pd.DataFrame(rows)
    target, audit = _eight_class_labels(table)

    assert audit["class_counts"] == {
        "dipper": 5,
        "eclipsing_binary_like": 5,
        "long_term_variable": 5,
        "long_period_variable": 5,
        "microlensing": 5,
        "quasi_periodic": 5,
        "brightening_event": 5,
        "artifact_or_nonvariable": 10,
    }
    assert set(table.loc[table["candidate_id"].str.startswith("ltv_"), target]) == {
        "long_term_variable"
    }
    assert set(table.loc[table["candidate_id"].str.startswith("lpv_"), target]) == {
        "long_period_variable"
    }
    assert set(table.loc[table["candidate_id"].str.startswith("artifact_"), target]) == {
        "artifact_or_nonvariable"
    }
    assert set(table.loc[table["candidate_id"].str.startswith("nonvariable_"), target]) == {
        "artifact_or_nonvariable"
    }
    assert audit["rejection_component_counts"] == {
        "artifact_or_bad_photometry": 5,
        "nonvariable_or_low_snr": 5,
    }


def test_eight_class_features_include_next_iteration_blocks() -> None:
    n_rows = 40
    frame = pd.DataFrame(
        {
            "candidate_id": [f"candidate_{index}" for index in range(n_rows)],
            "stats_signal": np.arange(n_rows, dtype=float),
        }
    )
    requested = (
        *ADDITIONAL_LC_FEATURES,
        *RECOVERY_BOUNDED_EVENT_FEATURES,
        *EIGHT_CLASS_CONTEXT_FEATURES,
    )
    for index, column in enumerate(requested):
        frame[column] = np.arange(n_rows, dtype=float) + index
    for index, column in enumerate(STATS_MODEL_FEATURE_EXCLUSION_COLUMNS):
        frame[column] = np.arange(n_rows, dtype=float) + 100 + index
    for column in (
        "periodicity_method",
        "dip_best_morph",
        "jump_best_morph",
        "left_event_boundary_type",
        "right_event_boundary_type",
        "dimming_complex_status",
    ):
        frame[column] = ["a", "b"] * (n_rows // 2)

    selected = _eight_class_features(
        frame, pd.Series(True, index=frame.index)
    )

    assert "long_ls_peak_power" in selected
    assert STATS_MODEL_FEATURE_EXCLUSION_COLUMNS.isdisjoint(selected)
    assert set(RECOVERY_BOUNDED_EVENT_FEATURES).issubset(selected)
    assert set(NEXT_ITERATION_CONTEXT_FEATURES).issubset(selected)
