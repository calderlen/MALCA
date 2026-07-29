from __future__ import annotations

import json

import pandas as pd

from malca.meta_analysis.ml.july1_hierarchical_training import (
    GATE_TARGET,
    LONG_TIMESCALE_TARGET,
    MICROLENSING_TARGET,
    PRIMARY_TARGET,
    QUASI_PERIODIC_TARGET,
    build_hierarchical_labels,
)


def _row(
    candidate_id: str,
    morphology_primary: str,
    *,
    secondary: list[str] | None = None,
    physical_primary: str = "",
    event_class: str = "",
) -> dict[str, object]:
    tags = secondary or []
    return {
        "candidate_id": candidate_id,
        "status": "reviewed",
        "workflow_status": "reviewed",
        "event_class": event_class,
        "morphology_primary": morphology_primary,
        "morphology_secondary": tags[0] if tags else "",
        "morphology_secondary_json": json.dumps(tags),
        "physical_primary": physical_primary,
    }


def test_hierarchy_separates_primary_morphology_from_subtype_axes() -> None:
    table = pd.DataFrame(
        [
            _row("artifact", "artifact_or_bad_photometry"),
            _row("flat", "nonvariable_or_low_snr"),
            _row(
                "qp_dipper",
                "dimming_event",
                secondary=["quasi_periodic_dips", "recurrent_dips"],
                event_class="dipper",
            ),
            _row(
                "eb",
                "periodic",
                secondary=["detached_binary_like"],
                physical_primary="eclipsing_or_geometric_binary",
            ),
            _row("periodic_other", "periodic", secondary=["pulsator_like"]),
            _row("ltv", "long_term_trend"),
            _row("lpv", "long_period_variability"),
            _row(
                "micro",
                "brightening_event",
                secondary=["possible_microlensing_event"],
                physical_primary="microlensing",
                event_class="microlensing",
            ),
            _row("bright", "brightening_event"),
            _row(
                "qp_bright",
                "quasi_periodic",
                secondary=["quasi_periodic_brightening"],
            ),
        ]
    )

    audit = build_hierarchical_labels(table)
    labels = table.set_index("candidate_id")

    assert labels.loc["artifact", GATE_TARGET] == "artifact_or_nonvariable"
    assert labels.loc["flat", GATE_TARGET] == "artifact_or_nonvariable"
    assert labels.loc["qp_dipper", PRIMARY_TARGET] == "dipper_dimming"
    assert labels.loc["qp_dipper", QUASI_PERIODIC_TARGET] == "quasi_periodic"
    assert labels.loc["eb", PRIMARY_TARGET] == "eb_geometric_periodic"
    assert (
        labels.loc["periodic_other", PRIMARY_TARGET]
        == "other_structured_variable"
    )
    assert labels.loc["ltv", PRIMARY_TARGET] == "long_timescale_variable"
    assert labels.loc["ltv", LONG_TIMESCALE_TARGET] == "long_term_variable"
    assert labels.loc["lpv", LONG_TIMESCALE_TARGET] == "long_period_variable"
    assert labels.loc["micro", PRIMARY_TARGET] == "brightening_transient"
    assert labels.loc["micro", MICROLENSING_TARGET] == "microlensing_like"
    assert (
        labels.loc["bright", MICROLENSING_TARGET]
        == "not_microlensing_like"
    )
    assert (
        labels.loc["qp_bright", PRIMARY_TARGET]
        == "other_structured_variable"
    )
    assert labels.loc["qp_bright", QUASI_PERIODIC_TARGET] == "quasi_periodic"
    assert audit["gate"]["class_counts"] == {
        "usable_astrophysical_variable": 8,
        "artifact_or_nonvariable": 2,
    }
