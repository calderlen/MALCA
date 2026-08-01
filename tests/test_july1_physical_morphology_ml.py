from __future__ import annotations

import pandas as pd

from malca.meta_analysis.ml.july1_physical_morphology_training import (
    ASSIGNMENT_PRECEDENCE,
    PHYSICAL_MORPHOLOGY_CLASS_ORDER,
    TARGET_COLUMN,
    construct_physical_morphology_target,
    physical_morphology_training_config,
)


def _row(
    candidate_id: str,
    primary: str,
    tags: list[str],
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "status": "reviewed",
        "workflow_status": "reviewed",
        "morphology_primary": primary,
        "morphology_secondary": tags[0] if tags else "",
        "morphology_secondary_json": str(tags).replace("'", '"'),
    }


def test_construct_physical_morphology_target_covers_requested_classes() -> None:
    table = pd.DataFrame(
        [
            _row("artifact", "artifact_or_bad_photometry", ["bad_photometry"]),
            _row("moving", "artifact_or_bad_photometry", ["moving_object"]),
            _row("eb", "periodic", ["detached_binary_like"]),
            _row("pulsator", "periodic", ["pulsator_like"]),
            _row("rotator", "periodic", ["rotator_like"]),
            _row("microlens", "brightening_event", ["possible_microlensing_event"]),
            _row("flare", "brightening_event", ["possible_flare"]),
            _row("yso", "quasi_periodic", ["quasi_periodic_accretion_variability"]),
            _row("dust", "dimming_event", ["color_dependent_dip"]),
            _row("compact", "brightening_event", ["possible_outburst"]),
            _row("unmatched", "quasi_periodic", ["quasi_periodic_brightening"]),
        ]
    )
    _target, audit = construct_physical_morphology_target(table)

    assert set(table[TARGET_COLUMN].dropna()) == set(
        PHYSICAL_MORPHOLOGY_CLASS_ORDER
    )
    assert pd.isna(
        table.loc[table["candidate_id"].eq("unmatched"), TARGET_COLUMN].iloc[0]
    )
    assert audit["n_trainable"] == 10
    assert audit["n_reviewed_excluded_without_requested_rule"] == 1


def test_specific_rules_win_over_broader_rules() -> None:
    table = pd.DataFrame(
        [
            _row("moving", "artifact_or_bad_photometry", ["moving_object"]),
            _row(
                "micro_outburst",
                "brightening_event",
                ["possible_microlensing_event", "possible_outburst"],
            ),
            _row(
                "flare_outburst",
                "brightening_event",
                ["possible_flare", "fast_rise_slow_decline"],
            ),
        ]
    )
    construct_physical_morphology_target(table)
    labels = dict(zip(table["candidate_id"], table[TARGET_COLUMN]))

    assert labels["moving"] == "solar_system_or_moving_object"
    assert labels["micro_outburst"] == "microlensing"
    assert (
        labels["flare_outburst"]
        == "flare_star_or_magnetically_active_star"
    )
    assert ASSIGNMENT_PRECEDENCE.index("microlensing") < (
        ASSIGNMENT_PRECEDENCE.index("cataclysmic_or_compact_accretor")
    )


def test_physical_morphology_config_retains_four_row_flare_class() -> None:
    config = physical_morphology_training_config()

    assert config.min_class_count == 4
    assert config.calibration_method == "none"
    assert config.class_weight == "balanced"
