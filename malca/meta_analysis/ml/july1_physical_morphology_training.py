"""Morphology-derived physical-candidate LightGBM training for July 1 Review.

This model deliberately predicts broad *candidate families* constructed from
human morphology labels.  The constructed target is not independent physical
ground truth.  Morphology, physical-review, and other taxonomy fields never
enter the predictor matrix; only the same stats/native-light-curve/context
feature block used by the July 1 eight-class model is retained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.candidate_features import (
    RECOVERY_FEATURE_SCHEMA_VERSION,
    add_recovery_bounded_event_features,
    default_recovery_feature_cache,
)
from malca.meta_analysis.ml.july1_review_training import (
    DEFAULT_DB_PATH,
    DEFAULT_RUN_DIR,
    _drop_exact_duplicate_features,
    _eight_class_features,
    _reviewed_mask,
    _secondary_tag_sets,
    load_review_population,
)
from malca.meta_analysis.ml.review_lightgbm import (
    TrainingConfig,
    save_target_model,
    score_target_model,
    train_target_model,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_RUN_DIR
    / "results"
    / "physical_morphology_ml"
    / "stats_plus_periodicity_dip_jump_context"
)
TARGET_COLUMN = "morphology_derived_physical_label"

# Report order follows the user's requested physical-family table. Assignment
# order below is different so specific subsets win over broader families.
PHYSICAL_MORPHOLOGY_CLASS_ORDER = (
    "false_positive_or_contaminant",
    "solar_system_or_moving_object",
    "eclipsing_or_geometric_binary",
    "pulsating_variable",
    "rotating_spotted_or_magnetic_variable",
    "microlensing",
    "flare_star_or_magnetically_active_star",
    "young_stellar_object_or_pms",
    "dust_obscuration_or_fading_variable",
    "cataclysmic_or_compact_accretor",
)
ASSIGNMENT_PRECEDENCE = (
    "solar_system_or_moving_object",
    "microlensing",
    "flare_star_or_magnetically_active_star",
    "eclipsing_or_geometric_binary",
    "pulsating_variable",
    "rotating_spotted_or_magnetic_variable",
    "young_stellar_object_or_pms",
    "dust_obscuration_or_fading_variable",
    "cataclysmic_or_compact_accretor",
    "false_positive_or_contaminant",
)

BINARY_TAGS = {
    "eclipsing_like",
    "contact_binary_like",
    "detached_binary_like",
    "semi_detached_binary_like",
    "ellipsoidal_like",
    "heartbeat_like",
}
YSO_ACCRETION_TAGS = {
    "quasi_periodic_accretion_variability",
    "yso_like_stochastic",
    "accretion_like_mixed_variability",
}
DUST_FADING_TAGS = {
    "color_dependent_dip",
    "secular_dimming",
    "monotonic_dimming",
    "long_duration_low_state",
    "dimming_without_recovery",
    "step_like_dimming",
}
COMPACT_ACCRETOR_OUTBURST_TAGS = {
    "possible_outburst",
    "fast_rise_exponential_decay",
    "fast_rise_slow_decline",
    "long_duration_outburst",
    "recurrent_brightenings",
}


def _has_any(tags: pd.Series, requested: set[str]) -> pd.Series:
    return tags.map(lambda values: bool(values & requested))


def construct_physical_morphology_target(
    table: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    """Assign ten mutually exclusive broad physical-candidate pseudo-labels."""

    reviewed = _reviewed_mask(table)
    primary = table.get(
        "morphology_primary", pd.Series("", index=table.index)
    ).fillna("").astype(str).str.strip()
    tags = _secondary_tag_sets(table)

    raw_masks = {
        "false_positive_or_contaminant": (
            reviewed & primary.eq("artifact_or_bad_photometry")
        ),
        "solar_system_or_moving_object": (
            reviewed
            & primary.eq("artifact_or_bad_photometry")
            & tags.map(lambda values: "moving_object" in values)
        ),
        "eclipsing_or_geometric_binary": (
            reviewed & primary.eq("periodic") & _has_any(tags, BINARY_TAGS)
        ),
        "pulsating_variable": (
            reviewed
            & (
                (
                    primary.eq("periodic")
                    & tags.map(lambda values: "pulsator_like" in values)
                )
                | (
                    primary.eq("long_period_variability")
                    & tags.map(
                        lambda values: "large_amplitude_pulsation" in values
                    )
                )
            )
        ),
        "rotating_spotted_or_magnetic_variable": (
            reviewed
            & (
                (
                    primary.eq("periodic")
                    & tags.map(lambda values: "rotator_like" in values)
                )
                | (
                    primary.eq("quasi_periodic")
                    & tags.map(
                        lambda values: "quasi_periodic_spot_modulation" in values
                    )
                )
            )
        ),
        "microlensing": (
            reviewed
            & primary.eq("brightening_event")
            & tags.map(lambda values: "possible_microlensing_event" in values)
        ),
        "flare_star_or_magnetically_active_star": (
            reviewed
            & primary.eq("brightening_event")
            & tags.map(lambda values: "possible_flare" in values)
        ),
        "young_stellar_object_or_pms": (
            reviewed
            & (
                primary.eq("mixed_dip_and_burst")
                | _has_any(tags, YSO_ACCRETION_TAGS)
            )
        ),
        "dust_obscuration_or_fading_variable": (
            reviewed
            & primary.eq("dimming_event")
            & _has_any(tags, DUST_FADING_TAGS)
        ),
        "cataclysmic_or_compact_accretor": (
            reviewed
            & primary.eq("brightening_event")
            & _has_any(tags, COMPACT_ACCRETOR_OUTBURST_TAGS)
        ),
    }

    membership_count = sum(
        mask.astype("int16") for mask in raw_masks.values()
    )
    target = pd.Series(pd.NA, index=table.index, dtype="string")
    source = pd.Series(
        "excluded_no_requested_morphology_rule", index=table.index, dtype="object"
    )
    assigned_masks: dict[str, pd.Series] = {}
    for label in ASSIGNMENT_PRECEDENCE:
        assign = raw_masks[label] & target.isna()
        target.loc[assign] = label
        source.loc[assign] = f"human_morphology_rule:{label}"
        assigned_masks[label] = assign

    table[TARGET_COLUMN] = target
    table["morphology_derived_physical_label_source"] = source
    table["morphology_derived_physical_tags"] = tags.map(
        lambda values: "|".join(sorted(values))
    )

    assigned_counts = {
        label: int(assigned_masks[label].sum())
        for label in PHYSICAL_MORPHOLOGY_CLASS_ORDER
    }
    raw_counts = {
        label: int(raw_masks[label].sum())
        for label in PHYSICAL_MORPHOLOGY_CLASS_ORDER
    }
    audit = {
        "target_column": TARGET_COLUMN,
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "n_trainable": int(target.notna().sum()),
        "class_order": list(PHYSICAL_MORPHOLOGY_CLASS_ORDER),
        "assignment_precedence": list(ASSIGNMENT_PRECEDENCE),
        "raw_rule_counts": raw_counts,
        "exclusive_class_counts": assigned_counts,
        "n_rows_matching_multiple_raw_rules": int(membership_count.gt(1).sum()),
        "n_reviewed_excluded_without_requested_rule": int(
            (reviewed & target.isna()).sum()
        ),
        "label_definition": (
            "Mutually exclusive broad physical-candidate pseudo-labels "
            "constructed from human primary/secondary morphology."
        ),
        "scientific_warning": (
            "These are morphology-derived candidate families, not independent "
            "physical ground truth or calibrated probabilities."
        ),
    }
    return TARGET_COLUMN, audit


def physical_morphology_training_config() -> TrainingConfig:
    """Return the same LightGBM regime as the eight-class Review model."""

    return TrainingConfig(
        random_state=42,
        val_size=0.15,
        test_size=0.20,
        cv_folds=5,
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=23,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        max_categorical_cardinality=50,
        min_class_count=4,
        class_weight="balanced",
        n_jobs=4,
        early_stopping_rounds=100,
        early_stopping_min_delta=0.0,
        early_stopping_selection_folds=3,
        calibration_method="none",
        reliability_bins=10,
    )


def _write_gain_importance(result: Any, output_dir: Path) -> None:
    booster = result.model.booster_
    pd.DataFrame(
        {
            "feature": result.feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False, ignore_index=True).to_csv(
        output_dir / "feature_importance_gain.csv", index=False
    )


def _human_unreviewed_mask(table: pd.DataFrame) -> pd.Series:
    status = table.get("status", pd.Series("", index=table.index))
    workflow = table.get("workflow_status", pd.Series("", index=table.index))
    event_class = table.get("event_class", pd.Series("", index=table.index))
    status = status.fillna("").astype(str).str.strip()
    workflow = workflow.fillna("").astype(str).str.strip()
    event_class = event_class.fillna("").astype(str).str.strip()
    return (
        status.isin(("", "unreviewed"))
        & workflow.isin(("", "unreviewed"))
        & event_class.isin(("", "unclassified"))
    )


def train_physical_morphology_model(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
    top_unreviewed_n: int = 500,
) -> dict[str, Any]:
    """Fit, save, and score the ten-class morphology-derived physical model."""

    db_path = Path(db_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        Path(recovery_feature_cache)
        if recovery_feature_cache is not None
        else default_recovery_feature_cache(db_path)
    )

    table = load_review_population(
        db_path, keep_morphology_secondary_json=True
    )
    table = add_recovery_bounded_event_features(
        table, cache_path, workers=recovery_workers
    )
    target, audit = construct_physical_morphology_target(table)
    if min(audit["exclusive_class_counts"].values()) < 3:
        raise ValueError(
            "A requested physical-morphology class has fewer than three "
            f"exclusive examples: {audit['exclusive_class_counts']}"
        )
    trainable = table[target].notna()
    feature_columns = _eight_class_features(table, trainable)
    feature_columns, duplicate_aliases = _drop_exact_duplicate_features(
        table.loc[trainable], feature_columns
    )
    audit["dropped_duplicate_feature_aliases"] = duplicate_aliases
    audit["n_features_requested"] = int(len(feature_columns))
    audit["feature_policy"] = (
        "Same stats, native light-curve, recovery, and astrophysical-context "
        "block as the July 1 eight-class model; all Review taxonomy fields "
        "are excluded."
    )

    model_input = table[["candidate_id", target, *feature_columns]].copy()
    config = physical_morphology_training_config()
    result = train_target_model(model_input, target, config=config)
    save_target_model(result, out_dir)
    _write_gain_importance(result, out_dir)

    scoring_input = table[["candidate_id", *feature_columns]].copy()
    predictions = score_target_model(out_dir, scoring_input)
    context_columns = [
        column
        for column in (
            "candidate_id",
            "asas_sn_id",
            "lc_path",
            "status",
            "workflow_status",
            "event_class",
            "morphology_primary",
            "morphology_secondary",
            "physical_primary",
            "physical_secondary",
            target,
            "morphology_derived_physical_label_source",
            "morphology_derived_physical_tags",
        )
        if column in table.columns
    ]
    prediction_columns = [
        "candidate_id",
        "y_pred",
        "prediction_confidence",
        *result.probability_columns,
    ]
    scores = table[context_columns].merge(
        predictions[prediction_columns], on="candidate_id", how="left"
    )
    matrix = scores[result.probability_columns].to_numpy(dtype=float)
    sorted_matrix = np.sort(matrix, axis=1)
    scores["score_margin"] = sorted_matrix[:, -1] - sorted_matrix[:, -2]
    clipped = np.clip(matrix, 1e-12, 1.0)
    scores["prediction_entropy"] = -(
        clipped * np.log(clipped)
    ).sum(axis=1) / math.log(len(result.probability_columns))
    scores["is_human_unreviewed"] = _human_unreviewed_mask(table).to_numpy()
    scores.to_parquet(
        out_dir / "all_candidates_physical_morphology_scores.parquet",
        index=False,
    )

    unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
    queues: list[pd.DataFrame] = []
    probability_by_label = dict(zip(result.label_classes, result.probability_columns))
    for label in PHYSICAL_MORPHOLOGY_CLASS_ORDER:
        probability_column = probability_by_label[label]
        queue = unreviewed.sort_values(
            probability_column, ascending=False
        ).head(top_unreviewed_n).copy()
        queue["queue_class"] = label
        queue["rank_within_class"] = np.arange(1, len(queue) + 1)
        queue.to_csv(
            out_dir / f"top{top_unreviewed_n}_unreviewed_{label}.csv",
            index=False,
        )
        queues.append(queue)
    pd.concat(queues, ignore_index=True).to_csv(
        out_dir / "top_unreviewed_by_physical_morphology_class.csv",
        index=False,
    )

    assignments = table.loc[
        trainable,
        [
            "candidate_id",
            "morphology_primary",
            "morphology_secondary",
            "morphology_derived_physical_tags",
            target,
            "morphology_derived_physical_label_source",
            "physical_primary",
            "physical_secondary",
        ],
    ].copy()
    assignments.to_parquet(
        out_dir / "constructed_training_labels.parquet", index=False
    )
    (out_dir / "label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_key": "physical_morphology",
        "db_path": str(db_path),
        "model_dir": str(out_dir),
        "target_column": target,
        "n_candidates": int(len(table)),
        "n_trainable": int(result.n_rows),
        "class_counts": result.class_counts,
        "n_features": int(result.n_features),
        "feature_columns": list(result.feature_columns),
        "holdout_metrics": result.holdout_metrics,
        "label_audit": audit,
        "recovery_feature_cache": str(cache_path.expanduser().resolve()),
        "recovery_feature_schema_version": RECOVERY_FEATURE_SCHEMA_VERSION,
        "warning": (
            "Morphology-derived candidate families and class-balanced scores "
            "are not independent physical truth or calibrated probabilities."
        ),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Trained physical_morphology model: {len(table):,} candidates, "
        f"{result.n_rows:,} labeled rows, {result.n_features} features -> {out_dir}"
    )
    return summary


__all__ = [
    "ASSIGNMENT_PRECEDENCE",
    "DEFAULT_OUTPUT_DIR",
    "PHYSICAL_MORPHOLOGY_CLASS_ORDER",
    "TARGET_COLUMN",
    "construct_physical_morphology_target",
    "physical_morphology_training_config",
    "train_physical_morphology_model",
]
