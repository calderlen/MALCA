"""Five-class morphology LightGBM training for the July 1 Review population.

The target deliberately reuses the human-label constructions from the
pre-existing July 1 classifiers:

* recurrent/non-recurrent dimming follows the dipper-recurrence model;
* eclipsing binary follows the eight-class EB-like rule;
* junk follows the eight-class combined artifact/nonvariable rejection rule;
* other contains the remaining clearly reviewed, non-dimming population.

Human taxonomy fields are label provenance only.  The predictor matrix is the
same stats/native-light-curve/recovery/context block used by the July 1
eight-class model.
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
    EB_REVIEW_TAGS,
    _clean_text,
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
    / "five_class_morphology_ml"
    / "stats_plus_periodicity_dip_jump_context"
)
TARGET_COLUMN = "human_five_class_label"
FIVE_CLASS_ORDER = (
    "recurrent_dimming_event",
    "non_recurrent_dimming_event",
    "eclipsing_binary",
    "junk",
    "other",
)

# These values intentionally match the earlier binary dipper and conditional
# recurrence models.  Ambiguous recurrence annotations are excluded rather
# than silently assigned to ``other``.
DIPPER_EVENT_CLASSES = {"dipper", "mixed_dip_and_burst"}
DIPPER_MORPHOLOGIES = {"dimming_event", "mixed_dip_and_burst"}
EXCLUDED_EVENT_CLASSES = {"", "unclassified"}
RECURRENT_DIP_TAGS = {
    "recurrent_dips",
    "periodic_dips",
    "quasi_periodic_dips",
    "aperiodic_dips",
    "stochastic_dips",
}
JUNK_MORPHOLOGIES = {
    "artifact_or_bad_photometry",
    "nonvariable_or_low_snr",
}


def construct_five_class_target(
    table: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    """Attach five mutually exclusive labels built from established rules."""

    reviewed = _reviewed_mask(table)
    event_class = _clean_text(table, "event_class")
    morphology_primary = _clean_text(table, "morphology_primary")
    physical_primary = _clean_text(table, "physical_primary")
    tags = _secondary_tag_sets(table)

    # Match the binary dipper model's clear-review cohort and positive-family
    # construction.  The recurrence split itself matches the dedicated
    # dipper-recurrence model's human secondary-tag rules.
    eligible = reviewed & ~event_class.isin(EXCLUDED_EVENT_CLASSES)
    dimming_family = eligible & (
        event_class.isin(DIPPER_EVENT_CLASSES)
        | morphology_primary.isin(DIPPER_MORPHOLOGIES)
    )
    has_recurrent_tag = dimming_family & tags.map(
        lambda values: bool(values & RECURRENT_DIP_TAGS)
    )
    has_single_tag = dimming_family & tags.map(
        lambda values: "single_dip" in values
    )
    recurrence_conflict = has_recurrent_tag & has_single_tag

    recurrent = has_recurrent_tag & ~has_single_tag
    non_recurrent = has_single_tag & ~has_recurrent_tag

    # Match the eight-class classifier's EB-like and combined rejection rules.
    eclipsing_binary = eligible & (
        tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
        | physical_primary.eq("eclipsing_or_geometric_binary")
    )
    junk = eligible & morphology_primary.isin(JUNK_MORPHOLOGIES)

    specific_masks = {
        "recurrent_dimming_event": recurrent,
        "non_recurrent_dimming_event": non_recurrent,
        "eclipsing_binary": eclipsing_binary,
        "junk": junk,
    }
    membership_count = sum(mask.astype("int16") for mask in specific_masks.values())
    if bool(membership_count.gt(1).any()):
        overlapping = table.loc[
            membership_count.gt(1), "candidate_id"
        ].astype(str).tolist()
        raise ValueError(
            "Established five-class rules overlap for "
            f"{len(overlapping)} candidates: {overlapping[:10]}"
        )

    # Do not turn recurrence-ambiguous dimmers into negative examples.  Other
    # is the remaining clear reviewed, non-dimming population.
    other = (
        eligible
        & ~dimming_family
        & ~eclipsing_binary
        & ~junk
    )
    masks = {**specific_masks, "other": other}

    target = pd.Series(pd.NA, index=table.index, dtype="string")
    source = pd.Series(
        "excluded_not_clear_review", index=table.index, dtype="object"
    )
    source.loc[reviewed & ~eligible] = (
        "excluded_blank_or_unclassified_event_class"
    )
    source.loc[
        dimming_family & ~(recurrent | non_recurrent)
    ] = "excluded_dimming_without_unambiguous_recurrence"
    source.loc[recurrence_conflict] = (
        "excluded_conflicting_human_recurrence_tags"
    )

    source_names = {
        "recurrent_dimming_event": "existing_human_recurrent_dip_morphology",
        "non_recurrent_dimming_event": "existing_human_single_dip_morphology",
        "eclipsing_binary": "existing_human_eb_review_rule",
        "junk": "existing_human_artifact_or_nonvariable_rule",
        "other": "remaining_clear_reviewed_non_dimming_class",
    }
    for label in FIVE_CLASS_ORDER:
        target.loc[masks[label]] = label
        source.loc[masks[label]] = source_names[label]

    table[TARGET_COLUMN] = target
    table["five_class_label_source"] = source
    table["five_class_human_secondary_tags"] = tags.map(
        lambda values: "|".join(sorted(values))
    )

    counts = {
        label: int(masks[label].sum()) for label in FIVE_CLASS_ORDER
    }
    audit = {
        "target_column": TARGET_COLUMN,
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "n_clear_review_eligible": int(eligible.sum()),
        "n_trainable": int(target.notna().sum()),
        "class_order": list(FIVE_CLASS_ORDER),
        "class_counts": counts,
        "n_overlap_rows": int(membership_count.gt(1).sum()),
        "n_reviewed_excluded_blank_or_unclassified": int(
            (reviewed & ~eligible).sum()
        ),
        "n_dimming_family": int(dimming_family.sum()),
        "n_dimming_excluded_without_unambiguous_recurrence": int(
            (dimming_family & ~(recurrent | non_recurrent)).sum()
        ),
        "n_conflicting_recurrence_tags": int(recurrence_conflict.sum()),
        "junk_component_counts": {
            morphology: int((junk & morphology_primary.eq(morphology)).sum())
            for morphology in sorted(JUNK_MORPHOLOGIES)
        },
        "eclipsing_binary_component_counts": {
            "human_eb_secondary_tag": int(
                (
                    eligible
                    & tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
                ).sum()
            ),
            "physical_primary_eclipsing_binary": int(
                (
                    eligible
                    & physical_primary.eq("eclipsing_or_geometric_binary")
                ).sum()
            ),
        },
        "label_definition": (
            "Five mutually exclusive human-review morphology classes using "
            "the established dipper-recurrence, eight-class EB, and "
            "eight-class rejection constructions."
        ),
        "other_definition": (
            "Clear reviewed rows that are not in the binary dipper family, "
            "the established EB-like class, or the established combined "
            "artifact/nonvariable rejection class."
        ),
        "scientific_warning": (
            "Rows in the dimming family without an unambiguous recurrent or "
            "single-dip human tag are excluded from training, not labeled other."
        ),
    }
    return TARGET_COLUMN, audit


def five_class_training_config() -> TrainingConfig:
    """Use the same LightGBM regime as the established eight-class model."""

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
        min_class_count=5,
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
    status = _clean_text(table, "status")
    workflow = _clean_text(table, "workflow_status")
    event_class = _clean_text(table, "event_class")
    return (
        status.isin(("", "unreviewed"))
        & workflow.isin(("", "unreviewed"))
        & event_class.isin(("", "unclassified"))
    )


def train_five_class_model(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
    top_unreviewed_n: int = 500,
) -> dict[str, Any]:
    """Fit, save, score, and queue the five-class morphology model."""

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
    target, audit = construct_five_class_target(table)
    if min(audit["class_counts"].values()) < 5:
        raise ValueError(
            "A requested five-class target has fewer than five examples: "
            f"{audit['class_counts']}"
        )

    trainable = table[target].notna()
    feature_columns = _eight_class_features(table, trainable)
    feature_columns, duplicate_aliases = _drop_exact_duplicate_features(
        table.loc[trainable], feature_columns
    )
    audit["dropped_duplicate_feature_aliases"] = duplicate_aliases
    audit["n_features_requested"] = int(len(feature_columns))
    audit["feature_policy"] = (
        "Same stats, native periodicity/dip/jump, recovery-bounded event, "
        "and astrophysical-context block as the July 1 eight-class model; "
        "all human Review taxonomy fields are excluded from predictors."
    )

    model_input = table[["candidate_id", target, *feature_columns]].copy()
    result = train_target_model(
        model_input, target, config=five_class_training_config()
    )
    save_target_model(result, out_dir)
    _write_gain_importance(result, out_dir)

    predictions = score_target_model(
        out_dir, table[["candidate_id", *feature_columns]].copy()
    )
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
            "five_class_label_source",
            "five_class_human_secondary_tags",
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
        out_dir / "all_candidates_five_class_scores.parquet", index=False
    )

    unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
    probability_by_label = dict(
        zip(result.label_classes, result.probability_columns)
    )
    queues: list[pd.DataFrame] = []
    for label in FIVE_CLASS_ORDER:
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
        out_dir / "top_unreviewed_by_five_class.csv", index=False
    )
    unreviewed.sort_values(
        ["prediction_entropy", "score_margin"], ascending=[False, True]
    ).head(500).to_csv(
        out_dir / "most_ambiguous_unreviewed.csv", index=False
    )

    label_columns = [
        column
        for column in (
            "candidate_id",
            "event_class",
            "morphology_primary",
            "morphology_secondary",
            "five_class_human_secondary_tags",
            target,
            "five_class_label_source",
            "physical_primary",
            "physical_secondary",
        )
        if column in table.columns
    ]
    table.loc[trainable, label_columns].to_parquet(
        out_dir / "constructed_training_labels.parquet", index=False
    )
    table.loc[_reviewed_mask(table), label_columns].to_parquet(
        out_dir / "reviewed_label_audit.parquet", index=False
    )
    (out_dir / "label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_key": "five_class_morphology",
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
            "Class-balanced outputs are uncalibrated ranking scores, not "
            "population probabilities."
        ),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Trained five_class_morphology model: {len(table):,} candidates, "
        f"{result.n_rows:,} labeled rows, {result.n_features} features -> "
        f"{out_dir}"
    )
    return summary


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "FIVE_CLASS_ORDER",
    "TARGET_COLUMN",
    "construct_five_class_target",
    "five_class_training_config",
    "train_five_class_model",
]
