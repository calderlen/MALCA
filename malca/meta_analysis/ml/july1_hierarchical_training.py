"""Hierarchical July 1 Review LightGBM training and scoring.

The hierarchy separates data quality, dominant morphology, and conditional
subtypes instead of forcing orthogonal human-review concepts into one flat
multiclass target.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.july1_review_training import (
    ADDITIONAL_LC_FEATURES,
    DEFAULT_DB_PATH,
    DEFAULT_RUN_DIR,
    EB_REVIEW_TAGS,
    EIGHT_CLASS_EXCLUDED_STATS,
    _clean_text,
    _dipper_recurrence_config,
    _dipper_recurrence_features,
    _dipper_recurrence_labels,
    _drop_exact_duplicate_features,
    _reviewed_mask,
    _secondary_tag_sets,
    _unreviewed_mask,
    is_usable_model_feature,
    load_review_population,
)
from malca.meta_analysis.ml.review_lightgbm import (
    ASTROPHYSICAL_CONTEXT_FEATURES,
    TrainingConfig,
    save_target_model,
    score_target_model,
    train_target_model,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_RUN_DIR
    / "results"
    / "hierarchical_review_ml"
    / "stats_plus_periodicity_dip_jump_context"
)

GATE_TARGET = "human_hierarchy_gate_label"
PRIMARY_TARGET = "human_primary_morphology_label"
QUASI_PERIODIC_TARGET = "human_quasi_periodic_label"
MICROLENSING_TARGET = "human_microlensing_given_brightening_label"
LONG_TIMESCALE_TARGET = "human_long_timescale_subtype_label"

REJECTION_LABEL = "artifact_or_nonvariable"
USABLE_LABEL = "usable_astrophysical_variable"

PRIMARY_CLASS_ORDER = (
    "dipper_dimming",
    "eb_geometric_periodic",
    "long_timescale_variable",
    "brightening_transient",
    "other_structured_variable",
)

REJECTION_MORPHOLOGIES = {
    "artifact_or_bad_photometry",
    "nonvariable_or_low_snr",
}
USABLE_MORPHOLOGIES = {
    "dimming_event",
    "brightening_event",
    "mixed_dip_and_burst",
    "periodic",
    "quasi_periodic",
    "stochastic",
    "long_term_trend",
    "long_period_variability",
}
QUASI_PERIODIC_TAGS = {
    "quasi_periodic_dips",
    "quasi_periodic_bursts",
    "quasi_periodic_dimming",
    "quasi_periodic_brightening",
    "quasi_periodic_symmetric_variability",
    "quasi_periodic_spot_modulation",
    "quasi_periodic_accretion_variability",
    "quasi_periodic_long_period",
}


@dataclass(frozen=True)
class HierarchyHead:
    key: str
    target_column: str
    feature_columns: tuple[str, ...]
    config: TrainingConfig
    label_audit: dict[str, Any]


def _multiclass_config() -> TrainingConfig:
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


def _binary_config(*, min_child_samples: int = 10) -> TrainingConfig:
    return TrainingConfig(
        random_state=42,
        val_size=0.15,
        test_size=0.20,
        cv_folds=5,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=min_child_samples,
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


def build_hierarchical_labels(table: pd.DataFrame) -> dict[str, Any]:
    """Populate all hierarchy targets from the current human taxonomy."""

    reviewed = _reviewed_mask(table)
    morphology = _clean_text(table, "morphology_primary")
    physical = _clean_text(table, "physical_primary")
    event_class = _clean_text(table, "event_class")
    tags = _secondary_tag_sets(table)
    table["human_secondary_tags"] = tags.map(
        lambda values: "|".join(sorted(values))
    )

    rejection = reviewed & morphology.isin(REJECTION_MORPHOLOGIES)
    usable = reviewed & morphology.isin(USABLE_MORPHOLOGIES)

    eb_like = usable & (
        physical.eq("eclipsing_or_geometric_binary")
        | tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
    )
    microlensing_like = usable & (
        physical.eq("microlensing")
        | event_class.eq("microlensing")
        | tags.map(lambda values: "possible_microlensing_event" in values)
    )

    primary_masks = {
        "eb_geometric_periodic": eb_like,
        "dipper_dimming": usable & morphology.eq("dimming_event") & ~eb_like,
        "long_timescale_variable": usable
        & morphology.isin({"long_term_trend", "long_period_variability"})
        & ~eb_like,
        "brightening_transient": usable
        & (morphology.eq("brightening_event") | microlensing_like)
        & ~eb_like,
    }
    assigned_primary = sum(mask.astype(int) for mask in primary_masks.values())
    primary_masks["other_structured_variable"] = usable & assigned_primary.eq(0)
    primary_membership = sum(mask.astype(int) for mask in primary_masks.values())
    if bool(primary_membership.gt(1).any()):
        raise ValueError(
            "Hierarchical primary morphology is not mutually exclusive for "
            f"{int(primary_membership.gt(1).sum())} rows"
        )
    if bool((usable & primary_membership.ne(1)).any()):
        raise ValueError(
            "Hierarchical primary morphology failed to map "
            f"{int((usable & primary_membership.ne(1)).sum())} usable rows"
        )

    table[GATE_TARGET] = pd.Series(pd.NA, index=table.index, dtype="string")
    table.loc[rejection, GATE_TARGET] = REJECTION_LABEL
    table.loc[usable, GATE_TARGET] = USABLE_LABEL

    table[PRIMARY_TARGET] = pd.Series(pd.NA, index=table.index, dtype="string")
    for label, mask in primary_masks.items():
        table.loc[mask, PRIMARY_TARGET] = label

    quasi_periodic = usable & (
        morphology.eq("quasi_periodic")
        | tags.map(lambda values: bool(values & QUASI_PERIODIC_TAGS))
    )
    table[QUASI_PERIODIC_TARGET] = pd.Series(
        pd.NA, index=table.index, dtype="string"
    )
    table.loc[usable & ~quasi_periodic, QUASI_PERIODIC_TARGET] = (
        "not_quasi_periodic"
    )
    table.loc[quasi_periodic, QUASI_PERIODIC_TARGET] = "quasi_periodic"

    brightening_parent = table[PRIMARY_TARGET].eq("brightening_transient")
    table[MICROLENSING_TARGET] = pd.Series(
        pd.NA, index=table.index, dtype="string"
    )
    table.loc[
        brightening_parent & ~microlensing_like, MICROLENSING_TARGET
    ] = "not_microlensing_like"
    table.loc[
        brightening_parent & microlensing_like, MICROLENSING_TARGET
    ] = "microlensing_like"

    long_parent = table[PRIMARY_TARGET].eq("long_timescale_variable")
    table[LONG_TIMESCALE_TARGET] = pd.Series(
        pd.NA, index=table.index, dtype="string"
    )
    table.loc[
        long_parent & morphology.eq("long_term_trend"), LONG_TIMESCALE_TARGET
    ] = "long_term_variable"
    table.loc[
        long_parent & morphology.eq("long_period_variability"),
        LONG_TIMESCALE_TARGET,
    ] = "long_period_variable"

    recurrence_target, _positive_label, recurrence_audit = (
        _dipper_recurrence_labels(table)
    )

    def counts(target: str) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in table[target].dropna().value_counts().items()
        }

    audit = {
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "gate": {
            "target": GATE_TARGET,
            "class_counts": counts(GATE_TARGET),
            "n_excluded_reviewed": int(
                (reviewed & table[GATE_TARGET].isna()).sum()
            ),
        },
        "primary_morphology": {
            "target": PRIMARY_TARGET,
            "class_counts": counts(PRIMARY_TARGET),
            "parent": USABLE_LABEL,
        },
        "quasi_periodic": {
            "target": QUASI_PERIODIC_TARGET,
            "class_counts": counts(QUASI_PERIODIC_TARGET),
            "parent": USABLE_LABEL,
            "temporary_negative_definition": (
                "usable reviewed rows without a quasi-periodic primary or "
                "secondary tag"
            ),
        },
        "microlensing_given_brightening": {
            "target": MICROLENSING_TARGET,
            "class_counts": counts(MICROLENSING_TARGET),
            "parent": "brightening_transient",
        },
        "long_timescale_subtype": {
            "target": LONG_TIMESCALE_TARGET,
            "class_counts": counts(LONG_TIMESCALE_TARGET),
            "parent": "long_timescale_variable",
        },
        "dipper_recurrence": recurrence_audit,
        "recurrence_target": recurrence_target,
        "definitions": {
            "gate": "artifact/nonvariable versus usable reviewed morphology",
            "primary": (
                "dipper/dimming, EB/geometric periodic, long-timescale, "
                "brightening/transient, or other structured variable"
            ),
            "subtypes": (
                "quasi-periodic across usable variables; microlensing within "
                "brightening; LPV versus LTV within long-timescale; recurrence "
                "within explicitly recurrence-labeled dippers"
            ),
        },
    }
    return audit


def _base_feature_candidates(
    table: pd.DataFrame, trainable: pd.Series
) -> list[str]:
    stats = [
        column
        for column in table.columns
        if column.startswith("stats_")
        and column not in EIGHT_CLASS_EXCLUDED_STATS
        and is_usable_model_feature(
            table.loc[trainable, column], min_non_null=20
        )
    ]
    stats = sorted(
        stats,
        key=lambda column: (
            -int(table.loc[trainable, column].notna().sum()),
            column,
        ),
    )
    requested = [*stats, *ADDITIONAL_LC_FEATURES, *ASTROPHYSICAL_CONTEXT_FEATURES]
    missing = [column for column in requested if column not in table.columns]
    if missing:
        raise KeyError(f"Hierarchy features missing from Review: {missing}")
    return requested


def _head_features(
    table: pd.DataFrame,
    trainable: pd.Series,
    base_features: Iterable[str],
    *,
    min_non_null: int = 20,
) -> tuple[list[str], dict[str, str]]:
    usable = [
        column
        for column in base_features
        if is_usable_model_feature(
            table.loc[trainable, column], min_non_null=min_non_null
        )
    ]
    return _drop_exact_duplicate_features(table.loc[trainable], usable)


def prepare_hierarchy(
    db_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, HierarchyHead], dict[str, Any]]:
    table = load_review_population(
        db_path, keep_morphology_secondary_json=True
    )
    audit = build_hierarchical_labels(table)
    gate_trainable = table[GATE_TARGET].notna()
    base_features = _base_feature_candidates(table, gate_trainable)

    head_definitions = {
        "gate": (GATE_TARGET, _binary_config()),
        "primary_morphology": (PRIMARY_TARGET, _multiclass_config()),
        "quasi_periodic": (QUASI_PERIODIC_TARGET, _binary_config()),
        "microlensing_given_brightening": (
            MICROLENSING_TARGET,
            _binary_config(min_child_samples=6),
        ),
        "long_timescale_subtype": (
            LONG_TIMESCALE_TARGET,
            _binary_config(min_child_samples=6),
        ),
    }
    heads: dict[str, HierarchyHead] = {}
    dropped_aliases: dict[str, dict[str, str]] = {}
    for key, (target, config) in head_definitions.items():
        trainable = table[target].notna()
        features, aliases = _head_features(
            table,
            trainable,
            base_features,
            min_non_null=20,
        )
        dropped_aliases[key] = aliases
        heads[key] = HierarchyHead(
            key=key,
            target_column=target,
            feature_columns=tuple(features),
            config=config,
            label_audit=dict(audit[key]),
        )

    recurrence_target = str(audit["recurrence_target"])
    recurrence_trainable = table[recurrence_target].notna()
    recurrence_features, recurrence_aliases = _dipper_recurrence_features(
        table, recurrence_trainable
    )
    dropped_aliases["dipper_recurrence"] = recurrence_aliases
    heads["dipper_recurrence"] = HierarchyHead(
        key="dipper_recurrence",
        target_column=recurrence_target,
        feature_columns=tuple(recurrence_features),
        config=_dipper_recurrence_config(),
        label_audit=dict(audit["dipper_recurrence"]),
    )
    audit["dropped_duplicate_feature_aliases"] = dropped_aliases
    audit["base_feature_count"] = len(base_features)
    return table, heads, audit


def _probability_for_label(
    result: Any, predictions: pd.DataFrame, label: str
) -> pd.Series:
    mapping = dict(zip(result.label_classes, result.probability_columns))
    if label not in mapping:
        raise KeyError(f"Saved head lacks label {label!r}: {result.label_classes}")
    return pd.to_numeric(predictions[mapping[label]], errors="coerce")


def _write_gain_importance(result: Any, output_dir: Path) -> None:
    booster = result.model.booster_
    pd.DataFrame(
        {
            "feature": result.feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False).to_csv(
        output_dir / "feature_importance_gain.csv", index=False
    )


def train_hierarchy(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_unreviewed_n: int = 500,
) -> dict[str, Any]:
    """Train every hierarchy head and write composed all-candidate scores."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, heads, audit = prepare_hierarchy(db_path)
    head_results: dict[str, Any] = {}
    head_predictions: dict[str, pd.DataFrame] = {}
    head_summaries: dict[str, Any] = {}

    for key, head in heads.items():
        model_input = table[
            ["candidate_id", head.target_column, *head.feature_columns]
        ].copy()
        result = train_target_model(
            model_input, head.target_column, config=head.config
        )
        head_dir = out_dir / key
        save_target_model(result, head_dir)
        _write_gain_importance(result, head_dir)
        predictions = score_target_model(head_dir, model_input)
        predictions[
            [
                "candidate_id",
                "y_pred",
                "prediction_confidence",
                *result.probability_columns,
            ]
        ].to_parquet(head_dir / "all_candidates_scores.parquet", index=False)
        head_results[key] = result
        head_predictions[key] = predictions
        head_summaries[key] = {
            "target_column": head.target_column,
            "n_trainable": result.n_rows,
            "n_features": result.n_features,
            "class_counts": result.class_counts,
            "holdout_metrics": result.holdout_metrics,
            "label_audit": head.label_audit,
        }

    gate_result = head_results["gate"]
    gate_pred = head_predictions["gate"]
    primary_result = head_results["primary_morphology"]
    primary_pred = head_predictions["primary_morphology"]
    qp_result = head_results["quasi_periodic"]
    qp_pred = head_predictions["quasi_periodic"]
    micro_result = head_results["microlensing_given_brightening"]
    micro_pred = head_predictions["microlensing_given_brightening"]
    long_result = head_results["long_timescale_subtype"]
    long_pred = head_predictions["long_timescale_subtype"]
    recurrence_result = head_results["dipper_recurrence"]
    recurrence_pred = head_predictions["dipper_recurrence"]

    scores = table[
        [
            column
            for column in (
                "candidate_id",
                "asas_sn_id",
                "lc_path",
                "status",
                "workflow_status",
                "morphology_primary",
                "morphology_secondary",
                "physical_primary",
                "event_class",
                GATE_TARGET,
                PRIMARY_TARGET,
                QUASI_PERIODIC_TARGET,
                MICROLENSING_TARGET,
                LONG_TIMESCALE_TARGET,
                recurrence_result.target_column,
            )
            if column in table.columns
        ]
    ].copy()

    p_reject = _probability_for_label(
        gate_result, gate_pred, REJECTION_LABEL
    )
    p_usable = _probability_for_label(gate_result, gate_pred, USABLE_LABEL)
    scores["prob_hierarchical_artifact_or_nonvariable"] = p_reject.to_numpy()
    scores["prob_usable_astrophysical_variable"] = p_usable.to_numpy()
    scores["predicted_hierarchy_gate"] = gate_pred["y_pred"].to_numpy()

    primary_gated_columns: dict[str, str] = {}
    for label in PRIMARY_CLASS_ORDER:
        conditional = _probability_for_label(
            primary_result, primary_pred, label
        ).clip(0.0, 1.0)
        conditional_column = f"prob_primary_{label}_given_usable"
        gated_column = f"prob_{label}"
        scores[conditional_column] = conditional.to_numpy()
        scores[gated_column] = (p_usable * conditional).to_numpy()
        primary_gated_columns[label] = gated_column
    scores["predicted_primary_morphology"] = primary_pred["y_pred"].to_numpy()
    scores["predicted_hierarchical_class"] = np.where(
        scores["predicted_hierarchy_gate"].eq(REJECTION_LABEL),
        REJECTION_LABEL,
        scores["predicted_primary_morphology"],
    )

    p_qp_conditional = _probability_for_label(
        qp_result, qp_pred, "quasi_periodic"
    ).clip(0.0, 1.0)
    scores["prob_quasi_periodic_given_usable"] = p_qp_conditional.to_numpy()
    scores["prob_quasi_periodic_hierarchical"] = (
        p_usable * p_qp_conditional
    ).to_numpy()
    scores["predicted_quasi_periodic"] = np.where(
        scores["predicted_hierarchy_gate"].eq(USABLE_LABEL),
        qp_pred["y_pred"].to_numpy(),
        "not_applicable",
    )

    p_micro_conditional = _probability_for_label(
        micro_result, micro_pred, "microlensing_like"
    ).clip(0.0, 1.0)
    scores["prob_microlensing_given_brightening"] = (
        p_micro_conditional.to_numpy()
    )
    scores["prob_microlensing_hierarchical"] = (
        scores["prob_brightening_transient"] * p_micro_conditional
    )
    scores["predicted_microlensing_like"] = np.where(
        scores["predicted_hierarchical_class"].eq("brightening_transient"),
        micro_pred["y_pred"].to_numpy(),
        "not_applicable",
    )

    p_lpv_conditional = _probability_for_label(
        long_result, long_pred, "long_period_variable"
    ).clip(0.0, 1.0)
    p_ltv_conditional = _probability_for_label(
        long_result, long_pred, "long_term_variable"
    ).clip(0.0, 1.0)
    scores["prob_long_period_variable_given_long_timescale"] = (
        p_lpv_conditional.to_numpy()
    )
    scores["prob_long_term_variable_given_long_timescale"] = (
        p_ltv_conditional.to_numpy()
    )
    scores["prob_long_period_variable_hierarchical"] = (
        scores["prob_long_timescale_variable"] * p_lpv_conditional
    )
    scores["prob_long_term_variable_hierarchical"] = (
        scores["prob_long_timescale_variable"] * p_ltv_conditional
    )
    scores["predicted_long_timescale_subtype"] = np.where(
        scores["predicted_hierarchical_class"].eq("long_timescale_variable"),
        long_pred["y_pred"].to_numpy(),
        "not_applicable",
    )

    p_recurrence_conditional = _probability_for_label(
        recurrence_result, recurrence_pred, "recurrent_given_dipper"
    ).clip(0.0, 1.0)
    scores["prob_recurrent_given_dipper"] = (
        p_recurrence_conditional.to_numpy()
    )
    p_single_conditional = 1.0 - p_recurrence_conditional
    scores["prob_single_given_dipper"] = p_single_conditional.to_numpy()
    scores["prob_recurrent_dipper_hierarchical"] = (
        scores["prob_dipper_dimming"] * p_recurrence_conditional
    )
    scores["prob_single_dipper_hierarchical"] = (
        scores["prob_dipper_dimming"] * p_single_conditional
    )
    scores["predicted_dipper_recurrence"] = np.where(
        scores["predicted_hierarchical_class"].eq("dipper_dimming"),
        np.where(
            p_recurrence_conditional.ge(0.5),
            "recurrent",
            "non_recurrent",
        ),
        "not_applicable",
    )

    primary_matrix = scores[
        [primary_gated_columns[label] for label in PRIMARY_CLASS_ORDER]
    ].to_numpy(dtype=float)
    sorted_primary = np.sort(primary_matrix, axis=1)
    scores["primary_score_margin"] = (
        sorted_primary[:, -1] - sorted_primary[:, -2]
    )
    clipped = np.clip(primary_matrix, 1e-12, 1.0)
    row_sum = clipped.sum(axis=1, keepdims=True)
    conditional_primary = clipped / np.maximum(row_sum, 1e-12)
    scores["primary_prediction_entropy"] = -(
        conditional_primary * np.log(conditional_primary)
    ).sum(axis=1) / math.log(len(PRIMARY_CLASS_ORDER))
    scores["is_human_unreviewed"] = _unreviewed_mask(scores)
    scores.to_parquet(
        out_dir / "all_candidates_hierarchical_scores.parquet", index=False
    )

    queue_columns = [
        "prob_hierarchical_artifact_or_nonvariable",
        "prob_usable_astrophysical_variable",
        *[primary_gated_columns[label] for label in PRIMARY_CLASS_ORDER],
        "prob_quasi_periodic_hierarchical",
        "prob_microlensing_hierarchical",
        "prob_long_period_variable_hierarchical",
        "prob_long_term_variable_hierarchical",
        "prob_recurrent_dipper_hierarchical",
        "prob_single_dipper_hierarchical",
    ]
    unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
    queue_frames = []
    for column in queue_columns:
        queue = unreviewed.sort_values(column, ascending=False).head(
            top_unreviewed_n
        )
        queue = queue.copy()
        queue["queue_score"] = column
        queue["rank_within_queue"] = np.arange(1, len(queue) + 1)
        queue.to_csv(
            out_dir / f"top{top_unreviewed_n}_unreviewed_{column}.csv",
            index=False,
        )
        queue_frames.append(queue)
    pd.concat(queue_frames, ignore_index=True).to_csv(
        out_dir / "top_unreviewed_by_hierarchy_score.csv", index=False
    )

    summary = {
        "db_path": str(Path(db_path)),
        "model_dir": str(out_dir),
        "n_candidates": int(len(table)),
        "n_human_unreviewed": int(len(unreviewed)),
        "heads": head_summaries,
        "label_audit": audit,
        "score_columns": queue_columns,
        "warning": (
            "All LightGBM outputs are class-balanced ranking scores, not "
            "calibrated population probabilities."
        ),
    }
    (out_dir / "hierarchy_contract.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        "Trained hierarchical Review model: "
        f"{len(table):,} candidates, {len(heads)} heads -> {out_dir}"
    )
    return summary


__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_OUTPUT_DIR",
    "GATE_TARGET",
    "LONG_TIMESCALE_TARGET",
    "MICROLENSING_TARGET",
    "PRIMARY_CLASS_ORDER",
    "PRIMARY_TARGET",
    "QUASI_PERIODIC_TARGET",
    "build_hierarchical_labels",
    "prepare_hierarchy",
    "train_hierarchy",
]
