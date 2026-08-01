"""Two-stage dimming hierarchy for the July 1 Review population.

The parent LightGBM model predicts four mutually exclusive human-review
families: dimming event, eclipsing binary, junk, and other.  A separate binary
head is trained only on dimming-family rows with an unambiguous human
recurrent/single-dip label.  This keeps recurrence conditional on the parent
dimming decision instead of making it compete directly with unrelated classes.
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
    _dipper_recurrence_config,
    _dipper_recurrence_features,
    _dipper_recurrence_labels,
    _drop_exact_duplicate_features,
    _eight_class_config,
    _eight_class_features,
    _reviewed_mask,
    _secondary_tag_sets,
    _unreviewed_mask,
    load_review_population,
)
from malca.meta_analysis.ml.review_lightgbm import (
    save_target_model,
    score_target_model,
    train_target_model,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_RUN_DIR
    / "results"
    / "dimming_hierarchy_ml"
    / "stats_plus_periodicity_dip_jump_context"
)
PARENT_TARGET = "human_four_class_parent_label"
PARENT_CLASS_ORDER = (
    "dimming_event",
    "eclipsing_binary",
    "junk",
    "other",
)
DIMMING_EVENT_CLASSES = {"dipper", "mixed_dip_and_burst"}
DIMMING_MORPHOLOGIES = {"dimming_event", "mixed_dip_and_burst"}
JUNK_MORPHOLOGIES = {
    "artifact_or_bad_photometry",
    "nonvariable_or_low_snr",
}
EXCLUDED_EVENT_CLASSES = {"", "unclassified"}

PARENT_SCORE_COLUMNS = tuple(
    f"score_parent_{label}" for label in PARENT_CLASS_ORDER
)
FINAL_SCORE_COLUMNS = (
    "score_hierarchical_recurrent_dimming_event",
    "score_hierarchical_non_recurrent_dimming_event",
    "score_parent_eclipsing_binary",
    "score_parent_junk",
    "score_parent_other",
)


def construct_four_class_parent_target(
    table: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    """Attach the four mutually exclusive parent labels."""

    reviewed = _reviewed_mask(table)
    event_class = _clean_text(table, "event_class")
    morphology_primary = _clean_text(table, "morphology_primary")
    physical_primary = _clean_text(table, "physical_primary")
    tags = _secondary_tag_sets(table)
    eligible = reviewed & ~event_class.isin(EXCLUDED_EVENT_CLASSES)

    dimming = eligible & (
        event_class.isin(DIMMING_EVENT_CLASSES)
        | morphology_primary.isin(DIMMING_MORPHOLOGIES)
    )
    eclipsing_binary = eligible & (
        tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
        | physical_primary.eq("eclipsing_or_geometric_binary")
    )
    junk = eligible & morphology_primary.isin(JUNK_MORPHOLOGIES)
    membership_count = (
        dimming.astype("int16")
        + eclipsing_binary.astype("int16")
        + junk.astype("int16")
    )
    if bool(membership_count.gt(1).any()):
        overlapping = table.loc[
            membership_count.gt(1), "candidate_id"
        ].astype(str).tolist()
        raise ValueError(
            "Established four-class parent rules overlap for "
            f"{len(overlapping)} candidates: {overlapping[:10]}"
        )

    other = eligible & ~dimming & ~eclipsing_binary & ~junk
    masks = {
        "dimming_event": dimming,
        "eclipsing_binary": eclipsing_binary,
        "junk": junk,
        "other": other,
    }
    target = pd.Series(pd.NA, index=table.index, dtype="string")
    source = pd.Series(
        "excluded_not_clear_review", index=table.index, dtype="object"
    )
    source.loc[reviewed & ~eligible] = (
        "excluded_blank_or_unclassified_event_class"
    )
    source_names = {
        "dimming_event": "existing_human_dipper_family",
        "eclipsing_binary": "existing_human_eb_review_rule",
        "junk": "existing_human_artifact_or_nonvariable_rule",
        "other": "remaining_clear_reviewed_non_dimming_class",
    }
    for label in PARENT_CLASS_ORDER:
        target.loc[masks[label]] = label
        source.loc[masks[label]] = source_names[label]

    table[PARENT_TARGET] = target
    table["four_class_parent_label_source"] = source
    table["four_class_parent_human_secondary_tags"] = tags.map(
        lambda values: "|".join(sorted(values))
    )
    counts = {
        label: int(masks[label].sum()) for label in PARENT_CLASS_ORDER
    }
    return PARENT_TARGET, {
        "target_column": PARENT_TARGET,
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "n_clear_review_eligible": int(eligible.sum()),
        "n_trainable": int(target.notna().sum()),
        "class_order": list(PARENT_CLASS_ORDER),
        "class_counts": counts,
        "n_overlap_rows": int(membership_count.gt(1).sum()),
        "n_reviewed_excluded_blank_or_unclassified": int(
            (reviewed & ~eligible).sum()
        ),
        "junk_component_counts": {
            morphology: int(
                (junk & morphology_primary.eq(morphology)).sum()
            )
            for morphology in sorted(JUNK_MORPHOLOGIES)
        },
        "label_definition": (
            "Four mutually exclusive human-review parent classes using the "
            "established dipper-family, EB, and combined artifact/nonvariable "
            "constructions; all remaining clear reviews are other."
        ),
    }


def _probability_for_label(
    result: Any,
    predictions: pd.DataFrame,
    label: str,
) -> pd.Series:
    probability_by_label = dict(
        zip(result.label_classes, result.probability_columns)
    )
    if label not in probability_by_label:
        raise KeyError(
            f"Saved model lacks label {label!r}: {result.label_classes}"
        )
    return pd.to_numeric(
        predictions[probability_by_label[label]], errors="coerce"
    ).clip(0.0, 1.0)


def _write_gain_importance(result: Any, output_dir: Path) -> None:
    booster = result.model.booster_
    pd.DataFrame(
        {
            "feature": result.feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(
        ["gain", "split"], ascending=False, ignore_index=True
    ).to_csv(output_dir / "feature_importance_gain.csv", index=False)


def compose_dimming_hierarchy_scores(
    table: pd.DataFrame,
    *,
    parent_result: Any,
    parent_predictions: pd.DataFrame,
    recurrence_result: Any,
    recurrence_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compose parent and conditional-head scores for every candidate."""

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
            PARENT_TARGET,
            "four_class_parent_label_source",
            recurrence_result.target_column,
            "human_dipper_recurrence_label_source",
            "human_dipper_recurrence_tags",
        )
        if column in table.columns
    ]
    scores = table[context_columns].copy()

    for label in PARENT_CLASS_ORDER:
        scores[f"score_parent_{label}"] = _probability_for_label(
            parent_result, parent_predictions, label
        ).to_numpy()
    scores["predicted_parent_class"] = (
        parent_predictions["y_pred"].astype(str).to_numpy()
    )

    recurrent = _probability_for_label(
        recurrence_result,
        recurrence_predictions,
        "recurrent_given_dipper",
    )
    non_recurrent = _probability_for_label(
        recurrence_result,
        recurrence_predictions,
        "non_recurrent_given_dipper",
    )
    scores["score_recurrent_given_dimming"] = recurrent.to_numpy()
    scores["score_non_recurrent_given_dimming"] = (
        non_recurrent.to_numpy()
    )
    scores["score_hierarchical_recurrent_dimming_event"] = (
        scores["score_parent_dimming_event"]
        * scores["score_recurrent_given_dimming"]
    )
    scores["score_hierarchical_non_recurrent_dimming_event"] = (
        scores["score_parent_dimming_event"]
        * scores["score_non_recurrent_given_dimming"]
    )

    parent_is_dimming = scores["predicted_parent_class"].eq(
        "dimming_event"
    )
    recurrence_class = np.where(
        recurrent.ge(non_recurrent),
        "recurrent_dimming_event",
        "non_recurrent_dimming_event",
    )
    scores["predicted_dimming_subclass"] = np.where(
        parent_is_dimming,
        recurrence_class,
        "not_applicable",
    )
    scores["predicted_hierarchical_class"] = np.where(
        parent_is_dimming,
        recurrence_class,
        scores["predicted_parent_class"],
    )

    parent_matrix = scores[list(PARENT_SCORE_COLUMNS)].to_numpy(
        dtype=float
    )
    sorted_parent = np.sort(parent_matrix, axis=1)
    scores["parent_score_margin"] = (
        sorted_parent[:, -1] - sorted_parent[:, -2]
    )
    clipped = np.clip(parent_matrix, 1e-12, 1.0)
    scores["parent_prediction_entropy"] = -(
        clipped * np.log(clipped)
    ).sum(axis=1) / math.log(len(PARENT_SCORE_COLUMNS))
    scores["recurrence_score_margin"] = (
        scores["score_recurrent_given_dimming"]
        - scores["score_non_recurrent_given_dimming"]
    ).abs()
    scores["is_human_unreviewed"] = _unreviewed_mask(scores)
    return scores


def validate_dimming_hierarchy_scores(
    scores: pd.DataFrame,
    *,
    candidate_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate score ranges, hierarchy identities, and optional coverage."""

    required = {
        "candidate_id",
        *PARENT_SCORE_COLUMNS,
        *FINAL_SCORE_COLUMNS,
        "score_recurrent_given_dimming",
        "score_non_recurrent_given_dimming",
        "predicted_parent_class",
        "predicted_dimming_subclass",
        "predicted_hierarchical_class",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Hierarchy scores are missing columns: {missing}")
    ids = scores["candidate_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError(
            "Hierarchy scores require one non-empty candidate ID per row"
        )
    score_columns = [
        *PARENT_SCORE_COLUMNS,
        "score_recurrent_given_dimming",
        "score_non_recurrent_given_dimming",
        *FINAL_SCORE_COLUMNS,
    ]
    values = scores[score_columns].to_numpy(dtype=float)
    nonfinite = int((~np.isfinite(values)).sum())
    outside_unit = int(((values < 0.0) | (values > 1.0)).sum())
    identity_errors = {
        "parent_sum": float(
            (
                scores[list(PARENT_SCORE_COLUMNS)].sum(axis=1) - 1.0
            ).abs().max()
        ),
        "conditional_dimming_subtype_sum": float(
            (
                scores["score_recurrent_given_dimming"]
                + scores["score_non_recurrent_given_dimming"]
                - 1.0
            ).abs().max()
        ),
        "gated_dimming_subtype_sum": float(
            (
                scores["score_hierarchical_recurrent_dimming_event"]
                + scores[
                    "score_hierarchical_non_recurrent_dimming_event"
                ]
                - scores["score_parent_dimming_event"]
            ).abs().max()
        ),
        "final_leaf_sum": float(
            (
                scores[list(FINAL_SCORE_COLUMNS)].sum(axis=1) - 1.0
            ).abs().max()
        ),
    }
    parent_is_dimming = scores["predicted_parent_class"].eq(
        "dimming_event"
    )
    applicability_violations = int(
        (
            scores["predicted_dimming_subclass"]
            .astype("string")
            .eq("not_applicable")
            != ~parent_is_dimming
        ).sum()
    )
    artifact_ids = set(ids.astype(str))
    missing_ids = (
        len(candidate_ids.difference(artifact_ids))
        if candidate_ids is not None
        else 0
    )
    unexpected_ids = (
        len(artifact_ids.difference(candidate_ids))
        if candidate_ids is not None
        else 0
    )
    report = {
        "n_scores": int(len(scores)),
        "n_unique_candidate_ids": int(ids.nunique()),
        "nonfinite_score_values": nonfinite,
        "score_values_outside_unit_interval": outside_unit,
        "subclass_applicability_violations": applicability_violations,
        "candidate_ids_missing_from_scores": missing_ids,
        "unexpected_candidate_ids": unexpected_ids,
        "identity_errors": identity_errors,
        "max_identity_error": max(identity_errors.values()),
    }
    if nonfinite or outside_unit or applicability_violations:
        raise ValueError(f"Invalid hierarchical score values: {report}")
    if candidate_ids is not None and artifact_ids != candidate_ids:
        raise ValueError(f"Hierarchy candidate coverage mismatch: {report}")
    if report["max_identity_error"] >= 1e-12:
        raise ValueError(f"Hierarchy score identities failed: {report}")
    return report


def train_dimming_hierarchy(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
    top_unreviewed_n: int = 500,
) -> dict[str, Any]:
    """Train the four-class parent and conditional recurrence head."""

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

    parent_target, parent_audit = construct_four_class_parent_target(
        table
    )
    recurrence_target, _positive_label, recurrence_audit = (
        _dipper_recurrence_labels(table)
    )
    parent_trainable = table[parent_target].notna()
    recurrence_trainable = table[recurrence_target].notna()

    parent_features = _eight_class_features(table, parent_trainable)
    parent_features, parent_aliases = _drop_exact_duplicate_features(
        table.loc[parent_trainable], parent_features
    )
    recurrence_features, recurrence_aliases = (
        _dipper_recurrence_features(table, recurrence_trainable)
    )
    parent_audit["dropped_duplicate_feature_aliases"] = parent_aliases
    recurrence_audit["dropped_duplicate_feature_aliases"] = (
        recurrence_aliases
    )
    parent_audit["feature_policy"] = (
        "Same stats, periodicity/dip/jump, recovery, and context block as "
        "the July 1 eight-class model; Review taxonomy fields are excluded."
    )
    recurrence_audit["hierarchy_role"] = (
        "Conditional child trained only on dimming-family rows with an "
        "unambiguous human recurrence label."
    )

    parent_input = table[
        ["candidate_id", parent_target, *parent_features]
    ].copy()
    recurrence_input = table[
        ["candidate_id", recurrence_target, *recurrence_features]
    ].copy()
    parent_result = train_target_model(
        parent_input,
        parent_target,
        config=_eight_class_config(),
    )
    recurrence_result = train_target_model(
        recurrence_input,
        recurrence_target,
        config=_dipper_recurrence_config(),
    )

    parent_dir = out_dir / "parent_four_class"
    recurrence_dir = out_dir / "recurrence_head"
    save_target_model(parent_result, parent_dir)
    save_target_model(recurrence_result, recurrence_dir)
    _write_gain_importance(parent_result, parent_dir)
    _write_gain_importance(recurrence_result, recurrence_dir)

    parent_predictions = score_target_model(
        parent_dir,
        table[["candidate_id", *parent_features]],
    )
    recurrence_predictions = score_target_model(
        recurrence_dir,
        table[["candidate_id", *recurrence_features]],
    )
    parent_predictions.to_parquet(
        parent_dir / "all_candidates_scores.parquet", index=False
    )
    recurrence_predictions.to_parquet(
        recurrence_dir / "all_candidates_scores.parquet", index=False
    )

    scores = compose_dimming_hierarchy_scores(
        table,
        parent_result=parent_result,
        parent_predictions=parent_predictions,
        recurrence_result=recurrence_result,
        recurrence_predictions=recurrence_predictions,
    )
    candidate_ids = set(table["candidate_id"].astype(str))
    validation = validate_dimming_hierarchy_scores(
        scores, candidate_ids=candidate_ids
    )
    scores.to_parquet(
        out_dir / "all_candidates_dimming_hierarchy_scores.parquet",
        index=False,
    )

    unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
    queue_columns = [
        *PARENT_SCORE_COLUMNS,
        "score_hierarchical_recurrent_dimming_event",
        "score_hierarchical_non_recurrent_dimming_event",
    ]
    queue_frames: list[pd.DataFrame] = []
    for column in queue_columns:
        queue = unreviewed.sort_values(column, ascending=False).head(
            top_unreviewed_n
        ).copy()
        queue["queue_score"] = column
        queue["rank_within_queue"] = np.arange(1, len(queue) + 1)
        queue.to_csv(
            out_dir / f"top{top_unreviewed_n}_unreviewed_{column}.csv",
            index=False,
        )
        queue_frames.append(queue)
    pd.concat(queue_frames, ignore_index=True).to_csv(
        out_dir / "top_unreviewed_by_dimming_hierarchy_score.csv",
        index=False,
    )
    unreviewed.sort_values(
        ["parent_prediction_entropy", "parent_score_margin"],
        ascending=[False, True],
    ).head(500).to_csv(
        out_dir / "most_ambiguous_parent_unreviewed.csv", index=False
    )

    label_columns = [
        column
        for column in (
            "candidate_id",
            "event_class",
            "morphology_primary",
            "morphology_secondary",
            "four_class_parent_human_secondary_tags",
            parent_target,
            "four_class_parent_label_source",
            recurrence_target,
            "human_dipper_recurrence_label_source",
            "physical_primary",
            "physical_secondary",
        )
        if column in table.columns
    ]
    table.loc[_reviewed_mask(table), label_columns].to_parquet(
        out_dir / "reviewed_hierarchy_labels.parquet", index=False
    )

    contract = {
        "parent": parent_audit,
        "recurrence_head": recurrence_audit,
        "parent_class_order": list(PARENT_CLASS_ORDER),
        "final_leaf_score_columns": list(FINAL_SCORE_COLUMNS),
        "score_composition": {
            "recurrent_dimming": (
                "score_parent_dimming_event * "
                "score_recurrent_given_dimming"
            ),
            "non_recurrent_dimming": (
                "score_parent_dimming_event * "
                "score_non_recurrent_given_dimming"
            ),
        },
        "scientific_warning": (
            "All components are uncalibrated class-balanced ranking scores. "
            "Products preserve the hierarchy but are not physical posteriors."
        ),
    }
    summary = {
        "model_key": "two_stage_dimming_hierarchy",
        "db_path": str(db_path),
        "model_dir": str(out_dir),
        "n_candidates": int(len(table)),
        "n_human_unreviewed": int(len(unreviewed)),
        "parent_four_class": {
            "n_trainable": int(parent_result.n_rows),
            "n_features": int(parent_result.n_features),
            "class_counts": parent_result.class_counts,
            "holdout_metrics": parent_result.holdout_metrics,
        },
        "recurrence_head": {
            "n_trainable": int(recurrence_result.n_rows),
            "n_features": int(recurrence_result.n_features),
            "class_counts": recurrence_result.class_counts,
            "holdout_metrics": recurrence_result.holdout_metrics,
        },
        "recovery_feature_cache": str(cache_path.expanduser().resolve()),
        "recovery_feature_schema_version": RECOVERY_FEATURE_SCHEMA_VERSION,
        "validation": validation,
        "warning": contract["scientific_warning"],
    }
    (out_dir / "hierarchy_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        "Trained two-stage dimming hierarchy: "
        f"{len(table):,} candidates, "
        f"{parent_result.n_rows:,} parent labels, "
        f"{recurrence_result.n_rows:,} recurrence labels -> {out_dir}"
    )
    return summary


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "FINAL_SCORE_COLUMNS",
    "PARENT_CLASS_ORDER",
    "PARENT_SCORE_COLUMNS",
    "PARENT_TARGET",
    "compose_dimming_hierarchy_scores",
    "construct_four_class_parent_target",
    "train_dimming_hierarchy",
    "validate_dimming_hierarchy_scores",
]
