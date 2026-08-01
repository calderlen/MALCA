"""Focused four-class -> dimming-recurrence hierarchy for July 1 Review.

Stage 1 predicts four mutually exclusive human-review morphology families:
``dimming_event``, ``eclipsing_binary``, ``junk``, and ``other``.  Stage 2 is
the established conditional dipper-recurrence model and is trained only on
dimming rows with an unambiguous recurrent or single-dip human tag.

The composed leaf scores partition the parent dimming score:

``P(recurrent dimming) = P(dimming) * P(recurrent | dimming)``

and likewise for non-recurrent dimming.  All outputs are uncalibrated ranking
scores rather than population probabilities.
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
from malca.meta_analysis.ml.july1_five_class_training import (
    DIPPER_EVENT_CLASSES,
    DIPPER_MORPHOLOGIES,
    EXCLUDED_EVENT_CLASSES,
    JUNK_MORPHOLOGIES,
    five_class_training_config,
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
    _eight_class_features,
    _reviewed_mask,
    _secondary_tag_sets,
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
    / "four_class_dimming_hierarchy_ml"
    / "stats_plus_periodicity_dip_jump_context"
)
PARENT_TARGET = "human_four_class_parent_label"
PARENT_CLASS_ORDER = (
    "dimming_event",
    "eclipsing_binary",
    "junk",
    "other",
)
LEAF_CLASS_ORDER = (
    "recurrent_dimming_event",
    "non_recurrent_dimming_event",
    "eclipsing_binary",
    "junk",
    "other",
)


def construct_four_class_parent_target(
    table: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    """Attach the four parent labels using established Review constructions."""

    reviewed = _reviewed_mask(table)
    event_class = _clean_text(table, "event_class")
    morphology = _clean_text(table, "morphology_primary")
    physical = _clean_text(table, "physical_primary")
    tags = _secondary_tag_sets(table)
    eligible = reviewed & ~event_class.isin(EXCLUDED_EVENT_CLASSES)

    dimming = eligible & (
        event_class.isin(DIPPER_EVENT_CLASSES)
        | morphology.isin(DIPPER_MORPHOLOGIES)
    )
    eclipsing_binary = eligible & (
        tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
        | physical.eq("eclipsing_or_geometric_binary")
    )
    junk = eligible & morphology.isin(JUNK_MORPHOLOGIES)
    specific = {
        "dimming_event": dimming,
        "eclipsing_binary": eclipsing_binary,
        "junk": junk,
    }
    membership_count = sum(mask.astype("int16") for mask in specific.values())
    overlap = eligible & membership_count.gt(1)
    if bool(overlap.any()):
        examples = (
            table.loc[overlap, "candidate_id"].astype(str).head(10).tolist()
        )
        raise ValueError(
            "Established four-class parent labels overlap for "
            f"{int(overlap.sum())} rows; examples={examples}"
        )

    other = eligible & membership_count.eq(0)
    masks = {**specific, "other": other}
    target = pd.Series(pd.NA, index=table.index, dtype="string")
    source = pd.Series(
        "excluded_not_clear_review", index=table.index, dtype="object"
    )
    source.loc[reviewed & ~eligible] = (
        "excluded_blank_or_unclassified_event_class"
    )
    source_names = {
        "dimming_event": "existing_binary_dipper_family_rule",
        "eclipsing_binary": "existing_human_eb_review_rule",
        "junk": "existing_human_artifact_or_nonvariable_rule",
        "other": "remaining_clear_reviewed_non_dimming_class",
    }
    for label in PARENT_CLASS_ORDER:
        target.loc[masks[label]] = label
        source.loc[masks[label]] = source_names[label]

    table[PARENT_TARGET] = target
    table["four_class_parent_label_source"] = source
    table["four_class_parent_secondary_tags"] = tags.map(
        lambda values: "|".join(sorted(values))
    )
    counts = {
        label: int(masks[label].sum()) for label in PARENT_CLASS_ORDER
    }
    audit = {
        "target_column": PARENT_TARGET,
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "n_clear_review_eligible": int(eligible.sum()),
        "n_trainable": int(target.notna().sum()),
        "class_order": list(PARENT_CLASS_ORDER),
        "class_counts": counts,
        "n_overlap_rows": int(overlap.sum()),
        "n_reviewed_excluded_blank_or_unclassified": int(
            (reviewed & ~eligible).sum()
        ),
        "dimming_definition": (
            "Existing binary dipper-family construction: reviewed clear row "
            "with event_class dipper/mixed_dip_and_burst or morphology_primary "
            "dimming_event/mixed_dip_and_burst."
        ),
        "other_definition": (
            "Remaining clear reviewed rows outside dimming, EB, and junk."
        ),
    }
    return PARENT_TARGET, audit


def build_hierarchy_targets(
    table: pd.DataFrame,
) -> tuple[str, str, dict[str, Any]]:
    """Attach parent and conditional targets and return their audit."""

    parent_target, parent_audit = construct_four_class_parent_target(table)
    recurrence_target, positive_label, recurrence_audit = (
        _dipper_recurrence_labels(table)
    )
    if positive_label != "recurrent_given_dipper":
        raise ValueError(
            f"Unexpected recurrence positive label: {positive_label}"
        )
    parent_dimming_ids = set(
        table.loc[
            table[parent_target].eq("dimming_event"), "candidate_id"
        ].astype(str)
    )
    recurrence_ids = set(
        table.loc[table[recurrence_target].notna(), "candidate_id"].astype(str)
    )
    if not recurrence_ids.issubset(parent_dimming_ids):
        unexpected = sorted(recurrence_ids - parent_dimming_ids)[:10]
        raise ValueError(
            "Conditional recurrence labels exist outside the parent dimming "
            f"class; examples={unexpected}"
        )
    audit = {
        "parent": parent_audit,
        "dimming_recurrence": recurrence_audit,
        "n_parent_dimming_without_subtype_label": int(
            len(parent_dimming_ids - recurrence_ids)
        ),
        "composition": {
            "recurrent_dimming_event": (
                "prob_dimming_event * prob_recurrent_given_dimming"
            ),
            "non_recurrent_dimming_event": (
                "prob_dimming_event * prob_non_recurrent_given_dimming"
            ),
        },
        "scientific_warning": (
            "Both heads use human morphology labels. Scores are class-balanced "
            "uncalibrated rankings, not physical or population probabilities."
        ),
    }
    return parent_target, recurrence_target, audit


def _probability_for_label(
    result: Any, predictions: pd.DataFrame, label: str
) -> pd.Series:
    mapping = dict(zip(result.label_classes, result.probability_columns))
    if label not in mapping:
        raise KeyError(
            f"Model lacks label {label!r}: {result.label_classes}"
        )
    return pd.to_numeric(predictions[mapping[label]], errors="coerce")


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


def compose_hierarchical_scores(
    parent_result: Any,
    parent_predictions: pd.DataFrame,
    recurrence_result: Any,
    recurrence_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compose all-candidate parent and conditional predictions."""

    parent = parent_predictions.set_index("candidate_id", drop=False)
    recurrence = recurrence_predictions.set_index("candidate_id", drop=False)
    if not parent.index.is_unique or not recurrence.index.is_unique:
        raise ValueError("Head predictions must have unique candidate IDs")
    if set(parent.index) != set(recurrence.index):
        raise ValueError("Parent and recurrence head candidate sets differ")
    recurrence = recurrence.reindex(parent.index)

    out = pd.DataFrame({"candidate_id": parent.index.astype(str)})
    out["predicted_parent_class"] = parent["y_pred"].astype(str).to_numpy()
    out["parent_prediction_confidence"] = pd.to_numeric(
        parent["prediction_confidence"], errors="coerce"
    ).to_numpy()
    out["conditional_recurrence_prediction"] = (
        recurrence["y_pred"].astype(str).to_numpy()
    )
    out["conditional_recurrence_confidence"] = pd.to_numeric(
        recurrence["prediction_confidence"], errors="coerce"
    ).to_numpy()

    for label in PARENT_CLASS_ORDER:
        out[f"prob_{label}"] = _probability_for_label(
            parent_result, parent, label
        ).to_numpy()
    p_recurrent = _probability_for_label(
        recurrence_result, recurrence, "recurrent_given_dipper"
    ).clip(0.0, 1.0)
    p_non_recurrent = _probability_for_label(
        recurrence_result, recurrence, "non_recurrent_given_dipper"
    ).clip(0.0, 1.0)
    out["prob_recurrent_given_dimming"] = p_recurrent.to_numpy()
    out["prob_non_recurrent_given_dimming"] = p_non_recurrent.to_numpy()
    out["prob_recurrent_dimming_event"] = (
        out["prob_dimming_event"] * out["prob_recurrent_given_dimming"]
    )
    out["prob_non_recurrent_dimming_event"] = (
        out["prob_dimming_event"]
        * out["prob_non_recurrent_given_dimming"]
    )
    out["predicted_dimming_subclass"] = np.where(
        out["predicted_parent_class"].eq("dimming_event"),
        np.where(
            out["conditional_recurrence_prediction"].eq(
                "recurrent_given_dipper"
            ),
            "recurrent_dimming_event",
            "non_recurrent_dimming_event",
        ),
        "not_applicable",
    )
    out["predicted_hierarchical_leaf"] = np.where(
        out["predicted_parent_class"].eq("dimming_event"),
        out["predicted_dimming_subclass"],
        out["predicted_parent_class"],
    )

    leaf_columns = [
        "prob_recurrent_dimming_event",
        "prob_non_recurrent_dimming_event",
        "prob_eclipsing_binary",
        "prob_junk",
        "prob_other",
    ]
    matrix = out[leaf_columns].to_numpy(dtype=float)
    sorted_matrix = np.sort(matrix, axis=1)
    out["hierarchical_leaf_score_margin"] = (
        sorted_matrix[:, -1] - sorted_matrix[:, -2]
    )
    clipped = np.clip(matrix, 1e-12, 1.0)
    out["hierarchical_leaf_entropy"] = -(
        clipped * np.log(clipped)
    ).sum(axis=1) / math.log(len(leaf_columns))
    return out.reset_index(drop=True)


def train_four_class_hierarchy(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
    top_unreviewed_n: int = 500,
) -> dict[str, Any]:
    """Train both heads and write composed scores and review queues."""

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
    parent_target, recurrence_target, audit = build_hierarchy_targets(table)

    parent_trainable = table[parent_target].notna()
    parent_features = _eight_class_features(table, parent_trainable)
    parent_features, parent_aliases = _drop_exact_duplicate_features(
        table.loc[parent_trainable], parent_features
    )
    recurrence_trainable = table[recurrence_target].notna()
    recurrence_features, recurrence_aliases = _dipper_recurrence_features(
        table, recurrence_trainable
    )
    audit["feature_policy"] = {
        "parent": (
            "Same full stats, periodicity, dip/jump, recovery, and context "
            "block as the July 1 eight-class model."
        ),
        "dimming_recurrence": (
            "Exact earlier conditional recurrence feature policy: stats plus "
            "astrophysical context only; direct detected recurrence fields "
            "are excluded."
        ),
    }
    audit["dropped_duplicate_feature_aliases"] = {
        "parent": parent_aliases,
        "dimming_recurrence": recurrence_aliases,
    }

    head_specs = {
        "parent_four_class": (
            parent_target,
            parent_features,
            five_class_training_config(),
        ),
        "dimming_recurrence": (
            recurrence_target,
            recurrence_features,
            _dipper_recurrence_config(),
        ),
    }
    results: dict[str, Any] = {}
    predictions: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for key, (target, features, config) in head_specs.items():
        model_input = table[["candidate_id", target, *features]].copy()
        result = train_target_model(model_input, target, config=config)
        head_dir = out_dir / key
        save_target_model(result, head_dir)
        _write_gain_importance(result, head_dir)
        prediction = score_target_model(
            head_dir, table[["candidate_id", *features]].copy()
        )
        prediction[
            [
                "candidate_id",
                "y_pred",
                "prediction_confidence",
                *result.probability_columns,
            ]
        ].to_parquet(
            head_dir / "all_candidates_scores.parquet", index=False
        )
        results[key] = result
        predictions[key] = prediction
        summaries[key] = {
            "target_column": target,
            "n_trainable": int(result.n_rows),
            "n_features": int(result.n_features),
            "class_counts": result.class_counts,
            "holdout_metrics": result.holdout_metrics,
        }

    composed = compose_hierarchical_scores(
        results["parent_four_class"],
        predictions["parent_four_class"],
        results["dimming_recurrence"],
        predictions["dimming_recurrence"],
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
            parent_target,
            recurrence_target,
            "four_class_parent_label_source",
            "human_dipper_recurrence_label_source",
        )
        if column in table.columns
    ]
    scores = table[context_columns].merge(
        composed, on="candidate_id", how="left", validate="one_to_one"
    )
    scores["is_human_unreviewed"] = _human_unreviewed_mask(table).to_numpy()
    score_path = out_dir / "all_candidates_four_class_hierarchical_scores.parquet"
    scores.to_parquet(score_path, index=False)

    queue_columns = [
        "prob_dimming_event",
        "prob_eclipsing_binary",
        "prob_junk",
        "prob_other",
        "prob_recurrent_dimming_event",
        "prob_non_recurrent_dimming_event",
    ]
    unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
    queues: list[pd.DataFrame] = []
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
        queues.append(queue)
    pd.concat(queues, ignore_index=True).to_csv(
        out_dir / "top_unreviewed_by_hierarchical_score.csv", index=False
    )
    unreviewed.sort_values(
        ["hierarchical_leaf_entropy", "hierarchical_leaf_score_margin"],
        ascending=[False, True],
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
            parent_target,
            recurrence_target,
            "four_class_parent_label_source",
            "human_dipper_recurrence_label_source",
        )
        if column in table.columns
    ]
    table.loc[_reviewed_mask(table), label_columns].to_parquet(
        out_dir / "reviewed_hierarchy_labels.parquet", index=False
    )
    summary = {
        "model_key": "four_class_dimming_hierarchy",
        "db_path": str(db_path),
        "model_dir": str(out_dir),
        "n_candidates": int(len(table)),
        "n_human_unreviewed": int(len(unreviewed)),
        "heads": summaries,
        "label_audit": audit,
        "score_path": str(score_path),
        "recovery_feature_cache": str(cache_path.expanduser().resolve()),
        "recovery_feature_schema_version": RECOVERY_FEATURE_SCHEMA_VERSION,
        "warning": (
            "Class-balanced LightGBM outputs and their products are "
            "uncalibrated ranking scores, not population probabilities."
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
        "Trained four-class -> dimming-recurrence hierarchy: "
        f"{len(table):,} candidates, parent={summaries['parent_four_class']['n_trainable']:,}, "
        f"subclass={summaries['dimming_recurrence']['n_trainable']:,} -> {out_dir}"
    )
    return summary


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "LEAF_CLASS_ORDER",
    "PARENT_CLASS_ORDER",
    "PARENT_TARGET",
    "build_hierarchy_targets",
    "compose_hierarchical_scores",
    "construct_four_class_parent_target",
    "train_four_class_hierarchy",
]
