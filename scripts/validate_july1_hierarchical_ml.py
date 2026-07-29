"""Validate the saved hierarchical Review scores and optional DB merge."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.july1_hierarchical_training import (
    DEFAULT_DB_PATH,
    DEFAULT_OUTPUT_DIR,
)
from malca.review.store import (
    HIERARCHICAL_ML_PREDICTION_COLUMNS,
    HIERARCHICAL_ML_PROBABILITY_COLUMNS,
    REVIEW_DB_SCHEMA_KEY,
)


PRIMARY_CONDITIONAL_COLUMNS = (
    "prob_primary_dipper_dimming_given_usable",
    "prob_primary_eb_geometric_periodic_given_usable",
    "prob_primary_long_timescale_variable_given_usable",
    "prob_primary_brightening_transient_given_usable",
    "prob_primary_other_structured_variable_given_usable",
)
PRIMARY_GATED_COLUMNS = (
    "prob_dipper_dimming",
    "prob_eb_geometric_periodic",
    "prob_long_timescale_variable",
    "prob_brightening_transient",
    "prob_other_structured_variable",
)


def _max_abs(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").abs().max())


def validate(
    *,
    scores_path: Path,
    db_path: Path,
    require_db_values: bool,
) -> dict[str, object]:
    scores = pd.read_parquet(scores_path)
    required = {
        "candidate_id",
        *HIERARCHICAL_ML_PROBABILITY_COLUMNS,
        *HIERARCHICAL_ML_PREDICTION_COLUMNS,
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Score artifact is missing columns: {missing}")
    if scores["candidate_id"].isna().any() or scores["candidate_id"].duplicated().any():
        raise ValueError("Score artifact candidate_id values must be non-null and unique")

    probabilities = scores[list(HIERARCHICAL_ML_PROBABILITY_COLUMNS)]
    probability_values = probabilities.to_numpy(dtype=float)
    nonfinite = int((~np.isfinite(probability_values)).sum())
    outside_unit = int(((probability_values < 0) | (probability_values > 1)).sum())
    blank_predictions = 0
    for column in HIERARCHICAL_ML_PREDICTION_COLUMNS:
        blank_predictions += int(
            scores[column].isna().sum()
            + scores[column].astype("string").str.strip().eq("").sum()
        )
    subtype_parent_masks = {
        "predicted_quasi_periodic": scores["predicted_hierarchy_gate"].eq(
            "usable_astrophysical_variable"
        ),
        "predicted_microlensing_like": scores[
            "predicted_hierarchical_class"
        ].eq("brightening_transient"),
        "predicted_long_timescale_subtype": scores[
            "predicted_hierarchical_class"
        ].eq("long_timescale_variable"),
        "predicted_dipper_recurrence": scores[
            "predicted_hierarchical_class"
        ].eq("dipper_dimming"),
    }
    subtype_parent_violations = {
        column: int(
            (
                scores[column].astype("string").eq("not_applicable")
                != ~parent_mask
            ).sum()
        )
        for column, parent_mask in subtype_parent_masks.items()
    }

    identities = {
        "gate_sum": _max_abs(
            scores["prob_hierarchical_artifact_or_nonvariable"]
            + scores["prob_usable_astrophysical_variable"]
            - 1.0
        ),
        "conditional_primary_sum": _max_abs(
            scores[list(PRIMARY_CONDITIONAL_COLUMNS)].sum(axis=1) - 1.0
        ),
        "gated_primary_sum": _max_abs(
            scores[list(PRIMARY_GATED_COLUMNS)].sum(axis=1)
            - scores["prob_usable_astrophysical_variable"]
        ),
        "long_subtype_sum": _max_abs(
            scores["prob_long_period_variable_hierarchical"]
            + scores["prob_long_term_variable_hierarchical"]
            - scores["prob_long_timescale_variable"]
        ),
        "quasi_periodic_parent_product": _max_abs(
            scores["prob_quasi_periodic_hierarchical"]
            - (
                scores["prob_usable_astrophysical_variable"]
                * scores["prob_quasi_periodic_given_usable"]
            )
        ),
        "microlensing_parent_product": _max_abs(
            scores["prob_microlensing_hierarchical"]
            - (
                scores["prob_brightening_transient"]
                * scores["prob_microlensing_given_brightening"]
            )
        ),
        "recurrence_parent_product": _max_abs(
            scores["prob_recurrent_dipper_hierarchical"]
            - (
                scores["prob_dipper_dimming"]
                * scores["prob_recurrent_given_dipper"]
            )
        ),
        "single_parent_product": _max_abs(
            scores["prob_single_dipper_hierarchical"]
            - (
                scores["prob_dipper_dimming"]
                * scores["prob_single_given_dipper"]
            )
        ),
        "conditional_dipper_subtype_sum": _max_abs(
            scores["prob_recurrent_given_dipper"]
            + scores["prob_single_given_dipper"]
            - 1.0
        ),
        "gated_dipper_subtype_sum": _max_abs(
            scores["prob_recurrent_dipper_hierarchical"]
            + scores["prob_single_dipper_hierarchical"]
            - scores["prob_dipper_dimming"]
        ),
    }

    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as conn:
        db_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")
        }
        db_ids = pd.read_sql_query("SELECT candidate_id FROM candidates", conn)
        merge_columns = (
            *HIERARCHICAL_ML_PROBABILITY_COLUMNS,
            *HIERARCHICAL_ML_PREDICTION_COLUMNS,
        )
        db_has_values = set(merge_columns).issubset(db_columns)
        db_scores = (
            pd.read_sql_query(
                "SELECT candidate_id, " + ", ".join(merge_columns) + " FROM candidates",
                conn,
            )
            if db_has_values
            else None
        )
        schema_row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?",
            (REVIEW_DB_SCHEMA_KEY,),
        ).fetchone()
        schema_version = int(schema_row[0]) if schema_row is not None else None

    artifact_ids = set(scores["candidate_id"].astype(str))
    database_ids = set(db_ids["candidate_id"].astype(str))
    report: dict[str, object] = {
        "scores_path": str(scores_path),
        "db_path": str(db_path),
        "n_scores": int(len(scores)),
        "n_database_candidates": int(len(db_ids)),
        "artifact_ids_missing_from_db": len(artifact_ids - database_ids),
        "db_ids_missing_from_artifact": len(database_ids - artifact_ids),
        "n_probability_columns": len(HIERARCHICAL_ML_PROBABILITY_COLUMNS),
        "n_prediction_columns": len(HIERARCHICAL_ML_PREDICTION_COLUMNS),
        "nonfinite_probability_values": nonfinite,
        "probability_values_outside_unit_interval": outside_unit,
        "blank_prediction_values": blank_predictions,
        "subtype_parent_prediction_violations": subtype_parent_violations,
        "max_probability_identity_error": max(identities.values()),
        "probability_identity_errors": identities,
        "db_schema_version": schema_version,
        "db_has_hierarchical_columns": db_has_values,
    }
    if db_scores is not None:
        comparison = scores[
            ["candidate_id", *merge_columns]
        ].merge(
            db_scores,
            on="candidate_id",
            how="inner",
            validate="one_to_one",
            suffixes=("", "__db"),
        )
        report["db_rows_with_complete_hierarchical_scores"] = int(
            db_scores[list(HIERARCHICAL_ML_PROBABILITY_COLUMNS)]
            .notna()
            .all(axis=1)
            .sum()
        )
        report["max_artifact_db_probability_difference"] = max(
            _max_abs(comparison[column] - comparison[f"{column}__db"])
            for column in HIERARCHICAL_ML_PROBABILITY_COLUMNS
        )
        report["artifact_db_prediction_mismatches"] = sum(
            int(
                (
                    comparison[column].astype("string")
                    != comparison[f"{column}__db"].astype("string")
                ).sum()
            )
            for column in HIERARCHICAL_ML_PREDICTION_COLUMNS
        )

    if (
        nonfinite
        or outside_unit
        or blank_predictions
        or any(subtype_parent_violations.values())
    ):
        raise ValueError("Score artifact contains invalid probabilities or predictions")
    if artifact_ids != database_ids:
        raise ValueError("Score artifact and Review DB candidate sets differ")
    if max(identities.values()) >= 1e-12:
        raise ValueError(f"Hierarchical probability identities failed: {identities}")
    if require_db_values:
        if db_scores is None:
            raise ValueError("Review DB lacks hierarchical score columns")
        if report["db_rows_with_complete_hierarchical_scores"] != len(scores):
            raise ValueError("Review DB has incomplete hierarchical score coverage")
        if report["max_artifact_db_probability_difference"] >= 1e-12:
            raise ValueError("Review DB probability values differ from the artifact")
        if report["artifact_db_prediction_mismatches"]:
            raise ValueError("Review DB prediction values differ from the artifact")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "all_candidates_hierarchical_scores.parquet",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--require-db-values", action="store_true")
    args = parser.parse_args()
    report = validate(
        scores_path=args.scores_path,
        db_path=args.db_path,
        require_db_values=args.require_db_values,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
