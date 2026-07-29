"""Attach recurrence-head predictions to the binary and eight-class dipper scores.

The recurrence model is trained only within human-labeled dippers.  This
script preserves the existing parent-model probabilities in Review and writes
parent-gated recurrent-dipper scores beside them.

Run after training the recurrence head:

    conda run -n malca python scripts/attach_july1_dipper_recurrence_ml.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from malca.io.table_io import read_parquet_table, write_parquet_table
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    merge_dipper_recurrence_ml_scores,
)


DEFAULT_RUN_DIR = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_DB_PATH = DEFAULT_RUN_DIR / "review" / "review.db"
DEFAULT_RECURRENCE_SCORES = (
    DEFAULT_RUN_DIR
    / "results"
    / "dipper_recurrence_ml"
    / "stats_plus_astrophysical_context"
    / "all_candidates_scores.parquet"
)
DEFAULT_BINARY_OVERLAY = (
    DEFAULT_RUN_DIR
    / "results"
    / "dipper_feature_selection"
    / "stats_plus_periodicity_dip_jump"
    / "all_candidates_recurrence_predictions.parquet"
)
DEFAULT_EIGHT_CLASS_OVERLAY = (
    DEFAULT_RUN_DIR
    / "results"
    / "eight_class_ml_separability"
    / "stats_plus_periodicity_dip_jump_context"
    / "all_candidates_recurrence_predictions.parquet"
)


def _require_exact_candidate_coverage(frame: pd.DataFrame, candidate_ids: set[str]) -> None:
    if "candidate_id" not in frame.columns:
        raise ValueError("Recurrence scores must include candidate_id")
    ids = frame["candidate_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated(keep=False).any():
        raise ValueError("Recurrence scores require one non-empty candidate_id per row")
    score_ids = set(ids.astype(str))
    if score_ids != candidate_ids:
        raise ValueError(
            "Recurrence scores do not match the review DB candidate set: "
            f"missing={len(candidate_ids.difference(score_ids))}, "
            f"unexpected={len(score_ids.difference(candidate_ids))}"
        )


def build_recurrence_overlays(
    recurrence_scores: pd.DataFrame,
    parent_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return binary and eight-class recurrence overlays without changing parents."""

    required = {"candidate_id", "prob_recurrent_given_dipper"}
    missing = sorted(required.difference(recurrence_scores.columns))
    if missing:
        raise ValueError(f"Recurrence score product is missing columns: {missing}")
    parents = parent_scores.set_index("candidate_id")
    recurrence = recurrence_scores.set_index("candidate_id")
    conditional = pd.to_numeric(
        recurrence["prob_recurrent_given_dipper"], errors="coerce"
    ).clip(0.0, 1.0)
    if conditional.isna().any():
        raise ValueError("Recurrence score product contains non-finite conditional scores")
    prediction = np.where(
        conditional.ge(0.5), "recurrent", "non_recurrent"
    )
    base = pd.DataFrame(
        {
            "candidate_id": recurrence.index.astype(str),
            "prob_recurrent_given_dipper": conditional.to_numpy(),
            "predicted_dipper_recurrence": prediction,
        }
    )
    binary_parent = pd.to_numeric(parents["prob_dipper_like"], errors="coerce").clip(0.0, 1.0)
    eight_parent = pd.to_numeric(parents["prob_dipper"], errors="coerce").clip(0.0, 1.0)
    if binary_parent.isna().any() or eight_parent.isna().any():
        raise ValueError("Review DB is missing parent dipper scores for recurrence attachment")

    binary = base.copy()
    binary["prob_dipper_like"] = binary["candidate_id"].map(binary_parent)
    binary["prob_recurrent_dipper_binary"] = (
        binary["prob_recurrent_given_dipper"] * binary["prob_dipper_like"]
    )
    eight_class = base.copy()
    eight_class["prob_dipper"] = eight_class["candidate_id"].map(eight_parent)
    eight_class["prob_recurrent_dipper_eight_class"] = (
        eight_class["prob_recurrent_given_dipper"] * eight_class["prob_dipper"]
    )
    return binary, eight_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--recurrence-scores", type=Path, default=DEFAULT_RECURRENCE_SCORES)
    parser.add_argument("--binary-output", type=Path, default=DEFAULT_BINARY_OVERLAY)
    parser.add_argument("--eight-class-output", type=Path, default=DEFAULT_EIGHT_CLASS_OVERLAY)
    args = parser.parse_args()

    ensure_review_db_schema(args.db_path)
    recurrence = read_parquet_table(args.recurrence_scores)
    with sqlite3.connect(f"file:{args.db_path.resolve()}?mode=ro", uri=True) as conn:
        parents = pd.read_sql_query(
            "SELECT candidate_id, prob_dipper_like, prob_dipper FROM candidates", conn
        )
    candidate_ids = set(parents["candidate_id"].astype(str))
    _require_exact_candidate_coverage(recurrence, candidate_ids)
    binary, eight_class = build_recurrence_overlays(recurrence, parents)
    args.binary_output.parent.mkdir(parents=True, exist_ok=True)
    args.eight_class_output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_table(binary, args.binary_output)
    write_parquet_table(eight_class, args.eight_class_output)
    with db_connect(args.db_path, initialize_if_missing=False) as conn:
        merge_dipper_recurrence_ml_scores(conn, args.binary_output)
        merge_dipper_recurrence_ml_scores(conn, args.eight_class_output)


if __name__ == "__main__":
    main()
