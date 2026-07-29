"""Attach the saved hierarchical ML scores to the July 1 Review database.

Run after training:

    conda run -n malca python scripts/attach_july1_hierarchical_ml.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from malca.meta_analysis.ml.july1_hierarchical_training import (
    DEFAULT_DB_PATH,
    DEFAULT_OUTPUT_DIR,
)
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    merge_hierarchical_ml_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "all_candidates_hierarchical_scores.parquet",
    )
    args = parser.parse_args()

    migrated = ensure_review_db_schema(args.db_path)
    with db_connect(args.db_path, initialize_if_missing=False) as conn:
        updated = merge_hierarchical_ml_scores(conn, args.scores_path)
    print(f"Review schema migrated: {migrated}")
    print(f"Hierarchical candidate rows updated: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
