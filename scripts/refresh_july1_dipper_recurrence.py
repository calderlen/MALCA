"""Refresh the observed recurrent/non-recurrent dipper classification in Review.

Run from the repository root:

    conda run -n malca python scripts/refresh_july1_dipper_recurrence.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    refresh_dipper_recurrence_classifications,
)


DEFAULT_DB_PATH = Path("output/runs/dat3-full-extended_2026-07-01-v4/review/review.db")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    ensure_review_db_schema(args.db_path)
    with db_connect(args.db_path, initialize_if_missing=False) as conn:
        refresh_dipper_recurrence_classifications(conn)


if __name__ == "__main__":
    main()
