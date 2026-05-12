from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from malca.review.store import (
    db_connect,
    load_candidates_file,
    merge_candidate_results,
    merge_vetting_results,
)
from malca.table_io import read_parquet_table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca review-maint",
        description="Review database maintenance commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_vetting = subparsers.add_parser(
        "merge-vetting",
        help="Merge vetting results into a review DB",
    )
    merge_vetting.add_argument("--review-db", required=True, type=Path, help="Review SQLite DB")
    merge_vetting.add_argument("--input", required=True, type=Path, help="Vetting parquet file")

    merge_candidates = subparsers.add_parser(
        "merge-candidates",
        help="Merge candidate columns into a review DB",
    )
    merge_candidates.add_argument("--review-db", required=True, type=Path, help="Review SQLite DB")
    merge_candidates.add_argument("--input", required=True, type=Path, help="Candidate Parquet file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    review_db = args.review_db.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with closing(db_connect(review_db)) as conn:
        if args.command == "merge-vetting":
            df = read_parquet_table(input_path)
            updated = merge_vetting_results(conn, df)
        elif args.command == "merge-candidates":
            df = load_candidates_file(input_path)
            updated = merge_candidate_results(conn, df)
        else:  # pragma: no cover - argparse enforces choices
            raise SystemExit(f"Unknown command: {args.command}")
    print(f"Updated {updated} candidates in {review_db}")


if __name__ == "__main__":
    main()
