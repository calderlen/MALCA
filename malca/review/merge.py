from __future__ import annotations

import argparse
from pathlib import Path

from malca.review.store import merge_review_databases


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge reviewed candidates from one MALCA review DB into another.")
    parser.add_argument("--source-review-db", required=True, help="Subset/source review DB")
    parser.add_argument("--target-review-db", required=True, help="Master/target review DB")
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Also copy unreviewed rows instead of merging reviewed rows only",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    source_db = Path(args.source_review_db).expanduser().resolve()
    target_db = Path(args.target_review_db).expanduser().resolve()

    print(f"Merging review DB {source_db}")
    print(f"  into {target_db}")
    result = merge_review_databases(
        source_db,
        target_db,
        only_reviewed=not bool(args.include_unreviewed),
    )
    print(
        "Merged {candidate_scope} candidate IDs | inserted {candidates_inserted} new candidates | "
        "reviews inserted={reviews_inserted}, updated={reviews_updated}, skipped={reviews_skipped} | "
        "history inserted={history_inserted}".format(**result)
    )


if __name__ == "__main__":
    main()
