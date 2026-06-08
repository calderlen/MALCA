from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from malca.config import DEFAULT_OUTPUT_DIR
from malca.migration.core import migrate_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m migrate",
        description="Mirror-copy MALCA outputs into the three-layer feature structure.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Input output tree or artifact to migrate. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Destination mirror tree. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Classify artifacts and write reports without creating a migrated mirror.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination if it already exists.",
    )
    parser.add_argument(
        "--prefer",
        choices=("fail", "canonical", "legacy"),
        default="fail",
        help="Conflict policy for legacy/canonical schema aliases in parquet products.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Migration report path. Default: <output>/migration_report.json.",
    )
    parser.add_argument(
        "--unclassified-columns",
        type=Path,
        default=None,
        help="Unclassified-column report path. Default: sibling of migration report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    summary = migrate_tree(
        args.input,
        args.output,
        scan_only=bool(args.scan_only),
        overwrite=bool(args.overwrite),
        prefer=args.prefer,
        report_path=args.report,
        unclassified_columns_path=args.unclassified_columns,
    )

    print(json.dumps(summary.to_dict(), sort_keys=True))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
