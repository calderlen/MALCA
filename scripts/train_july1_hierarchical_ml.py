"""Train and score the July 1 hierarchical Review LightGBM model.

Run from the repository root:

    conda run -n malca python scripts/train_july1_hierarchical_ml.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from malca.meta_analysis.ml.july1_hierarchical_training import (
    DEFAULT_DB_PATH,
    DEFAULT_OUTPUT_DIR,
    prepare_hierarchy,
    train_hierarchy,
)


def _snapshot_review_db(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".db",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_conn:
            with sqlite3.connect(temporary) as destination_conn:
                source_conn.backup(destination_conn)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-unreviewed", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.top_unreviewed < 1:
        raise SystemExit("--top-unreviewed must be positive")

    if args.dry_run:
        table, heads, audit = prepare_hierarchy(args.db_path)
        print(
            json.dumps(
                {
                    "db_path": str(args.db_path),
                    "n_candidates": len(table),
                    "heads": {
                        key: {
                            "target_column": head.target_column,
                            "n_trainable": int(
                                table[head.target_column].notna().sum()
                            ),
                            "class_counts": {
                                str(label): int(count)
                                for label, count in table[
                                    head.target_column
                                ].dropna().value_counts().items()
                            },
                            "n_features": len(head.feature_columns),
                        }
                        for key, head in heads.items()
                    },
                    "label_audit": audit,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    snapshot = args.output_dir / "training_review_snapshot.db"
    _snapshot_review_db(args.db_path, snapshot)
    (args.output_dir / "training_snapshot.json").write_text(
        json.dumps(
            {
                "source_db": str(args.db_path.expanduser().resolve()),
                "snapshot_db": str(snapshot.resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    train_hierarchy(
        db_path=snapshot,
        output_dir=args.output_dir,
        top_unreviewed_n=args.top_unreviewed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
