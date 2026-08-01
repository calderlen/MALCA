"""Train the July 1 four-class parent plus dimming-recurrence hierarchy."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from malca.meta_analysis.ml.candidate_features import (
    RECOVERY_FEATURE_SCHEMA_VERSION,
    add_recovery_bounded_event_features,
    default_recovery_feature_cache,
)
from malca.meta_analysis.ml.july1_dimming_hierarchy_training import (
    DEFAULT_OUTPUT_DIR,
    construct_four_class_parent_target,
    train_dimming_hierarchy,
)
from malca.meta_analysis.ml.july1_review_training import (
    DEFAULT_DB_PATH,
    _dipper_recurrence_features,
    _dipper_recurrence_labels,
    _drop_exact_duplicate_features,
    _eight_class_features,
    load_review_population,
)


def _snapshot_review_db(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-unreviewed", type=int, default=500)
    parser.add_argument("--recovery-feature-cache", type=Path, default=None)
    parser.add_argument("--recovery-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit both label/feature contracts without fitting.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(
        list(argv) if argv is not None else None
    )
    if args.top_unreviewed < 1:
        raise SystemExit("--top-unreviewed must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_cache = (
        args.recovery_feature_cache
        if args.recovery_feature_cache is not None
        else default_recovery_feature_cache(args.db_path)
    )
    snapshot_path = output_dir / "training_review_snapshot.db"
    _snapshot_review_db(args.db_path, snapshot_path)
    (output_dir / "training_snapshot.json").write_text(
        json.dumps(
            {
                "source_db": str(args.db_path.expanduser().resolve()),
                "snapshot_db": str(snapshot_path.resolve()),
                "recovery_feature_cache": str(
                    recovery_cache.expanduser().resolve()
                ),
                "recovery_feature_schema_version": (
                    RECOVERY_FEATURE_SCHEMA_VERSION
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        table = load_review_population(
            snapshot_path, keep_morphology_secondary_json=True
        )
        table = add_recovery_bounded_event_features(
            table, recovery_cache, workers=args.recovery_workers
        )
        parent_target, parent_audit = (
            construct_four_class_parent_target(table)
        )
        recurrence_target, _positive, recurrence_audit = (
            _dipper_recurrence_labels(table)
        )
        parent_features = _eight_class_features(
            table, table[parent_target].notna()
        )
        parent_features, parent_aliases = _drop_exact_duplicate_features(
            table.loc[table[parent_target].notna()], parent_features
        )
        recurrence_features, recurrence_aliases = (
            _dipper_recurrence_features(
                table, table[recurrence_target].notna()
            )
        )
        print(
            json.dumps(
                {
                    "db_path": str(snapshot_path),
                    "n_candidates": len(table),
                    "parent": {
                        "label_audit": parent_audit,
                        "n_features": len(parent_features),
                        "dropped_duplicate_feature_aliases": (
                            parent_aliases
                        ),
                    },
                    "recurrence_head": {
                        "label_audit": recurrence_audit,
                        "n_features": len(recurrence_features),
                        "dropped_duplicate_feature_aliases": (
                            recurrence_aliases
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    train_dimming_hierarchy(
        db_path=snapshot_path,
        output_dir=output_dir,
        recovery_feature_cache=recovery_cache,
        recovery_workers=args.recovery_workers,
        top_unreviewed_n=args.top_unreviewed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
