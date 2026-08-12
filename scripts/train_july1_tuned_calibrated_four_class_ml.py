"""Train a separate tuned and calibrated July 1 four-class hierarchy."""

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
    default_recovery_feature_cache,
)
from malca.meta_analysis.ml.july1_review_training import DEFAULT_DB_PATH
from malca.meta_analysis.ml.july1_tuned_calibrated_four_class_training import (
    DEFAULT_OUTPUT_DIR,
    train_tuned_calibrated_four_class_hierarchy,
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


def _default_timestamped_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR.parent / f"{DEFAULT_OUTPUT_DIR.name}_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recovery-feature-cache", type=Path, default=None)
    parser.add_argument("--recovery-workers", type=int, default=4)
    parser.add_argument("--parent-iterations", type=int, default=80)
    parser.add_argument("--recurrence-iterations", type=int, default=60)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--recurrence-cv-repeats", type=int, default=3)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.parent_iterations < 1 or args.recurrence_iterations < 1:
        raise SystemExit("Search iteration counts must be positive")
    if args.cv_folds < 2:
        raise SystemExit("--cv-folds must be at least 2")
    if args.recurrence_cv_repeats < 1:
        raise SystemExit("--recurrence-cv-repeats must be at least 1")

    output_dir = args.output_dir or _default_timestamped_output()
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
                "review_db_merge_requested": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    train_tuned_calibrated_four_class_hierarchy(
        db_path=snapshot_path,
        output_dir=output_dir,
        recovery_feature_cache=recovery_cache,
        recovery_workers=args.recovery_workers,
        parent_iterations=args.parent_iterations,
        recurrence_iterations=args.recurrence_iterations,
        cv_folds=args.cv_folds,
        recurrence_cv_repeats=args.recurrence_cv_repeats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
