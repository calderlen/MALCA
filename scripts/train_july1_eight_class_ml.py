"""Train and score the July 1 eight-class Review LightGBM model.

Run directly from the repository root:

    conda run -n malca python scripts/train_july1_eight_class_ml.py
"""

from __future__ import annotations

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
from malca.meta_analysis.ml.july1_review_training import (
    build_parser,
    script_main,
    train_review_model,
)


def _snapshot_review_db(source: Path, destination: Path) -> None:
    """Atomically snapshot Review so fitting and diagnostics share one label set."""

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


def main(argv: Iterable[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else None
    parser = build_parser("eight_class")
    args = parser.parse_args(args_list)
    if args.top_unreviewed < 1:
        raise SystemExit("--top-unreviewed must be positive")
    if args.dry_run:
        return script_main("eight_class", args_list)

    output_dir = Path(args.output_dir)
    recovery_feature_cache = (
        Path(args.recovery_feature_cache)
        if args.recovery_feature_cache is not None
        else default_recovery_feature_cache(args.db_path)
    )
    snapshot_path = output_dir / "training_review_snapshot.db"
    _snapshot_review_db(Path(args.db_path), snapshot_path)
    (output_dir / "training_snapshot.json").write_text(
        json.dumps(
            {
                "source_db": str(Path(args.db_path).expanduser().resolve()),
                "snapshot_db": str(snapshot_path.resolve()),
                "recovery_feature_cache": str(
                    recovery_feature_cache.expanduser().resolve()
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
    train_review_model(
        "eight_class",
        db_path=snapshot_path,
        output_dir=output_dir,
        top_unreviewed_n=args.top_unreviewed,
        recovery_feature_cache=recovery_feature_cache,
        recovery_workers=args.recovery_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
