"""Build and optionally apply exact-ID open-cluster membership sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from malca.enrichment.open_clusters import (
    OPEN_CLUSTER_OUTPUT_COLUMNS,
    OpenClusterMatchResult,
    add_open_cluster_context,
)
from malca.io.table_io import (
    is_layer_first_table,
    read_feature_table,
    read_parquet_table,
    write_feature_table,
    write_parquet_table,
)
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    replace_candidate_payload_fields,
    validate_review_db_integrity,
)


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(f"Input must be Parquet or CSV: {path}")
    if is_layer_first_table(path):
        return read_feature_table(path)
    return read_parquet_table(path)


def summarize_open_cluster_output(frame: pd.DataFrame) -> dict[str, object]:
    """Return stable coverage and classification counts for a sidecar."""
    def count_bool(column: str) -> int:
        if column not in frame:
            return 0
        return int(frame[column].fillna(False).astype(bool).sum())

    def status_counts(column: str) -> dict[str, int]:
        if column not in frame:
            return {}
        return {
            str(key): int(value)
            for key, value in frame[column].fillna("<missing>").astype(str).value_counts().items()
        }

    return {
        "rows": int(len(frame)),
        "unique_candidates": int(frame["candidate_id"].astype(str).nunique())
        if "candidate_id" in frame
        else None,
        "gaia_ids": int(frame.get("open_cluster_gaia_id", pd.Series(dtype=object)).notna().sum()),
        "ucc_listed_members": count_bool("ucc_listed_member"),
        "ucc_p50_members": count_bool("ucc_p50_member"),
        "ucc_good_members": count_bool("ucc_good_member"),
        "hr24_listed_members": count_bool("hr24_listed_member"),
        "hr24_p50_members": count_bool("hr24_p50_member"),
        "hr24_bound_members": count_bool("hr24_bound_member"),
        "hr24_high_quality_members": count_bool("hr24_high_quality_member"),
        "ucc_status_counts": status_counts("ucc_match_status"),
        "hr24_status_counts": status_counts("hr24_match_status"),
    }


def backup_review_db(review_db: Path, backup_path: Path | None = None) -> Path:
    """Create a transaction-consistent SQLite backup before applying fields."""
    if backup_path is None:
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        backup_path = review_db.with_name(
            f"{review_db.name}.pre-open-cluster-{stamp}.bak"
        )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(review_db)) as source, sqlite3.connect(str(backup_path)) as target:
        source.backup(target)
    return backup_path


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def apply_review_backfill(review_db: Path, frame: pd.DataFrame) -> int:
    """Apply only open-cluster fields to candidates present in a Review DB."""
    if "candidate_id" not in frame:
        raise ValueError("Open-cluster sidecar requires candidate_id for Review DB apply")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise ValueError("Open-cluster source sidecar contains duplicate candidate_id values")
    ensure_review_db_schema(review_db)
    updated = 0
    with db_connect(review_db, initialize_if_missing=False) as conn:
        existing_ids = {
            str(row[0]) for row in conn.execute("SELECT candidate_id FROM candidates").fetchall()
        }
        with conn:
            for _, row in frame.iterrows():
                candidate_id = str(row["candidate_id"])
                if candidate_id not in existing_ids:
                    continue
                updates = {
                    column: _json_scalar(row.get(column))
                    for column in OPEN_CLUSTER_OUTPUT_COLUMNS
                    if column in frame.columns
                }
                updated += int(
                    replace_candidate_payload_fields(
                        conn,
                        candidate_id,
                        updates,
                        commit=False,
                    )
                )
    return updated


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    tmp.replace(path)


def write_outputs(
    output_dir: Path,
    result: OpenClusterMatchResult,
    report: dict[str, object],
) -> dict[str, Path]:
    """Write canonical source output, plain all-match sidecar, and manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_membership.parquet"
    matches_path = output_dir / "all_member_matches.parquet"
    manifest_path = output_dir / "catalog_manifest.json"
    report_path = output_dir / "backfill_report.json"
    write_feature_table(result.sources, source_path)
    write_parquet_table(result.all_matches, matches_path)
    _write_json_atomic(manifest_path, result.manifest)
    _write_json_atomic(report_path, report)
    return {
        "sources": source_path,
        "matches": matches_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def run_backfill(
    frame: pd.DataFrame,
    *,
    ucc_dir: Path,
    hr24_dir: Path | None,
    include_proximity: bool,
) -> OpenClusterMatchResult:
    result = add_open_cluster_context(
        frame,
        ucc_dir=ucc_dir,
        hr24_dir=hr24_dir,
        include_proximity=include_proximity,
    )
    if len(result.sources) != len(frame):
        raise RuntimeError(
            f"Open-cluster source row count changed: input={len(frame)}, output={len(result.sources)}"
        )
    if "candidate_id" in result.sources and result.sources["candidate_id"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate candidate_id values")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join MALCA sources to pinned UCC and optional Hunt-Reffert member tables "
            "using exact Gaia DR3 source IDs."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Candidate/source Parquet or CSV")
    parser.add_argument("--ucc-dir", type=Path, required=True, help="Directory containing UCC_cat.csv and UCC_members.parquet")
    parser.add_argument("--hr24-dir", type=Path, default=None, help="Optional directory containing Hunt-Reffert clusters and members tables")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-proximity", action="store_true", help="Skip nearest reliable UCC-centre diagnostics")
    parser.add_argument("--review-db", type=Path, default=None, help="Review DB to update only when --apply is given")
    parser.add_argument("--apply", action="store_true", help="Apply the source sidecar to --review-db")
    parser.add_argument("--backup", type=Path, default=None, help="Explicit pre-apply SQLite backup path")
    parser.add_argument("--no-backup", action="store_true", help="Skip automatic SQLite backup")
    args = parser.parse_args(argv)

    input_path = args.input.expanduser()
    ucc_dir = args.ucc_dir.expanduser()
    hr24_dir = args.hr24_dir.expanduser() if args.hr24_dir is not None else None
    output_dir = args.output_dir.expanduser()
    review_db = args.review_db.expanduser() if args.review_db is not None else None
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.apply and review_db is None:
        parser.error("--apply requires --review-db")
    if review_db is not None and not review_db.exists():
        raise FileNotFoundError(review_db)
    if args.backup is not None and not args.apply:
        parser.error("--backup is only meaningful with --apply")
    if args.no_backup and args.backup is not None:
        parser.error("Choose either --backup or --no-backup")

    frame = _read_input(input_path)
    result = run_backfill(
        frame,
        ucc_dir=ucc_dir,
        hr24_dir=hr24_dir,
        include_proximity=not bool(args.no_proximity),
    )
    report: dict[str, object] = {
        "input": str(input_path.resolve()),
        "ucc_dir": str(ucc_dir.resolve()),
        "hr24_dir": str(hr24_dir.resolve()) if hr24_dir is not None else None,
        "include_proximity": not bool(args.no_proximity),
        "applied": False,
        "summary": summarize_open_cluster_output(result.sources),
    }

    if args.apply:
        backup = None
        if not args.no_backup:
            backup = backup_review_db(
                review_db,
                args.backup.expanduser() if args.backup is not None else None,
            )
            print(f"Created SQLite backup: {backup}")
        updated = apply_review_backfill(review_db, result.sources)
        integrity = validate_review_db_integrity(review_db)
        report.update(
            applied=True,
            candidates_updated=int(updated),
            review_db=str(review_db.resolve()),
            backup_path=str(backup) if backup is not None else None,
            integrity=integrity,
        )

    paths = write_outputs(output_dir, result, report)
    mode = "applied" if args.apply else "isolated"
    print(f"Open-cluster backfill {mode}: {json.dumps(report['summary'], sort_keys=True)}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
