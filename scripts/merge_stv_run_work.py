from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from malca.config import PARQUET_CACHE_COMPRESSION
from malca.enrichment.external_lcs import (
    _cache_only_source_merge_frames,
    _cache_only_specs,
    rebuild_external_lc_table_from_cache,
)
from malca.external_lc_manifest import (
    clear_external_lc_manifest_caches,
    normalize_external_lc_file_prefix,
    read_external_lc_manifest,
    scan_external_lc_manifest_rows,
    upsert_external_lc_manifest_rows,
)
from malca.io.table_io import read_feature_table, write_feature_table
from malca.review.store import (
    TAXONOMY_VERSION,
    db_connect,
    get_candidate_payload,
    merge_candidate_results,
    replace_candidate_payload_fields,
)


REVIEWED_EMPTY = {"", "unreviewed", "none", "nan", "null"}
EXTERNAL_STATUS_FILE = "_external_lc_status.parquet"
EXTERNAL_FINAL_TABLE = "lc_events_external_lcs.parquet"
EXTERNAL_CHECKPOINT_TABLE = "external_lcs_CHECKPOINT.parquet"
IGNORED_PAYLOAD_FILL_KEYS = {
    "candidate_id",
    "source_path",
    "payload_json",
    "imported_at",
    "asas_sn_id",
    "lc_path",
    "local_lc_path",
}


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    asas_sn_id: str


@dataclass
class MergeContext:
    source_run: Path
    source_review_db: Path
    target_run: Path
    target_review_db: Path
    source_candidates: pd.DataFrame
    target_candidates: pd.DataFrame
    source_by_asas: dict[str, CandidateRecord]
    target_by_asas: dict[str, CandidateRecord]
    source_duplicate_asas: set[str]
    target_duplicate_asas: set[str]
    overlap_asas: set[str]
    source_candidate_to_target: dict[str, str]
    source_cache_id_to_target: dict[str, str]

    @property
    def ambiguous_overlap_asas(self) -> set[str]:
        return (self.source_duplicate_asas | self.target_duplicate_asas) & self.overlap_asas

    @property
    def target_external_dir(self) -> Path:
        return self.target_run / "results" / "external_lcs"

    @property
    def source_results_dir(self) -> Path:
        return self.source_run / "results"

    @property
    def target_results_dir(self) -> Path:
        return self.target_run / "results"


@dataclass
class ReviewMergeResult:
    source_reviews: int = 0
    overlapping_reviewed: int = 0
    reviews_inserted: int = 0
    reviews_updated: int = 0
    reviews_skipped: int = 0
    history_inserted: int = 0
    history_skipped: int = 0


@dataclass
class ExternalCacheResult:
    manifest_rows: int = 0
    overlapping_manifest_rows: int = 0
    files_to_copy: int = 0
    files_copied: int = 0
    files_existing: int = 0
    files_missing: int = 0
    manifest_rows_upserted: int = 0
    status_rows: int = 0
    status_rows_to_insert: int = 0
    status_rows_inserted: int = 0
    status_rows_existing: int = 0
    sources: set[str] = field(default_factory=set)
    copied_source_keys: set[str] = field(default_factory=set)


@dataclass
class CacheRebuildResult:
    input_path: Path | None = None
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    rows: int = 0
    review_db_updates: dict[str, int] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)


@dataclass
class NativeAssetResult:
    source_assets: int = 0
    overlapping_assets: int = 0
    files_to_copy: int = 0
    files_copied: int = 0
    files_existing: int = 0


@dataclass
class PayloadFillResult:
    candidates_examined: int = 0
    candidates_updated: int = 0
    fields_filled: int = 0


def _clean_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _normalize_asas_id(value: object) -> str:
    text = _clean_id(value)
    if text.startswith("stv_"):
        text = text[4:]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in REVIEWED_EMPTY or value.strip() in {"[]", "{}"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(result.all())
    except Exception:
        return False


def _positive(value: object) -> bool:
    if _missing(value):
        return False
    if isinstance(value, bool):
        return value
    try:
        return float(value) > 0
    except Exception:
        return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_updated_at(value: object) -> datetime:
    text = _clean_id(value)
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _read_db_table(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Review DB not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, table):
            return pd.DataFrame()
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _candidate_records(df: pd.DataFrame) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for _, row in df.iterrows():
        candidate_id = _clean_id(row.get("candidate_id"))
        if not candidate_id:
            continue
        asas_sn_id = _normalize_asas_id(row.get("asas_sn_id")) or _normalize_asas_id(candidate_id)
        if not asas_sn_id:
            continue
        records.append(CandidateRecord(candidate_id=candidate_id, asas_sn_id=asas_sn_id))
    return records


def _unique_by_asas(records: Iterable[CandidateRecord]) -> tuple[dict[str, CandidateRecord], set[str]]:
    rows = list(records)
    counts = Counter(record.asas_sn_id for record in rows)
    duplicate_asas = {asas for asas, count in counts.items() if count > 1}
    unique = {
        record.asas_sn_id: record
        for record in rows
        if record.asas_sn_id not in duplicate_asas
    }
    return unique, duplicate_asas


def build_merge_context(
    source_run: Path,
    source_review_db: Path,
    target_run: Path,
    target_review_db: Path,
) -> MergeContext:
    source_run = source_run.expanduser().resolve()
    source_review_db = source_review_db.expanduser().resolve()
    target_run = target_run.expanduser().resolve()
    target_review_db = target_review_db.expanduser().resolve()

    source_candidates = _read_db_table(source_review_db, "candidates")
    target_candidates = _read_db_table(target_review_db, "candidates")

    source_by_asas, source_duplicate_asas = _unique_by_asas(_candidate_records(source_candidates))
    target_by_asas, target_duplicate_asas = _unique_by_asas(_candidate_records(target_candidates))
    overlap_asas = set(source_by_asas) & set(target_by_asas)

    source_candidate_to_target: dict[str, str] = {}
    source_cache_id_to_target: dict[str, str] = {}
    for asas in sorted(overlap_asas):
        source_record = source_by_asas[asas]
        target_record = target_by_asas[asas]
        source_candidate_to_target[source_record.candidate_id] = target_record.candidate_id
        source_cache_id_to_target[source_record.candidate_id] = target_record.candidate_id
        source_cache_id_to_target[source_record.asas_sn_id] = target_record.candidate_id

    return MergeContext(
        source_run=source_run,
        source_review_db=source_review_db,
        target_run=target_run,
        target_review_db=target_review_db,
        source_candidates=source_candidates,
        target_candidates=target_candidates,
        source_by_asas=source_by_asas,
        target_by_asas=target_by_asas,
        source_duplicate_asas=source_duplicate_asas,
        target_duplicate_asas=target_duplicate_asas,
        overlap_asas=overlap_asas,
        source_candidate_to_target=source_candidate_to_target,
        source_cache_id_to_target=source_cache_id_to_target,
    )


def _review_status(row: pd.Series) -> str:
    workflow = _clean_id(row.get("workflow_status"))
    if workflow:
        return workflow
    return _clean_id(row.get("status"))


def _reviewed_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return df.apply(lambda row: _review_status(row).strip().lower() not in REVIEWED_EMPTY, axis=1)


def _target_review_is_reviewed(row: pd.Series | None) -> bool:
    return row is not None and _review_status(row).strip().lower() not in REVIEWED_EMPTY


def _remapped_source_reviews(ctx: MergeContext) -> pd.DataFrame:
    reviews = _read_db_table(ctx.source_review_db, "reviews")
    if reviews.empty or "candidate_id" not in reviews.columns:
        return pd.DataFrame()
    reviewed = reviews[_reviewed_mask(reviews)].copy()
    reviewed["candidate_id"] = reviewed["candidate_id"].map(lambda value: ctx.source_candidate_to_target.get(_clean_id(value), ""))
    return reviewed[reviewed["candidate_id"].astype(str).str.len() > 0].copy()


def _remapped_source_history(ctx: MergeContext, source_review_ids: set[str]) -> pd.DataFrame:
    history = _read_db_table(ctx.source_review_db, "review_history")
    if history.empty or "candidate_id" not in history.columns:
        return pd.DataFrame(columns=["candidate_id", "event_type", "payload_json", "reviewer", "created_at"])
    history = history[history["candidate_id"].astype(str).map(_clean_id).isin(source_review_ids)].copy()
    history["candidate_id"] = history["candidate_id"].map(lambda value: ctx.source_candidate_to_target.get(_clean_id(value), ""))
    history = history[history["candidate_id"].astype(str).str.len() > 0].copy()
    for column in ["candidate_id", "event_type", "payload_json", "reviewer", "created_at"]:
        if column not in history.columns:
            history[column] = None
    return history[["candidate_id", "event_type", "payload_json", "reviewer", "created_at"]].copy()


def merge_reviews(ctx: MergeContext, *, apply: bool) -> ReviewMergeResult:
    source_reviews_raw = _read_db_table(ctx.source_review_db, "reviews")
    remapped = _remapped_source_reviews(ctx)
    source_review_ids = {
        _clean_id(cid)
        for cid in source_reviews_raw.loc[_reviewed_mask(source_reviews_raw), "candidate_id"].tolist()
        if _clean_id(cid) in ctx.source_candidate_to_target
    } if not source_reviews_raw.empty and "candidate_id" in source_reviews_raw.columns else set()
    history = _remapped_source_history(ctx, source_review_ids)

    result = ReviewMergeResult(
        source_reviews=len(source_reviews_raw),
        overlapping_reviewed=len(remapped),
    )
    if remapped.empty:
        return result

    target_reviews = _read_db_table(ctx.target_review_db, "reviews")
    target_review_map = {
        _clean_id(row["candidate_id"]): row
        for _, row in target_reviews.iterrows()
        if _clean_id(row.get("candidate_id"))
    }

    decisions: list[tuple[str, pd.Series, str]] = []
    for _, row in remapped.iterrows():
        candidate_id = _clean_id(row.get("candidate_id"))
        if not candidate_id:
            continue
        target_row = target_review_map.get(candidate_id)
        target_reviewed = _target_review_is_reviewed(target_row)
        source_updated = _parse_updated_at(row.get("updated_at"))
        target_updated = _parse_updated_at(target_row.get("updated_at")) if target_row is not None else datetime.min.replace(tzinfo=timezone.utc)
        if target_row is None:
            result.reviews_inserted += 1
            decisions.append(("insert", row, candidate_id))
        elif not target_reviewed or source_updated > target_updated:
            result.reviews_updated += 1
            decisions.append(("update", row, candidate_id))
        else:
            result.reviews_skipped += 1

    if not apply:
        with sqlite3.connect(ctx.target_review_db) as conn:
            existing_history = _existing_history_entries(conn)
        for _, row in history.iterrows():
            entry = _history_entry(row)
            if entry in existing_history:
                result.history_skipped += 1
            else:
                result.history_inserted += 1
        return result

    with db_connect(ctx.target_review_db) as conn:
        review_cols = _table_columns(conn, "reviews")
        common_cols = [col for col in remapped.columns if col in review_cols and col != "candidate_id"]
        for col in ("workflow_status", "taxonomy_version", "updated_at", "event_class", "status", "review_pass"):
            if col in review_cols and col not in common_cols:
                common_cols.append(col)

        for _, row, candidate_id in decisions:
            row_values = _review_row_values(row, common_cols)
            insert_cols = ["candidate_id", *common_cols]
            placeholders = ", ".join(["?"] * len(insert_cols))
            update_cols = [col for col in insert_cols if col != "candidate_id"]
            conflict_set = ", ".join(f"{col}=excluded.{col}" for col in update_cols)
            conn.execute(
                f"""
                INSERT INTO reviews ({', '.join(insert_cols)})
                VALUES ({placeholders})
                ON CONFLICT(candidate_id) DO UPDATE SET {conflict_set}
                """,
                (candidate_id, *(row_values[col] for col in common_cols)),
            )

        existing_history = _existing_history_entries(conn)
        for _, row in history.iterrows():
            entry = _history_entry(row)
            if entry in existing_history:
                result.history_skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO review_history (candidate_id, event_type, payload_json, reviewer, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                entry,
            )
            existing_history.add(entry)
            result.history_inserted += 1
        conn.commit()
    return result


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _review_row_values(row: pd.Series, columns: list[str]) -> dict[str, object]:
    values: dict[str, object] = {}
    for col in columns:
        value = row.get(col)
        if _missing(value):
            value = None
        values[col] = value
    if "workflow_status" in values and not values["workflow_status"]:
        values["workflow_status"] = row.get("status") or "reviewed"
    if "taxonomy_version" in values and not values["taxonomy_version"]:
        values["taxonomy_version"] = TAXONOMY_VERSION
    if "updated_at" in values and not values["updated_at"]:
        values["updated_at"] = _utc_now()
    if "event_class" in values and not values["event_class"]:
        values["event_class"] = "unclassified"
    if "status" in values and not values["status"]:
        values["status"] = values.get("workflow_status") or "reviewed"
    if "review_pass" in values and not values["review_pass"]:
        values["review_pass"] = 1
    return values


def _history_entry(row: pd.Series) -> tuple[str, str, str, object, str]:
    reviewer = row.get("reviewer")
    return (
        _clean_id(row.get("candidate_id")),
        _clean_id(row.get("event_type")) or "save",
        _clean_id(row.get("payload_json")) or "{}",
        None if _missing(reviewer) else str(reviewer),
        _clean_id(row.get("created_at")) or _utc_now(),
    )


def _existing_history_entries(conn: sqlite3.Connection) -> set[tuple[str, str, str, object, str]]:
    if not _table_exists(conn, "review_history"):
        return set()
    return {
        tuple(row)
        for row in conn.execute(
            "SELECT candidate_id, event_type, payload_json, reviewer, created_at FROM review_history"
        ).fetchall()
    }


def _manifest_roots(ctx: MergeContext) -> list[Path]:
    roots = [ctx.source_results_dir]
    nested = ctx.source_results_dir / "external_lcs"
    if nested.exists():
        roots.append(nested)
    return roots


def _read_source_manifest(ctx: MergeContext) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in _manifest_roots(ctx):
        manifest = read_external_lc_manifest(root)
        if manifest.empty:
            rows = scan_external_lc_manifest_rows(root)
            manifest = pd.DataFrame(rows)
        if manifest.empty:
            continue
        manifest = manifest.copy()
        manifest["_manifest_root"] = str(root)
        frames.append(manifest)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _source_path_from_manifest_row(row: pd.Series) -> Path:
    path_text = _clean_id(row.get("path"))
    if path_text:
        path = Path(path_text).expanduser()
        if path.exists():
            return path
    root = Path(_clean_id(row.get("_manifest_root"))).expanduser()
    rel = _clean_id(row.get("path_relative"))
    if rel:
        return root / rel
    return path if path_text else root


def _canonical_source(row: pd.Series) -> str:
    source = normalize_external_lc_file_prefix(row.get("source"))
    if source:
        return source
    return normalize_external_lc_file_prefix(row.get("file_prefix"))


def _filename_prefix(source: str) -> str:
    source = normalize_external_lc_file_prefix(source)
    return f"{source}_lc" if source else ""


def _manifest_row(root: Path, path: Path, candidate_id: str, source: str) -> dict[str, object]:
    stat = path.stat()
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
        path_text = str(path.resolve())
    except Exception:
        relative = path.name
        path_text = str(path)
    return {
        "candidate_id": candidate_id,
        "source": normalize_external_lc_file_prefix(source),
        "file_prefix": normalize_external_lc_file_prefix(source),
        "path": path_text,
        "path_relative": relative,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "updated_unix": float(time.time()),
    }


def _read_status_frames(root: Path) -> pd.DataFrame:
    paths: list[Path] = []
    direct = root / EXTERNAL_STATUS_FILE
    if direct.exists():
        paths.append(direct)
    if root.exists():
        for path in sorted(root.rglob(EXTERNAL_STATUS_FILE)):
            if path not in paths:
                paths.append(path)
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"External status warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _status_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        _clean_id(row.get("module")),
        _clean_id(row.get("candidate_id")),
        _clean_id(row.get("cache_key")),
    )


def merge_external_cache(ctx: MergeContext, *, apply: bool) -> ExternalCacheResult:
    result = ExternalCacheResult()
    manifest = _read_source_manifest(ctx)
    result.manifest_rows = len(manifest)
    target_external = ctx.target_external_dir
    manifest_rows: list[dict[str, object]] = []
    copy_jobs: list[tuple[Path, Path, str, str]] = []
    planned_destinations: set[Path] = set()

    for _, row in manifest.iterrows():
        old_id = _clean_id(row.get("candidate_id"))
        target_id = ctx.source_cache_id_to_target.get(old_id) or ctx.source_cache_id_to_target.get(_normalize_asas_id(old_id))
        if not target_id:
            continue
        result.overlapping_manifest_rows += 1
        source = _canonical_source(row)
        prefix = _filename_prefix(source)
        if not prefix:
            continue
        source_path = _source_path_from_manifest_row(row)
        if not source_path.exists():
            result.files_missing += 1
            continue
        dest_path = target_external / f"{prefix}_{target_id}.parquet"
        result.sources.add(source)
        result.copied_source_keys.add(source)
        if dest_path.exists():
            result.files_existing += 1
        elif dest_path not in planned_destinations:
            result.files_to_copy += 1
            copy_jobs.append((source_path, dest_path, target_id, source))
            planned_destinations.add(dest_path)
        manifest_rows.append(_manifest_row_for_existing_or_planned(target_external, dest_path, target_id, source))

    source_status = _read_status_frames(ctx.source_results_dir)
    result.status_rows = len(source_status)
    status_to_insert = _remap_status_rows(ctx, source_status)
    result.status_rows_to_insert, result.status_rows_existing = _count_new_status_rows(ctx, status_to_insert)

    if not apply:
        result.manifest_rows_upserted = len(manifest_rows)
        return result

    target_external.mkdir(parents=True, exist_ok=True)
    for source_path, dest_path, _, _ in copy_jobs:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        result.files_copied += 1

    materialized_manifest_rows = [
        _manifest_row(target_external, row_path, candidate_id, source)
        for row_path, candidate_id, source in {
            (target_external / Path(row["path_relative"]), str(row["candidate_id"]), str(row["source"]))
            for row in manifest_rows
            if (target_external / Path(str(row["path_relative"]))).exists()
        }
    ]
    if materialized_manifest_rows:
        _backup_if_exists(target_external / "external_lc_manifest.parquet")
        upsert_external_lc_manifest_rows(target_external, materialized_manifest_rows)
        clear_external_lc_manifest_caches()
        result.manifest_rows_upserted = len(materialized_manifest_rows)

    if not status_to_insert.empty:
        inserted = _write_merged_status_rows(ctx, status_to_insert)
        result.status_rows_inserted = inserted
    return result


def _manifest_row_for_existing_or_planned(root: Path, path: Path, candidate_id: str, source: str) -> dict[str, object]:
    try:
        relative = str(path.relative_to(root))
    except Exception:
        relative = path.name
    return {
        "candidate_id": candidate_id,
        "source": normalize_external_lc_file_prefix(source),
        "file_prefix": normalize_external_lc_file_prefix(source),
        "path": str(path),
        "path_relative": relative,
        "size_bytes": 0,
        "mtime_ns": -1,
        "updated_unix": float(time.time()),
    }


def _remap_status_rows(ctx: MergeContext, status: pd.DataFrame) -> pd.DataFrame:
    if status.empty or "candidate_id" not in status.columns:
        return pd.DataFrame()
    rows = status.copy()
    rows["candidate_id"] = rows["candidate_id"].map(
        lambda value: ctx.source_cache_id_to_target.get(_clean_id(value))
        or ctx.source_cache_id_to_target.get(_normalize_asas_id(value))
        or ""
    )
    rows = rows[rows["candidate_id"].astype(str).str.len() > 0].copy()
    return rows


def _count_new_status_rows(ctx: MergeContext, remapped: pd.DataFrame) -> tuple[int, int]:
    if remapped.empty:
        return (0, 0)
    existing = _read_target_status(ctx)
    existing_keys = {_status_key(row) for _, row in existing.iterrows()} if not existing.empty else set()
    new_count = 0
    existing_count = 0
    for _, row in remapped.iterrows():
        key = _status_key(row)
        if key in existing_keys:
            existing_count += 1
        else:
            new_count += 1
            existing_keys.add(key)
    return new_count, existing_count


def _read_target_status(ctx: MergeContext) -> pd.DataFrame:
    path = ctx.target_external_dir / EXTERNAL_STATUS_FILE
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _write_merged_status_rows(ctx: MergeContext, remapped: pd.DataFrame) -> int:
    target_status_path = ctx.target_external_dir / EXTERNAL_STATUS_FILE
    existing = _read_target_status(ctx)
    existing_keys = {_status_key(row) for _, row in existing.iterrows()} if not existing.empty else set()
    new_rows = []
    for _, row in remapped.iterrows():
        key = _status_key(row)
        if key in existing_keys:
            continue
        new_rows.append(row)
        existing_keys.add(key)
    if not new_rows:
        return 0
    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False) if not existing.empty else pd.DataFrame(new_rows)
    target_status_path.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_exists(target_status_path)
    out.to_parquet(target_status_path, index=False, compression=PARQUET_CACHE_COMPRESSION)
    return len(new_rows)


def _target_external_input_path(ctx: MergeContext) -> Path | None:
    candidates = [
        ctx.target_external_dir / EXTERNAL_CHECKPOINT_TABLE,
        ctx.target_results_dir / EXTERNAL_FINAL_TABLE,
        ctx.target_results_dir / "lc_events_vetted.parquet",
        ctx.target_results_dir / "lc_events_spectra.parquet",
        ctx.target_results_dir / "lc_events_neighbors.parquet",
        ctx.target_results_dir / "lc_events_classified.parquet",
        ctx.target_results_dir / "lc_events_characterized.parquet",
        ctx.target_results_dir / "lc_events_filtered.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def rebuild_external_summaries(
    ctx: MergeContext,
    *,
    source_keys: set[str] | None = None,
    apply: bool,
) -> CacheRebuildResult:
    if not apply:
        return CacheRebuildResult()
    input_path = _target_external_input_path(ctx)
    if input_path is None:
        return CacheRebuildResult()
    df_input = read_feature_table(input_path)
    source_keys = {normalize_external_lc_file_prefix(source) for source in (source_keys or set()) if source}
    if not source_keys:
        manifest = read_external_lc_manifest(ctx.target_external_dir)
        source_keys = {normalize_external_lc_file_prefix(source) for source in manifest.get("source", pd.Series(dtype=object)).dropna()}
    specs = _cache_only_specs()
    run_flags = {key: key in source_keys for key in specs}
    if not any(run_flags.values()):
        return CacheRebuildResult(input_path=input_path)

    out = rebuild_external_lc_table_from_cache(df_input, ctx.target_external_dir, run_flags)
    out = _preserve_target_external_values(df_input, out, run_flags)
    final_path = ctx.target_results_dir / EXTERNAL_FINAL_TABLE
    checkpoint_path = ctx.target_external_dir / EXTERNAL_CHECKPOINT_TABLE
    _backup_if_exists(final_path)
    _backup_if_exists(checkpoint_path)
    write_feature_table(out, final_path)
    write_feature_table(out, checkpoint_path)

    source_frames = _cache_only_source_merge_frames(out, run_flags)
    updates = _merge_external_frames_target_missing_only(ctx.target_review_db, source_frames, run_flags)
    return CacheRebuildResult(
        input_path=input_path,
        output_path=final_path,
        checkpoint_path=checkpoint_path,
        rows=len(out),
        review_db_updates=updates,
        sources={source for source, enabled in run_flags.items() if enabled},
    )


def _preserve_target_external_values(
    original: pd.DataFrame,
    rebuilt: pd.DataFrame,
    run_flags: dict[str, bool],
) -> pd.DataFrame:
    if original.empty:
        return rebuilt
    out = rebuilt.copy()
    specs = _cache_only_specs()
    touched = out.attrs.get("_external_lc_cache_only_touched", {})
    candidate_ids = out["candidate_id"].astype(str) if "candidate_id" in out.columns else pd.Series("", index=out.index)
    original_by_candidate = original.set_index(original["candidate_id"].astype(str), drop=False) if "candidate_id" in original.columns else pd.DataFrame()
    for source, enabled in run_flags.items():
        if not enabled or source not in specs:
            continue
        spec = specs[source]
        summary_cols = [col for col in spec["summary_cols"] if col in out.columns]
        if not summary_cols:
            continue
        touched_ids = {str(value) for value in touched.get(source, [])}
        marker = spec["match_col"]
        for idx, candidate_id in candidate_ids.items():
            if candidate_id not in original_by_candidate.index:
                continue
            original_row = original_by_candidate.loc[candidate_id]
            if isinstance(original_row, pd.DataFrame):
                original_row = original_row.iloc[-1]
            preserve = candidate_id not in touched_ids
            if not preserve and marker in original_row and marker in out.columns:
                preserve = _positive(original_row.get(marker)) and not _positive(out.loc[idx, marker])
            if not preserve:
                continue
            for col in summary_cols:
                if col in original_row:
                    out.loc[idx, col] = original_row.get(col)
    out.attrs.update(rebuilt.attrs)
    return out


def _merge_external_frames_target_missing_only(
    review_db: Path,
    source_frames: list[tuple[str, pd.DataFrame]],
    run_flags: dict[str, bool],
) -> dict[str, int]:
    if not source_frames:
        return {}
    specs = _cache_only_specs()
    updates: dict[str, int] = {}
    with db_connect(review_db) as conn:
        target_payloads: dict[str, dict] = {}
        for source, frame in source_frames:
            spec = specs.get(source)
            if spec is None or not run_flags.get(source):
                continue
            marker = spec["match_col"]
            keep_rows = []
            for _, row in frame.iterrows():
                candidate_id = _clean_id(row.get("candidate_id"))
                if not candidate_id:
                    continue
                if candidate_id not in target_payloads:
                    target_payloads[candidate_id] = get_candidate_payload(conn, candidate_id)
                if _positive(target_payloads[candidate_id].get(marker)):
                    continue
                keep_rows.append(row)
            if keep_rows:
                updates[source] = merge_candidate_results(conn, pd.DataFrame(keep_rows))
            else:
                updates[source] = 0
    return updates


def copy_native_assets(ctx: MergeContext, *, apply: bool) -> NativeAssetResult:
    source_dir = ctx.source_run / "bundle_assets" / "lightcurves"
    target_dir = ctx.target_run / "bundle_assets" / "lightcurves"
    result = NativeAssetResult()
    if not source_dir.exists():
        return result
    files = sorted(path for path in source_dir.glob("*.dat*") if path.is_file())
    result.source_assets = len(files)
    overlap = ctx.overlap_asas
    jobs: list[tuple[Path, Path]] = []
    for source_path in files:
        asas = _normalize_asas_id(source_path.stem)
        if asas not in overlap:
            continue
        result.overlapping_assets += 1
        dest_path = target_dir / source_path.name
        if dest_path.exists():
            result.files_existing += 1
        else:
            result.files_to_copy += 1
            jobs.append((source_path, dest_path))
    if apply and jobs:
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path, dest_path in jobs:
            shutil.copy2(source_path, dest_path)
            result.files_copied += 1
    return result


def merge_missing_candidate_payload_fields(ctx: MergeContext, *, apply: bool) -> PayloadFillResult:
    result = PayloadFillResult(candidates_examined=len(ctx.source_candidate_to_target))
    if not ctx.source_candidate_to_target:
        return result
    with db_connect(ctx.source_review_db) as source_conn, db_connect(ctx.target_review_db) as target_conn:
        for source_candidate_id, target_candidate_id in sorted(ctx.source_candidate_to_target.items()):
            source_payload = get_candidate_payload(source_conn, source_candidate_id)
            target_payload = get_candidate_payload(target_conn, target_candidate_id)
            updates: dict[str, object] = {}
            for key, value in source_payload.items():
                if key in IGNORED_PAYLOAD_FILL_KEYS or key.startswith("_"):
                    continue
                if _missing(value) or not _missing(target_payload.get(key)):
                    continue
                updates[key] = value
            if not updates:
                continue
            result.candidates_updated += 1
            result.fields_filled += len(updates)
            if apply:
                replace_candidate_payload_fields(
                    target_conn,
                    target_candidate_id,
                    updates,
                    commit=False,
                )
        if apply:
            target_conn.commit()
    return result


def _backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    suffix = time.strftime(".pre_merge_%Y%m%d-%H%M%S.bak")
    backup = path.with_name(path.name + suffix)
    shutil.copy2(path, backup)
    return backup


def _print_context(ctx: MergeContext) -> None:
    print("STV run work-reuse merge")
    print(f"  source run:      {ctx.source_run}")
    print(f"  source review:   {ctx.source_review_db}")
    print(f"  target run:      {ctx.target_run}")
    print(f"  target review:   {ctx.target_review_db}")
    print()
    print("Candidates")
    print(f"  source candidates:       {len(ctx.source_candidates)}")
    print(f"  target candidates:       {len(ctx.target_candidates)}")
    print(f"  ASAS overlap:            {len(ctx.overlap_asas)}")
    print(f"  source duplicate ASAS:   {len(ctx.source_duplicate_asas)}")
    print(f"  target duplicate ASAS:   {len(ctx.target_duplicate_asas)}")
    print(f"  ambiguous overlap ASAS:  {len(ctx.ambiguous_overlap_asas)}")


def _print_review_result(result: ReviewMergeResult, *, apply: bool) -> None:
    verb = "Applied" if apply else "Dry run"
    print()
    print(f"Reviews ({verb})")
    print(f"  source review rows:      {result.source_reviews}")
    print(f"  overlapping reviewed:    {result.overlapping_reviewed}")
    print(f"  reviews to insert:       {result.reviews_inserted}")
    print(f"  reviews to update:       {result.reviews_updated}")
    print(f"  reviews skipped:         {result.reviews_skipped}")
    print(f"  history to insert:       {result.history_inserted}")
    print(f"  history skipped:         {result.history_skipped}")


def _print_external_result(result: ExternalCacheResult, *, apply: bool) -> None:
    verb = "Applied" if apply else "Dry run"
    print()
    print(f"External LC caches ({verb})")
    print(f"  source manifest rows:    {result.manifest_rows}")
    print(f"  overlapping rows:        {result.overlapping_manifest_rows}")
    print(f"  files to copy:           {result.files_to_copy}")
    print(f"  files copied:            {result.files_copied}")
    print(f"  files already present:   {result.files_existing}")
    print(f"  missing source files:    {result.files_missing}")
    print(f"  manifest rows upserted:  {result.manifest_rows_upserted}")
    print(f"  source status rows:      {result.status_rows}")
    print(f"  status rows to insert:   {result.status_rows_to_insert}")
    print(f"  status rows inserted:    {result.status_rows_inserted}")
    print(f"  status rows existing:    {result.status_rows_existing}")
    if result.sources:
        print(f"  sources:                 {', '.join(sorted(result.sources))}")


def _print_rebuild_result(result: CacheRebuildResult) -> None:
    print()
    print("External LC cache-only rebuild")
    if result.input_path is None:
        print("  skipped: no candidate/external table input found")
        return
    if result.output_path is None:
        print(f"  skipped: no enabled external sources for {result.input_path}")
        return
    print(f"  input:                   {result.input_path}")
    print(f"  output:                  {result.output_path}")
    print(f"  checkpoint:              {result.checkpoint_path}")
    print(f"  rows:                    {result.rows}")
    print(f"  review DB updates:       {result.review_db_updates}")


def _print_native_result(result: NativeAssetResult, *, apply: bool) -> None:
    verb = "Applied" if apply else "Dry run"
    print()
    print(f"Native ASAS-SN assets ({verb})")
    print(f"  source assets:           {result.source_assets}")
    print(f"  overlapping assets:      {result.overlapping_assets}")
    print(f"  files to copy:           {result.files_to_copy}")
    print(f"  files copied:            {result.files_copied}")
    print(f"  files already present:   {result.files_existing}")


def _print_payload_result(result: PayloadFillResult, *, apply: bool) -> None:
    verb = "Applied" if apply else "Dry run"
    print()
    print(f"Missing candidate payload fills ({verb})")
    print(f"  candidates examined:     {result.candidates_examined}")
    print(f"  candidates to update:    {result.candidates_updated}")
    print(f"  fields to fill:          {result.fields_filled}")


def _tier_includes_caches(tier: str) -> bool:
    return tier in {"reviews-caches", "all"}


def _tier_includes_all(tier: str) -> bool:
    return tier == "all"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge already-done STV review/cache work from an older run into a newer run by ASAS-SN ID.",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-review-db", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--target-review-db", type=Path, required=True)
    parser.add_argument(
        "--tier",
        choices=["reviews", "reviews-caches", "all"],
        default="reviews-caches",
        help="Amount of reusable work to merge (default: reviews-caches)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually mutate the target run; default is dry-run")
    parser.add_argument(
        "--no-cache-rebuild",
        action="store_true",
        help="After copying external LC caches, skip cache-only summary/table/DB rebuild.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    ctx = build_merge_context(
        source_run=args.source_run,
        source_review_db=args.source_review_db,
        target_run=args.target_run,
        target_review_db=args.target_review_db,
    )
    _print_context(ctx)
    if ctx.ambiguous_overlap_asas:
        preview = ", ".join(sorted(ctx.ambiguous_overlap_asas)[:10])
        message = f"Ambiguous duplicate ASAS IDs in overlap; refusing apply. Examples: {preview}"
        if args.apply:
            parser.error(message)
        print(f"WARNING: {message}")

    if args.apply:
        _backup_if_exists(ctx.target_review_db)

    review_result = merge_reviews(ctx, apply=args.apply)
    _print_review_result(review_result, apply=args.apply)

    external_result = ExternalCacheResult()
    if _tier_includes_caches(args.tier):
        external_result = merge_external_cache(ctx, apply=args.apply)
        _print_external_result(external_result, apply=args.apply)
        if args.apply and not args.no_cache_rebuild:
            rebuild_result = rebuild_external_summaries(
                ctx,
                source_keys=external_result.copied_source_keys,
                apply=True,
            )
            _print_rebuild_result(rebuild_result)

    if _tier_includes_all(args.tier):
        native_result = copy_native_assets(ctx, apply=args.apply)
        _print_native_result(native_result, apply=args.apply)
        payload_result = merge_missing_candidate_payload_fields(ctx, apply=args.apply)
        _print_payload_result(payload_result, apply=args.apply)

    if not args.apply:
        print()
        print("Dry run only. Re-run with --apply to mutate the target run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
