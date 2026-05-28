"""Import/export a Git-trackable review bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from malca.review.taxonomy import (
    REVIEW_TAXONOMY_FIELDS,
    TAXONOMY_VERSION,
    derive_event_class,
    json_list,
    normalize_selection,
)
from malca.review.store import (
    _CANDIDATE_COLUMNS,
    _COL_NAMES,
    _COL_TYPE_MAP,
    _as_bool,
    _parse_updated_at,
    _to_float,
    _utc_now,
    db_connect,
    find_phase_plot_image,
    find_plot_image,
    get_candidate_payload,
)


SCHEMA_VERSION = 2
CANDIDATES_JSONL = "candidates.jsonl"
REVIEWS_JSONL = "reviews.jsonl"
ASSETS_MANIFEST_JSON = "assets_manifest.json"

REVIEW_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "interest_score",
    "review_pass",
    "notes",
    *REVIEW_TAXONOMY_FIELDS,
    "reviewer",
    "updated_at",
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


def _json_value(value: object) -> object:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(val) for val in value]
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_dumps(record: dict[str, object]) -> str:
    return json.dumps(_json_value(record), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")
    records: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path} line {line_no}")
        records.append(record)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    sorted_records = sorted(records, key=lambda row: str(row.get("candidate_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_json_dumps(record) + "\n" for record in sorted_records)
    path.write_text(text, encoding="utf-8")
    return len(sorted_records)


def _parse_payload_json(raw: object) -> dict[str, object]:
    if raw in (None, ""):
        return {}
    try:
        payload = json.loads(str(raw))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_column_value(column: str, value: object) -> object:
    if _is_missing(value):
        return None
    etype = _COL_TYPE_MAP.get(column)
    if etype == "bool":
        return bool(_as_bool(value))
    if etype == "float":
        return _to_float(value)
    return str(value)


def _sqlite_column_value(column: str, value: object) -> object:
    if _is_missing(value):
        return None
    etype = _COL_TYPE_MAP.get(column)
    if etype == "bool":
        return int(_as_bool(value))
    if etype == "float":
        return _to_float(value)
    return str(value)


def _candidate_records(conn: sqlite3.Connection, *, only_reviewed: bool = False) -> list[dict[str, object]]:
    candidate_cols = ["candidate_id", "source_path", *_COL_NAMES, "payload_json", "imported_at"]
    select_cols = ", ".join(f"c.{col}" for col in candidate_cols)
    query = f"SELECT {select_cols} FROM candidates c"
    if only_reviewed:
        query += " INNER JOIN reviews r ON r.candidate_id = c.candidate_id"
        query += " WHERE r.workflow_status IS NOT NULL AND r.workflow_status != 'unreviewed'"
    query += " ORDER BY c.candidate_id"
    cur = conn.execute(query)
    names = [desc[0] for desc in cur.description or []]

    records: list[dict[str, object]] = []
    first_class = set(candidate_cols) - {"payload_json"}
    for values in cur.fetchall():
        row = dict(zip(names, values))
        payload = _parse_payload_json(row.get("payload_json"))
        payload_extra = {
            str(key): value
            for key, value in payload.items()
            if str(key) not in first_class or _is_missing(row.get(str(key)))
        }

        record: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        for col in ("candidate_id", "source_path", *_COL_NAMES, "imported_at"):
            if col not in row:
                continue
            if col in _COL_TYPE_MAP:
                record[col] = _candidate_column_value(col, row.get(col))
            elif _is_missing(row.get(col)):
                record[col] = None
            else:
                record[col] = str(row.get(col))
        record["payload"] = _json_value(payload_extra)
        records.append(record)
    return records


def _review_records(conn: sqlite3.Connection, *, only_reviewed: bool = False) -> list[dict[str, object]]:
    query = f"SELECT {', '.join(REVIEW_FIELDS)} FROM reviews"
    if only_reviewed:
        query += " WHERE workflow_status IS NOT NULL AND workflow_status != 'unreviewed'"
    query += " ORDER BY candidate_id"
    cur = conn.execute(query)
    names = [desc[0] for desc in cur.description or []]
    records: list[dict[str, object]] = []
    for values in cur.fetchall():
        row = dict(zip(names, values))
        record: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        for field in REVIEW_FIELDS:
            value = row.get(field)
            if field in {"interest_score", "review_pass", "taxonomy_version"} and not _is_missing(value):
                record[field] = int(value)
            elif field in {"priority_tags_json", "evidence_flags_json", "model_tags_json"}:
                key = field.removesuffix("_json")
                try:
                    parsed = json.loads(str(value or "[]"))
                except Exception:
                    parsed = []
                record[key] = parsed if isinstance(parsed, list) else []
            elif _is_missing(value):
                record[field] = None
            else:
                record[field] = str(value)
        records.append(record)
    return records


def _conn_db_path(conn: sqlite3.Connection) -> Path | None:
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        return None
    for row in rows:
        if len(row) >= 3 and str(row[1]) == "main" and row[2]:
            return Path(str(row[2])).expanduser().resolve()
    return None


def _infer_run_root(path_text: object) -> Path | None:
    if path_text in (None, ""):
        return None
    path = Path(str(path_text)).expanduser()
    parts = path.parts
    for marker in ("results", "review", "plots", "bundle_assets"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return Path(*parts[:idx])
    if path.suffix:
        return path.parent
    return path


def _unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        if path is None:
            continue
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _asset_roots_for_candidate(
    payload: dict[str, object],
    *,
    db_path: Path | None,
    asset_roots: Sequence[Path] | None,
) -> list[Path]:
    roots: list[Path | None] = []
    roots.extend(Path(root) for root in (asset_roots or []))
    if db_path is not None:
        roots.append(db_path.parent)
        if db_path.parent.name == "review":
            roots.append(db_path.parent.parent)
    roots.append(_infer_run_root(payload.get("source_path")))
    roots.append(_infer_run_root(payload.get("lc_path")))
    roots.append(_infer_run_root(payload.get("path")))
    return _unique_paths(roots)


def _display_path(path: Path | None, roots: Sequence[Path]) -> str | None:
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path
    for root in roots:
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(resolved)


def _candidate_keys(payload: dict[str, object]) -> list[str]:
    keys: list[str] = []
    for key in ("candidate_id", "asas_sn_id"):
        value = payload.get(key)
        if value not in (None, ""):
            keys.append(str(value))
    for key in ("lc_path", "path"):
        value = payload.get(key)
        if value not in (None, ""):
            keys.append(Path(str(value)).stem)
    seen: set[str] = set()
    return [key for key in keys if key and not (key in seen or seen.add(key))]


def _expected_lightcurve_paths(keys: Sequence[str]) -> list[str]:
    suffixes = (".dat3", ".dat2", ".dat", ".csv")
    return [f"bundle_assets/lightcurves/{key}{suffix}" for key in keys for suffix in suffixes]


def _expected_raw_paths(keys: Sequence[str]) -> list[str]:
    return [f"bundle_assets/lightcurves/{key}.raw2" for key in keys]


def _expected_plot_paths(keys: Sequence[str], *, phase: bool = False) -> list[str]:
    suffixes = ("png", "jpg", "jpeg", "pdf")
    if phase:
        return [f"plots/*{key}*phase*.{suffix}" for key in keys for suffix in suffixes]
    return [f"plots/*{key}*.{suffix}" for key in keys for suffix in suffixes]


def _find_lightcurve(payload: dict[str, object], roots: Sequence[Path]) -> Path | None:
    keys = _candidate_keys(payload)
    for key in ("lc_path", "path"):
        value = payload.get(key)
        if value not in (None, ""):
            path = Path(str(value)).expanduser()
            if path.exists():
                return path
    for root in roots:
        for key in keys:
            for suffix in (".dat3", ".dat2", ".dat", ".csv"):
                path = root / "bundle_assets" / "lightcurves" / f"{key}{suffix}"
                if path.exists():
                    return path
    return None


def _find_raw_stats(payload: dict[str, object], roots: Sequence[Path], lightcurve: Path | None) -> Path | None:
    if lightcurve is not None:
        raw = lightcurve.with_suffix(".raw2")
        if raw.exists():
            return raw
    for root in roots:
        for key in _candidate_keys(payload):
            path = root / "bundle_assets" / "lightcurves" / f"{key}.raw2"
            if path.exists():
                return path
    return None


def _find_plot(payload: dict[str, object], roots: Sequence[Path], *, phase: bool = False) -> Path | None:
    finder = find_phase_plot_image if phase else find_plot_image
    for root in roots:
        plot_dir = root / "plots"
        found = finder(payload, plot_dir)
        if found is not None and found.exists():
            return found
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _asset_entry(
    *,
    kind: str,
    expected_paths: Sequence[str],
    resolved_path: Path | None,
    roots: Sequence[Path],
    hash_assets: bool,
) -> dict[str, object]:
    exists = bool(resolved_path is not None and resolved_path.exists())
    entry: dict[str, object] = {
        "kind": kind,
        "expected_paths": list(dict.fromkeys(str(path) for path in expected_paths)),
        "resolved_path": _display_path(resolved_path, roots) if exists else None,
        "exists": exists,
        "size_bytes": int(resolved_path.stat().st_size) if exists and resolved_path is not None else None,
    }
    if hash_assets:
        entry["sha256"] = _sha256(resolved_path) if exists and resolved_path is not None else None
    return entry


def _infer_source_run_id(payload: dict[str, object]) -> str | None:
    for key in ("source_run_id", "run_id"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    root = _infer_run_root(payload.get("source_path"))
    if root is not None and str(root) not in {"", "."}:
        return root.name
    return None


def build_assets_manifest(
    conn: sqlite3.Connection,
    asset_roots: Sequence[str | Path] | None = None,
    *,
    hash_assets: bool = False,
    candidate_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build a deterministic inventory of review assets."""
    scope = {str(candidate_id).strip() for candidate_id in (candidate_ids or []) if str(candidate_id).strip()}
    db_path = _conn_db_path(conn)
    explicit_roots = [Path(root).expanduser() for root in (asset_roots or [])]
    rows = conn.execute("SELECT candidate_id, source_path FROM candidates ORDER BY candidate_id").fetchall()

    assets: list[dict[str, object]] = []
    for candidate_id_raw, source_path_raw in rows:
        candidate_id = str(candidate_id_raw)
        if scope and candidate_id not in scope:
            continue
        payload = get_candidate_payload(conn, candidate_id)
        payload["candidate_id"] = candidate_id
        if source_path_raw not in (None, ""):
            payload["source_path"] = str(source_path_raw)
        roots = _asset_roots_for_candidate(payload, db_path=db_path, asset_roots=explicit_roots)
        keys = _candidate_keys(payload)

        lightcurve = _find_lightcurve(payload, roots)
        raw_stats = _find_raw_stats(payload, roots, lightcurve)
        plot = _find_plot(payload, roots, phase=False)
        phase_plot = _find_plot(payload, roots, phase=True)

        source_path = payload.get("source_path")
        if source_path in (None, ""):
            source_path = None

        entry = {
            "candidate_id": candidate_id,
            "asas_sn_id": None if payload.get("asas_sn_id") in (None, "") else str(payload.get("asas_sn_id")),
            "source_run_id": _infer_source_run_id(payload),
            "source_path": str(source_path) if source_path is not None else None,
            "assets": [
                _asset_entry(
                    kind="lightcurve",
                    expected_paths=_expected_lightcurve_paths(keys),
                    resolved_path=lightcurve,
                    roots=roots,
                    hash_assets=hash_assets,
                ),
                _asset_entry(
                    kind="raw_stats",
                    expected_paths=_expected_raw_paths(keys),
                    resolved_path=raw_stats,
                    roots=roots,
                    hash_assets=hash_assets,
                ),
                _asset_entry(
                    kind="plot",
                    expected_paths=_expected_plot_paths(keys),
                    resolved_path=plot,
                    roots=roots,
                    hash_assets=hash_assets,
                ),
                _asset_entry(
                    kind="phase_plot",
                    expected_paths=_expected_plot_paths(keys, phase=True),
                    resolved_path=phase_plot,
                    roots=roots,
                    hash_assets=hash_assets,
                ),
            ],
        }
        assets.append(entry)

    return {"schema_version": SCHEMA_VERSION, "assets": assets}


def export_review_bundle(
    db_path: str | Path,
    out_dir: str | Path = "reviews",
    *,
    hash_assets: bool = False,
    only_reviewed: bool = False,
) -> dict[str, object]:
    """Export candidates, reviews, and asset inventory to Git-friendly files."""
    db_path = Path(db_path).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    with db_connect(db_path) as conn:
        candidates = _candidate_records(conn, only_reviewed=only_reviewed)
        reviews = _review_records(conn, only_reviewed=only_reviewed)
        candidate_ids = [str(record["candidate_id"]) for record in candidates]
        manifest = build_assets_manifest(conn, hash_assets=hash_assets, candidate_ids=candidate_ids)

    candidates_path = out_dir / CANDIDATES_JSONL
    reviews_path = out_dir / REVIEWS_JSONL
    manifest_path = out_dir / ASSETS_MANIFEST_JSON
    candidate_count = _write_jsonl(candidates_path, candidates)
    review_count = _write_jsonl(reviews_path, reviews)
    manifest_path.write_text(
        json.dumps(_json_value(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "candidates_path": str(candidates_path),
        "reviews_path": str(reviews_path),
        "assets_manifest_path": str(manifest_path),
        "candidates_exported": candidate_count,
        "reviews_exported": review_count,
        "assets_candidates": len(manifest["assets"]),
    }


def auto_export_review_bundle(
    db_path: str | Path,
    out_dir: str | Path = "reviews",
    *,
    hash_assets: bool = False,
    logger: Callable[[str], object] | None = None,
) -> dict[str, object]:
    """Best-effort review Git-bundle export for DB-mutating workflows."""

    def emit(message: str) -> None:
        if logger is None:
            print(message)
            return
        try:
            logger(message)
        except Exception:
            print(message)

    try:
        result = export_review_bundle(
            db_path,
            out_dir,
            hash_assets=hash_assets,
            only_reviewed=False,
        )
    except Exception as exc:
        error_result = {
            "ok": False,
            "db_path": str(Path(db_path).expanduser()),
            "out_dir": str(Path(out_dir).expanduser()),
            "error": str(exc),
        }
        emit(f"Warning: review Git bundle export failed: {exc}")
        return error_result

    result = {**result, "ok": True}
    emit(
        "Exported review Git bundle to "
        f"{result['out_dir']} "
        f"({result['candidates_exported']} candidates, "
        f"{result['reviews_exported']} reviews, "
        f"{result['assets_candidates']} asset entries)"
    )
    return result


def _coerce_review_int(value: object, *, default: int | None = None) -> int | None:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_sql_rows(records: Sequence[dict[str, object]]) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for record in records:
        candidate_id = str(record.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("Candidate record missing candidate_id")

        payload = record.get("payload")
        payload_dict = dict(payload) if isinstance(payload, dict) else {}
        first_class_payload = {
            key: value
            for key, value in record.items()
            if key not in {"schema_version", "payload", "payload_json"}
            and not _is_missing(value)
        }
        payload_json = {**payload_dict, **first_class_payload}

        source_path = record.get("source_path")
        imported_at = record.get("imported_at") or _utc_now()
        values: list[object] = [
            candidate_id,
            None if _is_missing(source_path) else str(source_path),
        ]
        for col, _dtype, _etype in _CANDIDATE_COLUMNS:
            values.append(_sqlite_column_value(col, record.get(col)))
        values.append(_json_dumps(payload_json))
        values.append(str(imported_at))
        rows.append(tuple(values))
    return rows


def _upsert_candidate_records(conn: sqlite3.Connection, records: Sequence[dict[str, object]]) -> dict[str, int]:
    if not records:
        return {"candidates_written": 0, "candidates_inserted": 0}

    rows = _candidate_sql_rows(records)
    all_col_names = ["candidate_id", "source_path", *_COL_NAMES, "payload_json", "imported_at"]
    placeholders = ", ".join(["?"] * len(all_col_names))
    update_cols = [col for col in all_col_names if col != "candidate_id"]
    conflict_set = ", ".join(f"{col}=excluded.{col}" for col in update_cols)
    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO candidates ({', '.join(all_col_names)})
        VALUES ({placeholders})
        ON CONFLICT(candidate_id) DO UPDATE SET {conflict_set}
        """,
        rows,
    )
    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return {"candidates_written": len(rows), "candidates_inserted": int(after - before)}


def _import_review_records(
    conn: sqlite3.Connection,
    records: Sequence[dict[str, object]],
    *,
    merge_newer: bool = True,
) -> dict[str, int]:
    existing_candidates = {
        str(row[0])
        for row in conn.execute("SELECT candidate_id FROM candidates").fetchall()
    }
    existing_reviews = {
        str(row[0]): row[1]
        for row in conn.execute("SELECT candidate_id, updated_at FROM reviews").fetchall()
    }

    inserted = 0
    updated = 0
    skipped = 0
    for record in records:
        candidate_id = str(record.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("Review record missing candidate_id")
        if candidate_id not in existing_candidates:
            raise ValueError(f"Review references candidate not present in candidates.jsonl: {candidate_id}")

        updated_at = str(record.get("updated_at") or _utc_now())
        target_updated = existing_reviews.get(candidate_id)
        if merge_newer and target_updated is not None:
            if _parse_updated_at(updated_at) <= _parse_updated_at(target_updated):
                skipped += 1
                continue

        interest_score = _coerce_review_int(record.get("interest_score"))
        if interest_score is not None:
            interest_score = max(1, min(4, interest_score))
        review_pass = _coerce_review_int(record.get("review_pass"), default=1) or 1
        review_pass = max(1, review_pass)
        notes = "" if record.get("notes") is None else str(record.get("notes"))
        selection = normalize_selection(
            {
                "morphology_primary": record.get("morphology_primary"),
                "morphology_secondary": record.get("morphology_secondary"),
                "morphology_polarity": record.get("morphology_polarity"),
                "morphology_recurrence": record.get("morphology_recurrence"),
                "baseline_behavior": record.get("baseline_behavior"),
                "physical_primary": record.get("physical_primary"),
                "physical_secondary": record.get("physical_secondary"),
                "classification_confidence": record.get("classification_confidence"),
                "priority_tags": record.get("priority_tags") or record.get("priority_tags_json"),
                "evidence_flags": record.get("evidence_flags") or record.get("evidence_flags_json"),
                "model_tags": record.get("model_tags") or record.get("model_tags_json"),
                "disposition": record.get("disposition"),
                "duplicate_of": record.get("duplicate_of"),
                "known_object_id": record.get("known_object_id"),
                "known_object_source": record.get("known_object_source"),
            }
        )
        workflow_status = str(record.get("workflow_status") or "unreviewed")
        status = workflow_status
        event_class = derive_event_class(selection)
        reviewer = "" if record.get("reviewer") is None else str(record.get("reviewer"))
        taxonomy_values = {
            "workflow_status": workflow_status,
            "disposition": selection.get("disposition"),
            "morphology_primary": selection.get("morphology_primary"),
            "morphology_secondary": selection.get("morphology_secondary"),
            "morphology_polarity": selection.get("morphology_polarity"),
            "morphology_recurrence": selection.get("morphology_recurrence"),
            "baseline_behavior": selection.get("baseline_behavior"),
            "physical_primary": selection.get("physical_primary"),
            "physical_secondary": selection.get("physical_secondary"),
            "classification_confidence": selection.get("classification_confidence"),
            "priority_tags_json": json_list(selection.get("priority_tags")),
            "evidence_flags_json": json_list(selection.get("evidence_flags")),
            "model_tags_json": json_list(selection.get("model_tags")),
            "duplicate_of": selection.get("duplicate_of"),
            "known_object_id": selection.get("known_object_id"),
            "known_object_source": selection.get("known_object_source"),
            "taxonomy_version": _coerce_review_int(record.get("taxonomy_version"), default=TAXONOMY_VERSION) or TAXONOMY_VERSION,
            "legacy_review_json": "{}" if record.get("legacy_review_json") is None else str(record.get("legacy_review_json")),
        }
        taxonomy_cols = list(REVIEW_TAXONOMY_FIELDS)
        insert_cols = [
            "candidate_id",
            "interest_score",
            "event_class",
            "review_pass",
            "notes",
            "status",
            "reviewer",
            *taxonomy_cols,
            "updated_at",
        ]
        placeholders = ", ".join(["?"] * len(insert_cols))
        update_cols = [col for col in insert_cols if col != "candidate_id"]
        conflict_set = ",\n                ".join(f"{col}=excluded.{col}" for col in update_cols)

        conn.execute(
            f"""
            INSERT INTO reviews ({', '.join(insert_cols)})
            VALUES ({placeholders})
            ON CONFLICT(candidate_id) DO UPDATE SET
                {conflict_set}
            """,
            (
                candidate_id,
                interest_score,
                event_class,
                review_pass,
                notes,
                status,
                reviewer,
                *(taxonomy_values[col] for col in taxonomy_cols),
                updated_at,
            ),
        )
        if target_updated is None:
            inserted += 1
        else:
            updated += 1
        existing_reviews[candidate_id] = updated_at

    return {"reviews_inserted": inserted, "reviews_updated": updated, "reviews_skipped": skipped}


def _remove_db_files(db_path: Path) -> None:
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if path.exists():
            path.unlink()


def _validate_assets_manifest(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Required asset manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON asset manifest: {path}") from exc
    assets = manifest.get("assets") if isinstance(manifest, dict) else None
    if not isinstance(assets, list):
        raise ValueError(f"Asset manifest must contain an 'assets' list: {path}")

    candidate_count = 0
    missing_assets = 0
    for candidate in assets:
        if not isinstance(candidate, dict):
            continue
        candidate_count += 1
        for asset in candidate.get("assets", []):
            if isinstance(asset, dict) and not bool(asset.get("exists")):
                missing_assets += 1
    return {"asset_candidates": candidate_count, "manifest_missing_assets": missing_assets}


def import_review_bundle(
    in_dir: str | Path = "reviews",
    *,
    db_path: str | Path,
    replace: bool = False,
) -> dict[str, object]:
    """Import a Git-tracked review bundle into a SQLite review DB."""
    in_dir = Path(in_dir).expanduser()
    db_path = Path(db_path).expanduser()
    candidates_path = in_dir / CANDIDATES_JSONL
    reviews_path = in_dir / REVIEWS_JSONL
    manifest_path = in_dir / ASSETS_MANIFEST_JSON

    candidate_records = _read_jsonl(candidates_path)
    review_records = _read_jsonl(reviews_path)
    manifest_stats = _validate_assets_manifest(manifest_path)

    if replace:
        _remove_db_files(db_path)

    with db_connect(db_path) as conn:
        candidate_stats = _upsert_candidate_records(conn, candidate_records)
        review_stats = _import_review_records(conn, review_records, merge_newer=not replace)
        conn.commit()

    return {
        "db_path": str(db_path),
        "in_dir": str(in_dir),
        "replace": bool(replace),
        **candidate_stats,
        **review_stats,
        **manifest_stats,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="malca review-sync",
        description="Import/export a Git-trackable review bundle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export review DB to reviews/*.jsonl and manifest")
    export_parser.add_argument("--review-db", type=Path, default=Path("output/review/review.db"), help="Review SQLite DB path")
    export_parser.add_argument("--output-dir", type=Path, default=Path("reviews"), help="Output review bundle directory")
    export_parser.add_argument("--hash-assets", action="store_true", help="Include SHA-256 hashes for resolved assets")
    export_parser.add_argument("--only-reviewed", action="store_true", help="Export only reviewed/non-unreviewed rows")

    import_parser = subparsers.add_parser("import", help="Import reviews/*.jsonl into a review DB")
    import_parser.add_argument("--review-db", type=Path, default=Path("output/review/review.db"), help="Review SQLite DB path")
    import_parser.add_argument("--input-dir", type=Path, default=Path("reviews"), help="Input review bundle directory")
    import_parser.add_argument("--replace", action="store_true", help="Replace the target DB before import")

    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_review_bundle(
            args.review_db,
            args.output_dir,
            hash_assets=bool(args.hash_assets),
            only_reviewed=bool(args.only_reviewed),
        )
    else:
        result = import_review_bundle(args.input_dir, db_path=args.review_db, replace=bool(args.replace))
    print(json.dumps(_json_value(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
