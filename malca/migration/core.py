from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable, Literal

import pandas as pd

from malca.products.feature_layers import (
    ALL_FEATURE_LAYER_COLUMNS,
    DERIVED_STATS_LAYER,
    EXTERNAL_STATS_LAYER,
    FEATURE_LAYER_COLUMNS,
    FEATURE_LAYER_VERSION,
    FEATURE_LAYER_VERSION_COLUMN,
    LC_STATS_LAYER,
    feature_columns_by_layer,
    non_layer_feature_columns,
    to_layer_first_frame,
    to_layer_first_mapping,
    unclassified_non_feature_columns,
)
from malca.migration.schema_aliases import (
    CAMERA_FIELD_ALIAS_MAP,
    PreferPolicy,
    convert_product_frame,
    detect_product_timescale,
    migrate_camera_field_frame,
    migrate_camera_field_mapping,
)
from malca.config import DEFAULT_OUTPUT_DIR
from malca.io.table_io import read_parquet_table, write_parquet_table
from malca.review.store import (
    _COL_NAMES as REVIEW_CANDIDATE_COLUMNS,
    _flatten_review_payload,
    _review_payload_extra,
    _sql_value_for_column,
    init_db as init_review_db,
)


ArtifactAction = Literal["migrate", "copy"]

PRODUCT_PARQUET = "product_parquet"
PRODUCT_CSV = "product_csv"
REVIEW_DB = "review_db"
CANDIDATES_JSONL = "candidates_jsonl"
REVIEWS_JSONL = "reviews_jsonl"
ASSETS_MANIFEST = "assets_manifest"
COPIED_ASSET = "copied_asset"
COPIED_TABLE = "copied_table"
COPIED_DB = "copied_db"

_PRODUCT_NAME_HINTS = (
    "candidate",
    "candidates",
    "lc_events",
    "ltvar",
    "pipeline",
    "ranked",
    "review",
    "stochastic_features",
    "lightcurve_quality",
    "outlier",
)
_PRODUCT_CONTEXT_HINTS = {"results", "review", "runs", "transfer"}
_CACHE_OR_ASSET_PARTS = {
    ".dash_cache",
    "cache",
    "catalogs",
    "bundle_assets",
    "plots",
    "assets",
    "lightcurves",
}
_SIDE_TABLE_NAME_HINTS = (
    "sed_photometry",
    "sed_model_fits",
    "sed_model_curves",
    "dustycult_fits",
    "dustycult_predictive_curves",
    "phoebe_fits",
)
_INDEX_TABLE_NAME_HINTS = (
    "index",
    "index_chunks",
)
_CANDIDATE_IDENTITY_COLUMNS = {
    "candidate_id",
    "timescale",
    "lc_path",
    "path",
    "dat_path",
    "asas_sn_id",
}
_REQUIRED_LAYER_COUNTS = {
    LC_STATS_LAYER: 0,
    EXTERNAL_STATS_LAYER: 0,
    DERIVED_STATS_LAYER: 0,
}
_CAMERA_NAME_SQL_COLUMN_TYPES = {
    "camera_name_key": "TEXT",
    "camera_names": "TEXT",
    "camera_name_count": "REAL",
    "camera_name_key_fraction": "REAL",
}
_PATH_LIKE_KEYS = {
    "path",
    "paths",
    "lc_path",
    "dat_path",
    "source_path",
    "last_input_file",
}


@dataclass(frozen=True)
class Artifact:
    input_path: str
    relative_path: str
    artifact_type: str
    action: ArtifactAction
    reason: str = ""
    rows: int | None = None
    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactReport:
    input_path: str
    output_path: str | None
    relative_path: str
    artifact_type: str
    action: str
    reason: str = ""
    rows: int | None = None
    columns: list[str] = field(default_factory=list)
    layer_counts: dict[str, int] = field(default_factory=dict)
    migrated_columns: list[str] = field(default_factory=list)
    unclassified_columns: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MigrationSummary:
    input_root: str
    output_root: str | None
    report_path: str | None
    unclassified_columns_path: str | None
    artifacts: list[ArtifactReport]

    @property
    def errors(self) -> list[ArtifactReport]:
        return [item for item in self.artifacts if item.error]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_root": self.input_root,
            "output_root": self.output_root,
            "report_path": self.report_path,
            "unclassified_columns_path": self.unclassified_columns_path,
            "ok": self.ok,
            "errors": [item.to_dict() for item in self.errors],
            "artifact_count": len(self.artifacts),
        }


def default_output_root(now: datetime | None = None) -> Path:
    return DEFAULT_OUTPUT_DIR


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _json_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return not pd.notna(value)
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _is_parquet_dataset_dir(path: Path) -> bool:
    return path.is_dir() and any(child.is_file() and child.suffix.lower() == ".parquet" for child in path.glob("chunk_*.parquet"))


def _is_parquet_artifact(path: Path) -> bool:
    return (path.is_file() and path.suffix.lower() == ".parquet") or _is_parquet_dataset_dir(path)


def _path_parts_lower(path: Path) -> set[str]:
    return {part.lower() for part in path.parts}


def _is_cache_or_asset_path(path: Path) -> bool:
    return bool(_path_parts_lower(path) & _CACHE_OR_ASSET_PARTS)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _mirror_path(path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root
    return output_root / path.relative_to(input_root)


def _read_parquet_artifact(path: Path) -> pd.DataFrame:
    if path.is_file():
        return read_parquet_table(path)
    table = pd.read_parquet(path)
    return table.to_frame() if isinstance(table, pd.Series) else table


def _write_parquet_artifact(df: pd.DataFrame, path: Path, *, is_dataset: bool) -> None:
    if not is_dataset:
        write_parquet_table(df, path)
        return
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    write_parquet_table(df, path / "chunk_000000.parquet")


def _parquet_metadata(path: Path) -> tuple[list[str], int | None]:
    try:
        import pyarrow.parquet as pq

        if path.is_file():
            metadata = pq.read_metadata(path)
            return list(metadata.schema.names), int(metadata.num_rows)
        columns: list[str] = []
        rows = 0
        for chunk in sorted(path.glob("chunk_*.parquet")):
            metadata = pq.read_metadata(chunk)
            rows += int(metadata.num_rows)
            for name in metadata.schema.names:
                if name not in columns:
                    columns.append(name)
        return columns, rows
    except Exception:
        try:
            df = _read_parquet_artifact(path)
            return list(df.columns), int(len(df))
        except Exception:
            return [], None


def _csv_metadata(path: Path) -> tuple[list[str], int | None]:
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return [], None
    return columns, None


def _looks_like_product_table(path: Path, columns: Iterable[str]) -> tuple[bool, str]:
    if _is_cache_or_asset_path(path):
        return False, "cache_or_asset_table"

    cols = set(str(col) for col in columns)
    if not cols:
        return False, "empty_or_unreadable_schema"

    layer_cols = feature_columns_by_layer(cols)
    layer_count = sum(len(values) for values in layer_cols.values())
    has_identity = bool(cols & _CANDIDATE_IDENTITY_COLUMNS)
    has_layer_columns = set(FEATURE_LAYER_COLUMNS).issubset(cols)
    name = path.name.lower()
    path_text = path.as_posix().lower()
    if any(hint in name for hint in _SIDE_TABLE_NAME_HINTS):
        return False, "review_side_table"
    if any(hint in path_text for hint in _INDEX_TABLE_NAME_HINTS):
        return False, "index_table"
    name_hint = any(hint in name for hint in _PRODUCT_NAME_HINTS)
    context_hint = bool(_path_parts_lower(path) & _PRODUCT_CONTEXT_HINTS)

    if has_identity and (layer_count > 0 or has_layer_columns):
        return True, "candidate_table_with_features"
    if has_identity and name_hint and context_hint:
        return True, "candidate_table_path_hint"
    return False, "not_candidate_product"


def _sqlite_has_table(path: Path, table_name: str) -> bool:
    if _is_cache_or_asset_path(path):
        return False
    try:
        uri = f"file:{path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _classify_path(path: Path, root: Path) -> Artifact:
    relative = _relative_path(path, root)
    suffix = path.suffix.lower()

    if _is_parquet_artifact(path):
        columns, rows = _parquet_metadata(path)
        is_product, reason = _looks_like_product_table(path, columns)
        return Artifact(
            input_path=str(path),
            relative_path=relative,
            artifact_type=PRODUCT_PARQUET if is_product else COPIED_TABLE,
            action="migrate" if is_product else "copy",
            reason=reason,
            rows=rows,
            columns=columns,
        )

    if path.is_file() and suffix == ".csv":
        columns, rows = _csv_metadata(path)
        is_product, reason = _looks_like_product_table(path, columns)
        return Artifact(
            input_path=str(path),
            relative_path=relative,
            artifact_type=PRODUCT_CSV if is_product else COPIED_TABLE,
            action="migrate" if is_product else "copy",
            reason=reason,
            rows=rows,
            columns=columns,
        )

    if path.is_file() and suffix == ".db":
        is_review = _sqlite_has_table(path, "candidates")
        return Artifact(
            input_path=str(path),
            relative_path=relative,
            artifact_type=REVIEW_DB if is_review else COPIED_DB,
            action="migrate" if is_review else "copy",
            reason="sqlite_candidates_table" if is_review else "non_review_sqlite",
        )

    if path.is_file() and suffix == ".jsonl":
        if path.name == "candidates.jsonl":
            return Artifact(str(path), relative, CANDIDATES_JSONL, "migrate", "review_transfer_candidates")
        if path.name == "reviews.jsonl":
            return Artifact(str(path), relative, REVIEWS_JSONL, "copy", "review_transfer_reviews")

    if path.is_file() and path.name == "assets_manifest.json":
        return Artifact(str(path), relative, ASSETS_MANIFEST, "copy", "review_transfer_assets_manifest")

    return Artifact(str(path), relative, COPIED_ASSET, "copy", "copied_unchanged")


def discover_artifacts(input_root: str | Path = "output") -> list[Artifact]:
    root = Path(input_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Input not found: {root}")
    if root.is_file():
        return [_classify_path(root, root.parent)]

    artifacts: list[Artifact] = []
    dataset_dirs: set[Path] = set()
    for path in sorted(root.rglob("*")):
        if any(parent in dataset_dirs for parent in path.parents):
            continue
        if _is_parquet_dataset_dir(path):
            artifacts.append(_classify_path(path, root))
            dataset_dirs.add(path)
            continue
        if path.is_file():
            artifacts.append(_classify_path(path, root))
    return artifacts


def _report_from_artifact(
    artifact: Artifact,
    *,
    output_path: Path | None,
    action: str | None = None,
    error: str | None = None,
    layer_counts: dict[str, int] | None = None,
    migrated_columns: list[str] | None = None,
    unclassified_columns: list[str] | None = None,
    rows: int | None = None,
    columns: list[str] | None = None,
) -> ArtifactReport:
    return ArtifactReport(
        input_path=artifact.input_path,
        output_path=str(output_path) if output_path is not None else None,
        relative_path=artifact.relative_path,
        artifact_type=artifact.artifact_type,
        action=action or artifact.action,
        reason=artifact.reason,
        rows=artifact.rows if rows is None else rows,
        columns=artifact.columns if columns is None else columns,
        layer_counts={**_REQUIRED_LAYER_COUNTS, **(layer_counts or {})},
        migrated_columns=migrated_columns or [],
        unclassified_columns=unclassified_columns or [],
        error=error,
    )


def _layer_counts_from_frame(df: pd.DataFrame) -> dict[str, int]:
    counts = {layer: 0 for layer in FEATURE_LAYER_COLUMNS}
    if df.empty:
        return counts
    if any(layer not in df.columns for layer in FEATURE_LAYER_COLUMNS):
        return counts
    first = df.iloc[0]
    for layer in FEATURE_LAYER_COLUMNS:
        value = first.get(layer)
        if not isinstance(value, str):
            counts[layer] = 0
            continue
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = {}
        counts[layer] = len(parsed) if isinstance(parsed, dict) else 0
    return counts


def _convert_product_for_layers(
    df: pd.DataFrame,
    path: Path,
    *,
    prefer: PreferPolicy,
) -> pd.DataFrame:
    try:
        timescale = detect_product_timescale(path, df)
    except Exception:
        return migrate_camera_field_frame(df, prefer=prefer)
    return convert_product_frame(df, timescale, prefer=prefer)


def _layer_first_product_frame(
    original: pd.DataFrame,
    input_path: Path,
    *,
    prefer: PreferPolicy,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    canonical = _convert_product_for_layers(original, input_path, prefer=prefer)
    migrated_cols = non_layer_feature_columns(canonical.columns)
    unclassified = unclassified_non_feature_columns(canonical.columns)
    layered = to_layer_first_frame(canonical)
    return layered, migrated_cols, unclassified


def migrate_parquet_product(
    artifact: Artifact,
    output_path: Path,
    *,
    prefer: PreferPolicy = "fail",
) -> ArtifactReport:
    input_path = Path(artifact.input_path)
    try:
        original = _read_parquet_artifact(input_path)
        layered, migrated_cols, unclassified = _layer_first_product_frame(
            original,
            input_path,
            prefer=prefer,
        )
        _write_parquet_artifact(layered, output_path, is_dataset=input_path.is_dir())
        return _report_from_artifact(
            artifact,
            output_path=output_path,
            action="migrated",
            rows=len(layered),
            columns=list(layered.columns),
            layer_counts=_layer_counts_from_frame(layered),
            migrated_columns=migrated_cols,
            unclassified_columns=unclassified,
        )
    except Exception as exc:
        return _report_from_artifact(artifact, output_path=output_path, action="failed", error=str(exc))


def migrate_csv_product(
    artifact: Artifact,
    output_path: Path,
    *,
    prefer: PreferPolicy = "fail",
) -> ArtifactReport:
    input_path = Path(artifact.input_path)
    try:
        original = pd.read_csv(input_path)
        layered, migrated_cols, unclassified = _layer_first_product_frame(
            original,
            input_path,
            prefer=prefer,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        layered.to_csv(output_path, index=False)
        return _report_from_artifact(
            artifact,
            output_path=output_path,
            action="migrated",
            rows=len(layered),
            columns=list(layered.columns),
            layer_counts=_layer_counts_from_frame(layered),
            migrated_columns=migrated_cols,
            unclassified_columns=unclassified,
        )
    except Exception as exc:
        return _report_from_artifact(artifact, output_path=output_path, action="failed", error=str(exc))


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _artifact_roots(artifact: Artifact, output_path: Path) -> tuple[Path, Path]:
    """Return the source and migrated roots for an artifact mirror path."""
    rel_parts = Path(artifact.relative_path).parts
    input_root = Path(artifact.input_path).expanduser()
    output_root = output_path.expanduser()
    for _part in rel_parts:
        input_root = input_root.parent
        output_root = output_root.parent
    try:
        output_root = output_root.resolve()
    except Exception:
        pass
    return input_root, output_root


def _is_path_like_key(key: object) -> bool:
    text = str(key or "").strip().lower()
    return (
        text in _PATH_LIKE_KEYS
        or text.endswith("_path")
        or text.endswith("_paths")
        or text.endswith("_file")
    )


def _rewrite_output_path_line(text: str, output_root: Path) -> str:
    stripped = text.strip()
    if not stripped or "://" in stripped:
        return text

    suffix: str | None = None
    marker = "/output/"
    if stripped == "output":
        suffix = ""
    elif stripped.startswith("output/"):
        suffix = stripped[len("output/"):]
    elif marker in stripped:
        suffix = stripped.split(marker, 1)[1]

    if suffix is None:
        return text

    rewritten = str(output_root / suffix)
    return text.replace(stripped, rewritten, 1)


def _rewrite_output_path_text(value: str, output_root: Path) -> str:
    lines = value.splitlines()
    if len(lines) <= 1:
        return _rewrite_output_path_line(value, output_root)
    return "\n".join(_rewrite_output_path_line(line, output_root) for line in lines)


def _rewrite_path_values(value: Any, output_root: Path, *, key: object = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _rewrite_path_values(
                item_value,
                output_root,
                key=item_key,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_path_values(item, output_root, key=key) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_path_values(item, output_root, key=key) for item in value)
    if isinstance(value, str) and _is_path_like_key(key):
        return _rewrite_output_path_text(value, output_root)
    return value


def _zeroish(value: Any) -> bool:
    if _is_missing(value):
        return False
    try:
        return float(value) == 0.0
    except Exception:
        return False


def _prune_empty_camera_group(mapping: dict[str, Any], keys: tuple[str, str, str, str]) -> None:
    key_col, names_col, count_col, fraction_col = keys
    key_value = mapping.get(key_col)
    names_value = mapping.get(names_col)
    if (
        (key_value in (None, ""))
        and (names_value in (None, ""))
        and _zeroish(mapping.get(count_col))
    ):
        for col in keys:
            mapping.pop(col, None)


def _prune_empty_camera_name_metadata(layered: dict[str, Any]) -> dict[str, Any]:
    out = dict(layered)
    groups = (
        ("camera_name_key", "camera_names", "camera_name_count", "camera_name_key_fraction"),
        ("stats_camera_name_key", "stats_camera_names", "stats_camera_name_count", "stats_camera_name_key_fraction"),
    )
    for group in groups:
        _prune_empty_camera_group(out, group)
    for layer in FEATURE_LAYER_COLUMNS:
        layer_payload = out.get(layer)
        if not isinstance(layer_payload, dict):
            continue
        layer_payload = dict(layer_payload)
        for group in groups:
            _prune_empty_camera_group(layer_payload, group)
        out[layer] = layer_payload
    return out


def _candidate_row_layer_payload(
    row: sqlite3.Row | dict[str, Any],
    *,
    output_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    raw = dict(row)
    payload = _parse_json_object(raw.get("payload_json"))
    merged: dict[str, Any] = dict(payload)
    for key, value in raw.items():
        if key in {"rowid", "payload_json"}:
            continue
        if _is_missing(value):
            continue
        merged[str(key)] = value
    if output_root is not None:
        merged = _rewrite_path_values(merged, output_root)
    merged = migrate_camera_field_mapping(merged)
    layered = to_layer_first_mapping(merged, layer_values_as_json=False)
    layered = _prune_empty_camera_name_metadata(layered)
    layer_sql = {
        layer: _json_dumps(layered.get(layer) or {})
        for layer in FEATURE_LAYER_COLUMNS
    }
    return layered, layer_sql


def _layered_value(layered: dict[str, Any], key: str) -> Any:
    if key in layered:
        return layered.get(key)
    for layer in FEATURE_LAYER_COLUMNS:
        value = layered.get(layer)
        layer_payload = value if isinstance(value, dict) else _parse_json_object(value)
        if key in layer_payload:
            return layer_payload.get(key)
    return None


def _camera_sql_value(layered: dict[str, Any], key: str) -> Any:
    value = _layered_value(layered, key)
    if _is_missing(value):
        return None
    if key in {"camera_name_count", "camera_name_key_fraction"}:
        try:
            return float(value)
        except Exception:
            return None
    return str(value)


def _drop_legacy_camera_field_columns(conn: sqlite3.Connection, existing: set[str]) -> None:
    for old_col in CAMERA_FIELD_ALIAS_MAP:
        if old_col not in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE candidates DROP COLUMN {old_col}")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Could not drop legacy candidates.{old_col}: {exc}") from exc


def migrate_review_db(artifact: Artifact, output_path: Path) -> ArtifactReport:
    try:
        _input_root, output_root = _artifact_roots(artifact, output_path)
        with sqlite3.connect(str(output_path)) as conn:
            conn.row_factory = sqlite3.Row
            init_review_db(conn)
            existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}
            for col, dtype in _CAMERA_NAME_SQL_COLUMN_TYPES.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE candidates ADD COLUMN {col} {dtype}")
                    existing.add(col)

            rows = conn.execute("SELECT rowid, * FROM candidates ORDER BY rowid").fetchall()
            layer_counts = {layer: 0 for layer in FEATURE_LAYER_COLUMNS}
            for row in rows:
                raw = dict(row)
                merged = _flatten_review_payload(_parse_json_object(raw.get("payload_json")))
                for key, value in raw.items():
                    if key in {"rowid", "payload_json"} or _is_missing(value):
                        continue
                    merged[str(key)] = value
                merged = _rewrite_path_values(merged, output_root)
                merged = migrate_camera_field_mapping(merged)

                sql_cols = [col for col in REVIEW_CANDIDATE_COLUMNS if col in existing]
                assignments = ["source_path = ?", "payload_json = ?"]
                params: list[Any] = [
                    merged.get("source_path"),
                    _json_dumps(_review_payload_extra(merged, set(REVIEW_CANDIDATE_COLUMNS))),
                ]
                for col in sql_cols:
                    assignments.append(f"{col} = ?")
                    params.append(_sql_value_for_column(col, merged.get(col)))
                params.append(row["rowid"])
                conn.execute(
                    f"UPDATE candidates SET {', '.join(assignments)} WHERE rowid = ?",
                    params,
                )
            for key, value in conn.execute("SELECT key, value FROM app_state").fetchall():
                if _is_path_like_key(key):
                    rewritten = _rewrite_path_values(value, output_root, key=key)
                    if rewritten != value:
                        conn.execute("UPDATE app_state SET value = ? WHERE key = ?", (rewritten, key))
            _drop_legacy_camera_field_columns(conn, existing)
            conn.commit()
        return _report_from_artifact(
            artifact,
            output_path=output_path,
            action="migrated",
            rows=len(rows),
            layer_counts=layer_counts,
        )
    except Exception as exc:
        return _report_from_artifact(artifact, output_path=output_path, action="failed", error=str(exc))


def _migrate_candidate_json_record(
    record: dict[str, Any],
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    payload = record.get("payload")
    payload_dict = dict(payload) if isinstance(payload, dict) else _parse_json_object(record.get("payload_json"))
    first_class_keys = {
        str(key)
        for key in record
        if key not in {"schema_version", "payload", "payload_json"}
    }
    merged = dict(payload_dict)
    for key in first_class_keys:
        value = record.get(key)
        if not _is_missing(value):
            merged[key] = value
    if output_root is not None:
        merged = _rewrite_path_values(merged, output_root)
    merged = migrate_camera_field_mapping(merged)

    layered = to_layer_first_mapping(merged, layer_values_as_json=False)
    layered = _prune_empty_camera_name_metadata(layered)
    migrated: dict[str, Any] = {"schema_version": record.get("schema_version", 2)}
    for key in first_class_keys:
        if key in ALL_FEATURE_LAYER_COLUMNS or key in FEATURE_LAYER_COLUMNS:
            continue
        if key in layered:
            migrated[key] = layered.pop(key)
    migrated[FEATURE_LAYER_VERSION_COLUMN] = FEATURE_LAYER_VERSION
    migrated["payload"] = layered
    return migrated


def migrate_candidates_jsonl(artifact: Artifact, output_path: Path) -> ArtifactReport:
    try:
        _input_root, output_root = _artifact_roots(artifact, output_path)
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object on line {line_no}")
            records.append(_migrate_candidate_json_record(parsed, output_root=output_root))
        output_path.write_text("".join(_json_dumps(record) + "\n" for record in records), encoding="utf-8")
        layer_counts = {layer: 0 for layer in FEATURE_LAYER_COLUMNS}
        if records:
            payload = records[0].get("payload") if isinstance(records[0].get("payload"), dict) else {}
            for layer in FEATURE_LAYER_COLUMNS:
                value = payload.get(layer) if isinstance(payload, dict) else None
                layer_counts[layer] = len(value) if isinstance(value, dict) else 0
        return _report_from_artifact(
            artifact,
            output_path=output_path,
            action="migrated",
            rows=len(records),
            layer_counts=layer_counts,
        )
    except Exception as exc:
        return _report_from_artifact(artifact, output_path=output_path, action="failed", error=str(exc))


def _copy_input_tree(input_root: Path, output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()
    if input_root.is_file():
        output_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_root, output_root)
    else:
        shutil.copytree(input_root, output_root, symlinks=True)


def _write_reports(
    reports: list[ArtifactReport],
    *,
    report_path: Path,
    unclassified_columns_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_json_pretty([item.to_dict() for item in reports]) + "\n", encoding="ascii")

    unclassified: dict[str, list[str]] = {}
    for item in reports:
        if item.unclassified_columns:
            unclassified[item.relative_path] = item.unclassified_columns
    unclassified_columns_path.write_text(_json_pretty(unclassified) + "\n", encoding="ascii")


def _normal_report_paths(
    *,
    output_root: Path | None,
    report_path: str | Path | None,
    unclassified_columns_path: str | Path | None,
    scan_only: bool,
) -> tuple[Path | None, Path | None]:
    if report_path is not None:
        report = Path(report_path).expanduser()
    elif output_root is not None:
        report = output_root / "migration_report.json"
    elif scan_only:
        report = Path(f"migration_scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        report = None

    if unclassified_columns_path is not None:
        unclassified = Path(unclassified_columns_path).expanduser()
    elif report is not None:
        unclassified = report.parent / "unclassified_columns.json"
    else:
        unclassified = None
    return report, unclassified


def _is_output_inside_input(input_root: Path, output_root: Path) -> bool:
    try:
        output_root.resolve().relative_to(input_root.resolve())
        return True
    except ValueError:
        return False


def migrate_tree(
    input_root: str | Path = "output",
    output_root: str | Path | None = None,
    *,
    scan_only: bool = False,
    overwrite: bool = False,
    prefer: PreferPolicy = "fail",
    report_path: str | Path | None = None,
    unclassified_columns_path: str | Path | None = None,
) -> MigrationSummary:
    source = Path(input_root).expanduser()
    artifacts = discover_artifacts(source)
    target = None if scan_only else Path(output_root).expanduser() if output_root is not None else default_output_root()
    if target is not None and source.is_dir() and _is_output_inside_input(source, target):
        raise ValueError(f"Output root must not be inside input root: {target}")

    reports: list[ArtifactReport] = []
    if scan_only:
        for artifact in artifacts:
            migrated_columns = []
            unclassified_columns = []
            if artifact.action == "migrate" and artifact.artifact_type in {PRODUCT_PARQUET, PRODUCT_CSV}:
                migrated_columns = non_layer_feature_columns(artifact.columns)
                unclassified_columns = unclassified_non_feature_columns(artifact.columns)
            reports.append(
                _report_from_artifact(
                    artifact,
                    output_path=None,
                    action=f"would_{artifact.action}",
                    migrated_columns=migrated_columns,
                    unclassified_columns=unclassified_columns,
                )
            )
    else:
        assert target is not None
        _copy_input_tree(source, target, overwrite=overwrite)
        for artifact in artifacts:
            input_path = Path(artifact.input_path)
            output_path = _mirror_path(input_path, source, target)
            if artifact.action != "migrate":
                reports.append(_report_from_artifact(artifact, output_path=output_path, action="copied"))
                continue
            if artifact.artifact_type == PRODUCT_PARQUET:
                reports.append(migrate_parquet_product(artifact, output_path, prefer=prefer))
            elif artifact.artifact_type == PRODUCT_CSV:
                reports.append(migrate_csv_product(artifact, output_path, prefer=prefer))
            elif artifact.artifact_type == REVIEW_DB:
                reports.append(migrate_review_db(artifact, output_path))
            elif artifact.artifact_type == CANDIDATES_JSONL:
                reports.append(migrate_candidates_jsonl(artifact, output_path))
            else:
                reports.append(_report_from_artifact(artifact, output_path=output_path, action="copied"))

    report, unclassified = _normal_report_paths(
        output_root=target,
        report_path=report_path,
        unclassified_columns_path=unclassified_columns_path,
        scan_only=scan_only,
    )
    if report is not None and unclassified is not None:
        _write_reports(reports, report_path=report, unclassified_columns_path=unclassified)

    return MigrationSummary(
        input_root=str(source),
        output_root=str(target) if target is not None else None,
        report_path=str(report) if report is not None else None,
        unclassified_columns_path=str(unclassified) if unclassified is not None else None,
        artifacts=reports,
    )
