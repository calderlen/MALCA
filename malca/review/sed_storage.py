"""Versioned, provenance-preserving storage for SED measurements and fits.

The tables in this module intentionally live alongside the legacy
``sed_photometry`` and ``sed_model_*`` tables.  Native catalog measurements are
immutable; calibrations and other derived representations are stored as
separately versioned normalization rows.  This lets new calibration recipes be
tested without rewriting the catalog values that produced an older fit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


SED_MEASUREMENT_TABLE = "sed_measurements"
SED_NORMALIZATION_TABLE = "sed_measurement_normalizations"
SED_FIT_RUN_TABLE = "sed_fit_runs"
SED_FIT_INPUT_TABLE = "sed_fit_inputs"
SED_STORAGE_META_TABLE = "sed_storage_meta"
SED_STORAGE_SCHEMA_VERSION = 5

SED_MEASUREMENT_COLUMNS = [
    "measurement_id",
    "candidate_id",
    "source",
    "catalog",
    "release",
    "catalog_object_id",
    "catalog_measurement_id",
    "exposure_id",
    "instrument",
    "band",
    "epoch_mjd",
    "native_value",
    "native_error",
    "native_unit",
    "observable_kind",
    "is_upper_limit",
    "is_synthetic",
    "quality_flags",
    "quality_status",
    "match_sep_arcsec",
    "match_probability",
    "response_id",
    "calibration_id",
    "passband_fidelity",
    "fit_policy",
    "correlation_group",
    "raw_measurement_json",
    "provenance_json",
    "ingestion_version",
    "created_at",
]

SED_NORMALIZATION_COLUMNS = [
    "measurement_id",
    "normalization_version",
    "flux_nu_jy",
    "flux_nu_jy_err",
    "flux_lambda",
    "flux_lambda_err",
    "lambda_l_lambda",
    "lambda_l_lambda_err",
    "lambda_pivot_angstrom",
    "lambda_mean_angstrom",
    "lambda_nominal_angstrom",
    "lambda_reference_angstrom",
    "lambda_isophotal_angstrom",
    "lambda_effective_angstrom",
    "plot_lambda_angstrom",
    "plot_lambda_kind",
    "response_hash",
    "calibration_hash",
    "normalization_hash",
    "normalization_method",
    "provenance_json",
    "created_at",
]

SED_FIT_RUN_COLUMNS = [
    "fit_run_id",
    "candidate_id",
    "model_family",
    "fit_version",
    "photometry_method",
    "extinction_law",
    "status",
    "measurement_set_hash",
    "candidate_context_hash",
    "response_manifest_hash",
    "calibration_manifest_hash",
    "fit_recipe_hash",
    "model_grid_hash",
    "model_grid_provenance_json",
    "input_policy_manifest_hash",
    "fit_run_hash",
    "input_count",
    "used_input_count",
    "input_manifest_json",
    "recipe_json",
    "result_summary_json",
    "started_at",
    "completed_at",
    "created_at",
]

SED_FIT_INPUT_COLUMNS = [
    "fit_run_id",
    "measurement_id",
    "normalization_version",
    "fit_role",
    "used",
    "exclusion_reason",
    "correlation_group",
    "passband_fidelity",
    "fit_policy",
    "quality_flags",
    "fit_sigma_log",
    "fit_sigma_log_stat",
    "fit_sigma_log_systematic",
    "response_hash",
    "calibration_hash",
    "normalization_hash",
    "input_hash",
]

_JSON_COLUMNS = {
    "quality_flags",
    "raw_measurement_json",
    "provenance_json",
    "input_manifest_json",
    "recipe_json",
    "result_summary_json",
    "model_grid_provenance_json",
}
_DIRECT_JY_SOURCE_TOKENS = ("spitzer", "akari", "iras", "herschel")


class ImmutableSedRecordError(ValueError):
    """Raised when an existing immutable row would be changed in place."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_json(value: Any) -> str:
    """Return stable JSON suitable for provenance records and hashes."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def stable_sed_hash(value: Any) -> str:
    """Hash a JSON-like value after deterministic normalization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _records(rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return [dict(record) for record in rows.to_dict(orient="records")]
    if isinstance(rows, Mapping):
        return [dict(rows)]
    return [dict(record) for record in rows]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _bool_int(value: Any) -> int:
    if _missing(value):
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "t", "yes", "y"})
    return int(bool(value))


def _sql_value(value: Any, *, json_column: bool = False) -> Any:
    if _missing(value):
        return None
    if json_column and not isinstance(value, str):
        return canonical_json(value)
    if isinstance(value, (dict, list, tuple, set)):
        return canonical_json(value)
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def make_sed_measurement_id(row: Mapping[str, Any]) -> str:
    """Create a stable ID from the catalog observation's natural identity.

    Native flux or magnitude values are deliberately not included.  Reusing a
    catalog observation ID with different values is therefore detected as an
    attempted mutation instead of silently producing a second observation.
    """

    identity = canonical_sed_measurement_identity(row)
    if (
        not identity.get("candidate_id")
        or not (identity.get("source") or identity.get("catalog"))
        or not identity.get("band")
    ):
        raise ValueError("SED measurements require candidate_id, source/catalog, and band")
    return f"sedm_{stable_sed_hash(identity)[:32]}"


def _valid_identifier(value: Any) -> bool:
    if _missing(value):
        return False
    return str(value).strip().casefold() not in {"", "none", "nan", "<na>"}


def canonical_sed_measurement_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize every supported identity alias into the storage vocabulary."""

    return {
        "candidate_id": _jsonable(
            _first_present(row, "candidate_id", "asas_sn_id", "gaia_id", "source_id")
        ),
        "source": _jsonable(_first_present(row, "source", "catalog")),
        "catalog": _jsonable(_first_present(row, "catalog", "source")),
        "release": _jsonable(_first_present(row, "release", "catalog_release", "data_release")),
        "catalog_object_id": _jsonable(
            _first_present(row, "catalog_object_id", "source_object_id", "object_id")
        ),
        "catalog_measurement_id": _jsonable(
            _first_present(
                row,
                "catalog_measurement_id",
                "source_measurement_id",
                "observation_id",
            )
        ),
        "exposure_id": _jsonable(_first_present(row, "exposure_id", "visit_id")),
        "instrument": _jsonable(_first_present(row, "instrument", "camera")),
        "band": _jsonable(_first_present(row, "band")),
        "epoch_mjd": _jsonable(_first_present(row, "epoch_mjd", "observation_mjd", "mjd")),
    }


def make_sed_normalization_hash(row: Mapping[str, Any]) -> str:
    """Hash every stored scientific normalization field exactly once."""

    return stable_sed_hash(
        {
            column: _jsonable(row.get(column))
            for column in SED_NORMALIZATION_COLUMNS
            if column not in {"normalization_hash", "created_at"}
        }
    )


def make_sed_input_hash(row: Mapping[str, Any]) -> str:
    """Hash the complete per-measurement fit decision and calibration link."""

    return stable_sed_hash(
        {
            column: _jsonable(row.get(column))
            for column in SED_FIT_INPUT_COLUMNS
            if column not in {"fit_run_id", "input_hash"}
        }
    )


def make_sed_input_manifest_hash(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> str:
    inputs = []
    for raw in _records(rows):
        row = dict(raw)
        row["used"] = _bool_int(row.get("used"))
        row["input_hash"] = make_sed_input_hash(row)
        inputs.append(
            {
                column: _jsonable(row.get(column))
                for column in SED_FIT_INPUT_COLUMNS
                if column != "fit_run_id"
            }
        )
    inputs.sort(key=canonical_json)
    return stable_sed_hash(inputs)


def hash_sed_measurement_set(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any],
) -> str:
    """Hash the exact measurement/normalization inputs independent of row order."""

    records = []
    for raw in _records(rows):
        record = {
            key: _jsonable(raw.get(key))
            for key in (
                "measurement_id",
                "normalization_version",
                "normalization_hash",
                "response_hash",
                "calibration_hash",
            )
        }
        records.append(record)
    records.sort(key=canonical_json)
    return stable_sed_hash(records)


def make_sed_fit_run_hash(row: Mapping[str, Any]) -> str:
    """Hash all version identifiers that define a reproducible fit run."""

    return stable_sed_hash(
        {
            key: _jsonable(row.get(key))
            for key in (
                "candidate_id",
                "model_family",
                "fit_version",
                "photometry_method",
                "extinction_law",
                "measurement_set_hash",
                "candidate_context_hash",
                "response_manifest_hash",
                "calibration_manifest_hash",
                "fit_recipe_hash",
                "model_grid_hash",
                "input_policy_manifest_hash",
            )
        }
    )


def make_sed_fit_result_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return result content without self-referential legacy linkage fields."""

    return {
        str(key): _jsonable(value)
        for key, value in row.items()
        if str(key) not in {"fit_run_id"}
    }


def make_sed_result_fit_run_id(row: Mapping[str, Any]) -> str:
    supplied_run_hash = row.get("fit_run_hash")
    if not _valid_identifier(supplied_run_hash):
        raise ValueError("A result-linked fit_run_id requires fit_run_hash")
    run_hash = str(supplied_run_hash).strip()
    result_hash = stable_sed_hash(make_sed_fit_result_summary(row))
    return f"sedfit_{run_hash[:16]}_{result_hash[:16]}"


def _assert_sed_foreign_keys_enabled(conn: sqlite3.Connection) -> None:
    enabled = conn.execute("PRAGMA foreign_keys").fetchone()
    if enabled is None or int(enabled[0]) != 1:
        raise RuntimeError(
            "SED ledger writes require SQLite foreign-key enforcement; "
            "open the database with malca.review.store.db_connect()"
        )


def validate_sed_storage_integrity(conn: sqlite3.Connection) -> None:
    """Validate the additive SED schema and surface historical FK damage.

    This is intentionally diagnostic only: an older database is never rebuilt
    or rewritten to manufacture constraints.  Missing columns/constraints and
    orphaned rows fail loudly so a deliberate migration can preserve history.
    """

    _assert_sed_foreign_keys_enabled(conn)
    expected_columns = {
        SED_MEASUREMENT_TABLE: set(SED_MEASUREMENT_COLUMNS),
        SED_NORMALIZATION_TABLE: set(SED_NORMALIZATION_COLUMNS),
        SED_FIT_RUN_TABLE: set(SED_FIT_RUN_COLUMNS),
        SED_FIT_INPUT_TABLE: set(SED_FIT_INPUT_COLUMNS),
    }
    expected_references = {
        SED_MEASUREMENT_TABLE: {"candidates"},
        SED_NORMALIZATION_TABLE: {SED_MEASUREMENT_TABLE},
        SED_FIT_RUN_TABLE: {"candidates"},
        SED_FIT_INPUT_TABLE: {SED_FIT_RUN_TABLE, SED_NORMALIZATION_TABLE},
    }
    for table, required in expected_columns.items():
        actual = {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        missing = sorted(required - actual)
        if missing:
            raise RuntimeError(
                f"SED storage table {table!r} is missing required column(s): "
                + ", ".join(missing)
            )
        references = {
            str(row[2])
            for row in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        }
        missing_references = sorted(expected_references[table] - references)
        if missing_references:
            raise RuntimeError(
                f"SED storage table {table!r} is missing foreign-key reference(s): "
                + ", ".join(missing_references)
            )

        violations = conn.execute(f"PRAGMA foreign_key_check({table})").fetchall()
        if violations:
            preview = "; ".join(
                f"table={row[0]} rowid={row[1]} parent={row[2]} fk={row[3]}"
                for row in violations[:5]
            )
            raise sqlite3.IntegrityError(
                f"SED storage foreign-key violation(s) in {table}: {preview}"
            )


def ensure_sed_storage_schema(
    conn: sqlite3.Connection,
    *,
    validate: bool = False,
) -> bool:
    """Create or migrate the additive SED schema.

    The current-schema path is read-only: its metadata timestamp is not
    rewritten.  Full foreign-key validation is intentionally opt-in because it
    scans stored SED rows and does not belong on routine connection paths.

    Returns ``True`` when the stored schema version changed.
    """

    _assert_sed_foreign_keys_enabled(conn)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SED_STORAGE_META_TABLE} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    stored_version_row = conn.execute(
        f"SELECT value FROM {SED_STORAGE_META_TABLE} WHERE key = 'schema_version'"
    ).fetchone()
    stored_version = None
    if stored_version_row is not None:
        try:
            stored_version = int(stored_version_row[0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("SED storage schema_version is not an integer") from exc
        if stored_version > SED_STORAGE_SCHEMA_VERSION:
            raise RuntimeError(
                "This MALCA build only understands SED storage schema "
                f"{SED_STORAGE_SCHEMA_VERSION}, but the database is version {stored_version}"
            )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SED_MEASUREMENT_TABLE} (
            measurement_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            source TEXT NOT NULL,
            catalog TEXT,
            release TEXT,
            catalog_object_id TEXT,
            catalog_measurement_id TEXT,
            exposure_id TEXT,
            instrument TEXT,
            band TEXT NOT NULL,
            epoch_mjd REAL,
            native_value REAL,
            native_error REAL,
            native_unit TEXT,
            observable_kind TEXT,
            is_upper_limit INTEGER NOT NULL DEFAULT 0 CHECK(is_upper_limit IN (0, 1)),
            is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK(is_synthetic IN (0, 1)),
            quality_flags TEXT,
            quality_status TEXT,
            match_sep_arcsec REAL,
            match_probability REAL,
            response_id TEXT,
            calibration_id TEXT,
            passband_fidelity TEXT,
            fit_policy TEXT,
            correlation_group TEXT,
            raw_measurement_json TEXT,
            provenance_json TEXT,
            ingestion_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SED_NORMALIZATION_TABLE} (
            measurement_id TEXT NOT NULL,
            normalization_version TEXT NOT NULL,
            flux_nu_jy REAL,
            flux_nu_jy_err REAL,
            flux_lambda REAL,
            flux_lambda_err REAL,
            lambda_l_lambda REAL,
            lambda_l_lambda_err REAL,
            lambda_pivot_angstrom REAL,
            lambda_mean_angstrom REAL,
            lambda_nominal_angstrom REAL,
            lambda_reference_angstrom REAL,
            lambda_isophotal_angstrom REAL,
            lambda_effective_angstrom REAL,
            plot_lambda_angstrom REAL,
            plot_lambda_kind TEXT,
            response_hash TEXT,
            calibration_hash TEXT,
            normalization_hash TEXT NOT NULL,
            normalization_method TEXT,
            provenance_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(measurement_id, normalization_version),
            FOREIGN KEY(measurement_id) REFERENCES {SED_MEASUREMENT_TABLE}(measurement_id)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SED_FIT_RUN_TABLE} (
            fit_run_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            model_family TEXT,
            fit_version TEXT NOT NULL,
            photometry_method TEXT,
            extinction_law TEXT,
            status TEXT,
            measurement_set_hash TEXT NOT NULL,
            candidate_context_hash TEXT,
            response_manifest_hash TEXT,
            calibration_manifest_hash TEXT,
            fit_recipe_hash TEXT NOT NULL,
            model_grid_hash TEXT,
            model_grid_provenance_json TEXT,
            input_policy_manifest_hash TEXT,
            fit_run_hash TEXT NOT NULL,
            input_count INTEGER,
            used_input_count INTEGER,
            input_manifest_json TEXT,
            recipe_json TEXT,
            result_summary_json TEXT,
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_fit_run_columns = {
        str(row[1]).lower()
        for row in conn.execute(f"PRAGMA table_info({SED_FIT_RUN_TABLE})").fetchall()
    }
    for column, dtype in {
        "candidate_context_hash": "TEXT",
        "model_grid_hash": "TEXT",
        "model_grid_provenance_json": "TEXT",
        "input_policy_manifest_hash": "TEXT",
    }.items():
        if column not in existing_fit_run_columns:
            conn.execute(f"ALTER TABLE {SED_FIT_RUN_TABLE} ADD COLUMN {column} {dtype}")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SED_FIT_INPUT_TABLE} (
            fit_run_id TEXT NOT NULL,
            measurement_id TEXT NOT NULL,
            normalization_version TEXT NOT NULL,
            fit_role TEXT,
            used INTEGER NOT NULL DEFAULT 0 CHECK(used IN (0, 1)),
            exclusion_reason TEXT,
            correlation_group TEXT,
            passband_fidelity TEXT,
            fit_policy TEXT,
            quality_flags TEXT,
            fit_sigma_log REAL,
            fit_sigma_log_stat REAL,
            fit_sigma_log_systematic REAL,
            response_hash TEXT,
            calibration_hash TEXT,
            normalization_hash TEXT,
            input_hash TEXT NOT NULL,
            PRIMARY KEY(fit_run_id, measurement_id),
            FOREIGN KEY(fit_run_id) REFERENCES {SED_FIT_RUN_TABLE}(fit_run_id),
            FOREIGN KEY(measurement_id, normalization_version)
                REFERENCES {SED_NORMALIZATION_TABLE}(measurement_id, normalization_version)
        )
        """
    )
    existing_fit_input_columns = {
        str(row[1]).lower()
        for row in conn.execute(f"PRAGMA table_info({SED_FIT_INPUT_TABLE})").fetchall()
    }
    for column, dtype in {
        "passband_fidelity": "TEXT",
        "fit_policy": "TEXT",
        "quality_flags": "TEXT",
        "fit_sigma_log": "REAL",
        "fit_sigma_log_stat": "REAL",
        "fit_sigma_log_systematic": "REAL",
    }.items():
        if column not in existing_fit_input_columns:
            conn.execute(f"ALTER TABLE {SED_FIT_INPUT_TABLE} ADD COLUMN {column} {dtype}")

    index_statements = (
        f"CREATE INDEX IF NOT EXISTS idx_sed_measurements_candidate ON {SED_MEASUREMENT_TABLE}(candidate_id)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_measurements_source_band ON {SED_MEASUREMENT_TABLE}(source, band)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_measurements_catalog_object "
        f"ON {SED_MEASUREMENT_TABLE}(catalog, catalog_object_id)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_measurements_candidate_epoch "
        f"ON {SED_MEASUREMENT_TABLE}(candidate_id, epoch_mjd)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_normalizations_version "
        f"ON {SED_NORMALIZATION_TABLE}(normalization_version)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_normalizations_response "
        f"ON {SED_NORMALIZATION_TABLE}(response_hash, calibration_hash)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_fit_runs_candidate ON {SED_FIT_RUN_TABLE}(candidate_id, created_at)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_fit_runs_hash ON {SED_FIT_RUN_TABLE}(fit_run_hash)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_fit_inputs_measurement ON {SED_FIT_INPUT_TABLE}(measurement_id)",
        f"CREATE INDEX IF NOT EXISTS idx_sed_fit_inputs_used ON {SED_FIT_INPUT_TABLE}(fit_run_id, used)",
    )
    for statement in index_statements:
        conn.execute(statement)
    schema_changed = stored_version != SED_STORAGE_SCHEMA_VERSION
    if schema_changed:
        conn.execute(
            f"""
            INSERT INTO {SED_STORAGE_META_TABLE} (key, value, updated_at)
            VALUES ('schema_version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            WHERE {SED_STORAGE_META_TABLE}.value IS NOT excluded.value
            """,
            (str(SED_STORAGE_SCHEMA_VERSION), _utc_now()),
        )
    if validate:
        validate_sed_storage_integrity(conn)
    return schema_changed


@contextmanager
def _savepoint(conn: sqlite3.Connection, *, commit: bool) -> Iterable[None]:
    _assert_sed_foreign_keys_enabled(conn)
    name = f"sed_storage_{uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        conn.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name}")
        if commit:
            conn.commit()


def _normalize_columns(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return {
        column: _sql_value(row.get(column), json_column=column in _JSON_COLUMNS)
        for column in columns
    }


def _insert_immutable_rows(
    conn: sqlite3.Connection,
    *,
    table: str,
    columns: Sequence[str],
    key_columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    ignored_comparison_columns: Sequence[str] = ("created_at",),
) -> int:
    if not records:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    inserted = 0
    compare_columns = [column for column in columns if column not in ignored_comparison_columns]
    for raw in records:
        record = _normalize_columns(raw, columns)
        cursor = conn.execute(sql, [record[column] for column in columns])
        if cursor.rowcount == 1:
            inserted += 1
            continue
        where = " AND ".join(f"{column} = ?" for column in key_columns)
        existing_cursor = conn.execute(
            f"SELECT {', '.join(compare_columns)} FROM {table} WHERE {where}",
            [record[column] for column in key_columns],
        )
        existing_values = existing_cursor.fetchone()
        if existing_values is None:
            raise sqlite3.IntegrityError(f"Could not insert or locate {table} row")
        existing = dict(zip(compare_columns, existing_values))
        changed = [column for column in compare_columns if existing[column] != record[column]]
        if changed:
            key_text = ", ".join(f"{column}={record[column]!r}" for column in key_columns)
            raise ImmutableSedRecordError(
                f"Immutable {table} row ({key_text}) differs in: {', '.join(changed)}"
            )
    return inserted


def store_sed_measurements(
    conn: sqlite3.Connection,
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    commit: bool = True,
) -> int:
    """Insert native measurements, accepting exact repeats but never mutations."""

    prepared: list[dict[str, Any]] = []
    for raw in _records(rows):
        row = dict(raw)
        row["source"] = _first_present(row, "source", "catalog")
        row["catalog"] = _first_present(row, "catalog", "source")
        if any(_missing(row.get(column)) for column in ("candidate_id", "source", "band")):
            raise ValueError("SED measurements require candidate_id, source/catalog, and band")
        if not _valid_identifier(row.get("measurement_id")):
            row["measurement_id"] = make_sed_measurement_id(row)
        if _missing(row.get("ingestion_version")):
            row["ingestion_version"] = "sed-storage-v3"
        if _missing(row.get("created_at")):
            row["created_at"] = _utc_now()
        row["is_upper_limit"] = _bool_int(row.get("is_upper_limit"))
        row["is_synthetic"] = _bool_int(row.get("is_synthetic"))
        prepared.append(row)
    with _savepoint(conn, commit=commit):
        return _insert_immutable_rows(
            conn,
            table=SED_MEASUREMENT_TABLE,
            columns=SED_MEASUREMENT_COLUMNS,
            key_columns=("measurement_id",),
            records=prepared,
            # These legacy columns are interpretation snapshots, not native
            # catalog identity.  Current policy is content-addressed in the
            # fitted normalization and input ledger instead.  Ignoring them
            # avoids duplicating one native observation after a registry fix.
            ignored_comparison_columns=(
                "created_at",
                "response_id",
                "calibration_id",
                "passband_fidelity",
                "fit_policy",
                "correlation_group",
            ),
        )


def store_sed_normalizations(
    conn: sqlite3.Connection,
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    commit: bool = True,
) -> int:
    """Store immutable normalization versions derived from native measurements."""

    prepared: list[dict[str, Any]] = []
    for raw in _records(rows):
        row = dict(raw)
        if not _valid_identifier(row.get("measurement_id")) or not _valid_identifier(
            row.get("normalization_version")
        ):
            raise ValueError("SED normalizations require measurement_id and normalization_version")
        if _missing(row.get("created_at")):
            row["created_at"] = _utc_now()
        supplied_value = row.get("normalization_hash")
        supplied_hash = str(supplied_value).strip() if _valid_identifier(supplied_value) else ""
        expected_hash = make_sed_normalization_hash(row)
        if supplied_hash and supplied_hash != expected_hash:
            raise ValueError(
                f"normalization_hash mismatch for {row['measurement_id']!r}: "
                f"expected {expected_hash}, received {supplied_hash}"
            )
        row["normalization_hash"] = expected_hash
        prepared.append(row)
    with _savepoint(conn, commit=commit):
        return _insert_immutable_rows(
            conn,
            table=SED_NORMALIZATION_TABLE,
            columns=SED_NORMALIZATION_COLUMNS,
            key_columns=("measurement_id", "normalization_version"),
            records=prepared,
            # Pre-v5 rows used a narrower hash contract.  Their scientific
            # fields remain immutable and are compared above; the historical
            # hash itself must not make an exact legacy replay fail.
            ignored_comparison_columns=("created_at", "normalization_hash"),
        )


def _read_frame(conn: sqlite3.Connection, sql: str, params: Sequence[Any], columns: Sequence[str]) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn, params=tuple(params))
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return pd.DataFrame(columns=list(columns))


def load_sed_measurements(
    conn: sqlite3.Connection,
    candidate_id: str | None = None,
    *,
    measurement_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    where: list[str] = []
    params: list[Any] = []
    if candidate_id is not None:
        where.append("candidate_id = ?")
        params.append(str(candidate_id))
    ids = [str(value) for value in (measurement_ids or [])]
    if ids:
        where.append(f"measurement_id IN ({', '.join('?' for _ in ids)})")
        params.extend(ids)
    suffix = f" WHERE {' AND '.join(where)}" if where else ""
    return _read_frame(
        conn,
        f"SELECT {', '.join(SED_MEASUREMENT_COLUMNS)} FROM {SED_MEASUREMENT_TABLE}{suffix} "
        "ORDER BY candidate_id, epoch_mjd, source, band, measurement_id",
        params,
        SED_MEASUREMENT_COLUMNS,
    )


def load_sed_normalizations(
    conn: sqlite3.Connection,
    candidate_id: str | None = None,
    *,
    normalization_version: str | None = None,
    measurement_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    where: list[str] = []
    params: list[Any] = []
    if candidate_id is not None:
        where.append("m.candidate_id = ?")
        params.append(str(candidate_id))
    if normalization_version is not None:
        where.append("n.normalization_version = ?")
        params.append(str(normalization_version))
    ids = [str(value) for value in (measurement_ids or [])]
    if ids:
        where.append(f"n.measurement_id IN ({', '.join('?' for _ in ids)})")
        params.extend(ids)
    suffix = f" WHERE {' AND '.join(where)}" if where else ""
    return _read_frame(
        conn,
        f"SELECT {', '.join(f'n.{column}' for column in SED_NORMALIZATION_COLUMNS)} "
        f"FROM {SED_NORMALIZATION_TABLE} n JOIN {SED_MEASUREMENT_TABLE} m "
        f"ON m.measurement_id = n.measurement_id{suffix} "
        "ORDER BY m.candidate_id, n.measurement_id, n.normalization_version",
        params,
        SED_NORMALIZATION_COLUMNS,
    )


def load_prepared_sed_measurements(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    normalization_version: str,
) -> pd.DataFrame:
    """Load native and derived fields from one explicitly selected recipe."""

    m_columns = ", ".join(f"m.{column}" for column in SED_MEASUREMENT_COLUMNS)
    n_columns = ", ".join(
        f"n.{column} AS normalized_{column}"
        for column in SED_NORMALIZATION_COLUMNS
        if column != "measurement_id"
    )
    columns = [
        *SED_MEASUREMENT_COLUMNS,
        *(
            f"normalized_{column}"
            for column in SED_NORMALIZATION_COLUMNS
            if column != "measurement_id"
        ),
    ]
    return _read_frame(
        conn,
        f"SELECT {m_columns}, {n_columns} FROM {SED_MEASUREMENT_TABLE} m "
        f"JOIN {SED_NORMALIZATION_TABLE} n ON n.measurement_id = m.measurement_id "
        "WHERE m.candidate_id = ? AND n.normalization_version = ? "
        "ORDER BY n.plot_lambda_angstrom, m.source, m.band, m.measurement_id",
        (str(candidate_id), str(normalization_version)),
        columns,
    )


def _first_present(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if not _missing(value):
            return value
    return None


def _canonical_native_fields(row: Mapping[str, Any]) -> tuple[Any, Any, str | None, str]:
    """Extract the original catalog observable from a canonical SED row."""

    explicit_kind = str(_first_present(row, "observable_kind") or "").strip().lower()
    system = str(_first_present(row, "mag_system") or "").strip().upper()
    native_unit = str(_first_present(row, "native_unit", "native_flux_unit") or "").strip()
    is_quoted_fnu = explicit_kind == "quoted_fnu" or system == "JY" or native_unit.lower() == "jy"
    if is_quoted_fnu:
        return (
            _first_present(
                row,
                "native_value",
                "native_flux_nu_jy",
                "observed_flux_nu_jy",
                "flux_nu_jy",
            ),
            _first_present(
                row,
                "native_error",
                "native_flux_nu_jy_err",
                "observed_flux_nu_jy_err",
                "flux_nu_jy_err",
            ),
            "Jy",
            explicit_kind or "quoted_fnu",
        )
    magnitude = _first_present(row, "native_value", "mag")
    if magnitude is not None and (system in {"AB", "VEGA"} or explicit_kind.endswith("_mag")):
        kind = explicit_kind or ("ab_mag" if system == "AB" else "vega_mag")
        return magnitude, _first_present(row, "native_error", "mag_err"), "mag", kind
    if magnitude is not None and (native_unit.lower() in {"mag", "magnitude"} or system):
        return magnitude, _first_present(row, "native_error", "mag_err"), "mag", explicit_kind or "magnitude"
    flux_nu = _first_present(row, "observed_flux_nu_jy", "flux_nu_jy")
    if flux_nu is not None:
        return (
            flux_nu,
            _first_present(row, "observed_flux_nu_jy_err", "flux_nu_jy_err"),
            "Jy",
            explicit_kind or "flux_nu",
        )
    flux_lambda = row.get("flux_lambda")
    if not _missing(flux_lambda):
        return (
            flux_lambda,
            row.get("flux_lambda_err"),
            "erg s-1 cm-2 Angstrom-1",
            explicit_kind or "flux_lambda",
        )
    return None, None, native_unit or None, explicit_kind or "unknown"


def _canonical_native_payload(
    row: Mapping[str, Any],
    *,
    observable_kind: str,
) -> dict[str, Any]:
    """Retain catalog fields without folding derived v3 fluxes into identity."""

    identity_and_quality = (
        "candidate_id",
        "source",
        "catalog",
        "release",
        "catalog_release",
        "data_release",
        "catalog_object_id",
        "source_object_id",
        "object_id",
        "catalog_measurement_id",
        "source_measurement_id",
        "observation_id",
        "exposure_id",
        "visit_id",
        "instrument",
        "camera",
        "band",
        "epoch_mjd",
        "observation_mjd",
        "mjd",
        "is_upper_limit",
        "is_synthetic",
        "quality_flags",
        "quality_status",
        "match_sep_arcsec",
        "sep_arcsec",
        "match_probability",
    )
    native_fields = (
        ("mag", "mag_err", "mag_system", "native_value", "native_error", "native_unit")
        if observable_kind.endswith("_mag") or observable_kind == "magnitude"
        else (
            "flux_nu_jy",
            "flux_nu_jy_err",
            "native_flux_nu_jy",
            "native_flux_nu_jy_err",
            "native_value",
            "native_error",
            "native_unit",
            "native_flux_unit",
            "observable_kind",
        )
    )
    return {
        key: row.get(key)
        for key in (*identity_and_quality, *native_fields)
        if key in row and not _missing(row.get(key))
    }


def prepare_canonical_sed_rows(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    normalization_version: str = "sed-measurement-v3",
    ingestion_version: str = "canonical-sed-v3",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map fetched/canonical SED rows onto native and derived v3 records.

    AB and Vega rows retain their catalog magnitude as the native observable.
    Rows declared as ``Jy``/``quoted_fnu`` retain their catalog Jy value even
    when a compatibility magnitude is also present.  No input frame is
    modified.
    """

    measurements: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    for raw in _records(rows):
        source = str(_first_present(raw, "source", "catalog") or "").strip()
        candidate_id = str(_first_present(raw, "candidate_id") or "").strip()
        band = str(_first_present(raw, "band") or "").strip()
        if not candidate_id or not source or not band:
            raise ValueError("Canonical SED rows require candidate_id, source/catalog, and band")
        native_value, native_error, native_unit, observable_kind = _canonical_native_fields(raw)
        supplied_measurement_id = raw.get("measurement_id")
        measurement = {
            "measurement_id": supplied_measurement_id if _valid_identifier(supplied_measurement_id) else None,
            "candidate_id": candidate_id,
            "source": source,
            "catalog": _first_present(raw, "catalog") or source,
            "release": _first_present(raw, "release", "catalog_release", "data_release"),
            "catalog_object_id": _first_present(
                raw,
                "catalog_object_id",
                "source_object_id",
                "object_id",
            ),
            "catalog_measurement_id": _first_present(
                raw,
                "catalog_measurement_id",
                "source_measurement_id",
                "observation_id",
            ),
            "exposure_id": _first_present(raw, "exposure_id", "visit_id"),
            "instrument": _first_present(raw, "instrument", "camera"),
            "band": band,
            "epoch_mjd": _first_present(raw, "epoch_mjd", "observation_mjd", "mjd"),
            "native_value": native_value,
            "native_error": native_error,
            "native_unit": native_unit,
            "observable_kind": observable_kind,
            "is_upper_limit": raw.get("is_upper_limit"),
            "is_synthetic": raw.get("is_synthetic"),
            "quality_flags": raw.get("quality_flags"),
            "quality_status": _first_present(raw, "quality_status")
            or ("rejected" if "bad_quality" in str(raw.get("quality_flags") or "") else "unparsed"),
            "match_sep_arcsec": _first_present(raw, "match_sep_arcsec", "sep_arcsec"),
            "match_probability": raw.get("match_probability"),
            "response_id": _first_present(raw, "response_id", "svo_filter_id"),
            "calibration_id": _first_present(raw, "calibration_id", "calibration_source"),
            "passband_fidelity": _first_present(
                raw,
                "passband_fidelity",
                "response_kind",
            ),
            "fit_policy": raw.get("fit_policy"),
            "correlation_group": raw.get("correlation_group"),
            "raw_measurement_json": _first_present(raw, "raw_measurement_json")
            or _canonical_native_payload(raw, observable_kind=observable_kind),
            "provenance_json": _first_present(
                raw,
                "measurement_provenance_json",
                "provenance_json",
            )
            or {"prepared_from": "canonical_sed_row"},
            "ingestion_version": ingestion_version,
        }
        if not _valid_identifier(measurement.get("measurement_id")):
            measurement["measurement_id"] = make_sed_measurement_id(measurement)
        measurements.append(measurement)

        plot_lambda = _first_present(raw, "plot_lambda_angstrom", "lambda_eff_angstrom")
        normalizations.append(
            {
                "measurement_id": measurement["measurement_id"],
                "normalization_version": _first_present(raw, "normalization_version")
                or normalization_version,
                "flux_nu_jy": _first_present(raw, "observed_flux_nu_jy", "flux_nu_jy"),
                "flux_nu_jy_err": _first_present(
                    raw,
                    "observed_flux_nu_jy_err",
                    "flux_nu_jy_err",
                ),
                "flux_lambda": raw.get("flux_lambda"),
                "flux_lambda_err": raw.get("flux_lambda_err"),
                "lambda_l_lambda": raw.get("lambda_l_lambda"),
                "lambda_l_lambda_err": raw.get("lambda_l_lambda_err"),
                "lambda_pivot_angstrom": raw.get("lambda_pivot_angstrom"),
                "lambda_mean_angstrom": raw.get("lambda_mean_angstrom"),
                "lambda_nominal_angstrom": raw.get("lambda_nominal_angstrom"),
                "lambda_reference_angstrom": raw.get("lambda_reference_angstrom"),
                "lambda_isophotal_angstrom": raw.get("lambda_isophotal_angstrom"),
                "lambda_effective_angstrom": raw.get("lambda_eff_angstrom"),
                "plot_lambda_angstrom": plot_lambda,
                "plot_lambda_kind": _first_present(raw, "plot_lambda_kind")
                or ("legacy_effective" if plot_lambda is not None else None),
                "response_hash": raw.get("response_hash"),
                "calibration_hash": raw.get("calibration_hash"),
                # Recompute under the current complete normalization contract.
                # Stored legacy rows are accepted by scientific-field equality
                # in store_sed_normalizations without rewriting their old hash.
                "normalization_hash": None,
                "normalization_method": _first_present(
                    raw,
                    "normalization_method",
                    "photometry_method",
                )
                or normalization_version,
                "provenance_json": _first_present(raw, "normalization_provenance_json")
                or {"prepared_from": "canonical_sed_row"},
            }
        )
    return measurements, normalizations


def store_canonical_sed_rows(
    conn: sqlite3.Connection,
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    normalization_version: str = "sed-measurement-v3",
    ingestion_version: str = "canonical-sed-v3",
    commit: bool = True,
) -> tuple[int, int]:
    """Atomically persist fetched rows as native measurements and normalizations."""

    measurements, normalizations = prepare_canonical_sed_rows(
        rows,
        normalization_version=normalization_version,
        ingestion_version=ingestion_version,
    )
    with _savepoint(conn, commit=commit):
        n_measurements = store_sed_measurements(conn, measurements, commit=False)
        n_normalizations = store_sed_normalizations(conn, normalizations, commit=False)
    return n_measurements, n_normalizations


def store_sed_fit_run(
    conn: sqlite3.Connection,
    fit_run: Mapping[str, Any],
    inputs: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    *,
    commit: bool = True,
) -> tuple[int, int]:
    """Atomically store an immutable fit-run record and its exact inputs."""

    input_rows = _records(inputs)
    run = dict(fit_run)
    required = ("candidate_id", "fit_version", "measurement_set_hash", "fit_recipe_hash")
    missing = [field for field in required if not _valid_identifier(run.get(field))]
    if missing:
        raise ValueError(f"SED fit runs require: {', '.join(missing)}")
    if not input_rows:
        raise ValueError("A reproducible SED fit run requires at least one point/input")

    prepared_inputs: list[dict[str, Any]] = []
    seen_measurements: set[str] = set()
    for raw in input_rows:
        row = dict(raw)
        measurement_id = row.get("measurement_id")
        version = row.get("normalization_version")
        if not _valid_identifier(measurement_id) or not _valid_identifier(version):
            raise ValueError("SED fit inputs require measurement_id and normalization_version")
        measurement_id = str(measurement_id).strip()
        version = str(version).strip()
        if measurement_id in seen_measurements:
            raise ValueError(f"Duplicate SED fit input measurement_id: {measurement_id}")
        seen_measurements.add(measurement_id)
        stored = conn.execute(
            f"SELECT n.response_hash, n.calibration_hash, n.normalization_hash, m.candidate_id "
            f"FROM {SED_NORMALIZATION_TABLE} n JOIN {SED_MEASUREMENT_TABLE} m "
            "ON m.measurement_id = n.measurement_id "
            "WHERE n.measurement_id = ? AND n.normalization_version = ?",
            (measurement_id, version),
        ).fetchone()
        if stored is None:
            raise sqlite3.IntegrityError(
                f"SED fit input references missing normalization {measurement_id!r}/{version!r}"
            )
        if str(stored[3]) != str(run["candidate_id"]):
            raise sqlite3.IntegrityError(
                f"SED fit input {measurement_id!r} belongs to candidate {stored[3]!r}, "
                f"not {run['candidate_id']!r}"
            )
        for field, stored_value in zip(
            ("response_hash", "calibration_hash", "normalization_hash"),
            stored[:3],
            strict=True,
        ):
            supplied = row.get(field)
            if _jsonable(supplied) != _jsonable(stored_value):
                raise ValueError(
                    f"SED fit input {field} mismatch for {measurement_id!r}: "
                    f"stored={stored_value!r}, supplied={supplied!r}"
                )
        row["measurement_id"] = measurement_id
        row["normalization_version"] = version
        row["used"] = _bool_int(row.get("used"))
        expected_input_hash = make_sed_input_hash(row)
        supplied_input_hash = row.get("input_hash")
        if _valid_identifier(supplied_input_hash) and str(supplied_input_hash) != expected_input_hash:
            raise ValueError(
                f"SED fit input input_hash mismatch for {measurement_id!r}: "
                f"expected {expected_input_hash}, received {supplied_input_hash}"
            )
        row["input_hash"] = expected_input_hash
        prepared_inputs.append(row)

    expected_measurement_hash = hash_sed_measurement_set(prepared_inputs)
    if str(run.get("measurement_set_hash") or "") != expected_measurement_hash:
        raise ValueError(
            "SED fit run measurement_set_hash mismatch: "
            f"expected {expected_measurement_hash}, received {run.get('measurement_set_hash')}"
        )
    expected_input_count = len(prepared_inputs)
    expected_used_count = sum(_bool_int(item.get("used")) for item in prepared_inputs)
    for field, expected in (
        ("input_count", expected_input_count),
        ("used_input_count", expected_used_count),
    ):
        supplied = run.get(field)
        if supplied is not None and int(supplied) != expected:
            raise ValueError(f"SED fit run {field} mismatch: expected {expected}, received {supplied}")
        run[field] = expected

    manifest = [
        {
            column: _jsonable(row.get(column))
            for column in SED_FIT_INPUT_COLUMNS
            if column != "fit_run_id"
        }
        for row in prepared_inputs
    ]
    manifest.sort(key=canonical_json)
    supplied_manifest = run.get("input_manifest_json")
    if _valid_identifier(supplied_manifest):
        try:
            decoded_manifest = (
                json.loads(str(supplied_manifest))
                if isinstance(supplied_manifest, str)
                else supplied_manifest
            )
        except json.JSONDecodeError as exc:
            raise ValueError("SED fit run input_manifest_json is invalid JSON") from exc
        if canonical_json(decoded_manifest) != canonical_json(manifest):
            raise ValueError("SED fit run input_manifest_json does not match exact inputs")
    run["input_manifest_json"] = manifest
    expected_policy_hash = make_sed_input_manifest_hash(prepared_inputs)
    supplied_policy_hash = run.get("input_policy_manifest_hash")
    if _valid_identifier(supplied_policy_hash) and str(supplied_policy_hash) != expected_policy_hash:
        raise ValueError(
            "SED fit run input_policy_manifest_hash mismatch: "
            f"expected {expected_policy_hash}, received {supplied_policy_hash}"
        )
    run["input_policy_manifest_hash"] = expected_policy_hash

    expected_run_hash = make_sed_fit_run_hash(run)
    supplied_run_hash = run.get("fit_run_hash")
    if _valid_identifier(supplied_run_hash) and str(supplied_run_hash) != expected_run_hash:
        raise ValueError(
            f"SED fit run fit_run_hash mismatch: expected {expected_run_hash}, received {supplied_run_hash}"
        )
    run["fit_run_hash"] = expected_run_hash
    run["fit_run_id"] = run.get("fit_run_id") or f"sedfit_{expected_run_hash[:32]}"
    run["created_at"] = run.get("created_at") or _utc_now()
    for row in prepared_inputs:
        row["fit_run_id"] = run["fit_run_id"]

    with _savepoint(conn, commit=commit):
        n_runs = _insert_immutable_rows(
            conn,
            table=SED_FIT_RUN_TABLE,
            columns=SED_FIT_RUN_COLUMNS,
            key_columns=("fit_run_id",),
            records=(run,),
        )
        n_inputs = _insert_immutable_rows(
            conn,
            table=SED_FIT_INPUT_TABLE,
            columns=SED_FIT_INPUT_COLUMNS,
            key_columns=("fit_run_id", "measurement_id"),
            records=prepared_inputs,
            ignored_comparison_columns=(),
        )
    return n_runs, n_inputs


def sed_point_normalization_record(point: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical normalization ledger record for one fit point.

    Prepared fitter rows use the unprefixed ``flux_lambda`` fields, while the
    persisted diagnostic snapshot names the same values ``observed_*``.  This
    adapter is deliberately shared by hashing and persistence so those two
    representations cannot silently produce different normalization hashes.
    """

    def value(*names: str) -> Any:
        for name in names:
            if name in point:
                return point.get(name)
        return None

    provenance = point.get("normalization_provenance_json")
    if isinstance(provenance, str):
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = {"raw": provenance}
    if not isinstance(provenance, Mapping):
        provenance = {
            "calibration_id": point.get("calibration_id"),
            "fit_policy": point.get("fit_policy"),
            "passband_fidelity": point.get("passband_fidelity"),
        }
    method = point.get("normalization_method")
    if not _valid_identifier(method):
        method = "fitter_bandpass_calibrated_v8"
    plot_kind = point.get("plot_lambda_kind")
    plot_kind = str(plot_kind).strip() if _valid_identifier(plot_kind) else ""
    return {
        "measurement_id": str(point.get("measurement_id")).strip(),
        "normalization_version": str(point.get("normalization_version")).strip(),
        "flux_nu_jy": point.get("observed_flux_nu_jy"),
        "flux_nu_jy_err": point.get("observed_flux_nu_jy_err"),
        "flux_lambda": value("observed_flux_lambda", "flux_lambda"),
        "flux_lambda_err": value("observed_flux_lambda_err", "flux_lambda_err"),
        "lambda_l_lambda": value("observed_lambda_l_lambda", "lambda_l_lambda"),
        "lambda_l_lambda_err": value(
            "observed_lambda_l_lambda_err", "lambda_l_lambda_err"
        ),
        "lambda_effective_angstrom": point.get("lambda_eff_angstrom"),
        "lambda_pivot_angstrom": point.get("lambda_pivot_angstrom"),
        "lambda_mean_angstrom": point.get("lambda_mean_angstrom"),
        "lambda_nominal_angstrom": point.get("lambda_nominal_angstrom"),
        "lambda_reference_angstrom": point.get("lambda_reference_angstrom"),
        "lambda_isophotal_angstrom": point.get("lambda_isophotal_angstrom"),
        "plot_lambda_angstrom": point.get("plot_lambda_angstrom"),
        "plot_lambda_kind": plot_kind,
        "response_hash": point.get("response_hash")
        if _valid_identifier(point.get("response_hash"))
        else None,
        "calibration_hash": point.get("calibration_hash")
        if _valid_identifier(point.get("calibration_hash"))
        else None,
        "normalization_hash": point.get("normalization_hash"),
        "normalization_method": str(method),
        "provenance_json": dict(provenance),
    }


def store_sed_point_normalizations(
    conn: sqlite3.Connection,
    points: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    *,
    commit: bool = True,
) -> int:
    """Persist and verify the exact response-calibrated values used by a fit."""

    point_rows = _records(points)
    normalizations: list[dict[str, Any]] = []
    for point in point_rows:
        if not _valid_identifier(point.get("measurement_id")) or not _valid_identifier(
            point.get("normalization_version")
        ):
            raise ValueError("Fitted SED points require measurement_id and normalization_version")
        measurement_id = str(point.get("measurement_id")).strip()
        version = str(point.get("normalization_version")).strip()
        if not _valid_identifier(point.get("normalization_hash")):
            raise ValueError(
                f"Fitted SED point {measurement_id!r} requires a normalization_hash"
            )
        normalization = sed_point_normalization_record(point)
        normalization["measurement_id"] = measurement_id
        normalization["normalization_version"] = version
        normalizations.append(normalization)

    with _savepoint(conn, commit=commit):
        measurement_ids = sorted(
            {str(normalization["measurement_id"]) for normalization in normalizations}
        )
        existing_ids: set[str] = set()
        if measurement_ids:
            placeholders = ", ".join("?" for _ in measurement_ids)
            existing_ids = {
                str(row[0])
                for row in conn.execute(
                    f"SELECT measurement_id FROM {SED_MEASUREMENT_TABLE} "
                    f"WHERE measurement_id IN ({placeholders})",
                    measurement_ids,
                ).fetchall()
            }
        missing_ids = sorted(set(measurement_ids) - existing_ids)
        if missing_ids:
            raise sqlite3.IntegrityError(
                "Fitted SED normalizations reference missing native measurement(s): "
                + ", ".join(missing_ids[:5])
            )
        inserted = store_sed_normalizations(conn, normalizations, commit=False)
        for normalization in normalizations:
            stored = conn.execute(
                f"SELECT normalization_hash FROM {SED_NORMALIZATION_TABLE} "
                "WHERE measurement_id = ? AND normalization_version = ?",
                (
                    normalization["measurement_id"],
                    normalization["normalization_version"],
                ),
            ).fetchone()
            if stored is None or str(stored[0] or "") != str(normalization["normalization_hash"]):
                raise ImmutableSedRecordError(
                    "Stored SED normalization hash does not match the fitted point for "
                    f"{normalization['measurement_id']!r}"
                )
    return inserted


def store_sed_fit_results(
    conn: sqlite3.Connection,
    fits: pd.DataFrame | Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    points: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    *,
    commit: bool = True,
) -> tuple[int, int]:
    """Persist reproducible fitter outputs as immutable run/input records.

    Rows from older fitters that lack the input and recipe hashes are left in
    the legacy tables by their caller and skipped here.  Once those hashes are
    present, every diagnostic point must carry a measurement ID and a
    normalization version; accepting less would make the current run ledger claim a
    level of reproducibility it does not have.
    """

    fit_rows = _records(fits)
    point_rows = _records(points)
    total_runs = 0
    total_inputs = 0
    with _savepoint(conn, commit=commit):
        for raw_fit in fit_rows:
            fit = dict(raw_fit)
            required = (
                "candidate_id",
                "fit_version",
                "measurement_set_hash",
                "candidate_context_hash",
                "fit_recipe_hash",
                "model_grid_hash",
            )
            if any(_missing(fit.get(column)) for column in required):
                continue
            candidate_id = str(fit["candidate_id"])
            candidate_points = [
                dict(row)
                for row in point_rows
                if str(row.get("candidate_id") or "") == candidate_id
            ]
            if not candidate_points:
                raise ValueError(
                    f"Reproducible SED fit {candidate_id!r} requires at least one input point"
                )
            invalid_inputs = [
                row
                for row in candidate_points
                if str(row.get("measurement_id") or "").strip().lower()
                in {"", "none", "nan", "<na>"}
                or str(row.get("normalization_version") or "").strip().lower()
                in {"", "none", "nan", "<na>"}
            ]
            if invalid_inputs:
                raise ValueError(
                    f"Reproducible SED fit {candidate_id!r} has "
                    f"{len(invalid_inputs)} point(s) without measurement/version IDs"
                )
            measurement_ids = [str(row["measurement_id"]) for row in candidate_points]
            if len(measurement_ids) != len(set(measurement_ids)):
                raise ValueError(
                    f"Reproducible SED fit {candidate_id!r} contains duplicate measurement IDs"
                )

            # The run ledger must point at the exact response-calibrated values
            # used by the optimizer, not the earlier registry/fallback v3
            # conversion.  Store and hash-check these immutable versions first.
            store_sed_point_normalizations(conn, candidate_points, commit=False)

            inputs = [
                {
                    "measurement_id": row.get("measurement_id"),
                    "normalization_version": row.get("normalization_version"),
                    "fit_role": row.get("fit_role"),
                    "used": row.get("used"),
                    "exclusion_reason": row.get("exclusion_reason"),
                    "correlation_group": row.get("correlation_group"),
                    "passband_fidelity": row.get("passband_fidelity"),
                    "fit_policy": row.get("fit_policy"),
                    "quality_flags": row.get("quality_flags"),
                    "fit_sigma_log": row.get("fit_sigma_log"),
                    "fit_sigma_log_stat": row.get("fit_sigma_log_stat"),
                    "fit_sigma_log_systematic": row.get("fit_sigma_log_systematic"),
                    "response_hash": row.get("response_hash"),
                    "calibration_hash": row.get("calibration_hash"),
                    "normalization_hash": row.get("normalization_hash"),
                    "input_hash": row.get("input_hash"),
                }
                for row in candidate_points
            ]
            supplied_run_hash = fit.get("fit_run_hash")
            run_hash = (
                str(supplied_run_hash).strip()
                if _valid_identifier(supplied_run_hash)
                else ""
            )
            run_record = {
                "candidate_id": candidate_id,
                "model_family": fit.get("model_family"),
                "fit_version": fit.get("fit_version"),
                "photometry_method": fit.get("photometry_method"),
                "extinction_law": fit.get("extinction_law"),
                "status": fit.get("status"),
                "measurement_set_hash": fit.get("measurement_set_hash"),
                "candidate_context_hash": fit.get("candidate_context_hash"),
                "response_manifest_hash": fit.get("response_manifest_hash"),
                "calibration_manifest_hash": fit.get("calibration_manifest_hash"),
                "input_policy_manifest_hash": fit.get("input_policy_manifest_hash"),
                "fit_recipe_hash": fit.get("fit_recipe_hash"),
                "model_grid_hash": fit.get("model_grid_hash"),
                "model_grid_provenance_json": fit.get("model_grid_provenance_json"),
                "fit_run_hash": run_hash or None,
                "input_count": len(inputs),
                "used_input_count": sum(_bool_int(row.get("used")) for row in inputs),
                "recipe_json": {
                    "fit_version": fit.get("fit_version"),
                    "fit_recipe_hash": fit.get("fit_recipe_hash"),
                    "model_grid_hash": fit.get("model_grid_hash"),
                    "priors_json": fit.get("priors_json"),
                },
                "result_summary_json": None,
            }
            if not run_hash:
                run_hash = make_sed_fit_run_hash(run_record)
                run_record["fit_run_hash"] = run_hash
                fit["fit_run_hash"] = run_hash
            run_record["result_summary_json"] = make_sed_fit_result_summary(fit)
            # A run hash identifies the scientific inputs/recipe.  Include the
            # immutable result hash in the row ID so a failed attempt and a
            # later successful attempt can coexist under the same run hash,
            # while an exact replay remains idempotent.
            run_record["fit_run_id"] = make_sed_result_fit_run_id(fit)
            n_runs, n_inputs = store_sed_fit_run(
                conn,
                run_record,
                inputs,
                commit=False,
            )
            total_runs += n_runs
            total_inputs += n_inputs
    return total_runs, total_inputs


def load_sed_fit_runs(conn: sqlite3.Connection, candidate_id: str | None = None) -> pd.DataFrame:
    suffix = " WHERE candidate_id = ?" if candidate_id is not None else ""
    params: tuple[Any, ...] = (str(candidate_id),) if candidate_id is not None else ()
    return _read_frame(
        conn,
        f"SELECT {', '.join(SED_FIT_RUN_COLUMNS)} FROM {SED_FIT_RUN_TABLE}{suffix} "
        "ORDER BY created_at, fit_run_id",
        params,
        SED_FIT_RUN_COLUMNS,
    )


def load_sed_fit_inputs(conn: sqlite3.Connection, fit_run_id: str) -> pd.DataFrame:
    return _read_frame(
        conn,
        f"SELECT {', '.join(SED_FIT_INPUT_COLUMNS)} FROM {SED_FIT_INPUT_TABLE} "
        "WHERE fit_run_id = ? ORDER BY measurement_id",
        (str(fit_run_id),),
        SED_FIT_INPUT_COLUMNS,
    )


def _legacy_native_fields(row: Mapping[str, Any]) -> tuple[Any, Any, str | None, str]:
    source = str(row.get("source") or "").lower()
    flux_nu = row.get("flux_nu_jy")
    if any(token in source for token in _DIRECT_JY_SOURCE_TOKENS) and not _missing(flux_nu):
        return flux_nu, row.get("flux_nu_jy_err"), "Jy", "quoted_fnu"
    if not _missing(row.get("mag")):
        system = str(row.get("mag_system") or "").strip().lower()
        kind = "ab_mag" if system == "ab" else "vega_mag" if system == "vega" else "magnitude"
        return row.get("mag"), row.get("mag_err"), "mag", kind
    if not _missing(flux_nu):
        return flux_nu, row.get("flux_nu_jy_err"), "Jy", "flux_nu"
    if not _missing(row.get("flux_lambda")):
        return row.get("flux_lambda"), row.get("flux_lambda_err"), "erg s-1 cm-2 Angstrom-1", "flux_lambda"
    return None, None, None, "legacy_unknown"


def _legacy_passband_fidelity(source: Any) -> str:
    token = str(source or "").strip().lower()
    if "apass" in token:
        return "standardized_proxy"
    if token == "nsc" or "noirlab nsc" in token or "noirlab source catalog" in token:
        return "mixed_unknown"
    return "legacy_unknown"


def migrate_legacy_sed_photometry(
    conn: sqlite3.Connection,
    *,
    normalization_version: str = "legacy-stored-v1",
    ingestion_version: str = "legacy-sed-photometry-v1",
    commit: bool = True,
) -> tuple[int, int]:
    """Copy legacy rows into the immutable schema without altering the source.

    The complete legacy row is retained in ``raw_measurement_json``.  The old
    ambiguous wavelength is explicitly labeled ``legacy_effective`` rather
    than being reinterpreted as a pivot or mission reference wavelength.
    """

    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sed_photometry'"
    ).fetchone()
    if table_exists is None:
        return 0, 0
    cursor = conn.execute("SELECT * FROM sed_photometry ORDER BY candidate_id, source, band")
    names = [str(item[0]) for item in cursor.description or []]
    legacy_rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    measurements: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    for legacy in legacy_rows:
        source = str(legacy.get("source") or "legacy")
        candidate_id = str(legacy.get("candidate_id") or "")
        band = str(legacy.get("band") or "")
        native_value, native_error, native_unit, observable_kind = _legacy_native_fields(legacy)
        identity = {
            "candidate_id": candidate_id,
            "source": source,
            "catalog": source,
            "release": "legacy-review-db",
            "catalog_measurement_id": f"{candidate_id}:{source}:{band}:{ingestion_version}",
            "band": band,
        }
        measurement_id = make_sed_measurement_id(identity)
        mag_system = str(legacy.get("mag_system") or "").strip()
        quality_flags = legacy.get("quality_flags")
        calibration_id = (
            f"legacy:{source}:quoted_fnu"
            if observable_kind == "quoted_fnu"
            else f"legacy:{mag_system.lower()}" if mag_system else None
        )
        measurements.append(
            {
                **identity,
                "measurement_id": measurement_id,
                "native_value": native_value,
                "native_error": native_error,
                "native_unit": native_unit,
                "observable_kind": observable_kind,
                "is_upper_limit": legacy.get("is_upper_limit") or 0,
                "is_synthetic": legacy.get("is_synthetic") or 0,
                "quality_flags": quality_flags,
                "quality_status": "legacy_unparsed" if not _missing(quality_flags) else "unknown",
                "match_sep_arcsec": legacy.get("sep_arcsec"),
                "response_id": legacy.get("svo_filter_id"),
                "calibration_id": calibration_id,
                "passband_fidelity": _legacy_passband_fidelity(source),
                "raw_measurement_json": legacy,
                "provenance_json": {"migrated_from": "sed_photometry"},
                "ingestion_version": ingestion_version,
            }
        )
        old_wavelength = legacy.get("lambda_eff_angstrom")
        normalizations.append(
            {
                "measurement_id": measurement_id,
                "normalization_version": normalization_version,
                "flux_nu_jy": legacy.get("flux_nu_jy"),
                "flux_nu_jy_err": legacy.get("flux_nu_jy_err"),
                "flux_lambda": legacy.get("flux_lambda"),
                "flux_lambda_err": legacy.get("flux_lambda_err"),
                "lambda_l_lambda": legacy.get("lambda_l_lambda"),
                "lambda_l_lambda_err": legacy.get("lambda_l_lambda_err"),
                "lambda_effective_angstrom": old_wavelength,
                "plot_lambda_angstrom": old_wavelength,
                "plot_lambda_kind": "legacy_effective",
                "normalization_method": "legacy_stored_conversion",
                "provenance_json": {"migrated_from": "sed_photometry"},
            }
        )

    with _savepoint(conn, commit=commit):
        n_measurements = store_sed_measurements(conn, measurements, commit=False)
        n_normalizations = store_sed_normalizations(conn, normalizations, commit=False)
    return n_measurements, n_normalizations
