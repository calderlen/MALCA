from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import json
import math
import shutil
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd

from malca.config import GAIA_CHUNK_SIZE
from malca.config import LTV_MAX_PM
from malca.config import (
    VSX_CROSSMATCH_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CACHE_DIR,
    GAIA_CACHE_FILE,
    GAIA_LOCAL_CATALOG,
    LEGACY_GAIA_CACHE_FILE,
    REVIEW_IMPORTED_LC_CACHE_DIR,
)
from malca.core.derived_stats import DERIVED_FEATURE_COLUMNS
from malca.ltv.cmd import dustmaps_cmd_from_fields
from malca.products.feature_layers import (
    FEATURE_LAYER_VERSION_COLUMN,
    FEATURE_LAYER_COLUMNS,
    parse_layer_value,
    to_layer_first_frame,
)
from malca.catalogs.evidence import (
    CATALOG_NEIGHBOR_COLUMNS,
    DEFAULT_REVIEW_VETTING_RADIUS_ARCSEC,
    MAX_REVIEW_VETTING_RADIUS_ARCSEC,
    normalize_catalog_neighbor_frame,
)
from malca.ltv.multi_survey import LTV_MS_FEATURE_COLUMN_SPECS
from malca.enrichment.multi_survey_features import MS_FEATURE_COLUMN_SPECS
from malca.enrichment.open_clusters import (
    OPEN_CLUSTER_OUTPUT_COLUMNS as OPEN_CLUSTER_REVIEW_COLUMNS,
)
from malca.microlensing.schema import MICROLENSING_JOINT_COLUMN_SPECS
from malca.review.filter_schema import REVIEW_FILTER_COLUMN_TYPES
from malca.review.dipper_recurrence import (
    DIPPER_RECURRENCE_CLASS_COLUMN,
    DIPPER_RECURRENCE_EVIDENCE_COLUMN,
    add_observed_dipper_recurrence,
)
from malca.review.metadata import has_catalog_vetting_context, has_known_catalog_evidence, normalize_vsx_record
from malca.io.table_io import read_feature_table, read_parquet_table, write_feature_table, write_parquet_table
from malca.review.taxonomy import (
    REVIEW_TAXONOMY_FIELDS,
    REVIEW_TAXONOMY_SQL_COLUMNS,
    TAXONOMY_VERSION,
    derive_event_class,
    empty_taxonomy_selection,
    json_list,
    normalize_selection,
    selection_from_review,
)







DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / DEFAULT_OUTPUT_DIR / "review" / "review.db"
DEFAULT_STANDALONE_DB_PATH = Path(__file__).resolve().parents[2] / DEFAULT_OUTPUT_DIR / "review" / "standalone.db"
SQLITE_BUSY_TIMEOUT_MS = 30_000
REVIEW_CANDIDATES_SCHEMA_KEY = "review_candidates_schema"
REVIEW_CANDIDATES_SCHEMA_VERSION = "flat_v1"
REVIEW_DB_SCHEMA_KEY = "review_db_schema_version"
REVIEW_DB_SCHEMA_VERSION = 24
REVIEW_CONTENT_REVISION_SCOPE = "candidates_reviews"
STATUS_OPTIONS = ["unreviewed", "reviewed", "needs_followup"]
EVENT_CLASS_OPTIONS = [
    "unclassified",
    "dipper",
    "ltv",
    "microlensing",
    "flare",
    "instrumental",
    "other",
]


# These are class-balanced, eight-way LightGBM ranking scores.  The rejection
# class intentionally combines artifact/bad-photometry and nonvariable/low-SNR
# human labels.  They remain separate from the older binary
# ``prob_dipper_like`` score so a review run retains both model products.
EIGHT_CLASS_PROBABILITY_COLUMNS = (
    "prob_artifact_or_nonvariable",
    "prob_brightening_event",
    "prob_dipper",
    "prob_eclipsing_binary_like",
    "prob_long_period_variable",
    "prob_long_term_variable",
    "prob_microlensing",
    "prob_quasi_periodic",
)

HIERARCHICAL_ML_PROBABILITY_COLUMNS = (
    "prob_hierarchical_artifact_or_nonvariable",
    "prob_usable_astrophysical_variable",
    "prob_primary_dipper_dimming_given_usable",
    "prob_primary_eb_geometric_periodic_given_usable",
    "prob_primary_long_timescale_variable_given_usable",
    "prob_primary_brightening_transient_given_usable",
    "prob_primary_other_structured_variable_given_usable",
    "prob_dipper_dimming",
    "prob_eb_geometric_periodic",
    "prob_long_timescale_variable",
    "prob_brightening_transient",
    "prob_other_structured_variable",
    "prob_quasi_periodic_given_usable",
    "prob_quasi_periodic_hierarchical",
    "prob_microlensing_given_brightening",
    "prob_microlensing_hierarchical",
    "prob_long_period_variable_given_long_timescale",
    "prob_long_term_variable_given_long_timescale",
    "prob_long_period_variable_hierarchical",
    "prob_long_term_variable_hierarchical",
    "prob_recurrent_given_dipper",
    "prob_single_given_dipper",
    "prob_recurrent_dipper_hierarchical",
    "prob_single_dipper_hierarchical",
)

HIERARCHICAL_ML_PREDICTION_COLUMNS = (
    "predicted_hierarchy_gate",
    "predicted_primary_morphology",
    "predicted_hierarchical_class",
    "predicted_quasi_periodic",
    "predicted_microlensing_like",
    "predicted_long_timescale_subtype",
    "predicted_dipper_recurrence",
)

# Retained only so pre-existing nine-class artifacts remain importable.
NINE_CLASS_PROBABILITY_COLUMNS = (
    "prob_artifact_or_bad_photometry",
    "prob_brightening_event",
    "prob_dipper",
    "prob_eclipsing_binary_like",
    "prob_long_period_variable",
    "prob_long_term_variable",
    "prob_microlensing",
    "prob_nonvariable_or_low_snr",
    "prob_quasi_periodic",
)

DIPPER_RECURRENCE_ML_PROBABILITY_COLUMNS = (
    "prob_recurrent_given_dipper",
    "prob_recurrent_dipper_binary",
    "prob_recurrent_dipper_eight_class",
    "prob_recurrent_dipper_nine_class",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (str(table_name),),
    ).fetchone()
    return row is not None


def _stored_review_db_schema_version(conn: sqlite3.Connection) -> int | None:
    if not _table_exists(conn, "app_state"):
        return None
    row = conn.execute(
        "SELECT value FROM app_state WHERE key = ?",
        (REVIEW_DB_SCHEMA_KEY,),
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _ensure_review_content_revision_tracking(conn: sqlite3.Connection) -> None:
    """Install revision tracking for candidate/review content only.

    App state, cache metadata, SED tables, and other bookkeeping deliberately do
    not affect this revision.  Review caches therefore refresh only when a
    candidate or review row actually changes.
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_content_revision (
            scope TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO review_content_revision (scope, revision, updated_at)
        VALUES (?, 1, ?)
        """,
        (REVIEW_CONTENT_REVISION_SCOPE, _utc_now()),
    )
    for table_name in ("candidates", "reviews"):
        for operation in ("insert", "update", "delete"):
            trigger_name = f"review_content_revision_{table_name}_{operation}"
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS {trigger_name}
                AFTER {operation.upper()} ON {table_name}
                BEGIN
                    UPDATE review_content_revision
                    SET revision = revision + 1,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE scope = '{REVIEW_CONTENT_REVISION_SCOPE}';
                END
                """
            )


def _as_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _to_float(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        x = float(v)
        if np.isnan(x):
            return None
        return x
    except Exception:
        return None


def _normalize_large_integer_like_id(v) -> str | None:
    """Normalize large integer-like identifiers to plain strings.

    Converts values like 4.272990850383009e+17 -> "427299085038300900".
    Returns None for missing values.
    """
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s
    if d == d.to_integral_value():
        try:
            return format(d.to_integral_value(), "f")
        except Exception:
            return s
    return s


def _opt_str(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    return str(v) if v is not None else None


def _opt_bool(d: dict[str, Any], key: str) -> int | None:
    v = d.get(key)
    return int(_as_bool(v)) if v is not None else None


def _canonicalize_wise_fields(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    legacy_map = {
        "allwise_w3": "w3",
        "allwise_w3_err": "w3_err",
        "allwise_w4": "w4",
        "allwise_w4_err": "w4_err",
        "W1_W2": "w1_w2",
        "W1_W2_dered": "w1_w2_dered",
    }
    for old_col, new_col in legacy_map.items():
        if old_col not in out:
            continue
        if new_col not in out or out.get(new_col) is None:
            out[new_col] = out[old_col]
        out.pop(old_col, None)

    for left, right in (
        ("w1", "w2"),
        ("w1", "w3"),
        ("w1", "w4"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
    ):
        color_col = f"{left}_{right}"
        if color_col in out and out.get(color_col) is not None:
            continue
        left_value = _to_float(out.get(left))
        right_value = _to_float(out.get(right))
        if left_value is not None and right_value is not None:
            out[color_col] = left_value - right_value
    return out


def _payload_json_mapping(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _is_payload_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


_REVIEW_PAYLOAD_ALIASES: dict[str, tuple[str, ...]] = {
    "gaia_id": ("source_id_gaia",),
    "ruwe": ("ruwe_gaia",),
    "parallax": ("parallax_gaia",),
    "parallax_error": ("parallax_error_gaia",),
    "pmra": ("pmra_gaia",),
    "pmra_error": ("pmra_error_gaia",),
    "pmdec": ("pmdec_gaia",),
    "pmdec_error": ("pmdec_error_gaia",),
    "gaia_eb_period": ("period_gaia_eb_days",),
    "gaia_eb_morph": ("period_gaia_eb_class",),
}


def _canonicalize_review_payload_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill canonical Review fields from known pipeline aliases."""
    out = dict(payload)
    for canonical, aliases in _REVIEW_PAYLOAD_ALIASES.items():
        if not _is_payload_missing(out.get(canonical)):
            continue
        for alias in aliases:
            if not _is_payload_missing(out.get(alias)):
                out[canonical] = out[alias]
                break
    return out


def _flatten_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a flat review payload, unpacking any old layer/nested wrappers."""
    out: dict[str, Any] = {}
    nested_payload = _payload_json_mapping(payload.get("payload_json"))
    if nested_payload:
        out.update(_flatten_review_payload(nested_payload))

    for layer in FEATURE_LAYER_COLUMNS:
        layer_data = parse_layer_value(payload.get(layer))
        for key, value in layer_data.items():
            if not _is_payload_missing(value) or key not in out:
                out[key] = value

    for key, value in payload.items():
        name = str(key)
        if name in FEATURE_LAYER_COLUMNS or name in {FEATURE_LAYER_VERSION_COLUMN, "payload_json"}:
            continue
        if not _is_payload_missing(value) or name not in out:
            out[name] = value
    return out


def _payload_layer_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    for layer in FEATURE_LAYER_COLUMNS:
        layer_payload = parse_layer_value(payload.get(layer))
        if key in layer_payload:
            return layer_payload.get(key)
    return None


def _drop_payload_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    out = _flatten_review_payload(payload)
    for key in keys:
        out.pop(key, None)
    return out


def _merge_layer_payload_updates(payload: dict[str, Any], updates: dict[str, object]) -> dict[str, Any]:
    out = _flatten_review_payload(payload)
    for key, value in _payload_update_values(updates).items():
        out[str(key)] = value
    return out


def _payload_update_values(updates: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {
        str(key): value
        for key, value in updates.items()
        if str(key) not in FEATURE_LAYER_COLUMNS
    }
    for layer in FEATURE_LAYER_COLUMNS:
        values.update(parse_layer_value(updates.get(layer)))
    return values


def _review_payload_extra(payload: dict[str, Any], table_cols: set[str] | None = None) -> dict[str, Any]:
    """Return flat payload keys that are not stored as first-class SQL columns."""
    sql_cols = set(table_cols or _COL_NAMES)
    skip = {
        "candidate_id",
        "source_path",
        "imported_at",
        FEATURE_LAYER_VERSION_COLUMN,
        *FEATURE_LAYER_COLUMNS,
    }
    out: dict[str, Any] = {}
    for key, value in _flatten_review_payload(payload).items():
        name = str(key)
        if name in skip or name in sql_cols:
            continue
        if _is_payload_missing(value):
            continue
        out[name] = value
    return out


def _sql_value_for_column(column: str, value: object) -> object:
    if _is_payload_missing(value):
        return None
    etype = _COL_TYPE_MAP.get(column)
    if etype == "bool":
        return int(_as_bool(value))
    if etype == "float":
        return _to_float(value)
    return str(value)


def _candidate_insert_tuple_from_row_dict(
    row_dict: dict[str, Any],
    *,
    source_path: str | None = None,
    imported_at: str | None = None,
) -> tuple[Any, ...]:
    normalized = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
    normalized = _flatten_review_payload(normalized)
    normalized = normalize_vsx_record(normalized)
    normalized = _canonicalize_wise_fields(normalized)
    normalized = _canonicalize_review_payload_aliases(normalized)

    candidate_id = _normalize_large_integer_like_id(normalized.get("candidate_id"))
    if not candidate_id:
        raise ValueError("Candidate rows must include a non-empty candidate_id")
    normalized["candidate_id"] = candidate_id

    if _payload_layer_value(normalized, "gaia_id") is not None:
        normalized["gaia_id"] = _normalize_large_integer_like_id(_payload_layer_value(normalized, "gaia_id"))
    if _payload_layer_value(normalized, "source_id") is not None:
        normalized["source_id"] = _normalize_large_integer_like_id(_payload_layer_value(normalized, "source_id"))

    if not normalized.get("asassn_var_type") and normalized.get("period_asassn_var_class"):
        normalized["asassn_var_type"] = normalized.get("period_asassn_var_class")

    if not normalized.get("ztf_var_type") and normalized.get("period_ztf_periodic_class"):
        normalized["ztf_var_type"] = normalized.get("period_ztf_periodic_class")

    if _payload_layer_value(normalized, "high_pm_flag") in (None, "") and _payload_layer_value(normalized, "pm_total") not in (None, ""):
        pm_total = _to_float(_payload_layer_value(normalized, "pm_total"))
        if pm_total is not None:
            normalized["high_pm_flag"] = bool(pm_total > LTV_MAX_PM)

    if has_known_catalog_evidence(normalized):
        normalized["vetting_likely_known"] = True
    elif has_catalog_vetting_context(normalized):
        normalized["vetting_likely_known"] = False

    row_source_path = str(source_path if source_path is not None else normalized.get("source_path") or "")
    row_imported_at = str(imported_at or normalized.get("imported_at") or _utc_now())

    vals: list[Any] = [candidate_id, row_source_path]
    for col, _dtype, etype in _CANDIDATE_COLUMNS:
        raw = _payload_layer_value(normalized, col)
        if etype == "bool":
            vals.append(int(_as_bool(raw)) if raw is not None else None)
        elif etype == "float":
            vals.append(_to_float(raw))
        else:
            vals.append(str(raw) if raw is not None else None)
    vals.append(json.dumps(_review_payload_extra(normalized), default=str))
    vals.append(row_imported_at)
    return tuple(vals)


def upsert_candidates_frame(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    *,
    default_source_path: str | None = None,
) -> tuple[int, int]:
    """Upsert candidate rows from a DataFrame while preserving payload metadata."""
    if df.empty:
        return 0, 0

    df_use = df.copy()
    df_use["candidate_id"] = infer_candidate_id(df_use)

    rows = [
        _candidate_insert_tuple_from_row_dict(
            {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
            source_path=default_source_path,
        )
        for _, row in df_use.iterrows()
    ]

    all_col_names = ["candidate_id", "source_path"] + _COL_NAMES + ["payload_json", "imported_at"]
    candidate_cols = ", ".join(all_col_names)
    placeholders = ", ".join(["?"] * len(all_col_names))
    update_cols = [c for c in all_col_names if c != "candidate_id"]
    conflict_set = ", ".join(f"{c}=excluded.{c}" for c in update_cols)

    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO candidates ({candidate_cols})
        VALUES ({placeholders})
        ON CONFLICT(candidate_id) DO UPDATE SET {conflict_set}
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return len(rows), int(after - before)


def _parse_updated_at(value: object) -> datetime:
    if value in (None, "", b""):
        return datetime.min.replace(tzinfo=timezone.utc)
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def infer_candidate_id(df: pd.DataFrame) -> pd.Series:
    if "candidate_id" not in df.columns:
        raise ValueError("Input must include a 'candidate_id' column.")

    vals = df["candidate_id"].astype(str).str.strip()
    if not vals.nunique(dropna=True) == len(df):
        raise ValueError("'candidate_id' values must be unique.")
    return vals


def _candidate_ids_from_columns(df: pd.DataFrame) -> pd.Series:
    ids = pd.Series("", index=df.index, dtype="object")
    if "candidate_id" in df.columns:
        ids = df["candidate_id"].map(_normalize_large_integer_like_id).fillna("").astype(str).str.strip()

    missing = ids.eq("")
    for column in ("asas_sn_id", "source_id", "gaia_id"):
        if not bool(missing.any()) or column not in df.columns:
            continue
        fill = df.loc[missing, column].map(_normalize_large_integer_like_id).fillna("").astype(str).str.strip()
        ids.loc[missing] = fill
        missing = ids.eq("")

    for column in ("path", "dat_path", "lc_path", "local_lightcurve_path"):
        if not bool(missing.any()) or column not in df.columns:
            continue
        stems = df.loc[missing, column].map(
            lambda value: Path(str(value)).stem if value is not None and str(value).strip() else ""
        )
        ids.loc[missing] = stems.map(_normalize_large_integer_like_id).fillna("").astype(str).str.strip()
        missing = ids.eq("")

    return ids


def normalize_candidate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize review/reproduction candidate tables loaded from Parquet or CSV."""
    out = df.copy()
    out["candidate_id"] = _candidate_ids_from_columns(out)
    if "source_id" not in out.columns and "asas_sn_id" in out.columns:
        out["source_id"] = out["asas_sn_id"].map(_normalize_large_integer_like_id)
    if "lc_path" not in out.columns and "path" in out.columns:
        out["lc_path"] = out["path"].astype(str)
    return out


def load_candidates_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return normalize_candidate_input_frame(pd.read_csv(path))
    return normalize_candidate_input_frame(read_feature_table(path))


def detect_run_directory_files(run_dir: Path) -> dict[str, Path | None]:
    """
    Auto-detect MALCA review files from a run directory.

    Returns dict with keys:
    - 'candidates': Path to best candidates file found (or None)
    - 'plot_dir': Path to plots directory (or None)
    - 'gaia_cache': Path to gaia cache (or None)
    - 'run_params': Path to run_params.json (or None)
    - 'warnings': List of warning messages
    """
    results = {
        'candidates': None,
        'plot_dir': None,
        'gaia_cache': None,
        'run_params': None,
        'warnings': []
    }

    # Validate directory exists
    if not run_dir.exists():
        results['warnings'].append(f"Directory does not exist: {run_dir}")
        return results

    if not run_dir.is_dir():
        results['warnings'].append(f"Path is not a directory: {run_dir}")
        return results

    # Check for run_params.json (validates it's a run directory)
    run_params = run_dir / "run_params.json"
    if run_params.exists():
        results['run_params'] = run_params
    else:
        results['warnings'].append("run_params.json not found - may not be a MALCA run directory")

    # Detect candidates file.
    # Priority: vetted products first, then non-vetted products.
    candidates_priority: list[Path] = []

    vetted_exact = run_dir / "results" / "lc_events_vetted.parquet"
    if vetted_exact.exists():
        candidates_priority.append(vetted_exact)

    vetted_pattern = sorted(
        (run_dir / "results").glob("lc_events_vetted_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates_priority.extend(vetted_pattern)

    for rel_path in (
        "results/lc_events_spectra.parquet",
        "results/lc_events_neighbors.parquet",
        "results/lc_events_classified.parquet",
        "results/lc_events_enriched.parquet",
        "results/lc_events_characterized.parquet",
        "results/lc_events_filtered.parquet",
    ):
        candidate_file = run_dir / rel_path
        if candidate_file.exists():
            candidates_priority.append(candidate_file)

    for candidate_file in candidates_priority:
        if candidate_file.exists():
            results['candidates'] = candidate_file
            break

    if results['candidates'] is None:
        results['warnings'].append("No candidates file found in results/ directory")

    # Detect plot directory
    plot_dir = run_dir / "plots"
    if plot_dir.exists() and plot_dir.is_dir():
        results['plot_dir'] = plot_dir
    else:
        results['warnings'].append("plots/ directory not found")

    # Detect Gaia cache (optional, no warning if missing).  Prefer the unified
    # repo cache, then fall back to older per-run/global locations.
    for gaia_cache in (
        GAIA_LOCAL_CATALOG,
        GAIA_CACHE_FILE,
        run_dir / "gaia_cache" / "gaia_cache.parquet",
        LEGACY_GAIA_CACHE_FILE,
    ):
        if gaia_cache.exists():
            results['gaia_cache'] = gaia_cache
            break

    return results


# ---------------------------------------------------------------------------
# Single source of truth for all extracted candidate columns.
#
# Each entry: (column_name, sql_type, extract_type)
#   extract_type: 'bool' | 'float' | 'text'
#
# The order here determines column order in the DB table and INSERT.
# 'candidate_id', 'source_path', 'payload_json', 'imported_at' are handled
# separately (they aren't payload fields).
# ---------------------------------------------------------------------------
_CANDIDATE_COLUMNS: list[tuple[str, str, str]] = [
    # -- identification --
    ("timescale",                "TEXT",    "text"),
    ("asas_sn_id",               "TEXT",    "text"),
    ("lc_path",                  "TEXT",    "text"),
    ("source_id",                "TEXT",    "text"),
    ("gaia_id",                  "TEXT",    "text"),
    ("gaia_dr2_id",              "TEXT",    "text"),
    ("gaia_id_release",          "TEXT",    "text"),
    ("gaia_id_mapping_status",   "TEXT",    "text"),
    ("dr2_dr3_angular_distance_mas", "REAL", "float"),
    ("dr2_dr3_magnitude_difference", "REAL", "float"),
    ("ra",                       "REAL",    "float"),
    ("dec",                      "REAL",    "float"),
    ("asassn_field_key",         "TEXT",    "text"),
    ("asassn_fields",            "TEXT",    "text"),
    ("asassn_field_count",       "REAL",    "float"),
    ("asassn_field_key_fraction","REAL",    "float"),
    ("camera_name_key",         "TEXT",    "text"),
    ("camera_names",            "TEXT",    "text"),
    ("camera_name_count",       "REAL",    "float"),
    ("camera_name_key_fraction","REAL",    "float"),
    # -- top-level filter flags --
    ("failed_any",               "INTEGER", "bool"),
    ("filter_reason",            "TEXT",    "text"),
    ("periodic_flag",            "INTEGER", "bool"),
    ("catalog_match",            "INTEGER", "bool"),
    ("catalog_source",           "TEXT",    "text"),
    ("period_sources",           "TEXT",    "text"),
    ("period_n_sources",         "REAL",    "float"),
    ("period_consensus_days",    "REAL",    "float"),
    ("period_consensus_agree",   "INTEGER", "bool"),
    ("period_conflict_flag",     "INTEGER", "bool"),
    ("period_consensus_support", "REAL",    "float"),
    ("period_primary_source",    "TEXT",    "text"),
    ("period_source_periods",    "TEXT",    "text"),
    ("period_ogle_name",         "TEXT",    "text"),
    ("period_ogle_match",        "INTEGER", "bool"),
    ("period_ogle_days",         "REAL",    "float"),
    ("period_ogle_class",        "TEXT",    "text"),
    ("period_ogle_sep_arcsec",   "REAL",    "float"),
    ("high_ruwe_flag",           "INTEGER", "bool"),
    # -- periodicity --
    ("periodicity_score",        "REAL",    "float"),
    ("periodicity_period",       "REAL",    "float"),
    ("periodicity_method",       "TEXT",    "text"),
    ("periodicity_base_period",  "REAL",    "float"),
    ("periodicity_harmonic_factor","REAL",  "float"),
    ("periodicity_harmonic_objective","REAL","float"),
    ("periodicity_scatter_ratio","REAL",    "float"),
    ("periodicity_alias_flag",   "INTEGER", "bool"),
    ("periodicity_alias_matches","TEXT",    "text"),
    ("periodicity_bootstrap_sig","REAL",    "float"),
    ("periodicity_is_significant","INTEGER","bool"),
    ("periodicity_evidence_source", "TEXT",  "text"),
    ("periodicity_rejection_reason", "TEXT", "text"),
    ("periodicity_status",         "TEXT",    "text"),
    ("period_confidence",        "TEXT",    "text"),
    ("period_method",            "TEXT",    "text"),
    ("period_baseline_cycles",   "REAL",    "float"),
    ("period_confidence_reason", "TEXT",    "text"),
    ("period_native_days",       "REAL",    "float"),
    ("period_corrected_days",    "REAL",    "float"),
    ("period_for_fold_days",     "REAL",    "float"),
    ("period_evidence_summary",  "TEXT",    "text"),
    ("event_period_days",        "REAL",    "float"),
    ("event_period_method",      "TEXT",    "text"),
    ("event_period_n_events",    "REAL",    "float"),
    ("event_period_is_high_confidence", "INTEGER", "bool"),
    ("dip_run_epochs_json",      "TEXT",    "text"),
    ("jump_run_epochs_json",     "TEXT",    "text"),
    ("dip_epochs_source",        "TEXT",    "text"),
    ("dip_epochs_count",         "REAL",    "float"),
    ("long_ls_period_days",      "REAL",    "float"),
    ("long_ls_peak_power",       "REAL",    "float"),
    ("long_ls_fap_bootstrap",    "REAL",    "float"),
    ("long_ls_baseline_cycles",  "REAL",    "float"),
    ("long_ls_is_significant",   "INTEGER", "bool"),
    ("long_ls_status",           "TEXT",    "text"),
    ("pdm_period",               "REAL",    "float"),
    ("pdm_corrected_period",     "REAL",    "float"),
    ("pdm_harmonic_factor",      "REAL",    "float"),
    ("pdm_harmonic_objective",   "REAL",    "float"),
    ("pdm_harmonic_scatter_ratio","REAL",   "float"),
    ("pdm_alias_flag",           "INTEGER", "bool"),
    ("pdm_alias_matches",        "TEXT",    "text"),
    ("pdm_theta",                "REAL",    "float"),
    ("pdm_snr",                  "REAL",    "float"),
    ("pdm_bootstrap_sig",        "REAL",    "float"),
    ("pdm_is_significant",       "INTEGER", "bool"),
    ("ce_period",                "REAL",    "float"),
    ("ce_corrected_period",      "REAL",    "float"),
    ("ce_harmonic_factor",       "REAL",    "float"),
    ("ce_harmonic_objective",    "REAL",    "float"),
    ("ce_harmonic_scatter_ratio","REAL",    "float"),
    ("ce_alias_flag",            "INTEGER", "bool"),
    ("ce_alias_matches",         "TEXT",    "text"),
    ("ce_entropy",               "REAL",    "float"),
    ("ce_snr",                   "REAL",    "float"),
    ("ce_bootstrap_sig",         "REAL",    "float"),
    ("ce_is_significant",        "INTEGER", "bool"),
    ("lsp_bootstrap_sig",        "REAL",    "float"),
    ("lsp_power",                "REAL",    "float"),
    ("lsp_period",               "REAL",    "float"),
    ("lsp_is_alias",             "INTEGER", "bool"),
    ("lsp_is_significant",       "INTEGER", "bool"),
    ("phase_plot_ready",         "INTEGER", "bool"),
    ("phase_period_days",        "REAL",    "float"),
    ("phase_source",             "TEXT",    "text"),
    ("phase_quality_score",      "REAL",    "float"),
    # -- dip detection --
    ("dip_significant",          "INTEGER", "bool"),
    ("dip_best_morph",           "TEXT",    "text"),
    ("dip_best_log_bf",          "REAL",    "float"),
    ("dip_best_delta_bic",       "REAL",    "float"),
    ("dip_best_width_param",     "REAL",    "float"),
    ("dip_symmetry_score",       "REAL",    "float"),
    ("dip_best_amp",             "REAL",    "float"),
    ("dip_best_t0",              "REAL",    "float"),
    ("dip_best_alpha",           "REAL",    "float"),
    ("dip_best_tau",             "REAL",    "float"),
    ("dip_best_tau_rise",        "REAL",    "float"),
    ("dip_best_tau_fall",        "REAL",    "float"),
    ("dip_bayes_factor",         "REAL",    "float"),
    ("dip_best_p",               "REAL",    "float"),
    ("dip_best_mag_event",       "REAL",    "float"),
    ("dip_trigger_max",          "REAL",    "float"),
    ("dip_max_event_prob",       "REAL",    "float"),
    ("dip_trigger_threshold",    "REAL",    "float"),
    # -- dip runs --
    ("dip_count",                "REAL",    "float"),
    ("dip_run_count",            "REAL",    "float"),
    ("dip_max_run_points",       "REAL",    "float"),
    ("dip_max_run_duration",     "REAL",    "float"),
    ("dip_max_run_sum",          "REAL",    "float"),
    ("dip_max_run_max",          "REAL",    "float"),
    ("dip_max_run_cameras",      "REAL",    "float"),
    ("dip_max_log_bf_local",     "REAL",    "float"),
    # -- jump detection --
    ("jump_significant",         "INTEGER", "bool"),
    ("jump_best_morph",          "TEXT",    "text"),
    ("jump_best_log_bf",         "REAL",    "float"),
    ("jump_best_delta_bic",      "REAL",    "float"),
    ("jump_best_width_param",    "REAL",    "float"),
    ("jump_best_amp",            "REAL",    "float"),
    ("jump_best_t0",             "REAL",    "float"),
    ("jump_best_alpha",          "REAL",    "float"),
    ("jump_best_tau",            "REAL",    "float"),
    ("jump_best_tau_rise",       "REAL",    "float"),
    ("jump_best_tau_fall",       "REAL",    "float"),
    ("jump_bayes_factor",        "REAL",    "float"),
    ("jump_best_p",              "REAL",    "float"),
    ("jump_best_mag_event",      "REAL",    "float"),
    ("jump_trigger_max",         "REAL",    "float"),
    ("jump_max_event_prob",      "REAL",    "float"),
    ("jump_trigger_threshold",   "REAL",    "float"),
    # -- jump runs --
    ("jump_count",               "REAL",    "float"),
    ("jump_run_count",           "REAL",    "float"),
    ("jump_max_run_points",      "REAL",    "float"),
    ("jump_max_run_duration",    "REAL",    "float"),
    ("jump_max_run_sum",         "REAL",    "float"),
    ("jump_max_run_max",         "REAL",    "float"),
    ("jump_max_run_cameras",     "REAL",    "float"),
    ("jump_max_log_bf_local",    "REAL",    "float"),
    # -- dip recurrence --
    ("dipper_recurrence_class",         "TEXT",    "select"),
    ("dipper_recurrence_evidence",      "TEXT",    "text"),
    ("prob_recurrent_given_dipper",      "REAL",    "float"),
    ("prob_single_given_dipper",         "REAL",    "float"),
    ("prob_recurrent_dipper_binary",     "REAL",    "float"),
    ("prob_recurrent_dipper_eight_class", "REAL",   "float"),
    ("prob_recurrent_dipper_hierarchical", "REAL",  "float"),
    ("prob_single_dipper_hierarchical",  "REAL",    "float"),
    ("prob_recurrent_dipper_nine_class",  "REAL",   "float"),
    ("predicted_dipper_recurrence",      "TEXT",    "select"),
    ("dip_is_single_event",              "INTEGER", "bool"),
    ("dip_inter_event_spacing_median",   "REAL",    "float"),
    ("dip_inter_event_spacing_std",      "REAL",    "float"),
    ("dip_amplitude_consistency",        "REAL",    "float"),
    ("dip_duration_consistency",         "REAL",    "float"),
    # -- jump recurrence --
    ("jump_is_single_event",             "INTEGER", "bool"),
    ("jump_inter_event_spacing_median",  "REAL",    "float"),
    ("jump_inter_event_spacing_std",     "REAL",    "float"),
    ("jump_amplitude_consistency",       "REAL",    "float"),
    ("jump_duration_consistency",        "REAL",    "float"),
    # -- event scoring --
    ("prob_dipper_like",         "REAL",    "float"),
    ("prob_hierarchical_artifact_or_nonvariable", "REAL", "float"),
    ("prob_usable_astrophysical_variable", "REAL", "float"),
    ("prob_primary_dipper_dimming_given_usable", "REAL", "float"),
    ("prob_primary_eb_geometric_periodic_given_usable", "REAL", "float"),
    ("prob_primary_long_timescale_variable_given_usable", "REAL", "float"),
    ("prob_primary_brightening_transient_given_usable", "REAL", "float"),
    ("prob_primary_other_structured_variable_given_usable", "REAL", "float"),
    ("prob_dipper_dimming", "REAL", "float"),
    ("prob_eb_geometric_periodic", "REAL", "float"),
    ("prob_long_timescale_variable", "REAL", "float"),
    ("prob_brightening_transient", "REAL", "float"),
    ("prob_other_structured_variable", "REAL", "float"),
    ("prob_quasi_periodic_given_usable", "REAL", "float"),
    ("prob_quasi_periodic_hierarchical", "REAL", "float"),
    ("prob_microlensing_given_brightening", "REAL", "float"),
    ("prob_microlensing_hierarchical", "REAL", "float"),
    ("prob_long_period_variable_given_long_timescale", "REAL", "float"),
    ("prob_long_term_variable_given_long_timescale", "REAL", "float"),
    ("prob_long_period_variable_hierarchical", "REAL", "float"),
    ("prob_long_term_variable_hierarchical", "REAL", "float"),
    ("predicted_hierarchy_gate", "TEXT", "select"),
    ("predicted_primary_morphology", "TEXT", "select"),
    ("predicted_hierarchical_class", "TEXT", "select"),
    ("predicted_quasi_periodic", "TEXT", "select"),
    ("predicted_microlensing_like", "TEXT", "select"),
    ("predicted_long_timescale_subtype", "TEXT", "select"),
    ("prob_artifact_or_nonvariable", "REAL", "float"),
    ("prob_artifact_or_bad_photometry", "REAL", "float"),
    ("prob_brightening_event",   "REAL",    "float"),
    ("prob_dipper",              "REAL",    "float"),
    ("prob_eclipsing_binary_like", "REAL",  "float"),
    ("prob_long_period_variable", "REAL",   "float"),
    ("prob_long_term_variable",  "REAL",    "float"),
    ("prob_microlensing",        "REAL",    "float"),
    ("prob_nonvariable_or_low_snr", "REAL", "float"),
    ("prob_quasi_periodic",      "REAL",    "float"),
    ("dipper_score",             "REAL",    "float"),
    ("dipper_n_dips",            "REAL",    "float"),
    ("dipper_n_valid_dips",      "REAL",    "float"),
    ("jumper_score",             "REAL",    "float"),
    ("jumper_n_jumps",           "REAL",    "float"),
    ("jumper_n_valid_jumps",     "REAL",    "float"),
    # -- stellar parameters --
    ("ruwe",                     "REAL",    "float"),
    ("ref_epoch",                "REAL",    "float"),
    ("astrometric_params_solved", "REAL",   "float"),
    ("radial_velocity",          "REAL",    "float"),
    ("radial_velocity_error",    "REAL",    "float"),
    ("rv_amplitude_robust",      "REAL",    "float"),
    ("rv_nb_transits",           "REAL",    "float"),
    ("rv_chisq_pvalue",          "REAL",    "float"),
    ("rv_renormalised_gof",      "REAL",    "float"),
    ("rv_time_duration",         "REAL",    "float"),
    ("rv_method_used",           "REAL",    "float"),
    ("grvs_mag",                 "REAL",    "float"),
    ("teff_gspphot",             "REAL",    "float"),
    ("logg_gspphot",             "REAL",    "float"),
    ("mh_gspphot",               "REAL",    "float"),
    ("distance_gspphot",         "REAL",    "float"),
    ("parallax",                 "REAL",    "float"),
    ("parallax_error",           "REAL",    "float"),
    ("parallax_over_error",      "REAL",    "float"),
    ("pmra",                     "REAL",    "float"),
    ("pmra_error",               "REAL",    "float"),
    ("pmdec",                    "REAL",    "float"),
    ("pmdec_error",              "REAL",    "float"),
    ("parallax_pmra_corr",       "REAL",    "float"),
    ("parallax_pmdec_corr",      "REAL",    "float"),
    ("pmra_pmdec_corr",          "REAL",    "float"),
    ("astrometric_excess_noise", "REAL",    "float"),
    ("astrometric_excess_noise_sig", "REAL", "float"),
    ("astrometric_n_good_obs_al", "REAL",   "float"),
    ("astrometric_sigma5d_max",  "REAL",    "float"),
    ("visibility_periods_used",  "REAL",    "float"),
    ("ipd_frac_multi_peak",      "REAL",    "float"),
    ("ipd_frac_odd_win",         "REAL",    "float"),
    ("ipd_gof_harmonic_amplitude", "REAL",  "float"),
    ("duplicated_source",        "INTEGER", "bool"),
    ("non_single_star",          "REAL",    "float"),
    ("phot_variable_flag",       "TEXT",    "select"),
    ("has_epoch_photometry",     "INTEGER", "bool"),
    ("has_epoch_rv",             "INTEGER", "bool"),
    ("has_rvs",                  "INTEGER", "bool"),
    ("pm_total",                 "REAL",    "float"),
    ("high_pm_flag",             "INTEGER", "bool"),
    ("gaia_fetch_schema_version", "TEXT",   "text"),
    ("gaia_fetch_updated_at",    "TEXT",    "text"),
    ("gaia_enrichment_status",   "TEXT",    "select"),
    ("gaia_enrichment_source",   "TEXT",    "text"),
    ("gaia_astrometry_complete", "INTEGER", "bool"),
    ("gaia_banyan_input_complete", "INTEGER", "bool"),
    ("gaia_missing_fields_json", "TEXT",    "text"),
    ("gaia_enrichment_updated_at", "TEXT",  "text"),
    # -- photometry --
    ("phot_g_mean_mag",          "REAL",    "float"),
    ("phot_bp_mean_mag",         "REAL",    "float"),
    ("phot_rp_mean_mag",         "REAL",    "float"),
    ("bp_rp",                    "REAL",    "float"),
    ("phot_bp_rp_excess_factor", "REAL",    "float"),
    ("phot_bp_n_obs",            "REAL",    "float"),
    ("phot_rp_n_obs",            "REAL",    "float"),
    ("phot_bp_n_blended_transits", "REAL",  "float"),
    ("phot_rp_n_blended_transits", "REAL",  "float"),
    ("phot_bp_n_contaminated_transits", "REAL", "float"),
    ("phot_rp_n_contaminated_transits", "REAL", "float"),
    ("mg",                       "REAL",    "float"),
    ("mg0",                      "REAL",    "float"),
    ("bprp0",                    "REAL",    "float"),
    ("tmass_j",                  "REAL",    "float"),
    ("tmass_j_err",              "REAL",    "float"),
    ("tmass_h",                  "REAL",    "float"),
    ("tmass_h_err",              "REAL",    "float"),
    ("tmass_k",                  "REAL",    "float"),
    ("tmass_k_err",              "REAL",    "float"),
    ("w1",                       "REAL",    "float"),
    ("w1_err",                   "REAL",    "float"),
    ("w2",                       "REAL",    "float"),
    ("w2_err",                   "REAL",    "float"),
    ("w3",                       "REAL",    "float"),
    ("w3_err",                   "REAL",    "float"),
    ("w4",                       "REAL",    "float"),
    ("w4_err",                   "REAL",    "float"),
    ("apass_v",                  "REAL",    "float"),
    ("apass_v_err",              "REAL",    "float"),
    ("apass_b",                  "REAL",    "float"),
    ("apass_b_err",              "REAL",    "float"),
    ("apass_g",                  "REAL",    "float"),
    ("apass_g_err",              "REAL",    "float"),
    ("apass_r",                  "REAL",    "float"),
    ("apass_r_err",              "REAL",    "float"),
    ("apass_i",                  "REAL",    "float"),
    ("apass_i_err",              "REAL",    "float"),
    ("galex_fuv",                "REAL",    "float"),
    ("galex_fuv_err",            "REAL",    "float"),
    ("galex_nuv",                "REAL",    "float"),
    ("galex_nuv_err",            "REAL",    "float"),
    ("H_K",                      "REAL",    "float"),
    ("w1_w2",                    "REAL",    "float"),
    ("w1_w3",                    "REAL",    "float"),
    ("w1_w4",                    "REAL",    "float"),
    ("w2_w3",                    "REAL",    "float"),
    ("w2_w4",                    "REAL",    "float"),
    ("w3_w4",                    "REAL",    "float"),
    ("sed_alpha",                "REAL",    "float"),
    ("sed_alpha_class",          "TEXT",    "select"),
    ("sed_alpha_n_points",       "REAL",    "float"),
    ("sed_alpha_lambda_min_micron", "REAL", "float"),
    ("sed_alpha_lambda_max_micron", "REAL", "float"),
    ("sed_alpha_bands_json",     "TEXT",    "text"),
    ("sed_alpha_status",         "TEXT",    "select"),
    ("iphas_r_mag",              "REAL",    "float"),
    ("iphas_r_err",              "REAL",    "float"),
    ("iphas_i_mag",              "REAL",    "float"),
    ("iphas_i_err",              "REAL",    "float"),
    ("iphas_ha_mag",             "REAL",    "float"),
    ("iphas_ha_err",             "REAL",    "float"),
    ("iphas_r_i",                "REAL",    "float"),
    ("iphas_r_i_err",            "REAL",    "float"),
    ("iphas_r_ha",               "REAL",    "float"),
    ("iphas_r_ha_err",           "REAL",    "float"),
    ("iphas_sep_arcsec",         "REAL",    "float"),
    ("iphas_source_catalog",     "TEXT",    "text"),
    ("vphas_r_mag",              "REAL",    "float"),
    ("vphas_r_err",              "REAL",    "float"),
    ("vphas_i_mag",              "REAL",    "float"),
    ("vphas_i_err",              "REAL",    "float"),
    ("vphas_ha_mag",             "REAL",    "float"),
    ("vphas_ha_err",             "REAL",    "float"),
    ("vphas_r_i",                "REAL",    "float"),
    ("vphas_r_i_err",            "REAL",    "float"),
    ("vphas_r_ha",               "REAL",    "float"),
    ("vphas_r_ha_err",           "REAL",    "float"),
    ("vphas_sep_arcsec",         "REAL",    "float"),
    ("vphas_source_catalog",     "TEXT",    "text"),
    ("unwise_w1_zscore",         "REAL",    "float"),
    ("unwise_w2_zscore",         "REAL",    "float"),
    ("unwise_w1_var",            "INTEGER", "bool"),
    # -- galactic coordinates --
    ("gal_l",                    "REAL",    "float"),
    ("gal_b",                    "REAL",    "float"),
    # -- extinction & environment --
    ("A_v_3d",                   "REAL",    "float"),
    ("ebv_3d",                   "REAL",    "float"),
    ("dust_sigma",               "REAL",    "float"),
    ("population",               "TEXT",    "text"),
    ("age50",                    "REAL",    "float"),
    ("mass50",                   "REAL",    "float"),
    ("banyan_field_prob",        "REAL",    "float"),
    ("banyan_ya_prob",           "REAL",    "float"),
    ("banyan_best_assoc",        "TEXT",    "text"),
    ("banyan_best_assoc_prob",   "REAL",    "float"),
    ("banyan_probabilities_json", "TEXT",   "text"),
    ("banyan_input_mode",        "TEXT",    "select"),
    ("banyan_status",            "TEXT",    "select"),
    ("banyan_error",             "TEXT",    "text"),
    ("banyan_version",           "TEXT",    "text"),
    ("banyan_adapter_version",   "TEXT",    "text"),
    ("banyan_updated_at",        "TEXT",    "text"),
    # -- SFR environment versus stellar-association membership --
    ("sfr_environment_matches",  "TEXT",    "text"),
    ("sfr_environment_consistent", "INTEGER", "bool"),
    ("banyan_sfr_name",          "TEXT",    "text"),
    ("banyan_sfr_prob",          "REAL",    "float"),
    ("banyan_sfr_best_assoc",    "TEXT",    "text"),
    ("banyan_sfr_best_assoc_prob", "REAL",  "float"),
    ("banyan_sfr_agrees",        "INTEGER", "bool"),
    ("sfr_catalog_member",       "INTEGER", "bool"),
    ("sfr_catalog_match_status", "TEXT",    "select"),
    ("sfr_catalog_name",         "TEXT",    "text"),
    ("sfr_catalog_reference",    "TEXT",    "text"),
    ("sfr_catalog_membership_prob", "REAL", "float"),
    ("sfr_kinematic_name",       "TEXT",    "text"),
    ("sfr_kinematic_method",     "TEXT",    "select"),
    ("sfr_kinematic_consistent", "INTEGER", "bool"),
    ("sfr_kinematic_mahalanobis_sq", "REAL", "float"),
    ("sfr_kinematic_p_value",    "REAL",    "float"),
    ("sfr_kinematic_n_members",  "REAL",    "float"),
    ("sfr_membership_class",     "TEXT",    "select"),
    ("sfr_membership_name",      "TEXT",    "text"),
    ("sfr_membership_evidence",  "TEXT",    "select"),
    ("sfr_membership_status",    "TEXT",    "select"),
    ("sfr_membership_threshold", "REAL",    "float"),
    ("sfr_membership_version",   "TEXT",    "text"),
    # -- crossmatch details --
    ("vsx_class",                "TEXT",    "select"),
    ("vsx_sep_arcsec",           "REAL",    "float"),
    ("vsx_period",               "REAL",    "float"),
    ("nearby_vsx_dipper_contaminant", "INTEGER", "bool"),
    ("nearby_vsx_dipper_class",  "TEXT",    "select"),
    ("nearby_vsx_dipper_sep_arcsec", "REAL", "float"),
    ("nearby_vsx_dipper_period", "REAL",    "float"),
    ("sfr_name",                 "TEXT",    "text"),
    ("sfr_sep_arcmin",           "REAL",    "float"),
    ("cluster_name",             "TEXT",    "text"),
    ("cluster_membership_prob",  "REAL",    "float"),
    ("cluster_age_myr",          "REAL",    "float"),
    ("cluster_dist_pc",          "REAL",    "float"),
    ("cluster_catalog",          "TEXT",    "text"),
    ("open_cluster_match_version", "TEXT",  "text"),
    ("open_cluster_gaia_id",     "TEXT",    "text"),
    ("ucc_match_status",         "TEXT",    "select"),
    ("ucc_catalog_release",      "TEXT",    "text"),
    ("ucc_catalog_doi",          "TEXT",    "text"),
    ("ucc_n_matches",            "INTEGER", "int"),
    ("ucc_listed_member",        "INTEGER", "bool"),
    ("ucc_p50_member",           "INTEGER", "bool"),
    ("ucc_good_cluster",         "INTEGER", "bool"),
    ("ucc_good_member",          "INTEGER", "bool"),
    ("ucc_cluster",              "TEXT",    "text"),
    ("ucc_pmem",                 "REAL",    "float"),
    ("ucc_n_members",            "INTEGER", "int"),
    ("ucc_r50_arcmin",           "REAL",    "float"),
    ("ucc_core_radius_pc",       "REAL",    "float"),
    ("ucc_cluster_ra_deg",       "REAL",    "float"),
    ("ucc_cluster_dec_deg",      "REAL",    "float"),
    ("ucc_cluster_parallax_mas", "REAL",    "float"),
    ("ucc_cluster_pmra_masyr",   "REAL",    "float"),
    ("ucc_cluster_pmdec_masyr",  "REAL",    "float"),
    ("ucc_cluster_rv_kms",       "REAL",    "float"),
    ("ucc_distance_kpc",         "REAL",    "float"),
    ("ucc_distance_std",         "REAL",    "float"),
    ("ucc_av_mag",               "REAL",    "float"),
    ("ucc_av_std",               "REAL",    "float"),
    ("ucc_age_myr",              "REAL",    "float"),
    ("ucc_age_std",              "REAL",    "float"),
    ("ucc_feh_dex",              "REAL",    "float"),
    ("ucc_feh_std",              "REAL",    "float"),
    ("ucc_mass_msun",            "REAL",    "float"),
    ("ucc_mass_std",             "REAL",    "float"),
    ("ucc_c3",                   "TEXT",    "text"),
    ("ucc_pdup",                 "REAL",    "float"),
    ("ucc_uti",                  "REAL",    "float"),
    ("ucc_bad_oc",               "INTEGER", "bool"),
    ("ucc_nearest_cluster",      "TEXT",    "text"),
    ("ucc_nearest_sep_arcmin",   "REAL",    "float"),
    ("ucc_nearest_r50_arcmin",   "REAL",    "float"),
    ("ucc_nearest_sep_r50",      "REAL",    "float"),
    ("ucc_nearest_age_myr",      "REAL",    "float"),
    ("ucc_nearest_distance_kpc", "REAL",    "float"),
    ("ucc_nearest_uti",          "REAL",    "float"),
    ("ucc_nearest_dparallax_mas", "REAL",   "float"),
    ("ucc_nearest_dpmra_masyr",  "REAL",    "float"),
    ("ucc_nearest_dpmdec_masyr", "REAL",    "float"),
    ("hr24_match_status",        "TEXT",    "select"),
    ("hr24_catalog_id",          "TEXT",    "text"),
    ("hr24_n_matches",           "INTEGER", "int"),
    ("hr24_listed_member",       "INTEGER", "bool"),
    ("hr24_p50_member",          "INTEGER", "bool"),
    ("hr24_bound_member",        "INTEGER", "bool"),
    ("hr24_high_quality_member", "INTEGER", "bool"),
    ("hr24_cluster",             "TEXT",    "text"),
    ("hr24_pmem",                "REAL",    "float"),
    ("hr24_cluster_type",        "TEXT",    "select"),
    ("hr24_cst",                 "REAL",    "float"),
    ("hr24_cmd_class_median",    "REAL",    "float"),
    ("hr24_n_members",           "INTEGER", "int"),
    ("hr24_r50_deg",             "REAL",    "float"),
    ("hr24_r50_pc",              "REAL",    "float"),
    ("hr24_cluster_ra_deg",      "REAL",    "float"),
    ("hr24_cluster_dec_deg",     "REAL",    "float"),
    ("hr24_cluster_parallax_mas", "REAL",   "float"),
    ("hr24_cluster_pmra_masyr",  "REAL",    "float"),
    ("hr24_cluster_pmdec_masyr", "REAL",    "float"),
    ("hr24_distance_pc",         "REAL",    "float"),
    ("hr24_log_age_yr",          "REAL",    "float"),
    ("hr24_prob_jacobi",         "REAL",    "float"),
    ("hr24_mass_jacobi_msun",    "REAL",    "float"),
    ("hr24_mass_total_msun",     "REAL",    "float"),
    ("hr24_inside_jacobi_radius", "INTEGER", "bool"),
    ("hr24_inside_tidal_radius", "INTEGER", "bool"),
    # -- nuclear context and scoring --
    ("gaia_parallax_snr",        "REAL",    "float"),
    ("gaia_pm_snr",              "REAL",    "float"),
    ("gaia_stellar_veto_score",  "REAL",    "float"),
    ("gaia_extragalactic_prior_score", "REAL", "float"),
    ("host_match",               "INTEGER", "bool"),
    ("host_source",              "TEXT",    "text"),
    ("host_sep_arcsec",          "REAL",    "float"),
    ("nuclear_offset_arcsec",    "REAL",    "float"),
    ("host_nuclear_score",       "REAL",    "float"),
    ("host_assoc_status",        "TEXT",    "select"),
    ("radio_det",                "INTEGER", "bool"),
    ("radio_source_catalogs",    "TEXT",    "text"),
    ("radio_sep_arcsec",         "REAL",    "float"),
    ("radio_flux_mjy",           "REAL",    "float"),
    ("radio_agn_prior_score",    "REAL",    "float"),
    ("wise_agn_score",           "REAL",    "float"),
    ("neowise_variability_score","REAL",    "float"),
    ("xray_agn_prior_score",     "REAL",    "float"),
    ("uv_tde_score",             "REAL",    "float"),
    ("agn_prior_score",          "REAL",    "float"),
    ("agn_prior_reasons",        "TEXT",    "text"),
    ("tde_candidate_score",      "REAL",    "float"),
    ("tde_candidate_reasons",    "TEXT",    "text"),
    ("clagn_photometric_score",  "REAL",    "float"),
    ("clagn_reasons",            "TEXT",    "text"),
    ("redshift",                 "REAL",    "float"),
    ("redshift_source",          "TEXT",    "select"),
    ("spectral_type",            "TEXT",    "text"),
    ("spectral_type_source",     "TEXT",    "select"),
    ("host_spectral_class",      "TEXT",    "select"),
    ("prior_agn_spectrum_flag",  "INTEGER", "bool"),
    ("broad_line_flag",          "INTEGER", "bool"),
    ("spectrum_sources",         "TEXT",    "text"),
    ("spectrum_links",           "TEXT",    "text"),
    ("swift_uvot_obs",           "INTEGER", "bool"),
    ("swift_uvot_det",           "INTEGER", "bool"),
    ("swift_xrt_det",            "INTEGER", "bool"),
    ("swift_uvot_sep_arcsec",    "REAL",    "float"),
    ("swift_xrt_sep_arcsec",     "REAL",    "float"),
    ("swift_source_catalogs",    "TEXT",    "text"),
    ("swift_status",             "TEXT",    "select"),
    ("known_clagn_match",        "INTEGER", "bool"),
    ("known_clagn_source",       "TEXT",    "text"),
    ("known_clagn_type",         "TEXT",    "text"),
    ("known_clagn_name",         "TEXT",    "text"),
    ("known_clagn_sep_arcsec",   "REAL",    "float"),
    ("known_clagn_training_label","TEXT",   "select"),
    ("nuc_n_points",             "REAL",    "float"),
    ("nuc_time_span_days",       "REAL",    "float"),
    ("nuc_flux_frac_amp_p95_p05","REAL",    "float"),
    ("nuc_flux_slope_snr",       "REAL",    "float"),
    ("n_flare_events",           "REAL",    "float"),
    ("recurrence_count",         "REAL",    "float"),
    ("preflare_rms",             "REAL",    "float"),
    ("tde_single_flare_score",   "REAL",    "float"),
    ("tde_quiet_baseline_score", "REAL",    "float"),
    ("tde_no_recurrence_score",  "REAL",    "float"),
    ("tde_smooth_decline_score", "REAL",    "float"),
    ("fallback_fit_r2",          "REAL",    "float"),
    ("clagn_state_change_mag",   "REAL",    "float"),
    ("clagn_monotonicity_score", "REAL",    "float"),
    ("clagn_plateau_score",      "REAL",    "float"),
    ("lc_feature_status",        "TEXT",    "select"),
    # -- vetting classification --
    ("vetting_likely_known",     "INTEGER", "bool"),
    ("microlens_match",          "INTEGER", "bool"),
    ("microlens_catalog",        "TEXT",    "select"),
    ("asassn_var_type",          "TEXT",    "select"),
    ("gaia_var_class",           "TEXT",    "select"),
    ("simbad_otype",             "TEXT",    "select"),
    ("ztf_var_type",             "TEXT",    "select"),
    # -- vetting details: SIMBAD --
    ("simbad_main_id",           "TEXT",    "text"),
    ("simbad_nbref",             "REAL",    "float"),
    ("simbad_sep_arcsec",        "REAL",    "float"),
    # -- vetting details: Gaia variability --
    ("gaia_var_flag",            "INTEGER", "bool"),
    ("gaia_var_score",           "REAL",    "float"),
    # -- vetting details: Gaia EB --
    ("gaia_eb_period",           "REAL",    "float"),
    ("gaia_eb_morph",            "TEXT",    "text"),
    ("gaia_eb_global_ranking",   "REAL",    "float"),
    ("gaia_eb_solution_id",       "TEXT",    "text"),
    ("gaia_eb_period_error",      "REAL",    "float"),
    ("gaia_eb_reference_time",    "REAL",    "float"),
    ("gaia_eb_frequency",         "REAL",    "float"),
    ("gaia_eb_frequency_error",   "REAL",    "float"),
    ("gaia_eb_model_type",        "TEXT",    "text"),
    ("gaia_eb_reduced_chi2",      "REAL",    "float"),
    ("gaia_eb_geom_model_reference_level", "REAL", "float"),
    ("gaia_eb_geom_model_reference_level_error", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian1_phase", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian1_phase_error", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian1_sigma", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian1_sigma_error", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian1_depth", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian1_depth_error", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian2_phase", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian2_phase_error", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian2_sigma", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian2_sigma_error", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian2_depth", "REAL", "float"),
    ("gaia_eb_geom_model_gaussian2_depth_error", "REAL", "float"),
    ("gaia_eb_geom_model_cosine_half_period_amplitude", "REAL", "float"),
    ("gaia_eb_geom_model_cosine_half_period_amplitude_error", "REAL", "float"),
    ("gaia_eb_geom_model_cosine_half_period_phase", "REAL", "float"),
    ("gaia_eb_geom_model_cosine_half_period_phase_error", "REAL", "float"),
    ("gaia_eb_primary_phase",     "REAL",    "float"),
    ("gaia_eb_primary_phase_error", "REAL", "float"),
    ("gaia_eb_primary_duration",  "REAL",    "float"),
    ("gaia_eb_primary_duration_error", "REAL", "float"),
    ("gaia_eb_primary_depth",     "REAL",    "float"),
    ("gaia_eb_primary_depth_error", "REAL", "float"),
    ("gaia_eb_secondary_phase",   "REAL",    "float"),
    ("gaia_eb_secondary_phase_error", "REAL", "float"),
    ("gaia_eb_secondary_duration", "REAL",   "float"),
    ("gaia_eb_secondary_duration_error", "REAL", "float"),
    ("gaia_eb_secondary_depth",   "REAL",    "float"),
    ("gaia_eb_secondary_depth_error", "REAL", "float"),
    ("gaia_eb_depth_ratio",       "REAL",    "float"),
    ("gaia_eb_eclipse_phase_separation", "REAL", "float"),
    ("gaia_eb_two_eclipses",      "INTEGER", "bool"),
    # -- Gaia NSS and evidence-family summary --
    ("gaia_nss_solution_count",   "REAL",    "float"),
    ("gaia_nss_solution_types",   "TEXT",    "text"),
    ("gaia_nss_solution_type",    "TEXT",    "text"),
    ("gaia_nss_period",           "REAL",    "float"),
    ("gaia_nss_period_error",     "REAL",    "float"),
    ("gaia_nss_has_sb1",          "INTEGER", "bool"),
    ("gaia_nss_has_sb2",          "INTEGER", "bool"),
    ("gaia_nss_has_spectroscopic", "INTEGER", "bool"),
    ("gaia_nss_has_astrometric",  "INTEGER", "bool"),
    ("gaia_nss_has_eclipsing",    "INTEGER", "bool"),
    ("gaia_nss_has_eclipsing_spectro", "INTEGER", "bool"),
    ("gaia_nss_photometric_duplicate_of_eb", "INTEGER", "bool"),
    ("gaia_nss_semi_amplitude_primary", "REAL", "float"),
    ("gaia_nss_semi_amplitude_secondary", "REAL", "float"),
    ("gaia_nss_mass_ratio",       "REAL",    "float"),
    ("gaia_nss_inclination",      "REAL",    "float"),
    ("gaia_rv_variable_flag",     "INTEGER", "bool"),
    ("gaia_rv_large_amplitude_flag", "INTEGER", "bool"),
    ("gaia_astrometric_anomaly_flag", "INTEGER", "bool"),
    ("gaia_blend_contamination_flag", "INTEGER", "bool"),
    ("gaia_binary_reference_period", "REAL", "float"),
    ("gaia_binary_reference_period_source", "TEXT", "select"),
    ("gaia_binary_period_n_independent", "REAL", "float"),
    ("gaia_binary_period_agreement", "INTEGER", "bool"),
    ("gaia_binary_period_agreement_sources", "TEXT", "text"),
    ("gaia_binary_period_conflict", "INTEGER", "bool"),
    ("gaia_binary_period_conflict_sources", "TEXT", "text"),
    ("gaia_binary_evidence_families", "TEXT", "text"),
    ("gaia_binary_evidence_version", "TEXT", "select"),
    ("gaia_binary_evidence_score_kind", "TEXT", "select"),
    ("gaia_binary_n_evidence_families", "REAL", "float"),
    ("gaia_binary_evidence_level", "TEXT", "select"),
    ("gaia_binary_evidence_score", "REAL", "float"),
    ("gaia_eb_evidence_level",    "TEXT",    "select"),
    ("gaia_eb_evidence_score",    "REAL",    "float"),
    ("gaia_binary_evidence_summary", "TEXT", "text"),
    # -- vetting details: Gaia epoch --
    ("gaia_epoch_available",     "INTEGER", "bool"),
    ("gaia_epoch_n_obs",         "REAL",    "float"),
    ("gaia_epoch_g_range",       "REAL",    "float"),
    # -- vetting details: ASAS-SN --
    ("asassn_var_name",          "TEXT",    "text"),
    ("asassn_var_period",        "REAL",    "float"),
    # -- vetting details: microlensing catalogs --
    ("microlens_name",           "TEXT",    "text"),
    ("microlens_alt_name",       "TEXT",    "text"),
    ("microlens_te_days",        "REAL",    "float"),
    ("microlens_sep_arcsec",     "REAL",    "float"),
    # -- vetting details: ZTF --
    ("ztf_var_period",           "REAL",    "float"),
    ("ztf_var_amp",              "REAL",    "float"),
    # -- vetting details: TNS --
    ("tns_name",                 "TEXT",    "text"),
    ("tns_type",                 "TEXT",    "select"),
    ("tns_redshift",             "REAL",    "float"),
    ("tns_disc_date",            "TEXT",    "text"),
    # -- vetting details: ALeRCE --
    ("alerce_oid",               "TEXT",    "text"),
    ("alerce_ndet",              "REAL",    "float"),
    ("alerce_lc_class",          "TEXT",    "select"),
    ("alerce_lc_prob",           "REAL",    "float"),
    ("alerce_stamp_class",       "TEXT",    "text"),
    ("alerce_stamp_prob",        "REAL",    "float"),
    # -- vetting details: X-ray --
    ("erosita_det",              "INTEGER", "bool"),
    ("erosita_flux",             "REAL",    "float"),
    ("erosita_sep_arcsec",       "REAL",    "float"),
    ("chandra_det",              "INTEGER", "bool"),
    ("chandra_source_id",        "TEXT",    "text"),
    ("chandra_flux_05_7",        "REAL",    "float"),
    ("chandra_flux_broad",       "REAL",    "float"),
    ("chandra_significance",     "REAL",    "float"),
    ("chandra_likelihood",       "REAL",    "float"),
    ("chandra_likelihood_class", "TEXT",    "select"),
    ("chandra_pos_err_maj_arcsec","REAL",   "float"),
    ("chandra_pos_err_min_arcsec","REAL",   "float"),
    ("chandra_pos_err_pa_deg",   "REAL",    "float"),
    ("chandra_extended_flag",    "INTEGER", "bool"),
    ("chandra_variable_flag",    "INTEGER", "bool"),
    ("chandra_sep_arcsec",       "REAL",    "float"),
    ("xray_det",                 "INTEGER", "bool"),
    ("xray_flux",                "REAL",    "float"),
    ("xray_sep_arcsec",          "REAL",    "float"),
    ("xray_source_catalogs",     "TEXT",    "text"),
    # -- vetting details: proper motion --
    ("pm_cluster_offset_sigma",  "REAL",    "float"),
    # -- vetting details: ATLAS --
    ("atlas_has_phot",           "INTEGER", "bool"),
    ("atlas_n_det_cyan",         "REAL",    "float"),
    ("atlas_n_det_orange",       "REAL",    "float"),
    ("atlas_cyan_range",         "REAL",    "float"),
    ("atlas_orange_range",       "REAL",    "float"),
    ("atlas_preprocess_version", "TEXT",    "text"),
    ("atlas_n_raw",              "INTEGER", "float"),
    ("atlas_n_good",             "INTEGER", "float"),
    ("atlas_n_rejected",         "INTEGER", "float"),
    # -- vetting details: NEOWISE --
    ("neowise_n_epochs",         "REAL",    "float"),
    ("neowise_w1_range",         "REAL",    "float"),
    ("neowise_w2_range",         "REAL",    "float"),
    # -- external light curves: ZTF --
    ("ztf_lc_n_det",             "INTEGER", "float"),
    ("ztf_lc_g_range",           "REAL",    "float"),
    ("ztf_lc_r_range",           "REAL",    "float"),
    # -- external light curves: ZTF forced difference photometry --
    ("ztf_forced_lc_n_epochs",   "INTEGER", "float"),
    ("ztf_forced_lc_n_good",     "INTEGER", "float"),
    ("ztf_forced_lc_n_zg",       "INTEGER", "float"),
    ("ztf_forced_lc_n_zr",       "INTEGER", "float"),
    ("ztf_forced_lc_n_zi",       "INTEGER", "float"),
    # -- external light curves: Gaia epoch --
    ("gaia_epoch_lc_n_g",        "INTEGER", "float"),
    ("gaia_epoch_lc_g_range",    "REAL",    "float"),
    # -- external light curves: TESS --
    ("tess_n_sectors",           "INTEGER", "float"),
    ("tess_total_points",        "INTEGER", "float"),
    ("tess_flux_range",          "REAL",    "float"),
    # -- external light curves: Kepler --
    ("kepler_n_quarters",        "INTEGER", "float"),
    ("kepler_total_points",      "INTEGER", "float"),
    ("kepler_flux_range",        "REAL",    "float"),
    # -- external light curves: AAVSO --
    ("aavso_lc_n_points",        "INTEGER", "float"),
    # -- external light curves: OGLE --
    ("ogle_lc_n_points",         "INTEGER", "float"),
    ("ogle_lc_i_range",          "REAL",    "float"),
    ("ogle_lc_v_range",          "REAL",    "float"),
    # -- external light curves: SDSS Stripe 82 --
    ("stripe82_lc_n_points",     "INTEGER", "float"),
    ("stripe82_lc_u_range",      "REAL",    "float"),
    ("stripe82_lc_g_range",      "REAL",    "float"),
    ("stripe82_lc_r_range",      "REAL",    "float"),
    ("stripe82_lc_i_range",      "REAL",    "float"),
    ("stripe82_lc_z_range",      "REAL",    "float"),
    # -- external light curves: AllWISE Multiepoch --
    ("allwise_mep_n_epochs",     "INTEGER", "float"),
    ("allwise_mep_w1_range",     "REAL",    "float"),
    ("allwise_mep_w2_range",     "REAL",    "float"),
    ("allwise_mep_w3_range",     "REAL",    "float"),
    ("allwise_mep_w4_range",     "REAL",    "float"),
    # -- external light curves: VVVX/VIRAC2 --
    ("vvvx_virac_n_epochs",      "INTEGER", "float"),
    ("vvvx_virac_z_range",       "REAL",    "float"),
    ("vvvx_virac_y_range",       "REAL",    "float"),
    ("vvvx_virac_j_range",       "REAL",    "float"),
    ("vvvx_virac_h_range",       "REAL",    "float"),
    ("vvvx_virac_ks_range",      "REAL",    "float"),
    # -- external light curves: Pan-STARRS --
    ("ps1_lc_n_points",          "INTEGER", "float"),
    # -- external light curves: legacy transit surveys --
    ("superwasp_lc_n_points",    "INTEGER", "float"),
    ("superwasp_lc_time_span_days", "REAL", "float"),
    ("superwasp_lc_state",       "TEXT",    "select"),
    ("kelt_lc_n_points",         "INTEGER", "float"),
    ("kelt_lc_time_span_days",   "REAL",    "float"),
    ("kelt_lc_state",            "TEXT",    "select"),
    ("nsvs_lc_n_points",         "INTEGER", "float"),
    ("nsvs_lc_time_span_days",   "REAL",    "float"),
    ("nsvs_lc_state",            "TEXT",    "select"),
    ("asas3_lc_n_points",        "INTEGER", "float"),
    ("asas3_lc_time_span_days",  "REAL",    "float"),
    ("asas3_lc_state",           "TEXT",    "select"),
    # -- external light curves: CRTS --
    ("crts_lc_n_points",         "INTEGER", "float"),
    ("crts_lc_time_span_days",   "REAL",    "float"),
    ("crts_lc_state",            "TEXT",    "select"),
    # -- external light curves: DASCH --
    ("dasch_lc_n_points",        "INTEGER", "float"),
    ("dasch_lc_time_span_days",  "REAL",    "float"),
    ("dasch_lc_state",           "TEXT",    "select"),
    # -- multi-survey event-relative features --
    *MS_FEATURE_COLUMN_SPECS,
    # -- compact multi-survey microlensing fit summary --
    *MICROLENSING_JOINT_COLUMN_SPECS,
    # -- vetting details: other --
    ("iphas_ha_excess",          "REAL",    "float"),
    ("vphas_ha_excess",          "REAL",    "float"),
    # -- light curve basics --
    ("n_points",                 "REAL",    "float"),
    ("jd_first",                 "REAL",    "float"),
    ("jd_last",                  "REAL",    "float"),
    ("time_span_days",           "REAL",    "float"),
    ("n_unique_nights",          "REAL",    "float"),
    ("n_cameras",                "REAL",    "float"),
    ("baseline_mag",             "REAL",    "float"),
    ("baseline_source",          "TEXT",    "text"),
    ("pre_periodicity_label",    "TEXT",    "select"),
    ("pre_periodic_flag",        "INTEGER", "bool"),
    ("pre_periodicity_selected_period", "REAL", "float"),
    ("pre_periodicity_method",   "TEXT",    "select"),
    ("cadence_median_days",      "REAL",    "float"),
    ("trigger_mode",             "TEXT",    "text"),
    # -- YSO / classification --
    ("trigger_type",             "TEXT",    "text"),
    ("yso_class",                "TEXT",    "select"),
    ("final_class",              "TEXT",    "text"),
    ("P_eb",                     "REAL",    "float"),
    ("P_cv",                     "REAL",    "float"),
    ("P_starspot",               "REAL",    "float"),
    ("P_disk",                   "REAL",    "float"),
    ("a_circ_au",                "REAL",    "float"),
    ("transit_prob",             "REAL",    "float"),
    ("hill_radius_rsun",         "REAL",    "float"),
    # -- individual fail flags --
    ("failed_posterior_strength", "INTEGER", "bool"),
    ("failed_run_robustness",    "INTEGER", "bool"),
    ("failed_morphology",        "INTEGER", "bool"),
    ("failed_score",             "INTEGER", "bool"),
    ("failed_periodicity",       "INTEGER", "bool"),
    ("failed_gaia_ruwe",         "INTEGER", "bool"),
    ("failed_periodic_catalog",  "INTEGER", "bool"),
    ("failed_signal_amplitude",  "INTEGER", "bool"),
    ("bad_cameras_filtered",     "INTEGER", "bool"),
    # -- light curve statistics (from stats.py / enrichment) --
    ("stats_photometry_band_mode",                  "TEXT", "text"),
    ("stats_photometry_band_alignment",             "TEXT", "text"),
    ("stats_photometry_g_points",                   "INTEGER", "float"),
    ("stats_photometry_v_points",                   "INTEGER", "float"),
    ("stats_photometry_v_minus_g_offset_mag",       "REAL", "float"),
    ("stats_file_points_total",                    "REAL", "float"),
    ("stats_file_points_kept_after_filter",         "REAL", "float"),
    ("stats_jd_start",                             "REAL", "float"),
    ("stats_jd_end",                               "REAL", "float"),
    ("stats_time_span_days",                       "REAL", "float"),
    ("stats_n_unique_nights",                      "REAL", "float"),
    ("stats_duty_cycle_fraction",                  "REAL", "float"),
    ("stats_cadence_mean_dt_days",                 "REAL", "float"),
    ("stats_cadence_median_dt_days",               "REAL", "float"),
    ("stats_cadence_p05_dt_days",                  "REAL", "float"),
    ("stats_cadence_p95_dt_days",                  "REAL", "float"),
    ("stats_photometry_mean_mag",                  "REAL", "float"),
    ("stats_photometry_median_mag",                "REAL", "float"),
    ("stats_photometry_weighted_mean_mag",         "REAL", "float"),
    ("stats_photometry_weighted_mean_sem",         "REAL", "float"),
    ("stats_photometry_weighted_std_mag",          "REAL", "float"),
    ("stats_photometry_std_mag",                   "REAL", "float"),
    ("stats_photometry_robust_sigma_mag",          "REAL", "float"),
    ("stats_photometry_IQR_mag",                   "REAL", "float"),
    ("stats_photometry_p05_mag",                   "REAL", "float"),
    ("stats_photometry_p16_mag",                   "REAL", "float"),
    ("stats_photometry_p84_mag",                   "REAL", "float"),
    ("stats_photometry_p95_mag",                   "REAL", "float"),
    ("stats_clipped_mean_mag_3sigma_about_median", "REAL", "float"),
    ("stats_clipped_mean_mag_3sigma_about_median_g", "REAL", "float"),
    ("stats_clipped_mean_mag_3sigma_about_median_vband", "REAL", "float"),
    ("stats_clipped_std_mag_3sigma_about_median",  "REAL", "float"),
    ("stats_n_outliers_removed_robust_3sigma",     "REAL", "float"),
    ("stats_error_and_snr_stats_error_mean",       "REAL", "float"),
    ("stats_error_and_snr_stats_error_median",     "REAL", "float"),
    ("stats_error_and_snr_stats_error_p05",        "REAL", "float"),
    ("stats_error_and_snr_stats_error_p95",        "REAL", "float"),
    ("stats_error_and_snr_stats_snr_median",       "REAL", "float"),
    ("stats_error_and_snr_stats_snr_p05",          "REAL", "float"),
    ("stats_error_and_snr_stats_snr_p95",          "REAL", "float"),
    ("stats_variability_reduced_chi2_vs_constant", "REAL", "float"),
    ("stats_variability_von_neumann_ratio",        "REAL", "float"),
    ("stats_variability_roms",                     "REAL", "float"),
    ("stats_variability_sokolovsky_v",             "REAL", "float"),
    ("stats_variability_sokolovsky_v_g",           "REAL", "float"),
    ("stats_variability_sokolovsky_v_vband",       "REAL", "float"),
    ("stats_variability_lag1_autocorr",             "REAL", "float"),
    ("stats_variability_stetson_I",                "REAL", "float"),
    ("stats_variability_stetson_J",                "REAL", "float"),
    ("stats_variability_stetson_K",                "REAL", "float"),
    ("stats_variability_stetson_L",                "REAL", "float"),
    ("stats_variability_stetson_J_time",           "REAL", "float"),
    ("stats_variability_stetson_L_time",           "REAL", "float"),
    ("stats_variability_flux_asymmetry_m",         "REAL", "float"),
    ("stats_variability_quasi_periodicity_q",      "REAL", "float"),
    ("stats_variability_quasi_periodicity_method", "TEXT", "text"),
    ("stats_variability_quasi_periodicity_n_points", "REAL", "float"),
    ("stats_variability_quasi_periodicity_n_bins", "REAL", "float"),
    ("stats_variability_quasi_periodicity_populated_bins", "REAL", "float"),
    ("stats_variability_quasi_periodicity_bin_coverage", "REAL", "float"),
    ("stats_variability_quasi_periodicity_smooth_window_bins", "REAL", "float"),
    ("stats_variability_quasi_periodicity_template_amplitude", "REAL", "float"),
    ("stats_variability_quasi_periodicity_raw_scatter", "REAL", "float"),
    ("stats_variability_quasi_periodicity_resid_scatter", "REAL", "float"),
    ("stats_variability_quasi_periodicity_scatter_ratio", "REAL", "float"),
    ("stats_variability_quasi_periodicity_status", "TEXT", "text"),
    ("stats_variability_quasi_periodicity_evaluation", "TEXT", "text"),
    ("stats_variability_quasi_periodicity_n_folds", "REAL", "float"),
    ("stats_variability_periodic_feature_period_days", "REAL", "float"),
    ("stats_variability_periodic_feature_period_source", "TEXT", "text"),
    ("stats_variability_string_length_resid_total", "REAL", "float"),
    ("stats_variability_string_length_resid_mean_step", "REAL", "float"),
    ("stats_variability_string_length_resid_n_steps", "REAL", "float"),
    ("stats_variability_lomb_scargle_best_period_days", "REAL", "float"),
    ("stats_variability_lomb_scargle_peak_power",  "REAL", "float"),
    ("stats_variability_lomb_scargle_fap",         "REAL", "float"),
    ("stats_trend_slope_mag_per_day",              "REAL", "float"),
    ("stats_trend_slope_mag_per_year",             "REAL", "float"),
    ("stats_trend_r2",                             "REAL", "float"),
    # -- ALeRCE-style features --
    ("stats_amplitude",                            "REAL", "float"),
    ("stats_beyond_1_std",                         "REAL", "float"),
    ("stats_con",                                  "REAL", "float"),
    ("stats_delta_mag_fid",                        "REAL", "float"),
    ("stats_intrinsic_sigma_mag",                  "REAL", "float"),
    ("stats_first_mag",                            "REAL", "float"),
    ("stats_gskew",                                "REAL", "float"),
    ("stats_max_slope",                            "REAL", "float"),
    ("stats_meanvariance",                         "REAL", "float"),
    ("stats_median_abs_dev",                       "REAL", "float"),
    ("stats_median_brp",                           "REAL", "float"),
    ("stats_percent_amplitude",                    "REAL", "float"),
    ("stats_ahl_ratio",                            "REAL", "float"),
    ("stats_q31",                                  "REAL", "float"),
    ("stats_skew",                                 "REAL", "float"),
    ("stats_small_kurtosis",                       "REAL", "float"),
    ("stats_constancy_p_value",                    "REAL", "float"),
    ("stats_anderson_darling",                     "REAL", "float"),
    ("stats_pair_slope_trend",                     "REAL", "float"),
    ("stats_rcs",                                  "REAL", "float"),
    ("stats_autocor_length",                       "REAL", "float"),
    ("stats_sf_ml_amplitude",                      "REAL", "float"),
    ("stats_sf_ml_gamma",                          "REAL", "float"),
    # -- period-dependent features --
    ("stats_harmonics_order",                      "INTEGER", "float"),
    ("stats_harmonics_period",                     "REAL", "float"),
    ("stats_harmonics_a0",                         "REAL", "float"),
    ("stats_harmonics_model_amplitude",            "REAL", "float"),
    ("stats_harmonics_reduced_chi2",               "REAL", "float"),
    ("stats_harmonics_mag_1",                      "REAL", "float"),
    ("stats_harmonics_mag_2",                      "REAL", "float"),
    ("stats_harmonics_mag_3",                      "REAL", "float"),
    ("stats_harmonics_mag_4",                      "REAL", "float"),
    ("stats_harmonics_mag_5",                      "REAL", "float"),
    ("stats_harmonics_mag_6",                      "REAL", "float"),
    ("stats_harmonics_mag_7",                      "REAL", "float"),
    ("stats_harmonics_a1",                         "REAL", "float"),
    ("stats_harmonics_a2",                         "REAL", "float"),
    ("stats_harmonics_a3",                         "REAL", "float"),
    ("stats_harmonics_a4",                         "REAL", "float"),
    ("stats_harmonics_a5",                         "REAL", "float"),
    ("stats_harmonics_a6",                         "REAL", "float"),
    ("stats_harmonics_a7",                         "REAL", "float"),
    ("stats_harmonics_b1",                         "REAL", "float"),
    ("stats_harmonics_b2",                         "REAL", "float"),
    ("stats_harmonics_b3",                         "REAL", "float"),
    ("stats_harmonics_b4",                         "REAL", "float"),
    ("stats_harmonics_b5",                         "REAL", "float"),
    ("stats_harmonics_b6",                         "REAL", "float"),
    ("stats_harmonics_b7",                         "REAL", "float"),
    ("stats_harmonics_r21",                        "REAL", "float"),
    ("stats_harmonics_r31",                        "REAL", "float"),
    ("stats_harmonics_r41",                        "REAL", "float"),
    ("stats_harmonics_r51",                        "REAL", "float"),
    ("stats_harmonics_r61",                        "REAL", "float"),
    ("stats_harmonics_r71",                        "REAL", "float"),
    ("stats_harmonics_phase_2",                    "REAL", "float"),
    ("stats_harmonics_phase_3",                    "REAL", "float"),
    ("stats_harmonics_phase_4",                    "REAL", "float"),
    ("stats_harmonics_phase_5",                    "REAL", "float"),
    ("stats_harmonics_phase_6",                    "REAL", "float"),
    ("stats_harmonics_phase_7",                    "REAL", "float"),
    ("stats_harmonics_mse",                        "REAL", "float"),
    ("stats_psi_cs",                               "REAL", "float"),
    ("stats_psi_eta",                              "REAL", "float"),
    ("stats_lafler_kinman_t_time",                 "REAL", "float"),
    ("stats_lafler_kinman_t_phase",                "REAL", "float"),
    ("stats_lafler_kinman_delta",                  "REAL", "float"),
    ("stats_window_alias_period_1",                "REAL", "float"),
    ("stats_window_alias_power_1",                 "REAL", "float"),
    ("stats_window_alias_period_2",                "REAL", "float"),
    ("stats_window_alias_power_2",                 "REAL", "float"),
    ("stats_window_alias_period_3",                "REAL", "float"),
    ("stats_window_alias_power_3",                 "REAL", "float"),
    ("stats_window_alias_period_4",                "REAL", "float"),
    ("stats_window_alias_power_4",                 "REAL", "float"),
    ("stats_window_alias_period_5",                "REAL", "float"),
    ("stats_window_alias_power_5",                 "REAL", "float"),
    ("stats_eb_rminima",                           "REAL", "float"),
    ("stats_eb_primary_min_depth",                 "REAL", "float"),
    ("stats_eb_secondary_min_depth",               "REAL", "float"),
    # -- stochastic model features --
    ("stats_gp_drw_sigma",                         "REAL", "float"),
    ("stats_gp_drw_tau",                           "REAL", "float"),
    ("stats_gp_drw_model",                         "TEXT", "text"),
    ("stats_iar_phi",                              "REAL", "float"),
    ("stats_mhps_high",                            "REAL", "float"),
    ("stats_mhps_low",                             "REAL", "float"),
    ("stats_mhps_non_zero",                        "REAL", "float"),
    ("stats_mhps_pn_flag",                         "INTEGER", "bool"),
    ("stats_mhps_ratio",                           "REAL", "float"),
    ("stats_camera_loo_corr_min",                    "REAL", "float"),
    ("stats_camera_loo_corr_median",                 "REAL", "float"),
    ("stats_camera_loo_rms_max",                     "REAL", "float"),
    ("stats_compute_status",                        "TEXT", "text"),
    ("stats_compute_error",                         "TEXT", "text"),
    # -- derived statistics and color-magnitude quantities --
    *((col, "REAL", "float") for col in DERIVED_FEATURE_COLUMNS),
    # -- LTV: long-term variability core metrics --
    ("ltv_slope",                    "REAL",    "float"),  # mag/year linear slope
    ("ltv_slope_quad",               "REAL",    "float"),  # quadratic term (mag/yr^2)
    ("ltv_max_diff",                 "REAL",    "float"),  # max seasonal difference (mag)
    ("ltv_dispersion",               "REAL",    "float"),  # peak-to-peak dispersion (mag)
    ("ltv_median",                   "REAL",    "float"),  # median magnitude
    ("ltv_median_err",               "REAL",    "float"),  # robust LC scatter proxy from core output
    ("ltv_n_seasons",                "INTEGER", "float"),  # number of non-empty seasons
    ("ltv_ls_period",                "REAL",    "float"),  # best LS period (days)
    ("ltv_ls_power",                 "REAL",    "float"),  # LS power at best period
    ("ltv_ls_fap",                   "REAL",    "float"),  # LS false alarm probability
    ("ltv_coeff1",                   "REAL",    "float"),
    ("ltv_coeff2",                   "REAL",    "float"),
    ("ltv_vg_has_v",                 "INTEGER", "bool"),
    ("ltv_vg_overlap_days",          "REAL",    "float"),
    ("ltv_vg_overlap_fraction",      "REAL",    "float"),
    ("ltv_season_points_min",        "INTEGER", "float"),
    ("ltv_season_points_median",     "REAL",    "float"),
    ("ltv_season_points_max",        "INTEGER", "float"),
    ("ltv_season_span_days_mean",    "REAL",    "float"),
    ("ltv_season_span_days_median",  "REAL",    "float"),
    ("ltv_season_span_days_max",     "REAL",    "float"),
    ("ltv_season_step_max_mag",      "REAL",    "float"),
    ("ltv_season_step_mean_abs_mag", "REAL",    "float"),
    ("ltv_season_step_max_fraction", "REAL",    "float"),
    ("ltv_season_monotonicity_fraction", "REAL", "float"),
    ("ltv_season_spearman_rho",      "REAL",    "float"),
    ("ltv_season_kendall_tau",       "REAL",    "float"),
    ("ltv_leave1out_slope_std",      "REAL",    "float"),
    ("ltv_leave1out_slope_range",    "REAL",    "float"),
    ("ltv_trend_slope_mag_per_year", "REAL",    "float"),
    ("ltv_trend_quad_mag_per_year2", "REAL",    "float"),
    ("ltv_trend_slope_err_mag_per_year", "REAL", "float"),
    ("ltv_trend_slope_snr",          "REAL",    "float"),
    ("ltv_trend_r2",                 "REAL",    "float"),
    ("ltv_trend_delta_bic_linear",   "REAL",    "float"),
    ("ltv_trend_delta_bic_quadratic", "REAL",   "float"),
    ("ltv_smooth_p95_p5",            "REAL",    "float"),
    ("ltv_smooth_var",               "REAL",    "float"),
    ("ltv_resid_var",                "REAL",    "float"),
    ("ltv_long_short_var_ratio",     "REAL",    "float"),
    ("ltv_smooth_n_points",          "INTEGER", "float"),
    ("ltv_smooth_100d_p95_p5",       "REAL",    "float"),
    ("ltv_smooth_100d_smooth_var",   "REAL",    "float"),
    ("ltv_smooth_100d_resid_var",    "REAL",    "float"),
    ("ltv_smooth_100d_long_short_var_ratio", "REAL", "float"),
    ("ltv_smooth_100d_n_points",     "INTEGER", "float"),
    ("ltv_smooth_300d_p95_p5",       "REAL",    "float"),
    ("ltv_smooth_300d_smooth_var",   "REAL",    "float"),
    ("ltv_smooth_300d_resid_var",    "REAL",    "float"),
    ("ltv_smooth_300d_long_short_var_ratio", "REAL", "float"),
    ("ltv_smooth_300d_n_points",     "INTEGER", "float"),
    ("ltv_smooth_1000d_p95_p5",      "REAL",    "float"),
    ("ltv_smooth_1000d_smooth_var",  "REAL",    "float"),
    ("ltv_smooth_1000d_resid_var",   "REAL",    "float"),
    ("ltv_smooth_1000d_long_short_var_ratio", "REAL", "float"),
    ("ltv_smooth_1000d_n_points",    "INTEGER", "float"),
    ("ltv_binned_sf_n_bins",         "INTEGER", "float"),
    ("ltv_binned_sf_30d_mag2",       "REAL",    "float"),
    ("ltv_binned_sf_100d_mag2",      "REAL",    "float"),
    ("ltv_binned_sf_300d_mag2",      "REAL",    "float"),
    ("ltv_binned_sf_1000d_mag2",     "REAL",    "float"),
    ("ltv_binned_sf_3000d_mag2",     "REAL",    "float"),
    ("ltv_binned_sf_300d_30d_ratio", "REAL",    "float"),
    ("ltv_binned_sf_1000d_30d_ratio", "REAL",   "float"),
    ("ltv_binned_sf_3000d_30d_ratio", "REAL",   "float"),
    ("ltv_binned_sf_slope",          "REAL",    "float"),
    ("ltv_theil_sen_slope_mag_per_year", "REAL", "float"),
    ("ltv_theil_sen_intercept_mag",  "REAL",    "float"),
    ("ltv_theil_sen_low_slope_mag_per_year", "REAL", "float"),
    ("ltv_theil_sen_high_slope_mag_per_year", "REAL", "float"),
    ("ltv_bb_n_blocks",              "INTEGER", "float"),
    ("ltv_bb_n_change_points",       "INTEGER", "float"),
    ("ltv_bb_range_mag",             "REAL",    "float"),
    ("ltv_bb_largest_jump_mag",      "REAL",    "float"),
    ("ltv_bb_max_block_offset_mag",  "REAL",    "float"),
    ("ltv_lowess_p95_p5",            "REAL",    "float"),
    ("ltv_lowess_resid_std",         "REAL",    "float"),
    ("ltv_lowess_max_abs_resid",     "REAL",    "float"),
    ("ltv_variogram_short_mag2",     "REAL",    "float"),
    ("ltv_variogram_mid_mag2",       "REAL",    "float"),
    ("ltv_variogram_long_mag2",      "REAL",    "float"),
    ("ltv_variogram_long_short_ratio", "REAL",  "float"),
    ("ltv_variogram_slope",          "REAL",    "float"),
    # -- LTV: filter flags --
    ("ltv_failed_slope",             "INTEGER", "bool"),
    ("ltv_failed_max_diff",          "INTEGER", "bool"),
    ("ltv_failed_dec",               "INTEGER", "bool"),
    ("ltv_failed_refcat_offset",     "INTEGER", "bool"),
    ("ltv_failed_photometric_scatter", "INTEGER", "bool"),
    ("ltv_failed_high_pm",           "INTEGER", "bool"),
    ("ltv_failed_neighbor_high_pm",  "INTEGER", "bool"),
    ("ltv_failed_crowding",          "INTEGER", "bool"),
    ("ltv_class",                    "TEXT",    "text"),
    ("ltv_class_reason",             "TEXT",    "text"),
    ("ltv_interest_score",           "REAL",    "float"),
    ("ltv_dust_candidate",           "INTEGER", "bool"),   # dust-driven variability flag
    ("ltv_dust_excess",              "INTEGER", "bool"),   # mid-IR excess flag
    # -- LTV: crossmatch --
    ("ltv_vsx_match",                "INTEGER", "bool"),
    ("ltv_vsx_name",                 "TEXT",    "text"),
    ("ltv_milliquas_match",          "INTEGER", "bool"),   # AGN/QSO flag
    ("ltv_gaia_alert_match",         "INTEGER", "bool"),   # Gaia photometric alert
    # -- LTV: NEOWISE time-series --
    ("ltv_neowise_w1_slope",         "REAL",    "float"),  # NEOWISE W1 trend slope (mag/yr)
    ("ltv_neowise_w1_w2_slope",      "REAL",    "float"),  # NEOWISE W1-W2 color trend slope
    ("ltv_neowise_n_epochs",         "INTEGER", "float"),  # number of NEOWISE epochs
    # -- LTV: stochastic post-filter features --
    ("ltv_stoch_sf_ml_amplitude",    "REAL",    "float"),
    ("ltv_stoch_sf_ml_gamma",        "REAL",    "float"),
    ("ltv_stoch_iar_phi",            "REAL",    "float"),
    ("ltv_stoch_mhps_high",          "REAL",    "float"),
    ("ltv_stoch_mhps_low",           "REAL",    "float"),
    ("ltv_stoch_mhps_non_zero",      "REAL",    "float"),
    ("ltv_stoch_mhps_pn_flag",       "INTEGER", "bool"),
    ("ltv_stoch_mhps_ratio",         "REAL",    "float"),
    ("ltv_stoch_gp_drw_sigma",       "REAL",    "float"),
    ("ltv_stoch_gp_drw_tau",         "REAL",    "float"),
    # -- LTV: external multi-survey long-term summaries --
    *LTV_MS_FEATURE_COLUMN_SPECS,
]

# Derived helpers
_COL_NAMES = [c[0] for c in _CANDIDATE_COLUMNS]
_BOOL_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "bool"}
_FLOAT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "float"}
_TEXT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "text"}
_SELECT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "select"}
_COL_TYPE_MAP = {c[0]: c[2] for c in _CANDIDATE_COLUMNS}
_FALSE_INCLUDES_UNSET_BOOL_COLS = {
    "high_pm_flag",
    "microlens_match",
    "nearby_vsx_dipper_contaminant",
    "vetting_likely_known",
}
_REVIEW_FLOAT_COLS = {col for col, kind in REVIEW_FILTER_COLUMN_TYPES.items() if kind == "num"}
_REVIEW_TEXT_COLS = {col for col, kind in REVIEW_FILTER_COLUMN_TYPES.items() if kind == "text"}
_REVIEW_SELECT_COLS = {col for col, kind in REVIEW_FILTER_COLUMN_TYPES.items() if kind == "select"}


def _review_filter_expr(column: str) -> str:
    if column == "workflow_status":
        return "COALESCE(r.workflow_status, 'unreviewed')"
    return f"r.{column}"


def get_distinct_values(
    conn: sqlite3.Connection,
    column: str,
    *,
    source_path: str | None = None,
    source_paths: list[str] | None = None,
    source_path_fallback_like_any: list[str] | None = None,
    source_path_like: str | None = None,
    source_path_like_any: list[str] | None = None,
) -> list[str]:
    """Return sorted distinct non-empty values for a select-filter column."""
    is_review_col = column in _REVIEW_SELECT_COLS or column in _REVIEW_TEXT_COLS
    if column not in _SELECT_COLS and column not in _TEXT_COLS and not is_review_col:
        return []

    expr = _review_filter_expr(column) if is_review_col else f"c.{column}"
    where = [f"{expr} IS NOT NULL", f"{expr} != ''"]
    params: list[str] = []
    source_scope_terms: list[str] = []
    source_scope_params: list[str] = []
    if source_path:
        source_scope_terms.append("c.source_path = ?")
        source_scope_params.append(str(source_path))
    if source_paths:
        source_paths = [str(p) for p in source_paths if str(p)]
        if source_paths:
            placeholders = ",".join(["?"] * len(source_paths))
            source_scope_terms.append(f"c.source_path IN ({placeholders})")
            source_scope_params.extend(source_paths)
    if source_path_fallback_like_any:
        source_path_fallback_like_any = [str(v) for v in source_path_fallback_like_any if str(v)]
        if source_path_fallback_like_any:
            source_scope_terms.extend(["c.source_path LIKE ?"] * len(source_path_fallback_like_any))
            source_scope_params.extend([f"%{value}%" for value in source_path_fallback_like_any])
    if source_scope_terms:
        where.append("(" + " OR ".join(source_scope_terms) + ")")
        params.extend(source_scope_params)
    if source_path_like:
        where.append("c.source_path LIKE ?")
        params.append(f"%{str(source_path_like)}%")
    if source_path_like_any:
        source_path_like_any = [str(v) for v in source_path_like_any if str(v)]
        if source_path_like_any:
            where.append("(" + " OR ".join(["c.source_path LIKE ?"] * len(source_path_like_any)) + ")")
            params.extend([f"%{value}%" for value in source_path_like_any])

    rows = conn.execute(
        f"SELECT DISTINCT {expr} AS value FROM candidates c "
        f"LEFT JOIN reviews r ON r.candidate_id = c.candidate_id "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY value",
        params,
    ).fetchall()
    cleaned = set()
    for r in rows:
        val = r[0]
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8", errors="replace")
            except Exception:
                val = str(val)
        val_str = str(val).strip()
        if val_str and val_str not in ("None", "NaN", "nan"):
            cleaned.add(val_str)
    return sorted(list(cleaned))


def get_distinct_values_bulk(
    conn: sqlite3.Connection,
    columns: list[str] | tuple[str, ...],
    *,
    source_path: str | None = None,
    source_paths: list[str] | None = None,
    source_path_fallback_like_any: list[str] | None = None,
    source_path_like: str | None = None,
    source_path_like_any: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return distinct values for many filter columns with one candidate scan.

    Sidebar hydration previously issued one ``SELECT DISTINCT`` per field.  The
    review schema has more than one hundred categorical/text filters, so opening
    the sidebar repeatedly scanned the wide candidates table.  Fetching just the
    requested columns in one pass is substantially cheaper and keeps ordinary UI
    hydration read-only.
    """
    valid_columns: list[str] = []
    seen: set[str] = set()
    for column in columns:
        column = str(column)
        is_review_col = column in _REVIEW_SELECT_COLS or column in _REVIEW_TEXT_COLS
        if column in seen or (
            column not in _SELECT_COLS and column not in _TEXT_COLS and not is_review_col
        ):
            continue
        seen.add(column)
        valid_columns.append(column)

    if not valid_columns:
        return {}

    where: list[str] = []
    params: list[str] = []
    source_scope_terms: list[str] = []
    source_scope_params: list[str] = []
    if source_path:
        source_scope_terms.append("c.source_path = ?")
        source_scope_params.append(str(source_path))
    if source_paths:
        source_paths = [str(p) for p in source_paths if str(p)]
        if source_paths:
            placeholders = ",".join(["?"] * len(source_paths))
            source_scope_terms.append(f"c.source_path IN ({placeholders})")
            source_scope_params.extend(source_paths)
    if source_path_fallback_like_any:
        fallback_values = [str(v) for v in source_path_fallback_like_any if str(v)]
        if fallback_values:
            source_scope_terms.extend(["c.source_path LIKE ?"] * len(fallback_values))
            source_scope_params.extend([f"%{value}%" for value in fallback_values])
    if source_scope_terms:
        where.append("(" + " OR ".join(source_scope_terms) + ")")
        params.extend(source_scope_params)
    if source_path_like:
        where.append("c.source_path LIKE ?")
        params.append(f"%{str(source_path_like)}%")
    if source_path_like_any:
        like_values = [str(v) for v in source_path_like_any if str(v)]
        if like_values:
            where.append("(" + " OR ".join(["c.source_path LIKE ?"] * len(like_values)) + ")")
            params.extend([f"%{value}%" for value in like_values])

    expressions = [
        _review_filter_expr(column)
        if column in _REVIEW_SELECT_COLS or column in _REVIEW_TEXT_COLS
        else f"c.{column}"
        for column in valid_columns
    ]
    sql = (
        f"SELECT {', '.join(expressions)} FROM candidates c "
        "LEFT JOIN reviews r ON r.candidate_id = c.candidate_id"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)

    distinct: list[set[str]] = [set() for _ in valid_columns]
    cursor = conn.execute(sql, params)
    while True:
        rows = cursor.fetchmany(512)
        if not rows:
            break
        for row in rows:
            for index, value in enumerate(row):
                if value is None:
                    continue
                if isinstance(value, (bytes, bytearray)):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = str(value)
                value_text = str(value).strip()
                if value_text and value_text not in {"None", "NaN", "nan"}:
                    distinct[index].add(value_text)

    return {
        column: sorted(values)
        for column, values in zip(valid_columns, distinct)
    }


def get_numeric_bounds(
    conn: sqlite3.Connection,
    *,
    columns: list[str] | None = None,
    source_path: str | None = None,
    source_paths: list[str] | None = None,
    source_path_fallback_like_any: list[str] | None = None,
    source_path_like: str | None = None,
    source_path_like_any: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Return min/max bounds for numeric candidate columns."""
    selected_cols = columns or sorted(_FLOAT_COLS | _REVIEW_FLOAT_COLS)
    selected_cols = [col for col in selected_cols if col in _FLOAT_COLS or col in _REVIEW_FLOAT_COLS]
    if not selected_cols:
        return {}

    select_parts = []
    for col in selected_cols:
        expr = _review_filter_expr(col) if col in _REVIEW_FLOAT_COLS else f"c.{col}"
        select_parts.append(f"MIN({expr}) AS min_{col}")
        select_parts.append(f"MAX({expr}) AS max_{col}")

    where: list[str] = []
    params: list[str] = []
    source_scope_terms: list[str] = []
    source_scope_params: list[str] = []
    if source_path:
        source_scope_terms.append("c.source_path = ?")
        source_scope_params.append(str(source_path))
    if source_paths:
        source_paths = [str(p) for p in source_paths if str(p)]
        if source_paths:
            placeholders = ",".join(["?"] * len(source_paths))
            source_scope_terms.append(f"c.source_path IN ({placeholders})")
            source_scope_params.extend(source_paths)
    if source_path_fallback_like_any:
        source_path_fallback_like_any = [str(v) for v in source_path_fallback_like_any if str(v)]
        if source_path_fallback_like_any:
            source_scope_terms.extend(["c.source_path LIKE ?"] * len(source_path_fallback_like_any))
            source_scope_params.extend([f"%{value}%" for value in source_path_fallback_like_any])
    if source_scope_terms:
        where.append("(" + " OR ".join(source_scope_terms) + ")")
        params.extend(source_scope_params)
    if source_path_like:
        where.append("c.source_path LIKE ?")
        params.append(f"%{str(source_path_like)}%")
    if source_path_like_any:
        source_path_like_any = [str(v) for v in source_path_like_any if str(v)]
        if source_path_like_any:
            where.append("(" + " OR ".join(["c.source_path LIKE ?"] * len(source_path_like_any)) + ")")
            params.extend([f"%{value}%" for value in source_path_like_any])

    query = (
        f"SELECT {', '.join(select_parts)} FROM candidates c "
        f"LEFT JOIN reviews r ON r.candidate_id = c.candidate_id"
    )
    if where:
        query += " WHERE " + " AND ".join(where)

    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return {}
    col_index = {desc[0]: idx for idx, desc in enumerate(cursor.description or [])}

    bounds: dict[str, dict[str, float | None]] = {}
    for col in selected_cols:
        lo = row[col_index[f"min_{col}"]]
        hi = row[col_index[f"max_{col}"]]
        bounds[col] = {
            "min": float(lo) if lo is not None else None,
            "max": float(hi) if hi is not None else None,
        }
    return bounds


def _migrate_review_candidates_schema_flat_v1(conn: sqlite3.Connection) -> None:
    """Flatten old mixed review candidate rows and remove layer blob columns."""
    marker = conn.execute(
        "SELECT value FROM app_state WHERE key = ?",
        (REVIEW_CANDIDATES_SCHEMA_KEY,),
    ).fetchone()
    marker_value = str(marker[0]) if marker else ""

    table_info = conn.execute("PRAGMA table_info(candidates)").fetchall()
    existing = {str(row[1]) for row in table_info}
    layer_sql_cols = [
        col
        for col in (FEATURE_LAYER_VERSION_COLUMN, *FEATURE_LAYER_COLUMNS)
        if col in existing
    ]
    if marker_value == REVIEW_CANDIDATES_SCHEMA_VERSION and not layer_sql_cols:
        return

    sql_cols = [col for col in _COL_NAMES if col in existing]
    select_cols = list(dict.fromkeys(["rowid", "payload_json", *sql_cols, *layer_sql_cols]))
    cursor = conn.execute(f"SELECT {', '.join(select_cols)} FROM candidates ORDER BY rowid")
    names = [desc[0] for desc in cursor.description or []]
    for values in cursor.fetchall():
        raw = dict(zip(names, values))
        flat = _flatten_review_payload(_payload_json_mapping(raw.get("payload_json")))
        for layer in FEATURE_LAYER_COLUMNS:
            for key, value in parse_layer_value(raw.get(layer)).items():
                if key not in flat or _is_payload_missing(flat.get(key)):
                    flat[key] = value
        for col in sql_cols:
            value = raw.get(col)
            if not _is_payload_missing(value):
                flat[col] = value

        assignments = ["payload_json = ?"]
        params: list[object] = [json.dumps(_review_payload_extra(flat, set(sql_cols)), default=str)]
        for col in sql_cols:
            assignments.append(f"{col} = ?")
            params.append(_sql_value_for_column(col, flat.get(col)))
        params.append(raw["rowid"])
        conn.execute(
            f"UPDATE candidates SET {', '.join(assignments)} WHERE rowid = ?",
            params,
        )

    for col in layer_sql_cols:
        try:
            conn.execute(f"ALTER TABLE candidates DROP COLUMN {col}")
        except sqlite3.OperationalError as exc:
            if "no such column" not in str(exc).lower():
                raise

    conn.execute(
        """
        INSERT INTO app_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (REVIEW_CANDIDATES_SCHEMA_KEY, REVIEW_CANDIDATES_SCHEMA_VERSION, _utc_now()),
    )


def init_db(conn: sqlite3.Connection) -> None:
    col_defs = ",\n            ".join(
        f"{col} {dtype}" for col, dtype, _ in _CANDIDATE_COLUMNS
    )
    review_taxonomy_defs = ",\n            ".join(
        f"{col} {dtype}" for col, dtype in REVIEW_TAXONOMY_SQL_COLUMNS
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            source_path TEXT,
            {col_defs},
            payload_json TEXT NOT NULL,
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS reviews (
            candidate_id TEXT PRIMARY KEY,
            event_class TEXT DEFAULT 'unclassified',
            review_pass INTEGER,
            notes TEXT,
            status TEXT,
            reviewer TEXT,
            {review_taxonomy_defs},
            updated_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            reviewer TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sed_photometry (
            candidate_id TEXT NOT NULL,
            source TEXT NOT NULL,
            band TEXT NOT NULL,
            mag REAL,
            mag_err REAL,
            mag_system TEXT,
            lambda_eff_angstrom REAL,
            flux_lambda REAL,
            flux_lambda_err REAL,
            lambda_l_lambda REAL,
            lambda_l_lambda_err REAL,
            flux_nu_jy REAL,
            flux_nu_jy_err REAL,
            sep_arcsec REAL,
            is_synthetic INTEGER DEFAULT 0,
            is_upper_limit INTEGER DEFAULT 0,
            quality_flags TEXT,
            svo_filter_id TEXT,
            av_coeff REAL,
            PRIMARY KEY(candidate_id, source, band),
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_sed_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(sed_photometry)").fetchall()
    }
    sed_column_defs = {
        "candidate_id": "TEXT",
        "source": "TEXT",
        "band": "TEXT",
        "mag": "REAL",
        "mag_err": "REAL",
        "mag_system": "TEXT",
        "lambda_eff_angstrom": "REAL",
        "flux_lambda": "REAL",
        "flux_lambda_err": "REAL",
        "lambda_l_lambda": "REAL",
        "lambda_l_lambda_err": "REAL",
        "flux_nu_jy": "REAL",
        "flux_nu_jy_err": "REAL",
        "sep_arcsec": "REAL",
        "is_synthetic": "INTEGER DEFAULT 0",
        "is_upper_limit": "INTEGER DEFAULT 0",
        "quality_flags": "TEXT",
        "svo_filter_id": "TEXT",
        "av_coeff": "REAL",
    }
    for col, dtype in sed_column_defs.items():
        if col.lower() not in existing_sed_lower:
            try:
                conn.execute(f"ALTER TABLE sed_photometry ADD COLUMN {col} {dtype}")
                existing_sed_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sed_photometry_candidate
        ON sed_photometry(candidate_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sed_photometry_unique
        ON sed_photometry(candidate_id, source, band)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog_neighbors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            catalog TEXT NOT NULL,
            object_id TEXT,
            object_name TEXT,
            class_value TEXT,
            sep_arcsec REAL NOT NULL,
            period_days REAL,
            rank INTEGER,
            is_known_variable INTEGER DEFAULT 0,
            is_dipper_contaminant INTEGER DEFAULT 0,
            query_radius_arcsec REAL,
            raw_json TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_catalog_neighbor_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(catalog_neighbors)").fetchall()
    }
    catalog_neighbor_column_defs = {
        "candidate_id": "TEXT",
        "catalog": "TEXT",
        "object_id": "TEXT",
        "object_name": "TEXT",
        "class_value": "TEXT",
        "sep_arcsec": "REAL",
        "period_days": "REAL",
        "rank": "INTEGER",
        "is_known_variable": "INTEGER DEFAULT 0",
        "is_dipper_contaminant": "INTEGER DEFAULT 0",
        "query_radius_arcsec": "REAL",
        "raw_json": "TEXT",
    }
    for col, dtype in catalog_neighbor_column_defs.items():
        if col.lower() not in existing_catalog_neighbor_lower:
            try:
                conn.execute(f"ALTER TABLE catalog_neighbors ADD COLUMN {col} {dtype}")
                existing_catalog_neighbor_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_catalog_neighbors_candidate
        ON catalog_neighbors(candidate_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_catalog_neighbors_catalog_sep
        ON catalog_neighbors(catalog, sep_arcsec)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_catalog_neighbors_known_sep
        ON catalog_neighbors(is_known_variable, sep_arcsec)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_catalog_neighbors_dipper_sep
        ON catalog_neighbors(is_dipper_contaminant, sep_arcsec)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_catalog_neighbors_candidate_known_sep
        ON catalog_neighbors(candidate_id, is_known_variable, sep_arcsec)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_catalog_neighbors_candidate_dipper_sep
        ON catalog_neighbors(candidate_id, is_dipper_contaminant, sep_arcsec)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sed_model_fits (
            candidate_id TEXT PRIMARY KEY,
            model_family TEXT,
            fit_version TEXT,
            photometry_method TEXT,
            extinction_law TEXT,
            teff_k REAL,
            teff_err_k REAL,
            logg REAL,
            z REAL,
            av_fixed REAL,
            av_fit REAL,
            av_err REAL,
            rv REAL,
            apparent_scale REAL,
            scale REAL,
            luminosity_lsun REAL,
            radius_rsun REAL,
            chi2 REAL,
            reduced_chi2 REAL,
            n_fit_points INTEGER,
            n_available_points INTEGER,
            n_rejected_points INTEGER,
            fit_lambda_min REAL,
            fit_lambda_max REAL,
            fit_bands_json TEXT,
            priors_json TEXT,
            fit_param_names_json TEXT,
            fit_param_values_json TEXT,
            fit_covariance_json TEXT,
            fit_covariance_status TEXT,
            robust_objective REAL,
            response_manifest_hash TEXT,
            measurement_set_hash TEXT,
            candidate_context_hash TEXT,
            calibration_manifest_hash TEXT,
            input_policy_manifest_hash TEXT,
            fit_recipe_hash TEXT,
            model_grid_hash TEXT,
            model_grid_provenance_json TEXT,
            fit_run_hash TEXT,
            fit_run_id TEXT,
            boundary_flags TEXT,
            status TEXT,
            warning TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_sed_model_fit_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(sed_model_fits)").fetchall()
    }
    sed_model_fit_column_defs = {
        "candidate_id": "TEXT",
        "model_family": "TEXT",
        "fit_version": "TEXT",
        "photometry_method": "TEXT",
        "extinction_law": "TEXT",
        "teff_k": "REAL",
        "teff_err_k": "REAL",
        "logg": "REAL",
        "z": "REAL",
        "av_fixed": "REAL",
        "av_fit": "REAL",
        "av_err": "REAL",
        "rv": "REAL",
        "apparent_scale": "REAL",
        "scale": "REAL",
        "luminosity_lsun": "REAL",
        "radius_rsun": "REAL",
        "chi2": "REAL",
        "reduced_chi2": "REAL",
        "n_fit_points": "INTEGER",
        "n_available_points": "INTEGER",
        "n_rejected_points": "INTEGER",
        "fit_lambda_min": "REAL",
        "fit_lambda_max": "REAL",
        "fit_bands_json": "TEXT",
        "priors_json": "TEXT",
        "fit_param_names_json": "TEXT",
        "fit_param_values_json": "TEXT",
        "fit_covariance_json": "TEXT",
        "fit_covariance_status": "TEXT",
        "robust_objective": "REAL",
        "response_manifest_hash": "TEXT",
        "measurement_set_hash": "TEXT",
        "candidate_context_hash": "TEXT",
        "calibration_manifest_hash": "TEXT",
        "input_policy_manifest_hash": "TEXT",
        "fit_recipe_hash": "TEXT",
        "model_grid_hash": "TEXT",
        "model_grid_provenance_json": "TEXT",
        "fit_run_hash": "TEXT",
        "fit_run_id": "TEXT",
        "boundary_flags": "TEXT",
        "status": "TEXT",
        "warning": "TEXT",
    }
    for col, dtype in sed_model_fit_column_defs.items():
        if col.lower() not in existing_sed_model_fit_lower:
            try:
                conn.execute(f"ALTER TABLE sed_model_fits ADD COLUMN {col} {dtype}")
                existing_sed_model_fit_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sed_model_curves (
            candidate_id TEXT NOT NULL,
            model_family TEXT,
            fit_version TEXT,
            fit_run_hash TEXT,
            fit_run_id TEXT,
            wavelength_angstrom REAL NOT NULL,
            lambda_l_lambda REAL,
            flux_lambda REAL,
            lambda_l_lambda_intrinsic REAL,
            lambda_l_lambda_observed REAL,
            flux_lambda_intrinsic REAL,
            flux_lambda_observed REAL,
            teff_k REAL,
            av_fit REAL,
            scale REAL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_sed_model_curve_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(sed_model_curves)").fetchall()
    }
    sed_model_curve_column_defs = {
        "candidate_id": "TEXT",
        "model_family": "TEXT",
        "fit_version": "TEXT",
        "fit_run_hash": "TEXT",
        "fit_run_id": "TEXT",
        "wavelength_angstrom": "REAL",
        "lambda_l_lambda": "REAL",
        "flux_lambda": "REAL",
        "lambda_l_lambda_intrinsic": "REAL",
        "lambda_l_lambda_observed": "REAL",
        "flux_lambda_intrinsic": "REAL",
        "flux_lambda_observed": "REAL",
        "teff_k": "REAL",
        "av_fit": "REAL",
        "scale": "REAL",
    }
    for col, dtype in sed_model_curve_column_defs.items():
        if col.lower() not in existing_sed_model_curve_lower:
            try:
                conn.execute(f"ALTER TABLE sed_model_curves ADD COLUMN {col} {dtype}")
                existing_sed_model_curve_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sed_model_fits_candidate
        ON sed_model_fits(candidate_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sed_model_curves_candidate
        ON sed_model_curves(candidate_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sed_model_points (
            candidate_id TEXT NOT NULL,
            fit_version TEXT NOT NULL,
            fit_run_hash TEXT,
            fit_run_id TEXT,
            measurement_id TEXT,
            normalization_version TEXT,
            source TEXT NOT NULL,
            band TEXT NOT NULL,
            fit_role TEXT,
            used INTEGER DEFAULT 0,
            exclusion_reason TEXT,
            prediction_status TEXT,
            prediction_reason TEXT,
            observed_flux_nu_jy REAL,
            observed_flux_nu_jy_err REAL,
            observed_flux_lambda REAL,
            observed_flux_lambda_err REAL,
            observed_lambda_l_lambda REAL,
            observed_lambda_l_lambda_err REAL,
            model_flux_nu_jy REAL,
            model_flux_nu_jy_intrinsic REAL,
            observed_mag REAL,
            model_mag REAL,
            mag_system TEXT,
            residual_sigma REAL,
            lambda_eff_angstrom REAL,
            lambda_pivot_angstrom REAL,
            lambda_mean_angstrom REAL,
            lambda_nominal_angstrom REAL,
            lambda_reference_angstrom REAL,
            lambda_isophotal_angstrom REAL,
            plot_lambda_angstrom REAL,
            plot_lambda_kind TEXT,
            model_flux_lambda REAL,
            model_flux_lambda_intrinsic REAL,
            model_lambda_l_lambda REAL,
            model_lambda_l_lambda_intrinsic REAL,
            svo_filter_id TEXT,
            response_hash TEXT,
            calibration_id TEXT,
            calibration_hash TEXT,
            normalization_hash TEXT,
            normalization_method TEXT,
            normalization_provenance_json TEXT,
            passband_fidelity TEXT,
            fit_policy TEXT,
            quality_flags TEXT,
            fit_sigma_log REAL,
            fit_sigma_log_stat REAL,
            fit_sigma_log_systematic REAL,
            input_hash TEXT,
            correlation_group TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_sed_model_point_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(sed_model_points)").fetchall()
    }
    sed_model_point_column_defs = {
        "candidate_id": "TEXT",
        "fit_version": "TEXT",
        "fit_run_hash": "TEXT",
        "fit_run_id": "TEXT",
        "measurement_id": "TEXT",
        "normalization_version": "TEXT",
        "source": "TEXT",
        "band": "TEXT",
        "fit_role": "TEXT",
        "used": "INTEGER DEFAULT 0",
        "exclusion_reason": "TEXT",
        "prediction_status": "TEXT",
        "prediction_reason": "TEXT",
        "observed_flux_nu_jy": "REAL",
        "observed_flux_nu_jy_err": "REAL",
        "observed_flux_lambda": "REAL",
        "observed_flux_lambda_err": "REAL",
        "observed_lambda_l_lambda": "REAL",
        "observed_lambda_l_lambda_err": "REAL",
        "model_flux_nu_jy": "REAL",
        "model_flux_nu_jy_intrinsic": "REAL",
        "observed_mag": "REAL",
        "model_mag": "REAL",
        "mag_system": "TEXT",
        "residual_sigma": "REAL",
        "lambda_eff_angstrom": "REAL",
        "lambda_pivot_angstrom": "REAL",
        "lambda_mean_angstrom": "REAL",
        "lambda_nominal_angstrom": "REAL",
        "lambda_reference_angstrom": "REAL",
        "lambda_isophotal_angstrom": "REAL",
        "plot_lambda_angstrom": "REAL",
        "plot_lambda_kind": "TEXT",
        "model_flux_lambda": "REAL",
        "model_flux_lambda_intrinsic": "REAL",
        "model_lambda_l_lambda": "REAL",
        "model_lambda_l_lambda_intrinsic": "REAL",
        "svo_filter_id": "TEXT",
        "response_hash": "TEXT",
        "calibration_id": "TEXT",
        "calibration_hash": "TEXT",
        "normalization_hash": "TEXT",
        "normalization_method": "TEXT",
        "normalization_provenance_json": "TEXT",
        "passband_fidelity": "TEXT",
        "fit_policy": "TEXT",
        "quality_flags": "TEXT",
        "fit_sigma_log": "REAL",
        "fit_sigma_log_stat": "REAL",
        "fit_sigma_log_systematic": "REAL",
        "input_hash": "TEXT",
        "correlation_group": "TEXT",
    }
    for col, dtype in sed_model_point_column_defs.items():
        if col.lower() not in existing_sed_model_point_lower:
            try:
                conn.execute(f"ALTER TABLE sed_model_points ADD COLUMN {col} {dtype}")
                existing_sed_model_point_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sed_model_points_candidate
        ON sed_model_points(candidate_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sed_model_points_measurement
        ON sed_model_points(measurement_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dustycult_fits (
            candidate_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            runtime_sec REAL,
            artifact_dir TEXT,
            input_path TEXT,
            config_path TEXT,
            manifest_path TEXT,
            command_json TEXT,
            config_json TEXT,
            controls_json TEXT,
            window_json TEXT,
            stellar_json TEXT,
            posterior_json TEXT,
            summary_json TEXT,
            stderr_tail TEXT,
            stdout_tail TEXT,
            error TEXT,
            t0_jd REAL,
            start_jd REAL,
            end_jd REAL,
            n_input_points INTEGER,
            n_curve_points INTEGER,
            PRIMARY KEY(candidate_id, mode),
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_dustycult_fit_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(dustycult_fits)").fetchall()
    }
    dustycult_fit_column_defs = {
        "candidate_id": "TEXT",
        "mode": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
        "runtime_sec": "REAL",
        "artifact_dir": "TEXT",
        "input_path": "TEXT",
        "config_path": "TEXT",
        "manifest_path": "TEXT",
        "command_json": "TEXT",
        "config_json": "TEXT",
        "controls_json": "TEXT",
        "window_json": "TEXT",
        "stellar_json": "TEXT",
        "posterior_json": "TEXT",
        "summary_json": "TEXT",
        "stderr_tail": "TEXT",
        "stdout_tail": "TEXT",
        "error": "TEXT",
        "t0_jd": "REAL",
        "start_jd": "REAL",
        "end_jd": "REAL",
        "n_input_points": "INTEGER",
        "n_curve_points": "INTEGER",
    }
    for col, dtype in dustycult_fit_column_defs.items():
        if col.lower() not in existing_dustycult_fit_lower:
            try:
                conn.execute(f"ALTER TABLE dustycult_fits ADD COLUMN {col} {dtype}")
                existing_dustycult_fit_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dustycult_fits_candidate
        ON dustycult_fits(candidate_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dustycult_predictive_curves (
            candidate_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            point_id INTEGER,
            time REAL,
            band TEXT,
            observed REAL,
            error REAL,
            lower95 REAL,
            lower68 REAL,
            median REAL,
            upper68 REAL,
            upper95 REAL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_dustycult_curve_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(dustycult_predictive_curves)").fetchall()
    }
    dustycult_curve_column_defs = {
        "candidate_id": "TEXT",
        "mode": "TEXT",
        "point_id": "INTEGER",
        "time": "REAL",
        "band": "TEXT",
        "observed": "REAL",
        "error": "REAL",
        "lower95": "REAL",
        "lower68": "REAL",
        "median": "REAL",
        "upper68": "REAL",
        "upper95": "REAL",
    }
    for col, dtype in dustycult_curve_column_defs.items():
        if col.lower() not in existing_dustycult_curve_lower:
            try:
                conn.execute(f"ALTER TABLE dustycult_predictive_curves ADD COLUMN {col} {dtype}")
                existing_dustycult_curve_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dustycult_curves_candidate_mode
        ON dustycult_predictive_curves(candidate_id, mode)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS phoebe_fits (
            candidate_id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            runtime_sec REAL,
            model_kind TEXT,
            period_days REAL,
            period_source TEXT,
            manual_period_days REAL,
            t0_jd REAL,
            input_path TEXT,
            n_input_points INTEGER,
            params_json TEXT,
            metrics_json TEXT,
            plot_json TEXT,
            error TEXT,
            phoebe_version TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    existing_phoebe_fit_lower = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(phoebe_fits)").fetchall()
    }
    phoebe_fit_column_defs = {
        "candidate_id": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
        "runtime_sec": "REAL",
        "model_kind": "TEXT",
        "period_days": "REAL",
        "period_source": "TEXT",
        "manual_period_days": "REAL",
        "t0_jd": "REAL",
        "input_path": "TEXT",
        "n_input_points": "INTEGER",
        "params_json": "TEXT",
        "metrics_json": "TEXT",
        "plot_json": "TEXT",
        "error": "TEXT",
        "phoebe_version": "TEXT",
    }
    for col, dtype in phoebe_fit_column_defs.items():
        if col.lower() not in existing_phoebe_fit_lower:
            try:
                conn.execute(f"ALTER TABLE phoebe_fits ADD COLUMN {col} {dtype}")
                existing_phoebe_fit_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_phoebe_fits_candidate
        ON phoebe_fits(candidate_id)
        """
    )
    # Migrate: add any columns missing from older DBs.
    existing_candidate_columns = {
        str(row[1]).lower(): str(row[1])
        for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    legacy_candidate_column_renames = {
        "allwise_w3": "w3",
        "allwise_w3_err": "w3_err",
        "allwise_w4": "w4",
        "allwise_w4_err": "w4_err",
        "w1_w2": "w1_w2",
    }
    legacy_candidate_column_renames["W1_W2".lower()] = "w1_w2"
    for old_col, new_col in legacy_candidate_column_renames.items():
        old_actual = existing_candidate_columns.get(old_col)
        new_actual = existing_candidate_columns.get(new_col)
        if old_actual and (not new_actual or old_actual != new_col):
            try:
                conn.execute(f"ALTER TABLE candidates RENAME COLUMN {old_actual} TO {new_col}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
                continue
            existing_candidate_columns.pop(old_col, None)
            existing_candidate_columns[new_col] = new_col

    existing_lower = set(existing_candidate_columns)
    for col, dtype, _ in _CANDIDATE_COLUMNS:
        if col.lower() not in existing_lower:
            try:
                conn.execute(f"ALTER TABLE candidates ADD COLUMN {col} {dtype}")
                existing_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
                # Column already exists (e.g. race or schema drift); skip

    _migrate_review_candidates_schema_flat_v1(conn)

    existing_review_columns = {
        str(row[1]).lower(): str(row[1])
        for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
    }
    existing_review_lower = set(existing_review_columns)
    for col, dtype in REVIEW_TAXONOMY_SQL_COLUMNS:
        if col.lower() not in existing_review_lower:
            try:
                conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} {dtype}")
                existing_review_lower.add(col.lower())
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

    # Schema v6 has exactly one 1--4 review-score column.  Older databases
    # stored those integers in interest_score and carried a second, unused
    # classification_confidence column.  Prefer the original integers, use a
    # numeric/categorical confidence value only when the original is missing,
    # then remove the duplicate column and rename the integer column.
    review_columns = {
        str(row[1]).lower(): str(row[1])
        for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
    }
    legacy_score_col = review_columns.get("interest_score")
    confidence_col = review_columns.get("classification_confidence")
    if legacy_score_col is not None:
        if confidence_col is not None:
            conn.execute(
                f"""
                UPDATE reviews
                SET {legacy_score_col} = COALESCE(
                    {legacy_score_col},
                    CASE LOWER(TRIM(CAST({confidence_col} AS TEXT)))
                        WHEN '1' THEN 1
                        WHEN '2' THEN 2
                        WHEN '3' THEN 3
                        WHEN '4' THEN 4
                        WHEN 'morphology_only' THEN 1
                        WHEN 'possible' THEN 2
                        WHEN 'likely' THEN 3
                        WHEN 'secure' THEN 4
                    END
                )
                """
            )
            conn.execute(f"ALTER TABLE reviews DROP COLUMN {confidence_col}")
        conn.execute(
            f"ALTER TABLE reviews RENAME COLUMN {legacy_score_col} TO classification_confidence"
        )

    # Schema v5 removes the old synthetic ``lsp_*`` compatibility values from
    # candidate rows.  Match the complete alias signature so an independently
    # computed Lomb--Scargle result (which has a finite lsp_power or differs
    # from the selected PDM/CE solution) is never erased.  A fresh periodicity
    # run will repopulate these columns with real Lomb--Scargle measurements.
    conn.execute(
        """
        UPDATE candidates
        SET lsp_period = NULL,
            lsp_bootstrap_sig = NULL,
            lsp_is_alias = NULL,
            lsp_is_significant = NULL
        WHERE lsp_power IS NULL
          AND lsp_period IS NOT NULL
          AND lsp_period IS periodicity_period
          AND lsp_bootstrap_sig IS periodicity_bootstrap_sig
          AND lsp_is_alias IS periodicity_alias_flag
          AND lsp_is_significant IS periodicity_is_significant
        """
    )

    _ensure_review_content_revision_tracking(conn)

    # The provenance-preserving SED schema is additive.  Keep its definition in
    # a focused module so the legacy review tables remain available throughout
    # the v2 -> v3 shadow migration.
    from malca.review.sed_storage import ensure_sed_storage_schema

    ensure_sed_storage_schema(conn)
    conn.execute(
        """
        INSERT INTO app_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        WHERE app_state.value IS NOT excluded.value
        """,
        (REVIEW_DB_SCHEMA_KEY, str(REVIEW_DB_SCHEMA_VERSION), _utc_now()),
    )
    conn.commit()


def db_connect(
    db_path: Path,
    *,
    initialize_if_missing: bool = True,
) -> sqlite3.Connection:
    """Open and configure a review SQLite connection.

    Existing databases are never migrated here.  ``initialize_if_missing`` is
    retained only so established callers can create a brand-new review DB in a
    single step; application startup performs existing-schema migration via
    :func:`ensure_review_db_schema`.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
        check_same_thread=False,
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    if initialize_if_missing and not _table_exists(conn, "candidates"):
        init_db(conn)
    return conn


def ensure_review_db_schema(db_path: str | Path) -> bool:
    """Migrate a review DB once when its stored schema version is outdated.

    Returns ``True`` when initialization/migration ran and ``False`` when the
    database was already current.  The current-schema path performs no write.
    """

    path = Path(db_path).expanduser()
    with db_connect(path, initialize_if_missing=False) as conn:
        stored_version = _stored_review_db_schema_version(conn)
        if stored_version == REVIEW_DB_SCHEMA_VERSION:
            return False
        if stored_version is not None and stored_version > REVIEW_DB_SCHEMA_VERSION:
            raise RuntimeError(
                "This MALCA build only understands review DB schema "
                f"{REVIEW_DB_SCHEMA_VERSION}, but the database is version {stored_version}"
            )
        init_db(conn)
        return True


def validate_review_db_integrity(db_path: str | Path) -> dict[str, object]:
    """Run expensive review/SED integrity checks only when explicitly asked."""

    from malca.review.sed_storage import validate_sed_storage_integrity

    path = Path(db_path).expanduser()
    with db_connect(path, initialize_if_missing=False) as conn:
        quick_check_rows = conn.execute("PRAGMA quick_check").fetchall()
        quick_check = [str(row[0]) for row in quick_check_rows]
        if quick_check != ["ok"]:
            raise sqlite3.IntegrityError(
                "SQLite quick_check failed: " + "; ".join(quick_check[:10])
            )
        foreign_key_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_violations:
            preview = "; ".join(str(tuple(row)) for row in foreign_key_violations[:10])
            raise sqlite3.IntegrityError("SQLite foreign-key violations: " + preview)
        validate_sed_storage_integrity(conn)
        return {
            "path": str(path.resolve()),
            "quick_check": "ok",
            "foreign_key_violations": 0,
            "review_schema_version": _stored_review_db_schema_version(conn),
        }


def review_content_signature(db_path: str | Path) -> str:
    """Return a cache signature changed only by candidate/review row writes."""

    path = Path(db_path).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        uri = resolved.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
            row = conn.execute(
                "SELECT revision FROM review_content_revision WHERE scope = ?",
                (REVIEW_CONTENT_REVISION_SCOPE,),
            ).fetchone()
        if row is not None:
            return f"{resolved}|content:{int(row[0])}"
    except Exception:
        pass

    # Compatibility fallback for hand-built/legacy DBs not yet migrated.
    parts = [str(resolved)]
    for suffix, label in (("", "db"), ("-wal", "wal")):
        candidate = resolved if not suffix else Path(f"{resolved}{suffix}")
        try:
            stat = candidate.stat()
            parts.append(f"{label}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}")
        except Exception:
            parts.append(f"{label}:missing")
    return "|".join(parts)


def save_app_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO app_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=excluded.updated_at
        """,
        (key, value, _utc_now()),
    )
    conn.commit()


def load_app_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return default if row is None else str(row[0])


def upsert_catalog_neighbor_rows(
    conn: sqlite3.Connection,
    rows: pd.DataFrame | list[dict[str, object]] | None,
    *,
    commit: bool = True,
) -> int:
    """Replace long-form catalog-neighbor rows for touched candidate/catalog pairs."""
    frame = normalize_catalog_neighbor_frame(rows)
    if frame.empty:
        return 0

    pairs = (
        frame[["candidate_id", "catalog"]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    for candidate_id, catalog in pairs:
        conn.execute(
            "DELETE FROM catalog_neighbors WHERE candidate_id = ? AND catalog = ?",
            (candidate_id, catalog),
        )

    placeholders = ", ".join(["?"] * len(CATALOG_NEIGHBOR_COLUMNS))
    sql = (
        f"INSERT INTO catalog_neighbors ({', '.join(CATALOG_NEIGHBOR_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    values: list[tuple[object, ...]] = []
    for _, row in frame.iterrows():
        row_values: list[object] = []
        for column in CATALOG_NEIGHBOR_COLUMNS:
            value = row[column]
            if column in {"is_known_variable", "is_dipper_contaminant"}:
                row_values.append(int(bool(value)))
            elif value is None:
                row_values.append(None)
            else:
                try:
                    row_values.append(None if pd.isna(value) else value)
                except Exception:
                    row_values.append(value)
        values.append(tuple(row_values))

    conn.executemany(sql, values)
    if commit:
        conn.commit()
    return len(values)


def import_candidates(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source_path: str,
    *,
    characterize_before_import: bool = True,
    characterize_crossmatch: Path = VSX_CROSSMATCH_PATH,
    characterize_chunk_size: int = GAIA_CHUNK_SIZE,
    characterize_cache: Path = GAIA_CACHE_FILE,
    characterize_dust: bool = True,
    characterize_starhorse: str | None = "tap",
    vet_before_import: bool = True,
) -> tuple[int, int]:
    if df.empty:
        return 0, 0

    df_use = df
    if characterize_before_import:
        try:
            from malca.enrichment.characterize import characterize_candidates_df

            df_use = characterize_candidates_df(
                df,
                crossmatch=characterize_crossmatch,
                chunk_size=characterize_chunk_size,
                cache=characterize_cache,
                dust=characterize_dust,
                starhorse=characterize_starhorse,
            )
            if not isinstance(df_use, pd.DataFrame) or df_use.empty:
                df_use = df
        except Exception as e:
            print(f"Warning: characterization before import failed: {e}")
            df_use = df

    df_use = to_layer_first_frame(df_use)

    if vet_before_import:
        # Auto-detect completed vetting from positive evidence only. A populated
        # False/empty marker such as gaia_var_flag=False is not enough to prove
        # the lookup completed successfully.
        def _positive_vetting_mask(frame: pd.DataFrame) -> pd.Series:
            mask = pd.Series(False, index=frame.index, dtype=bool)
            completed_col = "_vetting_completed"
            if completed_col in frame.columns:
                completed = frame[completed_col].fillna(False).astype(str).str.strip().str.lower()
                mask |= completed.isin({"1", "true", "yes", "y", "t"})
            for col in _VET_STRING_EVIDENCE_COLS:
                if col not in frame.columns:
                    continue
                values = frame[col].fillna("").astype(str).str.strip()
                mask |= values != ""
            truthy = {"1", "true", "yes", "y", "t"}
            for col in _VET_BOOL_EVIDENCE_COLS:
                if col not in frame.columns:
                    continue
                values = frame[col].fillna(False)
                if values.dtype == bool:
                    mask |= values
                else:
                    mask |= values.astype(str).str.strip().str.lower().isin(truthy)
            return mask

        def _has_nonempty_value(cols: set[str]) -> bool:
            for col in cols:
                if col not in df_use.columns:
                    continue
                values = df_use[col].fillna("").astype(str).str.strip()
                if (values != "").any():
                    return True
            return False

        def _has_truthy_value(cols: set[str]) -> bool:
            truthy = {"1", "true", "yes", "y", "t"}
            for col in cols:
                if col not in df_use.columns:
                    continue
                values = df_use[col].fillna(False)
                if values.dtype == bool and values.any():
                    return True
                text_values = values.astype(str).str.strip().str.lower()
                if text_values.isin(truthy).any():
                    return True
            return False

        _VET_STRING_EVIDENCE_COLS = {
            "simbad_main_id",
            "gaia_var_class",
            "asassn_var_name",
            "asassn_var_type",
            "microlens_name",
            "ztf_var_type",
            "tns_name",
            "alerce_oid",
            "alerce_lc_class",
        }
        _VET_BOOL_EVIDENCE_COLS = {
            "vetting_likely_known",
            "gaia_var_flag",
            "microlens_match",
            "xray_det",
            "atlas_has_phot",
        }
        _has_vetting = (
            _has_nonempty_value(_VET_STRING_EVIDENCE_COLS)
            or _has_truthy_value(_VET_BOOL_EVIDENCE_COLS)
        )
        if _has_vetting:
            print("Vetting: columns already present in input, skipping re-vetting")
            vet_before_import = False

    if vet_before_import:
        try:
            from malca.enrichment.vetting import vet_candidates

            # --- vetting cache: skip candidates already vetted ----
            # Use the unified repo cache for real paths or fetch:// sources.
            if source_path:
                _vetting_cache_dir = (DEFAULT_CACHE_DIR / "vetting" / "review_import").expanduser()
                _vetting_cache_dir.mkdir(parents=True, exist_ok=True)
                _digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:16]
                _cache_token = "".join(
                    ch if ch.isalnum() or ch in {"_", "-"} else "_"
                    for ch in source_path.replace("fetch://", "")
                ).strip("_")[:80] or "source"
                _vetting_cache_path = _vetting_cache_dir / f"{_cache_token}_{_digest}.parquet"
                _legacy_vetting_cache_path = (
                    Path("output") / "cache" / "vetting_cache" / (source_path.replace("fetch://", "").replace("/", "_") + ".parquet")
                    if source_path.startswith("fetch://")
                    else Path(source_path + ".vetting_cache.parquet")
                )
                _use_cache = True
            else:
                _vetting_cache_path = None
                _legacy_vetting_cache_path = None
                _use_cache = False
            _cache_df = None
            _id_col = "candidate_id" if "candidate_id" in df_use.columns else None
            n_new = len(df_use)  # default: vet everything

            _vetting_read_cache_path = _vetting_cache_path
            if (
                _vetting_read_cache_path is not None
                and not _vetting_read_cache_path.exists()
                and _legacy_vetting_cache_path is not None
                and _legacy_vetting_cache_path.exists()
            ):
                _vetting_read_cache_path = _legacy_vetting_cache_path

            if _id_col and _vetting_read_cache_path is not None and _vetting_read_cache_path.exists():
                try:
                    _cache_df = pd.read_parquet(_vetting_read_cache_path)
                    valid_cache_mask = _positive_vetting_mask(_cache_df)
                    if not valid_cache_mask.all():
                        n_ignored = int((~valid_cache_mask).sum())
                        _cache_df = _cache_df.loc[valid_cache_mask].copy()
                        print(f"Vetting cache: ignoring {n_ignored} legacy entries without completion evidence")
                    cached_ids = set(_cache_df[_id_col])
                    mask_new = ~df_use[_id_col].isin(cached_ids)
                    n_cached = (~mask_new).sum()
                    n_new = mask_new.sum()
                    print(f"Vetting cache: {n_cached} cached, {n_new} to vet")
                except Exception:
                    _cache_df = None

            if _cache_df is not None and n_new == 0:
                # All candidates cached — merge vetting columns from cache
                cache_cols = [c for c in VETTING_COLUMNS if c in _cache_df.columns]
                df_use = df_use.merge(
                    _cache_df[[_id_col] + cache_cols],
                    on=_id_col, how="left", suffixes=("", "_cached"),
                )
                df_use = df_use[[c for c in df_use.columns if not c.endswith("_cached")]]
                if _vetting_cache_path is not None and _vetting_read_cache_path != _vetting_cache_path:
                    try:
                        _vetting_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        _cache_df.to_parquet(_vetting_cache_path, index=False)
                    except Exception:
                        pass
                print("Vetting: all candidates served from cache")
            else:
                if _cache_df is not None and n_new > 0:
                    # Vet only the new candidates
                    _run_tns = not (source_path and source_path.startswith("fetch://"))
                    df_new = vet_candidates(
                        df_use.loc[mask_new],
                        run_pm_check=False,
                        run_tns=_run_tns,
                        method="xmatch",
                    )
                    # Merge cached vetting columns onto cached rows
                    cache_cols = [c for c in VETTING_COLUMNS if c in _cache_df.columns]
                    df_old = df_use.loc[~mask_new].merge(
                        _cache_df[[_id_col] + cache_cols],
                        on=_id_col, how="left", suffixes=("", "_cached"),
                    )
                    df_old = df_old[[c for c in df_old.columns if not c.endswith("_cached")]]
                    df_use = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    # No cache or no candidate_id — vet everything
                    _run_tns = not (source_path and source_path.startswith("fetch://"))
                    df_use = vet_candidates(
                        df_use,
                        run_pm_check=False,
                        run_tns=_run_tns,
                        method="xmatch",
                    )

                # Update cache
                if _id_col and _vetting_cache_path is not None:
                    try:
                        vet_cols = [c for c in VETTING_COLUMNS if c in df_use.columns]
                        new_cache = df_use[[_id_col] + vet_cols].copy()
                        new_cache["_vetting_completed"] = True
                        if _cache_df is not None:
                            new_cache = pd.concat([
                                _cache_df[~_cache_df[_id_col].isin(new_cache[_id_col])],
                                new_cache,
                            ], ignore_index=True)
                        _vetting_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        new_cache.to_parquet(_vetting_cache_path, index=False)
                        print(f"Vetting cache saved: {len(new_cache)} entries → {_vetting_cache_path}")
                    except Exception as e:
                        print(f"Warning: failed to save vetting cache: {e}")
        except Exception as e:
            raise RuntimeError(f"Vetting before import failed: {e}") from e

    return upsert_candidates_frame(conn, df_use, default_source_path=str(source_path))


def _catalog_neighbor_filter_radius(value: object) -> float:
    radius = _to_float(value)
    if radius is None:
        radius = DEFAULT_REVIEW_VETTING_RADIUS_ARCSEC
    radius = max(0.0, min(float(radius), MAX_REVIEW_VETTING_RADIUS_ARCSEC))
    return radius


def _queue_where_params(filters: dict | None = None) -> tuple[list[str], list[object]]:
    """Build queue WHERE clauses and bound params from filter parameters."""
    if filters is None:
        filters = {}

    where: list[str] = []
    params: list[object] = []
    select_filter_mode = str(filters.get("select_filter_mode") or "exclude").strip().lower()
    if select_filter_mode not in {"include", "exclude"}:
        select_filter_mode = "exclude"
    select_filter_logic = str(filters.get("select_filter_logic") or "and").strip().lower()
    if select_filter_logic not in {"and", "or"}:
        select_filter_logic = "and"

    # --- review status ---
    if filters.get('only_unreviewed'):
        where.append("(r.workflow_status IS NULL OR r.workflow_status='unreviewed')")
    workflow_status_exact = str(
        filters.get("workflow_status_exact") or ""
    ).strip()
    if workflow_status_exact:
        where.append("(COALESCE(r.workflow_status, 'unreviewed') = ?)")
        params.append(workflow_status_exact)

    # Manifest-backed TUI external-photometry availability. JSON membership
    # keeps even large campaign ID sets within SQLite's bound-parameter limit.
    candidate_id_membership = filters.get("candidate_id_membership")
    if isinstance(candidate_id_membership, dict):
        if "required" in candidate_id_membership:
            required_ids = [
                str(value).strip()
                for value in (candidate_id_membership.get("required") or [])
                if str(value).strip()
            ]
            if required_ids:
                where.append(
                    "c.candidate_id IN "
                    "(SELECT CAST(value AS TEXT) FROM json_each(?))"
                )
                params.append(json.dumps(required_ids))
            else:
                where.append("(0 = 1)")
        excluded_ids = [
            str(value).strip()
            for value in (candidate_id_membership.get("excluded") or [])
            if str(value).strip()
        ]
        if excluded_ids:
            where.append(
                "c.candidate_id NOT IN "
                "(SELECT CAST(value AS TEXT) FROM json_each(?))"
            )
            params.append(json.dumps(excluded_ids))

    # --- failed_any shortcut ---
    if filters.get('require_failed_any_false'):
        where.append("(c.failed_any IS NULL OR c.failed_any = 0)")

    catalog_neighbor_radius = _catalog_neighbor_filter_radius(
        filters.get("catalog_neighbor_radius_arcsec")
    )
    if _as_bool(filters.get("exclude_known_catalog_neighbors")):
        where.append(
            """
            NOT EXISTS (
                SELECT 1
                FROM catalog_neighbors cn
                WHERE cn.candidate_id = c.candidate_id
                  AND cn.is_known_variable = 1
                  AND cn.sep_arcsec IS NOT NULL
                  AND cn.sep_arcsec <= ?
            )
            """.strip()
        )
        params.append(catalog_neighbor_radius)
    if _as_bool(filters.get("exclude_dipper_catalog_neighbors")):
        where.append(
            """
            NOT EXISTS (
                SELECT 1
                FROM catalog_neighbors cn
                WHERE cn.candidate_id = c.candidate_id
                  AND cn.is_dipper_contaminant = 1
                  AND cn.sep_arcsec IS NOT NULL
                  AND cn.sep_arcsec <= ?
            )
            """.strip()
        )
        params.append(catalog_neighbor_radius)

    # Campaign-local TUI catalog menus maintain explicit per-type keep/exclude
    # decisions.  These exclusions are independent of the legacy select-filter
    # include/exclude mode used by the browser and taxonomy controls.
    catalog_type_exclusions = filters.get("catalog_type_exclusions")
    if isinstance(catalog_type_exclusions, dict):
        allowed_catalog_columns = {
            "vsx_class",
            "gaia_var_class",
            "asassn_var_type",
            "simbad_otype",
            "ztf_var_type",
            "microlens_catalog",
            "tns_type",
            "alerce_lc_class",
            "yso_class",
        }
        for column in sorted(allowed_catalog_columns):
            raw_values = catalog_type_exclusions.get(column)
            if isinstance(raw_values, str):
                raw_values = [raw_values]
            if not isinstance(raw_values, (list, tuple, set)):
                continue
            values = [
                str(value).strip()
                for value in raw_values
                if str(value).strip()
            ]
            if not values:
                continue
            placeholders = ",".join(["?"] * len(values))
            where.append(
                f"(c.{column} IS NULL OR TRIM(c.{column}) = '' "
                f"OR TRIM(c.{column}) NOT IN ({placeholders}))"
            )
            params.extend(values)

    # --- optional source-path scoping (exact path, with portable run-token fallback) ---
    source_scope_terms: list[str] = []
    source_scope_params: list[object] = []
    source_path = filters.get('source_path')
    if source_path:
        source_scope_terms.append("c.source_path = ?")
        source_scope_params.append(str(source_path))

    source_paths = filters.get('source_paths')
    if source_paths:
        source_paths = [str(p) for p in source_paths if str(p)]
        if source_paths:
            placeholders = ",".join(["?"] * len(source_paths))
            source_scope_terms.append(f"c.source_path IN ({placeholders})")
            source_scope_params.extend(source_paths)

    source_path_fallback_like_any = filters.get('source_path_fallback_like_any')
    if source_path_fallback_like_any:
        source_path_fallback_like_any = [str(v) for v in source_path_fallback_like_any if str(v)]
        if source_path_fallback_like_any:
            source_scope_terms.extend(["c.source_path LIKE ?"] * len(source_path_fallback_like_any))
            source_scope_params.extend([f"%{value}%" for value in source_path_fallback_like_any])

    if source_scope_terms:
        where.append("(" + " OR ".join(source_scope_terms) + ")")
        params.extend(source_scope_params)

    # --- optional source-path scope token (bundle-like substring) ---
    source_path_like = filters.get('source_path_like')
    if source_path_like:
        where.append("(c.source_path LIKE ?)")
        params.append(f"%{str(source_path_like)}%")

    source_path_like_any = filters.get('source_path_like_any')
    if source_path_like_any:
        source_path_like_any = [str(v) for v in source_path_like_any if str(v)]
        if source_path_like_any:
            where.append("(" + " OR ".join(["c.source_path LIKE ?"] * len(source_path_like_any)) + ")")
            params.extend([f"%{value}%" for value in source_path_like_any])

    # --- Any / True / False / Unset bool-mode filters (auto-generated) ---
    mode_map = {"Any": None, "True": 1, "False": 0, "Unset": "unset"}
    for col in _BOOL_COLS:
        key = f"{col}_mode"
        mode = filters.get(key, "Any")
        val = mode_map.get(mode)
        if val == "unset":
            where.append(f"(c.{col} IS NULL)")
        elif val is not None:
            if col in _FALSE_INCLUDES_UNSET_BOOL_COLS and val == 0:
                where.append(f"(c.{col} IS NULL OR c.{col} = ?)")
            else:
                where.append(f"(c.{col} = ?)")
            params.append(val)

    # --- numeric range filters (auto-generated) ---
    # Convention: "min_<col>" → >=, "max_<col>" → <=
    for col in sorted(_FLOAT_COLS | _REVIEW_FLOAT_COLS):
        expr = _review_filter_expr(col) if col in _REVIEW_FLOAT_COLS else f"c.{col}"
        for prefix, op in [("min_", ">="), ("max_", "<=")]:
            key = f"{prefix}{col}"
            val = filters.get(key)
            if val is not None:
                where.append(f"({expr} IS NOT NULL AND {expr} {op} ?)")
                params.append(float(val))

    # --- string filters (auto-generated; exact match) ---
    for col in sorted(_TEXT_COLS | _REVIEW_TEXT_COLS):
        expr = _review_filter_expr(col) if col in _REVIEW_TEXT_COLS else f"c.{col}"
        val = filters.get(col)
        if val and val != "Any":
            val = str(val).strip()
            if val and val != "Any":
                where.append(f"({expr} IS NOT NULL AND {expr} = ?)")
                params.append(val)

    # --- select filters (multi-value dropdowns, grouped with AND or OR) ---
    select_where: list[str] = []
    select_params: list[object] = []
    for col in sorted(_SELECT_COLS | _REVIEW_SELECT_COLS):
        expr = _review_filter_expr(col) if col in _REVIEW_SELECT_COLS else f"c.{col}"
        exc = filters.get(f"exclude_{col}")
        if exc:
            placeholders = ",".join(["?"] * len(exc))
            if select_filter_mode == "include":
                select_where.append(f"({expr} IS NOT NULL AND {expr} IN ({placeholders}))")
            else:
                select_where.append(f"({expr} IS NULL OR {expr} NOT IN ({placeholders}))")
            select_params.extend(exc)
    if select_where:
        joiner = " OR " if select_filter_logic == "or" else " AND "
        where.append("(" + joiner.join(select_where) + ")")
        params.extend(select_params)

    return where, params


def _queue_order_clause(filters: dict | None = None) -> str:
    """Build the SQL ORDER BY clause for queue queries."""
    if filters is None:
        filters = {}

    # --- sorting (any float column + review columns, multi-column) ---
    _sortable = {c: f"c.{c}" for c in _FLOAT_COLS}
    _sortable.update({c: _review_filter_expr(c) for c in _REVIEW_FLOAT_COLS})
    _sortable["candidate_id"] = "c.candidate_id"
    _sortable.update({"updated_at": "r.updated_at", "review_pass": "r.review_pass"})
    sort_cols = filters.get('sort_cols') or [filters.get('sort_col', 'candidate_id')]
    direction = "DESC" if filters.get('sort_desc') else "ASC"
    order_parts = []
    for sc in sort_cols:
        col_expr = _sortable.get(sc)
        if col_expr:
            order_parts.append(f"{col_expr} {direction}")
    if not order_parts:
        order_parts.append(f"c.candidate_id {direction}")
    order_clause = ", ".join(order_parts)
    if "c.candidate_id" not in order_clause:
        order_clause += ", c.candidate_id ASC"
    return order_clause


def count_queue(
    conn: sqlite3.Connection,
    *,
    filters: dict | None = None,
) -> int:
    """Count queue rows matching the supplied filter parameters."""
    where, params = _queue_where_params(filters)

    query = """
        SELECT COUNT(*)
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if where:
        query += " WHERE " + " AND ".join(where)

    row = conn.execute(query, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def query_queue(
    conn: sqlite3.Connection,
    *,
    filters: dict | None = None,
    ids_only: bool = False,
) -> pd.DataFrame:
    """Query the candidate queue using filter parameters."""
    where, params = _queue_where_params(filters)
    order_clause = _queue_order_clause(filters)

    if ids_only:
        query = """
            SELECT c.candidate_id
            FROM candidates c
            LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
        """
    else:
        query = f"""
            SELECT
                c.candidate_id,
                c.asas_sn_id,
                c.lc_path,
                c.failed_any,
                c.periodic_flag,
                c.catalog_match,
                c.high_ruwe_flag,
                c.periodicity_score,
                c.periodicity_period,
                c.periodicity_method,
                c.periodicity_base_period,
                c.periodicity_harmonic_factor,
                c.periodicity_harmonic_objective,
                c.periodicity_scatter_ratio,
                c.periodicity_alias_flag,
                c.periodicity_alias_matches,
                c.periodicity_bootstrap_sig,
                c.periodicity_is_significant,
                c.lsp_bootstrap_sig,
                c.lsp_power,
                c.lsp_period,
                c.dip_best_log_bf,
                c.jump_best_log_bf,
                r.classification_confidence,
                r.review_pass,
                r.workflow_status AS status,
                r.notes,
                r.reviewer,
                r.updated_at
            FROM candidates c
            LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
        """
    if where:
        query += " WHERE " + " AND ".join(where)
    query += f" ORDER BY {order_clause}"
    return pd.read_sql_query(query, conn, params=params)


def get_candidate_payload(conn: sqlite3.Connection, candidate_id: str) -> dict:
    """Return merged payload for display: payload_json plus SQL columns (so vetting etc. show in GUI)."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(candidates)")
    actual_cols = set([c[1] for c in cursor.fetchall()])
    cols_to_fetch = [c for c in _COL_NAMES if c in actual_cols]
    col_list = ", ".join(["payload_json"] + cols_to_fetch)
    row = conn.execute(
        f"SELECT {col_list} FROM candidates WHERE candidate_id=?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return {}
    payload = _flatten_review_payload(_payload_json_mapping(row[0]))

    # Merge SQL columns into payload so asassn_var_type, ztf_var_type, tns_type etc. show when only in SQL
    for i, col in enumerate(cols_to_fetch):
        if i + 1 >= len(row):
            break
        raw = row[i + 1]
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            continue
        etype = _COL_TYPE_MAP.get(col)
        if etype == "bool":
            payload[col] = bool(_as_bool(raw))
        elif etype == "float":
            f = _to_float(raw)
            if f is not None:
                payload[col] = f
        else:
            payload[col] = str(raw).strip() if raw is not None else ""

    if _payload_layer_value(payload, "high_pm_flag") in (None, "") and _payload_layer_value(payload, "pm_total") not in (None, ""):
        pm_total = _to_float(_payload_layer_value(payload, "pm_total"))
        if pm_total is not None:
            payload["high_pm_flag"] = bool(pm_total > LTV_MAX_PM)
    if has_known_catalog_evidence(payload):
        payload["vetting_likely_known"] = True
    elif has_catalog_vetting_context(payload):
        payload["vetting_likely_known"] = False
    return payload


def replace_candidate_payload_fields(
    conn: sqlite3.Connection,
    candidate_id: str,
    updates: dict[str, object],
    *,
    clear_keys: set[str] | None = None,
    commit: bool = True,
) -> bool:
    """Replace selected payload fields while keeping unrelated candidate data.

    Keys in ``clear_keys`` are removed from ``payload_json`` before ``updates``
    are merged in. Matching SQL columns are cleared to ``NULL`` unless a new
    value is supplied in ``updates``.
    """
    row = conn.execute(
        "SELECT payload_json FROM candidates WHERE candidate_id=?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return False

    try:
        payload = json.loads(row[0]) if row[0] else {}
    except Exception:
        payload = {}
    clear = set(clear_keys or ())
    payload = _drop_payload_keys(payload if isinstance(payload, dict) else {}, clear)
    payload = _merge_layer_payload_updates(payload, updates)

    table_cols = {
        str(info[1])
        for info in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    update_values = _payload_update_values(updates)
    sql_targets = {key for key in clear if key in table_cols}
    sql_targets.update(key for key in update_values if key in table_cols)
    payload = _review_payload_extra(payload, table_cols)

    conn.execute(
        "UPDATE candidates SET payload_json = ? WHERE candidate_id = ?",
        (json.dumps(payload, default=str), candidate_id),
    )

    if sql_targets:
        assignments: list[str] = []
        params: list[object] = []
        for col in sorted(sql_targets):
            assignments.append(f"{col} = ?")
            if col not in update_values:
                params.append(None)
                continue

            params.append(_sql_value_for_column(col, update_values[col]))

        params.append(candidate_id)
        conn.execute(
            f"UPDATE candidates SET {', '.join(assignments)} WHERE candidate_id = ?",
            params,
        )

    if commit:
        conn.commit()
    return True


def _quote_sql_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _candidate_column_map(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("PRAGMA table_info(candidates)").fetchall()
    return {str(row[1]).lower(): str(row[1]) for row in rows}


def _json_value_expr(key: str) -> str:
    path = str(key).replace("'", "''")
    return (
        "COALESCE("
        f"json_extract(payload_json, '$.{path}'), "
        f"json_extract(json_extract(payload_json, '$.payload_json'), '$.{path}')"
        ")"
    )


def _background_value_expr(
    key: str,
    column_map: dict[str, str],
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    exprs: list[str] = []
    for candidate_key in (key, *aliases):
        actual_column = column_map.get(str(candidate_key).lower())
        if actual_column is not None:
            exprs.append(_quote_sql_identifier(actual_column))
        exprs.append(_json_value_expr(str(candidate_key)))
    if not exprs:
        return "NULL"
    if len(exprs) == 1:
        return exprs[0]
    return "COALESCE(" + ", ".join(exprs) + ")"


def _load_background_pair(
    conn: sqlite3.Connection,
    x_expr: str,
    y_expr: str,
    column_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired arrays for a diagnostic background plane."""
    columns = column_map if column_map is not None else _candidate_column_map(conn)
    x_sql = _background_value_expr(x_expr, columns)
    y_sql = _background_value_expr(y_expr, columns)
    rows = conn.execute(
        f"SELECT {x_sql}, {y_sql} FROM candidates"
    ).fetchall()
    if not rows:
        return np.empty(0), np.empty(0)
    arr = np.array(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty(0), np.empty(0)
    mask = np.isfinite(arr).all(axis=1)
    if not mask.any():
        return np.empty(0), np.empty(0)
    return arr[mask, 0], arr[mask, 1]


def get_diagnostic_background(conn: sqlite3.Connection) -> dict:
    """Load background arrays for diagnostic plots.

    Returns a dict with keys for the review GUI diagnostic planes.
    Values are numpy arrays (may be empty).
    """
    result: dict = {}
    column_map = _candidate_column_map(conn)

    # Kiel: prefer StarHorse teff50/logg50 from payload, fall back to GSP-Phot columns
    teff_gsp_expr = _background_value_expr("teff_gspphot", column_map)
    logg_gsp_expr = _background_value_expr("logg_gspphot", column_map)
    teff50_expr = _background_value_expr("teff50", column_map)
    logg50_expr = _background_value_expr("logg50", column_map)
    rows = conn.execute(
        f"SELECT {teff_gsp_expr}, {logg_gsp_expr}, {teff50_expr}, {logg50_expr} "
        "FROM candidates"
    ).fetchall()
    teff_list, logg_list = [], []
    for gsp_t, gsp_g, sh_t, sh_g in rows:
        t = sh_t if sh_t is not None else gsp_t
        g = sh_g if sh_g is not None else gsp_g
        if t is not None and g is not None:
            try:
                tf, gf = float(t), float(g)
                if math.isfinite(tf) and math.isfinite(gf):
                    teff_list.append(tf)
                    logg_list.append(gf)
            except (TypeError, ValueError):
                pass
    result["kiel_teff"] = np.array(teff_list, dtype=np.float64)
    result["kiel_logg"] = np.array(logg_list, dtype=np.float64)

    # Teff-alpha: use the same StarHorse -> GSP-Phot Teff precedence as Kiel.
    sed_alpha_expr = _background_value_expr("sed_alpha", column_map)
    rows = conn.execute(
        f"SELECT {teff_gsp_expr}, {teff50_expr}, {sed_alpha_expr} "
        "FROM candidates"
    ).fetchall()
    teff_alpha_t, teff_alpha_a = [], []
    for gsp_t, sh_t, alpha in rows:
        t = sh_t if sh_t is not None else gsp_t
        if t is None or alpha is None:
            continue
        try:
            tf, af = float(t), float(alpha)
            if math.isfinite(tf) and math.isfinite(af):
                teff_alpha_t.append(tf)
                teff_alpha_a.append(af)
        except (TypeError, ValueError):
            pass
    result["plane_teff_alpha_x"] = np.array(teff_alpha_t, dtype=np.float64)
    result["plane_teff_alpha_y"] = np.array(teff_alpha_a, dtype=np.float64)

    # CMD: dustmaps3d extinction from A_v_3d + Gaia photometry.
    phot_g_expr = _background_value_expr("phot_g_mean_mag", column_map)
    bp_rp_expr = _background_value_expr("bp_rp", column_map)
    phot_bp_expr = _background_value_expr("phot_bp_mean_mag", column_map)
    phot_rp_expr = _background_value_expr("phot_rp_mean_mag", column_map)
    dist_expr = _background_value_expr("distance_gspphot", column_map)
    parallax_expr = _background_value_expr("parallax", column_map)
    av_expr = _background_value_expr("A_v_3d", column_map)
    rows = conn.execute(
        f"SELECT {phot_g_expr}, {bp_rp_expr}, "
        f"{phot_bp_expr}, {phot_rp_expr}, {dist_expr}, {parallax_expr}, {av_expr} "
        "FROM candidates"
    ).fetchall()
    cmd_bprp0: list[float] = []
    cmd_mg0: list[float] = []
    for g_mag, bp_rp, bp_mag, rp_mag, dist, plx, av in rows:
        coords = dustmaps_cmd_from_fields(
            g_mag=g_mag,
            bp_rp=bp_rp,
            dist_pc=dist,
            a_v_3d=av,
            bp_mag=bp_mag,
            rp_mag=rp_mag,
            parallax_mas=plx,
        )
        if coords["cmd_coordinate_source"] == "missing":
            continue
        try:
            bprp0_f = float(coords["cmd_color"])
            mg0_f = float(coords["cmd_mag"])
            if math.isfinite(bprp0_f) and math.isfinite(mg0_f):
                cmd_mg0.append(mg0_f)
                cmd_bprp0.append(bprp0_f)
        except (TypeError, ValueError):
            pass
    result["cmd_bprp0"] = np.array(cmd_bprp0, dtype=np.float64)
    result["cmd_mg0"] = np.array(cmd_mg0, dtype=np.float64)

    # IR color-color: prefer dereddened from payload, fall back to observed
    h_expr = _background_value_expr("tmass_h", column_map)
    k_expr = _background_value_expr("tmass_k", column_map)
    w1_expr = _background_value_expr("w1", column_map)
    w2_expr = _background_value_expr("w2", column_map)
    hk_expr = _background_value_expr("H_K", column_map)
    w1w2_expr = _background_value_expr("w1_w2", column_map, aliases=("W1_W2",))
    hk_dered_expr = _background_value_expr("H_K_dered", column_map)
    w1w2_dered_expr = _background_value_expr("w1_w2_dered", column_map, aliases=("W1_W2_dered",))
    rows = conn.execute(
        f"SELECT {h_expr}, {k_expr}, {w1_expr}, {w2_expr}, "
        f"{hk_expr}, {w1w2_expr}, {hk_dered_expr}, {w1w2_dered_expr} "
        "FROM candidates"
    ).fetchall()
    hk_list, w1w2_list = [], []
    for h, k, w1, w2, hk_obs_col, w1w2_obs_col, hk_d, w1w2_d in rows:
        try:
            hk = hk_d if hk_d is not None else hk_obs_col
            w1w2 = w1w2_d if w1w2_d is not None else w1w2_obs_col
            if hk is None and h is not None and k is not None:
                hk = float(h) - float(k)
            if w1w2 is None and w1 is not None and w2 is not None:
                w1w2 = float(w1) - float(w2)
            if hk is None or w1w2 is None:
                continue
            hkf, wf = float(hk), float(w1w2)
            if math.isfinite(hkf) and math.isfinite(wf):
                hk_list.append(hkf)
                w1w2_list.append(wf)
        except (TypeError, ValueError):
            pass
    result["ir_hk"] = np.array(hk_list, dtype=np.float64)
    result["ir_w1w2"] = np.array(w1w2_list, dtype=np.float64)

    # RPM: H_G = G + 5*log10(pm_arcsec) + 5
    pmra_expr = _background_value_expr("pmra", column_map)
    pmdec_expr = _background_value_expr("pmdec", column_map)
    rows = conn.execute(
        f"SELECT {phot_g_expr}, {bp_rp_expr}, {phot_bp_expr}, {phot_rp_expr}, {pmra_expr}, {pmdec_expr} "
        "FROM candidates"
    ).fetchall()
    rpm_bprp_list, rpm_hg_list = [], []
    for g_mag, bprp, bp_mag, rp_mag, pmra, pmdec in rows:
        try:
            g_f = float(g_mag)
            bprp_f = float(bprp) if bprp is not None else float(bp_mag) - float(rp_mag)
            pm_total = math.sqrt(float(pmra) ** 2 + float(pmdec) ** 2)
            if pm_total > 0 and math.isfinite(g_f) and math.isfinite(bprp_f):
                pm_arcsec = pm_total / 1000.0
                h_g = g_f + 5.0 * math.log10(pm_arcsec) + 5.0
                rpm_bprp_list.append(bprp_f)
                rpm_hg_list.append(h_g)
        except (TypeError, ValueError):
            pass
    result["rpm_bprp"] = np.array(rpm_bprp_list, dtype=np.float64)
    result["rpm_hg"] = np.array(rpm_hg_list, dtype=np.float64)

    # UV-Optical: NUV - G vs BP-RP
    nuv_expr = _background_value_expr("galex_nuv", column_map)
    rows = conn.execute(
        f"SELECT {nuv_expr}, {phot_g_expr}, {bp_rp_expr}, {phot_bp_expr}, {phot_rp_expr} "
        "FROM candidates"
    ).fetchall()
    uv_bprp_list, uv_nuv_g_list = [], []
    for nuv, g_mag, bprp, bp_mag, rp_mag in rows:
        try:
            nuv_f = float(nuv)
            g_f = float(g_mag)
            bprp_f = float(bprp) if bprp is not None else float(bp_mag) - float(rp_mag)
            nuv_g = nuv_f - g_f
            if math.isfinite(nuv_g) and math.isfinite(bprp_f):
                uv_nuv_g_list.append(nuv_g)
                uv_bprp_list.append(bprp_f)
        except (TypeError, ValueError):
            pass
    result["uv_nuv_g"] = np.array(uv_nuv_g_list, dtype=np.float64)
    result["uv_bprp"] = np.array(uv_bprp_list, dtype=np.float64)

    pair_specs = (
        ("metric_periodicity_score", "metric_phase_quality_score", "periodicity_score", "phase_quality_score"),
        ("metric_dipper_score", "metric_jumper_score", "dipper_score", "jumper_score"),
        ("plane_catalog_support_x", "plane_catalog_support_y", "period_n_sources", "dip_run_count"),
        ("plane_recurrence_regularity_x", "plane_recurrence_regularity_y", "dip_inter_event_spacing_median", "dip_inter_event_spacing_std"),
        ("plane_dip_repeatability_x", "plane_dip_repeatability_y", "dip_amplitude_consistency", "dip_duration_consistency"),
        ("plane_var_strength_x", "plane_var_strength_y", "stats_photometry_robust_sigma_mag", "dipper_score"),
        ("plane_stetson_x", "plane_stetson_y", "stats_photometry_robust_sigma_mag", "stats_variability_stetson_J"),
        ("plane_shape_x", "plane_shape_y", "stats_skew", "stats_max_slope"),
        ("plane_harmonic_x", "plane_harmonic_y", "stats_harmonics_model_amplitude", "stats_harmonics_reduced_chi2"),
        ("plane_autocorr_x", "plane_autocorr_y", "stats_variability_lag1_autocorr", "stats_autocor_length"),
        ("plane_cluster_x", "plane_cluster_y", "pm_cluster_offset_sigma", "ruwe"),
        ("plane_classifier_x", "plane_classifier_y", "P_disk", "P_eb"),
        ("plane_atlas_x", "plane_atlas_y", "atlas_cyan_range", "atlas_orange_range"),
        ("plane_ztf_x", "plane_ztf_y", "ztf_lc_g_range", "ztf_lc_r_range"),
        ("plane_neowise_range_x", "plane_neowise_range_y", "neowise_w1_range", "neowise_w2_range"),
        ("plane_gaia_epoch_x", "plane_gaia_epoch_y", "gaia_epoch_n_obs", "gaia_epoch_g_range"),
        ("plane_ltv_x", "plane_ltv_y", "ltv_slope", "ltv_dispersion"),
        ("plane_neowise_trend_x", "plane_neowise_trend_y", "ltv_neowise_w1_slope", "ltv_neowise_w1_w2_slope"),
    )
    for x_key, y_key, x_expr, y_expr in pair_specs:
        result[x_key], result[y_key] = _load_background_pair(conn, x_expr, y_expr, column_map)

    return result


VETTING_COLUMNS = [
    "vetting_likely_known",
    "microlens_match", "microlens_catalog", "microlens_name",
    "microlens_alt_name", "microlens_te_days", "microlens_sep_arcsec",
    "simbad_main_id", "simbad_otype", "simbad_nbref", "simbad_sep_arcsec",
    "gaia_var_flag", "gaia_var_class", "gaia_var_score",
    "gaia_eb_period", "gaia_eb_morph", "gaia_eb_global_ranking",
    "gaia_epoch_available", "gaia_epoch_n_obs", "gaia_epoch_g_range",
    "asassn_var_name", "asassn_var_type", "asassn_var_period",
    "ztf_var_type", "ztf_var_period", "ztf_var_amp",
    "tns_name", "tns_type", "tns_redshift", "tns_disc_date",
    "alerce_oid", "alerce_ndet", "alerce_lc_class", "alerce_lc_prob",
    "alerce_stamp_class", "alerce_stamp_prob",
    "erosita_det", "erosita_flux", "erosita_sep_arcsec",
    "chandra_det", "chandra_source_id", "chandra_flux_05_7",
    "chandra_flux_broad", "chandra_significance", "chandra_likelihood",
    "chandra_likelihood_class", "chandra_pos_err_maj_arcsec",
    "chandra_pos_err_min_arcsec", "chandra_pos_err_pa_deg",
    "chandra_extended_flag", "chandra_variable_flag", "chandra_sep_arcsec",
    "xray_det", "xray_flux", "xray_sep_arcsec", "xray_source_catalogs",
    "vsx_class", "vsx_sep_arcsec", "vsx_period",
    "nearby_vsx_dipper_contaminant", "nearby_vsx_dipper_class",
    "nearby_vsx_dipper_sep_arcsec", "nearby_vsx_dipper_period",
    "sfr_name", "sfr_sep_arcmin",
    *OPEN_CLUSTER_REVIEW_COLUMNS,
    "banyan_best_assoc", "banyan_field_prob", "banyan_ya_prob",
    "banyan_best_assoc_prob", "banyan_probabilities_json", "banyan_input_mode",
    "banyan_status", "banyan_error", "banyan_version", "banyan_adapter_version",
    "banyan_updated_at",
    "yso_class",
    "iphas_r_mag", "iphas_r_err", "iphas_i_mag", "iphas_i_err",
    "iphas_ha_mag", "iphas_ha_err", "iphas_r_i", "iphas_r_i_err",
    "iphas_r_ha", "iphas_r_ha_err", "iphas_sep_arcsec", "iphas_source_catalog",
    "iphas_ha_excess",
    "vphas_r_mag", "vphas_r_err", "vphas_i_mag", "vphas_i_err",
    "vphas_ha_mag", "vphas_ha_err", "vphas_r_i", "vphas_r_i_err",
    "vphas_r_ha", "vphas_r_ha_err", "vphas_sep_arcsec", "vphas_source_catalog",
    "vphas_ha_excess",
    "pm_cluster_offset_sigma",
    "atlas_has_phot", "atlas_n_det_cyan", "atlas_n_det_orange",
    "atlas_cyan_range", "atlas_orange_range", "atlas_preprocess_version",
    "atlas_n_raw", "atlas_n_good", "atlas_n_rejected",
    "neowise_n_epochs", "neowise_w1_range", "neowise_w2_range",
    "kepler_n_quarters", "kepler_total_points", "kepler_flux_range",
    "aavso_lc_n_points",
    "ogle_lc_n_points", "ogle_lc_i_range", "ogle_lc_v_range",
    "stripe82_lc_n_points", "stripe82_lc_u_range", "stripe82_lc_g_range",
    "stripe82_lc_r_range", "stripe82_lc_i_range", "stripe82_lc_z_range",
    "allwise_mep_n_epochs", "allwise_mep_w1_range", "allwise_mep_w2_range",
    "allwise_mep_w3_range", "allwise_mep_w4_range",
    "vvvx_virac_n_epochs", "vvvx_virac_z_range", "vvvx_virac_y_range",
    "vvvx_virac_j_range", "vvvx_virac_h_range", "vvvx_virac_ks_range",
]


def merge_vetting_results(
    conn: sqlite3.Connection,
    vetting_df: pd.DataFrame,
    id_column: str | None = None,
) -> int:
    """Merge vetting results into existing candidate payload_json.

    Matches candidates by candidate_id or asas_sn_id. Updates only
    vetting-related columns in the payload, preserving all other data.

    Returns number of candidates updated.
    """
    if vetting_df.empty:
        return 0

    # Determine ID column
    if id_column is None:
        for col in ("candidate_id", "asas_sn_id"):
            if col in vetting_df.columns:
                id_column = col
                break
    if id_column is None:
        raise ValueError("Vetting DataFrame must have 'candidate_id' or 'asas_sn_id' column")

    # Build lookup: id -> vetting dict
    vetting_cols = [c for c in VETTING_COLUMNS if c in vetting_df.columns]
    if not vetting_cols:
        print("Warning: no vetting columns found in DataFrame")
        return 0

    vetting_df = vetting_df.copy()
    vetting_df[id_column] = vetting_df[id_column].astype(str).str.strip()
    lookup = {}
    for _, row in vetting_df.iterrows():
        cid = row[id_column]
        d = {}
        for col in vetting_cols:
            val = row[col]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                d[col] = val
        d = normalize_vsx_record(d)
        if has_known_catalog_evidence(d):
            d["vetting_likely_known"] = True
        elif has_catalog_vetting_context(d):
            d["vetting_likely_known"] = False
        if d:
            lookup[cid] = d

    if not lookup:
        return 0

    # Fetch all candidates and update payloads
    rows = conn.execute("SELECT candidate_id, asas_sn_id FROM candidates").fetchall()
    updated = 0
    for cid, asas_sn_id in rows:
        cid_str = str(cid).strip()
        vetting_data = lookup.get(cid_str)
        if vetting_data is None and asas_sn_id is not None:
            vetting_data = lookup.get(str(asas_sn_id).strip())
        if vetting_data is None:
            continue

        replace_candidate_payload_fields(
            conn,
            str(cid),
            vetting_data,
            commit=False,
        )
        updated += 1

    conn.commit()
    print(f"Merged vetting data for {updated}/{len(rows)} candidates ({len(vetting_cols)} columns)")
    return updated


def merge_candidate_results(
    conn: sqlite3.Connection,
    candidate_df: pd.DataFrame,
    id_column: str | None = None,
    *,
    clear_columns: Iterable[str] = (),
) -> int:
    """Merge candidate-table columns into existing review candidates.

    Matches by ``candidate_id`` or ``asas_sn_id``. Only candidate payload/SQL
    columns are updated; review tables are untouched.

    Non-null values present in ``candidate_df`` update pipeline-owned fields.
    Null values are sparse/no-op by default. Callers must explicitly name
    fields in ``clear_columns`` when a refresh intends to remove stale values.
    """
    if candidate_df.empty:
        return 0

    if id_column is None:
        for col in ("candidate_id", "asas_sn_id"):
            if col in candidate_df.columns:
                id_column = col
                break
    if id_column is None:
        raise ValueError("Candidate DataFrame must have 'candidate_id' or 'asas_sn_id' column")

    from malca.products.candidates import validate_candidate_ids
    from malca.products.feature_layers import expand_feature_layers

    candidate_df = expand_feature_layers(candidate_df)
    ignored_cols = {
        "candidate_id",
        "source_path",
        "payload_json",
        "imported_at",
        FEATURE_LAYER_VERSION_COLUMN,
        *FEATURE_LAYER_COLUMNS,
    }
    merge_cols = [c for c in candidate_df.columns if c not in ignored_cols]
    if not merge_cols:
        print("Warning: no candidate columns found in DataFrame")
        return 0

    candidate_df = candidate_df.copy()
    if id_column == "candidate_id":
        candidate_df[id_column] = validate_candidate_ids(
            candidate_df,
            key_col=id_column,
            require_unique=True,
        )
    else:
        candidate_df[id_column] = candidate_df[id_column].astype("string").str.strip()
        invalid = candidate_df[id_column].isna() | candidate_df[id_column].eq("")
        if bool(invalid.any()):
            raise ValueError(f"{id_column} contains blank/null match values")
        if bool(candidate_df[id_column].duplicated().any()):
            raise ValueError(f"{id_column} contains duplicate match values")

    requested_clears = {str(column) for column in clear_columns}
    protected_clears = requested_clears & {
        "candidate_id", "asas_sn_id", "source_path", "payload_json", "imported_at"
    }
    if protected_clears:
        raise ValueError(
            "clear_columns cannot clear candidate identity/provenance fields: "
            + ", ".join(sorted(protected_clears))
        )
    unknown_clears = requested_clears - set(merge_cols)
    if unknown_clears:
        raise ValueError(
            "clear_columns must also be present in the incoming product: "
            + ", ".join(sorted(unknown_clears))
        )

    rows = conn.execute("SELECT candidate_id, asas_sn_id FROM candidates").fetchall()
    candidate_ids: set[str] = set()
    asas_to_candidate: dict[str, str] = {}
    ambiguous_asas: set[str] = set()
    for raw_candidate_id, raw_asas_sn_id in rows:
        candidate_id = str(raw_candidate_id).strip()
        if candidate_id:
            candidate_ids.add(candidate_id)

        asas_sn_id = "" if raw_asas_sn_id is None else str(raw_asas_sn_id).strip()
        if not asas_sn_id:
            continue
        existing = asas_to_candidate.get(asas_sn_id)
        if existing is None:
            asas_to_candidate[asas_sn_id] = candidate_id
        elif existing != candidate_id:
            ambiguous_asas.add(asas_sn_id)
    for asas_sn_id in ambiguous_asas:
        asas_to_candidate.pop(asas_sn_id, None)

    updated = 0
    for _, row in candidate_df.iterrows():
        raw_match = str(row[id_column]).strip()
        if not raw_match:
            continue
        if id_column == "candidate_id":
            matched_candidate_id = raw_match if raw_match in candidate_ids else None
        else:
            matched_candidate_id = asas_to_candidate.get(raw_match)
        if not matched_candidate_id:
            continue

        clear_keys = set(requested_clears)
        updates: dict[str, object] = {}
        for col in merge_cols:
            value = row[col]
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            updates[col] = value
        updates = normalize_vsx_record(updates)
        if has_known_catalog_evidence(updates):
            updates["vetting_likely_known"] = True

        if not updates and not clear_keys:
            continue

        replace_candidate_payload_fields(
            conn,
            matched_candidate_id,
            updates,
            clear_keys=clear_keys,
            commit=False,
        )
        updated += 1

    conn.commit()
    print(f"Merged candidate data for {updated}/{len(rows)} candidates ({len(merge_cols)} columns)")
    return updated


def refresh_dipper_recurrence_classifications(conn: sqlite3.Connection) -> int:
    """Refresh deterministic observed dip-recurrence fields for all candidates.

    This is intentionally separate from ML-score merging: recurrence is a
    direct summary of the triggered-dip extraction, not a probability emitted
    by either the binary dipper ranker or the eight-class model.
    """

    required = {
        "candidate_id",
        "dip_run_count",
        "dip_is_single_event",
        DIPPER_RECURRENCE_CLASS_COLUMN,
        DIPPER_RECURRENCE_EVIDENCE_COLUMN,
    }
    actual = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    missing = sorted(required.difference(actual))
    if missing:
        raise RuntimeError(
            "Review DB is missing recurrence columns; run ensure_review_db_schema first: "
            f"{missing}"
        )

    source = pd.read_sql_query(
        "SELECT candidate_id, dip_run_count, dip_is_single_event FROM candidates",
        conn,
    )
    classified = add_observed_dipper_recurrence(source)
    temp_table = "temp_dipper_recurrence_classes"
    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    try:
        conn.execute(
            f"""
            CREATE TEMP TABLE {temp_table} (
                candidate_id TEXT PRIMARY KEY,
                {DIPPER_RECURRENCE_CLASS_COLUMN} TEXT NOT NULL,
                {DIPPER_RECURRENCE_EVIDENCE_COLUMN} TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            f"""
            INSERT INTO {temp_table} (
                candidate_id,
                {DIPPER_RECURRENCE_CLASS_COLUMN},
                {DIPPER_RECURRENCE_EVIDENCE_COLUMN}
            ) VALUES (?, ?, ?)
            """,
            [
                (
                    str(row.candidate_id),
                    str(getattr(row, DIPPER_RECURRENCE_CLASS_COLUMN)),
                    str(getattr(row, DIPPER_RECURRENCE_EVIDENCE_COLUMN)),
                )
                for row in classified.itertuples(index=False)
            ],
        )
        changed = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM candidates AS c
                JOIN {temp_table} AS t USING (candidate_id)
                WHERE c.{DIPPER_RECURRENCE_CLASS_COLUMN}
                    IS NOT t.{DIPPER_RECURRENCE_CLASS_COLUMN}
                   OR c.{DIPPER_RECURRENCE_EVIDENCE_COLUMN}
                    IS NOT t.{DIPPER_RECURRENCE_EVIDENCE_COLUMN}
                """
            ).fetchone()[0]
        )
        if changed:
            conn.execute(
                f"""
                UPDATE candidates
                SET {DIPPER_RECURRENCE_CLASS_COLUMN} = (
                        SELECT t.{DIPPER_RECURRENCE_CLASS_COLUMN}
                        FROM {temp_table} AS t
                        WHERE t.candidate_id = candidates.candidate_id
                    ),
                    {DIPPER_RECURRENCE_EVIDENCE_COLUMN} = (
                        SELECT t.{DIPPER_RECURRENCE_EVIDENCE_COLUMN}
                        FROM {temp_table} AS t
                        WHERE t.candidate_id = candidates.candidate_id
                    )
                WHERE candidate_id IN (SELECT candidate_id FROM {temp_table})
                  AND (
                    {DIPPER_RECURRENCE_CLASS_COLUMN} IS NOT (
                        SELECT t.{DIPPER_RECURRENCE_CLASS_COLUMN}
                        FROM {temp_table} AS t
                        WHERE t.candidate_id = candidates.candidate_id
                    )
                    OR {DIPPER_RECURRENCE_EVIDENCE_COLUMN} IS NOT (
                        SELECT t.{DIPPER_RECURRENCE_EVIDENCE_COLUMN}
                        FROM {temp_table} AS t
                        WHERE t.candidate_id = candidates.candidate_id
                    )
                  )
                """
            )
    except Exception:
        conn.rollback()
        raise
    else:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.commit()

    print(f"Updated dip-recurrence class for {changed} candidate(s)")
    return changed


def merge_model_probability_scores(
    conn: sqlite3.Connection,
    scores_path: str | Path,
    *,
    probability_columns: Iterable[str],
    require_complete_candidate_coverage: bool = False,
) -> int:
    """Merge one or more bounded ML score columns into review candidates.

    ``require_complete_candidate_coverage`` is for run-level model products:
    it rejects a score table whose candidate IDs do not exactly match the
    review database, preventing a partially scored or wrong-run artifact from
    silently changing a live review queue.
    """
    path = Path(scores_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ML score file not found: {path}")

    columns = tuple(dict.fromkeys(str(column).strip() for column in probability_columns))
    if not columns or any(not column for column in columns):
        raise ValueError("At least one non-empty ML score column is required")
    unknown_columns = set(columns).difference(_COL_NAMES)
    if unknown_columns:
        raise ValueError(f"Unknown candidate score column(s): {sorted(unknown_columns)}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        scores = pd.read_csv(path)
    else:
        scores = read_parquet_table(path)

    missing_columns = [column for column in columns if column not in scores.columns]
    if missing_columns:
        raise ValueError(f"Score file is missing required column(s): {missing_columns}")
    if "candidate_id" not in scores.columns and "asas_sn_id" not in scores.columns:
        raise ValueError("Score file must include candidate_id or asas_sn_id")

    id_column = "candidate_id" if "candidate_id" in scores.columns else "asas_sn_id"
    merge_frame = scores[[id_column, *columns]].copy()
    merge_frame[id_column] = merge_frame[id_column].astype("string").str.strip()
    for column in columns:
        merge_frame[column] = pd.to_numeric(
            merge_frame[column], errors="coerce"
        ).clip(lower=0.0, upper=1.0)

    invalid_rows = merge_frame[id_column].isna() | merge_frame[id_column].eq("")
    invalid_rows |= merge_frame[list(columns)].isna().any(axis=1)
    if require_complete_candidate_coverage and bool(invalid_rows.any()):
        raise ValueError(
            "Complete score coverage requires a finite ID and every requested "
            f"score; found {int(invalid_rows.sum())} invalid row(s) in {path}"
        )
    if require_complete_candidate_coverage and bool(
        merge_frame[id_column].duplicated(keep=False).any()
    ):
        duplicates = int(merge_frame[id_column].duplicated(keep=False).sum())
        raise ValueError(f"Complete score coverage rejects {duplicates} duplicate IDs in {path}")

    merge_frame = merge_frame.dropna(subset=[id_column, *columns])
    merge_frame = merge_frame.loc[merge_frame[id_column].ne("")]
    merge_frame = merge_frame.drop_duplicates(subset=[id_column], keep="last")
    if merge_frame.empty:
        return 0

    if require_complete_candidate_coverage:
        if id_column != "candidate_id":
            raise ValueError("Complete score coverage requires a candidate_id column")
        artifact_ids = set(merge_frame[id_column].astype(str))
        db_ids = {
            str(row[0])
            for row in conn.execute("SELECT candidate_id FROM candidates").fetchall()
        }
        missing = db_ids.difference(artifact_ids)
        unexpected = artifact_ids.difference(db_ids)
        if missing or unexpected:
            raise ValueError(
                "Score artifact does not match the review DB candidate set: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    temp_table = "temp_ml_probability_scores"
    quoted_columns = tuple(_quote_sql_identifier(column) for column in columns)
    column_definitions = ", ".join(f"{column} REAL NOT NULL" for column in quoted_columns)
    insert_columns = ", ".join(("match_id", *quoted_columns))
    placeholders = ", ".join("?" for _ in range(len(columns) + 1))
    changes = " OR ".join(
        f"c.{column} IS NOT s.{column}" for column in quoted_columns
    )
    target_id = "candidate_id" if id_column == "candidate_id" else "asas_sn_id"
    unique_asassn_clause = (
        ""
        if id_column == "candidate_id"
        else """
        AND (
            SELECT COUNT(*) FROM candidates c2
            WHERE c2.asas_sn_id = c.asas_sn_id
        ) = 1
        """
    )
    assignments = ", ".join(
        f"{column} = (SELECT s.{column} FROM {temp_table} s "
        f"WHERE s.match_id = candidates.{target_id})"
        for column in quoted_columns
    )
    update_uniqueness_clause = (
        ""
        if id_column == "candidate_id"
        else """
        AND (
            SELECT COUNT(*) FROM candidates c2
            WHERE c2.asas_sn_id = candidates.asas_sn_id
        ) = 1
        """
    )
    changed_subquery = " OR ".join(
        f"candidates.{column} IS NOT (SELECT s.{column} FROM {temp_table} s "
        f"WHERE s.match_id = candidates.{target_id})"
        for column in quoted_columns
    )

    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    try:
        conn.execute(
            f"CREATE TEMP TABLE {temp_table} (match_id TEXT PRIMARY KEY, {column_definitions})"
        )
        conn.executemany(
            f"INSERT INTO {temp_table} ({insert_columns}) VALUES ({placeholders})",
            [
                (str(row[id_column]), *(float(row[column]) for column in columns))
                for _, row in merge_frame.iterrows()
            ],
        )
        updated = int(conn.execute(
            f"""
            SELECT COUNT(*)
            FROM candidates c
            JOIN {temp_table} s ON s.match_id = c.{target_id}
            WHERE ({changes})
            {unique_asassn_clause}
            """
        ).fetchone()[0])
        if updated:
            conn.execute(
                f"""
                UPDATE candidates
                SET {assignments}
                WHERE {target_id} IN (SELECT match_id FROM {temp_table})
                {update_uniqueness_clause}
                  AND ({changed_subquery})
                """
            )
    except Exception:
        conn.rollback()
        raise
    else:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.commit()

    print(f"Updated {len(columns)} ML score column(s) for {updated} candidate(s)")
    return updated


def merge_dipper_probability_scores(
    conn: sqlite3.Connection,
    scores_path: str | Path,
    *,
    probability_column: str = "prob_dipper_like",
) -> int:
    """Merge a legacy binary ML dipper score into review candidates."""
    return merge_model_probability_scores(
        conn,
        scores_path,
        probability_columns=(probability_column,),
    )


def merge_nine_class_probability_scores(
    conn: sqlite3.Connection,
    scores_path: str | Path,
) -> int:
    """Merge a complete nine-class score artifact into the matching review DB."""
    return merge_model_probability_scores(
        conn,
        scores_path,
        probability_columns=NINE_CLASS_PROBABILITY_COLUMNS,
        require_complete_candidate_coverage=True,
    )


def merge_eight_class_probability_scores(
    conn: sqlite3.Connection,
    scores_path: str | Path,
) -> int:
    """Merge a complete eight-class score artifact into the matching Review DB."""
    return merge_model_probability_scores(
        conn,
        scores_path,
        probability_columns=EIGHT_CLASS_PROBABILITY_COLUMNS,
        require_complete_candidate_coverage=True,
    )


def merge_hierarchical_ml_scores(
    conn: sqlite3.Connection,
    scores_path: str | Path,
) -> int:
    """Merge a complete hierarchical score artifact into the Review DB."""

    path = Path(scores_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Hierarchical score file not found: {path}")
    scores = (
        pd.read_csv(path)
        if path.suffix.lower() == ".csv"
        else read_parquet_table(path)
    )
    required = {
        "candidate_id",
        *HIERARCHICAL_ML_PROBABILITY_COLUMNS,
        *HIERARCHICAL_ML_PREDICTION_COLUMNS,
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(
            f"Hierarchical score file is missing required columns: {missing}"
        )
    numeric_updated = merge_model_probability_scores(
        conn,
        path,
        probability_columns=HIERARCHICAL_ML_PROBABILITY_COLUMNS,
        require_complete_candidate_coverage=True,
    )
    incoming_predictions = scores[
        ["candidate_id", *HIERARCHICAL_ML_PREDICTION_COLUMNS]
    ].copy()
    for column in HIERARCHICAL_ML_PREDICTION_COLUMNS:
        invalid = (
            incoming_predictions[column].isna()
            | incoming_predictions[column].astype("string").str.strip().eq("")
        )
        if bool(invalid.any()):
            raise ValueError(
                f"Hierarchical score file has blank/null predictions in {column}"
            )
    current_predictions = pd.read_sql_query(
        "SELECT candidate_id, "
        + ", ".join(HIERARCHICAL_ML_PREDICTION_COLUMNS)
        + " FROM candidates",
        conn,
    )
    comparison = incoming_predictions.merge(
        current_predictions,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__current"),
    )
    changed = pd.Series(False, index=comparison.index, dtype=bool)
    for column in HIERARCHICAL_ML_PREDICTION_COLUMNS:
        changed |= (
            comparison[column].astype("string").fillna("<NULL>")
            != comparison[f"{column}__current"].astype("string").fillna("<NULL>")
        )
    prediction_updated = merge_candidate_results(
        conn,
        comparison.loc[
            changed, ["candidate_id", *HIERARCHICAL_ML_PREDICTION_COLUMNS]
        ],
    )
    return max(numeric_updated, prediction_updated)


def merge_dipper_recurrence_ml_scores(
    conn: sqlite3.Connection,
    scores_path: str | Path,
) -> int:
    """Merge conditional dipper-recurrence ML predictions into Review.

    The same conditional recurrence head is attached to the binary and active
    eight-class parent rankers, so each overlay supplies the shared conditional
    score plus the corresponding parent-gated recurrent-dipper score. Legacy
    nine-class overlays remain readable.
    """

    path = Path(scores_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dipper-recurrence score file not found: {path}")
    scores = pd.read_csv(path) if path.suffix.lower() == ".csv" else read_parquet_table(path)
    required = {"candidate_id", "prob_recurrent_given_dipper", "predicted_dipper_recurrence"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Dipper-recurrence score file is missing required columns: {missing}")
    available_probabilities = tuple(
        column for column in DIPPER_RECURRENCE_ML_PROBABILITY_COLUMNS if column in scores.columns
    )
    if len(available_probabilities) < 2:
        raise ValueError(
            "Dipper-recurrence overlay must include the conditional score and "
            "one parent-gated recurrent-dipper score"
        )
    numeric_updated = merge_model_probability_scores(
        conn,
        path,
        probability_columns=available_probabilities,
        require_complete_candidate_coverage=True,
    )

    classes = scores[["candidate_id", "predicted_dipper_recurrence"]].copy()
    classes["candidate_id"] = classes["candidate_id"].astype("string").str.strip()
    classes["predicted_dipper_recurrence"] = (
        classes["predicted_dipper_recurrence"].astype("string").str.strip()
    )
    valid_classes = {"recurrent", "non_recurrent"}
    invalid = (
        classes["candidate_id"].isna()
        | classes["candidate_id"].eq("")
        | ~classes["predicted_dipper_recurrence"].isin(valid_classes)
        | classes["candidate_id"].duplicated(keep=False)
    )
    if bool(invalid.any()):
        raise ValueError(
            "Dipper-recurrence overlay requires one valid recurrent/non_recurrent "
            f"prediction per candidate; found {int(invalid.sum())} invalid row(s)"
        )
    artifact_ids = set(classes["candidate_id"].astype(str))
    db_ids = {
        str(row[0]) for row in conn.execute("SELECT candidate_id FROM candidates").fetchall()
    }
    if artifact_ids != db_ids:
        raise ValueError(
            "Dipper-recurrence overlay does not match the review DB candidate set: "
            f"missing={len(db_ids.difference(artifact_ids))}, "
            f"unexpected={len(artifact_ids.difference(db_ids))}"
        )

    temp_table = "temp_dipper_recurrence_ml_classes"
    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    try:
        conn.execute(
            f"""
            CREATE TEMP TABLE {temp_table} (
                candidate_id TEXT PRIMARY KEY,
                predicted_dipper_recurrence TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            f"INSERT INTO {temp_table} (candidate_id, predicted_dipper_recurrence) VALUES (?, ?)",
            [
                (str(row.candidate_id), str(row.predicted_dipper_recurrence))
                for row in classes.itertuples(index=False)
            ],
        )
        class_updated = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM candidates AS c
                JOIN {temp_table} AS t USING (candidate_id)
                WHERE c.predicted_dipper_recurrence IS NOT t.predicted_dipper_recurrence
                """
            ).fetchone()[0]
        )
        if class_updated:
            conn.execute(
                f"""
                UPDATE candidates
                SET predicted_dipper_recurrence = (
                    SELECT t.predicted_dipper_recurrence
                    FROM {temp_table} AS t
                    WHERE t.candidate_id = candidates.candidate_id
                )
                WHERE candidate_id IN (SELECT candidate_id FROM {temp_table})
                  AND predicted_dipper_recurrence IS NOT (
                    SELECT t.predicted_dipper_recurrence
                    FROM {temp_table} AS t
                    WHERE t.candidate_id = candidates.candidate_id
                  )
                """
            )
    except Exception:
        conn.rollback()
        raise
    else:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.commit()

    updated = max(numeric_updated, class_updated)
    print(f"Updated dipper-recurrence ML fields for {updated} candidate(s)")
    return updated


def get_review(conn: sqlite3.Connection, candidate_id: str) -> dict:
    taxonomy_cols = ", ".join(REVIEW_TAXONOMY_FIELDS)
    row = conn.execute(
        f"""
        SELECT review_pass, notes, status, reviewer, updated_at, event_class,
               {taxonomy_cols}
        FROM reviews WHERE candidate_id=?
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        base = {
            "event_class": "unclassified",
            "review_pass": 1,
            "notes": "",
            "status": "unreviewed",
            "workflow_status": "unreviewed",
            "disposition": None,
            "reviewer": "",
            "updated_at": None,
        }
        base.update(empty_taxonomy_selection())
        base["workflow_status"] = "unreviewed"
        base["priority_tags_json"] = "[]"
        base["evidence_flags_json"] = "[]"
        base["model_tags_json"] = "[]"
        base["legacy_review_json"] = "{}"
        return base
    taxonomy_values = dict(zip(REVIEW_TAXONOMY_FIELDS, row[6:]))
    workflow_status = str(taxonomy_values.get("workflow_status") or "unreviewed")
    selection = selection_from_review(taxonomy_values)
    event_class = str(row[5]) if row[5] else derive_event_class(selection)
    out = {
        "event_class": event_class,
        "review_pass": 1 if row[0] is None else max(1, int(row[0])),
        "notes": "" if row[1] is None else str(row[1]),
        "status": workflow_status,
        "workflow_status": workflow_status,
        "reviewer": "" if row[3] is None else str(row[3]),
        "updated_at": row[4],
    }
    out.update(taxonomy_values)
    out.update(selection)
    out["priority_tags_json"] = json_list(out.get("priority_tags"))
    out["evidence_flags_json"] = json_list(out.get("evidence_flags"))
    out["model_tags_json"] = json_list(out.get("model_tags"))
    out["taxonomy_version"] = int(out.get("taxonomy_version") or TAXONOMY_VERSION)
    out["legacy_review_json"] = str(out.get("legacy_review_json") or "{}")
    return out


def save_review(
    conn: sqlite3.Connection,
    *,
    candidate_id: str,
    event_class: str = "unclassified",
    review_pass: int,
    notes: str,
    status: str | None = None,
    workflow_status: str | None = None,
    disposition: str | None = None,
    morphology_primary: str | None = None,
    morphology_secondary: str | None = None,
    morphology_secondary_json: Any = None,
    morphology_polarity: str | None = None,
    morphology_recurrence: str | None = None,
    baseline_behavior: str | None = None,
    physical_primary: str | None = None,
    physical_secondary: str | None = None,
    classification_confidence: int | None = None,
    priority_tags: Any = None,
    evidence_flags: Any = None,
    model_tags: Any = None,
    duplicate_of: str | None = None,
    known_object_id: str | None = None,
    known_object_source: str | None = None,
    legacy_review_json: str | None = None,
    reviewer: str = "",
    event_type: str = "save",
) -> None:
    ts = _utc_now()
    pass_int = max(1, int(review_pass))
    workflow = str(workflow_status or status or "reviewed")
    selection = normalize_selection(
        {
            "morphology_primary": morphology_primary,
            "morphology_secondary": morphology_secondary,
            "morphology_secondary_json": morphology_secondary_json,
            "morphology_polarity": morphology_polarity,
            "morphology_recurrence": morphology_recurrence,
            "baseline_behavior": baseline_behavior,
            "physical_primary": physical_primary,
            "physical_secondary": physical_secondary,
            "classification_confidence": classification_confidence,
            "priority_tags": priority_tags,
            "evidence_flags": evidence_flags,
            "model_tags": model_tags,
            "disposition": disposition,
            "duplicate_of": duplicate_of,
            "known_object_id": known_object_id,
            "known_object_source": known_object_source,
        }
    )
    ec = derive_event_class(selection)
    if ec == "unclassified" and event_class:
        ec = str(event_class)
    taxonomy_values = {
        "workflow_status": workflow,
        "disposition": selection.get("disposition"),
        "morphology_primary": selection.get("morphology_primary"),
        "morphology_secondary": selection.get("morphology_secondary"),
        "morphology_secondary_json": selection.get("morphology_secondary_json"),
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
        "taxonomy_version": TAXONOMY_VERSION,
        "legacy_review_json": legacy_review_json or "{}",
    }
    taxonomy_cols = list(REVIEW_TAXONOMY_FIELDS)
    insert_cols = [
        "candidate_id",
        "event_class",
        "review_pass",
        "notes",
        "status",
        "reviewer",
        *taxonomy_cols,
        "updated_at",
    ]
    placeholders = ", ".join(["?"] * len(insert_cols))
    conflict_cols = [col for col in insert_cols if col != "candidate_id"]
    conflict_set = ",\n            ".join(f"{col}=excluded.{col}" for col in conflict_cols)
    conn.execute(
        f"""
        INSERT INTO reviews ({', '.join(insert_cols)})
        VALUES ({placeholders})
        ON CONFLICT(candidate_id) DO UPDATE SET
            {conflict_set}
        """,
        (
            candidate_id,
            ec,
            pass_int,
            notes,
            workflow,
            reviewer,
            *(taxonomy_values[col] for col in taxonomy_cols),
            ts,
        ),
    )
    payload = {
        "event_class": ec,
        "review_pass": pass_int,
        "notes": notes,
        "status": workflow,
        **taxonomy_values,
        "reviewer": reviewer,
        "updated_at": ts,
    }
    conn.execute(
        """
        INSERT INTO review_history (candidate_id, event_type, payload_json, reviewer, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (candidate_id, event_type, json.dumps(payload, default=str), reviewer, ts),
    )
    conn.commit()


def recent_history(conn: sqlite3.Connection, limit: int = 5) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT candidate_id, event_type, reviewer, created_at
        FROM review_history
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=[int(limit)],
    )


def count_progress(conn: sqlite3.Connection) -> tuple[int, int]:
    total = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    reviewed = conn.execute("SELECT COUNT(*) FROM reviews WHERE workflow_status IS NOT NULL AND workflow_status != 'unreviewed'").fetchone()[0]
    return int(reviewed), int(total)


def find_plot_image(payload: dict, plot_dir: Path) -> Path | None:
    if not plot_dir.exists():
        return None
    keys = []
    for k in ("candidate_id", "asas_sn_id"):
        if k in payload and payload[k] is not None:
            keys.append(str(payload[k]))
    lc_path = payload.get("path")
    if lc_path:
        keys.append(Path(str(lc_path)).stem)
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    for key in keys:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.pdf"):
            matches = sorted(plot_dir.rglob(f"*{key}*{ext[1:]}"), key=lambda p: str(p))
            if not matches:
                continue
            non_phase = [p for p in matches if "phase" not in p.stem.lower()]
            return non_phase[0] if non_phase else matches[0]
    return None


def find_phase_plot_image(payload: dict, plot_dir: Path) -> Path | None:
    """Locate a phase-folded plot image for a candidate."""
    if not plot_dir.exists():
        return None
    keys = []
    for k in ("candidate_id", "asas_sn_id"):
        if k in payload and payload[k] is not None:
            keys.append(str(payload[k]))
    lc_path = payload.get("path")
    if lc_path:
        keys.append(Path(str(lc_path)).stem)
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    for key in keys:
        phase_patterns = (
            f"*{key}*candidate_phase*.png",
            f"*{key}*phase*.png",
            f"*{key}*candidate_phase*.jpg",
            f"*{key}*phase*.jpg",
            f"*{key}*candidate_phase*.jpeg",
            f"*{key}*phase*.jpeg",
            f"*{key}*candidate_phase*.pdf",
            f"*{key}*phase*.pdf",
        )
        for pattern in phase_patterns:
            matches = sorted(plot_dir.rglob(pattern), key=lambda p: str(p))
            if matches:
                return matches[0]
    return None


def export_reviews(conn: sqlite3.Connection, out_path: Path, only_reviewed: bool = True) -> None:
    candidate_cols = ["candidate_id", "source_path"] + _COL_NAMES
    review_cols = [
        "review_pass",
        "notes",
        "workflow_status",
        "disposition",
        "morphology_primary",
        "morphology_secondary",
        "morphology_secondary_json",
        "morphology_polarity",
        "morphology_recurrence",
        "baseline_behavior",
        "physical_primary",
        "physical_secondary",
        "classification_confidence",
        "priority_tags_json",
        "evidence_flags_json",
        "model_tags_json",
        "duplicate_of",
        "known_object_id",
        "known_object_source",
        "taxonomy_version",
        "legacy_review_json",
        "reviewer",
        "updated_at",
    ]
    select_cols = [
        *[f"c.{col}" for col in candidate_cols],
        *[f"r.{col}" for col in review_cols],
    ]
    select_clause = ",\n            ".join(select_cols)
    query = f"""
        SELECT
            {select_clause}
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if only_reviewed:
        query += " WHERE r.workflow_status IS NOT NULL AND r.workflow_status != 'unreviewed'"
    df = pd.read_sql_query(query, conn)
    write_feature_table(df, out_path)


def merge_review_databases(
    source_db: Path,
    target_db: Path,
    *,
    candidate_ids: Iterable[str] | None = None,
    only_reviewed: bool = True,
) -> dict[str, int]:
    """Merge review content from one DB into another.

    Reviews are matched by ``candidate_id``. When both source and target contain
    a review row, the row with the newer ``updated_at`` wins.
    """
    source_path = Path(source_db).expanduser().resolve()
    target_path = Path(target_db).expanduser().resolve()
    if source_path == target_path:
        raise ValueError("Source and target review DB paths must differ.")
    if not source_path.exists():
        raise FileNotFoundError(f"Source DB not found: {source_path}")

    candidate_scope = {str(cid).strip() for cid in (candidate_ids or []) if str(cid).strip()}

    with db_connect(source_path) as src_conn:
        candidate_query = "SELECT * FROM candidates"
        review_query = "SELECT * FROM reviews"
        history_query = "SELECT candidate_id, event_type, payload_json, reviewer, created_at FROM review_history"

        src_candidates = pd.read_sql_query(candidate_query, src_conn)
        src_reviews = pd.read_sql_query(review_query, src_conn)
        src_history = pd.read_sql_query(history_query, src_conn)

    if candidate_scope:
        src_candidates = src_candidates[src_candidates["candidate_id"].astype(str).isin(sorted(candidate_scope))].copy()
        src_reviews = src_reviews[src_reviews["candidate_id"].astype(str).isin(sorted(candidate_scope))].copy()
        src_history = src_history[src_history["candidate_id"].astype(str).isin(sorted(candidate_scope))].copy()

    if only_reviewed and not src_reviews.empty:
        if "workflow_status" in src_reviews.columns and "status" in src_reviews.columns:
            status_series = src_reviews["workflow_status"].where(
                src_reviews["workflow_status"].notna() & (src_reviews["workflow_status"].astype(str) != ""),
                src_reviews["status"],
            ).fillna("").astype(str)
        else:
            status_col = "workflow_status" if "workflow_status" in src_reviews.columns else "status"
            status_series = src_reviews[status_col].fillna("").astype(str)
        src_reviews = src_reviews[status_series.ne("") & status_series.ne("unreviewed")].copy()

    review_candidate_ids = {str(cid).strip() for cid in src_reviews.get("candidate_id", pd.Series(dtype="object")).tolist() if str(cid).strip()}
    if candidate_scope:
        scoped_candidate_ids = candidate_scope | review_candidate_ids
    else:
        scoped_candidate_ids = {str(cid).strip() for cid in src_candidates.get("candidate_id", pd.Series(dtype="object")).tolist() if str(cid).strip()}
        if only_reviewed:
            scoped_candidate_ids = scoped_candidate_ids | review_candidate_ids

    if scoped_candidate_ids and not src_candidates.empty:
        src_candidates = src_candidates[src_candidates["candidate_id"].astype(str).isin(sorted(scoped_candidate_ids))].copy()

    with db_connect(target_path) as dst_conn:
        existing_candidate_ids = {
            str(row[0]).strip()
            for row in dst_conn.execute("SELECT candidate_id FROM candidates").fetchall()
        }
        missing_candidates = src_candidates[~src_candidates["candidate_id"].astype(str).isin(sorted(existing_candidate_ids))].copy() if not src_candidates.empty else pd.DataFrame()
        inserted_candidate_rows = 0
        inserted_candidates = 0
        if not missing_candidates.empty:
            inserted_candidate_rows, inserted_candidates = upsert_candidates_frame(dst_conn, missing_candidates)

        target_reviews = pd.read_sql_query("SELECT * FROM reviews", dst_conn)
        target_review_map = {
            str(row["candidate_id"]).strip(): row
            for _, row in target_reviews.iterrows()
        }
        dst_review_cols = [str(row[1]) for row in dst_conn.execute("PRAGMA table_info(reviews)").fetchall()]
        common_review_cols = [
            col for col in src_reviews.columns
            if col in dst_review_cols and col != "candidate_id"
        ]
        if "workflow_status" not in common_review_cols and "workflow_status" in dst_review_cols:
            common_review_cols.append("workflow_status")
        if "taxonomy_version" not in common_review_cols and "taxonomy_version" in dst_review_cols:
            common_review_cols.append("taxonomy_version")

        inserted_reviews = 0
        updated_reviews = 0
        skipped_reviews = 0
        for _, row in src_reviews.iterrows():
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            target_row = target_review_map.get(candidate_id)
            source_updated = _parse_updated_at(row.get("updated_at"))
            target_updated = _parse_updated_at(target_row.get("updated_at")) if target_row is not None else datetime.min.replace(tzinfo=timezone.utc)
            if target_row is not None and source_updated <= target_updated:
                skipped_reviews += 1
                continue

            row_values: dict[str, object] = {}
            for col in common_review_cols:
                value = row.get(col)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    value = None
                row_values[col] = value
            if "workflow_status" in common_review_cols and not row_values.get("workflow_status"):
                row_values["workflow_status"] = row.get("status") or "unreviewed"
            if "taxonomy_version" in common_review_cols and not row_values.get("taxonomy_version"):
                row_values["taxonomy_version"] = TAXONOMY_VERSION
            if "updated_at" in common_review_cols and not row_values.get("updated_at"):
                row_values["updated_at"] = _utc_now()
            if "event_class" in common_review_cols and not row_values.get("event_class"):
                row_values["event_class"] = "unclassified"
            if "status" in common_review_cols and not row_values.get("status"):
                row_values["status"] = row_values.get("workflow_status") or "unreviewed"
            if "review_pass" in common_review_cols and not row_values.get("review_pass"):
                row_values["review_pass"] = 1

            insert_cols = ["candidate_id", *common_review_cols]
            placeholders = ", ".join(["?"] * len(insert_cols))
            update_cols = [col for col in insert_cols if col != "candidate_id"]
            conflict_set = ",\n                    ".join(f"{col}=excluded.{col}" for col in update_cols)

            dst_conn.execute(
                f"""
                INSERT INTO reviews ({', '.join(insert_cols)})
                VALUES ({placeholders})
                ON CONFLICT(candidate_id) DO UPDATE SET
                    {conflict_set}
                """,
                (candidate_id, *(row_values[col] for col in common_review_cols)),
            )
            if target_row is None:
                inserted_reviews += 1
            else:
                updated_reviews += 1

        existing_history = {
            tuple(row)
            for row in dst_conn.execute(
                "SELECT candidate_id, event_type, payload_json, reviewer, created_at FROM review_history"
            ).fetchall()
        }
        inserted_history = 0
        for _, row in src_history.iterrows():
            entry = (
                str(row.get("candidate_id") or "").strip(),
                str(row.get("event_type") or ""),
                str(row.get("payload_json") or "{}"),
                None if row.get("reviewer") in (None, "") else str(row.get("reviewer")),
                str(row.get("created_at") or _utc_now()),
            )
            if not entry[0] or entry in existing_history:
                continue
            dst_conn.execute(
                "INSERT INTO review_history (candidate_id, event_type, payload_json, reviewer, created_at) VALUES (?, ?, ?, ?, ?)",
                entry,
            )
            existing_history.add(entry)
            inserted_history += 1

        dst_conn.commit()

    return {
        "candidate_scope": len(scoped_candidate_ids),
        "candidate_rows_written": inserted_candidate_rows,
        "candidates_inserted": inserted_candidates,
        "reviews_inserted": inserted_reviews,
        "reviews_updated": updated_reviews,
        "reviews_skipped": skipped_reviews,
        "history_inserted": inserted_history,
    }


# ---------------------------------------------------------------------------
# Raw light-curve file import
# ---------------------------------------------------------------------------
_LC_CACHE_DIR = REVIEW_IMPORTED_LC_CACHE_DIR.expanduser()


def import_lightcurve_files(
    conn: sqlite3.Connection,
    file_path: Path,
    *,
    characterize: bool = False,
    vet: bool = False,
) -> tuple[int, int]:
    """Import raw light-curve CSV or Parquet files into the review DB.

    If the file has an ``asas_sn_id`` column with multiple unique values,
    each source is split into its own cached CSV.  Otherwise the file is
    treated as a single source and the filename stem is used as candidate_id.

    Returns (n_rows, n_new) like ``import_candidates``.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = read_parquet_table(file_path)
    if df.empty:
        return 0, 0

    _LC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Detect multi-source files
    id_col = None
    for col in ("asas_sn_id", "source_id", "candidate_id"):
        if col in df.columns and df[col].nunique() > 1:
            id_col = col
            break

    if id_col and df[id_col].nunique() > 1:
        # Multi-source: split into individual LC files
        rows = []
        for src_id, sub in df.groupby(id_col):
            cache_file = _LC_CACHE_DIR / f"{src_id}.csv"
            sub.to_csv(cache_file, index=False)
            rows.append({
                "candidate_id": str(src_id),
                "lc_path": str(cache_file),
            })
        candidate_df = pd.DataFrame(rows)
    else:
        # Single-source: copy to cache
        candidate_id = file_path.stem
        cache_file = _LC_CACHE_DIR / file_path.name
        if cache_file != file_path:

            shutil.copy2(file_path, cache_file)
        candidate_df = pd.DataFrame([{
            "candidate_id": candidate_id,
            "lc_path": str(cache_file),
        }])

    return import_candidates(
        conn,
        candidate_df,
        source_path=str(file_path),
        characterize_before_import=characterize,
        vet_before_import=vet,
    )
