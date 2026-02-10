from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from malca.review.metadata import normalize_vsx_record
from malca.config.config_paths import VSX_CROSSMATCH_PATH, GAIA_CACHE_FILE
from malca.config.config_characterize import GAIA_CHUNK_SIZE


DEFAULT_DB_PATH = "output/review/review.db"
STATUS_OPTIONS = ["unreviewed", "reviewed", "needs_followup"]
INTEREST_REASON_TAGS = [
    "clean_event",
    "multi_camera_support",
    "interesting_morphology",
    "periodic_contaminant",
    "camera_artifact",
    "known_object_nearby",
    "needs_followup_data",
]
EVENT_CLASS_OPTIONS = [
    "unclassified",
    "circumstellar_dust",
    "microlensing",
    "flare",
    "eclipsing_binary",
    "instrumental",
    "unknown_interesting",
    "not_real",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def infer_candidate_id(df: pd.DataFrame) -> pd.Series:
    if "candidate_id" not in df.columns:
        raise ValueError("Input must include a 'candidate_id' column.")

    vals = df["candidate_id"].astype(str).str.strip()
    if not vals.nunique(dropna=True) == len(df):
        raise ValueError("'candidate_id' values must be unique.")
    return vals


def load_candidates_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file type. Use CSV or Parquet.")


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

    # Detect candidates file (priority: most enriched first)
    candidates_priority = [
        "results/lc_events_spectra.parquet",
        "results/lc_events_neighbors.parquet",
        "results/lc_events_classified.parquet",
        "results/lc_events_enriched.parquet",
        "results/lc_events_characterized.parquet",
        "results/lc_events_filtered.parquet",
    ]

    for rel_path in candidates_priority:
        candidate_file = run_dir / rel_path
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

    # Detect gaia cache (optional, no warning if missing)
    gaia_cache = run_dir / "gaia_cache" / "gaia_cache.parquet"
    if gaia_cache.exists():
        results['gaia_cache'] = gaia_cache

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
    ("asas_sn_id",               "TEXT",    "text"),
    ("lc_path",                  "TEXT",    "text"),
    # -- top-level filter flags --
    ("failed_any",               "INTEGER", "bool"),
    ("periodic_flag",            "INTEGER", "bool"),
    ("catalog_match",            "INTEGER", "bool"),
    ("high_ruwe_flag",           "INTEGER", "bool"),
    # -- periodicity --
    ("periodicity_score",        "REAL",    "float"),
    ("lsp_bootstrap_sig",        "REAL",    "float"),
    ("lsp_power",                "REAL",    "float"),
    ("lsp_period",               "REAL",    "float"),
    ("lsp_is_alias",             "INTEGER", "bool"),
    ("lsp_is_significant",       "INTEGER", "bool"),
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
    ("dipper_score",             "REAL",    "float"),
    ("dipper_n_dips",            "REAL",    "float"),
    ("dipper_n_valid_dips",      "REAL",    "float"),
    ("jumper_score",             "REAL",    "float"),
    ("jumper_n_jumps",           "REAL",    "float"),
    ("jumper_n_valid_jumps",     "REAL",    "float"),
    # -- stellar parameters --
    ("ruwe",                     "REAL",    "float"),
    ("teff_gspphot",             "REAL",    "float"),
    ("logg_gspphot",             "REAL",    "float"),
    ("mh_gspphot",               "REAL",    "float"),
    ("distance_gspphot",         "REAL",    "float"),
    ("parallax",                 "REAL",    "float"),
    ("pmra",                     "REAL",    "float"),
    ("pmdec",                    "REAL",    "float"),
    # -- photometry --
    ("tmass_j",                  "REAL",    "float"),
    ("tmass_h",                  "REAL",    "float"),
    ("tmass_k",                  "REAL",    "float"),
    ("unwise_w1",                "REAL",    "float"),
    ("unwise_w2",                "REAL",    "float"),
    ("H_K",                      "REAL",    "float"),
    ("W1_W2",                    "REAL",    "float"),
    ("iphas_ha_mag",             "REAL",    "float"),
    ("unwise_w1_zscore",         "REAL",    "float"),
    ("unwise_w2_zscore",         "REAL",    "float"),
    # -- galactic coordinates --
    ("gal_l",                    "REAL",    "float"),
    ("gal_b",                    "REAL",    "float"),
    # -- extinction & environment --
    ("A_v_3d",                   "REAL",    "float"),
    ("ebv_3d",                   "REAL",    "float"),
    ("population",               "TEXT",    "text"),
    ("age50",                    "REAL",    "float"),
    ("mass50",                   "REAL",    "float"),
    ("banyan_field_prob",        "REAL",    "float"),
    ("banyan_best_assoc",        "TEXT",    "text"),
    # -- crossmatch details --
    ("vsx_class",                "TEXT",    "text"),
    ("vsx_sep_arcsec",           "REAL",    "float"),
    ("sfr_name",                 "TEXT",    "text"),
    ("sfr_sep_arcmin",           "REAL",    "float"),
    ("cluster_name",             "TEXT",    "text"),
    ("cluster_membership_prob",  "REAL",    "float"),
    # -- light curve basics --
    ("n_points",                 "REAL",    "float"),
    ("n_cameras",                "REAL",    "float"),
    ("baseline_mag",             "REAL",    "float"),
    ("baseline_source",          "TEXT",    "text"),
    ("cadence_median_days",      "REAL",    "float"),
    ("trigger_mode",             "TEXT",    "text"),
    # -- YSO / classification --
    ("trigger_type",             "TEXT",    "text"),
    ("yso_class",                "TEXT",    "text"),
    ("final_class",              "TEXT",    "text"),
    ("P_eb",                     "REAL",    "float"),
    ("P_cv",                     "REAL",    "float"),
    ("P_starspot",               "REAL",    "float"),
    ("P_disk",                   "REAL",    "float"),
    ("a_circ_au",                "REAL",    "float"),
    ("transit_prob",             "REAL",    "float"),
    ("hill_radius_rsun",         "REAL",    "float"),
    # -- individual fail flags --
    ("failed_sparse",            "INTEGER", "bool"),
    ("failed_multi_camera",      "INTEGER", "bool"),
    ("failed_vsx",               "INTEGER", "bool"),
    ("failed_evidence_strength", "INTEGER", "bool"),
    ("failed_run_robustness",    "INTEGER", "bool"),
    ("failed_morphology",        "INTEGER", "bool"),
    ("failed_score",             "INTEGER", "bool"),
    ("failed_periodicity",       "INTEGER", "bool"),
    ("failed_gaia_ruwe",         "INTEGER", "bool"),
    ("failed_periodic_catalog",  "INTEGER", "bool"),
    ("failed_signal_amplitude",  "INTEGER", "bool"),
    ("bad_cameras_filtered",     "INTEGER", "bool"),
]

# Derived helpers
_COL_NAMES = [c[0] for c in _CANDIDATE_COLUMNS]
_BOOL_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "bool"}
_FLOAT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "float"}
_TEXT_COLS = {c[0] for c in _CANDIDATE_COLUMNS if c[2] == "text"}


def _migrate_candidates_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema.  Safe to call repeatedly."""
    for col, dtype, _ in _CANDIDATE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE candidates ADD COLUMN {col} {dtype}")
        except Exception:
            pass  # column already exists
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    col_defs = ",\n            ".join(
        f"{col} {dtype}" for col, dtype, _ in _CANDIDATE_COLUMNS
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
        """
        CREATE TABLE IF NOT EXISTS reviews (
            candidate_id TEXT PRIMARY KEY,
            interest_score INTEGER,
            interest_reason TEXT,
            event_class TEXT DEFAULT 'unclassified',
            review_pass INTEGER,
            notes TEXT,
            status TEXT,
            reviewer TEXT,
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
    conn.commit()


def _migrate_reviews_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial reviews schema.  Safe to call repeatedly."""
    for col, dtype in [("event_class", "TEXT DEFAULT 'unclassified'")]:
        try:
            conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} {dtype}")
        except Exception:
            pass  # column already exists
    conn.commit()


def db_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    init_db(conn)
    _migrate_candidates_columns(conn)
    _migrate_reviews_columns(conn)
    return conn


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
) -> tuple[int, int]:
    if df.empty:
        return 0, 0

    df_use = df
    if characterize_before_import:
        try:
            from malca.characterize import characterize_candidates_df

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

    df_use = df_use.copy()
    df_use["candidate_id"] = infer_candidate_id(df_use)
    imported_at = _utc_now()

    def _opt_str(d, key):
        v = d.get(key)
        return str(v) if v is not None else None

    def _opt_bool(d, key):
        v = d.get(key)
        return int(_as_bool(v)) if v is not None else None

    # Map payload key → column; most are identical to the column name.
    _payload_alias = {"lc_path": "path"}

    rows = []
    for _, row in df_use.iterrows():
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        row_dict = normalize_vsx_record(row_dict)
        vals: list = [str(row_dict.get("candidate_id")), source_path]
        for col, _dtype, etype in _CANDIDATE_COLUMNS:
            payload_key = _payload_alias.get(col, col)
            raw = row_dict.get(payload_key)
            if etype == "bool":
                vals.append(_opt_bool(row_dict, payload_key))
            elif etype == "float":
                vals.append(_to_float(raw))
            else:
                vals.append(_opt_str(row_dict, payload_key))
        vals.append(json.dumps(row_dict, default=str))
        vals.append(imported_at)
        rows.append(tuple(vals))

    _all_col_names = ["candidate_id", "source_path"] + _COL_NAMES + ["payload_json", "imported_at"]
    _candidate_cols = ", ".join(_all_col_names)
    _placeholders = ", ".join(["?"] * len(_all_col_names))
    _update_cols = [c for c in _all_col_names if c != "candidate_id"]
    _conflict_set = ", ".join(f"{c}=excluded.{c}" for c in _update_cols)

    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.executemany(
        f"""
        INSERT INTO candidates ({_candidate_cols})
        VALUES ({_placeholders})
        ON CONFLICT(candidate_id) DO UPDATE SET {_conflict_set}
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return len(rows), int(after - before)


def query_queue(conn: sqlite3.Connection, *, filters: dict | None = None,
                # Legacy keyword args (still accepted for backward compat)
                only_unreviewed: bool | None = None,
                require_failed_any_false: bool | None = None,
                periodic_flag_mode: str | None = None,
                catalog_match_mode: str | None = None,
                high_ruwe_mode: str | None = None,
                min_periodicity_score: float | None = None,
                max_lsp_bootstrap_sig: float | None = None,
                min_lsp_power: float | None = None,
                sort_col: str | None = None,
                sort_desc: bool | None = None,
                ) -> pd.DataFrame:
    """Query the candidate queue with optional filters.

    Accepts either a *filters* dict or the legacy keyword arguments.
    If *filters* is provided it takes precedence.
    """
    if filters is None:
        filters = {}
    # Merge legacy kwargs as defaults (filters dict wins)
    _defaults = {
        'only_unreviewed': only_unreviewed if only_unreviewed is not None else False,
        'require_failed_any_false': require_failed_any_false if require_failed_any_false is not None else False,
        'periodic_flag_mode': periodic_flag_mode or 'Any',
        'catalog_match_mode': catalog_match_mode or 'Any',
        'high_ruwe_mode': high_ruwe_mode or 'Any',
        'min_periodicity_score': min_periodicity_score,
        'max_lsp_bootstrap_sig': max_lsp_bootstrap_sig,
        'min_lsp_power': min_lsp_power,
        'sort_col': sort_col or 'candidate_id',
        'sort_desc': sort_desc if sort_desc is not None else False,
    }
    for k, v in _defaults.items():
        filters.setdefault(k, v)

    where: list[str] = []
    params: list = []

    # --- review status ---
    if filters.get('only_unreviewed'):
        where.append("(r.status IS NULL OR r.status='unreviewed')")

    # --- failed_any shortcut ---
    if filters.get('require_failed_any_false'):
        where.append("(c.failed_any = 0)")

    # --- Any / True / False bool-mode filters (auto-generated) ---
    mode_map = {"Any": None, "True": 1, "False": 0}
    for col in _BOOL_COLS:
        key = f"{col}_mode"
        mode = filters.get(key, "Any")
        val = mode_map.get(mode)
        if val is not None:
            where.append(f"(c.{col} = ?)")
            params.append(val)

    # --- numeric range filters (auto-generated) ---
    # Convention: "min_<col>" → >=, "max_<col>" → <=
    for col in sorted(_FLOAT_COLS):
        for prefix, op in [("min_", ">="), ("max_", "<=")]:
            key = f"{prefix}{col}"
            val = filters.get(key)
            if val is not None:
                where.append(f"(c.{col} IS NOT NULL AND c.{col} {op} ?)")
                params.append(float(val))

    # --- string filters (auto-generated; exact match) ---
    for col in sorted(_TEXT_COLS):
        val = filters.get(col)
        if val:
            val = str(val).strip()
            if val:
                where.append(f"(c.{col} IS NOT NULL AND c.{col} = ?)")
                params.append(val)

    # --- sorting (any float column + review columns) ---
    _sortable = {c: f"c.{c}" for c in _FLOAT_COLS}
    _sortable["candidate_id"] = "c.candidate_id"
    _sortable.update({"updated_at": "r.updated_at", "interest_score": "r.interest_score",
                       "review_pass": "r.review_pass"})
    sc = filters.get('sort_col', 'candidate_id')
    order_col = _sortable.get(sc, "c.candidate_id")
    direction = "DESC" if filters.get('sort_desc') else "ASC"

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
            c.lsp_bootstrap_sig,
            c.lsp_power,
            c.lsp_period,
            c.dip_best_log_bf,
            c.jump_best_log_bf,
            r.interest_score,
            r.interest_reason,
            r.review_pass,
            r.status,
            r.notes,
            r.reviewer,
            r.updated_at
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if where:
        query += " WHERE " + " AND ".join(where)
    query += f" ORDER BY {order_col} {direction}, c.candidate_id ASC"
    return pd.read_sql_query(query, conn, params=params)


def get_candidate_payload(conn: sqlite3.Connection, candidate_id: str) -> dict:
    row = conn.execute("SELECT payload_json FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


def _parse_reason_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        out = json.loads(raw)
        if isinstance(out, list):
            return [str(x) for x in out if str(x).strip()]
    except Exception:
        pass
    return []


def get_review(conn: sqlite3.Connection, candidate_id: str) -> dict:
    row = conn.execute(
        """
        SELECT interest_score, interest_reason, review_pass, notes, status, reviewer, updated_at, event_class
        FROM reviews WHERE candidate_id=?
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        return {
            "interest_score": 2,
            "interest_reason": [],
            "event_class": "unclassified",
            "review_pass": 1,
            "notes": "",
            "status": "unreviewed",
            "reviewer": "",
            "updated_at": None,
        }
    score = 2 if row[0] is None else int(row[0])
    score = int(np.clip(score, 0, 5))
    return {
        "interest_score": score,
        "interest_reason": _parse_reason_list(row[1]),
        "event_class": str(row[7]) if row[7] else "unclassified",
        "review_pass": 1 if row[2] is None else max(1, int(row[2])),
        "notes": "" if row[3] is None else str(row[3]),
        "status": "unreviewed" if row[4] is None else str(row[4]),
        "reviewer": "" if row[5] is None else str(row[5]),
        "updated_at": row[6],
    }


def save_review(
    conn: sqlite3.Connection,
    *,
    candidate_id: str,
    interest_score: int,
    interest_reason: list[str],
    event_class: str = "unclassified",
    review_pass: int,
    notes: str,
    status: str,
    reviewer: str,
    event_type: str = "save",
) -> None:
    ts = _utc_now()
    score_int = int(np.clip(int(interest_score), 0, 5))
    pass_int = max(1, int(review_pass))
    reason_json = json.dumps(sorted(set(interest_reason)))
    ec = str(event_class) if event_class else "unclassified"
    conn.execute(
        """
        INSERT INTO reviews (candidate_id, interest_score, interest_reason, event_class, review_pass, notes, status, reviewer, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            interest_score=excluded.interest_score,
            interest_reason=excluded.interest_reason,
            event_class=excluded.event_class,
            review_pass=excluded.review_pass,
            notes=excluded.notes,
            status=excluded.status,
            reviewer=excluded.reviewer,
            updated_at=excluded.updated_at
        """,
        (candidate_id, score_int, reason_json, ec, pass_int, notes, status, reviewer, ts),
    )
    payload = {
        "interest_score": score_int,
        "interest_reason": sorted(set(interest_reason)),
        "event_class": ec,
        "review_pass": pass_int,
        "notes": notes,
        "status": status,
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
    reviewed = conn.execute("SELECT COUNT(*) FROM reviews WHERE status IS NOT NULL AND status != 'unreviewed'").fetchone()[0]
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
            matches = list(plot_dir.rglob(f"*{key}*{ext[1:]}"))
            if matches:
                return matches[0]
    return None


def export_reviews(conn: sqlite3.Connection, out_path: Path, only_reviewed: bool = True) -> None:
    query = """
        SELECT
            c.candidate_id,
            c.asas_sn_id,
            c.lc_path,
            c.failed_any,
            c.periodic_flag,
            c.catalog_match,
            c.high_ruwe_flag,
            c.periodicity_score,
            c.lsp_bootstrap_sig,
            c.lsp_power,
            c.lsp_period,
            c.dip_best_log_bf,
            c.jump_best_log_bf,
            r.interest_score,
            r.interest_reason,
            r.event_class,
            r.review_pass,
            r.notes,
            r.status,
            r.reviewer,
            r.updated_at,
            c.payload_json
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if only_reviewed:
        query += " WHERE r.status IS NOT NULL AND r.status != 'unreviewed'"
    df = pd.read_sql_query(query, conn)
    if not df.empty and "payload_json" in df.columns:
        payload_rows = []
        for value in df["payload_json"]:
            try:
                row_dict = json.loads(value) if isinstance(value, str) else {}
            except Exception:
                row_dict = {}
            payload_rows.append(normalize_vsx_record(row_dict))

        payload_df = pd.DataFrame(payload_rows)
        for col in ["vsx_class", "vsx_sep_arcsec", "population", "yso_class"]:
            if col in payload_df.columns and col not in df.columns:
                df[col] = payload_df[col]
        df = df.drop(columns=["payload_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False, compression="zstd")
    else:
        df.to_csv(out_path, index=False)
