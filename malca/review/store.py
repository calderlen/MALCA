from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


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
    for col in ("candidate_id", "asas_sn_id", "source_id", "id", "path"):
        if col in df.columns:
            vals = df[col].astype(str).str.strip()
            if vals.nunique(dropna=True) == len(df):
                return vals
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)


def load_candidates_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file type. Use CSV or Parquet.")


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            source_path TEXT,
            asas_sn_id TEXT,
            lc_path TEXT,
            failed_any INTEGER,
            periodic_flag INTEGER,
            catalog_match INTEGER,
            high_ruwe_flag INTEGER,
            periodicity_score REAL,
            lsp_bootstrap_sig REAL,
            lsp_power REAL,
            lsp_period REAL,
            dip_best_log_bf REAL,
            jump_best_log_bf REAL,
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
            review_pass INTEGER,
            notes TEXT,
            status TEXT,
            reviewer TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    # Backward-compatible columns from older schema
    _ensure_column(conn, "reviews", "label", "TEXT")
    _ensure_column(conn, "reviews", "confidence", "INTEGER")
    _ensure_column(conn, "reviews", "interest_reason", "TEXT")
    _ensure_column(conn, "reviews", "review_pass", "INTEGER")

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


def db_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    init_db(conn)
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


def import_candidates(conn: sqlite3.Connection, df: pd.DataFrame, source_path: str) -> tuple[int, int]:
    if df.empty:
        return 0, 0
    df = df.copy()
    df["candidate_id"] = infer_candidate_id(df)
    imported_at = _utc_now()

    rows = []
    for _, row in df.iterrows():
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        rows.append(
            (
                str(row_dict.get("candidate_id")),
                source_path,
                str(row_dict.get("asas_sn_id")) if row_dict.get("asas_sn_id") is not None else None,
                str(row_dict.get("path")) if row_dict.get("path") is not None else None,
                int(_as_bool(row_dict.get("failed_any"))),
                int(_as_bool(row_dict.get("periodic_flag"))),
                int(_as_bool(row_dict.get("catalog_match"))),
                int(_as_bool(row_dict.get("high_ruwe_flag"))),
                _to_float(row_dict.get("periodicity_score")),
                _to_float(row_dict.get("lsp_bootstrap_sig")),
                _to_float(row_dict.get("lsp_power")),
                _to_float(row_dict.get("lsp_period")),
                _to_float(row_dict.get("dip_best_log_bf")),
                _to_float(row_dict.get("jump_best_log_bf")),
                json.dumps(row_dict, default=str),
                imported_at,
            )
        )

    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.executemany(
        """
        INSERT INTO candidates (
            candidate_id, source_path, asas_sn_id, lc_path,
            failed_any, periodic_flag, catalog_match, high_ruwe_flag,
            periodicity_score, lsp_bootstrap_sig, lsp_power, lsp_period,
            dip_best_log_bf, jump_best_log_bf, payload_json, imported_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            source_path=excluded.source_path,
            asas_sn_id=excluded.asas_sn_id,
            lc_path=excluded.lc_path,
            failed_any=excluded.failed_any,
            periodic_flag=excluded.periodic_flag,
            catalog_match=excluded.catalog_match,
            high_ruwe_flag=excluded.high_ruwe_flag,
            periodicity_score=excluded.periodicity_score,
            lsp_bootstrap_sig=excluded.lsp_bootstrap_sig,
            lsp_power=excluded.lsp_power,
            lsp_period=excluded.lsp_period,
            dip_best_log_bf=excluded.dip_best_log_bf,
            jump_best_log_bf=excluded.jump_best_log_bf,
            payload_json=excluded.payload_json,
            imported_at=excluded.imported_at
        """,
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return len(rows), int(after - before)


def query_queue(
    conn: sqlite3.Connection,
    *,
    only_unreviewed: bool,
    require_failed_any_false: bool,
    periodic_flag_mode: str,
    catalog_match_mode: str,
    high_ruwe_mode: str,
    min_periodicity_score: float | None,
    max_lsp_bootstrap_sig: float | None,
    min_lsp_power: float | None,
    sort_col: str,
    sort_desc: bool,
) -> pd.DataFrame:
    where = []
    params: list = []
    if only_unreviewed:
        where.append("(r.status IS NULL OR r.status='unreviewed')")
    if require_failed_any_false:
        where.append("(c.failed_any = 0)")

    mode_map = {"Any": None, "True": 1, "False": 0}
    for mode, col in [
        (periodic_flag_mode, "c.periodic_flag"),
        (catalog_match_mode, "c.catalog_match"),
        (high_ruwe_mode, "c.high_ruwe_flag"),
    ]:
        val = mode_map[mode]
        if val is not None:
            where.append(f"({col} = ?)")
            params.append(val)
    if min_periodicity_score is not None:
        where.append("(c.periodicity_score IS NOT NULL AND c.periodicity_score >= ?)")
        params.append(float(min_periodicity_score))
    if max_lsp_bootstrap_sig is not None:
        where.append("(c.lsp_bootstrap_sig IS NOT NULL AND c.lsp_bootstrap_sig <= ?)")
        params.append(float(max_lsp_bootstrap_sig))
    if min_lsp_power is not None:
        where.append("(c.lsp_power IS NOT NULL AND c.lsp_power >= ?)")
        params.append(float(min_lsp_power))

    order_cols = {
        "candidate_id": "c.candidate_id",
        "periodicity_score": "c.periodicity_score",
        "lsp_bootstrap_sig": "c.lsp_bootstrap_sig",
        "lsp_power": "c.lsp_power",
        "dip_best_log_bf": "c.dip_best_log_bf",
        "jump_best_log_bf": "c.jump_best_log_bf",
        "updated_at": "r.updated_at",
        "interest_score": "r.interest_score",
        "review_pass": "r.review_pass",
    }
    order_col = order_cols.get(sort_col, "c.candidate_id")
    direction = "DESC" if sort_desc else "ASC"

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
        SELECT interest_score, interest_reason, review_pass, notes, status, reviewer, updated_at
        FROM reviews WHERE candidate_id=?
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        return {
            "interest_score": 2,
            "interest_reason": [],
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
    conn.execute(
        """
        INSERT INTO reviews (candidate_id, interest_score, interest_reason, review_pass, notes, status, reviewer, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            interest_score=excluded.interest_score,
            interest_reason=excluded.interest_reason,
            review_pass=excluded.review_pass,
            notes=excluded.notes,
            status=excluded.status,
            reviewer=excluded.reviewer,
            updated_at=excluded.updated_at
        """,
        (candidate_id, score_int, reason_json, pass_int, notes, status, reviewer, ts),
    )
    payload = {
        "interest_score": score_int,
        "interest_reason": sorted(set(interest_reason)),
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
    for k in ("asas_sn_id", "candidate_id", "source_id", "id"):
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
            r.review_pass,
            r.notes,
            r.status,
            r.reviewer,
            r.updated_at
        FROM candidates c
        LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
    """
    if only_reviewed:
        query += " WHERE r.status IS NOT NULL AND r.status != 'unreviewed'"
    df = pd.read_sql_query(query, conn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
