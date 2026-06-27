"""Persistent manifest helpers for cached external light-curve files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time

import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from malca.config import PARQUET_CACHE_COMPRESSION


EXTERNAL_LC_MANIFEST_FILE = "external_lc_manifest.parquet"
EXTERNAL_LC_MANIFEST_COLUMNS = (
    "candidate_id",
    "source",
    "file_prefix",
    "path",
    "path_relative",
    "size_bytes",
    "mtime_ns",
    "updated_unix",
)
_NEOWISE_ALIASES = {"wise", "neowise", "wise_w1_w2", "neowise_w1", "neowise_w2", "neowise_color"}


def normalize_external_lc_file_prefix(value: object) -> str:
    """Return the canonical external-LC file prefix without the trailing ``_lc``."""
    text = str(value or "").strip().lower()
    if text.endswith("_lc"):
        text = text[:-3]
    if text in _NEOWISE_ALIASES:
        return "neowise"
    return text


def external_lc_manifest_path(results_root: Path | str | None) -> Path | None:
    if results_root is None:
        return None
    return Path(results_root).expanduser() / EXTERNAL_LC_MANIFEST_FILE


def _file_signature(path: Path | None) -> tuple[int, int]:
    if path is None or not path.exists():
        return (0, -1)
    try:
        stat = path.stat()
    except OSError:
        return (0, -1)
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _canonical_manifest(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(EXTERNAL_LC_MANIFEST_COLUMNS))
    out = df.copy()
    for col in EXTERNAL_LC_MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[list(EXTERNAL_LC_MANIFEST_COLUMNS)]
    out["candidate_id"] = out["candidate_id"].fillna("").astype(str)
    out["source"] = out["source"].fillna("").astype(str).map(normalize_external_lc_file_prefix)
    out["file_prefix"] = out["file_prefix"].fillna("").astype(str).map(normalize_external_lc_file_prefix)
    out["path"] = out["path"].fillna("").astype(str)
    out["path_relative"] = out["path_relative"].fillna("").astype(str)
    out = out[out["candidate_id"].astype(str).str.len() > 0]
    out = out[out["file_prefix"].astype(str).str.len() > 0]
    return out.reset_index(drop=True)


@lru_cache(maxsize=64)
def _read_external_lc_manifest_cached(root_text: str, manifest_mtime_ns: int, manifest_size: int) -> pd.DataFrame:
    del manifest_mtime_ns, manifest_size
    manifest_path = external_lc_manifest_path(root_text)
    if manifest_path is None or not manifest_path.exists():
        return pd.DataFrame(columns=list(EXTERNAL_LC_MANIFEST_COLUMNS))
    try:
        return _canonical_manifest(pd.read_parquet(manifest_path))
    except Exception:
        return pd.DataFrame(columns=list(EXTERNAL_LC_MANIFEST_COLUMNS))


def clear_external_lc_manifest_caches() -> None:
    _read_external_lc_manifest_cached.cache_clear()
    index_external_lc_paths_from_manifest.cache_clear()


def read_external_lc_manifest(results_root: Path | str | None) -> pd.DataFrame:
    manifest_path = external_lc_manifest_path(results_root)
    if manifest_path is None:
        return pd.DataFrame(columns=list(EXTERNAL_LC_MANIFEST_COLUMNS))
    mtime_ns, size = _file_signature(manifest_path)
    root_text = str(Path(results_root).expanduser())
    return _read_external_lc_manifest_cached(root_text, mtime_ns, size).copy()


def _row_for_external_lc(
    results_root: Path,
    path: Path,
    *,
    candidate_id: str,
    source: str,
    file_prefix: str,
) -> dict[str, object] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        root_resolved = results_root.resolve()
        path_resolved = path.resolve()
        path_relative = str(path_resolved.relative_to(root_resolved))
        path_text = str(path_resolved)
    except Exception:
        path_relative = str(path)
        path_text = str(path)
    return {
        "candidate_id": str(candidate_id),
        "source": normalize_external_lc_file_prefix(source),
        "file_prefix": normalize_external_lc_file_prefix(file_prefix),
        "path": path_text,
        "path_relative": path_relative,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "updated_unix": float(time.time()),
    }


def _manifest_row_path(results_root: Path, row: pd.Series) -> Path:
    rel = row.get("path_relative")
    if pd.notna(rel) and str(rel).strip():
        return results_root / str(rel)
    path_text = row.get("path")
    return Path(str(path_text)).expanduser()


def _manifest_row_is_current(path: Path, row: pd.Series) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    try:
        return int(row.get("size_bytes")) == int(stat.st_size) and int(row.get("mtime_ns")) == int(stat.st_mtime_ns)
    except Exception:
        return False


def write_external_lc_manifest(results_root: Path | str | None, manifest: pd.DataFrame) -> bool:
    manifest_path = external_lc_manifest_path(results_root)
    if manifest_path is None:
        return False
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
        with open(lock_path, "a", encoding="ascii") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                _canonical_manifest(manifest).to_parquet(
                    manifest_path,
                    index=False,
                    compression=PARQUET_CACHE_COMPRESSION,
                )
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception:
        return False
    clear_external_lc_manifest_caches()
    return True


def upsert_external_lc_manifest_rows(results_root: Path | str | None, rows: list[dict[str, object]]) -> bool:
    if results_root is None or not rows:
        return False
    existing = read_external_lc_manifest(results_root)
    new = _canonical_manifest(pd.DataFrame(rows))
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    combined = _canonical_manifest(combined)
    combined = combined.drop_duplicates(subset=["candidate_id", "source", "file_prefix"], keep="last")
    return write_external_lc_manifest(results_root, combined)


def upsert_external_lc_manifest_entry(
    results_root: Path | str | None,
    *,
    candidate_id: str,
    source: str,
    file_prefix: str,
    path: Path | str,
) -> bool:
    if results_root is None:
        return False
    root = Path(results_root).expanduser()
    row = _row_for_external_lc(
        root,
        Path(path).expanduser(),
        candidate_id=str(candidate_id),
        source=source,
        file_prefix=file_prefix,
    )
    return upsert_external_lc_manifest_rows(root, [row]) if row is not None else False


def scan_external_lc_manifest_rows(
    results_root: Path | str | None,
    file_prefixes: tuple[str, ...] | list[str] | set[str] | None = None,
) -> list[dict[str, object]]:
    if results_root is None:
        return []
    root = Path(results_root).expanduser()
    if not root.exists():
        return []
    prefixes = [normalize_external_lc_file_prefix(p) for p in (file_prefixes or []) if str(p or "").strip()]
    if not prefixes:
        prefixes = sorted(
            {
                path.name.split("_lc_", 1)[0]
                for path in root.rglob("*_lc_*.parquet")
                if "_lc_" in path.name
            }
        )
    rows: list[dict[str, object]] = []
    for prefix in prefixes:
        stem_prefix = f"{prefix}_lc_"
        for path in root.rglob(f"{stem_prefix}*.parquet"):
            candidate_id = path.stem[len(stem_prefix):]
            if not candidate_id:
                continue
            row = _row_for_external_lc(
                root,
                path,
                candidate_id=candidate_id,
                source=prefix,
                file_prefix=prefix,
            )
            if row is not None:
                rows.append(row)
    return rows


def update_external_lc_manifest_from_scan(
    results_root: Path | str | None,
    file_prefixes: tuple[str, ...] | list[str] | set[str] | None = None,
) -> pd.DataFrame:
    rows = scan_external_lc_manifest_rows(results_root, file_prefixes)
    if rows:
        upsert_external_lc_manifest_rows(results_root, rows)
    return _canonical_manifest(pd.DataFrame(rows))


@lru_cache(maxsize=64)
def index_external_lc_paths_from_manifest(root_text: str, file_prefix: str) -> dict[str, str]:
    """Return candidate -> path using the manifest first, scanning only for misses/stale rows."""
    root = Path(root_text).expanduser()
    prefix = normalize_external_lc_file_prefix(file_prefix)
    if not prefix or not root.exists():
        return {}

    manifest = read_external_lc_manifest(root)
    mapping: dict[str, str] = {}
    repair_rows: list[dict[str, object]] = []
    stale_or_missing = manifest.empty

    if not manifest.empty:
        rows = manifest[manifest["file_prefix"].astype(str).map(normalize_external_lc_file_prefix) == prefix]
        if rows.empty:
            stale_or_missing = True
        for _, row in rows.iterrows():
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            path = _manifest_row_path(root, row)
            if not path.exists():
                stale_or_missing = True
                continue
            mapping[candidate_id] = str(path)
            if not _manifest_row_is_current(path, row):
                repair = _row_for_external_lc(
                    root,
                    path,
                    candidate_id=candidate_id,
                    source=str(row.get("source") or prefix),
                    file_prefix=prefix,
                )
                if repair is not None:
                    repair_rows.append(repair)

    if stale_or_missing:
        scanned_rows = scan_external_lc_manifest_rows(root, [prefix])
        if scanned_rows:
            upsert_external_lc_manifest_rows(root, scanned_rows)
            for row in scanned_rows:
                mapping[str(row["candidate_id"])] = str(_manifest_row_path(root, pd.Series(row)))
    elif repair_rows:
        upsert_external_lc_manifest_rows(root, repair_rows)

    return mapping
