"""Candidate-level features shared by the July 1 Review classifiers.

This module keeps next-iteration feature construction independent of human
labels.  Static context is derived from candidate-table columns, while
recovery-bounded event morphology is measured from every requested native
light curve and cached by candidate ID, light-curve fingerprint, and method
version.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from malca.stv.dimming_window import (
    DIMMING_WINDOW_METHOD_VERSION,
    measure_dimming_complex_window,
)


NEXT_ITERATION_CONTEXT_FEATURES = (
    "A_v_3d",
    "derived_wjk",
    "reduced_proper_motion_g",
    "iphas_r_ha",
    "vphas_r_ha",
)

RECOVERY_FEATURE_SCHEMA_VERSION = "recovery_bounded_ml_features_v1"

RECOVERY_BOUNDED_EVENT_FEATURES = (
    "dimming_complex_duration_days",
    "dimming_complex_is_lower_limit",
    "event_integrated_excess",
    "event_component_epochs",
    "delta_mag_peak",
    "left_baseline_recovered",
    "right_baseline_recovered",
    "event_window_gap_count",
    "event_window_max_gap_days",
    "left_event_boundary_type",
    "right_event_boundary_type",
    "dimming_complex_status",
)

RECOVERY_FEATURE_PROVENANCE_COLUMNS = (
    "recovery_feature_state",
    "recovery_feature_error",
    "recovery_feature_schema_version",
    "dimming_window_method_version",
    "recovery_feature_lc_path",
    "recovery_feature_lc_size_bytes",
    "recovery_feature_lc_mtime_ns",
)

_BOOLEAN_RECOVERY_FEATURES = {
    "dimming_complex_is_lower_limit",
    "left_baseline_recovered",
    "right_baseline_recovered",
}
_NO_SUPPORTED_EVENT_MESSAGE = (
    "no recovery-anchored dimming bracket with a supported event seed"
)


def _numeric_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index, dtype="float64")
    return pd.to_numeric(table[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def add_next_iteration_context_features(table: pd.DataFrame) -> pd.DataFrame:
    """Attach the selected static context without using external labels.

    Reduced proper motion follows

    ``H_G = G + 5 log10(mu / 1000) + 5``

    for total proper motion ``mu`` in mas/yr.  Non-positive or incomplete
    astrometry remains missing rather than being assigned a sentinel value.
    """

    out = table.copy()
    for column in ("A_v_3d", "derived_wjk", "iphas_r_ha", "vphas_r_ha"):
        out[column] = _numeric_column(out, column)

    pmra = _numeric_column(out, "pmra")
    pmdec = _numeric_column(out, "pmdec")
    g_mag = _numeric_column(out, "phot_g_mean_mag")
    proper_motion_mas_yr = np.hypot(pmra, pmdec)
    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = g_mag + 5.0 * np.log10(proper_motion_mas_yr / 1000.0) + 5.0
    out["reduced_proper_motion_g"] = pd.Series(reduced, index=out.index).where(
        np.isfinite(reduced) & (proper_motion_mas_yr > 0)
    )
    return out


def default_recovery_feature_cache(db_path: str | Path) -> Path:
    """Return the run-local recovery-feature cache for a Review database."""

    resolved = Path(db_path).expanduser().resolve()
    if resolved.parent.name == "review":
        run_dir = resolved.parent.parent
    else:
        run_dir = resolved.parent
    return (
        run_dir
        / "results"
        / "ml_feature_cache"
        / "recovery_bounded_event_features.parquet"
    )


def _lightcurve_fingerprint(lc_path: object) -> tuple[str, int, int]:
    text = "" if lc_path is None else str(lc_path).strip()
    if not text:
        return "", -1, -1
    path = Path(text).expanduser().resolve()
    try:
        stat = path.stat()
    except OSError:
        return str(path), -1, -1
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


def _empty_recovery_row(
    candidate_id: str,
    *,
    lc_path: str,
    lc_size_bytes: int,
    lc_mtime_ns: int,
    state: str,
    error: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": str(candidate_id),
        "recovery_feature_state": state,
        "recovery_feature_error": error,
        "recovery_feature_schema_version": RECOVERY_FEATURE_SCHEMA_VERSION,
        "dimming_window_method_version": DIMMING_WINDOW_METHOD_VERSION,
        "recovery_feature_lc_path": lc_path,
        "recovery_feature_lc_size_bytes": lc_size_bytes,
        "recovery_feature_lc_mtime_ns": lc_mtime_ns,
    }
    row.update({column: np.nan for column in RECOVERY_BOUNDED_EVENT_FEATURES})
    return row


def _measure_recovery_feature_task(
    task: tuple[str, str, int, int],
) -> dict[str, Any]:
    candidate_id, lc_path, lc_size_bytes, lc_mtime_ns = task
    if not lc_path or lc_size_bytes < 0:
        return _empty_recovery_row(
            candidate_id,
            lc_path=lc_path,
            lc_size_bytes=lc_size_bytes,
            lc_mtime_ns=lc_mtime_ns,
            state="no_lightcurve",
            error="native light curve is missing or unreadable",
        )
    try:
        measurement = measure_dimming_complex_window(candidate_id, lc_path)
        times = measurement.epochs["t"].to_numpy(float)
        metrics = measurement.window.to_metrics(times)
        row = _empty_recovery_row(
            candidate_id,
            lc_path=lc_path,
            lc_size_bytes=lc_size_bytes,
            lc_mtime_ns=lc_mtime_ns,
            state="measured",
            error="",
        )
        for column in RECOVERY_BOUNDED_EVENT_FEATURES:
            row[column] = metrics.get(column, np.nan)
        return row
    except RuntimeError as exc:
        message = str(exc)
        state = (
            "no_supported_event"
            if _NO_SUPPORTED_EVENT_MESSAGE in message
            else "error"
        )
        return _empty_recovery_row(
            candidate_id,
            lc_path=lc_path,
            lc_size_bytes=lc_size_bytes,
            lc_mtime_ns=lc_mtime_ns,
            state=state,
            error=f"{type(exc).__name__}: {message}",
        )
    except Exception as exc:  # keep a resumable per-candidate audit row
        return _empty_recovery_row(
            candidate_id,
            lc_path=lc_path,
            lc_size_bytes=lc_size_bytes,
            lc_mtime_ns=lc_mtime_ns,
            state="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _read_recovery_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.is_file():
        return pd.DataFrame()
    cache = pd.read_parquet(cache_path)
    required = {
        "candidate_id",
        *RECOVERY_BOUNDED_EVENT_FEATURES,
        *RECOVERY_FEATURE_PROVENANCE_COLUMNS,
    }
    missing = sorted(required.difference(cache.columns))
    if missing:
        raise ValueError(
            f"Recovery-feature cache has an incompatible schema ({cache_path}): "
            f"missing {missing}"
        )
    cache["candidate_id"] = cache["candidate_id"].astype(str)
    if cache["candidate_id"].duplicated().any():
        raise ValueError(f"Recovery-feature cache has duplicate candidate IDs: {cache_path}")
    return cache


def _write_recovery_cache(cache: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.temporary"
    )
    try:
        cache.sort_values("candidate_id", ignore_index=True).to_parquet(
            temporary, index=False
        )
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_row_is_current(
    row: dict[str, Any],
    *,
    lc_path: str,
    lc_size_bytes: int,
    lc_mtime_ns: int,
) -> bool:
    return (
        str(row.get("dimming_window_method_version", ""))
        == DIMMING_WINDOW_METHOD_VERSION
        and str(row.get("recovery_feature_schema_version", ""))
        == RECOVERY_FEATURE_SCHEMA_VERSION
        and str(row.get("recovery_feature_lc_path", "")) == lc_path
        and int(row.get("recovery_feature_lc_size_bytes", -2)) == lc_size_bytes
        and int(row.get("recovery_feature_lc_mtime_ns", -2)) == lc_mtime_ns
    )


def ensure_recovery_feature_cache(
    table: pd.DataFrame,
    cache_path: str | Path,
    *,
    workers: int = 4,
    checkpoint_every: int = 250,
    compute_missing: bool = True,
) -> pd.DataFrame:
    """Return a current, resumable recovery-feature cache for ``table``."""

    if "candidate_id" not in table.columns or "lc_path" not in table.columns:
        raise KeyError("Recovery features require candidate_id and lc_path columns")
    candidate_ids = table["candidate_id"].astype(str)
    if candidate_ids.duplicated().any():
        raise ValueError("Recovery features require unique candidate IDs")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least one")

    resolved_cache = Path(cache_path).expanduser().resolve()
    cache = _read_recovery_cache(resolved_cache)
    records = {
        str(row["candidate_id"]): row
        for row in cache.to_dict(orient="records")
    }
    requested: list[tuple[str, str, int, int]] = []
    for candidate_id, lc_path in zip(candidate_ids, table["lc_path"]):
        fingerprint = _lightcurve_fingerprint(lc_path)
        cached = records.get(candidate_id)
        if cached is None or not _cache_row_is_current(
            cached,
            lc_path=fingerprint[0],
            lc_size_bytes=fingerprint[1],
            lc_mtime_ns=fingerprint[2],
        ):
            requested.append((candidate_id, *fingerprint))

    if requested and not compute_missing:
        raise FileNotFoundError(
            f"{len(requested):,} candidates are absent or stale in recovery-feature "
            f"cache {resolved_cache}"
        )

    if requested:
        print(
            f"Recovery morphology: measuring {len(requested):,} missing/stale "
            f"candidates with {workers} worker(s)",
            flush=True,
        )
        if workers == 1:
            measured: Iterable[dict[str, Any]] = map(
                _measure_recovery_feature_task, requested
            )
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            measured = executor.map(
                _measure_recovery_feature_task, requested, chunksize=1
            )
        try:
            for index, row in enumerate(measured, start=1):
                records[str(row["candidate_id"])] = row
                if index % checkpoint_every == 0:
                    _write_recovery_cache(
                        pd.DataFrame.from_records(list(records.values())),
                        resolved_cache,
                    )
                    print(
                        f"Recovery morphology: cached {index:,}/{len(requested):,}",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown()
        cache = pd.DataFrame.from_records(list(records.values()))
        _write_recovery_cache(cache, resolved_cache)
    elif cache.empty:
        cache = pd.DataFrame.from_records(list(records.values()))

    requested_cache = cache.loc[
        cache["candidate_id"].astype(str).isin(set(candidate_ids))
    ].copy()
    if len(requested_cache) != len(table):
        raise RuntimeError(
            "Recovery-feature cache did not produce exactly one row per candidate"
        )
    return requested_cache


def add_recovery_bounded_event_features(
    table: pd.DataFrame,
    cache_path: str | Path,
    *,
    workers: int = 4,
    checkpoint_every: int = 250,
    compute_missing: bool = True,
) -> pd.DataFrame:
    """Merge current recovery morphology onto a candidate table."""

    cache = ensure_recovery_feature_cache(
        table,
        cache_path,
        workers=workers,
        checkpoint_every=checkpoint_every,
        compute_missing=compute_missing,
    )
    merge_columns = [
        "candidate_id",
        *RECOVERY_BOUNDED_EVENT_FEATURES,
        *RECOVERY_FEATURE_PROVENANCE_COLUMNS,
    ]
    out = table.drop(
        columns=[
            *RECOVERY_BOUNDED_EVENT_FEATURES,
            *RECOVERY_FEATURE_PROVENANCE_COLUMNS,
        ],
        errors="ignore",
    ).copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    cache = cache[merge_columns].copy()
    cache["candidate_id"] = cache["candidate_id"].astype(str)
    out = out.merge(cache, on="candidate_id", how="left", validate="one_to_one")
    for column in _BOOLEAN_RECOVERY_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    return out


__all__ = [
    "NEXT_ITERATION_CONTEXT_FEATURES",
    "RECOVERY_BOUNDED_EVENT_FEATURES",
    "RECOVERY_FEATURE_SCHEMA_VERSION",
    "RECOVERY_FEATURE_PROVENANCE_COLUMNS",
    "add_next_iteration_context_features",
    "add_recovery_bounded_event_features",
    "default_recovery_feature_cache",
    "ensure_recovery_feature_cache",
]
