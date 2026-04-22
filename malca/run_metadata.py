from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import shutil


def preserve_imported_run_snapshots(
    *,
    stage: str,
    import_bundle: Path | None,
    out_dir: Path,
    run_params_file: Path,
    run_summary_file: Path,
) -> tuple[Path | None, Path | None]:
    """Snapshot imported bundle metadata before a home-stage replay overwrites it."""
    if stage != "home" or import_bundle is None:
        return None, None

    imported_run_params_snapshot: Path | None = None
    imported_run_summary_snapshot: Path | None = None

    if run_params_file.exists():
        imported_run_params_snapshot = out_dir / "run_params_imported.json"
        if not imported_run_params_snapshot.exists():
            shutil.copy2(run_params_file, imported_run_params_snapshot)

    if run_summary_file.exists():
        imported_run_summary_snapshot = out_dir / "run_summary_imported.json"
        if not imported_run_summary_snapshot.exists():
            shutil.copy2(run_summary_file, imported_run_summary_snapshot)

    return imported_run_params_snapshot, imported_run_summary_snapshot


def load_summary_state(
    *,
    run_summary_file: Path,
    run_start_time: datetime,
    stage: str,
) -> dict[str, object]:
    """Load existing summary metadata and stamp the current run start info."""
    summary_state: dict[str, object] = {
        "run_info": {
            "start_time": run_start_time.isoformat(),
            "stage": stage,
        }
    }
    if not run_summary_file.exists():
        return summary_state

    try:
        existing_summary = json.loads(run_summary_file.read_text())
    except Exception:
        return summary_state

    existing_summary.update(summary_state)
    return existing_summary


def build_run_summary(
    *,
    previous_summary: dict[str, object] | None,
    run_start_time: datetime,
    run_end_time: datetime,
    config_fingerprint: dict[str, object],
    run_upstream: bool,
    manifest_total_sources: int | None,
    manifest_filtered_sources: int | None,
    artifact_context: dict[str, Any],
) -> dict[str, object]:
    """Merge final run-summary state while preserving imported bundle metadata."""
    summary = dict(previous_summary or {})
    previous_manifest_stats = (
        summary.get("manifest_stats", {})
        if isinstance(summary.get("manifest_stats", {}), dict)
        else {}
    )

    if not run_upstream:
        manifest_total_sources = _coerce_int(
            manifest_total_sources,
            fallback=previous_manifest_stats.get("total_sources"),
        )
        manifest_filtered_sources = _coerce_int(
            manifest_filtered_sources,
            fallback=previous_manifest_stats.get("filtered_sources"),
        )
    else:
        manifest_total_sources = _coerce_int(manifest_total_sources)
        manifest_filtered_sources = _coerce_int(manifest_filtered_sources)

    kept_fraction = _compute_fraction(
        numerator=manifest_filtered_sources,
        denominator=manifest_total_sources,
    )

    summary.update(
        {
            "run_info": {
                "start_time": run_start_time.isoformat(),
                "end_time": run_end_time.isoformat(),
                "duration_seconds": (run_end_time - run_start_time).total_seconds(),
            },
            "config_fingerprint": config_fingerprint,
            "manifest_stats": {
                "total_sources": manifest_total_sources,
                "filtered_sources": manifest_filtered_sources,
                "kept_fraction": kept_fraction,
            },
            "artifact_context": dict(artifact_context),
        }
    )
    return summary


def _coerce_int(value: object, *, fallback: object = None) -> int | None:
    candidate = value if value is not None else fallback
    if candidate is None:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def _compute_fraction(*, numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)
