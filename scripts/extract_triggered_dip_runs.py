#!/usr/bin/env python3
"""Replay the July 1 detector and persist one row per triggered dip run.

The July 1 candidate table persisted run counts but not the individual run
epochs.  This script recovers those epochs with the exact detector source
snapshot recorded by the run.  It extracts that snapshot from git into a
temporary directory, relaunches itself against the historical package, and
writes resumable parquet parts before assembling the final run table. Tiny
source-count differences caused by replay-environment numerical drift are
retained and explicitly reported instead of replacing runs with best points.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from typing import Any


DEFAULT_DETECTOR_COMMIT = "74adcdb6fb0523ae40c181fe8ca54e424647ba08"
DEFAULT_EVENTS_SHA256 = "7bbd252a69cfc8d081cd8128683a5b0bcbbab4b9711d7469dc33ad42c9ab36e8"
ASASSN_REDUCED_JD_OFFSET = 2_450_000.0
MJD_OFFSET = 2_400_000.5
RUN_TABLE_SCHEMA_VERSION = 2

_WORKER_CONFIG: dict[str, Any] | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--detector-commit", default=DEFAULT_DETECTOR_COMMIT)
    parser.add_argument("--events-sha256", default=DEFAULT_EVENTS_SHA256)
    parser.add_argument("--historical-runtime", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in (target, *target.parents):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def _prepare_historical_snapshot(repo_root: Path, commit: str, expected_hash: str) -> Path:
    snapshot_root = Path("/tmp") / f"malca-triggered-dip-runs-{commit[:12]}"
    events_path = snapshot_root / "malca" / "stv" / "events.py"
    if events_path.is_file() and _sha256(events_path) == expected_hash:
        return snapshot_root
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_root.mkdir(parents=True)
    archive_path = Path("/tmp") / f"malca-triggered-dip-runs-{commit[:12]}.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive_path}", commit],
        cwd=repo_root,
        check=True,
    )
    _safe_extract(archive_path, snapshot_root)
    archive_path.unlink(missing_ok=True)
    actual_hash = _sha256(events_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Historical detector hash mismatch: "
            f"expected {expected_hash}, got {actual_hash} from {commit}"
        )
    return snapshot_root


def _relaunch_with_historical_package(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    snapshot_root = _prepare_historical_snapshot(
        repo_root,
        str(args.detector_commit),
        str(args.events_sha256),
    )
    command = [*sys.argv, "--historical-runtime"]
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(snapshot_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    environment.setdefault("MPLCONFIGDIR", "/tmp/malca-triggered-dip-runs-mpl")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    environment.setdefault("NUMBA_NUM_THREADS", "1")
    completed = subprocess.run([sys.executable, *command], cwd=repo_root, env=environment)
    return int(completed.returncode)


def _load_candidates(run_root: Path, limit: int | None) -> list[dict[str, Any]]:
    database = run_root / "review" / "review.db"
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT candidate_id, dip_run_count
            FROM candidates
            WHERE CAST(COALESCE(dip_significant, 0) AS INTEGER) = 1
            ORDER BY candidate_id
            """
        ).fetchall()
    finally:
        connection.close()
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    lightcurve_dir = run_root / "bundle_assets" / "lightcurves"
    return [
        {
            "candidate_id": str(candidate_id),
            "expected_run_count": int(run_count),
            "lightcurve_path": str(lightcurve_dir / f"{str(candidate_id).removeprefix('stv_')}.dat3"),
        }
        for candidate_id, run_count in rows
    ]


def _load_quiet_exposure_candidates(run_root: Path, limit: int | None) -> list[dict[str, Any]]:
    """Return non-triggered sources needed only for unbiased exposure rates."""

    database = run_root / "review" / "review.db"
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT candidate_id
            FROM candidates
            WHERE CAST(COALESCE(dip_significant, 0) AS INTEGER) = 0
            ORDER BY candidate_id
            """
        ).fetchall()
    finally:
        connection.close()
    if limit is not None:
        return []
    lightcurve_dir = run_root / "bundle_assets" / "lightcurves"
    return [
        {
            "candidate_id": str(candidate_id),
            "lightcurve_path": str(lightcurve_dir / f"{str(candidate_id).removeprefix('stv_')}.dat3"),
        }
        for (candidate_id,) in rows
    ]


def _detector_config(run_root: Path) -> dict[str, Any]:
    from malca.stv.events import EVENTS_CONFIG_DEFAULTS

    saved = json.loads((run_root / "run_params.json").read_text())

    def get(name: str) -> Any:
        value = saved.get(name)
        return EVENTS_CONFIG_DEFAULTS.get(name) if value is None else value

    baseline_kwargs = {
        "S0": get("baseline_s0"),
        "w0": get("baseline_w0"),
        "q": get("baseline_q"),
        "jitter": get("baseline_jitter"),
        "sigma_floor": get("baseline_sigma_floor"),
        "add_sigma_eff_col": True,
    }
    return {
        "trigger_mode": get("trigger_mode"),
        "logbf_threshold_dip": get("logbf_threshold_dip"),
        "logbf_threshold_jump": get("logbf_threshold_jump"),
        "significance_threshold": get("significance_threshold"),
        "p_points": int(get("p_points")),
        "p_min_dip": get("p_min_dip"),
        "p_max_dip": get("p_max_dip"),
        "p_min_jump": get("p_min_jump"),
        "p_max_jump": get("p_max_jump"),
        "mag_points": int(get("mag_points")),
        "mag_min_dip": get("mag_min_dip"),
        "mag_max_dip": get("mag_max_dip"),
        "mag_min_jump": get("mag_min_jump"),
        "mag_max_jump": get("mag_max_jump"),
        "run_min_points": int(get("run_min_points")),
        "max_gap_points": int(get("run_max_gap_points")),
        "run_max_gap_days": get("run_max_gap_days"),
        "run_min_duration_days": get("run_min_duration_days"),
        "baseline_tag": get("baseline_func"),
        "baseline_kwargs": baseline_kwargs,
        "compute_event_prob": not bool(get("no_event_prob")),
        "auto_filter_bad_cameras": bool(get("filter_bad_cameras")),
        "bad_camera_scatter_ratio": float(get("bad_camera_scatter_ratio")),
    }


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)


def _modal(values: list[str]) -> str | None:
    cleaned = [str(value) for value in values if str(value).strip() not in {"", "nan", "None", "<NA>"}]
    if not cleaned:
        return None
    counts: dict[str, int] = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


def _modal_pair(first: list[Any], second: list[Any]) -> tuple[str | None, str | None]:
    pairs = [
        (str(left), str(right))
        for left, right in zip(first, second)
        if str(left).strip() not in {"", "nan", "None", "<NA>"}
        and str(right).strip() not in {"", "nan", "None", "<NA>"}
    ]
    if not pairs:
        return _modal(first), _modal(second)
    counts: dict[tuple[str, str], int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    return min(counts, key=lambda pair: (-counts[pair], pair))


def _json_list(values: list[Any]) -> str:
    return json.dumps(values, separators=(",", ":"), allow_nan=False)


def _float_or_nan(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return math.nan
    return converted if math.isfinite(converted) else math.nan


def _load_detector_frame(path: Path, config: dict[str, Any]) -> Any:
    import numpy as np
    import pandas as pd

    from malca.core.utils import filter_bad_cameras, read_lc_dat2

    dfg, dfv = read_lc_dat2(path.stem, str(path.parent), excluded_cameras=None, file_ext=path.suffix.lstrip("."))
    frame = pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()
    if frame.empty:
        raise ValueError("empty light curve")
    valid = (
        np.isfinite(frame["JD"])
        & np.isfinite(frame["mag"])
        & np.isfinite(frame["error"])
        & frame["error"].gt(0)
        & frame["error"].lt(10)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("no detector-valid observations")
    if config["auto_filter_bad_cameras"] and "camera#" in frame:
        frame, _ = filter_bad_cameras(
            frame,
            lc_path=str(path),
            filter_scatter=False,
            filter_offset=False,
            filter_catastrophic=True,
            scatter_ratio_threshold=config["bad_camera_scatter_ratio"],
        )
    return frame


def _score_path(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from malca.core.baseline import (
        global_median_baseline,
        per_camera_gp_baseline,
        per_camera_gp_baseline_masked,
        per_camera_median_baseline,
        phase_template_baseline,
    )
    from malca.stv.events import score_lightcurve

    frame = _load_detector_frame(path, config)
    baseline_functions = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
        "phase_template": phase_template_baseline,
    }
    baseline_function = baseline_functions.get(config["baseline_tag"], per_camera_gp_baseline)
    mag_grid_dip = None
    if config["mag_min_dip"] is not None and config["mag_max_dip"] is not None:
        mag_grid_dip = np.linspace(config["mag_min_dip"], config["mag_max_dip"], config["mag_points"])
    mag_grid_jump = None
    if config["mag_min_jump"] is not None and config["mag_max_jump"] is not None:
        mag_grid_jump = np.linspace(config["mag_min_jump"], config["mag_max_jump"], config["mag_points"])
    return score_lightcurve(
        frame,
        baseline_func=baseline_function,
        baseline_kwargs=config["baseline_kwargs"],
        filter_residual_bad_cameras_enabled=config["auto_filter_bad_cameras"],
        bad_camera_scatter_ratio=config["bad_camera_scatter_ratio"],
        p_points=config["p_points"],
        mag_points=config["mag_points"],
        trigger_mode=config["trigger_mode"],
        logbf_threshold_dip=config["logbf_threshold_dip"],
        logbf_threshold_jump=config["logbf_threshold_jump"],
        significance_threshold=config["significance_threshold"],
        run_min_points=config["run_min_points"],
        max_gap_points=config["max_gap_points"],
        run_max_gap_days=config["run_max_gap_days"],
        run_min_duration_days=config["run_min_duration_days"],
        compute_event_prob=config["compute_event_prob"],
        p_min_dip=config["p_min_dip"],
        p_max_dip=config["p_max_dip"],
        p_min_jump=config["p_min_jump"],
        p_max_jump=config["p_max_jump"],
        mag_grid_dip=mag_grid_dip,
        mag_grid_jump=mag_grid_jump,
    )


def _exposure_rows_from_frame(frame: Any, candidate_id: str) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd

    exposure_frame = frame.copy()
    exposure_frame["night_mjd"] = np.floor(
        pd.to_numeric(exposure_frame["JD"], errors="coerce")
        + ASASSN_REDUCED_JD_OFFSET
        - MJD_OFFSET
    ).astype("Int64")
    rows: list[dict[str, Any]] = []
    for (camera_name, field), group in exposure_frame.groupby(
        ["camera_name", "field"], observed=True, dropna=False
    ):
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_key": candidate_id,
                "camera_name_key": None if pd.isna(camera_name) else str(camera_name),
                "asassn_field_key": None if pd.isna(field) else str(field),
                "n_observations": int(len(group)),
                "n_observed_nights": int(group["night_mjd"].nunique(dropna=True)),
                "first_reduced_jd": float(pd.to_numeric(group["JD"], errors="coerce").min()),
                "last_reduced_jd": float(pd.to_numeric(group["JD"], errors="coerce").max()),
            }
        )
    return rows


def _extract_exposure_only(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("worker detector configuration was not initialized")
    candidate_id = str(task["candidate_id"])
    path = Path(task["lightcurve_path"])
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = _load_detector_frame(path, _WORKER_CONFIG)
        return {
            "rows": _exposure_rows_from_frame(frame, candidate_id),
            "status": {"candidate_id": candidate_id, "status": "ok", "error": None},
        }
    except Exception as exc:
        return {
            "rows": [],
            "status": {
                "candidate_id": candidate_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _extract_one(task: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    if _WORKER_CONFIG is None:
        raise RuntimeError("worker detector configuration was not initialized")
    candidate_id = str(task["candidate_id"])
    expected_count = int(task["expected_run_count"])
    path = Path(task["lightcurve_path"])
    started = time.perf_counter()
    status = {
        "candidate_id": candidate_id,
        "lightcurve_path": str(path),
        "expected_run_count": expected_count,
        "replayed_run_count": 0,
        "status": "error",
        "error": None,
        "elapsed_seconds": math.nan,
    }
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        result = _score_path(path, _WORKER_CONFIG)
        frame = result["df"].reset_index(drop=True)
        dip = result["dip"]
        summaries = sorted(dip.get("run_summaries", []), key=lambda row: (row["start_jd"], row["end_jd"]))
        status["replayed_run_count"] = len(summaries)
        count_matches = len(summaries) == expected_count
        event_indices = np.asarray(dip.get("event_indices", []), dtype=int)
        probabilities = dip.get("event_probability")
        probability_array = np.asarray(probabilities, dtype=float) if probabilities is not None else np.full(len(frame), np.nan)
        log_bf_array = np.asarray(dip.get("log_bf_local", np.full(len(frame), np.nan)), dtype=float)
        exposure_rows = _exposure_rows_from_frame(frame, candidate_id)
        rows: list[dict[str, Any]] = []
        for run_number, summary in enumerate(summaries, start=1):
            start_index = int(summary["start_idx"])
            end_index = int(summary["end_idx"])
            run_indices = event_indices[(event_indices >= start_index) & (event_indices <= end_index)]
            if len(run_indices) != int(summary["n_points"]):
                raise RuntimeError(
                    f"run {run_number}: summary has {summary['n_points']} points but recovered {len(run_indices)}"
                )
            if run_indices.size == 0:
                raise RuntimeError(f"run {run_number}: no triggered indices")
            reduced_jds = pd.to_numeric(frame.iloc[run_indices]["JD"], errors="coerce").to_numpy(float)
            full_jds = reduced_jds + ASASSN_REDUCED_JD_OFFSET
            cameras = frame.iloc[run_indices].get("camera_name", pd.Series(pd.NA, index=run_indices)).astype("string").tolist()
            fields = frame.iloc[run_indices].get("field", pd.Series(pd.NA, index=run_indices)).astype("string").tolist()
            camera_ids = frame.iloc[run_indices].get("camera#", pd.Series(pd.NA, index=run_indices)).astype("string").tolist()
            modal_field, modal_camera = _modal_pair(fields, cameras)
            start_reduced = float(summary["start_jd"])
            end_reduced = float(summary["end_jd"])
            fit_params = summary.get("params") if isinstance(summary.get("params"), dict) else {}
            fit_t0 = pd.to_numeric(pd.Series([fit_params.get("t0")]), errors="coerce").iloc[0]
            fit_t0_valid = bool(
                np.isfinite(fit_t0) and start_reduced <= float(fit_t0) <= end_reduced
            )
            ranking = probability_array[run_indices]
            if not np.isfinite(ranking).any():
                ranking = log_bf_array[run_indices]
            peak_offset = int(np.nanargmax(ranking)) if np.isfinite(ranking).any() else 0
            trigger_peak_index = int(run_indices[peak_offset])
            if fit_t0_valid:
                peak_reduced = float(fit_t0)
                peak_index = int(run_indices[np.nanargmin(np.abs(reduced_jds - peak_reduced))])
                peak_time_source = "morphology_fit_t0"
            else:
                peak_index = trigger_peak_index
                peak_reduced = float(frame.iloc[peak_index]["JD"])
                peak_time_source = "max_trigger_sample"
            peak_jd = peak_reduced + ASASSN_REDUCED_JD_OFFSET
            peak_mjd = peak_jd - MJD_OFFSET
            rows.append(
                {
                    "event_id": f"{candidate_id}:dip_run_{run_number:04d}",
                    "candidate_id": candidate_id,
                    "source_key": candidate_id,
                    "run_number": run_number,
                    "run_start_reduced_jd": start_reduced,
                    "run_end_reduced_jd": end_reduced,
                    "run_peak_reduced_jd": peak_reduced,
                    "run_peak_sample_reduced_jd": float(frame.iloc[peak_index]["JD"]),
                    "run_peak_time_source": peak_time_source,
                    "run_start_jd": start_reduced + ASASSN_REDUCED_JD_OFFSET,
                    "run_end_jd": end_reduced + ASASSN_REDUCED_JD_OFFSET,
                    "dip_jd": peak_jd,
                    "dip_mjd": peak_mjd,
                    "dip_night_mjd": int(math.floor(peak_mjd)),
                    "run_duration_days": float(summary["duration_days"]),
                    "n_trigger_points": int(summary["n_points"]),
                    "run_trigger_max": float(summary["run_max"]),
                    "run_trigger_sum": float(summary["run_sum"]),
                    "run_n_cameras": int(summary.get("run_n_cameras") or 0),
                    "run_peak_event_probability": float(probability_array[peak_index]) if np.isfinite(probability_array[peak_index]) else math.nan,
                    "run_peak_log_bf_local": float(log_bf_array[peak_index]) if np.isfinite(log_bf_array[peak_index]) else math.nan,
                    "run_peak_delta_mag": float(summary.get("peak_delta_mag", math.nan)),
                    "run_fit_amp": _float_or_nan(fit_params.get("amp")),
                    "run_fit_width_param": _float_or_nan(
                        fit_params.get("sigma", fit_params.get("tE", fit_params.get("width")))
                    ),
                    "run_morphology": str(summary.get("morphology", "none")),
                    "run_delta_bic_null": float(summary.get("delta_bic_null", math.nan)),
                    "run_symmetry_score": float(summary.get("symmetry_score", math.nan)),
                    "asassn_field_key": modal_field,
                    "camera_name_key": modal_camera,
                    "run_fields": ",".join(sorted({str(value) for value in fields if pd.notna(value)})),
                    "run_camera_names": ",".join(sorted({str(value) for value in cameras if pd.notna(value)})),
                    "run_camera_ids": ",".join(sorted({str(value) for value in camera_ids if pd.notna(value)})),
                    "trigger_jds_json": _json_list([float(value) for value in full_jds]),
                    "trigger_cameras_json": _json_list([None if pd.isna(value) else str(value) for value in cameras]),
                    "trigger_fields_json": _json_list([None if pd.isna(value) else str(value) for value in fields]),
                    "lc_path": str(path),
                    "detector_commit": DEFAULT_DETECTOR_COMMIT,
                    "run_table_schema_version": RUN_TABLE_SCHEMA_VERSION,
                }
            )
        status["status"] = "ok" if count_matches else "count_mismatch"
        if not count_matches:
            status["error"] = (
                f"stored run count {expected_count} != replayed run count {len(summaries)}; "
                "retained replayed intervals"
            )
        status["elapsed_seconds"] = time.perf_counter() - started
        return {"rows": rows, "exposures": exposure_rows, "status": status}
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["elapsed_seconds"] = time.perf_counter() - started
        return {"rows": [], "exposures": [], "status": status}


def _atomic_parquet(frame: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _historical_main(args: argparse.Namespace) -> int:
    import pandas as pd

    run_root = args.run_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.workers < 1 or args.batch_size < 1:
        raise SystemExit("--workers and --batch-size must be positive")
    tasks = _load_candidates(run_root, args.limit)
    config = _detector_config(run_root)
    parts_dir = output.parent / f"{output.stem}.v{RUN_TABLE_SCHEMA_VERSION}.parts"
    fingerprint_path = parts_dir / "fingerprint.json"
    fingerprint = {
        "detector_commit": str(args.detector_commit),
        "events_sha256": str(args.events_sha256),
        "run_table_schema_version": RUN_TABLE_SCHEMA_VERSION,
        "run_root": str(run_root),
        "n_sources": len(tasks),
        "limit": args.limit,
        "detector_config": config,
    }
    if args.overwrite and parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    if fingerprint_path.exists():
        existing = json.loads(fingerprint_path.read_text())
        if existing != fingerprint:
            raise SystemExit(f"Checkpoint fingerprint mismatch: {fingerprint_path}; use --overwrite")
    else:
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")

    n_batches = math.ceil(len(tasks) / args.batch_size) if tasks else 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(config,)) as executor:
        for batch_index in range(n_batches):
            run_part = parts_dir / f"runs_{batch_index:05d}.parquet"
            status_part = parts_dir / f"status_{batch_index:05d}.parquet"
            exposure_part = parts_dir / f"exposures_{batch_index:05d}.parquet"
            if run_part.is_file() and exposure_part.is_file() and status_part.is_file():
                cached_status = pd.read_parquet(status_part)
                if cached_status["status"].isin({"ok", "count_mismatch"}).all():
                    print(f"batch {batch_index + 1}/{n_batches}: cached", flush=True)
                    continue
            batch = tasks[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
            results = list(executor.map(_extract_one, batch, chunksize=1))
            run_rows = [row for result in results for row in result["rows"]]
            exposure_rows = [row for result in results for row in result["exposures"]]
            statuses = [result["status"] for result in results]
            _atomic_parquet(pd.DataFrame(run_rows), run_part)
            _atomic_parquet(pd.DataFrame(exposure_rows), exposure_part)
            _atomic_parquet(pd.DataFrame(statuses), status_part)
            usable_count = sum(status["status"] in {"ok", "count_mismatch"} for status in statuses)
            print(
                f"batch {batch_index + 1}/{n_batches}: {usable_count}/{len(batch)} usable sources, "
                f"{len(run_rows):,} runs, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    quiet_tasks = _load_quiet_exposure_candidates(run_root, args.limit)
    quiet_dir = parts_dir / "quiet_exposures"
    quiet_dir.mkdir(parents=True, exist_ok=True)
    quiet_fingerprint_path = quiet_dir / "fingerprint.json"
    quiet_fingerprint = {
        "run_root": str(run_root),
        "run_table_schema_version": RUN_TABLE_SCHEMA_VERSION,
        "n_sources": len(quiet_tasks),
        "detector_input_filter": {
            "auto_filter_bad_cameras": config["auto_filter_bad_cameras"],
            "bad_camera_scatter_ratio": config["bad_camera_scatter_ratio"],
        },
    }
    if quiet_fingerprint_path.exists():
        existing = json.loads(quiet_fingerprint_path.read_text())
        if existing != quiet_fingerprint:
            raise SystemExit(
                f"Quiet-exposure checkpoint fingerprint mismatch: {quiet_fingerprint_path}; use --overwrite"
            )
    else:
        quiet_fingerprint_path.write_text(json.dumps(quiet_fingerprint, indent=2, sort_keys=True) + "\n")
    quiet_batches = math.ceil(len(quiet_tasks) / args.batch_size) if quiet_tasks else 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(config,)) as executor:
        for batch_index in range(quiet_batches):
            exposure_part = quiet_dir / f"exposures_{batch_index:05d}.parquet"
            status_part = quiet_dir / f"status_{batch_index:05d}.parquet"
            if exposure_part.is_file() and status_part.is_file():
                cached_status = pd.read_parquet(status_part)
                if cached_status["status"].eq("ok").all():
                    print(f"quiet exposure batch {batch_index + 1}/{quiet_batches}: cached", flush=True)
                    continue
            batch = quiet_tasks[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
            results = list(executor.map(_extract_exposure_only, batch, chunksize=1))
            exposure_rows = [row for result in results for row in result["rows"]]
            statuses = [result["status"] for result in results]
            _atomic_parquet(pd.DataFrame(exposure_rows), exposure_part)
            _atomic_parquet(pd.DataFrame(statuses), status_part)
            print(
                f"quiet exposure batch {batch_index + 1}/{quiet_batches}: "
                f"{sum(status['status'] == 'ok' for status in statuses)}/{len(batch)} sources",
                flush=True,
            )

    quiet_status_parts = sorted(quiet_dir.glob("status_*.parquet"))
    quiet_exposure_parts = sorted(quiet_dir.glob("exposures_*.parquet"))
    quiet_statuses = (
        pd.concat([pd.read_parquet(path) for path in quiet_status_parts], ignore_index=True)
        if quiet_status_parts else pd.DataFrame(columns=["candidate_id", "status", "error"])
    )
    quiet_failures = quiet_statuses.loc[~quiet_statuses["status"].eq("ok")]
    if len(quiet_statuses) != len(quiet_tasks) or not quiet_failures.empty:
        examples = quiet_failures[["candidate_id", "error"]].head(10).to_dict(orient="records")
        raise SystemExit(
            f"Quiet-source exposure scan incomplete: statuses={len(quiet_statuses)}/{len(quiet_tasks)}, "
            f"failures={len(quiet_failures)}; examples={examples}"
        )

    status_parts = sorted(parts_dir.glob("status_*.parquet"))
    run_parts = sorted(parts_dir.glob("runs_*.parquet"))
    exposure_parts = sorted(parts_dir.glob("exposures_*.parquet"))
    statuses = pd.concat([pd.read_parquet(path) for path in status_parts], ignore_index=True) if status_parts else pd.DataFrame()
    failures = statuses.loc[~statuses["status"].isin({"ok", "count_mismatch"})] if not statuses.empty else statuses
    count_mismatches = statuses.loc[statuses["status"].eq("count_mismatch")] if not statuses.empty else statuses
    _atomic_parquet(statuses, output.with_name(f"{output.stem}_replay_status.parquet"))
    statuses.to_csv(output.with_name(f"{output.stem}_replay_status.csv"), index=False)
    if len(statuses) != len(tasks) or not failures.empty:
        examples = failures[["candidate_id", "error"]].head(10).to_dict(orient="records") if not failures.empty else []
        raise SystemExit(
            f"Run replay incomplete: statuses={len(statuses)}/{len(tasks)}, failures={len(failures)}; examples={examples}"
        )
    runs = pd.concat([pd.read_parquet(path) for path in run_parts], ignore_index=True) if run_parts else pd.DataFrame()
    exposure_frames = [pd.read_parquet(path) for path in exposure_parts]
    exposure_frames.extend(pd.read_parquet(path) for path in quiet_exposure_parts)
    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    expected_total = int(statuses["expected_run_count"].sum())
    replayed_total = int(statuses["replayed_run_count"].sum())
    if len(runs) != replayed_total or runs["event_id"].duplicated().any():
        raise SystemExit(
            f"Run-table integrity failed: rows={len(runs):,}, replayed={replayed_total:,}, "
            f"duplicate IDs={int(runs['event_id'].duplicated().sum()) if not runs.empty else 0}"
        )
    runs = runs.sort_values(["dip_mjd", "candidate_id", "run_number"], kind="mergesort").reset_index(drop=True)
    _atomic_parquet(runs, output)
    exposure_output = output.with_name(f"{output.stem}_exposures.parquet")
    exposures = exposures.sort_values(
        ["candidate_id", "camera_name_key", "asassn_field_key"], kind="mergesort"
    ).reset_index(drop=True)
    _atomic_parquet(exposures, exposure_output)
    manifest = {
        **fingerprint,
        "output": str(output),
        "exposure_output": str(exposure_output),
        "n_run_rows": len(runs),
        "n_exposure_rows": len(exposures),
        "n_exposure_sources": int(exposures["candidate_id"].nunique()) if not exposures.empty else 0,
        "n_quiet_exposure_sources": len(quiet_tasks),
        "expected_run_rows": expected_total,
        "replayed_run_rows": replayed_total,
        "run_count_delta": replayed_total - expected_total,
        "n_source_count_mismatches": len(count_mismatches),
        "n_sources": int(runs["candidate_id"].nunique()) if not runs.empty else 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {len(runs):,} triggered dip runs from {manifest['n_sources']:,} sources to {output}",
        flush=True,
    )
    if not count_mismatches.empty:
        print(
            f"warning: {len(count_mismatches):,} sources differ from their stored run count; "
            f"total replay delta={replayed_total - expected_total:+,}",
            flush=True,
        )
    return 0


def main() -> int:
    args = _parse_args()
    if not args.historical_runtime:
        return _relaunch_with_historical_package(args)
    return _historical_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
