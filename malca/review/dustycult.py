"""DustyCult fit integration for the MALCA review app."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    JD_OFFSET,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    MAG_POINTS,
    MJD_TO_JD,
    P_POINTS,
    RUN_MAX_GAP_POINTS,
    RUN_MIN_POINTS,
    SIGNIFICANCE_THRESHOLD,
    SKYPATROL_JD_OFFSET,
    TRIGGER_MODE,
)
from malca.core.phase import BAND_LABELS
from malca.stv.dimming_window import (
    DEFAULT_DIMMING_WINDOW_CONFIG,
    DIMMING_WINDOW_METHOD_VERSION,
    measure_dimming_complex_window,
)


DUSTYCULT_FIT_TABLE = "dustycult_fits"
DUSTYCULT_CURVE_TABLE = "dustycult_predictive_curves"

DUSTYCULT_FIT_COLUMNS = [
    "candidate_id",
    "mode",
    "status",
    "created_at",
    "updated_at",
    "runtime_sec",
    "artifact_dir",
    "input_path",
    "config_path",
    "manifest_path",
    "command_json",
    "config_json",
    "controls_json",
    "window_json",
    "stellar_json",
    "posterior_json",
    "summary_json",
    "stderr_tail",
    "stdout_tail",
    "error",
    "t0_jd",
    "start_jd",
    "end_jd",
    "n_input_points",
    "n_curve_points",
]

DUSTYCULT_CURVE_COLUMNS = [
    "candidate_id",
    "mode",
    "point_id",
    "time",
    "band",
    "observed",
    "error",
    "lower95",
    "lower68",
    "median",
    "upper68",
    "upper95",
]

DUSTYCULT_BANDPASS_NM = {
    "g": 477.0,
    "V": 545.0,
}

DUSTYCULT_MIN_POINTS = 12
DUSTYCULT_WARN_POINTS = 25
DUSTYCULT_WARN_SPAN_DAYS = 14.0
DUSTYCULT_MIN_SIDE_POINTS = 2
DUSTYCULT_REQUIRED_POSTERIOR = (
    "t0",
    "v",
    "b",
    "tau0",
    "lambda0",
    "alpha",
    "sigma_y",
    "sigma_x_plus",
    "sigma_x_minus",
)
DUSTYCULT_POSITIVE_POSTERIOR = {"v", "tau0", "lambda0", "sigma_y", "sigma_x_plus", "sigma_x_minus"}

QUICK_SAMPLING = {
    "n_samples": 200,
    "n_adapt": 200,
    "n_chains": 1,
    "grid_n": 51,
    "n_predictive_draws": 100,
}
FULL_SAMPLING = {
    "n_samples": 1000,
    "n_adapt": 1000,
    "n_chains": 4,
    "grid_n": 101,
    "n_predictive_draws": 200,
}
SAMPLING_BY_MODE = {
    "quick": QUICK_SAMPLING,
    "full": FULL_SAMPLING,
}

DEFAULT_CONTROLS = {
    "t0_width_days": 7.0,
    "log_v_width": 1.0,
    "b_center": 0.0,
    "b_width": 0.5,
    "log_tau0_width": 1.5,
    "alpha_center": 0.0,
    "alpha_width": 2.0,
    "log_sigma_width": 0.75,
    "star_R": 1.0,
    "star_u1": 0.0,
    "star_u2": 0.0,
}

DUSTYCULT_WINDOW_METADATA_FIELDS = (
    "method_version",
    "event_window_status",
    "dimming_complex_status",
    "dimming_complex_is_lower_limit",
    "left_boundary_type",
    "right_boundary_type",
)


@dataclass(frozen=True)
class DustyCultAvailability:
    ok: bool
    julia: str
    project_path: Path
    script_path: Path
    message: str


@dataclass(frozen=True)
class PreparedDustyCultInput:
    frame: pd.DataFrame
    window: dict[str, object]
    baseline_name: str
    baseline_warnings: list[str]
    quality: dict[str, object] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dustycult_project_path() -> Path:
    env_path = str(os.environ.get("MALCA_DUSTYCULT_PROJECT") or "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return _repo_root() / "external" / "dustycult"


def default_julia_executable() -> str:
    return str(os.environ.get("MALCA_JULIA") or "julia").strip() or "julia"


def default_dustycult_script_path(project_path: str | Path | None = None) -> Path:
    project = Path(project_path).expanduser() if project_path is not None else default_dustycult_project_path()
    return project / "scripts" / "fit_lightcurve.jl"


def check_dustycult_available(
    *,
    project_path: str | Path | None = None,
    julia: str | Path | None = None,
) -> DustyCultAvailability:
    project = Path(project_path).expanduser() if project_path is not None else default_dustycult_project_path()
    script = default_dustycult_script_path(project)
    julia_text = str(julia or default_julia_executable())
    julia_path = (
        julia_text
        if (Path(julia_text).exists() or shutil.which(julia_text))
        else ""
    )
    problems: list[str] = []
    if not project.exists():
        problems.append(f"DustyCult project not found at {project}")
    elif not (project / "Project.toml").exists():
        problems.append(f"DustyCult Project.toml not found at {project}")
    if not script.exists():
        problems.append(f"DustyCult fit CLI not found at {script}")
    if not julia_path:
        problems.append(f"Julia executable not found: {julia_text}")
    if problems:
        return DustyCultAvailability(False, julia_text, project, script, "; ".join(problems))
    return DustyCultAvailability(True, julia_text, project, script, "DustyCult is available")


def artifact_root_for_review_db(db_path: str | Path) -> Path:
    return Path(db_path).expanduser().resolve().parent / "dustycult"


def artifact_dir_for_candidate(db_path: str | Path, candidate_id: str, mode: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_id)).strip("_") or "unknown"
    safe_mode = normalize_mode(mode)
    return artifact_root_for_review_db(db_path) / safe_id / safe_mode


def normalize_mode(mode: object) -> str:
    text = str(mode or "quick").strip().lower()
    if text not in SAMPLING_BY_MODE:
        raise ValueError(f"Unsupported DustyCult mode: {mode}")
    return text


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        if value is None:
            return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _finite_or_none(value: object) -> float | None:
    return _safe_float(value, None)


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(value: object, default: Any) -> Any:
    if value is None:
        return default
    try:
        parsed = json.loads(str(value))
    except Exception:
        return default
    return parsed


def _sqlite_value(value: object) -> object:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    return value


def _time_value_to_lc_scale(value: object, lc_median: float | None) -> float | None:
    t = _safe_float(value)
    if t is None:
        return None
    if lc_median is None or not math.isfinite(lc_median):
        return t
    if lc_median > 2_000_000.0:
        if t > 2_000_000.0:
            return t
        if t > 50_000.0:
            return t + MJD_TO_JD
        return t + SKYPATROL_JD_OFFSET
    if lc_median > 50_000.0:
        if t > 2_000_000.0:
            return t - MJD_TO_JD
        if t < 50_000.0:
            return t + SKYPATROL_JD_OFFSET - MJD_TO_JD
        return t
    if t > 2_000_000.0:
        return t - SKYPATROL_JD_OFFSET
    if t > 50_000.0:
        return t + MJD_TO_JD - SKYPATROL_JD_OFFSET
    return t


def _lc_median_time(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "JD" not in df.columns:
        return None
    arr = pd.to_numeric(df["JD"], errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else None


def _derived_t0_width_days(half_width_days: object) -> float:
    half_width = _safe_float(half_width_days, 7.0) or 7.0
    return max(0.5, min(30.0, float(half_width) / 3.0))


def _apply_default_controls(defaults: Mapping[str, object] | None) -> dict[str, object]:
    out = dict(defaults or {})
    if out.get("t0_width_days") is None:
        out["t0_width_days"] = _derived_t0_width_days(out.get("half_width_days"))
    for key, value in DEFAULT_CONTROLS.items():
        out.setdefault(key, value)
    out.setdefault("log_v_width", DEFAULT_CONTROLS["log_v_width"])
    out.setdefault("status", "ok" if out.get("t0_jd") is not None else "missing")
    if not out.get("message"):
        out["message"] = "DustyCult defaults are ready." if out["status"] == "ok" else "Could not derive a dip window."
    return out


def _quality_status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "failed"
    if warnings:
        return "warning"
    return "ok"


def _band_count_dict(values: pd.Series | np.ndarray | list[object]) -> dict[str, int]:
    series = pd.Series(values).dropna().astype(str)
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _preflight_quality(
    frame: pd.DataFrame,
    window: Mapping[str, object],
    *,
    baseline_name: str,
    baseline_warnings: list[str] | None = None,
) -> dict[str, object]:
    warnings = [str(item) for item in (baseline_warnings or []) if str(item).strip()]
    errors: list[str] = []
    start = _safe_float(window.get("start_jd"))
    end = _safe_float(window.get("end_jd"))
    t0 = _safe_float(window.get("t0_jd"))
    n_points = int(len(frame) if frame is not None else 0)
    band_counts = _band_count_dict(frame["band"]) if frame is not None and "band" in frame.columns else {}
    time_span = None
    before_t0 = 0
    after_t0 = 0
    finite_errors = 0
    min_flux = None
    max_flux = None

    if start is None or end is None or end <= start:
        errors.append("DustyCult fit window is missing or invalid.")
    if t0 is None:
        errors.append("DustyCult t0 is missing.")
    elif start is not None and end is not None and not (start <= t0 <= end):
        errors.append("DustyCult t0 is outside the selected fit window.")

    if frame is None or frame.empty:
        errors.append("No valid g/V points fall inside the selected DustyCult window.")
    else:
        times = pd.to_numeric(frame.get("time"), errors="coerce").to_numpy(dtype=float)
        flux = pd.to_numeric(frame.get("relative_flux"), errors="coerce").to_numpy(dtype=float)
        errs = pd.to_numeric(frame.get("relative_flux_error"), errors="coerce").to_numpy(dtype=float)
        finite_time = times[np.isfinite(times)]
        finite_flux = flux[np.isfinite(flux)]
        finite_errors = int(np.isfinite(errs).sum())
        if finite_time.size:
            time_span = float(np.nanmax(finite_time) - np.nanmin(finite_time))
            if t0 is not None:
                before_t0 = int(np.sum(finite_time < float(t0)))
                after_t0 = int(np.sum(finite_time > float(t0)))
        if finite_flux.size:
            min_flux = float(np.nanmin(finite_flux))
            max_flux = float(np.nanmax(finite_flux))
        if n_points < DUSTYCULT_MIN_POINTS:
            errors.append(f"DustyCult needs at least {DUSTYCULT_MIN_POINTS} valid g/V points; found {n_points}.")
        elif n_points < DUSTYCULT_WARN_POINTS:
            warnings.append(f"DustyCult has only {n_points} valid g/V points.")
        if finite_errors <= 0:
            errors.append("DustyCult input has no finite flux errors.")
        if len(band_counts) <= 1:
            warnings.append("DustyCult input contains only one ASAS-SN band.")
        if time_span is not None and time_span < DUSTYCULT_WARN_SPAN_DAYS:
            warnings.append(f"DustyCult input spans only {time_span:.2f} days.")
        if t0 is not None and finite_time.size and (before_t0 < DUSTYCULT_MIN_SIDE_POINTS or after_t0 < DUSTYCULT_MIN_SIDE_POINTS):
            warnings.append("DustyCult input has sparse coverage on one side of t0.")

    window_span = float(end - start) if start is not None and end is not None and end >= start else None
    return {
        "status": _quality_status(errors, warnings),
        "warnings": warnings,
        "errors": errors,
        "n_points": n_points,
        "band_counts": band_counts,
        "time_span_days": time_span,
        "window_span_days": window_span,
        "before_t0_points": before_t0,
        "after_t0_points": after_t0,
        "finite_error_points": finite_errors,
        "min_relative_flux": min_flux,
        "max_relative_flux": max_flux,
        "baseline": {"name": str(baseline_name or ""), "warnings": list(baseline_warnings or [])},
        "window": {
            "start_jd": start,
            "end_jd": end,
            "t0_jd": t0,
            "source": str(window.get("source") or "manual_controls"),
        },
    }


def _preflight_failed(quality: Mapping[str, object] | None) -> bool:
    return str((quality or {}).get("status") or "").lower() == "failed"


def _quality_message(quality: Mapping[str, object] | None, *, include_warnings: bool = True) -> str:
    if not isinstance(quality, Mapping):
        return ""
    messages = [str(item) for item in quality.get("errors", []) or [] if str(item).strip()]
    if include_warnings:
        messages.extend(str(item) for item in quality.get("warnings", []) or [] if str(item).strip())
    return " ".join(messages)


def _defaults_viability_quality(
    payload: Mapping[str, object],
    defaults: Mapping[str, object],
    *,
    lc_path: str | Path | None,
    plot_dir: str | Path | None,
    run_params: Mapping[str, object] | None,
) -> dict[str, object] | None:
    try:
        controls = dict(defaults)
        controls["_dustycult_window_source"] = str(defaults.get("source") or "defaults")
        prepared = prepare_dustycult_input(
            payload,
            controls,
            lc_path=lc_path,
            plot_dir=plot_dir,
            run_params=run_params,
        )
        return dict(prepared.quality)
    except Exception as exc:
        return {
            "status": "failed",
            "warnings": [],
            "errors": [str(exc)],
            "window": {
                "start_jd": _safe_float(defaults.get("start_jd")),
                "end_jd": _safe_float(defaults.get("end_jd")),
                "t0_jd": _safe_float(defaults.get("t0_jd")),
                "source": str(defaults.get("source") or "defaults"),
            },
        }


def _window_from_t0_width(t0: float, width: float | None, duration: float | None) -> tuple[float, float, float]:
    candidates = [7.0]
    if width is not None and width > 0:
        candidates.append(3.0 * abs(float(width)))
    if duration is not None and duration > 0:
        candidates.append(0.5 * abs(float(duration)))
    half_width = min(max(max(candidates), 7.0), 120.0)
    return float(t0 - half_width), float(t0 + half_width), float(half_width)


def _stored_dip_defaults(payload: Mapping[str, object], lc_median: float | None = None) -> dict[str, object]:
    t0 = _time_value_to_lc_scale(payload.get("dip_best_t0"), lc_median)
    width = _safe_float(payload.get("dip_best_width_param"))
    duration = _safe_float(payload.get("dip_max_run_duration"))
    amp = _safe_float(payload.get("dip_best_amp"))
    if t0 is None:
        return {}
    start, end, half_width = _window_from_t0_width(t0, width, duration)
    return {
        "source": "stored_event_columns",
        "start_jd": start,
        "end_jd": end,
        "t0_jd": float(t0),
        "half_width_days": half_width,
        "width_param": width,
        "duration_days": duration,
        "amp_mag": amp,
        "message": "Loaded dip defaults from stored event columns.",
    }


def _deepest_point_defaults(df: pd.DataFrame) -> dict[str, object]:
    if df.empty or "JD" not in df.columns or "mag" not in df.columns:
        return {}
    work = df[np.isfinite(pd.to_numeric(df["JD"], errors="coerce")) & np.isfinite(pd.to_numeric(df["mag"], errors="coerce"))].copy()
    if work.empty:
        return {}
    idx = pd.to_numeric(work["mag"], errors="coerce").idxmax()
    t0 = float(work.loc[idx, "JD"])
    start, end, half_width = _window_from_t0_width(t0, None, None)
    return {
        "source": "deepest_point_fallback",
        "start_jd": start,
        "end_jd": end,
        "t0_jd": t0,
        "half_width_days": half_width,
        "width_param": None,
        "duration_days": None,
        "amp_mag": None,
        "message": "No stored dip window found; using deepest cleaned point.",
    }


def _dimming_complex_defaults(
    candidate_id: str,
    lc_path: str | Path,
) -> dict[str, object]:
    """Return DustyCult controls from the shared recovery-anchored window."""
    measurement = measure_dimming_complex_window(
        str(candidate_id),
        Path(lc_path).expanduser(),
        config=DEFAULT_DIMMING_WINDOW_CONFIG,
    )
    window = measurement.window
    duration = window.duration_days
    duration_upper = None if window.is_lower_limit else duration
    censoring = window.censoring_status
    qualifier = "lower-limit" if window.is_lower_limit else "recovery-bounded"
    return {
        "source": DIMMING_WINDOW_METHOD_VERSION,
        "method_version": DIMMING_WINDOW_METHOD_VERSION,
        "start_jd": float(window.start_jd),
        "end_jd": float(window.end_jd),
        "t0_jd": float(window.peak_jd),
        "half_width_days": 0.5 * float(duration),
        "width_param": None,
        "duration_days": float(duration),
        "duration_lower_days": float(duration),
        "duration_upper_days": duration_upper,
        "amp_mag": float(window.peak_depth_mag),
        "event_window_status": str(window.status),
        "dimming_complex_status": censoring,
        "dimming_complex_is_lower_limit": bool(window.is_lower_limit),
        "left_boundary_type": str(window.left_boundary_type),
        "right_boundary_type": str(window.right_boundary_type),
        "gap_count": int(window.gap_count),
        "max_gap_days": float(window.max_gap_days),
        "message": (
            "Loaded the shared recovery-anchored T_complex window "
            f"({qualifier}; {censoring})."
        ),
    }


def recompute_dip_defaults(df: pd.DataFrame, run_params: Mapping[str, object] | None = None) -> dict[str, object]:
    if df is None or df.empty:
        return {}
    from malca.stv.events import score_lightcurve
    from malca.review.interactive_plot import BASELINE_FUNCTIONS, _baseline_config_from_run_params

    baseline_name, baseline_kwargs, warnings = _baseline_config_from_run_params(dict(run_params or {}))
    baseline_func = BASELINE_FUNCTIONS.get(baseline_name)
    if baseline_func is None:
        baseline_func = BASELINE_FUNCTIONS["per_camera_gp"]
    res = score_lightcurve(
        df,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        trigger_mode=str((run_params or {}).get("trigger_mode") or TRIGGER_MODE),
        logbf_threshold_dip=float((run_params or {}).get("logbf_threshold_dip") or LOGBF_THRESHOLD_DIP),
        logbf_threshold_jump=float((run_params or {}).get("logbf_threshold_jump") or LOGBF_THRESHOLD_JUMP),
        significance_threshold=float((run_params or {}).get("significance_threshold") or SIGNIFICANCE_THRESHOLD),
        p_points=int((run_params or {}).get("p_points") or P_POINTS),
        mag_points=int((run_params or {}).get("mag_points") or MAG_POINTS),
        run_min_points=int((run_params or {}).get("run_min_points") or RUN_MIN_POINTS),
        max_gap_points=int((run_params or {}).get("run_max_gap_points") or RUN_MAX_GAP_POINTS),
        run_max_gap_days=_finite_or_none((run_params or {}).get("run_max_gap_days")),
        run_min_duration_days=_finite_or_none((run_params or {}).get("run_min_duration_days")),
        compute_event_prob=True,
    )
    run_summaries = list((res.get("dip") or {}).get("run_summaries") or [])
    if not run_summaries:
        fallback = _deepest_point_defaults(df)
        if fallback:
            fallback["source"] = "recompute_deepest_point_fallback"
            fallback["message"] = "Recomputed dip detection found no run; using deepest cleaned point."
        return fallback
    best = max(run_summaries, key=lambda item: _safe_float(item.get("run_max"), -np.inf) or -np.inf)
    params = best.get("params") if isinstance(best.get("params"), dict) else {}
    t0 = _safe_float(params.get("t0"), _safe_float(best.get("peak_jd")))
    start = _safe_float(best.get("start_jd"))
    end = _safe_float(best.get("end_jd"))
    width = _safe_float(params.get("sigma"), _safe_float(best.get("duration_days")))
    duration = _safe_float(best.get("duration_days"))
    if t0 is None:
        t0 = (float(start) + float(end)) / 2.0 if start is not None and end is not None else None
    if t0 is None:
        return _deepest_point_defaults(df)
    if start is None or end is None or end <= start:
        start, end, half_width = _window_from_t0_width(float(t0), width, duration)
    else:
        half_width = 0.5 * abs(float(end) - float(start))
    message = "Recomputed dip defaults from canonical cleaned light curve."
    if warnings:
        message += " " + " ".join(str(w) for w in warnings)
    return {
        "source": "recomputed_dip_run",
        "start_jd": float(start),
        "end_jd": float(end),
        "t0_jd": float(t0),
        "half_width_days": float(half_width),
        "width_param": width,
        "duration_days": duration,
        "amp_mag": _safe_float(params.get("amp")),
        "message": message,
        "run_summary": best,
    }


def load_canonical_cleaned_lightcurve(
    payload: Mapping[str, object],
    *,
    lc_path: str | Path | None = None,
    plot_dir: str | Path | None = None,
    run_params: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, Path]:
    plot_path = Path(plot_dir).expanduser() if plot_dir else None
    resolved = Path(lc_path).expanduser() if lc_path else None
    if resolved is None or not resolved.exists():
        from malca.review.interactive_plot import resolve_lightcurve_path

        resolved = resolve_lightcurve_path(dict(payload or {}), plot_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError("No local ASAS-SN light curve file is available for DustyCult.")

    params = dict(run_params or {})
    scatter_ratio = float(params.get("bad_camera_scatter_ratio", BAD_CAMERA_SCATTER_RATIO_THRESHOLD))
    clean_abs = float(params.get("clean_max_error_absolute", CLEAN_LC_MAX_ERROR_ABSOLUTE))
    clean_sig = float(params.get("clean_max_error_sigma", CLEAN_LC_MAX_ERROR_SIGMA))
    from malca.review.interactive_plot import _load_cleaned_df

    df, _filtered, _diag = _load_cleaned_df(
        resolved,
        filter_bad_cameras=True,
        scatter_ratio=scatter_ratio,
        clean_max_error_absolute=clean_abs,
        clean_max_error_sigma=clean_sig,
    )
    return df, resolved


def control_defaults_for_candidate(
    conn: sqlite3.Connection | None,
    candidate_id: str,
    payload: Mapping[str, object],
    *,
    lc_path: str | Path | None = None,
    plot_dir: str | Path | None = None,
    run_params: Mapping[str, object] | None = None,
    recompute: bool = False,
) -> dict[str, object]:
    df = pd.DataFrame()
    resolved: Path | None = None
    try:
        df, resolved = load_canonical_cleaned_lightcurve(
            payload,
            lc_path=lc_path,
            plot_dir=plot_dir,
            run_params=run_params,
        )
    except Exception:
        df = pd.DataFrame()
    lc_median = _lc_median_time(df)
    defaults: dict[str, object] = {}
    shared_error = ""
    if resolved is not None:
        try:
            defaults = _dimming_complex_defaults(str(candidate_id), resolved)
        except Exception as exc:
            shared_error = f"{type(exc).__name__}: {exc}"

    # Retain the prior event-column/STV behavior only as an explicit fallback
    # for unavailable or unmeasurable local light curves.
    should_recompute = bool(recompute) and not defaults
    if not defaults and not should_recompute and conn is not None:
        try:
            row = conn.execute(
                f"SELECT status FROM {DUSTYCULT_FIT_TABLE} WHERE candidate_id = ? ORDER BY updated_at DESC LIMIT 1",
                (str(candidate_id),),
            ).fetchone()
            should_recompute = bool(row and str(row[0] or "").lower() == "failed")
        except sqlite3.OperationalError:
            should_recompute = False
    if should_recompute and not df.empty:
        try:
            defaults = recompute_dip_defaults(df, run_params=run_params)
        except Exception as exc:
            defaults = _stored_dip_defaults(payload, lc_median)
            defaults["message"] = f"Recompute failed: {exc}. " + str(defaults.get("message") or "")
    if not defaults:
        defaults = _stored_dip_defaults(payload, lc_median)
        if defaults and not df.empty:
            quality = _defaults_viability_quality(
                payload,
                defaults,
                lc_path=lc_path,
                plot_dir=plot_dir,
                run_params=run_params,
            )
            if _preflight_failed(quality):
                try:
                    recomputed = recompute_dip_defaults(df, run_params=run_params)
                except Exception as exc:
                    recomputed = {}
                    defaults["message"] = (
                        f"Stored dip defaults failed preflight: {_quality_message(quality, include_warnings=False)} "
                        f"Recompute failed: {exc}. "
                        + str(defaults.get("message") or "")
                    ).strip()
                if recomputed:
                    recomputed["message"] = (
                        f"Stored dip defaults failed preflight: {_quality_message(quality, include_warnings=False)} "
                        + str(recomputed.get("message") or "")
                    ).strip()
                    defaults = recomputed
    if not defaults and not df.empty:
        try:
            defaults = recompute_dip_defaults(df, run_params=run_params)
        except Exception as exc:
            defaults = _deepest_point_defaults(df)
            if defaults:
                defaults["message"] = f"Recompute failed: {exc}. " + str(defaults.get("message") or "")
    if not defaults and not df.empty:
        defaults = _deepest_point_defaults(df)
    if defaults and shared_error:
        defaults["shared_window_fallback"] = True
        defaults["shared_window_error"] = shared_error
        defaults["message"] = (
            f"Shared {DIMMING_WINDOW_METHOD_VERSION} measurement failed: "
            f"{shared_error}. {defaults.get('message') or ''}"
        ).strip()
    defaults = _apply_default_controls(defaults)
    defaults.update(_stellar_defaults(conn, candidate_id, payload))
    return defaults


def _stellar_defaults(
    conn: sqlite3.Connection | None,
    candidate_id: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    radius = None
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT radius_rsun FROM sed_model_fits WHERE candidate_id = ? AND status = 'ok'",
                (str(candidate_id),),
            ).fetchone()
            if row:
                radius = _safe_float(row[0])
        except sqlite3.OperationalError:
            radius = None
    if radius is None:
        for key in ("radius_rsun", "stellar_radius", "radius", "gaia_radius"):
            radius = _safe_float(payload.get(key))
            if radius is not None and radius > 0:
                break
    if radius is None or radius <= 0:
        radius = DEFAULT_CONTROLS["star_R"]
    return {
        "star_R": float(radius),
        "star_u1": _safe_float(payload.get("ld_u1"), DEFAULT_CONTROLS["star_u1"]),
        "star_u2": _safe_float(payload.get("ld_u2"), DEFAULT_CONTROLS["star_u2"]),
    }


def normalize_controls(values: Mapping[str, object] | None) -> dict[str, object]:
    values = dict(values or {})
    controls = dict(DEFAULT_CONTROLS)
    for key, value in values.items():
        if key in controls or key in {"start_jd", "end_jd", "t0_jd"}:
            controls[key] = _safe_float(value, controls.get(key))
    for key in DUSTYCULT_WINDOW_METADATA_FIELDS:
        if key in values:
            controls[key] = values[key]
    if controls.get("start_jd") is not None and controls.get("end_jd") is not None:
        start = float(controls["start_jd"])
        end = float(controls["end_jd"])
        if end < start:
            start, end = end, start
        controls["start_jd"] = start
        controls["end_jd"] = end
    return controls


def prepare_dustycult_input(
    payload: Mapping[str, object],
    controls: Mapping[str, object],
    *,
    lc_path: str | Path | None = None,
    plot_dir: str | Path | None = None,
    run_params: Mapping[str, object] | None = None,
) -> PreparedDustyCultInput:
    raw_controls = dict(controls or {})
    window_source = str(
        raw_controls.get("_dustycult_window_source")
        or raw_controls.get("source")
        or "manual_controls"
    )
    controls = normalize_controls(controls)
    df, resolved_lc_path = load_canonical_cleaned_lightcurve(
        payload,
        lc_path=lc_path,
        plot_dir=plot_dir,
        run_params=run_params,
    )
    if df.empty:
        raise ValueError(f"No cleaned light-curve points remain in {resolved_lc_path}")
    params = dict(run_params or {})
    scatter_ratio = float(params.get("bad_camera_scatter_ratio", BAD_CAMERA_SCATTER_RATIO_THRESHOLD))
    clean_abs = float(params.get("clean_max_error_absolute", CLEAN_LC_MAX_ERROR_ABSOLUTE))
    clean_sig = float(params.get("clean_max_error_sigma", CLEAN_LC_MAX_ERROR_SIGMA))
    cache_key = (str(resolved_lc_path.resolve()), True, scatter_ratio, clean_abs, clean_sig)
    from malca.review.interactive_plot import _baseline_config_from_run_params, _compute_baseline_bands

    baseline_name, baseline_kwargs, baseline_warnings = _baseline_config_from_run_params(params)
    band_frames = _compute_baseline_bands(df, baseline_name, cache_key, baseline_kwargs=baseline_kwargs)
    if not band_frames:
        raise ValueError("No g/V light-curve bands are available for DustyCult.")
    merged = pd.concat([part.copy() for band, part in band_frames.items() if band in (0, 1)], ignore_index=True)
    if merged.empty:
        raise ValueError("No ASAS-SN g/V points are available for DustyCult.")
    if "baseline" not in merged.columns or not np.isfinite(pd.to_numeric(merged["baseline"], errors="coerce")).any():
        merged["baseline"] = merged.groupby("v_g_band")["mag"].transform("median")
        baseline_warnings = list(baseline_warnings) + ["Baseline model unavailable; used per-band median for DustyCult normalization."]

    start = _safe_float(controls.get("start_jd"))
    end = _safe_float(controls.get("end_jd"))
    derived_defaults: dict[str, object] = {}
    if start is None or end is None:
        try:
            derived_defaults = _dimming_complex_defaults(
                str(payload.get("candidate_id") or "unknown"),
                resolved_lc_path,
            )
        except Exception:
            derived_defaults = _stored_dip_defaults(
                payload,
                _lc_median_time(df),
            ) or _deepest_point_defaults(df)
        start = _safe_float(derived_defaults.get("start_jd"))
        end = _safe_float(derived_defaults.get("end_jd"))
        window_source = str(derived_defaults.get("source") or window_source)
    if start is None or end is None:
        raise ValueError("DustyCult fit window is missing.")
    if end < start:
        start, end = end, start

    work = merged.copy()
    for col in ("JD", "mag", "error", "baseline"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[(work["JD"] >= start) & (work["JD"] <= end)].copy()
    work = work[np.isfinite(work["JD"]) & np.isfinite(work["mag"]) & np.isfinite(work["error"]) & (work["error"] > 0) & np.isfinite(work["baseline"])]
    work = work[work["v_g_band"].isin([0, 1])].copy()
    if work.empty:
        raise ValueError("No valid g/V points fall inside the selected DustyCult window.")
    work["band"] = work["v_g_band"].map(lambda value: BAND_LABELS.get(int(value), str(value)))
    delta_mag = work["mag"].to_numpy(dtype=float) - work["baseline"].to_numpy(dtype=float)
    relative_flux = np.power(10.0, -0.4 * delta_mag)
    relative_flux_error = (math.log(10.0) / 2.5) * relative_flux * work["error"].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "time": work["JD"].to_numpy(dtype=float),
            "relative_flux": relative_flux,
            "relative_flux_error": relative_flux_error,
            "band": work["band"].astype(str).to_numpy(),
            "source_mag": work["mag"].to_numpy(dtype=float),
            "baseline_mag": work["baseline"].to_numpy(dtype=float),
        }
    )
    out = out[np.isfinite(out["time"]) & np.isfinite(out["relative_flux"]) & np.isfinite(out["relative_flux_error"]) & (out["relative_flux_error"] > 0)]
    if out.empty:
        raise ValueError("No finite relative-flux points could be prepared for DustyCult.")
    t0 = _safe_float(controls.get("t0_jd"))
    if t0 is None:
        t0 = _safe_float(derived_defaults.get("t0_jd"))
    if t0 is None:
        t0 = 0.5 * (float(start) + float(end))
    window = {
        "start_jd": float(start),
        "end_jd": float(end),
        "t0_jd": float(t0),
        "n_input_points": int(len(out)),
        "lc_path": str(resolved_lc_path),
        "source": window_source,
    }
    for key in DUSTYCULT_WINDOW_METADATA_FIELDS:
        if key in raw_controls:
            window[key] = raw_controls[key]
        elif key in derived_defaults:
            window[key] = derived_defaults[key]
    out = out.reset_index(drop=True)
    quality = _preflight_quality(
        out,
        window,
        baseline_name=baseline_name,
        baseline_warnings=list(baseline_warnings),
    )
    return PreparedDustyCultInput(out, window, baseline_name, list(baseline_warnings), quality)


def build_dustycult_config(input_csv: str | Path, controls: Mapping[str, object], mode: str) -> dict[str, object]:
    mode = normalize_mode(mode)
    controls = normalize_controls(controls)
    sampling = SAMPLING_BY_MODE[mode]
    prior_kwargs = {
        "t0_center": _safe_float(controls.get("t0_jd")),
        "t0_width": float(controls["t0_width_days"]),
        "log_v_width": float(controls["log_v_width"]),
        "b_center": float(controls["b_center"]),
        "b_width": float(controls["b_width"]),
        "log_tau0_width": float(controls["log_tau0_width"]),
        "alpha_center": float(controls["alpha_center"]),
        "alpha_width": float(controls["alpha_width"]),
        "log_sigma_width": float(controls["log_sigma_width"]),
    }
    config = {
        "lightcurve": {
            "path": str(input_csv),
            "format": "relative_flux",
            "columns": {
                "time": "time",
                "relative_flux": "relative_flux",
                "relative_flux_error": "relative_flux_error",
                "band": "band",
            },
        },
        "bandpass": {
            band: {"wavelength": wavelength}
            for band, wavelength in DUSTYCULT_BANDPASS_NM.items()
        },
        "star": {
            "R": float(controls["star_R"]),
            "I0": 1.0,
            "u1": float(controls["star_u1"]),
            "u2": float(controls["star_u2"]),
        },
        "grid": {"n": int(sampling["grid_n"])},
        "sampling": {
            "n_samples": int(sampling["n_samples"]),
            "n_adapt": int(sampling["n_adapt"]),
            "n_chains": int(sampling["n_chains"]),
            "progress": False,
        },
        "posterior_predictive": {"n_draws": int(sampling["n_predictive_draws"])},
        "prior_kwargs": prior_kwargs,
        "malca": {
            "mode": mode,
            "t0_jd": _safe_float(controls.get("t0_jd")),
            "start_jd": _safe_float(controls.get("start_jd")),
            "end_jd": _safe_float(controls.get("end_jd")),
        },
    }
    return config


def _tail_text(value: object, limit: int = 6000) -> str:
    text = str(value or "")
    return text[-limit:] if len(text) > limit else text


def _posterior_summary(samples: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    if samples is None or samples.empty:
        return summary
    for col in samples.columns:
        if col == "draw_id":
            continue
        series = pd.to_numeric(samples[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary[str(col)] = {
            "p16": float(series.quantile(0.16)),
            "median": float(series.quantile(0.50)),
            "p84": float(series.quantile(0.84)),
        }
    return summary


def _sample_diagnostics(samples: pd.DataFrame) -> dict[str, object]:
    if samples is None or samples.empty:
        return {"sample_count": 0, "finite_sample_count": 0, "degenerate_parameters": [], "all_required_degenerate": False}
    finite_count = 0
    degenerate: list[str] = []
    for col in samples.columns:
        if col == "draw_id":
            continue
        series = pd.to_numeric(samples[col], errors="coerce").dropna()
        if not series.empty:
            finite_count = max(finite_count, int(len(series)))
        if col in DUSTYCULT_REQUIRED_POSTERIOR and len(series) > 1 and float(series.std(ddof=0)) <= 1e-12:
            degenerate.append(str(col))
    required_present = [col for col in DUSTYCULT_REQUIRED_POSTERIOR if col in samples.columns]
    return {
        "sample_count": int(len(samples)),
        "finite_sample_count": int(finite_count),
        "degenerate_parameters": degenerate,
        "all_required_degenerate": bool(required_present and set(required_present).issubset(set(degenerate))),
    }


def _parse_divergent_transitions(stderr: str) -> int:
    values = [int(match) for match in re.findall(r"There were\s+(\d+)\s+divergent transitions", str(stderr or ""))]
    return max(values) if values else 0


def _posterior_median(summary: Mapping[str, object], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, Mapping):
        return _safe_float(value.get("median"))
    return _safe_float(value)


def _posterior_spread(summary: Mapping[str, object], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, Mapping):
        return None
    p16 = _safe_float(value.get("p16"))
    p84 = _safe_float(value.get("p84"))
    if p16 is None or p84 is None:
        return None
    return abs(float(p84) - float(p16))


def _validate_postfit_quality(
    preflight_quality: Mapping[str, object],
    artifact_summary: Mapping[str, object],
    curves: pd.DataFrame,
    *,
    stderr: str,
    controls: Mapping[str, object],
    window: Mapping[str, object],
) -> dict[str, object]:
    warnings = [str(item) for item in preflight_quality.get("warnings", []) or [] if str(item).strip()]
    errors = [str(item) for item in preflight_quality.get("errors", []) or [] if str(item).strip()]
    posterior = artifact_summary.get("posterior") if isinstance(artifact_summary.get("posterior"), Mapping) else {}
    sample_diag = artifact_summary.get("sample_diagnostics") if isinstance(artifact_summary.get("sample_diagnostics"), Mapping) else {}
    sample_count = int(artifact_summary.get("sample_count") or sample_diag.get("sample_count") or 0)
    divergent_count = _parse_divergent_transitions(stderr)

    if sample_count <= 0:
        errors.append("DustyCult wrote no posterior samples.")
    if divergent_count > 0:
        if sample_count > 0 and divergent_count >= sample_count:
            errors.append(f"All DustyCult samples diverged ({divergent_count}/{sample_count}).")
        else:
            warnings.append(f"DustyCult reported {divergent_count} divergent transitions.")
    if bool(sample_diag.get("all_required_degenerate")):
        errors.append("DustyCult posterior samples are degenerate.")

    for key in DUSTYCULT_REQUIRED_POSTERIOR:
        median = _posterior_median(posterior, key)
        if median is None:
            errors.append(f"DustyCult posterior is missing {key}.")
            continue
        if key in DUSTYCULT_POSITIVE_POSTERIOR and median <= 0:
            errors.append(f"DustyCult posterior {key} is non-positive.")
    zero_spread = [
        key
        for key in DUSTYCULT_REQUIRED_POSTERIOR
        if (_posterior_spread(posterior, key) is not None and (_posterior_spread(posterior, key) or 0.0) <= 1e-12)
    ]
    if len(zero_spread) >= len(DUSTYCULT_REQUIRED_POSTERIOR):
        errors.append("DustyCult posterior credible intervals are degenerate.")
    elif len(zero_spread) >= max(4, len(DUSTYCULT_REQUIRED_POSTERIOR) // 2):
        warnings.append("DustyCult posterior has weak or degenerate spread for many parameters.")

    t0 = _posterior_median(posterior, "t0")
    t0_center = _safe_float(controls.get("t0_jd"), _safe_float(window.get("t0_jd")))
    t0_width = _safe_float(controls.get("t0_width_days"), DEFAULT_CONTROLS["t0_width_days"]) or DEFAULT_CONTROLS["t0_width_days"]
    start = _safe_float(window.get("start_jd"))
    end = _safe_float(window.get("end_jd"))
    if t0 is not None and t0_center is not None and start is not None and end is not None:
        span = max(float(end) - float(start), 1.0)
        allowed = max(5.0 * float(t0_width), span)
        if abs(float(t0) - float(t0_center)) > allowed:
            errors.append("DustyCult posterior t0 is far outside the configured prior/window.")

    if curves is not None and not curves.empty and {"observed", "median"}.issubset(curves.columns):
        observed = pd.to_numeric(curves["observed"], errors="coerce").to_numpy(dtype=float)
        median = pd.to_numeric(curves["median"], errors="coerce").to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        median = median[np.isfinite(median)]
        if observed.size and median.size:
            obs_min = float(np.nanmin(observed))
            obs_range = float(np.nanmax(observed) - np.nanmin(observed))
            pred_range = float(np.nanmax(median) - np.nanmin(median))
            if pred_range <= 1e-5 and obs_min < 0.97 and obs_range > 0.03:
                errors.append("DustyCult predictive curve is flat while the observed input contains a significant dip.")

    quality = dict(preflight_quality)
    quality.update(
        {
            "status": _quality_status(errors, warnings),
            "warnings": warnings,
            "errors": errors,
            "divergent_transitions": divergent_count,
            "sample_count": sample_count,
            "sample_diagnostics": dict(sample_diag),
            "postfit": {
                "posterior_zero_spread_parameters": zero_spread,
            },
        }
    )
    return quality


def _load_artifact_frame(path_base: Path) -> pd.DataFrame:
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _load_fit_artifacts(output_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    manifest_path = output_dir / "manifest.json"
    manifest = _json_loads(manifest_path.read_text() if manifest_path.exists() else None, {})
    samples = _load_artifact_frame(output_dir / "samples")
    posterior = _posterior_summary(samples)
    sample_diagnostics = _sample_diagnostics(samples)
    curves = _load_artifact_frame(output_dir / "predictive_intervals")
    summary = {
        "manifest": manifest,
        "posterior": posterior,
        "sample_count": int(len(samples)),
        "sample_diagnostics": sample_diagnostics,
    }
    return summary, curves


def _curve_rows(candidate_id: str, mode: str, curves: pd.DataFrame) -> pd.DataFrame:
    if curves is None or curves.empty:
        return pd.DataFrame(columns=DUSTYCULT_CURVE_COLUMNS)
    frame = curves.copy()
    for col in ("point_id", "time", "band", "observed", "error", "lower95", "lower68", "median", "upper68", "upper95"):
        if col not in frame.columns:
            frame[col] = np.nan if col != "band" else ""
    frame["candidate_id"] = str(candidate_id)
    frame["mode"] = normalize_mode(mode)
    return frame[DUSTYCULT_CURVE_COLUMNS]


def parse_json_cell(value: object, default: Any | None = None) -> Any:
    return _json_loads(value, default if default is not None else {})


def preferred_fit_mode(fits: pd.DataFrame | None) -> str | None:
    if fits is None or fits.empty:
        return None
    frame = fits.copy()
    if "mode" not in frame.columns:
        return None
    status_series = (
        frame["status"].astype(str)
        if "status" in frame.columns
        else pd.Series([""] * len(frame), index=frame.index)
    )
    mode_series = frame["mode"].astype(str)
    for mode in ("full", "quick"):
        ok = frame[(mode_series == mode) & (status_series == "ok")]
        if not ok.empty:
            return mode
    for mode in ("full", "quick"):
        warning = frame[(mode_series == mode) & (status_series == "warning")]
        if not warning.empty:
            return mode
    for mode in ("full", "quick"):
        present = frame[mode_series == mode]
        if not present.empty:
            return mode
    return str(frame.iloc[0].get("mode") or "quick")


check_dustycult_availability = check_dustycult_available
resolve_fit_defaults = control_defaults_for_candidate


def upsert_dustycult_fit(
    conn: sqlite3.Connection,
    fit_row: Mapping[str, object],
    curves: pd.DataFrame | None = None,
) -> None:
    row = {col: fit_row.get(col) for col in DUSTYCULT_FIT_COLUMNS}
    row["candidate_id"] = str(row.get("candidate_id") or "")
    row["mode"] = normalize_mode(row.get("mode") or "quick")
    now = _utc_now()
    row["updated_at"] = row.get("updated_at") or now
    row["created_at"] = row.get("created_at") or now
    placeholders = ", ".join(["?"] * len(DUSTYCULT_FIT_COLUMNS))
    assignments = ", ".join([f"{col}=excluded.{col}" for col in DUSTYCULT_FIT_COLUMNS if col not in {"candidate_id", "mode", "created_at"}])
    sql = (
        f"INSERT INTO {DUSTYCULT_FIT_TABLE} ({', '.join(DUSTYCULT_FIT_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(candidate_id, mode) DO UPDATE SET {assignments}"
    )
    conn.execute(sql, [_sqlite_value(row[col]) for col in DUSTYCULT_FIT_COLUMNS])
    conn.execute(
        f"DELETE FROM {DUSTYCULT_CURVE_TABLE} WHERE candidate_id = ? AND mode = ?",
        (row["candidate_id"], row["mode"]),
    )
    if curves is not None and not curves.empty:
        frame = curves.copy()
        for col in DUSTYCULT_CURVE_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        frame = frame[DUSTYCULT_CURVE_COLUMNS]
        curve_sql = f"INSERT INTO {DUSTYCULT_CURVE_TABLE} ({', '.join(DUSTYCULT_CURVE_COLUMNS)}) VALUES ({', '.join(['?'] * len(DUSTYCULT_CURVE_COLUMNS))})"
        for _, curve_row in frame.iterrows():
            conn.execute(curve_sql, [_sqlite_value(curve_row[col]) for col in DUSTYCULT_CURVE_COLUMNS])
    conn.commit()


def load_dustycult_fits(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(DUSTYCULT_FIT_COLUMNS)} FROM {DUSTYCULT_FIT_TABLE} WHERE candidate_id = ? ORDER BY mode",
            conn,
            params=(str(candidate_id),),
        )
    except Exception:
        return pd.DataFrame(columns=DUSTYCULT_FIT_COLUMNS)


def load_dustycult_curve(conn: sqlite3.Connection, candidate_id: str, mode: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(DUSTYCULT_CURVE_COLUMNS)} FROM {DUSTYCULT_CURVE_TABLE} WHERE candidate_id = ? AND mode = ? ORDER BY time, point_id",
            conn,
            params=(str(candidate_id), normalize_mode(mode)),
        )
    except Exception:
        return pd.DataFrame(columns=DUSTYCULT_CURVE_COLUMNS)


load_dustycult_curves = load_dustycult_curve


def _failure_row(
    *,
    candidate_id: str,
    mode: str,
    controls: Mapping[str, object],
    started_at: float,
    error: str,
    output_dir: Path | None = None,
    command: list[str] | None = None,
    stdout: str = "",
    stderr: str = "",
    config: Mapping[str, object] | None = None,
    window: Mapping[str, object] | None = None,
    quality: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary = {"quality": dict(quality or {})}
    return {
        "candidate_id": str(candidate_id),
        "mode": normalize_mode(mode),
        "status": "failed",
        "runtime_sec": float(time.monotonic() - started_at),
        "artifact_dir": str(output_dir) if output_dir else "",
        "command_json": _json_dumps(command or []),
        "config_json": _json_dumps(config or {}),
        "controls_json": _json_dumps(dict(controls or {})),
        "window_json": _json_dumps(dict(window or {})),
        "summary_json": _json_dumps(summary),
        "stderr_tail": _tail_text(stderr),
        "stdout_tail": _tail_text(stdout),
        "error": str(error),
        "t0_jd": _safe_float((window or controls or {}).get("t0_jd")),
        "start_jd": _safe_float((window or controls or {}).get("start_jd")),
        "end_jd": _safe_float((window or controls or {}).get("end_jd")),
        "n_input_points": int((window or {}).get("n_input_points") or 0),
        "n_curve_points": 0,
    }


def run_dustycult_fit(
    conn: sqlite3.Connection,
    candidate_id: str,
    payload: Mapping[str, object],
    *,
    db_path: str | Path,
    controls: Mapping[str, object],
    mode: str,
    lc_path: str | Path | None = None,
    plot_dir: str | Path | None = None,
    run_params: Mapping[str, object] | None = None,
    project_path: str | Path | None = None,
    julia: str | Path | None = None,
) -> dict[str, object]:
    mode = normalize_mode(mode)
    raw_controls = dict(controls or {})
    window_source = str(raw_controls.get("_dustycult_window_source") or raw_controls.get("source") or "manual_controls")
    controls = normalize_controls(raw_controls)
    started = time.monotonic()
    output_dir = artifact_dir_for_candidate(db_path, candidate_id, mode)
    availability = check_dustycult_available(project_path=project_path, julia=julia)
    if not availability.ok:
        row = _failure_row(
            candidate_id=candidate_id,
            mode=mode,
            controls=controls,
            started_at=started,
            error=availability.message,
            output_dir=output_dir,
        )
        upsert_dustycult_fit(conn, row)
        return row

    try:
        prepare_controls = dict(controls)
        prepare_controls["_dustycult_window_source"] = window_source
        prepared = prepare_dustycult_input(
            payload,
            prepare_controls,
            lc_path=lc_path,
            plot_dir=plot_dir,
            run_params=run_params,
        )
        preflight_quality = dict(prepared.quality)
        if _preflight_failed(preflight_quality):
            row = _failure_row(
                candidate_id=candidate_id,
                mode=mode,
                controls=controls,
                started_at=started,
                error=_quality_message(preflight_quality, include_warnings=False) or "DustyCult preflight failed.",
                output_dir=output_dir,
                window=prepared.window,
                quality=preflight_quality,
            )
            upsert_dustycult_fit(conn, row)
            return row
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_csv = output_dir / "input_lightcurve.csv"
        prepared.frame[["time", "relative_flux", "relative_flux_error", "band"]].to_csv(input_csv, index=False)
        config = build_dustycult_config(input_csv, controls, mode)
        config["malca"].update(
            {
                "candidate_id": str(candidate_id),
                "window": prepared.window,
                "baseline": {
                    "name": prepared.baseline_name,
                    "warnings": prepared.baseline_warnings,
                },
                "quality": preflight_quality,
            }
        )
        config_path = output_dir / "run_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        row = _failure_row(
            candidate_id=candidate_id,
            mode=mode,
            controls=controls,
            started_at=started,
            error=str(exc),
            output_dir=output_dir,
            quality={"status": "failed", "warnings": [], "errors": [str(exc)]},
        )
        upsert_dustycult_fit(conn, row)
        return row

    if prepared.window.get("t0_jd") is None:
        row = _failure_row(
            candidate_id=candidate_id,
            mode=mode,
            controls=controls,
            started_at=started,
            error="DustyCult t0 is missing.",
            output_dir=output_dir,
            config=config,
            window=prepared.window,
            quality=prepared.quality,
        )
        row["input_path"] = str(input_csv)
        row["config_path"] = str(config_path)
        upsert_dustycult_fit(conn, row)
        return row

    command = [
        availability.julia,
        f"--project={availability.project_path}",
        str(availability.script_path),
        "--config",
        str(config_path),
        "--out",
        str(output_dir),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        row = _failure_row(
            candidate_id=candidate_id,
            mode=mode,
            controls=controls,
            started_at=started,
            error=f"Could not start DustyCult: {exc}",
            output_dir=output_dir,
            command=command,
            config=config,
            window=prepared.window,
            quality=prepared.quality,
        )
        row["input_path"] = str(input_csv)
        row["config_path"] = str(config_path)
        upsert_dustycult_fit(conn, row)
        return row
    if completed.returncode != 0:
        row = _failure_row(
            candidate_id=candidate_id,
            mode=mode,
            controls=controls,
            started_at=started,
            error=f"DustyCult exited with status {completed.returncode}",
            output_dir=output_dir,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            config=config,
            window=prepared.window,
            quality=prepared.quality,
        )
        row["input_path"] = str(input_csv)
        row["config_path"] = str(config_path)
        upsert_dustycult_fit(conn, row)
        return row

    try:
        artifact_summary, curves = _load_fit_artifacts(output_dir)
        curve_rows = _curve_rows(candidate_id, mode, curves)
        posterior = artifact_summary.get("posterior", {})
        final_quality = _validate_postfit_quality(
            prepared.quality,
            artifact_summary,
            curves,
            stderr=completed.stderr,
            controls=controls,
            window=prepared.window,
        )
        artifact_summary = dict(artifact_summary)
        artifact_summary["quality"] = final_quality
        status = str(final_quality.get("status") or "ok")
        status_message = _quality_message(final_quality, include_warnings=(status != "ok"))
        manifest_path = output_dir / "manifest.json"
        row = {
            "candidate_id": str(candidate_id),
            "mode": mode,
            "status": status,
            "runtime_sec": float(time.monotonic() - started),
            "artifact_dir": str(output_dir),
            "input_path": str(input_csv),
            "config_path": str(config_path),
            "manifest_path": str(manifest_path) if manifest_path.exists() else "",
            "command_json": _json_dumps(command),
            "config_json": _json_dumps(config),
            "controls_json": _json_dumps(dict(controls)),
            "window_json": _json_dumps(prepared.window),
            "stellar_json": _json_dumps(config.get("star", {})),
            "posterior_json": _json_dumps(posterior),
            "summary_json": _json_dumps(artifact_summary),
            "stderr_tail": _tail_text(completed.stderr),
            "stdout_tail": _tail_text(completed.stdout),
            "error": "" if status == "ok" else status_message,
            "t0_jd": _safe_float(prepared.window.get("t0_jd")),
            "start_jd": _safe_float(prepared.window.get("start_jd")),
            "end_jd": _safe_float(prepared.window.get("end_jd")),
            "n_input_points": int(prepared.window.get("n_input_points") or 0),
            "n_curve_points": int(len(curve_rows)),
        }
        upsert_dustycult_fit(conn, row, curve_rows)
        return row
    except Exception as exc:
        row = _failure_row(
            candidate_id=candidate_id,
            mode=mode,
            controls=controls,
            started_at=started,
            error=f"Could not load DustyCult artifacts: {exc}",
            output_dir=output_dir,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            config=config,
            window=prepared.window,
            quality=prepared.quality,
        )
        row["input_path"] = str(input_csv)
        row["config_path"] = str(config_path)
        upsert_dustycult_fit(conn, row)
        return row
