from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from malca.io.lightcurve_io import load_lightcurve_df


PHOEBE_FIT_TABLE = "phoebe_fits"
PHOEBE_FIT_COLUMNS = (
    "candidate_id",
    "status",
    "created_at",
    "updated_at",
    "runtime_sec",
    "model_kind",
    "period_days",
    "period_source",
    "manual_period_days",
    "t0_jd",
    "input_path",
    "n_input_points",
    "params_json",
    "metrics_json",
    "plot_json",
    "error",
    "phoebe_version",
)
PHOEBE_MODEL_KINDS = ("detached", "semidetached", "contact")
PHOEBE_DETACHED_FIT_PARAMETERS = (
    "incl@binary",
    "q@binary",
    "requiv@primary",
    "requiv@secondary",
)
PHOEBE_DEFAULT_MAX_ITERATIONS = 4
PHOEBE_DEFAULT_MAX_POINTS = 300


@dataclass(frozen=True)
class PhoebeAvailability:
    ok: bool
    message: str
    version: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _json_loads(value: object, default: Any) -> Any:
    if value in (None, "", b""):
        return default
    try:
        parsed = json.loads(str(value))
    except Exception:
        return default
    return parsed


def _sqlite_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return int(bool(value))
    return value


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _import_phoebe():
    return importlib.import_module("phoebe")


def check_phoebe_available() -> PhoebeAvailability:
    try:
        phoebe = _import_phoebe()
    except Exception as exc:
        return PhoebeAvailability(False, f"PHOEBE import failed: {exc}", None)
    version = str(getattr(phoebe, "__version__", "") or "unknown")
    return PhoebeAvailability(True, f"PHOEBE available ({version})", version)


def normalize_model_kind(value: object) -> str:
    text = str(value or "detached").strip().lower()
    return text if text in PHOEBE_MODEL_KINDS else "detached"


def infer_period_days(
    payload: Mapping[str, object] | None,
    manual_period_days: object | None = None,
) -> tuple[float | None, str]:
    """Return the best review-side EB period estimate and provenance label."""
    manual = _finite_positive(manual_period_days)
    if manual is not None:
        return manual, "manual"

    data = dict(payload or {})
    candidate_columns = (
        ("gaia_eb_period", "gaia_eb"),
        ("gaia_eb_period_days", "gaia_eb"),
        ("period_gaia_eb_period", "gaia_eb"),
        ("period_gaia_eb_period_days", "gaia_eb"),
        ("vsx_period_days", "vsx"),
        ("vsx_period", "vsx"),
        ("period_vsx_period", "vsx"),
        ("period_vsx_period_days", "vsx"),
        ("asassn_var_period_days", "asas_sn"),
        ("asassn_var_period", "asas_sn"),
        ("asassn_period", "asas_sn"),
        ("asassn_period_days", "asas_sn"),
        ("period_asassn_var_period", "asas_sn"),
        ("period_asassn_var_period_days", "asas_sn"),
        ("ztf_periodic_period_days", "ztf"),
        ("ztf_periodic_period", "ztf"),
        ("ztf_period", "ztf"),
        ("period_ztf_periodic_period", "ztf"),
        ("period_ztf_periodic_period_days", "ztf"),
        ("period_consensus_days", "period_consensus"),
        ("lsp_period", "lomb_scargle"),
        ("stats_variability_lomb_scargle_best_period_days", "lomb_scargle"),
        ("phase_period_days", "phase"),
    )
    for column, source in candidate_columns:
        period = _finite_positive(data.get(column))
        if period is not None:
            return period, source
    return None, "unavailable"


def _pick_numeric_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for name in candidates:
        col = lower_map.get(name.lower())
        if col is not None:
            values = pd.to_numeric(df[col], errors="coerce")
            if bool(np.isfinite(values).any()):
                return str(col)
    return None


def _prepare_lightcurve_frame(
    lc_path: str | Path,
    *,
    max_points: int = 2500,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = load_lightcurve_df(lc_path)
    if raw is None or raw.empty:
        raise ValueError(f"No light-curve rows loaded from {lc_path}")

    time_col = _pick_numeric_column(raw, ("JD", "jd", "HJD", "hjd", "MJD", "mjd", "time"))
    mag_col = _pick_numeric_column(raw, ("mag", "magnitude", "Mag"))
    err_col = _pick_numeric_column(raw, ("error", "mag_err", "magerr", "e_mag", "flux_err"))
    if time_col is None or mag_col is None:
        raise ValueError("Light curve must include finite time and magnitude columns.")

    work = pd.DataFrame(
        {
            "time": pd.to_numeric(raw[time_col], errors="coerce"),
            "mag": pd.to_numeric(raw[mag_col], errors="coerce"),
        }
    )
    if err_col is not None:
        work["mag_err"] = pd.to_numeric(raw[err_col], errors="coerce")
    else:
        work["mag_err"] = np.nan
    if "v_g_band" in raw.columns:
        work["band"] = raw["v_g_band"].map(lambda value: "V" if str(value).strip() == "1" else "g")
    elif "band" in raw.columns:
        work["band"] = raw["band"].astype(str)
    else:
        work["band"] = ""

    finite = np.isfinite(work["time"]) & np.isfinite(work["mag"])
    work = work.loc[finite].sort_values("time").reset_index(drop=True)
    if work.empty:
        raise ValueError("No finite light-curve points remain after cleaning.")

    if len(work) > max_points:
        keep = np.linspace(0, len(work) - 1, int(max_points), dtype=int)
        work = work.iloc[keep].reset_index(drop=True)

    ref_mag = float(np.nanmedian(work["mag"]))
    flux = np.power(10.0, -0.4 * (work["mag"].to_numpy(dtype=float) - ref_mag))
    mag_err = work["mag_err"].to_numpy(dtype=float)
    finite_err = np.isfinite(mag_err) & (mag_err > 0)
    fallback_mag_err = float(np.nanmedian(mag_err[finite_err])) if bool(finite_err.any()) else 0.02
    mag_err = np.where(finite_err, mag_err, fallback_mag_err)
    flux_err = np.log(10.0) * 0.4 * flux * mag_err

    work["relative_flux"] = flux
    work["relative_flux_error"] = flux_err
    meta = {
        "time_column": time_col,
        "mag_column": mag_col,
        "error_column": err_col or "",
        "reference_mag": ref_mag,
        "n_raw_points": int(len(raw)),
        "n_prepared_points": int(len(work)),
    }
    return work, meta


def _sample_for_plot(values: np.ndarray, max_points: int = 1600) -> np.ndarray:
    if values.size <= max_points:
        return np.arange(values.size, dtype=int)
    return np.linspace(0, values.size - 1, max_points, dtype=int)


def _try_set_value(bundle: object, qualifier: str, value: object) -> bool:
    try:
        bundle.set_value(qualifier, value)
        return True
    except Exception:
        return False


def _extract_model_flux(bundle: object, n_points: int) -> np.ndarray | None:
    if not hasattr(bundle, "get_value"):
        return None
    keys = (
        "fluxes@phoebe_model@model",
        "fluxes@lc01@phoebe_model",
        "fluxes@model",
        "fluxes@lc01@model",
    )
    for key in keys:
        try:
            values = np.asarray(bundle.get_value(key), dtype=float)
        except Exception:
            continue
        if values.size == n_points and bool(np.isfinite(values).any()):
            return values
    return None


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_min": float(np.nanmin(finite)),
        f"{prefix}_median": float(np.nanmedian(finite)),
        f"{prefix}_max": float(np.nanmax(finite)),
    }


def _normalize_model_flux(
    observed_flux: np.ndarray,
    model_flux: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    observed = np.asarray(observed_flux, dtype=float)
    raw_model = np.asarray(model_flux, dtype=float)
    valid_observed = observed[np.isfinite(observed)]
    valid_model = raw_model[np.isfinite(raw_model)]
    observed_median = float(np.nanmedian(valid_observed)) if valid_observed.size else np.nan
    model_median = float(np.nanmedian(valid_model)) if valid_model.size else np.nan
    if np.isfinite(observed_median) and np.isfinite(model_median) and model_median != 0:
        scale = float(observed_median / model_median)
    else:
        scale = 1.0
    normalized = raw_model * scale
    meta: dict[str, object] = {
        "model_flux_scale": scale,
        **_finite_summary(observed, "observed_flux"),
        **_finite_summary(raw_model, "model_flux_raw"),
        **_finite_summary(normalized, "model_flux_normalized"),
    }
    return normalized, meta


def _run_phoebe_model(
    phoebe: object,
    frame: pd.DataFrame,
    *,
    period_days: float,
    t0_jd: float,
    model_kind: str,
    max_iterations: int,
) -> tuple[np.ndarray | None, dict[str, object], dict[str, object]]:
    bundle = phoebe.default_binary()
    times = frame["time"].to_numpy(dtype=float)
    time_zero = float(np.nanmin(times))
    times_rel = times - time_zero
    flux = frame["relative_flux"].to_numpy(dtype=float)
    flux_err = frame["relative_flux_error"].to_numpy(dtype=float)

    params: dict[str, object] = {
        "time_zero_jd": time_zero,
        "period_days": period_days,
        "t0_jd": t0_jd,
        "t0_relative_days": float(t0_jd - time_zero),
        "model_kind": model_kind,
        "max_iterations": int(max_iterations),
    }
    set_results = {
        "period@binary": _try_set_value(bundle, "period@binary", period_days),
        "t0_supconj@binary": _try_set_value(bundle, "t0_supconj@binary", float(t0_jd - time_zero)),
    }
    params["set_results"] = set_results

    add_dataset_kwargs = {
        "times": times_rel,
        "fluxes": flux,
        "sigmas": flux_err,
        "dataset": "lc01",
        "overwrite": True,
    }
    try:
        bundle.add_dataset("lc", **add_dataset_kwargs)
    except TypeError:
        add_dataset_kwargs.pop("overwrite", None)
        bundle.add_dataset("lc", **add_dataset_kwargs)

    solver_status = "not_requested"
    solver_warning = ""
    fit_parameters = list(PHOEBE_DETACHED_FIT_PARAMETERS)
    params["fit_parameters"] = fit_parameters
    if hasattr(bundle, "add_solver") and hasattr(bundle, "run_solver") and max_iterations > 0:
        try:
            bundle.add_solver(
                "optimizer.nelder_mead",
                solver="nm_fit",
                maxiter=int(max_iterations),
                overwrite=True,
            )
            if not _try_set_value(bundle, "fit_parameters@nm_fit", fit_parameters):
                raise RuntimeError("failed to set fit_parameters@nm_fit")
            if hasattr(bundle, "run_checks"):
                checks = bundle.run_checks(solver="nm_fit")
                passed = bool(getattr(checks, "passed", False))
                params["solver_checks_passed"] = passed
                if not passed:
                    raise RuntimeError(f"PHOEBE solver checks failed: {checks}")
            bundle.run_solver(solver="nm_fit", solution="nm_solution", overwrite=True)
            if hasattr(bundle, "adopt_solution"):
                bundle.adopt_solution("nm_solution")
            solver_status = "ok"
        except Exception as exc:
            solver_status = f"skipped:{exc}"
            solver_warning = str(exc)
    params["solver_status"] = solver_status
    params["solver_warning"] = solver_warning

    compute_status = "not_run"
    try:
        bundle.run_compute(model="phoebe_model", overwrite=True)
        compute_status = "ok"
    except Exception as exc:
        compute_status = f"failed:{exc}"
    params["compute_status"] = compute_status

    model_flux = _extract_model_flux(bundle, len(frame))
    return model_flux, params, {
        "solver_status": solver_status,
        "solver_warning": solver_warning,
        "compute_status": compute_status,
    }


def _build_plot_payload(
    frame: pd.DataFrame,
    *,
    period_days: float,
    model_flux: np.ndarray | None,
) -> dict[str, object]:
    values = frame[["time", "relative_flux", "relative_flux_error"]].to_numpy(dtype=float)
    keep = _sample_for_plot(np.arange(len(frame), dtype=float))
    times = values[keep, 0]
    phase = ((times - times.min()) / period_days) % 1.0 if period_days > 0 else np.full_like(times, np.nan)
    payload = {
        "time": times.tolist(),
        "phase": phase.tolist(),
        "flux": values[keep, 1].tolist(),
        "flux_err": values[keep, 2].tolist(),
    }
    if model_flux is not None and len(model_flux) == len(frame):
        payload["model_flux"] = np.asarray(model_flux, dtype=float)[keep].tolist()
    return payload


def upsert_phoebe_fit(conn: sqlite3.Connection, fit_row: Mapping[str, object]) -> None:
    row = {col: fit_row.get(col) for col in PHOEBE_FIT_COLUMNS}
    row["candidate_id"] = str(row.get("candidate_id") or "")
    now = _utc_now()
    row["updated_at"] = row.get("updated_at") or now
    row["created_at"] = row.get("created_at") or now
    placeholders = ", ".join(["?"] * len(PHOEBE_FIT_COLUMNS))
    assignments = ", ".join(
        f"{col}=excluded.{col}"
        for col in PHOEBE_FIT_COLUMNS
        if col not in {"candidate_id", "created_at"}
    )
    sql = (
        f"INSERT INTO {PHOEBE_FIT_TABLE} ({', '.join(PHOEBE_FIT_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(candidate_id) DO UPDATE SET {assignments}"
    )
    conn.execute(sql, [_sqlite_value(row[col]) for col in PHOEBE_FIT_COLUMNS])
    conn.commit()


def load_phoebe_fits(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(PHOEBE_FIT_COLUMNS)} FROM {PHOEBE_FIT_TABLE} WHERE candidate_id = ?",
            conn,
            params=(str(candidate_id),),
        )
    except Exception:
        return pd.DataFrame(columns=PHOEBE_FIT_COLUMNS)


def _failure_row(
    *,
    candidate_id: str,
    started_at: float,
    model_kind: str,
    manual_period_days: object,
    period_days: float | None,
    period_source: str,
    input_path: str | None,
    error: str,
    phoebe_version: str | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": str(candidate_id),
        "status": "failed",
        "runtime_sec": float(time.monotonic() - started_at),
        "model_kind": normalize_model_kind(model_kind),
        "manual_period_days": _finite_positive(manual_period_days),
        "period_days": period_days,
        "period_source": period_source,
        "input_path": str(input_path or ""),
        "n_input_points": 0,
        "params_json": _json_dumps({}),
        "metrics_json": _json_dumps({}),
        "plot_json": _json_dumps({}),
        "error": str(error),
        "phoebe_version": phoebe_version,
    }


def run_phoebe_fit(
    conn: sqlite3.Connection,
    candidate_id: str,
    payload: Mapping[str, object] | None,
    *,
    lc_path: str | Path | None,
    manual_period_days: object | None = None,
    model_kind: str = "detached",
    max_iterations: int = PHOEBE_DEFAULT_MAX_ITERATIONS,
    max_points: int = PHOEBE_DEFAULT_MAX_POINTS,
) -> dict[str, object]:
    """Run a bounded PHOEBE model pass for one review candidate and persist it."""
    started_at = time.monotonic()
    cid = str(candidate_id)
    kind = normalize_model_kind(model_kind)
    period_days, period_source = infer_period_days(payload, manual_period_days)

    if lc_path is None or not str(lc_path).strip():
        row = _failure_row(
            candidate_id=cid,
            started_at=started_at,
            model_kind=kind,
            manual_period_days=manual_period_days,
            period_days=period_days,
            period_source=period_source,
            input_path=None,
            error="No local light-curve path is available.",
        )
        upsert_phoebe_fit(conn, row)
        return row
    if period_days is None:
        row = _failure_row(
            candidate_id=cid,
            started_at=started_at,
            model_kind=kind,
            manual_period_days=manual_period_days,
            period_days=None,
            period_source=period_source,
            input_path=str(lc_path),
            error="No positive EB period is available. Enter a period in days before running PHOEBE.",
        )
        upsert_phoebe_fit(conn, row)
        return row
    if kind != "detached":
        row = _failure_row(
            candidate_id=cid,
            started_at=started_at,
            model_kind=kind,
            manual_period_days=manual_period_days,
            period_days=period_days,
            period_source=period_source,
            input_path=str(lc_path),
            error=(
                "PHOEBE optimization currently supports only detached models; "
                f"{kind} is not implemented yet."
            ),
        )
        upsert_phoebe_fit(conn, row)
        return row

    try:
        phoebe = _import_phoebe()
    except Exception as exc:
        row = _failure_row(
            candidate_id=cid,
            started_at=started_at,
            model_kind=kind,
            manual_period_days=manual_period_days,
            period_days=period_days,
            period_source=period_source,
            input_path=str(lc_path),
            error=f"PHOEBE import failed: {exc}",
        )
        upsert_phoebe_fit(conn, row)
        return row

    phoebe_version = str(getattr(phoebe, "__version__", "") or "unknown")
    try:
        frame, prep_meta = _prepare_lightcurve_frame(lc_path, max_points=max_points)
        data = dict(payload or {})
        t0_jd = _finite_positive(data.get("dip_best_t0")) or float(np.nanmin(frame["time"]))
        model_flux, params, status_meta = _run_phoebe_model(
            phoebe,
            frame,
            period_days=period_days,
            t0_jd=t0_jd,
            model_kind=kind,
            max_iterations=max_iterations,
        )

        observed = frame["relative_flux"].to_numpy(dtype=float)
        sigma = frame["relative_flux_error"].to_numpy(dtype=float)
        model_scale_meta: dict[str, object] = {}
        plot_model: np.ndarray | None = None
        if model_flux is not None and len(model_flux) == len(observed):
            model, model_scale_meta = _normalize_model_flux(observed, np.asarray(model_flux, dtype=float))
            plot_model = model
            model_source = "phoebe"
        else:
            model = np.full_like(observed, float(np.nanmedian(observed)))
            model_source = "flat_fallback"
        residual = observed - model
        valid = np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0)
        chi2 = float(np.sum((residual[valid] / sigma[valid]) ** 2)) if bool(valid.any()) else np.nan
        dof = max(int(valid.sum()) - 4, 1)
        metrics = {
            "chi2": chi2,
            "reduced_chi2": float(chi2 / dof) if np.isfinite(chi2) else np.nan,
            "rms_residual": float(np.sqrt(np.nanmean(residual**2))),
            "mad_residual": float(np.nanmedian(np.abs(residual - np.nanmedian(residual)))),
            "model_flux_source": model_source,
            "n_model_points": int(len(observed)),
            **model_scale_meta,
            **status_meta,
        }
        params.update(prep_meta)
        params.update({"period_source": period_source, "manual_period_days": _finite_positive(manual_period_days)})
        params.update(model_scale_meta)

        compute_status = str(status_meta.get("compute_status") or "")
        solver_status = str(status_meta.get("solver_status") or "")
        if compute_status == "ok" and model_source == "phoebe" and solver_status == "ok":
            status = "ok"
            error = ""
        elif compute_status == "ok" and model_source == "phoebe":
            status = "warning"
            error = f"PHOEBE solver did not complete; diagnostic model only. {solver_status}".strip()
        else:
            status = "failed"
            error = compute_status if compute_status.startswith("failed:") else "PHOEBE model flux was unavailable."

        row = {
            "candidate_id": cid,
            "status": status,
            "runtime_sec": float(time.monotonic() - started_at),
            "model_kind": kind,
            "period_days": float(period_days),
            "period_source": period_source,
            "manual_period_days": _finite_positive(manual_period_days),
            "t0_jd": float(t0_jd),
            "input_path": str(lc_path),
            "n_input_points": int(len(frame)),
            "params_json": _json_dumps(params),
            "metrics_json": _json_dumps(metrics),
            "plot_json": _json_dumps(_build_plot_payload(frame, period_days=period_days, model_flux=plot_model)),
            "error": error,
            "phoebe_version": phoebe_version,
        }
    except Exception as exc:
        row = _failure_row(
            candidate_id=cid,
            started_at=started_at,
            model_kind=kind,
            manual_period_days=manual_period_days,
            period_days=period_days,
            period_source=period_source,
            input_path=str(lc_path),
            error=str(exc),
            phoebe_version=phoebe_version,
        )
    upsert_phoebe_fit(conn, row)
    return row


def parse_phoebe_json(value: object, default: Any | None = None) -> Any:
    return _json_loads(value, default if default is not None else {})
