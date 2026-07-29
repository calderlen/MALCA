#!/usr/bin/env python3
"""
Microlensing Injection-Recovery Pipeline

Injects synthetic Paczynski microlensing events into clean light curves and measures
the pipeline's detection efficiency as a function of tE and Amax.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from malca.config import SKYPATROL_CACHE_DIR
from malca.plotting.lightcurve_publication import apply_publication_rcparams, save_publication_figure, FIG_SINGLE_COL_HEATMAP, scaled_publication_text_sizes

import sys
sys.path.append(str(Path.cwd()))
from malca.evaluation.dip_injection import _resolve_lc_path
from malca.config import (
    INJECTION_MAX_ATTEMPTS,
    INJECTION_MAG_LO,
    INJECTION_MAG_HI,
    DEFAULT_OUTPUT_DIR,
)
from malca.evaluation.dip_injection import (
    ParquetAppendWriter,
    _write_checkpoint,
    _get_id_col,
    _processing_error_result,
    deterministic_trial_seed,
    summarize_injection_efficiency,
    _completed_trial_indices,
    _binned_efficiency_arrays,
    experiment_fingerprint,
    _assert_resume_fingerprint,
)

from scripts.microlensing import (
    fit_candidate_context,
    _prepare_lightcurve_df,
)

_GLOBAL: dict[str, object] = {}

def _solve_u0_from_A0(A0: float) -> float:
    """Find u0 corresponding to maximum magnification A0."""
    A_curr = float(A0)
    if A_curr <= 1.0:
        return float('inf')
    # Use exact formula if possible: A = (u^2 + 2)/(u * sqrt(u^2+4))
    # -> u^4 + 4u^2 - 4 / (A^2 - 1) = 0
    # -> u^2 = -2 + sqrt(4 + 4/(A^2-1))
    val = -2.0 + np.sqrt(4.0 + 4.0 / (A_curr**2 - 1.0))
    if val > 0:
        return float(np.sqrt(val))
    return 1e-8


def _Amax_from_u0(u0: float) -> float:
    u = float(u0)
    if not np.isfinite(u) or u <= 0:
        raise ValueError("u0 must be positive and finite")
    return float((u * u + 2.0) / (u * np.sqrt(u * u + 4.0)))


def _resolve_u0_range(
    *,
    u0_range: tuple[float, float] | None,
    Amax_range: tuple[float, float] | None,
) -> tuple[float, float]:
    if u0_range is not None and Amax_range is not None:
        raise ValueError("Specify u0_range or legacy Amax_range, not both")
    if u0_range is not None:
        lo, hi = map(float, u0_range)
    elif Amax_range is not None:
        amin, amax = map(float, Amax_range)
        if not (1.0 < amin <= amax):
            raise ValueError("Amax_range must be increasing and strictly above 1")
        # Amax decreases monotonically with impact parameter.
        lo, hi = _solve_u0_from_A0(amax), _solve_u0_from_A0(amin)
    else:
        raise ValueError("A physical u0_range is required")
    if not (0.0 < lo <= hi and np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("u0_range must be positive, finite, and increasing")
    return float(lo), float(hi)


def _paczynski_delta_mag(
    times: np.ndarray, *, t0: float, tE: float, u0: float
) -> np.ndarray:
    u_sq = float(u0) ** 2 + ((np.asarray(times, dtype=float) - float(t0)) / float(tE)) ** 2
    u = np.sqrt(u_sq)
    magnification = (u_sq + 2.0) / (u * np.sqrt(u_sq + 4.0))
    return -2.5 * np.log10(magnification)


def inject_paczynski(
    df_lc: pd.DataFrame,
    t0: float,
    tE: float,
    *,
    u0: float,
    mag_err_poly: np.poly1d | None = None,
    rng: np.random.Generator | None = None,
    mag_col: str = "mag",
    time_col: str = "JD",
    err_col: str = "error",
) -> pd.DataFrame:
    """Inject a physical Paczynski ``(t0, tE, u0)`` profile.

    The input is observed photometry, so its existing noise realization is
    preserved and no second random error draw is added.  ``mag_err_poly`` and
    ``rng`` remain accepted for compatibility with older callers.
    """
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out
        
    t = df_out[time_col].values
    mag_old = df_out[mag_col].values
    
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite")
    if not np.isfinite(tE) or float(tE) <= 0:
        raise ValueError("tE must be positive and finite")
    if not np.isfinite(u0) or float(u0) <= 0:
        raise ValueError("u0 must be positive and finite")
    # Calculate magnification A(t)
    dip_profile = _paczynski_delta_mag(t, t0=t0, tE=tE, u0=u0)
    
    df_out[mag_col] = mag_old + dip_profile
    
    return df_out


def _simulate_microlensing_trial(
    trial_index: int,
    *,
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    tE_range: tuple[float, float],
    mag_err_poly: np.poly1d | None,
    measure_pre_injection: bool,
    seed: int,
    Amax_range: tuple[float, float] | None = None,
    u0_range: tuple[float, float] | None = None,
    max_reduced_chi2: float = 10.0,
    max_t0_offset_tE: float = 1.0,
    recovered_tE_ratio_range: tuple[float, float] = (0.5, 2.0),
    experiment_id: str | None = None,
) -> dict:
    per_trial_seed = deterministic_trial_seed(seed, trial_index)
    rng = np.random.default_rng(per_trial_seed)
    if not np.isfinite(max_reduced_chi2) or float(max_reduced_chi2) <= 0:
        raise ValueError("max_reduced_chi2 must be positive and finite")
    if not np.isfinite(max_t0_offset_tE) or float(max_t0_offset_tE) <= 0:
        raise ValueError("max_t0_offset_tE must be positive and finite")
    ratio_lo, ratio_hi = map(float, recovered_tE_ratio_range)
    if not (0.0 < ratio_lo <= 1.0 <= ratio_hi and np.isfinite(ratio_hi)):
        raise ValueError("recovered_tE_ratio_range must be finite, positive, and bracket 1")

    resolved_u0_range = _resolve_u0_range(
        u0_range=u0_range, Amax_range=Amax_range
    )
    # Event trajectories are sampled in physical impact parameter, not in a
    # transformed/log-magnification coordinate.
    u0 = float(rng.uniform(resolved_u0_range[0], resolved_u0_range[1]))
    Amax = _Amax_from_u0(u0)
    tE = float(10 ** rng.uniform(np.log10(tE_range[0]), np.log10(tE_range[1])))
    base_values = {
        "experiment_seed": int(seed),
        "experiment_fingerprint": experiment_id,
        "Amax": Amax,
        "designed_Amax": Amax,
        "u0": u0,
        "designed_u0": u0,
        "tE": tE,
        "designed_tE_days": tE,
        "parameter_sampling": "uniform_u0_log_uniform_tE",
        "t0_sampling": "uniform_observed_time_span",
        "recovery_max_reduced_chi2": float(max_reduced_chi2),
        "recovery_definition": "quality_paczynski_fit_matched_in_t0_and_tE_and_not_control_recovery",
        "recovery_max_t0_offset_tE": float(max_t0_offset_tE),
        "recovery_tE_ratio_min": ratio_lo,
        "recovery_tE_ratio_max": ratio_hi,
    }

    for attempt in range(int(INJECTION_MAX_ATTEMPTS)):
        control_idx = int(rng.integers(0, len(control_ids)))
        asas_sn_id = str(control_ids[control_idx])
        raw_path = str(control_dirs[control_idx])
        try:
            from malca.io.fetch import download_lightcurve_by_id

            if not raw_path or raw_path == ".":
                lc_path, _ = download_lightcurve_by_id(
                    asas_sn_id, cache_dir=SKYPATROL_CACHE_DIR
                )
                if lc_path is None:
                    raise FileNotFoundError("lc_not_found")
            else:
                lc_path = Path(raw_path)
                if not lc_path.is_file():
                    files = sorted(lc_path.glob(f"*{asas_sn_id}*.dat"))
                    if not files:
                        raise FileNotFoundError("lc_not_found")
                    lc_path = files[0]

            df_lc, band_label = _prepare_lightcurve_df(lc_path, prefer_g_band=True)
            if df_lc.empty or len(df_lc) < 20:
                raise ValueError("empty_or_short_lc")
            median_mag = float(np.nanmedian(df_lc["mag"].values))
            if not INJECTION_MAG_LO <= median_mag <= INJECTION_MAG_HI:
                raise ValueError("magnitude_out_of_range")
            break
        except Exception as exc:
            if attempt == int(INJECTION_MAX_ATTEMPTS) - 1:
                return _processing_error_result(
                    trial_index,
                    per_trial_seed,
                    detected_key="recovered",
                    error_stage="load_control",
                    error=exc,
                    asas_sn_id=asas_sn_id,
                    control_attempts=int(attempt + 1),
                    **base_values,
                )

    injection_performed = False
    error_context: dict[str, object] = {
        "median_mag": median_mag,
        "control_attempts": int(attempt + 1),
        "n_points": int(len(df_lc)),
    }
    try:
        t_min = float(pd.to_numeric(df_lc["JD"], errors="coerce").min())
        t_max = float(pd.to_numeric(df_lc["JD"], errors="coerce").max())
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            raise ValueError("invalid_time_range")

        # Include edge/window losses instead of forcing t0 to be >2 tE from an
        # edge.  Conditional efficiency is reported from explicit support.
        t0 = float(rng.uniform(t_min, t_max))
        times = pd.to_numeric(df_lc["JD"], errors="coerce").to_numpy(dtype=float)
        delta_mag = _paczynski_delta_mag(times, t0=t0, tE=tE, u0=u0)
        theoretical_peak_mag = float(abs(-2.5 * np.log10(Amax)))
        observed_peak_mag = float(np.nanmax(np.abs(delta_mag)))
        observed_peak_fraction = (
            observed_peak_mag / theoretical_peak_mag if theoretical_peak_mag > 0 else np.nan
        )
        within_tE = np.abs(times - t0) <= tE
        within_2tE = np.abs(times - t0) <= 2.0 * tE
        n_within_tE = int(np.count_nonzero(within_tE))
        n_within_2tE = int(np.count_nonzero(within_2tE))
        overlap = max(0.0, min(t_max, t0 + 2.0 * tE) - max(t_min, t0 - 2.0 * tE))
        window_coverage_fraction = float(min(1.0, overlap / (4.0 * tE)))
        observable = bool(n_within_tE >= 1 and n_within_2tE >= 3)
        error_context.update(
            {
                "t0": t0,
                "t_center": t0,
                "designed_t0_jd": t0,
                "time_min_jd": t_min,
                "time_max_jd": t_max,
                "time_span_days": float(t_max - t_min),
                "n_points_within_tE": n_within_tE,
                "n_points_within_2tE": n_within_2tE,
                "observed_peak_fraction": float(observed_peak_fraction),
                "window_coverage_fraction": window_coverage_fraction,
                "window_coverage_definition": "fraction_of_t0_plusminus_2tE_inside_observed_span",
                "observable": observable,
                "observable_definition": "at_least_1_point_within_tE_and_3_points_within_2tE",
            }
        )

        def context(frame: pd.DataFrame, label: str) -> dict:
            return {
                "candidate_id": asas_sn_id,
                "asas_sn_id": asas_sn_id,
                "row": {},
                "payload": {
                    "candidate_id": asas_sn_id,
                    "ra_deg": 0.0,
                    "dec_deg": 0.0,
                },
                "lc_path": Path(f"{label}.dat"),
                "df": frame,
                "band_label": band_label,
            }

        def interpret_fit(fit_result: dict | None) -> dict:
            if fit_result is None:
                raise RuntimeError("fit_returned_none")
            summary = fit_result.get("summary", {})
            model_summary = (
                fit_result.get("best_seed_result", {})
                .get("fits", {})
                .get("paczynski", {})
            )
            fit_ok = bool(summary.get("fit_ok", False))
            best_model = summary.get("best_model")
            reduced = summary.get(
                "paczynski_reduced_chi2", model_summary.get("reduced_chi2", np.nan)
            )
            reduced_chi2 = float(reduced) if reduced is not None else np.nan
            recovered_tE = summary.get("raw_paczynski_tE_days", np.nan)
            recovered_t0 = summary.get("fit_t0_jd", np.nan)
            params = np.asarray(model_summary.get("params", []), dtype=float)
            recovered_Amax = float(params[0]) if params.size >= 1 else np.nan
            recovered_u0 = (
                float(_solve_u0_from_A0(recovered_Amax))
                if np.isfinite(recovered_Amax) and recovered_Amax > 1.0
                else np.nan
            )
            recovered = bool(
                fit_ok
                and best_model == "paczynski"
                and np.isfinite(reduced_chi2)
                and reduced_chi2 < float(max_reduced_chi2)
            )
            return {
                "recovered": recovered,
                "fit_ok": fit_ok,
                "best_model": None if best_model is None else str(best_model),
                "reduced_chi2": reduced_chi2,
                "recovered_tE": float(recovered_tE) if recovered_tE is not None else np.nan,
                "recovered_t0_jd": float(recovered_t0) if recovered_t0 is not None else np.nan,
                "recovered_Amax": recovered_Amax,
                "recovered_u0": recovered_u0,
            }

        pre_result = {
            "paired_control_evaluated": False,
            "pre_injection_recovered": None,
        }
        if measure_pre_injection:
            interpreted_pre = interpret_fit(fit_candidate_context(context(df_lc, "control")))
            pre_result = {
                "paired_control_evaluated": True,
                **{f"pre_injection_{key}": value for key, value in interpreted_pre.items()},
            }

        df_injected = inject_paczynski(
            df_lc, t0, tE, u0=u0, mag_err_poly=mag_err_poly, rng=rng
        )
        injection_performed = True
        error_context.update(
            {
                "injected_Amax": Amax,
                "injected_u0": u0,
                "injected_tE_days": tE,
                "injected_t0_jd": t0,
            }
        )
        interpreted = interpret_fit(fit_candidate_context(context(df_injected, "injected")))
        pre_recovered = pre_result.get("pre_injection_recovered")
        post_model_recovered = bool(interpreted["recovered"])
        recovered_t0 = float(interpreted.get("recovered_t0_jd", np.nan))
        recovered_tE = float(interpreted.get("recovered_tE", np.nan))
        t0_offset_days = (
            float(abs(recovered_t0 - t0)) if np.isfinite(recovered_t0) else np.nan
        )
        t0_offset_in_tE = t0_offset_days / tE if np.isfinite(t0_offset_days) else np.nan
        recovered_tE_ratio = recovered_tE / tE if np.isfinite(recovered_tE) else np.nan
        parameter_match = bool(
            post_model_recovered
            and np.isfinite(t0_offset_in_tE)
            and t0_offset_in_tE <= float(max_t0_offset_tE)
            and np.isfinite(recovered_tE_ratio)
            and ratio_lo <= recovered_tE_ratio <= ratio_hi
        )
        post_recovered = parameter_match
        paired_recovered = (
            post_recovered and not bool(pre_recovered)
            if pre_recovered is not None
            else post_recovered
        )
        return {
            "trial_index": int(trial_index),
            "trial_seed": int(per_trial_seed),
            "trial_status": "completed",
            "processing_error": False,
            "injection_performed": True,
            "error_stage": None,
            "error": None,
            **base_values,
            "t0": t0,
            "t_center": t0,
            "designed_t0_jd": t0,
            "injected_t0_jd": t0,
            "injected_Amax": Amax,
            "injected_u0": u0,
            "injected_tE_days": tE,
            "median_mag": median_mag,
            "asas_sn_id": asas_sn_id,
            "control_attempts": int(attempt + 1),
            "n_points": int(len(df_injected)),
            "time_min_jd": t_min,
            "time_max_jd": t_max,
            "time_span_days": float(t_max - t_min),
            "n_points_within_tE": n_within_tE,
            "n_points_within_2tE": n_within_2tE,
            "observed_peak_fraction": float(observed_peak_fraction),
            "window_coverage_fraction": window_coverage_fraction,
            "window_coverage_definition": "fraction_of_t0_plusminus_2tE_inside_observed_span",
            "observable": observable,
            "observable_definition": "at_least_1_point_within_tE_and_3_points_within_2tE",
            **interpreted,
            **pre_result,
            "post_injection_recovered": post_recovered,
            "post_injection_model_recovered": post_model_recovered,
            "post_injection_parameter_match": parameter_match,
            "recovery_t0_offset_days": t0_offset_days,
            "recovery_t0_offset_in_tE": t0_offset_in_tE,
            "recovered_to_injected_tE_ratio": recovered_tE_ratio,
            "recovered": paired_recovered,
            "recovered_above_paired_control": paired_recovered,
        }
    except Exception as exc:
        return _processing_error_result(
            trial_index,
            per_trial_seed,
            detected_key="recovered",
            error_stage="inject_or_fit",
            error=exc,
            asas_sn_id=asas_sn_id,
            injection_performed=injection_performed,
            **base_values,
            **error_context,
        )


def _init_worker(
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    Amax_range: tuple[float, float] | None,
    u0_range: tuple[float, float] | None,
    tE_range: tuple[float, float],
    mag_err_poly: np.poly1d | None,
    measure_pre_injection: bool,
    seed: int,
    max_reduced_chi2: float,
    max_t0_offset_tE: float,
    recovered_tE_ratio_range: tuple[float, float],
    experiment_id: str | None = None,
) -> None:
    _GLOBAL["control_ids"] = control_ids
    _GLOBAL["control_dirs"] = control_dirs
    _GLOBAL["Amax_range"] = Amax_range
    _GLOBAL["u0_range"] = u0_range
    _GLOBAL["tE_range"] = tE_range
    _GLOBAL["mag_err_poly"] = mag_err_poly
    _GLOBAL["measure_pre_injection"] = measure_pre_injection
    _GLOBAL["seed"] = seed
    _GLOBAL["max_reduced_chi2"] = max_reduced_chi2
    _GLOBAL["max_t0_offset_tE"] = max_t0_offset_tE
    _GLOBAL["recovered_tE_ratio_range"] = recovered_tE_ratio_range
    _GLOBAL["experiment_id"] = experiment_id


def _process_trial_batch(trial_indices: list[int]) -> list[dict]:
    results = []
    for trial_index in trial_indices:
        results.append(
            _simulate_microlensing_trial(
                trial_index,
                control_ids=_GLOBAL["control_ids"],
                control_dirs=_GLOBAL["control_dirs"],
                Amax_range=_GLOBAL["Amax_range"],
                u0_range=_GLOBAL["u0_range"],
                tE_range=_GLOBAL["tE_range"],
                mag_err_poly=_GLOBAL["mag_err_poly"],
                measure_pre_injection=bool(_GLOBAL["measure_pre_injection"]),
                seed=int(_GLOBAL["seed"]),
                max_reduced_chi2=float(_GLOBAL["max_reduced_chi2"]),
                max_t0_offset_tE=float(_GLOBAL["max_t0_offset_tE"]),
                recovered_tE_ratio_range=tuple(_GLOBAL["recovered_tE_ratio_range"]),
                experiment_id=_GLOBAL.get("experiment_id"),
            )
        )
    return results


def run_microlensing_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    total_trials: int = 1000,
    u0_range: tuple[float, float] | None = (0.01, 1.0),
    Amax_range: tuple[float, float] | None = None,
    tE_range: tuple[float, float] = (1.0, 300.0),
    measure_pre_injection: bool = True,
    mag_err_order: int = 5,
    mag_err_sample: int = 100,
    seed: int = 42,
    max_reduced_chi2: float = 10.0,
    max_t0_offset_tE: float = 1.0,
    recovered_tE_ratio_range: tuple[float, float] = (0.5, 2.0),
    workers: int = 1,
    task_size: int = 10,
    checkpoint_interval: int = 1000,
    chunk_size: int = 100,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame | None:
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite:
            output_path.unlink()
        if output_path.exists() and not resume and not overwrite:
            raise SystemExit(f"Output exists: {output_path} (use --overwrite or --no-resume)")
    if overwrite:
        resume = False

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    elif output_path is not None:
        checkpoint_path = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    if checkpoint_path and checkpoint_path.exists() and overwrite:
        checkpoint_path.unlink()

    if int(total_trials) < 1:
        raise ValueError("total_trials must be positive")
    # Preserve legacy programmatic callers that set Amax_range without knowing
    # about the new default u0_range.
    effective_u0_range = (
        None if Amax_range is not None and u0_range == (0.01, 1.0) else u0_range
    )
    resolved_u0_range = _resolve_u0_range(
        u0_range=effective_u0_range, Amax_range=Amax_range
    )
    if not (0.0 < float(tE_range[0]) <= float(tE_range[1])):
        raise ValueError("tE_range must be positive and increasing")
    if int(workers) < 1 or int(task_size) < 1 or int(checkpoint_interval) < 1:
        raise ValueError("workers, task_size, and checkpoint_interval must be positive")
    if not np.isfinite(max_reduced_chi2) or float(max_reduced_chi2) <= 0:
        raise ValueError("max_reduced_chi2 must be positive and finite")
    if not np.isfinite(max_t0_offset_tE) or float(max_t0_offset_tE) <= 0:
        raise ValueError("max_t0_offset_tE must be positive and finite")
    ratio_lo, ratio_hi = map(float, recovered_tE_ratio_range)
    if not (0.0 < ratio_lo <= 1.0 <= ratio_hi and np.isfinite(ratio_hi)):
        raise ValueError("recovered_tE_ratio_range must be finite, positive, and bracket 1")
    recovered_tE_ratio_range = (ratio_lo, ratio_hi)
    if int(INJECTION_MAX_ATTEMPTS) < 1:
        raise ValueError("INJECTION_MAX_ATTEMPTS must be positive")

    id_col = _get_id_col(control_sample)
    control_ids = control_sample[id_col].astype(str).to_numpy()
    control_dirs = []
    for _, row in control_sample.iterrows():
        lc_dir = _resolve_lc_path(row)
        if lc_dir is None:
            control_dirs.append("")
        else:
            control_dirs.append(str(lc_dir))
    control_dirs = np.asarray(control_dirs, dtype=object)

    if len(control_ids) == 0:
        raise SystemExit("Control sample is empty.")

    # Preserve the observed noise realization; do not fit/use a model to draw a
    # second layer of synthetic measurement noise.
    mag_err_poly = None

    tE_range = (float(tE_range[0]), float(tE_range[1]))
    experiment_id = experiment_fingerprint(
        {
            "contract_version": 2,
            "family": "paczynski",
            "seed": int(seed),
            "u0_range": resolved_u0_range,
            "tE_range": tE_range,
            "control_ids": control_ids,
            "control_paths": control_dirs,
            "paired_control": bool(measure_pre_injection),
            "fit_function": fit_candidate_context,
            "fit_reduced_chi2_max": float(max_reduced_chi2),
            "recovery_max_t0_offset_tE": float(max_t0_offset_tE),
            "recovery_tE_ratio_range": recovered_tE_ratio_range,
            "max_attempts": int(INJECTION_MAX_ATTEMPTS),
            "magnitude_limits": [float(INJECTION_MAG_LO), float(INJECTION_MAG_HI)],
        }
    )
    if resume:
        _assert_resume_fingerprint(output_path, experiment_id)
    completed_indices = _completed_trial_indices(output_path) if resume else set()
    invalid_indices = sorted(i for i in completed_indices if i < 0 or i >= int(total_trials))
    if invalid_indices:
        raise ValueError(
            "Resumable output contains trial indices outside the requested design: "
            f"{invalid_indices[:10]}"
        )
    pending_indices = [i for i in range(int(total_trials)) if i not in completed_indices]
    if not pending_indices:
        print("All designed trials are already present in the output table.")
        return None

    writer = ParquetAppendWriter(output_path) if output_path else None
    results: list[dict] = []

    pbar = tqdm(total=total_trials, initial=len(completed_indices), disable=not show_progress)

    def flush_results(is_final: bool = False) -> None:
        nonlocal results
        if not results:
            return
        if writer is None:
            return
        writer.write_chunk(results)
        if is_final:
            writer.close()
        results = []

    if workers <= 1:
        for trial_index in pending_indices:
            res = _simulate_microlensing_trial(
                trial_index,
                control_ids=control_ids,
                control_dirs=control_dirs,
                Amax_range=None,
                u0_range=resolved_u0_range,
                tE_range=tE_range,
                mag_err_poly=mag_err_poly,
                measure_pre_injection=measure_pre_injection,
                seed=seed,
                max_reduced_chi2=max_reduced_chi2,
                max_t0_offset_tE=max_t0_offset_tE,
                recovered_tE_ratio_range=recovered_tE_ratio_range,
                experiment_id=experiment_id,
            )
            results.append(res)
            pbar.update(1)
            if chunk_size and len(results) >= chunk_size:
                flush_results()
            if checkpoint_path and (trial_index + 1) % checkpoint_interval == 0:
                flush_results()
                _write_checkpoint(checkpoint_path, trial_index)
        flush_results(is_final=True)
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, total_trials - 1)
        pbar.close()
        if output_path:
            return None
        return pd.DataFrame(results).sort_values("trial_index", kind="stable").reset_index(drop=True)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            control_ids,
            control_dirs,
            None,
            resolved_u0_range,
            tE_range,
            mag_err_poly,
            measure_pre_injection,
            seed,
            max_reduced_chi2,
            max_t0_offset_tE,
            recovered_tE_ratio_range,
            experiment_id,
        ),
    ) as ex:
        for offset in range(0, len(pending_indices), checkpoint_interval):
            batch_indices = pending_indices[offset:offset + checkpoint_interval]
            tasks = [batch_indices[i:i + task_size] for i in range(0, len(batch_indices), task_size)]

            futures = {ex.submit(_process_trial_batch, task): task for task in tasks}
            for fut in as_completed(futures):
                batch_results = sorted(fut.result(), key=lambda row: int(row["trial_index"]))
                results.extend(batch_results)
                pbar.update(len(batch_results))
                if chunk_size and len(results) >= chunk_size:
                    flush_results()

            flush_results()
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, max(batch_indices))

    flush_results(is_final=True)
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, total_trials - 1)
    pbar.close()
    if output_path:
        return None
    return pd.DataFrame(results).sort_values("trial_index", kind="stable").reset_index(drop=True)


def compute_microlensing_efficiency_grid(
    df: pd.DataFrame,
    *,
    bins_tE: int = 15,
    bins_Amax: int = 15,
    confidence: float = 0.95,
) -> dict:
    """Compute unsmoothed microlensing efficiency, counts, and intervals."""
    if int(bins_tE) < 1 or int(bins_Amax) < 1:
        raise ValueError("Efficiency-grid bin counts must be positive")
    tE = pd.to_numeric(df["tE"], errors="coerce").to_numpy(dtype=float)
    Amax = pd.to_numeric(df["Amax"], errors="coerce").to_numpy(dtype=float)
    valid_tE = tE[np.isfinite(tE) & (tE > 0)]
    valid_Amax = Amax[np.isfinite(Amax) & (Amax > 1)]
    if valid_tE.size == 0 or valid_Amax.size == 0:
        raise ValueError("No finite positive tE/Amax values available for efficiency grid")

    def edges(values: np.ndarray, bins: int) -> np.ndarray:
        lo, hi = float(np.log10(values.min())), float(np.log10(values.max()))
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        return np.linspace(lo, hi, int(bins) + 1)

    tE_edges = edges(valid_tE, bins_tE)
    Amax_edges = edges(valid_Amax, bins_Amax)
    recovered = df["recovered"].astype("boolean")
    processing_error = df.get(
        "processing_error", df.get("error", pd.Series(None, index=df.index)).notna()
    )
    processing_error = pd.Series(processing_error, index=df.index).fillna(False).astype(bool).to_numpy()
    observable = df.get("observable", pd.Series(True, index=df.index))
    observable = pd.Series(observable, index=df.index).fillna(False).astype(bool).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sample = np.column_stack([np.log10(tE), np.log10(Amax)])
    arrays = _binned_efficiency_arrays(
        log_sample,
        recovered,
        processing_error,
        observable,
        [tE_edges, Amax_edges],
        confidence=confidence,
    )
    return {
        **arrays,
        "log_tE_edges": tE_edges,
        "log_Amax_edges": Amax_edges,
        "log_tE_centers": (tE_edges[:-1] + tE_edges[1:]) / 2.0,
        "log_Amax_centers": (Amax_edges[:-1] + Amax_edges[1:]) / 2.0,
    }


def plot_efficiency_map(
    df: pd.DataFrame,
    out_path: Path,
    bins_tE: int = 15,
    bins_Amax: int = 15,
    min_bin_trials: int = 5,
) -> dict:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    apply_publication_rcparams(plt)
    grid = compute_microlensing_efficiency_grid(
        df, bins_tE=bins_tE, bins_Amax=bins_Amax
    )
    stat = np.asarray(grid["efficiency_end_to_end"], dtype=float)
    counts = np.asarray(grid["n_designed"], dtype=int)
    stat[counts < max(1, int(min_bin_trials))] = np.nan
    plotted_eff = np.ma.masked_invalid(stat.T)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    text = scaled_publication_text_sizes(FIG_SINGLE_COL_HEATMAP)
    text["label"] = 10.0
    
    xc = grid["log_tE_centers"]
    yc = grid["log_Amax_centers"]
    log_tE = np.log10(pd.to_numeric(df["tE"], errors="coerce").dropna().to_numpy())
    log_Amax = np.log10(pd.to_numeric(df["Amax"], errors="coerce").dropna().to_numpy())
    
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color='0.9')
    im = ax.pcolormesh(
        grid["log_tE_edges"],
        grid["log_Amax_edges"],
        plotted_eff,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=True,
    )
    
    # Add contours
    mask = ~np.isnan(stat.T)
    if np.any(mask):
        try:
            cs = ax.contour(
                xc,
                yc,
                plotted_eff,
                levels=[0.5, 0.9, 0.99],
                colors='black',
                alpha=0.9,
                linewidths=0.6
            )
            
            # Find the longest continuous segment for each contour level to place exactly one label
            label_locations = []
            for p in cs.get_paths():
                polys = p.to_polygons()
                if not polys:
                    continue
                longest_poly = max(polys, key=lambda poly: len(poly))
                mid_idx = len(longest_poly) // 2
                midpoint = longest_poly[mid_idx]
                label_locations.append((midpoint[0], midpoint[1]))
            
            if label_locations:
                texts = ax.clabel(cs, inline=True, inline_spacing=4, fontsize=text["label"]*0.8, fmt='%g', manual=label_locations)
                for t in texts:
                    t.set_rotation(0)
        except Exception:
            pass
        
    divider = make_axes_locatable(ax)
    
    left_lim = min(0.0, log_tE.min())
    bottom_lim = min(0.0, log_Amax.min())
    
    ax.set_xlim(left=left_lim, right=log_tE.max())
    ax.set_ylim(bottom=bottom_lim, top=log_Amax.max())
    
    ax.set_xlabel(r'$t_E$ [days]', fontsize=text["label"])
    ax.set_ylabel(r'$A_{max}$', fontsize=text["label"])
    ax.tick_params(axis="y", labelleft=True)
    
    # Manually configure logarithmic ticks on the linear axes
    import matplotlib.ticker as ticker
    def set_log_ticks_on_linear_axis(axis_obj, vmin, vmax):
        # Only create major ticks strictly within the plot limits (or slightly outside to allow margin)
        major_ticks = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
        # Filter out major ticks that are way outside the limits to avoid extending the axis
        major_ticks = [x for x in major_ticks if x <= vmax + 0.1 and x >= vmin - 0.1]
        
        axis_obj.set_ticks(major_ticks)
        axis_obj.set_ticklabels([rf"$10^{{{int(x)}}}$" for x in major_ticks])
        
        minor_ticks = []
        for power in np.arange(np.floor(vmin)-1, np.ceil(vmax)+1):
            for mult in range(2, 10):
                val = np.log10(mult * 10**power)
                if vmin <= val <= vmax:
                    minor_ticks.append(val)
        axis_obj.set_ticks(minor_ticks, minor=True)

    set_log_ticks_on_linear_axis(ax.xaxis, left_lim, log_tE.max())
    set_log_ticks_on_linear_axis(ax.yaxis, bottom_lim, log_Amax.max())
    
    # Add a secondary y-axis for magnitude drop (Delta m)
    def logA_to_dm(logA):
        return 2.5 * logA

    def dm_to_logA(dm):
        return dm / 2.5

    secax = ax.secondary_yaxis('right', functions=(logA_to_dm, dm_to_logA))
    dm_ticks = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
    # Only keep dm_ticks that fall within the Y axis range
    dm_ticks = [dm for dm in dm_ticks if dm_to_logA(dm) <= log_Amax.max()]
    secax.set_yticks(dm_ticks)
    secax.set_yticklabels([f"{dm:.1f}" for dm in dm_ticks])
    secax.set_ylabel(r"$\Delta m$ [mag]", fontsize=text["label"])
    secax.tick_params(axis="y", labelsize=text["label"]*0.75)

    # STRICTLY set the axis limits at the very end to prevent matplotlib from autoscaling to the ticks!
    ax.set_xlim(left=log_tE.min(), right=log_tE.max())
    ax.set_ylim(bottom=log_Amax.min(), top=log_Amax.max())

    cax = divider.append_axes("right", size="7%", pad=0.7)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    import matplotlib.ticker as ticker
    cax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cbar.set_label("Efficiency", fontsize=text["label"], labelpad=8)
    cbar.ax.tick_params(labelsize=text["label"]*0.6)
    
    fig.tight_layout()
    save_publication_figure(fig, out_path, dpi=300)
    print(f"Efficiency map saved to {out_path}")
    return grid

def calculate_event_rate(
    df: pd.DataFrame,
    *,
    n_observed_events: int,
    exposure_star_years: float | None = None,
    n_stars_monitored: float | None = None,
    duration_years: float | None = None,
    efficiency_estimand: str = "end_to_end",
    confidence: float = 0.95,
) -> dict:
    """Calculate a rate only from explicit event count and survey exposure.

    No catalog size, survey duration, or observed-event count is assumed.  The
    caller may supply exposure directly, or supply both star count and duration.
    """
    observed_value = float(n_observed_events)
    if not np.isfinite(observed_value) or observed_value < 0 or not observed_value.is_integer():
        raise ValueError("n_observed_events must be a non-negative integer")
    observed_count = int(observed_value)
    if exposure_star_years is None:
        if n_stars_monitored is None or duration_years is None:
            raise ValueError(
                "Provide exposure_star_years or both n_stars_monitored and duration_years"
            )
        if float(n_stars_monitored) <= 0 or float(duration_years) <= 0:
            raise ValueError("n_stars_monitored and duration_years must both be positive")
        exposure_star_years = float(n_stars_monitored) * float(duration_years)
    exposure = float(exposure_star_years)
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("Survey exposure must be positive and finite")

    summary = summarize_injection_efficiency(
        df, detected_col="recovered", confidence=confidence
    )
    key = {
        "end_to_end": "end_to_end",
        "completed": "completed",
        "conditional_observable": "conditional_observable",
    }.get(str(efficiency_estimand))
    if key is None:
        raise ValueError(
            "efficiency_estimand must be end_to_end, completed, or conditional_observable"
        )
    efficiency = summary[key]
    value = efficiency["efficiency"]
    if value is None or value <= 0:
        raise ValueError(f"{key} recovery efficiency is zero or undefined")
    corrected_events = float(observed_count) / float(value)
    rate = corrected_events / exposure
    result = {
        "n_observed_events": observed_count,
        "exposure_star_years": exposure,
        "efficiency_estimand": key,
        "recovery_efficiency": float(value),
        "recovery_efficiency_ci_low": efficiency["ci_low"],
        "recovery_efficiency_ci_high": efficiency["ci_high"],
        "efficiency_corrected_events": corrected_events,
        "event_rate_per_star_year": float(rate),
    }
    print("\n--- Microlensing Event Rate Estimate ---")
    print(f"Recovery efficiency ({key}): {value:.2%}")
    print(f"Observed events: {observed_count}")
    print(f"Explicit exposure: {exposure:.6g} star-years")
    print(f"Estimated event rate: {rate:.3e} events/star/year")
    print("----------------------------------------\n")
    return result

def main():
    parser = argparse.ArgumentParser(description='Run microlensing injection-recovery and generate efficiency map')
    parser.add_argument('--manifest', type=str, default=str(DEFAULT_OUTPUT_DIR / "lc_manifest_all.parquet"),
                        help='Path to parquet or csv manifest of clean lightcurves (defaults to standard cluster manifest)')
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "microlensing_injection",
                        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR / 'microlensing_injection'})")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional tag to append to run directory name")
    parser.add_argument('--output', type=Path, default=None,
                        help='Override Parquet output path (default: <out-dir>/<timestamp>/microlensing_results.parquet)')
    parser.add_argument('--trials', type=int, default=1000,
                        help='Number of injection trials to run')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers for multiprocessing')
    parser.add_argument('--observed-events', type=int, default=None,
                        help='Explicit observed microlensing-event count; omit to skip rate calculation')
    parser.add_argument('--exposure-star-years', type=float, default=None,
                        help='Explicit survey exposure in star-years')
    parser.add_argument('--n-stars', type=float, default=None,
                        help='Explicit monitored-star count; requires --duration')
    parser.add_argument('--duration', type=float, default=None,
                        help='Explicit survey duration in years; requires --n-stars')
    parser.add_argument('--u0-min', type=float, default=0.01,
                        help='Minimum physical impact parameter u0')
    parser.add_argument('--u0-max', type=float, default=1.0,
                        help='Maximum physical impact parameter u0')
    parser.add_argument('--amp-min', type=float, default=None,
                        help='Legacy minimum Amax bound; requires --amp-max and overrides u0 bounds')
    parser.add_argument('--amp-max', type=float, default=None,
                        help='Legacy maximum Amax bound; requires --amp-min and overrides u0 bounds')
    parser.add_argument('--dur-min', type=float, default=1.0, help='Minimum injected Einstein time (tE in days)')
    parser.add_argument('--dur-max', type=float, default=500.0, help='Maximum injected Einstein time (tE in days)')
    parser.add_argument('--max-reduced-chi2', type=float, default=10.0,
                        help='Maximum Paczynski reduced chi-square for a recovery')
    parser.add_argument('--max-t0-offset-te', type=float, default=1.0,
                        help='Maximum fitted t0 offset, in injected tE units, for a recovery')
    parser.add_argument('--min-recovered-te-ratio', type=float, default=0.5,
                        help='Minimum fitted/injected tE ratio for a recovery')
    parser.add_argument('--max-recovered-te-ratio', type=float, default=2.0,
                        help='Maximum fitted/injected tE ratio for a recovery')
    parser.add_argument('--measure-pre-injection', dest='measure_pre_injection', action='store_true',
                        help='Run the paired no-injection fit (default)')
    parser.add_argument('--no-measure-pre-injection', dest='measure_pre_injection', action='store_false',
                        help='Disable the paired control for a diagnostic-only run')
    parser.set_defaults(measure_pre_injection=True)
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate the plot from an existing parquet file')
    parser.add_argument('--bins-te', type=int, default=15, help='Number of bins for tE in plot')
    parser.add_argument('--bins-amax', type=int, default=15, help='Number of bins for Amax in plot')
    parser.add_argument('--min-bin-trials', type=int, default=5,
                        help='Minimum designed trials required to display an efficiency bin')
    args = parser.parse_args()
    if (args.amp_min is None) != (args.amp_max is None):
        parser.error("--amp-min and --amp-max must be provided together")
    if args.plot_only and args.output is None:
        parser.error("--plot-only requires --output pointing to an existing results parquet")

    # Set up output paths with timestamped run directory
    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_tag}" if args.run_tag else timestamp
    
    if args.output:
        output_parquet_path = Path(args.output)
        run_dir = output_parquet_path.parent
    else:
        run_dir = base_out_dir / run_name
        output_parquet_path = run_dir / "microlensing_results.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_plot_path = output_parquet_path.with_suffix('.pdf')

    # Save run parameters to JSON
    run_params_file = run_dir / (
        "postprocess_params.json" if args.plot_only else "run_params.json"
    )
    run_params = vars(args).copy()
    for key, value in run_params.items():
        if isinstance(value, Path):
            run_params[key] = str(value)
    with open(run_params_file, "w") as f:
        json.dump(run_params, f, indent=2, default=str)

    # Create/update 'latest' symlink only if we created a new run_dir
    if not args.output:
        latest_link = base_out_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(run_name)
        except Exception as e:
            pass # Symlinks might fail on some filesystems

    if not args.plot_only:
        manifest_path = Path(args.manifest)
        if manifest_path.suffix == '.parquet':
            df_manifest = pd.read_parquet(manifest_path)
        else:
            df_manifest = pd.read_csv(manifest_path)

        legacy_Amax_range = (
            (args.amp_min, args.amp_max) if args.amp_min is not None else None
        )
        physical_u0_range = (
            None if legacy_Amax_range is not None else (args.u0_min, args.u0_max)
        )
        run_microlensing_injection_recovery(
            control_sample=df_manifest,
            total_trials=args.trials,
            workers=args.workers,
            u0_range=physical_u0_range,
            Amax_range=legacy_Amax_range,
            tE_range=(args.dur_min, args.dur_max),
            measure_pre_injection=args.measure_pre_injection,
            max_reduced_chi2=args.max_reduced_chi2,
            max_t0_offset_tE=args.max_t0_offset_te,
            recovered_tE_ratio_range=(
                args.min_recovered_te_ratio,
                args.max_recovered_te_ratio,
            ),
            output_path=output_parquet_path,
            overwrite=args.overwrite,
        )

    print(f"Generating efficiency map in {output_plot_path}...")
    df_results = pd.read_parquet(output_parquet_path)
    efficiency_summary = summarize_injection_efficiency(
        df_results, detected_col="recovered"
    )
    summary_path = output_parquet_path.with_name(
        f"{output_parquet_path.stem}_efficiency_summary.json"
    )
    summary_path.write_text(json.dumps(efficiency_summary, indent=2, sort_keys=True))
    print(f"Efficiency summary saved to {summary_path}")
    efficiency_grid = plot_efficiency_map(
        df_results,
        output_plot_path,
        bins_tE=args.bins_te,
        bins_Amax=args.bins_amax,
        min_bin_trials=args.min_bin_trials,
    )
    grid_path = output_parquet_path.with_name(
        f"{output_parquet_path.stem}_efficiency_grid.npz"
    )
    np.savez(
        grid_path,
        **{
            key: value
            for key, value in efficiency_grid.items()
            if isinstance(value, (np.ndarray, str, int, float, bool, np.number))
        },
    )
    print(f"Efficiency grid saved to {grid_path}")
    if args.observed_events is not None:
        rate_result = calculate_event_rate(
            df_results,
            n_observed_events=args.observed_events,
            exposure_star_years=args.exposure_star_years,
            n_stars_monitored=args.n_stars,
            duration_years=args.duration,
        )
        rate_path = output_parquet_path.with_name(
            f"{output_parquet_path.stem}_event_rate.json"
        )
        rate_path.write_text(json.dumps(rate_result, indent=2, sort_keys=True))
        print(f"Event-rate assumptions and result saved to {rate_path}")
    else:
        print("Event rate not calculated: pass --observed-events plus explicit survey exposure.")

if __name__ == '__main__':
    main()
