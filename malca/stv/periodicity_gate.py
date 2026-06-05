from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    PRE_PERIODICITY_CE_SNR_RESCUE_MARGIN,
    PRE_PERIODICITY_CE_SNR_THRESHOLD,
    PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_POINTS,
    PRE_PERIODICITY_MIN_PERIOD,
    PRE_PERIODICITY_N_PERIODS,
    PRE_PERIODICITY_PHASE_PEAK_SNR_MIN,
    PRE_PERIODICITY_PHASE_PEAK_REGION_MAX,
    PRE_PERIODICITY_PHASE_PEAK_WIDTH_MAX,
    PRE_PERIODICITY_SCATTER_RATIO_MAX,
    PRE_PERIODICITY_SCATTER_RATIO_RESCUE_MARGIN,
)
from malca.config import WORKERS
from malca.config import LS_ALIAS_PERIODS, LS_ALIAS_TOLERANCE
from malca.lightcurve_io import load_lightcurve_df, to_legacy_asassn_frame
from malca.phase import align_v_to_g_magnitude, phase_template, template_phase_lag
from malca.stats import compute_ce_stats
from malca.utils import clean_lc, compute_n_cameras


PREGATE_HARMONIC_FACTORS: tuple[float, ...] = (
    1.0,
    2.0,
    0.5,
    3.0,
    1.0 / 3.0,
    4.0,
    0.25,
    5.0,
    1.0 / 5.0,
    6.0,
    1.0 / 6.0,
    7.0,
    1.0 / 7.0,
    8.0,
    1.0 / 8.0,
)
PREGATE_HARMONIC_MIN_REL_IMPROVEMENT = 0.02
PREGATE_ROUTER_MODE = "ce_folded_scatter_phase_shape_v5"
PREGATE_CHECKPOINT_VERSION = "v8_ce_folded_scatter_phase_shape_lag"
PREGATE_RESULT_COLUMNS: list[str] = [
    "pre_periodicity_checkpoint_key",
    "pre_periodicity_path",
    "pre_n_points",
    "pre_n_cameras",
    "pre_ce_period",
    "pre_ce_corrected_period",
    "pre_ce_harmonic_factor",
    "pre_ce_entropy",
    "pre_ce_snr",
    "pre_ce_support",
    "pre_periodicity_router_mode",
    "pre_periodicity_v_minus_g_median_offset",
    "pre_periodicity_method",
    "pre_periodicity_base_period",
    "pre_periodicity_selected_period",
    "pre_periodicity_harmonic_factor",
    "pre_periodicity_selection_objective",
    "pre_periodicity_support_count",
    "pre_periodicity_score",
    "pre_periodicity_scatter_ratio",
    "pre_periodicity_phase_peak_snr",
    "pre_periodicity_phase_peak_width",
    "pre_periodicity_phase_peak_regions",
    "pre_periodicity_phase_peak_flag",
    "pre_periodicity_phase_lag_g_v_cycles",
    "pre_periodicity_phase_lag_g_v_abs_cycles",
    "pre_periodicity_alias_flag",
    "pre_periodicity_label",
    "pre_periodic_flag",
    "pre_periodicity_reason",
    "pre_periodicity_error",
]


def _empty_result(
    path: str,
    checkpoint_key: str,
    *,
    label: str = "non_periodic",
    reason: str = "",
    error: str | None = None,
) -> dict[str, object]:
    return {
        "pre_periodicity_checkpoint_key": checkpoint_key,
        "pre_periodicity_path": path,
        "pre_n_points": 0,
        "pre_n_cameras": 0,
        "pre_ce_period": np.nan,
        "pre_ce_corrected_period": np.nan,
        "pre_ce_harmonic_factor": np.nan,
        "pre_ce_entropy": np.nan,
        "pre_ce_snr": np.nan,
        "pre_ce_support": False,
        "pre_periodicity_router_mode": PREGATE_ROUTER_MODE,
        "pre_periodicity_v_minus_g_median_offset": np.nan,
        "pre_periodicity_method": None,
        "pre_periodicity_base_period": np.nan,
        "pre_periodicity_selected_period": np.nan,
        "pre_periodicity_harmonic_factor": np.nan,
        "pre_periodicity_selection_objective": np.nan,
        "pre_periodicity_support_count": 0,
        "pre_periodicity_score": np.nan,
        "pre_periodicity_scatter_ratio": np.nan,
        "pre_periodicity_phase_peak_snr": np.nan,
        "pre_periodicity_phase_peak_width": np.nan,
        "pre_periodicity_phase_peak_regions": np.nan,
        "pre_periodicity_phase_peak_flag": False,
        "pre_periodicity_phase_lag_g_v_cycles": np.nan,
        "pre_periodicity_phase_lag_g_v_abs_cycles": np.nan,
        "pre_periodicity_alias_flag": False,
        "pre_periodicity_label": label,
        "pre_periodic_flag": bool(label == "periodic"),
        "pre_periodicity_reason": reason,
        "pre_periodicity_error": error,
    }


def _excluded_camera_set(value: object) -> set[int]:
    if isinstance(value, str):
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    if isinstance(value, (set, list, tuple)):
        return {int(item) for item in value}
    return set()


def _checkpoint_key(path: str, excluded_cameras: object) -> str:
    token = ",".join(str(item) for item in sorted(_excluded_camera_set(excluded_cameras)))
    return f"{PREGATE_CHECKPOINT_VERSION}::{path}::{token}"


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _score_period_harmonic_candidate(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    period: float,
    *,
    n_bins: int = 48,
) -> dict[str, object]:
    if not np.isfinite(period) or period <= 0:
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "phase_peak_snr": np.nan,
            "phase_peak_width": np.nan,
            "phase_peak_regions": np.nan,
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }

    all_jd = [jd for jd, _ in band_resid.values() if jd.size > 0]
    if not all_jd:
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "phase_peak_snr": np.nan,
            "phase_peak_width": np.nan,
            "phase_peak_regions": np.nan,
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }
    jd0 = float(min(np.min(jd) for jd in all_jd))

    templates: dict[int, np.ndarray] = {}
    scatter_ratios: list[float] = []
    phase_peak_candidates: list[tuple[float, float, float]] = []
    for band, (jd, resid) in band_resid.items():
        phase = np.mod((jd - jd0) / float(period), 1.0)
        template, _ = phase_template(phase, resid, n_bins=n_bins)
        templates[int(band)] = template

        bin_idx = np.floor(phase * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        model = template[bin_idx]
        valid = np.isfinite(model) & np.isfinite(resid)
        if np.count_nonzero(valid) < 20:
            continue

        raw_sigma = _robust_sigma(resid[valid])
        folded_sigma = _robust_sigma(resid[valid] - model[valid])
        if np.isfinite(raw_sigma) and raw_sigma > 0 and np.isfinite(folded_sigma):
            scatter_ratios.append(float(folded_sigma / raw_sigma))

        finite_template = template[np.isfinite(template)]
        if finite_template.size < max(6, n_bins // 6):
            continue
        template_baseline = float(np.median(finite_template))
        template_peak = float(np.max(finite_template))
        peak_amp = float(template_peak - template_baseline)
        if not np.isfinite(peak_amp) or peak_amp <= 0 or not np.isfinite(raw_sigma) or raw_sigma <= 0:
            continue

        peak_snr = float(peak_amp / raw_sigma)
        half_peak_level = float(template_baseline + 0.5 * peak_amp)
        peak_mask = np.asarray(template >= half_peak_level, dtype=bool)
        if peak_mask.any():
            peak_regions = int(np.count_nonzero(peak_mask & ~np.roll(peak_mask, 1)))
        else:
            peak_regions = 0
        peak_width = float(np.mean(finite_template >= half_peak_level))
        if np.isfinite(peak_snr) and np.isfinite(peak_width):
            phase_peak_candidates.append((peak_snr, peak_width, float(peak_regions)))

    if not scatter_ratios:
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "phase_peak_snr": np.nan,
            "phase_peak_width": np.nan,
            "phase_peak_regions": np.nan,
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }

    scatter_ratio = float(np.mean(scatter_ratios))
    phase_peak_snr = np.nan
    phase_peak_width = np.nan
    phase_peak_regions = np.nan
    if phase_peak_candidates:
        phase_peak_snr, phase_peak_width, phase_peak_regions = max(
            phase_peak_candidates,
            key=lambda item: (float(item[0]), -float(item[1]), -float(item[2])),
        )
    phase_lag = np.nan
    if 0 in templates and 1 in templates:
        phase_lag = template_phase_lag(templates[0], templates[1], signed=True)
    phase_lag_abs = abs(float(phase_lag)) if np.isfinite(phase_lag) else np.nan
    alias_matches = [
        float(alias_period)
        for alias_period in LS_ALIAS_PERIODS
        if np.isfinite(alias_period) and abs(float(period) - float(alias_period)) <= float(LS_ALIAS_TOLERANCE)
    ]
    alias_flag = bool(alias_matches)
    return {
        "objective": scatter_ratio,
        "raw_objective": scatter_ratio,
        "scatter_ratio": scatter_ratio,
        "phase_peak_snr": float(phase_peak_snr),
        "phase_peak_width": float(phase_peak_width),
        "phase_peak_regions": float(phase_peak_regions),
        "phase_lag_g_v_cycles": float(phase_lag),
        "phase_lag_g_v_abs_cycles": float(phase_lag_abs),
        "alias_flag": alias_flag,
        "alias_matches": alias_matches,
    }


def _passes_phase_complexity(phase_peak_regions: object) -> bool:
    try:
        regions = float(phase_peak_regions)
    except (TypeError, ValueError):
        return True
    return (not np.isfinite(regions)) or regions <= float(PRE_PERIODICITY_PHASE_PEAK_REGION_MAX)


def _harmonic_candidate_sort_key(score: dict[str, object]) -> tuple[float, float, float, float]:
    scatter_ratio = float(score.get("scatter_ratio", np.nan))
    if not np.isfinite(scatter_ratio):
        scatter_ratio = np.inf

    phase_peak_regions = float(score.get("phase_peak_regions", np.nan))
    if not np.isfinite(phase_peak_regions):
        phase_peak_regions = np.inf

    phase_peak_snr = float(score.get("phase_peak_snr", np.nan))
    if not np.isfinite(phase_peak_snr):
        phase_peak_snr = -np.inf

    return (
        0.0 if _passes_phase_complexity(score.get("phase_peak_regions", np.nan)) else 1.0,
        float(scatter_ratio),
        float(phase_peak_regions),
        -float(phase_peak_snr),
    )


def _arbitrate_harmonic_period(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
    harmonic_factors: tuple[float, ...] = PREGATE_HARMONIC_FACTORS,
) -> tuple[float, float, dict[str, object]]:
    if not np.isfinite(base_period) or base_period <= 0 or not band_resid:
        return float(base_period), 1.0, {
            "objective": np.nan,
            "scatter_ratio": np.nan,
            "phase_peak_snr": np.nan,
            "phase_peak_width": np.nan,
            "phase_peak_regions": np.nan,
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
        }

    candidates: list[tuple[float, float, dict[str, object]]] = []
    for factor in harmonic_factors:
        period = float(base_period) * float(factor)
        if not np.isfinite(period) or period <= 0 or period < float(min_period) or period > float(max_period):
            continue
        if any(abs(period - prev_period) <= 1e-10 * max(1.0, abs(period), abs(prev_period)) for _, prev_period, _ in candidates):
            continue
        score = dict(_score_period_harmonic_candidate(band_resid, period))
        score["selection_objective"] = float(score.get("objective", np.inf))
        candidates.append((float(factor), period, score))

    if not candidates:
        return float(base_period), 1.0, {
            "objective": np.nan,
            "scatter_ratio": np.nan,
            "phase_peak_snr": np.nan,
            "phase_peak_width": np.nan,
            "phase_peak_regions": np.nan,
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
        }

    factor, period, score = min(candidates, key=lambda item: _harmonic_candidate_sort_key(item[2]))
    base_entry = next((item for item in candidates if abs(float(item[0]) - 1.0) < 1e-12), None)
    if base_entry is not None and abs(float(factor) - 1.0) > 1e-12:
        base_passes_complexity = _passes_phase_complexity(base_entry[2].get("phase_peak_regions", np.nan))
        best_passes_complexity = _passes_phase_complexity(score.get("phase_peak_regions", np.nan))
        if base_passes_complexity != best_passes_complexity:
            return float(period), float(factor), score
        base_selection_objective = float(
            base_entry[2].get("selection_objective", base_entry[2].get("objective", np.nan))
        )
        best_selection_objective = float(score.get("selection_objective", score.get("objective", np.nan)))
        improvement = (base_selection_objective - best_selection_objective) / max(abs(base_selection_objective), 1e-9)
        if not np.isfinite(improvement) or improvement < float(PREGATE_HARMONIC_MIN_REL_IMPROVEMENT):
            factor, period, score = base_entry
    return float(period), float(factor), score


def _build_band_residuals(df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if df.empty or "v_g_band" not in df.columns:
        return band_resid

    for band_value, band_df in df.groupby("v_g_band"):
        try:
            band = int(band_value)
        except Exception:
            continue
        jd = band_df["JD"].to_numpy(dtype=float)
        mag = band_df["mag"].to_numpy(dtype=float)
        valid = np.isfinite(jd) & np.isfinite(mag)
        if np.count_nonzero(valid) < 20:
            continue
        mag_valid = mag[valid]
        resid = mag_valid - float(np.median(mag_valid))
        band_resid[band] = (jd[valid], resid)
    return band_resid


def _harmonically_correct_period_candidate(
    method_name: str,
    raw_period: float,
    *,
    snr: float,
    supported: bool,
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    min_period: float,
    max_period: float,
) -> dict[str, object]:
    if not np.isfinite(raw_period) or raw_period <= 0:
        return {
            "method": method_name,
            "raw_period": np.nan,
            "corrected_period": np.nan,
            "harmonic_factor": np.nan,
            "snr": float(snr),
            "supported": bool(supported),
            "objective": np.nan,
            "selection_objective": np.nan,
            "scatter_ratio": np.nan,
            "phase_peak_snr": np.nan,
            "phase_peak_width": np.nan,
            "phase_peak_regions": np.nan,
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "alias_flag": False,
        }

    corrected_period, harmonic_factor, diag = _arbitrate_harmonic_period(
        band_resid,
        float(raw_period),
        min_period=float(min_period),
        max_period=float(max_period),
    )
    objective = float(diag.get("objective", np.nan))
    selection_objective = float(diag.get("selection_objective", np.nan))
    return {
        "method": method_name,
        "raw_period": float(raw_period),
        "corrected_period": float(corrected_period),
        "harmonic_factor": float(harmonic_factor),
        "snr": float(snr),
        "supported": bool(supported),
        "objective": objective,
        "selection_objective": selection_objective,
        "scatter_ratio": float(diag.get("scatter_ratio", np.nan)),
        "phase_peak_snr": float(diag.get("phase_peak_snr", np.nan)),
        "phase_peak_width": float(diag.get("phase_peak_width", np.nan)),
        "phase_peak_regions": float(diag.get("phase_peak_regions", np.nan)),
        "phase_lag_g_v_cycles": float(diag.get("phase_lag_g_v_cycles", np.nan)),
        "phase_lag_g_v_abs_cycles": float(diag.get("phase_lag_g_v_abs_cycles", np.nan)),
        "alias_flag": bool(diag.get("alias_flag", False)),
    }


def _period_candidate_sort_key(candidate: dict[str, object]) -> tuple[float, float]:
    selection_objective = float(candidate.get("selection_objective", np.nan))
    objective = float(candidate.get("objective", np.nan))
    rank_objective = selection_objective
    if not np.isfinite(rank_objective):
        rank_objective = objective
    if not np.isfinite(rank_objective):
        rank_objective = np.inf

    snr = float(candidate.get("snr", np.nan))
    if not np.isfinite(snr):
        snr = -np.inf

    return (float(rank_objective), -float(snr))


def _select_period_candidate(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    usable = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.get("corrected_period", np.nan))
        and float(candidate.get("corrected_period", np.nan)) > 0
    ]
    if not usable:
        return None

    supported_candidates = [candidate for candidate in usable if bool(candidate.get("supported", False))]
    pool = supported_candidates or usable
    return min(pool, key=_period_candidate_sort_key)


def _evaluate_periodicity_worker(args: tuple[object, ...]) -> dict[str, object]:
    (
        path_str,
        checkpoint_key,
        excluded_cameras,
        bad_camera_scatter_ratio,
        clean_max_error_absolute,
        clean_max_error_sigma,
        min_period,
        max_period,
        n_periods,
        ce_snr_threshold,
        min_points,
        scatter_ratio_max,
    ) = args

    try:
        df_lc = load_lightcurve_df(
            path_str,
            filter_bad_cameras_enabled=True,
            bad_camera_scatter_ratio=float(bad_camera_scatter_ratio),
        )
        df_lc = to_legacy_asassn_frame(df_lc)
        if df_lc.empty:
            return _empty_result(path_str, checkpoint_key, reason="empty_lc")

        excluded = _excluded_camera_set(excluded_cameras)
        if excluded and "camera#" in df_lc.columns:
            df_lc = df_lc[~df_lc["camera#"].isin(excluded)].reset_index(drop=True)

        df_lc = clean_lc(
            df_lc,
            max_error_absolute=float(clean_max_error_absolute),
            max_error_sigma=float(clean_max_error_sigma),
        )
        if df_lc.empty:
            return _empty_result(path_str, checkpoint_key, reason="empty_after_clean")

        n_points = int(len(df_lc))
        n_cameras = int(compute_n_cameras(df_lc))
        if n_points < int(min_points):
            result = _empty_result(path_str, checkpoint_key, reason="too_few_points")
            result["pre_n_points"] = n_points
            result["pre_n_cameras"] = n_cameras
            return result

        df_lc_aligned, v_minus_g_median_offset = align_v_to_g_magnitude(df_lc)

        jd = df_lc_aligned["JD"].to_numpy(dtype=float)
        mag = df_lc_aligned["mag"].to_numpy(dtype=float)
        err = df_lc_aligned["error"].to_numpy(dtype=float)

        ce_result = compute_ce_stats(
            jd,
            mag,
            err,
            min_period=float(min_period),
            max_period=float(max_period),
            n_periods=int(n_periods),
            n_bootstrap=0,
            refine=True,
        )

        ce_support = bool(
            np.isfinite(ce_result.get("ce_snr", np.nan))
            and float(ce_result["ce_snr"]) >= float(ce_snr_threshold)
        )
        support_count = int(ce_support)

        band_resid = _build_band_residuals(df_lc_aligned)
        ce_candidate = _harmonically_correct_period_candidate(
            "ce",
            float(ce_result.get("ce_period", np.nan)),
            snr=float(ce_result.get("ce_snr", np.nan)),
            supported=ce_support,
            band_resid=band_resid,
            min_period=float(min_period),
            max_period=float(max_period),
        )

        selected_candidate = _select_period_candidate([ce_candidate])
        method_name = str(selected_candidate.get("method")) if selected_candidate is not None else None
        base_period = float(selected_candidate.get("raw_period", np.nan)) if selected_candidate is not None else np.nan
        selected_period = float(selected_candidate.get("corrected_period", np.nan)) if selected_candidate is not None else np.nan
        harmonic_factor = float(selected_candidate.get("harmonic_factor", np.nan)) if selected_candidate is not None else np.nan
        selection_objective = (
            float(selected_candidate.get("selection_objective", np.nan))
            if selected_candidate is not None
            else np.nan
        )
        selected_snr = float(selected_candidate.get("snr", np.nan)) if selected_candidate is not None else np.nan

        scatter_ratio = float(selected_candidate.get("scatter_ratio", np.nan)) if selected_candidate is not None else np.nan
        phase_peak_snr = float(selected_candidate.get("phase_peak_snr", np.nan)) if selected_candidate is not None else np.nan
        phase_peak_width = (
            float(selected_candidate.get("phase_peak_width", np.nan))
            if selected_candidate is not None
            else np.nan
        )
        phase_peak_regions = (
            float(selected_candidate.get("phase_peak_regions", np.nan))
            if selected_candidate is not None
            else np.nan
        )
        phase_lag = (
            float(selected_candidate.get("phase_lag_g_v_cycles", np.nan))
            if selected_candidate is not None
            else np.nan
        )
        phase_lag_abs = (
            float(selected_candidate.get("phase_lag_g_v_abs_cycles", np.nan))
            if selected_candidate is not None
            else np.nan
        )
        alias_flag = bool(selected_candidate.get("alias_flag", False)) if selected_candidate is not None else False
        scatter_ok = bool(np.isfinite(scatter_ratio) and scatter_ratio <= float(scatter_ratio_max))
        phase_peak_ok = bool(
            np.isfinite(phase_peak_snr)
            and phase_peak_snr >= float(PRE_PERIODICITY_PHASE_PEAK_SNR_MIN)
            and np.isfinite(phase_peak_width)
            and phase_peak_width <= float(PRE_PERIODICITY_PHASE_PEAK_WIDTH_MAX)
        )
        phase_complexity_ok = _passes_phase_complexity(phase_peak_regions)
        scatter_rescue_ok = bool(
            np.isfinite(scatter_ratio)
            and scatter_ratio <= (float(scatter_ratio_max) + float(PRE_PERIODICITY_SCATTER_RATIO_RESCUE_MARGIN))
        )
        ce_near_support = bool(
            np.isfinite(selected_snr)
            and selected_snr >= (float(ce_snr_threshold) - float(PRE_PERIODICITY_CE_SNR_RESCUE_MARGIN))
        )

        periodic_by_base = bool(ce_support and scatter_ok and phase_complexity_ok)
        periodic_by_scatter_rescue = bool(
            ce_support
            and (not scatter_ok)
            and phase_peak_ok
            and phase_complexity_ok
            and scatter_rescue_ok
        )
        periodic_by_ce_rescue = bool(
            (not ce_support)
            and ce_near_support
            and phase_peak_ok
            and phase_complexity_ok
            and scatter_rescue_ok
        )
        label = "periodic" if (periodic_by_base or periodic_by_scatter_rescue or periodic_by_ce_rescue) else "non_periodic"

        score = float(selected_snr) if np.isfinite(selected_snr) else np.nan

        if periodic_by_base:
            reasons = ["ce", "folded_scatter"]
        elif periodic_by_scatter_rescue:
            reasons = ["ce", "phase_peak", "scatter_rescue"]
        elif periodic_by_ce_rescue:
            reasons = ["ce_near_threshold"]
            if scatter_ok:
                reasons.append("folded_scatter")
                reasons.append("phase_peak")
            else:
                reasons.extend(["phase_peak", "scatter_rescue"])
        else:
            reasons = []
            if ce_support:
                reasons.append("ce")
            elif ce_near_support and phase_peak_ok:
                reasons.append("ce_near_threshold")
            else:
                reasons.append("ce_below_threshold")
            if scatter_ok:
                reasons.append("folded_scatter")
            elif np.isfinite(scatter_ratio):
                reasons.append("scatter_too_high")
            else:
                reasons.append("no_folded_scatter")
            if phase_peak_ok:
                reasons.append("phase_peak")
            if np.isfinite(phase_peak_regions) and not phase_complexity_ok:
                reasons.append("phase_complexity")
            if alias_flag:
                reasons.append("alias")
        reason = ",".join(reasons) if reasons else "no_strong_periodicity"

        return {
            "pre_periodicity_checkpoint_key": checkpoint_key,
            "pre_periodicity_path": path_str,
            "pre_n_points": n_points,
            "pre_n_cameras": n_cameras,
            "pre_ce_period": float(ce_result.get("ce_period", np.nan)),
            "pre_ce_corrected_period": float(ce_candidate.get("corrected_period", np.nan)),
            "pre_ce_harmonic_factor": float(ce_candidate.get("harmonic_factor", np.nan)),
            "pre_ce_entropy": float(ce_result.get("ce_min_entropy", np.nan)),
            "pre_ce_snr": float(ce_result.get("ce_snr", np.nan)),
            "pre_ce_support": ce_support,
            "pre_periodicity_router_mode": PREGATE_ROUTER_MODE,
            "pre_periodicity_v_minus_g_median_offset": float(v_minus_g_median_offset),
            "pre_periodicity_method": method_name,
            "pre_periodicity_base_period": float(base_period),
            "pre_periodicity_selected_period": float(selected_period),
            "pre_periodicity_harmonic_factor": float(harmonic_factor),
            "pre_periodicity_selection_objective": float(selection_objective),
            "pre_periodicity_support_count": support_count,
            "pre_periodicity_score": score,
            "pre_periodicity_scatter_ratio": scatter_ratio,
            "pre_periodicity_phase_peak_snr": phase_peak_snr,
            "pre_periodicity_phase_peak_width": phase_peak_width,
            "pre_periodicity_phase_peak_regions": phase_peak_regions,
            "pre_periodicity_phase_peak_flag": phase_peak_ok,
            "pre_periodicity_phase_lag_g_v_cycles": phase_lag,
            "pre_periodicity_phase_lag_g_v_abs_cycles": phase_lag_abs,
            "pre_periodicity_alias_flag": alias_flag,
            "pre_periodicity_label": label,
            "pre_periodic_flag": bool(label == "periodic"),
            "pre_periodicity_reason": reason,
            "pre_periodicity_error": None,
        }
    except Exception as exc:
        return _empty_result(path_str, checkpoint_key, reason="error", error=str(exc))


def _result_is_usable(result: dict[str, object] | None) -> bool:
    if not isinstance(result, dict) or not result:
        return False
    if str(result.get("pre_periodicity_error") or "").strip():
        return False
    if str(result.get("pre_periodicity_router_mode") or "").strip() != PREGATE_ROUTER_MODE:
        return False
    required_columns = {
        "pre_periodicity_label",
        "pre_ce_corrected_period",
        "pre_periodicity_selection_objective",
    }
    return required_columns.issubset(result)


def _save_checkpoint(checkpoint_path: Path, results: dict[str, dict[str, object]]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(results.values()))
    if df.empty:
        return
    df = df.drop_duplicates(subset=["pre_periodicity_checkpoint_key"], keep="last")
    df.to_parquet(checkpoint_path, index=False)


def apply_pre_periodicity_gate(
    df: pd.DataFrame,
    *,
    path_col: str | None = None,
    excluded_cameras_col: str | None = "excluded_cameras",
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
    min_period: float = PRE_PERIODICITY_MIN_PERIOD,
    max_period: float = PRE_PERIODICITY_MAX_PERIOD,
    n_periods: int = PRE_PERIODICITY_N_PERIODS,
    ce_snr_threshold: float = PRE_PERIODICITY_CE_SNR_THRESHOLD,
    min_points: int = PRE_PERIODICITY_MIN_POINTS,
    scatter_ratio_max: float = PRE_PERIODICITY_SCATTER_RATIO_MAX,
    workers: int = WORKERS,
    checkpoint_path: str | Path | None = None,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    df_out = df.copy()
    existing_gate_cols = [column for column in PREGATE_RESULT_COLUMNS if column in df_out.columns]
    if existing_gate_cols:
        df_out = df_out.drop(columns=existing_gate_cols)
    if df_out.empty:
        for column in PREGATE_RESULT_COLUMNS:
            if column not in df_out.columns:
                df_out[column] = pd.Series(dtype="object")
        return df_out

    if path_col is None:
        if "dat_path" in df_out.columns:
            path_col = "dat_path"
        elif "path" in df_out.columns:
            path_col = "path"
        else:
            raise ValueError("Need 'dat_path' or 'path' column for pre-periodicity gate")

    df_out[path_col] = df_out[path_col].astype(str)
    checkpoint_keys = [
        _checkpoint_key(
            path_value,
            row.get(excluded_cameras_col) if excluded_cameras_col else None,
        )
        for path_value, (_, row) in zip(df_out[path_col].tolist(), df_out.iterrows())
    ]
    df_out["_pre_periodicity_checkpoint_key"] = checkpoint_keys

    cached_results: dict[str, dict[str, object]] = {}
    checkpoint_file = Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_file)
            for _, row in checkpoint_df.iterrows():
                result = row.to_dict()
                key = str(result.get("pre_periodicity_checkpoint_key") or "").strip()
                if key:
                    cached_results[key] = result
        except Exception:
            cached_results = {}

    worker_args: list[tuple[object, ...]] = []
    result_map: dict[str, dict[str, object]] = {}
    for _, row in df_out.iterrows():
        path_value = str(row[path_col])
        checkpoint_key = str(row["_pre_periodicity_checkpoint_key"])
        cached = cached_results.get(checkpoint_key)
        if _result_is_usable(cached):
            result_map[checkpoint_key] = cached
            continue
        worker_args.append(
            (
                path_value,
                checkpoint_key,
                row.get(excluded_cameras_col) if excluded_cameras_col else None,
                float(bad_camera_scatter_ratio),
                float(clean_max_error_absolute),
                float(clean_max_error_sigma),
                float(min_period),
                float(max_period),
                int(n_periods),
                float(ce_snr_threshold),
                int(min_points),
                float(scatter_ratio_max),
            )
        )

    if show_tqdm:
        tqdm.write(
            f"[pre_periodicity_gate] {len(result_map)} cached, processing {len(worker_args)} light curves"
        )

    checkpoint_interval = max(25, len(worker_args) // 20) if worker_args else 25
    processed_since_save = 0

    if workers > 1 and len(worker_args) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(worker_args))) as executor:
            futures = {
                executor.submit(_evaluate_periodicity_worker, item): item[1]
                for item in worker_args
            }
            iterator = as_completed(futures)
            if show_tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Pre-periodicity gate")
            for future in iterator:
                result = future.result()
                result_map[str(result["pre_periodicity_checkpoint_key"])] = result
                processed_since_save += 1
                if checkpoint_file is not None and processed_since_save >= checkpoint_interval:
                    _save_checkpoint(checkpoint_file, result_map)
                    processed_since_save = 0
    else:
        iterator = worker_args
        if show_tqdm and worker_args:
            iterator = tqdm(worker_args, desc="Pre-periodicity gate")
        for item in iterator:
            result = _evaluate_periodicity_worker(item)
            result_map[str(result["pre_periodicity_checkpoint_key"])] = result
            processed_since_save += 1
            if checkpoint_file is not None and processed_since_save >= checkpoint_interval:
                _save_checkpoint(checkpoint_file, result_map)
                processed_since_save = 0

    if checkpoint_file is not None and result_map:
        _save_checkpoint(checkpoint_file, result_map)

    results_df = pd.DataFrame(list(result_map.values()))
    if results_df.empty:
        for column in PREGATE_RESULT_COLUMNS:
            if column not in df_out.columns:
                df_out[column] = pd.Series(dtype="object")
        return df_out.drop(columns=["_pre_periodicity_checkpoint_key"])

    results_df = results_df.drop_duplicates(subset=["pre_periodicity_checkpoint_key"], keep="last")
    df_out = df_out.merge(
        results_df,
        left_on="_pre_periodicity_checkpoint_key",
        right_on="pre_periodicity_checkpoint_key",
        how="left",
    )
    df_out = df_out.drop(columns=["_pre_periodicity_checkpoint_key"])
    return df_out.reset_index(drop=True)
