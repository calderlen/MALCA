from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config.config_filters import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    PERIODIC_EVENTS_DEPTH_SNR_THRESHOLD,
    PERIODIC_EVENTS_MAX_DIP_WIDTH_PHASE,
    PERIODIC_EVENTS_MIN_CAMERA_SUPPORT,
    PERIODIC_EVENTS_MIN_CYCLE_SUPPORT,
    PERIODIC_EVENTS_MIN_POINTS,
    PERIODIC_EVENTS_MIN_SUPPORT_POINTS,
    PERIODIC_EVENTS_PHASE_BINS,
    PERIODIC_EVENTS_PROFILE_SMOOTH_WINDOW,
)
from malca.config.config_pipeline import WORKERS
from malca.lightcurve_io import load_lightcurve_df
from malca.stats import median_dt, robust_sigma
from malca.utils import clean_lc


PERIODIC_EVENT_RESULT_COLUMNS: list[str] = [
    "periodic_events_checkpoint_key",
    "periodic_events_path",
    "analysis_branch",
    "event_model",
    "event_type",
    "phase_period_days",
    "phase_reference_jd",
    "phase_dip_significant",
    "phase_dip_depth_mag",
    "phase_dip_depth_snr",
    "phase_dip_phase_center",
    "phase_dip_phase_start",
    "phase_dip_phase_end",
    "phase_dip_width_phase",
    "phase_dip_width_days",
    "phase_dip_support_points",
    "phase_dip_support_cycles",
    "phase_dip_support_cameras",
    "phase_secondary_depth_mag",
    "phase_secondary_depth_ratio",
    "phase_odd_depth_mag",
    "phase_even_depth_mag",
    "phase_odd_even_depth_diff_mag",
    "phase_odd_even_depth_ratio",
    "phase_profile_baseline_resid",
    "phase_profile_scatter",
    "phase_profile_bins",
    "phase_profile_finite_bins",
    "phase_profile_reason",
    "phase_profile_error",
    "path",
    "dip_significant",
    "jump_significant",
    "n_points",
    "jd_first",
    "jd_last",
    "cadence_median_days",
    "dip_best_morph",
    "dip_best_delta_bic",
    "dip_best_width_param",
    "dip_symmetry_score",
    "dip_best_amp",
    "dip_best_t0",
    "dip_best_alpha",
    "dip_best_tau",
    "jump_best_morph",
    "jump_best_delta_bic",
    "jump_best_width_param",
    "jump_best_amp",
    "jump_best_t0",
    "jump_best_alpha",
    "jump_best_tau",
    "dip_count",
    "jump_count",
    "dip_run_count",
    "jump_run_count",
    "dip_max_run_points",
    "jump_max_run_points",
    "dip_max_run_duration",
    "jump_max_run_duration",
    "dip_max_run_sum",
    "jump_max_run_sum",
    "dip_max_run_max",
    "jump_max_run_max",
    "dip_max_run_cameras",
    "jump_max_run_cameras",
    "dip_max_log_bf_local",
    "jump_max_log_bf_local",
    "dip_bayes_factor",
    "jump_bayes_factor",
    "baseline_mag",
    "dip_best_p",
    "jump_best_p",
    "dip_best_mag_event",
    "jump_best_mag_event",
    "dip_trigger_max",
    "jump_trigger_max",
    "dip_max_event_prob",
    "jump_max_event_prob",
    "n_cameras",
    "camera_ids",
    "camera_min_points",
    "camera_max_points",
    "dipper_score",
    "dipper_n_dips",
    "dipper_n_valid_dips",
    "jumper_score",
    "jumper_n_jumps",
    "jumper_n_valid_jumps",
    "baseline_source",
    "trigger_mode",
    "dip_trigger_threshold",
    "jump_trigger_threshold",
    "bad_cameras_filtered",
    "dip_is_single_event",
    "dip_inter_event_spacing_median",
    "dip_inter_event_spacing_std",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
    "jump_is_single_event",
    "jump_inter_event_spacing_median",
    "jump_inter_event_spacing_std",
    "jump_amplitude_consistency",
    "jump_duration_consistency",
]


def _excluded_camera_set(value: object) -> set[int]:
    if isinstance(value, str):
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    if isinstance(value, (set, list, tuple)):
        return {int(item) for item in value}
    return set()


def _checkpoint_key(path: str, period: float, excluded_cameras: object) -> str:
    token = ",".join(str(item) for item in sorted(_excluded_camera_set(excluded_cameras)))
    if np.isfinite(period):
        period_token = f"{float(period):.12g}"
    else:
        period_token = "nan"
    return f"{path}::{period_token}::{token}"


def _empty_result(
    path: str,
    checkpoint_key: str,
    *,
    period: float = np.nan,
    reason: str = "",
    error: str | None = None,
) -> dict[str, object]:
    return {
        "periodic_events_checkpoint_key": checkpoint_key,
        "periodic_events_path": path,
        "analysis_branch": "periodic",
        "event_model": "phase_folded_dip",
        "event_type": "none",
        "phase_period_days": float(period) if np.isfinite(period) else np.nan,
        "phase_reference_jd": np.nan,
        "phase_dip_significant": False,
        "phase_dip_depth_mag": np.nan,
        "phase_dip_depth_snr": np.nan,
        "phase_dip_phase_center": np.nan,
        "phase_dip_phase_start": np.nan,
        "phase_dip_phase_end": np.nan,
        "phase_dip_width_phase": np.nan,
        "phase_dip_width_days": np.nan,
        "phase_dip_support_points": 0,
        "phase_dip_support_cycles": 0,
        "phase_dip_support_cameras": 0,
        "phase_secondary_depth_mag": np.nan,
        "phase_secondary_depth_ratio": np.nan,
        "phase_odd_depth_mag": np.nan,
        "phase_even_depth_mag": np.nan,
        "phase_odd_even_depth_diff_mag": np.nan,
        "phase_odd_even_depth_ratio": np.nan,
        "phase_profile_baseline_resid": np.nan,
        "phase_profile_scatter": np.nan,
        "phase_profile_bins": 0,
        "phase_profile_finite_bins": 0,
        "phase_profile_reason": reason,
        "phase_profile_error": error,
        "path": path,
        "dip_significant": False,
        "jump_significant": False,
        "n_points": 0,
        "jd_first": np.nan,
        "jd_last": np.nan,
        "cadence_median_days": np.nan,
        "dip_best_morph": "phase_folded_dip",
        "dip_best_delta_bic": np.nan,
        "dip_best_width_param": np.nan,
        "dip_symmetry_score": np.nan,
        "dip_best_amp": np.nan,
        "dip_best_t0": np.nan,
        "dip_best_alpha": np.nan,
        "dip_best_tau": np.nan,
        "jump_best_morph": "none",
        "jump_best_delta_bic": 0.0,
        "jump_best_width_param": np.nan,
        "jump_best_amp": np.nan,
        "jump_best_t0": np.nan,
        "jump_best_alpha": np.nan,
        "jump_best_tau": np.nan,
        "dip_count": 0,
        "jump_count": 0,
        "dip_run_count": 0,
        "jump_run_count": 0,
        "dip_max_run_points": 0,
        "jump_max_run_points": 0,
        "dip_max_run_duration": np.nan,
        "jump_max_run_duration": np.nan,
        "dip_max_run_sum": np.nan,
        "jump_max_run_sum": np.nan,
        "dip_max_run_max": np.nan,
        "jump_max_run_max": np.nan,
        "dip_max_run_cameras": 0,
        "jump_max_run_cameras": 0,
        "dip_max_log_bf_local": np.nan,
        "jump_max_log_bf_local": np.nan,
        "dip_bayes_factor": 0.0,
        "jump_bayes_factor": 0.0,
        "baseline_mag": np.nan,
        "dip_best_p": np.nan,
        "jump_best_p": np.nan,
        "dip_best_mag_event": np.nan,
        "jump_best_mag_event": np.nan,
        "dip_trigger_max": np.nan,
        "jump_trigger_max": np.nan,
        "dip_max_event_prob": np.nan,
        "jump_max_event_prob": np.nan,
        "n_cameras": 0,
        "camera_ids": "",
        "camera_min_points": 0,
        "camera_max_points": 0,
        "dipper_score": np.nan,
        "dipper_n_dips": 0,
        "dipper_n_valid_dips": 0,
        "jumper_score": 0.0,
        "jumper_n_jumps": 0,
        "jumper_n_valid_jumps": 0,
        "baseline_source": "phase_folded_profile",
        "trigger_mode": "phase_folded_dip",
        "dip_trigger_threshold": float(PERIODIC_EVENTS_DEPTH_SNR_THRESHOLD),
        "jump_trigger_threshold": np.nan,
        "bad_cameras_filtered": "",
        "dip_is_single_event": False,
        "dip_inter_event_spacing_median": np.nan,
        "dip_inter_event_spacing_std": np.nan,
        "dip_amplitude_consistency": np.nan,
        "dip_duration_consistency": np.nan,
        "jump_is_single_event": False,
        "jump_inter_event_spacing_median": np.nan,
        "jump_inter_event_spacing_std": np.nan,
        "jump_amplitude_consistency": np.nan,
        "jump_duration_consistency": np.nan,
    }


def _result_is_usable(result: dict[str, object]) -> bool:
    if not isinstance(result, dict) or not result:
        return False
    if str(result.get("phase_profile_error") or "").strip():
        return False
    return "phase_dip_significant" in result


def _load_checkpoint(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or (not path.exists()):
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if "periodic_events_checkpoint_key" not in df.columns:
        return {}
    df = df.drop_duplicates(subset=["periodic_events_checkpoint_key"], keep="last")
    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        payload = row.to_dict()
        if _result_is_usable(payload):
            out[str(row["periodic_events_checkpoint_key"])] = payload
    return out


def _write_checkpoint(path: Path | None, results: dict[str, dict[str, object]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results.values())
    if df.empty:
        return
    df = df.drop_duplicates(subset=["periodic_events_checkpoint_key"], keep="last")
    df.to_parquet(path, index=False)


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0 or window <= 1:
        return arr.copy()
    half = max(0, int(window) // 2)
    out = np.full(n, np.nan, dtype=float)
    for idx in range(n):
        local = [arr[(idx + delta) % n] for delta in range(-half, half + 1)]
        local_arr = np.asarray(local, dtype=float)
        finite = local_arr[np.isfinite(local_arr)]
        if finite.size > 0:
            out[idx] = float(np.median(finite))
    return out


def _phase_profile(
    phase: np.ndarray,
    resid: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = (np.arange(n_bins, dtype=float) + 0.5) / float(n_bins)
    profile = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    valid = np.isfinite(phase) & np.isfinite(resid)
    if np.count_nonzero(valid) == 0:
        return centers, profile, counts
    phase_valid = np.mod(phase[valid], 1.0)
    resid_valid = resid[valid]
    indices = np.floor(phase_valid * n_bins).astype(int)
    indices = np.clip(indices, 0, n_bins - 1)
    for bin_idx in range(n_bins):
        vals = resid_valid[indices == bin_idx]
        if vals.size > 0:
            profile[bin_idx] = float(np.median(vals))
            counts[bin_idx] = int(vals.size)
    return centers, profile, counts


def _circular_segment(mask: np.ndarray, anchor: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    if n == 0:
        return np.array([], dtype=int)
    mask = mask.copy()
    anchor = int(anchor) % n
    mask[anchor] = True
    doubled = np.concatenate([mask, mask])
    anchor_idx = anchor + n

    start = anchor_idx
    while start > 0 and doubled[start - 1]:
        start -= 1

    end = anchor_idx
    while (end + 1) < doubled.size and doubled[end + 1]:
        end += 1

    length = min(n, end - start + 1)
    return np.asarray([(start + offset) % n for offset in range(length)], dtype=int)


def _phase_interval_mask(phase: np.ndarray, start: float, end: float) -> np.ndarray:
    phase = np.mod(np.asarray(phase, dtype=float), 1.0)
    start = float(np.mod(start, 1.0))
    end = float(np.mod(end, 1.0))
    if np.isclose(start, end):
        return np.ones_like(phase, dtype=bool)
    if start < end:
        return (phase >= start) & (phase < end)
    return (phase >= start) | (phase < end)


def _circular_center(phases: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.mod(np.asarray(phases, dtype=float), 1.0)
    finite = np.isfinite(values)
    if weights is None:
        weights_arr = np.ones_like(values, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        finite &= np.isfinite(weights_arr)
    if np.count_nonzero(finite) == 0:
        return np.nan
    values = values[finite]
    weights_arr = weights_arr[finite]
    angles = 2.0 * np.pi * values
    comp = np.sum(weights_arr * np.exp(1j * angles))
    if not np.isfinite(comp.real) or not np.isfinite(comp.imag) or abs(comp) == 0:
        return float(np.mod(np.median(values), 1.0))
    return float(np.mod(np.angle(comp) / (2.0 * np.pi), 1.0))


def _phase_offset(phase: np.ndarray, center: float) -> np.ndarray:
    return ((np.asarray(phase, dtype=float) - float(center) + 0.5) % 1.0) - 0.5


def _phase_window_symmetry(phases: np.ndarray, values: np.ndarray, center: float) -> float:
    offsets = _phase_offset(phases, center)
    finite = np.isfinite(offsets) & np.isfinite(values)
    if np.count_nonzero(finite) < 4:
        return np.nan
    offsets = offsets[finite]
    values = np.asarray(values, dtype=float)[finite]
    left = values[offsets < 0]
    right = values[offsets >= 0]
    if left.size == 0 or right.size == 0:
        return np.nan
    denom = max(abs(np.nanmedian(left)) + abs(np.nanmedian(right)), 1e-6)
    return float((np.nanmedian(right) - np.nanmedian(left)) / denom)


def _log10_score(depth: float, depth_snr: float, width_days: float, support_cycles: int, n_points: int) -> float:
    signal = max(float(depth), 1e-6)
    quality = max(float(depth_snr), 1.0)
    duration = max(float(width_days), 1e-4)
    support = max(int(support_cycles), 1)
    norm = max(int(n_points), 1)
    return float(np.log10(signal * quality * duration * support / norm))


def _evaluate_periodic_candidate_worker(args: tuple[object, ...]) -> dict[str, object]:
    (
        path_str,
        checkpoint_key,
        period,
        excluded_cameras,
        bad_camera_scatter_ratio,
        clean_max_error_absolute,
        clean_max_error_sigma,
        phase_bins,
        profile_smooth_window,
        min_points,
        min_support_points,
        min_cycle_support,
        min_camera_support,
        min_depth_snr,
        max_dip_width_phase,
    ) = args

    result = _empty_result(path_str, checkpoint_key, period=period)

    try:
        if not np.isfinite(period) or float(period) <= 0:
            result["phase_profile_reason"] = "missing_period"
            return result

        excluded = _excluded_camera_set(excluded_cameras)
        df_lc, bad_cameras = load_lightcurve_df(
            path_str,
            filter_bad_cameras_enabled=True,
            bad_camera_scatter_ratio=float(bad_camera_scatter_ratio),
            return_filtered_info=True,
        )
        if df_lc.empty:
            result["phase_profile_reason"] = "empty_lightcurve"
            return result

        if excluded and "camera#" in df_lc.columns:
            df_lc = df_lc[~df_lc["camera#"].isin(excluded)].reset_index(drop=True)

        df_lc = clean_lc(
            df_lc,
            max_error_absolute=float(clean_max_error_absolute),
            max_error_sigma=float(clean_max_error_sigma),
        )
        if df_lc.empty:
            result["phase_profile_reason"] = "empty_after_cleaning"
            return result

        n_points = int(len(df_lc))
        if n_points < int(min_points):
            result["phase_profile_reason"] = "too_few_points"
            result["n_points"] = n_points
            return result

        jd = df_lc["JD"].to_numpy(dtype=float)
        mag = df_lc["mag"].to_numpy(dtype=float)
        err = df_lc["error"].to_numpy(dtype=float)
        band = df_lc["v_g_band"].to_numpy(dtype=int) if "v_g_band" in df_lc.columns else np.zeros(n_points, dtype=int)
        camera = df_lc["camera#"].to_numpy(dtype=int) if "camera#" in df_lc.columns else np.ones(n_points, dtype=int)

        band_baseline = pd.Series(mag).groupby(band).transform("median").to_numpy(dtype=float)
        if "camera#" in df_lc.columns:
            group_keys = pd.MultiIndex.from_arrays([band, camera])
            group_sizes = pd.Series(mag).groupby(group_keys).transform("size").to_numpy(dtype=int)
            group_medians = pd.Series(mag).groupby(group_keys).transform("median").to_numpy(dtype=float)
            baseline = np.where(group_sizes >= 8, group_medians, band_baseline)
        else:
            baseline = band_baseline

        resid = mag - baseline
        jd0 = float(np.nanmin(jd))
        phase = np.mod((jd - jd0) / float(period), 1.0)
        cycle_index = np.floor((jd - jd0) / float(period)).astype(int)

        centers, profile, counts = _phase_profile(phase, resid, n_bins=int(phase_bins))
        smoothed = _circular_smooth(profile, int(profile_smooth_window))
        finite_profile = np.isfinite(smoothed)
        if not bool(finite_profile.any()):
            result["phase_profile_reason"] = "empty_phase_profile"
            return result

        baseline_resid = float(np.nanmedian(smoothed[finite_profile]))
        finite_bins = int(np.count_nonzero(finite_profile))
        profile_scatter = float(robust_sigma(smoothed[finite_profile] - baseline_resid))
        if not np.isfinite(profile_scatter) or profile_scatter <= 0:
            profile_scatter = float(robust_sigma(resid))
        if not np.isfinite(profile_scatter) or profile_scatter <= 0:
            profile_scatter = 1e-6

        peak_idx = int(np.nanargmax(smoothed))
        depth = float(smoothed[peak_idx] - baseline_resid)
        if not np.isfinite(depth) or depth <= 0:
            result.update(
                {
                    "n_points": n_points,
                    "jd_first": float(np.nanmin(jd)),
                    "jd_last": float(np.nanmax(jd)),
                    "cadence_median_days": float(median_dt(jd)),
                    "phase_reference_jd": jd0,
                    "phase_profile_baseline_resid": baseline_resid,
                    "phase_profile_scatter": profile_scatter,
                    "phase_profile_bins": int(phase_bins),
                    "phase_profile_finite_bins": finite_bins,
                    "baseline_mag": float(np.nanmedian(baseline)),
                    "n_cameras": int(pd.Series(camera).nunique()),
                    "camera_ids": ",".join(str(item) for item in sorted(pd.Series(camera).dropna().unique())),
                    "camera_min_points": int(pd.Series(camera).value_counts().min()),
                    "camera_max_points": int(pd.Series(camera).value_counts().max()),
                    "bad_cameras_filtered": ",".join(str(item) for item in sorted(set(int(c) for c in bad_cameras) | excluded)),
                    "phase_profile_reason": "no_positive_phase_dip",
                }
            )
            return result

        threshold = baseline_resid + max(0.35 * depth, 1.5 * profile_scatter)
        segment_mask = np.isfinite(smoothed) & (smoothed >= threshold)
        segment_indices = _circular_segment(segment_mask, peak_idx)
        if segment_indices.size == 0:
            segment_indices = np.asarray([peak_idx], dtype=int)

        width_phase = float(segment_indices.size / float(phase_bins))
        phase_start = float(segment_indices[0] / float(phase_bins))
        phase_end = float((segment_indices[-1] + 1) / float(phase_bins))
        phase_weights = np.maximum(smoothed[segment_indices] - baseline_resid, 0.0)
        phase_center = _circular_center(centers[segment_indices], phase_weights)
        if not np.isfinite(phase_center):
            phase_center = float(centers[peak_idx])

        point_mask = _phase_interval_mask(phase, phase_start, phase_end)
        support_points = int(np.count_nonzero(point_mask))
        if support_points == 0:
            result["phase_profile_reason"] = "no_phase_support_points"
            return result

        support_cycles = int(pd.Series(cycle_index[point_mask]).nunique())
        support_cameras = int(pd.Series(camera[point_mask]).nunique())
        support_by_cycle = pd.DataFrame(
            {
                "cycle": cycle_index[point_mask],
                "resid": resid[point_mask],
                "camera": camera[point_mask],
                "phase": phase[point_mask],
            }
        )
        cycle_group = support_by_cycle.groupby("cycle", sort=True)
        cycle_points = cycle_group.size()
        cycle_camera_counts = cycle_group["camera"].nunique()
        cycle_depths = cycle_group["resid"].median() - baseline_resid

        max_cycle_points = int(cycle_points.max()) if not cycle_points.empty else 0
        max_cycle_cameras = int(cycle_camera_counts.max()) if not cycle_camera_counts.empty else 0

        outside_mask = ~point_mask
        scatter = float(robust_sigma(resid[outside_mask]))
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = float(robust_sigma(resid))
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = profile_scatter
        scatter = max(scatter, 1e-6)

        depth_snr = float(depth / scatter)
        secondary_profile = smoothed.copy()
        secondary_profile[segment_indices] = np.nan
        secondary_depth = float(np.nanmax(secondary_profile) - baseline_resid) if np.isfinite(np.nanmax(secondary_profile)) else np.nan
        if not np.isfinite(secondary_depth):
            secondary_depth = np.nan
        secondary_ratio = float(secondary_depth / depth) if np.isfinite(secondary_depth) and depth > 0 else np.nan

        odd_mask = point_mask & ((cycle_index % 2) == 1)
        even_mask = point_mask & ((cycle_index % 2) == 0)
        odd_depth = float(np.nanmedian(resid[odd_mask]) - baseline_resid) if np.count_nonzero(odd_mask) > 0 else np.nan
        even_depth = float(np.nanmedian(resid[even_mask]) - baseline_resid) if np.count_nonzero(even_mask) > 0 else np.nan
        odd_even_diff = float(abs(odd_depth - even_depth)) if np.isfinite(odd_depth) and np.isfinite(even_depth) else np.nan
        odd_even_ratio = float(odd_even_diff / depth) if np.isfinite(odd_even_diff) and depth > 0 else np.nan

        cadence = float(median_dt(jd))
        width_days = float(width_phase * float(period))
        symmetry = _phase_window_symmetry(phase[point_mask], resid[point_mask] - baseline_resid, phase_center)

        amplitude_consistency = (
            float(robust_sigma(cycle_depths.to_numpy(dtype=float)) / depth)
            if depth > 0 and len(cycle_depths) > 1 else np.nan
        )
        cycle_times = sorted(float(jd0 + (cycle + phase_center) * float(period)) for cycle in cycle_group.groups)
        if len(cycle_times) > 1:
            spacing = np.diff(np.asarray(cycle_times, dtype=float))
            spacing_median = float(np.nanmedian(spacing))
            spacing_std = float(np.nanstd(spacing))
        else:
            spacing_median = np.nan
            spacing_std = np.nan

        n_cameras = int(pd.Series(camera).nunique())
        camera_counts = pd.Series(camera).value_counts()
        camera_ids = ",".join(str(item) for item in sorted(pd.Series(camera).dropna().unique()))
        bad_cameras_all = sorted(set(int(c) for c in bad_cameras) | excluded)

        significant = (
            depth_snr >= float(min_depth_snr)
            and support_points >= int(min_support_points)
            and support_cycles >= int(min_cycle_support)
            and support_cameras >= min(int(min_camera_support), max(n_cameras, 1))
            and width_phase <= float(max_dip_width_phase)
        )

        if support_cycles <= 1:
            reason = "single_cycle_only"
        elif support_points < int(min_support_points):
            reason = "insufficient_phase_support"
        elif depth_snr < float(min_depth_snr):
            reason = "low_depth_snr"
        elif support_cameras < min(int(min_camera_support), max(n_cameras, 1)):
            reason = "insufficient_camera_support"
        elif width_phase > float(max_dip_width_phase):
            reason = "dip_too_broad"
        else:
            reason = "significant_phase_dip" if significant else "weak_phase_dip"

        dip_bayes_factor = float(depth_snr ** 2 + np.log1p(max(support_cycles, 0)))
        dipper_score = _log10_score(depth, depth_snr, width_days, support_cycles, n_points)
        baseline_mag = float(np.nanmedian(baseline))
        dip_t0 = float(jd0 + phase_center * float(period))

        result.update(
            {
                "event_type": "phase_folded_dip" if significant else "none",
                "phase_reference_jd": jd0,
                "phase_dip_significant": bool(significant),
                "phase_dip_depth_mag": depth,
                "phase_dip_depth_snr": depth_snr,
                "phase_dip_phase_center": phase_center,
                "phase_dip_phase_start": phase_start,
                "phase_dip_phase_end": phase_end,
                "phase_dip_width_phase": width_phase,
                "phase_dip_width_days": width_days,
                "phase_dip_support_points": support_points,
                "phase_dip_support_cycles": support_cycles,
                "phase_dip_support_cameras": support_cameras,
                "phase_secondary_depth_mag": secondary_depth,
                "phase_secondary_depth_ratio": secondary_ratio,
                "phase_odd_depth_mag": odd_depth,
                "phase_even_depth_mag": even_depth,
                "phase_odd_even_depth_diff_mag": odd_even_diff,
                "phase_odd_even_depth_ratio": odd_even_ratio,
                "phase_profile_baseline_resid": baseline_resid,
                "phase_profile_scatter": scatter,
                "phase_profile_bins": int(phase_bins),
                "phase_profile_finite_bins": finite_bins,
                "phase_profile_reason": reason,
                "path": path_str,
                "dip_significant": bool(significant),
                "jump_significant": False,
                "n_points": n_points,
                "jd_first": float(np.nanmin(jd)),
                "jd_last": float(np.nanmax(jd)),
                "cadence_median_days": cadence,
                "dip_best_morph": "phase_folded_dip",
                "dip_best_delta_bic": float(depth_snr ** 2),
                "dip_best_width_param": width_phase,
                "dip_symmetry_score": symmetry,
                "dip_best_amp": depth,
                "dip_best_t0": dip_t0,
                "dip_count": support_cycles,
                "dip_run_count": support_cycles,
                "dip_max_run_points": max_cycle_points,
                "dip_max_run_duration": width_days,
                "dip_max_run_sum": float(depth * max_cycle_points),
                "dip_max_run_max": depth,
                "dip_max_run_cameras": max_cycle_cameras,
                "dip_max_log_bf_local": depth_snr,
                "dip_bayes_factor": dip_bayes_factor,
                "baseline_mag": baseline_mag,
                "dip_best_p": float(np.clip(depth / max(abs(baseline_mag), 1e-6), 0.0, np.inf)),
                "dip_best_mag_event": float(baseline_mag + depth),
                "dip_trigger_max": depth_snr,
                "n_cameras": n_cameras,
                "camera_ids": camera_ids,
                "camera_min_points": int(camera_counts.min()),
                "camera_max_points": int(camera_counts.max()),
                "dipper_score": dipper_score,
                "dipper_n_dips": support_cycles,
                "dipper_n_valid_dips": support_cycles if significant else 0,
                "bad_cameras_filtered": ",".join(str(item) for item in bad_cameras_all),
                "dip_is_single_event": bool(support_cycles <= 1),
                "dip_inter_event_spacing_median": spacing_median,
                "dip_inter_event_spacing_std": spacing_std,
                "dip_amplitude_consistency": amplitude_consistency,
                "dip_duration_consistency": 0.0,
            }
        )
        return result
    except Exception as exc:
        failed = _empty_result(path_str, checkpoint_key, period=period, error=str(exc), reason="error")
        return failed


def run_periodic_events(
    df: pd.DataFrame,
    *,
    path_col: str = "path",
    period_col: str = "pre_periodicity_selected_period",
    excluded_cameras_col: str | None = None,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
    phase_bins: int = PERIODIC_EVENTS_PHASE_BINS,
    profile_smooth_window: int = PERIODIC_EVENTS_PROFILE_SMOOTH_WINDOW,
    min_points: int = PERIODIC_EVENTS_MIN_POINTS,
    min_support_points: int = PERIODIC_EVENTS_MIN_SUPPORT_POINTS,
    min_cycle_support: int = PERIODIC_EVENTS_MIN_CYCLE_SUPPORT,
    min_camera_support: int = PERIODIC_EVENTS_MIN_CAMERA_SUPPORT,
    min_depth_snr: float = PERIODIC_EVENTS_DEPTH_SNR_THRESHOLD,
    max_dip_width_phase: float = PERIODIC_EVENTS_MAX_DIP_WIDTH_PHASE,
    workers: int = WORKERS,
    checkpoint_path: str | Path | None = None,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        if "path" not in out.columns and path_col in out.columns:
            out["path"] = out[path_col].astype(str)
        for col in PERIODIC_EVENT_RESULT_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    if path_col not in df.columns:
        raise ValueError(f"Need '{path_col}' column for periodic events")
    if period_col not in df.columns:
        raise ValueError(f"Need '{period_col}' column for periodic events")

    df_out = df.copy()
    df_out["path"] = df_out[path_col].astype(str)
    periods = pd.to_numeric(df_out[period_col], errors="coerce")
    excluded_values = (
        df_out[excluded_cameras_col]
        if excluded_cameras_col is not None and excluded_cameras_col in df_out.columns
        else pd.Series([None] * len(df_out), index=df_out.index)
    )
    checkpoint_keys = [
        _checkpoint_key(path, float(period) if np.isfinite(period) else np.nan, excluded)
        for path, period, excluded in zip(df_out["path"].astype(str), periods.to_numpy(dtype=float), excluded_values.tolist(), strict=True)
    ]
    df_out["_periodic_events_checkpoint_key"] = checkpoint_keys

    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
    result_map = _load_checkpoint(checkpoint)

    worker_args: list[tuple[object, ...]] = []
    for row_idx, row in df_out.iterrows():
        key = str(row["_periodic_events_checkpoint_key"])
        cached = result_map.get(key)
        if cached is not None and _result_is_usable(cached):
            continue
        period = periods.loc[row_idx]
        excluded = excluded_values.loc[row_idx]
        worker_args.append(
            (
                str(row["path"]),
                key,
                float(period) if np.isfinite(period) else np.nan,
                excluded,
                float(bad_camera_scatter_ratio),
                float(clean_max_error_absolute),
                float(clean_max_error_sigma),
                int(phase_bins),
                int(profile_smooth_window),
                int(min_points),
                int(min_support_points),
                int(min_cycle_support),
                int(min_camera_support),
                float(min_depth_snr),
                float(max_dip_width_phase),
            )
        )

    if show_tqdm:
        tqdm.write(
            f"[periodic_events] {len(result_map)} cached, processing {len(worker_args)} light curves"
        )

    if workers > 1 and len(worker_args) > 1:
        actual_workers = min(int(workers), len(worker_args))
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_evaluate_periodic_candidate_worker, item): item[1]
                for item in worker_args
            }
            iterator = as_completed(futures)
            if show_tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Periodic events")
            for future in iterator:
                result = future.result()
                result_map[str(result["periodic_events_checkpoint_key"])] = result
                _write_checkpoint(checkpoint, result_map)
    else:
        iterator = worker_args
        if show_tqdm:
            iterator = tqdm(worker_args, desc="Periodic events")
        for item in iterator:
            result = _evaluate_periodic_candidate_worker(item)
            result_map[str(result["periodic_events_checkpoint_key"])] = result
            _write_checkpoint(checkpoint, result_map)

    if not result_map:
        df_out = df_out.drop(columns=["_periodic_events_checkpoint_key"])
        return df_out

    results_df = pd.DataFrame(result_map.values())
    results_df = results_df.drop_duplicates(subset=["periodic_events_checkpoint_key"], keep="last")
    if "path" in df_out.columns and "path" in results_df.columns:
        results_df = results_df.drop(columns=["path"])
    df_out = df_out.merge(
        results_df,
        left_on="_periodic_events_checkpoint_key",
        right_on="periodic_events_checkpoint_key",
        how="left",
    )
    df_out = df_out.drop(columns=["_periodic_events_checkpoint_key"])
    return df_out
