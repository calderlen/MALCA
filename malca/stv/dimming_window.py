"""Recovery-anchored event-complex windows for MALCA light curves.

This module owns the generalized event-window estimator shared by scientific
diagnostic products and the Review/DustyCult integration.  It contains no
candidate-specific branches or manual window overrides.  Candidate identifiers
are retained only as provenance in the structured result. Dimming is the
default polarity; brightening uses the same estimator after orienting residuals
so the selected event is positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from malca.core.baseline import per_camera_gp_baseline_masked
from malca.core.utils import clean_lc
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame


DIMMING_WINDOW_METHOD_VERSION = "recovery_anchored_complex_v1"


@dataclass(frozen=True)
class DimmingWindowConfig:
    """Explicit thresholds for recovery-anchored event-window selection."""

    gp_s0: float = 0.0005
    gp_w0: float = 0.0031415926535897933
    gp_q: float = 0.7
    gp_jitter: float = 0.006
    recovery_window_days: float = 30.0
    quiet_core_percentile: float = 60.0
    quiet_scatter_floor_mag: float = 0.005
    quiet_scatter_ceiling_mag: float = 0.03
    uncertainty_floor_mag: float = 0.01
    recovery_tolerance_sigma: float = 2.0
    recovery_tolerance_floor_mag: float = 0.015
    recovery_tolerance_ceiling_mag: float = 0.025
    recovery_center_limit_mag: float = 0.015
    directional_center_limit_mag: float = 0.025
    recovery_window_sizes: tuple[int, ...] = (5, 4, 3)
    recovery_min_compatible_epochs: int = 3
    strong_dim_sigma: float = 3.0
    strong_dim_floor_mag: float = 0.02
    detection_sigma: float = 2.0
    detection_floor_mag: float = 0.02
    detection_ceiling_mag: float = 0.08
    peak_seed_epochs: int = 3
    peak_seed_required_strong_epochs: int = 2
    shallow_seed_min_epochs: int = 5
    crossing_gap_floor_days: float = 5.0
    crossing_gap_ceiling_days: float = 14.0
    crossing_gap_cadence_multiplier: float = 3.0


DEFAULT_DIMMING_WINDOW_CONFIG = DimmingWindowConfig()


@dataclass(frozen=True)
class DimmingComplexWindow:
    """Selected event envelope with explicit boundary and censoring state."""

    start_index: int
    stop_index: int
    start_jd: float
    end_jd: float
    status: str
    is_lower_limit: bool
    left_boundary_type: str
    right_boundary_type: str
    left_gap_state: str
    right_gap_state: str
    left_recovery_index: int | None
    right_recovery_index: int | None
    left_edge_dim_confirmed: bool
    right_edge_dim_confirmed: bool
    peak_jd: float
    peak_depth_mag: float
    peak_indices: tuple[int, ...]
    integrated_excess: float
    gap_count: int
    max_gap_days: float

    @property
    def duration_days(self) -> float:
        return max(0.0, float(self.end_jd - self.start_jd))

    @property
    def censoring_status(self) -> str:
        if not self.is_lower_limit:
            return "recovery_bounded"
        left_open = self.left_boundary_type != "recovery"
        right_open = self.right_boundary_type != "recovery"
        if left_open and right_open:
            return "both_censored"
        if left_open:
            return "left_censored"
        if right_open:
            return "right_censored"
        return "censored"

    def to_metrics(self, times: np.ndarray) -> dict[str, Any]:
        """Return stable flat fields used by tables, plots, and Review."""
        upper = np.nan if self.is_lower_limit else self.duration_days
        return {
            "dimming_window_method_version": DIMMING_WINDOW_METHOD_VERSION,
            "left_baseline_recovered": self.left_recovery_index is not None,
            "right_baseline_recovered": self.right_recovery_index is not None,
            "left_edge_dim_confirmed": self.left_edge_dim_confirmed,
            "right_edge_dim_confirmed": self.right_edge_dim_confirmed,
            "left_event_boundary_type": self.left_boundary_type,
            "right_event_boundary_type": self.right_boundary_type,
            "left_gap_boundary_state": self.left_gap_state,
            "right_gap_boundary_state": self.right_gap_state,
            "event_window_is_lower_limit": self.is_lower_limit,
            "left_recovery_jd": (
                float(times[self.left_recovery_index])
                if self.left_recovery_index is not None
                else np.nan
            ),
            "right_recovery_jd": (
                float(times[self.right_recovery_index])
                if self.right_recovery_index is not None
                else np.nan
            ),
            "left_recovery_is_gap_bracket": False,
            "right_recovery_is_gap_bracket": False,
            "event_window_start_jd": self.start_jd,
            "event_window_end_jd": self.end_jd,
            "event_window_duration_days": self.duration_days,
            "event_window_gap_count": self.gap_count,
            "event_window_max_gap_days": self.max_gap_days,
            "event_window_status": self.status,
            "event_continuity_assumed": bool(self.gap_count),
            "dimming_complex_start_jd": self.start_jd,
            "dimming_complex_end_jd": self.end_jd,
            "dimming_complex_duration_days": self.duration_days,
            "dimming_complex_duration_lower_days": self.duration_days,
            "dimming_complex_duration_upper_days": upper,
            "dimming_complex_duration_plot_days": self.duration_days,
            "dimming_complex_is_lower_limit": self.is_lower_limit,
            "dimming_complex_status": self.censoring_status,
            "event_selection_score": self.peak_depth_mag,
            "event_integrated_excess": self.integrated_excess,
            "event_component_epochs": self.stop_index - self.start_index + 1,
            "peak_initialization_points": len(self.peak_indices),
            "peak_jd": self.peak_jd,
            "delta_mag_peak": self.peak_depth_mag,
        }


@dataclass(frozen=True)
class DimmingComplexMeasurement:
    """Structured light-curve preparation and selected dimming window."""

    candidate_id: str
    lc_path: Path
    observations: pd.DataFrame
    epochs: pd.DataFrame
    smoothed_residual: np.ndarray
    smoothed_sigma: np.ndarray
    cadence_days: float
    smooth_window_days: float
    crossing_gap_limit_days: float
    quiet_scatter_mag: float
    detection_threshold_mag: float
    recovery_threshold_mag: float
    recovery_mask: np.ndarray
    recovery_support_mask: np.ndarray
    left_recovery_anchor_mask: np.ndarray
    right_recovery_anchor_mask: np.ndarray
    baseline_compatible_mask: np.ndarray
    strongly_dim_mask: np.ndarray
    detection_mask: np.ndarray
    window: DimmingComplexWindow
    n_good_observations: int


def _cadence_gap_cap(times: np.ndarray) -> tuple[float, float]:
    dt = np.diff(np.asarray(times, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan, 5.0
    cadence = float(np.median(dt))
    cadence_mad = float(1.4826 * np.median(np.abs(dt - cadence)))
    gap_cap = float(min(30.0, max(5.0 * cadence, cadence + 6.0 * cadence_mad, 1.0)))
    return cadence, gap_cap


def _local_epoch_median(
    times: np.ndarray,
    values: np.ndarray,
    cadence: float,
) -> tuple[np.ndarray, float]:
    """Return a three-epoch median without smoothing across major gaps."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    cadence_scale = cadence if np.isfinite(cadence) and cadence > 0 else 1.0
    smoothing_gap_limit = float(min(30.0, max(10.0, 5.0 * cadence_scale)))
    smoothed = np.full(values.shape, np.nan, dtype=float)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > smoothing_gap_limit) + 1, len(times)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = np.arange(start, stop)
        for index in block:
            order = np.argsort(np.abs(times[block] - times[index]))
            local = block[order[: min(3, len(block))]]
            finite = values[local][np.isfinite(values[local])]
            if finite.size:
                smoothed[index] = float(np.median(finite))
    return smoothed, smoothing_gap_limit


def _local_epoch_neighbor_matrix(times: np.ndarray, max_gap_days: float) -> np.ndarray:
    times = np.asarray(times, float)
    neighbors = np.full((len(times), 3), -1, dtype=int)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap_days) + 1, len(times)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = np.arange(start, stop)
        if len(block) < 3:
            continue
        for index in block:
            order = np.argsort(np.abs(times[block] - times[index]))
            neighbors[index] = block[order[:3]]
    return neighbors


def _mask_components(
    mask: np.ndarray,
    times: np.ndarray,
    max_gap_days: float,
) -> list[tuple[int, int]]:
    components: list[tuple[int, int]] = []
    index = 0
    while index < len(mask):
        if not bool(mask[index]):
            index += 1
            continue
        start = index
        while (
            index + 1 < len(mask)
            and bool(mask[index + 1])
            and times[index + 1] - times[index] <= max_gap_days
        ):
            index += 1
        components.append((start, index))
        index += 1
    return components


def _stable_recovery_mask(
    corrected_residual: np.ndarray,
    sigma: np.ndarray,
    times: np.ndarray,
    quiet_scatter_mag: float,
    max_gap_days: float,
    config: DimmingWindowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    corrected = np.asarray(corrected_residual, float)
    sigma = np.asarray(sigma, float)
    uncertainty = np.maximum.reduce(
        [
            np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0),
            np.full(corrected.shape, max(quiet_scatter_mag, 0.0)),
            np.full(corrected.shape, config.uncertainty_floor_mag),
        ]
    )
    tolerance = np.clip(
        config.recovery_tolerance_sigma * uncertainty,
        config.recovery_tolerance_floor_mag,
        config.recovery_tolerance_ceiling_mag,
    )
    baseline_compatible = np.abs(corrected) <= tolerance
    strongly_dim = corrected >= np.maximum(
        config.strong_dim_floor_mag,
        config.strong_dim_sigma * uncertainty,
    )
    support = np.zeros(corrected.shape, dtype=bool)
    confirmed = np.zeros(corrected.shape, dtype=bool)
    for start in range(len(corrected)):
        for width in config.recovery_window_sizes:
            stop = start + width
            if stop > len(corrected):
                continue
            window = np.arange(start, stop)
            if times[window[-1]] - times[window[0]] > max_gap_days:
                continue
            if int(np.sum(baseline_compatible[window])) < config.recovery_min_compatible_epochs:
                continue
            if abs(float(np.nanmedian(corrected[window]))) <= config.recovery_center_limit_mag:
                support[window] = True
                strict = window[
                    np.abs(corrected[window]) <= config.recovery_center_limit_mag
                ]
                if strict.size:
                    center = 0.5 * (window[0] + window[-1])
                    confirmed[int(strict[np.argmin(np.abs(strict - center))])] = True
                break
    return confirmed, support, baseline_compatible, strongly_dim, config.recovery_center_limit_mag


def _directional_recovery_anchor_mask(
    corrected_residual: np.ndarray,
    sigma: np.ndarray,
    times: np.ndarray,
    quiet_scatter_mag: float,
    max_gap_days: float,
    config: DimmingWindowConfig,
    *,
    side: str,
) -> np.ndarray:
    if side not in {"left", "right"}:
        raise ValueError(f"unknown recovery-anchor side {side!r}")
    corrected = np.asarray(corrected_residual, float)
    sigma = np.asarray(sigma, float)
    uncertainty = np.maximum.reduce(
        [
            np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0),
            np.full(corrected.shape, max(quiet_scatter_mag, 0.0)),
            np.full(corrected.shape, config.uncertainty_floor_mag),
        ]
    )
    tolerance = np.clip(
        config.recovery_tolerance_sigma * uncertainty,
        config.recovery_tolerance_floor_mag,
        config.recovery_tolerance_ceiling_mag,
    )
    compatible = np.abs(corrected) <= tolerance
    validated = np.zeros(corrected.shape, dtype=bool)
    strict_anchor = np.abs(corrected) <= config.recovery_center_limit_mag
    for anchor in np.flatnonzero(strict_anchor):
        for width in config.recovery_window_sizes:
            if side == "left":
                start, stop = int(anchor - width + 1), int(anchor + 1)
            else:
                start, stop = int(anchor), int(anchor + width)
            if start < 0 or stop > len(corrected):
                continue
            window = np.arange(start, stop)
            if times[window[-1]] - times[window[0]] > max_gap_days:
                continue
            if int(np.sum(compatible[window])) < config.recovery_min_compatible_epochs:
                continue
            if abs(float(np.nanmedian(corrected[window]))) <= config.directional_center_limit_mag:
                validated[anchor] = True
                break
    return validated


def _event_envelopes_from_recovery(
    recovery_mask: np.ndarray,
    left_anchor_mask: np.ndarray,
    right_anchor_mask: np.ndarray,
    times: np.ndarray,
    max_gap_days: float,
) -> list[dict[str, Any]]:
    plateaus = _mask_components(recovery_mask, times, max_gap_days)
    if not plateaus:
        return []
    directional: list[tuple[int | None, int | None]] = []
    for start, stop in plateaus:
        left_candidates = np.flatnonzero(
            left_anchor_mask
            & (times <= times[stop])
            & (times[stop] - times <= max_gap_days)
        )
        right_candidates = np.flatnonzero(
            right_anchor_mask
            & (times >= times[start])
            & (times - times[start] <= max_gap_days)
        )
        directional.append(
            (
                int(left_candidates[-1]) if left_candidates.size else None,
                int(right_candidates[0]) if right_candidates.size else None,
            )
        )

    envelopes: list[dict[str, Any]] = []
    first_start, _first_stop = plateaus[0]
    first_right = directional[0][1]
    first_boundary = first_right if first_right is not None else first_start
    if first_start > 0:
        envelopes.append(
            {
                "start": 0,
                "stop": int(first_boundary),
                "left_recovery": None,
                "right_recovery": first_right,
                "left_boundary_type": "data_edge",
                "right_boundary_type": (
                    "recovery" if first_right is not None else "unconfirmed_recovery"
                ),
            }
        )
    for plateau_index in range(len(plateaus) - 1):
        _, left_stop = plateaus[plateau_index]
        right_start, _ = plateaus[plateau_index + 1]
        left_anchor = directional[plateau_index][0]
        right_anchor = directional[plateau_index + 1][1]
        start = int(left_anchor if left_anchor is not None else left_stop)
        stop = int(right_anchor if right_anchor is not None else right_start)
        if stop > start + 1:
            envelopes.append(
                {
                    "start": start,
                    "stop": stop,
                    "left_recovery": left_anchor,
                    "right_recovery": right_anchor,
                    "left_boundary_type": (
                        "recovery" if left_anchor is not None else "unconfirmed_recovery"
                    ),
                    "right_boundary_type": (
                        "recovery" if right_anchor is not None else "unconfirmed_recovery"
                    ),
                }
            )
    _last_start, last_stop = plateaus[-1]
    last_left = directional[-1][0]
    last_boundary = last_left if last_left is not None else last_stop
    if last_stop + 1 < len(times):
        envelopes.append(
            {
                "start": int(last_boundary),
                "stop": len(times) - 1,
                "left_recovery": last_left,
                "right_recovery": None,
                "left_boundary_type": (
                    "recovery" if last_left is not None else "unconfirmed_recovery"
                ),
                "right_boundary_type": "data_edge",
            }
        )
    return envelopes


def _boundary_dim_supported(
    raw_residual: np.ndarray,
    corrected_residual: np.ndarray,
    detection_mask: np.ndarray,
    strongly_dim: np.ndarray,
    times: np.ndarray,
    index: int,
    *,
    direction: str,
    detection_threshold: float,
    max_gap_days: float,
) -> bool:
    if direction == "before":
        available = np.flatnonzero(
            (times <= times[index]) & (times[index] - times <= max_gap_days)
        )[-4:]
    elif direction == "after":
        available = np.flatnonzero(
            (times >= times[index]) & (times - times[index] <= max_gap_days)
        )[:4]
    else:
        raise ValueError(f"unknown boundary direction {direction!r}")
    if len(available) < 2:
        return False
    if raw_residual[index] < detection_threshold:
        return False
    if corrected_residual[index] < detection_threshold:
        return False
    required = 2 if len(available) < 4 else 3
    return bool(
        np.sum(detection_mask[available]) >= required
        or np.sum(strongly_dim[available]) >= 2
    )


def _edge_is_persistently_dim(
    raw_residual: np.ndarray,
    corrected_residual: np.ndarray,
    detection_mask: np.ndarray,
    strongly_dim: np.ndarray,
    times: np.ndarray,
    *,
    side: str,
    detection_threshold: float,
    max_gap_days: float,
) -> bool:
    if side not in {"left", "right"}:
        raise ValueError(f"unknown edge side {side!r}")
    edge = 0 if side == "left" else len(times) - 1
    if not np.isfinite(raw_residual[edge]) or raw_residual[edge] < detection_threshold:
        return False
    if not np.isfinite(corrected_residual[edge]) or corrected_residual[edge] < detection_threshold:
        return False
    if side == "left":
        available = np.flatnonzero(times - times[0] <= max_gap_days)[:4]
    else:
        available = np.flatnonzero(times[-1] - times <= max_gap_days)[-4:]
    if len(available) < 2:
        recent = (
            np.arange(min(4, len(times)))
            if side == "left"
            else np.arange(max(0, len(times) - 4), len(times))
        )
        return bool(
            strongly_dim[edge]
            and len(recent) >= 4
            and np.all(strongly_dim[recent])
        )
    required = 2 if len(available) < 4 else 3
    return bool(
        np.sum(strongly_dim[available]) >= 2
        or np.sum(detection_mask[available]) >= required
    )


def _capped_time_weights(times: np.ndarray, max_gap_days: float) -> np.ndarray:
    times = np.asarray(times, float)
    if len(times) == 1:
        return np.ones(1, dtype=float)
    weights = np.zeros(len(times), dtype=float)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap_days) + 1, len(times)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = times[start:stop]
        if len(block) == 1:
            weights[start] = 1.0
            continue
        gaps = np.diff(block)
        weights[start] = 0.5 * gaps[0]
        weights[stop - 1] = 0.5 * gaps[-1]
        if len(block) > 2:
            weights[start + 1 : stop - 1] = 0.5 * (gaps[:-1] + gaps[1:])
    return np.maximum(weights, 0.01)


def _close_single_epoch_holes(
    mask: np.ndarray,
    times: np.ndarray,
    max_span_days: float,
) -> np.ndarray:
    closed = np.asarray(mask, bool).copy()
    for index in range(1, len(closed) - 1):
        if (
            not closed[index]
            and closed[index - 1]
            and closed[index + 1]
            and times[index + 1] - times[index - 1] <= max_span_days
        ):
            closed[index] = True
    return closed


def _robust_peak_triplet(
    times: np.ndarray,
    residual: np.ndarray,
    local_profile: np.ndarray,
    start: int,
    stop: int,
    max_gap_days: float,
) -> tuple[float, float, np.ndarray]:
    neighbors = _local_epoch_neighbor_matrix(times, max_gap_days)
    best: tuple[float, float, np.ndarray] | None = None
    for anchor in range(start, stop + 1):
        indices = neighbors[anchor]
        indices = indices[indices >= 0]
        if len(indices) != 3 or np.any(indices < start) or np.any(indices > stop):
            continue
        depth = float(local_profile[anchor])
        if not np.isfinite(depth):
            continue
        raw_depth = float(np.nanmedian(residual[indices]))
        if not np.isclose(depth, raw_depth, rtol=0.0, atol=1e-12):
            continue
        peak_time = float(times[anchor])
        if best is None or depth > best[1]:
            best = (peak_time, depth, indices)
    if best is not None:
        return best
    return np.nan, np.nan, np.array([], dtype=int)


def _event_window_status(left: str, right: str) -> str:
    mapping = {
        ("recovery", "recovery"): "baseline_bounded",
        ("unconfirmed_recovery", "recovery"): "left_recovery_unconfirmed",
        ("recovery", "unconfirmed_recovery"): "right_recovery_unconfirmed",
        ("recovery", "data_edge"): "ongoing_right_censored",
        ("data_edge", "recovery"): "ongoing_left_censored",
        ("gap", "recovery"): "left_gap_censored",
        ("recovery", "gap"): "right_gap_censored",
        ("gap", "gap"): "both_gap_censored",
        ("gap", "data_edge"): "ongoing_right_left_gap_censored",
        ("data_edge", "gap"): "ongoing_left_right_gap_censored",
    }
    return mapping.get((left, right), "unanchored_no_baseline_recovery")


def select_dimming_complex_window(
    times: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    *,
    config: DimmingWindowConfig = DEFAULT_DIMMING_WINDOW_CONFIG,
    polarity: str = "dimming",
) -> tuple[DimmingComplexWindow, dict[str, Any]]:
    """Select the strongest recovery-anchored event from nightly residuals.

    Residuals are expected in magnitude units: positive values are dimmings and
    negative values are brightenings. Brightening selection negates residuals
    internally; returned times still refer to the original light curve.
    """
    times = np.asarray(times, float)
    residual = np.asarray(residual, float)
    sigma = np.asarray(sigma, float)
    event_polarity = str(polarity).strip().lower()
    if event_polarity not in {"dimming", "brightening"}:
        raise ValueError(f"unknown event polarity {polarity!r}")
    if event_polarity == "brightening":
        residual = -residual
    if not (len(times) == len(residual) == len(sigma)) or len(times) < 3:
        raise ValueError("dimming-window inputs must contain at least three aligned epochs")
    if np.any(np.diff(times) < 0):
        raise ValueError("dimming-window times must be sorted")

    cadence, _ = _cadence_gap_cap(times)
    smooth_residual, smooth_window = _local_epoch_median(times, residual, cadence)
    smooth_sigma, _ = _local_epoch_median(times, sigma, cadence)
    crossing_gap_limit = float(
        min(
            config.crossing_gap_ceiling_days,
            max(
                config.crossing_gap_floor_days,
                config.crossing_gap_cadence_multiplier * cadence
                if np.isfinite(cadence)
                else config.crossing_gap_floor_days,
            ),
        )
    )
    absolute_residual = np.abs(residual[np.isfinite(residual)])
    core_limit = (
        float(np.nanpercentile(absolute_residual, config.quiet_core_percentile))
        if absolute_residual.size
        else config.detection_floor_mag
    )
    quiet_values = residual[np.isfinite(residual) & (np.abs(residual) <= core_limit)]
    quiet_center = float(np.nanmedian(quiet_values)) if quiet_values.size else 0.0
    quiet_scatter = (
        float(1.4826 * np.nanmedian(np.abs(quiet_values - quiet_center)))
        if quiet_values.size
        else config.quiet_scatter_floor_mag
    )
    quiet_scatter = float(
        np.clip(
            quiet_scatter,
            config.quiet_scatter_floor_mag,
            config.quiet_scatter_ceiling_mag,
        )
    )
    (
        recovery_mask,
        recovery_support_mask,
        baseline_compatible,
        strongly_dim,
        recovery_threshold,
    ) = _stable_recovery_mask(
        residual,
        sigma,
        times,
        quiet_scatter,
        config.recovery_window_days,
        config,
    )
    left_anchor_mask = _directional_recovery_anchor_mask(
        residual,
        sigma,
        times,
        quiet_scatter,
        config.recovery_window_days,
        config,
        side="left",
    )
    right_anchor_mask = _directional_recovery_anchor_mask(
        residual,
        sigma,
        times,
        quiet_scatter,
        config.recovery_window_days,
        config,
        side="right",
    )
    finite_sigma = smooth_sigma[np.isfinite(smooth_sigma) & (smooth_sigma > 0)]
    typical_sigma = float(np.nanmedian(finite_sigma)) if finite_sigma.size else 0.01
    detection_threshold = float(
        max(
            config.detection_floor_mag,
            min(config.detection_ceiling_mag, config.detection_sigma * typical_sigma),
        )
    )
    detection_mask = _close_single_epoch_holes(
        smooth_residual >= detection_threshold,
        times,
        2.0 * crossing_gap_limit,
    )
    envelopes = _event_envelopes_from_recovery(
        recovery_mask,
        left_anchor_mask,
        right_anchor_mask,
        times,
        config.recovery_window_days,
    )
    time_weights = _capped_time_weights(times, smooth_window)
    candidates: list[dict[str, Any]] = []
    for envelope in envelopes:
        start = int(envelope["start"])
        stop = int(envelope["stop"])
        left_recovery = envelope["left_recovery"]
        right_recovery = envelope["right_recovery"]
        left_boundary = str(envelope["left_boundary_type"])
        right_boundary = str(envelope["right_boundary_type"])
        if stop - start + 1 < config.peak_seed_epochs:
            continue
        left_edge_dim = bool(
            left_boundary == "data_edge"
            and _edge_is_persistently_dim(
                residual,
                smooth_residual,
                detection_mask,
                strongly_dim,
                times,
                side="left",
                detection_threshold=detection_threshold,
                max_gap_days=smooth_window,
            )
        )
        right_edge_dim = bool(
            right_boundary == "data_edge"
            and _edge_is_persistently_dim(
                residual,
                smooth_residual,
                detection_mask,
                strongly_dim,
                times,
                side="right",
                detection_threshold=detection_threshold,
                max_gap_days=smooth_window,
            )
        )
        left_gap_dim = bool(
            left_boundary == "gap"
            and _boundary_dim_supported(
                residual,
                smooth_residual,
                detection_mask,
                strongly_dim,
                times,
                start,
                direction="after",
                detection_threshold=detection_threshold,
                max_gap_days=smooth_window,
            )
        )
        right_gap_dim = bool(
            right_boundary == "gap"
            and _boundary_dim_supported(
                residual,
                smooth_residual,
                detection_mask,
                strongly_dim,
                times,
                stop,
                direction="before",
                detection_threshold=detection_threshold,
                max_gap_days=smooth_window,
            )
        )
        left_gap_baseline = bool(left_boundary == "gap" and baseline_compatible[start])
        right_gap_baseline = bool(right_boundary == "gap" and baseline_compatible[stop])
        left_gap_state = (
            "none"
            if left_boundary != "gap"
            else ("dim" if left_gap_dim else ("baseline" if left_gap_baseline else "ambiguous"))
        )
        right_gap_state = (
            "none"
            if right_boundary != "gap"
            else ("dim" if right_gap_dim else ("baseline" if right_gap_baseline else "ambiguous"))
        )
        if left_boundary == "data_edge" and not left_edge_dim:
            continue
        if right_boundary == "data_edge" and not right_edge_dim:
            continue
        if left_boundary == "gap" and left_gap_state == "ambiguous":
            continue
        if right_boundary == "gap" and right_gap_state == "ambiguous":
            continue
        if left_recovery is None and right_recovery is None:
            continue

        strong_seed = False
        for seed_start in range(start, stop - config.peak_seed_epochs + 2):
            seed = np.arange(seed_start, seed_start + config.peak_seed_epochs)
            if seed[-1] > stop or times[seed[-1]] - times[seed[0]] > smooth_window:
                continue
            if int(np.sum(strongly_dim[seed])) >= config.peak_seed_required_strong_epochs:
                strong_seed = True
                break
        components = _mask_components(
            detection_mask[start : stop + 1],
            times[start : stop + 1],
            smooth_window,
        )
        shallow_seed = any(
            component_stop - component_start + 1 >= config.shallow_seed_min_epochs
            for component_start, component_stop in components
        )
        if not strong_seed and not shallow_seed:
            continue
        indices = np.arange(start, stop + 1)
        observed_excess = np.maximum(smooth_residual[indices] - recovery_threshold, 0.0)
        integrated_excess = float(np.sum(time_weights[indices] * observed_excess))
        peak_time, peak_depth, peak_indices = _robust_peak_triplet(
            times,
            residual,
            smooth_residual,
            start,
            stop,
            smooth_window,
        )
        if len(peak_indices) != config.peak_seed_epochs or not np.isfinite(peak_depth):
            continue
        candidates.append(
            {
                **envelope,
                "left_edge_dim": left_edge_dim,
                "right_edge_dim": right_edge_dim,
                "left_gap_state": left_gap_state,
                "right_gap_state": right_gap_state,
                "integrated_excess": integrated_excess,
                "peak_time": peak_time,
                "peak_depth": peak_depth,
                "peak_indices": peak_indices,
            }
        )
    if not candidates:
        raise RuntimeError(
            f"no recovery-anchored {event_polarity} bracket "
            "with a supported event seed"
        )
    selected = max(
        candidates,
        key=lambda candidate: (candidate["peak_depth"], candidate["integrated_excess"]),
    )
    start = int(selected["start"])
    stop = int(selected["stop"])
    left_boundary = str(selected["left_boundary_type"])
    right_boundary = str(selected["right_boundary_type"])
    gaps = np.diff(times[start : stop + 1])
    large_gaps = gaps[gaps > crossing_gap_limit]
    window = DimmingComplexWindow(
        start_index=start,
        stop_index=stop,
        start_jd=float(times[start]),
        end_jd=float(times[stop]),
        status=_event_window_status(left_boundary, right_boundary),
        is_lower_limit=left_boundary != "recovery" or right_boundary != "recovery",
        left_boundary_type=left_boundary,
        right_boundary_type=right_boundary,
        left_gap_state=str(selected["left_gap_state"]),
        right_gap_state=str(selected["right_gap_state"]),
        left_recovery_index=selected["left_recovery"],
        right_recovery_index=selected["right_recovery"],
        left_edge_dim_confirmed=bool(selected["left_edge_dim"]),
        right_edge_dim_confirmed=bool(selected["right_edge_dim"]),
        peak_jd=float(selected["peak_time"]),
        peak_depth_mag=float(selected["peak_depth"]),
        peak_indices=tuple(int(index) for index in selected["peak_indices"]),
        integrated_excess=float(selected["integrated_excess"]),
        gap_count=int(large_gaps.size),
        max_gap_days=float(np.max(large_gaps)) if large_gaps.size else 0.0,
    )
    diagnostics = {
        "event_polarity": event_polarity,
        "cadence_days": cadence,
        "smooth_window_days": smooth_window,
        "crossing_gap_limit_days": crossing_gap_limit,
        "smoothed_residual": smooth_residual,
        "smoothed_sigma": smooth_sigma,
        "quiet_scatter_mag": quiet_scatter,
        "detection_threshold_mag": detection_threshold,
        "recovery_threshold_mag": recovery_threshold,
        "recovery_mask": recovery_mask,
        "recovery_support_mask": recovery_support_mask,
        "left_recovery_anchor_mask": left_anchor_mask,
        "right_recovery_anchor_mask": right_anchor_mask,
        "baseline_compatible_mask": baseline_compatible,
        "strongly_dim_mask": strongly_dim,
        "detection_mask": detection_mask,
    }
    return window, diagnostics


def measure_dimming_complex_window(
    candidate_id: str,
    lc_path: str | Path,
    *,
    config: DimmingWindowConfig = DEFAULT_DIMMING_WINDOW_CONFIG,
    polarity: str = "dimming",
) -> DimmingComplexMeasurement:
    """Load, normalize, and measure one generalized event-complex window."""
    resolved_path = Path(lc_path).expanduser()
    canonical = load_lightcurve_df(
        resolved_path,
        filter_bad_cameras_enabled=True,
        apply_quality=True,
    )
    analysis = clean_lc(to_asassn_algorithm_frame(canonical))
    baseline = per_camera_gp_baseline_masked(
        analysis,
        S0=config.gp_s0,
        w0=config.gp_w0,
        q=config.gp_q,
        jitter=config.gp_jitter,
    )
    observations = pd.DataFrame(
        {
            "t": pd.to_numeric(baseline["JD"], errors="coerce"),
            "resid": pd.to_numeric(baseline["resid"], errors="coerce"),
            "sigma": pd.to_numeric(baseline["sigma_eff"], errors="coerce"),
        }
    ).dropna()
    observations = observations.loc[observations["sigma"] > 0].sort_values("t")
    if observations.empty:
        raise RuntimeError("no finite baseline residuals")
    observations["night"] = np.floor(observations["t"]).astype(int)
    epochs = (
        observations.groupby("night", sort=True)
        .agg(t=("t", "median"), resid=("resid", "median"), sigma=("sigma", "median"))
        .reset_index(drop=True)
    )
    times = epochs["t"].to_numpy(float)
    residual = epochs["resid"].to_numpy(float)
    sigma = epochs["sigma"].to_numpy(float)
    window, diagnostics = select_dimming_complex_window(
        times,
        residual,
        sigma,
        config=config,
        polarity=polarity,
    )
    return DimmingComplexMeasurement(
        candidate_id=str(candidate_id),
        lc_path=resolved_path,
        observations=observations,
        epochs=epochs,
        smoothed_residual=np.asarray(diagnostics["smoothed_residual"], float),
        smoothed_sigma=np.asarray(diagnostics["smoothed_sigma"], float),
        cadence_days=float(diagnostics["cadence_days"]),
        smooth_window_days=float(diagnostics["smooth_window_days"]),
        crossing_gap_limit_days=float(diagnostics["crossing_gap_limit_days"]),
        quiet_scatter_mag=float(diagnostics["quiet_scatter_mag"]),
        detection_threshold_mag=float(diagnostics["detection_threshold_mag"]),
        recovery_threshold_mag=float(diagnostics["recovery_threshold_mag"]),
        recovery_mask=np.asarray(diagnostics["recovery_mask"], bool),
        recovery_support_mask=np.asarray(diagnostics["recovery_support_mask"], bool),
        left_recovery_anchor_mask=np.asarray(diagnostics["left_recovery_anchor_mask"], bool),
        right_recovery_anchor_mask=np.asarray(diagnostics["right_recovery_anchor_mask"], bool),
        baseline_compatible_mask=np.asarray(diagnostics["baseline_compatible_mask"], bool),
        strongly_dim_mask=np.asarray(diagnostics["strongly_dim_mask"], bool),
        detection_mask=np.asarray(diagnostics["detection_mask"], bool),
        window=window,
        n_good_observations=int(len(analysis)),
    )


def dimming_complex_zoom_bounds(
    times: np.ndarray,
    *,
    start_jd: float,
    end_jd: float,
    peak_jd: float,
    cadence_days: float,
) -> tuple[float, float]:
    """Return the padded display window for a selected dimming complex.

    This is the display rule used by the all-dippers half-depth diagnostic
    atlas.  Selection of ``start_jd``/``end_jd`` is deliberately separate:
    those values come from the recovery-anchored complex estimator above.
    Keeping the modest padding rule here lets every consumer show the same
    selected dip without copying its edge handling.
    """
    finite_times = np.asarray(times, dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    if finite_times.size == 0:
        raise ValueError("cannot determine dimming zoom bounds without finite times")
    finite_times.sort()

    left = float(start_jd)
    right = float(end_jd)
    cadence = float(cadence_days)
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = 1.0
    if not (np.isfinite(left) and np.isfinite(right)) or right <= left:
        center = float(peak_jd)
        if not np.isfinite(center):
            center = float(np.nanmedian(finite_times))
        left = center - 0.5 * cadence
        right = center + 0.5 * cadence

    width = max(right - left, cadence, 0.5)
    margin = max(0.12 * width, 2.0 * cadence, 1.0)
    zoom_left = left if left <= finite_times[0] else max(finite_times[0], left - margin)
    zoom_right = right if right >= finite_times[-1] else min(finite_times[-1], right + margin)
    return float(zoom_left), float(zoom_right)


__all__ = [
    "DEFAULT_DIMMING_WINDOW_CONFIG",
    "DIMMING_WINDOW_METHOD_VERSION",
    "DimmingComplexMeasurement",
    "DimmingComplexWindow",
    "DimmingWindowConfig",
    "dimming_complex_zoom_bounds",
    "measure_dimming_complex_window",
    "select_dimming_complex_window",
]
