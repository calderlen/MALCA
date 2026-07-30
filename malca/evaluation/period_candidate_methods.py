"""Evaluation-only multi-method period candidate generation and scoring.

This module deliberately does not participate in MALCA's production period
pipeline.  It provides the common experimental machinery needed by the period
injection benchmark:

* independent global LS, PDM, CE, coarse-BLS, multiharmonic-LS,
  multiharmonic-AoV, Lafler--Kinman, SuperSmoother, and event-comb searches;
* frequency-resolution-aware peak extraction and candidate merging;
* explicit harmonic-family expansion; and
* fixed-period features for later deterministic or learned arbitration.

No bootstrap significance calculation is performed here.  Global searches are
deterministic for identical input arrays and configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import importlib
import math
from typing import Any, Literal

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle
from scipy.signal import find_peaks, peak_prominences

from malca.core.periodogram import (
    _ce_evaluate,
    _ce_scan_grid,
    _pdm_scan_grid,
    _pdm_scan_grid_plavchan,
    _pdm_theta,
    _pdm_theta_plavchan,
)
from malca.core.stats import (
    lafler_kinman_period_stats,
    phase_template_quasi_periodicity,
)


Objective = Literal["maximize", "minimize"]


GLOBAL_METHODS = (
    "ls_short",
    "ls_long",
    "multiharmonic_ls_long_2",
    "multiharmonic_ls_long_3",
    "pdm",
    "ce",
    "bls_coarse",
    "bls_adaptive",
    "multiharmonic_ls_2",
    "multiharmonic_ls_3",
    "multiharmonic_aov_2",
    "multiharmonic_aov_3",
    "lafler_kinman",
    "supersmoother",
    "event_comb",
)

FIXED_METHODS = (
    "ls",
    "pdm",
    "ce",
    "bls",
    "multiharmonic_fourier",
    "lafler_kinman",
    "supersmoother",
    "heldout_template",
    "odd_even",
    "event_coherence",
    "alias_evidence",
    "seasonal_stability",
    "null_model_comparison",
)


def _method_feature_key(method: str) -> str:
    """Return a stable identifier fragment for ranker one-hot columns."""
    return "".join(character if character.isalnum() else "_" for character in str(method)).strip("_")


def _independent_method_family(method: str) -> str:
    """Collapse deterministically related proposer views for support counts."""

    name = str(method)
    if name in {"ls_short", "ls_long", "local_ls", "local_ls_refinement"}:
        return "fourier_1"
    for order in (2, 3):
        if name in {
            f"multiharmonic_ls_{order}",
            f"multiharmonic_ls_long_{order}",
            f"multiharmonic_aov_{order}",
        }:
            # The AoV F value is a monotonic transform of the same nested
            # Fourier fit and therefore is not independent corroboration.
            return f"fourier_{order}"
    if name in {
        "bls_coarse",
        "bls_adaptive",
        "local_bls",
        "local_bls_refinement",
    }:
        return "bls"
    return name


@dataclass(frozen=True)
class PeriodCandidateMethodsConfig:
    """Configuration for the evaluation-only period candidate suite.

    The default grids are intentionally bounded so that a broad injection
    suite is practical.  For methods whose peaks narrow with observing
    baseline, ``general_min_samples_per_rayleigh`` raises the configured
    ``*_n_frequency`` floor to a baseline-aware resolution (subject to
    ``general_max_frequency_points``).  Set it to zero only for structural
    smoke tests.  All periodic grids are uniform in frequency.
    """

    enabled_global_methods: tuple[str, ...] = GLOBAL_METHODS
    enabled_fixed_methods: tuple[str, ...] = FIXED_METHODS
    # ``None`` scores the complete expanded bank.  The default retains that
    # bank for oracle-coverage accounting but applies expensive scorers only
    # to a generous proposal-priority shortlist.
    max_scored_candidates: int | None = 128
    shortlist_reserved_methods: tuple[str, ...] = (
        "ls_short",
        "ls_long",
        "multiharmonic_ls_long_2",
        "multiharmonic_ls_long_3",
        "pdm",
        "ce",
        "event_comb",
        "bls_adaptive",
        "multiharmonic_ls_2",
        "multiharmonic_ls_3",
        "lafler_kinman",
    )

    # Search bounds.  PDM/CE and the other general short searches use the
    # short interval; LS additionally receives a baseline-adaptive long band.
    short_min_period_days: float = 0.1
    short_max_period_days: float = 100.0
    long_min_period_days: float = 10.0
    long_max_baseline_fraction: float = 0.60
    long_absolute_max_period_days: float = 5000.0
    candidate_max_baseline_fraction: float = 0.80

    # Candidate counts and frequency resolution.
    top_k_ls: int = 10
    top_k_general: int = 5
    top_k_event: int = 10
    # Long searches reserve candidates by observed-cycle band so that dense
    # short-long overlap cannot consume every proposal slot.  Bounds are
    # expressed as increasing cycle counts over the realized baseline.
    long_cycle_band_edges: tuple[float, ...] = (1.5, 3.0, 8.0, 40.0)
    long_top_k_per_cycle_band: int = 5
    peak_min_rayleigh_separation: float = 1.0
    merge_rayleigh_separation: float = 1.0
    merge_relative_frequency_tolerance: float = 0.0
    harmonic_factors: tuple[float, ...] = (
        0.25,
        1.0 / 3.0,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
    )

    # Global search grids.
    ls_samples_per_peak: float = 5.0
    ls_max_frequency_points: int = 250_000
    general_min_samples_per_rayleigh: float = 1.0
    general_max_frequency_points: int = 50_000
    pdm_n_frequency: int = 8_000
    pdm_n_phase_bins: int = 20
    pdm_method: str = "classic"
    pdm_plavchan_phase_width: float = 0.05
    pdm_plavchan_min_neighbors: int = 3
    ce_n_frequency: int = 8_000
    ce_n_phase_bins: int = 20
    ce_n_mag_bins: int = 10
    bls_n_frequency: int = 2_000
    bls_period_groups: int = 12
    bls_max_period_ratio_per_group: float = 1.15
    bls_duration_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)
    bls_oversample: int = 5
    bls_adaptive_seed_top_k: int = 16
    bls_adaptive_top_k: int = 8
    bls_adaptive_refine_rayleigh_half_width: float = 3.0
    bls_adaptive_refine_frequency_oversample: float = 5.0
    bls_adaptive_refine_max_frequency_points_per_seed: int = 2_049
    multiharmonic_n_frequency: int = 8_000
    aov_n_frequency: int = 8_000
    lafler_kinman_n_frequency: int = 8_000
    supersmoother_n_frequency: int = 512

    # Fixed-period/local features.
    fixed_fourier_orders: tuple[int, ...] = (1, 2, 3)
    fixed_ls_refine_n_frequency: int = 101
    fixed_ls_refine_rayleigh_half_width: float = 2.0
    fixed_ls_refine_relative_half_width: float = 0.01
    fixed_bls_refine_n_frequency: int = 31
    fixed_bls_refine_rayleigh_half_width: float = 2.0
    fixed_bls_refine_relative_half_width: float = 0.0
    fixed_bls_refine_frequency_oversample: float = 5.0
    fixed_bls_refine_max_frequency_points: int = 4_097
    fixed_bls_max_refinement_seeds: int = 8
    local_refinement_sources: tuple[str, ...] = ("ls", "bls")
    max_refined_candidates: int | None = 64
    local_refinement_min_relative_shift: float = 1.0e-6
    template_n_phase_bins: int = 20
    template_min_bin_points: int = 2
    template_smooth_window_bins: int = 3
    template_min_bin_coverage: float = 0.30
    template_noise_subtract: bool = True
    odd_even_n_phase_bins: int = 20
    odd_even_min_cycles_per_parity: int = 2
    odd_even_min_common_bins: int = 3
    alias_periods_days: tuple[float, ...] = (0.5, 1.0, 29.53, 182.625, 365.25)
    alias_relative_tolerance: float = 0.10
    alias_rayleigh_tolerance: float = 1.0

    # Robust event-comb search.  Seeds are all dt/k hypotheses within bounds;
    # the cycle-count penalty prevents tiny integer subharmonics winning only
    # because they can align any finite set of event epochs.
    event_max_cycle_divisor: int = 64
    event_max_epochs: int = 64
    event_max_pair_lags: int = 12
    event_max_seed_hypotheses: int = 1_024
    event_phase_tolerance: float = 0.12
    event_min_inlier_fraction: float = 0.60
    event_min_events_for_ranking: int = 3
    event_min_cycle_span_for_ranking: int = 2
    event_cycle_complexity_penalty: float = 0.025
    event_seed_relative_tolerance: float = 2.0e-4

    # Candidate-level evidence used by the independent acceptance model.
    stability_segment_days: float = 365.25
    stability_min_points_per_segment: int = 12
    stability_min_segments: int = 2

    min_points: int = 10

    def __post_init__(self) -> None:
        unknown = sorted(set(self.enabled_global_methods) - set(GLOBAL_METHODS))
        if unknown:
            raise ValueError(f"Unknown global period methods: {unknown}")
        unknown_fixed = sorted(set(self.enabled_fixed_methods) - set(FIXED_METHODS))
        if unknown_fixed:
            raise ValueError(f"Unknown fixed-period methods: {unknown_fixed}")
        unknown_reserved = sorted(
            set(self.shortlist_reserved_methods) - set(GLOBAL_METHODS)
        )
        if unknown_reserved:
            raise ValueError(
                f"Unknown shortlist-reserved methods: {unknown_reserved}"
            )
        if self.max_scored_candidates is not None and int(self.max_scored_candidates) < 1:
            raise ValueError("max_scored_candidates must be positive or None")
        if not (
            np.isfinite(self.short_min_period_days)
            and np.isfinite(self.short_max_period_days)
            and 0 < self.short_min_period_days < self.short_max_period_days
        ):
            raise ValueError("short period bounds must be finite, positive, and increasing")
        if self.long_min_period_days <= 0:
            raise ValueError("long_min_period_days must be positive")
        for name in (
            "top_k_ls",
            "top_k_general",
            "top_k_event",
            "long_top_k_per_cycle_band",
            "pdm_n_frequency",
            "ce_n_frequency",
            "bls_n_frequency",
            "multiharmonic_n_frequency",
            "aov_n_frequency",
            "lafler_kinman_n_frequency",
            "supersmoother_n_frequency",
            "general_max_frequency_points",
            "bls_adaptive_seed_top_k",
            "bls_adaptive_top_k",
            "bls_adaptive_refine_max_frequency_points_per_seed",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            not np.isfinite(self.general_min_samples_per_rayleigh)
            or float(self.general_min_samples_per_rayleigh) < 0
        ):
            raise ValueError(
                "general_min_samples_per_rayleigh must be finite and non-negative"
            )
        if self.pdm_method not in {"classic", "plavchan"}:
            raise ValueError("pdm_method must be 'classic' or 'plavchan'")
        if any((not np.isfinite(v)) or v <= 0 for v in self.harmonic_factors):
            raise ValueError("harmonic_factors must be finite and positive")
        if (
            len(self.long_cycle_band_edges) < 2
            or any(
                (not np.isfinite(value)) or float(value) <= 0
                for value in self.long_cycle_band_edges
            )
            or any(
                float(right) <= float(left)
                for left, right in zip(
                    self.long_cycle_band_edges[:-1],
                    self.long_cycle_band_edges[1:],
                )
            )
        ):
            raise ValueError(
                "long_cycle_band_edges must contain at least two finite, "
                "positive, increasing cycle counts"
            )
        if any((not np.isfinite(v)) or not 0 < v < 1 for v in self.bls_duration_fractions):
            raise ValueError("bls_duration_fractions must lie strictly between zero and one")
        if (
            not np.isfinite(self.bls_max_period_ratio_per_group)
            or float(self.bls_max_period_ratio_per_group) <= 1.0
        ):
            raise ValueError("bls_max_period_ratio_per_group must exceed one")
        if any((not np.isfinite(v)) or v <= 0 for v in self.alias_periods_days):
            raise ValueError("alias_periods_days must be finite and positive")
        if self.fixed_ls_refine_n_frequency < 3:
            raise ValueError("fixed_ls_refine_n_frequency must be at least three")
        if self.fixed_bls_refine_n_frequency < 3:
            raise ValueError("fixed_bls_refine_n_frequency must be at least three")
        if (
            not np.isfinite(self.fixed_bls_refine_frequency_oversample)
            or float(self.fixed_bls_refine_frequency_oversample) <= 0
        ):
            raise ValueError(
                "fixed_bls_refine_frequency_oversample must be positive"
            )
        if int(self.fixed_bls_refine_max_frequency_points) < 3:
            raise ValueError(
                "fixed_bls_refine_max_frequency_points must be at least three"
            )
        if int(self.fixed_bls_max_refinement_seeds) < 1:
            raise ValueError("fixed_bls_max_refinement_seeds must be positive")
        if (
            not np.isfinite(self.bls_adaptive_refine_rayleigh_half_width)
            or float(self.bls_adaptive_refine_rayleigh_half_width) <= 0
        ):
            raise ValueError(
                "bls_adaptive_refine_rayleigh_half_width must be positive"
            )
        if (
            not np.isfinite(self.bls_adaptive_refine_frequency_oversample)
            or float(self.bls_adaptive_refine_frequency_oversample) <= 0
        ):
            raise ValueError(
                "bls_adaptive_refine_frequency_oversample must be positive"
            )
        if int(self.odd_even_min_cycles_per_parity) < 1:
            raise ValueError("odd_even_min_cycles_per_parity must be positive")
        if int(self.odd_even_min_common_bins) < 1:
            raise ValueError("odd_even_min_common_bins must be positive")
        unknown_refiners = sorted(set(self.local_refinement_sources) - {"ls", "bls"})
        if unknown_refiners:
            raise ValueError(f"Unknown local refinement sources: {unknown_refiners}")
        if self.max_refined_candidates is not None and int(self.max_refined_candidates) < 1:
            raise ValueError("max_refined_candidates must be positive or None")
        for name in (
            "event_max_cycle_divisor",
            "event_max_epochs",
            "event_max_pair_lags",
            "event_max_seed_hypotheses",
            "event_min_events_for_ranking",
            "event_min_cycle_span_for_ranking",
            "stability_min_points_per_segment",
            "stability_min_segments",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            not np.isfinite(self.stability_segment_days)
            or float(self.stability_segment_days) <= 0
        ):
            raise ValueError("stability_segment_days must be positive")


@dataclass
class PeriodCandidate:
    """One candidate period proposed by a global search or harmonic expansion."""

    method: str
    period_days: float
    frequency_per_day: float
    raw_score: float
    objective: Objective
    rank: int
    prominence: float = np.nan
    normalized_score: float = np.nan
    search_band: str = "all"
    parent_period_days: float = np.nan
    harmonic_factor: float = 1.0
    contributing_methods: tuple[str, ...] = ()
    was_scored: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a flat, serialization-friendly record."""
        record = asdict(self)
        metadata = record.pop("metadata")
        record.update({f"meta_{key}": value for key, value in metadata.items()})
        return record


@dataclass
class PeriodSearchResult:
    """Result of one global period search."""

    method: str
    objective: Objective
    candidates: list[PeriodCandidate] = field(default_factory=list)
    period_grid_days: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    metric_grid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    status: str = "ok"
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def candidate_records(self) -> list[dict[str, Any]]:
        """Return candidate records annotated with the search status."""
        records = []
        for candidate in self.candidates:
            record = candidate.to_record()
            record["search_status"] = self.status
            records.append(record)
        return records


@dataclass
class CandidateScore:
    """Fixed-period feature vector for one merged/expanded candidate."""

    period_days: float
    frequency_per_day: float
    features: dict[str, Any]
    status: str = "ok"
    contributing_methods: tuple[str, ...] = ()
    parent_period_days: float = np.nan
    harmonic_factor: float = 1.0
    proposal_method: str = ""
    proposal_raw_score: float = np.nan
    proposal_objective: str = ""
    proposal_rank: int = 0
    proposal_prominence: float = np.nan
    proposal_normalized_score: float = np.nan
    proposal_search_band: str = ""
    proposal_metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a flat record suitable for a candidate-level table."""
        representative_key = _method_feature_key(self.proposal_method)
        contributing = set(self.contributing_methods)
        independent_families = {
            _independent_method_family(method) for method in contributing
        }
        record = {
            "period_days": self.period_days,
            "frequency_per_day": self.frequency_per_day,
            "status": self.status,
            "contributing_methods": self.contributing_methods,
            "parent_period_days": self.parent_period_days,
            "harmonic_factor": self.harmonic_factor,
            "proposal_method": self.proposal_method,
            "proposal_raw_score": self.proposal_raw_score,
            "proposal_objective": self.proposal_objective,
            "proposal_rank": self.proposal_rank,
            "proposal_prominence": self.proposal_prominence,
            "proposal_normalized_score": self.proposal_normalized_score,
            "proposal_search_band": self.proposal_search_band,
            "proposal_contributing_methods": self.contributing_methods,
            "proposal_contributing_method_count": int(len(contributing)),
            "proposal_independent_method_family_count": int(
                len(independent_families)
            ),
            "proposal_parent_period_days": self.parent_period_days,
            "proposal_harmonic_factor": self.harmonic_factor,
            "proposal_is_harmonic_expansion": int(
                not np.isclose(self.harmonic_factor, 1.0)
                or self.proposal_method == "harmonic_expansion"
            ),
            "was_scored": 1,
            "proposal_merged_count": int(
                self.proposal_metadata.get("merged_candidate_count", 1)
            ),
            **self.features,
        }
        for method in GLOBAL_METHODS:
            key = _method_feature_key(method)
            record[f"proposal_method_is_{key}"] = int(key == representative_key)
            record[f"proposal_contributes_{key}"] = int(method in contributing)
        record["proposal_method_is_harmonic_expansion"] = int(
            self.proposal_method == "harmonic_expansion"
        )
        for key in (
            "seed_period_days",
            "seed_method",
            "merged_harmonic_factors",
            "merged_parent_periods_days",
        ):
            if key in self.proposal_metadata:
                record[f"proposal_{key}"] = self.proposal_metadata[key]
        return record


@dataclass
class PeriodCandidateSuiteResult:
    """All outputs from an end-to-end evaluation-only candidate run."""

    config: PeriodCandidateMethodsConfig
    search_results: dict[str, PeriodSearchResult]
    raw_candidates: list[PeriodCandidate]
    merged_candidates: list[PeriodCandidate]
    expanded_candidates: list[PeriodCandidate]
    candidate_scores: list[CandidateScore]
    status: str = "ok"

    def candidate_records(self) -> list[dict[str, Any]]:
        """Return flat fixed-period feature records."""
        return [score.to_record() for score in self.candidate_scores]

    def search_records(self) -> list[dict[str, Any]]:
        """Return flat records for every raw global-search proposal."""
        return [
            record
            for result in self.search_results.values()
            for record in result.candidate_records()
        ]

    def expanded_candidate_records(self) -> list[dict[str, Any]]:
        """Return the complete oracle bank, including ``was_scored`` flags."""
        return [candidate.to_record() for candidate in self.expanded_candidates]


def _prepare_light_curve(
    time: Sequence[float],
    mag: Sequence[float],
    err: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate, finite-filter, time-sort, and time-center a light curve."""
    time_arr = np.asarray(time, dtype=float)
    mag_arr = np.asarray(mag, dtype=float)
    if time_arr.ndim != 1 or mag_arr.ndim != 1 or time_arr.size != mag_arr.size:
        raise ValueError("time and mag must be one-dimensional arrays of equal length")

    err_arr: np.ndarray | None
    if err is None:
        err_arr = None
        mask = np.isfinite(time_arr) & np.isfinite(mag_arr)
    else:
        err_arr = np.asarray(err, dtype=float)
        if err_arr.ndim != 1 or err_arr.size != time_arr.size:
            raise ValueError("err must be one-dimensional and match time")
        mask = (
            np.isfinite(time_arr)
            & np.isfinite(mag_arr)
            & np.isfinite(err_arr)
            & (err_arr > 0)
        )

    time_arr = time_arr[mask]
    mag_arr = mag_arr[mask]
    if err_arr is not None:
        err_arr = err_arr[mask]
    if time_arr.size:
        order = np.argsort(time_arr, kind="mergesort")
        time_arr = time_arr[order]
        mag_arr = mag_arr[order]
        if err_arr is not None:
            err_arr = err_arr[order]
        time_arr = time_arr - float(time_arr[0])
    return time_arr, mag_arr, err_arr


def _frequency_grid(
    min_period_days: float,
    max_period_days: float,
    *,
    n_frequency: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return increasing frequency and corresponding decreasing period grids."""
    min_period = float(min_period_days)
    max_period = float(max_period_days)
    if not np.isfinite(min_period) or not np.isfinite(max_period) or min_period <= 0:
        raise ValueError("period bounds must be finite and positive")
    if max_period <= min_period:
        raise ValueError("max_period_days must exceed min_period_days")
    frequency = np.linspace(1.0 / max_period, 1.0 / min_period, max(2, int(n_frequency)))
    return frequency, 1.0 / frequency


def _resolution_aware_frequency_count(
    min_period_days: float,
    max_period_days: float,
    *,
    baseline_days: float,
    requested_n_frequency: int,
    config: PeriodCandidateMethodsConfig,
) -> tuple[int, dict[str, Any]]:
    """Resolve a bounded grid floor in units of independent Fourier spacing."""

    requested = max(2, int(requested_n_frequency))
    frequency_span = (
        1.0 / float(min_period_days) - 1.0 / float(max_period_days)
    )
    samples_per_rayleigh = float(config.general_min_samples_per_rayleigh)
    rayleigh_floor = (
        int(math.ceil(frequency_span * float(baseline_days) * samples_per_rayleigh))
        + 1
        if samples_per_rayleigh > 0
        else 2
    )
    capped_floor = min(
        max(2, rayleigh_floor),
        int(config.general_max_frequency_points),
    )
    resolved = max(requested, capped_floor)
    actual_samples = (
        float((resolved - 1) / (frequency_span * float(baseline_days)))
        if frequency_span > 0 and baseline_days > 0
        else np.nan
    )
    return resolved, {
        "requested_n_frequency": requested,
        "resolved_n_frequency": resolved,
        "requested_min_samples_per_rayleigh": samples_per_rayleigh,
        "actual_samples_per_rayleigh": actual_samples,
        "resolution_cap_hit": bool(
            samples_per_rayleigh > 0
            and rayleigh_floor > int(config.general_max_frequency_points)
            and requested < rayleigh_floor
        ),
    }


def _metric_normalized_score(value: float, metric: np.ndarray, maximize: bool) -> float:
    """Map a method-local metric to a robust 0--1 score."""
    finite = np.asarray(metric, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or not np.isfinite(value):
        return np.nan
    low, high = np.nanpercentile(finite, [5.0, 95.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.5
    score = (float(value) - low) / (high - low)
    if not maximize:
        score = 1.0 - score
    return float(np.clip(score, 0.0, 1.0))


def _metric_has_dynamic_range(metric: np.ndarray) -> bool:
    finite = np.asarray(metric, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return finite.size == 1
    scale = max(1.0, float(np.max(np.abs(finite))))
    return bool(float(np.ptp(finite)) > 64.0 * np.finfo(float).eps * scale)


def extract_frequency_separated_extrema(
    periods: Sequence[float],
    metric: Sequence[float],
    *,
    top_k: int,
    maximize: bool,
    baseline_days: float,
    min_frequency_separation: float | None = None,
    method: str = "unknown",
    search_band: str = "all",
) -> list[PeriodCandidate]:
    """Extract genuine local extrema separated in physical frequency.

    The input grid may be increasing or decreasing and need not be uniform.
    Candidate separation defaults to one Rayleigh resolution element,
    ``1 / baseline_days``.  Plateau extrema use SciPy's deterministic midpoint
    convention.  Grid endpoints are accepted only when they are strictly more
    favorable than their sole neighbor.
    """
    period_arr = np.asarray(periods, dtype=float)
    metric_arr = np.asarray(metric, dtype=float)
    if period_arr.ndim != 1 or metric_arr.ndim != 1 or period_arr.size != metric_arr.size:
        raise ValueError("periods and metric must be one-dimensional and equal length")
    valid = (
        np.isfinite(period_arr)
        & (period_arr > 0)
        & np.isfinite(metric_arr)
    )
    if not np.any(valid) or int(top_k) <= 0:
        return []

    frequency = 1.0 / period_arr[valid]
    values = metric_arr[valid]
    if not _metric_has_dynamic_range(values):
        return []
    order = np.argsort(frequency, kind="mergesort")
    frequency = frequency[order]
    values = values[order]
    periods_sorted = 1.0 / frequency

    target = values if maximize else -values
    peak_idx, _ = find_peaks(target, plateau_size=True)
    extrema = list(map(int, peak_idx))
    if target.size == 1:
        extrema = [0]
    elif target.size >= 2:
        # Require a strict improvement at a boundary.  With ``>=`` a metric
        # whose tails underflow to an exactly flat zero contributes a spurious
        # endpoint candidate after the genuine interior extrema.
        if target[0] > target[1]:
            extrema.append(0)
        if target[-1] > target[-2]:
            extrema.append(target.size - 1)
    if not extrema:
        extrema = [int(np.nanargmax(target))]
    extrema = sorted(set(extrema))

    prominence = np.zeros(len(extrema), dtype=float)
    interior_positions = [idx for idx, value in enumerate(extrema) if 0 < value < target.size - 1]
    if interior_positions:
        interior_indices = np.asarray([extrema[idx] for idx in interior_positions], dtype=int)
        interior_prom = peak_prominences(target, interior_indices)[0]
        for position, value in zip(interior_positions, interior_prom):
            prominence[position] = float(value)
    for pos, idx in enumerate(extrema):
        if idx == 0 and target.size > 1:
            prominence[pos] = max(prominence[pos], float(max(0.0, target[0] - target[1])))
        elif idx == target.size - 1 and target.size > 1:
            prominence[pos] = max(prominence[pos], float(max(0.0, target[-1] - target[-2])))

    extrema_order = sorted(
        range(len(extrema)),
        key=lambda pos: (-float(target[extrema[pos]]), -float(prominence[pos]), int(extrema[pos])),
    )
    baseline = float(baseline_days)
    if min_frequency_separation is None:
        min_sep = 1.0 / baseline if np.isfinite(baseline) and baseline > 0 else 0.0
    else:
        min_sep = max(0.0, float(min_frequency_separation))

    chosen: list[tuple[int, float]] = []
    for pos in extrema_order:
        idx = extrema[pos]
        candidate_frequency = float(frequency[idx])
        if any(abs(candidate_frequency - frequency[other]) < min_sep for other, _ in chosen):
            continue
        chosen.append((idx, float(prominence[pos])))
        if len(chosen) >= int(top_k):
            break

    objective: Objective = "maximize" if maximize else "minimize"
    candidates = []
    for rank, (idx, candidate_prominence) in enumerate(chosen, start=1):
        raw_score = float(values[idx])
        candidates.append(
            PeriodCandidate(
                method=str(method),
                period_days=float(periods_sorted[idx]),
                frequency_per_day=float(frequency[idx]),
                raw_score=raw_score,
                objective=objective,
                rank=rank,
                prominence=candidate_prominence,
                normalized_score=_metric_normalized_score(raw_score, values, maximize),
                search_band=str(search_band),
                contributing_methods=(str(method),),
            )
        )
    return candidates


def _empty_search(
    method: str,
    objective: Objective,
    status: str,
    message: str = "",
    **metadata: Any,
) -> PeriodSearchResult:
    return PeriodSearchResult(
        method=method,
        objective=objective,
        status=status,
        message=message,
        metadata=metadata,
    )


def _search_from_grid(
    method: str,
    objective: Objective,
    period_grid: np.ndarray,
    metric_grid: np.ndarray,
    *,
    top_k: int,
    baseline_days: float,
    min_rayleigh_separation: float,
    search_band: str,
    metadata: Mapping[str, Any] | None = None,
) -> PeriodSearchResult:
    finite_metric = np.asarray(metric_grid, dtype=float)
    finite_metric = finite_metric[np.isfinite(finite_metric)]
    candidates = extract_frequency_separated_extrema(
        period_grid,
        metric_grid,
        top_k=top_k,
        maximize=objective == "maximize",
        baseline_days=baseline_days,
        min_frequency_separation=(
            float(min_rayleigh_separation) / float(baseline_days)
            if np.isfinite(baseline_days) and baseline_days > 0
            else 0.0
        ),
        method=method,
        search_band=search_band,
    )
    return PeriodSearchResult(
        method=method,
        objective=objective,
        candidates=candidates,
        period_grid_days=np.asarray(period_grid, dtype=float),
        metric_grid=np.asarray(metric_grid, dtype=float),
        status=(
            "ok"
            if candidates
            else (
                "flat_metric"
                if finite_metric.size >= 2
                and not _metric_has_dynamic_range(finite_metric)
                else "no_finite_extrema"
            )
        ),
        metadata=dict(metadata or {}),
    )


def _global_ls(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    method: str,
    min_period: float,
    max_period: float,
    top_k: int,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
    search_band: str,
    nterms: int = 1,
    n_frequency: int | None = None,
) -> PeriodSearchResult:
    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period
    requested_n_frequency = n_frequency
    rayleigh_count = int(
        math.ceil(
            (max_frequency - min_frequency)
            * baseline
            * config.ls_samples_per_peak
        )
    ) + 1
    if n_frequency is None:
        n_frequency = min(max(2, rayleigh_count), int(config.ls_max_frequency_points))
    frequency = np.linspace(min_frequency, max_frequency, max(2, int(n_frequency)))
    try:
        ls = LombScargle(time, mag, err, nterms=int(nterms))
        power = np.asarray(ls.power(frequency, method="chi2"), dtype=float)
    except Exception as exc:
        return _empty_search(method, "maximize", "error", str(exc))
    return _search_from_grid(
        method,
        "maximize",
        1.0 / frequency,
        power,
        top_k=top_k,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band=search_band,
        metadata={
            "nterms": int(nterms),
            "grid_kind": "uniform_frequency",
            "requested_n_frequency": (
                None
                if requested_n_frequency is None
                else int(requested_n_frequency)
            ),
            "resolved_n_frequency": int(frequency.size),
            "requested_samples_per_rayleigh": float(
                config.ls_samples_per_peak
            ),
            "actual_samples_per_rayleigh": float(
                (frequency.size - 1)
                / ((max_frequency - min_frequency) * baseline)
            ),
            "resolution_cap_hit": bool(
                requested_n_frequency is None
                and rayleigh_count > int(config.ls_max_frequency_points)
            ),
        },
    )


def _long_cycle_bands(
    *,
    baseline: float,
    long_min_period: float,
    long_max_period: float,
    config: PeriodCandidateMethodsConfig,
) -> list[tuple[float, float, float, float]]:
    """Return non-overlapping long-period bounds with explicit cycle quotas.

    Each tuple is ``(minimum_period, maximum_period, minimum_cycles,
    maximum_cycles)``.  The configured cycle edges are extended to the
    realized search limits so ``ls_long`` still covers the complete requested
    period interval.
    """

    minimum_cycles = float(baseline) / float(long_max_period)
    maximum_cycles = float(baseline) / float(long_min_period)
    edges = [
        value
        for value in map(float, config.long_cycle_band_edges)
        if minimum_cycles < value < maximum_cycles
    ]
    cycle_edges = np.unique(
        np.asarray([minimum_cycles, *edges, maximum_cycles], dtype=float)
    )
    bands: list[tuple[float, float, float, float]] = []
    for low_cycles, high_cycles in zip(cycle_edges[:-1], cycle_edges[1:]):
        minimum_period = max(
            float(long_min_period),
            float(baseline) / float(high_cycles),
        )
        maximum_period = min(
            float(long_max_period),
            float(baseline) / float(low_cycles),
        )
        if maximum_period > minimum_period:
            bands.append(
                (
                    minimum_period,
                    maximum_period,
                    float(low_cycles),
                    float(high_cycles),
                )
            )
    return bands


def _global_cycle_banded_ls(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    method: str,
    baseline: float,
    long_max_period: float,
    config: PeriodCandidateMethodsConfig,
    nterms: int = 1,
) -> PeriodSearchResult:
    """Run LS independently in observed-cycle bands and retain a quota per band."""

    if long_max_period <= float(config.long_min_period_days):
        return _empty_search(
            method,
            "maximize",
            "invalid_long_bounds",
            long_max_period_days=float(long_max_period),
        )
    bands = _long_cycle_bands(
        baseline=baseline,
        long_min_period=float(config.long_min_period_days),
        long_max_period=float(long_max_period),
        config=config,
    )
    band_results: list[PeriodSearchResult] = []
    candidates: list[PeriodCandidate] = []
    for band_index, (
        minimum_period,
        maximum_period,
        minimum_cycles,
        maximum_cycles,
    ) in enumerate(bands):
        band_name = (
            f"long_cycles_{minimum_cycles:.3g}_{maximum_cycles:.3g}"
        )
        result = _global_ls(
            time,
            mag,
            err,
            method=method,
            min_period=minimum_period,
            max_period=maximum_period,
            top_k=int(config.long_top_k_per_cycle_band),
            baseline=baseline,
            config=config,
            search_band=band_name,
            nterms=int(nterms),
        )
        result.metadata.update(
            {
                "cycle_band_index": int(band_index),
                "cycle_band_min_cycles": float(minimum_cycles),
                "cycle_band_max_cycles": float(maximum_cycles),
                "cycle_band_min_period_days": float(minimum_period),
                "cycle_band_max_period_days": float(maximum_period),
            }
        )
        for candidate in result.candidates:
            candidate.metadata.update(result.metadata)
        band_results.append(result)
        candidates.extend(result.candidates)

    # LS powers are comparable across the bands because every band uses the
    # same normalization and harmonic order.  Keep each band's quota, then
    # assign one deterministic global rank without discarding sparse bands.
    candidates.sort(
        key=lambda candidate: (
            -float(candidate.raw_score)
            if np.isfinite(candidate.raw_score)
            else np.inf,
            float(candidate.frequency_per_day),
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = int(rank)

    periods = np.concatenate(
        [
            result.period_grid_days
            for result in band_results
            if result.period_grid_days.size
        ]
    ) if any(result.period_grid_days.size for result in band_results) else np.empty(0)
    metric = np.concatenate(
        [
            result.metric_grid
            for result in band_results
            if result.metric_grid.size
        ]
    ) if any(result.metric_grid.size for result in band_results) else np.empty(0)
    statuses = [result.status for result in band_results]
    return PeriodSearchResult(
        method=method,
        objective="maximize",
        candidates=candidates,
        period_grid_days=np.asarray(periods, dtype=float),
        metric_grid=np.asarray(metric, dtype=float),
        status=(
            "ok"
            if candidates
            else ("error" if any(status == "error" for status in statuses) else "no_finite_extrema")
        ),
        message="; ".join(
            result.message for result in band_results if result.message
        ),
        metadata={
            "search_strategy": "cycle_banded_frequency_quota",
            "nterms": int(nterms),
            "n_cycle_bands": int(len(bands)),
            "top_k_per_cycle_band": int(config.long_top_k_per_cycle_band),
            "cycle_bands": tuple(
                {
                    "minimum_period_days": float(minimum_period),
                    "maximum_period_days": float(maximum_period),
                    "minimum_cycles": float(minimum_cycles),
                    "maximum_cycles": float(maximum_cycles),
                }
                for (
                    minimum_period,
                    maximum_period,
                    minimum_cycles,
                    maximum_cycles,
                ) in bands
            ),
        },
    )


def _global_pdm(
    time: np.ndarray,
    mag: np.ndarray,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    n_frequency, resolution_metadata = _resolution_aware_frequency_count(
        config.short_min_period_days,
        config.short_max_period_days,
        baseline_days=baseline,
        requested_n_frequency=config.pdm_n_frequency,
        config=config,
    )
    _, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=n_frequency,
    )
    try:
        if config.pdm_method == "plavchan":
            _, theta = _pdm_scan_grid_plavchan(
                time,
                mag,
                periods,
                float(config.pdm_plavchan_phase_width),
                int(config.pdm_plavchan_min_neighbors),
                0.05,
                min(25, max(3, int(mag.size))),
            )
        else:
            _, theta = _pdm_scan_grid(
                time,
                mag,
                periods,
                int(config.pdm_n_phase_bins),
            )
    except Exception as exc:
        return _empty_search("pdm", "minimize", "error", str(exc))
    return _search_from_grid(
        "pdm",
        "minimize",
        periods,
        np.asarray(theta),
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "pdm_method": config.pdm_method,
            "n_phase_bins": config.pdm_n_phase_bins,
            "grid_kind": "uniform_frequency",
            **resolution_metadata,
        },
    )


def _global_ce(
    time: np.ndarray,
    mag: np.ndarray,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    n_frequency, resolution_metadata = _resolution_aware_frequency_count(
        config.short_min_period_days,
        config.short_max_period_days,
        baseline_days=baseline,
        requested_n_frequency=config.ce_n_frequency,
        config=config,
    )
    _, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=n_frequency,
    )
    try:
        _, entropy = _ce_scan_grid(
            time,
            mag,
            periods,
            int(config.ce_n_phase_bins),
            int(config.ce_n_mag_bins),
        )
    except Exception as exc:
        return _empty_search("ce", "minimize", "error", str(exc))
    return _search_from_grid(
        "ce",
        "minimize",
        periods,
        np.asarray(entropy),
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "n_phase_bins": config.ce_n_phase_bins,
            "n_mag_bins": config.ce_n_mag_bins,
            "grid_kind": "uniform_frequency",
            **resolution_metadata,
        },
    )


def _bls_period_groups(
    periods: np.ndarray,
    *,
    n_groups: int,
    max_period_ratio: float,
) -> list[np.ndarray]:
    """Return sorted index groups with a bounded max/min period ratio."""

    periods = np.asarray(periods, dtype=float)
    order = np.argsort(periods, kind="mergesort")
    if periods.size == 0:
        return []
    sorted_periods = periods[order]
    period_ratio = float(sorted_periods[-1] / sorted_periods[0])
    ratio_groups = (
        int(
            math.ceil(
                math.log(period_ratio) / math.log(float(max_period_ratio))
            )
        )
        if period_ratio > 1.0
        else 1
    )
    group_count = min(
        periods.size,
        max(1, int(n_groups), ratio_groups),
    )
    if group_count == 1:
        return [order]
    edges = np.geomspace(
        float(sorted_periods[0]),
        float(sorted_periods[-1]),
        group_count + 1,
    )
    labels = np.searchsorted(edges[1:-1], sorted_periods, side="right")
    return [
        order[labels == label]
        for label in range(group_count)
        if np.any(labels == label)
    ]


def _bls_grouped_power(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    periods: np.ndarray,
    *,
    duration_fractions: Sequence[float],
    n_groups: int,
    max_period_ratio: float,
    oversample: int,
) -> dict[str, np.ndarray]:
    """Evaluate BLS with approximately phase-scaled durations in period groups."""
    periods = np.asarray(periods, dtype=float)
    groups = _bls_period_groups(
        periods,
        n_groups=n_groups,
        max_period_ratio=max_period_ratio,
    )
    output = {
        name: np.full(periods.size, np.nan, dtype=float)
        for name in (
            "power",
            "duration",
            "transit_time",
            "depth",
            "depth_err",
            "depth_snr",
            "log_likelihood",
        )
    }
    flux = -(mag - float(np.nanmedian(mag)))
    bls = BoxLeastSquares(time, flux, err)
    fractions = np.asarray(duration_fractions, dtype=float)
    for group in groups:
        if group.size == 0:
            continue
        group_periods = periods[group]
        representative = float(np.exp(np.mean(np.log(group_periods))))
        max_duration = 0.8 * float(np.min(group_periods))
        durations = np.unique(np.minimum(representative * fractions, max_duration))
        durations = durations[np.isfinite(durations) & (durations > 0)]
        if durations.size == 0:
            continue
        result = bls.power(
            group_periods,
            durations,
            objective="likelihood",
            oversample=max(1, int(oversample)),
        )
        for name in output:
            if name in result:
                output[name][group] = np.asarray(result[name], dtype=float)
    return output


def _global_bls(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    _, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=config.bls_n_frequency,
    )
    try:
        values = _bls_grouped_power(
            time,
            mag,
            err,
            periods,
            duration_fractions=config.bls_duration_fractions,
            n_groups=config.bls_period_groups,
            max_period_ratio=config.bls_max_period_ratio_per_group,
            oversample=config.bls_oversample,
        )
    except Exception as exc:
        return _empty_search("bls_coarse", "maximize", "error", str(exc))
    result = _search_from_grid(
        "bls_coarse",
        "maximize",
        periods,
        values["power"],
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "grid_kind": "coarse_uniform_frequency",
            "duration_fractions": tuple(config.bls_duration_fractions),
            "period_groups": int(config.bls_period_groups),
        },
    )
    for candidate in result.candidates:
        idx = int(np.nanargmin(np.abs(periods - candidate.period_days)))
        candidate.metadata.update(
            {
                "duration_days": float(values["duration"][idx]),
                "depth": float(values["depth"][idx]),
                "depth_snr": float(values["depth_snr"][idx]),
                "transit_time": float(values["transit_time"][idx]),
            }
        )
    return result


def _global_adaptive_bls(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    """Run a coarse BLS proposal followed by duty-cycle-resolved local scans.

    A blind duration-resolved grid over the full baseline is prohibitively
    large.  This search uses the bounded coarse grid only to identify frequency
    neighborhoods, then resolves each neighborhood finely enough that the
    narrowest configured event does not drift by more than a fraction of its
    duration over the observing baseline.
    """

    frequency, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=config.bls_n_frequency,
    )
    try:
        coarse_values = _bls_grouped_power(
            time,
            mag,
            err,
            periods,
            duration_fractions=config.bls_duration_fractions,
            n_groups=config.bls_period_groups,
            max_period_ratio=config.bls_max_period_ratio_per_group,
            oversample=config.bls_oversample,
        )
    except Exception as exc:
        return _empty_search("bls_adaptive", "maximize", "error", str(exc))
    coarse = _search_from_grid(
        "bls_adaptive",
        "maximize",
        periods,
        coarse_values["power"],
        top_k=int(config.bls_adaptive_seed_top_k),
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="adaptive_coarse_seed",
        metadata={"grid_kind": "coarse_uniform_frequency"},
    )
    if not coarse.candidates:
        return PeriodSearchResult(
            method="bls_adaptive",
            objective="maximize",
            status=coarse.status,
            message=coarse.message,
            metadata={
                **coarse.metadata,
                "search_strategy": "coarse_to_duty_cycle_resolved",
                "n_coarse_seeds": 0,
            },
        )

    coarse_spacing = (
        float(np.max(np.diff(frequency)))
        if frequency.size >= 2
        else 1.0 / float(baseline)
    )
    minimum_duration_fraction = float(min(config.bls_duration_fractions))
    all_periods: list[np.ndarray] = []
    all_power: list[np.ndarray] = []
    all_fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "duration",
            "transit_time",
            "depth",
            "depth_err",
            "depth_snr",
            "log_likelihood",
        )
    }
    seed_metadata: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(coarse.candidates):
        center_frequency = float(seed.frequency_per_day)
        half_width = max(
            float(config.bls_adaptive_refine_rayleigh_half_width)
            / float(baseline),
            1.5 * coarse_spacing,
        )
        low_frequency = max(
            1.0 / float(config.short_max_period_days),
            center_frequency - half_width,
        )
        high_frequency = min(
            1.0 / float(config.short_min_period_days),
            center_frequency + half_width,
        )
        required = (
            int(
                math.ceil(
                    max(0.0, high_frequency - low_frequency)
                    * float(baseline)
                    * float(config.bls_adaptive_refine_frequency_oversample)
                    / minimum_duration_fraction
                )
            )
            + 1
        )
        resolved_n_frequency = min(
            max(3, required),
            int(config.bls_adaptive_refine_max_frequency_points_per_seed),
        )
        local_frequency = np.linspace(
            low_frequency,
            high_frequency,
            resolved_n_frequency,
        )
        local_frequency = np.unique(
            np.append(local_frequency, center_frequency)
        )
        local_periods = 1.0 / local_frequency
        try:
            local_values = _bls_grouped_power(
                time,
                mag,
                err,
                local_periods,
                duration_fractions=config.bls_duration_fractions,
                n_groups=max(1, min(config.bls_period_groups, 3)),
                max_period_ratio=config.bls_max_period_ratio_per_group,
                oversample=config.bls_oversample,
            )
        except Exception:
            continue
        all_periods.append(local_periods)
        all_power.append(local_values["power"])
        for name in all_fields:
            all_fields[name].append(local_values[name])
        seed_metadata.append(
            {
                "seed_index": int(seed_index),
                "seed_period_days": float(seed.period_days),
                "low_frequency_per_day": float(low_frequency),
                "high_frequency_per_day": float(high_frequency),
                "resolved_n_frequency": int(local_frequency.size),
                "resolution_cap_hit": bool(
                    required
                    > int(
                        config.bls_adaptive_refine_max_frequency_points_per_seed
                    )
                ),
            }
        )
    if not all_periods:
        return _empty_search(
            "bls_adaptive",
            "maximize",
            "adaptive_refinement_failed",
            n_coarse_seeds=int(len(coarse.candidates)),
        )

    refined_periods = np.concatenate(all_periods)
    refined_power = np.concatenate(all_power)
    order = np.argsort(1.0 / refined_periods, kind="mergesort")
    refined_periods = refined_periods[order]
    refined_power = refined_power[order]
    refined_fields = {
        name: np.concatenate(values)[order]
        for name, values in all_fields.items()
    }
    result = _search_from_grid(
        "bls_adaptive",
        "maximize",
        refined_periods,
        refined_power,
        top_k=int(config.bls_adaptive_top_k),
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="adaptive_refined",
        metadata={
            "grid_kind": "coarse_seeded_local_frequency",
            "search_strategy": "coarse_to_duty_cycle_resolved",
            "n_coarse_seeds": int(len(coarse.candidates)),
            "n_refined_frequency_points": int(refined_periods.size),
            "minimum_duration_fraction": minimum_duration_fraction,
            "seed_refinements": tuple(seed_metadata),
        },
    )
    for candidate in result.candidates:
        idx = int(
            np.nanargmin(np.abs(refined_periods - candidate.period_days))
        )
        candidate.metadata.update(
            {
                "duration_days": float(refined_fields["duration"][idx]),
                "depth": float(refined_fields["depth"][idx]),
                "depth_snr": float(refined_fields["depth_snr"][idx]),
                "transit_time": float(
                    refined_fields["transit_time"][idx]
                ),
                "adaptive_refinement": True,
            }
        )
    return result


def _multiharmonic_aov_f(power: np.ndarray, n_points: int, nterms: int) -> np.ndarray:
    """Convert standard multiharmonic LS power to its nested-model AoV F statistic.

    This is the regression/multiharmonic analysis-of-variance statistic:

    ``F = ((RSS0 - RSSm) / (2H)) / (RSSm / (N - 2H - 1))``.

    It is a true AoV-style test of the constant model against an H-harmonic
    Fourier model.  For a fixed H it is necessarily monotonic with standard
    multiharmonic LS power; it is kept as a separate, scientifically named
    method so the benchmark can quantify that redundancy explicitly.
    """
    power = np.asarray(power, dtype=float)
    df_model = 2 * int(nterms)
    df_resid = int(n_points) - df_model - 1
    if df_resid <= 0:
        return np.full(power.shape, np.nan, dtype=float)
    clipped = np.clip(power, 0.0, 1.0 - np.finfo(float).eps)
    return (clipped / float(df_model)) / ((1.0 - clipped) / float(df_resid))


def _global_multiharmonic(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
    nterms: int,
    as_aov: bool,
) -> PeriodSearchResult:
    method = f"multiharmonic_{'aov' if as_aov else 'ls'}_{int(nterms)}"
    requested_n_frequency = (
        config.aov_n_frequency if as_aov else config.multiharmonic_n_frequency
    )
    n_frequency, resolution_metadata = _resolution_aware_frequency_count(
        config.short_min_period_days,
        config.short_max_period_days,
        baseline_days=baseline,
        requested_n_frequency=requested_n_frequency,
        config=config,
    )
    frequency, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=n_frequency,
    )
    try:
        ls = LombScargle(time, mag, err, nterms=int(nterms), normalization="standard")
        power = np.asarray(ls.power(frequency, method="chi2"), dtype=float)
        metric = _multiharmonic_aov_f(power, mag.size, nterms) if as_aov else power
    except Exception as exc:
        return _empty_search(method, "maximize", "error", str(exc), nterms=int(nterms))
    return _search_from_grid(
        method,
        "maximize",
        periods,
        metric,
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "nterms": int(nterms),
            "statistic": "nested_fourier_model_f" if as_aov else "standard_ls_power",
            "grid_kind": "uniform_frequency",
            **resolution_metadata,
        },
    )


def _aov_result_from_multiharmonic_ls(
    ls_result: PeriodSearchResult,
    *,
    n_points: int,
    nterms: int,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    """Create the AoV F-statistic view of a cached multiharmonic LS search."""
    method = f"multiharmonic_aov_{int(nterms)}"
    if ls_result.status != "ok":
        return _empty_search(
            method,
            "maximize",
            ls_result.status,
            ls_result.message,
            nterms=int(nterms),
        )
    metric = _multiharmonic_aov_f(ls_result.metric_grid, n_points, nterms)
    return _search_from_grid(
        method,
        "maximize",
        ls_result.period_grid_days,
        metric,
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "nterms": int(nterms),
            "statistic": "nested_fourier_model_f",
            "grid_kind": "uniform_frequency",
            "derived_from": ls_result.method,
            **{
                key: value
                for key, value in ls_result.metadata.items()
                if key
                in {
                    "requested_n_frequency",
                    "resolved_n_frequency",
                    "requested_min_samples_per_rayleigh",
                    "actual_samples_per_rayleigh",
                    "resolution_cap_hit",
                }
            },
        },
    )


def _lafler_kinman_grid(time: np.ndarray, mag: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Evaluate normalized cyclic string length on a period grid."""
    values = np.full(periods.size, np.nan, dtype=float)
    centered = time - float(np.min(time))
    denom = float(np.sum(np.square(mag - np.mean(mag))))
    if not np.isfinite(denom) or denom <= 0:
        return values
    for idx, period in enumerate(periods):
        phase = np.mod(centered / float(period), 1.0)
        order = np.argsort(phase, kind="mergesort")
        folded = mag[order]
        diffs = np.diff(folded, append=folded[0])
        values[idx] = float(np.sum(np.square(diffs)) / denom)
    return values


def _global_lafler_kinman(
    time: np.ndarray,
    mag: np.ndarray,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    n_frequency, resolution_metadata = _resolution_aware_frequency_count(
        config.short_min_period_days,
        config.short_max_period_days,
        baseline_days=baseline,
        requested_n_frequency=config.lafler_kinman_n_frequency,
        config=config,
    )
    _, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=n_frequency,
    )
    try:
        string_length = _lafler_kinman_grid(time, mag, periods)
    except Exception as exc:
        return _empty_search("lafler_kinman", "minimize", "error", str(exc))
    return _search_from_grid(
        "lafler_kinman",
        "minimize",
        periods,
        string_length,
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "statistic": "normalized_cyclic_string_length",
            "grid_kind": "uniform_frequency",
            **resolution_metadata,
        },
    )


def _load_supersmoother() -> tuple[type[Any] | None, str]:
    """Load the optional ``supersmoother`` package without making it required."""
    try:
        module = importlib.import_module("supersmoother")
        smoother = getattr(module, "SuperSmoother")
        return smoother, str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return None, str(exc)


def _supersmoother_cv_error(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    smoother_class: type[Any],
) -> float:
    dy: np.ndarray | float = err if err is not None else 1.0
    smoother = smoother_class(period=float(period))
    smoother.fit(time, mag, dy=dy)
    value = float(smoother.cv_error())
    return value if np.isfinite(value) else np.nan


def _supersmoother_cv_metrics(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    smoother_class: type[Any],
) -> dict[str, float]:
    """Return standardized and magnitude-unit leave-one-out metrics.

    ``supersmoother.cv_residuals`` returns ``(y - prediction) / dy``.  The
    standardized residual is useful for comparing candidate periods, but it
    must be multiplied by the fitted uncertainty vector before comparison with
    the raw magnitude variance.
    """
    dy: np.ndarray | float = err if err is not None else 1.0
    smoother = smoother_class(period=float(period))
    smoother.fit(time, mag, dy=dy)
    standardized = np.asarray(smoother.cv_residuals(), dtype=float)
    fitted_dy = np.asarray(smoother.dy, dtype=float)
    if fitted_dy.ndim == 0:
        fitted_dy = np.full(standardized.size, float(fitted_dy), dtype=float)
    # Match supersmoother.cv_error's endpoint convention while also retaining
    # an MSE that can be compared to the raw magnitude variance.
    if standardized.size > 2:
        standardized = standardized[1:-1]
        fitted_dy = fitted_dy[1:-1]
    finite = (
        np.isfinite(standardized)
        & np.isfinite(fitted_dy)
        & (fitted_dy > 0)
    )
    standardized = standardized[finite]
    magnitude_residual = standardized * fitted_dy[finite]
    if standardized.size == 0:
        return {
            "standardized_mae": np.nan,
            "standardized_mse": np.nan,
            "magnitude_mae": np.nan,
            "magnitude_mse": np.nan,
        }
    return {
        "standardized_mae": float(np.mean(np.abs(standardized))),
        "standardized_mse": float(np.mean(np.square(standardized))),
        "magnitude_mae": float(np.mean(np.abs(magnitude_residual))),
        "magnitude_mse": float(np.mean(np.square(magnitude_residual))),
    }


def _global_supersmoother(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> PeriodSearchResult:
    smoother_class, version_or_error = _load_supersmoother()
    if smoother_class is None:
        return _empty_search(
            "supersmoother",
            "minimize",
            "unavailable",
            f"optional supersmoother dependency unavailable: {version_or_error}",
        )
    _, periods = _frequency_grid(
        config.short_min_period_days,
        config.short_max_period_days,
        n_frequency=config.supersmoother_n_frequency,
    )
    cv_error = np.full(periods.size, np.nan, dtype=float)
    for idx, period in enumerate(periods):
        try:
            cv_error[idx] = _supersmoother_cv_error(
                time,
                mag,
                err,
                float(period),
                smoother_class,
            )
        except Exception:
            continue
    result = _search_from_grid(
        "supersmoother",
        "minimize",
        periods,
        cv_error,
        top_k=config.top_k_general,
        baseline_days=baseline,
        min_rayleigh_separation=config.peak_min_rayleigh_separation,
        search_band="short",
        metadata={
            "statistic": "leave_one_out_cv_error",
            "dependency_version": version_or_error,
            "grid_kind": "coarse_uniform_frequency",
        },
    )
    if not np.isfinite(cv_error).any():
        result.status = "error"
        result.message = "SuperSmoother failed at every trial period"
    return result


def _event_phase_diagnostics(
    event_epochs: np.ndarray,
    period: float,
    *,
    tolerance: float,
    complexity_penalty: float,
    refine_period: bool = True,
) -> dict[str, Any]:
    """Robustly score an event-comb period hypothesis.

    ``refine_period=False`` holds the supplied period fixed and optimizes only
    the epoch offset.  This is the fixed-period evidence used by the ranker.
    The default also refits the ephemeris slope and is used only for explicit
    event-comb proposal/refinement fields.
    """
    epochs = np.asarray(event_epochs, dtype=float)
    epochs = np.sort(np.unique(epochs[np.isfinite(epochs)]))
    if epochs.size < 2 or not np.isfinite(period) or period <= 0:
        return {
            "period_days": np.nan,
            "score": np.nan,
            "phase_concentration": np.nan,
            "inlier_fraction": np.nan,
            "median_abs_oc_days": np.nan,
            "rms_oc_days": np.nan,
            "max_cycle": 0,
            "cycle_numbers": (),
        }

    current_period = float(period)
    epoch0 = float(epochs[0])
    cycles = np.rint((epochs - epoch0) / current_period).astype(np.int64)
    for _ in range(4):
        if np.unique(cycles).size < 2:
            break
        if refine_period:
            design = np.column_stack(
                (np.ones(cycles.size), cycles.astype(float))
            )
            try:
                coeff, _, _, _ = np.linalg.lstsq(
                    design, epochs, rcond=None
                )
            except np.linalg.LinAlgError:
                break
            fitted_period = float(coeff[1])
            if not np.isfinite(fitted_period) or fitted_period <= 0:
                break
            current_period = fitted_period
            epoch0 = float(coeff[0])
        else:
            epoch0 = float(
                np.median(epochs - cycles.astype(float) * current_period)
            )
        cycles = np.rint((epochs - epoch0) / current_period).astype(np.int64)

    model = epoch0 + cycles * current_period
    residual = epochs - model
    phase_residual = np.abs(residual) / current_period
    inlier = phase_residual <= float(tolerance)
    if int(np.count_nonzero(inlier)) >= 2 and not np.all(inlier):
        if refine_period:
            design = np.column_stack(
                (
                    np.ones(np.count_nonzero(inlier)),
                    cycles[inlier].astype(float),
                )
            )
            try:
                coeff, _, _, _ = np.linalg.lstsq(
                    design, epochs[inlier], rcond=None
                )
                if np.isfinite(coeff[1]) and coeff[1] > 0:
                    epoch0 = float(coeff[0])
                    current_period = float(coeff[1])
            except np.linalg.LinAlgError:
                pass
        else:
            epoch0 = float(
                np.median(
                    epochs[inlier]
                    - cycles[inlier].astype(float) * current_period
                )
            )
        cycles = np.rint((epochs - epoch0) / current_period).astype(np.int64)
        model = epoch0 + cycles * current_period
        residual = epochs - model
        phase_residual = np.abs(residual) / current_period
        inlier = phase_residual <= float(tolerance)

    angles = 2.0 * np.pi * np.mod((epochs - epoch0) / current_period, 1.0)
    concentration = float(np.abs(np.mean(np.exp(1j * angles))))
    inlier_fraction = float(np.mean(inlier))
    median_abs = float(np.median(np.abs(residual[inlier]))) if np.any(inlier) else np.nan
    rms = float(np.sqrt(np.mean(np.square(residual[inlier])))) if np.any(inlier) else np.nan
    cycle_span = int(np.ptp(cycles)) if cycles.size else 0
    timing_loss = (
        float(median_abs / current_period)
        if np.isfinite(median_abs) and current_period > 0
        else 1.0
    )
    # A smaller candidate period uses more integer cycles to explain the same
    # event span.  Penalize that flexibility gently; otherwise every dt/k
    # subharmonic can fit two events perfectly and a floating-point-GCD failure
    # simply reappears under another name.
    complexity = math.log1p(max(0, cycle_span))
    score = (
        inlier_fraction
        + concentration
        - timing_loss
        - float(complexity_penalty) * complexity
    )
    return {
        "period_days": current_period,
        "score": float(score),
        "phase_concentration": concentration,
        "inlier_fraction": inlier_fraction,
        "median_abs_oc_days": median_abs,
        "rms_oc_days": rms,
        "max_cycle": cycle_span,
        "cycle_numbers": tuple(int(value) for value in cycles),
    }


def event_epoch_detection_diagnostics(
    detected_epochs: Sequence[float] | None,
    injected_epochs: Sequence[float] | None,
    *,
    match_tolerance_days: float,
) -> dict[str, Any]:
    """Return one-to-one event-epoch precision, recall, and timing residuals.

    Matching is greedy in increasing absolute epoch separation.  Each detected
    and injected event can be used at most once, preventing one broad
    detection from claiming several injected events.
    """

    tolerance = float(match_tolerance_days)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("match_tolerance_days must be finite and positive")
    detected = (
        np.empty(0, dtype=float)
        if detected_epochs is None
        else np.sort(
            np.unique(
                np.asarray(detected_epochs, dtype=float)[
                    np.isfinite(np.asarray(detected_epochs, dtype=float))
                ]
            )
        )
    )
    injected = (
        np.empty(0, dtype=float)
        if injected_epochs is None
        else np.sort(
            np.unique(
                np.asarray(injected_epochs, dtype=float)[
                    np.isfinite(np.asarray(injected_epochs, dtype=float))
                ]
            )
        )
    )
    possible: list[tuple[float, int, int]] = []
    for detected_index, detected_epoch in enumerate(detected):
        separation = np.abs(injected - detected_epoch)
        for injected_index in np.flatnonzero(separation <= tolerance):
            possible.append(
                (
                    float(separation[injected_index]),
                    int(detected_index),
                    int(injected_index),
                )
            )
    possible.sort()
    used_detected: set[int] = set()
    used_injected: set[int] = set()
    residuals: list[float] = []
    for _, detected_index, injected_index in possible:
        if detected_index in used_detected or injected_index in used_injected:
            continue
        used_detected.add(detected_index)
        used_injected.add(injected_index)
        residuals.append(
            float(detected[detected_index] - injected[injected_index])
        )
    residual = np.asarray(residuals, dtype=float)
    matched = int(residual.size)
    precision = (
        float(matched / detected.size) if detected.size else np.nan
    )
    recall = float(matched / injected.size) if injected.size else np.nan
    return {
        "event_detection_status": (
            "ok"
            if injected.size
            else ("no_injected_events" if detected.size else "no_events")
        ),
        "event_detection_match_tolerance_days": tolerance,
        "event_detection_injected_count": int(injected.size),
        "event_detection_detected_count": int(detected.size),
        "event_detection_matched_count": matched,
        "event_detection_precision": precision,
        "event_detection_recall": recall,
        "event_detection_false_event_fraction": (
            float(1.0 - precision) if np.isfinite(precision) else np.nan
        ),
        "event_detection_missed_event_fraction": (
            float(1.0 - recall) if np.isfinite(recall) else np.nan
        ),
        "event_detection_epoch_bias_days": (
            float(np.mean(residual)) if residual.size else np.nan
        ),
        "event_detection_epoch_median_abs_error_days": (
            float(np.median(np.abs(residual))) if residual.size else np.nan
        ),
        "event_detection_epoch_rms_days": (
            float(np.sqrt(np.mean(np.square(residual))))
            if residual.size
            else np.nan
        ),
    }


def robust_event_comb_candidates(
    event_epochs: Sequence[float] | None,
    *,
    min_period_days: float,
    max_period_days: float,
    baseline_days: float,
    config: PeriodCandidateMethodsConfig | None = None,
) -> PeriodSearchResult:
    """Propose robust event-spacing periods from ``dt/k`` cycle hypotheses.

    Missing events are represented by integer cycle counts greater than one.
    Timing jitter and event outliers are handled by iterative ephemeris fitting
    and a phase-residual inlier rule.  With only two events the returned
    solutions remain intrinsically cycle-count ambiguous; that ambiguity is
    recorded in candidate metadata rather than hidden by a floating-point GCD.
    """
    cfg = config or PeriodCandidateMethodsConfig()
    if event_epochs is None:
        return _empty_search("event_comb", "maximize", "no_events")
    epochs_all = np.asarray(event_epochs, dtype=float)
    epochs_all = np.sort(np.unique(epochs_all[np.isfinite(epochs_all)]))
    n_events_input = int(epochs_all.size)
    if epochs_all.size < 2:
        return _empty_search(
            "event_comb",
            "maximize",
            "insufficient_events",
            n_events=n_events_input,
        )

    epochs_truncated = bool(epochs_all.size > int(cfg.event_max_epochs))
    if epochs_truncated:
        maximum_epochs = int(cfg.event_max_epochs)
        # Preserve both the full temporal span and local consecutive spacings.
        # Pure uniform index thinning turns a dense P-spaced train into an
        # apparently (k*P)-spaced train and can erase the fundamental.
        n_uniform = max(2, maximum_epochs // 2)
        uniform_indices = np.rint(
            np.linspace(0, epochs_all.size - 1, n_uniform)
        ).astype(int)
        n_adjacent_pairs = max(1, (maximum_epochs - n_uniform) // 2)
        pair_left = np.rint(
            np.linspace(0, epochs_all.size - 2, n_adjacent_pairs)
        ).astype(int)
        indices = np.unique(
            np.concatenate((uniform_indices, pair_left, pair_left + 1))
        )
        if indices.size > maximum_epochs:
            indices = indices[:maximum_epochs]
        epochs = epochs_all[indices]
    else:
        epochs = epochs_all

    # Preserve robust hypotheses from the unthinned consecutive spacings.
    # These anchors protect the fundamental when the coverage-oriented epoch
    # subset necessarily skips many events.
    priority_seeds: list[float] = []
    consecutive_all = np.diff(epochs_all)
    consecutive_all = consecutive_all[np.isfinite(consecutive_all) & (consecutive_all > 0)]
    if consecutive_all.size:
        spacing_summaries = np.quantile(
            consecutive_all,
            [0.50, 0.25, 0.75, 0.10, 0.90, 0.0, 1.0],
        )
        for spacing in spacing_summaries:
            max_divisor = min(
                int(cfg.event_max_cycle_divisor),
                max(1, int(math.floor(float(spacing) / float(min_period_days)))),
            )
            for divisor in range(1, max_divisor + 1):
                seed = float(spacing) / float(divisor)
                if min_period_days <= seed <= max_period_days:
                    priority_seeds.append(seed)

    seeds: list[float] = list(priority_seeds)
    pairs_considered = 0
    full_pair_count = int(epochs_all.size * (epochs_all.size - 1) // 2)
    seed_subset_pair_count = int(epochs.size * (epochs.size - 1) // 2)
    for left in range(epochs.size - 1):
        right_stop = min(
            int(epochs.size),
            left + 1 + int(cfg.event_max_pair_lags),
        )
        for right in range(left + 1, right_stop):
            pairs_considered += 1
            delta = float(epochs[right] - epochs[left])
            if delta <= 0:
                continue
            max_divisor = min(
                int(cfg.event_max_cycle_divisor),
                max(1, int(math.floor(delta / float(min_period_days)))),
            )
            for divisor in range(1, max_divisor + 1):
                seed = delta / float(divisor)
                if min_period_days <= seed <= max_period_days:
                    seeds.append(seed)
    if not seeds:
        return _empty_search(
            "event_comb",
            "maximize",
            "no_in_bounds_hypotheses",
            n_events_input=n_events_input,
            n_events_used=int(epochs.size),
            epochs_truncated=epochs_truncated,
            pairs_considered=pairs_considered,
        )

    raw_seed_count = int(len(seeds))
    # Sorted-adjacent relative de-duplication is O(S log S), unlike comparing
    # every new seed to every previously retained seed.
    unique_seeds: list[float] = []
    for seed in sorted(seeds, reverse=True):
        if unique_seeds and (
            abs(seed - unique_seeds[-1]) / max(seed, unique_seeds[-1])
            < cfg.event_seed_relative_tolerance
        ):
            continue
        unique_seeds.append(float(seed))

    unique_seed_count = int(len(unique_seeds))
    seeds_truncated = bool(unique_seed_count > int(cfg.event_max_seed_hypotheses))
    if seeds_truncated:
        # Preserve uniform frequency coverage rather than simply retaining the
        # first (longest-period) hypotheses.
        seed_frequency = np.sort(1.0 / np.asarray(unique_seeds, dtype=float))
        priority_unique: list[float] = []
        for seed in priority_seeds:
            if priority_unique and any(
                abs(seed - previous) / max(seed, previous)
                < cfg.event_seed_relative_tolerance
                for previous in priority_unique
            ):
                continue
            priority_unique.append(float(seed))
        priority_unique = priority_unique[: int(cfg.event_max_seed_hypotheses)]
        remaining = max(
            0,
            int(cfg.event_max_seed_hypotheses) - len(priority_unique),
        )
        target_frequency = np.geomspace(
            float(seed_frequency[0]),
            float(seed_frequency[-1]),
            max(1, remaining),
        )
        positions = np.searchsorted(seed_frequency, target_frequency)
        positions = np.clip(positions, 0, seed_frequency.size - 1)
        left_positions = np.maximum(positions - 1, 0)
        choose_left = (
            np.abs(seed_frequency[left_positions] - target_frequency)
            <= np.abs(seed_frequency[positions] - target_frequency)
        )
        positions[choose_left] = left_positions[choose_left]
        selected_frequency = seed_frequency[np.unique(positions)] if remaining else np.empty(0)
        selected_seeds = [float(1.0 / value) for value in selected_frequency]
        unique_seeds = list(priority_unique)
        for seed in selected_seeds:
            if any(
                abs(seed - previous) / max(seed, previous)
                < cfg.event_seed_relative_tolerance
                for previous in unique_seeds
            ):
                continue
            unique_seeds.append(seed)
            if len(unique_seeds) >= int(cfg.event_max_seed_hypotheses):
                break

    event_metadata = {
        "n_events": n_events_input,
        "n_events_input": n_events_input,
        "n_events_used": n_events_input,
        "n_events_seed_subset": int(epochs.size),
        "epochs_truncated": epochs_truncated,
        "full_pair_count": full_pair_count,
        "seed_subset_pair_count": seed_subset_pair_count,
        "pairs_considered": int(pairs_considered),
        "pair_lags_truncated": bool(
            epochs_truncated or pairs_considered < seed_subset_pair_count
        ),
        "n_raw_seed_hypotheses": raw_seed_count,
        "n_unique_seed_hypotheses": unique_seed_count,
        "seed_hypotheses_truncated": seeds_truncated,
        "n_evaluated_seed_hypotheses": int(len(unique_seeds)),
        "rescored_on_all_events": True,
    }
    seed_diagnostics = [
        _event_phase_diagnostics(
            epochs,
            seed,
            tolerance=cfg.event_phase_tolerance,
            complexity_penalty=cfg.event_cycle_complexity_penalty,
        )
        for seed in unique_seeds
    ]
    seed_diagnostics = [
        values
        for values in seed_diagnostics
        if np.isfinite(values["period_days"])
        and min_period_days <= values["period_days"] <= max_period_days
        and np.isfinite(values["score"])
    ]
    diagnostics: list[dict[str, Any]] = []
    for seed_values in seed_diagnostics:
        values = _event_phase_diagnostics(
            epochs_all,
            float(seed_values["period_days"]),
            tolerance=cfg.event_phase_tolerance,
            complexity_penalty=cfg.event_cycle_complexity_penalty,
        )
        values["seed_subset_score"] = float(seed_values["score"])
        values["seed_subset_inlier_fraction"] = float(
            seed_values["inlier_fraction"]
        )
        if (
            np.isfinite(values["period_days"])
            and min_period_days <= values["period_days"] <= max_period_days
            and np.isfinite(values["score"])
            and values["inlier_fraction"] >= cfg.event_min_inlier_fraction
        ):
            diagnostics.append(values)
    if not diagnostics:
        return _empty_search(
            "event_comb",
            "maximize",
            "no_coherent_hypotheses",
            **event_metadata,
        )

    # De-duplicate refined hypotheses in physical frequency and keep the best
    # score within each unresolved group.
    diagnostics.sort(
        key=lambda values: (
            -float(values["score"]),
            -float(values["period_days"]),
        )
    )
    min_df = (
        cfg.peak_min_rayleigh_separation / float(baseline_days)
        if np.isfinite(baseline_days) and baseline_days > 0
        else 0.0
    )
    kept: list[dict[str, Any]] = []
    for values in diagnostics:
        frequency = 1.0 / float(values["period_days"])
        if any(abs(frequency - 1.0 / float(previous["period_days"])) < min_df for previous in kept):
            continue
        kept.append(values)
        if len(kept) >= int(cfg.top_k_event):
            break

    scores = np.asarray([values["score"] for values in kept], dtype=float)
    candidates = []
    for rank, values in enumerate(kept, start=1):
        period = float(values["period_days"])
        candidates.append(
            PeriodCandidate(
                method="event_comb",
                period_days=period,
                frequency_per_day=1.0 / period,
                raw_score=float(values["score"]),
                objective="maximize",
                rank=rank,
                prominence=np.nan,
                normalized_score=_metric_normalized_score(
                    float(values["score"]),
                    scores,
                    True,
                ),
                search_band="event",
                contributing_methods=("event_comb",),
                metadata={
                    key: value
                    for key, value in values.items()
                    if key not in {"period_days", "score"}
                }
                | {
                    **event_metadata,
                    "two_event_cycle_count_ambiguous": bool(n_events_input == 2),
                },
            )
        )
    return PeriodSearchResult(
        method="event_comb",
        objective="maximize",
        candidates=candidates,
        period_grid_days=np.asarray([values["period_days"] for values in diagnostics]),
        metric_grid=np.asarray([values["score"] for values in diagnostics]),
        status="ok" if candidates else "no_coherent_hypotheses",
        metadata={
            **event_metadata,
            "two_event_cycle_count_ambiguous": bool(n_events_input == 2),
        },
    )


def run_global_period_searches(
    time: Sequence[float],
    mag: Sequence[float],
    err: Sequence[float] | None = None,
    *,
    event_epochs: Sequence[float] | None = None,
    config: PeriodCandidateMethodsConfig | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, PeriodSearchResult]:
    """Run selected independent global searches without bootstrapping.

    ``methods=("event_comb",)`` is useful for applying multiple event views to
    one cached set of photometric searches.
    """
    cfg = config or PeriodCandidateMethodsConfig()
    selected = tuple(cfg.enabled_global_methods if methods is None else methods)
    unknown = sorted(set(selected) - set(GLOBAL_METHODS))
    if unknown:
        raise ValueError(f"Unknown global period methods: {unknown}")
    time_arr, mag_arr, err_arr = _prepare_light_curve(time, mag, err)
    if time_arr.size < int(cfg.min_points):
        return {
            method: _empty_search(
                method,
                "minimize" if method in {"pdm", "ce", "lafler_kinman", "supersmoother"} else "maximize",
                "insufficient_points",
                n_points=int(time_arr.size),
            )
            for method in selected
        }
    baseline = float(np.ptp(time_arr))
    if not np.isfinite(baseline) or baseline <= 0:
        return {
            method: _empty_search(
                method,
                "minimize" if method in {"pdm", "ce", "lafler_kinman", "supersmoother"} else "maximize",
                "zero_baseline",
            )
            for method in selected
        }

    long_max = min(
        float(cfg.long_absolute_max_period_days),
        float(cfg.long_max_baseline_fraction) * baseline,
    )
    results: dict[str, PeriodSearchResult] = {}
    multiharmonic_ls_cache: dict[int, PeriodSearchResult] = {}

    def multiharmonic_ls_result(nterms: int) -> PeriodSearchResult:
        if nterms not in multiharmonic_ls_cache:
            multiharmonic_ls_cache[nterms] = _global_multiharmonic(
                time_arr,
                mag_arr,
                err_arr,
                baseline=baseline,
                config=cfg,
                nterms=nterms,
                as_aov=False,
            )
        return multiharmonic_ls_cache[nterms]

    for method in selected:
        if method == "ls_short":
            results[method] = _global_ls(
                time_arr,
                mag_arr,
                err_arr,
                method=method,
                min_period=cfg.short_min_period_days,
                max_period=cfg.short_max_period_days,
                top_k=cfg.top_k_ls,
                baseline=baseline,
                config=cfg,
                search_band="short",
            )
        elif method == "ls_long":
            results[method] = _global_cycle_banded_ls(
                time_arr,
                mag_arr,
                err_arr,
                method=method,
                baseline=baseline,
                long_max_period=long_max,
                config=cfg,
                nterms=1,
            )
        elif method.startswith("multiharmonic_ls_long_"):
            nterms = int(method.rsplit("_", 1)[-1])
            results[method] = _global_cycle_banded_ls(
                time_arr,
                mag_arr,
                err_arr,
                method=method,
                baseline=baseline,
                long_max_period=long_max,
                config=cfg,
                nterms=nterms,
            )
        elif method == "pdm":
            results[method] = _global_pdm(time_arr, mag_arr, baseline=baseline, config=cfg)
        elif method == "ce":
            results[method] = _global_ce(time_arr, mag_arr, baseline=baseline, config=cfg)
        elif method == "bls_coarse":
            results[method] = _global_bls(
                time_arr,
                mag_arr,
                err_arr,
                baseline=baseline,
                config=cfg,
            )
        elif method == "bls_adaptive":
            results[method] = _global_adaptive_bls(
                time_arr,
                mag_arr,
                err_arr,
                baseline=baseline,
                config=cfg,
            )
        elif method.startswith("multiharmonic_ls_"):
            nterms = int(method.rsplit("_", 1)[-1])
            results[method] = multiharmonic_ls_result(nterms)
        elif method.startswith("multiharmonic_aov_"):
            nterms = int(method.rsplit("_", 1)[-1])
            if int(cfg.aov_n_frequency) == int(cfg.multiharmonic_n_frequency):
                results[method] = _aov_result_from_multiharmonic_ls(
                    multiharmonic_ls_result(nterms),
                    n_points=int(mag_arr.size),
                    nterms=nterms,
                    baseline=baseline,
                    config=cfg,
                )
            else:
                results[method] = _global_multiharmonic(
                    time_arr,
                    mag_arr,
                    err_arr,
                    baseline=baseline,
                    config=cfg,
                    nterms=nterms,
                    as_aov=True,
                )
        elif method == "lafler_kinman":
            results[method] = _global_lafler_kinman(
                time_arr,
                mag_arr,
                baseline=baseline,
                config=cfg,
            )
        elif method == "supersmoother":
            results[method] = _global_supersmoother(
                time_arr,
                mag_arr,
                err_arr,
                baseline=baseline,
                config=cfg,
            )
        elif method == "event_comb":
            results[method] = robust_event_comb_candidates(
                event_epochs,
                min_period_days=cfg.short_min_period_days,
                max_period_days=min(
                    cfg.long_absolute_max_period_days,
                    cfg.candidate_max_baseline_fraction * baseline,
                ),
                baseline_days=baseline,
                config=cfg,
            )
    return results


def merge_search_results(
    *result_maps: Mapping[str, PeriodSearchResult],
) -> dict[str, PeriodSearchResult]:
    """Merge cached search maps, rejecting conflicting duplicate method keys."""
    merged: dict[str, PeriodSearchResult] = {}
    for result_map in result_maps:
        for method, result in result_map.items():
            if method in merged and merged[method] is not result:
                raise ValueError(f"Conflicting duplicate search result for {method!r}")
            merged[method] = result
    return merged


def _candidate_priority(
    candidate: PeriodCandidate,
) -> tuple[float, int, int, float, str, float]:
    normalized = (
        float(candidate.normalized_score)
        if np.isfinite(candidate.normalized_score)
        else -np.inf
    )
    is_expanded = 0 if np.isclose(candidate.harmonic_factor, 1.0) else 1
    harmonic_distance = abs(math.log(float(candidate.harmonic_factor)))
    return (
        -normalized,
        int(candidate.rank),
        is_expanded,
        harmonic_distance,
        str(candidate.method),
        -float(candidate.period_days),
    )


def _merge_candidates(
    candidates: Sequence[PeriodCandidate],
    *,
    baseline_days: float,
    config: PeriodCandidateMethodsConfig,
) -> list[PeriodCandidate]:
    """Merge unresolved frequencies while preserving all method provenance."""
    if not candidates:
        return []
    rayleigh = (
        float(config.merge_rayleigh_separation) / float(baseline_days)
        if np.isfinite(baseline_days) and baseline_days > 0
        else 0.0
    )
    groups: list[list[PeriodCandidate]] = []
    for candidate in sorted(candidates, key=_candidate_priority):
        frequency = float(candidate.frequency_per_day)
        matched: list[PeriodCandidate] | None = None
        for group in groups:
            representative = group[0]
            tolerance = max(
                rayleigh,
                config.merge_relative_frequency_tolerance
                * max(abs(frequency), abs(representative.frequency_per_day)),
            )
            if abs(frequency - representative.frequency_per_day) <= tolerance:
                matched = group
                break
        if matched is None:
            groups.append([candidate])
        else:
            matched.append(candidate)

    merged: list[PeriodCandidate] = []
    for group in groups:
        group.sort(key=_candidate_priority)
        representative = group[0]
        methods = tuple(sorted({method for item in group for method in item.contributing_methods}))
        factors = sorted({float(item.harmonic_factor) for item in group})
        parents = sorted(
            {
                float(item.parent_period_days)
                for item in group
                if np.isfinite(item.parent_period_days)
            }
        )
        merged.append(
            PeriodCandidate(
                method=representative.method,
                period_days=representative.period_days,
                frequency_per_day=representative.frequency_per_day,
                raw_score=representative.raw_score,
                objective=representative.objective,
                rank=representative.rank,
                prominence=representative.prominence,
                normalized_score=representative.normalized_score,
                search_band=representative.search_band,
                parent_period_days=representative.parent_period_days,
                harmonic_factor=representative.harmonic_factor,
                contributing_methods=methods,
                metadata={
                    **representative.metadata,
                    "merged_candidate_count": len(group),
                    "merged_methods": methods,
                    "merged_harmonic_factors": tuple(factors),
                    "merged_parent_periods_days": tuple(parents),
                },
            )
        )
    return sorted(merged, key=_candidate_priority)


def expand_harmonic_candidates(
    candidates: Sequence[PeriodCandidate],
    *,
    baseline_days: float,
    config: PeriodCandidateMethodsConfig | None = None,
) -> list[PeriodCandidate]:
    """Expand candidates by configured harmonic factors and merge duplicates."""
    cfg = config or PeriodCandidateMethodsConfig()
    maximum = min(
        cfg.long_absolute_max_period_days,
        cfg.candidate_max_baseline_fraction * float(baseline_days),
    )
    expanded: list[PeriodCandidate] = []
    for candidate in candidates:
        for factor in cfg.harmonic_factors:
            period = float(candidate.period_days) * float(factor)
            if not cfg.short_min_period_days <= period <= maximum:
                continue
            expanded.append(
                PeriodCandidate(
                    method=candidate.method if np.isclose(factor, 1.0) else "harmonic_expansion",
                    period_days=period,
                    frequency_per_day=1.0 / period,
                    raw_score=candidate.raw_score,
                    objective=candidate.objective,
                    rank=candidate.rank,
                    prominence=candidate.prominence,
                    normalized_score=candidate.normalized_score,
                    search_band=candidate.search_band,
                    parent_period_days=float(candidate.period_days),
                    harmonic_factor=float(factor),
                    contributing_methods=candidate.contributing_methods,
                    metadata={
                        **candidate.metadata,
                        "seed_method": candidate.method,
                        "seed_period_days": float(candidate.period_days),
                    },
                )
            )
    return _merge_candidates(expanded, baseline_days=baseline_days, config=cfg)


def build_candidate_bank(
    search_results: Mapping[str, PeriodSearchResult],
    *,
    baseline_days: float,
    config: PeriodCandidateMethodsConfig | None = None,
    expand_harmonics: bool = True,
) -> list[PeriodCandidate]:
    """Merge all successful global proposals and optionally expand harmonics."""
    cfg = config or PeriodCandidateMethodsConfig()
    raw = [
        candidate
        for result in search_results.values()
        for candidate in result.candidates
        if result.status == "ok"
    ]
    merged = _merge_candidates(raw, baseline_days=baseline_days, config=cfg)
    if not expand_harmonics:
        return merged
    return expand_harmonic_candidates(merged, baseline_days=baseline_days, config=cfg)


def _weighted_fourier_score(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    order: int,
) -> dict[str, float]:
    phase = np.mod(time / float(period), 1.0)
    columns = [np.ones(phase.size, dtype=float)]
    for harmonic in range(1, int(order) + 1):
        angle = 2.0 * np.pi * harmonic * phase
        columns.extend((np.cos(angle), np.sin(angle)))
    design = np.column_stack(columns)
    if err is None:
        sqrt_weight = np.ones(mag.size, dtype=float)
    else:
        sqrt_weight = 1.0 / err
    weighted_design = design * sqrt_weight[:, None]
    weighted_mag = mag * sqrt_weight
    try:
        coeff, _, _, _ = np.linalg.lstsq(weighted_design, weighted_mag, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "power": np.nan,
            "aov_f": np.nan,
            "rss": np.nan,
            "bic": np.nan,
            "amplitude": np.nan,
        }
    fitted = design @ coeff
    residual = (mag - fitted) * sqrt_weight
    rss = float(np.sum(np.square(residual)))
    constant = np.average(mag, weights=np.square(sqrt_weight))
    rss0 = float(np.sum(np.square((mag - constant) * sqrt_weight)))
    power = float(1.0 - rss / rss0) if rss0 > 0 else np.nan
    aov_f = float(
        _multiharmonic_aov_f(np.asarray([power]), mag.size, int(order))[0]
    ) if np.isfinite(power) else np.nan
    n_parameters = design.shape[1]
    bic = (
        float(mag.size * np.log(max(rss / mag.size, np.finfo(float).tiny)) + n_parameters * np.log(mag.size))
        if mag.size > n_parameters
        else np.nan
    )
    grid_phase = np.linspace(0.0, 1.0, 512, endpoint=False)
    grid_columns = [np.ones(grid_phase.size, dtype=float)]
    for harmonic in range(1, int(order) + 1):
        angle = 2.0 * np.pi * harmonic * grid_phase
        grid_columns.extend((np.cos(angle), np.sin(angle)))
    model_grid = np.column_stack(grid_columns) @ coeff
    amplitude = float(np.ptp(model_grid))
    return {
        "power": power,
        "aov_f": aov_f,
        "rss": rss,
        "bic": bic,
        "amplitude": amplitude,
    }


def _bounded_local_frequency_grid(
    period: float,
    *,
    baseline: float,
    rayleigh_half_width: float,
    relative_half_width: float,
    n_frequency: int,
    config: PeriodCandidateMethodsConfig,
) -> tuple[np.ndarray, float]:
    """Return a local grid clipped to the candidate bank's period bounds."""
    center_frequency = 1.0 / float(period)
    requested_half_width = max(
        float(rayleigh_half_width) / float(baseline),
        float(relative_half_width) * center_frequency,
    )
    minimum_period = float(config.short_min_period_days)
    maximum_period = min(
        float(config.long_absolute_max_period_days),
        float(config.candidate_max_baseline_fraction) * float(baseline),
    )
    minimum_frequency = 1.0 / maximum_period
    maximum_frequency = 1.0 / minimum_period
    low_frequency = max(minimum_frequency, center_frequency - requested_half_width)
    high_frequency = min(maximum_frequency, center_frequency + requested_half_width)
    if high_frequency < low_frequency:
        low_frequency = high_frequency = float(
            np.clip(center_frequency, minimum_frequency, maximum_frequency)
        )
    grid = np.linspace(
        low_frequency,
        high_frequency,
        max(3, int(n_frequency)),
    )
    # Preserve the exact seed score even when clipping makes the grid
    # asymmetric near the long- or short-period boundary.
    if minimum_frequency <= center_frequency <= maximum_frequency:
        grid = np.unique(np.append(grid, center_frequency))
    return grid, requested_half_width


def _fixed_ls_features(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> dict[str, Any]:
    """Evaluate exact LS power and refine locally in physical frequency."""
    center_frequency = 1.0 / float(period)
    frequency, half_width = _bounded_local_frequency_grid(
        period,
        baseline=baseline,
        rayleigh_half_width=config.fixed_ls_refine_rayleigh_half_width,
        relative_half_width=config.fixed_ls_refine_relative_half_width,
        n_frequency=config.fixed_ls_refine_n_frequency,
        config=config,
    )
    try:
        ls = LombScargle(time, mag, err, normalization="standard")
        power = np.asarray(ls.power(frequency), dtype=float)
    except Exception as exc:
        return {
            "ls_power": np.nan,
            "ls_local_best_power": np.nan,
            "ls_local_best_period_days": np.nan,
            "ls_local_relative_shift": np.nan,
            "ls_status": "error",
            "ls_error": str(exc),
        }
    if power.size == 0 or not np.isfinite(power).any():
        return {
            "ls_power": np.nan,
            "ls_local_best_power": np.nan,
            "ls_local_best_period_days": np.nan,
            "ls_local_relative_shift": np.nan,
            "ls_status": "no_finite_power",
        }
    exact_idx = int(np.nanargmin(np.abs(frequency - center_frequency)))
    best_idx = int(np.nanargmax(power))
    best_period = float(1.0 / frequency[best_idx])
    return {
        "ls_power": float(power[exact_idx]),
        "ls_local_best_power": float(power[best_idx]),
        "ls_local_best_period_days": best_period,
        "ls_local_relative_shift": float((best_period - period) / period),
        "ls_local_frequency_half_width": float(half_width),
        "ls_status": "ok",
    }


def _alias_features(
    period: float,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> dict[str, Any]:
    """Return nearest configured sampling-alias distances and flags."""
    aliases = np.asarray(config.alias_periods_days, dtype=float)
    if aliases.size == 0:
        return {
            "alias_status": "disabled_no_aliases",
            "alias_nearest_period_days": np.nan,
            "alias_frequency_distance_per_day": np.nan,
            "alias_relative_period_distance": np.nan,
            "alias_rayleigh_distance": np.nan,
            "alias_within_resolution": 0,
            "alias_within_relative_tolerance": 0,
        }
    frequency_distance = np.abs(1.0 / aliases - 1.0 / float(period))
    nearest_idx = int(np.nanargmin(frequency_distance))
    nearest = float(aliases[nearest_idx])
    delta_frequency = float(frequency_distance[nearest_idx])
    relative_period = float(abs(period - nearest) / nearest)
    rayleigh_distance = float(delta_frequency * baseline)
    return {
        "alias_status": "ok",
        "alias_nearest_period_days": nearest,
        "alias_nearest_index": nearest_idx,
        "alias_frequency_distance_per_day": delta_frequency,
        "alias_relative_period_distance": relative_period,
        "alias_rayleigh_distance": rayleigh_distance,
        "alias_within_resolution": int(
            rayleigh_distance <= float(config.alias_rayleigh_tolerance)
        ),
        "alias_within_relative_tolerance": int(
            relative_period <= float(config.alias_relative_tolerance)
        ),
    }


def _fixed_bls_features(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> dict[str, Any]:
    """Return BLS evidence at exactly the supplied candidate period.

    Local BLS refinement is intentionally deferred to
    :func:`refine_scored_candidates`, where only the strongest BLS seeds pay
    for a duty-cycle-resolved frequency grid.  This keeps fixed-period feature
    provenance unambiguous and prevents thousands of BLS evaluations for
    every candidate in the scoring bank.
    """

    periods = np.asarray([float(period)], dtype=float)
    try:
        values = _bls_grouped_power(
            time,
            mag,
            err,
            periods,
            duration_fractions=config.bls_duration_fractions,
            n_groups=1,
            max_period_ratio=config.bls_max_period_ratio_per_group,
            oversample=config.bls_oversample,
        )
    except Exception as exc:
        return {"bls_status": "error", "bls_error": str(exc)}
    finite = np.isfinite(values["power"])
    if not finite.any():
        return {"bls_status": "no_finite_power"}
    exact_idx = 0
    exact_duration = float(values["duration"][exact_idx])
    exact_depth = float(values["depth"][exact_idx])
    exact_depth_err = float(values["depth_err"][exact_idx])
    exact_depth_snr = float(values["depth_snr"][exact_idx])
    exact_log_likelihood = float(values["log_likelihood"][exact_idx])
    exact_transit_time = float(values["transit_time"][exact_idx])
    exact_power = float(values["power"][exact_idx])
    return {
        "bls_status": "ok",
        "bls_power": exact_power,
        "bls_exact_power": exact_power,
        "bls_exact_duration_days": exact_duration,
        "bls_exact_duration_phase_fraction": exact_duration / float(period),
        "bls_exact_transit_time": exact_transit_time,
        "bls_exact_depth": exact_depth,
        "bls_exact_depth_err": exact_depth_err,
        "bls_exact_depth_snr": exact_depth_snr,
        "bls_exact_log_likelihood": exact_log_likelihood,
        # Backward-compatible names are all exact-period values.
        "bls_duration_days": exact_duration,
        "bls_duration_phase_fraction": exact_duration / float(period),
        "bls_transit_time": exact_transit_time,
        "bls_depth": exact_depth,
        "bls_depth_err": exact_depth_err,
        "bls_depth_snr": exact_depth_snr,
        "bls_log_likelihood": exact_log_likelihood,
        "bls_local_status": "deferred",
        "bls_local_best_power": np.nan,
        "bls_refined_period_days": np.nan,
        "bls_refined_relative_shift": np.nan,
    }


def _refine_bls_period(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    *,
    baseline: float,
    config: PeriodCandidateMethodsConfig,
) -> dict[str, Any]:
    """Refine one BLS seed at duty-cycle-aware frequency resolution."""

    edge_grid, requested_half_width = _bounded_local_frequency_grid(
        period,
        baseline=baseline,
        rayleigh_half_width=config.fixed_bls_refine_rayleigh_half_width,
        relative_half_width=config.fixed_bls_refine_relative_half_width,
        n_frequency=3,
        config=config,
    )
    low_frequency = float(np.min(edge_grid))
    high_frequency = float(np.max(edge_grid))
    frequency_span = max(0.0, high_frequency - low_frequency)
    minimum_duration_fraction = float(min(config.bls_duration_fractions))
    required = (
        int(
            math.ceil(
                frequency_span
                * float(baseline)
                * float(config.fixed_bls_refine_frequency_oversample)
                / minimum_duration_fraction
            )
        )
        + 1
    )
    n_frequency = max(int(config.fixed_bls_refine_n_frequency), required, 3)
    cap_hit = n_frequency > int(config.fixed_bls_refine_max_frequency_points)
    n_frequency = min(
        n_frequency, int(config.fixed_bls_refine_max_frequency_points)
    )
    frequency = np.linspace(low_frequency, high_frequency, n_frequency)
    center_frequency = 1.0 / float(period)
    if low_frequency <= center_frequency <= high_frequency:
        frequency = np.unique(np.append(frequency, center_frequency))
    periods = 1.0 / frequency
    try:
        values = _bls_grouped_power(
            time,
            mag,
            err,
            periods,
            duration_fractions=config.bls_duration_fractions,
            n_groups=1,
            max_period_ratio=config.bls_max_period_ratio_per_group,
            oversample=config.bls_oversample,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    if not np.isfinite(values["power"]).any():
        return {"status": "no_finite_power"}
    best_idx = int(np.nanargmax(values["power"]))
    best_period = float(periods[best_idx])
    actual_frequency_spacing = (
        float(np.max(np.diff(np.sort(frequency))))
        if frequency.size >= 2
        else np.nan
    )
    return {
        "status": "ok",
        "period_days": best_period,
        "power": float(values["power"][best_idx]),
        "duration_days": float(values["duration"][best_idx]),
        "duration_phase_fraction": float(
            values["duration"][best_idx] / best_period
        ),
        "transit_time": float(values["transit_time"][best_idx]),
        "depth": float(values["depth"][best_idx]),
        "depth_err": float(values["depth_err"][best_idx]),
        "depth_snr": float(values["depth_snr"][best_idx]),
        "log_likelihood": float(values["log_likelihood"][best_idx]),
        "requested_frequency_half_width": float(requested_half_width),
        "n_frequency": int(frequency.size),
        "frequency_spacing_per_day": actual_frequency_spacing,
        "frequency_spacing_rayleigh": (
            actual_frequency_spacing * float(baseline)
            if np.isfinite(actual_frequency_spacing)
            else np.nan
        ),
        "resolution_cap_hit": bool(cap_hit),
    }


def _odd_even_features(
    time: np.ndarray,
    mag: np.ndarray,
    period: float,
    *,
    n_phase_bins: int,
    min_cycles_per_parity: int,
    min_common_bins: int,
) -> dict[str, float | int | str]:
    cycle = np.floor((time - float(np.min(time))) / float(period)).astype(np.int64)
    baseline_mag = float(np.nanpercentile(mag, 25.0))
    even = (cycle % 2) == 0
    odd = ~even
    n_even_cycles = int(np.unique(cycle[even]).size)
    n_odd_cycles = int(np.unique(cycle[odd]).size)

    def depth(mask: np.ndarray) -> float:
        values = mag[mask]
        return (
            float(np.nanpercentile(values, 90.0) - baseline_mag)
            if values.size >= 3
            else np.nan
        )

    even_depth = depth(even)
    odd_depth = depth(odd)
    absolute_difference = (
        abs(even_depth - odd_depth)
        if np.isfinite(even_depth) and np.isfinite(odd_depth)
        else np.nan
    )
    depth_ratio = (
        min(even_depth, odd_depth) / max(even_depth, odd_depth)
        if np.isfinite(even_depth)
        and np.isfinite(odd_depth)
        and min(even_depth, odd_depth) >= 0
        and max(even_depth, odd_depth) > 0
        else np.nan
    )

    phase = np.mod((time - float(np.min(time))) / float(period), 1.0)
    edges = np.linspace(0.0, 1.0, max(4, int(n_phase_bins)) + 1)
    even_template = np.full(edges.size - 1, np.nan)
    odd_template = np.full(edges.size - 1, np.nan)
    for idx in range(edges.size - 1):
        in_bin = (phase >= edges[idx]) & (phase < edges[idx + 1])
        if np.count_nonzero(in_bin & even) >= 2:
            even_template[idx] = float(np.median(mag[in_bin & even]))
        if np.count_nonzero(in_bin & odd) >= 2:
            odd_template[idx] = float(np.median(mag[in_bin & odd]))
    common = np.isfinite(even_template) & np.isfinite(odd_template)
    shape_rms = (
        float(np.sqrt(np.mean(np.square(even_template[common] - odd_template[common]))))
        if np.count_nonzero(common) >= 3
        else np.nan
    )
    n_common_bins = int(np.count_nonzero(common))
    sufficient = (
        n_even_cycles >= int(min_cycles_per_parity)
        and n_odd_cycles >= int(min_cycles_per_parity)
        and n_common_bins >= int(min_common_bins)
    )
    if not sufficient:
        even_depth = np.nan
        odd_depth = np.nan
        absolute_difference = np.nan
        depth_ratio = np.nan
        shape_rms = np.nan
    return {
        "odd_even_status": "ok" if sufficient else "insufficient_cycles",
        "odd_even_n_even_points": int(np.count_nonzero(even)),
        "odd_even_n_odd_points": int(np.count_nonzero(odd)),
        "odd_even_n_even_cycles": n_even_cycles,
        "odd_even_n_odd_cycles": n_odd_cycles,
        "odd_even_n_common_bins": n_common_bins,
        "odd_even_even_depth": even_depth,
        "odd_even_odd_depth": odd_depth,
        "odd_even_depth_abs_difference": absolute_difference,
        "odd_even_depth_ratio": depth_ratio,
        "odd_even_shape_rms": shape_rms,
    }


def _seasonal_stability_features(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    *,
    config: PeriodCandidateMethodsConfig,
) -> dict[str, Any]:
    """Measure whether fixed-period Fourier evidence persists across seasons."""

    segment_index = np.floor(
        (time - float(np.min(time))) / float(config.stability_segment_days)
    ).astype(np.int64)
    powers: list[float] = []
    amplitudes: list[float] = []
    phase_vectors: list[complex] = []
    total_segments = int(np.unique(segment_index).size)
    for segment in np.unique(segment_index):
        mask = segment_index == segment
        if int(np.count_nonzero(mask)) < int(
            config.stability_min_points_per_segment
        ):
            continue
        segment_time = time[mask]
        segment_mag = mag[mask]
        segment_err = None if err is None else err[mask]
        values = _weighted_fourier_score(
            segment_time,
            segment_mag,
            segment_err,
            period,
            1,
        )
        if np.isfinite(values["power"]):
            powers.append(float(values["power"]))
        if np.isfinite(values["amplitude"]):
            amplitudes.append(float(values["amplitude"]))

        weights = (
            np.ones(segment_mag.size, dtype=float)
            if segment_err is None
            else 1.0 / np.square(segment_err)
        )
        centered = segment_mag - float(
            np.average(segment_mag, weights=weights)
        )
        vector = np.sum(
            weights
            * centered
            * np.exp(
                -2.0j * np.pi * np.mod(segment_time / float(period), 1.0)
            )
        )
        if np.isfinite(vector.real) and np.isfinite(vector.imag) and abs(vector) > 0:
            phase_vectors.append(complex(vector / abs(vector)))

    valid_segments = len(powers)
    sufficient = valid_segments >= int(config.stability_min_segments)
    power_arr = np.asarray(powers, dtype=float)
    amplitude_arr = np.asarray(amplitudes, dtype=float)
    phase_resultant = (
        float(abs(np.mean(np.asarray(phase_vectors, dtype=complex))))
        if len(phase_vectors) >= int(config.stability_min_segments)
        else np.nan
    )
    amplitude_mean = (
        float(np.mean(amplitude_arr)) if amplitude_arr.size else np.nan
    )
    return {
        "stability_status": "ok" if sufficient else "insufficient_segments",
        "stability_total_segments": total_segments,
        "stability_valid_segments": int(valid_segments),
        "stability_valid_segment_fraction": (
            float(valid_segments / total_segments) if total_segments else 0.0
        ),
        "stability_fourier_power_min": (
            float(np.min(power_arr)) if sufficient else np.nan
        ),
        "stability_fourier_power_median": (
            float(np.median(power_arr)) if sufficient else np.nan
        ),
        "stability_fourier_power_std": (
            float(np.std(power_arr, ddof=1))
            if sufficient and power_arr.size >= 2
            else np.nan
        ),
        "stability_fourier_power_positive_fraction": (
            float(np.mean(power_arr > 0.0)) if sufficient else np.nan
        ),
        "stability_amplitude_cv": (
            float(np.std(amplitude_arr, ddof=1) / amplitude_mean)
            if sufficient
            and amplitude_arr.size >= 2
            and np.isfinite(amplitude_mean)
            and amplitude_mean > 0
            else np.nan
        ),
        "stability_phase_resultant": phase_resultant,
    }


def _design_bic(
    design: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
) -> float:
    """Return Gaussian weighted least-squares BIC for a fixed design matrix."""

    weight = (
        np.ones(mag.size, dtype=float)
        if err is None
        else 1.0 / np.asarray(err, dtype=float)
    )
    weighted_design = np.asarray(design, dtype=float) * weight[:, None]
    weighted_mag = np.asarray(mag, dtype=float) * weight
    try:
        coefficients, _, _, _ = np.linalg.lstsq(
            weighted_design,
            weighted_mag,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return np.nan
    residual = (mag - np.asarray(design, dtype=float) @ coefficients) * weight
    rss = float(np.sum(np.square(residual)))
    if mag.size <= design.shape[1] or not np.isfinite(rss):
        return np.nan
    return float(
        mag.size
        * np.log(max(rss / mag.size, np.finfo(float).tiny))
        + design.shape[1] * np.log(mag.size)
    )


def _null_model_features(
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    period: float,
    *,
    config: PeriodCandidateMethodsConfig,
) -> dict[str, Any]:
    """Compare a periodic model with constant, trend, and annual null models."""

    scaled_time = (
        (time - float(np.mean(time))) / max(float(np.ptp(time)), 1.0)
    )
    constant = np.ones((time.size, 1), dtype=float)
    linear = np.column_stack((constant[:, 0], scaled_time))
    quadratic = np.column_stack(
        (constant[:, 0], scaled_time, np.square(scaled_time))
    )
    annual_angle = 2.0 * np.pi * time / 365.25
    annual = np.column_stack(
        (
            constant[:, 0],
            scaled_time,
            np.cos(annual_angle),
            np.sin(annual_angle),
        )
    )
    constant_bic = _design_bic(constant, mag, err)
    linear_bic = _design_bic(linear, mag, err)
    quadratic_bic = _design_bic(quadratic, mag, err)
    annual_bic = _design_bic(annual, mag, err)
    periodic_bics = [
        _weighted_fourier_score(time, mag, err, period, int(order))["bic"]
        for order in config.fixed_fourier_orders
    ]
    finite_periodic = [
        float(value) for value in periodic_bics if np.isfinite(value)
    ]
    best_periodic_bic = (
        float(min(finite_periodic)) if finite_periodic else np.nan
    )
    differences = np.diff(mag)
    centered = mag - float(np.mean(mag))
    denominator = float(np.sum(np.square(centered)))
    von_neumann = (
        float(np.sum(np.square(differences)) / denominator)
        if denominator > 0 and differences.size
        else np.nan
    )

    def improvement(null_bic: float) -> float:
        return (
            float(null_bic - best_periodic_bic)
            if np.isfinite(null_bic) and np.isfinite(best_periodic_bic)
            else np.nan
        )

    return {
        "null_model_status": (
            "ok" if np.isfinite(best_periodic_bic) else "invalid_periodic_model"
        ),
        "null_constant_bic": constant_bic,
        "null_linear_bic": linear_bic,
        "null_quadratic_bic": quadratic_bic,
        "null_annual_bic": annual_bic,
        "null_best_periodic_bic": best_periodic_bic,
        "null_periodic_vs_constant_delta_bic": improvement(constant_bic),
        "null_periodic_vs_linear_delta_bic": improvement(linear_bic),
        "null_periodic_vs_quadratic_delta_bic": improvement(quadratic_bic),
        "null_periodic_vs_annual_delta_bic": improvement(annual_bic),
        "null_time_order_von_neumann_ratio": von_neumann,
    }


def score_fixed_period(
    time: Sequence[float],
    mag: Sequence[float],
    period_days: float,
    err: Sequence[float] | None = None,
    *,
    event_epochs: Sequence[float] | None = None,
    config: PeriodCandidateMethodsConfig | None = None,
) -> dict[str, Any]:
    """Compute all fixed-period arbitration features for one candidate."""
    cfg = config or PeriodCandidateMethodsConfig()
    period = float(period_days)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period_days must be finite and positive")
    time_arr, mag_arr, err_arr = _prepare_light_curve(time, mag, err)
    if time_arr.size < int(cfg.min_points):
        return {"feature_status": "insufficient_points", "n_points": int(time_arr.size)}
    baseline = float(np.ptp(time_arr))
    if not np.isfinite(baseline) or baseline <= 0:
        return {"feature_status": "zero_baseline", "n_points": int(time_arr.size)}

    features: dict[str, Any] = {
        "feature_status": "ok",
        "n_points": int(time_arr.size),
        "baseline_days": baseline,
        "baseline_cycles": float(baseline / period),
        "period_days": period,
        "frequency_per_day": 1.0 / period,
    }
    enabled = set(cfg.enabled_fixed_methods)
    if "ls" in enabled:
        features.update(
            _fixed_ls_features(
                time_arr,
                mag_arr,
                err_arr,
                period,
                baseline=baseline,
                config=cfg,
            )
        )
    else:
        features.update({"ls_power": np.nan, "ls_status": "disabled"})

    if "pdm" in enabled:
        try:
            if cfg.pdm_method == "plavchan":
                features["pdm_theta"] = float(
                    _pdm_theta_plavchan(
                        time_arr,
                        mag_arr,
                        period,
                        cfg.pdm_plavchan_phase_width,
                        cfg.pdm_plavchan_min_neighbors,
                        0.05,
                        min(25, max(3, int(mag_arr.size))),
                    )
                )
            else:
                features["pdm_theta"] = float(
                    _pdm_theta(time_arr, mag_arr, period, cfg.pdm_n_phase_bins)
                )
            features["pdm_status"] = "ok"
        except Exception as exc:
            features.update({"pdm_theta": np.nan, "pdm_status": "error", "pdm_error": str(exc)})
    else:
        features.update({"pdm_theta": np.nan, "pdm_status": "disabled"})

    if "ce" in enabled:
        try:
            features["ce_entropy"] = float(
                _ce_evaluate(
                    time_arr,
                    mag_arr,
                    period,
                    cfg.ce_n_phase_bins,
                    cfg.ce_n_mag_bins,
                )
            )
            features["ce_status"] = "ok"
        except Exception as exc:
            features.update({"ce_entropy": np.nan, "ce_status": "error", "ce_error": str(exc)})
    else:
        features.update({"ce_entropy": np.nan, "ce_status": "disabled"})

    if "bls" in enabled:
        features.update(
            _fixed_bls_features(
                time_arr,
                mag_arr,
                err_arr,
                period,
                baseline=baseline,
                config=cfg,
            )
        )
    else:
        features.update({"bls_status": "disabled", "bls_power": np.nan})

    if "multiharmonic_fourier" in enabled:
        for order in cfg.fixed_fourier_orders:
            values = _weighted_fourier_score(time_arr, mag_arr, err_arr, period, int(order))
            for name, value in values.items():
                features[f"fourier_{int(order)}_{name}"] = value
        features["multiharmonic_fourier_status"] = "ok"
    else:
        features["multiharmonic_fourier_status"] = "disabled"

    if "lafler_kinman" in enabled:
        lk = lafler_kinman_period_stats(time_arr, mag_arr, period)
        features.update(lk)
        features["lafler_kinman_status"] = (
            "ok" if np.isfinite(lk["lafler_kinman_t_phase"]) else "invalid"
        )
    else:
        features.update(
            {
                "lafler_kinman_status": "disabled",
                "lafler_kinman_t_time": np.nan,
                "lafler_kinman_t_phase": np.nan,
                "lafler_kinman_delta": np.nan,
            }
        )

    if "supersmoother" in enabled:
        smoother_class, version_or_error = _load_supersmoother()
        if smoother_class is None:
            features.update(
                {
                    "supersmoother_status": "unavailable",
                    "supersmoother_cv_error": np.nan,
                    "supersmoother_explained_fraction": np.nan,
                    "supersmoother_message": version_or_error,
                }
            )
        else:
            try:
                cv_metrics = _supersmoother_cv_metrics(
                    time_arr,
                    mag_arr,
                    err_arr,
                    period,
                    smoother_class,
                )
                raw_var = float(np.var(mag_arr, ddof=1))
                features.update(
                    {
                        "supersmoother_status": "ok",
                        # The package's native CV objective is standardized by
                        # each observation uncertainty.
                        "supersmoother_cv_error": cv_metrics["standardized_mae"],
                        "supersmoother_cv_standardized_mae": cv_metrics[
                            "standardized_mae"
                        ],
                        "supersmoother_cv_standardized_mse": cv_metrics[
                            "standardized_mse"
                        ],
                        # These values are de-standardized back into magnitude
                        # units and are the only ones compared with raw scatter.
                        "supersmoother_cv_mae": cv_metrics["magnitude_mae"],
                        "supersmoother_cv_mse": cv_metrics["magnitude_mse"],
                        "supersmoother_explained_fraction": (
                            float(1.0 - cv_metrics["magnitude_mse"] / raw_var)
                            if np.isfinite(cv_metrics["magnitude_mse"]) and raw_var > 0
                            else np.nan
                        ),
                        "supersmoother_dependency_version": version_or_error,
                    }
                )
            except Exception as exc:
                features.update(
                    {
                        "supersmoother_status": "error",
                        "supersmoother_cv_error": np.nan,
                        "supersmoother_explained_fraction": np.nan,
                        "supersmoother_message": str(exc),
                    }
                )
    else:
        features.update(
            {
                "supersmoother_status": "disabled",
                "supersmoother_cv_error": np.nan,
                "supersmoother_explained_fraction": np.nan,
            }
        )

    if "heldout_template" in enabled:
        if err_arr is None:
            scale = float(np.nanmedian(np.abs(mag_arr - np.nanmedian(mag_arr))) * 1.4826)
            template_err = np.full(mag_arr.size, max(scale * 0.05, 1.0e-6), dtype=float)
            features["template_error_source"] = "robust_scale_proxy"
        else:
            template_err = err_arr
            features["template_error_source"] = "input"
        try:
            template = phase_template_quasi_periodicity(
                mag_arr,
                time_arr,
                template_err,
                period,
                n_phase_bins=cfg.template_n_phase_bins,
                min_bin_points=cfg.template_min_bin_points,
                smooth_window_bins=cfg.template_smooth_window_bins,
                min_bin_coverage=cfg.template_min_bin_coverage,
                noise_subtract=cfg.template_noise_subtract,
            )
            features.update({f"template_{key}": value for key, value in template.items()})
        except Exception as exc:
            features.update({"template_status": "error", "template_error": str(exc)})
    else:
        features["template_status"] = "disabled"

    if "odd_even" in enabled:
        features.update(
            _odd_even_features(
                time_arr,
                mag_arr,
                period,
                n_phase_bins=cfg.odd_even_n_phase_bins,
                min_cycles_per_parity=cfg.odd_even_min_cycles_per_parity,
                min_common_bins=cfg.odd_even_min_common_bins,
            )
        )
    else:
        features["odd_even_status"] = "disabled"

    if "seasonal_stability" in enabled:
        features.update(
            _seasonal_stability_features(
                time_arr,
                mag_arr,
                err_arr,
                period,
                config=cfg,
            )
        )
    else:
        features["stability_status"] = "disabled"

    if "null_model_comparison" in enabled:
        features.update(
            _null_model_features(
                time_arr,
                mag_arr,
                err_arr,
                period,
                config=cfg,
            )
        )
    else:
        features["null_model_status"] = "disabled"

    if "event_coherence" not in enabled:
        features["event_status"] = "disabled"
    elif event_epochs is None:
        features["event_status"] = "no_events"
        features["event_n_events"] = 0
        features["event_evidence_enabled"] = 0
        for name in (
            "event_score",
            "event_phase_concentration",
            "event_inlier_fraction",
            "event_median_abs_oc_days",
            "event_rms_oc_days",
        ):
            features[name] = np.nan
    else:
        event_arr = np.asarray(event_epochs, dtype=float)
        event_arr = np.sort(np.unique(event_arr[np.isfinite(event_arr)]))
        exact_event = _event_phase_diagnostics(
            event_arr,
            period,
            tolerance=cfg.event_phase_tolerance,
            complexity_penalty=cfg.event_cycle_complexity_penalty,
            refine_period=False,
        )
        local_event = _event_phase_diagnostics(
            event_arr,
            period,
            tolerance=cfg.event_phase_tolerance,
            complexity_penalty=cfg.event_cycle_complexity_penalty,
            refine_period=True,
        )
        enough_events = event_arr.size >= int(cfg.event_min_events_for_ranking)
        enough_cycles = int(exact_event["max_cycle"]) >= int(
            cfg.event_min_cycle_span_for_ranking
        )
        enough_coherence = (
            np.isfinite(exact_event["inlier_fraction"])
            and float(exact_event["inlier_fraction"])
            >= float(cfg.event_min_inlier_fraction)
        )
        event_evidence_enabled = bool(
            enough_events and enough_cycles and enough_coherence
        )
        if event_arr.size < 2:
            event_status = "insufficient_events"
        elif not event_evidence_enabled:
            event_status = "insufficient_coherent_events"
        else:
            event_status = "ok"
        features.update(
            {
                "event_status": event_status,
                "event_n_events": int(event_arr.size),
                "event_evidence_enabled": int(event_evidence_enabled),
                "event_exact_period_days": period,
                "event_exact_score": exact_event["score"],
                "event_exact_phase_concentration": exact_event[
                    "phase_concentration"
                ],
                "event_exact_inlier_fraction": exact_event[
                    "inlier_fraction"
                ],
                "event_exact_median_abs_oc_days": exact_event[
                    "median_abs_oc_days"
                ],
                "event_exact_rms_oc_days": exact_event["rms_oc_days"],
                "event_exact_max_cycle": exact_event["max_cycle"],
                "event_local_refined_period_days": local_event[
                    "period_days"
                ],
                "event_local_score": local_event["score"],
                "event_local_phase_concentration": local_event[
                    "phase_concentration"
                ],
                "event_local_inlier_fraction": local_event[
                    "inlier_fraction"
                ],
                "event_local_median_abs_oc_days": local_event[
                    "median_abs_oc_days"
                ],
                "event_local_rms_oc_days": local_event["rms_oc_days"],
                "event_local_max_cycle": local_event["max_cycle"],
                # Preserve ungated diagnostics for detector work while the
                # backward-compatible ranker fields are enabled only when
                # at least three events span multiple cycles.
                "event_raw_score": exact_event["score"],
                "event_raw_phase_concentration": exact_event[
                    "phase_concentration"
                ],
                "event_raw_inlier_fraction": exact_event[
                    "inlier_fraction"
                ],
                "event_raw_median_abs_oc_days": exact_event[
                    "median_abs_oc_days"
                ],
                "event_raw_rms_oc_days": exact_event["rms_oc_days"],
                "event_score": (
                    exact_event["score"] if event_evidence_enabled else np.nan
                ),
                "event_phase_concentration": (
                    exact_event["phase_concentration"]
                    if event_evidence_enabled
                    else np.nan
                ),
                "event_inlier_fraction": (
                    exact_event["inlier_fraction"]
                    if event_evidence_enabled
                    else np.nan
                ),
                "event_median_abs_oc_days": (
                    exact_event["median_abs_oc_days"]
                    if event_evidence_enabled
                    else np.nan
                ),
                "event_rms_oc_days": (
                    exact_event["rms_oc_days"]
                    if event_evidence_enabled
                    else np.nan
                ),
                "event_max_cycle": (
                    exact_event["max_cycle"]
                    if event_evidence_enabled
                    else np.nan
                ),
                "event_refined_period_days": local_event["period_days"],
            }
        )
    if "alias_evidence" in enabled:
        features.update(_alias_features(period, baseline=baseline, config=cfg))
    else:
        features["alias_status"] = "disabled"
    return features


def score_candidate_bank(
    time: Sequence[float],
    mag: Sequence[float],
    candidates: Sequence[PeriodCandidate],
    err: Sequence[float] | None = None,
    *,
    event_epochs: Sequence[float] | None = None,
    config: PeriodCandidateMethodsConfig | None = None,
) -> list[CandidateScore]:
    """Compute fixed-period feature records for a candidate bank."""
    cfg = config or PeriodCandidateMethodsConfig()
    scores = []
    for candidate in candidates:
        candidate.was_scored = True
        features = score_fixed_period(
            time,
            mag,
            candidate.period_days,
            err,
            event_epochs=event_epochs,
            config=cfg,
        )
        scores.append(
            CandidateScore(
                period_days=float(candidate.period_days),
                frequency_per_day=float(candidate.frequency_per_day),
                features=features,
                status=str(features.get("feature_status", "ok")),
                contributing_methods=tuple(candidate.contributing_methods),
                parent_period_days=float(candidate.parent_period_days),
                harmonic_factor=float(candidate.harmonic_factor),
                proposal_method=str(candidate.method),
                proposal_raw_score=float(candidate.raw_score),
                proposal_objective=str(candidate.objective),
                proposal_rank=int(candidate.rank),
                proposal_prominence=float(candidate.prominence),
                proposal_normalized_score=float(candidate.normalized_score),
                proposal_search_band=str(candidate.search_band),
                proposal_metadata=dict(candidate.metadata),
            )
        )
    return scores


def refine_scored_candidates(
    candidates: Sequence[PeriodCandidate],
    scores: Sequence[CandidateScore],
    *,
    baseline_days: float,
    time: Sequence[float] | None = None,
    mag: Sequence[float] | None = None,
    err: Sequence[float] | None = None,
    config: PeriodCandidateMethodsConfig | None = None,
    include_original: bool = False,
) -> list[PeriodCandidate]:
    """Materialize bounded local-LS/BLS optima as candidate variants.

    Parameters
    ----------
    candidates, scores:
        Parallel sequences from a scoring shortlist and
        :func:`score_candidate_bank`.  Periods are validated before use so a
        stale/misaligned score table cannot silently refine the wrong seed.
    baseline_days:
        Used for the same frequency-resolution de-duplication as the global
        candidate bank.
    include_original:
        If true, return originals and variants together.  The default returns
        only genuinely shifted variants for an optional second scoring pass.
    time, mag, err:
        Prepared photometry used for high-resolution local BLS refinement.
        When omitted, LS variants remain available and BLS refinement is
        skipped rather than using an under-resolved proxy grid.

    Notes
    -----
    The local searches themselves are already bounded by the fixed LS/BLS
    refinement windows in the configuration.  This helper does not extrapolate
    beyond those windows and applies a separate candidate-count cap.
    """
    cfg = config or PeriodCandidateMethodsConfig()
    candidate_list = list(candidates)
    score_list = list(scores)
    if len(candidate_list) != len(score_list):
        raise ValueError("candidates and scores must be parallel sequences of equal length")
    minimum_period = float(cfg.short_min_period_days)
    maximum_period = min(
        float(cfg.long_absolute_max_period_days),
        float(cfg.candidate_max_baseline_fraction) * float(baseline_days),
    )
    bls_time: np.ndarray | None = None
    bls_mag: np.ndarray | None = None
    bls_err: np.ndarray | None = None
    if time is not None or mag is not None or err is not None:
        if time is None or mag is None:
            raise ValueError("time and mag must be supplied together for BLS refinement")
        bls_time, bls_mag, bls_err = _prepare_light_curve(time, mag, err)
        realized_baseline = float(np.ptp(bls_time)) if bls_time.size >= 2 else np.nan
        if not np.isfinite(realized_baseline) or not np.isclose(
            realized_baseline,
            float(baseline_days),
            rtol=0.0,
            atol=max(1.0e-8, 1.0e-10 * float(baseline_days)),
        ):
            raise ValueError(
                "BLS refinement photometry baseline does not match baseline_days"
            )

    bls_seed_indices: set[int] = set()
    if bls_time is not None and "bls" in cfg.local_refinement_sources:
        def bls_seed_power(index: int) -> float:
            try:
                value = float(score_list[index].features.get("bls_power", np.nan))
            except (TypeError, ValueError):
                return np.nan
            return value

        ordered_bls_seeds = sorted(
            (
                index
                for index in range(len(score_list))
                if np.isfinite(bls_seed_power(index))
            ),
            key=lambda index: (
                -bls_seed_power(index),
                _candidate_priority(candidate_list[index]),
            ),
        )
        bls_seed_indices = set(
            ordered_bls_seeds[: int(cfg.fixed_bls_max_refinement_seeds)]
        )

    variants: list[PeriodCandidate] = list(candidate_list) if include_original else []
    for candidate_index, (candidate, score) in enumerate(
        zip(candidate_list, score_list)
    ):
        if not np.isclose(
            float(candidate.period_days),
            float(score.period_days),
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("candidate and score periods are misaligned")
        feature_map = score.features
        for source in cfg.local_refinement_sources:
            if source == "ls":
                period_key = "ls_local_best_period_days"
                score_key = "ls_local_best_power"
                refinement_metadata: dict[str, Any] = {}
            else:
                if (
                    bls_time is None
                    or bls_mag is None
                    or candidate_index not in bls_seed_indices
                ):
                    continue
                bls_refinement = _refine_bls_period(
                    bls_time,
                    bls_mag,
                    bls_err,
                    float(candidate.period_days),
                    baseline=float(baseline_days),
                    config=cfg,
                )
                if bls_refinement.get("status") != "ok":
                    continue
                refined_period = float(bls_refinement["period_days"])
                refined_score = float(bls_refinement["power"])
                refinement_metadata = {
                    f"bls_local_{key}": value
                    for key, value in bls_refinement.items()
                    if key not in {"period_days", "power"}
                }
            if source == "ls":
                try:
                    refined_period = float(feature_map.get(period_key, np.nan))
                    refined_score = float(feature_map.get(score_key, np.nan))
                except (TypeError, ValueError):
                    continue
            if (
                not np.isfinite(refined_period)
                or not minimum_period <= refined_period <= maximum_period
            ):
                continue
            relative_shift = abs(refined_period - candidate.period_days) / candidate.period_days
            if relative_shift < float(cfg.local_refinement_min_relative_shift):
                continue
            refiner = f"local_{source}_refinement"
            variants.append(
                PeriodCandidate(
                    method=refiner,
                    period_days=refined_period,
                    frequency_per_day=1.0 / refined_period,
                    raw_score=refined_score,
                    objective="maximize",
                    rank=int(candidate.rank),
                    prominence=(
                        float(refined_score - feature_map.get(f"{source}_power", np.nan))
                        if np.isfinite(refined_score)
                        and np.isfinite(float(feature_map.get(f"{source}_power", np.nan)))
                        else np.nan
                    ),
                    normalized_score=float(candidate.normalized_score),
                    search_band="local_refinement",
                    parent_period_days=float(candidate.period_days),
                    harmonic_factor=float(candidate.harmonic_factor),
                    contributing_methods=tuple(
                        sorted(set(candidate.contributing_methods) | {refiner})
                    ),
                    metadata={
                        **candidate.metadata,
                        "refinement_source": source,
                        "refinement_seed_period_days": float(candidate.period_days),
                        "refinement_relative_shift": float(
                            (refined_period - candidate.period_days) / candidate.period_days
                        ),
                        "refinement_local_score": refined_score,
                        "seed_method": candidate.method,
                        **refinement_metadata,
                    },
                )
            )

    merged = _merge_candidates(variants, baseline_days=baseline_days, config=cfg)
    if cfg.max_refined_candidates is not None:
        merged = merged[: int(cfg.max_refined_candidates)]
    return merged


def select_scoring_shortlist(
    candidates: Sequence[PeriodCandidate],
    *,
    config: PeriodCandidateMethodsConfig | None = None,
) -> list[PeriodCandidate]:
    """Select a proposer-diverse expensive-scoring shortlist.

    The complete input bank is left in place for oracle-coverage accounting;
    its candidates receive explicit ``was_scored`` flags.  Set
    ``config.max_scored_candidates=None`` for an exhaustive shortlist.
    """
    cfg = config or PeriodCandidateMethodsConfig()
    maximum = cfg.max_scored_candidates
    ordered = list(candidates)
    for candidate in ordered:
        candidate.was_scored = False
    if maximum is None or len(ordered) <= int(maximum):
        for candidate in ordered:
            candidate.was_scored = True
        return ordered

    limit = int(maximum)
    chosen: list[PeriodCandidate] = []
    chosen_ids: set[int] = set()
    reserved_methods = tuple(
        method
        for method in cfg.shortlist_reserved_methods
        if method in cfg.enabled_global_methods
    )
    by_method = {
        method: [
            candidate
            for candidate in ordered
            if method in candidate.contributing_methods
        ]
        for method in reserved_methods
    }

    # Round-robin across proposers prevents a high-density LS or harmonic
    # family from consuming the expensive-scoring budget before a specialist
    # proposer (BLS, events, string length, or SuperSmoother) is represented.
    depth = 0
    while len(chosen) < limit:
        added = False
        for method in reserved_methods:
            method_candidates = by_method[method]
            if depth >= len(method_candidates):
                continue
            candidate = method_candidates[depth]
            identity = id(candidate)
            if identity in chosen_ids:
                continue
            chosen.append(candidate)
            chosen_ids.add(identity)
            added = True
            if len(chosen) >= limit:
                break
        if not added and all(
            depth + 1 >= len(values) for values in by_method.values()
        ):
            break
        depth += 1

    for candidate in ordered:
        if len(chosen) >= limit:
            break
        identity = id(candidate)
        if identity not in chosen_ids:
            chosen.append(candidate)
            chosen_ids.add(identity)
    chosen_set = {id(candidate) for candidate in chosen}
    for candidate in ordered:
        candidate.was_scored = id(candidate) in chosen_set
    # Restore proposal-priority order for deterministic downstream row order.
    return [candidate for candidate in ordered if candidate.was_scored]


def run_period_candidate_suite(
    time: Sequence[float],
    mag: Sequence[float],
    err: Sequence[float] | None = None,
    *,
    event_epochs: Sequence[float] | None = None,
    config: PeriodCandidateMethodsConfig | None = None,
) -> PeriodCandidateSuiteResult:
    """Run all configured global searches, candidate processing, and scoring."""
    cfg = config or PeriodCandidateMethodsConfig()
    prepared_time, _, _ = _prepare_light_curve(time, mag, err)
    baseline = float(np.ptp(prepared_time)) if prepared_time.size >= 2 else np.nan
    searches = run_global_period_searches(
        time,
        mag,
        err,
        event_epochs=event_epochs,
        config=cfg,
    )
    raw = [
        candidate
        for result in searches.values()
        for candidate in result.candidates
        if result.status == "ok"
    ]
    merged = _merge_candidates(raw, baseline_days=baseline, config=cfg)
    expanded = expand_harmonic_candidates(merged, baseline_days=baseline, config=cfg)
    scoring_shortlist = select_scoring_shortlist(
        expanded,
        config=cfg,
    )
    scores = score_candidate_bank(
        time,
        mag,
        scoring_shortlist,
        err,
        event_epochs=event_epochs,
        config=cfg,
    )
    status = "ok" if scores else "no_candidates"
    return PeriodCandidateSuiteResult(
        config=cfg,
        search_results=searches,
        raw_candidates=raw,
        merged_candidates=merged,
        expanded_candidates=expanded,
        candidate_scores=scores,
        status=status,
    )


__all__ = [
    "FIXED_METHODS",
    "GLOBAL_METHODS",
    "CandidateScore",
    "PeriodCandidate",
    "PeriodCandidateMethodsConfig",
    "PeriodCandidateSuiteResult",
    "PeriodSearchResult",
    "build_candidate_bank",
    "event_epoch_detection_diagnostics",
    "expand_harmonic_candidates",
    "extract_frequency_separated_extrema",
    "merge_search_results",
    "robust_event_comb_candidates",
    "refine_scored_candidates",
    "run_global_period_searches",
    "run_period_candidate_suite",
    "select_scoring_shortlist",
    "score_candidate_bank",
    "score_fixed_period",
]
