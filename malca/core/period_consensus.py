"""Unified period consensus with event-informed harmonic arbitration.

This module ties together the raw periodogram outputs from MALCA's short-period
methods (PDM, CE, LSP) and the new long-period LS discovery stage
(``malca.core.stats.long_period_ls_search``). It produces a single
``ConsensusResult`` that downstream pipeline stages, the review UI, and paper
notebooks all read from.

Design decisions:

* No backward compatibility with the previous ``periodicity_period`` /
  ``phase_period_days`` semantics: consensus is authoritative and every
  consumer is expected to switch to it.
* For long-period candidates (LS baseline cycles ~1-2), the raw LS peak is
  frequently at ``P/k`` because a Gaussian dip is spectrally rich. We build a
  harmonic ladder from the top-K LS peaks (``factor * peak`` for
  ``factor in {1, 2, 3, 4}``) and pick the ladder rung whose phase-folded
  dip epochs cluster tightest. When dip epochs are unavailable we fall back
  to LS power ranking.
* Confidence tiers make it explicit which consensus periods deserve trust:
  ``high``: multiple independent lines of evidence agree.
  ``tentative``: one method passes but another disagrees or events are absent.
  ``none``: no method reaches significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import math

import numpy as np

from malca.config import (
    LONG_PERIOD_BASELINE_FRACTION,
    POST_FILTER_CE_MIN_ENTROPY,
    POST_FILTER_CE_SNR_THRESHOLD,
    POST_FILTER_PDM_MIN_THETA,
    POST_FILTER_PDM_SNR_THRESHOLD,
)
from malca.core.period_arbitration import (
    NATIVE_PERIOD_WITH_MULTIPLES_FACTORS,
    finite_float,
    period_alias_matches,
)


HIGH = "high"
TENTATIVE = "tentative"
NONE = "none"
_VALID_CONFIDENCE = frozenset({HIGH, TENTATIVE, NONE})

# Confidence gates
MIN_BASELINE_CYCLES_HIGH = 2.0
MIN_BASELINE_CYCLES_TENTATIVE = 1.4
SHORT_LONG_METHOD_AGREE_REL_TOL = 0.05  # 5% period-space tolerance
LONG_LS_HIGH_FAP_THRESHOLD = 1e-3       # bootstrap FAP for "high" long-P confidence
LONG_LS_TENTATIVE_FAP_THRESHOLD = 0.01
EVENT_PHASE_HIGH_R = 0.85               # Rayleigh R for high-confidence event alignment
EVENT_PHASE_TENTATIVE_R = 0.6
# Ladder factors for long-P LS harmonic upward-search.
LONG_LS_LADDER_FACTORS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)


@dataclass
class ConsensusResult:
    """Structured consensus period result written verbatim to storage."""

    period_consensus_days: float
    period_confidence: str
    period_method: str
    period_baseline_cycles: float
    period_confidence_reason: str
    period_evidence: dict[str, object] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        return {
            "period_consensus_days": (
                float(self.period_consensus_days)
                if np.isfinite(self.period_consensus_days)
                else np.nan
            ),
            "period_confidence": str(self.period_confidence),
            "period_method": str(self.period_method),
            "period_baseline_cycles": (
                float(self.period_baseline_cycles)
                if np.isfinite(self.period_baseline_cycles)
                else np.nan
            ),
            "period_confidence_reason": str(self.period_confidence_reason),
            "period_evidence": self.period_evidence,
        }


def _empty_result(reason: str) -> ConsensusResult:
    return ConsensusResult(
        period_consensus_days=float("nan"),
        period_confidence=NONE,
        period_method="none",
        period_baseline_cycles=float("nan"),
        period_confidence_reason=reason,
    )


def _normalize_period_window(
    min_period_days: float | None,
    max_period_days: float | None,
) -> tuple[float | None, float | None]:
    """Return finite positive inclusive period bounds."""

    min_period = finite_float(min_period_days)
    max_period = finite_float(max_period_days)
    if min_period is not None and min_period <= 0:
        min_period = None
    if max_period is not None and max_period <= 0:
        max_period = None
    if (
        min_period is not None
        and max_period is not None
        and max_period < min_period
    ):
        min_period, max_period = max_period, min_period
    return min_period, max_period


def _period_in_window(
    period_days: float | None,
    *,
    min_period_days: float | None,
    max_period_days: float | None,
) -> bool:
    period = finite_float(period_days)
    if period is None or period <= 0:
        return False
    if min_period_days is not None and period < min_period_days:
        return False
    if max_period_days is not None and period > max_period_days:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase / event scoring
# ---------------------------------------------------------------------------

def phase_concentration_R(
    event_epochs: Sequence[float],
    period_days: float,
    *,
    reference_epoch: float | None = None,
) -> float:
    """Rayleigh R for how tightly a set of event epochs fold on ``period_days``.

    ``R`` is a value in [0, 1]. Perfect alignment -> 1. Random phases -> ~0.
    With only 2 events R = cos(pi * dphase) so it is still informative albeit
    noisy: two events sharing a phase give R=1, orthogonal phases give R=0.

    Returns ``np.nan`` when the input is empty or the period is invalid.
    """
    period = finite_float(period_days)
    if period is None or period <= 0:
        return float("nan")
    epochs = np.asarray(list(event_epochs), dtype=float)
    epochs = epochs[np.isfinite(epochs)]
    if epochs.size == 0:
        return float("nan")
    if reference_epoch is not None and np.isfinite(reference_epoch):
        epochs = epochs - float(reference_epoch)
    phases = 2.0 * math.pi * ((epochs / period) % 1.0)
    R = math.hypot(float(np.mean(np.cos(phases))), float(np.mean(np.sin(phases))))
    return float(R)


def event_fold_quality(
    event_epochs: Sequence[float],
    period_days: float,
    *,
    reference_epoch: float | None = None,
    phase_align_tol: float = 0.05,
) -> dict[str, float]:
    """Event-aligned fold quality for harmonic / consensus scoring.

    Combines Rayleigh phase concentration with a centeredness bonus when events
    cluster near phase 0 (within ``phase_align_tol``) after aligning the mean
    phase to zero. Lower ``objective_penalty`` is better (additive to scatter
    objectives used by harmonic arbitration).

    Returns
    -------
    dict with keys:
        R, R2, mean_phase, phase_span, n_events, align_bonus, objective_penalty.
    """
    empty = {
        "R": float("nan"),
        "R2": float("nan"),
        "mean_phase": float("nan"),
        "phase_span": float("nan"),
        "n_events": 0.0,
        "align_bonus": 0.0,
        "objective_penalty": 0.0,
    }
    period = finite_float(period_days)
    if period is None or period <= 0:
        return empty
    epochs = np.asarray(list(event_epochs), dtype=float)
    epochs = epochs[np.isfinite(epochs)]
    if epochs.size == 0:
        return empty

    R = phase_concentration_R(epochs, period, reference_epoch=reference_epoch)
    if not np.isfinite(R):
        return empty

    # Fold to [0, 1), then circular-mean-center so φ=0 is the event cluster.
    phases = ((epochs - (float(reference_epoch) if reference_epoch is not None and np.isfinite(reference_epoch) else 0.0)) / period) % 1.0
    angles = 2.0 * math.pi * phases
    mean_angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    mean_phase = (mean_angle / (2.0 * math.pi)) % 1.0
    centered = (phases - mean_phase + 0.5) % 1.0 - 0.5
    phase_span = float(np.max(centered) - np.min(centered)) if centered.size else float("nan")

    # Bonus when the cluster is tight and we treat φ=0 as the dip phase.
    align_bonus = 0.0
    if centered.size and float(np.max(np.abs(centered))) <= float(phase_align_tol):
        align_bonus = 0.15 * float(R)

    # Penalty grows when events are dispersed in phase (low R / large span).
    # Mapped into the same "lower is better" space as scatter_ratio objectives.
    dispersion = max(0.0, 1.0 - float(R))
    span_term = 0.0
    if np.isfinite(phase_span):
        span_term = min(1.0, max(0.0, phase_span / 0.5))
    objective_penalty = 0.35 * dispersion + 0.15 * span_term - align_bonus

    return {
        "R": float(R),
        "R2": float(R * R),
        "mean_phase": float(mean_phase),
        "phase_span": float(phase_span) if np.isfinite(phase_span) else float("nan"),
        "n_events": float(epochs.size),
        "align_bonus": float(align_bonus),
        "objective_penalty": float(objective_penalty),
    }


def _rel_diff(a: float, b: float) -> float:
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return float("inf")
    return abs(a - b) / max(a, b)


def _periods_agree(a: float, b: float, *, rel_tol: float = SHORT_LONG_METHOD_AGREE_REL_TOL) -> bool:
    if _rel_diff(a, b) <= rel_tol:
        return True
    # Also treat one being a small-integer multiple/divisor of the other as agreement.
    for k in (2.0, 3.0, 4.0):
        if _rel_diff(a * k, b) <= rel_tol or _rel_diff(a, b * k) <= rel_tol:
            return True
    return False


# ---------------------------------------------------------------------------
# Harmonic ladder for long-P LS peaks
# ---------------------------------------------------------------------------

def build_long_ls_harmonic_ladder(
    top_periods: Sequence[float],
    *,
    baseline_days: float,
    factors: Sequence[float] = LONG_LS_LADDER_FACTORS,
    baseline_fraction_max: float = LONG_PERIOD_BASELINE_FRACTION,
    min_period_days: float = 10.0,
) -> list[dict[str, float]]:
    """Return ladder candidates ``{seed, factor, period}`` up to ``fraction*baseline``.

    We only expand upward (``factor >= 1``) because the LS peak is usually at
    a subharmonic when only 1-2 cycles are observed. Downward candidates are
    handled by the standard short-period arbitration in
    ``malca.core.period_arbitration``.
    """
    baseline = finite_float(baseline_days)
    if baseline is None or baseline <= 0:
        return []
    max_period = float(baseline_fraction_max) * baseline
    min_period = float(min_period_days)

    seen: list[dict[str, float]] = []
    for seed in top_periods:
        seed_val = finite_float(seed)
        if seed_val is None or seed_val <= 0:
            continue
        for factor in factors:
            f = finite_float(factor)
            if f is None or f <= 0:
                continue
            period = seed_val * f
            if period < min_period or period > max_period:
                continue
            if any(
                abs(period - float(entry["period"])) <= 1e-6 * max(period, 1.0)
                for entry in seen
            ):
                continue
            seen.append({"seed_period": seed_val, "factor": float(f), "period": float(period)})
    return seen


# ---------------------------------------------------------------------------
# Method-supported helpers
# ---------------------------------------------------------------------------

def _pdm_supported(pdm_result: dict) -> bool:
    snr = finite_float(pdm_result.get("pdm_snr"))
    theta = finite_float(pdm_result.get("pdm_min_theta", pdm_result.get("pdm_theta")))
    if snr is None or theta is None or theta <= 0:
        return False
    return (
        snr >= float(POST_FILTER_PDM_SNR_THRESHOLD)
        and theta <= float(POST_FILTER_PDM_MIN_THETA)
    )


def _ce_supported(ce_result: dict) -> bool:
    snr = finite_float(ce_result.get("ce_snr"))
    entropy = finite_float(ce_result.get("ce_min_entropy", ce_result.get("ce_entropy")))
    if snr is None or entropy is None or entropy <= 0:
        return False
    return (
        snr >= float(POST_FILTER_CE_SNR_THRESHOLD)
        and entropy <= float(POST_FILTER_CE_MIN_ENTROPY)
    )


def _short_period_of(result: dict, key_period: str, key_corrected: str) -> float | None:
    corrected = finite_float(result.get(key_corrected))
    if corrected is not None and corrected > 0:
        return corrected
    raw = finite_float(result.get(key_period))
    return raw if (raw is not None and raw > 0) else None


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def resolve_period_consensus(
    *,
    baseline_days: float | None,
    pdm_result: dict | None = None,
    ce_result: dict | None = None,
    long_ls_result: dict | None = None,
    dip_epochs: Sequence[float] | None = None,
    reference_epoch: float | None = None,
    event_period_result: dict | None = None,
    min_period_days: float | None = None,
    max_period_days: float | None = None,
) -> ConsensusResult:
    """Fold PDM / CE / long-P LS / event-period + event epochs into a decision.

    The function is intentionally free of side effects and does not read light
    curves; every metric it needs must be supplied by the caller. This keeps
    it fast (call it 10^4 times/run) and trivially testable.

    Parameters
    ----------
    baseline_days:
        Time span of the light curve in days. Used for baseline-cycle
        confidence gating.
    pdm_result, ce_result:
        Per-method result dicts as produced by ``compute_pdm_stats`` /
        ``compute_ce_stats``.
    long_ls_result:
        Dict as produced by ``long_period_ls_search``. Must include
        ``long_ls_top_periods_days`` and ``long_ls_is_significant``.
    dip_epochs:
        Optional sequence of dip-center JDs. Used to score harmonic-ladder
        candidates by phase concentration (Rayleigh R). If ``None`` or empty
        we fall back to LS power ordering.
    reference_epoch:
        Optional epoch reference. When omitted the phase is computed modulo
        ``period_days`` so absolute origin is irrelevant.
    event_period_result:
        Optional dict from ``malca.stv.event_period.event_based_period``. When
        high-confidence (n>=3, stable GCD), it can win consensus outright;
        otherwise its period seeds the long-P ladder.
    min_period_days, max_period_days:
        Optional inclusive selection bounds. Every PDM, CE, long-LS,
        event-spacing, and harmonic-ladder candidate must lie inside this
        window. Omitting both preserves the production consensus behavior.
    """
    pdm_result = pdm_result or {}
    ce_result = ce_result or {}
    long_ls_result = long_ls_result or {}
    event_period_result = event_period_result or {}
    baseline = finite_float(baseline_days)

    min_period, max_period = _normalize_period_window(
        min_period_days,
        max_period_days,
    )

    pdm_period_raw = _short_period_of(
        pdm_result,
        "pdm_period",
        "pdm_corrected_period",
    )
    ce_period_raw = _short_period_of(
        ce_result,
        "ce_period",
        "ce_corrected_period",
    )
    pdm_period = (
        pdm_period_raw
        if _period_in_window(
            pdm_period_raw,
            min_period_days=min_period,
            max_period_days=max_period,
        )
        else None
    )
    ce_period = (
        ce_period_raw
        if _period_in_window(
            ce_period_raw,
            min_period_days=min_period,
            max_period_days=max_period,
        )
        else None
    )
    pdm_ok = bool(_pdm_supported(pdm_result) and pdm_period is not None)
    ce_ok = bool(_ce_supported(ce_result) and ce_period is not None)

    long_ls_significant = bool(long_ls_result.get("long_ls_is_significant"))
    long_ls_period_raw = finite_float(long_ls_result.get("long_ls_period_days"))
    long_ls_period = (
        long_ls_period_raw
        if _period_in_window(
            long_ls_period_raw,
            min_period_days=min_period,
            max_period_days=max_period,
        )
        else None
    )
    long_ls_fap = finite_float(long_ls_result.get("long_ls_fap_bootstrap"))
    top_periods_raw = long_ls_result.get("long_ls_top_periods_days") or []
    top_powers_raw = long_ls_result.get("long_ls_top_powers") or []
    top_periods: list[float] = []
    top_powers: list[float | None] = []
    for index, raw_period in enumerate(top_periods_raw):
        period = finite_float(raw_period)
        if not _period_in_window(
            period,
            min_period_days=min_period,
            max_period_days=max_period,
        ):
            continue
        top_periods.append(float(period))
        top_powers.append(
            finite_float(top_powers_raw[index])
            if index < len(top_powers_raw)
            else None
        )

    dip_epochs_arr = np.asarray(list(dip_epochs) if dip_epochs is not None else [], dtype=float)
    dip_epochs_arr = dip_epochs_arr[np.isfinite(dip_epochs_arr)]

    # Compute event-period on the fly when epochs are available but the caller
    # did not supply a precomputed result.
    if not event_period_result and dip_epochs_arr.size >= 1:
        try:
            from malca.stv.event_period import event_based_period

            event_period_result = event_based_period(
                dip_epochs_arr.tolist(),
                baseline_days=baseline,
            )
        except Exception:
            event_period_result = {}

    event_period_raw = finite_float(event_period_result.get("event_period_days"))
    event_period = (
        event_period_raw
        if _period_in_window(
            event_period_raw,
            min_period_days=min_period,
            max_period_days=max_period,
        )
        else None
    )
    event_n = int(event_period_result.get("event_period_n_events") or 0)
    event_high = bool(event_period_result.get("event_period_is_high_confidence"))
    event_method = str(event_period_result.get("event_period_method") or "")
    event_rel_std = finite_float(event_period_result.get("event_period_rel_std"))

    evidence: dict[str, object] = {
        "pdm_supported": bool(pdm_ok),
        "ce_supported": bool(ce_ok),
        "pdm_period_days": pdm_period,
        "ce_period_days": ce_period,
        "long_ls_significant": bool(long_ls_significant),
        "long_ls_period_days": long_ls_period,
        "long_ls_fap_bootstrap": long_ls_fap,
        "long_ls_top_periods_days": list(top_periods),
        "dip_epochs_n": int(dip_epochs_arr.size),
        "event_period_days": event_period,
        "event_period_n_events": event_n,
        "event_period_method": event_method,
        "event_period_is_high_confidence": event_high,
        "event_period_rel_std": event_rel_std,
        "search_min_period_days": min_period,
        "search_max_period_days": max_period,
        "out_of_window_periods": {
            key: float(period)
            for key, period in (
                ("pdm", pdm_period_raw),
                ("ce", ce_period_raw),
                ("long_ls", long_ls_period_raw),
                ("event_period", event_period_raw),
            )
            if period is not None
            and not _period_in_window(
                period,
                min_period_days=min_period,
                max_period_days=max_period,
            )
        },
    }

    # --- Branch 0: high-confidence event-period (n>=3, stable GCD) ----------
    if event_high and event_period is not None and event_period > 0 and baseline is not None:
        cycles = baseline / event_period if event_period > 0 else float("nan")
        # Cap at high only when cycle count also supports it.
        if np.isfinite(cycles) and cycles >= MIN_BASELINE_CYCLES_HIGH:
            confidence = HIGH
        else:
            confidence = TENTATIVE
        return ConsensusResult(
            period_consensus_days=float(event_period),
            period_confidence=confidence,
            period_method="event_period",
            period_baseline_cycles=float(cycles) if np.isfinite(cycles) else float("nan"),
            period_confidence_reason=(
                f"event_period {event_method} n={event_n} rel_std={event_rel_std:.3f}"
                if event_rel_std is not None
                else f"event_period {event_method} n={event_n}"
            ),
            period_evidence=evidence,
        )

    # --- Branch 1: long-P LS significant OR >=2 dip events available -------
    # Rationale: PDM/CE fixed at short periods routinely lock onto ultra-high
    # harmonics of a true long-recurrence signal (e.g. PDM=6d, CE=3d for a
    # 2000d dipper). Bootstrap-FAP-validated long-P LS is the strongest
    # evidence when it fires, but even a marginal LS peak combined with two
    # coherent dip epochs is enough to identify the fundamental via phase
    # concentration. Short-P branches remain the fallback when neither line
    # of evidence is available.
    event_derived_seeds = [
        period
        for period in _event_derived_period_seeds(dip_epochs_arr)
        if _period_in_window(
            period,
            min_period_days=min_period,
            max_period_days=max_period,
        )
    ]
    if event_period is not None and event_period > 0:
        # Soft prior / non-high-confidence event period seeds the ladder.
        event_derived_seeds = _combine_seed_periods(event_derived_seeds, [event_period])
    have_long_evidence = bool(long_ls_significant and top_periods) or bool(event_derived_seeds)
    if have_long_evidence and baseline is not None:
        # Seed set combines LS top peaks with event-Δt so we can find the true
        # fundamental even when LS itself misses it (spectrally rich Gaussian
        # dips often place LS peaks at P/k).
        combined_seeds = _combine_seed_periods(top_periods, event_derived_seeds)
        # With events, expand upward harmonics so phase concentration can pick
        # the true fundamental. Without events we cannot distinguish P from kP,
        # so keep only the raw LS peaks (factor=1) and let LS power decide.
        expand_factors: tuple[float, ...] = (
            LONG_LS_LADDER_FACTORS if dip_epochs_arr.size >= 2 else (1.0,)
        )
        ladder = [
            entry
            for entry in build_long_ls_harmonic_ladder(
                combined_seeds,
                baseline_days=baseline,
                factors=expand_factors,
                min_period_days=(
                    float(min_period)
                    if min_period is not None
                    else 10.0
                ),
            )
            if _period_in_window(
                entry.get("period"),
                min_period_days=min_period,
                max_period_days=max_period,
            )
        ]
        # Score ladder candidates. If we have >=2 events use phase concentration,
        # otherwise fall back to LS power via the seed peak.
        best = _score_long_ls_ladder(
            ladder=ladder,
            top_periods=top_periods,
            top_powers=top_powers,
            dip_epochs=dip_epochs_arr,
            reference_epoch=reference_epoch,
        )
        if best is not None:
            chosen_period = float(best["period"])
            cycles = baseline / chosen_period if chosen_period > 0 else float("nan")
            evidence["long_ls_ladder"] = ladder
            evidence["selected_ladder_entry"] = best
            evidence["event_derived_seeds_days"] = event_derived_seeds
            evidence["long_ls_significant"] = bool(long_ls_significant)
            # Prefer event_fold_quality R2 when available for confidence.
            fold_q = event_fold_quality(
                dip_epochs_arr, chosen_period, reference_epoch=reference_epoch
            )
            evidence["event_fold_quality"] = fold_q
            event_R = best.get("event_R")
            if finite_float(fold_q.get("R")) is not None:
                event_R = fold_q["R"]
            confidence, reason = _classify_long_ls_confidence(
                long_ls_fap=long_ls_fap,
                long_ls_significant=long_ls_significant,
                event_R=event_R,
                baseline_cycles=cycles,
                dip_n=int(dip_epochs_arr.size),
            )
            method_label = best.get("method_label", "long_ls")
            # If we entered this branch only because of events (LS not
            # significant and no LS seeds), reflect that in the label so
            # downstream analysis can distinguish it.
            if not long_ls_significant and not top_periods:
                method_label = "event_period"
            return ConsensusResult(
                period_consensus_days=chosen_period,
                period_confidence=confidence,
                period_method=method_label,
                period_baseline_cycles=float(cycles) if np.isfinite(cycles) else float("nan"),
                period_confidence_reason=reason,
                period_evidence=evidence,
            )

    # --- Branch 2: short-period PDM & CE agree ------------------------------
    if pdm_ok and ce_ok and pdm_period and ce_period and _periods_agree(pdm_period, ce_period):
        chosen = pdm_period if pdm_period <= ce_period else ce_period
        cycles = baseline / chosen if baseline and chosen > 0 else float("nan")
        confidence = HIGH if (np.isfinite(cycles) and cycles >= MIN_BASELINE_CYCLES_HIGH) else TENTATIVE
        return ConsensusResult(
            period_consensus_days=float(chosen),
            period_confidence=confidence,
            period_method="pdm+ce",
            period_baseline_cycles=float(cycles) if np.isfinite(cycles) else float("nan"),
            period_confidence_reason=(
                f"pdm={pdm_period:.4f}d agrees with ce={ce_period:.4f}d, cycles={cycles:.2f}"
                if np.isfinite(cycles)
                else f"pdm={pdm_period:.4f}d agrees with ce={ce_period:.4f}d"
            ),
            period_evidence=evidence,
        )

    # --- Branch 3: single short-period method ------------------------------
    if pdm_ok and pdm_period:
        cycles = baseline / pdm_period if baseline and pdm_period > 0 else float("nan")
        return ConsensusResult(
            period_consensus_days=float(pdm_period),
            period_confidence=TENTATIVE,
            period_method="pdm",
            period_baseline_cycles=float(cycles) if np.isfinite(cycles) else float("nan"),
            period_confidence_reason=(
                f"pdm passes ({pdm_period:.4f}d) but ce disagrees or fails support"
            ),
            period_evidence=evidence,
        )
    if ce_ok and ce_period:
        cycles = baseline / ce_period if baseline and ce_period > 0 else float("nan")
        return ConsensusResult(
            period_consensus_days=float(ce_period),
            period_confidence=TENTATIVE,
            period_method="ce",
            period_baseline_cycles=float(cycles) if np.isfinite(cycles) else float("nan"),
            period_confidence_reason=(
                f"ce passes ({ce_period:.4f}d) but pdm disagrees or fails support"
            ),
            period_evidence=evidence,
        )

    # --- Branch 4: nothing significant -------------------------------------
    reason = "no method reached significance"
    if long_ls_significant and top_periods:
        reason = "long_ls significant but baseline unknown; no ladder built"
    result = _empty_result(reason)
    result.period_evidence = evidence
    return result


def _score_long_ls_ladder(
    *,
    ladder: list[dict[str, float]],
    top_periods: Sequence[float],
    top_powers: Sequence[float | None],
    dip_epochs: np.ndarray,
    reference_epoch: float | None,
) -> dict[str, object] | None:
    """Rank ladder candidates. Uses dip phase concentration when possible."""
    if not ladder:
        return None
    power_by_seed: dict[float, float] = {}
    for seed, power in zip(list(top_periods), list(top_powers)):
        p = finite_float(power)
        if p is None:
            continue
        power_by_seed[float(seed)] = float(p)

    # Which seeds came from the event-Δt path (as opposed to LS peaks)? We
    # prefer these when R ties because event-derived seeds are direct period
    # candidates, whereas LS harmonic-ladder rungs are inferred.
    ls_seed_set: set[float] = set(float(s) for s in top_periods)

    scored: list[dict[str, object]] = []
    use_events = int(dip_epochs.size) >= 2
    for entry in ladder:
        period = float(entry["period"])
        seed = float(entry["seed_period"])
        raw_power = power_by_seed.get(seed, float("nan"))
        seed_is_event_only = not any(
            abs(seed - ls_seed) / max(seed, ls_seed) < 0.01 for ls_seed in ls_seed_set
        )
        R = phase_concentration_R(dip_epochs, period, reference_epoch=reference_epoch) if use_events else float("nan")
        if use_events:
            score = float(R) if np.isfinite(R) else -1.0
            method_label = "long_ls+events"
        else:
            score = float(raw_power) if np.isfinite(raw_power) else -1.0
            method_label = "long_ls"
        scored.append(
            {
                "period": period,
                "seed_period": seed,
                "factor": float(entry["factor"]),
                "ls_power": raw_power,
                "event_R": R,
                "score": score,
                "seed_is_event_only": seed_is_event_only,
                "method_label": method_label,
            }
        )

    # Sort:
    # * event-informed: primary key = round(R, 3) so 0.999 and 1.000 tie only
    #   at true numerical equivalence. Break ties in favor of event-derived
    #   seeds (factor=1 on Δt) because they are direct rather than inferred,
    #   then toward the *larger* period (fundamental preference), and finally
    #   by LS power at the seed peak.
    # * power-only: primary key = LS power at the seed peak; break ties toward
    #   larger period so a factor=2 ladder rung wins over factor=1 only when
    #   its seed has strictly higher power (never numerically).
    if use_events:
        scored.sort(
            key=lambda item: (
                round(float(item.get("score", -1.0)), 3),
                bool(item.get("seed_is_event_only", False)),
                float(item["period"]),
                float(item.get("ls_power") or 0.0),
            ),
            reverse=True,
        )
    else:
        scored.sort(
            key=lambda item: (
                float(item.get("score", -1.0)),
                float(item["period"]),
            ),
            reverse=True,
        )
    return scored[0] if scored else None


def _classify_long_ls_confidence(
    *,
    long_ls_fap: float | None,
    long_ls_significant: bool,
    event_R: object,
    baseline_cycles: float,
    dip_n: int,
) -> tuple[str, str]:
    """Return ``(confidence, reason)`` for a long-P consensus selection.

    ``long_ls_significant`` distinguishes the "LS fired" case from the
    "events-only" case. In the events-only case we cap confidence at
    ``tentative`` because a spurious LS backbone would let us silently claim
    high confidence off a couple of aligned dips.
    """
    fap = finite_float(long_ls_fap)
    R = finite_float(event_R)
    cycles = finite_float(baseline_cycles)

    high_ok_fap = fap is not None and fap <= LONG_LS_HIGH_FAP_THRESHOLD
    high_ok_cycles = cycles is not None and cycles >= MIN_BASELINE_CYCLES_HIGH
    high_ok_events = R is not None and R >= EVENT_PHASE_HIGH_R and dip_n >= 2

    tentative_ok_fap = fap is not None and fap <= LONG_LS_TENTATIVE_FAP_THRESHOLD
    tentative_ok_cycles = cycles is not None and cycles >= MIN_BASELINE_CYCLES_TENTATIVE

    if not long_ls_significant:
        # Events-only path (Branch 1 fired because dip_epochs_arr.size >= 2).
        # Require strong phase alignment for even "tentative".
        if R is None or R < EVENT_PHASE_TENTATIVE_R or dip_n < 2:
            parts = ["events available"]
            if R is not None:
                parts.append(f"R={R:.2f}")
            parts.append("phase-alignment insufficient")
            return NONE, "; ".join(parts)
        parts = [f"events R={R:.2f} align on {dip_n} epochs"]
        if fap is not None:
            parts.append(f"long_ls FAP={fap:.2f} (not significant)")
        if cycles is not None:
            parts.append(f"cycles={cycles:.2f}")
        return TENTATIVE, "; ".join(parts)

    reasons = []
    if high_ok_events and (high_ok_fap or high_ok_cycles):
        reasons.append(f"events R={R:.2f} align")
        if high_ok_fap:
            reasons.append(f"long_ls FAP={fap:.1e}")
        if high_ok_cycles:
            reasons.append(f"cycles={cycles:.2f}")
        return HIGH, "; ".join(reasons)
    if high_ok_fap and high_ok_cycles:
        return HIGH, f"long_ls FAP={fap:.1e} with cycles={cycles:.2f}"
    if tentative_ok_fap and tentative_ok_cycles:
        parts = [f"long_ls FAP={fap:.1e}", f"cycles={cycles:.2f}"]
        if R is not None:
            parts.append(f"events R={R:.2f}")
        return TENTATIVE, "; ".join(parts)
    parts = ["long_ls significant"]
    if fap is not None:
        parts.append(f"FAP={fap:.1e}")
    if cycles is not None:
        parts.append(f"cycles={cycles:.2f}")
    if R is not None:
        parts.append(f"events R={R:.2f}")
    return TENTATIVE, "; ".join(parts)


def _event_derived_period_seeds(dip_epochs: np.ndarray) -> list[float]:
    """Return candidate seed periods derived from pairwise dip Δt.

    For K epochs we return the sorted, deduplicated set of consecutive Δt and
    the total span (largest pairwise separation). These act as fresh seeds for
    the harmonic-ladder search when LS misses the fundamental. The routine
    intentionally does not attempt a GCD-style unfolding; that is what the
    ladder+R scoring does downstream.
    """
    epochs = np.asarray(dip_epochs, dtype=float)
    epochs = epochs[np.isfinite(epochs)]
    if epochs.size < 2:
        return []
    epochs = np.sort(np.unique(epochs))
    seeds: list[float] = []
    for i in range(epochs.size - 1):
        dt = float(epochs[i + 1] - epochs[i])
        if dt > 0:
            seeds.append(dt)
    span = float(epochs[-1] - epochs[0])
    if span > 0:
        seeds.append(span)
    # De-duplicate at 1% tolerance so a pile of near-identical Δt's does not
    # dominate the ladder.
    deduped: list[float] = []
    for s in sorted(seeds, reverse=True):
        if any(abs(s - t) / max(s, t) < 0.01 for t in deduped):
            continue
        deduped.append(s)
    return deduped


def _combine_seed_periods(
    ls_seeds: Sequence[float],
    event_seeds: Sequence[float],
) -> list[float]:
    """Merge LS-derived and event-derived seed periods (dedup at 1%)."""
    combined: list[float] = []
    for s in list(ls_seeds) + list(event_seeds):
        val = finite_float(s)
        if val is None or val <= 0:
            continue
        if any(abs(val - t) / max(val, t) < 0.01 for t in combined):
            continue
        combined.append(float(val))
    return combined


__all__ = [
    "HIGH",
    "NONE",
    "TENTATIVE",
    "ConsensusResult",
    "LONG_LS_LADDER_FACTORS",
    "MIN_BASELINE_CYCLES_HIGH",
    "MIN_BASELINE_CYCLES_TENTATIVE",
    "build_long_ls_harmonic_ladder",
    "event_fold_quality",
    "phase_concentration_R",
    "resolve_period_consensus",
]
