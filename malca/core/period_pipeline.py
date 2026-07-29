"""End-to-end period consensus for a single light curve.

Both the STV post-filter (:mod:`malca.stv.filter`) and the offline backfill
script share this entry point so their outputs are guaranteed to match. The
helper takes raw light-curve arrays plus already-computed short-period
diagnostics (PDM / CE / bootstrap-LS from the standard 0.2-365 d search), runs
the long-period LS discovery stage, and hands everything to
:func:`resolve_period_consensus`.

The result is a flat ``dict`` ready to be dumped into a checkpoint row or a
DB. There is no attempt to preserve legacy ``periodicity_period`` semantics --
the shape of the "canonical" period fields (``period_consensus_days``,
``period_confidence``, ``period_method``, ``period_baseline_cycles``,
``period_confidence_reason``) is authoritative.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from malca.core.event_epochs import (
    DipRunEpoch,
    detect_dip_epochs_lightweight,
    parse_run_epochs_json,
)
from malca.core.period_consensus import ConsensusResult, resolve_period_consensus
from malca.core.stats import long_period_ls_search


LONG_LS_EVIDENCE_KEYS: tuple[str, ...] = (
    "long_ls_period_days",
    "long_ls_peak_power",
    "long_ls_fap_bootstrap",
    "long_ls_baseline_cycles",
    "long_ls_is_significant",
    "long_ls_min_period_days",
    "long_ls_max_period_days",
    "long_ls_n_bootstrap_attempted",
    "long_ls_n_bootstrap_successful",
    "long_ls_status",
    "long_ls_top_periods_days",
    "long_ls_top_powers",
)


def compute_period_consensus_for_lc(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    *,
    pdm_result: dict | None = None,
    ce_result: dict | None = None,
    dip_epochs_json: str | bytes | None = None,
    dip_epochs_override: Sequence[float] | None = None,
    detect_dip_epochs_fallback: bool = True,
    use_events_pipeline: bool = False,
    df_lc: Any = None,
    reference_epoch: float | None = None,
    long_ls_kwargs: dict[str, Any] | None = None,
    dip_detector_kwargs: dict[str, Any] | None = None,
    score_kwargs: dict[str, Any] | None = None,
    event_period_result: dict[str, Any] | None = None,
    min_period_days: float | None = None,
    max_period_days: float | None = None,
) -> dict[str, Any]:
    """Run long-period LS + consensus for one cleaned light curve.

    Parameters
    ----------
    jd, mag, err:
        Cleaned light-curve arrays. ``err`` may be ``None``.
    pdm_result, ce_result:
        Optional PDM / CE dicts (same shape as ``compute_pdm_stats`` and
        ``compute_ce_stats`` return). When absent, consensus falls back to
        long-P LS alone.
    dip_epochs_json:
        Optional serialised ``dip_run_epochs_json`` blob from a previous
        events-pipeline run. When available, its center JDs are used for
        event-informed harmonic arbitration.
    dip_epochs_override:
        Explicit list of dip center JDs. Takes precedence over the JSON blob.
    detect_dip_epochs_fallback:
        When ``True`` and neither ``dip_epochs_json`` nor
        ``dip_epochs_override`` provided any epochs, run a dip detector.
    use_events_pipeline:
        When ``True`` and no JSON/override epochs are available, call the
        full ``score_lightcurve`` path (GP + ``build_runs``) instead of the
        lightweight detector. Prefer this for offline backfill.
    df_lc:
        Optional DataFrame required when ``use_events_pipeline=True``.
    reference_epoch:
        Passed through to :func:`phase_concentration_R` (via consensus).
    long_ls_kwargs, dip_detector_kwargs, score_kwargs:
        Optional per-call overrides.
    event_period_result:
        Optional precomputed ``event_based_period`` result.
    min_period_days, max_period_days:
        Optional inclusive bounds applied to every candidate entering the
        consensus decision. Callers that set these should also give the same
        bounds to the long-LS search through ``long_ls_kwargs``.
    """
    jd_arr = np.asarray(jd, dtype=float)
    mag_arr = np.asarray(mag, dtype=float)
    err_arr = np.asarray(err, dtype=float) if err is not None else None

    long_ls_kwargs = dict(long_ls_kwargs or {})
    dip_detector_kwargs = dict(dip_detector_kwargs or {})
    score_kwargs = dict(score_kwargs or {})

    long_ls_result = long_period_ls_search(jd_arr, mag_arr, err_arr, **long_ls_kwargs)

    dip_epochs, dip_source = _resolve_dip_epochs(
        jd=jd_arr,
        mag=mag_arr,
        err=err_arr,
        override=dip_epochs_override,
        json_blob=dip_epochs_json,
        run_fallback=detect_dip_epochs_fallback,
        use_events_pipeline=use_events_pipeline,
        detector_kwargs=dip_detector_kwargs,
        score_kwargs=score_kwargs,
        df_lc=df_lc,
    )

    baseline_days = _baseline_days(jd_arr)

    if event_period_result is None and dip_epochs:
        try:
            from malca.stv.event_period import event_based_period

            event_period_result = event_based_period(
                dip_epochs,
                baseline_days=baseline_days,
            )
        except Exception:
            event_period_result = None

    consensus = resolve_period_consensus(
        baseline_days=baseline_days,
        pdm_result=pdm_result,
        ce_result=ce_result,
        long_ls_result=long_ls_result,
        dip_epochs=dip_epochs,
        reference_epoch=reference_epoch,
        event_period_result=event_period_result,
        min_period_days=min_period_days,
        max_period_days=max_period_days,
    )

    out: dict[str, Any] = {
        key: long_ls_result.get(key)
        for key in LONG_LS_EVIDENCE_KEYS
        if key in long_ls_result
    }
    out.update(consensus.as_row())
    out["dip_epochs_used"] = list(dip_epochs)
    out["dip_epochs_source"] = dip_source
    out["dip_epochs_count"] = int(len(dip_epochs))
    if event_period_result:
        out["event_period_days"] = event_period_result.get("event_period_days", float("nan"))
        out["event_period_method"] = event_period_result.get("event_period_method", "none")
        out["event_period_n_events"] = event_period_result.get("event_period_n_events", 0)
        out["event_period_is_high_confidence"] = bool(
            event_period_result.get("event_period_is_high_confidence", False)
        )
    return out


def build_consensus_result(payload: dict[str, Any]) -> ConsensusResult:
    """Rebuild a :class:`ConsensusResult` from a dict produced by this helper.

    Useful when a caller has serialised the row to disk and wants the rich
    object back.
    """
    return ConsensusResult(
        period_consensus_days=float(payload.get("period_consensus_days", float("nan"))),
        period_confidence=str(payload.get("period_confidence", "none")),
        period_method=str(payload.get("period_method", "none")),
        period_baseline_cycles=float(payload.get("period_baseline_cycles", float("nan"))),
        period_confidence_reason=str(payload.get("period_confidence_reason", "")),
        period_evidence=dict(payload.get("period_evidence") or {}),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _baseline_days(jd: np.ndarray) -> float | None:
    finite = jd[np.isfinite(jd)]
    if finite.size < 2:
        return None
    baseline = float(finite.max() - finite.min())
    return baseline if baseline > 0 else None


def _resolve_dip_epochs(
    *,
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None,
    override: Sequence[float] | None,
    json_blob: str | bytes | None,
    run_fallback: bool,
    use_events_pipeline: bool,
    detector_kwargs: dict[str, Any],
    score_kwargs: dict[str, Any],
    df_lc: Any,
) -> tuple[list[float], str]:
    """Pick the best available dip epoch list. Returns ``(epochs, source)``."""
    if override is not None:
        cleaned = [float(v) for v in override if np.isfinite(float(v))]
        cleaned.sort()
        if cleaned:
            return cleaned, "override"

    parsed = parse_run_epochs_json(json_blob)
    if parsed:
        centers = _centers(parsed)
        if centers:
            return centers, "json"

    if run_fallback and use_events_pipeline and df_lc is not None:
        from malca.core.event_epochs import detect_dip_epochs_via_events

        detected = detect_dip_epochs_via_events(df_lc, score_kwargs=score_kwargs)
        centers = _centers(detected)
        if centers:
            return centers, "events"

    if run_fallback:
        detected = detect_dip_epochs_lightweight(
            jd,
            mag,
            err,
            **detector_kwargs,
        )
        centers = _centers(detected)
        if centers:
            return centers, "detected"

    return [], "none"


def _centers(epochs: Sequence[DipRunEpoch]) -> list[float]:
    out = [float(e.center_jd) for e in epochs if np.isfinite(e.center_jd)]
    out.sort()
    return out


__all__ = [
    "LONG_LS_EVIDENCE_KEYS",
    "build_consensus_result",
    "compute_period_consensus_for_lc",
]
