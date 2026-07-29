"""End-to-end tests for compute_period_consensus_for_lc."""
from __future__ import annotations

import json

import numpy as np
import pytest

from malca.core.event_epochs import serialize_run_summaries
from malca.core.period_pipeline import (
    LONG_LS_EVIDENCE_KEYS,
    build_consensus_result,
    compute_period_consensus_for_lc,
)


def _make_dipper(
    *,
    dip_centers: tuple[float, ...] = (1000.0, 3011.0),
    baseline_days: float = 3653.0,
    n_points: int = 1500,
    dip_width_days: float = 25.0,
    dip_depth_mag: float = 0.7,
    noise_mag: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(0.0, baseline_days, size=n_points))
    mag = np.full(jd.size, 15.0)
    for center in dip_centers:
        mag += dip_depth_mag * np.exp(-0.5 * ((jd - center) / (dip_width_days / 2.355)) ** 2)
    mag += rng.normal(0.0, noise_mag, size=jd.size)
    err = np.full_like(mag, noise_mag)
    return jd, mag, err


def _make_short_periodic(
    *,
    period_days: float = 3.0,
    baseline_days: float = 500.0,
    n_points: int = 500,
    amplitude_mag: float = 0.2,
    noise_mag: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(0.0, baseline_days, size=n_points))
    mag = 15.0 + amplitude_mag * np.sin(2 * np.pi * jd / period_days)
    mag += rng.normal(0.0, noise_mag, size=jd.size)
    err = np.full_like(mag, noise_mag)
    return jd, mag, err


# ---------------------------------------------------------------------------
# Basic shape / plumbing
# ---------------------------------------------------------------------------

def test_returns_all_long_ls_keys() -> None:
    jd, mag, err = _make_short_periodic()
    out = compute_period_consensus_for_lc(jd, mag, err)
    for key in LONG_LS_EVIDENCE_KEYS:
        assert key in out
    for key in (
        "period_consensus_days",
        "period_confidence",
        "period_method",
        "period_baseline_cycles",
        "period_confidence_reason",
        "period_evidence",
        "dip_epochs_used",
        "dip_epochs_source",
        "dip_epochs_count",
    ):
        assert key in out


def test_build_consensus_result_round_trip() -> None:
    jd, mag, err = _make_short_periodic()
    payload = compute_period_consensus_for_lc(jd, mag, err)
    result = build_consensus_result(payload)
    assert result.period_confidence == payload["period_confidence"]
    assert result.period_method == payload["period_method"]


# ---------------------------------------------------------------------------
# Long-P dipper — the driving science case
# ---------------------------------------------------------------------------

def test_long_period_dipper_uses_detected_events() -> None:
    jd, mag, err = _make_dipper(dip_centers=(1000.0, 3011.0), baseline_days=3653.0)
    out = compute_period_consensus_for_lc(jd, mag, err, detect_dip_epochs_fallback=True)

    assert out["dip_epochs_count"] >= 2
    assert out["dip_epochs_source"] in {"detected", "json", "override"}
    # Method should reflect the long-period branch (event-arbitrated)
    assert out["period_method"] in {"long_ls", "long_ls+events", "event_period"}
    period = out["period_consensus_days"]
    assert np.isfinite(period)
    # Consensus period should match Δt = 2011 d within 10%
    assert abs(period - 2011.0) / 2011.0 < 0.10


def test_long_period_dipper_with_persisted_epochs_matches_detection() -> None:
    """Providing dip_epochs_json should give the same consensus as detection."""
    jd, mag, err = _make_dipper()
    summaries = [
        {"start_jd": 985.0, "end_jd": 1015.0, "run_max": 6.0, "n_points": 4, "kept": True},
        {"start_jd": 2995.0, "end_jd": 3025.0, "run_max": 5.5, "n_points": 4, "kept": True},
    ]
    blob = serialize_run_summaries(summaries)
    assert blob is not None

    out = compute_period_consensus_for_lc(
        jd,
        mag,
        err,
        dip_epochs_json=blob,
        detect_dip_epochs_fallback=False,
    )
    assert out["dip_epochs_source"] == "json"
    assert out["dip_epochs_count"] == 2
    assert out["period_method"] in {"long_ls", "long_ls+events", "event_period"}
    assert abs(out["period_consensus_days"] - 2011.0) / 2011.0 < 0.10


def test_override_takes_precedence_over_json() -> None:
    jd, mag, err = _make_dipper()
    blob = serialize_run_summaries(
        [{"start_jd": 100.0, "end_jd": 110.0, "kept": True}]
    )
    out = compute_period_consensus_for_lc(
        jd, mag, err,
        dip_epochs_json=blob,
        dip_epochs_override=[1000.0, 3011.0],
        detect_dip_epochs_fallback=False,
    )
    assert out["dip_epochs_source"] == "override"
    assert out["dip_epochs_used"] == [1000.0, 3011.0]


def test_no_dip_epochs_when_fallback_disabled_and_no_json() -> None:
    jd, mag, err = _make_dipper()
    out = compute_period_consensus_for_lc(
        jd, mag, err,
        detect_dip_epochs_fallback=False,
    )
    assert out["dip_epochs_source"] == "none"
    assert out["dip_epochs_count"] == 0


# ---------------------------------------------------------------------------
# Short-period sinusoid: LS should not fabricate a long period
# ---------------------------------------------------------------------------

def test_short_period_sinusoid_stays_short() -> None:
    jd, mag, err = _make_short_periodic(period_days=3.0, baseline_days=500.0)
    # Emulate a PDM detection so consensus has a short-P branch to consider
    pdm_result = {
        "pdm_period": 3.0,
        "pdm_corrected_period": 3.0,
        "pdm_min_theta": 0.05,
        "pdm_snr": 25.0,
        "pdm_bootstrap_sig": 1e-4,
        "pdm_is_significant": True,
    }
    ce_result = {
        "ce_period": 3.0,
        "ce_corrected_period": 3.0,
        "ce_min_entropy": 0.2,
        "ce_snr": 20.0,
        "ce_bootstrap_sig": 1e-4,
        "ce_is_significant": True,
    }
    out = compute_period_consensus_for_lc(
        jd, mag, err,
        pdm_result=pdm_result,
        ce_result=ce_result,
        detect_dip_epochs_fallback=False,
    )
    assert out["period_method"] in {"pdm+ce", "pdm", "ce", "short_and_long_agree"}
    assert abs(out["period_consensus_days"] - 3.0) / 3.0 < 0.05


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_lightcurve_returns_none_confidence() -> None:
    jd = np.array([])
    mag = np.array([])
    err = np.array([])
    out = compute_period_consensus_for_lc(jd, mag, err, detect_dip_epochs_fallback=False)
    assert out["period_confidence"] == "none"


def test_bad_dip_epochs_json_is_gracefully_ignored() -> None:
    jd, mag, err = _make_short_periodic()
    out = compute_period_consensus_for_lc(
        jd, mag, err,
        dip_epochs_json="not valid json",
        detect_dip_epochs_fallback=False,
    )
    assert out["dip_epochs_source"] == "none"
    assert out["dip_epochs_count"] == 0
