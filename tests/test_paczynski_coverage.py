"""Tests for Paczynski τ-coverage scoring (microlensing LC quality)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.microlensing import (
    PAC_COVERAGE_N_BINS,
    PAC_COVERAGE_TAU_MAX,
    _paczynski_weighted_coverage,
)


def _minimal_pac(*, A0: float = 1.8, t0: float = 61234.0, tE: float = 35.0) -> dict:
    return {
        'success': True,
        'params': np.array([A0, t0, tE, 14.0, 0.0], dtype=float),
        't_ref': float(t0),
    }


def test_uniform_sampling_high_coverage():
    t0, tE = 61234.0, 35.0
    pac = _minimal_pac(t0=t0, tE=tE)
    # Dense sampling across Paczynski τ window
    jd = np.linspace(t0 - PAC_COVERAGE_TAU_MAX * tE, t0 + PAC_COVERAGE_TAU_MAX * tE, 800)
    out = _paczynski_weighted_coverage(jd, pac)
    assert np.isfinite(out['paczynski_tau_coverage_score'])
    assert float(out['paczynski_tau_coverage_score']) > 0.92
    assert int(out['paczynski_coverage_n_bins_hit']) >= int(0.9 * PAC_COVERAGE_N_BINS)


def test_gap_at_peak_hurts_more_than_far_shoulder_gap():
    t0, tE = 61234.0, 40.0
    pac = _minimal_pac(t0=t0, tE=tE)
    span = PAC_COVERAGE_TAU_MAX * tE
    jd_dense = np.linspace(t0 - span, t0 + span, 900)

    # Remove points near τ=0 (peak region)
    tau = (jd_dense - t0) / tE
    peak_mask = np.abs(tau) >= 0.35
    jd_peak_hole = jd_dense[peak_mask]

    # Same fraction removed but away from core: drop |τ|∈[4, 5] only
    shoulder_mask = (np.abs(tau) < 4.0) | (np.abs(tau) > 5.0)
    jd_shoulder_hole = jd_dense[shoulder_mask]

    s_peak = float(_paczynski_weighted_coverage(jd_peak_hole, pac)['paczynski_tau_coverage_score'])
    s_shoulder = float(_paczynski_weighted_coverage(jd_shoulder_hole, pac)['paczynski_tau_coverage_score'])
    assert s_peak < s_shoulder


def test_no_pac_success_returns_nan_score():
    pac = {'success': False, 'params': np.array([1.8, 60000.0, 30.0, 14.0, 0.0])}
    out = _paczynski_weighted_coverage(np.array([60000.0, 60010.0]), pac)
    assert not np.isfinite(out['paczynski_tau_coverage_score'])


def test_weighted_gap_larger_when_contiguous_hole_at_core():
    t0, tE = 55000.0, 25.0
    pac = _minimal_pac(t0=t0, tE=tE)
    span = PAC_COVERAGE_TAU_MAX * tE
    jd_full = np.linspace(t0 - span, t0 + span, 600)
    tau = (jd_full - t0) / tE
    jd_hole = jd_full[np.abs(tau) >= 0.5]

    g_full = _paczynski_weighted_coverage(jd_full, pac)['paczynski_coverage_max_weighted_gap']
    g_hole = _paczynski_weighted_coverage(jd_hole, pac)['paczynski_coverage_max_weighted_gap']
    assert np.isfinite(g_full) and np.isfinite(g_hole)
    assert float(g_hole) > float(g_full) + 0.01
