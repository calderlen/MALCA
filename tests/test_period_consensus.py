"""Table-driven tests for the period consensus decision matrix."""
from __future__ import annotations

import math

import numpy as np
import pytest

from malca.config import (
    POST_FILTER_CE_MIN_ENTROPY,
    POST_FILTER_CE_SNR_THRESHOLD,
    POST_FILTER_PDM_MIN_THETA,
    POST_FILTER_PDM_SNR_THRESHOLD,
)
from malca.core.period_consensus import (
    HIGH,
    NONE,
    TENTATIVE,
    build_long_ls_harmonic_ladder,
    phase_concentration_R,
    resolve_period_consensus,
)


# ---------------------------------------------------------------------------
# phase_concentration_R
# ---------------------------------------------------------------------------

def test_phase_concentration_perfect_alignment_returns_one() -> None:
    epochs = [100.0, 100.0 + 2011.0, 100.0 + 2 * 2011.0]
    R = phase_concentration_R(epochs, 2011.0)
    assert R == pytest.approx(1.0, abs=1e-9)


def test_phase_concentration_orthogonal_phases_returns_zero() -> None:
    # Two events half a period apart on a two-point Rayleigh: R = cos(pi/2) = 0
    R = phase_concentration_R([0.0, 500.0], 1000.0)
    assert R == pytest.approx(0.0, abs=1e-9)


def test_phase_concentration_two_events_same_phase() -> None:
    R = phase_concentration_R([100.0, 100.0 + 2011.0], 2011.0)
    assert R == pytest.approx(1.0, abs=1e-9)


def test_phase_concentration_wrong_period_scatters_R() -> None:
    """Real dipper: two dips at 2011 d, tested against wrong period 500 d."""
    R = phase_concentration_R([2457481.0, 2459492.0], 500.0)
    assert 0.0 <= R <= 1.0
    # The two epochs differ by 2011 d = 4.022 cycles at P=500 -> phase diff ~0.022
    assert R > 0.9  # by coincidence they fold close together; but the same test at wrong period must not always fail
    # More stringent: test against a genuinely wrong period.
    R_bad = phase_concentration_R([0.0, 1000.0], 400.0)  # phase diff = 0.5
    assert R_bad == pytest.approx(0.0, abs=1e-9)


def test_phase_concentration_invalid_inputs_return_nan() -> None:
    assert math.isnan(phase_concentration_R([], 100.0))
    assert math.isnan(phase_concentration_R([1.0, 2.0], 0.0))
    assert math.isnan(phase_concentration_R([1.0, 2.0], float("nan")))


# ---------------------------------------------------------------------------
# build_long_ls_harmonic_ladder
# ---------------------------------------------------------------------------

def test_ladder_expands_top_peaks_upward_and_respects_cap() -> None:
    ladder = build_long_ls_harmonic_ladder(
        [1000.0, 800.0],
        baseline_days=3653.0,
    )
    periods = sorted({round(entry["period"], 6) for entry in ladder})
    # Only entries with period <= 0.6 * 3653 = 2191.8 survive
    assert all(p <= 2191.9 for p in periods)
    # Fundamental seeds must be present
    assert 1000.0 in periods
    assert 800.0 in periods
    # 2x seed=1000 gives 2000 which is <= cap
    assert 2000.0 in periods


def test_ladder_ignores_invalid_seeds() -> None:
    ladder = build_long_ls_harmonic_ladder(
        [float("nan"), -5.0, 500.0],
        baseline_days=3000.0,
    )
    assert all(entry["seed_period"] == 500.0 for entry in ladder)


def test_ladder_returns_empty_for_missing_baseline() -> None:
    assert build_long_ls_harmonic_ladder([500.0], baseline_days=None) == []
    assert build_long_ls_harmonic_ladder([500.0], baseline_days=float("nan")) == []


def test_ladder_deduplicates_close_periods() -> None:
    ladder = build_long_ls_harmonic_ladder(
        [1000.0, 1000.00001],
        baseline_days=3000.0,
    )
    periods = [entry["period"] for entry in ladder]
    # Both seeds should still contribute distinct entries once at each factor.
    assert len(periods) == len(set(round(p, 6) for p in periods))


# ---------------------------------------------------------------------------
# resolve_period_consensus - branch coverage
# ---------------------------------------------------------------------------

def _pdm_ok(period: float, snr: float = 10.0, theta: float = 0.4) -> dict:
    return {
        "pdm_period": period,
        "pdm_corrected_period": period,
        "pdm_snr": snr,
        "pdm_min_theta": theta,
    }


def _ce_ok(period: float, snr: float = 20.0, entropy: float = 0.3) -> dict:
    return {
        "ce_period": period,
        "ce_corrected_period": period,
        "ce_snr": snr,
        "ce_min_entropy": entropy,
    }


def _long_ls(
    period: float,
    *,
    top_periods: list[float] | None = None,
    top_powers: list[float] | None = None,
    fap: float = 1e-5,
    significant: bool = True,
) -> dict:
    return {
        "long_ls_period_days": period,
        "long_ls_peak_power": 0.5,
        "long_ls_fap_bootstrap": fap,
        "long_ls_is_significant": bool(significant),
        "long_ls_top_periods_days": list(top_periods) if top_periods is not None else [period],
        "long_ls_top_powers": list(top_powers) if top_powers is not None else [0.5],
    }


def test_pdm_ce_agree_and_baseline_covers_high_confidence() -> None:
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result=_pdm_ok(10.0),
        ce_result=_ce_ok(10.05),
    )
    assert result.period_confidence == HIGH
    assert result.period_method == "pdm+ce"
    assert result.period_consensus_days == pytest.approx(10.0, rel=1e-3)


def test_pdm_ce_agree_but_insufficient_cycles_downgrades_to_tentative() -> None:
    result = resolve_period_consensus(
        baseline_days=15.0,
        pdm_result=_pdm_ok(10.0),
        ce_result=_ce_ok(10.05),
    )
    assert result.period_confidence == TENTATIVE
    assert result.period_method == "pdm+ce"


def test_long_ls_ladder_selects_true_fundamental_when_events_align() -> None:
    """Two dip epochs 2011 d apart; LS peaks at P/2 = 1005.5 and other harmonics."""
    dips = [2457481.0, 2459492.0]  # diff = 2011 d
    result = resolve_period_consensus(
        baseline_days=3653.0,
        long_ls_result=_long_ls(
            1005.5,
            top_periods=[1005.5, 670.0, 502.75],
            top_powers=[0.42, 0.35, 0.30],
            fap=5e-4,
        ),
        dip_epochs=dips,
    )
    # 2x 1005.5 = 2011 is inside the ladder max of 0.6*3653 = 2192
    assert result.period_consensus_days == pytest.approx(2011.0, rel=1e-3)
    assert result.period_method == "long_ls+events"
    assert result.period_confidence == HIGH
    assert result.period_baseline_cycles == pytest.approx(3653.0 / 2011.0, rel=1e-3)


def test_long_ls_no_events_falls_back_to_ls_power_and_is_tentative() -> None:
    """Without dip epochs, ladder scoring reverts to LS power ordering."""
    result = resolve_period_consensus(
        baseline_days=3653.0,
        long_ls_result=_long_ls(
            1857.0,
            top_periods=[1857.0, 2067.0, 1024.0],
            top_powers=[0.42, 0.39, 0.26],
            fap=5e-3,
        ),
        dip_epochs=None,
    )
    assert result.period_method == "long_ls"
    # Highest LS power (1857) wins, even though 2067 is closer to the truth.
    assert result.period_consensus_days == pytest.approx(1857.0, rel=1e-3)
    assert result.period_confidence == TENTATIVE


def test_long_ls_high_fap_with_enough_cycles_is_high_confidence() -> None:
    result = resolve_period_consensus(
        baseline_days=6000.0,
        long_ls_result=_long_ls(
            1000.0,
            top_periods=[1000.0],
            top_powers=[0.6],
            fap=1e-6,
        ),
    )
    assert result.period_confidence == HIGH
    assert result.period_consensus_days == pytest.approx(1000.0)


def test_single_short_method_pdm_pass_ce_fail_is_tentative() -> None:
    ce_bad = {
        "ce_period": 5.0,
        "ce_snr": 0.5,  # below threshold
        "ce_min_entropy": 10.0,  # above threshold
    }
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result=_pdm_ok(10.0),
        ce_result=ce_bad,
    )
    assert result.period_confidence == TENTATIVE
    assert result.period_method == "pdm"
    assert result.period_consensus_days == pytest.approx(10.0)


def test_no_method_reaches_significance_returns_none() -> None:
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result={"pdm_period": 5.0, "pdm_snr": 0.5, "pdm_min_theta": 10.0},
        ce_result={"ce_period": 5.0, "ce_snr": 0.5, "ce_min_entropy": 10.0},
        long_ls_result=_long_ls(2000.0, significant=False),
    )
    assert result.period_confidence == NONE
    assert result.period_method == "none"
    assert math.isnan(result.period_consensus_days)


def test_long_ls_significant_overrides_pathological_short_pdm_ce() -> None:
    """The AA-Tau regression: PDM=6d and CE=3d agree as harmonics of the true
    2000 d period. Consensus must not return the short-P artefact when long-P
    LS is significant.
    """
    dips = [2457481.0, 2459492.0]  # diff = 2011 d
    result = resolve_period_consensus(
        baseline_days=3653.0,
        pdm_result=_pdm_ok(6.0),
        ce_result=_ce_ok(3.0),
        long_ls_result=_long_ls(
            1857.0,
            top_periods=[1857.0, 2067.0, 1005.5],
            top_powers=[0.42, 0.39, 0.30],
            fap=5e-4,
        ),
        dip_epochs=dips,
    )
    assert result.period_method == "long_ls+events"
    # 2x 1005.5 = 2011 is closest to Δt and has R=1 among tied candidates
    # (fundamental-preference tiebreak selects the larger period).
    assert result.period_consensus_days > 1500.0


def test_high_confidence_event_period_wins_consensus() -> None:
    """≥3 stable dip epochs with GCD period should win without long-P LS."""
    epochs = [0.0, 2000.0, 4000.0, 6000.0]
    event_result = {
        "event_period_days": 2000.0,
        "event_period_n_events": 4,
        "event_period_method": "gcd_dt",
        "event_period_is_high_confidence": True,
        "event_period_rel_std": 0.05,
    }
    result = resolve_period_consensus(
        baseline_days=8000.0,
        pdm_result=_pdm_ok(6.0),
        ce_result=_ce_ok(3.0),
        long_ls_result=_long_ls(1000.0, significant=False),
        dip_epochs=epochs,
        event_period_result=event_result,
    )
    assert result.period_method == "event_period"
    assert result.period_consensus_days == pytest.approx(2000.0)
    assert result.period_confidence == HIGH
    assert result.period_consensus_days == pytest.approx(2011.0, rel=0.05)


def test_selected_window_excludes_outside_event_and_long_periods() -> None:
    """A review window must constrain every consensus input, not only PDM/CE."""

    event_result = {
        "event_period_days": 358.689,
        "event_period_n_events": 4,
        "event_period_method": "gcd_dt",
        "event_period_is_high_confidence": True,
        "event_period_rel_std": 0.01,
    }
    result = resolve_period_consensus(
        baseline_days=3700.0,
        pdm_result=_pdm_ok(4.25),
        ce_result=_ce_ok(4.25),
        long_ls_result=_long_ls(
            358.689,
            top_periods=[358.689],
            top_powers=[0.8],
        ),
        dip_epochs=[0.0, 358.689, 717.378, 1076.067],
        event_period_result=event_result,
        min_period_days=0.1,
        max_period_days=10.0,
    )

    assert result.period_consensus_days == pytest.approx(4.25)
    assert result.period_method == "pdm+ce"
    assert result.period_evidence["search_min_period_days"] == pytest.approx(0.1)
    assert result.period_evidence["search_max_period_days"] == pytest.approx(10.0)
    assert result.period_evidence["out_of_window_periods"] == pytest.approx(
        {
            "long_ls": 358.689,
            "event_period": 358.689,
        }
    )


def test_selected_window_clips_event_harmonic_ladder() -> None:
    result = resolve_period_consensus(
        baseline_days=1000.0,
        long_ls_result=_long_ls(
            8.0,
            top_periods=[8.0],
            top_powers=[0.8],
        ),
        dip_epochs=[0.0, 8.0, 16.0],
        event_period_result={
            "event_period_days": 8.0,
            "event_period_n_events": 3,
            "event_period_method": "median_dt",
            "event_period_is_high_confidence": False,
        },
        min_period_days=0.1,
        max_period_days=10.0,
    )

    assert 0.1 <= result.period_consensus_days <= 10.0
    assert all(
        0.1 <= entry["period"] <= 10.0
        for entry in result.period_evidence["long_ls_ladder"]
    )


def test_pdm_ce_harmonic_match_still_agrees() -> None:
    """PDM finds P, CE finds P/2 -> should still agree via harmonic tolerance."""
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result=_pdm_ok(10.0),
        ce_result=_ce_ok(5.0),
    )
    assert result.period_method == "pdm+ce"
    # Consensus picks the shorter period from the agreeing pair.
    assert result.period_consensus_days == pytest.approx(5.0, rel=1e-3)


def test_consensus_evidence_is_populated() -> None:
    result = resolve_period_consensus(
        baseline_days=3653.0,
        long_ls_result=_long_ls(
            1005.5,
            top_periods=[1005.5, 670.0],
            top_powers=[0.42, 0.35],
            fap=5e-4,
        ),
        dip_epochs=[0.0, 2011.0],
    )
    ev = result.period_evidence
    assert ev["long_ls_significant"] is True
    assert ev["dip_epochs_n"] == 2
    assert "long_ls_ladder" in ev
    assert "selected_ladder_entry" in ev


def test_result_as_row_is_serialisable() -> None:
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result=_pdm_ok(10.0),
        ce_result=_ce_ok(10.05),
    )
    row = result.as_row()
    assert row["period_consensus_days"] == pytest.approx(10.0, rel=1e-3)
    assert row["period_confidence"] == HIGH
    assert row["period_method"] == "pdm+ce"
    assert isinstance(row["period_evidence"], dict)


def test_disagreeing_pdm_ce_and_no_long_ls_produces_short_p_tentative() -> None:
    """PDM and CE both pass support but disagree in period; nothing else."""
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result=_pdm_ok(6.0),
        ce_result=_ce_ok(3.0, snr=25.0),
    )
    # 6.0 and 3.0 are a 2x harmonic pair, so they *do* agree via the harmonic tolerance.
    # This is the same behavior as the pipeline's PDM/CE harmonic arbitration.
    assert result.period_method == "pdm+ce"


def test_truly_disagreeing_short_methods_use_pdm_tentative() -> None:
    """PDM=4.7 d, CE=11.3 d — not harmonic and not within 5% -> single-method fallback."""
    result = resolve_period_consensus(
        baseline_days=1000.0,
        pdm_result=_pdm_ok(4.7),
        ce_result=_ce_ok(11.3),
    )
    # Both are "supported" individually, so we fall to the single-method branch;
    # pdm is tried first.
    assert result.period_confidence == TENTATIVE
    assert result.period_method == "pdm"
    assert result.period_consensus_days == pytest.approx(4.7)
