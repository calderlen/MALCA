from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.stv.score import _find_runs, compute_event_score


def _scored_dip(offset: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    jd = np.linspace(0.0, 80.0, 161)
    baseline = np.full_like(jd, 14.0 + offset)
    dip = 0.45 * np.exp(-0.5 * ((jd - 40.0) / 5.0) ** 2)
    quiet_structure = 0.01 * np.sin(jd / 3.0)
    df = pd.DataFrame(
        {
            "JD": jd,
            "mag": baseline + quiet_structure + dip,
            "error": np.full_like(jd, 0.02),
        }
    )
    return df, baseline


def test_event_score_is_invariant_to_absolute_magnitude_shift() -> None:
    df_a, baseline_a = _scored_dip(0.0)
    df_b, baseline_b = _scored_dip(5.0)

    score_a, events_a = compute_event_score(
        df_a,
        event_type="dip",
        sigma_threshold=1.0,
        min_fwhm_days=1.0,
        baseline_mags=baseline_a,
    )
    score_b, events_b = compute_event_score(
        df_b,
        event_type="dip",
        sigma_threshold=1.0,
        min_fwhm_days=1.0,
        baseline_mags=baseline_b,
    )

    assert np.isfinite(score_a)
    assert score_b == pytest.approx(score_a, rel=1e-9, abs=1e-9)
    assert [event.delta for event in events_b] == pytest.approx(
        [event.delta for event in events_a]
    )
    assert all(event.dof > 0 for event in events_a if event.valid)
    assert all(np.isfinite(event.chi2_reduced) for event in events_a if event.valid)


def test_event_score_rejects_misaligned_baseline() -> None:
    df, baseline = _scored_dip()

    with pytest.raises(ValueError, match="position-aligned"):
        compute_event_score(df, baseline_mags=baseline[:-1])


def test_event_score_runs_do_not_bridge_seasonal_gap() -> None:
    jd = np.array([0.0, 1.0, 2.0, 300.0, 301.0, 302.0])
    mask = np.ones(len(jd), dtype=bool)

    assert _find_runs(mask, jd=jd) == [(0, 2), (3, 5)]
