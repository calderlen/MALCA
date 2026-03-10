from __future__ import annotations

import numpy as np

from malca.periodogram import ce_find_period, lsp_find_period, pdm_find_period


def _make_split_epoch_signal(true_period: float, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """Build a long-baseline signal with early/late epoch blocks."""
    rng = np.random.default_rng(seed)
    t_early = np.sort(rng.uniform(0.0, 1200.0, 360))
    t_late = np.sort(rng.uniform(2600.0, 3800.0, 540))
    times = np.concatenate([t_early, t_late])

    phase = np.mod((times - times.min()) / true_period, 1.0)
    signal = (
        0.12 * np.cos(2.0 * np.pi * phase + 0.2)
        + 0.04 * np.cos(4.0 * np.pi * phase + 1.1)
    )
    values = signal + rng.normal(0.0, 0.018, size=times.size)
    return times, values


def test_pdm_refinement_improves_period_accuracy() -> None:
    min_p, max_p, n_periods = 1.3, 1.6, 401
    coarse_step = (max_p - min_p) / float(n_periods - 1)
    true_period = min_p + (173.37 * coarse_step)
    times, values = _make_split_epoch_signal(true_period, seed=11)

    p_coarse, _, _ = pdm_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        n_periods=n_periods,
        refine=False,
    )
    p_refined, _, _ = pdm_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        n_periods=n_periods,
        refine=True,
        refine_n_grid=2001,
    )

    assert abs(p_refined - true_period) < abs(p_coarse - true_period)
    assert abs(p_refined - true_period) < coarse_step


def test_ce_refinement_improves_period_accuracy() -> None:
    min_p, max_p, n_periods = 1.3, 1.6, 401
    coarse_step = (max_p - min_p) / float(n_periods - 1)
    true_period = min_p + (211.63 * coarse_step)
    times, values = _make_split_epoch_signal(true_period, seed=22)

    p_coarse, _, _ = ce_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        n_periods=n_periods,
        refine=False,
    )
    p_refined, _, _ = ce_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        n_periods=n_periods,
        refine=True,
        refine_n_grid=2001,
    )

    assert abs(p_refined - true_period) < abs(p_coarse - true_period)
    assert abs(p_refined - true_period) < coarse_step


def test_lsp_refinement_improves_period_accuracy() -> None:
    min_p, max_p = 1.3, 1.6
    true_period = 1.445583
    times, values = _make_split_epoch_signal(true_period, seed=33)

    p_coarse, _, _ = lsp_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        samples_per_peak=4,
        refine=False,
    )
    p_refined, _, _ = lsp_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        samples_per_peak=4,
        refine=True,
        refine_n_grid=2001,
    )

    assert abs(p_refined - true_period) < abs(p_coarse - true_period)
