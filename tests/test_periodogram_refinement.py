from __future__ import annotations

import numpy as np

from malca.config.config_stats import PDM_PLAVCHAN_GRID_FACTOR, PDM_PLAVCHAN_WORST_FRAC, PDM_PLAVCHAN_WORST_MIN
from malca.periodogram import _build_plavchan_period_grid, ce_find_period, lsp_find_period, pdm_find_period


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


def _manual_plavchan_theta(
    times: np.ndarray,
    values: np.ndarray,
    *,
    period: float,
    phase_width: float,
    worst_frac: float,
    worst_min: int,
) -> float:
    phase = np.mod(times - np.min(times), period) / period
    order = np.argsort(phase)
    phase_sorted = phase[order]
    values_sorted = values[order]
    half_width = 0.5 * float(phase_width)

    smooth = np.empty_like(values_sorted)
    for idx, phase_value in enumerate(phase_sorted):
        delta = np.abs(phase_sorted - phase_value)
        delta = np.minimum(delta, 1.0 - delta)
        mask = delta <= half_width
        smooth[idx] = float(np.mean(values_sorted[mask]))

    resid2 = np.square(values_sorted - smooth)
    worst_count = max(int(np.ceil(float(worst_frac) * values_sorted.size)), int(worst_min))
    worst_count = min(max(worst_count, 1), values_sorted.size)
    worst_mean = float(np.mean(np.sort(resid2)[-worst_count:]))
    total_var = float(np.var(values_sorted, ddof=1))
    return float(worst_mean / total_var)


def test_plavchan_period_grid_uses_adaptive_spacing() -> None:
    times = np.array([0.0, 250.0], dtype=float)
    min_period = 0.5
    max_period = 6.0
    grid = _build_plavchan_period_grid(times, min_period, max_period)

    assert np.isclose(grid[0], min_period)
    assert np.isclose(grid[-1], max_period)
    assert np.all(np.diff(grid) > 0)

    baseline = float(times.max() - times.min())
    for idx in range(grid.size - 2):
        expected = PDM_PLAVCHAN_GRID_FACTOR * grid[idx] * grid[idx] / baseline
        assert np.isclose(grid[idx + 1] - grid[idx], expected, rtol=1e-12, atol=1e-12)

    final_expected = PDM_PLAVCHAN_GRID_FACTOR * grid[-2] * grid[-2] / baseline
    assert (grid[-1] - grid[-2]) <= final_expected + 1e-12


def _make_alias_prone_signal(true_period: float, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Build a semi-regular nightly cadence that produces strong daily aliases."""
    rng = np.random.default_rng(seed)
    nights = np.arange(0.0, 140.0, 1.0)
    keep = rng.random(nights.size) > 0.18
    nights = nights[keep]
    offsets = rng.normal(0.08, 0.015, size=nights.size)
    times = np.sort(nights + offsets)

    phase = np.mod(times / true_period, 1.0)
    eclipse = ((phase < 0.045) | (phase > 0.955)).astype(float)
    shoulder = ((phase > 0.46) & (phase < 0.54)).astype(float)
    values = (
        14.0
        + 0.95 * eclipse
        + 0.14 * shoulder
        + rng.normal(0.0, 0.03, size=times.size)
    )
    return times, values


def test_plavchan_metric_matches_direct_boxcar_smoothing() -> None:
    times = np.linspace(0.0, 12.0, 60, endpoint=False)
    values = 14.0 + 0.3 * np.sin(2.0 * np.pi * times / 3.0) + 0.1 * np.cos(4.0 * np.pi * times / 3.0)
    period = 3.0
    phase_width = 0.30

    _, _, theta = pdm_find_period(
        times,
        values,
        min_period=period,
        max_period=period,
        n_periods=1,
        method="plavchan",
        phase_width=phase_width,
        min_neighbors=1,
        refine=False,
    )

    expected = _manual_plavchan_theta(
        times,
        values,
        period=period,
        phase_width=phase_width,
        worst_frac=PDM_PLAVCHAN_WORST_FRAC,
        worst_min=PDM_PLAVCHAN_WORST_MIN,
    )
    assert np.isclose(theta[0], expected, rtol=1e-10, atol=1e-12)


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


def test_plavchan_pdm_refinement_improves_period_accuracy() -> None:
    min_p, max_p, n_periods = 1.3, 1.6, 401
    true_period = 1.4415
    times, values = _make_split_epoch_signal(true_period, seed=44)

    p_coarse, period_arr, _ = pdm_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        n_periods=n_periods,
        method="plavchan",
        refine=False,
    )
    p_refined, _, _ = pdm_find_period(
        times,
        values,
        min_period=min_p,
        max_period=max_p,
        n_periods=n_periods,
        method="plavchan",
        refine=True,
        refine_n_grid=2001,
    )

    nearest_grid_distance = float(np.min(np.abs(period_arr - true_period)))
    assert abs(p_refined - true_period) < abs(p_coarse - true_period)
    assert abs(p_refined - true_period) <= nearest_grid_distance + 1e-12


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


def test_plavchan_pdm_beats_classic_on_alias_prone_nightly_cadence() -> None:
    true_period = 2.413
    times, values = _make_alias_prone_signal(true_period, seed=55)

    p_classic, _, _ = pdm_find_period(
        times,
        values,
        min_period=0.8,
        max_period=3.5,
        n_periods=1800,
        method="classic",
        refine=True,
        refine_n_grid=2001,
    )
    p_plavchan, _, _ = pdm_find_period(
        times,
        values,
        min_period=0.8,
        max_period=3.5,
        n_periods=1800,
        method="plavchan",
        refine=True,
        refine_n_grid=2001,
    )

    assert abs(p_plavchan - true_period) < abs(p_classic - true_period)
    assert abs(p_plavchan - true_period) < 0.08
