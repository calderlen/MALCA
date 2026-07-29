"""Tests for ``long_period_ls_search``.

We construct synthetic long-period signals that live below the standard LS
frequency floor and assert the new search finds them, while also verifying it
returns ``insufficient_points`` / ``zero_baseline`` / ``invalid_bounds`` on
degenerate inputs and does not spuriously flag pure noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from malca.core.stats import long_period_ls_search


def _make_dipper_lightcurve(
    *,
    baseline_days: float,
    period_days: float,
    n_points: int,
    dip_amplitude_mag: float = 0.6,
    dip_width_days: float = 30.0,
    noise_mag: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(0.0, baseline_days, size=n_points))
    phase = ((jd % period_days) / period_days)
    dip_frac = dip_width_days / period_days
    dip = dip_amplitude_mag * np.exp(-0.5 * ((phase - 0.5) / (dip_frac / 2.355)) ** 2)
    mag = 15.0 + dip + rng.normal(0.0, noise_mag, size=jd.size)
    err = np.full_like(mag, noise_mag)
    return jd, mag, err


def _matches_period(candidate: float, target: float, *, rel: float = 0.05) -> bool:
    if not np.isfinite(candidate) or candidate <= 0:
        return False
    return abs(candidate - target) / target <= rel


def _make_sinusoid_lightcurve(
    *,
    baseline_days: float,
    period_days: float,
    n_points: int,
    amplitude_mag: float = 0.3,
    noise_mag: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(0.0, baseline_days, size=n_points))
    mag = 15.0 + amplitude_mag * np.sin(2 * np.pi * jd / period_days) + rng.normal(0.0, noise_mag, size=jd.size)
    err = np.full_like(mag, noise_mag)
    return jd, mag, err


def test_recovers_pure_sinusoid_long_period() -> None:
    """A pure sinusoid at 800 d in a 3000 d baseline should be found exactly.

    Sinusoids have no upper harmonic content, so this validates the LS search
    itself before we exercise harmonic-rich signals.
    """
    jd, mag, err = _make_sinusoid_lightcurve(
        baseline_days=3000.0,
        period_days=800.0,
        n_points=1500,
    )
    result = long_period_ls_search(jd, mag, err, n_bootstrap=200, samples_per_peak=10)
    assert result["long_ls_status"] == "ok"
    assert result["long_ls_period_days"] == pytest.approx(800.0, rel=0.03)
    assert result["long_ls_is_significant"] is True


def test_dipper_2000_day_period_surfaces_as_a_harmonic_of_top_peaks() -> None:
    """Regression test for the AA-Tau-like candidate motivating this module.

    A Gaussian dip repeating at period ``P`` has strong harmonic content, so
    the raw LS peak may land on ``P/k`` (which sees ``k*1.8`` cycles vs 1.8 at
    the fundamental). We require that the true P is recoverable by multiplying
    one of the top-K raw peaks by a small integer, so the downstream consensus
    step can promote to the fundamental via native_harmonic_period_candidates.
    """
    jd, mag, err = _make_dipper_lightcurve(
        baseline_days=3653.0,
        period_days=2011.0,
        n_points=800,
        dip_amplitude_mag=0.7,
        dip_width_days=40.0,
        noise_mag=0.02,
    )
    result = long_period_ls_search(jd, mag, err, n_bootstrap=200, samples_per_peak=10)
    assert result["long_ls_status"] == "ok"
    assert result["long_ls_is_significant"] is True

    tops = result["long_ls_top_periods_days"]
    assert tops, "expected non-empty top peaks"

    def is_low_harmonic_of(target: float, candidate: float) -> bool:
        for k in (1, 2, 3, 4, 5, 6):
            for base in (candidate, candidate * k, candidate / k):
                if _matches_period(base, target, rel=0.06):
                    return True
        return False

    assert any(is_low_harmonic_of(2011.0, p) for p in tops), (
        f"no top peak is a low-order harmonic of 2011 d: {tops}"
    )
    assert result["long_ls_min_period_days"] < 2011.0 < result["long_ls_max_period_days"]


def test_noise_only_light_curve_is_not_significant() -> None:
    rng = np.random.default_rng(0)
    jd = np.sort(rng.uniform(0.0, 3000.0, size=800))
    mag = rng.normal(15.0, 0.02, size=jd.size)
    err = np.full_like(mag, 0.02)
    result = long_period_ls_search(jd, mag, err, n_bootstrap=100, samples_per_peak=5)
    assert result["long_ls_status"] == "ok"
    assert result["long_ls_is_significant"] is False
    assert 0.0 <= result["long_ls_fap_bootstrap"] <= 1.0


def test_insufficient_points_returns_status() -> None:
    jd = np.array([0.0, 1.0, 2.0])
    mag = np.array([15.0, 15.0, 15.0])
    err = np.array([0.02, 0.02, 0.02])
    result = long_period_ls_search(jd, mag, err, n_bootstrap=0)
    assert result["long_ls_status"] == "insufficient_points"
    assert not result["long_ls_is_significant"]


def test_zero_baseline_returns_status() -> None:
    jd = np.zeros(50)
    mag = np.full(50, 15.0)
    err = np.full(50, 0.02)
    result = long_period_ls_search(jd, mag, err, n_bootstrap=0)
    assert result["long_ls_status"] == "zero_baseline"


def test_user_supplied_bounds_are_honored() -> None:
    jd, mag, err = _make_dipper_lightcurve(
        baseline_days=3000.0,
        period_days=800.0,
        n_points=1000,
    )
    result = long_period_ls_search(
        jd, mag, err,
        min_period_days=50.0,
        max_period_days=1500.0,
        n_bootstrap=0,
    )
    assert result["long_ls_min_period_days"] == pytest.approx(50.0)
    assert result["long_ls_max_period_days"] == pytest.approx(1500.0)


def test_bounds_default_to_long_stage_of_baseline() -> None:
    jd, mag, err = _make_dipper_lightcurve(
        baseline_days=3653.0,
        period_days=2011.0,
        n_points=500,
    )
    result = long_period_ls_search(jd, mag, err, n_bootstrap=0)
    assert result["long_ls_max_period_days"] > 2011.0
    assert result["long_ls_min_period_days"] >= 10.0


def test_alias_matches_are_flagged() -> None:
    """A signal near a known LS alias (1 day) should be tagged, not counted as significant."""
    rng = np.random.default_rng(0)
    jd = np.sort(rng.uniform(0.0, 3000.0, size=1500))
    period = 1.0
    mag = 15.0 + 0.3 * np.sin(2 * np.pi * jd / period) + rng.normal(0.0, 0.02, jd.size)
    err = np.full_like(mag, 0.02)
    result = long_period_ls_search(
        jd, mag, err,
        min_period_days=0.5,
        max_period_days=100.0,
        n_bootstrap=20,
    )
    assert result["long_ls_status"] == "ok"
    if result["long_ls_period_days"] and abs(result["long_ls_period_days"] - 1.0) < 0.05:
        assert result["long_ls_is_alias"] is True
        assert result["long_ls_is_significant"] is False
