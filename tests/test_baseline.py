"""Tests for baseline computation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.baseline import (
    global_median_baseline,
    per_camera_gp_baseline,
    per_camera_median_baseline,
    phase_template_baseline,
)


def make_synthetic_lc(
    n_points: int = 200,
    n_cameras: int = 2,
    base_mag: float = 14.0,
    scatter: float = 0.02,
    error: float = 0.015,
    jd_start: float = 2458000.0,
    cadence: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic quiescent light curve with all required columns."""
    rng = np.random.default_rng(seed)
    
    n_per_cam = n_points // n_cameras
    rows = []
    
    for cam in range(n_cameras):
        jd = jd_start + np.arange(n_per_cam) * cadence + rng.uniform(0, 1, n_per_cam)
        mag = base_mag + rng.normal(0, scatter, n_per_cam)
        err = np.full(n_per_cam, error) + rng.uniform(0, 0.005, n_per_cam)
        
        for j, m, e in zip(jd, mag, err):
            rows.append({
                "JD": j, 
                "mag": m, 
                "error": e, 
                "camera#": f"b{cam}",
                "saturated": 0,  # Required by clean_lc
            })
    
    return pd.DataFrame(rows).sort_values("JD").reset_index(drop=True)


def inject_dip(
    df: pd.DataFrame,
    t0: float,
    amplitude: float = 0.5,
    sigma: float = 10.0,
) -> pd.DataFrame:
    """Inject a Gaussian dip into the light curve."""
    df = df.copy()
    gaussian = amplitude * np.exp(-0.5 * ((df["JD"] - t0) / sigma) ** 2)
    df["mag"] = df["mag"] + gaussian  # Positive = fainter = dip
    return df


def inject_jump(
    df: pd.DataFrame,
    t0: float,
    amplitude: float = 0.5,
    sigma: float = 10.0,
) -> pd.DataFrame:
    """Inject a Gaussian brightening into the light curve."""
    df = df.copy()
    gaussian = amplitude * np.exp(-0.5 * ((df["JD"] - t0) / sigma) ** 2)
    df["mag"] = df["mag"] - gaussian  # Negative = brighter = jump
    return df


def make_periodic_template_lc(
    n_points: int = 320,
    period_days: float = 7.0,
    base_mag: float = 14.0,
    scatter: float = 0.015,
    seed: int = 123,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic multi-camera, multi-band periodic source for phase-template tests."""
    rng = np.random.default_rng(seed)
    jd = 2458000.0 + np.sort(rng.uniform(0.0, 180.0, n_points))
    phase = np.mod((jd - jd.min()) / period_days, 1.0)
    waveform = 0.18 * np.sin(2.0 * np.pi * phase) + 0.06 * np.cos(4.0 * np.pi * phase)

    camera = np.where(np.arange(n_points) % 2 == 0, 1, 2)
    band = np.where(np.arange(n_points) % 3 == 0, 1, 0)
    camera_offset = np.where(camera == 2, 0.05, 0.0)
    band_offset = np.where(band == 1, 0.45, 0.0)
    noise = rng.normal(0.0, scatter, n_points)

    df = pd.DataFrame(
        {
            "JD": jd,
            "mag": base_mag + waveform + camera_offset + band_offset + noise,
            "error": np.full(n_points, 0.02, dtype=float),
            "camera#": camera,
            "v_g_band": band,
            "saturated": 0,
        }
    )
    return df, waveform


class TestSigmaFloorConsistency:
    """Test that GP and masked GP produce consistent sigma_eff values."""

    def test_gp_baseline_has_sigma_eff(self):
        """Verify per_camera_gp_baseline produces sigma_eff column."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert "sigma_eff" in result.columns
        assert result["sigma_eff"].notna().all()
        assert (result["sigma_eff"] > 0).all()

    def test_median_baselines_have_sigma_eff(self):
        df = make_synthetic_lc(seed=123)
        res_global = global_median_baseline(df)
        res_cam = per_camera_median_baseline(df)
        assert "sigma_eff" in res_global.columns
        assert "sigma_eff" in res_cam.columns
        assert (res_global["sigma_eff"] > 0).all()
        assert (res_cam["sigma_eff"] > 0).all()

    def test_sigma_eff_includes_sigma_floor(self):
        """sigma_eff should be larger than just mag_err due to sigma_floor."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        # sigma_eff should generally be >= error (due to floor and GP variance)
        median_sigma_eff = result["sigma_eff"].median()
        median_error = result["error"].median()
        
        assert median_sigma_eff >= median_error * 0.9  # Allow small tolerance


class TestRobustSigmaFloor:
    """Test the robust_sigma_floor estimation."""

    def test_sigma_floor_nonnegative(self):
        """sigma_floor should never produce negative values."""
        df = make_synthetic_lc()
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        # sigma_eff² = mag_err² + floor² + var, so sigma_eff >= mag_err
        # This indirectly tests floor >= 0
        assert (result["sigma_eff"] >= 0).all()

    def test_sigma_floor_robust_to_outliers(self):
        """sigma_floor should be stable when dips are present."""
        df_clean = make_synthetic_lc(seed=456)
        df_with_dip = inject_dip(df_clean, t0=df_clean["JD"].median(), amplitude=0.8)
        
        result_clean = per_camera_gp_baseline(df_clean, add_sigma_eff_col=True)
        result_dip = per_camera_gp_baseline(df_with_dip, add_sigma_eff_col=True)
        
        # Median sigma_eff shouldn't change dramatically due to one dip
        ratio = result_dip["sigma_eff"].median() / result_clean["sigma_eff"].median()
        assert 0.7 < ratio < 2.0, f"sigma_eff changed too much with dip: {ratio:.2f}"


class TestBaselineFallbacks:
    """Test fallback behavior when GP fit fails."""

    def test_fallback_with_few_points(self):
        """Should fall back to median with insufficient points."""
        df = make_synthetic_lc(n_points=8, n_cameras=1)
        # per_camera_gp_baseline has internal min points check
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert "baseline" in result.columns
        assert result["baseline"].notna().all()

    def test_fallback_still_has_sigma_eff(self):
        """Fallback case should still compute sigma_eff."""
        df = make_synthetic_lc(n_points=8, n_cameras=1)
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert "sigma_eff" in result.columns
        assert result["sigma_eff"].notna().all()


class TestEdgeCases:
    """Test edge cases in baseline computation."""

    def test_gp_baseline_accepts_mag_err_col_keyword(self):
        df = make_synthetic_lc().rename(columns={"error": "mag_err"})
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True, mag_err_col="mag_err")

        assert len(result) == len(df)
        assert result["baseline"].notna().all()
        assert result["sigma_eff"].notna().all()

    def test_single_camera(self):
        """Handle single-camera light curves."""
        df = make_synthetic_lc(n_cameras=1)
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert len(result) == len(df)
        assert result["baseline"].notna().all()

    def test_some_nan_errors(self):
        """Handle light curves with some NaN errors."""
        df = make_synthetic_lc()
        df.loc[::3, "error"] = np.nan  # Every 3rd error is NaN
        
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        assert result["baseline"].notna().all()
        assert result.loc[df["error"].notna(), "sigma_eff"].notna().all()
        assert result.loc[df["error"].isna(), "sigma_eff"].isna().all()

    def test_empty_dataframe(self):
        """Handle empty dataframe gracefully."""
        df = pd.DataFrame(columns=["JD", "mag", "error", "camera#", "saturated"])
        result = per_camera_gp_baseline(df, add_sigma_eff_col=True)
        
        assert len(result) == 0


class TestPhaseTemplateBaseline:
    def test_phase_template_removes_periodic_waveform_and_keeps_stochastic_dip(self):
        period_days = 7.0
        df, waveform = make_periodic_template_lc(period_days=period_days, seed=1001)
        t0 = float(df["JD"].median())
        df = inject_dip(df, t0=t0, amplitude=0.45, sigma=0.8)

        result = phase_template_baseline(df, period_days=period_days)
        event_mask = np.abs(result["JD"] - t0) <= 1.5

        assert set(result["baseline_source"].astype(str)) == {"phase_template"}
        assert np.nanstd(result.loc[~event_mask, "resid"]) < 0.08
        assert float(result.loc[event_mask, "resid"].max()) > 0.20
        assert np.nanstd(waveform) > np.nanstd(result.loc[~event_mask, "resid"])

    def test_phase_template_keeps_brightening_residual(self):
        period_days = 5.5
        df, _waveform = make_periodic_template_lc(period_days=period_days, seed=1002)
        t0 = float(df["JD"].median())
        df = inject_jump(df, t0=t0, amplitude=0.40, sigma=0.8)

        result = phase_template_baseline(df, period_days=period_days)
        event_mask = np.abs(result["JD"] - t0) <= 1.5

        assert set(result["baseline_source"].astype(str)) == {"phase_template"}
        assert np.nanstd(result.loc[~event_mask, "resid"]) < 0.08
        assert float(result.loc[event_mask, "resid"].min()) < -0.18
