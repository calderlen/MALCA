"""Tests for baseline computation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.core.baseline import (
    global_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
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


def _make_late_onset_lc(
    *,
    n_anchor_cams: int = 3,
    n_late_cams: int = 1,
    base_mag: float = 14.0,
    scatter: float = 0.02,
    error: float = 0.015,
    jd_start: float = 7000.0,
    jd_end: float = 11000.0,
    late_onset_jd: float = 9500.0,
    dip_center_jd: float = 9600.0,
    dip_amplitude: float = 0.5,
    dip_sigma: float = 15.0,
    cadence: float = 5.0,
    band: int = 0,
    seed: int = 99,
) -> pd.DataFrame:
    """Build a lightcurve where late-onset cameras start observing during a dip.

    Anchor cameras span the full time range. Late-onset cameras only begin
    observing near the dip, so their per-camera stiff GP would track the dip
    as quiescent if not corrected by the consensus mechanism.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for cam_i in range(n_anchor_cams):
        jd = np.arange(jd_start, jd_end, cadence) + rng.uniform(-1, 1, int((jd_end - jd_start) / cadence))
        jd = np.sort(jd)
        mag = base_mag + rng.normal(0, scatter, len(jd))
        dip = dip_amplitude * np.exp(-0.5 * ((jd - dip_center_jd) / dip_sigma) ** 2)
        mag += dip
        err = np.full(len(jd), error) + rng.uniform(0, 0.003, len(jd))
        for j, m, e in zip(jd, mag, err):
            rows.append({
                "JD": j, "mag": m, "error": e,
                "camera#": f"anchor_{cam_i}", "v_g_band": band, "saturated": 0,
            })

    for cam_i in range(n_late_cams):
        jd = np.arange(late_onset_jd, jd_end, cadence) + rng.uniform(-1, 1, int((jd_end - late_onset_jd) / cadence))
        jd = np.sort(jd)
        mag = base_mag + rng.normal(0, scatter, len(jd))
        dip = dip_amplitude * np.exp(-0.5 * ((jd - dip_center_jd) / dip_sigma) ** 2)
        mag += dip
        err = np.full(len(jd), error) + rng.uniform(0, 0.003, len(jd))
        for j, m, e in zip(jd, mag, err):
            rows.append({
                "JD": j, "mag": m, "error": e,
                "camera#": f"late_{cam_i}", "v_g_band": band, "saturated": 0,
            })

    return pd.DataFrame(rows).sort_values("JD").reset_index(drop=True)


def _make_staggered_rollout_lc(
    *,
    base_mag: float = 14.0,
    scatter: float = 0.02,
    error: float = 0.015,
    jd_start: float = 7000.0,
    jd_end: float = 11000.0,
    dip_center_jd: float = 10000.0,
    dip_amplitude: float = 0.5,
    dip_sigma: float = 20.0,
    cadence: float = 5.0,
    band: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Staggered g-band rollout: mid-band cameras have long baselines; one starts near dip."""
    rng = np.random.default_rng(seed)
    rows = []
    camera_starts = {
        "bj": jd_start,
        "bn": jd_start + 500.0,
        "bF": jd_start + 1000.0,
        "cB": jd_start + 2800.0,
    }

    def _add_camera(name: str, start_jd: float):
        jd = np.arange(start_jd, jd_end, cadence) + rng.uniform(-1, 1, int((jd_end - start_jd) / cadence))
        jd = np.sort(jd)
        mag = base_mag + rng.normal(0, scatter, len(jd))
        dip = dip_amplitude * np.exp(-0.5 * ((jd - dip_center_jd) / dip_sigma) ** 2)
        mag += dip
        err = np.full(len(jd), error) + rng.uniform(0, 0.003, len(jd))
        for j, m, e in zip(jd, mag, err):
            rows.append({
                "JD": j, "mag": m, "error": e,
                "camera#": name, "v_g_band": band, "saturated": 0,
            })

    for name, start in camera_starts.items():
        _add_camera(name, start)

    return pd.DataFrame(rows).sort_values("JD").reset_index(drop=True)


class TestLateOnsetConsensus:
    """Test that late-onset cameras use the band consensus for masking."""

    def test_late_onset_camera_classified(self):
        """A camera starting 500+ days after band start should be late-onset."""
        df = _make_late_onset_lc(late_onset_jd=9500.0)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=300.0, min_anchor_overlap_days=30.0,
        )
        assert "base_rough" in result.columns
        assert result["baseline"].notna().all()

    def test_late_onset_base_rough_near_quiescent(self):
        """Late-onset camera's base_rough should be near the quiescent level,
        not tracking the dip."""
        df = _make_late_onset_lc(dip_amplitude=0.5)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=300.0, min_anchor_overlap_days=30.0,
        )
        late_mask = result["camera#"].str.startswith("late_")
        late_base_rough = result.loc[late_mask, "base_rough"].to_numpy(float)
        assert np.all(np.abs(late_base_rough - 14.0) < 0.15), (
            f"Late-onset base_rough deviates too far from quiescent: "
            f"mean={np.mean(late_base_rough):.3f}"
        )

    def test_late_onset_dip_residuals_positive(self):
        """With consensus baseline, the dip should produce positive residuals
        (fainter than baseline) in the late-onset camera, not inverted."""
        df = _make_late_onset_lc(dip_amplitude=0.5, dip_sigma=15.0)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=300.0, min_anchor_overlap_days=30.0,
        )
        late_mask = result["camera#"].str.startswith("late_")
        dip_window = np.abs(result["JD"] - 9600.0) < 30.0
        dip_resid = result.loc[late_mask & dip_window, "resid"].to_numpy(float)
        assert np.nanmax(dip_resid) > 0.1, (
            f"Expected positive residuals in dip window, got max={np.nanmax(dip_resid):.3f}"
        )

    def test_late_onset_baseline_does_not_track_dip(self):
        """Late-onset final baseline should stay near quiescent, not follow the dip."""
        df = _make_late_onset_lc(dip_amplitude=0.5, dip_sigma=15.0)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=300.0, min_anchor_overlap_days=30.0,
        )
        late_mask = result["camera#"].str.startswith("late_")
        dip_window = np.abs(result["JD"] - 9600.0) < 30.0
        late_baseline = result.loc[late_mask & dip_window, "baseline"].to_numpy(float)
        assert np.all(np.abs(late_baseline - 14.0) < 0.15), (
            f"Late-onset baseline tracks dip: mean={np.mean(late_baseline):.3f}"
        )

    def test_anchor_cameras_unaffected(self):
        """Anchor and mid-band cameras with long baselines match disabled consensus."""
        df = _make_late_onset_lc()
        result_with = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=100.0, min_anchor_overlap_days=30.0,
        )
        result_without = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=0,
        )
        anchor_mask = result_with["camera#"].str.startswith("anchor_")
        np.testing.assert_allclose(
            result_with.loc[anchor_mask, "baseline"].to_numpy(float),
            result_without.loc[anchor_mask, "baseline"].to_numpy(float),
            atol=1e-10,
        )

    def test_staggered_rollout_only_excursion_camera_differs(self):
        """Mid-band cameras with quiet history are unchanged; only cB-like camera is fixed."""
        df = _make_staggered_rollout_lc()
        result_with = per_camera_gp_baseline_masked(
            df,
            late_onset_buffer_days=100.0,
            min_quiet_baseline_days=250.0,
            min_anchor_overlap_days=30.0,
        )
        result_without = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=0,
        )

        for cam in ("bj", "bn", "bF"):
            mask = result_with["camera#"] == cam
            np.testing.assert_allclose(
                result_with.loc[mask, "baseline"].to_numpy(float),
                result_without.loc[mask, "baseline"].to_numpy(float),
                atol=1e-10,
                err_msg=f"{cam} baseline should be unchanged by consensus logic",
            )

        cB_mask = result_with["camera#"] == "cB"
        dip_window = np.abs(result_with["JD"] - 10000.0) < 40.0
        dip_resid = result_with.loc[cB_mask & dip_window, "resid"].to_numpy(float)
        assert np.nanmax(dip_resid) > 0.08, (
            f"cB-like camera should retain positive dip residuals, max={np.nanmax(dip_resid):.3f}"
        )

    def test_staggered_rollout_cB_base_rough_near_quiescent(self):
        df = _make_staggered_rollout_lc()
        result = per_camera_gp_baseline_masked(
            df,
            late_onset_buffer_days=100.0,
            min_quiet_baseline_days=250.0,
            min_anchor_overlap_days=30.0,
        )
        cB_base = result.loc[result["camera#"] == "cB", "base_rough"].to_numpy(float)
        assert np.all(np.abs(cB_base - 14.0) < 0.2)

    def test_camera_starts_in_anchor_excursion_window(self):
        """Camera whose first obs fall inside anchor excursion window needs consensus."""
        rng = np.random.default_rng(55)
        rows = []
        base_mag = 14.0
        dip_center = 10000.0
        for cam_i in range(2):
            jd = np.arange(7000.0, 11000.0, 5.0) + rng.uniform(-1, 1, 800)
            jd = np.sort(jd)
            mag = base_mag + rng.normal(0, 0.02, len(jd))
            mag += 0.6 * np.exp(-0.5 * ((jd - dip_center) / 15.0) ** 2)
            err = np.full(len(jd), 0.015)
            for j, m, e in zip(jd, mag, err):
                rows.append({
                    "JD": j, "mag": m, "error": e,
                    "camera#": f"anchor_{cam_i}", "v_g_band": 0, "saturated": 0,
                })
        jd_late = np.arange(9980.0, 11000.0, 5.0)
        mag_late = base_mag + 0.55 + rng.normal(0, 0.02, len(jd_late))
        mag_late += 0.6 * np.exp(-0.5 * ((jd_late - dip_center) / 15.0) ** 2)
        for j, m in zip(jd_late, mag_late):
            rows.append({
                "JD": j, "mag": m, "error": 0.015,
                "camera#": "late_in_dip", "v_g_band": 0, "saturated": 0,
            })
        df = pd.DataFrame(rows).sort_values("JD").reset_index(drop=True)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=100.0, min_quiet_baseline_days=250.0,
        )
        late_base = result.loc[result["camera#"] == "late_in_dip", "base_rough"].to_numpy(float)
        assert np.all(np.abs(late_base - base_mag) < 0.2)

    def test_multiple_anchors_contribute_to_consensus(self):
        """Consensus uses all qualifying anchors, not only the earliest."""
        df = _make_staggered_rollout_lc(seed=7)
        result = per_camera_gp_baseline_masked(
            df,
            late_onset_buffer_days=100.0,
            min_quiet_baseline_days=250.0,
            min_anchor_overlap_days=30.0,
        )
        cB_base = result.loc[result["camera#"] == "cB", "base_rough"].to_numpy(float)
        bj_only = result.loc[result["camera#"] == "bj", "base_rough"].to_numpy(float)
        overlap_len = min(len(cB_base), len(bj_only))
        assert not np.allclose(cB_base[:overlap_len], bj_only[:overlap_len], atol=1e-6)

    def test_no_band_col_graceful(self):
        """Without a band column, late-onset detection is skipped gracefully."""
        df = _make_late_onset_lc().drop(columns=["v_g_band"])
        result = per_camera_gp_baseline_masked(df, late_onset_buffer_days=300.0)
        assert result["baseline"].notna().all()

    def test_single_camera_band_not_flagged(self):
        """A band with only one camera should not be flagged as late-onset."""
        df = _make_late_onset_lc(n_anchor_cams=0, n_late_cams=1, late_onset_jd=7000.0)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=300.0, min_anchor_overlap_days=30.0,
        )
        assert result["baseline"].notna().all()

    def test_cross_band_fallback(self):
        """When all cameras in a new band start late, fall back to previous band."""
        rng = np.random.default_rng(77)
        rows = []
        base_mag = 14.0
        dip_center = 9600.0

        for cam_i in range(3):
            jd = np.arange(7000.0, 11000.0, 5.0) + rng.uniform(-1, 1, 800)
            jd = np.sort(jd)
            mag = base_mag + rng.normal(0, 0.02, len(jd))
            dip = 0.5 * np.exp(-0.5 * ((jd - dip_center) / 15.0) ** 2)
            mag += dip
            err = np.full(len(jd), 0.015)
            for j, m, e in zip(jd, mag, err):
                rows.append({
                    "JD": j, "mag": m, "error": e,
                    "camera#": f"old_band_{cam_i}", "v_g_band": 0, "saturated": 0,
                })

        for cam_i in range(2):
            jd = np.arange(9500.0, 11000.0, 5.0) + rng.uniform(-1, 1, 300)
            jd = np.sort(jd)
            mag = base_mag + rng.normal(0, 0.02, len(jd))
            dip = 0.5 * np.exp(-0.5 * ((jd - dip_center) / 15.0) ** 2)
            mag += dip
            err = np.full(len(jd), 0.015)
            for j, m, e in zip(jd, mag, err):
                rows.append({
                    "JD": j, "mag": m, "error": e,
                    "camera#": f"new_band_{cam_i}", "v_g_band": 1, "saturated": 0,
                })

        df = pd.DataFrame(rows).sort_values("JD").reset_index(drop=True)
        result = per_camera_gp_baseline_masked(
            df, late_onset_buffer_days=300.0, min_anchor_overlap_days=30.0,
        )
        new_band_mask = result["camera#"].str.startswith("new_band_")
        new_band_base_rough = result.loc[new_band_mask, "base_rough"].to_numpy(float)
        assert np.all(np.abs(new_band_base_rough - base_mag) < 0.15), (
            f"Cross-band fallback base_rough deviates from quiescent: "
            f"mean={np.mean(new_band_base_rough):.3f}"
        )

    def test_cross_band_transfer_requires_and_records_colour_calibration(self):
        rng = np.random.default_rng(711)
        rows = []
        dip_center = 9600.0
        for camera in ("g1", "g2"):
            jd = np.arange(7000.0, 11000.0, 5.0)
            mag = 14.0 + rng.normal(0.0, 0.015, len(jd))
            mag += 0.45 * np.exp(-0.5 * ((jd - dip_center) / 15.0) ** 2)
            rows.extend(
                {
                    "JD": t,
                    "mag": m,
                    "error": 0.015,
                    "camera#": camera,
                    "v_g_band": 0,
                    "saturated": 0,
                }
                for t, m in zip(jd, mag)
            )
        for camera in ("v1", "v2"):
            jd = np.arange(9500.0, 11000.0, 5.0)
            mag = 15.0 + rng.normal(0.0, 0.015, len(jd))
            mag += 0.45 * np.exp(-0.5 * ((jd - dip_center) / 15.0) ** 2)
            rows.extend(
                {
                    "JD": t,
                    "mag": m,
                    "error": 0.015,
                    "camera#": camera,
                    "v_g_band": 1,
                    "saturated": 0,
                }
                for t, m in zip(jd, mag)
            )

        result = per_camera_gp_baseline_masked(
            pd.DataFrame(rows).sort_values("JD").reset_index(drop=True),
            late_onset_buffer_days=300.0,
            min_anchor_overlap_days=30.0,
            allow_cross_band_consensus=True,
            cross_band_min_overlap_points=50,
        )
        target = result["camera#"].astype(str).str.startswith("v")

        assert result.loc[target, "cross_band_calibrated"].all()
        assert np.nanmedian(result.loc[target, "cross_band_offset_mag"]) == pytest.approx(1.0, abs=0.05)
        assert set(result.loc[target, "baseline_source"]) == {"cross_band_consensus_calibrated"}
        assert np.nanmedian(result.loc[target, "baseline"]) == pytest.approx(15.0, abs=0.05)
