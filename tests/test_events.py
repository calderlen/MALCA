"""Tests for event detection functions."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from malca.stv.events import (
    build_runs,
    classify_run_morphology,
    compute_recurrence_stats,
    filter_runs,
    main as events_main,
    merge_event_result_metadata,
    score_events_bayesian,
    score_lightcurve,
    signal_amplitude_pass_mask,
)
from malca.core.baseline import per_camera_gp_baseline
from malca.core.utils import fred
from malca.products.feature_layers import to_layer_first_frame, with_feature_columns


def _global_constant_baseline(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = df.copy()
    baseline = np.full(len(out), float(np.nanmedian(out["mag"].to_numpy(dtype=float))))
    resid = out["mag"].to_numpy(dtype=float) - baseline
    sigma_eff = np.full(len(out), 0.02)
    out["baseline"] = baseline
    out["resid"] = resid
    out["sigma_eff"] = sigma_eff
    out["sigma_resid"] = resid / sigma_eff
    out["baseline_source"] = "test_global_constant"
    return out


def test_event_metadata_cannot_overwrite_measured_counts() -> None:
    result = {
        "lc_path": "candidate.dat2",
        "raw_n_points": 10,
        "raw_n_cameras": 2,
        "n_points": 6,
    }
    metadata = {
        "raw_n_points": 10,
        "raw_n_cameras": 2,
        "clean_n_points": 8,
        "tag_stats_status": "ok",
        "tag_stats_error": "",
    }

    merged = merge_event_result_metadata(result, metadata, lc_path="candidate.dat2")

    assert merged["raw_n_points"] == 10
    assert merged["clean_n_points"] == 8
    assert merged["tag_stats_status"] == "ok"
    with pytest.raises(ValueError, match="raw_n_points"):
        merge_event_result_metadata(
            result,
            {**metadata, "raw_n_points": 11},
            lc_path="candidate.dat2",
        )
    with pytest.raises(ValueError, match="Point-count invariant"):
        merge_event_result_metadata(
            result,
            {**metadata, "clean_n_points": 5},
            lc_path="candidate.dat2",
        )


def test_event_metadata_comparison_is_column_aware_and_camera_counts_balance() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        merge_event_result_metadata(
            {"candidate_id": "001"},
            {"candidate_id": 1},
            lc_path="candidate.dat2",
        )
    with pytest.raises(ValueError, match="pre_periodic_flag"):
        merge_event_result_metadata(
            {"pre_periodic_flag": True},
            {"pre_periodic_flag": "false"},
            lc_path="candidate.dat2",
        )
    merged = merge_event_result_metadata(
        {"pre_periodic_flag": False},
        {"pre_periodic_flag": "no"},
        lc_path="candidate.dat2",
    )
    assert merged["pre_periodic_flag"] is False
    with pytest.raises(ValueError, match="raw_n_points"):
        merge_event_result_metadata(
            {"raw_n_points": 10},
            {"raw_n_points": 10.0000000001},
            lc_path="candidate.dat2",
        )
    with pytest.raises(ValueError, match="Camera-count invariant"):
        merge_event_result_metadata(
            {"raw_n_cameras": 1, "n_cameras": 2},
            {},
            lc_path="candidate.dat2",
        )


def _write_events_config(tmp_path: Path, **options: object) -> Path:
    config_path = tmp_path / "events_config.json"
    config_path.write_text(json.dumps({"events": options}), encoding="ascii")
    return config_path


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
    df["mag"] = df["mag"] + gaussian
    return df


def _write_dat3(path: Path, times: np.ndarray, mags: np.ndarray, *, error: float = 0.03) -> None:
    lines: list[str] = []
    for idx, (time_value, mag_value) in enumerate(zip(times, mags, strict=True)):
        camera = 1 + (idx % 2)
        band = idx % 2
        lines.append(
            f"{float(time_value):.6f} {float(mag_value):.6f} {error:.6f} 1 {camera:d} {band:d} 0 cam{camera}/field1"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _periodic_dip_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 120.0, 240)
    phase = np.mod(times, 6.0) / 6.0
    periodic = 14.0 + 0.20 * np.sin(2.0 * np.pi * phase) + 0.05 * np.cos(4.0 * np.pi * phase)
    stochastic_dip = 0.45 * np.exp(-0.5 * ((times - 60.0) / 0.8) ** 2)
    return times, periodic + stochastic_dip


class TestBuildRuns:
    """Test run building from triggered indices.
    
    Note: build_runs returns a list of numpy arrays, where each array
    contains the indices belonging to that run.
    """

    def test_contiguous_triggers_form_single_run(self):
        """Adjacent triggered indices should form one run."""
        jd = np.arange(100.0)
        trig_idx = np.array([10, 11, 12, 13, 14])
        
        runs = build_runs(trig_idx, jd, max_gap_points=0)
        
        assert len(runs) == 1
        assert list(runs[0]) == [10, 11, 12, 13, 14]

    def test_gap_breaks_runs(self):
        """Large gaps should break runs into separate groups."""
        jd = np.arange(100.0)
        trig_idx = np.array([10, 11, 12, 50, 51, 52])  # Gap between 12 and 50
        
        runs = build_runs(trig_idx, jd, max_gap_points=0, max_gap_days=5.0)
        
        assert len(runs) == 2
        assert list(runs[0]) == [10, 11, 12]
        assert list(runs[1]) == [50, 51, 52]

    def test_allow_gap_points(self):
        """Small index gaps should be allowed within a run."""
        jd = np.arange(100.0)
        trig_idx = np.array([10, 11, 13, 14])  # Missing index 12
        
        # max_gap_points=1 means max_index_step = 2
        # But we also need max_gap_days large enough (dt between 11 and 13 is 2 days)
        runs = build_runs(trig_idx, jd, max_gap_points=1, max_gap_days=5.0)
        
        # With max_gap_points=1 and sufficient max_gap_days, gap of 2 indices should merge
        assert len(runs) == 1

    def test_max_gap_days_auto_calculation(self):
        """max_gap_days should default to 99.73th percentile of gaps."""
        # Create JD with mostly 3-day cadence but one 30-day gap
        jd = np.concatenate([
            np.arange(0, 100, 3),      # Regular cadence
            np.arange(130, 200, 3),    # After 30-day gap
        ])
        trig_idx = np.array([0, 1, 2])
        
        # With auto max_gap_days, should still form runs
        runs = build_runs(trig_idx, jd, max_gap_days=None)
        
        assert len(runs) >= 1

    def test_explicit_max_gap_days_overrides(self):
        """Explicit max_gap_days should override automatic calculation."""
        jd = np.arange(100.0)
        # Triggers with 10-index gap (10 days since cadence=1)
        trig_idx = np.array([10, 11, 12, 22, 23, 24])
        
        # With max_gap_days=5, the 10-day gap should break runs
        runs_short = build_runs(trig_idx, jd, max_gap_days=5.0, max_gap_points=0)
        
        # With max_gap_days=15, should be 1 run (but still need max_gap_points
        # to allow skipping indices 13-21)
        runs_long = build_runs(trig_idx, jd, max_gap_days=15.0, max_gap_points=10)
        
        assert len(runs_short) == 2
        assert len(runs_long) == 1

    def test_default_gap_never_bridges_an_observing_season(self):
        jd = np.array([0.0, 1.0, 2.0, 300.0, 301.0, 302.0])
        trig_idx = np.arange(len(jd))

        runs = build_runs(trig_idx, jd, max_gap_points=0)

        assert [run.tolist() for run in runs] == [[0, 1, 2], [3, 4, 5]]


def test_signal_amplitude_uses_delta_mag_without_subtracting_baseline() -> None:
    rows = pd.DataFrame(
        {
            "baseline_mag": [14.0, 19.0, 14.0],
            "dip_significant": [True, True, True],
            "jump_significant": [False, False, False],
            "dip_best_delta_mag": [0.05, 0.15, 0.10],
            "jump_best_delta_mag": [np.nan, np.nan, np.nan],
        }
    )

    passed = signal_amplitude_pass_mask(rows, 0.10)

    assert passed.tolist() == [False, True, True]


def test_recurrence_separates_photometric_depth_from_trigger_strength() -> None:
    summaries = [
        {
            "start_jd": 0.0,
            "end_jd": 2.0,
            "duration_days": 2.0,
            "run_max": 10.0,
            "peak_delta_mag": 0.20,
        },
        {
            "start_jd": 10.0,
            "end_jd": 12.0,
            "duration_days": 2.0,
            "run_max": 20.0,
            "peak_delta_mag": 0.20,
        },
    ]

    result = compute_recurrence_stats(summaries)

    assert result["amplitude_consistency"] == pytest.approx(0.0)
    assert result["trigger_strength_cv"] > 0.0


class TestFilterRuns:
    """Test run filtering logic.
    
    Note: filter_runs returns (kept_runs, summaries) where:
    - kept_runs: list of numpy arrays (indices for runs that passed filters)
    - summaries: list of dicts with info for ALL runs (including failed ones)
    """

    def test_min_points_filter(self):
        """Runs with too few points should be rejected."""
        jd = np.arange(100.0)
        score_vec = np.ones(100) * 10.0
        # Build runs from indices
        runs = [np.array([10, 11]), np.array([20, 21, 22, 23, 24, 25])]  # 2 vs 6 points
        
        kept, summaries = filter_runs(runs, jd, score_vec, min_points=3)
        
        # Only the longer run should pass
        assert len(kept) == 1
        assert len(kept[0]) == 6

    def test_per_point_threshold(self):
        """Runs without high enough max score should be rejected."""
        jd = np.arange(100.0)
        score_vec = np.ones(100) * 5.0
        score_vec[20:26] = 15.0  # High scores only in second run
        runs = [np.array([10, 11, 12, 13, 14, 15]), np.array([20, 21, 22, 23, 24, 25])]
        
        kept, summaries = filter_runs(runs, jd, score_vec, per_point_threshold=10.0)
        
        assert len(kept) == 1
        assert kept[0][0] == 20  # Should be the second run

    def test_min_duration_days(self):
        """Runs shorter than min_duration should be rejected."""
        jd = np.arange(100.0)  # 1 day cadence
        score_vec = np.ones(100) * 10.0
        runs = [np.array([10, 11, 12]), np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])]
        
        kept, summaries = filter_runs(runs, jd, score_vec, min_duration_days=5.0)
        
        assert len(kept) == 1
        # Check that the kept run has duration >= 5 days
        assert summaries[1]["duration_days"] >= 5.0

    def test_runs_pass_without_sum_threshold(self):
        """Runs should pass without sum_threshold parameter (removed)."""
        jd = np.arange(100.0)
        score_vec = np.ones(100) * 2.0  # Low individual scores
        runs = [np.array([10, 11, 12, 13, 14, 15])]
        
        kept, summaries = filter_runs(runs, jd, score_vec, min_points=2)
        
        assert len(kept) == 1


class TestRunBayesianSignificance:
    """Test the high-level score_lightcurve function."""

    def test_basic_usage(self):
        """Basic usage should work without errors."""
        df = make_synthetic_lc(seed=400)
        
        result = score_lightcurve(
            df,
            logbf_threshold_dip=5.0,
            logbf_threshold_jump=5.0,
            trigger_mode="logbf",
        )
        
        assert "dip" in result
        assert "jump" in result
        assert "significant" in result["dip"]
        assert "significant" in result["jump"]

    def test_uses_sigma_eff(self):
        """Baselines should provide sigma_eff by default."""
        df = make_synthetic_lc(seed=500)

        result = score_lightcurve(df)
        assert "dip" in result

    def test_quiescent_lc_no_detection(self):
        """Flat light curve should not trigger detections."""
        df = make_synthetic_lc(scatter=0.01, seed=100)
        
        result = score_lightcurve(
            df,
            logbf_threshold_dip=5.0,
            logbf_threshold_jump=5.0,
            trigger_mode="logbf",
        )
        
        # Should not be significant
        assert not result["dip"]["significant"]

    def test_dip_detection(self):
        """Injected dip should be detected."""
        df = make_synthetic_lc(n_points=300, scatter=0.015, seed=200)
        t0 = df["JD"].median()
        df = inject_dip(df, t0=t0, amplitude=0.4, sigma=8.0)
        
        result = score_lightcurve(
            df,
            logbf_threshold_dip=3.0,  # Lower threshold for test
            trigger_mode="logbf",
        )
        
        # Should detect the dip (check for triggers or runs)
        has_triggers = len(result["dip"].get("event_indices", [])) > 0
        is_significant = result["dip"]["significant"]
        assert has_triggers or is_significant

    def test_residual_bad_camera_filter_refits_before_scoring(self):
        rows = []
        for cam, offset in ((1, 0.0), (2, 0.0), (3, 1.0)):
            jd = 2458000.0 + np.arange(140, dtype=float)
            mag = np.full_like(jd, 14.0 + offset, dtype=float)
            for t, m in zip(jd, mag):
                rows.append(
                    {
                        "JD": t,
                        "mag": m,
                        "error": 0.02,
                        "camera#": cam,
                        "saturated": 0,
                    }
                )
        df = pd.DataFrame(rows)

        result = score_lightcurve(
            df,
            baseline_func=_global_constant_baseline,
            baseline_kwargs={},
            filter_residual_bad_cameras_enabled=True,
            trigger_mode="logbf",
        )

        assert result["bad_cameras_filtered"] == {3}
        assert 3 not in set(result["df"]["camera#"])
        assert len(result["df_base"]) == len(result["df"])

    def test_kept_run_summaries_stay_aligned_after_rejections(self, monkeypatch):
        """Rejected early runs must not shift diagnostics onto later kept runs."""
        jd = np.arange(60.0)
        df = pd.DataFrame(
            {
                "JD": jd,
                "mag": np.full_like(jd, 14.0, dtype=float),
                "error": np.full_like(jd, 0.02, dtype=float),
                "camera#": ["b0"] * len(jd),
            }
        )
        df_base = df.copy()
        df_base["baseline"] = 14.0
        df_base["sigma_eff"] = 0.02
        df_base["baseline_source"] = "test"

        point_significance = np.zeros(len(jd), dtype=float)
        point_significance[[5, 6]] = 10.0
        point_significance[20:31] = 10.0

        def fake_resolve_trigger_indices(**kwargs):
            return {
                "point_significance": point_significance,
                "event_indices": np.array([5, 6, *range(20, 31)], dtype=int),
                "trigger_threshold": 5.0,
                "trigger_max": 10.0,
            }

        monkeypatch.setattr("malca.stv.events.resolve_trigger_indices", fake_resolve_trigger_indices)
        monkeypatch.setattr(
            "malca.stv.events.classify_run_morphology",
            lambda *args, **kwargs: {"morphology": "test", "bic": 0.0, "delta_bic_null": 0.0, "params": {}},
        )
        monkeypatch.setattr("malca.stv.events.compute_symmetry_score", lambda *args, **kwargs: 0.0)

        result = score_events_bayesian(
            df,
            kind="dip",
            baseline_func=None,
            df_base=df_base,
            trigger_mode="logbf",
            logbf_threshold=5.0,
            p_points=2,
            mag_grid=np.array([0.2]),
            run_min_points=3,
            run_min_duration_days=5.0,
            compute_event_prob=False,
        )

        assert len(result["run_summaries"]) == 1
        assert result["run_summaries"][0]["start_idx"] == 20
        assert result["run_summaries"][0]["duration_days"] == 10.0

    def test_morphology_receives_sigma_eff_from_baseline(self, monkeypatch):
        """Morphology fitting should see baseline-derived sigma_eff, not raw errors."""
        jd = np.arange(40.0)
        raw_error = np.full_like(jd, 0.2, dtype=float)
        sigma_eff = np.full_like(jd, 0.05, dtype=float)
        df = pd.DataFrame(
            {
                "JD": jd,
                "mag": np.full_like(jd, 14.0, dtype=float),
                "error": raw_error,
                "camera#": ["b0"] * len(jd),
            }
        )
        df_base = df.copy()
        df_base["baseline"] = 14.0
        df_base["sigma_eff"] = sigma_eff
        df_base["baseline_source"] = "test"

        point_significance = np.zeros(len(jd), dtype=float)
        point_significance[10:16] = 10.0
        seen_sigma_eff: list[np.ndarray] = []

        def fake_resolve_trigger_indices(**kwargs):
            return {
                "point_significance": point_significance,
                "event_indices": np.arange(10, 16, dtype=int),
                "trigger_threshold": 5.0,
                "trigger_max": 10.0,
            }

        def fake_classify_run_morphology(jd_arr, mag_arr, sigma_eff_arr, run_idx, **kwargs):
            seen_sigma_eff.append(np.asarray(sigma_eff_arr, dtype=float).copy())
            return {"morphology": "test", "bic": 0.0, "delta_bic_null": 0.0, "params": {}}

        monkeypatch.setattr("malca.stv.events.resolve_trigger_indices", fake_resolve_trigger_indices)
        monkeypatch.setattr("malca.stv.events.classify_run_morphology", fake_classify_run_morphology)
        monkeypatch.setattr("malca.stv.events.compute_symmetry_score", lambda *args, **kwargs: 0.0)

        result = score_events_bayesian(
            df,
            kind="dip",
            baseline_func=None,
            df_base=df_base,
            trigger_mode="logbf",
            logbf_threshold=5.0,
            p_points=2,
            mag_grid=np.array([0.2]),
            run_min_points=3,
            run_min_duration_days=5.0,
            compute_event_prob=False,
        )

        assert result["run_summaries"]
        assert len(seen_sigma_eff) == 1
        np.testing.assert_allclose(seen_sigma_eff[0], sigma_eff)


class TestEdgeCases:
    """Test edge cases in event detection."""

    def test_short_light_curve(self):
        """Handle very short light curves."""
        df = make_synthetic_lc(n_points=20, n_cameras=1)
        
        result = score_lightcurve(df)
        
        # Should complete without error
        assert "dip" in result

    def test_single_point_run(self):
        """Single-point triggers should not form valid runs with min_points=2."""
        jd = np.arange(100.0)
        trig_idx = np.array([50])  # Single trigger
        
        runs = build_runs(trig_idx, jd)
        score_vec = np.zeros(100)
        score_vec[50] = 100.0
        
        kept, summaries = filter_runs(runs, jd, score_vec, min_points=2)
        
        assert len(kept) == 0

    def test_all_triggered(self):
        """Handle case where all points are triggered."""
        jd = np.arange(50.0)
        trig_idx = np.arange(50)
        
        runs = build_runs(trig_idx, jd)
        
        # Should form one big run
        assert len(runs) == 1
        assert len(runs[0]) == 50
        assert runs[0][0] == 0
        assert runs[0][-1] == 49


class TestMorphology:
    @pytest.mark.parametrize(
        ("kind", "expected_counts"),
        [("dip", [0, 3, 4]), ("jump", [0, 3, 4])],
    )
    def test_paper_parameter_counts_are_used_for_bic(
        self,
        monkeypatch,
        kind,
        expected_counts,
    ):
        jd = np.linspace(0.0, 20.0, 80)
        baseline = np.full_like(jd, 14.0)
        sign = 1.0 if kind == "dip" else -1.0
        mag = baseline + sign * 0.3 * np.exp(-0.5 * ((jd - 10.0) / 2.0) ** 2)
        err = np.full_like(jd, 0.02)
        run_idx = np.arange(30, 50)
        seen_counts = []

        def fake_curve_fit(_model, _x, _y, *, p0, **_kwargs):
            return np.asarray(p0, dtype=float), np.eye(len(p0))

        def fake_bic(_resid, _err, n_params):
            seen_counts.append(n_params)
            return [100.0, 80.0, 60.0][len(seen_counts) - 1]

        monkeypatch.setattr("malca.stv.events._curve_fit_quiet", fake_curve_fit)
        monkeypatch.setattr("malca.stv.events.bic", fake_bic)

        classify_run_morphology(
            jd,
            mag,
            err,
            run_idx,
            baseline=baseline,
            kind=kind,
        )

        assert seen_counts == expected_counts

    def test_dip_morphology_is_fit_in_gp_residual_space(self):
        jd = np.linspace(0.0, 30.0, 160)
        baseline = 14.0 + 0.02 * jd
        injected = 0.45 * np.exp(-0.5 * ((jd - 15.0) / 2.0) ** 2)
        mag = baseline + injected
        err = np.full_like(jd, 0.01)
        run_idx = np.flatnonzero(injected > 0.08)

        out = classify_run_morphology(
            jd,
            mag,
            err,
            run_idx,
            baseline=baseline,
            kind="dip",
        )

        assert out["morphology"] == "gaussian"
        assert "baseline" not in out["params"]
        assert out["params"]["amp"] == pytest.approx(0.45, rel=0.02)
        assert out["model_bics"].keys() >= {"null", "gaussian", "skew_gaussian"}

    def test_jump_bazin_fred_persists_paper_parameters(self):
        jd = np.linspace(0.0, 40.0, 240)
        baseline = 14.0 + 0.002 * jd
        injected = fred(jd, -0.65, 18.0, 1.5, 8.0)
        mag = baseline + injected
        err = np.full_like(jd, 0.008)
        run_idx = np.flatnonzero(injected < -0.04)

        out = classify_run_morphology(
            jd,
            mag,
            err,
            run_idx,
            baseline=baseline,
            kind="jump",
        )

        assert out["morphology"] == "fred"
        assert out["params"]["delta_m_peak"] == pytest.approx(-0.65, rel=0.05)
        assert out["params"]["t_peak"] == pytest.approx(18.0, abs=0.3)
        assert out["params"]["tau_rise"] == pytest.approx(1.5, rel=0.15)
        assert out["params"]["tau_fall"] == pytest.approx(8.0, rel=0.15)

    def test_jump_morphology_never_returns_gaussian(self):
        """Gaussian is not a valid winner for jump morphology."""
        jd = np.linspace(0.0, 30.0, 120)
        baseline = np.full_like(jd, 14.0)
        # Synthetic brightening-like feature (negative in mag)
        mag = baseline - 0.4 * np.exp(-0.5 * ((jd - 15.0) / 3.0) ** 2)
        err = np.full_like(jd, 0.02)
        run_idx = np.arange(45, 75)

        out = classify_run_morphology(jd, mag, err, run_idx, baseline=baseline, kind="jump")
        assert out["morphology"] != "gaussian"


def test_events_main_phase_template_uses_metadata_period_and_keeps_audit_fields(tmp_path: Path, monkeypatch) -> None:
    lc_path = tmp_path / "periodic_with_dip.dat3"
    times, mags = _periodic_dip_lightcurve()
    _write_dat3(lc_path, times, mags)

    input_file = tmp_path / "paths.txt"
    input_file.write_text(f"{lc_path}\n", encoding="ascii")

    metadata_file = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "lc_path": [str(lc_path)],
            "excluded_cameras": [""],
            "raw_median_suspect_cameras": ["5"],
            "pre_periodicity_label": ["periodic"],
            "pre_periodic_flag": [True],
            "pre_periodicity_selected_period": [6.0],
            "pre_periodicity_method": ["pdm"],
        }
    ).to_parquet(metadata_file, index=False)

    output_path = tmp_path / "lc_events_results.parquet"
    config_path = _write_events_config(
        tmp_path,
        output_format="parquet",
        baseline_func="phase_template",
        logbf_threshold_dip=2.0,
        logbf_threshold_jump=2.0,
        run_min_points=2,
    )
    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--metadata",
            str(metadata_file),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    events_main()

    out = with_feature_columns(
        pd.read_parquet(output_path),
        [
            "baseline_source",
            "pre_periodicity_label",
            "pre_periodic_flag",
            "pre_periodicity_selected_period",
            "pre_periodicity_method",
            "asassn_field_key",
            "asassn_fields",
            "asassn_field_count",
            "camera_name_key",
            "raw_median_suspect_cameras",
        ],
    )
    assert len(out) == 1
    assert out.loc[0, "baseline_source"] == "phase_template"
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert np.isclose(out.loc[0, "pre_periodicity_selected_period"], 6.0)
    assert out.loc[0, "pre_periodicity_method"] == "pdm"
    assert out.loc[0, "asassn_field_key"] == "field1"
    assert out.loc[0, "asassn_fields"] == "field1"
    assert int(out.loc[0, "asassn_field_count"]) == 1
    assert out.loc[0, "camera_name_key"] == "cam1"
    assert out.loc[0, "raw_median_suspect_cameras"] == "5"


def test_events_main_fails_on_high_error_fraction_and_only_checkpoints_successes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = ["ok.dat2", "bad1.dat2", "bad2.dat2"]
    input_file = tmp_path / "paths.txt"
    input_file.write_text("\n".join(paths) + "\n", encoding="ascii")

    output_path = tmp_path / "lc_events_results.parquet"
    config_path = _write_events_config(
        tmp_path,
        output_format="parquet",
        min_mag_offset=0,
        max_error_fraction=0.1,
    )

    def fake_process_one(path: str, path_metadata: dict | None) -> dict:
        if path.startswith("bad"):
            raise RuntimeError(f"synthetic failure for {path}")
        return {
            "lc_path": path,
            "dip_significant": False,
            "jump_significant": False,
        }

    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        events_main()

    assert exc_info.value.code == 2

    out = pd.read_parquet(output_path)
    assert out["lc_path"].tolist() == ["ok.dat2"]

    checkpoint = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")
    assert checkpoint.read_text(encoding="ascii").splitlines() == ["ok.dat2"]

    error_log = output_path.with_name(f"{output_path.stem}_ERRORS.parquet")
    errors = pd.read_parquet(error_log)
    assert set(errors["lc_path"]) == {"bad1.dat2", "bad2.dat2"}


@pytest.mark.parametrize(
    ("metadata_paths", "message"),
    [
        ([None], "null or blank"),
        (["   "], "null or blank"),
        (["same.dat2", "same.dat2"], "duplicate"),
    ],
)
def test_events_main_rejects_invalid_metadata_paths(
    tmp_path: Path,
    monkeypatch,
    metadata_paths: list[str | None],
    message: str,
) -> None:
    input_file = tmp_path / "paths.txt"
    input_file.write_text("candidate.dat2\n", encoding="ascii")
    metadata_file = tmp_path / "metadata.parquet"
    pd.DataFrame({"lc_path": metadata_paths}).to_parquet(metadata_file, index=False)
    output_path = tmp_path / "lc_events_results.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--metadata",
            str(metadata_file),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit, match=message):
        events_main()


@pytest.mark.parametrize("output_format", ["parquet", "parquet_chunk"])
def test_events_main_reconciles_resume_output_and_clears_resolved_errors(
    tmp_path: Path,
    monkeypatch,
    output_format: str,
) -> None:
    paths = ["good.dat2", "duplicate.dat2", "checkpoint_only.dat2"]
    input_file = tmp_path / "paths.txt"
    input_file.write_text("\n".join(paths) + "\n", encoding="ascii")
    output_path = (
        tmp_path / "lc_events_results.parquet"
        if output_format == "parquet"
        else tmp_path / "lc_events_results"
    )
    existing = to_layer_first_frame(
        pd.DataFrame(
            {
                "candidate_id": ["stv_good", "stv_duplicate", "stv_duplicate"],
                "timescale": ["stv", "stv", "stv"],
                "lc_path": ["good.dat2", "duplicate.dat2", "duplicate.dat2"],
                "dip_significant": [False, False, False],
                "jump_significant": [False, False, False],
            }
        )
    )
    if output_format == "parquet":
        existing.to_parquet(output_path, index=False)
    else:
        output_path.mkdir()
        existing.to_parquet(output_path / "chunk_000000.parquet", index=False)
    checkpoint = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")
    checkpoint.write_text("\n".join(paths) + "\n", encoding="ascii")
    error_output = (
        output_path.with_name(f"{output_path.stem}_ERRORS.parquet")
        if output_format == "parquet"
        else output_path.parent / f"{output_path.name}_ERRORS.parquet"
    )
    pd.DataFrame(
        {
            "lc_path": ["duplicate.dat2", "checkpoint_only.dat2", "unresolved.dat2"],
            "error": ["old duplicate", "old missing row", "still unresolved"],
            "traceback": ["", "", ""],
        }
    ).to_parquet(error_output, index=False)
    config_path = _write_events_config(
        tmp_path,
        output_format=output_format,
        min_mag_offset=0,
    )
    attempted: list[str] = []

    def fake_process_one(path: str, path_metadata: dict | None) -> dict:
        attempted.append(path)
        return {
            "lc_path": path,
            "dip_significant": False,
            "jump_significant": False,
        }

    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    events_main()

    assert set(attempted) == {"duplicate.dat2", "checkpoint_only.dat2"}
    if output_format == "parquet":
        out = pd.read_parquet(output_path)
    else:
        out = pd.concat(
            [pd.read_parquet(path) for path in sorted(output_path.glob("chunk_*.parquet"))],
            ignore_index=True,
        )
    assert sorted(out["lc_path"].tolist()) == sorted(paths)
    assert out["lc_path"].is_unique
    assert out["candidate_id"].is_unique
    assert set(checkpoint.read_text(encoding="ascii").splitlines()) == set(paths)
    unresolved = pd.read_parquet(error_output)
    assert unresolved[["lc_path", "error"]].to_dict("records") == [
        {"lc_path": "unresolved.dat2", "error": "still unresolved"}
    ]


def test_events_main_rejects_candidate_id_collisions_before_processing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_file = tmp_path / "paths.txt"
    input_file.write_text("first/shared.dat2\nsecond/shared.dat2\n", encoding="ascii")
    output_path = tmp_path / "lc_events_results.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit, match="same canonical candidate_id"):
        events_main()


def test_events_main_preserves_existing_rows_when_parquet_metadata_type_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "lc_events_results.parquet"

    def run_once(path_value: str, observer_note: str | None, priority_rank: int) -> None:
        input_file = tmp_path / f"{path_value}.txt"
        input_file.write_text(f"{path_value}\n", encoding="ascii")
        config_path = _write_events_config(tmp_path, output_format="parquet", min_mag_offset=0)
        metadata_file = tmp_path / f"{path_value}.metadata.parquet"
        pd.DataFrame(
            {
                "lc_path": [path_value],
                "observer_note": [observer_note],
                "priority_rank": [priority_rank],
            }
        ).to_parquet(metadata_file, index=False)

        def fake_process_one(path: str, path_metadata: dict | None) -> dict:
            return {
                "lc_path": path,
                "dip_significant": False,
                "jump_significant": False,
            }

        monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
        monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "malca.stv.events",
                "--input-file",
                str(input_file),
                "--metadata",
                str(metadata_file),
                "--output",
                str(output_path),
                "--config",
                str(config_path),
                "--workers",
                "1",
            ],
        )

        events_main()

    run_once("first.dat2", None, 1)
    run_once("second.dat2", "present", 2)

    out = pd.read_parquet(output_path)
    assert out["lc_path"].tolist() == ["first.dat2", "second.dat2"]
    assert "observer_note" in out.columns
    assert "priority_rank" in out.columns
    assert pd.isna(out.loc[0, "observer_note"])
    assert out.loc[1, "observer_note"] == "present"
    assert out["priority_rank"].tolist() == [1, 2]

    checkpoint = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")
    assert checkpoint.read_text(encoding="ascii").splitlines() == ["first.dat2", "second.dat2"]


def test_events_main_unexpected_worker_keys_go_to_extra_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = ["first.dat2", "second.dat2"]
    input_file = tmp_path / "paths.txt"
    input_file.write_text("\n".join(paths) + "\n", encoding="ascii")
    output_path = tmp_path / "lc_events_results.parquet"
    config_path = _write_events_config(
        tmp_path,
        output_format="parquet",
        chunk_size=1,
        min_mag_offset=0,
    )

    def fake_process_one(path: str, path_metadata: dict | None) -> dict:
        row = {
            "lc_path": path,
            "dip_significant": False,
            "jump_significant": False,
        }
        if path == "second.dat2":
            row["worker_debug_note"] = "late"
        return row

    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    events_main()

    out = pd.read_parquet(output_path)
    assert "worker_debug_note" not in out.columns
    payloads = [json.loads(value) for value in out["extra_json"]]
    assert payloads == [{}, {"worker_debug_note": "late"}]


def test_events_main_parquet_chunk_files_share_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = ["first.dat2", "second.dat2"]
    input_file = tmp_path / "paths.txt"
    input_file.write_text("\n".join(paths) + "\n", encoding="ascii")
    output_dir = tmp_path / "events_dataset"
    config_path = _write_events_config(
        tmp_path,
        output_format="parquet_chunk",
        chunk_size=1,
        min_mag_offset=0,
    )

    def fake_process_one(path: str, path_metadata: dict | None) -> dict:
        row = {
            "lc_path": path,
            "dip_significant": False,
            "jump_significant": False,
        }
        if path == "second.dat2":
            row["late_metric"] = 3.5
        return row

    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_dir),
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    events_main()

    chunk_files = sorted(output_dir.glob("chunk_*.parquet"))
    assert len(chunk_files) == 2
    schemas = [pq.read_schema(path) for path in chunk_files]
    assert schemas[0].names == schemas[1].names
    assert schemas[0].types == schemas[1].types
    assert "late_metric" not in schemas[0].names
    assert "extra_json" in schemas[0].names


def test_events_main_normalizes_older_parquet_before_append(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "lc_events_results.parquet"
    pd.DataFrame(
        {
            "lc_path": ["old.dat2"],
            "dip_significant": [False],
            "jump_significant": [False],
            "legacy_metric": ["old-value"],
        }
    ).to_parquet(output_path, index=False)

    input_file = tmp_path / "paths.txt"
    input_file.write_text("new.dat2\n", encoding="ascii")
    config_path = _write_events_config(tmp_path, output_format="parquet", min_mag_offset=0)

    def fake_process_one(path: str, path_metadata: dict | None) -> dict:
        return {
            "lc_path": path,
            "dip_significant": False,
            "jump_significant": False,
        }

    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    events_main()

    out = pd.read_parquet(output_path)
    assert out["lc_path"].tolist() == ["old.dat2", "new.dat2"]
    assert "legacy_metric" not in out.columns
    assert json.loads(out.loc[0, "extra_json"]) == {"legacy_metric": "old-value"}
    assert json.loads(out.loc[1, "extra_json"]) == {}


def test_events_main_parquet_schema_uses_canonical_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_file = tmp_path / "paths.txt"
    input_file.write_text("first.dat2\n", encoding="ascii")
    output_path = tmp_path / "lc_events_results.parquet"
    config_path = _write_events_config(tmp_path, output_format="parquet", min_mag_offset=0)

    def fake_process_one(path: str, path_metadata: dict | None) -> dict:
        return {
            "lc_path": path,
            "dip_significant": False,
            "jump_significant": False,
        }

    monkeypatch.setattr("malca.stv.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr("malca.stv.events._process_one", fake_process_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.stv.events",
            "--input-file",
            str(input_file),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            "1",
        ],
    )

    events_main()

    columns = pd.read_parquet(output_path).columns.tolist()
    assert columns == [
        "candidate_id",
        "timescale",
        "asas_sn_id",
        "lc_path",
        "extra_json",
        "lc_stats",
        "external_stats",
        "derived_stats",
        "feature_layer_version",
    ]
