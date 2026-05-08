"""Tests for event detection functions."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.events import (
    build_runs,
    classify_run_morphology,
    filter_runs,
    main as events_main,
    score_events_bayesian,
    score_lightcurve,
)
from malca.baseline import per_camera_gp_baseline


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

        monkeypatch.setattr("malca.events.resolve_trigger_indices", fake_resolve_trigger_indices)
        monkeypatch.setattr(
            "malca.events.classify_run_morphology",
            lambda *args, **kwargs: {"morphology": "test", "bic": 0.0, "delta_bic_null": 0.0, "params": {}},
        )
        monkeypatch.setattr("malca.events.compute_symmetry_score", lambda *args, **kwargs: 0.0)

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

        monkeypatch.setattr("malca.events.resolve_trigger_indices", fake_resolve_trigger_indices)
        monkeypatch.setattr("malca.events.classify_run_morphology", fake_classify_run_morphology)
        monkeypatch.setattr("malca.events.compute_symmetry_score", lambda *args, **kwargs: 0.0)

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

    metadata_file = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "path": [str(lc_path)],
            "excluded_cameras": [""],
            "pre_periodicity_label": ["periodic"],
            "pre_periodic_flag": [True],
            "pre_periodicity_selected_period": [6.0],
            "pre_periodicity_method": ["pdm"],
        }
    ).to_csv(metadata_file, index=False)

    output_path = tmp_path / "lc_events_results.parquet"
    monkeypatch.setattr("malca.events.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca.events",
            "--input-file",
            str(input_file),
            "--metadata-csv",
            str(metadata_file),
            "--output",
            str(output_path),
            "--output-format",
            "parquet",
            "--baseline-func",
            "phase_template",
            "--workers",
            "1",
            "--logbf-threshold-dip",
            "2.0",
            "--logbf-threshold-jump",
            "2.0",
            "--run-min-points",
            "2",
        ],
    )

    events_main()

    out = pd.read_parquet(output_path)
    assert len(out) == 1
    assert out.loc[0, "baseline_source"] == "phase_template"
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert np.isclose(out.loc[0, "pre_periodicity_selected_period"], 6.0)
    assert out.loc[0, "pre_periodicity_method"] == "pdm"
