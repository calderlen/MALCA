from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from malca.evaluation import dip_injection as dip
from malca.evaluation import microlensing_injection as micro


def _lightcurve(n: int = 101) -> pd.DataFrame:
    # Deliberately retain a nontrivial observed residual pattern.  Injection
    # must preserve it exactly rather than draw another noise realization.
    times = np.linspace(0.0, 100.0, n)
    return pd.DataFrame(
        {
            "JD": times,
            "mag": 14.0 + 0.01 * np.sin(times / 3.0),
            "error": np.linspace(0.01, 0.03, n),
            "camera": "a",
        }
    )


def _fake_dip_detection(frame: pd.DataFrame, _kwargs: dict, min_mag_offset: float = 0.0) -> dict:
    detected = bool(frame["mag"].max() - frame["mag"].median() > 0.05)
    return {
        "detected": detected,
        "dip_significant": detected,
        "jump_significant": False,
        "dip_bayes_factor": 10.0 if detected else 0.0,
        "jump_bayes_factor": 0.0,
        "dip_best_p": 0.5,
        "jump_best_p": 0.5,
        "baseline_mag": float(frame["mag"].median()),
        "dip_best_mag_event": float(frame["mag"].max()),
        "jump_best_mag_event": float(frame["mag"].min()),
        "dip_best_t0": float(frame.loc[frame["mag"].idxmax(), "JD"]),
        "jump_best_t0": float(frame.loc[frame["mag"].idxmin(), "JD"]),
    }


def test_trial_seed_depends_only_on_seed_and_trial_identity() -> None:
    seeds = [dip.deterministic_trial_seed(91, i) for i in (8, 2, 8, 4)]
    assert seeds[0] == seeds[2]
    assert len(set(seeds)) == 3
    assert dip.deterministic_trial_seed(92, 8) != seeds[0]


def test_dip_profile_is_not_renormalized_to_the_sampled_cadence() -> None:
    missed_peak = dip.skewnormal_dip(
        np.array([0.0, 1.0]), t_center=50.0, duration=2.0, amplitude=0.7
    )
    assert float(missed_peak.max()) < 1e-6


def test_dip_trial_records_actual_injection_support_and_paired_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _lightcurve()
    monkeypatch.setattr(dip, "_load_lc", lambda *_args, **_kwargs: observed.copy())
    monkeypatch.setattr(dip, "_default_detection_func", _fake_dip_detection)

    kwargs = dict(
        control_ids=np.array(["control-a"]),
        control_dirs=np.array(["/unused"], dtype=object),
        amp_range=(0.4, 0.4),
        dur_range=(8.0, 8.0),
        skew_range=(0.0, 0.0),
        mag_err_poly=np.poly1d([99.0]),
        detection_kwargs={},
        min_mag_offset=0.0,
        measure_pre_injection=True,
        seed=1234,
    )
    first = dip._simulate_trial(7, **kwargs)
    second = dip._simulate_trial(7, **kwargs)

    assert first == second
    assert first["trial_status"] == "completed"
    assert first["processing_error"] is False
    assert first["paired_control_evaluated"] is True
    assert first["pre_injection_detected"] is False
    assert first["detected"] is True
    assert first["injected_amplitude_mag"] == pytest.approx(0.4)
    assert first["injected_duration_days"] == pytest.approx(8.0)
    assert first["injected_t0_jd"] == pytest.approx(first["t_center"])
    assert first["n_support_points"] >= first["n_fwhm_points"]
    assert 0.0 <= first["window_coverage_fraction"] <= 1.0
    assert 0.0 <= first["observed_peak_fraction"] <= 1.0 + 1e-12


def test_processing_failure_is_not_encoded_as_a_nondetection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dip, "INJECTION_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(dip, "_load_lc", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad file")))
    result = dip._simulate_trial(
        0,
        control_ids=np.array(["broken"]),
        control_dirs=np.array(["/broken"], dtype=object),
        amp_range=(0.2, 0.2),
        dur_range=(5.0, 5.0),
        skew_range=(0.0, 0.0),
        mag_err_poly=None,
        detection_kwargs={},
        min_mag_offset=0.0,
        measure_pre_injection=True,
        seed=1,
    )
    assert result["trial_status"] == "processing_error"
    assert result["processing_error"] is True
    assert result["detected"] is None
    assert result["error_stage"] == "load_control"


def test_efficiency_summary_reports_both_denominators_and_intervals() -> None:
    results = pd.DataFrame(
        {
            "detected": pd.array([True, False, None, True, False], dtype="boolean"),
            "processing_error": [False, False, True, False, False],
            "observable": [True, False, False, True, True],
            "trial_status": ["completed", "completed", "processing_error", "completed", "completed"],
        }
    )
    summary = dip.summarize_injection_efficiency(results)
    assert summary["end_to_end"]["successes"] == 2
    assert summary["end_to_end"]["trials"] == 5
    assert summary["end_to_end"]["efficiency"] == pytest.approx(0.4)
    assert summary["completed"]["trials"] == 4
    assert summary["conditional_observable"]["trials"] == 3
    assert summary["conditional_observable"]["efficiency"] == pytest.approx(2 / 3)
    assert 0.0 <= summary["end_to_end"]["ci_low"] < 0.4
    assert 0.4 < summary["end_to_end"]["ci_high"] <= 1.0
    assert dip.binomial_confidence_interval(0, 0) == (None, None)


def test_binned_efficiency_keeps_empty_bins_missing() -> None:
    results = pd.DataFrame(
        {
            "amplitude": [0.1, 0.1, 0.9],
            "duration": [1.0, 1.0, 10.0],
            "detected": pd.array([True, False, True], dtype="boolean"),
            "processing_error": [False, False, False],
            "observable": [True, True, True],
        }
    )
    grid = dip.compute_detection_efficiency_details(
        results, amplitude_bins=3, duration_bins=3
    )
    assert np.isnan(grid["efficiency_end_to_end"][grid["n_designed"] == 0]).all()
    assert np.isnan(grid["end_to_end_ci_low"][grid["n_designed"] == 0]).all()


def test_parquet_checkpoint_writer_unions_schema_and_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "trials.parquet"
    writer = dip.ParquetAppendWriter(output)
    writer.write_chunk(
        [{"trial_index": 1, "trial_status": "processing_error", "error": "first"}]
    )
    writer.write_chunk(
        [
            {"trial_index": 0, "trial_status": "completed", "detected": True},
            {"trial_index": 1, "trial_status": "completed", "detected": False},
        ]
    )
    saved = pd.read_parquet(output)
    assert saved["trial_index"].tolist() == [0, 1]
    assert "error" in saved.columns
    assert "detected" in saved.columns
    assert saved.loc[saved["trial_index"].eq(1), "trial_status"].item() == "completed"


def test_resume_rejects_a_different_experiment_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "trials.parquet"
    pd.DataFrame(
        {"trial_index": [0], "experiment_fingerprint": ["configuration-a"]}
    ).to_parquet(output, index=False)
    dip._assert_resume_fingerprint(output, "configuration-a")
    with pytest.raises(ValueError, match="different experiment configuration"):
        dip._assert_resume_fingerprint(output, "configuration-b")


def test_paczynski_injection_uses_t0_tE_u0_without_extra_noise() -> None:
    observed = _lightcurve()
    first = micro.inject_paczynski(
        observed, 50.0, 10.0, u0=0.4, rng=np.random.default_rng(1)
    )
    second = micro.inject_paczynski(
        observed, 50.0, 10.0, u0=0.4, rng=np.random.default_rng(999)
    )
    pd.testing.assert_frame_equal(first, second)
    expected_delta = micro._paczynski_delta_mag(
        observed["JD"].to_numpy(), t0=50.0, tE=10.0, u0=0.4
    )
    np.testing.assert_allclose(first["mag"] - observed["mag"], expected_delta)
    assert micro._Amax_from_u0(0.4) == pytest.approx(
        micro._Amax_from_u0(micro._solve_u0_from_A0(micro._Amax_from_u0(0.4)))
    )


def test_microlensing_trial_records_physical_parameters_and_paired_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lc_path = tmp_path / "control.dat"
    lc_path.touch()
    observed = _lightcurve()
    seen_frames: list[pd.DataFrame] = []
    monkeypatch.setattr(
        micro, "_prepare_lightcurve_df", lambda *_args, **_kwargs: (observed.copy(), "g")
    )

    def fake_fit(context: dict) -> dict:
        seen_frames.append(context["df"].copy())
        return {
            "summary": {
                "fit_ok": True,
                "best_model": "paczynski",
                "paczynski_reduced_chi2": 1.0,
                "raw_paczynski_tE_days": 12.0,
                "fit_t0_jd": float(
                    context["df"].loc[context["df"]["mag"].idxmin(), "JD"]
                ),
            }
        }

    monkeypatch.setattr(micro, "fit_candidate_context", fake_fit)
    result = micro._simulate_microlensing_trial(
        3,
        control_ids=np.array(["control"]),
        control_dirs=np.array([str(lc_path)], dtype=object),
        Amax_range=(2.0, 2.0),
        tE_range=(12.0, 12.0),
        mag_err_poly=np.poly1d([50.0]),
        measure_pre_injection=True,
        seed=44,
    )
    assert result["trial_status"] == "completed"
    assert result["injected_tE_days"] == pytest.approx(12.0)
    assert result["injected_Amax"] == pytest.approx(2.0)
    assert result["injected_u0"] == pytest.approx(micro._solve_u0_from_A0(2.0))
    assert result["injected_t0_jd"] == pytest.approx(result["t0"])
    assert result["paired_control_evaluated"] is True
    assert result["post_injection_recovered"] is True
    assert result["recovered"] is False
    assert len(seen_frames) == 2
    pd.testing.assert_frame_equal(seen_frames[0], observed)
    expected = micro.inject_paczynski(
        observed, result["t0"], result["tE"], u0=result["u0"]
    )
    pd.testing.assert_frame_equal(seen_frames[1], expected)


def test_microlensing_event_rate_requires_explicit_exposure() -> None:
    results = pd.DataFrame(
        {
            "recovered": pd.array([True, False, None, True], dtype="boolean"),
            "processing_error": [False, False, True, False],
            "observable": [True, True, False, True],
        }
    )
    with pytest.raises(ValueError, match="exposure"):
        micro.calculate_event_rate(results, n_observed_events=10)
    rate = micro.calculate_event_rate(
        results, n_observed_events=10, exposure_star_years=100.0
    )
    assert rate["recovery_efficiency"] == pytest.approx(0.5)
    assert rate["event_rate_per_star_year"] == pytest.approx(0.2)


def test_microlensing_grid_does_not_fill_unmeasured_bins() -> None:
    results = pd.DataFrame(
        {
            "tE": [1.0, 1.0, 100.0],
            "Amax": [1.1, 1.1, 10.0],
            "recovered": pd.array([True, False, True], dtype="boolean"),
            "processing_error": [False, False, False],
            "observable": [True, True, True],
        }
    )
    grid = micro.compute_microlensing_efficiency_grid(
        results, bins_tE=4, bins_Amax=4
    )
    empty = grid["n_designed"] == 0
    assert empty.any()
    assert np.isnan(grid["efficiency_end_to_end"][empty]).all()
    assert np.isnan(grid["end_to_end_ci_low"][empty]).all()


def test_efficiency_plotting_leaves_empty_cells_masked(tmp_path: Path) -> None:
    fig = dip.plot_efficiency_jointplot(
        np.array([1.5, 3.0]),
        np.array([0.25, 0.75]),
        np.array([1.0, 2.0, 4.0]),
        np.array([0.0, 0.5, 1.0]),
        np.array([[0.1, np.nan], [0.6, 0.9]]),
        xlog=True,
        show=False,
        contour_kwargs={"levels": [0.5], "manual1": [], "manual2": []},
    )
    plotted = fig.axes[0].collections[0].get_array()
    assert np.ma.getmaskarray(plotted).sum() == 1
    plt.close(fig)

    micro_results = pd.DataFrame(
        {
            "tE": [1.0] * 5 + [100.0] * 5,
            "Amax": [1.1] * 5 + [10.0] * 5,
            "recovered": [False] * 5 + [True] * 5,
            "processing_error": False,
            "observable": True,
        }
    )
    out = tmp_path / "microlensing_efficiency.pdf"
    grid = micro.plot_efficiency_map(
        micro_results, out, bins_tE=3, bins_Amax=3, min_bin_trials=2
    )
    assert out.exists()
    assert (grid["n_designed"] == 0).any()
