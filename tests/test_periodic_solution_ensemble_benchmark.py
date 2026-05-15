from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from malca.evaluation.periodic_branch_simulation_benchmark import generate_trial_design
from malca.evaluation.periodic_solution_ensemble_benchmark import (
    PeriodicSolutionBenchmarkConfig,
    PeriodicSolutionBenchmarkRun,
    select_baseline_gallery_trials,
    write_solution_baseline_gallery,
)


def test_select_baseline_gallery_trials_is_limited_and_unique() -> None:
    rows = []
    for trial_id in range(12):
        rows.append(
            {
                "trial_id": trial_id,
                "mode": "current_template_true_period",
                "status": "ok",
                "has_dip": trial_id % 3 != 0,
                "truth_observable_actual": trial_id % 2 == 0,
                "target_recovered": trial_id in {2, 4},
                "off_target_detection": trial_id == 5,
                "false_positive": trial_id == 6,
                "phase_local_detected": trial_id in {7, 8},
                "dip_bayes_factor": float(trial_id),
                "dip_amp_mag": 0.05 * trial_id,
                "phase_local_truth_peak_snr": float(12 - trial_id),
                "waveform_kind": "sinusoid" if trial_id % 2 else "spot_like",
                "dip_amp_bin": "small" if trial_id % 2 else "medium",
                "period_bin": "<2d" if trial_id % 2 else "2-5d",
            }
        )
    df = pd.DataFrame(rows)

    selected = select_baseline_gallery_trials(df, n_trials=6, seed=42)

    assert len(selected) == 6
    assert len(set(selected)) == len(selected)
    assert set(selected).issubset(set(df["trial_id"]))


def test_write_solution_baseline_gallery_smoke(tmp_path: Path) -> None:
    modes = ("current_template_true_period", "fourier_3harmonic")
    config = PeriodicSolutionBenchmarkConfig(
        output_base_dir=tmp_path,
        run_tag="smoke",
        n_trials=1,
        seed=20260514,
        workers=1,
        show_progress=False,
        mode_names=modes,
        compute_event_prob=True,
    )
    design = generate_trial_design(config)
    run = PeriodicSolutionBenchmarkRun(
        config=config,
        run_dir=tmp_path,
        trial_design=design,
        solution_results=pd.DataFrame(),
        summary_overall=pd.DataFrame(),
        summary_slices={},
    )

    paths = write_solution_baseline_gallery(
        run,
        trial_ids=[int(design["trial_id"].iloc[0])],
        modes=modes,
        output_dir=tmp_path / "gallery",
        dpi=60,
        show_progress=False,
    )

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].stat().st_size > 0
