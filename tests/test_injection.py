from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.baseline import per_camera_gp_baseline_masked
from malca.evaluation import detection_rate
from malca.evaluation import injection

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_detection_args() -> argparse.Namespace:
    return argparse.Namespace(
        trigger_mode="posterior_prob",
        logbf_threshold_dip=5.0,
        logbf_threshold_jump=5.0,
        significance_threshold=5.0,
        p_points=12,
        p_min_dip=None,
        p_max_dip=None,
        p_min_jump=None,
        p_max_jump=None,
        mag_points=12,
        mag_min_dip=None,
        mag_max_dip=None,
        mag_min_jump=None,
        mag_max_jump=None,
        run_min_points=3,
        run_max_gap_points=2,
        run_max_gap_days=None,
        run_min_duration_days=0.0,
        baseline_func="gp_masked",
        baseline_s0=0.002,
        baseline_w0=0.003,
        baseline_q=1.0 / np.sqrt(2.0),
        baseline_jitter=0.01,
        baseline_sigma_floor=None,
        no_event_prob=False,
        min_mag_offset=0.2,
    )


def test_inject_dip_is_deterministic_with_explicit_rng() -> None:
    df = pd.DataFrame(
        {
            "JD": np.linspace(0.0, 20.0, 32),
            "mag": np.full(32, 14.0),
            "error": np.full(32, 0.02),
        }
    )

    first = injection.inject_dip(df, 10.0, 3.0, 0.4, rng=np.random.default_rng(123))
    second = injection.inject_dip(df, 10.0, 3.0, 0.4, rng=np.random.default_rng(123))
    different = injection.inject_dip(df, 10.0, 3.0, 0.4, rng=np.random.default_rng(456))

    pd.testing.assert_frame_equal(first, second)
    assert not np.allclose(first["mag"].to_numpy(), different["mag"].to_numpy())


def test_injection_and_detection_rate_use_gp_masked_baseline() -> None:
    args = _base_detection_args()

    assert injection._build_detection_kwargs(args)["baseline_func"] is per_camera_gp_baseline_masked
    assert detection_rate._build_detection_kwargs(args)["baseline_func"] is per_camera_gp_baseline_masked


def test_injection_baseline_names_fail_loudly() -> None:
    args = _base_detection_args()
    args.baseline_func = "does_not_exist"

    with pytest.raises(ValueError, match="Unsupported baseline_func"):
        injection._build_detection_kwargs(args)
    with pytest.raises(ValueError, match="Unsupported baseline_func"):
        detection_rate._build_detection_kwargs(args)


def test_run_injection_recovery_keeps_in_memory_rows_when_chunking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_sample = pd.DataFrame({"source_id": ["control-1"], "lc_dir": [str(tmp_path)]})
    lc = pd.DataFrame(
        {
            "JD": np.linspace(0.0, 10.0, 12),
            "mag": np.full(12, 14.0),
            "error": np.full(12, 0.02),
        }
    )

    monkeypatch.setattr(injection, "_load_lc", lambda *_args, **_kwargs: lc.copy())

    def fake_simulate_trial(trial_index: int, **_kwargs: object) -> dict[str, int]:
        return {"trial_index": int(trial_index)}

    monkeypatch.setattr(injection, "_simulate_trial", fake_simulate_trial)

    out = injection.run_injection_recovery(
        control_sample,
        detection_kwargs={},
        total_trials=3,
        workers=1,
        chunk_size=1,
        checkpoint_interval=1,
        output_path=None,
        checkpoint_path=None,
        mag_err_order=0,
        show_progress=False,
    )

    assert out is not None
    assert out["trial_index"].tolist() == [0, 1, 2]


def test_injection_cli_exposes_diagnostics_detection_flags() -> None:
    help_text = subprocess.run(
        [sys.executable, "-m", "malca", "injection", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    for flag in (
        "--baseline-func",
        "--logbf-threshold-dip",
        "--measure-pre-injection",
        "--total-trials",
    ):
        assert flag in help_text

    for flag in ("--amp-steps", "--dur-steps", "--n-injections-per-grid", "--max-trials"):
        assert flag not in help_text


def test_diagnostics_scripts_use_current_cli_entrypoints() -> None:
    script_text = "\n".join(path.read_text() for path in (REPO_ROOT / "diagnostics").glob("*.sh"))

    assert "python -m malca.injection" not in script_text
    assert "python -m malca.detection_rate" not in script_text
    assert "--sample-size" not in script_text
