from __future__ import annotations

import json
from pathlib import Path

from malca.stv.pipeline import find_latest_run_dir


def _write_run_params(run_dir: Path, params: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_params.json").write_text(json.dumps(params))


def test_find_latest_run_dir_requires_matching_fingerprint(tmp_path: Path) -> None:
    base_root = tmp_path / "output"
    current_fingerprint = {
        "version": 1,
        "params": {"mag_bin": ["13_13.5"], "min_bayes_factor": 10.0},
        "code": {"stv/pipeline.py": "same"},
    }

    older_matching = base_root / "runs" / "stv" / "20250101_000000"
    newer_mismatch = base_root / "runs" / "stv" / "20250102_000000"
    _write_run_params(
        older_matching,
        {
            "mag_bin": ["13_13.5"],
            "run_reuse_fingerprint": current_fingerprint,
        },
    )
    _write_run_params(
        newer_mismatch,
        {
            "mag_bin": ["13_13.5"],
            "run_reuse_fingerprint": {
                "version": 1,
                "params": {"mag_bin": ["13_13.5"], "min_bayes_factor": 5.0},
                "code": {"stv/pipeline.py": "same"},
            },
        },
    )

    assert find_latest_run_dir(base_root, ["13_13.5"], current_fingerprint) == older_matching


def test_find_latest_run_dir_does_not_auto_reuse_legacy_params(tmp_path: Path) -> None:
    base_root = tmp_path / "output"
    current_fingerprint = {
        "version": 1,
        "params": {"mag_bin": ["13_13.5"]},
        "code": {"stv/pipeline.py": "same"},
    }

    legacy_run = base_root / "runs" / "stv" / "20250101_000000"
    _write_run_params(legacy_run, {"mag_bin": ["13_13.5"]})

    assert find_latest_run_dir(base_root, ["13_13.5"], current_fingerprint) is None


def test_find_latest_run_dir_preserves_mag_bin_lookup_without_fingerprint(tmp_path: Path) -> None:
    base_root = tmp_path / "output"
    legacy_run = base_root / "runs" / "stv" / "20250101_000000"
    _write_run_params(legacy_run, {"mag_bin": ["13_13.5"]})

    assert find_latest_run_dir(base_root, ["13_13.5"]) == legacy_run
