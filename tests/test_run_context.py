from __future__ import annotations

import json
from pathlib import Path

from malca.products.run_context import (
    init_pipeline_run_context,
    run_dir_from_bundle,
    timestamped_run_dir,
    update_latest_symlink,
    write_run_log,
    write_run_params,
    write_run_summary,
)


def test_timestamped_run_dir_is_under_root(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "stv"
    run_dir = timestamped_run_dir(root)

    assert run_dir.parent == root
    assert len(run_dir.name) == len("20260527_123456")


def test_run_dir_from_bundle_strips_bundle_suffix_and_handles_collision(tmp_path: Path) -> None:
    root = tmp_path / "runs" / "stv"
    bundle = tmp_path / "20260527_120000_bundle.zip"

    assert run_dir_from_bundle(bundle, root) == root / "20260527_120000"

    (root / "20260527_120000").mkdir(parents=True)
    assert run_dir_from_bundle(bundle, root, collision_suffix="_home") == root / "20260527_120000_home"
    assert run_dir_from_bundle(bundle, root, collision_suffix="_home", overwrite=True) == root / "20260527_120000"
    assert run_dir_from_bundle(bundle, root, collision_suffix=None) == root / "20260527_120000"


def test_run_context_writes_ascii_json_and_logs(tmp_path: Path) -> None:
    ctx = init_pipeline_run_context("stv", tmp_path / "run")
    tagged = ctx.run_dir / "run_params_tagged.json"

    write_run_params(ctx, {"name": "Cafe\u00e9", "n": 1}, extra_paths=[tagged])
    write_run_log(ctx, ["first", "second"])
    write_run_summary(ctx, {"status": "ok"})

    assert json.loads(ctx.run_params_file.read_text(encoding="ascii")) == {"name": "Cafe\u00e9", "n": 1}
    assert tagged.read_bytes().isascii()
    assert ctx.run_log_file.read_text(encoding="ascii") == "first\nsecond\n"
    assert json.loads(ctx.run_summary_file.read_text(encoding="ascii")) == {"status": "ok"}


def test_update_latest_symlink_is_best_effort(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ltv" / "20260527_120000"
    run_dir.mkdir(parents=True)
    latest = tmp_path / "runs" / "ltv" / "latest"

    update_latest_symlink(run_dir, latest, label="LTV")

    assert latest.exists()
    assert latest.resolve() == run_dir.resolve()
