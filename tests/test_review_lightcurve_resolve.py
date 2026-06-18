from __future__ import annotations

from pathlib import Path

from malca.review.interactive_plot import _bundle_lightcurve_dir, resolve_lightcurve_path


def test_bundle_lightcurve_dir_accepts_run_root_or_plots_anchor(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs_march18_bundle_all"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    (run_dir / "plots").mkdir()

    assert _bundle_lightcurve_dir(run_dir) == bundle_dir
    assert _bundle_lightcurve_dir(run_dir / "plots") == bundle_dir


def test_resolve_lightcurve_path_falls_back_to_bundle_when_stored_path_is_stale(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs_march18_bundle_all"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    lc_path = bundle_dir / "549755992463.dat3"
    lc_path.write_text("stub", encoding="utf-8")

    payload = {
        "candidate_id": "549755992463",
        "lc_path": "/home/calder/code/malca/output/runs/runs_march18_bundle_all/bundle_assets/lightcurves/549755992463.dat3",
    }

    assert resolve_lightcurve_path(payload, run_dir) == lc_path
    assert resolve_lightcurve_path(payload, run_dir / "plots") == lc_path
