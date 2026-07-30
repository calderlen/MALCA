"""Tests for malca.plotting.paper_figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.plotting.paper_figures import (
    _deredden_ir_colors,
    _galactic_radians,
    _median_g,
    default_context,
    load_survey_candidates,
    resolve_efficiency_cube_path,
    resolve_injection_recovery_path,
    resolve_injection_results_path,
)


def test_median_g_prefers_stats_photometry_median_mag() -> None:
    frame = pd.DataFrame(
        {
            "stats_photometry_median_mag": [14.0],
            "phot_g_mean_mag": [15.0],
        }
    )
    assert _median_g(frame).iloc[0] == pytest.approx(14.0)


def test_deredden_ir_colors_applies_av() -> None:
    frame = pd.DataFrame(
        {
            "A_v_3d": [2.0],
            "tmass_k": [10.0],
            "w2": [9.0],
            "w4": [8.0],
        }
    )
    out = _deredden_ir_colors(frame)
    assert out["ks_w2_0"].iloc[0] == pytest.approx((10.0 - 0.112 * 2.0) - (9.0 - 0.047 * 2.0))
    assert out["ks_w4_0"].iloc[0] == pytest.approx((10.0 - 0.112 * 2.0) - (8.0 - 0.0 * 2.0))


def test_galactic_radians_from_ra_dec() -> None:
    frame = pd.DataFrame({"ra": [83.6331], "dec": [22.0145]})
    l_rad, b_rad, good = _galactic_radians(frame)
    assert good.all()
    assert l_rad.size == 1
    assert b_rad.size == 1


def test_default_context_paths() -> None:
    ctx = default_context()
    assert ctx.run_root.name == "dat3-full-extended_2026-07-01-v4"
    assert ctx.output_dir.parts[-2:] == ("notebooks", "paper_figures")


def test_resolve_injection_results_prefers_matching_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "output" / "runs" / "test-run"
    manifest = run_root / "results" / "external_lc_manifest.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"manifest")

    def _write_results(base: Path, *, manifest_rel: str, stamp: str) -> Path:
        run_dir = base / stamp
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True)
        results_path = results_dir / "injection_results.parquet"
        pd.DataFrame(
            {
                "fractional_depth": [0.1, 0.2],
                "duration": [5.0, 10.0],
                "median_mag": [14.0, 14.5],
                "detected": [True, False],
            }
        ).to_parquet(results_path, index=False)
        (run_dir / "run_params.json").write_text(
            json.dumps({"manifest": manifest_rel}),
            encoding="utf-8",
        )
        return results_path

    old_results = _write_results(
        repo_root / "output" / "dip_injection",
        manifest_rel=str(manifest.relative_to(repo_root)),
        stamp="20260101_000000_old",
    )
    new_results = _write_results(
        repo_root / "output" / "dip_injection",
        manifest_rel=str(manifest.relative_to(repo_root)),
        stamp="20260201_000000_new",
    )

    resolved = resolve_injection_results_path(repo_root, run_root)
    assert resolved == new_results.resolve()

    recovery = resolve_injection_recovery_path(repo_root, run_root)
    assert recovery == new_results.resolve()


def test_load_efficiency_grid_depth_timescale_from_results(tmp_path: Path) -> None:
    from malca.evaluation.dip_injection import load_efficiency_grid_depth_timescale

    rng = np.random.default_rng(0)
    n = 400
    results_path = tmp_path / "injection_results.parquet"
    pd.DataFrame(
        {
            "fractional_depth": rng.uniform(0.05, 0.5, n),
            "duration": rng.uniform(2.0, 200.0, n),
            "median_mag": rng.uniform(12.0, 16.0, n),
            "detected": rng.random(n) > 0.4,
        }
    ).to_parquet(results_path, index=False)

    dur, depth, eff = load_efficiency_grid_depth_timescale(results_path)
    assert dur.ndim == 1
    assert depth.ndim == 1
    assert eff.ndim == 2
    assert eff.shape == (len(depth), len(dur))


def test_resolve_efficiency_cube_prefers_matching_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "output" / "runs" / "test-run"
    manifest = run_root / "results" / "external_lc_manifest.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"manifest")

    def _write_cube(base: Path, *, manifest_rel: str, stamp: str) -> Path:
        run_dir = base / stamp
        cubes_dir = run_dir / "cubes"
        cubes_dir.mkdir(parents=True)
        cube_path = cubes_dir / "efficiency_cube.npz"
        cube_path.write_bytes(b"npz")
        (run_dir / "run_params.json").write_text(
            json.dumps({"manifest": manifest_rel}),
            encoding="utf-8",
        )
        return cube_path

    old_cube = _write_cube(
        repo_root / "output" / "dip_injection",
        manifest_rel=str(manifest.relative_to(repo_root)),
        stamp="20260101_000000_old",
    )
    new_cube = _write_cube(
        repo_root / "output" / "dip_injection",
        manifest_rel=str(manifest.relative_to(repo_root)),
        stamp="20260201_000000_new",
    )
    _write_cube(
        repo_root / "output" / "injection",
        manifest_rel="output/other_manifest.parquet",
        stamp="20260301_000000_unrelated",
    )

    resolved = resolve_efficiency_cube_path(repo_root, run_root)
    assert resolved == new_cube.resolve()

    explicit = resolve_efficiency_cube_path(repo_root, run_root, explicit=old_cube)
    assert explicit == old_cube.resolve()


def test_resolve_efficiency_cube_explicit_missing_falls_back(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "output" / "runs" / "test-run"
    run_root.mkdir(parents=True)
    cube_path = repo_root / "output" / "dip_injection" / "20260101_000000" / "cubes" / "efficiency_cube.npz"
    cube_path.parent.mkdir(parents=True)
    cube_path.write_bytes(b"npz")

    resolved = resolve_efficiency_cube_path(
        repo_root,
        run_root,
        explicit=repo_root / "missing.npz",
    )
    assert resolved == cube_path.resolve()


def test_load_survey_candidates_sql(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "review.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                stats_amplitude REAL,
                stats_photometry_median_mag REAL,
                stats_error_and_snr_stats_error_median REAL,
                stats_variability_quasi_periodicity_q REAL,
                stats_variability_flux_asymmetry_m REAL,
                stats_variability_periodic_feature_period_source TEXT,
                ra REAL, dec REAL, gal_l REAL, gal_b REAL,
                bp_rp REAL,
                phot_g_mean_mag REAL,
                A_v_3d REAL,
                age50 REAL, period_consensus_days REAL, period_primary_source TEXT,
                tmass_k REAL, tmass_k_err REAL,
                w1 REAL, w1_err REAL, w2 REAL, w2_err REAL,
                w3 REAL, w3_err REAL, w4 REAL, w4_err REAL,
                sed_alpha REAL,
                iphas_r_ha REAL, vphas_r_ha REAL,
                pmra REAL, pmdec REAL, parallax REAL,
                distance_gspphot REAL,
                dip_best_mag_event REAL, dip_max_run_duration REAL
            );
            CREATE TABLE reviews (
                candidate_id TEXT PRIMARY KEY,
                event_class TEXT,
                status TEXT
            );
            INSERT INTO candidates VALUES (
                'c1', 0.1, 14.0, 0.02, 0.5, 0.1, NULL,
                10.0, 10.0, 180.0, 0.0,
                1.0, 14.0, 0.5,
                0.05, 10.0, NULL,
                10.0, 0.1, 9.0, 0.1, 8.5, 0.1, 8.0, 0.1, 7.5, 0.1,
                -2.0, NULL, NULL, 1.0, -1.0, 2.0, 100.0,
                0.2, 5.0
            );
            INSERT INTO reviews VALUES ('c1', 'dipper', 'reviewed');
            """
        )
    frame = load_survey_candidates(db)
    assert len(frame) == 1
    assert bool(frame["is_dipper"].iloc[0])
