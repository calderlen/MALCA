from __future__ import annotations

import numpy as np
import pandas as pd

from malca.microlensing import datasets as dataset_module
from malca.microlensing.pspl import pspl_magnification


def test_asassn_loader_detrends_with_pipeline_gp_and_combines_cameras(monkeypatch, tmp_path):
    canonical = pd.DataFrame({"sentinel": [1]})
    algorithm = pd.DataFrame(
        {
            "JD": 2_459_000.0 + np.arange(10, dtype=float),
            "mag": [15.0, 14.1, 15.2, 14.3, 17.0, 16.2, 20.0, 19.0, 14.0, 13.1],
            "error": np.full(10, 0.01),
            "good_bad": np.ones(10, dtype=int),
            "saturated": np.zeros(10, dtype=int),
            "camera#": [1, 1, 1, 1, 2, 2, 8, 9, 3, 3],
            "v_g_band": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        }
    )
    calls: list[object] = []

    def fake_load(path, *, apply_quality):
        calls.append(("load", apply_quality))
        return canonical.copy()

    def fake_convert(frame):
        calls.append("convert")
        assert frame.equals(canonical)
        return algorithm.copy()

    def fake_clean(frame):
        calls.append("clean")
        return frame.copy()

    def fake_filter_bad(frame, **kwargs):
        calls.append(("catastrophic", kwargs))
        return frame.loc[frame["camera#"] != 8].reset_index(drop=True), {8}

    baseline_calls: list[tuple[int, ...]] = []

    def fake_baseline(frame):
        baseline_calls.append(tuple(sorted(frame["camera#"].unique())))
        out = frame.copy()
        offset = out["JD"] - 2_459_000.0
        out["baseline"] = np.select(
            [out["camera#"] == 1, out["camera#"] == 2, out["camera#"] == 3],
            [15.0 + 0.1 * offset, 16.2 + 0.2 * offset, 13.2 + 0.1 * offset],
            default=20.0,
        )
        out["is_masked"] = out["mag"] < out["baseline"]
        out["sigma_eff"] = out["camera#"].map({1: 0.10, 2: 0.20, 3: 0.05, 9: 0.30})
        out["resid"] = out["mag"] - out["baseline"]
        return out

    def fake_filter_residual(frame, baseline):
        calls.append("residual")
        assert set(baseline["camera#"]) == {1, 2, 3, 9}
        return frame.loc[frame["camera#"] != 9].reset_index(drop=True), {9}

    monkeypatch.setattr(dataset_module, "load_lightcurve_df", fake_load)
    monkeypatch.setattr(dataset_module, "to_asassn_algorithm_frame", fake_convert)
    monkeypatch.setattr(dataset_module, "clean_lc", fake_clean)
    monkeypatch.setattr(dataset_module, "filter_bad_cameras", fake_filter_bad)
    monkeypatch.setattr(dataset_module, "per_camera_gp_baseline_masked", fake_baseline)
    monkeypatch.setattr(dataset_module, "filter_residual_bad_cameras", fake_filter_residual)

    path = tmp_path / "candidate.dat3"
    path.touch()
    datasets = dataset_module.load_asassn_datasets(path, min_points=2)

    assert [dataset.dataset_id for dataset in datasets] == [
        "asassn:g:combined",
        "asassn:v:combined",
    ]
    assert all(dataset.instrument == "combined_cameras" for dataset in datasets)
    assert all(dataset.reference_mag == 0.0 for dataset in datasets)
    assert baseline_calls == [(1, 2, 3, 9), (1, 2, 3)]
    assert calls[0:3] == [("load", True), "convert", "clean"]
    assert calls[3][0] == "catastrophic"
    assert calls[3][1]["filter_scatter"] is False
    assert calls[3][1]["filter_offset"] is False
    assert calls[3][1]["filter_catastrophic"] is True
    assert calls[4] == "residual"

    g_dataset = datasets[0]
    v_dataset = datasets[1]
    expected_bright_flux = 10.0**0.4
    np.testing.assert_allclose(
        g_dataset.flux,
        [1.0, expected_bright_flux, 1.0, expected_bright_flux, 1.0, expected_bright_flux],
    )
    expected_g_sigma = (np.log(10.0) / 2.5) * g_dataset.flux * np.array(
        [0.10, 0.10, 0.10, 0.10, 0.20, 0.20]
    )
    np.testing.assert_allclose(g_dataset.flux_error, expected_g_sigma)
    np.testing.assert_allclose(v_dataset.flux, [1.0, expected_bright_flux])
    assert set(g_dataset.time_jd) == set(algorithm.loc[algorithm["camera#"].isin([1, 2]), "JD"])
    assert set(v_dataset.time_jd) == set(algorithm.loc[algorithm["camera#"] == 3, "JD"])


def test_atlas_loader_and_calibration_keep_native_flux_metadata(tmp_path):
    n_rows = 180
    mjd = 59_000.0 + np.arange(n_rows, dtype=float)
    t0_jd = 2_459_090.5
    u0 = 0.25
    tE_days = 8.0
    magnification = pspl_magnification(mjd + 2_400_000.5, t0_jd, u0, tE_days)
    baseline_pattern = np.tile([-6.0, -3.0, 0.0, 3.0, 6.0], 36)
    flux_ujy = 100.0 + 25.0 * (magnification - 1.0) + baseline_pattern
    raw = pd.DataFrame(
        {
            "MJD": mjd,
            "m": np.full(n_rows, -20.0),
            "dm": np.full(n_rows, 99.0),
            "uJy": flux_ujy,
            "duJy": np.full(n_rows, 1.5),
            "F": np.full(n_rows, "c"),
            "err": np.zeros(n_rows, dtype=int),
            "chi/N": np.linspace(5.0, 500.0, n_rows),
            "x": np.full(n_rows, 500.0),
            "y": np.full(n_rows, 500.0),
            "maj": np.full(n_rows, 2.5),
            "min": np.full(n_rows, 2.2),
            "apfit": np.full(n_rows, -0.5),
            "mag5sig": np.full(n_rows, 19.0),
            "Sky": np.full(n_rows, 20.0),
            "Obs": [
                f"{'01' if index % 2 == 0 else '02'}a{index:05d}"
                for index in range(n_rows)
            ],
            "atlas_image_type": np.full(n_rows, "reduced"),
        }
    )
    raw.attrs["atlas_image_types"] = ["reduced"]
    path = tmp_path / "atlas_lc_candidate.parquet"
    raw.to_parquet(path, index=False)

    loaded = dataset_module.load_atlas_datasets(path)

    assert len(loaded) == 1
    atlas = loaded[0]
    assert atlas.dataset_id == "atlas:c:forced"
    assert atlas.n_points == n_rows
    assert set(atlas.point_metadata["atlas_obs_site_code"]) == {"01", "02"}
    np.testing.assert_allclose(
        atlas.flux * atlas.metadata["atlas_reference_flux_ujy"],
        atlas.point_metadata["atlas_flux_ujy"],
    )
    np.testing.assert_allclose(
        atlas.flux_error * atlas.metadata["atlas_reference_flux_ujy"],
        atlas.point_metadata["atlas_flux_error_formal_ujy"],
    )

    calibrated = dataset_module.calibrate_atlas_datasets(
        loaded,
        t0_jd=t0_jd,
        u0=u0,
        tE_days=tE_days,
        min_calibration_points=30,
    )

    assert len(calibrated) == 1
    atlas = calibrated[0]
    assert atlas.metadata["atlas_noise_status"] == "ok"
    assert atlas.metadata["atlas_noise_model_version"] == "atlas-empirical-noise-v1"
    assert len(atlas.metadata["atlas_calibration_counts_by_site"]) == 2
    assert np.all(atlas.point_metadata["atlas_noise_floor_ujy"] > 0.0)
    assert np.all(atlas.point_metadata["atlas_robust_weight"] > 0.0)
    assert np.all(atlas.point_metadata["atlas_robust_weight"] <= 1.0)
    assert np.all(
        atlas.point_metadata["atlas_flux_error_eff_ujy"]
        >= atlas.point_metadata["atlas_flux_error_formal_ujy"]
    )
    np.testing.assert_allclose(
        atlas.flux_error * atlas.metadata["atlas_reference_flux_ujy"],
        atlas.point_metadata["atlas_fit_error_ujy"],
    )
