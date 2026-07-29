from __future__ import annotations

import importlib
import json
import math
import sys
import types
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import malca.enrichment.sed_photometry as sed_photometry
from malca.review.sed import SED_BANDPASSES, build_sed_dataframe
from malca.review.store import db_connect, upsert_candidates_frame
from malca.enrichment.sed_model import (
    NORMALIZATION_VERSION,
    LSUN_ERG_S,
    PC_CM,
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    SED_MODEL_FIT_VERSION,
    SED_MODEL_POINT_COLUMNS,
    PystellibsSetupError,
    _extend_stellar_spectrum_rayleigh_jeans,
    _finalize_point_rows,
    _is_stellar_fit_band,
    _patch_pystellibs_kurucz_libsdir,
    _prepare_candidate_points,
    fit_sed_models,
)
from malca.enrichment.photometric_calibration import mission_quoted_fnu_calibration
from malca.enrichment.synthetic_photometry import (
    FilterResponse,
    apply_extinction,
    bandpass_flux_nu_jy,
    response_pivot_wavelength_angstrom,
    top_hat_response,
)
from malca.io.table_io import read_parquet_table, write_feature_table

_trapezoid = getattr(np, "trapezoid", np.trapz)
_FILTER_CENTERS = {
    bandpass.svo_filter_id: bandpass.lambda_eff_angstrom
    for bandpass in SED_BANDPASSES.values()
    if bandpass.svo_filter_id
}


def _test_response_loader(filter_id: str, mag_system: str) -> FilterResponse:
    center = _FILTER_CENTERS[filter_id]
    return top_hat_response(filter_id, center, max(20.0, center * 0.001), mag_system=mag_system)


class FakeKurucz:
    source = "/tmp/fake-kurucz2004.grid.fits"

    def __init__(self) -> None:
        self._wavelength = np.geomspace(2500.0, 250000.0, 2500)

    @property
    def logT(self):
        return np.array([3.55, 3.65, 3.75, 3.9, 4.1, 4.4])

    @property
    def logg(self):
        return np.array([0.0, 2.5, 4.5])

    @property
    def Z(self):
        return np.array([0.02])

    def generate_stellar_spectrum(self, logT, logg, logL, Z, raise_extrapolation=False):
        wave = self._wavelength
        teff = 10.0 ** float(logT)
        c2 = 1.438776877e8
        exponent = np.clip(c2 / (wave * teff), 1.0e-4, 700.0)
        shape = 1.0 / (np.power(wave, 5.0) * np.expm1(exponent))
        shape = shape / _trapezoid(shape, wave)
        return shape * LSUN_ERG_S * (10.0 ** float(logL))


def _rows_from_fake_model(candidate_id: str, *, teff: float = 6000.0, scale: float = 2.5) -> pd.DataFrame:
    library = FakeKurucz()
    spectrum = library.generate_stellar_spectrum(math.log10(teff), 4.5, 0.0, 0.02)
    distance_cm = 1000.0 * PC_CM
    optical = [
        ("Gaia GSPC", "SDSS_u", 3543.0),
        ("APASS", "V", 5500.0),
        ("Pan-STARRS", "g", 4810.0),
        ("Pan-STARRS", "r", 6170.0),
        ("Pan-STARRS", "i", 7520.0),
        ("Pan-STARRS", "z", 8660.0),
        ("Pan-STARRS", "y", 9620.0),
        ("2MASS", "J", 12350.0),
    ]
    rows = []
    for source, band, lam in optical:
        spec = float(np.interp(lam, library._wavelength, spectrum))
        l_lam = scale * lam * spec
        mag_system = "Vega" if source in {"APASS", "2MASS"} else "AB"
        rows.append({
            "candidate_id": candidate_id,
            "source": source,
            "band": band,
            "mag": np.nan,
            "mag_err": np.nan,
            "mag_system": mag_system,
            "lambda_eff_angstrom": lam,
            "flux_lambda": scale * spec / (4.0 * math.pi * distance_cm * distance_cm),
            "flux_lambda_err": 0.05 * scale * spec / (4.0 * math.pi * distance_cm * distance_cm),
            "lambda_l_lambda": l_lam,
            "lambda_l_lambda_err": 0.05 * l_lam,
            "flux_nu_jy": np.nan,
            "flux_nu_jy_err": np.nan,
            "sep_arcsec": 0.0,
            "is_synthetic": int(source == "Gaia GSPC"),
            "is_upper_limit": 0,
            "quality_flags": "",
            "svo_filter_id": "",
            "av_coeff": 0.0,
        })
    lam = 33526.0
    spec = float(np.interp(lam, library._wavelength, spectrum))
    rows.append({
        "candidate_id": candidate_id,
        "source": "AllWISE",
        "band": "W1",
        "mag": np.nan,
        "mag_err": np.nan,
        "mag_system": "Vega",
        "lambda_eff_angstrom": lam,
        "flux_lambda": 25.0 * scale * spec / (4.0 * math.pi * distance_cm * distance_cm),
        "flux_lambda_err": 0.05 * scale * spec / (4.0 * math.pi * distance_cm * distance_cm),
        "lambda_l_lambda": 25.0 * scale * lam * spec,
        "lambda_l_lambda_err": 0.05 * scale * lam * spec,
        "flux_nu_jy": np.nan,
        "flux_nu_jy_err": np.nan,
        "sep_arcsec": 0.0,
        "is_synthetic": 0,
        "is_upper_limit": 0,
        "quality_flags": "",
        "svo_filter_id": "",
        "av_coeff": 0.061,
    })
    return pd.DataFrame(rows)


def test_finalize_point_rows_parses_persisted_false_boolean_strings() -> None:
    rows = pd.DataFrame(
        {
            "candidate_id": ["false-word", "zero-word", "true-word"],
            "used": ["False", "0", "true"],
        }
    )

    finalized = _finalize_point_rows(rows, "run")

    assert finalized["used"].tolist() == [0, 0, 1]


def test_kurucz_fitter_ignores_ir_excess_and_recovers_teff_scale() -> None:
    candidate = pd.DataFrame([{
        "candidate_id": "sed-cand",
        "teff_gspphot": 5900.0,
        "logg_gspphot": 4.5,
        "mh_gspphot": 0.0,
        "distance_gspphot": 1000.0,
        "A_v_3d": 0.0,
    }])
    sed_rows = _rows_from_fake_model("sed-cand", teff=6000.0, scale=2.5)

    fits, curves, points = fit_sed_models(
        candidate,
        sed_rows,
        library=FakeKurucz(),
        curve_points=96,
        response_loader=_test_response_loader,
        allow_bandpass_download=False,
        return_points=True,
    )

    fit = fits.iloc[0]
    assert fit["status"] == "ok"
    assert int(fit["n_fit_points"]) == 6
    assert fit["fit_version"] == SED_MODEL_FIT_VERSION
    covariance = np.asarray(json.loads(fit["fit_covariance_json"]), dtype=float)
    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, covariance.T)
    assert str(fit["fit_covariance_status"]) in {"ok", "singular"}
    assert abs(float(fit["teff_k"]) - 6000.0) < 400.0
    assert abs(float(fit["scale"]) - 2.5) / 2.5 < 0.15
    assert float(fit["luminosity_lsun"]) > 0
    fit_bands = json.loads(fit["fit_bands_json"])
    assert all(item["source"] != "2MASS" for item in fit_bands)
    assert all(item["source"] != "AllWISE" for item in fit_bands)
    assert not curves.empty
    assert float(curves["wavelength_angstrom"].min()) == pytest.approx(
        float(FakeKurucz()._wavelength[0])
    )
    assert float(curves["wavelength_angstrom"].max()) == pytest.approx(
        float(FakeKurucz()._wavelength[-1])
    )
    assert int((curves["wavelength_angstrom"] > 25000.0).sum()) > 10
    assert set(SED_MODEL_CURVE_COLUMNS).issubset(curves.columns)
    assert set(SED_MODEL_POINT_COLUMNS).issubset(points.columns)
    assert points.loc[points["source"] == "AllWISE", "exclusion_reason"].iloc[0] == "ir_excess_diagnostic"
    assert points.loc[points["source"] == "2MASS", "exclusion_reason"].iloc[0] == "ir_excess_diagnostic"
    assert np.isfinite(points["model_flux_nu_jy"]).all()


def test_post_fit_rayleigh_jeans_tail_covers_far_ir_diagnostic_only() -> None:
    native_wave = np.array([100000.0, 175000.0, 250000.0])
    native_flux = np.array([16.0, 4.0, 1.0])
    target_wave = 2200000.0
    extended_wave, extended_flux = _extend_stellar_spectrum_rayleigh_jeans(
        native_wave,
        native_flux,
        target_wave,
    )

    assert np.array_equal(extended_wave[: len(native_wave)], native_wave)
    assert np.array_equal(extended_flux[: len(native_flux)], native_flux)
    assert extended_wave[len(native_wave) - 1] == native_wave[-1]
    assert extended_flux[len(native_flux) - 1] == native_flux[-1]
    assert extended_wave[-1] == target_wave
    tail_slope = np.diff(np.log(extended_flux[len(native_flux) - 1 :])) / np.diff(
        np.log(extended_wave[len(native_wave) - 1 :])
    )
    assert np.allclose(tail_slope, -4.0, rtol=0.0, atol=1.0e-12)

    candidate = pd.DataFrame([{
        "candidate_id": "sed-far-ir-tail",
        "teff_gspphot": 5900.0,
        "logg_gspphot": 4.5,
        "mh_gspphot": 0.0,
        "distance_gspphot": 1000.0,
        "A_v_3d": 0.0,
    }])
    rows = _rows_from_fake_model("sed-far-ir-tail", teff=6000.0, scale=2.5)
    rows = pd.concat(
        [
            rows,
            pd.DataFrame([{
                "candidate_id": "sed-far-ir-tail",
                "source": "Herschel",
                "band": "PACS160",
                "mag_system": "Jy",
                "observable_kind": "quoted_fnu",
                "lambda_eff_angstrom": 1600000.0,
                "lambda_reference_angstrom": 1600000.0,
                "flux_nu_jy": 0.01,
                "flux_nu_jy_err": 0.001,
                "sep_arcsec": 0.0,
                "is_synthetic": 0,
                "is_upper_limit": 0,
                "quality_flags": "",
                "svo_filter_id": "Herschel/Pacs.red",
            }]),
        ],
        ignore_index=True,
    )
    far_response = FilterResponse(
        filter_id="Herschel/Pacs.red",
        wavelength_angstrom=np.array([1200000.0, 1600000.0, target_wave]),
        throughput=np.array([0.0, 1.0, 0.0]),
        detector_type="energy",
        mag_system="Jy",
    )

    def response_loader(filter_id: str, mag_system: str) -> FilterResponse:
        if filter_id == far_response.filter_id:
            return far_response
        return _test_response_loader(filter_id, mag_system)

    fits, curves, points = fit_sed_models(
        candidate,
        rows,
        library=FakeKurucz(),
        curve_points=96,
        response_loader=response_loader,
        allow_bandpass_download=False,
        return_points=True,
    )

    fit = fits.iloc[0]
    far_point = points[(points["source"] == "Herschel") & (points["band"] == "PACS160")].iloc[0]
    assert fit["status"] == "ok"
    assert int(fit["n_fit_points"]) == 6
    assert float(fit["fit_lambda_max"]) <= 10000.0
    assert not bool(far_point["used"])
    assert far_point["exclusion_reason"] == "fit_policy:diagnostic_only"
    assert far_point["prediction_status"] == "ok"
    assert np.isfinite(float(far_point["model_flux_nu_jy"]))
    assert float(curves["wavelength_angstrom"].min()) == float(FakeKurucz()._wavelength[0])
    assert float(curves["wavelength_angstrom"].max()) == target_wave
    assert int(
        (
            (curves["wavelength_angstrom"] > native_wave[-1])
            & (curves["wavelength_angstrom"] < target_wave)
        ).sum()
    ) > 10


def test_kurucz_bandpass_fit_recovers_joint_temperature_extinction_and_scale() -> None:
    library = FakeKurucz()
    teff_true = 5200.0
    av_true = 1.2
    scale_true = 1.8
    distance_pc = 750.0
    distance_cm = distance_pc * PC_CM
    wave = library._wavelength
    intrinsic = library.generate_stellar_spectrum(math.log10(teff_true), 4.5, 0.0, 0.02)
    observed = apply_extinction(wave, intrinsic, av_true)
    apparent_scale = scale_true / (4.0 * math.pi * distance_cm**2)
    selected = [
        ("Pan-STARRS", "g"),
        ("Pan-STARRS", "r"),
        ("Pan-STARRS", "i"),
        ("Pan-STARRS", "z"),
        ("Pan-STARRS", "y"),
    ]
    rows = []
    for source, band in selected:
        bandpass = SED_BANDPASSES[f"{source.lower()}:{band.lower()}"]
        response = _test_response_loader(str(bandpass.svo_filter_id), bandpass.mag_system)
        fnu = apparent_scale * bandpass_flux_nu_jy(wave, observed, response)
        rows.append({
            "candidate_id": "sed-extincted",
            "source": source,
            "band": band,
            "mag": np.nan,
            "mag_err": np.nan,
            "mag_system": bandpass.mag_system,
            "lambda_eff_angstrom": bandpass.lambda_eff_angstrom,
            "flux_nu_jy": fnu,
            "flux_nu_jy_err": 0.03 * fnu,
            "flux_lambda": np.nan,
            "flux_lambda_err": np.nan,
            "lambda_l_lambda": np.nan,
            "lambda_l_lambda_err": np.nan,
            "sep_arcsec": 0.0,
            "is_synthetic": 0,
            "is_upper_limit": 0,
            "quality_flags": "",
            "svo_filter_id": bandpass.svo_filter_id,
            "av_coeff": bandpass.av_coeff,
        })
    candidate = pd.DataFrame([{
        "candidate_id": "sed-extincted",
        "teff_gspphot": 5600.0,
        "logg_gspphot": 4.5,
        "distance_gspphot": distance_pc,
        "A_v_3d": 0.4,
    }])

    fits, _curves = fit_sed_models(
        candidate,
        pd.DataFrame(rows),
        library=library,
        response_loader=_test_response_loader,
        allow_bandpass_download=False,
    )
    fit = fits.iloc[0]

    assert fit["status"] == "ok"
    assert float(fit["teff_k"]) == pytest.approx(teff_true, abs=150.0)
    assert float(fit["av_fit"]) == pytest.approx(av_true, abs=0.12)
    assert float(fit["scale"]) == pytest.approx(scale_true, rel=0.08)


def test_kurucz_bandpass_fit_records_robust_outlier_decision() -> None:
    candidate = pd.DataFrame([{
        "candidate_id": "sed-outlier",
        "teff_gspphot": 6000.0,
        "distance_gspphot": 1000.0,
        "A_v_3d": 0.0,
    }])
    rows = _rows_from_fake_model("sed-outlier", teff=6000.0, scale=2.5)
    bad = (rows["source"] == "Pan-STARRS") & (rows["band"] == "g")
    rows.loc[bad, "flux_lambda"] *= 20.0
    rows.loc[bad, "flux_lambda_err"] *= 20.0

    fits, _curves, points = fit_sed_models(
        candidate,
        rows,
        library=FakeKurucz(),
        response_loader=_test_response_loader,
        allow_bandpass_download=False,
        return_points=True,
    )

    fit = fits.iloc[0]
    outlier = points[(points["source"] == "Pan-STARRS") & (points["band"] == "g")].iloc[0]
    assert fit["status"] == "ok"
    assert int(fit["n_rejected_points"]) == 1
    assert not bool(outlier["used"])
    assert outlier["exclusion_reason"] == "robust_outlier"


def test_kurucz_fitter_rejects_correlated_and_pointed_sed_sources() -> None:
    base = {
        "band": "sample",
        "lambda_eff_angstrom": 5000.0,
        "is_upper_limit": 0,
    }

    assert not _is_stellar_fit_band(pd.Series({**base, "source": "Gaia XP", "quality_flags": "correlated_spectrum"}))
    assert not _is_stellar_fit_band(pd.Series({**base, "source": "Swift/UVOT", "quality_flags": "non_simultaneous_pointed"}))
    assert not _is_stellar_fit_band(pd.Series({**base, "source": "XMM-OM", "quality_flags": "non_simultaneous_pointed"}))


def test_new_fitter_normalization_uses_current_pivot_and_actual_mission_calibration() -> None:
    apass_response = FilterResponse(
        filter_id="Generic/Johnson.B",
        wavelength_angstrom=np.array([3800.0, 4300.0, 4900.0]),
        throughput=np.array([0.0, 1.0, 0.0]),
        detector_type="energy",
        mag_system="Vega",
        zero_point_jy=4063.0,
    )
    spitzer_response = FilterResponse(
        filter_id="Spitzer/IRAC.I1",
        wavelength_angstrom=np.array([31000.0, 35500.0, 40000.0]),
        throughput=np.array([0.0, 1.0, 0.0]),
        detector_type="energy",
        mag_system="Jy",
    )
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "sed-current-response",
                "source": "APASS",
                "band": "B",
                "mag": 12.0,
                "mag_err": 0.03,
                "mag_system": "Vega",
                "observable_kind": "vega_mag",
                "lambda_pivot_angstrom": 9999.0,
                "plot_lambda_angstrom": 9999.0,
                "plot_lambda_kind": "response_pivot",
                "svo_filter_id": "Generic/Johnson.B",
            },
            {
                "candidate_id": "sed-current-response",
                "source": "Spitzer SEIP",
                "band": "IRAC1",
                "mag_system": "Jy",
                "observable_kind": "quoted_fnu",
                "flux_nu_jy": 2.0,
                "flux_nu_jy_err": 0.1,
                "lambda_reference_angstrom": 35500.0,
                "lambda_pivot_angstrom": 99999.0,
                "plot_lambda_angstrom": 35500.0,
                "plot_lambda_kind": "mission_reference",
                "svo_filter_id": "Spitzer/IRAC.I1",
                "calibration_id": "Spitzer/IRAC.I1:quoted_fnu",
            },
        ]
    )
    responses = {
        ("Generic/Johnson.B", "Vega"): apass_response,
        ("Spitzer/IRAC.I1", "Jy"): spitzer_response,
    }

    prepared = _prepare_candidate_points(
        "sed-current-response",
        {"candidate_id": "sed-current-response"},
        rows,
        responses,
        {},
    ).set_index("source")

    apass = prepared.loc["APASS"]
    spitzer = prepared.loc["Spitzer SEIP"]
    assert float(apass["lambda_pivot_angstrom"]) == pytest.approx(
        response_pivot_wavelength_angstrom(apass_response)
    )
    assert float(apass["plot_lambda_angstrom"]) == pytest.approx(
        response_pivot_wavelength_angstrom(apass_response)
    )
    assert float(apass["lambda_pivot_angstrom"]) != pytest.approx(9999.0)

    mission = mission_quoted_fnu_calibration("Spitzer/IRAC.I1", 35500.0)
    assert float(spitzer["lambda_pivot_angstrom"]) == pytest.approx(
        response_pivot_wavelength_angstrom(spitzer_response)
    )
    assert float(spitzer["plot_lambda_angstrom"]) == pytest.approx(35500.0)
    assert spitzer["plot_lambda_kind"] == "mission_reference"
    assert spitzer["calibration_id"] == mission.calibration_id
    assert spitzer["calibration_hash"] == mission.calibration_hash
    provenance = json.loads(str(spitzer["normalization_provenance_json"]))
    assert provenance["calibration_id"] == mission.calibration_id
    assert str(spitzer["normalization_version"]).startswith(f"{NORMALIZATION_VERSION}:")
    assert spitzer["normalization_method"] == "fitter_bandpass_calibrated_v8"


def test_kurucz_fitter_reports_insufficient_data_with_eligible_bands() -> None:
    payload = {
        "candidate_id": "sed-two-bands",
        "apass_b": 15.2,
        "apass_v": 14.8,
        "w1": 11.9,
        "distance_gspphot": 1000.0,
        "A_v_3d": 0.0,
    }
    candidate = pd.DataFrame([payload])
    sed_rows = build_sed_dataframe(payload, candidate_id="sed-two-bands", extinction_mode="observed")

    fits, curves = fit_sed_models(
        candidate,
        sed_rows,
        library=FakeKurucz(),
        curve_points=32,
        response_loader=_test_response_loader,
        allow_bandpass_download=False,
    )

    fit = fits.iloc[0]
    assert fit["status"] == "insufficient_data"
    assert "Need at least 5 bandpass-calibrated photospheric SED points" in str(fit["warning"])
    assert "found 2: APASS B,V" in str(fit["warning"])
    assert curves.empty


def test_missing_pystellibs_raises_actionable_error(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == "pystellibs":
            raise ModuleNotFoundError("No module named 'pystellibs'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("malca.enrichment.sed_model.importlib.import_module", fake_import_module)
    candidate = pd.DataFrame([{"candidate_id": "sed-cand"}])
    sed_rows = _rows_from_fake_model("sed-cand")

    with pytest.raises(PystellibsSetupError, match="pystellibs is required"):
        fit_sed_models(candidate, sed_rows)


def test_pystellibs_kurucz_libsdir_falls_back_to_packaged_libs(tmp_path: Path, monkeypatch) -> None:
    package_root = tmp_path / "pystellibs"
    (package_root / "libs").mkdir(parents=True)
    (package_root / "ezunits" / "libs").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="ascii")
    packaged_grid = package_root / "libs" / "kurucz2004.grid.fits"
    packaged_grid.write_text("grid", encoding="ascii")

    fake_pystellibs = types.SimpleNamespace(__file__=str(package_root / "__init__.py"))
    fake_config = types.SimpleNamespace(libsdir=str(package_root / "ezunits" / "libs"))
    fake_kurucz = types.SimpleNamespace(libsdir=str(package_root / "ezunits" / "libs"))
    monkeypatch.setitem(sys.modules, "pystellibs.config", fake_config)
    monkeypatch.setitem(sys.modules, "pystellibs.kurucz", fake_kurucz)

    _patch_pystellibs_kurucz_libsdir(fake_pystellibs)

    assert fake_config.libsdir == str(package_root / "libs")
    assert fake_kurucz.libsdir == str(package_root / "libs")


def test_sed_photometry_cli_writes_fit_and_curve_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.parquet"
    write_feature_table(pd.DataFrame([{
        "candidate_id": "sed-cand", "timescale": "stv", "ra": 1.0, "dec": 2.0,
        "failed_any": False,
    }]), input_path)
    sed_rows = _rows_from_fake_model("sed-cand")
    fits = pd.DataFrame([{col: None for col in SED_MODEL_FIT_COLUMNS}])
    fits.loc[0, "candidate_id"] = "sed-cand"
    fits.loc[0, "model_family"] = "Castelli/Kurucz 2004"
    fits.loc[0, "status"] = "ok"
    curves = pd.DataFrame([{col: None for col in SED_MODEL_CURVE_COLUMNS}])
    curves.loc[0, "candidate_id"] = "sed-cand"
    curves.loc[0, "model_family"] = "Castelli/Kurucz 2004"
    curves.loc[0, "wavelength_angstrom"] = 5000.0
    curves.loc[0, "lambda_l_lambda"] = 1.0e33
    curves.loc[0, "flux_lambda"] = 1.0e-12
    points = pd.DataFrame([{col: None for col in SED_MODEL_POINT_COLUMNS}])
    points.loc[0, "candidate_id"] = "sed-cand"
    points.loc[0, "fit_version"] = SED_MODEL_FIT_VERSION
    points.loc[0, "source"] = "Pan-STARRS"
    points.loc[0, "band"] = "g"

    monkeypatch.setattr(sed_photometry, "fetch_sed_photometry", lambda *args, **kwargs: sed_rows)
    monkeypatch.setattr(sed_photometry, "fit_sed_models", lambda *args, **kwargs: (fits, curves, points))

    args = sed_photometry.build_arg_parser().parse_args([str(input_path), "--sources", "payload"])
    output_path = sed_photometry.run(args)
    fit_path, curve_path = sed_photometry._default_model_output_paths(output_path)
    point_path = sed_photometry._default_model_point_output_path(output_path)

    assert output_path.exists()
    assert fit_path.exists()
    assert curve_path.exists()
    assert point_path.exists()
    layer_cols = {"lc_stats", "external_stats", "derived_stats", "feature_layer_version"}
    sed_out = read_parquet_table(output_path)
    fit_out = read_parquet_table(fit_path)
    curve_out = read_parquet_table(curve_path)
    point_out = read_parquet_table(point_path)
    assert len(sed_out) == len(sed_rows)
    assert len(fit_out) == len(fits)
    assert len(curve_out) == len(curves)
    assert len(point_out) == len(points)
    assert layer_cols.isdisjoint(sed_out.columns)
    assert layer_cols.isdisjoint(fit_out.columns)
    assert layer_cols.isdisjoint(curve_out.columns)


def test_sed_photometry_cli_writes_alpha_output_without_atmosphere_fit(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.parquet"
    write_feature_table(
        pd.DataFrame([{
            "candidate_id": "sed-alpha-cand", "timescale": "stv", "ra": 1.0, "dec": 2.0,
            "failed_any": False,
        }]),
        input_path,
    )
    sed_rows = pd.DataFrame(
        [
            {
                "candidate_id": "sed-alpha-cand",
                "source": "Test",
                "band": f"b{idx}",
                "lambda_eff_angstrom": wave_micron * 1.0e4,
                "lambda_l_lambda": wave_micron ** -0.5,
                "flux_lambda": wave_micron ** -0.5 / (wave_micron * 1.0e4),
                "is_upper_limit": False,
                "is_synthetic": False,
            }
            for idx, wave_micron in enumerate((2.159, 3.4, 12.0, 22.0))
        ]
    )

    monkeypatch.setattr(sed_photometry, "fetch_sed_photometry", lambda *args, **kwargs: sed_rows)

    args = sed_photometry.build_arg_parser().parse_args(
        [str(input_path), "--sources", "payload", "--no-fit-atmosphere"]
    )
    output_path = sed_photometry.run(args)
    alpha_path = sed_photometry._default_alpha_output_path(output_path)
    alpha_rows = pd.read_parquet(alpha_path)

    assert output_path.exists()
    assert alpha_path.exists()
    sed_out = read_parquet_table(output_path)
    assert len(sed_out) == len(sed_rows)
    assert {"lc_stats", "external_stats", "derived_stats", "feature_layer_version"}.isdisjoint(sed_out.columns)
    assert alpha_rows.loc[0, "sed_alpha_status"] == "ok"
    assert alpha_rows.loc[0, "sed_alpha"] == pytest.approx(-0.5, abs=1.0e-12)
    assert {"lc_stats", "external_stats", "derived_stats", "feature_layer_version"}.isdisjoint(alpha_rows.columns)


def test_sed_photometry_cli_detaches_fetch_manifest_before_chunked_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "candidates.parquet"
    write_feature_table(
        pd.DataFrame([{
            "candidate_id": "sed-manifest-cand",
            "timescale": "stv",
            "ra": 1.0,
            "dec": 2.0,
            "failed_any": False,
        }]),
        input_path,
    )
    sed_rows = pd.DataFrame(
        [
            {
                "candidate_id": "sed-manifest-cand",
                "source": "Payload",
                "band": f"b{idx}",
                "lambda_eff_angstrom": 5000.0 + idx,
                "flux_lambda": 1.0e-12,
                "quality_flags": None if idx < 2 else "late-object-value",
            }
            for idx in range(3)
        ]
    )

    def fake_fetch(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        out = sed_rows.copy()
        out.attrs[sed_photometry.SED_FETCH_MANIFEST_ATTR] = (
            sed_photometry.build_sed_fetch_manifest(
                df,
                sources=["payload"],
                fetched_rows=out,
            )
        )
        return out

    real_write_parquet_table = sed_photometry.write_parquet_table

    def force_chunked_write(df: pd.DataFrame, path: Path) -> None:
        real_write_parquet_table(df, path, chunk_rows=2)

    monkeypatch.setattr(sed_photometry, "fetch_sed_photometry", fake_fetch)
    monkeypatch.setattr(sed_photometry, "write_parquet_table", force_chunked_write)

    args = sed_photometry.build_arg_parser().parse_args(
        [
            str(input_path),
            "--sources",
            "payload",
            "--no-fit-atmosphere",
            "--no-alpha",
        ]
    )
    output_path = sed_photometry.run(args)
    manifest_path = sed_photometry._default_fetch_manifest_output_path(output_path)

    assert output_path.exists()
    assert manifest_path.exists()
    assert len(read_parquet_table(output_path)) == len(sed_rows)
    manifest = read_parquet_table(manifest_path)
    assert len(manifest) == 1
    assert manifest.loc[0, "candidate_id"] == "sed-manifest-cand"
    assert bool(manifest.loc[0, "is_complete"])


def test_sed_photometry_cli_writes_empty_plain_output(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.parquet"
    write_feature_table(
        pd.DataFrame([{
            "candidate_id": "sed-empty-cand", "timescale": "stv", "ra": 1.0, "dec": 2.0,
            "failed_any": False,
        }]),
        input_path,
    )
    monkeypatch.setattr(
        sed_photometry,
        "fetch_sed_photometry",
        lambda *args, **kwargs: pd.DataFrame(columns=sed_photometry.CANONICAL_SED_COLUMNS),
    )

    args = sed_photometry.build_arg_parser().parse_args(
        [str(input_path), "--sources", "payload", "--no-fit-atmosphere"]
    )
    output_path = sed_photometry.run(args)

    out = read_parquet_table(output_path)
    assert out.empty
    assert set(sed_photometry.CANONICAL_SED_COLUMNS).issubset(out.columns)
    assert {"lc_stats", "external_stats", "derived_stats", "feature_layer_version"}.isdisjoint(out.columns)


def test_sed_photometry_cli_reads_candidates_from_review_db_by_default(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with closing(db_connect(db_path)) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame([{
                "candidate_id": "sed-cand",
                "ra": 1.0,
                "dec": 2.0,
                "gaia_id": "123456789",
                "failed_any": False,
            }]),
        )

    sed_rows = _rows_from_fake_model("sed-cand")
    fits = pd.DataFrame([{col: None for col in SED_MODEL_FIT_COLUMNS}])
    fits.loc[0, "candidate_id"] = "sed-cand"
    fits.loc[0, "model_family"] = "Castelli/Kurucz 2004"
    fits.loc[0, "status"] = "ok"
    curves = pd.DataFrame([{col: None for col in SED_MODEL_CURVE_COLUMNS}])
    curves.loc[0, "candidate_id"] = "sed-cand"
    curves.loc[0, "model_family"] = "Castelli/Kurucz 2004"
    curves.loc[0, "wavelength_angstrom"] = 5000.0
    curves.loc[0, "lambda_l_lambda"] = 1.0e33
    curves.loc[0, "flux_lambda"] = 1.0e-12
    points = pd.DataFrame([{col: None for col in SED_MODEL_POINT_COLUMNS}])
    points.loc[0, "candidate_id"] = "sed-cand"
    points.loc[0, "fit_version"] = SED_MODEL_FIT_VERSION
    points.loc[0, "source"] = "Pan-STARRS"
    points.loc[0, "band"] = "g"

    seen: dict[str, object] = {}

    def fake_fetch(df: pd.DataFrame, *args, **kwargs):
        seen["input_rows"] = len(df)
        seen["candidate_id"] = str(df.loc[0, "candidate_id"])
        seen["ra"] = float(df.loc[0, "ra"])
        seen["dec"] = float(df.loc[0, "dec"])
        seen["gaia_id"] = str(df.loc[0, "gaia_id"])
        return sed_rows

    monkeypatch.setattr(sed_photometry, "fetch_sed_photometry", fake_fetch)
    monkeypatch.setattr(sed_photometry, "fit_sed_models", lambda *args, **kwargs: (fits, curves, points))

    args = sed_photometry.build_arg_parser().parse_args([str(db_path), "--sources", "payload"])
    output_path = sed_photometry.run(args)

    assert output_path.exists()
    assert seen == {
        "input_rows": 1,
        "candidate_id": "sed-cand",
        "ra": 1.0,
        "dec": 2.0,
        "gaia_id": "123456789",
    }
    with closing(db_connect(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sed_photometry").fetchone()[0] == len(sed_rows)
        assert conn.execute("SELECT COUNT(*) FROM sed_model_fits").fetchone()[0] == len(fits)
