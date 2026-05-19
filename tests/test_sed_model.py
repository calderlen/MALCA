from __future__ import annotations

import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import malca.sed_photometry as sed_photometry
from malca.review.sed import build_sed_dataframe
from malca.sed_model import (
    LSUN_ERG_S,
    PC_CM,
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    PystellibsSetupError,
    _patch_pystellibs_kurucz_libsdir,
    fit_sed_models,
)
from malca.table_io import write_parquet_table

_trapezoid = getattr(np, "trapezoid", np.trapz)


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
        ("Pan-STARRS", "g", 4810.0),
        ("Pan-STARRS", "r", 6170.0),
        ("Pan-STARRS", "i", 7520.0),
        ("Pan-STARRS", "z", 8660.0),
    ]
    rows = []
    for source, band, lam in optical:
        spec = float(np.interp(lam, library._wavelength, spectrum))
        l_lam = scale * lam * spec
        rows.append({
            "candidate_id": candidate_id,
            "source": source,
            "band": band,
            "mag": np.nan,
            "mag_err": np.nan,
            "mag_system": "AB",
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

    fits, curves = fit_sed_models(candidate, sed_rows, library=FakeKurucz(), curve_points=96)

    fit = fits.iloc[0]
    assert fit["status"] == "ok"
    assert int(fit["n_fit_points"]) == 5
    assert abs(float(fit["teff_k"]) - 6000.0) < 400.0
    assert abs(float(fit["scale"]) - 2.5) / 2.5 < 0.15
    assert float(fit["luminosity_lsun"]) > 0
    fit_bands = json.loads(fit["fit_bands_json"])
    assert all(item["source"] != "AllWISE" for item in fit_bands)
    assert not curves.empty
    assert set(SED_MODEL_CURVE_COLUMNS).issubset(curves.columns)


def test_kurucz_fitter_reports_insufficient_data_with_eligible_bands() -> None:
    payload = {
        "candidate_id": "sed-two-bands",
        "apass_b": 15.2,
        "apass_v": 14.8,
        "tmass_j": 12.4,
        "w1": 11.9,
        "distance_gspphot": 1000.0,
        "A_v_3d": 0.0,
    }
    candidate = pd.DataFrame([payload])
    sed_rows = build_sed_dataframe(payload, candidate_id="sed-two-bands", extinction_mode="observed")

    fits, curves = fit_sed_models(candidate, sed_rows, library=FakeKurucz(), curve_points=32)

    fit = fits.iloc[0]
    assert fit["status"] == "insufficient_data"
    assert "Need at least 3 finite optical photospheric SED points" in str(fit["warning"])
    assert "found 2: APASS B,V" in str(fit["warning"])
    assert curves.empty


def test_missing_pystellibs_raises_actionable_error(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, *args, **kwargs):
        if name == "pystellibs":
            raise ModuleNotFoundError("No module named 'pystellibs'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("malca.sed_model.importlib.import_module", fake_import_module)
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
    write_parquet_table(pd.DataFrame([{"candidate_id": "sed-cand", "ra_deg": 1.0, "dec_deg": 2.0}]), input_path)
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

    monkeypatch.setattr(sed_photometry, "fetch_sed_photometry", lambda *args, **kwargs: sed_rows)
    monkeypatch.setattr(sed_photometry, "fit_sed_models", lambda *args, **kwargs: (fits, curves))

    args = sed_photometry.build_arg_parser().parse_args([str(input_path), "--sources", "payload"])
    output_path = sed_photometry.run(args)
    fit_path, curve_path = sed_photometry._default_model_output_paths(output_path)

    assert output_path.exists()
    assert fit_path.exists()
    assert curve_path.exists()
