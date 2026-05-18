from __future__ import annotations

import math
import sys
import types
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from malca.review.sed import (
    LSUN_ERG_S,
    SED_COLUMNS,
    bandpass_for,
    build_sed_dataframe,
    build_sed_figure,
    distance_pc_from_payload,
    extinction_av_from_payload,
    flux_lambda_from_flux_nu_jy,
    flux_nu_jy_from_mag,
    load_sed_rows,
    query_gaia_gspc_photometry,
    resolve_sed_sources,
    rows_from_payload,
    upsert_sed_rows,
)
from malca.review.pipeline import detect_sed_model_status, detect_sed_photometry_status
from malca.review.store import db_connect
from malca.sed_model import (
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    load_sed_model_curves,
    load_sed_model_fits,
    upsert_sed_model_results,
)


def test_ab_and_vega_flux_conversions() -> None:
    ps1_g = bandpass_for("Pan-STARRS", "g")
    wise_w1 = bandpass_for("AllWISE", "W1")

    assert ps1_g is not None
    assert wise_w1 is not None
    assert math.isclose(flux_nu_jy_from_mag(0.0, ps1_g), 3631.0, rel_tol=1e-12)
    assert math.isclose(flux_nu_jy_from_mag(0.0, wise_w1), 309.540, rel_tol=1e-12)
    assert flux_lambda_from_flux_nu_jy(3631.0, ps1_g.lambda_eff_angstrom) > 0


def test_sed_sources_always_include_far_ir_catalogs() -> None:
    default_sources = set(resolve_sed_sources("default"))
    custom_sources = set(resolve_sed_sources("payload,ps1"))

    assert "payload" in default_sources
    assert "ps1" in default_sources
    assert {"akari", "iras", "herschel"}.issubset(default_sources)
    assert {"akari", "iras", "herschel"}.issubset(custom_sources)
    assert set(resolve_sed_sources("far-ir")) == set(resolve_sed_sources("default"))


def test_rows_from_payload_computes_luminosity_with_distance() -> None:
    payload = {
        "candidate_id": "cand-1",
        "phot_g_mean_mag": 15.0,
        "tmass_j": 12.5,
        "tmass_j_err": 0.04,
        "w1": 11.7,
        "w1_err": 0.03,
        "distance_gspphot": 1000.0,
    }

    rows = rows_from_payload(payload)

    assert {"Gaia DR3", "2MASS", "AllWISE"}.issubset(set(rows["source"]))
    assert np.isfinite(rows["flux_lambda"]).all()
    assert np.isfinite(rows["lambda_l_lambda"]).all()


def test_sed_luminosity_plot_uses_solar_units() -> None:
    payload = {"candidate_id": "cand-lsun", "w1": 11.7, "distance_gspphot": 1000.0}

    fig, rows, warnings = build_sed_figure(payload)

    assert not warnings
    assert "L_{\\odot}" in fig.layout.yaxis.title.text
    assert len(fig.data) == 1
    plotted_y = float(fig.data[0].y[0])
    expected_y = float(rows["lambda_l_lambda"].iloc[0]) / LSUN_ERG_S
    assert math.isclose(plotted_y, expected_y, rel_tol=1.0e-12)
    assert fig.data[0].marker.opacity == 1.0
    assert fig.data[0].marker.size >= 10


def test_distance_fallback_uses_positive_parallax() -> None:
    assert math.isclose(distance_pc_from_payload({"parallax": 10.0}), 100.0)
    assert distance_pc_from_payload({"parallax": -2.0}) is None


def test_extinction_fallback_uses_ebv_when_av_missing() -> None:
    assert math.isclose(extinction_av_from_payload({"ebv_3d": 0.2}), 0.62)
    assert extinction_av_from_payload({"A_v_3d": 0.0, "ebv_3d": 0.2}) == 0.0


def test_observed_mode_leaves_wise_ir_points_unchanged() -> None:
    payload = {
        "candidate_id": "cand-2",
        "w1": 10.0,
        "w1_err": 0.01,
        "A_v_3d": 5.0,
        "distance_gspphot": 1000.0,
    }

    observed = rows_from_payload(payload, extinction_mode="observed")
    corrected = rows_from_payload(payload, extinction_mode="corrected")

    assert float(observed.loc[observed["band"] == "W1", "mag"].iloc[0]) == 10.0
    assert float(corrected.loc[corrected["band"] == "W1", "mag"].iloc[0]) == 10.0 - 5.0 * 0.061


def test_missing_distance_plots_flux_only_with_warning() -> None:
    payload = {"candidate_id": "cand-3", "w1": 11.0}

    fig, rows, warnings = build_sed_figure(payload)

    assert not rows.empty
    assert rows["lambda_l_lambda"].isna().all()
    assert any("No distance available" in warning for warning in warnings)
    assert "F_{\\lambda}" in fig.layout.yaxis.title.text
    assert "\\mathring{\\mathrm{A}}" in fig.layout.xaxis.title.text
    assert "\\mathring{\\mathrm{A}}" in fig.layout.yaxis.title.text


def test_external_rows_merge_with_payload() -> None:
    payload = {"candidate_id": "cand-4", "w1": 12.0, "distance_gspphot": 1000.0}
    external = pd.DataFrame([
        {
            "candidate_id": "cand-4",
            "source": "Pan-STARRS",
            "band": "g",
            "mag": 17.2,
            "mag_err": 0.03,
            "mag_system": "AB",
            "lambda_eff_angstrom": 4810.0,
        }
    ])

    rows = build_sed_dataframe(payload, external_rows=external)

    assert set(rows["source"]) == {"AllWISE", "Pan-STARRS"}
    assert np.isfinite(rows["flux_lambda"]).all()


def test_gaia_gspc_adapter_uses_aip_available_columns(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeTapService:
        def __init__(self, url: str) -> None:
            captured["url"] = url

        def search(self, query: str):
            captured["query"] = query
            table = Table(rows=[
                (
                    123,
                    17.0, 2.0e-30, 4.0e-32,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    16.2, 3.0e-30, 9.0e-32,
                )
            ], names=[
                "source_id",
                "u_sdss_mag", "u_sdss_flux", "u_sdss_flux_error",
                "g_sdss_mag", "g_sdss_flux", "g_sdss_flux_error",
                "r_sdss_mag", "r_sdss_flux", "r_sdss_flux_error",
                "i_sdss_mag", "i_sdss_flux", "i_sdss_flux_error",
                "z_sdss_mag", "z_sdss_flux", "z_sdss_flux_error",
                "y_ps1_mag", "y_ps1_flux", "y_ps1_flux_error",
            ])
            return types.SimpleNamespace(to_table=lambda: table)

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)

    rows = query_gaia_gspc_photometry(
        pd.DataFrame([{"asas_sn_id": "cand-gspc", "gaia_id": "123", "distance_gspphot": 1000.0}])
    )

    assert captured["url"] == "https://gea.esac.esa.int/tap-server/tap"
    assert "g_ps1_mag" not in captured["query"]
    assert "u_sdss_mag_error" not in captured["query"]
    assert "u_sdss_flux_error" in captured["query"]
    assert "y_ps1_mag" in captured["query"]
    assert set(rows["band"]) == {"SDSS_u", "PS1_y"}
    assert np.isfinite(rows["mag_err"]).all()
    assert rows["is_synthetic"].all()


def test_nonpositive_magnitude_errors_are_treated_as_missing() -> None:
    payload = {"candidate_id": "cand-4a", "distance_gspphot": 1000.0}
    external = pd.DataFrame([
        {
            "candidate_id": "cand-4a",
            "source": "Pan-STARRS",
            "band": "g",
            "mag": 17.2,
            "mag_err": -999.0,
            "mag_system": "AB",
            "lambda_eff_angstrom": 4810.0,
        }
    ])

    rows = build_sed_dataframe(payload, external_rows=external)

    assert pd.isna(rows.loc[0, "mag_err"])
    assert pd.isna(rows.loc[0, "flux_nu_jy_err"])


def test_jy_catalog_rows_roundtrip_as_flux_density() -> None:
    payload = {"candidate_id": "cand-4b", "distance_gspphot": 1000.0}
    external = pd.DataFrame([
        {
            "candidate_id": "cand-4b",
            "source": "AKARI",
            "band": "S9W",
            "mag": 18.0,
            "mag_system": "AB",
            "flux_nu_jy": 2.0,
            "flux_nu_jy_err": 0.2,
            "lambda_eff_angstrom": 90000.0,
            "quality_flags": "confusion_risk;flux_catalog",
        }
    ])

    rows = build_sed_dataframe(payload, external_rows=external)

    assert len(rows) == 1
    assert math.isclose(float(rows.loc[0, "flux_nu_jy"]), 2.0)
    assert rows.loc[0, "mag_system"] == "AB"
    assert "confusion_risk" in rows.loc[0, "quality_flags"]


def test_sed_rows_roundtrip_review_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    row = {col: None for col in SED_COLUMNS}
    row.update({
        "candidate_id": "cand-5",
        "source": "Pan-STARRS",
        "band": "g",
        "mag": 17.2,
        "mag_system": "AB",
        "lambda_eff_angstrom": 4810.0,
        "flux_lambda": 1.0e-16,
    })

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-5", "{}", "2026-05-14T00:00:00"),
        )
        assert detect_sed_photometry_status(conn, "cand-5") == "missing"
        assert detect_sed_photometry_status(conn, "cand-5", {"sed_photometry_checked": True}) == "complete"
        assert upsert_sed_rows(conn, pd.DataFrame([row])) == 1
        assert detect_sed_photometry_status(conn, "cand-5") == "complete"
        loaded = load_sed_rows(conn, "cand-5")

    assert len(loaded) == 1
    assert loaded.loc[0, "source"] == "Pan-STARRS"
    assert loaded.loc[0, "band"] == "g"


def test_sed_model_rows_roundtrip_review_db_and_overlay(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    fit = {col: None for col in SED_MODEL_FIT_COLUMNS}
    fit.update({
        "candidate_id": "cand-6",
        "model_family": "Castelli/Kurucz 2004",
        "teff_k": 5750.0,
        "logg": 4.5,
        "z": 0.02,
        "av_fixed": 0.0,
        "scale": 1.0,
        "luminosity_lsun": 1.0,
        "radius_rsun": 1.0,
        "chi2": 1.2,
        "reduced_chi2": 0.4,
        "n_fit_points": 5,
        "fit_lambda_min": 3500.0,
        "fit_lambda_max": 9000.0,
        "fit_bands_json": "[]",
        "status": "ok",
        "warning": "",
    })
    curves = pd.DataFrame([
        {
            "candidate_id": "cand-6",
            "model_family": "Castelli/Kurucz 2004",
            "wavelength_angstrom": wave,
            "lambda_l_lambda": value,
            "flux_lambda": value * 1.0e-45,
            "teff_k": 5750.0,
            "scale": 1.0,
        }
        for wave, value in [(3500.0, 1.0e33), (5500.0, 2.0e33), (9000.0, 8.0e32)]
    ], columns=SED_MODEL_CURVE_COLUMNS)

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-6", "{}", "2026-05-14T00:00:00"),
        )
        assert detect_sed_model_status(conn, "cand-6") == "missing"
        n_fit, n_curve = upsert_sed_model_results(conn, pd.DataFrame([fit]), curves)
        assert n_fit == 1
        assert n_curve == 3
        assert detect_sed_model_status(conn, "cand-6") == "complete"
        loaded_fits = load_sed_model_fits(conn, "cand-6")
        loaded_curves = load_sed_model_curves(conn, "cand-6")

    fig, _rows, warnings = build_sed_figure(
        {"candidate_id": "cand-6", "distance_gspphot": 1000.0},
        external_rows=pd.DataFrame([
            {
                "candidate_id": "cand-6",
                "source": "Pan-STARRS",
                "band": "g",
                "mag": 17.2,
                "mag_system": "AB",
                "lambda_eff_angstrom": 4810.0,
                "lambda_l_lambda": 1.5e33,
            }
        ]),
        model_curve_rows=loaded_curves,
        model_fit_rows=loaded_fits,
        extinction_mode="corrected",
    )

    assert any("Castelli/Kurucz" in str(trace.name) for trace in fig.data)
    assert any("CK fit" in warning for warning in warnings)

    observed_fig, _rows, observed_warnings = build_sed_figure(
        {"candidate_id": "cand-6", "distance_gspphot": 1000.0},
        external_rows=pd.DataFrame([
            {
                "candidate_id": "cand-6",
                "source": "Pan-STARRS",
                "band": "g",
                "mag": 17.2,
                "mag_system": "AB",
                "lambda_eff_angstrom": 4810.0,
                "lambda_l_lambda": 1.5e33,
            }
        ]),
        model_curve_rows=loaded_curves,
        model_fit_rows=loaded_fits,
        extinction_mode="observed",
    )

    assert not any("Castelli/Kurucz" in str(trace.name) for trace in observed_fig.data)
    assert any("dereddened" in warning for warning in observed_warnings)


def test_sed_axis_crop_uses_photometry_not_model_extent() -> None:
    external_rows = pd.DataFrame(
        [
            {
                "candidate_id": "cand-crop",
                "source": "Catalog",
                "band": f"b{idx}",
                "mag": 14.0 + idx,
                "mag_system": "AB",
                "lambda_eff_angstrom": wave,
            }
            for idx, wave in enumerate([5000.0, 10000.0, 20000.0])
        ]
    )
    model_curve_rows = pd.DataFrame(
        [
            {
                "candidate_id": "cand-crop",
                "model_family": "Castelli/Kurucz 2004",
                "wavelength_angstrom": wave,
                "lambda_l_lambda": value,
                "flux_lambda": value * 1.0e-45,
                "teff_k": 6000.0,
                "scale": 1.0,
            }
            for wave, value in [(100.0, 1.0e28), (5000.0, 1.0e33), (1.0e6, 1.0e28)]
        ],
        columns=SED_MODEL_CURVE_COLUMNS,
    )
    model_fit_rows = pd.DataFrame(
        [
            {
                **{col: None for col in SED_MODEL_FIT_COLUMNS},
                "candidate_id": "cand-crop",
                "model_family": "Castelli/Kurucz 2004",
                "status": "ok",
                "n_fit_points": 3,
            }
        ],
        columns=SED_MODEL_FIT_COLUMNS,
    )

    fig, rows, _warnings = build_sed_figure(
        {"candidate_id": "cand-crop", "distance_gspphot": 1000.0},
        external_rows=external_rows,
        model_curve_rows=model_curve_rows,
        model_fit_rows=model_fit_rows,
        extinction_mode="corrected",
    )

    assert not rows.empty
    assert any("Castelli/Kurucz" in str(trace.name) for trace in fig.data)
    x0, x1 = fig.layout.xaxis.range
    assert x0 > math.log10(100.0)
    assert x1 < math.log10(1.0e6)
    assert x0 < math.log10(float(rows["lambda_eff_angstrom"].min()))
    assert x1 > math.log10(float(rows["lambda_eff_angstrom"].max()))
