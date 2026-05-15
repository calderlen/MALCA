from __future__ import annotations

import math
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from malca.review.sed import (
    SED_COLUMNS,
    bandpass_for,
    build_sed_dataframe,
    build_sed_figure,
    distance_pc_from_payload,
    extinction_av_from_payload,
    flux_lambda_from_flux_nu_jy,
    flux_nu_jy_from_mag,
    load_sed_rows,
    resolve_sed_sources,
    rows_from_payload,
    upsert_sed_rows,
)
from malca.review.pipeline import detect_sed_photometry_status
from malca.review.store import db_connect


def test_ab_and_vega_flux_conversions() -> None:
    ps1_g = bandpass_for("Pan-STARRS", "g")
    wise_w1 = bandpass_for("AllWISE", "W1")

    assert ps1_g is not None
    assert wise_w1 is not None
    assert math.isclose(flux_nu_jy_from_mag(0.0, ps1_g), 3631.0, rel_tol=1e-12)
    assert math.isclose(flux_nu_jy_from_mag(0.0, wise_w1), 309.540, rel_tol=1e-12)
    assert flux_lambda_from_flux_nu_jy(3631.0, ps1_g.lambda_eff_angstrom) > 0


def test_default_sed_sources_exclude_far_ir_catalogs() -> None:
    default_sources = set(resolve_sed_sources("default"))

    assert "payload" in default_sources
    assert "ps1" in default_sources
    assert "akari" not in default_sources
    assert set(resolve_sed_sources("far-ir")) == {"akari", "iras", "herschel"}
    assert "herschel" in set(resolve_sed_sources("all"))


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
    assert "F_lambda" in fig.layout.yaxis.title.text


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
