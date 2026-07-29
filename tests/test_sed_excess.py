from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from malca.enrichment.sed_excess import (
    NULL_LOCUS_COLUMNS,
    SED_EXCESS_VERSION,
    compute_sed_excess_posteriors,
    draw_fit_parameters,
    evaluate_wise_quality,
    fit_empirical_null_locus,
    _model_flux_draws,
    _model_flux_draws_direct,
    refresh_allwise_quality,
    summarize_sed_excess,
    upsert_sed_excess_results,
)
from malca.enrichment.sed_model import LSUN_ERG_S, SED_MODEL_FIT_VERSION
from malca.enrichment.synthetic_photometry import bandpass_flux_nu_jy, top_hat_response
from malca.review.sed import bandpass_for


class FakeKurucz:
    def __init__(self) -> None:
        self._wavelength = np.geomspace(2500.0, 250000.0, 1800)

    @property
    def logT(self):
        return np.array([3.55, 3.65, 3.75, 3.9, 4.1])

    @property
    def logg(self):
        return np.array([0.0, 2.5, 4.5])

    @property
    def Z(self):
        return np.array([0.02])

    def generate_stellar_spectrum(self, logT, logg, logL, Z, raise_extrapolation=False):
        wave = self._wavelength
        teff = 10.0 ** float(logT)
        exponent = np.clip(1.438776877e8 / (wave * teff), 1.0e-4, 700.0)
        shape = 1.0 / (wave**5 * np.expm1(exponent))
        shape /= np.trapezoid(shape, wave)
        return shape * LSUN_ERG_S


def _response_loader(filter_id: str, mag_system: str):
    band = next(band for band in ("W1", "W2", "W3", "W4") if bandpass_for("AllWISE", band).svo_filter_id == filter_id)
    center = bandpass_for("AllWISE", band).lambda_eff_angstrom
    return top_hat_response(filter_id, center, center * 0.01, mag_system=mag_system)


def _fit(candidate_id: str = "candidate") -> pd.DataFrame:
    center = [math.log10(6000.0), 0.2, -40.0]
    covariance = np.array([
        [2.0e-6, 1.0e-5, -2.0e-6],
        [1.0e-5, 4.0e-3, -1.0e-4],
        [-2.0e-6, -1.0e-4, 1.0e-4],
    ])
    return pd.DataFrame([{
        "candidate_id": candidate_id,
        "fit_version": SED_MODEL_FIT_VERSION,
        "fit_run_hash": "run",
        "status": "ok",
        "teff_k": 6000.0,
        "teff_err_k": 120.0,
        "av_fit": 0.2,
        "apparent_scale": 1.0e-40,
        "logg": 4.5,
        "z": 0.02,
        "rv": 3.1,
        "reduced_chi2": 1.0,
        "boundary_flags": "",
        "fit_param_names_json": json.dumps(["log10_teff", "av", "log10_apparent_scale"]),
        "fit_param_values_json": json.dumps(center),
        "fit_covariance_json": json.dumps(covariance.tolist()),
        "fit_covariance_status": "ok",
    }])


def _wise_rows(candidate_id: str = "candidate", ratios: dict[str, float] | None = None) -> pd.DataFrame:
    ratios = ratios or {band: 1.0 for band in ("W1", "W2", "W3", "W4")}
    library = FakeKurucz()
    spectrum = library.generate_stellar_spectrum(math.log10(6000.0), 4.5, 0.0, 0.02)
    rows = []
    for band in ("W1", "W2", "W3", "W4"):
        bp = bandpass_for("AllWISE", band)
        response = _response_loader(bp.svo_filter_id, bp.mag_system)
        flux = 1.0e-40 * bandpass_flux_nu_jy(library._wavelength, spectrum, response) * ratios[band]
        rows.append({
            "candidate_id": candidate_id,
            "source": "AllWISE",
            "band": band,
            "mag": np.nan,
            "mag_err": 0.02,
            "mag_system": "Vega",
            "lambda_eff_angstrom": bp.lambda_eff_angstrom,
            "flux_nu_jy": flux,
            "flux_nu_jy_err": flux * 0.02,
            "sep_arcsec": 0.2,
            "is_synthetic": 0,
            "is_upper_limit": 0,
            "quality_flags": "qph=AAAA;ccf=0000;ex=0;nb=1;na=0;snr1=30;snr2=30;snr3=30;snr4=30;chi2w1=1;chi2w2=1;chi2w3=1;chi2w4=1;sat1=0;sat2=0;sat3=0;sat4=0",
            "svo_filter_id": bp.svo_filter_id,
            "av_coeff": bp.av_coeff,
        })
    return pd.DataFrame(rows)


def _zero_null_locus() -> pd.DataFrame:
    rows = []
    for band in ("W1", "W2", "W3", "W4"):
        rows.append({
            "excess_version": SED_EXCESS_VERSION,
            "null_locus_version": "test-null",
            "band": band,
            "n_control": 100,
            "status": "ok",
            "feature_names_json": "[]",
            "feature_centers_json": "{}",
            "feature_scales_json": "{}",
            "coefficients_json": "[0.0]",
            "scatter_dex": 0.01,
            "clip_fraction": 0.0,
        })
    return pd.DataFrame(rows, columns=NULL_LOCUS_COLUMNS)


def test_wise_quality_is_band_specific_and_rejects_isolated_artifact() -> None:
    row = {
        "quality_flags": "qph=AAAA;ccf=00H0;ex=0;nb=1;na=0;snr3=25;chi2w3=1.2;sat3=0",
        "sep_arcsec": 0.4,
        "is_upper_limit": 0,
    }
    result = evaluate_wise_quality(row, "W3")
    assert result["quality_status"] == "fail"
    assert "artifact_h" in result["quality_reasons"]
    assert evaluate_wise_quality(row, "W4")["quality_status"] == "pass"


@pytest.mark.parametrize("false_value", ["0", "false", "False", "no"])
def test_wise_quality_does_not_treat_false_text_as_an_upper_limit(false_value: str) -> None:
    row = {
        "quality_flags": "qph=AAAA;ccf=0000;ex=0;nb=1;na=0;snr3=25;chi2w3=1.2;sat3=0",
        "sep_arcsec": 0.4,
        "is_upper_limit": false_value,
    }

    result = evaluate_wise_quality(row, "W3")

    assert result["quality_status"] == "pass"
    assert "upper_limit" not in result["quality_reasons"]


def test_mc_excess_is_deterministic_distance_invariant_and_recovers_adjacent_excess() -> None:
    candidates = pd.DataFrame([{"candidate_id": "candidate", "distance_gspphot": 100.0}])
    kwargs = dict(
        candidates=candidates,
        fits=_fit(),
        sed_rows=_wise_rows(ratios={"W1": 1.0, "W2": 1.05, "W3": 2.0, "W4": 2.4}),
        null_locus=_zero_null_locus(),
        n_draws=96,
        seed=41,
        library=FakeKurucz(),
        response_loader=_response_loader,
        allow_bandpass_download=False,
        non_simultaneity_floor_mag=0.0,
    )
    bands_a, summary_a = compute_sed_excess_posteriors(**kwargs)
    kwargs["candidates"] = pd.DataFrame([{"candidate_id": "candidate", "distance_gspphot": 5000.0}])
    bands_b, summary_b = compute_sed_excess_posteriors(**kwargs)

    pd.testing.assert_frame_equal(bands_a, bands_b)
    assert summary_a.loc[0, "excess_class"] == "robust"
    assert summary_a.loc[0, "primary_band"] == "W3"
    assert summary_a.loc[0, "w3_ratio_p50"] == pytest.approx(2.0, rel=0.12)
    assert summary_b.loc[0, "w4_ratio_p50"] == pytest.approx(summary_a.loc[0, "w4_ratio_p50"])


def test_pure_photosphere_is_not_classified_as_excess() -> None:
    bands, summary = compute_sed_excess_posteriors(
        candidates=pd.DataFrame([{"candidate_id": "candidate"}]),
        fits=_fit(),
        sed_rows=_wise_rows(),
        null_locus=_zero_null_locus(),
        n_draws=256,
        seed=81,
        library=FakeKurucz(),
        response_loader=_response_loader,
        allow_bandpass_download=False,
        non_simultaneity_floor_mag=0.0,
    )

    assert not bands["p_excess_calibrated"].gt(0.997).any()
    assert summary.loc[0, "excess_class"] == "none"


def test_isolated_w4_is_not_robust() -> None:
    bands = pd.DataFrame([
        {"candidate_id": "c", "band": "W3", "ratio_p50": 1.0, "ratio_p16": 0.9, "ratio_p84": 1.1, "p_excess_calibrated": 0.5, "quality_pass": True, "quality_status": "pass", "posterior_reliable": True},
        {"candidate_id": "c", "band": "W4", "ratio_p50": 4.0, "ratio_p16": 3.0, "ratio_p84": 5.0, "p_excess_calibrated": 0.999, "quality_pass": True, "quality_status": "pass", "posterior_reliable": True},
    ])
    summary = summarize_sed_excess(bands)
    assert summary.loc[0, "excess_class"] == "isolated_w4"


def test_failed_model_posterior_is_unassessable_not_uncalibrated() -> None:
    bands = pd.DataFrame([{
        "candidate_id": "c", "band": "W3", "p_excess_calibrated": np.nan,
        "quality_pass": True, "quality_status": "pass", "quality_reasons": "",
        "posterior_reliable": False, "posterior_status": "model_prediction_failed",
        "null_locus_version": "control-softl1-v1",
    }])

    summary = summarize_sed_excess(bands)

    assert summary.loc[0, "excess_class"] == "unassessable"
    assert summary.loc[0, "classification_reason"] == "posterior_unreliable"


def test_bandpass_response_grid_matches_direct_draw_evaluation() -> None:
    rng = np.random.default_rng(14)
    draws = np.column_stack((
        rng.uniform(math.log10(5600.0), math.log10(6400.0), 48),
        rng.uniform(0.0, 0.8, 48),
        rng.normal(-40.0, 0.04, 48),
    ))
    responses = {}
    for band in ("W1", "W2", "W3", "W4"):
        bp = bandpass_for("AllWISE", band)
        responses[(bp.svo_filter_id, bp.mag_system)] = _response_loader(bp.svo_filter_id, bp.mag_system)
    fit = _fit().iloc[0].to_dict()
    direct = _model_flux_draws_direct(draws, fit, FakeKurucz(), responses)
    gridded = _model_flux_draws(draws, fit, FakeKurucz(), responses)

    for band in ("W1", "W2", "W3", "W4"):
        np.testing.assert_allclose(gridded[band], direct[band], rtol=2.0e-3, atol=0.0)
    single_direct = _model_flux_draws_direct(draws[:1], fit, FakeKurucz(), responses)
    single_gridded = _model_flux_draws(draws[:1], fit, FakeKurucz(), responses)
    for band in ("W1", "W2", "W3", "W4"):
        np.testing.assert_allclose(single_gridded[band], single_direct[band], rtol=1.0e-12, atol=0.0)


def test_correlated_covariance_changes_model_prediction_uncertainty() -> None:
    correlated_fit = _fit().iloc[0].to_dict()
    covariance = np.array([
        [4.0e-4, 0.0, 3.6e-4],
        [0.0, 1.0e-4, 0.0],
        [3.6e-4, 0.0, 4.0e-4],
    ])
    correlated_fit["fit_covariance_json"] = json.dumps(covariance.tolist())
    diagonal_fit = dict(correlated_fit)
    diagonal_fit["fit_covariance_json"] = json.dumps(np.diag(np.diag(covariance)).tolist())
    bounds = (math.log10(3500.0), math.log10(12000.0))
    correlated_draws, _ = draw_fit_parameters(correlated_fit, 2000, seed=22, logt_bounds=bounds)
    diagonal_draws, _ = draw_fit_parameters(diagonal_fit, 2000, seed=22, logt_bounds=bounds)
    responses = {}
    for band in ("W1", "W2", "W3", "W4"):
        bp = bandpass_for("AllWISE", band)
        responses[(bp.svo_filter_id, bp.mag_system)] = _response_loader(bp.svo_filter_id, bp.mag_system)
    correlated_model = _model_flux_draws(correlated_draws, correlated_fit, FakeKurucz(), responses)["W1"]
    diagonal_model = _model_flux_draws(diagonal_draws, diagonal_fit, FakeKurucz(), responses)["W1"]

    correlated_width = np.std(np.log10(correlated_model))
    diagonal_width = np.std(np.log10(diagonal_model))
    assert max(correlated_width, diagonal_width) > 1.15 * min(correlated_width, diagonal_width)
    assert np.corrcoef(correlated_draws[:, 0], correlated_draws[:, 2])[0, 1] > 0.8
    assert abs(np.corrcoef(diagonal_draws[:, 0], diagonal_draws[:, 2])[0, 1]) < 0.1


def test_empirical_null_locus_recovers_band_offset_and_scatter() -> None:
    rng = np.random.default_rng(9)
    controls = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(120)],
        "band": "W3",
        "log_ratio_p50": 0.12 + rng.normal(0.0, 0.025, 120),
        "teff_k": rng.uniform(4500.0, 7500.0, 120),
        "observed_mag": rng.uniform(7.0, 11.0, 120),
        "reduced_chi2": 1.0,
        "av_fit": 0.3,
        "quality_status": "pass",
    })
    locus = fit_empirical_null_locus(controls, minimum_controls=30)
    w3 = locus[locus["band"] == "W3"].iloc[0]
    assert w3["status"] == "ok"
    coefficients = json.loads(w3["coefficients_json"])
    assert coefficients[0] == pytest.approx(0.12, abs=0.015)
    assert float(w3["scatter_dex"]) == pytest.approx(0.025, abs=0.012)


def test_empirical_null_locus_handles_empty_control_table() -> None:
    locus = fit_empirical_null_locus(pd.DataFrame())

    assert locus["band"].tolist() == ["W1", "W2", "W3", "W4"]
    assert locus["status"].eq("insufficient_controls").all()
    assert locus["n_control"].eq(0).all()


def test_allwise_quality_refresh_retries_only_unmatched_chunks() -> None:
    candidates = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(7)],
        "ra": np.arange(7, dtype=float),
        "dec": np.arange(7, dtype=float),
    })
    calls: list[list[str]] = []

    def fake_crossmatch(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        ids = result["candidate_id"].astype(str).tolist()
        calls.append(ids)
        result["allwise_id"] = ""
        if len(result) <= 2:
            result["allwise_id"] = "WISE-" + result["candidate_id"].astype(str)
        return result

    refreshed = refresh_allwise_quality(
        candidates,
        batch_size=4,
        minimum_retry_size=1,
        max_retry_depth=3,
        crossmatcher=fake_crossmatch,
    )

    assert refreshed["allwise_id"].str.startswith("WISE-").all()
    assert calls[0] == ["c0", "c1", "c2", "c3"]
    assert calls[-1] == ["c5", "c6"]


def test_wise_quality_requires_core_metadata_and_uses_flux_snr_fallback() -> None:
    adequate = evaluate_wise_quality(
        {"observed_flux_nu_jy": 1.0, "observed_flux_nu_jy_err": 0.1, "sep_arcsec": 0.2},
        "W3",
    )
    low_snr = evaluate_wise_quality(
        {"observed_flux_nu_jy": 1.0, "observed_flux_nu_jy_err": 0.5, "sep_arcsec": 0.2},
        "W3",
    )

    assert adequate["quality_status"] == "unknown"
    assert adequate["quality_pass"] is False
    assert low_snr["quality_status"] == "fail"
    assert "low_snr" in low_snr["quality_reasons"]


def test_candidate_without_wise_photometry_is_explicitly_unassessable() -> None:
    bands, summary = compute_sed_excess_posteriors(
        pd.DataFrame([{"candidate_id": "missing", "gal_b": 20.0}]),
        _fit("missing"),
        pd.DataFrame(columns=["candidate_id", "source", "band"]),
        null_locus=_zero_null_locus(),
        n_draws=20,
        library=FakeKurucz(),
        response_loader=_response_loader,
        allow_bandpass_download=False,
    )

    assert bands["band"].tolist() == ["W1", "W2", "W3", "W4"]
    assert bands["quality_status"].eq("missing").all()
    assert summary.loc[0, "candidate_id"] == "missing"
    assert summary.loc[0, "excess_class"] == "unassessable"
    assert summary.loc[0, "classification_reason"] == "no_wise_photometry"


def test_excess_results_upsert_to_versioned_review_tables() -> None:
    bands = pd.DataFrame([{
        "candidate_id": "c", "excess_version": SED_EXCESS_VERSION,
        "fit_version": SED_MODEL_FIT_VERSION, "fit_run_hash": "run",
        "source": "AllWISE", "band": "W3", "ratio_p50": 2.1,
    }])
    summaries = pd.DataFrame([{
        "candidate_id": "c", "excess_version": SED_EXCESS_VERSION,
        "excess_class": "robust", "primary_band": "W3",
    }])
    conn = sqlite3.connect(":memory:")

    assert upsert_sed_excess_results(conn, bands, summaries) == (1, 1)
    band_payload = json.loads(conn.execute(
        "SELECT payload_json FROM sed_excess_bands WHERE candidate_id = 'c'"
    ).fetchone()[0])
    summary_payload = json.loads(conn.execute(
        "SELECT payload_json FROM sed_excess_summary WHERE candidate_id = 'c'"
    ).fetchone()[0])

    assert band_payload["ratio_p50"] == 2.1
    assert summary_payload["excess_class"] == "robust"
