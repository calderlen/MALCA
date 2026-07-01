from __future__ import annotations

import numpy as np
import pytest

from malca.enrich.spectrum_fetch import SpectrumData
from malca.review.spectrum_plot import (
    APOGEE_STRONG_STELLAR_LINE_WINDOWS_AA,
    DEFAULT_CONTINUUM_DEGREE,
    DEFAULT_CONTINUUM_MODE,
    _fit_smooth_continuum,
    _repair_segment_continuum,
    analyze_spectrum,
    build_spectrum_figure,
    line_fits_to_dataframe,
    render_spectrum_pdf,
)


def test_build_spectrum_figure_draws_uncertainty_as_underlay_envelope() -> None:
    wavelength = np.array([5000.0, 5001.0, 5002.0, 5003.0])
    flux = np.array([1.0, 1.2, 0.9, 1.1])
    flux_err = np.array([0.1, -0.2, np.nan, 0.05])

    fig = build_spectrum_figure(wavelength, flux, flux_err=flux_err, survey="synthetic", theme="white")

    assert len(fig.data) == 2
    envelope, spectrum = fig.data
    assert envelope.fill == "toself"
    assert envelope.showlegend is False
    assert envelope.hoverinfo == "skip"
    assert envelope.legendgroup == spectrum.legendgroup == "spectrum"
    assert spectrum.name == "synthetic"
    np.testing.assert_allclose(
        np.asarray(envelope.y, dtype=float),
        np.array([1.1, 1.4, np.nan, 1.15, 1.05, np.nan, 1.0, 0.9]),
        equal_nan=True,
    )


def test_analyze_spectrum_fits_specutils_significant_lines() -> None:
    pytest.importorskip("specutils")

    wavelength = np.linspace(6500.0, 6600.0, 2000)
    continuum = 10.0 + 0.01 * (wavelength - 6550.0)
    emission = 1.8 * np.exp(-0.5 * ((wavelength - 6540.0) / 0.35) ** 2)
    absorption = -1.5 * np.exp(-0.5 * ((wavelength - 6570.0) / 0.45) ** 2)
    flux = continuum + emission + absorption
    flux_err = np.full_like(wavelength, 0.05)

    analysis = analyze_spectrum(
        SpectrumData(wavelength=wavelength, flux=flux, flux_err=flux_err),
        continuum_degree=1,
        line_noise_factor=8.0,
        line_window_AA=1.5,
    )

    assert len(analysis.detected_lines) == 2
    assert len(analysis.line_fits) == 2
    assert {line.line_type for line in analysis.line_fits} == {"emission", "absorption"}
    np.testing.assert_allclose([line.mean for line in analysis.line_fits], [6540.0, 6570.0], atol=0.08)
    assert np.nanmedian(np.abs(analysis.normalized_residual_flux)) < 0.003
    assert analysis.normalized_flux_err is not None
    np.testing.assert_allclose(np.nanmedian(analysis.normalized_flux_err), 0.005, rtol=0.15)
    np.testing.assert_allclose([line.amplitude for line in analysis.line_fits], [0.18, -0.15], atol=0.03)
    table = line_fits_to_dataframe(analysis)
    assert list(table["line_type"]) == ["emission", "absorption"]
    np.testing.assert_allclose(table["fit_mean"], [6540.0, 6570.0], atol=0.08)
    assert {"fit_fwhm", "signed_equivalent_width"}.issubset(table.columns)
    assert np.all(np.isfinite(table["signed_equivalent_width"]))
    assert table.loc[0, "signed_equivalent_width"] < 0
    assert table.loc[1, "signed_equivalent_width"] > 0


def test_analyze_spectrum_default_continuum_is_chipwise_pseudo_continuum() -> None:
    pytest.importorskip("specutils")

    wavelength = np.concatenate([np.linspace(6500.0, 6540.0, 250), np.linspace(6560.0, 6600.0, 250)])
    flux = 10.0 + 0.02 * (wavelength - 6550.0) + 0.0008 * (wavelength - 6550.0) ** 2
    flux_err = np.full_like(wavelength, 0.05)

    analysis = analyze_spectrum(SpectrumData(wavelength=wavelength, flux=flux, flux_err=flux_err))

    assert analysis.continuum_model.mode == DEFAULT_CONTINUUM_MODE
    assert getattr(analysis.continuum_model, "degree", None) == DEFAULT_CONTINUUM_DEGREE
    assert len(analysis.continuum_model.segments) == 2
    assert analysis.continuum_fit_mask is not None


def test_pseudo_continuum_handles_apogee_chip_curvature_and_absorption_forest() -> None:
    pytest.importorskip("specutils")

    rng = np.random.default_rng(42)
    chip_a = np.linspace(15150.0, 15805.0, 900)
    chip_b = np.linspace(15870.0, 16240.0, 520)
    chip_c = np.linspace(16480.0, 16960.0, 690)
    wavelength = np.concatenate([chip_a, chip_b, chip_c])
    centered = wavelength - 16000.0
    true_continuum = 7600.0 - 0.42 * centered + 180.0 * np.sin((wavelength - 15150.0) / 390.0)
    true_continuum += np.where(wavelength > 16400.0, -0.0016 * (wavelength - 16720.0) ** 2, 0.0)
    flux = true_continuum.copy()
    for center in np.linspace(15195.0, 16910.0, 46):
        local_depth = rng.uniform(0.025, 0.11)
        width = rng.uniform(0.18, 0.55)
        flux -= local_depth * true_continuum * np.exp(-0.5 * ((wavelength - center) / width) ** 2)
    flux += rng.normal(0.0, 0.004 * true_continuum, size=len(wavelength))
    flux_err = np.full_like(wavelength, 0.004 * np.nanmedian(true_continuum))

    analysis = analyze_spectrum(
        SpectrumData(wavelength=wavelength, flux=flux, flux_err=flux_err),
        line_noise_factor=1000.0,
    )

    fit_mask = analysis.continuum_fit_mask
    assert fit_mask is not None
    assert analysis.valid_pixel_mask is not None
    assert analysis.continuum_output_mask is not None
    assert analysis.continuum_model.mode == "pseudo"
    assert len(analysis.continuum_model.segments) == 3
    assert np.count_nonzero(fit_mask) < np.count_nonzero(analysis.finite)

    edge_region = np.zeros_like(wavelength, dtype=bool)
    for chip in (chip_a, chip_b, chip_c):
        edge_region |= ((wavelength >= chip.min()) & (wavelength < chip.min() + 12.0))
        edge_region |= ((wavelength > chip.max() - 12.0) & (wavelength <= chip.max()))
    assert not np.any(fit_mask & edge_region)

    output_edge_region = np.zeros_like(wavelength, dtype=bool)
    for chip in (chip_a, chip_b, chip_c):
        output_edge_region |= ((wavelength >= chip.min()) & (wavelength < chip.min() + 30.0))
        output_edge_region |= ((wavelength > chip.max() - 30.0) & (wavelength <= chip.max()))
    assert not np.any(analysis.continuum_output_mask & output_edge_region)

    strong_line_region = np.zeros_like(wavelength, dtype=bool)
    for lo, hi in APOGEE_STRONG_STELLAR_LINE_WINDOWS_AA:
        strong_line_region |= (wavelength >= lo) & (wavelength <= hi)
    assert np.any(strong_line_region & analysis.continuum_output_mask)
    assert not np.any(fit_mask & strong_line_region)

    rel_error = analysis.continuum / true_continuum - 1.0
    np.testing.assert_allclose(np.nanmedian(rel_error), 0.0, atol=0.025)
    assert np.nanpercentile(np.abs(rel_error), 90.0) < 0.07


def test_spectrum_source_mask_does_not_delete_samples_from_static_analysis() -> None:
    pytest.importorskip("specutils")
    from astropy import units as u
    from astropy.nddata import StdDevUncertainty
    from specutils import Spectrum

    wavelength = np.linspace(6500.0, 6700.0, 700)
    flux = 12.0 + 0.015 * (wavelength - 6600.0)
    flux_err = np.full_like(wavelength, 0.05)
    source_mask = (wavelength > 6575.0) & (wavelength < 6625.0)
    spectrum = Spectrum(
        spectral_axis=wavelength * u.AA,
        flux=flux * u.dimensionless_unscaled,
        uncertainty=StdDevUncertainty(flux_err * u.dimensionless_unscaled),
        mask=source_mask,
    )

    analysis = analyze_spectrum(spectrum, line_noise_factor=1000.0)

    central = source_mask & np.isfinite(analysis.continuum)
    assert np.count_nonzero(central) > 0
    assert analysis.valid_pixel_mask is not None
    assert np.count_nonzero(analysis.valid_pixel_mask & source_mask) > 0
    assert np.nanmedian(np.abs(analysis.normalized_residual_flux[central])) < 0.05


def test_smooth_continuum_uses_envelope_instead_of_extrapolated_fit_tails() -> None:
    x_eval = np.linspace(16065.0, 16126.0, 280)
    x_fit = np.linspace(16081.0, 16103.0, 64)
    y_fit = 560.0 + 5.0 * np.sin((x_fit - 16092.0) / 5.0)
    flux = 555.0 + 8.0 * np.sin((x_eval - 16090.0) / 8.0)
    envelope = np.full_like(x_eval, 560.0)

    continuum, _model = _fit_smooth_continuum(
        x_fit,
        y_fit,
        x_eval,
        weights=None,
        knot_spacing_angstrom=60.0,
    )
    repaired = _repair_segment_continuum(x_eval, flux, continuum, envelope)

    outside_support = (x_eval < x_fit.min()) | (x_eval > x_fit.max())
    assert np.all(np.isnan(continuum[outside_support]))
    np.testing.assert_allclose(repaired[outside_support], envelope[outside_support])
    assert np.nanmax(repaired) < 590.0
    assert np.nanmin(repaired) > 530.0


def test_analyze_spectrum_can_use_global_continuum() -> None:
    pytest.importorskip("specutils")

    wavelength = np.concatenate([np.linspace(6500.0, 6540.0, 250), np.linspace(6560.0, 6600.0, 250)])
    flux = 10.0 + 0.02 * (wavelength - 6550.0)
    flux_err = np.full_like(wavelength, 0.05)

    analysis = analyze_spectrum(
        SpectrumData(wavelength=wavelength, flux=flux, flux_err=flux_err),
        continuum_mode="global",
    )

    assert analysis.continuum_model.mode == "global"
    assert len(analysis.continuum_model.segments) == 1


def test_render_spectrum_pdf_uses_matplotlib_line_analysis() -> None:
    pytest.importorskip("specutils")

    wavelength = np.linspace(6500.0, 6510.0, 400)
    continuum = np.full_like(wavelength, 1.0)
    emission = 0.4 * np.exp(-0.5 * ((wavelength - 6505.0) / 0.2) ** 2)
    flux = continuum + emission
    flux_err = np.full_like(wavelength, 0.04)

    pdf = render_spectrum_pdf(
        SpectrumData(wavelength=wavelength, flux=flux, flux_err=flux_err),
        survey="synthetic",
        candidate_id="line-test",
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_render_spectrum_pdf_accepts_analysis_configuration() -> None:
    pytest.importorskip("specutils")

    wavelength = np.linspace(6500.0, 6510.0, 400)
    flux = 1.0 + 0.3 * np.exp(-0.5 * ((wavelength - 6505.0) / 0.2) ** 2)
    flux_err = np.full_like(wavelength, 0.04)

    pdf = render_spectrum_pdf(
        SpectrumData(wavelength=wavelength, flux=flux, flux_err=flux_err),
        continuum_degree=3,
        continuum_mode="global",
        line_noise_factor=6.0,
        line_window_AA=1.0,
        max_line_fits=5,
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
