from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from malca.enrichment.sed_image_photometry import (
    NoUsableCoverageError,
    measure_fits_aperture,
)


def _write_test_image(path: Path, data: np.ndarray, *, bunit: str = "MJy/sr") -> None:
    from astropy.io import fits
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [51.0, 51.0]
    wcs.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    header["BUNIT"] = bunit
    fits.PrimaryHDU(data=np.asarray(data, dtype=float), header=header).writeto(path)


def test_measure_fits_aperture_returns_provisional_calibrated_flux(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    y, x = np.indices((101, 101))
    image = 2.0 + rng.normal(0.0, 0.02, size=(101, 101))
    image += 20.0 * np.exp(-0.5 * ((x - 50.0) ** 2 + (y - 50.0) ** 2) / 1.2**2)
    science = tmp_path / "science.fits"
    coverage = tmp_path / "coverage.fits"
    _write_test_image(science, image)
    _write_test_image(coverage, np.ones_like(image), bunit="count")

    result = measure_fits_aperture(
        science,
        ra_deg=10.0,
        dec_deg=20.0,
        band="IRAC1",
        coverage_path=coverage,
    )

    assert result.flux_jy > 0
    assert result.flux_error_jy > 0
    assert result.is_upper_limit is False
    assert result.native_unit == "MJy/sr"
    assert result.n_aperture_pixels > 3
    assert result.n_annulus_pixels > 12


def test_measure_fits_aperture_requires_positive_coverage(tmp_path: Path) -> None:
    science = tmp_path / "science.fits"
    coverage = tmp_path / "coverage.fits"
    _write_test_image(science, np.ones((101, 101)))
    _write_test_image(coverage, np.zeros((101, 101)), bunit="count")

    with pytest.raises(NoUsableCoverageError, match="coverage"):
        measure_fits_aperture(
            science,
            ra_deg=10.0,
            dec_deg=20.0,
            band="IRAC1",
            coverage_path=coverage,
        )
