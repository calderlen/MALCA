from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.enrich.spectrum_fetch import (
    FetchBackend,
    FetchStatus,
    SpectrumData,
    SURVEY_BACKEND_MAP,
    _cache_path,
    _parse_fits_spectrum,
    fetch_spectrum,
    load_spectrum_cache,
    prefetch_spectra,
    save_spectrum_cache,
)
from malca.enrich.spectrum_config import load_spectrum_fetch_config


def test_survey_backend_map_covers_all_expected_keys() -> None:
    expected_fetchable = {
        "sdss_dr16_spec", "sdss_boss", "sdss_eboss", "sdss_legacy",
        "sdss_segue", "sdss_spiders", "sdss_tdss", "sdss2_sn",
        "desi_dr1", "galah_dr3", "galah_dr4", "lamost_dr7",
        "apogee_dr16", "apogee_dr17", "rave_dr6",
        "tns_spectra", "gaia_rvs", "gaia_xp", "gaia_eso",
        "osc", "pessto", "vvds", "zcosmos", "vandels", "vipers",
        "3d_hst", "sdss_v",
    }
    for key in expected_fetchable:
        assert key in SURVEY_BACKEND_MAP, f"Missing backend mapping for {key}"


def test_survey_backend_map_link_only_keys() -> None:
    link_only_keys = {
        "sixdf_gs", "2dfgrs", "ozdes", "deep2", "wigglez",
        "simbad", "ned", "milliquas", "tns", "manga_dr17",
    }
    for key in link_only_keys:
        assert key in SURVEY_BACKEND_MAP
        backend, _ = SURVEY_BACKEND_MAP[key]
        assert backend == FetchBackend.LINK_ONLY, f"{key} should be LINK_ONLY"


def test_sdss_maps_to_sdss_backend() -> None:
    for key in ["sdss_dr16_spec", "sdss_boss", "sdss_eboss"]:
        backend, _ = SURVEY_BACKEND_MAP[key]
        assert backend == FetchBackend.SDSS


def test_eso_collections_mapped_correctly() -> None:
    eso_keys = {"gaia_eso": "Gaia-ESO", "pessto": "PESSTO", "vvds": "VVDS", "vandels": "VANDELS", "vipers": "VIPERS"}
    for key, expected_collection in eso_keys.items():
        backend, kwargs = SURVEY_BACKEND_MAP[key]
        assert backend == FetchBackend.ESO
        assert kwargs.get("collection") == expected_collection


def test_cache_roundtrip(tmp_path: Path) -> None:
    data = SpectrumData(
        wavelength=np.linspace(4000, 9000, 100),
        flux=np.random.randn(100),
        flux_err=np.abs(np.random.randn(100)),
    )
    path = tmp_path / "test.npz"
    save_spectrum_cache(path, data)
    loaded = load_spectrum_cache(path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded.wavelength, data.wavelength)
    np.testing.assert_array_equal(loaded.flux, data.flux)
    np.testing.assert_array_equal(loaded.flux_err, data.flux_err)


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    assert load_spectrum_cache(tmp_path / "nonexistent.npz") is None


def test_fetch_uses_cache(tmp_path: Path) -> None:
    data = SpectrumData(
        wavelength=np.linspace(3000, 8000, 50),
        flux=np.ones(50),
    )
    cache_dir = tmp_path / "cache"
    save_spectrum_cache(_cache_path(cache_dir, "C1", "sdss_boss"), data)

    row = pd.Series({"candidate_id": "C1", "survey": "sdss_boss"})
    result = fetch_spectrum(row, cache_dir=cache_dir)

    assert result.status == FetchStatus.OK
    assert result.data is not None
    np.testing.assert_array_equal(result.data.wavelength, data.wavelength)


def test_fetch_apogee_cache_preserves_catalog_metadata(tmp_path: Path) -> None:
    data = SpectrumData(
        wavelength=np.linspace(15100, 16900, 50),
        flux=np.ones(50),
    )
    cache_dir = tmp_path / "cache"
    save_spectrum_cache(_cache_path(cache_dir, "C1", "apogee_dr17"), data)

    row = pd.Series(
        {
            "candidate_id": "C1",
            "survey": "apogee_dr17",
            "APOGEE_ID": "2M02541269+6041444",
            "TEFF": 4100.0,
            "LOGG": 3.9,
            "FE_H": -0.2,
            "VHELIO_AVG": 22.4,
            "SNR": 120.0,
            "NVISITS": 3,
            "STARFLAG": 0,
            "ASPCAPFLAG": 0,
            "VSINI": 14.0,
            "MG_FE": 0.1,
        }
    )

    result = fetch_spectrum(row, cache_dir=cache_dir)

    assert result.status == FetchStatus.OK
    assert result.metadata["APOGEE_ID"] == "2M02541269+6041444"
    assert result.metadata["TEFF"] == 4100.0
    assert result.metadata["LOGG"] == 3.9
    assert result.metadata["FE_H"] == -0.2
    assert result.metadata["VHELIO_AVG"] == 22.4
    assert result.metadata["MG_FE"] == 0.1


def test_fetch_link_only_survey() -> None:
    row = pd.Series({"candidate_id": "C1", "survey": "simbad", "link": "https://simbad.cds.unistra.fr/"})
    result = fetch_spectrum(row)
    assert result.status == FetchStatus.LINK_ONLY
    assert result.link == "https://simbad.cds.unistra.fr/"


def test_fetch_unknown_survey() -> None:
    row = pd.Series({"candidate_id": "C1", "survey": "unknown_survey_xyz"})
    result = fetch_spectrum(row)
    assert result.status == FetchStatus.LINK_ONLY


def test_prefetch_skips_link_only(tmp_path: Path) -> None:
    spectra_long = pd.DataFrame([
        {"candidate_id": "C1", "survey": "simbad", "link": "https://simbad.cds.unistra.fr/"},
        {"candidate_id": "C2", "survey": "ned", "link": "https://ned.ipac.caltech.edu/"},
    ])
    cache_dir = tmp_path / "spectra_cache"
    index = prefetch_spectra(spectra_long, cache_dir=cache_dir)
    assert index.empty or len(index) == 0


def test_prefetch_uses_existing_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "spectra_cache"
    data = SpectrumData(wavelength=np.linspace(4000, 7000, 30), flux=np.ones(30))
    save_spectrum_cache(_cache_path(cache_dir, "C1", "sdss_boss"), data)

    spectra_long = pd.DataFrame([
        {"candidate_id": "C1", "survey": "sdss_boss"},
    ])
    index = prefetch_spectra(spectra_long, cache_dir=cache_dir)
    assert len(index) == 1
    assert index.iloc[0]["status"] == "cached"


def test_load_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TNS_API_KEY", "test-key-123")
    monkeypatch.setenv("ESO_USERNAME", "testuser")
    config = load_spectrum_fetch_config()
    assert config.tns_api_key == "test-key-123"
    assert config.eso_username == "testuser"


def test_load_config_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("TNS_API_KEY", "env-key")
    config = load_spectrum_fetch_config(tns_api_key="explicit-key")
    assert config.tns_api_key == "explicit-key"


def test_spectrum_data_without_error() -> None:
    data = SpectrumData(wavelength=np.array([1, 2, 3]), flux=np.array([4, 5, 6]))
    assert data.flux_err is None
    assert len(data.wavelength) == 3


def test_fetch_sdss_missing_ids() -> None:
    from malca.enrich.spectrum_fetch import _fetch_sdss
    row = pd.Series({"candidate_id": "C1", "survey": "sdss_boss"})
    result = _fetch_sdss(row)
    assert result.status == FetchStatus.LINK_ONLY


def test_fetch_gaia_missing_source_id() -> None:
    from malca.enrich.spectrum_fetch import _fetch_gaia
    row = pd.Series({"candidate_id": "C1", "survey": "gaia_rvs"})
    result = _fetch_gaia(row)
    assert result.status == FetchStatus.LINK_ONLY


def test_fetch_eso_missing_object_name() -> None:
    from malca.enrich.spectrum_fetch import _fetch_eso
    row = pd.Series({"candidate_id": "C1", "survey": "pessto"})
    result = _fetch_eso(row, collection="PESSTO")
    assert result.status == FetchStatus.LINK_ONLY


def test_fetch_tns_missing_api_key() -> None:
    from malca.enrich.spectrum_fetch import _fetch_tns
    row = pd.Series({"candidate_id": "C1", "survey": "tns_spectra", "provenance_name": "AT2020abc"})
    result = _fetch_tns(row)
    assert result.status == FetchStatus.LINK_ONLY


def test_fetch_lamost_missing_obsid() -> None:
    from malca.enrich.spectrum_fetch import _fetch_lamost
    row = pd.Series({"candidate_id": "C1", "survey": "lamost_dr7"})
    result = _fetch_lamost(row)
    assert result.status == FetchStatus.LINK_ONLY


def test_fetch_galah_accepts_vizier_galah_identifier(monkeypatch) -> None:
    import malca.enrich.spectrum_fetch as spectrum_fetch

    lookup = pd.DataFrame(
        [{"sobject_id": "161008000000000", "file_path": "", "url": np.nan}]
    ).set_index("sobject_id")
    monkeypatch.setattr(spectrum_fetch, "_GALAH_LOOKUP", lookup)

    row = pd.Series({"candidate_id": "C1", "survey": "galah_dr3", "GALAH": 161008000000000.0})
    result = spectrum_fetch._fetch_galah(row)

    assert result.status == FetchStatus.LINK_ONLY
    assert result.message == "GALAH lookup row missing file_path and url"


def test_apogee_apstar_parser_reads_error_hdu_not_flux_row(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")

    flux_rows = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [100.0, 101.0, 102.0, 103.0],
        ],
        dtype=np.float32,
    )
    error_rows = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.1, 1.2, 1.3, 1.4],
        ],
        dtype=np.float32,
    )

    primary = fits.PrimaryHDU()
    primary.header["NVISITS"] = 2
    primary.header["SFILE1"] = "apVisit-r3-0000-00000-000.fits"

    flux_hdu = fits.ImageHDU(flux_rows)
    flux_hdu.header["CRVAL1"] = 4.0
    flux_hdu.header["CDELT1"] = 0.001
    flux_hdu.header["CRPIX1"] = 1
    flux_hdu.header["CTYPE1"] = "LOG-LINEAR"
    flux_hdu.header["DC-FLAG"] = 1
    flux_hdu.header["BUNIT"] = "Flux (10^-17 erg/s/cm^2/Ang)"

    error_hdu = fits.ImageHDU(error_rows)
    error_hdu.header["BUNIT"] = "Err (10^-17 erg/s/cm^2/Ang)"

    path = tmp_path / "apStar-test.fits"
    fits.HDUList([primary, flux_hdu, error_hdu]).writeto(path)

    result = _parse_fits_spectrum(str(path))

    assert result.status == FetchStatus.OK
    assert result.data is not None
    np.testing.assert_allclose(result.data.flux, flux_rows[0])
    np.testing.assert_allclose(result.data.flux_err, error_rows[0])
    assert not np.allclose(result.data.flux_err, flux_rows[1])
    np.testing.assert_allclose(result.data.wavelength, 10.0 ** np.array([4.0, 4.001, 4.002, 4.003]))


def test_lamost_style_vector_columns_are_squeezed_to_1d(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")

    path = tmp_path / "lamost-vector-columns.fits"
    table_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="wavelength", format="4D", array=np.array([[5000.0, 5001.0, 5002.0, 5003.0]])),
            fits.Column(name="flux", format="4D", array=np.array([[1.0, 1.1, 0.9, 1.2]])),
            fits.Column(name="ivar", format="4D", array=np.array([[100.0, 25.0, 4.0, 1.0]])),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path)

    result = _parse_fits_spectrum(str(path))

    assert result.status == FetchStatus.OK
    assert result.data is not None
    assert result.data.wavelength.shape == (4,)
    assert result.data.flux.shape == (4,)
    assert result.data.flux_err is not None
    assert result.data.flux_err.shape == (4,)
    np.testing.assert_allclose(result.data.flux_err, [0.1, 0.2, 0.5, 1.0])


def test_spectrum_data_round_trips_through_specutils() -> None:
    pytest.importorskip("specutils")

    data = SpectrumData(
        wavelength=np.array([5000.0, 5001.0, 5002.0]),
        flux=np.array([1.0, 1.1, 0.9]),
        flux_err=np.array([0.1, 0.2, 0.3]),
    )

    spectrum = data.to_specutils()
    restored = SpectrumData.from_specutils(spectrum)

    np.testing.assert_allclose(restored.wavelength, data.wavelength)
    np.testing.assert_allclose(restored.flux, data.flux)
    np.testing.assert_allclose(restored.flux_err, data.flux_err)
