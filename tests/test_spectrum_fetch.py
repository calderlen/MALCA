from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.enrich.spectrum_fetch import (
    FetchBackend,
    FetchStatus,
    SpectrumData,
    SpectrumFetchResult,
    SURVEY_BACKEND_MAP,
    _cache_path,
    fetch_spectrum,
    load_spectrum_cache,
    prefetch_spectra,
    save_spectrum_cache,
)
from malca.enrich.spectrum_config import SpectrumFetchConfig, load_spectrum_fetch_config


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
