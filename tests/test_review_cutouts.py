from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from malca.review.cutouts import (
    CUTOUT_SURVEY_BY_KEY,
    DEFAULT_CUTOUT_FOV_ARCSEC,
    DEFAULT_CUTOUT_SIZE_PX,
    build_cutout_url,
    build_hips2fits_url,
    candidate_coordinates,
    cutout_payload_for_candidate,
)


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def test_hips2fits_url_contains_expected_cutout_parameters() -> None:
    survey = CUTOUT_SURVEY_BY_KEY["panstarrs-dr1-color"]
    url = build_hips2fits_url(
        survey,
        240.48595227,
        -55.342,
        fov_arcsec=DEFAULT_CUTOUT_FOV_ARCSEC,
        size_px=DEFAULT_CUTOUT_SIZE_PX,
    )
    params = _query(url)

    assert urlparse(url).netloc == "alasky.cds.unistra.fr"
    assert "CDS%2FP%2FPanSTARRS%2FDR1%2Fcolor-i-r-g" in url
    assert params["hips"] == ["CDS/P/PanSTARRS/DR1/color-i-r-g"]
    assert params["ra"] == ["240.48595227"]
    assert params["dec"] == ["-55.34200000"]
    assert float(params["fov"][0]) == pytest.approx(60.0 / 3600.0)
    assert params["width"] == ["512"]
    assert params["height"] == ["512"]
    assert params["projection"] == ["TAN"]
    assert params["format"] == ["jpg"]


def test_legacy_url_contains_dr10_layer_bands_and_pixel_size() -> None:
    url = build_cutout_url(
        "desi-legacy-dr10",
        10.0,
        20.0,
        fov_arcsec=60.0,
        size_px=420,
    )
    params = _query(url)

    assert urlparse(url).netloc == "www.legacysurvey.org"
    assert params["layer"] == ["ls-dr10"]
    assert params["bands"] == ["grz"]
    assert params["size"] == ["420"]
    assert float(params["pixscale"][0]) == pytest.approx(60.0 / 420.0)


def test_candidate_coordinates_use_existing_aliases() -> None:
    coords = candidate_coordinates({"RAJ2000": "12.5", "DEJ2000": "-4.25"})

    assert coords == (12.5, -4.25)


def test_missing_or_invalid_coordinates_return_empty_cutout_payload() -> None:
    missing = cutout_payload_for_candidate({"candidate_id": "C1"})
    invalid = cutout_payload_for_candidate({"ra": 361.0, "dec": 0.0})

    assert missing["has_coordinates"] is False
    assert missing["image_url"] == ""
    assert "RA/Dec" in str(missing["message"])
    assert invalid["has_coordinates"] is False
    assert invalid["image_url"] == ""
