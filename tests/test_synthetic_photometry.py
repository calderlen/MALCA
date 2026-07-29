from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

import malca.enrichment.synthetic_photometry as synthetic_photometry

from malca.enrichment.photometric_calibration import (
    CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT,
    REFERENCE_NU_FNU_CONSTANT,
    ab_calibration,
    mission_quoted_fnu_calibration,
    quoted_fnu_calibration,
    reference_fnu_jy,
    response_matched_vega_zero_point_calibration,
    vega_zero_point_calibration,
)
from malca.enrichment.synthetic_photometry import (
    CACHE_FORMAT_VERSION,
    C_ANGSTROM_PER_S,
    JY_CGS,
    FilterResponse,
    _legacy_cache_stem,
    _metadata_hash,
    apply_extinction,
    bandpass_flux_nu_jy,
    build_response_map,
    fetch_filter_response,
    load_cached_filter_response,
    predict_native_observable,
    response_audit_manifest_hash,
    response_manifest_hash,
    response_matched_zero_point_jy,
    response_pivot_wavelength_angstrom,
    save_filter_response,
    svo_calibration_reference_wavelength_angstrom,
    top_hat_response,
)


_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def test_flat_one_jy_spectrum_integrates_to_one_jy() -> None:
    wave = np.geomspace(3000.0, 25000.0, 4000)
    flux_lambda = 1.0e-23 * 2.99792458e18 / np.square(wave)
    response = top_hat_response("TEST/flat", 7000.0, 3000.0)

    assert bandpass_flux_nu_jy(wave, flux_lambda, response) == pytest.approx(1.0, rel=2.0e-5)


def test_g23_extinction_is_applied_before_band_integration() -> None:
    wave = np.geomspace(2500.0, 25000.0, 5000)
    intrinsic = np.ones_like(wave)
    extincted = apply_extinction(wave, intrinsic, 1.0, rv=3.1)
    blue = top_hat_response("TEST/blue", 4000.0, 300.0)
    infrared = top_hat_response("TEST/ir", 20000.0, 1000.0)

    blue_ratio = bandpass_flux_nu_jy(wave, extincted, blue) / bandpass_flux_nu_jy(wave, intrinsic, blue)
    infrared_ratio = bandpass_flux_nu_jy(wave, extincted, infrared) / bandpass_flux_nu_jy(wave, intrinsic, infrared)

    assert 0 < blue_ratio < infrared_ratio < 1


def test_filter_response_cache_roundtrip(tmp_path) -> None:
    response = top_hat_response("TEST/cache", 5500.0, 800.0, mag_system="AB")
    data_path, meta_path = save_filter_response(response, tmp_path)

    assert data_path.exists()
    assert meta_path.exists()
    assert json.loads(meta_path.read_text())["response_hash"] == response.response_hash
    loaded = load_cached_filter_response("TEST/cache", "AB", tmp_path)
    assert loaded is not None
    assert loaded.response_hash == response.response_hash
    assert np.array_equal(loaded.wavelength_angstrom, response.wavelength_angstrom)


def test_fetch_filter_response_parses_svo_votable_and_caches(tmp_path) -> None:
    payload = b"""<?xml version='1.0'?>
    <VOTABLE version='1.3' xmlns='http://www.ivoa.net/xml/VOTable/v1.3'>
      <RESOURCE type='results'>
        <INFO name='QUERY_STATUS' value='OK'/>
        <TABLE>
          <PARAM name='DetectorType' datatype='int' value='1'/>
          <PARAM name='MagSys' datatype='char' arraysize='*' value='AB'/>
          <PARAM name='ZeroPoint' datatype='double' value='3631'/>
          <PARAM name='WavelengthRef' datatype='double' value='5000'/>
          <FIELD name='Wavelength' datatype='double' unit='Angstrom'/>
          <FIELD name='Transmission' datatype='double'/>
          <DATA><TABLEDATA>
            <TR><TD>4000</TD><TD>0</TD></TR>
            <TR><TD>4500</TD><TD>1</TD></TR>
            <TR><TD>5500</TD><TD>1</TD></TR>
            <TR><TD>6000</TD><TD>0</TD></TR>
          </TABLEDATA></DATA>
        </TABLE>
      </RESOURCE>
    </VOTABLE>"""

    class FakeResponse:
        content = payload

        @staticmethod
        def raise_for_status() -> None:
            return None

    class FakeSession:
        @staticmethod
        def get(url: str, timeout: float):
            assert "PhotCalID=TEST%2Fremote%2FAB" in url
            assert timeout == 5.0
            return FakeResponse()

    response = fetch_filter_response(
        "TEST/remote",
        "AB",
        cache_dir=tmp_path,
        timeout=5.0,
        session=FakeSession(),
    )

    assert isinstance(response, FilterResponse)
    assert response.detector_type == "photon"
    assert response.zero_point_jy == pytest.approx(3631.0)
    assert response.cache_format_version == CACHE_FORMAT_VERSION
    assert response.refresh_provenance == "cache_miss_upstream_fetch"
    assert len(response.upstream_query_id) == 64
    query = json.loads(response.upstream_query_json)
    assert query["requested_mag_system"] == "AB"
    assert query["parameters"] == {"PhotCalID": "TEST/remote/AB", "VERB": "2"}
    assert datetime.fromisoformat(response.retrieved_at_utc).tzinfo is not None
    assert datetime.fromisoformat(response.cached_at_utc).tzinfo is not None

    metadata_path = next(tmp_path.glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["upstream_query_id"] == response.upstream_query_id
    assert metadata["retrieved_at_utc"] == response.retrieved_at_utc
    assert metadata["refresh_provenance"] == "cache_miss_upstream_fetch"

    response_set = {("TEST/remote", "AB"): response}
    initial_manifest = response_manifest_hash(response_set)
    initial_audit_manifest = response_audit_manifest_hash(response_set)
    refreshed = fetch_filter_response(
        "TEST/remote",
        "AB",
        cache_dir=tmp_path,
        timeout=5.0,
        session=FakeSession(),
        force=True,
    )
    assert refreshed.upstream_query_id == response.upstream_query_id
    assert refreshed.refresh_provenance == "forced_upstream_refresh"
    assert datetime.fromisoformat(refreshed.retrieved_at_utc).tzinfo is not None
    refreshed_set = {("TEST/remote", "AB"): refreshed}
    assert response_manifest_hash(refreshed_set) == initial_manifest
    assert response_audit_manifest_hash(refreshed_set) != initial_audit_manifest

    offline = load_cached_filter_response("TEST/remote", "AB", tmp_path)
    assert offline is not None
    assert offline.upstream_query_id == refreshed.upstream_query_id
    assert offline.retrieved_at_utc == refreshed.retrieved_at_utc
    assert offline.refresh_provenance == "forced_upstream_refresh"


def test_response_science_manifest_tracks_science_not_cache_operations(monkeypatch) -> None:
    base = FilterResponse(
        filter_id="TEST/manifest",
        wavelength_angstrom=np.asarray([4000.0, 5000.0, 6000.0]),
        throughput=np.asarray([0.0, 1.0, 0.0]),
        detector_type="photon",
        mag_system="Vega",
        zero_point_jy=3600.0,
        svo_calibration_wavelength_ref_angstrom=5050.0,
        zero_point_contract=CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT,
        source_url="https://example.invalid/first",
        retrieved_at_utc="2026-01-01T00:00:00+00:00",
        cached_at_utc="2026-01-01T00:00:01+00:00",
        upstream_query_id="query-v1",
        upstream_query_json='{"ID":"TEST/manifest"}',
        refresh_provenance="cache_miss_upstream_fetch",
        cache_format_version=2,
    )
    key = (base.filter_id, "Vega")
    science_hash = response_manifest_hash({key: base})
    audit_hash = response_audit_manifest_hash({key: base})

    operational_refresh = replace(
        base,
        source_url="https://example.invalid/refreshed",
        retrieved_at_utc="2026-02-02T00:00:00+00:00",
        cached_at_utc="2026-02-02T00:00:01+00:00",
        refresh_provenance="forced_upstream_refresh",
        cache_format_version=99,
    )
    assert response_manifest_hash({key: operational_refresh}) == science_hash
    assert response_audit_manifest_hash({key: operational_refresh}) != audit_hash

    changed_curve = replace(
        base,
        throughput=np.asarray([0.0, 0.8, 0.0]),
        response_hash="",
    )
    science_changes = (
        changed_curve,
        replace(base, detector_type="energy"),
        replace(base, zero_point_jy=3550.0),
        replace(base, zero_point_contract="different-count-ratio-contract"),
        replace(base, svo_calibration_wavelength_ref_angstrom=5100.0),
        replace(base, upstream_query_id="query-v2"),
    )
    for changed in science_changes:
        assert response_manifest_hash({key: changed}) != science_hash

    monkeypatch.setattr(
        synthetic_photometry,
        "SYNTHETIC_PHOTOMETRY_VERSION",
        "bandpass-union-grid-test-next",
    )
    assert response_manifest_hash({key: base}) != science_hash


def test_cache_identity_is_filter_only_and_calibrations_are_views(tmp_path) -> None:
    ab_response = top_hat_response("TEST/shared", 5500.0, 800.0, mag_system="AB")
    ab_paths = save_filter_response(ab_response, tmp_path, requested_mag_system="AB")
    vega_response = FilterResponse(
        filter_id=ab_response.filter_id,
        wavelength_angstrom=ab_response.wavelength_angstrom,
        throughput=ab_response.throughput,
        detector_type=ab_response.detector_type,
        mag_system="Vega",
        zero_point_jy=3600.0,
        wavelength_ref_angstrom=5500.0,
    )
    vega_paths = save_filter_response(vega_response, tmp_path, requested_mag_system="Vega")

    assert ab_paths == vega_paths
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    loaded_ab = load_cached_filter_response("TEST/shared", "AB", tmp_path)
    loaded_vega = load_cached_filter_response("TEST/shared", "Vega", tmp_path)
    loaded_jy = load_cached_filter_response("TEST/shared", "Jy", tmp_path)
    assert loaded_ab is not None and loaded_ab.zero_point_jy == pytest.approx(3631.0)
    assert loaded_vega is not None and loaded_vega.zero_point_jy == pytest.approx(3600.0)
    assert loaded_jy is not None
    assert loaded_jy.mag_system == "Jy"
    assert loaded_jy.zero_point_jy is None
    assert loaded_jy.response_hash == loaded_ab.response_hash == loaded_vega.response_hash

    responses, failures = build_response_map(
        [("TEST/shared", "AB"), ("TEST/shared", "Vega"), ("TEST/shared", "Jy")],
        cache_dir=tmp_path,
        allow_download=False,
    )
    assert not failures
    assert set(responses) == {
        ("TEST/shared", "AB"),
        ("TEST/shared", "Vega"),
        ("TEST/shared", "Jy"),
    }


def test_native_jy_svo_request_reloads_when_svo_returns_vega(tmp_path) -> None:
    payload = b"""<?xml version='1.0'?>
    <VOTABLE version='1.3' xmlns='http://www.ivoa.net/xml/VOTable/v1.3'>
      <RESOURCE type='results'>
        <INFO name='QUERY_STATUS' value='OK'/>
        <TABLE>
          <PARAM name='DetectorType' datatype='int' value='1'/>
          <PARAM name='MagSys' datatype='char' arraysize='*' value='Vega'/>
          <PARAM name='ZeroPoint' datatype='double' value='280.0'/>
          <PARAM name='WavelengthRef' datatype='double' value='35000'/>
          <FIELD name='Wavelength' datatype='double' unit='Angstrom'/>
          <FIELD name='Transmission' datatype='double'/>
          <DATA><TABLEDATA>
            <TR><TD>30000</TD><TD>0</TD></TR>
            <TR><TD>32000</TD><TD>1</TD></TR>
            <TR><TD>38000</TD><TD>1</TD></TR>
            <TR><TD>40000</TD><TD>0</TD></TR>
          </TABLEDATA></DATA>
        </TABLE>
      </RESOURCE>
    </VOTABLE>"""

    class FakeResponse:
        content = payload

        @staticmethod
        def raise_for_status() -> None:
            return None

    class FakeSession:
        @staticmethod
        def get(url: str, timeout: float):
            assert "ID=TEST%2Finfrared" in url
            assert "PhotCalID" not in url
            return FakeResponse()

    fetched = fetch_filter_response(
        "TEST/infrared",
        "Jy",
        cache_dir=tmp_path,
        session=FakeSession(),
    )
    offline = load_cached_filter_response("TEST/infrared", "Jy", tmp_path)

    assert fetched.mag_system == "Jy"
    assert fetched.zero_point_jy is None
    assert offline is not None
    assert offline.mag_system == "Jy"
    assert offline.response_hash == fetched.response_hash
    assert svo_calibration_reference_wavelength_angstrom(offline) == pytest.approx(35000.0)


def test_legacy_vega_keyed_cache_can_serve_native_jy_registration(tmp_path) -> None:
    response = top_hat_response("TEST/legacy-jy", 35000.0, 5000.0, mag_system="Vega")
    stem = _legacy_cache_stem(response.filter_id, "Vega")
    np.savez_compressed(
        tmp_path / f"{stem}.npz",
        wavelength_angstrom=response.wavelength_angstrom,
        throughput=response.throughput,
    )
    (tmp_path / f"{stem}.json").write_text(
        json.dumps(
            {
                "filter_id": response.filter_id,
                "detector_type": response.detector_type,
                "mag_system": "Vega",
                "zero_point_jy": 280.0,
                "wavelength_ref_angstrom": 35000.0,
                "source_url": "legacy",
                "response_hash": response.response_hash,
            }
        )
    )

    loaded = load_cached_filter_response(response.filter_id, "Jy", tmp_path)
    assert loaded is not None
    assert loaded.mag_system == "Jy"
    assert loaded.zero_point_jy is None
    assert loaded.response_hash == response.response_hash
    assert loaded.cache_format_version == 1
    assert loaded.refresh_provenance == "legacy_cache_v1"


def test_v2_hash_validated_cache_remains_readable_offline(tmp_path) -> None:
    response = top_hat_response("TEST/v2-cache", 5500.0, 800.0, mag_system="AB")
    _, meta_path = save_filter_response(response, tmp_path)
    metadata = json.loads(meta_path.read_text())
    metadata["cache_format_version"] = 2
    for key in (
        "cached_at_utc",
        "refresh_provenance",
        "retrieved_at_utc",
        "svo_calibration_wavelength_ref_angstrom",
        "upstream_query_id",
        "upstream_query_json",
        "zero_point_contract",
    ):
        metadata.pop(key, None)
    for calibration in metadata.get("calibrations", {}).values():
        calibration.pop("svo_calibration_wavelength_ref_angstrom", None)
        calibration.pop("zero_point_contract", None)
    metadata["metadata_hash"] = _metadata_hash(metadata)
    meta_path.write_text(json.dumps(metadata))

    loaded = load_cached_filter_response(response.filter_id, "AB", tmp_path)
    assert loaded is not None
    assert loaded.response_hash == response.response_hash
    assert loaded.cache_format_version == 2
    assert loaded.refresh_provenance == "pre_provenance_cache_v2"


def test_cache_rejects_array_or_metadata_tampering(tmp_path) -> None:
    response = top_hat_response("TEST/hash", 5500.0, 800.0, mag_system="AB")
    data_path, meta_path = save_filter_response(response, tmp_path)
    with np.load(data_path, allow_pickle=False) as arrays:
        wavelength = np.asarray(arrays["wavelength_angstrom"])
        throughput = np.asarray(arrays["throughput"]).copy()
        generation = np.asarray(arrays["cache_generation"])
    throughput[1] *= 0.5
    np.savez_compressed(
        data_path,
        wavelength_angstrom=wavelength,
        throughput=throughput,
        cache_generation=generation,
    )
    assert load_cached_filter_response("TEST/hash", "AB", tmp_path) is None

    # Re-save a valid pair, then alter metadata without updating its content
    # hash.  Calibration corruption must be rejected just like curve corruption.
    save_filter_response(response, tmp_path)
    metadata = json.loads(meta_path.read_text())
    metadata["detector_type"] = "energy"
    meta_path.write_text(json.dumps(metadata))
    assert load_cached_filter_response("TEST/hash", "AB", tmp_path) is None


def test_union_grid_preserves_model_structure_between_response_nodes() -> None:
    wave = np.linspace(3500.0, 6500.0, 6001)
    fnu_jy = np.ones_like(wave)
    fnu_jy[np.abs(wave - 5000.0) <= 100.0] = 0.05
    flux_lambda = fnu_jy * JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    response = FilterResponse(
        filter_id="TEST/sparse-response",
        wavelength_angstrom=np.array([4000.0, 4001.0, 5999.0, 6000.0]),
        throughput=np.array([0.0, 1.0, 1.0, 0.0]),
    )

    predicted = bandpass_flux_nu_jy(wave, flux_lambda, response)
    throughput = np.interp(wave, response.wavelength_angstrom, response.throughput, left=0.0, right=0.0)
    numerator = _TRAPEZOID(flux_lambda * throughput * wave, wave)
    reference = JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    denominator = _TRAPEZOID(reference * throughput * wave, wave)

    assert predicted == pytest.approx(numerator / denominator, rel=2.0e-6)
    assert predicted < 0.93  # A response-node-only sampler would incorrectly return about one Jy.


def test_native_ab_and_vega_magnitude_operators() -> None:
    wave = np.geomspace(3000.0, 10000.0, 5000)
    response = top_hat_response("TEST/native-mag", 5500.0, 1800.0)

    ab_flux = 3631.0 * JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    assert predict_native_observable(wave, ab_flux, response, ab_calibration()) == pytest.approx(
        0.0,
        abs=2.0e-5,
    )

    vega_zero_point = 2500.0
    vega_flux = vega_zero_point * JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    vega = vega_zero_point_calibration(vega_zero_point, calibration_id="TEST/Vega")
    assert predict_native_observable(wave, vega_flux, response, vega) == pytest.approx(
        0.0,
        abs=2.0e-5,
    )


def test_response_matched_vega_zero_point_is_exact_count_rate_ratio() -> None:
    wave = np.linspace(4000.0, 7000.0, 3001)
    throughput = np.sin(np.pi * (wave - wave[0]) / (wave[-1] - wave[0])) ** 2
    response = FilterResponse(
        filter_id="TEST/vega-count-ratio",
        wavelength_angstrom=wave,
        throughput=throughput,
        detector_type="photon",
    )

    # This is an arbitrary, explicitly supplied reference spectrum.  The
    # contract and algebra do not synthesize or approximate a Vega spectrum.
    flat_one_jy_flam = JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    reference_flux = 2800.0 * flat_one_jy_flam * (
        1.0 + 0.18 * np.sin(2.0 * np.pi * (wave - 4000.0) / 3000.0)
    )
    model_flux = 1900.0 * flat_one_jy_flam * np.power(wave / 5500.0, -0.7)

    zero_point_jy = response_matched_zero_point_jy(wave, reference_flux, response)
    calibration = response_matched_vega_zero_point_calibration(
        zero_point_jy,
        calibration_id="TEST/pinned-reference-zero-point",
    )
    predicted_magnitude = predict_native_observable(
        wave,
        model_flux,
        response,
        calibration,
    )

    reference_count_rate = _TRAPEZOID(reference_flux * throughput * wave, wave)
    model_count_rate = _TRAPEZOID(model_flux * throughput * wave, wave)
    direct_count_ratio_magnitude = -2.5 * np.log10(model_count_rate / reference_count_rate)

    assert calibration.forward_contract == CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT
    assert predicted_magnitude == pytest.approx(direct_count_ratio_magnitude, rel=2.0e-12)


def test_pivot_svo_and_mission_reference_wavelengths_are_distinct() -> None:
    response = FilterResponse(
        filter_id="TEST/pivot",
        wavelength_angstrom=np.array([4000.0, 4500.0, 6000.0, 7000.0]),
        throughput=np.array([0.0, 1.0, 0.8, 0.0]),
        wavelength_ref_angstrom=6000.0,
    )
    numerator = _TRAPEZOID(
        response.throughput * response.wavelength_angstrom,
        response.wavelength_angstrom,
    )
    denominator = _TRAPEZOID(
        response.throughput / response.wavelength_angstrom,
        response.wavelength_angstrom,
    )

    pivot = response_pivot_wavelength_angstrom(response)
    mission_calibration = quoted_fnu_calibration("TEST/mission-reference", 6500.0)
    assert pivot == pytest.approx(np.sqrt(numerator / denominator), rel=1.0e-12)
    assert svo_calibration_reference_wavelength_angstrom(response) == pytest.approx(6000.0)
    assert mission_calibration.reference_wavelength_angstrom == pytest.approx(6500.0)
    assert pivot != pytest.approx(svo_calibration_reference_wavelength_angstrom(response))
    assert mission_calibration.reference_wavelength_angstrom != pytest.approx(
        svo_calibration_reference_wavelength_angstrom(response)
    )


def test_energy_detector_pivot_uses_energy_response_weighting() -> None:
    response = FilterResponse(
        filter_id="TEST/energy-pivot",
        wavelength_angstrom=np.array([5200.0, 5800.0, 6900.0, 7600.0]),
        throughput=np.array([0.0, 0.9, 1.0, 0.0]),
        detector_type="energy",
    )
    wave = response.wavelength_angstrom
    throughput = response.throughput
    expected_energy_pivot = np.sqrt(
        _TRAPEZOID(throughput, wave)
        / _TRAPEZOID(throughput / np.square(wave), wave)
    )
    incorrect_photon_pivot = np.sqrt(
        _TRAPEZOID(throughput * wave, wave)
        / _TRAPEZOID(throughput / wave, wave)
    )

    pivot = response_pivot_wavelength_angstrom(response)

    assert pivot == pytest.approx(expected_energy_pivot, rel=1.0e-12)
    assert pivot != pytest.approx(incorrect_photon_pivot, rel=1.0e-6)


def test_quoted_fnu_operator_reproduces_reference_spectrum() -> None:
    wave = np.geomspace(4000.0, 12000.0, 6000)
    response = top_hat_response("TEST/quoted", 7000.0, 3000.0)
    calibration = quoted_fnu_calibration(
        "TEST/nuFnu",
        7000.0,
        reference_spectrum=REFERENCE_NU_FNU_CONSTANT,
    )
    reference_flux_lambda = (
        reference_fnu_jy(wave, calibration) * JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    )

    assert predict_native_observable(wave, reference_flux_lambda, response, calibration) == pytest.approx(
        1.0,
        rel=2.0e-5,
    )


def test_mips_blackbody_quoted_fnu_operator_reproduces_one_jy_reference() -> None:
    wave = np.geomspace(150000.0, 320000.0, 6000)
    response = top_hat_response("Spitzer/MIPS.24mu", 235000.0, 90000.0)
    calibration = mission_quoted_fnu_calibration("Spitzer/MIPS.24mu", 235000.0)
    reference_flux_lambda = (
        reference_fnu_jy(wave, calibration) * JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    )

    assert predict_native_observable(wave, reference_flux_lambda, response, calibration) == pytest.approx(
        1.0,
        rel=2.0e-5,
    )
