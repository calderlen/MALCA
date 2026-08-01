from __future__ import annotations

import math
import json
import sys
import types
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import Table

import malca.review.sed as review_sed
from malca.review.sed import (
    ALL_CATALOG_SOURCES,
    BROAD_CLASSIFICATION_SED_SOURCES,
    CANONICAL_SED_COLUMNS,
    LSUN_ERG_S,
    NSC_INSTRUMENT_BANDPASSES,
    SED_COLUMNS,
    SOURCE_COLORS,
    APASS_B_RED_LEAK_COLOR_THRESHOLD,
    bandpass_for,
    build_sed_dataframe,
    build_sed_figure,
    distance_pc_from_payload,
    extinction_av_from_payload,
    flux_lambda_from_flux_nu_jy,
    flux_nu_jy_from_mag,
    load_sed_rows,
    query_gaia_gspc_photometry,
    query_gaia_xp_sampled,
    query_nsc_photometry,
    prepare_canonical_sed_measurements,
    prepare_sed_measurement_row,
    resolve_sed_sources,
    rows_from_payload,
    sed_source_statuses,
    upsert_sed_rows,
)
from malca.review.pipeline import (
    ReviewStageExecutionError,
    detect_sed_model_status,
    detect_sed_photometry_status,
    run_missing_stages,
)
from malca.review.store import db_connect
from malca.review.sed_storage import (
    CANONICAL_SED_NORMALIZATION_VERSION,
    make_sed_normalization_hash,
    store_sed_normalizations,
)
from malca.enrich.swift import DEFAULT_SWIFT_CATALOGS
from malca.enrichment.sed_model import (
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    SED_MODEL_FIT_VERSION,
    SED_MODEL_POINT_COLUMNS,
    load_sed_model_curves,
    load_sed_model_fits,
    load_sed_model_points,
    upsert_sed_model_results,
)
from malca.enrichment.synthetic_photometry import (
    FilterResponse,
    build_response_map,
    response_pivot_wavelength_angstrom,
    save_filter_response,
)


def _table_from_dicts(rows: list[dict]) -> Table:
    if not rows:
        return Table()
    names = list(rows[0])
    values = [tuple(row.get(name) for name in names) for row in rows]
    return Table(rows=values, names=names)


def test_matplotlib_sed_legend_follows_first_plotted_wavelength() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {
            "sed_mode": ["observed"] * 5,
            "source": ["AllWISE", "2MASS", "IPHAS", "Gaia DR3", "APASS"],
            "plot_lambda_angstrom": [
                33526.0,
                12350.0,
                6568.0,
                5320.0,
                4380.0,
            ],
            "lambda_eff_angstrom": [
                33526.0,
                12350.0,
                6568.0,
                5320.0,
                4380.0,
            ],
            "lambda_l_lambda": [
                5.0e33,
                4.0e33,
                3.0e33,
                2.0e33,
                1.0e33,
            ],
            "quality_flags": [""] * 5,
        }
    )
    fig, axis = plt.subplots()
    try:
        review_sed._draw_sed_photometry_matplotlib(
            axis,
            frame,
            y_col="lambda_l_lambda",
            mode="observed",
        )
        _handles, labels = axis.get_legend_handles_labels()
    finally:
        plt.close(fig)

    assert labels == ["APASS", "Gaia DR3", "IPHAS", "2MASS", "AllWISE"]


def test_iphas_color_is_between_adjacent_sed_sources() -> None:
    def rgb(hex_color: str) -> tuple[int, int, int]:
        return tuple(
            int(hex_color[index : index + 2], 16)
            for index in (1, 3, 5)
        )

    gaia = rgb(SOURCE_COLORS["Gaia DR3"])
    two_mass = rgb(SOURCE_COLORS["2MASS"])
    expected_midpoint = tuple(
        round((left + right) / 2)
        for left, right in zip(gaia, two_mass)
    )

    assert rgb(SOURCE_COLORS["IPHAS"]) == expected_midpoint


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("False", 0),
        ("0", 0),
        (np.nan, 0),
        (pd.NA, 0),
        (None, 0),
        ("true", 1),
        (1, 1),
    ],
)
def test_to_bool_int_handles_persisted_null_and_text_encodings(value: object, expected: int) -> None:
    assert review_sed._to_bool_int(value) == expected


def _patch_vizier_query(monkeypatch, tables_by_catalog: dict[str, Table]) -> None:
    vizier_module = pytest.importorskip("astroquery.vizier")

    class FakeVizier:
        def __init__(self, *_args, **_kwargs) -> None:
            self.TIMEOUT = None

        def query_region(self, *_args, catalog=None, **_kwargs):
            table = tables_by_catalog.get(str(catalog))
            return [] if table is None else [table]

    monkeypatch.setattr(vizier_module, "Vizier", FakeVizier)


def test_ab_and_vega_flux_conversions() -> None:
    ps1_g = bandpass_for("Pan-STARRS", "g")
    wise_w1 = bandpass_for("AllWISE", "W1")

    assert ps1_g is not None
    assert wise_w1 is not None
    assert math.isclose(flux_nu_jy_from_mag(0.0, ps1_g), 3631.0, rel_tol=1e-12)
    assert math.isclose(flux_nu_jy_from_mag(0.0, wise_w1), 309.540, rel_tol=1e-12)
    assert flux_lambda_from_flux_nu_jy(3631.0, ps1_g.lambda_eff_angstrom) > 0


def test_sed_sources_default_is_bounded_and_all_is_explicit() -> None:
    default_sources = resolve_sed_sources("default")
    custom_sources = resolve_sed_sources("payload,ps1")

    assert default_sources == BROAD_CLASSIFICATION_SED_SOURCES
    assert resolve_sed_sources("all") == ALL_CATALOG_SOURCES
    assert resolve_sed_sources("broad") == BROAD_CLASSIFICATION_SED_SOURCES
    assert resolve_sed_sources("classification") == BROAD_CLASSIFICATION_SED_SOURCES
    assert custom_sources == ("payload", "ps1")
    assert resolve_sed_sources("far-ir") == ("akari", "iras", "herschel", "apex_laboca")
    assert {"akari", "iras", "herschel"}.issubset(set(resolve_sed_sources("all")))
    assert {
        "gaia_xp", "galex", "catwise", "nsc", "vhs", "viking", "swift_uvot", "xmm_om",
    }.issubset(set(resolve_sed_sources("all")))


def test_new_catalog_bandpasses_are_registered() -> None:
    expected = {
        "GALEX": {"FUV", "NUV"},
        "CatWISE2020": {"W1", "W2"},
        "NOIRLab NSC DR2": {"u", "g", "r", "i", "z", "Y", "VR"},
        "VISTA/VHS": {"Y", "J", "H", "Ks"},
        "VISTA/VIKING": {"Z", "Y", "J", "H", "Ks"},
        "Swift/UVOT": {"UVW2", "UVM2", "UVW1", "U", "B", "V"},
        "XMM-OM": {"UVW2", "UVM2", "UVW1", "U", "B", "V"},
    }

    for source, bands in expected.items():
        assert all(bandpass_for(source, band) is not None for band in bands)


def test_exact_nsc_and_mission_reference_registrations() -> None:
    expected = {
        ("c4d", "u"): "CTIO/DECam.u",
        ("c4d", "g"): "CTIO/DECam.g",
        ("c4d", "r"): "CTIO/DECam.r",
        ("c4d", "i"): "CTIO/DECam.i",
        ("c4d", "z"): "CTIO/DECam.z",
        ("c4d", "y"): "CTIO/DECam.Y",
        ("c4d", "vr"): "CTIO/DECam.VR_filter",
        ("k4m", "z"): "KPNO/MOSAIC.zd_DECam",
        ("ksb", "g"): "BOK/BASS.g",
        ("ksb", "r"): "BOK/BASS.r",
    }
    assert {
        key: value.svo_filter_id for key, value in NSC_INSTRUMENT_BANDPASSES.items()
    } == expected
    mosaic_z = bandpass_for("NOIRLab NSC DR2", "z", instrument="Mosaic3")
    assert mosaic_z is not None and mosaic_z.svo_filter_id == "KPNO/MOSAIC.zd_DECam"
    assert mosaic_z.response_kind == "filter_only_proxy"
    assert mosaic_z.fit_policy == "diagnostic_only"
    decam_vr = bandpass_for("NOIRLab NSC DR2", "VR", instrument="c4d")
    assert decam_vr is not None and decam_vr.response_kind == "filter_only_proxy"
    assert decam_vr.fit_policy == "diagnostic_only"
    assert bandpass_for("NOIRLab NSC DR2", "g", instrument="c4d").fit_policy == "photosphere"
    registered_ids = {bp.svo_filter_id for bp in review_sed.SED_BANDPASSES.values()}
    assert set(expected.values()).issubset(registered_ids)

    mips24 = bandpass_for("Spitzer SEIP", "MIPS24")
    assert mips24 is not None
    assert mips24.lambda_reference_angstrom == pytest.approx(236750.0)
    uvot_bands = ("UVW2", "UVM2", "UVW1", "U", "B", "V")
    assert all(bandpass_for("Swift/UVOT", band).mag_system == "AB" for band in uvot_bands)
    assert all(bandpass_for("XMM-OM", band).mag_system == "AB" for band in uvot_bands)


def test_nsc_filter_only_response_registration_loads_from_offline_cache(tmp_path: Path) -> None:
    bandpass = bandpass_for("NOIRLab NSC DR2", "VR", instrument="c4d")
    assert bandpass is not None
    cached = FilterResponse(
        filter_id=str(bandpass.svo_filter_id),
        wavelength_angstrom=np.array([5800.0, 6200.0, 7000.0]),
        throughput=np.array([0.0, 1.0, 0.0]),
        mag_system="AB",
    )
    save_filter_response(cached, tmp_path)

    responses, failures = build_response_map(
        [(str(bandpass.svo_filter_id), bandpass.mag_system)],
        cache_dir=tmp_path,
        allow_download=False,
    )

    assert failures == {}
    assert responses[("CTIO/DECam.VR_filter", "AB")].response_hash == cached.response_hash


def test_v3_registry_marks_apass_proxies_and_nsc_mixed_means() -> None:
    apass_b = bandpass_for("APASS", "B")
    apass_g = bandpass_for("APASS", "g")
    nsc_u = bandpass_for("NOIRLab NSC DR2", "u")

    assert apass_b is not None and apass_g is not None and nsc_u is not None
    assert apass_b.response_kind == "standardized_system_proxy"
    assert apass_g.fit_policy == "photosphere_proxy"
    assert "apass_b_red_leak_risk" not in apass_b.policy_flags
    assert nsc_u.response_kind == "mixed_unknown"
    assert nsc_u.fit_policy == "diagnostic_only"
    assert nsc_u.svo_filter_id is None


def test_vphas_and_halpha_registry_semantics_are_explicit() -> None:
    for band in ("u", "g", "r", "i"):
        bandpass = bandpass_for("VPHAS+", band)
        assert bandpass is not None
        assert bandpass.mag_system == "Vega"
        assert bandpass.response_kind == "filter_ccd_natural_system_proxy"
        assert bandpass.fit_policy == "photosphere_proxy"
        assert bandpass.systematic_floor_mag == pytest.approx(0.04)
        assert {
            "natural_system_proxy",
            "filter_plus_ccd_response",
            "systematic_floor_mag=0.04",
        }.issubset(bandpass.policy_flags)

    for source in ("IPHAS", "VPHAS+"):
        halpha = bandpass_for(source, "Halpha")
        assert halpha is not None
        assert halpha.mag_system == "Vega"
        assert halpha.response_kind == "emission_line_filter"
        assert halpha.fit_policy == "diagnostic_only"
        assert {"emission_line", "diagnostic_only"}.issubset(halpha.policy_flags)


def test_apass_b_red_leak_policy_excludes_known_very_red_color() -> None:
    rows = rows_from_payload({
        "candidate_id": "cand-apass-red",
        "apass_b": 18.0,
        "cousins_i": 18.0 - APASS_B_RED_LEAK_COLOR_THRESHOLD - 0.1,
    })

    flags = str(rows.iloc[0]["quality_flags"])
    assert "standardized_system_proxy" in flags
    assert "apass_b_red_leak_likely" in flags
    assert "bad_quality" in flags


def test_apass_b_red_leak_policy_is_explicit_when_color_unavailable() -> None:
    rows = rows_from_payload({"candidate_id": "cand-apass-unknown", "apass_b": 15.0})

    assert "apass_b_red_leak_unassessed" in str(rows.iloc[0]["quality_flags"])


def test_swift_enrichment_uses_real_uvot_catalog() -> None:
    assert DEFAULT_SWIFT_CATALOGS["swift_uvotssc1"] == "II/339/uvotssc1"
    assert "II/363/uvotssc2" not in DEFAULT_SWIFT_CATALOGS.values()


def test_payload_rows_include_vphas_photometry() -> None:
    rows = rows_from_payload(
        {
            "candidate_id": "cand-vphas",
            "vphas_u": 17.1,
            "vphas_g": 16.8,
            "vphas_r": 16.5,
            "vphas_i": 16.2,
            "vphas_ha": 16.0,
        }
    )

    assert set(rows["source"]) == {"VPHAS+"}
    assert set(rows["band"]) == {"u", "g", "r", "i", "Halpha"}
    assert set(rows["mag_system"]) == {"Vega"}
    broad = rows.loc[rows["band"] != "Halpha"]
    halpha = rows.loc[rows["band"] == "Halpha"].iloc[0]
    assert set(broad["passband_fidelity"]) == {"natural_system_proxy"}
    assert set(pd.to_numeric(broad["systematic_floor_mag"])) == {0.04}
    assert set(broad["fit_policy"]) == {"photosphere_proxy"}
    assert halpha["fit_policy"] == "diagnostic_only"
    assert halpha["passband_fidelity"] != "exact"


@pytest.mark.parametrize(
    ("payload", "filter_id", "zero_point_jy"),
    [
        ({"candidate_id": "cand-gaia-cache", "phot_g_mean_mag": 10.0}, "GAIA/GAIA3.G", 3228.75),
        ({"candidate_id": "cand-wise-cache", "w3": 8.0}, "WISE/WISE.W3", 29.75),
    ],
)
def test_payload_rows_use_cached_response_pivot_hash_and_vega_zero_point(
    monkeypatch,
    payload: dict,
    filter_id: str,
    zero_point_jy: float,
) -> None:
    response = FilterResponse(
        filter_id=filter_id,
        wavelength_angstrom=np.array([5000.0, 6200.0, 8000.0]),
        throughput=np.array([0.0, 1.0, 0.0]),
        mag_system="Vega",
        zero_point_jy=zero_point_jy,
    )
    calls: list[tuple[str, str]] = []

    def cache_loader(requested_filter_id: str, requested_system: str) -> FilterResponse | None:
        calls.append((requested_filter_id, requested_system))
        return response if requested_filter_id == filter_id and requested_system == "Vega" else None

    monkeypatch.setattr(review_sed, "_load_cached_registered_response", cache_loader)

    rows = rows_from_payload(payload)
    row = rows.iloc[0]
    magnitude = float(row["mag"])
    expected_pivot = response_pivot_wavelength_angstrom(response)

    assert float(row["observed_flux_nu_jy"]) == pytest.approx(
        zero_point_jy * 10.0 ** (-0.4 * magnitude)
    )
    assert float(row["plot_lambda_angstrom"]) == pytest.approx(expected_pivot)
    assert row["plot_lambda_kind"] == "response_pivot"
    assert row["response_hash"] == response.response_hash
    assert str(row["calibration_hash"])
    assert row["calibration_source"] == "response_calibration"
    assert "legacy_vega_zero_point_fallback" not in str(row["quality_flags"])
    assert (filter_id, "Vega") in calls


def test_real_cache_only_response_loader_memoizes_only_positive_hits(monkeypatch) -> None:
    import malca.enrichment.synthetic_photometry as synthetic_photometry

    sentinel = object()
    calls: list[tuple[str, str]] = []

    responses: list[object | None] = [None, sentinel]

    def disk_loader(filter_id: str, mag_system: str) -> object | None:
        calls.append((filter_id, mag_system))
        return responses.pop(0) if responses else sentinel

    monkeypatch.setattr(synthetic_photometry, "load_cached_filter_response", disk_loader)
    review_sed._clear_cached_registered_responses()
    try:
        first = review_sed._load_cached_registered_response("TEST/memoized", "Vega")
        second = review_sed._load_cached_registered_response("TEST/memoized", "Vega")
        third = review_sed._load_cached_registered_response("TEST/memoized", "Vega")

        assert first is None
        assert second is sentinel and third is sentinel
        assert calls == [("TEST/memoized", "Vega"), ("TEST/memoized", "Vega")]
        assert len(review_sed._REGISTERED_RESPONSE_CACHE) <= 256
    finally:
        review_sed._clear_cached_registered_responses()


def test_vphas_cached_vega_calibration_preserves_proxy_classification(monkeypatch) -> None:
    bandpass = bandpass_for("VPHAS+", "g")
    assert bandpass is not None
    response = FilterResponse(
        filter_id=str(bandpass.svo_filter_id),
        wavelength_angstrom=np.array([4000.0, 4800.0, 5600.0]),
        throughput=np.array([0.0, 1.0, 0.0]),
        mag_system="Vega",
        zero_point_jy=3900.0,
    )
    monkeypatch.setattr(
        review_sed,
        "_load_cached_registered_response",
        lambda filter_id, mag_system: response
        if filter_id == bandpass.svo_filter_id and mag_system == "Vega"
        else None,
    )

    row = rows_from_payload({"candidate_id": "cand-vphas-cache", "vphas_g": 15.0}).iloc[0]

    assert row["mag_system"] == "Vega"
    assert row["observable_kind"] == "vega_mag"
    assert row["passband_fidelity"] == "natural_system_proxy"
    assert row["fit_policy"] == "photosphere_proxy"
    assert float(row["systematic_floor_mag"]) == pytest.approx(0.04)
    assert float(row["observed_flux_nu_jy"]) == pytest.approx(3900.0e-6)


def test_legacy_ab_cache_never_supplies_a_vphas_vega_zero_point(monkeypatch) -> None:
    bandpass = bandpass_for("VPHAS+", "g")
    assert bandpass is not None
    ab_response = FilterResponse(
        filter_id=str(bandpass.svo_filter_id),
        wavelength_angstrom=np.array([4000.0, 4800.0, 5600.0]),
        throughput=np.array([0.0, 1.0, 0.0]),
        mag_system="AB",
        zero_point_jy=3631.0,
    )
    monkeypatch.setattr(
        review_sed,
        "_load_cached_registered_response",
        lambda filter_id, mag_system: ab_response
        if filter_id == bandpass.svo_filter_id and mag_system == ""
        else None,
    )

    row = rows_from_payload({"candidate_id": "cand-vphas-old-cache", "vphas_g": 0.0}).iloc[0]

    assert float(row["observed_flux_nu_jy"]) == pytest.approx(bandpass.fnu_zero_jy)
    assert float(row["observed_flux_nu_jy"]) != pytest.approx(3631.0)
    assert row["response_hash"] == ab_response.response_hash
    assert "legacy_vega_zero_point_fallback" in str(row["quality_flags"])


def test_direct_jy_rows_never_consult_response_cache_for_display_wavelength(monkeypatch) -> None:
    def forbidden_loader(_filter_id: str, _mag_system: str) -> None:
        raise AssertionError("direct-Jy normalization must not consult SVO cache metadata")

    monkeypatch.setattr(review_sed, "_load_cached_registered_response", forbidden_loader)

    row = prepare_canonical_sed_measurements(
        [{"source": "AKARI", "band": "S9W", "flux_nu_jy": 2.0}],
        candidate_id="cand-direct-jy-cache",
    ).iloc[0]

    assert float(row["plot_lambda_angstrom"]) == pytest.approx(90000.0)
    assert row["plot_lambda_kind"] == "mission_reference"


def test_sed_source_statuses_reports_cached_miss_as_no_match(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame([
        {
            "_cache_candidate_id": "cand-ps1",
            "_cache_status": "miss",
            "_cache_updated_at": "2026-05-19T00:00:00Z",
            "candidate_id": "cand-ps1",
            "source": "ps1",
            "band": None,
        }
    ]).to_parquet(tmp_path / "ps1.parquet", index=False)

    statuses = sed_source_statuses(
        "cand-ps1",
        external_rows=pd.DataFrame(),
        sources=["ps1", "sdss"],
    )
    by_key = {str(item["key"]): item for item in statuses}

    assert by_key["ps1"]["status"] == "miss"
    assert by_key["ps1"]["message"] == "queried; no catalog match"
    assert by_key["sdss"]["status"] == "not_queried"


def test_sed_source_statuses_reports_cached_hit_row_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame([
        {
            "_cache_candidate_id": "cand-gspc",
            "_cache_status": "hit",
            "_cache_updated_at": "2026-05-19T00:00:00Z",
            "candidate_id": "cand-gspc",
            "source": "Gaia GSPC",
            "band": "SDSS_u",
        },
        {
            "_cache_candidate_id": "cand-gspc",
            "_cache_status": "hit",
            "_cache_updated_at": "2026-05-19T00:00:00Z",
            "candidate_id": "cand-gspc",
            "source": "Gaia GSPC",
            "band": "SDSS_g",
        },
    ]).to_parquet(tmp_path / "gaia_gspc.parquet", index=False)

    statuses = sed_source_statuses(
        "cand-gspc",
        external_rows=pd.DataFrame(),
        sources=["gaia_gspc"],
    )
    gspc = {str(item["key"]): item for item in statuses}["gaia_gspc"]

    assert gspc["status"] == "hit"
    assert gspc["n_rows"] == 2
    assert gspc["source_names"] == ["Gaia GSPC"]
    assert gspc["bands"] == ["SDSS_g", "SDSS_u"]


@pytest.mark.parametrize(
    ("cache_status", "expected_message"),
    [
        ("outside_footprint", "outside the catalog footprint"),
        ("error", "retryable"),
    ],
)
def test_sed_source_statuses_distinguishes_non_match_states(
    tmp_path: Path,
    monkeypatch,
    cache_status: str,
    expected_message: str,
) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame([{
        "_cache_candidate_id": "cand-status",
        "_cache_status": cache_status,
        "_cache_updated_at": "2026-07-18T00:00:00Z",
        "candidate_id": "cand-status",
        "source": "ps1",
        "band": None,
    }]).to_parquet(tmp_path / "ps1.parquet", index=False)

    status = sed_source_statuses("cand-status", sources=["ps1"])[0]

    assert status["status"] == cache_status
    assert expected_message in str(status["message"])


def test_retryable_cache_error_is_refetched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame([{
        "_cache_candidate_id": "cand-retry",
        "_cache_status": "error",
        "_cache_updated_at": "2026-07-18T00:00:00Z",
        "candidate_id": "cand-retry",
        "source": "ps1",
        "band": None,
    }]).to_parquet(tmp_path / "ps1.parquet", index=False)
    calls: list[str] = []

    def fetcher(frame: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        calls.extend(frame["candidate_id"].astype(str))
        return review_sed._fetch_result([], {"cand-retry": "miss"})

    review_sed._fetch_sed_source_with_cache(
        "ps1",
        fetcher,
        pd.DataFrame([{"candidate_id": "cand-retry", "ra_deg": 10.0, "dec_deg": 20.0}]),
    )

    assert calls == ["cand-retry"]
    refreshed = pd.read_parquet(tmp_path / "ps1.parquet")
    assert set(refreshed["_cache_status"]) == {"miss"}


def test_unannotated_empty_fetch_remains_retryable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    calls = 0

    def fetcher(_frame: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)

    review_sed._fetch_sed_source_with_cache(
        "catwise",
        fetcher,
        pd.DataFrame([{"candidate_id": "cand-unknown", "ra": 10.0, "dec": 20.0}]),
        chunk_size=1,
        max_attempts=2,
        retry_base_seconds=0.0,
    )

    assert calls == 2
    cached = pd.read_parquet(tmp_path / "catwise.parquet")
    assert set(cached["_cache_status"]) == {"error"}


def test_successful_fetch_replaces_unreadable_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    (tmp_path / "catwise.parquet").write_bytes(b"not a parquet file")

    def fetcher(frame: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        cid = str(frame.iloc[0]["candidate_id"])
        return review_sed._fetch_result([], {cid: "miss"})

    review_sed._fetch_sed_source_with_cache(
        "catwise",
        fetcher,
        pd.DataFrame([{"candidate_id": "repair-cache", "ra": 10.0, "dec": 20.0}]),
        max_attempts=1,
        retry_base_seconds=0.0,
    )

    repaired = pd.read_parquet(tmp_path / "catwise.parquet")
    assert repaired.loc[0, "_cache_candidate_id"] == "repair-cache"
    assert repaired.loc[0, "_cache_status"] == "miss"


def test_source_chunks_checkpoint_before_interruption(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    calls = 0

    def fetcher(frame: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        cid = str(frame.iloc[0]["candidate_id"])
        if calls == 2:
            raise KeyboardInterrupt("simulated interruption")
        return review_sed._fetch_result([], {cid: "miss"})

    candidates = pd.DataFrame([
        {"candidate_id": "checkpoint-a", "ra": 10.0, "dec": 20.0},
        {"candidate_id": "checkpoint-b", "ra": 11.0, "dec": 21.0},
    ])
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        review_sed._fetch_sed_source_with_cache(
            "catwise",
            fetcher,
            candidates,
            chunk_size=1,
            max_attempts=1,
            retry_base_seconds=0.0,
        )

    cached = pd.read_parquet(tmp_path / "catwise.parquet")
    assert set(cached["_cache_candidate_id"].astype(str)) == {"checkpoint-a"}
    assert set(cached["_cache_status"]) == {"miss"}


def test_fetch_manifest_certifies_candidate_source_matrix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame([
        {
            "_cache_candidate_id": "manifest-a",
            "_cache_status": "miss",
            "_cache_updated_at": "2026-07-22T00:00:00Z",
            "candidate_id": "manifest-a",
            "source": "catwise",
            "band": None,
        },
        {
            "_cache_candidate_id": "manifest-b",
            "_cache_status": "error",
            "_cache_updated_at": "2026-07-22T00:00:00Z",
            "candidate_id": "manifest-b",
            "source": "catwise",
            "band": None,
        },
    ]).to_parquet(tmp_path / "catwise.parquet", index=False)
    candidates = pd.DataFrame([
        {"candidate_id": "manifest-a"},
        {"candidate_id": "manifest-b"},
    ])

    manifest = review_sed.build_sed_fetch_manifest(candidates, sources=["catwise"])

    assert len(manifest) == 2
    by_id = manifest.set_index("candidate_id")
    assert by_id.loc["manifest-a", "status"] == "miss"
    assert bool(by_id.loc["manifest-a", "is_complete"])
    assert by_id.loc["manifest-b", "status"] == "error"
    assert not bool(by_id.loc["manifest-b", "is_complete"])
    complete, errors = review_sed.validate_sed_fetch_manifest(
        manifest,
        candidates,
        sources=["catwise"],
    )
    assert not complete
    assert any("non-terminal statuses" in error for error in errors)


@pytest.mark.parametrize(
    "cache_status",
    [
        "catalog_detection",
        "image_detection",
        "upper_limit",
        "ambiguous_counterpart",
        "unusable_measurement",
        "reduction_required",
        "covered_no_detection",
        "catalog_no_match",
        "not_observed",
    ],
)
def test_fetch_manifest_preserves_expanded_terminal_statuses(
    tmp_path: Path,
    monkeypatch,
    cache_status: str,
) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame(
        [
            {
                "_cache_candidate_id": "expanded-status",
                "_cache_status": cache_status,
                "_cache_updated_at": "2026-07-30T00:00:00Z",
                "candidate_id": "expanded-status",
                "source": "AllWISE",
                "band": "W1" if cache_status.endswith("detection") else None,
            }
        ]
    ).to_parquet(tmp_path / "allwise.parquet", index=False)
    candidates = pd.DataFrame([{"candidate_id": "expanded-status"}])

    manifest = review_sed.build_sed_fetch_manifest(
        candidates,
        sources=["allwise"],
    )

    assert manifest.loc[0, "status"] == cache_status
    assert bool(manifest.loc[0, "is_complete"])
    complete, errors = review_sed.validate_sed_fetch_manifest(
        manifest,
        candidates,
        sources=["allwise"],
    )
    assert complete
    assert errors == []


def test_fetch_manifest_preserves_expanded_retryable_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame(
        [
            {
                "_cache_candidate_id": "retryable-status",
                "_cache_status": "query_error",
                "_cache_updated_at": "2026-07-30T00:00:00Z",
                "candidate_id": "retryable-status",
                "source": "AllWISE",
                "band": None,
            }
        ]
    ).to_parquet(tmp_path / "allwise.parquet", index=False)
    candidates = pd.DataFrame([{"candidate_id": "retryable-status"}])

    manifest = review_sed.build_sed_fetch_manifest(
        candidates,
        sources=["allwise"],
    )

    assert manifest.loc[0, "status"] == "query_error"
    assert not bool(manifest.loc[0, "is_complete"])


def test_fetch_manifest_validation_requires_exact_cross_product(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    candidates = pd.DataFrame([
        {"candidate_id": "matrix-a"},
        {"candidate_id": "matrix-b"},
    ])
    manifest = review_sed.build_sed_fetch_manifest(candidates, sources=["payload", "catwise"])
    complete, errors = review_sed.validate_sed_fetch_manifest(
        manifest,
        candidates,
        sources=["payload", "catwise"],
    )

    assert not complete
    assert any("non-terminal statuses" in error for error in errors)

    completed = manifest.copy()
    completed["status"] = "miss"
    completed["is_complete"] = True
    complete, errors = review_sed.validate_sed_fetch_manifest(
        completed,
        candidates,
        sources=["payload", "catwise"],
    )
    assert complete
    assert errors == []

    wrong_candidates = candidates.copy()
    wrong_candidates.loc[1, "candidate_id"] = "matrix-c"
    complete, errors = review_sed.validate_sed_fetch_manifest(
        completed,
        wrong_candidates,
        sources=["payload", "catwise"],
    )
    assert not complete
    assert any("candidate set mismatch" in error for error in errors)


def test_sed_cache_preserves_canonical_multi_instrument_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    calls = 0

    def fetcher(_frame: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        rows = []
        for measurement_id, instrument, band, response_id in (
            ("sedm-c4d", "c4d", "g", "CTIO/DECam.g"),
            ("sedm-ksb", "ksb", "g", "BOK/BASS.g"),
        ):
            row = {column: None for column in CANONICAL_SED_COLUMNS}
            row.update({
                "candidate_id": "cand-cache-nsc",
                "source": "NOIRLab NSC DR2",
                "band": band,
                "measurement_id": measurement_id,
                "instrument": instrument,
                "svo_filter_id": response_id,
                "passband_fidelity": "exact",
            })
            rows.append(row)
        return pd.DataFrame(rows, columns=CANONICAL_SED_COLUMNS)

    candidates = pd.DataFrame([{
        "candidate_id": "cand-cache-nsc", "ra_deg": 10.0, "dec_deg": 20.0,
    }])
    first = review_sed._fetch_sed_source_with_cache("nsc", fetcher, candidates)
    second = review_sed._fetch_sed_source_with_cache("nsc", fetcher, candidates)

    assert calls == 1
    assert len(first) == len(second) == 2
    assert set(second["measurement_id"]) == {"sedm-c4d", "sedm-ksb"}
    assert set(second["instrument"]) == {"c4d", "ksb"}
    assert set(second["svo_filter_id"]) == {"CTIO/DECam.g", "BOK/BASS.g"}


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


def test_gaia_gspc_service_failure_is_retryable(monkeypatch) -> None:
    class FakeTapService:
        def __init__(self, _url: str) -> None:
            pass

        def search(self, _query: str):
            raise RuntimeError("temporary Gaia outage")

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)
    rows = query_gaia_gspc_photometry(pd.DataFrame([{
        "candidate_id": "gspc-error", "gaia_id": "123",
    }]))

    assert rows.empty
    assert review_sed._fetch_statuses(rows) == {"gspc-error": "error"}


def test_gaia_xp_adapter_retains_sampled_spectrum_as_correlated_rows(monkeypatch) -> None:
    gaia_module = pytest.importorskip("astroquery.gaia")
    table = Table({
        "source_id": [123, 123],
        "wavelength": [400.0, 500.0] * u.nm,
        "flux": [1.0e-16, 2.0e-16] * (u.W / u.m**2 / u.nm),
        "flux_error": [1.0e-17, 2.0e-17] * (u.W / u.m**2 / u.nm),
    })
    product = types.SimpleNamespace(to_table=lambda: table)

    def fake_load_data(*, ids, retrieval_type, data_release):
        assert ids == [123]
        assert retrieval_type == "XP_SAMPLED"
        assert data_release == "Gaia DR3"
        return {"XP_SAMPLED-Gaia DR3 123.xml": [product]}

    monkeypatch.setattr(gaia_module.Gaia, "load_data", staticmethod(fake_load_data))

    rows = query_gaia_xp_sampled(pd.DataFrame([{
        "candidate_id": "cand-xp",
        "gaia_id": "123",
        "distance_gspphot": 1000.0,
    }]))

    assert len(rows) == 2
    assert set(rows["source"]) == {"Gaia XP"}
    assert np.allclose(rows["lambda_eff_angstrom"], [4000.0, 5000.0])
    assert np.allclose(rows["flux_lambda"], [1.0e-14, 2.0e-14])
    assert rows["quality_flags"].str.contains("correlated_spectrum", regex=False).all()
    assert np.isfinite(rows["lambda_l_lambda"]).all()


def test_gaia_xp_service_failure_is_retryable(monkeypatch) -> None:
    gaia_module = pytest.importorskip("astroquery.gaia")

    def fail_load_data(**_kwargs):
        raise RuntimeError("temporary Gaia outage")

    monkeypatch.setattr(gaia_module.Gaia, "load_data", staticmethod(fail_load_data))
    rows = query_gaia_xp_sampled(pd.DataFrame([{
        "candidate_id": "xp-error", "gaia_id": "123",
    }]))

    assert rows.empty
    assert review_sed._fetch_statuses(rows) == {"xp-error": "error"}


def test_gaia_xp_malformed_product_is_retryable(monkeypatch) -> None:
    gaia_module = pytest.importorskip("astroquery.gaia")
    malformed = types.SimpleNamespace(
        to_table=lambda: Table({"unexpected": [1.0]}),
    )

    monkeypatch.setattr(
        gaia_module.Gaia,
        "load_data",
        staticmethod(lambda **_kwargs: {"XP_SAMPLED-Gaia DR3 123.xml": [malformed]}),
    )
    rows = query_gaia_xp_sampled(pd.DataFrame([{
        "candidate_id": "xp-malformed", "gaia_id": "123",
    }]))

    assert rows.empty
    assert review_sed._fetch_statuses(rows) == {"xp-malformed": "error"}


def test_gaia_xp_uses_small_service_batches(monkeypatch) -> None:
    gaia_module = pytest.importorskip("astroquery.gaia")
    batches: list[list[int]] = []

    def fake_load_data(*, ids, **_kwargs):
        batches.append(list(ids))
        return {}

    monkeypatch.setattr(gaia_module.Gaia, "load_data", staticmethod(fake_load_data))
    candidates = pd.DataFrame([
        {"candidate_id": f"xp-{index}", "gaia_id": str(1000 + index)}
        for index in range(21)
    ])

    rows = query_gaia_xp_sampled(candidates)

    assert rows.empty
    assert [len(batch) for batch in batches] == [20, 1]


def test_decaps_service_failure_is_retryable(monkeypatch) -> None:
    class FakeTapService:
        def __init__(self, _url: str) -> None:
            pass

        def search(self, _query: str):
            raise RuntimeError("temporary NOIRLab outage")

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)
    rows = review_sed.query_decaps_photometry(pd.DataFrame([{
        "candidate_id": "decaps-error", "ra_deg": 10.0, "dec_deg": 20.0,
    }]))

    assert rows.empty
    assert review_sed._fetch_statuses(rows) == {"decaps-error": "error"}


def test_new_vizier_catalog_adapters_normalize_expected_bands(monkeypatch) -> None:
    candidates = pd.DataFrame([{
        "candidate_id": "cand-catalogs",
        "ra_deg": 10.0,
        "dec_deg": 20.0,
        "distance_gspphot": 1000.0,
    }])
    tables = {
        "II/335/galex_ais": _table_from_dicts([{
            "RAJ2000": 10.0, "DEJ2000": 20.0,
            "FUVmag": 20.1, "e_FUVmag": 0.1, "NUVmag": 19.5, "e_NUVmag": 0.05,
            "Fafl": 0, "Nafl": 0,
        }]),
        "II/365/catwise": _table_from_dicts([{
            "RAPMdeg": 10.0, "DEPMdeg": 20.0,
            "W1mproPM": 12.1, "e_W1mproPM": 0.03,
            "W2mproPM": 11.8, "e_W2mproPM": 0.04,
            "pmQual": "1N000", "ccf": "0000", "abf": "00",
        }]),
        "II/367/vhs_dr5": _table_from_dicts([{
            "RAJ2000": 10.0, "DEJ2000": 20.0,
            "Yap3": 15.0, "e_Yap3": 0.02, "Jap3": 14.7, "e_Jap3": 0.02,
            "Hap3": 14.4, "e_Hap3": 0.03, "Ksap3": 14.2, "e_Ksap3": 0.03,
            "Mclass": -1, "pStar": 0.99,
        }]),
        "II/382/viking4": _table_from_dicts([{
            "RAJ2000": 10.0, "DEJ2000": 20.0,
            "Zap3": 15.4, "e_Zap3": 0.03, "Yap3": 15.1, "e_Yap3": 0.03,
            "Jap3": 14.8, "e_Jap3": 0.03, "Hap3": 14.5, "e_Hap3": 0.03,
            "Ksap3": 14.3, "e_Ksap3": 0.03,
        }]),
        "II/339/uvotssc1": _table_from_dicts([{
            "RAJ2000": 10.0, "DEJ2000": 20.0,
            "UVW2-AB": 19.0, "e_UVW2": 0.1, "UVM2-AB": 18.9, "e_UVM2": 0.1,
            "UVW1-AB": 18.5, "e_UVW1": 0.08, "U-AB": 17.5, "e_Umag": 0.05,
            "B-AB": 17.0, "e_Bmag": 0.04, "V-AB": 16.7, "e_Vmag": 0.04,
            "Nd": 2,
        }]),
        "II/378/xmmom6s": _table_from_dicts([{
            "RAJ2000": 10.0, "DEJ2000": 20.0,
            "UVW2mAB": 19.1, "e_UVW2mAB": 0.1, "UVM2mAB": 19.0, "e_UVM2mAB": 0.1,
            "UVW1mAB": 18.6, "e_UVW1mAB": 0.08, "UmAB": 17.6, "e_UmAB": 0.05,
            "BmAB": 17.1, "e_BmAB": 0.04, "VmAB": 16.8, "e_VmAB": 0.04,
            "Nobs": 1,
        }]),
    }
    _patch_vizier_query(monkeypatch, tables)

    expected = {
        "galex": {"FUV", "NUV"},
        "catwise": {"W1", "W2"},
        "vhs": {"Y", "J", "H", "Ks"},
        "viking": {"Z", "Y", "J", "H", "Ks"},
        "swift_uvot": {"UVW2", "UVM2", "UVW1", "U", "B", "V"},
        "xmm_om": {"UVW2", "UVM2", "UVW1", "U", "B", "V"},
    }
    for key, bands in expected.items():
        rows = review_sed.query_vizier_source(candidates, key)
        assert set(rows["band"]) == bands
        assert np.isfinite(rows["flux_lambda"]).all()

    catwise = review_sed.query_vizier_source(candidates, "catwise")
    assert catwise["quality_flags"].str.contains("pmQual=1N000", regex=False).all()
    assert catwise["quality_flags"].str.contains("confusion_risk", regex=False).all()
    swift = review_sed.query_vizier_source(candidates, "swift_uvot")
    assert swift["quality_flags"].str.contains("non_simultaneous_pointed", regex=False).all()


def test_catwise_native_artifact_and_snr_flags_are_band_specific(monkeypatch) -> None:
    candidates = pd.DataFrame([{
        "candidate_id": "catwise-quality", "ra": 10.0, "dec": 20.0,
    }])
    _patch_vizier_query(monkeypatch, {
        "II/365/catwise": _table_from_dicts([{
            "RAPMdeg": 10.0,
            "DEPMdeg": 20.0,
            "W1mproPM": 12.1,
            "e_W1mproPM": 0.03,
            "W2mproPM": 11.8,
            "e_W2mproPM": 0.04,
            "ccf": "D000",
            "abf": "00",
            "snrW1pm": 2.0,
            "snrW2pm": 20.0,
        }]),
    })

    rows = review_sed.query_vizier_source(candidates, "catwise").set_index("band")

    assert "catwise_artifact" in str(rows.loc["W1", "quality_flags"])
    assert "catwise_low_snr" in str(rows.loc["W1", "quality_flags"])
    assert "bad_quality" in str(rows.loc["W1", "quality_flags"])
    assert rows.loc["W1", "fit_policy"] == "diagnostic_only"
    assert "bad_quality" not in str(rows.loc["W2", "quality_flags"])


def test_vizier_bulk_xmatch_selects_nearest_and_flags_ambiguity(monkeypatch) -> None:
    calls = 0

    class FakeXMatch:
        TIMEOUT = 60

        @staticmethod
        def query(**kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["cat2"] == "vizier:II/365/catwise"
            assert kwargs["colRA1"] == "malca_ra"
            return Table(rows=[
                ("bulk-a", 0.20, 10.0, 20.0, 12.1, 0.03, 11.8, 0.04),
                ("bulk-a", 0.45, 10.0, 20.0, 13.1, 0.03, 12.8, 0.04),
                ("bulk-b", 0.10, 11.0, 21.0, 14.1, 0.03, 13.8, 0.04),
            ], names=[
                "malca_candidate_id", "angDist", "RAPMdeg", "DEPMdeg",
                "W1mproPM", "e_W1mproPM", "W2mproPM", "e_W2mproPM",
            ])

    monkeypatch.setitem(sys.modules, "astroquery.xmatch", types.SimpleNamespace(XMatch=FakeXMatch))
    monkeypatch.setattr(review_sed, "SED_BULK_XMATCH_MIN_CANDIDATES", 2)
    candidates = pd.DataFrame([
        {"candidate_id": "bulk-a", "ra": 10.0, "dec": 20.0},
        {"candidate_id": "bulk-b", "ra": 11.0, "dec": 21.0},
    ])

    rows = review_sed.query_vizier_source(candidates, "catwise")

    assert calls == 1
    assert len(rows) == 4
    assert rows.loc[rows["candidate_id"] == "bulk-a", "mag"].min() == pytest.approx(11.8)
    assert rows.loc[
        rows["candidate_id"] == "bulk-a", "quality_flags"
    ].str.contains("ambiguous_counterpart", regex=False).all()


def test_vizier_bulk_schema_failure_is_retryable(monkeypatch) -> None:
    class FakeXMatch:
        TIMEOUT = 60

        @staticmethod
        def query(**_kwargs):
            return Table(rows=[
                ("schema-a", 0.20, 10.0, 20.0),
                ("schema-b", 0.10, 11.0, 21.0),
            ], names=["malca_candidate_id", "angDist", "RAPMdeg", "DEPMdeg"])

    monkeypatch.setitem(sys.modules, "astroquery.xmatch", types.SimpleNamespace(XMatch=FakeXMatch))
    monkeypatch.setattr(review_sed, "SED_BULK_XMATCH_MIN_CANDIDATES", 2)
    candidates = pd.DataFrame([
        {"candidate_id": "schema-a", "ra": 10.0, "dec": 20.0},
        {"candidate_id": "schema-b", "ra": 11.0, "dec": 21.0},
    ])

    rows = review_sed.query_vizier_source(candidates, "catwise")

    assert rows.empty
    assert review_sed._fetch_statuses(rows) == {
        "schema-a": "error",
        "schema-b": "error",
    }


def test_ps1_large_batch_uses_one_bulk_crossmatch(monkeypatch) -> None:
    calls = 0

    class FakeXMatch:
        TIMEOUT = 60

        @staticmethod
        def query(**kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["cat2"] == "vizier:II/349/ps1"
            return Table(rows=[
                ("ps1-a", 0.15, 10.0, 20.0, 17.1, 0.02, 16.9, 0.02),
                ("ps1-b", 0.20, 11.0, 21.0, 18.1, 0.03, 17.9, 0.03),
            ], names=[
                "malca_candidate_id", "angDist", "RAJ2000", "DEJ2000",
                "gmag", "e_gmag", "rmag", "e_rmag",
            ])

    monkeypatch.setitem(sys.modules, "astroquery.xmatch", types.SimpleNamespace(XMatch=FakeXMatch))
    monkeypatch.setattr(review_sed, "SED_BULK_XMATCH_MIN_CANDIDATES", 2)
    candidates = pd.DataFrame([
        {"candidate_id": "ps1-a", "ra": 10.0, "dec": 20.0},
        {"candidate_id": "ps1-b", "ra": 11.0, "dec": 21.0},
    ])

    rows = review_sed.query_ps1_mean_photometry(candidates)

    assert calls == 1
    assert set(rows["candidate_id"]) == {"ps1-a", "ps1-b"}
    assert set(rows["band"]) == {"g", "r"}


def test_nsc_adapter_rejects_missing_value_sentinels(monkeypatch) -> None:
    captured: dict[str, str] = {}
    table = Table(rows=[(
        10.0, 20.0,
        99.99, 9.99, 17.0, 0.02, 16.8, 0.02, 16.7, 0.03,
        16.6, 0.03, 16.5, 0.04, 99.99, 9.99, 0, 0.98, 12,
    )], names=[
        "ra", "dec",
        "umag", "uerr", "gmag", "gerr", "rmag", "rerr", "imag", "ierr",
        "zmag", "zerr", "ymag", "yerr", "vrmag", "vrerr", "flags", "class_star", "ndet",
    ])

    class FakeTapService:
        def __init__(self, url: str) -> None:
            captured["url"] = url

        def search(self, query: str):
            captured["query"] = query
            return types.SimpleNamespace(to_table=lambda: table)

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)

    rows = query_nsc_photometry(pd.DataFrame([{
        "candidate_id": "cand-nsc", "ra_deg": 10.0, "dec_deg": 20.0,
    }]))

    assert captured["url"] == "https://datalab.noirlab.edu/tap"
    assert "FROM nsc_dr2.object" in captured["query"]
    assert set(rows["band"]) == {"g", "r", "i", "z", "Y"}
    assert rows["quality_flags"].str.contains("class_star=0.98", regex=False).all()
    assert rows["quality_flags"].str.contains("mixed_instrument_mean", regex=False).all()
    assert rows["quality_flags"].str.contains("instrument_provenance_required", regex=False).all()
    assert rows["quality_flags"].str.contains("nsc_exact_unavailable", regex=False).all()
    assert rows["quality_flags"].str.contains("diagnostic_only", regex=False).all()
    assert rows["svo_filter_id"].isna().all()


def test_nsc_adapter_uses_measurement_exposure_instrument_responses(monkeypatch) -> None:
    object_table = Table(
        rows=[(
            "obj-123", 10.0, 20.0,
            18.0, 0.03, 17.1, 0.02, 16.9, 0.02, 16.8, 0.03,
            16.7, 0.03, 16.6, 0.04, 16.85, 0.03, 0, 0.98, 12,
        )],
        names=[
            "objectid", "ra", "dec",
            "umag", "uerr", "gmag", "gerr", "rmag", "rerr", "imag", "ierr",
            "zmag", "zerr", "ymag", "yerr", "vrmag", "vrerr", "flags", "class_star", "ndet",
        ],
    )
    measurement_table = Table(
        rows=[
            ("m1", "obj-123", "e1", "g", 59000.0, 17.0, 0.02, 0, 0.97, "c4d", "g", 59000.0),
            ("m2", "obj-123", "e2", "g", 59100.0, 17.2, 0.02, 0, 0.96, "c4d", "g", 59100.0),
            ("m3", "obj-123", "e3", "z", 59050.0, 16.7, 0.03, 0, 0.95, "k4m", "z", 59050.0),
            ("m4", "obj-123", "e4", "r", 59060.0, 16.9, 0.03, 0, 0.94, "ksb", "r", 59060.0),
        ],
        names=[
            "measid", "objectid", "exposure", "meas_filter", "mjd", "mag_auto", "magerr_auto",
            "meas_flags", "meas_class_star", "instrument", "exposure_filter", "exposure_mjd",
        ],
    )
    queries: list[str] = []

    class FakeTapService:
        def __init__(self, _url: str) -> None:
            pass

        def search(self, query: str):
            queries.append(query)
            table = measurement_table if "FROM nsc_dr2.meas" in query else object_table
            return types.SimpleNamespace(to_table=lambda: table)

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)

    rows = query_nsc_photometry(pd.DataFrame([{
        "candidate_id": "cand-nsc-exact", "ra_deg": 10.0, "dec_deg": 20.0,
    }]))

    assert any("SELECT TOP 1000 id, ra, dec" in query for query in queries)
    assert any("JOIN nsc_dr2.exposure" in query for query in queries)
    assert set(CANONICAL_SED_COLUMNS).issubset(rows.columns)
    exact = rows[rows["instrument"].notna()].copy()
    fallback = rows[rows["instrument"].isna()].copy()
    assert set(zip(exact["instrument"], exact["band"])) == {("c4d", "g"), ("k4m", "z"), ("ksb", "r")}
    assert set(fallback["band"]) == {"u", "i", "Y", "VR"}
    assert set(fallback["passband_fidelity"]) == {"mixed_unknown"}
    assert set(fallback["fit_policy"]) == {"diagnostic_only"}
    assert set(exact["svo_filter_id"]) == {"CTIO/DECam.g", "KPNO/MOSAIC.zd_DECam", "BOK/BASS.r"}
    assert set(exact.loc[exact["instrument"].isin(["c4d", "ksb"]), "passband_fidelity"]) == {"exact"}
    assert set(exact.loc[exact["instrument"].isin(["c4d", "ksb"]), "fit_policy"]) == {"photosphere"}
    mosaic = exact[exact["instrument"] == "k4m"].iloc[0]
    assert mosaic["passband_fidelity"] == "filter_only_proxy"
    assert mosaic["fit_policy"] == "diagnostic_only"
    assert rows["measurement_id"].astype(str).str.startswith("sedm_").all()
    c4d_g = exact[(exact["instrument"] == "c4d") & (exact["band"] == "g")].iloc[0]
    assert float(c4d_g["mag"]) == pytest.approx(17.1)
    assert float(c4d_g["mag_err"]) == pytest.approx(0.14826)
    assert "nsc_aggregate_n=2" in str(c4d_g["quality_flags"])
    assert str(c4d_g["catalog_measurement_id"]).startswith("nsc-dr2-aggregate:")
    provenance = json.loads(str(c4d_g["provenance_json"]))
    assert provenance["measurement_ids"] == ["m1", "m2"]
    assert provenance["exposure_ids"] == ["e1", "e2"]


def test_nsc_adapter_falls_back_to_diagnostic_object_means_on_join_error(monkeypatch) -> None:
    object_table = Table(
        rows=[("obj-fallback", 10.0, 20.0, 17.0, 0.02, 0, 0.98, 12)],
        names=["objectid", "ra", "dec", "gmag", "gerr", "flags", "class_star", "ndet"],
    )

    class FakeTapService:
        def __init__(self, _url: str) -> None:
            pass

        def search(self, query: str):
            if "FROM nsc_dr2.meas" in query:
                raise RuntimeError("measurement schema unavailable")
            return types.SimpleNamespace(to_table=lambda: object_table)

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)
    rows = query_nsc_photometry(pd.DataFrame([{
        "candidate_id": "cand-nsc-fallback", "ra_deg": 10.0, "dec_deg": 20.0,
    }]))

    assert list(rows["band"]) == ["g"]
    assert pd.isna(rows.iloc[0]["instrument"])
    assert pd.isna(rows.iloc[0]["svo_filter_id"])
    assert rows.iloc[0]["passband_fidelity"] == "mixed_unknown"
    assert "nsc_exact_unavailable" in str(rows.iloc[0]["quality_flags"])


def test_nsc_measurement_quality_and_aggregate_provenance_are_deterministic() -> None:
    measurements = pd.DataFrame([
        {
            "measid": "m-good",
            "objectid": "obj-quality",
            "exposure": "e-good",
            "meas_filter": "g",
            "mjd": 59000.0,
            "mag_auto": 17.0,
            "magerr_auto": 0.02,
            "meas_flags": 0,
            "meas_class_star": 0.2,
            "instrument": "c4d",
        },
        {
            "measid": "m-saturated",
            "objectid": "obj-quality",
            "exposure": "e-saturated",
            "meas_filter": "g",
            "mjd": 59001.0,
            "mag_auto": 15.0,
            "magerr_auto": 0.02,
            "meas_flags": 4,
            "meas_class_star": 0.99,
            "instrument": "c4d",
        },
        {
            "measid": "m-invalid-error",
            "objectid": "obj-quality",
            "exposure": "e-invalid-error",
            "meas_filter": "g",
            "mjd": 59002.0,
            "mag_auto": 16.0,
            "magerr_auto": -1.0,
            "meas_flags": 0,
            "meas_class_star": 0.99,
            "instrument": "c4d",
        },
    ])

    first = review_sed._nsc_measurement_aggregate_rows(
        measurements,
        candidate_id="cand-quality",
        object_id="obj-quality",
        distance_pc=None,
        sep_arcsec=0.2,
    )[0]
    second = review_sed._nsc_measurement_aggregate_rows(
        measurements.iloc[::-1],
        candidate_id="cand-quality",
        object_id="obj-quality",
        distance_pc=None,
        sep_arcsec=0.2,
    )[0]

    assert first["measurement_id"] == second["measurement_id"]
    assert first["provenance_json"] == second["provenance_json"]
    assert float(first["mag"]) == pytest.approx(17.0)
    assert first["fit_policy"] == "diagnostic_only"
    assert "nsc_rejected_quality_n=2" in str(first["quality_flags"])
    assert "nsc_nonstellar_morphology" in str(first["quality_flags"])
    assert "bad_quality" in str(first["quality_flags"])
    provenance = json.loads(str(first["provenance_json"]))
    assert provenance["measurement_ids"] == ["m-good", "m-invalid-error", "m-saturated"]
    assert provenance["accepted_measurement_ids"] == ["m-good"]
    assert provenance["rejected_measurement_ids"] == ["m-invalid-error", "m-saturated"]


def test_nsc_adapter_rejects_matches_outside_requested_radius(monkeypatch) -> None:
    object_table = Table(
        rows=[("obj-far", 10.001, 20.0, 17.0, 0.02, 0, 0.98, 12)],
        names=["objectid", "ra", "dec", "gmag", "gerr", "flags", "class_star", "ndet"],
    )

    class FakeTapService:
        def __init__(self, _url: str) -> None:
            pass

        def search(self, _query: str):
            return types.SimpleNamespace(to_table=lambda: object_table)

    fake_pyvo = types.SimpleNamespace(dal=types.SimpleNamespace(TAPService=FakeTapService))
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)

    rows = query_nsc_photometry(pd.DataFrame([{
        "candidate_id": "cand-nsc-far", "ra_deg": 10.0, "dec_deg": 20.0,
    }]))

    assert rows.empty
    assert rows.attrs[review_sed.SED_FETCH_STATUS_ATTR]["cand-nsc-far"] == "miss"


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
    assert rows.loc[0, "mag_system"] == "Jy"
    assert pd.isna(rows.loc[0, "mag"])
    assert rows.loc[0, "native_flux_unit"] == "Jy"
    assert rows.loc[0, "plot_lambda_kind"] == "mission_reference"
    assert "confusion_risk" in rows.loc[0, "quality_flags"]


def test_canonical_measurement_uses_response_zero_point_and_explicit_wavelength() -> None:
    rows = prepare_canonical_sed_measurements(
        pd.DataFrame([{
            "candidate_id": "cand-canonical",
            "source": "Gaia DR3",
            "band": "G",
            "mag": 10.0,
            "mag_system": "Vega",
            "lambda_pivot_angstrom": 6217.59,
        }]),
        candidate_id="cand-canonical",
        response_zero_points_jy={"GAIA/GAIA3.G": 3228.75},
    )

    expected = 3228.75 * 10.0 ** -4
    assert math.isclose(float(rows.loc[0, "flux_nu_jy"]), expected, rel_tol=1.0e-12)
    assert rows.loc[0, "calibration_source"] == "response_calibration"
    assert rows.loc[0, "plot_lambda_kind"] == "response_pivot"
    assert float(rows.loc[0, "lambda_eff_angstrom"]) == pytest.approx(6217.59, abs=1.0e-3)
    assert math.isclose(
        float(rows.loc[0, "lambda_eff_angstrom"]),
        float(rows.loc[0, "plot_lambda_angstrom"]),
    )


def test_current_response_pivot_replaces_stale_unversioned_input_pivot(monkeypatch) -> None:
    monkeypatch.setattr(
        review_sed,
        "_cached_bandpass_response_metadata",
        lambda _bandpass: {
            "lambda_pivot_angstrom": 7000.0,
            "plot_lambda_angstrom": 7000.0,
            "plot_lambda_kind": "response_pivot",
            "response_hash": "response-current",
            "response_zero_point_jy": 3228.75,
        },
    )
    rows = prepare_canonical_sed_measurements(
        pd.DataFrame(
            [
                {
                    "candidate_id": "cand-stale-pivot",
                    "source": "Gaia DR3",
                    "band": "G",
                    "mag": 10.0,
                    "mag_system": "Vega",
                    "lambda_pivot_angstrom": 6000.0,
                    "response_hash": "response-stale",
                }
            ]
        ),
        candidate_id="cand-stale-pivot",
    )

    assert float(rows.loc[0, "lambda_pivot_angstrom"]) == pytest.approx(7000.0)
    assert float(rows.loc[0, "plot_lambda_angstrom"]) == pytest.approx(7000.0)
    assert rows.loc[0, "response_hash"] == "response-current"


def test_shared_single_row_preparation_exposes_observed_flux_aliases() -> None:
    response = types.SimpleNamespace(
        filter_id="GAIA/GAIA3.G",
        zero_point_jy=3228.75,
        wavelength_ref_angstrom=6217.59,
    )
    row = prepare_sed_measurement_row(
        {"source": "Gaia DR3", "band": "G", "mag": 10.0},
        candidate_id="cand-shared",
        response=response,
    )

    assert row is not None
    assert row["observed_flux_nu_jy"] == pytest.approx(row["flux_nu_jy"])
    assert row["observed_flux_nu_jy_err"] == row["flux_nu_jy_err"]
    assert row["plot_lambda_angstrom"] == pytest.approx(row["lambda_eff_angstrom"])
    assert row["lambda_reference_angstrom"] == pytest.approx(6217.59)


def test_canonical_measurement_accepts_flux_lambda_only_inputs() -> None:
    wavelength = 4810.0
    flux_lambda = 2.5e-16
    rows = prepare_canonical_sed_measurements(pd.DataFrame([{
        "candidate_id": "cand-flam",
        "source": "Pan-STARRS",
        "band": "g",
        "mag": np.nan,
        "flux_nu_jy": np.nan,
        "flux_lambda": flux_lambda,
        "flux_lambda_err": 0.1 * flux_lambda,
        "lambda_eff_angstrom": wavelength,
    }]))

    expected_fnu = (flux_lambda * u.erg / u.s / u.cm**2 / u.AA).to_value(
        u.Jy,
        equivalencies=u.spectral_density(wavelength * u.AA),
    )
    assert len(rows) == 1
    assert float(rows.loc[0, "observed_flux_nu_jy"]) == pytest.approx(expected_fnu)
    assert float(rows.loc[0, "observed_flux_nu_jy_err"]) == pytest.approx(0.1 * expected_fnu)


def test_canonical_identity_preserves_multi_epoch_measurements() -> None:
    external = pd.DataFrame([
        {
            "candidate_id": "cand-epochs",
            "source": "Pan-STARRS",
            "band": "g",
            "mag": 16.0,
            "mag_err": 0.02,
            "catalog_release": "DR2",
            "source_object_id": "obj-epochs",
            "catalog_measurement_id": "visit-1",
            "instrument": "GPC1",
            "exposure_id": "exp-1",
            "epoch_mjd": 59000.0,
        },
        {
            "candidate_id": "cand-epochs",
            "source": "Pan-STARRS",
            "band": "g",
            "mag": 16.2,
            "mag_err": 0.02,
            "release": "DR2",
            "catalog_object_id": "obj-epochs",
            "source_measurement_id": "visit-2",
            "camera": "GPC1",
            "visit_id": "exp-2",
            "observation_mjd": 59100.0,
        },
    ])

    canonical = prepare_canonical_sed_measurements(
        external,
        candidate_id="cand-epochs",
    )
    combined = build_sed_dataframe(
        {"candidate_id": "cand-epochs"},
        external_rows=external,
    )

    assert len(canonical) == 2
    assert canonical["measurement_id"].nunique() == 2
    assert list(canonical["exposure_id"]) == ["exp-1", "exp-2"]
    assert list(canonical["catalog_measurement_id"]) == ["visit-1", "visit-2"]
    assert len(combined) == 2


def test_direct_jy_plot_wavelength_ignores_svo_reference() -> None:
    response = types.SimpleNamespace(
        filter_id="AKARI/IRC.S9W",
        zero_point_jy=123.0,
        wavelength_ref_angstrom=99999.0,
        response_hash="response-hash",
    )
    row = prepare_sed_measurement_row(
        {"source": "AKARI", "band": "S9W", "flux_nu_jy": 2.0},
        candidate_id="cand-akari-wave",
        response=response,
    )

    assert row is not None
    assert row["plot_lambda_angstrom"] == pytest.approx(90000.0)
    assert row["lambda_reference_angstrom"] == pytest.approx(90000.0)


def test_plot_substitution_requires_matching_measurement_normalization_identity() -> None:
    observation = {
        "candidate_id": "cand-identity",
        "source": "Pan-STARRS",
        "band": "g",
        "mag": 16.0,
        "mag_system": "AB",
        "measurement_id": "sedm-current",
        "normalization_version": "sed-measurement-v3",
        "normalization_hash": "normalization-current",
        "observed_flux_nu_jy": 1.0,
        "observed_flux_nu_jy_err": 0.1,
        "plot_lambda_angstrom": 4810.0,
    }
    point = {
        **observation,
        "observed_flux_nu_jy": 2.0,
        "observed_flux_nu_jy_err": 0.2,
    }
    _, matched_rows, matched_warnings = build_sed_figure(
        {"candidate_id": "cand-identity"},
        external_rows=pd.DataFrame([observation]),
        model_point_rows=pd.DataFrame([point]),
    )
    assert float(matched_rows.iloc[0]["observed_flux_nu_jy"]) == pytest.approx(2.0)
    assert not any("do not match" in warning for warning in matched_warnings)

    stale_point = {**point, "normalization_hash": "normalization-stale", "observed_flux_nu_jy": 9.0}
    _, stale_rows, stale_warnings = build_sed_figure(
        {"candidate_id": "cand-identity"},
        external_rows=pd.DataFrame([observation]),
        model_point_rows=pd.DataFrame([stale_point]),
        extinction_mode="both",
    )
    observed = stale_rows[stale_rows["sed_mode"] == "Observed"].iloc[0]
    assert observed["measurement_id"] == "sedm-current"
    assert float(observed["observed_flux_nu_jy"]) == pytest.approx(1.0)
    assert any("do not match" in warning for warning in stale_warnings)

    _, corrected_rows, corrected_warnings = build_sed_figure(
        {"candidate_id": "cand-identity"},
        external_rows=pd.DataFrame([observation]),
        model_point_rows=pd.DataFrame([stale_point]),
        extinction_mode="corrected",
    )
    assert corrected_rows.iloc[0]["sed_mode"] == "ISM-corrected"
    assert any("do not match" in warning for warning in corrected_warnings)

    removed_input = {
        **point,
        "measurement_id": "sedm-removed",
        "source": "SDSS",
        "band": "r",
    }
    _, removed_rows, removed_warnings = build_sed_figure(
        {"candidate_id": "cand-identity"},
        external_rows=pd.DataFrame([observation]),
        model_point_rows=pd.DataFrame([point, removed_input]),
    )
    assert float(removed_rows.iloc[0]["observed_flux_nu_jy"]) == pytest.approx(1.0)
    assert any("do not match" in warning for warning in removed_warnings)

    _, empty_rows, empty_warnings = build_sed_figure(
        {"candidate_id": "cand-identity-empty"},
        model_point_rows=pd.DataFrame([removed_input]),
    )
    assert empty_rows.empty
    assert any("do not match" in warning for warning in empty_warnings)


def test_corrected_fit_points_use_exact_bandpass_model_ratio_not_payload_av() -> None:
    payload = {"candidate_id": "cand-fit-ratio", "A_v_3d": 9.0}
    observation = {
        "candidate_id": payload["candidate_id"],
        "source": "Pan-STARRS",
        "band": "g",
        "mag": 16.0,
        "mag_system": "AB",
        "measurement_id": "sedm-fit-ratio",
        "normalization_version": "sed-measurement-v5-bandpass:ratio-test",
        "normalization_hash": "normalization-ratio-test",
        "observed_flux_nu_jy": 2.0,
        "observed_flux_nu_jy_err": 0.2,
        "plot_lambda_angstrom": 4810.0,
    }
    point = {column: None for column in SED_MODEL_POINT_COLUMNS}
    point.update(
        {
            **observation,
            "used": 1,
            "model_flux_nu_jy": 1.0,
            "model_flux_nu_jy_intrinsic": 4.0,
            "model_flux_lambda": 1.0e-15,
            "model_flux_lambda_intrinsic": 4.0e-15,
        }
    )
    fit = {
        "candidate_id": payload["candidate_id"],
        "fit_version": "ck04-bandpass-v5",
        "status": "ok",
        "av_fit": 1.0,
    }
    curve = {
        "candidate_id": payload["candidate_id"],
        "wavelength_angstrom": 4810.0,
        "flux_lambda_observed": 1.0e-15,
        "flux_lambda_intrinsic": 4.0e-15,
        "teff_k": 5750.0,
    }

    figure, corrected, warnings = build_sed_figure(
        payload,
        external_rows=pd.DataFrame([observation]),
        model_point_rows=pd.DataFrame([point]),
        model_fit_rows=pd.DataFrame([fit]),
        model_curve_rows=pd.DataFrame([curve]),
        extinction_mode="corrected",
    )
    assert float(corrected.loc[0, "observed_flux_nu_jy"]) == pytest.approx(2.0)
    assert float(corrected.loc[0, "flux_nu_jy"]) == pytest.approx(8.0)
    assert not any("Intrinsic CK comparison is hidden" in warning for warning in warnings)
    assert any("intrinsic fit" in str(trace.name) for trace in figure.data)

    unavailable_point = {**point, "model_flux_nu_jy_intrinsic": None}
    unavailable_figure, _, unavailable_warnings = build_sed_figure(
        payload,
        external_rows=pd.DataFrame([observation]),
        model_point_rows=pd.DataFrame([unavailable_point]),
        model_fit_rows=pd.DataFrame([fit]),
        model_curve_rows=pd.DataFrame([curve]),
        extinction_mode="corrected",
    )
    assert any("Intrinsic CK comparison is hidden" in warning for warning in unavailable_warnings)
    assert not any("intrinsic fit" in str(trace.name) for trace in unavailable_figure.data)


def test_ck_predicted_markers_share_canonical_observed_plot_wavelengths() -> None:
    observations = prepare_canonical_sed_measurements(
        pd.DataFrame([
            {
                "candidate_id": "cand-marker-x",
                "source": "AKARI",
                "band": "S9W",
                "flux_nu_jy": 2.0,
                "flux_nu_jy_err": 0.2,
                "lambda_eff_angstrom": 99999.0,
            },
            {
                "candidate_id": "cand-marker-x",
                "source": "Gaia DR3",
                "band": "G",
                "mag": 12.0,
                "mag_err": 0.01,
                "lambda_eff_angstrom": 9999.0,
                "lambda_pivot_angstrom": 6217.59,
            },
        ]),
        candidate_id="cand-marker-x",
    )
    observations["normalization_hash"] = ["hash-akari", "hash-gaia"]
    points = observations.copy()
    points["lambda_eff_angstrom"] = [99999.0, 9999.0]
    points["model_flux_lambda"] = pd.to_numeric(
        points["flux_lambda"], errors="coerce"
    ) * 1.1
    points["used"] = 1
    for column in SED_MODEL_POINT_COLUMNS:
        if column not in points.columns:
            points[column] = None
    fit = pd.DataFrame([
        {
            "candidate_id": "cand-marker-x",
            "fit_version": SED_MODEL_FIT_VERSION,
            "status": "ok",
        }
    ])

    fig, plotted_rows, warnings = build_sed_figure(
        {"candidate_id": "cand-marker-x"},
        external_rows=observations,
        model_fit_rows=fit,
        model_point_rows=points,
    )

    assert not any("do not match" in warning for warning in warnings)
    observed_x = {
        float(value)
        for trace in fig.data
        if str(trace.name) in {"AKARI", "Gaia DR3"}
        for value in trace.x
    }
    predicted_trace = next(
        trace for trace in fig.data if str(trace.name) == "CK synthetic fitted bands"
    )
    predicted_x = {float(value) for value in predicted_trace.x}
    expected = sorted([90000.0, 6217.59])
    assert sorted(observed_x) == pytest.approx(expected)
    assert sorted(predicted_x) == pytest.approx(expected)
    assert sorted(pd.to_numeric(plotted_rows["plot_lambda_angstrom"])) == pytest.approx(expected)


def test_ukidss_j_uses_epoch_specific_aliases(monkeypatch) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-ukidss", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([
        {
            "RAJ2000": 10.0,
            "DEJ2000": 20.0,
            "Jmag1": 16.1,
            "e_Jmag1": 0.03,
            "Jmag2": np.nan,
            "e_Jmag2": np.nan,
        }
    ])
    _patch_vizier_query(monkeypatch, {"II/319/las9": table})

    rows = review_sed.query_vizier_source(candidates, "ukidss")

    j_row = rows.loc[rows["band"] == "J"].iloc[0]
    assert math.isclose(float(j_row["mag"]), 16.1)
    assert math.isclose(float(j_row["mag_err"]), 0.03)


def test_ukidss_j_falls_back_to_second_epoch(monkeypatch) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-ukidss-2", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([
        {
            "RAJ2000": 10.0,
            "DEJ2000": 20.0,
            "Jmag1": np.nan,
            "e_Jmag1": np.nan,
            "Jmag2": 16.4,
            "e_Jmag2": 0.05,
        }
    ])
    _patch_vizier_query(monkeypatch, {"II/319/las9": table})

    rows = review_sed.query_vizier_source(candidates, "ukidss")

    j_row = rows.loc[rows["band"] == "J"].iloc[0]
    assert math.isclose(float(j_row["mag"]), 16.4)
    assert math.isclose(float(j_row["mag_err"]), 0.05)


def test_des_vizier_source_prefers_psf_columns(monkeypatch) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-des", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([
        {
            "RA_ICRS": 10.0,
            "DE_ICRS": 20.0,
            "gmagPSF": 18.2,
            "e_gmagPSF": 0.02,
            "gmag": 19.9,
            "e_gmag": 0.3,
        }
    ])
    _patch_vizier_query(monkeypatch, {"II/371/des_dr2": table})

    rows = review_sed.query_des_photometry(candidates)

    g_row = rows.loc[rows["band"] == "g"].iloc[0]
    assert math.isclose(float(g_row["mag"]), 18.2)
    assert math.isclose(float(g_row["mag_err"]), 0.02)


def test_direct_irsa_spitzer_provenance_is_json_and_cache_writable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates = pd.DataFrame([{
        "candidate_id": "cand-spitzer",
        "ra_deg": 10.0,
        "dec_deg": 20.0,
    }])
    table = pd.DataFrame([{
        "ra": 10.0,
        "dec": 20.0,
        "objid": "SEIP-test-source",
        "i1_f_ap1": 1500.0,
        "i1_df_ap1": 120.0,
        "i1_fluxtype": 1,
        "i1_fluxflag": 0,
        "i1_softsatflag": 0,
    }])
    monkeypatch.setattr(
        review_sed,
        "_irsa_query_region_frame",
        lambda *args, **kwargs: (table, 10.0, 20.0, "test_propagation"),
    )
    monkeypatch.setattr(review_sed, "_sleep_after_sed_request", lambda: None)

    rows = review_sed.query_irsa_spitzer_photometry(candidates)

    assert len(rows) == 1
    provenance = json.loads(rows.iloc[0]["provenance_json"])
    assert provenance["catalog"] == "slphotdr4"
    assert provenance["coordinate_method"] == "test_propagation"
    assert rows.iloc[0]["catalog_release"] == "slphotdr4"

    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    cache_rows = review_sed._cache_rows_for_sed_result(
        "spitzer",
        {"cand-spitzer"},
        rows,
        status_by_candidate={"cand-spitzer": "hit"},
    )
    assert review_sed._write_sed_source_cache("spitzer", cache_rows)
    cached = pd.read_parquet(tmp_path / "spitzer.parquet")
    cached_provenance = cached.loc[
        cached["_cache_status"].astype(str) == "hit",
        "provenance_json",
    ].iloc[0]
    assert json.loads(cached_provenance)["catalog"] == "slphotdr4"


def test_direct_irsa_allwise_is_canonical_and_payload_is_not_a_fallback(
    monkeypatch,
) -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "cand-allwise",
                "ra_deg": 10.0,
                "dec_deg": 20.0,
                "w1": 12.1,
                "w1_err": 0.03,
            }
        ]
    )
    table = pd.DataFrame(
        [
            {
                "ra": 10.0,
                "dec": 20.0,
                "designation": "J004000.00+200000.0",
                "w1mpro": 12.0,
                "w1sigmpro": 0.02,
                "ph_qual": "A---",
                "cc_flags": "0000",
                "ext_flg": 0,
                "w1snr": 50.0,
                "w1rchi2": 1.0,
                "w1sat": 0,
            }
        ]
    )
    monkeypatch.setattr(
        review_sed,
        "_irsa_query_region_frame",
        lambda *args, **kwargs: (table, 10.0, 20.0, "test_propagation"),
    )
    monkeypatch.setattr(review_sed, "_sleep_after_sed_request", lambda: None)

    rows = review_sed.query_irsa_allwise_photometry(candidates)
    payload_rows = review_sed.rows_from_candidate_frame(candidates)

    assert rows["band"].tolist() == ["W1"]
    assert rows.iloc[0]["catalog_release"] == "allwise_p3as_psd"
    assert rows.iloc[0]["mag"] == pytest.approx(12.0)
    assert "AllWISE" not in set(payload_rows.get("source", pd.Series(dtype=str)))
    assert review_sed.CATALOG_FETCHERS["allwise"] is review_sed.query_irsa_allwise_photometry
    assert review_sed.CATALOG_FETCHERS["spitzer"] is review_sed.query_irsa_spitzer_photometry


def test_sed_cache_signature_invalidates_changed_astrometry() -> None:
    candidate = pd.Series(
        {
            "candidate_id": "signature-candidate",
            "ra_deg": 10.0,
            "dec_deg": 20.0,
            "pmra": 1.0,
            "pmdec": 2.0,
            "ref_epoch": 2016.0,
        }
    )
    original_hash = review_sed._candidate_astrometry_hash(candidate)
    cache = review_sed._cache_rows_for_sed_result(
        "allwise",
        {"signature-candidate"},
        pd.DataFrame(columns=review_sed.CANONICAL_SED_COLUMNS),
        status_by_candidate={"signature-candidate": "covered_no_detection"},
        astrometry_hashes={"signature-candidate": original_hash},
    )
    assert review_sed._cache_signature_mask(
        cache,
        "allwise",
        {"signature-candidate": original_hash},
    ).all()

    moved = candidate.copy()
    moved["ra_deg"] = 10.001
    moved_hash = review_sed._candidate_astrometry_hash(moved)
    assert not review_sed._cache_signature_mask(
        cache,
        "allwise",
        {"signature-candidate": moved_hash},
    ).any()


def test_sed_cache_read_normalizes_legacy_struct_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(review_sed, "SED_CACHE_DIR", tmp_path)
    pd.DataFrame([{
        "_cache_candidate_id": "cand-legacy-provenance",
        "_cache_status": "hit",
        "candidate_id": "cand-legacy-provenance",
        "provenance_json": {"quality": {"flux_quality": 3}},
    }]).to_parquet(tmp_path / "akari.parquet", index=False)

    cached = review_sed._read_sed_source_cache("akari")

    provenance_text = cached.iloc[0]["provenance_json"]
    assert isinstance(provenance_text, str)
    assert json.loads(provenance_text) == {"quality": {"flux_quality": 3}}


def test_sed_json_normalizer_does_not_copy_scalar_only_frame() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["cand-null", "cand-json"],
        "provenance_json": [None, '{"quality":{}}'],
    })

    normalized = review_sed._normalize_sed_json_text_columns(frame)

    assert normalized is frame


def test_akari_irc_uses_flux_columns_not_flags(monkeypatch) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-akari", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([
        {
            "S09": 0.1599,
            "e_S09": 0.0444,
            "S18": 0.25,
            "e_S18": 0.05,
            "f09": "9",
            "f18": "8",
        }
    ])
    _patch_vizier_query(monkeypatch, {"II/297/irc": table, "II/298/fis": Table()})

    rows = review_sed.query_flux_catalog_source(candidates, "akari")
    by_band = rows.set_index("band")

    assert math.isclose(float(by_band.loc["S9W", "flux_nu_jy"]), 0.1599)
    assert math.isclose(float(by_band.loc["S9W", "flux_nu_jy_err"]), 0.0444)
    assert not math.isclose(float(by_band.loc["S9W", "flux_nu_jy"]), 9.0)
    assert math.isclose(float(by_band.loc["L18W", "flux_nu_jy"]), 0.25)


def test_flux_catalog_explicitly_selects_nearest_match_and_preserves_ambiguity(
    monkeypatch,
) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-nearest", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([
        {
            "_r": 3.2,
            "objName": "AKARI-farther",
            "S09": 9.0,
            "e_S09": 0.9,
            "S18": 8.0,
            "e_S18": 0.8,
            "q_S09": 3,
            "q_S18": 3,
        },
        {
            "_r": 0.4,
            "objName": "AKARI-nearest",
            "S09": 1.5,
            "e_S09": 0.15,
            "S18": 1.0,
            "e_S18": 0.10,
            "q_S09": 3,
            "q_S18": 3,
        },
    ])
    _patch_vizier_query(monkeypatch, {"II/297/irc": table, "II/298/fis": Table()})

    rows = review_sed.query_flux_catalog_source(candidates, "akari")

    s9w = rows.loc[rows["band"] == "S9W"].iloc[0]
    assert float(s9w["flux_nu_jy"]) == pytest.approx(1.5)
    assert float(s9w["sep_arcsec"]) == pytest.approx(0.4)
    assert s9w["source_object_id"] == "AKARI-nearest"
    assert "multiple_catalog_matches" in str(s9w["quality_flags"])
    provenance = json.loads(s9w["provenance_json"])
    assert provenance["match_count_returned"] == 2
    assert provenance["second_nearest_sep_arcsec"] == pytest.approx(3.2)


def test_iras_flux_quality_distinguishes_upper_limits_and_moderate_detections(
    monkeypatch,
) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-iras-quality", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([{
        "_r": 1.25,
        "IRAS": "IRAS-quality-source",
        "Fnu_12": 2.0,
        "e_Fnu_12": 10.0,
        "q_Fnu_12": 1,
        "Fnu_25": 3.0,
        "e_Fnu_25": 20.0,
        "q_Fnu_25": 2,
        "Disc": "1",
        "Confuse": "2",
        "HSDFlag": "0",
        "SES1_12": 1,
        "SES2_12": 0,
        "CC_12": "A",
    }])
    _patch_vizier_query(monkeypatch, {"II/125/main": table})

    rows = review_sed.query_flux_catalog_source(candidates, "iras").set_index("band")

    assert int(rows.loc["12", "is_upper_limit"]) == 1
    assert int(rows.loc["25", "is_upper_limit"]) == 0
    assert float(rows.loc["12", "flux_nu_jy_err"]) == pytest.approx(0.2)
    assert float(rows.loc["25", "flux_nu_jy_err"]) == pytest.approx(0.6)
    assert "iras_upper_limit" in str(rows.loc["12", "quality_flags"])
    assert "iras_moderate_quality" in str(rows.loc["25", "quality_flags"])
    assert "iras_discrepant_flux" in str(rows.loc["12", "quality_flags"])
    assert "iras_nearby_seconds_confirmed_extension" in str(rows.loc["12", "quality_flags"])
    assert rows.loc["12", "source_object_id"] == "IRAS-quality-source"


def test_akari_low_quality_is_diagnostic_but_never_reinterpreted_as_upper_limit(
    monkeypatch,
) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-akari-quality", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([{
        "_r": 0.8,
        "objName": "AKARI-quality-source",
        "S09": 0.5,
        "e_S09": 0.05,
        "q_S09": 1,
        "S18": 0.8,
        "e_S18": 0.08,
        "q_S18": 2,
    }])
    _patch_vizier_query(monkeypatch, {"II/297/irc": table, "II/298/fis": Table()})

    rows = review_sed.query_flux_catalog_source(candidates, "akari").set_index("band")

    assert int(rows.loc["S9W", "is_upper_limit"]) == 0
    assert int(rows.loc["L18W", "is_upper_limit"]) == 0
    assert "akari_source_unconfirmed" in str(rows.loc["S9W", "quality_flags"])
    assert "akari_flux_unreliable" in str(rows.loc["L18W", "quality_flags"])
    assert "bad_quality" in str(rows.loc["S9W", "quality_flags"])
    assert "bad_quality" in str(rows.loc["L18W", "quality_flags"])


def test_herschel_pacs_flux_and_noise_are_scaled_from_mjy(monkeypatch) -> None:
    candidates = pd.DataFrame([{"candidate_id": "cand-herschel", "ra_deg": 10.0, "dec_deg": 20.0}])
    table = _table_from_dicts([
        {
            "_r": 2.5,
            "Name": "HPPSC070-test-source",
            "ObsId": "1342000001",
            "Flux": 2163.374054,
            "snrnoise": 2.05664964214913,
            "rms": 91.181079,
            "Edge": 1,
            "Blend": 1,
            "Warmat": 0,
            "SSOmap": 0,
        }
    ])
    _patch_vizier_query(
        monkeypatch,
        {
            "VIII/106/hppsc070": table,
            "VIII/106/hppsc100": Table(),
            "VIII/106/hppsc160": Table(),
        },
    )

    rows = review_sed.query_flux_catalog_source(candidates, "herschel")

    pacs70 = rows.loc[rows["band"] == "PACS70"].iloc[0]
    assert math.isclose(float(pacs70["flux_nu_jy"]), 2.163374054)
    assert math.isclose(float(pacs70["flux_nu_jy_err"]), 0.00205664964214913)
    assert float(pacs70["sep_arcsec"]) == pytest.approx(2.5)
    assert pacs70["source_object_id"] == "HPPSC070-test-source"
    assert pacs70["exposure_id"] == "1342000001"
    assert "herschel_edge" in str(pacs70["quality_flags"])
    assert "herschel_blended" in str(pacs70["quality_flags"])
    assert json.loads(pacs70["provenance_json"])["quality"]["Edge"] == 1


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


def test_canonical_roundtrip_and_plot_preserve_exact_normalization(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    canonical = prepare_canonical_sed_measurements(
        pd.DataFrame([{
            "candidate_id": "cand-v3-exact",
            "source": "Gaia DR3",
            "band": "G",
            "mag": 10.0,
            "mag_err": 0.01,
            "lambda_pivot_angstrom": 6217.59,
        }]),
        candidate_id="cand-v3-exact",
        response_zero_points_jy={"GAIA/GAIA3.G": 3228.75},
    )
    expected_fnu = float(canonical.loc[0, "observed_flux_nu_jy"])

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-v3-exact", "{}", "2026-07-18T00:00:00"),
        )
        assert upsert_sed_rows(conn, canonical) == 1
        loaded = load_sed_rows(conn, "cand-v3-exact")

    assert loaded.loc[0, "normalization_version"] == CANONICAL_SED_NORMALIZATION_VERSION
    assert float(loaded.loc[0, "observed_flux_nu_jy"]) == pytest.approx(expected_fnu)
    assert float(loaded.loc[0, "plot_lambda_angstrom"]) == pytest.approx(6217.59)
    _, plotted, _ = build_sed_figure(
        {"candidate_id": "cand-v3-exact"},
        external_rows=loaded,
    )
    assert float(plotted.loc[0, "observed_flux_nu_jy"]) == pytest.approx(expected_fnu)
    assert float(plotted.loc[0, "plot_lambda_angstrom"]) == pytest.approx(6217.59)


def test_review_loads_exact_fit_normalization_and_never_mixes_refreshed_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {
        "candidate_id": "cand-fit-normalization",
        "phot_g_mean_mag": 10.0,
        "distance_gspphot": 1000.0,
    }
    cache_state = {"response_hash": "response-old", "pivot": 6000.0}

    def cached_metadata(_bandpass):
        return {
            "response_hash": cache_state["response_hash"],
            "lambda_pivot_angstrom": cache_state["pivot"],
            "plot_lambda_angstrom": cache_state["pivot"],
            "plot_lambda_kind": "response_pivot",
            "response_zero_point_jy": 3000.0,
            "calibration_id": "gaia-test-calibration",
            "calibration_hash": "calibration-old",
        }

    monkeypatch.setattr(review_sed, "_cached_bandpass_response_metadata", cached_metadata)
    native_rows = rows_from_payload(payload, candidate_id=payload["candidate_id"])
    assert len(native_rows) == 1

    with closing(db_connect(tmp_path / "review.db")) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            (payload["candidate_id"], json.dumps(payload), "2026-07-18T00:00:00+00:00"),
        )
        assert upsert_sed_rows(conn, native_rows) == 1
        measurement_id = str(
            conn.execute(
                "SELECT measurement_id FROM sed_measurements WHERE candidate_id = ?",
                (payload["candidate_id"],),
            ).fetchone()[0]
        )
        version = "sed-measurement-v5-bandpass:exact-test"
        observed_fnu = 2.0
        observed_fnu_err = 0.2
        plot_lambda = 6000.0
        normalization = {
            "measurement_id": measurement_id,
            "normalization_version": version,
            "flux_nu_jy": observed_fnu,
            "flux_nu_jy_err": observed_fnu_err,
            "flux_lambda": flux_lambda_from_flux_nu_jy(observed_fnu, plot_lambda),
            "flux_lambda_err": flux_lambda_from_flux_nu_jy(observed_fnu_err, plot_lambda),
            "lambda_pivot_angstrom": plot_lambda,
            "lambda_effective_angstrom": plot_lambda,
            "plot_lambda_angstrom": plot_lambda,
            "plot_lambda_kind": "response_pivot",
            "response_hash": "response-old",
            "calibration_hash": "calibration-old",
            "normalization_method": "fitter_bandpass_calibrated_v5",
            "provenance_json": {"contract": "exact-test"},
        }
        normalization_hash = make_sed_normalization_hash(normalization)
        normalization["normalization_hash"] = normalization_hash
        assert store_sed_normalizations(conn, normalization) == 1

        conn.execute(
            """
            INSERT INTO sed_fit_runs (
                fit_run_id, candidate_id, fit_version, measurement_set_hash,
                fit_recipe_hash, fit_run_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fitrun-exact-test",
                payload["candidate_id"],
                "ck04-bandpass-v5",
                "measurement-set-test",
                "recipe-test",
                "run-hash-test",
                "2026-07-18T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO sed_fit_inputs (
                fit_run_id, measurement_id, normalization_version, used,
                response_hash, calibration_hash, normalization_hash, input_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fitrun-exact-test",
                measurement_id,
                version,
                1,
                "response-old",
                "calibration-old",
                normalization_hash,
                "input-hash-test",
            ),
        )
        conn.execute(
            "INSERT INTO sed_model_fits (candidate_id, fit_version, fit_run_hash, fit_run_id, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                payload["candidate_id"],
                "ck04-bandpass-v5",
                "run-hash-test",
                "fitrun-exact-test",
                "ok",
            ),
        )
        conn.execute(
            """
            INSERT INTO sed_model_points (
                candidate_id, fit_version, fit_run_hash, fit_run_id,
                measurement_id, normalization_version, source, band, used,
                observed_flux_nu_jy, observed_flux_nu_jy_err,
                model_flux_nu_jy, model_flux_nu_jy_intrinsic,
                lambda_eff_angstrom, lambda_pivot_angstrom,
                plot_lambda_angstrom, plot_lambda_kind,
                response_hash, calibration_hash, normalization_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["candidate_id"],
                "ck04-bandpass-v5",
                "run-hash-test",
                "fitrun-exact-test",
                measurement_id,
                version,
                "Gaia DR3",
                "G",
                1,
                observed_fnu,
                observed_fnu_err,
                1.0,
                2.0,
                plot_lambda,
                plot_lambda,
                plot_lambda,
                "response_pivot",
                "response-old",
                "calibration-old",
                normalization_hash,
            ),
        )
        conn.commit()
        loaded = load_sed_rows(conn, payload["candidate_id"])
        model_points = load_sed_model_points(conn, payload["candidate_id"])

    assert len(loaded) == 1
    assert loaded.loc[0, "normalization_version"] == version
    assert loaded.loc[0, "normalization_hash"] == normalization_hash
    assert float(loaded.loc[0, "observed_flux_nu_jy"]) == pytest.approx(observed_fnu)

    # The payload produces the same native measurement ID, but must not hide
    # the exact stored fit normalization during deduplication.
    combined = build_sed_dataframe(payload, external_rows=loaded)
    assert len(combined) == 1
    assert combined.loc[0, "normalization_version"] == version
    assert combined.loc[0, "normalization_hash"] == normalization_hash

    _, valid_rows, valid_warnings = build_sed_figure(
        payload,
        external_rows=loaded,
        model_point_rows=model_points,
    )
    assert float(valid_rows.loc[0, "observed_flux_nu_jy"]) == pytest.approx(observed_fnu)
    assert not any("do not match" in warning for warning in valid_warnings)

    mismatched_points = model_points.copy()
    mismatched_points.loc[0, "normalization_hash"] = "normalization-mismatch"
    mismatched_points.loc[0, "observed_flux_nu_jy"] = 9.0
    _, mismatched_rows, mismatched_warnings = build_sed_figure(
        payload,
        external_rows=loaded,
        model_point_rows=mismatched_points,
    )
    assert float(mismatched_rows.loc[0, "observed_flux_nu_jy"]) == pytest.approx(observed_fnu)
    assert any("do not match" in warning for warning in mismatched_warnings)

    # A refreshed response must neither mutate a complete stored snapshot
    # while retaining its old hash nor pass exact fit substitution.
    cache_state.update(response_hash="response-new", pivot=7100.0)
    immutable = prepare_canonical_sed_measurements(
        loaded,
        payload=payload,
        candidate_id=payload["candidate_id"],
    )
    assert float(immutable.loc[0, "plot_lambda_angstrom"]) == pytest.approx(plot_lambda)
    assert immutable.loc[0, "response_hash"] == "response-old"
    assert immutable.loc[0, "normalization_hash"] == normalization_hash
    refreshed_points = model_points.copy()
    refreshed_points.loc[0, "observed_flux_nu_jy"] = 9.0
    _, refreshed_rows, refreshed_warnings = build_sed_figure(
        payload,
        external_rows=loaded,
        model_point_rows=refreshed_points,
    )
    assert float(refreshed_rows.loc[0, "observed_flux_nu_jy"]) == pytest.approx(observed_fnu)
    assert float(refreshed_rows.loc[0, "plot_lambda_angstrom"]) == pytest.approx(plot_lambda)
    assert refreshed_rows.loc[0, "response_hash"] == "response-old"
    assert any("do not match" in warning for warning in refreshed_warnings)


def test_sed_model_rows_roundtrip_review_db_and_overlay(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    fit = {col: None for col in SED_MODEL_FIT_COLUMNS}
    fit.update({
        "candidate_id": "cand-6",
        "model_family": "Castelli/Kurucz 2004",
        "fit_version": SED_MODEL_FIT_VERSION,
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
            "lambda_l_lambda_intrinsic": value,
            "lambda_l_lambda_observed": value * 0.9,
            "flux_lambda_intrinsic": value * 1.0e-45,
            "flux_lambda_observed": value * 0.9e-45,
            "teff_k": 5750.0,
            "scale": 1.0,
        }
        for wave, value in [(3500.0, 1.0e33), (5500.0, 2.0e33), (9000.0, 8.0e32)]
    ], columns=SED_MODEL_CURVE_COLUMNS)
    points = pd.DataFrame([{
        **{col: None for col in SED_MODEL_POINT_COLUMNS},
        "candidate_id": "cand-6",
        "fit_version": SED_MODEL_FIT_VERSION,
        "source": "Pan-STARRS",
        "band": "g",
        "fit_role": "photosphere",
        "used": 1,
        "exclusion_reason": "",
        "prediction_status": "ok",
        "prediction_reason": "",
        "lambda_eff_angstrom": 4810.0,
        "model_lambda_l_lambda": 1.4e33,
        "model_lambda_l_lambda_intrinsic": 1.6e33,
        "residual_sigma": 0.3,
    }], columns=SED_MODEL_POINT_COLUMNS)

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-6", "{}", "2026-05-14T00:00:00"),
        )
        assert detect_sed_model_status(conn, "cand-6") == "missing"
        n_fit, n_curve = upsert_sed_model_results(conn, pd.DataFrame([fit]), curves, points)
        assert n_fit == 1
        assert n_curve == 3
        assert detect_sed_model_status(conn, "cand-6") == "complete"
        loaded_fits = load_sed_model_fits(conn, "cand-6")
        loaded_curves = load_sed_model_curves(conn, "cand-6")
        loaded_points = load_sed_model_points(conn, "cand-6")

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
        model_point_rows=loaded_points,
        extinction_mode="corrected",
    )

    assert not any("Castelli/Kurucz intrinsic" in str(trace.name) for trace in fig.data)
    assert not any("CK synthetic fitted bands" in str(trace.name) for trace in fig.data)
    assert any("Intrinsic CK comparison is hidden" in warning for warning in warnings)
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
        model_point_rows=loaded_points,
        extinction_mode="observed",
    )

    assert any("Castelli/Kurucz observed" in str(trace.name) for trace in observed_fig.data)
    assert not any("dereddened" in warning for warning in observed_warnings)


def test_sed_model_stage_failure_is_not_marked_complete(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    sed_row = {col: None for col in SED_COLUMNS}
    sed_row.update(
        {
            "candidate_id": "cand-sed-fail",
            "source": "Pan-STARRS",
            "band": "g",
            "mag": 17.2,
            "mag_system": "AB",
            "lambda_eff_angstrom": 4810.0,
            "flux_lambda": 1.0e-16,
        }
    )

    def fail_fit(*_args, **_kwargs):
        raise RuntimeError("kurucz grid missing")

    monkeypatch.setattr("malca.enrichment.sed_model.fit_sed_models", fail_fit)
    log_lines: list[str] = []
    completed: list[str] = []

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-sed-fail", '{"candidate_id": "cand-sed-fail"}', "2026-05-14T00:00:00"),
        )
        assert upsert_sed_rows(conn, pd.DataFrame([sed_row])) == 1

        with pytest.raises(ReviewStageExecutionError, match="kurucz grid missing"):
            run_missing_stages(
                conn,
                "cand-sed-fail",
                progress_callback=log_lines.append,
                stage_complete_callback=completed.append,
                force_stages=["sed_model_fit"],
                only_force=True,
            )
        payload_json = conn.execute(
            "SELECT payload_json FROM candidates WHERE candidate_id = ?",
            ("cand-sed-fail",),
        ).fetchone()[0]

        assert detect_sed_model_status(conn, "cand-sed-fail") == "missing"

    assert completed == []
    assert '"review_stage_sed_model_fit_status": "error"' in payload_json
    assert '"review_stage_sed_model_fit_error": "kurucz grid missing"' in payload_json
    assert any("sed_model_fit failed: kurucz grid missing" in line for line in log_lines)


def test_sed_photometry_stage_requires_certified_fetch_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "review.db"
    monkeypatch.setattr(
        review_sed,
        "fetch_sed_photometry",
        lambda *_args, **_kwargs: pd.DataFrame(columns=CANONICAL_SED_COLUMNS),
    )
    completed: list[str] = []

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            (
                "cand-sed-manifest",
                '{"candidate_id":"cand-sed-manifest","ra":10.0,"dec":20.0}',
                "2026-07-22T00:00:00",
            ),
        )
        with pytest.raises(ReviewStageExecutionError, match="SED photometry incomplete"):
            run_missing_stages(
                conn,
                "cand-sed-manifest",
                stage_complete_callback=completed.append,
                force_stages=["sed_photometry"],
                only_force=True,
            )
        payload_json = conn.execute(
            "SELECT payload_json FROM candidates WHERE candidate_id = ?",
            ("cand-sed-manifest",),
        ).fetchone()[0]

    assert completed == []
    assert '"review_stage_sed_photometry_status": "error"' in payload_json


def test_sed_model_stage_uses_payload_sed_rows_when_sidecar_is_empty(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    payload = {
        "candidate_id": "cand-payload-sed",
        "apass_b": 14.1,
        "apass_v": 13.8,
        "tmass_j": 12.1,
        "tmass_h": 11.7,
        "tmass_k": 11.4,
        "w1": 10.8,
        "w2": 10.6,
        "distance_gspphot": 1000.0,
    }
    seen: dict[str, pd.DataFrame] = {}

    def fake_fit(candidates, sed_rows, **_kwargs):
        seen["sed_rows"] = pd.DataFrame(sed_rows)
        fit = {col: None for col in SED_MODEL_FIT_COLUMNS}
        fit.update(
            {
                "candidate_id": "cand-payload-sed",
                "model_family": "Castelli/Kurucz 2004",
                "fit_version": SED_MODEL_FIT_VERSION,
                "teff_k": 5000.0,
                "status": "ok",
                "warning": "",
                "n_fit_points": int(len(sed_rows)),
            }
        )
        curve = {col: None for col in SED_MODEL_CURVE_COLUMNS}
        curve.update(
            {
                "candidate_id": "cand-payload-sed",
                "model_family": "Castelli/Kurucz 2004",
                "wavelength_angstrom": 5000.0,
                "lambda_l_lambda": 1.0,
                "flux_lambda": 1.0,
                "teff_k": 5000.0,
                "scale": 1.0,
            }
        )
        point = {col: None for col in SED_MODEL_POINT_COLUMNS}
        point.update({
            "candidate_id": "cand-payload-sed",
            "fit_version": SED_MODEL_FIT_VERSION,
            "source": "APASS",
            "band": "V",
        })
        return pd.DataFrame([fit]), pd.DataFrame([curve]), pd.DataFrame([point])

    monkeypatch.setattr("malca.enrichment.sed_model.fit_sed_models", fake_fit)
    log_lines: list[str] = []

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-payload-sed", json.dumps(payload), "2026-05-14T00:00:00"),
        )

        stages = run_missing_stages(
            conn,
            "cand-payload-sed",
            progress_callback=log_lines.append,
            force_stages=["sed_model_fit"],
            only_force=True,
        )

        stored_measurements = load_sed_rows(conn, "cand-payload-sed")
        assert len(stored_measurements) == len(seen["sed_rows"])
        assert set(stored_measurements["measurement_id"]) == set(seen["sed_rows"]["measurement_id"])
        assert {"APASS", "2MASS", "AllWISE"}.issubset(set(stored_measurements["source"]))
        assert detect_sed_model_status(conn, "cand-payload-sed") == "complete"
        assert len(load_sed_model_fits(conn, "cand-payload-sed")) == 1
        assert len(load_sed_model_curves(conn, "cand-payload-sed")) == 1

    assert stages == ["sed_model_fit"]
    assert "sed_rows" in seen
    assert {"APASS", "2MASS", "AllWISE"}.issubset(set(seen["sed_rows"]["source"]))
    assert any("using 7 payload SED rows" in line for line in log_lines)


def test_non_ok_sed_model_fit_row_is_partial_not_complete(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    fit = {col: None for col in SED_MODEL_FIT_COLUMNS}
    fit.update(
        {
            "candidate_id": "cand-sed-partial",
            "model_family": "Castelli/Kurucz 2004",
            "status": "fit_failed",
            "warning": "optimizer failed",
        }
    )

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("cand-sed-partial", "{}", "2026-05-14T00:00:00"),
        )
        n_fit, n_curve = upsert_sed_model_results(conn, pd.DataFrame([fit]), pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS))

        assert n_fit == 1
        assert n_curve == 0
        assert detect_sed_model_status(conn, "cand-sed-partial") == "partial"


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
                "lambda_l_lambda_observed": value,
                "flux_lambda": value * 1.0e-45,
                "flux_lambda_observed": value * 1.0e-45,
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
                "fit_version": SED_MODEL_FIT_VERSION,
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
        extinction_mode="observed",
    )

    assert not rows.empty
    assert any("Castelli/Kurucz" in str(trace.name) for trace in fig.data)
    x0, x1 = fig.layout.xaxis.range
    assert x0 > math.log10(100.0)
    assert x1 < math.log10(1.0e6)
    assert x0 < math.log10(float(rows["lambda_eff_angstrom"].min()))
    assert x1 > math.log10(float(rows["lambda_eff_angstrom"].max()))
