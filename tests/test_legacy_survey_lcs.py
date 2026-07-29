from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from malca.enrichment import legacy_survey_lcs as legacy
from malca.enrichment import vetting
from malca.enrichment import external_lcs
from malca.enrichment.external_lcs import build_arg_parser, _source_run_flags
from malca.external_lc_manifest import (
    read_external_lc_manifest,
    upsert_external_lc_manifest_entry,
)
from malca.review.lightcurve_sources import (
    EXTERNAL_LC_SPECS,
    EXTERNAL_SOURCE_ORDER,
    normalize_external_lc_dataframe,
)


class _Response:
    def __init__(self, *, content: bytes = b"x", text: str = "", payload=None):
        self.content = content
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_superwasp_preserves_raw_and_sysrem(monkeypatch):
    matches = pd.DataFrame(
        {
            "sourceid": ["1SWASP J100000.00+200000.0"],
            "ra": [150.0],
            "dec": [20.0],
            "tile": ["1SWASP"],
            "npts": [2],
        }
    )
    table = pd.DataFrame(
        {
            "hjd": [2454000.0, 2454001.0],
            "mag2": [12.1, 12.2],
            "mag2_err": [0.03, 0.04],
            "tammag2": [12.0, 12.05],
            "tammag2_err": [0.02, 0.02],
        }
    )
    monkeypatch.setattr(legacy, "_nasa_cone_query", lambda *args, **kwargs: matches)
    monkeypatch.setattr(legacy.requests, "get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(legacy, "_read_ipac_table", lambda content: table)

    result = legacy.query_superwasp_lightcurve(150.0, 20.0)

    assert set(result["proc_type"]) == {"raw", "sysrem"}
    assert set(result["photometry_level"]) == {"raw", "minimally_detrended"}
    assert len(result) == 4
    assert result.groupby("proc_type").size().to_dict() == {"raw": 2, "sysrem": 2}


def test_kelt_download_index_keeps_orientation_and_processing(tmp_path, monkeypatch):
    script = "\n".join(
        [
            "wget -O 'KELT_N12_lc_000141_V01_east_raw_lc.tbl' "
            "'http://example.test/east_raw.tbl'",
            "wget -O 'KELT_N12_lc_000141_V01_west_raw_lc.tbl' "
            "'http://example.test/west_raw.tbl'",
            "wget -O 'KELT_N12_lc_000141_V01_west_tfa_lc.tbl' "
            "'http://example.test/west_tfa.tbl'",
        ]
    ).encode()
    archive = tmp_path / "KELT_wget.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("KELT_N12_wget.bat")
        info.size = len(script)
        bundle.addfile(info, BytesIO(script))
    monkeypatch.setattr(legacy, "_download_cached_file", lambda *args, **kwargs: archive)

    source_id = "KELT_N12_lc_000141_V01"
    matches = pd.DataFrame(
        {
            "kelt_sourceid": [source_id, source_id, source_id],
            "kelt_field": ["N12", "N12", "N12"],
            "orientation": ["east", "west", "west"],
            "proc_type": ["raw", "raw", "tfa"],
            "sep_arcsec": [1.0, 1.0, 1.0],
        }
    )
    indexed = legacy._kelt_urls_for_matches(matches, tmp_path)

    products = {
        (str(metadata["orientation"]), str(metadata["proc_type"]))
        for metadata, _ in indexed
    }
    assert products == {("east", "raw"), ("west", "raw"), ("west", "tfa")}
    assert all(url.startswith("https://") for _, url in indexed)


def test_asas3_parser_preserves_apertures_and_marks_selected():
    text = """
#ndata= 1
#dataset= 1 ; 1 F1000+20_42
#desig= 100000+2000.0
#cra= 10.000000
#cdec= 20.000000
#class= 0
#ra= 10.000010
#dec= 20.000010
#cmag_0= 11.10
#cmag_1= 11.11
#cmer_0= 0.080
#cmer_1= 0.040
#cmer_2= 0.060
#cmer_3= 0.090
#cmer_4= 0.100
#nskip_0= 2
#nskip_1= 3
  3000.0 11.0 11.1 11.2 11.3 11.4 0.1 0.1 0.1 0.1 0.1 A 42
#dataset= 2 ; unrelated
#desig= 100100+2000.0
#cra= 10.016667
#cdec= 20.000000
#cmer_0= 0.030
  3001.0 12.0 12.1 12.2 12.3 12.4 0.1 0.1 0.1 0.1 0.1 A 43
"""
    result = legacy.parse_asas3_lightcurve(text, ra=150.0, dec=20.0)

    assert len(result) == 5
    assert set(result["aperture"]) == {0, 1, 2, 3, 4}
    assert result["selected"].sum() == 1
    assert int(result.loc[result["selected"], "aperture"].iloc[0]) == 1
    assert np.allclose(result["hjd"], 2453000.0)
    assert result["mag_err"].isna().all()
    assert np.allclose(result["frame_error"], 0.1)
    assert set(result["dataset_descriptor"]) == {"1 F1000+20_42"}
    assert set(result["dataset_entry"]) == {"1"}
    assert set(result["dataset_field"]) == {"F1000+20_42"}
    assert np.allclose(result["catalog_ra_deg"], 150.0)
    assert np.allclose(result["measured_ra_deg"], 150.00015)
    selected = result[result["selected"]].iloc[0]
    assert selected["mag"] == 11.1
    assert selected["cmer_aperture"] == 0.04
    assert selected["nskip_aperture"] == 3
    assert selected["response_ndata"] == 1
    assert selected["catalog_class"] == "0"
    assert '"cmer_1":"0.040"' in selected["dataset_header_json"]
    assert bool(selected["quality_pass"])


def test_asas3_parser_retains_sentinels_and_grade_rejections():
    text = """
#dataset= 1 ; 1 F1000+20_42
#desig= 100000+2000.0
#cra= 10.000000
#cdec= 20.000000
#cmer_0= 0.080
  3000.0 99.999 12.0 29.999 12.3 12.4 0.1 0.1 0.1 0.1 0.1 D 42
"""
    result = legacy.parse_asas3_lightcurve(text, ra=150.0, dec=20.0)

    assert len(result) == 5
    assert set(result["measurement_status"]) == {
        "detection",
        "below_detection_threshold",
        "negative_aperture_flux",
    }
    assert result.loc[result["aperture"].eq(0), "mag"].isna().all()
    assert result.loc[result["aperture"].eq(0), "raw_mag"].iloc[0] == 99.999
    assert not result["grade_pass"].any()
    assert not result["quality_pass"].any()


def test_dasch_api_records_are_normalized(monkeypatch):
    query_payload = [
        "ref_text,ref_number,gsc_bin_index,ra_deg,dec_deg,num_matches",
        "N1,123,456,150.0,20.0,2",
        "N2,124,457,150.005,20.0,5",
    ]
    lc_payload = [
        (
            "date_jd,series,plate_number,magcal_magdep,magcal_magdep_rms,"
            "magcal_local_error,magcal_local_rms,limiting_mag_local,reject_flag,"
            "ra_deg,dec_deg,aflags,a2flags,bflags,b2flags,ellipticity"
        ),
        "2410000.5,a,1,12.3,,0.2,0.3,14.0,0,150.0001,20.0,128,0,0,0,0.1",
        "2410001.5,a,2,,,,,14.2,1,,,,,,,",
    ]
    responses = iter(
        [
            _Response(payload=query_payload),
            _Response(payload=lc_payload),
        ]
    )
    monkeypatch.setattr(
        legacy.requests,
        "post",
        lambda *args, **kwargs: next(responses),
    )

    result = legacy.query_dasch_lightcurve(150.0, 20.0)

    assert len(result) == 2
    assert result.loc[0, "mag"] == 12.3
    assert result.loc[0, "mag_err"] == 0.3
    assert result.loc[1, "limiting_mag"] == 14.2
    assert set(result["refcat"]) == {"apass"}
    assert result.loc[0, "magcal_local_error"] == 0.2
    assert result.loc[0, "aflags"] == 128
    assert result.loc[0, "ellipticity"] == 0.1
    assert result.loc[0, "epoch_sep_arcsec"] > 0
    assert result.loc[0, "standard_aflag_mask"] == 128
    assert bool(result.loc[0, "standard_aflag_reject"])
    assert not bool(result.loc[0, "quality_standard_pass"])
    assert result.loc[0, "n_refcat_candidates"] == 2
    assert result.loc[0, "nearest_alternative_source_id"] == "N2"
    assert "N2" in result.loc[0, "refcat_candidates_json"]


def test_fetch_wrapper_records_matched_no_coverage_and_failure(tmp_path):
    candidates = pd.DataFrame(
        {
            "candidate_id": ["matched", "empty", "failed"],
            "ra": [1.0, 2.0, 3.0],
            "dec": [4.0, 5.0, 6.0],
        }
    )

    def query(ra, dec):
        if ra == 1.0:
            return pd.DataFrame(
                {"hjd": [2450000.0, 2450002.0], "mag": [12.0, 12.1]}
            )
        if ra == 2.0:
            return pd.DataFrame(columns=["hjd", "mag"])
        raise RuntimeError("service unavailable")

    result = vetting._fetch_legacy_coordinate_lightcurves(
        candidates,
        module="Fixture LCs",
        file_prefix="fixture_lc",
        prefix="fixture_lc",
        radius_arcsec=10.0,
        query_func=query,
        summarize_func=lambda lc: vetting._summarize_legacy_mag_lc(
            lc,
            "fixture_lc",
        ),
        output_dir=tmp_path,
        workers=2,
        refresh_cache=False,
    ).set_index("candidate_id")

    assert result["fixture_lc_state"].to_dict() == {
        "matched": "matched",
        "empty": "no_coverage",
        "failed": "fetch_failed",
    }
    assert result.loc["matched", "fixture_lc_n_points"] == 2
    assert result.loc["matched", "fixture_lc_time_span_days"] == 2.0
    assert (tmp_path / "fixture_lc_matched.parquet").exists()
    assert not (tmp_path / "fixture_lc_empty.parquet").exists()
    assert not (tmp_path / "fixture_lc_failed.parquet").exists()

    statuses = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert statuses.set_index("candidate_id")["status"].to_dict() == {
        "matched": "fetched",
        "empty": "no_data",
        "failed": "error",
    }


def test_cache_only_persists_inferred_state_for_historical_status(tmp_path):
    candidates = pd.DataFrame(
        [{"candidate_id": "C1", "ra": 150.0, "dec": 20.0}]
    )
    cache_key = vetting._coord_lookup_cache_key(
        candidates,
        0,
        vetting.CRTS_MATCH_RADIUS_ARCSEC,
        "crts",
    )
    vetting._write_external_lc_status(
        tmp_path,
        [
            {
                "module": "CRTS LCs",
                "candidate_id": "C1",
                "cache_key": cache_key,
                "status": "no_data",
                "updated_unix": 1.0,
                "crts_lc_n_points": 0,
            }
        ],
    )

    result = external_lcs.rebuild_external_lc_table_from_cache(
        candidates,
        tmp_path,
        {"crts": True},
    )

    assert result.loc[0, "crts_lc_state"] == "no_coverage"
    statuses = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    latest = statuses.sort_values("updated_unix").iloc[-1]
    assert latest["status"] == "no_data"
    assert latest["crts_lc_state"] == "no_coverage"


def test_cache_only_preserves_fetch_failure_status(tmp_path):
    candidates = pd.DataFrame(
        [{"candidate_id": "C1", "ra": 150.0, "dec": 20.0}]
    )
    cache_key = vetting._coord_lookup_cache_key(
        candidates,
        0,
        vetting.CRTS_MATCH_RADIUS_ARCSEC,
        "crts",
    )
    vetting._write_external_lc_status(
        tmp_path,
        [
            {
                "module": "CRTS LCs",
                "candidate_id": "C1",
                "cache_key": cache_key,
                "status": "error",
                "updated_unix": 1.0,
                "crts_lc_n_points": 0,
            }
        ],
    )

    result = external_lcs.rebuild_external_lc_table_from_cache(
        candidates,
        tmp_path,
        {"crts": True},
    )

    assert result.loc[0, "crts_lc_state"] == "fetch_failed"
    statuses = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    latest = statuses.sort_values("updated_unix").iloc[-1]
    assert latest["status"] == "error"
    assert latest["crts_lc_state"] == "fetch_failed"


def test_cache_only_indexes_direct_fallback_file_missing_from_manifest(tmp_path):
    results_root = tmp_path / "results"
    output_dir = results_root / "external_lcs"
    output_dir.mkdir(parents=True)
    unrelated = output_dir / "dasch_lc_other.parquet"
    pd.DataFrame({"jd": [2450000.0], "mag": [12.0]}).to_parquet(
        unrelated,
        index=False,
    )
    assert upsert_external_lc_manifest_entry(
        results_root,
        candidate_id="other",
        source="dasch",
        file_prefix="dasch_lc",
        path=unrelated,
    )
    crts_path = output_dir / "crts_lc_C1.parquet"
    pd.DataFrame(
        {"mjd": [59000.0, 59001.0], "mag": [14.0, 14.1]}
    ).to_parquet(crts_path, index=False)
    candidates = pd.DataFrame(
        [{"candidate_id": "C1", "ra": 150.0, "dec": 20.0}]
    )

    result = external_lcs.rebuild_external_lc_table_from_cache(
        candidates,
        output_dir,
        {"crts": True},
        results_root=results_root,
    )

    assert result.loc[0, "crts_lc_state"] == "matched"
    manifest = read_external_lc_manifest(results_root)
    row = manifest[
        manifest["candidate_id"].eq("C1")
        & manifest["source"].eq("crts")
    ]
    assert len(row) == 1
    assert row.iloc[0]["path_relative"] == "external_lcs/crts_lc_C1.parquet"


def test_review_sources_and_asas_selected_aperture_normalization():
    for source in ("superwasp", "kelt", "nsvs", "asas3", "dasch"):
        assert source in EXTERNAL_LC_SPECS
        assert source in EXTERNAL_SOURCE_ORDER

    raw = pd.DataFrame(
        {
            "hjd": [2450000.0, 2450000.0],
            "mag": [11.0, 12.0],
            "mag_err": [0.1, 0.2],
            "selected": [False, True],
        }
    )
    normalized = normalize_external_lc_dataframe("asas3", raw)
    assert len(normalized) == 1
    assert normalized.iloc[0]["mag"] == 12.0
    assert normalized.iloc[0]["frame_error"] == 0.2
    assert pd.isna(normalized.iloc[0]["mag_err"])


def test_legacy_surveys_only_cli_selects_exact_six_sources(tmp_path):
    args = build_arg_parser().parse_args(
        [str(tmp_path / "review.db"), "--legacy-surveys-only"]
    )
    enabled = {
        source for source, should_run in _source_run_flags(args).items() if should_run
    }
    assert enabled == {"superwasp", "kelt", "nsvs", "asas3", "crts", "dasch"}


def test_legacy_surveys_only_cli_respects_individual_skip_flags(tmp_path):
    args = build_arg_parser().parse_args(
        [
            str(tmp_path / "review.db"),
            "--legacy-surveys-only",
            "--no-superwasp",
            "--no-kelt",
            "--no-nsvs",
            "--no-crts",
        ]
    )
    enabled = {
        source for source, should_run in _source_run_flags(args).items() if should_run
    }
    assert enabled == {"asas3", "dasch"}
