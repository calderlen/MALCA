from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from malca.enrich import neighbor
from malca.enrich.spectra import _normalize_spectra_long, _summarize_spectra_long
from malca.enrichment import characterize, vetting
from malca.enrichment.astrometry import angular_separation_arcsec, propagate_linear_icrs
from malca.enrichment.classify import compute_all_classifications
from malca.enrichment.external_lcs import rebuild_external_lc_table_from_cache
from malca.enrichment.multi_survey_features import compute_candidate_multi_survey_features
from malca.external_lc_manifest import read_external_lc_manifest, upsert_external_lc_manifest_entry
from malca.ltv.dust import apply_dust_flags


def test_classifier_uses_explicit_inputs_and_strongest_contaminant() -> None:
    frame = pd.DataFrame(
        [{
            "candidate_id": "c1",
            "event_duration_days": 1.0,
            "event_depth_mag": 0.01,
            "is_known_cv": True,
            "tmass_h": 12.0,
            "tmass_k": 11.9,
            "w1": 11.8,
            "w2": 11.75,
        }]
    )

    out = compute_all_classifications(frame)

    assert out.loc[0, "final_class"] == "Likely CV"
    assert out.loc[0, "classification_score"] == out.loc[0, "P_cv"]
    assert out.loc[0, "classification_duration_source"] == "event_duration_days"
    assert out.loc[0, "classification_depth_source"] == "event_depth_mag"
    assert out.loc[0, "classification_version"] == "heuristic-v2"
    assert '"duration":"event_duration_days"' in out.loc[0, "classification_input_map_json"]


def test_classifier_skip_switches_are_not_ignored() -> None:
    frame = pd.DataFrame([{"candidate_id": "c1", "is_known_cv": True}])

    out = compute_all_classifications(frame, run_eb=False, run_cv=False, run_starspot=False)

    assert pd.isna(out.loc[0, "P_cv"])
    assert out.loc[0, "cv_classifier_status"] == "disabled"
    assert out.loc[0, "final_class"] == "Unknown Dipper"


def test_classifier_does_not_treat_false_string_as_positive_catalog_evidence() -> None:
    frame = pd.DataFrame([{"candidate_id": "c1", "is_known_cv": "False"}])

    out = compute_all_classifications(frame)

    assert out.loc[0, "cv_classifier_status"] != "known_catalog_cv"
    assert "Known CV" not in out.loc[0, "cv_notes"]


def test_dust_flags_distinguish_missing_from_measured_zero() -> None:
    frame = pd.DataFrame(
        {
            "ltv_slope": [np.nan, 0.0, 0.04],
            "ltv_neowise_w1_w2_median": [np.nan, 0.0, 0.4],
            "ltv_neowise_w1_w2_slope": [np.nan, 0.0, 0.02],
        }
    )

    out = apply_dust_flags(frame)

    assert pd.isna(out.loc[0, "ltv_dust_candidate"])
    assert out.loc[0, "ltv_dust_status"] == "missing_inputs"
    assert bool(out.loc[1, "ltv_dust_candidate"]) is False
    assert out.loc[1, "ltv_dust_status"] == "ok"
    assert bool(out.loc[2, "ltv_dust_candidate"]) is True
    assert out.loc[2, "ltv_dust_trend_class"] == "redder+fainter"


def test_neighbor_summary_excludes_self_and_deduplicates_catalog_rows(monkeypatch, tmp_path: Path) -> None:
    matches = pd.DataFrame(
        [
            {"candidate_id": "c1", "catalog": "I/355/gaiadr3", "Source": "123", "sep_arcsec": 0.0},
            {"candidate_id": "c1", "catalog": "I/355/gaiadr3", "Source": "123", "sep_arcsec": 0.0},
            {"candidate_id": "c1", "catalog": "I/355/gaiadr3", "Source": "456", "sep_arcsec": 3.0},
            {"candidate_id": "c1", "catalog": "I/355/gaiadr3", "Source": "456", "sep_arcsec": 3.0},
            {"candidate_id": "c1", "catalog": "I/355/gaiadr3", "Source": "789", "sep_arcsec": 20.0},
        ]
    )
    monkeypatch.setattr(neighbor, "_query_catalog_bulk", lambda *_args, **_kwargs: matches.copy())
    targets = pd.DataFrame(
        [{"candidate_id": "c1", "source_id": "123", "ra_deg": 1.0, "dec_deg": 2.0}]
    )

    long_rows, summary = neighbor.run_neighbor_enrichment(
        targets,
        out_dir=tmp_path,
        catalogs={"gaia_dr3": "I/355/gaiadr3"},
    )

    assert len(long_rows) == 3
    assert int(summary.loc[0, "neighbor_target_match_count"]) == 1
    assert int(summary.loc[0, "neighbor_unique_count"]) == 2
    assert int(summary.loc[0, "local_density_n_15as"]) == 1
    assert float(summary.loc[0, "nearest_sep_arcsec"]) == 3.0


def test_spectrum_records_have_stable_keys_dedup_and_conflict_flags() -> None:
    rows = pd.DataFrame(
        [
            {"candidate_id": "c1", "survey": "sdss", "catalog": "sdss", "SpecObjID": "a", "source_id": "added-later", "sep_arcsec": 0.2, "z": 0.1},
            {"candidate_id": "c1", "survey": "sdss", "catalog": "sdss", "SpecObjID": "a", "sep_arcsec": 0.2, "z": 0.1},
            {"candidate_id": "c1", "survey": "desi", "catalog": "desi", "TARGETID": "b", "sep_arcsec": 0.3, "z": 0.8},
        ]
    )

    normalized = _normalize_spectra_long(rows)
    summary = _summarize_spectra_long(pd.DataFrame([{"candidate_id": "c1"}]), normalized)

    assert len(normalized) == 2
    assert normalized["spectrum_record_key"].nunique() == 2
    assert int(summary.loc[0, "spectrum_n_unique_records"]) == 2
    assert bool(summary.loc[0, "spectrum_redshift_conflict"]) is True


def test_manifest_upserts_are_locked_and_lossless(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    paths = []
    for index in range(12):
        path = root / f"tess_lc_c{index}.parquet"
        pd.DataFrame({"time": [1.0], "flux": [1.0]}).to_parquet(path, index=False)
        paths.append(path)

    def write(index: int) -> bool:
        return upsert_external_lc_manifest_entry(
            root,
            candidate_id=f"c{index}",
            source="tess",
            file_prefix="tess",
            path=paths[index],
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        assert all(executor.map(write, range(len(paths))))

    manifest = read_external_lc_manifest(root)
    assert set(manifest["candidate_id"]) == {f"c{index}" for index in range(len(paths))}


def test_neowise_identity_uses_proper_motion_and_rejects_neighbor() -> None:
    target_mjd = np.array([61040.0])  # about ten years after the Gaia reference epoch
    propagated_ra, propagated_dec, method = propagate_linear_icrs(
        10.0, 0.0, target_mjd, pmra_mas_per_year=1000.0, pmdec_mas_per_year=0.0
    )
    lc = pd.DataFrame(
        [
            {"ra": propagated_ra[0], "dec": propagated_dec[0], "mjd": target_mjd[0], "w1mpro": 12.0},
            {"ra": propagated_ra[0] + 2.0 / 3600.0, "dec": propagated_dec[0], "mjd": target_mjd[0], "w1mpro": 14.0},
        ]
    )

    matched, status = vetting._filter_neowise_candidate_identity(
        lc,
        {"ra": 10.0, "dec": 0.0, "pmra": 1000.0, "pmdec": 0.0, "ref_epoch": 2016.0},
        max_sep_arcsec=3.0,
    )

    assert method == "proper_motion_linear"
    assert status == "matched"
    assert len(matched) == 1
    assert float(matched.loc[0, "w1mpro"]) == 12.0
    assert vetting._neowise_query_radius_arcsec(
        {"pmra": 1000.0, "pmdec": 0.0, "ref_epoch": 2016.0}, 3.0
    ) > 10.0
    assert float(angular_separation_arcsec(propagated_ra, propagated_dec, propagated_ra, propagated_dec)[0]) == 0.0


def test_tess_search_selection_keeps_one_requested_target() -> None:
    class FakeSearch:
        def __init__(self, table: Table):
            self.table = table

        def __len__(self) -> int:
            return len(self.table)

        def __getitem__(self, item):
            return FakeSearch(self.table[item])

    search = FakeSearch(
        Table(
            {
                "target_name": ["TIC 101", "TIC 202", "TIC 202"],
                "distance": [0.1, 0.5, 0.5],
                "author": ["SPOC", "SPOC", "QLP"],
            }
        )
    )

    selected, target_id, separation, method = vetting._select_tess_target_search_result(
        search, {"tic_id": 202}
    )

    assert selected is not None
    assert len(selected) == 2
    assert set(selected.table["target_name"]) == {"TIC 202"}
    assert target_id == "TIC 202"
    assert separation == 0.5
    assert method == "catalog_identifier"


def test_identity_verified_cache_summaries_reject_legacy_files() -> None:
    with pytest.raises(ValueError, match="verified target identity"):
        vetting._summarize_verified_neowise_lc(
            pd.DataFrame({"mjd": [58000.0], "w1mpro": [12.0]})
        )
    with pytest.raises(ValueError, match="verified target identity"):
        vetting._summarize_verified_tess_lc(
            pd.DataFrame({"time": [1400.0], "flux": [1.0], "quality": [0], "sector": [1]})
        )


def test_normal_neowise_fetch_refreshes_legacy_unverified_cache(monkeypatch, tmp_path: Path) -> None:
    pd.DataFrame(
        {"mjd": [58000.0], "w1mpro": [12.0], "w2mpro": [11.8]}
    ).to_parquet(tmp_path / "neowise_lc_C1.parquet", index=False)
    calls: list[str] = []

    class Result:
        def to_table(self) -> Table:
            return Table(
                {
                    "ra": [10.0, 10.0],
                    "dec": [20.0, 20.0],
                    "mjd": [59000.0, 59001.0],
                    "w1mpro": [12.0, 12.2],
                    "w1sigmpro": [0.03, 0.03],
                    "w2mpro": [11.8, 11.9],
                    "w2sigmpro": [0.04, 0.04],
                    "w1snr": [20.0, 20.0],
                    "w2snr": [20.0, 20.0],
                    "qual_frame": [10, 10],
                    "qi_fact": [1.0, 1.0],
                    "cc_flags": ["0000", "0000"],
                }
            )

    def query_tap(query: str) -> Result:
        calls.append(query)
        return Result()

    monkeypatch.setattr(vetting.Irsa, "query_tap", staticmethod(query_tap))
    out = vetting.query_neowise_lightcurves(
        pd.DataFrame([{"candidate_id": "C1", "ra": 10.0, "dec": 20.0}]),
        output_dir=tmp_path,
        workers=1,
    )

    assert len(calls) == 1
    assert out.loc[0, "neowise_identity_status"] == "matched"
    refreshed = pd.read_parquet(tmp_path / "neowise_lc_C1.parquet")
    assert set(refreshed["target_identity_status"]) == {"matched"}


def test_cache_only_legacy_identity_is_not_recorded_as_fetched(tmp_path: Path) -> None:
    pd.DataFrame(
        {"time": [1400.0, 1401.0], "flux": [1.0, 0.9], "quality": [0, 0], "sector": [1, 1]}
    ).to_parquet(tmp_path / "tess_lc_C1.parquet", index=False)

    out = rebuild_external_lc_table_from_cache(
        pd.DataFrame([{"candidate_id": "C1", "ra": 10.0, "dec": 20.0}]),
        tmp_path,
        {"tess": True},
    )

    assert out.loc[0, "tess_identity_status"] == "legacy_unverified"
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert status.loc[0, "status"] == "identity_unverified"


def test_characterization_disabled_is_not_a_completed_checkpoint_state() -> None:
    frame = characterize._run_optional_module(
        pd.DataFrame([{"candidate_id": "c1"}]),
        module="unwise",
        enabled=False,
        description="unused",
        func=lambda value: value,
    )

    assert frame.loc[0, "char_status_unwise"] == "disabled"
    assert characterize._module_completed(frame, "unwise") is False


def test_multi_survey_reports_legacy_identity_as_unverified(tmp_path: Path) -> None:
    pd.DataFrame({"mjd": [58000.0], "w1mpro": [12.0]}).to_parquet(
        tmp_path / "neowise_lc_c1.parquet", index=False
    )
    pd.DataFrame({"time": [1400.0], "flux": [1.0], "quality": [0]}).to_parquet(
        tmp_path / "tess_lc_c1.parquet", index=False
    )
    row = {
        "candidate_id": "c1",
        "dip_best_t0": 8500.0,
        "dip_significant": True,
        "dip_bayes_factor": 2.0,
        "dip_best_width_param": 2.0,
    }

    out = compute_candidate_multi_survey_features(row, external_lc_dir=tmp_path)

    assert out["ms_neowise_identity_status"] == "legacy_unverified"
    assert out["ms_tess_identity_status"] == "legacy_unverified"
    assert out["ms_neowise_status"] == "identity_unverified:legacy_unverified"
    assert out["ms_tess_status"] == "identity_unverified:legacy_unverified"
    assert int(out["ms_neowise_n_near"]) == 0
    assert int(out["ms_tess_n_event"]) == 0
    assert '"tess_identity":"legacy_unverified"' in out["ms_source_status_json"]
