from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id
from malca.enrich.spectra import _generate_link, _normalize_spectra_long, _summarize_spectra_long, run_spectra_availability
from malca.enrich.spectra_catalogs import (
    DEFAULT_SPECTRA_CATALOGS,
    DEFAULT_SPECTRA_CATALOG_SPECS,
    LEGACY_SPECTRA_CATALOG_ALIASES,
    resolve_spectra_catalogs,
)
from malca.enrich.spectra_provenance import merge_external_spectral_provenance


def test_resolve_legacy_catalog_aliases() -> None:
    resolved = resolve_spectra_catalogs({"rave_dr5": "III/283/ravedr6"})
    assert "rave_dr6" in resolved
    assert resolved["rave_dr6"].vizier_id == "III/283/ravedr6"


def test_legacy_alias_map_covers_old_keys() -> None:
    assert LEGACY_SPECTRA_CATALOG_ALIASES["sdss_dr17_spec"] == "sdss_dr16_spec"


def test_default_spectra_catalog_specs_avoid_known_broken_schema() -> None:
    desi = DEFAULT_SPECTRA_CATALOG_SPECS["desi_dr1"]
    assert desi.mode == "xmatch"
    assert desi.tap_table is None
    assert desi.ra_col == "RAICRS"
    assert desi.dec_col == "DEICRS"

    gaia_rvs = DEFAULT_SPECTRA_CATALOG_SPECS["gaia_rvs"]
    assert gaia_rvs.mode == "tap"
    assert gaia_rvs.tap_table == '"I/355/rvsmean"'
    assert gaia_rvs.ra_col == "RAICRS"
    assert gaia_rvs.dec_col == "DEICRS"

    gaia_xp = DEFAULT_SPECTRA_CATALOG_SPECS["gaia_xp"]
    assert gaia_xp.mode == "tap"
    assert gaia_xp.vizier_id == "I/355/xpsummary"
    assert gaia_xp.tap_table == '"I/355/xpsummary"'
    assert gaia_xp.ra_col == "RA_ICRS"
    assert gaia_xp.dec_col == "DE_ICRS"

    lamost = DEFAULT_SPECTRA_CATALOG_SPECS["lamost_dr7"]
    assert lamost.mode == "tap_cone"
    assert lamost.vizier_id == "V/156/dr7lrs"
    assert lamost.tap_table == '"V/156/dr7lrs"'
    assert lamost.ra_col == "RAJ2000"
    assert lamost.dec_col == "DEJ2000"

    assert DEFAULT_SPECTRA_CATALOGS["sdss_dr16_spec"] == "V/154/sdss16"
    for key in ("sdss_boss", "sdss_eboss", "sdss_legacy", "sdss_segue", "sdss_spiders", "sdss_tdss"):
        assert not DEFAULT_SPECTRA_CATALOG_SPECS[key].enabled_by_default
        assert key not in DEFAULT_SPECTRA_CATALOGS


def test_rave_link_generator() -> None:
    link = _generate_link(pd.Series({"survey": "rave_dr6", "RAVEID": "12345"}))
    assert link is not None
    assert "rave-survey.org" in link


def test_desi_link_generator_accepts_vizier_targetid_case() -> None:
    link = _generate_link(pd.Series({"survey": "desi_dr1", "TargetID": 39633355007553139}))
    assert link is not None
    assert "39633355007553139" in link


def test_galah_link_generator_accepts_vizier_galah_identifier() -> None:
    link = _generate_link(pd.Series({"survey": "galah_dr3", "GALAH": 161008000000000}))
    assert link is not None
    assert "161008000000000" in link


def test_apogee_link_generator_accepts_vizier_id_column() -> None:
    link = _generate_link(pd.Series({"survey": "apogee_dr16", "ID": "2M02541269+6041444"}))
    assert link is not None
    assert "2M02541269+6041444" in link


def test_link_generator_handles_missing_link_value() -> None:
    assert _generate_link(pd.Series({"survey": "unknown", "link": pd.NA})) is None


def test_summary_uses_nearest_match() -> None:
    coords = pd.DataFrame([{"candidate_id": "C1"}])
    long_df = pd.DataFrame(
        [
            {"candidate_id": "C1", "survey": "desi_dr1", "sep_arcsec": 2.0, "spectrum_redshift": 0.5, "spectrum_spectral_type": "GAL", "link": "http://a"},
            {"candidate_id": "C1", "survey": "sdss_boss", "sep_arcsec": 0.2, "spectrum_redshift": 0.12, "spectrum_spectral_type": "QSO", "link": "http://b"},
        ]
    )
    summary = _summarize_spectra_long(coords, long_df)
    assert summary.loc[0, "spectrum_redshift"] == 0.12
    assert summary.loc[0, "spectrum_spectral_type"] == "QSO"
    assert summary.loc[0, "spectrum_sep_arcsec"] == 0.2
    assert "desi_dr1" in summary.loc[0, "spectrum_sources"]
    assert "sdss_boss" in summary.loc[0, "spectrum_sources"]


def test_apogee_metadata_columns_are_normalized_and_summarized() -> None:
    coords = pd.DataFrame([{"candidate_id": "C1"}])
    long_df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "survey": "apogee_dr17",
                "catalog": "III/284/allstars",
                "sep_arcsec": 0.3,
                "ID": "2M02541269+6041444",
                "teff": 4100.0,
                "logg": 3.9,
                "[M/H]": -0.15,
                "[Fe/H]": -0.2,
                "HRV": 22.4,
                "e_HRV": 0.08,
                "s_HRV": 0.4,
                "SNR": 120.0,
                "Nvis": 3,
                "SFlag": 0,
                "AFlag": 0,
                "Vsini": 14.0,
                "[C/Fe]": 0.05,
                "[Mg/Fe]": 0.1,
            }
        ]
    )

    normalized = _normalize_spectra_long(long_df)
    assert normalized.loc[0, "APOGEE_ID"] == "2M02541269+6041444"
    assert normalized.loc[0, "TEFF"] == 4100.0
    assert normalized.loc[0, "LOGG"] == 3.9
    assert normalized.loc[0, "NVISITS"] == 3

    summary = _summarize_spectra_long(coords, normalized)
    assert summary.loc[0, "apogee_teff"] == 4100.0
    assert summary.loc[0, "apogee_logg"] == 3.9
    assert summary.loc[0, "apogee_fe_h"] == -0.2
    assert summary.loc[0, "apogee_vhelio_avg"] == 22.4
    assert summary.loc[0, "apogee_c_fe"] == 0.05
    assert summary.loc[0, "apogee_mg_fe"] == 0.1


def test_merge_external_provenance_from_input_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "ra_deg": 10.0,
                "dec_deg": 20.0,
                "tns_name": "AT2020abc",
                "tns_type": "SN Ia",
                "tns_redshift": 0.04,
                "simbad_main_id": "NGC 1234",
                "simbad_otype": "Sy1",
            }
        ]
    )
    merged = merge_external_spectral_provenance(df, pd.DataFrame())
    surveys = set(merged["survey"].astype(str))
    assert "tns" in surveys
    assert "simbad" in surveys
    assert merged.loc[merged["survey"] == "tns", "link"].iloc[0].startswith("https://www.wis-tns.org/object/")


def test_ensure_candidate_id_reads_layer_first_coords() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "external_stats": {"ra": 319.95, "dec": 60.72},
            }
        ]
    )
    out = _ensure_candidate_id(df)
    assert out.loc[0, "ra_deg"] == 319.95
    assert out.loc[0, "dec_deg"] == 60.72


def test_parquet_safe_frame_handles_mixed_catalog_ids(tmp_path: Path) -> None:
    from malca.enrich.spectra import _parquet_safe_frame

    df = pd.DataFrame(
        [
            {"candidate_id": "C1", "survey": "sdss_boss", "objID": 1234567890123456789},
            {"candidate_id": "C2", "survey": "sdss_boss", "objID": "J202733.2-305100"},
        ]
    )
    out = _parquet_safe_frame(df)
    path = tmp_path / "mixed_objid.parquet"
    out.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    assert str(loaded.loc[1, "objID"]) == "J202733.2-305100"


def test_run_spectra_availability_with_mocked_query(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])

    def fake_query(coords, *, survey_specs, representative, radius_arcsec, chunk_size, show_progress=False, progress_desc=None, **kwargs):
        survey = next(iter(survey_specs))
        return pd.DataFrame(
            [
                {
                    "candidate_id": "C1",
                    "sep_arcsec": 0.4,
                    "catalog": representative.vizier_id,
                    "survey": survey,
                    "z": 0.2,
                    "Class": "STAR",
                }
            ]
        )

    monkeypatch.setattr("malca.enrich.spectra.query_spectra_catalog_group", fake_query)
    long_df, summary = run_spectra_availability(targets, out_dir=tmp_path / "spectra", catalogs={"rave_dr6": "III/283/ravedr6"})
    assert summary.loc[0, "has_spectrum"]
    assert summary.loc[0, "spectrum_redshift"] == 0.2
    assert (tmp_path / "spectra" / "spectra_long.parquet").exists()


def test_lamost_spectra_availability_uses_tap(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])
    calls = []

    class FakeResults:
        def __len__(self):
            return 1

        def to_pandas(self):
            return pd.DataFrame(
                [
                    {
                        "ObsID": 300702165,
                        "Target": "J030001.08+000110.8",
                        "Class": "STAR",
                        "subClass": "K4",
                        "RAJ2000": 10.0,
                        "DEJ2000": 20.0,
                    }
                ]
            )

    class FakeJob:
        def get_results(self):
            return FakeResults()

    class FakeTap:
        def __init__(self, *, url):
            calls.append({"url": url})

        def launch_job(self, query, verbose=False):
            calls.append({"query": query, "verbose": verbose})
            return FakeJob()

    monkeypatch.setattr("malca.enrich.spectra_queries.TapPlus", FakeTap)

    long_df, summary = run_spectra_availability(
        targets,
        out_dir=tmp_path / "spectra",
        catalogs={"lamost_dr7": "V/156/dr7lrs"},
        merge_provenance_from_input=False,
    )

    query = next(call["query"] for call in calls if "query" in call)
    assert 'FROM "V/156/dr7lrs" AS c' in query
    assert "TAP_UPLOAD" not in query
    assert "c.RAJ2000" in query
    assert "c.DEJ2000" in query
    assert long_df.loc[0, "survey"] == "lamost_dr7"
    assert long_df.loc[0, "link"] == "https://www.lamost.org/dr7/v2.0/spectrum/view?obsid=300702165"
    assert long_df.loc[0, "sep_arcsec"] == 0.0
    assert summary.loc[0, "has_spectrum"]
    status = pd.read_parquet(tmp_path / "spectra" / "spectra_query_status.parquet")
    assert status.loc[0, "mode"] == "tap_cone"


def test_run_spectra_availability_records_xmatch_query_failure(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])

    def fail_query(*_args, **_kwargs):
        raise ValueError("catalog is not available on the XMatch server")

    monkeypatch.setattr("malca.enrich.neighbor.XMatch.query", fail_query)

    long_df, summary = run_spectra_availability(
        targets,
        out_dir=tmp_path / "spectra",
        catalogs={"rave_dr6": "III/283/ravedr6"},
        merge_provenance_from_input=False,
    )

    assert long_df.empty
    assert not bool(summary.loc[0, "has_spectrum"])
    status = pd.read_parquet(tmp_path / "spectra" / "spectra_query_status.parquet")
    assert status.loc[0, "catalog"] == "III/283/ravedr6"
    assert status.loc[0, "survey_keys"] == "rave_dr6"
    assert status.loc[0, "mode"] == "xmatch"
    assert status.loc[0, "status"] == "error"
    assert "not available" in status.loc[0, "error_message"]
