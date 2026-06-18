from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id
from malca.enrich.spectra import _generate_link, _summarize_spectra_long, run_spectra_availability
from malca.enrich.spectra_catalogs import LEGACY_SPECTRA_CATALOG_ALIASES, resolve_spectra_catalogs
from malca.enrich.spectra_provenance import merge_external_spectral_provenance


def test_resolve_legacy_catalog_aliases() -> None:
    resolved = resolve_spectra_catalogs({"rave_dr5": "III/283/ravedr6"})
    assert "rave_dr6" in resolved
    assert resolved["rave_dr6"].vizier_id == "III/283/ravedr6"


def test_legacy_alias_map_covers_old_keys() -> None:
    assert LEGACY_SPECTRA_CATALOG_ALIASES["sdss_dr17_spec"] == "sdss_dr16_spec"


def test_rave_link_generator() -> None:
    link = _generate_link(pd.Series({"survey": "rave_dr6", "RAVEID": "12345"}))
    assert link is not None
    assert "rave-survey.org" in link


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


def test_run_spectra_availability_with_mocked_query(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])

    def fake_query(coords, *, survey_specs, representative, radius_arcsec, chunk_size, show_progress=False, progress_desc=None):
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
