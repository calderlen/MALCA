from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.enrich.host import host_nuclear_score, run_host_association
from malca.enrich.radio import run_radio_enrichment
from malca.enrich.spectra import run_spectra_availability
from malca.enrich.swift import run_swift_enrichment
from malca.nuclear.clagn_catalogs import load_known_clagn_catalogs, match_known_clagn_catalogs
from malca.nuclear.context import NuclearContextConfig, run_nuclear_context
from malca.nuclear.redshift import resolve_redshift_spectral_types
from malca.nuclear.scoring import score_nuclear_candidates
from malca.nuclear.targets import normalize_nuclear_targets
from malca.review.store import db_connect, get_candidate_payload, import_candidates


def test_normalize_nuclear_targets_adds_ids_and_coordinate_aliases() -> None:
    df = pd.DataFrame([{"name": "AT2024abc", "RA": 12.3, "DEC": -4.5}])

    out = normalize_nuclear_targets(df)

    assert out.loc[0, "candidate_id"] == "AT2024abc"
    assert out.loc[0, "ra"] == 12.3
    assert out.loc[0, "dec_deg"] == -4.5
    assert out.loc[0, "timescale"] == "nuclear"


def test_score_nuclear_candidates_promotes_agn_and_demotes_stars() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "AGN",
                "parallax": 0.0,
                "parallax_error": 0.2,
                "pm_total": 0.1,
                "w1": 12.0,
                "w2": 11.0,
                "radio_det": True,
                "xray_det": True,
                "simbad_otype": "QSO",
                "host_nuclear_score": 1.0,
            },
            {
                "candidate_id": "STAR",
                "parallax": 5.0,
                "parallax_error": 0.2,
                "pm_total": 50.0,
                "pmra_error": 1.0,
                "pmdec_error": 1.0,
                "host_nuclear_score": 0.2,
            },
        ]
    )

    out = score_nuclear_candidates(df)

    agn = out.set_index("candidate_id").loc["AGN"]
    star = out.set_index("candidate_id").loc["STAR"]
    assert agn["agn_prior_score"] >= 0.9
    assert agn["wise_agn_score"] >= 0.9
    assert star["gaia_stellar_veto_score"] >= 0.9
    assert star["gaia_extragalactic_prior_score"] <= 0.2


def test_tde_score_uses_single_flare_quiet_nonstellar_and_low_agn_prior() -> None:
    out = score_nuclear_candidates(
        pd.DataFrame(
            [
                {
                    "candidate_id": "TDE",
                    "parallax": 0.0,
                    "parallax_error": 0.2,
                    "pm_total": 0.0,
                    "host_nuclear_score": 1.0,
                    "tde_single_flare_score": 1.0,
                    "tde_quiet_baseline_score": 1.0,
                    "tde_no_recurrence_score": 1.0,
                    "tde_smooth_decline_score": 1.0,
                    "agn_prior_score": 0.0,
                    "galex_nuv": 19.0,
                    "phot_g_mean_mag": 18.0,
                },
                {
                    "candidate_id": "AGN_FLARE",
                    "host_nuclear_score": 1.0,
                    "tde_single_flare_score": 1.0,
                    "tde_quiet_baseline_score": 0.1,
                    "tde_no_recurrence_score": 0.0,
                    "tde_smooth_decline_score": 0.4,
                    "simbad_otype": "Seyfert 1",
                },
            ]
        )
    ).set_index("candidate_id")

    assert out.loc["TDE", "tde_candidate_score"] > out.loc["AGN_FLARE", "tde_candidate_score"]
    assert "single flare" in out.loc["TDE", "tde_candidate_reasons"]
    assert "demoted by prior AGN evidence" in out.loc["AGN_FLARE", "tde_candidate_reasons"]


def test_redshift_spectral_resolver_maps_sources_and_agn_flags() -> None:
    out = resolve_redshift_spectral_types(
        pd.DataFrame(
            [
                {"candidate_id": "C1", "desi_z": 0.123, "desi_spectype": "QSO broad-line"},
                {"candidate_id": "C2", "tns_redshift": 0.04, "simbad_otype": "Galaxy"},
            ]
        )
    )

    assert out.loc[0, "redshift"] == 0.123
    assert out.loc[0, "redshift_source"] == "DESI"
    assert out.loc[0, "host_spectral_class"] == "broad_line_agn"
    assert bool(out.loc[0, "prior_agn_spectrum_flag"])
    assert out.loc[1, "redshift_source"] == "TNS"


def test_clagn_catalog_loader_and_matcher(tmp_path: Path) -> None:
    path = tmp_path / "desi_clagn.csv"
    path.write_text("object,ra,dec,z,classification\nJ0001,10.0,20.0,0.1,type 1.9\n", encoding="ascii")
    catalog = load_known_clagn_catalogs({"desi": path})

    out = match_known_clagn_catalogs(
        pd.DataFrame([{"candidate_id": "X", "ra_deg": 10.0001, "dec_deg": 20.0001}]),
        catalog,
        radius_arcsec=3.0,
    )

    assert bool(out.loc[0, "known_clagn_match"])
    assert out.loc[0, "known_clagn_source"] == "desi"
    assert out.loc[0, "known_clagn_training_label"] == "known_clagn"


def test_host_radio_swift_enrichment_summaries(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])

    def fake_host_query(coords, *, catalog, radius_arcsec, chunk_size, show_progress=False, progress_desc=None):
        return pd.DataFrame([{"candidate_id": "C1", "sep_arcsec": 0.25, "catalog": catalog}])

    monkeypatch.setattr("malca.enrich.host._query_catalog_bulk", fake_host_query)
    _host_long, host_summary = run_host_association(targets, out_dir=tmp_path / "host", catalogs={"ps1": "fake"})
    assert bool(host_summary.loc[0, "host_match"])
    assert host_summary.loc[0, "host_nuclear_score"] > 0.9
    assert host_nuclear_score(pd.Series([0.25])).iloc[0] > host_nuclear_score(pd.Series([2.5])).iloc[0]

    def fake_radio_query(coords, *, catalog, radius_arcsec, chunk_size, show_progress=False, progress_desc=None):
        return pd.DataFrame([{"candidate_id": "C1", "sep_arcsec": 1.5, "Fpeak": 12.0, "catalog": catalog}])

    monkeypatch.setattr("malca.enrich.radio._query_catalog_bulk", fake_radio_query)
    _radio_long, radio_summary = run_radio_enrichment(targets, out_dir=tmp_path / "radio", catalogs={"first": "fake"})
    assert bool(radio_summary.loc[0, "radio_det"])
    assert radio_summary.loc[0, "radio_flux_mjy"] == 12.0

    def fake_swift_query(coords, *, catalog, radius_arcsec, chunk_size, show_progress=False, progress_desc=None):
        return pd.DataFrame([{"candidate_id": "C1", "sep_arcsec": 2.0, "catalog": catalog}])

    monkeypatch.setattr("malca.enrich.swift._query_catalog_bulk", fake_swift_query)
    _swift_long, swift_summary = run_swift_enrichment(targets, out_dir=tmp_path / "swift", catalogs={"swift_2sxps": "fake"})
    assert bool(swift_summary.loc[0, "swift_xrt_det"])
    assert swift_summary.loc[0, "swift_status"] == "matched"


def test_enrichment_no_match_outputs_are_valid(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])

    def empty_query(coords, *, catalog, radius_arcsec, chunk_size, show_progress=False, progress_desc=None):
        return pd.DataFrame()

    monkeypatch.setattr("malca.enrich.host._query_catalog_bulk", empty_query)
    monkeypatch.setattr("malca.enrich.radio._query_catalog_bulk", empty_query)
    monkeypatch.setattr("malca.enrich.swift._query_catalog_bulk", empty_query)

    _host_long, host_summary = run_host_association(targets, out_dir=tmp_path / "host_no", catalogs={"ps1": "fake"})
    _radio_long, radio_summary = run_radio_enrichment(targets, out_dir=tmp_path / "radio_no", catalogs={"first": "fake"})
    _swift_long, swift_summary = run_swift_enrichment(targets, out_dir=tmp_path / "swift_no", catalogs={"swift_2sxps": "fake"})

    assert not bool(host_summary.loc[0, "host_match"])
    assert not bool(radio_summary.loc[0, "radio_det"])
    assert not bool(swift_summary.loc[0, "swift_xrt_det"])
    assert (tmp_path / "host_no" / "host_long.parquet").exists()
    assert (tmp_path / "radio_no" / "radio_long.parquet").exists()
    assert (tmp_path / "swift_no" / "swift_long.parquet").exists()


def test_spectra_summary_carries_redshift_and_type(monkeypatch, tmp_path: Path) -> None:
    targets = pd.DataFrame([{"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0}])

    def fake_spectra_query(coords, *, catalog, radius_arcsec, chunk_size, show_progress=False, progress_desc=None):
        return pd.DataFrame(
            [
                {
                    "candidate_id": "C1",
                    "sep_arcsec": 0.5,
                    "catalog": catalog,
                    "z": 0.123,
                    "Class": "QSO",
                }
            ]
        )

    monkeypatch.setattr("malca.enrich.spectra._query_catalog_bulk", fake_spectra_query)
    _long, summary = run_spectra_availability(targets, out_dir=tmp_path / "spectra", catalogs={"sdss": "fake"})

    assert summary.loc[0, "spectrum_redshift"] == 0.123
    assert summary.loc[0, "spectrum_spectral_type"] == "QSO"


def test_run_nuclear_context_local_only_writes_scores(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "ra": 10.0,
                "dec": 20.0,
                "parallax": 0.0,
                "parallax_error": 0.1,
                "pm_total": 0.0,
                "host_nuclear_score": 1.0,
                "w1": 12.0,
                "w2": 11.1,
            }
        ]
    )
    config = NuclearContextConfig(
        run_dir=tmp_path / "run",
        run_characterize=False,
        run_ltv_crossmatch=False,
        run_vetting=False,
        run_external_lcs=False,
        run_spectra=False,
        run_host=False,
        run_radio=False,
        run_swift=False,
        run_clagn_catalogs=False,
    )

    out = run_nuclear_context(df, config)

    assert "agn_prior_score" in out.columns
    assert out.loc[0, "wise_agn_score"] > 0.9
    assert (tmp_path / "run" / "results" / "nuclear_context.parquet").exists()


def test_review_import_preserves_nuclear_score_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "asas_sn_id": "C1",
                "agn_prior_score": 0.9,
                "tde_candidate_score": 0.2,
                "host_nuclear_score": 1.0,
                "known_clagn_match": True,
            }
        ]
    )

    with db_connect(db_path) as conn:
        import_candidates(conn, frame, source_path=str(tmp_path), characterize_before_import=False, vet_before_import=False)
        payload = get_candidate_payload(conn, "C1")
        row = conn.execute(
            "SELECT agn_prior_score, tde_candidate_score, host_nuclear_score, known_clagn_match FROM candidates WHERE candidate_id='C1'"
        ).fetchone()

    assert payload["agn_prior_score"] == 0.9
    assert row == (0.9, 0.2, 1.0, 1)
