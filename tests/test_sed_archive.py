from __future__ import annotations

import pandas as pd

from malca.enrichment import sed_archive


def _candidate() -> pd.DataFrame:
    return pd.DataFrame(
        [{"candidate_id": "archive-candidate", "ra_deg": 10.0, "dec_deg": 20.0}]
    )


def test_spitzer_seip_discovery_records_products_and_resumable_job(monkeypatch) -> None:
    monkeypatch.setattr(
        sed_archive,
        "_archive_query_position",
        lambda row, epoch_jyear: (10.0, 20.0, "test_propagation"),
    )
    result = pd.DataFrame(
        [
            {
                "energy_bandpassname": "IRAC1",
                "instrument_name": "IRAC",
                "obs_id": "seip-1",
                "dataproduct_subtype": "science",
                "access_url": "https://example.test/seip.mean.fits",
                "access_format": "image/fits",
                "calib_level": 3,
                "t_min": 55000.0,
                "t_max": 55001.0,
            },
            {
                "energy_bandpassname": "IRAC1",
                "instrument_name": "IRAC",
                "obs_id": "seip-1",
                "dataproduct_subtype": "coverage",
                "access_url": "https://example.test/seip.cov.fits",
                "access_format": "image/fits",
                "calib_level": 3,
                "t_min": 55000.0,
                "t_max": 55001.0,
            },
        ]
    )
    from astroquery.ipac.irsa import Irsa

    monkeypatch.setattr(Irsa, "query_sia", lambda **kwargs: result)

    coverage, products, jobs = sed_archive.discover_spitzer_seip(_candidate())

    assert len(coverage) == 1
    assert coverage[0]["coverage_status"] == "covered_product"
    assert coverage[0]["band"] == "IRAC1"
    assert coverage[0]["product_count"] == 2
    assert {row["product_type"] for row in products} == {
        "science_image",
        "coverage_map",
    }
    assert len(jobs) == 1
    assert jobs[0]["job_status"] == "queued"
    assert jobs[0]["job_type"] == "spitzer_seip_forced_photometry"


def test_spitzer_sia_discovery_adds_timeout_and_emits_target_checkpoint(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sed_archive,
        "_archive_query_position",
        lambda row, epoch_jyear: (10.0, 20.0, "test_propagation"),
    )
    from astroquery.ipac.irsa import Irsa

    seen: dict[str, object] = {}

    def fake_request(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return object()

    def fake_query_sia(**kwargs):
        Irsa._session.get("https://example.test/sia")
        return pd.DataFrame()

    monkeypatch.setattr(Irsa._session, "request", fake_request)
    monkeypatch.setattr(Irsa, "query_sia", fake_query_sia)

    checkpoints: list[tuple[str, int, int, int]] = []
    coverage, products, jobs = sed_archive.discover_spitzer_seip(
        _candidate(),
        query_timeout_seconds=7.0,
        checkpoint_callback=lambda source, cov, prod, queued: checkpoints.append(
            (source, len(cov), len(prod), len(queued))
        ),
    )

    assert seen["timeout"] == (7.0, 7.0)
    assert checkpoints == [("spitzer", 1, 0, 0)]
    assert coverage[0]["coverage_status"] == "not_observed"
    assert not products
    assert not jobs


def test_archive_discovery_skips_completed_checkpoint_targets(monkeypatch) -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "done", "ra_deg": 10.0, "dec_deg": 20.0},
            {"candidate_id": "pending", "ra_deg": 11.0, "dec_deg": 21.0},
        ]
    )
    seen: list[str] = []

    def fake_discover(frame, **kwargs):
        seen.extend(frame["candidate_id"].astype(str))
        return [], [], []

    monkeypatch.setattr(sed_archive, "discover_spitzer_seip", fake_discover)
    sed_archive.discover_sed_archive_products(
        candidates,
        sources=["spitzer"],
        completed_candidate_ids_by_source={"spitzer": {"done"}},
    )

    assert seen == ["pending"]


def test_herschel_discovery_marks_center_matches_as_coverage_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        sed_archive,
        "_archive_query_position",
        lambda row, epoch_jyear: (10.0, 20.0, "test_propagation"),
    )
    from astroquery.esa.hsa import HSA

    monkeypatch.setattr(
        HSA,
        "query_hsa_tap",
        lambda *args, **kwargs: pd.DataFrame(
            [{"observation_id": "1342", "instrument_oid": 1, "target_name": "target"}]
        ),
    )

    coverage, products, jobs = sed_archive.discover_herschel_hsa(_candidate())

    assert len(coverage) == 1
    assert coverage[0]["coverage_status"] == "covered_product"
    assert coverage[0]["instrument"] == "PACS"
    assert products[0]["product_type"] == "observation_bundle"
    assert jobs[0]["job_type"] == "herschel_map_validate_photometry"


def test_apex_discovery_never_queues_raw_bolometer_data_as_fits_photometry(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sed_archive,
        "_archive_query_position",
        lambda row, epoch_jyear: (10.0, 20.0, "test_propagation"),
    )

    class FakeTap:
        def search(self, query):
            return pd.DataFrame(
                [
                    {
                        "dp_id": "APEX.1",
                        "datalink_url": "https://example.test/apex/datalink",
                        "access_format": "application/x-votable+xml",
                        "mjd_obs": 56000.0,
                        "exposure": 120.0,
                        "prog_id": "0000.A-0000",
                    }
                ]
            )

    import pyvo

    monkeypatch.setattr(pyvo.dal, "TAPService", lambda url: FakeTap())

    coverage, products, jobs = sed_archive.discover_apex_bolometer(_candidate())

    assert coverage[0]["coverage_status"] == "reduction_required"
    assert products[0]["product_status"] == "reduction_required"
    assert products[0]["product_type"] == "raw_bolometer_observation"
    assert jobs[0]["job_status"] == "reduction_required"
    assert jobs[0]["job_type"] == "apex_bolometer_classify_reduce"
