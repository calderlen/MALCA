from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.io.table_io import read_feature_table, read_parquet_table, write_feature_table
from malca.products.feature_layers import with_feature_columns
from malca.stv import tag
from malca.review.maintenance import backfill_vsx_live_results, backfill_vsx_live_run, backfill_vsx_results
from malca.review.store import db_connect, get_candidate_payload, merge_candidate_results, upsert_candidates_frame
from malca.vsx import crossmatch as vsx_crossmatch
from malca.vsx.filter import filter_vsx, normalize_vsx_catalog, write_clean_outputs
from malca.vsx.nearby import VsxNeighbor


def test_vsx_all_keeps_classes_filtered_from_cleaned(tmp_path: Path) -> None:
    df_asassn = pd.DataFrame({"asas_sn_id": ["A1"], "ra_deg": [10.0], "dec_deg": [5.0]})
    df_vsx_raw = pd.DataFrame(
        {
            "id_vsx": [1, 2],
            "name": ["NSVS 1", "NSVS 2"],
            "ra": [10.0, 11.0],
            "dec": [5.0, 6.0],
            "class": ["GCAS", "VAR"],
            "period": [591.9, 12.3],
        }
    )

    vsx_all = normalize_vsx_catalog(df_vsx_raw)
    vsx_clean = filter_vsx(df_vsx_raw)
    _asas_path, clean_path, all_path = write_clean_outputs(df_asassn, vsx_clean, vsx_all, output_dir=tmp_path)

    clean = pd.read_parquet(clean_path)
    all_rows = pd.read_parquet(all_path)

    assert "GCAS" in set(all_rows["class"])
    assert "GCAS" not in set(clean["class"])


def test_full_vsx_crossmatch_preserves_gcas_label(tmp_path: Path) -> None:
    asassn_path = tmp_path / "asassn.parquet"
    vsx_path = tmp_path / "vsx_all.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": ["292058860969"],
            "ra_deg": [107.46549],
            "dec_deg": [-7.98314],
            "pm_ra": [0.0],
            "pm_dec": [0.0],
        }
    ).to_parquet(asassn_path, index=False)
    pd.DataFrame(
        {
            "ra": [107.46550],
            "dec": [-7.98314],
            "class": ["GCAS"],
            "period": [591.8991],
        }
    ).to_parquet(vsx_path, index=False)

    out = vsx_crossmatch.crossmatch_asassn_vsx(asassn_path, vsx_path)

    assert out.loc[0, "vsx_class"] == "GCAS"
    assert out.loc[0, "vsx_period"] == 591.8991


def test_attach_vsx_info_uses_full_crossmatch_labels(tmp_path: Path) -> None:
    xmatch_path = tmp_path / "xmatch.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": ["292058860969"],
            "sep_arcsec": [0.25],
            "class": ["GCAS"],
            "period": [591.8991],
        }
    ).to_parquet(xmatch_path, index=False)

    out = tag.attach_vsx_info(
        pd.DataFrame({"asas_sn_id": ["292058860969"]}),
        vsx_crossmatch_csv=xmatch_path,
    )

    assert out.loc[0, "vsx_class"] == "GCAS"
    assert out.loc[0, "vsx_sep_arcsec"] == 0.25
    assert out.loc[0, "vsx_period"] == 591.8991


def test_backfill_vsx_updates_payload_and_sql_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    xmatch_path = tmp_path / "xmatch.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": ["292058860969"],
            "vsx_class": ["GCAS"],
            "vsx_sep_arcsec": [0.4],
            "vsx_period": [591.8991],
        }
    ).to_parquet(xmatch_path, index=False)

    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "292058860969",
                        "asas_sn_id": "292058860969",
                    }
                ]
            ),
        )
        updated = backfill_vsx_results(
            conn,
            crossmatch=xmatch_path,
            raw_vsx=None,
            radius_arcsec=3.0,
            chunksize=100,
        )
        payload = get_candidate_payload(conn, "292058860969")
        row = conn.execute(
            """
            SELECT vsx_class, vsx_sep_arcsec, vsx_period, vetting_likely_known
            FROM candidates
            WHERE candidate_id=?
            """,
            ("292058860969",),
        ).fetchone()

    assert updated == 1
    assert payload["vsx_class"] == "GCAS"
    assert payload["vetting_likely_known"] is True
    assert row == ("GCAS", 0.4, 591.8991, 1)


def test_backfill_vsx_live_updates_missing_payload_and_sql_columns(tmp_path: Path, monkeypatch) -> None:
    import malca.review.maintenance as maintenance

    def fake_nearby(ra, dec, *, limit, radius_arcsec, timeout_sec):
        assert ra == 10.0
        assert dec == 20.0
        assert limit == 3
        assert radius_arcsec == 3.0
        assert timeout_sec == 2.0
        return [
            VsxNeighbor(
                sep_arcsec=0.61,
                oid="101",
                name="NSVS 12455434",
                ra_deg=10.0,
                dec_deg=20.0,
                vsx_type="EB",
                type_label="EB - Beta Lyrae-type eclipsing binary",
                period_days=3.6826,
                url="https://vsx.aavso.org/index.php?view=detail.top&oid=101",
            )
        ]

    monkeypatch.setattr(maintenance, "find_nearby_vsx", fake_nearby)
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "stv_231929035169",
                        "asas_sn_id": "231929035169",
                    }
                ]
            ),
        )
        merge_candidate_results(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "stv_231929035169",
                        "ra": 10.0,
                        "dec": 20.0,
                    }
                ]
            ),
        )
        updated = backfill_vsx_live_results(
            conn,
            radius_arcsec=3.0,
            timeout_sec=2.0,
            limit=3,
        )
        payload = get_candidate_payload(conn, "stv_231929035169")
        row = conn.execute(
            """
            SELECT vsx_class, vsx_sep_arcsec, vsx_period, vetting_likely_known
            FROM candidates
            WHERE candidate_id=?
            """,
            ("stv_231929035169",),
        ).fetchone()

    assert updated == 1
    assert payload["vsx_class"] == "EB"
    assert payload["vetting_likely_known"] is True
    assert row == ("EB", 0.61, 3.6826, 1)


def test_backfill_vsx_live_skips_existing_class_by_default(tmp_path: Path, monkeypatch) -> None:
    import malca.review.maintenance as maintenance

    def fail_nearby(*args, **kwargs):
        raise AssertionError("live VSX should not be queried for populated rows")

    monkeypatch.setattr(maintenance, "find_nearby_vsx", fail_nearby)
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "stv-existing",
                        "asas_sn_id": "1",
                        "vsx_class": "ROT",
                        "vsx_sep_arcsec": 0.2,
                        "vsx_period": 1.5,
                    }
                ]
            ),
        )
        merge_candidate_results(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "stv-existing",
                        "ra": 10.0,
                        "dec": 20.0,
                    }
                ]
            ),
        )
        updated = backfill_vsx_live_results(
            conn,
            radius_arcsec=3.0,
            timeout_sec=2.0,
            limit=3,
        )
        row = conn.execute(
            """
            SELECT vsx_class, vsx_sep_arcsec, vsx_period
            FROM candidates
            WHERE candidate_id=?
            """,
            ("stv-existing",),
        ).fetchone()

    assert updated == 0
    assert row == ("ROT", 0.2, 1.5)


def test_backfill_vsx_live_run_writes_sidecar_products_and_db(tmp_path: Path, monkeypatch) -> None:
    import malca.review.maintenance as maintenance

    calls: list[tuple[float, float]] = []

    def fake_nearby(ra, dec, *, limit, radius_arcsec, timeout_sec):
        calls.append((float(ra), float(dec)))
        assert limit == 3
        assert radius_arcsec == 3.0
        assert timeout_sec == 2.0
        if float(ra) == 10.0:
            return [
                VsxNeighbor(
                    sep_arcsec=0.61,
                    oid="101",
                    name="NSVS 12455434",
                    ra_deg=10.0,
                    dec_deg=20.0,
                    vsx_type="EB",
                    type_label="EB - Beta Lyrae-type eclipsing binary",
                    period_days=3.6826,
                    url="https://vsx.aavso.org/index.php?view=detail.top&oid=101",
                ),
                VsxNeighbor(
                    sep_arcsec=1.2,
                    oid="202",
                    name="VSX no period",
                    ra_deg=10.0001,
                    dec_deg=20.0001,
                    vsx_type="EW",
                    type_label="EW - W Ursae Majoris-type eclipsing binary",
                    period_days=None,
                    url="https://vsx.aavso.org/index.php?view=detail.top&oid=202",
                ),
            ]
        return []

    monkeypatch.setattr(maintenance, "find_nearby_vsx", fake_nearby)
    run_dir = tmp_path / "run"
    results_dir = run_dir / "results"
    review_dir = run_dir / "review"
    results_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)

    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "stv-hit",
                "asas_sn_id": "231929035169",
                "gaia_id": "123",
                "ra": 10.0,
                "dec": 20.0,
            },
            {
                "candidate_id": "stv-miss",
                "asas_sn_id": "2",
                "gaia_id": "456",
                "ra": 11.0,
                "dec": 21.0,
            },
            {
                "candidate_id": "stv-missing-coords",
                "asas_sn_id": "3",
                "gaia_id": "789",
                "ra": pd.NA,
                "dec": 22.0,
            },
        ]
    )
    write_feature_table(candidates, results_dir / "lc_events_vetted.parquet")
    write_feature_table(candidates, results_dir / "lc_events_neighbors.parquet")

    db_path = review_dir / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(conn, candidates[["candidate_id", "asas_sn_id"]])
        merge_candidate_results(conn, candidates[["candidate_id", "ra", "dec"]])

    stats = backfill_vsx_live_run(
        run_dir,
        review_db=None,
        output_path=None,
        radius_arcsec=3.0,
        timeout_sec=2.0,
        limit=3,
    )

    sidecar_path = results_dir / "vsx_live_backfill.parquet"
    sidecar = read_parquet_table(sidecar_path).sort_values("candidate_id").reset_index(drop=True)
    statuses = dict(zip(sidecar["candidate_id"], sidecar["vsx_live_status"]))
    hit = sidecar.loc[sidecar["candidate_id"].eq("stv-hit")].iloc[0]
    vetted = with_feature_columns(
        read_feature_table(results_dir / "lc_events_vetted.parquet"),
        ["vsx_class", "vsx_sep_arcsec", "vsx_period"],
    )
    neighbors = with_feature_columns(
        read_feature_table(results_dir / "lc_events_neighbors.parquet"),
        ["vsx_class", "vsx_sep_arcsec", "vsx_period"],
    )
    with db_connect(db_path) as conn:
        db_row = conn.execute(
            """
            SELECT vsx_class, vsx_sep_arcsec, vsx_period, vetting_likely_known
            FROM candidates
            WHERE candidate_id=?
            """,
            ("stv-hit",),
        ).fetchone()

    assert stats["sidecar_rows"] == 3
    assert stats["matched"] == 1
    assert stats["no_match"] == 1
    assert stats["missing_coords"] == 1
    assert stats["parquets_updated"] == 2
    assert stats["parquet_rows_updated"] == 2
    assert stats["db_candidates_updated"] == 1
    assert calls == [(10.0, 20.0), (11.0, 21.0)]
    assert statuses == {
        "stv-hit": "matched",
        "stv-miss": "no_match",
        "stv-missing-coords": "missing_coords",
    }
    assert hit["vsx_class"] == "EB"
    assert hit["vsx_sep_arcsec"] == 0.61
    assert hit["vsx_period"] == 3.6826
    assert hit["vsx_oid"] == "101"
    assert hit["vsx_neighbor_oids"] == "101|202"
    assert hit["vsx_neighbor_classes"] == "EB|EW"
    assert hit["vsx_neighbor_sep_arcsec"] == "0.61|1.2"
    assert vetted.loc[vetted["candidate_id"].eq("stv-hit"), "vsx_class"].iloc[0] == "EB"
    assert neighbors.loc[neighbors["candidate_id"].eq("stv-hit"), "vsx_class"].iloc[0] == "EB"
    assert pd.isna(vetted.loc[vetted["candidate_id"].eq("stv-miss"), "vsx_class"].iloc[0])
    assert db_row == ("EB", 0.61, 3.6826, 1)
    assert list(results_dir.glob("lc_events_vetted.parquet.pre_vsx_live_backfill_*.bak"))
    assert list(results_dir.glob("lc_events_neighbors.parquet.pre_vsx_live_backfill_*.bak"))
    assert list(review_dir.glob("review.db.pre_vsx_live_backfill_*.bak"))
