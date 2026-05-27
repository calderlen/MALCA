from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca import tag
from malca.review.maintenance import backfill_vsx_results
from malca.review.store import db_connect, get_candidate_payload, upsert_candidates_frame
from malca.vsx import crossmatch as vsx_crossmatch
from malca.vsx.filter import filter_vsx, normalize_vsx_catalog, write_clean_outputs


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
            "SELECT vsx_class, vsx_sep_arcsec, vsx_period FROM candidates WHERE candidate_id=?",
            ("292058860969",),
        ).fetchone()

    assert updated == 1
    assert payload["vsx_class"] == "GCAS"
    assert row == ("GCAS", 0.4, 591.8991)
