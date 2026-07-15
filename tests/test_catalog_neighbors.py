from __future__ import annotations

import multiprocessing as mp
import time

import pandas as pd
import pytest
from astropy.table import Table
from astroquery.xmatch import XMatch

from malca.catalogs.evidence import catalog_neighbor_record, normalize_catalog_neighbor_frame
from malca.enrich import neighbor
from malca.enrichment import vetting
from malca.products.feature_layers import to_layer_first_frame


def _fork_context_or_skip():
    try:
        return mp.get_context("fork")
    except ValueError:
        pytest.skip("fork multiprocessing context is unavailable")


def test_catalog_neighbor_normalization_flags_and_ranks_v1_catalogs() -> None:
    specs = [
        ("vsx", "vsx_class", "EA", True),
        ("simbad", "simbad_otype", "EB*", True),
        ("asassn_variables", "asassn_var_type", "EA", True),
        ("ztf_periodic_variables", "ztf_var_type", "EA", True),
        ("tns", "tns_type", "CV", True),
        ("microlensing_catalogs", "microlens_catalog", "OGLE-EWS", False),
    ]
    rows = []
    for catalog, class_column, class_value, is_dipper in specs:
        for sep in (30.0, 5.0, 15.0):
            rows.append(
                catalog_neighbor_record(
                    candidate_id=f"C-{catalog}",
                    catalog=catalog,
                    object_id=f"{catalog}-{sep:g}",
                    object_name=f"{catalog} object",
                    class_column=class_column,
                    class_value=class_value,
                    sep_arcsec=sep,
                    period_days=2.5,
                    query_radius_arcsec=30.0,
                    raw={"sep": sep},
                )
            )

    normalized = normalize_catalog_neighbor_frame(rows)

    assert len(normalized) == 18
    assert set(normalized["sep_arcsec"]) == {5.0, 15.0, 30.0}
    for catalog, _class_column, _class_value, is_dipper in specs:
        subset = normalized[normalized["catalog"].eq(catalog)]
        assert subset["rank"].tolist() == [1, 2, 3]
        assert subset["is_known_variable"].tolist() == [True, True, True]
        assert subset["is_dipper_contaminant"].tolist() == [is_dipper, is_dipper, is_dipper]


def test_catalog_neighbor_coords_read_layer_first_ra_dec() -> None:
    frame = to_layer_first_frame(
        pd.DataFrame(
            [
                {"candidate_id": "C1", "ra": 12.5, "dec": -1.25},
                {"candidate_id": "C2", "ra": 14.0, "dec": 3.5},
            ]
        ),
        run_derived=False,
    )

    coords = vetting._catalog_neighbor_coords(frame)

    assert coords[["candidate_id", "ra_deg", "dec_deg"]].to_dict("records") == [
        {"candidate_id": "C1", "ra_deg": 12.5, "dec_deg": -1.25},
        {"candidate_id": "C2", "ra_deg": 14.0, "dec_deg": 3.5},
    ]


def test_catalog_neighbor_xmatch_timeout_is_restored() -> None:
    previous = XMatch.TIMEOUT

    with vetting._temporary_xmatch_timeout(7):
        assert XMatch.TIMEOUT == 7

    assert XMatch.TIMEOUT == previous


def test_xmatch_chunk_subprocess_returns_frame(monkeypatch) -> None:
    monkeypatch.setattr(neighbor, "_xmatch_process_context", _fork_context_or_skip)

    def fake_query(**kwargs):
        assert kwargs["cat2"] == "vizier:B/vsx/vsx"
        return Table(
            {
                "candidate_id": ["C1"],
                "angDist": [2.5],
                "Name": ["VSX C1"],
            }
        )

    monkeypatch.setattr(neighbor.XMatch, "query", fake_query)

    result = neighbor.query_xmatch_chunk(
        pd.DataFrame([{"candidate_id": "C1", "ra_deg": 12.5, "dec_deg": -1.25}]),
        cat2="vizier:B/vsx/vsx",
        radius_arcsec=30.0,
        timeout_sec=5.0,
    )

    assert result.to_dict("records") == [
        {"candidate_id": "C1", "angDist": 2.5, "Name": "VSX C1"}
    ]


def test_query_catalog_bulk_hard_times_out_stuck_xmatch_chunk(monkeypatch) -> None:
    monkeypatch.setattr(neighbor, "_xmatch_process_context", _fork_context_or_skip)

    def slow_query(**kwargs):
        time.sleep(10)
        return Table()

    monkeypatch.setattr(neighbor.XMatch, "query", slow_query)

    status_rows: list[dict] = []
    t0 = time.perf_counter()
    result = neighbor._query_catalog_bulk(
        pd.DataFrame([{"candidate_id": "C1", "ra_deg": 12.5, "dec_deg": -1.25}]),
        catalog="B/vsx/vsx",
        radius_arcsec=30.0,
        chunk_size=1,
        xmatch_timeout_sec=0.2,
        status_rows=status_rows,
    )
    elapsed = time.perf_counter() - t0

    assert result.empty
    assert elapsed < 3.0
    assert status_rows == [
        {
            "catalog": "B/vsx/vsx",
            "mode": "xmatch",
            "chunk_start": 0,
            "chunk_stop": 1,
            "attempted": 1,
            "matched": 0,
            "error_message": "XMatch chunk timed out after 0.2s for vizier:B/vsx/vsx",
            "status": "timeout",
        }
    ]


def test_collect_vsx_catalog_neighbors_prefers_local_catalog(monkeypatch, tmp_path) -> None:
    vsx_path = tmp_path / "vsx_all.parquet"
    pd.DataFrame(
        [
            {
                "id_vsx": 101,
                "name": "VSX Close",
                "ra": 10.0,
                "dec": 20.0,
                "class": "EA",
                "period": 1.25,
                "var_flag": 0,
            },
            {
                "id_vsx": 202,
                "name": "VSX Outside",
                "ra": 10.1,
                "dec": 20.1,
                "class": "EA",
                "period": 2.5,
                "var_flag": 0,
            },
            {
                "id_vsx": 303,
                "name": "VSX Constant Flag",
                "ra": 11.0,
                "dec": 21.0,
                "class": "EA",
                "period": 3.5,
                "var_flag": 2,
            },
        ]
    ).to_parquet(vsx_path, index=False)

    def fail_remote_lookup(*args, **kwargs):
        pytest.fail("remote VSX lookup should not run when local vsx_all.parquet exists")

    monkeypatch.setattr(vetting, "VSX_ALL_CATALOG_PATH", vsx_path)
    monkeypatch.setattr(vetting, "batch_tap_crossmatch", fail_remote_lookup)
    coords = pd.DataFrame(
        [
            {"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0},
            {"candidate_id": "C2", "ra_deg": 11.0, "dec_deg": 21.0},
        ]
    )

    records = vetting._collect_vsx_catalog_neighbors(
        coords,
        radius_arcsec=30.0,
        chunk_size=250,
        method="tap",
        xmatch_timeout_sec=45.0,
        show_progress=True,
    )

    by_id = {record["object_id"]: record for record in records}
    assert set(by_id) == {"101", "303"}
    assert by_id["101"]["candidate_id"] == "C1"
    assert by_id["101"]["catalog"] == "vsx"
    assert by_id["101"]["object_name"] == "VSX Close"
    assert by_id["101"]["class_value"] == "EA"
    assert by_id["101"]["period_days"] == 1.25
    assert by_id["101"]["sep_arcsec"] == pytest.approx(0.0)
    assert by_id["101"]["is_known_variable"] is True
    assert by_id["101"]["is_dipper_contaminant"] is True
    assert by_id["303"]["candidate_id"] == "C2"
    assert by_id["303"]["is_known_variable"] is False
    assert by_id["303"]["is_dipper_contaminant"] is False


def test_builtin_vsx_catalog_rejects_partial_parquet(monkeypatch, tmp_path) -> None:
    vsx_path = tmp_path / "vsx_all.parquet"
    pd.DataFrame(
        [
            {"id_vsx": 101, "name": "Partial", "ra": 10.0, "dec": 20.0, "class": "EA"},
            {"id_vsx": 202, "name": "Partial 2", "ra": 11.0, "dec": 21.0, "class": "EB"},
        ]
    ).to_parquet(vsx_path, index=False)

    monkeypatch.setattr(vetting, "VSX_BUILTIN_ALL_CATALOG_PATH", vsx_path)
    monkeypatch.setattr(vetting, "VSX_BUILTIN_RAW_CATALOG_PATH", tmp_path / "missing_raw.dat")
    monkeypatch.setattr(vetting, "VSX_LOCAL_CATALOG_MIN_ROWS", 5)

    with pytest.raises(RuntimeError, match="appears incomplete"):
        vetting._validate_vsx_local_catalog_ready(vsx_path)


def test_collect_vsx_catalog_neighbors_can_use_vizier_tap(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_batch_tap_crossmatch(
        coords_df,
        *,
        tap_url,
        catalog_table,
        select_cols,
        ra_col,
        dec_col,
        match_radius_arcsec,
        chunk_size,
        n_workers,
        verbose,
        desc,
        timeout,
        raise_on_all_failed,
        raise_on_failed_chunk,
    ):
        calls.append(
            {
                "coords": coords_df.to_dict("records"),
                "tap_url": tap_url,
                "catalog_table": catalog_table,
                "select_cols": select_cols,
                "ra_col": ra_col,
                "dec_col": dec_col,
                "match_radius_arcsec": match_radius_arcsec,
                "chunk_size": chunk_size,
                "n_workers": n_workers,
                "verbose": verbose,
                "desc": desc,
                "timeout": timeout,
                "raise_on_all_failed": raise_on_all_failed,
                "raise_on_failed_chunk": raise_on_failed_chunk,
            }
        )
        return pd.DataFrame(
            [
                {
                    "_idx": 1,
                    "OID": 202,
                    "Name": "VSX B",
                    "Type": "EA",
                    "Period": 1.25,
                    "sep_arcsec": 12.0,
                }
            ]
        )

    monkeypatch.setattr(vetting, "batch_tap_crossmatch", fake_batch_tap_crossmatch)
    monkeypatch.setattr(vetting, "VSX_ALL_CATALOG_PATH", tmp_path / "missing_vsx_all.parquet")
    coords = pd.DataFrame(
        [
            {"candidate_id": "C1", "ra_deg": 10.0, "dec_deg": 20.0},
            {"candidate_id": "C2", "ra_deg": 11.0, "dec_deg": 21.0},
        ]
    )

    records = vetting._collect_vsx_catalog_neighbors(
        coords,
        radius_arcsec=30.0,
        chunk_size=250,
        method="tap",
        xmatch_timeout_sec=45.0,
        show_progress=True,
    )

    assert calls == [
        {
            "coords": [
                {"_idx": 0, "ra": 10.0, "dec": 20.0},
                {"_idx": 1, "ra": 11.0, "dec": 21.0},
            ],
            "tap_url": vetting.VIZIER_TAP_URL,
            "catalog_table": '"B/vsx/vsx"',
            "select_cols": 'c."OID", c."Name", c."RAJ2000", c."DEJ2000", c."Type", c."Period"',
            "ra_col": "RAJ2000",
            "dec_col": "DEJ2000",
            "match_radius_arcsec": 30.0,
            "chunk_size": 250,
            "n_workers": 4,
            "verbose": True,
            "desc": "catalog-neighbors:VSX TAP",
            "timeout": 45.0,
            "raise_on_all_failed": True,
            "raise_on_failed_chunk": True,
        }
    ]
    assert len(records) == 1
    assert records[0]["candidate_id"] == "C2"
    assert records[0]["catalog"] == "vsx"
    assert records[0]["object_id"] == "202"
    assert records[0]["object_name"] == "VSX B"
    assert records[0]["class_value"] == "EA"
    assert records[0]["sep_arcsec"] == 12.0
    assert records[0]["period_days"] == 1.25


def test_collect_catalog_neighbors_reports_progress_and_passes_timeout(monkeypatch, capsys) -> None:
    calls = []

    def fake_collect_vsx(
        coords,
        *,
        radius_arcsec,
        chunk_size,
        method,
        xmatch_timeout_sec,
        show_progress,
    ):
        calls.append(
            {
                "n_coords": len(coords),
                "radius_arcsec": radius_arcsec,
                "chunk_size": chunk_size,
                "method": method,
                "xmatch_timeout_sec": xmatch_timeout_sec,
                "show_progress": show_progress,
            }
        )
        return [
            catalog_neighbor_record(
                candidate_id="C1",
                catalog="vsx",
                class_column="vsx_class",
                class_value="EA",
                sep_arcsec=3.0,
            )
        ]

    monkeypatch.setattr(vetting, "_collect_vsx_catalog_neighbors", fake_collect_vsx)

    neighbors = vetting.collect_catalog_neighbors(
        pd.DataFrame([{"candidate_id": "C1", "ra": 12.5, "dec": -1.25}]),
        catalogs=["vsx"],
        radius_arcsec=30.0,
        chunk_size=17,
        xmatch_timeout_sec=9.0,
        show_progress=True,
    )
    captured = capsys.readouterr().out

    assert len(neighbors) == 1
    assert calls == [
        {
            "n_coords": 1,
            "radius_arcsec": 30.0,
            "chunk_size": 17,
            "method": "xmatch",
            "xmatch_timeout_sec": 9.0,
            "show_progress": True,
        }
    ]
    assert "Catalog neighbors: 1 candidate(s)" in captured
    assert "Catalog neighbors/vsx: starting" in captured
    assert "Catalog neighbors/vsx: 1 row(s)" in captured
