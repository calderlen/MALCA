from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

import malca.characterize as characterize


def test_get_dust_extinction_handles_duplicate_dustmaps_index(monkeypatch) -> None:
    def fake_dustmaps3d(l, b, d):
        n = len(l)
        idx = [42] * n
        ebv = pd.Series(np.linspace(0.1, 0.2, n), index=idx)
        density = pd.Series(np.zeros(n), index=idx)
        sigma = pd.Series(np.full(n, 0.01), index=idx)
        max_dist = pd.Series(np.full(n, 3.0), index=idx)
        return ebv, density, sigma, max_dist

    monkeypatch.setattr(characterize, "dustmaps3d", fake_dustmaps3d)

    df = pd.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [-10.0, 5.0],
            "distance_gspphot": [1000.0, 1500.0],
        }
    )

    out = characterize.get_dust_extinction(df)

    assert "A_v_3d" in out.columns
    assert "ebv_3d" in out.columns
    assert out["ebv_3d"].notna().sum() == 2
    assert (out["A_v_3d"] > 0).sum() == 2


def test_starhorse_tap_query_uses_cache_and_schema_available_columns(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "starhorse_cache.parquet"
    stats: dict[str, object] = {"data_queries": 0, "last_data_query": ""}

    class FakeResult:
        def __init__(self, table: Table) -> None:
            self._table = table

        def to_table(self) -> Table:
            return self._table

    class FakeTapService:
        def search(self, query: str | None = None):
            q = query or ""
            if "TAP_SCHEMA.columns" in q:
                return FakeResult(
                    Table(
                        {
                            "column_name": [
                                "source_id",
                                "teff50",
                                "logg50",
                                "met50",
                                "dist50",
                                "av50",
                                "mass50",
                            ]
                        }
                    )
                )

            stats["data_queries"] = int(stats["data_queries"]) + 1
            stats["last_data_query"] = q
            ids_str = q.split("IN (", 1)[1].rsplit(")", 1)[0]
            ids = [tok.strip() for tok in ids_str.split(",") if tok.strip()]
            table = Table(
                {
                    "source_id": [int(x) for x in ids],
                    "teff50": [5000.0] * len(ids),
                    "logg50": [4.3] * len(ids),
                    "met50": [-0.1] * len(ids),
                    "dist50": [1000.0] * len(ids),
                    "av50": [0.2] * len(ids),
                    "mass50": [1.0] * len(ids),
                }
            )
            return FakeResult(table)

    monkeypatch.setattr(characterize.pyvo.dal, "TAPService", lambda *_a, **_k: FakeTapService())

    first = characterize.query_starhorse_by_ids(
        ["1", "2"],
        use_tap=True,
        cache_file=cache_path,
    )
    assert len(first) == 2
    assert cache_path.exists()
    assert int(stats["data_queries"]) == 1
    assert "age50" not in str(stats["last_data_query"])

    second = characterize.query_starhorse_by_ids(
        ["1", "2"],
        use_tap=True,
        cache_file=cache_path,
    )
    assert len(second) == 2
    # Fully served from cache on second call.
    assert int(stats["data_queries"]) == 1


def test_open_cluster_crossmatch_uses_metadata_mapping(monkeypatch) -> None:
    meta = pd.DataFrame(
        {
            "cluster_name": ["MyCluster"],
            "cluster_age_myr": [123.0],
            "cluster_dist_pc": [456.0],
        }
    )

    def fake_load_meta(_cache_file=None):
        return meta

    def fake_xmatch_query(**kwargs):
        _ = kwargs
        return Table(
            {
                "_idx": [0],
                "Cluster": ["MyCluster"],
                "angDist": [0.2],
            }
        )

    monkeypatch.setattr(characterize, "_load_open_cluster_metadata", fake_load_meta)
    monkeypatch.setattr(characterize.XMatch, "query", fake_xmatch_query)

    df = pd.DataFrame({"ra": [10.0, 20.0], "dec": [1.0, -1.0]})
    out = characterize.crossmatch_open_clusters(df)

    assert out.loc[0, "cluster_name"] == "MyCluster"
    assert out.loc[0, "cluster_age_myr"] == 123.0
    assert out.loc[0, "cluster_dist_pc"] == 456.0
    assert out.loc[1, "cluster_name"] == ""


def test_unwise_variability_resumes_from_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "unwise_ckpt.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "unwise_w1_zscore": [1.0, 2.0],
            "unwise_w2_zscore": [0.5, 0.7],
            "unwise_w1_var": [False, False],
        }
    ).to_parquet(checkpoint_path, index=False)

    calls: list[str] = []

    def fake_query_single(candidate_id: str, ra: float, dec: float, *, max_sep_arcsec: float, max_retries: int):
        _ = (ra, dec, max_sep_arcsec, max_retries)
        calls.append(candidate_id)
        return {
            "candidate_id": candidate_id,
            "unwise_w1_zscore": 4.2,
            "unwise_w2_zscore": 1.1,
            "unwise_w1_var": True,
        }

    monkeypatch.setattr(characterize, "_query_unwise_single", fake_query_single)

    df = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3", "c4"],
            "ra": [10.0, 11.0, 12.0, 13.0],
            "dec": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = characterize.query_unwise_variability(
        df,
        workers=2,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )

    assert set(calls) == {"c3", "c4"}
    assert out["unwise_w1_zscore"].notna().sum() == 4
    assert int(out["unwise_w1_var"].sum()) == 2

    ckpt = pd.read_parquet(checkpoint_path)
    assert len(ckpt) == 4
