from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

post_filter = pytest.importorskip("malca.filter")


def _empty_catalog() -> pd.DataFrame:
    return pd.DataFrame(columns=["gaia_id", "ra", "dec", "period", "var_type"])


def test_fetch_gaia_dr3_eb_periods_caches_negative_lookups(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeTap:
        def __init__(self, _url: str) -> None:
            pass

        def run_sync(self, query: str) -> list[dict[str, object]]:
            calls.append(query)
            return [
                {
                    "source_id": 1,
                    "frequency": 0.5,
                    "model_type": "ECL",
                    "global_ranking": 0.8,
                }
            ]

    monkeypatch.setattr(post_filter.pyvo.dal, "TAPService", FakeTap)

    first = post_filter.fetch_gaia_dr3_eb_periods(
        [1, 2],
        cache_dir=tmp_path,
        chunk_size=2,
        show_tqdm=False,
    )
    second = post_filter.fetch_gaia_dr3_eb_periods(
        [1, 2],
        cache_dir=tmp_path,
        chunk_size=2,
        show_tqdm=False,
    )

    assert len(calls) == 1
    assert first["source_id"].astype(int).tolist() == [1]
    assert second["source_id"].astype(int).tolist() == [1]

    cache = pd.read_parquet(tmp_path / "gaia_dr3_eb_periods.parquet")
    by_id = cache.set_index(cache["source_id"].astype(int))
    assert set(by_id.index) == {1, 2}
    assert bool(by_id.loc[1, "matched"]) is True
    assert bool(by_id.loc[2, "matched"]) is False


def test_validate_periodic_catalog_builds_multisource_consensus(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "path": ["/tmp/1001.dat2", "/tmp/2002.dat2"],
            "gaia_id": ["1", "2"],
            "ra_deg": [10.0, 20.0],
            "dec_deg": [-5.0, 1.0],
        }
    )

    monkeypatch.setattr(
        post_filter,
        "fetch_gaia_dr3_eb_periods",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "source_id": [1],
                "period": [2.0],
                "var_type": ["ECL"],
                "global_ranking": [0.8],
            }
        ),
    )
    monkeypatch.setattr(
        post_filter,
        "fetch_asassn_variable_catalog",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "gaia_id": [1],
                "ra": [10.0],
                "dec": [-5.0],
                "period": [4.0],
                "var_type": ["ROT"],
            }
        ),
    )
    monkeypatch.setattr(
        post_filter,
        "fetch_chen2020_ztf_periodic",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "gaia_id": [1],
                "ra": [10.0],
                "dec": [-5.0],
                "period": [2.02],
                "var_type": ["BY"],
            }
        ),
    )
    monkeypatch.setattr(
        post_filter,
        "fetch_vsx_period_catalog",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "asas_sn_id": ["1001"],
                "period": [2.01],
                "var_type": ["VAR"],
                "vsx_sep_arcsec": [0.2],
            }
        ),
    )
    monkeypatch.setattr(post_filter, "fetch_ogle_periodic_catalog", lambda *args, **kwargs: _empty_catalog())

    out = post_filter.validate_periodic_catalog(df, show_tqdm=False)
    by_path = out.set_index("path")

    row0 = by_path.loc["/tmp/1001.dat2"]
    assert bool(row0["catalog_match"]) is True
    assert int(row0["period_n_sources"]) == 4
    assert bool(row0["period_consensus_agree"]) is True
    assert bool(row0["period_conflict_flag"]) is False
    assert str(row0["period_sources"]) == "gaia_eb|vsx|asassn_var|ztf_periodic"
    assert str(row0["catalog_source"]) == "gaia_eb"
    assert float(row0["catalog_period"]) == pytest.approx(2.005, rel=1e-3)

    row1 = by_path.loc["/tmp/2002.dat2"]
    assert bool(row1["catalog_match"]) is False
    assert int(row1["period_n_sources"]) == 0
    assert bool(row1["period_conflict_flag"]) is False


def test_validate_periodic_catalog_flags_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "path": ["/tmp/1001.dat2"],
            "gaia_id": ["1"],
            "ra_deg": [10.0],
            "dec_deg": [-5.0],
        }
    )

    monkeypatch.setattr(
        post_filter,
        "fetch_gaia_dr3_eb_periods",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "source_id": [1],
                "period": [2.0],
                "var_type": ["ECL"],
                "global_ranking": [0.8],
            }
        ),
    )
    monkeypatch.setattr(
        post_filter,
        "fetch_asassn_variable_catalog",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "gaia_id": [1],
                "ra": [10.0],
                "dec": [-5.0],
                "period": [5.0],
                "var_type": ["ROT"],
            }
        ),
    )
    monkeypatch.setattr(post_filter, "fetch_chen2020_ztf_periodic", lambda *args, **kwargs: _empty_catalog())
    monkeypatch.setattr(post_filter, "fetch_vsx_period_catalog", lambda *args, **kwargs: pd.DataFrame(columns=["asas_sn_id", "period", "var_type"]))
    monkeypatch.setattr(post_filter, "fetch_ogle_periodic_catalog", lambda *args, **kwargs: _empty_catalog())

    out = post_filter.validate_periodic_catalog(df, show_tqdm=False)
    row = out.iloc[0]

    assert bool(row["catalog_match"]) is True
    assert int(row["period_n_sources"]) == 2
    assert bool(row["period_consensus_agree"]) is False
    assert bool(row["period_conflict_flag"]) is True


def test_validate_periodic_catalog_overwrites_existing_output_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "path": ["/tmp/1001.dat2"],
            "gaia_id": ["1"],
            "ra_deg": [10.0],
            "dec_deg": [-5.0],
            "catalog_match": [False],
            "catalog_source": ["stale"],
            "period_sources": ["stale"],
            "period_n_sources": [99],
            "period_gaia_eb_match": [False],
            "period_gaia_eb_days": [99.0],
            "period_gaia_eb_class": ["OLD"],
            "period_gaia_eb_sep_arcsec": [99.0],
        }
    )

    monkeypatch.setattr(
        post_filter,
        "fetch_gaia_dr3_eb_periods",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "source_id": [1],
                "period": [2.0],
                "var_type": ["ECL"],
                "global_ranking": [0.8],
            }
        ),
    )
    monkeypatch.setattr(post_filter, "fetch_asassn_variable_catalog", lambda *args, **kwargs: _empty_catalog())
    monkeypatch.setattr(post_filter, "fetch_chen2020_ztf_periodic", lambda *args, **kwargs: _empty_catalog())
    monkeypatch.setattr(post_filter, "fetch_vsx_period_catalog", lambda *args, **kwargs: pd.DataFrame(columns=["asas_sn_id", "period", "var_type"]))
    monkeypatch.setattr(post_filter, "fetch_ogle_periodic_catalog", lambda *args, **kwargs: _empty_catalog())

    out = post_filter.validate_periodic_catalog(df, show_tqdm=False)

    assert out.columns.tolist().count("period_gaia_eb_days") == 1
    row = out.iloc[0]
    assert bool(row["catalog_match"]) is True
    assert str(row["catalog_source"]) == "gaia_eb"
    assert str(row["period_sources"]) == "gaia_eb"
    assert int(row["period_n_sources"]) == 1
    assert float(row["period_gaia_eb_days"]) == pytest.approx(2.0)


def test_apply_filters_home_validations_only_check_upstream_passers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "path": ["/tmp/a.dat2", "/tmp/b.dat2", "/tmp/c.dat2", "/tmp/d.dat2"],
            "gaia_id": ["1", "2", "3", "4"],
            "failed_posterior_strength": [False, True, False, False],
            "failed_periodic_catalog": [True, False, False, False],
            "failed_gaia_ruwe": [True, False, False, False],
            "failed_gaia_pm": [False, False, False, False],
            "failed_any": [True, True, False, False],
            "catalog_match": [True, True, False, False],
            "catalog_source": ["stale_a", "stale_b", "", ""],
            "period_sources": ["gaia_eb", "gaia_eb", "", ""],
            "period_n_sources": [1, 1, 0, 0],
            "ruwe": [9.9, 8.8, np.nan, np.nan],
            "high_ruwe_flag": [True, True, False, False],
            "pmra": [1.0, 2.0, np.nan, np.nan],
            "pmdec": [1.0, 2.0, np.nan, np.nan],
            "pm_total": [1.4, 2.8, np.nan, np.nan],
            "high_pm_flag": [False, True, False, False],
        }
    )

    checked: dict[str, list[str]] = {}

    def fake_periodic_catalog(subset: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        checked["periodic_catalog"] = subset["path"].tolist()
        out = subset.copy()
        out["catalog_match"] = False
        out["catalog_period"] = np.nan
        out["catalog_class"] = ""
        out["catalog_source"] = ""
        out["period_sources"] = ""
        out["period_n_sources"] = 0
        out["period_consensus_days"] = np.nan
        out["period_consensus_agree"] = False
        out["period_conflict_flag"] = False
        out["period_consensus_support"] = np.nan
        out["period_primary_source"] = ""
        out["period_source_periods"] = ""
        for src in ("gaia_eb", "vsx", "asassn_var", "ztf_periodic", "ogle"):
            out[f"period_{src}_match"] = False
            out[f"period_{src}_days"] = np.nan
            out[f"period_{src}_class"] = ""
            out[f"period_{src}_sep_arcsec"] = np.nan
        return out

    def fake_gaia_ruwe(subset: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        checked["gaia_ruwe"] = subset["path"].tolist()
        out = subset.copy()
        out["ruwe"] = [1.0, 2.2, 1.1]
        out["high_ruwe_flag"] = [False, True, False]
        return out

    def fake_gaia_pm(subset: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        checked["gaia_pm"] = subset["path"].tolist()
        out = subset.copy()
        out["pmra"] = [1.0, 2.0, 120.0]
        out["pmdec"] = [1.0, 2.0, 0.0]
        out["pm_total"] = [1.4, 2.8, 120.0]
        out["high_pm_flag"] = [False, False, True]
        return out

    monkeypatch.setattr(post_filter, "validate_periodic_catalog", fake_periodic_catalog)
    monkeypatch.setattr(post_filter, "validate_gaia_ruwe", fake_gaia_ruwe)
    monkeypatch.setattr(post_filter, "validate_gaia_proper_motion", fake_gaia_pm)

    out = post_filter.apply_filters(
        df,
        apply_evidence_strength=False,
        apply_significant_detection=False,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodic_catalog_validation=True,
        periodic_catalog_flag_only=False,
        apply_gaia_ruwe_validation=True,
        gaia_flag_only=False,
        apply_gaia_pm_validation=True,
        gaia_pm_flag_only=False,
        apply_periodicity_validation=False,
        home_passers_only=True,
        show_tqdm=False,
    )

    assert checked["periodic_catalog"] == ["/tmp/a.dat2", "/tmp/c.dat2", "/tmp/d.dat2"]
    assert checked["gaia_ruwe"] == ["/tmp/a.dat2", "/tmp/c.dat2", "/tmp/d.dat2"]
    assert checked["gaia_pm"] == ["/tmp/a.dat2", "/tmp/c.dat2", "/tmp/d.dat2"]

    by_path = out.set_index("path")

    row_a = by_path.loc["/tmp/a.dat2"]
    assert bool(row_a["failed_any"]) is False
    assert bool(row_a["failed_periodic_catalog"]) is False
    assert bool(row_a["failed_gaia_ruwe"]) is False
    assert bool(row_a["failed_gaia_pm"]) is False

    row_b = by_path.loc["/tmp/b.dat2"]
    assert bool(row_b["failed_posterior_strength"]) is True
    assert bool(row_b["catalog_match"]) is False
    assert str(row_b["catalog_source"]) == ""
    assert str(row_b["period_sources"]) == ""
    assert pd.isna(row_b["ruwe"])
    assert bool(row_b["high_ruwe_flag"]) is False
    assert pd.isna(row_b["pm_total"])
    assert bool(row_b["high_pm_flag"]) is False
    assert bool(row_b["failed_periodic_catalog"]) is False
    assert bool(row_b["failed_gaia_ruwe"]) is False
    assert bool(row_b["failed_gaia_pm"]) is False

    row_c = by_path.loc["/tmp/c.dat2"]
    assert bool(row_c["failed_gaia_ruwe"]) is True
    assert bool(row_c["failed_any"]) is True

    row_d = by_path.loc["/tmp/d.dat2"]
    assert bool(row_d["failed_gaia_pm"]) is True
    assert bool(row_d["failed_any"]) is True
