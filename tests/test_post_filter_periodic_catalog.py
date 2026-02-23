from __future__ import annotations

import pandas as pd
import pytest

post_filter = pytest.importorskip("malca.post_filter")


def _empty_catalog() -> pd.DataFrame:
    return pd.DataFrame(columns=["gaia_id", "ra", "dec", "period", "var_type"])


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
