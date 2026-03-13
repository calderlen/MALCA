from __future__ import annotations

import pandas as pd

from malca.ltv import filter as ltv_filter


def test_query_gaia_proper_motions_batch_keeps_target_gaia_match(monkeypatch) -> None:
    def _fake_batch_gaia_cone_query(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "_idx": [0],
                "source_id": [123456789],
                "pmra": [3.0],
                "pmdec": [4.0],
                "sep_arcsec": [0.2],
            }
        )

    monkeypatch.setattr(ltv_filter, "batch_gaia_cone_query", _fake_batch_gaia_cone_query)

    df = pd.DataFrame({"ra_deg": [10.0], "dec_deg": [-5.0]})
    out = ltv_filter.query_gaia_proper_motions_batch(df, verbose=False)

    assert out.loc[0, "gaia_source_id"] == 123456789
    assert out.loc[0, "gaia_sep_arcsec"] == 0.2
    assert out.loc[0, "gaia_pm_total"] == 5.0


def test_query_neighbor_high_pm_batch_skips_exact_self_match(monkeypatch) -> None:
    def _fake_batch_gaia_cone_query(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "_idx": [0, 0],
                "source_id": [111, 222],
                "ra": [100.0, 100.02],
                "dec": [20.0, 20.0],
                "pmra": [10.0, 0.0],
                "pmdec": [0.0, 0.0],
                "phot_g_mean_mag": [12.0, 12.0],
                "sep_arcsec": [0.001, 67.7],
            }
        )

    monkeypatch.setattr(ltv_filter, "batch_gaia_cone_query", _fake_batch_gaia_cone_query)

    df = pd.DataFrame(
        {
            "ra_deg": [100.0],
            "dec_deg": [20.0],
            "Pstarss gmag": [12.2],
            "gaia_source_id": pd.Series([111], dtype="Int64"),
        }
    )

    out = ltv_filter.query_neighbor_high_pm_batch(df, verbose=False)

    assert bool(out.loc[0, "neighbor_pm_contam"]) is False


def test_query_neighbor_high_pm_batch_uses_tiny_sep_fallback_without_target_id(monkeypatch) -> None:
    def _fake_batch_gaia_cone_query(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "_idx": [0, 0],
                "source_id": [111, 222],
                "ra": [100.0, 100.02],
                "dec": [20.0, 20.0],
                "pmra": [10.0, 0.0],
                "pmdec": [0.0, 0.0],
                "phot_g_mean_mag": [12.0, 12.0],
                "sep_arcsec": [0.001, 67.7],
            }
        )

    monkeypatch.setattr(ltv_filter, "batch_gaia_cone_query", _fake_batch_gaia_cone_query)

    df = pd.DataFrame(
        {
            "ra_deg": [100.0],
            "dec_deg": [20.0],
            "Pstarss gmag": [12.2],
        }
    )

    out = ltv_filter.query_neighbor_high_pm_batch(df, verbose=False)

    assert bool(out.loc[0, "neighbor_pm_contam"]) is False


def test_query_neighbor_high_pm_batch_flags_real_close_neighbor(monkeypatch) -> None:
    def _fake_batch_gaia_cone_query(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "_idx": [0, 0],
                "source_id": [111, 222],
                "ra": [100.0, 100.0055555556],
                "dec": [20.0, 20.0],
                "pmra": [10.0, -2000.0],
                "pmdec": [0.0, 0.0],
                "phot_g_mean_mag": [12.0, 12.5],
                "sep_arcsec": [0.001, 18.8],
            }
        )

    monkeypatch.setattr(ltv_filter, "batch_gaia_cone_query", _fake_batch_gaia_cone_query)

    df = pd.DataFrame(
        {
            "ra_deg": [100.0],
            "dec_deg": [20.0],
            "Pstarss gmag": [12.2],
            "gaia_source_id": pd.Series([111], dtype="Int64"),
        }
    )

    out = ltv_filter.query_neighbor_high_pm_batch(df, verbose=False)

    assert bool(out.loc[0, "neighbor_pm_contam"]) is True


def test_apply_all_filters_does_not_use_vg_overlap(monkeypatch) -> None:
    called = {"vg": False}

    def _fake_vg_overlap(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        called["vg"] = True
        return df

    monkeypatch.setattr(ltv_filter, "filter_vg_overlap", _fake_vg_overlap)

    df = pd.DataFrame({"dummy": [1]})
    out = ltv_filter.apply_all_filters(
        df,
        query_gaia=False,
        run_neighbor_pm_filter=False,
        verbose=False,
    )

    assert called["vg"] is False
    assert out.equals(df)
