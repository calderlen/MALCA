from __future__ import annotations

import pandas as pd

import malca.ltv.crossmatch as ltv_crossmatch


def test_ltv_batch_tap_crossmatch_delegates_to_shared_helper(monkeypatch) -> None:
    coords = pd.DataFrame({"_idx": [0], "ra": [10.0], "dec": [-5.0]})
    expected = pd.DataFrame({"_idx": [0], "main_id": ["SIMBAD 1"], "sep_arcsec": [0.2]})
    captured: dict[str, object] = {}

    def fake_batch_tap_crossmatch(coords_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        captured["coords_df"] = coords_df.copy()
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(ltv_crossmatch, "shared_batch_tap_crossmatch", fake_batch_tap_crossmatch)

    result = ltv_crossmatch._batch_tap_crossmatch(
        coords,
        tap_service="https://example.test/tap",
        catalog_table="basic",
        select_cols="c.main_id",
        ra_col="ra",
        dec_col="dec",
        match_radius_arcsec=2.5,
        chunk_size=123,
        n_workers=7,
        verbose=True,
        desc="SIMBAD TAP",
    )

    assert result.equals(expected)
    assert captured["coords_df"].equals(coords)
    assert captured["kwargs"] == {
        "tap_url": "https://example.test/tap",
        "catalog_table": "basic",
        "select_cols": "c.main_id",
        "ra_col": "ra",
        "dec_col": "dec",
        "match_radius_arcsec": 2.5,
        "chunk_size": 123,
        "n_workers": 7,
        "verbose": True,
        "desc": "SIMBAD TAP",
    }
