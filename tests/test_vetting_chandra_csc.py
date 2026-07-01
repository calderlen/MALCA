from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.table import Table

import malca.enrichment.vetting as vetting


def test_chandra_csc_crossmatch_populates_source_fields_and_xray_aggregate(monkeypatch) -> None:
    df = pd.DataFrame({"ra": [10.0, 20.0], "dec": [-5.0, 15.0]})

    def fake_batch_tap_crossmatch(coords_df, **kwargs):
        assert kwargs["catalog_table"] == '"IX/70/csc21mas"'
        assert kwargs["ra_col"] == '"RAICRS"'
        assert kwargs["dec_col"] == '"DEICRS"'
        assert kwargs["match_radius_arcsec"] == 3.0
        assert "FPL0.5-7" in kwargs["select_cols"]
        assert list(coords_df["_idx"]) == [0, 1]
        return pd.DataFrame(
            [
                {
                    "_idx": 1,
                    "chandra_source_id": "2CXO Jfar",
                    "chandra_flux_05_7": 9.0e-14,
                    "chandra_flux_broad": 8.0e-14,
                    "chandra_significance": 4.0,
                    "chandra_likelihood": 11.0,
                    "chandra_likelihood_class": "TRUE",
                    "chandra_pos_err_maj_arcsec": 1.4,
                    "chandra_pos_err_min_arcsec": 0.8,
                    "chandra_pos_err_pa_deg": 20.0,
                    "chandra_extended_flag": "False",
                    "chandra_variable_flag": "False",
                    "sep_arcsec": 2.5,
                },
                {
                    "_idx": 1,
                    "chandra_source_id": "2CXO Jnear",
                    "chandra_flux_05_7": 1.2e-14,
                    "chandra_flux_broad": 1.0e-14,
                    "chandra_significance": 7.5,
                    "chandra_likelihood": 33.0,
                    "chandra_likelihood_class": "TRUE",
                    "chandra_pos_err_maj_arcsec": 0.6,
                    "chandra_pos_err_min_arcsec": 0.3,
                    "chandra_pos_err_pa_deg": 42.0,
                    "chandra_extended_flag": "True",
                    "chandra_variable_flag": "False",
                    "sep_arcsec": 0.7,
                },
            ]
        )

    monkeypatch.setattr(vetting, "batch_tap_crossmatch", fake_batch_tap_crossmatch)

    out = vetting.crossmatch_chandra_csc(df, method="tap")

    assert bool(out.loc[0, "chandra_det"]) is False
    assert bool(out.loc[0, "xray_det"]) is False
    assert bool(out.loc[1, "chandra_det"]) is True
    assert out.loc[1, "chandra_source_id"] == "2CXO Jnear"
    assert float(out.loc[1, "chandra_flux_05_7"]) == 1.2e-14
    assert float(out.loc[1, "chandra_sep_arcsec"]) == 0.7
    assert bool(out.loc[1, "chandra_extended_flag"]) is True
    assert bool(out.loc[1, "chandra_variable_flag"]) is False
    assert bool(out.loc[1, "xray_det"]) is True
    assert float(out.loc[1, "xray_flux"]) == 1.2e-14
    assert float(out.loc[1, "xray_sep_arcsec"]) == 0.7
    assert out.loc[1, "xray_source_catalogs"] == "Chandra CSC 2.1"


def test_chandra_csc_xmatch_path_maps_csc_columns(monkeypatch) -> None:
    df = pd.DataFrame({"ra": [91.64698296514], "dec": [-86.6438]})

    def fake_xmatch_query(*, cat1, cat2, max_distance, colRA1, colDec1):
        assert cat2 == "vizier:IX/70/csc21mas"
        assert colRA1 == "ra"
        assert colDec1 == "dec"
        assert list(cat1["_idx"]) == [0]
        table = Table()
        table["angDist"] = [0.536]
        table["_idx"] = [0]
        table["2CXO"] = ["2CXO J060635.2-863837"]
        table["FPL0.5-7"] = [2.0e-14]
        table["Favgb"] = [3.0e-14]
        table["signi"] = [12.0]
        table["like"] = [35.0]
        table["likeClass"] = ["TRUE"]
        table["r0"] = [0.7]
        table["r1"] = [0.4]
        table["PA"] = [10.0]
        table["fe"] = ["False"]
        table["fv"] = ["True"]
        return table

    monkeypatch.setattr(vetting.XMatch, "query", fake_xmatch_query)

    out = vetting.crossmatch_chandra_csc(df, method="xmatch")

    assert bool(out.loc[0, "chandra_det"]) is True
    assert out.loc[0, "chandra_source_id"] == "2CXO J060635.2-863837"
    assert float(out.loc[0, "chandra_flux_05_7"]) == 2.0e-14
    assert float(out.loc[0, "chandra_sep_arcsec"]) == 0.536
    assert bool(out.loc[0, "chandra_variable_flag"]) is True
    assert out.loc[0, "xray_source_catalogs"] == "Chandra CSC 2.1"


def test_xray_aggregate_handles_erosita_chandra_and_both_catalog_hits() -> None:
    df = pd.DataFrame(
        {
            "erosita_det": [True, False, True, True],
            "erosita_flux": [5.0e-13, np.nan, 2.0e-13, 3.0e-13],
            "erosita_sep_arcsec": [1.2, np.nan, 0.4, 2.1],
            "chandra_det": [False, True, True, True],
            "chandra_flux_05_7": [np.nan, np.nan, 8.0e-14, 4.0e-14],
            "chandra_flux_broad": [np.nan, 7.0e-14, 9.0e-14, 5.0e-14],
            "chandra_sep_arcsec": [np.nan, 0.6, 0.9, 0.2],
        }
    )

    out = vetting._sync_xray_aggregate_fields(df)

    assert bool(out.loc[0, "xray_det"]) is True
    assert float(out.loc[0, "xray_flux"]) == 5.0e-13
    assert float(out.loc[0, "xray_sep_arcsec"]) == 1.2
    assert out.loc[0, "xray_source_catalogs"] == "eROSITA"

    assert bool(out.loc[1, "xray_det"]) is True
    assert float(out.loc[1, "xray_flux"]) == 7.0e-14
    assert float(out.loc[1, "xray_sep_arcsec"]) == 0.6
    assert out.loc[1, "xray_source_catalogs"] == "Chandra CSC 2.1"

    assert bool(out.loc[2, "xray_det"]) is True
    assert float(out.loc[2, "xray_flux"]) == 2.0e-13
    assert float(out.loc[2, "xray_sep_arcsec"]) == 0.4
    assert out.loc[2, "xray_source_catalogs"] == "eROSITA,Chandra CSC 2.1"
    assert float(out.loc[2, "chandra_flux_05_7"]) == 8.0e-14

    assert bool(out.loc[3, "xray_det"]) is True
    assert float(out.loc[3, "xray_flux"]) == 4.0e-14
    assert float(out.loc[3, "xray_sep_arcsec"]) == 0.2
    assert out.loc[3, "xray_source_catalogs"] == "eROSITA,Chandra CSC 2.1"
