from __future__ import annotations

import pandas as pd

from malca.ltv.review import map_ltv_columns
from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.metadata import REVIEW_METADATA_GROUPS
from malca.review.store import _COL_NAMES


def test_map_ltv_columns_renames_stochastic_outputs() -> None:
    df = pd.DataFrame(
        {
            "ASAS-SN ID": [123],
            "stoch_sf_ml_amplitude": [0.21],
            "stoch_sf_ml_gamma": [0.54],
            "stoch_iar_phi": [0.87],
            "stoch_mhps_high": [1.2],
            "stoch_mhps_low": [2.3],
            "stoch_mhps_non_zero": [12.0],
            "stoch_mhps_pn_flag": [1.0],
            "stoch_mhps_ratio": [1.9],
            "stoch_gp_drw_sigma": [0.04],
            "stoch_gp_drw_tau": [140.0],
        }
    )

    out = map_ltv_columns(df)

    assert out.loc[0, "candidate_id"] == "ltv_123"
    assert out.loc[0, "ltv_stoch_sf_ml_amplitude"] == 0.21
    assert out.loc[0, "ltv_stoch_sf_ml_gamma"] == 0.54
    assert out.loc[0, "ltv_stoch_iar_phi"] == 0.87
    assert out.loc[0, "ltv_stoch_mhps_ratio"] == 1.9
    assert out.loc[0, "ltv_stoch_gp_drw_tau"] == 140.0
    assert out.loc[0, "ltv_stoch_mhps_pn_flag"] == 1
    assert "stoch_sf_ml_amplitude" not in out.columns
    assert "stoch_mhps_pn_flag" not in out.columns


def test_review_schema_and_ui_include_ltv_stochastic_columns() -> None:
    expected = {
        "ltv_stoch_sf_ml_amplitude",
        "ltv_stoch_sf_ml_gamma",
        "ltv_stoch_iar_phi",
        "ltv_stoch_mhps_high",
        "ltv_stoch_mhps_low",
        "ltv_stoch_mhps_non_zero",
        "ltv_stoch_mhps_pn_flag",
        "ltv_stoch_mhps_ratio",
        "ltv_stoch_gp_drw_sigma",
        "ltv_stoch_gp_drw_tau",
    }

    assert expected.issubset(set(_COL_NAMES))

    sidebar_cols = {
        col
        for group_name, items in SIDEBAR_GROUPS
        if group_name == "LTV Stochastic"
        for _ftype, col in items
    }
    assert expected == sidebar_cols

    metadata_cols = {
        key
        for group_name, fields in REVIEW_METADATA_GROUPS
        if group_name == "LTV Stochastic"
        for _label, key in fields
    }
    assert expected == metadata_cols
