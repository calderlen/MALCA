from __future__ import annotations

import pandas as pd

from malca.feature_layers import with_feature_columns
from malca.ltv.review import map_ltv_columns
from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.metadata import REVIEW_METADATA_GROUPS
from malca.review.store import _COL_NAMES


def _canonical_base(asas_sn_id: int) -> dict[str, list[object]]:
    return {
        "asas_sn_id": [asas_sn_id],
        "lc_path": [f"/tmp/{asas_sn_id}.dat2"],
        "ra": [1.0],
        "dec": [2.0],
    }


def test_map_ltv_columns_preserves_stochastic_outputs() -> None:
    df = pd.DataFrame(
        {
            **_canonical_base(123),
            "ltv_stoch_sf_ml_amplitude": [0.21],
            "ltv_stoch_sf_ml_gamma": [0.54],
            "ltv_stoch_iar_phi": [0.87],
            "ltv_stoch_mhps_high": [1.2],
            "ltv_stoch_mhps_low": [2.3],
            "ltv_stoch_mhps_non_zero": [12.0],
            "ltv_stoch_mhps_pn_flag": [1.0],
            "ltv_stoch_mhps_ratio": [1.9],
            "ltv_stoch_gp_drw_sigma": [0.04],
            "ltv_stoch_gp_drw_tau": [140.0],
        }
    )

    out = map_ltv_columns(df)
    view = with_feature_columns(
        out,
        [
            "ltv_stoch_sf_ml_amplitude",
            "ltv_stoch_sf_ml_gamma",
            "ltv_stoch_iar_phi",
            "ltv_stoch_mhps_ratio",
            "ltv_stoch_gp_drw_tau",
            "ltv_stoch_mhps_pn_flag",
        ],
    )

    assert out.loc[0, "candidate_id"] == "ltv_123"
    assert view.loc[0, "ltv_stoch_sf_ml_amplitude"] == 0.21
    assert view.loc[0, "ltv_stoch_sf_ml_gamma"] == 0.54
    assert view.loc[0, "ltv_stoch_iar_phi"] == 0.87
    assert view.loc[0, "ltv_stoch_mhps_ratio"] == 1.9
    assert view.loc[0, "ltv_stoch_gp_drw_tau"] == 140.0
    assert view.loc[0, "ltv_stoch_mhps_pn_flag"] == 1
    assert out.loc[0, "timescale"] == "ltv"


def test_map_ltv_columns_preserves_extended_core_outputs() -> None:
    canonical_cols = {
        "ltv_slope": [0.4],
        "ltv_slope_quad": [0.01],
        "ltv_max_diff": [0.6],
        "ltv_dispersion": [0.02],
        "ltv_median": [13.0],
        "ltv_median_err": [0.02],
        "ltv_n_seasons": [4],
        "ltv_ls_period": [22.0],
        "ltv_ls_power": [0.7],
        "ltv_ls_fap": [0.01],
        "ltv_vg_has_v": [1],
    }
    df = pd.DataFrame({**_canonical_base(456), **canonical_cols})

    out = map_ltv_columns(df)
    view = with_feature_columns(out, canonical_cols)

    assert out.loc[0, "candidate_id"] == "ltv_456"
    for dst in canonical_cols:
        assert dst in view.columns
    assert view.loc[0, "ltv_vg_has_v"] == 1


def test_map_ltv_columns_derives_pm_total_when_missing() -> None:
    df = pd.DataFrame(
        {
            **_canonical_base(789),
            "pmra": [3.0],
            "pmdec": [4.0],
        }
    )

    out = map_ltv_columns(df)
    view = with_feature_columns(out, ["pmra", "pmdec", "pm_total", "high_pm_flag"])

    assert view.loc[0, "pmra"] == 3.0
    assert view.loc[0, "pmdec"] == 4.0
    assert view.loc[0, "pm_total"] == 5.0
    assert view.loc[0, "high_pm_flag"] == 0


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


def test_review_schema_and_ui_include_ltv_long_term_feature_columns() -> None:
    expected = {
        "ltv_smooth_p95_p5",
        "ltv_smooth_var",
        "ltv_resid_var",
        "ltv_long_short_var_ratio",
        "ltv_smooth_n_points",
        "ltv_smooth_100d_p95_p5",
        "ltv_smooth_100d_smooth_var",
        "ltv_smooth_100d_resid_var",
        "ltv_smooth_100d_long_short_var_ratio",
        "ltv_smooth_100d_n_points",
        "ltv_smooth_300d_p95_p5",
        "ltv_smooth_300d_smooth_var",
        "ltv_smooth_300d_resid_var",
        "ltv_smooth_300d_long_short_var_ratio",
        "ltv_smooth_300d_n_points",
        "ltv_smooth_1000d_p95_p5",
        "ltv_smooth_1000d_smooth_var",
        "ltv_smooth_1000d_resid_var",
        "ltv_smooth_1000d_long_short_var_ratio",
        "ltv_smooth_1000d_n_points",
        "ltv_binned_sf_n_bins",
        "ltv_binned_sf_30d_mag2",
        "ltv_binned_sf_100d_mag2",
        "ltv_binned_sf_300d_mag2",
        "ltv_binned_sf_1000d_mag2",
        "ltv_binned_sf_3000d_mag2",
        "ltv_binned_sf_300d_30d_ratio",
        "ltv_binned_sf_1000d_30d_ratio",
        "ltv_binned_sf_3000d_30d_ratio",
        "ltv_binned_sf_slope",
        "ltv_theil_sen_slope_mag_per_year",
        "ltv_theil_sen_intercept_mag",
        "ltv_theil_sen_low_slope_mag_per_year",
        "ltv_theil_sen_high_slope_mag_per_year",
        "ltv_bb_n_blocks",
        "ltv_bb_n_change_points",
        "ltv_bb_range_mag",
        "ltv_bb_largest_jump_mag",
        "ltv_bb_max_block_offset_mag",
        "ltv_lowess_p95_p5",
        "ltv_lowess_resid_std",
        "ltv_lowess_max_abs_resid",
        "ltv_variogram_short_mag2",
        "ltv_variogram_mid_mag2",
        "ltv_variogram_long_mag2",
        "ltv_variogram_long_short_ratio",
        "ltv_variogram_slope",
    }

    assert expected.issubset(set(_COL_NAMES))

    sidebar_cols = {
        col
        for group_name, items in SIDEBAR_GROUPS
        if group_name == "LTV Long-Term Features"
        for _ftype, col in items
    }
    assert expected == sidebar_cols

    metadata_cols = {
        key
        for group_name, fields in REVIEW_METADATA_GROUPS
        if group_name == "LTV Long-Term Features"
        for _label, key in fields
    }
    assert expected == metadata_cols
