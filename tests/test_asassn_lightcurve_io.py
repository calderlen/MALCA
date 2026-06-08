from __future__ import annotations

import numpy as np
import pandas as pd

from malca.lightcurve_io import (
    CANONICAL_ASASSN_COLUMNS,
    AB_ZERO_POINT_JY,
    V_VEGA_ZERO_POINT_JY,
    filter_asassn_quality,
    load_lightcurve_df,
    normalize_asassn_lightcurve,
    to_asassn_algorithm_frame,
)


def test_load_skypatrol_csv_returns_canonical_schema(tmp_path):
    path = tmp_path / "ASASSN-test.csv"
    path.write_text(
        "JD,Flux,Flux Error,Mag,Mag Error,Limit,FWHM,Filter,Quality,Camera\n"
        "2459000.5,1.2,0.1,15.0,0.02,16.0,2.5,g,G,ba\n"
        "2459001.5,2.0,0.2,14.5,0.03,16.1,2.7,V,B,bb\n"
        "2459002.5,0.8,0.1,,,16.2,2.8,g,G,bc\n",
        encoding="utf-8",
    )

    df = load_lightcurve_df(path)

    assert list(df.columns) == CANONICAL_ASASSN_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "jd"] == 2459000.5
    assert df.loc[0, "mjd"] == 59000.0
    assert df.loc[0, "band"] == "g"
    assert df.loc[0, "flux"] == 1.2
    assert df.loc[0, "flux_density_mjy"] == 1.2
    assert df.loc[0, "flux_provenance"] == "asassn_survey_flux"
    assert df.loc[0, "quality"] == "G"
    assert df.loc[0, "camera"] == "ba"
    assert df.loc[0, "camera_name"] == "ba"
    assert df.loc[0, "field"] == ""
    assert df.loc[0, "source_path"] == str(path)
    assert pd.isna(df.loc[1, "mag"])
    assert df.loc[1, "flux"] == 0.8
    assert df.loc[1, "flux_density_mjy"] == 0.8

    unfiltered = load_lightcurve_df(path, apply_quality=False)
    assert len(unfiltered) == 3
    assert unfiltered["is_good"].tolist() == [True, False, True]


def test_load_dat_preserves_camera_field_split(tmp_path):
    path = tmp_path / "123.dat2"
    path.write_text(
        "\n".join(
            [
                "7479.8 14.10 0.02 1 4 0 0 ba/F1",
                "7480.8 14.20 0.03 1 5 1 0 bb/F2",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    df = load_lightcurve_df(path, apply_quality=False)

    assert df["camera"].tolist() == ["4", "5"]
    assert df["camera_name"].tolist() == ["ba", "bb"]
    assert df["field"].tolist() == ["F1", "F2"]
    assert "camera_field" not in df.columns


def test_field_only_camera_field_normalizes_to_field():
    raw = pd.DataFrame(
        {
            "jd": [2459000.5],
            "band": ["g"],
            "mag": [15.0],
            "mag_err": [0.02],
            "quality": ["G"],
            "camera": ["4"],
            "saturated": [0],
            "camera_field": ["F1"],
        }
    )

    df = normalize_asassn_lightcurve(raw, apply_quality=False)

    assert df.loc[0, "camera"] == "4"
    assert df.loc[0, "camera_name"] == ""
    assert df.loc[0, "field"] == "F1"
    assert "camera_field" not in df.columns


def test_mag_only_legacy_rows_get_labeled_flux_density_not_survey_flux():
    raw = pd.DataFrame(
        {
            "JD": [9000.5, 9001.5],
            "mag": [15.0, 15.0],
            "error": [0.02, 0.03],
            "good_bad": [0, 0],
            "camera#": [1, 2],
            "v_g_band": [0, 1],
            "saturated": [0, 0],
            "cam_field": ["1/a", "2/b"],
        }
    )

    df = normalize_asassn_lightcurve(raw)

    expected_g = AB_ZERO_POINT_JY * 1000.0 * 10.0 ** (-0.4 * 15.0)
    expected_v = V_VEGA_ZERO_POINT_JY * 1000.0 * 10.0 ** (-0.4 * 15.0)
    assert df["jd"].tolist() == [2459000.5, 2459001.5]
    assert df["flux"].isna().all()
    assert np.isclose(df.loc[0, "flux_density_mjy"], expected_g)
    assert np.isclose(df.loc[1, "flux_density_mjy"], expected_v)
    assert df["flux_provenance"].tolist() == ["mag_zero_point_g_ab", "mag_zero_point_v_vega"]
    assert np.isclose(df.loc[0, "rel_flux"], 1.0)
    assert np.isclose(df.loc[1, "rel_flux"], 1.0)


def test_quality_filter_handles_quality_saturation_and_bad_errors():
    raw = pd.DataFrame(
        {
            "jd": [2459000.5, 2459001.5, 2459002.5, 2459003.5],
            "band": ["g", "g", "g", "g"],
            "mag": [15.0, 15.1, 15.2, 15.3],
            "mag_err": [0.02, 0.03, np.nan, -0.1],
            "quality": ["G", "B", "G", "G"],
            "saturated": [0, 0, 1, 0],
        }
    )

    unfiltered = normalize_asassn_lightcurve(raw, apply_quality=False)
    filtered = filter_asassn_quality(raw)

    assert unfiltered["is_good"].tolist() == [True, False, False, False]
    assert filtered["jd"].tolist() == [2459000.5]


def test_algorithm_adapter_uses_split_camera_and_field_columns():
    canonical = pd.DataFrame(
        {
            "jd": [2459000.5, 2459001.5],
            "mjd": [59000.0, 59001.0],
            "band": ["g", "V"],
            "mag": [15.0, 14.8],
            "mag_err": [0.02, 0.03],
            "quality": ["G", "G"],
            "camera": ["ba", "bb"],
            "saturated": [False, False],
            "camera_name": ["ba", "bb"],
            "field": ["1", "2"],
        }
    )

    legacy = to_asassn_algorithm_frame(canonical)

    assert legacy.columns.tolist() == [
        "JD",
        "mag",
        "error",
        "good_bad",
        "camera#",
        "v_g_band",
        "saturated",
        "camera_name",
        "field",
    ]
    assert legacy["JD"].tolist() == [2459000.5, 2459001.5]
    assert legacy["error"].tolist() == [0.02, 0.03]
    assert legacy["good_bad"].tolist() == [1, 1]
    assert legacy["camera#"].tolist() == ["ba", "bb"]
    assert legacy["v_g_band"].tolist() == [0.0, 1.0]
    assert legacy["camera_name"].tolist() == ["ba", "bb"]
    assert legacy["field"].tolist() == ["1", "2"]
