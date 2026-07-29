from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from malca.ltv.multi_survey import (
    LTV_MS_FEATURE_VERSION,
    compute_ltv_multi_survey_features,
)
from malca.ltv.review import ingest_ltv_results


def test_ltv_multi_survey_features_summarize_external_lc_files(tmp_path: Path) -> None:
    external_dir = tmp_path / "external_lcs"
    external_dir.mkdir()
    pd.DataFrame(
        {
            "MJD": [59000.0, 59365.25, 59400.0],
            "m": [-20.0, 99.0, -99.0],
            "dm": [99.0, 99.0, 99.0],
            "uJy": [100.0, 50.0, -1.0],
            "duJy": [10.0, 5.0, 1.0],
            "F": ["c", "c", "c"],
            "err": [0, 0, 0],
            "x": [500.0, 500.0, 500.0],
            "y": [500.0, 500.0, 500.0],
            "maj": [2.5, 2.5, 2.5],
            "min": [2.2, 2.2, 2.2],
            "apfit": [-0.5, -0.5, -0.5],
            "mag5sig": [19.0, 19.0, 19.0],
            "Sky": [20.0, 20.0, 20.0],
            "atlas_image_type": ["reduced", "reduced", "reduced"],
        }
    ).to_parquet(external_dir / "atlas_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "jd": [2450000.0, 2450365.25],
            "mag": [13.0, 14.0],
            "band": ["zg", "zg"],
        }
    ).to_parquet(external_dir / "ztf_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "mjd": [58000.0, 58365.25],
            "w1mpro": [10.0, 11.0],
            "w2mpro": [9.5, 9.0],
        }
    ).to_parquet(external_dir / "neowise_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "mjd": [55400.0, 55765.25],
            "w1mpro": [12.0, 12.5],
            "w3mpro": [8.0, 8.4],
        }
    ).to_parquet(external_dir / "allwise_mep_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "time": [100.0, 465.25],
            "flux": [1.0, 1.2],
        }
    ).to_parquet(external_dir / "kepler_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "mjd": [59000.0, 59365.25],
            "mag": [13.0, 13.5],
            "band": ["V", "V"],
        }
    ).to_parquet(external_dir / "aavso_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "mjd": [57000.0, 57365.25],
            "mag": [15.0, 15.4],
            "band": ["I", "I"],
        }
    ).to_parquet(external_dir / "ogle_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "mjd": [52000.0, 52365.25],
            "mag": [18.0, 17.8],
            "band": ["g", "g"],
        }
    ).to_parquet(external_dir / "stripe82_lc_ltv_123.parquet", index=False)
    pd.DataFrame(
        {
            "mjd": [57000.0, 57365.25],
            "mag": [14.0, 14.7],
            "band": ["ks", "ks"],
        }
    ).to_parquet(external_dir / "vvvx_virac_lc_ltv_123.parquet", index=False)

    out = compute_ltv_multi_survey_features(
        pd.DataFrame({"asas_sn_id": ["123"], "candidate_id": ["ltv_123"]}),
        external_lc_dir=external_dir,
    )

    assert out.loc[0, "ltv_ms_feature_status"] == "ok"
    assert out.loc[0, "ltv_ms_feature_version"] == LTV_MS_FEATURE_VERSION == "2"
    assert out.loc[0, "ltv_ms_atlas_n_points"] == 2
    assert out.loc[0, "ltv_ms_atlas_time_span_days"] == 365.25
    assert round(float(out.loc[0, "ltv_ms_atlas_mag_range"]), 6) == 0.752575
    assert round(float(out.loc[0, "ltv_ms_atlas_mag_slope_per_year"]), 6) == 0.752575
    assert out.loc[0, "ltv_ms_ztf_n_points"] == 2
    assert out.loc[0, "ltv_ms_ztf_time_span_days"] == 365.25
    assert out.loc[0, "ltv_ms_ztf_mag_range"] == 1.0
    assert round(float(out.loc[0, "ltv_ms_ztf_mag_slope_per_year"]), 6) == 1.0
    assert out.loc[0, "ltv_ms_neowise_n_points"] == 2
    assert out.loc[0, "ltv_ms_neowise_w1_range"] == 1.0
    assert round(float(out.loc[0, "ltv_ms_neowise_w1_slope_per_year"]), 6) == 1.0
    assert out.loc[0, "ltv_ms_allwise_mep_n_points"] == 2
    assert out.loc[0, "ltv_ms_allwise_mep_w1_range"] == 0.5
    assert out.loc[0, "ltv_ms_kepler_n_points"] == 2
    assert round(float(out.loc[0, "ltv_ms_kepler_flux_range"]), 6) == 0.2
    assert out.loc[0, "ltv_ms_aavso_n_points"] == 2
    assert out.loc[0, "ltv_ms_aavso_mag_range"] == 0.5
    assert out.loc[0, "ltv_ms_ogle_n_points"] == 2
    assert round(float(out.loc[0, "ltv_ms_ogle_mag_range"]), 6) == 0.4
    assert out.loc[0, "ltv_ms_stripe82_n_points"] == 2
    assert round(float(out.loc[0, "ltv_ms_stripe82_mag_slope_per_year"]), 6) == -0.2
    assert out.loc[0, "ltv_ms_vvvx_virac_n_points"] == 2
    assert round(float(out.loc[0, "ltv_ms_vvvx_virac_ks_range"]), 6) == 0.7


def test_ltv_multi_survey_columns_persist_in_review_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    ingest_ltv_results(
        db_path,
        pd.DataFrame(
            {
                "asas_sn_id": ["123"],
                "candidate_id": ["ltv_123"],
                "timescale": ["ltv"],
                "lc_path": [str(tmp_path / "123.dat2")],
                "ra": [1.0],
                "dec": [2.0],
                "ltv_ms_feature_status": ["ok"],
                "ltv_ms_ztf_n_points": [2],
                "ltv_ms_ztf_mag_range": [1.0],
            }
        ),
        run_characterize=False,
        run_stats=False,
        verbose=False,
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "select ltv_ms_feature_status, ltv_ms_ztf_n_points, ltv_ms_ztf_mag_range from candidates"
        ).fetchone()

    assert row == ("ok", 2, 1.0)
