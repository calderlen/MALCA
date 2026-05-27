from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from malca.ltv.multi_survey import compute_ltv_multi_survey_features
from malca.ltv.review import ingest_ltv_results


def test_ltv_multi_survey_features_summarize_external_lc_files(tmp_path: Path) -> None:
    external_dir = tmp_path / "external_lcs"
    external_dir.mkdir()
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

    out = compute_ltv_multi_survey_features(
        pd.DataFrame({"asas_sn_id": ["123"], "candidate_id": ["ltv_123"]}),
        external_lc_dir=external_dir,
    )

    assert out.loc[0, "ltv_ms_feature_status"] == "ok"
    assert out.loc[0, "ltv_ms_ztf_n_points"] == 2
    assert out.loc[0, "ltv_ms_ztf_time_span_days"] == 365.25
    assert out.loc[0, "ltv_ms_ztf_mag_range"] == 1.0
    assert round(float(out.loc[0, "ltv_ms_ztf_mag_slope_per_year"]), 6) == 1.0
    assert out.loc[0, "ltv_ms_neowise_n_points"] == 2
    assert out.loc[0, "ltv_ms_neowise_w1_range"] == 1.0
    assert round(float(out.loc[0, "ltv_ms_neowise_w1_slope_per_year"]), 6) == 1.0


def test_ltv_multi_survey_columns_persist_in_review_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    ingest_ltv_results(
        db_path,
        pd.DataFrame(
            {
                "asas_sn_id": ["123"],
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
