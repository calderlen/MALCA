from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from malca.config import GAIA_TCB_EPOCH_JD, MJD_TO_JD, TESS_BTJD_OFFSET
from malca.multi_survey_features import (
    _derive_event_window,
    compute_multi_survey_features,
    run as run_multi_survey_features,
)
from malca.review.pipeline import (
    _run_multi_survey_features_stage,
    detect_pipeline_status,
)
from malca.review.store import db_connect, get_candidate_payload, upsert_candidates_frame
from malca.table_io import read_parquet_table, write_parquet_table


def _base_event_row(candidate_id: str = "C1") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "asas_sn_id": candidate_id,
        "dip_significant": True,
        "dip_bayes_factor": 50.0,
        "dip_best_t0": 8500.0,
        "dip_best_width_param": 2.0,
        "dip_max_run_duration": 12.0,
        "jump_significant": False,
        "jump_bayes_factor": 1.0,
        "jump_best_t0": 8510.0,
        "jump_best_width_param": 3.0,
        "jump_max_run_duration": 10.0,
        "gaia_var_flag": True,
        "gaia_var_class": "ECL",
        "gaia_var_score": 0.91,
        "gaia_eb_period": 5.0,
        "gaia_eb_morph": "detached",
        "gaia_eb_global_ranking": 0.8,
        "radial_velocity": 12.3,
        "rv_amplitude_robust": 4.5,
        "period_consensus_days": 10.0,
    }


def _write_asassn_dat2(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "8490 14.0 0.02 0 1 0 0 cam/field",
                "8500 15.0 0.02 0 1 0 0 cam/field",
                "8512 14.2 0.02 0 1 0 0 cam/field",
                "8490 13.5 0.02 0 1 1 0 cam/field",
                "8500 14.4 0.02 0 1 1 0 cam/field",
                "8512 13.7 0.02 0 1 1 0 cam/field",
            ]
        )
        + "\n"
    )


def _write_external_lcs(root: Path, candidate_id: str) -> Path:
    t0_jd = 2458500.0
    asassn_path = root / f"{candidate_id}.dat2"
    _write_asassn_dat2(asassn_path)

    pd.DataFrame(
        [
            {"mjd": t0_jd - 10.0 - MJD_TO_JD, "mag": 14.5, "mag_err": 0.03, "band": "zg"},
            {"mjd": t0_jd - 9.8 - MJD_TO_JD, "mag": 14.2, "mag_err": 0.03, "band": "zr"},
            {"mjd": t0_jd - MJD_TO_JD, "mag": 15.2, "mag_err": 0.03, "band": "zg"},
            {"mjd": t0_jd + 0.2 - MJD_TO_JD, "mag": 14.6, "mag_err": 0.03, "band": "zr"},
        ]
    ).to_parquet(root / f"ztf_lc_{candidate_id}.parquet", index=False)

    pd.DataFrame(
        [
            {"mjd": t0_jd - 400.0 - MJD_TO_JD, "w1mpro": 12.0, "w2mpro": 11.7},
            {"mjd": t0_jd - MJD_TO_JD, "w1mpro": 13.0, "w2mpro": 12.5},
        ]
    ).to_parquet(root / f"neowise_lc_{candidate_id}.parquet", index=False)

    pd.DataFrame(
        {
            "time": np.array([t0_jd - 10, t0_jd - 2, t0_jd - 1, t0_jd, t0_jd + 1, t0_jd + 2, t0_jd + 10])
            - TESS_BTJD_OFFSET,
            "flux": [1.0, 0.98, 0.9, 0.8, 0.9, 0.98, 1.0],
            "quality": [0, 0, 0, 0, 0, 0, 0],
        }
    ).to_parquet(root / f"tess_lc_{candidate_id}.parquet", index=False)

    pd.DataFrame(
        {
            "time": np.array([t0_jd - 20, t0_jd, t0_jd + 20]) - GAIA_TCB_EPOCH_JD,
            "mag": [14.0, 14.8, 14.2],
        }
    ).to_parquet(root / f"gaia_epoch_lc_{candidate_id}.parquet", index=False)

    return asassn_path


def test_dominant_event_selection_and_window() -> None:
    row = {
        "dip_significant": False,
        "dip_bayes_factor": 100.0,
        "dip_best_t0": 8400.0,
        "dip_best_width_param": 2.0,
        "dip_max_run_duration": 10.0,
        "jump_significant": True,
        "jump_bayes_factor": 2.0,
        "jump_best_t0": 8500.0,
        "jump_best_width_param": 4.0,
        "jump_max_run_duration": 40.0,
    }

    event = _derive_event_window(row)

    assert event is not None
    assert event["event_type"] == "jump"
    assert event["t0_jd"] == 2458500.0
    assert event["half_width_days"] == 20.0
    assert event["start_jd"] == 2458480.0
    assert event["end_jd"] == 2458520.0


def test_multisurvey_features_from_cached_lcs(tmp_path: Path) -> None:
    row = _base_event_row()
    row["lc_path"] = str(_write_external_lcs(tmp_path, "C1"))

    out = compute_multi_survey_features(pd.DataFrame([row]), external_lc_dir=tmp_path).iloc[0]

    assert out["ms_feature_status"] == "ok"
    assert out["ms_event_type"] == "dip"
    assert out["ms_event_t0_jd"] == 2458500.0
    assert out["ms_asassn_n_event_g"] == 1
    assert out["ms_asassn_n_baseline_g"] == 2
    assert np.isclose(out["ms_asassn_delta_g"], 0.9)
    assert np.isclose(out["ms_asassn_g_minus_v_delta"], 0.1)
    assert out["ms_ztf_gr_event_pairs"] == 1
    assert out["ms_ztf_gr_baseline_pairs"] == 1
    assert np.isclose(out["ms_ztf_gr_delta"], 0.3)
    assert out["ms_neowise_n_near"] == 1
    assert out["ms_neowise_n_baseline"] == 1
    assert np.isclose(out["ms_neowise_w1_delta"], 1.0)
    assert bool(out["ms_tess_event_overlap"]) is True
    assert np.isclose(out["ms_tess_flux_frac_delta"], -0.1)
    assert out["ms_tess_half_depth_duration_days"] == 2.0
    assert np.isfinite(out["ms_tess_ingress_slope_per_day"])
    assert np.isfinite(out["ms_tess_egress_slope_per_day"])
    assert np.isclose(out["ms_gaia_epoch_g_delta"], 0.7)
    assert np.isclose(out["ms_gaia_eb_period_ratio"], 2.0)
    assert out["ms_gaia_var_flag"] is True or out["ms_gaia_var_flag"] == 1


def test_missing_files_are_graceful(tmp_path: Path) -> None:
    row = _base_event_row()

    out = compute_multi_survey_features(pd.DataFrame([row]), external_lc_dir=tmp_path).iloc[0]

    assert out["ms_feature_status"] == "ok"
    assert out["ms_ztf_gr_event_pairs"] == 0
    assert out["ms_neowise_n_near"] == 0
    assert out["ms_tess_n_event"] == 0
    assert pd.isna(out["ms_ztf_gr_delta"])
    assert pd.isna(out["ms_neowise_w1_delta"])
    assert pd.isna(out["ms_tess_flux_frac_delta"])


def test_no_event_stage_status_is_complete_after_check() -> None:
    payload = {"candidate_id": "C1", "ms_feature_status": "no_event", "ms_event_type": "none"}

    assert detect_pipeline_status(payload)["multi_survey_features"] == "complete"


def test_cli_writes_enriched_parquet_and_merges_review_db(tmp_path: Path) -> None:
    row = _base_event_row()
    input_path = tmp_path / "candidates.parquet"
    output_path = tmp_path / "features.parquet"
    db_path = tmp_path / "review.db"
    write_parquet_table(pd.DataFrame([row]), input_path)

    with closing(db_connect(db_path)) as conn:
        upsert_candidates_frame(conn, pd.DataFrame([row]))

    args = argparse.Namespace(
        input=input_path,
        output=output_path,
        external_lc_dir=tmp_path,
        review_db=db_path,
    )
    result_path = run_multi_survey_features(args)

    out = read_parquet_table(result_path)
    assert result_path == output_path
    assert out.loc[0, "ms_feature_status"] == "ok"
    with closing(db_connect(db_path)) as conn:
        payload = get_candidate_payload(conn, "C1")
    assert payload["ms_feature_status"] == "ok"
    assert payload["ms_event_type"] == "dip"


def test_review_stage_status_and_runner(tmp_path: Path) -> None:
    payload = _base_event_row()

    assert detect_pipeline_status(payload)["multi_survey_features"] == "missing"
    _run_multi_survey_features_stage(payload, tmp_path)

    assert payload["ms_feature_status"] == "ok"
    assert detect_pipeline_status(payload)["multi_survey_features"] == "complete"
