from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("astroquery")
pytest.importorskip("celerite2")

from malca.evaluation import reproduce


def test_main_impl_passes_trigger_mode_and_significance(monkeypatch, tmp_path: Path) -> None:
    parser = reproduce.build_parser()
    args = parser.parse_args([])
    args.trigger_mode = "posterior_prob"
    args.significance_threshold = 97.5

    captured: dict[str, object] = {}

    def fake_resolve_candidates(_spec):
        return []

    def fake_build_reproduction_report(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(reproduce, "resolve_candidates", fake_resolve_candidates)
    monkeypatch.setattr(reproduce, "build_reproduction_report", fake_build_reproduction_report)

    reproduce._main_impl(args, plot_out_dir=tmp_path)

    assert captured["trigger_mode"] == "posterior_prob"
    assert captured["significance_threshold"] == 97.5


def test_records_from_skypatrol_dir_accepts_light_curve_suffix(tmp_path: Path) -> None:
    csv_path = tmp_path / "120259184943-light-curves.csv"
    csv_path.write_text("jd,mag\n")

    targets = pd.DataFrame(
        [{"source_id": "120259184943", "mag_bin": "13_13.5"}]
    )

    records = reproduce.records_from_skypatrol_dir(targets, tmp_path)

    assert "13_13.5" in records
    assert records["13_13.5"][0]["dat_path"] == str(csv_path)
    assert records["13_13.5"][0]["found"] is True


def test_ordered_reproduction_columns_tolerates_missing_optional_fields() -> None:
    frame = pd.DataFrame(
        {
            "source": ["demo"],
            "source_id": ["120259184943"],
            "mag_bin": ["13_13.5"],
            "detected": [True],
            "detection_details": ["g_bayes_dip"],
            "g_n_points": [10],
        }
    )

    cols = reproduce._ordered_reproduction_columns(frame)

    assert cols[:5] == [
        "source",
        "source_id",
        "mag_bin",
        "detected",
        "detection_details",
    ]
    assert "category" not in cols
    assert "rejection_reason" not in cols


def test_build_reproduction_report_preserves_scored_rows(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "335007754417-light-curves.csv"
    csv_path.write_text("jd,mag\n")

    lc_df = pd.DataFrame(
        {
            "JD": [1.0, 2.0, 3.0],
            "mag": [14.5, 15.0, 14.4],
            "error": [0.05, 0.05, 0.05],
            "v_g_band": [0, 0, 0],
            "camera#": ["ba", "ba", "ba"],
            "saturated": [0, 0, 0],
        }
    )

    run_summary = {
        "morphology": "paczynski",
        "params": {"t0": 2.0, "amplitude": 0.5, "tE": 10.0},
        "n_points": 3,
        "start_jd": 1.0,
        "end_jd": 3.0,
        "delta_bic_null": 9.0,
        "symmetry_score": 1.0,
    }

    def fake_score_lightcurve(*_args, **_kwargs):
        return {
            "dip": {
                "significant": True,
                "event_indices": np.array([1], dtype=int),
                "bayes_factor": 12.0,
                "max_event_prob": 1.0,
                "max_log_bf_local": 5.0,
                "run_summaries": [run_summary],
                "n_runs": 1,
                "max_run_points": 3,
                "max_run_duration": 2.0,
                "max_run_sum": 1.5,
                "max_run_max": 0.5,
                "max_run_cameras": 1,
                "baseline_mag": 14.5,
                "best_p": 0.1,
                "best_mag_event": 15.0,
                "trigger_max": 1.0,
                "used_sigma_eff": True,
                "baseline_source": "global_median",
                "trigger_mode": "posterior_prob",
                "trigger_threshold": 0.999,
            },
            "jump": {
                "significant": False,
                "event_indices": np.array([], dtype=int),
                "bayes_factor": 0.0,
                "max_event_prob": 0.0,
                "max_log_bf_local": np.nan,
                "run_summaries": [],
                "n_runs": 0,
                "max_run_points": 0,
                "max_run_duration": np.nan,
                "max_run_sum": np.nan,
                "max_run_max": np.nan,
                "max_run_cameras": 0,
                "baseline_mag": 14.5,
                "best_p": np.nan,
                "best_mag_event": np.nan,
                "trigger_max": 0.0,
                "used_sigma_eff": True,
                "baseline_source": "global_median",
                "trigger_mode": "posterior_prob",
                "trigger_threshold": 0.999,
            },
        }

    monkeypatch.setattr(
        reproduce,
        "records_from_skypatrol_dir",
        lambda *_args, **_kwargs: {
            "14.5_15": [
                {
                    "mag_bin": "14.5_15",
                    "index_num": None,
                    "index_csv": None,
                    "lc_dir": str(tmp_path),
                    "asas_sn_id": "335007754417",
                    "dat_path": str(csv_path),
                    "found": True,
                }
            ]
        },
    )
    monkeypatch.setattr(reproduce, "read_skypatrol_csv", lambda _path: lc_df.copy())
    monkeypatch.setattr(reproduce, "score_lightcurve", fake_score_lightcurve)
    monkeypatch.setattr(reproduce, "compute_event_score", lambda *_args, **_kwargs: (0.0, []))

    report = reproduce.build_reproduction_report(
        candidates=[
            {
                "source": "J073234-200049",
                "source_id": "335007754417",
                "category": "Single Eclipse Binaries",
                "mag_bin": "14.5_15",
            }
        ],
        skypatrol_dir=tmp_path,
        n_workers=1,
        baseline_func="global_median",
        skip_tags=True,
        skip_vsx=True,
        min_mag_offset=0.0,
        verbose=False,
    )

    assert "g_bayes_dip_significant" in report.columns
    assert bool(report.loc[0, "g_bayes_dip_significant"]) is True
    assert bool(report.loc[0, "detected"]) is True
    assert "g_bayes_dip" in report.loc[0, "detection_details"]
