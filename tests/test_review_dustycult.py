from __future__ import annotations

import subprocess
import sys
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd

from malca.review.dustycult import (
    DEFAULT_CONTROLS,
    DustyCultAvailability,
    PreparedDustyCultInput,
    build_dustycult_config,
    control_defaults_for_candidate,
    load_dustycult_curve,
    load_dustycult_fits,
    prepare_dustycult_input,
    run_dustycult_fit,
    upsert_dustycult_fit,
)
from malca.review.store import db_connect


def test_control_defaults_use_stored_dip_columns_for_window(tmp_path: Path) -> None:
    payload = {
        "dip_best_t0": 2459000.0,
        "dip_best_width_param": 2.0,
        "dip_max_run_duration": 20.0,
        "dip_best_amp": 0.6,
    }
    with db_connect(tmp_path / "review.db") as conn:
        defaults = control_defaults_for_candidate(conn, "cand-1", payload)

    assert defaults["source"] == "stored_event_columns"
    assert defaults["t0_jd"] == 2459000.0
    assert defaults["start_jd"] == 2458990.0
    assert defaults["end_jd"] == 2459010.0
    assert defaults["star_R"] == 1.0


def test_control_defaults_recompute_when_stored_window_missing(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"JD": [1.0, 2.0, 3.0], "mag": [12.0, 13.0, 12.1], "error": [0.02, 0.02, 0.02]})

    monkeypatch.setattr(
        "malca.review.dustycult.load_canonical_cleaned_lightcurve",
        lambda *args, **kwargs: (df, tmp_path / "lc.dat2"),
    )
    monkeypatch.setattr(
        "malca.review.dustycult.recompute_dip_defaults",
        lambda *_args, **_kwargs: {
            "source": "recomputed_dip_run",
            "start_jd": 1.0,
            "end_jd": 3.0,
            "t0_jd": 2.0,
            "half_width_days": 1.0,
            "message": "recomputed",
        },
    )

    with db_connect(tmp_path / "review.db") as conn:
        defaults = control_defaults_for_candidate(conn, "cand-1", {})

    assert defaults["source"] == "recomputed_dip_run"
    assert defaults["t0_jd"] == 2.0


def test_build_dustycult_config_uses_quick_and_full_sampling() -> None:
    controls = dict(DEFAULT_CONTROLS)
    controls.update({"start_jd": 10.0, "end_jd": 20.0, "t0_jd": 15.0})

    quick = build_dustycult_config("input.csv", controls, "quick")
    full = build_dustycult_config("input.csv", controls, "full")

    assert quick["sampling"]["n_samples"] == 200
    assert quick["sampling"]["n_adapt"] == 200
    assert quick["sampling"]["n_chains"] == 1
    assert quick["grid"]["n"] == 51
    assert quick["posterior_predictive"]["n_draws"] == 100
    assert quick["prior_kwargs"]["t0_center"] == 15.0
    assert full["sampling"]["n_samples"] == 1000
    assert full["sampling"]["n_adapt"] == 1000
    assert full["sampling"]["n_chains"] == 4
    assert full["grid"]["n"] == 101
    assert full["posterior_predictive"]["n_draws"] == 200


def test_prepare_dustycult_input_uses_cleaned_gv_relative_flux(tmp_path: Path, monkeypatch) -> None:
    lc_path = tmp_path / "cand-1.dat2"
    lc_path.write_text(
        "\n".join(
            [
                "2458998.0 14.0 0.02 1 1 0 0 c1/f1",
                "2459000.0 14.0 0.02 1 1 0 0 c1/f1",
                "2459002.0 14.5 0.02 1 1 0 0 c1/f1",
                "2459004.0 14.0 0.02 1 1 0 0 c1/f1",
                "2459000.5 13.5 0.03 1 2 1 0 c2/f1",
                "2459002.5 13.8 0.03 1 2 1 0 c2/f1",
                "2459004.5 13.5 0.03 1 2 1 0 c2/f1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_interactive_plot = types.ModuleType("malca.review.interactive_plot")

    def fake_load_cleaned_df(*args, **kwargs):
        return pd.DataFrame(
            {
                "JD": [2458998.0, 2459000.0, 2459002.0, 2459004.0, 2459000.5, 2459002.5, 2459004.5],
                "mag": [14.0, 14.0, 14.5, 14.0, 13.5, 13.8, 13.5],
                "error": [0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03],
                "v_g_band": [0, 0, 0, 0, 1, 1, 1],
            }
        ), set(), {}

    def fake_compute_baseline_bands(df, *_args, **_kwargs):
        out = {}
        for band in (0, 1):
            part = df[df["v_g_band"] == band].copy()
            part["baseline"] = 14.0 if band == 0 else 13.5
            out[band] = part
        return out

    fake_interactive_plot._load_cleaned_df = fake_load_cleaned_df
    fake_interactive_plot._baseline_config_from_run_params = lambda _params: ("global_median", {}, [])
    fake_interactive_plot._compute_baseline_bands = fake_compute_baseline_bands
    fake_interactive_plot.resolve_lightcurve_path = lambda _payload, _plot_path=None: lc_path
    monkeypatch.setitem(sys.modules, "malca.review.interactive_plot", fake_interactive_plot)

    prepared = prepare_dustycult_input(
        {"candidate_id": "cand-1", "lc_path": str(lc_path)},
        {"start_jd": 2458999.0, "end_jd": 2459005.0, "t0_jd": 2459002.0},
        lc_path=lc_path,
        run_params={"baseline_func": "global_median"},
    )

    assert prepared.window["n_input_points"] == 6
    assert set(prepared.frame["band"]) == {"g", "V"}
    dip_row = prepared.frame.loc[np.abs(prepared.frame["time"] - 2459002.0) < 1e-6].iloc[0]
    assert np.isclose(dip_row["relative_flux"], 10 ** (-0.4 * 0.5))
    assert dip_row["relative_flux_error"] > 0


def test_upsert_dustycult_fit_replaces_mode_and_curve_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    curves = pd.DataFrame(
        {
            "candidate_id": ["cand-1", "cand-1"],
            "mode": ["quick", "quick"],
            "point_id": [1, 2],
            "time": [1.0, 2.0],
            "band": ["g", "g"],
            "observed": [0.9, 1.0],
            "error": [0.02, 0.02],
            "lower95": [0.8, 0.9],
            "lower68": [0.85, 0.95],
            "median": [0.9, 1.0],
            "upper68": [0.95, 1.05],
            "upper95": [1.0, 1.1],
        }
    )
    with db_connect(db_path) as conn:
        upsert_dustycult_fit(
            conn,
            {
                "candidate_id": "cand-1",
                "mode": "quick",
                "status": "failed",
                "error": "first",
                "n_curve_points": 2,
            },
            curves,
        )
        upsert_dustycult_fit(
            conn,
            {
                "candidate_id": "cand-1",
                "mode": "quick",
                "status": "ok",
                "error": "",
                "n_curve_points": 0,
            },
            pd.DataFrame(),
        )
        fits = load_dustycult_fits(conn, "cand-1")
        stored_curves = load_dustycult_curve(conn, "cand-1", "quick")

    assert len(fits) == 1
    assert fits.iloc[0]["status"] == "ok"
    assert stored_curves.empty


def test_run_dustycult_fit_imports_mocked_artifacts(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [10.0, 11.0],
            "relative_flux": [0.9, 1.0],
            "relative_flux_error": [0.02, 0.02],
            "band": ["g", "V"],
        }
    )
    prepared = PreparedDustyCultInput(
        frame=frame,
        window={"start_jd": 9.0, "end_jd": 12.0, "t0_jd": 10.5, "n_input_points": 2},
        baseline_name="median",
        baseline_warnings=[],
    )

    def fake_prepare(*args, **kwargs):
        return prepared

    def fake_available(*args, **kwargs):
        return DustyCultAvailability(
            True,
            sys.executable,
            tmp_path,
            tmp_path / "scripts" / "fit_lightcurve.jl",
            "available",
        )

    seen = {}

    def fake_run(command, check, capture_output, text):
        seen["command"] = command
        out_dir = Path(command[command.index("--out") + 1])
        pd.DataFrame({"t0": [10.4, 10.6], "v": [1.0, 1.2]}).to_csv(out_dir / "samples.csv", index=False)
        pd.DataFrame(
            {
                "point_id": [1, 2],
                "time": [10.0, 11.0],
                "band": ["g", "V"],
                "observed": [0.9, 1.0],
                "error": [0.02, 0.02],
                "lower95": [0.8, 0.9],
                "lower68": [0.85, 0.95],
                "median": [0.9, 1.0],
                "upper68": [0.95, 1.05],
                "upper95": [1.0, 1.1],
            }
        ).to_csv(out_dir / "predictive_intervals.csv", index=False)
        (out_dir / "manifest.json").write_text('{"status":"ok"}\n')
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr("malca.review.dustycult.prepare_dustycult_input", fake_prepare)
    monkeypatch.setattr("malca.review.dustycult.check_dustycult_available", fake_available)
    monkeypatch.setattr("malca.review.dustycult.subprocess.run", fake_run)

    controls = dict(DEFAULT_CONTROLS)
    controls.update({"start_jd": 9.0, "end_jd": 12.0, "t0_jd": 10.5})
    with db_connect(tmp_path / "review.db") as conn:
        row = run_dustycult_fit(
            conn,
            "cand-1",
            {},
            db_path=tmp_path / "review.db",
            controls=controls,
            mode="quick",
        )
        curves = load_dustycult_curve(conn, "cand-1", "quick")

    assert row["status"] == "ok"
    assert "--config" in seen["command"]
    assert "--out" in seen["command"]
    assert "--input" not in seen["command"]
    config_path = Path(seen["command"][seen["command"].index("--config") + 1])
    config = json.loads(config_path.read_text())
    assert config["prior_kwargs"]["t0_center"] == 10.5
    assert row["n_curve_points"] == 2
    assert len(curves) == 2
