from __future__ import annotations

import subprocess
import sys
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from malca.review.dustycult_display import (
    build_dustycult_corner_figure,
    build_dustycult_fit_figure,
    dustycult_samples_path,
    dustycult_fit_metadata_rows,
    dustycult_geometry_rows,
    dustycult_posterior_rows,
    load_dustycult_samples,
    select_dustycult_display_row,
)
from malca.review.dustycult_visualization import (
    DustOcculterParameters,
    DustStarParameters,
    occulter_absorption_grid,
    occulter_parameters_from_fit,
)
from malca.review.store import db_connect, upsert_candidates_frame


def _insert_candidate(conn, candidate_id: str = "cand-1") -> None:
    upsert_candidates_frame(conn, pd.DataFrame([{"candidate_id": candidate_id}]))


_DUSTYCULT_REQUIRED_SAMPLE_COLUMNS = (
    "t0",
    "v",
    "b",
    "tau0",
    "lambda0",
    "alpha",
    "sigma_y",
    "sigma_x_plus",
    "sigma_x_minus",
)


def _fit_row_with_posterior(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "cand-1",
        "mode": "quick",
        "status": "ok",
        "runtime_sec": 1.25,
        "start_jd": 10.0,
        "end_jd": 12.0,
        "artifact_dir": "/tmp/dustycult",
        "t0_jd": 10.5,
        "posterior_json": json.dumps(
            {
                "t0": {"median": 10.6, "p16": 10.5, "p84": 10.7},
                "v": {"median": 2.0, "p16": 1.8, "p84": 2.2},
                "b": {"median": 0.2, "p16": 0.1, "p84": 0.3},
                "tau0": {"median": 0.4, "p16": 0.3, "p84": 0.5},
                "lambda0": {"median": 510.0, "p16": 500.0, "p84": 520.0},
                "alpha": {"median": 1.2, "p16": 1.0, "p84": 1.4},
                "sigma_y": {"median": 0.3, "p16": 0.2, "p84": 0.4},
                "sigma_x_plus": {"median": 0.4, "p16": 0.3, "p84": 0.5},
                "sigma_x_minus": {"median": 0.5, "p16": 0.4, "p84": 0.6},
            }
        ),
        "stellar_json": json.dumps({"R": 1.3, "u1": 0.2, "u2": 0.1}),
    }
    row.update(overrides)
    return row


def _prepared_quality(frame: pd.DataFrame, window: dict[str, object], *, warnings: list[str] | None = None, errors: list[str] | None = None) -> dict[str, object]:
    warnings = list(warnings or [])
    errors = list(errors or [])
    return {
        "status": "failed" if errors else ("warning" if warnings else "ok"),
        "warnings": warnings,
        "errors": errors,
        "n_points": int(len(frame)),
        "band_counts": {str(k): int(v) for k, v in frame["band"].value_counts().items()},
        "time_span_days": float(frame["time"].max() - frame["time"].min()) if not frame.empty else None,
        "window_span_days": float(window["end_jd"] - window["start_jd"]),
        "finite_error_points": int(np.isfinite(frame["relative_flux_error"]).sum()) if not frame.empty else 0,
        "before_t0_points": int((frame["time"] < window["t0_jd"]).sum()) if not frame.empty else 0,
        "after_t0_points": int((frame["time"] > window["t0_jd"]).sum()) if not frame.empty else 0,
        "baseline": {"name": "median", "warnings": []},
        "window": {
            "start_jd": window["start_jd"],
            "end_jd": window["end_jd"],
            "t0_jd": window["t0_jd"],
            "source": window.get("source", "test"),
        },
    }


def _prepared_input(n_points: int = 30, *, one_band: bool = False, warnings: list[str] | None = None, errors: list[str] | None = None) -> PreparedDustyCultInput:
    times = np.linspace(0.0, 20.0, n_points)
    flux = 1.0 - 0.08 * np.exp(-0.5 * ((times - 10.0) / 1.2) ** 2)
    bands = ["g"] * n_points if one_band else ["g" if i % 2 == 0 else "V" for i in range(n_points)]
    frame = pd.DataFrame(
        {
            "time": times,
            "relative_flux": flux,
            "relative_flux_error": np.full(n_points, 0.02),
            "band": bands,
        }
    )
    window = {"start_jd": 0.0, "end_jd": 20.0, "t0_jd": 10.0, "n_input_points": n_points, "source": "test"}
    return PreparedDustyCultInput(
        frame=frame,
        window=window,
        baseline_name="median",
        baseline_warnings=[],
        quality=_prepared_quality(frame, window, warnings=warnings, errors=errors),
    )


def _sample_artifacts(n_samples: int = 30, *, degenerate: bool = False) -> pd.DataFrame:
    if degenerate:
        values = np.zeros(n_samples)
    else:
        values = np.linspace(-0.05, 0.05, n_samples)
    return pd.DataFrame(
        {
            "t0": 10.0 + values,
            "v": 1.0 + 0.1 * values,
            "b": 0.1 + 0.05 * values,
            "tau0": 0.3 + 0.05 * values,
            "lambda0": 510.0 + values,
            "alpha": 0.2 + 0.1 * values,
            "sigma_y": 0.25 + 0.02 * values,
            "sigma_x_plus": 0.25 + 0.02 * values,
            "sigma_x_minus": 0.25 + 0.02 * values,
        }
    )


def _predictive_artifacts(prepared: PreparedDustyCultInput, *, flat: bool = False) -> pd.DataFrame:
    frame = prepared.frame.reset_index(drop=True)
    median = np.ones(len(frame)) if flat else frame["relative_flux"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "point_id": np.arange(1, len(frame) + 1),
            "time": frame["time"].to_numpy(dtype=float),
            "band": frame["band"].astype(str).to_numpy(),
            "observed": frame["relative_flux"].to_numpy(dtype=float),
            "error": frame["relative_flux_error"].to_numpy(dtype=float),
            "lower95": median - 0.04,
            "lower68": median - 0.02,
            "median": median,
            "upper68": median + 0.02,
            "upper95": median + 0.04,
        }
    )


def _write_mock_fit_artifacts(output_dir: Path, prepared: PreparedDustyCultInput, *, degenerate: bool = False, flat: bool = False) -> None:
    _sample_artifacts(degenerate=degenerate).to_csv(output_dir / "samples.csv", index=False)
    _predictive_artifacts(prepared, flat=flat).to_csv(output_dir / "predictive_intervals.csv", index=False)
    (output_dir / "manifest.json").write_text('{"status":"ok"}\n', encoding="utf-8")


def test_dustycult_display_selects_full_ok_then_quick_ok_then_latest() -> None:
    fits = pd.DataFrame(
        [
            {"mode": "quick", "status": "ok", "updated_at": "2026-01-01T00:00:00Z"},
            {"mode": "full", "status": "failed", "updated_at": "2026-01-02T00:00:00Z"},
            {"mode": "full", "status": "ok", "updated_at": "2026-01-03T00:00:00Z"},
        ]
    )

    assert select_dustycult_display_row(fits).get("mode") == "full"
    assert select_dustycult_display_row(fits, mode="quick").get("mode") == "quick"
    no_ok = fits.assign(status=["failed", "failed", "failed"])
    assert select_dustycult_display_row(no_ok).get("updated_at") == "2026-01-03T00:00:00Z"
    warning_only = fits.assign(status=["failed", "warning", "failed"])
    assert select_dustycult_display_row(warning_only).get("mode") == "full"


def test_dustycult_display_rows_extract_metadata_geometry_and_posterior() -> None:
    row = pd.Series(_fit_row_with_posterior())

    metadata = dict(dustycult_fit_metadata_rows(row))
    geometry = dict(dustycult_geometry_rows(row))
    posterior = dustycult_posterior_rows(row, limit=None)

    assert metadata["mode"] == "quick"
    assert metadata["status"] == "ok"
    assert metadata["artifact"] == "/tmp/dustycult"
    assert geometry["R_star"] == "1.3"
    assert geometry["b / R_star"] != "-"
    assert ("t0", "10.6", "10.5", "10.7") in posterior


def test_dustycult_display_fit_figure_uses_stored_predictive_curves() -> None:
    row = pd.Series(_fit_row_with_posterior())
    curves = pd.DataFrame(
        {
            "time": [10.0, 11.0],
            "band": ["g", "g"],
            "observed": [0.98, 0.9],
            "error": [0.02, 0.02],
            "lower95": [0.8, 0.82],
            "lower68": [0.9, 0.88],
            "median": [0.95, 0.92],
            "upper68": [1.0, 0.98],
            "upper95": [1.05, 1.02],
        }
    )

    fig = build_dustycult_fit_figure(curves, row, theme="white")

    assert "DustyCult Quick Fit" in fig.layout.title.text
    assert "$F/F" in fig.layout.yaxis.title.text
    assert any(trace.name == "g median" for trace in fig.data)


def test_dustycult_corner_figure_loads_samples_from_search_root(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "review" / "dustycult" / "cand-1" / "quick"
    artifact_dir.mkdir(parents=True)
    _sample_artifacts(n_samples=24).to_parquet(artifact_dir / "samples.parquet", index=False)
    row = pd.Series(_fit_row_with_posterior(artifact_dir=str(tmp_path / "stale"), candidate_id="cand-1", mode="quick"))

    assert dustycult_samples_path(row, search_roots=[tmp_path / "review" / "dustycult"]) == artifact_dir / "samples.parquet"
    samples = load_dustycult_samples(row, search_roots=[tmp_path / "review" / "dustycult"])
    fig = build_dustycult_corner_figure(
        row,
        parameters=("t0", "v", "b"),
        search_roots=[tmp_path / "review" / "dustycult"],
        theme="white",
    )

    assert samples.shape == (24, 9)
    assert "cand-1 - DustyCult Quick Posterior Corner" in fig.layout.title.text
    assert sum(trace.type == "histogram" for trace in fig.data) == 3
    assert sum(trace.type == "scatter" for trace in fig.data) == 3


def test_dustycult_corner_figure_handles_missing_samples() -> None:
    row = pd.Series(_fit_row_with_posterior(artifact_dir="/definitely/missing/dustycult"))

    fig = build_dustycult_corner_figure(row, parameters=("t0", "v"), theme="white")

    assert len(fig.data) == 0
    assert "No DustyCult posterior sample artifact" in fig.layout.annotations[0].text


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
    assert np.isclose(defaults["t0_width_days"], 10.0 / 3.0)
    assert defaults["star_R"] == 1.0


def test_control_defaults_broad_stored_event_gets_broad_t0_prior(tmp_path: Path) -> None:
    payload = {
        "dip_best_t0": 2459000.0,
        "dip_best_width_param": 100.0,
        "dip_max_run_duration": 400.0,
    }
    with db_connect(tmp_path / "review.db") as conn:
        defaults = control_defaults_for_candidate(conn, "cand-1", payload)

    assert defaults["start_jd"] == 2458880.0
    assert defaults["end_jd"] == 2459120.0
    assert defaults["half_width_days"] == 120.0
    assert defaults["t0_width_days"] == 30.0


def test_control_defaults_fall_back_when_stored_window_fails_preflight(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"JD": [1.0, 2.0, 3.0], "mag": [12.0, 13.0, 12.1], "error": [0.02, 0.02, 0.02]})

    monkeypatch.setattr(
        "malca.review.dustycult.load_canonical_cleaned_lightcurve",
        lambda *args, **kwargs: (df, tmp_path / "lc.dat2"),
    )
    monkeypatch.setattr(
        "malca.review.dustycult._defaults_viability_quality",
        lambda *_args, **_kwargs: {"status": "failed", "warnings": [], "errors": ["stored window has no points"]},
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
        defaults = control_defaults_for_candidate(conn, "cand-1", {"dip_best_t0": 100.0})

    assert defaults["source"] == "recomputed_dip_run"
    assert defaults["t0_jd"] == 2.0
    assert "stored window has no points" in defaults["message"]


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
        _insert_candidate(conn)
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
    prepared = _prepared_input()

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
        _write_mock_fit_artifacts(out_dir, prepared)
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr("malca.review.dustycult.prepare_dustycult_input", fake_prepare)
    monkeypatch.setattr("malca.review.dustycult.check_dustycult_available", fake_available)
    monkeypatch.setattr("malca.review.dustycult.subprocess.run", fake_run)

    controls = dict(DEFAULT_CONTROLS)
    controls.update({"start_jd": 0.0, "end_jd": 20.0, "t0_jd": 10.0})
    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn)
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
    assert config["prior_kwargs"]["t0_center"] == 10.0
    assert row["n_curve_points"] == len(prepared.frame)
    assert len(curves) == len(prepared.frame)


def test_run_dustycult_fit_preflight_failure_skips_julia(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared_input(
        4,
        errors=["DustyCult needs at least 12 valid g/V points; found 4."],
    )

    def fake_available(*args, **kwargs):
        return DustyCultAvailability(
            True,
            sys.executable,
            tmp_path,
            tmp_path / "scripts" / "fit_lightcurve.jl",
            "available",
        )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("preflight failure should not launch DustyCult")

    monkeypatch.setattr("malca.review.dustycult.prepare_dustycult_input", lambda *args, **kwargs: prepared)
    monkeypatch.setattr("malca.review.dustycult.check_dustycult_available", fake_available)
    monkeypatch.setattr("malca.review.dustycult.subprocess.run", fail_run)

    controls = dict(DEFAULT_CONTROLS)
    controls.update({"start_jd": 0.0, "end_jd": 20.0, "t0_jd": 10.0})
    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn)
        row = run_dustycult_fit(
            conn,
            "cand-1",
            {},
            db_path=tmp_path / "review.db",
            controls=controls,
            mode="quick",
        )

    assert row["status"] == "failed"
    assert row["n_input_points"] == 4
    assert "at least 12" in row["error"]


def test_run_dustycult_fit_one_band_result_is_warning(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared_input(30, one_band=True, warnings=["DustyCult input contains only one ASAS-SN band."])

    def fake_available(*args, **kwargs):
        return DustyCultAvailability(
            True,
            sys.executable,
            tmp_path,
            tmp_path / "scripts" / "fit_lightcurve.jl",
            "available",
        )

    def fake_run(command, check, capture_output, text):
        out_dir = Path(command[command.index("--out") + 1])
        _write_mock_fit_artifacts(out_dir, prepared)
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr("malca.review.dustycult.prepare_dustycult_input", lambda *args, **kwargs: prepared)
    monkeypatch.setattr("malca.review.dustycult.check_dustycult_available", fake_available)
    monkeypatch.setattr("malca.review.dustycult.subprocess.run", fake_run)

    controls = dict(DEFAULT_CONTROLS)
    controls.update({"start_jd": 0.0, "end_jd": 20.0, "t0_jd": 10.0})
    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn)
        row = run_dustycult_fit(
            conn,
            "cand-1",
            {},
            db_path=tmp_path / "review.db",
            controls=controls,
            mode="quick",
        )

    assert row["status"] == "warning"
    assert "one ASAS-SN band" in row["error"]


def test_run_dustycult_fit_all_divergent_degenerate_samples_fail(tmp_path: Path, monkeypatch) -> None:
    prepared = _prepared_input()

    def fake_available(*args, **kwargs):
        return DustyCultAvailability(
            True,
            sys.executable,
            tmp_path,
            tmp_path / "scripts" / "fit_lightcurve.jl",
            "available",
        )

    def fake_run(command, check, capture_output, text):
        out_dir = Path(command[command.index("--out") + 1])
        _write_mock_fit_artifacts(out_dir, prepared, degenerate=True, flat=True)
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="There were 30 divergent transitions after tuning")

    monkeypatch.setattr("malca.review.dustycult.prepare_dustycult_input", lambda *args, **kwargs: prepared)
    monkeypatch.setattr("malca.review.dustycult.check_dustycult_available", fake_available)
    monkeypatch.setattr("malca.review.dustycult.subprocess.run", fake_run)

    controls = dict(DEFAULT_CONTROLS)
    controls.update({"start_jd": 0.0, "end_jd": 20.0, "t0_jd": 10.0})
    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn)
        row = run_dustycult_fit(
            conn,
            "cand-1",
            {},
            db_path=tmp_path / "review.db",
            controls=controls,
            mode="quick",
        )

    assert row["status"] == "failed"
    assert "All DustyCult samples diverged" in row["error"]


def test_occulter_parameters_extract_physical_and_log_posteriors() -> None:
    row = {
        "t0_jd": 10.5,
        "posterior_json": json.dumps(
            {
                "t0": {"median": 10.6},
                "log_v": {"median": np.log(2.0)},
                "b": {"median": 0.2},
                "log_tau0": {"median": np.log(0.4)},
                "log_lambda0": {"median": np.log(510.0)},
                "alpha": {"median": 1.2},
                "log_sigma_y": {"median": np.log(0.3)},
                "sigma_x_plus": {"median": 0.4},
                "log_sigma_x_minus": {"median": np.log(0.5)},
            }
        ),
        "stellar_json": json.dumps({"R": 1.3, "u1": 0.2, "u2": 0.1}),
    }

    params = occulter_parameters_from_fit(row)

    assert np.isclose(params.t0, 10.6)
    assert np.isclose(params.v, 2.0)
    assert np.isclose(params.b, 0.2)
    assert np.isclose(params.tau0, 0.4)
    assert np.isclose(params.lambda0, 510.0)
    assert np.isclose(params.alpha, 1.2)
    assert np.isclose(params.sigma_y, 0.3)
    assert np.isclose(params.sigma_x_plus, 0.4)
    assert np.isclose(params.sigma_x_minus, 0.5)


def test_occulter_parameters_report_missing_required_values() -> None:
    row = {"posterior_json": json.dumps({"t0": {"median": 1.0}})}

    with pytest.raises(ValueError, match="Missing DustyCult posterior parameters"):
        occulter_parameters_from_fit(row)


def test_occulter_absorption_grid_matches_dustycult_formula() -> None:
    dust = DustOcculterParameters(
        t0=0.0,
        v=1.0,
        b=0.0,
        tau0=0.5,
        lambda0=500.0,
        alpha=1.0,
        sigma_y=0.25,
        sigma_x_plus=0.25,
        sigma_x_minus=0.25,
    )
    star = DustStarParameters(R=1.0, I0=1.0, u1=0.2, u2=0.1)

    x, y, absorption, _extent = occulter_absorption_grid(dust, star, 500.0, grid_n=51)

    center = absorption[len(y) // 2, len(x) // 2]
    expected = 1.0 - np.exp(-0.5)
    assert np.isclose(center, expected)

    _x2, _y2, blue_absorption, _extent2 = occulter_absorption_grid(dust, star, 250.0, grid_n=51)
    assert blue_absorption[len(y) // 2, len(x) // 2] > center
