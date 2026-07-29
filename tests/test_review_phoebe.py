from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from malca.review.phoebe_fit import (
    PHOEBE_DETACHED_FIT_PARAMETERS,
    infer_period_days,
    load_phoebe_fits,
    parse_phoebe_json,
    run_phoebe_fit,
)
from malca.review.store import db_connect, upsert_candidates_frame


def _insert_candidate(conn, candidate_id: str) -> None:
    upsert_candidates_frame(conn, pd.DataFrame([{"candidate_id": candidate_id}]))


class _FakeChecks:
    passed = True


class _FakeBundle:
    def __init__(self, *, solver_error: Exception | None = None, model_multiplier: float = 1.0) -> None:
        self.fluxes: np.ndarray | None = None
        self.solver_error = solver_error
        self.model_multiplier = model_multiplier
        self.set_values: dict[str, object] = {}
        self.solver_added = False
        self.solver_ran = False

    def set_value(self, qualifier, value, *_args, **_kwargs) -> None:
        self.set_values[str(qualifier)] = value
        return None

    def add_dataset(self, _kind: str, **kwargs) -> None:
        self.fluxes = np.asarray(kwargs["fluxes"], dtype=float)

    def add_solver(self, *_args, **_kwargs) -> None:
        self.solver_added = True
        return None

    def run_checks(self, *_args, **_kwargs) -> _FakeChecks:
        return _FakeChecks()

    def run_solver(self, *_args, **_kwargs) -> None:
        if self.solver_error is not None:
            raise self.solver_error
        self.solver_ran = True
        return None

    def adopt_solution(self, *_args, **_kwargs) -> None:
        return None

    def run_compute(self, *_args, **_kwargs) -> None:
        return None

    def get_value(self, _key: str):
        if self.fluxes is None:
            return []
        return np.full_like(self.fluxes, np.nanmedian(self.fluxes) * self.model_multiplier)


def test_infer_period_prefers_manual_then_catalog() -> None:
    payload = {
        "gaia_eb_period": 3.0,
        "stats_variability_lomb_scargle_best_period_days": 5.0,
    }

    assert infer_period_days(payload, 2.5) == (2.5, "manual")
    assert infer_period_days(payload) == (3.0, "gaia_eb")


def test_run_phoebe_fit_persists_success(tmp_path: Path, monkeypatch) -> None:
    lc_path = tmp_path / "cand.csv"
    lc_path.write_text("unused\n", encoding="ascii")
    monkeypatch.setattr(
        "malca.review.phoebe_fit.load_lightcurve_df",
        lambda _path: pd.DataFrame(
            {
                "JD": [2459000.0, 2459001.0, 2459002.0, 2459003.0],
                "mag": [14.0, 14.2, 14.0, 14.2],
                "error": [0.02, 0.02, 0.02, 0.02],
            }
        ),
    )
    bundle = _FakeBundle(model_multiplier=2.0)
    fake_phoebe = types.SimpleNamespace(__version__="test", default_binary=lambda: bundle)
    monkeypatch.setattr("malca.review.phoebe_fit._import_phoebe", lambda: fake_phoebe)

    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn, "cand-1")
        row = run_phoebe_fit(
            conn,
            "cand-1",
            {"stats_variability_lomb_scargle_best_period_days": 2.0},
            lc_path=lc_path,
        )
        fits = load_phoebe_fits(conn, "cand-1")

    assert row["status"] == "ok"
    assert len(fits) == 1
    assert fits.loc[0, "status"] == "ok"
    assert fits.loc[0, "period_days"] == 2.0
    assert bundle.solver_added
    assert bundle.solver_ran
    assert bundle.set_values["fit_parameters@nm_fit"] == list(PHOEBE_DETACHED_FIT_PARAMETERS)

    params = parse_phoebe_json(row["params_json"])
    metrics = parse_phoebe_json(row["metrics_json"])
    plot = parse_phoebe_json(row["plot_json"])
    assert params["fit_parameters"] == list(PHOEBE_DETACHED_FIT_PARAMETERS)
    assert metrics["model_flux_source"] == "phoebe"
    assert metrics["solver_status"] == "ok"
    assert metrics["model_flux_scale"] == pytest.approx(0.5)
    assert np.nanmedian(plot["model_flux"]) == pytest.approx(np.nanmedian(plot["flux"]))


def test_run_phoebe_fit_warns_when_solver_skips_but_compute_succeeds(tmp_path: Path, monkeypatch) -> None:
    lc_path = tmp_path / "cand.csv"
    lc_path.write_text("unused\n", encoding="ascii")
    monkeypatch.setattr(
        "malca.review.phoebe_fit.load_lightcurve_df",
        lambda _path: pd.DataFrame(
            {
                "JD": [2459000.0, 2459001.0, 2459002.0, 2459003.0],
                "mag": [14.0, 14.2, 14.0, 14.2],
                "error": [0.02, 0.02, 0.02, 0.02],
            }
        ),
    )
    bundle = _FakeBundle(solver_error=RuntimeError("no valid parameters"), model_multiplier=2.0)
    fake_phoebe = types.SimpleNamespace(__version__="test", default_binary=lambda: bundle)
    monkeypatch.setattr("malca.review.phoebe_fit._import_phoebe", lambda: fake_phoebe)

    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn, "cand-warning")
        row = run_phoebe_fit(
            conn,
            "cand-warning",
            {"stats_variability_lomb_scargle_best_period_days": 2.0},
            lc_path=lc_path,
        )
        fits = load_phoebe_fits(conn, "cand-warning")

    assert row["status"] == "warning"
    assert fits.loc[0, "status"] == "warning"
    assert "diagnostic model only" in row["error"]
    metrics = parse_phoebe_json(row["metrics_json"])
    plot = parse_phoebe_json(row["plot_json"])
    assert metrics["solver_status"].startswith("skipped:")
    assert metrics["model_flux_scale"] == pytest.approx(0.5)
    assert np.nanmedian(plot["model_flux"]) == pytest.approx(np.nanmedian(plot["flux"]))


def test_run_phoebe_fit_persists_failure_when_import_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "malca.review.phoebe_fit._import_phoebe",
        lambda: (_ for _ in ()).throw(ImportError("missing phoebe")),
    )

    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn, "cand-2")
        row = run_phoebe_fit(
            conn,
            "cand-2",
            {"stats_variability_lomb_scargle_best_period_days": 2.0},
            lc_path=tmp_path / "missing.dat2",
        )
        fits = load_phoebe_fits(conn, "cand-2")

    assert row["status"] == "failed"
    assert "PHOEBE import failed" in str(row["error"])
    assert fits.loc[0, "status"] == "failed"


def test_run_phoebe_fit_rejects_unsupported_model_kind(tmp_path: Path, monkeypatch) -> None:
    lc_path = tmp_path / "cand.csv"
    lc_path.write_text("unused\n", encoding="ascii")

    with db_connect(tmp_path / "review.db") as conn:
        _insert_candidate(conn, "cand-contact")
        row = run_phoebe_fit(
            conn,
            "cand-contact",
            {"stats_variability_lomb_scargle_best_period_days": 2.0},
            lc_path=lc_path,
            model_kind="contact",
        )
        fits = load_phoebe_fits(conn, "cand-contact")

    assert row["status"] == "failed"
    assert "supports only detached" in str(row["error"])
    assert fits.loc[0, "model_kind"] == "contact"
