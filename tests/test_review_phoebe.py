from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pandas as pd

from malca.review.phoebe_fit import (
    infer_period_days,
    load_phoebe_fits,
    run_phoebe_fit,
)
from malca.review.store import db_connect


class _FakeBundle:
    def __init__(self) -> None:
        self.fluxes: np.ndarray | None = None

    def set_value(self, *_args, **_kwargs) -> None:
        return None

    def add_dataset(self, _kind: str, **kwargs) -> None:
        self.fluxes = np.asarray(kwargs["fluxes"], dtype=float)

    def add_solver(self, *_args, **_kwargs) -> None:
        return None

    def run_solver(self, *_args, **_kwargs) -> None:
        return None

    def adopt_solution(self, *_args, **_kwargs) -> None:
        return None

    def run_compute(self, *_args, **_kwargs) -> None:
        return None

    def get_value(self, _key: str):
        if self.fluxes is None:
            return []
        return np.full_like(self.fluxes, np.nanmedian(self.fluxes))


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
    fake_phoebe = types.SimpleNamespace(__version__="test", default_binary=lambda: _FakeBundle())
    monkeypatch.setattr("malca.review.phoebe_fit._import_phoebe", lambda: fake_phoebe)

    with db_connect(tmp_path / "review.db") as conn:
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


def test_run_phoebe_fit_persists_failure_when_import_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "malca.review.phoebe_fit._import_phoebe",
        lambda: (_ for _ in ()).throw(ImportError("missing phoebe")),
    )

    with db_connect(tmp_path / "review.db") as conn:
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
