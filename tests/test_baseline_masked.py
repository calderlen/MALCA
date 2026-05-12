from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "celerite2" not in sys.modules:
    fake_celerite2 = types.ModuleType("celerite2")

    class _ImportDummyGP:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

    class _ImportDummyTerm:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def __add__(self, other):
            return self

    fake_celerite2.GaussianProcess = _ImportDummyGP
    fake_celerite2.terms = types.SimpleNamespace(SHOTerm=_ImportDummyTerm, RealTerm=_ImportDummyTerm)
    sys.modules["celerite2"] = fake_celerite2

import malca.baseline as baseline_module


class _DummySHOTerm:
    def __init__(self, *, S0: float, w0: float, Q: float) -> None:
        self.S0 = S0
        self.w0 = w0
        self.Q = Q


class _DummyGP:
    instances: list["_DummyGP"] = []

    def __init__(self, kernel) -> None:
        self.kernel = kernel
        self.compute_t = None
        self.compute_diag = None
        _DummyGP.instances.append(self)

    def compute(self, t, diag) -> None:
        self.compute_t = np.asarray(t, dtype=float)
        self.compute_diag = np.asarray(diag, dtype=float)

    def predict(self, y_centered, t_pred, return_var=True):
        t_pred = np.asarray(t_pred, dtype=float)
        mu = np.zeros_like(t_pred, dtype=float)
        var = np.zeros_like(t_pred, dtype=float)
        if return_var:
            return mu, var
        return mu


def test_gp_masked_iterative_pass_masks_long_dip(monkeypatch) -> None:
    _DummyGP.instances.clear()
    monkeypatch.setattr(
        baseline_module,
        "terms",
        types.SimpleNamespace(SHOTerm=_DummySHOTerm),
    )
    monkeypatch.setattr(baseline_module, "GaussianProcess", _DummyGP)
    monkeypatch.setattr(
        baseline_module,
        "rolling_time_median",
        lambda jd, mag, **kwargs: np.asarray(mag, dtype=float),
    )

    t = np.arange(40, dtype=float)
    y = np.full_like(t, 10.0, dtype=float)
    y[15:25] = 14.0
    yerr = np.full_like(t, 0.05, dtype=float)
    df = pd.DataFrame({"JD": t, "mag": y, "error": yerr, "camera#": "camA"})

    out = baseline_module.per_camera_gp_baseline_masked(
        df,
        pad_days=0.0,
        min_gp_points=5,
        iterative_masking=True,
    )

    assert len(_DummyGP.instances) == 2
    assert len(_DummyGP.instances[0].compute_t) == 40
    assert len(_DummyGP.instances[1].compute_t) == 30
    assert _DummyGP.instances[0].kernel.w0 < _DummyGP.instances[1].kernel.w0
    assert np.allclose(out.loc[:14, "baseline"], 10.0)
    assert np.allclose(out.loc[25:, "baseline"], 10.0)
