from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

if "celerite2" not in sys.modules:
    fake_celerite2 = types.ModuleType("celerite2")

    class _DummyGP:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

    class _DummyTerm:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def __add__(self, other):
            return self

    fake_celerite2.GaussianProcess = _DummyGP
    fake_celerite2.terms = types.SimpleNamespace(SHOTerm=_DummyTerm, RealTerm=_DummyTerm)
    sys.modules["celerite2"] = fake_celerite2

if "iar.IARModel" not in sys.modules:
    fake_iar_pkg = types.ModuleType("iar")
    fake_iar_model = types.ModuleType("iar.IARModel")

    def _dummy_iar(*args, **kwargs):
        _ = args, kwargs
        return float("nan")

    fake_iar_model.IARphikalman = _dummy_iar
    fake_iar_pkg.IARModel = fake_iar_model
    sys.modules["iar"] = fake_iar_pkg
    sys.modules["iar.IARModel"] = fake_iar_model

import malca.stats as stats


def _identity_baseline(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    _ = kwargs
    out = df.copy()
    out["resid"] = pd.to_numeric(out["mag"], errors="coerce")
    return out


def test_string_length_ignores_time_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "per_camera_gp_baseline", _identity_baseline)

    df = pd.DataFrame(
        {
            "JD": [2450000.0, 2450001.0, 2451000.0],
            "mag": [0.0, 1.0, 2.0],
            "error": [0.1, 0.1, 0.1],
            "camera_name": ["camA", "camA", "camA"],
        }
    )

    out = stats.baseline_subtracted_string_length(df)
    assert out["string_length_total"] == pytest.approx(2.0)
    assert out["string_length_mean_step"] == pytest.approx(1.0)
    assert out["string_length_n_steps"] == pytest.approx(2.0)


def test_string_length_higher_for_jittery_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "per_camera_gp_baseline", _identity_baseline)

    base = {
        "JD": [1.0, 2.0, 3.0, 4.0, 5.0],
        "error": [0.1, 0.1, 0.1, 0.1, 0.1],
        "camera_name": ["camA", "camA", "camA", "camA", "camA"],
    }
    smooth = pd.DataFrame({**base, "mag": [0.0, 0.1, 0.2, 0.3, 0.4]})
    jitter = pd.DataFrame({**base, "mag": [0.0, 1.0, -1.0, 1.0, -1.0]})

    smooth_out = stats.baseline_subtracted_string_length(smooth)
    jitter_out = stats.baseline_subtracted_string_length(jitter)

    assert jitter_out["string_length_total"] > smooth_out["string_length_total"]
    assert jitter_out["string_length_mean_step"] > smooth_out["string_length_mean_step"]


def test_string_length_handles_missing_camera_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats, "per_camera_gp_baseline", _identity_baseline)

    df = pd.DataFrame(
        {
            "JD": [10.0, 11.0, 12.0],
            "mag": [1.0, 1.5, 1.0],
            "error": [0.1, 0.1, 0.1],
        }
    )

    out = stats.baseline_subtracted_string_length(df)
    assert out["string_length_total"] == pytest.approx(1.0)
    assert out["string_length_n_steps"] == pytest.approx(2.0)
