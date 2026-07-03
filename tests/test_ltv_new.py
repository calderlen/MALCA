from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.ltv_new.api import fit_ltv_evidence
from malca.ltv_new.likelihood import LightCurveData, gaussian_log_likelihood
from malca.ltv_new.models import evaluate_mean
from malca.ltv_new.priors import build_prior_transform
from malca.ltv_new.samplers import SamplerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_data() -> LightCurveData:
    jd = 2450000.0 + np.linspace(0.0, 100.0, 24)
    mag = 14.0 + 0.2 * ((jd - np.median(jd)) / 365.25)
    err = np.full_like(jd, 0.02)
    band = np.zeros(jd.size, dtype=int)
    return LightCurveData(jd=jd, mag=mag, err=err, band=band, target_id="synthetic")


def test_ltv_new_model_curves_are_deterministic() -> None:
    t = np.array([0.0, 5.0, 10.0])

    assert np.allclose(evaluate_mean("flat", t, {"mu": 14.0}, t_ref=0.0), 14.0)
    assert evaluate_mean("linear", t, {"mu": 14.0, "slope": 0.36525}, t_ref=0.0)[-1] == pytest.approx(14.01)
    assert evaluate_mean("step", t, {"mu": 14.0, "amp": 0.5, "t0": 5.0}, t_ref=0.0).tolist() == [
        14.0,
        14.5,
        14.5,
    ]
    assert evaluate_mean(
        "sinusoid",
        np.array([0.0, 25.0]),
        {"mu": 14.0, "amp": 1.0, "period": 100.0, "phase": 0.0},
        t_ref=0.0,
    )[1] == pytest.approx(15.0)
    fred = evaluate_mean(
        "fred",
        t,
        {"mu": 14.0, "amp": 0.5, "t0": 0.0, "tau_rise": 1.0, "tau_decay": 10.0},
        t_ref=0.0,
    )
    assert np.all(np.isfinite(fred))
    assert fred.max() > 14.0


def test_ltv_new_prior_transform_outputs_finite_physical_values() -> None:
    data = _synthetic_data()
    transform = build_prior_transform("soft_step", data, include_band_offset=True)
    params = transform.transform(np.full(transform.ndim, 0.5))

    assert set(params) == {"mu", "amp", "t0", "tau"}
    assert all(np.isfinite(value) for value in params.values())
    assert data.jd_min <= params["t0"] <= data.jd_max
    assert params["tau"] > 0.0


def test_ltv_new_likelihood_prefers_matching_linear_model() -> None:
    data = _synthetic_data()
    linear = gaussian_log_likelihood(
        data,
        "linear",
        {"mu": 14.0, "slope": 0.2},
    )
    flat = gaussian_log_likelihood(data, "flat", {"mu": 14.0})

    assert linear > flat


def test_ltv_new_band_offset_improves_two_band_likelihood() -> None:
    jd = 2450000.0 + np.linspace(0.0, 50.0, 20)
    mag = np.full(jd.size, 14.0)
    band = np.zeros(jd.size, dtype=int)
    band[10:] = 1
    mag[band == 1] += 0.3
    data = LightCurveData(jd=jd, mag=mag, err=np.full_like(jd, 0.02), band=band)

    without = gaussian_log_likelihood(data, "flat", {"mu": 14.0, "delta_vg": 0.0}, include_band_offset=True)
    with_offset = gaussian_log_likelihood(data, "flat", {"mu": 14.0, "delta_vg": 0.3}, include_band_offset=True)

    assert with_offset > without


def test_ltv_new_fit_evidence_smoke_with_monte_carlo_backend() -> None:
    result = fit_ltv_evidence(
        _synthetic_data(),
        model_names=("flat", "linear"),
        sampler_config=SamplerConfig(backend="monte-carlo", mc_samples=32, seed=7),
    )

    assert {row["model_name"] for row in result.model_rows} == {"flat", "linear"}
    assert result.summary["target_id"] == "synthetic"
    assert result.summary["n_models_ok"] == 2


def test_ltv_new_cli_fit_writes_tables(tmp_path: Path) -> None:
    lc_path = tmp_path / "lc.csv"
    out_dir = tmp_path / "out"
    jd = 2450000.0 + np.linspace(0.0, 30.0, 12)
    pd.DataFrame(
        {
            "JD": jd,
            "mag": 14.0 + 0.01 * np.arange(jd.size),
            "error": np.full(jd.size, 0.03),
            "v_g_band": np.zeros(jd.size, dtype=int),
            "saturated": np.zeros(jd.size, dtype=int),
        }
    ).to_csv(lc_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "malca",
            "ltv-new",
            "fit",
            "--input",
            str(lc_path),
            "--output",
            str(out_dir),
            "--models",
            "flat,linear",
            "--backend",
            "monte-carlo",
            "--mc-samples",
            "16",
            "--seed",
            "3",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    model_table = pd.read_parquet(out_dir / "ltv_new_model_evidence.parquet")
    summary = pd.read_parquet(out_dir / "ltv_new_summary.parquet")
    assert set(model_table["model_name"]) == {"flat", "linear"}
    assert summary.loc[0, "target_id"] == "lc"
