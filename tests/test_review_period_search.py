from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.review import period_search


def _make_band_df(jd: np.ndarray, true_period: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phase = np.mod((jd - jd.min()) / true_period, 1.0)
    resid = 0.20 * np.exp(-0.5 * ((phase - 0.35) / 0.06) ** 2)
    resid += 0.05 * np.exp(-0.5 * ((phase - 0.62) / 0.03) ** 2)
    resid += rng.normal(0.0, 0.01, size=jd.size)
    return pd.DataFrame({"JD": jd, "resid": resid})


def test_score_period_harmonic_candidate_flags_common_alias_period() -> None:
    jd = np.linspace(0.0, 60.0, 240)
    band_resid = {
        0: (jd, _make_band_df(jd, 1.0, seed=1)["resid"].to_numpy()),
        1: (jd + 0.03, _make_band_df(jd + 0.03, 1.0, seed=2)["resid"].to_numpy()),
    }

    score = period_search._score_period_harmonic_candidate(band_resid, 1.0)

    assert score["alias_flag"] is True
    assert 1.0 in score["alias_matches"]
    assert np.isfinite(float(score["objective"]))


def test_arbitrate_harmonic_period_checks_multiples() -> None:
    jd = np.linspace(0.0, 120.0, 600)
    band_dfs = {
        0: _make_band_df(jd, 2.0, seed=11),
        1: _make_band_df(jd + 0.03, 2.0, seed=12),
    }

    best_period, harmonic_factor, diag = period_search.arbitrate_harmonic_period(
        band_dfs,
        base_period=1.0,
        min_period=0.1,
        max_period=10.0,
    )

    assert abs(best_period - 2.0) < 0.15
    assert harmonic_factor == 2.0
    assert diag["alias_flag"] is False


def test_run_period_search_for_payload_auto_uses_multi_method_search(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lc_path = tmp_path / "candidate.csv"
    lc_path.write_text("JD,Mag\n", encoding="ascii")

    jd = np.linspace(0.0, 120.0, 600)
    band_dfs = {
        0: _make_band_df(jd, 2.0, seed=21),
        1: _make_band_df(jd + 0.03, 2.0, seed=22),
    }

    calls: list[str] = []

    monkeypatch.setattr(period_search, "resolve_lightcurve_path", lambda payload, plot_dir: lc_path)
    monkeypatch.setattr(
        period_search,
        "_load_cleaned_df",
        lambda *args, **kwargs: (pd.DataFrame({"JD": jd, "mag": np.zeros_like(jd)}), None, None),
    )
    monkeypatch.setattr(period_search, "_compute_baseline_bands", lambda *args, **kwargs: band_dfs)

    def _fake_pdm(times, values, min_period, max_period, refine):
        calls.append("pdm")
        return 1.0, np.array([1.0]), np.array([0.1])

    def _fake_ce(times, values, min_period, max_period, refine):
        calls.append("ce")
        return 2.0, np.array([2.0]), np.array([0.1])

    monkeypatch.setattr(period_search, "pdm_find_period", _fake_pdm)
    monkeypatch.setattr(period_search, "ce_find_period", _fake_ce)

    result, label = period_search.run_period_search_for_payload(
        {"candidate_id": "TEST-1"},
        plot_dir=tmp_path,
        min_period=0.1,
        max_period=10.0,
        method="auto",
    )

    assert calls == ["pdm", "ce"]
    assert isinstance(result, dict)
    assert result["auto"] is True
    assert abs(float(result["best_period"]) - 2.0) < 0.15
    assert set(result["searched_methods"]) == {"PDM", "CE"}
    assert label.startswith("Auto CE/PDM: P=")
