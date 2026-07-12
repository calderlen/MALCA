from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

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

import malca.stv.periodicity_gate as periodicity_gate
from malca.stv.periodicity_gate import apply_pre_periodicity_gate


def _write_dat3(path: Path, times: np.ndarray, mags: np.ndarray) -> None:
    lines: list[str] = []
    for idx, (time_value, mag_value) in enumerate(zip(times, mags, strict=True)):
        camera = 1 + (idx % 2)
        lines.append(
            f"{float(time_value):.6f} {float(mag_value):.6f} 0.030000 1 {camera:d} 0 0 cam{camera}/field1"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _periodic_eclipse_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 120.0, 240)
    phase = np.mod(times, 4.0) / 4.0
    eclipse = ((phase < 0.08) | (phase > 0.92)).astype(float)
    mags = 14.0 + 1.5 * eclipse + 0.05 * np.sin(2.0 * np.pi * times / 4.0)
    return times, mags


def _single_dip_lightcurve() -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 120.0, 240)
    mags = np.full_like(times, 14.0)
    dip_mask = np.abs(times - 60.0) <= 1.0
    mags[dip_mask] += 0.9
    return times, mags


def _dummy_band_residuals() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {0: (np.array([0.0, 1.0], dtype=float), np.array([0.0, 0.0], dtype=float))}


def _two_band_offset_frame() -> pd.DataFrame:
    jd_g = np.linspace(0.0, 120.0, 80)
    jd_v = jd_g + 0.15
    signal_g = 14.0 + 0.10 * np.sin(2.0 * np.pi * jd_g / 4.0)
    signal_v = 14.0 + 0.10 * np.sin(2.0 * np.pi * jd_v / 4.0) + 0.8
    return pd.DataFrame(
        {
            "JD": np.concatenate([jd_g, jd_v]),
            "mag": np.concatenate([signal_g, signal_v]),
            "error": np.full(jd_g.size + jd_v.size, 0.03, dtype=float),
            "camera#": np.concatenate([np.ones(jd_g.size, dtype=int), np.full(jd_v.size, 2, dtype=int)]),
            "v_g_band": np.concatenate([np.zeros(jd_g.size, dtype=int), np.ones(jd_v.size, dtype=int)]),
        }
    )


def test_apply_pre_periodicity_gate_flags_strong_periodic_lightcurve(tmp_path: Path) -> None:
    periodic_path = tmp_path / "periodic.dat3"
    times, mags = _periodic_eclipse_lightcurve()
    _write_dat3(periodic_path, times, mags)

    df = pd.DataFrame({"source_id": ["periodic"], "dat_path": [str(periodic_path)]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        ce_snr_threshold=5.0,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert float(out.loc[0, "pre_periodicity_selected_period"]) > 0.0


def test_apply_pre_periodicity_gate_keeps_single_dip_in_stochastic_branch(tmp_path: Path) -> None:
    dip_path = tmp_path / "single_dip.dat3"
    times, mags = _single_dip_lightcurve()
    _write_dat3(dip_path, times, mags)

    df = pd.DataFrame({"source_id": ["single_dip"], "dat_path": [str(dip_path)]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is False
    assert out.loc[0, "pre_periodicity_label"] == "non_periodic"


def test_apply_pre_periodicity_gate_aligns_simple_v_minus_g_median_offset(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()
    captured: dict[str, np.ndarray] = {}

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        assert _kwargs.get("refine") is True
        captured["ce_mag"] = np.asarray(mag, dtype=float).copy()
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.7,
            "ce_snr": 10.0,
            "ce_bootstrap_sig": 0.5,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        _base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        return 4.0, 1.0, {"scatter_ratio": 1.0, "lag_phase": np.nan, "alias_flag": False}

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["two_band"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    g_count = int((df_lc["v_g_band"] == 0).sum())
    expected_offset = float(
        np.median(df_lc.loc[df_lc["v_g_band"] == 1, "mag"].to_numpy())
        - np.median(df_lc.loc[df_lc["v_g_band"] == 0, "mag"].to_numpy())
    )
    mag = captured["ce_mag"]
    assert np.isclose(np.median(mag[:g_count]), np.median(mag[g_count:]), atol=1e-10)
    assert out.loc[0, "pre_periodicity_router_mode"] == periodicity_gate.PREGATE_ROUTER_MODE
    assert np.isclose(out.loc[0, "pre_periodicity_v_minus_g_median_offset"], expected_offset, atol=1e-10)
    assert "pre_periodicity_pdm_method" not in out.columns


def test_apply_pre_periodicity_gate_uses_ce_corrected_period_as_selected_period(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        assert _kwargs.get("refine") is True
        return {
            "ce_period": 8.0,
            "ce_min_entropy": 0.3,
            "ce_snr": 11.0,
            "ce_bootstrap_sig": 2e-4,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        if np.isclose(base_period, 8.0):
            return 4.0, 0.5, {
                "objective": 0.3,
                "selection_objective": 0.3,
                "scatter_ratio": 0.5,
                "lag_phase": np.nan,
                "alias_flag": False,
            }
        raise AssertionError(f"unexpected base period {base_period}")

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["harmonic_pair"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert np.isclose(out.loc[0, "pre_ce_period"], 8.0)
    assert np.isclose(out.loc[0, "pre_ce_corrected_period"], 4.0)
    assert np.isclose(out.loc[0, "pre_ce_harmonic_factor"], 0.5)
    assert out.loc[0, "pre_periodicity_method"] == "ce"
    assert np.isclose(out.loc[0, "pre_periodicity_base_period"], 8.0)
    assert np.isclose(out.loc[0, "pre_periodicity_selected_period"], 4.0)
    assert np.isclose(out.loc[0, "pre_periodicity_harmonic_factor"], 0.5)
    for legacy_column in (
        "pre_periodicity_pdm_method",
        "pre_periodicity_methods_agree",
        "pre_periodicity_agreement_factor",
        "pre_periodicity_agreement_rel_err",
        "pre_pdm_period",
        "pre_pdm_corrected_period",
        "pre_pdm_harmonic_factor",
        "pre_pdm_bootstrap_sig",
        "pre_ce_bootstrap_sig",
        "pre_periodicity_bootstrap_sig",
        "pre_periodicity_is_significant",
        "pre_periodicity_lag_phase",
    ):
        assert legacy_column not in out.columns


def test_apply_pre_periodicity_gate_uses_ce_snr_and_folded_scatter_only(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.95,
            "ce_snr": 11.0,
            "ce_bootstrap_sig": 0.1,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        assert np.isclose(base_period, 4.0)
        return 4.0, 1.0, {
            "objective": 0.45,
            "selection_objective": 0.45,
            "scatter_ratio": 0.45,
            "lag_phase": np.nan,
            "phase_peak_regions": 1.0,
            "alias_flag": False,
        }

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["single_support"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert int(out.loc[0, "pre_periodicity_support_count"]) == 1
    assert out.loc[0, "pre_periodicity_reason"] == "ce,folded_scatter"


def test_apply_pre_periodicity_gate_rescues_phase_concentrated_scatter_near_miss(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.5,
            "ce_snr": 11.0,
            "ce_bootstrap_sig": 0.1,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        assert np.isclose(base_period, 4.0)
        return 4.0, 1.0, {
            "objective": 0.95,
            "selection_objective": 0.95,
            "scatter_ratio": 0.95,
            "phase_peak_snr": 3.8,
            "phase_peak_width": 0.18,
            "phase_peak_regions": 1.0,
            "alias_flag": False,
        }

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["scatter_near_miss"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert bool(out.loc[0, "pre_periodicity_phase_peak_flag"]) is True
    assert out.loc[0, "pre_periodicity_reason"] == "ce,phase_peak,scatter_rescue"


def test_apply_pre_periodicity_gate_rescues_phase_concentrated_ce_near_threshold(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.6,
            "ce_snr": 8.7,
            "ce_bootstrap_sig": 0.1,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        assert np.isclose(base_period, 4.0)
        return 4.0, 1.0, {
            "objective": 0.72,
            "selection_objective": 0.72,
            "scatter_ratio": 0.72,
            "phase_peak_snr": 3.4,
            "phase_peak_width": 0.16,
            "phase_peak_regions": 1.0,
            "alias_flag": False,
        }

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["ce_near_threshold"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert int(out.loc[0, "pre_periodicity_support_count"]) == 0
    assert bool(out.loc[0, "pre_periodicity_phase_peak_flag"]) is True
    assert out.loc[0, "pre_periodicity_reason"] == "ce_near_threshold,folded_scatter,phase_peak"


def test_apply_pre_periodicity_gate_rejects_complex_multi_peak_fold(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.5,
            "ce_snr": 15.0,
            "ce_bootstrap_sig": 0.1,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        assert np.isclose(base_period, 4.0)
        return 4.0, 1.0, {
            "objective": 0.55,
            "selection_objective": 0.55,
            "scatter_ratio": 0.55,
            "phase_peak_snr": 4.1,
            "phase_peak_width": 0.14,
            "phase_peak_regions": 4.0,
            "alias_flag": False,
        }

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["complex_fold"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is False
    assert out.loc[0, "pre_periodicity_label"] == "non_periodic"
    assert np.isclose(out.loc[0, "pre_periodicity_phase_peak_regions"], 4.0)
    assert "phase_complexity" in str(out.loc[0, "pre_periodicity_reason"])


def test_apply_pre_periodicity_gate_allows_alias_periodic_branch(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        return {
            "ce_period": 1.0,
            "ce_min_entropy": 0.4,
            "ce_snr": 15.0,
            "ce_bootstrap_sig": 0.1,
        }

    def fake_arbitrate(
        _band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
        base_period: float,
        *,
        min_period: float,
        max_period: float,
        **_kwargs: object,
    ) -> tuple[float, float, dict[str, object]]:
        assert min_period == 0.5
        assert max_period == 10.0
        assert np.isclose(base_period, 1.0)
        return 1.0, 1.0, {
            "objective": 0.45,
            "selection_objective": 0.45,
            "scatter_ratio": 0.45,
            "phase_peak_snr": 4.0,
            "phase_peak_width": 0.18,
            "phase_peak_regions": 1.0,
            "alias_flag": True,
        }

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_arbitrate_harmonic_period", fake_arbitrate)

    df = pd.DataFrame({"source_id": ["alias_fold"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert bool(out.loc[0, "pre_periodicity_alias_flag"]) is True
    assert out.loc[0, "pre_periodicity_reason"] == "ce,folded_scatter"


def test_arbitrate_harmonic_period_does_not_evaluate_longer_aliases(monkeypatch) -> None:
    raw_scores = {
        1.0: 0.40,
        2.0: 0.35,
        4.0: 0.01,
    }
    evaluated: list[float] = []

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        evaluated.append(float(period))
        objective = float(raw_scores.get(round(float(period), 10), np.inf))
        return {
            "objective": objective,
            "raw_objective": objective,
            "scatter_ratio": objective,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _ = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        2.0,
        min_period=0.2,
        max_period=20.0,
    )

    assert periodicity_gate.PREGATE_HARMONIC_FACTORS == (
        1.0,
        0.5,
        1.0 / 3.0,
        0.25,
    )
    assert evaluated == pytest.approx([2.0, 1.0, 2.0 / 3.0, 0.5])
    assert all(period <= 2.0 for period in evaluated)
    assert factor == 1.0
    assert selected_period == 2.0


def test_arbitrate_harmonic_period_uses_shortlist_half_when_best(monkeypatch) -> None:
    raw_scores = {
        10.0: 0.12,
        20.0: 0.18,
        40.0: 0.30,
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        objective = float(raw_scores.get(round(float(period), 10), np.inf))
        return {
            "objective": objective,
            "raw_objective": objective,
            "scatter_ratio": objective,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _ = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        20.0,
        min_period=0.2,
        max_period=100.0,
    )

    assert np.isclose(factor, 0.5)
    assert np.isclose(selected_period, 10.0)


def test_arbitrate_harmonic_period_keeps_base_when_harmonic_gain_is_too_small(monkeypatch) -> None:
    raw_scores = {
        5.0: 0.50,
        2.5: 0.495,
        10.0: 0.60,
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        objective = float(raw_scores.get(round(float(period), 10), np.inf))
        return {
            "objective": objective,
            "raw_objective": objective,
            "scatter_ratio": objective,
            "phase_peak_snr": 3.0,
            "phase_peak_width": 0.2,
            "phase_peak_regions": 1.0,
            "lag_phase": 0.0,
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _ = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        5.0,
        min_period=0.2,
        max_period=20.0,
    )

    assert factor == 1.0
    assert selected_period == 5.0


def test_arbitrate_harmonic_period_prefers_passing_subharmonic_over_complex_lower_scatter_fold(monkeypatch) -> None:
    raw_scores = {
        4.0: {"scatter_ratio": 0.24, "phase_peak_regions": 8.0, "phase_peak_snr": 1.2},
        2.0: {"scatter_ratio": 0.18, "phase_peak_regions": 4.0, "phase_peak_snr": 1.3},
        1.0: {"scatter_ratio": 0.19, "phase_peak_regions": 1.0, "phase_peak_snr": 1.4},
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        diag = raw_scores.get(round(float(period), 10))
        if diag is None:
            return {
                "objective": np.inf,
                "raw_objective": np.inf,
                "scatter_ratio": np.inf,
                "phase_peak_snr": np.nan,
                "phase_peak_width": np.nan,
                "phase_peak_regions": np.nan,
                "alias_flag": False,
                "alias_matches": [],
            }
        return {
            "objective": float(diag["scatter_ratio"]),
            "raw_objective": float(diag["scatter_ratio"]),
            "scatter_ratio": float(diag["scatter_ratio"]),
            "phase_peak_snr": float(diag["phase_peak_snr"]),
            "phase_peak_width": 0.2,
            "phase_peak_regions": float(diag["phase_peak_regions"]),
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _ = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        4.0,
        min_period=0.2,
        max_period=20.0,
    )

    assert np.isclose(factor, 0.25)
    assert np.isclose(selected_period, 1.0)


def test_arbitrate_harmonic_period_ignores_alias_status_when_ranking(monkeypatch) -> None:
    raw_scores = {
        4.0: {"scatter_ratio": 0.30, "phase_peak_regions": 3.0, "phase_peak_snr": 1.0, "alias_flag": False},
        2.0: {"scatter_ratio": 0.15, "phase_peak_regions": 1.0, "phase_peak_snr": 1.2, "alias_flag": False},
        1.0: {"scatter_ratio": 0.10, "phase_peak_regions": 1.0, "phase_peak_snr": 1.1, "alias_flag": True},
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        diag = raw_scores.get(round(float(period), 10))
        if diag is None:
            return {
                "objective": np.inf,
                "raw_objective": np.inf,
                "scatter_ratio": np.inf,
                "phase_peak_snr": np.nan,
                "phase_peak_width": np.nan,
                "phase_peak_regions": np.nan,
                "alias_flag": False,
                "alias_matches": [],
            }
        return {
            "objective": float(diag["scatter_ratio"]),
            "raw_objective": float(diag["scatter_ratio"]),
            "scatter_ratio": float(diag["scatter_ratio"]),
            "phase_peak_snr": float(diag["phase_peak_snr"]),
            "phase_peak_width": 0.2,
            "phase_peak_regions": float(diag["phase_peak_regions"]),
            "alias_flag": bool(diag["alias_flag"]),
            "alias_matches": [0.5] if bool(diag["alias_flag"]) else [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, diag = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        4.0,
        min_period=0.2,
        max_period=20.0,
    )

    assert np.isclose(factor, 0.25)
    assert np.isclose(selected_period, 1.0)
    assert bool(diag["alias_flag"]) is True


def test_arbitrate_harmonic_period_keeps_simpler_harmonic_even_when_gain_is_small(monkeypatch) -> None:
    raw_scores = {
        5.0: {"scatter_ratio": 0.50, "phase_peak_regions": 4.0, "phase_peak_snr": 1.2},
        2.5: {"scatter_ratio": 0.495, "phase_peak_regions": 1.0, "phase_peak_snr": 1.3},
        10.0: {"scatter_ratio": 0.60, "phase_peak_regions": 5.0, "phase_peak_snr": 1.0},
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        diag = raw_scores.get(round(float(period), 10))
        if diag is None:
            return {
                "objective": np.inf,
                "raw_objective": np.inf,
                "scatter_ratio": np.inf,
                "phase_peak_snr": np.nan,
                "phase_peak_width": np.nan,
                "phase_peak_regions": np.nan,
                "alias_flag": False,
                "alias_matches": [],
            }
        return {
            "objective": float(diag["scatter_ratio"]),
            "raw_objective": float(diag["scatter_ratio"]),
            "scatter_ratio": float(diag["scatter_ratio"]),
            "phase_peak_snr": float(diag["phase_peak_snr"]),
            "phase_peak_width": 0.2,
            "phase_peak_regions": float(diag["phase_peak_regions"]),
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _ = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        5.0,
        min_period=0.2,
        max_period=20.0,
    )

    assert np.isclose(factor, 0.5)
    assert np.isclose(selected_period, 2.5)


def test_arbitrate_harmonic_period_breaks_passing_ties_by_scatter_then_regions(monkeypatch) -> None:
    raw_scores = {
        4.0: {"scatter_ratio": 0.20, "phase_peak_regions": 2.0, "phase_peak_snr": 1.0},
        2.0: {"scatter_ratio": 0.15, "phase_peak_regions": 2.0, "phase_peak_snr": 1.2},
        1.0: {"scatter_ratio": 0.15, "phase_peak_regions": 1.0, "phase_peak_snr": 1.1},
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        diag = raw_scores.get(round(float(period), 10))
        if diag is None:
            return {
                "objective": np.inf,
                "raw_objective": np.inf,
                "scatter_ratio": np.inf,
                "phase_peak_snr": np.nan,
                "phase_peak_width": np.nan,
                "phase_peak_regions": np.nan,
                "alias_flag": False,
                "alias_matches": [],
            }
        return {
            "objective": float(diag["scatter_ratio"]),
            "raw_objective": float(diag["scatter_ratio"]),
            "scatter_ratio": float(diag["scatter_ratio"]),
            "phase_peak_snr": float(diag["phase_peak_snr"]),
            "phase_peak_width": 0.2,
            "phase_peak_regions": float(diag["phase_peak_regions"]),
            "alias_flag": False,
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    selected_period, factor, _ = periodicity_gate._arbitrate_harmonic_period(
        _dummy_band_residuals(),
        4.0,
        min_period=0.2,
        max_period=20.0,
    )

    assert np.isclose(factor, 0.25)
    assert np.isclose(selected_period, 1.0)


def test_apply_pre_periodicity_gate_selects_passing_subharmonic_regression(monkeypatch) -> None:
    df_lc = _two_band_offset_frame()

    def fake_load_lightcurve_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df_lc.copy()

    def fake_clean_lc(df: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def fake_ce_stats(_jd: np.ndarray, _mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.5,
            "ce_snr": 15.0,
            "ce_bootstrap_sig": 0.1,
        }

    raw_scores = {
        4.0: {"scatter_ratio": 0.24, "phase_peak_regions": 8.0, "phase_peak_snr": 1.2, "alias_flag": False},
        2.0: {"scatter_ratio": 0.18, "phase_peak_regions": 4.0, "phase_peak_snr": 1.3, "alias_flag": False},
        1.0: {"scatter_ratio": 0.19, "phase_peak_regions": 1.0, "phase_peak_snr": 1.4, "alias_flag": False},
    }

    def fake_score(_band_resid: dict[int, tuple[np.ndarray, np.ndarray]], period: float, **_kwargs: object) -> dict[str, object]:
        diag = raw_scores.get(round(float(period), 10))
        if diag is None:
            return {
                "objective": np.inf,
                "raw_objective": np.inf,
                "scatter_ratio": np.inf,
                "phase_peak_snr": np.nan,
                "phase_peak_width": np.nan,
                "phase_peak_regions": np.nan,
                "alias_flag": False,
                "alias_matches": [],
            }
        return {
            "objective": float(diag["scatter_ratio"]),
            "raw_objective": float(diag["scatter_ratio"]),
            "scatter_ratio": float(diag["scatter_ratio"]),
            "phase_peak_snr": float(diag["phase_peak_snr"]),
            "phase_peak_width": 0.2,
            "phase_peak_regions": float(diag["phase_peak_regions"]),
            "alias_flag": bool(diag["alias_flag"]),
            "alias_matches": [],
        }

    monkeypatch.setattr(periodicity_gate, "load_lightcurve_df", fake_load_lightcurve_df)
    monkeypatch.setattr(periodicity_gate, "clean_lc", fake_clean_lc)
    monkeypatch.setattr(periodicity_gate, "compute_ce_stats", fake_ce_stats)
    monkeypatch.setattr(periodicity_gate, "_score_period_harmonic_candidate", fake_score)

    df = pd.DataFrame({"source_id": ["subharmonic_regression"], "dat_path": ["dummy.dat3"]})
    out = apply_pre_periodicity_gate(
        df,
        n_periods=800,
        min_period=0.5,
        max_period=10.0,
        workers=1,
        show_tqdm=False,
    )

    assert bool(out.loc[0, "pre_periodic_flag"]) is True
    assert out.loc[0, "pre_periodicity_label"] == "periodic"
    assert np.isclose(out.loc[0, "pre_periodicity_harmonic_factor"], 0.25)
    assert np.isclose(out.loc[0, "pre_periodicity_selected_period"], 1.0)
    assert out.loc[0, "pre_periodicity_reason"] == "ce,folded_scatter"


def test_pre_periodicity_empty_result_includes_phase_lag_columns() -> None:
    result = periodicity_gate._empty_result("dummy.dat3", "dummy")

    assert "pre_periodicity_phase_lag_g_v_cycles" in result
    assert "pre_periodicity_phase_lag_g_v_abs_cycles" in result
    assert np.isnan(result["pre_periodicity_phase_lag_g_v_cycles"])
    assert np.isnan(result["pre_periodicity_phase_lag_g_v_abs_cycles"])
    assert "pre_periodicity_phase_lag_g_v_cycles" in periodicity_gate.PREGATE_RESULT_COLUMNS
    assert "pre_periodicity_phase_lag_g_v_abs_cycles" in periodicity_gate.PREGATE_RESULT_COLUMNS
