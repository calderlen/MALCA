from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.ltv.core import Config
from malca.ltv import injection as ltv_injection


def _write_dat2(path: Path, mags: list[float]) -> None:
    lines = [
        f"{1000.0 + 10.0 * i:.1f} {mag:.3f} 0.05 1 1 0 0 cam1/field1"
        for i, mag in enumerate(mags)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _test_cfg() -> Config:
    return Config(
        root=Path("."),
        mag_bin="test",
        output=Path("."),
        dspring=2460023.5,
        ra_is_deg=True,
        max_seasons=12,
        min_points_per_season=1,
        min_seasons_for_quadratic=3,
        write_per_dir=False,
        band_mode="g_only",
        workers=1,
        chunk_size=1,
        overwrite=False,
    )


def test_inject_trend_adds_monotonic_signal() -> None:
    df = pd.DataFrame(
        {
            "JD": [1000.0, 1010.0, 1020.0, 1030.0],
            "mag": [14.0, 14.0, 14.0, 14.0],
            "error": [0.05, 0.05, 0.05, 0.05],
            "good_bad": [1, 1, 1, 1],
            "camera": ["1", "1", "1", "1"],
            "v_g_band": [0, 0, 0, 0],
            "saturated": [0, 0, 0, 0],
            "cam_field": ["cam1/field1"] * 4,
        }
    )

    out = ltv_injection.inject_trend(
        df,
        amplitude_mag=0.6,
        timescale_days=5.0,
        direction=1,
        profile="tanh",
    )

    assert np.allclose(df["mag"].to_numpy(), 14.0)
    assert out["mag"].iloc[0] < out["mag"].iloc[-1]
    assert not np.allclose(out["mag"].to_numpy(), df["mag"].to_numpy())


def test_compute_rejection_summary_and_plot_tables() -> None:
    df = pd.DataFrame(
        {
            "amplitude_mag": [0.1, 0.1, 0.5, 0.5],
            "timescale_days": [10.0, 100.0, 10.0, 100.0],
            "pstarrs_g_mag": [13.0, 14.0, 15.0, 16.0],
            "passed": [True, False, True, False],
            "filter_reason": ["passed", "mock_filter", "passed", "core_no_metrics"],
        }
    )

    summary = ltv_injection.compute_rejection_summary(df)
    assert set(summary["filter_reason"]) == {"passed", "mock_filter", "core_no_metrics"}

    plot_tables = ltv_injection.compute_plot_tables(
        df,
        amplitude_values=np.array([0.1, 0.5]),
        timescale_values=np.array([10.0, 100.0]),
        top_n_reasons=2,
    )
    assert "pass_fraction" in plot_tables
    assert plot_tables["pass_fraction"].shape == (2, 2)
    mag_slices = ltv_injection.compute_magnitude_slices(df, n_slices=2)
    assert len(mag_slices) == 2


def test_generate_plots_writes_expected_files(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "amplitude_mag": [0.1, 0.1, 0.5, 0.5],
            "timescale_days": [10.0, 100.0, 10.0, 100.0],
            "pstarrs_g_mag": [13.0, 14.0, 15.0, 16.0],
            "passed": [True, False, True, False],
            "filter_reason": ["passed", "mock_filter", "passed", "core_no_metrics"],
        }
    )

    ltv_injection.generate_plots(
        df,
        amplitude_values=np.array([0.1, 0.5]),
        timescale_values=np.array([10.0, 100.0]),
        output_dir=tmp_path,
        top_n_reasons=2,
        n_mag_slices=2,
    )

    assert (tmp_path / "rejection_reason_counts.png").exists()
    assert (tmp_path / "pass_fraction_heatmap.png").exists()
    assert (tmp_path / "plot_tables" / "pass_fraction.csv").exists()
    slice_files = list((tmp_path / "magnitude_slices").glob("*_pass_fraction_heatmap.png"))
    assert len(slice_files) == 2
    slice_table_dirs = list((tmp_path / "plot_tables" / "magnitude_slices").glob("gmag_*"))
    assert len(slice_table_dirs) == 2


def test_run_injection_recovery_writes_trial_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dat2_path = tmp_path / "123.dat2"
    _write_dat2(dat2_path, [14.0, 14.0, 14.0, 14.0])

    control_sample = pd.DataFrame(
        {
            "asas_sn_id": [123],
            "ra_deg": [10.0],
            "dec_deg": [20.0],
            "pstarrs_g_mag": [14.2],
            "dat_path": [str(dat2_path)],
        }
    )

    def _fake_process_one_lc(path: str, meta, cfg) -> dict:
        df = pd.read_csv(path, header=None, sep=r"\s+")
        delta = float(df.iloc[-1, 1] - df.iloc[0, 1])
        return {
            "ASAS-SN ID": meta.asas_sn_id,
            "ra_deg": meta.ra_deg,
            "dec_deg": meta.dec_deg,
            "Pstarss gmag": meta.pstarrs_g_mag,
            "Median": float(df.iloc[:, 1].median()),
            "Slope": abs(delta),
            "max diff": abs(delta),
            "ls_fap": 0.5,
            "lc_path": path,
        }

    def _fake_apply_all_filters(df: pd.DataFrame, *, return_rejected: bool = False, **kwargs):
        if float(df.iloc[0]["Slope"]) >= 0.12:
            passed = df.copy()
            rejected = pd.DataFrame(columns=list(df.columns) + ["filter_reason"])
        else:
            passed = df.iloc[0:0].copy()
            rejected = df.copy()
            rejected["filter_reason"] = "mock_filter"
        if return_rejected:
            return passed, rejected
        return passed

    monkeypatch.setattr(ltv_injection, "process_one_lc", _fake_process_one_lc)
    monkeypatch.setattr(ltv_injection, "apply_all_filters", _fake_apply_all_filters)

    output_path = tmp_path / "results.csv"
    results_df = ltv_injection.run_injection_recovery(
        control_sample,
        amplitude_values=np.array([0.1, 0.5]),
        timescale_values=np.array([10.0, 100.0]),
        repeats_per_grid=1,
        profile="tanh",
        direction_mode="positive",
        cfg=_test_cfg(),
        filter_kwargs={"query_gaia": False, "verbose": False},
        seed=1,
        workers=1,
        output_path=output_path,
        checkpoint_path=tmp_path / "checkpoint.txt",
        resume=False,
        overwrite=True,
    )

    assert output_path.exists()
    assert len(results_df) == 4
    assert set(results_df["filter_reason"]) == {"mock_filter", "passed"}

    plot_tables = ltv_injection.generate_plots(
        results_df,
        amplitude_values=np.array([0.1, 0.5]),
        timescale_values=np.array([10.0, 100.0]),
        output_dir=tmp_path / "plots",
        n_mag_slices=1,
    )
    ltv_injection.save_results_artifacts(results_df, results_dir=tmp_path / "artifacts", plot_tables=plot_tables)

    assert (tmp_path / "artifacts" / "ltv_injection_trials.csv").exists()
    assert (tmp_path / "artifacts" / "ltv_rejection_summary.csv").exists()
    assert (tmp_path / "plots" / "pass_fraction_heatmap.png").exists()
