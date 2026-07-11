from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.io.table_io import read_parquet_table, write_parquet_table
from malca.nuclear.arbitration import arbitrate_nuclear_scores
from malca.nuclear.injection import (
    build_amplitude_grid,
    build_timescale_grid,
    compute_recovery_summary,
    inject_agn_variability,
    inject_clagn_transition,
    inject_tde_flare,
    load_manifest,
    run_nuclear_injection_recovery,
    generate_plots,
    save_results_artifacts,
    score_injected_lightcurve,
    select_control_sample,
    synthetic_context_for_class,
    write_dat2_table,
)
from malca.nuclear.features import compute_nuclear_lightcurve_features
from malca.nuclear.redshift import resolve_redshift_spectral_types
from malca.nuclear.scoring import score_nuclear_candidates


def _light_curve(n_points: int = 160, *, span_days: float = 2600.0, mag: float = 17.0) -> pd.DataFrame:
    jd = np.linspace(2458000.0, 2458000.0 + span_days, n_points)
    return pd.DataFrame(
        {
            "JD": jd,
            "mag": np.full(n_points, mag),
            "error": np.full(n_points, 0.03),
            "good_bad": np.ones(n_points, dtype=int),
            "camera": ["1"] * n_points,
            "v_g_band": np.zeros(n_points, dtype=int),
            "saturated": np.zeros(n_points, dtype=int),
            "cam_field": ["cam/a"] * n_points,
        }
    )


def _source_row(dat_path: Path, source_id: str = "1001") -> pd.Series:
    return pd.Series(
        {
            "asas_sn_id": source_id,
            "dat_path": str(dat_path),
            "ra_deg": 10.0,
            "dec_deg": 20.0,
            "pstarrs_g_mag": 17.0,
            "n_points": 160,
        }
    )


def test_analytic_templates_create_expected_light_curve_morphologies() -> None:
    lc = _light_curve()
    rng = np.random.default_rng(7)

    agn = inject_agn_variability(lc, amplitude_mag=0.8, timescale_days=300.0, rng=rng)
    assert float(agn["mag"].max() - agn["mag"].min()) > 0.4

    tde, peak_mjd = inject_tde_flare(lc, amplitude_mag=1.5, timescale_days=250.0, rng=rng)
    tde_features = compute_nuclear_lightcurve_features(tde, peak_mjd=peak_mjd)
    assert float(lc["mag"].median() - tde["mag"].min()) > 1.0
    assert tde_features["tde_single_flare_score"] == 1.0
    assert tde_features["tde_quiet_baseline_score"] > 0.8

    clagn, direction = inject_clagn_transition(lc, amplitude_mag=1.2, timescale_days=400.0, rng=rng)
    clagn_features = compute_nuclear_lightcurve_features(clagn)
    assert direction in {-1, 1}
    assert abs(float(clagn_features["clagn_state_change_mag"])) > 0.7
    assert float(clagn_features["clagn_monotonicity_score"]) > 0.8


def test_synthetic_context_scores_and_arbitrates_to_expected_classes() -> None:
    rows = []
    rows.append(
        {
            "candidate_id": "AGN",
            **synthetic_context_for_class("agn"),
            "nuc_flux_frac_amp_p95_p05": 0.4,
        }
    )
    rows.append(
        {
            "candidate_id": "TDE",
            **synthetic_context_for_class("tde"),
            "tde_single_flare_score": 1.0,
            "tde_quiet_baseline_score": 1.0,
            "tde_no_recurrence_score": 1.0,
            "tde_smooth_decline_score": 1.0,
        }
    )
    rows.append(
        {
            "candidate_id": "CLAGN",
            **synthetic_context_for_class("clagn"),
            "clagn_state_change_mag": 1.5,
            "clagn_monotonicity_score": 1.0,
            "clagn_plateau_score": 1.0,
        }
    )
    rows.append({"candidate_id": "CONTROL", **synthetic_context_for_class("control")})

    scored = score_nuclear_candidates(resolve_redshift_spectral_types(pd.DataFrame(rows)))
    out = arbitrate_nuclear_scores(scored, min_score=0.5, min_margin=0.05).set_index("candidate_id")

    assert out.loc["AGN", "nuclear_primary_hypothesis"] == "agn"
    assert out.loc["TDE", "nuclear_primary_hypothesis"] == "tde"
    assert out.loc["CLAGN", "nuclear_primary_hypothesis"] == "clagn"
    assert out.loc["CONTROL", "nuclear_primary_hypothesis"] == "control"


def test_arbitration_handles_margin_and_low_score_controls() -> None:
    out = arbitrate_nuclear_scores(
        pd.DataFrame(
            [
                {"candidate_id": "A", "agn_prior_score": 0.8, "tde_candidate_score": 0.2, "clagn_photometric_score": 0.1},
                {"candidate_id": "B", "agn_prior_score": 0.51, "tde_candidate_score": 0.49, "clagn_photometric_score": 0.2},
                {"candidate_id": "C", "agn_prior_score": 0.2, "tde_candidate_score": 0.1, "clagn_photometric_score": 0.0},
            ]
        ),
        min_score=0.5,
        min_margin=0.05,
    ).set_index("candidate_id")

    assert out.loc["A", "nuclear_primary_hypothesis"] == "agn"
    assert out.loc["B", "nuclear_primary_hypothesis"] == "ambiguous"
    assert out.loc["C", "nuclear_primary_hypothesis"] == "control"


def test_score_injected_lightcurve_uses_template_features_and_context(tmp_path: Path) -> None:
    lc = _light_curve()
    dat_path = tmp_path / "1001.dat2"
    write_dat2_table(lc, dat_path)
    source = _source_row(dat_path)

    rng = np.random.default_rng(9)
    tde_lc, peak_mjd = inject_tde_flare(lc, amplitude_mag=1.5, timescale_days=250.0, rng=rng)
    scored = score_injected_lightcurve(
        tde_lc,
        source_row=source,
        id_col="asas_sn_id",
        dat_path=dat_path,
        truth_class="tde",
        peak_mjd=peak_mjd,
        min_score=0.5,
        min_margin=0.05,
    )

    assert scored.loc[0, "tde_candidate_score"] > 0.6
    assert scored.loc[0, "nuclear_primary_hypothesis"] == "tde"


def test_tiny_nuclear_injection_recovery_run_writes_resumable_results(tmp_path: Path) -> None:
    dat_paths = []
    for idx in range(2):
        path = tmp_path / f"{1000 + idx}.dat2"
        write_dat2_table(_light_curve(mag=17.0 + idx * 0.1), path)
        dat_paths.append(path)

    manifest_path = tmp_path / "manifest.parquet"
    manifest = pd.DataFrame(
        [
            {
                "asas_sn_id": str(1000 + idx),
                "dat_path": str(path),
                "ra_deg": 10.0 + idx,
                "dec_deg": 20.0 + idx,
                "pstarrs_g_mag": 17.0 + idx * 0.1,
                "n_points": 160,
            }
            for idx, path in enumerate(dat_paths)
        ]
    )
    write_parquet_table(manifest, manifest_path)

    loaded = load_manifest(manifest_path)
    sample = select_control_sample(loaded, n_sample=2, min_points=50, seed=1)
    output_path = tmp_path / "results" / "nuclear_injection_trials.parquet"
    results = run_nuclear_injection_recovery(
        sample,
        amplitude_values=build_amplitude_grid(1.2, 1.2, 1),
        timescale_values=build_timescale_grid(300.0, 300.0, 1),
        repeats_per_grid=1,
        classes=["agn", "tde", "clagn", "control"],
        seed=3,
        output_path=output_path,
        checkpoint_path=tmp_path / "results" / "nuclear_injection_trials_PROCESSED.txt",
        workers=1,
        chunk_size=2,
        max_trials=None,
    )

    assert output_path.exists()
    written = read_parquet_table(output_path)
    assert len(results) == 5
    assert len(written) == 5
    assert set(written["truth_class"]) == {"agn", "tde", "clagn", "control"}
    assert {"agn_prior_score", "tde_candidate_score", "clagn_photometric_score"}.issubset(written.columns)
    assert {"nuclear_primary_hypothesis", "recovered", "failure_reason"}.issubset(written.columns)
    assert "broad_line_change_flag" in written.columns

    summary = compute_recovery_summary(written)
    assert set(summary["truth_class"]) == {"agn", "tde", "clagn", "control"}

    plot_tables = generate_plots(
        written,
        amplitude_values=build_amplitude_grid(1.2, 1.2, 1),
        timescale_values=build_timescale_grid(300.0, 300.0, 1),
        output_dir=tmp_path / "plots",
        top_n_outcomes=4,
        n_mag_slices=0,
    )
    save_results_artifacts(written, results_dir=tmp_path / "artifacts", plot_tables=plot_tables)
    assert (tmp_path / "plots" / "arbitration_outcome_counts.png").exists()
    assert (tmp_path / "plots" / "confusion_matrix.png").exists()
    assert (tmp_path / "plots" / "agn_recovery_fraction_heatmap.png").exists()
    assert (tmp_path / "artifacts" / "aggregates" / "agn_recovery_fraction.parquet").exists()
