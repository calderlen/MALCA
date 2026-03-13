from __future__ import annotations

import numpy as np
import pandas as pd

from malca.pca import (
    DEFAULT_VARIABILITY_PCA_COLUMNS,
    apply_variability_pca,
    build_standardized_feature_matrix,
    fit_apply_variability_pca,
    fit_variability_pca,
    load_pca_model,
    save_pca_model,
)


def _synthetic_variability_frame(n_rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    mag = rng.uniform(11.0, 17.5, size=n_rows)
    latent_a = rng.normal(size=n_rows)
    latent_b = rng.normal(size=n_rows)

    df = pd.DataFrame({"baseline_mag": mag})
    for idx, col in enumerate(DEFAULT_VARIABILITY_PCA_COLUMNS):
        slope = 0.45 + 0.03 * idx
        noise = 0.2 + 0.02 * (idx % 4)
        values = slope * mag + 0.7 * latent_a + 0.25 * latent_b + rng.normal(scale=noise, size=n_rows)
        df[col] = values

        missing_mask = ((np.arange(n_rows) + idx) % (7 + (idx % 3))) == 0
        df.loc[missing_mask, col] = np.nan

    return df


def test_fit_apply_variability_pca_adds_expected_columns() -> None:
    df = _synthetic_variability_frame()

    out, model = fit_apply_variability_pca(
        df,
        n_components=3,
        bin_width=1.0,
        min_bin_count=15,
    )

    for col in model.normalized_columns:
        assert col in out.columns
    for col in model.pca_columns:
        assert col in out.columns
        assert out[col].notna().all()

    standardized = build_standardized_feature_matrix(df, model)
    assert list(standardized.columns) == model.feature_columns
    np.testing.assert_allclose(
        standardized.mean(axis=0).to_numpy(dtype=float),
        np.zeros(len(model.feature_columns), dtype=float),
        atol=1e-8,
    )


def test_magnitude_normalization_reduces_raw_mag_trend() -> None:
    df = _synthetic_variability_frame()
    feature_columns = [
        "stats_variability_stetson_J_time",
        "stats_variability_stetson_I",
        "stats_variability_von_neumann_ratio",
    ]

    model = fit_variability_pca(
        df,
        feature_columns=feature_columns,
        n_components=2,
        bin_width=1.0,
        min_bin_count=15,
    )
    out = apply_variability_pca(df, model)

    raw_corr = abs(df["stats_variability_stetson_J_time"].corr(df["baseline_mag"]))
    norm_corr = abs(out["stats_variability_stetson_J_time_magz"].corr(df["baseline_mag"]))

    assert np.isfinite(raw_corr)
    assert np.isfinite(norm_corr)
    assert norm_corr < raw_corr


def test_pca_model_round_trips_via_joblib(tmp_path) -> None:
    df = _synthetic_variability_frame()

    model = fit_variability_pca(
        df,
        n_components=2,
        bin_width=1.0,
        min_bin_count=15,
    )
    model_path = tmp_path / "variability_pca.joblib"
    save_pca_model(model, model_path)
    loaded = load_pca_model(model_path)

    out_expected = apply_variability_pca(df, model)
    out_loaded = apply_variability_pca(df, loaded)

    np.testing.assert_allclose(
        out_expected[model.pca_columns].to_numpy(dtype=float),
        out_loaded[loaded.pca_columns].to_numpy(dtype=float),
    )
