from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess_atlas_photometry import (
    ATLAS_PREPROCESS_VERSION,
    preprocess_atlas_frame,
    run_preprocessing,
)
from malca.enrichment.atlas_forced_photometry import (
    apply_atlas_noise_model,
    atlas_huber_weights,
    estimate_atlas_noise_model,
    summarize_atlas_residuals,
)
from malca.review.lightcurve_sources import normalize_external_lc_dataframe


def _atlas_frame(rows: int = 1) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "MJD": [59000.0 + index for index in range(rows)],
            "m": [-20.0] * rows,
            "dm": [99.0] * rows,
            "uJy": [100.0] * rows,
            "duJy": [10.0] * rows,
            "F": ["c"] * rows,
            "err": [0] * rows,
            "chi/N": [1.0] * rows,
            "x": [500.0] * rows,
            "y": [500.0] * rows,
            "maj": [2.5] * rows,
            "min": [2.2] * rows,
            "apfit": [-0.5] * rows,
            "mag5sig": [19.0] * rows,
            "Sky": [20.0] * rows,
            "atlas_image_type": ["reduced"] * rows,
            "Obs": [f"01a{index:05d}" for index in range(rows)],
        }
    )
    frame.attrs["atlas_image_types"] = ["reduced"]
    return frame


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("duJy", 10000.0),
        ("err", 1),
        ("x", 100.0),
        ("x", 10460.0),
        ("y", 100.0),
        ("y", 10460.0),
        ("maj", 1.6),
        ("maj", 5.0),
        ("min", 1.6),
        ("min", 5.0),
        ("apfit", -1.0),
        ("apfit", -0.1),
        ("mag5sig", 17.0),
        ("Sky", 17.0),
    ],
)
def test_each_faq_boundary_is_strictly_rejected(column: str, value: float) -> None:
    raw = _atlas_frame()
    raw.loc[0, column] = value

    flagged = preprocess_atlas_frame(raw)

    assert not bool(flagged.loc[0, "atlas_faq_good"])
    assert not bool(flagged.loc[0, "atlas_good"])
    assert flagged.loc[0, "atlas_reject_reason"]


def test_clean_magnitude_is_derived_from_flux_without_changing_raw_magnitude() -> None:
    raw = _atlas_frame()

    flagged = preprocess_atlas_frame(raw, snr_min=5)

    assert bool(flagged.loc[0, "atlas_good"])
    assert flagged.loc[0, "m"] == -20.0
    assert flagged.loc[0, "dm"] == 99.0
    assert flagged.loc[0, "flux_snr"] == pytest.approx(10.0)
    assert flagged.loc[0, "m_clean"] == pytest.approx(
        23.9 - 2.5 * np.log10(100.0)
    )
    assert flagged.loc[0, "dm_clean"] == pytest.approx(
        (2.5 / np.log(10.0)) * 10.0 / 100.0
    )
    assert flagged.loc[0, "atlas_obs_site_code"] == "01"
    assert flagged.loc[0, "atlas_obs_site"] == "maunaloa"
    assert flagged.loc[0, "atlas_camera"] == "a"
    assert flagged.loc[0, "atlas_obs"] == "01a00000"
    assert flagged.loc[0, "atlas_flux_ujy"] == 100.0
    assert flagged.loc[0, "atlas_flux_error_formal_ujy"] == 10.0


def test_empirical_noise_model_adds_site_floor_without_overwriting_formal_errors() -> None:
    raw = _atlas_frame(rows=100)
    raw["duJy"] = 2.0
    raw["uJy"] = 100.0 + np.tile([-8.0, -4.0, 0.0, 4.0, 8.0], 20)
    raw["Obs"] = [
        f"{'01' if index < 50 else '02'}a{index:05d}" for index in range(len(raw))
    ]
    flagged = preprocess_atlas_frame(raw)
    original_formal = flagged["duJy"].copy()

    model = estimate_atlas_noise_model(
        flagged,
        calibration_mask=np.ones(len(flagged), dtype=bool),
        min_points=20,
        min_time_span_days=20.0,
    )
    calibrated = apply_atlas_noise_model(flagged, model)

    site_models = model.loc[model["atlas_noise_scope"].eq("site")]
    assert len(site_models) == 2
    assert site_models["atlas_noise_usable"].all()
    assert (site_models["atlas_noise_floor_ujy"] > 0.0).all()
    pd.testing.assert_series_equal(calibrated["duJy"], original_formal)
    assert np.all(
        calibrated.loc[calibrated["atlas_good"], "atlas_flux_error_eff_ujy"]
        > calibrated.loc[calibrated["atlas_good"], "atlas_flux_error_formal_ujy"]
    )


def test_huber_weights_and_residual_diagnostics_preserve_chi_n_as_diagnostic() -> None:
    raw = _atlas_frame(rows=60)
    raw["duJy"] = 2.0
    raw["uJy"] = 100.0 + np.tile([-5.0, 0.0, 5.0], 20)
    raw["chi/N"] = np.linspace(1.0, 1200.0, len(raw))
    flagged = preprocess_atlas_frame(raw)
    model = estimate_atlas_noise_model(
        flagged,
        calibration_mask=np.ones(len(flagged), dtype=bool),
        min_points=20,
        min_time_span_days=20.0,
    )
    calibrated = apply_atlas_noise_model(flagged, model)
    residual = calibrated["atlas_flux_ujy"].to_numpy(dtype=float) - 100.0
    weights = atlas_huber_weights(
        np.array([0.0, 10.0]), np.array([1.0, 1.0]), tuning=5.0
    )
    np.testing.assert_allclose(weights, [1.0, 0.5])

    diagnostics = summarize_atlas_residuals(
        calibrated,
        residual_flux_ujy=residual,
    )

    assert {"site", "chi_n_bin"}.issubset(set(diagnostics["diagnostic_scope"]))
    site = diagnostics.loc[diagnostics["diagnostic_scope"].eq("site")].iloc[0]
    assert site["reduced_chi2_effective"] < site["reduced_chi2_formal"]
    assert site["chi_n_p90"] > site["chi_n_median"]


def test_flux_filter_snr_and_image_type_are_separate_from_faq_mask() -> None:
    raw = pd.concat([_atlas_frame() for _ in range(4)], ignore_index=True)
    raw["atlas_image_type"] = ["reduced", "reduced", "reduced", "difference"]
    raw["F"] = ["c", "o", "H", "c"]
    raw["uJy"] = [100.0, -100.0, 100.0, 100.0]
    raw["duJy"] = [10.0, 10.0, 10.0, 10.0]

    flagged = preprocess_atlas_frame(raw, snr_min=5)

    assert flagged["atlas_faq_good"].tolist() == [True, True, True, True]
    assert flagged["atlas_good"].tolist() == [True, False, False, False]
    assert "uJy_not_positive_or_finite" in flagged.loc[1, "atlas_reject_reason"]
    assert "filter_not_c_or_o" in flagged.loc[2, "atlas_reject_reason"]
    assert "image_type_not_reduced" in flagged.loc[3, "atlas_reject_reason"]
    assert flagged.loc[1:, "m_clean"].isna().all()


def test_low_snr_positive_flux_is_retained_but_flagged_as_not_a_detection() -> None:
    raw = _atlas_frame()
    raw.loc[0, "uJy"] = 40.0
    raw.loc[0, "duJy"] = 10.0

    flagged = preprocess_atlas_frame(raw, snr_min=5)

    assert bool(flagged.loc[0, "atlas_flux_good"])
    assert not bool(flagged.loc[0, "atlas_snr_good"])
    assert not bool(flagged.loc[0, "atlas_good"])
    assert flagged.loc[0, "atlas_reject_reason"] == "flux_snr_lt_5"
    assert pd.isna(flagged.loc[0, "m_clean"])


def test_review_normalization_uses_only_clean_flux_derived_magnitudes() -> None:
    raw = _atlas_frame(rows=2)
    raw.loc[1, "uJy"] = -20.0

    normalized = normalize_external_lc_dataframe("atlas", raw)

    assert len(normalized) == 1
    assert normalized["atlas_good"].tolist() == [True]
    assert normalized.loc[0, "mag"] == pytest.approx(
        23.9 - 2.5 * np.log10(100.0)
    )
    assert normalized.loc[0, "mag_err"] == pytest.approx(
        (2.5 / np.log(10.0)) * 10.0 / 100.0
    )
    assert normalized.loc[0, "m"] == -20.0
    assert normalized.loc[0, "dm"] == 99.0


def test_batch_processing_writes_clean_and_flagged_copies_without_touching_input(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "preprocessed"
    input_dir.mkdir()
    source = input_dir / "atlas_lc_C0.parquet"

    raw = _atlas_frame(rows=2)
    raw.loc[1, "uJy"] = -20.0
    raw.to_parquet(source, index=False)
    original = pd.read_parquet(source)

    summary = run_preprocessing(input_dir, output_dir)

    clean_path = output_dir / "clean" / source.name
    flagged_path = output_dir / "flagged" / source.name
    clean = pd.read_parquet(clean_path)
    flagged = pd.read_parquet(flagged_path)
    unchanged = pd.read_parquet(source)

    assert summary["files_processed"] == 1
    assert summary["rows_total"] == 2
    assert summary["rows_good"] == 1
    assert len(clean) == 1
    assert len(flagged) == 2
    assert clean.loc[0, "m"] == -20.0
    assert np.isfinite(clean.loc[0, "m_clean"])
    assert clean.loc[0, "mag"] == pytest.approx(clean.loc[0, "m_clean"])
    assert clean.loc[0, "mag_err"] == pytest.approx(clean.loc[0, "dm_clean"])
    assert clean.loc[0, "mag_raw"] == -20.0
    assert flagged["atlas_good"].tolist() == [True, False]
    assert flagged.attrs["atlas_preprocess_version"] == ATLAS_PREPROCESS_VERSION
    pd.testing.assert_frame_equal(unchanged, original)
    assert "atlas_good" not in unchanged.columns
    assert (output_dir / "atlas_preprocess_summary.json").exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "preprocessed"
    input_dir.mkdir()
    _atlas_frame().to_parquet(input_dir / "atlas_lc_C0.parquet", index=False)

    summary = run_preprocessing(input_dir, output_dir, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["rows_good"] == 1
    assert not output_dir.exists()
