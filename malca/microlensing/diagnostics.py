"""Transparent table diagnostics for joint microlensing fits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from malca.enrichment.atlas_forced_photometry import summarize_atlas_residuals

from .datasets import PhotometryDataset
from .joint_fit import JointFitResult
from .parallax import ParallaxResult
from .schema import MICROLENSING_JOINT_VERSION


ATLAS_DIAGNOSTIC_COLUMNS = (
    "candidate_id",
    "dataset_id",
    "survey",
    "band",
    "instrument",
    "provenance_path",
    "atlas_noise_status",
    "atlas_noise_model_version",
    "atlas_filter",
    "atlas_obs_site_code",
    "atlas_obs_site",
    "diagnostic_scope",
    "chi_n_bin",
    "n_raw",
    "n_good",
    "n_rejected_quality",
    "n_loaded_good",
    "n_calibration",
    "n_noise_usable",
    "n_retained_fit",
    "n_downweighted_pre_fit",
    "n_excluded_noise_or_robust",
    "n_points",
    "median_formal_error_ujy",
    "median_effective_error_ujy",
    "median_noise_floor_ujy",
    "chi_n_median",
    "chi_n_p90",
    "chi_n_p99",
    "residual_median_ujy",
    "residual_robust_scatter_ujy",
    "reduced_chi2_formal",
    "reduced_chi2_effective",
    "fraction_gt3_formal",
    "fraction_gt5_formal",
    "fraction_gt10_formal",
    "fraction_gt3_effective",
    "fraction_gt5_effective",
    "fraction_gt10_effective",
    "median_robust_weight",
    "n_downweighted",
    "n_excluded",
)


def peak_coverage(dataset: PhotometryDataset, fit: JointFitResult) -> dict[str, object]:
    if not fit.success or not np.isfinite(fit.tE_days) or fit.tE_days <= 0.0:
        return {
            "peak_n_within_1_tE": 0,
            "peak_n_within_2_tE": 0,
            "peak_n_before": 0,
            "peak_n_after": 0,
            "peak_nearest_abs_tau": np.nan,
        }
    tau = (dataset.time_jd - fit.t0_jd) / fit.tE_days
    return {
        "peak_n_within_1_tE": int(np.sum(np.abs(tau) <= 1.0)),
        "peak_n_within_2_tE": int(np.sum(np.abs(tau) <= 2.0)),
        "peak_n_before": int(np.sum((tau >= -2.0) & (tau < 0.0))),
        "peak_n_after": int(np.sum((tau > 0.0) & (tau <= 2.0))),
        "peak_nearest_abs_tau": float(np.nanmin(np.abs(tau))) if tau.size else np.nan,
    }


def candidate_result_row(
    candidate_id: str,
    datasets: list[PhotometryDataset],
    fit: JointFitResult,
    *,
    parallax: ParallaxResult | None = None,
) -> dict[str, object]:
    surveys = sorted({dataset.survey for dataset in datasets})
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "microlensing_joint_version": MICROLENSING_JOINT_VERSION,
        "microlensing_joint_status": fit.status,
        "microlensing_joint_n_surveys": len(surveys),
        "microlensing_joint_n_datasets": len(datasets),
        "microlensing_joint_n_points": int(sum(dataset.n_points for dataset in datasets)),
        "microlensing_joint_surveys": ",".join(surveys),
        "microlensing_joint_dataset_ids_json": json.dumps([dataset.dataset_id for dataset in datasets]),
        "microlensing_joint_t0_jd": fit.t0_jd,
        "microlensing_joint_u0": fit.u0,
        "microlensing_joint_tE_days": fit.tE_days,
        "microlensing_joint_chi2": fit.chi2,
        "microlensing_joint_reduced_chi2": fit.reduced_chi2,
        "microlensing_joint_bic": fit.bic,
        "microlensing_joint_flat_chi2": fit.flat_chi2,
        "microlensing_joint_flat_bic": fit.flat_bic,
        "microlensing_joint_delta_bic_flat": fit.delta_bic_flat,
    }
    parallax = parallax or ParallaxResult(False, False, False, "not_requested")
    best = parallax.best
    row.update(
        {
            "microlensing_joint_parallax_attempted": parallax.attempted,
            "microlensing_joint_parallax_fit_ok": parallax.fit_ok,
            "microlensing_joint_parallax_preferred": parallax.preferred,
            "microlensing_joint_parallax_status": parallax.status,
            "microlensing_joint_parallax_best_branch": parallax.best_branch,
            "microlensing_joint_parallax_delta_bic": parallax.delta_bic,
            "microlensing_joint_parallax_branch_delta_bic": parallax.branch_delta_bic,
            "microlensing_joint_parallax_t0_jd": best.t0_jd if best else np.nan,
            "microlensing_joint_parallax_u0": best.u0 if best else np.nan,
            "microlensing_joint_parallax_tE_days": best.tE_days if best else np.nan,
            "microlensing_joint_piE_N": best.piE_N if best else np.nan,
            "microlensing_joint_piE_E": best.piE_E if best else np.nan,
            "microlensing_joint_piE": best.piE if best else np.nan,
            "microlensing_joint_parallax_reduced_chi2": best.reduced_chi2 if best else np.nan,
            "microlensing_joint_parallax_mcmc_json": json.dumps(best.mcmc_summary, sort_keys=True) if best else "{}",
        }
    )
    return row


def dataset_result_rows(
    candidate_id: str,
    datasets: list[PhotometryDataset],
    fit: JointFitResult,
    *,
    individual_fits: dict[str, JointFitResult],
    leave_one_survey_out: dict[str, JointFitResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        solution = fit.dataset_solutions.get(dataset.dataset_id)
        individual = individual_fits.get(dataset.dataset_id, JointFitResult(False, "not_run"))
        omitted = leave_one_survey_out.get(dataset.survey, JointFitResult(False, "not_run"))
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "microlensing_joint_version": MICROLENSING_JOINT_VERSION,
            "dataset_id": dataset.dataset_id,
            "survey": dataset.survey,
            "band": dataset.band,
            "instrument": dataset.instrument,
            "flux_kind": dataset.flux_kind,
            "provenance_path": dataset.provenance_path,
            "reference_mag": dataset.reference_mag,
            "n_points": dataset.n_points,
            "jd_first": float(np.nanmin(dataset.time_jd)),
            "jd_last": float(np.nanmax(dataset.time_jd)),
            "joint_source_flux": solution.source_flux if solution else np.nan,
            "joint_blend_flux": solution.blend_flux if solution else np.nan,
            "joint_difference_offset_flux": solution.reference_difference_flux if solution else np.nan,
            "joint_dataset_chi2": solution.chi2 if solution else np.nan,
            "joint_dataset_reduced_chi2": (
                solution.chi2 / max(dataset.n_points - 2, 1) if solution else np.nan
            ),
            "individual_status": individual.status,
            "individual_t0_jd": individual.t0_jd,
            "individual_u0": individual.u0,
            "individual_tE_days": individual.tE_days,
            "individual_reduced_chi2": individual.reduced_chi2,
            "individual_delta_bic_flat": individual.delta_bic_flat,
            "loo_omitted_survey": dataset.survey,
            "loo_status": omitted.status,
            "loo_t0_jd": omitted.t0_jd,
            "loo_u0": omitted.u0,
            "loo_tE_days": omitted.tE_days,
            "loo_reduced_chi2": omitted.reduced_chi2,
            "loo_delta_bic_flat": omitted.delta_bic_flat,
        }
        row.update(peak_coverage(dataset, fit))
        if dataset.survey == "atlas":
            floor = np.asarray(
                dataset.point_metadata.get("atlas_noise_floor_ujy", np.array([], dtype=float)),
                dtype=float,
            )
            row.update(
                {
                    "atlas_noise_model_version": dataset.metadata.get(
                        "atlas_noise_model_version", ""
                    ),
                    "atlas_noise_status": dataset.metadata.get("atlas_noise_status", ""),
                    "atlas_reference_flux_ujy": dataset.metadata.get(
                        "atlas_reference_flux_ujy", np.nan
                    ),
                    "atlas_median_noise_floor_ujy": (
                        float(np.nanmedian(floor)) if np.any(np.isfinite(floor)) else np.nan
                    ),
                    "atlas_n_calibration": dataset.metadata.get("atlas_n_calibration", 0),
                    "atlas_n_noise_unusable": dataset.metadata.get(
                        "atlas_n_noise_unusable", 0
                    ),
                    "atlas_n_downweighted": dataset.metadata.get("atlas_n_downweighted", 0),
                    "atlas_n_excluded_robust": dataset.metadata.get(
                        "atlas_n_excluded_robust", 0
                    ),
                    "atlas_huber_tuning": dataset.metadata.get("atlas_huber_tuning", np.nan),
                }
            )
        rows.append(row)
    return rows


def atlas_diagnostic_rows(
    candidate_id: str,
    datasets: list[PhotometryDataset],
    fit: JointFitResult,
) -> list[dict[str, object]]:
    """Return site- and chi/N-resolved ATLAS residual diagnostics."""
    output: list[dict[str, object]] = []
    for dataset in datasets:
        if dataset.survey != "atlas":
            continue

        quality_counts = {
            str(row.get("atlas_obs_site_code", "")): dict(row)
            for row in dataset.metadata.get("atlas_counts_by_site", [])
        }
        calibration_counts = {
            str(row.get("atlas_obs_site_code", "")): dict(row)
            for row in dataset.metadata.get("atlas_calibration_counts_by_site", [])
        }
        solution = fit.dataset_solutions.get(dataset.dataset_id) if fit.success else None
        calibrated_columns = {
            "atlas_filter",
            "atlas_obs_site_code",
            "atlas_obs_site",
            "atlas_chi_n",
            "atlas_flux_error_formal_ujy",
            "atlas_flux_error_eff_ujy",
        }
        frame = pd.DataFrame(dataset.point_metadata)
        summaries = pd.DataFrame()
        if solution is not None and calibrated_columns.issubset(frame.columns):
            reference_flux_ujy = float(
                dataset.metadata.get("atlas_reference_flux_ujy", np.nan)
            )
            if np.isfinite(reference_flux_ujy) and reference_flux_ujy > 0.0:
                residual_ujy = (dataset.flux - solution.model_flux) * reference_flux_ujy
                robust_weights = dataset.point_metadata.get("atlas_robust_weight")
                summaries = summarize_atlas_residuals(
                    frame,
                    residual_flux_ujy=residual_ujy,
                    robust_weights=robust_weights,
                )

        summary_site_codes = (
            set(summaries["atlas_obs_site_code"].astype(str))
            if not summaries.empty
            else set()
        )
        all_site_codes = sorted(set(quality_counts) | set(calibration_counts) | summary_site_codes)
        if summaries.empty:
            summaries = pd.DataFrame(
                [
                    {
                        "atlas_filter": dataset.band,
                        "atlas_obs_site_code": site_code,
                        "atlas_obs_site": (
                            calibration_counts.get(site_code, {}).get("atlas_obs_site")
                            or quality_counts.get(site_code, {}).get("atlas_obs_site")
                            or "unknown"
                        ),
                        "diagnostic_scope": "site",
                        "chi_n_bin": "all",
                    }
                    for site_code in all_site_codes
                ]
            )
        else:
            missing_site_codes = sorted(set(all_site_codes) - summary_site_codes)
            if missing_site_codes:
                summaries = pd.concat(
                    [
                        summaries,
                        pd.DataFrame(
                            [
                                {
                                    "atlas_filter": dataset.band,
                                    "atlas_obs_site_code": site_code,
                                    "atlas_obs_site": (
                                        calibration_counts.get(site_code, {}).get("atlas_obs_site")
                                        or quality_counts.get(site_code, {}).get("atlas_obs_site")
                                        or "unknown"
                                    ),
                                    "diagnostic_scope": "site",
                                    "chi_n_bin": "all",
                                }
                                for site_code in missing_site_codes
                            ]
                        ),
                    ],
                    ignore_index=True,
                    sort=False,
                )

        for summary in summaries.to_dict(orient="records"):
            site_code = str(summary.get("atlas_obs_site_code", ""))
            quality = quality_counts.get(site_code, {})
            calibration = calibration_counts.get(site_code, {})
            row = {
                "candidate_id": candidate_id,
                "dataset_id": dataset.dataset_id,
                "survey": dataset.survey,
                "band": dataset.band,
                "instrument": dataset.instrument,
                "provenance_path": dataset.provenance_path,
                "atlas_noise_status": dataset.metadata.get("atlas_noise_status", "not_calibrated"),
                "atlas_noise_model_version": dataset.metadata.get(
                    "atlas_noise_model_version", ""
                ),
                "n_raw": quality.get("n_raw", 0),
                "n_good": quality.get("n_good", 0),
                "n_rejected_quality": quality.get("n_rejected_quality", 0),
                "n_loaded_good": calibration.get("n_loaded_good", quality.get("n_good", 0)),
                "n_calibration": calibration.get("n_calibration", 0),
                "n_noise_usable": calibration.get("n_noise_usable", 0),
                "n_retained_fit": calibration.get("n_retained_fit", 0),
                "n_downweighted_pre_fit": calibration.get("n_downweighted", 0),
                "n_excluded_noise_or_robust": calibration.get(
                    "n_excluded_noise_or_robust", 0
                ),
            }
            row.update(summary)
            output.append({column: row.get(column, np.nan) for column in ATLAS_DIAGNOSTIC_COLUMNS})
    return output


__all__ = [
    "ATLAS_DIAGNOSTIC_COLUMNS",
    "atlas_diagnostic_rows",
    "candidate_result_row",
    "dataset_result_rows",
    "peak_coverage",
]
