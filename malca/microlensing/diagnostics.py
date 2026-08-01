"""Transparent table diagnostics for joint microlensing fits."""

from __future__ import annotations

import json

import numpy as np

from .datasets import PhotometryDataset
from .joint_fit import JointFitResult
from .parallax import ParallaxResult
from .schema import MICROLENSING_JOINT_VERSION


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
        rows.append(row)
    return rows


__all__ = ["candidate_result_row", "dataset_result_rows", "peak_coverage"]
