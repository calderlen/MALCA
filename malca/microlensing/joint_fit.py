"""Profiled multi-survey PSPL geometry fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from malca.config import (
    PACZYNSKI_TE_MAX_FACTOR,
    PACZYNSKI_TE_MIN_DAYS,
    PACZYNSKI_U0_MAX,
    PACZYNSKI_U0_MIN,
)

from .datasets import PhotometryDataset
from .pspl import LinearFluxSolution, pspl_magnification, solve_linear_flux_parameters


@dataclass
class JointFitResult:
    success: bool
    status: str
    t0_jd: float = np.nan
    u0: float = np.nan
    tE_days: float = np.nan
    chi2: float = np.nan
    reduced_chi2: float = np.nan
    bic: float = np.nan
    flat_chi2: float = np.nan
    flat_bic: float = np.nan
    delta_bic_flat: float = np.nan
    n_points: int = 0
    n_datasets: int = 0
    n_parameters: int = 0
    dataset_solutions: dict[str, LinearFluxSolution] = field(default_factory=dict)
    magnifications: dict[str, np.ndarray] = field(default_factory=dict)
    optimizer_message: str = ""


def _flat_statistics(datasets: list[PhotometryDataset]) -> tuple[float, float]:
    chi2 = 0.0
    n_points = 0
    for dataset in datasets:
        weights = 1.0 / np.square(dataset.flux_error)
        mean = float(np.sum(weights * dataset.flux) / np.sum(weights))
        chi2 += float(np.sum(np.square((dataset.flux - mean) / dataset.flux_error)))
        n_points += dataset.n_points
    n_parameters = len(datasets)
    bic = chi2 + n_parameters * np.log(max(n_points, 2))
    return float(chi2), float(bic)


def _profile_magnifications(
    datasets: list[PhotometryDataset],
    magnifications: dict[str, np.ndarray],
) -> tuple[dict[str, LinearFluxSolution], float, np.ndarray]:
    solutions: dict[str, LinearFluxSolution] = {}
    residuals: list[np.ndarray] = []
    chi2 = 0.0
    for dataset in datasets:
        solution = solve_linear_flux_parameters(
            magnifications[dataset.dataset_id],
            dataset.flux,
            dataset.flux_error,
            flux_kind=dataset.flux_kind,
        )
        solutions[dataset.dataset_id] = solution
        if not solution.success or not np.all(np.isfinite(solution.residuals)):
            return solutions, np.nan, np.full(sum(item.n_points for item in datasets), 1e6)
        residuals.append(solution.residuals)
        chi2 += solution.chi2
    return solutions, float(chi2), np.concatenate(residuals)


def profile_joint_pspl(
    datasets: Iterable[PhotometryDataset],
    *,
    t0: float,
    u0: float,
    tE: float,
) -> tuple[dict[str, LinearFluxSolution], float, np.ndarray]:
    """Profile all per-dataset flux terms at one shared PSPL geometry."""
    dataset_list = list(datasets)
    magnifications = {
        dataset.dataset_id: pspl_magnification(dataset.time_jd, t0=t0, u0=u0, tE=tE)
        for dataset in dataset_list
    }
    return _profile_magnifications(dataset_list, magnifications)


def _geometry_bounds(datasets: list[PhotometryDataset]) -> tuple[np.ndarray, np.ndarray]:
    all_time = np.concatenate([dataset.time_jd for dataset in datasets])
    span = max(float(np.nanmax(all_time) - np.nanmin(all_time)), 30.0)
    tE_max = min(max(25.0, PACZYNSKI_TE_MAX_FACTOR * span), 5000.0)
    lower = np.array([np.log(PACZYNSKI_U0_MIN), np.nanmin(all_time), np.log(PACZYNSKI_TE_MIN_DAYS)])
    upper = np.array([np.log(PACZYNSKI_U0_MAX), np.nanmax(all_time), np.log(tE_max)])
    return lower, upper


def _data_seed(datasets: list[PhotometryDataset]) -> tuple[float, float, float]:
    best_score = -np.inf
    best_time = np.nan
    all_time = np.concatenate([dataset.time_jd for dataset in datasets])
    for dataset in datasets:
        baseline = float(np.nanmedian(dataset.flux))
        score = (dataset.flux - baseline) / dataset.flux_error
        if not np.any(np.isfinite(score)):
            continue
        index = int(np.nanargmax(score))
        if float(score[index]) > best_score:
            best_score = float(score[index])
            best_time = float(dataset.time_jd[index])
    if not np.isfinite(best_time):
        best_time = float(np.nanmedian(all_time))
    span = max(float(np.nanmax(all_time) - np.nanmin(all_time)), 30.0)
    return best_time, 0.2, max(PACZYNSKI_TE_MIN_DAYS, 0.08 * span)


def _fit_pspl(
    datasets: list[PhotometryDataset],
    *,
    seed: tuple[float, float, float] | None = None,
) -> JointFitResult:
    if not datasets:
        return JointFitResult(False, "no_datasets")
    if sum(dataset.n_points for dataset in datasets) < 8:
        return JointFitResult(False, "too_few_points")

    lower, upper = _geometry_bounds(datasets)
    t0_seed, u0_seed, tE_seed = seed or _data_seed(datasets)
    t0_seed = float(np.clip(t0_seed, lower[1], upper[1]))
    u0_seed = float(np.clip(abs(u0_seed), np.exp(lower[0]), np.exp(upper[0])))
    tE_seed = float(np.clip(abs(tE_seed), np.exp(lower[2]), np.exp(upper[2])))
    data_t0, _, data_tE = _data_seed(datasets)
    starts = [
        (u0_seed, t0_seed, tE_seed),
        (0.05, data_t0, max(0.5 * tE_seed, PACZYNSKI_TE_MIN_DAYS)),
        (0.25, data_t0, data_tE),
        (1.0, data_t0, min(2.0 * data_tE, np.exp(upper[2]))),
    ]

    def residual_function(opt: np.ndarray) -> np.ndarray:
        _, _, residuals = profile_joint_pspl(
            datasets,
            t0=float(opt[1]),
            u0=float(np.exp(opt[0])),
            tE=float(np.exp(opt[2])),
        )
        return residuals

    best_result = None
    best_chi2 = np.inf
    last_message = "least_squares_failed"
    for u0_start, t0_start, tE_start in starts:
        x0 = np.array([np.log(u0_start), t0_start, np.log(tE_start)], dtype=float)
        try:
            result = least_squares(
                residual_function,
                np.clip(x0, lower + 1e-9, upper - 1e-9),
                bounds=(lower, upper),
                loss="linear",
                max_nfev=3000,
            )
        except Exception as exc:
            last_message = str(exc)
            continue
        last_message = str(result.message)
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        chi2 = float(np.sum(np.square(result.fun)))
        if chi2 < best_chi2:
            best_result = result
            best_chi2 = chi2

    if best_result is None:
        return JointFitResult(False, "fit_failed", optimizer_message=last_message)

    t0 = float(best_result.x[1])
    u0 = float(np.exp(best_result.x[0]))
    tE = float(np.exp(best_result.x[2]))
    magnifications = {
        dataset.dataset_id: pspl_magnification(dataset.time_jd, t0=t0, u0=u0, tE=tE)
        for dataset in datasets
    }
    solutions, chi2, _ = _profile_magnifications(datasets, magnifications)
    if not np.isfinite(chi2):
        return JointFitResult(False, "flux_profile_failed", optimizer_message=str(best_result.message))

    n_points = sum(dataset.n_points for dataset in datasets)
    n_parameters = 3 + 2 * len(datasets)
    dof = max(n_points - n_parameters, 1)
    bic = float(chi2 + n_parameters * np.log(max(n_points, 2)))
    flat_chi2, flat_bic = _flat_statistics(datasets)
    return JointFitResult(
        True,
        "ok",
        t0_jd=t0,
        u0=u0,
        tE_days=tE,
        chi2=chi2,
        reduced_chi2=float(chi2 / dof),
        bic=bic,
        flat_chi2=flat_chi2,
        flat_bic=flat_bic,
        delta_bic_flat=float(flat_bic - bic),
        n_points=n_points,
        n_datasets=len(datasets),
        n_parameters=n_parameters,
        dataset_solutions=solutions,
        magnifications=magnifications,
        optimizer_message=str(best_result.message),
    )


def fit_asassn_only_pspl(datasets: Iterable[PhotometryDataset]) -> JointFitResult:
    """Fit the shared geometry using only ASAS-SN datasets."""
    return _fit_pspl([dataset for dataset in datasets if dataset.survey == "asassn"])


def fit_joint_pspl(
    datasets: Iterable[PhotometryDataset],
    *,
    seed: tuple[float, float, float] | None = None,
) -> JointFitResult:
    """Fit one PSPL geometry shared by every supplied survey dataset."""
    dataset_list = list(datasets)
    if seed is None and any(dataset.survey == "asassn" for dataset in dataset_list):
        asassn_fit = fit_asassn_only_pspl(dataset_list)
        if asassn_fit.success:
            seed = (asassn_fit.t0_jd, asassn_fit.u0, asassn_fit.tE_days)
    return _fit_pspl(dataset_list, seed=seed)


def fit_individual_dataset_pspl(
    datasets: Iterable[PhotometryDataset],
    *,
    joint_seed: JointFitResult | None = None,
) -> dict[str, JointFitResult]:
    """Return raw independent geometry fits for cross-survey inspection."""
    seed = None
    if joint_seed is not None and joint_seed.success:
        seed = (joint_seed.t0_jd, joint_seed.u0, joint_seed.tE_days)
    return {dataset.dataset_id: _fit_pspl([dataset], seed=seed) for dataset in datasets}


def fit_leave_one_survey_out(
    datasets: Iterable[PhotometryDataset],
    *,
    joint_seed: JointFitResult | None = None,
) -> dict[str, JointFitResult]:
    """Refit after removing each survey as a direct robustness diagnostic."""
    dataset_list = list(datasets)
    seed = None
    if joint_seed is not None and joint_seed.success:
        seed = (joint_seed.t0_jd, joint_seed.u0, joint_seed.tE_days)
    output: dict[str, JointFitResult] = {}
    for survey in sorted({dataset.survey for dataset in dataset_list}):
        retained = [dataset for dataset in dataset_list if dataset.survey != survey]
        output[survey] = _fit_pspl(retained, seed=seed)
    return output


__all__ = [
    "JointFitResult",
    "fit_asassn_only_pspl",
    "fit_individual_dataset_pspl",
    "fit_joint_pspl",
    "fit_leave_one_survey_out",
    "profile_joint_pspl",
]
