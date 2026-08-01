"""Shared annual-parallax PSPL fits for multiple photometric datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from scipy.optimize import least_squares

from .datasets import PhotometryDataset
from .joint_fit import JointFitResult
from .pspl import LinearFluxSolution, magnification_from_tau_beta, solve_linear_flux_parameters

PARALLAX_MIN_TE_DAYS = 80.0
PARALLAX_MIN_POINTS = 80
PARALLAX_MIN_SPAN_DAYS = 240.0
PARALLAX_MAX_ABS_PIE = 1.5
PARALLAX_REQUIRED_DELTA_BIC = 6.0


@dataclass
class ParallaxBranchResult:
    success: bool
    status: str
    branch_sign: int
    t0_jd: float = np.nan
    u0: float = np.nan
    tE_days: float = np.nan
    piE_N: float = np.nan
    piE_E: float = np.nan
    chi2: float = np.nan
    reduced_chi2: float = np.nan
    bic: float = np.nan
    dataset_solutions: dict[str, LinearFluxSolution] = field(default_factory=dict)
    magnifications: dict[str, np.ndarray] = field(default_factory=dict)
    opt_params: np.ndarray = field(default_factory=lambda: np.empty(0))
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    mcmc_summary: dict[str, float] = field(default_factory=dict)

    @property
    def piE(self) -> float:
        return float(np.hypot(self.piE_N, self.piE_E))


@dataclass
class ParallaxResult:
    attempted: bool
    fit_ok: bool
    preferred: bool
    status: str
    best_branch: str = ""
    delta_bic: float = np.nan
    branch_delta_bic: float = np.nan
    t0_ref_jd: float = np.nan
    branches: dict[str, ParallaxBranchResult] = field(default_factory=dict)

    @property
    def best(self) -> ParallaxBranchResult | None:
        return self.branches.get(self.best_branch)


def _sky_basis(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    east = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=float)
    north = np.array([-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)], dtype=float)
    return east, north


def project_earth_orbit_geocentric(
    time_jd: np.ndarray,
    *,
    ra_deg: float,
    dec_deg: float,
    t0_ref_jd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Earth displacement on the sky after removing position and velocity at t0."""
    time_jd = np.asarray(time_jd, dtype=float)
    east, north = _sky_basis(ra_deg, dec_deg)
    times = Time(time_jd, format="jd", scale="utc").tdb
    reference = Time(float(t0_ref_jd), format="jd", scale="utc").tdb
    position, _ = get_body_barycentric_posvel("earth", times)
    position_ref, velocity_ref = get_body_barycentric_posvel("earth", reference)
    xyz = position.xyz.to_value(u.au).T
    xyz_ref = position_ref.xyz.to_value(u.au)
    velocity = velocity_ref.xyz.to_value(u.au / u.day)
    delta = xyz - xyz_ref[None, :] - (times.jd - reference.jd)[:, None] * velocity[None, :]
    return np.asarray(delta @ north, dtype=float), np.asarray(delta @ east, dtype=float)


def _profile_branch(
    opt: np.ndarray,
    *,
    branch_sign: int,
    datasets: list[PhotometryDataset],
    ephemerides: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, LinearFluxSolution], dict[str, np.ndarray], float, np.ndarray]:
    u0 = float(branch_sign * np.exp(opt[0]))
    t0 = float(opt[1])
    tE = float(np.exp(opt[2]))
    piE_N = float(opt[3])
    piE_E = float(opt[4])
    solutions: dict[str, LinearFluxSolution] = {}
    magnifications: dict[str, np.ndarray] = {}
    residuals: list[np.ndarray] = []
    chi2 = 0.0
    for dataset in datasets:
        delta_n, delta_e = ephemerides[dataset.dataset_id]
        tau = (dataset.time_jd - t0) / tE + piE_N * delta_n + piE_E * delta_e
        beta = u0 + piE_N * delta_e - piE_E * delta_n
        magnification = magnification_from_tau_beta(tau, beta)
        solution = solve_linear_flux_parameters(
            magnification,
            dataset.flux,
            dataset.flux_error,
            flux_kind=dataset.flux_kind,
        )
        solutions[dataset.dataset_id] = solution
        magnifications[dataset.dataset_id] = magnification
        if not solution.success or not np.all(np.isfinite(solution.residuals)):
            size = sum(item.n_points for item in datasets)
            return solutions, magnifications, np.nan, np.full(size, 1e6)
        residuals.append(solution.residuals)
        chi2 += solution.chi2
    return solutions, magnifications, float(chi2), np.concatenate(residuals)


def _fit_branch(
    *,
    branch_sign: int,
    datasets: list[PhotometryDataset],
    base_fit: JointFitResult,
    ephemerides: dict[str, tuple[np.ndarray, np.ndarray]],
) -> ParallaxBranchResult:
    all_time = np.concatenate([dataset.time_jd for dataset in datasets])
    span = max(float(np.nanmax(all_time) - np.nanmin(all_time)), 30.0)
    u0_abs = max(abs(base_fit.u0), 1e-3)
    u0_lo = max(1e-3, u0_abs / 3.0)
    u0_hi = min(2.0, max(0.05, 3.0 * u0_abs))
    if u0_hi <= u0_lo:
        u0_lo, u0_hi = 1e-3, 2.0
    tE_lo = max(5.0, 0.35 * base_fit.tE_days)
    tE_hi = min(max(25.0, 4.0 * span), 5000.0, max(30.0, 3.0 * base_fit.tE_days))
    if tE_hi <= tE_lo:
        tE_hi = min(5000.0, max(tE_lo + 5.0, 1.5 * tE_lo))
    lower = np.array([np.log(u0_lo), np.nanmin(all_time), np.log(tE_lo), -PARALLAX_MAX_ABS_PIE, -PARALLAX_MAX_ABS_PIE])
    upper = np.array([np.log(u0_hi), np.nanmax(all_time), np.log(tE_hi), PARALLAX_MAX_ABS_PIE, PARALLAX_MAX_ABS_PIE])

    starts = [
        (0.0, 0.0),
        (0.10, 0.0),
        (-0.10, 0.0),
        (0.0, 0.10),
        (0.0, -0.10),
        (0.20, 0.20),
    ]

    def residual_function(opt: np.ndarray) -> np.ndarray:
        return _profile_branch(
            opt,
            branch_sign=branch_sign,
            datasets=datasets,
            ephemerides=ephemerides,
        )[3]

    best = None
    best_chi2 = np.inf
    message = "least_squares_failed"
    for piE_N, piE_E in starts:
        x0 = np.array([np.log(u0_abs), base_fit.t0_jd, np.log(base_fit.tE_days), piE_N, piE_E])
        try:
            result = least_squares(
                residual_function,
                np.clip(x0, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                loss="linear",
                max_nfev=5000,
            )
        except Exception as exc:
            message = str(exc)
            continue
        message = str(result.message)
        if result.success and np.all(np.isfinite(result.x)):
            chi2 = float(np.sum(np.square(result.fun)))
            if chi2 < best_chi2:
                best = result
                best_chi2 = chi2
    if best is None:
        return ParallaxBranchResult(False, message, branch_sign)

    solutions, magnifications, chi2, _ = _profile_branch(
        best.x,
        branch_sign=branch_sign,
        datasets=datasets,
        ephemerides=ephemerides,
    )
    if not np.isfinite(chi2):
        return ParallaxBranchResult(False, "flux_profile_failed", branch_sign)
    n_points = sum(dataset.n_points for dataset in datasets)
    n_parameters = 5 + 2 * len(datasets)
    dof = max(n_points - n_parameters, 1)
    return ParallaxBranchResult(
        True,
        "ok",
        branch_sign,
        t0_jd=float(best.x[1]),
        u0=float(branch_sign * np.exp(best.x[0])),
        tE_days=float(np.exp(best.x[2])),
        piE_N=float(best.x[3]),
        piE_E=float(best.x[4]),
        chi2=chi2,
        reduced_chi2=float(chi2 / dof),
        bic=float(chi2 + n_parameters * np.log(max(n_points, 2))),
        dataset_solutions=solutions,
        magnifications=magnifications,
        opt_params=np.asarray(best.x, dtype=float),
        bounds=(lower, upper),
    )


def _run_mcmc(
    branch: ParallaxBranchResult,
    *,
    datasets: list[PhotometryDataset],
    ephemerides: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, float]:
    if not branch.success or branch.bounds is None:
        return {}
    lower, upper = branch.bounds
    rng = np.random.default_rng(seed)
    scales = np.array([0.03, max(0.6, 0.01 * branch.tE_days), 0.03, 0.015, 0.015])
    samples: list[np.ndarray] = []
    accepted = 0
    proposed = 0

    def log_probability(opt: np.ndarray) -> float:
        if np.any(opt <= lower) or np.any(opt >= upper):
            return -np.inf
        chi2 = _profile_branch(
            opt,
            branch_sign=branch.branch_sign,
            datasets=datasets,
            ephemerides=ephemerides,
        )[2]
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    for _ in range(6):
        current = np.clip(branch.opt_params + rng.normal(scale=0.2 * scales), lower + 1e-8, upper - 1e-8)
        current_logp = log_probability(current)
        if not np.isfinite(current_logp):
            current = branch.opt_params.copy()
            current_logp = log_probability(current)
        for step in range(600):
            proposal = np.clip(current + rng.normal(scale=scales), lower + 1e-8, upper - 1e-8)
            proposal_logp = log_probability(proposal)
            proposed += 1
            if np.isfinite(proposal_logp) and (
                proposal_logp >= current_logp or np.log(rng.random()) < proposal_logp - current_logp
            ):
                current, current_logp = proposal, proposal_logp
                accepted += 1
            if step >= 200 and (step - 200) % 2 == 0:
                samples.append(current.copy())
    if not samples:
        return {}
    values = np.asarray(samples)
    physical = {
        "u0": branch.branch_sign * np.exp(values[:, 0]),
        "t0_jd": values[:, 1],
        "tE_days": np.exp(values[:, 2]),
        "piE_N": values[:, 3],
        "piE_E": values[:, 4],
        "piE": np.hypot(values[:, 3], values[:, 4]),
    }
    summary: dict[str, float] = {
        "n_samples": float(len(values)),
        "acceptance_rate": float(accepted / max(proposed, 1)),
    }
    for name, array in physical.items():
        q16, q50, q84 = np.nanpercentile(array, [16.0, 50.0, 84.0])
        summary[f"{name}_p16"] = float(q16)
        summary[f"{name}_p50"] = float(q50)
        summary[f"{name}_p84"] = float(q84)
    return summary


def fit_joint_parallax(
    datasets: list[PhotometryDataset],
    base_fit: JointFitResult,
    *,
    ra_deg: float | None,
    dec_deg: float | None,
    run_mcmc: bool = False,
    random_seed: int = 20260322,
) -> ParallaxResult:
    """Fit both signed-u0 annual-parallax branches after the base joint fit."""
    if not base_fit.success:
        return ParallaxResult(False, False, False, "not_attempted:no_base_fit")
    if ra_deg is None or dec_deg is None or not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
        return ParallaxResult(False, False, False, "not_attempted:missing_coordinates")
    n_points = sum(dataset.n_points for dataset in datasets)
    all_time = np.concatenate([dataset.time_jd for dataset in datasets])
    span = float(np.nanmax(all_time) - np.nanmin(all_time))
    if n_points < PARALLAX_MIN_POINTS:
        return ParallaxResult(False, False, False, "not_attempted:too_few_points")
    if span < PARALLAX_MIN_SPAN_DAYS:
        return ParallaxResult(False, False, False, "not_attempted:fit_span_too_short")
    if base_fit.tE_days < PARALLAX_MIN_TE_DAYS:
        return ParallaxResult(False, False, False, "not_attempted:tE_below_threshold")

    ephemerides = {
        dataset.dataset_id: project_earth_orbit_geocentric(
            dataset.time_jd,
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            t0_ref_jd=base_fit.t0_jd,
        )
        for dataset in datasets
    }
    branches = {
        "u0_pos": _fit_branch(
            branch_sign=+1,
            datasets=datasets,
            base_fit=base_fit,
            ephemerides=ephemerides,
        ),
        "u0_neg": _fit_branch(
            branch_sign=-1,
            datasets=datasets,
            base_fit=base_fit,
            ephemerides=ephemerides,
        ),
    }
    successful = [(name, branch) for name, branch in branches.items() if branch.success]
    if not successful:
        return ParallaxResult(True, False, False, "fit_failed", t0_ref_jd=base_fit.t0_jd, branches=branches)
    successful.sort(key=lambda item: item[1].bic)
    best_name, best = successful[0]
    if run_mcmc:
        for index, (_, branch) in enumerate(successful):
            branch.mcmc_summary = _run_mcmc(
                branch,
                datasets=datasets,
                ephemerides=ephemerides,
                seed=random_seed + index,
            )
    delta_bic = float(base_fit.bic - best.bic)
    branch_delta = float(successful[1][1].bic - best.bic) if len(successful) > 1 else np.nan
    near_pi_bound = max(abs(best.piE_N), abs(best.piE_E)) >= 0.98 * PARALLAX_MAX_ABS_PIE
    preferred = bool(
        delta_bic >= PARALLAX_REQUIRED_DELTA_BIC
        and best.reduced_chi2 <= 10.0
        and not near_pi_bound
    )
    return ParallaxResult(
        True,
        True,
        preferred,
        "ok",
        best_branch=best_name,
        delta_bic=delta_bic,
        branch_delta_bic=branch_delta,
        t0_ref_jd=base_fit.t0_jd,
        branches=branches,
    )


__all__ = [
    "ParallaxBranchResult",
    "ParallaxResult",
    "fit_joint_parallax",
    "project_earth_orbit_geocentric",
]
