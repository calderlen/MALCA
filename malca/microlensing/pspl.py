"""Canonical PSPL physics and profiled flux nuisance parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class LinearFluxSolution:
    """Best linear flux terms for one fixed magnification curve."""

    success: bool
    source_flux: float
    offset_flux: float
    model_flux: np.ndarray
    residuals: np.ndarray
    chi2: float
    flux_kind: str

    @property
    def blend_flux(self) -> float:
        return self.offset_flux if self.flux_kind == "direct" else np.nan

    @property
    def reference_difference_flux(self) -> float:
        return self.offset_flux if self.flux_kind == "difference" else np.nan


def pspl_magnification(
    time: np.ndarray,
    t0: float,
    u0: float,
    tE: float,
) -> np.ndarray:
    """Point-source point-lens magnification at ``time``."""
    time = np.asarray(time, dtype=float)
    tE = max(abs(float(tE)), 1e-12)
    u = np.sqrt(float(u0) ** 2 + np.square((time - float(t0)) / tE))
    u = np.maximum(u, 1e-12)
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


def magnification_from_tau_beta(tau: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """PSPL magnification from trajectory coordinates, used by parallax fits."""
    tau = np.asarray(tau, dtype=float)
    beta = np.asarray(beta, dtype=float)
    u = np.sqrt(np.maximum(np.square(tau) + np.square(beta), 1e-24))
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


def magnitude_to_relative_flux(
    mag: np.ndarray,
    mag_error: np.ndarray,
    reference_mag: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert magnitudes and their uncertainties to relative direct flux."""
    mag = np.asarray(mag, dtype=float)
    mag_error = np.asarray(mag_error, dtype=float)
    if reference_mag is None or not np.isfinite(reference_mag):
        reference_mag = float(np.nanmedian(mag))
    flux = np.power(10.0, -0.4 * (mag - float(reference_mag)))
    flux_error = (np.log(10.0) / 2.5) * flux * np.clip(mag_error, 1e-4, None)
    return flux, np.clip(flux_error, 1e-8, None), float(reference_mag)


def _failed_solution(size: int, flux_kind: str) -> LinearFluxSolution:
    values = np.full(int(size), np.nan, dtype=float)
    return LinearFluxSolution(False, np.nan, np.nan, values.copy(), values, np.nan, flux_kind)


def solve_linear_flux_parameters(
    magnification: np.ndarray,
    flux: np.ndarray,
    flux_error: np.ndarray,
    *,
    flux_kind: str = "direct",
) -> LinearFluxSolution:
    """Profile the two linear flux terms for a fixed PSPL geometry.

    Direct photometry uses ``F = Fs*A + Fb`` with ``Fs,Fb >= 0``.
    Difference photometry uses ``dF = Fs*(A-1) + F0`` with ``Fs >= 0`` and
    an unconstrained reference offset ``F0``.
    """
    magnification = np.asarray(magnification, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_error = np.asarray(flux_error, dtype=float)
    if not (magnification.shape == flux.shape == flux_error.shape):
        raise ValueError("magnification, flux, and flux_error must have matching shapes")
    if flux_kind not in {"direct", "difference"}:
        raise ValueError("flux_kind must be 'direct' or 'difference'")

    valid = (
        np.isfinite(magnification)
        & np.isfinite(flux)
        & np.isfinite(flux_error)
        & (flux_error > 0.0)
    )
    if int(valid.sum()) < 2:
        return _failed_solution(len(flux), flux_kind)

    profile = magnification[valid] if flux_kind == "direct" else magnification[valid] - 1.0
    design = np.column_stack((profile, np.ones_like(profile)))
    inv_sigma = 1.0 / flux_error[valid]
    design_weighted = design * inv_sigma[:, None]
    flux_weighted = flux[valid] * inv_sigma
    lower = np.array([0.0, 0.0 if flux_kind == "direct" else -np.inf])
    upper = np.array([np.inf, np.inf])

    try:
        result = lsq_linear(
            design_weighted,
            flux_weighted,
            bounds=(lower, upper),
            method="trf",
            lsmr_tol="auto",
        )
    except Exception:
        return _failed_solution(len(flux), flux_kind)
    if not result.success or not np.all(np.isfinite(result.x)):
        return _failed_solution(len(flux), flux_kind)

    source_flux, offset_flux = (float(result.x[0]), float(result.x[1]))
    full_profile = magnification if flux_kind == "direct" else magnification - 1.0
    model = source_flux * full_profile + offset_flux
    residuals = np.full_like(flux, np.nan, dtype=float)
    residuals[valid] = (flux[valid] - model[valid]) / flux_error[valid]
    chi2 = float(np.sum(np.square(residuals[valid])))
    return LinearFluxSolution(
        True,
        source_flux,
        offset_flux,
        np.asarray(model, dtype=float),
        residuals,
        chi2,
        flux_kind,
    )


def evaluate_pspl_dataset(
    time: np.ndarray,
    flux: np.ndarray,
    flux_error: np.ndarray,
    *,
    t0: float,
    u0: float,
    tE: float,
    flux_kind: str = "direct",
) -> LinearFluxSolution:
    """Evaluate and profile one dataset at a fixed PSPL geometry."""
    magnification = pspl_magnification(time, t0=t0, u0=u0, tE=tE)
    return solve_linear_flux_parameters(
        magnification,
        flux,
        flux_error,
        flux_kind=flux_kind,
    )
