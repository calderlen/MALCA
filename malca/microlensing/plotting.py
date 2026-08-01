"""Compact scientific plots for shared-geometry microlensing fits."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from .datasets import PhotometryDataset
from .joint_fit import JointFitResult
from .pspl import pspl_magnification


def _normalized_flux_space(dataset: PhotometryDataset, fit: JointFitResult):
    solution = fit.dataset_solutions[dataset.dataset_id]
    if dataset.flux_kind == "direct":
        scale = solution.source_flux + solution.offset_flux
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        values = dataset.flux / scale

        def model_from_magnification(magnification: np.ndarray) -> np.ndarray:
            return (solution.source_flux * magnification + solution.offset_flux) / scale
    else:
        if not np.isfinite(solution.source_flux) or solution.source_flux <= 0.0:
            return None
        values = (dataset.flux - solution.offset_flux) / solution.source_flux + 1.0
        scale = solution.source_flux

        def model_from_magnification(magnification: np.ndarray) -> np.ndarray:
            return magnification
    errors = dataset.flux_error / scale
    return values, errors, solution.residuals, model_from_magnification


def plot_joint_fit(
    candidate_id: str,
    datasets: list[PhotometryDataset],
    fit: JointFitResult,
    output_dir: Path | str,
) -> Path:
    """Plot the full baseline, event-scale overlay, and standardized residuals."""
    if not fit.success:
        raise ValueError("Cannot plot an unsuccessful joint fit")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate_id))
    output_path = output_dir / f"{safe_id}_joint_pspl.pdf"

    figure, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    all_tau: list[np.ndarray] = []
    all_residuals: list[np.ndarray] = []
    for index, dataset in enumerate(datasets):
        normalized = _normalized_flux_space(dataset, fit)
        if normalized is None:
            continue
        values, errors, residuals, model_from_magnification = normalized
        color = colors(index % 10)
        label = f"{dataset.survey} {dataset.band} {dataset.instrument}"
        axes[0].errorbar(dataset.time_jd, values, yerr=errors, fmt=".", ms=3, alpha=0.65, color=color, label=label)
        tau = (dataset.time_jd - fit.t0_jd) / fit.tE_days
        axes[1].errorbar(tau, values, yerr=errors, fmt=".", ms=3, alpha=0.7, color=color, label=label)
        axes[2].scatter(tau, residuals, s=9, alpha=0.7, color=color, label=label)
        full_grid = np.linspace(np.nanmin(dataset.time_jd), np.nanmax(dataset.time_jd), 700)
        full_A = pspl_magnification(full_grid, fit.t0_jd, fit.u0, fit.tE_days)
        axes[0].plot(full_grid, model_from_magnification(full_A), color=color, lw=1.0)
        tau_grid = np.linspace(-5.0, 5.0, 1000)
        event_A = pspl_magnification(fit.t0_jd + tau_grid * fit.tE_days, fit.t0_jd, fit.u0, fit.tE_days)
        axes[1].plot(tau_grid, model_from_magnification(event_A), color=color, lw=1.0)
        all_tau.append(tau)
        all_residuals.append(residuals[np.isfinite(residuals)])

    tau_grid = np.linspace(-5.0, 5.0, 1400)
    time_grid = fit.t0_jd + tau_grid * fit.tE_days
    axes[1].plot(tau_grid, pspl_magnification(time_grid, fit.t0_jd, fit.u0, fit.tE_days), color="black", lw=1.2, ls="--", label="intrinsic A(t)")
    axes[2].axhline(0.0, color="black", lw=1.0)
    axes[2].axhline(+3.0, color="0.7", lw=0.8, ls="--")
    axes[2].axhline(-3.0, color="0.7", lw=0.8, ls="--")
    observed_tau = np.concatenate(all_tau) if all_tau else np.array([5.0])
    tau_limit = float(np.clip(1.1 * np.nanpercentile(np.abs(observed_tau), 99.0), 1.2, 5.0))
    axes[1].set_xlim(-tau_limit, tau_limit)
    axes[2].set_xlim(-tau_limit, tau_limit)
    residual_values = np.concatenate(all_residuals) if all_residuals else np.array([0.0])
    residual_limit = float(np.clip(1.2 * np.nanpercentile(np.abs(residual_values), 95.0), 5.0, 25.0))
    axes[2].set_ylim(-residual_limit, residual_limit)
    n_clipped = int(np.sum(np.abs(residual_values) > residual_limit))
    if n_clipped:
        axes[2].text(0.99, 0.04, f"{n_clipped} residuals outside limits", transform=axes[2].transAxes, ha="right", va="bottom", fontsize=8)
    axes[0].set_ylabel("flux / fitted baseline")
    axes[1].set_ylabel("normalized flux")
    axes[2].set_ylabel("residual / uncertainty")
    axes[0].set_xlabel("JD")
    axes[1].set_xlabel(r"$(t-t_0)/t_E$")
    axes[2].set_xlabel(r"$(t-t_0)/t_E$")
    axes[0].set_title(
        f"{candidate_id}: shared PSPL  t0={fit.t0_jd:.2f}, u0={fit.u0:.3g}, tE={fit.tE_days:.2f} d"
    )
    axes[0].legend(fontsize=7, ncol=2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


__all__ = ["plot_joint_fit"]
