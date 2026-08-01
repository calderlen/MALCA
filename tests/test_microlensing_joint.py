from __future__ import annotations

import numpy as np

from malca.microlensing.datasets import PhotometryDataset
from malca.microlensing.joint_fit import fit_joint_pspl
from malca.microlensing.pspl import pspl_magnification, solve_linear_flux_parameters


def test_pspl_magnification_is_symmetric_and_has_known_u1_value():
    time = np.array([-1.0, 0.0, 1.0])
    magnification = pspl_magnification(time, t0=0.0, u0=1.0, tE=1.0)
    assert np.isclose(magnification[0], magnification[2])
    assert np.isclose(magnification[1], 3.0 / np.sqrt(5.0))


def test_direct_flux_solver_recovers_source_and_blend_flux():
    magnification = np.linspace(1.0, 4.0, 80)
    flux_error = np.full(80, 0.02)
    flux = 1.7 * magnification + 0.35
    solution = solve_linear_flux_parameters(magnification, flux, flux_error, flux_kind="direct")
    assert solution.success
    assert np.isclose(solution.source_flux, 1.7, atol=1e-8)
    assert np.isclose(solution.blend_flux, 0.35, atol=1e-8)


def test_difference_flux_solver_recovers_source_and_signed_offset():
    magnification = np.linspace(1.0, 4.0, 80)
    flux_error = np.full(80, 2.0)
    difference_flux = 125.0 * (magnification - 1.0) - 17.0
    solution = solve_linear_flux_parameters(
        magnification,
        difference_flux,
        flux_error,
        flux_kind="difference",
    )
    assert solution.success
    assert np.isclose(solution.source_flux, 125.0, atol=1e-8)
    assert np.isclose(solution.reference_difference_flux, -17.0, atol=1e-8)


def test_joint_fit_recovers_one_geometry_across_direct_and_difference_datasets():
    t0 = 2_459_100.0
    u0 = 0.18
    tE = 32.0
    time_a = t0 + np.linspace(-110.0, 105.0, 150)
    time_b = t0 + np.linspace(-90.0, 125.0, 125)
    time_c = t0 + np.linspace(-130.0, 115.0, 140)
    magnification_a = pspl_magnification(time_a, t0=t0, u0=u0, tE=tE)
    magnification_b = pspl_magnification(time_b, t0=t0, u0=u0, tE=tE)
    magnification_c = pspl_magnification(time_c, t0=t0, u0=u0, tE=tE)
    datasets = [
        PhotometryDataset(
            "asassn:g:bi", "asassn", "g", "bi", time_a,
            1.2 * magnification_a + 0.45, np.full(time_a.size, 0.01), "direct", "synthetic-a",
        ),
        PhotometryDataset(
            "atlas:o:forced", "atlas", "o", "forced", time_b,
            0.75 * magnification_b + 1.1, np.full(time_b.size, 0.015), "direct", "synthetic-b",
        ),
        PhotometryDataset(
            "ztf_forced:zr:forced", "ztf_forced", "zr", "forced", time_c,
            240.0 * (magnification_c - 1.0) - 11.0, np.full(time_c.size, 1.5), "difference", "synthetic-c",
        ),
    ]
    fit = fit_joint_pspl(datasets, seed=(t0 + 2.0, 0.25, 28.0))
    assert fit.success
    assert np.isclose(fit.t0_jd, t0, atol=0.05)
    assert np.isclose(fit.u0, u0, atol=0.01)
    assert np.isclose(fit.tE_days, tE, atol=0.1)
