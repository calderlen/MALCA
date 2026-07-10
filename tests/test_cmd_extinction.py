"""Tests for dustmaps3d CMD dereddening helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.config import CMD_A_G_PER_AV, CMD_E_BP_RP_PER_AV
from malca.ltv.cmd import (
    cmd_uncertainty_from_fields,
    compute_cmd_features,
    dustmaps_cmd_from_fields,
    estimate_cmd_masses,
    mist_mass_tracks,
    normalize_mist_cmd_grid,
)


def test_dustmaps_cmd_from_fields_applies_extinction() -> None:
    coords = dustmaps_cmd_from_fields(
        g_mag=12.0,
        bp_rp=1.0,
        dist_pc=100.0,
        a_v_3d=0.5,
    )
    mg = 12.0 - 5.0 * np.log10(100.0) + 5.0
    assert coords["cmd_coordinate_source"] == "dustmaps3d"
    assert coords["mg"] == pytest.approx(mg)
    assert coords["mg0"] == pytest.approx(mg - CMD_A_G_PER_AV * 0.5)
    assert coords["bprp0"] == pytest.approx(1.0 - CMD_E_BP_RP_PER_AV * 0.5)


def test_dustmaps_cmd_zero_extinction_is_observed_no_extinction() -> None:
    coords = dustmaps_cmd_from_fields(
        g_mag=11.0,
        bp_rp=0.8,
        dist_pc=200.0,
        a_v_3d=0.0,
    )
    mg = 11.0 - 5.0 * np.log10(200.0) + 5.0
    assert coords["cmd_coordinate_source"] == "observed_no_extinction"
    assert coords["mg0"] == pytest.approx(mg)
    assert coords["bprp0"] == pytest.approx(0.8)


def test_dustmaps_cmd_missing_av_uses_observed_fallback() -> None:
    coords = dustmaps_cmd_from_fields(
        g_mag=11.0,
        bp_rp=0.8,
        dist_pc=200.0,
        a_v_3d=None,
    )
    mg = 11.0 - 5.0 * np.log10(200.0) + 5.0
    assert coords["cmd_coordinate_source"] == "observed_fallback"
    assert coords["cmd_mag"] == pytest.approx(mg)
    assert coords["cmd_color"] == pytest.approx(0.8)
    assert np.isnan(coords["mg0"])


def test_dustmaps_cmd_uses_precomputed_bp_rp_without_bp_rp_mags() -> None:
    coords = dustmaps_cmd_from_fields(
        g_mag=10.0,
        bp_rp=0.6,
        dist_pc=150.0,
        a_v_3d=0.2,
    )
    assert coords["cmd_coordinate_source"] == "dustmaps3d"
    assert coords["bp_rp"] == pytest.approx(0.6)


def test_dustmaps_cmd_ignores_stored_starhorse_mg0_bprp0() -> None:
    """Helper only uses passed fields; stored SH values are not inputs."""
    coords = dustmaps_cmd_from_fields(
        g_mag=10.0,
        bp_rp=1.0,
        dist_pc=100.0,
        a_v_3d=1.0,
    )
    sh_mg0 = 99.0
    sh_bprp0 = -9.0
    assert coords["mg0"] != sh_mg0
    assert coords["bprp0"] != sh_bprp0


def test_compute_cmd_features_with_bp_rp_only_and_av() -> None:
    df = pd.DataFrame(
        {
            "phot_g_mean_mag": [12.0],
            "bp_rp": [1.2],
            "distance_gspphot": [250.0],
            "A_v_3d": [0.4],
        }
    )
    out = compute_cmd_features(df)
    mg = 12.0 - 5.0 * np.log10(250.0) + 5.0
    assert out.loc[0, "bp_rp"] == pytest.approx(1.2)
    assert out.loc[0, "mg"] == pytest.approx(mg)
    assert out.loc[0, "mg0"] == pytest.approx(mg - CMD_A_G_PER_AV * 0.4)
    assert out.loc[0, "bprp0"] == pytest.approx(1.2 - CMD_E_BP_RP_PER_AV * 0.4)


def test_cmd_uncertainty_from_fields_propagates_photometry_parallax_and_extinction() -> None:
    coords = cmd_uncertainty_from_fields(
        g_mag_err=0.01,
        bp_mag_err=0.02,
        rp_mag_err=0.03,
        parallax_mas=10.0,
        parallax_err_mas=0.5,
        a_v_3d=0.2,
        a_v_3d_err=0.1,
    )
    color_err = np.sqrt(0.02**2 + 0.03**2)
    dist_mod_err = 5.0 / np.log(10.0) * 0.5 / 10.0
    mg_err = np.sqrt(0.01**2 + dist_mod_err**2)
    assert coords["bp_rp_err"] == pytest.approx(color_err)
    assert coords["mg_err"] == pytest.approx(mg_err)
    assert coords["cmd_color_err"] == pytest.approx(
        np.sqrt(color_err**2 + (CMD_E_BP_RP_PER_AV * 0.1) ** 2)
    )
    assert coords["cmd_mag_err"] == pytest.approx(
        np.sqrt(mg_err**2 + (CMD_A_G_PER_AV * 0.1) ** 2)
    )
    assert coords["cmd_distance_error_source"] == "parallax_error"
    assert coords["cmd_extinction_error_source"] == "av_error"


def test_normalize_mist_cmd_grid_accepts_mist_gaia_columns() -> None:
    grid = pd.DataFrame(
        {
            "log10_isochrone_age_yr": [6.0],
            "initial_mass": [0.5],
            "star_mass": [0.49],
            "Gaia_G_EDR3": [6.1],
            "Gaia_BP_EDR3": [7.3],
            "Gaia_RP_EDR3": [5.8],
        }
    )
    out = normalize_mist_cmd_grid(grid)
    assert out.loc[0, "mist_age_myr"] == pytest.approx(1.0)
    assert out.loc[0, "mist_star_mass"] == pytest.approx(0.49)
    assert out.loc[0, "mist_gaia_bp_rp"] == pytest.approx(1.5)


def test_estimate_cmd_masses_uses_nearest_isochrone_point() -> None:
    grid = pd.DataFrame(
        {
            "age_myr": [1.0, 1.0, 3.0, 3.0],
            "initial_mass": [0.5, 1.0, 0.5, 1.0],
            "star_mass": [0.48, 0.98, 0.47, 0.97],
            "gaia_g": [5.0, 3.0, 5.4, 3.4],
            "gaia_bp": [6.0, 3.8, 6.3, 4.1],
            "gaia_rp": [4.9, 3.2, 5.0, 3.4],
        }
    )
    df = pd.DataFrame(
        {
            "cmd_color": [0.61],
            "cmd_mag": [3.02],
            "cmd_color_err": [0.04],
            "cmd_mag_err": [0.05],
        }
    )
    out = estimate_cmd_masses(df, grid, ages_myr=[1.0, 3.0])
    assert out.loc[0, "cmd_mass_1myr"] == pytest.approx(0.98)
    assert out.loc[0, "cmd_mass_best"] == pytest.approx(0.98)
    assert out.loc[0, "cmd_mass_best_age_myr"] == pytest.approx(1.0)
    assert out.loc[0, "cmd_mass_source"] == "mist_nearest_isochrone"


def test_mist_mass_tracks_interpolates_between_grid_masses() -> None:
    grid = pd.DataFrame(
        {
            "age_myr": [1.0, 1.0, 3.0, 3.0],
            "initial_mass": [0.5, 1.0, 0.5, 1.0],
            "star_mass": [0.5, 1.0, 0.5, 1.0],
            "gaia_g": [5.0, 3.0, 5.4, 3.4],
            "gaia_bp": [6.0, 3.8, 6.3, 4.1],
            "gaia_rp": [4.8, 3.1, 5.0, 3.4],
        }
    )
    tracks = mist_mass_tracks(grid, [0.75], ages_myr=[1.0, 3.0])
    assert len(tracks) == 2
    assert set(tracks["age_myr"]) == {1.0, 3.0}
    assert tracks.loc[tracks["age_myr"].eq(1.0), "gaia_g"].iloc[0] == pytest.approx(4.0)
