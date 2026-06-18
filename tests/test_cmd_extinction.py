"""Tests for dustmaps3d CMD dereddening helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.config import CMD_A_G_PER_AV, CMD_E_BP_RP_PER_AV
from malca.ltv.cmd import compute_cmd_features, dustmaps_cmd_from_fields


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
