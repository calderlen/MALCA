from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.core.baseline import per_camera_gp_baseline_masked
from malca.core.utils import (
    compute_field_summary,
    fred,
    filter_bad_cameras,
    identify_bad_cameras,
    identify_offset_cameras,
    identify_residual_bad_cameras,
)


def test_peak_centered_bazin_fred_matches_requested_peak_and_quiescent_tails() -> None:
    t = np.linspace(-100.0, 100.0, 4001)
    profile = fred(
        t,
        delta_m_peak=-0.75,
        t_peak=0.0,
        tau_rise=2.0,
        tau_fall=12.0,
    )

    assert np.all(np.isfinite(profile))
    assert fred(np.array([0.0]), -0.75, 0.0, 2.0, 12.0)[0] == pytest.approx(-0.75)
    assert t[int(np.argmin(profile))] == pytest.approx(0.0)
    assert abs(profile[0]) < 1e-8
    assert abs(profile[-1]) < 5e-4


def test_peak_centered_bazin_fred_requires_ordered_positive_timescales() -> None:
    with pytest.raises(ValueError, match="0 < tau_rise < tau_fall"):
        fred(np.array([0.0]), -0.5, 0.0, 2.0, 2.0)


def _make_camera_df(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = 120
    for cam in (1, 2):
        jd = 2458000.0 + np.arange(n, dtype=float)
        mag = 14.0 + rng.normal(0.0, 0.02, size=n)
        err = np.full(n, 0.02)
        for t, m, e in zip(jd, mag, err):
            rows.append({"JD": t, "mag": m, "error": e, "camera#": cam})
    return pd.DataFrame(rows)


def test_compute_field_summary_single_field() -> None:
    df = pd.DataFrame(
        {
            "camera_name": ["cam1", "cam1", "cam2"],
            "field": ["field1", "field1", "field1"],
        }
    )

    out = compute_field_summary(df)

    assert out["asassn_field_key"] == "field1"
    assert out["asassn_fields"] == "field1"
    assert out["asassn_field_count"] == 1
    assert out["asassn_field_key_fraction"] == 1.0
    assert out["camera_name_key"] == "cam1"
    assert out["camera_names"] == "cam1,cam2"
    assert out["camera_name_count"] == 2
    assert out["camera_name_key_fraction"] == 2 / 3


def test_compute_field_summary_mixed_fields() -> None:
    df = pd.DataFrame(
        {
            "camera_name": ["cam1", "cam1", "cam2", "cam3"],
            "field": ["field1", "field2", "field2", "field2"],
        }
    )

    out = compute_field_summary(df)

    assert out["asassn_field_key"] == "field2"
    assert out["asassn_fields"] == "field1,field2"
    assert out["asassn_field_count"] == 2
    assert out["asassn_field_key_fraction"] == 0.75


def test_compute_field_summary_ties_are_deterministic() -> None:
    df = pd.DataFrame(
        {
            "cam_field": [
                "cam2/fieldB",
                "cam1/fieldA",
                "cam3/fieldB",
                "cam4/fieldA",
            ]
        }
    )

    out = compute_field_summary(df)

    assert out["asassn_field_key"] == "fieldA"
    assert out["asassn_fields"] == "fieldA,fieldB"
    assert out["asassn_field_key_fraction"] == 0.5
    assert out["camera_name_key"] == "cam1"
    assert out["camera_names"] == "cam1,cam2,cam3,cam4"
    assert out["camera_name_count"] == 4
    assert out["camera_name_key_fraction"] == 0.25


def test_filter_bad_cameras_flags_isolated_catastrophic_camera():
    df = _make_camera_df()

    # Inject a single catastrophic outlier in camera 1 only.
    idx = df.index[(df["camera#"] == 1) & (df["JD"] == 2458060.0)]
    assert len(idx) == 1
    df.loc[idx[0], "mag"] = 18.2

    df_filtered, bad = filter_bad_cameras(
        df,
        catastrophic_mag_excursion=3.0,
        catastrophic_min_count=1,
        catastrophic_max_fraction=0.05,
    )

    assert 1 in bad
    assert (df_filtered["camera#"] == 1).sum() == 0
    assert (df_filtered["camera#"] == 2).sum() > 0


def test_raw2_scatter_stats_do_not_hard_remove_camera():
    rng = np.random.default_rng(2)
    rows = []
    for cam, scatter in ((1, 0.12), (2, 0.01)):
        jd = 2458000.0 + np.arange(80, dtype=float)
        mag = 14.0 + rng.normal(0.0, scatter, size=80)
        for t, m in zip(jd, mag):
            rows.append({"JD": t, "mag": m, "error": 0.02, "camera#": cam})
    df = pd.DataFrame(rows)
    raw2 = pd.DataFrame(
        {
            "camera#": [1, 2],
            "median": [14.0, 14.0],
            "sig1_low": [13.99, 13.99],
            "sig1_high": [14.01, 14.01],
            "p90_low": [13.98, 13.98],
            "p90_high": [14.02, 14.02],
            "expected_scatter": [0.01, 0.01],
        }
    )

    assert identify_bad_cameras(df, raw2_df=raw2) == set()


def test_per_camera_baseline_corrected_raw_offset_is_not_residual_bad():
    rng = np.random.default_rng(3)
    rows = []
    for cam, offset in ((1, 0.0), (2, 0.0), (3, 0.45)):
        jd = 2458000.0 + np.arange(140, dtype=float)
        mag = 14.0 + offset + rng.normal(0.0, 0.01, size=140)
        for t, m in zip(jd, mag):
            rows.append({"JD": t, "mag": m, "error": 0.02, "camera#": cam, "saturated": 0})
    df = pd.DataFrame(rows)

    raw_offset_bad, _ = identify_offset_cameras(df)
    df_base = per_camera_gp_baseline_masked(df)
    residual_bad = identify_residual_bad_cameras(df_base)

    assert 3 in raw_offset_bad
    assert residual_bad == set()


def test_residual_scatter_filter_flags_camera_after_baseline():
    rng = np.random.default_rng(4)
    rows = []
    for cam, scatter in ((1, 0.01), (2, 0.01), (3, 0.18)):
        jd = 2458000.0 + np.arange(140, dtype=float)
        resid = rng.normal(0.0, scatter, size=140)
        for t, r in zip(jd, resid):
            rows.append({"JD": t, "resid": r, "camera#": cam})
    df_base = pd.DataFrame(rows)

    residual_bad = identify_residual_bad_cameras(df_base)

    assert residual_bad == {3}
