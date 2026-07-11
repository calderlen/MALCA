import numpy as np
import pandas as pd

from malca.core.phase import (
    BAND_LABELS,
    align_v_to_g_magnitude,
    camera_labels,
    compute_band_phase_lag,
    phase_fold_dataframe,
    phase_time_dataframe,
    resolve_phase_epoch,
    resolve_phase_period,
    template_phase_lag,
)


def test_resolve_phase_period_prefers_manual_then_payload_priority() -> None:
    payload = {
        "period_consensus_days": 2.0,
        "vsx_period": 3.0,
        "periodicity_period": 4.0,
    }

    assert resolve_phase_period(payload, override_period=1.5) == (1.5, "manual/search")
    assert resolve_phase_period(payload) == (2.0, "period_consensus_days")
    assert resolve_phase_period({"phase_period_days": 1.25, "phase_source": "pdm"}) == (1.25, "pdm")
    assert resolve_phase_period({"vsx_period": 3.0}) == (None, "")
    assert resolve_phase_period({"catalog_period": 3.0}) == (None, "")
    assert resolve_phase_period({"periodicity_period": "4.5"}) == (4.5, "periodicity_period")
    assert resolve_phase_period({"periodicity_period": "4.5"}, include_periodogram_periods=False) == (None, "")
    assert resolve_phase_period({"lsp_period": "5.5"}) == (5.5, "lsp_period")
    assert resolve_phase_period({"lsp_period": "5.5"}, include_lsp=False) == (None, "")
    assert resolve_phase_period({"lsp_period": "bad"}) == (None, "")
    assert resolve_phase_period({"pre_periodicity_selected_period": 6.0}) == (6.0, "pre_periodicity_selected_period")
    assert resolve_phase_period({"pre_periodicity_selected_period": 6.0, "periodicity_period": 5.5}) == (
        6.0,
        "pre_periodicity_selected_period",
    )
    assert resolve_phase_period({"pdm_period": 7.0}) == (7.0, "pdm_period")
    assert resolve_phase_period({"ce_period": 8.0}) == (8.0, "ce_period")
    assert resolve_phase_period({"stats_variability_lomb_scargle_best_period_days": 9.0}) == (
        9.0,
        "stats_variability_lomb_scargle_best_period_days",
    )
    assert resolve_phase_period(
        {
            "period_consensus_days": 2.0,
            "stats_variability_lomb_scargle_best_period_days": 9.0,
        },
        include_periodogram_periods=False,
    ) == (2.0, "period_consensus_days")
    assert resolve_phase_period(
        {"stats_variability_lomb_scargle_best_period_days": 9.0},
        include_periodogram_periods=False,
    ) == (None, "")
    assert resolve_phase_period({"pdm_period": 7.0}, include_periodogram_periods=False) == (None, "")
    assert resolve_phase_period({"ce_period": 8.0}, include_periodogram_periods=False) == (None, "")


def test_resolve_phase_epoch_uses_minimum_finite_jd() -> None:
    df = pd.DataFrame({"JD": [np.nan, 12.0, 10.0, 11.0]})

    assert resolve_phase_epoch(df) == 10.0
    assert resolve_phase_epoch(df, explicit_epoch_jd=9.5) == 9.5


def test_align_v_to_g_magnitude_subtracts_v_minus_g_offset() -> None:
    df = pd.DataFrame(
        {
            "v_g_band": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "mag": [14.0, 14.2, 13.8, 14.1, 13.9, 15.0, 15.2, 14.8, 15.1, 14.9],
        }
    )

    aligned, offset = align_v_to_g_magnitude(df)

    assert np.isclose(offset, 1.0)
    assert np.allclose(aligned.loc[aligned["v_g_band"] == 0, "mag"], df.loc[df["v_g_band"] == 0, "mag"])
    assert np.isclose(aligned.loc[aligned["v_g_band"] == 1, "mag"].median(), 14.0)


def test_phase_fold_dataframe_supports_residual_mode_labels_and_duplicate_cycles() -> None:
    df = pd.DataFrame(
        {
            "JD": [10.0, 10.5, 11.0, 11.5],
            "mag": [14.0, 14.1, 14.2, 14.3],
            "resid": [0.0, -0.1, 0.2, -0.2],
            "error": [0.01, 0.01, 0.02, 0.02],
            "v_g_band": [0, 1, 0, 1],
            "camera#": [1, 1, 2, 2],
        }
    )

    folded, diag = phase_fold_dataframe(df, 1.0, value_mode="resid")

    assert len(folded) == 8
    assert set(folded["band_label"]) == set(BAND_LABELS.values())
    assert set(folded["camera_label"]) == {"1", "2"}
    assert set(np.round(folded["phase"], 6)) == {0.0, 0.5, 1.0, 1.5}
    assert np.allclose(folded.iloc[:4]["phase_value"], df["resid"])
    assert diag["epoch_jd"] == 10.0
    assert diag["value_col"] == "resid"


def test_phase_time_dataframe_tracks_cycles_and_duplicates_phase() -> None:
    df = pd.DataFrame(
        {
            "JD": [10.0, 10.25, 10.75, 11.25],
            "mag": [14.0, 14.1, 14.2, 14.3],
            "resid": [0.0, 0.3, -0.2, 0.5],
            "error": [0.01, 0.01, 0.02, 0.02],
            "v_g_band": [0, 1, 0, 1],
            "camera#": [1, 1, 2, 2],
        }
    )

    phase_time, diag = phase_time_dataframe(df, 1.0, value_mode="resid")
    original = phase_time.iloc[: len(df)]
    duplicate = phase_time.iloc[len(df):].reset_index(drop=True)

    assert len(phase_time) == 8
    assert np.all((original["phase"] >= 0.0) & (original["phase"] < 1.0))
    assert np.allclose(duplicate["phase"], original["phase"].to_numpy() + 1.0)
    assert original["cycle"].tolist() == [0, 0, 0, 1]
    assert np.allclose(original["phase_value"], df["resid"])
    assert set(phase_time["band_label"]) == set(BAND_LABELS.values())
    assert set(phase_time["camera_label"]) == {"1", "2"}
    assert diag["epoch_jd"] == 10.0
    assert diag["value_col"] == "resid"


def test_camera_labels_prefers_existing_and_common_columns() -> None:
    assert camera_labels(pd.DataFrame({"camera_label": ["cam-a"]})).tolist() == ["cam-a"]
    assert camera_labels(pd.DataFrame({"camera_name": ["bd"]})).tolist() == ["bd"]
    assert camera_labels(pd.DataFrame({"camera#": [3]})).tolist() == ["3"]
    assert camera_labels(pd.DataFrame({"camera": ["c1"]})).tolist() == ["c1"]
    assert camera_labels(pd.DataFrame({"JD": [1]})).tolist() == ["unknown"]


def test_template_phase_lag_signed_positive_when_second_template_is_later() -> None:
    n_bins = 40
    template_g = np.zeros(n_bins)
    template_v = np.zeros(n_bins)
    template_g[8] = 1.0
    template_v[12] = 1.0

    assert np.isclose(template_phase_lag(template_g, template_v), 0.1)
    assert np.isclose(template_phase_lag(template_g, template_v, signed=True), 0.1)
    assert np.isclose(template_phase_lag(template_v, template_g, signed=True), -0.1)


def test_compute_band_phase_lag_from_folded_points() -> None:
    phase = np.linspace(0.0, 1.0, 240, endpoint=False)
    g_values = np.exp(-0.5 * ((np.mod(phase - 0.20 + 0.5, 1.0) - 0.5) / 0.03) ** 2)
    v_values = np.exp(-0.5 * ((np.mod(phase - 0.30 + 0.5, 1.0) - 0.5) / 0.03) ** 2)
    df = pd.DataFrame(
        {
            "phase": np.concatenate([phase, phase]),
            "phase_value": np.concatenate([g_values, v_values]),
            "v_g_band": np.concatenate([np.zeros_like(phase), np.ones_like(phase)]),
        }
    )

    lag = compute_band_phase_lag(df, n_bins=40)

    assert np.isclose(lag["phase_lag_g_v_cycles"], 0.1, atol=0.025)
    assert np.isclose(lag["phase_lag_g_v_abs_cycles"], 0.1, atol=0.025)
