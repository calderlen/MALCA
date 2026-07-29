from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from malca.meta_analysis.ml import candidate_features
from malca.meta_analysis.ml.candidate_features import (
    RECOVERY_BOUNDED_EVENT_FEATURES,
    add_next_iteration_context_features,
    add_recovery_bounded_event_features,
)


def test_next_iteration_context_derives_reduced_proper_motion() -> None:
    table = pd.DataFrame(
        {
            "A_v_3d": [0.3, 0.8],
            "derived_wjk": [1.2, 2.1],
            "phot_g_mean_mag": [15.0, 14.0],
            "pmra": [30.0, 0.0],
            "pmdec": [40.0, 0.0],
            "iphas_r_ha": [0.2, np.nan],
            "vphas_r_ha": [np.nan, 0.4],
        }
    )

    out = add_next_iteration_context_features(table)

    expected_h_g = 15.0 + 5.0 * np.log10(50.0 / 1000.0) + 5.0
    assert out.loc[0, "reduced_proper_motion_g"] == pytest.approx(expected_h_g)
    assert np.isnan(out.loc[1, "reduced_proper_motion_g"])
    assert out.loc[0, "A_v_3d"] == pytest.approx(0.3)
    assert out.loc[1, "derived_wjk"] == pytest.approx(2.1)
    assert out.loc[0, "iphas_r_ha"] == pytest.approx(0.2)
    assert out.loc[1, "vphas_r_ha"] == pytest.approx(0.4)


def test_recovery_feature_cache_is_population_wide_and_reused(
    tmp_path, monkeypatch
) -> None:
    first_lc = tmp_path / "first.dat"
    second_lc = tmp_path / "second.dat"
    first_lc.write_text("first\n", encoding="utf-8")
    second_lc.write_text("second\n", encoding="utf-8")
    table = pd.DataFrame(
        {
            "candidate_id": ["first", "second"],
            "lc_path": [str(first_lc), str(second_lc)],
        }
    )

    class FakeWindow:
        def to_metrics(self, _times):
            return {
                "dimming_complex_duration_days": 12.5,
                "dimming_complex_is_lower_limit": False,
                "event_integrated_excess": 1.7,
                "event_component_epochs": 6,
                "delta_mag_peak": 0.8,
                "left_baseline_recovered": True,
                "right_baseline_recovered": True,
                "event_window_gap_count": 1,
                "event_window_max_gap_days": 20.0,
                "left_event_boundary_type": "recovery",
                "right_event_boundary_type": "recovery",
                "dimming_complex_status": "recovery_bounded",
            }

    calls: list[str] = []

    def fake_measure(candidate_id, _lc_path):
        calls.append(str(candidate_id))
        if candidate_id == "second":
            raise RuntimeError(
                "no recovery-anchored dimming bracket with a supported event seed"
            )
        return SimpleNamespace(
            epochs=pd.DataFrame({"t": [1.0, 2.0, 3.0]}),
            window=FakeWindow(),
        )

    monkeypatch.setattr(
        candidate_features, "measure_dimming_complex_window", fake_measure
    )
    cache_path = tmp_path / "recovery.parquet"
    out = add_recovery_bounded_event_features(
        table, cache_path, workers=1, checkpoint_every=1
    )

    assert calls == ["first", "second"]
    assert out.loc[0, "dimming_complex_duration_days"] == pytest.approx(12.5)
    assert out.loc[0, "left_baseline_recovered"] == pytest.approx(1.0)
    assert out.loc[1, "recovery_feature_state"] == "no_supported_event"
    assert out.loc[1, list(RECOVERY_BOUNDED_EVENT_FEATURES)].isna().all()

    def fail_if_remeasured(*_args, **_kwargs):
        raise AssertionError("current cache rows should be reused")

    monkeypatch.setattr(
        candidate_features, "measure_dimming_complex_window", fail_if_remeasured
    )
    reused = add_recovery_bounded_event_features(
        table, cache_path, workers=1, checkpoint_every=1
    )
    assert reused.loc[0, "event_integrated_excess"] == pytest.approx(1.7)

    first_lc.write_text("first changed\n", encoding="utf-8")
    calls.clear()
    monkeypatch.setattr(
        candidate_features, "measure_dimming_complex_window", fake_measure
    )
    refreshed = add_recovery_bounded_event_features(
        table, cache_path, workers=1, checkpoint_every=1
    )
    assert calls == ["first"]
    assert refreshed.loc[0, "dimming_complex_duration_days"] == pytest.approx(
        12.5
    )
