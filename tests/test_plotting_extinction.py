from __future__ import annotations

import pandas as pd
import pytest

from malca.plotting.extinction import add_dereddened_ir_magnitudes, dereddened_color


def test_add_dereddened_ir_magnitudes_uses_project_band_coefficients() -> None:
    frame = pd.DataFrame(
        {
            "A_v_3d": [2.0],
            "tmass_j": [12.5],
            "tmass_h": [12.0],
            "tmass_k": [11.0],
            "w1": [10.0],
            "w2": [9.5],
            "w3": [8.0],
            "w4": [7.0],
        }
    )

    corrected = add_dereddened_ir_magnitudes(frame)

    assert corrected.loc[0, "tmass_j_0"] == pytest.approx(12.5 - 0.282 * 2.0)
    assert corrected.loc[0, "tmass_h_0"] == pytest.approx(12.0 - 0.175 * 2.0)
    assert corrected.loc[0, "tmass_k_0"] == pytest.approx(11.0 - 0.112 * 2.0)
    assert corrected.loc[0, "w1_0"] == pytest.approx(10.0 - 0.061 * 2.0)
    assert corrected.loc[0, "w2_0"] == pytest.approx(9.5 - 0.047 * 2.0)
    assert corrected.loc[0, "w3_0"] == pytest.approx(8.0)
    assert corrected.loc[0, "w4_0"] == pytest.approx(7.0)
    assert dereddened_color(corrected, "w1", "w2").iloc[0] == pytest.approx(0.5 - 0.014 * 2.0)


def test_missing_extinction_preserves_observed_magnitudes() -> None:
    corrected = add_dereddened_ir_magnitudes(pd.DataFrame({"tmass_k": [11.0], "w3": [8.0]}))

    assert corrected.loc[0, "tmass_k_0"] == pytest.approx(11.0)
    assert corrected.loc[0, "w3_0"] == pytest.approx(8.0)
