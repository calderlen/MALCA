from __future__ import annotations

import pytest

from malca.ltv.core import compute_trend_metrics


def test_compute_trend_metrics_linear_trend_has_mag_scale_diff() -> None:
    indexes = [1.0, 2.0, 3.0]
    meds = [14.0, 14.1, 14.2]

    lin_slope, quad_slope, coeff1, coeff2, max_diff = compute_trend_metrics(indexes, meds)

    assert lin_slope == pytest.approx(0.1, abs=1e-10)
    assert quad_slope == pytest.approx(0.0, abs=1e-10)
    assert coeff1 == pytest.approx(0.1, abs=1e-10)
    assert coeff2 == pytest.approx(13.9, abs=1e-10)
    assert max_diff == pytest.approx(0.2, abs=1e-10)


def test_compute_trend_metrics_quadratic_uses_true_vertex_for_max_diff() -> None:
    indexes = [1.0, 2.0, 3.0]
    meds = [11.0, 10.0, 11.0]

    lin_slope, quad_slope, coeff1, coeff2, max_diff = compute_trend_metrics(indexes, meds)

    assert lin_slope == pytest.approx(0.0, abs=1e-10)
    assert quad_slope == pytest.approx(1.0, abs=1e-10)
    assert coeff1 == pytest.approx(-4.0, abs=1e-10)
    assert coeff2 == pytest.approx(14.0, abs=1e-10)
    assert max_diff == pytest.approx(1.0, abs=1e-10)
