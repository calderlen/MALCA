from __future__ import annotations

import numpy as np
import pytest

from malca.plotting.irac import irac_vega_magnitude, irac_vega_magnitude_error


def test_irac_vega_magnitude_uses_standard_zero_point() -> None:
    magnitude = irac_vega_magnitude(np.array([280.9, 28.09]), "IRAC1")

    assert magnitude[0] == pytest.approx(0.0)
    assert magnitude[1] == pytest.approx(2.5)


def test_irac_vega_magnitude_error_propagates_flux_uncertainty() -> None:
    error = irac_vega_magnitude_error(np.array([10.0]), np.array([1.0]))

    assert error[0] == pytest.approx(2.5 / np.log(10.0) * 0.1)
