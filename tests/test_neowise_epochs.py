from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.catalogs.neowise_epochs import combine_neowise_epochs
from malca.enrichment import vetting


def test_combine_neowise_epochs_uses_visit_medians_and_scatter_errors() -> None:
    raw = pd.DataFrame(
        {
            "mjd": [59000.0, 59000.1, 59000.2, 59180.0, 59180.1],
            "w1mpro": [12.0, 12.1, 15.0, 12.5, 12.7],
            "w1sigmpro": [0.03, 0.03, 0.03, 0.04, 0.04],
            "w2mpro": [11.5, 11.6, 11.7, 12.0, 12.2],
            "w2sigmpro": [0.04, 0.04, 0.04, 0.05, 0.05],
        }
    )

    epochs = combine_neowise_epochs(raw)

    assert len(epochs) == 2
    assert epochs.loc[0, "mjd"] == pytest.approx(59000.1)
    assert epochs.loc[0, "w1mpro"] == pytest.approx(12.1)
    assert epochs.loc[0, "w1_scatter"] == pytest.approx(0.14826)
    assert epochs.loc[0, "w1sigmpro"] == pytest.approx(0.14826 / np.sqrt(3.0))
    assert epochs.loc[0, "w1err"] == epochs.loc[0, "w1sigmpro"]
    assert epochs.loc[0, "n_points"] == 3
    assert epochs.loc[0, "w1_n_points"] == 3
    assert bool(epochs.loc[0, "neowise_epoch_binned"])


def test_combine_neowise_epochs_handles_bands_independently() -> None:
    raw = pd.DataFrame(
        {
            "mjd": [59000.0, 59000.2],
            "w1mpro": [12.0, 12.2],
            "w1sigmpro": [0.04, 0.04],
            "w2mpro": [11.5, np.nan],
            "w2sigmpro": [0.05, np.nan],
        }
    )

    epochs = combine_neowise_epochs(raw)

    assert len(epochs) == 1
    assert epochs.loc[0, "w1_n_points"] == 2
    assert epochs.loc[0, "w2_n_points"] == 1
    assert epochs.loc[0, "w2mpro"] == pytest.approx(11.5)
    assert epochs.loc[0, "w2sigmpro"] == pytest.approx(0.05)


def test_combine_neowise_epochs_is_idempotent() -> None:
    raw = pd.DataFrame(
        {
            "mjd": [59000.0, 59000.1],
            "w1mpro": [12.0, 12.2],
            "w1sigmpro": [0.04, 0.04],
            "w2mpro": [11.5, 11.7],
            "w2sigmpro": [0.05, 0.05],
        }
    )

    once = combine_neowise_epochs(raw)
    twice = combine_neowise_epochs(once)

    pd.testing.assert_frame_equal(once, twice)


def test_neowise_cache_summary_counts_binned_visits() -> None:
    raw = pd.DataFrame(
        {
            "mjd": [59000.0, 59000.1, 59180.0],
            "w1mpro": [12.0, 12.2, 12.6],
            "w1sigmpro": [0.04, 0.04, 0.05],
            "w2mpro": [11.5, 11.7, 12.0],
            "w2sigmpro": [0.05, 0.05, 0.06],
            "target_identity_status": ["matched", "matched", "matched"],
            "target_sep_arcsec": [0.2, 0.3, 0.25],
        }
    )

    summary = vetting._summarize_neowise_lc(raw)

    assert summary["neowise_n_epochs"] == 2
    assert summary["neowise_w1_range"] == pytest.approx(0.5)
    assert summary["neowise_w2_range"] == pytest.approx(0.4)
    assert summary["neowise_identity_status"] == "matched"
