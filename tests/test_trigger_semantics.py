from __future__ import annotations

import numpy as np
import pytest

from malca.triggering import (
    normalize_trigger_block,
    posterior_probability_threshold,
    resolve_trigger_indices,
)


def test_posterior_probability_threshold_accepts_percent_or_fraction() -> None:
    assert posterior_probability_threshold(99.9) == pytest.approx(0.999)
    assert posterior_probability_threshold(0.975) == pytest.approx(0.975)


def test_resolve_trigger_indices_logbf_mode() -> None:
    out = resolve_trigger_indices(
        trigger_mode="logbf",
        log_bf_local=np.array([1.0, 4.9, 5.0, np.nan, 6.2]),
        event_probability=None,
        logbf_threshold=5.0,
        significance_threshold=99.0,
    )
    assert out["trigger_mode"] == "logbf"
    assert np.array_equal(out["event_indices"], np.array([2, 4]))
    assert float(out["trigger_threshold"]) == pytest.approx(5.0)
    assert float(out["trigger_max"]) == pytest.approx(6.2)


def test_resolve_trigger_indices_posterior_mode() -> None:
    out = resolve_trigger_indices(
        trigger_mode="posterior_prob",
        log_bf_local=np.array([0.1, 0.2]),
        event_probability=np.array([0.8, 0.95, np.nan, 0.97]),
        logbf_threshold=5.0,
        significance_threshold=95.0,
    )
    assert out["trigger_mode"] == "posterior_prob"
    assert np.array_equal(out["event_indices"], np.array([1, 3]))
    assert float(out["trigger_threshold"]) == pytest.approx(0.95)
    assert float(out["trigger_max"]) == pytest.approx(0.97)


def test_resolve_trigger_indices_requires_event_prob_in_posterior_mode() -> None:
    with pytest.raises(RuntimeError, match="requires event probabilities"):
        resolve_trigger_indices(
            trigger_mode="posterior_prob",
            log_bf_local=np.array([1.0]),
            event_probability=None,
            logbf_threshold=5.0,
            significance_threshold=99.0,
        )


def test_normalize_trigger_block_backfills_summary_fields() -> None:
    block = {
        "event_indices": [1, 4, 7],
        "log_bf_local": np.array([0.0, 2.0, np.nan]),
        "event_probability": np.array([0.1, 0.2, 0.3]),
    }
    out = normalize_trigger_block(block, kind="dip", default_trigger_mode="posterior_prob")
    assert out["n_dips"] == 3
    assert float(out["max_log_bf_local"]) == pytest.approx(2.0)
    assert float(out["max_event_prob"]) == pytest.approx(0.3)
    assert out["trigger_mode"] == "posterior_prob"
