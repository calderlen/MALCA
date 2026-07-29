from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.train_july1_dipper_ml import (
    ADDITIONAL_LC_FEATURES,
    EXPANDED_FEATURE_SET,
    EXTERNAL_CONTEXT_FEATURES,
    EXTERNAL_CONTEXT_FEATURE_SET,
    RECOVERY_BOUNDED_EVENT_FEATURES,
    REDUNDANT_MODEL_FEATURES,
    TARGET_COLUMN,
    add_external_context_features,
    select_feature_columns,
)


def _external_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bprp0": [1.2, 0.8],
            "derived_mrp": [4.0, 5.1],
            "parallax": [10.0, -1.0],
            "parallax_error": [2.0, 0.0],
            "ruwe": [1.1, 1.8],
            "derived_j_k": [0.8, 0.6],
            "w1_w2": [0.2, 0.3],
            "w1_w3": [0.9, 0.2],
            "w2_w3": [0.7, 0.1],
            "w3_err": [0.1, 0.0],
            "w4_err": [0.2, float("nan")],
            "sed_alpha": [-1.2, -0.7],
            "tess_flux_range": [0.08, 0.15],
        }
    )


def test_add_external_context_features_uses_shared_requested_block() -> None:
    out = add_external_context_features(_external_source_frame())

    assert out.loc[0, "bprp0"] == pytest.approx(1.2)
    assert out.loc[0, "derived_mrp"] == pytest.approx(4.0)
    assert out.loc[0, "parallax_snr"] == pytest.approx(5.0)
    assert out.loc[0, "derived_j_k"] == pytest.approx(0.8)
    assert out.loc[0, "w1_w2"] == pytest.approx(0.2)
    assert out.loc[0, "w1_w3"] == pytest.approx(0.9)
    assert out.loc[0, "w2_w3"] == pytest.approx(0.7)
    assert out.loc[0, "wise_w3_missing"] == 0
    assert out.loc[1, "wise_w3_missing"] == 1
    assert np.isnan(out.loc[1, "parallax_snr"])


def test_external_context_feature_set_strictly_includes_curated_blocks() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            TARGET_COLUMN: ["dipper_like", "not_dipper", "dipper_like", "not_dipper"],
            "stats_signal": [0.0, 1.0, 0.2, 0.8],
        }
    )
    for column in (
        *ADDITIONAL_LC_FEATURES,
        *RECOVERY_BOUNDED_EVENT_FEATURES,
        *EXTERNAL_CONTEXT_FEATURES,
    ):
        frame[column] = [0.0, 1.0, 0.25, 0.75]
    for column in (
        "periodicity_method",
        "dip_best_morph",
        "jump_best_morph",
        "left_event_boundary_type",
        "right_event_boundary_type",
        "dimming_complex_status",
    ):
        frame[column] = ["a", "b", "a", "b"]

    selected = select_feature_columns(
        frame,
        feature_set=EXTERNAL_CONTEXT_FEATURE_SET,
        min_non_null=2,
        max_cardinality=50,
        max_features=250,
    )

    assert len(selected) == (
        1
        + len(ADDITIONAL_LC_FEATURES)
        + len(RECOVERY_BOUNDED_EVENT_FEATURES)
        + len(EXTERNAL_CONTEXT_FEATURES)
    )
    assert set(ADDITIONAL_LC_FEATURES).issubset(selected)
    assert set(RECOVERY_BOUNDED_EVENT_FEATURES).issubset(selected)
    assert set(EXTERNAL_CONTEXT_FEATURES).issubset(selected)


def test_curated_feature_selection_excludes_known_redundant_columns() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            TARGET_COLUMN: ["dipper_like", "not_dipper", "dipper_like", "not_dipper"],
            "stats_signal": [0.0, 1.0, 0.2, 0.8],
        }
    )
    for index, column in enumerate(
        (
            *ADDITIONAL_LC_FEATURES,
            *RECOVERY_BOUNDED_EVENT_FEATURES,
            *EXTERNAL_CONTEXT_FEATURES,
        )
    ):
        frame[column] = np.asarray([0.0, 1.0, 0.25, 0.75]) + index
    for column in (
        "periodicity_method",
        "dip_best_morph",
        "jump_best_morph",
        "left_event_boundary_type",
        "right_event_boundary_type",
        "dimming_complex_status",
    ):
        frame[column] = ["a", "b", "a", "b"]
    for column in REDUNDANT_MODEL_FEATURES:
        frame[column] = [1.0, 2.0, 3.0, 4.0]

    selected = select_feature_columns(
        frame,
        feature_set=EXPANDED_FEATURE_SET,
        min_non_null=2,
        max_cardinality=50,
        max_features=250,
    )

    assert set(REDUNDANT_MODEL_FEATURES).isdisjoint(selected)
    assert set(RECOVERY_BOUNDED_EVENT_FEATURES).issubset(selected)
    assert set(EXTERNAL_CONTEXT_FEATURES).issubset(selected)
