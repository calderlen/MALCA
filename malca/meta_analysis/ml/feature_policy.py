"""Shared feature inclusion and exclusion policy for MALCA ML models."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


# Retain these fields in candidate/statistics products for provenance and
# diagnostics, but never pass them to a model.  Each is either an absolute
# observing epoch, a deterministic rescaling of a retained feature, or an
# explicit missingness flag that LightGBM already represents natively.
MODEL_FEATURE_EXCLUSION_COLUMNS = frozenset(
    {
        "stats_jd_start",
        "stats_jd_end",
        "stats_trend_slope_mag_per_year",
        "stats_median_abs_dev",
        "stats_variability_quasi_periodicity_populated_bins",
        "wise_w3_missing",
        "wise_w4_missing",
    }
)

STATS_MODEL_FEATURE_EXCLUSION_COLUMNS = frozenset(
    column
    for column in MODEL_FEATURE_EXCLUSION_COLUMNS
    if column.startswith("stats_")
)


def restore_legacy_excluded_model_features(
    model_input: pd.DataFrame,
    source_frame: pd.DataFrame,
    required_features: Iterable[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Restore excluded fields only when scoring an older saved model.

    The shared policy prevents these deterministic aliases from entering new
    training runs.  Older persisted models may still require them, however.
    This bridge copies the original diagnostic columns from the same candidate
    rows for backwards-compatible scoring; it never changes the feature set
    selected for a new model.
    """

    restored = model_input.copy()
    required = tuple(str(column) for column in required_features)
    missing_legacy = tuple(
        column
        for column in required
        if column in MODEL_FEATURE_EXCLUSION_COLUMNS
        and column not in restored.columns
    )
    if not missing_legacy:
        return restored, ()
    if "candidate_id" not in restored or "candidate_id" not in source_frame:
        raise ValueError("Legacy feature restoration requires candidate_id in both frames")

    source = source_frame.drop_duplicates("candidate_id").set_index("candidate_id")
    restored_ids = restored["candidate_id"].astype(str)
    restored_columns: list[str] = []
    for column in missing_legacy:
        if column not in source.columns:
            continue
        values = source[column].copy()
        values.index = values.index.astype(str)
        restored[column] = restored_ids.map(values)
        restored_columns.append(column)
    return restored, tuple(restored_columns)


__all__ = [
    "MODEL_FEATURE_EXCLUSION_COLUMNS",
    "STATS_MODEL_FEATURE_EXCLUSION_COLUMNS",
    "restore_legacy_excluded_model_features",
]
