"""Observed dip-recurrence labels shared by Review and ML score products.

These labels describe the triggered-dip extraction, not an astrophysical
claim that an object is intrinsically one-off or recurrent outside the
available observing baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DIPPER_RECURRENCE_CLASS_COLUMN = "dipper_recurrence_class"
DIPPER_RECURRENCE_EVIDENCE_COLUMN = "dipper_recurrence_evidence"

RECURRENT_DIPPER = "recurrent"
NON_RECURRENT_DIPPER = "non_recurrent"
UNKNOWN_DIPPER_RECURRENCE = "unknown"


def _numeric_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index, dtype="float64")
    return pd.to_numeric(table[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _boolean_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index, dtype="bool")
    raw = table[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    truthy_text = raw.fillna("").astype(str).str.strip().str.lower().isin(
        {"true", "t", "yes", "y", "1"}
    )
    return numeric.eq(1).fillna(False) | truthy_text


def add_observed_dipper_recurrence(table: pd.DataFrame) -> pd.DataFrame:
    """Attach a three-state observed dip-recurrence classification.

    A source is ``recurrent`` only when the event extractor finds at least two
    dip runs.  It is ``non_recurrent`` when exactly one run is detected, or
    when the pipeline's explicit single-event flag is set.  Everything else is
    ``unknown``.  This deliberately does *not* use ``dipper_n_valid_dips``:
    it counts valid dip detections within a run and is not an independent
    event-complex count.
    """

    out = table.copy()
    run_count = _numeric_column(out, "dip_run_count")
    single_event = _boolean_column(out, "dip_is_single_event")

    recurrent = run_count.ge(2)
    non_recurrent = ~recurrent & (run_count.eq(1) | single_event)

    out[DIPPER_RECURRENCE_CLASS_COLUMN] = np.select(
        [recurrent, non_recurrent],
        [RECURRENT_DIPPER, NON_RECURRENT_DIPPER],
        default=UNKNOWN_DIPPER_RECURRENCE,
    )
    out[DIPPER_RECURRENCE_EVIDENCE_COLUMN] = np.select(
        [recurrent, run_count.eq(1), non_recurrent],
        [
            "two_or_more_detected_dip_runs",
            "one_detected_dip_run",
            "pipeline_single_event_flag",
        ],
        default="insufficient_dip_recurrence_evidence",
    )
    return out


__all__ = [
    "DIPPER_RECURRENCE_CLASS_COLUMN",
    "DIPPER_RECURRENCE_EVIDENCE_COLUMN",
    "NON_RECURRENT_DIPPER",
    "RECURRENT_DIPPER",
    "UNKNOWN_DIPPER_RECURRENCE",
    "add_observed_dipper_recurrence",
]
