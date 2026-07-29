from __future__ import annotations

import numpy as np
import pandas as pd

from malca.review.dipper_recurrence import (
    DIPPER_RECURRENCE_CLASS_COLUMN,
    DIPPER_RECURRENCE_EVIDENCE_COLUMN,
    add_observed_dipper_recurrence,
)


def test_observed_dipper_recurrence_uses_dip_runs_not_dip_counts() -> None:
    source = pd.DataFrame(
        {
            "dip_run_count": [3, 1, np.nan, 0, 1],
            "dip_is_single_event": [False, False, True, False, True],
            # A source can contain many valid dip detections inside one run;
            # that does not establish recurrence.
            "dipper_n_valid_dips": [3, 40, 12, 0, 20],
        }
    )

    out = add_observed_dipper_recurrence(source)

    assert out[DIPPER_RECURRENCE_CLASS_COLUMN].tolist() == [
        "recurrent",
        "non_recurrent",
        "non_recurrent",
        "unknown",
        "non_recurrent",
    ]
    assert out[DIPPER_RECURRENCE_EVIDENCE_COLUMN].tolist() == [
        "two_or_more_detected_dip_runs",
        "one_detected_dip_run",
        "pipeline_single_event_flag",
        "insufficient_dip_recurrence_evidence",
        "one_detected_dip_run",
    ]
    assert DIPPER_RECURRENCE_CLASS_COLUMN not in source


def test_observed_dipper_recurrence_handles_missing_source_columns() -> None:
    out = add_observed_dipper_recurrence(pd.DataFrame({"candidate_id": ["a", "b"]}))

    assert out[DIPPER_RECURRENCE_CLASS_COLUMN].tolist() == ["unknown", "unknown"]
