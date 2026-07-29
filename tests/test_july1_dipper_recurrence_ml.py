from __future__ import annotations

import pandas as pd

from malca.meta_analysis.ml.july1_review_training import _dipper_recurrence_labels


def test_dipper_recurrence_labels_use_human_morphology_not_run_count() -> None:
    table = pd.DataFrame(
        {
            "status": ["reviewed"] * 5,
            "workflow_status": ["reviewed"] * 5,
            "event_class": ["dipper", "dipper", "dipper", "dipper", "other"],
            "morphology_secondary": ["", "", "", "", "single_dip"],
            "morphology_secondary_json": [
                '["recurrent_dips"]',
                '["single_dip"]',
                '["single_dip", "recurrent_dips"]',
                "[]",
                '["single_dip"]',
            ],
            # These must not control the human training labels.
            "dip_run_count": [1, 5, 3, 8, 1],
        }
    )

    target, positive, audit = _dipper_recurrence_labels(table)

    assert positive == "recurrent_given_dipper"
    assert table.loc[0, target] == "recurrent_given_dipper"
    assert table.loc[1, target] == "non_recurrent_given_dipper"
    assert table.loc[2:, target].isna().all()
    assert audit["n_recurrent"] == 1
    assert audit["n_non_recurrent"] == 1
    assert audit["n_conflicting_recurrence_tags"] == 1
