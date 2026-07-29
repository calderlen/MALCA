from __future__ import annotations

import pandas as pd

from scripts.attach_july1_dipper_recurrence_ml import build_recurrence_overlays


def test_build_recurrence_overlays_gates_each_parent_dipper_score() -> None:
    recurrence = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "prob_recurrent_given_dipper": [0.8, 0.3],
        }
    )
    parents = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "prob_dipper_like": [0.5, 0.9],
            "prob_dipper": [0.25, 0.4],
        }
    )

    binary, eight_class = build_recurrence_overlays(recurrence, parents)

    assert binary["predicted_dipper_recurrence"].tolist() == ["recurrent", "non_recurrent"]
    assert binary["prob_recurrent_dipper_binary"].tolist() == [0.4, 0.27]
    assert eight_class["prob_recurrent_dipper_eight_class"].tolist() == [0.2, 0.12]
