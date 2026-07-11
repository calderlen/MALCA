from __future__ import annotations

import numpy as np
import pandas as pd


NUCLEAR_SCORE_COLUMNS: dict[str, str] = {
    "agn": "agn_prior_score",
    "tde": "tde_candidate_score",
    "clagn": "clagn_photometric_score",
}


def _score(row: pd.Series, column: str) -> float:
    try:
        value = float(row.get(column, np.nan))
    except Exception:
        return 0.0
    return value if np.isfinite(value) else 0.0


def arbitrate_nuclear_scores(
    df: pd.DataFrame,
    *,
    min_score: float = 0.5,
    min_margin: float = 0.05,
) -> pd.DataFrame:
    """Choose the primary AGN/TDE/CLAGN hypothesis from nuclear score columns."""
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        ranked = sorted(
            (
                (hypothesis, _score(row, column))
                for hypothesis, column in NUCLEAR_SCORE_COLUMNS.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        best_hypothesis, best_score = ranked[0]
        runner_up_hypothesis, runner_up_score = ranked[1]
        margin = float(best_score - runner_up_score)

        if best_score < float(min_score):
            primary = "control"
            status = "control"
        elif margin < float(min_margin):
            primary = "ambiguous"
            status = "ambiguous"
        else:
            primary = best_hypothesis
            status = "classified"

        rows.append(
            {
                "nuclear_primary_hypothesis": primary,
                "nuclear_primary_score": float(best_score),
                "nuclear_best_score_hypothesis": best_hypothesis,
                "nuclear_best_score": float(best_score),
                "nuclear_runner_up_hypothesis": runner_up_hypothesis,
                "nuclear_runner_up_score": float(runner_up_score),
                "nuclear_hypothesis_margin": margin,
                "nuclear_hypothesis_status": status,
            }
        )

    arbitration = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for column in arbitration.columns:
        out[column] = arbitration[column]
    return out
