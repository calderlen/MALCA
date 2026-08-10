from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.evaluation.period_arbitration_audit import (
    annotate_external_period_groups,
    current_default_config_table,
    select_healthy_period_cohort,
    topk_catalog_recovery,
    utility_family_table,
)


def test_current_default_config_table_is_portable() -> None:
    table = current_default_config_table()
    values = table.set_index("parameter")["value"]
    assert values["pdm_method"] == "classic"
    assert values["max_scored_candidates"] == "128"
    assert table["value"].map(type).eq(str).all()


def test_balanced_cohort_separates_absent_from_conflicted_external(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(18):
        path = tmp_path / f"lc_{index}.dat3"
        path.touch()
        group = index % 3
        rows.append(
            {
                "candidate_id": f"source-{index:02d}",
                "lc_path": str(path),
                "median_mag": 12.1 + 0.15 * index,
                "n_points": 300 + 10 * index,
                "catalog_reference_available": group == 0,
                "catalog_reference_period": 2.0 if group == 0 else np.nan,
                "gaia_eb_period": 2.0 if group in {0, 1} else np.nan,
                "vsx_period": np.nan,
                "asassn_var_period": np.nan,
                "ztf_var_period": np.nan,
                "period_ogle_days": np.nan,
            }
        )
    frame = annotate_external_period_groups(pd.DataFrame(rows))
    assert set(frame["external_period_group"]) == {
        "external_reference",
        "no_external_period",
        "unclean_or_conflicting_external",
    }

    cohort = select_healthy_period_cohort(frame, n_per_group=3, seed=7)
    assert cohort.groupby("cohort_group").size().to_dict() == {
        "external_reference": 3,
        "no_external_period": 3,
    }
    assert not cohort["external_period_group"].eq(
        "unclean_or_conflicting_external"
    ).any()


def test_topk_recovery_and_utility_family_collapse() -> None:
    scores = pd.DataFrame(
        {
            "source_id": ["a", "a", "b", "b"],
            "candidate_id": ["a1", "a2", "b1", "b2"],
            "period_days": [1.0, 2.0, 3.0, 4.0],
            "baseline_rank": [1, 2, 1, 2],
            "baseline_score": [0.8, 0.7, 0.9, 0.6],
            "candidate_stage": ["shortlist"] * 4,
            "catalog_exact_match": [False, True, True, False],
            "catalog_family_match": [True, True, True, False],
            "baseline_score__proposal_normalized_score": [0.8, 0.2, 0.7, 0.1],
            "baseline_score__ls_power": [0.9, 0.3, 0.8, 0.2],
            "baseline_score__bls_power": [0.4, 0.7, 0.6, 0.1],
        }
    )
    recovery = topk_catalog_recovery(scores, top_k=(1, 2))
    assert recovery.loc[recovery["top_k"].eq(1), "exact_recovery"].iloc[0] == 0.5
    assert recovery.loc[recovery["top_k"].eq(2), "exact_recovery"].iloc[0] == 1.0

    utilities = utility_family_table(scores)
    assert np.isclose(utilities.loc[0, "proposal/support"], 0.8)
    assert np.isclose(utilities.loc[0, "LS/Fourier"], 0.9)
    assert np.isclose(utilities.loc[0, "BLS"], 0.4)
