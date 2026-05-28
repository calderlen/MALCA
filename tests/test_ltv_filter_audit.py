from __future__ import annotations

import pandas as pd

from malca.ltv.filter import apply_all_filters_audit


def test_apply_all_filters_audit_preserves_rejected_rows() -> None:
    df = pd.DataFrame(
        {
            "asas_sn_id": ["pass", "low_slope", "low_diff", "south"],
            "ltv_slope": [0.1, 0.01, 0.1, 0.1],
            "ltv_max_diff": [0.5, 0.5, 0.1, 0.5],
            "dec": [0.0, 0.0, 0.0, -89.0],
            "ltv_median": [13.0, 13.0, 13.0, 13.0],
            "baseline_mag": [13.0, 13.0, 13.0, 13.0],
            "ltv_dispersion": [0.02, 0.02, 0.02, 0.02],
            "ltv_median_err": [0.02, 0.02, 0.02, 0.02],
            "pm_total": [0.0, 0.0, 0.0, 0.0],
            "neighbor_pm_contam": [False, False, False, False],
            "crowding_count": [0, 0, 0, 0],
        }
    )

    audit, passers = apply_all_filters_audit(
        df,
        min_slope=0.03,
        min_diff=0.3,
        min_dec=-80.0,
        query_gaia=False,
        return_passers=True,
    )

    assert len(audit) == 4
    assert passers["asas_sn_id"].tolist() == ["pass"]
    by_id = audit.set_index("asas_sn_id")
    assert bool(by_id.loc["low_slope", "ltv_failed_slope"]) is True
    assert bool(by_id.loc["low_diff", "ltv_failed_max_diff"]) is True
    assert bool(by_id.loc["south", "ltv_failed_dec"]) is True
    assert int(audit["failed_any"].sum()) == 3
