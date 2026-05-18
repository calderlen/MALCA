from __future__ import annotations

import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id


def test_enrichment_coordinate_helper_accepts_gaia_ra_dec_columns() -> None:
    out = _ensure_candidate_id(
        pd.DataFrame(
            {
                "asas_sn_id": ["A"],
                "ra": ["12.3"],
                "dec": ["-45.6"],
            }
        )
    )

    assert out.loc[0, "candidate_id"] == "A"
    assert float(out.loc[0, "ra_deg"]) == 12.3
    assert float(out.loc[0, "dec_deg"]) == -45.6
