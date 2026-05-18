from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.evaluation.reproduce import load_candidates_df
from malca.review.store import load_candidates_file


def test_review_candidate_loader_accepts_14_14_5_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "lc_events_collect_candidates_14_14.5.csv"
    pd.DataFrame(
        {
            "path": ["/data/lcsv2/14_14.5/123.dat2"],
            "asas_sn_id": ["123"],
            "mag_bin": ["14_14.5"],
            "ra_deg": [10.5],
            "dec_deg": [-20.25],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_candidates_file(csv_path)

    assert loaded.loc[0, "candidate_id"] == "123"
    assert loaded.loc[0, "source_id"] == "123"
    assert loaded.loc[0, "lc_path"] == "/data/lcsv2/14_14.5/123.dat2"


def test_reproduce_candidate_loader_accepts_14_14_5_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "lc_events_collect_candidates_14_14.5.csv"
    pd.DataFrame(
        {
            "path": ["/data/lcsv2/14_14.5/456.dat2"],
            "asas_sn_id": ["456"],
            "mag_bin": ["14_14.5"],
            "ra_deg": [11.5],
            "dec_deg": [-21.25],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_candidates_df(csv_path)

    assert loaded.loc[0, "candidate_id"] == "456"
    assert loaded.loc[0, "source_id"] == "456"
    assert loaded.loc[0, "path"].endswith("456.dat2")
