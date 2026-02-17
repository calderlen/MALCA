from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

post_filter = pytest.importorskip("malca.post_filter")


def test_validate_gaia_proper_motion_flags_without_reject(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "path": ["a.csv", "b.csv"],
            "gaia_id": ["1001", "1002"],
        }
    )

    def fake_fetch(source_ids: list[int], show_tqdm: bool = False, **_kwargs) -> pd.DataFrame:
        _ = show_tqdm
        assert sorted(source_ids) == [1001, 1002]
        return pd.DataFrame(
            {
                "source_id": [1001, 1002],
                "ruwe": [1.0, 1.0],
                "pmra": [30.0, 120.0],
                "pmdec": [40.0, 160.0],
            }
        )

    monkeypatch.setattr(post_filter, "fetch_gaia_dr3_ruwe", fake_fetch)

    out = post_filter.validate_gaia_proper_motion(df, max_pm=100.0, flag_only=True)

    assert len(out) == 2
    assert list(out["high_pm_flag"].astype(bool)) == [False, True]
    assert np.isclose(float(out.loc[0, "pm_total"]), 50.0)
    assert np.isclose(float(out.loc[1, "pm_total"]), 200.0)


def test_validate_gaia_proper_motion_rejects_high_pm(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "path": ["a.csv", "b.csv"],
            "gaia_id": ["2001", "2002"],
        }
    )

    def fake_fetch(source_ids: list[int], show_tqdm: bool = False, **_kwargs) -> pd.DataFrame:
        _ = show_tqdm
        assert sorted(source_ids) == [2001, 2002]
        return pd.DataFrame(
            {
                "source_id": [2001, 2002],
                "ruwe": [1.0, 1.0],
                "pmra": [20.0, 90.0],
                "pmdec": [20.0, 80.0],
            }
        )

    monkeypatch.setattr(post_filter, "fetch_gaia_dr3_ruwe", fake_fetch)

    out = post_filter.validate_gaia_proper_motion(df, max_pm=100.0, flag_only=False)

    assert len(out) == 1
    assert out.iloc[0]["path"] == "a.csv"
    assert bool(out.iloc[0]["high_pm_flag"]) is False
