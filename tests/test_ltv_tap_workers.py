from __future__ import annotations

import pandas as pd

from malca.ltv import pipeline


def test_run_full_pipeline_uses_tap_workers_for_gaia_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_apply_all_filters(df: pd.DataFrame, **kwargs):
        captured.update(kwargs)
        return df.copy(), pd.DataFrame()

    monkeypatch.setattr(pipeline, "apply_all_filters", fake_apply_all_filters)

    df = pd.DataFrame({"asas_sn_id": ["candidate"], "ra": [10.0], "dec": [5.0]})
    out = pipeline.run_full_pipeline(
        df,
        run_crossmatch=False,
        run_neowise=False,
        run_extinction=False,
        run_dust_flags=False,
        run_gaia_epoch=False,
        run_bailer_jones=False,
        run_cmd=False,
        n_workers=10,
        tap_workers=1,
        verbose=False,
    )

    assert captured["n_workers"] == 1
    assert out["filter_reason"].tolist() == ["passed"]
