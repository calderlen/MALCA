from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


RUN_DIR = Path("output") / "runs" / "ltv_march18"


pytestmark = pytest.mark.skipif(
    not (RUN_DIR / "review" / "review.db").exists(),
    reason="migrated LTV March 18 bundle is not present",
)


def test_ltv_march18_review_db_counts() -> None:
    with sqlite3.connect(RUN_DIR / "review" / "review.db") as conn:
        assert conn.execute("select count(*) from candidates").fetchone()[0] == 868
        assert conn.execute("select count(*) from reviews").fetchone()[0] == 588


def test_ltv_march18_lightcurve_asset_count() -> None:
    lightcurves = RUN_DIR / "bundle_assets" / "lightcurves"
    assert len(list(lightcurves.glob("*.dat2"))) == 867


def test_ltv_march18_pipeline_outputs_remain_glob_discoverable() -> None:
    results = RUN_DIR / "results"
    pipeline_outputs = sorted(path.name for path in results.glob("*_pipeline.parquet"))

    assert pipeline_outputs == [
        "LTvar12-12.5_pipeline.parquet",
        "LTvar12.5-13_pipeline.parquet",
        "LTvar13-13.5_pipeline.parquet",
        "LTvar13.5-14_pipeline.parquet",
        "LTvar14-14.5_pipeline.parquet",
        "LTvar14.5-15_pipeline.parquet",
    ]


def test_ltv_march18_bundle_omits_intermediate_outputs() -> None:
    results = RUN_DIR / "results"

    assert not [path for path in results.glob("LTvar*.parquet") if path.is_dir()]
    assert not list(results.glob("LTvar*_PROCESSED.txt"))
    assert not list(results.glob("ltv_pca_model_*.joblib"))
    assert not (RUN_DIR / "artifacts").exists()
