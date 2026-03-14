from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from astroquery.ipac.irsa import Irsa
from astropy.table import Table

from malca.ltv import pipeline as ltv_pipeline
from malca.ltv import neowise as ltv_neowise

@pytest.fixture
def mock_irsa_query_region(monkeypatch):
    mock_query = MagicMock()
    monkeypatch.setattr(Irsa, "query_region", mock_query)
    return mock_query

@pytest.fixture
def mock_read_parquet(monkeypatch):
    mock_read = MagicMock()
    monkeypatch.setattr(pd, "read_parquet", mock_read)
    return mock_read

def _write_mock_dat2(path: Path) -> None:
    lines = [
        "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
        "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
        "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

def test_run_full_pipeline_with_neowise_and_filters(
    mock_irsa_query_region,
    mock_read_parquet,
    monkeypatch,
    tmp_path: Path,
) -> None:
    # 1. Mock NEOWISE results
    # For simplicity, we only test one target here to verify pipeline flow.
    # The actual bulk query logic is tested in tests/test_neowise_bulk.py
    ra, dec = 10.0, 20.0
    # The new implementation uses 'cntr', 'dist', 'w1mpro', 'w1sigmpro', 'mjd'
    neowise_results = Table({
        'ra': [ra], 'dec': [dec], 'mjd': [56000.0],
        'w1mpro': [15.0], 'w1sigmpro': [0.1],
        'w2mpro': [14.0], 'w2sigmpro': [0.1],
        # The new implementation uses 'cntr' and 'dist' from crossmatch
        'cntr': [123], 'dist': [0.1],
    })
    mock_irsa_query_region.return_value = neowise_results

    # 2. Mock light curve loading (stochastic postfilter depends on it)
    dat2_path = tmp_path / "123.dat2"
    _write_mock_dat2(dat2_path)
    
    # stochastic postfilter features are NAN in the test case where lc_path is not found or empty
    # For now, let's just make sure it passes the filter checks and proceeds to neowise

    # 3. Construct input DataFrame
    # Slope and max diff must exceed LTV_MIN_SLOPE (0.03) and LTV_MIN_DIFF (0.3) to pass filters.
    df = pd.DataFrame({
        "ra": [ra],
        "dec": [dec],
        "ra_deg": [ra],
        "dec_deg": [dec],
        "lc_path": [str(dat2_path)],
        "Slope": [0.05],
        "max diff": [0.4],
    })

    # Mock stochastic stage so it doesn't try to load the light curves which might fail
    # if paths are not correct.
    monkeypatch.setattr(ltv_pipeline, "add_stochastic_postfilter_features", lambda df, **kwargs: df)

    # 4. Run pipeline
    out = ltv_pipeline.run_full_pipeline(
        df,
        run_filters=True, # Critical
        run_neowise=True, # Critical
        run_stochastic_postfilter=False, # We mock it out
        run_crossmatch=False,
        run_extinction=False,
        run_dust_flags=False,
        run_cmd=False,
        run_bailer_jones=False,
        run_gaia_epoch=False,
        verbose=False,
    )

    # 5. Verify results
    assert "filter_reason" in out.columns
    # NEOWISE stage adds these columns (bulk implementation)
    assert "neowise_n_epochs" in out.columns
    assert "w1_slope" in out.columns

