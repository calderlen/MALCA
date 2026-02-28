import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from malca.vetting import (
    fetch_kepler_k2_lightcurves,
    fetch_aavso_lightcurves,
    fetch_panstarrs_lightcurves,
    fetch_crts_lightcurves
)

def get_mock_df():
    return pd.DataFrame({
        "candidate_id": ["cand_1", "cand_2"],
        "ra": [10.0, 20.0],
        "dec": [-10.0, -20.0],
        "best_name": ["V* TEST 1", "V* TEST 2"]
    })

def test_fetch_kepler_k2(mock_df, tmp_path):
    import sys
    mock_lk = MagicMock()
    sys.modules["lightkurve"] = mock_lk
    try:
        mock_search = MagicMock()
        mock_search.__len__.return_value = 1
        mock_lk.search_lightcurve.return_value = mock_search
        mock_lc1 = MagicMock()
        mock_lc1.time.value = np.array([1, 2, 3])
        mock_lc1.flux.value = np.array([100, 101, 100])
        mock_lc1.flux_err.value = np.array([1, 1, 1])
        mock_lc1.quality.value = np.array([0, 0, 0])
        mock_lc1.meta.QUARTER = 1
        
        mock_collection = MagicMock()
        mock_collection.__len__.return_value = 1
        mock_collection.__iter__.return_value = iter([mock_lc1])
        mock_search.download_all.return_value = mock_collection
        
        df_out = fetch_kepler_k2_lightcurves(mock_df, output_dir=tmp_path)
        
        assert "kepler_n_quarters" in df_out.columns
        assert df_out.loc[0, "kepler_n_quarters"] == 1
        assert df_out.loc[0, "kepler_total_points"] == 3
        
        assert (tmp_path / "kepler_lc_cand_1.parquet").exists()
    finally:
        del sys.modules["lightkurve"]

def test_fetch_aavso(mock_df, tmp_path):
    with patch("malca.vetting.requests.get") as mock_get, \
         patch("malca.vetting.pd.read_html") as mock_read_html:
        
        mock_response = MagicMock()
        mock_response.text = "<html><table>...</table></html>"
        mock_get.return_value = mock_response
        
        mock_df_html = pd.DataFrame({
            "JD": ["2459000.5", "2459001.5"],
            "Mag": ["10.0", "10.1"],
            "Err": ["0.1", "0.1"],
            "Filter": ["V", "B"],
            "Observer": ["XXX", "YYY"]
        })
        mock_read_html.return_value = [mock_df_html]
        
        df_out = fetch_aavso_lightcurves(mock_df, output_dir=tmp_path)
        
        assert "aavso_lc_n_points" in df_out.columns
        assert df_out.loc[0, "aavso_lc_n_points"] == 2
        assert (tmp_path / "aavso_lc_cand_1.parquet").exists()

def test_fetch_panstarrs(mock_df, tmp_path):
    with patch("malca.vetting.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "obsTime,filterID,psfFlux,psfFluxErr\n59000.0,1,0.0001,0.00001\n59001.0,2,0.0002,0.00002\n"
        mock_get.return_value = mock_response
        
        df_out = fetch_panstarrs_lightcurves(mock_df, output_dir=tmp_path)
        
        assert "ps1_lc_n_points" in df_out.columns
        assert df_out.loc[0, "ps1_lc_n_points"] == 2
        assert (tmp_path / "ps1_lc_cand_1.parquet").exists()

def test_fetch_crts(mock_df, tmp_path):
    try:
        import pyvo
    except ImportError:
        pytest.skip("pyvo not installed")

    with patch("malca.vetting.batch_tap_crossmatch") as mock_prss, \
         patch("malca.vetting.pyvo.dal.TAPService") as mock_tap:
        
        mock_prss.return_value = pd.DataFrame({
            "ID": ["111", "222"],
            "sep_arcsec": [1.0, 2.0],
            "_idx": [0, 1]
        })
        
        mock_tap_instance = MagicMock()
        mock_tap.return_value = mock_tap_instance
        
        mock_result = MagicMock()
        mock_result.to_table.return_value.to_pandas.return_value = pd.DataFrame({
            "ID": ["111", "111", "222"],
            "mjd": [55000.0, 55001.0, 56000.0],
            "mag": [15.0, 15.1, 14.5],
            "magerr": [0.05, 0.06, 0.04]
        })
        mock_tap_instance.search.return_value = mock_result
        
        df_out = fetch_crts_lightcurves(mock_df, output_dir=tmp_path)
        
        assert "crts_lc_n_points" in df_out.columns
        assert df_out.loc[0, "crts_lc_n_points"] == 2 
        assert df_out.loc[1, "crts_lc_n_points"] == 1
        assert (tmp_path / "crts_lc_cand_1.parquet").exists()

if __name__ == "__main__":
    import tempfile
    df = get_mock_df()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_fetch_kepler_k2(df, p)
        test_fetch_aavso(df, p)
        test_fetch_panstarrs(df, p)
        test_fetch_crts(df, p)
        print("All tests passed!")
