from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

import malca.vetting as vetting


def test_query_simbad_batch_calls_xmatch(monkeypatch) -> None:
    def fake_xmatch(df: pd.DataFrame, valid, n: int, radius_arcsec: float) -> pd.DataFrame:
        _ = (valid, n, radius_arcsec)
        out = df.copy()
        out["simbad_main_id"] = ["SIMBAD 1"]
        out["simbad_otype"] = ["YSO"]
        out["simbad_nbref"] = [3]
        out["simbad_sep_arcsec"] = [0.4]
        return out

    monkeypatch.setattr(vetting, "_simbad_via_xmatch", fake_xmatch)

    df = pd.DataFrame({"ra": [10.0], "dec": [-5.0]})
    out = vetting.query_simbad_batch(df)

    assert out.loc[0, "simbad_main_id"] == "SIMBAD 1"
    assert out.loc[0, "simbad_otype"] == "YSO"
    assert out.loc[0, "simbad_nbref"] == 3


def test_fetch_microlensing_event_catalog_combines_sources(monkeypatch, tmp_path: Path) -> None:
    # We mock pd.read_csv to return dummy data for KMTNet and OGLE-EWS
    original_read_csv = pd.read_csv
    
    def mock_read_csv(filepath, *args, **kwargs):
        path_str = str(filepath)
        if "kmtnet" in path_str:
            return pd.DataFrame({
                "event": ["KMT-2024-BLG-0001"],
                "ra_deg": [270.0],
                "dec_deg": [-30.0],
                "t_e": [18.2],
                "classification": ["clear"],
            })
        elif "ogle" in path_str:
            return pd.DataFrame({
                "event": ["2024-BLG-0001"],
                "ra_deg": [270.0],
                "dec_deg": [-30.0],
                "tau": [28.5],
            })
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    # Also patch Path.exists to always return True for these local CSVs
    original_exists = Path.exists
    def mock_exists(self):
        if "kmtnet" in str(self) or "ogle" in str(self):
            return True
        return original_exists(self)
    
    monkeypatch.setattr(Path, "exists", mock_exists)

    out = vetting.fetch_microlensing_event_catalog()

    assert set(out["source"]) == {"OGLE-EWS", "KMTNet"}


def test_crossmatch_microlensing_catalogs_prefers_closest_match_and_keeps_secondary_aliases(monkeypatch) -> None:
    monkeypatch.setattr(
        vetting,
        "fetch_microlensing_event_catalog",
        lambda **kwargs: pd.DataFrame([
            {
                "source": "KMTNet", "event_id": "KMT-2024-BLG-0001", "alias": "OGLE-2024-BLG-0001",
                "ra": 270.0, "dec": -30.0, "timescale_days": 18.2, "timescale_kind": "te",
                "status": "clear", "source_url": "https://kmtnet", "event_year": 2024, "source_rank": 1,
            },
            {
                "source": "OGLE-EWS", "event_id": "OGLE-2024-BLG-0001", "alias": "2024-BLG-0001",
                "ra": 270.0002, "dec": -30.0002, "timescale_days": 28.5, "timescale_kind": "tau",
                "status": "", "source_url": "https://ogle", "event_year": 2024, "source_rank": 0,
            },
        ]),
    )

    df = pd.DataFrame({"ra": [270.0], "dec": [-30.0]})
    out = vetting.crossmatch_microlensing_catalogs(df, radius_arcsec=2.0)

    assert bool(out.loc[0, "microlens_match"])
    assert out.loc[0, "microlens_catalog"] == "KMTNet"
    assert out.loc[0, "microlens_name"] == "KMT-2024-BLG-0001"
    assert np.isclose(out.loc[0, "microlens_te_days"], 18.2)
    assert "OGLE-2024-BLG-0001" in out.loc[0, "microlens_alt_name"]


def test_print_vetting_summary_marks_microlens_matches_as_known() -> None:
    df = pd.DataFrame({
        "microlens_match": [True, False],
        "microlens_catalog": ["OGLE-IV", ""],
    })

    vetting._print_vetting_summary(df, time.perf_counter())

    assert bool(df.loc[0, "vetting_likely_known"])
    assert not bool(df.loc[1, "vetting_likely_known"])
