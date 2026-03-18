from __future__ import annotations

from pathlib import Path

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


def test_asassn_tap_falls_back_to_local(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fail_tap(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("tap down")

    def fake_local(
        df: pd.DataFrame,
        valid,
        n_valid: int,
        radius_arcsec: float,
        local_csv: Path | str | None = None,
    ) -> pd.DataFrame:
        _ = (valid, n_valid, radius_arcsec)
        seen["local_csv"] = local_csv
        out = df.copy()
        out["asassn_var_name"] = ["ASASSN-V J000000.00+000000.0"]
        out["asassn_var_type"] = ["EA"]
        out["asassn_var_period"] = [1.23]
        return out

    monkeypatch.setattr(vetting, "batch_tap_crossmatch", fail_tap)
    monkeypatch.setattr(vetting, "_asassn_via_local", fake_local)

    local_csv = tmp_path / "asassn.csv"
    df = pd.DataFrame({"ra": [10.0], "dec": [-5.0]})
    out = vetting.crossmatch_asassn_variables(df, method="tap", local_csv=local_csv)

    assert seen["local_csv"] == local_csv
    assert out.loc[0, "asassn_var_name"] == "ASASSN-V J000000.00+000000.0"
    assert out.loc[0, "asassn_var_type"] == "EA"
    assert np.isclose(out.loc[0, "asassn_var_period"], 1.23)
