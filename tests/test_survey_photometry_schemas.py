from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.enrichment import characterize


class _FakeXMatchResult:
    def __init__(self, rows: list[dict[str, object]]):
        self._df = pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


def test_iphas_crossmatch_uses_lowercase_dr2_columns(monkeypatch) -> None:
    def fake_query(**_kwargs):
        return _FakeXMatchResult(
            [
                {
                    "_idx": 0,
                    "angDist": 0.4,
                    "r": 12.20,
                    "i": 11.87,
                    "ha": 12.04,
                    "rmi": 0.33,
                    "rmha": 0.16,
                    "rErr": 0.01,
                    "iErr": 0.02,
                    "haErr": 0.03,
                }
            ]
        )

    monkeypatch.setattr(characterize.XMatch, "query", fake_query)

    out = characterize.crossmatch_iphas(pd.DataFrame({"ra": [1.0], "dec": [2.0]}))

    assert float(out.loc[0, "iphas_r_mag"]) == 12.20
    assert float(out.loc[0, "iphas_i_mag"]) == 11.87
    assert float(out.loc[0, "iphas_ha_mag"]) == 12.04
    assert float(out.loc[0, "iphas_r_i"]) == 0.33
    assert float(out.loc[0, "iphas_r_ha"]) == 0.16
    assert float(out.loc[0, "iphas_sep_arcsec"]) == 0.4
    assert out.loc[0, "iphas_source_catalog"] == "II/321/iphas2"


def test_vphas_crossmatch_prefers_dr3_columns(monkeypatch) -> None:
    queried_catalogs: list[str] = []

    def fake_query(**kwargs):
        queried_catalogs.append(str(kwargs["cat2"]).removeprefix("vizier:"))
        return _FakeXMatchResult(
            [
                {
                    "_idx": 0,
                    "angDist": 0.5,
                    "rap3": 15.0,
                    "iap3": 14.4,
                    "Haap3": 14.7,
                    "r-ipnt": 0.6,
                    "r-Hapnt": 0.3,
                    "e_rap3": 0.02,
                    "e_iap3": 0.03,
                    "e_Haap3": 0.04,
                }
            ]
        )

    monkeypatch.setattr(characterize.XMatch, "query", fake_query)

    out = characterize.crossmatch_vphas(pd.DataFrame({"ra": [1.0], "dec": [2.0]}))

    assert queried_catalogs == ["II/386/vphasplus32"]
    assert float(out.loc[0, "vphas_r_mag"]) == 15.0
    assert float(out.loc[0, "vphas_i_mag"]) == 14.4
    assert float(out.loc[0, "vphas_ha_mag"]) == 14.7
    assert float(out.loc[0, "vphas_r_i"]) == 0.6
    assert float(out.loc[0, "vphas_r_ha"]) == 0.3
    assert out.loc[0, "vphas_source_catalog"] == "II/386/vphasplus32"


def test_vphas_crossmatch_falls_back_to_dr2_columns(monkeypatch) -> None:
    queried_catalogs: list[str] = []

    def fake_query(**kwargs):
        catalog = str(kwargs["cat2"]).removeprefix("vizier:")
        queried_catalogs.append(catalog)
        if catalog == "II/386/vphasplus32":
            return _FakeXMatchResult([])
        return _FakeXMatchResult(
            [
                {
                    "_idx": 0,
                    "angDist": 0.7,
                    "rmag": 16.0,
                    "imag": 15.3,
                    "Hamag": 15.5,
                    "r-i": 0.7,
                    "r-ha": 0.5,
                    "e_rmag": 0.02,
                    "e_imag": 0.03,
                    "e_Hamag": 0.04,
                }
            ]
        )

    monkeypatch.setattr(characterize.XMatch, "query", fake_query)

    out = characterize.crossmatch_vphas(pd.DataFrame({"ra": [1.0], "dec": [2.0]}))

    assert queried_catalogs == ["II/386/vphasplus32", "II/341/vphasp"]
    assert float(out.loc[0, "vphas_r_mag"]) == 16.0
    assert float(out.loc[0, "vphas_i_mag"]) == 15.3
    assert float(out.loc[0, "vphas_ha_mag"]) == 15.5
    assert float(out.loc[0, "vphas_r_i"]) == 0.7
    assert float(out.loc[0, "vphas_r_ha"]) == 0.5
    assert out.loc[0, "vphas_source_catalog"] == "II/341/vphasp"


def test_apass_crossmatch_uses_apostrophe_column_names(monkeypatch) -> None:
    def fake_query(**_kwargs):
        return _FakeXMatchResult(
            [
                {
                    "_idx": 0,
                    "angDist": 0.2,
                    "Bmag": 14.0,
                    "e_Bmag": 0.02,
                    "Vmag": 13.5,
                    "e_Vmag": 0.02,
                    "g'mag": 13.7,
                    "e_g'mag": 0.03,
                    "r'mag": 13.2,
                    "e_r'mag": 0.03,
                    "i'mag": 13.0,
                    "e_i'mag": 0.04,
                }
            ]
        )

    monkeypatch.setattr(characterize.XMatch, "query", fake_query)

    out = characterize.crossmatch_apass(pd.DataFrame({"ra": [1.0], "dec": [2.0]}))

    assert float(out.loc[0, "apass_g"]) == 13.7
    assert float(out.loc[0, "apass_g_err"]) == 0.03
    assert float(out.loc[0, "apass_r"]) == 13.2
    assert float(out.loc[0, "apass_i"]) == 13.0


def test_galex_crossmatch_uses_ais_fuv_nuv_columns(monkeypatch) -> None:
    def fake_query(**_kwargs):
        return _FakeXMatchResult(
            [
                {
                    "_idx": 0,
                    "angDist": 0.2,
                    "FUV": 19.1,
                    "e_FUV": 0.11,
                    "NUV": 18.4,
                    "e_NUV": 0.09,
                }
            ]
        )

    monkeypatch.setattr(characterize.XMatch, "query", fake_query)

    out = characterize.crossmatch_galex(pd.DataFrame({"ra": [1.0], "dec": [2.0]}))

    assert float(out.loc[0, "galex_fuv"]) == 19.1
    assert float(out.loc[0, "galex_fuv_err"]) == 0.11
    assert float(out.loc[0, "galex_nuv"]) == 18.4
    assert float(out.loc[0, "galex_nuv_err"]) == 0.09


def test_module_completed_requires_expected_output_columns() -> None:
    assert characterize._module_completed(pd.DataFrame({"char_status_iphas": ["ok"]}), "iphas") is False
    assert characterize._module_completed(pd.DataFrame({"char_status_iphas": ["skipped"]}), "iphas") is True

    frame = pd.DataFrame({"char_status_iphas": ["ok"]})
    for col in characterize.IPHAS_CACHE_COLUMNS:
        frame[col] = np.nan
    assert characterize._module_completed(frame, "iphas") is True


def test_characterization_cache_reruns_when_schema_is_stale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(characterize, "CHARACTERIZE_CACHE_DIR", tmp_path / "characterize")
    cache_path = characterize._characterize_cache_path("apass")
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "_cache_key": ["gaia:123"],
            "apass_v": [14.0],
            "_cache_status": ["ok"],
            "_cache_updated_at": ["2026-06-30T00:00:00+00:00"],
        }
    ).to_parquet(cache_path, index=False)

    calls = {"n": 0}

    def fake_apass(frame: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        out = frame.copy()
        out["apass_v"] = 14.2
        out["apass_v_err"] = 0.03
        return out

    out = characterize._run_cached_characterization_module(
        pd.DataFrame({"source_id": ["123"], "ra": [1.0], "dec": [2.0]}),
        module="apass",
        func=fake_apass,
        output_columns=["apass_v", "apass_v_err"],
    )

    assert calls == {"n": 1}
    assert float(out.loc[0, "apass_v"]) == 14.2
    assert float(out.loc[0, "apass_v_err"]) == 0.03
