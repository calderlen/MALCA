from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malca import vetting


class _FakeResponse:
    def __init__(self, text: str = "", *, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise vetting.requests.HTTPError(f"HTTP {self.status_code}")


class _FakeTapResult:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_table(self) -> "_FakeTapResult":
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


def test_new_external_lc_parsers_normalize_fixture_lightcurves() -> None:
    aavso = vetting._parse_aavso_vsx_response(
        """<?xml version="1.0" encoding="UTF-8"?>
        <VSXObject><Name>T CrB</Name><AUID>000-BBW-825</AUID><Data><![CDATA[
JD,mag,uncert,band,by,starName,mtype,obsID,fainterThan,obsType
2460000.5,10.2,0.03,V,ABC,T CRB,STD,123,0,CCD
2460001.5,<10.8,,Vis.,DEF,T CRB,,124,1,Visual
]]></Data><Count>2</Count></VSXObject>"""
    )
    assert aavso["mjd"].tolist() == [60000.0, 60001.0]
    assert aavso["mag"].tolist() == [10.2, 10.8]
    assert aavso["mag_err"].iloc[0] == pytest.approx(0.03)
    assert pd.isna(aavso["mag_err"].iloc[1])
    assert aavso["band"].tolist() == ["V", "Vis."]
    assert aavso["auid"].iloc[0] == "000-BBW-825"
    assert aavso["vsx_name"].iloc[0] == "T CrB"

    ogle = vetting._parse_ogle_dat(
        "2457000.5 15.1 0.02\n2457001.0 15.4 0.03\n",
        source_name="OGLE-BLG-RRLYR-00001",
        band="I",
    )
    assert list(ogle.columns) == ["mjd", "mag", "mag_err", "band", "ogle_name", "source_url"]
    assert ogle["band"].tolist() == ["I", "I"]
    assert ogle["mjd"].iloc[0] == 57000.0

    allwise = vetting._normalize_allwise_mep_table(
        pd.DataFrame(
            {
                "mjd_ep": [55400.0, 55401.0],
                "w1mpro_ep": [12.0, 12.3],
                "w1sigmpro_ep": [0.03, 0.04],
                "w4mpro_ep": [8.0, 8.2],
                "w4sigmpro_ep": [0.2, 0.25],
            }
        )
    )
    assert {"mjd", "w1mpro", "w1sigmpro", "w4mpro", "w4sigmpro"}.issubset(allwise.columns)
    assert allwise["w1mpro"].tolist() == [12.0, 12.3]

    vvvx = vetting._normalize_vvvx_virac_table(
        pd.DataFrame(
            {
                "mjdobs": [57000.0, 57001.0],
                "filter": ["K_s", "J"],
                "mag": [14.0, 15.0],
                "magerr": [0.05, 0.06],
                "sourceid": [123, 123],
            }
        )
    )
    assert vvvx[["mjd", "mag", "mag_err", "band"]].to_dict("list") == {
        "mjd": [57000.0, 57001.0],
        "mag": [14.0, 15.0],
        "mag_err": [0.05, 0.06],
        "band": ["ks", "j"],
    }
    vvvx_summary = vetting._summarize_vvvx_virac_lc(vvvx)
    assert vvvx_summary["vvvx_virac_n_epochs"] == 2
    assert "vvvx_virac_n_points" not in vvvx_summary
    assert pd.isna(vvvx_summary["vvvx_virac_ks_range"])

    stripe_summary = vetting._summarize_stripe82_lc(
        pd.DataFrame(
            {
                "mjd": [52000.0, 52001.0, 52002.0],
                "band": ["g", "g", "r"],
                "mag": [18.0, 18.5, 17.0],
                "mag_err": [0.02, 0.03, 0.02],
            }
        )
    )
    assert stripe_summary["stripe82_lc_n_points"] == 3
    assert stripe_summary["stripe82_lc_g_range"] == 0.5


def test_stripe82_link_candidates_keep_known_fallbacks_first(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html>
      <a href="S82variables.dat.gz">stale master</a>
      <a href="AllLCs.tar.gz">light curves</a>
    </html>
    """

    monkeypatch.setattr(vetting.requests, "get", lambda *_args, **_kwargs: _FakeResponse(html))

    master_urls, archive_urls = vetting._stripe82_link_candidates()

    assert master_urls[0] == vetting.STRIPE82_MASTER_FALLBACK_URLS[0]
    assert archive_urls[0] == vetting.STRIPE82_LC_ARCHIVE_FALLBACK_URLS[0]
    assert any(url.endswith("S82variables.dat.gz") for url in master_urls[1:])


def test_allwise_mep_query_uses_published_qi_fact_column(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    def fake_query_tap(query: str) -> _FakeTapResult:
        queries.append(query)
        if "FROM allwise_p3as_psd" in query:
            return _FakeTapResult(pd.DataFrame([{"source_id": "0000p000_ac51-000001", "cntr": 1}]))
        if "FROM allwise_p3as_mep" in query:
            lowered = query.lower()
            assert "qi_fact" in lowered
            assert "qual_frame" not in lowered
            return _FakeTapResult(
                pd.DataFrame(
                    {
                        "mjd": [55400.0],
                        "w1mpro_ep": [12.0],
                        "w1sigmpro_ep": [0.03],
                        "qi_fact": [1.0],
                    }
                )
            )
        raise AssertionError(f"unexpected AllWISE MEP query: {query}")

    monkeypatch.setattr(vetting.Irsa, "query_tap", staticmethod(fake_query_tap))

    out = vetting._query_allwise_mep_one(1.0, 1.0)

    assert out["w1mpro"].tolist() == [12.0]
    assert len(queries) == 2


def test_allwise_mep_fetch_records_fetched_no_data_and_remote_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        [
            {"candidate_id": "C1", "ra": 1.0, "dec": 1.0},
            {"candidate_id": "C2", "ra": 2.0, "dec": 2.0},
            {"candidate_id": "C3", "ra": 3.0, "dec": 3.0},
        ]
    )

    def fake_query(ra: float, dec: float, max_sep_arcsec: float = vetting.ALLWISE_MEP_MAX_SEP_ARCSEC) -> pd.DataFrame:
        if ra == 1.0:
            return pd.DataFrame(
                {
                    "mjd": [55400.0, 55401.0],
                    "w1mpro": [12.0, 12.5],
                    "w1sigmpro": [0.03, 0.04],
                    "w2mpro": [11.5, 11.4],
                    "w2sigmpro": [0.03, 0.04],
                }
            )
        if ra == 2.0:
            return pd.DataFrame()
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(vetting, "_query_allwise_mep_one", fake_query)

    with pytest.raises(RuntimeError, match="AllWISE MEP LCs"):
        vetting.fetch_allwise_mep_lightcurves(df, output_dir=tmp_path, workers=1)

    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE).sort_values("candidate_id").reset_index(drop=True)
    assert status["status"].tolist() == ["fetched", "no_data", "error"]
    assert int(status.loc[0, "allwise_mep_n_epochs"]) == 2
    assert int(status.loc[1, "allwise_mep_n_epochs"]) == 0
    assert "synthetic outage" in status.loc[2, "error_message"]
    assert (tmp_path / "allwise_mep_lc_C1.parquet").exists()


def test_allwise_mep_fetch_reuses_cached_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 1.0}])
    lc = pd.DataFrame(
        {
            "mjd": [55400.0, 55401.0],
            "w1mpro": [12.0, 12.4],
            "w1sigmpro": [0.03, 0.04],
        }
    )
    lc.to_parquet(tmp_path / "allwise_mep_lc_C1.parquet", index=False)
    cache_key = vetting._coord_lookup_cache_key(df, 0, vetting.ALLWISE_MEP_MAX_SEP_ARCSEC, "allwise_mep")
    pd.DataFrame(
        [
            {
                "module": "AllWISE MEP LCs",
                "candidate_id": "C1",
                "cache_key": cache_key,
                "status": "fetched",
                "allwise_mep_n_epochs": 2,
            }
        ]
    ).to_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE, index=False)

    def fail_query(*_args, **_kwargs):
        raise AssertionError("cache hit should avoid live query")

    monkeypatch.setattr(vetting, "_query_allwise_mep_one", fail_query)

    out = vetting.fetch_allwise_mep_lightcurves(df, output_dir=tmp_path, workers=1)

    assert int(out.loc[0, "allwise_mep_n_epochs"]) == 2
    assert out.loc[0, "allwise_mep_w1_range"] == pytest.approx(0.4)


def test_vvvx_virac_query_uses_published_de_column_and_sourceid(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    class FakeTapService:
        def __init__(self, url: str) -> None:
            assert url == vetting.ESO_TAP_CAT_URL

        def search(self, query: str) -> _FakeTapResult:
            queries.append(query)
            assert "POINT('ICRS', ra, dec)" not in query
            if "VVVX_VIRAC_V2_SOURCES" in query:
                assert "POINT('ICRS', ra, de)" in query
                return _FakeTapResult(pd.DataFrame([{"sourceid": 123, "ra": 10.0, "de": -1.0}]))
            if "WHERE sourceid = 123" in query:
                return _FakeTapResult(
                    pd.DataFrame(
                        {
                            "sourceid": [123],
                            "mjdobs": [57000.0],
                            "filter": ["Ks"],
                            "mag": [14.0],
                            "emag": [0.04],
                        }
                    )
                )
            raise AssertionError(f"unexpected VIRAC2 query: {query}")

    monkeypatch.setattr(vetting.pyvo.dal, "TAPService", FakeTapService)

    lc = vetting._query_vvvx_virac_one(10.0, -1.0)

    assert lc[["mjd", "mag", "mag_err", "band"]].to_dict("records") == [
        {"mjd": 57000.0, "mag": 14.0, "mag_err": 0.04, "band": "ks"}
    ]
    assert len(queries) == 2


def test_vvvx_virac_empty_lightcurve_is_no_data_not_bad_alias_error(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    class FakeTapService:
        def __init__(self, _url: str) -> None:
            return None

        def search(self, query: str) -> _FakeTapResult:
            queries.append(query)
            assert "source_id" not in query
            assert "virac_id" not in query
            assert "srcid" not in query
            assert "POINT('ICRS', ra, dec)" not in query
            if "VVVX_VIRAC_V2_SOURCES" in query:
                return _FakeTapResult(pd.DataFrame([{"sourceid": 456, "ra": 11.0, "de": -2.0}]))
            if "WHERE sourceid = 456" in query:
                return _FakeTapResult(pd.DataFrame())
            assert "POINT('ICRS', ra, de)" in query
            return _FakeTapResult(pd.DataFrame())

    monkeypatch.setattr(vetting.pyvo.dal, "TAPService", FakeTapService)

    lc = vetting._query_vvvx_virac_one(11.0, -2.0)

    assert lc.empty
    assert len(queries) == 3


def test_aavso_fetch_uses_vsx_api_names_not_candidate_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        [
            {"candidate_id": "123456789"},
            {"candidate_id": "C2", "asassn_var_name": "T CrB", "jd_first": 2459000.0, "jd_last": 2459010.0},
        ]
    )
    calls: list[tuple[str, float, float]] = []

    def fake_query(identifier: str, from_jd: float, to_jd: float, max_points: int = vetting.AAVSO_MAX_POINTS) -> pd.DataFrame:
        calls.append((identifier, from_jd, to_jd))
        return pd.DataFrame(
            {
                "mjd": [59000.0, 59001.0],
                "mag": [10.0, 10.3],
                "mag_err": [0.02, 0.03],
                "band": ["V", "V"],
            }
        )

    monkeypatch.setattr(vetting, "_query_aavso_vsx_lightcurve", fake_query)

    out = vetting.fetch_aavso_lightcurves(df, output_dir=tmp_path, workers=1)

    assert calls == [("T CrB", pytest.approx(2458635.0), pytest.approx(2459375.0))]
    assert int(out.loc[0, "aavso_lc_n_points"]) == 0
    assert int(out.loc[1, "aavso_lc_n_points"]) == 2
    assert (tmp_path / "aavso_lc_C2.parquet").exists()


def test_aavso_vsx_405_and_human_verification_are_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, **_kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse("Not Allowed", status_code=405)
        return _FakeResponse("<html><title>Human Verification</title><script>AwsWaf</script></html>")

    monkeypatch.setattr(vetting.requests, "get", fake_get)

    lc = vetting._query_aavso_vsx_lightcurve("J194916.15+252549.7", 2456000.5, 2461198.8)

    assert lc.empty
    assert calls == list(vetting.AAVSO_VSX_API_URLS)
