from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.io import fetch as skypatrol_fetch
from malca.products.feature_layers import feature_mapping_get
from malca.review import fetch as review_fetch
from malca.enrichment import vetting


def _write_skypatrol_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "JD": [2450000.5, 2450001.5],
            "Flux": [1.0, 1.1],
            "Flux Error": [0.1, 0.1],
            "Mag": [14.0, 13.9],
            "Mag Error": [0.02, 0.02],
            "Limit": [pd.NA, pd.NA],
            "FWHM": [2.0, 2.1],
            "Filter": ["g", "g"],
            "Quality": ["G", "G"],
            "Camera": ["bi", "bi"],
        }
    ).to_csv(path, index=False)


def test_skypatrol_download_by_id_uses_valid_cached_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = tmp_path / "123.csv"
    _write_skypatrol_csv(cached)
    (tmp_path / "123.csv.meta.json").write_text('{"asas_sn_id": 123, "ra_deg": 1.5}', encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise AssertionError("SkyPatrol backend should not be queried on cache hit")

    monkeypatch.setattr(skypatrol_fetch, "_sp2_query_catalog_info", fail)
    monkeypatch.setattr(skypatrol_fetch, "_sp2_download_lc", fail)

    out, meta = skypatrol_fetch.download_lightcurve_by_id("123", cache_dir=tmp_path, backend="skypatrol2")

    assert out == cached.resolve()
    assert meta["asas_sn_id"] == 123
    assert meta["ra_deg"] == 1.5


def test_skypatrol_refresh_cache_forces_backend_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = tmp_path / "123.csv"
    _write_skypatrol_csv(cached)
    calls = {"download": 0}

    monkeypatch.setattr(
        skypatrol_fetch,
        "_sp2_query_catalog_info",
        lambda *_args, **_kwargs: {"asas_sn_id": 123, "ra_deg": 3.0},
    )

    def fake_download(*_args, **_kwargs):
        calls["download"] += 1
        return pd.DataFrame({"jd": [2450101.5, 2450100.5], "mag": [13.0, 13.2], "mag_err": [0.02, 0.02]})

    monkeypatch.setattr(skypatrol_fetch, "_sp2_download_lc", fake_download)

    out, meta = skypatrol_fetch.download_lightcurve_by_id(
        "123",
        cache_dir=tmp_path,
        backend="skypatrol2",
        refresh_cache=True,
    )

    refreshed = pd.read_csv(out)
    assert calls["download"] == 1
    assert meta["ra_deg"] == 3.0
    assert refreshed["JD"].tolist() == [2450100.5, 2450101.5]


def test_review_stats_cache_reuses_and_invalidates_by_lc_file_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lc_path = tmp_path / "CACHED.csv"
    _write_skypatrol_csv(lc_path)
    calls = {"stats": 0}

    import malca.core.stats as stats_mod

    def fake_compute_stats(_candidate_id: str, _parent: str, *, compute_ls: bool = True):
        calls["stats"] += 1
        return pd.DataFrame(), {"photometry_mean_mag": 14.0 + calls["stats"]}

    monkeypatch.setattr(stats_mod, "compute_stats", fake_compute_stats)

    first = review_fetch._compute_stats_from_skypatrol_csv(lc_path)
    second = review_fetch._compute_stats_from_skypatrol_csv(lc_path)

    assert calls["stats"] == 1
    assert first == second
    assert feature_mapping_get(first, "stats_photometry_mean_mag") == 15.0

    with lc_path.open("a", encoding="utf-8") as f:
        f.write("2450002.5,1.2,0.1,13.8,0.02,,2.0,g,G,bi\n")

    third = review_fetch._compute_stats_from_skypatrol_csv(lc_path)

    assert calls["stats"] == 2
    assert feature_mapping_get(third, "stats_photometry_mean_mag") == 16.0


def test_fetch_ztf_lightcurves_reuses_cached_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0, 59003.0],
            "mag": [15.0, 14.0, 16.0, 13.0],
            "band": ["zg", "zg", "zr", "zr"],
        }
    ).to_parquet(tmp_path / "ztf_lc_C1.parquet", index=False)

    def fail(*_args, **_kwargs):
        raise AssertionError("IRSA ZTF API should not be called on cache hit")

    monkeypatch.setattr(vetting.requests, "get", fail)

    out = vetting.fetch_ztf_lightcurves(df, output_dir=tmp_path)

    assert int(out.loc[0, "ztf_lc_n_det"]) == 4
    assert out.loc[0, "ztf_lc_g_range"] == 1.0
    assert out.loc[0, "ztf_lc_r_range"] == 3.0


def test_fetch_ztf_lightcurves_corrupt_cache_falls_back_to_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    (tmp_path / "ztf_lc_C1.parquet").write_text("not parquet", encoding="ascii")

    calls: list[dict] = []

    class FakeResponse:
        text = (
            "oid,hjd,hmjd,mag,MAG,magerr,filtercode,FILTER,catflags\n"
            "123,2450000.5,,15.0,,0.1,1,g,0\n"
            "123,2450001.5,,14.0,,0.1,1,g,0\n"
        )

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(vetting.requests, "get", fake_get)

    out = vetting.fetch_ztf_lightcurves(df, output_dir=tmp_path, workers=1)

    assert len(calls) == 1
    assert calls[0]["url"] == vetting.ZTF_LC_API_URL
    assert calls[0]["params"]["COLLECTION"] == "ztf_dr22"
    assert calls[0]["params"]["FORMAT"] == "CSV"
    assert int(out.loc[0, "ztf_lc_n_det"]) == 2
    assert out.loc[0, "ztf_lc_g_range"] == 1.0


def test_fetch_ztf_lightcurves_reuses_no_data_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    cache_key = vetting._coord_lookup_cache_key(df, 0, 2.0, "ztf_dr22")
    pd.DataFrame(
        [
            {
                "module": "ZTF LCs",
                "candidate_id": "C1",
                "cache_key": cache_key,
                "status": "no_data",
                "ztf_lc_n_det": 0,
                "ztf_lc_g_range": pd.NA,
                "ztf_lc_r_range": pd.NA,
            }
        ]
    ).to_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE, index=False)

    def fail(*_args, **_kwargs):
        raise AssertionError("IRSA ZTF API should not be called on no-data cache hit")

    monkeypatch.setattr(vetting.requests, "get", fail)

    out = vetting.fetch_ztf_lightcurves(df, output_dir=tmp_path)

    assert int(out.loc[0, "ztf_lc_n_det"]) == 0


def test_fetch_panstarrs_lightcurves_retries_http_429(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 10.0, "dec": 20.0}])
    csv_text = "obsTime,filterID,psfFlux,psfFluxErr,infoFlag\n2459000.5,1,1000.0,10.0,0\n"
    calls: list[tuple[str, dict]] = []
    sleeps: list[float] = []

    class FakeResponse:
        def __init__(self, status_code: int, text: str = "", headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

    responses = [
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(200, csv_text),
    ]

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(vetting.requests, "get", fake_get)
    monkeypatch.setattr(vetting.time, "sleep", lambda delay: sleeps.append(delay))

    out = vetting.fetch_panstarrs_lightcurves(df, output_dir=tmp_path, workers=1)

    assert len(calls) == 2
    assert calls[0][1]["timeout"] == vetting.VETTING_HTTP_TIMEOUT
    assert sleeps == [0.0]
    assert int(out.loc[0, "ps1_lc_n_points"]) == 1
    saved = pd.read_parquet(tmp_path / "ps1_lc_C1.parquet")
    assert saved.loc[0, "mjd"] == pytest.approx(59000.0)
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert status.loc[0, "status"] == "fetched"


def test_fetch_external_lcs_continues_after_module_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    messages: list[str] = []

    def fail_ztf(_df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        raise RuntimeError("synthetic ZTF outage")

    def fake_tess(in_df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        out = in_df.copy()
        out["tess_n_sectors"] = 1
        out["tess_total_points"] = 12
        out["tess_flux_range"] = 0.1
        return out

    monkeypatch.setattr(vetting, "fetch_ztf_lightcurves", fail_ztf)
    monkeypatch.setattr(vetting, "fetch_tess_lightcurves", fake_tess)

    out = vetting.fetch_external_lcs(
        df,
        output_dir=tmp_path,
        run_atlas=False,
        run_ztf=True,
        run_gaia_epoch=False,
        run_tess=True,
        run_neowise=False,
        run_kepler=False,
        run_aavso=False,
        run_ogle=False,
        run_stripe82=False,
        run_allwise_mep=False,
        run_vvvx_virac=False,
        run_ps1=False,
        run_crts=False,
        progress_callback=messages.append,
    )

    assert int(out.loc[0, "tess_n_sectors"]) == 1
    assert out.attrs["external_lc_failures"] == ["ZTF LCs failed: synthetic ZTF outage"]
    assert any("ZTF LCs failed" in message for message in messages)
    assert any("TESS LCs completed" in message for message in messages)


def test_fetch_external_lcs_module_failure_survives_closed_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])

    def fail_ztf(_df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        raise RuntimeError("synthetic ZTF outage")

    monkeypatch.setattr(vetting, "fetch_ztf_lightcurves", fail_ztf)
    closed_stdout = io.StringIO()
    closed_stdout.close()
    fallback_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", closed_stdout)
    monkeypatch.setattr(sys, "__stderr__", fallback_stderr)

    out = vetting.fetch_external_lcs(
        df,
        output_dir=tmp_path,
        run_atlas=False,
        run_ztf=True,
        run_gaia_epoch=False,
        run_tess=False,
        run_neowise=False,
        run_kepler=False,
        run_aavso=False,
        run_ogle=False,
        run_stripe82=False,
        run_allwise_mep=False,
        run_vvvx_virac=False,
        run_ps1=False,
        run_crts=False,
    )

    assert out.attrs["external_lc_failures"] == ["ZTF LCs failed: synthetic ZTF outage"]
    assert "ZTF LCs failed: synthetic ZTF outage" in fallback_stderr.getvalue()


def test_fetch_external_lcs_module_print_survives_closed_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])

    def fake_ztf(_df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        print("synthetic ZTF summary")
        out = _df.copy()
        out["ztf_lc_n_det"] = 1
        out["ztf_lc_g_range"] = float("nan")
        out["ztf_lc_r_range"] = float("nan")
        return out

    monkeypatch.setattr(vetting, "fetch_ztf_lightcurves", fake_ztf)
    closed_stdout = io.StringIO()
    closed_stdout.close()
    fallback_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", closed_stdout)
    monkeypatch.setattr(sys, "__stderr__", fallback_stderr)

    out = vetting.fetch_external_lcs(
        df,
        output_dir=tmp_path,
        run_atlas=False,
        run_ztf=True,
        run_gaia_epoch=False,
        run_tess=False,
        run_neowise=False,
        run_kepler=False,
        run_aavso=False,
        run_ogle=False,
        run_stripe82=False,
        run_allwise_mep=False,
        run_vvvx_virac=False,
        run_ps1=False,
        run_crts=False,
    )

    assert int(out.loc[0, "ztf_lc_n_det"]) == 1
    assert "external_lc_failures" not in out.attrs
    assert "synthetic ZTF summary" in fallback_stderr.getvalue()


def test_fetch_external_lcs_resume_retries_error_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    checkpoint_path = tmp_path / "external_lcs_CHECKPOINT.parquet"
    checkpoint = df.copy()
    checkpoint["ztf_lc_n_det"] = 12
    checkpoint.to_parquet(checkpoint_path, index=False)
    pd.DataFrame(
        [
            {
                "module": "ZTF LCs",
                "candidate_id": "C1",
                "cache_key": "ztf-error-key",
                "status": "error",
            }
        ]
    ).to_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE, index=False)
    calls: list[bool] = []

    def fake_ztf(in_df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        calls.append(True)
        return in_df

    monkeypatch.setattr(vetting, "fetch_ztf_lightcurves", fake_ztf)

    vetting.fetch_external_lcs(
        df,
        output_dir=tmp_path,
        run_atlas=False,
        run_ztf=True,
        run_gaia_epoch=False,
        run_tess=False,
        run_neowise=False,
        run_kepler=False,
        run_aavso=False,
        run_ogle=False,
        run_stripe82=False,
        run_allwise_mep=False,
        run_vvvx_virac=False,
        run_ps1=False,
        run_crts=False,
        checkpoint_path=checkpoint_path,
        progress_callback=lambda _msg: None,
    )

    assert calls == [True]


def test_fetch_tess_lightcurves_records_lookup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        [
            {"candidate_id": "C1", "ra": 1.0, "dec": 2.0},
            {"candidate_id": "C2", "ra": 3.0, "dec": 4.0},
        ]
    )

    def fake_search_lightcurve(coord, **_kwargs):
        if float(coord.ra.deg) == 1.0:
            raise RuntimeError("mast stream blew up")
        return []

    monkeypatch.setattr(vetting.lk, "search_lightcurve", fake_search_lightcurve)

    out = vetting.fetch_tess_lightcurves(df, output_dir=tmp_path, workers=1)

    assert out["tess_n_sectors"].tolist() == [0, 0]
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    error_row = status[status["candidate_id"] == "C1"].iloc[0]
    no_data_row = status[status["candidate_id"] == "C2"].iloc[0]
    assert error_row["status"] == "error"
    assert "mast stream blew up" in error_row["error_message"]
    assert no_data_row["status"] == "no_data"


class _FakeGaiaEpochTapTable:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __len__(self) -> int:
        return len(self._df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


class _FakeGaiaEpochTapResult:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_table(self) -> _FakeGaiaEpochTapTable:
        return _FakeGaiaEpochTapTable(self._df)


class _FakeGaiaEpochRawTapResult:
    def __init__(self, table: object) -> None:
        self._table = table

    def to_table(self) -> object:
        return self._table


class _FakeMaskedGaiaEpochTapTable:
    colnames = ["source_id", "transit_id", "time", "mag", "band"]

    def __init__(self, source_id: int) -> None:
        self._data = {
            "source_id": np.ma.array([source_id, source_id], mask=[False, False]),
            "transit_id": np.ma.array([301, 0], mask=[False, True]),
            "time": np.ma.array([1800.0, 1801.0], mask=[False, False]),
            "mag": np.ma.array([13.0, 13.4], mask=[False, False]),
            "band": np.array(["G", "G"]),
        }

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def to_pandas(self) -> pd.DataFrame:
        raise AssertionError("Gaia epoch fetcher should not call table.to_pandas() directly")


def test_fetch_gaia_epoch_lcs_expands_array_valued_tap_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "2192699590525791232"
    df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "gaia_id": source_id,
                "gaia_epoch_available": True,
            }
        ]
    )
    tap_df = pd.DataFrame(
        {
            "source_id": [int(source_id)],
            "transit_id": [np.ma.array([101, 102, 103], mask=[False, False, False])],
            "time": [np.ma.array([1688.1, 1688.3, 1688.5], mask=[False, False, False])],
            "mag": [np.ma.array([12.7, 12.5, 13.1], mask=[False, False, False])],
            "band": ["G"],
        }
    )

    monkeypatch.setattr(vetting, "_connect_gaia_taps_until_available", lambda *args, **kwargs: ["fake-tap"])
    monkeypatch.setattr(
        vetting,
        "_run_gaia_tap_query_until_success",
        lambda *_args, **_kwargs: _FakeGaiaEpochTapResult(tap_df),
    )

    out = vetting.fetch_gaia_epoch_lcs(df, output_dir=tmp_path)

    assert int(out.loc[0, "gaia_epoch_lc_n_g"]) == 3
    assert float(out.loc[0, "gaia_epoch_lc_g_range"]) == pytest.approx(0.6)
    saved = pd.read_parquet(tmp_path / "gaia_epoch_lc_C1.parquet")
    assert saved["time"].tolist() == [1688.1, 1688.3, 1688.5]
    assert saved["mag"].tolist() == [12.7, 12.5, 13.1]
    assert saved["band"].tolist() == ["G", "G", "G"]
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert status.loc[0, "status"] == "fetched"
    assert int(status.loc[0, "gaia_epoch_lc_n_g"]) == 3


def test_fetch_gaia_epoch_lcs_accepts_scalar_tap_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "427299085038300928"
    df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "gaia_id": source_id,
                "gaia_epoch_available": True,
            }
        ]
    )
    tap_df = pd.DataFrame(
        {
            "source_id": [int(source_id), int(source_id)],
            "transit_id": [201, 202],
            "time": [1700.0, 1701.0],
            "mag": [11.9, 12.2],
            "band": ["G", "G"],
            "mag_error": [0.01, 0.02],
        }
    )

    monkeypatch.setattr(vetting, "_connect_gaia_taps_until_available", lambda *args, **kwargs: ["fake-tap"])
    monkeypatch.setattr(
        vetting,
        "_run_gaia_tap_query_until_success",
        lambda *_args, **_kwargs: _FakeGaiaEpochTapResult(tap_df),
    )

    out = vetting.fetch_gaia_epoch_lcs(df, output_dir=tmp_path)

    assert int(out.loc[0, "gaia_epoch_lc_n_g"]) == 2
    assert float(out.loc[0, "gaia_epoch_lc_g_range"]) == pytest.approx(0.3)
    saved = pd.read_parquet(tmp_path / "gaia_epoch_lc_C1.parquet")
    assert saved["transit_id"].tolist() == [201, 202]
    assert saved["mag_error"].tolist() == [0.01, 0.02]


def test_fetch_gaia_epoch_lcs_handles_masked_integer_tap_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "427299085038300928"
    df = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "gaia_id": source_id,
                "gaia_epoch_available": True,
            }
        ]
    )

    monkeypatch.setattr(vetting, "_connect_gaia_taps_until_available", lambda *args, **kwargs: ["fake-tap"])
    monkeypatch.setattr(
        vetting,
        "_run_gaia_tap_query_until_success",
        lambda *_args, **_kwargs: _FakeGaiaEpochRawTapResult(_FakeMaskedGaiaEpochTapTable(int(source_id))),
    )

    out = vetting.fetch_gaia_epoch_lcs(df, output_dir=tmp_path)

    assert int(out.loc[0, "gaia_epoch_lc_n_g"]) == 2
    assert float(out.loc[0, "gaia_epoch_lc_g_range"]) == pytest.approx(0.4)
    saved = pd.read_parquet(tmp_path / "gaia_epoch_lc_C1.parquet")
    assert saved["time"].tolist() == [1800.0, 1801.0]
    assert saved["mag"].tolist() == [13.0, 13.4]
    assert pd.isna(saved.loc[1, "transit_id"])


class _FakeCRTSResponse:
    def __init__(self, text: str, *, status_code: int = 200, url: str = "http://nunuku.caltech.edu/cgi-bin/getcssconedb_priv.cgi") -> None:
        self.text = text
        self.status_code = status_code
        self.url = url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise vetting.requests.HTTPError(f"HTTP {self.status_code}")


def _crts_csv(rows: list[tuple[object, float, float, float, float, float, int]]) -> str:
    body = ["MasterID,Mag,Magerr,RA,Dec,MJD,Blend"]
    body.extend(",".join(str(value) for value in row) for row in rows)
    return "\n".join(body)


def test_crts_cgi_html_link_and_csv_normalization() -> None:
    html = '<html><a href="/DataRelease/upload/result_web_file.csv">Download CSV</a></html>'

    assert vetting._extract_crts_csv_url(html, vetting.CRTS_CGI_URL).endswith(
        "/DataRelease/upload/result_web_file.csv"
    )

    raw = pd.DataFrame(
        {
            "MasterID": ["near", "near", "far"],
            "Mag": [14.0, 14.2, 16.0],
            "Magerr": [0.05, 0.06, 0.08],
            "RA": [10.0, 10.0, 10.001],
            "Dec": [20.0, 20.0, 20.0],
            "MJD": [56000.0, 56001.0, 56000.5],
            "Blend": [0, 0, 0],
        }
    )

    lc = vetting._normalize_crts_cgi_lightcurve(
        raw,
        ra=10.0,
        dec=20.0,
        radius_arcsec=10.0,
        catalog="photcat",
    )

    assert lc["crts_id"].unique().tolist() == ["near"]
    assert lc["mag"].tolist() == [14.0, 14.2]
    assert lc["catalog"].unique().tolist() == ["photcat"]
    assert lc["mjd"].tolist() == [56000.0, 56001.0]


def test_fetch_crts_lightcurves_cgi_success_writes_parquet_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    calls: list[tuple[str, dict]] = []
    csv_text = _crts_csv(
        [
            (1149024031442, 14.1, 0.08, 1.0, 2.0, 56000.0, 0),
            (1149024031442, 14.3, 0.09, 1.0, 2.0, 56001.0, 0),
        ]
    )

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        params = kwargs.get("params") or {}
        if url == vetting.CRTS_CGI_URL:
            assert params["DB"] == "photcat"
            return _FakeCRTSResponse(
                '<html><a href="/DataRelease/upload/result.csv">CSV</a></html>',
                url=vetting.CRTS_CGI_URL,
            )
        assert url.endswith("/DataRelease/upload/result.csv")
        return _FakeCRTSResponse(csv_text, url=url)

    monkeypatch.setattr(vetting.requests, "get", fake_get)

    out = vetting.fetch_crts_lightcurves(df, output_dir=tmp_path)

    assert int(out.loc[0, "crts_lc_n_points"]) == 2
    assert [call[1].get("params", {}).get("DB") for call in calls if call[0] == vetting.CRTS_CGI_URL] == ["photcat"]
    parquet = tmp_path / "crts_lc_C1.parquet"
    assert parquet.exists()
    saved = pd.read_parquet(parquet)
    assert saved["crts_id"].astype(str).unique().tolist() == ["1149024031442"]
    assert saved["mag"].tolist() == [14.1, 14.3]
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert status.loc[0, "module"] == "CRTS LCs"
    assert status.loc[0, "status"] == "fetched"
    assert int(status.loc[0, "crts_lc_n_points"]) == 2


def test_fetch_crts_lightcurves_retries_orphancat_when_photcat_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    db_calls: list[str] = []

    def fake_get(url: str, **kwargs):
        params = kwargs.get("params") or {}
        if url == vetting.CRTS_CGI_URL:
            db_calls.append(params["DB"])
            if params["DB"] == "photcat":
                return _FakeCRTSResponse("There were 0 lines returned", url=vetting.CRTS_CGI_URL)
            return _FakeCRTSResponse('<a href="/DataRelease/upload/orphan.csv">CSV</a>', url=vetting.CRTS_CGI_URL)
        return _FakeCRTSResponse(
            _crts_csv([(9001, 15.1, 0.1, 1.0, 2.0, 56010.0, 0)]),
            url=url,
        )

    monkeypatch.setattr(vetting.requests, "get", fake_get)

    out = vetting.fetch_crts_lightcurves(df, output_dir=tmp_path)

    assert db_calls == ["photcat", "orphancat"]
    assert int(out.loc[0, "crts_lc_n_points"]) == 1
    saved = pd.read_parquet(tmp_path / "crts_lc_C1.parquet")
    assert saved["catalog"].unique().tolist() == ["orphancat"]


def test_fetch_crts_lightcurves_missing_csv_link_is_no_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])

    def fake_get(url: str, **kwargs):
        if url == vetting.CRTS_CGI_URL:
            return _FakeCRTSResponse("<html>There were 4 lines but no CSV link</html>", url=vetting.CRTS_CGI_URL)
        raise AssertionError("CSV URL should not be fetched")

    monkeypatch.setattr(vetting.requests, "get", fake_get)

    out = vetting.fetch_crts_lightcurves(df, output_dir=tmp_path)

    assert int(out.loc[0, "crts_lc_n_points"]) == 0
    assert not (tmp_path / "crts_lc_C1.parquet").exists()
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert status.loc[0, "module"] == "CRTS LCs"
    assert status.loc[0, "status"] == "no_data"
    assert int(status.loc[0, "crts_lc_n_points"]) == 0


def test_fetch_external_lcs_records_crts_failure_attr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])

    def fake_get(url: str, **kwargs):
        if url == vetting.CRTS_CGI_URL:
            return _FakeCRTSResponse("", status_code=502, url=vetting.CRTS_CGI_URL)
        raise AssertionError("CSV URL should not be fetched")

    monkeypatch.setattr(vetting.requests, "get", fake_get)

    out = vetting.fetch_external_lcs(
        df,
        output_dir=tmp_path,
        run_atlas=False,
        run_ztf=False,
        run_gaia_epoch=False,
        run_tess=False,
        run_neowise=False,
        run_kepler=False,
        run_aavso=False,
        run_ogle=False,
        run_stripe82=False,
        run_allwise_mep=False,
        run_vvvx_virac=False,
        run_ps1=False,
        run_crts=True,
    )

    assert out.attrs["external_lc_failures"][0].startswith("CRTS LCs failed:")
    status = pd.read_parquet(tmp_path / vetting.EXTERNAL_LC_STATUS_FILE)
    assert status.loc[0, "status"] == "error"
