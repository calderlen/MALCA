from __future__ import annotations

from pathlib import Path

from astropy.table import Table
import pandas as pd
import pytest

from malca import fetch as skypatrol_fetch
from malca.review import fetch as review_fetch
from malca import vetting


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

    import malca.stats as stats_mod

    def fake_compute_stats(_candidate_id: str, _parent: str, *, compute_ls: bool = True):
        calls["stats"] += 1
        return pd.DataFrame(), {"photometry_mean_mag": 14.0 + calls["stats"]}

    monkeypatch.setattr(stats_mod, "compute_stats", fake_compute_stats)

    first = review_fetch._compute_stats_from_skypatrol_csv(lc_path)
    second = review_fetch._compute_stats_from_skypatrol_csv(lc_path)

    assert calls["stats"] == 1
    assert first == second
    assert first["stats_photometry_mean_mag"] == 15.0

    with lc_path.open("a", encoding="utf-8") as f:
        f.write("2450002.5,1.2,0.1,13.8,0.02,,2.0,g,G,bi\n")

    third = review_fetch._compute_stats_from_skypatrol_csv(lc_path)

    assert calls["stats"] == 2
    assert third["stats_photometry_mean_mag"] == 16.0


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
        raise AssertionError("IRSA TAP should not be constructed on cache hit")

    monkeypatch.setattr(vetting.pyvo.dal, "TAPService", fail)

    out = vetting.fetch_ztf_lightcurves(df, output_dir=tmp_path)

    assert int(out.loc[0, "ztf_lc_n_det"]) == 4
    assert out.loc[0, "ztf_lc_g_range"] == 1.0
    assert out.loc[0, "ztf_lc_r_range"] == 3.0


def test_fetch_ztf_lightcurves_corrupt_cache_falls_back_to_tap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    (tmp_path / "ztf_lc_C1.parquet").write_text("not parquet", encoding="ascii")

    class Result:
        def __init__(self, table: Table):
            self._table = table

        def to_table(self) -> Table:
            return self._table

    class FakeTap:
        def __init__(self):
            self.calls = 0

        def run_sync(self, query: str):
            self.calls += 1
            if "lightcurve" in query:
                return Result(
                    Table(
                        {
                            "oid": [123, 123],
                            "hjd": [2450000.5, 2450001.5],
                            "mag": [15.0, 14.0],
                            "magerr": [0.1, 0.1],
                            "filtercode": [1, 1],
                            "catflags": [0, 0],
                        }
                    )
                )
            return Result(Table({"oid": [123]}))

    fake_tap = FakeTap()
    monkeypatch.setattr(vetting.pyvo.dal, "TAPService", lambda *_args, **_kwargs: fake_tap)

    out = vetting.fetch_ztf_lightcurves(df, output_dir=tmp_path, workers=1)

    assert fake_tap.calls == 2
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
        raise AssertionError("IRSA TAP should not be constructed on no-data cache hit")

    monkeypatch.setattr(vetting.pyvo.dal, "TAPService", fail)

    out = vetting.fetch_ztf_lightcurves(df, output_dir=tmp_path)

    assert int(out.loc[0, "ztf_lc_n_det"]) == 0
