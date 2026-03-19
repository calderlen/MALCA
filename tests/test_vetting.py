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


def test_parse_ogle_ews_html_extracts_events() -> None:
    html = """
    <html><body>
    <table>
      <tr><th>Event</th><th>Field</th><th>RA</th><th>Dec</th><th>tau</th></tr>
      <tr>
        <td><a href="blg-0001.html">2024-BLG-0001</a></td>
        <td>BLG100</td>
        <td>17:47:03.53</td>
        <td>-31:00:17.4</td>
        <td>28.5</td>
      </tr>
    </table>
    </body></html>
    """

    out = vetting._parse_ogle_ews_html(
        html,
        page_url="https://ogle.astrouw.edu.pl/ogle4/ews/2024/ews.html",
        default_year=2024,
    )

    assert len(out) == 1
    assert out.loc[0, "source"] == "OGLE-EWS"
    assert out.loc[0, "event_id"] == "OGLE-2024-BLG-0001"
    assert out.loc[0, "alias"] == "2024-BLG-0001"
    assert np.isclose(out.loc[0, "timescale_days"], 28.5)
    assert out.loc[0, "timescale_kind"] == "tau"
    assert out.loc[0, "event_year"] == 2024
    assert out.loc[0, "source_url"] == "https://ogle.astrouw.edu.pl/ogle4/ews/2024/blg-0001.html"
    assert np.isfinite(out.loc[0, "ra"])
    assert np.isfinite(out.loc[0, "dec"])


def test_parse_kmtnet_listpage_text_extracts_events() -> None:
    text = (
        "Event  Field  Star #  Clear/Probable/possible  Related event  RA  Dec  t_0  t_E\n"
        "KMT-2024-BLG-0001  BLG01  12345  clear  OGLE-2024-BLG-0001  17:47:03.53  -31:00:17.4  2460482.0  18.2\n"
    )

    out = vetting._parse_kmtnet_listpage_text(text, year=2024)

    assert len(out) == 1
    assert out.loc[0, "source"] == "KMTNet"
    assert out.loc[0, "event_id"] == "KMT-2024-BLG-0001"
    assert out.loc[0, "alias"] == "OGLE-2024-BLG-0001"
    assert out.loc[0, "status"] == "clear"
    assert np.isclose(out.loc[0, "timescale_days"], 18.2)
    assert out.loc[0, "timescale_kind"] == "te"
    assert out.loc[0, "source_url"] == "https://kmtnet.kasi.re.kr/ulens/event/2024/"


def test_parse_moa_events_html_extracts_events() -> None:
    html = """
    <html><body>
    <table>
      <tr><th>EventNum</th><th>RA</th><th>Dec</th><th>CUSPWIDTH</th><th>LC_URL</th></tr>
      <tr>
        <td>2021-BLG-123</td>
        <td>270.123</td>
        <td>-30.456</td>
        <td>1.7</td>
        <td>https://example.test/moa_lc.dat</td>
      </tr>
    </table>
    </body></html>
    """

    out = vetting._parse_moa_events_html(html)

    assert len(out) == 1
    assert out.loc[0, "source"] == "MOA"
    assert out.loc[0, "event_id"] == "MOA-2021-BLG-123"
    assert np.isclose(out.loc[0, "ra"], 270.123)
    assert np.isclose(out.loc[0, "dec"], -30.456)
    assert np.isclose(out.loc[0, "timescale_days"], 1.7)
    assert out.loc[0, "timescale_kind"] == "cuspwidth"
    assert out.loc[0, "source_url"] == "https://example.test/moa_lc.dat"


def test_fetch_microlensing_event_catalog_combines_sources_and_caches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        vetting,
        "fetch_ogle_ews_microlensing_events",
        lambda **kwargs: pd.DataFrame([{
            "source": "OGLE-EWS", "event_id": "OGLE-2024-BLG-0001", "alias": "",
            "ra": 270.0, "dec": -30.0, "timescale_days": 28.5, "timescale_kind": "tau",
            "status": "", "source_url": "https://ogle", "event_year": 2024,
        }]),
    )
    monkeypatch.setattr(
        vetting,
        "fetch_kmtnet_microlensing_events",
        lambda **kwargs: pd.DataFrame([{
            "source": "KMTNet", "event_id": "KMT-2024-BLG-0001", "alias": "OGLE-2024-BLG-0001",
            "ra": 270.001, "dec": -30.001, "timescale_days": 18.2, "timescale_kind": "te",
            "status": "clear", "source_url": "https://kmtnet", "event_year": 2024,
        }]),
    )
    monkeypatch.setattr(
        vetting,
        "fetch_moa_microlensing_events",
        lambda **kwargs: pd.DataFrame([{
            "source": "MOA", "event_id": "MOA-2021-BLG-123", "alias": "",
            "ra": 271.0, "dec": -31.0, "timescale_days": 1.7, "timescale_kind": "cuspwidth",
            "status": "", "source_url": "https://moa", "event_year": 2021,
        }]),
    )

    out = vetting.fetch_microlensing_event_catalog(cache_dir=tmp_path, force_download=True, show_tqdm=False)

    assert set(out["source"]) == {"OGLE-EWS", "KMTNet", "MOA"}
    assert (tmp_path / "microlens_union.parquet").exists()


def test_fetch_microlensing_event_catalog_uses_stale_cache_on_refresh_failure(monkeypatch, tmp_path: Path) -> None:
    cached = pd.DataFrame([{
        "source": "OGLE-EWS", "event_id": "OGLE-2024-BLG-0001", "alias": "",
        "ra": 270.0, "dec": -30.0, "timescale_days": 28.5, "timescale_kind": "tau",
        "status": "", "source_url": "https://ogle", "event_year": 2024, "source_rank": 0,
    }])
    cached.to_parquet(tmp_path / "microlens_union.parquet", index=False)

    monkeypatch.setattr(vetting, "fetch_ogle_ews_microlensing_events", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("ogle down")))
    monkeypatch.setattr(vetting, "fetch_kmtnet_microlensing_events", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("kmtnet down")))
    monkeypatch.setattr(vetting, "fetch_moa_microlensing_events", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("moa down")))

    out = vetting.fetch_microlensing_event_catalog(cache_dir=tmp_path, force_download=True, show_tqdm=False)

    assert len(out) == 1
    assert out.loc[0, "event_id"] == "OGLE-2024-BLG-0001"


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


def test_crossmatch_microlensing_catalogs_uses_published_ogle_fallback_when_union_fetch_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        vetting,
        "fetch_microlensing_event_catalog",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("union down")),
    )

    def fake_fallback(df: pd.DataFrame, *, radius_arcsec: float, chunk_size: int, method: str) -> pd.DataFrame:
        out = df.copy()
        out["microlens_match"] = [True]
        out["microlens_catalog"] = ["OGLE-IV"]
        out["microlens_name"] = ["OGLE-2015-BLG-0001"]
        out["microlens_alt_name"] = [""]
        out["microlens_te_days"] = [33.0]
        out["microlens_sep_arcsec"] = [0.4]
        return out

    monkeypatch.setattr(vetting, "_crossmatch_published_ogle_microlensing", fake_fallback)

    df = pd.DataFrame({"ra": [270.0], "dec": [-30.0]})
    out = vetting.crossmatch_microlensing_catalogs(df)

    assert bool(out.loc[0, "microlens_match"])
    assert out.loc[0, "microlens_catalog"] == "OGLE-IV"


def test_print_vetting_summary_marks_microlens_matches_as_known() -> None:
    df = pd.DataFrame({
        "microlens_match": [True, False],
        "microlens_catalog": ["OGLE-IV", ""],
    })

    vetting._print_vetting_summary(df, time.perf_counter())

    assert bool(df.loc[0, "vetting_likely_known"])
    assert not bool(df.loc[1, "vetting_likely_known"])
