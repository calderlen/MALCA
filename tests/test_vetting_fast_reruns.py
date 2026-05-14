from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import malca.vetting as vetting


def test_asassn_variables_local_csv_does_not_call_tap(monkeypatch, tmp_path) -> None:
    local_csv = tmp_path / "asassn_variables_220326.csv"
    pd.DataFrame(
        {
            "ID": ["ASASSN-V LOCAL"],
            "RAJ2000": [10.0],
            "DEJ2000": [-5.0],
            "ML_classification": ["EA"],
            "Period": [1.25],
        }
    ).to_csv(local_csv, index=False)

    def fail_tap(*_args, **_kwargs):
        raise AssertionError("TAP should not be called for local ASAS-SN matching")

    monkeypatch.setattr(vetting, "batch_tap_crossmatch", fail_tap)

    df = pd.DataFrame({"ra": [10.0], "dec": [-5.0]})
    out = vetting.crossmatch_asassn_variables(df, method="local", local_csv=local_csv)

    assert out.loc[0, "asassn_var_name"] == "ASASSN-V LOCAL"
    assert out.loc[0, "asassn_var_type"] == "EA"
    assert float(out.loc[0, "asassn_var_period"]) == 1.25


def test_asassn_transients_populate_tns_fields(tmp_path) -> None:
    vetting._tns_cache.clear()
    transients_csv = tmp_path / "asassn_transients.csv"
    pd.DataFrame(
        {
            "asassn_id": ["ASASSN-26bv"],
            "other_ids": [""],
            "atel_tns": ["-"],
            "ra": ["6:2:44.21"],
            "dec": ["-4:11:6.2"],
            "discovery_ut": ["2026-03-19.19"],
            "spectroscopic_class": ["-"],
            "comments": ["CV candidate, matches to PS1 g=21.3"],
            "ra_deg": [90.684208],
            "dec_deg": [-4.185056],
        }
    ).to_csv(transients_csv, index=False)

    df = pd.DataFrame({"ra": [90.684208], "dec": [-4.185056]})
    out = vetting.crossmatch_tns(df, local_csvs=[transients_csv])

    assert out.loc[0, "tns_name"] == "ASAS-SN:ASASSN-26bv"
    assert out.loc[0, "tns_type"] == "CV candidate"
    assert out.loc[0, "tns_disc_date"] == "2026-03-19"


def test_gaia_variability_cache_hit_and_refresh(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame({"gaia_id": ["123"]})
    calls = {"n": 0}

    monkeypatch.setattr(vetting, "_connect_gaia_taps_until_available", lambda *args, **kwargs: ["tap"])

    def fake_run_query(_taps, query: str, *, label: str):
        calls["n"] += 1
        if "gaiadr3.vari_summary" in query:
            return [{"source_id": 123, "in_vari_classification_result": True}]
        if "gaiadr3.vari_classifier_result" in query:
            return [{"source_id": 123, "best_class_name": "LPV", "best_class_score": 0.8}]
        raise AssertionError(label)

    monkeypatch.setattr(vetting, "_run_gaia_tap_query_until_success", fake_run_query)

    first = vetting.query_gaia_variability(df, cache_dir=tmp_path)
    assert first.loc[0, "gaia_var_class"] == "LPV"
    assert calls["n"] == 2

    def fail_run_query(*_args, **_kwargs):
        raise AssertionError("Gaia TAP should not be called on cache hit")

    monkeypatch.setattr(vetting, "_run_gaia_tap_query_until_success", fail_run_query)
    second = vetting.query_gaia_variability(df, cache_dir=tmp_path)
    assert second.loc[0, "gaia_var_class"] == "LPV"

    monkeypatch.setattr(vetting, "_run_gaia_tap_query_until_success", fake_run_query)
    refreshed = vetting.query_gaia_variability(df, cache_dir=tmp_path, refresh_cache=True)
    assert refreshed.loc[0, "gaia_var_class"] == "LPV"
    assert calls["n"] == 4


def test_simbad_cache_hit_and_refresh(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
    calls = {"n": 0}

    def fake_simbad(query_df, valid, n, radius_arcsec):
        calls["n"] += 1
        query_df = query_df.copy()
        query_df["simbad_main_id"] = "SIMBAD OBJ"
        query_df["simbad_otype"] = "V*"
        query_df["simbad_nbref"] = 7
        query_df["simbad_sep_arcsec"] = 0.2
        return query_df

    monkeypatch.setattr(vetting, "_simbad_via_xmatch", fake_simbad)
    first = vetting.query_simbad_batch(df, cache_dir=tmp_path)
    assert first.loc[0, "simbad_main_id"] == "SIMBAD OBJ"
    assert calls["n"] == 1

    monkeypatch.setattr(
        vetting,
        "_simbad_via_xmatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SIMBAD should use cache")),
    )
    second = vetting.query_simbad_batch(df, cache_dir=tmp_path)
    assert second.loc[0, "simbad_main_id"] == "SIMBAD OBJ"

    monkeypatch.setattr(vetting, "_simbad_via_xmatch", fake_simbad)
    refreshed = vetting.query_simbad_batch(df, cache_dir=tmp_path, refresh_cache=True)
    assert refreshed.loc[0, "simbad_main_id"] == "SIMBAD OBJ"
    assert calls["n"] == 2


def test_alerce_cache_hit_and_refresh(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
    calls = {"n": 0}

    def fake_alerce(_ra, _dec, _radius):
        calls["n"] += 1
        return {
            "alerce_oid": "ZTF-test",
            "alerce_ndet": 5,
            "alerce_lc_class": "YSO",
            "alerce_lc_prob": 0.7,
            "alerce_stamp_class": "",
            "alerce_stamp_prob": np.nan,
        }

    monkeypatch.setattr(vetting, "_alerce_query_single", fake_alerce)
    first = vetting.query_alerce(df, workers=1, cache_dir=tmp_path)
    assert first.loc[0, "alerce_oid"] == "ZTF-test"
    assert calls["n"] == 1

    monkeypatch.setattr(
        vetting,
        "_alerce_query_single",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ALeRCE should use cache")),
    )
    second = vetting.query_alerce(df, workers=1, cache_dir=tmp_path)
    assert second.loc[0, "alerce_oid"] == "ZTF-test"

    monkeypatch.setattr(vetting, "_alerce_query_single", fake_alerce)
    refreshed = vetting.query_alerce(df, workers=1, cache_dir=tmp_path, refresh_cache=True)
    assert refreshed.loc[0, "alerce_oid"] == "ZTF-test"
    assert calls["n"] == 2


def test_vetting_cli_only_cache_refresh_and_skip_existing(monkeypatch, tmp_path) -> None:
    captured = {}
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(vetting, "read_parquet_table", lambda path: pd.DataFrame({"ra": [1.0], "dec": [2.0]}))
    monkeypatch.setattr(vetting, "write_parquet_table", lambda df, path: captured.setdefault("output_path", path))

    def fake_vet_candidates(df, **kwargs):
        captured["kwargs"] = kwargs
        return df

    monkeypatch.setattr(vetting, "vet_candidates", fake_vet_candidates)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "malca",
            str(input_path),
            "--output",
            str(output_path),
            "--no-checkpoint",
            "--only",
            "simbad,asassn-var",
            "--cache-dir",
            str(cache_dir),
            "--refresh-cache",
            "--skip-existing",
        ],
    )

    vetting.main()

    kwargs = captured["kwargs"]
    assert kwargs["run_simbad"] is True
    assert kwargs["run_asassn_var"] is True
    assert kwargs["run_gaia_var"] is False
    assert kwargs["run_atlas"] is False
    assert kwargs["cache_dir"] == cache_dir
    assert kwargs["refresh_cache"] is True
    assert kwargs["skip_existing"] is True
    assert captured["output_path"] == output_path
