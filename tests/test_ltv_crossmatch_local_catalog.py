from __future__ import annotations

import pandas as pd
import pytest

from malca.ltv.crossmatch import clear_catalog_cache, load_local_catalog, merge_local_catalog


@pytest.fixture(autouse=True)
def _clear_local_catalog_cache():
    clear_catalog_cache()
    yield
    clear_catalog_cache()


def _catalog_frame(**overrides) -> pd.DataFrame:
    data = {
        "asas_sn_id": ["asassn-a", "asassn-b"],
        "gaia_id": [1234567890123, 9876543210987],
        "plx": [1.25, 2.5],
        "pm_ra": [3.0, 4.0],
        "pm_dec": [-1.0, -2.0],
        "gaia_mag": [13.1, 14.2],
        "id_vsx": [101, 202],
        "name": ["VSX A", "VSX B"],
        "class": ["EA", "M"],
        "period": [1.5, 2.5],
        "ra_deg": [10.0, 20.0],
        "dec_deg": [-5.0, 6.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_load_local_catalog_reads_parquet_and_renames_columns(tmp_path) -> None:
    path = tmp_path / "vsx.parquet"
    _catalog_frame().to_parquet(path, index=False)

    out = load_local_catalog(path)

    assert "parallax" in out.columns
    assert "pmra" in out.columns
    assert "phot_g_mean_mag" in out.columns
    assert "vsx_oid" in out.columns
    assert "vsx_name" in out.columns
    assert out.loc[0, "gaia_id"] == "1234567890123"
    assert out.loc[0, "vsx_type"] == "EA"


def test_load_local_catalog_keeps_legacy_csv_support(tmp_path) -> None:
    path = tmp_path / "vsx.csv"
    _catalog_frame().to_csv(path, index=False)

    out = load_local_catalog(path)

    assert out.loc[1, "parallax"] == 2.5
    assert out.loc[1, "vsx_name"] == "VSX B"


def test_merge_local_catalog_reads_parquet_path(tmp_path) -> None:
    path = tmp_path / "vsx.parquet"
    _catalog_frame().to_parquet(path, index=False)
    candidates = pd.DataFrame({"asas_sn_id": ["asassn-a", "missing"]})

    out = merge_local_catalog(candidates, catalog_path=path)

    matched = out.loc[out["asas_sn_id"] == "asassn-a"].iloc[0]
    missing = out.loc[out["asas_sn_id"] == "missing"].iloc[0]
    assert matched["gaia_id"] == "1234567890123"
    assert matched["vsx_name"] == "VSX A"
    assert pd.isna(missing["gaia_id"])


def test_load_local_catalog_cache_is_path_aware(tmp_path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _catalog_frame(gaia_id=[111, 222], name=["FIRST A", "FIRST B"]).to_parquet(first, index=False)
    _catalog_frame(gaia_id=[333, 444], name=["SECOND A", "SECOND B"]).to_parquet(second, index=False)

    first_out = load_local_catalog(first)
    second_out = load_local_catalog(second)

    assert first_out.loc[0, "gaia_id"] == "111"
    assert second_out.loc[0, "gaia_id"] == "333"
    assert second_out.loc[0, "vsx_name"] == "SECOND A"
