from __future__ import annotations

import pandas as pd

from malca.characterize import query_gaia_by_ids
from malca.detect import _add_gaia_ids_from_index
from malca.gaia_fetch import _normalize_gaia_ids
from malca.gaia_ids import normalize_gaia_source_id_series, parse_gaia_source_id


LARGE_GAIA_ID = 3564313717372918912


def test_parse_gaia_source_id_rejects_unsafe_large_float() -> None:
    assert parse_gaia_source_id(LARGE_GAIA_ID) == str(LARGE_GAIA_ID)
    assert parse_gaia_source_id(f"{LARGE_GAIA_ID}.0") == str(LARGE_GAIA_ID)
    assert parse_gaia_source_id("4.56e2") == "456"
    assert parse_gaia_source_id(float(LARGE_GAIA_ID)) is None


def test_normalize_gaia_ids_accepts_numeric_strings_without_duplicates() -> None:
    assert _normalize_gaia_ids(["123.0", "4.56e2", 123, "bad", None]) == ["123", "456"]


def test_detect_index_merge_preserves_large_gaia_ids_as_strings(tmp_path) -> None:
    index_path = tmp_path / "asassn_index.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": [101, 202],
            "gaia_id": [LARGE_GAIA_ID, 2237594329615223552],
        }
    ).to_parquet(index_path, index=False)

    events = pd.DataFrame({"path": ["/data/101.dat3", "/data/999.dat3"]})
    out = _add_gaia_ids_from_index(events, index_path)

    assert out.loc[0, "gaia_id"] == str(LARGE_GAIA_ID)
    assert pd.isna(out.loc[1, "gaia_id"])
    assert pd.api.types.is_string_dtype(out["gaia_id"])


def test_normalize_gaia_source_id_series_vectorizes_integer_columns() -> None:
    series = pd.Series([LARGE_GAIA_ID], dtype="int64")

    out = normalize_gaia_source_id_series(series)

    assert out.tolist() == [str(LARGE_GAIA_ID)]
    assert pd.api.types.is_string_dtype(out)


def test_query_gaia_by_ids_matches_numeric_like_requested_ids(tmp_path) -> None:
    cache_path = tmp_path / "gaia_cache.parquet"
    pd.DataFrame(
        {
            "source_id": [LARGE_GAIA_ID],
            "ra": [12.3],
            "dec": [-45.6],
        }
    ).to_parquet(cache_path, index=False)

    out = query_gaia_by_ids([f"{LARGE_GAIA_ID}.0"], cache_file=str(cache_path))

    assert len(out) == 1
    assert out.loc[out.index[0], "source_id"] == str(LARGE_GAIA_ID)
    assert float(out.loc[out.index[0], "ra"]) == 12.3
