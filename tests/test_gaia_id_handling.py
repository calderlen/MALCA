from __future__ import annotations

import pandas as pd

import malca.characterize as characterize
from malca.characterize import query_gaia_by_ids
from malca.feature_layers import with_feature_columns
from malca.stv.pipeline import _add_gaia_ids_from_index
from malca.gaia_fetch import (
    _GAIA_QUERY_TEMPLATE,
    _ensure_gaia_schema,
    _extract_gaia_ids,
    _has_current_gaia_fetch_schema,
    _normalize_gaia_ids,
)
from malca.gaia_ids import normalize_gaia_source_id_series, parse_gaia_source_id
from malca.table_io import write_feature_table


LARGE_GAIA_ID = 3564313717372918912


def test_parse_gaia_source_id_rejects_unsafe_large_float() -> None:
    assert parse_gaia_source_id(LARGE_GAIA_ID) == str(LARGE_GAIA_ID)
    assert parse_gaia_source_id(f"{LARGE_GAIA_ID}.0") == str(LARGE_GAIA_ID)
    assert parse_gaia_source_id("4.56e2") == "456"
    assert parse_gaia_source_id(float(LARGE_GAIA_ID)) is None


def test_normalize_gaia_ids_accepts_numeric_strings_without_duplicates() -> None:
    assert _normalize_gaia_ids(["123.0", "4.56e2", 123, "bad", None]) == ["123", "456"]


def test_extract_gaia_ids_reads_direct_gaia_passers(tmp_path) -> None:
    input_path = tmp_path / "candidates.parquet"
    xmatch_path = tmp_path / "unused_xmatch.parquet"
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_1001", "stv_1002", "stv_bad"],
                "timescale": ["stv", "stv", "stv"],
                "gaia_id": ["1001", "1002", "bad"],
                "failed_any": [False, True, False],
                "large_payload": ["x" * 1000, "y" * 1000, "z" * 1000],
            }
        ),
        input_path,
    )

    out = _extract_gaia_ids(input_path, xmatch_path, only_passers=True)

    assert out == ["1001"]


def test_extract_gaia_ids_uses_minimal_crossmatch_columns(tmp_path) -> None:
    input_path = tmp_path / "candidates.parquet"
    xmatch_path = tmp_path / "xmatch.parquet"
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_A", "stv_B"],
                "timescale": ["stv", "stv"],
                "asas_sn_id": ["A", "B"],
                "failed_any": [False, True],
                "large_payload": ["x" * 1000, "y" * 1000],
            }
        ),
        input_path,
    )
    pd.DataFrame(
        {
            "asas_sn_id": ["A", "B"],
            "gaia_id": ["2001", "2002"],
            "unused_payload": ["u" * 1000, "v" * 1000],
        }
    ).to_parquet(xmatch_path, index=False)

    out = _extract_gaia_ids(input_path, xmatch_path, only_passers=True)

    assert out == ["2001"]


def test_detect_index_merge_preserves_large_gaia_ids_as_strings(tmp_path) -> None:
    index_path = tmp_path / "asassn_index.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": [101, 202],
            "gaia_id": [LARGE_GAIA_ID, 2237594329615223552],
        }
    ).to_parquet(index_path, index=False)

    events = pd.DataFrame({"lc_path": ["/data/101.dat3", "/data/999.dat3"]})
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


def test_gaia_fetch_current_schema_requires_separate_bp_rp() -> None:
    stale = pd.DataFrame(
        {
            "source_id": ["123"],
            "w1": [11.0],
            "w1_err": [0.1],
            "w2": [10.8],
            "w2_err": [0.1],
            "w3": [10.0],
            "w3_err": [0.2],
            "w4": [9.5],
            "w4_err": [0.3],
        }
    )
    current = stale.assign(phot_bp_mean_mag=[15.1], phot_rp_mean_mag=[14.2])

    assert not _has_current_gaia_fetch_schema(stale)
    assert _has_current_gaia_fetch_schema(current)
    assert "g.phot_bp_mean_mag" in _GAIA_QUERY_TEMPLATE
    assert "g.phot_rp_mean_mag" in _GAIA_QUERY_TEMPLATE

    normalized = _ensure_gaia_schema(current)
    assert "phot_bp_mean_mag" in normalized.columns
    assert "phot_rp_mean_mag" in normalized.columns


def test_characterize_uses_existing_gaia_id_without_crossmatch(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_query_gaia_by_ids(gaia_ids, **_kwargs):
        seen["gaia_ids"] = list(gaia_ids)
        return pd.DataFrame(
            {
                "source_id": [str(LARGE_GAIA_ID)],
                "ra": [12.3],
                "dec": [-45.6],
                "phot_g_mean_mag": [14.0],
                "bp_rp": [1.2],
                "parallax": [2.0],
                "pmra": [1.0],
                "pmdec": [2.0],
            }
        )

    monkeypatch.setattr(characterize, "query_gaia_by_ids", fake_query_gaia_by_ids)
    monkeypatch.setattr(characterize, "_module_completed", lambda *_args, **_kwargs: True)

    out = characterize.characterize_candidates_df(
        pd.DataFrame({"asas_sn_id": ["A"], "gaia_id": [str(LARGE_GAIA_ID)]}),
        crossmatch=tmp_path / "missing_crossmatch.parquet",
        cache=None,
        dust=False,
        starhorse=None,
        run_banyan=False,
        run_iphas=False,
        run_sfr=False,
        run_clusters=False,
        run_unwise=False,
    )

    assert seen["gaia_ids"] == [str(LARGE_GAIA_ID)]
    view = with_feature_columns(out, ["source_id", "ra"])
    assert view.loc[0, "source_id"] == str(LARGE_GAIA_ID)
    assert float(view.loc[0, "ra"]) == 12.3
