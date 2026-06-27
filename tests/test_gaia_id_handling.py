from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

import malca.enrichment.characterize as characterize
import malca.catalogs.gaia_ids as gaia_ids_module
from malca.enrichment.characterize import query_gaia_by_ids
from malca.products.feature_layers import with_feature_columns
from malca.catalogs.gaia_id_repair import main as gaia_id_repair_main
from malca.catalogs.gaia_id_repair import repair_gaia_ids_frame, repair_review_db
from malca.ltv.review import _add_gaia_ids_from_index_ltv
from malca.stv.pipeline import _add_gaia_ids_from_index
from malca.catalogs.gaia_fetch import (
    _GAIA_QUERY_TEMPLATE,
    _ensure_gaia_schema,
    _extract_gaia_ids,
    _has_current_gaia_fetch_schema,
    _normalize_gaia_ids,
)
from malca.catalogs.gaia_ids import (
    GAIA_ID_MAPPING_COLUMNS,
    canonicalize_gaia_ids,
    canonicalize_gaia_ids_in_frame,
    normalize_gaia_source_id_series,
    parse_gaia_source_id,
)
from malca.io.table_io import write_feature_table


LARGE_GAIA_ID = 3564313717372918912
DR2_ID = "5885468452501959424"
DR3_ID = "5885468456822377088"


def _dr2_mapping_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_gaia_id": DR2_ID,
                "source_id": DR3_ID,
                "gaia_id": DR3_ID,
                "gaia_dr2_id": DR2_ID,
                "gaia_id_release": "dr2_translated",
                "gaia_id_mapping_status": "dr2_translated",
                "dr2_dr3_angular_distance_mas": 7.678524,
                "dr2_dr3_magnitude_difference": 0.3726778,
            }
        ],
        columns=list(GAIA_ID_MAPPING_COLUMNS),
    )


def _identity_mapping_frame(source_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_gaia_id": source_id,
                "source_id": source_id,
                "gaia_id": source_id,
                "gaia_dr2_id": pd.NA,
                "gaia_id_release": "dr3",
                "gaia_id_mapping_status": "dr3",
                "dr2_dr3_angular_distance_mas": 0.0,
                "dr2_dr3_magnitude_difference": 0.0,
            }
        ],
        columns=list(GAIA_ID_MAPPING_COLUMNS),
    )


def test_parse_gaia_source_id_rejects_unsafe_large_float() -> None:
    assert parse_gaia_source_id(LARGE_GAIA_ID) == str(LARGE_GAIA_ID)
    assert parse_gaia_source_id(f"{LARGE_GAIA_ID}.0") == str(LARGE_GAIA_ID)
    assert parse_gaia_source_id("4.56e2") == "456"
    assert parse_gaia_source_id(float(LARGE_GAIA_ID)) is None


def test_normalize_gaia_ids_accepts_numeric_strings_without_duplicates() -> None:
    assert _normalize_gaia_ids(["123.0", "4.56e2", 123, "bad", None]) == ["123", "456"]


def test_canonicalize_gaia_ids_keeps_local_dr3_and_translates_dr2(tmp_path, monkeypatch) -> None:
    gaia_cache = tmp_path / "gaia_dr3.parquet"
    mapping_cache = tmp_path / "gaia_id_mapping.parquet"
    pd.DataFrame({"source_id": [DR3_ID]}).to_parquet(gaia_cache, index=False)
    seen: dict[str, list[str]] = {}

    def fake_query(ids, **_kwargs):
        seen["ids"] = list(ids)
        return _dr2_mapping_frame()

    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", fake_query)

    out = canonicalize_gaia_ids(
        [DR3_ID, DR2_ID],
        gaia_cache_path=gaia_cache,
        mapping_cache_path=mapping_cache,
    )

    by_input = out.set_index("input_gaia_id")
    assert by_input.loc[DR3_ID, "source_id"] == DR3_ID
    assert by_input.loc[DR3_ID, "gaia_id_mapping_status"] == "dr3"
    assert by_input.loc[DR2_ID, "source_id"] == DR3_ID
    assert by_input.loc[DR2_ID, "gaia_dr2_id"] == DR2_ID
    assert by_input.loc[DR2_ID, "gaia_id_mapping_status"] == "dr2_translated"
    assert seen["ids"] == [DR2_ID]
    assert mapping_cache.exists()


def test_canonicalize_unmapped_id_marks_status_without_crashing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        gaia_ids_module,
        "query_dr2_neighbourhood_mappings",
        lambda *_args, **_kwargs: pd.DataFrame(columns=list(GAIA_ID_MAPPING_COLUMNS)),
    )

    with pytest.warns(RuntimeWarning, match="No Gaia DR3 DR2-neighbourhood mapping"):
        out = canonicalize_gaia_ids(
            ["999999999999999999"],
            gaia_cache_path=tmp_path / "missing_gaia.parquet",
            mapping_cache_path=tmp_path / "mapping.parquet",
        )

    row = out.iloc[0]
    assert row["source_id"] == "999999999999999999"
    assert row["gaia_id_mapping_status"] == "unmapped"


def test_canonicalize_checks_remote_dr3_before_dr2_neighbourhood(tmp_path, monkeypatch) -> None:
    calls = {"dr2": 0}

    def fake_dr2(*_args, **_kwargs):
        calls["dr2"] += 1
        return _identity_mapping_frame(DR3_ID)

    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda ids, **_kwargs: {str(value) for value in ids})
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", fake_dr2)

    out = canonicalize_gaia_ids(
        [DR3_ID],
        gaia_cache_path=tmp_path / "missing_gaia.parquet",
        mapping_cache_path=tmp_path / "mapping.parquet",
    )

    row = out.iloc[0]
    assert row["source_id"] == DR3_ID
    assert row["gaia_id_mapping_status"] == "dr3"
    assert pd.isna(row["gaia_dr2_id"])
    assert calls["dr2"] == 0


def test_canonicalize_frame_translates_gaia_id_when_source_id_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", lambda *_args, **_kwargs: _dr2_mapping_frame())

    out = canonicalize_gaia_ids_in_frame(
        pd.DataFrame({"candidate_id": ["618475536448"], "gaia_id": [DR2_ID], "source_id": [pd.NA]}),
        gaia_cache_path=tmp_path / "missing_gaia.parquet",
        mapping_cache_path=tmp_path / "mapping.parquet",
    )

    assert out.loc[0, "gaia_id"] == DR3_ID
    assert out.loc[0, "source_id"] == DR3_ID
    assert out.loc[0, "gaia_dr2_id"] == DR2_ID
    assert out.loc[0, "gaia_id_release"] == "dr2_translated"


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


def test_detect_index_merge_preserves_large_gaia_ids_as_strings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gaia_ids_module, "_write_gaia_id_mapping_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        gaia_ids_module,
        "_ids_present_in_local_dr3_cache",
        lambda ids, **_kwargs: {str(LARGE_GAIA_ID), "2237594329615223552"},
    )
    monkeypatch.setattr(
        gaia_ids_module,
        "query_dr2_neighbourhood_mappings",
        lambda *_args, **_kwargs: pd.DataFrame(columns=list(GAIA_ID_MAPPING_COLUMNS)),
    )
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


def test_ltv_index_merge_translates_dr2_ids(tmp_path, monkeypatch) -> None:
    index_path = tmp_path / "asassn_index.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": [101, 202],
            "gaia_id": [DR2_ID, DR3_ID],
        }
    ).to_parquet(index_path, index=False)
    monkeypatch.setattr(gaia_ids_module, "_write_gaia_id_mapping_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gaia_ids_module, "_ids_present_in_local_dr3_cache", lambda ids, **_kwargs: {DR3_ID})
    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", lambda *_args, **_kwargs: _dr2_mapping_frame())

    out = _add_gaia_ids_from_index_ltv(pd.DataFrame({"asas_sn_id": ["101", "202"]}), index_path, verbose=False)

    assert out.loc[0, "gaia_id"] == DR3_ID
    assert out.loc[0, "source_id"] == DR3_ID
    assert out.loc[0, "gaia_dr2_id"] == DR2_ID
    assert out.loc[1, "gaia_id"] == DR3_ID


def test_repair_frame_translates_and_merges_local_gaia(tmp_path, monkeypatch) -> None:
    gaia_cache = tmp_path / "gaia_dr3.parquet"
    mapping_cache = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "source_id": [DR3_ID],
            "ra": [233.51822502],
            "dec": [-54.18837841],
            "parallax": [0.5153323],
            "phot_g_mean_mag": [13.134209],
        }
    ).to_parquet(gaia_cache, index=False)
    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", lambda *_args, **_kwargs: _dr2_mapping_frame())

    repaired, stats = repair_gaia_ids_frame(
        pd.DataFrame({"candidate_id": ["618475536448"], "gaia_id": [DR2_ID], "source_id": [pd.NA]}),
        gaia_cache_path=gaia_cache,
        mapping_cache_path=mapping_cache,
    )

    assert stats["changed"] == 1
    assert stats["translated"] == 1
    assert repaired.loc[0, "gaia_id"] == DR3_ID
    assert repaired.loc[0, "source_id"] == DR3_ID
    assert repaired.loc[0, "gaia_dr2_id"] == DR2_ID
    assert float(repaired.loc[0, "parallax"]) == 0.5153323
    assert float(repaired.loc[0, "phot_g_mean_mag"]) == 13.134209


def test_repair_frame_only_changes_rows_with_missing_source_id(tmp_path, monkeypatch) -> None:
    gaia_cache = tmp_path / "gaia_dr3.parquet"
    mapping_cache = tmp_path / "mapping.parquet"
    pd.DataFrame(
        {
            "source_id": [DR3_ID],
            "parallax": [0.5153323],
        }
    ).to_parquet(gaia_cache, index=False)
    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", lambda *_args, **_kwargs: _dr2_mapping_frame())

    repaired, stats = repair_gaia_ids_frame(
        pd.DataFrame(
            {
                "candidate_id": ["needs-repair", "already-good"],
                "gaia_id": [DR2_ID, DR3_ID],
                "source_id": [pd.NA, DR3_ID],
            }
        ),
        gaia_cache_path=gaia_cache,
        mapping_cache_path=mapping_cache,
    )

    assert stats["changed"] == 1
    assert stats["translated"] == 1
    assert repaired.loc[0, "source_id"] == DR3_ID
    assert repaired.loc[1, "source_id"] == DR3_ID
    assert pd.isna(repaired.loc[1, "gaia_id_mapping_status"])


def test_repair_review_db_dry_run_does_not_migrate_schema(tmp_path) -> None:
    db_path = tmp_path / "old_review.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO candidates(candidate_id, payload_json) VALUES (?, ?)",
        ("618475536448", f'{{"gaia_id":"{DR2_ID}"}}'),
    )
    conn.commit()
    before_cols = [row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()]
    conn.close()

    stats = repair_review_db(
        db_path,
        write=False,
        gaia_cache_path=tmp_path / "missing_gaia.parquet",
        mapping_cache_path=tmp_path / "mapping.parquet",
        write_mapping_cache=False,
        query_tap=False,
    )

    conn = sqlite3.connect(db_path)
    after_cols = [row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()]
    conn.close()
    assert stats["rows"] == 1
    assert before_cols == ["candidate_id", "payload_json"]
    assert after_cols == before_cols
    assert not (tmp_path / "mapping.parquet").exists()


def test_repair_review_db_write_mode_migrates_and_updates_gaia_fields(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "old_review.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO candidates(candidate_id, payload_json) VALUES (?, ?)",
        ("618475536448", f'{{"gaia_id":"{DR2_ID}"}}'),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(gaia_ids_module, "query_dr3_source_ids", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(gaia_ids_module, "query_dr2_neighbourhood_mappings", lambda *_args, **_kwargs: _dr2_mapping_frame())

    stats = repair_review_db(
        db_path,
        write=True,
        gaia_cache_path=tmp_path / "missing_gaia.parquet",
        mapping_cache_path=tmp_path / "mapping.parquet",
        write_mapping_cache=False,
    )

    conn = sqlite3.connect(db_path)
    cols = [row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()]
    row = conn.execute(
        "SELECT source_id, gaia_id, gaia_dr2_id, gaia_id_mapping_status, payload_json FROM candidates WHERE candidate_id=?",
        ("618475536448",),
    ).fetchone()
    conn.close()
    payload = json.loads(row[4])
    external_stats = json.loads(payload["external_stats"])
    assert stats["changed"] == 1
    assert stats["translated"] == 1
    assert "source_id" in cols
    assert row[:4] == (DR3_ID, DR3_ID, DR2_ID, "dr2_translated")
    assert payload["source_id"] == DR3_ID
    assert payload["gaia_id"] == DR3_ID
    assert external_stats["source_id"] == DR3_ID
    assert external_stats["gaia_id"] == DR3_ID
    assert external_stats["gaia_dr2_id"] == DR2_ID


def test_gaia_id_repair_rejects_fetch_gaia_in_dry_run(tmp_path) -> None:
    db_path = tmp_path / "review.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit):
        gaia_id_repair_main(["--fetch-gaia", str(db_path)])


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
    monkeypatch.setattr(characterize, "canonicalize_gaia_ids_in_frame", lambda frame, **_kwargs: frame)
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


def test_characterize_canonicalizes_dr2_before_gaia_query(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_canonicalize(frame, **_kwargs):
        out = frame.copy()
        out["gaia_id"] = DR3_ID
        out["source_id"] = DR3_ID
        out["gaia_dr2_id"] = DR2_ID
        out["gaia_id_mapping_status"] = "dr2_translated"
        return out

    def fake_query_gaia_by_ids(gaia_ids, **_kwargs):
        seen["gaia_ids"] = list(gaia_ids)
        return pd.DataFrame(
            {
                "source_id": [DR3_ID],
                "ra": [233.5],
                "dec": [-54.1],
                "phot_g_mean_mag": [13.1],
                "bp_rp": [1.0],
                "parallax": [0.515],
                "pmra": [1.0],
                "pmdec": [2.0],
            }
        )

    monkeypatch.setattr(characterize, "canonicalize_gaia_ids_in_frame", fake_canonicalize)
    monkeypatch.setattr(characterize, "query_gaia_by_ids", fake_query_gaia_by_ids)
    monkeypatch.setattr(characterize, "_module_completed", lambda *_args, **_kwargs: True)

    out = characterize.characterize_candidates_df(
        pd.DataFrame({"asas_sn_id": ["618475536448"], "gaia_id": [DR2_ID]}),
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

    assert seen["gaia_ids"] == [DR3_ID]
    view = with_feature_columns(out, ["source_id", "gaia_id", "gaia_dr2_id", "parallax"])
    assert view.loc[0, "source_id"] == DR3_ID
    assert view.loc[0, "gaia_id"] == DR3_ID
    assert view.loc[0, "gaia_dr2_id"] == DR2_ID
    assert float(view.loc[0, "parallax"]) == 0.515
