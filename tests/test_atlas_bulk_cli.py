from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from malca.enrichment import external_lcs
from malca.external_lc_manifest import upsert_external_lc_manifest_entry
from malca.review.store import db_connect, get_candidate_payload, upsert_candidates_frame


def _install_review_cohort_loader(
    monkeypatch: pytest.MonkeyPatch,
    cohort: pd.DataFrame,
    calls: list[dict[str, object]],
) -> None:
    module = types.ModuleType("malca.review.paper_candidates")

    def fake_load_reviewed_cohort(review_db, **kwargs):
        calls.append({"review_db": Path(review_db), **kwargs})
        return cohort.copy()

    module.load_reviewed_cohort = fake_load_reviewed_cohort
    monkeypatch.setitem(sys.modules, "malca.review.paper_candidates", module)


def _install_atlas_fetcher(
    monkeypatch: pytest.MonkeyPatch,
    fetch,
) -> None:
    atlas_module = types.ModuleType("malca.enrichment.atlas_forced_photometry")
    atlas_module.query_atlas_forced_phot = fetch
    monkeypatch.setitem(
        sys.modules,
        "malca.enrichment.atlas_forced_photometry",
        atlas_module,
    )

    vetting_module = types.ModuleType("malca.enrichment.vetting")

    def fail_vetting_orchestrator(*_args, **_kwargs):
        raise AssertionError("--atlas-only must not call the multi-survey vetting orchestrator")

    vetting_module.fetch_external_lcs = fail_vetting_orchestrator
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", vetting_module)


def _atlas_only_args(review_db: Path, *extra: str):
    return external_lcs.build_arg_parser().parse_args(
        [
            str(review_db),
            "--atlas-only",
            "--review-classes",
            "dipper",
            "ltv",
            "microlensing",
            "--atlas-token",
            "test-token",
            *extra,
        ]
    )


def test_atlas_only_uses_publication_cohort_and_dedicated_fetcher_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    review_db = run_dir / "review" / "review.db"
    review_db.parent.mkdir(parents=True)
    review_db.touch()

    cohort = pd.DataFrame(
        [
            {"candidate_id": "C2", "ra": 2.0, "dec": -2.0, "review_bucket": "LTV"},
            {"candidate_id": "C1", "ra": 1.0, "dec": -1.0, "review_bucket": "Dipper"},
            {
                "candidate_id": "C3",
                "ra": 3.0,
                "dec": -3.0,
                "review_bucket": "Microlensing",
            },
        ]
    )
    loader_calls: list[dict[str, object]] = []
    _install_review_cohort_loader(monkeypatch, cohort, loader_calls)

    fetch_calls: list[dict[str, object]] = []

    def fake_query_atlas_forced_phot(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        fetch_calls.append({"df": df.copy(), **kwargs})
        out = df.copy()
        out["atlas_has_phot"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        for column in external_lcs.ATLAS_SUMMARY_COLUMNS[1:]:
            out[column] = pd.NA
        return out

    _install_atlas_fetcher(monkeypatch, fake_query_atlas_forced_phot)

    written: dict[str, object] = {}
    monkeypatch.setattr(
        external_lcs,
        "write_feature_table",
        lambda df, path: written.update({"df": df.copy(), "path": Path(path)}),
    )
    merge_calls: list[dict[str, object]] = []

    def fake_merge(review_db_path, merge_df, *, clear_columns=()):
        merge_calls.append(
            {
                "review_db": Path(review_db_path),
                "frame": merge_df.copy(),
                "clear_columns": tuple(clear_columns),
            }
        )
        return 0

    monkeypatch.setattr(external_lcs, "_merge_into_review_db_with_retries", fake_merge)

    output_path = external_lcs.run(_atlas_only_args(review_db))

    assert loader_calls == [
        {
            "review_db": review_db,
            "buckets": ["Dipper", "LTV", "Microlensing"],
            "only_reviewed": True,
            "publication_only": True,
        }
    ]
    assert len(fetch_calls) == 1
    assert fetch_calls[0]["df"]["candidate_id"].tolist() == ["C1", "C2", "C3"]

    results_root = run_dir / "results"
    lightcurve_dir = results_root / "external_lcs"
    assert fetch_calls[0]["output_dir"] == lightcurve_dir
    assert fetch_calls[0]["results_root"] == results_root
    assert fetch_calls[0]["task_checkpoint"] is None
    assert (fetch_calls[0]["task_checkpoint"] or lightcurve_dir / "atlas_forced_phot_tasks.parquet") == (
        lightcurve_dir / "atlas_forced_phot_tasks.parquet"
    )
    assert output_path == results_root / "atlas_reviewed_events_external_lcs.parquet"
    assert written["path"] == output_path

    assert len(merge_calls) == 1
    assert merge_calls[0]["review_db"] == review_db
    assert merge_calls[0]["frame"].empty
    assert merge_calls[0]["clear_columns"] == external_lcs.ATLAS_SUMMARY_COLUMNS


def test_atlas_only_review_db_defaults_to_three_marked_event_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_db = tmp_path / "run" / "review" / "review.db"
    review_db.parent.mkdir(parents=True)
    review_db.touch()
    loader_calls: list[dict[str, object]] = []
    _install_review_cohort_loader(
        monkeypatch,
        pd.DataFrame(
            [{"candidate_id": "C1", "ra": 1.0, "dec": 2.0, "review_bucket": "Dipper"}]
        ),
        loader_calls,
    )

    args = external_lcs.build_arg_parser().parse_args(
        [str(review_db), "--atlas-only", "--dry-run"]
    )
    external_lcs.run(args)

    assert loader_calls[0]["buckets"] == ["Dipper", "LTV", "Microlensing"]
    assert loader_calls[0]["only_reviewed"] is True
    assert loader_calls[0]["publication_only"] is True


def test_atlas_only_all_candidates_bypasses_publication_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_db = tmp_path / "run" / "review" / "review.db"
    review_db.parent.mkdir(parents=True)
    review_db.touch()
    read_calls: list[object] = []
    rebuilt: list[pd.DataFrame] = []

    def fake_read(_path: Path, *, review_classes=None) -> pd.DataFrame:
        read_calls.append(review_classes)
        return pd.DataFrame(
            {
                "candidate_id": ["C1", "C2"],
                "ra": [1.0, 2.0],
                "dec": [3.0, 4.0],
            }
        )

    def fake_rebuild(df: pd.DataFrame, *_args, **_kwargs) -> pd.DataFrame:
        rebuilt.append(df.copy())
        out = df.copy()
        for column in external_lcs.ATLAS_SUMMARY_COLUMNS:
            out[column] = pd.NA
        return out

    monkeypatch.setattr(external_lcs, "_read_input_candidates", fake_read)
    monkeypatch.setattr(
        external_lcs,
        "rebuild_external_lc_table_from_cache",
        fake_rebuild,
    )
    monkeypatch.setattr(external_lcs, "write_feature_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        external_lcs,
        "_merge_into_review_db_with_retries",
        lambda *_args, **_kwargs: 0,
    )

    args = external_lcs.build_arg_parser().parse_args(
        [str(review_db), "--atlas-only", "--cache-only", "--all-candidates"]
    )
    external_lcs.run(args)

    assert read_calls == [None]
    assert rebuilt[0]["candidate_id"].tolist() == ["C1", "C2"]


def test_atlas_only_dry_run_reports_batches_without_network_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    review_db = run_dir / "review" / "review.db"
    review_db.parent.mkdir(parents=True)
    review_db.touch()

    cohort = pd.DataFrame(
        [
            {
                "candidate_id": f"C{index:03d}",
                "ra": float(index),
                "dec": -float(index),
                "review_bucket": (
                    "Dipper" if index < 40 else "LTV" if index < 80 else "Microlensing"
                ),
            }
            for index in range(101)
        ]
    )
    _install_review_cohort_loader(monkeypatch, cohort, [])

    def fail_network(*_args, **_kwargs):
        raise AssertionError("dry-run must not call the ATLAS API")

    _install_atlas_fetcher(monkeypatch, fail_network)
    monkeypatch.setattr(
        external_lcs,
        "write_feature_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not write the summary table")
        ),
    )
    monkeypatch.setattr(
        external_lcs,
        "_merge_into_review_db_with_retries",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not write the review DB")
        ),
    )

    output_path = external_lcs.run(
        _atlas_only_args(review_db, "--dry-run", "--atlas-batch-size", "100")
    )

    results_root = run_dir / "results"
    lightcurve_dir = results_root / "external_lcs"
    captured = capsys.readouterr().out
    assert "ATLAS dry run: 101 coordinate(s) in 2 batch(es) of at most 100" in captured
    assert f"Task journal: {lightcurve_dir / 'atlas_forced_phot_tasks.parquet'}" in captured
    assert f"Manifest: {results_root / 'external_lc_manifest.parquet'}" in captured
    assert f"Summary table: {output_path}" in captured
    assert output_path == results_root / "atlas_reviewed_events_external_lcs.parquet"
    assert not results_root.exists()


def test_atlas_only_passes_terminal_narrow_frame_and_authoritative_clears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    review_db = run_dir / "review" / "review.db"
    review_db.parent.mkdir(parents=True)
    review_db.touch()

    cohort = pd.DataFrame(
        [
            {"candidate_id": "fetched", "ra": 1.0, "dec": 2.0, "review_bucket": "Dipper"},
            {"candidate_id": "pending", "ra": 3.0, "dec": 4.0, "review_bucket": "LTV"},
            {
                "candidate_id": "no-data",
                "ra": 5.0,
                "dec": 6.0,
                "review_bucket": "Microlensing",
            },
        ]
    )
    _install_review_cohort_loader(monkeypatch, cohort, [])

    def fake_query_atlas_forced_phot(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        out = df.copy()
        summaries = {
            "fetched": (True, 4, 5, 0.4, 0.5),
            "pending": (pd.NA, pd.NA, pd.NA, pd.NA, pd.NA),
            "no-data": (False, 0, 0, pd.NA, pd.NA),
        }
        values = [summaries[str(candidate_id)] for candidate_id in out["candidate_id"]]
        out["atlas_has_phot"] = pd.Series(
            [value[0] for value in values],
            index=out.index,
            dtype="boolean",
        )
        out["atlas_n_det_cyan"] = pd.Series(
            [value[1] for value in values],
            index=out.index,
            dtype="Int64",
        )
        out["atlas_n_det_orange"] = pd.Series(
            [value[2] for value in values],
            index=out.index,
            dtype="Int64",
        )
        out["atlas_cyan_range"] = [value[3] for value in values]
        out["atlas_orange_range"] = [value[4] for value in values]
        out["atlas_preprocess_version"] = [
            "atlas-reduced-direct-v1",
            pd.NA,
            "atlas-reduced-direct-v1",
        ]
        out["atlas_n_raw"] = pd.Series([9, pd.NA, 0], dtype="Int64")
        out["atlas_n_good"] = pd.Series([9, pd.NA, 0], dtype="Int64")
        out["atlas_n_rejected"] = pd.Series([0, pd.NA, 0], dtype="Int64")
        out["tess_n_sectors"] = 99
        return out

    _install_atlas_fetcher(monkeypatch, fake_query_atlas_forced_phot)
    monkeypatch.setattr(external_lcs, "write_feature_table", lambda *_args, **_kwargs: None)

    merge_calls: list[dict[str, object]] = []

    def fake_merge(review_db_path, merge_df, *, clear_columns=()):
        merge_calls.append(
            {
                "review_db": Path(review_db_path),
                "frame": merge_df.copy(),
                "clear_columns": tuple(clear_columns),
            }
        )
        return len(merge_df)

    monkeypatch.setattr(external_lcs, "_merge_into_review_db_with_retries", fake_merge)

    external_lcs.run(_atlas_only_args(review_db))

    assert len(merge_calls) == 1
    merge = merge_calls[0]
    assert merge["review_db"] == review_db
    assert merge["frame"].columns.tolist() == [
        "candidate_id",
        *external_lcs.ATLAS_SUMMARY_COLUMNS,
    ]
    assert merge["frame"]["candidate_id"].tolist() == ["fetched", "no-data"]
    assert "pending" not in set(merge["frame"]["candidate_id"])
    assert "tess_n_sectors" not in merge["frame"].columns
    assert merge["clear_columns"] == external_lcs.ATLAS_SUMMARY_COLUMNS


def test_terminal_merge_leaves_pending_review_db_row_untouched(tmp_path: Path) -> None:
    review_db = tmp_path / "review.db"
    with db_connect(review_db) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "terminal",
                        "atlas_has_phot": True,
                        "atlas_n_det_cyan": 10,
                        "atlas_n_det_orange": 11,
                        "atlas_cyan_range": 1.0,
                        "atlas_orange_range": 1.1,
                        "atlas_preprocess_version": "old",
                        "atlas_n_raw": 11,
                        "atlas_n_good": 11,
                        "atlas_n_rejected": 0,
                    },
                    {
                        "candidate_id": "pending",
                        "atlas_has_phot": True,
                        "atlas_n_det_cyan": 20,
                        "atlas_n_det_orange": 21,
                        "atlas_cyan_range": 2.0,
                        "atlas_orange_range": 2.1,
                        "atlas_preprocess_version": "old",
                        "atlas_n_raw": 22,
                        "atlas_n_good": 21,
                        "atlas_n_rejected": 1,
                    },
                ]
            ),
        )

    fetched = pd.DataFrame(
        [
            {
                "candidate_id": "terminal",
                "atlas_has_phot": False,
                "atlas_n_det_cyan": 0,
                "atlas_n_det_orange": 0,
                "atlas_cyan_range": pd.NA,
                "atlas_orange_range": pd.NA,
                "atlas_preprocess_version": "atlas-reduced-direct-v1",
                "atlas_n_raw": 0,
                "atlas_n_good": 0,
                "atlas_n_rejected": 0,
            },
            {
                "candidate_id": "pending",
                "atlas_has_phot": pd.NA,
                "atlas_n_det_cyan": pd.NA,
                "atlas_n_det_orange": pd.NA,
                "atlas_cyan_range": pd.NA,
                "atlas_orange_range": pd.NA,
                "atlas_preprocess_version": pd.NA,
                "atlas_n_raw": pd.NA,
                "atlas_n_good": pd.NA,
                "atlas_n_rejected": pd.NA,
            },
        ]
    )
    merge_frame = external_lcs._atlas_terminal_merge_frame(fetched)

    updated = external_lcs._merge_into_review_db_with_retries(
        review_db,
        merge_frame,
        clear_columns=external_lcs.ATLAS_SUMMARY_COLUMNS,
    )

    with db_connect(review_db) as conn:
        terminal = get_candidate_payload(conn, "terminal")
        pending = get_candidate_payload(conn, "pending")

    assert updated == 1
    assert terminal["atlas_has_phot"] is False
    assert terminal["atlas_n_det_cyan"] == 0.0
    assert terminal["atlas_n_det_orange"] == 0.0
    assert terminal.get("atlas_cyan_range") in (None, "")
    assert terminal.get("atlas_orange_range") in (None, "")
    assert terminal["atlas_preprocess_version"] == "atlas-reduced-direct-v1"
    assert terminal["atlas_n_raw"] == 0.0
    assert terminal["atlas_n_good"] == 0.0
    assert terminal["atlas_n_rejected"] == 0.0
    assert pending["atlas_has_phot"] is True
    assert pending["atlas_n_det_cyan"] == 20.0
    assert pending["atlas_n_det_orange"] == 21.0
    assert pending["atlas_cyan_range"] == 2.0
    assert pending["atlas_orange_range"] == 2.1
    assert pending["atlas_preprocess_version"] == "old"
    assert pending["atlas_n_raw"] == 22.0
    assert pending["atlas_n_good"] == 21.0
    assert pending["atlas_n_rejected"] == 1.0


def test_atlas_cache_only_restores_header_only_no_data_product(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    lightcurve_dir = results_root / "external_lcs"
    lightcurve_dir.mkdir(parents=True)
    lightcurve_path = lightcurve_dir / "atlas_lc_C1.parquet"
    empty_result = pd.DataFrame(
        columns=["MJD", "m", "dm", "F", "mjd", "mag", "mag_err", "filter"]
    )
    empty_result.attrs["atlas_image_type"] = "reduced"
    empty_result.attrs["atlas_image_types"] = ["reduced"]
    empty_result.to_parquet(lightcurve_path, index=False)
    assert upsert_external_lc_manifest_entry(
        results_root,
        candidate_id="C1",
        source="atlas",
        file_prefix="atlas",
        path=lightcurve_path,
    )

    out = external_lcs.rebuild_external_lc_table_from_cache(
        pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
        lightcurve_dir,
        {"atlas": True},
        results_root=results_root,
    )

    assert not bool(out.loc[0, "atlas_has_phot"])
    assert int(out.loc[0, "atlas_n_det_cyan"]) == 0
    assert int(out.loc[0, "atlas_n_det_orange"]) == 0
    frames = dict(external_lcs._cache_only_source_merge_frames(out, {"atlas": True}))
    assert frames["atlas"]["candidate_id"].tolist() == ["C1"]
