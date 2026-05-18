from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def _install_explorer_import_stubs() -> None:
    if "celerite2" not in sys.modules and importlib.util.find_spec("celerite2") is None:
        fake_celerite2 = types.ModuleType("celerite2")
        fake_terms = types.ModuleType("celerite2.terms")

        class _FakeGaussianProcess:
            def __init__(self, *args, **kwargs):
                pass

        class _FakeTerm:
            def __init__(self, *args, **kwargs):
                pass

            def __add__(self, other):
                return self

        fake_terms.SHOTerm = _FakeTerm
        fake_terms.RealTerm = _FakeTerm
        fake_celerite2.GaussianProcess = _FakeGaussianProcess
        fake_celerite2.terms = fake_terms
        sys.modules["celerite2"] = fake_celerite2
        sys.modules["celerite2.terms"] = fake_terms

    if "multiprocess" not in sys.modules and importlib.util.find_spec("multiprocess") is None:
        fake_multiprocess = types.ModuleType("multiprocess")
        fake_multiprocess.get_all_start_methods = lambda: ["spawn"]
        fake_multiprocess.set_start_method = lambda *args, **kwargs: None
        sys.modules["multiprocess"] = fake_multiprocess


_install_explorer_import_stubs()

from malca.review.explore_data import (
    add_eda_columns,
    CandidateSourceData,
    CombinedCandidateData,
    find_candidate_key,
    get_candidate_record_by_key,
    infer_plot_dir_from_source,
    infer_source_kind,
    load_review_db,
    load_combined_source_data,
    load_source_data,
)
from malca.review.explorer import (
    _resolve_initial_candidate_key,
    _resolve_sources,
    _review_db_paths_from_frame,
    build_arg_parser,
    build_explorer_app,
)
from malca.review import explorer as review_explorer


def _write_review_db(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)")
        for row in rows:
            conn.execute(
                "INSERT INTO candidates VALUES (?, ?, ?)",
                (
                    row["candidate_id"],
                    row.get("source_path", ""),
                    json.dumps(row),
                ),
            )
        conn.commit()


def _collect_layout_text(node: object) -> list[str]:
    text: list[str] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if isinstance(item, str):
            text.append(item)
            return
        if item is None or isinstance(item, (int, float, bool)):
            return
        walk(getattr(item, "children", None))

    walk(node)
    return text


def _collect_layout_ids(node: object) -> list[object]:
    ids: list[object] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        cid = getattr(item, "id", None)
        if cid is not None:
            ids.append(cid)
        walk(getattr(item, "children", None))

    walk(node)
    return ids


def _collect_graph_configs(node: object) -> list[dict]:
    configs: list[dict] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            return
        config = getattr(item, "config", None)
        if isinstance(config, dict):
            configs.append(config)
        walk(getattr(item, "children", None))

    walk(node)
    return configs


def test_load_combined_source_data_adds_candidate_keys(tmp_path: Path) -> None:
    db1 = tmp_path / "output" / "runs" / "run_a" / "review" / "review.db"
    db2 = tmp_path / "output" / "runs" / "run_b" / "review" / "review.db"
    _write_review_db(db1, [{"candidate_id": "A1", "asas_sn_id": "A1", "dipper_score": 12.0}])
    _write_review_db(db2, [{"candidate_id": "A1", "asas_sn_id": "B1", "dipper_score": 9.0}])

    combined = load_combined_source_data(sources=[db1, db2])

    assert len(combined.df) == 2
    assert combined.df["candidate_key"].nunique() == 2
    assert find_candidate_key(combined, "A1") is not None


def test_explorer_header_uses_compact_source_row_count(tmp_path: Path) -> None:
    row_count = 13_743
    frame = pd.DataFrame(
        {
            "candidate_key": [f"C{i}" for i in range(row_count)],
            "candidate_id": [f"C{i}" for i in range(row_count)],
            "source_label": ["demo"] * row_count,
            "dipper_score": [0.0] * row_count,
        }
    )
    source = CandidateSourceData(
        source_path=tmp_path / "review.db",
        source_kind="db",
        source_label="demo",
        df=frame,
        lookup={"C0": 0},
        default_plot_dir=None,
    )
    combined = CombinedCandidateData(
        df=frame,
        sources=[source],
        key_lookup={"C0": 0},
        id_lookup={"C0": ["C0"]},
    )

    app = build_explorer_app(combined, host="127.0.0.1", port=8050)
    text = _collect_layout_text(app.layout)

    assert "MALCA Explorer" not in text
    assert "[1/13,743]" in text


def test_explorer_layout_has_publication_export_targets(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["C0"],
            "candidate_id": ["C0"],
            "source_label": ["demo"],
            "dipper_score": [0.0],
        }
    )
    source = CandidateSourceData(
        source_path=tmp_path / "review.db",
        source_kind="db",
        source_label="demo",
        df=frame,
        lookup={"C0": 0},
        default_plot_dir=None,
    )
    combined = CombinedCandidateData(
        df=frame,
        sources=[source],
        key_lookup={"C0": 0},
        id_lookup={"C0": ["C0"]},
    )

    app = build_explorer_app(combined, host="127.0.0.1", port=8050)
    ids = _collect_layout_ids(app.layout)

    assert "explorer-dustycult-download" in ids
    assert "explorer-mini-plot-download" in ids
    assert "dustycult-export-fit-btn" in ids
    assert "dustycult-export-occulter-btn" in ids


def test_explorer_layout_graphs_disable_plotly_image_export(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["C0"],
            "candidate_id": ["C0"],
            "source_label": ["demo"],
            "dipper_score": [0.0],
        }
    )
    source = CandidateSourceData(
        source_path=tmp_path / "review.db",
        source_kind="db",
        source_label="demo",
        df=frame,
        lookup={"C0": 0},
        default_plot_dir=None,
    )
    combined = CombinedCandidateData(
        df=frame,
        sources=[source],
        key_lookup={"C0": 0},
        id_lookup={"C0": ["C0"]},
    )

    app = build_explorer_app(combined, host="127.0.0.1", port=8050)
    configs = _collect_graph_configs(app.layout)

    assert configs
    assert all("toImage" in config.get("modeBarButtonsToRemove", []) for config in configs)


def test_explorer_mini_plot_export_passes_button_ids_and_clicks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["C0"],
            "candidate_id": ["cand-0"],
            "source_label": ["demo"],
            "dipper_score": [0.0],
        }
    )
    source = CandidateSourceData(
        source_path=tmp_path / "review.db",
        source_kind="db",
        source_label="demo",
        df=frame,
        lookup={"C0": 0},
        default_plot_dir=None,
    )
    combined = CombinedCandidateData(
        df=frame,
        sources=[source],
        key_lookup={"C0": 0},
        id_lookup={"cand-0": ["C0"]},
    )
    app = build_explorer_app(combined, host="127.0.0.1", port=8050)
    callback = next(
        spec["callback"].__wrapped__
        for key, spec in app.callback_map.items()
        if "explorer-mini-plot-download" in key
    )
    seen = {}

    def fake_export(triggered_id, button_ids, clicks, graph_ids, figures, candidate_id):
        seen.update(
            triggered_id=triggered_id,
            button_ids=button_ids,
            clicks=clicks,
            graph_ids=graph_ids,
            figures=figures,
            candidate_id=candidate_id,
        )
        return "download", "status"

    triggered_id = {"type": "mini-plot-export-btn", "panel": "external", "name": "cmd"}
    button_id = {"type": "mini-plot-export-btn", "panel": "external", "name": "cmd"}
    graph_id = {"type": "mini-plot-export-graph", "panel": "external", "name": "cmd"}
    monkeypatch.setattr(review_explorer, "_export_mini_plot_pdf_from_state", fake_export)
    monkeypatch.setattr(
        review_explorer.dash,
        "callback_context",
        types.SimpleNamespace(triggered_id=triggered_id, inputs_list=[[{"id": button_id}]]),
    )

    data, status = callback([1], [graph_id], [{"data": []}], "C0", {})

    assert (data, status) == ("download", "status")
    assert seen == {
        "triggered_id": triggered_id,
        "button_ids": [button_id],
        "clicks": [1],
        "graph_ids": [graph_id],
        "figures": [{"data": []}],
        "candidate_id": "cand-0",
    }


def test_review_explore_parser_uses_review_db_cli(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    _write_review_db(db_path, [{"candidate_id": "A1"}])

    args = build_arg_parser().parse_args(["--review-db", str(db_path), "--no-browser"])

    assert _resolve_sources(args) == [db_path.resolve()]
    assert args.no_browser is True
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--source", str(db_path)])


def test_review_db_paths_from_frame_requires_existing_db_sources(tmp_path: Path) -> None:
    db1 = tmp_path / "run_a" / "review" / "review.db"
    db2 = tmp_path / "run_b" / "review" / "review.db"
    _write_review_db(db1, [{"candidate_id": "A1"}])
    _write_review_db(db2, [{"candidate_id": "B1"}])
    missing = tmp_path / "missing.db"

    frame = pd.DataFrame(
        {
            "source_file": [
                str(db1),
                str(db1),
                str(db2),
                str(missing),
                str(tmp_path / "candidates.parquet"),
                "",
            ]
        }
    )

    assert _review_db_paths_from_frame(frame) == [db1.resolve(), db2.resolve()]


def test_add_eda_columns_builds_proxy_fields(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_a" / "review" / "review.db"
    _write_review_db(
        db,
        [
            {
                "candidate_id": "C1",
                "asas_sn_id": "C1",
                "catalog_match": True,
                "period_consensus_agree": True,
                "period_n_sources": 3,
                "dip_run_count": 4,
                "dipper_n_valid_dips": 12,
                "vetting_likely_known": True,
                "dipper_score": 10,
            },
            {
                "candidate_id": "C2",
                "asas_sn_id": "C2",
                "catalog_match": False,
                "period_consensus_agree": False,
                "period_n_sources": 0,
                "dip_run_count": 1,
                "dip_is_single_event": True,
                "dipper_n_valid_dips": 8,
                "vetting_likely_known": False,
                "dipper_score": 15,
            },
        ],
    )

    combined = load_combined_source_data(sources=[db])
    frame = add_eda_columns(combined.df)

    row1 = frame.loc[frame["candidate_id"] == "C1"].iloc[0]
    row2 = frame.loc[frame["candidate_id"] == "C2"].iloc[0]
    assert bool(row1["known_periodic_catalog"]) is True
    assert bool(row1["strong_catalog_period"]) is True
    assert bool(row2["proxy_oneoff_dipper"]) is True
    assert frame.attrs["default_target_col"] == "proxy_oneoff_dipper"


def test_add_eda_columns_fills_catalog_type_aliases(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_alias" / "review" / "review.db"
    _write_review_db(
        db,
        [
            {
                "candidate_id": "A1",
                "period_asassn_var_class": "EA",
                "period_ztf_periodic_class": "EW",
                "asassn_var_type": "",
                "ztf_var_type": "",
            },
            {
                "candidate_id": "A2",
                "period_asassn_var_class": "",
                "period_ztf_periodic_class": "",
                "asassn_var_type": "DSCT",
                "ztf_var_type": "RR",
            },
        ],
    )

    combined = load_combined_source_data(sources=[db])
    frame = add_eda_columns(combined.df)

    row1 = frame.loc[frame["candidate_id"] == "A1"].iloc[0]
    row2 = frame.loc[frame["candidate_id"] == "A2"].iloc[0]

    assert row1["asassn_var_type"] == "EA"
    assert row1["ztf_var_type"] == "EW"
    assert row2["asassn_var_type"] == "DSCT"
    assert row2["ztf_var_type"] == "RR"


def test_get_candidate_record_by_key_round_trips(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_a" / "review" / "review.db"
    _write_review_db(db, [{"candidate_id": "C3", "asas_sn_id": "C3", "dipper_score": 4.0}])
    combined = load_combined_source_data(sources=[db])
    key = str(combined.df.iloc[0]["candidate_key"])

    record = get_candidate_record_by_key(combined, key)

    assert record is not None
    assert record["candidate_id"] == "C3"


def test_load_review_db_merges_review_columns(tmp_path: Path) -> None:
    db = tmp_path / "review.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)")
        conn.execute(
            "CREATE TABLE reviews (candidate_id TEXT, interest_score INTEGER, event_class TEXT, review_pass INTEGER, notes TEXT, status TEXT, reviewer TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            ("C4", "/tmp/run", json.dumps({"candidate_id": "C4", "dipper_score": 8.0})),
        )
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("C4", 3, "dipper", 2, "note", "reviewed", "tester", "2026-03-10T00:00:00Z"),
        )
        conn.commit()

    df = load_review_db(db)

    assert df.loc[0, "interest_score"] == 3
    assert df.loc[0, "event_class"] == "dipper"
    assert df.loc[0, "status"] == "reviewed"


def test_infer_source_kind() -> None:
    assert infer_source_kind("/tmp/review.db") == "db"
    assert infer_source_kind("/tmp/candidates.parquet") == "parquet"
    with pytest.raises(ValueError):
        infer_source_kind("/tmp/candidates.csv")


def test_infer_plot_dir_from_source_for_run_local_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    plot_dir = run_dir / "plots"
    review_dir = run_dir / "review"
    results_dir = run_dir / "results"
    plot_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    db_path = review_dir / "review.db"
    db_path.touch()
    parquet_path = results_dir / "lc_events_vetted.parquet"
    parquet_path.touch()

    assert infer_plot_dir_from_source(db_path) == plot_dir.resolve()
    assert infer_plot_dir_from_source(parquet_path) == plot_dir.resolve()


def test_load_source_data_builds_id_lookup(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)"
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, ?)",
            (
                "CAND-2",
                "/tmp/run",
                json.dumps({"candidate_id": "CAND-2", "asas_sn_id": "ASAS-2", "gaia_id": "1234"}),
            ),
        )
        conn.commit()

    source = load_source_data(db_path)

    assert source.lookup["CAND-2"] == 0
    assert source.lookup["ASAS-2"] == 0
    assert source.lookup["1234"] == 0
    assert source.default_candidate_id == "CAND-2"


def test_resolve_initial_candidate_key_prefers_cli_candidate(tmp_path: Path) -> None:
    db = tmp_path / "output" / "runs" / "run_init" / "review" / "review.db"
    _write_review_db(
        db,
        [
            {"candidate_id": "C1", "asas_sn_id": "ASAS-1", "dipper_score": 1.0},
            {"candidate_id": "C2", "asas_sn_id": "ASAS-2", "dipper_score": 2.0},
        ],
    )
    combined = load_combined_source_data(sources=[db])

    resolved = _resolve_initial_candidate_key(combined, candidate="ASAS-2")

    assert resolved == str(combined.df.loc[combined.df["candidate_id"] == "C2", "candidate_key"].iloc[0])
