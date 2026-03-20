from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from malca.review import explorer as review_explorer
from malca.review.explore_data import CandidateSourceData, CombinedCandidateData, add_eda_columns
from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.handoff import build_review_command


def test_style_plot_separates_title_from_horizontal_legend() -> None:
    fig = go.Figure()
    fig.update_layout(title="Legend-safe explorer plot")

    review_explorer._style_plot(fig, height=320)

    assert fig.layout.title.text == "Legend-safe explorer plot"
    assert fig.layout.title.x == 0.01
    assert fig.layout.title.y == 0.98
    assert fig.layout.margin.t == 92
    assert fig.layout.legend.y == 1.14


def test_style_plot_supports_light_theme() -> None:
    fig = go.Figure()
    fig.update_layout(title="Light explorer plot")

    review_explorer._style_plot(fig, height=300, theme="white", uirevision="plot:1")

    assert fig.layout.paper_bgcolor == "#f2f6fa"
    assert fig.layout.plot_bgcolor == "#ffffff"
    assert fig.layout.font.family.startswith("Inter")
    assert fig.layout.uirevision == "plot:1"


def test_style_native_explorer_figure_reserves_title_space() -> None:
    fig = go.Figure()
    fig.update_layout(title="Native explorer candidate", margin={"l": 12, "r": 14, "t": 68, "b": 20})

    review_explorer._style_native_explorer_figure(fig)

    assert fig.layout.title.text == "Native explorer candidate"
    assert fig.layout.title.x == 0.01
    assert fig.layout.margin.t == 84
    assert fig.layout.margin.l == 12
    assert fig.layout.margin.r == 14


def test_journal_export_figure_uses_fixed_white_layout() -> None:
    fig = go.Figure()
    fig.update_layout(title="Export me")

    export_fig = review_explorer._journal_export_figure(fig)

    assert export_fig.layout.paper_bgcolor == "white"
    assert export_fig.layout.plot_bgcolor == "white"
    assert export_fig.layout.width == 1400
    assert export_fig.layout.height == 900
    assert export_fig.layout.font.family.startswith("Helvetica")


def test_candidate_scope_from_plot_uses_current_ranges() -> None:
    scope = review_explorer._candidate_scope_from_plot(
        {
            "xaxis.range[0]": 1.0,
            "xaxis.range[1]": 2.5,
            "yaxis.range[0]": 10.0,
            "yaxis.range[1]": 15.0,
        },
        x_metric="period_n_sources",
        y_metric="dipper_score",
        log_flags=[],
    )

    assert scope["mode"] == "view"
    assert scope["x_range"] == [1.0, 2.5]
    assert scope["y_range"] == [10.0, 15.0]


def test_apply_candidate_scope_filters_to_captured_view() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["A", "B", "C"],
            "period_n_sources": [1.0, 2.0, 3.0],
            "dipper_score": [8.0, 12.0, 16.0],
        }
    )

    scoped = review_explorer._apply_candidate_scope(
        frame,
        scope_state={
            "mode": "view",
            "x_metric": "period_n_sources",
            "y_metric": "dipper_score",
            "log_x": False,
            "log_y": False,
            "x_range": [1.5, 3.1],
            "y_range": [10.0, 20.0],
        },
        x_metric="period_n_sources",
        y_metric="dipper_score",
        log_flags=[],
    )

    assert scoped["candidate_key"].tolist() == ["B", "C"]


def test_main_schedules_browser_open(monkeypatch) -> None:
    opened: list[str] = []
    timer_calls: list[tuple[float, object]] = []
    run_calls: list[tuple[str, int, bool]] = []

    class _Parser:
        def parse_args(self):
            return Namespace(
                source=None,
                source_glob=None,
                source_kind=None,
                plot_dir=None,
                host="127.0.0.1",
                port=8062,
                debug=False,
            )

    class _Timer:
        def __init__(self, delay, callback):
            timer_calls.append((delay, callback))

        def start(self):
            timer_calls[-1][1]()

    class _App:
        def run(self, host: str, port: int, debug: bool):
            run_calls.append((host, port, debug))

    monkeypatch.setattr(review_explorer, "build_arg_parser", lambda: _Parser())
    monkeypatch.setattr(review_explorer, "_resolve_sources", lambda _args: [])
    monkeypatch.setattr(review_explorer, "load_combined_source_data", lambda **_kwargs: type("Combined", (), {"df": object(), "sources": []})())
    monkeypatch.setattr(review_explorer, "add_eda_columns", lambda df: df)
    monkeypatch.setattr(review_explorer, "build_explorer_app", lambda *args, **kwargs: _App())
    monkeypatch.setattr(review_explorer, "Timer", _Timer)
    monkeypatch.setattr(review_explorer.webbrowser, "open", lambda url: opened.append(url))

    review_explorer.main()

    assert opened == ["http://127.0.0.1:8062"]
    assert timer_calls and timer_calls[0][0] == 0.1
    assert run_calls == [("127.0.0.1", 8062, False)]


def test_build_explorer_app_uses_custom_plot_sidebar_layout() -> None:
    combined = CombinedCandidateData(df=pd.DataFrame(), sources=[], key_lookup={}, id_lookup={})
    app = review_explorer.build_explorer_app(combined, host="127.0.0.1", port=8062)

    def _collect_ids(node) -> set[str]:
        found: set[str] = set()
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            found.add(node_id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                found.update(_collect_ids(child))
        elif children is not None:
            found.update(_collect_ids(children))
        return found

    ids = _collect_ids(app.layout)

    assert "explorer-shell" in ids
    assert "explorer-sidebar" in ids
    assert "explorer-sidebar-toggle" in ids
    assert "explorer-keyboard-input" in ids
    assert "explorer-keyboard-init" in ids
    assert "explorer-help-open" in ids
    assert "explorer-help-modal" in ids
    assert "custom-graph" in ids
    assert "export-native-pdf-btn" in ids
    assert "camera-checklist" in ids
    assert "band-checklist" in ids
    assert "theme-mode" in ids
    assert "saved-explorer-gui-state" in ids
    assert "save-explorer-gui-state-btn" in ids
    assert "explorer-review-overrides" in ids
    assert "explorer-review-save-request" in ids
    assert "explorer-review-launch-url" in ids
    assert "explorer-review-launch-pending" in ids
    assert "explorer-review-launch-opened" in ids
    assert "explorer-review-class" in ids
    assert "explorer-review-confidence" in ids
    assert "explorer-review-followup" in ids
    assert "explorer-review-notes" in ids
    assert "explorer-review-save-btn" in ids
    assert "sample-size" not in ids
    assert "main-graph" not in ids
    assert "catalog-graph" not in ids
    assert "explorer-main-splitter" not in ids


def test_table_rows_returns_all_filtered_candidates() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": [f"cand-{i}" for i in range(55)],
            "candidate_id": [f"ASASSN-{i}" for i in range(55)],
            "dipper_score": list(range(55)),
        }
    )

    rows = review_explorer._table_rows(frame, "dipper_score")

    assert len(rows) == 55
    assert rows[0]["candidate_key"] == "cand-54"
    assert rows[-1]["candidate_key"] == "cand-0"


def test_explorer_shortcut_action_toggles_class() -> None:
    action = review_explorer._explorer_shortcut_action(
        key="d",
        selected_key="cand-1",
        table_data=[{"candidate_key": "cand-1"}],
        event_class="unclassified",
        interest_score=None,
        followup_value=[],
        notes="",
        save_request={"nonce": 0},
    )

    assert action["event_class"] == "dipper"
    assert action["status"] == "Class: dipper"


def test_explorer_shortcut_action_sets_confidence_and_saves() -> None:
    action = review_explorer._explorer_shortcut_action(
        key="3",
        selected_key="cand-2",
        table_data=[{"candidate_key": "cand-2"}],
        event_class="flare",
        interest_score=None,
        followup_value=["followup"],
        notes="check phase fold",
        save_request={"nonce": 2},
    )

    assert action["interest_score"] == 3
    assert action["status"] == "✓ Confidence: 3"
    assert action["save_request"] == {
        "nonce": 3,
        "candidate_key": "cand-2",
        "interest_score": 3,
        "event_class": "flare",
        "needs_followup": True,
        "notes": "check phase fold",
        "increment_pass": False,
        "event_type": "keyboard",
    }


def test_explorer_shortcut_action_enter_saves_and_advances() -> None:
    action = review_explorer._explorer_shortcut_action(
        key="Enter",
        selected_key="cand-2",
        table_data=[{"candidate_key": "cand-1"}, {"candidate_key": "cand-2"}, {"candidate_key": "cand-3"}],
        event_class="ltv",
        interest_score=4,
        followup_value=[],
        notes="strong trend",
        save_request={"nonce": 5},
    )

    assert action["selected_key"] == "cand-3"
    assert action["status"] == "✓ Saved + Next →"
    assert action["save_request"]["candidate_key"] == "cand-2"
    assert action["save_request"]["increment_pass"] is True


def test_build_review_command_disables_child_browser_auto_open(monkeypatch) -> None:
    monkeypatch.setattr("malca.review.handoff._find_open_port", lambda preferred_port, host="127.0.0.1": 8123)
    command, url = build_review_command(db_path="/tmp/review.db", candidate="cand-1")

    assert "--no-browser" in command
    assert "--candidate" in command
    assert url == "http://127.0.0.1:8123"


def test_filter_controls_and_selected_candidate_are_persistent() -> None:
    bool_control = review_explorer._bool_filter_control("failed_any")
    bool_dropdown = bool_control.children[1]
    assert bool_dropdown.persistence is True
    assert bool_dropdown.persistence_type == "local"

    num_control = review_explorer._num_filter_control("dipper_score")
    num_min = num_control.children[1].children[0]
    num_max = num_control.children[1].children[1]
    assert num_min.persistence is True
    assert num_min.persistence_type == "local"
    assert num_max.persistence is True
    assert num_max.persistence_type == "local"

    text_control = review_explorer._text_filter_control(pd.DataFrame({"final_class": ["dipper"]}), "final_class")
    text_dropdown = text_control.children[1]
    assert text_dropdown.persistence is True
    assert text_dropdown.persistence_type == "local"

    select_control = review_explorer._select_filter_control(pd.DataFrame({"source_label": ["run-a"]}), "source_label")
    select_dropdown = select_control.children[1]
    assert select_dropdown.persistence is True
    assert select_dropdown.persistence_type == "local"

    combined = CombinedCandidateData(df=pd.DataFrame(), sources=[], key_lookup={}, id_lookup={})
    app = review_explorer.build_explorer_app(combined, host="127.0.0.1", port=8062)

    def _find_by_id(node, target_id):
        if getattr(node, "id", None) == target_id:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                found = _find_by_id(child, target_id)
                if found is not None:
                    return found
        elif children is not None:
            return _find_by_id(children, target_id)
        return None

    selected_store = _find_by_id(app.layout, "selected-key-store")
    source_filter = _find_by_id(app.layout, "source-filter")
    query_input = _find_by_id(app.layout, "query-input")
    only_unreviewed = _find_by_id(app.layout, "only-unreviewed")
    require_failed = _find_by_id(app.layout, "require-failed-any-false")

    assert selected_store.storage_type == "local"
    assert source_filter.persistence is True
    assert source_filter.persistence_type == "local"
    assert query_input.persistence is True
    assert query_input.persistence_type == "local"
    assert only_unreviewed.persistence is True
    assert only_unreviewed.persistence_type == "local"
    assert require_failed.persistence is True
    assert require_failed.persistence_type == "local"


def test_explorer_select_filters_pick_up_catalog_type_aliases() -> None:
    frame = add_eda_columns(
        pd.DataFrame(
            {
                "candidate_id": ["A1", "A2"],
                "period_asassn_var_class": ["EA", "DSCT"],
                "period_ztf_periodic_class": ["EW", "RR"],
                "asassn_var_type": ["", ""],
                "ztf_var_type": ["", ""],
            }
        )
    )

    asassn_control = review_explorer._select_filter_control(frame, "asassn_var_type")
    ztf_control = review_explorer._select_filter_control(frame, "ztf_var_type")

    asassn_options = [opt["value"] for opt in asassn_control.children[1].options]
    ztf_options = [opt["value"] for opt in ztf_control.children[1].options]

    assert asassn_options == ["DSCT", "EA"]
    assert ztf_options == ["EW", "RR"]


def test_filter_schema_exposes_pm_and_fail_flags() -> None:
    groups = {name: items for name, items in SIDEBAR_GROUPS}

    assert ("num", "pm_total") in groups["Vetting"]
    assert ("bool", "high_pm_flag") in groups["Vetting"]
    assert ("bool", "failed_any") in groups["Fail Flags"]
    assert ("bool", "period_conflict_flag") in groups["Fail Flags"]


def test_build_auto_filter_groups_exposes_unlisted_columns() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A1", "A2"],
            "custom_bad_flag": [True, False],
            "custom_score": [1.5, 2.5],
            "custom_label": ["foo", "bar"],
        }
    )

    groups = {name: items for name, items in review_explorer._build_auto_filter_groups(frame)}

    assert ("bool", "custom_bad_flag") in groups["Additional Flags"]
    assert ("num", "custom_score") in groups["Additional Numeric"]
    assert ("select", "custom_label") in groups["Additional Categorical"]


def test_explorer_state_db_path_requires_single_db_source() -> None:
    empty = pd.DataFrame()
    db_source = CandidateSourceData(
        source_path=Path("/tmp/demo/review.db"),
        source_kind="db",
        source_label="demo",
        df=empty,
        lookup={},
        default_plot_dir=None,
    )
    csv_source = CandidateSourceData(
        source_path=Path("/tmp/demo.csv"),
        source_kind="csv",
        source_label="csv",
        df=empty,
        lookup={},
        default_plot_dir=None,
    )
    second_db = CandidateSourceData(
        source_path=Path("/tmp/other/review.db"),
        source_kind="db",
        source_label="other",
        df=empty,
        lookup={},
        default_plot_dir=None,
    )

    combined = CombinedCandidateData(df=empty, sources=[db_source, csv_source], key_lookup={}, id_lookup={})
    assert review_explorer._explorer_state_db_path(combined) == Path("/tmp/demo/review.db")

    mixed = CombinedCandidateData(df=empty, sources=[db_source, second_db], key_lookup={}, id_lookup={})
    assert review_explorer._explorer_state_db_path(mixed) is None


def test_explorer_advanced_ui_values_roundtrip() -> None:
    saved = {
        "advanced": {
            "bool": {"failed_any": "False"},
            "num": {"dipper_score": {"min": 3.0, "max": 9.0}},
            "text": {"final_class": "dipper"},
            "select": {"source_label": ["run-a"]},
            "only_unreviewed": True,
            "require_failed_any_false": False,
        }
    }
    bool_ids = [{"col": "failed_any"}]
    num_min_ids = [{"col": "dipper_score"}]
    num_max_ids = [{"col": "dipper_score"}]
    text_ids = [{"col": "final_class"}]
    select_ids = [{"col": "source_label"}]

    (
        bool_values,
        num_min_values,
        num_max_values,
        text_values,
        select_values,
        only_unreviewed,
        require_failed,
    ) = review_explorer._explorer_advanced_ui_values_from_state(
        saved,
        bool_ids=bool_ids,
        num_min_ids=num_min_ids,
        num_max_ids=num_max_ids,
        text_ids=text_ids,
        select_ids=select_ids,
    )

    assert bool_values == ["False"]
    assert num_min_values == [3.0]
    assert num_max_values == [9.0]
    assert text_values == ["dipper"]
    assert select_values == [["run-a"]]
    assert only_unreviewed == ["yes"]
    assert require_failed == []


def test_apply_review_overrides_updates_review_state_columns() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["cand-1", "cand-2"],
            "candidate_id": ["A1", "A2"],
            "event_class": ["", ""],
            "review_event_class": ["", ""],
            "status": ["unreviewed", "unreviewed"],
            "interest_score": [None, None],
        }
    )

    updated = review_explorer._apply_review_overrides(
        frame,
        {
            "cand-2": {
                "event_class": "dipper",
                "status": "reviewed",
                "interest_score": 4,
                "notes": "looks real",
                "review_pass": 2,
            }
        },
    )

    row = updated.loc[updated["candidate_key"] == "cand-2"].iloc[0]
    assert row["event_class"] == "dipper"
    assert row["status"] == "reviewed"
    assert row["interest_score"] == 4
    assert row["review_label"] == "dipper"
    assert bool(row["is_reviewed"]) is True
    assert bool(row["is_reviewed_dipper"]) is True


def test_record_review_state_normalizes_unknown_class_and_score() -> None:
    state = review_explorer._record_review_state(
        {
            "interest_score": 9,
            "event_class": "mystery",
            "review_pass": 0,
            "status": "odd",
            "notes": None,
        }
    )

    assert state["interest_score"] == 4
    assert state["event_class"] == "other"
    assert state["review_pass"] == 1
    assert state["status"] == "reviewed"
    assert state["notes"] == ""
