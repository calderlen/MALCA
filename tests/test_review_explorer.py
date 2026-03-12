from __future__ import annotations

from argparse import Namespace

import pandas as pd
import plotly.graph_objects as go

from malca.review import explorer as review_explorer
from malca.review.explore_data import CombinedCandidateData, add_eda_columns
from malca.review.filter_schema import SIDEBAR_GROUPS


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
    assert "custom-graph" in ids
    assert "camera-checklist" in ids
    assert "band-checklist" in ids
    assert "theme-mode" in ids
    assert "sample-size" not in ids
    assert "main-graph" not in ids
    assert "catalog-graph" not in ids


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
