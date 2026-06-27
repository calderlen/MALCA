from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.review.eda_panel import (
    candidate_ids_from_eda_table_context,
    candidate_ids_from_plotly_selection,
    eda_plot_row_counts,
    eda_publication_figure,
    eda_scatter_figure,
    eda_table_page,
    eda_table_rows,
    load_review_eda_frame,
    queue_eda_frame,
    selected_candidate_row_style,
    selected_row_style,
)


def _write_review_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT, source_path TEXT, payload_json TEXT)")
        conn.execute(
            "CREATE TABLE reviews (candidate_id TEXT, interest_score REAL, event_class TEXT, review_pass INTEGER, notes TEXT, status TEXT, reviewer TEXT, updated_at TEXT)"
        )
        for candidate_id, score, period_sources, dip_runs in (
            ("A", 3.0, 2, 1),
            ("B", 1.0, 0, 4),
            ("C", 4.0, 1, 2),
        ):
            payload = {
                "candidate_id": candidate_id,
                "asas_sn_id": f"ASAS-{candidate_id}",
                "dipper_score": score * 2,
                "period_n_sources": period_sources,
                "dip_run_count": dip_runs,
            }
            conn.execute(
                "INSERT INTO candidates VALUES (?, ?, ?)",
                (candidate_id, str(path.parent.parent), json.dumps(payload)),
            )
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("B", 2.0, "dipper", 1, "note", "reviewed", "tester", "2026-01-01"),
        )
        conn.commit()


def test_load_review_eda_frame_adds_review_and_proxy_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "run" / "review" / "review.db"
    _write_review_db(db_path)

    frame = load_review_eda_frame(db_path, "sig1")

    assert set(frame["candidate_id"]) == {"A", "B", "C"}
    assert "periodic_evidence_bucket" in frame.columns
    assert "is_reviewed" in frame.columns
    assert frame.loc[frame["candidate_id"] == "B", "event_class"].iloc[0] == "dipper"


def test_queue_eda_frame_preserves_queue_order(tmp_path: Path) -> None:
    db_path = tmp_path / "run" / "review" / "review.db"
    _write_review_db(db_path)
    frame = load_review_eda_frame(db_path, "sig2")

    queue_frame = queue_eda_frame(frame, ["C", "A"])

    assert queue_frame["candidate_id"].tolist() == ["C", "A"]


def test_eda_table_rows_tolerate_missing_optional_columns() -> None:
    rows = eda_table_rows(pd.DataFrame({"candidate_id": ["A"], "dipper_score": [1.5]}))

    assert rows == [{"candidate_id": "A", "dipper_score": 1.5, "id": "A"}]


def test_eda_table_rows_keep_candidate_key_payload() -> None:
    rows = eda_table_rows(pd.DataFrame({"candidate_id": ["A"], "candidate_key": ["K"], "dipper_score": [1.5]}))

    assert rows == [{"candidate_id": "A", "dipper_score": 1.5, "candidate_key": "K", "id": "A"}]


def test_eda_table_page_slices_server_side() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C"],
            "dipper_score": [1.0, 3.0, 2.0],
        }
    )

    rows, page_count, total = eda_table_page(frame, page_current=1, page_size=2)

    assert [row["candidate_id"] for row in rows] == ["C"]
    assert page_count == 2
    assert total == 3


def test_eda_table_page_sorts_before_paging() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C"],
            "dipper_score": [1.0, 3.0, 2.0],
        }
    )

    rows, page_count, total = eda_table_page(
        frame,
        page_current=0,
        page_size=2,
        sort_by=[{"column_id": "dipper_score", "direction": "desc"}],
    )

    assert [row["candidate_id"] for row in rows] == ["B", "C"]
    assert page_count == 2
    assert total == 3


def test_eda_table_page_filters_before_paging() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["Alpha", "Beta", "Gamma"],
            "dipper_score": [1.0, 3.0, 2.0],
        }
    )

    rows, page_count, total = eda_table_page(
        frame,
        page_current=0,
        page_size=10,
        sort_by=[{"column_id": "dipper_score", "direction": "asc"}],
        filter_query='{candidate_id} contains "a" && {dipper_score} >= 2',
    )

    assert [row["candidate_id"] for row in rows] == ["Gamma", "Beta"]
    assert page_count == 1
    assert total == 2


def test_candidate_ids_from_eda_table_context_prefers_row_id() -> None:
    candidate_ids = candidate_ids_from_eda_table_context(
        {"row": 0, "row_id": "C-from-row-id"},
        [{"candidate_id": "C-from-viewport"}],
        [{"candidate_id": "C-from-virtual"}],
        [{"candidate_id": "C-from-table"}],
    )

    assert candidate_ids[0] == "C-from-row-id"


def test_candidate_ids_from_eda_table_context_uses_current_viewport_page() -> None:
    virtual_rows = [
        {"candidate_id": f"C{i}"}
        for i in range(24)
    ]
    viewport_rows = virtual_rows[12:24]

    candidate_ids = candidate_ids_from_eda_table_context(
        {"row": 2},
        viewport_rows,
        virtual_rows,
        virtual_rows,
        page_current=1,
        page_size=12,
    )

    assert candidate_ids[0] == "C14"
    assert "C2" not in candidate_ids[:2]


def test_candidate_ids_from_eda_table_context_uses_absolute_page_fallback() -> None:
    virtual_rows = [
        {"candidate_id": f"C{i}"}
        for i in range(24)
    ]

    candidate_ids = candidate_ids_from_eda_table_context(
        {"row": 2},
        None,
        virtual_rows,
        virtual_rows,
        page_current=1,
        page_size=12,
    )

    assert candidate_ids[0] == "C14"


def test_candidate_ids_from_plotly_selection_reads_customdata_once() -> None:
    selected = {
        "points": [
            {"customdata": ["B"]},
            {"customdata": ["C", "ignored"]},
            {"customdata": ["B"]},
            {"customdata": "D"},
            {"text": "candidate_id: E<br>dipper_score: 1.0"},
            {"x": 1.0},
        ]
    }

    assert candidate_ids_from_plotly_selection(selected) == ["B", "C", "D", "E"]


def test_candidate_ids_from_plotly_selection_uses_trace_point_indices() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C"],
            "dipper_score": [1.0, 2.0, 3.0],
            "period_n_sources": [0, 1, 2],
            "periodic_evidence_bucket": ["none", "one", "one"],
        }
    )
    fig = eda_scatter_figure(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        color_metric="periodic_evidence_bucket",
    )
    traces_by_name = {str(trace.name): idx for idx, trace in enumerate(fig.data)}
    selected = {
        "points": [
            {"curveNumber": traces_by_name["none"], "pointNumber": 0},
            {"curveNumber": traces_by_name["one"], "pointNumber": 1},
        ]
    }

    assert candidate_ids_from_plotly_selection(selected, fig.to_dict()) == ["A", "C"]


def test_eda_scatter_figure_highlights_selected_candidate() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "dipper_score": [1.0, 3.0],
            "period_n_sources": [0, 2],
        }
    )

    fig = eda_scatter_figure(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        selected_candidate_id="B",
        theme="black",
    )

    assert len(fig.data) >= 2
    assert fig.data[-1].name == "current"
    assert fig.data[-1].customdata[0][0] == "B"
    assert fig.data[-1].showlegend is False
    assert fig.layout.title.text in (None, "")
    assert fig.layout.xaxis.title.text == "period_n_sources"
    assert fig.layout.yaxis.title.text == "dipper_score"


def test_eda_scatter_figure_current_trace_is_empty_for_absent_candidate() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "dipper_score": [1.0, 3.0],
            "period_n_sources": [0, 2],
        }
    )

    fig = eda_scatter_figure(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        selected_candidate_id="Z",
    )

    assert fig.data[-1].name == "current"
    assert len(fig.data[-1].x) == 0
    assert len(fig.data[-1].customdata) == 0


def test_eda_scatter_figure_uses_plain_list_trace_data() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "dipper_score": [1.0, 3.0],
            "period_n_sources": [0, 2],
        }
    )

    fig = eda_scatter_figure(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        selected_candidate_id="B",
    )
    raw = fig.to_dict()["data"][0]

    assert raw["x"] == [0.0, 2.0]
    assert raw["y"] == [1.0, 3.0]
    assert not isinstance(raw["x"], dict)
    assert fig.data[-1].x == (2.0,)
    assert fig.data[-1].y == (3.0,)


def test_eda_scatter_figure_reports_log_filtered_empty_data() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "dipper_score": [1.0, 3.0],
            "period_n_sources": [0, -2],
        }
    )

    counts = eda_plot_row_counts(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        log_x=True,
    )
    fig = eda_scatter_figure(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        log_x=True,
    )

    assert counts["plottable_rows"] == 0
    assert counts["dropped_nonpositive"] == 2
    assert len(fig.data) == 1
    assert fig.data[-1].name == "current"
    assert len(fig.data[-1].x) == 0
    assert "log-axis filtering" in fig.layout.annotations[0].text


def test_eda_publication_figure_uses_publication_style_and_highlight() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "dipper_score": [1.0, 3.0],
            "period_n_sources": [0, 2],
        }
    )

    fig = eda_publication_figure(
        frame,
        x_metric="period_n_sources",
        y_metric="dipper_score",
        selected_candidate_id="B",
    )

    assert fig.layout.paper_bgcolor == "white"
    assert fig.layout.plot_bgcolor == "white"
    assert PUBLICATION_PLOTLY_FONT.split(",")[0].strip() in fig.layout.font.family
    assert fig.layout.xaxis.title.text == "period_n_sources"
    assert fig.layout.yaxis.title.text == "dipper_score"
    assert fig.data[-1].name == "current"
    assert fig.data[-1].marker.line.color == "#111827"


def test_selected_row_style_targets_selected_candidate() -> None:
    style = selected_row_style(
        [{"candidate_id": "A"}, {"candidate_id": "B"}],
        "B",
        theme="white",
    )

    assert style[0]["if"] == {"filter_query": '{candidate_id} = "B"'}
    assert style[0]["fontWeight"] == "700"


def test_selected_candidate_row_style_does_not_require_table_rows() -> None:
    style = selected_candidate_row_style("B", theme="black")

    assert style[0]["if"] == {"filter_query": '{candidate_id} = "B"'}
    assert style[0]["fontWeight"] == "700"
