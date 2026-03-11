from __future__ import annotations

import argparse
from typing import Any
from urllib.parse import parse_qs
from uuid import uuid4

import dash
from dash import Input, Output, State, dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from malca.review.explore_data import (
    CandidateSourceData,
    _normalized_id,
    get_candidate_record,
    infer_plot_dir_for_record,
    infer_plot_dir_from_source,
    infer_source_kind,
    load_candidate_source,
    load_review_db,
    load_run_params,
    load_source_data,
)
from malca.review.interactive_plot import build_interactive_lightcurve_figure


def clickable_figure_html(
    fig: go.Figure,
    *,
    viewer_url: str,
    candidate_index: int = 0,
    window_name: str = "malca-mini-viewer",
    include_plotlyjs: str | bool = False,
) -> str:
    div_id = f"malca-click-{uuid4().hex}"
    html_str = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id=div_id)
    glue = "&" if "?" in viewer_url else "?"
    script = f"""
<script>
(function() {{
  var gd = document.getElementById('{div_id}');
  if (!gd || gd.__malcaClickBound) return;
  gd.__malcaClickBound = true;
  gd.on('plotly_click', function(evt) {{
    if (!evt || !evt.points || !evt.points.length) return;
    var point = evt.points[0];
    var custom = point.customdata;
    var candidate = Array.isArray(custom) ? custom[{candidate_index}] : custom;
    if (candidate === undefined || candidate === null || candidate === '') return;
    var url = '{viewer_url}{glue}candidate_id=' + encodeURIComponent(String(candidate));
    window.open(url, '{window_name}');
  }});
}})();
</script>
"""
    return html_str + script


def display_clickable_scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    viewer_url: str,
    color: str | None = None,
    symbol: str | None = None,
    query: str | None = None,
    hover_cols: list[str] | None = None,
    max_points: int | None = 4000,
    title: str | None = None,
):
    from IPython.display import HTML, display

    data = frame.query(query, engine="python").copy() if query else frame.copy()
    if x not in data.columns or y not in data.columns:
        missing = [col for col in (x, y) if col not in data.columns]
        raise KeyError(f"Missing columns: {missing}")
    data = data.loc[data[x].notna() & data[y].notna()].copy()
    if max_points is not None and len(data) > int(max_points):
        data = data.sample(int(max_points), random_state=42).copy()
    if "candidate_id" not in data.columns:
        raise KeyError("Dataframe must include candidate_id for click-through viewer links")
    if hover_cols is None:
        hover_cols = [col for col in ["candidate_id", "asas_sn_id", "gaia_id", "final_class"] if col in data.columns]

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color if color and color in data.columns else None,
        symbol=symbol if symbol and symbol in data.columns else None,
        hover_data=hover_cols,
        custom_data=["candidate_id"],
        opacity=0.7,
        title=title or f"{y} vs {x}",
    )
    fig.update_layout(template="plotly_white", height=520)
    fig.update_traces(marker={"size": 8})
    display(HTML(clickable_figure_html(fig, viewer_url=viewer_url, include_plotlyjs="cdn")))
    return fig


def _summary_items(record: dict[str, object]) -> list[tuple[str, str]]:
    fields = [
        ("Candidate", record.get("candidate_id")),
        ("ASAS-SN", record.get("asas_sn_id")),
        ("Gaia", record.get("gaia_id")),
        ("Final class", record.get("final_class")),
        ("Known?", record.get("vetting_likely_known")),
        ("Period (d)", record.get("period_consensus_days")),
        ("N sources", record.get("period_n_sources")),
        ("Dipper score", record.get("dipper_score")),
    ]
    items: list[tuple[str, str]] = []
    for label, value in fields:
        text = _normalized_id(value)
        if text:
            items.append((label, text))
    return items


def _render_summary(record: dict[str, object]) -> html.Div:
    items = _summary_items(record)
    if not items:
        return html.Div("No candidate summary available.", style={"color": "#c9d5df", "fontSize": "12px"})
    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{label}: ", style={"fontSize": "11px", "color": "#88a5bb"}),
                    html.Span(value, style={"fontSize": "11px", "color": "#f5f9fc"}),
                ],
                style={"padding": "2px 0", "display": "flex", "gap": "4px", "flexWrap": "wrap"},
            )
            for label, value in items
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(180px, 1fr))", "columnGap": "14px", "rowGap": "2px"},
    )


def _render_stats(stat_rows: list[tuple[str, str]]) -> html.Div:
    if not stat_rows:
        return html.Div("No precomputed light-curve stats available.", style={"color": "#c9d5df", "fontSize": "12px"})
    return html.Div(
        [
            html.Div(
                [
                    html.Span(key, style={"color": "#86a7bd", "fontSize": "10px", "textTransform": "uppercase"}),
                    html.Span(value, style={"color": "#eef3f7", "fontSize": "11px", "textAlign": "right", "wordBreak": "break-word"}),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "12px", "padding": "5px 0", "borderBottom": "1px solid rgba(60, 86, 104, 0.4)"},
            )
            for key, value in stat_rows
        ],
        style={"maxHeight": "360px", "overflowY": "auto", "padding": "4px 0"},
    )


def build_mini_viewer_app(source_data: CandidateSourceData, *, host: str, port: int) -> dash.Dash:
    app = dash.Dash(__name__, title="MALCA Mini Viewer")

    default_candidate = source_data.default_candidate_id

    app.layout = html.Div(
        [
            dcc.Location(id="viewer-url", refresh=False),
            html.Div(
                [
                    html.Div("MALCA Mini Viewer", style={"fontSize": "22px", "fontWeight": "600", "color": "#eef3f7"}),
                    html.Div(str(source_data.source_path), style={"fontSize": "12px", "color": "#8aa4b8"}),
                ],
                style={"paddingBottom": "12px"},
            ),
            html.Div(
                [
                    dcc.Input(id="candidate-input", value=default_candidate, debounce=True, placeholder="candidate_id / asas_sn_id / gaia_id", style={"width": "320px", "padding": "8px", "background": "#101923", "color": "#eef3f7", "border": "1px solid #32485c", "borderRadius": "6px"}),
                    dcc.Dropdown(id="camera-dropdown", multi=True, placeholder="All cameras", style={"minWidth": "280px", "color": "#111"}),
                    dcc.Checklist(
                        id="panel-options",
                        options=[
                            {"label": " Raw", "value": "raw"},
                            {"label": " Residuals", "value": "resid"},
                            {"label": " Phase", "value": "phase"},
                            {"label": " Baseline", "value": "baseline"},
                            {"label": " Events", "value": "events"},
                            {"label": " Diagnostics", "value": "diagnostics"},
                            {"label": " Filter bad cameras", "value": "filter_bad_cameras"},
                        ],
                        value=["raw", "resid", "phase", "baseline", "events", "filter_bad_cameras"],
                        inline=True,
                        style={"color": "#d8e3eb", "fontSize": "12px", "display": "flex", "flexWrap": "wrap", "gap": "10px"},
                    ),
                    dcc.Input(id="period-input", type="number", debounce=True, placeholder="Override period (days)", style={"width": "180px", "padding": "8px", "background": "#101923", "color": "#eef3f7", "border": "1px solid #32485c", "borderRadius": "6px"}),
                    dcc.RadioItems(
                        id="yaxis-mode",
                        options=[{"label": " mag", "value": "mag"}, {"label": " flux", "value": "flux"}],
                        value="mag",
                        inline=True,
                        style={"color": "#d8e3eb", "fontSize": "12px"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "center", "padding": "10px", "background": "#111a23", "border": "1px solid #223240", "borderRadius": "10px"},
            ),
            html.Div(id="viewer-status", style={"padding": "10px 2px", "color": "#c7d4df", "fontSize": "12px"}),
            html.Div(id="viewer-summary", style={"paddingBottom": "12px"}),
            dcc.Graph(id="lightcurve-graph", config={"displaylogo": False, "scrollZoom": True}, style={"height": "74vh", "border": "1px solid #223240", "borderRadius": "10px", "background": "#0c1218"}),
            html.Div(
                [
                    html.Div("Light-curve stats", style={"fontSize": "13px", "fontWeight": "600", "color": "#eef3f7", "paddingBottom": "6px"}),
                    html.Div(id="viewer-stats"),
                ],
                style={"marginTop": "12px", "padding": "10px 12px", "background": "#111a23", "border": "1px solid #223240", "borderRadius": "10px"},
            ),
        ],
        style={"background": "#0a1016", "minHeight": "100vh", "padding": "16px", "fontFamily": "Monaco, Courier New, monospace"},
    )

    @app.callback(Output("candidate-input", "value"), Input("viewer-url", "search"), State("candidate-input", "value"), prevent_initial_call=False)
    def sync_candidate_from_url(search: str | None, current_value: str | None):
        params = parse_qs((search or "").lstrip("?"))
        clicked = params.get("candidate_id", [None])[0]
        if clicked:
            return clicked
        return current_value or default_candidate

    @app.callback(
        Output("lightcurve-graph", "figure"),
        Output("camera-dropdown", "options"),
        Output("camera-dropdown", "value"),
        Output("viewer-status", "children"),
        Output("viewer-summary", "children"),
        Output("viewer-stats", "children"),
        Input("candidate-input", "value"),
        Input("camera-dropdown", "value"),
        Input("panel-options", "value"),
        Input("period-input", "value"),
        Input("yaxis-mode", "value"),
    )
    def render_candidate(candidate_value, selected_cameras, panel_options, period_value, yaxis_mode):
        record = get_candidate_record(source_data, candidate_value)
        if record is None:
            figure = go.Figure()
            figure.update_layout(template="plotly_dark", title=f"Candidate not found: {candidate_value}")
            return figure, [], [], f"Candidate not found: {candidate_value}", html.Div(), html.Div()

        plot_dir = infer_plot_dir_for_record(record, source_data.default_plot_dir)
        run_params = load_run_params(plot_dir)
        options = set(panel_options or [])
        override_period = None
        if period_value not in (None, ""):
            try:
                override_period = float(period_value)
            except Exception:
                override_period = None
        yaxis_literal = "flux" if str(yaxis_mode or "mag") == "flux" else "mag"

        native = build_interactive_lightcurve_figure(
            record,
            plot_dir=plot_dir,
            selected_cameras=selected_cameras or [],
            filter_bad_cameras="filter_bad_cameras" in options,
            show_baseline="baseline" in options,
            show_event_markers="events" in options,
            show_residuals="resid" in options,
            show_phase_fold="phase" in options,
            show_raw_mag="raw" in options,
            override_period=override_period,
            show_diagnostics="diagnostics" in options,
            confidence_colors=True,
            run_params=run_params,
            uirevision_key=f"mini:{_normalized_id(record.get('candidate_id'))}",
            theme="black",
            yaxis_mode=yaxis_literal,
        )

        status_bits = [native.get("status_message", "")]
        if plot_dir is not None:
            status_bits.append(f"Plot dir: {plot_dir}")
        warnings = native.get("warnings", []) or []
        if warnings:
            status_bits.append("Warnings: " + "; ".join(str(w) for w in warnings))
        status_text = " | ".join(bit for bit in status_bits if bit)
        return (
            native["figure"],
            native.get("camera_options", []),
            native.get("camera_values", []),
            status_text,
            _render_summary(record),
            _render_stats(native.get("stat_rows", [])),
        )

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight Dash viewer for MALCA light curves from a review DB or candidate table.")
    parser.add_argument("--source", required=True, help="Review DB, parquet, or CSV containing candidate rows")
    parser.add_argument("--source-kind", default=None, choices=["db", "parquet", "csv"], help="Optional explicit source kind")
    parser.add_argument("--plot-dir", default=None, help="Optional plot dir used to resolve bundle light curves")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8061, help="Dash port")
    parser.add_argument("--debug", action="store_true", help="Run Dash with debug enabled")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    source_data = load_source_data(args.source, source_kind=args.source_kind, plot_dir=args.plot_dir)
    app = build_mini_viewer_app(source_data, host=str(args.host), port=int(args.port))
    print(f"Starting MALCA Mini Viewer on http://{args.host}:{args.port}")
    print(f"  Source: {source_data.source_path}")
    if source_data.default_plot_dir is not None:
        print(f"  Plot dir: {source_data.default_plot_dir}")
    app.run(host=str(args.host), port=int(args.port), debug=bool(args.debug))


if __name__ == "__main__":
    main()
