"""Unified review light-curve plot specification assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    DEFAULT_OUTPUT_DIR,
    JD_OFFSET,
    REVIEW_RESIDUAL_FRACTION,
)
from malca.io.lightcurve_io import stable_camera_color
from malca.plotting.lightcurve_publication import BAND_COLORS
from malca.core.phase import BAND_LABELS, phase_fold_dataframe, phase_time_dataframe, resolve_phase_epoch, resolve_phase_period
from malca.review.interactive_plot import (
    DIP_EVENT_COLOR,
    JUMP_EVENT_COLOR,
    REQUIRED_COLUMNS,
    _baseline_config_from_run_params,
    _build_stat_rows,
    _build_title,
    _camera_labels,
    _compute_baseline_bands,
    _coerce_finite_float,
    _event_entries,
    _flux_err_from_mag_err,
    _load_cleaned_df,
    _mag_to_flux,
    _zero_centered_color_bounds,
    resolve_lightcurve_path,
)
from malca.review.coordinate_labels import publication_coordinate_headers
from malca.review.lightcurve_sources import (
    build_external_traces,
    coerce_external_source_values,
    discover_external_lcs,
    external_source_label,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS_ROOT = _REPO_ROOT / DEFAULT_OUTPUT_DIR / "results"

MARKER_MAP: dict[str, dict[str, str]] = {
    "circle": {"canonical": "circle", "mpl": "o"},
    "square": {"canonical": "square", "mpl": "s"},
    "diamond": {"canonical": "diamond", "mpl": "D"},
    "diamond-open": {"canonical": "diamond-open", "mpl": "D"},
    "triangle-up": {"canonical": "triangle-up", "mpl": "^"},
    "triangle-down": {"canonical": "triangle-down", "mpl": "v"},
    "triangle-down-open": {"canonical": "triangle-down-open", "mpl": "v"},
    "square-open": {"canonical": "square-open", "mpl": "s"},
    "star": {"canonical": "star", "mpl": "*"},
    "hexagon": {"canonical": "hexagon", "mpl": "h"},
    "x": {"canonical": "x", "mpl": "x"},
    "cross": {"canonical": "cross", "mpl": "+"},
}

_BAND_MARKERS = {0: "circle", 1: "square"}
NativeColorMode = Literal["camera", "band"]


def _native_trace_color(
    *,
    band: int,
    cam: str,
    band_labels: dict[int, str],
    color_mode: NativeColorMode,
) -> str:
    if color_mode == "band":
        return BAND_COLORS.get(band_labels[band], BAND_COLORS["unknown"])
    return stable_camera_color(cam)


def _native_trace_style(
    *,
    band: int,
    cam: str,
    band_labels: dict[int, str],
    color_mode: NativeColorMode,
    band_legend_shown: set[str],
) -> tuple[str, str, bool, str | None]:
    """Return color, label, showlegend, and legendgroup for a native ASAS-SN trace."""
    band_label = band_labels[band]
    trace_name = f"{cam} ({band_label})"
    if color_mode == "band":
        color = BAND_COLORS.get(band_label, BAND_COLORS["unknown"])
        showlegend = band_label not in band_legend_shown
        if showlegend:
            band_legend_shown.add(band_label)
        return color, (band_label if showlegend else trace_name), showlegend, band_label
    return stable_camera_color(cam), trace_name, True, None


@dataclass(frozen=True)
class PlotTrace:
    panel_id: str
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None = None
    color: str | None = None
    marker: str | None = None
    label: str | None = None
    alpha: float = 1.0
    marker_size: float = 6.0
    kind: str = "scatter"
    cmap_values: np.ndarray | None = None
    cmap_vmin: float | None = None
    cmap_vmax: float | None = None
    showlegend: bool = True
    legendgroup: str | None = None
    line_width: float = 1.6
    customdata: np.ndarray | None = None
    hovertemplate: str | None = None


@dataclass(frozen=True)
class PlotPanel:
    panel_id: str
    kind: str
    y_label: str = ""
    x_label: str = ""
    invert_y: bool = False
    height_ratio: float = 1.0
    external_source: str | None = None
    y_autorange: bool | str = True
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class PlotEventOverlay:
    panel_id: str
    x0: float
    half_width: float
    kind: str
    confidence: float = 1.0
    show_span: bool = False
    color: str = ""


@dataclass(frozen=True)
class PlotAnnotation:
    panel_id: str | None
    text: str
    x: float
    y: float
    xref: str = "paper"
    yref: str = "paper"
    font_size: float | None = None
    color: str | None = None
    bgcolor: str | None = None
    bordercolor: str | None = None
    borderwidth: float | None = None
    opacity: float | None = None
    xanchor: str | None = None
    yanchor: str | None = None
    showarrow: bool = False
    hovertext: str | None = None


@dataclass(frozen=True)
class PlotHLine:
    panel_id: str
    y: float
    color: str = ""
    dash: str = "dot"
    line_width: float = 1.0


@dataclass(frozen=True)
class PlotVLine:
    panel_id: str
    x: float
    color: str = ""
    dash: str = "dot"
    line_width: float = 1.0


@dataclass(frozen=True)
class ReviewLightCurvePlotSpec:
    title: str
    jd_offset: float
    panels: tuple[PlotPanel, ...]
    traces: tuple[PlotTrace, ...]
    baselines: tuple[PlotTrace, ...]
    events: tuple[PlotEventOverlay, ...]
    hlines: tuple[PlotHLine, ...]
    vlines: tuple[PlotVLine, ...]
    annotations: tuple[PlotAnnotation, ...]
    legend_panel_id: str | None
    warnings: tuple[str, ...]
    status: str
    status_message: str
    stat_rows: tuple[tuple[str, str], ...]
    camera_diagnostics: dict[str, Any]
    camera_options: tuple[dict[str, str], ...]
    camera_values: tuple[str, ...]
    phase_diagnostics: dict[str, Any] = field(default_factory=dict)
    phase_period: float | None = None
    phase_panel_mode: Literal["fold", "time"] = "fold"
    is_flux: bool = False
    show_native_raw: bool = True
    yaxis_mode: Literal["mag", "flux"] = "mag"
    baseline_opacity: float = 0.5
    phase_requested: bool = False
    phase_enabled: bool = False
    phase_period_pending: bool = False
    suppress_catalog_phase_period: bool = False
    phase_source: str = ""
    show_diagnostics: bool = False
    confidence_colors: bool = False
    jd_match_panel_id: str | None = None
    header_left: str | None = None
    header_right: str | None = None


@dataclass(frozen=True)
class ReviewPlotRequest:
    payload: dict
    plot_dir: Path | None
    selected_cameras: list[str] | None
    filter_bad_cameras: bool
    show_baseline: bool
    show_event_markers: bool
    show_residuals: bool
    show_phase_fold: bool
    phase_panel_mode: Literal["fold", "time"]
    show_raw_mag: bool
    override_period: float | None
    override_period_source: str
    phase_period_pending: bool
    phase_period_pending_source: str
    suppress_catalog_phase_period: bool
    show_diagnostics: bool
    confidence_colors: bool
    run_params: dict | None
    residual_fraction: float
    baseline_opacity: float
    yaxis_mode: Literal["mag", "flux"]
    external_lcs: dict[str, Path] | None
    external_source_view: str | list[str]
    external_panel_mode: Literal["overlay", "split"]
    selected_bands: list[str] | None
    native_color_mode: NativeColorMode = "camera"
    candidate_id: str | None = None
    discover_external: bool = True

    @classmethod
    def from_kwargs(
        cls,
        payload: dict,
        *,
        plot_dir: Path | None,
        selected_cameras: list[str] | None,
        filter_bad_cameras: bool,
        show_baseline: bool,
        show_event_markers: bool,
        show_residuals: bool,
        show_phase_fold: bool = False,
        phase_panel_mode: Literal["fold", "time"] = "fold",
        show_raw_mag: bool = True,
        override_period: float | None = None,
        override_period_source: str = "manual/search",
        phase_period_pending: bool = False,
        phase_period_pending_source: str = "",
        suppress_catalog_phase_period: bool = False,
        show_diagnostics: bool,
        confidence_colors: bool,
        run_params: dict | None,
        residual_fraction: float = REVIEW_RESIDUAL_FRACTION,
        baseline_opacity: float = 0.5,
        yaxis_mode: Literal["mag", "flux"] = "mag",
        external_lcs: dict[str, Path] | None = None,
        external_source_view: str | list[str] = "asassn",
        external_panel_mode: Literal["overlay", "split"] = "overlay",
        selected_bands: list[str] | None = None,
        native_color_mode: NativeColorMode = "camera",
        candidate_id: str | None = None,
        discover_external: bool = True,
    ) -> ReviewPlotRequest:
        return cls(
            payload=payload,
            plot_dir=Path(plot_dir) if plot_dir else None,
            selected_cameras=selected_cameras,
            filter_bad_cameras=filter_bad_cameras,
            show_baseline=show_baseline,
            show_event_markers=show_event_markers,
            show_residuals=show_residuals,
            show_phase_fold=show_phase_fold,
            phase_panel_mode=phase_panel_mode,
            show_raw_mag=show_raw_mag,
            override_period=override_period,
            override_period_source=override_period_source,
            phase_period_pending=phase_period_pending,
            phase_period_pending_source=str(phase_period_pending_source or ""),
            suppress_catalog_phase_period=suppress_catalog_phase_period,
            show_diagnostics=show_diagnostics,
            confidence_colors=confidence_colors,
            run_params=run_params,
            residual_fraction=residual_fraction,
            baseline_opacity=baseline_opacity,
            yaxis_mode=yaxis_mode,
            external_lcs=external_lcs,
            external_source_view=external_source_view,
            external_panel_mode=external_panel_mode,
            selected_bands=selected_bands,
            native_color_mode=native_color_mode,
            candidate_id=candidate_id,
            discover_external=discover_external,
        )


def trace_from_dict(d: dict) -> PlotTrace:
    """Convert an external-trace dict from build_external_traces into a PlotTrace."""
    return PlotTrace(
        panel_id=str(d["panel_id"]),
        x=np.asarray(d["x"], dtype=float),
        y=np.asarray(d["y"], dtype=float),
        yerr=None if d.get("yerr") is None else np.asarray(d["yerr"], dtype=float),
        color=d.get("color"),
        marker=d.get("marker"),
        label=d.get("label"),
        alpha=float(d.get("alpha", 0.8)),
        marker_size=float(d.get("marker_size", 6)),
        kind=str(d.get("kind", "scatter")),
        showlegend=bool(d.get("showlegend", True)),
        legendgroup=d.get("legendgroup"),
        customdata=d.get("customdata"),
        hovertemplate=d.get("hovertemplate"),
    )


def _empty_spec(
    *,
    status: str,
    status_message: str,
    warnings: list[str] | None = None,
    camera_diagnostics: dict | None = None,
    camera_options: list[dict[str, str]] | None = None,
    camera_values: list[str] | None = None,
) -> ReviewLightCurvePlotSpec:
    return ReviewLightCurvePlotSpec(
        title="",
        jd_offset=float(JD_OFFSET),
        panels=(),
        traces=(),
        baselines=(),
        events=(),
        hlines=(),
        vlines=(),
        annotations=(),
        legend_panel_id=None,
        warnings=tuple(warnings or []),
        status=status,
        status_message=status_message,
        stat_rows=(),
        camera_diagnostics=dict(camera_diagnostics or {}),
        camera_options=tuple(camera_options or ()),
        camera_values=tuple(camera_values or ()),
    )


def assemble_review_lightcurve_plot(request: ReviewPlotRequest) -> ReviewLightCurvePlotSpec:
    """Build a backend-neutral review light-curve plot specification."""
    phase_panel_mode: Literal["fold", "time"] = (
        "time" if str(request.phase_panel_mode or "fold").strip().lower() == "time" else "fold"
    )
    source_values = coerce_external_source_values(request.external_source_view)
    external_panel_mode = (
        "split" if str(request.external_panel_mode or "").strip().lower() == "split" else "overlay"
    )
    native_source_enabled = "asassn" in source_values
    requested_external_sources = [src for src in source_values if src != "asassn"]

    external_lcs_by_source = {
        str(k).strip().lower(): Path(v) for k, v in dict(request.external_lcs or {}).items()
    }
    if request.candidate_id and not external_lcs_by_source and request.discover_external:
        discovered = discover_external_lcs(
            str(request.candidate_id),
            request.payload,
            request.plot_dir,
            requested_external_sources,
            default_results_root=_DEFAULT_RESULTS_ROOT,
        )
        external_lcs_by_source = {k: Path(v) for k, v in discovered.items()}

    active_external_lcs = {
        src: external_lcs_by_source[src]
        for src in requested_external_sources
        if src in external_lcs_by_source
    }
    missing_external_sources = [src for src in requested_external_sources if src not in external_lcs_by_source]
    split_external_lcs = active_external_lcs if external_panel_mode == "split" else {}
    overlay_external_lcs = active_external_lcs if external_panel_mode != "split" else {}
    show_native_raw = bool(request.show_raw_mag and native_source_enabled)
    show_raw_panel = bool(show_native_raw or (request.show_raw_mag and overlay_external_lcs))

    plot_dir = request.plot_dir
    lc_path = resolve_lightcurve_path(request.payload, plot_dir)
    if lc_path is None:
        return _empty_spec(
            status="missing-file",
            status_message="Missing light-curve file. Check candidate path or imported bundle assets.",
            warnings=["Missing LC file"],
        )

    scatter_ratio = (
        float(request.run_params.get("bad_camera_scatter_ratio", BAD_CAMERA_SCATTER_RATIO_THRESHOLD))
        if request.run_params
        else BAD_CAMERA_SCATTER_RATIO_THRESHOLD
    )
    clean_abs = (
        float(request.run_params.get("clean_max_error_absolute", CLEAN_LC_MAX_ERROR_ABSOLUTE))
        if request.run_params
        else CLEAN_LC_MAX_ERROR_ABSOLUTE
    )
    clean_sig = (
        float(request.run_params.get("clean_max_error_sigma", CLEAN_LC_MAX_ERROR_SIGMA))
        if request.run_params
        else CLEAN_LC_MAX_ERROR_SIGMA
    )

    df, filtered_cameras, camera_diagnostics = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=request.filter_bad_cameras,
        scatter_ratio=scatter_ratio,
        clean_max_error_absolute=clean_abs,
        clean_max_error_sigma=clean_sig,
    )

    missing_cols = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
    if missing_cols:
        return _empty_spec(
            status="missing-columns",
            status_message=f"Missing columns: {', '.join(missing_cols)}",
            warnings=["Missing required columns"],
            camera_diagnostics=camera_diagnostics,
        )

    if df.empty:
        return _empty_spec(
            status="empty-after-filter",
            status_message="No points remain after filtering.",
            warnings=["No data after filtering"],
            camera_diagnostics=camera_diagnostics,
        )

    median_jd = float(df["JD"].median())
    jd_offset = JD_OFFSET if median_jd > 2000000 else 8000.0
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    df["JD_plot"] = df["JD"] - jd_offset
    df["camera_label"] = _camera_labels(df, request.payload)

    camera_ids = sorted(df["camera_label"].dropna().unique().tolist())
    selected = [str(c) for c in (request.selected_cameras or []) if str(c) in camera_ids]
    if not selected:
        selected = list(camera_ids)
    df = df[df["camera_label"].isin(selected)].copy()

    camera_options = tuple({"label": f"{cam}", "value": str(cam)} for cam in camera_ids)
    if df.empty:
        return _empty_spec(
            status="empty-camera-selection",
            status_message="Current camera selection has no points.",
            warnings=["No points for selected cameras"],
            camera_diagnostics=camera_diagnostics,
            camera_options=list(camera_options),
            camera_values=selected,
        )

    band_labels = BAND_LABELS
    available_band_labels = [
        label for band, label in band_labels.items() if int((df["v_g_band"] == band).sum()) > 0
    ]
    selected_band_lookup = {
        str(value).strip().lower()
        for value in (request.selected_bands if request.selected_bands is not None else available_band_labels)
        if str(value).strip()
    }
    active_bands = [
        band
        for band, label in band_labels.items()
        if label.lower() in selected_band_lookup and label in available_band_labels
    ]
    if not active_bands:
        return _empty_spec(
            status="empty-band-selection",
            status_message="Current band selection has no visible points.",
            warnings=["No points for selected bands"],
            camera_diagnostics=camera_diagnostics,
            camera_options=list(camera_options),
            camera_values=selected,
        )

    baseline_name, baseline_kwargs, baseline_warnings = _baseline_config_from_run_params(request.run_params)
    baseline_cache_key = (
        str(lc_path.resolve()),
        tuple(sorted(str(c) for c in selected)),
        bool(request.filter_bad_cameras),
        float(scatter_ratio),
        float(clean_abs),
        float(clean_sig),
    )
    band_dfs = _compute_baseline_bands(
        df,
        baseline_name,
        baseline_cache_key,
        baseline_kwargs=baseline_kwargs,
    )

    warnings: list[str] = list(baseline_warnings)
    for source_name in missing_external_sources:
        warnings.append(f"{external_source_label(source_name)} light curve is not available for this candidate.")

    phase_requested = bool(request.show_phase_fold)
    if phase_requested:
        period_payload = {} if request.suppress_catalog_phase_period else request.payload
        phase_period, phase_source = resolve_phase_period(
            period_payload,
            override_period=request.override_period,
            override_source=request.override_period_source or "manual/search",
        )
    else:
        phase_period = None
        phase_source = ""
    phase_enabled = bool(phase_requested and phase_period is not None)
    pending_period_source = str(request.phase_period_pending_source or "Auto period search")
    if phase_requested and not phase_enabled:
        if request.phase_period_pending:
            warnings.append(f"{pending_period_source} is running; the phase panel will update when it finishes.")
        elif request.suppress_catalog_phase_period:
            warnings.append("Auto harmonic check did not return a valid period. Run Find Period or enter Manual P.")
        else:
            warnings.append("Phase panel requested, but no valid period was found. Use Find Period to search manually.")

    try:
        residual_fraction = float(request.residual_fraction)
    except Exception:
        residual_fraction = REVIEW_RESIDUAL_FRACTION
    if not np.isfinite(residual_fraction):
        residual_fraction = REVIEW_RESIDUAL_FRACTION
    residual_fraction = float(np.clip(residual_fraction, 0.15, 0.85))

    panels: list[PlotPanel] = []
    if show_raw_panel:
        panels.append(PlotPanel(panel_id="raw", kind="raw", height_ratio=1.0))
    for source_name in split_external_lcs:
        panels.append(
            PlotPanel(
                panel_id=f"external:{source_name}",
                kind="external",
                height_ratio=0.55,
                external_source=source_name,
                y_autorange="reversed",
            )
        )
    if request.show_residuals:
        panels.append(
            PlotPanel(
                panel_id="resid",
                kind="resid",
                height_ratio=float(np.clip(residual_fraction, 0.15, 0.45)),
                y_autorange="reversed",
            )
        )
    if phase_requested:
        panels.append(PlotPanel(panel_id="phase", kind="phase", height_ratio=0.45))

    if not panels:
        return ReviewLightCurvePlotSpec(
            title=_build_title(request.payload, df),
            jd_offset=jd_offset,
            panels=(),
            traces=(),
            baselines=(),
            events=(),
            hlines=(),
            vlines=(),
            annotations=(),
            legend_panel_id=None,
            warnings=tuple([*warnings, "No panels selected"]),
            status="ok",
            status_message="No panels selected.",
            stat_rows=tuple(_build_stat_rows(request.payload, df, filtered_cameras)),
            camera_diagnostics=camera_diagnostics,
            camera_options=camera_options,
            camera_values=tuple(selected),
        )

    is_flux = request.yaxis_mode == "flux"
    traces: list[PlotTrace] = []
    baselines: list[PlotTrace] = []
    events: list[PlotEventOverlay] = []
    hlines: list[PlotHLine] = []
    vlines: list[PlotVLine] = []
    annotations: list[PlotAnnotation] = []
    color_mode: NativeColorMode = (
        "band" if request.native_color_mode == "band" else "camera"
    )
    band_legend_shown: set[str] = set()

    for band in active_bands:
        bdf = band_dfs.get(band)
        if bdf is None or bdf.empty:
            continue
        for cam in selected:
            cdf = bdf[bdf["camera_label"] == cam]
            if cdf.empty:
                continue
            color, trace_name, show_legend, legendgroup = _native_trace_style(
                band=band,
                cam=cam,
                band_labels=band_labels,
                color_mode=color_mode,
                band_legend_shown=band_legend_shown,
            )
            err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
            resid = cdf["resid"].to_numpy() if "resid" in cdf.columns else np.full(len(cdf), np.nan)
            baseline = cdf["baseline"].to_numpy() if "baseline" in cdf.columns else np.full(len(cdf), np.nan)
            jd_full = cdf["JD_plot"].to_numpy() + JD_OFFSET
            mag_raw = cdf["mag"].to_numpy()

            if show_native_raw:
                y_raw = _mag_to_flux(mag_raw) if is_flux else mag_raw
                err_raw = _flux_err_from_mag_err(y_raw, err) if is_flux else err
                hover_raw = np.column_stack([jd_full, err, resid, baseline, err_raw, mag_raw])
                raw_hovertemplate = (
                    "<b>%{fullData.name}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>"
                )
                if is_flux:
                    raw_hovertemplate += (
                        "F: %{y:.4e}<br>"
                        "σ<sub>F</sub>: %{customdata[4]:.3e}<br>"
                        "m: %{customdata[5]:.4f}<br>"
                        "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                        "Δm: %{customdata[2]:.4f}<br>"
                        "m<sub>base</sub>: %{customdata[3]:.4f}<extra></extra>"
                    )
                else:
                    raw_hovertemplate += (
                        "m: %{y:.4f}<br>"
                        "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                        "Δm: %{customdata[2]:.4f}<br>"
                        "m<sub>base</sub>: %{customdata[3]:.4f}<extra></extra>"
                    )
                traces.append(
                    PlotTrace(
                        panel_id="raw",
                        x=cdf["JD_plot"].to_numpy(dtype=float),
                        y=y_raw,
                        yerr=err_raw,
                        color=color,
                        marker=_BAND_MARKERS[band],
                        label=trace_name,
                        alpha=1.0,
                        marker_size=7.0,
                        kind="scatter",
                        showlegend=show_legend,
                        legendgroup=legendgroup,
                        customdata=hover_raw,
                        hovertemplate=raw_hovertemplate,
                    )
                )

            if request.show_residuals:
                y_resid = (_mag_to_flux(resid) - 1.0) if is_flux else resid
                err_resid = _flux_err_from_mag_err(_mag_to_flux(resid), err) if is_flux else err
                hover_resid = np.column_stack([jd_full, err, resid, baseline, err_resid, mag_raw])
                resid_hovertemplate = (
                    "<b>%{fullData.name}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>"
                )
                if is_flux:
                    resid_hovertemplate += (
                        "ΔF/F: %{y:.4f}<br>"
                        "σ<sub>F</sub>: %{customdata[4]:.3e}<br>"
                        "Δm: %{customdata[2]:.4f}<extra></extra>"
                    )
                else:
                    resid_hovertemplate += (
                        "Δm: %{y:.4f}<br>"
                        "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>"
                    )
                traces.append(
                    PlotTrace(
                        panel_id="resid",
                        x=cdf["JD_plot"].to_numpy(dtype=float),
                        y=y_resid,
                        color=color,
                        marker=_BAND_MARKERS[band],
                        label=trace_name,
                        alpha=1.0,
                        marker_size=6.0,
                        kind="scatter",
                        showlegend=False,
                        customdata=hover_resid,
                        hovertemplate=resid_hovertemplate,
                    )
                )

        if show_native_raw and request.show_baseline and "baseline" in bdf.columns:
            for cam in selected:
                cbase = bdf[(bdf["camera_label"] == cam) & np.isfinite(bdf["baseline"])].sort_values("JD_plot")
                if cbase.empty:
                    continue
                base_color = _native_trace_color(
                    band=band,
                    cam=cam,
                    band_labels=band_labels,
                    color_mode=color_mode,
                )
                y_base = _mag_to_flux(cbase["baseline"].to_numpy()) if is_flux else cbase["baseline"].to_numpy()
                baselines.append(
                    PlotTrace(
                        panel_id="raw",
                        x=cbase["JD_plot"].to_numpy(dtype=float),
                        y=y_base,
                        color=base_color,
                        kind="line",
                        showlegend=False,
                        line_width=1.6,
                        alpha=float(request.baseline_opacity),
                        customdata=cbase["JD_plot"].to_numpy() + JD_OFFSET,
                        hovertemplate=(
                            f"JD: %{{customdata:.5f}}<br>JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>"
                            f"F<sub>base</sub>: %{{y:.4e}}<extra></extra>"
                            if is_flux
                            else f"JD: %{{customdata:.5f}}<br>JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>"
                            f"m<sub>base</sub>: %{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

    event_entries = _event_entries(request.payload, jd_offset, request.run_params, lc_median=median_jd)
    if request.show_event_markers and show_native_raw:
        visible_raw_df = df[df["v_g_band"].isin(active_bands)].copy() if "v_g_band" in df.columns else df.copy()
        if visible_raw_df.empty:
            visible_raw_df = df.copy()
        raw_event_y_values = (
            _mag_to_flux(visible_raw_df["mag"].to_numpy(dtype=float))
            if is_flux
            else visible_raw_df["mag"].to_numpy(dtype=float)
        )
        finite_raw_event_y = raw_event_y_values[np.isfinite(raw_event_y_values)]
        if finite_raw_event_y.size:
            raw_event_span = float(np.nanmax(finite_raw_event_y) - np.nanmin(finite_raw_event_y))
        else:
            raw_event_span = 1.0
        raw_event_pad = raw_event_span * 0.10
        if not np.isfinite(raw_event_pad) or raw_event_pad <= 0:
            raw_event_pad = 0.05 * max(1.0, abs(float(finite_raw_event_y[0])) if finite_raw_event_y.size else 1.0)

        for entry in event_entries:
            color = str(entry["base_color"])
            conf = float(entry["confidence"])
            if request.confidence_colors:
                alpha = 0.35 + 0.55 * conf
                if entry["kind"] == "dip":
                    color = f"rgba(255,107,107,{alpha:.3f})"
                else:
                    color = f"rgba(0,150,255,{alpha:.3f})"
            events.append(
                PlotEventOverlay(
                    panel_id="raw",
                    x0=float(entry["x0"]),
                    half_width=float(entry["half_width"]),
                    kind=str(entry["kind"]),
                    confidence=conf,
                    show_span=bool(request.show_diagnostics and float(entry["half_width"]) > 0),
                    color=color,
                )
            )
            bf_text = "n/a" if entry["bf"] is None else f"{float(entry['bf']):.2f}"
            logbf_thr = entry.get("logbf_threshold")
            sig_thr = entry.get("sig_threshold")
            logbf_thr_text = "n/a" if logbf_thr is None else f"{float(logbf_thr):.2f}"
            sig_thr_text = "n/a" if sig_thr is None else f"{float(sig_thr):.2f}"
            hover_text = (
                f"{str(entry['kind']).title()} event<br>"
                f"t0 [JD]: {float(entry['t0']):.5f}<br>"
                f"w: {float(entry['half_width']) / 2.0:.3f}<br>"
                f"log BF: {bf_text}<br>"
                f"log BF thr: {logbf_thr_text}<br>"
                f"sig thr: {sig_thr_text}<br>"
                f"morph: {entry['morph'] or 'n/a'}<br>"
                f"c: {float(entry['confidence']):.2f}"
            )
            annotations.append(
                PlotAnnotation(
                    panel_id="raw",
                    text="◆",
                    x=float(entry["x0"]),
                    y=0.0,
                    xref="axis",
                    yref="axis",
                    font_size=18.0,
                    color=color,
                    hovertext=hover_text,
                )
            )
            if request.show_diagnostics and float(entry["half_width"]) > 0:
                annotations.append(
                    PlotAnnotation(
                        panel_id="raw",
                        text=(
                            f"{str(entry['kind']).title()} thr logBF={logbf_thr_text}, sig={sig_thr_text}"
                        ),
                        x=float(entry["x0"]),
                        y=0.0,
                        xref="axis",
                        yref="axis",
                        font_size=9.0,
                        opacity=0.92,
                    )
                )

    phase_diag: dict[str, object] = {}
    if phase_enabled and phase_period is not None:
        phase_zero_jd = resolve_phase_epoch(df)
        phase_inputs = [
            band_dfs[band]
            for band in active_bands
            if band in band_dfs and band_dfs[band] is not None and not band_dfs[band].empty
        ]
        phase_source_df = pd.concat(phase_inputs, ignore_index=True) if phase_inputs else pd.DataFrame()

        if phase_panel_mode == "time":
            phase_time_df = pd.DataFrame()
            if not phase_source_df.empty:
                phase_time_df, phase_diag = phase_time_dataframe(
                    phase_source_df,
                    float(phase_period),
                    epoch_jd=phase_zero_jd,
                    value_mode="resid",
                    duplicate_cycles=True,
                )
            color_values = (
                pd.to_numeric(phase_time_df.get("phase_value"), errors="coerce").to_numpy(dtype=float)
                if not phase_time_df.empty and "phase_value" in phase_time_df.columns
                else np.array([], dtype=float)
            )
            cmin, cmax = _zero_centered_color_bounds(color_values)
            for band in active_bands:
                if phase_time_df.empty or "v_g_band" not in phase_time_df.columns:
                    continue
                band_phase_df = phase_time_df[pd.to_numeric(phase_time_df["v_g_band"], errors="coerce") == band]
                if band_phase_df.empty:
                    continue
                for cam in selected:
                    cdf = band_phase_df[band_phase_df["camera_label"] == cam]
                    if cdf.empty:
                        continue
                    err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
                    resid = pd.to_numeric(cdf["phase_value"], errors="coerce").to_numpy(dtype=float)
                    jd_full_phase = cdf["JD_plot"].to_numpy() + JD_OFFSET
                    hover_phase = np.column_stack([
                        jd_full_phase,
                        err,
                        pd.to_numeric(cdf["mag"], errors="coerce").to_numpy(dtype=float),
                        resid,
                        pd.to_numeric(cdf["cycle"], errors="coerce").to_numpy(dtype=float),
                        pd.to_numeric(cdf["v_g_band"], errors="coerce").to_numpy(dtype=float),
                    ])
                    traces.append(
                        PlotTrace(
                            panel_id="phase",
                            x=cdf["phase"].to_numpy(dtype=float),
                            y=cdf["cycle"].to_numpy(dtype=float),
                            cmap_values=resid,
                            cmap_vmin=cmin,
                            cmap_vmax=cmax,
                            marker=_BAND_MARKERS[band],
                            label=f"{cam} ({band_labels[band]})",
                            kind="phase_scatter_cmap",
                            showlegend=False,
                            marker_size=6.0,
                            customdata=hover_phase,
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "φ: %{x:.4f}<br>"
                                "cycle: %{customdata[4]:.0f}<br>"
                                "JD: %{customdata[0]:.5f}<br>"
                                "Δm: %{customdata[3]:.4f}<br>"
                                "m: %{customdata[2]:.4f}<br>"
                                "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                                "band: %{customdata[5]:.0f}<extra></extra>"
                            ),
                        )
                    )
        else:
            phase_bdf = pd.DataFrame()
            if not phase_source_df.empty:
                phase_bdf, phase_diag = phase_fold_dataframe(
                    phase_source_df,
                    float(phase_period),
                    epoch_jd=phase_zero_jd,
                    value_mode="resid",
                    duplicate_cycles=True,
                )
            for band in active_bands:
                if phase_bdf.empty or "v_g_band" not in phase_bdf.columns:
                    continue
                band_phase_df = phase_bdf[pd.to_numeric(phase_bdf["v_g_band"], errors="coerce") == band]
                if band_phase_df.empty:
                    continue
                for cam in selected:
                    cdf = band_phase_df[band_phase_df["camera_label"] == cam]
                    if cdf.empty:
                        continue
                    color = _native_trace_color(
                        band=band,
                        cam=cam,
                        band_labels=band_labels,
                        color_mode=color_mode,
                    )
                    err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
                    resid = cdf["phase_value"].to_numpy()
                    y_phase = (_mag_to_flux(resid) - 1.0) if is_flux else resid
                    err_phase = _flux_err_from_mag_err(_mag_to_flux(resid), err) if is_flux else err
                    jd_full_phase = cdf["JD_plot"].to_numpy() + JD_OFFSET
                    hover_phase = np.column_stack([jd_full_phase, err, cdf["mag"].to_numpy(), resid, err_phase])
                    phase_hovertemplate = (
                        "<b>%{fullData.name}</b><br>"
                        "φ: %{x:.4f}<br>"
                        "JD: %{customdata[0]:.5f}<br>"
                    )
                    if is_flux:
                        phase_hovertemplate += (
                            "ΔF/F: %{y:.4f}<br>"
                            "σ<sub>F</sub>: %{customdata[4]:.3e}<br>"
                            "Δm: %{customdata[3]:.4f}<br>"
                            "m: %{customdata[2]:.4f}<br>"
                            "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>"
                        )
                    else:
                        phase_hovertemplate += (
                            "Δm: %{y:.4f}<br>"
                            "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                            "m: %{customdata[2]:.4f}<extra></extra>"
                        )
                    traces.append(
                        PlotTrace(
                            panel_id="phase",
                            x=cdf["phase"].to_numpy(dtype=float),
                            y=y_phase,
                            yerr=err_phase,
                            color=color,
                            marker=_BAND_MARKERS[band],
                            label=f"{cam} ({band_labels[band]})",
                            kind="scatter",
                            showlegend=False,
                            marker_size=6.0,
                            customdata=hover_phase,
                            hovertemplate=phase_hovertemplate,
                        )
                    )

            phase_lag = float(phase_diag.get("phase_lag_g_v_cycles", np.nan))
            phase_lag_abs = float(phase_diag.get("phase_lag_g_v_abs_cycles", np.nan))
            if np.isfinite(phase_lag):
                lag_text = f"g-V lag {phase_lag:+.3f} cyc"
                if np.isfinite(phase_lag_abs):
                    lag_text += f" (|lag| {phase_lag_abs:.3f})"
                annotations.append(
                    PlotAnnotation(
                        panel_id="phase",
                        text=lag_text,
                        x=0.99,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        xanchor="right",
                        yanchor="top",
                        font_size=11.0,
                        borderwidth=1.0,
                        opacity=0.9,
                    )
                )

        for x in (0.0, 1.0, 2.0):
            vlines.append(PlotVLine(panel_id="phase", x=x))
        if phase_panel_mode == "fold":
            hlines.append(PlotHLine(panel_id="phase", y=0.0))

    raw_y_range: tuple[float, float] | None = None
    if show_raw_panel:
        raw_panel = next(p for p in panels if p.panel_id == "raw")
        y_label = r"$F$ [arb]" if is_flux else r"$m$ [mag]"
        if show_native_raw:
            if is_flux:
                y_vals = _mag_to_flux(df["mag"].to_numpy())
                if request.show_baseline and "baseline" in df.columns:
                    b_vals = df["baseline"].dropna().to_numpy()
                    if b_vals.size > 0:
                        y_vals = np.concatenate([y_vals, _mag_to_flux(b_vals)])
            else:
                y_vals = df["mag"].to_numpy()
                if request.show_baseline and "baseline" in df.columns:
                    b_vals = df["baseline"].dropna().to_numpy()
                    if b_vals.size > 0:
                        y_vals = np.concatenate([y_vals, b_vals])
            if y_vals.size > 0:
                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                y_pad_fraction = 0.10 if request.show_event_markers and event_entries else 0.05
                y_pad = (y_max - y_min) * y_pad_fraction
                if y_pad == 0:
                    y_pad = 0.5 if not is_flux else y_max * 0.05
                if is_flux:
                    raw_y_range = (max(0, y_min - y_pad), y_max + y_pad)
                else:
                    raw_y_range = (y_max + y_pad, y_min - y_pad)
                raw_panel = PlotPanel(
                    panel_id="raw",
                    kind="raw",
                    height_ratio=raw_panel.height_ratio,
                    y_label=y_label,
                    invert_y=not is_flux,
                    y_autorange=False,
                    y_range=raw_y_range,
                )
            else:
                raw_panel = PlotPanel(
                    panel_id="raw",
                    kind="raw",
                    height_ratio=raw_panel.height_ratio,
                    y_label=y_label,
                    invert_y=not is_flux,
                    y_autorange="reversed" if not is_flux else True,
                )
        else:
            raw_panel = PlotPanel(
                panel_id="raw",
                kind="raw",
                height_ratio=raw_panel.height_ratio,
                y_label=y_label,
                invert_y=not is_flux,
                y_autorange="reversed" if not is_flux else True,
            )
        panels = [raw_panel if p.panel_id == "raw" else p for p in panels]

    if request.show_residuals:
        panels = [
            PlotPanel(
                panel_id=p.panel_id,
                kind=p.kind,
                y_label=r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]",
                x_label=p.x_label,
                invert_y=not is_flux,
                height_ratio=p.height_ratio,
                external_source=p.external_source,
                y_autorange="reversed" if not is_flux else True,
                x_range=p.x_range,
                y_range=p.y_range,
            )
            if p.panel_id == "resid"
            else p
            for p in panels
        ]
        hlines.append(PlotHLine(panel_id="resid", y=0.0))

    if phase_enabled and phase_period is not None:
        phase_axis_title = (
            rf"$\phi\ \mathrm{{vs.}}\ E\,(P={phase_period:.5f}\,\mathrm{{d}})$"
            if phase_panel_mode == "time"
            else rf"$\phi\,(P={phase_period:.5f}\,\mathrm{{d}})$"
        )
        panels = [
            PlotPanel(
                panel_id=p.panel_id,
                kind=p.kind,
                y_label="Cycle E" if phase_panel_mode == "time" else (r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]"),
                x_label=phase_axis_title,
                invert_y=False if phase_panel_mode == "time" else (not is_flux),
                height_ratio=p.height_ratio,
                external_source=p.external_source,
                y_autorange=True if phase_panel_mode == "time" else ("reversed" if not is_flux else True),
                x_range=(-0.02, 2.02),
                y_range=p.y_range,
            )
            if p.panel_id == "phase"
            else p
            for p in panels
        ]
    elif phase_requested:
        panels = [
            PlotPanel(
                panel_id=p.panel_id,
                kind=p.kind,
                y_label=r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]",
                x_label=r"$\phi$",
                invert_y=not is_flux,
                height_ratio=p.height_ratio,
                external_source=p.external_source,
                y_autorange=True,
                x_range=(0.0, 1.0),
                y_range=p.y_range,
            )
            if p.panel_id == "phase"
            else p
            for p in panels
        ]
        if request.phase_period_pending:
            phase_placeholder = f"{pending_period_source} is running..."
        elif request.suppress_catalog_phase_period:
            phase_placeholder = "No automatic harmonic-check period available. Run Find Period or enter Manual P."
        else:
            phase_placeholder = "No phase period available. Run Find Period or enter Manual P."
        annotations.append(
            PlotAnnotation(
                panel_id="phase",
                text=phase_placeholder,
                x=0.5,
                y=0.5,
                xref="domain",
                yref="domain",
                font_size=12.0,
            )
        )

    jd_panel_ids = [p.panel_id for p in panels if p.kind in {"raw", "resid", "external"}]
    if jd_panel_ids:
        x_label = rf"$\mathrm{{JD}} - {int(JD_OFFSET)}\ [\mathrm{{d}}]$"
        last_jd = jd_panel_ids[-1]
        panels = [
            PlotPanel(
                panel_id=p.panel_id,
                kind=p.kind,
                y_label=p.y_label,
                x_label=x_label if p.panel_id == last_jd else p.x_label,
                invert_y=p.invert_y,
                height_ratio=p.height_ratio,
                external_source=p.external_source,
                y_autorange=p.y_autorange,
                x_range=p.x_range,
                y_range=p.y_range,
            )
            for p in panels
        ]

    mag_anchor = _coerce_finite_float(request.payload.get("baseline_mag"))
    if mag_anchor is None:
        finite_mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy(dtype=float)
        finite_mag = finite_mag[np.isfinite(finite_mag)]
        if finite_mag.size:
            mag_anchor = float(np.nanmedian(finite_mag))

    ext_sources_with_traces: set[str] = set()
    if overlay_external_lcs and show_raw_panel:
        ext_traces, ext_sources_with_traces = build_external_traces(
            "raw",
            overlay_external_lcs,
            jd_offset,
            is_flux,
            mag_anchor=mag_anchor,
            warnings=warnings,
        )
        traces.extend(trace_from_dict(t) for t in ext_traces)
        panels = [
            PlotPanel(
                panel_id=p.panel_id,
                kind=p.kind,
                y_label=p.y_label,
                x_label=p.x_label,
                invert_y=p.invert_y,
                height_ratio=p.height_ratio,
                external_source=p.external_source,
                y_autorange=True if is_flux else "reversed",
                x_range=p.x_range,
                y_range=None,
            )
            if p.panel_id == "raw"
            else p
            for p in panels
        ]

    for source_name, lc_path in split_external_lcs.items():
        panel_id = f"external:{source_name}"
        ext_traces, split_sources = build_external_traces(
            panel_id,
            {source_name: lc_path},
            jd_offset,
            is_flux,
            mag_anchor=_coerce_finite_float(request.payload.get("baseline_mag")),
            warnings=warnings,
        )
        traces.extend(trace_from_dict(t) for t in ext_traces)
        ext_sources_with_traces.update(split_sources)
        annotations.append(
            PlotAnnotation(
                panel_id=panel_id,
                text=external_source_label(source_name),
                x=0.01,
                y=0.96,
                xref="domain",
                yref="domain",
                xanchor="left",
                yanchor="top",
                font_size=11.0,
                borderwidth=1.0,
                opacity=0.9,
            )
        )
        panels = [
            PlotPanel(
                panel_id=p.panel_id,
                kind=p.kind,
                y_label=r"$F$ [arb]" if is_flux else r"$m$ [mag]",
                x_label=p.x_label,
                invert_y=not is_flux,
                height_ratio=p.height_ratio,
                external_source=p.external_source,
                y_autorange=True if is_flux else "reversed",
                x_range=p.x_range,
                y_range=p.y_range,
            )
            if p.panel_id == panel_id
            else p
            for p in panels
        ]

    for source_name in active_external_lcs:
        if source_name not in ext_sources_with_traces:
            warnings.append(
                f"{external_source_label(source_name)} light curve has no plottable points for this candidate."
            )

    status_message = ""
    if phase_enabled and phase_period is not None:
        phase_bits = [
            f"{'Phase-time' if phase_panel_mode == 'time' else 'Phase-fold'} P={float(phase_period):.5f} d"
        ]
        if phase_source:
            phase_bits.append(f"source={phase_source}")
        phase_lag = float(phase_diag.get("phase_lag_g_v_cycles", np.nan))
        phase_lag_abs = float(phase_diag.get("phase_lag_g_v_abs_cycles", np.nan))
        if np.isfinite(phase_lag):
            phase_bits.append(f"g-V lag={phase_lag:+.3f} cycles")
            if np.isfinite(phase_lag_abs):
                phase_bits.append(f"|lag|={phase_lag_abs:.3f}")
        status_message = "; ".join(phase_bits)
    elif phase_requested and request.phase_period_pending:
        status_message = f"{pending_period_source}: searching..."
    elif phase_requested and request.suppress_catalog_phase_period:
        status_message = "Auto harmonic check: no valid period"

    header_left, header_right = publication_coordinate_headers(request.payload)

    return ReviewLightCurvePlotSpec(
        title=_build_title(request.payload, df),
        jd_offset=jd_offset,
        panels=tuple(panels),
        traces=tuple(traces),
        baselines=tuple(baselines),
        events=tuple(events),
        hlines=tuple(hlines),
        vlines=tuple(vlines),
        annotations=tuple(annotations),
        legend_panel_id="raw" if show_native_raw else None,
        header_left=header_left,
        header_right=header_right,
        warnings=tuple(warnings),
        status="ok",
        status_message=status_message,
        stat_rows=tuple(_build_stat_rows(request.payload, df, filtered_cameras)),
        camera_diagnostics=camera_diagnostics,
        camera_options=camera_options,
        camera_values=tuple(selected),
        phase_diagnostics=phase_diag,
        phase_period=phase_period,
        phase_panel_mode=phase_panel_mode,
        is_flux=is_flux,
        show_native_raw=show_native_raw,
        yaxis_mode=request.yaxis_mode,
        baseline_opacity=float(request.baseline_opacity),
        phase_requested=phase_requested,
        phase_enabled=phase_enabled,
        phase_period_pending=request.phase_period_pending,
        suppress_catalog_phase_period=request.suppress_catalog_phase_period,
        phase_source=phase_source,
        show_diagnostics=request.show_diagnostics,
        confidence_colors=request.confidence_colors,
        jd_match_panel_id=jd_panel_ids[0] if jd_panel_ids else None,
    )
