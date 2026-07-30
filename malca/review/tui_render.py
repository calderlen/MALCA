"""Fast light-curve image rendering for the terminal review interface.

The module's startup path deliberately has no Dash or Plotly imports.
Matplotlib, the MALCA light-curve loader, and the browser-compatible period
search helpers are imported only inside the render worker so starting the TUI
remains cheap.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Callable, Optional, Union
from urllib.request import Request, urlopen
import zlib

from malca.review.tui_photometry import (
    DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES,
    normalize_tui_external_photometry_sources,
)


PathLike = Union[str, os.PathLike]

VIEWER_CHOICES = ("window", "quicklook", "none")
QUICKLOOK_CLOSE_TIMEOUT_SECONDS = 1.0
LIGHTCURVE_EXTENSIONS = (".dat3", ".dat2", ".dat", ".csv")
PNG_FIGSIZE = (13.5, 10.0)
PNG_DPI = 180
RENDER_CACHE_VERSION = "tui-publication-panels-periodogram-sed-v57-period-window"
PHASE_MIN_PERIOD_DAYS = 0.1
PHASE_MAX_PERIOD_DAYS = 2000.0
PHASE_SEARCH_METHOD = "pipeline"
TUI_TIME_WINDOW_MODES = ("full", "asassn")
ASASSN_VISIT_GAP_DAYS = 0.5
DEFAULT_ASASSN_CADENCE_WINDOW_DAYS = 1.0

# Preset ``(min_days, max_days)`` search windows cycled via the TUI ``W`` key.
# Index 0 is overridden per candidate with baseline-adaptive review bounds.
PHASE_SEARCH_WINDOWS: tuple[tuple[float, float], ...] = (
    (0.1, 2000.0),
    (0.05, 1.0),
    (0.1, 10.0),
    (0.5, 100.0),
    (5.0, 1000.0),
)
DEFAULT_TUI_SURVEY_KEY = "decaps-dr2"
CUTOUT_FALLBACK_SURVEY_KEY = "dss2"
CUTOUT_TIMEOUT_SECONDS = 2.0
CUTOUT_BYTE_CACHE_SIZE = 24
CUTOUT_BYTE_CACHE_LIMIT = 24 * 1024 * 1024

_CUTOUT_BYTES_CACHE: OrderedDict[str, bytes] = OrderedDict()
_CMD_BACKGROUND_CACHE: OrderedDict[str, dict] = OrderedDict()
_CMD_BACKGROUND_CACHE_SIZE = 2

# Retain the established public name for tests and downstream imports while
# making the selected sources part of each render request.
TUI_EXTERNAL_LC_SOURCES = DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES
TUI_CADENCE_BIN_SOURCES = frozenset({"asas3", "crts"})
TUI_EXTERNAL_MARKER_MAP = {
    "diamond": "D",
    "triangle-up": "^",
    "x": "x",
    "cross": "+",
    "cross-open": "+",
    "star": "*",
    "hexagon": "h",
    "square": "s",
    "square-open": "s",
    "diamond-open": "D",
    "triangle-down-open": "v",
    "circle": "o",
}

# Publication-style light curve.  Single dark color for all ASAS-SN points and
# matching-color errorbars keep the panel visually indistinguishable from the
# paper's ``PUBLICATION_STYLE`` figures.
TUI_PUB_POINT_COLOR = "0.12"
TUI_PUB_ERROR_COLOR = "#cc2222"
TUI_PUB_LC_YLABEL = r"$m$ [mag]"
TUI_PUB_RESID_YLABEL = r"$\Delta m$ [mag]"
TUI_PUB_PHASE_XLABEL = r"$\phi$"

# Overlay defaults for contemporaneous survey photometry.  Points are drawn on
# top of the ASAS-SN scatter without changing the y-axis so absolute mags never
# affect scaling.
TUI_EXTERNAL_MARKER_SIZE_PT = 3.4
TUI_EXTERNAL_MARKER_ALPHA = 0.78
TUI_UNFILLED_MARKERS = frozenset({"x", "+", "X", "|", "_"})
TUI_PLOT_THEME_CYCLE = ("white", "black")
TUI_TICK_MAJOR_LEN = 7.5
TUI_TICK_MAJOR_WIDTH = 1.1
TUI_TICK_MINOR_LEN = 4.0
TUI_TICK_MINOR_WIDTH = 0.85
TUI_LEGEND_FONTSIZE = 9
TUI_LEGEND_HANDLE_LENGTH = 1.2
TUI_LEGEND_HANDLE_TEXTPAD = 0.4
TUI_LEGEND_COLUMN_SPACING = 0.9
TUI_LEGEND_BORDERPAD = 0.35
TUI_SED_LEGEND_INSET = (0.965, 0.965)
TUI_SED_LEGEND_FONTSIZE = 8.0
_CMU_BRIGHT_NATIVE_GLOB = "cmunb*.otf"
_CMU_BRIGHT_REGULAR_FILE = "cmunbmr.otf"
_CMU_BRIGHT_FAMILY = "CMU Bright"


def _format_tui_coordinate_header(ra_deg: float, dec_deg: float) -> str:
    """Format the decimal-coordinate box used in a rendered TUI image."""

    return f"α = {float(ra_deg):.4f}°, δ = {float(dec_deg):+.4f}°"


@dataclass(frozen=True)
class TuiPlotTheme:
    """Matplotlib colors for one review TUI raster theme."""

    name: str
    figure: str
    axes: str
    text: str
    spine: str
    tick: str
    grid: str
    point: str
    error: str
    placeholder_bg: str
    placeholder_text: str
    annotation_text: str
    annotation_face: str
    annotation_edge: str
    header_text: str
    header_face: str
    header_edge: str
    legend_face: str
    legend_edge: str
    legend_text: str
    cmd_bg_scatter: str
    phase_message: str
    phase_guide: str
    grid_alpha: float


def tui_plot_theme(mode: str | None) -> TuiPlotTheme:
    """Resolve TUI plot colors, matching Dash review themes."""
    normalized = str(mode or "black").strip().lower()
    if normalized == "black":
        return TuiPlotTheme(
            name="black",
            figure="#0d0d0d",
            axes="#0d0d0d",
            text="#dce8f2",
            spine="#8aa0b3",
            tick="#c5d4e3",
            grid="#607482",
            point="#e8f0f8",
            error="#ff7070",
            placeholder_bg="#141414",
            placeholder_text="#9fb0c0",
            annotation_text="#dce8f2",
            annotation_face="#1a2430",
            annotation_edge="#5f7385",
            header_text="#dce8f2",
            header_face="#141414",
            header_edge="#5f7385",
            legend_face="#141414",
            legend_edge="#5f7385",
            legend_text="#dce8f2",
            cmd_bg_scatter="#4a6880",
            phase_message="#9fb0c0",
            phase_guide="#9fb0c0",
            grid_alpha=0.55,
        )
    return TuiPlotTheme(
        name="white",
        figure="#ffffff",
        axes="#ffffff",
        text="#1c2733",
        spine="#1c2733",
        tick="#1c2733",
        grid="#688095",
        point=TUI_PUB_POINT_COLOR,
        error=TUI_PUB_ERROR_COLOR,
        placeholder_bg="#eeeeee",
        placeholder_text="#666666",
        annotation_text="0.10",
        annotation_face="#ffffff",
        annotation_edge="0.15",
        header_text="0.12",
        header_face="#ffffff",
        header_edge="0.15",
        legend_face="#ffffff",
        legend_edge="0.15",
        legend_text="#1c2733",
        cmd_bg_scatter="#96b7d4",
        phase_message="0.4",
        phase_guide="0.45",
        grid_alpha=0.18,
    )


@lru_cache(maxsize=1)
def _register_native_cmu_bright_fonts() -> bool:
    """Register CMU Bright OpenType faces without installing system fonts."""

    from matplotlib import font_manager

    if any(
        str(entry.name).strip() == _CMU_BRIGHT_FAMILY
        for entry in font_manager.fontManager.ttflist
    ):
        return True

    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return False
    try:
        completed = subprocess.run(
            [kpsewhich, _CMU_BRIGHT_REGULAR_FILE],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    regular_path = Path(str(completed.stdout or "").strip()).expanduser()
    if completed.returncode != 0 or not regular_path.is_file():
        return False

    registered = False
    for font_path in sorted(regular_path.parent.glob(_CMU_BRIGHT_NATIVE_GLOB)):
        try:
            font_manager.fontManager.addfont(str(font_path))
            registered = True
        except (OSError, RuntimeError, ValueError):
            continue
    return registered


def _review_style_for_theme(theme: TuiPlotTheme) -> dict[str, object]:
    from malca.plotting.lightcurve_publication import REVIEW_LIGHTCURVE_STYLE

    style = {
        **REVIEW_LIGHTCURVE_STYLE,
        "figure.facecolor": theme.figure,
        "axes.facecolor": theme.axes,
        "text.color": theme.text,
        "axes.labelcolor": theme.text,
        "axes.edgecolor": theme.spine,
        "xtick.color": theme.tick,
        "ytick.color": theme.tick,
        "xtick.major.size": TUI_TICK_MAJOR_LEN,
        "ytick.major.size": TUI_TICK_MAJOR_LEN,
        "xtick.major.width": TUI_TICK_MAJOR_WIDTH,
        "ytick.major.width": TUI_TICK_MAJOR_WIDTH,
        "xtick.minor.size": TUI_TICK_MINOR_LEN,
        "ytick.minor.size": TUI_TICK_MINOR_LEN,
        "xtick.minor.width": TUI_TICK_MINOR_WIDTH,
        "ytick.minor.width": TUI_TICK_MINOR_WIDTH,
    }
    if _register_native_cmu_bright_fonts():
        # CMU Bright is the OpenType form of Computer Modern Bright.  Native
        # FreeType/MathText rendering preserves that face without spawning a
        # separate latex and dvipng process for every uncached text string.
        style.pop("text.latex.preamble", None)
        style.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": [_CMU_BRIGHT_FAMILY],
                "mathtext.fontset": "custom",
                "mathtext.rm": _CMU_BRIGHT_FAMILY,
                "mathtext.it": f"{_CMU_BRIGHT_FAMILY}:italic",
                "mathtext.bf": f"{_CMU_BRIGHT_FAMILY}:bold",
                "mathtext.sf": _CMU_BRIGHT_FAMILY,
                "mathtext.cal": _CMU_BRIGHT_FAMILY,
                "mathtext.fallback": "cm",
            }
        )
    return style


def _apply_tui_tick_style(ax) -> None:
    """Enlarge in-axis ticks for the high-DPI TUI raster."""
    ax.tick_params(
        which="major",
        length=TUI_TICK_MAJOR_LEN,
        width=TUI_TICK_MAJOR_WIDTH,
        direction="in",
    )
    ax.tick_params(
        which="minor",
        length=TUI_TICK_MINOR_LEN,
        width=TUI_TICK_MINOR_WIDTH,
        direction="in",
    )


def _apply_tui_axis_theme(ax, theme: TuiPlotTheme) -> None:
    ax.set_facecolor(theme.axes)
    for spine in ax.spines.values():
        spine.set_color(theme.spine)
    ax.tick_params(axis="both", colors=theme.tick, which="both")
    ax.xaxis.label.set_color(theme.text)
    ax.yaxis.label.set_color(theme.text)
    title = ax.title
    if title is not None:
        title.set_color(theme.text)
    ax.grid(
        True,
        which="major",
        color=theme.grid,
        linewidth=0.4,
        alpha=theme.grid_alpha,
    )


def _style_tui_legend(ax, theme: TuiPlotTheme) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(theme.legend_face)
    frame.set_edgecolor(theme.legend_edge)
    frame.set_alpha(0.92)
    for text in legend.get_texts():
        text.set_color(theme.legend_text)


def _style_tui_sed_legend(ax, theme: TuiPlotTheme) -> None:
    """Match the light-curve legend while clearing inward axis ticks."""

    handles, labels = ax.get_legend_handles_labels()
    source_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if not str(label).startswith("Castelli/Kurucz")
    ]
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    if not source_entries:
        return
    handles, labels = zip(*source_entries)
    ax.legend(
        handles=handles[:6],
        labels=labels[:6],
        loc="upper right",
        bbox_to_anchor=TUI_SED_LEGEND_INSET,
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        fontsize=TUI_SED_LEGEND_FONTSIZE,
        frameon=True,
        framealpha=1.0,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=TUI_LEGEND_COLUMN_SPACING,
        borderpad=0.22,
        labelspacing=0.28,
    )
    _style_tui_legend(ax, theme)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(1.0)


def _draw_tui_placeholder_panel(ax, message: str, theme: TuiPlotTheme) -> None:
    ax.set_facecolor(theme.placeholder_bg)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=theme.placeholder_text,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


@dataclass(frozen=True)
class RenderRequest:
    """Everything needed to locate and label one review light curve."""

    candidate_id: str
    asas_sn_id: Optional[str] = None
    lc_path: Optional[PathLike] = None
    source_path: Optional[PathLike] = None
    db_path: Optional[PathLike] = None
    run_dir: Optional[PathLike] = None
    payload: Optional[dict] = None
    stored_phase_period_days: Optional[float] = None
    stored_phase_source: Optional[str] = None
    manual_phase_period_days: Optional[float] = None
    manual_phase_source: Optional[str] = None
    phase_multiplier: float = 1.0
    force_period_search: bool = False
    force_period_search_token: Optional[str] = None
    camera_view: str = "all"
    show_event_markers: bool = False
    color_by_camera: bool = False
    show_external_lightcurves: bool = True
    external_lightcurve_sources: tuple[str, ...] = (
        DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES
    )
    time_window_mode: str = "full"
    asassn_window_padding_days: float = 0.0
    plot_theme: str = "black"
    survey_key: str = DEFAULT_TUI_SURVEY_KEY
    phase_search_min_days: Optional[float] = None
    phase_search_max_days: Optional[float] = None


@dataclass(frozen=True)
class ImageStatus:
    """Current image state returned to the TUI without blocking."""

    state: str = "idle"
    candidate_id: Optional[str] = None
    path: Optional[Path] = None
    error: Optional[str] = None
    generation: int = 0
    phase_period_days: Optional[float] = None
    phase_source: Optional[str] = None
    survey_label: Optional[str] = None


@dataclass(frozen=True)
class _CutoutPanel:
    image: object | None
    label: str
    message: str
    overlay_fraction: float = 0.0


def _path(value: Optional[PathLike]) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "://" in text:
        return None
    return Path(text).expanduser()


def _existing_file(value: Optional[PathLike]) -> Optional[Path]:
    candidate = _path(value)
    if candidate is None:
        return None
    try:
        if candidate.is_file():
            return candidate.resolve()
    except OSError:
        return None
    return None


def _run_dir_from_anchor(value: Optional[PathLike], *, db: bool = False) -> Optional[Path]:
    """Infer a run root from a run/source path or a review DB path."""
    anchor = _path(value)
    if anchor is None:
        return None

    if db or anchor.suffix.lower() == ".db":
        anchor = anchor.parent
    elif anchor.is_file():
        anchor = anchor.parent

    # Common anchors are <run>, <run>/review, <run>/plots, a results file,
    # or <run>/bundle_assets/lightcurves.  Limit the walk so an unrelated
    # ancestor cannot unexpectedly become the active run.
    candidates = [anchor]
    candidates.extend(list(anchor.parents)[:3])
    for candidate in candidates:
        try:
            if (candidate / "bundle_assets" / "lightcurves").is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _identifier_names(request: RenderRequest) -> list[str]:
    names: list[str] = []
    for value in (request.asas_sn_id, request.candidate_id):
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none", "null", "<na>"}:
            continue
        # Candidate identifiers are names, not paths.  Taking the basename
        # prevents a malformed identifier from escaping the bundle directory.
        name = Path(text).name
        if name and name not in names:
            names.append(name)
    return names


def resolve_lightcurve_path(request: RenderRequest) -> Optional[Path]:
    """Resolve the local light curve for ``request`` without network access.

    An existing stored light-curve path wins.  Otherwise the active run is
    inferred from the explicit run directory, review database, or source path,
    in that order, and its bundled light curves are searched by ASAS-SN ID and
    then candidate ID.
    """
    stored_candidate = _path(request.lc_path)
    if (
        stored_candidate is not None
        and stored_candidate.suffix.lower() in LIGHTCURVE_EXTENSIONS
    ):
        stored = _existing_file(stored_candidate)
        if stored is not None:
            return stored

    # Some standalone imports store the light-curve file itself as source_path.
    # Candidate tables and other run artifacts can also be valid source paths,
    # however, and must not be handed to the light-curve loader.
    source_candidate = _path(request.source_path)
    if (
        source_candidate is not None
        and source_candidate.suffix.lower() in LIGHTCURVE_EXTENSIONS
    ):
        stored_source = _existing_file(source_candidate)
        if stored_source is not None:
            return stored_source

    run_dirs: list[Path] = []
    for anchor, is_db in (
        (request.run_dir, False),
        (request.db_path, True),
        (request.source_path, False),
    ):
        run_dir = _run_dir_from_anchor(anchor, db=is_db)
        if run_dir is not None and run_dir not in run_dirs:
            run_dirs.append(run_dir)

    for run_dir in run_dirs:
        lightcurve_dir = run_dir / "bundle_assets" / "lightcurves"
        for identifier in _identifier_names(request):
            for extension in LIGHTCURVE_EXTENSIONS:
                candidate = lightcurve_dir / f"{identifier}{extension}"
                try:
                    if candidate.is_file():
                        return candidate.resolve()
                except OSError:
                    continue
    return None


def _finite_positive(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _phase_multiplier(value: object) -> float:
    multiplier = _finite_positive(value)
    return multiplier if multiplier is not None else 1.0


def _camera_view(request: RenderRequest) -> str:
    value = str(request.camera_view or "all").strip().lower()
    return value if value in {"cleaned", "all"} else "all"


def _time_window_mode(request: RenderRequest) -> str:
    value = str(request.time_window_mode or "full").strip().lower()
    return value if value in TUI_TIME_WINDOW_MODES else "full"


def _asassn_window_padding_days(request: RenderRequest) -> float:
    try:
        padding = float(request.asassn_window_padding_days)
    except (TypeError, ValueError):
        return 0.0
    return padding if math.isfinite(padding) and padding >= 0.0 else 0.0


def _asassn_jd_window(
    jd,
    np,
    *,
    padding_days: float,
) -> tuple[float, float] | None:
    values = np.asarray(jd, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    padding = max(0.0, float(padding_days))
    if upper <= lower:
        padding = max(padding, 0.5)
    return lower - padding, upper + padding


def _asassn_cadence_window_days(jd, np) -> float:
    """Estimate one ASAS-SN visit-to-visit cadence from the displayed data."""

    values = np.asarray(jd, dtype=float)
    finite = np.unique(values[np.isfinite(values)])
    if finite.size < 2:
        return DEFAULT_ASASSN_CADENCE_WINDOW_DAYS
    finite.sort()

    # A target can have several camera measurements during one observing
    # visit. Collapse those before estimating the cadence so an intra-night
    # sequence does not turn the display bin width into a few minutes.
    visits: list[float] = []
    start = 0
    for index in range(1, finite.size):
        if float(finite[index] - finite[start]) > ASASSN_VISIT_GAP_DAYS:
            visits.append(float(np.median(finite[start:index])))
            start = index
    visits.append(float(np.median(finite[start:])))

    if len(visits) < 2:
        return DEFAULT_ASASSN_CADENCE_WINDOW_DAYS
    gaps = np.diff(np.asarray(visits, dtype=float))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if gaps.size == 0:
        return DEFAULT_ASASSN_CADENCE_WINDOW_DAYS
    cadence = float(np.median(gaps))
    return (
        cadence
        if math.isfinite(cadence) and cadence > 0
        else DEFAULT_ASASSN_CADENCE_WINDOW_DAYS
    )


def _combine_external_magnitude_cadence_bins(
    times,
    magnitudes,
    errors,
    *,
    window_days: float,
    np,
):
    """Combine external magnitudes in non-overlapping cadence-sized windows.

    Finite, positive uncertainties define inverse-variance weights. Missing
    uncertainties within an otherwise weighted bin receive that bin's median
    valid uncertainty so the corresponding photometric point is retained.
    """

    time_values = np.asarray(times, dtype=float)
    mag_values = np.asarray(magnitudes, dtype=float)
    if time_values.size != mag_values.size:
        raise ValueError("times and magnitudes must have equal length")

    error_values = None
    if errors is not None:
        error_values = np.asarray(errors, dtype=float)
        if error_values.size != time_values.size:
            raise ValueError("errors must match times and magnitudes")

    window = float(window_days)
    if not math.isfinite(window) or window <= 0:
        raise ValueError("window_days must be a positive finite number")

    good = np.isfinite(time_values) & np.isfinite(mag_values)
    time_values = time_values[good]
    mag_values = mag_values[good]
    if error_values is not None:
        error_values = error_values[good]
    if time_values.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty.copy(), empty.copy(), np.asarray([], dtype=int)

    order = np.argsort(time_values, kind="stable")
    time_values = time_values[order]
    mag_values = mag_values[order]
    if error_values is not None:
        error_values = error_values[order]

    combined_time: list[float] = []
    combined_mag: list[float] = []
    combined_err: list[float] = []
    combined_count: list[int] = []
    start = 0
    while start < time_values.size:
        stop = int(
            np.searchsorted(
                time_values,
                time_values[start] + window,
                side="right",
            )
        )
        bin_time = time_values[start:stop]
        bin_mag = mag_values[start:stop]
        bin_err = error_values[start:stop] if error_values is not None else None

        valid_err = (
            np.isfinite(bin_err) & (bin_err > 0)
            if bin_err is not None
            else np.zeros(bin_time.size, dtype=bool)
        )
        if valid_err.any():
            fallback_err = float(np.median(bin_err[valid_err]))
            effective_err = np.where(valid_err, bin_err, fallback_err)
            weights = np.reciprocal(np.square(effective_err))
            weight_sum = float(np.sum(weights))
            combined_time.append(float(np.sum(weights * bin_time) / weight_sum))
            combined_mag.append(float(np.sum(weights * bin_mag) / weight_sum))
            combined_err.append(float(np.sqrt(1.0 / weight_sum)))
        else:
            combined_time.append(float(np.mean(bin_time)))
            combined_mag.append(float(np.mean(bin_mag)))
            combined_err.append(float("nan"))
        combined_count.append(int(bin_time.size))
        start = stop

    return (
        np.asarray(combined_time, dtype=float),
        np.asarray(combined_mag, dtype=float),
        np.asarray(combined_err, dtype=float),
        np.asarray(combined_count, dtype=int),
    )


def _with_phase_multiplier(
    request: RenderRequest,
    period: Optional[float],
    source: str,
    warning: str,
) -> tuple[Optional[float], str, str]:
    """Apply the display-period multiplier after resolving the base period."""
    if period is None:
        return None, source, warning
    multiplier = _phase_multiplier(request.phase_multiplier)
    if multiplier == 1.0:
        return period, source, warning
    return period * multiplier, f"{source} ×{multiplier:g}", warning


def _run_root_for_request(
    request: RenderRequest,
    lc_path: Optional[Path] = None,
) -> Optional[Path]:
    for anchor, is_db in (
        (request.run_dir, False),
        (request.db_path, True),
        (request.source_path, False),
        (lc_path, False),
    ):
        run_dir = _run_dir_from_anchor(anchor, db=is_db)
        if run_dir is not None:
            return run_dir
    return None


def _run_params_path(request: RenderRequest, lc_path: Optional[Path] = None) -> Optional[Path]:
    run_root = _run_root_for_request(request, lc_path)
    if run_root is None:
        return None
    candidate = run_root / "run_params.json"
    try:
        return candidate.resolve() if candidate.is_file() else None
    except OSError:
        return None


def _load_run_params(request: RenderRequest, lc_path: Path) -> dict:
    params_path = _run_params_path(request, lc_path)
    if params_path is None:
        return {}
    try:
        with params_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _config_float(params: dict, key: str, default: float) -> float:
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _period_processing_config(request: RenderRequest, lc_path: Path) -> dict:
    """Return the same cleaning and baseline policy used by browser review."""
    from malca.config import (
        BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
        CLEAN_LC_MAX_ERROR_ABSOLUTE,
        CLEAN_LC_MAX_ERROR_SIGMA,
    )
    from malca.review.native_lightcurve import _baseline_config_from_run_params

    params = _load_run_params(request, lc_path)
    baseline_name, baseline_kwargs, _warnings = _baseline_config_from_run_params(params)
    camera_view = _camera_view(request)
    return {
        # The TUI switch is authoritative.  It controls both the visible raw
        # panel and the data used for automatic period search / phase folding.
        "filter_bad_cameras": camera_view == "cleaned",
        "scatter_ratio": _config_float(
            params,
            "bad_camera_scatter_ratio",
            BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
        ),
        "clean_max_error_absolute": _config_float(
            params,
            "clean_max_error_absolute",
            CLEAN_LC_MAX_ERROR_ABSOLUTE,
        ),
        "clean_max_error_sigma": _config_float(
            params,
            "clean_max_error_sigma",
            CLEAN_LC_MAX_ERROR_SIGMA,
        ),
        "baseline_name": baseline_name,
        "baseline_kwargs": baseline_kwargs,
    }


def _period_cleaning_kwargs(config: dict) -> dict:
    """Return only the camera-cleaning kwargs accepted by pipeline period search."""

    return {
        key: config[key]
        for key in (
            "filter_bad_cameras",
            "scatter_ratio",
            "clean_max_error_absolute",
            "clean_max_error_sigma",
        )
        if key in config
    }


def _period_payload(request: RenderRequest, lc_path: Path) -> Optional[dict]:
    if not isinstance(request.payload, dict):
        return None
    payload = dict(request.payload)
    payload["candidate_id"] = str(request.candidate_id)
    if request.asas_sn_id:
        payload["asas_sn_id"] = str(request.asas_sn_id)
    # Force both native resolver fields to the exact local file already chosen
    # by the TUI.  Imported payloads can retain a stale cluster-only path.
    payload["path"] = str(lc_path)
    payload["lc_path"] = str(lc_path)
    return payload


def _resolve_phase_search_bounds(
    request: RenderRequest,
    lc_path: Path,
    *,
    payload: dict | None = None,
) -> tuple[float, float]:
    """Return the min/max period window used for search and periodogram plots."""

    stored_period = _finite_positive(request.stored_phase_period_days)
    if payload is None:
        payload = _period_payload(request, lc_path)

    min_period = _finite_positive(request.phase_search_min_days)
    max_period = _finite_positive(request.phase_search_max_days)
    if min_period is None or max_period is None or max_period <= min_period:
        if isinstance(payload, dict):
            from malca.review.period_search import adaptive_review_period_bounds

            min_period, max_period = adaptive_review_period_bounds(payload)
        else:
            min_period, max_period = PHASE_MIN_PERIOD_DAYS, PHASE_MAX_PERIOD_DAYS
    if max_period <= min_period:
        min_period, max_period = PHASE_MIN_PERIOD_DAYS, PHASE_MAX_PERIOD_DAYS
    if (
        not request.force_period_search
        and stored_period is not None
        and float(stored_period) > float(max_period)
    ):
        from malca.review.period_search import payload_baseline_days

        baseline = None
        if isinstance(payload, dict):
            baseline = payload_baseline_days(payload)
        expanded_max = float(stored_period) * 1.05
        if baseline is not None and math.isfinite(baseline) and baseline > 0:
            expanded_max = min(expanded_max, float(baseline) / 1.2)
        max_period = max(float(max_period), expanded_max)
    return float(min_period), float(max_period)


def _resolve_best_phase_period(
    request: RenderRequest,
    lc_path: Path,
) -> tuple[Optional[float], str, str]:
    """Resolve the browser-equivalent automatic period for one TUI image.

    Uses the STV pipeline consensus search when no trustworthy stored period
    exists (or when forced), and a harmonic check otherwise. Search bounds
    default to baseline-adaptive review limits when not overridden.
    """
    manual_period = _finite_positive(request.manual_phase_period_days)
    if manual_period is not None:
        manual_source = str(request.manual_phase_source or "Manual").strip() or "Manual"
        return _with_phase_multiplier(request, manual_period, manual_source, "")

    stored_period = _finite_positive(request.stored_phase_period_days)
    stored_source = str(request.stored_phase_source or "stored period").strip()
    payload = _period_payload(request, lc_path)

    # Direct renderer callers may intentionally omit a payload.  In that case
    # an explicitly supplied period is already final and no hidden search runs.
    if payload is None:
        if stored_period is not None and not request.force_period_search:
            return _with_phase_multiplier(request, stored_period, stored_source, "")
        return None, "", "No candidate payload for automatic period search"

    config = _period_processing_config(request, lc_path)
    run_root = _run_root_for_request(request, lc_path)
    plot_dir = run_root / "plots" if run_root is not None else None

    min_period, max_period = _resolve_phase_search_bounds(
        request,
        lc_path,
        payload=payload,
    )

    try:
        from malca.review.period_search import (
            run_harmonic_check_for_payload,
            run_pipeline_period_search_for_payload,
        )

        needs_pipeline = bool(
            request.force_period_search
            or stored_period is None
        )
        if needs_pipeline:
            result, message = run_pipeline_period_search_for_payload(
                payload,
                plot_dir=plot_dir,
                min_period=min_period,
                max_period=max_period,
                **_period_cleaning_kwargs(config),
            )
            if request.force_period_search:
                source = "Pipeline search (forced)"
            else:
                source = "Pipeline search"
        else:
            payload["phase_period_days"] = stored_period
            payload["phase_source"] = stored_source
            result, message = run_harmonic_check_for_payload(
                payload,
                plot_dir=plot_dir,
                min_period=min_period,
                max_period=max_period,
                **config,
            )
            source = "Auto harmonic check"
    except Exception as exc:
        if stored_period is not None and not request.force_period_search:
            return _with_phase_multiplier(
                request,
                stored_period,
                stored_source,
                f"Automatic period search failed: {exc}",
            )
        return None, "", f"Automatic period search failed: {exc}"

    best_period = (
        _finite_positive(result.get("best_period"))
        if isinstance(result, dict)
        else None
    )
    if isinstance(result, dict):
        result_method = str(result.get("period_method") or result.get("method") or "")
        if result_method.endswith("_review_candidate"):
            method_name = result_method.removesuffix("_review_candidate").upper()
            source = f"{method_name} candidate (weak)"
    if (
        request.force_period_search
        and best_period is not None
        and not (float(min_period) <= best_period <= float(max_period))
    ):
        rejected_period = best_period
        best_period = None
        message = (
            f"Rejected P={rejected_period:.9g} d outside forced "
            f"{min_period:g}–{max_period:g} d window"
        )
    if best_period is not None:
        return _with_phase_multiplier(request, best_period, source, "")
    if stored_period is not None and not request.force_period_search:
        note = str(message or "Automatic period search failed")
        return _with_phase_multiplier(request, stored_period, stored_source, note)
    return None, source, str(message or "No valid automatic period")


def _prepare_review_phase_source(request: RenderRequest, lc_path: Path):
    """Prepare browser-equivalent detrended data for the publication phase plot."""
    import pandas as pd

    from malca.core.phase import resolve_phase_epoch
    from malca.review.native_lightcurve import (
        _compute_baseline_bands,
        _load_cleaned_df,
    )

    config = _period_processing_config(request, lc_path)
    cleaned, _, _ = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=bool(config["filter_bad_cameras"]),
        scatter_ratio=float(config["scatter_ratio"]),
        clean_max_error_absolute=float(config["clean_max_error_absolute"]),
        clean_max_error_sigma=float(config["clean_max_error_sigma"]),
    )
    if cleaned is None or cleaned.empty:
        return pd.DataFrame(), None

    baseline_cache_key = (
        str(lc_path.resolve()),
        (),
        bool(config["filter_bad_cameras"]),
        float(config["scatter_ratio"]),
        float(config["clean_max_error_absolute"]),
        float(config["clean_max_error_sigma"]),
    )
    band_dfs = _compute_baseline_bands(
        cleaned,
        str(config["baseline_name"]),
        baseline_cache_key,
        baseline_kwargs=dict(config["baseline_kwargs"]),
    )
    inputs = [
        band_dfs[band]
        for band in (0, 1)
        if band in band_dfs and band_dfs[band] is not None and not band_dfs[band].empty
    ]
    if not inputs:
        return pd.DataFrame(), resolve_phase_epoch(cleaned)
    source = pd.concat(inputs, ignore_index=True)
    return source, resolve_phase_epoch(cleaned)


def _ensure_mpl_config_dir() -> Path:
    """Select a deterministic writable Matplotlib config directory."""
    config_dir = Path(tempfile.gettempdir()) / "malca-review-tui-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    configured = os.environ.get("MPLCONFIGDIR")
    if not configured:
        os.environ["MPLCONFIGDIR"] = str(config_dir)
        return config_dir

    configured_path = Path(configured).expanduser()
    try:
        configured_path.mkdir(parents=True, exist_ok=True)
        if os.access(configured_path, os.W_OK):
            return configured_path
    except OSError:
        pass
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    return config_dir


def _load_render_dependencies():
    """Lazy-load plotting and light-curve dependencies in the worker."""
    _ensure_mpl_config_dir()

    import warnings

    # SED assembly still emits pandas FutureWarnings on some sparse tables.
    # The TUI render worker shares stderr with curses, so keep it quiet.
    warnings.simplefilter("ignore", category=FutureWarning)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    from malca.io.lightcurve_io import load_lightcurve_df

    return plt, np, load_lightcurve_df


def _load_display_frame(request: RenderRequest, lc_path: Path, load_lightcurve_df):
    """Load the raw panel with the same camera policy as the phase panel."""
    if _camera_view(request) == "all":
        return load_lightcurve_df(lc_path, apply_quality=True)

    import pandas as pd

    from malca.review.native_lightcurve import _load_cleaned_df

    config = _period_processing_config(request, lc_path)
    cleaned, _filtered_cameras, _diagnostics = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=True,
        scatter_ratio=float(config["scatter_ratio"]),
        clean_max_error_absolute=float(config["clean_max_error_absolute"]),
        clean_max_error_sigma=float(config["clean_max_error_sigma"]),
    )
    if cleaned is None or cleaned.empty:
        return pd.DataFrame()

    frame = pd.DataFrame(index=cleaned.index)
    frame["jd"] = pd.to_numeric(cleaned.get("JD"), errors="coerce")
    frame["mag"] = pd.to_numeric(cleaned.get("mag"), errors="coerce")
    frame["mag_err"] = pd.to_numeric(cleaned.get("error"), errors="coerce")
    if "camera_name" in cleaned.columns:
        frame["camera_name"] = cleaned["camera_name"]
    else:
        frame["camera_name"] = cleaned.get("camera#", "unknown")
    frame["camera"] = cleaned.get("camera#", frame["camera_name"])
    if "v_g_band" in cleaned.columns:
        frame["band"] = cleaned["v_g_band"].map(
            lambda value: "g" if str(value).strip() in {"0", "0.0", "g", "G"} else "V"
        )
    else:
        frame["band"] = ""
    return frame.reset_index(drop=True)


def _fetch_cutout_bytes(url: str, timeout: float = CUTOUT_TIMEOUT_SECONDS) -> bytes:
    """Fetch a survey tile into a small bounded in-memory cache."""
    cached = _CUTOUT_BYTES_CACHE.get(url)
    if cached is not None:
        _CUTOUT_BYTES_CACHE.move_to_end(url)
        return cached

    request = Request(url, headers={"User-Agent": "MALCA-review-tui/1"})
    with urlopen(request, timeout=float(timeout)) as response:
        content = response.read(12 * 1024 * 1024 + 1)
    if not content:
        raise ValueError("survey returned an empty response")
    if len(content) > 12 * 1024 * 1024:
        raise ValueError("survey image exceeded 12 MiB")

    _CUTOUT_BYTES_CACHE[url] = content
    _CUTOUT_BYTES_CACHE.move_to_end(url)
    while (
        len(_CUTOUT_BYTES_CACHE) > CUTOUT_BYTE_CACHE_SIZE
        or sum(len(value) for value in _CUTOUT_BYTES_CACHE.values())
        > CUTOUT_BYTE_CACHE_LIMIT
    ):
        _CUTOUT_BYTES_CACHE.popitem(last=False)
    return content


def _decode_cutout_bytes(content: bytes, plt):
    # HiPS2FITS serves JPEG bytes even though a file-like object has no suffix.
    # ``matplotlib.imread`` consequently assumes PNG and rejects a valid tile.
    # Pillow sniffs the actual bytes and handles either JPEG or PNG reliably.
    del plt
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(content)) as decoded:
        return np.asarray(decoded.convert("RGB"))


def _is_blank_cutout(image, np) -> bool:
    """Treat uniform and near-uniform tiles as missing survey coverage."""
    values = np.asarray(image, dtype=float)
    if values.size == 0:
        return True
    if values.ndim >= 3:
        values = values[..., :3].mean(axis=-1)
    finite = values[np.isfinite(values)]
    if finite.size < 16:
        return True
    if float(np.nanmax(finite)) > 1.5:
        finite = finite / 255.0
    low, high = np.nanpercentile(finite, [1.0, 99.0])
    return bool(float(high - low) < 0.015 or float(np.nanstd(finite)) < 0.004)


def _survey_cutout(request: RenderRequest, plt, np) -> _CutoutPanel:
    """Fetch DECaPS first and transparently fall back to DSS2 when blank."""
    from malca.review.cutouts import cutout_payload_for_candidate

    payload = request.payload if isinstance(request.payload, dict) else None
    primary = cutout_payload_for_candidate(
        payload,
        selected_key=request.survey_key or DEFAULT_TUI_SURVEY_KEY,
        prefer_compatible=False,
    )
    overlay_fraction = float(primary.get("asassn_fwhm_overlay_fraction") or 0.0)
    if not primary.get("has_coordinates") or not primary.get("image_url"):
        return _CutoutPanel(
            None,
            "",
            str(primary.get("message") or "No RA/Dec"),
            overlay_fraction,
        )

    def fetch(cutout_payload: dict):
        content = _fetch_cutout_bytes(str(cutout_payload["image_url"]))
        decoded = _decode_cutout_bytes(content, plt)
        if _is_blank_cutout(decoded, np):
            raise ValueError("blank or near-uniform tile")
        return decoded

    primary_error = ""
    try:
        image = fetch(primary)
        return _CutoutPanel(
            image,
            "",
            "",
            overlay_fraction,
        )
    except Exception as exc:
        primary_error = str(exc)

    if str(primary.get("selected_key") or "") != CUTOUT_FALLBACK_SURVEY_KEY:
        fallback = cutout_payload_for_candidate(
            payload,
            selected_key=CUTOUT_FALLBACK_SURVEY_KEY,
            prefer_compatible=False,
        )
        if fallback.get("has_coordinates") and fallback.get("image_url"):
            try:
                image = fetch(fallback)
                return _CutoutPanel(
                    image,
                    "",
                    "",
                    float(fallback.get("asassn_fwhm_overlay_fraction") or overlay_fraction),
                )
            except Exception as exc:
                if primary_error:
                    primary_error = f"{primary_error}; fallback failed ({exc})"
                else:
                    primary_error = f"fallback failed ({exc})"

    return _CutoutPanel(
        None,
        "",
        "Image unavailable" if primary_error else "No coordinates",
        overlay_fraction,
    )


def _load_cmd_background(request: RenderRequest) -> dict | None:
    """Load review-candidate CMD background arrays from the active review DB."""
    db_path = _path(request.db_path)
    if db_path is None:
        return None
    try:
        if not db_path.is_file():
            return None
    except OSError:
        return None

    key = str(db_path.resolve())
    cached = _CMD_BACKGROUND_CACHE.get(key)
    if cached is not None:
        _CMD_BACKGROUND_CACHE.move_to_end(key)
        return cached

    try:
        from contextlib import closing

        from malca.review.store import db_connect, get_diagnostic_background

        with closing(db_connect(db_path)) as conn:
            background = get_diagnostic_background(conn)
    except Exception:
        return None

    _CMD_BACKGROUND_CACHE[key] = background
    _CMD_BACKGROUND_CACHE.move_to_end(key)
    while len(_CMD_BACKGROUND_CACHE) > _CMD_BACKGROUND_CACHE_SIZE:
        _CMD_BACKGROUND_CACHE.popitem(last=False)
    return background


def _plot_cmd_background(
    ax,
    bg_x,
    bg_y,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    np,
    scatter_color: str = "#96b7d4",
) -> None:
    if bg_x.size == 0 or bg_y.size == 0:
        return
    mask = (
        (bg_x >= xlim[0])
        & (bg_x <= xlim[1])
        & (bg_y >= ylim[0])
        & (bg_y <= ylim[1])
    )
    if not mask.any():
        return
    visible_x = np.asarray(bg_x[mask], dtype=float)
    visible_y = np.asarray(bg_y[mask], dtype=float)
    if visible_x.size > 8000:
        step = int(np.ceil(visible_x.size / 8000.0))
        visible_x = visible_x[::step]
        visible_y = visible_y[::step]
    ax.scatter(
        visible_x,
        visible_y,
        s=2.8,
        c=scatter_color,
        alpha=0.28,
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )


def _draw_cutout_axis(ax, cutout: _CutoutPanel, plt, *, theme: TuiPlotTheme) -> None:
    """Render a single-band survey cutout without any title decoration."""
    if cutout.image is not None:
        ax.imshow(cutout.image, origin="upper", extent=(0.0, 1.0, 0.0, 1.0))
        if cutout.overlay_fraction > 0:
            ax.add_patch(
                plt.Circle(
                    (0.5, 0.5),
                    cutout.overlay_fraction / 2.0,
                    fill=False,
                    edgecolor="#00e5ff",
                    linewidth=1.3,
                    alpha=0.95,
                )
            )
    else:
        _draw_tui_placeholder_panel(ax, cutout.message or "Cutout unavailable", theme)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _metadata_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def _write_render_metadata(image_path: Path, metadata: dict[str, object]) -> None:
    destination = _metadata_path(image_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".json", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_render_metadata(image_path: Path) -> dict[str, object]:
    try:
        with _metadata_path(image_path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_replace_from_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix or ".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _write_error_placeholder_png(destination: Path) -> Path:
    """Write a dependency-free PNG that cannot be mistaken for a light curve."""
    width = int(PNG_FIGSIZE[0] * PNG_DPI)
    height = int(PNG_FIGSIZE[1] * PNG_DPI)
    accent_row = b"\x00" + bytes((176, 48, 60)) * width
    body_row = b"\x00" + bytes((242, 242, 242)) * width
    raw_pixels = accent_row * 56 + body_row * (height - 56)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw_pixels, level=9))
        + _png_chunk(b"IEND", b"")
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(png)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _jd_plot_offset(jd_values, np) -> float:
    """Return the plot-time JD offset applied to the ASAS-SN frame.

    ``plot_lightcurve_panel`` accepts a numeric ``time_offset``; sharing one
    offset value between the ASAS-SN panel, external overlays, and event
    markers keeps every artist on the same x-axis without recomputing.
    """
    from malca.config import JD_OFFSET

    finite = np.asarray(jd_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0
    return float(JD_OFFSET) if float(np.nanmedian(finite)) > 2_000_000.0 else 0.0


def _external_lc_jd_values(df_ext, spec, np):
    """Return a NumPy array of JD (TDB-agnostic) times or ``None`` when missing."""
    import pandas as pd

    from malca.config import (
        GAIA_TCB_EPOCH_JD,
        KEPLER_BKJD_OFFSET,
        MJD_TO_JD,
        TESS_BTJD_OFFSET,
    )

    time_col = str(spec.get("time_col") or "")
    if not time_col:
        return None
    actual_time = next(
        (column for column in df_ext.columns if column.lower() == time_col.lower()),
        None,
    )
    if actual_time is None:
        return None
    times = np.asarray(
        pd.to_numeric(df_ext[actual_time], errors="coerce"), dtype=float
    )
    jd_sys = str(spec.get("jd_system", "mjd") or "mjd")
    if jd_sys == "mjd":
        finite_t = times[np.isfinite(times)]
        if finite_t.size and float(np.nanmedian(finite_t)) > 1_000_000.0:
            return times
        return times + MJD_TO_JD
    if jd_sys == "bjd_gaia":
        return times + GAIA_TCB_EPOCH_JD
    if jd_sys == "btjd":
        return times + TESS_BTJD_OFFSET
    if jd_sys == "bkjd":
        return times + KEPLER_BKJD_OFFSET
    return times


def _overlay_tui_external_lightcurves(
    ax,
    request: RenderRequest,
    lc_path: Path,
    *,
    asas_median: float,
    jd_offset: float,
    np,
    jd_window: tuple[float, float] | None = None,
    cadence_window_days: float | None = None,
) -> list[tuple[str, str, str]]:
    """Overlay selected median-shifted external photometry on the raw LC axis.

    Returns a ``(label, color, marker_char)`` tuple per band actually drawn
    so the caller can build a legend keyed on the artists that made it onto
    the plot.  Every branch swallows exceptions so a malformed external
    artifact never prevents the main light curve from rendering.
    """
    import pandas as pd

    drawn: list[tuple[str, str, str]] = []

    try:
        from malca.review.lightcurve_sources import (
            EXTERNAL_LC_SPECS,
            discover_external_lcs,
            load_external_lc_frame,
        )
    except Exception:
        return drawn

    payload = request.payload if isinstance(request.payload, dict) else {}
    run_root = _run_root_for_request(request, lc_path)
    default_results = run_root / "results" if run_root is not None else None
    try:
        requested_sources = normalize_tui_external_photometry_sources(
            request.external_lightcurve_sources
        )
        if not requested_sources:
            return drawn
        external = discover_external_lcs(
            request.candidate_id,
            payload,
            lc_path.parent,
            list(requested_sources),
            default_results_root=default_results,
        )
    except Exception:
        return drawn
    if not external:
        return drawn

    if not math.isfinite(asas_median):
        return drawn

    for source_name, external_path in external.items():
        spec = EXTERNAL_LC_SPECS.get(source_name)
        if spec is None or spec.get("is_flux"):
            continue
        try:
            df_ext = load_external_lc_frame(source_name, external_path)
        except Exception:
            continue
        if df_ext is None or df_ext.empty:
            continue

        jd = _external_lc_jd_values(df_ext, spec, np)
        if jd is None or jd.size == 0:
            continue

        col_lookup = {str(column).lower(): column for column in df_ext.columns}
        filter_col = spec.get("filter_col")
        actual_filt = col_lookup.get(str(filter_col or "").lower()) if filter_col else None
        default_mag_col = str(spec.get("mag_col") or "")

        for band_key, band_info in spec.get("bands", {}).items():
            if source_name == "neowise" and band_key not in {"W1", "W2"}:
                continue
            band_mag_col = str(band_info.get("mag_col") or default_mag_col)
            band_err_col = str(band_info.get("err_col") or spec.get("err_col") or "")
            actual_mag = col_lookup.get(band_mag_col.lower())
            actual_err = col_lookup.get(band_err_col.lower()) if band_err_col else None
            if actual_mag is None:
                continue

            if actual_filt is not None:
                mask = np.asarray(
                    df_ext[actual_filt].astype(str) == str(band_key), dtype=bool
                )
                if not mask.any():
                    continue
                band_jd = jd[mask] if jd.size == mask.size else jd
                raw_y = np.asarray(
                    pd.to_numeric(df_ext.loc[mask, actual_mag], errors="coerce"),
                    dtype=float,
                )
                raw_err = (
                    np.asarray(
                        pd.to_numeric(df_ext.loc[mask, actual_err], errors="coerce"),
                        dtype=float,
                    )
                    if actual_err is not None
                    else None
                )
            else:
                band_jd = jd
                raw_y = np.asarray(
                    pd.to_numeric(df_ext[actual_mag], errors="coerce"), dtype=float
                )
                raw_err = (
                    np.asarray(
                        pd.to_numeric(df_ext[actual_err], errors="coerce"), dtype=float
                    )
                    if actual_err is not None
                    else None
                )

            good = np.isfinite(band_jd) & np.isfinite(raw_y)
            if jd_window is not None:
                good &= (band_jd >= float(jd_window[0])) & (
                    band_jd <= float(jd_window[1])
                )
            if not good.any():
                continue

            band_jd = band_jd[good]
            raw_y = raw_y[good]
            raw_err = raw_err[good] if raw_err is not None else None
            if (
                source_name in TUI_CADENCE_BIN_SOURCES
                and cadence_window_days is not None
            ):
                try:
                    band_jd, raw_y, binned_err, _bin_counts = (
                        _combine_external_magnitude_cadence_bins(
                            band_jd,
                            raw_y,
                            raw_err,
                            window_days=cadence_window_days,
                            np=np,
                        )
                    )
                    raw_err = binned_err
                except (TypeError, ValueError):
                    pass
            if band_jd.size == 0:
                continue

            band_median = float(np.nanmedian(raw_y))
            if not math.isfinite(band_median):
                continue

            display_y = raw_y - band_median + asas_median
            plot_x = band_jd - float(jd_offset)
            display_err = None
            if raw_err is not None:
                if np.isfinite(raw_err).any():
                    display_err = raw_err

            marker_name = str(band_info.get("marker") or "o")
            marker_char = TUI_EXTERNAL_MARKER_MAP.get(marker_name, "o")
            label = str(band_info.get("label") or f"{source_name} {band_key}")
            scatter_kwargs = {
                "s": TUI_EXTERNAL_MARKER_SIZE_PT**2,
                "marker": marker_char,
                "linewidths": 0.55,
                "alpha": TUI_EXTERNAL_MARKER_ALPHA,
                "zorder": 2,
                "rasterized": True,
                "label": label,
            }
            if marker_char in TUI_UNFILLED_MARKERS:
                scatter_kwargs["color"] = band_info["color"]
            elif marker_name.endswith("-open"):
                scatter_kwargs["facecolors"] = "none"
                scatter_kwargs["edgecolors"] = band_info["color"]
            else:
                scatter_kwargs["facecolors"] = band_info["color"]
                scatter_kwargs["edgecolors"] = band_info["color"]
            if display_err is not None:
                valid_err = np.isfinite(display_err) & (display_err > 0)
                if valid_err.any():
                    ax.errorbar(
                        plot_x[valid_err],
                        display_y[valid_err],
                        yerr=display_err[valid_err],
                        fmt="none",
                        ecolor=band_info["color"],
                        elinewidth=0.55,
                        capsize=1.2,
                        alpha=0.55,
                        zorder=1,
                    )
            ax.scatter(
                plot_x,
                display_y,
                **scatter_kwargs,
            )
            drawn.append((label, str(band_info["color"]), marker_char))

    return drawn


def _render_cmd_panel(
    ax,
    payload: object | None,
    np,
    *,
    background: dict | None = None,
    theme: TuiPlotTheme,
) -> bool:
    from malca.ltv.cmd import dustmaps_cmd_from_fields

    if not isinstance(payload, dict):
        _draw_tui_placeholder_panel(ax, "CMD unavailable", theme)
        return False

    coords = dustmaps_cmd_from_fields(
        g_mag=payload.get("phot_g_mean_mag") or payload.get("gaia_phot_g_mean_mag"),
        bp_rp=payload.get("bp_rp") or payload.get("derived_bp_rp"),
        dist_pc=payload.get("distance_gspphot"),
        a_v_3d=payload.get("A_v_3d"),
        bp_mag=payload.get("phot_bp_mean_mag"),
        rp_mag=payload.get("phot_rp_mean_mag"),
        parallax_mas=payload.get("parallax"),
    )
    if coords.get("cmd_coordinate_source") == "missing":
        _draw_tui_placeholder_panel(ax, "CMD unavailable", theme)
        return False

    bp_rp = float(coords["bp_rp"])
    mg = float(coords["mg"])
    bp_rp0 = float(coords["cmd_color"])
    mg0 = float(coords["cmd_mag"])

    bg_x = np.asarray((background or {}).get("cmd_bprp0"), dtype=float).reshape(-1)
    bg_y = np.asarray((background or {}).get("cmd_mg0"), dtype=float).reshape(-1)
    if bg_x.size and bg_y.size == bg_x.size:
        finite = np.isfinite(bg_x) & np.isfinite(bg_y)
        bg_x = bg_x[finite]
        bg_y = bg_y[finite]
    else:
        bg_x = np.empty(0, dtype=float)
        bg_y = np.empty(0, dtype=float)

    x_values = [bp_rp0, bp_rp, -0.5, 5.0]
    if bg_x.size:
        x_values.extend([float(np.nanquantile(bg_x, 0.01)), float(np.nanquantile(bg_x, 0.99))])
    y_values = [mg0, mg, -8.0, 16.0]
    if bg_y.size:
        y_values.extend([float(np.nanquantile(bg_y, 0.01)), float(np.nanquantile(bg_y, 0.99))])
    xlim = (max(-0.8, min(x_values) - 0.2), min(5.2, max(x_values) + 0.2))
    ylim = (max(-8.0, min(y_values) - 0.6), min(16.0, max(y_values) + 0.6))

    _plot_cmd_background(
        ax,
        bg_x,
        bg_y,
        xlim=xlim,
        ylim=ylim,
        np=np,
        scatter_color=theme.cmd_bg_scatter,
    )

    if abs(bp_rp0 - bp_rp) > 0.01 or abs(mg0 - mg) > 0.01:
        ax.scatter(
            [bp_rp],
            [mg],
            s=24,
            c="#d97706",
            edgecolors="0.15",
            linewidths=0.4,
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(bp_rp0, mg0),
            xytext=(bp_rp, mg),
            arrowprops={
                "arrowstyle": "->",
                "color": "#6b7280",
                "linewidth": 0.8,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=3,
        )

    ax.scatter(
        [bp_rp0],
        [mg0],
        s=36,
        c="#e8c547",
        edgecolors="0.15",
        linewidths=0.45,
        zorder=5,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_xlabel(r"$(\mathrm{BP}-\mathrm{RP})_0$")
    ax.set_ylabel(r"$M_{G,0}$")
    _apply_tui_tick_style(ax)
    return True


def _compute_periodogram_curve(
    phase_source_frame,
    *,
    min_period: float,
    max_period: float,
    np,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Lomb-Scargle power spectrum over the active review search window."""

    if phase_source_frame is None or phase_source_frame.empty:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    time_col = "JD" if "JD" in phase_source_frame.columns else "jd"
    value_col = "resid" if "resid" in phase_source_frame.columns else "mag"
    if time_col not in phase_source_frame.columns or value_col not in phase_source_frame.columns:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    times = np.asarray(phase_source_frame[time_col], dtype=float)
    values = np.asarray(phase_source_frame[value_col], dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.size < 20 or max_period <= min_period:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    from malca.core.periodogram import lsp_find_period

    try:
        _, periods, power = lsp_find_period(
            times,
            values,
            min_period=float(min_period),
            max_period=float(max_period),
            refine=False,
        )
    except Exception:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    order = np.argsort(periods)
    return np.asarray(periods[order], dtype=float), np.asarray(power[order], dtype=float)


def _render_periodogram_panel(
    ax,
    periods,
    power,
    adopted_period: float | None,
    *,
    min_period: float,
    max_period: float,
    theme: TuiPlotTheme,
    np,
) -> None:
    """Compact Lomb-Scargle strip beneath the phase panel."""

    if periods.size == 0 or power.size == 0:
        _draw_tui_placeholder_panel(ax, "Periodogram unavailable", theme)
        return

    finite = np.isfinite(periods) & np.isfinite(power) & (periods > 0)
    periods = periods[finite]
    power = power[finite]
    if periods.size == 0:
        _draw_tui_placeholder_panel(ax, "Periodogram unavailable", theme)
        return

    search_min = float(min_period)
    search_max = float(max_period)
    if not (math.isfinite(search_min) and math.isfinite(search_max) and search_max > search_min):
        search_min = float(np.nanmin(periods))
        search_max = float(np.nanmax(periods))

    span = search_max / max(search_min, 1.0e-6)
    plot_kwargs = dict(color=theme.point, linewidth=0.95, alpha=0.92, rasterized=True)
    if span >= 30.0:
        ax.semilogx(periods, power, **plot_kwargs)
    else:
        ax.plot(periods, power, **plot_kwargs)

    adopted = _finite_positive(adopted_period)
    if adopted is not None and search_min <= adopted <= search_max:
        ax.axvline(
            adopted,
            color=theme.error,
            linestyle="--",
            linewidth=1.0,
            alpha=0.88,
            zorder=4,
        )

    peak_idx = int(np.nanargmax(power))
    ax.scatter(
        [float(periods[peak_idx])],
        [float(power[peak_idx])],
        s=14,
        facecolors=theme.error,
        edgecolors=theme.error,
        linewidths=0.0,
        zorder=5,
    )
    ax.set_xlim(search_min, search_max)
    power_peak = float(np.nanmax(power))
    ax.set_ylim(0.0, power_peak * 1.06 if power_peak > 0 else 1.0)
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("LSP power")
    _apply_tui_tick_style(ax)


@dataclass(frozen=True)
class _DimmingComplexZoom:
    """One selected recovery-anchored event complex for TUI display."""

    zoom_start_jd: float
    zoom_end_jd: float
    event_start_jd: float
    event_end_jd: float
    peak_jd: float
    status: str
    polarity: str = "dimming"


def _event_window_polarity(payload: dict | None) -> str:
    """Choose dimming or brightening from review labels and event evidence."""
    values = payload if isinstance(payload, dict) else {}

    def normalized(key: str) -> str:
        return (
            str(values.get(key) or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def truthy(key: str) -> bool:
        value = values.get(key)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"", "0", "false", "no", "none", "null", "nan"}:
                return False
            if lowered in {"1", "true", "yes"}:
                return True
        return bool(value)

    def finite(key: str) -> float | None:
        try:
            number = float(values.get(key))
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    morphology = normalized("morphology_primary")
    event_class = normalized("event_class")
    if morphology == "brightening_event" or event_class in {
        "brightening",
        "brightening_event",
        "jumper",
    }:
        return "brightening"
    if morphology == "dimming_event" or event_class in {
        "dimming",
        "dimming_event",
        "dipper",
    }:
        return "dimming"

    dip_significant = truthy("dip_significant")
    jump_significant = truthy("jump_significant")
    if jump_significant != dip_significant:
        return "brightening" if jump_significant else "dimming"

    for suffix in (
        "best_delta_bic",
        "best_log_bf",
        "bayes_factor",
        "max_log_bf_local",
    ):
        dip_score = finite(f"dip_{suffix}")
        jump_score = finite(f"jump_{suffix}")
        if dip_score is None and jump_score is None:
            continue
        if jump_score is None:
            return "dimming"
        if dip_score is None:
            return "brightening"
        if not math.isclose(dip_score, jump_score):
            return "brightening" if jump_score > dip_score else "dimming"

    jump_t0 = finite("jump_best_t0")
    dip_t0 = finite("dip_best_t0")
    if jump_t0 is not None and dip_t0 is None:
        return "brightening"
    return "dimming"


@lru_cache(maxsize=32)
def _cached_atlas_dimming_zoom(
    lc_path: str,
    source_mtime_ns: int,
    polarity: str,
) -> _DimmingComplexZoom:
    """Measure one recovery-anchored complex with the requested polarity."""
    del source_mtime_ns  # It intentionally invalidates this path-keyed cache.
    from malca.stv.dimming_window import (
        dimming_complex_zoom_bounds,
        measure_dimming_complex_window,
    )

    measurement = measure_dimming_complex_window(
        "tui",
        Path(lc_path),
        polarity=polarity,
    )
    window = measurement.window
    zoom_start_jd, zoom_end_jd = dimming_complex_zoom_bounds(
        measurement.epochs["t"].to_numpy(float),
        start_jd=window.start_jd,
        end_jd=window.end_jd,
        peak_jd=window.peak_jd,
        cadence_days=measurement.cadence_days,
    )
    return _DimmingComplexZoom(
        zoom_start_jd=zoom_start_jd,
        zoom_end_jd=zoom_end_jd,
        event_start_jd=float(window.start_jd),
        event_end_jd=float(window.end_jd),
        peak_jd=float(window.peak_jd),
        status=str(window.status),
        polarity=polarity,
    )


def _atlas_dimming_zoom(
    lc_path: Path,
    *,
    polarity: str,
) -> _DimmingComplexZoom | None:
    """Return a recovery-anchored event window for the requested polarity."""
    try:
        resolved = lc_path.resolve()
        source_mtime_ns = resolved.stat().st_mtime_ns
        return _cached_atlas_dimming_zoom(
            str(resolved),
            source_mtime_ns,
            polarity,
        )
    except Exception:
        return None


def _render_event_zoom_panel(
    ax,
    frame,
    dimming_zoom: _DimmingComplexZoom | None,
    *,
    jd_offset: float,
    jd_xlabel: str,
    np,
    theme: TuiPlotTheme,
) -> None:
    """Render one recovery-anchored event window.

    Unlike the raw-panel annotation markers, this panel never uses fitted
    dip/jump ``t0`` values.  Its selected interval and its surrounding margin
    use the same shared complex logic as the half-depth atlas, with residuals
    oriented according to the selected event polarity.
    """
    if dimming_zoom is None:
        _draw_tui_placeholder_panel(ax, "No recovery-anchored event window", theme)
        return

    jd = np.asarray(frame["jd"], dtype=float)
    mag = np.asarray(frame["mag"], dtype=float)
    finite = np.isfinite(jd) & np.isfinite(mag)
    in_window = finite & (jd >= dimming_zoom.zoom_start_jd) & (
        jd <= dimming_zoom.zoom_end_jd
    )
    if not in_window.any():
        _draw_tui_placeholder_panel(ax, "No displayed samples in event window", theme)
        return

    error = None
    if "mag_err" in frame.columns:
        candidate_error = np.asarray(frame["mag_err"], dtype=float)
        if candidate_error.size == jd.size:
            error = candidate_error

    plot_time = jd[in_window] - jd_offset
    selected_mag = mag[in_window]
    event_start = dimming_zoom.event_start_jd - jd_offset
    event_end = dimming_zoom.event_end_jd - jd_offset
    ax.axvspan(event_start, event_end, color=theme.error, alpha=0.08, linewidth=0)
    if error is not None:
        selected_error = error[in_window]
        valid_error = np.isfinite(selected_error) & (selected_error > 0)
        if valid_error.any():
            ax.errorbar(
                plot_time[valid_error],
                selected_mag[valid_error],
                yerr=selected_error[valid_error],
                fmt="none",
                ecolor=theme.error,
                elinewidth=0.5,
                capsize=1.1,
                alpha=0.52,
                zorder=1,
            )
    ax.scatter(
        plot_time,
        selected_mag,
        s=10,
        color=theme.point,
        alpha=0.9,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )
    ax.axvline(
        dimming_zoom.peak_jd - jd_offset,
        color=theme.annotation_text,
        linestyle="--",
        linewidth=0.8,
        alpha=0.82,
    )
    ax.set_xlim(
        dimming_zoom.zoom_start_jd - jd_offset,
        dimming_zoom.zoom_end_jd - jd_offset,
    )
    ax.invert_yaxis()
    ax.set_xlabel(jd_xlabel, labelpad=3)
    ax.set_ylabel(TUI_PUB_LC_YLABEL)


def _load_tui_sed_context(request: RenderRequest):
    """Load the same SED inputs used by browser review."""

    import pandas as pd

    from malca.enrichment.sed_model import (
        load_sed_model_curves,
        load_sed_model_fits,
        load_sed_model_points,
    )
    from malca.review.sed import load_sed_rows

    payload = dict(request.payload) if isinstance(request.payload, dict) else {}
    external = pd.DataFrame()
    curves = pd.DataFrame()
    fits = pd.DataFrame()
    points = pd.DataFrame()
    db_path = _existing_file(request.db_path)
    if db_path is not None and request.candidate_id:
        try:
            import sqlite3

            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                external = load_sed_rows(conn, str(request.candidate_id))
                curves = load_sed_model_curves(conn, str(request.candidate_id))
                fits = load_sed_model_fits(conn, str(request.candidate_id))
                points = load_sed_model_points(conn, str(request.candidate_id))
        except Exception:
            pass
    return payload, external, curves, fits, points


def _render_sed_panel(ax, request: RenderRequest, np, *, theme: TuiPlotTheme) -> None:
    """Compact matplotlib SED using the browser-review build path."""

    try:
        from malca.review.sed import render_sed_matplotlib

        payload, external, curves, fits, points = _load_tui_sed_context(request)
        fit_version = (
            str(fits.iloc[0].get("fit_version") or "")
            if fits is not None and not fits.empty
            else ""
        )
        fit_status = (
            str(fits.iloc[0].get("status") or "").strip().lower()
            if fits is not None and not fits.empty
            else ""
        )
        extinction_mode = (
            "corrected"
            if fit_status == "ok" and not fit_version.startswith("ck04-bandpass-v")
            else "observed"
        )
        render_sed_matplotlib(
            ax,
            payload,
            candidate_id=request.candidate_id,
            external_rows=external,
            model_curve_rows=curves,
            model_fit_rows=fits,
            model_point_rows=points,
            extinction_mode=extinction_mode,
            theme=theme.name,
            y_axis_side="left",
        )
        _style_tui_sed_legend(ax, theme)
    except Exception as exc:
        _draw_tui_placeholder_panel(ax, f"SED error: {exc}", theme)


def render_lightcurve_png(request: RenderRequest, output_path: PathLike) -> Path:
    """Render a raw light curve and browser-equivalent best phase fold.

    The destination is replaced atomically, so the image viewer never observes
    a partially written PNG.
    """
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise ValueError("TUI light-curve output must use a .png extension")

    lc_path = resolve_lightcurve_path(request)
    if lc_path is None:
        raise FileNotFoundError(
            f"No local light curve found for candidate {request.candidate_id!r}"
        )

    plt, np, load_lightcurve_df = _load_render_dependencies()
    frame = _load_display_frame(request, lc_path, load_lightcurve_df)
    if frame is None or frame.empty:
        raise ValueError(f"No {_camera_view(request)}-camera observations in {lc_path}")

    jd = np.asarray(frame["jd"], dtype=float)
    magnitude = np.asarray(frame["mag"], dtype=float)
    finite = np.isfinite(jd) & np.isfinite(magnitude)
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise ValueError(f"No finite magnitude observations in {lc_path}")
    asassn_cadence_window_days = _asassn_cadence_window_days(jd, np)
    asassn_jd_window = (
        _asassn_jd_window(
            jd,
            np,
            padding_days=_asassn_window_padding_days(request),
        )
        if _time_window_mode(request) == "asassn"
        else None
    )

    if "camera_name" in frame.columns:
        camera = frame["camera_name"].fillna("").astype(str).str.strip()
    else:
        camera = frame["camera"].fillna("").astype(str).str.strip()
    if "camera" in frame.columns:
        fallback = frame["camera"].fillna("").astype(str).str.strip()
        camera = camera.mask(camera.eq(""), fallback)
    frame["_tui_camera"] = camera.mask(camera.eq(""), "unknown")
    frame["_tui_band"] = frame["band"].fillna("").astype(str).str.strip()

    # Survey IO and period resolution are independent.  Overlap them so a
    # blank DECaPS tile plus DSS2 fallback does not add its full network latency
    # on top of PDM/harmonic computation.
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="malca-tui-cutout",
    ) as cutout_pool:
        cutout_future = cutout_pool.submit(_survey_cutout, request, plt, np)
        phase_period, phase_source, phase_warning = _resolve_best_phase_period(
            request,
            lc_path,
        )
        phase_source_frame = None
        phase_epoch = None
        if phase_period is not None:
            try:
                phase_source_frame, phase_epoch = _prepare_review_phase_source(
                    request, lc_path
                )
            except Exception as exc:
                phase_warning = f"Phase preparation failed: {exc}"
            has_residuals = bool(
                phase_source_frame is not None
                and not phase_source_frame.empty
                and "resid" in phase_source_frame.columns
                and np.isfinite(
                    np.asarray(phase_source_frame["resid"], dtype=float)
                ).any()
            )
            if not has_residuals:
                from malca.plotting.lightcurve_publication import (
                    prepare_median_centered_phase_source,
                )

                phase_source_frame = prepare_median_centered_phase_source(
                    frame,
                    time_col="jd",
                    value_col="mag",
                    error_col="mag_err",
                    band_col="_tui_band",
                    camera_col="_tui_camera",
                )
                phase_epoch = float(frame["jd"].min())
                fallback_note = "using median-centered fallback"
                phase_warning = (
                    f"{phase_warning}; {fallback_note}"
                    if phase_warning
                    else fallback_note
                )
        try:
            cutout = cutout_future.result()
        except Exception as exc:
            cutout = _CutoutPanel(None, "", f"Cutout error: {exc}", 0.0)

    cmd_background = _load_cmd_background(request)
    plot_theme = tui_plot_theme(request.plot_theme)
    event_polarity = _event_window_polarity(request.payload)
    dimming_zoom = _atlas_dimming_zoom(
        lc_path,
        polarity=event_polarity,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".png", dir=str(destination.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)

    figure = None
    try:
        from malca.plotting.lightcurve_publication import (
            plot_lightcurve_panel,
            plot_phase_panel,
            plot_phase_placeholder,
        )
        from malca.review.coordinate_labels import (
            format_j_designation,
            payload_ra_dec,
        )
        from malca.review.lightcurve_pdf import _draw_header_boxes
        from malca.review.tui_controller import build_review_display_title

        with plt.rc_context(_review_style_for_theme(plot_theme)):
            figure = plt.figure(figsize=PNG_FIGSIZE, dpi=PNG_DPI)
            figure.patch.set_facecolor(plot_theme.figure)
            grid = figure.add_gridspec(
                2,
                2,
                height_ratios=(1.35, 1.45),
                width_ratios=(1.55, 1.25),
                hspace=0.14,
                wspace=0.22,
            )
            raw_axis = figure.add_subplot(grid[0, 0])
            left_stack = grid[1, 0].subgridspec(
                2,
                1,
                height_ratios=(0.82, 1.0),
                hspace=0.32,
            )
            event_zoom_axis = figure.add_subplot(left_stack[0, 0])
            phase_axis = figure.add_subplot(left_stack[1, 0])
            right_stack = grid[:, 1].subgridspec(
                3,
                2,
                # The full-width LSP is one dip-zoom panel tall after
                # accounting for the parent and nested GridSpec spacing.
                height_ratios=(1.0, 1.0, 0.61),
                wspace=0.30,
                hspace=0.30,
            )
            cutout_axis = figure.add_subplot(right_stack[0, 0])
            cmd_axis = figure.add_subplot(right_stack[0, 1])
            sed_axis = figure.add_subplot(right_stack[1, 0])
            periodogram_axis = figure.add_subplot(right_stack[2, :])

            payload = request.payload if isinstance(request.payload, dict) else {}
            display_title = build_review_display_title(
                payload,
                asas_sn_id=request.asas_sn_id,
            )
            asas_median = float(np.nanmedian(magnitude))
            jd_offset = _jd_plot_offset(jd, np)
            jd_xlabel = rf"JD $-$ {int(jd_offset)} [d]" if jd_offset else "JD [d]"

            # Compact, review-oriented header labels.
            coords = payload_ra_dec(payload) if payload else None
            if coords is not None:
                ra_deg, dec_deg = coords
                header_left = format_j_designation(ra_deg, dec_deg)
                header_right = _format_tui_coordinate_header(ra_deg, dec_deg)
            else:
                header_left = str(request.candidate_id or "")
                header_right = None

            events: list = []
            if payload:
                try:
                    from malca.review.native_lightcurve import _event_entries

                    events = _event_entries(
                        payload,
                        0.0,
                        _load_run_params(request, lc_path),
                        lc_median=float(np.nanmedian(jd)),
                    )
                except Exception:
                    events = []

            # ``plot_lightcurve_panel`` renders both the color-by-camera and the
            # single-color publication styles.  Setting ``group_by="none"`` in the
            # publication branch collapses every point into one uniformly styled
            # series that matches the paper figures.
            error_col = "mag_err" if "mag_err" in frame.columns else None
            common_lc_kwargs = dict(
                time_col="jd",
                value_col="mag",
                error_col=error_col,
                band_col="_tui_band",
                camera_col="_tui_camera",
                time_offset=float(jd_offset),
                xlabel=jd_xlabel,
                ylabel=TUI_PUB_LC_YLABEL,
                invert_y=True,
                rasterized=True,
                title=None,
                annotated_events=events if request.show_event_markers else (),
                margins=(0.0, 0.02),
            )

            if request.color_by_camera:
                plot_lightcurve_panel(
                    raw_axis,
                    frame,
                    group_by="band-camera",
                    color_by="camera",
                    marker_by="band",
                    show_errorbars=False,
                    marker_size=12**0.5,
                    marker_alpha=0.82,
                    marker_edgecolor=None,
                    marker_edgewidth=0.0,
                    legend_max_groups=14,
                    dense_group_note="color = camera   ○ g   △ V",
                    **common_lc_kwargs,
                )
            else:
                plot_lightcurve_panel(
                    raw_axis,
                    frame,
                    group_by="none",
                    color_map={"all": plot_theme.point},
                    show_errorbars=True,
                    error_color=plot_theme.error,
                    marker_size=2.6,
                    marker_alpha=0.9,
                    marker_edgecolor=None,
                    marker_edgewidth=0.0,
                    legend="none",
                    **common_lc_kwargs,
                )

            external_legend: list[tuple[str, str, str]] = []
            if request.show_external_lightcurves:
                try:
                    external_legend = _overlay_tui_external_lightcurves(
                        raw_axis,
                        request,
                        lc_path,
                        asas_median=asas_median,
                        jd_offset=jd_offset,
                        np=np,
                        jd_window=asassn_jd_window,
                        cadence_window_days=asassn_cadence_window_days,
                    )
                except Exception:
                    external_legend = []
            if asassn_jd_window is not None:
                raw_axis.set_xlim(
                    asassn_jd_window[0] - float(jd_offset),
                    asassn_jd_window[1] - float(jd_offset),
                )

            # Build a compact legend that spells out which survey/band each
            # marker/color denotes.  We do this via ``Line2D`` proxies rather
            # than the automatic legend so ASAS-SN (drawn without a matplotlib
            # label) is always the first entry.
            try:
                from matplotlib.lines import Line2D

                if request.color_by_camera:
                    asas_handle = Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        markerfacecolor="0.35",
                        markeredgecolor="0.35",
                        markersize=4.0,
                        label="ASAS-SN (camera colored)",
                    )
                else:
                    asas_handle = Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=plot_theme.point,
                        markeredgecolor=plot_theme.point,
                        markersize=4.0,
                        label="ASAS-SN",
                    )
                handles = [asas_handle]
                for label, color, marker_char in external_legend:
                    face = "none" if marker_char in {"x", "+"} else color
                    handles.append(
                        Line2D(
                            [],
                            [],
                            marker=marker_char,
                            linestyle="none",
                            markerfacecolor=face,
                            markeredgecolor=color,
                            markersize=4.4,
                            markeredgewidth=0.9,
                            label=label,
                        )
                    )
                raw_axis.legend(
                    handles=handles,
                    loc="lower right",
                    fontsize=TUI_LEGEND_FONTSIZE,
                    frameon=True,
                    framealpha=0.9,
                    ncol=min(4, max(1, len(handles))),
                    handlelength=TUI_LEGEND_HANDLE_LENGTH,
                    handletextpad=TUI_LEGEND_HANDLE_TEXTPAD,
                    columnspacing=TUI_LEGEND_COLUMN_SPACING,
                    borderpad=TUI_LEGEND_BORDERPAD,
                )
            except Exception:
                pass

            try:
                _draw_header_boxes(
                    raw_axis,
                    left=header_left,
                    right=header_right,
                    font_size=13.5,
                    text_color=plot_theme.header_text,
                    bbox={
                        "boxstyle": "square,pad=0.52",
                        "facecolor": plot_theme.header_face,
                        "edgecolor": plot_theme.header_edge,
                        "linewidth": 0.95,
                        "alpha": 1.0,
                    },
                )
            except Exception:
                pass

            # Larger, more visible ticks — the paper style's defaults are too
            # small for the TUI raster.
            _apply_tui_tick_style(raw_axis)

            _render_event_zoom_panel(
                event_zoom_axis,
                frame,
                dimming_zoom,
                jd_offset=jd_offset,
                jd_xlabel=jd_xlabel,
                np=np,
                theme=plot_theme,
            )

            phase_rendered = False
            if (
                phase_period is not None
                and phase_source_frame is not None
                and not phase_source_frame.empty
            ):
                phase_camera_col = next(
                    (
                        column
                        for column in (
                            "camera_name",
                            "camera_label",
                            "camera#",
                            "camera",
                        )
                        if column in phase_source_frame.columns
                    ),
                    None,
                )
                phase_error_col = (
                    "error" if "error" in phase_source_frame.columns else None
                )
                phase_common = dict(
                    period_days=phase_period,
                    epoch_jd=phase_epoch,
                    value_mode="resid",
                    duplicate_cycles=True,
                    time_col="JD",
                    value_col="mag",
                    error_col=phase_error_col,
                    band_col="v_g_band",
                    camera_col=phase_camera_col,
                    residual_col="resid",
                    title=None,
                    rasterized=True,
                    xlim=(0.0, 2.0),
                    xlabel=TUI_PUB_PHASE_XLABEL,
                    ylabel=TUI_PUB_RESID_YLABEL,
                    margins=(0.0, 0.02),
                    notice=None,
                )
                try:
                    if request.color_by_camera:
                        plot_phase_panel(
                            phase_axis,
                            phase_source_frame,
                            group_by="band-camera",
                            color_by="camera",
                            marker_by="band",
                            show_errorbars=False,
                            legend="none",
                            marker_size=10**0.5,
                            marker_alpha=0.78,
                            marker_edgecolor=None,
                            marker_edgewidth=0.0,
                            **phase_common,
                        )
                    else:
                        plot_phase_panel(
                            phase_axis,
                            phase_source_frame,
                            group_by="none",
                            color_map={"all": plot_theme.point},
                            show_errorbars=True,
                            error_color=plot_theme.error,
                            legend="none",
                            marker_size=2.4,
                            marker_alpha=0.9,
                            marker_edgecolor=None,
                            marker_edgewidth=0.0,
                            **phase_common,
                        )
                    phase_rendered = True
                except Exception as exc:
                    phase_warning = (
                        f"{phase_warning}; phase plot failed: {exc}"
                        if phase_warning
                        else f"Phase plot failed: {exc}"
                    )

            if phase_rendered:
                _apply_tui_tick_style(phase_axis)
                phase_axis.set_xlabel(TUI_PUB_PHASE_XLABEL, labelpad=6)
                from malca.review.tui_controller import _format_period_days

                period_label = (
                    rf"$P = {_format_period_days(phase_period)}$ d"
                    if phase_period is not None and math.isfinite(phase_period)
                    else "no valid period"
                )
                phase_axis.set_title(
                    period_label,
                    fontsize=10,
                    color=plot_theme.annotation_text,
                    pad=4,
                )
            else:
                phase_axis.clear()
                plot_phase_placeholder(
                    phase_axis,
                    phase_warning or "No valid best period",
                    xlabel=TUI_PUB_PHASE_XLABEL,
                    ylabel=TUI_PUB_RESID_YLABEL,
                )
                phase_axis.set_xlabel(TUI_PUB_PHASE_XLABEL, labelpad=6)
                for line in phase_axis.get_lines():
                    line.set_color(plot_theme.phase_guide)
                    line.set_alpha(min(1.0, max(0.0, plot_theme.grid_alpha + 0.35)))
                for text in phase_axis.texts:
                    position = text.get_position()
                    if len(position) >= 2 and abs(float(position[0]) - 0.5) < 0.01:
                        text.set_color(plot_theme.phase_message)

            try:
                _render_cmd_panel(
                    cmd_axis,
                    payload,
                    np,
                    background=cmd_background,
                    theme=plot_theme,
                )
                cmd_axis.set_box_aspect(1)
            except Exception as exc:
                cmd_axis.clear()
                _draw_tui_placeholder_panel(cmd_axis, f"CMD error: {exc}", plot_theme)
                cmd_axis.set_box_aspect(1)

            _render_sed_panel(sed_axis, request, np, theme=plot_theme)
            sed_axis.set_box_aspect(1)

            min_period, max_period = _resolve_phase_search_bounds(
                request,
                lc_path,
                payload=payload if isinstance(payload, dict) else None,
            )
            try:
                periods, power = _compute_periodogram_curve(
                    phase_source_frame,
                    min_period=min_period,
                    max_period=max_period,
                    np=np,
                )
                _render_periodogram_panel(
                    periodogram_axis,
                    periods,
                    power,
                    phase_period,
                    min_period=min_period,
                    max_period=max_period,
                    theme=plot_theme,
                    np=np,
                )
            except Exception as exc:
                periodogram_axis.clear()
                _draw_tui_placeholder_panel(
                    periodogram_axis,
                    f"Periodogram error: {exc}",
                    plot_theme,
                )

            _draw_cutout_axis(cutout_axis, cutout, plt, theme=plot_theme)
            cutout_axis.set_box_aspect(1)
            cmd_axis.set_box_aspect(1)
            sed_axis.set_box_aspect(1)

            panel_axes = (
                raw_axis,
                event_zoom_axis,
                phase_axis,
                periodogram_axis,
                cmd_axis,
                sed_axis,
            )
            for axis in panel_axes:
                _apply_tui_axis_theme(axis, plot_theme)
                _apply_tui_tick_style(axis)
            event_zoom_axis.grid(False, which="both")
            if cutout.image is None:
                _apply_tui_axis_theme(cutout_axis, plot_theme)
            _style_tui_legend(raw_axis, plot_theme)

            figure.subplots_adjust(
                left=0.08,
                right=0.965,
                bottom=0.08,
                top=0.96,
            )
            figure.savefig(temporary, format="png", dpi=PNG_DPI)
        os.replace(temporary, destination)
        _write_render_metadata(
            destination,
            {
                "phase_period_days": phase_period,
                "phase_source": phase_source or None,
                "display_title": display_title,
                "camera_view": _camera_view(request),
                "time_window_mode": _time_window_mode(request),
                "asassn_window_padding_days": _asassn_window_padding_days(
                    request
                ),
                "external_cadence_bin_days": asassn_cadence_window_days,
            },
        )
    finally:
        if figure is not None:
            plt.close(figure)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    return destination


def quicklook_command(image_path: PathLike) -> list[str]:
    """Return the argv for one owned macOS Quick Look preview process."""
    return [
        "/usr/bin/qlmanage",
        "-p",
        str(Path(image_path).expanduser()),
    ]


def persistent_viewer_command(manifest_path: PathLike) -> list[str]:
    """Return argv for the geometry-preserving MALCA image-window helper."""

    return [
        sys.executable,
        "-m",
        "malca.review.tui_image_viewer",
        "--manifest",
        str(Path(manifest_path).expanduser()),
    ]


def launch_viewer(image_path: PathLike, viewer: str = "window"):
    """Launch an owned native viewer process and return it to the caller."""
    normalized = str(viewer).strip().lower()
    if normalized not in VIEWER_CHOICES:
        raise ValueError(f"viewer must be one of: {', '.join(VIEWER_CHOICES)}")
    if normalized == "none":
        return None
    command = (
        persistent_viewer_command(image_path)
        if normalized == "window"
        else quicklook_command(image_path)
    )
    return subprocess.Popen(
        command,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class ImageCoordinator:
    """Coordinate one current render and lightweight one-candidate prefetching."""

    def __init__(
        self,
        cache_dir: PathLike,
        *,
        viewer: str = "window",
        cache_size: int = 4,
        renderer: Callable[[RenderRequest, PathLike], Path] = render_lightcurve_png,
    ) -> None:
        normalized_viewer = str(viewer).strip().lower()
        if normalized_viewer not in VIEWER_CHOICES:
            raise ValueError(f"viewer must be one of: {', '.join(VIEWER_CHOICES)}")
        if int(cache_size) < 1:
            raise ValueError("cache_size must be at least 1")

        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.cache_dir / "current.png"
        self.viewer_manifest_path = self.cache_dir / "viewer.json"
        self.viewer = normalized_viewer
        self.cache_size = int(cache_size)
        self._renderer = renderer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="malca-tui-image")
        self._jobs: dict[str, Future] = {}
        self._cache: OrderedDict[str, Path] = OrderedDict()
        self._current_key: Optional[str] = None
        self._generation = 0
        self._ready_generation = -1
        self._closed = False
        self._viewer_process: subprocess.Popen | None = None
        self._status = ImageStatus(path=self.current_path)
        self._prune_disk_cache()

    @property
    def status(self) -> ImageStatus:
        return self._status

    @staticmethod
    def _request_key(request: RenderRequest) -> str:
        resolved = resolve_lightcurve_path(request)
        if resolved is not None:
            try:
                stat = resolved.stat()
                source_token = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
            except OSError:
                source_token = str(resolved)
        else:
            source_token = "|".join(
                str(value or "")
                for value in (
                    request.lc_path,
                    request.run_dir,
                    request.db_path,
                    request.source_path,
                )
            )
        params_path = _run_params_path(request, resolved)
        if params_path is not None:
            try:
                params_stat = params_path.stat()
                params_token = (
                    f"{params_path}|{params_stat.st_size}|{params_stat.st_mtime_ns}"
                )
            except OSError:
                params_token = str(params_path)
        else:
            params_token = ""
        stored_period = _finite_positive(request.stored_phase_period_days)
        manual_period = _finite_positive(request.manual_phase_period_days)
        try:
            from malca.review.cutouts import candidate_coordinates

            coordinates = candidate_coordinates(
                request.payload if isinstance(request.payload, dict) else None
            )
        except Exception:
            coordinates = None
        event_token = ""
        if request.show_event_markers and isinstance(request.payload, dict):
            event_token = json.dumps(
                {
                    key: request.payload.get(key)
                    for key in (
                        "dip_best_t0",
                        "jump_best_t0",
                        "dip_best_width_param",
                        "jump_best_width_param",
                        "dip_bayes_factor",
                        "jump_bayes_factor",
                        "dip_best_morph",
                        "jump_best_morph",
                    )
                },
                sort_keys=True,
                default=str,
            )
        identity = "|".join(
            (
                RENDER_CACHE_VERSION,
                str(request.candidate_id),
                str(request.asas_sn_id or ""),
                source_token,
                params_token,
                f"stored_period={stored_period!r}",
                f"stored_source={request.stored_phase_source or ''}",
                f"manual_period={manual_period!r}",
                f"manual_source={request.manual_phase_source or ''}",
                f"phase_multiplier={_phase_multiplier(request.phase_multiplier)!r}",
                f"force_search={bool(request.force_period_search)}",
                f"force_search_token={request.force_period_search_token or ''}",
                f"camera_view={_camera_view(request)}",
                f"color_by_camera={bool(request.color_by_camera)}",
                f"event_markers={bool(request.show_event_markers)}:{event_token}",
                f"event_polarity={_event_window_polarity(request.payload)}",
                f"external_lcs={bool(request.show_external_lightcurves)}",
                "external_sources="
                + ",".join(
                    normalize_tui_external_photometry_sources(
                        request.external_lightcurve_sources
                    )
                ),
                f"time_window={_time_window_mode(request)}:"
                f"{_asassn_window_padding_days(request)!r}",
                f"plot_theme={tui_plot_theme(request.plot_theme).name}",
                f"survey={request.survey_key or DEFAULT_TUI_SURVEY_KEY}",
                f"coordinates={coordinates!r}",
                f"search={PHASE_SEARCH_METHOD}:"
                f"{request.phase_search_min_days!r}:"
                f"{request.phase_search_max_days!r}",
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("ImageCoordinator is closed")

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"render-{key[:20]}.png"

    def _prune_disk_cache(self, protected: tuple[Path, ...] = ()) -> None:
        """Bound cache artifacts left by earlier TUI sessions."""
        try:
            cached_paths = sorted(
                self.cache_dir.glob("render-*.png"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return
        protected_existing = [path for path in protected if path in cached_paths]
        cached_paths = protected_existing + [
            path for path in cached_paths if path not in protected_existing
        ]
        for old_path in cached_paths[self.cache_size :]:
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                continue
            try:
                _metadata_path(old_path).unlink()
            except (FileNotFoundError, OSError):
                pass
            for key, tracked_path in list(self._cache.items()):
                if tracked_path == old_path:
                    del self._cache[key]

    def _cached_path(self, key: str) -> Optional[Path]:
        cached = self._cache.get(key)
        if cached is None:
            disk_path = self._cache_path(key)
            if disk_path.is_file():
                cached = disk_path
                self._remember(key, cached)
        if cached is None or not cached.is_file():
            return None
        self._cache.move_to_end(key)
        try:
            cached.touch()
        except OSError:
            pass
        return cached

    def _submit(self, request: RenderRequest, key: str) -> None:
        if key in self._jobs:
            return
        destination = self._cache_path(key)
        self._jobs[key] = self._executor.submit(self._renderer, request, destination)

    def _cancel_superseded_jobs(self, keep_key: str) -> None:
        """Drop queued renders that rapid navigation made irrelevant.

        A render already running cannot be interrupted safely, but the executor
        has only one worker.  Canceling everything else bounds the wait for the
        newest current image to at most that one in-flight render.
        """
        for key, future in list(self._jobs.items()):
            if key == keep_key:
                continue
            if future.cancel():
                del self._jobs[key]

    def _remember(self, key: str, path: Path) -> None:
        try:
            path.touch()
        except OSError:
            pass
        self._cache[key] = path
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            old_key, old_path = self._cache.popitem(last=False)
            if old_key == self._current_key:
                self._cache[old_key] = old_path
                self._cache.move_to_end(old_key)
                continue
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
            try:
                _metadata_path(old_path).unlink()
            except (FileNotFoundError, OSError):
                pass
        protected = [path]
        current_cached = self._cache.get(self._current_key or "")
        if current_cached is not None and current_cached not in protected:
            protected.append(current_cached)
        self._prune_disk_cache(tuple(protected))

    def _clear_error_placeholders(self, keep: Optional[Path] = None) -> None:
        for placeholder in self.cache_dir.glob("error-*.png"):
            if keep is not None and placeholder == keep:
                continue
            try:
                placeholder.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                continue

    def _close_viewer(self) -> None:
        """Close only the image process launched by this coordinator."""
        process = self._viewer_process
        self._viewer_process = None
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=QUICKLOOK_CLOSE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=QUICKLOOK_CLOSE_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    # The process handle is still owned, but there is nothing
                    # safe to do here beyond kill: never target Preview or a
                    # system-wide Quick Look service by name.
                    pass
        except (OSError, ProcessLookupError):
            # The owned process exited between poll and terminate/wait.
            return

    def _show_viewer(self, image_path: PathLike, *, title: str | None = None) -> None:
        """Show the image without resetting persistent-window geometry."""

        if self.viewer == "none":
            return
        if self.viewer == "window":
            self._write_viewer_manifest(Path(image_path), title=title)
            process = self._viewer_process
            if process is not None and process.poll() is None:
                return
            self._viewer_process = launch_viewer(
                self.viewer_manifest_path, self.viewer
            )
            return
        # Legacy Quick Look cannot reliably replace content in an existing
        # owned preview, so explicit --viewer quicklook retains restart behavior.
        self._close_viewer()
        self._viewer_process = launch_viewer(image_path, self.viewer)

    def _write_viewer_manifest(
        self,
        image_path: Path,
        *,
        title: str | None = None,
    ) -> None:
        temporary = self.viewer_manifest_path.with_suffix(".json.tmp")
        try:
            image_version = image_path.stat().st_mtime_ns
        except OSError:
            image_version = 0
        manifest_title = str(title or "").strip() or "MALCA Review"
        payload = {
            "path": str(image_path.expanduser().resolve()),
            "token": f"{self._generation}:{image_version}",
            "title": manifest_title,
        }
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.viewer_manifest_path)

    def _install_error_placeholder(self, key: str) -> Optional[str]:
        placeholder = self.cache_dir / f"error-{key[:20]}.png"
        try:
            self._clear_error_placeholders(keep=placeholder)
            _write_error_placeholder_png(placeholder)
            _atomic_replace_from_file(placeholder, self.current_path)
            self._show_viewer(placeholder)
        except Exception as exc:
            return f"error placeholder failed: {exc}"
        return None

    def _install_current(self, cached_path: Path) -> None:
        generation = self._generation
        metadata = _read_render_metadata(cached_path)
        display_title = str(metadata.get("display_title") or "").strip() or None
        try:
            _atomic_replace_from_file(cached_path, self.current_path)
            # The viewer receives the immutable cache path.  The persistent
            # helper reloads only its canvas and retains user window geometry.
            self._show_viewer(cached_path, title=display_title)
            self._clear_error_placeholders()
        except Exception as exc:
            self._status = ImageStatus(
                state="error",
                candidate_id=self._status.candidate_id,
                path=self.current_path,
                error=f"Image display error: {exc}",
                generation=generation,
            )
            self._ready_generation = generation
            return
        self._ready_generation = generation
        self._status = ImageStatus(
            state="ready",
            candidate_id=self._status.candidate_id,
            path=self.current_path,
            generation=generation,
            phase_period_days=_finite_positive(metadata.get("phase_period_days")),
            phase_source=str(metadata.get("phase_source") or "") or None,
            survey_label=None,
        )

    def request_current(self, request: RenderRequest) -> ImageStatus:
        """Request the visible candidate image and return immediately."""
        self._assert_open()
        self._generation += 1
        self._ready_generation = -1
        key = self._request_key(request)
        self._current_key = key
        self._cancel_superseded_jobs(key)
        self._status = ImageStatus(
            state="rendering",
            candidate_id=request.candidate_id,
            path=self.current_path,
            generation=self._generation,
        )

        cached = self._cached_path(key)
        if cached is not None:
            self._install_current(cached)
        else:
            self._submit(request, key)
        return self._status

    def prefetch(self, request: RenderRequest) -> bool:
        """Queue an image for cache without changing the visible candidate."""
        self._assert_open()
        key = self._request_key(request)
        cached = self._cached_path(key)
        if cached is not None:
            return False
        if key in self._jobs:
            return False
        self._submit(request, key)
        return True

    def poll(self) -> ImageStatus:
        """Harvest completed work without blocking and return current status."""
        self._assert_open()
        for key, future in list(self._jobs.items()):
            if not future.done():
                continue
            del self._jobs[key]
            try:
                rendered_path = Path(future.result())
                if not rendered_path.is_file():
                    raise FileNotFoundError(f"Renderer did not create {rendered_path}")
                self._remember(key, rendered_path)
            except Exception as exc:
                if key == self._current_key:
                    display_error = self._install_error_placeholder(key)
                    error_text = f"{type(exc).__name__}: {exc}"
                    if display_error:
                        error_text += f"; {display_error}"
                    self._status = ImageStatus(
                        state="error",
                        candidate_id=self._status.candidate_id,
                        path=self.current_path,
                        error=error_text,
                        generation=self._generation,
                    )
                    self._ready_generation = self._generation
                continue

            # A completed old request may warm the cache, but it must never
            # replace the image chosen by a newer generation.
            if key == self._current_key and self._ready_generation != self._generation:
                self._install_current(rendered_path)
        return self._status

    def close(self) -> None:
        """Stop the render worker and close this session's owned image window."""
        if self._closed:
            return
        self._closed = True
        self._close_viewer()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._status = ImageStatus(
            state="closed",
            candidate_id=self._status.candidate_id,
            path=self.current_path,
            generation=self._generation,
        )

    def __enter__(self) -> "ImageCoordinator":
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
