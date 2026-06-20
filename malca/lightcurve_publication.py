from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from malca.lightcurve_io import load_lightcurve_df, normalize_asassn_lightcurve


JD_OFFSET = 2450000.0
DIP_EVENT_COLOR = "#ff6b6b"
JUMP_EVENT_COLOR = "#0096FF"

BAND_COLORS = {
    "g": "#238b45",
    "V": "#6a51a3",
    "B": "#2171b5",
    "u": "#08519c",
    "r": "#cb181d",
    "i": "#d94801",
    "z": "#8c6d31",
    "all": "#252525",
    "unknown": "#737373",
}

MARKERS = ("o", "s", "D", "^", "v", "P", "X", "<", ">", "h")

# LaTeX Computer Modern–style serif (STIXGeneral + cm mathtext in matplotlib).
PUBLICATION_SERIF_FONTS = ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"]
PUBLICATION_PLOTLY_FONT = ", ".join(PUBLICATION_SERIF_FONTS)

PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": PUBLICATION_SERIF_FONTS,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Publication figure widths (inches). ApJ-style single column = 3.5; two-column span = 7.0.
FIG_SINGLE_COL_WIDTH = 3.5
FIG_TWO_COL_WIDTH = 7.0

FIG_SINGLE_COL_SQUARE = (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH)
FIG_LC_SINGLE_COL = (FIG_SINGLE_COL_WIDTH, 2.0)
FIG_LC_TWO_COL = (FIG_TWO_COL_WIDTH, 3.0)
FIG_ROC_PR_TWO_COL = (FIG_TWO_COL_WIDTH, 3.0)

# Gaia CMD plot styling for FIG_SINGLE_COL_SQUARE (3.5 x 3.5 in).
CMD_BG_SCATTER_SIZE = 2.0
CMD_BG_SCATTER_ALPHA = 0.33
CMD_BG_HOLLOW_SCATTER_SIZE = 3.5
CMD_AXIS_LABEL_FONTSIZE = 8.0
CMD_TICK_LABEL_FONTSIZE = 7.0
CMD_LEGEND_FONTSIZE = 6.5
CMD_TICK_LENGTH = 2.2
CMD_TICK_WIDTH = 0.5
CMD_MARKER_EDGE_SOLID = 0.3
CMD_MARKER_EDGE_HOLLOW = 0.45
CMD_LEGEND_MARKERSCALE = 0.7
CMD_BUCKET_STYLE: dict[str, dict[str, object]] = {
    "Dipper": {"color": "#E41A1C", "marker": "v", "size": 16, "zorder": 6},
    "Interesting": {"color": "#009E73", "marker": "o", "size": 10, "zorder": 4},
    "LTV": {"color": "#ED9224", "marker": "v", "size": 16, "zorder": 7},
    "Microlensing": {"color": "#F0E442", "marker": "*", "size": 28, "zorder": 9},
    "Eclipsing binary": {"color": "#CC79A7", "marker": "s", "size": 18, "zorder": 8},
    "Unknown": {"color": "#6A3D9A", "marker": "D", "size": 12, "zorder": 5},
}

# Legacy-sized figures scaled to publication column widths (3.5" single / 7.0" two-column).
FIG_SINGLE_COL_HEATMAP = (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH)  # was (10, 10)
FIG_SINGLE_COL_LC_WIDE = (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 6 / 10)  # was (10, 6)
FIG_SINGLE_COL_MEDIUM = (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 6 / 8)  # was (8, 6)
FIG_SINGLE_COL_COMPACT = (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 5 / 6)  # was (6, 5)
FIG_SINGLE_COL_PORTRAIT = (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 5 / 7)  # was (7, 5)
FIG_TWO_COL_STANDARD = (FIG_TWO_COL_WIDTH, FIG_TWO_COL_WIDTH * 8 / 14)  # was (14, 8)
FIG_TWO_COL_LC_WIDE = (FIG_TWO_COL_WIDTH, FIG_TWO_COL_WIDTH * 6 / 13)  # was (13, 6)
FIG_TWO_COL_TRIPLE = (FIG_TWO_COL_WIDTH, FIG_TWO_COL_WIDTH * 6 / 18)  # was (18, 6)

FIG_HEATMAP_REFERENCE = (10.0, 8.0)
FIG_GRID_PANEL_WIDTH = 3.5  # matches malca.config.GRID_PANEL_WIDTH
FIG_GRID_ROW_HEIGHT = 2.5
FIG_HEATMAP_ROW_HEIGHT = 0.28
FIG_HEATMAP_MIN_HEIGHT = 4.0

COLUMN_ALIASES = {
    "time": (
        "jd",
        "hjd",
        "bjd",
        "mjd",
        "time",
        "date",
        "julian date",
        "julian_date",
    ),
    "mag": (
        "mag",
        "magnitude",
        "m",
        "mag auto",
        "mag_auto",
        "psfmag",
        "apmag",
        "calmag",
    ),
    "mag_error": (
        "mag_err",
        "mag error",
        "mag_error",
        "magerr",
        "mag err",
        "magnitude error",
        "e mag",
        "e_mag",
        "emag",
        "error",
        "err",
        "sigma",
        "sigma mag",
        "sigma_mag",
    ),
    "flux": (
        "flux",
        "f",
        "flux density",
        "flux_density",
    ),
    "flux_error": (
        "flux error",
        "flux_error",
        "fluxerr",
        "flux err",
        "e flux",
        "e_flux",
        "eflux",
    ),
    "band": (
        "filter",
        "phot filter",
        "phot_filter",
        "passband",
        "band",
        "filter band",
        "filter_band",
        "v g band",
        "v_g_band",
        "vgband",
    ),
    "camera": (
        "camera",
        "camera#",
        "camera id",
        "camera_id",
        "camera number",
        "camera_number",
        "camera name",
        "camera_name",
        "cam",
        "cam id",
        "cam_id",
    ),
    "quality": (
        "quality",
        "quality flag",
        "quality_flag",
        "good bad",
        "good_bad",
        "goodbad",
        "flag",
    ),
    "saturated": (
        "saturated",
        "sat",
        "saturation",
    ),
}


@dataclass(frozen=True)
class PublicationLightCurve:
    df: pd.DataFrame
    source_path: Path
    time_column: str
    y_kind: str
    y_label: str
    default_invert_y: bool


@dataclass(frozen=True)
class PanelPlotResult:
    ax: Any
    frame: pd.DataFrame
    diagnostics: dict[str, object] | None = None


@dataclass(frozen=True)
class _PreparedPanelFrame:
    df: pd.DataFrame
    source_column: str
    y_label: str
    default_invert_y: bool


def _normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in df.columns:
        lookup.setdefault(_normalize_column_name(column), str(column))
    return lookup


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = _column_lookup(df)
    for alias in aliases:
        column = lookup.get(_normalize_column_name(alias))
        if column is not None:
            return column
    return None


def _format_band(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)):
        rounded = int(round(float(numeric)))
        if abs(float(numeric) - rounded) < 1e-9:
            if rounded == 0:
                return "g"
            if rounded == 1:
                return "V"

    text = str(value).strip()
    if not text:
        return "unknown"
    lower = text.lower()
    if lower == "v":
        return "V"
    if lower in {"g", "u", "r", "i", "z"}:
        return lower
    if lower == "b":
        return "B"
    return text


def _as_boolean_series(values: pd.Series, *, default: bool) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)

    numeric = pd.to_numeric(values, errors="coerce")
    non_null = values.notna()
    if non_null.any() and numeric[non_null].notna().all():
        return numeric.fillna(1 if default else 0).astype(float) != 0.0

    text = values.astype("string").fillna("").str.strip().str.upper()
    true_tokens = {"1", "TRUE", "T", "YES", "Y", "GOOD", "G", "OK", "A"}
    false_tokens = {"0", "FALSE", "F", "NO", "N", "BAD", "B"}
    out = pd.Series(default, index=values.index, dtype=bool)
    out[text.isin(true_tokens)] = True
    out[text.isin(false_tokens)] = False
    return out


def _quality_good_series(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(True, index=index, dtype=bool)
    return _as_boolean_series(pd.Series(values, index=index), default=True)


def _saturated_series(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index, dtype=bool)
    return _as_boolean_series(pd.Series(values, index=index), default=False)


def _read_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", skip_blank_lines=True)


def _looks_like_asassn_csv(df: pd.DataFrame) -> bool:
    cols = {str(col).strip().lower() for col in df.columns}
    has_jd = "jd" in cols
    has_measurement = bool({"flux", "flux error", "flux_error", "mag", "mag error", "mag_err"} & cols)
    has_asassn_context = bool({"filter", "quality", "camera", "limit", "fwhm", "band"} & cols)
    return has_jd and has_measurement and has_asassn_context


def _load_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        default_config = Path.home() / ".config" / "matplotlib"
        if not os.access(default_config, os.W_OK):
            cache_dir = Path(tempfile.gettempdir()) / "malca-matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    apply_publication_rcparams(plt)
    return plt, AutoMinorLocator


def apply_publication_rcparams(plt=None) -> None:
    """Apply MALCA publication rcParams to the active matplotlib runtime."""
    if plt is None:
        import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_STYLE)


PUBLICATION_TIGHT_LAYOUT_PAD = 0.3


def finalize_publication_figure(
    fig,
    *,
    pad: float = PUBLICATION_TIGHT_LAYOUT_PAD,
    w_pad: float | None = None,
    h_pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Apply aggressive tight_layout to a figure before saving.

    All publication plots should call this instead of fig.tight_layout()
    or using constrained_layout / bbox_inches="tight".
    """
    fig.tight_layout(pad=pad, w_pad=w_pad or pad, h_pad=h_pad or pad, rect=rect)


def save_publication_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 300,
    pad: float = PUBLICATION_TIGHT_LAYOUT_PAD,
    w_pad: float | None = None,
    h_pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
    close: bool = True,
    **kwargs,
) -> None:
    """Finalize layout and save a publication figure."""
    finalize_publication_figure(fig, pad=pad, w_pad=w_pad, h_pad=h_pad, rect=rect)
    fig.savefig(path, dpi=dpi, bbox_inches=None, **kwargs)
    if close:
        import matplotlib.pyplot as _plt
        _plt.close(fig)


def figsize_scale(
    figsize: tuple[float, float],
    *,
    reference: tuple[float, float] = FIG_HEATMAP_REFERENCE,
) -> float:
    """Return scale factor relative to a reference figure size (min of width/height ratios)."""
    return min(figsize[0] / reference[0], figsize[1] / reference[1])


def scaled_scatter_size(base_size: float, scale: float, *, minimum: float = 2.0) -> float:
    """Scale matplotlib scatter ``s`` (points^2) for a smaller figure."""
    return max(minimum, float(base_size) * scale**2)


def scaled_font_size(base_size: float, scale: float, *, minimum: float = 7.0) -> float:
    return max(minimum, float(base_size) * scale)


def figsize_from_legacy(width: float, height: float) -> tuple[float, float]:
    """Scale a legacy figure size to single- or two-column publication width."""
    target_w = FIG_SINGLE_COL_WIDTH if width <= 10.0 else FIG_TWO_COL_WIDTH
    scale = target_w / float(width)
    return (target_w, float(height) * scale)


def figsize_heatmap_single_col(*, aspect: float = 8 / 10) -> tuple[float, float]:
    """Return a single-column heatmap size with the given width/height aspect."""
    return (FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * aspect)


def figsize_two_col_grid(
    ncols: int,
    nrows: int,
    *,
    row_height: float = FIG_GRID_ROW_HEIGHT,
) -> tuple[float, float]:
    """Multi-panel figure spanning two-column width."""
    del ncols  # subplot grid divides the fixed publication width internally
    return (FIG_TWO_COL_WIDTH, row_height * nrows)


def scaled_publication_text_sizes(
    figsize: tuple[float, float],
    *,
    reference: tuple[float, float] = FIG_HEATMAP_REFERENCE,
    label: float = 14.0,
    title: float = 16.0,
    colorbar: float = 14.0,
) -> dict[str, float]:
    """Scale explicit label/title sizes for a smaller publication figsize."""
    scale = figsize_scale(figsize, reference=reference)
    return {
        "label": scaled_font_size(label, scale),
        "title": scaled_font_size(title, scale, minimum=8.0),
        "colorbar": scaled_font_size(colorbar, scale),
    }


def figsize_heatmap_two_col(
    n_rows: int,
    *,
    row_height: float = FIG_HEATMAP_ROW_HEIGHT,
    min_height: float = FIG_HEATMAP_MIN_HEIGHT,
) -> tuple[float, float]:
    return (FIG_TWO_COL_WIDTH, max(min_height, row_height * n_rows))


def figsize_feature_grid(
    ncols: int,
    nrows: int,
    *,
    panel_width: float = FIG_GRID_PANEL_WIDTH,
    row_height: float = FIG_GRID_ROW_HEIGHT,
) -> tuple[float, float]:
    return (panel_width * ncols, row_height * nrows)


def _read_raw_lightcurve(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".dat", ".dat2", ".dat3"}:
        return load_lightcurve_df(path, file_ext=suffix[1:])
    if suffix == ".csv":
        df = _read_csv_frame(path)
        if _looks_like_asassn_csv(df):
            try:
                return normalize_asassn_lightcurve(df, source_path=path)
            except Exception:
                return df
        return df
    return _read_csv_frame(path)


def normalize_lightcurve_frame(df: pd.DataFrame, source_path: str | Path) -> PublicationLightCurve:
    """Normalize a generic light-curve table to plotting columns."""
    source = Path(source_path)
    if df.empty:
        raise ValueError(f"Light curve table is empty: {source}")

    time_col = _find_column(df, COLUMN_ALIASES["time"])
    mag_col = _find_column(df, COLUMN_ALIASES["mag"])
    mag_err_col = _find_column(df, COLUMN_ALIASES["mag_error"])
    flux_col = _find_column(df, COLUMN_ALIASES["flux"])
    flux_err_col = _find_column(df, COLUMN_ALIASES["flux_error"])
    band_col = _find_column(df, COLUMN_ALIASES["band"])
    camera_col = _find_column(df, COLUMN_ALIASES["camera"])
    quality_col = _find_column(df, COLUMN_ALIASES["quality"])
    saturated_col = _find_column(df, COLUMN_ALIASES["saturated"])

    if time_col is None:
        raise ValueError(
            "Could not infer a time column. Expected one of: "
            f"{', '.join(COLUMN_ALIASES['time'])}"
        )

    if mag_col is not None:
        y_col = mag_col
        y_err_col = mag_err_col
        y_kind = "mag"
        y_label = "Magnitude [mag]"
        default_invert_y = True
    elif flux_col is not None:
        y_col = flux_col
        y_err_col = flux_err_col
        y_kind = "flux"
        y_label = "Flux"
        default_invert_y = False
    else:
        raise ValueError(
            "Could not infer a magnitude or flux column. Expected one of: "
            f"{', '.join(COLUMN_ALIASES['mag'] + COLUMN_ALIASES['flux'])}"
        )

    out = pd.DataFrame(index=df.index)
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    out["value"] = pd.to_numeric(df[y_col], errors="coerce")
    if y_err_col is not None:
        out["value_error"] = pd.to_numeric(df[y_err_col], errors="coerce")
    else:
        out["value_error"] = np.nan

    if band_col is not None:
        out["band"] = df[band_col].map(_format_band)
    else:
        out["band"] = "all"

    if camera_col is not None:
        camera = df[camera_col].astype("string").fillna("").str.strip()
        out["camera"] = camera.where(camera != "", "unknown")
    else:
        out["camera"] = "all"

    out["good_quality"] = _quality_good_series(
        df[quality_col] if quality_col is not None else None,
        df.index,
    )
    out["saturated"] = _saturated_series(
        df[saturated_col] if saturated_col is not None else None,
        df.index,
    )

    return PublicationLightCurve(
        df=out.reset_index(drop=True),
        source_path=source,
        time_column=time_col,
        y_kind=y_kind,
        y_label=y_label,
        default_invert_y=default_invert_y,
    )


def load_lightcurve(path: str | Path) -> PublicationLightCurve:
    """Load SkyPatrol/generic CSV or ASAS-SN dat/dat2/dat3 light curves."""
    lc_path = Path(path).expanduser()
    if not lc_path.exists():
        raise FileNotFoundError(f"Light curve file not found: {lc_path}")
    return normalize_lightcurve_frame(_read_raw_lightcurve(lc_path), lc_path)


def _parse_band_filter(bands: Sequence[str] | None) -> set[str] | None:
    if not bands:
        return None
    return {_format_band(band) for band in bands}


def filter_lightcurve(
    lc: PublicationLightCurve,
    *,
    bands: Sequence[str] | None = None,
    include_bad_quality: bool = False,
    include_saturated: bool = False,
    max_error: float | None = 1.0,
) -> pd.DataFrame:
    """Apply conservative publication-plot filters."""
    df = lc.df.copy()
    mask = np.isfinite(df["time"]) & np.isfinite(df["value"])

    err = pd.to_numeric(df["value_error"], errors="coerce")
    err_valid = np.isfinite(err) & (err > 0)
    df.loc[~err_valid, "value_error"] = np.nan

    if lc.y_kind == "mag" and max_error is not None and np.isfinite(float(max_error)):
        mask &= (~err_valid) | (err <= float(max_error))

    if not include_bad_quality:
        mask &= df["good_quality"].fillna(True).astype(bool)

    if not include_saturated:
        mask &= ~df["saturated"].fillna(False).astype(bool)

    wanted_bands = _parse_band_filter(bands)
    if wanted_bands is not None:
        mask &= df["band"].isin(wanted_bands)

    return df.loc[mask].sort_values("time").reset_index(drop=True)


def _stable_color(label: str) -> str:
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 10
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    return palette[hue]


def _stable_marker(label: str) -> str:
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    return MARKERS[int(digest[:8], 16) % len(MARKERS)]


def _band_sort_key(label: str) -> tuple[int, str]:
    order = {"u": 0, "B": 1, "V": 2, "g": 3, "r": 4, "i": 5, "z": 6, "all": 7}
    return order.get(label, 99), label


def _default_title(path: Path) -> str:
    stem = path.stem
    for suffix in ("-light-curves", "_light_curves", "_lightcurve", "-lightcurve"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_time_axis(
    time: pd.Series,
    *,
    source_column: str,
    offset: str = "auto",
) -> tuple[pd.Series, str]:
    """Return plotted time values and an axis label."""
    requested = str(offset).strip().lower()
    norm_col = _normalize_column_name(source_column)
    raw = pd.to_numeric(time, errors="coerce")
    finite = raw[np.isfinite(raw)]
    median = float(finite.median()) if not finite.empty else np.nan

    if requested in {"none", "raw", "0"}:
        if "mjd" in norm_col:
            return raw, "MJD"
        if "jd" in norm_col and np.isfinite(median) and median < 100000:
            return raw, "JD - 2450000 [d]"
        return raw, f"{source_column} [days]"

    if requested == "auto":
        if "mjd" in norm_col:
            return raw, "MJD"
        if "jd" in norm_col:
            if np.isfinite(median) and median > 2000000:
                return raw - JD_OFFSET, "JD - 2450000 [d]"
            return raw, "JD - 2450000 [d]"
        return raw, f"{source_column} [days]"

    try:
        numeric_offset = float(requested)
    except ValueError as exc:
        raise ValueError("--time-offset must be 'auto', 'none', or a numeric value") from exc
    return raw - numeric_offset, f"{source_column} - {numeric_offset:g} [d]"


def _group_label(row: pd.Series, group_by: str) -> str:
    if group_by == "none":
        return "all"
    if group_by in {"group", "column"}:
        return str(row["group"]) if "group" in row.index else "all"
    if group_by == "band":
        return str(row["band"])
    if group_by == "camera":
        return str(row["camera"])
    if group_by == "band-camera":
        return f"{row['band']} / {row['camera']}"
    raise ValueError(f"Unknown group_by value: {group_by}")


def _group_color(label: str, group_by: str) -> str:
    if group_by == "band":
        return BAND_COLORS.get(label, _stable_color(label))
    if group_by == "band-camera":
        band = label.split("/", 1)[0].strip()
        return BAND_COLORS.get(band, _stable_color(label))
    return BAND_COLORS.get(label, _stable_color(label))


def publication_style_context():
    """Return a matplotlib rc_context for MALCA publication-style figures."""
    plt, _ = _load_matplotlib()
    return plt.rc_context(PUBLICATION_STYLE)


def style_publication_axis(
    ax,
    *,
    grid: bool = True,
    minor_ticks: bool = True,
    top: bool = True,
    right: bool = True,
) -> Any:
    """Apply publication tick/grid styling to an existing matplotlib axis."""
    _, auto_minor_locator = _load_matplotlib()
    if grid:
        ax.grid(True, which="major", linewidth=0.4, alpha=0.28)
    if minor_ticks:
        ax.xaxis.set_minor_locator(auto_minor_locator())
        ax.yaxis.set_minor_locator(auto_minor_locator())
    ax.tick_params(which="both", direction="in", top=top, right=right)
    ax.tick_params(which="major", length=4.0, width=0.8)
    ax.tick_params(which="minor", length=2.0, width=0.6)
    return ax


def _specified_or_found(df: pd.DataFrame, specified: str | None, aliases: Sequence[str]) -> str | None:
    if specified is not None:
        if specified not in df.columns:
            raise KeyError(f"Column {specified!r} not found")
        return specified
    return _find_column(df, aliases)


def _first_present_column(df: pd.DataFrame, columns: Sequence[str]) -> str | None:
    lookup = _column_lookup(df)
    for column in columns:
        match = lookup.get(_normalize_column_name(column))
        if match is not None:
            return match
    return None


def _ensure_panel_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "value_error" not in out.columns:
        out["value_error"] = np.nan
    if "band" not in out.columns:
        out["band"] = "all"
    else:
        out["band"] = out["band"].map(_format_band)
    if "camera" not in out.columns:
        out["camera"] = "all"
    else:
        camera = out["camera"].astype("string").fillna("").str.strip()
        out["camera"] = camera.where(camera != "", "unknown")
    if "good_quality" not in out.columns:
        out["good_quality"] = True
    if "saturated" not in out.columns:
        out["saturated"] = False
    return out


def _prepare_panel_frame(
    data: PublicationLightCurve | pd.DataFrame,
    df: pd.DataFrame | None = None,
    *,
    source_path: str | Path = "<dataframe>",
    time_col: str | None = None,
    value_col: str | None = None,
    error_col: str | None = None,
    band_col: str | None = None,
    camera_col: str | None = None,
    group_col: str | None = None,
    y_label: str | None = None,
    default_invert_y: bool | None = None,
    extra_numeric_cols: Sequence[str] = (),
) -> _PreparedPanelFrame:
    if isinstance(data, PublicationLightCurve):
        frame = data.df if df is None else df
        out = _ensure_panel_columns(frame)
        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out["value_error"] = pd.to_numeric(out["value_error"], errors="coerce")
        if group_col is not None:
            if group_col not in frame.columns:
                raise KeyError(f"Column {group_col!r} not found")
            group = frame[group_col].astype("string").fillna("").str.strip()
            out["group"] = group.where(group != "", "unknown")
        for column in extra_numeric_cols:
            if column in frame.columns:
                out[column] = pd.to_numeric(frame[column], errors="coerce")
        return _PreparedPanelFrame(
            df=out.reset_index(drop=True),
            source_column=data.time_column,
            y_label=y_label or data.y_label,
            default_invert_y=data.default_invert_y if default_invert_y is None else bool(default_invert_y),
        )

    if df is not None:
        raise TypeError("Pass either a PublicationLightCurve plus optional df, or a raw DataFrame")

    raw = pd.DataFrame(data).copy()
    if raw.empty:
        raise ValueError(f"Light curve table is empty: {source_path}")

    if {"time", "value"}.issubset(raw.columns) and time_col is None and value_col is None:
        out = _ensure_panel_columns(raw)
        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out["value_error"] = pd.to_numeric(out["value_error"], errors="coerce")
        source_column = "time"
        label = y_label or "Magnitude [mag]"
        invert = True if default_invert_y is None else bool(default_invert_y)
    else:
        source_column = _specified_or_found(raw, time_col, COLUMN_ALIASES["time"])
        if source_column is None:
            raise ValueError(
                "Could not infer a time column. Expected one of: "
                f"{', '.join(COLUMN_ALIASES['time'])}"
            )

        mag_col = _specified_or_found(raw, value_col, COLUMN_ALIASES["mag"])
        flux_col = None if value_col is not None else _find_column(raw, COLUMN_ALIASES["flux"])
        if mag_col is not None:
            y_col = mag_col
            y_err_col = _specified_or_found(raw, error_col, COLUMN_ALIASES["mag_error"])
            label = y_label or ("Residual [mag]" if _normalize_column_name(y_col) in {"resid", "residual"} else "Magnitude [mag]")
            invert = True if default_invert_y is None else bool(default_invert_y)
        elif flux_col is not None:
            y_col = flux_col
            y_err_col = _specified_or_found(raw, error_col, COLUMN_ALIASES["flux_error"])
            label = y_label or "Flux"
            invert = False if default_invert_y is None else bool(default_invert_y)
        else:
            raise ValueError(
                "Could not infer a magnitude or flux column. Expected one of: "
                f"{', '.join(COLUMN_ALIASES['mag'] + COLUMN_ALIASES['flux'])}"
            )

        band_name = _specified_or_found(raw, band_col, COLUMN_ALIASES["band"])
        camera_name = _specified_or_found(raw, camera_col, COLUMN_ALIASES["camera"])
        quality_name = _find_column(raw, COLUMN_ALIASES["quality"])
        saturated_name = _find_column(raw, COLUMN_ALIASES["saturated"])

        out = pd.DataFrame(index=raw.index)
        out["time"] = pd.to_numeric(raw[source_column], errors="coerce")
        out["value"] = pd.to_numeric(raw[y_col], errors="coerce")
        out["value_error"] = pd.to_numeric(raw[y_err_col], errors="coerce") if y_err_col is not None else np.nan
        out["band"] = raw[band_name].map(_format_band) if band_name is not None else "all"
        if camera_name is not None:
            camera = raw[camera_name].astype("string").fillna("").str.strip()
            out["camera"] = camera.where(camera != "", "unknown")
        else:
            out["camera"] = "all"
        out["good_quality"] = _quality_good_series(raw[quality_name] if quality_name is not None else None, raw.index)
        out["saturated"] = _saturated_series(raw[saturated_name] if saturated_name is not None else None, raw.index)

    resolved_group_col = group_col
    if resolved_group_col is not None:
        if resolved_group_col not in raw.columns:
            raise KeyError(f"Column {resolved_group_col!r} not found")
        group = raw[resolved_group_col].astype("string").fillna("").str.strip()
        out["group"] = group.where(group != "", "unknown")

    for column in extra_numeric_cols:
        actual = _first_present_column(raw, (column,))
        if actual is not None:
            out[column] = pd.to_numeric(raw[actual], errors="coerce")

    return _PreparedPanelFrame(
        df=out.reset_index(drop=True),
        source_column=source_column,
        y_label=label,
        default_invert_y=invert,
    )


def _plot_vertical_lines(
    ax,
    values: Sequence[object] | None,
    *,
    source_column: str,
    time_offset: str,
    default_style: dict[str, object] | None = None,
) -> None:
    if values is None:
        return
    base_style = {"color": "0.35", "linestyle": "--", "linewidth": 0.9, "alpha": 0.7}
    if default_style:
        base_style.update(default_style)
    for item in values:
        if isinstance(item, dict):
            x_raw = item.get("x", item.get("time", item.get("jd")))
            style = {k: v for k, v in item.items() if k not in {"x", "time", "jd"}}
        else:
            x_raw = item
            style = {}
        if x_raw is None:
            continue
        x_plot, _ = resolve_time_axis(pd.Series([x_raw]), source_column=source_column, offset=time_offset)
        x_val = pd.to_numeric(x_plot, errors="coerce").iloc[0]
        if pd.notna(x_val) and np.isfinite(float(x_val)):
            kwargs = dict(base_style)
            kwargs.update(style)
            ax.axvline(float(x_val), **kwargs)


def _plot_vertical_spans(
    ax,
    spans: Sequence[object] | None,
    *,
    source_column: str,
    time_offset: str,
    default_style: dict[str, object] | None = None,
) -> None:
    if spans is None:
        return
    base_style = {"color": "0.5", "alpha": 0.12, "linewidth": 0}
    if default_style:
        base_style.update(default_style)
    for item in spans:
        if isinstance(item, dict):
            x0_raw = item.get("x0", item.get("start", item.get("start_jd")))
            x1_raw = item.get("x1", item.get("end", item.get("end_jd")))
            style = {k: v for k, v in item.items() if k not in {"x0", "x1", "start", "end", "start_jd", "end_jd"}}
        else:
            try:
                x0_raw, x1_raw = item
            except Exception:
                continue
            style = {}
        x_plot, _ = resolve_time_axis(pd.Series([x0_raw, x1_raw]), source_column=source_column, offset=time_offset)
        vals = pd.to_numeric(x_plot, errors="coerce")
        if vals.notna().all() and np.isfinite(vals.to_numpy(dtype=float)).all():
            kwargs = dict(base_style)
            kwargs.update(style)
            ax.axvspan(float(vals.iloc[0]), float(vals.iloc[1]), **kwargs)


def _event_overlay_specs(event_runs: Sequence[dict[str, object]], event_kind: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    color = DIP_EVENT_COLOR if event_kind == "dip" else JUMP_EVENT_COLOR if event_kind == "jump" else "0.35"
    lines: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    for run in event_runs:
        start = run.get("start_jd", run.get("start", run.get("jd_start")))
        end = run.get("end_jd", run.get("end", run.get("jd_end")))
        if start is not None and end is not None:
            spans.append({"start_jd": start, "end_jd": end, "color": color, "alpha": 0.08})
            lines.append({"x": start, "color": color, "linestyle": "--", "linewidth": 1.0, "alpha": 0.7})
            if end != start:
                lines.append({"x": end, "color": color, "linestyle": "--", "linewidth": 1.0, "alpha": 0.7})
        params = run.get("params")
        if isinstance(params, dict):
            t0 = params.get("t0")
            if t0 is not None:
                lines.append({"x": t0, "color": color, "linestyle": "--", "linewidth": 1.0, "alpha": 0.9})
    return lines, spans


def _plot_baseline_overlay(
    ax,
    baseline: pd.DataFrame,
    *,
    source_column: str,
    time_offset: str,
    time_col: str | None,
    value_col: str,
    group_col: str | None,
    color_map: dict[str, str] | None = None,
    label: str | None = "Baseline",
    style: dict[str, object] | None = None,
) -> None:
    if baseline.empty:
        return
    t_col = time_col or _first_present_column(baseline, ("time", "JD", "jd", source_column))
    if t_col is None or value_col not in baseline.columns:
        return
    plot = baseline.copy()
    plot["_baseline_time"], _ = resolve_time_axis(plot[t_col], source_column=source_column, offset=time_offset)
    plot["_baseline_value"] = pd.to_numeric(plot[value_col], errors="coerce")
    plot = plot[np.isfinite(plot["_baseline_time"]) & np.isfinite(plot["_baseline_value"])]
    if plot.empty:
        return

    line_style = {"linestyle": "-", "linewidth": 1.5, "alpha": 0.85, "zorder": 5}
    if style:
        line_style.update(style)

    if group_col is not None and group_col in plot.columns:
        first_label = True
        for group_value, group_df in plot.groupby(group_col, dropna=False):
            part = group_df.sort_values("_baseline_time")
            kwargs = dict(line_style)
            if color_map is not None:
                kwargs.setdefault("color", color_map.get(str(group_value), _stable_color(str(group_value))))
            ax.plot(
                part["_baseline_time"],
                part["_baseline_value"],
                label=label if first_label else None,
                **kwargs,
            )
            first_label = False
    else:
        plot = plot.sort_values("_baseline_time")
        kwargs = {"color": "0.1", **line_style}
        ax.plot(plot["_baseline_time"], plot["_baseline_value"], label=label, **kwargs)


def _legend_ncol(labels: Sequence[str]) -> int:
    return 1 if len(labels) <= 5 else 2


def plot_lightcurve_panel(
    ax,
    data: PublicationLightCurve | pd.DataFrame,
    df: pd.DataFrame | None = None,
    *,
    title: str | None = None,
    group_by: str = "band",
    group_col: str | None = None,
    show_errorbars: bool = True,
    invert_y: bool | None = None,
    time_offset: str = "auto",
    legend: str = "auto",
    marker_size: float = 3.5,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    time_col: str | None = None,
    value_col: str | None = None,
    error_col: str | None = None,
    band_col: str | None = None,
    camera_col: str | None = None,
    y_label: str | None = None,
    default_invert_y: bool | None = None,
    baseline: pd.DataFrame | None = None,
    baseline_col: str = "baseline",
    baseline_time_col: str | None = None,
    baseline_group_col: str | None = None,
    baseline_label: str | None = "Baseline",
    baseline_style: dict[str, object] | None = None,
    vertical_lines: Sequence[object] | None = None,
    vertical_spans: Sequence[object] | None = None,
    event_runs: Sequence[dict[str, object]] | None = None,
    event_kind: str = "dip",
    highlight_mask: Sequence[bool] | pd.Series | np.ndarray | None = None,
    highlight_label: str | None = None,
    highlight_style: dict[str, object] | None = None,
) -> PanelPlotResult:
    prepared = _prepare_panel_frame(
        data,
        df,
        time_col=time_col,
        value_col=value_col,
        error_col=error_col,
        band_col=band_col,
        camera_col=camera_col,
        group_col=group_col,
        y_label=y_label,
        default_invert_y=default_invert_y,
        extra_numeric_cols=(baseline_col,),
    )

    plot_df = prepared.df.copy()
    if plot_df.empty:
        raise ValueError("No light-curve points remain after filtering")

    plot_df["time_plot"], x_label = resolve_time_axis(
        plot_df["time"],
        source_column=prepared.source_column,
        offset=time_offset,
    )
    plot_df["plot_group"] = plot_df.apply(_group_label, axis=1, group_by=group_by)
    mask = np.isfinite(plot_df["time_plot"]) & np.isfinite(plot_df["value"])
    plot_df = plot_df.loc[mask].reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No finite light-curve points remain after filtering")

    err = pd.to_numeric(plot_df["value_error"], errors="coerce")
    err_valid = np.isfinite(err) & (err > 0)
    plot_df.loc[~err_valid, "value_error"] = np.nan

    groups = sorted(plot_df["plot_group"].dropna().unique(), key=_band_sort_key)
    color_map: dict[str, str] = {}
    for label in groups:
        part = plot_df[plot_df["plot_group"] == label]
        if part.empty:
            continue
        color = _group_color(str(label), group_by)
        color_map[str(label)] = color
        marker = (
            _stable_marker(str(label))
            if group_by != "band"
            else _stable_marker(str(part["band"].iloc[0]))
        )
        plot_label = None if str(label) == "all" else str(label)
        yerr = part["value_error"].to_numpy()
        has_yerr = show_errorbars and np.isfinite(yerr).any()

        if has_yerr:
            ax.errorbar(
                part["time_plot"],
                part["value"],
                yerr=yerr,
                fmt=marker,
                linestyle="none",
                markersize=marker_size,
                markeredgecolor="0.15",
                markeredgewidth=0.35,
                color=color,
                ecolor=color,
                elinewidth=0.55,
                capsize=1.2,
                alpha=0.82,
                label=plot_label,
            )
        else:
            ax.scatter(
                part["time_plot"],
                part["value"],
                s=marker_size**2,
                marker=marker,
                linewidths=0.35,
                edgecolors="0.15",
                color=color,
                alpha=0.82,
                label=plot_label,
            )

    if highlight_mask is not None:
        highlight = np.asarray(highlight_mask, dtype=bool)
        if highlight.size == len(plot_df) and highlight.any():
            style = {
                "s": max(marker_size**2 * 2.4, 18.0),
                "facecolors": "none",
                "edgecolors": "crimson",
                "linewidths": 0.9,
                "alpha": 0.9,
                "label": highlight_label,
                "zorder": 6,
            }
            if highlight_style:
                style.update(highlight_style)
            ax.scatter(plot_df.loc[highlight, "time_plot"], plot_df.loc[highlight, "value"], **style)

    baseline_frame = baseline
    if baseline_frame is None and baseline_col in plot_df.columns:
        baseline_frame = plot_df.rename(columns={"time_plot": "_time_plot"})
    if baseline_frame is not None:
        resolved_group_col = baseline_group_col
        if resolved_group_col is None:
            if group_by == "camera" and "camera" in baseline_frame.columns:
                resolved_group_col = "camera"
            elif group_by == "group" and "group" in baseline_frame.columns:
                resolved_group_col = "group"
        _plot_baseline_overlay(
            ax,
            baseline_frame,
            source_column=prepared.source_column,
            time_offset=time_offset,
            time_col=baseline_time_col,
            value_col=baseline_col,
            group_col=resolved_group_col,
            color_map=color_map,
            label=baseline_label,
            style=baseline_style,
        )

    event_lines: list[dict[str, object]] = []
    event_spans: list[dict[str, object]] = []
    if event_runs:
        event_lines, event_spans = _event_overlay_specs(event_runs, event_kind)
    span_items = list(vertical_spans) if vertical_spans is not None else []
    line_items = list(vertical_lines) if vertical_lines is not None else []
    _plot_vertical_spans(
        ax,
        [*span_items, *event_spans],
        source_column=prepared.source_column,
        time_offset=time_offset,
    )
    _plot_vertical_lines(
        ax,
        [*line_items, *event_lines],
        source_column=prepared.source_column,
        time_offset=time_offset,
    )

    if title:
        ax.set_title(title, pad=8)

    ax.set_xlabel(xlabel or x_label)
    ax.set_ylabel(ylabel or prepared.y_label)
    should_invert_y = invert_y if invert_y is not None else prepared.default_invert_y
    if should_invert_y:
        ax.invert_yaxis()

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    style_publication_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    if legend != "none" and handles:
        if legend == "outside":
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
        else:
            ax.legend(handles, labels, loc="best", frameon=False, ncol=_legend_ncol(labels))

    return PanelPlotResult(ax=ax, frame=plot_df)


def _add_residual_thresholds(ax, threshold: float | None, *, shade: bool = True) -> None:
    if threshold is None or not np.isfinite(float(threshold)):
        return
    thr = abs(float(threshold))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    if shade:
        ax.fill_between([x0, x1], thr, y_max, color="0.85", alpha=0.45, zorder=0)
        ax.fill_between([x0, x1], y_min, -thr, color="0.85", alpha=0.38, zorder=0)
    ax.axhline(0.0, color="0.1", linestyle="--", alpha=0.45, linewidth=0.8, zorder=1)
    ax.axhline(thr, color="0.1", linestyle="-", alpha=0.75, linewidth=0.8, zorder=1)
    ax.axhline(-thr, color="0.1", linestyle="-", alpha=0.75, linewidth=0.8, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def plot_residual_panel(
    ax,
    data: PublicationLightCurve | pd.DataFrame,
    *,
    residual_col: str = "resid",
    error_col: str | None = None,
    threshold: float | None = None,
    shade_threshold: bool = True,
    ylabel: str = "Residual [mag]",
    invert_y: bool = True,
    **kwargs,
) -> PanelPlotResult:
    result = plot_lightcurve_panel(
        ax,
        data,
        value_col=residual_col,
        error_col=error_col,
        y_label=ylabel,
        default_invert_y=invert_y,
        **kwargs,
    )
    _add_residual_thresholds(ax, threshold, shade=shade_threshold)
    return result


def _phase_input_frame(
    data: PublicationLightCurve | pd.DataFrame,
    *,
    time_col: str | None,
    value_col: str | None,
    error_col: str | None,
    band_col: str | None,
    camera_col: str | None,
    residual_col: str,
) -> pd.DataFrame:
    prepared = _prepare_panel_frame(
        data,
        time_col=time_col,
        value_col=value_col,
        error_col=error_col,
        band_col=band_col,
        camera_col=camera_col,
        extra_numeric_cols=(residual_col,),
    )
    raw = pd.DataFrame(
        {
            "JD": prepared.df["time"],
            "mag": prepared.df["value"],
            "error": prepared.df["value_error"],
            "camera#": prepared.df["camera"],
        }
    )
    band_map = {"g": 0, "V": 1}
    raw["v_g_band"] = prepared.df["band"].map(lambda value: band_map.get(str(value), np.nan))
    if residual_col in prepared.df.columns:
        raw[residual_col] = prepared.df[residual_col]
    return raw


def plot_phase_panel(
    ax,
    data: PublicationLightCurve | pd.DataFrame,
    *,
    period_days: float,
    epoch_jd: float | None = None,
    value_mode: str = "mag",
    align_v_to_g: bool = False,
    duplicate_cycles: bool = True,
    time_col: str | None = None,
    value_col: str | None = None,
    error_col: str | None = None,
    band_col: str | None = None,
    camera_col: str | None = None,
    residual_col: str = "resid",
    group_by: str = "band",
    title: str | None = None,
    show_errorbars: bool = True,
    legend: str = "auto",
    marker_size: float = 3.5,
    xlim: Sequence[float] | None = (0.0, 2.0),
    ylabel: str | None = None,
) -> PanelPlotResult:
    if period_days <= 0 or not np.isfinite(float(period_days)):
        raise ValueError("period_days must be positive and finite")

    from malca.phase import phase_fold_dataframe

    raw = _phase_input_frame(
        data,
        time_col=time_col,
        value_col=value_col,
        error_col=error_col,
        band_col=band_col,
        camera_col=camera_col,
        residual_col=residual_col,
    )
    phase_df, diagnostics = phase_fold_dataframe(
        raw,
        float(period_days),
        epoch_jd=epoch_jd,
        value_mode=value_mode,
        align_v_to_g=align_v_to_g,
        duplicate_cycles=duplicate_cycles,
        resid_col=residual_col,
    )
    if phase_df.empty:
        raise ValueError("No finite points for phase folding")

    phase_df = phase_df.copy()
    if "camera_label" not in phase_df.columns:
        phase_df["camera_label"] = phase_df["camera#"].astype(str) if "camera#" in phase_df.columns else "unknown"

    result = plot_lightcurve_panel(
        ax,
        phase_df,
        time_col="phase",
        value_col="phase_value",
        error_col="error" if "error" in phase_df.columns else None,
        band_col="v_g_band",
        camera_col="camera_label",
        group_by=group_by,
        show_errorbars=show_errorbars,
        legend=legend,
        marker_size=marker_size,
        time_offset="none",
        xlabel="Phase",
        ylabel=ylabel or ("Residual magnitude [mag]" if value_mode == "resid" else "Magnitude [mag]"),
        title=title,
        xlim=xlim,
    )
    for x in (0.0, 1.0, 2.0):
        ax.axvline(x, color="0.45", linestyle="--", linewidth=0.8, alpha=0.55)
    return PanelPlotResult(ax=ax, frame=result.frame, diagnostics=diagnostics)


def plot_lightcurve(
    lc: PublicationLightCurve,
    df: pd.DataFrame,
    *,
    output: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = FIG_LC_TWO_COL,
    dpi: int = 300,
    group_by: str = "band",
    show_errorbars: bool = True,
    invert_y: bool | None = None,
    time_offset: str = "auto",
    legend: str = "auto",
    marker_size: float = 3.5,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
) -> tuple[Any, Any]:
    if df.empty:
        raise ValueError("No light-curve points remain after filtering")

    plt, _ = _load_matplotlib()

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        if title is None:
            title = _default_title(lc.source_path)
        plot_lightcurve_panel(
            ax,
            lc,
            df,
            title=title,
            group_by=group_by,
            show_errorbars=show_errorbars,
            invert_y=invert_y,
            time_offset=time_offset,
            legend=legend,
            marker_size=marker_size,
            xlim=xlim,
            ylim=ylim,
        )

        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_publication_figure(fig, out_path, dpi=dpi, close=False)

        if show:
            plt.show()
        if close:
            plt.close(fig)
        return fig, ax


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca lc-plot",
        description="Create a publication-quality light-curve plot from SkyPatrol/generic CSV or ASAS-SN dat/dat2/dat3 files.",
    )
    parser.add_argument("path", type=Path, help="Light-curve CSV/dat/dat2/dat3 path.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output figure path. Defaults to '<input>_lightcurve.pdf' unless --show is used.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    parser.add_argument("--title", default=None, help="Figure title. Pass an empty string to suppress the title.")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), default=FIG_LC_TWO_COL, help="Figure size in inches.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument("--group-by", choices=("band", "camera", "band-camera", "none"), default="band", help="Legend/grouping dimension.")
    parser.add_argument("--bands", nargs="+", default=None, help="Only plot these bands, e.g. --bands g V.")
    parser.add_argument("--include-bad-quality", action="store_true", help="Include points flagged as bad quality.")
    parser.add_argument("--include-saturated", action="store_true", help="Include saturated points.")
    parser.add_argument("--max-error", type=float, default=1.0, help="Drop points with uncertainty above this value. Use --no-max-error to disable.")
    parser.add_argument("--no-max-error", action="store_true", help="Disable uncertainty filtering.")
    parser.add_argument("--no-errorbars", action="store_true", help="Plot markers without error bars.")
    parser.add_argument("--no-invert-y", action="store_true", help="Do not invert the y-axis for magnitudes.")
    parser.add_argument("--time-offset", default="auto", help="Time offset to subtract: auto, none, or a numeric value.")
    parser.add_argument("--legend", choices=("auto", "outside", "none"), default="auto", help="Legend placement.")
    parser.add_argument("--markersize", type=float, default=3.5, help="Marker size in points.")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"), default=None, help="X-axis limits after time offset.")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"), default=None, help="Y-axis limits.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    lc = load_lightcurve(args.path)
    max_error = None if args.no_max_error else args.max_error
    plot_df = filter_lightcurve(
        lc,
        bands=args.bands,
        include_bad_quality=args.include_bad_quality,
        include_saturated=args.include_saturated,
        max_error=max_error,
    )
    if plot_df.empty:
        raise SystemExit("No finite light-curve points remain after filtering.")

    output = args.output
    if output is None and not args.show:
        output = args.path.with_name(f"{args.path.stem}_lightcurve.pdf")

    invert_y = False if args.no_invert_y else None
    plot_lightcurve(
        lc,
        plot_df,
        output=output,
        show=args.show,
        close=not args.show,
        title=args.title,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=args.dpi,
        group_by=args.group_by,
        show_errorbars=not args.no_errorbars,
        invert_y=invert_y,
        time_offset=args.time_offset,
        legend=args.legend,
        marker_size=args.markersize,
        xlim=args.xlim,
        ylim=args.ylim,
    )

    band_summary = ", ".join(sorted(plot_df["band"].dropna().astype(str).unique(), key=_band_sort_key))
    print(
        f"Plotted {len(plot_df)}/{len(lc.df)} points from {args.path}"
        + (f" in bands: {band_summary}" if band_summary else "")
    )
    if output is not None:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
