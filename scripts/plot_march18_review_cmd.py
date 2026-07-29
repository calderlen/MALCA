#!/usr/bin/env python
"""Plot a Gaia CMD for selected review candidates."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.plotting.lightcurve_publication import (
    CMD_AXIS_LABEL_FONTSIZE,
    CMD_BG_SCATTER_SIZE,
    CMD_BG_SCATTER_ALPHA,
    CMD_BUCKET_STYLE,
    CMD_LEGEND_FONTSIZE,
    CMD_LEGEND_MARKERSCALE,
    CMD_MARKER_EDGE_HOLLOW,
    CMD_MARKER_EDGE_SOLID,
    CMD_TICK_LABEL_FONTSIZE,
    CMD_TICK_LENGTH,
    CMD_TICK_WIDTH,
    FIG_SINGLE_COL_SQUARE,
    apply_publication_rcparams,
    save_publication_figure,
)

apply_publication_rcparams(plt)

from malca.config import CMD_A_G_PER_AV, CMD_E_BP_RP_PER_AV, DEFAULT_OUTPUT_DIR, MIST_GRID_PATH
from malca.ltv.cmd import (
    cmd_uncertainty_from_fields,
    dustmaps_cmd_from_fields,
    estimate_cmd_masses,
    load_mist_grid,
    mist_mass_tracks,
    normalize_mist_cmd_grid,
)


MARCH18_RUN = DEFAULT_OUTPUT_DIR / "runs" / "runs_march18_bundle_all"
JULY1_RUN = DEFAULT_OUTPUT_DIR / "runs" / "dat3-full-extended_2026-07-01-v4"
MARCH18_REVIEW_DB = MARCH18_RUN / "review" / "review.taxonomy_filled.db"
MARCH18_CANDIDATES = MARCH18_RUN / "results" / "lc_events_classified.parquet"
MARCH18_OUTPUT = MARCH18_RUN / "results" / "march18_review_cmd_selected.png"
JULY1_REVIEW_DB = JULY1_RUN / "review" / "review.db"
JULY1_CANDIDATES = JULY1_RUN / "results" / "lc_events_vetted.parquet"
JULY1_OUTPUT = JULY1_RUN / "results" / "july1_review_cmd_dippers.png"
CMD_FIGSIZE = FIG_SINGLE_COL_SQUARE
CMD_XLIM = (-1.0, 3.0)
CMD_YLIM = (-4.0, 10.0)
DEFAULT_CMD_MODE = "dereddened"
DEFAULT_ISOCHRONE_AGES_MYR = (1.0, 3.0, 10.0, 30.0)
DEFAULT_MASS_LABELS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.3, 3.0)
ISOCHRONE_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")
ISOCHRONE_LINEWIDTH = 1.0
ISOCHRONE_ALPHA = 0.88
MASS_TRACK_COLOR = "#333333"
MASS_TRACK_ALPHA = 0.58
MASS_TRACK_LINEWIDTH = 0.75
MASS_LABEL_FONTSIZE = 5.6
ERRORBAR_ALPHA = 0.72
ERRORBAR_LINEWIDTH = 0.46
ERRORBAR_CAPSIZE = 1.1
ERRORBAR_CAPTHICK = 0.46
SOLID_CMD_SOURCES = frozenset({"dustmaps3d", "observed_no_extinction"})
HOLLOW_CMD_SOURCES = frozenset({"observed_fallback"})
PLOTTABLE_CMD_SOURCES = SOLID_CMD_SOURCES | HOLLOW_CMD_SOURCES

BUCKET_ORDER = ["Dipper", "Interesting", "LTV", "Microlensing", "Eclipsing binary", "Unknown"]
PRESETS = {
    "july1-dippers": {
        "review_db": JULY1_REVIEW_DB,
        "candidates": JULY1_CANDIDATES,
        "output": JULY1_OUTPUT,
        "buckets": ["Dipper"],
    },
    "march18": {
        "review_db": MARCH18_REVIEW_DB,
        "candidates": MARCH18_CANDIDATES,
        "output": MARCH18_OUTPUT,
        "buckets": BUCKET_ORDER,
    },
}
DEFAULT_PRESET = "july1-dippers"


def _json_frame(series: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    for value in series:
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = {}
            rows.append(parsed if isinstance(parsed, dict) else {})
        else:
            rows.append({})
    return pd.DataFrame(rows, index=series.index)


def _copy_external_column(
    out: pd.DataFrame,
    external: pd.DataFrame,
    target: str,
    aliases: tuple[str, ...] = (),
) -> None:
    for col in (target, *aliases):
        if col in external.columns:
            out[target] = external[col]
            return
    out[target] = np.nan


def _read_reviews(db_path: Path, *, only_reviewed: bool = True) -> pd.DataFrame:
    where = ""
    if only_reviewed:
        where = " WHERE workflow_status = 'reviewed' OR status = 'reviewed'"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT
                candidate_id,
                event_class,
                classification_confidence,
                morphology_primary,
                morphology_secondary,
                physical_primary,
                physical_secondary,
                workflow_status,
                status
            FROM reviews
            {where}
            """,
            conn,
        ).astype({"candidate_id": "string"})


def _compute_cmd_coordinates(candidates: pd.DataFrame) -> pd.DataFrame:
    derived = _json_frame(candidates.get("derived_stats", pd.Series(index=candidates.index, dtype=object)))
    external = _json_frame(candidates.get("external_stats", pd.Series(index=candidates.index, dtype=object)))

    out = pd.DataFrame({"candidate_id": candidates["candidate_id"].astype("string")})
    for col in ("bp_rp",):
        out[col] = pd.to_numeric(derived.get(col), errors="coerce")
    _copy_external_column(out, external, "phot_g_mean_mag")
    _copy_external_column(out, external, "phot_bp_mean_mag")
    _copy_external_column(out, external, "phot_rp_mean_mag")
    _copy_external_column(out, external, "phot_g_mean_mag_error", ("phot_g_mean_mag_err", "g_mag_error", "g_mag_err"))
    _copy_external_column(
        out,
        external,
        "phot_bp_mean_mag_error",
        ("phot_bp_mean_mag_err", "bp_mag_error", "bp_mag_err"),
    )
    _copy_external_column(
        out,
        external,
        "phot_rp_mean_mag_error",
        ("phot_rp_mean_mag_err", "rp_mag_error", "rp_mag_err"),
    )
    _copy_external_column(out, external, "distance_gspphot")
    _copy_external_column(
        out,
        external,
        "distance_gspphot_error",
        ("distance_gspphot_err", "dist_pc_error", "dist_pc_err"),
    )
    _copy_external_column(out, external, "parallax")
    _copy_external_column(out, external, "parallax_error", ("parallax_error_gaia", "plx_d", "plx_err"))
    _copy_external_column(out, external, "A_v_3d")
    _copy_external_column(out, external, "dust_sigma", ("A_v_3d_error", "A_v_3d_err", "av_error", "av_err"))
    _copy_external_column(out, external, "source_id")
    _copy_external_column(out, external, "gaia_id")

    cmd_rows: list[dict[str, object]] = []
    err_rows: list[dict[str, object]] = []
    for idx in out.index:
        row = out.loc[idx]
        coords = dustmaps_cmd_from_fields(
            g_mag=row.get("phot_g_mean_mag"),
            bp_rp=row.get("bp_rp"),
            dist_pc=row.get("distance_gspphot"),
            a_v_3d=row.get("A_v_3d"),
            bp_mag=row.get("phot_bp_mean_mag"),
            rp_mag=row.get("phot_rp_mean_mag"),
            parallax_mas=row.get("parallax"),
        )
        cmd_rows.append(coords)
        err_rows.append(
            cmd_uncertainty_from_fields(
                g_mag_err=row.get("phot_g_mean_mag_error"),
                bp_mag_err=row.get("phot_bp_mean_mag_error"),
                rp_mag_err=row.get("phot_rp_mean_mag_error"),
                dist_pc=row.get("distance_gspphot"),
                dist_pc_err=row.get("distance_gspphot_error"),
                parallax_mas=row.get("parallax"),
                parallax_err_mas=row.get("parallax_error"),
                a_v_3d=row.get("A_v_3d"),
                a_v_3d_err=row.get("dust_sigma"),
            )
        )

    cmd_df = pd.DataFrame(cmd_rows, index=out.index)
    if "bp_rp" in cmd_df.columns:
        out["bp_rp"] = pd.to_numeric(out["bp_rp"], errors="coerce").combine_first(
            pd.to_numeric(cmd_df["bp_rp"], errors="coerce")
        )
        cmd_df = cmd_df.drop(columns=["bp_rp"])
    err_df = pd.DataFrame(err_rows, index=out.index)
    out = pd.concat([out, cmd_df, err_df], axis=1)
    return out


def _assign_bucket(row: pd.Series) -> str | None:
    event_class = str(row.get("event_class") or "").strip().lower()
    morph_secondary = str(row.get("morphology_secondary") or "").strip().lower()
    physical_primary = str(row.get("physical_primary") or "").strip().lower()

    if event_class == "dipper":
        return "Dipper"
    if event_class == "ltv":
        return "LTV"
    if event_class == "microlensing":
        return "Microlensing"
    if event_class == "periodic" or morph_secondary in {"detached_binary_like", "eclipsing_like"}:
        return "Eclipsing binary"
    if physical_primary == "unknown" or event_class == "unknown":
        return "Unknown"
    return None


def _plottable_mask(df: pd.DataFrame) -> pd.Series:
    x_col = "cmd_plot_color" if "cmd_plot_color" in df.columns else "cmd_color"
    y_col = "cmd_plot_mag" if "cmd_plot_mag" in df.columns else "cmd_mag"
    return (
        df["cmd_coordinate_source"].isin(PLOTTABLE_CMD_SOURCES)
        & df[x_col].notna()
        & df[y_col].notna()
    )


def _apply_cmd_mode(plot_df: pd.DataFrame, cmd_mode: str) -> pd.DataFrame:
    out = plot_df.copy()
    if cmd_mode == "observed":
        out["cmd_plot_color"] = pd.to_numeric(out["bp_rp"], errors="coerce")
        out["cmd_plot_mag"] = pd.to_numeric(out["mg"], errors="coerce")
        out["cmd_plot_color_err"] = pd.to_numeric(out["bp_rp_err"], errors="coerce")
        out["cmd_plot_mag_err"] = pd.to_numeric(out["mg_err"], errors="coerce")
    elif cmd_mode == "dereddened":
        out["cmd_plot_color"] = pd.to_numeric(out["cmd_color"], errors="coerce")
        out["cmd_plot_mag"] = pd.to_numeric(out["cmd_mag"], errors="coerce")
        out["cmd_plot_color_err"] = pd.to_numeric(out["cmd_color_err"], errors="coerce")
        out["cmd_plot_mag_err"] = pd.to_numeric(out["cmd_mag_err"], errors="coerce")
    else:
        raise ValueError(f"Unsupported CMD mode: {cmd_mode}")
    out["cmd_plot_mode"] = cmd_mode
    return out


def _draw_errorbars(
    ax,
    df: pd.DataFrame,
    *,
    color,
    x_col: str,
    y_col: str,
    xerr_col: str,
    yerr_col: str,
    zorder: float,
) -> None:
    if df.empty or xerr_col not in df.columns or yerr_col not in df.columns:
        return
    xerr = pd.to_numeric(df[xerr_col], errors="coerce").to_numpy(dtype=float)
    yerr = pd.to_numeric(df[yerr_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xerr) | np.isfinite(yerr)
    if not ok.any():
        return
    ax.errorbar(
        pd.to_numeric(df.loc[ok, x_col], errors="coerce"),
        pd.to_numeric(df.loc[ok, y_col], errors="coerce"),
        xerr=xerr[ok],
        yerr=yerr[ok],
        fmt="none",
        ecolor=color,
        alpha=ERRORBAR_ALPHA,
        elinewidth=ERRORBAR_LINEWIDTH,
        capsize=ERRORBAR_CAPSIZE,
        capthick=ERRORBAR_CAPTHICK,
        zorder=zorder,
    )


def _scatter_solid(
    ax,
    df: pd.DataFrame,
    *,
    color,
    size,
    marker="o",
    alpha=1.0,
    zorder=2,
    label=None,
    edge_lw: float = CMD_MARKER_EDGE_SOLID,
    edgecolors=None,
    x_col: str = "cmd_plot_color",
    y_col: str = "cmd_plot_mag",
    xerr_col: str = "cmd_plot_color_err",
    yerr_col: str = "cmd_plot_mag_err",
    show_errorbars: bool = False,
) -> None:
    if df.empty:
        return
    if show_errorbars:
        _draw_errorbars(
            ax,
            df,
            color=color,
            x_col=x_col,
            y_col=y_col,
            xerr_col=xerr_col,
            yerr_col=yerr_col,
            zorder=zorder - 0.2,
        )
    ax.scatter(
        df[x_col],
        df[y_col],
        s=size,
        c=color,
        marker=marker,
        alpha=alpha,
        edgecolors=edgecolors if edgecolors is not None else "black",
        linewidths=edge_lw,
        label=label,
        zorder=zorder,
    )


def _scatter_hollow(
    ax,
    df: pd.DataFrame,
    *,
    color,
    size,
    marker="o",
    alpha=1.0,
    zorder=2,
    label=None,
    edge_lw: float = CMD_MARKER_EDGE_HOLLOW,
    edgecolors=None,
    x_col: str = "cmd_plot_color",
    y_col: str = "cmd_plot_mag",
    xerr_col: str = "cmd_plot_color_err",
    yerr_col: str = "cmd_plot_mag_err",
    show_errorbars: bool = False,
) -> None:
    if df.empty:
        return
    if show_errorbars:
        _draw_errorbars(
            ax,
            df,
            color=color,
            x_col=x_col,
            y_col=y_col,
            xerr_col=xerr_col,
            yerr_col=yerr_col,
            zorder=zorder - 0.2,
        )
    ax.scatter(
        df[x_col],
        df[y_col],
        s=size,
        facecolors="none",
        edgecolors=edgecolors if edgecolors is not None else "black",
        marker=marker,
        alpha=alpha,
        linewidths=edge_lw,
        label=label,
        zorder=zorder,
    )


CMD_DENSITY_BINS = 130
CMD_DENSITY_CMAP = "magma"
CMD_DENSITY_SMOOTH_SIGMA = 2.6
CMD_DENSITY_UPSAMPLE_FACTOR = 2
CMD_DENSITY_CONTOUR_LEVELS = 20
CMD_DENSITY_CUTOFF_PERCENTILE = 30.0
CMD_DENSITY_CUTOFF_FRAC_MAX = 0.04
CMD_MARKER_SIZE_SCALE = 0.72
CMD_REVIEW_BUCKET_STYLE = {bucket: dict(style) for bucket, style in CMD_BUCKET_STYLE.items()}
CMD_REVIEW_BUCKET_STYLE["LTV"].update({"marker": "o", "size": 16})
CMD_REVIEW_BUCKET_STYLE["Microlensing"].update({"marker": "^", "size": 16})
DEFAULT_BACKGROUND_STYLE = "density"
BACKGROUND_SAMPLE_CHOICES = ("gaia", "gaia+candidates", "candidates")
DEFAULT_BACKGROUND_SAMPLE = "gaia+candidates"
GAIA_BG_SOURCE_CHOICES = ("file", "tap")
GAIA_BG_CMD_MODE_CHOICES = ("match", "observed", "dereddened")
DEFAULT_GAIA_BG_SOURCE = "file"
DEFAULT_GAIA_BG_CMD_MODE = "match"
DEFAULT_GAIA_TAP_TABLE = "gaiadr3.gaia_source"
DEFAULT_GAIA_TAP_LIMIT = 200_000
DEFAULT_GAIA_TAP_RUWE_MAX = 1.4
DEFAULT_GAIA_DUST_CHUNK_SIZE = 50_000
GAIA_BG_PATH = Path("input/gaia/gaia_dr3_crossmatched.parquet")

# "density" background style: full-field pcolormesh, viridis, lighter smoothing.
CMD_DENSITY_STYLE_CMAP = "viridis"
CMD_DENSITY_STYLE_BINS = 480
CMD_DENSITY_STYLE_SMOOTH_SIGMA = 1.0
CMD_DENSITY_STYLE_CUTOFF_PERCENTILE = 8.0
CMD_DENSITY_STYLE_CUTOFF_FRAC_MAX = 0.005

# "corner" background style: monochrome scatter + line contours (corner-plot-like).
CMD_CORNER_STYLE_BINS = 100
CMD_CORNER_STYLE_SMOOTH_SIGMA = 1.2
CMD_CORNER_STYLE_SCATTER_COLOR = "#888888"
CMD_CORNER_STYLE_SCATTER_ALPHA = 0.28
CMD_CORNER_STYLE_CONTOUR_LEVELS = 6
CMD_CORNER_STYLE_CONTOUR_COLOR = "#333333"
CMD_CORNER_STYLE_CONTOUR_LINEWIDTH = 0.55


def _build_cmd_density_field(
    bg_x: np.ndarray,
    bg_y: np.ndarray,
) -> dict[str, object] | None:
    """Build a smoothed 2D number-density grid for filled CMD contours."""
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    if len(bg_x) == 0:
        return None

    xrange = (CMD_XLIM[0], CMD_XLIM[1])
    yrange = (CMD_YLIM[0], CMD_YLIM[1])
    hist, xedges, yedges = np.histogram2d(
        bg_x,
        bg_y,
        bins=CMD_DENSITY_BINS,
        range=[xrange, yrange],
    )
    density = gaussian_filter(hist.T.astype(float), sigma=CMD_DENSITY_SMOOTH_SIGMA)
    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size == 0:
        return None

    cutoff = max(
        float(np.nanmax(positive)) * CMD_DENSITY_CUTOFF_FRAC_MAX,
        float(np.nanpercentile(positive, CMD_DENSITY_CUTOFF_PERCENTILE)),
    )
    vmin = max(cutoff, float(np.nanpercentile(positive, 12)))
    vmax = float(np.nanmax(positive))
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    return {
        "density": density,
        "xedges": xedges,
        "yedges": yedges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "cutoff": cutoff,
        "levels": np.geomspace(vmin, vmax, CMD_DENSITY_CONTOUR_LEVELS),
        "norm": LogNorm(vmin=vmin, vmax=vmax),
    }


def _local_density_at_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    density: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    if len(x) == 0:
        return np.empty(0, dtype=float)

    interp = RegularGridInterpolator(
        (y_centers, x_centers),
        density,
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(np.column_stack([y, x]))


def _upsample_density_grid(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    density: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import zoom

    if factor <= 1:
        return x_centers, y_centers, density

    upsampled = zoom(density, factor, order=3)
    x_hi = np.linspace(x_centers[0], x_centers[-1], upsampled.shape[1])
    y_hi = np.linspace(y_centers[0], y_centers[-1], upsampled.shape[0])
    return x_hi, y_hi, upsampled


def _plot_cmd_background(ax, bg_x: np.ndarray, bg_y: np.ndarray) -> None:
    """Sparse CMD background: per-point scatter below density cutoff, contourf above."""
    if len(bg_x) == 0:
        return

    density_field = _build_cmd_density_field(bg_x, bg_y)
    if density_field is None:
        ax.scatter(
            bg_x,
            bg_y,
            s=CMD_BG_SCATTER_SIZE,
            c=plt.get_cmap(CMD_DENSITY_CMAP)(0.15),
            alpha=CMD_BG_SCATTER_ALPHA,
            edgecolors="none",
            linewidths=0.0,
            zorder=0,
        )
        return

    density = density_field["density"]
    cutoff = float(density_field["cutoff"])
    norm = density_field["norm"]
    cmap = plt.get_cmap(CMD_DENSITY_CMAP)
    local_density = _local_density_at_points(
        bg_x,
        bg_y,
        density=density,
        x_centers=density_field["x_centers"],
        y_centers=density_field["y_centers"],
    )

    sparse_mask = local_density < cutoff
    if sparse_mask.any():
        sparse_density = np.clip(local_density[sparse_mask], norm.vmin, norm.vmax)
        ax.scatter(
            bg_x[sparse_mask],
            bg_y[sparse_mask],
            s=CMD_BG_SCATTER_SIZE,
            c=cmap(norm(sparse_density)),
            alpha=CMD_BG_SCATTER_ALPHA,
            edgecolors="none",
            linewidths=0.0,
            zorder=0,
        )

    contour_x, contour_y, contour_density = _upsample_density_grid(
        density_field["x_centers"],
        density_field["y_centers"],
        density,
        CMD_DENSITY_UPSAMPLE_FACTOR,
    )
    density_masked = np.ma.masked_where(contour_density < cutoff, contour_density)
    ax.contourf(
        contour_x,
        contour_y,
        density_masked,
        levels=density_field["levels"],
        cmap=CMD_DENSITY_CMAP,
        norm=norm,
        extend="max",
        antialiased=True,
        corner_mask=True,
        zorder=1,
    )


def _plot_cmd_background_density(ax, bg_x: np.ndarray, bg_y: np.ndarray) -> None:
    """Full-field pcolormesh density map (viridis + LogNorm), reference-style."""
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    if len(bg_x) == 0:
        return

    xrange = (CMD_XLIM[0], CMD_XLIM[1])
    yrange = (CMD_YLIM[0], CMD_YLIM[1])
    hist, xedges, yedges = np.histogram2d(
        bg_x,
        bg_y,
        bins=CMD_DENSITY_STYLE_BINS,
        range=[xrange, yrange],
    )
    density = gaussian_filter(hist.T.astype(float), sigma=CMD_DENSITY_STYLE_SMOOTH_SIGMA)
    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size == 0:
        return

    cutoff = max(
        float(np.nanmax(positive)) * CMD_DENSITY_STYLE_CUTOFF_FRAC_MAX,
        float(np.nanpercentile(positive, CMD_DENSITY_STYLE_CUTOFF_PERCENTILE)),
    )
    vmin = max(cutoff, float(np.nanpercentile(positive, 5)))
    vmax = float(np.nanmax(positive))

    density_masked = np.ma.masked_where(density < cutoff, density)
    ax.pcolormesh(
        xedges,
        yedges,
        density_masked,
        cmap=CMD_DENSITY_STYLE_CMAP,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        rasterized=True,
        zorder=0,
    )


def _plot_cmd_background_corner(ax, bg_x: np.ndarray, bg_y: np.ndarray) -> None:
    """Monochrome scatter + line-density contours, corner-plot style."""
    from scipy.ndimage import gaussian_filter

    if len(bg_x) == 0:
        return

    ax.scatter(
        bg_x,
        bg_y,
        s=CMD_BG_SCATTER_SIZE,
        c=CMD_CORNER_STYLE_SCATTER_COLOR,
        alpha=CMD_CORNER_STYLE_SCATTER_ALPHA,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
        zorder=0,
    )

    xrange = (CMD_XLIM[0], CMD_XLIM[1])
    yrange = (CMD_YLIM[0], CMD_YLIM[1])
    hist, xedges, yedges = np.histogram2d(
        bg_x,
        bg_y,
        bins=CMD_CORNER_STYLE_BINS,
        range=[xrange, yrange],
    )
    density = gaussian_filter(hist.T.astype(float), sigma=CMD_CORNER_STYLE_SMOOTH_SIGMA)
    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size == 0:
        return

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    percentiles = np.linspace(55, 95, CMD_CORNER_STYLE_CONTOUR_LEVELS)
    levels = sorted({float(np.percentile(positive, p)) for p in percentiles})
    if not levels:
        return

    ax.contour(
        x_centers,
        y_centers,
        density,
        levels=levels,
        colors=CMD_CORNER_STYLE_CONTOUR_COLOR,
        linewidths=CMD_CORNER_STYLE_CONTOUR_LINEWIDTH,
        zorder=1,
    )


def _gaia_background_arrays_from_frame(
    df: pd.DataFrame,
    *,
    g_min: float | None = None,
    g_max: float | None = None,
    cmd_mode: str = "observed",
    dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if cmd_mode not in {"observed", "dereddened"}:
        raise ValueError(f"Gaia background cmd mode must be observed or dereddened, got {cmd_mode!r}")

    work = pd.DataFrame(index=df.index)
    work["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    work["phot_g_mean_mag"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
    work["parallax"] = pd.to_numeric(df["parallax"], errors="coerce")
    if "distance_gspphot" in df.columns:
        work["distance_gspphot"] = pd.to_numeric(df["distance_gspphot"], errors="coerce")
    if "ra" in df.columns:
        work["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    if "dec" in df.columns:
        work["dec"] = pd.to_numeric(df["dec"], errors="coerce")

    ok = (work["parallax"] > 0) & work["phot_g_mean_mag"].notna() & work["bp_rp"].notna()
    if g_min is not None:
        ok &= work["phot_g_mean_mag"] >= float(g_min)
    if g_max is not None:
        ok &= work["phot_g_mean_mag"] <= float(g_max)
    work = work.loc[ok].copy()
    if work.empty:
        return np.empty(0), np.empty(0), {"gaia_background_cmd_mode": cmd_mode}

    dist_pc = pd.Series(np.nan, index=work.index, dtype=float)
    if "distance_gspphot" in work.columns:
        gsp = work["distance_gspphot"]
        dist_pc = dist_pc.where(~((gsp > 0) & np.isfinite(gsp)), gsp)
    plx_dist_pc = 1000.0 / work["parallax"]
    dist_pc = dist_pc.where(np.isfinite(dist_pc) & (dist_pc > 0), plx_dist_pc)

    finite_dist = np.isfinite(dist_pc) & (dist_pc > 0)
    work = work.loc[finite_dist].copy()
    dist_pc = dist_pc.loc[finite_dist]
    if work.empty:
        return np.empty(0), np.empty(0), {"gaia_background_cmd_mode": cmd_mode}

    bp_rp = work["bp_rp"].to_numpy(dtype=float)
    g = work["phot_g_mean_mag"].to_numpy(dtype=float)
    dist = dist_pc.to_numpy(dtype=float)
    mg = g - 5.0 * np.log10(dist) + 5.0
    if cmd_mode == "observed":
        return bp_rp, mg, {"gaia_background_cmd_mode": "observed"}

    if "ra" not in work.columns or "dec" not in work.columns:
        raise ValueError("Dereddened Gaia background requires ra/dec columns; use --gaia-bg-source tap or a Gaia file with ra/dec")

    av = _query_gaia_background_dust_av(
        work["ra"].to_numpy(dtype=float),
        work["dec"].to_numpy(dtype=float),
        dist,
        chunk_size=dust_chunk_size,
    )
    av = np.where(np.isfinite(av) & (av >= 0), av, 0.0)
    bp_rp0 = bp_rp - CMD_E_BP_RP_PER_AV * av
    mg0 = mg - CMD_A_G_PER_AV * av
    finite_av = av[np.isfinite(av)]
    return bp_rp0, mg0, {
        "gaia_background_cmd_mode": "dereddened",
        "gaia_background_dust_points": int(len(av)),
        "gaia_background_mean_av": None if finite_av.size == 0 else float(np.mean(finite_av)),
        "gaia_background_median_av": None if finite_av.size == 0 else float(np.median(finite_av)),
    }


def _query_gaia_background_dust_av(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    dist_pc: np.ndarray,
    *,
    chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
) -> np.ndarray:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from dustmaps3d import dustmaps3d

    if chunk_size <= 0:
        raise ValueError("--gaia-bg-dust-chunk-size must be positive")

    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    dist = np.asarray(dist_pc, dtype=float)
    av = np.full(len(ra), np.nan, dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dist) & (dist > 0)
    if not valid.any():
        return av

    valid_idx = np.flatnonzero(valid)
    print(f"Dereddening Gaia background with dustmaps3d for {len(valid_idx):,} sources...")
    for start in range(0, len(valid_idx), int(chunk_size)):
        idx = valid_idx[start : start + int(chunk_size)]
        coords = SkyCoord(ra=ra[idx] * u.deg, dec=dec[idx] * u.deg, frame="icrs")
        galactic = coords.galactic
        ebv, _dust_density, _sigma, _max_dist = dustmaps3d(
            galactic.l.deg,
            galactic.b.deg,
            dist[idx] / 1000.0,
        )
        av[idx] = 3.1 * np.asarray(ebv, dtype=float)
        print(f"  dustmaps3d: {min(start + int(chunk_size), len(valid_idx)):,}/{len(valid_idx):,}")
    finite_av = av[np.isfinite(av)]
    if finite_av.size:
        print(f"Gaia background dust complete. Mean A_V={finite_av.mean():.3f}, median A_V={np.median(finite_av):.3f}")
    else:
        print("Gaia background dust complete. No finite A_V values returned.")
    return av


def _load_gaia_background_file(
    bg_path: Path | str = GAIA_BG_PATH,
    *,
    g_min: float | None = None,
    g_max: float | None = None,
    cmd_mode: str = "observed",
    dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load a local Gaia parquet sample for CMD density background."""
    import pyarrow.parquet as pq

    path = Path(bg_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        return np.empty(0), np.empty(0), {"gaia_background_cmd_mode": cmd_mode}
    wanted = ["bp_rp", "phot_g_mean_mag", "parallax"]
    if cmd_mode == "dereddened":
        wanted.extend(["ra", "dec", "distance_gspphot"])
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [col for col in wanted if col in available]
    missing = sorted(set(wanted[:3]) - available)
    if missing:
        raise ValueError(f"Gaia background file is missing required columns: {missing}")
    df = pd.read_parquet(path, columns=columns)
    return _gaia_background_arrays_from_frame(
        df,
        g_min=g_min,
        g_max=g_max,
        cmd_mode=cmd_mode,
        dust_chunk_size=dust_chunk_size,
    )


def _adql_float(value: float) -> str:
    return f"{float(value):.12g}"


def _query_gaia_background_tap(
    *,
    g_min: float,
    g_max: float,
    row_limit: int = DEFAULT_GAIA_TAP_LIMIT,
    ruwe_max: float | None = DEFAULT_GAIA_TAP_RUWE_MAX,
    table: str = DEFAULT_GAIA_TAP_TABLE,
    cmd_mode: str = "observed",
    dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Query Gaia DR3 live via TAP and return apparent-G-filtered CMD background arrays."""
    from astroquery.gaia import Gaia
    from astroquery.utils.tap.core import TapPlus

    if row_limit <= 0:
        raise ValueError("--gaia-tap-limit must be positive")

    conditions = [
        f"phot_g_mean_mag >= {_adql_float(g_min)}",
        f"phot_g_mean_mag <= {_adql_float(g_max)}",
        "phot_g_mean_mag IS NOT NULL",
        "bp_rp IS NOT NULL",
        "parallax IS NOT NULL",
        "parallax > 0",
    ]
    if ruwe_max is not None:
        conditions.extend(["ruwe IS NOT NULL", f"ruwe <= {_adql_float(ruwe_max)}"])

    query = f"""
SELECT TOP {int(row_limit)}
    source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, ruwe, distance_gspphot
FROM {table}
WHERE
    {" AND ".join(conditions)}
ORDER BY random_index
"""
    print("Querying Gaia TAP for background sample:")
    print(query.strip())
    Gaia.ROW_LIMIT = -1
    job = TapPlus.launch_job_async(
        Gaia,
        query=query,
        output_format="votable_gzip",
        dump_to_file=False,
        maxrec=int(row_limit),
    )
    df = job.get_results().to_pandas()
    if len(df) < row_limit:
        print(
            f"Gaia TAP returned {len(df):,} rows for requested limit {row_limit:,}; "
            "the archive may have fewer matching rows or may still be applying a service cap."
        )
    return _gaia_background_arrays_from_frame(
        df,
        g_min=g_min,
        g_max=g_max,
        cmd_mode=cmd_mode,
        dust_chunk_size=dust_chunk_size,
    )


def _load_gaia_background(
    bg_path: Path | str = GAIA_BG_PATH,
    *,
    source: str = DEFAULT_GAIA_BG_SOURCE,
    g_min: float | None = None,
    g_max: float | None = None,
    tap_limit: int = DEFAULT_GAIA_TAP_LIMIT,
    tap_ruwe_max: float | None = DEFAULT_GAIA_TAP_RUWE_MAX,
    tap_table: str = DEFAULT_GAIA_TAP_TABLE,
    cmd_mode: str = "observed",
    dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if source == "file":
        return _load_gaia_background_file(
            bg_path,
            g_min=g_min,
            g_max=g_max,
            cmd_mode=cmd_mode,
            dust_chunk_size=dust_chunk_size,
        )
    if source == "tap":
        if g_min is None or g_max is None:
            raise ValueError("--gaia-bg-source tap requires both --gaia-bg-g-min and --gaia-bg-g-max")
        return _query_gaia_background_tap(
            g_min=float(g_min),
            g_max=float(g_max),
            row_limit=int(tap_limit),
            ruwe_max=tap_ruwe_max,
            table=tap_table,
            cmd_mode=cmd_mode,
            dust_chunk_size=dust_chunk_size,
        )
    raise ValueError(f"gaia background source must be one of {GAIA_BG_SOURCE_CHOICES}, got {source!r}")


def _candidate_background_rows(
    plottable: pd.DataFrame,
    *,
    g_min: float | None = None,
    g_max: float | None = None,
) -> pd.DataFrame:
    """Candidate CMD rows for the background, optionally filtered by apparent Gaia G."""
    mask = pd.Series(True, index=plottable.index)
    if g_min is not None or g_max is not None:
        g = pd.to_numeric(plottable["phot_g_mean_mag"], errors="coerce")
        if g_min is not None:
            mask &= g >= float(g_min)
        if g_max is not None:
            mask &= g <= float(g_max)
    return plottable[mask].copy()


def _build_cmd_background_arrays(
    plottable: pd.DataFrame,
    *,
    background_sample: str = DEFAULT_BACKGROUND_SAMPLE,
    bg_path: Path | str = GAIA_BG_PATH,
    gaia_bg_source: str = DEFAULT_GAIA_BG_SOURCE,
    gaia_bg_cmd_mode: str = DEFAULT_GAIA_BG_CMD_MODE,
    gaia_bg_g_min: float | None = None,
    gaia_bg_g_max: float | None = None,
    gaia_tap_limit: int = DEFAULT_GAIA_TAP_LIMIT,
    gaia_tap_ruwe_max: float | None = DEFAULT_GAIA_TAP_RUWE_MAX,
    gaia_tap_table: str = DEFAULT_GAIA_TAP_TABLE,
    gaia_dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
    candidate_bg_g_min: float | None = None,
    candidate_bg_g_max: float | None = None,
    plot_cmd_mode: str = DEFAULT_CMD_MODE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if background_sample not in BACKGROUND_SAMPLE_CHOICES:
        raise ValueError(f"background_sample must be one of {BACKGROUND_SAMPLE_CHOICES}, got {background_sample!r}")
    if gaia_bg_cmd_mode not in GAIA_BG_CMD_MODE_CHOICES:
        raise ValueError(f"gaia_bg_cmd_mode must be one of {GAIA_BG_CMD_MODE_CHOICES}, got {gaia_bg_cmd_mode!r}")

    uses_gaia = background_sample in {"gaia", "gaia+candidates"}
    uses_candidates = background_sample in {"candidates", "gaia+candidates"}
    resolved_gaia_cmd_mode = plot_cmd_mode if gaia_bg_cmd_mode == "match" else gaia_bg_cmd_mode
    if uses_gaia:
        gaia_x, gaia_y, gaia_meta = _load_gaia_background(
            bg_path,
            source=gaia_bg_source,
            g_min=gaia_bg_g_min,
            g_max=gaia_bg_g_max,
            tap_limit=gaia_tap_limit,
            tap_ruwe_max=gaia_tap_ruwe_max,
            tap_table=gaia_tap_table,
            cmd_mode=resolved_gaia_cmd_mode,
            dust_chunk_size=gaia_dust_chunk_size,
        )
    else:
        gaia_x = np.empty(0)
        gaia_y = np.empty(0)
        gaia_meta = {"gaia_background_cmd_mode": resolved_gaia_cmd_mode}
    candidate_bg = _candidate_background_rows(
        plottable,
        g_min=candidate_bg_g_min,
        g_max=candidate_bg_g_max,
    )
    cand_x = candidate_bg["cmd_plot_color"].to_numpy(dtype=float)
    cand_y = candidate_bg["cmd_plot_mag"].to_numpy(dtype=float)

    if background_sample == "gaia":
        bg_x = gaia_x
        bg_y = gaia_y
    elif background_sample == "candidates":
        bg_x = cand_x
        bg_y = cand_y
    else:
        bg_x = np.concatenate([gaia_x, cand_x])
        bg_y = np.concatenate([gaia_y, cand_y])

    finite_mask = np.isfinite(bg_x) & np.isfinite(bg_y)
    bg_x = bg_x[finite_mask]
    bg_y = bg_y[finite_mask]
    counts = {
        "background_sample": background_sample,
        "gaia_background_source": gaia_bg_source,
        "gaia_background_cmd_mode_requested": gaia_bg_cmd_mode,
        "gaia_background_cmd_mode": resolved_gaia_cmd_mode,
        "gaia_bg_g_min": None if gaia_bg_g_min is None else float(gaia_bg_g_min),
        "gaia_bg_g_max": None if gaia_bg_g_max is None else float(gaia_bg_g_max),
        "gaia_tap_limit": int(gaia_tap_limit) if gaia_bg_source == "tap" else None,
        "gaia_tap_ruwe_max": (
            None
            if gaia_bg_source != "tap" or gaia_tap_ruwe_max is None
            else float(gaia_tap_ruwe_max)
        ),
        "candidate_bg_g_min": None if candidate_bg_g_min is None else float(candidate_bg_g_min),
        "candidate_bg_g_max": None if candidate_bg_g_max is None else float(candidate_bg_g_max),
        "gaia_background_pool_points": int(len(gaia_x)) if uses_gaia else None,
        "candidate_background_pool_points": int(len(candidate_bg)) if uses_candidates else None,
        "gaia_background_points": int(len(gaia_x)) if uses_gaia else 0,
        "candidate_background_points": int(len(candidate_bg)) if uses_candidates else 0,
        "background_points": int(len(bg_x)),
    }
    counts.update(gaia_meta)
    return bg_x, bg_y, counts


def _resolve_plot_ages(grid: pd.DataFrame, ages_myr: tuple[float, ...] | list[float]) -> list[float]:
    available = np.array(sorted(pd.to_numeric(grid["mist_age_myr"], errors="coerce").dropna().unique()), dtype=float)
    resolved: list[float] = []
    for requested in ages_myr:
        if available.size == 0:
            break
        idx = int(np.nanargmin(np.abs(available - float(requested))))
        age = float(available[idx])
        if age not in resolved:
            resolved.append(age)
    return resolved


def _visible_cmd_rows(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    return df[
        x.between(CMD_XLIM[0], CMD_XLIM[1])
        & y.between(CMD_YLIM[0], CMD_YLIM[1])
    ].copy()


def _plot_mist_isochrones(
    ax,
    mist_grid: pd.DataFrame,
    *,
    ages_myr: tuple[float, ...] | list[float],
) -> None:
    grid = normalize_mist_cmd_grid(mist_grid)
    if grid.empty:
        return
    ages = _resolve_plot_ages(grid, ages_myr)
    for i, age in enumerate(ages):
        iso = grid[np.isclose(grid["mist_age_myr"], age, rtol=5e-3, atol=1e-6)].sort_values("mist_initial_mass")
        if iso.empty:
            continue
        color = ISOCHRONE_COLORS[i % len(ISOCHRONE_COLORS)]
        ax.plot(
            iso["mist_gaia_bp_rp"],
            iso["mist_gaia_g"],
            color=color,
            linestyle="--",
            linewidth=ISOCHRONE_LINEWIDTH,
            alpha=ISOCHRONE_ALPHA,
            zorder=2.1,
        )
        visible = _visible_cmd_rows(iso, "mist_gaia_bp_rp", "mist_gaia_g")
        if not visible.empty:
            label_row = visible.iloc[min(len(visible) - 1, max(0, int(0.8 * len(visible))))]
            ax.text(
                label_row["mist_gaia_bp_rp"],
                label_row["mist_gaia_g"],
                f"{age:g} Myr",
                color=color,
                fontsize=MASS_LABEL_FONTSIZE,
                ha="left",
                va="center",
                clip_on=True,
                zorder=2.2,
            )


def _plot_mist_mass_tracks(
    ax,
    mist_grid: pd.DataFrame,
    *,
    masses: tuple[float, ...] | list[float],
    ages_myr: tuple[float, ...] | list[float],
) -> None:
    tracks = mist_mass_tracks(mist_grid, masses, ages_myr=ages_myr)
    if tracks.empty:
        return
    for mass in masses:
        track = tracks[np.isclose(pd.to_numeric(tracks["mass"], errors="coerce"), float(mass))]
        track = track.sort_values("age_myr")
        if track.empty:
            continue
        ax.plot(
            track["gaia_bp_rp"],
            track["gaia_g"],
            color=MASS_TRACK_COLOR,
            alpha=MASS_TRACK_ALPHA,
            linewidth=MASS_TRACK_LINEWIDTH,
            zorder=2.0,
        )
        visible = _visible_cmd_rows(track, "gaia_bp_rp", "gaia_g")
        if visible.empty:
            continue
        label_row = visible.iloc[-1]
        ax.text(
            label_row["gaia_bp_rp"],
            label_row["gaia_g"],
            rf"{mass:g}$M_\odot$",
            color=MASS_TRACK_COLOR,
            fontsize=MASS_LABEL_FONTSIZE,
            ha="left",
            va="center",
            alpha=0.86,
            clip_on=True,
            zorder=2.2,
        )


def _plot_cmd(
    plot_df: pd.DataFrame,
    out_path: Path,
    *,
    bucket_order: list[str] | None = None,
    background_style: str = DEFAULT_BACKGROUND_STYLE,
    background_sample: str = DEFAULT_BACKGROUND_SAMPLE,
    bg_path: Path | str = GAIA_BG_PATH,
    gaia_bg_source: str = DEFAULT_GAIA_BG_SOURCE,
    gaia_bg_cmd_mode: str = DEFAULT_GAIA_BG_CMD_MODE,
    gaia_bg_g_min: float | None = None,
    gaia_bg_g_max: float | None = None,
    gaia_tap_limit: int = DEFAULT_GAIA_TAP_LIMIT,
    gaia_tap_ruwe_max: float | None = DEFAULT_GAIA_TAP_RUWE_MAX,
    gaia_tap_table: str = DEFAULT_GAIA_TAP_TABLE,
    gaia_dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
    candidate_bg_g_min: float | None = None,
    candidate_bg_g_max: float | None = None,
    cmd_mode: str = DEFAULT_CMD_MODE,
    mist_grid: pd.DataFrame | None = None,
    with_isochrones: bool = False,
    isochrone_ages_myr: tuple[float, ...] | list[float] = DEFAULT_ISOCHRONE_AGES_MYR,
    mass_labels: tuple[float, ...] | list[float] = DEFAULT_MASS_LABELS,
    show_errorbars: bool = False,
) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_order = bucket_order or BUCKET_ORDER
    plottable = plot_df[_plottable_mask(plot_df)].copy()
    selected = plottable[plottable["review_cmd_bucket"].isin(bucket_order)].copy()

    fig, ax = plt.subplots(figsize=CMD_FIGSIZE)

    bg_x, bg_y, background_counts = _build_cmd_background_arrays(
        plottable,
        background_sample=background_sample,
        bg_path=bg_path,
        gaia_bg_source=gaia_bg_source,
        gaia_bg_cmd_mode=gaia_bg_cmd_mode,
        gaia_bg_g_min=gaia_bg_g_min,
        gaia_bg_g_max=gaia_bg_g_max,
        gaia_tap_limit=gaia_tap_limit,
        gaia_tap_ruwe_max=gaia_tap_ruwe_max,
        gaia_tap_table=gaia_tap_table,
        gaia_dust_chunk_size=gaia_dust_chunk_size,
        candidate_bg_g_min=candidate_bg_g_min,
        candidate_bg_g_max=candidate_bg_g_max,
        plot_cmd_mode=cmd_mode,
    )

    if background_style == "density":
        _plot_cmd_background_density(ax, bg_x, bg_y)
    elif background_style == "corner":
        _plot_cmd_background_corner(ax, bg_x, bg_y)
    else:
        _plot_cmd_background(ax, bg_x, bg_y)

    ax.set_xlim(*CMD_XLIM)
    ax.set_ylim(*CMD_YLIM)
    ax.invert_yaxis()

    if with_isochrones and mist_grid is not None:
        _plot_mist_mass_tracks(ax, mist_grid, masses=mass_labels, ages_myr=isochrone_ages_myr)
        _plot_mist_isochrones(ax, mist_grid, ages_myr=isochrone_ages_myr)

    for bucket in bucket_order:
        sub = selected[selected["review_cmd_bucket"] == bucket]
        if sub.empty:
            continue
        style = CMD_REVIEW_BUCKET_STYLE[bucket]
        sub_solid = sub[sub["cmd_coordinate_source"].isin(SOLID_CMD_SOURCES)]
        sub_hollow = sub[sub["cmd_coordinate_source"].isin(HOLLOW_CMD_SOURCES)]
        marker_size = float(style["size"]) * CMD_MARKER_SIZE_SCALE
        solid_label = f"{bucket} ({len(sub):,})" if sub_hollow.empty else f"{bucket} solid ({len(sub_solid):,})"
        _scatter_solid(
            ax,
            sub_solid,
            color=style["color"],
            size=marker_size,
            marker=style["marker"],
            zorder=style["zorder"],
            label=solid_label if not sub_solid.empty else None,
            show_errorbars=show_errorbars,
        )
        _scatter_hollow(
            ax,
            sub_hollow,
            color=style["color"],
            size=marker_size,
            marker=style["marker"],
            zorder=style["zorder"],
            label=f"{bucket} observed ({len(sub_hollow):,})" if not sub_hollow.empty else None,
            show_errorbars=show_errorbars,
        )

    if cmd_mode == "dereddened":
        ax.set_xlabel(r"$(G_{\mathrm{BP}} - G_{\mathrm{RP}})_0$ [mag]", fontsize=CMD_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$M_{G,0}$ [mag]", fontsize=CMD_AXIS_LABEL_FONTSIZE)
    else:
        ax.set_xlabel(r"$G_{\mathrm{BP}} - G_{\mathrm{RP}}$ [mag]", fontsize=CMD_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$M_G$ [mag]", fontsize=CMD_AXIS_LABEL_FONTSIZE)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        labelsize=CMD_TICK_LABEL_FONTSIZE,
        length=CMD_TICK_LENGTH,
        width=CMD_TICK_WIDTH,
    )
    ax.minorticks_on()
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        framealpha=0.92,
        edgecolor="black",
        fontsize=CMD_LEGEND_FONTSIZE,
        markerscale=CMD_LEGEND_MARKERSCALE,
    )
    save_publication_figure(fig, out_path, dpi=220, close=False, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    plt.close(fig)
    return background_counts


def build_plot(
    review_db: Path,
    candidates_path: Path,
    output_path: Path,
    *,
    buckets: list[str] | None = None,
    title: str | None = None,
    only_reviewed: bool = True,
    background_style: str = DEFAULT_BACKGROUND_STYLE,
    background_sample: str = DEFAULT_BACKGROUND_SAMPLE,
    bg_path: Path | str = GAIA_BG_PATH,
    gaia_bg_source: str = DEFAULT_GAIA_BG_SOURCE,
    gaia_bg_cmd_mode: str = DEFAULT_GAIA_BG_CMD_MODE,
    gaia_bg_g_min: float | None = None,
    gaia_bg_g_max: float | None = None,
    gaia_tap_limit: int = DEFAULT_GAIA_TAP_LIMIT,
    gaia_tap_ruwe_max: float | None = DEFAULT_GAIA_TAP_RUWE_MAX,
    gaia_tap_table: str = DEFAULT_GAIA_TAP_TABLE,
    gaia_dust_chunk_size: int = DEFAULT_GAIA_DUST_CHUNK_SIZE,
    candidate_bg_g_min: float | None = None,
    candidate_bg_g_max: float | None = None,
    cmd_mode: str = DEFAULT_CMD_MODE,
    with_isochrones: bool = False,
    isochrone_grid: Path | str = MIST_GRID_PATH,
    isochrone_ages_myr: tuple[float, ...] | list[float] = DEFAULT_ISOCHRONE_AGES_MYR,
    mass_labels: tuple[float, ...] | list[float] = DEFAULT_MASS_LABELS,
    show_errorbars: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    bucket_order = buckets or BUCKET_ORDER
    reviews = _read_reviews(review_db, only_reviewed=only_reviewed)
    candidates = pd.read_parquet(candidates_path)
    cmd = _compute_cmd_coordinates(candidates)
    plot_df = cmd.merge(reviews, on="candidate_id", how="left")
    plot_df["review_cmd_bucket"] = plot_df.apply(_assign_bucket, axis=1)
    plot_df = _apply_cmd_mode(plot_df, cmd_mode)
    show_errorbars = with_isochrones if show_errorbars is None else show_errorbars

    mist_grid = None
    if with_isochrones:
        mist_grid = load_mist_grid(isochrone_grid)
        plot_df = estimate_cmd_masses(
            plot_df,
            mist_grid,
            color_col="cmd_plot_color",
            mag_col="cmd_plot_mag",
            color_err_col="cmd_plot_color_err",
            mag_err_col="cmd_plot_mag_err",
            ages_myr=isochrone_ages_myr,
        )

    plottable_mask = _plottable_mask(plot_df)
    selected_mask = plot_df["review_cmd_bucket"].isin(bucket_order)
    plotted_mask = selected_mask & plottable_mask
    solid_mask = plottable_mask & plot_df["cmd_coordinate_source"].isin(SOLID_CMD_SOURCES)
    hollow_mask = plottable_mask & plot_df["cmd_coordinate_source"].isin(HOLLOW_CMD_SOURCES)
    counts = {
        "all_candidates": int(len(plot_df)),
        "all_cmd_points_solid": int(solid_mask.sum()),
        "all_cmd_points_hollow": int(hollow_mask.sum()),
        "all_cmd_points": int(plottable_mask.sum()),
        "selected_candidates": int(selected_mask.sum()),
        "selected_cmd_points": int(plotted_mask.sum()),
    }
    for bucket in bucket_order:
        bucket_mask = plot_df["review_cmd_bucket"].eq(bucket)
        counts[f"{bucket}_selected"] = int(bucket_mask.sum())
        counts[f"{bucket}_plotted"] = int((bucket_mask & plottable_mask).sum())
        counts[f"{bucket}_plotted_solid"] = int(
            (bucket_mask & plot_df["cmd_coordinate_source"].isin(SOLID_CMD_SOURCES) & plottable_mask).sum()
        )
        counts[f"{bucket}_plotted_hollow"] = int(
            (bucket_mask & plot_df["cmd_coordinate_source"].isin(HOLLOW_CMD_SOURCES) & plottable_mask).sum()
        )

    if with_isochrones:
        mass_mask = selected_mask & plottable_mask & plot_df["cmd_mass_best"].notna()
        counts["selected_mass_estimates"] = int(mass_mask.sum())

    background_counts = _plot_cmd(
        plot_df,
        output_path,
        bucket_order=bucket_order,
        background_style=background_style,
        background_sample=background_sample,
        bg_path=bg_path,
        gaia_bg_source=gaia_bg_source,
        gaia_bg_cmd_mode=gaia_bg_cmd_mode,
        gaia_bg_g_min=gaia_bg_g_min,
        gaia_bg_g_max=gaia_bg_g_max,
        gaia_tap_limit=gaia_tap_limit,
        gaia_tap_ruwe_max=gaia_tap_ruwe_max,
        gaia_tap_table=gaia_tap_table,
        gaia_dust_chunk_size=gaia_dust_chunk_size,
        candidate_bg_g_min=candidate_bg_g_min,
        candidate_bg_g_max=candidate_bg_g_max,
        cmd_mode=cmd_mode,
        mist_grid=mist_grid,
        with_isochrones=with_isochrones,
        isochrone_ages_myr=isochrone_ages_myr,
        mass_labels=mass_labels,
        show_errorbars=bool(show_errorbars),
    )
    counts.update(background_counts)
    csv_path = output_path.with_suffix(".csv")
    plot_df.loc[selected_mask].sort_values(["review_cmd_bucket", "candidate_id"]).to_csv(csv_path, index=False)
    return plot_df, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default=DEFAULT_PRESET,
        help=(
            "Path/bucket preset to use when --review-db/--candidates/--output/--buckets "
            "are omitted. Default: july1-dippers."
        ),
    )
    parser.add_argument("--review-db", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--buckets",
        nargs="+",
        choices=BUCKET_ORDER,
        default=None,
        help="Restrict plot/CSV to these review buckets (default: preset buckets).",
    )
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument(
        "--background-style",
        choices=("hybrid", "density", "corner"),
        default=DEFAULT_BACKGROUND_STYLE,
        help=(
            "Background rendering: 'density' (pcolormesh viridis, default), "
            "'hybrid' (scatter+contourf magma), or 'corner' (grey scatter+line contours)."
        ),
    )
    parser.add_argument(
        "--background-sample",
        choices=BACKGROUND_SAMPLE_CHOICES,
        default=DEFAULT_BACKGROUND_SAMPLE,
        help=(
            "CMD background source: Gaia field sample, MALCA candidate CMD points, "
            "or both combined. Default preserves the previous Gaia+candidate background."
        ),
    )
    parser.add_argument(
        "--candidate-bg-g-min",
        type=float,
        default=None,
        help="Minimum apparent Gaia G magnitude for candidate background points only.",
    )
    parser.add_argument(
        "--candidate-bg-g-max",
        type=float,
        default=None,
        help="Maximum apparent Gaia G magnitude for candidate background points only.",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include non-reviewed rows from the reviews table.",
    )
    parser.add_argument(
        "--gaia-bg",
        type=Path,
        default=GAIA_BG_PATH,
        help="Path to the Gaia parquet file to use for the background when --gaia-bg-source=file.",
    )
    parser.add_argument(
        "--gaia-bg-source",
        choices=GAIA_BG_SOURCE_CHOICES,
        default=DEFAULT_GAIA_BG_SOURCE,
        help="Use a local Gaia parquet file or query Gaia DR3 live via TAP for the background.",
    )
    parser.add_argument(
        "--gaia-bg-cmd-mode",
        choices=GAIA_BG_CMD_MODE_CHOICES,
        default=DEFAULT_GAIA_BG_CMD_MODE,
        help=(
            "CMD coordinates for Gaia background. 'match' uses --cmd-mode, so "
            "a dereddened candidate overlay gets a dereddened Gaia background too."
        ),
    )
    parser.add_argument(
        "--gaia-bg-g-min",
        type=float,
        default=None,
        help="Minimum apparent Gaia G magnitude for Gaia field background points only.",
    )
    parser.add_argument(
        "--gaia-bg-g-max",
        type=float,
        default=None,
        help="Maximum apparent Gaia G magnitude for Gaia field background points only.",
    )
    parser.add_argument(
        "--gaia-tap-limit",
        type=int,
        default=DEFAULT_GAIA_TAP_LIMIT,
        help="Maximum rows to request from Gaia TAP when --gaia-bg-source=tap.",
    )
    parser.add_argument(
        "--gaia-tap-ruwe-max",
        type=float,
        default=DEFAULT_GAIA_TAP_RUWE_MAX,
        help="Maximum RUWE for Gaia TAP background rows. Default follows the all-sky helper script.",
    )
    parser.add_argument(
        "--gaia-tap-no-ruwe-cut",
        action="store_true",
        help="Do not apply a RUWE cut to Gaia TAP background rows.",
    )
    parser.add_argument(
        "--gaia-bg-dust-chunk-size",
        type=int,
        default=DEFAULT_GAIA_DUST_CHUNK_SIZE,
        help="Number of Gaia background rows per dustmaps3d dereddening chunk.",
    )
    parser.add_argument(
        "--gaia-tap-table",
        default=DEFAULT_GAIA_TAP_TABLE,
        help="Gaia TAP table to query for live background rows.",
    )
    parser.add_argument(
        "--cmd-mode",
        choices=("dereddened", "observed"),
        default=DEFAULT_CMD_MODE,
        help=(
            "CMD coordinates for the selected candidates. Use 'dereddened' for "
            "MIST comparison and mass estimates; use 'observed' for raw Gaia CMD axes."
        ),
    )
    parser.add_argument(
        "--with-isochrones",
        action="store_true",
        help="Overlay MIST Gaia isochrones/mass tracks and write nearest-track mass estimates.",
    )
    parser.add_argument(
        "--isochrone-grid",
        type=Path,
        default=MIST_GRID_PATH,
        help="Path to the compact MIST Gaia isochrone grid.",
    )
    parser.add_argument(
        "--isochrone-ages-myr",
        nargs="+",
        type=float,
        default=list(DEFAULT_ISOCHRONE_AGES_MYR),
        help="MIST isochrone ages to plot and use for mass estimates, in Myr.",
    )
    parser.add_argument(
        "--mass-labels",
        nargs="+",
        type=float,
        default=list(DEFAULT_MASS_LABELS),
        help="Fixed masses to draw as tracks through the requested isochrone ages.",
    )
    errorbar_group = parser.add_mutually_exclusive_group()
    errorbar_group.add_argument(
        "--show-errorbars",
        dest="show_errorbars",
        action="store_true",
        default=None,
        help="Draw propagated CMD uncertainty bars on candidate points.",
    )
    errorbar_group.add_argument(
        "--hide-errorbars",
        dest="show_errorbars",
        action="store_false",
        help="Suppress CMD uncertainty bars, even when plotting isochrones.",
    )
    return parser.parse_args()


def _apply_output_suffix(output: Path, background_style: str) -> Path:
    """Insert style suffix before the extension for non-default background styles."""
    if background_style == DEFAULT_BACKGROUND_STYLE:
        return output
    stem = output.stem
    if not stem.endswith(f"_{background_style}"):
        return output.with_name(f"{stem}_{background_style}{output.suffix}")
    return output


def _format_cli_float(value: float | None) -> str:
    return "None" if value is None else f"{float(value):g}"


def main() -> None:
    args = parse_args()
    if (
        args.candidate_bg_g_min is not None
        and args.candidate_bg_g_max is not None
        and args.candidate_bg_g_min > args.candidate_bg_g_max
    ):
        raise SystemExit("--candidate-bg-g-min must be <= --candidate-bg-g-max")
    if (
        args.gaia_bg_g_min is not None
        and args.gaia_bg_g_max is not None
        and args.gaia_bg_g_min > args.gaia_bg_g_max
    ):
        raise SystemExit("--gaia-bg-g-min must be <= --gaia-bg-g-max")
    if args.gaia_bg_dust_chunk_size <= 0:
        raise SystemExit("--gaia-bg-dust-chunk-size must be positive")
    uses_gaia_background = args.background_sample in {"gaia", "gaia+candidates"}
    if args.gaia_bg_source == "tap" and uses_gaia_background:
        if args.gaia_bg_g_min is None or args.gaia_bg_g_max is None:
            raise SystemExit("--gaia-bg-source tap requires both --gaia-bg-g-min and --gaia-bg-g-max")
        if args.gaia_tap_limit <= 0:
            raise SystemExit("--gaia-tap-limit must be positive")
    gaia_tap_ruwe_max = None if args.gaia_tap_no_ruwe_cut else args.gaia_tap_ruwe_max
    preset = PRESETS[args.preset]
    review_db = args.review_db or preset["review_db"]
    candidates = args.candidates or preset["candidates"]
    buckets = args.buckets if args.buckets is not None else list(preset["buckets"])
    output = _apply_output_suffix(args.output or preset["output"], args.background_style)
    print(f"Preset: {args.preset}")
    print(f"Review DB: {review_db}")
    print(f"Candidates: {candidates}")
    print(f"Buckets: {', '.join(buckets)}")
    print(f"Background sample: {args.background_sample}")
    print(f"Gaia background source: {args.gaia_bg_source}")
    print(f"Gaia background CMD mode: {args.gaia_bg_cmd_mode}")
    if args.gaia_bg_g_min is not None or args.gaia_bg_g_max is not None:
        print(
            "Gaia background G range: "
            f"{_format_cli_float(args.gaia_bg_g_min)}-{_format_cli_float(args.gaia_bg_g_max)}"
        )
    if args.gaia_bg_source == "tap":
        print(f"Gaia TAP row limit: {args.gaia_tap_limit}")
        print(f"Gaia TAP RUWE max: {_format_cli_float(gaia_tap_ruwe_max)}")
    if args.candidate_bg_g_min is not None or args.candidate_bg_g_max is not None:
        print(
            "candidate background G range: "
            f"{_format_cli_float(args.candidate_bg_g_min)}-{_format_cli_float(args.candidate_bg_g_max)}"
        )
    _plot_df, counts = build_plot(
        review_db,
        candidates,
        output,
        buckets=buckets,
        title=args.title,
        only_reviewed=not args.include_unreviewed,
        background_style=args.background_style,
        background_sample=args.background_sample,
        bg_path=args.gaia_bg,
        gaia_bg_source=args.gaia_bg_source,
        gaia_bg_cmd_mode=args.gaia_bg_cmd_mode,
        gaia_bg_g_min=args.gaia_bg_g_min,
        gaia_bg_g_max=args.gaia_bg_g_max,
        gaia_tap_limit=args.gaia_tap_limit,
        gaia_tap_ruwe_max=gaia_tap_ruwe_max,
        gaia_tap_table=args.gaia_tap_table,
        gaia_dust_chunk_size=args.gaia_bg_dust_chunk_size,
        candidate_bg_g_min=args.candidate_bg_g_min,
        candidate_bg_g_max=args.candidate_bg_g_max,
        cmd_mode=args.cmd_mode,
        with_isochrones=args.with_isochrones,
        isochrone_grid=args.isochrone_grid,
        isochrone_ages_myr=args.isochrone_ages_myr,
        mass_labels=args.mass_labels,
        show_errorbars=args.show_errorbars,
    )
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")
    print(f"Wrote {output.with_suffix('.csv')}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
