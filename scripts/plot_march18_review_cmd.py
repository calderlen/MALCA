#!/usr/bin/env python
"""Plot a Gaia CMD for selected March 18 review buckets."""
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

from malca.lightcurve_publication import (
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

from malca.config import DEFAULT_OUTPUT_DIR
from malca.ltv.cmd import dustmaps_cmd_from_fields


MARCH18_RUN = DEFAULT_OUTPUT_DIR / "runs" / "runs_march18_bundle_all"
DEFAULT_REVIEW_DB = MARCH18_RUN / "review" / "review.taxonomy_filled.db"
DEFAULT_CANDIDATES = MARCH18_RUN / "results" / "lc_events_classified.parquet"
DEFAULT_OUTPUT = MARCH18_RUN / "results" / "march18_review_cmd_selected.png"
CMD_FIGSIZE = FIG_SINGLE_COL_SQUARE
CMD_XLIM = (-1.0, 3.0)
CMD_YLIM = (-5.0, 10.0)
SOLID_CMD_SOURCES = frozenset({"dustmaps3d", "observed_no_extinction"})
HOLLOW_CMD_SOURCES = frozenset({"observed_fallback"})
PLOTTABLE_CMD_SOURCES = SOLID_CMD_SOURCES | HOLLOW_CMD_SOURCES

BUCKET_ORDER = ["Dipper", "Interesting", "LTV", "Microlensing", "Eclipsing binary", "Unknown"]


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
                interest_score,
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
    for col in (
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "distance_gspphot",
        "parallax",
        "A_v_3d",
        "source_id",
        "gaia_id",
    ):
        if col in external.columns:
            out[col] = external[col]
        else:
            out[col] = np.nan

    cmd_rows: list[dict[str, object]] = []
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

    cmd_df = pd.DataFrame(cmd_rows, index=out.index)
    out = pd.concat([out, cmd_df], axis=1)
    return out


def _assign_bucket(row: pd.Series) -> str | None:
    event_class = str(row.get("event_class") or "").strip().lower()
    morph_secondary = str(row.get("morphology_secondary") or "").strip().lower()
    physical_primary = str(row.get("physical_primary") or "").strip().lower()

    if event_class == "dipper":
        return "Dipper"
    if event_class == "unknown_interesting":
        return "Interesting"
    if event_class == "ltv":
        return "LTV"
    if event_class == "microlensing":
        return "Microlensing"
    if event_class == "periodic" or morph_secondary in {"detached_binary_like", "eclipsing_like"}:
        return "Eclipsing binary"
    if physical_primary == "unknown" or event_class in {"unknown", "unclear"}:
        return "Unknown"
    return None


def _plottable_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["cmd_coordinate_source"].isin(PLOTTABLE_CMD_SOURCES)
        & df["cmd_color"].notna()
        & df["cmd_mag"].notna()
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
) -> None:
    if df.empty:
        return
    ax.scatter(
        df["cmd_color"],
        df["cmd_mag"],
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
) -> None:
    if df.empty:
        return
    ax.scatter(
        df["cmd_color"],
        df["cmd_mag"],
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
CMD_MARKER_SIZE_SCALE = 1.25
GAIA_BG_PATH = Path("input/gaia/gaia_dr3_crossmatched.parquet")

# "density" background style: full-field pcolormesh, viridis, lighter smoothing.
CMD_DENSITY_STYLE_CMAP = "viridis"
CMD_DENSITY_STYLE_BINS = 120
CMD_DENSITY_STYLE_SMOOTH_SIGMA = 1.0
CMD_DENSITY_STYLE_CUTOFF_PERCENTILE = 8.0
CMD_DENSITY_STYLE_CUTOFF_FRAC_MAX = 0.012


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


def _load_gaia_background() -> tuple[np.ndarray, np.ndarray]:
    """Load the full Gaia DR3 crossmatched sample for CMD density background."""
    import pandas as pd
    path = Path(__file__).resolve().parents[1] / GAIA_BG_PATH
    if not path.is_file():
        return np.empty(0), np.empty(0)
    df = pd.read_parquet(path, columns=["bp_rp", "phot_g_mean_mag", "parallax"])
    bp_rp = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(dtype=float)
    ok = (plx > 0) & np.isfinite(g) & np.isfinite(bp_rp)
    mg = g[ok] + 5 * np.log10(plx[ok] / 1000.0) + 5.0
    return bp_rp[ok], mg


def _plot_cmd(
    plot_df: pd.DataFrame,
    out_path: Path,
    *,
    bucket_order: list[str] | None = None,
    background_style: str = "hybrid",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_order = bucket_order or BUCKET_ORDER
    plottable = plot_df[_plottable_mask(plot_df)].copy()
    selected = plottable[plottable["review_cmd_bucket"].isin(bucket_order)].copy()

    fig, ax = plt.subplots(figsize=CMD_FIGSIZE)

    # Combine Gaia DR3 background + all pipeline candidates
    gaia_x, gaia_y = _load_gaia_background()
    cand_x = plottable["cmd_color"].to_numpy(dtype=float)
    cand_y = plottable["cmd_mag"].to_numpy(dtype=float)
    bg_x = np.concatenate([gaia_x, cand_x])
    bg_y = np.concatenate([gaia_y, cand_y])
    mask = np.isfinite(bg_x) & np.isfinite(bg_y)
    bg_x = bg_x[mask]
    bg_y = bg_y[mask]

    if background_style == "density":
        _plot_cmd_background_density(ax, bg_x, bg_y)
    else:
        _plot_cmd_background(ax, bg_x, bg_y)

    ax.set_xlim(*CMD_XLIM)
    ax.set_ylim(*CMD_YLIM)
    ax.invert_yaxis()

    for bucket in bucket_order:
        sub = selected[selected["review_cmd_bucket"] == bucket]
        if sub.empty:
            continue
        style = CMD_BUCKET_STYLE[bucket]
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
        )
        _scatter_hollow(
            ax,
            sub_hollow,
            color=style["color"],
            size=marker_size,
            marker=style["marker"],
            zorder=style["zorder"],
            label=f"{bucket} observed ({len(sub_hollow):,})" if not sub_hollow.empty else None,
        )

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
        frameon=True,
        framealpha=0.92,
        fontsize=CMD_LEGEND_FONTSIZE,
        markerscale=CMD_LEGEND_MARKERSCALE,
    )
    save_publication_figure(fig, out_path, dpi=220, close=False, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    plt.close(fig)


def build_plot(
    review_db: Path,
    candidates_path: Path,
    output_path: Path,
    *,
    buckets: list[str] | None = None,
    title: str | None = None,
    only_reviewed: bool = True,
    background_style: str = "hybrid",
) -> tuple[pd.DataFrame, dict[str, int]]:
    bucket_order = buckets or BUCKET_ORDER
    reviews = _read_reviews(review_db, only_reviewed=only_reviewed)
    candidates = pd.read_parquet(candidates_path)
    cmd = _compute_cmd_coordinates(candidates)
    plot_df = cmd.merge(reviews, on="candidate_id", how="left")
    plot_df["review_cmd_bucket"] = plot_df.apply(_assign_bucket, axis=1)

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

    _plot_cmd(plot_df, output_path, bucket_order=bucket_order, background_style=background_style)
    csv_path = output_path.with_suffix(".csv")
    plot_df.loc[selected_mask].sort_values(["review_cmd_bucket", "candidate_id"]).to_csv(csv_path, index=False)
    return plot_df, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--buckets",
        nargs="+",
        choices=BUCKET_ORDER,
        default=None,
        help="Restrict plot/CSV to these review buckets (default: all buckets).",
    )
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument(
        "--background-style",
        choices=("hybrid", "density"),
        default="hybrid",
        help="Background rendering: 'hybrid' (scatter+contourf magma) or 'density' (pcolormesh viridis).",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include non-reviewed rows from the reviews table.",
    )
    return parser.parse_args()


def _apply_output_suffix(output: Path, background_style: str) -> Path:
    """Insert style suffix (e.g. '_density') before the extension when not hybrid."""
    if background_style == "hybrid":
        return output
    stem = output.stem
    if not stem.endswith(f"_{background_style}"):
        return output.with_name(f"{stem}_{background_style}{output.suffix}")
    return output


def main() -> None:
    args = parse_args()
    output = _apply_output_suffix(args.output, args.background_style)
    _plot_df, counts = build_plot(
        args.review_db,
        args.candidates,
        output,
        buckets=args.buckets,
        title=args.title,
        only_reviewed=not args.include_unreviewed,
        background_style=args.background_style,
    )
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")
    print(f"Wrote {output.with_suffix('.csv')}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
