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
    CMD_BG_HOLLOW_SCATTER_SIZE,
    CMD_BG_SCATTER_SIZE,
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
    alpha=0.88,
    zorder=2,
    label=None,
    edge_lw: float = CMD_MARKER_EDGE_SOLID,
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
        edgecolors="white",
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
    alpha=0.88,
    zorder=2,
    label=None,
    edge_lw: float = CMD_MARKER_EDGE_HOLLOW,
) -> None:
    if df.empty:
        return
    ax.scatter(
        df["cmd_color"],
        df["cmd_mag"],
        s=size,
        facecolors="none",
        edgecolors=color,
        marker=marker,
        alpha=alpha,
        linewidths=edge_lw,
        label=label,
        zorder=zorder,
    )


def _plot_cmd(
    plot_df: pd.DataFrame,
    out_path: Path,
    *,
    bucket_order: list[str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_order = bucket_order or BUCKET_ORDER
    plottable = plot_df[_plottable_mask(plot_df)].copy()
    selected = plottable[plottable["review_cmd_bucket"].isin(bucket_order)].copy()
    background = plottable[~plottable["review_cmd_bucket"].isin(bucket_order)].copy()

    fig, ax = plt.subplots(figsize=CMD_FIGSIZE)

    bg_solid = background[background["cmd_coordinate_source"].isin(SOLID_CMD_SOURCES)]
    bg_hollow = background[background["cmd_coordinate_source"].isin(HOLLOW_CMD_SOURCES)]
    _scatter_solid(
        ax,
        bg_solid,
        color="0.45",
        size=CMD_BG_SCATTER_SIZE,
        alpha=0.35,
        zorder=1,
        label=f"Background ({len(bg_solid):,})",
    )
    if not bg_hollow.empty:
        _scatter_hollow(
            ax,
            bg_hollow,
            color="0.55",
            size=CMD_BG_HOLLOW_SCATTER_SIZE,
            alpha=0.35,
            zorder=1,
            label=f"Background, observed ({len(bg_hollow):,})",
        )

    for bucket in bucket_order:
        sub = selected[selected["review_cmd_bucket"] == bucket]
        if sub.empty:
            continue
        style = CMD_BUCKET_STYLE[bucket]
        sub_solid = sub[sub["cmd_coordinate_source"].isin(SOLID_CMD_SOURCES)]
        sub_hollow = sub[sub["cmd_coordinate_source"].isin(HOLLOW_CMD_SOURCES)]
        solid_label = f"{bucket} ({len(sub):,})" if sub_hollow.empty else f"{bucket} solid ({len(sub_solid):,})"
        marker_size = float(style["size"])
        _scatter_solid(
            ax,
            sub_solid,
            color=style["color"],
            size=marker_size,
            marker=style["marker"],
            zorder=style["zorder"],
            label=solid_label if not sub_solid.empty else None,
        )
        if not sub_hollow.empty:
            _scatter_hollow(
                ax,
                sub_hollow,
                color=style["color"],
                size=marker_size,
                marker=style["marker"],
                zorder=style["zorder"],
                label=f"{bucket} observed ({len(sub_hollow):,})",
            )

    ax.set_xlabel(r"$G_{\mathrm{BP}} - G_{\mathrm{RP}}$ [mag]", fontsize=CMD_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$M_G$ [mag]", fontsize=CMD_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(*CMD_XLIM)
    ax.set_ylim(*CMD_YLIM)
    ax.invert_yaxis()
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
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        fontsize=CMD_LEGEND_FONTSIZE,
        markerscale=CMD_LEGEND_MARKERSCALE,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def build_plot(
    review_db: Path,
    candidates_path: Path,
    output_path: Path,
    *,
    buckets: list[str] | None = None,
    title: str | None = None,
    only_reviewed: bool = True,
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

    _plot_cmd(plot_df, output_path, bucket_order=bucket_order)
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
        "--include-unreviewed",
        action="store_true",
        help="Include non-reviewed rows from the reviews table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _plot_df, counts = build_plot(
        args.review_db,
        args.candidates,
        args.output,
        buckets=args.buckets,
        title=args.title,
        only_reviewed=not args.include_unreviewed,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.csv')}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
