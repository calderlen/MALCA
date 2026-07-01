#!/usr/bin/env python
"""Plot compact publication-style light-curve grids for March 18 review classes."""
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, StrMethodFormatter

from malca.config import DEFAULT_OUTPUT_DIR
from malca.io.lightcurve_io import load_lightcurve_df
from malca.plotting.lightcurve_publication import FIG_TWO_COL_WIDTH
from malca.review.coordinate_labels import format_j_designation


MARCH18_RUN = DEFAULT_OUTPUT_DIR / "runs" / "runs_march18_bundle_all"
DEFAULT_REVIEW_DB = MARCH18_RUN / "review" / "review.taxonomy_filled.db"
DEFAULT_OUTPUT_DIR = MARCH18_RUN / "results" / "review_lightcurve_grids_publication_singlepages"
DEFAULT_FIG_WIDTH = FIG_TWO_COL_WIDTH
DEFAULT_FIG_HEIGHT = FIG_TWO_COL_WIDTH * 11.0 / 8.5

CLASS_ORDER = ("dipper", "ltv", "microlensing")
CLASS_LABELS = {
    "dipper": "Dippers",
    "ltv": "LTV candidates",
    "microlensing": "Microlensing candidates",
}
CLASS_SLUGS = {
    "dipper": "dippers",
    "ltv": "ltv",
    "microlensing": "microlensing",
}
ERRORBAR_COLORS = {
    "dipper": "#d62728",
    "ltv": "#2ca02c",
    "microlensing": "#1f77b4",
}
CLASS_SORT_RANK = {name: idx for idx, name in enumerate(CLASS_ORDER)}

EVENT_MARKERS = (
    "dip_best_t0",
    "jump_best_t0",
)
HEADER_BOX = {
    "facecolor": "white",
    "edgecolor": "0.15",
    "linewidth": 0.50,
    "alpha": 1.0,
}
HEADER_FONT_SIZE = 5.4
HEADER_NAME_LEFT_X = 0.035
HEADER_NAME_WIDTH = 0.270
HEADER_BOX_GAP = 0.055
HEADER_COORD_LEFT_X = HEADER_NAME_LEFT_X + HEADER_NAME_WIDTH + HEADER_BOX_GAP
HEADER_COORD_WIDTH = 0.570
HEADER_BOX_HEIGHT = 0.056
SUBPLOT_LEFT = 0.067
SUBPLOT_RIGHT = 0.992
SUBPLOT_BOTTOM = 0.065
SUBPLOT_TOP = 0.978
SUBPLOT_WSPACE = 0.12
SUBPLOT_HSPACE = 0.13
X_LABEL_Y = 0.030
Y_LABEL_X = 0.026

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _read_review_rows(db_path: Path) -> pd.DataFrame:
    query = """
        SELECT
            r.candidate_id,
            lower(r.event_class) AS event_class,
            r.interest_score,
            r.workflow_status,
            r.status,
            c.lc_path,
            c.asas_sn_id,
            coalesce(
                c.ra,
                json_extract(c.payload_json, '$.ra'),
                json_extract(json_extract(c.payload_json, '$.payload_json'), '$.ra')
            ) AS ra,
            coalesce(
                c.dec,
                json_extract(c.payload_json, '$.dec'),
                json_extract(json_extract(c.payload_json, '$.payload_json'), '$.dec')
            ) AS dec,
            c.dip_best_t0,
            c.jump_best_t0
        FROM reviews AS r
        JOIN candidates AS c USING(candidate_id)
        WHERE lower(r.event_class) IN ('dipper', 'ltv', 'microlensing')
        ORDER BY
            CASE lower(r.event_class)
                WHEN 'dipper' THEN 0
                WHEN 'ltv' THEN 1
                WHEN 'microlensing' THEN 2
                ELSE 3
            END,
            coalesce(r.interest_score, 0) DESC,
            r.candidate_id
    """
    with sqlite3.connect(db_path) as conn:
        rows = pd.read_sql_query(query, conn)
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["event_class"] = rows["event_class"].astype(str)
    return rows


def _resolve_lightcurve_path(row: pd.Series, run_root: Path) -> Path:
    candidates: list[Path] = []
    raw_path = str(row.get("lc_path") or "").strip()
    if raw_path:
        candidates.append(Path(raw_path))

    for key in ("candidate_id", "asas_sn_id"):
        value = str(row.get(key) or "").strip()
        if value:
            for ext in ("dat3", "dat2", "dat"):
                candidates.append(run_root / "bundle_assets" / "lightcurves" / f"{value}.{ext}")

    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else Path()


def _plot_time_values(df: pd.DataFrame) -> pd.Series:
    jd = pd.to_numeric(df["jd"], errors="coerce")
    if jd.dropna().empty:
        return jd
    if float(jd.dropna().median()) > 50000.0:
        return jd - 2458000.0
    return jd - 8000.0


def _plot_time_value(value: float) -> float:
    if value > 50000.0:
        return value - 2458000.0
    return value - 8000.0


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _coordinate_headers(row: pd.Series) -> tuple[str | None, str | None]:
    ra = _finite_float(row.get("ra"))
    dec = _finite_float(row.get("dec"))
    if ra is None or dec is None:
        return str(row.get("candidate_id") or ""), None
    return (
        format_j_designation(ra, dec),
        rf"RA={ra:.4f}$\!^\circ$, DEC={dec:+.4f}$\!^\circ$",
    )


def _draw_compact_header_boxes(ax, *, left: str | None, right: str | None) -> None:
    def draw_boxed_text(x: float, width: float, text: str) -> None:
        y = 1.0
        ax.add_patch(
            Rectangle(
                (x, y - HEADER_BOX_HEIGHT / 2.0),
                width,
                HEADER_BOX_HEIGHT,
                transform=ax.transAxes,
                clip_on=False,
                zorder=29,
                **HEADER_BOX,
            )
        )
        ax.text(
            x + width / 2.0,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=HEADER_FONT_SIZE,
            color="0.10",
            clip_on=False,
            zorder=30,
        )

    if left:
        draw_boxed_text(
            HEADER_NAME_LEFT_X,
            HEADER_NAME_WIDTH,
            left,
        )
    if right:
        draw_boxed_text(
            HEADER_COORD_LEFT_X,
            HEADER_COORD_WIDTH,
            right,
        )


def _set_lightcurve_limits(ax, x: pd.Series, y: pd.Series) -> None:
    finite_x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    finite_y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    finite_y = finite_y[np.isfinite(finite_y)]

    if finite_x.size:
        xmin = float(np.nanmin(finite_x))
        xmax = float(np.nanmax(finite_x))
        xpad = max(3.0, (xmax - xmin) * 0.005)
        ax.set_xlim(
            math.floor((xmin - xpad) / 250.0) * 250.0,
            math.ceil((xmax + xpad) / 250.0) * 250.0,
        )

    if finite_y.size:
        ymin = float(np.nanmin(finite_y))
        ymax = float(np.nanmax(finite_y))
        ypad = max(0.015, (ymax - ymin) * 0.045) if ymax > ymin else 0.1
        ax.set_ylim(ymax + ypad, ymin - ypad)


def _style_panel(ax) -> None:
    ax.tick_params(
        which="major",
        direction="in",
        top=False,
        right=True,
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
        labelright=False,
        labeltop=False,
        labelsize=5.6,
        length=4.1,
        width=0.65,
        pad=1.4,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=False,
        right=True,
        left=True,
        bottom=True,
        length=2.2,
        width=0.55,
    )
    ax.grid(True, which="major", linewidth=0.24, alpha=0.14)
    ax.xaxis.set_major_locator(MultipleLocator(1000.0))
    ax.xaxis.set_minor_locator(MultipleLocator(250.0))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def _plot_candidate_panel(ax, row: pd.Series, *, run_root: Path) -> dict[str, object]:
    candidate_id = str(row["candidate_id"])
    lc_path = _resolve_lightcurve_path(row, run_root)
    status: dict[str, object] = {
        "candidate_id": candidate_id,
        "event_class": str(row["event_class"]),
        "lc_path": str(lc_path),
        "plot_status": "ok",
        "plot_points": 0,
        "has_coordinate_headers": False,
    }

    _style_panel(ax)
    header_left, header_right = _coordinate_headers(row)
    status["has_coordinate_headers"] = bool(header_left and header_right)

    if not lc_path.exists():
        status["plot_status"] = "missing_lightcurve"
        ax.text(0.5, 0.5, "missing light curve", ha="center", va="center", fontsize=5.4, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_compact_header_boxes(ax, left=header_left, right=header_right)
        return status

    try:
        df = load_lightcurve_df(lc_path, apply_quality=True)
    except Exception as exc:
        status["plot_status"] = f"load_error: {exc}"
        ax.text(0.5, 0.5, "load error", ha="center", va="center", fontsize=5.4, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_compact_header_boxes(ax, left=header_left, right=header_right)
        return status

    if df.empty:
        status["plot_status"] = "empty_lightcurve"
        ax.text(0.5, 0.5, "empty light curve", ha="center", va="center", fontsize=5.4, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_compact_header_boxes(ax, left=header_left, right=header_right)
        return status

    x = _plot_time_values(df)
    y = pd.to_numeric(df["mag"], errors="coerce")
    status["plot_points"] = int(np.isfinite(x.to_numpy(dtype=float)).sum())

    yerr = pd.to_numeric(df.get("mag_err"), errors="coerce") if "mag_err" in df.columns else pd.Series(np.nan, index=df.index)
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    err_mask = mask & np.isfinite(yerr.to_numpy(dtype=float)) & (yerr.to_numpy(dtype=float) > 0)
    errorbar_color = ERRORBAR_COLORS.get(str(row["event_class"]), "#d62728")
    if err_mask.any():
        ax.errorbar(
            x[err_mask],
            y[err_mask],
            yerr=yerr[err_mask],
            fmt="none",
            ecolor=errorbar_color,
            elinewidth=0.24,
            capsize=0.85,
            capthick=0.24,
            zorder=1,
        )
    if mask.any():
        ax.scatter(
            x[mask],
            y[mask],
            s=1.65,
            color="black",
            linewidths=0,
            rasterized=False,
            zorder=3,
        )

    for col in EVENT_MARKERS:
        event_t0 = _finite_float(row.get(col))
        if event_t0 is not None:
            ax.axvline(_plot_time_value(event_t0), color="black", linestyle="--", linewidth=0.55, zorder=2)

    _set_lightcurve_limits(ax, x, y)
    _draw_compact_header_boxes(ax, left=header_left, right=header_right)
    return status


def _page_chunks(df: pd.DataFrame, page_size: int):
    for start in range(0, len(df), page_size):
        yield start // page_size + 1, df.iloc[start : start + page_size]


def _active_page_geometry(*, active_rows: int, max_rows: int, full_page_height: float) -> tuple[float, float, float, float, float]:
    active_rows = max(1, int(active_rows))
    max_rows = max(1, int(max_rows))
    full_grid_height = full_page_height * (SUBPLOT_TOP - SUBPLOT_BOTTOM)
    full_row_units = max_rows + max(0, max_rows - 1) * SUBPLOT_HSPACE
    panel_height = full_grid_height / full_row_units
    active_grid_height = panel_height * (active_rows + max(0, active_rows - 1) * SUBPLOT_HSPACE)

    bottom_margin = full_page_height * SUBPLOT_BOTTOM
    top_margin = full_page_height * (1.0 - SUBPLOT_TOP)
    page_height = active_grid_height + bottom_margin + top_margin

    bottom = bottom_margin / page_height
    top = 1.0 - top_margin / page_height
    xlabel_y = (full_page_height * X_LABEL_Y) / page_height
    ylabel_y = 0.5 * (bottom + top)
    return page_height, bottom, top, xlabel_y, ylabel_y


def _make_page_figure(
    page_rows: pd.DataFrame,
    *,
    event_class: str,
    page_number: int,
    n_pages: int,
    rows: int,
    cols: int,
    run_root: Path,
    page_width: float,
    page_height: float,
) -> tuple[object, list[dict[str, object]]]:
    active_rows = max(1, math.ceil(len(page_rows) / cols))
    active_page_height, subplot_bottom, subplot_top, xlabel_y, ylabel_y = _active_page_geometry(
        active_rows=active_rows,
        max_rows=rows,
        full_page_height=page_height,
    )
    fig, axes = plt.subplots(active_rows, cols, figsize=(page_width, active_page_height), squeeze=False)

    statuses: list[dict[str, object]] = []
    flat_axes = axes.ravel()
    for idx, (_, row) in enumerate(page_rows.iterrows()):
        statuses.append(_plot_candidate_panel(flat_axes[idx], row, run_root=run_root))

    for ax in flat_axes[len(page_rows) :]:
        ax.axis("off")

    for row_axes in axes:
        for ax in row_axes:
            if not ax.has_data():
                continue
            ax.tick_params(axis="both", labelleft=True, labelbottom=True)

    fig.text(0.525, xlabel_y, "JD - 2458000 [d]", ha="center", va="bottom", fontsize=8.2, color="0.10")
    fig.text(Y_LABEL_X, ylabel_y, "m [mag]", ha="center", va="center", rotation="vertical", fontsize=8.2, color="0.10")
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=subplot_bottom,
        top=subplot_top,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )
    return fig, statuses


def _write_class_pdfs(
    class_rows: pd.DataFrame,
    *,
    event_class: str,
    output_dir: Path,
    rows: int,
    cols: int,
    dpi: int,
    run_root: Path,
    page_width: float,
    page_height: float,
) -> tuple[Path, list[dict[str, object]]]:
    page_size = rows * cols
    n_pages = max(1, math.ceil(len(class_rows) / page_size))
    slug = CLASS_SLUGS.get(event_class, event_class)
    pdf_paths: list[Path] = []
    all_statuses: list[dict[str, object]] = []

    for stale_path in output_dir.glob(f"march18_review_{slug}_lightcurve_grid_sheet*.pdf"):
        stale_path.unlink()

    for page_number, page_rows in _page_chunks(class_rows, page_size):
        pdf_path = output_dir / f"march18_review_{slug}_lightcurve_grid_sheet{page_number:02d}.pdf"
        fig, statuses = _make_page_figure(
            page_rows,
            event_class=event_class,
            page_number=page_number,
            n_pages=n_pages,
            rows=rows,
            cols=cols,
            run_root=run_root,
            page_width=page_width,
            page_height=page_height,
        )
        with PdfPages(pdf_path) as class_pdf:
            class_pdf.savefig(fig, dpi=dpi, metadata={"Creator": "MALCA"})
        plt.close(fig)
        pdf_paths.append(pdf_path)
        for item in statuses:
            item.update({"sheet": page_number, "pdf_path": str(pdf_path)})
        all_statuses.extend(statuses)
    return pdf_paths, all_statuses


def make_lightcurve_grids(
    *,
    review_db: Path,
    run_root: Path,
    output_dir: Path,
    rows: int,
    cols: int,
    dpi: int,
    page_width: float,
    page_height: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = _read_review_rows(review_db)
    manifest_path = output_dir / "march18_review_lightcurve_grid_manifest.csv"

    pdf_paths: dict[str, list[Path]] = {}
    statuses: list[dict[str, object]] = []
    for event_class in CLASS_ORDER:
        class_rows = review_rows[review_rows["event_class"] == event_class].copy()
        if class_rows.empty:
            continue
        class_pdf_paths, class_statuses = _write_class_pdfs(
            class_rows,
            event_class=event_class,
            output_dir=output_dir,
            rows=rows,
            cols=cols,
            dpi=dpi,
            run_root=run_root,
            page_width=page_width,
            page_height=page_height,
        )
        pdf_paths[event_class] = class_pdf_paths
        statuses.extend(class_statuses)

    manifest = pd.DataFrame(statuses)
    if not manifest.empty:
        manifest["class_order"] = manifest["event_class"].map(CLASS_SORT_RANK)
        manifest = manifest.sort_values(["class_order", "sheet", "candidate_id"]).drop(columns=["class_order"])
    manifest.to_csv(manifest_path, index=False)

    counts = review_rows.groupby("event_class").size().to_dict()
    return {
        "counts": {key: int(counts.get(key, 0)) for key in CLASS_ORDER},
        "pdfs": pdf_paths,
        "manifest": manifest_path,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--run-root", type=Path, default=MARCH18_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--page-width", type=float, default=DEFAULT_FIG_WIDTH)
    parser.add_argument("--page-height", type=float, default=DEFAULT_FIG_HEIGHT)
    args = parser.parse_args()

    result = make_lightcurve_grids(
        review_db=args.review_db,
        run_root=args.run_root,
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        dpi=args.dpi,
        page_width=args.page_width,
        page_height=args.page_height,
    )
    print("Wrote March 18 review light-curve grids")
    print(f"  counts: {result['counts']}")
    for event_class, pdf_paths in result["pdfs"].items():
        for pdf_path in pdf_paths:
            print(f"  {event_class}: {pdf_path}")
    print(f"  manifest: {result['manifest']}")
    print(f"  output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
