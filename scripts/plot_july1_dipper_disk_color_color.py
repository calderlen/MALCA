#!/usr/bin/env python
"""Plot July 1 reviewed dippers on a 2MASS/WISE disk color-color diagram."""
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

from malca.extinction import MID_IR_EXTINCTION_POLICY_VERSION
from malca.plotting.color_color_labels import LABEL_KS_W3_0, LABEL_KS_W4_0
from malca.plotting.extinction import (
    IR_AV_COEFFICIENTS,
    add_dereddened_ir_magnitudes,
    dereddened_color,
)
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from malca.plotting.blackbody_locus import add_blackbody_locus
from malca.plotting.lightcurve_publication import PUBLICATION_STYLE


DEFAULT_RUN_ROOT = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_REVIEW_DB = DEFAULT_RUN_ROOT / "review" / "review.db"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "results" / "dipper_disk_color_color"


# Figure-style region demarcations from the supplied 2MASS-WISE diagram.
# These are drawn as empirical reference guides in the corrected color plane:
# x = (Ks - W4)_0, y = (Ks - W3)_0.
BOUNDARY_SEGMENTS = {
    "diskless_debris": [(0.00, 0.50), (0.42, -0.32)],
    "debris_evolved": [(1.50, 1.25), (2.40, -0.25)],
    "evolved_full": [(2.50, 2.50), (3.50, 1.50), (5.00, 0.00)],
    "evolved_transition": [(3.50, 1.50), (6.55, 2.70)],
}

REGION_LABELS = {
    "Diskless": (-0.36, -0.56),
    "Debris": (1.55, -0.56),
    "Evolved": (3.55, -0.56),
    "Full": (4.00, 2.72),
    "Transition": (5.28, 1.62),
}

REGION_ORDER = ("Diskless", "Debris", "Evolved", "Full", "Transition")
REGION_ERROR_COLORS = {
    "Diskless": "#4c72b0",
    "Debris": "#55a868",
    "Evolved": "#dd8452",
    "Full": "#c44e52",
    "Transition": "#8172b3",
}


plt.rcParams.update(
    {
        **PUBLICATION_STYLE,
        "axes.formatter.use_mathtext": True,
    }
)


def _read_dippers(db_path: Path) -> pd.DataFrame:
    query = """
        SELECT
            c.candidate_id,
            c.asas_sn_id,
            c.ra,
            c.dec,
            c.A_v_3d,
            r.event_class,
            r.workflow_status,
            r.status,
            r.physical_primary,
            r.physical_secondary,
            r.classification_confidence,
            c.yso_class,
            c.tmass_k,
            c.tmass_k_err,
            c.w3,
            c.w3_err,
            c.w4,
            c.w4_err,
            c.w1,
            c.w2,
            c.w1_w2,
            c.w1_w3,
            c.w1_w4,
            c.w2_w3,
            c.w2_w4,
            c.w3_w4,
            c.sed_alpha,
            c.sed_alpha_class,
            c.sed_alpha_status,
            c.vsx_class,
            c.asassn_var_type,
            c.gaia_var_class,
            c.simbad_otype,
            c.nearby_vsx_dipper_contaminant
        FROM candidates AS c
        JOIN reviews AS r USING(candidate_id)
        WHERE lower(r.event_class) = 'dipper'
        ORDER BY c.candidate_id
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def _add_colors(df: pd.DataFrame) -> pd.DataFrame:
    out = add_dereddened_ir_magnitudes(df)
    for col in ("tmass_k", "tmass_k_err", "w3", "w3_err", "w4", "w4_err"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ks_w3"] = out["tmass_k"] - out["w3"]
    out["ks_w4"] = out["tmass_k"] - out["w4"]
    out["ks_w3_0"] = dereddened_color(out, "tmass_k", "w3")
    out["ks_w4_0"] = dereddened_color(out, "tmass_k", "w4")
    out["has_disk_color_axes"] = out[["tmass_k", "w3", "w4"]].notna().all(axis=1)
    out["has_disk_color_errors"] = out[["tmass_k_err", "w3_err", "w4_err"]].notna().all(axis=1)
    out["ks_w3_err"] = np.sqrt(out["tmass_k_err"] ** 2 + out["w3_err"] ** 2)
    out["ks_w4_err"] = np.sqrt(out["tmass_k_err"] ** 2 + out["w4_err"] ** 2)
    out.loc[~out["has_disk_color_errors"], ["ks_w3_err", "ks_w4_err"]] = np.nan
    out["disk_region"] = [
        classify_disk_region(x, y) if np.isfinite(x) and np.isfinite(y) else np.nan
        for x, y in zip(out["ks_w4_0"], out["ks_w3_0"])
    ]
    return out


def _line_side(x0: float, y0: float, x1: float, y1: float, x: float, y: float) -> float:
    return (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)


def classify_disk_region(ks_w4: float, ks_w3: float) -> str:
    """Assign a disk-stage region from the figure-style boundary segments."""
    x = float(ks_w4)
    y = float(ks_w3)
    s_diskless_debris = _line_side(0.0, 0.5, 0.42, -0.32, x, y)
    s_debris_evolved = _line_side(1.5, 1.25, 2.4, -0.25, x, y)
    s_evolved_transition = _line_side(3.5, 1.5, 6.55, 2.7, x, y)
    s_upper = (
        _line_side(2.5, 2.5, 3.5, 1.5, x, y)
        if x <= 3.5
        else _line_side(3.5, 1.5, 5.0, 0.0, x, y)
    )
    if s_upper > 0:
        return "Full" if s_evolved_transition > 0 else "Transition"
    if s_diskless_debris < 0:
        return "Diskless"
    if s_debris_evolved < 0:
        return "Debris"
    return "Evolved"


def _write_boundary_table(output_dir: Path) -> pd.DataFrame:
    rows = []
    for segment_name, points in BOUNDARY_SEGMENTS.items():
        for vertex_index, (ks_w4, ks_w3) in enumerate(points):
            rows.append(
                {
                    "segment": segment_name,
                    "vertex_index": vertex_index,
                    "ks_w4": ks_w4,
                    "ks_w3": ks_w3,
                    "x_axis": "(tmass_k - w4)_0",
                    "y_axis": "(tmass_k - w3)_0",
                    "note": "Figure-style boundary vertices used as empirical reference guides in the extinction-corrected 2MASS-WISE plane.",
                }
            )
    boundary_df = pd.DataFrame(rows)
    boundary_df.to_csv(output_dir / "july1_dipper_disk_color_boundaries.csv", index=False)
    with (output_dir / "july1_dipper_disk_color_boundaries.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "axes": {
                    "x": "(K_s - W4)_0 = tmass_k_0 - w4_0",
                    "y": "(K_s - W3)_0 = tmass_k_0 - w3_0",
                },
                "boundary_segments": BOUNDARY_SEGMENTS,
                "region_labels": REGION_LABELS,
                "provenance": (
                    "Figure-style boundaries digitized from the supplied 2MASS-WISE "
                    "disk color-color diagram and used as empirical guides after applying "
                    "the project's foreground-extinction correction; Luhman & Mamajek "
                    "(2012) describe the underlying disk-stage criteria in excess-color space."
                ),
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    return boundary_df


def _plot(df: pd.DataFrame, output_dir: Path) -> None:
    plotted = df[df["has_disk_color_axes"]].copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if plotted["has_disk_color_errors"].any():
        err = plotted[plotted["has_disk_color_errors"]]
        for region_name in REGION_ORDER:
            region = err.loc[err["disk_region"] == region_name]
            if region.empty:
                continue
            ax.errorbar(
                region["ks_w4_0"],
                region["ks_w3_0"],
                xerr=region["ks_w4_err"],
                yerr=region["ks_w3_err"],
                fmt="none",
                ecolor=REGION_ERROR_COLORS[region_name],
                elinewidth=0.85,
                capsize=2.8,
                capthick=0.75,
                alpha=0.9,
                zorder=1,
            )

    ax.scatter(
        plotted["ks_w4_0"],
        plotted["ks_w3_0"],
        s=22,
        marker="o",
        facecolor="black",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.95,
        zorder=3,
    )

    for segment_name, points in BOUNDARY_SEGMENTS.items():
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            color="black",
            linestyle=(0, (5, 2.5)),
            linewidth=1.4,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=2,
        )

    for label, (x, y) in REGION_LABELS.items():
        ax.text(x, y, label, fontsize=17, ha="left", va="center", color="black", zorder=4)

    ax.set_xlim(-0.5, 6.8)
    ax.set_ylim(-0.75, 4.6)
    ax.set_xlabel(LABEL_KS_W4_0, fontsize=18)
    ax.set_ylabel(LABEL_KS_W3_0, fontsize=18)
    add_blackbody_locus(
        ax,
        ("Ks", "W4"),
        ("Ks", "W3"),
        label_placement="above",
    )

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=5, width=0.9, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    stem = output_dir / "july1_dipper_2mass_wise_disk_color_color"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = _add_colors(_read_dippers(args.review_db))
    df.to_csv(args.output_dir / "july1_dipper_2mass_wise_disk_color_color.csv", index=False)
    _write_boundary_table(args.output_dir)
    _plot(df, args.output_dir)

    plotted = int(df["has_disk_color_axes"].sum())
    region_counts = {
        region: int((df["disk_region"] == region).sum())
        for region in REGION_ORDER
    }
    with (args.output_dir / "july1_dipper_2mass_wise_disk_color_color_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "review_db": str(args.review_db),
                "n_dippers": int(len(df)),
                "n_plotted": plotted,
                "n_missing_axes": int(len(df) - plotted),
                "n_with_errors": int(df["has_disk_color_errors"].sum()),
                "disk_region_counts": region_counts,
                "extinction_correction": "Foreground A_v_3d with scalar R_V=3.1 band coefficients.",
                "extinction_policy_version": MID_IR_EXTINCTION_POLICY_VERSION,
                "band_av_coefficients": {
                    "2MASS_Ks": IR_AV_COEFFICIENTS["tmass_k"],
                    "AllWISE_W3": IR_AV_COEFFICIENTS["w3"],
                    "AllWISE_W4": IR_AV_COEFFICIENTS["w4"],
                },
                "output_dir": str(args.output_dir),
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote disk color-color plot for {plotted}/{len(df)} dippers to {args.output_dir}")


if __name__ == "__main__":
    main()
