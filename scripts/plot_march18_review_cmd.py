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


DEFAULT_REVIEW_DB = Path("output/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db")
DEFAULT_CANDIDATES = Path(
    "output_migrated_camera_field_20260606/runs/runs_march18_bundle_all/results/"
    "lc_events_classified.parquet"
)
DEFAULT_OUTPUT = Path("output/runs/runs_march18_bundle_all/results/march18_review_cmd_selected.png")


BUCKET_ORDER = ["Dipper", "Interesting", "LTV", "Eclipsing binary", "Unknown"]
BUCKET_STYLE = {
    "Dipper": {"color": "#0072B2", "marker": "o", "size": 54, "zorder": 6},
    "Interesting": {"color": "#D55E00", "marker": "o", "size": 34, "zorder": 4},
    "LTV": {"color": "#009E73", "marker": "^", "size": 64, "zorder": 7},
    "Eclipsing binary": {"color": "#CC79A7", "marker": "s", "size": 74, "zorder": 8},
    "Unknown": {"color": "#6A3D9A", "marker": "D", "size": 44, "zorder": 5},
}


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


def _read_reviews(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
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
            """,
            conn,
        ).astype({"candidate_id": "string"})


def _compute_cmd_coordinates(candidates: pd.DataFrame) -> pd.DataFrame:
    derived = _json_frame(candidates.get("derived_stats", pd.Series(index=candidates.index, dtype=object)))
    external = _json_frame(candidates.get("external_stats", pd.Series(index=candidates.index, dtype=object)))

    out = pd.DataFrame({"candidate_id": candidates["candidate_id"].astype("string")})
    for col in ("bp_rp", "bprp0", "mg0"):
        out[col] = pd.to_numeric(derived.get(col), errors="coerce")
    for col in ("phot_g_mean_mag", "distance_gspphot", "parallax", "source_id", "gaia_id"):
        if col in external.columns:
            out[col] = external[col]
        else:
            out[col] = np.nan

    dist_pc = pd.to_numeric(out["distance_gspphot"], errors="coerce")
    parallax = pd.to_numeric(out["parallax"], errors="coerce")
    dist_pc = dist_pc.where(dist_pc > 0, np.where(parallax > 0, 1000.0 / parallax, np.nan))
    observed_mg = pd.to_numeric(out["phot_g_mean_mag"], errors="coerce") - 5.0 * np.log10(dist_pc) + 5.0

    out["cmd_color"] = out["bprp0"].where(out["bprp0"].notna(), out["bp_rp"])
    out["cmd_mag"] = out["mg0"].where(out["mg0"].notna(), observed_mg)
    out["cmd_coordinate_source"] = np.where(
        out["bprp0"].notna() & out["mg0"].notna(),
        "dereddened_bprp0_mg0",
        np.where(out["cmd_color"].notna() & out["cmd_mag"].notna(), "observed_fallback", "missing"),
    )
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
    if event_class == "periodic" or morph_secondary in {"detached_binary_like", "eclipsing_like"}:
        return "Eclipsing binary"
    if physical_primary == "unknown" or event_class in {"unknown", "unclear"}:
        return "Unknown"
    return None


def _plot_cmd(plot_df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_cmd = plot_df[plot_df["cmd_color"].notna() & plot_df["cmd_mag"].notna()].copy()
    selected = all_cmd[all_cmd["review_cmd_bucket"].notna()].copy()

    fig, ax = plt.subplots(figsize=(7.1, 6.4))
    ax.scatter(
        all_cmd["cmd_color"],
        all_cmd["cmd_mag"],
        s=8,
        c="0.78",
        alpha=0.22,
        edgecolors="none",
        linewidths=0,
        label=f"March 18 background ({len(all_cmd):,})",
        zorder=1,
    )

    for bucket in BUCKET_ORDER:
        sub = selected[selected["review_cmd_bucket"] == bucket]
        if sub.empty:
            continue
        style = BUCKET_STYLE[bucket]
        ax.scatter(
            sub["cmd_color"],
            sub["cmd_mag"],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.55,
            label=f"{bucket} ({len(sub):,})",
            zorder=style["zorder"],
        )

    ax.set_xlabel(r"Gaia $BP-RP$ color")
    ax.set_ylabel(r"Gaia $M_G$")
    ax.invert_yaxis()
    ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_title(title)
    ax.text(
        0.02,
        0.02,
        "Dereddened bprp0/mg0 where available; observed Gaia fallback otherwise.",
        transform=ax.transAxes,
        fontsize=8,
        color="0.32",
        ha="left",
        va="bottom",
    )
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def build_plot(review_db: Path, candidates_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    reviews = _read_reviews(review_db)
    candidates = pd.read_parquet(candidates_path)
    cmd = _compute_cmd_coordinates(candidates)
    plot_df = cmd.merge(reviews, on="candidate_id", how="left")
    plot_df["review_cmd_bucket"] = plot_df.apply(_assign_bucket, axis=1)

    plotted_mask = (
        plot_df["review_cmd_bucket"].notna()
        & plot_df["cmd_color"].notna()
        & plot_df["cmd_mag"].notna()
    )
    selected_mask = plot_df["review_cmd_bucket"].notna()
    counts = {
        "all_candidates": int(len(plot_df)),
        "all_cmd_points": int((plot_df["cmd_color"].notna() & plot_df["cmd_mag"].notna()).sum()),
        "selected_candidates": int(selected_mask.sum()),
        "selected_cmd_points": int(plotted_mask.sum()),
    }
    for bucket in BUCKET_ORDER:
        bucket_mask = plot_df["review_cmd_bucket"].eq(bucket)
        counts[f"{bucket}_selected"] = int(bucket_mask.sum())
        counts[f"{bucket}_plotted"] = int(
            (bucket_mask & plot_df["cmd_color"].notna() & plot_df["cmd_mag"].notna()).sum()
        )

    _plot_cmd(
        plot_df,
        output_path,
        title="March 18 Reviewed Candidates on Gaia CMD",
    )
    csv_path = output_path.with_suffix(".csv")
    plot_df.loc[selected_mask].sort_values(["review_cmd_bucket", "candidate_id"]).to_csv(csv_path, index=False)
    return plot_df, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _plot_df, counts = build_plot(args.review_db, args.candidates, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.csv')}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
