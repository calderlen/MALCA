#!/usr/bin/env python
"""Plot fractional dip depth versus timescale for all July 1 MALCA dippers."""
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
from matplotlib.ticker import LogLocator, NullFormatter

from malca.io.lightcurve_io import load_lightcurve_df
from malca.plotting.lightcurve_publication import PUBLICATION_STYLE


DEFAULT_RUN_ROOT = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_REVIEW_DB = DEFAULT_RUN_ROOT / "review" / "review.db"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "results" / "dipper_depth_timescale"

FALLBACK_BIN_DAYS = 14.0
FALLBACK_MAX_BIN_GAP_DAYS = 56.0
FALLBACK_DEPTH_THRESHOLD_FRACTION = 0.30
FALLBACK_MIN_PEAK_FRACTION = 0.80
FALLBACK_MIN_POINTS_PER_BIN = 2


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
            c.lc_path,
            c.ra,
            c.dec,
            c.dip_best_mag_event,
            c.dip_max_run_duration,
            c.dip_run_count,
            c.dip_best_morph,
            c.dip_significant,
            r.event_class,
            r.status,
            r.classification_confidence,
            r.morphology_primary,
            r.morphology_secondary,
            r.physical_primary
        FROM candidates AS c
        JOIN reviews AS r USING(candidate_id)
        WHERE lower(r.event_class) = 'dipper'
          AND lower(r.status) = 'reviewed'
        ORDER BY c.candidate_id
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(query, conn)


def _lightcurve_columns(lightcurve: pd.DataFrame) -> tuple[str, str]:
    time_col = "jd" if "jd" in lightcurve.columns else "JD"
    if time_col not in lightcurve.columns:
        raise ValueError("Light curve has no jd/JD time column")

    for candidate in ("camera", "camera#", "band"):
        if candidate in lightcurve.columns:
            return time_col, candidate
    raise ValueError("Light curve has no camera or band column for baseline alignment")


def _fallback_duration_from_lightcurve(lc_path: Path, depth_mag: float) -> float:
    """Estimate a broad-dip duration when the event table has no kept run.

    The fallback is deliberately conservative and auditable. It aligns each
    camera to its median, measures 14-day bin medians, and returns the longest
    connected sequence whose residual exceeds 30% of the stored dip depth and
    whose peak reaches at least 80% of that depth.
    """
    lightcurve = load_lightcurve_df(lc_path, apply_quality=True).copy()
    time_col, group_col = _lightcurve_columns(lightcurve)
    lightcurve[time_col] = pd.to_numeric(lightcurve[time_col], errors="coerce")
    lightcurve["mag"] = pd.to_numeric(lightcurve["mag"], errors="coerce")
    lightcurve = lightcurve.dropna(subset=[time_col, "mag", group_col])
    if lightcurve.empty:
        return np.nan

    lightcurve["camera_residual_mag"] = lightcurve["mag"] - lightcurve.groupby(
        group_col, observed=True
    )["mag"].transform("median")
    lightcurve["time_bin_start"] = (
        np.floor(lightcurve[time_col] / FALLBACK_BIN_DAYS) * FALLBACK_BIN_DAYS
    )
    binned = (
        lightcurve.groupby([group_col, "time_bin_start"], observed=True)
        .agg(
            n_points=("mag", "size"),
            residual_mag=("camera_residual_mag", "median"),
        )
        .reset_index()
    )
    threshold = FALLBACK_DEPTH_THRESHOLD_FRACTION * float(depth_mag)
    binned = binned.loc[
        (binned["n_points"] >= FALLBACK_MIN_POINTS_PER_BIN)
        & (binned["residual_mag"] >= threshold)
    ]

    durations: list[float] = []
    for _, camera_bins in binned.groupby(group_col, observed=True):
        camera_bins = camera_bins.sort_values("time_bin_start").reset_index(drop=True)
        if camera_bins.empty:
            continue
        starts = [0]
        starts.extend(
            (np.flatnonzero(np.diff(camera_bins["time_bin_start"]) > FALLBACK_MAX_BIN_GAP_DAYS) + 1).tolist()
        )
        ends = [index - 1 for index in starts[1:]] + [len(camera_bins) - 1]
        for start, end in zip(starts, ends):
            run = camera_bins.iloc[start : end + 1]
            if run["residual_mag"].max() < FALLBACK_MIN_PEAK_FRACTION * float(depth_mag):
                continue
            duration = (
                float(run["time_bin_start"].iloc[-1])
                - float(run["time_bin_start"].iloc[0])
                + FALLBACK_BIN_DAYS
            )
            if duration > 0:
                durations.append(duration)

    return max(durations) if durations else np.nan


def _measure_plot_values(df: pd.DataFrame) -> pd.DataFrame:
    measured = df.copy()
    measured["depth_mag"] = pd.to_numeric(measured["dip_best_mag_event"], errors="coerce")
    measured["fractional_depth"] = 1.0 - 10.0 ** (-0.4 * measured["depth_mag"])
    measured["timescale_days"] = pd.to_numeric(
        measured["dip_max_run_duration"], errors="coerce"
    )
    measured["depth_source"] = "dip_best_mag_event"
    measured["timescale_source"] = "dip_max_run_duration"

    missing_duration = ~np.isfinite(measured["timescale_days"]) | measured["timescale_days"].le(0)
    for index, row in measured.loc[missing_duration].iterrows():
        lc_path = Path(str(row["lc_path"])).expanduser()
        fallback = _fallback_duration_from_lightcurve(lc_path, float(row["depth_mag"]))
        measured.loc[index, "timescale_days"] = fallback
        measured.loc[index, "timescale_source"] = (
            "fallback_binned_lightcurve_14d_30pct_depth_56d_gap"
        )

    measured["has_plot_values"] = (
        np.isfinite(measured["fractional_depth"])
        & measured["fractional_depth"].gt(0)
        & measured["fractional_depth"].lt(1)
        & np.isfinite(measured["timescale_days"])
        & measured["timescale_days"].gt(0)
    )
    if measured["candidate_id"].duplicated().any():
        duplicates = measured.loc[measured["candidate_id"].duplicated(), "candidate_id"].tolist()
        raise ValueError(f"Duplicate dipper candidate IDs: {duplicates}")
    if not measured["has_plot_values"].all():
        missing = measured.loc[~measured["has_plot_values"], "candidate_id"].tolist()
        raise ValueError(f"Dippers still missing valid depth/timescale values: {missing}")
    return measured


def _plot(df: pd.DataFrame, output_dir: Path) -> None:
    plotted = df.loc[df["has_plot_values"]].copy()
    x = plotted["timescale_days"].to_numpy(dtype=float)
    y = plotted["fractional_depth"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 7.3), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        x,
        y,
        s=54,
        marker="o",
        facecolor="#455a54",
        edgecolor="#202724",
        linewidth=0.65,
        alpha=0.78,
        zorder=3,
        label=f"MALCA dippers ($N={len(plotted)}$)",
    )

    guide_color = "#999999"
    for timescale, label in ((1.0, "Day"), (30.0, "Month"), (365.0, "Year")):
        ax.axvline(timescale, color=guide_color, linestyle="--", linewidth=1.15, zorder=1)
        ax.text(
            timescale,
            1.012,
            label,
            color="#777777",
            fontsize=15,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    lower = 10.0 ** np.floor(np.log10(x.min()))
    upper = 10.0 ** np.ceil(np.log10(x.max()))
    ax.set_xscale("log")
    ax.set_xlim(lower, max(upper, 1000.0))
    ax.set_ylim(-0.025, 1.04)
    ax.set_xlabel("Timescale [days]", fontsize=19)
    ax.set_ylabel("Transit Depth Fraction", fontsize=19)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=6, width=1.0, labelsize=13)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#202020")

    legend = ax.legend(loc="upper left", frameon=True, fontsize=12)
    legend.get_frame().set_edgecolor("#bdbdbd")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)

    stem = output_dir / "july1_malca_dipper_depth_timescale"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.review_db.exists():
        raise FileNotFoundError(f"Review database not found: {args.review_db}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured = _measure_plot_values(_read_dippers(args.review_db))
    csv_path = args.output_dir / "july1_malca_dipper_depth_timescale_values.csv"
    measured.to_csv(csv_path, index=False)
    _plot(measured, args.output_dir)

    fallback = measured["timescale_source"].ne("dip_max_run_duration")
    summary = {
        "review_db": str(args.review_db),
        "cohort_query": "reviewed reviews.event_class = dipper",
        "n_dippers": int(len(measured)),
        "n_plotted": int(measured["has_plot_values"].sum()),
        "n_stored_durations": int((~fallback).sum()),
        "n_fallback_durations": int(fallback.sum()),
        "fallback_candidate_ids": measured.loc[fallback, "candidate_id"].tolist(),
        "fractional_depth_min": float(measured["fractional_depth"].min()),
        "fractional_depth_median": float(measured["fractional_depth"].median()),
        "fractional_depth_max": float(measured["fractional_depth"].max()),
        "timescale_days_min": float(measured["timescale_days"].min()),
        "timescale_days_median": float(measured["timescale_days"].median()),
        "timescale_days_max": float(measured["timescale_days"].max()),
        "depth_definition": "1 - 10**(-0.4 * dip_best_mag_event)",
        "timescale_definition": "dip_max_run_duration with binned-lightcurve fallback when missing",
        "fallback_parameters": {
            "bin_days": FALLBACK_BIN_DAYS,
            "max_bin_gap_days": FALLBACK_MAX_BIN_GAP_DAYS,
            "depth_threshold_fraction": FALLBACK_DEPTH_THRESHOLD_FRACTION,
            "minimum_peak_fraction": FALLBACK_MIN_PEAK_FRACTION,
            "minimum_points_per_bin": FALLBACK_MIN_POINTS_PER_BIN,
        },
        "output_dir": str(args.output_dir),
    }
    summary_path = args.output_dir / "july1_malca_dipper_depth_timescale_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"Wrote depth-timescale plot for {summary['n_plotted']}/{summary['n_dippers']} "
        f"reviewed dippers to {args.output_dir}"
    )
    if summary["n_fallback_durations"]:
        print(
            "Used binned-lightcurve fallback durations for: "
            + ", ".join(summary["fallback_candidate_ids"])
        )


if __name__ == "__main__":
    main()
