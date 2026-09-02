#!/usr/bin/env python
"""Plot Event Window Construction depth versus FWHM for July 1 dippers.

This plotter consumes the provenance-rich metrics written by
``scripts/plot_all_dipper_diagnostics.py``.  It refuses stale or partial metric
tables: candidate IDs must exactly match the live ``event_class='dipper'``
cohort, and the saved dimming-window/FWHM method versions must match the current
implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
)

from malca.plotting.lightcurve_publication import (
    FIG_SINGLE_COL_WIDTH,
    PUBLICATION_STYLE,
)
from malca.stv.dimming_window import DIMMING_WINDOW_METHOD_VERSION


DEFAULT_RUN_ROOT = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_REVIEW_DB = DEFAULT_RUN_ROOT / "review" / "review.db"
DEFAULT_METRICS_CSV = Path(
    "output/pdf/all_dipper_diagnostics/"
    "all_dippers_half_depth_diagnostic_atlas_with_pipeline_runs_metrics.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "results" / "dipper_event_window_depth_timescale"
DEFAULT_SMPLOTLIB_OUTPUT_DIR = (
    DEFAULT_RUN_ROOT / "results" / "dipper_event_window_depth_timescale_smplotlib"
)
PLOT_STYLES = ("malca", "smplotlib")

EXPECTED_FWHM_METHOD_VERSION = "persistent_half_depth_5of6_7d_v1"
EXPECTED_EVENT_METRICS_SCHEMA_VERSION = "dimming_complex_duration_v1"

RESOLVED_COLOR = "#0072B2"
INTERVAL_COLOR = "#E69F00"
LIMIT_COLOR = "#009E73"
NEUTRAL_COLOR = "#333333"
BLACK_COLOR = "#000000"

SINGLE_COLUMN_FIGURE_SIZE = (FIG_SINGLE_COL_WIDTH, 3.15)
SINGLE_COLUMN_STYLE = {
    "font.size": 9.0,
    "axes.labelsize": 11.0,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.size": 2.7,
    "ytick.minor.size": 2.7,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
}


plt.rcParams.update(
    {
        **PUBLICATION_STYLE,
        **SINGLE_COLUMN_STYLE,
        "axes.formatter.use_mathtext": True,
    }
)


def _apply_plot_style(style: str) -> str | None:
    """Apply the requested style and return its package version, if any."""
    if style == "malca":
        plt.rcParams.update(
            {
                **PUBLICATION_STYLE,
                **SINGLE_COLUMN_STYLE,
                "axes.formatter.use_mathtext": True,
            }
        )
        return None
    if style == "smplotlib":
        import smplotlib

        style_path = Path(smplotlib.__file__).resolve().parent / "smplot.mplstyle"
        plt.style.use(style_path)
        smplotlib.set_style()
        plt.rcParams.update(SINGLE_COLUMN_STYLE)
        plt.rcParams["savefig.bbox"] = None
        return str(getattr(smplotlib, "__version__", "unknown"))
    raise ValueError(f"Unsupported plot style: {style}")


def _read_live_dippers(review_db: Path) -> pd.DataFrame:
    """Read the complete live Dipper cohort without mutating the WAL database."""
    query = """
        SELECT
            c.candidate_id,
            c.asas_sn_id,
            c.lc_path,
            r.event_class,
            r.status,
            r.workflow_status,
            r.disposition
        FROM candidates AS c
        JOIN reviews AS r USING(candidate_id)
        WHERE lower(trim(coalesce(r.event_class, ''))) = 'dipper'
        ORDER BY c.candidate_id
    """
    uri = f"file:{review_db.resolve().as_posix()}?immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        frame = pd.read_sql_query(query, conn)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.empty:
        raise RuntimeError(f"No Dippers were found in {review_db}")
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("The live Dipper query returned duplicate candidate IDs")
    return frame


def _validate_single_version(
    metrics: pd.DataFrame,
    column: str,
    expected: str,
) -> None:
    if column not in metrics:
        raise ValueError(f"Metrics table is missing required column {column!r}")
    actual = set(metrics[column].dropna().astype(str))
    if actual != {expected}:
        raise ValueError(
            f"Metrics table has {column}={sorted(actual)!r}; expected {expected!r}"
        )


def _load_plot_values(review_db: Path, metrics_csv: Path) -> pd.DataFrame:
    cohort = _read_live_dippers(review_db)
    metrics = pd.read_csv(metrics_csv, dtype={"candidate_id": str})
    if metrics["candidate_id"].duplicated().any():
        raise ValueError("Event-window metrics contain duplicate candidate IDs")

    cohort_ids = set(cohort["candidate_id"])
    metric_ids = set(metrics["candidate_id"])
    missing = sorted(cohort_ids - metric_ids)
    stale = sorted(metric_ids - cohort_ids)
    if missing or stale:
        raise ValueError(
            "Event-window metrics do not match the live Dipper cohort: "
            f"missing={missing}, stale={stale}"
        )

    _validate_single_version(
        metrics,
        "dimming_window_method_version",
        DIMMING_WINDOW_METHOD_VERSION,
    )
    _validate_single_version(
        metrics,
        "fwhm_method_version",
        EXPECTED_FWHM_METHOD_VERSION,
    )
    _validate_single_version(
        metrics,
        "event_metrics_schema_version",
        EXPECTED_EVENT_METRICS_SCHEMA_VERSION,
    )

    required = {
        "tau_peak",
        "delta_mag_peak",
        "dimming_complex_duration_lower_days",
        "dimming_complex_duration_upper_days",
        "dimming_complex_duration_plot_days",
        "dimming_complex_is_lower_limit",
        "duration_status",
        "duration_lower_days",
        "duration_upper_days",
        "duration_plot_days",
        "duration_is_lower_limit",
        "measurement_error",
    }
    absent = sorted(required - set(metrics.columns))
    if absent:
        raise ValueError(f"Event-window metrics are missing columns: {absent}")

    values = cohort.merge(metrics, on="candidate_id", how="left", validate="one_to_one")
    numeric = [
        "tau_peak",
        "delta_mag_peak",
        "dimming_complex_duration_lower_days",
        "dimming_complex_duration_upper_days",
        "dimming_complex_duration_plot_days",
        "duration_lower_days",
        "duration_upper_days",
        "duration_plot_days",
        "tau_peak_mc_err_minus",
        "tau_peak_mc_err_plus",
        "duration_mc_err_minus",
        "duration_mc_err_plus",
    ]
    for column in numeric:
        if column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce")

    valid_depth = (
        np.isfinite(values["tau_peak"])
        & values["tau_peak"].gt(0)
        & values["tau_peak"].lt(1)
    )
    values["fwhm_plot_included"] = (
        valid_depth
        & np.isfinite(values["duration_lower_days"])
        & values["duration_lower_days"].gt(0)
        & values["duration_status"].ne("measurement_failed")
    )
    values["full_window_plot_included"] = (
        valid_depth
        & np.isfinite(values["dimming_complex_duration_lower_days"])
        & values["dimming_complex_duration_lower_days"].gt(0)
    )
    # Retain the original column as a backwards-compatible alias for the FWHM
    # product and its summary table.
    values["plot_included"] = values["fwhm_plot_included"]
    values["plot_x_days"] = values["duration_plot_days"]
    limited = values["duration_is_lower_limit"].eq(True)
    values.loc[limited, "plot_x_days"] = values.loc[limited, "duration_lower_days"]
    values["full_window_plot_x_days"] = values[
        "dimming_complex_duration_plot_days"
    ]
    full_window_limited = values["dimming_complex_is_lower_limit"].eq(True)
    values.loc[full_window_limited, "full_window_plot_x_days"] = values.loc[
        full_window_limited, "dimming_complex_duration_lower_days"
    ]
    return values


def _amplitude_yerr(rows: pd.DataFrame) -> np.ndarray:
    minus = pd.to_numeric(
        rows.get("tau_peak_mc_err_minus", pd.Series(0.0, index=rows.index)),
        errors="coerce",
    ).fillna(0.0)
    plus = pd.to_numeric(
        rows.get("tau_peak_mc_err_plus", pd.Series(0.0, index=rows.index)),
        errors="coerce",
    ).fillna(0.0)
    depth = rows["tau_peak"].to_numpy(float)
    return np.vstack(
        [
            np.minimum(np.maximum(minus.to_numpy(float), 0.0), 0.95 * depth),
            np.maximum(plus.to_numpy(float), 0.0),
        ]
    )


def _plot_group(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    color: str,
    marker: str,
    facecolor: str | None = None,
    edgecolor: str = NEUTRAL_COLOR,
    marker_size: float = 20.0,
    marker_linewidth: float = 0.4,
    xerr: np.ndarray | None = None,
) -> None:
    if rows.empty:
        return
    ax.errorbar(
        rows["plot_x_days"].to_numpy(float),
        rows["tau_peak"].to_numpy(float),
        xerr=xerr,
        yerr=_amplitude_yerr(rows),
        fmt="none",
        ecolor=color,
        elinewidth=0.65,
        capsize=1.3,
        capthick=0.65,
        alpha=1.0,
        zorder=2,
    )
    ax.scatter(
        rows["plot_x_days"],
        rows["tau_peak"],
        s=marker_size,
        marker=marker,
        facecolor=color if facecolor is None else facecolor,
        edgecolor=edgecolor,
        linewidth=marker_linewidth,
        alpha=1.0,
        zorder=3,
    )


def _fractional_depth_to_mag(depth: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(depth, dtype=float), 0.0, 1.0 - 1e-9)
    return -2.5 * np.log10(1.0 - values)


def _mag_to_fractional_depth(delta_mag: np.ndarray | float) -> np.ndarray:
    return 1.0 - np.power(10.0, -0.4 * np.asarray(delta_mag, dtype=float))


def _plain_log_decade(value: float, _position: float) -> str:
    """Format the selected log-axis decades as compact plain numbers."""
    for tick in (1.0, 10.0, 100.0, 1000.0):
        if np.isclose(value, tick):
            return str(int(tick))
    return ""


def _compact_decimal_tick(value: float, _position: float) -> str:
    """Keep zero compact while retaining one decimal elsewhere."""
    return "0" if np.isclose(value, 0.0) else f"{value:.1f}"


def _plot(
    values: pd.DataFrame,
    output_dir: Path,
    *,
    plot_style: str,
    x_axis: str,
) -> Path:
    if x_axis == "fwhm":
        plotted = values.loc[values["fwhm_plot_included"]].copy()
        x_column = "plot_x_days"
        upper_column = "duration_upper_days"
        resolved = plotted.loc[plotted["duration_status"].eq("resolved")]
        interval = plotted.loc[plotted["duration_status"].eq("interval_censored")]
        limited = plotted.loc[plotted["duration_is_lower_limit"].eq(True)]

        resolved_xerr = np.zeros((2, len(resolved)), dtype=float)
        reported = resolved.get(
            "duration_mc_reporting_status",
            pd.Series("", index=resolved.index, dtype=object),
        ).eq("reported_resolved")
        if reported.any():
            resolved_xerr[0, reported.to_numpy()] = pd.to_numeric(
                resolved.loc[reported, "duration_mc_err_minus"], errors="coerce"
            ).fillna(0.0)
            resolved_xerr[1, reported.to_numpy()] = pd.to_numeric(
                resolved.loc[reported, "duration_mc_err_plus"], errors="coerce"
            ).fillna(0.0)

        interval_x = interval[x_column].to_numpy(float)
        interval_xerr = np.vstack(
            [
                np.maximum(
                    interval_x
                    - interval["duration_lower_days"].to_numpy(float),
                    0.0,
                ),
                np.maximum(
                    interval["duration_upper_days"].to_numpy(float) - interval_x,
                    0.0,
                ),
            ]
        )
        xlabel = r"$T_{\rm window,FWHM}$ [d]"
        stem_suffix = "fwhm_depth_timescale"
        resolved_color = RESOLVED_COLOR
        resolved_facecolor = RESOLVED_COLOR
        resolved_edgecolor = NEUTRAL_COLOR
        limited_color = LIMIT_COLOR
        limited_facecolor = LIMIT_COLOR
        limited_edgecolor = NEUTRAL_COLOR
        resolved_marker_size = 20.0
        limited_marker_size = 20.0
        limited_marker_linewidth = 0.4
    elif x_axis == "full_window":
        plotted = values.loc[values["full_window_plot_included"]].copy()
        x_column = "full_window_plot_x_days"
        upper_column = "dimming_complex_duration_upper_days"
        full_window_limited = plotted["dimming_complex_is_lower_limit"].eq(True)
        resolved = plotted.loc[~full_window_limited]
        interval = plotted.iloc[0:0]
        limited = plotted.loc[full_window_limited]
        resolved_xerr = None
        interval_xerr = None
        xlabel = r"$T_{\rm window}$ [d]"
        stem_suffix = "full_window_depth_timescale"
        resolved_color = BLACK_COLOR
        resolved_facecolor = BLACK_COLOR
        resolved_edgecolor = BLACK_COLOR
        limited_color = BLACK_COLOR
        # A white face preserves the hollow appearance on the white panel while
        # masking the error-bar segment underneath the triangle.
        limited_facecolor = "white"
        limited_edgecolor = BLACK_COLOR
        resolved_marker_size = 12.0
        limited_marker_size = 15.0
        limited_marker_linewidth = 0.55
    else:
        raise ValueError(f"Unsupported x-axis definition: {x_axis}")

    fig, ax = plt.subplots(figsize=SINGLE_COLUMN_FIGURE_SIZE, layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_box_aspect(1.0)

    resolved_for_plot = resolved.copy()
    resolved_for_plot["plot_x_days"] = resolved_for_plot[x_column]
    _plot_group(
        ax,
        resolved_for_plot,
        color=resolved_color,
        marker="o",
        facecolor=resolved_facecolor,
        edgecolor=resolved_edgecolor,
        marker_size=resolved_marker_size,
        xerr=resolved_xerr,
    )
    interval_for_plot = interval.copy()
    interval_for_plot["plot_x_days"] = interval_for_plot[x_column]
    _plot_group(
        ax,
        interval_for_plot,
        color=INTERVAL_COLOR,
        marker="D",
        xerr=interval_xerr,
    )
    limited_for_plot = limited.copy()
    limited_for_plot["plot_x_days"] = limited_for_plot[x_column]
    _plot_group(
        ax,
        limited_for_plot,
        color=limited_color,
        marker=">",
        facecolor=limited_facecolor,
        edgecolor=limited_edgecolor,
        marker_size=limited_marker_size,
        marker_linewidth=limited_marker_linewidth,
    )

    xmax = 1.75 * float(
        np.nanmax(
            [
                plotted[x_column].max(),
                plotted[upper_column].max(skipna=True),
            ]
        )
    )
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(1500.0, xmax))
    ax.xaxis.set_major_locator(FixedLocator([1.0, 10.0, 100.0, 1000.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_log_decade))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    fractional_depth_at_one_mag = float(_mag_to_fractional_depth(1.0))
    ax.set_ylim(0.0, fractional_depth_at_one_mag)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_decimal_tick))
    ax.set_xlabel(xlabel, fontsize=11, labelpad=2)
    ax.set_ylabel(r"$\delta$", fontsize=11, labelpad=2)
    ax.grid(visible=False, which="both", axis="both")
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=False,
        labelsize=9,
        length=5.0,
        width=0.9,
        pad=2,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=False,
        length=2.7,
        width=0.7,
    )

    secondary = ax.secondary_yaxis(
        "right",
        functions=(_fractional_depth_to_mag, _mag_to_fractional_depth),
    )
    secondary.set_ylabel(r"$\Delta m$ [mag]", fontsize=10, labelpad=3)
    major_mag_ticks = np.linspace(0.0, 1.0, 6)
    minor_mag_ticks = np.asarray(
        [
            value
            for value in np.arange(0.05, 1.0, 0.05)
            if not np.any(np.isclose(value, major_mag_ticks))
        ],
        dtype=float,
    )
    secondary.set_yticks(major_mag_ticks)
    secondary.set_yticks(minor_mag_ticks, minor=True)
    secondary.yaxis.set_major_formatter(FuncFormatter(_compact_decimal_tick))
    secondary.tick_params(
        axis="y",
        which="minor",
        right=True,
        direction="in",
        length=2.7,
        width=0.7,
    )
    secondary.tick_params(
        axis="y",
        which="major",
        right=True,
        direction="in",
        labelsize=9,
        length=5.0,
        width=0.9,
        pad=2,
    )
    for tick_value, tick_label in zip(
        secondary.get_yticks(), secondary.get_yticklabels()
    ):
        if np.isclose(tick_value, 1.0):
            tick_label.set_verticalalignment("top")
            tick_label.set_horizontalalignment("left")
            tick_label.set_y(1.0)

    suffix = "_smplotlib" if plot_style == "smplotlib" else ""
    stem = output_dir / (
        "july1_malca_dipper_event_window_" + stem_suffix + suffix
    )
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-style", choices=PLOT_STYLES, default="malca")
    args = parser.parse_args()

    review_db = args.review_db.expanduser().resolve()
    metrics_csv = args.metrics_csv.expanduser().resolve()
    default_output_dir = (
        DEFAULT_SMPLOTLIB_OUTPUT_DIR
        if args.plot_style == "smplotlib"
        else DEFAULT_OUTPUT_DIR
    )
    output_dir = (
        args.output_dir if args.output_dir is not None else default_output_dir
    ).expanduser().resolve()
    if not review_db.is_file():
        raise FileNotFoundError(f"Review database not found: {review_db}")
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Event-window metrics not found: {metrics_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)

    style_version = _apply_plot_style(args.plot_style)
    values = _load_plot_values(review_db, metrics_csv)
    fwhm_pdf_path = _plot(
        values,
        output_dir,
        plot_style=args.plot_style,
        x_axis="fwhm",
    )
    full_window_pdf_path = _plot(
        values,
        output_dir,
        plot_style=args.plot_style,
        x_axis="full_window",
    )
    suffix = "_smplotlib" if args.plot_style == "smplotlib" else ""
    values_path = output_dir / (
        "july1_malca_dipper_event_window_fwhm_depth_timescale_values" + suffix + ".csv"
    )
    values.to_csv(values_path, index=False)

    plotted = values.loc[values["fwhm_plot_included"]]
    full_window_plotted = values.loc[values["full_window_plot_included"]]
    unavailable = values.loc[~values["fwhm_plot_included"]]
    full_window_unavailable = values.loc[~values["full_window_plot_included"]]
    summary = {
        "review_db": str(review_db),
        "event_window_metrics_csv": str(metrics_csv),
        "event_window_metrics_sha256": _sha256(metrics_csv),
        "cohort_query": "lower(trim(reviews.event_class)) = 'dipper'",
        "n_dippers": int(len(values)),
        "n_measured": int(len(plotted)),
        "n_unavailable": int(len(unavailable)),
        "n_full_window_measured": int(len(full_window_plotted)),
        "n_full_window_unavailable": int(len(full_window_unavailable)),
        "unavailable_candidate_ids": unavailable["candidate_id"].tolist(),
        "unavailable_errors": dict(
            zip(
                unavailable["candidate_id"],
                unavailable["measurement_error"].fillna("").astype(str),
            )
        ),
        "duration_status_counts": {
            str(key): int(value)
            for key, value in values["duration_status"].value_counts(dropna=False).items()
        },
        "fractional_depth_min": _json_float(plotted["tau_peak"].min()),
        "fractional_depth_median": _json_float(plotted["tau_peak"].median()),
        "fractional_depth_max": _json_float(plotted["tau_peak"].max()),
        "fwhm_plot_days_min": _json_float(plotted["plot_x_days"].min()),
        "fwhm_plot_days_median": _json_float(plotted["plot_x_days"].median()),
        "fwhm_plot_days_max": _json_float(plotted["plot_x_days"].max()),
        "full_window_plot_days_min": _json_float(
            full_window_plotted["full_window_plot_x_days"].min()
        ),
        "full_window_plot_days_median": _json_float(
            full_window_plotted["full_window_plot_x_days"].median()
        ),
        "full_window_plot_days_max": _json_float(
            full_window_plotted["full_window_plot_x_days"].max()
        ),
        "depth_definition": "tau_peak = 1 - 10**(-0.4 * delta_mag_peak)",
        "fwhm_timescale_definition": (
            "persistent half-depth FWHM within the selected recovery-anchored "
            "event window; finite gaps are intervals and open sides are lower limits"
        ),
        "full_window_timescale_definition": (
            "full recovery-anchored dimming-complex span; open event-window "
            "boundaries are lower limits"
        ),
        "dimming_window_method_version": DIMMING_WINDOW_METHOD_VERSION,
        "fwhm_method_version": EXPECTED_FWHM_METHOD_VERSION,
        "event_metrics_schema_version": EXPECTED_EVENT_METRICS_SCHEMA_VERSION,
        "plot_style": args.plot_style,
        "smplotlib_version": style_version,
        "rendering": {
            "figure_size_inches": list(SINGLE_COLUMN_FIGURE_SIZE),
            "axes_box_aspect": 1.0,
            "aastex_target": "single_column_square_axes",
            "axis_label_fontsize_points": 11.0,
            "tick_label_fontsize_points": 9.0,
            "secondary_axis_label_fontsize_points": 10.0,
            "major_tick_length_points": 5.0,
            "minor_tick_length_points": 2.7,
            "marker_area_points_squared": 20.0,
            "fwhm_point_style": {
                "resolved": "filled_blue_circle",
                "interval": "filled_orange_diamond",
                "lower_limit": "filled_green_right_triangle",
            },
            "full_window_point_style": {
                "bounded": "small_filled_black_circle",
                "lower_limit": "small_hollow_appearance_black_right_triangle",
            },
            "full_window_marker_area_points_squared": {
                "bounded": 12.0,
                "lower_limit": 15.0,
            },
            "full_window_triangle_errorbar_masked_inside_marker": True,
            "point_alpha": 1.0,
            "errorbar_alpha": 1.0,
            "grid": False,
            "legend": False,
            "lower_limit_arrows": False,
            "lower_limit_marker": ">",
            "right_axis_max_mag": 1.0,
            "left_axis_label": "delta",
            "right_axis_label": "Delta m [mag]",
            "right_axis_minor_tick_interval_mag": 0.05,
            "x_major_tick_labels": ["1", "10", "100", "1000"],
            "x_minor_ticks_per_decade": [2, 5],
            "zero_tick_label": "0",
            "primary_ticks_mirrored_on_right": False,
            "right_axis_major_tick_pad_points": 2,
            "x_axis_min_days": 0.8,
            "x_axis_unit": "d",
            "output_format": "pdf_only",
        },
        "fwhm_pdf": str(fwhm_pdf_path),
        "full_window_pdf": str(full_window_pdf_path),
        "values_csv": str(values_path),
    }
    summary_path = output_dir / (
        "july1_malca_dipper_event_window_fwhm_depth_timescale_summary"
        + suffix
        + ".json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"Wrote FWHM and full-window depth-timescale PDFs for "
        f"{len(plotted)}/{len(values)} Dippers with {args.plot_style} style "
        f"to {output_dir}"
    )
    if len(unavailable):
        print("Unavailable: " + ", ".join(unavailable["candidate_id"]))


if __name__ == "__main__":
    main()
