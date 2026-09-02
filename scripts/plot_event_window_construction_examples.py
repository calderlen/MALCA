#!/usr/bin/env python
"""Plot stacked event-window constructions at two-column width.

Each requested light curve is written to its own PDF and PNG. The plot uses
the live recovery-anchored event-window and persistent half-depth measurement,
with the full light curve above an event-window zoom. It does not reproduce
those boundaries approximately from cached values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from malca.core.baseline import per_camera_gp_baseline_masked
from malca.core.utils import clean_lc
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame
from malca.plotting.lightcurve_publication import PUBLICATION_STYLE
from malca.review.coordinate_labels import format_j_designation
from malca.stv.dimming_window import (
    DEFAULT_DIMMING_WINDOW_CONFIG,
    DIMMING_WINDOW_METHOD_VERSION,
    dimming_complex_zoom_bounds,
)

try:
    from scripts.plot_all_dipper_diagnostics import (
        EVENT_METRICS_SCHEMA_VERSION,
        FWHM_METHOD_VERSION,
        _relative_flux_from_residual,
        measure_half_depth_event,
        read_all_dippers,
    )
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from plot_all_dipper_diagnostics import (  # type: ignore[no-redef]
        EVENT_METRICS_SCHEMA_VERSION,
        FWHM_METHOD_VERSION,
        _relative_flux_from_residual,
        measure_half_depth_event,
        read_all_dippers,
    )


DEFAULT_RUN_ROOT = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_REVIEW_DB = DEFAULT_RUN_ROOT / "review" / "review.db"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "results" / "event_window_construction_examples"
TIME_OFFSET = 2_458_000.0
FIGURE_WIDTH_INCHES = 7.2
FIGURE_HEIGHT_INCHES = 3.2

OBS_COLOR = "#8A8A8A"
EPOCH_COLOR = "#111111"
PROFILE_COLOR = "#009E73"
PEAK_COLOR = "#E69F00"
WINDOW_COLOR = "#6A3D9A"
WINDOW_FILL_COLOR = "#CAB2D6"
FWHM_COLOR = "#0072B2"
FWHM_FILL_COLOR = "#56B4E9"
DEPTH_COLOR = "#CC4C02"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _json_float(value: Any) -> float | None:
    number = _finite(value)
    return number if np.isfinite(number) else None


def _fwhm_core(
    measurement: dict[str, Any],
    trace: dict[str, Any],
) -> tuple[float, float]:
    """Return the observed half-depth core used by the duration estimator."""
    epoch_times = trace["epochs"]["t"].to_numpy(float)
    left_status = str(measurement["left_crossing_status"])
    right_status = str(measurement["right_crossing_status"])
    left_bounds = trace["left_crossing_bounds"]
    right_bounds = trace["right_crossing_bounds"]

    if left_status == "exact":
        left = _finite(trace["left_crossing_time"])
    elif left_status == "interval":
        left = _finite(left_bounds[1])
    else:
        left = float(epoch_times[int(trace["left_inside"])])
    if right_status == "exact":
        right = _finite(trace["right_crossing_time"])
    elif right_status == "interval":
        right = _finite(right_bounds[0])
    else:
        right = float(epoch_times[int(trace["right_inside"])])

    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        peak = float(measurement["peak_jd"])
        width = float(measurement["duration_plot_days"])
        left, right = peak - 0.5 * width, peak + 0.5 * width
    return left, right


def _flux_limits(values: np.ndarray, peak_flux: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lower = float(np.nanpercentile(finite, 0.5)) if finite.size else peak_flux
    upper = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
    lower = min(lower, peak_flux)
    upper = max(upper, 1.0)
    span = max(upper - lower, 0.08)
    return max(0.0, lower - 0.06 * span), upper + 0.09 * span


def _window_label(measurement: dict[str, Any]) -> str:
    relation = r"\geq" if bool(measurement["event_window_is_lower_limit"]) else "="
    duration = float(measurement["event_window_duration_days"])
    return rf"$T_{{\rm window}}{relation}{duration:.1f}\,\rm d$"


def _fwhm_label(measurement: dict[str, Any]) -> str:
    if bool(measurement["duration_is_interval_censored"]):
        lower = float(measurement["duration_lower_days"])
        upper = float(measurement["duration_upper_days"])
        return rf"$T_{{\rm window,FWHM}}\in[{lower:.1f},\,{upper:.1f}]\,\rm d$"
    if bool(measurement["duration_is_lower_limit"]):
        duration = float(measurement["duration_plot_days"])
        return rf"$T_{{\rm window,FWHM}}\geq{duration:.1f}\,\rm d$"
    duration = float(measurement["duration_plot_days"])
    return rf"$T_{{\rm window,FWHM}}={duration:.1f}\,\rm d$"


def _depth_label(measurement: dict[str, Any]) -> str:
    depth = float(measurement["tau_peak"])
    err_minus = _finite(measurement.get("tau_peak_mc_err_minus"))
    err_plus = _finite(measurement.get("tau_peak_mc_err_plus"))
    if np.isfinite(err_minus) and np.isfinite(err_plus):
        return rf"$\delta={depth:.3f}_{{-{err_minus:.3f}}}^{{+{err_plus:.3f}}}$"
    return rf"$\delta={depth:.3f}$"


def _coordinate_title(candidate: Any) -> str:
    """Return a source title derived only from the stored sky coordinates."""
    ra_deg = _finite(candidate.ra)
    dec_deg = _finite(candidate.dec)
    if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
        raise ValueError(f"Missing finite RA/Dec for {candidate.candidate_id}")
    return format_j_designation(ra_deg, dec_deg)


def _add_window_legend(ax: plt.Axes, measurement: dict[str, Any]) -> None:
    """Add the two duration encodings in a compact publication legend."""
    handles = [
        Patch(
            facecolor=WINDOW_FILL_COLOR,
            edgecolor=WINDOW_COLOR,
            alpha=0.65,
            label=_window_label(measurement),
        ),
        Patch(
            facecolor=FWHM_FILL_COLOR,
            edgecolor=FWHM_COLOR,
            alpha=0.55,
            label=_fwhm_label(measurement),
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderpad=0.30,
        labelspacing=0.14,
        handlelength=1.25,
        handleheight=0.65,
        handletextpad=0.45,
        borderaxespad=0.55,
        fontsize=9.0,
    )
    legend.get_frame().set_linewidth(0.8)


def _add_unavailable_window_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(
            facecolor=WINDOW_FILL_COLOR,
            edgecolor=WINDOW_COLOR,
            alpha=0.65,
            label=r"$T_{\rm window}$ unavailable",
        ),
        Patch(
            facecolor=FWHM_FILL_COLOR,
            edgecolor=FWHM_COLOR,
            alpha=0.55,
            label=r"$T_{\rm window,FWHM}$ unavailable",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderpad=0.30,
        labelspacing=0.14,
        handlelength=1.25,
        handleheight=0.65,
        handletextpad=0.45,
        borderaxespad=0.55,
        fontsize=9.0,
    )
    legend.get_frame().set_linewidth(0.8)


def _draw_depth_measurement(
    ax: plt.Axes,
    measurement: dict[str, Any],
    *,
    peak_x: float,
    peak_flux: float,
    xmin: float,
    xmax: float,
) -> None:
    """Show the fractional transit depth from the fitted baseline to the peak."""
    err_minus = _finite(measurement.get("tau_peak_mc_err_minus"))
    err_plus = _finite(measurement.get("tau_peak_mc_err_plus"))
    if np.isfinite(err_minus) and np.isfinite(err_plus):
        ax.errorbar(
            [peak_x],
            [peak_flux],
            yerr=np.array([[err_plus], [err_minus]]),
            fmt="none",
            ecolor=DEPTH_COLOR,
            elinewidth=0.9,
            capsize=2.0,
            zorder=4.2,
        )
    ax.annotate(
        "",
        xy=(peak_x, 1.0),
        xytext=(peak_x, peak_flux),
        arrowprops={"arrowstyle": "<->", "color": DEPTH_COLOR, "linewidth": 1.35},
        zorder=4.4,
    )
    label_on_left = peak_x > 0.5 * (xmin + xmax)
    depth_text = ax.annotate(
        _depth_label(measurement),
        xy=(peak_x, peak_flux + 0.58 * (1.0 - peak_flux)),
        xytext=(-8 if label_on_left else 8, 0),
        textcoords="offset points",
        ha="right" if label_on_left else "left",
        va="center",
        color="#8C2D04",
        fontsize=8.3,
        zorder=4.6,
    )
    depth_text.set_path_effects(
        [path_effects.withStroke(linewidth=1.0, foreground="white")]
    )


def _plot_style() -> dict[str, Any]:
    return {
        **PUBLICATION_STYLE,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
    }


def _output_paths(output_dir: Path, display_id: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"july1_dipper_event_window_construction_{display_id}"
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def _prepare_fallback_trace(lc_path: Path) -> dict[str, pd.DataFrame]:
    """Prepare the baseline-normalized full curve even when selection fails."""
    config = DEFAULT_DIMMING_WINDOW_CONFIG
    canonical = load_lightcurve_df(
        lc_path,
        filter_bad_cameras_enabled=True,
        apply_quality=True,
    )
    analysis = clean_lc(to_asassn_algorithm_frame(canonical))
    baseline = per_camera_gp_baseline_masked(
        analysis,
        S0=config.gp_s0,
        w0=config.gp_w0,
        q=config.gp_q,
        jitter=config.gp_jitter,
    )
    observations = pd.DataFrame(
        {
            "t": pd.to_numeric(baseline["JD"], errors="coerce"),
            "resid": pd.to_numeric(baseline["resid"], errors="coerce"),
            "sigma": pd.to_numeric(baseline["sigma_eff"], errors="coerce"),
        }
    ).dropna()
    observations = observations.loc[observations["sigma"] > 0].sort_values("t")
    if observations.empty:
        raise RuntimeError("no finite baseline residuals for fallback plot")
    observations["night"] = np.floor(observations["t"]).astype(int)
    epochs = (
        observations.groupby("night", sort=True)
        .agg(t=("t", "median"), resid=("resid", "median"), sigma=("sigma", "median"))
        .reset_index(drop=True)
    )
    return {"observations": observations, "epochs": epochs}


def _plot_full_light_curve(
    candidate: Any,
    measurement: dict[str, Any],
    trace: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    observations = trace["observations"]
    epochs = trace["epochs"]
    obs_times = observations["t"].to_numpy(float)
    epoch_times = epochs["t"].to_numpy(float)
    obs_x = obs_times - TIME_OFFSET
    epoch_x = epoch_times - TIME_OFFSET
    obs_flux = _relative_flux_from_residual(observations["resid"].to_numpy(float))
    epoch_flux = _relative_flux_from_residual(epochs["resid"].to_numpy(float))
    profile_flux = _relative_flux_from_residual(trace["corrected_smoothed_residual"])

    event_left = float(measurement["event_window_start_jd"]) - TIME_OFFSET
    event_right = float(measurement["event_window_end_jd"]) - TIME_OFFSET
    fwhm_left, fwhm_right = _fwhm_core(measurement, trace)
    fwhm_left -= TIME_OFFSET
    fwhm_right -= TIME_OFFSET
    peak_x = float(measurement["peak_jd"]) - TIME_OFFSET
    peak_flux = float(1.0 - measurement["tau_peak"])
    zoom_left_jd, zoom_right_jd = dimming_complex_zoom_bounds(
        epoch_times,
        start_jd=float(measurement["event_window_start_jd"]),
        end_jd=float(measurement["event_window_end_jd"]),
        peak_jd=float(measurement["peak_jd"]),
        cadence_days=float(measurement["cadence_days"]),
    )
    zoom_left = zoom_left_jd - TIME_OFFSET
    zoom_right = zoom_right_jd - TIME_OFFSET

    with plt.rc_context(_plot_style()):
        fig, (full_ax, zoom_ax) = plt.subplots(
            2,
            1,
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
            layout="constrained",
            gridspec_kw={"height_ratios": [0.86, 1.0]},
        )
        gap_limit = float(measurement["crossing_gap_limit_days"])
        starts = np.r_[0, np.flatnonzero(np.diff(epoch_times) > gap_limit) + 1]
        stops = np.r_[starts[1:], len(epoch_times)]

        for ax, is_zoom in ((full_ax, False), (zoom_ax, True)):
            ax.axvspan(
                event_left,
                event_right,
                color=WINDOW_FILL_COLOR,
                alpha=0.20,
                linewidth=0,
                zorder=0,
            )
            ax.axvline(event_left, color=WINDOW_COLOR, linewidth=0.9, zorder=1)
            ax.axvline(event_right, color=WINDOW_COLOR, linewidth=0.9, zorder=1)
            ax.axvspan(
                fwhm_left,
                fwhm_right,
                color=FWHM_FILL_COLOR,
                alpha=0.20,
                linewidth=0,
                zorder=0.4,
            )
            ax.axvline(fwhm_left, color=FWHM_COLOR, linewidth=0.95, zorder=1.2)
            ax.axvline(fwhm_right, color=FWHM_COLOR, linewidth=0.95, zorder=1.2)

            observation_mask = (
                (obs_x >= zoom_left) & (obs_x <= zoom_right)
                if is_zoom
                else np.ones(len(obs_x), dtype=bool)
            )
            epoch_mask = (
                (epoch_x >= zoom_left) & (epoch_x <= zoom_right)
                if is_zoom
                else np.ones(len(epoch_x), dtype=bool)
            )
            ax.scatter(
                obs_x[observation_mask],
                obs_flux[observation_mask],
                s=3.8 if is_zoom else 2.4,
                color=OBS_COLOR,
                alpha=0.30 if is_zoom else 0.24,
                linewidths=0,
                rasterized=True,
                zorder=1.5,
            )
            ax.scatter(
                epoch_x[epoch_mask],
                epoch_flux[epoch_mask],
                s=11.0 if is_zoom else 7.5,
                color=EPOCH_COLOR,
                linewidths=0,
                zorder=2.5,
            )
            for start, stop in zip(starts, stops):
                segment_mask = epoch_mask[start:stop]
                if not np.any(segment_mask):
                    continue
                ax.plot(
                    epoch_x[start:stop][segment_mask],
                    profile_flux[start:stop][segment_mask],
                    color=PROFILE_COLOR,
                    linewidth=0.85 if is_zoom else 0.70,
                    alpha=0.9,
                    zorder=2,
                )
            ax.axhline(
                1.0,
                color="#555555",
                linestyle=(0, (3, 2)),
                linewidth=0.7,
                zorder=0.8,
            )
            ax.scatter(
                peak_x,
                peak_flux,
                marker="*",
                s=70 if is_zoom else 48,
                facecolor=PEAK_COLOR,
                edgecolor=EPOCH_COLOR,
                linewidth=0.5,
                zorder=4,
            )
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.tick_params(which="major", length=3.5, width=0.75)
            ax.tick_params(which="minor", length=1.8, width=0.55)
            ax.minorticks_on()

        xmin = float(np.nanmin(obs_x))
        xmax = float(np.nanmax(obs_x))
        full_ax.set_xlim(xmin, xmax)
        full_ax.set_ylim(*_flux_limits(obs_flux, peak_flux))

        zoom_flux = obs_flux[(obs_x >= zoom_left) & (obs_x <= zoom_right)]
        zoom_ax.set_xlim(zoom_left, zoom_right)
        zoom_ax.set_ylim(*_flux_limits(zoom_flux, peak_flux))
        _draw_depth_measurement(
            zoom_ax,
            measurement,
            peak_x=peak_x,
            peak_flux=peak_flux,
            xmin=zoom_left,
            xmax=zoom_right,
        )

        display_id = str(candidate.asas_sn_id or candidate.candidate_id)
        full_ax.set_title(_coordinate_title(candidate), loc="left", pad=2.0)
        fig.supxlabel(r"JD $-\,2458000$ [d]")
        fig.supylabel("Relative flux")
        _add_window_legend(zoom_ax, measurement)

        png_path, pdf_path = _output_paths(output_dir, display_id)
        fig.savefig(png_path, dpi=450)
        fig.savefig(pdf_path)
        plt.close(fig)
    return png_path, pdf_path


def _plot_unavailable_full_light_curve(
    candidate: Any,
    measurement: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Retain an unavailable cohort member without inventing a zoom window."""
    fallback = _prepare_fallback_trace(Path(str(candidate.lc_path)).expanduser())
    observations = fallback["observations"]
    epochs = fallback["epochs"]
    obs_x = observations["t"].to_numpy(float) - TIME_OFFSET
    epoch_x = epochs["t"].to_numpy(float) - TIME_OFFSET
    obs_flux = _relative_flux_from_residual(observations["resid"].to_numpy(float))
    epoch_flux = _relative_flux_from_residual(epochs["resid"].to_numpy(float))
    deepest_flux = float(np.nanmin(epoch_flux))

    with plt.rc_context(_plot_style()):
        fig, (full_ax, zoom_ax) = plt.subplots(
            2,
            1,
            figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
            layout="constrained",
            gridspec_kw={"height_ratios": [0.86, 1.0]},
        )
        full_ax.scatter(
            obs_x,
            obs_flux,
            s=2.4,
            color=OBS_COLOR,
            alpha=0.24,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        full_ax.scatter(
            epoch_x,
            epoch_flux,
            s=7.5,
            color=EPOCH_COLOR,
            linewidths=0,
            zorder=2,
        )
        full_ax.axhline(
            1.0,
            color="#555555",
            linestyle=(0, (3, 2)),
            linewidth=0.7,
            zorder=0.8,
        )
        full_ax.set_xlim(float(np.nanmin(obs_x)), float(np.nanmax(obs_x)))
        full_ax.set_ylim(*_flux_limits(obs_flux, deepest_flux))
        display_id = str(candidate.asas_sn_id or candidate.candidate_id)
        full_ax.set_title(
            _coordinate_title(candidate),
            loc="left",
            pad=2.0,
        )
        full_ax.tick_params(which="both", direction="in", top=True, right=True)
        full_ax.tick_params(which="major", length=3.5, width=0.75)
        full_ax.tick_params(which="minor", length=1.8, width=0.55)
        full_ax.minorticks_on()
        _add_unavailable_window_legend(full_ax)

        zoom_ax.axis("off")
        zoom_ax.text(
            0.5,
            0.5,
            "Event-window zoom unavailable\n"
            "No supported recovery-anchored event window",
            transform=zoom_ax.transAxes,
            ha="center",
            va="center",
            color="#B2182B",
            fontsize=9.0,
        )
        fig.supxlabel(r"JD $-\,2458000$ [d]")
        fig.supylabel("Relative flux")
        png_path, pdf_path = _output_paths(output_dir, display_id)
        fig.savefig(png_path, dpi=450)
        fig.savefig(pdf_path)
        plt.close(fig)
    return png_path, pdf_path


def _select_cohort(
    review_db: Path,
    candidate_ids: list[str] | None,
) -> pd.DataFrame:
    cohort = read_all_dippers(review_db)
    cohort["candidate_id"] = cohort["candidate_id"].astype(str)
    if candidate_ids is None:
        return cohort.sort_values("candidate_id").reset_index(drop=True)
    indexed = cohort.set_index("candidate_id", drop=False)
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in indexed.index]
    if missing:
        raise ValueError(f"Requested candidates are not live reviewed dippers: {missing}")
    return pd.DataFrame([indexed.loc[candidate_id] for candidate_id in candidate_ids])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-id",
        action="append",
        dest="candidate_ids",
        help=(
            "Optional live reviewed Dipper candidate ID; repeat to restrict the "
            "export. Omit to plot the complete live reviewed-Dipper cohort."
        ),
    )
    args = parser.parse_args()

    review_db = args.review_db.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not review_db.is_file():
        raise FileNotFoundError(f"Review database not found: {review_db}")
    cohort = _select_cohort(review_db, args.candidate_ids)

    outputs = []
    for index, candidate in enumerate(cohort.itertuples(index=False), start=1):
        candidate_id = str(candidate.candidate_id)
        measurement = measure_half_depth_event(
            candidate_id,
            Path(str(candidate.lc_path)).expanduser(),
            include_trace=True,
        )
        trace = measurement.pop("_trace", None)
        if trace is None:
            png_path, pdf_path = _plot_unavailable_full_light_curve(
                candidate,
                measurement,
                output_dir,
            )
        else:
            png_path, pdf_path = _plot_full_light_curve(
                candidate,
                measurement,
                trace,
                output_dir,
            )
        outputs.append(
            {
                "candidate_id": candidate_id,
                "asas_sn_id": str(candidate.asas_sn_id),
                "ra_deg": _json_float(candidate.ra),
                "dec_deg": _json_float(candidate.dec),
                "coordinate_title": _coordinate_title(candidate),
                "lightcurve_path": str(Path(str(candidate.lc_path)).resolve()),
                "lightcurve_sha256": _sha256(Path(str(candidate.lc_path)).resolve()),
                "measurement_error": str(measurement["measurement_error"]),
                "event_window_status": str(measurement["event_window_status"]),
                "event_window_duration_days": _json_float(
                    measurement["event_window_duration_days"]
                ),
                "duration_status": str(measurement["duration_status"]),
                "duration_lower_days": _json_float(measurement.get("duration_lower_days")),
                "duration_upper_days": _json_float(measurement.get("duration_upper_days")),
                "duration_plot_days": _json_float(measurement.get("duration_plot_days")),
                "tau_peak": _json_float(measurement.get("tau_peak")),
                "tau_peak_mc_err_minus": _json_float(
                    measurement.get("tau_peak_mc_err_minus")
                ),
                "tau_peak_mc_err_plus": _json_float(
                    measurement.get("tau_peak_mc_err_plus")
                ),
                "png": str(png_path),
                "pdf": str(pdf_path),
            }
        )
        print(
            f"[{index:03d}/{len(cohort):03d}] {candidate_id}: "
            f"{measurement['event_window_status']}",
            flush=True,
        )

    summary = {
        "review_db": str(review_db),
        "review_db_sha256": _sha256(review_db),
        "cohort_query": "lower(trim(reviews.event_class)) = 'dipper'",
        "candidate_ids": cohort["candidate_id"].astype(str).tolist(),
        "n_reviewed_dippers": int(len(cohort)),
        "n_plot_files": int(len(outputs)),
        "n_measured": int(
            sum(item["event_window_status"] != "measurement_failed" for item in outputs)
        ),
        "n_unavailable": int(
            sum(item["event_window_status"] == "measurement_failed" for item in outputs)
        ),
        "figure_width_inches": FIGURE_WIDTH_INCHES,
        "figure_height_inches": FIGURE_HEIGHT_INCHES,
        "plot_scope": "stacked_full_light_curve_and_event_window_zoom",
        "panel_titles": {
            "top": "Jhhmmss+ddmmss derived from candidates.ra and candidates.dec",
            "bottom": None,
        },
        "legend": {
            "location": "best",
            "frame_alpha": 1.0,
            "frame_edgecolor": "black",
            "compact_vertical_spacing": True,
        },
        "event_window_style": {
            "fill_color": WINDOW_FILL_COLOR,
            "fill_alpha": 0.20,
            "boundary_color": WINDOW_COLOR,
        },
        "fwhm_window_style": {
            "fill_color": FWHM_FILL_COLOR,
            "fill_alpha": 0.20,
            "boundary_color": FWHM_COLOR,
        },
        "depth_annotation": (
            "unboxed fractional_flux_depth_tau_peak_with_mc68_uncertainty"
        ),
        "dimming_window_method_version": DIMMING_WINDOW_METHOD_VERSION,
        "fwhm_method_version": FWHM_METHOD_VERSION,
        "event_metrics_schema_version": EVENT_METRICS_SCHEMA_VERSION,
        "outputs": outputs,
    }
    summary_path = output_dir / "july1_dipper_event_window_construction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(
        f"Wrote {len(outputs)} separate stacked event-window plots "
        f"for {len(cohort)} reviewed Dippers"
    )


if __name__ == "__main__":
    main()
