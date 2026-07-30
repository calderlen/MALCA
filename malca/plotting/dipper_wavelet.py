from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LogNorm

from malca.io.lightcurve_io import load_lightcurve_df
from malca.io.notebook_paths import resolve_local_lightcurve_path
from malca.plotting.lightcurve_publication import PUBLICATION_STYLE


@dataclass(frozen=True)
class WWZConfig:
    """Configuration for an irregularly sampled weighted wavelet Z-transform."""

    min_scale_days: float = 2.0
    max_scale_days: float | None = None
    max_scale_fraction_of_span: float = 0.50
    n_scales: int = 72
    n_time_bins: int = 180
    decay_constant: float = 0.0125
    min_effective_points: float = 10.0
    min_nights: int = 30

    def validate(self) -> None:
        if self.min_scale_days <= 0:
            raise ValueError("min_scale_days must be positive")
        if (
            self.max_scale_days is not None
            and self.max_scale_days <= self.min_scale_days
        ):
            raise ValueError("max_scale_days must be greater than min_scale_days")
        if not 0 < self.max_scale_fraction_of_span <= 0.5:
            raise ValueError("max_scale_fraction_of_span must be in (0, 0.5]")
        if self.n_scales < 8:
            raise ValueError("n_scales must be at least 8")
        if self.n_time_bins < 16:
            raise ValueError("n_time_bins must be at least 16")
        if self.decay_constant <= 0:
            raise ValueError("decay_constant must be positive")
        if self.min_effective_points <= 3:
            raise ValueError("min_effective_points must be greater than 3")
        if self.min_nights < 10:
            raise ValueError("min_nights must be at least 10")


@dataclass(frozen=True)
class WWZResult:
    times: np.ndarray
    relative_flux: np.ndarray
    time_centers: np.ndarray
    scales_days: np.ndarray
    power: np.ndarray
    effective_points: np.ndarray
    edge_reliable: np.ndarray
    global_power: np.ndarray
    dominant_scale_days: float
    dominant_scale_low_days: float
    dominant_scale_high_days: float
    dominant_peak_at_upper_boundary: bool
    dominant_band_hits_upper_boundary: bool
    dominant_scale_edge_reliable_fraction: float
    max_scale_analyzed_days: float
    median_cadence_days: float


def read_review_labeled_candidates(
    review_db: Path | str,
    *,
    event_class: str = "dipper",
) -> pd.DataFrame:
    """Read the current unique Review rows for one event class."""
    db_path = Path(review_db).expanduser().resolve()
    query = """
        SELECT
            c.candidate_id,
            c.asas_sn_id,
            c.lc_path,
            c.ra,
            c.dec,
            c.periodicity_period,
            c.periodicity_method,
            r.event_class,
            r.classification_confidence,
            r.workflow_status,
            r.status
        FROM reviews AS r
        JOIN candidates AS c USING(candidate_id)
        WHERE lower(trim(coalesce(r.event_class, ''))) = lower(trim(?))
        ORDER BY
            coalesce(r.classification_confidence, 0) DESC,
            c.candidate_id
    """
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        candidates = pd.read_sql_query(query, conn, params=(event_class,))

    if candidates.empty:
        raise RuntimeError(f"No {event_class!r} rows were found in {db_path}")
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    if candidates["candidate_id"].duplicated().any():
        duplicates = candidates.loc[
            candidates["candidate_id"].duplicated(keep=False), "candidate_id"
        ].tolist()
        raise RuntimeError(f"Review query returned duplicate candidate IDs: {duplicates[:5]}")
    return candidates


def prepare_relative_flux(
    lightcurve: pd.DataFrame,
    *,
    nightly_bin: bool = True,
) -> pd.DataFrame:
    """Band-normalize a canonical ASAS-SN light curve and optionally bin by night.

    Normalizing each passband independently removes the V/g zero-point change
    without fitting away long-timescale variability.  Nightly medians prevent a
    night with many exposures from dominating the local wavelet regression.
    """
    required = {"jd", "band", "mag"}
    missing = required - set(lightcurve.columns)
    if missing:
        raise ValueError(f"Canonical light curve is missing columns: {sorted(missing)}")

    frame = lightcurve.copy()
    frame["jd"] = pd.to_numeric(frame["jd"], errors="coerce")
    frame["mag"] = pd.to_numeric(frame["mag"], errors="coerce")
    if "mag_err" in frame:
        frame["mag_err"] = pd.to_numeric(frame["mag_err"], errors="coerce")
    else:
        frame["mag_err"] = np.nan
    frame["band"] = (
        frame["band"].astype("string").fillna("").str.strip().replace("", "all")
    )
    frame = frame.loc[np.isfinite(frame["jd"]) & np.isfinite(frame["mag"])].copy()
    if frame.empty:
        raise RuntimeError("No finite time/magnitude observations remain after quality filtering")

    band_median = frame.groupby("band", observed=True)["mag"].transform("median")
    frame["relative_flux"] = np.power(10.0, -0.4 * (frame["mag"] - band_median))
    frame["relative_flux_err"] = (
        np.log(10.0) * 0.4 * frame["relative_flux"] * frame["mag_err"]
    )
    frame = frame.loc[
        np.isfinite(frame["relative_flux"]) & (frame["relative_flux"] > 0)
    ].copy()

    if nightly_bin:
        # JD changes at noon UTC; JD - 0.5 groups a conventional UTC observing night.
        frame["night"] = np.floor(frame["jd"] - 0.5).astype(np.int64)
        prepared = (
            frame.groupby("night", sort=True, observed=True)
            .agg(
                jd=("jd", "median"),
                relative_flux=("relative_flux", "median"),
                relative_flux_err=("relative_flux_err", "median"),
                n_exposures=("relative_flux", "size"),
                bands=("band", lambda values: ",".join(sorted(set(map(str, values))))),
            )
            .reset_index(drop=True)
        )
    else:
        prepared = frame[
            ["jd", "relative_flux", "relative_flux_err", "band"]
        ].rename(columns={"band": "bands"})
        prepared["n_exposures"] = 1

    prepared = prepared.sort_values("jd").reset_index(drop=True)
    scale = float(np.nanmedian(prepared["relative_flux"]))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Relative-flux normalization is not finite and positive")
    prepared["relative_flux"] = prepared["relative_flux"] / scale
    prepared["relative_flux_err"] = prepared["relative_flux_err"] / scale
    return prepared


def weighted_wavelet_z(
    times: np.ndarray,
    values: np.ndarray,
    time_centers: np.ndarray,
    scales_days: np.ndarray,
    *,
    decay_constant: float = 0.0125,
    min_effective_points: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Foster's weighted wavelet Z statistic on irregular observations.

    At each time and trial scale, a constant plus sine/cosine model is fit with
    Gaussian time weights.  The returned statistic is undefined where the local
    effective sample size is too small or the weighted regression is singular.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    tau = np.asarray(time_centers, dtype=float)
    scales = np.asarray(scales_days, dtype=float)

    finite = np.isfinite(t) & np.isfinite(x)
    t = t[finite]
    x = x[finite]
    if t.size < 4:
        raise ValueError("At least four finite observations are required")
    if tau.ndim != 1 or scales.ndim != 1:
        raise ValueError("time_centers and scales_days must be one-dimensional")
    if np.any(scales <= 0):
        raise ValueError("All scales must be positive")

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    x_median = float(np.median(x))
    x_sigma = float(1.4826 * np.median(np.abs(x - x_median)))
    if not np.isfinite(x_sigma) or x_sigma <= 0:
        x_sigma = float(np.std(x))
    if not np.isfinite(x_sigma) or x_sigma <= 0:
        raise ValueError("Input values have no finite variability")
    x = (x - x_median) / x_sigma

    power = np.full((scales.size, tau.size), np.nan, dtype=float)
    effective_points = np.zeros_like(power)
    dt = t[None, :] - tau[:, None]

    for scale_idx, scale in enumerate(scales):
        omega = 2.0 * np.pi / float(scale)
        phase = omega * dt
        weights = np.exp(-float(decay_constant) * np.square(phase))
        weight_sum = weights.sum(axis=1)
        weight_square_sum = np.square(weights).sum(axis=1)
        neff = np.divide(
            np.square(weight_sum),
            weight_square_sum,
            out=np.zeros_like(weight_sum),
            where=weight_square_sum > 0,
        )
        effective_points[scale_idx] = neff

        cosine = np.cos(phase)
        sine = np.sin(phase)
        design = np.stack(
            (np.ones_like(cosine), cosine, sine),
            axis=2,
        )
        normal = np.einsum("tn,tni,tnj->tij", weights, design, design, optimize=True)
        rhs = np.einsum("tn,tni,n->ti", weights, design, x, optimize=True)

        trace = np.trace(normal, axis1=1, axis2=2)
        ridge = np.maximum(trace, 1.0) * 1e-12
        normal[:, 0, 0] += ridge
        normal[:, 1, 1] += ridge
        normal[:, 2, 2] += ridge
        try:
            coefficients = np.linalg.solve(normal, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficients = np.full_like(rhs, np.nan)
            for idx in range(tau.size):
                coefficients[idx] = np.linalg.lstsq(
                    normal[idx],
                    rhs[idx],
                    rcond=None,
                )[0]

        model = np.einsum("tni,ti->tn", design, coefficients, optimize=True)
        mean_x = np.divide(
            weights @ x,
            weight_sum,
            out=np.full_like(weight_sum, np.nan),
            where=weight_sum > 0,
        )
        mean_model = np.divide(
            np.sum(weights * model, axis=1),
            weight_sum,
            out=np.full_like(weight_sum, np.nan),
            where=weight_sum > 0,
        )
        variance_x = np.divide(
            np.sum(weights * np.square(x[None, :] - mean_x[:, None]), axis=1),
            weight_sum,
            out=np.full_like(weight_sum, np.nan),
            where=weight_sum > 0,
        )
        variance_model = np.divide(
            np.sum(weights * np.square(model - mean_model[:, None]), axis=1),
            weight_sum,
            out=np.full_like(weight_sum, np.nan),
            where=weight_sum > 0,
        )
        residual_variance = variance_x - variance_model
        good = (
            (neff >= float(min_effective_points))
            & np.isfinite(variance_x)
            & np.isfinite(variance_model)
            & (variance_x > 0)
            & (variance_model >= 0)
            & (residual_variance > np.maximum(1e-12, variance_x * 1e-10))
        )
        power[scale_idx, good] = (
            (neff[good] - 3.0)
            * variance_model[good]
            / (2.0 * residual_variance[good])
        )

    power[~np.isfinite(power) | (power <= 0)] = np.nan
    return power, effective_points


def _dominant_scale_band(
    scales_days: np.ndarray,
    global_power: np.ndarray,
) -> tuple[float, float, float, bool, bool]:
    finite = np.isfinite(global_power) & (global_power > 0)
    if not finite.any():
        return np.nan, np.nan, np.nan, False, False

    scores = np.where(finite, global_power, -np.inf)
    peak_idx = int(np.argmax(scores))
    last_valid_idx = int(np.flatnonzero(finite)[-1])
    peak = float(global_power[peak_idx])
    threshold = 0.5 * peak

    low_idx = peak_idx
    while low_idx > 0 and np.isfinite(global_power[low_idx - 1]) and global_power[low_idx - 1] >= threshold:
        low_idx -= 1
    high_idx = peak_idx
    while (
        high_idx < len(global_power) - 1
        and np.isfinite(global_power[high_idx + 1])
        and global_power[high_idx + 1] >= threshold
    ):
        high_idx += 1
    return (
        float(scales_days[peak_idx]),
        float(scales_days[low_idx]),
        float(scales_days[high_idx]),
        peak_idx == last_valid_idx,
        high_idx == last_valid_idx,
    )


def analyze_wavelet(
    prepared: pd.DataFrame,
    *,
    config: WWZConfig = WWZConfig(),
) -> WWZResult:
    """Analyze one prepared nightly relative-flux light curve."""
    config.validate()
    times_jd = pd.to_numeric(prepared["jd"], errors="coerce").to_numpy(float)
    relative_flux = pd.to_numeric(
        prepared["relative_flux"], errors="coerce"
    ).to_numpy(float)
    finite = np.isfinite(times_jd) & np.isfinite(relative_flux)
    times_jd = times_jd[finite]
    relative_flux = relative_flux[finite]
    if times_jd.size < config.min_nights:
        raise RuntimeError(
            f"Only {times_jd.size} finite nights; at least {config.min_nights} are required"
        )

    order = np.argsort(times_jd)
    times_jd = times_jd[order]
    relative_flux = relative_flux[order]
    times = times_jd - float(times_jd[0])
    unique_times = np.unique(times)
    cadence = (
        float(np.median(np.diff(unique_times)))
        if unique_times.size > 1
        else np.nan
    )
    span = float(times[-1] - times[0])
    max_scale = span * float(config.max_scale_fraction_of_span)
    if config.max_scale_days is not None:
        max_scale = min(max_scale, float(config.max_scale_days))
    if max_scale <= config.min_scale_days:
        raise RuntimeError(
            f"Light-curve span {span:.3f} d does not support the configured scales"
        )

    scales = np.geomspace(
        float(config.min_scale_days),
        max_scale,
        int(config.n_scales),
    )
    time_centers = np.linspace(times[0], times[-1], int(config.n_time_bins))
    power, effective_points = weighted_wavelet_z(
        times,
        relative_flux,
        time_centers,
        scales,
        decay_constant=config.decay_constant,
        min_effective_points=config.min_effective_points,
    )
    edge_half_width = scales[:, None] / (
        2.0 * math.pi * math.sqrt(float(config.decay_constant))
    )
    edge_reliable = (
        (time_centers[None, :] - times[0] >= edge_half_width)
        & (times[-1] - time_centers[None, :] >= edge_half_width)
    )
    valid_counts = np.sum(np.isfinite(power), axis=1)
    global_power = np.divide(
        np.nansum(power, axis=1),
        valid_counts,
        out=np.full(scales.size, np.nan),
        where=valid_counts > 0,
    )
    dominant, low, high, peak_at_upper, band_hits_upper = _dominant_scale_band(
        scales,
        global_power,
    )
    if np.isfinite(dominant):
        dominant_idx = int(np.argmin(np.abs(scales - dominant)))
        dominant_edge_reliable_fraction = float(
            np.mean(edge_reliable[dominant_idx])
        )
    else:
        dominant_edge_reliable_fraction = np.nan
    return WWZResult(
        times=times,
        relative_flux=relative_flux,
        time_centers=time_centers,
        scales_days=scales,
        power=power,
        effective_points=effective_points,
        edge_reliable=edge_reliable,
        global_power=global_power,
        dominant_scale_days=dominant,
        dominant_scale_low_days=low,
        dominant_scale_high_days=high,
        dominant_peak_at_upper_boundary=peak_at_upper,
        dominant_band_hits_upper_boundary=band_hits_upper,
        dominant_scale_edge_reliable_fraction=dominant_edge_reliable_fraction,
        max_scale_analyzed_days=float(scales[-1]),
        median_cadence_days=cadence,
    )


def _latex_escape(text: Any) -> str:
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _positive_color_limits(power: np.ndarray) -> tuple[float, float]:
    values = np.asarray(power, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1e-3, 1.0
    low = float(np.percentile(values, 10.0))
    high = float(np.percentile(values, 99.5))
    low = max(low, float(np.min(values)), 1e-8)
    high = max(high, low * 10.0)
    return low, high


def plot_wavelet_figure(
    result: WWZResult,
    *,
    candidate_id: str,
    asas_sn_id: str | None = None,
    period_days: float | None = None,
    period_source: str | None = None,
    figure_size: tuple[float, float] = (8.0, 5.2),
) -> plt.Figure:
    """Create a light curve, phase fold, WWZ scalogram, and power profile."""
    with plt.rc_context(PUBLICATION_STYLE):
        fig = plt.figure(figsize=figure_size)
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=(1.0, 1.15),
            width_ratios=(4.6, 1.55),
            hspace=0.08,
            wspace=0.12,
        )
        ax_lc = fig.add_subplot(grid[0, 0])
        ax_phase = fig.add_subplot(grid[0, 1])
        ax_map = fig.add_subplot(grid[1, 0], sharex=ax_lc)
        ax_global = fig.add_subplot(grid[1, 1], sharey=ax_map)

        ax_lc.scatter(
            result.times,
            result.relative_flux,
            s=3.0,
            color="black",
            linewidths=0,
            rasterized=True,
        )
        ax_lc.axhline(1.0, color="0.55", linewidth=0.55, zorder=0)
        ax_lc.set_ylabel("Relative flux")
        title = _latex_escape(candidate_id)
        if asas_sn_id is not None and str(asas_sn_id).strip():
            title += rf" \quad ASAS-SN {_latex_escape(asas_sn_id)}"
        ax_lc.set_title(title)
        ax_lc.tick_params(labelbottom=False)

        try:
            phase_period = float(period_days) if period_days is not None else np.nan
        except (TypeError, ValueError):
            phase_period = np.nan
        if np.isfinite(phase_period) and phase_period > 0:
            phase = np.mod(result.times / phase_period, 1.0)
            ax_phase.scatter(
                np.concatenate((phase, phase + 1.0)),
                np.tile(result.relative_flux, 2),
                s=2.4,
                color="black",
                alpha=0.62,
                linewidths=0,
                rasterized=True,
            )
            ax_phase.axhline(1.0, color="0.55", linewidth=0.55, zorder=0)
            source_label = str(period_source or "pipeline").upper()
            ax_phase.set_title(
                f"Pipeline phase fold\nP={phase_period:.5g} d ({_latex_escape(source_label)})",
                fontsize=8,
            )
        else:
            ax_phase.text(
                0.5,
                0.5,
                "No pipeline period",
                transform=ax_phase.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="0.4",
            )
            ax_phase.set_title("Pipeline phase fold", fontsize=8)
        ax_phase.set_xlim(0.0, 2.0)
        ax_phase.set_ylim(ax_lc.get_ylim())
        ax_phase.set_xlabel("Phase")
        ax_phase.set_ylabel("Relative flux")

        vmin, vmax = _positive_color_limits(result.power)
        ax_map.pcolormesh(
            result.time_centers,
            result.scales_days,
            result.power,
            shading="auto",
            cmap="turbo",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        ax_map.set_facecolor("0.86")
        edge_unreliable = np.ma.masked_where(
            result.edge_reliable,
            np.ones_like(result.power),
        )
        ax_map.pcolormesh(
            result.time_centers,
            result.scales_days,
            edge_unreliable,
            shading="auto",
            cmap=ListedColormap([(0.72, 0.72, 0.72, 0.58)]),
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        if np.any(result.edge_reliable) and np.any(~result.edge_reliable):
            ax_map.contour(
                result.time_centers,
                result.scales_days,
                result.edge_reliable.astype(float),
                levels=[0.5],
                colors=["white"],
                linewidths=0.55,
                alpha=0.9,
            )
        valid = np.isfinite(result.power).astype(float)
        if np.nanmin(valid) < 0.5 < np.nanmax(valid):
            ax_map.contour(
                result.time_centers,
                result.scales_days,
                valid,
                levels=[0.5],
                colors=["white"],
                linewidths=0.45,
                alpha=0.85,
            )
        ax_map.set_yscale("log")
        ax_map.set_xlabel("Time since first observation (days)")
        ax_map.set_ylabel("Wavelet scale (days)")

        ax_global.plot(
            result.global_power,
            result.scales_days,
            color="black",
            linewidth=1.0,
        )
        ax_global.set_xlabel("Mean WWZ")
        ax_global.tick_params(labelleft=False)
        ax_global.grid(axis="x", color="0.85", linewidth=0.4)

        for scale in (
            result.dominant_scale_low_days,
            result.dominant_scale_high_days,
        ):
            if np.isfinite(scale):
                ax_map.axhline(scale, color="#d62728", linestyle=(0, (5, 4)), linewidth=1.0)
                ax_global.axhline(scale, color="#d62728", linestyle=(0, (5, 4)), linewidth=1.0)

        if np.isfinite(result.dominant_scale_days):
            ax_global.axhline(
                result.dominant_scale_days,
                color="#d62728",
                linewidth=0.75,
                alpha=0.85,
            )
            quality_notes: list[str] = []
            if result.dominant_peak_at_upper_boundary:
                quality_notes.append("upper-bound limited")
            if result.dominant_scale_edge_reliable_fraction <= 0:
                quality_notes.append("edge-dominated")
            elif result.dominant_scale_edge_reliable_fraction < 0.5:
                quality_notes.append("partly edge-affected")
            note = f"\n{'; '.join(quality_notes)}" if quality_notes else ""
            ax_global.text(
                0.96,
                0.04,
                (
                    f"Dominant {result.dominant_scale_days:.3g} d\n"
                    f"[{result.dominant_scale_low_days:.3g}, "
                    f"{result.dominant_scale_high_days:.3g}] d"
                    f"{note}"
                ),
                transform=ax_global.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.5,
                },
            )
        ax_lc.set_xlim(result.time_centers[0], result.time_centers[-1])
        ax_map.set_ylim(result.scales_days[0], result.scales_days[-1])
        ax_global.set_ylim(result.scales_days[0], result.scales_days[-1])
        for axis in (ax_lc, ax_phase, ax_map, ax_global):
            axis.tick_params(direction="in", top=True, right=True)
        return fig


def _safe_stem(candidate_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(candidate_id)).strip("._")
    return stem or "candidate"


def generate_wavelet_atlas(
    candidates: pd.DataFrame,
    *,
    run_root: Path | str,
    output_dir: Path | str,
    config: WWZConfig = WWZConfig(),
    max_candidates: int | None = None,
    write_png: bool = True,
    write_pdf: bool = True,
    build_atlas_pdf: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate one wavelet figure per candidate plus a manifest and PDF atlas."""
    config.validate()
    run_path = Path(run_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    png_dir = output_path / "png"
    pdf_dir = output_path / "pdf"
    output_path.mkdir(parents=True, exist_ok=True)
    if write_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if write_pdf:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    selected = candidates.copy()
    if max_candidates is not None:
        selected = selected.head(int(max_candidates)).copy()

    atlas_path = output_path / "all_dipper_wavelet_atlas.pdf"
    atlas_context = PdfPages(atlas_path) if build_atlas_pdf else None
    rows: list[dict[str, Any]] = []
    try:
        total = len(selected)
        for candidate_index, row in enumerate(selected.itertuples(index=False), start=1):
            candidate_id = str(row.candidate_id)
            if progress:
                print(f"[{candidate_index}/{total}] {candidate_id}", flush=True)
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "asas_sn_id": str(getattr(row, "asas_sn_id", "") or ""),
                "phase_period_days": np.nan,
                "phase_source": "",
                "plot_status": "ok",
                "error": "",
                "lc_path": "",
                "n_quality_exposures": 0,
                "n_nights": 0,
                "span_days": np.nan,
                "median_cadence_days": np.nan,
                "dominant_scale_days": np.nan,
                "dominant_scale_low_days": np.nan,
                "dominant_scale_high_days": np.nan,
                "dominant_peak_at_upper_boundary": False,
                "dominant_band_hits_upper_boundary": False,
                "dominant_scale_edge_reliable_fraction": np.nan,
                "max_scale_analyzed_days": np.nan,
                "valid_wwz_fraction": np.nan,
                "edge_reliable_wwz_fraction": np.nan,
                "png_path": "",
                "pdf_path": "",
            }
            fig: plt.Figure | None = None
            try:
                lc_path = resolve_local_lightcurve_path(
                    getattr(row, "lc_path", None),
                    run_dir=run_path,
                )
                if lc_path is None:
                    raise FileNotFoundError(f"Could not resolve light curve: {getattr(row, 'lc_path', '')}")
                record["lc_path"] = str(lc_path)
                lightcurve = load_lightcurve_df(
                    lc_path,
                    filter_bad_cameras_enabled=True,
                    apply_quality=True,
                )
                record["n_quality_exposures"] = int(len(lightcurve))
                prepared = prepare_relative_flux(lightcurve)
                record["n_nights"] = int(len(prepared))
                result = analyze_wavelet(prepared, config=config)
                record.update(
                    {
                        "span_days": float(result.times[-1] - result.times[0]),
                        "median_cadence_days": result.median_cadence_days,
                        "dominant_scale_days": result.dominant_scale_days,
                        "dominant_scale_low_days": result.dominant_scale_low_days,
                        "dominant_scale_high_days": result.dominant_scale_high_days,
                        "dominant_peak_at_upper_boundary": result.dominant_peak_at_upper_boundary,
                        "dominant_band_hits_upper_boundary": result.dominant_band_hits_upper_boundary,
                        "dominant_scale_edge_reliable_fraction": result.dominant_scale_edge_reliable_fraction,
                        "max_scale_analyzed_days": result.max_scale_analyzed_days,
                        "valid_wwz_fraction": float(np.mean(np.isfinite(result.power))),
                        "edge_reliable_wwz_fraction": float(np.mean(result.edge_reliable)),
                    }
                )
                fig = plot_wavelet_figure(
                    result,
                    candidate_id=candidate_id,
                    asas_sn_id=record["asas_sn_id"],
                    period_days=getattr(row, "periodicity_period", None),
                    period_source=getattr(row, "periodicity_method", None),
                )
                record["phase_period_days"] = getattr(
                    row,
                    "periodicity_period",
                    np.nan,
                )
                record["phase_source"] = str(
                    getattr(row, "periodicity_method", "") or ""
                )
                stem = _safe_stem(candidate_id)
                if write_png:
                    png_path = png_dir / f"{stem}.png"
                    fig.savefig(png_path, dpi=220, bbox_inches="tight")
                    record["png_path"] = str(png_path)
                if write_pdf:
                    pdf_path = pdf_dir / f"{stem}.pdf"
                    fig.savefig(pdf_path, bbox_inches="tight")
                    record["pdf_path"] = str(pdf_path)
                if atlas_context is not None:
                    atlas_context.savefig(fig, bbox_inches="tight")
            except Exception as exc:
                record["plot_status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                if progress:
                    print(f"  ERROR: {record['error']}", flush=True)
            finally:
                if fig is not None:
                    plt.close(fig)
            rows.append(record)
    finally:
        if atlas_context is not None:
            atlas_context.close()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_path / "wavelet_manifest.csv", index=False)
    return manifest


__all__ = [
    "WWZConfig",
    "WWZResult",
    "analyze_wavelet",
    "generate_wavelet_atlas",
    "plot_wavelet_figure",
    "prepare_relative_flux",
    "read_review_labeled_candidates",
    "weighted_wavelet_z",
]
