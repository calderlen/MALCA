"""Native Matplotlib publication exports for review light curves."""

from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    JD_OFFSET,
    REVIEW_RESIDUAL_FRACTION,
)
from malca.lightcurve_publication import PUBLICATION_STYLE, _load_matplotlib, style_publication_axis
from malca.lightcurve_io import stable_camera_color
from malca.phase import BAND_LABELS, phase_fold_dataframe, phase_time_dataframe, resolve_phase_epoch, resolve_phase_period
from malca.review.interactive_plot import (
    DIP_EVENT_COLOR,
    JUMP_EVENT_COLOR,
    PHASE_TIME_COLORSCALE,
    REQUIRED_COLUMNS,
    _baseline_config_from_run_params,
    _camera_labels,
    _compute_baseline_bands,
    _event_entries,
    _flux_err_from_mag_err,
    _load_cleaned_df,
    _mag_to_flux,
    resolve_lightcurve_path,
    _zero_centered_color_bounds,
)


_BAND_MARKERS = {0: "o", 1: "s"}


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _run_param_float(run_params: dict | None, key: str, default: float) -> float:
    value = _finite_float((run_params or {}).get(key))
    return float(default if value is None else value)


def _selected_bands(selected_bands: list[str] | None, available_labels: list[str]) -> list[int]:
    selected_lookup = {
        str(value).strip().lower()
        for value in (selected_bands if selected_bands is not None else available_labels)
        if str(value).strip()
    }
    bands = [
        band
        for band, label in BAND_LABELS.items()
        if label.lower() in selected_lookup and label in available_labels
    ]
    return bands or [band for band, label in BAND_LABELS.items() if label in available_labels]


def _axis_label_for_offset(jd_offset: float) -> str:
    if abs(float(jd_offset) - round(float(jd_offset))) < 1e-6:
        return rf"$\mathrm{{JD}} - {int(round(float(jd_offset)))}$"
    return rf"$\mathrm{{JD}} - {float(jd_offset):.1f}$"


def _plot_points(
    ax,
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    err_col: str | None,
    color: str,
    marker: str,
    label: str | None,
    marker_size: float = 3.4,
    alpha: float = 0.86,
) -> None:
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    x = x[mask]
    y = y[mask]
    yerr = None
    if err_col and err_col in frame.columns:
        err = pd.to_numeric(frame[err_col], errors="coerce").to_numpy(dtype=float)[mask]
        if np.isfinite(err).any():
            yerr = err
    if yerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            linestyle="none",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="0.12",
            markeredgewidth=0.35,
            ecolor=color,
            elinewidth=0.45,
            capsize=0.0,
            alpha=alpha,
            label=label,
            zorder=4,
        )
    else:
        ax.scatter(
            x,
            y,
            s=marker_size**2,
            marker=marker,
            facecolors=color,
            edgecolors="0.12",
            linewidths=0.35,
            alpha=alpha,
            label=label,
            zorder=4,
        )


def _set_robust_limits(ax, values: list[np.ndarray], *, inverted: bool, pad_fraction: float = 0.05) -> None:
    finite = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values if np.asarray(v).size])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        pad = max(0.1, abs(lo) * pad_fraction)
        lo -= pad
        hi += pad
    else:
        pad = max(0.03, (hi - lo) * pad_fraction)
        lo -= pad
        hi += pad
    ax.set_ylim((hi, lo) if inverted else (lo, hi))


def _robust_color_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    lo, hi = np.nanpercentile(vals, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, None
    if lo == hi:
        pad = max(abs(float(lo)) * 0.05, 0.05)
        lo = float(lo) - pad
        hi = float(hi) + pad
    return float(lo), float(hi)


def _phase_time_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "malca_phase_time_delta_m",
        [item[1] for item in PHASE_TIME_COLORSCALE],
    )


def _style_lightcurve_axis(ax) -> None:
    style_publication_axis(ax, top=False, right=False)
    ax.grid(True, which="major", linewidth=0.42, alpha=0.24)
    ax.tick_params(which="both", direction="out", top=False, right=False)


def build_review_lightcurve_publication_pdf(
    payload: dict,
    *,
    plot_dir: str | Path | None,
    selected_cameras: list[str] | None,
    selected_bands: list[str] | None,
    filter_bad_cameras: bool,
    show_baseline: bool,
    show_event_markers: bool,
    show_residuals: bool,
    show_phase_fold: bool,
    show_raw_mag: bool,
    phase_panel_mode: Literal["fold", "time"] = "fold",
    override_period: float | None,
    override_period_source: str = "manual/search",
    phase_period_pending: bool = False,
    suppress_catalog_phase_period: bool = False,
    show_diagnostics: bool,
    confidence_colors: bool,
    run_params: dict | None,
    residual_fraction: float = REVIEW_RESIDUAL_FRACTION,
    baseline_opacity: float = 0.55,
    yaxis_mode: Literal["mag", "flux"] = "mag",
) -> bytes:
    """Render the review light-curve view directly from data as a Matplotlib PDF."""
    phase_panel_mode = "time" if str(phase_panel_mode or "fold").strip().lower() == "time" else "fold"
    plot_dir_path = Path(plot_dir) if plot_dir else None
    lc_path = resolve_lightcurve_path(payload, plot_dir_path)
    if lc_path is None:
        raise FileNotFoundError("No light-curve file found for this candidate.")

    scatter_ratio = _run_param_float(run_params, "bad_camera_scatter_ratio", BAD_CAMERA_SCATTER_RATIO_THRESHOLD)
    clean_abs = _run_param_float(run_params, "clean_max_error_absolute", CLEAN_LC_MAX_ERROR_ABSOLUTE)
    clean_sig = _run_param_float(run_params, "clean_max_error_sigma", CLEAN_LC_MAX_ERROR_SIGMA)
    df, _filtered_cameras, _camera_diagnostics = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=bool(filter_bad_cameras),
        scatter_ratio=scatter_ratio,
        clean_max_error_absolute=clean_abs,
        clean_max_error_sigma=clean_sig,
    )
    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required light-curve columns: {', '.join(missing_cols)}")
    if df.empty:
        raise ValueError("No light-curve points remain after cleaning.")

    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    if df.empty:
        raise ValueError("No finite light-curve points remain after cleaning.")
    median_jd = float(df["JD"].median())
    jd_offset = JD_OFFSET if median_jd > 2_000_000 else 8000.0
    df["JD_plot"] = pd.to_numeric(df["JD"], errors="coerce") - jd_offset
    df["camera_label"] = _camera_labels(df, payload)

    camera_ids = sorted(df["camera_label"].dropna().astype(str).unique().tolist())
    selected = [str(c) for c in (selected_cameras or []) if str(c) in camera_ids] or camera_ids
    df = df[df["camera_label"].astype(str).isin(selected)].copy()
    if df.empty:
        raise ValueError("Selected cameras contain no light-curve points.")

    available_band_labels = [
        label
        for band, label in BAND_LABELS.items()
        if int((df["v_g_band"] == band).sum()) > 0
    ]
    active_bands = _selected_bands(selected_bands, available_band_labels)
    if not active_bands:
        raise ValueError("Selected bands contain no light-curve points.")

    baseline_name, baseline_kwargs, _warnings = _baseline_config_from_run_params(run_params)
    baseline_cache_key = (
        str(lc_path.resolve()),
        tuple(sorted(str(c) for c in selected)),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_abs),
        float(clean_sig),
    )
    band_dfs = _compute_baseline_bands(
        df,
        baseline_name,
        baseline_cache_key,
        baseline_kwargs=baseline_kwargs,
    )

    if show_phase_fold and not phase_period_pending:
        period_payload = {} if suppress_catalog_phase_period else payload
        phase_period, _phase_source = resolve_phase_period(
            period_payload,
            override_period=override_period,
            override_source=override_period_source or "manual/search",
        )
    else:
        phase_period, _phase_source = (None, "")
    phase_enabled = bool(show_phase_fold and phase_period is not None)
    panels: list[str] = []
    if show_raw_mag:
        panels.append("raw")
    if show_residuals:
        panels.append("resid")
    if phase_enabled:
        panels.append("phase")
    if not panels:
        raise ValueError("No light-curve panels selected for export.")

    is_flux = str(yaxis_mode or "mag") == "flux"
    n_rows = len(panels)
    if n_rows == 1:
        height_ratios = [1.0]
    elif n_rows == 2:
        height_ratios = [1.45, 1.0]
    else:
        lower = float(np.clip(residual_fraction, 0.18, 0.34))
        height_ratios = [1.0 - 2.0 * lower, lower, lower]

    plt, _auto_minor = _load_matplotlib()
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(7.35, 2.15 + 1.35 * n_rows),
            sharex=False,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes_list = list(np.atleast_1d(axes))
        ax_by_panel = {panel: axes_list[idx] for idx, panel in enumerate(panels)}

        legend_handles = []
        legend_labels = []
        raw_values: list[np.ndarray] = []
        resid_values: list[np.ndarray] = []
        phase_values: list[np.ndarray] = []

        for band in active_bands:
            bdf = band_dfs.get(band)
            if bdf is None or bdf.empty:
                continue
            marker = _BAND_MARKERS.get(band, "o")
            band_label = BAND_LABELS.get(band, str(band))
            for cam in selected:
                cdf = bdf[bdf["camera_label"].astype(str) == str(cam)].copy()
                if cdf.empty:
                    continue
                color = stable_camera_color(cam)
                label = f"{cam} ({band_label})"
                err = pd.to_numeric(cdf.get("error"), errors="coerce").to_numpy(dtype=float)
                mag = pd.to_numeric(cdf["mag"], errors="coerce").to_numpy(dtype=float)
                resid = pd.to_numeric(cdf.get("resid"), errors="coerce").to_numpy(dtype=float)

                if "raw" in ax_by_panel:
                    if is_flux:
                        cdf["_raw_y"] = _mag_to_flux(mag)
                        cdf["_raw_err"] = _flux_err_from_mag_err(cdf["_raw_y"].to_numpy(dtype=float), err)
                    else:
                        cdf["_raw_y"] = mag
                        cdf["_raw_err"] = err
                    _plot_points(
                        ax_by_panel["raw"],
                        cdf,
                        x_col="JD_plot",
                        y_col="_raw_y",
                        err_col="_raw_err",
                        color=color,
                        marker=marker,
                        label=label,
                        marker_size=3.15,
                    )
                    raw_values.append(cdf["_raw_y"].to_numpy(dtype=float))
                    if show_baseline and "baseline" in cdf.columns:
                        base = cdf[np.isfinite(pd.to_numeric(cdf["baseline"], errors="coerce"))].sort_values("JD_plot")
                        if not base.empty:
                            y_base = _mag_to_flux(base["baseline"].to_numpy(dtype=float)) if is_flux else base["baseline"].to_numpy(dtype=float)
                            ax_by_panel["raw"].plot(
                                base["JD_plot"],
                                y_base,
                                color=color,
                                linewidth=0.95,
                                alpha=float(np.clip(baseline_opacity, 0.0, 1.0)),
                                zorder=3,
                            )
                            raw_values.append(np.asarray(y_base, dtype=float))

                if "resid" in ax_by_panel:
                    if is_flux:
                        cdf["_resid_y"] = _mag_to_flux(resid) - 1.0
                        cdf["_resid_err"] = _flux_err_from_mag_err(_mag_to_flux(resid), err)
                    else:
                        cdf["_resid_y"] = resid
                        cdf["_resid_err"] = err
                    _plot_points(
                        ax_by_panel["resid"],
                        cdf,
                        x_col="JD_plot",
                        y_col="_resid_y",
                        err_col=None,
                        color=color,
                        marker=marker,
                        label=None,
                        marker_size=2.85,
                        alpha=0.82,
                    )
                    resid_values.append(cdf["_resid_y"].to_numpy(dtype=float))

        if show_event_markers and "raw" in ax_by_panel:
            event_ax = ax_by_panel["raw"]
            for entry in _event_entries(payload, jd_offset, run_params):
                color = DIP_EVENT_COLOR if entry["kind"] == "dip" else JUMP_EVENT_COLOR
                if confidence_colors:
                    alpha = 0.28 + 0.42 * float(entry.get("confidence") or 0.0)
                else:
                    alpha = 0.22
                x0 = float(entry["x0"])
                half_width = float(entry.get("half_width") or 0.0)
                if show_diagnostics and half_width > 0:
                    event_ax.axvspan(x0 - half_width, x0 + half_width, color=color, alpha=0.08, linewidth=0, zorder=1)
                event_ax.axvline(x0, color=color, linestyle="--", linewidth=0.9, alpha=alpha + 0.25, zorder=2)

        phase_diag: dict[str, object] = {}
        if "phase" in ax_by_panel and phase_period is not None:
            phase_inputs = [
                band_dfs[band]
                for band in active_bands
                if band in band_dfs and band_dfs[band] is not None and not band_dfs[band].empty
            ]
            if phase_inputs:
                phase_source = pd.concat(phase_inputs, ignore_index=True)
                if phase_panel_mode == "time":
                    phase_df, phase_diag = phase_time_dataframe(
                        phase_source,
                        float(phase_period),
                        epoch_jd=resolve_phase_epoch(df),
                        value_mode="resid",
                        duplicate_cycles=True,
                    )
                else:
                    phase_df, phase_diag = phase_fold_dataframe(
                        phase_source,
                        float(phase_period),
                        epoch_jd=resolve_phase_epoch(df),
                        value_mode="resid",
                        duplicate_cycles=True,
                    )
                if phase_panel_mode == "time" and not phase_df.empty and "v_g_band" in phase_df.columns:
                    phase_ax = ax_by_panel["phase"]
                    color_values = pd.to_numeric(phase_df.get("phase_value"), errors="coerce").to_numpy(dtype=float)
                    cmin, cmax = _zero_centered_color_bounds(color_values)
                    cmap = _phase_time_colormap()
                    scatter_for_colorbar = None
                    for band in active_bands:
                        band_df = phase_df[pd.to_numeric(phase_df["v_g_band"], errors="coerce") == band]
                        if band_df.empty:
                            continue
                        marker = _BAND_MARKERS.get(band, "o")
                        for cam in selected:
                            cdf = band_df[band_df["camera_label"].astype(str) == str(cam)].copy()
                            if cdf.empty:
                                continue
                            phase = pd.to_numeric(cdf["phase"], errors="coerce").to_numpy(dtype=float)
                            cycle = pd.to_numeric(cdf["cycle"], errors="coerce").to_numpy(dtype=float)
                            resid = pd.to_numeric(cdf["phase_value"], errors="coerce").to_numpy(dtype=float)
                            valid = np.isfinite(phase) & np.isfinite(cycle) & np.isfinite(resid)
                            if not valid.any():
                                continue
                            scatter_kwargs = {
                                "s": 8.0,
                                "marker": marker,
                                "c": resid[valid],
                                "cmap": cmap,
                                "edgecolors": "0.12",
                                "linewidths": 0.25,
                                "alpha": 0.84,
                                "zorder": 4,
                            }
                            if cmin is not None and cmax is not None:
                                scatter_kwargs["vmin"] = cmin
                                scatter_kwargs["vmax"] = cmax
                            sc = phase_ax.scatter(phase[valid], cycle[valid], **scatter_kwargs)
                            phase_values.append(resid[valid])
                            if scatter_for_colorbar is None:
                                scatter_for_colorbar = sc
                    if scatter_for_colorbar is not None:
                        cbar = fig.colorbar(scatter_for_colorbar, ax=phase_ax, pad=0.012, aspect=14)
                        cbar.set_label(r"$\Delta m$", fontsize=7.0)
                        cbar.ax.tick_params(labelsize=6.5)
                    for x in (0.0, 1.0, 2.0):
                        phase_ax.axvline(x, color="0.55", linestyle=":", linewidth=0.65, alpha=0.7, zorder=1)
                elif not phase_df.empty and "v_g_band" in phase_df.columns:
                    phase_ax = ax_by_panel["phase"]
                    for band in active_bands:
                        band_df = phase_df[pd.to_numeric(phase_df["v_g_band"], errors="coerce") == band]
                        if band_df.empty:
                            continue
                        marker = _BAND_MARKERS.get(band, "o")
                        for cam in selected:
                            cdf = band_df[band_df["camera_label"].astype(str) == str(cam)].copy()
                            if cdf.empty:
                                continue
                            color = stable_camera_color(cam)
                            resid = pd.to_numeric(cdf["phase_value"], errors="coerce").to_numpy(dtype=float)
                            err = pd.to_numeric(cdf.get("error"), errors="coerce").to_numpy(dtype=float)
                            if is_flux:
                                cdf["_phase_y"] = _mag_to_flux(resid) - 1.0
                                cdf["_phase_err"] = _flux_err_from_mag_err(_mag_to_flux(resid), err)
                            else:
                                cdf["_phase_y"] = resid
                                cdf["_phase_err"] = err
                            _plot_points(
                                phase_ax,
                                cdf,
                                x_col="phase",
                                y_col="_phase_y",
                                err_col=None,
                                color=color,
                                marker=marker,
                                label=None,
                                marker_size=2.75,
                                alpha=0.82,
                            )
                            phase_values.append(cdf["_phase_y"].to_numpy(dtype=float))
                    for x in (0.0, 1.0, 2.0):
                        phase_ax.axvline(x, color="0.55", linestyle=":", linewidth=0.65, alpha=0.7, zorder=1)
                    phase_ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.65, alpha=0.75, zorder=1)
                    phase_lag = _finite_float(phase_diag.get("phase_lag_g_v_cycles"))
                    phase_lag_abs = _finite_float(phase_diag.get("phase_lag_g_v_abs_cycles"))
                    if phase_lag is not None:
                        lag = rf"$g-V={phase_lag:+.3f}$ cyc"
                        if phase_lag_abs is not None:
                            lag += rf", $|g-V|={phase_lag_abs:.3f}$"
                        phase_ax.text(0.985, 0.94, lag, transform=phase_ax.transAxes, ha="right", va="top", fontsize=7.0, color="0.20")

        for panel, ax in ax_by_panel.items():
            _style_lightcurve_axis(ax)
            if panel == "raw":
                ax.set_ylabel(r"$F$ [arb.]" if is_flux else r"$m$ [mag]")
                _set_robust_limits(ax, raw_values, inverted=not is_flux)
            elif panel == "resid":
                ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.65, alpha=0.8, zorder=1)
                ax.set_ylabel(r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]")
                _set_robust_limits(ax, resid_values, inverted=not is_flux)
            elif panel == "phase":
                ax.set_xlim(-0.02, 2.02)
                if phase_panel_mode == "time":
                    ax.set_ylabel(r"Cycle $E$")
                    ax.set_xlabel(rf"$\phi\ \mathrm{{vs.}}\ E\,(P={float(phase_period):.5f}\,\mathrm{{d}})$")
                else:
                    ax.set_ylabel(r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]")
                    ax.set_xlabel(rf"$\phi\,(P={float(phase_period):.5f}\,\mathrm{{d}})$")
                    _set_robust_limits(ax, phase_values, inverted=not is_flux)

        time_label = _axis_label_for_offset(jd_offset)
        time_panel = "resid" if "resid" in ax_by_panel else "raw" if "raw" in ax_by_panel else None
        if time_panel is not None:
            ax_by_panel[time_panel].set_xlabel(time_label)
        if "raw" in ax_by_panel and time_panel != "raw":
            ax_by_panel["raw"].tick_params(axis="x", labelbottom=False)

        if "raw" in ax_by_panel:
            handles, labels = ax_by_panel["raw"].get_legend_handles_labels()
            seen: set[str] = set()
            for handle, label in zip(handles, labels):
                if not label or label in seen:
                    continue
                seen.add(label)
                legend_handles.append(handle)
                legend_labels.append(label)
            if legend_handles:
                ncol = 1 if len(legend_labels) <= 9 else 2
                ax_by_panel["raw"].legend(
                    legend_handles,
                    legend_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    frameon=False,
                    borderaxespad=0.0,
                    fontsize=6.7,
                    ncol=ncol,
                    columnspacing=0.7,
                    handletextpad=0.35,
                )

        label = str(payload.get("asas_sn_id") or payload.get("candidate_id") or "").strip()
        if label:
            fig.text(0.09, 0.985, label, ha="left", va="top", fontsize=8.0, color="0.15")
        fig.subplots_adjust(
            left=0.085,
            right=0.80 if legend_handles else 0.965,
            bottom=0.075,
            top=0.955,
            hspace=0.16 if len(panels) > 1 else 0.08,
        )

        buf = BytesIO()
        try:
            fig.savefig(buf, format="pdf", dpi=300, metadata={"Creator": "MALCA"}, bbox_inches="tight", pad_inches=0.035)
            return buf.getvalue()
        finally:
            plt.close(fig)
