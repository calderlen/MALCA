                      
"""
Plot light curves with Bayesian event detection results, showing run fits overlaid.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence
import argparse
import hashlib
import json
import shlex
import sys
import time

from matplotlib.lines import Line2D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.core.baseline import (
    global_median_baseline,
    phase_template_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.config import (
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
)
from malca.config import (
    WORKERS, JD_OFFSET, PLOT_FIGSIZE,
    LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP,
)
from malca.config import MAD_SCALE
from malca.stv.events import score_lightcurve
from malca.plotting.lightcurve_publication import (
    DIP_EVENT_COLOR,
    JUMP_EVENT_COLOR,
    FIG_SINGLE_COL_LC_WIDE,
    PUBLICATION_STYLE,
    apply_publication_rcparams,
    finalize_publication_figure,
    save_publication_figure,
    plot_phase_panel,
)

apply_publication_rcparams(plt)
from malca.core.phase import camera_labels
from malca.review.metadata import REVIEW_METADATA_FIELDS, normalize_vsx_record
from malca.io.table_io import read_feature_table
from malca.products.feature_layers import expand_feature_layers, is_layer_first_frame
from malca.io.lightcurve_io import (
    load_lightcurve_df as _canonical_load_lightcurve_df,
    read_asassn_dat as _read_asassn_dat,
    to_asassn_algorithm_frame,
)
from malca.core.utils import clean_lc, filter_bad_cameras as filter_bad_camera_observations
from malca.core.utils import gaussian, paczynski_kernel, read_skypatrol_csv as _read_skypatrol_csv

read_skypatrol_csv = _read_skypatrol_csv



matplotlib.use('Agg')


CAMERA_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]


def _stable_camera_color(camera_label: str) -> str:
    """Return a deterministic color for a camera label across all plots."""
    s = str(camera_label)
    try:
        idx = int(s) % len(CAMERA_COLOR_PALETTE)
    except Exception:
        digest = hashlib.md5(s.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(CAMERA_COLOR_PALETTE)
    return CAMERA_COLOR_PALETTE[idx]

asassn_columns = [
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera#",
    "v_g_band",
    "saturated",
    "camera_name",
    "field",
]

asassn_raw_columns = [
    "cam#",
    "median",
    "1siglow",
    "1sighigh",
    "90percentlow",
    "90percenthigh",
]


def read_asassn_dat(dat_path):
    """
    Read an ASAS-SN .dat file using whitespace separation.
    """
    return _read_asassn_dat(dat_path)




def load_lightcurve_df(
    path,
    *,
    filter_bad_cameras_enabled: bool = False,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    return_filtered_info: bool = False,
):
    """
    Dispatch loader based on file extension (.csv -> SkyPatrol, else ASAS-SN .dat).
    
    Parameters
    ----------
    path : Path-like
        Path to light curve file
    filter_bad_cameras_enabled : bool
        If True, filter out cameras with anomalously high scatter
    bad_camera_scatter_ratio : float
        Scatter ratio threshold for bad camera filtering
    return_filtered_info : bool
        If True, return (df, filtered_cameras) tuple instead of just df

    Returns
    -------
    df : pd.DataFrame
        Light curve data (or empty DataFrame if loading fails)
    filtered_cameras : set[int]
        Only returned if return_filtered_info=True. Set of camera IDs removed.
    """
    loaded = _canonical_load_lightcurve_df(
        path,
        filter_bad_cameras_enabled=filter_bad_cameras_enabled,
        bad_camera_scatter_ratio=bad_camera_scatter_ratio,
        return_filtered_info=return_filtered_info,
    )
    if return_filtered_info:
        df, filtered_cameras = loaded
        return to_asassn_algorithm_frame(df), filtered_cameras
    return to_asassn_algorithm_frame(loaded)


def load_events_paths(
    events_path: Path,
    *,
    path_col: str = "path",
    only_significant: bool = False,
    max_plots: int | None = None,
) -> list[Path]:
    """
    Load events/post-filter output and return unique LC paths.
    Supports Parquet.
    """
    events_path = Path(events_path)
    df = read_feature_table(events_path)

    if path_col not in df.columns:
        raise KeyError(f"Missing '{path_col}' column in {events_path}")

    if only_significant and {"dip_significant", "jump_significant"}.issubset(df.columns):
        df = df[(df["dip_significant"].fillna(False)) | (df["jump_significant"].fillna(False))]

    paths = df[path_col].dropna().astype(str).tolist()
    seen: set[str] = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    if max_plots is not None:
        paths = paths[:max_plots]
    return [Path(p) for p in paths]


def plot_phase_folded_lightcurve(
    csv_path: Path,
    *,
    period_days: float,
    phase_epoch_jd: float | None = None,
    value_mode: str = "mag",
    align_v_to_g: bool = False,
    out_path: Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = FIG_SINGLE_COL_LC_WIDE,
    clean_max_error_absolute: float = 1.0,
    clean_max_error_sigma: float = 5.0,
    filter_bad_cameras: bool = True,
    bad_camera_scatter_ratio: float = 2.5,
    return_filtered_cameras: bool = False,
) -> set[int] | None:
    """Plot a phase-folded light curve (0-2 cycles) for a given period."""
    if period_days <= 0 or (not np.isfinite(period_days)):
        raise ValueError("period_days must be positive and finite")

    df, filtered_cameras = load_lightcurve_df(
        csv_path,
        filter_bad_cameras_enabled=filter_bad_cameras,
        bad_camera_scatter_ratio=bad_camera_scatter_ratio,
        return_filtered_info=True,
    )
    df = clean_lc(
        df,
        max_error_absolute=clean_max_error_absolute,
        max_error_sigma=clean_max_error_sigma,
    )
    if df.empty:
        raise ValueError(f"Light curve file is empty after cleaning: {csv_path}")

    df["camera_label"] = camera_labels(df)

    asas_sn_id = csv_path.stem.split("-")[0]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        phase_plot = plot_phase_panel(
            ax,
            df,
            period_days=float(period_days),
            epoch_jd=phase_epoch_jd,
            value_mode=value_mode,
            align_v_to_g=align_v_to_g,
            group_by="band-camera",
            camera_col="camera_label",
            marker_size=4.0,
            title=f"{asas_sn_id} phase-folded (P={float(period_days):.5f} d)",
        )
        phase_diag = phase_plot.diagnostics or {}

    lag = float(phase_diag.get("phase_lag_g_v_cycles", np.nan))
    lag_abs = float(phase_diag.get("phase_lag_g_v_abs_cycles", np.nan))
    if np.isfinite(lag):
        lag_label = f"g-V lag {lag:+.3f} cyc"
        if np.isfinite(lag_abs):
            lag_label += f" (|lag| {lag_abs:.3f})"
        ax.text(
            0.99,
            0.98,
            lag_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 3},
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # De-duplicate legend labels while preserving order.
        seen: set[str] = set()
        uniq_h = []
        uniq_l = []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, fontsize=8, ncol=2, loc="best")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_publication_figure(fig, out_path, dpi=150, close=False)

    if show:
        finalize_publication_figure(fig)
        plt.show()
    else:
        plt.close(fig)

    if return_filtered_cameras:
        return filtered_cameras
    return None


def load_detection_results(table_path):
    """
    Load detection results Parquet with trimmed strings; used for metadata lookup.
    """
    if table_path is None:
        return None

    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"detection_results file not found: {table_path}")

    df = read_feature_table(table_path)
    if is_layer_first_frame(df):
        df = expand_feature_layers(df)
    df = df.fillna("")
    df = df.apply(lambda x: x.astype(str).str.strip() if x.dtype == "object" else x)
    path_col = next(
        (column for column in ("lc_path", "path", "DAT_Path") if column in df.columns),
        None,
    )
    if path_col is not None:
        df["Match_ID"] = df[path_col].apply(lambda p: Path(str(p)).stem if p else "")
    return df


def lookup_source_metadata(asassn_id=None, *, source_name=None, dat_path=None, csv_path=None):
    """
    Look up metadata (source/category/vsx class) from detection_results.
    """
    if csv_path is None:
        return None

    df = load_detection_results(csv_path)
    if df is None:
        return None
    matches = pd.DataFrame(columns=df.columns)

    # An exact path is the strongest identity available. Identifier fallback
    # is intentionally later because alternate reductions can share a stem.
    if dat_path:
        requested_path = str(dat_path).strip()
        for column in ("lc_path", "path", "DAT_Path"):
            if column not in df.columns:
                continue
            path_matches = df[column].astype("string").str.strip().eq(requested_path).fillna(False)
            if bool(path_matches.any()):
                matches = df.loc[path_matches]
                break
    if matches.empty and asassn_id:
        requested_id = str(asassn_id).strip()
        for column in ("Match_ID", "asas_sn_id"):
            if column not in df.columns:
                continue
            id_matches = df[column].astype("string").str.strip().eq(requested_id).fillna(False)
            if bool(id_matches.any()):
                matches = df.loc[id_matches]
                break
    if matches.empty and source_name and "Source" in df.columns:
        source_matches = (
            df["Source"].astype("string").str.strip().str.lower()
            .eq(str(source_name).strip().lower())
            .fillna(False)
        )
        matches = df.loc[source_matches]

    if matches.empty:
        return None

    row = matches.iloc[-1]
    matched_path = next(
        (
            row.get(column)
            for column in ("lc_path", "path", "DAT_Path")
            if column in row.index and not _is_missing_scalar(row.get(column))
        ),
        None,
    )
    stored_asas_sn_id = row.get("asas_sn_id")
    source_id = (
        row.get("Match_ID")
        if _is_missing_scalar(stored_asas_sn_id)
        else stored_asas_sn_id
    )
    return {
        "dat_path": matched_path,
        "source": row.get("Source"),
        "source_id": source_id,
        "category": row.get("Category"),
        "vsx_class": row.get("VSX_Class"),
    }


def lookup_metadata_for_path(path: Path, detection_results_csv=None):
    """
    Infer metadata for a given light-curve path.
    Falls back to brayden_candidates if no detection results table is provided.
    """
    path = Path(path)
    stem = path.stem
    source_type = "SkyPatrol" if path.suffix.lower() == ".csv" else "Internal"

    meta = lookup_source_metadata(asassn_id=stem, dat_path=str(path), csv_path=detection_results_csv)

    if not meta and "-" in stem:
        meta = lookup_source_metadata(asassn_id=stem.split("-")[0], csv_path=detection_results_csv)

    # Fallback to reproduction candidate metadata if no table metadata found.
    if not meta:
        from malca.evaluation.reproduce import brayden_candidates
        source_id = stem.split("-")[0]
        for candidate in brayden_candidates:
            if candidate.get("source_id") == source_id:
                meta = {
                    "source": candidate.get("source"),
                    "source_id": source_id,
                    "category": candidate.get("category"),
                    "data_source": source_type,
                }
                return meta
    
    if not meta:
        return {"data_source": source_type}

    meta = dict(meta)
    meta["data_source"] = source_type
    return meta


def load_passing_candidates(*args, **kwargs):
    """Forward to candidate-table loader used by consolidated plotting."""
    from malca.review.plot_batch import load_passing_candidates as _load_impl
    return _load_impl(*args, **kwargs)


def plot_passing_candidates(*args, **kwargs):
    """Forward to candidate-table plotting implementation."""
    from malca.review.plot_batch import plot_passing_candidates as _plot_impl
    return _plot_impl(*args, **kwargs)



BASELINE_FUNCTIONS = {
    "global_median": global_median_baseline,
    "per_camera_median": per_camera_median_baseline,
    "per_camera_gp": per_camera_gp_baseline,
    "gp": per_camera_gp_baseline,
    "gp_masked": per_camera_gp_baseline_masked,
    "phase_template": phase_template_baseline,
}

PER_CAMERA_BASELINES = {
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    phase_template_baseline,
}

PER_CAMERA_BASELINE_NAMES = {
    "per_camera_median",
    "per_camera_gp",
    "gp",
    "gp_masked",
    "phase_template",
}


def _is_missing_scalar(value: object) -> bool:
    """Return whether a table/config scalar carries no usable value."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    """Parse a persisted optional boolean without truthifying non-empty strings."""
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes", "y"}:
            return True
        if lowered in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def _match_detection_result_row(
    stored_results: pd.DataFrame,
    csv_path: Path,
    *,
    asas_sn_id: str | None = None,
) -> pd.Series | None:
    """Match a persisted result, preferring the exact LC path over identifiers."""
    if stored_results is None or stored_results.empty:
        return None

    requested_path = str(csv_path).strip()
    for column in ("lc_path", "path", "DAT_Path"):
        if column not in stored_results.columns:
            continue
        values = stored_results[column].astype("string").str.strip()
        matches = values.eq(requested_path).fillna(False)
        if bool(matches.any()):
            return stored_results.loc[matches].iloc[-1]

    # Identifier matching is deliberately a fallback. Two files can share an
    # ASAS-SN identifier (for example, alternate reductions), so it must never
    # override an exact-path row.
    stem = Path(csv_path).stem
    identifier_checks: list[tuple[str, str]] = []
    if "Match_ID" in stored_results.columns:
        identifier_checks.append(("Match_ID", stem))
    if asas_sn_id and "asas_sn_id" in stored_results.columns:
        identifier_checks.append(("asas_sn_id", str(asas_sn_id).strip()))
    for column, requested_id in identifier_checks:
        values = stored_results[column].astype("string").str.strip()
        matches = values.eq(requested_id).fillna(False)
        if bool(matches.any()):
            return stored_results.loc[matches].iloc[-1]
    return None


def _resolve_replay_baseline(
    baseline_func,
    baseline_name: str | None,
    baseline_kwargs: dict | None,
    stored_row: pd.Series | dict | None,
):
    """Resolve the candidate-level production baseline and its row-level inputs."""
    kwargs = dict(baseline_kwargs or {})
    source = ""
    if stored_row is not None:
        source_value = stored_row.get("baseline_source")
        baseline_value = stored_row.get("baseline_func")
        source = "" if _is_missing_scalar(source_value) else str(source_value).strip().lower()
        stored_baseline = "" if _is_missing_scalar(baseline_value) else str(baseline_value).strip().lower()
        if "phase_template" in source or stored_baseline == "phase_template":
            baseline_func = phase_template_baseline
            baseline_name = "phase_template"

    if baseline_func is phase_template_baseline or baseline_name == "phase_template":
        # The production periodic branch uses this exact row-level period. A
        # missing/invalid value intentionally reproduces its median fallback.
        if stored_row is not None:
            period_value = stored_row.get("pre_periodicity_selected_period")
            try:
                kwargs["period_days"] = float(period_value)
            except (TypeError, ValueError):
                kwargs["period_days"] = np.nan
    return baseline_func, baseline_name, kwargs


def _build_replay_score_kwargs(
    run_params: dict | None,
    *,
    logbf_threshold_dip: float,
    logbf_threshold_jump: float,
    filter_bad_cameras: bool,
    bad_camera_scatter_ratio: float,
) -> dict[str, object]:
    """Build the exact event-scoring arguments persisted by the detection run."""
    params = dict(run_params or {})
    score_kwargs: dict[str, object] = {
        "logbf_threshold_dip": params.get("logbf_threshold_dip", logbf_threshold_dip),
        "logbf_threshold_jump": params.get("logbf_threshold_jump", logbf_threshold_jump),
        "compute_event_prob": True,
        "filter_residual_bad_cameras_enabled": filter_bad_cameras,
        "bad_camera_scatter_ratio": bad_camera_scatter_ratio,
    }
    for key in (
        "p_points",
        "mag_points",
        "trigger_mode",
        "significance_threshold",
        "run_min_points",
        "max_gap_points",
        "run_max_gap_days",
        "run_min_duration_days",
        "p_min_dip",
        "p_max_dip",
        "p_min_jump",
        "p_max_jump",
    ):
        if key in params and params[key] is not None:
            score_kwargs[key] = params[key]
    if params.get("run_max_gap_points") is not None:
        score_kwargs["max_gap_points"] = params["run_max_gap_points"]

    if "compute_event_prob" in params:
        parsed = _optional_bool(params["compute_event_prob"], field_name="compute_event_prob")
        if parsed is not None:
            score_kwargs["compute_event_prob"] = parsed
    elif "no_event_prob" in params:
        parsed = _optional_bool(params["no_event_prob"], field_name="no_event_prob")
        if parsed is not None:
            score_kwargs["compute_event_prob"] = not parsed

    mag_points = int(score_kwargs.get("mag_points", 12))
    if mag_points <= 0:
        raise ValueError(f"mag_points must be positive for replay, got {mag_points}")
    for kind in ("dip", "jump"):
        direct_key = f"mag_grid_{kind}"
        if direct_key in params and not _is_missing_scalar(params[direct_key]):
            grid = np.asarray(params[direct_key], dtype=float)
            if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
                raise ValueError(f"Invalid configured {direct_key}: {params[direct_key]!r}")
            score_kwargs[direct_key] = grid
            continue
        lower = params.get(f"mag_min_{kind}")
        upper = params.get(f"mag_max_{kind}")
        if not _is_missing_scalar(lower) and not _is_missing_scalar(upper):
            score_kwargs[direct_key] = np.linspace(float(lower), float(upper), mag_points)
    return score_kwargs


def compare_detection_replay(
    stored: pd.Series | dict,
    replay: dict,
    *,
    atol: float = 1e-6,
) -> list[str]:
    """Return material differences between a stored event row and plot replay."""
    mismatches: list[str] = []
    for kind in ("dip", "jump"):
        stored_sig = stored.get(f"{kind}_significant")
        replay_kind = replay.get(kind) if isinstance(replay, dict) else None
        replay_kind = replay_kind if isinstance(replay_kind, dict) else {}
        replay_sig = replay_kind.get("significant")
        stored_sig_available = not _is_missing_scalar(stored_sig)
        replay_sig_available = not _is_missing_scalar(replay_sig)
        if stored_sig_available and not replay_sig_available:
            mismatches.append(f"{kind}_significant_replay_unavailable")
        elif replay_sig_available and not stored_sig_available:
            mismatches.append(f"{kind}_significant_stored_unavailable")
        elif stored_sig_available and replay_sig_available:
            try:
                stored_bool = _optional_bool(stored_sig, field_name=f"{kind}_significant")
                replay_bool = _optional_bool(replay_sig, field_name=f"replay.{kind}.significant")
                if stored_bool != replay_bool:
                    mismatches.append(f"{kind}_significant")
            except ValueError:
                mismatches.append(f"{kind}_significant_invalid")

        stored_delta = stored.get(f"{kind}_best_delta_mag")
        if _is_missing_scalar(stored_delta):
            stored_delta = stored.get(f"{kind}_best_mag_event")
        replay_delta = replay_kind.get("best_delta_mag")
        if _is_missing_scalar(replay_delta):
            replay_delta = replay_kind.get("best_mag_event")
        stored_delta_available = not _is_missing_scalar(stored_delta)
        replay_delta_available = not _is_missing_scalar(replay_delta)
        if stored_delta_available and not replay_delta_available:
            mismatches.append(f"{kind}_best_delta_mag_replay_unavailable")
            continue
        if replay_delta_available and not stored_delta_available:
            mismatches.append(f"{kind}_best_delta_mag_stored_unavailable")
            continue
        if not stored_delta_available and not replay_delta_available:
            continue
        try:
            stored_float = float(stored_delta)
            replay_float = float(replay_delta)
            if not np.isfinite(stored_float) and not np.isfinite(replay_float):
                continue
            if not np.isfinite(stored_float):
                mismatches.append(f"{kind}_best_delta_mag_stored_unavailable")
            elif not np.isfinite(replay_float):
                mismatches.append(f"{kind}_best_delta_mag_replay_unavailable")
            elif not np.isclose(stored_float, replay_float, atol=atol, rtol=1e-6):
                mismatches.append(f"{kind}_best_delta_mag")
        except (TypeError, ValueError):
            mismatches.append(f"{kind}_best_delta_mag_invalid")
    return mismatches


def plot_bayes_results(
    csv_path: Path,
    results_csv: Path | None = None,
    *,
    out_path: Path | None = None,
    figsize=PLOT_FIGSIZE,
    show=False,
    baseline_func=None,
    baseline_kwargs=None,
    logbf_threshold_dip=LOGBF_THRESHOLD_DIP,
    logbf_threshold_jump=LOGBF_THRESHOLD_JUMP,
    skip_events=False,
    plot_fits=False,
    jd_offset=JD_OFFSET,
    detection_results_csv=None,
    clean_max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
    annotations: dict[str, str] | None = None,
    # New parameters
    metadata: dict | None = None,
    run_params: dict | None = None,
    unified_layout: bool = True,
    legend_outside: bool = True,
    robust_threshold: bool = True,
    show_timestamp: bool = True,
    filter_bad_cameras: bool = True,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    return_filtered_cameras: bool = False,
) -> set[int] | None:
    """Plot a light curve with Bayesian detection results and run fits.
    
    Returns
    -------
    filtered_cameras : set[int] | None
        Set of camera IDs that were filtered out (only if return_filtered_cameras=True).
    """
                      
    df, filtered_cameras = load_lightcurve_df(
        csv_path,
        filter_bad_cameras_enabled=False,
        bad_camera_scatter_ratio=bad_camera_scatter_ratio,
        return_filtered_info=True,
    )
    df = clean_lc(
        df,
        max_error_absolute=clean_max_error_absolute,
        max_error_sigma=clean_max_error_sigma,
    )
    if df.empty:
        raise ValueError(f"Light curve file is empty: {csv_path.name}")
    if filter_bad_cameras and "camera#" in df.columns:
        df, pre_baseline_bad = filter_bad_camera_observations(
            df,
            lc_path=str(csv_path),
            filter_scatter=False,
            filter_offset=False,
            filter_catastrophic=True,
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )
        filtered_cameras = set(filtered_cameras or set()) | set(pre_baseline_bad)


    asas_sn_id = csv_path.stem.split("-")[0]

    # Load the persisted candidate row before resolving the baseline: a single
    # run can contain both the stochastic branch and row-level phase-template
    # branch, while run_params.json records only the run-wide default.
    replay_results_path = results_csv if results_csv is not None else detection_results_csv
    stored_result_row: pd.Series | None = None
    if not skip_events and replay_results_path is not None:
        replay_path = Path(replay_results_path)
        if not replay_path.exists():
            annotations = dict(annotations or {})
            annotations["replay_unavailable"] = f"stored results file not found: {replay_path}"
        else:
            try:
                stored_results = load_detection_results(replay_path)
                stored_result_row = _match_detection_result_row(
                    stored_results,
                    csv_path,
                    asas_sn_id=asas_sn_id,
                )
                if stored_result_row is None:
                    annotations = dict(annotations or {})
                    annotations["replay_unavailable"] = "no stored result row matched this light-curve path or identifier"
            except Exception as exc:
                annotations = dict(annotations or {})
                annotations["replay_unavailable"] = f"could not read stored result row ({type(exc).__name__}: {exc})"
    
                                                                   
    baseline_name = None
    if baseline_func is None:
        requested_baseline = None
        if run_params:
            requested_baseline = run_params.get("baseline_func", run_params.get("baseline"))
        baseline_func = BASELINE_FUNCTIONS.get(str(requested_baseline), per_camera_gp_baseline_masked)
    baseline_kwargs = dict(baseline_kwargs or {})
    if run_params:
        for source_key, target_key in (
            ("baseline_s0", "S0"),
            ("baseline_w0", "w0"),
            ("baseline_q", "q"),
            ("baseline_jitter", "jitter"),
            ("baseline_sigma_floor", "sigma_floor"),
        ):
            if source_key in run_params and run_params[source_key] is not None:
                baseline_kwargs.setdefault(target_key, run_params[source_key])
    # allow alias strings for baseline selection
    if isinstance(baseline_func, str):
        baseline_name = baseline_func
        baseline_func = BASELINE_FUNCTIONS.get(baseline_func, per_camera_gp_baseline)
    if baseline_func is per_camera_gp_baseline_masked and run_params:
        for key in (
            "allow_cross_band_consensus",
            "cross_band_min_overlap_points",
            "cross_band_min_overlap_days",
            "cross_band_clip_sigma",
        ):
            if key in run_params and run_params[key] is not None:
                baseline_kwargs.setdefault(key, run_params[key])
    if baseline_name is None:
        for name, func in BASELINE_FUNCTIONS.items():
            if func is baseline_func:
                baseline_name = name
                break
    baseline_func, baseline_name, baseline_kwargs = _resolve_replay_baseline(
        baseline_func,
        baseline_name,
        baseline_kwargs,
        stored_result_row,
    )
    per_camera_baseline = (
        baseline_func in PER_CAMERA_BASELINES
        or (baseline_name in PER_CAMERA_BASELINE_NAMES)
    )
    
    print(f"Analyzing {asas_sn_id}...")
    
                                             
    combined_result = None
    if skip_events:
        empty_res = {"significant": False, "run_summaries": [], "n_runs": 0}
        band_results = {0: {"dip": empty_res, "jump": empty_res}, 1: {"dip": empty_res, "jump": empty_res}}
        shared_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None
    else:
        # Replay the production decision on the combined, cleaned light curve.
        # Splitting bands here used to produce figures that could not reproduce
        # the stored candidate-level event decision.
        if baseline_func in {per_camera_gp_baseline, per_camera_gp_baseline_masked}:
            baseline_kwargs.setdefault("add_sigma_eff_col", True)

        score_kwargs = _build_replay_score_kwargs(
            run_params,
            logbf_threshold_dip=logbf_threshold_dip,
            logbf_threshold_jump=logbf_threshold_jump,
            filter_bad_cameras=filter_bad_cameras,
            bad_camera_scatter_ratio=bad_camera_scatter_ratio,
        )

        combined_result = score_lightcurve(
            df,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            **score_kwargs,
        )
        df = combined_result["df"].copy()
        shared_base = combined_result.get("df_base")
        replay_payload = {"dip": combined_result["dip"], "jump": combined_result["jump"]}
        band_results = {0: replay_payload, 1: replay_payload}
        filtered_cameras = set(filtered_cameras or set()) | set(
            combined_result.get("bad_cameras_filtered", set()) or set()
        )

        if stored_result_row is not None:
            mismatches = compare_detection_replay(stored_result_row, replay_payload)
            if mismatches:
                annotations = dict(annotations or {})
                annotations["replay_warning"] = ", ".join(mismatches)
    
                               
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    median_jd = df["JD"].median()
    if median_jd > 2000000:
        df["JD_plot"] = df["JD"] - jd_offset
    else:
        df["JD_plot"] = df["JD"] - 8000.0
    
    bands = [0, 1]
    band_labels = {0: "g", 1: "V"}
    band_markers = {0: "o", 1: "s"}

    # Set up camera labels
    if "camera_name" in df.columns:
        df["camera_label"] = df["camera_name"].astype(str)
    elif "camera#" in df.columns:
        df["camera_label"] = df["camera#"].astype(str)
    elif "camera" in df.columns:
        df["camera_label"] = df["camera"].astype(str)
    else:
        df["camera_label"] = "unknown"

    camera_ids = sorted(df["camera_label"].dropna().unique())
    camera_colors = {cam: _stable_camera_color(cam) for cam in camera_ids}

    # Use the exact baseline frame used for event replay.
    if shared_base is not None:
        if len(shared_base) != len(df):
            raise RuntimeError("Plot replay baseline is not aligned with the prepared light curve")
        shared_base = shared_base.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df["baseline"] = pd.to_numeric(shared_base["baseline"], errors="coerce")
        df["resid"] = df["mag"] - df["baseline"]

    # Split only for rendering after the combined scientific decision is fixed.
    band_dfs = {}
    all_resids = []
    for band in bands:
        band_df = df[df["v_g_band"] == band].copy()
        if band_df.empty:
            continue
        if "resid" in band_df.columns:
            all_resids.extend(band_df["resid"].dropna().tolist())
        band_dfs[band] = band_df

    # Compute robust 3-sigma threshold from all residuals
    if robust_threshold and len(all_resids) > 10:
        all_resids_arr = np.array(all_resids)
        all_resids_finite = all_resids_arr[np.isfinite(all_resids_arr)]
        if len(all_resids_finite) > 10:
            mad = MAD_SCALE * np.median(np.abs(all_resids_finite - np.median(all_resids_finite)))
            sigma_5 = 5 * mad
        else:
            sigma_5 = 0.5  # fallback ~5-sigma for typical scatter
    else:
        sigma_5 = 0.5

    # Create unified 2x1 layout (both bands on same axes)
    if unified_layout:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax_main = axes[0]
        ax_resid = axes[1]
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col")

    # Track legend handles for unified legend
    legend_handles = []
    legend_labels_seen = set()

    # Track if we have significant dips/jumps for legend
    has_dip = False
    has_jump = False

    # Plot each band
    for band in bands:
        if band not in band_dfs:
            continue
        band_df = band_dfs[band]
        band_label = band_labels[band]
        marker = band_markers[band]

        if not unified_layout:
            band_idx = 0 if band == 1 else 1  # V first, then g
            ax_main = axes[0, band_idx]
            ax_resid = axes[1, band_idx]
            ax_main.invert_yaxis()

        # Plot data points by camera
        for cam in camera_ids:
            subset = band_df[band_df["camera_label"] == cam]
            if subset.empty:
                continue
            color = camera_colors[cam]

            # Label includes band for unified layout
            if unified_layout:
                label = f"{cam} ({band_label})"
            else:
                label = f"{cam}"

            # Avoid duplicate legend entries
            if label in legend_labels_seen:
                label = None
            else:
                legend_labels_seen.add(label)

            ax_main.errorbar(
                subset["JD_plot"],
                subset["mag"],
                yerr=subset["error"],
                fmt=marker,
                ms=5,
                color=color,
                alpha=0.7,
                ecolor=color,
                elinewidth=0.8,
                capsize=1.5,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=label,
            )

        # Plot baseline
        if "baseline" in band_df.columns:
            baseline_finite = band_df[np.isfinite(band_df["baseline"])]
            if not baseline_finite.empty:
                if per_camera_baseline:
                    for cam in camera_ids:
                        cam_baseline = baseline_finite[baseline_finite["camera_label"] == cam]
                        if cam_baseline.empty:
                            continue
                        cam_sorted = cam_baseline.sort_values("JD_plot")
                        ax_main.plot(
                            cam_sorted["JD_plot"],
                            cam_sorted["baseline"],
                            color=camera_colors[cam],
                            linestyle="-",
                            linewidth=1.6,
                            alpha=0.8,
                            zorder=5,
                        )
                else:
                    baseline_sorted = baseline_finite.sort_values("JD_plot")
                    ax_main.plot(
                        baseline_sorted["JD_plot"],
                        baseline_sorted["baseline"],
                        color="orange",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                        label="Baseline",
                        zorder=5,
                    )

        # Plot event markers
        band_res = band_results[band]
        dip = band_res["dip"]
        jump = band_res["jump"]

        # Plot dips
        if (not skip_events) and dip["significant"] and dip.get("run_summaries"):
            has_dip = True
            for run_summary in dip["run_summaries"]:
                jd_start = run_summary["start_jd"]
                jd_end = run_summary["end_jd"]

                jd_plot_start = jd_start - (jd_offset if median_jd > 2000000 else 8000.0)
                jd_plot_end = jd_end - (jd_offset if median_jd > 2000000 else 8000.0)
                ax_main.axvline(jd_plot_start, color=DIP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.5)
                if jd_plot_end != jd_plot_start:
                    ax_main.axvline(jd_plot_end, color=DIP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.5)

                morph = run_summary.get("morphology", "none")
                params = run_summary.get("params", {})

                if morph == "gaussian" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color=DIP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.0)
                    if plot_fits:
                        sigma = params.get("sigma", (jd_end - jd_start) / 4)
                        amp = params.get("amp", 0.1)
                        baseline_val = params.get("baseline", band_df["mag"].median())
                        t_fit = np.linspace(jd_start - 3 * sigma, jd_end + 3 * sigma, 100)
                        mag_fit = gaussian(t_fit, amp, t0, sigma, baseline_val)
                        t_fit_plot = t_fit - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.plot(
                            t_fit_plot,
                            mag_fit,
                            color=DIP_EVENT_COLOR,
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

                elif morph == "paczynski" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color=DIP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.0)
                    if plot_fits:
                        tE = params.get("tE", (jd_end - jd_start) / 2)
                        amp = params.get("amp", -0.1)
                        baseline_val = params.get("baseline", band_df["mag"].median())
                        t_fit = np.linspace(jd_start - 3 * tE, jd_end + 3 * tE, 100)
                        mag_fit = paczynski_kernel(t_fit, amp, t0, tE, baseline_val)
                        t_fit_plot = t_fit - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.plot(
                            t_fit_plot,
                            mag_fit,
                            color=DIP_EVENT_COLOR,
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

        # Plot jumps
        if (not skip_events) and jump["significant"] and jump.get("run_summaries"):
            has_jump = True
            for run_summary in jump["run_summaries"]:
                jd_start = run_summary["start_jd"]
                jd_end = run_summary["end_jd"]

                jd_plot_start = jd_start - (jd_offset if median_jd > 2000000 else 8000.0)
                jd_plot_end = jd_end - (jd_offset if median_jd > 2000000 else 8000.0)
                ax_main.axvline(jd_plot_start, color=JUMP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.5)
                if jd_plot_end != jd_plot_start:
                    ax_main.axvline(jd_plot_end, color=JUMP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.5)

                morph = run_summary.get("morphology", "none")
                params = run_summary.get("params", {})

                if morph == "gaussian" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color=JUMP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.0)
                    if plot_fits:
                        sigma = params.get("sigma", (jd_end - jd_start) / 4)
                        amp = params.get("amp", -0.1)
                        baseline_val = params.get("baseline", band_df["mag"].median())
                        t_fit = np.linspace(jd_start - 3 * sigma, jd_end + 3 * sigma, 100)
                        mag_fit = gaussian(t_fit, amp, t0, sigma, baseline_val)
                        t_fit_plot = t_fit - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.plot(
                            t_fit_plot,
                            mag_fit,
                            color=JUMP_EVENT_COLOR,
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

                elif morph == "paczynski" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color=JUMP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.0)
                    if plot_fits:
                        tE = params.get("tE", (jd_end - jd_start) / 2)
                        amp = params.get("amp", -0.1)
                        baseline_val = params.get("baseline", band_df["mag"].median())
                        t_fit = np.linspace(jd_start - 3 * tE, jd_end + 3 * tE, 100)
                        mag_fit = paczynski_kernel(t_fit, amp, t0, tE, baseline_val)
                        t_fit_plot = t_fit - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.plot(
                            t_fit_plot,
                            mag_fit,
                            color=JUMP_EVENT_COLOR,
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

                elif morph == "fred" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color=JUMP_EVENT_COLOR, linestyle="--", alpha=0.7, linewidth=1.0)
                    if plot_fits:
                        tau = params.get("tau", 0.05)
                        amp = params.get("amp", -0.1)
                        baseline_val = params.get("baseline", band_df["mag"].median())
                        t_fit = np.linspace(jd_start - 3 * tau, jd_end + 3 * tau, 100)
                        dt = t_fit - t0
                        decay = np.where(dt >= 0, np.exp(-dt / tau), 0.0)
                        mag_fit = baseline_val + amp * decay

                        t_fit_plot = t_fit - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.plot(
                            t_fit_plot,
                            mag_fit,
                            color=JUMP_EVENT_COLOR,
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

        # Plot residuals
        if "resid" in band_df.columns:
            for cam in camera_ids:
                subset = band_df[band_df["camera_label"] == cam]
                if subset.empty:
                    continue
                color = camera_colors[cam]
                ax_resid.scatter(
                    subset["JD_plot"],
                    subset["resid"],
                    s=15,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    marker=marker,
                )

        # For non-unified layout, configure axes per-band
        if not unified_layout:
            ax_main.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax_main.set_xlabel(f"JD - {int(jd_offset)} [d]", fontsize=10)
            ax_main.xaxis.set_label_position("top")
            ax_main.set_ylabel(f"{band_labels[band]} band [mag]", fontsize=12)
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc="best", fontsize=8, ncol=2)

            if "resid" in band_df.columns:
                jd_min = band_df["JD_plot"].min()
                jd_max = band_df["JD_plot"].max()
                ax_resid.fill_between([jd_min, jd_max], sigma_5, 100, color="lightgrey", alpha=0.5, zorder=0)
                ax_resid.fill_between([jd_min, jd_max], -sigma_5, -100, color="lightgrey", alpha=0.45, zorder=0)
                ax_resid.axhline(0.0, color="black", linestyle="--", alpha=0.4, zorder=1)
                ax_resid.axhline(sigma_5, color="black", linestyle="-", linewidth=0.8, zorder=1)
                ax_resid.axhline(-sigma_5, color="black", linestyle="-", linewidth=0.8, zorder=1)
                ax_resid.set_ylabel(f"{band_labels[band]} residual [mag]", fontsize=12)
                ax_resid.grid(True, alpha=0.3)
                ax_resid.invert_yaxis()
                resid_min, resid_max = band_df["resid"].min(), band_df["resid"].max()
                pad = (resid_max - resid_min) * 0.1 if resid_max != resid_min else 0.1
                ax_resid.set_ylim(max(resid_max + pad, sigma_5 + 0.05), min(resid_min - pad, -sigma_5 - 0.05))
            ax_resid.set_xlabel(f"JD - {int(jd_offset)} [d]", fontsize=10)

    # Configure unified axes (after plotting all bands)
    if unified_layout:
        ax_main.invert_yaxis()
        ax_main.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax_main.set_xlabel(f"JD - {int(jd_offset)} [d]", fontsize=10)
        ax_main.xaxis.set_label_position("top")
        ax_main.set_ylabel("Magnitude [mag]", fontsize=12)
        ax_main.grid(True, alpha=0.3)

        # Build legend with Line2D handles for events
        handles, labels = ax_main.get_legend_handles_labels()
        legend_handles = list(zip(handles, labels))

        # Add event color legend entries
        if has_dip:
            legend_handles.append((Line2D([0], [0], color=DIP_EVENT_COLOR, linestyle="--", linewidth=1.5), "Dip"))
        if has_jump:
            legend_handles.append((Line2D([0], [0], color=JUMP_EVENT_COLOR, linestyle="--", linewidth=1.5), "Jump"))

        if legend_handles:
            final_handles = [h for h, _ in legend_handles]
            final_labels = [l for _, l in legend_handles]
            if legend_outside:
                ax_main.legend(final_handles, final_labels, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
            else:
                ax_main.legend(final_handles, final_labels, loc="best", fontsize=8, ncol=2)

        # Residual panel for unified layout
        jd_min = df["JD_plot"].min()
        jd_max = df["JD_plot"].max()
        ax_resid.fill_between([jd_min, jd_max], sigma_5, 100, color="lightgrey", alpha=0.5, zorder=0)
        ax_resid.fill_between([jd_min, jd_max], -sigma_5, -100, color="lightgrey", alpha=0.45, zorder=0)
        ax_resid.axhline(0.0, color="black", linestyle="--", alpha=0.4, zorder=1)
        ax_resid.axhline(sigma_5, color="black", linestyle="-", linewidth=0.8, zorder=1)
        ax_resid.axhline(-sigma_5, color="black", linestyle="-", linewidth=0.8, zorder=1)
        ax_resid.set_ylabel("Residual [mag]", fontsize=12)
        ax_resid.grid(True, alpha=0.3)
        ax_resid.invert_yaxis()

        if all_resids:
            resid_min, resid_max = min(all_resids), max(all_resids)
            pad = (resid_max - resid_min) * 0.1 if resid_max != resid_min else 0.1
            ax_resid.set_ylim(max(resid_max + pad, sigma_5 + 0.05), min(resid_min - pad, -sigma_5 - 0.05))

        ax_resid.set_xlabel(f"JD - {int(jd_offset)} [d]", fontsize=10)
    


    # Merge metadata from lookup with passed-in metadata (passed-in takes precedence)
    meta = lookup_metadata_for_path(csv_path, detection_results_csv=detection_results_csv) or {}
    if metadata:
        meta.update(metadata)
    meta = normalize_vsx_record(meta)

    # Build title: "Source (ID) – VSX Class – Category – JD range"
    source_name = meta.get("source")
    vsx_class = meta.get("vsx_class")
    category = meta.get("category")
    external_id = meta.get("external_id")
    trigger_type = meta.get("trigger_type")

    # Start with source name (ID) format
    if source_name and asas_sn_id:
        label = f"{source_name} ({asas_sn_id})"
    elif asas_sn_id:
        label = str(asas_sn_id)
    elif source_name:
        label = str(source_name)
    else:
        label = "Source"

    # Calculate JD range from the data
    jd_start_val = float(df["JD"].min())
    jd_end_val = float(df["JD"].max())
    jd_label = f"JD {jd_start_val:.0f}-{jd_end_val:.0f}"

    title_parts = [label]
    if vsx_class:
        title_parts.append(f"VSX: {vsx_class}")
    if category:
        title_parts.append(str(category))
    title_parts.append(jd_label)

    if not skip_events:
        combined_dip = band_results[0]["dip"]
        combined_jump = band_results[0]["jump"]

        if combined_dip["significant"]:
            title_parts.append(f"Dips: {combined_dip.get('n_runs', 0)} combined-band runs")
        if combined_jump["significant"]:
            title_parts.append(f"Jumps: {combined_jump.get('n_runs', 0)} combined-band runs")

    fig.suptitle(" – ".join(title_parts), fontsize=14)

    # Build info panel content (displayed at bottom-right, expands upward)
    info_lines = []

    # Source classification
    if vsx_class:
        info_lines.append(f"VSX class: {vsx_class}")

    # External IDs
    if external_id:
        info_lines.append(f"External ID: {external_id}")
    if meta.get("asas_sn_id"):
        info_lines.append(f"ASAS-SN ID: {meta['asas_sn_id']}")

    # Coordinates and Gaia
    if annotations:
        if annotations.get("RA") is not None and annotations.get("Dec") is not None:
            info_lines.append(f"RA/Dec: {annotations['RA']}, {annotations['Dec']}")
        if annotations.get("Gaia_ID") is not None:
            info_lines.append(f"Gaia ID: {annotations['Gaia_ID']}")
        if annotations.get("RUWE") is not None:
            info_lines.append(f"Gaia RUWE: {annotations['RUWE']}")

    if trigger_type:
        info_lines.append(f"Trigger: {trigger_type}")

    # Scores and filter results from annotations
    if annotations:
        if annotations.get("replay_unavailable") is not None:
            info_lines.append(f"REPLAY UNAVAILABLE: {annotations['replay_unavailable']}")
        if annotations.get("replay_warning") is not None:
            info_lines.append(f"REPLAY MISMATCH: {annotations['replay_warning']}")
        # Dipper/jumper scores
        if annotations.get("dipper_score") is not None:
            info_lines.append(f"Dipper score: {annotations['dipper_score']}")
        if annotations.get("jumper_score") is not None:
            info_lines.append(f"Jumper score: {annotations['jumper_score']}")
        # Bayes factors (already log scale)
        if annotations.get("dip_logBF") is not None:
            info_lines.append(f"Dip logBF: {annotations['dip_logBF']}")
        if annotations.get("jump_logBF") is not None:
            info_lines.append(f"Jump logBF: {annotations['jump_logBF']}")
        # Morphology
        if annotations.get("dip_morph") is not None:
            info_lines.append(f"Dip morph: {annotations['dip_morph']}")
        if annotations.get("jump_morph") is not None:
            info_lines.append(f"Jump morph: {annotations['jump_morph']}")
        # Run counts
        if annotations.get("dip_runs") is not None:
            info_lines.append(f"Dip runs: {annotations['dip_runs']}")
        if annotations.get("jump_runs") is not None:
            info_lines.append(f"Jump runs: {annotations['jump_runs']}")
        # Periodic match
        if annotations.get("periodic") is not None:
            info_lines.append(f"Periodic match: {annotations['periodic']}")

    # Run params (compact format)
    if run_params:
        params_to_show = ["baseline", "logbf_threshold_dip", "logbf_threshold_jump"]
        for key in params_to_show:
            if key in run_params:
                # Shorten key names for display
                display_key = key.replace("logbf_threshold_", "logBF_thr_")
                info_lines.append(f"{display_key}: {run_params[key]}")

    # Filtered cameras
    if filtered_cameras:
        cams_str = ",".join(str(c) for c in sorted(filtered_cameras))
        info_lines.append(f"Cameras filtered: {cams_str}")

    # Timestamp
    if show_timestamp:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        info_lines.append(f"Generated: {timestamp_str}")

    # Shared metadata fields for GUI/TUI/PNG parity
    existing_lines = set(info_lines)
    for label, key in REVIEW_METADATA_FIELDS:
        if key not in meta:
            continue
        val = meta.get(key)
        if val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        if isinstance(val, str) and not val.strip():
            continue
        line = f"{label}: {val}"
        if line not in existing_lines:
            info_lines.append(line)
            existing_lines.add(line)

    # Display info panel at bottom-right, expanding upward
    if info_lines and unified_layout:
        info_text = "\n".join(info_lines)
        fig.text(
            1.02, 0.02, info_text,
            transform=ax_resid.transAxes,
            ha="left", va="bottom", fontsize=9,
            fontfamily="monospace", color="black",
            bbox=dict(boxstyle="round,pad=0.4", fc="0.95", ec="0.7", alpha=0.95),
        )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_publication_figure(fig, out_path, dpi=150, close=False)
        print(f"Saved plot to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_filtered_cameras:
        return filtered_cameras
    return fig


def _prepare_results_mode_input(
    results_path: Path,
    *,
    path_col: str,
    only_significant: bool,
) -> pd.DataFrame:
    """Apply results-mode CLI selection while preserving the complete rows."""
    df = read_feature_table(Path(results_path))
    if is_layer_first_frame(df):
        df = expand_feature_layers(df)

    effective_path_col = str(path_col)
    if effective_path_col not in df.columns:
        # Canonical products use lc_path; keep the historical default usable.
        if effective_path_col == "path" and "lc_path" in df.columns:
            effective_path_col = "lc_path"
        else:
            raise KeyError(f"Missing '{path_col}' column in {results_path}")

    if only_significant:
        significance_columns = [
            column for column in ("dip_significant", "jump_significant") if column in df.columns
        ]
        if not significance_columns:
            raise KeyError(
                "--only-significant requires dip_significant and/or jump_significant in the results table"
            )
        significant = pd.Series(False, index=df.index, dtype=bool)
        for column in significance_columns:
            parsed = df[column].map(
                lambda value: _optional_bool(value, field_name=column)
            )
            significant |= parsed.eq(True).fillna(False)
        df = df.loc[significant].copy()
    else:
        df = df.copy()

    # plot_passing_candidates consumes canonical lc_path. Always overwrite it
    # so an explicit --path-col cannot be shadowed by a stale existing column.
    df["lc_path"] = df[effective_path_col]
    return df


def _write_plot_log(
    args: argparse.Namespace,
    *,
    gp_kwargs: dict[str, object],
    total_plots: int,
    summary: dict[str, object] | None = None,
) -> None:
    """Write the detect-run plot log on every successful CLI exit path."""
    if not args.detect_run:
        return
    detect_run = Path(args.detect_run).expanduser()
    plot_log_file = detect_run / "plot_log.json"
    orig_argv = getattr(sys, "orig_argv", None)
    cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)
    plot_log: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "command": cmd,
        "results_file": str(args.results) if args.results else None,
        "output_dir": str(args.out_dir),
        "plot_params": {
            "baseline": args.baseline,
            "logbf_threshold_dip": args.logbf_threshold_dip,
            "logbf_threshold_jump": args.logbf_threshold_jump,
            "skip_events": args.skip_events,
            "plot_fits": args.plot_fits,
            "format": args.format,
            "only_significant": args.only_significant,
            "path_col": args.path_col,
            "jd_offset": args.jd_offset,
            "clean_max_error_absolute": args.clean_max_error_absolute,
            "clean_max_error_sigma": args.clean_max_error_sigma,
        },
        "results": {
            "total_plots": int(total_plots),
            "max_plots_limit": args.max_plots,
        },
    }
    if gp_kwargs:
        plot_log["plot_params"]["gp_params"] = gp_kwargs
    if summary is not None:
        plot_log["results"]["batch_summary"] = summary
    plot_log_file.parent.mkdir(parents=True, exist_ok=True)
    plot_log_file.write_text(
        json.dumps(plot_log, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves with Bayesian event detection results"
    )
    g_input = parser.add_argument_group("Input")
    g_selection = parser.add_argument_group("Selection")
    g_output = parser.add_argument_group("Output")
    g_baseline = parser.add_argument_group("Baseline & detection")
    g_gp = parser.add_argument_group("GP parameters")
    g_general = parser.add_argument_group("General")

    g_input.add_argument(
        "--detect-run",
        type=Path,
        default=None,
        help=f"STV run directory (e.g., {DEFAULT_OUTPUT_DIR / 'runs' / 'stv' / '20250121_143052'}). If specified, reads events from <detect-run>/results/ and writes plots to <detect-run>/plots/",
    )
    g_input.add_argument(
        "--input",
        nargs="+",
        help="Path(s) to light curve file(s) (glob patterns supported)",
    )
    g_input.add_argument(
        "--results",
        type=Path,
        help="Events/post-filter output Parquet with a path column (overrides --detect-run).",
    )
    g_input.add_argument(
        "--path-col",
        default="path",
        help="Column in events/post-filter output that contains LC paths.",
    )
    g_selection.add_argument(
        "--only-significant",
        action="store_true",
        help="If events output has dip_significant/jump_significant, plot only those.",
    )
    g_selection.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Maximum number of light curves to plot.",
    )
    g_selection.add_argument(
        "--ignore-failed-any",
        action="store_true",
        help="Do not require failed_any == False when plotting from --results.",
    )
    g_selection.add_argument(
        "--require-flag",
        action="append",
        default=[],
        help="Require this boolean flag column to be True (repeatable, --results mode).",
    )
    g_selection.add_argument(
        "--exclude-flag",
        action="append",
        default=[],
        help="Exclude rows where this boolean flag column is True (repeatable, --results mode).",
    )
    g_selection.add_argument(
        "--min-lsp-power",
        type=float,
        default=None,
        help="Require lsp_power >= this value (--results mode).",
    )
    g_selection.add_argument(
        "--max-lsp-bootstrap-sig",
        type=float,
        default=None,
        help="Require lsp_bootstrap_sig <= this value (--results mode).",
    )
    g_selection.add_argument(
        "--min-periodicity-score",
        type=float,
        default=None,
        help="Require periodicity_score >= this value (--results mode).",
    )
    g_output.add_argument(
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to <detect-run>/plots/ if --detect-run is used)",
    )
    g_baseline.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINE_FUNCTIONS.keys()),
        default=None,
        help="Override the detection baseline (otherwise read from --detect-run)",
    )
    g_baseline.add_argument(
        "--logbf-threshold-dip",
        type=float,
        default=None,
        help="Override the detection-run dip log-BF threshold",
    )
    g_baseline.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=None,
        help="Override the detection-run jump log-BF threshold",
    )
    g_baseline.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip Bayesian event detection; plot baseline/residuals only",
    )
    g_baseline.add_argument(
        "--plot-fits",
        action="store_true",
        help="Plot Gaussian/Paczynski fit curves in addition to peak markers.",
    )
    g_baseline.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output format for plots (default: png).",
    )
    g_baseline.add_argument(
        "--jd-offset",
        type=float,
        default=JD_OFFSET,
        help="JD offset for plotting (default: 2458000.0)",
    )
    g_baseline.add_argument(
        "--clean-max-error-absolute",
        type=float,
        default=CLEAN_LC_MAX_ERROR_ABSOLUTE,
        help="Absolute error cutoff for clean_lc (default: 1.0)",
    )
    g_baseline.add_argument(
        "--clean-max-error-sigma",
        type=float,
        default=CLEAN_LC_MAX_ERROR_SIGMA,
        help="Sigma cutoff for clean_lc MAD filter (default: 5.0)",
    )
    g_gp.add_argument("--gp-sigma", type=float, default=None, help="GP sigma parameter.")
    g_gp.add_argument("--gp-rho", type=float, default=None, help="GP rho parameter.")
    g_gp.add_argument("--gp-q", type=float, default=None, help="GP Q parameter (default: 0.7).")
    g_gp.add_argument("--gp-s0", type=float, default=None, help="GP S0 parameter (alt parameterization).")
    g_gp.add_argument("--gp-w0", type=float, default=None, help="GP w0 parameter (alt parameterization).")
    g_gp.add_argument("--gp-jitter", type=float, default=None, help="GP jitter term (default: 0.006).")
    g_gp.add_argument("--gp-sigma-floor", type=float, default=None, help="Extra GP sigma floor.")
    g_gp.add_argument("--gp-floor-clip", type=float, default=None, help="Sigma floor clipping threshold.")
    g_gp.add_argument("--gp-floor-iters", type=int, default=None, help="Sigma floor clipping iterations.")
    g_gp.add_argument("--gp-min-floor-points", type=int, default=None, help="Minimum points for sigma floor.")
    g_general.add_argument("--show", action="store_true", help="Show plots interactively")
    g_general.add_argument("--workers", type=int, default=WORKERS, help="Parallel workers for candidate plotting")
    g_general.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    g_general.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    g_general.add_argument(
        "--no-filter-bad-cameras",
        dest="filter_bad_cameras",
        action="store_false",
        default=True,
        help="Disable bad-camera filtering before plotting",
    )
    g_general.add_argument(
        "--bad-camera-scatter-ratio",
        type=float,
        default=2.5,
        help="Scatter ratio threshold for bad camera filtering",
    )

    args = parser.parse_args()

    run_params: dict = {}

    # Handle --detect-run for events and output directory
    if args.detect_run:
        detect_run = args.detect_run.expanduser()

        run_params_path = detect_run / "run_params.json"
        if not run_params_path.exists():
            raise FileNotFoundError(
                f"Detection run is missing run_params.json required for exact plot replay: {run_params_path}"
            )
        try:
            with run_params_path.open() as handle:
                loaded_params = json.load(handle)
        except Exception as exc:
            raise ValueError(f"Could not read detection run parameters from {run_params_path}: {exc}") from exc
        if not isinstance(loaded_params, dict):
            raise ValueError(f"Detection run parameters are not a JSON object: {run_params_path}")
        run_params = dict(loaded_params)

        # Set events path if not explicitly provided
        if not args.results:
            results_dir = detect_run / "results"
            from malca.review.plot_batch import _resolve_filtered_result

            args.results = _resolve_filtered_result(results_dir)
            print(f"Using results from: {args.results}")

        # Set out_dir if not explicitly provided
        if not args.out_dir:
            args.out_dir = detect_run / "plots"

    # Validate that we have an output directory
    if not args.out_dir:
        raise ValueError("Must specify either --output-dir or --detect-run")

    effective_baseline = args.baseline or str(run_params.get("baseline_func") or "").strip() or None
    if effective_baseline is None:
        raise ValueError(
            "Exact plot replay requires --baseline when --detect-run is not provided"
        )
    if effective_baseline not in BASELINE_FUNCTIONS:
        raise ValueError(f"Unknown detection baseline: {effective_baseline}")
    args.baseline = effective_baseline
    run_params["baseline_func"] = effective_baseline

    if args.logbf_threshold_dip is None:
        stored = run_params.get("logbf_threshold_dip")
        args.logbf_threshold_dip = float(LOGBF_THRESHOLD_DIP if stored is None else stored)
    else:
        run_params["logbf_threshold_dip"] = float(args.logbf_threshold_dip)
    if args.logbf_threshold_jump is None:
        stored = run_params.get("logbf_threshold_jump")
        args.logbf_threshold_jump = float(LOGBF_THRESHOLD_JUMP if stored is None else stored)
    else:
        run_params["logbf_threshold_jump"] = float(args.logbf_threshold_jump)

    baseline_func = BASELINE_FUNCTIONS[args.baseline]
    baseline_kwargs = {}

    gp_kwargs = {
        "sigma": args.gp_sigma,
        "rho": args.gp_rho,
        "q": args.gp_q,
        "S0": args.gp_s0,
        "w0": args.gp_w0,
        "jitter": args.gp_jitter,
        "sigma_floor": args.gp_sigma_floor,
        "floor_clip": args.gp_floor_clip,
        "floor_iters": args.gp_floor_iters,
        "min_floor_points": args.gp_min_floor_points,
    }
    gp_kwargs = {k: v for k, v in gp_kwargs.items() if v is not None}
    if gp_kwargs:
        if baseline_func in {per_camera_gp_baseline, per_camera_gp_baseline_masked}:
            baseline_kwargs.update(gp_kwargs)
        else:
            print("Warning: GP parameters were provided but baseline is not a GP baseline; ignoring.", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)


    if args.results:
        plot_input = _prepare_results_mode_input(
            args.results,
            path_col=args.path_col,
            only_significant=bool(args.only_significant),
        )
        summary = plot_passing_candidates(
            plot_input,
            args.out_dir,
            require_failed_any_false=not args.ignore_failed_any,
            require_flags=args.require_flag,
            exclude_flags=args.exclude_flag,
            min_lsp_power=args.min_lsp_power,
            max_lsp_bootstrap_sig=args.max_lsp_bootstrap_sig,
            min_periodicity_score=args.min_periodicity_score,
            max_plots=args.max_plots,
            baseline=args.baseline,
            baseline_kwargs=baseline_kwargs,
            skip_events=args.skip_events,
            plot_fits=args.plot_fits,
            format=args.format,
            show=args.show,
            verbose=args.verbose,
            workers=args.workers,
            logbf_threshold_dip=args.logbf_threshold_dip,
            logbf_threshold_jump=args.logbf_threshold_jump,
            jd_offset=args.jd_offset,
            clean_max_error_absolute=args.clean_max_error_absolute,
            clean_max_error_sigma=args.clean_max_error_sigma,
            detection_results_csv=args.results,
            run_params=run_params,
            filter_bad_cameras=args.filter_bad_cameras,
            bad_camera_scatter_ratio=args.bad_camera_scatter_ratio,
            show_tqdm=not args.no_progress,
        )
        print(f"Generated {summary.get('plotted', 0)} candidate plots in {args.out_dir}")
        _write_plot_log(
            args,
            gp_kwargs=gp_kwargs,
            total_plots=int(summary.get("plotted", 0)),
            summary=summary,
        )
        return
    elif args.input:
        csv_paths = [Path(p) for p in args.input]
    else:
        csv_paths = []

    if args.results and args.results.exists():
        results_df = read_feature_table(args.results)

        results_ids: set[str] = set()
        for p in results_df["path"].dropna().astype(str):
            results_ids.add(Path(p).stem.split("-")[0])

        csv_paths = [p for p in csv_paths if p.stem.split("-")[0] in results_ids]
        print(f"Filtered to {len(csv_paths)} light curves from results Parquet")

    if not csv_paths:
        raise SystemExit("No light curve paths provided (use --input or --results).")

    if args.max_plots is not None:
        csv_paths = csv_paths[: args.max_plots]
    
                           
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Light curve file does not exist: {csv_path}")

        asas_sn_id = csv_path.stem.split("-")[0]
        out_path = args.out_dir / f"{asas_sn_id}_dips.{args.format}"

        plot_bayes_results(
            csv_path,
            out_path=out_path,
            show=args.show,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            logbf_threshold_dip=args.logbf_threshold_dip,
            logbf_threshold_jump=args.logbf_threshold_jump,
            skip_events=args.skip_events,
            plot_fits=args.plot_fits,
            jd_offset=args.jd_offset,
            detection_results_csv=args.results,
            run_params=run_params,
            clean_max_error_absolute=args.clean_max_error_absolute,
            clean_max_error_sigma=args.clean_max_error_sigma,
        )

    _write_plot_log(
        args,
        gp_kwargs=gp_kwargs,
        total_plots=len(csv_paths),
    )


if __name__ == "__main__":
    main()
