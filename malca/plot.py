                      
"""
Plot light curves with Bayesian event detection results, showing run fits overlaid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from datetime import datetime
from typing import Sequence

from malca.events import run_bayesian_significance
from malca.utils import gaussian, paczynski_kernel
from malca.utils import clean_lc, read_lc_dat2, filter_bad_cameras
from malca.baseline import (
    per_camera_gp_baseline,
    global_mean_baseline,
    global_median_baseline,
    global_rolling_median_baseline,
    global_rolling_mean_baseline,
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
    per_camera_gp_baseline_masked,
)


JD_OFFSET = 2458000.0

asassn_columns = [
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera#",
    "v_g_band",
    "saturated",
    "cam_field",
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
    import pandas as pd

    df = pd.read_csv(
        dat_path,
        sep=r"\s+",
        names=asassn_columns,
        dtype={
            "JD": float,
            "mag": float,
            "error": float,
            "good_bad": int,
            "camera#": int,
            "v_g_band": int,
            "saturated": int,
            "cam_field": str,
        },
        comment="#",
    )
    return df


def read_skypatrol_csv(csv_path):
    """
    Read a SkyPatrol CSV, remapping columns to the ASAS-SN schema.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        comment="#",
        skip_blank_lines=True,
        dtype={
            "JD": float,
            "Flux": float,
            "Flux Error": float,
            "Mag": float,
            "Mag Error": float,
            "Limit": float,
            "FWHM": float,
            "Filter": "string",
            "Quality": "string",
            "Camera": "string",
        },
    )
    rename_map = {
        "Flux": "flux",
        "Flux Error": "flux_error",
        "Mag": "mag",
        "Mag Error": "error",
        "Limit": "limit",
        "FWHM": "fwhm",
        "Filter": "filter_band",
        "Quality": "quality_flag",
        "Camera": "camera",
    }
    df = df.rename(columns=rename_map)
    df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["error"] = pd.to_numeric(df["error"], errors="coerce")
    df["flux"] = pd.to_numeric(df.get("flux"), errors="coerce")
    df["flux_error"] = pd.to_numeric(df.get("flux_error"), errors="coerce")
    df["camera"] = df["camera"].astype(str).str.strip()
    df["camera#"] = df["camera"]
    df["cam_field"] = df["camera#"]
    df["quality_flag"] = df["quality_flag"].astype(str).str.strip().str.upper()
    df["good_bad"] = (df["quality_flag"] == "G").astype(int)
    df["saturated"] = 0

    filt = df["filter_band"].astype(str).str.strip().str.lower()
    band_map = {"v": 1, "g": 0}
    df["v_g_band"] = filt.map(band_map)
    df = df[df["v_g_band"].notna()].copy()
    df["v_g_band"] = df["v_g_band"].astype(int)

    df = df[pd.notna(df["JD"]) & pd.notna(df["mag"])]
    df = df.sort_values("JD").reset_index(drop=True)
    return df


def load_lightcurve_df(
    path,
    *,
    filter_bad_cameras_enabled: bool = False,
    bad_camera_scatter_ratio: float = 2.5,
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
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dat2":
        dfg, dfv = read_lc_dat2(path.stem, str(path.parent))
        if dfg.empty and dfv.empty:
            return (pd.DataFrame(), set()) if return_filtered_info else pd.DataFrame()
        df = pd.concat([dfg, dfv], ignore_index=True)
    elif suffix == ".csv":
        df = read_skypatrol_csv(path)
    elif suffix == ".dat":
        df = read_asassn_dat(path)
    else:
        df = read_asassn_dat(path)
    
    # Optionally filter bad cameras
    filtered_cameras: set[int] = set()
    if filter_bad_cameras_enabled and not df.empty and "camera#" in df.columns:
        df, filtered_cameras = filter_bad_cameras(df, lc_path=str(path), scatter_ratio_threshold=bad_camera_scatter_ratio)
    
    if return_filtered_info:
        return df, filtered_cameras
    return df


def load_events_paths(
    events_path: Path,
    *,
    path_col: str = "path",
    only_significant: bool = False,
    max_plots: int | None = None,
) -> list[Path]:
    """
    Load events/post-filter output and return unique LC paths.
    Supports CSV/Parquet.
    """
    events_path = Path(events_path)
    suffix = events_path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(events_path)
    else:
        df = pd.read_csv(events_path)

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


def load_detection_results(csv_path):
    """
    Load detection_results.csv with trimmed strings; used for metadata lookup.
    """
    if csv_path is None:
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"detection_results file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df["Match_ID"] = df["DAT_Path"].apply(lambda p: Path(p).stem if p else "")
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
    matches = pd.DataFrame()

    if asassn_id:
        matches = df[df["Match_ID"] == str(asassn_id).strip()]
    if matches.empty and dat_path:
        matches = df[df["DAT_Path"] == str(dat_path).strip()]
    if matches.empty and source_name:
        matches = df[df["Source"].str.lower() == str(source_name).strip().lower()]

    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "dat_path": row.get("DAT_Path"),
        "source": row.get("Source"),
        "source_id": row.get("Match_ID"),
        "category": row.get("Category"),
        "vsx_class": row.get("VSX_Class"),
    }


def lookup_metadata_for_path(path: Path, detection_results_csv=None):
    """
    Infer metadata for a given light-curve path.
    Falls back to brayden_candidates if no detection_results CSV is provided.
    """
    path = Path(path)
    stem = path.stem
    source_type = "SkyPatrol" if path.suffix.lower() == ".csv" else "Internal"

    meta = lookup_source_metadata(asassn_id=stem, dat_path=str(path), csv_path=detection_results_csv)

    if not meta and "-light-curves" in stem:
        meta = lookup_source_metadata(asassn_id=stem.replace("-light-curves", ""), csv_path=detection_results_csv)

    if not meta and "-" in stem:
        meta = lookup_source_metadata(asassn_id=stem.split("-")[0], csv_path=detection_results_csv)

    # Fallback to brayden_candidates if no metadata found from CSV
    if not meta:
        try:
            from tests.reproduce import brayden_candidates
            
            # Extract source_id from filename
            source_id = stem.replace("-light-curves", "").split("-")[0]
            
            # Look up in brayden_candidates
            for candidate in brayden_candidates:
                if candidate.get("source_id") == source_id:
                    meta = {
                        "source": candidate.get("source"),
                        "source_id": source_id,
                        "category": candidate.get("category"),
                        "data_source": source_type,
                    }
                    return meta
        except ImportError:
            pass
    
    if not meta:
        return {"data_source": source_type}

    meta = dict(meta)
    meta["data_source"] = source_type
    return meta



BASELINE_FUNCTIONS = {
    "global_mean": global_mean_baseline,
    "global_median": global_median_baseline,
    "global_rolling_median": global_rolling_median_baseline,
    "global_rolling_mean": global_rolling_mean_baseline,
    "per_camera_mean": per_camera_mean_baseline,
    "per_camera_median": per_camera_median_baseline,
    "per_camera_trend": per_camera_trend_baseline,
    "per_camera_gp": per_camera_gp_baseline,
    "per_camera_gp_masked": per_camera_gp_baseline_masked,
}

PER_CAMERA_BASELINES = {
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
}

PER_CAMERA_BASELINE_NAMES = {
    "per_camera_mean",
    "per_camera_median",
    "per_camera_trend",
    "per_camera_gp",
    "per_camera_gp_masked",
}


def plot_bayes_results(
    csv_path: Path,
    results_csv: Path | None = None,
    *,
    out_path: Path | None = None,
    figsize=(14, 8),
    show=False,
    baseline_func=None,
    baseline_kwargs=None,
    logbf_threshold_dip=5.0,
    logbf_threshold_jump=5.0,
    skip_events=False,
    plot_fits=False,
    jd_offset=2458000.0,
    detection_results_csv=None,
    clean_max_error_absolute=1.0,
    clean_max_error_sigma=5.0,
    annotations: dict[str, str] | None = None,
    # New parameters
    metadata: dict | None = None,
    run_params: dict | None = None,
    unified_layout: bool = True,
    legend_outside: bool = True,
    robust_threshold: bool = True,
    show_timestamp: bool = True,
    filter_bad_cameras: bool = True,
    bad_camera_scatter_ratio: float = 2.5,
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
        raise ValueError(f"Light curve file is empty: {csv_path.name}")


    asas_sn_id = csv_path.stem.split("-")[0]
    
                                                                   
    baseline_name = None
    if baseline_func is None:
        baseline_func = per_camera_gp_baseline
    if baseline_kwargs is None:
        baseline_kwargs = {}
    # allow alias strings for baseline selection
    if isinstance(baseline_func, str):
        baseline_name = baseline_func
        baseline_func = BASELINE_FUNCTIONS.get(baseline_func, per_camera_gp_baseline)
    if baseline_name is None:
        for name, func in BASELINE_FUNCTIONS.items():
            if func is baseline_func:
                baseline_name = name
                break
    per_camera_baseline = (
        baseline_func in PER_CAMERA_BASELINES
        or (baseline_name in PER_CAMERA_BASELINE_NAMES)
    )
    
    print(f"Analyzing {asas_sn_id}...")
    
                                             
    df_g = df[df["v_g_band"] == 0].copy()
    df_v = df[df["v_g_band"] == 1].copy()
    
    if skip_events:
        empty_res = {"significant": False, "run_summaries": [], "n_runs": 0}
        band_results = {0: {"dip": empty_res, "jump": empty_res}, 1: {"dip": empty_res, "jump": empty_res}}
    else:
        # For GP baselines, ensure add_sigma_eff_col is enabled for sigma_eff computation
        if baseline_func in (per_camera_gp_baseline, per_camera_gp_baseline_masked):
            baseline_kwargs.setdefault("add_sigma_eff_col", True)
        
        res_g = run_bayesian_significance(
            df_g,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            logbf_threshold_dip=logbf_threshold_dip,
            logbf_threshold_jump=logbf_threshold_jump,
            compute_event_prob=True,
        ) if not df_g.empty else {"dip": {"significant": False, "run_summaries": [], "n_runs": 0}, "jump": {"significant": False, "run_summaries": [], "n_runs": 0}}
        
        res_v = run_bayesian_significance(
            df_v,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            logbf_threshold_dip=logbf_threshold_dip,
            logbf_threshold_jump=logbf_threshold_jump,
            compute_event_prob=True,
        ) if not df_v.empty else {"dip": {"significant": False, "run_summaries": [], "n_runs": 0}, "jump": {"significant": False, "run_summaries": [], "n_runs": 0}}

        
        band_results = {0: res_g, 1: res_v}
    
                               
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
    cmap = plt.get_cmap("tab20", max(len(camera_ids), 1))
    camera_colors = {cam: cmap(i % cmap.N) for i, cam in enumerate(camera_ids)}

    # Compute baselines and residuals for each band
    band_dfs = {}
    all_resids = []
    for band in bands:
        band_df = df[df["v_g_band"] == band].copy()
        if band_df.empty:
            continue
        if baseline_func:
            band_df_baseline = baseline_func(band_df, **baseline_kwargs)
            if "baseline" in band_df_baseline.columns:
                band_df["baseline"] = band_df_baseline["baseline"]
                band_df["resid"] = band_df["mag"] - band_df["baseline"]
                all_resids.extend(band_df["resid"].dropna().tolist())
        band_dfs[band] = band_df

    # Compute robust 3-sigma threshold from all residuals
    if robust_threshold and len(all_resids) > 10:
        all_resids_arr = np.array(all_resids)
        all_resids_finite = all_resids_arr[np.isfinite(all_resids_arr)]
        if len(all_resids_finite) > 10:
            mad = 1.4826 * np.median(np.abs(all_resids_finite - np.median(all_resids_finite)))
            sigma_5 = 5 * mad
        else:
            sigma_5 = 0.5  # fallback ~5-sigma for typical scatter
    else:
        sigma_5 = 0.5

    # Create unified 2x1 layout (both bands on same axes)
    if unified_layout:
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
        ax_main = axes[0]
        ax_resid = axes[1]
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex="col")

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
                ax_main.axvline(jd_plot_start, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
                if jd_plot_end != jd_plot_start:
                    ax_main.axvline(jd_plot_end, color="red", linestyle="--", alpha=0.7, linewidth=1.5)

                morph = run_summary.get("morphology", "none")
                params = run_summary.get("params", {})

                if morph == "gaussian" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color="red", linestyle="--", alpha=0.7, linewidth=1.0)
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
                            color="red",
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

                elif morph == "paczynski" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color="red", linestyle="--", alpha=0.7, linewidth=1.0)
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
                            color="blue",
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
                ax_main.axvline(jd_plot_start, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
                if jd_plot_end != jd_plot_start:
                    ax_main.axvline(jd_plot_end, color="green", linestyle="--", alpha=0.7, linewidth=1.5)

                morph = run_summary.get("morphology", "none")
                params = run_summary.get("params", {})

                if morph == "gaussian" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color="green", linestyle="--", alpha=0.7, linewidth=1.0)
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
                            color="green",
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

                elif morph == "paczynski" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color="green", linestyle="--", alpha=0.7, linewidth=1.0)
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
                            color="cyan",
                            linestyle="-",
                            linewidth=2,
                            alpha=0.8,
                        )

                elif morph == "fred" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    if t0 is not None and np.isfinite(t0):
                        t0_plot = t0 - (jd_offset if median_jd > 2000000 else 8000.0)
                        ax_main.axvline(t0_plot, color="green", linestyle="--", alpha=0.7, linewidth=1.0)
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
                            color="magenta",
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
            ax_main.set_xlabel(f"JD - {int(jd_offset)}", fontsize=10)
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
            ax_resid.set_xlabel(f"JD - {int(jd_offset)}", fontsize=10)

    # Configure unified axes (after plotting all bands)
    if unified_layout:
        ax_main.invert_yaxis()
        ax_main.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax_main.set_xlabel(f"JD - {int(jd_offset)}", fontsize=10)
        ax_main.xaxis.set_label_position("top")
        ax_main.set_ylabel("Magnitude [mag]", fontsize=12)
        ax_main.grid(True, alpha=0.3)

        # Build legend with Line2D handles for events
        handles, labels = ax_main.get_legend_handles_labels()
        legend_handles = list(zip(handles, labels))

        # Add event color legend entries
        if has_dip:
            legend_handles.append((Line2D([0], [0], color="red", linestyle="--", linewidth=1.5), "Dip"))
        if has_jump:
            legend_handles.append((Line2D([0], [0], color="green", linestyle="--", linewidth=1.5), "Jump"))

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

        ax_resid.set_xlabel(f"JD - {int(jd_offset)}", fontsize=10)
    


    # Merge metadata from lookup with passed-in metadata (passed-in takes precedence)
    meta = lookup_metadata_for_path(csv_path, detection_results_csv=detection_results_csv) or {}
    if metadata:
        meta.update(metadata)

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
        g_dip = band_results[0]["dip"]
        g_jump = band_results[0]["jump"]
        v_dip = band_results[1]["dip"]
        v_jump = band_results[1]["jump"]

        if g_dip["significant"] or v_dip["significant"]:
            total_dips = g_dip.get("n_runs", 0) + v_dip.get("n_runs", 0)
            title_parts.append(f"Dips: {total_dips} runs (g:{g_dip.get('n_runs', 0)}, V:{v_dip.get('n_runs', 0)})")
        if g_jump["significant"] or v_jump["significant"]:
            total_jumps = g_jump.get("n_runs", 0) + v_jump.get("n_runs", 0)
            title_parts.append(f"Jumps: {total_jumps} runs (g:{g_jump.get('n_runs', 0)}, V:{v_jump.get('n_runs', 0)})")

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
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    if return_filtered_cameras:
        return filtered_cameras
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves with Bayesian event detection results"
    )
    parser.add_argument(
        "--detect-run",
        type=Path,
        default=None,
        help="Detect run directory (e.g., output/runs/20250121_143052). If specified, reads events from <detect-run>/results/ and writes plots to <detect-run>/plots/",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="Path(s) to light curve file(s) (glob patterns supported)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="Events/post-filter output (CSV/Parquet) with a path column (overrides --detect-run).",
    )
    parser.add_argument(
        "--path-col",
        default="path",
        help="Column in events/post-filter output that contains LC paths.",
    )
    parser.add_argument(
        "--only-significant",
        action="store_true",
        help="If events output has dip_significant/jump_significant, plot only those.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Maximum number of light curves to plot.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Path to results CSV (optional, for filtering which to plot)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to <detect-run>/plots/ if --detect-run is used)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINE_FUNCTIONS.keys()),
        default="per_camera_gp",
        help="Baseline function to use",
    )
    parser.add_argument(
        "--logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Log BF threshold for dips",
    )
    parser.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Log BF threshold for jumps",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip Bayesian event detection; plot baseline/residuals only",
    )
    parser.add_argument(
        "--plot-fits",
        action="store_true",
        help="Plot Gaussian/Paczynski fit curves in addition to peak markers.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output format for plots (default: png).",
    )
    parser.add_argument(
        "--jd-offset",
        type=float,
        default=2458000.0,
        help="JD offset for plotting (default: 2458000.0)",
    )
    parser.add_argument(
        "--clean-max-error-absolute",
        type=float,
        default=1.0,
        help="Absolute error cutoff for clean_lc (default: 1.0)",
    )
    parser.add_argument(
        "--clean-max-error-sigma",
        type=float,
        default=5.0,
        help="Sigma cutoff for clean_lc MAD filter (default: 5.0)",
    )
    parser.add_argument("--gp-sigma", type=float, default=None, help="GP sigma parameter.")
    parser.add_argument("--gp-rho", type=float, default=None, help="GP rho parameter.")
    parser.add_argument("--gp-q", type=float, default=None, help="GP Q parameter (default: 0.7).")
    parser.add_argument("--gp-s0", type=float, default=None, help="GP S0 parameter (alt parameterization).")
    parser.add_argument("--gp-w0", type=float, default=None, help="GP w0 parameter (alt parameterization).")
    parser.add_argument("--gp-jitter", type=float, default=None, help="GP jitter term (default: 0.006).")
    parser.add_argument("--gp-sigma-floor", type=float, default=None, help="Extra GP sigma floor.")
    parser.add_argument("--gp-floor-clip", type=float, default=None, help="Sigma floor clipping threshold.")
    parser.add_argument("--gp-floor-iters", type=int, default=None, help="Sigma floor clipping iterations.")
    parser.add_argument("--gp-min-floor-points", type=int, default=None, help="Minimum points for sigma floor.")
    parser.add_argument(
        "--detection-results",
        type=Path,
        default=None,
        help="Optional detection results CSV for metadata lookup",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    # Handle --detect-run for events and output directory
    if args.detect_run:
        detect_run = args.detect_run.expanduser()

        # Set events path if not explicitly provided
        if not args.events:
            results_dir = detect_run / "results"
            # Look for filtered results first, then raw results
            candidates = (list(results_dir.glob("*filtered.csv")) +
                         list(results_dir.glob("*filtered.parquet")) +
                         list(results_dir.glob("*events_results.csv")) +
                         list(results_dir.glob("*events_results.parquet")))
            if candidates:
                args.events = candidates[0]
                print(f"Using events from: {args.events}")

        # Set out_dir if not explicitly provided
        if not args.out_dir:
            args.out_dir = detect_run / "plots"

    # Validate that we have an output directory
    if not args.out_dir:
        raise ValueError("Must specify either --out-dir or --detect-run")

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
        if baseline_func in (per_camera_gp_baseline, per_camera_gp_baseline_masked):
            baseline_kwargs.update(gp_kwargs)
        else:
            print("Warning: GP parameters were provided but baseline is not a GP baseline; ignoring.", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)


    if args.events:
        csv_paths = load_events_paths(
            args.events,
            path_col=args.path_col,
            only_significant=args.only_significant,
            max_plots=args.max_plots,
        )
        print(f"Loaded {len(csv_paths)} light curves from {args.events}")
    elif args.input:
        csv_paths = [Path(p) for p in args.input]
    else:
        csv_paths = []

    if args.results_csv and args.results_csv.exists():
        results_df = pd.read_csv(args.results_csv)

        results_ids = set()
        for path in results_df["path"]:
            if "-light-curves.csv" in str(path):
                id_str = str(path).split("/")[-1].replace("-light-curves.csv", "")
                results_ids.add(id_str)

        csv_paths = [p for p in csv_paths if p.stem.split("-")[0] in results_ids]
        print(f"Filtered to {len(csv_paths)} light curves from results CSV")

    if not csv_paths:
        raise SystemExit("No light curve paths provided (use --input or --events).")

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
            detection_results_csv=args.detection_results,
            clean_max_error_absolute=args.clean_max_error_absolute,
            clean_max_error_sigma=args.clean_max_error_sigma,
        )

    # Generate plot log with comprehensive statistics
    if args.detect_run:
        try:
            import json
            import sys
            import shlex
            from datetime import datetime

            detect_run = args.detect_run.expanduser()
            plot_log_file = detect_run / "plot_log.json"

            orig_argv = getattr(sys, "orig_argv", None)
            cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)

            plot_log = {
                "timestamp": datetime.now().isoformat(),
                "command": cmd,
                "events_file": str(args.events) if args.events else None,
                "output_dir": str(args.out_dir),
                "plot_params": {
                    "baseline": args.baseline,
                    "logbf_threshold_dip": args.logbf_threshold_dip,
                    "logbf_threshold_jump": args.logbf_threshold_jump,
                    "skip_events": args.skip_events,
                    "plot_fits": args.plot_fits,
                    "format": args.format,
                    "only_significant": args.only_significant,
                    "jd_offset": args.jd_offset,
                    "clean_max_error_absolute": args.clean_max_error_absolute,
                    "clean_max_error_sigma": args.clean_max_error_sigma,
                },
                "results": {
                    "total_plots": len(csv_paths),
                    "max_plots_limit": args.max_plots,
                },
            }

            # Add GP parameters if used
            if gp_kwargs:
                plot_log["plot_params"]["gp_params"] = gp_kwargs

            with open(plot_log_file, "w") as f:
                json.dump(plot_log, f, indent=2, default=str)

        except Exception as e:
            if args.verbose:
                print(f"Warning: could not write plot log: {e}")


if __name__ == "__main__":
    main()
