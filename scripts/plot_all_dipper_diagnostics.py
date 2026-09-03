#!/usr/bin/env python
"""Generate publication-style diagnostic plots for every reviewed dipper.

The sample is selected only by ``reviews.event_class = 'dipper'``.  In
particular, this script never filters on ``final_class`` or ``yso_class``.

The figures are analogues of the distance, 2MASS/WISE, H-alpha, WISE CMD, and
timescale-depth plots in Tzanidakis et al. (2025).  Bailer-Jones distances are
fetched explicitly and cached; other layers that are not stored in MALCA
(literature comparison samples, stellar loci, and BT-NextGen disk tracks) are
not silently fabricated.
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
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext

from malca.config import (
    AU_M,
    DAY_S,
    GRAVITATIONAL_CONSTANT_SI,
    SOLAR_MASS_KG,
    SOLAR_RADIUS_M,
    UNWISE_EXPECTED_SCATTER_BASE,
    UNWISE_EXPECTED_SCATTER_MAG_REF,
    UNWISE_EXPECTED_SCATTER_SLOPE,
    YSO_CLASS_I_W1W2,
    YSO_CLASS_II_HK,
    YSO_CLASS_II_W1W2_MIN,
)
from malca.core.baseline import per_camera_gp_baseline_masked
from malca.core.utils import clean_lc
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame
from malca.plotting.blackbody_locus import add_blackbody_locus
from malca.plotting.color_color_labels import (
    LABEL_H_KS,
    LABEL_R_HALPHA,
    LABEL_R_I,
    LABEL_W1_W2,
    LABEL_W3,
    LABEL_W3_W4,
)
from malca.plotting.lightcurve_publication import PUBLICATION_STYLE
from malca.stv.dimming_window import (
    DEFAULT_DIMMING_WINDOW_CONFIG,
    DIMMING_WINDOW_METHOD_VERSION,
    dimming_complex_zoom_bounds,
    measure_dimming_complex_window,
)


DEFAULT_RUN_ROOT = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_REVIEW_DB = DEFAULT_RUN_ROOT / "review" / "review.db"
DEFAULT_EXTERNAL_MANIFEST = DEFAULT_RUN_ROOT / "results" / "external_lc_manifest.parquet"
DEFAULT_ALLWISE_QUALITY = (
    DEFAULT_RUN_ROOT / "results" / "marked_dipper_seds" / "marked_dipper_allwise_quality.parquet"
)
DEFAULT_OUTPUT_DIR = Path("output/pdf/all_dipper_diagnostics")
DEFAULT_BAILER_JONES_FILENAME = "all_dippers_bailer_jones_distances.csv"
DEFAULT_TRIGGERED_DIP_RUNS = Path(
    "output/notebooks/july1_triggered_dip_systematics/triggered_dip_runs.parquet"
)
DEFAULT_EVENT_MC_DRAWS = 1024
MIN_DURATION_MC_RESOLVED_FRACTION = 0.90
MIN_DURATION_MC_RESOLVED_DRAWS = 200
FWHM_METHOD_VERSION = "persistent_half_depth_5of6_7d_v1"
EVENT_METRICS_SCHEMA_VERSION = "dimming_complex_duration_v1"
HALF_DEPTH_RECOVERY_WINDOW_EPOCHS = 6
HALF_DEPTH_RECOVERY_REQUIRED_EPOCHS = 5
HALF_DEPTH_RECOVERY_MIN_SPAN_DAYS = 7.0
PIPELINE_DIP_RUN_COLOR = "#cc79a7"
PIPELINE_DIP_RUN_EDGE_COLOR = "#8e2c6a"
PIPELINE_DIP_RUN_COLUMNS = (
    "event_id",
    "candidate_id",
    "run_number",
    "run_start_jd",
    "run_end_jd",
    "dip_jd",
    "n_trigger_points",
    "run_peak_event_probability",
    "trigger_jds_json",
    "detector_commit",
    "run_table_schema_version",
)

CLASS_ORDER = ("Class I", "Class II", "Flat", "Class III/photosphere", "Unknown")
CLASS_STYLE = {
    "Class I": {"marker": "^", "color": "#c0392b", "label": "Class I"},
    "Class II": {"marker": "s", "color": "#5f8f79", "label": "Class II"},
    "Flat": {"marker": "v", "color": "#d9a441", "label": "Flat"},
    "Class III/photosphere": {
        "marker": "o",
        "color": "#c05a84",
        "label": "Class III/photosphere",
    },
    "Unknown": {"marker": "D", "color": "#777777", "label": "Unknown"},
}
TIMESCALE_CLASS_OUTLINE = {
    "Class I": "#ff6f3c",
    "Class II": "#5ec7ff",
    "Flat": CLASS_STYLE["Flat"]["color"],
    "Class III/photosphere": CLASS_STYLE["Class III/photosphere"]["color"],
    "Unknown": "#bbbbbb",
}
DURATION_DEPTH_XMIN_DAYS = 1.0  # 10^0 days
DURATION_DEPTH_XMAX_DAYS = 3_000.0  # 3 × 10^3 days

plt.rcParams.update(
    {
        **PUBLICATION_STYLE,
        "axes.linewidth": 1.15,
        "savefig.facecolor": "white",
    }
)


def _finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _dimming_complex_metrics(
    start_jd: float,
    end_jd: float,
    event_status: str,
    is_lower_limit: bool,
) -> dict[str, float | bool | str]:
    """Describe the full recovery-anchored dimming complex.

    This is intentionally separate from the peak-centered FWHM.  The measured
    span is the outer event envelope already selected by the recovery logic.
    A completed envelope is recovery-bounded; any missing outer recovery makes
    the observed span a lower limit rather than an exact physical timescale.
    """
    start = _finite_number(start_jd)
    end = _finite_number(end_jd)
    span = max(0.0, end - start) if np.isfinite(start) and np.isfinite(end) else np.nan
    event_status = str(event_status)
    lower_limit = bool(is_lower_limit)

    left_open_statuses = {
        "ongoing_left_censored",
        "left_recovery_unconfirmed",
        "left_gap_censored",
        "ongoing_left_right_gap_censored",
        "both_gap_censored",
    }
    right_open_statuses = {
        "ongoing_right_censored",
        "right_recovery_unconfirmed",
        "right_gap_censored",
        "ongoing_right_left_gap_censored",
        "both_gap_censored",
    }
    left_open = event_status in left_open_statuses
    right_open = event_status in right_open_statuses
    if not lower_limit:
        status = "recovery_bounded"
    elif left_open and right_open:
        status = "both_censored"
    elif left_open:
        status = "left_censored"
    elif right_open:
        status = "right_censored"
    else:
        status = "censored"

    return {
        "dimming_complex_start_jd": start,
        "dimming_complex_end_jd": end,
        "dimming_complex_duration_days": span,
        "dimming_complex_duration_lower_days": span,
        "dimming_complex_duration_upper_days": span if not lower_limit else np.nan,
        "dimming_complex_duration_plot_days": span,
        "dimming_complex_is_lower_limit": lower_limit,
        "dimming_complex_status": status,
    }


def _payload_value(payload_text: Any, key: str) -> float:
    if payload_text is None:
        return np.nan
    try:
        payload = json.loads(str(payload_text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.nan
    value = payload.get(key)
    if value is None and isinstance(payload.get("external"), dict):
        value = payload["external"].get(key)
    return _finite_number(value)


def _display_sed_alpha_class(sed_alpha_class: Any) -> str:
    """Map stored ``sed_alpha_class`` values to the plot legend buckets."""
    value = str(sed_alpha_class or "").strip()
    if not value or value.lower() == "unknown":
        return "Unknown"
    normalized = {
        "class i": "Class I",
        "class ii": "Class II",
        "flat": "Flat",
        "class iii/photosphere": "Class III/photosphere",
    }
    return normalized.get(value.lower(), "Unknown")


def read_all_dippers(review_db: Path) -> pd.DataFrame:
    """Read the complete live dipper sample without a stellar-class cut."""
    query = """
        SELECT
            c.candidate_id,
            c.asas_sn_id,
            c.lc_path,
            c.source_id,
            c.gaia_id,
            c.ra,
            c.dec,
            c.final_class,
            c.yso_class,
            c.sed_alpha_class,
            c.parallax,
            c.parallax_error,
            c.distance_gspphot,
            c.tmass_h,
            c.tmass_h_err,
            c.tmass_k,
            c.tmass_k_err,
            c.w1,
            c.w1_err,
            c.w2,
            c.w2_err,
            c.w3,
            c.w3_err,
            c.w4,
            c.w4_err,
            c.iphas_r_i,
            c.iphas_r_i_err,
            c.iphas_r_ha,
            c.iphas_r_ha_err,
            c.vphas_r_i,
            c.vphas_r_i_err,
            c.vphas_r_ha,
            c.vphas_r_ha_err,
            c.dip_best_mag_event,
            c.dip_symmetry_score,
            c.dip_max_run_duration,
            c.mass50,
            c.payload_json,
            r.event_class,
            r.morphology_primary,
            r.physical_primary,
            r.workflow_status,
            r.status
        FROM candidates AS c
        JOIN reviews AS r ON r.candidate_id = c.candidate_id
        WHERE lower(trim(coalesce(r.event_class, ''))) = 'dipper'
        ORDER BY c.candidate_id
    """
    uri = f"file:{review_db.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        frame = pd.read_sql_query(query, conn)

    if frame.empty:
        raise RuntimeError(f"No reviewed dippers were found in {review_db}")
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("The live dipper query returned duplicate candidate IDs")

    numeric = [
        col
        for col in frame.columns
        if col
        not in {
            "candidate_id",
            "asas_sn_id",
            "lc_path",
            "source_id",
            "gaia_id",
            "final_class",
            "yso_class",
            "sed_alpha_class",
            "payload_json",
            "event_class",
            "morphology_primary",
            "physical_primary",
            "workflow_status",
            "status",
        }
    ]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    for key in ("dist16", "dist50", "dist84", "mass16", "mass84"):
        frame[f"starhorse_{key}"] = frame["payload_json"].map(lambda value, k=key: _payload_value(value, k))
    frame["starhorse_distance_pc"] = 1000.0 * frame["starhorse_dist50"]
    frame["inverse_parallax_distance_pc"] = np.where(
        frame["parallax"] > 0.0,
        1000.0 / frame["parallax"],
        np.nan,
    )
    frame["plot_class"] = frame["sed_alpha_class"].map(_display_sed_alpha_class)
    return frame


def read_pipeline_dip_runs(
    replay_path: Path,
    candidate_ids: pd.Series | np.ndarray | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Read provenance-locked pipeline dip runs for the requested candidates."""
    resolved = Path(replay_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pipeline dip-run replay not found: {resolved}")
    runs = pd.read_parquet(resolved, columns=list(PIPELINE_DIP_RUN_COLUMNS))
    missing = set(PIPELINE_DIP_RUN_COLUMNS) - set(runs.columns)
    if missing:
        raise RuntimeError(
            f"Pipeline dip-run replay is missing columns: {sorted(missing)}"
        )
    requested = {str(value) for value in candidate_ids}
    runs["candidate_id"] = runs["candidate_id"].astype(str)
    runs = runs.loc[runs["candidate_id"].isin(requested)].copy()
    for column in (
        "run_number",
        "run_start_jd",
        "run_end_jd",
        "dip_jd",
        "n_trigger_points",
        "run_peak_event_probability",
        "run_table_schema_version",
    ):
        runs[column] = pd.to_numeric(runs[column], errors="coerce")
    invalid = (
        ~np.isfinite(runs["run_start_jd"])
        | ~np.isfinite(runs["run_end_jd"])
        | (runs["run_end_jd"] < runs["run_start_jd"])
    )
    if bool(invalid.any()):
        examples = runs.loc[
            invalid, ["event_id", "candidate_id", "run_start_jd", "run_end_jd"]
        ].head(10)
        raise RuntimeError(
            "Pipeline dip-run replay contains invalid selected intervals: "
            f"{examples.to_dict(orient='records')}"
        )
    if runs["event_id"].astype(str).duplicated().any():
        raise RuntimeError("Pipeline dip-run replay contains duplicate selected event IDs")
    runs.attrs["source_path"] = str(resolved)
    return runs.sort_values(
        ["candidate_id", "run_start_jd", "run_end_jd", "run_number"],
        kind="mergesort",
        ignore_index=True,
    )


def _json_float_list(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    output: list[float] = []
    for item in parsed:
        number = _finite_number(item)
        if np.isfinite(number):
            output.append(float(number))
    return output


def _pipeline_trigger_jds(runs: pd.DataFrame | None) -> np.ndarray:
    if runs is None or runs.empty or "trigger_jds_json" not in runs:
        return np.array([], dtype=float)
    values = [
        jd
        for blob in runs["trigger_jds_json"]
        for jd in _json_float_list(blob)
    ]
    if not values:
        return np.array([], dtype=float)
    return np.unique(np.asarray(values, dtype=float))


def _pipeline_run_overlay_metrics(
    runs: pd.DataFrame | None,
    *,
    event_start_jd: float,
    event_end_jd: float,
    peak_jd: float,
) -> dict[str, Any]:
    if runs is None or runs.empty:
        return {
            "pipeline_dip_run_count": 0,
            "pipeline_dip_runs_overlapping_complex": 0,
            "pipeline_trigger_point_count": 0,
            "atlas_peak_inside_pipeline_dip_run": False,
        }
    starts = pd.to_numeric(runs["run_start_jd"], errors="coerce").to_numpy(float)
    ends = pd.to_numeric(runs["run_end_jd"], errors="coerce").to_numpy(float)
    overlaps = (starts <= float(event_end_jd)) & (ends >= float(event_start_jd))
    contains_peak = (starts <= float(peak_jd)) & (ends >= float(peak_jd))
    return {
        "pipeline_dip_run_count": int(len(runs)),
        "pipeline_dip_runs_overlapping_complex": int(np.sum(overlaps)),
        "pipeline_trigger_point_count": int(len(_pipeline_trigger_jds(runs))),
        "atlas_peak_inside_pipeline_dip_run": bool(np.any(contains_peak)),
    }


def _nearest_observation_indices(
    observation_jds: np.ndarray,
    target_jds: np.ndarray,
    *,
    tolerance_days: float = 1.0e-4,
) -> np.ndarray:
    observations = np.asarray(observation_jds, dtype=float)
    targets = np.asarray(target_jds, dtype=float)
    if observations.size == 0 or targets.size == 0:
        return np.array([], dtype=int)
    indices: list[int] = []
    for target in targets[np.isfinite(targets)]:
        index = int(np.nanargmin(np.abs(observations - target)))
        if abs(float(observations[index] - target)) <= float(tolerance_days):
            indices.append(index)
    return np.unique(np.asarray(indices, dtype=int)) if indices else np.array([], dtype=int)


def _draw_pipeline_dip_run_overlay(
    ax: plt.Axes,
    runs: pd.DataFrame | None,
    *,
    observation_jds: np.ndarray,
    observation_flux: np.ndarray,
    time_offset: float,
    observation_mask: np.ndarray | None = None,
) -> None:
    """Overlay historical pipeline-run windows and their triggered observations."""
    if runs is None or runs.empty:
        return
    for run in runs.itertuples(index=False):
        start = float(run.run_start_jd) - float(time_offset)
        end = float(run.run_end_jd) - float(time_offset)
        if end > start:
            ax.axvspan(
                start,
                end,
                color=PIPELINE_DIP_RUN_COLOR,
                alpha=0.10,
                linewidth=0,
                zorder=0.35,
            )
            ax.plot(
                [start, end],
                [0.985, 0.985],
                transform=ax.get_xaxis_transform(),
                color=PIPELINE_DIP_RUN_EDGE_COLOR,
                linewidth=2.2,
                solid_capstyle="butt",
                clip_on=True,
                zorder=7,
            )
        else:
            ax.axvline(
                start,
                color=PIPELINE_DIP_RUN_EDGE_COLOR,
                linewidth=1.0,
                alpha=0.9,
                zorder=7,
            )
    trigger_jds = _pipeline_trigger_jds(runs)
    trigger_indices = _nearest_observation_indices(observation_jds, trigger_jds)
    if observation_mask is not None and trigger_indices.size:
        visible = np.asarray(observation_mask, dtype=bool)
        trigger_indices = trigger_indices[visible[trigger_indices]]
    if trigger_indices.size:
        ax.scatter(
            np.asarray(observation_jds, dtype=float)[trigger_indices] - float(time_offset),
            np.asarray(observation_flux, dtype=float)[trigger_indices],
            s=28,
            marker="D",
            facecolors="none",
            edgecolors=PIPELINE_DIP_RUN_EDGE_COLOR,
            linewidths=0.9,
            zorder=6.5,
        )


def add_color_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["h_ks"] = out["tmass_h"] - out["tmass_k"]
    out["w1_w2_color"] = out["w1"] - out["w2"]
    out["w3_w4_color"] = out["w3"] - out["w4"]
    out["h_ks_err"] = np.sqrt(out["tmass_h_err"] ** 2 + out["tmass_k_err"] ** 2)
    out["w1_w2_err"] = np.sqrt(out["w1_err"] ** 2 + out["w2_err"] ** 2)
    out["w3_w4_err"] = np.sqrt(out["w3_err"] ** 2 + out["w4_err"] ** 2)
    out.loc[
        ~((out["tmass_h_err"] > 0) & (out["tmass_k_err"] > 0)),
        "h_ks_err",
    ] = np.nan
    out.loc[
        ~((out["w1_err"] > 0) & (out["w2_err"] > 0)),
        "w1_w2_err",
    ] = np.nan
    out.loc[
        ~((out["w3_err"] > 0) & (out["w4_err"] > 0)),
        "w3_w4_err",
    ] = np.nan
    out["fractional_depth_stored"] = 1.0 - np.power(
        10.0,
        -0.4 * pd.to_numeric(out["dip_best_mag_event"], errors="coerce"),
    )
    return out


def _allwise_intrinsic_scatter(path: Path) -> dict[str, Any]:
    """Measure W1 scatter from one homogeneous AllWISE multiepoch file."""
    result = {
        "allwise_mep_n_w1": 0,
        "allwise_mep_w1_median": np.nan,
        "allwise_mep_w1_robust_sigma": np.nan,
        "allwise_mep_w1_median_error": np.nan,
        "allwise_mep_w1_intrinsic_scatter": np.nan,
        "allwise_mep_w1_scatter_ratio": np.nan,
    }
    try:
        lc = pd.read_parquet(path)
    except Exception:
        return result
    if not {"w1mpro", "w1sigmpro"}.issubset(lc.columns):
        return result

    mag = pd.to_numeric(lc["w1mpro"], errors="coerce")
    err = pd.to_numeric(lc["w1sigmpro"], errors="coerce")
    good = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if "moon_masked" in lc.columns:
        moon = lc["moon_masked"].astype(str).str.strip()
        good &= moon.eq("") | moon.str.startswith("0")
    mag = mag.loc[good].to_numpy(float)
    err = err.loc[good].to_numpy(float)
    if mag.size < 3:
        return result

    median = float(np.median(mag))
    robust_sigma = float(1.4826 * np.median(np.abs(mag - median)))
    median_error = float(np.sqrt(np.median(np.square(err))))
    intrinsic = float(np.sqrt(max(robust_sigma**2 - median_error**2, 0.0)))
    expected = float(
        UNWISE_EXPECTED_SCATTER_BASE
        + UNWISE_EXPECTED_SCATTER_SLOPE * max(0.0, median - UNWISE_EXPECTED_SCATTER_MAG_REF)
    )
    result.update(
        {
            "allwise_mep_n_w1": int(mag.size),
            "allwise_mep_w1_median": median,
            "allwise_mep_w1_robust_sigma": robust_sigma,
            "allwise_mep_w1_median_error": median_error,
            "allwise_mep_w1_intrinsic_scatter": intrinsic,
            "allwise_mep_w1_scatter_ratio": robust_sigma / expected if expected > 0 else np.nan,
        }
    )
    return result


def measure_allwise_variability(
    candidates: pd.DataFrame,
    manifest_path: Path,
    run_root: Path,
) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "allwise_mep_path",
        "allwise_mep_n_w1",
        "allwise_mep_w1_median",
        "allwise_mep_w1_robust_sigma",
        "allwise_mep_w1_median_error",
        "allwise_mep_w1_intrinsic_scatter",
        "allwise_mep_w1_scatter_ratio",
    ]
    if not manifest_path.exists():
        return pd.DataFrame(columns=columns)

    manifest = pd.read_parquet(manifest_path)
    manifest["candidate_id"] = manifest["candidate_id"].astype(str)
    wanted = set(candidates["candidate_id"].astype(str))
    subset = manifest.loc[
        manifest["candidate_id"].isin(wanted)
        & manifest["source"].astype(str).str.lower().eq("allwise_mep")
    ].copy()
    if subset["candidate_id"].duplicated().any():
        subset = subset.sort_values("updated_unix").drop_duplicates("candidate_id", keep="last")

    rows: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        raw_path = Path(str(getattr(row, "path", ""))).expanduser()
        if not raw_path.exists():
            raw_path = run_root / "results" / str(getattr(row, "path_relative", ""))
        metrics = _allwise_intrinsic_scatter(raw_path)
        rows.append(
            {
                "candidate_id": str(row.candidate_id),
                "allwise_mep_path": str(raw_path),
                **metrics,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def read_allwise_quality(path: Path) -> pd.DataFrame:
    wanted = [
        "candidate_id",
        "allwise_id",
        "allwise_ph_qual",
        "allwise_cc_flags",
        "allwise_w3_snr",
        "allwise_w4_snr",
    ]
    if not path.exists():
        return pd.DataFrame(columns=wanted)
    quality = pd.read_parquet(path)
    quality["candidate_id"] = quality["candidate_id"].astype(str)
    if quality["candidate_id"].duplicated().any():
        raise RuntimeError(f"Duplicate candidate IDs in {path}")
    for col in wanted:
        if col not in quality.columns:
            quality[col] = np.nan
    return quality[wanted].copy()


def load_bailer_jones_distances(
    candidates: pd.DataFrame,
    cache_path: Path,
    *,
    fetch: bool,
) -> pd.DataFrame:
    """Read or fetch Bailer-Jones EDR3 distance posterior medians."""
    columns = ["candidate_id", "gaia_id", "bj_r_med_photogeo", "bj_r_med_geo"]
    if cache_path.exists() and not fetch:
        cached = pd.read_csv(cache_path, dtype={"candidate_id": str, "gaia_id": str})
        for col in columns:
            if col not in cached.columns:
                cached[col] = np.nan
        if cached["candidate_id"].duplicated().any():
            raise RuntimeError(f"Duplicate candidate IDs in {cache_path}")
        return cached[columns].copy()

    if not fetch:
        return pd.DataFrame(columns=columns)

    from malca.ltv.cmd import fetch_bailer_jones_distances

    query = candidates[["candidate_id", "gaia_id"]].copy()
    query["candidate_id"] = query["candidate_id"].astype(str)
    query["gaia_id"] = query["gaia_id"].astype("string").str.strip()
    valid = query["gaia_id"].str.fullmatch(r"\d+", na=False)
    fetched = fetch_bailer_jones_distances(
        query.loc[valid].copy(),
        source_id_col="gaia_id",
        chunk_size=1000,
        n_workers=1,
        verbose=True,
    )
    for col in ("bj_r_med_photogeo", "bj_r_med_geo"):
        fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
    output = query.merge(
        fetched[["candidate_id", "bj_r_med_photogeo", "bj_r_med_geo"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    n_matched = int((output["bj_r_med_photogeo"] > 0).sum())
    if n_matched == 0:
        raise RuntimeError("The Bailer-Jones TAP query returned no photogeometric distances")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output[columns].to_csv(cache_path, index=False)
    return output[columns]


def _cadence_gap_cap(times: np.ndarray) -> tuple[float, float]:
    dt = np.diff(np.asarray(times, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan, 5.0
    cadence = float(np.median(dt))
    cadence_mad = float(1.4826 * np.median(np.abs(dt - cadence)))
    gap_cap = float(min(30.0, max(5.0 * cadence, cadence + 6.0 * cadence_mad, 1.0)))
    return cadence, gap_cap


def _crossing_time(t0: float, y0: float, t1: float, y1: float, level: float) -> float:
    if y1 == y0:
        return 0.5 * (t0 + t1)
    fraction = float(np.clip((level - y0) / (y1 - y0), 0.0, 1.0))
    return float(t0 + fraction * (t1 - t0))


def _local_epoch_median(
    times: np.ndarray,
    values: np.ndarray,
    cadence: float,
) -> tuple[np.ndarray, float]:
    """Return a three-epoch median without smoothing across major gaps."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    cadence_scale = cadence if np.isfinite(cadence) and cadence > 0 else 1.0
    smoothing_gap_limit = float(min(30.0, max(10.0, 5.0 * cadence_scale)))
    smoothed = np.full(values.shape, np.nan, dtype=float)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > smoothing_gap_limit) + 1, len(times)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = np.arange(start, stop)
        for index in block:
            order = np.argsort(np.abs(times[block] - times[index]))
            local = block[order[: min(3, len(block))]]
            finite = values[local][np.isfinite(values[local])]
            if finite.size:
                smoothed[index] = float(np.median(finite))
    return smoothed, smoothing_gap_limit


def _mask_components(
    mask: np.ndarray,
    times: np.ndarray,
    max_gap_days: float,
) -> list[tuple[int, int]]:
    """Return true-mask components, splitting genuinely separate observing blocks."""
    components: list[tuple[int, int]] = []
    index = 0
    while index < len(mask):
        if not bool(mask[index]):
            index += 1
            continue
        start = index
        while (
            index + 1 < len(mask)
            and bool(mask[index + 1])
            and times[index + 1] - times[index] <= max_gap_days
        ):
            index += 1
        components.append((start, index))
        index += 1
    return components


def _stable_recovery_mask(
    corrected_residual: np.ndarray,
    sigma: np.ndarray,
    times: np.ndarray,
    quiet_scatter_mag: float,
    max_gap_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Confirm recovery plateaus from independent nightly epochs.

    A plateau needs a robust five-night center near the fixed GP-residual
    baseline and at least three individually compatible nights.  The robust
    window median lets minority outliers coexist with a majority-baseline
    recovery.  ``support`` records the full qualifying window;
    ``confirmed`` records only its individually baseline-compatible members.
    """
    corrected = np.asarray(corrected_residual, float)
    sigma = np.asarray(sigma, float)
    uncertainty = np.maximum.reduce(
        [
            np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0),
            np.full(corrected.shape, max(quiet_scatter_mag, 0.0)),
            np.full(corrected.shape, 0.01),
        ]
    )
    # Let noisy individual nights scatter more broadly, but require the robust
    # multi-night center itself to remain close to zero.  Actual event
    # boundaries are chosen only from the stricter <=0.015 mag subset below.
    recovery_tolerance = np.clip(2.0 * uncertainty, 0.015, 0.025)
    baseline_compatible = np.abs(corrected) <= recovery_tolerance
    strongly_dim = corrected >= np.maximum(0.02, 3.0 * uncertainty)
    support = np.zeros(corrected.shape, dtype=bool)
    confirmed = np.zeros(corrected.shape, dtype=bool)
    for start in range(len(corrected)):
        for width in (5, 4, 3):
            stop = start + width
            if stop > len(corrected):
                continue
            window = np.arange(start, stop)
            if times[window[-1]] - times[window[0]] > max_gap_days:
                continue
            if int(np.sum(baseline_compatible[window])) < 3:
                continue
            robust_center = float(np.nanmedian(corrected[window]))
            if abs(robust_center) <= 0.015:
                support[window] = True
                strict = window[np.abs(corrected[window]) <= 0.015]
                if strict.size:
                    window_center = 0.5 * (window[0] + window[-1])
                    representative = int(
                        strict[np.argmin(np.abs(strict - window_center))]
                    )
                    confirmed[representative] = True
                break
    return (
        confirmed,
        support,
        baseline_compatible,
        strongly_dim,
        0.015,
    )


def _nearest_recovery_indices(
    recovery_mask: np.ndarray,
    event_start: int,
    event_stop: int,
) -> tuple[int | None, int | None]:
    left_candidates = np.flatnonzero(recovery_mask[:event_start])
    right_candidates = np.flatnonzero(recovery_mask[event_stop + 1 :]) + event_stop + 1
    left = int(left_candidates[-1]) if left_candidates.size else None
    right = int(right_candidates[0]) if right_candidates.size else None
    return left, right


def _event_envelopes_from_recovery(
    recovery_mask: np.ndarray,
    recovery_support_mask: np.ndarray,
    left_anchor_mask: np.ndarray,
    right_anchor_mask: np.ndarray,
    times: np.ndarray,
    max_gap_days: float,
) -> list[dict[str, Any]]:
    """Enumerate recovery-anchored search brackets and edge-censored cases."""
    # Each true entry is the near-unity representative of a qualifying robust
    # recovery window.  Componentize those representatives—not the overlapping
    # support windows, which can otherwise chain across a shallow dip.
    plateaus = _mask_components(recovery_mask, times, max_gap_days)
    if not plateaus:
        # With neither side anchored at the quiescent baseline, the data do not
        # support either a completed event or a one-sided ongoing event.
        return []
    directional: list[tuple[int | None, int | None]] = []
    for start, stop in plateaus:
        left_candidates = np.flatnonzero(
            left_anchor_mask
            & (times <= times[stop])
            & (times[stop] - times <= max_gap_days)
        )
        right_candidates = np.flatnonzero(
            right_anchor_mask
            & (times >= times[start])
            & (times - times[start] <= max_gap_days)
        )
        directional.append(
            (
                int(left_candidates[-1]) if left_candidates.size else None,
                int(right_candidates[0]) if right_candidates.size else None,
            )
        )

    envelopes: list[dict[str, Any]] = []
    first_start, first_stop = plateaus[0]
    first_right = directional[0][1]
    first_boundary = first_right if first_right is not None else first_start
    if first_start > 0:
        envelopes.append(
            {
                "start": 0,
                "stop": int(first_boundary),
                "left_recovery": None,
                "right_recovery": first_right,
                "left_boundary_type": "data_edge",
                "right_boundary_type": (
                    "recovery" if first_right is not None else "unconfirmed_recovery"
                ),
            }
        )
    for plateau_index in range(len(plateaus) - 1):
        _, left_stop = plateaus[plateau_index]
        right_start, _ = plateaus[plateau_index + 1]
        left_anchor = directional[plateau_index][0]
        right_anchor = directional[plateau_index + 1][1]
        start = int(left_anchor if left_anchor is not None else left_stop)
        stop = int(right_anchor if right_anchor is not None else right_start)
        if stop > start + 1:
            envelopes.append(
                {
                    "start": start,
                    "stop": stop,
                    "left_recovery": left_anchor,
                    "right_recovery": right_anchor,
                    "left_boundary_type": (
                        "recovery" if left_anchor is not None else "unconfirmed_recovery"
                    ),
                    "right_boundary_type": (
                        "recovery" if right_anchor is not None else "unconfirmed_recovery"
                    ),
                }
            )
    last_start, last_stop = plateaus[-1]
    last_left = directional[-1][0]
    last_boundary = last_left if last_left is not None else last_stop
    if last_stop + 1 < len(times):
        envelopes.append(
            {
                "start": int(last_boundary),
                "stop": len(times) - 1,
                "left_recovery": last_left,
                "right_recovery": None,
                "left_boundary_type": (
                    "recovery" if last_left is not None else "unconfirmed_recovery"
                ),
                "right_boundary_type": "data_edge",
            }
        )
    return envelopes


def _directional_recovery_anchor_mask(
    corrected_residual: np.ndarray,
    sigma: np.ndarray,
    times: np.ndarray,
    quiet_scatter_mag: float,
    max_gap_days: float,
    *,
    side: str,
) -> np.ndarray:
    """Validate a recovery using only the quiescent side of its boundary.

    A left event anchor must be supported by the nights ending at that anchor;
    a right anchor by the nights starting there.  This prevents a mixed window
    that straddles the dip boundary from manufacturing a baseline recovery.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"unknown recovery-anchor side {side!r}")
    corrected = np.asarray(corrected_residual, float)
    sigma = np.asarray(sigma, float)
    uncertainty = np.maximum.reduce(
        [
            np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0),
            np.full(corrected.shape, max(quiet_scatter_mag, 0.0)),
            np.full(corrected.shape, 0.01),
        ]
    )
    tolerance = np.clip(2.0 * uncertainty, 0.015, 0.025)
    compatible = np.abs(corrected) <= tolerance
    validated = np.zeros(corrected.shape, dtype=bool)
    strict_anchor = np.abs(corrected) <= 0.015
    for anchor in np.flatnonzero(strict_anchor):
        for width in (5, 4, 3):
            if side == "left":
                start, stop = int(anchor - width + 1), int(anchor + 1)
            else:
                start, stop = int(anchor), int(anchor + width)
            if start < 0 or stop > len(corrected):
                continue
            window = np.arange(start, stop)
            if times[window[-1]] - times[window[0]] > max_gap_days:
                continue
            if int(np.sum(compatible[window])) < 3:
                continue
            if abs(float(np.nanmedian(corrected[window]))) <= 0.025:
                validated[anchor] = True
                break
    return validated


def _boundary_dim_supported(
    raw_residual: np.ndarray,
    corrected_residual: np.ndarray,
    detection_mask: np.ndarray,
    strongly_dim: np.ndarray,
    times: np.ndarray,
    index: int,
    *,
    direction: str,
    detection_threshold: float,
    max_gap_days: float,
) -> bool:
    """Test whether an observing-block edge is confidently still in the dip."""
    if direction == "before":
        available = np.flatnonzero(
            (times <= times[index]) & (times[index] - times <= max_gap_days)
        )[-4:]
    elif direction == "after":
        available = np.flatnonzero(
            (times >= times[index]) & (times - times[index] <= max_gap_days)
        )[:4]
    else:
        raise ValueError(f"unknown boundary direction {direction!r}")
    if len(available) < 2:
        return False
    if raw_residual[index] < detection_threshold:
        return False
    if corrected_residual[index] < detection_threshold:
        return False
    required = 2 if len(available) < 4 else 3
    return bool(
        np.sum(detection_mask[available]) >= required
        or np.sum(strongly_dim[available]) >= 2
    )


def _bridgeable_sampling_gaps(
    raw_residual: np.ndarray,
    corrected_residual: np.ndarray,
    detection_mask: np.ndarray,
    strongly_dim: np.ndarray,
    times: np.ndarray,
    *,
    detection_threshold: float,
    max_gap_days: float,
) -> np.ndarray:
    """Bridge a seasonal gap only when both observed edges remain confidently dim."""
    bridgeable = np.zeros(max(0, len(times) - 1), dtype=bool)
    for index in np.flatnonzero(np.diff(times) > max_gap_days):
        left_dim = _boundary_dim_supported(
            raw_residual,
            corrected_residual,
            detection_mask,
            strongly_dim,
            times,
            int(index),
            direction="before",
            detection_threshold=detection_threshold,
            max_gap_days=max_gap_days,
        )
        right_dim = _boundary_dim_supported(
            raw_residual,
            corrected_residual,
            detection_mask,
            strongly_dim,
            times,
            int(index + 1),
            direction="after",
            detection_threshold=detection_threshold,
            max_gap_days=max_gap_days,
        )
        bridgeable[index] = left_dim and right_dim
    return bridgeable


def _split_envelopes_at_unbridged_gaps(
    envelopes: list[tuple[int, int, int | None, int | None]],
    recovery_mask: np.ndarray,
    times: np.ndarray,
    bridgeable_gaps: np.ndarray,
    max_gap_days: float,
) -> list[dict[str, Any]]:
    """Split candidate events at gaps lacking dim-state support on both edges."""
    split: list[dict[str, Any]] = []
    large_gaps = np.diff(times) > max_gap_days
    barriers = large_gaps & ~bridgeable_gaps

    def recovery_before_gap(gap_index: int) -> int | None:
        candidates = np.flatnonzero(
            recovery_mask[: gap_index + 1]
            & (times[gap_index] - times[: gap_index + 1] <= max_gap_days)
        )
        return int(candidates[-1]) if candidates.size else None

    def recovery_after_gap(gap_index: int) -> int | None:
        candidates = np.flatnonzero(
            recovery_mask[gap_index + 1 :]
            & (times[gap_index + 1 :] - times[gap_index + 1] <= max_gap_days)
        )
        return int(gap_index + 1 + candidates[0]) if candidates.size else None

    for start, stop, left_recovery, right_recovery in envelopes:
        cuts = np.flatnonzero(barriers[start:stop]) + start
        segment_starts = np.r_[start, cuts + 1]
        segment_stops = np.r_[cuts, stop]
        for segment_index, (segment_start, segment_stop) in enumerate(
            zip(segment_starts, segment_stops)
        ):
            is_first = segment_index == 0
            is_last = segment_index == len(segment_starts) - 1
            segment_left_recovery = left_recovery if is_first else None
            segment_right_recovery = right_recovery if is_last else None
            left_gap_index = int(segment_start - 1) if not is_first else None
            right_gap_index = int(segment_stop) if not is_last else None
            split.append(
                {
                    "start": int(segment_start),
                    "stop": int(segment_stop),
                    "left_recovery": segment_left_recovery,
                    "right_recovery": segment_right_recovery,
                    "left_gap_recovery": (
                        recovery_before_gap(left_gap_index)
                        if left_gap_index is not None
                        else None
                    ),
                    "right_gap_recovery": (
                        recovery_after_gap(right_gap_index)
                        if right_gap_index is not None
                        else None
                    ),
                    # Preserve the outer baseline anchors even when an
                    # unbridged sampling gap forces the measurable event
                    # segment to begin/end inside that larger bracket.
                    "left_recovery_reference": left_recovery,
                    "right_recovery_reference": right_recovery,
                    "left_boundary_type": (
                        "recovery"
                        if segment_left_recovery is not None
                        else ("data_edge" if is_first and start == 0 else "gap")
                    ),
                    "right_boundary_type": (
                        "recovery"
                        if segment_right_recovery is not None
                        else (
                            "data_edge"
                            if is_last and stop == len(times) - 1
                            else "gap"
                        )
                    ),
                }
            )
    return split


def _capped_time_weights(times: np.ndarray, max_gap_days: float) -> np.ndarray:
    """Observed-time weights that never integrate across an unobserved long gap."""
    times = np.asarray(times, float)
    if len(times) == 1:
        return np.ones(1, dtype=float)
    weights = np.zeros(len(times), dtype=float)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap_days) + 1, len(times)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = times[start:stop]
        if len(block) == 1:
            weights[start] = 1.0
            continue
        gaps = np.diff(block)
        weights[start] = 0.5 * gaps[0]
        weights[stop - 1] = 0.5 * gaps[-1]
        if len(block) > 2:
            weights[start + 1 : stop - 1] = 0.5 * (gaps[:-1] + gaps[1:])
    return np.maximum(weights, 0.01)


def _edge_is_persistently_dim(
    raw_residual: np.ndarray,
    corrected_residual: np.ndarray,
    detection_mask: np.ndarray,
    strongly_dim: np.ndarray,
    times: np.ndarray,
    *,
    side: str,
    detection_threshold: float,
    max_gap_days: float,
) -> bool:
    """Require a coherent dim state at the true data edge before censoring there."""
    if side not in {"left", "right"}:
        raise ValueError(f"unknown edge side {side!r}")
    edge = 0 if side == "left" else len(times) - 1
    if not np.isfinite(raw_residual[edge]):
        return False
    # The actual terminal epoch must itself still be dim.  A nearest-neighbour
    # smoother is not allowed to turn a recovered edge point into an ongoing
    # event.
    if raw_residual[edge] < detection_threshold:
        return False
    if not np.isfinite(corrected_residual[edge]):
        return False
    if corrected_residual[edge] < detection_threshold:
        return False

    if side == "left":
        available = np.flatnonzero(times - times[0] <= max_gap_days)[:4]
    else:
        available = np.flatnonzero(times[-1] - times <= max_gap_days)[-4:]
    if len(available) < 2:
        # Long dips can remain active across a seasonal gap.  Retain that
        # one-sided, lower-limit interpretation only when the terminal epoch
        # and all four most recent *observed* epochs are strongly dim.  The
        # unobserved gap remains explicit in the event-envelope metadata.
        recent = (
            np.arange(min(4, len(times)))
            if side == "left"
            else np.arange(max(0, len(times) - 4), len(times))
        )
        return bool(
            strongly_dim[edge]
            and len(recent) >= 4
            and np.all(strongly_dim[recent])
        )
    required = 2 if len(available) < 4 else 3
    return bool(
        np.sum(strongly_dim[available]) >= 2
        or np.sum(detection_mask[available]) >= required
    )


def _robust_peak_triplet(
    times: np.ndarray,
    residual: np.ndarray,
    local_profile: np.ndarray,
    start: int,
    stop: int,
    max_gap_days: float,
) -> tuple[float, float, np.ndarray]:
    """Return the strongest gap-safe nearest-three local median in an event."""
    neighbors = _local_epoch_neighbor_matrix(times, max_gap_days)
    best: tuple[float, float, np.ndarray] | None = None
    for anchor in range(start, stop + 1):
        indices = neighbors[anchor]
        indices = indices[indices >= 0]
        if len(indices) != 3 or np.any(indices < start) or np.any(indices > stop):
            continue
        depth = float(local_profile[anchor])
        if not np.isfinite(depth):
            continue
        # This should equal the supplied local profile by construction.  Keep
        # the explicit raw median as a guard against future neighborhood drift.
        raw_depth = float(np.nanmedian(residual[indices]))
        if not np.isclose(depth, raw_depth, rtol=0.0, atol=1e-12):
            continue
        peak_time = float(times[anchor])
        if best is None or depth > best[1]:
            best = (peak_time, depth, indices)
    if best is not None:
        return best
    return np.nan, np.nan, np.array([], dtype=int)


def _local_epoch_neighbor_matrix(times: np.ndarray, max_gap_days: float) -> np.ndarray:
    """Return the fixed nearest-three neighborhood used by the local median."""
    times = np.asarray(times, float)
    neighbors = np.full((len(times), 3), -1, dtype=int)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap_days) + 1, len(times)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = np.arange(start, stop)
        for index in block:
            order = np.argsort(np.abs(times[block] - times[index]))
            local = block[order[: min(3, len(block))]]
            neighbors[index, : len(local)] = local
    return neighbors


def _mc_interval(
    values: np.ndarray,
    point: float,
    prefix: str,
) -> dict[str, float | int]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 16 or not np.isfinite(point):
        return {
            f"{prefix}_raw_p16": np.nan,
            f"{prefix}_raw_p50": np.nan,
            f"{prefix}_raw_p84": np.nan,
            f"{prefix}_p16": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p84": np.nan,
            f"{prefix}_err_minus": np.nan,
            f"{prefix}_err_plus": np.nan,
            f"{prefix}_n_valid": int(finite.size),
        }
    raw_p16, raw_p50, raw_p84 = np.nanpercentile(finite, [16.0, 50.0, 84.0])
    # The perturb-and-reselect maximum is predictably shifted upward by the
    # winner's-curse operation itself.  Use the Monte Carlo spread, centered on
    # the observed estimator, rather than presenting that simulation bias as a
    # one-sided physical confidence interval.
    err_minus = float(max(0.0, raw_p50 - raw_p16))
    err_plus = float(max(0.0, raw_p84 - raw_p50))
    return {
        f"{prefix}_raw_p16": float(raw_p16),
        f"{prefix}_raw_p50": float(raw_p50),
        f"{prefix}_raw_p84": float(raw_p84),
        f"{prefix}_p16": float(max(0.0, point - err_minus)),
        f"{prefix}_p50": float(point),
        f"{prefix}_p84": float(point + err_plus),
        f"{prefix}_err_minus": err_minus,
        f"{prefix}_err_plus": err_plus,
        f"{prefix}_n_valid": int(finite.size),
    }


def _estimate_event_uncertainties(
    candidate_id: str,
    times: np.ndarray,
    raw_residual: np.ndarray,
    sigma: np.ndarray,
    *,
    quiet_scatter_mag: float,
    event_start: int,
    event_stop: int,
    peak_indices: np.ndarray,
    peak_index: int,
    smooth_window_days: float,
    crossing_gap_limit_days: float,
    tau_point: float,
    delta_mag_point: float,
    duration_point: float,
    duration_status: str,
    left_event_boundary_type: str,
    right_event_boundary_type: str,
    n_draws: int = DEFAULT_EVENT_MC_DRAWS,
) -> dict[str, float | int | str]:
    """Conditional Monte Carlo errors for amplitude and measured FWHM support.

    The accepted recovery-anchored search bracket is held fixed.  Each draw
    perturbs all nightly residuals and reselects the strongest gap-safe
    three-night peak for the amplitude error.  The FWHM error stays attached to
    the selected physical peak, recomputes its local profile/crossings, and
    retains the same censoring rules as the point estimate.
    """
    result: dict[str, float | int | str] = {
        "uncertainty_method": "conditional_nightly_parametric_mc_v2",
        "uncertainty_draws": int(n_draws),
        "duration_mc_resolved_fraction": np.nan,
        "duration_mc_reporting_status": "not_evaluated",
    }
    if event_stop - event_start + 1 < 3:
        result["uncertainty_method"] = "unavailable_event_too_short"
        return result

    neighbor_indices = _local_epoch_neighbor_matrix(times, smooth_window_days)
    triplets = []
    for anchor in range(event_start, event_stop + 1):
        indices = neighbor_indices[anchor]
        indices = indices[indices >= 0]
        if (
            len(indices) == 3
            and np.all(indices >= event_start)
            and np.all(indices <= event_stop)
        ):
            triplets.append(indices)
    if not triplets:
        result["uncertainty_method"] = "unavailable_no_gap_safe_triplet"
        return result
    triplet_indices = np.asarray(triplets, dtype=int)

    digest = hashlib.sha256(str(candidate_id).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    scale = np.maximum.reduce(
        [
            np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0),
            np.full(len(times), max(quiet_scatter_mag, 0.005)),
            np.full(len(times), 0.005),
        ]
    )
    draws = raw_residual[None, :] + rng.normal(
        loc=0.0,
        scale=scale[None, :],
        size=(n_draws, len(times)),
    )

    triplet_depths = np.nanmedian(draws[:, triplet_indices], axis=2)
    winners = np.nanargmax(triplet_depths, axis=1)
    draw_rows = np.arange(n_draws)
    delta_draws = triplet_depths[draw_rows, winners]
    tau_draws = 1.0 - np.power(10.0, -0.4 * delta_draws)
    fixed_peak_indices = np.asarray(peak_indices, dtype=int)
    fixed_peak_depth_draws = np.nanmedian(draws[:, fixed_peak_indices], axis=1)

    safe_neighbors = np.where(neighbor_indices >= 0, neighbor_indices, 0)
    neighborhood = draws[:, safe_neighbors]
    neighborhood = np.where(
        neighbor_indices[None, :, :] >= 0,
        neighborhood,
        np.nan,
    )
    smooth_draws = np.nanmedian(neighborhood, axis=2)
    recovery_extension = HALF_DEPTH_RECOVERY_WINDOW_EPOCHS - 1
    recovery_window_indices = _persistent_recovery_window_indices(
        times,
        max(0, event_start - recovery_extension),
        min(len(times) - 1, event_stop + recovery_extension),
        crossing_gap_limit_days,
    )

    duration_draws = np.full(n_draws, np.nan, dtype=float)
    resolved_draws = np.zeros(n_draws, dtype=bool)
    for draw_index in range(n_draws):
        delta_draw = float(fixed_peak_depth_draws[draw_index])
        tau_draw = float(1.0 - 10.0 ** (-0.4 * delta_draw))
        if not np.isfinite(delta_draw) or delta_draw <= 0 or not (0 < tau_draw < 1):
            continue
        half_level = float(-2.5 * np.log10(max(1.0 - 0.5 * tau_draw, 1e-9)))
        profile = smooth_draws[draw_index]
        inside = _close_single_epoch_holes(
            profile >= half_level,
            times,
            2.0 * crossing_gap_limit_days,
        )
        inside[:event_start] = False
        inside[event_stop + 1 :] = False
        anchor = int(peak_index)
        if not inside[anchor]:
            anchor = int(
                fixed_peak_indices[
                    np.nanargmax(profile[fixed_peak_indices])
                ]
            )
        if not inside[anchor]:
            continue
        try:
            half_depth_draw = _persistent_half_depth_measurement(
                times,
                draws[draw_index],
                profile,
                half_level=half_level,
                anchor=anchor,
                event_start=event_start,
                event_stop=event_stop,
                crossing_gap_limit_days=crossing_gap_limit_days,
                left_event_boundary_type=left_event_boundary_type,
                right_event_boundary_type=right_event_boundary_type,
                recovery_window_indices=recovery_window_indices,
            )
        except RuntimeError:
            continue
        duration_draw = _finite_number(half_depth_draw["duration_plot_days"])
        if np.isfinite(duration_draw):
            duration_draws[draw_index] = duration_draw
        resolved_draws[draw_index] = half_depth_draw["duration_status"] == "resolved"

    result.update(_mc_interval(delta_draws, delta_mag_point, "delta_mag_peak_mc"))
    result.update(_mc_interval(tau_draws, tau_point, "tau_peak_mc"))
    finite_duration = np.isfinite(duration_draws)
    result["duration_mc_resolved_fraction"] = float(np.mean(resolved_draws))
    resolved_duration_draws = duration_draws[finite_duration & resolved_draws]
    resolved_fraction = _finite_number(result["duration_mc_resolved_fraction"])
    duration_is_reportable = bool(
        duration_status == "resolved"
        and np.isfinite(resolved_fraction)
        and resolved_fraction >= MIN_DURATION_MC_RESOLVED_FRACTION
        and resolved_duration_draws.size >= MIN_DURATION_MC_RESOLVED_DRAWS
    )
    if duration_is_reportable:
        result.update(
            _mc_interval(resolved_duration_draws, duration_point, "duration_mc")
        )
        result["duration_mc_reporting_status"] = "reported_resolved"
    else:
        result.update(_mc_interval(np.array([], dtype=float), duration_point, "duration_mc"))
        result["duration_mc_n_valid"] = int(resolved_duration_draws.size)
        if duration_status != "resolved":
            result["duration_mc_reporting_status"] = "structurally_censored"
        elif resolved_duration_draws.size < MIN_DURATION_MC_RESOLVED_DRAWS:
            result["duration_mc_reporting_status"] = "insufficient_resolved_draws"
        else:
            result["duration_mc_reporting_status"] = "unstable_crossing_classification"
    return result


def _close_single_epoch_holes(
    mask: np.ndarray,
    times: np.ndarray,
    max_span_days: float,
) -> np.ndarray:
    """Suppress a lone noisy epoch inside an otherwise coherent dim state."""
    closed = np.asarray(mask, bool).copy()
    for index in range(1, len(closed) - 1):
        if (
            not closed[index]
            and closed[index - 1]
            and closed[index + 1]
            and times[index + 1] - times[index - 1] <= max_span_days
        ):
            closed[index] = True
    return closed


def _persistent_recovery_windows(
    times: np.ndarray,
    recovered_mask: np.ndarray,
    start: int,
    stop: int,
    max_gap_days: float,
    *,
    window_epochs: int = HALF_DEPTH_RECOVERY_WINDOW_EPOCHS,
    required_epochs: int = HALF_DEPTH_RECOVERY_REQUIRED_EPOCHS,
    min_span_days: float = HALF_DEPTH_RECOVERY_MIN_SPAN_DAYS,
) -> list[tuple[int, int]]:
    """Return persistent half-depth recovery windows in one observing block.

    A window contains consecutive *observed nightly bins*, not necessarily
    consecutive calendar dates.  It must contain at least five recovered
    epochs out of six, span at least seven days, and contain no cadence gap
    larger than the FWHM crossing limit.
    """
    recovered = np.asarray(recovered_mask, bool)
    candidate_windows = _persistent_recovery_window_indices(
        times,
        start,
        stop,
        max_gap_days,
        window_epochs=window_epochs,
        min_span_days=min_span_days,
    )
    if not candidate_windows.size:
        return []
    cumulative = np.r_[0, np.cumsum(recovered.astype(int))]
    counts = cumulative[candidate_windows[:, 1] + 1] - cumulative[candidate_windows[:, 0]]
    sizes = candidate_windows[:, 1] - candidate_windows[:, 0] + 1
    required = np.ceil(sizes * required_epochs / window_epochs).astype(int)
    qualified = candidate_windows[counts >= required]
    return [(int(window[0]), int(window[1])) for window in qualified]


def _persistent_recovery_window_indices(
    times: np.ndarray,
    start: int,
    stop: int,
    max_gap_days: float,
    *,
    window_epochs: int = HALF_DEPTH_RECOVERY_WINDOW_EPOCHS,
    min_span_days: float = HALF_DEPTH_RECOVERY_MIN_SPAN_DAYS,
) -> np.ndarray:
    """Return bounds of time-valid recovery sequences.

    Six well-spaced epochs remain the minimum.  For denser cadence, the
    shortest consecutive sequence reaching seven days can contain more than
    six epochs and must retain the same five-sixths recovered fraction.
    """
    times = np.asarray(times, float)
    start = max(0, int(start))
    stop = min(len(times) - 1, int(stop))
    if window_epochs <= 0 or stop - start + 1 < window_epochs:
        return np.empty((0, 2), dtype=int)
    bounds: list[tuple[int, int]] = []
    block_starts = np.r_[start, np.flatnonzero(np.diff(times[start : stop + 1]) > max_gap_days) + start + 1]
    block_stops = np.r_[block_starts[1:] - 1, stop]
    for block_start, block_stop in zip(block_starts, block_stops):
        for window_start in range(int(block_start), int(block_stop) - window_epochs + 2):
            minimum_stop = window_start + window_epochs - 1
            possible = np.flatnonzero(
                times[minimum_stop : int(block_stop) + 1] - times[window_start]
                >= min_span_days
            )
            if possible.size:
                bounds.append((window_start, int(minimum_stop + possible[0])))
    return np.asarray(bounds, dtype=int).reshape(-1, 2)


def _persistent_half_depth_measurement(
    times: np.ndarray,
    raw_residual: np.ndarray,
    profile: np.ndarray,
    *,
    half_level: float,
    anchor: int,
    event_start: int,
    event_stop: int,
    crossing_gap_limit_days: float,
    left_event_boundary_type: str,
    right_event_boundary_type: str,
    recovery_window_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure a peak-centered half-depth span using persistent recovery.

    Sampling gaps never terminate the scan.  A side terminates only after a
    five-of-six nightly recovery window spanning at least seven days within a
    single observing block.  A crossing hidden by a gap is returned as a
    finite interval; a side lacking persistent recovery remains open-censored.
    """
    times = np.asarray(times, float)
    raw_residual = np.asarray(raw_residual, float)
    profile = np.asarray(profile, float)
    anchor = int(anchor)
    event_start = int(event_start)
    event_stop = int(event_stop)

    inside = _close_single_epoch_holes(
        np.isfinite(profile) & (profile >= half_level),
        times,
        2.0 * crossing_gap_limit_days,
    )
    inside[:event_start] = False
    inside[event_stop + 1 :] = False
    if not inside[anchor]:
        event_indices = np.arange(event_start, event_stop + 1)
        supported = event_indices[inside[event_indices]]
        if supported.size:
            anchor = int(supported[np.nanargmax(profile[supported])])
    if not inside[anchor]:
        raise RuntimeError("selected peak is not supported by the half-depth profile")

    recovered = np.isfinite(raw_residual) & (raw_residual < half_level)
    extension = HALF_DEPTH_RECOVERY_WINDOW_EPOCHS - 1
    search_start = max(0, event_start - extension)
    search_stop = min(len(times) - 1, event_stop + extension)
    if recovery_window_indices is None:
        recovery_window_indices = _persistent_recovery_window_indices(
            times,
            search_start,
            search_stop,
            crossing_gap_limit_days,
        )
    recovery_window_indices = np.asarray(recovery_window_indices, dtype=int)
    if recovery_window_indices.size:
        cumulative = np.r_[0, np.cumsum(recovered.astype(int))]
        counts = (
            cumulative[recovery_window_indices[:, 1] + 1]
            - cumulative[recovery_window_indices[:, 0]]
        )
        sizes = recovery_window_indices[:, 1] - recovery_window_indices[:, 0] + 1
        required = np.ceil(
            sizes
            * HALF_DEPTH_RECOVERY_REQUIRED_EPOCHS
            / HALF_DEPTH_RECOVERY_WINDOW_EPOCHS
        ).astype(int)
        qualified = recovery_window_indices[counts >= required]
        windows = [(int(window[0]), int(window[1])) for window in qualified]
    else:
        windows = []
    left_windows = [window for window in windows if window[1] < anchor]
    right_windows = [window for window in windows if window[0] > anchor]
    left_window = max(left_windows, key=lambda window: window[1]) if left_windows else None
    right_window = min(right_windows, key=lambda window: window[0]) if right_windows else None

    def left_side() -> dict[str, Any]:
        if left_window is not None:
            window_start, window_stop = left_window
            profile_recovered = np.flatnonzero(
                np.isfinite(profile[window_start : window_stop + 1])
                & (profile[window_start : window_stop + 1] < half_level)
            ) + window_start
            if profile_recovered.size:
                outside_index = int(profile_recovered[-1])
                inside_candidates = np.flatnonzero(inside[outside_index + 1 : anchor + 1])
                if inside_candidates.size:
                    inside_index = int(outside_index + 1 + inside_candidates[0])
                    lower = float(times[outside_index])
                    upper = float(times[inside_index])
                    exact = bool(
                        inside_index == outside_index + 1
                        and upper - lower <= crossing_gap_limit_days
                    )
                    point = (
                        _crossing_time(
                            lower,
                            float(profile[outside_index]),
                            upper,
                            float(profile[inside_index]),
                            half_level,
                        )
                        if exact
                        else np.nan
                    )
                    return {
                        "status": "exact" if exact else "interval",
                        "source": "persistent_recovery" if exact else "sampling_gap",
                        "point": point,
                        "lower": lower,
                        "upper": upper,
                        "inside_index": inside_index,
                        "outside_index": outside_index,
                        "window": left_window,
                    }
        inside_candidates = np.flatnonzero(inside[event_start : anchor + 1])
        inside_index = int(event_start + inside_candidates[0])
        return {
            "status": "censored",
            "source": str(left_event_boundary_type),
            "point": np.nan,
            "lower": np.nan,
            "upper": float(times[inside_index]),
            "inside_index": inside_index,
            "outside_index": None,
            "window": None,
        }

    def right_side() -> dict[str, Any]:
        if right_window is not None:
            window_start, window_stop = right_window
            profile_recovered = np.flatnonzero(
                np.isfinite(profile[window_start : window_stop + 1])
                & (profile[window_start : window_stop + 1] < half_level)
            ) + window_start
            if profile_recovered.size:
                outside_index = int(profile_recovered[0])
                inside_candidates = np.flatnonzero(inside[anchor:outside_index])
                if inside_candidates.size:
                    inside_index = int(anchor + inside_candidates[-1])
                    lower = float(times[inside_index])
                    upper = float(times[outside_index])
                    exact = bool(
                        outside_index == inside_index + 1
                        and upper - lower <= crossing_gap_limit_days
                    )
                    point = (
                        _crossing_time(
                            lower,
                            float(profile[inside_index]),
                            upper,
                            float(profile[outside_index]),
                            half_level,
                        )
                        if exact
                        else np.nan
                    )
                    return {
                        "status": "exact" if exact else "interval",
                        "source": "persistent_recovery" if exact else "sampling_gap",
                        "point": point,
                        "lower": lower,
                        "upper": upper,
                        "inside_index": inside_index,
                        "outside_index": outside_index,
                        "window": right_window,
                    }
        inside_candidates = np.flatnonzero(inside[anchor : event_stop + 1])
        inside_index = int(anchor + inside_candidates[-1])
        return {
            "status": "censored",
            "source": str(right_event_boundary_type),
            "point": np.nan,
            "lower": float(times[inside_index]),
            "upper": np.nan,
            "inside_index": inside_index,
            "outside_index": None,
            "window": None,
        }

    left = left_side()
    right = right_side()
    left_inside = int(left["inside_index"])
    right_inside = int(right["inside_index"])
    if right_inside < left_inside:
        left_inside = right_inside = anchor
    support_mask = inside.copy()
    support_mask[:left_inside] = False
    support_mask[right_inside + 1 :] = False
    internal_gaps = np.diff(times[left_inside : right_inside + 1])
    large_internal_gaps = internal_gaps[internal_gaps > crossing_gap_limit_days]
    internal_gap_count = int(large_internal_gaps.size)

    left_finite = left["status"] in {"exact", "interval"}
    right_finite = right["status"] in {"exact", "interval"}
    duration = np.nan
    duration_lower = np.nan
    duration_upper = np.nan
    duration_plot = np.nan
    lower_limit = False
    interval_censored = False
    if left_finite and right_finite:
        duration_lower = max(0.0, float(right["lower"] - left["upper"]))
        duration_upper = max(duration_lower, float(right["upper"] - left["lower"]))
        if left["status"] == "exact" and right["status"] == "exact":
            duration = max(0.0, float(right["point"] - left["point"]))
            duration_plot = duration
            duration_status = "resolved"
        else:
            interval_censored = True
            duration_status = "interval_censored"
            duration_plot = (
                float(np.sqrt(duration_lower * duration_upper))
                if duration_lower > 0 and duration_upper > 0
                else 0.5 * (duration_lower + duration_upper)
            )
    elif left_finite:
        lower_limit = True
        left_reference = float(left["point"] if left["status"] == "exact" else left["upper"])
        duration_lower = max(0.0, float(times[right_inside] - left_reference))
        duration_plot = duration_lower if duration_lower > 0 else np.nan
        duration_status = "right_censored"
    elif right_finite:
        lower_limit = True
        right_reference = float(right["point"] if right["status"] == "exact" else right["lower"])
        duration_lower = max(0.0, float(right_reference - times[left_inside]))
        duration_plot = duration_lower if duration_lower > 0 else np.nan
        duration_status = "left_censored"
    else:
        lower_limit = True
        duration_lower = max(0.0, float(times[right_inside] - times[left_inside]))
        duration_plot = duration_lower if duration_lower > 0 else np.nan
        duration_status = "both_censored"

    return {
        "anchor": anchor,
        "inside_mask": support_mask,
        "left": left,
        "right": right,
        "left_inside": left_inside,
        "right_inside": right_inside,
        "n_half_depth_epochs": int(np.sum(support_mask[left_inside : right_inside + 1])),
        "observed_half_depth_span_days": float(times[right_inside] - times[left_inside]),
        "internal_gap_count": internal_gap_count,
        "max_internal_gap_days": (
            float(np.max(large_internal_gaps)) if large_internal_gaps.size else 0.0
        ),
        "half_depth_continuity_assumed": bool(internal_gap_count),
        "half_depth_duration_days": duration,
        "duration_lower_days": duration_lower,
        "duration_upper_days": duration_upper,
        "duration_plot_days": duration_plot,
        "duration_is_lower_limit": lower_limit,
        "duration_is_interval_censored": interval_censored,
        "duration_status": duration_status,
    }


def _robust_local_baseline(values: np.ndarray) -> float:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    scatter = float(1.4826 * np.median(np.abs(finite - center)))
    if np.isfinite(scatter) and scatter > 0:
        clipped = finite[np.abs(finite - center) <= 3.0 * scatter]
        if clipped.size:
            center = float(np.median(clipped))
    return center


def _legacy_inline_measure_half_depth_event(
    candidate_id: str,
    lc_path: Path,
    *,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Measure one recovery-anchored ASAS-SN dimming episode.

    The dip depth is initialized from the deepest local-median epochs rather
    than one noisy point.  The outer bracket extends to statistically stable
    recovery near relative flux one.  A completed bracket needs directional
    recovery on both sides; otherwise a mixed boundary remains explicitly
    unconfirmed, or a genuinely ongoing event is censored at the actual data
    edge.  The half-depth scan crosses sampling gaps and terminates only after
    five of six nightly medians recover within one observing block spanning at
    least seven days.  A crossing hidden by a gap remains a finite interval;
    only a truly open side is reported as a lower limit.
    """
    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "lc_path_measured": str(lc_path),
        "n_good_observations": 0,
        "n_epoch_bins": 0,
        "cadence_days": np.nan,
        "gap_cap_days": np.nan,
        "smooth_window_days": np.nan,
        "recovery_window_days": np.nan,
        "crossing_gap_limit_days": np.nan,
        "local_baseline_residual_mag": np.nan,
        "quiet_scatter_mag": np.nan,
        "peak_initialization_points": 0,
        "detection_threshold_mag": np.nan,
        "event_selection_score": np.nan,
        "event_integrated_excess": np.nan,
        "event_component_epochs": 0,
        "recovery_threshold_mag": np.nan,
        "recovery_flux_threshold": np.nan,
        "left_baseline_recovered": False,
        "right_baseline_recovered": False,
        "left_edge_dim_confirmed": False,
        "right_edge_dim_confirmed": False,
        "left_event_boundary_type": "unknown",
        "right_event_boundary_type": "unknown",
        "left_gap_boundary_state": "none",
        "right_gap_boundary_state": "none",
        "event_window_is_lower_limit": False,
        "left_recovery_jd": np.nan,
        "right_recovery_jd": np.nan,
        "left_recovery_is_gap_bracket": False,
        "right_recovery_is_gap_bracket": False,
        "event_window_start_jd": np.nan,
        "event_window_end_jd": np.nan,
        "event_window_duration_days": np.nan,
        "event_window_gap_count": 0,
        "event_window_max_gap_days": 0.0,
        "event_window_status": "measurement_failed",
        "event_metrics_schema_version": EVENT_METRICS_SCHEMA_VERSION,
        "dimming_complex_start_jd": np.nan,
        "dimming_complex_end_jd": np.nan,
        "dimming_complex_duration_days": np.nan,
        "dimming_complex_duration_lower_days": np.nan,
        "dimming_complex_duration_upper_days": np.nan,
        "dimming_complex_duration_plot_days": np.nan,
        "dimming_complex_is_lower_limit": False,
        "dimming_complex_status": "measurement_failed",
        "support_status": "measurement_failed",
        "fwhm_method_version": FWHM_METHOD_VERSION,
        "peak_jd": np.nan,
        "delta_mag_peak": np.nan,
        "tau_peak": np.nan,
        "half_depth_delta_mag": np.nan,
        "n_half_depth_epochs": 0,
        "observed_half_depth_span_days": np.nan,
        "half_depth_recovery_window_epochs": HALF_DEPTH_RECOVERY_WINDOW_EPOCHS,
        "half_depth_recovery_required_epochs": HALF_DEPTH_RECOVERY_REQUIRED_EPOCHS,
        "half_depth_recovery_min_span_days": HALF_DEPTH_RECOVERY_MIN_SPAN_DAYS,
        "left_bracketed": False,
        "right_bracketed": False,
        "left_crossing_status": "unavailable",
        "right_crossing_status": "unavailable",
        "left_crossing_source": "unavailable",
        "right_crossing_source": "unavailable",
        "left_crossing_resolved": False,
        "right_crossing_resolved": False,
        "left_crossing_gap_censored": False,
        "right_crossing_gap_censored": False,
        "left_crossing_time": np.nan,
        "right_crossing_time": np.nan,
        "left_crossing_lower_jd": np.nan,
        "left_crossing_upper_jd": np.nan,
        "right_crossing_lower_jd": np.nan,
        "right_crossing_upper_jd": np.nan,
        "half_depth_duration_days": np.nan,
        "duration_lower_days": np.nan,
        "duration_upper_days": np.nan,
        "duration_plot_days": np.nan,
        "duration_is_lower_limit": False,
        "duration_is_interval_censored": False,
        "internal_gap_count": 0,
        "max_internal_gap_days": 0.0,
        "half_depth_continuity_assumed": False,
        "event_continuity_assumed": False,
        "duration_status": "measurement_failed",
        "uncertainty_method": "not_computed",
        "uncertainty_draws": 0,
        "delta_mag_peak_mc_p16": np.nan,
        "delta_mag_peak_mc_p50": np.nan,
        "delta_mag_peak_mc_p84": np.nan,
        "delta_mag_peak_mc_raw_p16": np.nan,
        "delta_mag_peak_mc_raw_p50": np.nan,
        "delta_mag_peak_mc_raw_p84": np.nan,
        "delta_mag_peak_mc_err_minus": np.nan,
        "delta_mag_peak_mc_err_plus": np.nan,
        "delta_mag_peak_mc_n_valid": 0,
        "tau_peak_mc_p16": np.nan,
        "tau_peak_mc_p50": np.nan,
        "tau_peak_mc_p84": np.nan,
        "tau_peak_mc_raw_p16": np.nan,
        "tau_peak_mc_raw_p50": np.nan,
        "tau_peak_mc_raw_p84": np.nan,
        "tau_peak_mc_err_minus": np.nan,
        "tau_peak_mc_err_plus": np.nan,
        "tau_peak_mc_n_valid": 0,
        "duration_mc_p16": np.nan,
        "duration_mc_p50": np.nan,
        "duration_mc_p84": np.nan,
        "duration_mc_raw_p16": np.nan,
        "duration_mc_raw_p50": np.nan,
        "duration_mc_raw_p84": np.nan,
        "duration_mc_err_minus": np.nan,
        "duration_mc_err_plus": np.nan,
        "duration_mc_n_valid": 0,
        "duration_mc_resolved_fraction": np.nan,
        "duration_mc_reporting_status": "not_computed",
        "measurement_error": "",
    }
    try:
        canonical = load_lightcurve_df(
            lc_path,
            filter_bad_cameras_enabled=True,
            apply_quality=True,
        )
        analysis = clean_lc(to_asassn_algorithm_frame(canonical))
        baseline = per_camera_gp_baseline_masked(
            analysis,
            S0=0.0005,
            w0=0.0031415926535897933,
            q=0.7,
            jitter=0.006,
        )
        obs = pd.DataFrame(
            {
                "t": pd.to_numeric(baseline["JD"], errors="coerce"),
                "resid": pd.to_numeric(baseline["resid"], errors="coerce"),
                "sigma": pd.to_numeric(baseline["sigma_eff"], errors="coerce"),
            }
        ).dropna()
        obs = obs.loc[obs["sigma"] > 0].sort_values("t")
        if obs.empty:
            raise RuntimeError("no finite baseline residuals")

        obs["night"] = np.floor(obs["t"]).astype(int)
        epoch = (
            obs.groupby("night", sort=True)
            .agg(t=("t", "median"), resid=("resid", "median"), sigma=("sigma", "median"))
            .reset_index(drop=True)
        )
        times = epoch["t"].to_numpy(float)
        residual = epoch["resid"].to_numpy(float)
        sigma = epoch["sigma"].to_numpy(float)
        cadence, _ = _cadence_gap_cap(times)
        smooth_residual, smooth_window = _local_epoch_median(times, residual, cadence)
        smooth_sigma, _ = _local_epoch_median(times, sigma, cadence)
        crossing_gap_limit = float(
            min(14.0, max(5.0, 3.0 * cadence if np.isfinite(cadence) else 5.0))
        )
        recovery_window = 30.0
        # ``resid == 0`` is the fixed GP-normalized quiescent baseline.  Do not
        # move relative flux one after choosing an event.
        local_baseline = 0.0
        corrected_residual = smooth_residual.copy()
        raw_corrected = residual.copy()
        absolute_residual = np.abs(raw_corrected[np.isfinite(raw_corrected)])
        core_limit = (
            float(np.nanpercentile(absolute_residual, 60.0))
            if absolute_residual.size
            else 0.02
        )
        quiet_values = raw_corrected[
            np.isfinite(raw_corrected) & (np.abs(raw_corrected) <= core_limit)
        ]
        quiet_center = float(np.nanmedian(quiet_values)) if quiet_values.size else 0.0
        quiet_scatter = (
            float(1.4826 * np.nanmedian(np.abs(quiet_values - quiet_center)))
            if quiet_values.size
            else 0.005
        )
        quiet_scatter = float(np.clip(quiet_scatter, 0.005, 0.03))
        (
            recovery_mask,
            recovery_support_mask,
            baseline_compatible,
            strongly_dim,
            recovery_threshold,
        ) = (
            _stable_recovery_mask(
                raw_corrected,
                sigma,
                times,
                quiet_scatter,
                recovery_window,
            )
        )
        left_recovery_anchor_mask = _directional_recovery_anchor_mask(
            raw_corrected,
            sigma,
            times,
            quiet_scatter,
            recovery_window,
            side="left",
        )
        right_recovery_anchor_mask = _directional_recovery_anchor_mask(
            raw_corrected,
            sigma,
            times,
            quiet_scatter,
            recovery_window,
            side="right",
        )
        finite_sigma = smooth_sigma[np.isfinite(smooth_sigma) & (smooth_sigma > 0)]
        typical_sigma = float(np.nanmedian(finite_sigma)) if finite_sigma.size else 0.01
        detection_threshold = float(max(0.02, min(0.08, 2.0 * typical_sigma)))
        detection_mask = _close_single_epoch_holes(
            corrected_residual >= detection_threshold,
            times,
            2.0 * crossing_gap_limit,
        )
        # This is an outer recovery-to-recovery *search bracket*, not a claim
        # that the source stayed dim throughout every unobserved gap.  Long
        # dips can span seasons, so retain the nearest observed recovery
        # plateaus and report all intervening gaps.  The local FWHM component
        # below is measured separately and never bridges a sampling gap.
        envelopes = []
        for recovery_envelope in _event_envelopes_from_recovery(
            recovery_mask,
            recovery_support_mask,
            left_recovery_anchor_mask,
            right_recovery_anchor_mask,
            times,
            recovery_window,
        ):
            start = int(recovery_envelope["start"])
            stop = int(recovery_envelope["stop"])
            left_recovery = recovery_envelope["left_recovery"]
            right_recovery = recovery_envelope["right_recovery"]
            envelopes.append(
                {
                    "start": start,
                    "stop": stop,
                    "left_recovery": left_recovery,
                    "right_recovery": right_recovery,
                    "left_gap_recovery": None,
                    "right_gap_recovery": None,
                    "left_recovery_reference": left_recovery,
                    "right_recovery_reference": right_recovery,
                    "left_boundary_type": recovery_envelope["left_boundary_type"],
                    "right_boundary_type": recovery_envelope["right_boundary_type"],
                }
            )
        time_weights = _capped_time_weights(times, smooth_window)
        candidates: list[dict[str, Any]] = []
        for envelope in envelopes:
            start = int(envelope["start"])
            stop = int(envelope["stop"])
            left_recovery = envelope["left_recovery"]
            right_recovery = envelope["right_recovery"]
            left_gap_recovery = envelope["left_gap_recovery"]
            right_gap_recovery = envelope["right_gap_recovery"]
            left_recovery_reference = envelope["left_recovery_reference"]
            right_recovery_reference = envelope["right_recovery_reference"]
            left_boundary_type = str(envelope["left_boundary_type"])
            right_boundary_type = str(envelope["right_boundary_type"])
            if stop - start + 1 < 3:
                continue
            left_edge_dim = bool(
                left_boundary_type == "data_edge"
                and _edge_is_persistently_dim(
                    raw_corrected,
                    corrected_residual,
                    detection_mask,
                    strongly_dim,
                    times,
                    side="left",
                    detection_threshold=detection_threshold,
                    max_gap_days=smooth_window,
                )
            )
            right_edge_dim = bool(
                right_boundary_type == "data_edge"
                and _edge_is_persistently_dim(
                    raw_corrected,
                    corrected_residual,
                    detection_mask,
                    strongly_dim,
                    times,
                    side="right",
                    detection_threshold=detection_threshold,
                    max_gap_days=smooth_window,
                )
            )
            # A one-sided envelope is an ongoing event only when the light
            # curve is demonstrably still dim at that actual data boundary.
            left_gap_dim = bool(
                left_boundary_type == "gap"
                and _boundary_dim_supported(
                    raw_corrected,
                    corrected_residual,
                    detection_mask,
                    strongly_dim,
                    times,
                    start,
                    direction="after",
                    detection_threshold=detection_threshold,
                    max_gap_days=smooth_window,
                )
            )
            right_gap_dim = bool(
                right_boundary_type == "gap"
                and _boundary_dim_supported(
                    raw_corrected,
                    corrected_residual,
                    detection_mask,
                    strongly_dim,
                    times,
                    stop,
                    direction="before",
                    detection_threshold=detection_threshold,
                    max_gap_days=smooth_window,
                )
            )
            left_gap_baseline = bool(
                left_boundary_type == "gap" and baseline_compatible[start]
            )
            right_gap_baseline = bool(
                right_boundary_type == "gap" and baseline_compatible[stop]
            )
            left_gap_state = (
                "none"
                if left_boundary_type != "gap"
                else (
                    "dim"
                    if left_gap_dim
                    else ("baseline" if left_gap_baseline else "ambiguous")
                )
            )
            right_gap_state = (
                "none"
                if right_boundary_type != "gap"
                else (
                    "dim"
                    if right_gap_dim
                    else ("baseline" if right_gap_baseline else "ambiguous")
                )
            )
            if left_boundary_type == "data_edge" and not left_edge_dim:
                continue
            if right_boundary_type == "data_edge" and not right_edge_dim:
                continue
            if left_boundary_type == "gap" and left_gap_state == "ambiguous":
                continue
            if right_boundary_type == "gap" and right_gap_state == "ambiguous":
                continue
            # The selected observed segment itself needs a confirmed return to
            # unity.  A remote recovery beyond an unbridged gap cannot anchor
            # this segment, and a gap-to-gap segment is not an identified dip.
            if left_recovery is None and right_recovery is None:
                continue
            strong_seed = False
            for seed_start in range(start, stop - 1):
                seed = np.arange(seed_start, seed_start + 3)
                if seed[-1] > stop or times[seed[-1]] - times[seed[0]] > smooth_window:
                    continue
                if int(np.sum(strongly_dim[seed])) >= 2:
                    strong_seed = True
                    break
            local_detection = detection_mask[start : stop + 1]
            detection_components = _mask_components(
                local_detection,
                times[start : stop + 1],
                smooth_window,
            )
            shallow_seed = any(
                component_stop - component_start + 1 >= 5
                for component_start, component_stop in detection_components
            )
            if not strong_seed and not shallow_seed:
                continue
            indices = np.arange(start, stop + 1)
            observed_excess = np.maximum(
                corrected_residual[indices] - recovery_threshold,
                0.0,
            )
            integrated_excess = float(np.sum(time_weights[indices] * observed_excess))
            candidate_peak_time, candidate_peak_depth, candidate_peak_indices = (
                _robust_peak_triplet(
                    times,
                    raw_corrected,
                    corrected_residual,
                    start,
                    stop,
                    smooth_window,
                )
            )
            if len(candidate_peak_indices) != 3 or not np.isfinite(candidate_peak_depth):
                continue
            candidates.append(
                {
                    "start": start,
                    "stop": stop,
                    "left_recovery": left_recovery,
                    "right_recovery": right_recovery,
                    "left_recovery_reference": left_recovery_reference,
                    "right_recovery_reference": right_recovery_reference,
                    "left_edge_dim": left_edge_dim,
                    "right_edge_dim": right_edge_dim,
                    "left_boundary_type": left_boundary_type,
                    "right_boundary_type": right_boundary_type,
                    "left_gap_state": left_gap_state,
                    "right_gap_state": right_gap_state,
                    "integrated_excess": integrated_excess,
                    "peak_time": candidate_peak_time,
                    "peak_depth": candidate_peak_depth,
                    "peak_indices": candidate_peak_indices,
                }
            )
        if not candidates:
            raise RuntimeError("no recovery-anchored dimming bracket with a supported dip seed")
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["peak_depth"],
                candidate["integrated_excess"],
            ),
        )
        event_window_start = int(selected["start"])
        event_window_stop = int(selected["stop"])
        left_recovery = selected["left_recovery"]
        right_recovery = selected["right_recovery"]
        left_recovery_reference = selected["left_recovery_reference"]
        right_recovery_reference = selected["right_recovery_reference"]
        left_baseline_recovered = left_recovery is not None
        right_baseline_recovered = right_recovery is not None
        left_recovery_is_gap_bracket = False
        right_recovery_is_gap_bracket = False
        left_edge_dim_confirmed = bool(selected["left_edge_dim"])
        right_edge_dim_confirmed = bool(selected["right_edge_dim"])
        left_boundary_type = str(selected["left_boundary_type"])
        right_boundary_type = str(selected["right_boundary_type"])
        left_gap_state = str(selected["left_gap_state"])
        right_gap_state = str(selected["right_gap_state"])
        if left_boundary_type == "recovery" and right_boundary_type == "recovery":
            event_window_status = "baseline_bounded"
        elif (
            left_boundary_type == "unconfirmed_recovery"
            and right_boundary_type == "recovery"
        ):
            event_window_status = "left_recovery_unconfirmed"
        elif (
            left_boundary_type == "recovery"
            and right_boundary_type == "unconfirmed_recovery"
        ):
            event_window_status = "right_recovery_unconfirmed"
        elif left_boundary_type == "recovery" and right_boundary_type == "data_edge":
            event_window_status = "ongoing_right_censored"
        elif left_boundary_type == "data_edge" and right_boundary_type == "recovery":
            event_window_status = "ongoing_left_censored"
        elif left_boundary_type == "gap" and right_boundary_type == "recovery":
            event_window_status = "left_gap_censored"
        elif left_boundary_type == "recovery" and right_boundary_type == "gap":
            event_window_status = "right_gap_censored"
        elif left_boundary_type == "gap" and right_boundary_type == "gap":
            event_window_status = "both_gap_censored"
        elif left_boundary_type == "gap" and right_boundary_type == "data_edge":
            event_window_status = "ongoing_right_left_gap_censored"
        elif left_boundary_type == "data_edge" and right_boundary_type == "gap":
            event_window_status = "ongoing_left_right_gap_censored"
        else:
            event_window_status = "unanchored_no_baseline_recovery"
        support_status = "baseline_envelope_seeded"
        event_start = event_window_start
        event_stop = event_window_stop
        event_selection_score = float(selected["peak_depth"])
        event_integrated_excess = float(selected["integrated_excess"])
        deepest = np.asarray(selected["peak_indices"], dtype=int)
        peak_count = int(len(deepest))
        delta_mag = float(selected["peak_depth"])
        if not np.isfinite(delta_mag) or delta_mag <= 0:
            raise RuntimeError("robust peak depth is not positive")
        peak_time = float(selected["peak_time"])
        peak_index = int(np.nanargmin(np.abs(times - peak_time)))
        anchor = peak_index
        tau = float(1.0 - 10.0 ** (-0.4 * delta_mag))
        half_level = float(-2.5 * np.log10(max(1.0 - 0.5 * tau, 1e-9)))

        half_depth = _persistent_half_depth_measurement(
            times,
            raw_corrected,
            corrected_residual,
            half_level=half_level,
            anchor=anchor,
            event_start=event_window_start,
            event_stop=event_window_stop,
            crossing_gap_limit_days=crossing_gap_limit,
            left_event_boundary_type=left_boundary_type,
            right_event_boundary_type=right_boundary_type,
        )
        anchor = int(half_depth["anchor"])
        inside_mask = np.asarray(half_depth["inside_mask"], bool)
        left_side = half_depth["left"]
        right_side = half_depth["right"]
        left_inside = int(half_depth["left_inside"])
        right_inside = int(half_depth["right_inside"])
        n_half = int(half_depth["n_half_depth_epochs"])
        observed_span = float(half_depth["observed_half_depth_span_days"])
        left_crossing_status = str(left_side["status"])
        right_crossing_status = str(right_side["status"])
        left_crossing_source = str(left_side["source"])
        right_crossing_source = str(right_side["source"])
        left_bracketed = left_crossing_status in {"exact", "interval"}
        right_bracketed = right_crossing_status in {"exact", "interval"}
        left_resolved = left_crossing_status == "exact"
        right_resolved = right_crossing_status == "exact"
        left_crossing_gap_censored = left_crossing_status == "interval"
        right_crossing_gap_censored = right_crossing_status == "interval"
        left_time = float(left_side["point"])
        right_time = float(right_side["point"])
        left_lower = float(left_side["lower"])
        left_upper = float(left_side["upper"])
        right_lower = float(right_side["lower"])
        right_upper = float(right_side["upper"])
        internal_gap_count = int(half_depth["internal_gap_count"])
        max_internal_gap = float(half_depth["max_internal_gap_days"])
        window_gaps = np.diff(times[event_window_start : event_window_stop + 1])
        large_window_gaps = window_gaps[window_gaps > crossing_gap_limit]
        event_window_gap_count = int(large_window_gaps.size)
        event_window_max_gap = (
            float(np.max(large_window_gaps)) if large_window_gaps.size else 0.0
        )
        continuity_assumed = bool(event_window_gap_count)
        dimming_complex = _dimming_complex_metrics(
            float(times[event_window_start]),
            float(times[event_window_stop]),
            event_window_status,
            bool(
                left_boundary_type != "recovery"
                or right_boundary_type != "recovery"
            ),
        )

        duration = float(half_depth["half_depth_duration_days"])
        duration_lower = float(half_depth["duration_lower_days"])
        duration_upper = float(half_depth["duration_upper_days"])
        duration_plot = float(half_depth["duration_plot_days"])
        lower_limit = bool(half_depth["duration_is_lower_limit"])
        interval_censored = bool(half_depth["duration_is_interval_censored"])
        duration_status = str(half_depth["duration_status"])

        uncertainty = _estimate_event_uncertainties(
            candidate_id,
            times,
            raw_corrected,
            sigma,
            quiet_scatter_mag=quiet_scatter,
            event_start=event_window_start,
            event_stop=event_window_stop,
            peak_indices=deepest,
            peak_index=peak_index,
            smooth_window_days=smooth_window,
            crossing_gap_limit_days=crossing_gap_limit,
            tau_point=tau,
            delta_mag_point=delta_mag,
            duration_point=duration_plot,
            duration_status=duration_status,
            left_event_boundary_type=left_boundary_type,
            right_event_boundary_type=right_boundary_type,
        )

        result.update(
            {
                "n_good_observations": int(len(analysis)),
                "n_epoch_bins": int(len(epoch)),
                "cadence_days": cadence,
                "gap_cap_days": crossing_gap_limit,
                "smooth_window_days": smooth_window,
                "recovery_window_days": recovery_window,
                "crossing_gap_limit_days": crossing_gap_limit,
                "local_baseline_residual_mag": local_baseline,
                "quiet_scatter_mag": quiet_scatter,
                "peak_initialization_points": peak_count,
                "detection_threshold_mag": detection_threshold,
                "event_selection_score": event_selection_score,
                "event_integrated_excess": event_integrated_excess,
                "event_component_epochs": int(event_stop - event_start + 1),
                "recovery_threshold_mag": recovery_threshold,
                "recovery_flux_threshold": float(10.0 ** (-0.4 * recovery_threshold)),
                "left_baseline_recovered": left_baseline_recovered,
                "right_baseline_recovered": right_baseline_recovered,
                "left_edge_dim_confirmed": left_edge_dim_confirmed,
                "right_edge_dim_confirmed": right_edge_dim_confirmed,
                "left_event_boundary_type": left_boundary_type,
                "right_event_boundary_type": right_boundary_type,
                "left_gap_boundary_state": left_gap_state,
                "right_gap_boundary_state": right_gap_state,
                "event_window_is_lower_limit": bool(
                    left_boundary_type != "recovery"
                    or right_boundary_type != "recovery"
                ),
                "left_recovery_jd": (
                    float(times[left_recovery])
                    if left_recovery is not None
                    else np.nan
                ),
                "right_recovery_jd": (
                    float(times[right_recovery])
                    if right_recovery is not None
                    else np.nan
                ),
                "left_recovery_is_gap_bracket": left_recovery_is_gap_bracket,
                "right_recovery_is_gap_bracket": right_recovery_is_gap_bracket,
                "event_window_start_jd": float(times[event_window_start]),
                "event_window_end_jd": float(times[event_window_stop]),
                "event_window_duration_days": float(
                    times[event_window_stop] - times[event_window_start]
                ),
                "event_window_gap_count": event_window_gap_count,
                "event_window_max_gap_days": event_window_max_gap,
                "event_window_status": event_window_status,
                "event_metrics_schema_version": EVENT_METRICS_SCHEMA_VERSION,
                **dimming_complex,
                "support_status": support_status,
                "fwhm_method_version": FWHM_METHOD_VERSION,
                "peak_jd": peak_time,
                "delta_mag_peak": delta_mag,
                "tau_peak": tau,
                "half_depth_delta_mag": half_level,
                "n_half_depth_epochs": n_half,
                "observed_half_depth_span_days": observed_span,
                "half_depth_recovery_window_epochs": HALF_DEPTH_RECOVERY_WINDOW_EPOCHS,
                "half_depth_recovery_required_epochs": HALF_DEPTH_RECOVERY_REQUIRED_EPOCHS,
                "half_depth_recovery_min_span_days": HALF_DEPTH_RECOVERY_MIN_SPAN_DAYS,
                "left_bracketed": left_bracketed,
                "right_bracketed": right_bracketed,
                "left_crossing_status": left_crossing_status,
                "right_crossing_status": right_crossing_status,
                "left_crossing_source": left_crossing_source,
                "right_crossing_source": right_crossing_source,
                "left_crossing_resolved": left_resolved,
                "right_crossing_resolved": right_resolved,
                "left_crossing_gap_censored": left_crossing_gap_censored,
                "right_crossing_gap_censored": right_crossing_gap_censored,
                "left_crossing_time": left_time,
                "right_crossing_time": right_time,
                "left_crossing_lower_jd": left_lower,
                "left_crossing_upper_jd": left_upper,
                "right_crossing_lower_jd": right_lower,
                "right_crossing_upper_jd": right_upper,
                "half_depth_duration_days": duration,
                "duration_lower_days": duration_lower,
                "duration_upper_days": duration_upper,
                "duration_plot_days": duration_plot,
                "duration_is_lower_limit": lower_limit,
                "duration_is_interval_censored": interval_censored,
                "internal_gap_count": internal_gap_count,
                "max_internal_gap_days": max_internal_gap,
                "half_depth_continuity_assumed": bool(
                    half_depth["half_depth_continuity_assumed"]
                ),
                "event_continuity_assumed": continuity_assumed,
                "duration_status": duration_status,
                **uncertainty,
            }
        )
        if include_trace:
            result["_trace"] = {
                "observations": obs.copy(),
                "epochs": epoch.copy(),
                "anchor_index": anchor,
                "peak_index": peak_index,
                "smoothed_residual": smooth_residual,
                "corrected_smoothed_residual": corrected_residual,
                "inside_mask": inside_mask,
                "event_start": event_start,
                "event_stop": event_stop,
                "detection_mask": detection_mask,
                "recovery_mask": recovery_mask,
                "recovery_support_mask": recovery_support_mask,
                "left_recovery_anchor_mask": left_recovery_anchor_mask,
                "right_recovery_anchor_mask": right_recovery_anchor_mask,
                "baseline_compatible_mask": baseline_compatible,
                "strongly_dim_mask": strongly_dim,
                "left_inside": left_inside,
                "right_inside": right_inside,
                "left_crossing_time": left_time,
                "right_crossing_time": right_time,
                "left_crossing_bounds": (left_lower, left_upper),
                "right_crossing_bounds": (right_lower, right_upper),
                "left_crossing_status": left_crossing_status,
                "right_crossing_status": right_crossing_status,
            }
    except Exception as exc:
        result["measurement_error"] = f"{type(exc).__name__}: {exc}"
        if include_trace:
            result["_trace"] = None
    return result


def measure_half_depth_event(
    candidate_id: str,
    lc_path: Path,
    *,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Measure FWHM inside the shared recovery-anchored dimming window."""
    result: dict[str, Any] = {
        "candidate_id": str(candidate_id),
        "lc_path_measured": str(lc_path),
        "n_good_observations": 0,
        "n_epoch_bins": 0,
        "left_baseline_recovered": False,
        "right_baseline_recovered": False,
        "left_edge_dim_confirmed": False,
        "right_edge_dim_confirmed": False,
        "left_event_boundary_type": "unknown",
        "right_event_boundary_type": "unknown",
        "event_window_is_lower_limit": False,
        "event_window_start_jd": np.nan,
        "event_window_end_jd": np.nan,
        "event_window_duration_days": np.nan,
        "event_window_status": "measurement_failed",
        "dimming_complex_duration_days": np.nan,
        "dimming_complex_duration_lower_days": np.nan,
        "dimming_complex_duration_upper_days": np.nan,
        "dimming_complex_is_lower_limit": False,
        "dimming_complex_status": "measurement_failed",
        "left_crossing_time": np.nan,
        "right_crossing_time": np.nan,
        "duration_status": "measurement_failed",
        "uncertainty_method": "not_computed",
        "uncertainty_draws": 0,
        "event_metrics_schema_version": EVENT_METRICS_SCHEMA_VERSION,
        "dimming_window_method_version": DIMMING_WINDOW_METHOD_VERSION,
        "fwhm_method_version": FWHM_METHOD_VERSION,
        "measurement_error": "",
    }
    try:
        complex_measurement = measure_dimming_complex_window(
            str(candidate_id),
            lc_path,
            config=DEFAULT_DIMMING_WINDOW_CONFIG,
        )
        obs = complex_measurement.observations
        epoch = complex_measurement.epochs
        times = epoch["t"].to_numpy(float)
        raw_corrected = epoch["resid"].to_numpy(float)
        sigma = epoch["sigma"].to_numpy(float)
        corrected_residual = complex_measurement.smoothed_residual
        smooth_sigma = complex_measurement.smoothed_sigma
        cadence = complex_measurement.cadence_days
        smooth_window = complex_measurement.smooth_window_days
        crossing_gap_limit = complex_measurement.crossing_gap_limit_days
        quiet_scatter = complex_measurement.quiet_scatter_mag
        detection_threshold = complex_measurement.detection_threshold_mag
        recovery_threshold = complex_measurement.recovery_threshold_mag
        window = complex_measurement.window
        event_start = window.start_index
        event_stop = window.stop_index
        peak_time = window.peak_jd
        peak_index = int(np.nanargmin(np.abs(times - peak_time)))
        deepest = np.asarray(window.peak_indices, dtype=int)
        delta_mag = window.peak_depth_mag
        tau = float(1.0 - 10.0 ** (-0.4 * delta_mag))
        half_level = float(-2.5 * np.log10(max(1.0 - 0.5 * tau, 1e-9)))

        half_depth = _persistent_half_depth_measurement(
            times,
            raw_corrected,
            corrected_residual,
            half_level=half_level,
            anchor=peak_index,
            event_start=event_start,
            event_stop=event_stop,
            crossing_gap_limit_days=crossing_gap_limit,
            left_event_boundary_type=window.left_boundary_type,
            right_event_boundary_type=window.right_boundary_type,
        )
        anchor = int(half_depth["anchor"])
        left_side = half_depth["left"]
        right_side = half_depth["right"]
        left_crossing_status = str(left_side["status"])
        right_crossing_status = str(right_side["status"])
        duration_status = str(half_depth["duration_status"])
        duration_plot = float(half_depth["duration_plot_days"])
        uncertainty = _estimate_event_uncertainties(
            str(candidate_id),
            times,
            raw_corrected,
            sigma,
            quiet_scatter_mag=quiet_scatter,
            event_start=event_start,
            event_stop=event_stop,
            peak_indices=deepest,
            peak_index=peak_index,
            smooth_window_days=smooth_window,
            crossing_gap_limit_days=crossing_gap_limit,
            tau_point=tau,
            delta_mag_point=delta_mag,
            duration_point=duration_plot,
            duration_status=duration_status,
            left_event_boundary_type=window.left_boundary_type,
            right_event_boundary_type=window.right_boundary_type,
        )
        window_metrics = window.to_metrics(times)
        result.update(
            {
                "n_good_observations": complex_measurement.n_good_observations,
                "n_epoch_bins": int(len(epoch)),
                "cadence_days": cadence,
                "gap_cap_days": crossing_gap_limit,
                "smooth_window_days": smooth_window,
                "recovery_window_days": DEFAULT_DIMMING_WINDOW_CONFIG.recovery_window_days,
                "crossing_gap_limit_days": crossing_gap_limit,
                "local_baseline_residual_mag": 0.0,
                "quiet_scatter_mag": quiet_scatter,
                "detection_threshold_mag": detection_threshold,
                "recovery_threshold_mag": recovery_threshold,
                "recovery_flux_threshold": float(10.0 ** (-0.4 * recovery_threshold)),
                **window_metrics,
                "event_metrics_schema_version": EVENT_METRICS_SCHEMA_VERSION,
                "support_status": "baseline_envelope_seeded",
                "fwhm_method_version": FWHM_METHOD_VERSION,
                "peak_jd": peak_time,
                "delta_mag_peak": delta_mag,
                "tau_peak": tau,
                "half_depth_delta_mag": half_level,
                "n_half_depth_epochs": int(half_depth["n_half_depth_epochs"]),
                "observed_half_depth_span_days": float(
                    half_depth["observed_half_depth_span_days"]
                ),
                "half_depth_recovery_window_epochs": HALF_DEPTH_RECOVERY_WINDOW_EPOCHS,
                "half_depth_recovery_required_epochs": HALF_DEPTH_RECOVERY_REQUIRED_EPOCHS,
                "half_depth_recovery_min_span_days": HALF_DEPTH_RECOVERY_MIN_SPAN_DAYS,
                "left_bracketed": left_crossing_status in {"exact", "interval"},
                "right_bracketed": right_crossing_status in {"exact", "interval"},
                "left_crossing_status": left_crossing_status,
                "right_crossing_status": right_crossing_status,
                "left_crossing_source": str(left_side["source"]),
                "right_crossing_source": str(right_side["source"]),
                "left_crossing_resolved": left_crossing_status == "exact",
                "right_crossing_resolved": right_crossing_status == "exact",
                "left_crossing_gap_censored": left_crossing_status == "interval",
                "right_crossing_gap_censored": right_crossing_status == "interval",
                "left_crossing_time": float(left_side["point"]),
                "right_crossing_time": float(right_side["point"]),
                "left_crossing_lower_jd": float(left_side["lower"]),
                "left_crossing_upper_jd": float(left_side["upper"]),
                "right_crossing_lower_jd": float(right_side["lower"]),
                "right_crossing_upper_jd": float(right_side["upper"]),
                "half_depth_duration_days": float(half_depth["half_depth_duration_days"]),
                "duration_lower_days": float(half_depth["duration_lower_days"]),
                "duration_upper_days": float(half_depth["duration_upper_days"]),
                "duration_plot_days": duration_plot,
                "duration_is_lower_limit": bool(half_depth["duration_is_lower_limit"]),
                "duration_is_interval_censored": bool(
                    half_depth["duration_is_interval_censored"]
                ),
                "internal_gap_count": int(half_depth["internal_gap_count"]),
                "max_internal_gap_days": float(half_depth["max_internal_gap_days"]),
                "half_depth_continuity_assumed": bool(
                    half_depth["half_depth_continuity_assumed"]
                ),
                "duration_status": duration_status,
                **uncertainty,
            }
        )
        if include_trace:
            result["_trace"] = {
                "observations": obs.copy(),
                "epochs": epoch.copy(),
                "anchor_index": anchor,
                "peak_index": peak_index,
                "smoothed_residual": corrected_residual,
                "corrected_smoothed_residual": corrected_residual,
                "inside_mask": np.asarray(half_depth["inside_mask"], bool),
                "event_start": event_start,
                "event_stop": event_stop,
                "detection_mask": complex_measurement.detection_mask,
                "recovery_mask": complex_measurement.recovery_mask,
                "recovery_support_mask": complex_measurement.recovery_support_mask,
                "left_recovery_anchor_mask": complex_measurement.left_recovery_anchor_mask,
                "right_recovery_anchor_mask": complex_measurement.right_recovery_anchor_mask,
                "baseline_compatible_mask": complex_measurement.baseline_compatible_mask,
                "strongly_dim_mask": complex_measurement.strongly_dim_mask,
                "left_inside": int(half_depth["left_inside"]),
                "right_inside": int(half_depth["right_inside"]),
                "left_crossing_time": float(left_side["point"]),
                "right_crossing_time": float(right_side["point"]),
                "left_crossing_bounds": (
                    float(left_side["lower"]),
                    float(left_side["upper"]),
                ),
                "right_crossing_bounds": (
                    float(right_side["lower"]),
                    float(right_side["upper"]),
                ),
                "left_crossing_status": left_crossing_status,
                "right_crossing_status": right_crossing_status,
            }
    except Exception as exc:
        result["measurement_error"] = f"{type(exc).__name__}: {exc}"
        if include_trace:
            result["_trace"] = None
    return result


def measure_all_half_depth_events(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in candidates[["candidate_id", "lc_path"]].itertuples(index=False):
        rows.append(measure_half_depth_event(str(row.candidate_id), Path(str(row.lc_path)).expanduser()))
    return pd.DataFrame(rows)


def _relative_flux_from_residual(delta_mag: np.ndarray | pd.Series | float) -> np.ndarray:
    """Convert baseline-relative magnitude residuals to relative flux."""
    return np.power(10.0, -0.4 * np.asarray(delta_mag, dtype=float))


def _fractional_depth_to_delta_mag(
    fractional_depth: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Convert a fractional flux depth to a positive magnitude depth."""
    depth = np.asarray(fractional_depth, dtype=float)
    return -2.5 * np.log10(1.0 - depth)


def _delta_mag_to_fractional_depth(
    delta_mag: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Convert a positive magnitude depth to a fractional flux depth."""
    return 1.0 - np.power(10.0, -0.4 * np.asarray(delta_mag, dtype=float))


def _event_zoom_bounds(measurement: dict[str, Any], trace: dict[str, Any]) -> tuple[float, float]:
    epochs = trace["epochs"]
    return dimming_complex_zoom_bounds(
        epochs["t"].to_numpy(float),
        start_jd=float(measurement["event_window_start_jd"]),
        end_jd=float(measurement["event_window_end_jd"]),
        peak_jd=float(measurement["peak_jd"]),
        cadence_days=float(measurement["cadence_days"]),
    )


def _flux_limits(flux: np.ndarray, peak_flux: float) -> tuple[float, float]:
    finite = np.asarray(flux, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lower = float(np.nanpercentile(finite, 1.0))
        upper = float(np.nanpercentile(finite, 99.0))
    else:
        lower, upper = peak_flux, 1.0
    lower = min(lower, peak_flux)
    upper = max(upper, 1.0)
    span = max(upper - lower, 0.08)
    return max(0.0, lower - 0.08 * span), upper + 0.08 * span


def _plot_half_depth_candidate(
    full_ax: plt.Axes,
    zoom_ax: plt.Axes,
    metrics_ax: plt.Axes,
    candidate: pd.Series,
    measurement: dict[str, Any],
    trace: dict[str, Any],
    pipeline_dip_runs: pd.DataFrame | None = None,
) -> None:
    observations = trace["observations"]
    epochs = trace["epochs"]
    obs_times = observations["t"].to_numpy(float)
    local_baseline = float(measurement["local_baseline_residual_mag"])
    obs_flux = _relative_flux_from_residual(observations["resid"].to_numpy(float) - local_baseline)
    epoch_times = epochs["t"].to_numpy(float)
    epoch_flux = _relative_flux_from_residual(epochs["resid"].to_numpy(float) - local_baseline)
    smooth_flux = _relative_flux_from_residual(trace["corrected_smoothed_residual"])
    time_offset = 2_458_000.0
    obs_x = obs_times - time_offset
    epoch_x = epoch_times - time_offset

    peak_time = float(measurement["peak_jd"])
    peak_x = peak_time - time_offset
    peak_flux = float(1.0 - measurement["tau_peak"])
    half_flux = float(1.0 - 0.5 * measurement["tau_peak"])
    zoom_left, zoom_right = _event_zoom_bounds(measurement, trace)
    zoom_left_x = zoom_left - time_offset
    zoom_right_x = zoom_right - time_offset
    event_left = float(measurement["event_window_start_jd"])
    event_right = float(measurement["event_window_end_jd"])
    event_left_x = event_left - time_offset
    event_right_x = event_right - time_offset

    full_ax.scatter(
        obs_x,
        obs_flux,
        s=3,
        color="#8a8a8a",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    full_ax.scatter(epoch_x, epoch_flux, s=8, color="#111111", linewidths=0, zorder=2)
    gap_cap = float(measurement["crossing_gap_limit_days"])
    full_starts = np.r_[0, np.flatnonzero(np.diff(epoch_times) > gap_cap) + 1]
    full_ends = np.r_[full_starts[1:], len(epoch_times)]
    for start, end in zip(full_starts, full_ends):
        full_ax.plot(
            epoch_x[start:end],
            smooth_flux[start:end],
            color="#009e73",
            linewidth=0.85,
            alpha=0.9,
            zorder=2.5,
        )
    _draw_pipeline_dip_run_overlay(
        full_ax,
        pipeline_dip_runs,
        observation_jds=obs_times,
        observation_flux=obs_flux,
        time_offset=time_offset,
    )
    full_ax.axhline(1.0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
    full_ax.axvspan(zoom_left_x, zoom_right_x, color="#56b4e9", alpha=0.18, linewidth=0)
    full_ax.axvline(peak_x, color="#e69f00", linewidth=1.0, zorder=3)
    full_ax.set_xlim(float(np.nanmin(obs_x)), float(np.nanmax(obs_x)))
    full_ax.set_ylim(*_flux_limits(obs_flux, peak_flux))
    display_id = str(candidate.get("asas_sn_id") or candidate["candidate_id"])
    full_ax.set_title(f"ASAS-SN {display_id}  |  {candidate['plot_class']}", loc="left", fontsize=9.3)
    full_ax.set_ylabel("Relative flux", fontsize=8.5)

    in_zoom = (obs_times >= zoom_left) & (obs_times <= zoom_right)
    zoom_ax.axvspan(
        event_left_x,
        event_right_x,
        color="#999999",
        alpha=0.15,
        linewidth=0,
        zorder=0,
    )
    zoom_ax.scatter(
        obs_x[in_zoom],
        obs_flux[in_zoom],
        s=8,
        color="#8a8a8a",
        alpha=0.48,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    epoch_in_zoom = (epoch_times >= zoom_left) & (epoch_times <= zoom_right)
    zoom_indices = np.flatnonzero(epoch_in_zoom)
    if zoom_indices.size:
        starts = np.r_[0, np.flatnonzero(np.diff(epoch_times[zoom_indices]) > gap_cap) + 1]
        ends = np.r_[starts[1:], len(zoom_indices)]
        for start, end in zip(starts, ends):
            selected = zoom_indices[start:end]
            zoom_ax.plot(
                epoch_x[selected],
                smooth_flux[selected],
                color="#009e73",
                linewidth=0.95,
                alpha=0.9,
                zorder=2,
            )
    zoom_ax.scatter(
        epoch_x[epoch_in_zoom],
        epoch_flux[epoch_in_zoom],
        s=22,
        facecolor="#111111",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    _draw_pipeline_dip_run_overlay(
        zoom_ax,
        pipeline_dip_runs,
        observation_jds=obs_times,
        observation_flux=obs_flux,
        time_offset=time_offset,
        observation_mask=in_zoom,
    )

    left_inside = int(trace["left_inside"])
    right_inside = int(trace["right_inside"])
    selected_indices = np.flatnonzero(trace["inside_mask"])
    selected_indices = selected_indices[
        (selected_indices >= left_inside) & (selected_indices <= right_inside)
    ]
    zoom_ax.scatter(
        epoch_x[selected_indices],
        epoch_flux[selected_indices],
        s=34,
        facecolor="#009eaa",
        edgecolor="#111111",
        linewidth=0.55,
        zorder=4,
    )
    zoom_ax.scatter(
        peak_x,
        peak_flux,
        marker="*",
        s=105,
        facecolor="#e69f00",
        edgecolor="#111111",
        linewidth=0.65,
        zorder=6,
    )
    tau_err_minus = _finite_number(measurement.get("tau_peak_mc_err_minus"))
    tau_err_plus = _finite_number(measurement.get("tau_peak_mc_err_plus"))
    if np.isfinite(tau_err_minus) and np.isfinite(tau_err_plus):
        zoom_ax.errorbar(
            [peak_x],
            [peak_flux],
            yerr=np.array([[tau_err_plus], [tau_err_minus]]),
            fmt="none",
            ecolor="#cc4c02",
            elinewidth=0.9,
            capsize=2.0,
            zorder=5.5,
        )
    zoom_ax.axhline(1.0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
    recovery_flux = float(measurement["recovery_flux_threshold"])
    zoom_ax.axhline(
        recovery_flux,
        color="#009e73",
        linestyle=(0, (2, 2)),
        linewidth=0.7,
        alpha=0.8,
        zorder=1,
    )
    if str(measurement["left_event_boundary_type"]) == "recovery":
        zoom_ax.axvline(event_left_x, color="#6a3d9a", linewidth=1.0, alpha=0.95, zorder=2)
    elif str(measurement["left_event_boundary_type"]) == "unconfirmed_recovery":
        zoom_ax.axvline(
            event_left_x,
            color="#b2182b",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            alpha=0.95,
            zorder=2,
        )
    elif str(measurement["left_event_boundary_type"]) == "gap":
        zoom_ax.axvline(
            event_left_x,
            color="#b2182b",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            alpha=0.9,
            zorder=2,
        )
    if str(measurement["right_event_boundary_type"]) == "recovery":
        zoom_ax.axvline(event_right_x, color="#6a3d9a", linewidth=1.0, alpha=0.95, zorder=2)
    elif str(measurement["right_event_boundary_type"]) == "unconfirmed_recovery":
        zoom_ax.axvline(
            event_right_x,
            color="#d55e00",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            alpha=0.95,
            zorder=2,
        )
    elif str(measurement["right_event_boundary_type"]) == "gap":
        zoom_ax.axvline(
            event_right_x,
            color="#d55e00",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            alpha=0.9,
            zorder=2,
        )
    zoom_ax.axhline(half_flux, color="#0072b2", linestyle=(0, (5, 2)), linewidth=1.05, zorder=2)

    left_crossing = _finite_number(trace["left_crossing_time"])
    right_crossing = _finite_number(trace["right_crossing_time"])
    left_crossing_status = str(measurement.get("left_crossing_status", "censored"))
    right_crossing_status = str(measurement.get("right_crossing_status", "censored"))
    left_bounds = trace["left_crossing_bounds"]
    right_bounds = trace["right_crossing_bounds"]
    lower_limit = bool(measurement["duration_is_lower_limit"])
    if left_crossing_status == "exact" and np.isfinite(left_crossing):
        interval_left = left_crossing
    elif left_crossing_status == "interval" and np.isfinite(left_bounds[1]):
        interval_left = float(left_bounds[1])
    else:
        interval_left = float(epoch_times[left_inside])
    if right_crossing_status == "exact" and np.isfinite(right_crossing):
        interval_right = right_crossing
    elif right_crossing_status == "interval" and np.isfinite(right_bounds[0]):
        interval_right = float(right_bounds[0])
    else:
        interval_right = float(epoch_times[right_inside])
    if interval_right <= interval_left:
        proxy_width = float(measurement["duration_plot_days"])
        interval_left = peak_time - 0.5 * proxy_width
        interval_right = peak_time + 0.5 * proxy_width

    zoom_ax.axvspan(
        interval_left - time_offset,
        interval_right - time_offset,
        color="#56b4e9",
        alpha=0.2,
        linewidth=0,
        zorder=0,
    )
    for crossing_status, (bound_left, bound_right) in (
        (left_crossing_status, left_bounds),
        (right_crossing_status, right_bounds),
    ):
        if (
            crossing_status == "interval"
            and np.isfinite(bound_left)
            and np.isfinite(bound_right)
        ):
            zoom_ax.axvspan(
                bound_left - time_offset,
                bound_right - time_offset,
                color="#d55e00",
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
    zoom_ax.annotate(
        "",
        xy=(interval_right - time_offset, half_flux),
        xytext=(interval_left - time_offset, half_flux),
        arrowprops={"arrowstyle": "<->", "color": "#0072b2", "linewidth": 1.35},
        zorder=5,
    )
    outward = 0.075 * (zoom_right - zoom_left)
    if lower_limit and left_crossing_status == "censored":
        zoom_ax.annotate(
            "",
            xy=(interval_left - outward - time_offset, half_flux),
            xytext=(interval_left - time_offset, half_flux),
            arrowprops={"arrowstyle": "-|>", "color": "#0072b2", "linewidth": 1.35},
            zorder=5,
        )
    if lower_limit and right_crossing_status == "censored":
        zoom_ax.annotate(
            "",
            xy=(interval_right + outward - time_offset, half_flux),
            xytext=(interval_right - time_offset, half_flux),
            arrowprops={"arrowstyle": "-|>", "color": "#0072b2", "linewidth": 1.35},
            zorder=5,
        )

    zoom_ax.annotate(
        "",
        xy=(peak_x, 1.0),
        xytext=(peak_x, peak_flux),
        arrowprops={"arrowstyle": "<->", "color": "#cc4c02", "linewidth": 1.35},
        zorder=5,
    )
    delta_label = rf"$\delta={float(measurement['tau_peak']):.3f}$"
    if np.isfinite(tau_err_minus) and np.isfinite(tau_err_plus):
        delta_label = (
            rf"$\delta={float(measurement['tau_peak']):.3f}"
            rf"_{{-{tau_err_minus:.3f}}}^{{+{tau_err_plus:.3f}}}$"
        )
    zoom_ax.annotate(
        delta_label,
        xy=(peak_x, peak_flux + 0.72 * (1.0 - peak_flux)),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.2,
        color="#8c2d04",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )

    if bool(measurement["duration_is_interval_censored"]):
        duration_expression = (
            rf"\tau_{{\rm FWHM}}\in[{float(measurement['duration_lower_days']):.2f},"
            rf"{float(measurement['duration_upper_days']):.2f}]\,\rm d"
        )
    elif lower_limit and np.isfinite(_finite_number(measurement["duration_plot_days"])):
        duration_expression = (
            rf"\tau_{{\rm FWHM}}\geq{float(measurement['duration_plot_days']):.2f}\,\rm d"
        )
    elif lower_limit:
        duration_expression = r"\tau_{\rm FWHM}\;\mathrm{unconstrained}"
    else:
        duration_expression = rf"\tau_{{\rm FWHM}}={float(measurement['duration_plot_days']):.2f}\,\rm d"
    duration_mc_p16 = _finite_number(measurement.get("duration_mc_p16"))
    duration_mc_p84 = _finite_number(measurement.get("duration_mc_p84"))
    if (
        measurement.get("duration_mc_reporting_status") == "reported_resolved"
        and np.isfinite(duration_mc_p16)
        and np.isfinite(duration_mc_p84)
    ):
        duration_expression += (
            rf"\;[\mathrm{{MC68}}:\,{duration_mc_p16:.2f},{duration_mc_p84:.2f}\,\rm d]"
        )
    event_status_labels = {
        "baseline_bounded": "recovered on both sides",
        "ongoing_right_censored": "left recovered\nongoing at final epoch",
        "ongoing_left_censored": "right recovered\nstarts before first epoch",
        "left_recovery_unconfirmed": "right recovered\nleft edge unconfirmed",
        "right_recovery_unconfirmed": "left recovered\nright edge unconfirmed",
        "left_gap_censored": "right recovered\nstart hidden by gap",
        "right_gap_censored": "left recovered\nend hidden by gap",
        "both_gap_censored": "both edges hidden by gaps",
        "ongoing_right_left_gap_censored": "start hidden by gap\nongoing at final epoch",
        "ongoing_left_right_gap_censored": "starts before data\nend hidden by gap",
        "unanchored_no_baseline_recovery": "no confirmed recovery",
    }
    event_status_text = event_status_labels.get(
        str(measurement["event_window_status"]),
        str(measurement["event_window_status"]).replace("_", " "),
    )
    gap_note = ""
    if bool(measurement["event_continuity_assumed"]):
        gap_note = f"\ngaps crossed: {int(measurement['event_window_gap_count'])}"
    complex_duration = float(measurement["dimming_complex_duration_days"])
    if bool(measurement["dimming_complex_is_lower_limit"]):
        complex_expression = (
            rf"$T_{{\rm complex}}\geq{complex_duration:.2f}\,\rm d$"
        )
        complex_qualifier = str(measurement["dimming_complex_status"]).replace("_", " ")
    else:
        complex_expression = rf"$T_{{\rm complex}}={complex_duration:.2f}\,\rm d$"
        complex_qualifier = "recovery bounded"

    metrics_ax.set_facecolor("#fafafa")
    metrics_ax.axis("off")
    metrics_ax.text(
        0.02,
        0.98,
        "Individual dip",
        transform=metrics_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        weight="bold",
        color="#005a8d",
    )
    metrics_ax.text(
        0.02,
        0.83,
        rf"$\delta={float(measurement['tau_peak']):.3f}$"
        + "\n"
        + f"${duration_expression}$"
        + "\n"
        + str(measurement["duration_status"]).replace("_", " "),
        transform=metrics_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        linespacing=1.25,
    )
    overlay = _pipeline_run_overlay_metrics(
        pipeline_dip_runs,
        event_start_jd=event_left,
        event_end_jd=event_right,
        peak_jd=peak_time,
    )
    if pipeline_dip_runs is not None:
        run_count = int(overlay["pipeline_dip_run_count"])
        overlap_count = int(overlay["pipeline_dip_runs_overlapping_complex"])
        peak_relation = (
            "peak inside a run"
            if bool(overlay["atlas_peak_inside_pipeline_dip_run"])
            else "peak outside runs"
        )
        metrics_ax.text(
            0.02,
            0.51,
            (
                f"Pipeline runs: {run_count}; overlap: {overlap_count}\n"
                f"{peak_relation}"
            ),
            transform=metrics_ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            linespacing=1.08,
            color=PIPELINE_DIP_RUN_EDGE_COLOR,
        )
        complex_title_y = 0.37
        complex_text_y = 0.25
    else:
        complex_title_y = 0.50
        complex_text_y = 0.36
    metrics_ax.text(
        0.02,
        complex_title_y,
        "Dimming complex",
        transform=metrics_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        weight="bold",
        color="#5e3c99",
    )
    metrics_ax.text(
        0.02,
        complex_text_y,
        f"{complex_expression}\n{complex_qualifier}\n{event_status_text}{gap_note}",
        transform=metrics_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        linespacing=1.15,
    )
    metrics_ax.text(
        0.02,
        0.59,
        rf"$t_{{\rm peak}}-2{{,}}458{{,}}000={peak_x:.2f}$",
        transform=metrics_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#444444",
    )
    zoom_flux = obs_flux[in_zoom] if np.any(in_zoom) else obs_flux
    zoom_ax.set_xlim(zoom_left_x, zoom_right_x)
    zoom_ax.set_ylim(*_flux_limits(zoom_flux, peak_flux))

    for ax in (full_ax, zoom_ax):
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=7.5)
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)


def plot_half_depth_diagnostic_atlas(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    rows_per_page: int = 4,
    pipeline_dip_runs: pd.DataFrame | None = None,
    atlas_stem: str = "all_dippers_half_depth_diagnostic_atlas",
    compare_cached_metrics: bool = True,
) -> dict[str, Any]:
    """Write a paginated visual audit of every plotted half-depth measurement."""
    if Path(atlas_stem).name != atlas_stem or Path(atlas_stem).suffix:
        raise ValueError("atlas_stem must be a filename stem without a directory or suffix")
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("candidate_id").reset_index(drop=True)
    atlas_path = output_dir / f"{atlas_stem}.pdf"
    preview_path = output_dir / f"{atlas_stem}_page01.png"
    metrics_path = output_dir / f"{atlas_stem}_metrics.csv"
    audit_rows: list[dict[str, Any]] = []
    n_pages = int(np.ceil(len(ordered) / rows_per_page))
    overlay_enabled = pipeline_dip_runs is not None
    overlay_source = (
        str(pipeline_dip_runs.attrs.get("source_path", ""))
        if pipeline_dip_runs is not None
        else ""
    )
    if pipeline_dip_runs is not None:
        empty_runs = pipeline_dip_runs.iloc[0:0].copy()
        run_groups = {
            str(candidate_id): group.copy()
            for candidate_id, group in pipeline_dip_runs.groupby(
                "candidate_id", sort=False
            )
        }
    else:
        empty_runs = None
        run_groups = {}
    legend_handles = [
        Line2D([0], [0], marker=".", linestyle="none", color="#8a8a8a", label="Individual observations"),
        Line2D([0], [0], marker="o", linestyle="none", color="#111111", markersize=4, label="Nightly medians"),
        Line2D([0], [0], linestyle="-", color="#009e73", linewidth=1.1, label="Local-median dip profile"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#009eaa", markeredgecolor="#111111", markersize=5, label="At/above half depth"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#e69f00", markeredgecolor="#111111", markersize=8, label="Selected peak"),
        Line2D([0], [0], color="#0072b2", linestyle=(0, (5, 2)), label="Half-depth level"),
        Line2D([0], [0], color="#009e73", linestyle=(0, (2, 2)), label="Baseline-recovery threshold"),
        Line2D([0], [0], marker="|", linestyle="none", color="#6a3d9a", markersize=10, markeredgewidth=1.4, label="Confirmed recovery epoch"),
        Line2D([0], [0], color="#b2182b", linestyle=(0, (3, 2)), linewidth=1.0, label="Unconfirmed recovery boundary"),
        Patch(facecolor="#999999", edgecolor="none", alpha=0.24, label="Outer recovery bracket (gaps unobserved)"),
        Patch(facecolor="#56b4e9", edgecolor="none", alpha=0.25, label=r"FWHM core"),
        Patch(facecolor="#d55e00", edgecolor="none", alpha=0.18, label="Sampling crossing interval"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color="#cc4c02",
            markerfacecolor="white",
            markersize=4,
            linewidth=0.9,
            label="Conditional MC 68% error",
        ),
    ]
    if overlay_enabled:
        legend_handles.extend(
            [
                Patch(
                    facecolor=PIPELINE_DIP_RUN_COLOR,
                    edgecolor=PIPELINE_DIP_RUN_EDGE_COLOR,
                    alpha=0.18,
                    label="Production dip-run interval",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor=PIPELINE_DIP_RUN_EDGE_COLOR,
                    markersize=5,
                    label="Pipeline-triggered observation",
                ),
            ]
        )

    with PdfPages(
        atlas_path,
        metadata={
            "Title": (
                "All-dipper ASAS-SN half-depth and production dip-run comparison atlas"
                if overlay_enabled
                else "All-dipper ASAS-SN FWHM and dimming-complex diagnostic atlas"
            ),
            "Creator": "MALCA",
        },
    ) as atlas:
        for page_index in range(n_pages):
            fig, axes = plt.subplots(
                rows_per_page,
                3,
                figsize=(15.5, 10.5),
                squeeze=False,
                gridspec_kw={"width_ratios": [1.0, 1.08, 0.42]},
            )
            start = page_index * rows_per_page
            stop = min(start + rows_per_page, len(ordered))
            for row_index, (_, candidate) in enumerate(ordered.iloc[start:stop].iterrows()):
                candidate_id = str(candidate["candidate_id"])
                candidate_runs = (
                    run_groups.get(candidate_id, empty_runs)
                    if overlay_enabled
                    else None
                )
                measurement = measure_half_depth_event(
                    candidate_id,
                    Path(str(candidate["lc_path"])).expanduser(),
                    include_trace=True,
                )
                trace = measurement.pop("_trace", None)
                if compare_cached_metrics:
                    cached_tau = _finite_number(candidate.get("tau_peak"))
                    cached_duration = _finite_number(candidate.get("duration_plot_days"))
                    recomputed_tau = _finite_number(measurement["tau_peak"])
                    recomputed_duration = _finite_number(measurement["duration_plot_days"])
                    cached_status = str(candidate.get("duration_status", ""))
                    recomputed_status = str(measurement["duration_status"])
                    tau_match = bool(
                        (not np.isfinite(cached_tau) and not np.isfinite(recomputed_tau))
                        or np.isclose(cached_tau, recomputed_tau, rtol=0, atol=1e-10)
                    )
                    duration_match = bool(
                        (
                            not np.isfinite(cached_duration)
                            and not np.isfinite(recomputed_duration)
                        )
                        or np.isclose(
                            cached_duration, recomputed_duration, rtol=0, atol=1e-8
                        )
                    )
                    bound_matches = []
                    for field in (
                        "duration_lower_days",
                        "duration_upper_days",
                        "dimming_complex_duration_lower_days",
                        "dimming_complex_duration_upper_days",
                    ):
                        cached_bound = _finite_number(candidate.get(field))
                        recomputed_bound = _finite_number(measurement[field])
                        bound_matches.append(
                            bool(
                                (
                                    not np.isfinite(cached_bound)
                                    and not np.isfinite(recomputed_bound)
                                )
                                or np.isclose(
                                    cached_bound,
                                    recomputed_bound,
                                    rtol=0,
                                    atol=1e-8,
                                )
                            )
                        )
                    metric_match = bool(
                        tau_match
                        and duration_match
                        and all(bound_matches)
                        and cached_status == recomputed_status
                        and str(candidate.get("fwhm_method_version", ""))
                        == FWHM_METHOD_VERSION
                        and str(candidate.get("event_metrics_schema_version", ""))
                        == EVENT_METRICS_SCHEMA_VERSION
                        and str(candidate.get("dimming_window_method_version", ""))
                        == DIMMING_WINDOW_METHOD_VERSION
                    )
                    metric_audit_mode = "compared_with_cached_measurement"
                else:
                    metric_match = True
                    metric_audit_mode = "fresh_measurement"
                overlay_metrics = _pipeline_run_overlay_metrics(
                    candidate_runs,
                    event_start_jd=_finite_number(
                        measurement.get("event_window_start_jd")
                    ),
                    event_end_jd=_finite_number(
                        measurement.get("event_window_end_jd")
                    ),
                    peak_jd=_finite_number(measurement.get("peak_jd")),
                )
                audit_rows.append(
                    {
                        **measurement,
                        **overlay_metrics,
                        "dip_run_overlay_enabled": overlay_enabled,
                        "dip_run_overlay_source": overlay_source,
                        "metric_audit_mode": metric_audit_mode,
                        "matches_plotted_metric": metric_match,
                    }
                )
                if trace is None:
                    for ax in axes[row_index]:
                        ax.axis("off")
                    row_box = axes[row_index, 1].get_position()
                    run_count = int(overlay_metrics["pipeline_dip_run_count"])
                    fig.text(
                        0.5,
                        0.5 * (row_box.y0 + row_box.y1),
                        (
                            f"{candidate_id}\n"
                            "Atlas half-depth estimator unavailable: "
                            f"{measurement['measurement_error']}\n"
                            f"Historical production replay: {run_count} dip runs. "
                            "Candidate retained in the cohort."
                        ),
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        linespacing=1.35,
                        color="#444444",
                    )
                    continue
                _plot_half_depth_candidate(
                    axes[row_index, 0],
                    axes[row_index, 1],
                    axes[row_index, 2],
                    candidate,
                    measurement,
                    trace,
                    pipeline_dip_runs=candidate_runs,
                )
                if compare_cached_metrics and not metric_match:
                    axes[row_index, 2].text(
                        0.02,
                        0.68,
                        "WARNING: recomputed metric differs from plotted cache",
                        transform=axes[row_index, 2].transAxes,
                        ha="left",
                        va="top",
                        color="#b2182b",
                        fontsize=7.0,
                        weight="bold",
                        wrap=True,
                    )

            for row_index in range(stop - start, rows_per_page):
                for ax in axes[row_index]:
                    ax.axis("off")
            fig.text(
                0.205,
                0.895,
                "Full locally normalized light curve",
                ha="center",
                va="bottom",
                fontsize=10.5,
                weight="bold",
            )
            fig.text(
                0.595,
                0.895,
                "Selected event and FWHM measurement",
                ha="center",
                va="bottom",
                fontsize=10.5,
                weight="bold",
            )
            fig.text(
                0.885,
                0.895,
                "Measurements",
                ha="center",
                va="bottom",
                fontsize=10.5,
                weight="bold",
            )
            for ax in axes[-1, :2]:
                if ax.axison:
                    ax.set_xlabel("JD - 2,458,000 [days]", fontsize=8.5)
            fig.suptitle(
                (
                    "ASAS-SN half-depth and production dip-run comparison"
                    if overlay_enabled
                    else "ASAS-SN dip measurements"
                )
                + f"  |  page {page_index + 1} of {n_pages}",
                fontsize=13,
                y=0.995,
            )
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.962),
                ncol=8 if overlay_enabled else 7,
                fontsize=6.8 if overlay_enabled else 7.4,
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="black",
            )
            fig.text(
                0.5,
                0.012,
                (
                    "Magenta: historical production dip-run intervals and their triggered observations. "
                    if overlay_enabled
                    else ""
                )
                + "Grey: recovery-bracketed dimming complex. Blue: individual-dip FWHM; finite intervals are bounds and open event edges are lower limits.",
                ha="center",
                va="bottom",
                fontsize=7.8 if overlay_enabled else 8.2,
            )
            fig.subplots_adjust(
                left=0.055,
                right=0.985,
                top=0.85,
                bottom=0.055,
                hspace=0.5,
                wspace=0.18,
            )
            atlas.savefig(fig, dpi=180)
            if page_index == 0:
                fig.savefig(preview_path, dpi=190, bbox_inches="tight")
            plt.close(fig)

    pd.DataFrame(audit_rows).to_csv(metrics_path, index=False)
    return {
        "n_candidates": int(len(ordered)),
        "n_pages": n_pages,
        "n_metrics_matching_plot": int(sum(row["matches_plotted_metric"] for row in audit_rows)),
        "n_candidates_with_pipeline_dip_runs": int(
            sum(row["pipeline_dip_run_count"] > 0 for row in audit_rows)
        ),
        "n_pipeline_dip_runs": int(
            sum(row["pipeline_dip_run_count"] for row in audit_rows)
        ),
        "n_atlas_peaks_inside_pipeline_dip_run": int(
            sum(row["atlas_peak_inside_pipeline_dip_run"] for row in audit_rows)
        ),
        "n_candidates_with_run_overlapping_complex": int(
            sum(row["pipeline_dip_runs_overlapping_complex"] > 0 for row in audit_rows)
        ),
        "dip_run_overlay_source": overlay_source,
        "pdf": str(atlas_path),
        "preview": str(preview_path),
        "metrics": str(metrics_path),
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=6, width=1.0, labelsize=11)
    ax.tick_params(which="minor", length=3, width=0.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)


def _class_handles(*, white_face: bool = False, neutral_edges: bool = False) -> list[Line2D]:
    handles = []
    for name in CLASS_ORDER:
        style = CLASS_STYLE[name]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor="white" if white_face else style["color"],
                markeredgecolor=(
                    "#222222"
                    if neutral_edges
                    else style["color"] if white_face else "#222222"
                ),
                markeredgewidth=1.2,
                markersize=7,
                label=style["label"],
            )
        )
    return handles


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight: bool = True,
) -> None:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(pdf, **save_kwargs)
    fig.savefig(png, dpi=260, **save_kwargs)
    plt.close(fig)


def plot_distance_histogram(frame: pd.DataFrame, output_dir: Path, max_pc: float) -> dict[str, int]:
    estimators = [
        ("Inverse parallax", frame["inverse_parallax_distance_pc"], "#27a7b8", "-"),
        ("Bailer-Jones photogeometric", frame["bj_r_med_photogeo"], "#70466e", "-"),
        ("StarHorse", frame["starhorse_distance_pc"], "#e88973", "-."),
        ("Gaia GSP-Phot", frame["distance_gspphot"], "#456a9a", "--"),
    ]
    bin_width = 500.0
    bins = np.arange(0.0, max_pc + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(8.1, 5.5), layout="constrained")
    coverage: dict[str, int] = {}
    overflow_lines = []
    for label, values, color, linestyle in estimators:
        data = pd.to_numeric(values, errors="coerce").to_numpy(float)
        data = data[np.isfinite(data) & (data > 0)]
        coverage[label] = int(data.size)
        in_range = data[data <= max_pc]
        overflow = int(np.sum(data > max_pc))
        ax.hist(
            in_range,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
            label=f"{label} (N={data.size})",
        )
        if data.size:
            median = float(np.median(data))
            if median <= max_pc:
                ax.axvline(median, color=color, linestyle=":", linewidth=1.15, alpha=0.9)
        if overflow:
            overflow_lines.append(f"{label}: {overflow}")
    ax.set_xlim(0, max_pc)
    ax.set_xlabel("Distance [pc]", fontsize=17)
    ax.set_ylabel("Number of dippers", fontsize=17)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.text(
        0.02,
        0.97,
        "All 87 reviewed dippers\nBailer-Jones: EDR3 photogeometric posterior median.",
        transform=ax.transAxes,
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    if overflow_lines:
        ax.text(
            0.98,
            0.03,
            f"Beyond {max_pc / 1000:g} kpc: " + "; ".join(overflow_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
    _style_axis(ax)
    _save_figure(fig, output_dir, "all_dippers_distance_comparison")
    return coverage


def plot_2mass_wise(frame: pd.DataFrame, output_dir: Path) -> int:
    plotted = frame.dropna(subset=["h_ks", "w1_w2_color"]).copy()
    values = pd.to_numeric(plotted["allwise_mep_w1_intrinsic_scatter"], errors="coerce")
    finite = values[np.isfinite(values)]
    vmax = max(0.05, float(np.nanpercentile(finite, 95))) if not finite.empty else 0.1
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="constrained")
    with_errors = plotted[["h_ks_err", "w1_w2_err"]].notna().all(axis=1)
    if with_errors.any():
        err = plotted.loc[with_errors]
        ax.errorbar(
            err["w1_w2_color"],
            err["h_ks"],
            xerr=err["w1_w2_err"],
            yerr=err["h_ks_err"],
            fmt="none",
            ecolor="#777777",
            elinewidth=0.55,
            alpha=0.35,
            zorder=1,
        )
    for class_name in CLASS_ORDER:
        subset = plotted.loc[plotted["plot_class"] == class_name]
        if subset.empty:
            continue
        metric = pd.to_numeric(subset["allwise_mep_w1_intrinsic_scatter"], errors="coerce")
        has_metric = np.isfinite(metric)
        if has_metric.any():
            ax.scatter(
                subset.loc[has_metric, "w1_w2_color"],
                subset.loc[has_metric, "h_ks"],
                c=metric.loc[has_metric],
                cmap=cmap,
                norm=norm,
                marker=CLASS_STYLE[class_name]["marker"],
                s=54,
                edgecolor="#202020",
                linewidth=0.8,
                zorder=3,
            )
        if (~has_metric).any():
            ax.scatter(
                subset.loc[~has_metric, "w1_w2_color"],
                subset.loc[~has_metric, "h_ks"],
                facecolor="none",
                edgecolor="#555555",
                marker=CLASS_STYLE[class_name]["marker"],
                s=54,
                linewidth=1.0,
                zorder=3,
            )

    ax.axvline(YSO_CLASS_II_W1W2_MIN, color="black", linestyle=(0, (5, 3)), linewidth=1.2)
    ax.axvline(YSO_CLASS_I_W1W2, color="black", linestyle=(0, (5, 3)), linewidth=1.2)
    ax.plot(
        [YSO_CLASS_II_W1W2_MIN, YSO_CLASS_I_W1W2],
        [YSO_CLASS_II_HK, YSO_CLASS_II_HK],
        color="black",
        linestyle=(0, (5, 3)),
        linewidth=1.2,
    )
    ax.text(0.02, 0.96, "MALCA heuristic boundaries", transform=ax.transAxes, va="top", fontsize=9.5)
    ax.set_xlabel(LABEL_W1_W2, fontsize=17)
    ax.set_ylabel(LABEL_H_KS, fontsize=17)
    add_blackbody_locus(ax, ("W1", "W2"), ("H", "Ks"))
    ax.legend(
        handles=_class_handles(white_face=True, neutral_edges=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=len(CLASS_ORDER),
        fontsize=9.5,
        title="Marker shape = SED class",
        title_fontsize=9.5,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.025, extend="max")
    cbar.set_label("AllWISE MEP W1 intrinsic scatter [mag]", fontsize=13)
    ax.text(
        0.98,
        0.03,
        f"N={len(plotted)}; network-free AllWISE metric\n(not the paper's unWISE Z-score)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
    )
    _style_axis(ax)
    _save_figure(fig, output_dir, "all_dippers_2mass_wise_hk_w1w2")
    return int(len(plotted))


def _plot_halpha_panel(ax: plt.Axes, frame: pd.DataFrame, prefix: str, title: str) -> int:
    xcol = f"{prefix}_r_i"
    ycol = f"{prefix}_r_ha"
    xerr_col = f"{prefix}_r_i_err"
    yerr_col = f"{prefix}_r_ha_err"
    plotted = frame.dropna(subset=[xcol, ycol]).copy()
    for class_name in CLASS_ORDER:
        subset = plotted.loc[plotted["plot_class"] == class_name]
        if subset.empty:
            continue
        valid_error = (subset[xerr_col] > 0) & (subset[yerr_col] > 0)
        if valid_error.any():
            err = subset.loc[valid_error]
            ax.errorbar(
                err[xcol],
                err[ycol],
                xerr=err[xerr_col],
                yerr=err[yerr_col],
                fmt="none",
                ecolor=CLASS_STYLE[class_name]["color"],
                alpha=0.45,
                linewidth=0.7,
                zorder=1,
            )
        ax.scatter(
            subset[xcol],
            subset[ycol],
            s=56,
            marker=CLASS_STYLE[class_name]["marker"],
            facecolor=CLASS_STYLE[class_name]["color"],
            edgecolor="#222222",
            linewidth=0.75,
            zorder=2,
        )
    ax.set_title(f"{title} (N={len(plotted)})", fontsize=15)
    ax.set_xlabel(LABEL_R_I, fontsize=15)
    ax.set_ylabel(LABEL_R_HALPHA, fontsize=15)
    ax.text(
        0.97,
        0.04,
        "Catalog colors only; no stellar-locus grid stored",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    _style_axis(ax)
    return int(len(plotted))


def plot_halpha(frame: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.3), layout="constrained")
    n_iphas = _plot_halpha_panel(axes[0], frame, "iphas", "IPHAS")
    n_vphas = _plot_halpha_panel(axes[1], frame, "vphas", "VPHAS+")
    fig.legend(
        handles=_class_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(CLASS_ORDER),
        fontsize=10,
    )
    _save_figure(fig, output_dir, "all_dippers_iphas_vphas_halpha")
    return {"IPHAS": n_iphas, "VPHAS+": n_vphas}


def _w4_quality_status(row: pd.Series) -> str:
    snr = _finite_number(row.get("allwise_w4_snr"))
    ph_qual = str(row.get("allwise_ph_qual") or "")
    w4_is_upper = len(ph_qual) >= 4 and ph_qual[3].upper() == "U"
    if np.isfinite(snr):
        return "limit" if snr < 2.0 or w4_is_upper else "detection"
    return "unknown"


def plot_wise_cmd(frame: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    plotted = frame.dropna(subset=["w3_w4_color", "w3"]).copy()
    plotted["w4_quality_status"] = plotted.apply(_w4_quality_status, axis=1)
    face = {"detection": "#5f8f79", "limit": "#f5bf24", "unknown": "white"}
    edge = {"detection": "#333333", "limit": "#555555", "unknown": "#888888"}
    fig, ax = plt.subplots(figsize=(7.3, 6.6), layout="constrained")

    valid_error = plotted[["w3_w4_err", "w3_err"]].notna().all(axis=1) & (plotted["w3_err"] > 0)
    if valid_error.any():
        err = plotted.loc[valid_error]
        ax.errorbar(
            err["w3_w4_color"],
            err["w3"],
            xerr=err["w3_w4_err"],
            yerr=err["w3_err"],
            fmt="none",
            ecolor="#777777",
            alpha=0.35,
            linewidth=0.6,
            zorder=1,
        )
    for class_name in CLASS_ORDER:
        for quality_status in ("detection", "limit", "unknown"):
            subset = plotted.loc[
                (plotted["plot_class"] == class_name)
                & (plotted["w4_quality_status"] == quality_status)
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["w3_w4_color"],
                subset["w3"],
                s=62,
                marker=CLASS_STYLE[class_name]["marker"],
                facecolor=face[quality_status],
                edgecolor=edge[quality_status],
                linewidth=1.0,
                zorder=2,
            )
    ax.invert_yaxis()
    ax.set_xlabel(LABEL_W3_W4, fontsize=17)
    ax.set_ylabel(LABEL_W3, fontsize=17)
    class_legend = ax.legend(
        handles=_class_handles(),
        title="SED class",
        loc="upper left",
        fontsize=9.2,
        title_fontsize=9.5,
    )
    ax.add_artist(class_legend)
    quality_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=face["detection"], markeredgecolor=edge["detection"], label=r"W4 S/N $\geq 2$"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=face["limit"], markeredgecolor=edge["limit"], label="W4 upper/low S/N"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=face["unknown"], markeredgecolor=edge["unknown"], label="W4 quality unknown"),
    ]
    ax.legend(handles=quality_handles, title="AllWISE quality", loc="upper right", fontsize=9.2, title_fontsize=9.5)
    _style_axis(ax)
    _save_figure(fig, output_dir, "all_dippers_wise_w3w4_cmd")
    counts = plotted["w4_quality_status"].value_counts()
    return {name: int(counts.get(name, 0)) for name in ("detection", "limit", "unknown")}


def _semimajor_axis_grid(
    duration_days: np.ndarray,
    depth: np.ndarray,
    mass_solar: float,
    radius_solar: float,
) -> np.ndarray:
    duration_seconds = np.asarray(duration_days, float) * DAY_S
    tau = np.asarray(depth, float)
    occulter_radius_m = radius_solar * SOLAR_RADIUS_M * np.sqrt(tau)
    stellar_radius_m = radius_solar * SOLAR_RADIUS_M
    mass_kg = mass_solar * SOLAR_MASS_KG
    # Full crossing: v = 2(R_star + R_occ)/duration and a = GM/v^2.
    axis_m = (
        GRAVITATIONAL_CONSTANT_SI
        * mass_kg
        * duration_seconds**2
        / (4.0 * np.square(occulter_radius_m + stellar_radius_m))
    )
    return axis_m / AU_M


def _eclipse_probability_grid(
    duration_days: np.ndarray,
    depth: np.ndarray,
    mass_solar: float,
    radius_solar: float,
) -> np.ndarray:
    """Return the dimensionless probability proxy (R_star + R_occ) / a_proxy."""
    axis_au = _semimajor_axis_grid(duration_days, depth, mass_solar, radius_solar)
    radius_sum_m = radius_solar * SOLAR_RADIUS_M * (1.0 + np.sqrt(np.asarray(depth, float)))
    return radius_sum_m / (axis_au * AU_M)


def add_fwhm_proxy_bounds(
    frame: pd.DataFrame,
    *,
    mass_solar: float = 1.0,
    radius_solar: float = 1.0,
) -> pd.DataFrame:
    """Add physical-proxy bounds without promoting interval midpoints.

    Resolved durations map to one value.  Finite duration intervals map to
    finite lower/upper proxy bounds.  A duration lower limit maps to a lower
    limit on ``a_proxy`` and an upper limit on ``P_ecl,proxy``.
    """
    out = frame.copy()
    status = out["duration_status"].astype(str)
    duration_point = pd.to_numeric(out["duration_plot_days"], errors="coerce")
    duration_lower = pd.to_numeric(out["duration_lower_days"], errors="coerce")
    duration_upper = pd.to_numeric(out["duration_upper_days"], errors="coerce")
    depth = pd.to_numeric(out["tau_peak"], errors="coerce")

    resolved = status.eq("resolved")
    finite_interval = status.eq("interval_censored")
    lower_limit = out["duration_is_lower_limit"].fillna(False).astype(bool)
    proxy_duration_lower = duration_lower.where(~resolved, duration_point)
    proxy_duration_upper = duration_upper.where(~resolved, duration_point)
    proxy_duration_upper = proxy_duration_upper.where(~lower_limit, np.nan)
    proxy_duration_lower = proxy_duration_lower.where(
        resolved | finite_interval | lower_limit,
        np.nan,
    )

    valid_lower = proxy_duration_lower.gt(0) & depth.gt(0)
    valid_upper = proxy_duration_upper.gt(0) & depth.gt(0)
    a_lower = np.full(len(out), np.nan, dtype=float)
    a_upper = np.full(len(out), np.nan, dtype=float)
    p_upper = np.full(len(out), np.nan, dtype=float)
    p_lower = np.full(len(out), np.nan, dtype=float)
    if valid_lower.any():
        a_lower[valid_lower] = _semimajor_axis_grid(
            proxy_duration_lower[valid_lower].to_numpy(float),
            depth[valid_lower].to_numpy(float),
            mass_solar,
            radius_solar,
        )
        p_upper[valid_lower] = _eclipse_probability_grid(
            proxy_duration_lower[valid_lower].to_numpy(float),
            depth[valid_lower].to_numpy(float),
            mass_solar,
            radius_solar,
        )
    if valid_upper.any():
        a_upper[valid_upper] = _semimajor_axis_grid(
            proxy_duration_upper[valid_upper].to_numpy(float),
            depth[valid_upper].to_numpy(float),
            mass_solar,
            radius_solar,
        )
        p_lower[valid_upper] = _eclipse_probability_grid(
            proxy_duration_upper[valid_upper].to_numpy(float),
            depth[valid_upper].to_numpy(float),
            mass_solar,
            radius_solar,
        )
    p_lower[lower_limit.to_numpy(bool) & valid_lower.to_numpy(bool)] = 0.0

    out["fwhm_proxy_duration_lower_days"] = proxy_duration_lower
    out["fwhm_proxy_duration_upper_days"] = proxy_duration_upper
    out["a_proxy_lower_au"] = a_lower
    out["a_proxy_upper_au"] = a_upper
    out["p_ecl_proxy_lower"] = p_lower
    out["p_ecl_proxy_upper"] = p_upper
    return out


def _draw_timescale_depth_points(
    ax: plt.Axes,
    plotted: pd.DataFrame,
    *,
    marker_size: float = 47,
    marker_linewidth: float = 0.65,
    arrow_linewidth: float = 0.9,
    arrow_mutation_scale: float = 8.5,
    arrow_outline_linewidth: float = 1.6,
) -> None:
    """Draw bounded FWHM durations and open-side lower limits."""
    for class_name in CLASS_ORDER:
        subset = plotted.loc[plotted["plot_class"] == class_name]
        if subset.empty:
            continue
        limited = subset.loc[subset["duration_is_lower_limit"].astype(bool)]
        bounded = subset.loc[~subset["duration_is_lower_limit"].astype(bool)]

        def amplitude_errors(rows: pd.DataFrame) -> np.ndarray:
            y_values = rows["tau_peak"].to_numpy(float)
            y_minus = pd.to_numeric(
                rows.get("tau_peak_mc_err_minus", pd.Series(0.0, index=rows.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(float)
            y_plus = pd.to_numeric(
                rows.get("tau_peak_mc_err_plus", pd.Series(0.0, index=rows.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(float)
            return np.vstack([np.minimum(y_minus, 0.95 * y_values), y_plus])

        if not bounded.empty:
            bounded_x = bounded["duration_plot_days"].to_numpy(float)
            x_minus = np.zeros(len(bounded), dtype=float)
            x_plus = np.zeros(len(bounded), dtype=float)
            bounded_interval = bounded["duration_is_interval_censored"].fillna(False).astype(bool)
            if bounded_interval.any():
                x_minus[bounded_interval.to_numpy()] = (
                    bounded.loc[bounded_interval, "duration_plot_days"]
                    - bounded.loc[bounded_interval, "duration_lower_days"]
                ).clip(lower=0).to_numpy(float)
                x_plus[bounded_interval.to_numpy()] = (
                    bounded.loc[bounded_interval, "duration_upper_days"]
                    - bounded.loc[bounded_interval, "duration_plot_days"]
                ).clip(lower=0).to_numpy(float)
            reporting_status = bounded.get(
                "duration_mc_reporting_status",
                pd.Series("", index=bounded.index, dtype=object),
            )
            reported = (~bounded_interval) & reporting_status.eq("reported_resolved")
            if reported.any():
                x_minus[reported.to_numpy()] = np.minimum(
                    pd.to_numeric(
                        bounded.loc[reported, "duration_mc_err_minus"], errors="coerce"
                    ).fillna(0.0).to_numpy(float),
                    0.9 * bounded_x[reported.to_numpy()],
                )
                x_plus[reported.to_numpy()] = pd.to_numeric(
                    bounded.loc[reported, "duration_mc_err_plus"], errors="coerce"
                ).fillna(0.0).to_numpy(float)
            ax.errorbar(
                bounded_x,
                bounded["tau_peak"].to_numpy(float),
                xerr=np.vstack([x_minus, x_plus]),
                yerr=amplitude_errors(bounded),
                fmt="none",
                ecolor=TIMESCALE_CLASS_OUTLINE[class_name],
                elinewidth=max(0.5, 0.85 * marker_linewidth),
                capsize=1.4,
                alpha=0.9,
                zorder=3.6,
            )
            ax.scatter(
                bounded_x,
                bounded["tau_peak"],
                marker=CLASS_STYLE[class_name]["marker"],
                s=marker_size,
                facecolor="#111111",
                edgecolor=TIMESCALE_CLASS_OUTLINE[class_name],
                linewidth=marker_linewidth,
                zorder=4,
            )
        if not limited.empty:
            limited_x = limited["duration_lower_days"].to_numpy(float)
            ax.errorbar(
                limited_x,
                limited["tau_peak"].to_numpy(float),
                yerr=amplitude_errors(limited),
                fmt="none",
                ecolor=TIMESCALE_CLASS_OUTLINE[class_name],
                elinewidth=max(0.5, 0.85 * marker_linewidth),
                capsize=1.4,
                alpha=0.9,
                zorder=3.6,
            )
            ax.scatter(
                limited_x,
                limited["tau_peak"],
                marker=CLASS_STYLE[class_name]["marker"],
                s=marker_size,
                facecolor="#111111",
                edgecolor=TIMESCALE_CLASS_OUTLINE[class_name],
                linewidth=marker_linewidth,
                zorder=4,
            )
            for duration, depth in zip(
                limited_x,
                limited["tau_peak"].to_numpy(float),
            ):
                arrow = ax.annotate(
                    "",
                    xy=(duration * 1.35, depth),
                    xytext=(duration, depth),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#111111",
                        "linewidth": arrow_linewidth,
                        "mutation_scale": arrow_mutation_scale,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    zorder=3.9,
                )
                arrow.arrow_patch.set_path_effects(
                    [
                        path_effects.Stroke(
                            linewidth=arrow_outline_linewidth,
                            foreground=TIMESCALE_CLASS_OUTLINE[class_name],
                        ),
                        path_effects.Normal(),
                    ]
                )


def _add_timescale_depth_class_legend(
    ax: plt.Axes,
    *,
    title: str | None = "SED class",
    fontsize: float = 9.2,
    title_fontsize: float = 9.5,
    marker_size: float = 7,
    marker_edgewidth: float = 0.65,
    frame_linewidth: float = 1.0,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (0.025, 0.975),
    borderpad: float = 0.4,
    handletextpad: float = 0.8,
    labelspacing: float = 0.5,
) -> Any:
    class_handles = [
        Line2D(
            [0],
            [0],
            marker=CLASS_STYLE[class_name]["marker"],
            linestyle="none",
            markerfacecolor="#111111",
            markeredgecolor=TIMESCALE_CLASS_OUTLINE[class_name],
            markeredgewidth=marker_edgewidth,
            markersize=marker_size,
            label=CLASS_STYLE[class_name]["label"],
        )
        for class_name in CLASS_ORDER
    ]
    class_legend = ax.legend(
        handles=class_handles,
        title=title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderpad=borderpad,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
    )
    class_legend.get_frame().set_linewidth(frame_linewidth)
    return class_legend


def _duration_status_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#111111",
            markeredgecolor="#777777",
            markersize=5,
            label="FWHM",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#333333",
            marker=">",
            markevery=[1],
            linewidth=1.1,
            markersize=5,
            label="FWHM lower limit",
        ),
    ]


def plot_timescale_depth(frame: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    plotted = frame.dropna(subset=["duration_plot_days", "tau_peak"]).copy()
    plotted = plotted.loc[(plotted["duration_plot_days"] > 0) & (plotted["tau_peak"] > 0)].copy()
    xgrid = np.logspace(
        np.log10(DURATION_DEPTH_XMIN_DAYS),
        np.log10(DURATION_DEPTH_XMAX_DAYS),
        260,
    )
    ymax = 0.6
    ygrid = np.linspace(0.0, ymax, 220)
    xx, yy = np.meshgrid(xgrid, ygrid)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True, layout="constrained")
    scenarios = [(1.0, 1.0, r"Assumed $1.0\,M_\odot$, $1.0\,R_\odot$"), (0.8, 0.8, r"Assumed $0.8\,M_\odot$, $0.8\,R_\odot$")]
    axis_grids = [_semimajor_axis_grid(xx, yy, mass, radius) for mass, radius, _ in scenarios]
    color_min = min(float(np.nanmin(grid)) for grid in axis_grids)
    color_max = max(float(np.nanmax(grid)) for grid in axis_grids)
    levels = np.geomspace(color_min, color_max, 256)
    norm = LogNorm(vmin=color_min, vmax=color_max)
    line_exponents = np.arange(
        int(np.ceil(np.log10(color_min))),
        int(np.floor(np.log10(color_max))) + 1,
    )
    line_levels = np.power(10.0, line_exponents)
    contour_fill = None
    label_depth_fractions = np.array([0.35, 0.48, 0.60, 0.82, 0.70, 0.58, 0.82, 0.16])
    for ax, (mass, radius, title), axis_grid in zip(axes, scenarios, axis_grids):
        contour_fill = ax.contourf(
            xx,
            yy,
            axis_grid,
            levels=levels,
            norm=norm,
            cmap="viridis",
            zorder=0,
        )
        contour_fill.set_edgecolor("face")
        ax.set_rasterization_zorder(1)
        lines = ax.contour(
            xx,
            yy,
            axis_grid,
            levels=line_levels,
            colors="black",
            linewidths=0.55,
            alpha=0.72,
            zorder=2,
        )
        label_depths = ymax * label_depth_fractions
        radius_m = radius * SOLAR_RADIUS_M
        label_durations = (
            2.0
            * radius_m
            * (1.0 + np.sqrt(label_depths))
            * np.sqrt(line_levels * AU_M / (GRAVITATIONAL_CONSTANT_SI * mass * SOLAR_MASS_KG))
            / DAY_S
        )
        labels = ax.clabel(
            lines,
            fmt=lambda value: rf"$10^{{{int(np.rint(np.log10(value)))}}}$",
            fontsize=11,
            inline=True,
            inline_spacing=1,
            manual=list(zip(label_durations, label_depths)),
        )
        for label in labels:
            label.set_rotation(0)
            label.set_zorder(6)
            label.set_color("black")
            label.set_alpha(1.0)

        _draw_timescale_depth_points(ax, plotted, marker_size=38)
        ax.set_xscale("log")
        ax.set_xlim(DURATION_DEPTH_XMIN_DAYS, DURATION_DEPTH_XMAX_DAYS)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(r"$\tau_{\mathrm{FWHM}}$ [days]", fontsize=14)
        _style_axis(ax)
    axes[0].set_ylabel(r"$\delta$", fontsize=15)
    _add_timescale_depth_class_legend(axes[0])
    fig.legend(
        handles=_duration_status_handles(),
        loc="lower center",
        bbox_to_anchor=(0.47, -0.015),
        ncol=3,
        fontsize=8.4,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        handlelength=2.4,
        columnspacing=1.5,
    )
    if contour_fill is not None:
        color_mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        color_mappable.set_array([])
        cbar = fig.colorbar(color_mappable, ax=axes, pad=0.018)
        cbar.set_ticks(line_levels)
        cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        cbar.set_label(r"$a_{\mathrm{proxy}}$ [AU]", fontsize=12)
    _save_figure(fig, output_dir, "all_dippers_timescale_depth")
    statuses = plotted["duration_status"].value_counts()
    coverage = {str(key): int(value) for key, value in statuses.items()}
    coverage["unconstrained_no_finite_width"] = int(len(frame) - len(plotted))
    return coverage


def _duration_depth_axis_limits(_plotted: pd.DataFrame) -> tuple[float, float]:
    return DURATION_DEPTH_XMIN_DAYS, DURATION_DEPTH_XMAX_DAYS


def _add_depth_magnitude_axis(ax: plt.Axes, *, ymax: float = 0.6) -> None:
    ax.tick_params(axis="y", which="both", right=False)
    magnitude_axis = ax.secondary_yaxis(
        "right",
        functions=(_fractional_depth_to_delta_mag, _delta_mag_to_fractional_depth),
    )
    magnitude_axis.set_ylabel(r"$\Delta m$ [mag]", fontsize=14)
    magnitude_axis.set_yticks(
        np.linspace(0.0, float(_fractional_depth_to_delta_mag(ymax)), 6)
    )
    magnitude_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    magnitude_axis.minorticks_on()
    magnitude_axis.tick_params(which="both", direction="in", labelsize=11)
    magnitude_axis.tick_params(which="major", length=6, width=1.0)
    magnitude_axis.tick_params(which="minor", length=3, width=0.8)


def plot_fwhm_duration_depth(frame: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    """Plot peak-centered FWHM duration against depth with full censoring support."""
    plotted = frame.dropna(subset=["duration_plot_days", "tau_peak"]).copy()
    plotted = plotted.loc[
        (plotted["duration_plot_days"] > 0) & (plotted["tau_peak"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    ymax = 0.6
    _draw_timescale_depth_points(
        ax,
        plotted,
        marker_size=34,
        marker_linewidth=0.65,
        arrow_linewidth=0.9,
        arrow_mutation_scale=8.5,
        arrow_outline_linewidth=1.6,
    )
    x_min, x_max = _duration_depth_axis_limits(plotted)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(r"$\tau_{\mathrm{FWHM}}$ [days]", fontsize=13)
    ax.set_ylabel(r"$\delta$", fontsize=14)
    ax.grid(which="major", color="#d0d0d0", linewidth=0.55, alpha=0.55, zorder=0)
    _style_axis(ax)
    _add_depth_magnitude_axis(ax, ymax=ymax)
    class_legend = _add_timescale_depth_class_legend(
        ax,
        fontsize=8.4,
        title_fontsize=8.8,
        marker_size=6,
        bbox_to_anchor=(0.02, 0.98),
    )
    ax.add_artist(class_legend)
    _save_figure(fig, output_dir, "all_dippers_fwhm_duration_depth")

    status_counts = plotted["duration_status"].value_counts()
    coverage = {str(key): int(value) for key, value in status_counts.items()}
    coverage["unavailable"] = int(len(frame) - len(plotted))
    return coverage


def plot_dimming_complex_duration(
    frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    """Plot the full recovery-anchored dimming-complex span separately."""
    plotted = frame.dropna(
        subset=["dimming_complex_duration_plot_days", "tau_peak"]
    ).copy()
    plotted = plotted.loc[
        (plotted["dimming_complex_duration_plot_days"] > 0)
        & (plotted["tau_peak"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    for class_name in CLASS_ORDER:
        subset = plotted.loc[plotted["plot_class"] == class_name]
        if subset.empty:
            continue
        bounded = subset.loc[~subset["dimming_complex_is_lower_limit"].astype(bool)]
        limited = subset.loc[subset["dimming_complex_is_lower_limit"].astype(bool)]

        for rows in (bounded, limited):
            if rows.empty:
                continue
            y_values = rows["tau_peak"].to_numpy(float)
            y_minus = pd.to_numeric(
                rows.get("tau_peak_mc_err_minus", pd.Series(0.0, index=rows.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(float)
            y_plus = pd.to_numeric(
                rows.get("tau_peak_mc_err_plus", pd.Series(0.0, index=rows.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(float)
            ax.errorbar(
                rows["dimming_complex_duration_lower_days"].to_numpy(float),
                y_values,
                yerr=np.vstack([np.minimum(y_minus, 0.95 * y_values), y_plus]),
                fmt="none",
                ecolor=TIMESCALE_CLASS_OUTLINE[class_name],
                elinewidth=0.65,
                capsize=1.5,
                alpha=0.88,
                zorder=2.8,
            )
            ax.scatter(
                rows["dimming_complex_duration_lower_days"],
                y_values,
                marker=CLASS_STYLE[class_name]["marker"],
                s=34,
                facecolor="#111111",
                edgecolor=TIMESCALE_CLASS_OUTLINE[class_name],
                linewidth=0.65,
                zorder=3,
            )

        for duration, depth in zip(
            limited["dimming_complex_duration_lower_days"].to_numpy(float),
            limited["tau_peak"].to_numpy(float),
        ):
            arrow = ax.annotate(
                "",
                xy=(duration * 1.35, depth),
                xytext=(duration, depth),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#111111",
                    "linewidth": 0.9,
                    "mutation_scale": 8.5,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=3.1,
            )
            arrow.arrow_patch.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=1.6,
                        foreground=TIMESCALE_CLASS_OUTLINE[class_name],
                    ),
                    path_effects.Normal(),
                ]
            )

    ymax = 0.6
    ax.set_xscale("log")
    ax.set_xlim(DURATION_DEPTH_XMIN_DAYS, DURATION_DEPTH_XMAX_DAYS)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(r"$\tau$ [days]", fontsize=13)
    ax.set_ylabel(r"$\delta$", fontsize=14)
    ax.grid(which="major", color="#d0d0d0", linewidth=0.55, alpha=0.55, zorder=0)
    _style_axis(ax)
    _add_depth_magnitude_axis(ax, ymax=ymax)
    class_legend = _add_timescale_depth_class_legend(
        ax,
        fontsize=8.4,
        title_fontsize=8.8,
        marker_size=6,
        bbox_to_anchor=(0.02, 0.98),
    )
    ax.add_artist(class_legend)
    _save_figure(fig, output_dir, "all_dippers_dimming_complex_duration_depth")

    status_counts = plotted["dimming_complex_status"].value_counts()
    coverage = {str(key): int(value) for key, value in status_counts.items()}
    coverage["unavailable"] = int(len(frame) - len(plotted))
    return coverage


def plot_eclipse_probability_proxy(frame: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    """Plot the geometric eclipse-probability proxy tied to a_proxy."""
    plotted = frame.dropna(subset=["duration_plot_days", "tau_peak"]).copy()
    plotted = plotted.loc[(plotted["duration_plot_days"] > 0) & (plotted["tau_peak"] > 0)].copy()
    xgrid = np.logspace(
        np.log10(DURATION_DEPTH_XMIN_DAYS),
        np.log10(DURATION_DEPTH_XMAX_DAYS),
        260,
    )
    ymax = 0.6
    ygrid = np.linspace(0.0, ymax, 220)
    xx, yy = np.meshgrid(xgrid, ygrid)

    # The extra horizontal canvas margin keeps the left ylabel fully inside
    # the export while the box aspect and fixed height preserve panel width.
    fig, ax = plt.subplots(figsize=(3.58, 2.535), layout="constrained")
    mass = 1.0
    radius = 1.0
    probability_grid = _eclipse_probability_grid(xx, yy, mass, radius)
    color_min = float(np.nanmin(probability_grid))
    color_max = float(np.nanmax(probability_grid))
    levels = np.geomspace(color_min, color_max, 256)
    norm = LogNorm(vmin=color_min, vmax=color_max)
    line_exponents = np.arange(
        int(np.ceil(np.log10(color_min))),
        int(np.floor(np.log10(color_max))) + 1,
    )
    line_levels = np.power(10.0, line_exponents)
    label_depth_fractions = np.array(
        [
            0.08,  # 10^-9
            0.18,  # 10^-8
            0.48,  # 10^-7
            0.60,  # 10^-6
            0.82,  # 10^-5
            0.58,  # 10^-4; keep below the legend
            0.52,  # 10^-3
        ]
    )
    contour_fill = ax.contourf(
        xx,
        yy,
        probability_grid,
        levels=levels,
        norm=norm,
        cmap="viridis",
        zorder=0,
    )
    contour_fill.set_edgecolor("face")
    ax.set_rasterization_zorder(1)
    lines = ax.contour(
        xx,
        yy,
        probability_grid,
        levels=line_levels,
        colors="black",
        linewidths=0.45,
        alpha=0.72,
        zorder=2,
    )
    label_depths = ymax * label_depth_fractions
    probability_coefficient = (
        4.0
        * (radius * SOLAR_RADIUS_M) ** 3
        / (GRAVITATIONAL_CONSTANT_SI * mass * SOLAR_MASS_KG * DAY_S**2)
    )
    label_durations = np.sqrt(
        probability_coefficient
        * np.power(1.0 + np.sqrt(label_depths), 3)
        / line_levels
    )
    labels = ax.clabel(
        lines,
        fmt=lambda value: rf"$10^{{{int(np.rint(np.log10(value)))}}}$",
        fontsize=6.2,
        inline=True,
        inline_spacing=-1,
        manual=list(zip(label_durations, label_depths)),
    )
    for label in labels:
        label.set_rotation(0)
        label.set_zorder(6)
        label.set_color("black")
        label.set_alpha(1.0)

    _draw_timescale_depth_points(
        ax,
        plotted,
        marker_size=14,
        marker_linewidth=0.45,
        arrow_linewidth=0.65,
        arrow_mutation_scale=5.8,
        arrow_outline_linewidth=1.15,
    )
    ax.set_xscale("log")
    ax.set_xlim(DURATION_DEPTH_XMIN_DAYS, DURATION_DEPTH_XMAX_DAYS)
    ax.set_ylim(0, ymax)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$\tau_{\mathrm{FWHM}}$ [days]", fontsize=9.5)
    ax.set_ylabel(r"$\delta$", fontsize=10)
    _style_axis(ax)
    ax.tick_params(axis="y", which="both", right=False)
    ax.tick_params(which="major", length=4, width=0.8, labelsize=7.5)
    ax.tick_params(which="minor", length=2, width=0.6)
    magnitude_axis = ax.secondary_yaxis(
        "right",
        functions=(_fractional_depth_to_delta_mag, _delta_mag_to_fractional_depth),
    )
    # Keep the secondary axis above the z=1 rasterization threshold used only
    # for the filled contour background so its ticks and text stay vectorized.
    magnitude_axis.set_zorder(3)
    magnitude_axis.set_ylabel(r"$\Delta m$ [mag]", fontsize=9)
    magnitude_axis.set_yticks(
        np.linspace(0.0, float(_fractional_depth_to_delta_mag(ymax)), 6)
    )
    magnitude_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    magnitude_axis.minorticks_on()
    magnitude_axis.tick_params(
        which="both",
        direction="in",
        labelsize=7.5,
    )
    magnitude_axis.tick_params(which="major", length=4, width=0.8)
    magnitude_axis.tick_params(which="minor", length=2, width=0.6)
    _add_timescale_depth_class_legend(
        ax,
        title=None,
        fontsize=5.4,
        title_fontsize=5.4,
        marker_size=4.0,
        marker_edgewidth=0.4,
        frame_linewidth=0.65,
        borderpad=0.28,
        handletextpad=0.55,
        labelspacing=0.32,
    )
    color_mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    color_mappable.set_array([])
    cbar = fig.colorbar(color_mappable, ax=ax, pad=0.018)
    cbar.set_ticks(line_levels)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    cbar.ax.tick_params(labelsize=7.2, length=3, width=0.7)
    cbar.set_label(r"$P_{\mathrm{ecl,proxy}}$", fontsize=9)
    _save_figure(
        fig,
        output_dir,
        "all_dippers_eclipse_probability_proxy",
        tight=False,
    )
    statuses = plotted["duration_status"].value_counts()
    coverage = {str(key): int(value) for key, value in statuses.items()}
    coverage["unconstrained_no_finite_width"] = int(len(frame) - len(plotted))
    return coverage


def plot_symmetry_duration(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Plot stored dip-shape symmetry against the maximum detected dip-run duration."""
    plotted = frame.dropna(subset=["dip_symmetry_score", "dip_max_run_duration"]).copy()
    plotted = plotted.loc[plotted["dip_max_run_duration"] > 0].copy()

    fig, ax = plt.subplots(figsize=(3.5, 3.35), layout="constrained")
    for class_name in CLASS_ORDER:
        subset = plotted.loc[plotted["plot_class"] == class_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["dip_symmetry_score"],
            subset["dip_max_run_duration"],
            marker=CLASS_STYLE[class_name]["marker"],
            s=18,
            facecolor="#111111",
            edgecolor=TIMESCALE_CLASS_OUTLINE[class_name],
            linewidth=0.5,
            zorder=3,
        )

    ax.axvline(0.0, color="#333333", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    x_min = float(plotted["dip_symmetry_score"].min())
    x_max = float(plotted["dip_symmetry_score"].max())
    x_pad = 0.06 * (x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_yscale("log")
    ax.set_ylim(3.0, 1_000.0)
    ax.set_xlabel(r"Dip symmetry score, $S_{\mathrm{dip}}$", fontsize=9.5)
    ax.set_ylabel("Maximum dip-run duration [days]", fontsize=9.5)
    _style_axis(ax)
    ax.tick_params(which="major", length=4, width=0.8, labelsize=7.5)
    ax.tick_params(which="minor", length=2, width=0.6)
    _add_timescale_depth_class_legend(
        ax,
        fontsize=6.2,
        title_fontsize=6.6,
        marker_size=4.7,
        marker_edgewidth=0.45,
        frame_linewidth=0.75,
        loc="lower right",
        bbox_to_anchor=(0.975, 0.025),
    )
    ax.text(
        0.025,
        0.975,
        f"N={len(plotted)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    _save_figure(
        fig,
        output_dir,
        "all_dippers_symmetry_duration",
        tight=False,
    )
    return {
        "n_plotted": int(len(plotted)),
        "n_missing": int(len(frame) - len(plotted)),
        "class_counts": _jsonable_counts(plotted["plot_class"]),
        "symmetry_range": [x_min, x_max],
        "duration_range_days": [
            float(plotted["dip_max_run_duration"].min()),
            float(plotted["dip_max_run_duration"].max()),
        ],
    }


def _jsonable_counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}


def write_summary(
    output_dir: Path,
    review_db: Path,
    frame: pd.DataFrame,
    coverage: dict[str, Any],
) -> None:
    summary = {
        "review_db": str(review_db),
        "selection": "lower(trim(reviews.event_class)) = 'dipper'; no final_class/yso_class cut",
        "n_all_dippers": int(len(frame)),
        "class_counts": _jsonable_counts(frame["plot_class"]),
        "coverage": coverage,
        "bailer_jones": {
            "catalog": "Gaia EDR3 distance catalogue via Gaia TAP external.gaiaedr3_distance",
            "plotted_field": "r_med_photogeo",
            "n_photogeometric": int(frame["bj_r_med_photogeo"].notna().sum()),
            "n_geometric": int(frame["bj_r_med_geo"].notna().sum()),
            "unmatched_candidate_ids": frame.loc[
                frame["bj_r_med_photogeo"].isna(), "candidate_id"
            ].tolist(),
        },
        "figure_notes": {
            "distance": "Bailer-Jones photogeometric distances are fetched from Gaia TAP external.gaiaedr3_distance and remain distinct from inverse parallax, StarHorse, and Gaia GSP-Phot.",
            "2mass_wise": "Color is homogeneous AllWISE MEP W1 intrinsic scatter, not the unavailable unWISE W1 Z-score.",
            "halpha": "IPHAS and VPHAS+ remain separate; no literature stellar loci or EW tracks are stored.",
            "wise_cmd": "Static AllWISE photometry with W4 quality flags; no BT-NextGen disk-model grid is stored.",
            "timescale_depth": "Fresh locally normalized ASAS-SN measurement of the deepest gap-safe nearest-three-night local-median dip inside a recovery-anchored search bracket. Half-depth scans continue across sampling gaps and stop only after 5 of 6 independent nightly medians recover above the half-depth line within one observing block spanning at least 7 days. Gap-hidden crossings are drawn at the geometric midpoint with horizontal spans when needed; open sides are lower-limit arrows. Vertical bars show recentered conditional MC68 amplitude spread. Horizontal Monte Carlo errors are reported only for nominally resolved FWHM values whose crossings remain resolved in at least 90% of evaluable draws and at least 200 draws.",
            "fwhm_duration_depth": "Peak-centered FWHM duration against depth on the same axes as the dimming-complex companion plot. Bounded FWHM values are solid class markers with horizontal and vertical error bars when available; arrows mark open-side lower limits.",
            "dimming_complex_duration": "Computationally measured outer recovery-bracket span for every light curve. Recovery-bounded events are finite spans. Ongoing, data-edge, gap, or unconfirmed boundaries are reported as observed lower limits rather than exact durations.",
            "half_depth_atlas": "Each row separates the individual-dip FWHM from the outer recovery-bracket dimming-complex duration in a dedicated measurements column. A half-depth side is recovered only after 5 of 6 independent nightly medians lie above the half-depth line within one observing block spanning at least 7 days. Brief excursions do not split the event, and the scan continues across sampling gaps. A gap may bracket a finite FWHM interval but never supplies an exact crossing; a truly open side remains a lower limit. The same classifier is applied in every conditional Monte Carlo draw.",
            "eclipse_probability_proxy": "Geometric proxy P_ecl = (R_star + R_occ) / a_proxy with R_occ = R_star * sqrt(delta); it is not a measured occurrence probability. Bounded FWHM durations are shown as solid class markers with horizontal spans when needed; duration lower limits map to lower limits on a_proxy and upper limits on P_ecl,proxy. The right axis converts fractional depth using Delta m = -2.5 log10(1 - delta).",
            "symmetry_duration": "Stored dip_symmetry_score is the uncertainty-normalized ingress-minus-egress area of the best detected dip run. Stored dip_max_run_duration is the longest detected dip run and is not guaranteed to refer to that same event; no symmetry-score uncertainty is stored.",
        },
    }
    with (output_dir / "all_dippers_diagnostic_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--external-manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST)
    parser.add_argument("--allwise-quality", type=Path, default=DEFAULT_ALLWISE_QUALITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distance-max-pc", type=float, default=9000.0)
    parser.add_argument(
        "--bailer-jones-cache",
        type=Path,
        default=None,
        help="CSV cache path; defaults to the output directory.",
    )
    parser.add_argument(
        "--fetch-bailer-jones",
        action="store_true",
        help="Fetch or refresh Bailer-Jones photogeometric distances from Gaia TAP.",
    )
    parser.add_argument(
        "--reuse-event-metrics",
        action="store_true",
        help="Reuse all_dippers_half_depth_metrics.csv if it already exists.",
    )
    parser.add_argument(
        "--half-depth-atlas-only",
        action="store_true",
        help=(
            "Generate only the half-depth atlas, skipping catalog enrichment and "
            "the other diagnostic figures."
        ),
    )
    parser.add_argument(
        "--dip-run-overlay",
        type=Path,
        default=None,
        help=(
            "Provenance-locked triggered_dip_runs.parquet to overlay as production "
            "dip-run intervals and triggered observations."
        ),
    )
    parser.add_argument(
        "--half-depth-atlas-stem",
        default=None,
        help="Output filename stem for the atlas, preview, metrics, and manifest.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_db = args.review_db.expanduser().resolve()
    frame = add_color_columns(read_all_dippers(review_db))
    pipeline_dip_runs = (
        read_pipeline_dip_runs(args.dip_run_overlay, frame["candidate_id"])
        if args.dip_run_overlay is not None
        else None
    )
    atlas_stem = args.half_depth_atlas_stem or (
        "all_dippers_half_depth_diagnostic_atlas_with_pipeline_runs"
        if pipeline_dip_runs is not None
        else "all_dippers_half_depth_diagnostic_atlas"
    )

    if args.half_depth_atlas_only:
        atlas_coverage = plot_half_depth_diagnostic_atlas(
            frame,
            output_dir,
            pipeline_dip_runs=pipeline_dip_runs,
            atlas_stem=atlas_stem,
            compare_cached_metrics=False,
        )
        overlay_manifest_path = (
            Path(str(pipeline_dip_runs.attrs.get("source_path", ""))).with_suffix(
                ".manifest.json"
            )
            if pipeline_dip_runs is not None
            else None
        )
        manifest = {
            "review_db": str(review_db),
            "selection": (
                "lower(trim(reviews.event_class)) = 'dipper'; no "
                "final_class/yso_class cut"
            ),
            "n_all_dippers": int(len(frame)),
            "atlas_metrics": (
                "Freshly recomputed from each light curve during atlas generation"
            ),
            "half_depth_method_versions": {
                "dimming_window": DIMMING_WINDOW_METHOD_VERSION,
                "fwhm": FWHM_METHOD_VERSION,
                "event_metrics_schema": EVENT_METRICS_SCHEMA_VERSION,
            },
            "pipeline_dip_run_replay": {
                "table": (
                    str(pipeline_dip_runs.attrs.get("source_path", ""))
                    if pipeline_dip_runs is not None
                    else None
                ),
                "manifest": (
                    str(overlay_manifest_path)
                    if overlay_manifest_path is not None
                    and overlay_manifest_path.is_file()
                    else None
                ),
                "interpretation": (
                    "Historical production Bayesian trigger runs are overlaid for "
                    "comparison only; they do not define or alter the atlas window, "
                    "peak, amplitude, or FWHM."
                    if pipeline_dip_runs is not None
                    else None
                ),
            },
            "coverage": atlas_coverage,
        }
        manifest_path = output_dir / f"{atlas_stem}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(f"Selected {len(frame)} reviewed dippers with no stellar-class cut.")
        print(f"Wrote atlas: {atlas_coverage['pdf']}")
        print(f"Wrote atlas metrics: {atlas_coverage['metrics']}")
        print(f"Wrote atlas manifest: {manifest_path}")
        return

    bj_cache = (
        args.bailer_jones_cache.expanduser().resolve()
        if args.bailer_jones_cache is not None
        else output_dir / DEFAULT_BAILER_JONES_FILENAME
    )
    bailer_jones = load_bailer_jones_distances(
        frame,
        bj_cache,
        fetch=args.fetch_bailer_jones,
    )
    frame = frame.merge(bailer_jones, on=["candidate_id", "gaia_id"], how="left", validate="one_to_one")

    variability = measure_allwise_variability(frame, args.external_manifest, args.run_root)
    frame = frame.merge(variability, on="candidate_id", how="left", validate="one_to_one")
    quality = read_allwise_quality(args.allwise_quality)
    frame = frame.merge(quality, on="candidate_id", how="left", validate="one_to_one")

    event_path = output_dir / "all_dippers_half_depth_metrics.csv"
    if args.reuse_event_metrics and event_path.exists():
        event_metrics = pd.read_csv(event_path)
        event_metrics["candidate_id"] = event_metrics["candidate_id"].astype(str)
        cache_is_current = bool(
            "fwhm_method_version" in event_metrics
            and event_metrics["fwhm_method_version"].eq(FWHM_METHOD_VERSION).all()
            and "event_metrics_schema_version" in event_metrics
            and event_metrics["event_metrics_schema_version"]
            .eq(EVENT_METRICS_SCHEMA_VERSION)
            .all()
            and "dimming_window_method_version" in event_metrics
            and event_metrics["dimming_window_method_version"]
            .eq(DIMMING_WINDOW_METHOD_VERSION)
            .all()
            and {
                "left_crossing_status",
                "right_crossing_status",
                "duration_lower_days",
                "duration_upper_days",
                "dimming_complex_duration_lower_days",
                "dimming_complex_duration_upper_days",
                "dimming_complex_is_lower_limit",
                "dimming_complex_status",
            }.issubset(event_metrics.columns)
        )
        if not cache_is_current:
            print(
                "Cached event metrics use an older measurement schema; recomputing "
                f"with {DIMMING_WINDOW_METHOD_VERSION} / {FWHM_METHOD_VERSION} / "
                f"{EVENT_METRICS_SCHEMA_VERSION}."
            )
            event_metrics = measure_all_half_depth_events(frame)
            event_metrics.to_csv(event_path, index=False)
    else:
        event_metrics = measure_all_half_depth_events(frame)
        event_metrics.to_csv(event_path, index=False)
    frame = frame.merge(event_metrics, on="candidate_id", how="left", validate="one_to_one")
    frame = add_fwhm_proxy_bounds(frame)

    frame["has_2mass_wise"] = frame[["h_ks", "w1_w2_color"]].notna().all(axis=1)
    frame["has_iphas"] = frame[["iphas_r_i", "iphas_r_ha"]].notna().all(axis=1)
    frame["has_vphas"] = frame[["vphas_r_i", "vphas_r_ha"]].notna().all(axis=1)
    frame["has_wise_cmd"] = frame[["w3_w4_color", "w3"]].notna().all(axis=1)
    frame.drop(columns=["payload_json"], errors="ignore").to_csv(
        output_dir / "all_dippers_plot_inputs.csv",
        index=False,
    )
    variability.to_csv(output_dir / "all_dippers_allwise_mep_variability.csv", index=False)

    coverage: dict[str, Any] = {}
    coverage["distance"] = plot_distance_histogram(frame, output_dir, args.distance_max_pc)
    coverage["2mass_wise"] = plot_2mass_wise(frame, output_dir)
    coverage["halpha"] = plot_halpha(frame, output_dir)
    coverage["wise_cmd"] = plot_wise_cmd(frame, output_dir)
    coverage["timescale_depth"] = plot_timescale_depth(frame, output_dir)
    coverage["fwhm_duration_depth"] = plot_fwhm_duration_depth(frame, output_dir)
    coverage["dimming_complex_duration"] = plot_dimming_complex_duration(frame, output_dir)
    coverage["eclipse_probability_proxy"] = plot_eclipse_probability_proxy(frame, output_dir)
    coverage["symmetry_duration"] = plot_symmetry_duration(frame, output_dir)
    coverage["half_depth_atlas"] = plot_half_depth_diagnostic_atlas(
        frame,
        output_dir,
        pipeline_dip_runs=pipeline_dip_runs,
        atlas_stem=atlas_stem,
    )
    write_summary(output_dir, args.review_db, frame, coverage)

    print(f"Selected {len(frame)} reviewed dippers with no stellar-class cut.")
    print(f"Class counts: {_jsonable_counts(frame['plot_class'])}")
    print(f"Wrote plots and provenance tables to {output_dir}")


if __name__ == "__main__":
    main()
