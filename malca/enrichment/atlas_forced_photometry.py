"""Resumable bulk client for the ATLAS forced-photometry service.

The ATLAS service creates one asynchronous task per coordinate, even when up
to 100 coordinates are submitted in a single ``radeclist`` request.  This
module therefore keeps a permanent, atomic task ledger separate from MALCA's
candidate-table checkpoints.  A queued or running remote task is never
treated as a no-data result and is never resubmitted just because polling was
interrupted.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from malca.config import ATLAS_API_BASE, PARQUET_CACHE_COMPRESSION
from malca.external_lc_manifest import upsert_external_lc_manifest_entry


ATLAS_SUMMARY_COLUMNS = (
    "atlas_has_phot",
    "atlas_n_det_cyan",
    "atlas_n_det_orange",
    "atlas_cyan_range",
    "atlas_orange_range",
    "atlas_preprocess_version",
    "atlas_n_raw",
    "atlas_n_good",
    "atlas_n_rejected",
)

ATLAS_PREPROCESS_VERSION = "atlas-reduced-direct-v2"
ATLAS_NOISE_MODEL_VERSION = "atlas-empirical-noise-v1"
ATLAS_PREPROCESS_DEFAULT_FILTERS = ("c", "o")
ATLAS_PREPROCESS_DEFAULT_SNR_MIN = 5.0
ATLAS_OBS_SITE_NAMES = {
    "01": "maunaloa",
    "02": "haleakala",
    "03": "south_africa",
    "04": "chile",
}
ATLAS_CHI_N_BINS = (
    ("lt10", 0.0, 10.0),
    ("10_30", 10.0, 30.0),
    ("30_100", 30.0, 100.0),
    ("100_300", 100.0, 300.0),
    ("300_1000", 300.0, 1000.0),
    ("ge1000", 1000.0, np.inf),
)
ATLAS_PREPROCESS_FAQ_REQUIRED_COLUMNS = (
    "duJy",
    "err",
    "x",
    "y",
    "maj",
    "min",
    "apfit",
    "mag5sig",
    "Sky",
)
ATLAS_PREPROCESS_REQUIRED_COLUMNS = (
    *ATLAS_PREPROCESS_FAQ_REQUIRED_COLUMNS,
    "uJy",
    "F",
)

_TASK_CHECKPOINT_FILENAME = "atlas_forced_phot_tasks.parquet"
_TASK_LEDGER_VERSION = "1"
_REQUEST_VERSION = "atlas-forced-phot-v1"
_HTTP_TIMEOUT_SECONDS = 60.0
_SUBMISSION_AMBIGUITY_GRACE_SECONDS = 300.0
_ACTIVE_TASK_STATUSES = frozenset(
    {"planned", "submitting", "queued", "running", "finished", "download_retry"}
)
_TERMINAL_TASK_STATUSES = frozenset({"downloaded", "no_data", "error"})
_TASK_COLUMNS = (
    "ledger_version",
    "request_key",
    "candidate_id",
    "ra",
    "dec",
    "mjd_min",
    "mjd_max",
    "image_type",
    "use_reduced",
    "batch_comment",
    "task_id",
    "task_url",
    "result_url",
    "status",
    "submitted_unix",
    "started_at",
    "finished_at",
    "downloaded_unix",
    "last_polled_unix",
    "updated_unix",
    "attempts",
    "queue_position",
    "error_message",
    "output_path",
)


def parse_atlas_result(text: str) -> pd.DataFrame:
    """Parse a complete ATLAS result table without discarding its header.

    ATLAS names the first header field ``###MJD``.  It is a real table header,
    not a disposable comment.  The returned frame retains every supplied row
    and column, with only that header name normalized to ``MJD``.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ATLAS result is empty and has no table header")

    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        first_field = line.lstrip().split(maxsplit=1)[0] if line.strip() else ""
        if first_field in {"###MJD", "MJD"}:
            header_index = index
            break
    if header_index is None:
        raise ValueError("ATLAS result is missing the ###MJD table header")

    table_text = "\n".join(lines[header_index:])
    try:
        result = pd.read_csv(StringIO(table_text), sep=r"\s+")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("ATLAS result contains no parseable table") from exc
    if "###MJD" in result.columns:
        result = result.rename(columns={"###MJD": "MJD"})
    if "MJD" not in result.columns:
        raise ValueError("ATLAS result table does not contain an MJD column")
    return result


def _atlas_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _append_atlas_reject_reason(
    reasons: pd.Series,
    rejected: pd.Series,
    reason: str,
) -> None:
    rejected = rejected.fillna(True).astype(bool)
    first = rejected & reasons.eq("")
    later = rejected & reasons.ne("")
    reasons.loc[first] = reason
    reasons.loc[later] = reasons.loc[later] + ";" + reason


def _atlas_reduced_image_mask(frame: pd.DataFrame) -> pd.Series:
    """Resolve reduced/direct provenance without treating unknown data as safe."""
    if "atlas_image_type" in frame.columns:
        values = frame["atlas_image_type"].astype(str).str.strip().str.lower()
        return values.eq("reduced")

    if "atlas_use_reduced" in frame.columns:
        values = frame["atlas_use_reduced"]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)
        normalized = values.astype(str).str.strip().str.lower()
        return normalized.isin({"true", "1", "yes"})

    attr_modes = frame.attrs.get("atlas_image_types")
    if isinstance(attr_modes, str):
        attr_modes = [attr_modes]
    if isinstance(attr_modes, (list, tuple, set)):
        modes = {str(value).strip().lower() for value in attr_modes}
        if modes == {"reduced"}:
            return pd.Series(True, index=frame.index, dtype=bool)

    attr_mode = str(frame.attrs.get("atlas_image_type", "")).strip().lower()
    if attr_mode == "reduced":
        return pd.Series(True, index=frame.index, dtype=bool)

    raise ValueError(
        "Cannot verify that this is reduced/direct ATLAS photometry: expected "
        "an atlas_image_type/atlas_use_reduced column or reduced-image metadata"
    )


def preprocess_atlas_frame(
    frame: pd.DataFrame,
    *,
    snr_min: float = ATLAS_PREPROCESS_DEFAULT_SNR_MIN,
    filters: Iterable[str] = ATLAS_PREPROCESS_DEFAULT_FILTERS,
) -> pd.DataFrame:
    """Return all ATLAS rows with shared quality flags and clean magnitudes."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    snr_min = float(snr_min)
    if not math.isfinite(snr_min) or snr_min < 0:
        raise ValueError("snr_min must be a finite non-negative number")

    allowed_filters = tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in filters
            if str(value).strip()
        )
    )
    if not allowed_filters:
        raise ValueError("At least one ATLAS filter must be selected")

    missing = [
        column
        for column in ATLAS_PREPROCESS_REQUIRED_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError("Missing required ATLAS columns: " + ", ".join(missing))

    out = frame.copy()
    reasons = pd.Series("", index=out.index, dtype="object")

    dujy = _atlas_numeric(out, "duJy")
    err = _atlas_numeric(out, "err")
    x = _atlas_numeric(out, "x")
    y = _atlas_numeric(out, "y")
    maj = _atlas_numeric(out, "maj")
    minor = _atlas_numeric(out, "min")
    apfit = _atlas_numeric(out, "apfit")
    mag5sig = _atlas_numeric(out, "mag5sig")
    sky = _atlas_numeric(out, "Sky")
    ujy = _atlas_numeric(out, "uJy")
    chi_n = (
        pd.to_numeric(out["chi/N"], errors="coerce")
        if "chi/N" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    obs = (
        out["Obs"].fillna("").astype(str).str.strip()
        if "Obs" in out.columns
        else pd.Series("", index=out.index, dtype=str)
    )
    site_code = obs.str.slice(0, 2).where(obs.str.slice(0, 2).str.fullmatch(r"\d{2}"), "")
    camera = obs.str.slice(2, 3).where(obs.str.len() >= 3, "")
    time_col = _first_present(out, ("MJD", "mjd"))
    atlas_mjd = (
        pd.to_numeric(out[time_col], errors="coerce")
        if time_col is not None
        else pd.Series(np.nan, index=out.index, dtype=float)
    )

    faq_checks = (
        (dujy < 10000, "duJy_not_lt_10000"),
        (err == 0, "err_not_zero"),
        ((x > 100) & (x < 10460), "x_outside_100_10460"),
        ((y > 100) & (y < 10460), "y_outside_100_10460"),
        ((maj > 1.6) & (maj < 5), "maj_outside_1p6_5"),
        ((minor > 1.6) & (minor < 5), "min_outside_1p6_5"),
        ((apfit > -1) & (apfit < -0.1), "apfit_outside_minus1_minus0p1"),
        (mag5sig > 17, "mag5sig_not_gt_17"),
        (sky > 17, "Sky_not_gt_17"),
    )

    faq_good = pd.Series(True, index=out.index, dtype=bool)
    for accepted, reason in faq_checks:
        accepted = accepted.fillna(False).astype(bool)
        faq_good &= accepted
        _append_atlas_reject_reason(reasons, ~accepted, reason)

    normalized_filter = out["F"].astype(str).str.strip().str.lower()
    filter_good = normalized_filter.isin(allowed_filters)
    _append_atlas_reject_reason(
        reasons,
        ~filter_good,
        "filter_not_" + "_or_".join(allowed_filters),
    )

    reduced_good = _atlas_reduced_image_mask(out)
    _append_atlas_reject_reason(
        reasons,
        ~reduced_good,
        "image_type_not_reduced",
    )

    positive_ujy = np.isfinite(ujy) & (ujy > 0)
    positive_dujy = np.isfinite(dujy) & (dujy > 0)
    flux_good = positive_ujy & positive_dujy
    _append_atlas_reject_reason(
        reasons,
        ~positive_ujy,
        "uJy_not_positive_or_finite",
    )
    _append_atlas_reject_reason(
        reasons,
        ~positive_dujy,
        "duJy_not_positive_or_finite",
    )

    flux_snr = pd.Series(np.nan, index=out.index, dtype=float)
    flux_snr.loc[positive_dujy] = (
        ujy.loc[positive_dujy] / dujy.loc[positive_dujy]
    )
    snr_good = np.isfinite(flux_snr) & (flux_snr >= snr_min)
    _append_atlas_reject_reason(
        reasons,
        flux_good & ~snr_good,
        f"flux_snr_lt_{snr_min:g}",
    )

    atlas_good = faq_good & filter_good & reduced_good & flux_good & snr_good
    out["atlas_faq_good"] = faq_good
    out["atlas_filter_good"] = filter_good
    out["atlas_reduced_good"] = reduced_good
    out["atlas_flux_good"] = flux_good
    out["atlas_snr_good"] = snr_good
    out["atlas_good"] = atlas_good
    out["flux_snr"] = flux_snr
    out["atlas_row_id"] = np.arange(len(out), dtype=np.int64)
    out["atlas_mjd"] = atlas_mjd
    out["atlas_filter"] = normalized_filter
    out["atlas_chi_n"] = chi_n
    out["atlas_obs"] = obs
    out["atlas_obs_site_code"] = site_code
    out["atlas_obs_site"] = site_code.map(ATLAS_OBS_SITE_NAMES).fillna("unknown")
    out["atlas_camera"] = camera
    out["atlas_flux_ujy"] = ujy
    out["atlas_flux_error_formal_ujy"] = dujy
    out["m_clean"] = np.nan
    out["dm_clean"] = np.nan
    out.loc[atlas_good, "m_clean"] = (
        23.9 - 2.5 * np.log10(ujy.loc[atlas_good])
    )
    out.loc[atlas_good, "dm_clean"] = (
        (2.5 / np.log(10.0))
        * dujy.loc[atlas_good]
        / ujy.loc[atlas_good]
    )
    out["atlas_reject_reason"] = reasons

    out.attrs.update(frame.attrs)
    out.attrs.update(
        {
            "atlas_preprocess_version": ATLAS_PREPROCESS_VERSION,
            "atlas_preprocess_snr_min": snr_min,
            "atlas_preprocess_filters": list(allowed_filters),
            "atlas_preprocess_rows_total": int(len(out)),
            "atlas_preprocess_rows_good": int(atlas_good.sum()),
        }
    )
    return out


def atlas_science_view(
    frame: pd.DataFrame,
    *,
    snr_min: float = ATLAS_PREPROCESS_DEFAULT_SNR_MIN,
    filters: Iterable[str] = ATLAS_PREPROCESS_DEFAULT_FILTERS,
) -> pd.DataFrame:
    """Return accepted reduced/direct detections with cleaned canonical columns.

    Raw ``m``/``dm`` columns remain untouched.  ``mag``/``mag_err`` are the
    positive-flux AB measurements used by all scientific consumers.
    """
    flagged = preprocess_atlas_frame(frame, snr_min=snr_min, filters=filters)
    science = flagged.loc[flagged["atlas_good"]].copy()

    raw_mag_col = _first_present(science, ("m", "mag"))
    raw_err_col = _first_present(science, ("dm", "mag_err", "magerr"))
    if raw_mag_col is not None:
        science["mag_raw"] = science[raw_mag_col]
    if raw_err_col is not None:
        science["mag_err_raw"] = science[raw_err_col]

    time_col = _first_present(science, ("mjd", "MJD"))
    filter_col = _first_present(science, ("filter", "F"))
    if time_col is not None:
        science["mjd"] = pd.to_numeric(science[time_col], errors="coerce")
    if filter_col is not None:
        science["filter"] = (
            science[filter_col].astype(str).str.strip().str.lower()
        )
    science["mag"] = pd.to_numeric(science["m_clean"], errors="coerce")
    science["mag_err"] = pd.to_numeric(science["dm_clean"], errors="coerce")
    science.attrs.update(flagged.attrs)
    return science


def _aligned_array(
    values: Iterable[object] | np.ndarray | pd.Series,
    length: int,
    *,
    name: str,
    dtype: object,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or len(array) != int(length):
        raise ValueError(f"{name} must be a one-dimensional array aligned to the ATLAS frame")
    return array


def _iterative_robust_location_scatter(
    values: np.ndarray,
    *,
    clip_sigma: float,
    max_iterations: int,
) -> tuple[float, float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if not np.any(keep):
        return np.nan, np.nan, keep
    for _ in range(max(1, int(max_iterations))):
        selected = values[keep]
        location = float(np.nanmedian(selected))
        scatter = float(1.4826 * np.nanmedian(np.abs(selected - location)))
        if not np.isfinite(scatter) or scatter <= 0.0:
            break
        next_keep = np.isfinite(values) & (np.abs(values - location) <= float(clip_sigma) * scatter)
        if np.array_equal(next_keep, keep):
            keep = next_keep
            break
        keep = next_keep
        if not np.any(keep):
            break
    if not np.any(keep):
        return np.nan, np.nan, keep
    selected = values[keep]
    location = float(np.nanmedian(selected))
    scatter = float(1.4826 * np.nanmedian(np.abs(selected - location)))
    if not np.isfinite(scatter):
        scatter = np.nan
    return location, scatter, keep


def estimate_atlas_noise_model(
    frame: pd.DataFrame,
    *,
    calibration_mask: Iterable[bool] | np.ndarray | pd.Series,
    reference_flux_ujy: Iterable[float] | np.ndarray | pd.Series | None = None,
    group_columns: tuple[str, ...] = ("atlas_filter", "atlas_obs_site_code"),
    include_band_fallback: bool = True,
    min_points: int = 30,
    min_time_span_days: float = 30.0,
    clip_sigma: float = 5.0,
    max_iterations: int = 5,
) -> pd.DataFrame:
    """Estimate empirical ATLAS flux-error floors from caller-defined quiet data.

    The caller owns ``calibration_mask`` because only the scientific analysis
    can distinguish quiescent measurements from real variability.  The raw
    ``duJy`` values are never modified.  Site-level estimates are accompanied
    by optional passband-level fallback estimates.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = {
        "atlas_good",
        "atlas_filter",
        "atlas_obs_site_code",
        "atlas_flux_ujy",
        "atlas_flux_error_formal_ujy",
        "atlas_mjd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("ATLAS frame is missing preprocessing columns: " + ", ".join(missing))
    if not group_columns or any(column not in frame.columns for column in group_columns):
        raise ValueError("group_columns must name existing ATLAS preprocessing columns")
    if int(min_points) < 2:
        raise ValueError("min_points must be at least 2")
    if not math.isfinite(float(min_time_span_days)) or float(min_time_span_days) < 0.0:
        raise ValueError("min_time_span_days must be finite and non-negative")
    if not math.isfinite(float(clip_sigma)) or float(clip_sigma) <= 0.0:
        raise ValueError("clip_sigma must be finite and positive")

    calibration = _aligned_array(
        calibration_mask,
        len(frame),
        name="calibration_mask",
        dtype=bool,
    )
    reference = None
    if reference_flux_ujy is not None:
        reference = _aligned_array(
            reference_flux_ujy,
            len(frame),
            name="reference_flux_ujy",
            dtype=float,
        )

    work = frame.copy()
    work["_atlas_calibration_mask"] = calibration
    if reference is not None:
        work["_atlas_reference_flux_ujy"] = reference

    rows: list[dict[str, object]] = []

    def append_groups(columns: tuple[str, ...], scope: str) -> None:
        grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
        for key, group in work.groupby(grouper, sort=True, dropna=False):
            keys = key if isinstance(key, tuple) else (key,)
            values = dict(zip(columns, keys))
            flux = pd.to_numeric(group["atlas_flux_ujy"], errors="coerce").to_numpy(dtype=float)
            formal = pd.to_numeric(
                group["atlas_flux_error_formal_ujy"], errors="coerce"
            ).to_numpy(dtype=float)
            mjd = pd.to_numeric(group["atlas_mjd"], errors="coerce").to_numpy(dtype=float)
            good = group["atlas_good"].fillna(False).to_numpy(dtype=bool)
            selected = group["_atlas_calibration_mask"].to_numpy(dtype=bool)
            finite = good & selected & np.isfinite(flux) & np.isfinite(formal) & (formal > 0.0)
            if reference is not None:
                group_reference = pd.to_numeric(
                    group["_atlas_reference_flux_ujy"], errors="coerce"
                ).to_numpy(dtype=float)
                finite &= np.isfinite(group_reference)
                residual = flux - group_reference
                reference_mode = "supplied"
            else:
                residual = flux.copy()
                reference_mode = "group_median"

            calibration_values = residual[finite]
            location, scatter, robust_keep = _iterative_robust_location_scatter(
                calibration_values,
                clip_sigma=float(clip_sigma),
                max_iterations=int(max_iterations),
            )
            selected_times = mjd[finite]
            selected_formal = formal[finite]
            n_calibration = int(len(calibration_values))
            n_used = int(np.sum(robust_keep))
            time_span = (
                float(np.nanmax(selected_times) - np.nanmin(selected_times))
                if np.sum(np.isfinite(selected_times)) >= 2
                else 0.0
            )
            median_formal = (
                float(np.nanmedian(selected_formal[robust_keep]))
                if n_used and len(selected_formal) == len(robust_keep)
                else np.nan
            )
            median_formal_variance = (
                float(np.nanmedian(np.square(selected_formal[robust_keep])))
                if n_used and len(selected_formal) == len(robust_keep)
                else np.nan
            )
            usable = True
            status = "ok"
            if n_used < int(min_points):
                usable = False
                status = "insufficient_calibration_points"
            elif time_span < float(min_time_span_days):
                usable = False
                status = "insufficient_time_span"
            elif not np.isfinite(scatter) or not np.isfinite(median_formal_variance):
                usable = False
                status = "nonfinite_scatter"
            floor = (
                float(np.sqrt(max(float(scatter) ** 2 - median_formal_variance, 0.0)))
                if usable
                else np.nan
            )
            row: dict[str, object] = {
                "atlas_noise_model_version": ATLAS_NOISE_MODEL_VERSION,
                "atlas_noise_scope": scope,
                "atlas_noise_status": status,
                "atlas_noise_usable": bool(usable),
                "atlas_noise_reference_mode": reference_mode,
                "atlas_noise_n_total": int(len(group)),
                "atlas_noise_n_good": int(np.sum(good)),
                "atlas_noise_n_calibration": n_calibration,
                "atlas_noise_n_used": n_used,
                "atlas_noise_n_clipped": max(n_calibration - n_used, 0),
                "atlas_noise_time_span_days": time_span,
                "atlas_noise_robust_location_ujy": location,
                "atlas_noise_robust_scatter_ujy": scatter,
                "atlas_noise_median_formal_error_ujy": median_formal,
                "atlas_noise_floor_ujy": floor,
            }
            row.update(values)
            if "atlas_filter" not in row:
                row["atlas_filter"] = ""
            if "atlas_obs_site_code" not in row:
                row["atlas_obs_site_code"] = ""
            row["atlas_noise_group"] = (
                f"{row['atlas_filter']}:{row['atlas_obs_site_code']}"
                if scope == "site"
                else f"{row['atlas_filter']}:all_sites"
            )
            rows.append(row)

    append_groups(tuple(group_columns), "site")
    if include_band_fallback and "atlas_filter" in group_columns:
        append_groups(("atlas_filter",), "band")
    return pd.DataFrame(rows)


def apply_atlas_noise_model(frame: pd.DataFrame, noise_model: pd.DataFrame) -> pd.DataFrame:
    """Attach effective uncertainties from an empirical ATLAS noise model."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(noise_model, pd.DataFrame):
        raise TypeError("frame and noise_model must be pandas DataFrames")
    required = {
        "atlas_filter",
        "atlas_obs_site_code",
        "atlas_flux_ujy",
        "atlas_flux_error_formal_ujy",
        "atlas_good",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("ATLAS frame is missing preprocessing columns: " + ", ".join(missing))

    out = frame.copy()
    n_rows = len(out)
    floor = np.full(n_rows, np.nan, dtype=float)
    location = np.full(n_rows, np.nan, dtype=float)
    scope = np.full(n_rows, "", dtype=object)
    group_name = np.full(n_rows, "", dtype=object)
    status = np.full(n_rows, "no_usable_noise_model", dtype=object)
    usable = np.zeros(n_rows, dtype=bool)

    site_models: dict[tuple[str, str], object] = {}
    band_models: dict[str, object] = {}
    if not noise_model.empty:
        for row in noise_model.itertuples(index=False):
            if not bool(getattr(row, "atlas_noise_usable", False)):
                continue
            band = str(getattr(row, "atlas_filter", ""))
            model_scope = str(getattr(row, "atlas_noise_scope", ""))
            if model_scope == "site":
                site_models[(band, str(getattr(row, "atlas_obs_site_code", "")))] = row
            elif model_scope == "band":
                band_models[band] = row

    bands = out["atlas_filter"].fillna("").astype(str).to_numpy()
    sites = out["atlas_obs_site_code"].fillna("").astype(str).to_numpy()
    good = out["atlas_good"].fillna(False).to_numpy(dtype=bool)
    for index, (band, site) in enumerate(zip(bands, sites)):
        if not good[index]:
            status[index] = "rejected_by_epoch_quality"
            continue
        model = site_models.get((band, site))
        if model is None:
            model = band_models.get(band)
        if model is None:
            continue
        floor[index] = float(getattr(model, "atlas_noise_floor_ujy"))
        location[index] = float(getattr(model, "atlas_noise_robust_location_ujy"))
        scope[index] = str(getattr(model, "atlas_noise_scope"))
        group_name[index] = str(getattr(model, "atlas_noise_group"))
        status[index] = str(getattr(model, "atlas_noise_status"))
        usable[index] = True

    formal = pd.to_numeric(
        out["atlas_flux_error_formal_ujy"], errors="coerce"
    ).to_numpy(dtype=float)
    flux = pd.to_numeric(out["atlas_flux_ujy"], errors="coerce").to_numpy(dtype=float)
    effective = np.sqrt(np.square(formal) + np.square(floor))
    effective[~usable | ~np.isfinite(effective) | (effective <= 0.0)] = np.nan
    mag_error_effective = np.full(n_rows, np.nan, dtype=float)
    positive = usable & np.isfinite(flux) & (flux > 0.0) & np.isfinite(effective)
    mag_error_effective[positive] = (
        (2.5 / np.log(10.0)) * effective[positive] / flux[positive]
    )

    out["atlas_noise_model_version"] = ATLAS_NOISE_MODEL_VERSION
    out["atlas_noise_model_usable"] = usable
    out["atlas_noise_status"] = status
    out["atlas_noise_scope"] = scope
    out["atlas_noise_group"] = group_name
    out["atlas_noise_location_ujy"] = location
    out["atlas_noise_floor_ujy"] = floor
    out["atlas_flux_error_eff_ujy"] = effective
    out["atlas_mag_error_eff"] = mag_error_effective
    out.attrs.update(frame.attrs)
    out.attrs["atlas_noise_model_version"] = ATLAS_NOISE_MODEL_VERSION
    return out


def atlas_huber_weights(
    residuals: Iterable[float] | np.ndarray | pd.Series,
    uncertainties: Iterable[float] | np.ndarray | pd.Series,
    *,
    tuning: float = 5.0,
) -> np.ndarray:
    """Return fixed Huber weights for already defined ATLAS residuals."""
    if not math.isfinite(float(tuning)) or float(tuning) <= 0.0:
        raise ValueError("tuning must be finite and positive")
    residual = np.asarray(residuals, dtype=float)
    uncertainty = np.asarray(uncertainties, dtype=float)
    if residual.shape != uncertainty.shape:
        raise ValueError("residuals and uncertainties must have matching shapes")
    weights = np.zeros_like(residual, dtype=float)
    valid = np.isfinite(residual) & np.isfinite(uncertainty) & (uncertainty > 0.0)
    standardized = np.full_like(residual, np.nan, dtype=float)
    standardized[valid] = np.abs(residual[valid] / uncertainty[valid])
    weights[valid] = 1.0
    tail = valid & (standardized > float(tuning))
    weights[tail] = float(tuning) / standardized[tail]
    return weights


def summarize_atlas_residuals(
    frame: pd.DataFrame,
    *,
    residual_flux_ujy: Iterable[float] | np.ndarray | pd.Series,
    robust_weights: Iterable[float] | np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Summarize model residuals by ATLAS passband, site, and ``chi/N`` bin."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = {
        "atlas_filter",
        "atlas_obs_site_code",
        "atlas_obs_site",
        "atlas_chi_n",
        "atlas_flux_error_formal_ujy",
        "atlas_flux_error_eff_ujy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("ATLAS frame is missing calibrated columns: " + ", ".join(missing))
    residual = _aligned_array(
        residual_flux_ujy,
        len(frame),
        name="residual_flux_ujy",
        dtype=float,
    )
    weights = (
        np.ones(len(frame), dtype=float)
        if robust_weights is None
        else _aligned_array(robust_weights, len(frame), name="robust_weights", dtype=float)
    )
    formal = pd.to_numeric(
        frame["atlas_flux_error_formal_ujy"], errors="coerce"
    ).to_numpy(dtype=float)
    effective = pd.to_numeric(
        frame["atlas_flux_error_eff_ujy"], errors="coerce"
    ).to_numpy(dtype=float)
    floor = (
        pd.to_numeric(frame["atlas_noise_floor_ujy"], errors="coerce").to_numpy(dtype=float)
        if "atlas_noise_floor_ujy" in frame.columns
        else np.full(len(frame), np.nan, dtype=float)
    )
    chi_n = pd.to_numeric(frame["atlas_chi_n"], errors="coerce").to_numpy(dtype=float)
    pre_z = np.divide(
        residual,
        formal,
        out=np.full_like(residual, np.nan),
        where=np.isfinite(formal) & (formal > 0.0),
    )
    post_z = np.divide(
        residual,
        effective,
        out=np.full_like(residual, np.nan),
        where=np.isfinite(effective) & (effective > 0.0),
    )
    bands = frame["atlas_filter"].fillna("").astype(str).to_numpy()
    site_codes = frame["atlas_obs_site_code"].fillna("").astype(str).to_numpy()
    site_names = frame["atlas_obs_site"].fillna("unknown").astype(str).to_numpy()
    rows: list[dict[str, object]] = []

    def append_row(mask: np.ndarray, *, band: str, site_code: str, site_name: str, scope: str, chi_bin: str) -> None:
        valid = mask & np.isfinite(residual)
        n_points = int(np.sum(valid))
        if not n_points:
            return
        selected_residual = residual[valid]
        residual_location = float(np.nanmedian(selected_residual))
        residual_scatter = float(
            1.4826 * np.nanmedian(np.abs(selected_residual - residual_location))
        )
        selected_pre = pre_z[valid & np.isfinite(pre_z)]
        selected_post = post_z[valid & np.isfinite(post_z)]
        selected_chi = chi_n[valid & np.isfinite(chi_n)]
        selected_weights = weights[valid & np.isfinite(weights)]
        rows.append(
            {
                "atlas_noise_model_version": ATLAS_NOISE_MODEL_VERSION,
                "atlas_filter": band,
                "atlas_obs_site_code": site_code,
                "atlas_obs_site": site_name,
                "diagnostic_scope": scope,
                "chi_n_bin": chi_bin,
                "n_points": n_points,
                "median_formal_error_ujy": float(np.nanmedian(formal[valid])),
                "median_effective_error_ujy": float(np.nanmedian(effective[valid])),
                "median_noise_floor_ujy": float(np.nanmedian(floor[valid])),
                "chi_n_median": float(np.nanmedian(selected_chi)) if selected_chi.size else np.nan,
                "chi_n_p90": float(np.nanpercentile(selected_chi, 90.0)) if selected_chi.size else np.nan,
                "chi_n_p99": float(np.nanpercentile(selected_chi, 99.0)) if selected_chi.size else np.nan,
                "residual_median_ujy": residual_location,
                "residual_robust_scatter_ujy": residual_scatter,
                "reduced_chi2_formal": (
                    float(np.sum(np.square(selected_pre)) / max(len(selected_pre) - 2, 1))
                    if selected_pre.size
                    else np.nan
                ),
                "reduced_chi2_effective": (
                    float(np.sum(np.square(selected_post)) / max(len(selected_post) - 2, 1))
                    if selected_post.size
                    else np.nan
                ),
                "fraction_gt3_formal": float(np.mean(np.abs(selected_pre) > 3.0)) if selected_pre.size else np.nan,
                "fraction_gt5_formal": float(np.mean(np.abs(selected_pre) > 5.0)) if selected_pre.size else np.nan,
                "fraction_gt10_formal": float(np.mean(np.abs(selected_pre) > 10.0)) if selected_pre.size else np.nan,
                "fraction_gt3_effective": (
                    float(np.mean(np.abs(selected_post) > 3.0))
                    if selected_post.size
                    else np.nan
                ),
                "fraction_gt5_effective": (
                    float(np.mean(np.abs(selected_post) > 5.0))
                    if selected_post.size
                    else np.nan
                ),
                "fraction_gt10_effective": (
                    float(np.mean(np.abs(selected_post) > 10.0))
                    if selected_post.size
                    else np.nan
                ),
                "median_robust_weight": float(np.nanmedian(selected_weights)) if selected_weights.size else np.nan,
                "n_downweighted": int(np.sum(selected_weights < 1.0)) if selected_weights.size else 0,
                "n_excluded": int(np.sum(selected_weights <= 0.0)) if selected_weights.size else 0,
            }
        )

    for band in sorted(set(bands)):
        for site_code in sorted(set(site_codes[bands == band])):
            base = (bands == band) & (site_codes == site_code)
            names = site_names[base]
            site_name = str(names[0]) if names.size else ATLAS_OBS_SITE_NAMES.get(site_code, "unknown")
            append_row(base, band=band, site_code=site_code, site_name=site_name, scope="site", chi_bin="all")
            for label, lower, upper in ATLAS_CHI_N_BINS:
                chi_mask = base & np.isfinite(chi_n) & (chi_n >= lower) & (chi_n < upper)
                append_row(
                    chi_mask,
                    band=band,
                    site_code=site_code,
                    site_name=site_name,
                    scope="chi_n_bin",
                    chi_bin=label,
                )
    return pd.DataFrame(rows)


def summarize_atlas_lc(lc: pd.DataFrame) -> dict[str, object]:
    """Return cleaned ATLAS summary fields and preprocessing audit counts."""
    summary: dict[str, object] = {
        "atlas_has_phot": False,
        "atlas_n_det_cyan": 0,
        "atlas_n_det_orange": 0,
        "atlas_cyan_range": np.nan,
        "atlas_orange_range": np.nan,
        "atlas_preprocess_version": ATLAS_PREPROCESS_VERSION,
        "atlas_n_raw": 0,
        "atlas_n_good": 0,
        "atlas_n_rejected": 0,
    }
    if lc is None or lc.empty:
        return summary

    phot = atlas_science_view(lc)
    summary["atlas_n_raw"] = int(len(lc))
    summary["atlas_n_good"] = int(len(phot))
    summary["atlas_n_rejected"] = int(len(lc) - len(phot))
    summary["atlas_has_phot"] = bool(len(phot))
    if phot.empty:
        return summary

    filters = phot["filter"]
    magnitudes = phot["mag"]
    cyan = magnitudes.loc[filters.eq("c")].dropna()
    orange = magnitudes.loc[filters.eq("o")].dropna()
    summary["atlas_n_det_cyan"] = int(len(cyan))
    summary["atlas_n_det_orange"] = int(len(orange))
    if len(cyan) >= 2:
        summary["atlas_cyan_range"] = round(float(cyan.max() - cyan.min()), 4)
    if len(orange) >= 2:
        summary["atlas_orange_range"] = round(float(orange.max() - orange.min()), 4)
    return summary


def query_atlas_forced_phot(
    df: pd.DataFrame,
    *,
    token: str | None = None,
    output_dir: Path | str | None = None,
    results_root: Path | str | None = None,
    refresh_cache: bool = False,
    task_checkpoint: Path | str | None = None,
    batch_size: int = 100,
    poll_interval: float = 60.0,
    mjd_min: float = 57000.0,
    mjd_max: float | None = None,
    image_type: str = "reduced",
    max_wait_seconds: float | None = None,
    submit_only: bool = False,
    session: Any | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fetch ATLAS forced photometry in resumable batches.

    Parameters are part of each request's persistent identity.  Re-running the
    same call first consumes cached parquets and saved task URLs, and submits
    only requests with neither.  ``max_wait_seconds`` and ``submit_only`` may
    return rows with null summary fields; their queued tasks remain resumable.
    """
    out = _prepare_input_frame(df)
    if out.empty:
        return out

    if output_dir is None:
        raise ValueError("output_dir is required for resumable ATLAS forced photometry")
    lc_dir = Path(output_dir).expanduser()
    lc_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = _resolve_results_root(lc_dir, results_root)
    manifest_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(task_checkpoint).expanduser()
        if task_checkpoint is not None
        else lc_dir / _TASK_CHECKPOINT_FILENAME
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = int(batch_size)
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and the ATLAS limit of 100")
    poll_interval = float(poll_interval)
    if not math.isfinite(poll_interval) or poll_interval < 0:
        raise ValueError("poll_interval must be a finite non-negative number")
    mjd_min = float(mjd_min)
    if not math.isfinite(mjd_min):
        raise ValueError("mjd_min must be finite")
    if mjd_max is not None:
        mjd_max = float(mjd_max)
        if not math.isfinite(mjd_max) or mjd_max <= mjd_min:
            raise ValueError("mjd_max must be finite and greater than mjd_min")
    if max_wait_seconds is not None:
        max_wait_seconds = float(max_wait_seconds)
        if not math.isfinite(max_wait_seconds) or max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be finite and non-negative")

    modes = _normalize_image_types(image_type)
    specs = _build_request_specs(out, modes, mjd_min=mjd_min, mjd_max=mjd_max)
    auth_token = token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN")
    owned_session = session is None
    client = requests.Session() if owned_session else session
    started_monotonic = time.monotonic()
    deadline = (
        None
        if max_wait_seconds is None
        else started_monotonic + max_wait_seconds
    )

    try:
        with _task_process_lock(checkpoint_path):
            return _query_locked(
                out,
                specs=specs,
                modes=modes,
                token=auth_token,
                lc_dir=lc_dir,
                results_root=manifest_root,
                checkpoint_path=checkpoint_path,
                refresh_cache=bool(refresh_cache),
                batch_size=batch_size,
                poll_interval=poll_interval,
                deadline=deadline,
                submit_only=bool(submit_only),
                client=client,
                sleep_func=sleep_func,
                progress=progress,
            )
    finally:
        if owned_session and client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def _query_locked(
    out: pd.DataFrame,
    *,
    specs: list[dict[str, object]],
    modes: tuple[str, ...],
    token: str | None,
    lc_dir: Path,
    results_root: Path,
    checkpoint_path: Path,
    refresh_cache: bool,
    batch_size: int,
    poll_interval: float,
    deadline: float | None,
    submit_only: bool,
    client: Any,
    sleep_func: Callable[[float], None],
    progress: Callable[[str], None] | None,
) -> pd.DataFrame:
    ledger = _read_task_ledger(checkpoint_path)
    current_keys = {str(spec["request_key"]) for spec in specs}
    if refresh_cache and not ledger.empty:
        ledger = ledger.loc[~ledger["request_key"].astype(str).isin(current_keys)].copy()
        _write_task_ledger(ledger, checkpoint_path)

    ledger, changed = _seed_ledger_from_cache(
        ledger,
        specs,
        modes=modes,
        lc_dir=lc_dir,
        results_root=results_root,
        refresh_cache=refresh_cache,
    )
    if changed:
        _write_task_ledger(ledger, checkpoint_path)

    work_rows = _ledger_for_keys(ledger, current_keys)
    requires_network = bool(
        work_rows["status"].astype(str).isin(_ACTIVE_TASK_STATUSES).any()
    )
    if requires_network and not token:
        raise ValueError(
            "ATLAS API token is required for uncached or pending work; set "
            "MALCA_ATLAS_TOKEN or pass token="
        )
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    } if token else {}

    ambiguous = work_rows.loc[
        work_rows["status"].astype(str).eq("submitting")
        & work_rows["task_url"].fillna("").astype(str).str.strip().eq("")
    ]
    if not ambiguous.empty:
        remote_tasks = _list_remote_tasks(client, headers=headers)
        ledger, reconciled, deferred = _reconcile_submitting_rows(
            ledger,
            ambiguous,
            remote_tasks,
        )
        _write_task_ledger(ledger, checkpoint_path)
        if reconciled:
            _emit(progress, f"ATLAS: recovered {reconciled} submitted task URL(s) from the remote queue")
        if deferred:
            _emit(
                progress,
                f"ATLAS: deferring {deferred} ambiguous recent submission(s) to avoid duplicate jobs",
            )

    spec_by_key = {str(spec["request_key"]): spec for spec in specs}
    planned_keys = [
        key
        for key in spec_by_key
        if _task_status(ledger, key) == "planned"
    ]
    grouped: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    for key in planned_keys:
        grouped[str(spec_by_key[key]["image_type"])].append(spec_by_key[key])

    for mode in modes:
        mode_specs = grouped.get(mode, [])
        for offset in range(0, len(mode_specs), batch_size):
            batch = mode_specs[offset : offset + batch_size]
            batch_comment = _batch_comment(batch)
            now = time.time()
            for spec in batch:
                key = str(spec["request_key"])
                ledger = _update_task(
                    ledger,
                    key,
                    status="submitting",
                    batch_comment=batch_comment,
                    submitted_unix=now,
                    updated_unix=now,
                    attempts=_task_attempts(ledger, key) + 1,
                    error_message="",
                )
            # The pre-submit write is intentional: if the process dies after
            # server acceptance, the deterministic comment can be reconciled.
            _write_task_ledger(ledger, checkpoint_path)
            _emit(progress, f"ATLAS: submitting {len(batch)} {mode} coordinate(s)")
            items = _submit_batch(
                client,
                batch,
                batch_comment=batch_comment,
                headers=headers,
                deadline=deadline,
                sleep_func=sleep_func,
            )
            if items is None:
                # A 429 consumed the caller's wait budget.  No task was
                # accepted, so these requests can safely return to planned.
                for spec in batch:
                    ledger = _update_task(
                        ledger,
                        str(spec["request_key"]),
                        status="planned",
                        batch_comment="",
                        submitted_unix=np.nan,
                        updated_unix=time.time(),
                    )
                _write_task_ledger(ledger, checkpoint_path)
                return _apply_terminal_summaries(out, ledger, specs, modes, lc_dir)

            assignments = _assign_response_items(batch, items)
            now = time.time()
            for key, item in assignments.items():
                state = _remote_task_state(item)
                ledger = _update_task(
                    ledger,
                    key,
                    status=state,
                    task_id=_remote_task_id(item),
                    task_url=_absolute_task_url(item.get("url")),
                    result_url=str(item.get("result_url") or ""),
                    started_at=str(item.get("starttimestamp") or ""),
                    finished_at=str(item.get("finishtimestamp") or ""),
                    queue_position=item.get("queuepos", np.nan),
                    error_message=str(item.get("error_msg") or ""),
                    updated_unix=now,
                )
            # Every URL from a successful POST is persisted before any poll.
            _write_task_ledger(ledger, checkpoint_path)

    if submit_only:
        pending = _count_active(ledger, current_keys)
        _emit(progress, f"ATLAS: submit-only complete; {pending} task(s) remain pending")
        return _apply_terminal_summaries(out, ledger, specs, modes, lc_dir)

    while True:
        active = _ledger_for_keys(ledger, current_keys)
        active = active.loc[active["status"].astype(str).isin(_ACTIVE_TASK_STATUSES)]
        if active.empty:
            break
        pollable = active.loc[
            active["task_url"].fillna("").astype(str).str.strip().ne("")
        ]
        if pollable.empty:
            # A recent crash-before-response is intentionally left ambiguous
            # for one grace period rather than risking duplicate submissions.
            break
        if deadline is not None and time.monotonic() >= deadline:
            break

        for _, task in pollable.iterrows():
            key = str(task["request_key"])
            status = str(task["status"])
            if status in {"finished", "download_retry"} and str(task.get("result_url") or "").strip():
                ledger = _finish_result_download(
                    ledger,
                    key,
                    client=client,
                    headers=headers,
                    lc_dir=lc_dir,
                    results_root=results_root,
                    checkpoint_path=checkpoint_path,
                )
                continue

            task_url = str(task["task_url"])
            try:
                response = client.get(task_url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
            except Exception as exc:
                ledger = _update_task(
                    ledger,
                    key,
                    last_polled_unix=time.time(),
                    updated_unix=time.time(),
                    error_message=f"poll failed: {_short_error(exc)}",
                )
                _write_task_ledger(ledger, checkpoint_path)
                continue

            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in {401, 403}:
                raise RuntimeError(f"ATLAS task poll authorization failed with HTTP {status_code}")
            if status_code == 404:
                ledger = _update_task(
                    ledger,
                    key,
                    status="error",
                    last_polled_unix=time.time(),
                    updated_unix=time.time(),
                    error_message="ATLAS task URL is no longer available (HTTP 404)",
                )
                _write_task_ledger(ledger, checkpoint_path)
                continue
            if status_code == 429 or status_code >= 500 or status_code == 0:
                ledger = _update_task(
                    ledger,
                    key,
                    last_polled_unix=time.time(),
                    updated_unix=time.time(),
                    error_message=f"transient task poll HTTP {status_code}",
                )
                _write_task_ledger(ledger, checkpoint_path)
                continue
            if status_code != 200:
                raise RuntimeError(_http_error("poll ATLAS task", response))

            try:
                detail = response.json()
            except Exception as exc:
                retained_status = "running" if status == "running" else "queued"
                ledger = _update_task(
                    ledger,
                    key,
                    status=retained_status,
                    last_polled_unix=time.time(),
                    updated_unix=time.time(),
                    error_message=f"transient invalid ATLAS task JSON: {_short_error(exc)}",
                )
                _write_task_ledger(ledger, checkpoint_path)
                continue
            if not isinstance(detail, dict):
                retained_status = "running" if status == "running" else "queued"
                ledger = _update_task(
                    ledger,
                    key,
                    status=retained_status,
                    last_polled_unix=time.time(),
                    updated_unix=time.time(),
                    error_message="transient ATLAS task response was not a JSON object",
                )
                _write_task_ledger(ledger, checkpoint_path)
                continue

            now = time.time()
            server_error = str(detail.get("error_msg") or "").strip()
            finished_at = str(detail.get("finishtimestamp") or "").strip()
            result_url = str(detail.get("result_url") or "").strip()
            if finished_at and server_error:
                next_status = "error"
                error_message = server_error
            elif finished_at and result_url:
                next_status = "finished"
                error_message = ""
            elif finished_at:
                next_status = "error"
                error_message = "ATLAS task finished without a result URL"
            elif detail.get("starttimestamp"):
                next_status = "running"
                error_message = ""
            else:
                next_status = "queued"
                error_message = ""
            ledger = _update_task(
                ledger,
                key,
                status=next_status,
                task_id=_remote_task_id(detail) or task.get("task_id", ""),
                result_url=result_url,
                started_at=str(detail.get("starttimestamp") or ""),
                finished_at=finished_at,
                last_polled_unix=now,
                updated_unix=now,
                queue_position=detail.get("queuepos", np.nan),
                error_message=error_message,
            )
            _write_task_ledger(ledger, checkpoint_path)
            if next_status == "finished":
                ledger = _finish_result_download(
                    ledger,
                    key,
                    client=client,
                    headers=headers,
                    lc_dir=lc_dir,
                    results_root=results_root,
                    checkpoint_path=checkpoint_path,
                )

        remaining = _count_active(ledger, current_keys)
        if not remaining:
            break
        if deadline is not None:
            seconds_left = deadline - time.monotonic()
            if seconds_left <= 0:
                break
            delay = min(poll_interval, seconds_left)
        else:
            delay = poll_interval
        _emit(progress, f"ATLAS: {remaining} task(s) pending; polling again in {delay:g}s")
        if delay > 0:
            sleep_func(delay)

    pending = _count_active(ledger, current_keys)
    errors = int(
        _ledger_for_keys(ledger, current_keys)["status"].astype(str).eq("error").sum()
    )
    if pending:
        _emit(progress, f"ATLAS: returning with {pending} resumable task(s) still pending")
    if errors:
        _emit(progress, f"ATLAS: {errors} task(s) ended in error; their summaries remain null")
    return _apply_terminal_summaries(out, ledger, specs, modes, lc_dir)


def _prepare_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    out = df.copy()
    for column in ATLAS_SUMMARY_COLUMNS:
        out[column] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    if out.empty:
        return out
    if "ra" not in out.columns and "ra_deg" in out.columns:
        out["ra"] = out["ra_deg"]
    if "dec" not in out.columns and "dec_deg" in out.columns:
        out["dec"] = out["dec_deg"]
    if "ra" not in out.columns or "dec" not in out.columns:
        raise ValueError("ATLAS input requires ra/dec or ra_deg/dec_deg columns")
    if "candidate_id" not in out.columns:
        if "asas_sn_id" in out.columns:
            out["candidate_id"] = out["asas_sn_id"].astype(str)
        else:
            out["candidate_id"] = [str(value) for value in out.index]
    candidate_ids = out["candidate_id"].astype(str).str.strip()
    if bool(candidate_ids.eq("").any()):
        raise ValueError("ATLAS input contains a blank candidate_id")
    if bool(candidate_ids.duplicated().any()):
        duplicates = candidate_ids.loc[candidate_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"ATLAS input candidate_id values must be unique: {duplicates[:5]}")
    for candidate_id in candidate_ids:
        if Path(candidate_id).name != candidate_id or candidate_id in {".", ".."}:
            raise ValueError(f"candidate_id is unsafe for an output filename: {candidate_id!r}")
    out["candidate_id"] = candidate_ids

    ra = pd.to_numeric(out["ra"], errors="coerce")
    dec = pd.to_numeric(out["dec"], errors="coerce")
    invalid = ~np.isfinite(ra) | ~np.isfinite(dec) | (ra < 0) | (ra >= 360) | (dec < -90) | (dec > 90)
    if bool(invalid.any()):
        bad = out.loc[invalid, "candidate_id"].astype(str).tolist()
        raise ValueError(f"ATLAS input contains invalid coordinates for: {bad[:5]}")
    out["ra"] = ra.astype(float)
    out["dec"] = dec.astype(float)
    return out


def _normalize_image_types(image_type: str) -> tuple[str, ...]:
    value = str(image_type or "").strip().lower().replace("_", "-")
    aliases = {
        "reduced": ("reduced",),
        "target": ("reduced",),
        "target-image": ("reduced",),
        "difference": ("difference",),
        "diff": ("difference",),
        "difference-image": ("difference",),
        "both": ("reduced", "difference"),
    }
    if value not in aliases:
        raise ValueError("image_type must be 'reduced', 'difference', or 'both'")
    return aliases[value]


def _build_request_specs(
    df: pd.DataFrame,
    modes: Iterable[str],
    *,
    mjd_min: float,
    mjd_max: float | None,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for position in range(len(df)):
        row = df.iloc[position]
        for mode in modes:
            spec = {
                "candidate_id": str(row["candidate_id"]),
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "mjd_min": float(mjd_min),
                "mjd_max": np.nan if mjd_max is None else float(mjd_max),
                "image_type": str(mode),
                "use_reduced": bool(mode == "reduced"),
            }
            spec["request_key"] = _request_key(spec)
            specs.append(spec)
    return specs


def _request_key(spec: dict[str, object]) -> str:
    payload = {
        "version": _REQUEST_VERSION,
        "candidate_id": str(spec["candidate_id"]),
        "ra": f"{float(spec['ra']):.10f}",
        "dec": f"{float(spec['dec']):.10f}",
        "mjd_min": f"{float(spec['mjd_min']):.5f}",
        "mjd_max": None if _is_missing(spec.get("mjd_max")) else f"{float(spec['mjd_max']):.5f}",
        "image_type": str(spec["image_type"]),
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest


def _batch_comment(batch: list[dict[str, object]]) -> str:
    keys = sorted(str(spec["request_key"]) for spec in batch)
    digest = sha256("|".join(keys).encode("ascii")).hexdigest()[:32]
    return f"malca-atlas-{digest}"


def _new_task_row(spec: dict[str, object], *, status: str = "planned") -> dict[str, object]:
    now = time.time()
    return {
        "ledger_version": _TASK_LEDGER_VERSION,
        "request_key": str(spec["request_key"]),
        "candidate_id": str(spec["candidate_id"]),
        "ra": float(spec["ra"]),
        "dec": float(spec["dec"]),
        "mjd_min": float(spec["mjd_min"]),
        "mjd_max": spec.get("mjd_max", np.nan),
        "image_type": str(spec["image_type"]),
        "use_reduced": bool(spec["use_reduced"]),
        "batch_comment": "",
        "task_id": "",
        "task_url": "",
        "result_url": "",
        "status": status,
        "submitted_unix": np.nan,
        "started_at": "",
        "finished_at": "",
        "downloaded_unix": np.nan,
        "last_polled_unix": np.nan,
        "updated_unix": now,
        "attempts": 0,
        "queue_position": np.nan,
        "error_message": "",
        "output_path": "",
    }


def _canonical_task_ledger(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_TASK_COLUMNS))
    ledger = frame.copy()
    for column in _TASK_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = pd.NA
    ledger = ledger[list(_TASK_COLUMNS)]
    for column in (
        "ledger_version", "request_key", "candidate_id", "image_type",
        "batch_comment", "task_id", "task_url", "result_url", "status",
        "started_at", "finished_at", "error_message", "output_path",
    ):
        ledger[column] = ledger[column].astype("string").fillna("").astype(str)
    for column in (
        "ra", "dec", "mjd_min", "mjd_max", "submitted_unix",
        "downloaded_unix", "last_polled_unix", "updated_unix", "attempts",
        "queue_position",
    ):
        ledger[column] = pd.to_numeric(ledger[column], errors="coerce")
    ledger["use_reduced"] = ledger["use_reduced"].map(_coerce_bool)
    ledger = ledger.loc[ledger["request_key"].str.strip().ne("")]
    ledger = ledger.drop_duplicates(subset=["request_key"], keep="last")
    return ledger.reset_index(drop=True)


def _read_task_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _canonical_task_ledger(None)
    try:
        return _canonical_task_ledger(pd.read_parquet(path))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read ATLAS task checkpoint {path}; refusing to risk duplicate submissions: {_short_error(exc)}"
        ) from exc


def _write_task_ledger(ledger: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_task_ledger(ledger)
    _atomic_write_parquet(canonical, path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression=PARQUET_CACHE_COMPRESSION)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def _task_process_lock(checkpoint_path: Path):
    lock_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="ascii") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _upsert_task_rows(ledger: pd.DataFrame, rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _canonical_task_ledger(ledger)
    new_rows = pd.DataFrame(rows)
    combined = new_rows if ledger.empty else pd.concat([ledger, new_rows], ignore_index=True)
    return _canonical_task_ledger(combined)


def _update_task(ledger: pd.DataFrame, request_key: str, **updates: object) -> pd.DataFrame:
    matches = ledger.index[ledger["request_key"].astype(str).eq(str(request_key))].tolist()
    if not matches:
        raise KeyError(f"ATLAS task ledger has no request_key {request_key}")
    row_index = matches[-1]
    for column, value in updates.items():
        if column not in _TASK_COLUMNS:
            raise KeyError(f"Unknown ATLAS task ledger column: {column}")
        ledger.at[row_index, column] = value
    return _canonical_task_ledger(ledger)


def _task_status(ledger: pd.DataFrame, request_key: str) -> str:
    rows = ledger.loc[ledger["request_key"].astype(str).eq(str(request_key))]
    return "" if rows.empty else str(rows.iloc[-1]["status"])


def _task_attempts(ledger: pd.DataFrame, request_key: str) -> int:
    rows = ledger.loc[ledger["request_key"].astype(str).eq(str(request_key))]
    if rows.empty:
        return 0
    value = pd.to_numeric(pd.Series([rows.iloc[-1].get("attempts")]), errors="coerce").iloc[0]
    return 0 if pd.isna(value) else int(value)


def _ledger_for_keys(ledger: pd.DataFrame, request_keys: set[str]) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    return ledger.loc[ledger["request_key"].astype(str).isin(request_keys)].copy()


def _seed_ledger_from_cache(
    ledger: pd.DataFrame,
    specs: list[dict[str, object]],
    *,
    modes: tuple[str, ...],
    lc_dir: Path,
    results_root: Path,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, bool]:
    changed = False
    file_cache: dict[str, pd.DataFrame | None] = {}
    for spec in specs:
        key = str(spec["request_key"])
        candidate_id = str(spec["candidate_id"])
        mode = str(spec["image_type"])
        path = _candidate_lc_path(lc_dir, candidate_id)
        if candidate_id not in file_cache:
            file_cache[candidate_id] = _read_parquet_if_valid(path, allow_empty=True)
        cached = file_cache[candidate_id]
        file_has_mode = (
            not refresh_cache
            and cached is not None
            and _cached_frame_satisfies_mode(cached, mode, requested_modes=modes)
        )
        existing = ledger.loc[ledger["request_key"].astype(str).eq(key)]
        if file_has_mode:
            _ensure_manifest(results_root, path, candidate_id)
            cached_status = "no_data" if cached is not None and cached.empty else "downloaded"
            if existing.empty:
                row = _new_task_row(spec, status=cached_status)
                row.update(
                    {
                        "downloaded_unix": time.time(),
                        "updated_unix": time.time(),
                        "output_path": str(path),
                    }
                )
                ledger = _upsert_task_rows(ledger, [row])
            else:
                ledger = _update_task(
                    ledger,
                    key,
                    status=cached_status,
                    downloaded_unix=time.time(),
                    updated_unix=time.time(),
                    output_path=str(path),
                    error_message="",
                )
            changed = True
            continue

        if existing.empty:
            ledger = _upsert_task_rows(ledger, [_new_task_row(spec)])
            changed = True
            continue

        status = str(existing.iloc[-1]["status"])
        if status == "downloaded" and not path.exists():
            task_url = str(existing.iloc[-1].get("task_url") or "").strip()
            result_url = str(existing.iloc[-1].get("result_url") or "").strip()
            if result_url:
                replacement = "finished"
            elif task_url:
                replacement = "queued"
            else:
                replacement = "error"
            ledger = _update_task(
                ledger,
                key,
                status=replacement,
                updated_unix=time.time(),
                error_message=(
                    "cached ATLAS parquet is missing and no saved task URL can recover it"
                    if replacement == "error"
                    else ""
                ),
            )
            changed = True
        elif status not in _ACTIVE_TASK_STATUSES | _TERMINAL_TASK_STATUSES:
            raise RuntimeError(f"Unknown ATLAS task status {status!r} for {candidate_id}")
    return ledger, changed


def _cached_frame_satisfies_mode(
    frame: pd.DataFrame,
    mode: str,
    *,
    requested_modes: tuple[str, ...],
) -> bool:
    attr_modes = frame.attrs.get("atlas_image_types")
    if isinstance(attr_modes, str):
        attr_modes = [attr_modes]
    if not isinstance(attr_modes, (list, tuple, set)):
        single_attr_mode = frame.attrs.get("atlas_image_type")
        attr_modes = [single_attr_mode] if isinstance(single_attr_mode, str) else []
    normalized_attr_modes = {str(value).strip().lower() for value in attr_modes if str(value).strip()}
    if mode in normalized_attr_modes:
        return True
    if frame.empty:
        # Header-only output is valid no-data, but without persisted mode
        # provenance it must not let a difference request suppress a reduced
        # request (or vice versa).  A matching task-ledger row still adopts it.
        return False
    mode_col = _first_present(frame, ("atlas_image_type", "atlas_image_mode"))
    if mode_col is None:
        # The removed legacy fetcher omitted ``use_reduced`` and therefore
        # produced the ATLAS server's default difference-image photometry.
        return len(requested_modes) == 1 and mode == "difference"
    values = frame[mode_col].astype(str).str.strip().str.lower()
    return bool(values.eq(mode).any())


def _submit_batch(
    client: Any,
    batch: list[dict[str, object]],
    *,
    batch_comment: str,
    headers: dict[str, str],
    deadline: float | None,
    sleep_func: Callable[[float], None],
) -> list[dict[str, object]] | None:
    if not batch:
        return []
    mode = str(batch[0]["image_type"])
    if any(str(spec["image_type"]) != mode for spec in batch):
        raise ValueError("ATLAS batch contains mixed image types")
    data: dict[str, object] = {
        "radeclist": "\n".join(
            f"{float(spec['ra']):.10f},{float(spec['dec']):.10f}" for spec in batch
        ),
        "mjd_min": float(batch[0]["mjd_min"]),
        "use_reduced": bool(batch[0]["use_reduced"]),
        "send_email": False,
        "comment": batch_comment,
    }
    if not _is_missing(batch[0].get("mjd_max")):
        data["mjd_max"] = float(batch[0]["mjd_max"])

    while True:
        try:
            response = client.post(
                f"{ATLAS_API_BASE}/queue/",
                headers=headers,
                data=data,
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise RuntimeError(f"ATLAS batch submission failed: {_short_error(exc)}") from exc
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code == 201:
            try:
                payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"ATLAS submission returned invalid JSON: {_short_error(exc)}") from exc
            items = payload if isinstance(payload, list) else [payload]
            if not all(isinstance(item, dict) for item in items):
                raise RuntimeError("ATLAS submission response was not a task object list")
            return list(items)
        if status_code == 429:
            wait_seconds = _retry_wait_seconds(response)
            if deadline is not None and time.monotonic() + wait_seconds > deadline:
                return None
            sleep_func(wait_seconds)
            continue
        raise RuntimeError(_http_error("submit ATLAS batch", response))


def _assign_response_items(
    batch: list[dict[str, object]],
    items: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if len(items) != len(batch):
        raise RuntimeError(
            f"ATLAS returned {len(items)} task objects for a {len(batch)}-coordinate batch"
        )
    assignments: dict[str, dict[str, object]] = {}
    unused = list(range(len(items)))
    for position, spec in enumerate(batch):
        chosen = None
        if position in unused and _submission_item_matches_spec(items[position], spec):
            chosen = position
        else:
            for candidate in unused:
                if _submission_item_matches_spec(items[candidate], spec):
                    chosen = candidate
                    break
        if chosen is None:
            raise RuntimeError(
                f"ATLAS response could not be matched to coordinate "
                f"{float(spec['ra']):.8f},{float(spec['dec']):.8f}"
            )
        item = items[chosen]
        if not str(item.get("url") or "").strip():
            raise RuntimeError("ATLAS submission task object is missing its URL")
        unused.remove(chosen)
        assignments[str(spec["request_key"])] = item
    return assignments


def _submission_item_matches_spec(remote: dict[str, object], spec: dict[str, object]) -> bool:
    """Match a create response, tolerating a URL-only test/proxy response.

    The official serializer returns RA/Dec and request options.  If an
    intermediary returns only the task URL, DRF list-serializer ordering is
    still the authoritative mapping for that response.
    """
    if _finite_float(remote.get("ra")) is None and _finite_float(remote.get("dec")) is None:
        return True
    return _remote_matches_spec(remote, spec)


def _list_remote_tasks(client: Any, *, headers: dict[str, str]) -> list[dict[str, object]]:
    url: str | None = f"{ATLAS_API_BASE}/queue/?format=json&pagesize=500"
    tasks: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    while url:
        if url in seen_urls:
            raise RuntimeError("ATLAS queue pagination returned a loop")
        seen_urls.add(url)
        try:
            response = client.get(url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reconcile ATLAS submitting tasks against the remote queue: {_short_error(exc)}"
            ) from exc
        if int(getattr(response, "status_code", 0) or 0) != 200:
            raise RuntimeError(_http_error("list ATLAS queue", response))
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"ATLAS queue returned invalid JSON: {_short_error(exc)}") from exc
        if isinstance(payload, list):
            page = payload
            url = None
        elif isinstance(payload, dict):
            page = payload.get("results", [])
            next_url = payload.get("next")
            url = str(next_url) if next_url else None
        else:
            raise RuntimeError("ATLAS queue response was not a JSON object or list")
        tasks.extend(item for item in page if isinstance(item, dict))
    return tasks


def _reconcile_submitting_rows(
    ledger: pd.DataFrame,
    ambiguous: pd.DataFrame,
    remote_tasks: list[dict[str, object]],
) -> tuple[pd.DataFrame, int, int]:
    reconciled = 0
    deferred = 0
    unused = set(range(len(remote_tasks)))
    now = time.time()
    for _, row in ambiguous.iterrows():
        spec = row.to_dict()
        comment = str(row.get("batch_comment") or "")
        match_index = None
        for index in sorted(unused):
            remote = remote_tasks[index]
            if str(remote.get("comment") or "") != comment:
                continue
            if _remote_matches_spec(remote, spec):
                match_index = index
                break
        key = str(row["request_key"])
        if match_index is not None:
            remote = remote_tasks[match_index]
            unused.remove(match_index)
            ledger = _update_task(
                ledger,
                key,
                status=_remote_task_state(remote),
                task_id=_remote_task_id(remote),
                task_url=_absolute_task_url(remote.get("url")),
                result_url=str(remote.get("result_url") or ""),
                started_at=str(remote.get("starttimestamp") or ""),
                finished_at=str(remote.get("finishtimestamp") or ""),
                queue_position=remote.get("queuepos", np.nan),
                error_message=str(remote.get("error_msg") or ""),
                updated_unix=now,
            )
            reconciled += 1
            continue

        submitted = _finite_float(row.get("submitted_unix"))
        if submitted is not None and now - submitted < _SUBMISSION_AMBIGUITY_GRACE_SECONDS:
            deferred += 1
            continue
        ledger = _update_task(
            ledger,
            key,
            status="planned",
            batch_comment="",
            submitted_unix=np.nan,
            updated_unix=now,
            error_message="",
        )
    return ledger, reconciled, deferred


def _remote_matches_spec(remote: dict[str, object], spec: dict[str, object]) -> bool:
    remote_ra = _finite_float(remote.get("ra"))
    remote_dec = _finite_float(remote.get("dec"))
    if remote_ra is None or remote_dec is None:
        return False
    if abs(remote_ra - float(spec["ra"])) > 1e-7 or abs(remote_dec - float(spec["dec"])) > 1e-7:
        return False
    if "use_reduced" in remote and _coerce_bool(remote.get("use_reduced")) != bool(spec["use_reduced"]):
        return False
    remote_min = _finite_float(remote.get("mjd_min"))
    if remote_min is not None and abs(remote_min - float(spec["mjd_min"])) > 1e-5:
        return False
    expected_max = _finite_float(spec.get("mjd_max"))
    remote_max = _finite_float(remote.get("mjd_max"))
    if expected_max is None:
        if remote_max is not None:
            return False
    elif remote_max is None or abs(remote_max - expected_max) > 1e-5:
        return False
    return True


def _remote_task_state(task: dict[str, object]) -> str:
    if task.get("finishtimestamp"):
        return "error" if str(task.get("error_msg") or "").strip() else "finished"
    if task.get("starttimestamp"):
        return "running"
    return "queued"


def _remote_task_id(task: dict[str, object]) -> object:
    value = task.get("id")
    if not _is_missing(value):
        return value
    url = str(task.get("url") or "").rstrip("/")
    return url.rsplit("/", 1)[-1] if url else ""


def _absolute_task_url(value: object) -> str:
    text = str(value or "").strip()
    return urljoin(f"{ATLAS_API_BASE}/", text) if text else ""


def _finish_result_download(
    ledger: pd.DataFrame,
    request_key: str,
    *,
    client: Any,
    headers: dict[str, str],
    lc_dir: Path,
    results_root: Path,
    checkpoint_path: Path,
) -> pd.DataFrame:
    task_rows = ledger.loc[ledger["request_key"].astype(str).eq(request_key)]
    if task_rows.empty:
        raise KeyError(request_key)
    task = task_rows.iloc[-1]
    result_url = str(task.get("result_url") or "").strip()
    if not result_url:
        return _update_task(
            ledger,
            request_key,
            status="error",
            updated_unix=time.time(),
            error_message="ATLAS finished task has no result URL",
        )
    try:
        response = client.get(result_url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
    except Exception as exc:
        ledger = _update_task(
            ledger,
            request_key,
            status="download_retry",
            updated_unix=time.time(),
            error_message=f"result download failed: {_short_error(exc)}",
        )
        _write_task_ledger(ledger, checkpoint_path)
        return ledger

    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code in {401, 403}:
        raise RuntimeError(f"ATLAS result download authorization failed with HTTP {status_code}")
    if status_code == 429 or status_code >= 500 or status_code == 0:
        ledger = _update_task(
            ledger,
            request_key,
            status="download_retry",
            updated_unix=time.time(),
            error_message=f"transient result download HTTP {status_code}",
        )
        _write_task_ledger(ledger, checkpoint_path)
        return ledger
    if status_code == 404:
        # Static result links are explicitly short-lived.  The durable task
        # URL may return a refreshed result URL, so go back to task polling
        # instead of creating a new task or declaring a permanent failure.
        ledger = _update_task(
            ledger,
            request_key,
            status="queued",
            result_url="",
            updated_unix=time.time(),
            error_message="ATLAS result URL expired; repolling the saved task URL",
        )
        _write_task_ledger(ledger, checkpoint_path)
        return ledger
    if status_code != 200:
        ledger = _update_task(
            ledger,
            request_key,
            status="error",
            updated_unix=time.time(),
            error_message=_http_error("download ATLAS result", response),
        )
        _write_task_ledger(ledger, checkpoint_path)
        return ledger

    try:
        phot = parse_atlas_result(str(getattr(response, "text", "")))
    except Exception as exc:
        ledger = _update_task(
            ledger,
            request_key,
            status="error",
            updated_unix=time.time(),
            error_message=f"invalid ATLAS result table: {_short_error(exc)}",
        )
        _write_task_ledger(ledger, checkpoint_path)
        return ledger

    mode = str(task["image_type"])
    candidate_id = str(task["candidate_id"])
    phot = _add_canonical_columns(
        phot,
        image_type=mode,
        candidate_id=candidate_id,
        request_key=request_key,
        task_id=task.get("task_id"),
        task_url=task.get("task_url"),
        result_url=result_url,
    )
    path = _write_candidate_mode(
        lc_dir,
        candidate_id,
        phot,
        image_type=mode,
    )
    _ensure_manifest(results_root, path, candidate_id)
    now = time.time()
    ledger = _update_task(
        ledger,
        request_key,
        status="no_data" if phot.empty else "downloaded",
        downloaded_unix=now,
        updated_unix=now,
        error_message="",
        output_path=str(path),
    )
    _write_task_ledger(ledger, checkpoint_path)
    return ledger


def _add_canonical_columns(
    phot: pd.DataFrame,
    *,
    image_type: str,
    candidate_id: str,
    request_key: str,
    task_id: object,
    task_url: object,
    result_url: object,
) -> pd.DataFrame:
    out = phot.copy()
    aliases = {
        "mjd": ("MJD", "mjd"),
        "mag": ("m", "mag"),
        "mag_err": ("dm", "mag_err"),
        "filter": ("F", "filter"),
    }
    for canonical, names in aliases.items():
        if canonical in out.columns:
            continue
        source = _first_present(out, names)
        if source is not None:
            out[canonical] = out[source]
    out["atlas_image_type"] = str(image_type)
    out["atlas_use_reduced"] = bool(image_type == "reduced")
    out["candidate_id"] = str(candidate_id)
    out["request_key"] = str(request_key)
    out["task_id"] = "" if _is_missing(task_id) else str(task_id)
    out["task_url"] = "" if _is_missing(task_url) else str(task_url)
    out["result_url"] = "" if _is_missing(result_url) else str(result_url)
    out["atlas_request_key"] = str(request_key)
    out["atlas_task_id"] = "" if _is_missing(task_id) else str(task_id)
    out["atlas_task_url"] = "" if _is_missing(task_url) else str(task_url)
    out["atlas_result_url"] = "" if _is_missing(result_url) else str(result_url)
    # pandas persists JSON-compatible DataFrame attrs in parquet metadata.
    # Unlike scalar columns, this also preserves provenance for zero-row
    # header-only results.
    out.attrs.update(
        {
            "atlas_image_type": str(image_type),
            "atlas_image_types": [str(image_type)],
            "atlas_candidate_id": str(candidate_id),
            "atlas_request_key": str(request_key),
            "atlas_task_url": "" if _is_missing(task_url) else str(task_url),
            "atlas_result_url": "" if _is_missing(result_url) else str(result_url),
        }
    )
    return out


def _write_candidate_mode(
    lc_dir: Path,
    candidate_id: str,
    phot: pd.DataFrame,
    *,
    image_type: str,
) -> Path:
    path = _candidate_lc_path(lc_dir, candidate_id)
    existing = _read_parquet_if_valid(path, allow_empty=True)
    if existing is None:
        combined = phot.copy()
        existing_modes: set[str] = set()
    else:
        existing_modes = _frame_provenance_modes(existing)
        mode_col = _first_present(existing, ("atlas_image_type", "atlas_image_mode"))
        if mode_col is None:
            # Unknown legacy provenance cannot safely be combined with a new
            # explicit mode, so a successful explicit fetch replaces it.
            keep = existing.iloc[0:0].copy()
        else:
            old_modes = existing[mode_col].astype(str).str.strip().str.lower()
            keep = existing.loc[~old_modes.eq(image_type)].copy()
        combined = pd.concat([keep, phot], ignore_index=True, sort=False)
    combined.attrs.update(phot.attrs)
    combined.attrs["atlas_image_type"] = str(image_type)
    combined.attrs["atlas_image_types"] = sorted(existing_modes | {str(image_type)})
    _atomic_write_parquet(combined, path)
    return path


def _frame_provenance_modes(frame: pd.DataFrame) -> set[str]:
    values = frame.attrs.get("atlas_image_types")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        single = frame.attrs.get("atlas_image_type")
        values = [single] if isinstance(single, str) else []
    modes = {str(value).strip().lower() for value in values if str(value).strip()}
    mode_col = _first_present(frame, ("atlas_image_type", "atlas_image_mode"))
    if mode_col is not None and not frame.empty:
        modes.update(frame[mode_col].astype(str).str.strip().str.lower().dropna().tolist())
    return modes


def _candidate_lc_path(lc_dir: Path, candidate_id: str) -> Path:
    return lc_dir / f"atlas_lc_{candidate_id}.parquet"


def _read_parquet_if_valid(path: Path, *, allow_empty: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read cached ATLAS parquet {path}: {_short_error(exc)}") from exc
    return frame if allow_empty or not frame.empty else None


def _ensure_manifest(results_root: Path, path: Path, candidate_id: str) -> None:
    ok = upsert_external_lc_manifest_entry(
        results_root,
        candidate_id=candidate_id,
        source="atlas",
        file_prefix="atlas",
        path=path,
    )
    if not ok:
        raise RuntimeError(f"Could not update external-LC manifest for {path}")


def _resolve_results_root(lc_dir: Path, results_root: Path | str | None) -> Path:
    if results_root is not None:
        return Path(results_root).expanduser()
    return lc_dir.parent if lc_dir.name == "external_lcs" else lc_dir


def _apply_terminal_summaries(
    out: pd.DataFrame,
    ledger: pd.DataFrame,
    specs: list[dict[str, object]],
    modes: tuple[str, ...],
    lc_dir: Path,
) -> pd.DataFrame:
    specs_by_candidate: dict[str, list[dict[str, object]]] = {}
    for spec in specs:
        specs_by_candidate.setdefault(str(spec["candidate_id"]), []).append(spec)

    column_positions = {column: out.columns.get_loc(column) for column in ATLAS_SUMMARY_COLUMNS}
    for row_position in range(len(out)):
        candidate_id = str(out.iloc[row_position]["candidate_id"])
        candidate_specs = specs_by_candidate[candidate_id]
        task_rows = []
        for spec in candidate_specs:
            rows = ledger.loc[ledger["request_key"].astype(str).eq(str(spec["request_key"]))]
            if rows.empty:
                task_rows = []
                break
            task_rows.append(rows.iloc[-1])
        if len(task_rows) != len(modes):
            continue
        statuses = {str(row["status"]) for row in task_rows}
        if statuses & _ACTIVE_TASK_STATUSES or "error" in statuses:
            continue
        if statuses == {"no_data"}:
            summary = summarize_atlas_lc(pd.DataFrame())
        else:
            path = _candidate_lc_path(lc_dir, candidate_id)
            cached = _read_parquet_if_valid(path, allow_empty=True)
            if cached is None:
                continue
            summary = summarize_atlas_lc(cached)
        for column in ATLAS_SUMMARY_COLUMNS:
            out.iat[row_position, column_positions[column]] = summary[column]
    return out


def _count_active(ledger: pd.DataFrame, request_keys: set[str]) -> int:
    rows = _ledger_for_keys(ledger, request_keys)
    return int(rows["status"].astype(str).isin(_ACTIVE_TASK_STATUSES).sum())


def _retry_wait_seconds(response: Any) -> float:
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    if retry_after is not None:
        try:
            value = float(retry_after)
            if math.isfinite(value) and value >= 0:
                return value
        except (TypeError, ValueError):
            pass
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or "")
    except Exception:
        detail = str(getattr(response, "text", "") or "")
    seconds = re.search(r"available in\s+(\d+)\s+seconds?", detail, flags=re.IGNORECASE)
    if seconds:
        return float(seconds.group(1))
    minutes = re.search(r"available in\s+(\d+)\s+minutes?", detail, flags=re.IGNORECASE)
    if minutes:
        return float(minutes.group(1)) * 60.0
    return 10.0


def _http_error(action: str, response: Any) -> str:
    status_code = int(getattr(response, "status_code", 0) or 0)
    detail = ""
    try:
        payload = response.json()
        detail = json.dumps(payload, sort_keys=True)
    except Exception:
        detail = str(getattr(response, "text", "") or "")
    detail = " ".join(detail.split())[:500]
    return f"Failed to {action}: HTTP {status_code}{': ' + detail if detail else ''}"


def _first_present(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_missing(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "t"}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null", "<na>"}
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _finite_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _short_error(exc: Exception, max_len: int = 240) -> str:
    text = str(exc).splitlines()[0].strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(message)
