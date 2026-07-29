"""Dip-run epoch extraction shared by the pipeline and the backfill script.

The events pipeline (``malca.stv.events.score_lightcurve``) already computes
``run_summaries`` with per-run ``start_jd`` / ``end_jd`` timestamps, but only
aggregate summaries (``dip_max_run_duration`` etc.) are persisted per
candidate. This module adds two entry points:

* :func:`serialize_run_summaries` / :func:`parse_run_epochs_json` — turn the
  events-pipeline ``run_summaries`` list into a compact JSON string suitable
  for storing in a single DB column, and read it back.

* :func:`detect_dip_epochs_lightweight` — a fast standalone dip finder for
  cases where the events pipeline has not been re-run (backfill). It reuses
  ``build_runs`` / ``filter_runs`` from ``malca.stv.events`` so triggered-point
  clustering follows the same semantics as the pipeline. The trigger is a
  simple rolling-median residual sigma threshold on a per-camera cleaned
  light curve (no GP posterior), so it runs orders of magnitude faster than
  ``score_lightcurve`` while producing epochs the pipeline would accept.

Both entry points return a plain ``list[float]`` of dip center JDs so the
consumer (``resolve_period_consensus``) is agnostic to how the epochs were
sourced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import json

import numpy as np
import pandas as pd

from malca.config import MAD_SCALE


DIP_RUN_EPOCHS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DipRunEpoch:
    """Compact structure for a single dip run persisted to DB."""

    start_jd: float
    end_jd: float
    center_jd: float
    peak_significance: float
    n_points: int
    duration_days: float


# ---------------------------------------------------------------------------
# Persistence: (de)serialise dip_run_epochs_json
# ---------------------------------------------------------------------------

def serialize_run_summaries(
    run_summaries: Sequence[dict[str, Any]] | None,
    *,
    kept_only: bool = True,
) -> str | None:
    """Return a compact JSON blob suitable for a ``dip_run_epochs_json`` column.

    ``run_summaries`` is the list emitted by ``filter_runs`` /
    ``summarize_kept_runs`` in the events pipeline. Each entry must include
    ``start_jd`` and ``end_jd`` in JD, plus optionally ``run_max`` (peak
    significance), ``n_points``, ``duration_days``, and a ``kept`` flag.

    Returns ``None`` when ``run_summaries`` is empty or contains no kept runs
    so callers can persist ``NULL`` rather than an empty JSON literal.
    """
    if not run_summaries:
        return None
    epochs: list[dict[str, float | int]] = []
    for summary in run_summaries:
        if kept_only and not bool(summary.get("kept", True)):
            continue
        start = _finite(summary.get("start_jd"))
        end = _finite(summary.get("end_jd"))
        if start is None or end is None:
            continue
        peak = _finite(summary.get("run_max"))
        duration = _finite(summary.get("duration_days"))
        if duration is None:
            duration = max(float(end - start), 0.0)
        center = 0.5 * (start + end)
        epochs.append(
            {
                "start_jd": float(start),
                "end_jd": float(end),
                "center_jd": float(center),
                "peak_significance": float(peak) if peak is not None else float("nan"),
                "n_points": int(summary.get("n_points") or 0),
                "duration_days": float(duration),
            }
        )
    if not epochs:
        return None
    payload = {
        "schema": DIP_RUN_EPOCHS_SCHEMA_VERSION,
        "epochs": epochs,
    }
    return json.dumps(payload, separators=(",", ":"))


def parse_run_epochs_json(blob: str | bytes | None) -> list[DipRunEpoch]:
    """Inverse of :func:`serialize_run_summaries`. Never raises on bad input."""
    if blob is None:
        return []
    if isinstance(blob, bytes):
        try:
            blob = blob.decode("utf-8")
        except UnicodeDecodeError:
            return []
    if not isinstance(blob, str):
        return []
    blob = blob.strip()
    if not blob:
        return []
    try:
        payload = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(payload, list):
        raw_epochs = payload
    elif isinstance(payload, dict):
        raw_epochs = payload.get("epochs") or []
    else:
        return []
    out: list[DipRunEpoch] = []
    for entry in raw_epochs:
        if not isinstance(entry, dict):
            continue
        start = _finite(entry.get("start_jd"))
        end = _finite(entry.get("end_jd"))
        center = _finite(entry.get("center_jd"))
        if center is None and start is not None and end is not None:
            center = 0.5 * (start + end)
        if center is None:
            continue
        out.append(
            DipRunEpoch(
                start_jd=float(start) if start is not None else float("nan"),
                end_jd=float(end) if end is not None else float("nan"),
                center_jd=float(center),
                peak_significance=float(_finite(entry.get("peak_significance")) or float("nan")),
                n_points=int(entry.get("n_points") or 0),
                duration_days=float(_finite(entry.get("duration_days")) or float("nan")),
            )
        )
    return out


def dip_center_jds(epochs: Sequence[DipRunEpoch]) -> list[float]:
    """Return sorted list of dip center JDs."""
    finite = [float(e.center_jd) for e in epochs if np.isfinite(e.center_jd)]
    finite.sort()
    return finite


# ---------------------------------------------------------------------------
# Lightweight dip detection for backfill
# ---------------------------------------------------------------------------

def detect_dip_epochs_lightweight(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None = None,
    *,
    cam_vec: np.ndarray | None = None,
    n_sigma: float = 4.0,
    rolling_window_days: float = 300.0,
    min_run_points: int = 2,
    max_gap_points: int = 1,
    min_run_duration_days: float | None = None,
    min_depth_mag: float = 0.1,
) -> list[DipRunEpoch]:
    """Detect dip-run epochs using a rolling-median residual trigger.

    This is a deliberately cheap standalone alternative to
    ``score_lightcurve``. It exists so the backfill script can produce
    ``dip_run_epochs_json`` for a candidate whose events pipeline never
    persisted that field.

    The pipeline itself should call ``score_lightcurve`` and pass the resulting
    ``run_summaries`` through :func:`serialize_run_summaries` rather than use
    this heuristic. When both are available the pipeline output is preferred.

    Parameters
    ----------
    jd, mag, err:
        Cleaned light-curve arrays. ``err`` is optional; when omitted the
        trigger falls back to a MAD-based estimate.
    cam_vec:
        Optional camera-label array so ``filter_runs`` can count cameras per
        run.
    n_sigma:
        Threshold on the residual-over-sigma ratio for triggering a point.
        A dip is a **positive** residual (fainter -> larger mag).
    rolling_window_days:
        Boxcar median-window width for the baseline.
    min_run_points, max_gap_points, min_run_duration_days:
        Passed straight through to ``build_runs``/``filter_runs``.
    min_depth_mag:
        Extra safety threshold: reject runs whose peak residual is below this
        magnitude even if they clear the per-point sigma threshold.

    Returns
    -------
    list[DipRunEpoch]
    """
    # Deferred to break the circular import cycle
    # ``malca.stv.events`` -> ``malca.core.event_epochs`` -> ``malca.stv.events``.
    from malca.stv.events import build_runs, filter_runs

    jd_arr = np.asarray(jd, dtype=float)
    mag_arr = np.asarray(mag, dtype=float)
    err_arr = np.asarray(err, dtype=float) if err is not None else np.array([])

    mask = np.isfinite(jd_arr) & np.isfinite(mag_arr)
    if err_arr.size == mag_arr.size:
        mask &= np.isfinite(err_arr) & (err_arr > 0)
    if mask.sum() < max(3 * int(min_run_points), 20):
        return []

    order = np.argsort(jd_arr[mask], kind="stable")
    jd_s = jd_arr[mask][order]
    mag_s = mag_arr[mask][order]
    err_s = err_arr[mask][order] if err_arr.size == mag_arr.size else None
    cam_s = np.asarray(cam_vec)[mask][order] if cam_vec is not None else None

    baseline = _rolling_median_by_time(jd_s, mag_s, window_days=float(rolling_window_days))
    residual = mag_s - baseline

    if err_s is not None and np.any(np.isfinite(err_s) & (err_s > 0)):
        sigma_eff = np.where(np.isfinite(err_s) & (err_s > 0), err_s, np.nan)
        finite_mask = np.isfinite(sigma_eff)
        if finite_mask.any():
            fallback = float(np.nanmedian(sigma_eff[finite_mask]))
            sigma_eff = np.where(finite_mask, sigma_eff, fallback)
        else:
            sigma_eff = np.full_like(mag_s, np.nan)
    else:
        sigma_eff = np.full_like(mag_s, np.nan)

    if not np.any(np.isfinite(sigma_eff) & (sigma_eff > 0)):
        finite_resid = residual[np.isfinite(residual)]
        if finite_resid.size == 0:
            return []
        mad = MAD_SCALE * float(np.median(np.abs(finite_resid - float(np.median(finite_resid)))))
        if not np.isfinite(mad) or mad <= 0:
            return []
        sigma_eff = np.full_like(mag_s, mad)

    z = residual / sigma_eff
    trig_idx = np.where(np.isfinite(z) & (z >= float(n_sigma)) & (residual >= float(min_depth_mag)))[0]
    if trig_idx.size == 0:
        return []

    runs = build_runs(trig_idx, jd_s, max_gap_points=int(max_gap_points))
    if not runs:
        return []
    _, summaries = filter_runs(
        runs,
        jd_s,
        z,
        min_points=int(min_run_points),
        min_duration_days=min_run_duration_days,
        per_point_threshold=float(n_sigma),
        cam_vec=cam_s,
    )

    out: list[DipRunEpoch] = []
    for summary in summaries:
        if not summary.get("kept"):
            continue
        start = _finite(summary.get("start_jd"))
        end = _finite(summary.get("end_jd"))
        if start is None or end is None:
            continue
        # Peak location within the run gives a better center than the midpoint
        # of the sparse endpoints.
        i0 = int(summary["start_idx"])
        i1 = int(summary["end_idx"]) + 1
        window = residual[i0:i1]
        if window.size == 0:
            center = 0.5 * (start + end)
            peak_significance = float(summary.get("run_max") or float("nan"))
        else:
            k = int(np.argmax(window))
            center = float(jd_s[i0 + k])
            peak_significance = float(summary.get("run_max") or float("nan"))
        out.append(
            DipRunEpoch(
                start_jd=float(start),
                end_jd=float(end),
                center_jd=float(center),
                peak_significance=peak_significance,
                n_points=int(summary.get("n_points") or 0),
                duration_days=float(end - start),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _rolling_median_by_time(
    jd: np.ndarray,
    values: np.ndarray,
    *,
    window_days: float,
) -> np.ndarray:
    """Rolling median with a time-based window (centered).

    We use a plain two-pointer sweep rather than a general rolling apply since
    JD arrays here are already sorted; this keeps the routine O(n log w) and
    avoids importing scipy for such a simple task. For each point we take the
    median of all values whose JD lies within ``window_days/2`` on either
    side.
    """
    if window_days <= 0 or jd.size == 0:
        return values.astype(float).copy()
    half = float(window_days) / 2.0
    n = jd.size
    out = np.empty_like(values, dtype=float)
    lo = 0
    hi = 0
    for i in range(n):
        while lo < n and jd[lo] < jd[i] - half:
            lo += 1
        while hi < n and jd[hi] <= jd[i] + half:
            hi += 1
        window = values[lo:hi]
        if window.size == 0:
            out[i] = float(values[i])
        else:
            out[i] = float(np.median(window))
    return out


def detect_dip_epochs_via_events(
    df_lc: pd.DataFrame,
    *,
    score_kwargs: dict[str, Any] | None = None,
) -> list[DipRunEpoch]:
    """Detect dip-run epochs by calling the full events ``score_lightcurve``.

    This is the preferred path when we need epochs that match the pipeline
    exactly (same GP baseline, same trigger, same ``build_runs`` /
    ``filter_runs`` clustering). It is substantially more expensive than
    :func:`detect_dip_epochs_lightweight` and should be reserved for the
    backfill / reprocessing path rather than the per-candidate hot loop.
    """
    from malca.stv.events import score_lightcurve

    if df_lc is None or df_lc.empty:
        return []
    score_kwargs = dict(score_kwargs or {})
    try:
        results = score_lightcurve(df_lc, **score_kwargs)
    except Exception:
        return []
    dip = results.get("dip") if isinstance(results, dict) else None
    if not isinstance(dip, dict):
        return []
    summaries = dip.get("run_summaries") or []
    blob = serialize_run_summaries(summaries)
    return parse_run_epochs_json(blob)


__all__ = [
    "DIP_RUN_EPOCHS_SCHEMA_VERSION",
    "DipRunEpoch",
    "detect_dip_epochs_lightweight",
    "detect_dip_epochs_via_events",
    "dip_center_jds",
    "parse_run_epochs_json",
    "serialize_run_summaries",
]
