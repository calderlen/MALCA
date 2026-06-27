"""SED spectral-index features for YSO diagnostics."""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd


SED_ALPHA_COLUMNS = [
    "candidate_id",
    "sed_alpha",
    "sed_alpha_class",
    "sed_alpha_n_points",
    "sed_alpha_lambda_min_micron",
    "sed_alpha_lambda_max_micron",
    "sed_alpha_bands_json",
    "sed_alpha_status",
]

SED_ALPHA_LAMBDA_MIN_MICRON = 2.0
SED_ALPHA_LAMBDA_MAX_MICRON = 24.0
SED_ALPHA_BLUE_ANCHOR_MAX_MICRON = 3.0
SED_ALPHA_RED_ANCHOR_MIN_MICRON = 10.0
SED_ALPHA_MIN_POINTS = 3


def _safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        if value is None or pd.isna(value):
            return False
    except Exception:
        if value is None:
            return False
    return bool(value)


def _candidate_id_for_row(row: pd.Series, index: object) -> str:
    for col in ("candidate_id", "asas_sn_id", "gaia_id", "source_id"):
        if col not in row:
            continue
        value = row.get(col)
        try:
            if value is None or pd.isna(value):
                continue
        except Exception:
            if value is None:
                continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            return text
    return str(index)


def _candidate_lookup(candidates: pd.DataFrame | Iterable[dict] | None) -> dict[str, dict]:
    if candidates is None:
        return {}
    frame = pd.DataFrame(candidates)
    if frame.empty:
        return {}
    lookup: dict[str, dict] = {}
    for index, row in frame.iterrows():
        lookup[_candidate_id_for_row(row, index)] = row.to_dict()
    return lookup


def _empty_alpha_row(candidate_id: str, status: str) -> dict:
    return {
        "candidate_id": str(candidate_id),
        "sed_alpha": np.nan,
        "sed_alpha_class": "unknown",
        "sed_alpha_n_points": 0,
        "sed_alpha_lambda_min_micron": np.nan,
        "sed_alpha_lambda_max_micron": np.nan,
        "sed_alpha_bands_json": "[]",
        "sed_alpha_status": status,
    }


def classify_sed_alpha(alpha: float | None) -> str:
    """Return the standard YSO class implied by an infrared SED slope."""
    value = _safe_float(alpha)
    if value is None:
        return "unknown"
    if value >= 0.3:
        return "Class I"
    if value >= -0.3:
        return "Flat"
    if value >= -1.6:
        return "Class II"
    return "Class III/photosphere"


def _prepared_alpha_points(
    sed_rows: pd.DataFrame,
    candidate: dict | pd.Series | None,
) -> pd.DataFrame:
    if sed_rows is None or sed_rows.empty:
        return pd.DataFrame()

    frame = pd.DataFrame(sed_rows).copy()
    for col in ("lambda_eff_angstrom", "flux_lambda", "lambda_l_lambda"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "lambda_eff_angstrom" not in frame.columns:
        return pd.DataFrame()

    frame["lambda_micron"] = frame["lambda_eff_angstrom"] * 1.0e-4
    good = (
        np.isfinite(frame["lambda_micron"])
        & (frame["lambda_micron"] >= SED_ALPHA_LAMBDA_MIN_MICRON)
        & (frame["lambda_micron"] <= SED_ALPHA_LAMBDA_MAX_MICRON)
    )
    if "is_upper_limit" in frame.columns:
        good &= ~frame["is_upper_limit"].map(_to_bool)
    if "is_synthetic" in frame.columns:
        good &= ~frame["is_synthetic"].map(_to_bool)
    frame = frame.loc[good].copy()
    if frame.empty:
        return frame

    lum = (
        pd.to_numeric(frame["lambda_l_lambda"], errors="coerce")
        if "lambda_l_lambda" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    flux = (
        pd.to_numeric(frame["flux_lambda"], errors="coerce")
        if "flux_lambda" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    lam_ang = pd.to_numeric(frame["lambda_eff_angstrom"], errors="coerce")

    if np.isfinite(lum).all() and (lum > 0).all():
        y = lum.to_numpy(dtype=float)
    else:
        y = (lam_ang * flux).to_numpy(dtype=float)

    av = _safe_float(candidate.get("A_v_3d")) if candidate is not None else None
    if av is not None and av > 0 and "av_coeff" in frame.columns:
        coeff = pd.to_numeric(frame["av_coeff"], errors="coerce").to_numpy(dtype=float)
        flags = frame.get("quality_flags")
        if flags is None:
            already_corrected = np.zeros(len(frame), dtype=bool)
        else:
            already_corrected = flags.fillna("").astype(str).str.contains(
                "ism_corrected", case=False, regex=False
            ).to_numpy()
        can_correct = np.isfinite(coeff) & ~already_corrected
        y[can_correct] = y[can_correct] * (10.0 ** (0.4 * av * coeff[can_correct]))

    frame["alpha_y"] = y
    frame = frame[np.isfinite(frame["alpha_y"]) & (frame["alpha_y"] > 0)].copy()
    sort_cols = [col for col in ("lambda_micron", "source", "band") if col in frame.columns]
    return frame.sort_values(sort_cols, na_position="last") if sort_cols else frame


def fit_sed_alpha_for_candidate(
    candidate_id: str,
    sed_rows: pd.DataFrame,
    *,
    candidate: dict | pd.Series | None = None,
) -> dict:
    """Fit one candidate's 2-24 micron spectral index from normalized SED rows."""
    cid = str(candidate_id)
    points = _prepared_alpha_points(sed_rows, candidate)
    if points.empty:
        return _empty_alpha_row(cid, "no_valid_points")
    if len(points) < SED_ALPHA_MIN_POINTS:
        row = _empty_alpha_row(cid, "insufficient_valid_points")
        row["sed_alpha_n_points"] = int(len(points))
        return row
    if not (points["lambda_micron"] <= SED_ALPHA_BLUE_ANCHOR_MAX_MICRON).any():
        row = _empty_alpha_row(cid, "missing_blue_anchor")
        row["sed_alpha_n_points"] = int(len(points))
        return row
    if not (points["lambda_micron"] >= SED_ALPHA_RED_ANCHOR_MIN_MICRON).any():
        row = _empty_alpha_row(cid, "missing_red_anchor")
        row["sed_alpha_n_points"] = int(len(points))
        return row

    x = np.log10(points["lambda_micron"].to_numpy(dtype=float))
    y = np.log10(points["alpha_y"].to_numpy(dtype=float))
    if np.unique(x).size < 2:
        row = _empty_alpha_row(cid, "degenerate_wavelengths")
        row["sed_alpha_n_points"] = int(len(points))
        return row

    slope, _intercept = np.polyfit(x, y, deg=1)
    alpha = float(slope)
    bands = [
        {
            "source": str(row.get("source") or ""),
            "band": str(row.get("band") or ""),
            "lambda_eff_angstrom": _safe_float(row.get("lambda_eff_angstrom")),
        }
        for _, row in points.iterrows()
    ]
    return {
        "candidate_id": cid,
        "sed_alpha": alpha,
        "sed_alpha_class": classify_sed_alpha(alpha),
        "sed_alpha_n_points": int(len(points)),
        "sed_alpha_lambda_min_micron": float(points["lambda_micron"].min()),
        "sed_alpha_lambda_max_micron": float(points["lambda_micron"].max()),
        "sed_alpha_bands_json": json.dumps(bands, separators=(",", ":")),
        "sed_alpha_status": "ok",
    }


def compute_sed_alpha_features(
    candidates: pd.DataFrame | Iterable[dict] | None,
    sed_rows: pd.DataFrame | Iterable[dict],
) -> pd.DataFrame:
    """Compute SED-alpha rows for every candidate represented in *candidates*."""
    rows = pd.DataFrame(sed_rows)
    lookup = _candidate_lookup(candidates)
    if rows.empty:
        candidate_ids = sorted(lookup)
        return pd.DataFrame(
            [_empty_alpha_row(cid, "no_sed_rows") for cid in candidate_ids],
            columns=SED_ALPHA_COLUMNS,
        )
    if "candidate_id" not in rows.columns:
        return pd.DataFrame(columns=SED_ALPHA_COLUMNS)

    row_ids = {str(cid) for cid in rows["candidate_id"].dropna().astype(str).unique()}
    candidate_ids = sorted(set(lookup) | row_ids)
    out = []
    for cid in candidate_ids:
        candidate_rows = rows[rows["candidate_id"].astype(str) == cid]
        if candidate_rows.empty:
            out.append(_empty_alpha_row(cid, "no_sed_rows"))
            continue
        out.append(fit_sed_alpha_for_candidate(cid, candidate_rows, candidate=lookup.get(cid, {})))
    return pd.DataFrame(out, columns=SED_ALPHA_COLUMNS)


def _sqlite_value(value: object) -> object:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def upsert_sed_alpha_results(conn: sqlite3.Connection, alpha_rows: pd.DataFrame) -> int:
    """Merge SED-alpha feature rows into review candidate columns and payloads."""
    if alpha_rows is None or alpha_rows.empty:
        return 0
    from malca.review.store import replace_candidate_payload_fields

    frame = alpha_rows.copy()
    for col in SED_ALPHA_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    count = 0
    for _, row in frame[SED_ALPHA_COLUMNS].iterrows():
        cid = str(row.get("candidate_id") or "").strip()
        if not cid:
            continue
        updates = {col: _sqlite_value(row[col]) for col in SED_ALPHA_COLUMNS if col != "candidate_id"}
        if replace_candidate_payload_fields(conn, cid, updates, commit=False):
            count += 1
    conn.commit()
    return count
