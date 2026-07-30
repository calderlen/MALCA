"""Stable MALCA adapter for the GitHub BANYAN Sigma implementation."""

from __future__ import annotations

import importlib.metadata
import json
import math
from typing import Callable

import numpy as np
import pandas as pd


BANYAN_ADAPTER_VERSION = "1"
BANYAN_OUTPUT_COLUMNS = (
    "banyan_field_prob",
    "banyan_ya_prob",
    "banyan_best_assoc",
    "banyan_best_assoc_prob",
    "banyan_probabilities_json",
    "banyan_input_mode",
    "banyan_status",
    "banyan_error",
    "banyan_version",
    "banyan_adapter_version",
    "banyan_updated_at",
)


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _package_version() -> str:
    try:
        return importlib.metadata.version("banyan-sigma")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _column(frame: pd.DataFrame, *tokens: str) -> pd.Series | None:
    wanted = tuple(token.upper() for token in tokens)
    for column in frame.columns:
        if isinstance(column, tuple):
            parts = tuple(str(part).upper() for part in column)
        else:
            parts = (str(column).upper(),)
        if all(token in parts for token in wanted):
            return frame[column]
    return None


def _association_probability_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    probabilities: dict[str, pd.Series] = {}
    for column in frame.columns:
        if not isinstance(column, tuple) or len(column) < 2:
            continue
        first, second = str(column[0]).upper(), str(column[1]).upper()
        if first == "ALL" and "FIELD" not in second and second not in {"GLOBAL", "METRICS"}:
            probabilities[second] = pd.to_numeric(frame[column], errors="coerce")
    return probabilities


def _parse_membership_output(result: pd.DataFrame, *, association_threshold: float) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise RuntimeError("BANYAN Sigma returned no rows")

    association_columns = _association_probability_columns(result)
    field_columns = []
    for column in result.columns:
        parts = tuple(str(part).upper() for part in column) if isinstance(column, tuple) else (str(column).upper(),)
        if "ALL" in parts and any("FIELD" in part for part in parts):
            field_columns.append(pd.to_numeric(result[column], errors="coerce"))
    field_prob = (
        pd.concat(field_columns, axis=1).sum(axis=1, min_count=1)
        if field_columns
        else pd.Series(np.nan, index=result.index, dtype=float)
    )

    ya_prob_raw = _column(result, "GLOBAL", "YA_PROB")
    if ya_prob_raw is None:
        ya_prob_raw = _column(result, "YA_PROB")
    ya_prob = (
        pd.to_numeric(ya_prob_raw, errors="coerce")
        if ya_prob_raw is not None
        else 1.0 - field_prob
    )
    best_ya_raw = _column(result, "GLOBAL", "BEST_YA")
    if best_ya_raw is None:
        best_ya_raw = _column(result, "BEST_YA")
    best_ya = (
        best_ya_raw.fillna("").astype(str)
        if best_ya_raw is not None
        else pd.Series("", index=result.index, dtype=object)
    )

    parsed_rows: list[dict[str, object]] = []
    for index in result.index:
        association = str(best_ya.loc[index]).strip().upper()
        assoc_prob = math.nan
        if association and association != "FIELD" and association in association_columns:
            value = association_columns[association].loc[index]
            assoc_prob = float(value) if pd.notna(value) and math.isfinite(float(value)) else math.nan
        named = association if math.isfinite(assoc_prob) and assoc_prob > association_threshold else ""
        probability_map = {
            name: float(values.loc[index])
            for name, values in association_columns.items()
            if pd.notna(values.loc[index]) and math.isfinite(float(values.loc[index]))
        }
        parsed_rows.append(
            {
                "banyan_field_prob": float(field_prob.loc[index]),
                "banyan_ya_prob": float(ya_prob.loc[index]),
                "banyan_best_assoc": named,
                "banyan_best_assoc_prob": assoc_prob,
                "banyan_probabilities_json": json.dumps(
                    probability_map, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(parsed_rows, index=result.index)


def _membership_callable() -> Callable[..., pd.DataFrame]:
    try:
        import banyan_sigma
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        raise RuntimeError(f"Could not import banyan_sigma: {exc}") from exc
    func = getattr(banyan_sigma, "membership_probability", None)
    if not callable(func):
        raise RuntimeError("banyan_sigma does not expose membership_probability()")
    return func


def compute_banyan_membership(
    candidates: pd.DataFrame,
    *,
    association_threshold: float = 0.1,
    membership_func: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return *candidates* with explicit BANYAN results or ineligibility reasons."""
    out = candidates.copy()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    package_version = _package_version()
    defaults: dict[str, object] = {
        "banyan_field_prob": np.nan,
        "banyan_ya_prob": np.nan,
        "banyan_best_assoc": "",
        "banyan_best_assoc_prob": np.nan,
        "banyan_probabilities_json": "{}",
        "banyan_input_mode": "",
        "banyan_status": "missing_inputs",
        "banyan_error": "",
        "banyan_version": package_version,
        "banyan_adapter_version": BANYAN_ADAPTER_VERSION,
        "banyan_updated_at": now,
    }
    for column, value in defaults.items():
        out[column] = value
    if out.empty:
        return out

    ra = _finite_series(out, "ra")
    dec = _finite_series(out, "dec")
    pmra = _finite_series(out, "pmra")
    pmdec = _finite_series(out, "pmdec")
    epmra = _finite_series(out, "pmra_error")
    epmdec = _finite_series(out, "pmdec_error")
    plx = _finite_series(out, "parallax")
    eplx = _finite_series(out, "parallax_error")
    rv = _finite_series(out, "radial_velocity")
    erv = _finite_series(out, "radial_velocity_error")

    coordinates_ok = ra.notna() & dec.notna() & ra.between(0, 360, inclusive="left") & dec.between(-90, 90)
    motion_ok = pmra.notna() & pmdec.notna()
    motion_error_ok = epmra.notna() & epmdec.notna() & (epmra > 0) & (epmdec > 0)
    eligible = coordinates_ok & motion_ok & motion_error_ok

    out.loc[~coordinates_ok, "banyan_status"] = "missing_coordinates"
    out.loc[coordinates_ok & ~motion_ok, "banyan_status"] = "missing_proper_motion"
    out.loc[coordinates_ok & motion_ok & ~motion_error_ok, "banyan_status"] = "missing_proper_motion_error"

    parallax_ok = plx.notna() & eplx.notna() & (plx > 0) & (eplx > 0)
    rv_ok = rv.notna() & erv.notna() & (erv > 0)
    out.loc[eligible, "banyan_input_mode"] = "pm"
    out.loc[eligible & parallax_ok, "banyan_input_mode"] = "pm+plx"
    out.loc[eligible & rv_ok, "banyan_input_mode"] = "pm+rv"
    out.loc[eligible & parallax_ok & rv_ok, "banyan_input_mode"] = "pm+plx+rv"
    out.loc[eligible, "banyan_status"] = "pending"

    if not eligible.any():
        return out
    try:
        func = membership_func or _membership_callable()
    except Exception as exc:
        out.loc[eligible, "banyan_status"] = "package_api_mismatch"
        out.loc[eligible, "banyan_error"] = str(exc)
        return out

    for mode in ("pm", "pm+plx", "pm+rv", "pm+plx+rv"):
        indices = out.index[eligible & out["banyan_input_mode"].eq(mode)]
        if len(indices) == 0:
            continue
        kwargs: dict[str, object] = {
            "ra": ra.loc[indices].to_numpy(dtype=float),
            "dec": dec.loc[indices].to_numpy(dtype=float),
            "pmra": pmra.loc[indices].to_numpy(dtype=float),
            "pmdec": pmdec.loc[indices].to_numpy(dtype=float),
            "epmra": epmra.loc[indices].to_numpy(dtype=float),
            "epmdec": epmdec.loc[indices].to_numpy(dtype=float),
        }
        if "plx" in mode:
            kwargs.update(
                plx=plx.loc[indices].to_numpy(dtype=float),
                eplx=eplx.loc[indices].to_numpy(dtype=float),
                use_plx=True,
            )
        if "rv" in mode:
            kwargs.update(
                rv=rv.loc[indices].to_numpy(dtype=float),
                erv=erv.loc[indices].to_numpy(dtype=float),
                use_rv=True,
            )
        try:
            result = func(**kwargs)
            result = result.copy()
            result.index = indices
            parsed = _parse_membership_output(
                result, association_threshold=float(association_threshold)
            )
            for column in parsed.columns:
                out.loc[indices, column] = parsed[column]
            finite = pd.to_numeric(
                out.loc[indices, "banyan_field_prob"], errors="coerce"
            ).notna()
            good_indices = finite.index[finite]
            bad_indices = finite.index[~finite]
            out.loc[good_indices, "banyan_status"] = "ok"
            out.loc[bad_indices, "banyan_status"] = "calculation_error"
            out.loc[bad_indices, "banyan_error"] = "BANYAN returned no finite field probability"
        except Exception as exc:
            out.loc[indices, "banyan_status"] = "calculation_error"
            out.loc[indices, "banyan_error"] = str(exc)

    return out
