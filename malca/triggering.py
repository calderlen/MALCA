from __future__ import annotations

from typing import Literal

import numpy as np


TriggerMode = Literal["logbf", "posterior_prob"]


def posterior_probability_threshold(significance_threshold: float) -> float:
    """Normalize posterior thresholds from [0, 1] or percent space."""
    thr = float(significance_threshold)
    return thr / 100.0 if thr > 1.0 else thr


def resolve_trigger_indices(
    *,
    trigger_mode: str,
    log_bf_local: np.ndarray | None,
    event_probability: np.ndarray | None,
    logbf_threshold: float,
    significance_threshold: float,
) -> dict[str, object]:
    """Resolve per-point trigger vector and passing indices for one branch."""
    mode = str(trigger_mode).strip().lower()

    if mode == "logbf":
        point_significance = np.asarray(log_bf_local if log_bf_local is not None else [], float)
        threshold = float(logbf_threshold)
    elif mode == "posterior_prob":
        if event_probability is None:
            raise RuntimeError("trigger_mode='posterior_prob' requires event probabilities")
        point_significance = np.asarray(event_probability, float)
        threshold = posterior_probability_threshold(significance_threshold)
    else:
        raise ValueError("trigger_mode must be 'logbf' or 'posterior_prob'")

    event_indices = np.nonzero(np.isfinite(point_significance) & (point_significance >= threshold))[0].astype(int)
    trigger_max = float(np.nanmax(point_significance)) if point_significance.size and np.isfinite(point_significance).any() else np.nan

    return {
        "trigger_mode": mode,
        "point_significance": point_significance,
        "event_indices": event_indices,
        "trigger_threshold": float(threshold),
        "trigger_max": trigger_max,
    }


def normalize_trigger_block(block: dict, *, kind: Literal["dip", "jump"], default_trigger_mode: str) -> dict:
    """Backfill trigger-related summary fields for a result block."""
    out = dict(block or {})
    idx = np.asarray(out.get("event_indices", np.array([], dtype=int)), dtype=int)
    out["event_indices"] = idx

    max_log_bf_local = out.get("max_log_bf_local", np.nan)
    if not np.isfinite(max_log_bf_local):
        log_bf_local = out.get("log_bf_local", None)
        if log_bf_local is None:
            out["max_log_bf_local"] = np.nan
        else:
            lb = np.asarray(log_bf_local, float)
            out["max_log_bf_local"] = float(np.nanmax(lb)) if lb.size and np.isfinite(lb).any() else np.nan

    max_event_prob = out.get("max_event_prob", np.nan)
    if not np.isfinite(max_event_prob):
        event_prob = out.get("event_probability", None)
        if event_prob is None:
            out["max_event_prob"] = np.nan
        else:
            ep = np.asarray(event_prob, float)
            out["max_event_prob"] = float(np.nanmax(ep)) if ep.size and np.isfinite(ep).any() else np.nan

    out.setdefault("trigger_mode", str(default_trigger_mode))

    if kind == "dip":
        out["n_dips"] = int(len(idx))
    else:
        out["n_jumps"] = int(len(idx))
    return out
