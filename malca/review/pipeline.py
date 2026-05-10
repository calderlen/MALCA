"""Pipeline stage detection and on-demand runner for the review widget.

Detects which analysis stages have been run (by checking for signature
columns in the candidate payload) and can run missing stages on demand.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Callable, Optional
import json
import sqlite3

import numpy as np
import pandas as pd

from malca.config import (
    TRIGGER_MODE, LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP,
    SIGNIFICANCE_THRESHOLD, P_POINTS, MAG_POINTS,
    RUN_MIN_POINTS, RUN_MAX_GAP_POINTS, BASELINE_FUNC,
)
from malca.review.stats_merge import merge_stats_summary_into_payload as _merge_stats_summary_into_payload
from malca.review.store import _CANDIDATE_COLUMNS, _as_bool, _to_float


P_MIN_DIP = None
P_MAX_DIP = None
P_MIN_JUMP = None
P_MAX_JUMP = None
MAX_GAP_POINTS = RUN_MAX_GAP_POINTS
RUN_MAX_GAP_DAYS = None
RUN_MIN_DURATION_DAYS = 0.0
BASELINE_TAG = BASELINE_FUNC


class _ProgressCaptureStream(io.TextIOBase):
    """Stream that forwards stdout/stderr lines to a progress callback."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__()
        self._sink = sink
        self._buffer = ""
        self._last_line = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        for ch in str(text):
            if ch in "\r\n":
                self._emit_buffer()
            else:
                self._buffer += ch
        return len(text)

    def flush(self) -> None:
        self._emit_buffer()

    def _emit_buffer(self) -> None:
        line = self._buffer.strip()
        self._buffer = ""
        if not line:
            return
        if line == self._last_line:
            return
        self._last_line = line
        try:
            self._sink(line)
        except Exception:
            pass


def _run_with_progress_capture(func: Callable[[], object], p: Callable[[str], None] | None) -> object:
    """Run *func* while forwarding printed lines to progress callback *p*."""
    if p is None:
        return func()
    stream = _ProgressCaptureStream(p)
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = func()
    stream.flush()
    return result








# ---------------------------------------------------------------------------
# Stage signatures: columns whose presence indicates a stage has run
# ---------------------------------------------------------------------------
STAGE_SIGNATURES: dict[str, list[str]] = {
    "stats": [
        "n_points",
        "cadence_median_days",
        "baseline_mag",
        "stats_photometry_mean_mag",
    ],
    "events": [
        "dip_significant",
        "dip_best_morph",
        "jump_significant",
    ],
    "characterize": [
        "parallax",
        "tmass_j",
        "gal_l",
    ],
    "vetting": [
        "simbad_main_id",
        "gaia_var_flag",
        "vetting_likely_known",
    ],
    "external_lcs": [
        "atlas_has_phot",
        "ztf_lc_n_det",
        "gaia_epoch_lc_n_g",
        "ps1_lc_n_points",
        "crts_lc_n_points",
    ],
}


def detect_pipeline_status(payload: dict) -> dict[str, str]:
    """Determine which pipeline stages have completed for a candidate.

    Returns a dict mapping stage name → status:
      "complete"  = all signature columns present and non-null
      "partial"   = some signature columns present
      "missing"   = no signature columns present
    """
    result = {}
    for stage, sig_cols in STAGE_SIGNATURES.items():
        present = sum(
            1 for c in sig_cols
            if c in payload and payload[c] is not None
            and not (isinstance(payload[c], float) and np.isnan(payload[c]))
        )
        if present == 0:
            result[stage] = "missing"
        elif present == len(sig_cols):
            result[stage] = "complete"
        else:
            result[stage] = "partial"
    return result


def run_missing_stages(
    conn: sqlite3.Connection,
    candidate_id: str,
    progress_callback: Callable[[str], None] | None = None,
    stage_complete_callback: Callable[[str], None] | None = None,
    force_stages: list[str] | None = None,
    only_force: bool = False,
) -> list[str]:
    """Detect and run missing pipeline stages for a candidate.

    Returns a list of stage names that were executed.
    """
    # 1. Load payload from DB
    row = conn.execute(
        "SELECT payload_json, lc_path FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    conn.commit()  # Release structural read lock during heavy API wait times
    
    if row is None:
        raise ValueError(f"Candidate {candidate_id} not found in DB")

    payload = json.loads(row[0]) if row[0] else {}
    lc_path = row[1] or payload.get("lc_path")

    # 2. Detect current status
    status = detect_pipeline_status(payload)
    stages_run: list[str] = []

    def p(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"[pipeline] {msg}")

    def mark_stage_complete(stage_name: str) -> None:
        """Persist payload after each stage and notify listeners."""
        update_candidate_payload(conn, candidate_id, payload)
        if stage_complete_callback:
            stage_complete_callback(stage_name)

    # 3. Run missing stages in order
    force = set(force_stages or [])

    def should_run(stage_name: str) -> bool:
        if only_force:
            return stage_name in force
        return (status.get(stage_name) in ("missing", "partial")) or (stage_name in force)

    if should_run("stats"):
        if lc_path and Path(lc_path).exists():
            p("Computing LC stats...")
            _run_stats_stage(payload, lc_path, p)
            stages_run.append("stats")
            mark_stage_complete("stats")

    if should_run("events"):
        if lc_path and Path(lc_path).exists():
            p("Running event detection...")
            _run_events_stage(payload, lc_path, p)
            stages_run.append("events")
            mark_stage_complete("events")

    if should_run("characterize"):
        ra = payload.get("ra_deg")
        dec = payload.get("dec_deg")
        if ra is not None and dec is not None:
            p("Characterizing...")
            _run_characterize_stage(payload, p)
            stages_run.append("characterize")
            mark_stage_complete("characterize")

    if should_run("vetting"):
        ra = payload.get("ra_deg")
        dec = payload.get("dec_deg")
        if ra is not None and dec is not None:
            p("Vetting crossmatches...")
            _run_vetting_stage(payload, p)
            stages_run.append("vetting")
            mark_stage_complete("vetting")

    if should_run("external_lcs"):
        ra = payload.get("ra_deg")
        dec = payload.get("dec_deg")
        if ra is not None and dec is not None:
            p("Fetching external LCs...")
            _run_external_lcs_stage(payload, output_dir=_resolve_output_dir(conn, candidate_id), p=p)
            stages_run.append("external_lcs")
            mark_stage_complete("external_lcs")

    return stages_run


def update_candidate_payload(
    conn: sqlite3.Connection,
    candidate_id: str,
    updates: dict,
) -> None:
    """Merge *updates* into the existing payload_json for a candidate."""
    row = conn.execute(
        "SELECT payload_json FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return
    existing = json.loads(row[0]) if row[0] else {}
    existing.update(updates)
    conn.execute(
        "UPDATE candidates SET payload_json = ? WHERE candidate_id = ?",
        (json.dumps(existing, default=str), candidate_id),
    )

    # Also update extracted columns if they match _CANDIDATE_COLUMNS

    col_updates = []
    params = []
    for col, _dtype, etype in _CANDIDATE_COLUMNS:
        if col in updates:
            col_updates.append(f"{col} = ?")
            raw = updates[col]
            if etype == "bool":
                params.append(int(_as_bool(raw)) if raw is not None else None)
            elif etype == "float":
                params.append(_to_float(raw))
            else:
                params.append(str(raw) if raw is not None else None)
    if col_updates:
        params.append(candidate_id)
        conn.execute(
            f"UPDATE candidates SET {', '.join(col_updates)} WHERE candidate_id = ?",
            params,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_stats_stage(payload: dict, lc_path: str, p: Callable | None = None) -> None:
    """Run compute_stats and merge results into payload."""
    try:
        from malca.stats import compute_stats

        candidate_id = Path(lc_path).stem
        parent = str(Path(lc_path).parent)
        _df, summary = compute_stats(candidate_id, parent, compute_ls=True)
        _merge_stats_summary_into_payload(payload, summary)
    except Exception as e:
        if p: p(f"Stats stage failed: {e}")
        else: print(f"[pipeline] Stats stage failed: {e}")


def _run_events_stage(payload: dict, lc_path: str, p: Callable | None = None) -> None:
    """Run process_lightcurve and merge results into payload."""
    try:
        from malca.events import process_lightcurve

        result = process_lightcurve(
            str(lc_path),
            trigger_mode=TRIGGER_MODE,
            logbf_threshold_dip=LOGBF_THRESHOLD_DIP,
            logbf_threshold_jump=LOGBF_THRESHOLD_JUMP,
            significance_threshold=SIGNIFICANCE_THRESHOLD,
            p_points=P_POINTS,
            p_min_dip=P_MIN_DIP,
            p_max_dip=P_MAX_DIP,
            p_min_jump=P_MIN_JUMP,
            p_max_jump=P_MAX_JUMP,
            mag_points=MAG_POINTS,
            run_min_points=RUN_MIN_POINTS,
            max_gap_points=MAX_GAP_POINTS,
            run_max_gap_days=RUN_MAX_GAP_DAYS,
            run_min_duration_days=RUN_MIN_DURATION_DAYS,
            baseline_tag=BASELINE_TAG,
            compute_event_prob=True,
        )
        if isinstance(result, dict):
            payload.update(result)
    except Exception as e:
        if p: p(f"Events stage failed: {e}")
        else: print(f"[pipeline] Events stage failed: {e}")


def _run_characterize_stage(payload: dict, p: Callable | None = None) -> None:
    """Run characterize_candidates_df on a 1-row DataFrame."""
    try:
        from malca.characterize import characterize_candidates_df

        df = pd.DataFrame([payload])
        df_out = _run_with_progress_capture(lambda: characterize_candidates_df(df), p)
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
    except Exception as e:
        if p: p(f"Characterize stage failed: {e}")
        else: print(f"[pipeline] Characterize stage failed: {e}")


def _run_vetting_stage(payload: dict, p: Callable | None = None) -> None:
    """Run vet_candidates on a 1-row DataFrame."""
    try:
        from malca.vetting import vet_candidates

        df = pd.DataFrame([payload])
        df_out = _run_with_progress_capture(
            lambda: vet_candidates(
                df,
                run_atlas=False,
                # (other vetting happens unconditionally in vet_candidates if columns missing)
            ),
            p,
        )
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
    except Exception as e:
        if p: p(f"Vetting stage failed: {e}")
        else: print(f"[pipeline] Vetting stage failed: {e}")


def _resolve_output_dir(conn: sqlite3.Connection, candidate_id: str) -> Path:
    """Resolve the results output directory for a candidate's run."""
    row = conn.execute(
        "SELECT source_path FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row and row[0]:
        src = Path(str(row[0]))
        # If source_path is under a run directory, use its results/ dir
        if src.parent.name == "results":
            return src.parent
        if (src.parent / "results").is_dir():
            return src.parent / "results"
    # Fallback: use the default output directory
    default = Path(__file__).resolve().parents[2] / "output" / "results"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _run_external_lcs_stage(payload: dict, output_dir: Path, p: Callable | None = None) -> None:
    """Run fetch_external_lcs on a 1-row DataFrame."""
    try:
        from malca.vetting import fetch_external_lcs

        df = pd.DataFrame([payload])
        df_out = _run_with_progress_capture(
            lambda: fetch_external_lcs(
                df,
                output_dir=output_dir,
                run_atlas=True,
                run_ztf=True,
                run_gaia_epoch=True,
                run_tess=False,
                run_kepler=False,
                run_aavso=False,
                run_ps1=True,
                run_crts=True,
                progress_callback=p,
            ),
            p,
        )
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
    except Exception as e:
        if p: p(f"External LCs stage failed: {e}")
        else: print(f"[pipeline] External LCs stage failed: {e}")
