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
        "ztf_lc_n_det",
        "gaia_epoch_lc_n_g",
        "tess_n_sectors",
        "neowise_n_epochs",
        "ps1_lc_n_points",
        "crts_lc_n_points",
    ],
    "multi_survey_features": [
        "ms_feature_status",
        "ms_event_type",
        "ms_event_t0_jd",
    ],
}


def _coordinate_value(payload: dict, key: str) -> object | None:
    value = payload.get(key)
    return value if _to_float(value) is not None else None


def _normalize_coordinate_aliases(payload: dict) -> None:
    """Keep both coordinate naming conventions available to stage runners."""
    ra_deg = _coordinate_value(payload, "ra_deg")
    dec_deg = _coordinate_value(payload, "dec_deg")
    ra = _coordinate_value(payload, "ra")
    dec = _coordinate_value(payload, "dec")
    if ra_deg is None and ra is not None:
        payload["ra_deg"] = ra
    if dec_deg is None and dec is not None:
        payload["dec_deg"] = dec
    if ra is None and ra_deg is not None:
        payload["ra"] = ra_deg
    if dec is None and dec_deg is not None:
        payload["dec"] = dec_deg


def _has_coordinates(payload: dict) -> bool:
    return _coordinate_value(payload, "ra_deg") is not None and _coordinate_value(payload, "dec_deg") is not None


def detect_pipeline_status(payload: dict) -> dict[str, str]:
    """Determine which pipeline stages have completed for a candidate.

    Returns a dict mapping stage name → status:
      "complete"  = all signature columns present and non-null
      "partial"   = some signature columns present
      "missing"   = no signature columns present
    """
    result = {}

    def _is_present(value: object) -> bool:
        if value is None:
            return False
        try:
            return not bool(pd.isna(value))
        except Exception:
            return True

    for stage, sig_cols in STAGE_SIGNATURES.items():
        if stage == "multi_survey_features":
            status_value = str(payload.get("ms_feature_status") or "").strip().lower()
            event_type = str(payload.get("ms_event_type") or "").strip()
            has_t0 = _is_present(payload.get("ms_event_t0_jd"))
            if not status_value:
                result[stage] = "missing"
            elif status_value == "ok":
                result[stage] = "complete" if event_type and has_t0 else "partial"
            else:
                result[stage] = "complete" if event_type else "partial"
            continue

        present = sum(1 for c in sig_cols if c in payload and _is_present(payload[c]))
        if present == 0:
            result[stage] = "missing"
        elif present == len(sig_cols):
            result[stage] = "complete"
        else:
            result[stage] = "partial"
    return result


def detect_sed_photometry_status(conn: sqlite3.Connection, candidate_id: str, payload: dict | None = None) -> str:
    """Return completion status for the SED sidecar table."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM sed_photometry WHERE candidate_id = ?",
            (str(candidate_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return "missing"
    count = int(row[0] or 0) if row else 0
    if count > 0:
        return "complete"
    if isinstance(payload, dict) and bool(payload.get("sed_photometry_checked")):
        return "complete"
    return "missing"


def detect_sed_model_status(conn: sqlite3.Connection, candidate_id: str, payload: dict | None = None) -> str:
    """Return completion status for the SED model sidecar tables."""
    try:
        rows = conn.execute(
            "SELECT status FROM sed_model_fits WHERE candidate_id = ?",
            (str(candidate_id),),
        ).fetchall()
    except sqlite3.OperationalError:
        return "missing"
    statuses = [str(row[0] or "").strip().lower() for row in rows]
    if any(status == "ok" for status in statuses):
        return "complete"
    if statuses:
        return "partial"
    return "missing"


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
    _normalize_coordinate_aliases(payload)
    lc_path = row[1] or payload.get("lc_path")

    # 2. Detect current status
    status = detect_pipeline_status(payload)
    status["sed_photometry"] = detect_sed_photometry_status(conn, candidate_id, payload)
    status["sed_model_fit"] = detect_sed_model_status(conn, candidate_id, payload)
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
        if _has_coordinates(payload):
            p("Characterizing...")
            _run_characterize_stage(payload, p)
            stages_run.append("characterize")
            mark_stage_complete("characterize")
        else:
            p("Characterize skipped: missing coordinates")

    if should_run("sed_photometry"):
        p("Fetching SED photometry...")
        n_sed_rows = _run_sed_photometry_stage(
            conn,
            candidate_id,
            payload,
            p,
            replace=("sed_photometry" in force),
        )
        payload["sed_photometry_checked"] = True
        payload["sed_photometry_n_rows"] = int(n_sed_rows)
        stages_run.append("sed_photometry")
        mark_stage_complete("sed_photometry")

    if should_run("sed_model_fit"):
        p("Fitting SED atmosphere model...")
        n_fit_rows, n_curve_rows = _run_sed_model_fit_stage(
            conn,
            candidate_id,
            payload,
            p,
            replace=("sed_model_fit" in force),
        )
        sed_model_status = detect_sed_model_status(conn, candidate_id, payload)
        if sed_model_status == "complete":
            payload["sed_model_fit_checked"] = True
            payload["sed_model_fit_n_rows"] = int(n_fit_rows)
            payload["sed_model_curve_n_rows"] = int(n_curve_rows)
            stages_run.append("sed_model_fit")
            mark_stage_complete("sed_model_fit")
        else:
            payload["sed_model_fit_checked"] = False
            payload["sed_model_fit_n_rows"] = 0
            payload["sed_model_curve_n_rows"] = int(n_curve_rows)
            update_candidate_payload(conn, candidate_id, payload)
            p(f"SED model fit did not complete successfully; status is {sed_model_status}")

    if should_run("vetting"):
        if _has_coordinates(payload):
            p("Vetting crossmatches...")
            _run_vetting_stage(payload, p)
            stages_run.append("vetting")
            mark_stage_complete("vetting")
        else:
            p("Vetting skipped: missing coordinates")

    if should_run("external_lcs"):
        if _has_coordinates(payload):
            p("Fetching external LCs...")
            external_lcs_ok = _run_external_lcs_stage(
                payload,
                output_dir=_resolve_output_dir(conn, candidate_id),
                p=p,
                refresh_cache=("external_lcs" in force),
            )
            stages_run.append("external_lcs")
            if external_lcs_ok:
                mark_stage_complete("external_lcs")
            else:
                update_candidate_payload(conn, candidate_id, payload)
                p("External LCs finished with failures; status may remain partial")
        else:
            p("External LCs skipped: missing coordinates")

    if should_run("multi_survey_features"):
        p("Computing multi-survey features...")
        _run_multi_survey_features_stage(payload, output_dir=_resolve_output_dir(conn, candidate_id), p=p)
        stages_run.append("multi_survey_features")
        mark_stage_complete("multi_survey_features")

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
            p_min_dip=None,
            p_max_dip=None,
            p_min_jump=None,
            p_max_jump=None,
            mag_points=MAG_POINTS,
            run_min_points=RUN_MIN_POINTS,
            max_gap_points=RUN_MAX_GAP_POINTS,
            run_max_gap_days=None,
            run_min_duration_days=0.0,
            baseline_tag=BASELINE_FUNC,
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


def _run_sed_photometry_stage(
    conn: sqlite3.Connection,
    candidate_id: str,
    payload: dict,
    p: Callable | None = None,
    *,
    replace: bool = False,
    sources: str = "default",
) -> int:
    """Run SED photometry fetch/normalization for a 1-row candidate payload."""
    try:
        from malca.review.sed import fetch_sed_photometry, upsert_sed_rows

        df = pd.DataFrame([payload])
        if "candidate_id" not in df.columns:
            df["candidate_id"] = str(candidate_id)
        sed_rows = _run_with_progress_capture(
            lambda: fetch_sed_photometry(df, sources=sources, progress_callback=p),
            p,
        )
        if replace:
            conn.execute("DELETE FROM sed_photometry WHERE candidate_id = ?", (str(candidate_id),))
            conn.commit()
        n_rows = upsert_sed_rows(conn, sed_rows) if isinstance(sed_rows, pd.DataFrame) else 0
        if p:
            p(f"SED photometry: {n_rows} rows")
        return int(n_rows)
    except Exception as e:
        if p: p(f"SED photometry stage failed: {e}")
        else: print(f"[pipeline] SED photometry stage failed: {e}")
        return 0


def _run_sed_model_fit_stage(
    conn: sqlite3.Connection,
    candidate_id: str,
    payload: dict,
    p: Callable | None = None,
    *,
    replace: bool = False,
) -> tuple[int, int]:
    """Run Castelli/Kurucz SED atmosphere fitting for one candidate."""
    try:
        from malca.review.sed import build_sed_dataframe, load_sed_rows
        from malca.sed_model import fit_sed_models, upsert_sed_model_results

        sed_rows = load_sed_rows(conn, str(candidate_id))
        model_rows = build_sed_dataframe(
            payload,
            candidate_id=str(candidate_id),
            external_rows=sed_rows,
            extinction_mode="observed",
        )
        if model_rows.empty:
            if p:
                p("SED model: no SED photometry rows available")
            return 0, 0
        if p and sed_rows.empty:
            p(f"SED model: using {len(model_rows)} payload SED rows")
        df = pd.DataFrame([payload])
        if "candidate_id" not in df.columns:
            df["candidate_id"] = str(candidate_id)
        fits, curves = _run_with_progress_capture(
            lambda: fit_sed_models(df, model_rows, progress_callback=p),
            p,
        )
        n_fit_rows, n_curve_rows = upsert_sed_model_results(
            conn,
            fits,
            curves,
            replace_candidate_ids=[str(candidate_id)] if replace else None,
        )
        if p:
            ok = 0
            if isinstance(fits, pd.DataFrame) and "status" in fits.columns:
                ok = int((fits["status"].astype(str) == "ok").sum())
            p(f"SED model: {ok}/{n_fit_rows} fits ok; {n_curve_rows} curve rows")
        return int(n_fit_rows), int(n_curve_rows)
    except Exception as e:
        if p: p(f"SED model fit stage failed: {e}")
        else: print(f"[pipeline] SED model fit stage failed: {e}")
        return 0, 0


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


def _run_external_lcs_stage(
    payload: dict,
    output_dir: Path,
    p: Callable | None = None,
    *,
    refresh_cache: bool = False,
) -> bool:
    """Run fetch_external_lcs on a 1-row DataFrame."""
    try:
        from malca.vetting import fetch_external_lcs

        _normalize_coordinate_aliases(payload)
        df = pd.DataFrame([payload])
        df_out = _run_with_progress_capture(
            lambda: fetch_external_lcs(
                df,
                output_dir=output_dir,
                run_atlas=False,
                run_ztf=True,
                run_gaia_epoch=True,
                run_tess=True,
                run_neowise=True,
                run_kepler=False,
                run_aavso=False,
                run_ps1=True,
                run_crts=True,
                refresh_cache=refresh_cache,
                progress_callback=p,
            ),
            p,
        )
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for k, v in row.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload[k] = v
            failures = list(getattr(df_out, "attrs", {}).get("external_lc_failures") or [])
            if failures:
                if p:
                    p(f"External LCs finished with module failure(s): {'; '.join(failures[:3])}")
                return False
        return True
    except Exception as e:
        if p: p(f"External LCs stage failed: {e}")
        else: print(f"[pipeline] External LCs stage failed: {e}")
        return False


def _run_multi_survey_features_stage(payload: dict, output_dir: Path, p: Callable | None = None) -> None:
    """Compute event-relative multi-survey features for a 1-row payload."""
    try:
        from malca.multi_survey_features import MS_FEATURE_COLUMNS, compute_multi_survey_features

        df = pd.DataFrame([payload])
        df_out = _run_with_progress_capture(
            lambda: compute_multi_survey_features(df, external_lc_dir=output_dir),
            p,
        )
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            row = df_out.iloc[0].to_dict()
            for key in MS_FEATURE_COLUMNS:
                value = row.get(key)
                if value is None:
                    payload[key] = None
                    continue
                try:
                    if pd.isna(value):
                        payload[key] = None
                        continue
                except Exception:
                    pass
                payload[key] = value
    except Exception as e:
        if p: p(f"Multi-survey features stage failed: {e}")
        else: print(f"[pipeline] Multi-survey features stage failed: {e}")
