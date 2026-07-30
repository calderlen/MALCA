#!/usr/bin/env python3
"""Backfill period consensus for an existing review DB.

Recomputes long-period LS + event-informed consensus for every candidate and
flips ``periodicity_period`` / ``phase_period_days`` to the consensus result.
Dip epochs prefer, in order:

1. Existing ``dip_run_epochs_json`` on the candidate row (from a prior events run).
2. Full ``score_lightcurve`` (GP + ``build_runs``) -- default, matches the pipeline.
3. Lightweight ``build_runs`` residual detector (``--fast``).

Example
-------
::

    python scripts/backfill_period_consensus.py \\
        --review-db output/runs/dat3-full-extended_2026-07-01-v4/review/review.db \\
        --workers 8 \\
        --limit 100   # dry-run subset first

"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from malca.core.event_epochs import (  # noqa: E402
    detect_dip_epochs_via_events,
    dip_center_jds,
    parse_run_epochs_json,
    serialize_run_summaries,
)
from malca.core.period_pipeline import compute_period_consensus_for_lc  # noqa: E402
from malca.core.stats import compute_ce_stats, compute_pdm_stats  # noqa: E402
from malca.core.utils import clean_lc  # noqa: E402
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame  # noqa: E402
from malca.review.store import ensure_review_db_schema  # noqa: E402
from malca.stv.periodicity_gate import prepare_periodicity_lightcurve  # noqa: E402


UPDATE_COLUMNS = (
    "periodicity_period",
    "periodicity_method",
    "phase_period_days",
    "phase_source",
    "period_confidence",
    "period_method",
    "period_baseline_cycles",
    "period_confidence_reason",
    "dip_run_epochs_json",
    "dip_epochs_source",
    "dip_epochs_count",
    "long_ls_period_days",
    "long_ls_peak_power",
    "long_ls_fap_bootstrap",
    "long_ls_baseline_cycles",
    "long_ls_is_significant",
    "long_ls_status",
)


def _resolve_lc_path(row: dict, lc_root: Path | None) -> Path | None:
    for key in ("lc_path", "source_path"):
        raw = row.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path
        if lc_root is not None:
            candidate = lc_root / path.name
            if candidate.is_file():
                return candidate
            # Common layout: bundle_assets/lightcurves/<id>.dat3
            for pattern in (
                lc_root / "bundle_assets" / "lightcurves" / path.name,
                lc_root / "lightcurves" / path.name,
            ):
                if pattern.is_file():
                    return pattern
    asas = row.get("asas_sn_id") or row.get("source_id") or ""
    asas = str(asas).strip()
    if asas and lc_root is not None:
        for ext in (".dat3", ".dat2", ".raw2", ".dat"):
            for pattern in (
                lc_root / "bundle_assets" / "lightcurves" / f"{asas}{ext}",
                lc_root / "lightcurves" / f"{asas}{ext}",
                lc_root / f"{asas}{ext}",
            ):
                if pattern.is_file():
                    return pattern
    return None


def _process_one(payload: dict) -> dict:
    """Worker entry point. Returns a flat update dict keyed by candidate_id."""
    candidate_id = str(payload["candidate_id"])
    lc_path = payload.get("lc_path")
    use_events = bool(payload.get("use_events_pipeline", True))
    n_bootstrap = int(payload.get("n_bootstrap", 200))
    existing_json = payload.get("dip_run_epochs_json")

    out = {
        "candidate_id": candidate_id,
        "ok": False,
        "error": None,
    }
    for col in UPDATE_COLUMNS:
        out[col] = None

    if not lc_path:
        out["error"] = "no_lc_path"
        return out

    try:
        raw = load_lightcurve_df(lc_path)
        df = to_asassn_algorithm_frame(raw)
        df = clean_lc(df)
        df = prepare_periodicity_lightcurve(df)
        if len(df) < 20:
            out["error"] = f"too_few_points:{len(df)}"
            return out

        jd = df["JD"].to_numpy(float)
        mag = df["mag"].to_numpy(float)
        err = df["error"].to_numpy(float) if "error" in df.columns else None

        # Prefer existing events JSON; otherwise run score_lightcurve (or lightweight).
        dip_json = existing_json
        if not dip_json and use_events:
            epochs = detect_dip_epochs_via_events(df)
            if epochs:
                dip_json = serialize_run_summaries(
                    [
                        {
                            "start_jd": e.start_jd,
                            "end_jd": e.end_jd,
                            "run_max": e.peak_significance,
                            "n_points": e.n_points,
                            "duration_days": e.duration_days,
                            "kept": True,
                        }
                        for e in epochs
                    ]
                )

        pdm = compute_pdm_stats(jd, mag, err if err is not None else np.full_like(mag, 0.02), n_bootstrap=0)
        ce = compute_ce_stats(jd, mag, err if err is not None else np.full_like(mag, 0.02), n_bootstrap=0)

        consensus = compute_period_consensus_for_lc(
            jd,
            mag,
            err,
            pdm_result=pdm,
            ce_result=ce,
            dip_epochs_json=dip_json,
            detect_dip_epochs_fallback=True,
            use_events_pipeline=False,  # already handled above
            long_ls_kwargs={"n_bootstrap": n_bootstrap},
        )

        # Persist whatever epochs consensus actually used.
        if not dip_json and consensus.get("dip_epochs_used"):
            used = list(consensus["dip_epochs_used"])
            dip_json = serialize_run_summaries(
                [
                    {
                        "start_jd": float(c),
                        "end_jd": float(c),
                        "run_max": float("nan"),
                        "n_points": 1,
                        "duration_days": 0.0,
                        "kept": True,
                    }
                    for c in used
                ]
            )

        period = consensus.get("period_consensus_days")
        method = consensus.get("period_method") or "none"
        confidence = consensus.get("period_confidence") or "none"

        out.update(
            {
                "ok": True,
                "periodicity_period": period,
                "periodicity_method": method,
                "phase_period_days": period,
                "phase_source": method,
                "period_confidence": confidence,
                "period_method": method,
                "period_baseline_cycles": consensus.get("period_baseline_cycles"),
                "period_confidence_reason": consensus.get("period_confidence_reason"),
                "dip_run_epochs_json": dip_json,
                "dip_epochs_source": consensus.get("dip_epochs_source"),
                "dip_epochs_count": consensus.get("dip_epochs_count"),
                "long_ls_period_days": consensus.get("long_ls_period_days"),
                "long_ls_peak_power": consensus.get("long_ls_peak_power"),
                "long_ls_fap_bootstrap": consensus.get("long_ls_fap_bootstrap"),
                "long_ls_baseline_cycles": consensus.get("long_ls_baseline_cycles"),
                "long_ls_is_significant": int(bool(consensus.get("long_ls_is_significant"))),
                "long_ls_status": consensus.get("long_ls_status"),
            }
        )
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        return out


def _load_candidates(conn: sqlite3.Connection, *, limit: int | None, candidate_ids: list[str] | None) -> list[dict]:
    cols = [
        "candidate_id",
        "asas_sn_id",
        "source_id",
        "lc_path",
        "source_path",
        "dip_run_epochs_json",
    ]
    # Some older DBs may lack dip_run_epochs_json until ensure_schema runs.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(candidates)")}
    cols = [c for c in cols if c in existing or c == "candidate_id"]
    sql = f"SELECT {', '.join(cols)} FROM candidates"
    params: list[object] = []
    if candidate_ids:
        placeholders = ",".join("?" for _ in candidate_ids)
        sql += f" WHERE candidate_id IN ({placeholders})"
        params.extend(candidate_ids)
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    return [dict(zip(cols, row)) for row in rows]


def _apply_updates(conn: sqlite3.Connection, updates: list[dict], *, dry_run: bool) -> int:
    if dry_run or not updates:
        return 0
    set_clause = ", ".join(f"{col} = ?" for col in UPDATE_COLUMNS)
    sql = f"UPDATE candidates SET {set_clause} WHERE candidate_id = ?"
    n = 0
    for upd in updates:
        if not upd.get("ok"):
            continue
        values = [upd.get(col) for col in UPDATE_COLUMNS]
        values.append(upd["candidate_id"])
        conn.execute(sql, values)
        n += 1
    conn.commit()
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-db", type=Path, required=True)
    parser.add_argument("--lc-root", type=Path, default=None, help="Run dir or lightcurve root for path resolution")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-id", action="append", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use lightweight build_runs residual detector instead of score_lightcurve",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args(argv)

    db_path = args.review_db.expanduser().resolve()
    if not db_path.is_file():
        raise SystemExit(f"review DB not found: {db_path}")

    lc_root = args.lc_root.expanduser().resolve() if args.lc_root else db_path.parent.parent

    conn = sqlite3.connect(str(db_path))
    ensure_review_db_schema(db_path)
    candidates = _load_candidates(conn, limit=args.limit, candidate_ids=args.candidate_id)
    print(f"Loaded {len(candidates)} candidates from {db_path}")

    payloads = []
    for row in candidates:
        lc_path = _resolve_lc_path(row, lc_root)
        payloads.append(
            {
                "candidate_id": row["candidate_id"],
                "lc_path": str(lc_path) if lc_path else None,
                "dip_run_epochs_json": row.get("dip_run_epochs_json"),
                "use_events_pipeline": not bool(args.fast),
                "n_bootstrap": int(args.n_bootstrap),
            }
        )

    n_ok = 0
    n_err = 0
    pending: list[dict] = []
    t0 = time.time()

    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = ((_process_one(p), p) for p in payloads)
        results_iter = ((res, p) for res, p in iterator)
        # Sequential
        for i, payload in enumerate(payloads, start=1):
            result = _process_one(payload)
            if result.get("ok"):
                n_ok += 1
                pending.append(result)
            else:
                n_err += 1
                print(f"[{i}/{len(payloads)}] FAIL {result['candidate_id']}: {result.get('error')}")
            if len(pending) >= args.checkpoint_every:
                written = _apply_updates(conn, pending, dry_run=args.dry_run)
                print(f"  checkpoint wrote {written} (ok={n_ok} err={n_err})")
                pending = []
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, p): p for p in payloads}
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                if result.get("ok"):
                    n_ok += 1
                    pending.append(result)
                else:
                    n_err += 1
                    print(f"[{i}/{len(payloads)}] FAIL {result['candidate_id']}: {result.get('error')}")
                if len(pending) >= args.checkpoint_every:
                    written = _apply_updates(conn, pending, dry_run=args.dry_run)
                    print(f"  checkpoint wrote {written} (ok={n_ok} err={n_err})")
                    pending = []

    written = _apply_updates(conn, pending, dry_run=args.dry_run)
    elapsed = time.time() - t0
    print(
        f"Done. ok={n_ok} err={n_err} written={written} "
        f"dry_run={args.dry_run} elapsed={elapsed:.1f}s"
    )
    conn.close()
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
