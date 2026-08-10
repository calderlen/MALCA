"""Reusable helpers for auditing MALCA period-candidate arbitration on real data.

This module is intentionally evaluation-only.  It runs the current
``PeriodCandidateMethodsConfig`` through proposal generation, harmonic
expansion, fixed-period scoring, local refinement, and the documented
deterministic ranker.  The helpers are used by the companion notebook to make
the intermediate candidate pool inspectable without modifying production or
Review database state.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from malca.core.event_epochs import parse_run_epochs_json
from malca.core.phase import align_v_to_g_magnitude
from malca.core.utils import read_lc_dat2
from malca.evaluation.period_candidate_methods import (
    PeriodCandidateMethodsConfig,
    build_candidate_bank,
    expand_harmonic_candidates,
    refine_scored_candidates,
    run_global_period_searches,
    score_candidate_bank,
    select_scoring_shortlist,
)
from malca.evaluation.period_candidate_ranker import rank_deterministic_baseline
from malca.evaluation.period_cost_accuracy import period_match_arrays
from malca.stv.periodicity_gate import prepare_periodicity_lightcurve


EXTERNAL_PERIOD_COLUMNS: tuple[str, ...] = (
    "gaia_eb_period",
    "vsx_period",
    "asassn_var_period",
    "ztf_var_period",
    "period_ogle_days",
)
DEFAULT_MATCH_TOLERANCE = 0.05


def current_default_config_table(
    config: PeriodCandidateMethodsConfig | None = None,
) -> pd.DataFrame:
    """Return the current candidate-suite configuration as a readable table."""

    cfg = config or PeriodCandidateMethodsConfig()
    rows: list[dict[str, Any]] = []
    for name, value in asdict(cfg).items():
        if isinstance(value, (tuple, list, dict)):
            rendered = json.dumps(value, sort_keys=True)
        elif value is None:
            rendered = "None"
        else:
            rendered = str(value)
        rows.append({"parameter": name, "value": rendered, "python_type": type(value).__name__})
    return pd.DataFrame(rows)


def annotate_external_period_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Label clean-reference, no-external-period, and conflicted rows."""

    result = frame.copy()
    available = pd.DataFrame(index=result.index)
    for column in EXTERNAL_PERIOD_COLUMNS:
        if column in result:
            values = pd.to_numeric(result[column], errors="coerce")
            available[column] = np.isfinite(values) & values.gt(0)
        else:
            available[column] = False
    result["has_any_external_period"] = available.any(axis=1)
    clean = result.get(
        "catalog_reference_available",
        pd.Series(False, index=result.index),
    ).fillna(False).astype(bool)
    result["external_period_group"] = np.select(
        [clean, ~result["has_any_external_period"]],
        ["external_reference", "no_external_period"],
        default="unclean_or_conflicting_external",
    )
    return result


def _stable_token(value: object, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{value}".encode()).hexdigest()


def _balanced_take(frame: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0].copy()
    work = frame.copy()
    work["_magnitude_bin"] = pd.cut(
        pd.to_numeric(work["median_mag"], errors="coerce"),
        bins=[12.0, 13.0, 14.0, 15.0],
        labels=["12-13", "13-14", "14-15"],
        include_lowest=True,
    ).astype("string")
    point_rank = pd.to_numeric(work["n_points"], errors="coerce").rank(
        method="first", pct=True
    )
    work["_point_bin"] = pd.cut(
        point_rank,
        bins=[0.0, 0.5, 1.0],
        labels=["lower_n", "higher_n"],
        include_lowest=True,
    ).astype("string")
    work["sample_stratum"] = (
        work["external_period_group"].astype(str)
        + "|"
        + work["_magnitude_bin"].fillna("unknown").astype(str)
        + "|"
        + work["_point_bin"].fillna("unknown").astype(str)
    )
    work["_stable_token"] = work["candidate_id"].map(
        lambda value: _stable_token(value, seed)
    )
    queues = {
        stratum: list(group.sort_values("_stable_token", kind="stable").index)
        for stratum, group in work.groupby("sample_stratum", sort=True)
    }
    selected: list[object] = []
    while len(selected) < min(int(n), len(work)):
        added = False
        for stratum in sorted(queues):
            if queues[stratum]:
                selected.append(queues[stratum].pop(0))
                added = True
                if len(selected) >= min(int(n), len(work)):
                    break
        if not added:
            break
    return work.loc[selected].drop(
        columns=["_magnitude_bin", "_point_bin", "_stable_token"]
    )


def select_healthy_period_cohort(
    frame: pd.DataFrame,
    *,
    n_per_group: int = 12,
    seed: int = 20260803,
    magnitude_min: float = 12.0,
    magnitude_max: float = 15.0,
    min_points: int = 100,
) -> pd.DataFrame:
    """Select balanced healthy light curves with and without external periods.

    ``no_external_period`` means that none of the five integrated external
    period fields is finite and positive; it does not merely mean that the
    catalog-consensus cleaning step rejected a conflicting reference.
    """

    required = {"candidate_id", "lc_path", "median_mag", "n_points"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cohort frame is missing required columns: {missing}")
    work = annotate_external_period_groups(frame)
    work["lc_exists"] = work["lc_path"].map(
        lambda value: Path(str(value)).is_file() if pd.notna(value) else False
    )
    magnitude = pd.to_numeric(work["median_mag"], errors="coerce")
    n_points = pd.to_numeric(work["n_points"], errors="coerce")
    healthy = (
        work["lc_exists"]
        & magnitude.between(float(magnitude_min), float(magnitude_max), inclusive="both")
        & n_points.ge(int(min_points))
    )
    eligible = work.loc[
        healthy
        & work["external_period_group"].isin(
            ["external_reference", "no_external_period"]
        )
    ].copy()
    pieces = []
    for offset, group_name in enumerate(("external_reference", "no_external_period")):
        group = eligible.loc[eligible["external_period_group"].eq(group_name)]
        if len(group) < int(n_per_group):
            raise ValueError(
                f"Only {len(group)} healthy rows are available for {group_name}; "
                f"requested {n_per_group}"
            )
        pieces.append(_balanced_take(group, int(n_per_group), seed=int(seed) + offset))
    result = pd.concat(pieces, ignore_index=True)
    result["cohort_group"] = result["external_period_group"]
    result["cohort_seed"] = int(seed)
    result["cohort_n_per_group"] = int(n_per_group)
    return result.sort_values(
        ["cohort_group", "sample_stratum", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def load_prepared_light_curve(record: Mapping[str, Any]) -> pd.DataFrame:
    """Load and apply the same canonical preparation used by the V3 benchmark."""

    path = Path(str(record["lc_path"]))
    df_g, df_v = read_lc_dat2(
        path.stem,
        str(path.parent),
        file_ext=path.suffix.lstrip("."),
    )
    frame = pd.concat([df_g, df_v], ignore_index=True)
    frame = prepare_periodicity_lightcurve(frame)
    frame, alignment = align_v_to_g_magnitude(frame)
    frame = frame.sort_values("JD", kind="stable").reset_index(drop=True)
    frame.attrs["v_to_g_alignment"] = alignment
    return frame


def event_epochs_from_record(record: Mapping[str, Any]) -> list[float]:
    return [
        float(epoch.center_jd)
        for epoch in parse_run_epochs_json(record.get("dip_run_epochs_json"))
    ]


def _finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _match_single(
    period: object,
    reference: object,
    *,
    tolerance: float,
) -> tuple[bool | float, bool | float]:
    if not (_finite_positive(period) and _finite_positive(reference)):
        return np.nan, np.nan
    match = period_match_arrays(
        [float(period)],
        [float(reference)],
        tolerance=float(tolerance),
    )
    return bool(match["is_exact"][0]), bool(match["is_harmonic_family"][0])


def _match_any(
    periods: Sequence[object],
    reference: object,
    *,
    tolerance: float,
) -> tuple[bool | float, bool | float]:
    if not _finite_positive(reference):
        return np.nan, np.nan
    values = [float(value) for value in periods if _finite_positive(value)]
    if not values:
        return False, False
    match = period_match_arrays(
        values,
        np.full(len(values), float(reference)),
        tolerance=float(tolerance),
    )
    return bool(match["is_exact"].any()), bool(match["is_harmonic_family"].any())


def _jsonify_objects(frame: pd.DataFrame) -> pd.DataFrame:
    """Make nested provenance columns safe for portable Parquet output."""

    result = frame.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        def convert(value: object) -> object:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (tuple, list, dict, set)):
                return json.dumps(
                    list(value) if isinstance(value, set) else value,
                    sort_keys=True,
                    default=str,
                )
            return value

        result[column] = result[column].map(convert)
    return result


def _candidate_records(
    candidates: Sequence[Any],
    *,
    source_id: str,
    stage: str,
) -> pd.DataFrame:
    rows = []
    for index, candidate in enumerate(candidates):
        row = candidate.to_record()
        row.update(
            {
                "source_id": source_id,
                "candidate_stage": stage,
                "stage_index": int(index),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_current_deterministic_arbitration(
    record: Mapping[str, Any],
    *,
    config: PeriodCandidateMethodsConfig | None = None,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> dict[str, pd.DataFrame]:
    """Run every current arbitration stage for one real light curve."""

    cfg = config or PeriodCandidateMethodsConfig()
    source_id = str(record["candidate_id"])
    total_started = time.perf_counter()

    prepare_started = time.perf_counter()
    light_curve = load_prepared_light_curve(record)
    jd = light_curve["JD"].to_numpy(dtype=float)
    mag = light_curve["mag"].to_numpy(dtype=float)
    err = light_curve["error"].to_numpy(dtype=float)
    prepare_seconds = time.perf_counter() - prepare_started
    if jd.size < int(cfg.min_points):
        raise ValueError(f"Only {jd.size} prepared points remain for {source_id}")
    baseline = float(np.ptp(jd))
    event_epochs = event_epochs_from_record(record)

    search_started = time.perf_counter()
    searches = run_global_period_searches(
        jd,
        mag,
        err,
        event_epochs=event_epochs,
        config=cfg,
        methods=cfg.enabled_global_methods,
    )
    search_seconds = time.perf_counter() - search_started
    raw_candidates = [
        candidate
        for result in searches.values()
        for candidate in result.candidates
        if result.status == "ok"
    ]
    search_status = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "method": method,
                "status": result.status,
                "candidate_count": len(result.candidates),
                "grid_count": int(len(result.period_grid_days)),
                "objective": result.objective,
                "message": result.message,
            }
            for method, result in searches.items()
        ]
    )
    proposal_rows = pd.concat(
        [
            _candidate_records(
                result.candidates,
                source_id=source_id,
                stage="global_proposal",
            ).assign(search_method=method)
            for method, result in searches.items()
            if result.candidates
        ],
        ignore_index=True,
    ) if raw_candidates else pd.DataFrame()

    bank_started = time.perf_counter()
    merged = build_candidate_bank(
        searches,
        baseline_days=baseline,
        config=cfg,
        expand_harmonics=False,
    )
    expanded = expand_harmonic_candidates(
        merged,
        baseline_days=baseline,
        config=cfg,
    )
    shortlist = select_scoring_shortlist(expanded, config=cfg)
    bank_seconds = time.perf_counter() - bank_started

    score_started = time.perf_counter()
    initial_scores = score_candidate_bank(
        jd,
        mag,
        shortlist,
        err,
        event_epochs=event_epochs,
        config=cfg,
    )
    score_seconds = time.perf_counter() - score_started

    refine_started = time.perf_counter()
    refined = refine_scored_candidates(
        shortlist,
        initial_scores,
        baseline_days=baseline,
        time=jd,
        mag=mag,
        err=err,
        config=cfg,
        include_original=False,
    )
    refined_scores = score_candidate_bank(
        jd,
        mag,
        refined,
        err,
        event_epochs=event_epochs,
        config=cfg,
    )
    refine_seconds = time.perf_counter() - refine_started

    rows: list[dict[str, Any]] = []
    for stage, stage_scores in (
        ("scored_shortlist", initial_scores),
        ("local_refinement", refined_scores),
    ):
        for index, score in enumerate(stage_scores):
            row = score.to_record()
            row.update(
                {
                    "candidate_id": f"{source_id}::{stage}::{index:04d}",
                    "source_id": source_id,
                    "base_view_id": source_id,
                    "candidate_stage": stage,
                    "stage_index": int(index),
                    "baseline_days": baseline,
                }
            )
            rows.append(row)
    score_frame = pd.DataFrame(rows)
    if score_frame.empty:
        raise ValueError(f"No candidates reached deterministic arbitration for {source_id}")

    rank_started = time.perf_counter()
    ranked = rank_deterministic_baseline(
        score_frame,
        group_col="base_view_id",
        include_components=True,
    )
    rank_seconds = time.perf_counter() - rank_started
    ranked = ranked.sort_values("baseline_rank", kind="stable").reset_index(drop=True)

    reference = record.get("catalog_reference_period")
    has_reference = _finite_positive(reference)
    if has_reference:
        match = period_match_arrays(
            pd.to_numeric(ranked["period_days"], errors="coerce").to_numpy(),
            np.full(len(ranked), float(reference)),
            tolerance=float(match_tolerance),
        )
        ranked["catalog_exact_match"] = match["is_exact"]
        ranked["catalog_family_match"] = match["is_harmonic_family"]
        ranked["catalog_best_harmonic"] = match["nearest_harmonic_factor"]
        ranked["catalog_relative_error"] = match["relative_error"]
    else:
        ranked["catalog_exact_match"] = np.nan
        ranked["catalog_family_match"] = np.nan
        ranked["catalog_best_harmonic"] = np.nan
        ranked["catalog_relative_error"] = np.nan

    selected = ranked.iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else None
    selected_period = float(selected["period_days"])
    selected_exact, selected_family = _match_single(
        selected_period,
        reference,
        tolerance=float(match_tolerance),
    )
    all_oracle_exact, all_oracle_family = _match_any(
        [candidate.period_days for candidate in expanded]
        + [candidate.period_days for candidate in refined],
        reference,
        tolerance=float(match_tolerance),
    )
    ranked_oracle_exact, ranked_oracle_family = _match_any(
        ranked["period_days"].tolist(),
        reference,
        tolerance=float(match_tolerance),
    )
    stored_period = record.get("periodicity_period")
    selected_stored_exact, selected_stored_family = _match_single(
        selected_period,
        stored_period,
        tolerance=float(match_tolerance),
    )
    score_margin = (
        float(selected["baseline_score"] - runner_up["baseline_score"])
        if runner_up is not None
        else np.nan
    )

    selected_methods = selected.get("proposal_contributing_methods", ())
    if isinstance(selected_methods, str):
        selected_methods_json = selected_methods
    else:
        selected_methods_json = json.dumps(selected_methods, default=str)
    summary = pd.DataFrame(
        [
            {
                "candidate_id": source_id,
                "cohort_group": record.get("cohort_group"),
                "sample_stratum": record.get("sample_stratum"),
                "lc_path": str(record.get("lc_path", "")),
                "median_mag": float(record.get("median_mag", np.nan)),
                "n_points_input": float(record.get("n_points", np.nan)),
                "n_points_prepared": int(len(light_curve)),
                "baseline_days": baseline,
                "event_epoch_count": int(len(event_epochs)),
                "catalog_reference_period": (
                    float(reference) if has_reference else np.nan
                ),
                "catalog_reference_tier": record.get("catalog_reference_tier"),
                "stored_period_days": (
                    float(stored_period) if _finite_positive(stored_period) else np.nan
                ),
                "selected_period_days": selected_period,
                "runner_up_period_days": (
                    float(runner_up["period_days"]) if runner_up is not None else np.nan
                ),
                "selected_candidate_stage": selected["candidate_stage"],
                "selected_proposal_method": selected.get("proposal_method"),
                "selected_contributing_methods": selected_methods_json,
                "selected_independent_method_family_count": float(
                    selected.get("proposal_independent_method_family_count", np.nan)
                ),
                "selected_score": float(selected["baseline_score"]),
                "runner_up_score": (
                    float(runner_up["baseline_score"]) if runner_up is not None else np.nan
                ),
                "score_margin": score_margin,
                "selected_exact_match": selected_exact,
                "selected_family_match": selected_family,
                "all_candidate_oracle_exact": all_oracle_exact,
                "all_candidate_oracle_family": all_oracle_family,
                "ranked_pool_oracle_exact": ranked_oracle_exact,
                "ranked_pool_oracle_family": ranked_oracle_family,
                "selected_vs_stored_exact": selected_stored_exact,
                "selected_vs_stored_family": selected_stored_family,
                "selected_template_q": float(selected.get("template_q", np.nan)),
                "selected_template_scatter_ratio": float(
                    selected.get("template_scatter_ratio", np.nan)
                ),
                "raw_candidate_count": int(len(raw_candidates)),
                "merged_candidate_count": int(len(merged)),
                "expanded_candidate_count": int(len(expanded)),
                "scored_candidate_count": int(len(initial_scores)),
                "refined_candidate_count": int(len(refined_scores)),
                "ranked_candidate_count": int(len(ranked)),
                "preparation_seconds": prepare_seconds,
                "global_search_seconds": search_seconds,
                "candidate_bank_seconds": bank_seconds,
                "fixed_scoring_seconds": score_seconds,
                "refinement_seconds": refine_seconds,
                "ranking_seconds": rank_seconds,
                "total_seconds": time.perf_counter() - total_started,
                "status": "ok",
                "error": "",
            }
        ]
    )
    stages = pd.DataFrame(
        {
            "candidate_id": source_id,
            "cohort_group": record.get("cohort_group"),
            "stage": [
                "raw proposals",
                "merged proposals",
                "harmonic expansion",
                "fixed scoring shortlist",
                "local refinements",
                "final ranked pool",
            ],
            "stage_order": np.arange(6, dtype=int),
            "candidate_count": [
                len(raw_candidates),
                len(merged),
                len(expanded),
                len(initial_scores),
                len(refined_scores),
                len(ranked),
            ],
        }
    )
    return {
        "summary": _jsonify_objects(summary),
        "candidate_scores": _jsonify_objects(ranked),
        "proposals": _jsonify_objects(proposal_rows),
        "stage_counts": _jsonify_objects(stages),
        "search_status": _jsonify_objects(search_status),
    }


def run_current_deterministic_arbitration_safe(
    record: Mapping[str, Any],
    *,
    config: PeriodCandidateMethodsConfig | None = None,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> dict[str, pd.DataFrame]:
    """Return an error summary instead of losing an entire parallel batch."""

    try:
        return run_current_deterministic_arbitration(
            record,
            config=config,
            match_tolerance=match_tolerance,
        )
    except Exception as exc:  # pragma: no cover - exercised by real-data failures
        source_id = str(record.get("candidate_id", "<missing>"))
        summary = pd.DataFrame(
            [
                {
                    "candidate_id": source_id,
                    "cohort_group": record.get("cohort_group"),
                    "sample_stratum": record.get("sample_stratum"),
                    "lc_path": str(record.get("lc_path", "")),
                    "catalog_reference_period": record.get("catalog_reference_period"),
                    "status": "error",
                    "error": repr(exc),
                }
            ]
        )
        empty = pd.DataFrame()
        return {
            "summary": summary,
            "candidate_scores": empty,
            "proposals": empty,
            "stage_counts": empty,
            "search_status": empty,
        }


def _safe_source_name(source_id: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_id)).strip("._")
    if not safe:
        safe = hashlib.sha256(str(source_id).encode()).hexdigest()[:16]
    return safe


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def save_candidate_audit(
    result: Mapping[str, pd.DataFrame],
    *,
    checkpoint_root: Path,
) -> Path:
    summary = result["summary"]
    if summary.empty:
        raise ValueError("Audit result has no summary row")
    source_id = str(summary.iloc[0]["candidate_id"])
    root = Path(checkpoint_root) / _safe_source_name(source_id)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        _atomic_parquet(frame, root / f"{name}.parquet")
    return root


def completed_audit_source_ids(checkpoint_root: Path) -> set[str]:
    completed: set[str] = set()
    for path in Path(checkpoint_root).glob("*/summary.parquet"):
        try:
            frame = pd.read_parquet(path, columns=["candidate_id"])
        except Exception:
            continue
        if not frame.empty:
            completed.add(str(frame.iloc[0]["candidate_id"]))
    return completed


def collect_candidate_audits(checkpoint_root: Path) -> dict[str, pd.DataFrame]:
    collections: dict[str, list[pd.DataFrame]] = {
        "summary": [],
        "candidate_scores": [],
        "proposals": [],
        "stage_counts": [],
        "search_status": [],
    }
    for root in sorted(Path(checkpoint_root).iterdir() if Path(checkpoint_root).is_dir() else []):
        if not root.is_dir():
            continue
        for name in collections:
            path = root / f"{name}.parquet"
            if path.is_file():
                collections[name].append(pd.read_parquet(path))
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in collections.items()
    }


def write_aggregate_audits(
    audits: Mapping[str, pd.DataFrame],
    *,
    output_root: Path,
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in audits.items():
        _atomic_parquet(frame, root / f"{name}.parquet")


def topk_catalog_recovery(
    candidate_scores: pd.DataFrame,
    *,
    top_k: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> pd.DataFrame:
    """Calculate selected-pool catalog recovery as a function of rank depth."""

    required = {
        "source_id",
        "baseline_rank",
        "catalog_exact_match",
        "catalog_family_match",
    }
    if candidate_scores.empty or not required.issubset(candidate_scores.columns):
        return pd.DataFrame()
    rows = []
    for k in top_k:
        subset = candidate_scores.loc[
            pd.to_numeric(candidate_scores["baseline_rank"], errors="coerce").le(int(k))
        ]
        per_source = subset.groupby("source_id", as_index=False).agg(
            exact=("catalog_exact_match", "max"),
            family=("catalog_family_match", "max"),
        )
        exact = pd.to_numeric(per_source["exact"], errors="coerce")
        family = pd.to_numeric(per_source["family"], errors="coerce")
        rows.append(
            {
                "top_k": int(k),
                "n_sources": int(exact.notna().sum()),
                "exact_recovery": float(exact.mean()) if exact.notna().any() else np.nan,
                "family_recovery": float(family.mean()) if family.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def choose_case_studies(
    summaries: pd.DataFrame,
    *,
    per_group: int = 4,
) -> pd.DataFrame:
    """Choose informative catalog matches/misses and high/low-evidence unknowns."""

    if summaries.empty:
        return pd.DataFrame(columns=["candidate_id", "case_reason"])
    good = summaries.loc[summaries.get("status", "ok").eq("ok")].copy()
    chosen: list[dict[str, str]] = []
    used: set[str] = set()

    def add(row: pd.Series | None, reason: str) -> None:
        if row is None:
            return
        source_id = str(row["candidate_id"])
        if source_id in used:
            return
        used.add(source_id)
        chosen.append({"candidate_id": source_id, "case_reason": reason})

    known = good.loc[good["cohort_group"].eq("external_reference")]
    if not known.empty:
        exact = known.loc[known["selected_exact_match"].eq(True)]
        family_only = known.loc[
            known["selected_family_match"].eq(True)
            & ~known["selected_exact_match"].eq(True)
        ]
        mismatch = known.loc[~known["selected_family_match"].eq(True)]
        add(exact.sort_values("score_margin", ascending=False).iloc[0] if not exact.empty else None, "catalog exact match")
        add(family_only.sort_values("score_margin", ascending=False).iloc[0] if not family_only.empty else None, "catalog harmonic-family match")
        add(mismatch.sort_values("score_margin", ascending=False).iloc[0] if not mismatch.empty else None, "catalog-family disagreement")
        for _, row in known.sort_values("score_margin", ascending=True).iterrows():
            if sum(item["candidate_id"] in set(known["candidate_id"].astype(str)) for item in chosen) >= int(per_group):
                break
            add(row, "small arbitration margin")

    unknown = good.loc[good["cohort_group"].eq("no_external_period")]
    if not unknown.empty:
        add(unknown.sort_values("score_margin", ascending=False).iloc[0], "no catalog: strongest score margin")
        add(unknown.sort_values("score_margin", ascending=True).iloc[0], "no catalog: weakest score margin")
        add(
            unknown.sort_values("selected_independent_method_family_count", ascending=False).iloc[0],
            "no catalog: broadest method support",
        )
        add(
            unknown.sort_values("selected_template_q", ascending=True).iloc[0],
            "no catalog: strongest phase repeatability",
        )

    cases = pd.DataFrame(chosen)
    if cases.empty:
        return cases
    return cases.merge(good, on="candidate_id", how="left")


def utility_family_table(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse the deterministic ranker's many utility columns for display."""

    utility_columns = [
        column for column in candidate_scores if column.startswith("baseline_score__")
    ]
    if not utility_columns:
        return pd.DataFrame()
    families = {
        "proposal/support": ("proposal_",),
        "LS/Fourier": ("ls_", "fourier_"),
        "dispersion/smoother": ("pdm_", "ce_", "lafler_", "supersmoother_"),
        "BLS": ("bls_",),
        "phase repeatability": ("template_", "odd_even_"),
        "event timing": ("event_",),
    }
    result = candidate_scores[[
        column
        for column in (
            "source_id",
            "candidate_id",
            "period_days",
            "baseline_rank",
            "baseline_score",
            "candidate_stage",
        )
        if column in candidate_scores
    ]].copy()
    for family, prefixes in families.items():
        members = [
            column
            for column in utility_columns
            if any(column.removeprefix("baseline_score__").startswith(prefix) for prefix in prefixes)
        ]
        result[family] = (
            candidate_scores[members].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            if members
            else np.nan
        )
    return result


def harmonic_label(ratio: object) -> str:
    if not _finite_positive(ratio):
        return "missing"
    ratio_value = float(ratio)
    factors = (0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0)
    factor = min(factors, key=lambda value: abs(math.log(ratio_value / value)))
    return f"{factor:g}x"


__all__ = [
    "DEFAULT_MATCH_TOLERANCE",
    "EXTERNAL_PERIOD_COLUMNS",
    "annotate_external_period_groups",
    "choose_case_studies",
    "collect_candidate_audits",
    "completed_audit_source_ids",
    "current_default_config_table",
    "event_epochs_from_record",
    "harmonic_label",
    "load_prepared_light_curve",
    "run_current_deterministic_arbitration",
    "run_current_deterministic_arbitration_safe",
    "save_candidate_audit",
    "select_healthy_period_cohort",
    "topk_catalog_recovery",
    "utility_family_table",
    "write_aggregate_audits",
]
