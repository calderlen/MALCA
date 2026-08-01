"""Utilities for period-finding cost-versus-accuracy evaluations.

The functions in this module deliberately keep three questions separate:

* coverage: did a strategy return any period?
* selected-period agreement: did one selected period agree with an
  independently supplied reference?
* candidate oracle coverage: did a multi-method candidate pool contain a
  reference-compatible period before arbitration?

Catalog-derived strategies must not be scored against the same catalog
consensus used as their input.  ``evaluate_stored_strategies`` therefore marks
those rows as coverage-only instead of reporting circular accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_HARMONIC_FACTORS: tuple[float, ...] = (
    0.25,
    1.0 / 3.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
)


@dataclass(frozen=True)
class StoredPeriodStrategy:
    """Describe a stored-period selection rule.

    ``period_columns`` are checked from left to right.  This represents
    fallback routing without materializing another wide table.
    """

    name: str
    label: str
    period_columns: tuple[str, ...]
    family: str
    reference_independent: bool
    description: str


DEFAULT_STORED_STRATEGIES: tuple[StoredPeriodStrategy, ...] = (
    StoredPeriodStrategy(
        "catalog_consensus_only",
        "Catalog consensus only",
        ("catalog_reference_period",),
        "catalog",
        False,
        "Use the clean integrated external-catalog consensus and otherwise abstain.",
    ),
    StoredPeriodStrategy(
        "lsp_top_period",
        "Stored Lomb-Scargle",
        ("lsp_period",),
        "single_method",
        True,
        "Stored short-window Lomb-Scargle peak.",
    ),
    StoredPeriodStrategy(
        "pdm_top_period",
        "Stored Plavchan PDM",
        ("pdm_period",),
        "single_method",
        True,
        "Stored Plavchan-style PDM period.",
    ),
    StoredPeriodStrategy(
        "ce_top_period",
        "Stored conditional entropy",
        ("ce_period",),
        "single_method",
        True,
        "Stored conditional-entropy period.",
    ),
    StoredPeriodStrategy(
        "long_ls_top_period",
        "Stored long Lomb-Scargle",
        ("long_ls_period_days",),
        "single_method",
        True,
        "Stored baseline-adaptive long-period Lomb-Scargle peak.",
    ),
    StoredPeriodStrategy(
        "event_period",
        "Stored event-spacing period",
        ("event_period_days",),
        "single_method",
        True,
        "Period inferred from detected event epochs when available.",
    ),
    StoredPeriodStrategy(
        "production_consensus",
        "Stored dip-focused production consensus",
        ("periodicity_period",),
        "production_ensemble",
        True,
        "The selected internal long-period/event consensus from the July 1 periodicity backfill.",
    ),
    StoredPeriodStrategy(
        "phase_selected",
        "Stored phase-folding period",
        ("phase_period_days",),
        "production_ensemble",
        False,
        "The period currently selected for phase folding; it can inherit an external catalog period.",
    ),
    StoredPeriodStrategy(
        "catalog_then_production",
        "Catalog first, then production consensus",
        ("catalog_reference_period", "periodicity_period"),
        "hybrid_route",
        False,
        "Use catalog consensus when present and compute/fall back internally otherwise.",
    ),
)


_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "lc_path",
    "n_points",
    "stats_photometry_median_mag",
    "baseline_mag",
    "period_sources",
    "period_n_sources",
    "period_consensus_days",
    "period_consensus_agree",
    "period_conflict_flag",
    "period_consensus_support",
    "period_primary_source",
    "period_source_periods",
    "gaia_eb_period",
    "vsx_period",
    "asassn_var_period",
    "ztf_var_period",
    "period_ogle_days",
    "lsp_period",
    "pdm_period",
    "ce_period",
    "long_ls_period_days",
    "event_period_days",
    "event_period_is_high_confidence",
    "periodicity_period",
    "periodicity_method",
    "phase_period_days",
    "phase_source",
    "period_confidence",
    "period_method",
    "period_baseline_cycles",
    "dip_run_epochs_json",
)


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def load_review_period_snapshot(
    review_db: str | Path,
    *,
    columns: Sequence[str] = _SNAPSHOT_COLUMNS,
) -> pd.DataFrame:
    """Read the period-relevant candidate fields from ``review.db`` read-only."""

    path = Path(review_db).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"{path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        table_info = connection.execute("PRAGMA table_info(candidates)").fetchall()
        available = {str(row[1]) for row in table_info}
        requested = tuple(dict.fromkeys(str(name) for name in columns))
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"Candidate table is missing requested fields: {missing}")
        query = "SELECT " + ", ".join(map(_quote_identifier, requested)) + " FROM candidates"
        frame = pd.read_sql_query(query, connection)
    frame["median_mag"] = pd.to_numeric(
        frame["stats_photometry_median_mag"], errors="coerce"
    ).fillna(pd.to_numeric(frame["baseline_mag"], errors="coerce"))
    return frame


def add_catalog_reference(
    frame: pd.DataFrame,
    *,
    minimum_sources: int = 1,
    require_agreement: bool = True,
    reject_conflicts: bool = True,
) -> pd.DataFrame:
    """Add a clean external-catalog reference without filling missing periods."""

    required = {
        "period_consensus_days",
        "period_n_sources",
        "period_consensus_agree",
        "period_conflict_flag",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing catalog reference fields: {missing}")
    result = frame.copy()
    period = pd.to_numeric(result["period_consensus_days"], errors="coerce")
    source_count = pd.to_numeric(result["period_n_sources"], errors="coerce").fillna(0)
    valid = period.notna() & period.gt(0) & source_count.ge(int(minimum_sources))
    if require_agreement:
        valid &= result["period_consensus_agree"].fillna(False).astype(bool)
    if reject_conflicts:
        valid &= ~result["period_conflict_flag"].fillna(False).astype(bool)
    result["catalog_reference_period"] = period.where(valid)
    result["catalog_reference_available"] = valid
    result["catalog_reference_tier"] = np.select(
        [
            valid & source_count.ge(2),
            valid,
        ],
        [
            "clean_multi_catalog",
            "clean_single_catalog",
        ],
        default="none",
    )
    return result


def resolve_strategy_period(
    frame: pd.DataFrame,
    strategy: StoredPeriodStrategy,
) -> pd.Series:
    """Resolve a strategy's first valid positive period for every row."""

    missing = sorted(set(strategy.period_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Strategy {strategy.name!r} is missing columns: {missing}")
    resolved = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in strategy.period_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.where(np.isfinite(values) & values.gt(0))
        resolved = resolved.fillna(values)
    return resolved


def period_match_arrays(
    candidate_period: Sequence[float],
    reference_period: Sequence[float],
    *,
    tolerance: float = 0.05,
    harmonic_factors: Sequence[float] = DEFAULT_HARMONIC_FACTORS,
) -> dict[str, np.ndarray]:
    """Return exact and harmonic-family relative-period agreement arrays."""

    if not np.isfinite(tolerance) or float(tolerance) < 0:
        raise ValueError("tolerance must be finite and non-negative")
    factors = np.asarray(tuple(harmonic_factors), dtype=float)
    if factors.size == 0 or np.any(~np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError("harmonic_factors must contain finite positive values")
    if not np.any(np.isclose(factors, 1.0, rtol=0.0, atol=1.0e-12)):
        factors = np.append(factors, 1.0)
    factors = np.unique(factors)

    candidate = np.asarray(candidate_period, dtype=float)
    reference = np.asarray(reference_period, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("candidate_period and reference_period must have equal shapes")
    valid = (
        np.isfinite(candidate)
        & (candidate > 0)
        & np.isfinite(reference)
        & (reference > 0)
    )
    errors = np.full((candidate.size, factors.size), np.inf, dtype=float)
    if valid.any():
        expected = reference[valid, None] * factors[None, :]
        errors[valid] = np.abs(candidate[valid, None] - expected) / expected
    nearest_index = np.argmin(errors, axis=1)
    nearest_error = errors[np.arange(candidate.size), nearest_index]
    exact_error = np.full(candidate.size, np.inf, dtype=float)
    exact_error[valid] = np.abs(candidate[valid] - reference[valid]) / reference[valid]
    return {
        "valid": valid,
        "is_exact": valid & (exact_error <= float(tolerance)),
        "is_harmonic_family": valid & (nearest_error <= float(tolerance)),
        "relative_error": np.where(valid, exact_error, np.nan),
        "nearest_harmonic_factor": np.where(valid, factors[nearest_index], np.nan),
        "nearest_harmonic_error": np.where(valid, nearest_error, np.nan),
    }


def evaluate_stored_strategies(
    frame: pd.DataFrame,
    *,
    strategies: Sequence[StoredPeriodStrategy] = DEFAULT_STORED_STRATEGIES,
    reference_col: str = "catalog_reference_period",
    tolerance: float = 0.05,
    harmonic_factors: Sequence[float] = DEFAULT_HARMONIC_FACTORS,
) -> pd.DataFrame:
    """Summarize stored strategies without circular catalog scoring."""

    if reference_col not in frame:
        raise ValueError(f"Missing reference column {reference_col!r}")
    reference = pd.to_numeric(frame[reference_col], errors="coerce")
    reference_mask = np.isfinite(reference) & reference.gt(0)
    reference_total = int(reference_mask.sum())
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        estimate = resolve_strategy_period(frame, strategy)
        available = np.isfinite(estimate) & estimate.gt(0)
        comparable = reference_mask & available
        record: dict[str, object] = {
            "strategy": strategy.name,
            "label": strategy.label,
            "family": strategy.family,
            "description": strategy.description,
            "reference_independent": bool(strategy.reference_independent),
            "n_all": int(len(frame)),
            "n_period_available": int(available.sum()),
            "coverage_all": float(available.mean()) if len(frame) else np.nan,
            "n_reference": reference_total,
            "n_available_on_reference": int(comparable.sum()),
            "coverage_on_reference": (
                float(comparable.sum() / reference_total)
                if reference_total
                else np.nan
            ),
        }
        if strategy.reference_independent and comparable.any():
            matched = period_match_arrays(
                estimate.to_numpy(dtype=float),
                reference.to_numpy(dtype=float),
                tolerance=tolerance,
                harmonic_factors=harmonic_factors,
            )
            exact = matched["is_exact"] & comparable.to_numpy()
            family = matched["is_harmonic_family"] & comparable.to_numpy()
            record.update(
                {
                    "exact_agreement_conditional": float(exact.sum() / comparable.sum()),
                    "family_agreement_conditional": float(
                        family.sum() / comparable.sum()
                    ),
                    "exact_yield_on_reference": (
                        float(exact.sum() / reference_total)
                        if reference_total
                        else np.nan
                    ),
                    "family_yield_on_reference": (
                        float(family.sum() / reference_total)
                        if reference_total
                        else np.nan
                    ),
                    "median_relative_error": float(
                        np.nanmedian(matched["relative_error"][comparable.to_numpy()])
                    ),
                }
            )
        else:
            record.update(
                {
                    "exact_agreement_conditional": np.nan,
                    "family_agreement_conditional": np.nan,
                    "exact_yield_on_reference": np.nan,
                    "family_yield_on_reference": np.nan,
                    "median_relative_error": np.nan,
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _stable_token(value: object, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest()


def stratified_runtime_sample(
    frame: pd.DataFrame,
    n: int,
    *,
    seed: int = 20260730,
    magnitude_min: float | None = 12.0,
    magnitude_max: float | None = 15.0,
    id_col: str = "candidate_id",
    magnitude_col: str = "median_mag",
    points_col: str = "n_points",
    path_col: str = "lc_path",
) -> pd.DataFrame:
    """Select a deterministic, weighted runtime sample across key strata."""

    if int(n) < 1:
        raise ValueError("n must be positive")
    required = {id_col, magnitude_col, points_col, path_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sample fields: {missing}")
    pool = frame.copy()
    magnitude = pd.to_numeric(pool[magnitude_col], errors="coerce")
    if magnitude_min is not None:
        pool = pool.loc[magnitude.ge(float(magnitude_min))].copy()
        magnitude = pd.to_numeric(pool[magnitude_col], errors="coerce")
    if magnitude_max is not None:
        pool = pool.loc[magnitude.le(float(magnitude_max))].copy()
    pool = pool.loc[pool[path_col].notna() & pool[id_col].notna()].copy()
    if pool.empty:
        raise ValueError("No rows remain after runtime-sample filtering")

    pool["_reference_stratum"] = np.where(
        pool.get(
            "catalog_reference_available",
            pd.Series(False, index=pool.index),
        ).fillna(False),
        "catalog_reference",
        "no_catalog_reference",
    )
    mag = pd.to_numeric(pool[magnitude_col], errors="coerce")
    pool["_magnitude_stratum"] = pd.cut(
        mag,
        bins=[-np.inf, 13.0, 14.0, np.inf],
        labels=["bright", "middle", "faint"],
    ).astype("string").fillna("missing")
    points = pd.to_numeric(pool[points_col], errors="coerce")
    try:
        point_bins = pd.qcut(
            points.rank(method="first"),
            q=min(3, int(points.notna().sum())),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        point_bins = pd.Series(0, index=pool.index, dtype=float)
    pool["_points_stratum"] = (
        pd.to_numeric(point_bins, errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    pool["sample_stratum"] = (
        pool[["_reference_stratum", "_magnitude_stratum", "_points_stratum"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    pool["_stable_token"] = pool[id_col].map(lambda value: _stable_token(value, seed))
    ordered_groups = {
        str(name): group.sort_values("_stable_token", kind="stable").index.tolist()
        for name, group in pool.groupby("sample_stratum", sort=True)
    }
    selected_indices: list[object] = []
    while len(selected_indices) < min(int(n), len(pool)):
        added = False
        for name in sorted(ordered_groups):
            values = ordered_groups[name]
            if not values:
                continue
            selected_indices.append(values.pop(0))
            added = True
            if len(selected_indices) >= min(int(n), len(pool)):
                break
        if not added:
            break
    result = pool.loc[selected_indices].copy()
    population_counts = pool["sample_stratum"].value_counts()
    sample_counts = result["sample_stratum"].value_counts()
    result["sample_stratum_population_n"] = result["sample_stratum"].map(
        population_counts
    ).astype(int)
    result["sample_stratum_n"] = result["sample_stratum"].map(sample_counts).astype(int)
    result["sample_weight"] = (
        result["sample_stratum_population_n"] / result["sample_stratum_n"]
    )
    return result.drop(
        columns=[
            "_reference_stratum",
            "_magnitude_stratum",
            "_points_stratum",
            "_stable_token",
        ]
    ).reset_index(drop=True)


def project_runtime_days(
    *,
    n_sources: int,
    seconds_per_source: float,
    workers_per_machine: int,
    machines: int = 1,
    parallel_efficiency: float = 1.0,
) -> float:
    """Project elapsed days under an explicit outer-loop scaling assumption."""

    values = (
        int(n_sources),
        float(seconds_per_source),
        int(workers_per_machine),
        int(machines),
        float(parallel_efficiency),
    )
    if values[0] < 0 or values[1] < 0:
        raise ValueError("n_sources and seconds_per_source must be non-negative")
    if values[2] < 1 or values[3] < 1:
        raise ValueError("workers_per_machine and machines must be positive")
    if not 0 < values[4] <= 1:
        raise ValueError("parallel_efficiency must lie in (0, 1]")
    effective_workers = values[2] * values[3] * values[4]
    return float(values[0] * values[1] / effective_workers / 86_400.0)


def mark_pareto_frontier(
    summary: pd.DataFrame,
    *,
    cost_col: str,
    accuracy_col: str,
    coverage_col: str | None = None,
    minimum_coverage: float = 0.0,
) -> pd.DataFrame:
    """Mark strategies not dominated in cost and accuracy."""

    required = {cost_col, accuracy_col}
    if coverage_col is not None:
        required.add(coverage_col)
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing Pareto fields: {missing}")
    result = summary.copy()
    cost = pd.to_numeric(result[cost_col], errors="coerce").to_numpy(dtype=float)
    accuracy = pd.to_numeric(result[accuracy_col], errors="coerce").to_numpy(
        dtype=float
    )
    eligible = np.isfinite(cost) & np.isfinite(accuracy)
    if coverage_col is not None:
        coverage = pd.to_numeric(result[coverage_col], errors="coerce").to_numpy(
            dtype=float
        )
        eligible &= np.isfinite(coverage) & (coverage >= float(minimum_coverage))
    pareto = np.zeros(len(result), dtype=bool)
    for index in np.flatnonzero(eligible):
        dominates = (
            eligible
            & (cost <= cost[index])
            & (accuracy >= accuracy[index])
            & ((cost < cost[index]) | (accuracy > accuracy[index]))
        )
        pareto[index] = not bool(dominates.any())
    result["is_pareto"] = pareto
    return result


__all__ = [
    "DEFAULT_HARMONIC_FACTORS",
    "DEFAULT_STORED_STRATEGIES",
    "StoredPeriodStrategy",
    "add_catalog_reference",
    "evaluate_stored_strategies",
    "load_review_period_snapshot",
    "mark_pareto_frontier",
    "period_match_arrays",
    "project_runtime_days",
    "resolve_strategy_period",
    "stratified_runtime_sample",
]
