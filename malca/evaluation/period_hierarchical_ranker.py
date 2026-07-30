"""Evaluation-only hierarchical arbitration for period candidate families.

This module is deliberately not imported by MALCA's production period
pipeline.  It implements the second-generation injection experiment as three
separate learned decisions:

1. rank harmonic families with graded relevance;
2. resolve the fundamental within the selected family; and
3. accept or reject the selected solution with an independent null model.

Every fitted model learns on the training partition.  Probability calibrators
learn only on the separate calibration partition.  Threshold tuning remains a
downstream operation on a third, disjoint threshold-development partition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import pickle

import numpy as np
import pandas as pd

from malca.evaluation.period_candidate_ranker import (
    CandidateRankerArtifact,
    RankerConfig,
    StatusThresholds,
    assign_solution_status,
    fit_candidate_ranker,
    score_candidate_ranker,
    validate_feature_allowlist,
)


HIERARCHICAL_ARTIFACT_SCHEMA_VERSION = "period-hierarchical-arbitrator-v2"
DEFAULT_RESOLVER_COMPARISON_FEATURES: tuple[str, ...] = (
    "proposal_normalized_score",
    "proposal_contributing_method_count",
    "proposal_independent_method_family_count",
    "baseline_cycles",
    "ls_power",
    "ls_local_best_power",
    "pdm_theta",
    "ce_entropy",
    "bls_power",
    "bls_exact_depth_snr",
    "bls_exact_log_likelihood",
    "fourier_1_power",
    "fourier_1_bic",
    "fourier_2_power",
    "fourier_2_bic",
    "fourier_3_power",
    "fourier_3_bic",
    "lafler_kinman_t_phase",
    "supersmoother_explained_fraction",
    "template_q",
    "template_bin_coverage",
    "template_scatter_ratio",
    "odd_even_depth_abs_difference",
    "odd_even_depth_ratio",
    "odd_even_shape_rms",
    "event_evidence_enabled",
    "event_score",
    "event_phase_concentration",
    "event_inlier_fraction",
    "event_rms_oc_days",
    "alias_rayleigh_distance",
    "alias_within_resolution",
    "stability_valid_segment_fraction",
    "stability_fourier_power_median",
    "stability_fourier_power_std",
    "stability_amplitude_cv",
    "stability_phase_resultant",
    "null_periodic_vs_linear_delta_bic",
    "null_periodic_vs_quadratic_delta_bic",
    "null_periodic_vs_annual_delta_bic",
    "null_time_order_von_neumann_ratio",
)


@dataclass(frozen=True)
class HierarchicalArbitratorConfig:
    """Configuration for family ranking, fundamental resolution, and rejection."""

    group_col: str = "base_view_id"
    split_group_col: str = "base_trial_id"
    candidate_id_col: str = "candidate_id"
    period_col: str = "period_days"
    harmonic_factors: tuple[float, ...] = (
        0.25,
        1.0 / 3.0,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
    )
    family_relative_tolerance: float = 0.05
    family_rayleigh_tolerance: float = 2.0
    resolver_relative_tolerance: float = 0.08
    resolver_comparison_features: tuple[str, ...] = (
        DEFAULT_RESOLVER_COMPARISON_FEATURES
    )
    aggregation_statistics: tuple[str, ...] = ("max", "min", "mean", "std")
    calibration_margin_weight: float = 0.5
    random_state: int = 20260726
    n_jobs: int = 1
    n_estimators: int = 400
    learning_rate: float = 0.025
    num_leaves: int = 31
    min_child_samples: int = 12
    calibration_method: str = "isotonic"
    hard_negative_weights: tuple[tuple[str, float], ...] = (
        ("seasonal_trend", 3.0),
        ("red_noise", 3.0),
        ("aperiodic_dips", 3.0),
        ("isolated_dip", 3.0),
        ("evolving_trend", 2.0),
        ("step_change", 2.0),
    )
    secure_acceptance_threshold: float = 0.80
    secure_exact_threshold: float = 0.80
    secure_family_threshold: float = 0.85
    tentative_acceptance_threshold: float = 0.35

    def __post_init__(self) -> None:
        if any(
            (not np.isfinite(value)) or float(value) <= 0
            for value in self.harmonic_factors
        ):
            raise ValueError("harmonic_factors must be finite and positive")
        if (
            not np.isfinite(self.family_relative_tolerance)
            or not 0 < float(self.family_relative_tolerance) < 1
        ):
            raise ValueError("family_relative_tolerance must lie in (0, 1)")
        if (
            not np.isfinite(self.family_rayleigh_tolerance)
            or float(self.family_rayleigh_tolerance) <= 0
        ):
            raise ValueError("family_rayleigh_tolerance must be positive")
        if (
            not np.isfinite(self.resolver_relative_tolerance)
            or not 0 < float(self.resolver_relative_tolerance) < 1
        ):
            raise ValueError("resolver_relative_tolerance must lie in (0, 1)")
        unknown_statistics = set(self.aggregation_statistics) - {
            "max",
            "min",
            "mean",
            "std",
        }
        if unknown_statistics:
            raise ValueError(
                f"Unknown aggregation statistics: {sorted(unknown_statistics)}"
            )
        for morphology, weight in self.hard_negative_weights:
            if not str(morphology):
                raise ValueError("Hard-negative morphology names cannot be empty")
            if not np.isfinite(weight) or float(weight) <= 0:
                raise ValueError("Hard-negative weights must be positive")

    def status_thresholds(self) -> StatusThresholds:
        return StatusThresholds(
            secure_acceptance_threshold=self.secure_acceptance_threshold,
            secure_exact_threshold=self.secure_exact_threshold,
            secure_family_threshold=self.secure_family_threshold,
            tentative_acceptance_threshold=self.tentative_acceptance_threshold,
        )


@dataclass
class HierarchicalPeriodArbitratorArtifact:
    """Serializable fitted state for the three-stage arbitrator."""

    family_ranker: CandidateRankerArtifact
    resolver_ranker: CandidateRankerArtifact
    acceptance_model: CandidateRankerArtifact
    config: HierarchicalArbitratorConfig
    candidate_feature_columns: tuple[str, ...]
    family_feature_columns: tuple[str, ...]
    resolver_feature_columns: tuple[str, ...]
    acceptance_feature_columns: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalScoreResult:
    """Auditable intermediate tables and final one-row-per-view solutions."""

    clustered_candidates: pd.DataFrame
    family_scores: pd.DataFrame
    resolver_scores: pd.DataFrame
    acceptance_scores: pd.DataFrame
    solutions: pd.DataFrame


def _baseline_for_group(
    frame: pd.DataFrame,
    *,
    baseline_col: str,
    default_baseline_days: float,
) -> float:
    if baseline_col in frame:
        values = pd.to_numeric(frame[baseline_col], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            return float(finite.iloc[0])
    return float(default_baseline_days)


def _same_harmonic_family(
    left_period: float,
    right_period: float,
    *,
    baseline_days: float,
    factors: Sequence[float],
    relative_tolerance: float,
    rayleigh_tolerance: float,
) -> bool:
    if (
        not np.isfinite(left_period)
        or not np.isfinite(right_period)
        or left_period <= 0
        or right_period <= 0
    ):
        return False
    for factor in factors:
        expected = float(right_period) * float(factor)
        relative_error = abs(float(left_period) - expected) / expected
        if relative_error > float(relative_tolerance):
            continue
        frequency_error = abs(1.0 / float(left_period) - 1.0 / expected)
        if frequency_error * float(baseline_days) <= float(rayleigh_tolerance):
            return True
    return False


def cluster_harmonic_families(
    candidates: pd.DataFrame,
    *,
    config: HierarchicalArbitratorConfig | None = None,
    baseline_col: str = "baseline_days",
    default_baseline_days: float = 4000.0,
) -> pd.DataFrame:
    """Assign leakage-free connected components under harmonic equivalence."""

    cfg = config or HierarchicalArbitratorConfig()
    required = {cfg.group_col, cfg.candidate_id_col, cfg.period_col}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing clustering columns: {missing}")
    parts: list[pd.DataFrame] = []
    for group_value, group in candidates.groupby(
        cfg.group_col,
        sort=False,
        dropna=False,
    ):
        work = group.copy().reset_index(drop=False).rename(
            columns={"index": "_original_index"}
        )
        periods = pd.to_numeric(work[cfg.period_col], errors="coerce").to_numpy(
            dtype=float
        )
        baseline = _baseline_for_group(
            work,
            baseline_col=baseline_col,
            default_baseline_days=default_baseline_days,
        )
        parent = np.arange(len(work), dtype=int)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return int(index)

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for left in range(len(work) - 1):
            for right in range(left + 1, len(work)):
                if _same_harmonic_family(
                    periods[left],
                    periods[right],
                    baseline_days=baseline,
                    factors=cfg.harmonic_factors,
                    relative_tolerance=cfg.family_relative_tolerance,
                    rayleigh_tolerance=cfg.family_rayleigh_tolerance,
                ):
                    union(left, right)

        components: dict[int, list[int]] = {}
        for index in range(len(work)):
            components.setdefault(find(index), []).append(index)
        ordered_components = sorted(
            components.values(),
            key=lambda indices: (
                float(np.nanmin(periods[indices])),
                min(str(work.loc[index, cfg.candidate_id_col]) for index in indices),
            ),
        )
        family_by_index: dict[int, str] = {}
        family_rank_by_index: dict[int, int] = {}
        for family_rank, indices in enumerate(ordered_components, start=1):
            family_id = f"{group_value}::hf-{family_rank:04d}"
            for index in indices:
                family_by_index[index] = family_id
                family_rank_by_index[index] = int(family_rank)
        work["harmonic_family_id"] = [
            family_by_index[index] for index in range(len(work))
        ]
        work["harmonic_family_rank"] = [
            family_rank_by_index[index] for index in range(len(work))
        ]
        work["harmonic_family_size"] = work.groupby(
            "harmonic_family_id", sort=False
        )[cfg.candidate_id_col].transform("size")
        work["family_instance_id"] = work["harmonic_family_id"]
        parts.append(
            work.set_index("_original_index", drop=True)
        )
    if not parts:
        result = candidates.copy()
        result["harmonic_family_id"] = pd.Series(dtype="string")
        result["harmonic_family_rank"] = pd.Series(dtype="int64")
        result["harmonic_family_size"] = pd.Series(dtype="int64")
        result["family_instance_id"] = pd.Series(dtype="string")
        return result
    result = pd.concat(parts).sort_index(kind="mergesort")
    result.index.name = candidates.index.name
    return result


def _method_tokens(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(sorted({str(item) for item in value if str(item)}))
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("[") or text.startswith("("):
        try:
            payload = json.loads(text.replace("(", "[").replace(")", "]"))
        except Exception:
            payload = None
        if isinstance(payload, list):
            return tuple(sorted({str(item) for item in payload if str(item)}))
    separator = "|" if "|" in text else ","
    return tuple(
        sorted(
            {
                token.strip().strip("'\"")
                for token in text.split(separator)
                if token.strip().strip("'\"")
            }
        )
    )


def _numeric_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    group_col: str,
) -> tuple[str, ...]:
    allowed = validate_feature_allowlist(
        frame,
        tuple(feature_columns),
        group_col=group_col,
    )
    numeric: list[str] = []
    for name in allowed:
        converted = pd.to_numeric(frame[name], errors="coerce")
        if converted.notna().any():
            numeric.append(name)
    return tuple(numeric)


def _family_source_features(
    candidate_feature_columns: Sequence[str],
    *,
    config: HierarchicalArbitratorConfig,
) -> tuple[str, ...]:
    """Select compact family evidence instead of aggregating every column."""

    explicit = set(config.resolver_comparison_features) | {
        "period_days",
        "harmonic_factor",
        "proposal_harmonic_factor",
        "proposal_rank",
        "proposal_prominence",
        "proposal_normalized_score",
        "proposal_contributing_method_count",
        "proposal_independent_method_family_count",
        "proposal_merged_count",
        "proposal_is_harmonic_expansion",
    }
    return tuple(
        name
        for name in candidate_feature_columns
        if name in explicit
        or name.startswith("proposal_contributes_")
        or name.startswith("proposal_method_is_")
    )


def build_harmonic_family_table(
    clustered_candidates: pd.DataFrame,
    *,
    candidate_feature_columns: Sequence[str],
    config: HierarchicalArbitratorConfig | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Aggregate candidate evidence into one row per harmonic family."""

    cfg = config or HierarchicalArbitratorConfig()
    if "harmonic_family_id" not in clustered_candidates:
        raise ValueError("Candidates must be clustered before family aggregation")
    numeric_features = _numeric_feature_columns(
        clustered_candidates,
        candidate_feature_columns,
        group_col=cfg.group_col,
    )
    rows: list[dict[str, Any]] = []
    generated_features: list[str] = [
        "support_family_candidate_count",
        "support_family_method_count",
        "support_family_log_period_span",
        "support_family_harmonic_factor_span",
    ]
    for feature in numeric_features:
        for statistic in cfg.aggregation_statistics:
            generated_features.append(
                f"support_aggregate_{feature}_{statistic}"
            )

    for family_id, family in clustered_candidates.groupby(
        "harmonic_family_id",
        sort=False,
        dropna=False,
    ):
        period = pd.to_numeric(family[cfg.period_col], errors="coerce")
        finite_period = period[np.isfinite(period) & (period > 0)]
        methods: set[str] = set()
        method_column = (
            "contributing_methods"
            if "contributing_methods" in family
            else (
                "proposal_method"
                if "proposal_method" in family
                else None
            )
        )
        if method_column is not None:
            for value in family[method_column]:
                methods.update(_method_tokens(value))
        record: dict[str, Any] = {
            cfg.group_col: family[cfg.group_col].iloc[0],
            cfg.split_group_col: (
                family[cfg.split_group_col].iloc[0]
                if cfg.split_group_col in family
                else family[cfg.group_col].iloc[0]
            ),
            "harmonic_family_id": family_id,
            "candidate_id": family_id,
            "period_days": (
                float(np.exp(np.mean(np.log(finite_period))))
                if not finite_period.empty
                else np.nan
            ),
            "support_family_candidate_count": int(len(family)),
            "support_family_method_count": int(len(methods)),
            "support_family_log_period_span": (
                float(
                    np.log(float(finite_period.max()) / float(finite_period.min()))
                )
                if len(finite_period) >= 2
                else 0.0
            ),
            "support_family_harmonic_factor_span": (
                float(
                    pd.to_numeric(
                        family.get(
                            "harmonic_factor",
                            pd.Series(np.ones(len(family))),
                        ),
                        errors="coerce",
                    ).max()
                    - pd.to_numeric(
                        family.get(
                            "harmonic_factor",
                            pd.Series(np.ones(len(family))),
                        ),
                        errors="coerce",
                    ).min()
                )
                if len(family)
                else 0.0
            ),
            "family_candidate_ids": tuple(
                family[cfg.candidate_id_col].astype(str)
            ),
        }
        for context_name in (
            "split",
            "event_mode",
            "detected_event_backend",
            "morphology",
            "nuisance_mode",
            "truth_is_periodic",
            "true_period_days",
        ):
            if context_name in family:
                record[context_name] = family[context_name].iloc[0]
        for feature in numeric_features:
            values = pd.to_numeric(family[feature], errors="coerce").to_numpy(
                dtype=float
            )
            finite = values[np.isfinite(values)]
            for statistic in cfg.aggregation_statistics:
                key = f"support_aggregate_{feature}_{statistic}"
                if not finite.size:
                    record[key] = np.nan
                elif statistic == "max":
                    record[key] = float(np.max(finite))
                elif statistic == "min":
                    record[key] = float(np.min(finite))
                elif statistic == "mean":
                    record[key] = float(np.mean(finite))
                else:
                    record[key] = (
                        float(np.std(finite, ddof=1))
                        if finite.size >= 2
                        else 0.0
                    )
        if "rank_relevance" in family:
            relevance = pd.to_numeric(
                family["rank_relevance"], errors="coerce"
            )
            record["family_relevance"] = int(relevance.max())
        if "is_exact" in family:
            record["family_contains_exact"] = bool(
                family["is_exact"].fillna(False).astype(bool).any()
            )
        if "is_harmonic_family" in family:
            record["family_is_true"] = bool(
                family["is_harmonic_family"].fillna(False).astype(bool).any()
            )
        rows.append(record)
    return pd.DataFrame(rows), tuple(generated_features)


def add_harmonic_resolver_features(
    clustered_candidates: pd.DataFrame,
    *,
    candidate_feature_columns: Sequence[str],
    config: HierarchicalArbitratorConfig | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add pairwise P/2 and 2P evidence within each harmonic family."""

    cfg = config or HierarchicalArbitratorConfig()
    if "family_instance_id" not in clustered_candidates:
        raise ValueError("Candidates must be clustered before resolver features")
    numeric = _numeric_feature_columns(
        clustered_candidates,
        candidate_feature_columns,
        group_col=cfg.group_col,
    )
    comparison = tuple(
        name
        for name in cfg.resolver_comparison_features
        if name in numeric
    )
    work = clustered_candidates.copy()
    generated = [
        "support_resolver_family_size",
        "support_resolver_log_period_from_family_median",
        "support_resolver_has_half",
        "support_resolver_has_double",
        "support_resolver_half_relative_error",
        "support_resolver_double_relative_error",
    ]
    for feature in comparison:
        generated.extend(
            (
                f"support_resolver_{feature}_minus_half",
                f"support_resolver_{feature}_minus_double",
            )
        )
    for name in generated:
        work[name] = np.nan

    for _, family in work.groupby(
        "family_instance_id",
        sort=False,
        dropna=False,
    ):
        periods = pd.to_numeric(
            family[cfg.period_col], errors="coerce"
        ).to_numpy(dtype=float)
        indices = family.index.to_numpy()
        finite_period = periods[np.isfinite(periods) & (periods > 0)]
        median_period = (
            float(np.exp(np.mean(np.log(finite_period))))
            if finite_period.size
            else np.nan
        )
        for local_index, row_index in enumerate(indices):
            period = periods[local_index]
            work.at[row_index, "support_resolver_family_size"] = int(
                len(family)
            )
            work.at[
                row_index,
                "support_resolver_log_period_from_family_median",
            ] = (
                float(np.log(period / median_period))
                if np.isfinite(period)
                and period > 0
                and np.isfinite(median_period)
                and median_period > 0
                else np.nan
            )
            sibling_by_name: dict[str, int | None] = {}
            for sibling_name, ratio in (("half", 0.5), ("double", 2.0)):
                target = float(period) * ratio
                relative = np.abs(periods - target) / target
                relative[local_index] = np.inf
                sibling = (
                    int(np.nanargmin(relative))
                    if np.isfinite(relative).any()
                    else None
                )
                if (
                    sibling is None
                    or not np.isfinite(relative[sibling])
                    or relative[sibling] > cfg.resolver_relative_tolerance
                ):
                    sibling = None
                sibling_by_name[sibling_name] = sibling
                work.at[
                    row_index, f"support_resolver_has_{sibling_name}"
                ] = int(sibling is not None)
                work.at[
                    row_index,
                    f"support_resolver_{sibling_name}_relative_error",
                ] = (
                    float(relative[sibling]) if sibling is not None else np.nan
                )
            for feature in comparison:
                value = pd.to_numeric(
                    pd.Series([family.loc[row_index, feature]]),
                    errors="coerce",
                ).iloc[0]
                for sibling_name, sibling in sibling_by_name.items():
                    sibling_value = (
                        pd.to_numeric(
                            pd.Series(
                                [family.iloc[int(sibling)][feature]]
                            ),
                            errors="coerce",
                        ).iloc[0]
                        if sibling is not None
                        else np.nan
                    )
                    work.at[
                        row_index,
                        f"support_resolver_{feature}_minus_{sibling_name}",
                    ] = (
                        float(value - sibling_value)
                        if np.isfinite(value) and np.isfinite(sibling_value)
                        else np.nan
                    )
    return work, tuple((*numeric, *generated))


def _top_with_diagnostics(
    scored: pd.DataFrame,
    *,
    group_col: str,
    rank_col: str = "candidate_rank",
    score_col: str = "ranker_score_raw",
) -> pd.DataFrame:
    """Select rank one and attach margin and normalized score entropy."""

    rows: list[pd.Series] = []
    for _, group in scored.groupby(group_col, sort=False, dropna=False):
        ordered = group.sort_values(
            [rank_col, score_col],
            ascending=[True, False],
            kind="mergesort",
        )
        top = ordered.iloc[0].copy()
        raw = pd.to_numeric(ordered[score_col], errors="coerce").to_numpy(
            dtype=float
        )
        finite = raw[np.isfinite(raw)]
        if finite.size >= 2:
            sorted_raw = np.sort(finite)[::-1]
            margin = float(sorted_raw[0] - sorted_raw[1])
        else:
            margin = np.nan
        if finite.size >= 2:
            shifted = finite - float(np.max(finite))
            probability = np.exp(np.clip(shifted, -50.0, 0.0))
            probability /= float(np.sum(probability))
            entropy = float(
                -np.sum(probability * np.log(np.clip(probability, 1.0e-15, 1.0)))
                / np.log(finite.size)
            )
        else:
            entropy = 0.0
        top["ranker_score_margin"] = margin
        top["ranker_score_entropy"] = entropy
        top["ranker_group_size"] = int(len(group))
        rows.append(top)
    return pd.DataFrame(rows).reset_index(drop=True)


def _selection_signal(
    selected: pd.DataFrame,
    *,
    margin_weight: float,
) -> np.ndarray:
    top = pd.to_numeric(
        selected["ranker_score_raw"], errors="coerce"
    ).to_numpy(dtype=float)
    margin = pd.to_numeric(
        selected["ranker_score_margin"], errors="coerce"
    ).to_numpy(dtype=float)
    margin = np.where(np.isfinite(margin), margin, 0.0)
    return top + float(margin_weight) * margin


def _selected_family_and_resolver(
    candidates: pd.DataFrame,
    family_ranker: CandidateRankerArtifact,
    resolver_ranker: CandidateRankerArtifact,
    *,
    candidate_feature_columns: Sequence[str],
    config: HierarchicalArbitratorConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    clustered = cluster_harmonic_families(candidates, config=config)
    family_source_features = _family_source_features(
        candidate_feature_columns,
        config=config,
    )
    family_table, _ = build_harmonic_family_table(
        clustered,
        candidate_feature_columns=family_source_features,
        config=config,
    )
    family_scores = score_candidate_ranker(family_table, family_ranker)
    selected_family = _top_with_diagnostics(
        family_scores,
        group_col=config.group_col,
    )
    family_map = selected_family.set_index(config.group_col)[
        "harmonic_family_id"
    ]
    resolver_table, _ = add_harmonic_resolver_features(
        clustered,
        candidate_feature_columns=candidate_feature_columns,
        config=config,
    )
    selected_family_id = resolver_table[config.group_col].map(family_map)
    resolver_table = resolver_table.loc[
        resolver_table["harmonic_family_id"].eq(selected_family_id)
    ].copy()
    resolver_scores = score_candidate_ranker(
        resolver_table,
        resolver_ranker,
    )
    selected_candidate = _top_with_diagnostics(
        resolver_scores,
        group_col=config.group_col,
    )
    return clustered, family_scores, selected_family, resolver_scores, selected_candidate


def _acceptance_feature_table(
    candidates: pd.DataFrame,
    selected_family: pd.DataFrame,
    family_scores: pd.DataFrame,
    selected_candidate: pd.DataFrame,
    resolver_scores: pd.DataFrame,
    *,
    candidate_feature_columns: Sequence[str],
    config: HierarchicalArbitratorConfig,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build one truth-free feature row per light-curve view."""

    numeric = _numeric_feature_columns(
        candidates,
        candidate_feature_columns,
        group_col=config.group_col,
    )
    candidate_count = candidates.groupby(
        config.group_col, sort=False
    ).size()
    family_count = family_scores.groupby(
        config.group_col, sort=False
    ).size()
    rows: list[dict[str, Any]] = []
    feature_names = [
        "support_acceptance_candidate_count",
        "support_acceptance_family_count",
        "support_acceptance_family_top_score",
        "support_acceptance_family_margin",
        "support_acceptance_family_entropy",
        "support_acceptance_resolver_top_score",
        "support_acceptance_resolver_margin",
        "support_acceptance_resolver_entropy",
        "support_acceptance_selected_family_size",
        "support_acceptance_selected_family_method_count",
    ]
    feature_names.extend(
        f"support_selected_{feature}" for feature in numeric
    )
    family_by_group = selected_family.set_index(config.group_col)
    selected_by_group = selected_candidate.set_index(config.group_col)
    for group_value in selected_by_group.index:
        candidate = selected_by_group.loc[group_value]
        family = family_by_group.loc[group_value]
        record: dict[str, Any] = {
            config.group_col: group_value,
            config.split_group_col: candidate.get(
                config.split_group_col,
                group_value,
            ),
            "candidate_id": f"{group_value}::acceptance",
            "period_days": candidate.get(config.period_col, np.nan),
            "support_acceptance_candidate_count": int(
                candidate_count.get(group_value, 0)
            ),
            "support_acceptance_family_count": int(
                family_count.get(group_value, 0)
            ),
            "support_acceptance_family_top_score": family.get(
                "ranker_score_raw", np.nan
            ),
            "support_acceptance_family_margin": family.get(
                "ranker_score_margin", np.nan
            ),
            "support_acceptance_family_entropy": family.get(
                "ranker_score_entropy", np.nan
            ),
            "support_acceptance_resolver_top_score": candidate.get(
                "ranker_score_raw", np.nan
            ),
            "support_acceptance_resolver_margin": candidate.get(
                "ranker_score_margin", np.nan
            ),
            "support_acceptance_resolver_entropy": candidate.get(
                "ranker_score_entropy", np.nan
            ),
            "support_acceptance_selected_family_size": family.get(
                "support_family_candidate_count", np.nan
            ),
            "support_acceptance_selected_family_method_count": family.get(
                "support_family_method_count", np.nan
            ),
        }
        for context_name in (
            "split",
            "event_mode",
            "detected_event_backend",
            "morphology",
            "nuisance_mode",
            "truth_is_periodic",
            "true_period_days",
            "is_exact",
            "is_harmonic_family",
            "is_harmonic_only",
        ):
            if context_name in candidate:
                record[context_name] = candidate[context_name]
        for feature in numeric:
            record[f"support_selected_{feature}"] = candidate.get(
                feature, np.nan
            )
        rows.append(record)
    table = pd.DataFrame(rows)
    hard_negative_weights = dict(config.hard_negative_weights)
    if not table.empty:
        table["training_sample_weight"] = [
            (
                float(hard_negative_weights.get(str(morphology), 1.0))
                if not bool(periodic)
                else 1.0
            )
            for morphology, periodic in zip(
                table.get(
                    "morphology",
                    pd.Series([""] * len(table)),
                ),
                table.get(
                    "truth_is_periodic",
                    pd.Series([False] * len(table)),
                ),
            )
        ]
    return table, tuple(feature_names)


def _ranker_common_kwargs(
    config: HierarchicalArbitratorConfig,
) -> dict[str, Any]:
    return {
        "random_state": int(config.random_state),
        "n_estimators": int(config.n_estimators),
        "learning_rate": float(config.learning_rate),
        "num_leaves": int(config.num_leaves),
        "min_child_samples": int(config.min_child_samples),
        "calibration_method": str(config.calibration_method),
        "calibration_margin_weight": float(config.calibration_margin_weight),
        "n_jobs": int(config.n_jobs),
        "secure_acceptance_threshold": float(
            config.secure_acceptance_threshold
        ),
        "secure_exact_threshold": float(config.secure_exact_threshold),
        "secure_family_threshold": float(config.secure_family_threshold),
        "tentative_acceptance_threshold": float(
            config.tentative_acceptance_threshold
        ),
    }


def fit_hierarchical_period_arbitrator(
    train_candidates: pd.DataFrame,
    calibration_candidates: pd.DataFrame,
    *,
    candidate_feature_columns: Sequence[str],
    config: HierarchicalArbitratorConfig | None = None,
) -> HierarchicalPeriodArbitratorArtifact:
    """Fit the family ranker, resolver, and independent acceptance model."""

    cfg = config or HierarchicalArbitratorConfig()
    if train_candidates.empty or calibration_candidates.empty:
        raise ValueError("Training and calibration candidates must be non-empty")
    candidate_features = _numeric_feature_columns(
        train_candidates,
        candidate_feature_columns,
        group_col=cfg.group_col,
    )
    validate_feature_allowlist(
        calibration_candidates,
        candidate_features,
        group_col=cfg.group_col,
    )
    train_clustered = cluster_harmonic_families(
        train_candidates,
        config=cfg,
    )
    calibration_clustered = cluster_harmonic_families(
        calibration_candidates,
        config=cfg,
    )
    train_family, family_features = build_harmonic_family_table(
        train_clustered,
        candidate_feature_columns=_family_source_features(
            candidate_features,
            config=cfg,
        ),
        config=cfg,
    )
    calibration_family, _ = build_harmonic_family_table(
        calibration_clustered,
        candidate_feature_columns=_family_source_features(
            candidate_features,
            config=cfg,
        ),
        config=cfg,
    )
    family_config = RankerConfig(
        feature_columns=family_features,
        group_col=cfg.group_col,
        split_group_col=cfg.split_group_col,
        candidate_id_col="harmonic_family_id",
        period_col="period_days",
        target_col="family_relevance",
        exact_target_col="family_is_true",
        family_target_col="family_is_true",
        periodic_target_col="truth_is_periodic",
        model_kind="ranker",
        **_ranker_common_kwargs(cfg),
    )
    family_ranker = fit_candidate_ranker(
        train_family,
        calibration_family,
        config=family_config,
    )

    train_resolver, resolver_features = add_harmonic_resolver_features(
        train_clustered,
        candidate_feature_columns=candidate_features,
        config=cfg,
    )
    calibration_resolver, _ = add_harmonic_resolver_features(
        calibration_clustered,
        candidate_feature_columns=candidate_features,
        config=cfg,
    )
    train_true_family_ids = set(
        train_family.loc[
            train_family["family_is_true"].astype(bool),
            "harmonic_family_id",
        ]
    )
    calibration_true_family_ids = set(
        calibration_family.loc[
            calibration_family["family_is_true"].astype(bool),
            "harmonic_family_id",
        ]
    )
    train_resolver = train_resolver.loc[
        train_resolver["harmonic_family_id"].isin(train_true_family_ids)
    ].copy()
    calibration_resolver = calibration_resolver.loc[
        calibration_resolver["harmonic_family_id"].isin(
            calibration_true_family_ids
        )
    ].copy()
    if train_resolver.empty or calibration_resolver.empty:
        raise ValueError("No truth-matched harmonic families exist for resolver fit")
    resolver_config = RankerConfig(
        feature_columns=resolver_features,
        group_col="family_instance_id",
        split_group_col=cfg.split_group_col,
        candidate_id_col=cfg.candidate_id_col,
        period_col=cfg.period_col,
        target_col="rank_relevance",
        exact_target_col="is_exact",
        family_target_col="is_harmonic_family",
        periodic_target_col="truth_is_periodic",
        model_kind="ranker",
        **_ranker_common_kwargs(cfg),
    )
    resolver_ranker = fit_candidate_ranker(
        train_resolver,
        calibration_resolver,
        config=resolver_config,
    )

    (
        _,
        train_family_scores,
        train_selected_family,
        train_resolver_scores,
        train_selected_candidate,
    ) = _selected_family_and_resolver(
        train_candidates,
        family_ranker,
        resolver_ranker,
        candidate_feature_columns=candidate_features,
        config=cfg,
    )
    train_acceptance, acceptance_features = _acceptance_feature_table(
        train_candidates,
        train_selected_family,
        train_family_scores,
        train_selected_candidate,
        train_resolver_scores,
        candidate_feature_columns=candidate_features,
        config=cfg,
    )
    (
        _,
        calibration_family_scores,
        calibration_selected_family,
        calibration_resolver_scores,
        calibration_selected_candidate,
    ) = _selected_family_and_resolver(
        calibration_candidates,
        family_ranker,
        resolver_ranker,
        candidate_feature_columns=candidate_features,
        config=cfg,
    )
    calibration_acceptance, _ = _acceptance_feature_table(
        calibration_candidates,
        calibration_selected_family,
        calibration_family_scores,
        calibration_selected_candidate,
        calibration_resolver_scores,
        candidate_feature_columns=candidate_features,
        config=cfg,
    )
    acceptance_config = RankerConfig(
        feature_columns=acceptance_features,
        group_col=cfg.group_col,
        split_group_col=cfg.split_group_col,
        candidate_id_col="candidate_id",
        period_col="period_days",
        target_col="truth_is_periodic",
        exact_target_col="truth_is_periodic",
        family_target_col="truth_is_periodic",
        periodic_target_col="truth_is_periodic",
        model_kind="classifier",
        equal_group_weight=False,
        sample_weight_col="training_sample_weight",
        **_ranker_common_kwargs(cfg),
    )
    acceptance_model = fit_candidate_ranker(
        train_acceptance,
        calibration_acceptance,
        config=acceptance_config,
    )
    metadata_core = {
        "schema_version": HIERARCHICAL_ARTIFACT_SCHEMA_VERSION,
        "config": asdict(cfg),
        "candidate_feature_columns": list(candidate_features),
        "family_feature_columns": list(family_features),
        "resolver_feature_columns": list(resolver_features),
        "acceptance_feature_columns": list(acceptance_features),
        "family_ranker_fingerprint": family_ranker.metadata.get(
            "artifact_fingerprint"
        ),
        "resolver_ranker_fingerprint": resolver_ranker.metadata.get(
            "artifact_fingerprint"
        ),
        "acceptance_model_fingerprint": acceptance_model.metadata.get(
            "artifact_fingerprint"
        ),
        "calibration_partition_role": "calibration_only",
        "threshold_partition_role": "not_consumed_by_fit",
        "hard_negative_weights": dict(cfg.hard_negative_weights),
    }
    metadata = {
        **metadata_core,
        "artifact_fingerprint": sha256(
            json.dumps(
                metadata_core,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }
    return HierarchicalPeriodArbitratorArtifact(
        family_ranker=family_ranker,
        resolver_ranker=resolver_ranker,
        acceptance_model=acceptance_model,
        config=cfg,
        candidate_feature_columns=candidate_features,
        family_feature_columns=family_features,
        resolver_feature_columns=resolver_features,
        acceptance_feature_columns=acceptance_features,
        metadata=metadata,
    )


def score_hierarchical_period_arbitrator(
    candidates: pd.DataFrame,
    artifact: HierarchicalPeriodArbitratorArtifact,
    *,
    status_thresholds: StatusThresholds | Mapping[str, float] | None = None,
    expected_trials: pd.DataFrame | None = None,
) -> HierarchicalScoreResult:
    """Apply all three stages and return auditable intermediate predictions."""

    cfg = artifact.config
    (
        clustered,
        family_scores,
        selected_family,
        resolver_scores,
        selected_candidate,
    ) = _selected_family_and_resolver(
        candidates,
        artifact.family_ranker,
        artifact.resolver_ranker,
        candidate_feature_columns=artifact.candidate_feature_columns,
        config=cfg,
    )
    acceptance_table, _ = _acceptance_feature_table(
        candidates,
        selected_family,
        family_scores,
        selected_candidate,
        resolver_scores,
        candidate_feature_columns=artifact.candidate_feature_columns,
        config=cfg,
    )
    acceptance_scores = score_candidate_ranker(
        acceptance_table,
        artifact.acceptance_model,
    )
    acceptance_by_group = acceptance_scores.set_index(cfg.group_col)
    family_by_group = selected_family.set_index(cfg.group_col)
    solution = selected_candidate.copy()
    family_signal = _selection_signal(
        selected_family,
        margin_weight=cfg.calibration_margin_weight,
    )
    resolver_signal = _selection_signal(
        selected_candidate,
        margin_weight=cfg.calibration_margin_weight,
    )
    acceptance_signal = pd.to_numeric(
        acceptance_by_group.loc[
            solution[cfg.group_col],
            "ranker_score_raw",
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    family_probability = artifact.family_ranker.family_calibrator.predict(
        family_signal
    )
    exact_probability = artifact.resolver_ranker.exact_calibrator.predict(
        resolver_signal
    )
    acceptance_probability = (
        artifact.acceptance_model.acceptance_calibrator.predict(
            acceptance_signal
        )
    )
    family_probability_by_group = pd.Series(
        family_probability,
        index=selected_family[cfg.group_col].to_numpy(),
    )
    exact_probability_by_group = pd.Series(
        exact_probability,
        index=selected_candidate[cfg.group_col].to_numpy(),
    )
    acceptance_probability_by_group = pd.Series(
        acceptance_probability,
        index=solution[cfg.group_col].to_numpy(),
    )
    solution = solution.rename(
        columns={
            "ranker_score_raw": "resolver_score_raw",
            "ranker_score": "resolver_score",
            "ranker_score_margin": "resolver_score_margin",
            "ranker_score_entropy": "resolver_score_entropy",
        }
    )
    solution["selected_period_days"] = pd.to_numeric(
        solution[cfg.period_col], errors="coerce"
    )
    solution["family_score_raw"] = pd.to_numeric(
        family_by_group.loc[
            solution[cfg.group_col],
            "ranker_score_raw",
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    solution["family_score_margin"] = pd.to_numeric(
        family_by_group.loc[
            solution[cfg.group_col],
            "ranker_score_margin",
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    solution["family_score_entropy"] = pd.to_numeric(
        family_by_group.loc[
            solution[cfg.group_col],
            "ranker_score_entropy",
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    solution["acceptance_score_raw"] = acceptance_signal
    solution["acceptance_probability"] = solution[cfg.group_col].map(
        acceptance_probability_by_group
    )
    solution["family_probability"] = solution[cfg.group_col].map(
        family_probability_by_group
    )
    solution["exact_probability"] = solution[cfg.group_col].map(
        exact_probability_by_group
    )
    solution["harmonic_ambiguity_probability"] = (
        1.0 - solution["exact_probability"]
    )
    if status_thresholds is None:
        thresholds = cfg.status_thresholds()
    elif isinstance(status_thresholds, StatusThresholds):
        thresholds = status_thresholds
    else:
        thresholds = StatusThresholds(**dict(status_thresholds))
    solution["solution_status"] = assign_solution_status(
        solution["acceptance_probability"],
        solution["exact_probability"],
        solution["family_probability"],
        **thresholds.as_dict(),
    )
    solution["no_candidate"] = False

    if expected_trials is not None:
        if cfg.group_col not in expected_trials:
            raise ValueError(
                f"Expected-trial table is missing {cfg.group_col!r}"
            )
        expected = expected_trials.drop_duplicates(cfg.group_col).copy()
        missing = expected.loc[
            ~expected[cfg.group_col].isin(solution[cfg.group_col])
        ].copy()
        if not missing.empty:
            missing["candidate_id"] = pd.NA
            missing["period_days"] = np.nan
            missing["selected_period_days"] = np.nan
            missing["acceptance_probability"] = 0.0
            missing["family_probability"] = 0.0
            missing["exact_probability"] = 0.0
            missing["harmonic_ambiguity_probability"] = 1.0
            missing["solution_status"] = "abstain"
            missing["no_candidate"] = True
            solution = pd.concat((solution, missing), ignore_index=True)
        order = {
            value: index
            for index, value in enumerate(expected[cfg.group_col])
        }
        solution["_expected_order"] = solution[cfg.group_col].map(order)
        solution = solution.sort_values(
            "_expected_order", kind="mergesort"
        ).drop(columns="_expected_order")

    return HierarchicalScoreResult(
        clustered_candidates=clustered,
        family_scores=family_scores,
        resolver_scores=resolver_scores,
        acceptance_scores=acceptance_scores,
        solutions=solution.reset_index(drop=True),
    )


def save_hierarchical_period_arbitrator(
    artifact: HierarchicalPeriodArbitratorArtifact,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        import joblib

        joblib.dump(artifact, temporary)
    except ImportError:  # pragma: no cover
        with temporary.open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_hierarchical_period_arbitrator(
    path: str | Path,
) -> HierarchicalPeriodArbitratorArtifact:
    source = Path(path)
    try:
        import joblib

        artifact = joblib.load(source)
    except ImportError:  # pragma: no cover
        with source.open("rb") as handle:
            artifact = pickle.load(handle)
    if not isinstance(artifact, HierarchicalPeriodArbitratorArtifact):
        raise TypeError(f"{source} does not contain a hierarchical artifact")
    if (
        artifact.metadata.get("schema_version")
        != HIERARCHICAL_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported hierarchical artifact schema: "
            f"{artifact.metadata.get('schema_version')!r}"
        )
    metadata_core = {
        key: value
        for key, value in artifact.metadata.items()
        if key != "artifact_fingerprint"
    }
    expected = sha256(
        json.dumps(
            metadata_core,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if artifact.metadata.get("artifact_fingerprint") != expected:
        raise ValueError("Hierarchical artifact metadata fingerprint mismatch")
    return artifact


__all__ = [
    "DEFAULT_RESOLVER_COMPARISON_FEATURES",
    "HIERARCHICAL_ARTIFACT_SCHEMA_VERSION",
    "HierarchicalArbitratorConfig",
    "HierarchicalPeriodArbitratorArtifact",
    "HierarchicalScoreResult",
    "add_harmonic_resolver_features",
    "build_harmonic_family_table",
    "cluster_harmonic_families",
    "fit_hierarchical_period_arbitrator",
    "load_hierarchical_period_arbitrator",
    "save_hierarchical_period_arbitrator",
    "score_hierarchical_period_arbitrator",
]
