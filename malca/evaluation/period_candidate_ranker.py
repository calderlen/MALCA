"""Evaluation-only ranking and calibration for period candidate banks.

This module deliberately has no dependency on the production period pipeline.
It consumes a *long* candidate table: one row per proposed period and one or
more rows per injected light curve.  The helpers here cover the experimental
workflow used by the period-injection notebook:

1. assign leakage-safe train/validation/test splits at the light-curve level;
2. label candidates against injected truth;
3. rank candidates with either a transparent baseline or LightGBM;
4. calibrate source acceptance, exact-period confidence, and harmonic-family
   confidence on validation winners;
5. select one solution per light curve and summarize recovery.

All model features come from an explicit allowlist.  Truth, injection, split,
oracle, recovery, and model-output columns are rejected even if a caller puts
them in that allowlist.  LightGBM and scikit-learn are imported lazily so that
candidate generation and deterministic evaluation remain usable without the
optional ML environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence
import importlib.metadata
import json
import math
import pickle
import re
import warnings

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = "period-candidate-ranker-v1"
DEFAULT_HARMONIC_FACTORS: tuple[float, ...] = (
    0.25,
    1.0 / 3.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
)

CANDIDATE_PERIOD_ALIASES: tuple[str, ...] = (
    "period_days",
    "candidate_period_days",
    "candidate_period",
    "period",
)
TRUTH_PERIOD_ALIASES: tuple[str, ...] = (
    "true_period_days",
    "truth_period_days",
    "injected_period_days",
    "injected_period",
)
METHOD_COLUMN_ALIASES: tuple[str, ...] = (
    "proposal_method",
    "method",
    "source_method",
    "candidate_method",
    "contributing_methods",
)
CANDIDATE_ID_ALIASES: tuple[str, ...] = (
    "candidate_id",
    "period_candidate_id",
    "candidate_key",
)

TRUTH_OUTPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "truth_is_periodic",
        "truth_exact_relative_error",
        "truth_family_relative_error",
        "truth_exact_rayleigh_error",
        "truth_family_rayleigh_error",
        "truth_matched_harmonic_factor",
        "truth_matched_harmonic_factor_relative",
        "truth_match_criterion",
        "truth_rayleigh_tolerance",
        "truth_require_relative_with_rayleigh",
        "candidate_truth_period_ratio",
        "is_exact",
        "is_harmonic_family",
        "is_harmonic_only",
        "is_exact_relative",
        "is_harmonic_family_relative",
        "is_harmonic_only_relative",
        "is_exact_resolution_consistent",
        "is_harmonic_family_resolution_consistent",
        "is_harmonic_only_resolution_consistent",
        "is_wrong_harmonic",
        "is_wrong_harmonic_relative",
        "nearest_harmonic_factor",
        "relative_period_error",
        "rayleigh_frequency_error",
        "rank_relevance",
    }
)

_FORBIDDEN_FEATURE_EXACT: frozenset[str] = frozenset(
    {
        "split",
        "fold",
        "base_trial_id",
        "trial_id",
        "candidate_id",
        "period_candidate_id",
        "candidate_key",
        "morphology",
        "period_regime",
        "signal_kind",
        "has_signal",
        "is_signal",
        "is_null",
        "nuisance_mode",
        "event_mode",
        "alias_target_days",
        "amplitude_snr",
        "width_fraction",
        "sampling_density",
        "median_error",
        "signal_seed",
        "trial_seed",
        "requested_baseline_days",
        "design_version",
        # Arbitrary coordinates or duplicate absolute refinements are not
        # evidence.  Their dimensionless shifts/scores remain deployable.
        "bls_transit_time",
        "bls_exact_transit_time",
        "ls_local_best_period_days",
        "bls_refined_period_days",
        "event_exact_period_days",
        "event_local_refined_period_days",
        "event_refined_period_days",
        "alias_nearest_period_days",
        "alias_nearest_index",
        "parent_period_days",
        "proposal_parent_period_days",
        "proposal_seed_period_days",
        "is_exact",
        "is_harmonic_family",
        "is_harmonic_only",
        "is_exact_relative",
        "is_harmonic_family_relative",
        "is_harmonic_only_relative",
        "is_exact_resolution_consistent",
        "is_harmonic_family_resolution_consistent",
        "is_harmonic_only_resolution_consistent",
        "is_wrong_harmonic",
        "is_wrong_harmonic_relative",
        "nearest_harmonic_factor",
        "relative_period_error",
        "rayleigh_frequency_error",
        "rank_relevance",
        "ranker_score",
        "ranker_score_raw",
        "candidate_rank",
        "baseline_score",
        "baseline_rank",
        "acceptance_probability",
        "exact_probability",
        "family_probability",
        "solution_status",
        "selected_period_days",
        "training_sample_weight",
    }
)
_FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = (
    "true_",
    "truth_",
    "injected_",
    "oracle_",
    "recovered_",
    "recovery_",
    "selected_is_",
)
_FORBIDDEN_FEATURE_TOKENS = re.compile(
    r"(^|_)(?:true|truth|injected|oracle|recovered|recovery|label|target|"
    r"ground_truth|is_exact|harmonic_family|correct_candidate|match_truth)(?:_|$)"
)

_SAFE_FEATURE_PREFIXES: tuple[str, ...] = (
    "alias_",
    "baseline_",
    "bls_",
    "candidate_period",
    "ce_",
    "cycle_",
    "event_",
    "feature_",
    "fourier_",
    "harmonic_factor",
    "heldout_",
    "lafler_",
    "ls_",
    "method_",
    "multiharmonic_",
    "n_cycles",
    "n_event",
    "n_method",
    "n_points",
    "n_support",
    "odd_even_",
    "pdm_",
    "period_days",
    "phase_",
    "proposal_",
    "sampling_",
    "string_length",
    "stability_",
    "support_",
    "supersmoother_",
    "template_",
    "null_",
)
_SAFE_CATEGORICAL_NAMES: frozenset[str] = frozenset(
    (
        *METHOD_COLUMN_ALIASES,
        "search_band",
        "proposal_search_band",
        "event_view",
        "event_information_arm",
    )
)

_REDUNDANT_AOV_FEATURE_TOKEN = re.compile(r"(^|_)aov(?:_|$)")


def _is_redundant_aov_feature(name: str) -> bool:
    """Identify AoV columns derived from the same multiharmonic Fourier fit."""

    return bool(_REDUNDANT_AOV_FEATURE_TOKEN.search(str(name).strip().casefold()))


@dataclass(frozen=True)
class BaselineFeatureSpec:
    """One transparent component of the deterministic baseline score.

    Values are converted to within-light-curve percentile utilities, so feature
    scales need not be comparable.  Missing values receive ``missing_utility``.
    """

    column: str
    weight: float = 1.0
    higher_is_better: bool = True
    missing_utility: float = 0.0


@dataclass(frozen=True)
class RankerConfig:
    """Configuration for a candidate-level LightGBM model.

    ``feature_columns`` is a strict allowlist.  If it is empty,
    :func:`default_feature_columns` is applied to the training table.  The
    resulting concrete allowlist is stored in the returned artifact.
    """

    feature_columns: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()
    group_col: str = "base_trial_id"
    split_group_col: str = "base_trial_id"
    candidate_id_col: str = "candidate_id"
    period_col: str = "period_days"
    method_col: str = "proposal_method"
    target_col: str = "is_exact"
    exact_target_col: str = "is_exact"
    family_target_col: str = "is_harmonic_family"
    periodic_target_col: str = "truth_is_periodic"
    model_kind: str = "classifier"
    random_state: int = 20260725
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 12
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    class_weight: str | None = "balanced"
    equal_group_weight: bool = True
    sample_weight_col: str | None = None
    n_jobs: int = 1
    early_stopping_rounds: int = 0
    calibration_method: str = "isotonic"
    min_calibration_samples: int = 20
    calibration_margin_weight: float = 0.5
    secure_acceptance_threshold: float = 0.80
    secure_exact_threshold: float = 0.80
    secure_family_threshold: float = 0.85
    tentative_acceptance_threshold: float = 0.35

    def __post_init__(self) -> None:
        kind = str(self.model_kind).lower()
        if kind not in {"classifier", "ranker"}:
            raise ValueError("model_kind must be 'classifier' or 'ranker'")
        object.__setattr__(self, "model_kind", kind)
        calibration = str(self.calibration_method).lower()
        if calibration not in {"isotonic", "platt", "binned", "none"}:
            raise ValueError(
                "calibration_method must be isotonic, platt, binned, or none"
            )
        object.__setattr__(self, "calibration_method", calibration)
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        object.__setattr__(
            self, "categorical_features", tuple(self.categorical_features)
        )
        for name in (
            "secure_acceptance_threshold",
            "secure_exact_threshold",
            "secure_family_threshold",
            "tentative_acceptance_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if (
            self.tentative_acceptance_threshold
            > self.secure_acceptance_threshold
        ):
            raise ValueError(
                "tentative_acceptance_threshold cannot exceed "
                "secure_acceptance_threshold"
            )
        if int(self.early_stopping_rounds) < 0:
            raise ValueError("early_stopping_rounds must be non-negative")
        if self.sample_weight_col is not None and not str(
            self.sample_weight_col
        ).strip():
            raise ValueError("sample_weight_col must be a non-empty name or None")


@dataclass
class CandidateFeatureEncoder:
    """Deterministic numeric/categorical encoding fitted on training rows."""

    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a candidate table without learning new categories."""

        missing = [name for name in self.feature_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Candidate table is missing model features: {missing}")
        encoded: dict[str, pd.Series] = {}
        categorical = set(self.categorical_columns)
        for name in self.feature_columns:
            if name in categorical:
                values = frame[name].astype("string").fillna("<missing>")
                mapping = {
                    value: index for index, value in enumerate(self.categories[name])
                }
                encoded[name] = values.map(mapping).fillna(-1).astype("int32")
            else:
                values = pd.to_numeric(frame[name], errors="coerce").astype(float)
                encoded[name] = values.where(np.isfinite(values), np.nan)
        # Construct once. Repeated frame insertion produces a heavily
        # fragmented 500+ column family matrix and floods long notebook runs
        # with pandas PerformanceWarnings.
        return pd.DataFrame(encoded, index=frame.index)


@dataclass
class BinaryProbabilityCalibrator:
    """Serializable one-dimensional probability calibrator."""

    method: str
    model: Any | None = field(default=None, repr=False)
    constant: float | None = None
    x_points: tuple[float, ...] = ()
    y_points: tuple[float, ...] = ()

    def metadata(self) -> dict[str, Any]:
        """Return deterministic state sufficient to reproduce predictions."""

        payload: dict[str, Any] = {
            "method": self.method,
            "constant": (
                None if self.constant is None else float(self.constant)
            ),
            "x_points": [float(value) for value in self.x_points],
            "y_points": [float(value) for value in self.y_points],
        }
        if self.method == "isotonic" and self.model is not None:
            payload["model_state"] = {
                "x_thresholds": [
                    float(value)
                    for value in np.asarray(
                        getattr(self.model, "X_thresholds_", ()), dtype=float
                    )
                ],
                "y_thresholds": [
                    float(value)
                    for value in np.asarray(
                        getattr(self.model, "y_thresholds_", ()), dtype=float
                    )
                ],
            }
        elif self.method == "platt" and self.model is not None:
            payload["model_state"] = {
                "coef": np.asarray(
                    getattr(self.model, "coef_", ()), dtype=float
                ).tolist(),
                "intercept": np.asarray(
                    getattr(self.model, "intercept_", ()), dtype=float
                ).tolist(),
                "classes": np.asarray(
                    getattr(self.model, "classes_", ())
                ).tolist(),
            }
        return payload

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        """Return clipped probabilities for raw selection scores."""

        values = np.asarray(scores, dtype=float)
        result = np.full(values.shape, np.nan, dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            return result
        if self.method == "constant":
            result[finite] = float(self.constant if self.constant is not None else 0.5)
        elif self.method == "isotonic":
            result[finite] = np.asarray(self.model.predict(values[finite]), dtype=float)
        elif self.method == "platt":
            result[finite] = np.asarray(
                self.model.predict_proba(values[finite].reshape(-1, 1))[:, 1],
                dtype=float,
            )
        elif self.method == "binned":
            result[finite] = np.interp(
                values[finite],
                np.asarray(self.x_points, dtype=float),
                np.asarray(self.y_points, dtype=float),
            )
        elif self.method == "identity":
            result[finite] = values[finite]
        else:  # pragma: no cover - guards corrupted external artifacts
            raise ValueError(f"Unknown calibrator method: {self.method!r}")
        return np.clip(result, 0.0, 1.0)


@dataclass
class CandidateRankerArtifact:
    """Fitted ranker, encoder, validation calibrators, and provenance."""

    model: Any = field(repr=False)
    encoder: CandidateFeatureEncoder
    config: RankerConfig
    feature_columns: tuple[str, ...]
    model_kind: str
    candidate_calibrator: BinaryProbabilityCalibrator
    acceptance_calibrator: BinaryProbabilityCalibrator
    exact_calibrator: BinaryProbabilityCalibrator
    family_calibrator: BinaryProbabilityCalibrator
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StatusThresholds:
    """Frozen thresholds mapping calibrated confidence to reported status."""

    secure_acceptance_threshold: float = 0.80
    secure_exact_threshold: float = 0.80
    secure_family_threshold: float = 0.85
    tentative_acceptance_threshold: float = 0.35

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if self.tentative_acceptance_threshold > self.secure_acceptance_threshold:
            raise ValueError(
                "tentative_acceptance_threshold cannot exceed the secure "
                "acceptance threshold"
            )

    def as_dict(self) -> dict[str, float]:
        """Return a plain mapping suitable for ``select_trial_solutions``."""

        return {
            name: float(value)
            for name, value in asdict(self).items()
        }


@dataclass
class ThresholdTuningResult:
    """Validation-only threshold search result.

    ``thresholds`` is the frozen policy to apply unchanged to a held-out test.
    ``sweep`` contains every evaluated policy and makes constraint selection
    auditable.
    """

    thresholds: StatusThresholds
    sweep: pd.DataFrame
    constraints_satisfied: bool
    metadata: dict[str, Any]


def _resolve_column(
    frame: pd.DataFrame,
    requested: str,
    aliases: Sequence[str],
    *,
    role: str,
    required: bool = True,
) -> str | None:
    candidates = tuple(dict.fromkeys((requested, *aliases)))
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(
            f"Missing {role} column. Tried: {', '.join(repr(name) for name in candidates)}"
        )
    return None


def _safe_bool(values: pd.Series, *, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(default).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.casefold()
    result = numeric.fillna(0).ne(0)
    result.loc[text.isin({"true", "yes", "y", "periodic", "signal"})] = True
    result.loc[text.isin({"false", "no", "n", "null", "control", "none"})] = False
    return result.fillna(default).astype(bool)


def _stable_group_token(value: object, seed: int) -> int:
    payload = f"{int(seed)}\x1f{type(value).__name__}\x1f{value!s}".encode(
        "utf-8", errors="surrogatepass"
    )
    return int.from_bytes(sha256(payload).digest()[:8], byteorder="big", signed=False)


def _split_counts(n_groups: int, fractions: np.ndarray) -> np.ndarray:
    raw = fractions * int(n_groups)
    counts = np.floor(raw).astype(int)
    remainder = int(n_groups) - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    positive = np.flatnonzero(fractions > 0)
    if n_groups >= len(positive):
        for index in positive:
            if counts[index] > 0:
                continue
            donors = [
                donor
                for donor in np.argsort(-counts, kind="stable")
                if counts[donor] > 1
            ]
            if donors:
                counts[donors[0]] -= 1
                counts[index] += 1
    return counts


def assign_grouped_splits(
    frame: pd.DataFrame,
    group_col: str = "base_trial_id",
    seed: int = 20260725,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    split_col: str = "split",
    stratify_col: str | Sequence[str] | None = None,
    split_names: tuple[str, str, str] = ("train", "validation", "test"),
) -> pd.DataFrame:
    """Assign deterministic train/validation/test splits by whole trial.

    Candidate rows belonging to the same ``group_col`` can never cross splits.
    Assignment is stable to input row order.  Optional stratification columns
    must be constant within each group.
    """

    if group_col not in frame.columns:
        raise ValueError(f"Missing grouping column {group_col!r}")
    if frame[group_col].isna().any():
        raise ValueError(f"{group_col!r} contains missing group identifiers")
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=float
    )
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("Split fractions must be finite and non-negative")
    if not np.isclose(float(fractions.sum()), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to one")
    if len(split_names) != 3 or len(set(split_names)) != 3:
        raise ValueError("split_names must contain three distinct names")

    strata = (
        []
        if stratify_col is None
        else [stratify_col]
        if isinstance(stratify_col, str)
        else list(stratify_col)
    )
    missing_strata = [name for name in strata if name not in frame.columns]
    if missing_strata:
        raise ValueError(f"Missing stratification columns: {missing_strata}")

    group_rows: list[dict[str, Any]] = []
    for group_value, subset in frame.groupby(group_col, sort=False, dropna=False):
        record: dict[str, Any] = {group_col: group_value}
        for name in strata:
            unique = subset[name].astype("string").fillna("<missing>").unique()
            if len(unique) != 1:
                raise ValueError(
                    f"Stratification column {name!r} varies within group "
                    f"{group_value!r}"
                )
            record[name] = str(unique[0])
        record["_stable_token"] = _stable_group_token(group_value, seed)
        group_rows.append(record)

    groups = pd.DataFrame(group_rows)
    if groups.empty:
        result = frame.copy()
        result[split_col] = pd.Series(index=result.index, dtype="string")
        return result
    groups["_stratum"] = (
        groups[strata].astype("string").agg("\x1f".join, axis=1)
        if strata
        else "__all__"
    )
    assignment: dict[object, str] = {}
    for _, subset in groups.groupby("_stratum", sort=True, dropna=False):
        ordered = subset.sort_values(
            ["_stable_token", group_col],
            key=lambda series: series.astype(str) if series.name == group_col else series,
            kind="stable",
        )
        counts = _split_counts(len(ordered), fractions)
        start = 0
        for name, count in zip(split_names, counts):
            for group_value in ordered.iloc[start : start + int(count)][group_col]:
                assignment[group_value] = name
            start += int(count)

    result = frame.copy()
    result[split_col] = result[group_col].map(assignment).astype("string")
    if result[split_col].isna().any():  # pragma: no cover - defensive invariant
        raise RuntimeError("Internal error: at least one group was not assigned")
    return result


def split_candidate_frame(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return independent train, validation, and test candidate frames."""

    if split_col not in frame.columns:
        raise ValueError(f"Missing split column {split_col!r}")
    normalized = frame[split_col].astype("string").str.casefold()
    validation_names = {"validation", "val", "valid"}
    train = frame.loc[normalized.eq("train")].copy()
    validation = frame.loc[normalized.isin(validation_names)].copy()
    test = frame.loc[normalized.eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty")
    return train, validation, test


def _period_error(
    candidate_period: np.ndarray,
    target_period: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    if metric == "relative_period":
        return np.abs(candidate_period - target_period) / target_period
    if metric == "relative_frequency":
        candidate_frequency = 1.0 / candidate_period
        target_frequency = 1.0 / target_period
        return np.abs(candidate_frequency - target_frequency) / target_frequency
    raise ValueError("match_metric must be 'relative_period' or 'relative_frequency'")


def label_candidates(
    frame: pd.DataFrame,
    period_col: str = "period_days",
    truth_col: str = "true_period_days",
    *,
    harmonic_factors: Sequence[float] = DEFAULT_HARMONIC_FACTORS,
    tolerance: float = 0.05,
    exact_tolerance: float | None = None,
    match_metric: str = "relative_period",
    truth_periodic_col: str | None = None,
    baseline_days: float | Sequence[float] | str | None = None,
    rayleigh_tolerance: float | None = None,
    require_relative_with_rayleigh: bool = True,
) -> pd.DataFrame:
    """Label exact and harmonic-family recovery against injected periods.

    Harmonic factors are defined as ``candidate_period / true_period``.  Thus a
    factor of ``0.5`` labels a half-period candidate and ``2`` a double-period
    candidate.  Null/control trials should have a non-positive or missing truth
    period, or provide ``truth_periodic_col=False``.

    Relative-period labels are always retained in ``is_exact_relative`` and
    ``is_harmonic_family_relative`` for comparison with legacy benchmarks.  If
    ``rayleigh_tolerance`` is provided, the principal ``is_exact`` and
    ``is_harmonic_family`` labels require
    ``abs(f_candidate - f_target) * baseline_days <= rayleigh_tolerance``.
    By default they also retain the declared relative-period cap, preventing
    the Rayleigh criterion from becoming excessively permissive when only a
    few long-period cycles are observed.
    """

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    exact_tol = tolerance if exact_tolerance is None else float(exact_tolerance)
    if not np.isfinite(exact_tol) or exact_tol < 0:
        raise ValueError("exact_tolerance must be finite and non-negative")
    if rayleigh_tolerance is not None and (
        not np.isfinite(rayleigh_tolerance) or float(rayleigh_tolerance) < 0
    ):
        raise ValueError("rayleigh_tolerance must be None or finite and non-negative")
    factors = np.asarray(tuple(harmonic_factors), dtype=float)
    if (
        factors.size == 0
        or np.any(~np.isfinite(factors))
        or np.any(factors <= 0)
    ):
        raise ValueError("harmonic_factors must contain finite positive values")
    if not np.any(np.isclose(factors, 1.0, rtol=0.0, atol=1e-12)):
        factors = np.append(factors, 1.0)
    factors = np.unique(factors)

    candidate_name = _resolve_column(
        frame,
        period_col,
        CANDIDATE_PERIOD_ALIASES,
        role="candidate period",
    )
    truth_name = _resolve_column(
        frame,
        truth_col,
        TRUTH_PERIOD_ALIASES,
        role="truth period",
    )
    candidate = pd.to_numeric(frame[candidate_name], errors="coerce").to_numpy(
        dtype=float
    )
    truth = pd.to_numeric(frame[truth_name], errors="coerce").to_numpy(dtype=float)
    valid_candidate = np.isfinite(candidate) & (candidate > 0)
    valid_truth = np.isfinite(truth) & (truth > 0)
    if truth_periodic_col is not None:
        if truth_periodic_col not in frame.columns:
            raise ValueError(f"Missing truth periodicity column {truth_periodic_col!r}")
        valid_truth &= _safe_bool(frame[truth_periodic_col]).to_numpy()

    exact_error = np.full(len(frame), np.inf, dtype=float)
    valid = valid_candidate & valid_truth
    if valid.any():
        exact_error[valid] = _period_error(
            candidate[valid], truth[valid], metric=match_metric
        )

    family_errors = np.full((len(frame), len(factors)), np.inf, dtype=float)
    if valid.any():
        for index, factor in enumerate(factors):
            target = truth[valid] * factor
            family_errors[valid, index] = _period_error(
                candidate[valid], target, metric=match_metric
            )
    best_index = np.argmin(family_errors, axis=1)
    family_error = family_errors[np.arange(len(frame)), best_index]
    matched_factor = factors[best_index].astype(float)
    matched_factor[~valid] = np.nan

    is_exact_relative = valid & (exact_error <= exact_tol)
    is_family_relative = valid & (family_error <= float(tolerance))

    exact_rayleigh_error = np.full(len(frame), np.inf, dtype=float)
    family_rayleigh_errors = np.full((len(frame), len(factors)), np.inf, dtype=float)
    baseline = np.full(len(frame), np.nan, dtype=float)
    if baseline_days is not None:
        if isinstance(baseline_days, str):
            if baseline_days not in frame.columns:
                raise ValueError(
                    f"Missing observing-baseline column {baseline_days!r}"
                )
            baseline = pd.to_numeric(
                frame[baseline_days], errors="coerce"
            ).to_numpy(dtype=float)
        elif np.isscalar(baseline_days):
            baseline.fill(float(baseline_days))
        else:
            baseline = np.asarray(baseline_days, dtype=float)
            if baseline.shape != (len(frame),):
                raise ValueError(
                    "baseline_days sequence must have one value per candidate row"
                )
    valid_baseline = valid & np.isfinite(baseline) & (baseline > 0)
    if valid_baseline.any():
        exact_rayleigh_error[valid_baseline] = (
            np.abs(
                1.0 / candidate[valid_baseline]
                - 1.0 / truth[valid_baseline]
            )
            * baseline[valid_baseline]
        )
        for index, factor in enumerate(factors):
            target = truth[valid_baseline] * factor
            family_rayleigh_errors[valid_baseline, index] = (
                np.abs(1.0 / candidate[valid_baseline] - 1.0 / target)
                * baseline[valid_baseline]
            )
    rayleigh_best_index = np.argmin(family_rayleigh_errors, axis=1)
    family_rayleigh_error = family_rayleigh_errors[
        np.arange(len(frame)), rayleigh_best_index
    ]

    if rayleigh_tolerance is None:
        is_exact = is_exact_relative
        is_family = is_family_relative
        is_exact_resolution = np.zeros(len(frame), dtype=bool)
        is_family_resolution = np.zeros(len(frame), dtype=bool)
        match_criterion = match_metric
    else:
        if baseline_days is None:
            raise ValueError(
                "baseline_days is required when rayleigh_tolerance is provided"
            )
        is_exact_resolution = valid_baseline & (
            exact_rayleigh_error <= float(rayleigh_tolerance)
        )
        is_family_resolution = valid_baseline & (
            family_rayleigh_error <= float(rayleigh_tolerance)
        )
        if require_relative_with_rayleigh:
            is_exact = is_exact_resolution & is_exact_relative
            per_factor_match = (
                family_rayleigh_errors <= float(rayleigh_tolerance)
            ) & (family_errors <= float(tolerance))
            is_family = valid_baseline & np.any(per_factor_match, axis=1)
            combined_error = np.where(
                per_factor_match,
                family_rayleigh_errors,
                np.inf,
            )
            combined_best_index = np.argmin(combined_error, axis=1)
            selected_index = np.where(
                is_family, combined_best_index, rayleigh_best_index
            )
            match_criterion = "rayleigh_frequency_and_relative_period"
        else:
            is_exact = is_exact_resolution
            is_family = is_family_resolution
            selected_index = rayleigh_best_index
            match_criterion = "rayleigh_frequency"
        matched_factor = factors[selected_index].astype(float)
        matched_factor[~valid_baseline] = np.nan
    ratio = np.full(len(frame), np.nan, dtype=float)
    ratio[valid] = candidate[valid] / truth[valid]

    result = frame.copy()
    result["truth_is_periodic"] = valid_truth
    result["truth_exact_relative_error"] = np.where(
        np.isfinite(exact_error), exact_error, np.nan
    )
    result["truth_family_relative_error"] = np.where(
        np.isfinite(family_error), family_error, np.nan
    )
    result["truth_exact_rayleigh_error"] = np.where(
        np.isfinite(exact_rayleigh_error), exact_rayleigh_error, np.nan
    )
    result["truth_family_rayleigh_error"] = np.where(
        np.isfinite(family_rayleigh_error), family_rayleigh_error, np.nan
    )
    result["truth_matched_harmonic_factor_relative"] = factors[
        best_index
    ].astype(float)
    result.loc[~valid, "truth_matched_harmonic_factor_relative"] = np.nan
    result["truth_matched_harmonic_factor"] = matched_factor
    result["candidate_truth_period_ratio"] = ratio
    result["truth_match_criterion"] = match_criterion
    result["truth_rayleigh_tolerance"] = (
        np.nan if rayleigh_tolerance is None else float(rayleigh_tolerance)
    )
    result["truth_require_relative_with_rayleigh"] = bool(
        require_relative_with_rayleigh
    )
    result["is_exact_relative"] = is_exact_relative
    result["is_harmonic_family_relative"] = is_family_relative
    result["is_harmonic_only_relative"] = (
        is_family_relative & ~is_exact_relative
    )
    result["is_exact_resolution_consistent"] = is_exact_resolution
    result["is_harmonic_family_resolution_consistent"] = (
        is_family_resolution
    )
    result["is_harmonic_only_resolution_consistent"] = (
        is_family_resolution & ~is_exact_resolution
    )
    result["is_wrong_harmonic_relative"] = result[
        "is_harmonic_only_relative"
    ]
    result["is_exact"] = is_exact
    result["is_harmonic_family"] = is_family
    result["is_harmonic_only"] = is_family & ~is_exact
    result["is_wrong_harmonic"] = result["is_harmonic_only"]
    result["nearest_harmonic_factor"] = matched_factor
    result["relative_period_error"] = np.where(
        np.isfinite(exact_error), exact_error, np.nan
    )
    result["rayleigh_frequency_error"] = np.where(
        np.isfinite(exact_rayleigh_error), exact_rayleigh_error, np.nan
    )
    result["rank_relevance"] = np.where(is_exact, 2, np.where(is_family, 1, 0)).astype(
        "int8"
    )
    return result


def _feature_leakage_reason(name: str) -> str | None:
    normalized = str(name).strip().casefold()
    if normalized in _FORBIDDEN_FEATURE_EXACT:
        return "reserved grouping, truth, split, or model-output column"
    if normalized in TRUTH_OUTPUT_COLUMNS:
        return "truth-derived label"
    if any(normalized.startswith(prefix) for prefix in _FORBIDDEN_FEATURE_PREFIXES):
        return "truth/injection/oracle-derived prefix"
    if _FORBIDDEN_FEATURE_TOKENS.search(normalized):
        return "truth/target token"
    return None


def validate_feature_allowlist(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    group_col: str = "base_trial_id",
) -> tuple[str, ...]:
    """Validate an explicit feature allowlist and reject likely leakage."""

    features = tuple(str(name) for name in feature_columns)
    if not features:
        raise ValueError("At least one feature column is required")
    if len(set(features)) != len(features):
        duplicates = sorted({name for name in features if features.count(name) > 1})
        raise ValueError(f"Duplicate feature columns are not allowed: {duplicates}")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Feature allowlist columns are missing: {missing}")
    violations: dict[str, str] = {}
    for name in features:
        reason = _feature_leakage_reason(name)
        if reason is not None:
            violations[name] = reason
        elif name == group_col:
            violations[name] = "group identifier"
    if violations:
        detail = ", ".join(f"{name!r} ({reason})" for name, reason in violations.items())
        raise ValueError(f"Feature allowlist contains leakage-prone columns: {detail}")
    return features


def default_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Infer a conservative deployable feature allowlist.

    Inference is intentionally based on known method/score prefixes rather than
    accepting every numeric column.  Callers should inspect and persist the
    returned tuple; :func:`fit_candidate_ranker` stores the concrete result.
    """

    selected: list[str] = []
    for raw_name in frame.columns:
        name = str(raw_name)
        normalized = name.casefold()
        if _feature_leakage_reason(name) is not None:
            continue
        if _is_redundant_aov_feature(name):
            continue
        if normalized in _SAFE_CATEGORICAL_NAMES:
            selected.append(name)
            continue
        if normalized.startswith(_SAFE_FEATURE_PREFIXES) and (
            pd.api.types.is_numeric_dtype(frame[name])
            or pd.api.types.is_bool_dtype(frame[name])
        ):
            selected.append(name)
    return tuple(selected)


_DEFAULT_BASELINE_FEATURE_DIRECTIONS: tuple[tuple[str, bool], ...] = (
    # Proposal evidence.  These are dimensionless method-normalized/support
    # quantities; raw method scores and arbitrary candidate coordinates are
    # intentionally excluded.
    ("proposal_normalized_score", True),
    ("proposal_prominence", True),
    ("proposal_independent_method_family_count", True),
    # Fixed-period photometric evidence.
    ("ls_power", True),
    ("ls_local_best_power", True),
    ("pdm_theta", False),
    ("ce_entropy", False),
    ("bls_power", True),
    ("bls_local_best_power", True),
    ("bls_depth_snr", True),
    ("bls_log_likelihood", True),
    ("fourier_1_power", True),
    ("fourier_2_power", True),
    ("fourier_3_power", True),
    ("lafler_kinman_t_phase", False),
    ("lafler_kinman_delta", False),
    ("supersmoother_cv_standardized_mse", False),
    ("supersmoother_explained_fraction", True),
    ("template_q", False),
    ("template_scatter_ratio", False),
    # Cycle-to-cycle and event-timing consistency.
    ("odd_even_depth_abs_difference", False),
    ("odd_even_shape_rms", False),
    ("event_score", True),
    ("event_phase_concentration", True),
    ("event_inlier_fraction", True),
    ("event_median_abs_oc_days", False),
    ("event_rms_oc_days", False),
)


def default_baseline_feature_specs(
    frame: pd.DataFrame,
) -> tuple[BaselineFeatureSpec, ...]:
    """Return a small, explicit, scientifically directed baseline.

    This comparator deliberately does *not* infer direction from column names
    and does not reward coordinates such as candidate period, transit epoch,
    proposal rank, or alias index.  New features must be reviewed and added to
    ``_DEFAULT_BASELINE_FEATURE_DIRECTIONS`` explicitly before they influence
    the published deterministic baseline.
    """

    return tuple(
        BaselineFeatureSpec(
            name,
            weight=1.0,
            higher_is_better=higher_is_better,
            missing_utility=0.0,
        )
        for name, higher_is_better in _DEFAULT_BASELINE_FEATURE_DIRECTIONS
        if name in frame.columns
        and (
            pd.api.types.is_numeric_dtype(frame[name])
            or pd.api.types.is_bool_dtype(frame[name])
        )
    )


def _within_group_utility(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if not finite.any():
        return result
    valid = numeric.loc[finite]
    if valid.nunique(dropna=True) <= 1:
        result.loc[valid.index] = 0.5
        return result
    ranks = valid.rank(method="average", ascending=True)
    utility = (ranks - 1.0) / float(len(valid) - 1)
    if not higher_is_better:
        utility = 1.0 - utility
    result.loc[valid.index] = utility
    return result


def _deterministic_candidate_rank(
    frame: pd.DataFrame,
    *,
    group_col: str,
    score_col: str,
    period_col: str | None,
    candidate_id_col: str | None,
) -> pd.Series:
    work = pd.DataFrame(index=frame.index)
    work["_group"] = frame[group_col].astype("string").fillna("<missing>")
    work["_score"] = pd.to_numeric(frame[score_col], errors="coerce").fillna(-np.inf)
    if period_col is not None and period_col in frame.columns:
        work["_period"] = pd.to_numeric(frame[period_col], errors="coerce").fillna(
            np.inf
        )
    else:
        work["_period"] = np.inf
    if candidate_id_col is not None and candidate_id_col in frame.columns:
        work["_candidate"] = (
            frame[candidate_id_col].astype("string").fillna("<missing>")
        )
    else:
        work["_candidate"] = frame.index.astype(str)
    work["_original_index"] = np.arange(len(work), dtype=int)
    ordered = work.sort_values(
        ["_group", "_score", "_period", "_candidate", "_original_index"],
        ascending=[True, False, True, True, True],
        kind="stable",
    )
    ordered["_rank"] = ordered.groupby("_group", sort=False).cumcount() + 1
    result = pd.Series(ordered["_rank"].to_numpy(), index=ordered.index, dtype="int64")
    return result.reindex(frame.index)


def rank_deterministic_baseline(
    frame: pd.DataFrame,
    feature_specs: Sequence[BaselineFeatureSpec] | None = None,
    *,
    group_col: str = "base_trial_id",
    period_col: str = "period_days",
    candidate_id_col: str = "candidate_id",
    score_col: str = "baseline_score",
    rank_col: str = "baseline_rank",
    include_components: bool = False,
) -> pd.DataFrame:
    """Rank candidates with a documented weighted percentile score."""

    if group_col not in frame.columns:
        raise ValueError(f"Missing grouping column {group_col!r}")
    specs = tuple(feature_specs or default_baseline_feature_specs(frame))
    if not specs:
        raise ValueError("No usable deterministic baseline features were supplied")
    for spec in specs:
        if spec.column not in frame.columns:
            raise ValueError(f"Missing baseline feature {spec.column!r}")
        if not np.isfinite(spec.weight) or spec.weight < 0:
            raise ValueError(f"Invalid non-negative weight for {spec.column!r}")
        if not 0.0 <= float(spec.missing_utility) <= 1.0:
            raise ValueError(
                f"missing_utility for {spec.column!r} must lie in [0, 1]"
            )
    if sum(float(spec.weight) for spec in specs) <= 0:
        raise ValueError("At least one baseline feature must have positive weight")

    result = frame.copy()
    weighted = np.zeros(len(result), dtype=float)
    total_weight = 0.0
    for spec in specs:
        utility = result.groupby(group_col, sort=False, dropna=False)[
            spec.column
        ].transform(
            lambda values: _within_group_utility(
                values, higher_is_better=spec.higher_is_better
            )
        )
        utility = utility.fillna(float(spec.missing_utility)).astype(float)
        weighted += float(spec.weight) * utility.to_numpy()
        total_weight += float(spec.weight)
        if include_components:
            result[f"{score_col}__{spec.column}"] = utility
    result[score_col] = weighted / total_weight
    period_name = _resolve_column(
        result,
        period_col,
        CANDIDATE_PERIOD_ALIASES,
        role="candidate period",
        required=False,
    )
    candidate_name = _resolve_column(
        result,
        candidate_id_col,
        CANDIDATE_ID_ALIASES,
        role="candidate ID",
        required=False,
    )
    result[rank_col] = _deterministic_candidate_rank(
        result,
        group_col=group_col,
        score_col=score_col,
        period_col=period_name,
        candidate_id_col=candidate_name,
    )
    return result


def _fit_feature_encoder(
    frame: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> CandidateFeatureEncoder:
    explicit_categorical = set(categorical_features)
    unknown = explicit_categorical.difference(features)
    if unknown:
        raise ValueError(
            f"categorical_features are not in feature_columns: {sorted(unknown)}"
        )
    categorical: list[str] = []
    categories: dict[str, tuple[str, ...]] = {}
    for name in features:
        series = frame[name]
        is_categorical = (
            name in explicit_categorical
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        )
        if is_categorical:
            categorical.append(name)
            values = series.astype("string").fillna("<missing>")
            categories[name] = tuple(sorted(str(value) for value in values.unique()))
            continue
        # A deployable feature can be structurally unavailable in one training
        # arm (for example event coherence in an event-free ablation).  Keep an
        # explicitly numeric all-NaN column so LightGBM can learn its missing
        # branch and the frozen schema remains usable when that feature appears
        # in another event arm.
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            raise ValueError(
                f"Feature {name!r} is neither categorical nor numerically usable"
            )
    return CandidateFeatureEncoder(
        feature_columns=tuple(features),
        categorical_columns=tuple(categorical),
        categories=categories,
    )


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - depends on active environment
        raise ImportError(
            "LightGBM is required to fit the experimental candidate ranker. "
            "Candidate labeling, deterministic ranking, selection, and metrics "
            "remain available without it."
        ) from exc
    return lgb


def _fit_binned_calibrator(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    max_bins: int = 10,
) -> BinaryProbabilityCalibrator:
    order = np.argsort(scores, kind="stable")
    chunks = np.array_split(order, min(max_bins, max(2, len(order) // 5)))
    x_points: list[float] = []
    y_points: list[float] = []
    global_rate = float(np.mean(y_true))
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        x_points.append(float(np.mean(scores[chunk])))
        positives = float(np.sum(y_true[chunk]))
        y_points.append((positives + 1.0 + global_rate) / (len(chunk) + 3.0))
    unique_x: list[float] = []
    unique_y: list[float] = []
    for x_value, y_value in zip(x_points, y_points):
        if unique_x and np.isclose(x_value, unique_x[-1]):
            unique_y[-1] = float((unique_y[-1] + y_value) / 2.0)
        else:
            unique_x.append(x_value)
            unique_y.append(y_value)
    if len(unique_x) < 2:
        return BinaryProbabilityCalibrator("constant", constant=global_rate)
    monotonic_y = np.maximum.accumulate(np.asarray(unique_y, dtype=float))
    return BinaryProbabilityCalibrator(
        "binned",
        x_points=tuple(unique_x),
        y_points=tuple(float(value) for value in monotonic_y),
    )


def _fit_probability_calibrator(
    y_true: Sequence[object],
    scores: Sequence[float],
    *,
    method: str,
    min_samples: int,
    seed: int,
) -> BinaryProbabilityCalibrator:
    y = pd.Series(y_true).fillna(False).astype(bool).astype(int).to_numpy()
    raw = np.asarray(scores, dtype=float)
    finite = np.isfinite(raw)
    y = y[finite]
    raw = raw[finite]
    if len(y) == 0:
        return BinaryProbabilityCalibrator("constant", constant=0.0)
    rate = float(np.mean(y))
    if len(np.unique(y)) < 2 or len(y) < int(min_samples):
        return BinaryProbabilityCalibrator("constant", constant=rate)
    if method == "none":
        if np.nanmin(raw) >= 0.0 and np.nanmax(raw) <= 1.0:
            return BinaryProbabilityCalibrator("identity")
        return _fit_binned_calibrator(y, raw)
    if method == "binned":
        return _fit_binned_calibrator(y, raw)
    try:
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(raw, y)
            return BinaryProbabilityCalibrator("isotonic", model=model)
        if method == "platt":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                max_iter=2000,
                random_state=int(seed),
                solver="lbfgs",
            )
            model.fit(raw.reshape(-1, 1), y)
            return BinaryProbabilityCalibrator("platt", model=model)
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        warnings.warn(
            f"Falling back to deterministic binned calibration because "
            f"{method!r} calibration is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return _fit_binned_calibrator(y, raw)


def _raw_model_score(
    model: Any,
    matrix: pd.DataFrame,
    *,
    model_kind: str,
) -> np.ndarray:
    if model_kind == "classifier":
        probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Classifier did not return a two-class probability matrix")
        return probabilities[:, 1]
    return np.asarray(model.predict(matrix), dtype=float).reshape(-1)


def _ordered_group_frame(
    frame: pd.DataFrame,
    *,
    group_col: str,
    period_col: str | None,
    candidate_id_col: str | None,
) -> pd.DataFrame:
    work = frame.copy()
    work["_ranker_original_order"] = np.arange(len(work), dtype=int)
    sort_columns = [group_col]
    if period_col is not None and period_col in work.columns:
        sort_columns.append(period_col)
    if candidate_id_col is not None and candidate_id_col in work.columns:
        sort_columns.append(candidate_id_col)
    sort_columns.append("_ranker_original_order")
    return work.sort_values(
        sort_columns,
        key=lambda series: (
            series.astype("string")
            if series.name in {group_col, candidate_id_col}
            else series
        ),
        kind="stable",
    )


def _validate_group_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    group_col: str,
) -> None:
    if group_col not in train.columns or group_col not in validation.columns:
        raise ValueError(f"Both frames require group column {group_col!r}")
    overlap = set(train[group_col].dropna()).intersection(
        set(validation[group_col].dropna())
    )
    if overlap:
        examples = sorted(str(value) for value in overlap)[:5]
        raise ValueError(
            "Training and validation groups overlap; this would leak light curves "
            f"across calibration: {examples}"
        )


def _equal_group_sample_weight(
    frame: pd.DataFrame,
    *,
    group_col: str,
) -> np.ndarray:
    """Give every light-curve group equal total weight and mean weight one."""

    sizes = frame.groupby(group_col, sort=False)[group_col].transform("size")
    n_groups = int(frame[group_col].nunique())
    if n_groups <= 0 or (sizes <= 0).any():
        raise ValueError("Cannot construct group weights for an empty group set")
    scale = float(len(frame)) / float(n_groups)
    return scale / sizes.to_numpy(dtype=float)


def _selection_signal(
    selected: pd.DataFrame,
    *,
    margin_weight: float,
) -> np.ndarray:
    top = pd.to_numeric(selected["ranker_score_raw"], errors="coerce").to_numpy(
        dtype=float
    )
    margin = pd.to_numeric(
        selected["ranker_score_margin"], errors="coerce"
    ).to_numpy(dtype=float)
    margin = np.where(np.isfinite(margin), margin, 0.0)
    return top + float(margin_weight) * margin


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _frame_fingerprint(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_col: str,
) -> str:
    usable = [name for name in dict.fromkeys((group_col, *columns)) if name in frame]
    work = frame[usable].copy()
    for name in work.columns:
        if pd.api.types.is_object_dtype(work[name]) or pd.api.types.is_string_dtype(
            work[name]
        ):
            work[name] = work[name].astype("string").fillna("<missing>")
    work["_fingerprint_row_hash"] = pd.util.hash_pandas_object(
        work, index=False, categorize=True
    ).astype("uint64")
    values = np.sort(work["_fingerprint_row_hash"].to_numpy(dtype="uint64"))
    digest = sha256()
    digest.update(json.dumps(usable, separators=(",", ":")).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _model_fingerprint(model: Any) -> str | None:
    try:
        text = model.booster_.model_to_string()
    except Exception:
        return None
    return sha256(text.encode("utf-8")).hexdigest()


def _artifact_metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in metadata.items()
        if key != "artifact_fingerprint"
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metadata_config(config: RankerConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["feature_columns"] = list(config.feature_columns)
    payload["categorical_features"] = list(config.categorical_features)
    return payload


def fit_candidate_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    config: RankerConfig,
) -> CandidateRankerArtifact:
    """Fit a candidate classifier/ranker and validation-only calibrators.

    The candidate model learns only on ``train``.  Early stopping and all four
    probability calibrators use ``validation``.  The test split must therefore
    be held back and passed only to :func:`score_candidate_ranker`.
    """

    if train.empty or validation.empty:
        raise ValueError("Training and validation candidate tables must be non-empty")
    split_group_name = (
        config.split_group_col
        if config.split_group_col in train.columns
        and config.split_group_col in validation.columns
        else config.group_col
    )
    _validate_group_disjoint(
        train, validation, group_col=split_group_name
    )
    required_targets = {config.target_col, config.exact_target_col}
    missing_targets = [
        name
        for name in required_targets
        if name not in train.columns or name not in validation.columns
    ]
    if missing_targets:
        raise ValueError(f"Missing candidate target columns: {missing_targets}")

    inferred = config.feature_columns or default_feature_columns(train)
    features = validate_feature_allowlist(
        train, inferred, group_col=config.group_col
    )
    validate_feature_allowlist(validation, features, group_col=config.group_col)
    redundant_aov = sorted(name for name in features if _is_redundant_aov_feature(name))
    if redundant_aov:
        raise ValueError(
            "New candidate rankers cannot include redundant AoV features derived "
            f"from multiharmonic Fourier power: {redundant_aov}"
        )
    if config.split_group_col in features:
        raise ValueError(
            f"Split grouping column {config.split_group_col!r} cannot be a model feature"
        )
    if config.sample_weight_col in features:
        raise ValueError(
            f"Sample-weight column {config.sample_weight_col!r} cannot be a model feature"
        )
    if (
        config.sample_weight_col is not None
        and config.sample_weight_col not in train.columns
    ):
        raise ValueError(
            f"Training table is missing sample-weight column "
            f"{config.sample_weight_col!r}"
        )
    encoder = _fit_feature_encoder(
        train, features, config.categorical_features
    )
    period_name = _resolve_column(
        train,
        config.period_col,
        CANDIDATE_PERIOD_ALIASES,
        role="candidate period",
        required=False,
    )
    candidate_name = _resolve_column(
        train,
        config.candidate_id_col,
        CANDIDATE_ID_ALIASES,
        role="candidate ID",
        required=False,
    )
    ordered_train = _ordered_group_frame(
        train,
        group_col=config.group_col,
        period_col=period_name,
        candidate_id_col=candidate_name,
    )
    ordered_validation = _ordered_group_frame(
        validation,
        group_col=config.group_col,
        period_col=period_name,
        candidate_id_col=candidate_name,
    )
    X_train = encoder.transform(ordered_train)
    X_validation = encoder.transform(ordered_validation)
    y_train = pd.to_numeric(
        ordered_train[config.target_col], errors="raise"
    ).astype(int)
    y_validation = pd.to_numeric(
        ordered_validation[config.target_col], errors="raise"
    ).astype(int)
    exact_validation = _safe_bool(
        ordered_validation[config.exact_target_col]
    ).astype(int)
    if config.model_kind == "classifier" and not set(y_train.unique()).issubset({0, 1}):
        raise ValueError(
            "Classifier target must be binary; use model_kind='ranker' for "
            "relevance labels"
        )
    if y_train.nunique() < 2:
        raise ValueError(
            f"Training target {config.target_col!r} must contain both classes"
        )

    lgb = _require_lightgbm()
    common_kwargs = {
        "n_estimators": int(config.n_estimators),
        "learning_rate": float(config.learning_rate),
        "num_leaves": int(config.num_leaves),
        "max_depth": int(config.max_depth),
        "min_child_samples": int(config.min_child_samples),
        "subsample": float(config.subsample),
        "colsample_bytree": float(config.colsample_bytree),
        "reg_alpha": float(config.reg_alpha),
        "reg_lambda": float(config.reg_lambda),
        "random_state": int(config.random_state),
        "n_jobs": int(config.n_jobs),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    categorical = list(encoder.categorical_columns)
    callbacks: list[Any] = []
    if int(config.early_stopping_rounds) > 0:
        callbacks.append(
            lgb.early_stopping(
                int(config.early_stopping_rounds), verbose=False
            )
        )
    if config.model_kind == "classifier":
        model = lgb.LGBMClassifier(
            objective="binary",
            class_weight=config.class_weight,
            **common_kwargs,
        )
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(X_validation, y_validation)],
            "eval_metric": "binary_logloss",
            "callbacks": callbacks,
        }
        train_weight = np.ones(len(ordered_train), dtype=float)
        if config.equal_group_weight:
            train_weight *= _equal_group_sample_weight(
                ordered_train, group_col=config.group_col
            )
            fit_kwargs["eval_sample_weight"] = [
                _equal_group_sample_weight(
                    ordered_validation, group_col=config.group_col
                )
            ]
        if config.sample_weight_col is not None:
            configured_weight = pd.to_numeric(
                ordered_train[config.sample_weight_col],
                errors="coerce",
            ).to_numpy(dtype=float)
            if (
                not np.isfinite(configured_weight).all()
                or np.any(configured_weight <= 0)
            ):
                raise ValueError(
                    f"Sample weights in {config.sample_weight_col!r} must be "
                    "finite and positive"
                )
            train_weight *= configured_weight
        if config.equal_group_weight or config.sample_weight_col is not None:
            fit_kwargs["sample_weight"] = train_weight
    else:
        model = lgb.LGBMRanker(objective="lambdarank", **common_kwargs)
        train_groups = (
            ordered_train.groupby(config.group_col, sort=False).size().astype(int).tolist()
        )
        validation_groups = (
            ordered_validation.groupby(config.group_col, sort=False)
            .size()
            .astype(int)
            .tolist()
        )
        fit_kwargs = {
            "group": train_groups,
            "eval_set": [(X_validation, y_validation)],
            "eval_group": [validation_groups],
            "eval_at": [1, 3, 5],
            "callbacks": callbacks,
        }
    if categorical:
        fit_kwargs["categorical_feature"] = categorical
    model.fit(X_train, y_train, **fit_kwargs)

    validation_raw = _raw_model_score(
        model, X_validation, model_kind=config.model_kind
    )
    candidate_calibrator = _fit_probability_calibrator(
        exact_validation,
        validation_raw,
        method=config.calibration_method,
        min_samples=config.min_calibration_samples,
        seed=config.random_state,
    )
    validation_scored = ordered_validation.drop(
        columns=["_ranker_original_order"], errors="ignore"
    ).copy()
    validation_scored["ranker_score_raw"] = validation_raw
    validation_scored["ranker_score"] = candidate_calibrator.predict(validation_raw)
    validation_scored["candidate_rank"] = _deterministic_candidate_rank(
        validation_scored,
        group_col=config.group_col,
        score_col="ranker_score_raw",
        period_col=period_name,
        candidate_id_col=candidate_name,
    )
    selected_validation = _select_top_rows(
        validation_scored,
        group_col=config.group_col,
        score_col="ranker_score_raw",
        rank_col="candidate_rank",
    )
    signal = _selection_signal(
        selected_validation,
        margin_weight=config.calibration_margin_weight,
    )
    if config.periodic_target_col in selected_validation.columns:
        acceptance_target = _safe_bool(
            selected_validation[config.periodic_target_col]
        )
    else:
        acceptance_target = _safe_bool(
            selected_validation.get(
                config.family_target_col,
                selected_validation[config.target_col],
            )
        )
    exact_target = _safe_bool(selected_validation[config.exact_target_col])
    family_target = _safe_bool(
        selected_validation.get(config.family_target_col, exact_target)
    )
    calibrator_kwargs = {
        "method": config.calibration_method,
        "min_samples": config.min_calibration_samples,
        "seed": config.random_state,
    }
    acceptance_calibrator = _fit_probability_calibrator(
        acceptance_target, signal, **calibrator_kwargs
    )
    exact_calibrator = _fit_probability_calibrator(
        exact_target, signal, **calibrator_kwargs
    )
    family_calibrator = _fit_probability_calibrator(
        family_target, signal, **calibrator_kwargs
    )

    metadata: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": _metadata_config(config),
        "resolved_feature_columns": list(features),
        "categorical_features": list(encoder.categorical_columns),
        "categorical_levels": {
            key: list(value) for key, value in sorted(encoder.categories.items())
        },
        "n_train_rows": int(len(train)),
        "n_train_groups": int(train[config.group_col].nunique()),
        "n_train_split_groups": int(train[split_group_name].nunique()),
        "n_validation_rows": int(len(validation)),
        "n_validation_groups": int(validation[config.group_col].nunique()),
        "n_validation_split_groups": int(
            validation[split_group_name].nunique()
        ),
        "train_fingerprint": _frame_fingerprint(
            train,
            (
                *features,
                config.target_col,
                config.exact_target_col,
                config.family_target_col,
                config.periodic_target_col,
            ),
            group_col=config.group_col,
        ),
        "validation_fingerprint": _frame_fingerprint(
            validation,
            (
                *features,
                config.target_col,
                config.exact_target_col,
                config.family_target_col,
                config.periodic_target_col,
            ),
            group_col=config.group_col,
        ),
        "dependency_versions": {
            "lightgbm": _package_version("lightgbm"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
        },
        "model_fingerprint": _model_fingerprint(model),
        "calibrators": {
            "candidate": candidate_calibrator.metadata(),
            "acceptance": acceptance_calibrator.metadata(),
            "exact": exact_calibrator.metadata(),
            "family": family_calibrator.metadata(),
        },
    }
    metadata["artifact_fingerprint"] = _artifact_metadata_fingerprint(metadata)
    return CandidateRankerArtifact(
        model=model,
        encoder=encoder,
        config=config,
        feature_columns=features,
        model_kind=config.model_kind,
        candidate_calibrator=candidate_calibrator,
        acceptance_calibrator=acceptance_calibrator,
        exact_calibrator=exact_calibrator,
        family_calibrator=family_calibrator,
        metadata=metadata,
    )


def score_candidate_ranker(
    frame: pd.DataFrame,
    artifact: CandidateRankerArtifact,
) -> pd.DataFrame:
    """Score and deterministically rank every candidate within its trial."""

    config = artifact.config
    if config.group_col not in frame.columns:
        raise ValueError(f"Missing grouping column {config.group_col!r}")
    validate_feature_allowlist(
        frame, artifact.feature_columns, group_col=config.group_col
    )
    matrix = artifact.encoder.transform(frame)
    raw = _raw_model_score(
        artifact.model, matrix, model_kind=artifact.model_kind
    )
    result = frame.copy()
    result["ranker_score_raw"] = raw
    result["ranker_score"] = artifact.candidate_calibrator.predict(raw)
    period_name = _resolve_column(
        result,
        config.period_col,
        CANDIDATE_PERIOD_ALIASES,
        role="candidate period",
        required=False,
    )
    candidate_name = _resolve_column(
        result,
        config.candidate_id_col,
        CANDIDATE_ID_ALIASES,
        role="candidate ID",
        required=False,
    )
    result["candidate_rank"] = _deterministic_candidate_rank(
        result,
        group_col=config.group_col,
        score_col="ranker_score_raw",
        period_col=period_name,
        candidate_id_col=candidate_name,
    )
    return result


def _select_top_rows(
    scored: pd.DataFrame,
    *,
    group_col: str,
    score_col: str,
    rank_col: str | None,
) -> pd.DataFrame:
    if group_col not in scored.columns:
        raise ValueError(f"Missing grouping column {group_col!r}")
    if score_col not in scored.columns:
        raise ValueError(f"Missing score column {score_col!r}")
    work = scored.copy()
    if rank_col is None or rank_col not in work.columns:
        work["_temporary_rank"] = work.groupby(group_col, sort=False)[
            score_col
        ].rank(method="first", ascending=False)
        rank_name = "_temporary_rank"
    else:
        rank_name = rank_col
    selected = work.loc[pd.to_numeric(work[rank_name], errors="coerce").eq(1)].copy()
    if selected[group_col].duplicated().any():
        raise ValueError("Rank column selects more than one candidate in a trial")

    sorted_scores = (
        work.groupby(group_col, sort=False)[score_col]
        .apply(
            lambda values: np.sort(
                pd.to_numeric(values, errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )[::-1]
        )
    )
    margins: dict[object, float] = {}
    runner_up: dict[object, float] = {}
    for group_value, values in sorted_scores.items():
        if len(values) >= 2:
            runner_up[group_value] = float(values[1])
            margins[group_value] = float(values[0] - values[1])
        else:
            runner_up[group_value] = np.nan
            margins[group_value] = np.nan
    selected["ranker_score_runner_up"] = selected[group_col].map(runner_up)
    selected["ranker_score_margin"] = selected[group_col].map(margins)
    return selected.drop(columns=["_temporary_rank"], errors="ignore")


def assign_solution_status(
    acceptance_probability: Sequence[float],
    exact_probability: Sequence[float],
    family_probability: Sequence[float],
    *,
    secure_acceptance_threshold: float = 0.80,
    secure_exact_threshold: float = 0.80,
    secure_family_threshold: float = 0.85,
    tentative_acceptance_threshold: float = 0.35,
) -> np.ndarray:
    """Map three independently calibrated confidences to solution states.

    A secure fundamental must also clear the family threshold.  This prevents a
    calibration inconsistency from declaring the more specific event secure
    while its containing harmonic-family event is not secure.
    """

    acceptance = np.asarray(acceptance_probability, dtype=float)
    exact = np.asarray(exact_probability, dtype=float)
    family = np.asarray(family_probability, dtype=float)
    if not (acceptance.shape == exact.shape == family.shape):
        raise ValueError("Acceptance, exact, and family arrays must have equal shape")
    status = np.full(acceptance.shape, "abstain", dtype=object)
    finite = np.isfinite(acceptance) & np.isfinite(exact) & np.isfinite(family)
    tentative = finite & (acceptance >= float(tentative_acceptance_threshold))
    status[tentative] = "tentative"
    family_secure = (
        finite
        & (acceptance >= float(secure_acceptance_threshold))
        & (family >= float(secure_family_threshold))
    )
    status[family_secure] = "secure_family_harmonic_ambiguous"
    exact_secure = (
        family_secure
        & (exact >= float(secure_exact_threshold))
    )
    status[exact_secure] = "secure_fundamental"
    return status


def select_trial_solutions(
    scored: pd.DataFrame,
    artifact: CandidateRankerArtifact | None = None,
    *,
    group_col: str | None = None,
    period_col: str | None = None,
    candidate_id_col: str | None = None,
    score_col: str = "ranker_score_raw",
    rank_col: str = "candidate_rank",
    status_thresholds: (
        Mapping[str, float] | StatusThresholds | ThresholdTuningResult | None
    ) = None,
    expected_trials: pd.DataFrame | Sequence[object] | None = None,
) -> pd.DataFrame:
    """Select one candidate per trial and assign calibrated confidence states.

    Passing ``artifact`` is required for scientifically calibrated confidence.
    Without it, the helper still selects winners and returns ``NaN``
    probabilities with ``solution_status='abstain'``; this is useful for oracle or
    deterministic-baseline diagnostics without pretending scores are calibrated.

    If ``expected_trials`` is supplied, every expected trial is retained in the
    denominator.  Trials with no candidate become explicit ``abstain`` rows
    with ``no_candidate=True`` rather than silently disappearing.
    """

    config = artifact.config if artifact is not None else RankerConfig()
    group_name = group_col or config.group_col
    period_requested = period_col or config.period_col
    candidate_requested = candidate_id_col or config.candidate_id_col
    selected = _select_top_rows(
        scored,
        group_col=group_name,
        score_col=score_col,
        rank_col=rank_col,
    )
    period_name = _resolve_column(
        selected,
        period_requested,
        CANDIDATE_PERIOD_ALIASES,
        role="candidate period",
        required=False,
    )
    candidate_name = _resolve_column(
        selected,
        candidate_requested,
        CANDIDATE_ID_ALIASES,
        role="candidate ID",
        required=False,
    )
    selected["selected_period_days"] = (
        pd.to_numeric(selected[period_name], errors="coerce")
        if period_name is not None
        else np.nan
    )
    selected["selected_candidate_id"] = (
        selected[candidate_name].astype("string")
        if candidate_name is not None
        else selected.index.astype(str)
    )

    if artifact is None:
        selected["acceptance_probability"] = np.nan
        selected["exact_probability"] = np.nan
        selected["family_probability"] = np.nan
        selected["solution_status"] = "abstain"
    else:
        signal = _selection_signal(
            selected,
            margin_weight=config.calibration_margin_weight,
        )
        selected["acceptance_probability"] = artifact.acceptance_calibrator.predict(
            signal
        )
        selected["exact_probability"] = artifact.exact_calibrator.predict(signal)
        selected["family_probability"] = artifact.family_calibrator.predict(signal)
        thresholds = {
            "secure_acceptance_threshold": config.secure_acceptance_threshold,
            "secure_exact_threshold": config.secure_exact_threshold,
            "secure_family_threshold": config.secure_family_threshold,
            "tentative_acceptance_threshold": config.tentative_acceptance_threshold,
        }
        if isinstance(status_thresholds, ThresholdTuningResult):
            supplied_thresholds: Mapping[str, float] | None = (
                status_thresholds.thresholds.as_dict()
            )
        elif isinstance(status_thresholds, StatusThresholds):
            supplied_thresholds = status_thresholds.as_dict()
        else:
            supplied_thresholds = status_thresholds
        if supplied_thresholds:
            unknown = set(supplied_thresholds).difference(thresholds)
            if unknown:
                raise ValueError(f"Unknown status thresholds: {sorted(unknown)}")
            thresholds.update(
                {key: float(value) for key, value in supplied_thresholds.items()}
            )
        selected["solution_status"] = assign_solution_status(
            selected["acceptance_probability"],
            selected["exact_probability"],
            selected["family_probability"],
            **thresholds,
        )

    selected["no_candidate"] = False
    if expected_trials is None:
        return selected.reset_index(drop=True)

    if isinstance(expected_trials, pd.DataFrame):
        if group_name not in expected_trials.columns:
            raise ValueError(
                f"expected_trials table is missing grouping column {group_name!r}"
            )
        expected = expected_trials.copy()
    else:
        expected = pd.DataFrame({group_name: list(expected_trials)})
    if expected[group_name].duplicated().any():
        raise ValueError("expected_trials must contain one row per trial")
    expected["_expected_order"] = np.arange(len(expected), dtype=int)
    unexpected_groups = selected.loc[
        ~selected[group_name].isin(expected[group_name]), group_name
    ].drop_duplicates()
    if not unexpected_groups.empty:
        raise ValueError(
            "Scored candidates contain trials absent from expected_trials: "
            f"{unexpected_groups.astype(str).head(5).tolist()}"
        )

    context = expected.set_index(group_name, drop=False)
    for name in expected.columns:
        if name in {group_name, "_expected_order"}:
            continue
        mapped = selected[group_name].map(context[name])
        if name not in selected.columns:
            selected[name] = mapped
        else:
            selected[name] = selected[name].where(selected[name].notna(), mapped)

    missing = expected.loc[~expected[group_name].isin(selected[group_name])].copy()
    if not missing.empty:
        missing["selected_period_days"] = np.nan
        missing["selected_candidate_id"] = pd.NA
        for name in (
            "ranker_score_raw",
            "ranker_score",
            "candidate_rank",
            "ranker_score_runner_up",
            "ranker_score_margin",
        ):
            missing[name] = np.nan
        missing["acceptance_probability"] = 0.0
        missing["exact_probability"] = 0.0
        missing["family_probability"] = 0.0
        missing["solution_status"] = "abstain"
        missing["no_candidate"] = True
        for name in (
            config.target_col,
            config.exact_target_col,
            config.family_target_col,
            "is_exact_relative",
            "is_harmonic_family_relative",
            "is_harmonic_only",
            "is_harmonic_only_relative",
        ):
            if name not in missing.columns:
                missing[name] = False
        # Columns that are entirely missing in the placeholder block already
        # exist on ``selected``; omitting them here preserves stable dtypes and
        # avoids pandas' deprecated all-NA concat inference.
        missing_for_concat = missing.dropna(axis=1, how="all")
        selected = pd.concat(
            (selected, missing_for_concat), ignore_index=True, sort=False
        )

    order = expected.set_index(group_name)["_expected_order"]
    selected["_expected_order"] = selected[group_name].map(order)
    return (
        selected.sort_values("_expected_order", kind="stable")
        .drop(columns="_expected_order", errors="ignore")
        .reset_index(drop=True)
    )


def _validated_probability_grid(
    values: Sequence[float] | None,
    *,
    default: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    grid = np.asarray(tuple(default if values is None else values), dtype=float)
    if grid.size == 0 or np.any(~np.isfinite(grid)):
        raise ValueError(f"{name} must contain finite thresholds")
    if np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError(f"{name} thresholds must lie in [0, 1]")
    return tuple(float(value) for value in np.unique(grid))


def _wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)
    proportion = float(successes) / float(trials)
    denominator = 1.0 + z * z / float(trials)
    center = (
        proportion + z * z / (2.0 * float(trials))
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / float(trials)
            + z * z / (4.0 * float(trials) ** 2)
        )
        / denominator
    )
    return (
        float(max(0.0, center - half_width)),
        float(min(1.0, center + half_width)),
    )


def sweep_solution_thresholds(
    validation_solutions: pd.DataFrame,
    *,
    secure_acceptance_grid: Sequence[float] | None = None,
    secure_exact_grid: Sequence[float] | None = None,
    secure_family_grid: Sequence[float] | None = None,
    tentative_acceptance_threshold: float = 0.35,
    target_null_accepted_rate: float = 0.05,
    target_harmonic_only_rate: float | None = 0.05,
    minimum_exact_recovery_rate: float | None = None,
    minimum_family_recovery_rate: float | None = None,
    minimum_periodic_coverage: float | None = None,
    use_confidence_bounds_for_constraints: bool = False,
    constraint_confidence_level: float = 0.95,
    exact_weight: float = 1.0,
    family_weight: float = 0.25,
    coverage_weight: float = 0.05,
    group_col: str = "base_trial_id",
    exact_col: str = "is_exact",
    family_col: str = "is_harmonic_family",
    periodic_col: str = "truth_is_periodic",
) -> pd.DataFrame:
    """Evaluate validation-only status policies on a deterministic grid.

    "Accepted" means one of the two secure states; tentative candidates are
    retained for inspection but do not count toward coverage or false-positive
    rate.  ``harmonic_only_rate`` uses all periodic validation trials as its
    denominator, matching the usual unconditional recovery accounting.
    """

    required = {
        group_col,
        exact_col,
        family_col,
        periodic_col,
        "acceptance_probability",
        "exact_probability",
        "family_probability",
    }
    missing = sorted(required.difference(validation_solutions.columns))
    if missing:
        raise ValueError(f"Threshold sweep is missing columns: {missing}")
    if validation_solutions[group_col].duplicated().any():
        raise ValueError(
            "Threshold sweep requires one selected validation row per trial"
        )
    for name, value in (
        ("target_null_accepted_rate", target_null_accepted_rate),
        ("tentative_acceptance_threshold", tentative_acceptance_threshold),
    ):
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if target_harmonic_only_rate is not None and (
        not np.isfinite(target_harmonic_only_rate)
        or not 0.0 <= float(target_harmonic_only_rate) <= 1.0
    ):
        raise ValueError("target_harmonic_only_rate must be None or lie in [0, 1]")
    for name, value in (
        ("minimum_exact_recovery_rate", minimum_exact_recovery_rate),
        ("minimum_family_recovery_rate", minimum_family_recovery_rate),
        ("minimum_periodic_coverage", minimum_periodic_coverage),
    ):
        if value is not None and (
            not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be None or lie in [0, 1]")
    if (
        not np.isfinite(constraint_confidence_level)
        or not 0.0 < float(constraint_confidence_level) < 1.0
    ):
        raise ValueError("constraint_confidence_level must lie strictly in (0, 1)")
    for name, value in (
        ("exact_weight", exact_weight),
        ("family_weight", family_weight),
        ("coverage_weight", coverage_weight),
    ):
        if not np.isfinite(value) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    default_grid = tuple(np.linspace(0.50, 0.95, 10)) + (0.975, 0.99)
    acceptance_grid = _validated_probability_grid(
        secure_acceptance_grid,
        default=default_grid,
        name="secure_acceptance_grid",
    )
    exact_grid = _validated_probability_grid(
        secure_exact_grid,
        default=default_grid,
        name="secure_exact_grid",
    )
    family_grid = _validated_probability_grid(
        secure_family_grid,
        default=default_grid,
        name="secure_family_grid",
    )
    acceptance_probability = pd.to_numeric(
        validation_solutions["acceptance_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    exact_probability = pd.to_numeric(
        validation_solutions["exact_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    family_probability = pd.to_numeric(
        validation_solutions["family_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    exact_truth = _safe_bool(validation_solutions[exact_col]).to_numpy()
    family_truth = _safe_bool(validation_solutions[family_col]).to_numpy()
    periodic_truth = _safe_bool(validation_solutions[periodic_col]).to_numpy()
    n_trials = int(len(validation_solutions))
    n_periodic = int(periodic_truth.sum())
    n_null = int((~periodic_truth).sum())

    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else np.nan

    rows: list[dict[str, Any]] = []
    for acceptance_threshold in acceptance_grid:
        for exact_threshold in exact_grid:
            for family_threshold in family_grid:
                status = assign_solution_status(
                    acceptance_probability,
                    exact_probability,
                    family_probability,
                    secure_acceptance_threshold=acceptance_threshold,
                    secure_exact_threshold=exact_threshold,
                    secure_family_threshold=family_threshold,
                    tentative_acceptance_threshold=tentative_acceptance_threshold,
                )
                accepted = np.isin(
                    status,
                    (
                        "secure_fundamental",
                        "secure_family_harmonic_ambiguous",
                    ),
                )
                periodic_accepted = periodic_truth & accepted
                exact_success = periodic_accepted & exact_truth
                family_success = periodic_accepted & family_truth
                harmonic_only = periodic_accepted & family_truth & ~exact_truth
                catastrophic = periodic_accepted & ~family_truth
                null_accepted = ~periodic_truth & accepted
                null_rate = rate(int(null_accepted.sum()), n_null)
                harmonic_rate = rate(int(harmonic_only.sum()), n_periodic)
                exact_rate = rate(int(exact_success.sum()), n_periodic)
                family_rate = rate(int(family_success.sum()), n_periodic)
                periodic_coverage = rate(int(periodic_accepted.sum()), n_periodic)
                exact_ci = _wilson_interval(
                    int(exact_success.sum()),
                    n_periodic,
                    confidence_level=constraint_confidence_level,
                )
                family_ci = _wilson_interval(
                    int(family_success.sum()),
                    n_periodic,
                    confidence_level=constraint_confidence_level,
                )
                harmonic_ci = _wilson_interval(
                    int(harmonic_only.sum()),
                    n_periodic,
                    confidence_level=constraint_confidence_level,
                )
                coverage_ci = _wilson_interval(
                    int(periodic_accepted.sum()),
                    n_periodic,
                    confidence_level=constraint_confidence_level,
                )
                null_ci = _wilson_interval(
                    int(null_accepted.sum()),
                    n_null,
                    confidence_level=constraint_confidence_level,
                )
                null_constraint_value = (
                    null_ci[1]
                    if use_confidence_bounds_for_constraints
                    else null_rate
                )
                harmonic_constraint_value = (
                    harmonic_ci[1]
                    if use_confidence_bounds_for_constraints
                    else harmonic_rate
                )
                exact_constraint_value = (
                    exact_ci[0]
                    if use_confidence_bounds_for_constraints
                    else exact_rate
                )
                family_constraint_value = (
                    family_ci[0]
                    if use_confidence_bounds_for_constraints
                    else family_rate
                )
                coverage_constraint_value = (
                    coverage_ci[0]
                    if use_confidence_bounds_for_constraints
                    else periodic_coverage
                )
                null_ok = n_null == 0 or null_constraint_value <= float(
                    target_null_accepted_rate
                )
                harmonic_ok = (
                    target_harmonic_only_rate is None
                    or n_periodic == 0
                    or harmonic_constraint_value
                    <= float(target_harmonic_only_rate)
                )
                exact_ok = (
                    minimum_exact_recovery_rate is None
                    or n_periodic == 0
                    or exact_constraint_value
                    >= float(minimum_exact_recovery_rate)
                )
                family_ok = (
                    minimum_family_recovery_rate is None
                    or n_periodic == 0
                    or family_constraint_value
                    >= float(minimum_family_recovery_rate)
                )
                coverage_ok = (
                    minimum_periodic_coverage is None
                    or n_periodic == 0
                    or coverage_constraint_value
                    >= float(minimum_periodic_coverage)
                )
                objective = (
                    float(exact_weight) * (0.0 if np.isnan(exact_rate) else exact_rate)
                    + float(family_weight)
                    * (0.0 if np.isnan(family_rate) else family_rate)
                    + float(coverage_weight)
                    * (
                        0.0
                        if np.isnan(periodic_coverage)
                        else periodic_coverage
                    )
                )
                rows.append(
                    {
                        "secure_acceptance_threshold": acceptance_threshold,
                        "secure_exact_threshold": exact_threshold,
                        "secure_family_threshold": family_threshold,
                        "tentative_acceptance_threshold": float(
                            tentative_acceptance_threshold
                        ),
                        "n_trials": n_trials,
                        "n_periodic": n_periodic,
                        "n_null": n_null,
                        "accepted_n": int(accepted.sum()),
                        "periodic_coverage": periodic_coverage,
                        "periodic_coverage_ci_low": coverage_ci[0],
                        "periodic_coverage_ci_high": coverage_ci[1],
                        "exact_recovery_rate": exact_rate,
                        "exact_recovery_ci_low": exact_ci[0],
                        "exact_recovery_ci_high": exact_ci[1],
                        "family_recovery_rate": family_rate,
                        "family_recovery_ci_low": family_ci[0],
                        "family_recovery_ci_high": family_ci[1],
                        "harmonic_only_rate": harmonic_rate,
                        "harmonic_only_ci_low": harmonic_ci[0],
                        "harmonic_only_ci_high": harmonic_ci[1],
                        "conditional_harmonic_only_rate": rate(
                            int(harmonic_only.sum()), int(periodic_accepted.sum())
                        ),
                        "catastrophic_rate": rate(
                            int(catastrophic.sum()), n_periodic
                        ),
                        "conditional_catastrophic_rate": rate(
                            int(catastrophic.sum()), int(periodic_accepted.sum())
                        ),
                        "null_accepted_rate": null_rate,
                        "null_accepted_ci_low": null_ci[0],
                        "null_accepted_ci_high": null_ci[1],
                        "constraint_confidence_level": float(
                            constraint_confidence_level
                        ),
                        "constraints_use_confidence_bounds": bool(
                            use_confidence_bounds_for_constraints
                        ),
                        "null_constraint_satisfied": bool(null_ok),
                        "harmonic_constraint_satisfied": bool(harmonic_ok),
                        "exact_constraint_satisfied": bool(exact_ok),
                        "family_constraint_satisfied": bool(family_ok),
                        "coverage_constraint_satisfied": bool(coverage_ok),
                        "constraints_satisfied": bool(
                            null_ok
                            and harmonic_ok
                            and exact_ok
                            and family_ok
                            and coverage_ok
                        ),
                        "objective": float(objective),
                    }
                )
    sweep = pd.DataFrame(rows)
    return sweep.sort_values(
        [
            "constraints_satisfied",
            "objective",
            "exact_recovery_rate",
            "family_recovery_rate",
            "periodic_coverage",
            "null_accepted_rate",
            "harmonic_only_rate",
            "secure_acceptance_threshold",
            "secure_exact_threshold",
            "secure_family_threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def tune_solution_thresholds(
    validation_frame: pd.DataFrame,
    artifact: CandidateRankerArtifact | None = None,
    *,
    require_feasible: bool = True,
    group_col: str | None = None,
    **sweep_kwargs: Any,
) -> ThresholdTuningResult:
    """Tune thresholds on validation data and return a frozen test policy.

    ``validation_frame`` may be a scored long candidate table or an already
    selected one-row-per-trial table.  If confidence columns are absent,
    ``artifact`` is required and is used to score/select validation candidates.
    This helper must never be called on the frozen test split.
    """

    group_name = group_col or (
        artifact.config.group_col if artifact is not None else "base_trial_id"
    )
    confidence_columns = {
        "acceptance_probability",
        "exact_probability",
        "family_probability",
    }
    if confidence_columns.issubset(validation_frame.columns) and not (
        validation_frame[group_name].duplicated().any()
    ):
        validation_solutions = validation_frame.copy()
    else:
        if artifact is None:
            raise ValueError(
                "An artifact is required to score/select long validation candidates"
            )
        scored = (
            validation_frame
            if {"ranker_score_raw", "candidate_rank"}.issubset(
                validation_frame.columns
            )
            else score_candidate_ranker(validation_frame, artifact)
        )
        validation_solutions = select_trial_solutions(
            scored,
            artifact,
            group_col=group_name,
        )
    sweep = sweep_solution_thresholds(
        validation_solutions,
        group_col=group_name,
        **sweep_kwargs,
    )
    feasible = sweep.loc[sweep["constraints_satisfied"]]
    if feasible.empty:
        if require_feasible:
            minimum_null = pd.to_numeric(
                sweep["null_accepted_rate"], errors="coerce"
            ).min()
            minimum_harmonic = pd.to_numeric(
                sweep["harmonic_only_rate"], errors="coerce"
            ).min()
            raise ValueError(
                "No validation threshold policy satisfies the requested "
                "constraints. Minimum observed null accepted rate="
                f"{minimum_null!r}, harmonic-only rate={minimum_harmonic!r}; "
                "inspect the saved sweep for exact, family, and coverage ceilings."
            )
        selected = sweep.iloc[0]
        constraints_satisfied = False
    else:
        selected = feasible.iloc[0]
        constraints_satisfied = True
    thresholds = StatusThresholds(
        secure_acceptance_threshold=float(
            selected["secure_acceptance_threshold"]
        ),
        secure_exact_threshold=float(selected["secure_exact_threshold"]),
        secure_family_threshold=float(selected["secure_family_threshold"]),
        tentative_acceptance_threshold=float(
            selected["tentative_acceptance_threshold"]
        ),
    )
    fingerprint_columns = [
        "acceptance_probability",
        "exact_probability",
        "family_probability",
        sweep_kwargs.get("exact_col", "is_exact"),
        sweep_kwargs.get("family_col", "is_harmonic_family"),
        sweep_kwargs.get("periodic_col", "truth_is_periodic"),
    ]
    metadata = {
        "schema_version": "period-candidate-thresholds-v1",
        "selection_split": "validation",
        "validation_fingerprint": _frame_fingerprint(
            validation_solutions,
            fingerprint_columns,
            group_col=group_name,
        ),
        "n_validation_trials": int(len(validation_solutions)),
        "constraints_satisfied": bool(constraints_satisfied),
        "target_null_accepted_rate": float(
            sweep_kwargs.get("target_null_accepted_rate", 0.05)
        ),
        "target_harmonic_only_rate": (
            None
            if sweep_kwargs.get("target_harmonic_only_rate", 0.05) is None
            else float(sweep_kwargs.get("target_harmonic_only_rate", 0.05))
        ),
        "minimum_exact_recovery_rate": (
            None
            if sweep_kwargs.get("minimum_exact_recovery_rate") is None
            else float(sweep_kwargs["minimum_exact_recovery_rate"])
        ),
        "minimum_family_recovery_rate": (
            None
            if sweep_kwargs.get("minimum_family_recovery_rate") is None
            else float(sweep_kwargs["minimum_family_recovery_rate"])
        ),
        "minimum_periodic_coverage": (
            None
            if sweep_kwargs.get("minimum_periodic_coverage") is None
            else float(sweep_kwargs["minimum_periodic_coverage"])
        ),
        "use_confidence_bounds_for_constraints": bool(
            sweep_kwargs.get("use_confidence_bounds_for_constraints", False)
        ),
        "constraint_confidence_level": float(
            sweep_kwargs.get("constraint_confidence_level", 0.95)
        ),
        "thresholds": thresholds.as_dict(),
    }
    metadata["threshold_policy_fingerprint"] = sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ThresholdTuningResult(
        thresholds=thresholds,
        sweep=sweep,
        constraints_satisfied=constraints_satisfied,
        metadata=metadata,
    )


def _trial_denominator(
    candidates: pd.DataFrame,
    *,
    group_col: str,
    periodic_col: str,
    trial_ids: Sequence[object] | pd.DataFrame | None,
    periodic_only: bool,
) -> pd.Index:
    if trial_ids is None:
        frame = candidates
    elif isinstance(trial_ids, pd.DataFrame):
        if group_col not in trial_ids.columns:
            raise ValueError(f"trial_ids table is missing {group_col!r}")
        frame = trial_ids
    else:
        return pd.Index(pd.unique(pd.Series(list(trial_ids))), name=group_col)
    if periodic_only and periodic_col in frame.columns:
        frame = frame.loc[_safe_bool(frame[periodic_col])]
    return pd.Index(pd.unique(frame[group_col]), name=group_col)


def candidate_oracle_summary(
    candidates: pd.DataFrame,
    *,
    group_col: str = "base_trial_id",
    exact_col: str = "is_exact",
    family_col: str = "is_harmonic_family",
    periodic_col: str = "truth_is_periodic",
    rank_col: str | None = "candidate_rank",
    score_col: str | None = None,
    top_ks: Sequence[int] = (1, 3, 5, 10),
    trial_ids: Sequence[object] | pd.DataFrame | None = None,
    periodic_only: bool = True,
) -> pd.DataFrame:
    """Summarize exact/family candidate ceiling and optional top-K coverage."""

    required = [group_col, exact_col, family_col]
    missing = [name for name in required if name not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate oracle table is missing columns: {missing}")
    denominator = _trial_denominator(
        candidates,
        group_col=group_col,
        periodic_col=periodic_col,
        trial_ids=trial_ids,
        periodic_only=periodic_only,
    )
    work = candidates.loc[candidates[group_col].isin(denominator)].copy()
    if score_col is not None:
        if score_col not in work.columns:
            raise ValueError(f"Missing score column {score_col!r}")
        work["_oracle_rank"] = work.groupby(group_col, sort=False)[score_col].rank(
            method="first", ascending=False
        )
        rank_name: str | None = "_oracle_rank"
    elif rank_col is not None and rank_col in work.columns:
        rank_name = rank_col
    else:
        rank_name = None

    exact_any = work.groupby(group_col, sort=False)[exact_col].any().reindex(
        denominator, fill_value=False
    )
    family_any = work.groupby(group_col, sort=False)[family_col].any().reindex(
        denominator, fill_value=False
    )
    payload: dict[str, Any] = {
        "n_trials": int(len(denominator)),
        "n_trials_with_candidates": int(work[group_col].nunique()),
        "exact_oracle_n": int(exact_any.sum()),
        "exact_oracle_rate": float(exact_any.mean()) if len(exact_any) else np.nan,
        "family_oracle_n": int(family_any.sum()),
        "family_oracle_rate": float(family_any.mean()) if len(family_any) else np.nan,
    }
    if rank_name is not None:
        for raw_k in top_ks:
            k = int(raw_k)
            if k <= 0:
                raise ValueError("top_ks must contain positive integers")
            subset = work.loc[pd.to_numeric(work[rank_name], errors="coerce") <= k]
            exact = subset.groupby(group_col, sort=False)[exact_col].any().reindex(
                denominator, fill_value=False
            )
            family = subset.groupby(group_col, sort=False)[family_col].any().reindex(
                denominator, fill_value=False
            )
            payload[f"exact_top_{k}_n"] = int(exact.sum())
            payload[f"exact_top_{k}_rate"] = (
                float(exact.mean()) if len(exact) else np.nan
            )
            payload[f"family_top_{k}_n"] = int(family.sum())
            payload[f"family_top_{k}_rate"] = (
                float(family.mean()) if len(family) else np.nan
            )
    return pd.DataFrame([payload])


def _method_tokens(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, list):
            return tuple(
                str(item).strip() for item in decoded if str(item).strip()
            )
    return tuple(
        token.strip()
        for token in re.split(r"[,;|+]", text)
        if token.strip()
    )


def method_ablation_summary(
    candidates: pd.DataFrame,
    *,
    method_col: str = "proposal_method",
    group_col: str = "base_trial_id",
    exact_col: str = "is_exact",
    family_col: str = "is_harmonic_family",
    periodic_col: str = "truth_is_periodic",
    trial_ids: Sequence[object] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Measure leave-one-proposer-out oracle coverage.

    A row listing multiple contributing methods is retained when at least one
    non-ablated contributor remains.  This matches a merged candidate bank:
    removing one proposer should not erase an independently proposed period.
    """

    method_name = _resolve_column(
        candidates,
        method_col,
        METHOD_COLUMN_ALIASES,
        role="proposal method",
    )
    token_series = candidates[method_name].map(_method_tokens)
    methods = sorted({token for tokens in token_series for token in tokens})
    baseline = candidate_oracle_summary(
        candidates,
        group_col=group_col,
        exact_col=exact_col,
        family_col=family_col,
        periodic_col=periodic_col,
        rank_col=None,
        trial_ids=trial_ids,
    ).iloc[0]
    rows = [
        {
            "ablation": "none",
            "removed_method": None,
            "n_candidate_rows": int(len(candidates)),
            "exact_oracle_rate": baseline["exact_oracle_rate"],
            "family_oracle_rate": baseline["family_oracle_rate"],
            "delta_exact_oracle_rate": 0.0,
            "delta_family_oracle_rate": 0.0,
        }
    ]
    for method in methods:
        keep = token_series.map(
            lambda tokens: method not in tokens or any(token != method for token in tokens)
        )
        ablated = candidates.loc[keep]
        summary = candidate_oracle_summary(
            ablated,
            group_col=group_col,
            exact_col=exact_col,
            family_col=family_col,
            periodic_col=periodic_col,
            rank_col=None,
            trial_ids=trial_ids if trial_ids is not None else candidates,
        ).iloc[0]
        rows.append(
            {
                "ablation": f"without_{method}",
                "removed_method": method,
                "n_candidate_rows": int(len(ablated)),
                "exact_oracle_rate": summary["exact_oracle_rate"],
                "family_oracle_rate": summary["family_oracle_rate"],
                "delta_exact_oracle_rate": float(
                    summary["exact_oracle_rate"] - baseline["exact_oracle_rate"]
                ),
                "delta_family_oracle_rate": float(
                    summary["family_oracle_rate"] - baseline["family_oracle_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_recovery(
    solutions: pd.DataFrame,
    *,
    group_col: str = "base_trial_id",
    exact_col: str = "is_exact",
    family_col: str = "is_harmonic_family",
    periodic_col: str = "truth_is_periodic",
    status_col: str = "solution_status",
    accepted_statuses: Sequence[str] = (
        "secure_fundamental",
        "secure_family_harmonic_ambiguous",
    ),
) -> pd.DataFrame:
    """Return one-row recovery, coverage, abstention, and null-FPR metrics."""

    required = [group_col, exact_col, family_col]
    missing = [name for name in required if name not in solutions.columns]
    if missing:
        raise ValueError(f"Solution table is missing columns: {missing}")
    if solutions[group_col].duplicated().any():
        raise ValueError("summarize_recovery expects one selected row per trial")
    exact = _safe_bool(solutions[exact_col]).to_numpy()
    family = _safe_bool(solutions[family_col]).to_numpy()
    periodic = (
        _safe_bool(solutions[periodic_col]).to_numpy()
        if periodic_col in solutions.columns
        else np.ones(len(solutions), dtype=bool)
    )
    if status_col in solutions.columns:
        status = solutions[status_col].astype("string").fillna("abstain")
        accepted = status.isin(tuple(accepted_statuses)).to_numpy()
    else:
        status = pd.Series("accepted", index=solutions.index, dtype="string")
        accepted = np.ones(len(solutions), dtype=bool)

    n_all = int(len(solutions))
    n_periodic = int(periodic.sum())
    n_null = int((~periodic).sum())
    exact_success = periodic & accepted & exact
    family_success = periodic & accepted & family
    harmonic_only = periodic & accepted & family & ~exact
    catastrophic = periodic & accepted & ~family
    periodic_accepted = periodic & accepted
    null_accepted = ~periodic & accepted

    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else np.nan

    payload = {
        "n_trials": n_all,
        "n_periodic": n_periodic,
        "n_null": n_null,
        "n_accepted": int(accepted.sum()),
        "coverage": rate(int(accepted.sum()), n_all),
        "abstention_rate": rate(int((~accepted).sum()), n_all),
        "exact_recovery_n": int(exact_success.sum()),
        "exact_recovery_rate": rate(int(exact_success.sum()), n_periodic),
        "family_recovery_n": int(family_success.sum()),
        "family_recovery_rate": rate(int(family_success.sum()), n_periodic),
        "harmonic_only_n": int(harmonic_only.sum()),
        "harmonic_only_rate": rate(int(harmonic_only.sum()), n_periodic),
        "conditional_harmonic_only_rate": rate(
            int(harmonic_only.sum()), int(periodic_accepted.sum())
        ),
        "catastrophic_n": int(catastrophic.sum()),
        "catastrophic_rate": rate(int(catastrophic.sum()), n_periodic),
        "conditional_catastrophic_rate": rate(
            int(catastrophic.sum()), int(periodic_accepted.sum())
        ),
        "periodic_accepted_n": int(periodic_accepted.sum()),
        "periodic_coverage": rate(int(periodic_accepted.sum()), n_periodic),
        "conditional_exact_rate": rate(
            int(exact_success.sum()), int(periodic_accepted.sum())
        ),
        "conditional_family_rate": rate(
            int(family_success.sum()), int(periodic_accepted.sum())
        ),
        "null_accepted_n": int(null_accepted.sum()),
        "null_accepted_rate": rate(int(null_accepted.sum()), n_null),
        "secure_fundamental_n": int(status.eq("secure_fundamental").sum()),
        "secure_family_harmonic_ambiguous_n": int(
            status.eq("secure_family_harmonic_ambiguous").sum()
        ),
        "tentative_n": int(status.eq("tentative").sum()),
        "abstain_n": int(status.eq("abstain").sum()),
    }
    return pd.DataFrame([payload])


def summarize_recovery_by_slice(
    solutions: pd.DataFrame,
    slice_columns: Sequence[str],
    **summary_kwargs: Any,
) -> pd.DataFrame:
    """Apply :func:`summarize_recovery` to requested morphology/regime slices."""

    missing = [name for name in slice_columns if name not in solutions.columns]
    if missing:
        raise ValueError(f"Missing recovery slice columns: {missing}")
    rows: list[pd.DataFrame] = []
    for name in slice_columns:
        for value, subset in solutions.groupby(name, dropna=False, sort=True):
            summary = summarize_recovery(subset, **summary_kwargs)
            summary.insert(0, "slice_value", value)
            summary.insert(0, "slice_column", name)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def artifact_metadata_json(
    artifact: CandidateRankerArtifact,
    *,
    indent: int = 2,
) -> str:
    """Serialize deterministic artifact metadata, excluding fitted objects."""

    return json.dumps(
        artifact.metadata,
        sort_keys=True,
        indent=int(indent),
        allow_nan=False,
    ) + "\n"


def save_candidate_ranker_artifact(
    artifact: CandidateRankerArtifact,
    path: str | Path,
) -> None:
    """Persist a fitted experimental artifact with joblib or pickle."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        import joblib

        joblib.dump(artifact, temporary)
    except ImportError:  # pragma: no cover - joblib is a project dependency
        with temporary.open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_candidate_ranker_artifact(
    path: str | Path,
) -> CandidateRankerArtifact:
    """Load and schema-check an experimental candidate-ranker artifact."""

    source = Path(path)
    try:
        import joblib

        artifact = joblib.load(source)
    except ImportError:  # pragma: no cover - joblib is a project dependency
        with source.open("rb") as handle:
            artifact = pickle.load(handle)
    if not isinstance(artifact, CandidateRankerArtifact):
        raise TypeError(f"{source} does not contain a CandidateRankerArtifact")
    if artifact.metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema: {artifact.metadata.get('schema_version')!r}"
        )
    stored_fingerprint = artifact.metadata.get("artifact_fingerprint")
    if (
        not isinstance(stored_fingerprint, str)
        or stored_fingerprint
        != _artifact_metadata_fingerprint(artifact.metadata)
    ):
        raise ValueError("Candidate-ranker artifact metadata fingerprint mismatch")
    if list(artifact.feature_columns) != artifact.metadata.get(
        "resolved_feature_columns"
    ):
        raise ValueError("Candidate-ranker feature schema does not match metadata")
    if _metadata_config(artifact.config) != artifact.metadata.get("config"):
        raise ValueError("Candidate-ranker configuration does not match metadata")
    if _model_fingerprint(artifact.model) != artifact.metadata.get(
        "model_fingerprint"
    ):
        raise ValueError("Candidate-ranker fitted model fingerprint mismatch")
    calibrators = {
        "candidate": artifact.candidate_calibrator,
        "acceptance": artifact.acceptance_calibrator,
        "exact": artifact.exact_calibrator,
        "family": artifact.family_calibrator,
    }
    if {
        name: calibrator.metadata()
        for name, calibrator in calibrators.items()
    } != artifact.metadata.get("calibrators"):
        raise ValueError("Candidate-ranker calibrator state does not match metadata")
    return artifact


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_HARMONIC_FACTORS",
    "BaselineFeatureSpec",
    "BinaryProbabilityCalibrator",
    "CandidateFeatureEncoder",
    "CandidateRankerArtifact",
    "RankerConfig",
    "StatusThresholds",
    "ThresholdTuningResult",
    "artifact_metadata_json",
    "assign_grouped_splits",
    "assign_solution_status",
    "candidate_oracle_summary",
    "default_baseline_feature_specs",
    "default_feature_columns",
    "fit_candidate_ranker",
    "label_candidates",
    "load_candidate_ranker_artifact",
    "method_ablation_summary",
    "rank_deterministic_baseline",
    "save_candidate_ranker_artifact",
    "score_candidate_ranker",
    "select_trial_solutions",
    "split_candidate_frame",
    "summarize_recovery",
    "summarize_recovery_by_slice",
    "sweep_solution_thresholds",
    "tune_solution_thresholds",
    "validate_feature_allowlist",
]
