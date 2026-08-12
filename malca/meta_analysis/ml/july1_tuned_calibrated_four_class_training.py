"""Standalone tuned and calibrated four-class -> recurrence hierarchy.

This module deliberately does not replace the existing Review-LightGBM stack.
Each head uses a locked stratified 80/20 test split, randomized cross-validation
on the development partition, and sigmoid probability calibration.  The saved
production models reuse the selected hyperparameters and are fit on all labels.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from malca.meta_analysis.ml.candidate_features import (
    RECOVERY_FEATURE_SCHEMA_VERSION,
    add_recovery_bounded_event_features,
    default_recovery_feature_cache,
)
from malca.meta_analysis.ml.july1_four_class_hierarchical_training import (
    _human_unreviewed_mask,
    build_hierarchy_targets,
    compose_hierarchical_scores,
)
from malca.meta_analysis.ml.july1_review_training import (
    DEFAULT_DB_PATH,
    DEFAULT_RUN_DIR,
    _dipper_recurrence_features,
    _drop_exact_duplicate_features,
    _eight_class_features,
    _reviewed_mask,
    load_review_population,
)
from malca.meta_analysis.ml.review_lightgbm import (
    _feature_columns_for_target,
    _fit_feature_encoder,
    _raise_for_exact_duplicate_features,
    transform_features,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_RUN_DIR
    / "results"
    / "tuned_calibrated_four_class_dimming_hierarchy"
)
PROBABILITY_SCOPE = "reviewed_candidate_distribution"


@dataclass(frozen=True)
class SearchSpec:
    """Small search configuration for one hierarchy head."""

    n_iter: int
    cv_folds: int
    random_state: int
    parameter_distributions: Mapping[str, Sequence[Any]]
    cv_repeats: int = 1
    refit_metric: str = "neg_log_loss"


@dataclass
class TunedHeadResult:
    """Artifacts from one tuned and calibrated hierarchy head."""

    target_column: str
    label_classes: list[str]
    feature_columns: list[str]
    categorical_features: list[str]
    categorical_maps: dict[str, list[str]]
    class_counts: dict[str, int]
    best_params: dict[str, Any]
    split_assignments: pd.DataFrame
    search_trials: pd.DataFrame
    test_predictions: pd.DataFrame
    raw_test_metrics: dict[str, Any]
    calibrated_test_metrics: dict[str, Any]
    raw_reliability: pd.DataFrame
    calibrated_reliability: pd.DataFrame
    raw_model: lgb.LGBMClassifier
    calibrated_model: CalibratedClassifierCV
    evaluation_raw_model: lgb.LGBMClassifier
    evaluation_calibrated_model: CalibratedClassifierCV
    all_candidate_scores: pd.DataFrame
    search_spec: SearchSpec


def parent_search_spec(*, n_iter: int = 80, cv_folds: int = 5) -> SearchSpec:
    return SearchSpec(
        n_iter=n_iter,
        cv_folds=cv_folds,
        random_state=42,
        parameter_distributions={
            "n_estimators": [100, 200, 400, 700, 1000],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
            "num_leaves": [7, 15, 23, 31, 47],
            "min_child_samples": [10, 20, 40, 80],
            "colsample_bytree": [0.5, 0.65, 0.8, 1.0],
            "subsample": [0.65, 0.8, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0, 10.0],
            "reg_lambda": [0.0, 0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    )


def recurrence_search_spec(
    *, n_iter: int = 60, cv_folds: int = 5, cv_repeats: int = 3
) -> SearchSpec:
    return SearchSpec(
        n_iter=n_iter,
        cv_folds=cv_folds,
        random_state=43,
        parameter_distributions={
            "n_estimators": [100, 150, 250, 400],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [3, 5, 7],
            "min_child_samples": [16, 24, 32, 40],
            "colsample_bytree": [0.5, 0.7, 0.9],
            "subsample": [0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0, 10.0],
            "reg_lambda": [0.0, 0.1, 1.0, 10.0],
            "class_weight": [None],
        },
        cv_repeats=cv_repeats,
    )


def _safe_token(value: object) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z]+", "_", str(value).lower()).strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _dump_joblib(value: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path)


def _load_joblib(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def _probability_columns(label_classes: Sequence[str]) -> list[str]:
    return [f"prob_{_safe_token(label)}" for label in label_classes]


def _raw_score_columns(label_classes: Sequence[str]) -> list[str]:
    return [f"raw_score_{_safe_token(label)}" for label in label_classes]


def _prediction_frame(
    *,
    candidate_ids: Sequence[str],
    raw_model: lgb.LGBMClassifier,
    calibrated_model: CalibratedClassifierCV,
    features: pd.DataFrame,
    label_classes: Sequence[str],
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    raw = np.asarray(raw_model.predict_proba(features), dtype=float)
    calibrated = np.asarray(
        calibrated_model.predict_proba(features), dtype=float
    )
    if raw.ndim == 1:
        raw = np.column_stack([1.0 - raw, raw])
    if calibrated.ndim == 1:
        calibrated = np.column_stack([1.0 - calibrated, calibrated])
    predicted = calibrated.argmax(axis=1)
    out = pd.DataFrame({"candidate_id": [str(v) for v in candidate_ids]})
    if y_true is not None:
        out["y_true"] = [label_classes[int(value)] for value in y_true]
    out["y_pred"] = [label_classes[int(value)] for value in predicted]
    out["prediction_confidence"] = calibrated.max(axis=1)
    for idx, column in enumerate(_raw_score_columns(label_classes)):
        out[column] = raw[:, idx]
    for idx, column in enumerate(_probability_columns(label_classes)):
        out[column] = calibrated[:, idx]
    return out


def _metric_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_classes: Sequence[str],
) -> dict[str, Any]:
    n_classes = len(label_classes)
    predicted = probabilities.argmax(axis=1)
    precision, recall, per_class_f1, support = (
        precision_recall_fscore_support(
            y_true,
            predicted,
            labels=list(range(n_classes)),
            zero_division=0,
        )
    )
    truth = np.eye(n_classes, dtype=float)[y_true]
    metrics: dict[str, Any] = {
        "n_eval": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=list(range(n_classes)))
        ),
        "brier_macro": float(np.mean((truth - probabilities) ** 2)),
    }
    for idx, label in enumerate(label_classes):
        token = _safe_token(label)
        metrics[f"precision_class_{token}"] = float(precision[idx])
        metrics[f"recall_class_{token}"] = float(recall[idx])
        metrics[f"f1_class_{token}"] = float(per_class_f1[idx])
        metrics[f"support_class_{token}"] = int(support[idx])
        metrics[f"brier_class_{token}"] = float(
            np.mean((truth[:, idx] - probabilities[:, idx]) ** 2)
        )
    try:
        if n_classes == 2:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, probabilities[:, 1])
            )
            metrics["pr_auc"] = float(
                average_precision_score(truth[:, 1], probabilities[:, 1])
            )
        else:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    truth, probabilities, average="macro", multi_class="ovr"
                )
            )
            metrics["pr_auc_macro"] = float(
                average_precision_score(truth, probabilities, average="macro")
            )
    except ValueError:
        pass
    return metrics


def _reliability_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_classes: Sequence[str],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for class_idx, label in enumerate(label_classes):
        observed = (y_true == class_idx).astype(float)
        scores = probabilities[:, class_idx]
        bin_ids = np.clip(np.digitize(scores, bins[1:-1]), 0, n_bins - 1)
        ece = 0.0
        class_rows: list[dict[str, Any]] = []
        for bin_idx in range(n_bins):
            mask = bin_ids == bin_idx
            if not bool(mask.any()):
                continue
            mean_probability = float(scores[mask].mean())
            observed_rate = float(observed[mask].mean())
            count = int(mask.sum())
            ece += count / len(scores) * abs(mean_probability - observed_rate)
            class_rows.append(
                {
                    "class_label": str(label),
                    "probability_bin": bin_idx + 1,
                    "bin_lower": float(bins[bin_idx]),
                    "bin_upper": float(bins[bin_idx + 1]),
                    "n": count,
                    "mean_probability": mean_probability,
                    "observed_rate": observed_rate,
                }
            )
        for row in class_rows:
            row["expected_calibration_error"] = float(ece)
            rows.append(row)
    return pd.DataFrame(rows)


def _base_classifier(n_classes: int, random_state: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
        subsample_freq=1,
    )


def fit_tuned_calibrated_head(
    table: pd.DataFrame,
    *,
    target_column: str,
    requested_features: Sequence[str],
    population: pd.DataFrame,
    search_spec: SearchSpec,
    test_size: float = 0.20,
) -> TunedHeadResult:
    """Fit one standalone tuned and sigmoid-calibrated classifier head."""

    trainable = table[target_column].notna()
    work = table.loc[
        trainable, ["candidate_id", target_column, *requested_features]
    ].copy()
    if work["candidate_id"].duplicated().any():
        raise ValueError("Trainable candidate IDs must be unique")
    labels = work[target_column].astype(str)
    label_classes = sorted(labels.unique())
    label_to_int = {label: idx for idx, label in enumerate(label_classes)}
    y = labels.map(label_to_int).to_numpy(dtype=int)
    positions = np.arange(len(work))
    dev_pos, test_pos = train_test_split(
        positions,
        test_size=test_size,
        stratify=y,
        random_state=search_spec.random_state,
    )
    dev = work.iloc[dev_pos]
    test = work.iloc[test_pos]
    y_dev = y[dev_pos]
    y_test = y[test_pos]

    feature_columns, categorical_features = _feature_columns_for_target(
        dev,
        target_col=target_column,
        drop_columns={"candidate_id"},
        max_categorical_cardinality=50,
    )
    X_dev, categorical_maps = _fit_feature_encoder(
        dev, feature_columns, categorical_features
    )
    _raise_for_exact_duplicate_features(X_dev)
    X_test = transform_features(
        test,
        feature_columns=feature_columns,
        categorical_maps=categorical_maps,
    )
    X_trainable = transform_features(
        work,
        feature_columns=feature_columns,
        categorical_maps=categorical_maps,
    )
    X_population = transform_features(
        population,
        feature_columns=feature_columns,
        categorical_maps=categorical_maps,
    )

    minimum_class_count = int(pd.Series(y_dev).value_counts().min())
    n_folds = min(search_spec.cv_folds, minimum_class_count)
    if n_folds < 2:
        raise ValueError("Not enough development labels for stratified CV")
    if search_spec.cv_repeats < 1:
        raise ValueError("cv_repeats must be at least 1")
    calibration_cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=search_spec.random_state,
    )
    search_cv = calibration_cv
    if search_spec.cv_repeats > 1:
        search_cv = RepeatedStratifiedKFold(
            n_splits=n_folds,
            n_repeats=search_spec.cv_repeats,
            random_state=search_spec.random_state,
        )
    search = RandomizedSearchCV(
        estimator=_base_classifier(
            len(label_classes), search_spec.random_state
        ),
        param_distributions=dict(search_spec.parameter_distributions),
        n_iter=search_spec.n_iter,
        scoring={
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "neg_log_loss": "neg_log_loss",
        },
        refit=search_spec.refit_metric,
        cv=search_cv,
        random_state=search_spec.random_state,
        n_jobs=4,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_dev, y_dev)

    evaluation_raw_model = search.best_estimator_
    evaluation_calibrated_model = CalibratedClassifierCV(
        estimator=clone(evaluation_raw_model),
        method="sigmoid",
        cv=calibration_cv,
        n_jobs=1,
        ensemble=False,
    )
    evaluation_calibrated_model.fit(X_dev, y_dev)
    raw_test = np.asarray(
        evaluation_raw_model.predict_proba(X_test), dtype=float
    )
    calibrated_test = np.asarray(
        evaluation_calibrated_model.predict_proba(X_test), dtype=float
    )

    production_raw_model = clone(evaluation_raw_model)
    production_raw_model.fit(X_trainable, y)
    production_cv = StratifiedKFold(
        n_splits=min(n_folds, int(pd.Series(y).value_counts().min())),
        shuffle=True,
        random_state=search_spec.random_state + 1000,
    )
    production_calibrated_model = CalibratedClassifierCV(
        estimator=clone(evaluation_raw_model),
        method="sigmoid",
        cv=production_cv,
        n_jobs=1,
        ensemble=False,
    )
    production_calibrated_model.fit(X_trainable, y)

    split = pd.DataFrame(
        {
            "candidate_id": work["candidate_id"].astype(str),
            "target_column": target_column,
            "label": labels,
            "split": "development",
        }
    )
    split.iloc[test_pos, split.columns.get_loc("split")] = "test"
    test_predictions = _prediction_frame(
        candidate_ids=test["candidate_id"].astype(str),
        raw_model=evaluation_raw_model,
        calibrated_model=evaluation_calibrated_model,
        features=X_test,
        label_classes=label_classes,
        y_true=y_test,
    )
    all_candidate_scores = _prediction_frame(
        candidate_ids=population["candidate_id"].astype(str),
        raw_model=production_raw_model,
        calibrated_model=production_calibrated_model,
        features=X_population,
        label_classes=label_classes,
    )
    selected_rank_column = f"rank_test_{search_spec.refit_metric}"
    trials = pd.DataFrame(search.cv_results_).sort_values(
        selected_rank_column, ignore_index=True
    )
    keep = [
        column
        for column in trials.columns
        if column == "params"
        or column.startswith("param_")
        or column.startswith("mean_test_")
        or column.startswith("std_test_")
        or column.startswith("rank_test_")
    ]
    return TunedHeadResult(
        target_column=target_column,
        label_classes=label_classes,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        categorical_maps={key: list(value) for key, value in categorical_maps.items()},
        class_counts={
            str(key): int(value)
            for key, value in labels.value_counts().sort_index().items()
        },
        best_params=dict(search.best_params_),
        split_assignments=split.reset_index(drop=True),
        search_trials=trials[keep],
        test_predictions=test_predictions,
        raw_test_metrics=_metric_summary(y_test, raw_test, label_classes),
        calibrated_test_metrics=_metric_summary(
            y_test, calibrated_test, label_classes
        ),
        raw_reliability=_reliability_table(
            y_test, raw_test, label_classes
        ),
        calibrated_reliability=_reliability_table(
            y_test, calibrated_test, label_classes
        ),
        raw_model=production_raw_model,
        calibrated_model=production_calibrated_model,
        evaluation_raw_model=evaluation_raw_model,
        evaluation_calibrated_model=evaluation_calibrated_model,
        all_candidate_scores=all_candidate_scores,
        search_spec=search_spec,
    )


def _model_bundle(
    result: TunedHeadResult,
    *,
    evaluation: bool,
) -> dict[str, Any]:
    return {
        "bundle_version": 2,
        "raw_model": (
            result.evaluation_raw_model if evaluation else result.raw_model
        ),
        "calibrated_model": (
            result.evaluation_calibrated_model
            if evaluation
            else result.calibrated_model
        ),
        "target_column": result.target_column,
        "feature_columns": list(result.feature_columns),
        "categorical_features": list(result.categorical_features),
        "categorical_maps": {
            key: list(value) for key, value in result.categorical_maps.items()
        },
        "label_classes": list(result.label_classes),
        "raw_score_columns": _raw_score_columns(result.label_classes),
        "probability_columns": _probability_columns(result.label_classes),
        "calibration_method": "sigmoid_cross_validated",
        "probability_scope": PROBABILITY_SCOPE,
        "artifact_role": "evaluation" if evaluation else "production",
        "best_params": dict(result.best_params),
    }


def score_tuned_calibrated_head(
    bundle_or_path: Mapping[str, Any] | str | Path,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Score rows with a saved standalone tuned/calibrated head bundle."""

    bundle = (
        _load_joblib(Path(bundle_or_path))
        if isinstance(bundle_or_path, (str, Path))
        else dict(bundle_or_path)
    )
    if int(bundle.get("bundle_version", 0)) != 2:
        raise ValueError("Expected a tuned/calibrated bundle_version=2")
    X = transform_features(
        table,
        feature_columns=list(bundle["feature_columns"]),
        categorical_maps={
            str(key): list(value)
            for key, value in dict(bundle["categorical_maps"]).items()
        },
    )
    return _prediction_frame(
        candidate_ids=table["candidate_id"].astype(str),
        raw_model=bundle["raw_model"],
        calibrated_model=bundle["calibrated_model"],
        features=X,
        label_classes=list(bundle["label_classes"]),
    )


def _save_head(result: TunedHeadResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_joblib(_model_bundle(result, evaluation=False), output_dir / "model.joblib")
    _dump_joblib(
        _model_bundle(result, evaluation=True),
        output_dir / "evaluation_model.joblib",
    )
    result.raw_model.booster_.save_model(
        str(output_dir / "lightgbm_model.txt")
    )
    result.split_assignments.to_parquet(
        output_dir / "split_assignments.parquet", index=False
    )
    result.search_trials.to_csv(output_dir / "search_trials.csv", index=False)
    result.test_predictions.to_parquet(
        output_dir / "test_predictions.parquet", index=False
    )
    result.all_candidate_scores.to_parquet(
        output_dir / "all_candidates_scores.parquet", index=False
    )
    result.raw_reliability.to_csv(
        output_dir / "raw_calibration_by_bin.csv", index=False
    )
    result.calibrated_reliability.to_csv(
        output_dir / "calibrated_calibration_by_bin.csv", index=False
    )
    importance = pd.DataFrame(
        {
            "feature": result.feature_columns,
            "gain": result.raw_model.booster_.feature_importance(
                importance_type="gain"
            ),
            "split": result.raw_model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values(["gain", "split"], ascending=False, ignore_index=True)
    importance.to_csv(output_dir / "feature_importance_gain.csv", index=False)
    metadata = {
        "bundle_version": 2,
        "target_column": result.target_column,
        "n_rows": int(sum(result.class_counts.values())),
        "n_features": len(result.feature_columns),
        "class_counts": result.class_counts,
        "label_classes": result.label_classes,
        "best_params": result.best_params,
        "search": asdict(result.search_spec),
        "calibration_method": "sigmoid_cross_validated",
        "probability_scope": PROBABILITY_SCOPE,
        "population_probability_claim_supported": False,
        "raw_test_metrics": result.raw_test_metrics,
        "calibrated_test_metrics": result.calibrated_test_metrics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "best_params.json").write_text(
        json.dumps(
            result.best_params,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def train_tuned_calibrated_four_class_hierarchy(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
    parent_iterations: int = 80,
    recurrence_iterations: int = 60,
    cv_folds: int = 5,
    recurrence_cv_repeats: int = 3,
) -> dict[str, Any]:
    """Train both standalone heads and write calibrated composed scores."""

    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        Path(recovery_feature_cache)
        if recovery_feature_cache is not None
        else default_recovery_feature_cache(db_path)
    )
    table = load_review_population(
        db_path, keep_morphology_secondary_json=True
    )
    table = add_recovery_bounded_event_features(
        table, cache_path, workers=recovery_workers
    )
    parent_target, recurrence_target, audit = build_hierarchy_targets(table)

    parent_trainable = table[parent_target].notna()
    parent_features = _eight_class_features(table, parent_trainable)
    parent_features, parent_aliases = _drop_exact_duplicate_features(
        table.loc[parent_trainable], parent_features
    )
    recurrence_trainable = table[recurrence_target].notna()
    recurrence_features, recurrence_aliases = _dipper_recurrence_features(
        table, recurrence_trainable
    )

    parent = fit_tuned_calibrated_head(
        table,
        target_column=parent_target,
        requested_features=parent_features,
        population=table,
        search_spec=parent_search_spec(
            n_iter=parent_iterations, cv_folds=cv_folds
        ),
    )
    recurrence = fit_tuned_calibrated_head(
        table,
        target_column=recurrence_target,
        requested_features=recurrence_features,
        population=table,
        search_spec=recurrence_search_spec(
            n_iter=recurrence_iterations,
            cv_folds=cv_folds,
            cv_repeats=recurrence_cv_repeats,
        ),
    )
    _save_head(parent, output_dir / "parent_four_class")
    _save_head(recurrence, output_dir / "dimming_recurrence")

    parent_view = SimpleNamespace(
        label_classes=parent.label_classes,
        probability_columns=_probability_columns(parent.label_classes),
    )
    recurrence_view = SimpleNamespace(
        label_classes=recurrence.label_classes,
        probability_columns=_probability_columns(recurrence.label_classes),
    )
    composed = compose_hierarchical_scores(
        parent_view,
        parent.all_candidate_scores,
        recurrence_view,
        recurrence.all_candidate_scores,
    )
    context_columns = [
        column
        for column in (
            "candidate_id",
            "asas_sn_id",
            "lc_path",
            "status",
            "workflow_status",
            "event_class",
            "morphology_primary",
            "morphology_secondary",
            "physical_primary",
            "physical_secondary",
            parent_target,
            recurrence_target,
            "four_class_parent_label_source",
            "human_dipper_recurrence_label_source",
        )
        if column in table.columns
    ]
    scores = table[context_columns].merge(
        composed, on="candidate_id", how="left", validate="one_to_one"
    )
    raw_parent = parent.all_candidate_scores[
        ["candidate_id", *_raw_score_columns(parent.label_classes)]
    ].rename(
        columns={
            column: f"parent_{column}"
            for column in _raw_score_columns(parent.label_classes)
        }
    )
    raw_recurrence = recurrence.all_candidate_scores[
        ["candidate_id", *_raw_score_columns(recurrence.label_classes)]
    ].rename(
        columns={
            column: f"recurrence_{column}"
            for column in _raw_score_columns(recurrence.label_classes)
        }
    )
    scores = scores.merge(
        raw_parent, on="candidate_id", how="left", validate="one_to_one"
    ).merge(
        raw_recurrence,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    scores["is_human_unreviewed"] = _human_unreviewed_mask(table).to_numpy()
    score_path = output_dir / "all_candidates_tuned_calibrated_scores.parquet"
    scores.to_parquet(score_path, index=False)

    leaf_columns = [
        "prob_recurrent_dimming_event",
        "prob_non_recurrent_dimming_event",
        "prob_eclipsing_binary",
        "prob_junk",
        "prob_other",
    ]
    leaf_sum_error = float(
        (scores[leaf_columns].sum(axis=1) - 1.0).abs().max()
    )
    audit["dropped_duplicate_feature_aliases"] = {
        "parent": parent_aliases,
        "dimming_recurrence": recurrence_aliases,
    }
    audit["probability_semantics"] = {
        "method": "cross-validated sigmoid calibration",
        "scope": PROBABILITY_SCOPE,
        "population_probability_claim_supported": False,
        "recurrence_qualification": (
            "Conditional on dimming candidates with an unambiguous recurrent "
            "or single-dip review label."
        ),
    }
    (output_dir / "hierarchy_contract.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_key": "tuned_calibrated_four_class_dimming_hierarchy",
        "db_path": str(db_path),
        "output_dir": str(output_dir),
        "n_candidates": int(len(table)),
        "n_reviewed": int(_reviewed_mask(table).sum()),
        "score_path": str(score_path),
        "recovery_feature_cache": str(cache_path.expanduser().resolve()),
        "recovery_feature_schema_version": RECOVERY_FEATURE_SCHEMA_VERSION,
        "parent": {
            "class_counts": parent.class_counts,
            "n_features": len(parent.feature_columns),
            "search": asdict(parent.search_spec),
            "best_params": parent.best_params,
            "raw_test_metrics": parent.raw_test_metrics,
            "calibrated_test_metrics": parent.calibrated_test_metrics,
        },
        "dimming_recurrence": {
            "class_counts": recurrence.class_counts,
            "n_features": len(recurrence.feature_columns),
            "search": asdict(recurrence.search_spec),
            "best_params": recurrence.best_params,
            "raw_test_metrics": recurrence.raw_test_metrics,
            "calibrated_test_metrics": recurrence.calibrated_test_metrics,
        },
        "max_leaf_sum_error": leaf_sum_error,
        "review_db_modified": False,
        "warning": (
            "Probabilities are sigmoid-calibrated to the reviewed candidate "
            "distribution, which is not a random sample of the full survey."
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    print(
        "Trained standalone tuned/calibrated hierarchy: "
        f"{len(table):,} candidates -> {output_dir}"
    )
    return summary


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "SearchSpec",
    "TunedHeadResult",
    "fit_tuned_calibrated_head",
    "parent_search_spec",
    "recurrence_search_spec",
    "score_tuned_calibrated_head",
    "train_tuned_calibrated_four_class_hierarchy",
]
