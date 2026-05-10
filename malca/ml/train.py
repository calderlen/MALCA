from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from malca.config import (
    ML_N_ESTIMATORS,
    ML_LEARNING_RATE,
    ML_NUM_LEAVES,
    ML_SUBSAMPLE,
    ML_COLSAMPLE_BYTREE,
    ML_MIN_SAMPLES,
    ML_CV_FOLDS,
    ML_TOP_FEATURES,
)
from malca.ml.features import (
    ML_LABEL_COLUMN,
    build_ml_feature_schema,
    transform_ml_features,
)






def _load_input_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
        if isinstance(table, pd.Series):
            return table.to_frame()
        return table
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if isinstance(table, pd.Series):
            return table.to_frame()
        return table
    raise ValueError("Unsupported input file type. Use CSV or Parquet.")


def _build_model(seed: int) -> object:
    return lgb.LGBMClassifier(
        n_estimators=ML_N_ESTIMATORS,
        learning_rate=ML_LEARNING_RATE,
        num_leaves=ML_NUM_LEAVES,
        subsample=ML_SUBSAMPLE,
        colsample_bytree=ML_COLSAMPLE_BYTREE,
        random_state=seed,
        class_weight="balanced",
    )


def _filter_training_rows(
    df: pd.DataFrame,
    *,
    label_col: str,
    drop_unclassified: bool,
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    labels = df[label_col].astype("string").str.strip()
    valid = labels.notna() & (labels != "")
    if drop_unclassified:
        valid &= labels.str.lower() != "unclassified"

    out = df.loc[valid].copy()
    out[label_col] = labels.loc[valid].astype(str)
    return out


def _prepare_xy(
    df: pd.DataFrame,
    *,
    label_col: str,
    feature_schema: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    y = pd.Series(df[label_col], index=df.index).astype(str)
    schema = feature_schema or build_ml_feature_schema(df)
    x = transform_ml_features(df, schema)
    return x, y, schema


def _resolve_cv_folds(y: pd.Series, requested_folds: int) -> tuple[int, dict[str, int]]:
    class_counts = y.value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError("Need at least 2 distinct classes to train a classifier.")

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError("Each class needs at least 2 labeled examples for cross-validation.")

    effective_folds = min(int(requested_folds), min_class_count)
    if effective_folds < 2:
        raise ValueError("Cross-validation requires at least 2 folds.")

    if effective_folds != int(requested_folds):
        print(
            f"Warning: reducing --cv-folds from {requested_folds} to {effective_folds} "
            f"because the smallest class has {min_class_count} samples"
        )

    return effective_folds, {str(k): int(v) for k, v in class_counts.items()}


def _run_cross_validation(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    classes: list[str],
    seed: int,
    cv_folds: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_pred = pd.Series(index=y.index, dtype="object")
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = _build_model(seed + fold_idx)

        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(x_train, y_train)
        pred_valid = pd.Series(model.predict(x_valid), index=y_valid.index)
        oof_pred.loc[y_valid.index] = pred_valid

        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "accuracy": float(accuracy_score(y_valid, pred_valid)),
                "macro_f1": float(f1_score(y_valid, pred_valid, average="macro", zero_division=0)),
            }
        )

    oof_pred = oof_pred.fillna("<missing>")
    report_text = classification_report(y, oof_pred, labels=classes, zero_division=0)
    report_dict = classification_report(y, oof_pred, labels=classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y, oof_pred, labels=classes)

    print("\nCross-validation classification report (OOF):")
    print(report_text)
    print("Confusion matrix (rows=true, cols=pred):")
    print(conf_mat)

    fold_acc = [m["accuracy"] for m in fold_metrics]
    fold_macro_f1 = [m["macro_f1"] for m in fold_metrics]

    return {
        "cv_folds": int(cv_folds),
        "cv_fold_metrics": fold_metrics,
        "cv_accuracy": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_macro_f1": float(np.mean(fold_macro_f1)),
        "cv_macro_f1_std": float(np.std(fold_macro_f1)),
        "cv_classification_report": report_dict,
        "cv_confusion_matrix": conf_mat.tolist(),
    }


def train_baseline_model(
    df: pd.DataFrame,
    *,
    label_col: str,
    seed: int = 42,
    cv_folds: int = ML_CV_FOLDS,
    min_samples: int = ML_MIN_SAMPLES,
    drop_unclassified: bool = True,
) -> tuple[object, dict[str, Any], dict[str, Any]]:
    df_train = _filter_training_rows(df, label_col=label_col, drop_unclassified=drop_unclassified)
    if len(df_train) < int(min_samples):
        raise ValueError(
            f"Refusing to train: need at least {min_samples} labeled samples, got {len(df_train)}"
        )
    if len(df_train) < 100:
        print(f"Warning: only {len(df_train)} labeled samples available; model quality may be unstable")

    x, y, feature_schema = _prepare_xy(df_train, label_col=label_col)
    classes = sorted(y.unique().tolist())
    effective_folds, class_counts = _resolve_cv_folds(y, cv_folds)
    cv_metrics = _run_cross_validation(x, y, classes=classes, seed=seed, cv_folds=effective_folds)

    model = _build_model(seed)

    model.fit(x, y)
    train_pred = model.predict(x)
    train_acc = float(accuracy_score(y, train_pred))
    train_macro_f1 = float(f1_score(y, train_pred, average="macro", zero_division=0))

    top_features: list[dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        importances = list(zip(x.columns.tolist(), model.feature_importances_.tolist()))
        importances.sort(key=lambda t: t[1], reverse=True)
        top_features = [
            {"feature": str(name), "importance": float(score)}
            for name, score in importances[:ML_TOP_FEATURES]
        ]

    metrics = {
        "n_samples": int(len(df_train)),
        "n_features": int(x.shape[1]),
        "classes": classes,
        "class_counts": class_counts,
        "requested_cv_folds": int(cv_folds),
        **cv_metrics,
        "train_accuracy": train_acc,
        "train_macro_f1": train_macro_f1,
    }
    if top_features:
        metrics["feature_importance_top20"] = top_features

    return model, metrics, feature_schema


def save_model_artifacts(
    model: object,
    *,
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "candidate_classifier.joblib")

    (out_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2, sort_keys=True))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline candidate classifier model")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet with reviewer labels")
    parser.add_argument("--label-col", type=str, default=ML_LABEL_COLUMN, help="Supervised label column (default: event_class)")
    parser.add_argument("--out-dir", type=Path, default=Path("output/ml"), help="Output directory for model artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=ML_CV_FOLDS, help="Requested stratified CV folds")
    parser.add_argument("--min-samples", type=int, default=ML_MIN_SAMPLES, help="Minimum labeled sample count required to train")
    parser.add_argument(
        "--include-unclassified",
        action="store_true",
        help="Include rows labeled 'unclassified' instead of dropping them",
    )
    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if args.min_samples < 2:
        raise ValueError("--min-samples must be at least 2")

    df = _load_input_table(args.input)

    model, metrics, feature_schema = train_baseline_model(
        df,
        label_col=args.label_col,
        seed=args.seed,
        cv_folds=args.cv_folds,
        min_samples=args.min_samples,
        drop_unclassified=not args.include_unclassified,
    )
    save_model_artifacts(model, metrics=metrics, feature_schema=feature_schema, out_dir=args.out_dir)
    print(f"Saved model artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
