from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb


def _prepare_xy(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    y = df[label_col].astype(str)
    drop_cols = {label_col, "candidate_id", "path", "asas_sn_id"}
    x = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    for col in x.columns:
        if x[col].dtype == object:
            x[col] = x[col].astype("category").cat.codes
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, y


def train_baseline_model(
    df: pd.DataFrame,
    *,
    label_col: str,
    seed: int = 42,
) -> tuple[object, dict]:
    x, y = _prepare_xy(df, label_col)
    classes = sorted(y.unique().tolist())
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        class_weight="balanced",
    )

    model.fit(x, y)
    train_acc = float((model.predict(x) == y).mean())
    metrics = {
        "n_samples": int(len(df)),
        "n_features": int(x.shape[1]),
        "classes": classes,
        "train_accuracy": train_acc,
    }
    return model, metrics


def save_model_artifacts(
    model: object,
    *,
    metrics: dict,
    feature_columns: list[str],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "candidate_classifier.joblib")

    (out_dir / "feature_schema.json").write_text(json.dumps({"features": feature_columns}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline candidate classifier model")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet with reviewer labels")
    parser.add_argument("--label-col", type=str, default="label", help="Supervised label column")
    parser.add_argument("--out-dir", type=Path, default=Path("output/ml"), help="Output directory for model artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    model, metrics = train_baseline_model(df, label_col=args.label_col, seed=args.seed)
    x, _ = _prepare_xy(df, args.label_col)
    save_model_artifacts(model, metrics=metrics, feature_columns=list(x.columns), out_dir=args.out_dir)
    print(f"Saved model artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
