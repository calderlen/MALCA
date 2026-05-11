from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import pandas as pd

from malca.meta_analysis.ml.features import transform_ml_features


class _PredictModel(Protocol):
    classes_: Any

    def predict(self, x: pd.DataFrame) -> Any:
        ...

    def predict_proba(self, x: pd.DataFrame) -> Any:
        ...


def _load_table(path: Path) -> pd.DataFrame:
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


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False, compression="zstd")
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported output file type. Use CSV or Parquet.")


def load_model(model_dir: Path) -> tuple[_PredictModel, dict[str, Any]]:
    model_dir = Path(model_dir)
    model_path = model_dir / "candidate_classifier.joblib"
    schema_path = model_dir / "feature_schema.json"

    model = cast(_PredictModel, joblib.load(model_path))
    feature_schema = json.loads(schema_path.read_text())

    required_keys = {"features", "categorical_mappings", "unknown_category_code"}
    if not isinstance(feature_schema, dict) or not required_keys.issubset(feature_schema):
        raise ValueError(f"Invalid feature schema at {schema_path}")

    return model, feature_schema


def predict_candidates(
    model: _PredictModel,
    feature_schema: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    x = transform_ml_features(df, feature_schema)
    if x.empty:
        raise ValueError("No model features found in input for prediction")

    preds = model.predict(x)
    result = df.copy()
    result["ml_predicted_class"] = preds

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probs = model.predict_proba(x)
        classes = [str(cls) for cls in list(model.classes_)]
        if probs.shape[1] != len(classes):
            raise ValueError("Model class/probability shape mismatch")
        for idx, cls in enumerate(classes):
            result[f"ml_prob_{cls}"] = probs[:, idx]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with a trained ML model")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing model artifacts")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet of candidates")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet with predictions appended")
    args = parser.parse_args()

    model, feature_schema = load_model(args.model_dir)
    df = _load_table(args.input)
    scored = predict_candidates(model, feature_schema, df)
    _save_table(scored, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
