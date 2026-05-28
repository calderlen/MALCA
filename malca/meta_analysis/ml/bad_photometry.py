"""Bad-photometry dropout models for MALCA reprocessing experiments.

This module treats candidates that appear in an original run but disappear
after subtraction-photometry recomputation as operational bad-photometry
examples.  It provides three comparable model paths:

* derived-only LightGBM on v1 candidate-table features,
* raw-only PyTorch model over native ASAS-SN light curves,
* hybrid LightGBM using derived features plus frozen raw-curve embeddings.

Heavy ML dependencies are imported lazily so the rest of MALCA can import this
module in environments that are not configured for model training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from malca.table_io import read_parquet_table, write_parquet_table


ID_COLUMNS = (
    "candidate_id",
    "asas_sn_id",
    "asassn_id",
    "source_id",
    "gaia_id",
    "object_id",
    "id",
)
PATH_COLUMNS = (
    "resolved_lc_path",
    "lc_path",
    "dat_path",
    "path",
    "source_path",
)
LIGHTCURVE_EXTENSIONS = (".dat3", ".dat2", ".dat", ".csv")
ASASSN_COLUMNS = (
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera#",
    "v_g_band",
    "saturated",
    "cam_field",
)
INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")
RAW_FEATURE_NAMES = (
    "time_norm",
    "time_rel_event",
    "log_dt_prev",
    "log_dt_next",
    "mag_minus_median",
    "mag_minus_band_median",
    "mag_robust_z",
    "log_error",
    "residual_over_error",
    "good_bad",
    "camera_hash",
    "band",
    "saturated",
    "cam_field_hash",
)
EMBEDDING_PREFIX = "raw_emb_"


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
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _dump_pickle(obj: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(obj, out)
    except Exception:
        with out.open("wb") as handle:
            pickle.dump(obj, handle)


def _load_pickle(path: str | Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "LightGBM is required for derived/hybrid bad-photometry models. "
            "Install the MALCA ML dependencies in the active environment."
        ) from exc
    return lgb


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "PyTorch is required for raw light-curve bad-photometry models. "
            "Install torch in the active environment."
        ) from exc
    return torch


def _maybe_sklearn_calibration():
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
    except Exception:  # pragma: no cover - depends on optional env
        return None, None
    return IsotonicRegression, LogisticRegression


def normalize_id(value: object) -> str:
    """Normalize candidate IDs for v1/v2 set arithmetic."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    match = INTEGER_FLOAT_RE.match(text)
    if match:
        text = match.group(1)
    return text.casefold()


def id_from_path(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    return Path(text).stem


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    if not table_path.exists():
        raise FileNotFoundError(f"Table not found: {table_path}")
    suffix = table_path.suffix.lower()
    if suffix == ".parquet" or table_path.is_dir():
        return read_parquet_table(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported table type: {table_path}")


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path).expanduser()
    if out.suffix.lower() == ".parquet":
        write_parquet_table(df, out)
    elif out.suffix.lower() == ".csv":
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    else:
        raise ValueError(f"Output table must be .parquet or .csv: {out}")


def choose_id_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Missing requested ID column {requested!r}")
        return requested
    for column in ID_COLUMNS:
        if column in df.columns:
            return column
    for column in PATH_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not auto-detect candidate ID column. "
        f"Expected one of: {', '.join((*ID_COLUMNS, *PATH_COLUMNS))}"
    )


def _source_id_series(df: pd.DataFrame, key: str) -> pd.Series:
    if key in PATH_COLUMNS:
        return df[key].map(id_from_path).map(normalize_id)
    return df[key].map(normalize_id)


def _display_id_series(df: pd.DataFrame, key: str) -> pd.Series:
    if key in PATH_COLUMNS:
        return df[key].map(id_from_path)
    return df[key].astype(str).str.strip()


def resolve_lightcurve_path(
    row: Mapping[str, Any],
    *,
    source_id: str | None = None,
    flat_lightcurve_dir: str | Path | None = None,
    extensions: Sequence[str] = LIGHTCURVE_EXTENSIONS,
) -> str:
    for column in PATH_COLUMNS:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        if candidate.exists() and candidate.suffix.lower() in extensions:
            return str(candidate)

    if flat_lightcurve_dir is None:
        return ""
    stem = str(source_id or row.get("dropout_source_id", "")).strip()
    if not stem:
        return ""
    flat_dir = Path(flat_lightcurve_dir).expanduser()
    for ext in extensions:
        candidate = flat_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return ""


def build_dropout_dataset(
    v1_candidates: str | Path,
    v2_candidates: str | Path,
    *,
    output: str | Path | None = None,
    key: str | None = None,
    v2_key: str | None = None,
    require_v2_subset: bool = True,
    flat_lightcurve_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build the canonical v1/v2 dropout training table.

    ``bad_photometry`` is 1 for rows present in v1 but absent from v2.
    ``survived_reprocessing`` is the complementary boolean label.
    """
    v1 = _read_table(v1_candidates)
    v2 = _read_table(v2_candidates)
    v1_key = choose_id_column(v1, key)
    effective_v2_key = v2_key
    if effective_v2_key is None and key is not None and key in v2.columns:
        effective_v2_key = key
    v2_key_name = choose_id_column(v2, effective_v2_key)

    out = v1.copy()
    out["dropout_source_id"] = _source_id_series(out, v1_key)
    out["dropout_display_id"] = _display_id_series(out, v1_key)
    if out["dropout_source_id"].eq("").any():
        bad = out.loc[out["dropout_source_id"].eq(""), [v1_key]].head(10)
        raise ValueError(f"v1 has empty normalized IDs; sample:\n{bad.to_string(index=False)}")

    v2_ids = set(_source_id_series(v2, v2_key_name))
    v2_ids.discard("")
    v1_ids = set(out["dropout_source_id"])
    extra_v2 = sorted(v2_ids - v1_ids)
    if require_v2_subset and extra_v2:
        sample = ", ".join(extra_v2[:10])
        raise ValueError(f"v2 contains IDs absent from v1; sample: {sample}")

    out["survived_reprocessing"] = out["dropout_source_id"].isin(v2_ids)
    out["bad_photometry"] = (~out["survived_reprocessing"]).astype("int8")
    out["reprocessing_label_source"] = "v1_minus_v2_dropout"
    out["v1_id_column"] = v1_key
    out["v2_id_column"] = v2_key_name

    if "candidate_id" not in out.columns:
        out["candidate_id"] = out["dropout_display_id"]

    if "resolved_lc_path" not in out.columns:
        resolved: list[str] = []
        for _, row in out.iterrows():
            resolved.append(
                resolve_lightcurve_path(
                    row,
                    source_id=str(row["dropout_display_id"]),
                    flat_lightcurve_dir=flat_lightcurve_dir,
                )
            )
        out["resolved_lc_path"] = resolved

    if output is not None:
        _write_table(out, output)
    return out


def make_stratified_group_split(
    df: pd.DataFrame,
    *,
    output: str | Path | None = None,
    label_col: str = "bad_photometry",
    group_col: str = "dropout_source_id",
    split_col: str = "split",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a deterministic stratified random split while keeping groups together."""
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train/validation/test fractions must sum to 1.0")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    work = df.copy()
    labels = pd.to_numeric(work[label_col], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"Label column {label_col!r} contains non-numeric values")
    work[label_col] = labels.astype(int)
    grouped = (
        work[[group_col, label_col]]
        .drop_duplicates()
        .groupby(group_col, dropna=False)[label_col]
        .max()
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    split_by_group: dict[str, str] = {}
    for label_value in sorted(grouped[label_col].unique()):
        ids = grouped.loc[grouped[label_col].eq(label_value), group_col].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * validation_fraction))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        else:
            n_train = min(n, max(n_train, 1))
            n_val = max(0, min(n - n_train, n_val))
        train_ids = ids[:n_train]
        val_ids = ids[n_train : n_train + n_val]
        test_ids = ids[n_train + n_val :]
        split_by_group.update({str(value): "train" for value in train_ids})
        split_by_group.update({str(value): "val" for value in val_ids})
        split_by_group.update({str(value): "test" for value in test_ids})

    work[split_col] = work[group_col].astype(str).map(split_by_group).fillna("train")
    if output is not None:
        _write_table(work, output)
    return work


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[sorter[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_np(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(score)
    sum_pos = float(ranks[y == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_np(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def risk_deciles(scores: Sequence[float]) -> np.ndarray:
    series = pd.Series(scores, dtype="float64")
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), 10, dtype=int)
    ranks = series.rank(method="first", na_option="bottom")
    deciles = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
    out = pd.Series(deciles, index=series.index).fillna(0).astype(int).to_numpy() + 1
    if out.max() < 10:
        scale = 10 / max(out.max(), 1)
        out = np.ceil(out * scale).astype(int)
    return out


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    top_fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    base_rate = float(y.mean()) if len(y) else float("nan")
    metrics: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int((y == 1).sum()),
        "base_rate": base_rate,
        "pr_auc": average_precision_np(y, score),
        "roc_auc": roc_auc_np(y, score),
    }
    if len(y) == 0:
        return metrics
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    total_pos = int((y == 1).sum())
    for frac in top_fractions:
        k = max(1, int(math.ceil(len(y) * float(frac))))
        top = y_sorted[:k]
        precision = float(top.mean()) if k else float("nan")
        recall = float(top.sum() / total_pos) if total_pos else float("nan")
        lift = float(precision / base_rate) if base_rate and np.isfinite(base_rate) else float("nan")
        key = f"top_{int(round(frac * 100))}pct"
        metrics[f"{key}_n"] = int(k)
        metrics[f"{key}_precision"] = precision
        metrics[f"{key}_recall"] = recall
        metrics[f"{key}_lift"] = lift
    return metrics


def calibration_by_decile(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "label": np.asarray(y_true, dtype=int),
            "score": np.asarray(y_score, dtype=float),
        }
    )
    df = df.loc[np.isfinite(df["score"].to_numpy(dtype=float))].copy()
    if df.empty:
        return pd.DataFrame(columns=["risk_decile", "n", "mean_score", "observed_rate"])
    df["risk_decile"] = risk_deciles(df["score"])
    return (
        df.groupby("risk_decile", as_index=False)
        .agg(
            n=("label", "size"),
            mean_score=("score", "mean"),
            observed_rate=("label", "mean"),
        )
        .sort_values("risk_decile")
        .reset_index(drop=True)
    )


TABULAR_EXACT_DROP_COLUMNS = {
    "bad_photometry",
    "survived_reprocessing",
    "reprocessing_label_source",
    "split",
    "candidate_id",
    "dropout_source_id",
    "dropout_display_id",
    "source_id",
    "asas_sn_id",
    "asassn_id",
    "gaia_id",
    "object_id",
    "id",
    "path",
    "lc_path",
    "dat_path",
    "source_path",
    "resolved_lc_path",
    "index_csv",
    "lc_dir",
    "camera_ids",
    "excluded_cameras",
    "payload_json",
    "notes",
    "reviewer",
    "updated_at",
    "v1_id_column",
    "v2_id_column",
    "_raw_row_id",
    "raw_bad_photometry_probability",
}
TABULAR_DROP_PREFIXES = (
    "v2_",
    "review_",
    "legacy_review",
)
TABULAR_DROP_SUFFIXES = (
    "_json",
    "_path",
    "_id",
)


@dataclass
class TabularModelConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    class_weight: str | None = "balanced"
    random_state: int = 42
    n_jobs: int = 1
    max_categorical_cardinality: int = 50
    calibration: str = "isotonic"


@dataclass
class TabularFeatureEncoder:
    feature_columns: list[str]
    categorical_columns: list[str]
    categorical_maps: dict[str, list[str]]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for col in self.feature_columns:
            if col in self.categorical_maps:
                values = (
                    df[col].fillna("").astype(str).str.strip()
                    if col in df.columns
                    else pd.Series("", index=df.index)
                )
                mapping = {value: idx for idx, value in enumerate(self.categorical_maps[col])}
                columns[col] = values.map(mapping).fillna(-1).astype("int32")
            else:
                values = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
                columns[col] = (
                    pd.to_numeric(values, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype("float64")
                )
        return pd.DataFrame(columns, index=df.index)


@dataclass
class ProbabilityCalibrator:
    method: str = "identity"
    model: Any = field(default=None, repr=False)

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        arr = np.asarray(scores, dtype=float)
        if self.method == "identity" or self.model is None:
            return np.clip(arr, 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(self.model.predict(arr), 0.0, 1.0)
        if self.method == "platt":
            return np.clip(self.model.predict_proba(arr.reshape(-1, 1))[:, 1], 0.0, 1.0)
        return np.clip(arr, 0.0, 1.0)


def _is_tabular_drop_column(column: str, *, target_col: str) -> bool:
    if column == target_col:
        return True
    if column.startswith(EMBEDDING_PREFIX):
        return False
    if column in TABULAR_EXACT_DROP_COLUMNS:
        return True
    if any(column.startswith(prefix) for prefix in TABULAR_DROP_PREFIXES):
        return True
    if any(column.endswith(suffix) for suffix in TABULAR_DROP_SUFFIXES):
        return True
    return False


def choose_tabular_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str = "bad_photometry",
    max_categorical_cardinality: int = 50,
    extra_drop_columns: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    extra_drop = set(extra_drop_columns)
    feature_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in df.columns:
        if col in extra_drop or _is_tabular_drop_column(col, target_col=target_col):
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            feature_columns.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        nonempty = series.dropna().astype(str).str.strip()
        if nonempty.empty:
            continue
        if nonempty.nunique() <= max_categorical_cardinality:
            feature_columns.append(col)
            categorical_columns.append(col)
    return feature_columns, categorical_columns


def fit_tabular_encoder(
    df: pd.DataFrame,
    *,
    target_col: str = "bad_photometry",
    max_categorical_cardinality: int = 50,
    extra_drop_columns: Iterable[str] = (),
) -> TabularFeatureEncoder:
    feature_columns, categorical_columns = choose_tabular_feature_columns(
        df,
        target_col=target_col,
        max_categorical_cardinality=max_categorical_cardinality,
        extra_drop_columns=extra_drop_columns,
    )
    categorical_maps: dict[str, list[str]] = {}
    for col in categorical_columns:
        values = df[col].fillna("").astype(str).str.strip()
        categorical_maps[col] = sorted(value for value in values.unique() if value)
    return TabularFeatureEncoder(
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        categorical_maps=categorical_maps,
    )


def fit_probability_calibrator(
    y_true: Sequence[int],
    raw_scores: Sequence[float],
    *,
    method: str = "isotonic",
) -> ProbabilityCalibrator:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(raw_scores, dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(np.unique(y)) < 2 or len(y) < 10:
        return ProbabilityCalibrator("identity")
    isotonic_cls, logistic_cls = _maybe_sklearn_calibration()
    if method == "isotonic" and isotonic_cls is not None:
        model = isotonic_cls(out_of_bounds="clip")
        model.fit(score, y)
        return ProbabilityCalibrator("isotonic", model)
    if method == "platt" and logistic_cls is not None:
        model = logistic_cls(max_iter=1000)
        model.fit(score.reshape(-1, 1), y)
        return ProbabilityCalibrator("platt", model)
    return ProbabilityCalibrator("identity")


def train_lightgbm_classifier(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    model_name: str,
    target_col: str = "bad_photometry",
    split_col: str = "split",
    config: TabularModelConfig | None = None,
    extra_drop_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Train one LightGBM classifier and write all standard artifacts."""
    lgb = _require_lightgbm()
    cfg = config or TabularModelConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_col not in df.columns:
        raise ValueError(f"Missing split column: {split_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train = df.loc[df[split_col].eq("train")].copy()
    val = df.loc[df[split_col].eq("val")].copy()
    test = df.loc[df[split_col].eq("test")].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, val, and test splits must all be non-empty")

    encoder = fit_tabular_encoder(
        train,
        target_col=target_col,
        max_categorical_cardinality=cfg.max_categorical_cardinality,
        extra_drop_columns=extra_drop_columns,
    )
    if not encoder.feature_columns:
        raise ValueError("No usable tabular feature columns found")
    X_train = encoder.transform(train)
    X_val = encoder.transform(val)
    X_test = encoder.transform(test)
    y_train = pd.to_numeric(train[target_col], errors="raise").astype(int).to_numpy()
    y_val = pd.to_numeric(val[target_col], errors="raise").astype(int).to_numpy()
    y_test = pd.to_numeric(test[target_col], errors="raise").astype(int).to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_samples=cfg.min_child_samples,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {}
    if encoder.categorical_columns:
        fit_kwargs["categorical_feature"] = [
            col for col in encoder.categorical_columns if col in X_train.columns
        ]
    model.fit(X_train, y_train, **fit_kwargs)

    val_raw = model.predict_proba(X_val)[:, 1]
    calibrator = fit_probability_calibrator(y_val, val_raw, method=cfg.calibration)
    test_raw = model.predict_proba(X_test)[:, 1]
    test_score = calibrator.predict(test_raw)
    metrics = binary_metrics(y_test, test_score)
    calibration = calibration_by_decile(y_test, test_score)

    predictions = test[["candidate_id", target_col]].copy() if "candidate_id" in test.columns else pd.DataFrame()
    if predictions.empty:
        predictions["candidate_id"] = test.index.astype(str)
        predictions[target_col] = y_test
    predictions["bad_photometry_risk"] = test_score
    predictions["risk_decile"] = risk_deciles(test_score)
    predictions["model_name"] = model_name

    feature_importance = pd.DataFrame(
        {
            "feature": encoder.feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    bundle = {
        "model": model,
        "encoder": encoder,
        "calibrator": calibrator,
        "config": cfg,
        "model_name": model_name,
    }
    model_path = out_dir / "model.joblib"
    _dump_pickle(bundle, model_path)
    try:
        model.booster_.save_model(str(out_dir / "lightgbm_model.txt"))
    except Exception:
        pass
    _write_table(predictions, out_dir / "test_predictions.parquet")
    _write_table(calibration, out_dir / "calibration_by_decile.parquet")
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)
    (out_dir / "feature_columns.json").write_text(
        json.dumps(encoder.feature_columns, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_json(
        out_dir / "metrics.json",
        {
            "model_name": model_name,
            "metrics": metrics,
            "config": asdict(cfg),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "n_features": int(len(encoder.feature_columns)),
            "categorical_features": encoder.categorical_columns,
        },
    )
    return {
        "model_name": model_name,
        "metrics": metrics,
        "model_path": model_path,
        "predictions": predictions,
        "calibration": calibration,
        "feature_importance": feature_importance,
    }


@dataclass
class RawPreprocessConfig:
    full_max_points: int = 2048
    event_max_points: int = 512
    default_event_half_width_days: float = 180.0
    min_event_half_width_days: float = 90.0
    max_event_half_width_days: float = 730.0
    hash_buckets: int = 1024
    error_floor: float = 1.0e-4


@dataclass
class RawModelConfig:
    d_model: int = 128
    residual_blocks: int = 3
    transformer_layers: int = 4
    attention_heads: int = 4
    dropout: float = 0.1
    embedding_dim: int = 128


@dataclass
class RawTrainingConfig:
    max_epochs: int = 50
    patience: int = 6
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


def read_native_lightcurve(path: str | Path) -> pd.DataFrame:
    """Read a native MALCA light curve without importing the full pipeline stack."""
    lc_path = Path(path).expanduser()
    if not lc_path.exists():
        raise FileNotFoundError(f"Light curve not found: {lc_path}")
    if lc_path.suffix.lower() == ".csv":
        return pd.read_csv(lc_path)
    df = pd.read_csv(
        lc_path,
        sep=r"\s+",
        names=list(ASASSN_COLUMNS),
        comment="#",
        engine="python",
    )
    for col in ("JD", "mag", "error", "good_bad", "camera#", "v_g_band", "saturated"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stable_hash(value: object, *, buckets: int = 1024) -> float:
    text = str(value)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % max(int(buckets), 1)
    if buckets <= 1:
        return 0.0
    return float((bucket / (buckets - 1)) * 2.0 - 1.0)


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def choose_primary_event(row: Mapping[str, Any]) -> dict[str, Any]:
    dip_score = _finite_float(row.get("dipper_score"))
    jump_score = _finite_float(row.get("jumper_score"))
    dip_t0 = _finite_float(row.get("dip_best_t0"))
    jump_t0 = _finite_float(row.get("jump_best_t0"))
    use_dip = True
    if math.isfinite(jump_score) and (not math.isfinite(dip_score) or jump_score > dip_score):
        use_dip = False
    if use_dip and not math.isfinite(dip_t0) and math.isfinite(jump_t0):
        use_dip = False
    if not use_dip and not math.isfinite(jump_t0) and math.isfinite(dip_t0):
        use_dip = True

    if use_dip:
        duration = _finite_float(row.get("dip_max_run_duration"))
        return {"kind": "dip", "t0": dip_t0, "duration_days": duration}
    duration = _finite_float(row.get("jump_max_run_duration"))
    return {"kind": "jump", "t0": jump_t0, "duration_days": duration}


def event_half_width_days(
    duration_days: float,
    *,
    config: RawPreprocessConfig,
) -> float:
    duration = duration_days if math.isfinite(duration_days) and duration_days > 0 else 0.0
    half_width = max(config.default_event_half_width_days, 3.0 * duration)
    return float(
        np.clip(
            half_width,
            config.min_event_half_width_days,
            config.max_event_half_width_days,
        )
    )


def _quantile_subsample_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(int))


def _event_subsample_indices(jd: np.ndarray, t0: float, cap: int) -> np.ndarray:
    n = len(jd)
    if n <= cap:
        return np.arange(n, dtype=int)
    nearest_count = max(1, cap // 2)
    nearest = np.argsort(np.abs(jd - t0), kind="mergesort")[:nearest_count]
    shoulder_count = max(1, cap - len(nearest))
    shoulder = _quantile_subsample_indices(n, shoulder_count)
    combined = np.unique(np.concatenate([nearest, shoulder]))
    if len(combined) > cap:
        order = np.argsort(np.abs(jd[combined] - t0), kind="mergesort")[:cap]
        combined = combined[order]
    return np.sort(combined)


def _robust_mad(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    return 1.4826 * mad


def _lightcurve_stats(df: pd.DataFrame) -> dict[str, Any]:
    jd = pd.to_numeric(df.get("JD"), errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df.get("mag"), errors="coerce").to_numpy(dtype=float)
    band = pd.to_numeric(df.get("v_g_band"), errors="coerce")
    finite_mag = mag[np.isfinite(mag)]
    median = float(np.nanmedian(finite_mag)) if finite_mag.size else 0.0
    sigma = _robust_mad(mag)
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(finite_mag)) if finite_mag.size else 1.0
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    band_medians: dict[float, float] = {}
    if "v_g_band" in df.columns:
        for value in sorted(band.dropna().unique()):
            mask = band.eq(value).to_numpy()
            vals = mag[mask & np.isfinite(mag)]
            if vals.size:
                band_medians[float(value)] = float(np.nanmedian(vals))
    finite_jd = jd[np.isfinite(jd)]
    jd_min = float(np.nanmin(finite_jd)) if finite_jd.size else 0.0
    jd_max = float(np.nanmax(finite_jd)) if finite_jd.size else jd_min
    return {
        "mag_median": median,
        "mag_sigma": sigma,
        "band_medians": band_medians,
        "jd_min": jd_min,
        "jd_max": jd_max,
    }


def _point_feature_matrix(
    df: pd.DataFrame,
    *,
    stats: Mapping[str, Any],
    t0: float,
    half_width: float,
    config: RawPreprocessConfig,
) -> np.ndarray:
    jd = pd.to_numeric(df.get("JD"), errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df.get("mag"), errors="coerce").to_numpy(dtype=float)
    err_source = df["error"] if "error" in df.columns else pd.Series([0.03] * len(df), index=df.index)
    good_source = df["good_bad"] if "good_bad" in df.columns else pd.Series([1] * len(df), index=df.index)
    band_source = df["v_g_band"] if "v_g_band" in df.columns else pd.Series([-1] * len(df), index=df.index)
    sat_source = df["saturated"] if "saturated" in df.columns else pd.Series([0] * len(df), index=df.index)
    err = pd.to_numeric(err_source, errors="coerce").to_numpy(dtype=float)
    good_bad = pd.to_numeric(good_source, errors="coerce").fillna(0).to_numpy(dtype=float)
    camera = df["camera#"] if "camera#" in df.columns else pd.Series([""] * len(df), index=df.index)
    band = pd.to_numeric(band_source, errors="coerce").fillna(-1).to_numpy(dtype=float)
    saturated = pd.to_numeric(sat_source, errors="coerce").fillna(0).to_numpy(dtype=float)
    cam_field = df["cam_field"] if "cam_field" in df.columns else pd.Series([""] * len(df), index=df.index)

    jd_min = float(stats["jd_min"])
    jd_max = float(stats["jd_max"])
    span = jd_max - jd_min if jd_max > jd_min else 1.0
    time_norm = ((jd - jd_min) / span) * 2.0 - 1.0
    if math.isfinite(t0) and half_width > 0:
        time_rel_event = np.clip((jd - t0) / half_width, -10.0, 10.0)
    else:
        time_rel_event = np.zeros_like(jd)

    dt_prev = np.zeros_like(jd)
    dt_next = np.zeros_like(jd)
    if len(jd) > 1:
        diffs = np.diff(jd)
        dt_prev[1:] = diffs
        dt_next[:-1] = diffs
    log_dt_prev = np.log1p(np.clip(np.abs(dt_prev), 0.0, None))
    log_dt_next = np.log1p(np.clip(np.abs(dt_next), 0.0, None))

    mag_median = float(stats["mag_median"])
    mag_sigma = float(stats["mag_sigma"])
    mag_minus_median = mag - mag_median
    band_medians = stats.get("band_medians", {})
    mag_minus_band = np.empty_like(mag)
    for idx, value in enumerate(band):
        median = band_medians.get(float(value), mag_median)
        mag_minus_band[idx] = mag[idx] - float(median)
    mag_robust_z = mag_minus_median / mag_sigma
    err_clipped = np.clip(err, config.error_floor, None)
    log_error = np.log10(err_clipped)
    residual_over_error = np.clip(mag_minus_median / err_clipped, -50.0, 50.0)

    camera_hash = np.asarray(
        [_stable_hash(value, buckets=config.hash_buckets) for value in camera],
        dtype=float,
    )
    cam_field_hash = np.asarray(
        [_stable_hash(value, buckets=config.hash_buckets) for value in cam_field],
        dtype=float,
    )
    matrix = np.column_stack(
        [
            time_norm,
            time_rel_event,
            log_dt_prev,
            log_dt_next,
            mag_minus_median,
            mag_minus_band,
            mag_robust_z,
            log_error,
            residual_over_error,
            good_bad,
            camera_hash,
            band,
            saturated,
            cam_field_hash,
        ]
    )
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix.astype("float32")


def _pad_matrix(matrix: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((cap, len(RAW_FEATURE_NAMES)), dtype="float32")
    mask = np.zeros(cap, dtype=bool)
    n = min(len(matrix), cap)
    if n:
        out[:n, :] = matrix[:n, :]
        mask[:n] = True
    return out, mask


def preprocess_raw_lightcurve(
    path: str | Path,
    row: Mapping[str, Any],
    *,
    config: RawPreprocessConfig | None = None,
) -> dict[str, Any]:
    """Convert one raw light curve into full-curve and event-window tensors."""
    cfg = config or RawPreprocessConfig()
    result = {
        "full_x": np.zeros((cfg.full_max_points, len(RAW_FEATURE_NAMES)), dtype="float32"),
        "full_mask": np.zeros(cfg.full_max_points, dtype=bool),
        "event_x": np.zeros((cfg.event_max_points, len(RAW_FEATURE_NAMES)), dtype="float32"),
        "event_mask": np.zeros(cfg.event_max_points, dtype=bool),
        "flags": np.zeros(2, dtype="float32"),
        "raw_available": False,
        "event_window_available": False,
        "event_kind": "",
    }
    try:
        df = read_native_lightcurve(path)
    except Exception:
        return result
    if df.empty or "JD" not in df.columns or "mag" not in df.columns:
        return result
    df = df.copy()
    df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df = df.loc[df["JD"].notna() & df["mag"].notna()].sort_values("JD").reset_index(drop=True)
    if df.empty:
        return result

    event = choose_primary_event(row)
    t0 = float(event["t0"])
    half_width = event_half_width_days(float(event["duration_days"]), config=cfg)
    stats = _lightcurve_stats(df)

    full_idx = _quantile_subsample_indices(len(df), cfg.full_max_points)
    full_matrix = _point_feature_matrix(
        df.iloc[full_idx].reset_index(drop=True),
        stats=stats,
        t0=t0,
        half_width=half_width,
        config=cfg,
    )
    full_x, full_mask = _pad_matrix(full_matrix, cfg.full_max_points)
    result["full_x"] = full_x
    result["full_mask"] = full_mask
    result["raw_available"] = True
    result["flags"][0] = 1.0
    result["event_kind"] = str(event["kind"])

    if math.isfinite(t0):
        jd = df["JD"].to_numpy(dtype=float)
        in_window = np.abs(jd - t0) <= half_width
        event_df = df.loc[in_window].copy().reset_index(drop=True)
        if not event_df.empty:
            event_jd = event_df["JD"].to_numpy(dtype=float)
            event_idx = _event_subsample_indices(event_jd, t0, cfg.event_max_points)
            event_matrix = _point_feature_matrix(
                event_df.iloc[event_idx].reset_index(drop=True),
                stats=stats,
                t0=t0,
                half_width=half_width,
                config=cfg,
            )
            event_x, event_mask = _pad_matrix(event_matrix, cfg.event_max_points)
            result["event_x"] = event_x
            result["event_mask"] = event_mask
            result["event_window_available"] = True
            result["flags"][1] = 1.0
    return result


def resolve_model_lightcurve_path(row: Mapping[str, Any]) -> str:
    for column in ("resolved_lc_path", *PATH_COLUMNS):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


class RawLightCurveDataset:
    """Lazy raw-light-curve dataset for PyTorch DataLoader."""

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        target_col: str = "bad_photometry",
        preprocess_config: RawPreprocessConfig | None = None,
        include_labels: bool = True,
    ) -> None:
        self.table = table.reset_index(drop=True).copy()
        self.target_col = target_col
        self.preprocess_config = preprocess_config or RawPreprocessConfig()
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.table.iloc[idx]
        lc_path = resolve_model_lightcurve_path(row)
        processed = preprocess_raw_lightcurve(lc_path, row, config=self.preprocess_config)
        candidate_id = str(row.get("candidate_id", row.get("dropout_display_id", idx)))
        item = {
            "candidate_id": candidate_id,
            "full_x": processed["full_x"],
            "full_mask": processed["full_mask"],
            "event_x": processed["event_x"],
            "event_mask": processed["event_mask"],
            "flags": processed["flags"],
            "raw_available": bool(processed["raw_available"]),
            "event_window_available": bool(processed["event_window_available"]),
        }
        if self.include_labels:
            item["label"] = int(row[self.target_col])
        return item


def raw_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = _require_torch()
    labels = [item.get("label") for item in batch]
    out = {
        "candidate_id": [str(item["candidate_id"]) for item in batch],
        "full_x": torch.tensor(np.stack([item["full_x"] for item in batch]), dtype=torch.float32),
        "full_mask": torch.tensor(np.stack([item["full_mask"] for item in batch]), dtype=torch.bool),
        "event_x": torch.tensor(np.stack([item["event_x"] for item in batch]), dtype=torch.float32),
        "event_mask": torch.tensor(np.stack([item["event_mask"] for item in batch]), dtype=torch.bool),
        "flags": torch.tensor(np.stack([item["flags"] for item in batch]), dtype=torch.float32),
        "raw_available": [bool(item["raw_available"]) for item in batch],
        "event_window_available": [bool(item["event_window_available"]) for item in batch],
    }
    if labels and labels[0] is not None:
        out["label"] = torch.tensor(labels, dtype=torch.float32)
    return out


def create_raw_model(
    *,
    input_dim: int = len(RAW_FEATURE_NAMES),
    config: RawModelConfig | None = None,
):
    torch = _require_torch()
    nn = torch.nn
    cfg = config or RawModelConfig()

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.net(x)

    class MaskedAttentionPool(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.score = nn.Linear(dim, 1)

        def forward(self, x, mask):
            scores = self.score(x).squeeze(-1)
            scores = scores.masked_fill(~mask, -1.0e9)
            weights = torch.softmax(scores, dim=1)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            weights = weights / denom
            return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

    class FullCurveEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(input_dim, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model),
            )
            self.blocks = nn.ModuleList(
                [ResidualBlock(cfg.d_model, cfg.dropout) for _ in range(cfg.residual_blocks)]
            )
            self.attn_pool = MaskedAttentionPool(cfg.d_model)
            self.out = nn.Sequential(
                nn.Linear(cfg.d_model * 3, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model),
            )

        def forward(self, x, mask):
            h = self.input(x)
            for block in self.blocks:
                h = block(h)
            mask_f = mask.unsqueeze(-1).float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            mean = (h * mask_f).sum(dim=1) / denom
            max_values = h.masked_fill(~mask.unsqueeze(-1), -1.0e9).max(dim=1).values
            max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
            attn = self.attn_pool(h, mask)
            return self.out(torch.cat([mean, max_values, attn], dim=-1))

    class EventEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(input_dim, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.attention_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
            self.attn_pool = MaskedAttentionPool(cfg.d_model)
            self.out = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))

        def forward(self, x, mask):
            h = self.input(x)
            padding_mask = ~mask
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            pooled = self.attn_pool(h, mask)
            return self.out(pooled)

    class RawDropoutModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.full_encoder = FullCurveEncoder()
            self.event_encoder = EventEncoder()
            self.embedding = nn.Sequential(
                nn.Linear(cfg.d_model * 2 + 2, cfg.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.embedding_dim),
                nn.Dropout(cfg.dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(cfg.embedding_dim, 64),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, 1),
            )

        def forward(self, full_x, full_mask, event_x, event_mask, flags, *, return_embedding: bool = False):
            z_full = self.full_encoder(full_x, full_mask)
            z_event = self.event_encoder(event_x, event_mask)
            z_raw = self.embedding(torch.cat([z_full, z_event, flags], dim=-1))
            logits = self.classifier(z_raw).squeeze(-1)
            if return_embedding:
                return logits, z_raw
            return logits

    return RawDropoutModel()


def _torch_device(config: RawTrainingConfig):
    torch = _require_torch()
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _raw_loader(
    df: pd.DataFrame,
    *,
    preprocess_config: RawPreprocessConfig,
    training_config: RawTrainingConfig,
    include_labels: bool,
    shuffle: bool,
):
    torch = _require_torch()
    dataset = RawLightCurveDataset(
        df,
        preprocess_config=preprocess_config,
        include_labels=include_labels,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        num_workers=training_config.num_workers,
        collate_fn=raw_collate,
    )


def _evaluate_raw_model(model, loader, device) -> tuple[pd.DataFrame, np.ndarray | None]:
    torch = _require_torch()
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            full_x = batch["full_x"].to(device)
            full_mask = batch["full_mask"].to(device)
            event_x = batch["event_x"].to(device)
            event_mask = batch["event_mask"].to(device)
            flags = batch["flags"].to(device)
            logits, z_raw = model(
                full_x,
                full_mask,
                event_x,
                event_mask,
                flags,
                return_embedding=True,
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            z_np = z_raw.detach().cpu().numpy()
            embeddings.append(z_np)
            labels = batch.get("label")
            label_np = labels.detach().cpu().numpy() if labels is not None else [np.nan] * len(probs)
            for idx, candidate_id in enumerate(batch["candidate_id"]):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "bad_photometry": int(label_np[idx]) if np.isfinite(label_np[idx]) else np.nan,
                        "bad_photometry_risk": float(probs[idx]),
                        "raw_available": bool(batch["raw_available"][idx]),
                        "event_window_available": bool(batch["event_window_available"][idx]),
                    }
                )
    emb = np.concatenate(embeddings, axis=0) if embeddings else None
    return pd.DataFrame(rows), emb


def train_raw_neural_model(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    split_col: str = "split",
    target_col: str = "bad_photometry",
    preprocess_config: RawPreprocessConfig | None = None,
    model_config: RawModelConfig | None = None,
    training_config: RawTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train the raw two-view PyTorch model and write artifacts."""
    torch = _require_torch()
    pp_cfg = preprocess_config or RawPreprocessConfig()
    model_cfg = model_config or RawModelConfig()
    train_cfg = training_config or RawTrainingConfig()
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = df.loc[df[split_col].eq("train")].copy()
    val = df.loc[df[split_col].eq("val")].copy()
    test = df.loc[df[split_col].eq("test")].copy()
    if train.empty or val.empty:
        raise ValueError("Train and val splits must be non-empty for raw training")
    if test.empty:
        test = val.copy()

    device = _torch_device(train_cfg)
    model = create_raw_model(config=model_cfg).to(device)
    y_train = pd.to_numeric(train[target_col], errors="raise").astype(int)
    n_bad = int((y_train == 1).sum())
    n_good = int((y_train == 0).sum())
    pos_weight = float(n_good / n_bad) if n_bad else 1.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    train_loader = _raw_loader(
        train,
        preprocess_config=pp_cfg,
        training_config=train_cfg,
        include_labels=True,
        shuffle=True,
    )
    val_loader = _raw_loader(
        val,
        preprocess_config=pp_cfg,
        training_config=train_cfg,
        include_labels=True,
        shuffle=False,
    )

    best_metric = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["full_x"].to(device),
                batch["full_mask"].to(device),
                batch["event_x"].to(device),
                batch["event_mask"].to(device),
                batch["flags"].to(device),
            )
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_predictions, _ = _evaluate_raw_model(model, val_loader, device)
        val_metric = average_precision_np(
            val_predictions["bad_photometry"].astype(int),
            val_predictions["bad_photometry_risk"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_pr_auc": val_metric,
            }
        )
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= train_cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = _raw_loader(
        test,
        preprocess_config=pp_cfg,
        training_config=train_cfg,
        include_labels=True,
        shuffle=False,
    )
    test_predictions, _ = _evaluate_raw_model(model, test_loader, device)
    test_predictions["risk_decile"] = risk_deciles(test_predictions["bad_photometry_risk"])
    test_predictions["model_name"] = "raw_neural"
    metrics = binary_metrics(
        test_predictions["bad_photometry"].astype(int),
        test_predictions["bad_photometry_risk"],
    )
    calibration = calibration_by_decile(
        test_predictions["bad_photometry"].astype(int),
        test_predictions["bad_photometry_risk"],
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "preprocess_config": asdict(pp_cfg),
            "feature_names": list(RAW_FEATURE_NAMES),
        },
        out_dir / "raw_model.pt",
    )
    _write_table(test_predictions, out_dir / "test_predictions.parquet")
    _write_table(calibration, out_dir / "calibration_by_decile.parquet")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    _write_json(
        out_dir / "metrics.json",
        {
            "model_name": "raw_neural",
            "metrics": metrics,
            "preprocess_config": asdict(pp_cfg),
            "model_config": asdict(model_cfg),
            "training_config": asdict(train_cfg),
            "best_val_pr_auc": best_metric,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "pos_weight": pos_weight,
        },
    )
    embeddings = generate_raw_embeddings(
        model,
        df,
        out_dir / "raw_embeddings.parquet",
        preprocess_config=pp_cfg,
        training_config=train_cfg,
        device=device,
    )
    return {
        "model": model,
        "model_path": out_dir / "raw_model.pt",
        "metrics": metrics,
        "predictions": test_predictions,
        "embeddings": embeddings,
    }


def generate_raw_embeddings(
    model,
    df: pd.DataFrame,
    output: str | Path | None = None,
    *,
    preprocess_config: RawPreprocessConfig | None = None,
    training_config: RawTrainingConfig | None = None,
    device: Any | None = None,
) -> pd.DataFrame:
    torch = _require_torch()
    pp_cfg = preprocess_config or RawPreprocessConfig()
    train_cfg = training_config or RawTrainingConfig()
    if device is None:
        device = _torch_device(train_cfg)
    model = model.to(device)
    loader = _raw_loader(
        df,
        preprocess_config=pp_cfg,
        training_config=train_cfg,
        include_labels="bad_photometry" in df.columns,
        shuffle=False,
    )
    predictions, embeddings = _evaluate_raw_model(model, loader, device)
    if embeddings is None:
        embeddings = np.empty((0, RawModelConfig().embedding_dim), dtype="float32")
    emb_cols = [f"{EMBEDDING_PREFIX}{idx:03d}" for idx in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    id_cols = []
    for column in ("_raw_row_id", "candidate_id", "dropout_source_id", "split", "bad_photometry"):
        if column in df.columns:
            id_cols.append(column)
    out = df[id_cols].reset_index(drop=True).copy()
    out = pd.concat([out, emb_df], axis=1)
    out["raw_available"] = predictions["raw_available"].to_numpy(dtype=bool)
    out["event_window_available"] = predictions["event_window_available"].to_numpy(dtype=bool)
    out["raw_bad_photometry_probability"] = predictions["bad_photometry_risk"].to_numpy(dtype=float)
    if output is not None:
        _write_table(out, output)
    return out


def merge_raw_embeddings(df: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    key = "_raw_row_id" if "_raw_row_id" in df.columns and "_raw_row_id" in embeddings.columns else None
    if key is None and "candidate_id" in df.columns and "candidate_id" in embeddings.columns:
        key = "candidate_id"
    if key is None and "dropout_source_id" in df.columns and "dropout_source_id" in embeddings.columns:
        key = "dropout_source_id"
    if key is None:
        raise ValueError("Could not find a shared candidate key for embeddings")
    emb_cols = [
        column
        for column in embeddings.columns
        if column.startswith(EMBEDDING_PREFIX)
        or column in {"raw_available", "event_window_available", "raw_bad_photometry_probability", key}
    ]
    return df.merge(embeddings[emb_cols], on=key, how="left", validate="many_to_one")


def _assign_stratified_group_folds(
    df: pd.DataFrame,
    *,
    n_folds: int,
    label_col: str = "bad_photometry",
    group_col: str = "dropout_source_id",
    seed: int = 42,
) -> pd.Series:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    work = df[[group_col, label_col]].copy()
    work[group_col] = work[group_col].astype(str)
    grouped = work.groupby(group_col, dropna=False)[label_col].max().reset_index()
    rng = np.random.default_rng(seed)
    fold_by_group: dict[str, int] = {}
    for label_value in sorted(grouped[label_col].unique()):
        groups = grouped.loc[grouped[label_col].eq(label_value), group_col].astype(str).to_numpy()
        rng.shuffle(groups)
        for idx, group in enumerate(groups):
            fold_by_group[str(group)] = int(idx % n_folds)
    return df[group_col].astype(str).map(fold_by_group).astype(int)


def generate_oof_raw_embeddings(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    final_embeddings: pd.DataFrame,
    n_folds: int = 5,
    split_col: str = "split",
    preprocess_config: RawPreprocessConfig | None = None,
    model_config: RawModelConfig | None = None,
    training_config: RawTrainingConfig | None = None,
) -> pd.DataFrame:
    """Generate out-of-fold raw embeddings for training rows.

    Validation/test rows keep embeddings from the final raw model trained on the
    full training split. Training rows are replaced with embeddings from fold
    models that did not train on that row.
    """
    if n_folds < 2:
        return final_embeddings.copy()
    if "_raw_row_id" not in df.columns:
        raise ValueError("df must include _raw_row_id before generating OOF embeddings")
    if "_raw_row_id" not in final_embeddings.columns:
        raise ValueError("final_embeddings must include _raw_row_id")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pp_cfg = preprocess_config or RawPreprocessConfig()
    model_cfg = model_config or RawModelConfig()
    train_cfg = training_config or RawTrainingConfig()

    train_rows = df.loc[df[split_col].eq("train")].copy()
    if train_rows.empty:
        return final_embeddings.copy()
    group_labels = train_rows.groupby("dropout_source_id", dropna=False)["bad_photometry"].max()
    label_counts = group_labels.value_counts()
    if len(label_counts) < 2 or int(label_counts.min()) < 2:
        return final_embeddings.copy()
    n_folds = min(int(n_folds), int(label_counts.min()))
    folds = _assign_stratified_group_folds(
        train_rows,
        n_folds=n_folds,
        seed=train_cfg.seed,
    )
    train_rows["_oof_fold"] = folds.to_numpy(dtype=int)
    oof_parts: list[pd.DataFrame] = []
    for fold in range(n_folds):
        fold_table = train_rows.copy()
        fold_table[split_col] = np.where(fold_table["_oof_fold"].eq(fold), "val", "train")
        if fold_table[split_col].eq("train").sum() == 0 or fold_table[split_col].eq("val").sum() == 0:
            continue
        fold_result = train_raw_neural_model(
            fold_table.drop(columns=["_oof_fold"]),
            out_dir / f"fold_{fold:02d}",
            split_col=split_col,
            preprocess_config=pp_cfg,
            model_config=model_cfg,
            training_config=train_cfg,
        )
        fold_embeddings = fold_result["embeddings"]
        val_ids = set(fold_table.loc[fold_table[split_col].eq("val"), "_raw_row_id"].astype(int))
        oof_parts.append(fold_embeddings.loc[fold_embeddings["_raw_row_id"].astype(int).isin(val_ids)].copy())

    if not oof_parts:
        return final_embeddings.copy()
    oof_train = pd.concat(oof_parts, ignore_index=True)
    non_train = final_embeddings.loc[
        ~final_embeddings["_raw_row_id"].astype(int).isin(set(train_rows["_raw_row_id"].astype(int)))
    ].copy()
    combined = pd.concat([oof_train, non_train], ignore_index=True)
    combined = combined.drop_duplicates("_raw_row_id", keep="first")
    missing_ids = set(final_embeddings["_raw_row_id"].astype(int)) - set(combined["_raw_row_id"].astype(int))
    if missing_ids:
        fallback = final_embeddings.loc[final_embeddings["_raw_row_id"].astype(int).isin(missing_ids)].copy()
        combined = pd.concat([combined, fallback], ignore_index=True)
    combined = combined.sort_values("_raw_row_id").reset_index(drop=True)
    _write_table(combined, out_dir / "raw_embeddings_oof.parquet")
    return combined


def train_all_models(
    dataset: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    split_col: str = "split",
    make_split_if_missing: bool = True,
    tabular_config: TabularModelConfig | None = None,
    raw_preprocess_config: RawPreprocessConfig | None = None,
    raw_model_config: RawModelConfig | None = None,
    raw_training_config: RawTrainingConfig | None = None,
    train_raw: bool = True,
    oof_folds: int = 5,
) -> dict[str, Any]:
    """Train derived-only, raw-only, and hybrid models on one split."""
    df = _read_table(dataset) if not isinstance(dataset, pd.DataFrame) else dataset.copy()
    if "_raw_row_id" not in df.columns:
        df["_raw_row_id"] = np.arange(len(df), dtype=int)
    if split_col not in df.columns:
        if not make_split_if_missing:
            raise ValueError(f"Missing split column: {split_col}")
        df = make_stratified_group_split(df, split_col=split_col)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_table(df, out_dir / "dataset_with_split.parquet")

    results: dict[str, Any] = {}
    results["derived"] = train_lightgbm_classifier(
        df,
        out_dir / "derived_only",
        model_name="derived_only",
        split_col=split_col,
        config=tabular_config,
    )
    if train_raw:
        results["raw"] = train_raw_neural_model(
            df,
            out_dir / "raw_neural",
            split_col=split_col,
            preprocess_config=raw_preprocess_config,
            model_config=raw_model_config,
            training_config=raw_training_config,
        )
        embeddings = results["raw"]["embeddings"]
        if oof_folds >= 2:
            embeddings = generate_oof_raw_embeddings(
                df,
                out_dir / "raw_neural_oof",
                final_embeddings=embeddings,
                n_folds=oof_folds,
                split_col=split_col,
                preprocess_config=raw_preprocess_config,
                model_config=raw_model_config,
                training_config=raw_training_config,
            )
        hybrid_df = merge_raw_embeddings(df, embeddings)
        _write_table(hybrid_df, out_dir / "hybrid_training_table.parquet")
        results["hybrid"] = train_lightgbm_classifier(
            hybrid_df,
            out_dir / "hybrid",
            model_name="hybrid",
            split_col=split_col,
            config=tabular_config,
        )
    comparison_rows = []
    for name, result in results.items():
        if not isinstance(result, Mapping) or "metrics" not in result:
            continue
        row = {"model": name}
        row.update(result["metrics"])
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    return results


def score_lightgbm_bundle(
    model_path: str | Path,
    candidates: str | Path | pd.DataFrame,
    output: str | Path | None = None,
) -> pd.DataFrame:
    bundle = _load_pickle(model_path)
    df = _read_table(candidates) if not isinstance(candidates, pd.DataFrame) else candidates.copy()
    encoder: TabularFeatureEncoder = bundle["encoder"]
    X = encoder.transform(df)
    raw = bundle["model"].predict_proba(X)[:, 1]
    scores = bundle["calibrator"].predict(raw)
    out = df.copy()
    out["bad_photometry_risk"] = scores
    out["risk_decile"] = risk_deciles(scores)
    out["model_name"] = bundle.get("model_name", "lightgbm")
    if output is not None:
        _write_table(out, output)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca bad-photometry",
        description="Train and apply MALCA bad-photometry dropout models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="Build v1/v2 dropout label table")
    build.add_argument("--v1", required=True, type=Path, help="Original v1 candidate table")
    build.add_argument("--v2", required=True, type=Path, help="Surviving v2 candidate table")
    build.add_argument("--output", required=True, type=Path, help="Output labeled table")
    build.add_argument("--key", default=None, help="Candidate ID column in v1 and v2 when shared")
    build.add_argument("--v2-key", default=None, help="Candidate ID column in v2 when different")
    build.add_argument("--flat-lightcurve-dir", default=None, type=Path)
    build.add_argument("--allow-v2-outside-v1", action="store_true")

    split = subparsers.add_parser("make-split", help="Add stratified random train/val/test split")
    split.add_argument("--dataset", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", default=42, type=int)

    train = subparsers.add_parser("train", help="Train derived/raw/hybrid models")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--raw-max-epochs", default=50, type=int)
    train.add_argument("--raw-batch-size", default=64, type=int)
    train.add_argument("--oof-folds", default=5, type=int, help="OOF folds for hybrid raw embeddings")
    train.add_argument("--skip-raw", action="store_true", help="Train derived-only model only")

    score = subparsers.add_parser("score-lightgbm", help="Score candidates with derived/hybrid LightGBM bundle")
    score.add_argument("--model", required=True, type=Path, help="Path to model.joblib")
    score.add_argument("--candidates", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        df = build_dropout_dataset(
            args.v1,
            args.v2,
            output=args.output,
            key=args.key,
            v2_key=args.v2_key,
            require_v2_subset=not args.allow_v2_outside_v1,
            flat_lightcurve_dir=args.flat_lightcurve_dir,
        )
        print(f"Wrote {len(df)} labeled candidates to {args.output}")
        print(f"Bad-photometry/dropout rows: {int(df['bad_photometry'].sum())}")
        return 0
    if args.command == "make-split":
        df = _read_table(args.dataset)
        out = make_stratified_group_split(df, output=args.output, seed=args.seed)
        counts = out.groupby(["split", "bad_photometry"]).size().reset_index(name="n")
        print(f"Wrote split table to {args.output}")
        print(counts.to_string(index=False))
        return 0
    if args.command == "train":
        raw_training = RawTrainingConfig(
            max_epochs=args.raw_max_epochs,
            batch_size=args.raw_batch_size,
        )
        results = train_all_models(
            args.dataset,
            args.output_dir,
            raw_training_config=raw_training,
            train_raw=not args.skip_raw,
            oof_folds=args.oof_folds,
        )
        print(f"Wrote model artifacts to {args.output_dir}")
        for name, result in results.items():
            if isinstance(result, Mapping) and "metrics" in result:
                metrics = result["metrics"]
                print(f"{name}: PR-AUC={metrics.get('pr_auc')}, ROC-AUC={metrics.get('roc_auc')}")
        return 0
    if args.command == "score-lightgbm":
        out = score_lightgbm_bundle(args.model, args.candidates, output=args.output)
        print(f"Wrote {len(out)} scored candidates to {args.output}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
