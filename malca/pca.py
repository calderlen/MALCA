"""
1. Normalize each metric by its expected center and scatter vs magnitude
2. Impute missing normalized values
3. Standardize to zero mean and unit variance
4. Project onto principal components
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# User-requested core metrics.
BASE_VARIABILITY_PCA_COLUMNS: list[str] = [
    "stats_variability_stetson_J_time",
    "stats_variability_stetson_I",
    "stats_variability_stetson_K",
    "stats_variability_stetson_L_time",
    "stats_variability_von_neumann_ratio",
    "stats_median_abs_dev",
    "stats_photometry_IQR_mag",
    "stats_variability_string_length_resid_total",
    "stats_variability_reduced_chi2_vs_constant",
    "stats_anderson_darling",
]

# Recommended additions: complementary variability structure metrics that are
# not algebraically derived from the requested Stetson family.
RECOMMENDED_EXTRA_VARIABILITY_PCA_COLUMNS: list[str] = [
    "stats_variability_roms",
    "stats_variability_lag1_autocorr",
]

DEFAULT_VARIABILITY_PCA_COLUMNS: list[str] = (
    BASE_VARIABILITY_PCA_COLUMNS + RECOMMENDED_EXTRA_VARIABILITY_PCA_COLUMNS
)
DEFAULT_MAGNITUDE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "baseline_mag",
    "stats_photometry_median_mag",
    "stats_photometry_mean_mag",
)
DEFAULT_MAGNITUDE_NORMALIZED_SUFFIX = "_magz"
DEFAULT_PCA_PREFIX = "stats_variability_pc"


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _robust_scale(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > 0:
        return scale
    if vals.size >= 2:
        scale = float(np.std(vals, ddof=1))
        if np.isfinite(scale) and scale > 0:
            return scale
    return 1.0


def infer_magnitude_column(
    df: pd.DataFrame,
    *,
    preferred: str | None = None,
    candidates: Sequence[str] = DEFAULT_MAGNITUDE_COLUMN_CANDIDATES,
) -> str:
    """Infer which magnitude column should drive the normalization."""
    ordered = []
    if preferred:
        ordered.append(str(preferred))
    ordered.extend(str(col) for col in candidates)
    for col in _unique_preserve_order(ordered):
        if col in df.columns:
            return col
    raise ValueError(
        "Could not infer a magnitude column. "
        f"Tried: {', '.join(_unique_preserve_order(ordered))}"
    )


def resolve_feature_columns(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (present_features, missing_features) in stable order."""
    requested = _unique_preserve_order(feature_columns or DEFAULT_VARIABILITY_PCA_COLUMNS)
    present = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    return present, missing


@dataclass
class MagnitudeNormalizationCurve:
    """Per-feature center/scale trend versus magnitude."""

    feature: str
    mag_points: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    global_center: float
    global_scale: float
    n_valid: int
    bin_width: float
    min_bin_count: int


@dataclass
class VariabilityPCAModel:
    """Fitted population-level transform for variability-metric PCA."""

    mag_column: str
    feature_columns: list[str]
    normalized_columns: list[str]
    pca_columns: list[str]
    curves: dict[str, MagnitudeNormalizationCurve]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    metadata: dict[str, Any] = field(default_factory=dict)


def _fit_curve_for_feature(
    mag: np.ndarray,
    values: np.ndarray,
    *,
    feature: str,
    bin_width: float,
    min_bin_count: int,
) -> MagnitudeNormalizationCurve | None:
    mask = np.isfinite(mag) & np.isfinite(values)
    if not mask.any():
        return None

    mag_valid = mag[mask]
    values_valid = values[mask]
    global_center = float(np.median(values_valid))
    global_scale = _robust_scale(values_valid)

    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be > 0")

    mag_min = float(np.min(mag_valid))
    mag_max = float(np.max(mag_valid))
    edge_start = np.floor(mag_min / bin_width) * bin_width
    edge_stop = np.ceil(mag_max / bin_width) * bin_width
    if not np.isfinite(edge_start) or not np.isfinite(edge_stop):
        edge_start = mag_min
        edge_stop = mag_max
    if edge_stop <= edge_start:
        edge_stop = edge_start + float(bin_width)

    edges = np.arange(edge_start, edge_stop + bin_width, bin_width, dtype=float)
    if edges.size < 2:
        edges = np.array([edge_start, edge_start + bin_width], dtype=float)

    mag_points: list[float] = []
    centers: list[float] = []
    scales: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (mag_valid >= left) & (mag_valid < right)
        if right == edges[-1]:
            in_bin = (mag_valid >= left) & (mag_valid <= right)
        if int(np.sum(in_bin)) < int(min_bin_count):
            continue
        subset = values_valid[in_bin]
        mag_points.append(float(0.5 * (left + right)))
        centers.append(float(np.median(subset)))
        scales.append(float(_robust_scale(subset)))

    if not mag_points:
        midpoint = float(np.median(mag_valid))
        mag_points = [midpoint - 0.5 * bin_width, midpoint + 0.5 * bin_width]
        centers = [global_center, global_center]
        scales = [global_scale, global_scale]
    elif len(mag_points) == 1:
        midpoint = mag_points[0]
        mag_points = [midpoint - 0.5 * bin_width, midpoint + 0.5 * bin_width]
        centers = [centers[0], centers[0]]
        scales = [scales[0], scales[0]]

    return MagnitudeNormalizationCurve(
        feature=feature,
        mag_points=np.asarray(mag_points, dtype=float),
        centers=np.asarray(centers, dtype=float),
        scales=np.asarray(scales, dtype=float),
        global_center=float(global_center),
        global_scale=float(global_scale),
        n_valid=int(mask.sum()),
        bin_width=float(bin_width),
        min_bin_count=int(min_bin_count),
    )


def fit_magnitude_normalizer(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    mag_column: str | None = None,
    bin_width: float = 0.5,
    min_bin_count: int = 50,
) -> tuple[str, dict[str, MagnitudeNormalizationCurve], dict[str, Any]]:
    """Fit magnitude-conditioned center/scale trends for each feature."""
    mag_col = infer_magnitude_column(df, preferred=mag_column)
    present, missing = resolve_feature_columns(df, feature_columns=feature_columns)
    if not present:
        raise ValueError("No PCA feature columns were found in the input DataFrame.")

    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float)
    curves: dict[str, MagnitudeNormalizationCurve] = {}
    skipped: list[str] = []
    for feature in present:
        values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        curve = _fit_curve_for_feature(
            mag,
            values,
            feature=feature,
            bin_width=bin_width,
            min_bin_count=min_bin_count,
        )
        if curve is None:
            skipped.append(feature)
            continue
        curves[feature] = curve

    if len(curves) < 2:
        raise ValueError(
            "Need at least 2 fitted feature curves for PCA. "
            f"Found {len(curves)} usable features."
        )

    meta = {
        "requested_features": _unique_preserve_order(feature_columns or DEFAULT_VARIABILITY_PCA_COLUMNS),
        "missing_features_at_fit": missing,
        "skipped_features_at_fit": skipped,
        "bin_width": float(bin_width),
        "min_bin_count": int(min_bin_count),
    }
    return mag_col, curves, meta


def _interp_curve(
    mag_values: np.ndarray,
    *,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    default: float,
) -> np.ndarray:
    out = np.full(mag_values.shape, float(default), dtype=float)
    finite = np.isfinite(mag_values)
    if not finite.any():
        return out
    out[finite] = np.interp(
        mag_values[finite],
        np.asarray(curve_x, dtype=float),
        np.asarray(curve_y, dtype=float),
        left=float(curve_y[0]),
        right=float(curve_y[-1]),
    )
    return out


def apply_magnitude_normalizer(
    df: pd.DataFrame,
    curves: dict[str, MagnitudeNormalizationCurve],
    *,
    mag_column: str,
    suffix: str = DEFAULT_MAGNITUDE_NORMALIZED_SUFFIX,
) -> pd.DataFrame:
    """Return magnitude-normalized feature columns using fitted curves."""
    if mag_column not in df.columns:
        mag_values = np.full(len(df), np.nan, dtype=float)
    else:
        mag_values = pd.to_numeric(df[mag_column], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)
    for feature, curve in curves.items():
        raw = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float) if feature in df.columns else np.full(len(df), np.nan, dtype=float)
        centers = _interp_curve(
            mag_values,
            curve_x=curve.mag_points,
            curve_y=curve.centers,
            default=curve.global_center,
        )
        scales = _interp_curve(
            mag_values,
            curve_x=curve.mag_points,
            curve_y=curve.scales,
            default=curve.global_scale,
        )
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, curve.global_scale)
        values = np.full(len(df), np.nan, dtype=float)
        finite_raw = np.isfinite(raw)
        values[finite_raw] = (raw[finite_raw] - centers[finite_raw]) / scales[finite_raw]
        out[f"{feature}{suffix}"] = values
    return out


def build_standardized_feature_matrix(
    df: pd.DataFrame,
    model: VariabilityPCAModel,
) -> pd.DataFrame:
    """Transform *df* into the standardized matrix consumed by PCA."""
    normalized = apply_magnitude_normalizer(
        df,
        model.curves,
        mag_column=model.mag_column,
        suffix=DEFAULT_MAGNITUDE_NORMALIZED_SUFFIX,
    )
    normalized = normalized.reindex(columns=model.normalized_columns)
    matrix = model.imputer.transform(normalized)
    matrix = model.scaler.transform(matrix)
    return pd.DataFrame(matrix, index=df.index, columns=model.feature_columns)


def fit_variability_pca(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    mag_column: str | None = None,
    n_components: int | float | str | None = 2,
    bin_width: float = 0.5,
    min_bin_count: int = 50,
    impute_strategy: str = "median",
    svd_solver: str = "auto",
    random_state: int | None = 42,
) -> VariabilityPCAModel:
    """Fit the full magnitude-normalization + PCA pipeline."""
    mag_col, curves, meta = fit_magnitude_normalizer(
        df,
        feature_columns=feature_columns,
        mag_column=mag_column,
        bin_width=bin_width,
        min_bin_count=min_bin_count,
    )

    fitted_features = list(curves.keys())
    normalized_columns = [
        f"{feature}{DEFAULT_MAGNITUDE_NORMALIZED_SUFFIX}" for feature in fitted_features
    ]
    normalized = apply_magnitude_normalizer(
        df,
        curves,
        mag_column=mag_col,
        suffix=DEFAULT_MAGNITUDE_NORMALIZED_SUFFIX,
    )
    normalized = normalized.reindex(columns=normalized_columns)

    imputer = SimpleImputer(strategy=impute_strategy)
    matrix = imputer.fit_transform(normalized)

    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)

    pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    pca.fit(matrix)

    n_components_fit = int(pca.components_.shape[0])
    pca_columns = [f"{DEFAULT_PCA_PREFIX}{idx + 1}" for idx in range(n_components_fit)]
    meta["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    return VariabilityPCAModel(
        mag_column=mag_col,
        feature_columns=fitted_features,
        normalized_columns=normalized_columns,
        pca_columns=pca_columns,
        curves=curves,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        metadata=meta,
    )


def apply_variability_pca(
    df: pd.DataFrame,
    model: VariabilityPCAModel,
    *,
    include_mag_normalized: bool = True,
) -> pd.DataFrame:
    """Apply a fitted variability PCA model and append derived columns."""
    out = df.copy()
    normalized = apply_magnitude_normalizer(
        df,
        model.curves,
        mag_column=model.mag_column,
        suffix=DEFAULT_MAGNITUDE_NORMALIZED_SUFFIX,
    )
    normalized = normalized.reindex(columns=model.normalized_columns)

    if include_mag_normalized:
        for col in normalized.columns:
            out[col] = normalized[col].astype(float)

    matrix = model.imputer.transform(normalized)
    matrix = model.scaler.transform(matrix)
    pcs = model.pca.transform(matrix)
    for idx, col in enumerate(model.pca_columns):
        out[col] = pcs[:, idx].astype(float)

    return out


def fit_apply_variability_pca(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    mag_column: str | None = None,
    n_components: int | float | str | None = 2,
    bin_width: float = 0.5,
    min_bin_count: int = 50,
    impute_strategy: str = "median",
    svd_solver: str = "auto",
    random_state: int | None = 42,
    include_mag_normalized: bool = True,
) -> tuple[pd.DataFrame, VariabilityPCAModel]:
    """Fit and immediately apply the variability PCA pipeline."""
    model = fit_variability_pca(
        df,
        feature_columns=feature_columns,
        mag_column=mag_column,
        n_components=n_components,
        bin_width=bin_width,
        min_bin_count=min_bin_count,
        impute_strategy=impute_strategy,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    out = apply_variability_pca(
        df,
        model,
        include_mag_normalized=include_mag_normalized,
    )
    return out, model


def summarize_pca_model(model: VariabilityPCAModel) -> dict[str, Any]:
    """Return a JSON-serializable summary of the fitted PCA model."""
    return {
        "mag_column": model.mag_column,
        "feature_columns": list(model.feature_columns),
        "normalized_columns": list(model.normalized_columns),
        "pca_columns": list(model.pca_columns),
        "explained_variance_ratio": model.pca.explained_variance_ratio_.tolist(),
        **dict(model.metadata),
    }


def save_pca_model(model: VariabilityPCAModel, path: str | Path) -> None:
    """Serialize the fitted PCA model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_pca_model(path: str | Path) -> VariabilityPCAModel:
    """Load a serialized PCA model."""
    return joblib.load(Path(path))


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        out = pd.read_parquet(path)
        return out.to_frame() if isinstance(out, pd.Series) else out
    if path.suffix.lower() == ".csv":
        out = pd.read_csv(path)
        return out.to_frame() if isinstance(out, pd.Series) else out
    raise ValueError("Unsupported input file type. Use CSV or Parquet.")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError("Unsupported output file type. Use CSV or Parquet.")


def _feature_args_to_list(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    out: list[str] = []
    for value in raw:
        parts = [part.strip() for part in str(value).split(",")]
        out.extend(part for part in parts if part)
    return _unique_preserve_order(out)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet table")
    parser.add_argument(
        "--feature",
        action="append",
        help=(
            "Feature column to include. Repeat or pass a comma-separated list. "
            "Defaults to the built-in variability feature set."
        ),
    )
    parser.add_argument("--mag-col", type=str, default=None, help="Magnitude column for trend fitting")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Magnitude-bin width for trend fitting")
    parser.add_argument("--min-bin-count", type=int, default=50, help="Minimum rows per magnitude bin")
    parser.add_argument("--impute-strategy", type=str, default="median", choices=["mean", "median", "most_frequent", "constant"])
    parser.add_argument("--svd-solver", type=str, default="auto", choices=["auto", "full", "covariance_eigh", "arpack", "randomized"])
    parser.add_argument("--n-components", type=float, default=2, help="PCA n_components; integer counts are accepted")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PCA solvers that use one")


def _parse_n_components(raw: float) -> int | float:
    as_int = int(raw)
    if float(raw) == float(as_int):
        return as_int
    return float(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit/apply magnitude-normalized PCA on variability metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_fit = subparsers.add_parser("fit", help="Fit a PCA model and save it")
    _add_common_args(p_fit)
    p_fit.add_argument("--model-out", type=Path, required=True, help="Output .joblib path")
    p_fit.add_argument("--summary-out", type=Path, default=None, help="Optional JSON summary output")

    p_apply = subparsers.add_parser("apply", help="Apply a saved PCA model to a table")
    p_apply.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet table")
    p_apply.add_argument("--model-in", type=Path, required=True, help="Input .joblib model path")
    p_apply.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet table")
    p_apply.add_argument("--no-magz", action="store_true", help="Do not append intermediate *_magz columns")

    p_fit_apply = subparsers.add_parser("fit-apply", help="Fit a PCA model and immediately apply it")
    _add_common_args(p_fit_apply)
    p_fit_apply.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet table")
    p_fit_apply.add_argument("--model-out", type=Path, default=None, help="Optional .joblib output path")
    p_fit_apply.add_argument("--summary-out", type=Path, default=None, help="Optional JSON summary output")
    p_fit_apply.add_argument("--no-magz", action="store_true", help="Do not append intermediate *_magz columns")

    args = parser.parse_args()

    if args.command == "apply":
        df = _load_table(args.input)
        model = load_pca_model(args.model_in)
        out = apply_variability_pca(df, model, include_mag_normalized=not args.no_magz)
        _write_table(out, args.output)
        return

    df = _load_table(args.input)
    feature_columns = _feature_args_to_list(args.feature)
    n_components = _parse_n_components(args.n_components)
    model = fit_variability_pca(
        df,
        feature_columns=feature_columns,
        mag_column=args.mag_col,
        n_components=n_components,
        bin_width=args.bin_width,
        min_bin_count=args.min_bin_count,
        impute_strategy=args.impute_strategy,
        svd_solver=args.svd_solver,
        random_state=args.seed,
    )

    if args.command == "fit":
        save_pca_model(model, args.model_out)
        if args.summary_out is not None:
            args.summary_out.write_text(json.dumps(summarize_pca_model(model), indent=2, sort_keys=True), encoding="utf-8")
        return

    out = apply_variability_pca(df, model, include_mag_normalized=not args.no_magz)
    _write_table(out, args.output)
    if args.model_out is not None:
        save_pca_model(model, args.model_out)
    if args.summary_out is not None:
        args.summary_out.write_text(json.dumps(summarize_pca_model(model), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
