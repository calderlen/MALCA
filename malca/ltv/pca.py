"""
LTV PCA: unsupervised PCA on LTV statistics.

Fits PCA (impute → standardize → PCA) on numeric LTV columns and adds
ltv_pc1, ltv_pc2, ... for exploration, visualization, and optional scoring.
No labels required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Columns to use for PCA when present (core + optional pipeline/stochastic).
# Only columns present in df and numeric are used.
LTV_PCA_FEATURE_CANDIDATES: tuple[str, ...] = (
    # Core
    "Slope",
    "Quad Slope",
    "Median",
    "Median_err",
    "Dispersion",
    "max diff",
    "n_points",
    "time_span_days",
    "n_unique_nights",
    "trend_slope_mag_per_year",
    "trend_quad_mag_per_year2",
    "trend_slope_err_mag_per_year",
    "trend_slope_snr",
    "trend_r2",
    "trend_delta_bic_linear",
    "trend_delta_bic_quadratic",
    "inverse_von_neumann_ratio",
    "reduced_chi2_vs_constant",
    "roms",
    "n_seasons",
    "season_points_min",
    "season_points_median",
    "season_points_max",
    "season_span_days_mean",
    "season_span_days_median",
    "season_span_days_max",
    "season_step_max_mag",
    "season_step_mean_abs_mag",
    "season_step_max_fraction",
    "season_monotonicity_fraction",
    "season_spearman_rho",
    "season_kendall_tau",
    "leave1out_slope_std",
    "leave1out_slope_range",
    "ls_period",
    "ls_power",
    "ls_fap",
    "vg_overlap_days",
    "vg_overlap_fraction",
    # Optional pipeline
    "w1_slope",
    "w1_w2_slope",
    "neowise_n_epochs",
    "stoch_sf_ml_amplitude",
    "stoch_sf_ml_gamma",
    "stoch_iar_phi",
    "stoch_mhps_high",
    "stoch_mhps_low",
    "stoch_mhps_ratio",
    "stoch_gp_drw_sigma",
    "stoch_gp_drw_tau",
)

LTV_PCA_PREFIX = "ltv_pc"


def resolve_feature_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> list[str]:
    """Return list of numeric columns from *df* suitable for PCA.

    If *feature_columns* is ``None``, intersect ``LTV_PCA_FEATURE_CANDIDATES``
    with the columns actually present in *df* and keep only numeric ones.
    """
    if feature_columns is not None:
        candidates = list(feature_columns)
    else:
        candidates = list(LTV_PCA_FEATURE_CANDIDATES)
    return [
        c for c in candidates
        if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]


def coerce_n_components(raw: int | float, n_samples: int, n_features: int) -> int | float:
    """Normalise *raw* ``n_components`` for sklearn ``PCA``.

    * Whole-number floats (e.g. ``10.0``) are converted to ``int``.
    * Integer counts are clamped to ``min(n_samples, n_features)`` so
      sklearn never raises.
    * Fractional values in ``(0, 1)`` are passed through as a variance
      fraction — sklearn handles them natively.
    """
    if isinstance(raw, float) and raw == int(raw):
        raw = int(raw)
    if isinstance(raw, int):
        upper = min(n_samples, n_features)
        return max(1, min(raw, upper))
    return raw


@dataclass
class LtvPCAModel:
    """Fitted LTV PCA model (imputer + scaler + PCA)."""

    feature_columns: list[str]
    pca_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    explained_variance_ratio_: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def fit_ltv_pca(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    n_components: int | float = 10,
    impute_strategy: str = "median",
    svd_solver: str = "auto",
    random_state: int | None = 42,
) -> LtvPCAModel:
    """Fit imputer, scaler, and PCA on numeric LTV stats.

    *n_components* can be an integer count or a float in ``(0, 1)`` for a
    variance-fraction target.  Integer counts are automatically clamped to
    ``min(n_samples, n_features)`` so the call never fails when the table
    is smaller than the requested dimensionality.
    """
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to fit PCA.")

    resolved = resolve_feature_columns(df, feature_columns)
    if len(resolved) < 2:
        raise ValueError(
            f"Need at least 2 numeric LTV feature columns for PCA; found {len(resolved)}."
        )

    X = df[resolved].astype(float)
    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    safe_nc = coerce_n_components(n_components, n_samples=len(df), n_features=len(resolved))

    pca = PCA(
        n_components=safe_nc,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    pca.fit(X_scaled)

    n_comp = int(pca.components_.shape[0])
    pca_cols = [f"{LTV_PCA_PREFIX}{i + 1}" for i in range(n_comp)]
    evr = pca.explained_variance_ratio_.tolist()

    return LtvPCAModel(
        feature_columns=resolved,
        pca_columns=pca_cols,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        explained_variance_ratio_=evr,
        metadata={
            "n_samples_fit": len(df),
            "n_features": len(resolved),
            "n_components": n_comp,
            "n_components_requested": n_components,
        },
    )


def apply_ltv_pca(df: pd.DataFrame, model: LtvPCAModel) -> pd.DataFrame:
    """Apply a fitted LTV PCA model and add ltv_pc1, ltv_pc2, ... columns."""
    out = df.copy()
    for c in model.feature_columns:
        if c not in out.columns:
            out[c] = np.nan
    X = out[model.feature_columns].astype(float)
    X_imputed = model.imputer.transform(X)
    X_scaled = model.scaler.transform(X_imputed)
    pcs = model.pca.transform(X_scaled)
    for i, col in enumerate(model.pca_columns):
        out[col] = pcs[:, i].astype(float)
    return out


def fit_apply_ltv_pca(
    df: pd.DataFrame,
    **kwargs,
) -> tuple[pd.DataFrame, LtvPCAModel]:
    """Fit PCA on df and apply it; return (df with PC columns, model)."""
    model = fit_ltv_pca(df, **kwargs)
    out = apply_ltv_pca(df, model)
    return out, model


def save_ltv_pca_model(model: LtvPCAModel, path: str | Path) -> None:
    """Save fitted model to path (joblib)."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_ltv_pca_model(path: str | Path) -> LtvPCAModel:
    """Load fitted model from path (joblib)."""
    import joblib
    return joblib.load(Path(path))


def _summarize_ltv_pca_model(model: LtvPCAModel) -> dict:
    """Summary dict for JSON export (explained_variance_ratio, feature_columns, etc.)."""
    return {
        "feature_columns": model.feature_columns,
        "pca_columns": model.pca_columns,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "cumulative_variance": sum(model.explained_variance_ratio_),
        **model.metadata,
    }


# -----------------------------------------------------------------------------
# Standalone CLI
# -----------------------------------------------------------------------------

def _load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Fit/apply LTV PCA on numeric LTV statistics (adds ltv_pc1, ltv_pc2, ...)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_fit_apply = subparsers.add_parser("fit-apply", help="Fit PCA and apply to the same table")
    p_fit_apply.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet (LTV pipeline output)")
    p_fit_apply.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet with ltv_pc1, ltv_pc2, ...")
    p_fit_apply.add_argument("--n-components", type=float, default=10, metavar="N", help="Number of components or variance fraction (e.g. 0.95). Default: 10")
    p_fit_apply.add_argument("--model-out", type=Path, default=None, help="Optional path to save fitted model (.joblib)")
    p_fit_apply.add_argument("--summary-out", type=Path, default=None, help="Optional JSON path with explained_variance_ratio, feature_columns")

    p_apply = subparsers.add_parser("apply", help="Apply a saved LTV PCA model to a table")
    p_apply.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet table")
    p_apply.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet with PC columns")
    p_apply.add_argument("--model", type=Path, required=True, help="Path to fitted model (.joblib)")

    args = parser.parse_args()

    if args.command == "apply":
        df = _load_table(args.input)
        model = load_ltv_pca_model(args.model)
        out = apply_ltv_pca(df, model)
        _write_table(out, args.output)
        return

    # fit-apply
    df = _load_table(args.input)
    if len(df) < 2:
        raise SystemExit("Need at least 2 rows to fit LTV PCA.")
    feats = resolve_feature_columns(df)
    if len(feats) < 2:
        raise SystemExit(f"Need at least 2 numeric LTV feature columns; found {len(feats)}.")

    # coerce_n_components handles float→int and clamping
    nc = coerce_n_components(args.n_components, n_samples=len(df), n_features=len(feats))
    out, model = fit_apply_ltv_pca(df, n_components=nc)
    _write_table(out, args.output)
    print(
        f"Wrote {len(out):,} rows x {len(model.pca_columns)} PC columns "
        f"(cumulative variance: {sum(model.explained_variance_ratio_):.3f})"
    )
    if args.model_out is not None:
        save_ltv_pca_model(model, args.model_out)
        print(f"Saved model to {args.model_out}")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(_summarize_ltv_pca_model(model), indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
