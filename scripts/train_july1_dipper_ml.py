"""Train and score July 1 dipper triage models.

This is intentionally a research script rather than a new MALCA CLI command.
It can train three binary LightGBM feature sets:

``stats_only``
    Uses only ``stats_*`` columns from the review DB.
``stats_only_early_stopping``
    Uses the same feature definition as ``stats_only`` but writes a separate
    artifact so the historical fixed-tree model is not overwritten.
``stats_plus_periodicity_dip_jump``
    Adds curated native periodicity, dip, jump, recovery-bounded morphology,
    and astrophysical-context measurements to usable ``stats_*`` columns.
``stats_plus_periodicity_dip_jump_external_context``
    Compatibility name for the same expanded context-enabled model.
``full``
    Uses the broader non-label candidate feature set.

Each model targets reviewed dipper-like candidates versus reviewed non-dippers,
then scores every candidate in the July 1 review DB. The default run trains the
curated expanded model with fold-local early stopping.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from malca.meta_analysis.ml.candidate_features import (
    NEXT_ITERATION_CONTEXT_FEATURES,
    RECOVERY_BOUNDED_EVENT_FEATURES,
    RECOVERY_FEATURE_SCHEMA_VERSION,
    add_next_iteration_context_features,
    add_recovery_bounded_event_features,
    default_recovery_feature_cache,
)
from malca.meta_analysis.ml.feature_policy import (
    MODEL_FEATURE_EXCLUSION_COLUMNS,
)
from malca.meta_analysis.ml.review_lightgbm import (
    ASTROPHYSICAL_CONTEXT_FEATURES,
    DEFAULT_DROP_COLUMNS,
    TrainingConfig,
    add_astrophysical_context_features,
    score_target_model,
    save_target_model,
    train_target_model,
)
from malca.review.dipper_recurrence import add_observed_dipper_recurrence


DEFAULT_RUN_DIR = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_DB_PATH = DEFAULT_RUN_DIR / "review" / "review.db"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "results" / "dipper_feature_selection"

TARGET_COLUMN = "dipper_like_label"
POSITIVE_LABEL = "dipper_like"
NEGATIVE_LABEL = "not_dipper"

POSITIVE_EVENT_CLASSES = {"dipper", "mixed_dip_and_burst"}
POSITIVE_MORPHOLOGIES = {"dimming_event", "mixed_dip_and_burst"}
EXCLUDED_EVENT_CLASSES = {"", "unclassified"}

WINDOW_COVERAGES = (1.0, 0.95, 0.90, 0.80)
SKIP_DB_COLUMNS = {
    "payload_json",
    "legacy_review_json",
    "notes",
}

STATS_ONLY_FEATURE_SETS = {"stats_only", "stats_only_early_stopping"}
EXPANDED_FEATURE_SET = "stats_plus_periodicity_dip_jump"
EXTERNAL_CONTEXT_FEATURE_SET = "stats_plus_periodicity_dip_jump_external_context"
CURATED_LC_FEATURE_SETS = {EXPANDED_FEATURE_SET, EXTERNAL_CONTEXT_FEATURE_SET}

ADDITIONAL_PERIODICITY_FEATURES = (
    "periodicity_period",
    "periodicity_method",
    "periodicity_harmonic_factor",
    "periodicity_harmonic_objective",
    "periodicity_scatter_ratio",
    "periodicity_alias_flag",
    "periodicity_bootstrap_sig",
    "lsp_bootstrap_sig",
    "lsp_is_alias",
    "pdm_theta",
    "pdm_snr",
    "pdm_bootstrap_sig",
    "ce_entropy",
    "ce_snr",
    "ce_bootstrap_sig",
    "long_ls_peak_power",
)

ADDITIONAL_DIP_FEATURES = (
    "dip_best_morph",
    "dip_best_delta_bic",
    "dip_best_width_param",
    "dip_symmetry_score",
    "dip_best_amp",
    "dip_bayes_factor",
    "dip_best_p",
    "dip_max_event_prob",
    "dip_count",
    "dip_run_count",
    "dip_max_run_points",
    "dip_max_run_duration",
    "dip_max_run_sum",
    "dip_max_run_max",
    "dip_max_run_cameras",
    "dip_max_log_bf_local",
    "dip_is_single_event",
    "dip_inter_event_spacing_median",
    "dip_inter_event_spacing_std",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
    "dipper_n_dips",
    "dipper_n_valid_dips",
)

ADDITIONAL_JUMP_FEATURES = (
    "jump_best_morph",
    "jump_best_delta_bic",
    "jump_best_width_param",
    "jump_best_amp",
    "jump_bayes_factor",
    "jump_best_p",
    "jump_max_event_prob",
    "jump_count",
    "jump_run_count",
    "jump_max_run_points",
    "jump_max_run_duration",
    "jump_max_run_sum",
    "jump_max_run_max",
    "jump_max_run_cameras",
    "jump_max_log_bf_local",
    "jump_is_single_event",
    "jump_inter_event_spacing_median",
    "jump_inter_event_spacing_std",
    "jump_amplitude_consistency",
    "jump_duration_consistency",
    "jumper_n_jumps",
    "jumper_n_valid_jumps",
)

ADDITIONAL_LC_FEATURES = (
    *ADDITIONAL_PERIODICITY_FEATURES,
    *ADDITIONAL_DIP_FEATURES,
    *ADDITIONAL_JUMP_FEATURES,
)
if len(ADDITIONAL_LC_FEATURES) != 61 or len(set(ADDITIONAL_LC_FEATURES)) != 61:
    raise RuntimeError("The curated expanded model must contain 61 unique additional features")

# These columns remain useful in stored products as provenance or operational
# diagnostics, but they must not enter a model alongside their canonical value.
# The LSP fields are intentionally absent from this list: once recomputed by the
# genuine Lomb--Scargle path, they are independent of the selected PDM/CE result.
REDUNDANT_MODEL_FEATURES = (
    "stats_mhps_non_zero",
    "stats_harmonics_period",
    # In the July 1 population, ``none`` maps exactly to g-only and
    # ``v_median_to_g_median`` maps exactly to g+V. Keep the more direct
    # photometry-coverage field in the model.
    "stats_photometry_band_alignment",
    "dip_trigger_max",
    "jump_trigger_max",
    *sorted(MODEL_FEATURE_EXCLUSION_COLUMNS),
)

EXTERNAL_CONTEXT_FEATURES = (
    *ASTROPHYSICAL_CONTEXT_FEATURES,
    *NEXT_ITERATION_CONTEXT_FEATURES,
)
SLIM_SCORE_COLUMNS = (
    "candidate_id",
    "asas_sn_id",
    "event_class",
    "morphology_primary",
    "workflow_status",
    "dipper_score",
    "dipper_n_dips",
    "dipper_n_valid_dips",
    "dip_run_count",
    "dip_is_single_event",
    "dipper_recurrence_class",
    "dipper_recurrence_evidence",
    "dip_significant",
    "stats_amplitude",
    "stats_percent_amplitude",
    "stats_photometry_robust_sigma_mag",
    "stats_variability_reduced_chi2_vs_constant",
    "stats_variability_stetson_J",
    "stats_variability_stetson_L",
    "stats_variability_von_neumann_ratio",
    "stats_variability_quasi_periodicity_q",
    "stats_variability_flux_asymmetry_m",
    "stats_skew",
    "stats_max_slope",
    "vetting_likely_known",
    "catalog_match",
    "catalog_source",
    "simbad_otype",
    "vsx_class",
)


def _sqlite_ro(path: Path) -> sqlite3.Connection:
    db_path = path.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Review DB not found: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return [str(row[1]) for row in rows]


def _selected_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    columns = []
    for column in _table_columns(conn, table):
        if column in SKIP_DB_COLUMNS or column.endswith("_json"):
            continue
        columns.append(column)
    return columns


def _read_selected_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    columns = _selected_columns(conn, table)
    select_list = ", ".join(_quote_identifier(column) for column in columns)
    return pd.read_sql_query(f"SELECT {select_list} FROM {_quote_identifier(table)}", conn)


def load_candidate_review_table(db_path: Path) -> pd.DataFrame:
    """Load all candidate rows with review labels left-joined when present."""

    with _sqlite_ro(db_path) as conn:
        candidates = _read_selected_table(conn, "candidates")
        reviews = _read_selected_table(conn, "reviews")

    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    reviews["candidate_id"] = reviews["candidate_id"].astype(str)
    table = candidates.merge(reviews, on="candidate_id", how="left", suffixes=("", "_review"))
    return table.loc[:, ~table.columns.duplicated()].copy()


def _clean_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].fillna("").astype(str).str.strip()


def add_dipper_like_target(table: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary training label for reviewed non-unclassified rows."""

    out = table.copy()
    workflow = _clean_text_series(out, "workflow_status")
    status = _clean_text_series(out, "status")
    event_class = _clean_text_series(out, "event_class")
    morphology = _clean_text_series(out, "morphology_primary")

    reviewed = (
        workflow.ne("")
        & workflow.ne("unreviewed")
        | (status.ne("") & status.ne("unreviewed"))
    )
    trainable_review = reviewed & ~event_class.isin(EXCLUDED_EVENT_CLASSES)
    positive = event_class.isin(POSITIVE_EVENT_CLASSES) | morphology.isin(POSITIVE_MORPHOLOGIES)

    out[TARGET_COLUMN] = pd.NA
    out.loc[trainable_review & positive, TARGET_COLUMN] = POSITIVE_LABEL
    out.loc[trainable_review & ~positive, TARGET_COLUMN] = NEGATIVE_LABEL
    return out


def add_external_context_features(table: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the shared astrophysical-context helper."""

    return add_next_iteration_context_features(
        add_astrophysical_context_features(table)
    )


def _is_usable_feature_series(series: pd.Series, *, min_non_null: int, max_cardinality: int) -> bool:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return int(numeric.notna().sum()) >= min_non_null and numeric.nunique(dropna=True) > 1

    nonempty = series.dropna().astype(str).str.strip()
    nonempty = nonempty.loc[nonempty.ne("")]
    if int(nonempty.size) < min_non_null:
        return False
    cardinality = int(nonempty.nunique())
    return 1 < cardinality <= max_cardinality


def select_feature_columns(
    table: pd.DataFrame,
    *,
    feature_set: str,
    min_non_null: int,
    max_cardinality: int,
    max_features: int | None,
) -> list[str]:
    """Select modelable columns after removing labels, identifiers, and constants."""

    trainable = table.loc[table[TARGET_COLUMN].notna()].copy()
    drop = set(DEFAULT_DROP_COLUMNS) | {TARGET_COLUMN}
    required_features: tuple[str, ...] = ()
    if feature_set in CURATED_LC_FEATURE_SETS:
        required_features = (
            *ADDITIONAL_LC_FEATURES,
            *RECOVERY_BOUNDED_EVENT_FEATURES,
            *EXTERNAL_CONTEXT_FEATURES,
        )
    if required_features:
        missing = [column for column in required_features if column not in table.columns]
        if missing:
            raise KeyError(f"Requested curated-model features are missing: {missing}")
        unusable = [
            column
            for column in required_features
            if not _is_usable_feature_series(
                trainable[column],
                min_non_null=min_non_null,
                max_cardinality=max_cardinality,
            )
        ]
        if unusable:
            raise ValueError(f"Requested curated-model features are unusable: {unusable}")

    feature_cols: list[str] = []
    for column in table.columns:
        if column in drop or column.endswith("_review"):
            continue
        if column in REDUNDANT_MODEL_FEATURES:
            continue
        if feature_set in STATS_ONLY_FEATURE_SETS and not column.startswith("stats_"):
            continue
        if (
            feature_set in CURATED_LC_FEATURE_SETS
            and not column.startswith("stats_")
            and column not in required_features
        ):
            continue
        if feature_set == "full" and column.startswith("payload"):
            continue
        if _is_usable_feature_series(
            trainable[column],
            min_non_null=min_non_null,
            max_cardinality=max_cardinality,
        ):
            feature_cols.append(column)
    ranked = sorted(
        feature_cols,
        key=lambda column: (
            0 if column.startswith("stats_") else 1,
            -int(trainable[column].notna().sum()),
            column,
        ),
    )
    if max_features is not None and max_features > 0:
        ranked = ranked[:max_features]
    if required_features:
        omitted = [column for column in required_features if column not in ranked]
        if omitted:
            raise ValueError(
                "max_features excluded requested curated-model features; "
                f"increase the cap above {max_features}: {omitted}"
            )
    return ranked


def training_frame(table: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    columns = ["candidate_id", TARGET_COLUMN, *feature_columns]
    return table.loc[:, [column for column in columns if column in table.columns]].copy()


def _model_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        random_state=args.random_state,
        val_size=args.val_size,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        max_categorical_cardinality=args.max_categorical_cardinality,
        min_class_count=args.min_class_count,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.early_stopping_rounds,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_selection_folds=args.early_stopping_selection_folds,
        calibration_method="none",
        reliability_bins=args.reliability_bins,
    )


def write_gain_importance(result: Any, output_dir: Path) -> None:
    if result.model is None:
        return
    booster = result.model.booster_
    frame = pd.DataFrame(
        {
            "feature": result.feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False, ignore_index=True)
    frame.to_csv(output_dir / "feature_importance_gain.csv", index=False)


def write_feature_policy_metadata(output_dir: Path) -> None:
    """Record the enforced de-duplication policy beside the saved model."""

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_policy"] = {
        "exact_duplicate_check": "passed",
        "excluded_redundant_features": list(REDUNDANT_MODEL_FEATURES),
        "lsp_policy": "retain only when independently recomputed by Lomb-Scargle",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def feature_window_table(
    table: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    all_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize percentile windows that keep a target fraction of dippers."""

    target = table[TARGET_COLUMN].fillna("").astype(str)
    positive_mask = target.eq(POSITIVE_LABEL)
    negative_mask = target.eq(NEGATIVE_LABEL)
    n_positive = int(positive_mask.sum())
    n_negative = int(negative_mask.sum())
    n_all = int(len(all_candidates))

    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature not in table.columns or feature not in all_candidates.columns:
            continue
        reviewed_values = pd.to_numeric(table[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        positive_values = reviewed_values.loc[positive_mask].dropna()
        if positive_values.empty:
            continue
        negative_values = reviewed_values.loc[negative_mask]
        all_values = pd.to_numeric(all_candidates[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)

        for coverage in WINDOW_COVERAGES:
            tail = (1.0 - coverage) / 2.0
            lo = float(positive_values.quantile(tail))
            hi = float(positive_values.quantile(1.0 - tail))
            reviewed_in_window = reviewed_values.between(lo, hi, inclusive="both")
            all_in_window = all_values.between(lo, hi, inclusive="both")

            pos_kept = int((reviewed_in_window & positive_mask).sum())
            neg_kept = int((reviewed_in_window & negative_mask).sum())
            reviewed_kept = pos_kept + neg_kept
            all_kept = int(all_in_window.sum())
            rows.append(
                {
                    "feature": feature,
                    "target_coverage": coverage,
                    "lo": lo,
                    "hi": hi,
                    "positive_kept": pos_kept,
                    "positive_total": n_positive,
                    "positive_recall": pos_kept / n_positive if n_positive else np.nan,
                    "reviewed_nondipper_kept": neg_kept,
                    "reviewed_nondipper_total": n_negative,
                    "reviewed_kept": reviewed_kept,
                    "reviewed_precision": pos_kept / reviewed_kept if reviewed_kept else np.nan,
                    "all_candidates_kept": all_kept,
                    "all_candidates_total": n_all,
                    "all_candidate_fraction_kept": all_kept / n_all if n_all else np.nan,
                    "positive_nonmissing": int(positive_values.size),
                    "positive_nonmissing_fraction": positive_values.size / n_positive if n_positive else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["target_coverage", "positive_recall", "all_candidate_fraction_kept", "reviewed_precision"],
        ascending=[False, False, True, False],
        ignore_index=True,
    )


def slim_scores(scored: pd.DataFrame, *, probability_column: str) -> pd.DataFrame:
    columns = [column for column in SLIM_SCORE_COLUMNS if column in scored.columns]
    score_columns = ["y_pred", "prediction_confidence", probability_column]
    out = scored.loc[:, [*columns, *score_columns]].copy()
    return out.sort_values(
        [probability_column, "dipper_score"],
        ascending=[False, False],
        na_position="last",
        ignore_index=True,
    )


def unreviewed_score_rows(scored: pd.DataFrame) -> pd.DataFrame:
    workflow = _clean_text_series(scored, "workflow_status")
    event_class = _clean_text_series(scored, "event_class")
    return scored.loc[
        workflow.isin(("", "unreviewed")) & event_class.isin(("", "unclassified"))
    ].copy()


def train_and_score_feature_set(
    table: pd.DataFrame,
    *,
    feature_set: str,
    output_dir: Path,
    config: TrainingConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    feature_columns = select_feature_columns(
        table,
        feature_set=feature_set,
        min_non_null=args.min_non_null,
        max_cardinality=args.max_categorical_cardinality,
        max_features=args.max_features,
    )
    print(f"{feature_set}: selected {len(feature_columns)} feature columns", flush=True)
    model_input = training_frame(table, feature_columns)
    print(f"{feature_set}: training LightGBM", flush=True)
    result = train_target_model(model_input, TARGET_COLUMN, config=config)

    model_dir = output_dir / feature_set
    save_target_model(result, model_dir)
    write_feature_policy_metadata(model_dir)
    write_gain_importance(result, model_dir)

    print(f"{feature_set}: scoring all candidates", flush=True)
    scoring_input = training_frame(table, feature_columns)
    scored = score_target_model(model_dir, scoring_input)
    probability_column = f"prob_{POSITIVE_LABEL}"
    prediction_columns = ["y_pred", "prediction_confidence", probability_column]
    score_base = table.drop(columns=prediction_columns, errors="ignore")
    scored_slim = slim_scores(
        score_base.merge(
            scored[["candidate_id", *prediction_columns]],
            on="candidate_id",
            how="left",
        ),
        probability_column=probability_column,
    )
    scored_slim.to_parquet(model_dir / "all_candidates_scores.parquet", index=False)
    unreviewed_score_rows(scored_slim).head(args.high_priority_rows).to_csv(
        model_dir / "high_priority_review_queue.csv",
        index=False,
    )

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(table[column]) or column.startswith("stats_")
    ]
    windows = feature_window_table(
        model_input,
        feature_columns=numeric_features,
        all_candidates=table,
    )
    windows.to_csv(model_dir / "feature_windows.csv", index=False)
    print(f"{feature_set}: wrote outputs to {model_dir}", flush=True)

    return {
        "feature_set": feature_set,
        "model_dir": str(model_dir),
        "n_features": len(feature_columns),
        "n_trainable_rows": result.n_rows,
        "class_counts": result.class_counts,
        "label_classes": result.label_classes,
        "probability_column": probability_column,
        "excluded_redundant_features": list(REDUNDANT_MODEL_FEATURES),
        "additional_lc_features": list(ADDITIONAL_LC_FEATURES) if feature_set in CURATED_LC_FEATURE_SETS else [],
        "recovery_bounded_event_features": (
            list(RECOVERY_BOUNDED_EVENT_FEATURES)
            if feature_set in CURATED_LC_FEATURE_SETS
            else []
        ),
        "external_context_features": (
            list(EXTERNAL_CONTEXT_FEATURES) if feature_set in CURATED_LC_FEATURE_SETS else []
        ),
        "top_features": result.feature_importance.head(20).to_dict(orient="records"),
        "test_metrics": result.holdout_metrics,
    }


def write_label_audit(table: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    target = table[TARGET_COLUMN].fillna("").astype(str)
    audit = {
        "n_candidates": int(len(table)),
        "n_reviewed_trainable": int(target.ne("").sum()),
        "target_counts": target.loc[target.ne("")].value_counts().to_dict(),
        "event_class_counts": _clean_text_series(table, "event_class").value_counts().head(30).to_dict(),
        "morphology_primary_counts": _clean_text_series(table, "morphology_primary").value_counts().head(30).to_dict(),
    }
    (output_dir / "label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-set",
        choices=(
            "stats_only",
            "stats_only_early_stopping",
            EXPANDED_FEATURE_SET,
            EXTERNAL_CONTEXT_FEATURE_SET,
            "full",
            "both",
            "comparison",
        ),
        default=EXPANDED_FEATURE_SET,
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--min-non-null", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=250)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-selection-folds", type=int, default=3)
    parser.add_argument("--high-priority-rows", type=int, default=300)
    parser.add_argument(
        "--recovery-feature-cache",
        type=Path,
        default=default_recovery_feature_cache(DEFAULT_DB_PATH),
        help=(
            "Versioned candidate-independent recovery-morphology cache. "
            "Missing or stale rows are measured before fitting."
        ),
    )
    parser.add_argument(
        "--recovery-workers",
        type=int,
        default=4,
        help="Parallel workers used only while filling the recovery-feature cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set == "both":
        feature_sets = ("stats_only", "full")
    elif args.feature_set == "comparison":
        feature_sets = ("stats_only_early_stopping", EXPANDED_FEATURE_SET)
    else:
        feature_sets = (args.feature_set,)

    table = load_candidate_review_table(args.db_path)
    if any(feature_set in CURATED_LC_FEATURE_SETS for feature_set in feature_sets):
        table = add_external_context_features(table)
        table = add_recovery_bounded_event_features(
            table,
            args.recovery_feature_cache,
            workers=args.recovery_workers,
        )
    table = add_observed_dipper_recurrence(table)
    table = add_dipper_like_target(table)
    audit = write_label_audit(table, output_dir)
    config = _model_config(args)

    summaries = [
        train_and_score_feature_set(
            table,
            feature_set=feature_set,
            output_dir=output_dir,
            config=config,
            args=args,
        )
        for feature_set in feature_sets
    ]

    run_summary = {
        "db_path": str(args.db_path),
        "output_dir": str(output_dir),
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "positive_event_classes": sorted(POSITIVE_EVENT_CLASSES),
        "positive_morphologies": sorted(POSITIVE_MORPHOLOGIES),
        "recovery_feature_cache": str(args.recovery_feature_cache),
        "recovery_feature_schema_version": RECOVERY_FEATURE_SCHEMA_VERSION,
        "label_audit": audit,
        "models": summaries,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for summary in summaries:
        print(
            f"{summary['feature_set']}: "
            f"{summary['n_trainable_rows']} training rows, "
            f"{summary['n_features']} features, "
            f"outputs -> {summary['model_dir']}"
        )


if __name__ == "__main__":
    main()
