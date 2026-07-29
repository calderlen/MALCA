"""Reproducible training products for the July 1 Review classifiers.

The review notebooks remain the place for figures and investigation.  This
module owns the data contract, labels, feature selection, fitting, and saved
score artifacts so retraining does not depend on executing notebook cells in a
particular order.  It deliberately provides functions for standalone Python
scripts rather than registering new ``malca`` CLI commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
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
    STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
)
from malca.meta_analysis.ml.review_lightgbm import (
    ASTROPHYSICAL_CONTEXT_FEATURES,
    TrainingConfig,
    add_astrophysical_context_features,
    save_target_model,
    score_target_model,
    train_target_model,
)
from malca.review.dipper_recurrence import add_observed_dipper_recurrence


DEFAULT_RUN_DIR = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_DB_PATH = DEFAULT_RUN_DIR / "review" / "review.db"

MODEL_KEYS = (
    "eight_class",
    "dipper_recurrence",
    "eb",
    "ltv",
    "microlensing",
)

DEFAULT_MODEL_DIRS = {
    "eight_class": DEFAULT_RUN_DIR / "results" / "eight_class_ml_separability" / "stats_plus_periodicity_dip_jump_context",
    "dipper_recurrence": DEFAULT_RUN_DIR / "results" / "dipper_recurrence_ml" / "stats_plus_astrophysical_context",
    "eb": DEFAULT_RUN_DIR / "results" / "eb_feature_selection" / "stats_plus_astrophysical_context",
    "ltv": DEFAULT_RUN_DIR / "results" / "ltv_feature_selection" / "stats_plus_astrophysical_context",
    "microlensing": DEFAULT_RUN_DIR / "results" / "microlensing_feature_selection" / "stats_plus_astrophysical_context",
}

MODEL_EXCLUDED_STATS = {
    *STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
    "stats_variability_lomb_scargle_best_period_days",
}
EIGHT_CLASS_EXCLUDED_STATS = {
    *MODEL_EXCLUDED_STATS,
    "stats_mhps_non_zero",
    "stats_harmonics_period",
    # The current July 1 table encodes this exactly as
    # ``stats_photometry_band_mode``.  Keep the direct coverage value and
    # prevent the shared trainer from fitting duplicate columns.
    "stats_photometry_band_alignment",
}

EB_REVIEW_TAGS = {
    "eclipsing_like",
    "detached_binary_like",
    "semi_detached_binary_like",
    "contact_binary_like",
    "ellipsoidal_like",
    "heartbeat_like",
}

LTV_NEGATIVE_EVENT_CLASSES = {
    "instrumental",
    "periodic",
    "quasi_periodic",
    "dipper",
    "mixed_dip_and_burst",
    "brightening_event",
    "nonvariable_or_low_snr",
    "microlensing",
    "flare",
    "stochastic",
}
MICROLENS_NEGATIVE_EVENT_CLASSES = {
    "instrumental",
    "periodic",
    "quasi_periodic",
    "dipper",
    "mixed_dip_and_burst",
    "brightening_event",
    "nonvariable_or_low_snr",
    "ltv",
    "flare",
    "stochastic",
}
EB_NEGATIVE_EVENT_CLASSES = {
    "instrumental",
    "periodic",
    "quasi_periodic",
    "brightening_event",
    "nonvariable_or_low_snr",
    "dipper",
    "mixed_dip_and_burst",
    "ltv",
    "microlensing",
    "flare",
    "stochastic",
}

EIGHT_CLASS_ORDER = (
    "dipper",
    "eclipsing_binary_like",
    "long_term_variable",
    "long_period_variable",
    "microlensing",
    "quasi_periodic",
    "brightening_event",
    "artifact_or_nonvariable",
)

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
EIGHT_CLASS_CONTEXT_FEATURES = (
    *ASTROPHYSICAL_CONTEXT_FEATURES,
    *NEXT_ITERATION_CONTEXT_FEATURES,
)


@dataclass(frozen=True)
class PreparedReviewModel:
    """The complete, inspectable data contract for one training run."""

    model_key: str
    target_column: str
    positive_label: str | None
    table: pd.DataFrame
    model_input: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_counts: dict[str, int]
    label_audit: dict[str, Any]
    probability_columns: tuple[str, ...]
    score_context_columns: tuple[str, ...]


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _read_table_without_payloads(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    keep_morphology_secondary_json: bool,
) -> pd.DataFrame:
    schema = pd.read_sql_query(f"PRAGMA table_info({_quote_identifier(table_name)})", conn)
    columns = [
        str(name)
        for name in schema["name"]
        if (
            not str(name).endswith("_json")
            or (keep_morphology_secondary_json and str(name) == "morphology_secondary_json")
        )
        and str(name) not in {"payload_json", "legacy_review_json", "notes"}
    ]
    select_list = ", ".join(_quote_identifier(column) for column in columns)
    return pd.read_sql_query(
        f"SELECT {select_list} FROM {_quote_identifier(table_name)}", conn
    )


def load_review_population(db_path: str | Path, *, keep_morphology_secondary_json: bool) -> pd.DataFrame:
    """Load the current Review DB as one flat candidate-plus-review table."""

    path = Path(db_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as conn:
        candidates = _read_table_without_payloads(
            conn,
            "candidates",
            keep_morphology_secondary_json=keep_morphology_secondary_json,
        )
        reviews = _read_table_without_payloads(
            conn,
            "reviews",
            keep_morphology_secondary_json=keep_morphology_secondary_json,
        )
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    reviews["candidate_id"] = reviews["candidate_id"].astype(str)
    table = candidates.merge(reviews, on="candidate_id", how="left", suffixes=("", "_review"))
    table = table.loc[:, ~table.columns.duplicated()].copy()
    table = add_astrophysical_context_features(table)
    table = add_next_iteration_context_features(table)
    return add_observed_dipper_recurrence(table)


def _clean_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _parse_secondary_tags(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, list):
        return {str(item).strip() for item in parsed if str(item).strip()}
    return {part.strip() for part in text.split("|") if part.strip()}


def _secondary_tag_sets(table: pd.DataFrame) -> pd.Series:
    scalar = _clean_text(table, "morphology_secondary")
    raw_sets = table.get(
        "morphology_secondary_json", pd.Series("", index=table.index)
    ).map(_parse_secondary_tags)
    return pd.Series(
        [tags | ({value} if value else set()) for tags, value in zip(raw_sets, scalar)],
        index=table.index,
    )


def _reviewed_mask(table: pd.DataFrame) -> pd.Series:
    status = _clean_text(table, "status")
    workflow = _clean_text(table, "workflow_status")
    return (status.ne("") & status.ne("unreviewed")) | (
        workflow.ne("") & workflow.ne("unreviewed")
    )


def is_usable_model_feature(
    series: pd.Series,
    *,
    min_non_null: int,
    max_cardinality: int = 50,
) -> bool:
    """Match the notebook's conservative feature-eligibility contract."""

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return int(numeric.notna().sum()) >= min_non_null and numeric.nunique(dropna=True) > 1
    values = series.dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    return int(values.size) >= min_non_null and 1 < int(values.nunique()) <= max_cardinality


def _stats_plus_context_features(
    table: pd.DataFrame,
    trainable: pd.Series,
    *,
    excluded_stats: Iterable[str],
    min_non_null: int,
) -> list[str]:
    stats_columns = [column for column in table.columns if column.startswith("stats_")]
    excluded = set(excluded_stats)
    features = [
        column
        for column in stats_columns
        if column not in excluded
        and is_usable_model_feature(
            table.loc[trainable, column], min_non_null=min_non_null
        )
    ]
    features = sorted(
        features,
        key=lambda column: (-int(table.loc[trainable, column].notna().sum()), column),
    )
    unusable_context = [
        column
        for column in ASTROPHYSICAL_CONTEXT_FEATURES
        if not is_usable_model_feature(
            table.loc[trainable, column], min_non_null=min_non_null
        )
    ]
    if unusable_context:
        raise ValueError(
            f"Requested astrophysical-context features are unusable: {unusable_context}"
        )
    return [*features, *ASTROPHYSICAL_CONTEXT_FEATURES]


def _drop_exact_duplicate_features(
    frame: pd.DataFrame, columns: Iterable[str]
) -> tuple[list[str], dict[str, str]]:
    """Drop aliases deterministically before fitting a binary Review model."""

    signature_groups: dict[bytes, list[str]] = {}
    normalized: dict[str, pd.Series] = {}
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            values = (
                pd.to_numeric(series, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype("float64")
            )
        else:
            text = series.fillna("").astype(str).str.strip()
            categories = sorted(value for value in text.unique() if value)
            values = text.map({value: index for index, value in enumerate(categories)}).fillna(-1).astype("float64")
        normalized[column] = values
        signature = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64").tobytes()
        signature_groups.setdefault(signature, []).append(column)

    aliases: dict[str, str] = {}
    for group in signature_groups.values():
        retained: list[str] = []
        for column in group:
            duplicate_of = next(
                (kept for kept in retained if normalized[column].equals(normalized[kept])),
                None,
            )
            if duplicate_of is None:
                retained.append(column)
            else:
                aliases[column] = duplicate_of
    return [column for column in columns if column not in aliases], aliases


def _binary_labels(table: pd.DataFrame, model_key: str) -> tuple[str, str, str, dict[str, Any]]:
    reviewed = _reviewed_mask(table)
    event_class = _clean_text(table, "event_class")
    if model_key == "ltv":
        target, positive_label, negative_label = "ltv_like_label", "ltv_like", "not_ltv"
        positive = reviewed & event_class.eq("ltv")
        negative = reviewed & event_class.isin(LTV_NEGATIVE_EVENT_CLASSES) & ~positive
        definition = "reviewed event_class=ltv versus reviewed clear non-LTV classes"
    elif model_key == "eb":
        target, positive_label, negative_label = "eb_like_label", "eb_like", "not_eb"
        tags = _secondary_tag_sets(table)
        physical_primary = _clean_text(table, "physical_primary")
        positive = reviewed & (
            tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
            | physical_primary.eq("eclipsing_or_geometric_binary")
        )
        negative = reviewed & event_class.isin(EB_NEGATIVE_EVENT_CLASSES) & ~positive
        table["human_eb_tags"] = tags.map(lambda values: "|".join(sorted(values & EB_REVIEW_TAGS)))
        definition = "human-review EB-like secondary morphology or binary physical label versus clear non-EB classes"
    elif model_key == "microlensing":
        target, positive_label, negative_label = (
            "microlensing_like_label",
            "microlensing_like",
            "not_microlensing",
        )
        tags = _secondary_tag_sets(table)
        possible_tag = tags.map(lambda values: "possible_microlensing_event" in values)
        positive = reviewed & (event_class.eq("microlensing") | possible_tag)
        negative = reviewed & event_class.isin(MICROLENS_NEGATIVE_EVENT_CLASSES) & ~positive
        table["human_microlensing_like"] = positive.astype(bool)
        table["microlensing_label_source"] = np.select(
            [
                positive & event_class.eq("microlensing") & possible_tag,
                positive & event_class.eq("microlensing"),
                positive & possible_tag,
                negative,
            ],
            [
                "human_event_and_morphology_microlensing",
                "human_event_class_microlensing",
                "human_morphology_microlensing",
                "human_review_non_microlensing",
            ],
            default="unlabeled",
        )
        definition = "reviewed event_class=microlensing or possible_microlensing_event tag"
    else:
        raise ValueError(f"Not a binary model key: {model_key}")

    table[target] = pd.Series(pd.NA, index=table.index, dtype="string")
    table.loc[positive, target] = positive_label
    table.loc[negative, target] = negative_label
    audit = {
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "n_trainable": int((positive | negative).sum()),
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
        "n_reviewed_excluded": int((reviewed & ~(positive | negative)).sum()),
        "label_definition": definition,
    }
    return target, positive_label, negative_label, audit


def _eight_class_labels(table: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    target = "human_eight_class_label"
    reviewed = _reviewed_mask(table)
    event_class = _clean_text(table, "event_class")
    morphology_primary = _clean_text(table, "morphology_primary")
    physical_primary = _clean_text(table, "physical_primary")
    tags = _secondary_tag_sets(table)
    table["human_secondary_tags"] = tags.map(lambda values: "|".join(sorted(values)))
    microlensing = reviewed & (
        event_class.eq("microlensing")
        | tags.map(lambda values: "possible_microlensing_event" in values)
    )
    masks = {
        "dipper": reviewed & event_class.isin({"dipper", "mixed_dip_and_burst"}),
        "eclipsing_binary_like": reviewed
        & (
            tags.map(lambda values: bool(values & EB_REVIEW_TAGS))
            | physical_primary.eq("eclipsing_or_geometric_binary")
        ),
        "long_term_variable": reviewed & morphology_primary.eq("long_term_trend"),
        "long_period_variable": reviewed
        & morphology_primary.eq("long_period_variability"),
        "microlensing": microlensing,
        "quasi_periodic": reviewed & event_class.eq("quasi_periodic"),
        "brightening_event": reviewed & event_class.eq("brightening_event") & ~microlensing,
        "artifact_or_nonvariable": reviewed
        & morphology_primary.isin(
            {"artifact_or_bad_photometry", "nonvariable_or_low_snr"}
        ),
    }
    membership_count = sum(mask.astype(int) for mask in masks.values())
    if bool(membership_count.gt(1).any()):
        raise ValueError(
            f"Found {int(membership_count.gt(1).sum())} reviewed candidates in more than one eight-class target"
        )
    table[target] = pd.Series(pd.NA, index=table.index, dtype="string")
    for label, mask in masks.items():
        table.loc[mask, target] = label
    counts = table.loc[table[target].notna(), target].value_counts().reindex(EIGHT_CLASS_ORDER, fill_value=0)
    if int(counts.min()) < 5:
        raise ValueError(f"An eight-class target has fewer than 5 examples: {counts.to_dict()}")
    source = pd.Series("unlabeled_for_eight_class_model", index=table.index, dtype="object")
    source.loc[masks["dipper"]] = "human_event_class_dipper"
    source.loc[masks["eclipsing_binary_like"]] = "human_secondary_eb_morphology"
    source.loc[masks["long_term_variable"]] = "human_morphology_long_term_trend"
    source.loc[masks["long_period_variable"]] = "human_morphology_long_period_variability"
    source.loc[masks["microlensing"]] = "human_event_or_morphology_microlensing"
    source.loc[masks["quasi_periodic"]] = "human_event_class_quasi_periodic"
    source.loc[masks["brightening_event"]] = "human_event_class_brightening_non_microlensing"
    rejection_mask = masks["artifact_or_nonvariable"]
    source.loc[
        rejection_mask & morphology_primary.eq("artifact_or_bad_photometry")
    ] = "human_morphology_artifact_or_bad_photometry"
    source.loc[
        rejection_mask & morphology_primary.eq("nonvariable_or_low_snr")
    ] = "human_morphology_nonvariable_or_low_snr"
    table["human_label_source"] = source
    return target, {
        "n_candidates": int(len(table)),
        "n_reviewed": int(reviewed.sum()),
        "n_trainable": int(table[target].notna().sum()),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "n_overlap_rows": int(membership_count.gt(1).sum()),
        "n_other_reviewed_excluded": int((reviewed & table[target].isna()).sum()),
        "rejection_component_counts": {
            "artifact_or_bad_photometry": int(
                (rejection_mask & morphology_primary.eq("artifact_or_bad_photometry")).sum()
            ),
            "nonvariable_or_low_snr": int(
                (rejection_mask & morphology_primary.eq("nonvariable_or_low_snr")).sum()
            ),
        },
        "label_definition": (
            "eight mutually exclusive human review classes; artifact/bad "
            "photometry and nonvariable/low-SNR are one rejection class"
        ),
    }


def _dipper_recurrence_labels(table: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    """Build a human-labeled recurrent versus non-recurrent dipper target.

    The target deliberately uses reviewer morphology tags, rather than the
    triggered-dip run count used for the observed recurrence display field.
    Direct recurrence measurements are excluded from this model's feature
    policy by selecting only stats and astrophysical-context features below.
    """

    target = "human_dipper_recurrence_label"
    positive_label = "recurrent_given_dipper"
    negative_label = "non_recurrent_given_dipper"
    reviewed = _reviewed_mask(table)
    event_class = _clean_text(table, "event_class")
    tags = _secondary_tag_sets(table)
    dipper = reviewed & event_class.isin({"dipper", "mixed_dip_and_burst"})
    recurrent_tags = {
        "recurrent_dips",
        "periodic_dips",
        "quasi_periodic_dips",
        "aperiodic_dips",
        "stochastic_dips",
    }
    has_recurrent_tag = dipper & tags.map(lambda values: bool(values & recurrent_tags))
    single = dipper & tags.map(lambda values: "single_dip" in values)
    conflicts = has_recurrent_tag & single
    recurrent = has_recurrent_tag & ~single
    non_recurrent = single & ~has_recurrent_tag

    table[target] = pd.Series(pd.NA, index=table.index, dtype="string")
    table.loc[recurrent, target] = positive_label
    table.loc[non_recurrent, target] = negative_label
    table["human_dipper_recurrence_tags"] = tags.map(
        lambda values: "|".join(sorted(values & (recurrent_tags | {"single_dip"})))
    )
    source = pd.Series("unlabeled", index=table.index, dtype="object")
    source.loc[recurrent] = "human_recurrent_dip_morphology"
    source.loc[non_recurrent] = "human_single_dip_morphology"
    source.loc[conflicts] = "excluded_conflicting_human_morphology"
    table["human_dipper_recurrence_label_source"] = source
    audit = {
        "n_candidates": int(len(table)),
        "n_reviewed_dipper_or_mixed": int(dipper.sum()),
        "n_trainable": int((recurrent | non_recurrent).sum()),
        "n_recurrent": int(recurrent.sum()),
        "n_non_recurrent": int(non_recurrent.sum()),
        "n_conflicting_recurrence_tags": int(conflicts.sum()),
        "n_dipper_rows_without_recurrence_label": int(
            (dipper & ~(recurrent | non_recurrent)).sum()
        ),
        "label_definition": (
            "reviewed dipper or mixed-dip-and-burst candidates: recurrent "
            "morphology tags versus single_dip without a recurrent tag"
        ),
        "feature_policy": (
            "stats plus astrophysical context only; direct triggered-dip "
            "recurrence measurements are excluded"
        ),
    }
    return target, positive_label, audit


def _dipper_recurrence_features(
    table: pd.DataFrame, trainable: pd.Series
) -> tuple[list[str], dict[str, str]]:
    features = _stats_plus_context_features(
        table,
        trainable,
        excluded_stats=MODEL_EXCLUDED_STATS,
        min_non_null=20,
    )
    return _drop_exact_duplicate_features(table.loc[trainable], features)


def _eight_class_features(table: pd.DataFrame, trainable: pd.Series) -> list[str]:
    requested_native = (
        *ADDITIONAL_LC_FEATURES,
        *RECOVERY_BOUNDED_EVENT_FEATURES,
    )
    missing = [column for column in requested_native if column not in table.columns]
    if missing:
        raise KeyError(f"Requested additional features are missing: {missing}")
    unusable = [
        column
        for column in (*requested_native, *NEXT_ITERATION_CONTEXT_FEATURES)
        if not is_usable_model_feature(table.loc[trainable, column], min_non_null=20)
    ]
    if unusable:
        raise ValueError(f"Requested additional features are unusable: {unusable}")
    stats = _stats_plus_context_features(
        table,
        trainable,
        excluded_stats=EIGHT_CLASS_EXCLUDED_STATS,
        min_non_null=20,
    )
    stats_without_context = [
        column
        for column in stats
        if column not in ASTROPHYSICAL_CONTEXT_FEATURES
    ]
    return [
        *stats_without_context,
        *requested_native,
        *EIGHT_CLASS_CONTEXT_FEATURES,
    ]


def _binary_config(model_key: str) -> TrainingConfig:
    common = dict(
        random_state=42,
        val_size=0.15,
        test_size=0.20,
        cv_folds=5,
        n_estimators=220,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        max_categorical_cardinality=50,
        min_class_count=5,
        class_weight="balanced",
        calibration_method="none",
    )
    if model_key == "eb":
        return TrainingConfig(num_leaves=23, min_child_samples=20, n_jobs=4, reliability_bins=10, **common)
    if model_key == "ltv":
        return TrainingConfig(num_leaves=15, min_child_samples=10, n_jobs=1, reliability_bins=8, **common)
    if model_key == "microlensing":
        return TrainingConfig(num_leaves=15, min_child_samples=8, n_jobs=1, reliability_bins=8, **common)
    raise ValueError(f"Not a binary model key: {model_key}")


def _eight_class_config() -> TrainingConfig:
    return TrainingConfig(
        random_state=42,
        val_size=0.15,
        test_size=0.20,
        cv_folds=5,
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=23,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        max_categorical_cardinality=50,
        min_class_count=5,
        class_weight="balanced",
        n_jobs=4,
        early_stopping_rounds=100,
        early_stopping_min_delta=0.0,
        early_stopping_selection_folds=3,
        calibration_method="none",
        reliability_bins=10,
    )


def _dipper_recurrence_config() -> TrainingConfig:
    return TrainingConfig(
        random_state=42,
        val_size=0.15,
        test_size=0.20,
        cv_folds=5,
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=8,
        max_categorical_cardinality=50,
        min_class_count=20,
        class_weight="balanced",
        n_jobs=1,
        calibration_method="none",
        reliability_bins=8,
    )


def prepare_review_model(
    model_key: str,
    db_path: str | Path,
    *,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
) -> tuple[PreparedReviewModel, TrainingConfig]:
    """Prepare one current-DB training frame without writing model artifacts."""

    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model_key!r}; choose one of {MODEL_KEYS}")
    keep_secondary_json = model_key in {"eight_class", "dipper_recurrence", "eb", "microlensing"}
    table = load_review_population(db_path, keep_morphology_secondary_json=keep_secondary_json)

    if model_key == "eight_class":
        cache_path = (
            Path(recovery_feature_cache)
            if recovery_feature_cache is not None
            else default_recovery_feature_cache(db_path)
        )
        table = add_recovery_bounded_event_features(
            table,
            cache_path,
            workers=recovery_workers,
        )
        target, audit = _eight_class_labels(table)
        trainable = table[target].notna()
        features = _eight_class_features(table, trainable)
        probability_columns = tuple(f"prob_{label}" for label in EIGHT_CLASS_ORDER)
        positive_label = None
        config = _eight_class_config()
        score_context = (
            "candidate_id", "asas_sn_id", "lc_path", "status", "workflow_status",
            "event_class", "morphology_primary", "morphology_secondary", "physical_primary",
            "dip_run_count", "dip_is_single_event", "dipper_recurrence_class",
            "dipper_recurrence_evidence",
            "human_secondary_tags", target, "human_label_source", "interest_score",
            "catalog_match", "catalog_source", "gaia_var_class", "gaia_eb_period",
            "gaia_eb_morph", "vsx_class", "asassn_var_type", "ztf_var_type",
            "simbad_otype", "microlens_match", "microlens_catalog", "microlens_name",
            "microlens_te_days",
        )
    elif model_key == "dipper_recurrence":
        target, positive_label, audit = _dipper_recurrence_labels(table)
        trainable = table[target].notna()
        features, duplicate_aliases = _dipper_recurrence_features(table, trainable)
        audit["dropped_duplicate_feature_aliases"] = duplicate_aliases
        probability_columns = (f"prob_{positive_label}",)
        config = _dipper_recurrence_config()
        score_context = (
            "candidate_id", "asas_sn_id", "lc_path", "status", "workflow_status",
            "event_class", "morphology_primary", "morphology_secondary", "physical_primary",
            "human_dipper_recurrence_tags", target,
            "human_dipper_recurrence_label_source", "interest_score",
            "catalog_match", "catalog_source", "gaia_var_class", "vsx_class",
            "asassn_var_type", "ztf_var_type", "simbad_otype",
        )
    else:
        target, positive_label, _negative_label, audit = _binary_labels(table, model_key)
        trainable = table[target].notna()
        minimum = 30 if model_key == "eb" else 20
        features = _stats_plus_context_features(
            table,
            trainable,
            excluded_stats=MODEL_EXCLUDED_STATS,
            min_non_null=minimum,
        )
        features, duplicate_aliases = _drop_exact_duplicate_features(
            table.loc[trainable], features
        )
        audit["dropped_duplicate_feature_aliases"] = duplicate_aliases
        probability_columns = (f"prob_{positive_label}",)
        config = _binary_config(model_key)
        score_context = (
            "candidate_id", "asas_sn_id", "lc_path", "status", "workflow_status",
            "event_class", "morphology_primary", "morphology_secondary", "physical_primary",
            "interest_score", "catalog_match", "catalog_source", "vetting_likely_known",
            "simbad_otype", "vsx_class", "gaia_var_class", "microlens_match",
            "microlens_catalog", "microlens_name", "microlens_te_days", target,
            "human_microlensing_like", "microlensing_label_source",
        )

    model_input = table[["candidate_id", target, *features]].copy()
    counts = {
        str(key): int(value)
        for key, value in table.loc[trainable, target].value_counts().items()
    }
    return PreparedReviewModel(
        model_key=model_key,
        target_column=target,
        positive_label=positive_label,
        table=table,
        model_input=model_input,
        feature_columns=tuple(features),
        label_counts=counts,
        label_audit=audit,
        probability_columns=probability_columns,
        score_context_columns=tuple(column for column in score_context if column in table.columns),
    ), config


def _unreviewed_mask(scores: pd.DataFrame) -> pd.Series:
    status = _clean_text(scores, "status")
    workflow = _clean_text(scores, "workflow_status")
    event_class = _clean_text(scores, "event_class")
    return (
        status.isin(("", "unreviewed"))
        & workflow.isin(("", "unreviewed"))
        & event_class.isin(("", "unclassified"))
    )


def _write_gain_importance(result: Any, output_dir: Path) -> None:
    booster = result.model.booster_
    gain_importance = pd.DataFrame(
        {
            "feature": result.feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False, ignore_index=True)
    gain_importance.to_csv(output_dir / "feature_importance_gain.csv", index=False)


def train_review_model(
    model_key: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path | None = None,
    top_unreviewed_n: int = 500,
    recovery_feature_cache: str | Path | None = None,
    recovery_workers: int = 4,
) -> dict[str, Any]:
    """Fit and score one Review model, writing its complete core artifact set."""

    prepared, config = prepare_review_model(
        model_key,
        db_path,
        recovery_feature_cache=recovery_feature_cache,
        recovery_workers=recovery_workers,
    )
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_MODEL_DIRS[model_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_target_model(prepared.model_input, prepared.target_column, config=config)
    save_target_model(result, out_dir)
    _write_gain_importance(result, out_dir)

    predictions = score_target_model(out_dir, prepared.model_input)
    prediction_columns = [
        "candidate_id", "y_pred", "prediction_confidence", *prepared.probability_columns
    ]
    scores = prepared.table[list(prepared.score_context_columns)].merge(
        predictions[prediction_columns], on="candidate_id", how="left"
    )
    unreviewed = scores.loc[_unreviewed_mask(scores)].copy()
    summary: dict[str, Any] = {
        "model_key": model_key,
        "db_path": str(Path(db_path)),
        "model_dir": str(out_dir),
        "target_column": prepared.target_column,
        "positive_label": prepared.positive_label,
        "n_candidates": int(len(prepared.table)),
        "n_trainable": int(prepared.model_input[prepared.target_column].notna().sum()),
        "class_counts": result.class_counts,
        "n_features": int(result.n_features),
        "feature_columns": list(result.feature_columns),
        "holdout_metrics": result.holdout_metrics,
        "label_audit": prepared.label_audit,
        "warning": "Class-balanced outputs are uncalibrated ranking scores, not population probabilities.",
    }
    if model_key == "eight_class":
        resolved_recovery_cache = (
            Path(recovery_feature_cache)
            if recovery_feature_cache is not None
            else default_recovery_feature_cache(db_path)
        )
        summary["recovery_feature_cache"] = str(
            resolved_recovery_cache.expanduser().resolve()
        )
        summary["recovery_feature_schema_version"] = (
            RECOVERY_FEATURE_SCHEMA_VERSION
        )

    if model_key == "eight_class":
        matrix = scores[list(prepared.probability_columns)].to_numpy(dtype=float)
        sorted_matrix = np.sort(matrix, axis=1)
        scores["score_margin"] = sorted_matrix[:, -1] - sorted_matrix[:, -2]
        clipped = np.clip(matrix, 1e-12, 1.0)
        scores["prediction_entropy"] = -(
            clipped * np.log(clipped)
        ).sum(axis=1) / math.log(len(prepared.probability_columns))
        scores["is_human_unreviewed"] = _unreviewed_mask(scores)
        unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
        scores.to_parquet(out_dir / "all_candidates_eight_class_scores.parquet", index=False)
        queue_frames = []
        for label, probability_column in zip(EIGHT_CLASS_ORDER, prepared.probability_columns):
            queue = unreviewed.sort_values(probability_column, ascending=False).head(top_unreviewed_n).copy()
            queue["queue_class"] = label
            queue["rank_within_class"] = np.arange(1, len(queue) + 1)
            queue.to_csv(out_dir / f"top{top_unreviewed_n}_unreviewed_{label}.csv", index=False)
            queue_frames.append(queue)
        pd.concat(queue_frames, ignore_index=True).to_csv(
            out_dir / "top_unreviewed_by_class.csv", index=False
        )
        unreviewed.sort_values(["prediction_entropy", "score_margin"], ascending=[False, True]).head(500).to_csv(
            out_dir / "most_ambiguous_unreviewed.csv", index=False
        )
    else:
        probability_column = prepared.probability_columns[0]
        scores.to_parquet(out_dir / "all_candidates_scores.parquet", index=False)
        queue = unreviewed.sort_values(probability_column, ascending=False).head(top_unreviewed_n).copy()
        queue["rank"] = np.arange(1, len(queue) + 1)
        queue.to_csv(out_dir / f"top{top_unreviewed_n}_unreviewed.csv", index=False)
        unreviewed.sort_values(probability_column, ascending=False).head(1000).to_csv(
            out_dir / "high_priority_review_queue.csv", index=False
        )

    (out_dir / "label_audit.json").write_text(
        json.dumps(prepared.label_audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Trained {model_key} model: {len(prepared.table):,} candidates, "
        f"{summary['n_trainable']:,} labeled rows, {result.n_features} features -> {out_dir}"
    )
    return summary


def build_parser(model_key: str) -> argparse.ArgumentParser:
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model_key!r}")
    parser = argparse.ArgumentParser(
        description=f"Train and score the July 1 Review {model_key} LightGBM model."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIRS[model_key])
    parser.add_argument("--top-unreviewed", type=int, default=500)
    parser.add_argument(
        "--recovery-feature-cache",
        type=Path,
        default=None,
        help=(
            "Versioned candidate-independent recovery-morphology cache. "
            "The default is derived from --db-path."
        ),
    )
    parser.add_argument(
        "--recovery-workers",
        type=int,
        default=4,
        help="Parallel workers used only while filling the recovery-feature cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate labels/features and print the contract without fitting; "
            "the recovery-feature cache may still be filled."
        ),
    )
    return parser


def script_main(model_key: str, argv: Iterable[str] | None = None) -> int:
    """Entry point used by the four standalone ``scripts/train_*.py`` files."""

    args = build_parser(model_key).parse_args(list(argv) if argv is not None else None)
    if args.top_unreviewed < 1:
        raise SystemExit("--top-unreviewed must be positive")
    if args.dry_run:
        prepared, _config = prepare_review_model(
            model_key,
            args.db_path,
            recovery_feature_cache=args.recovery_feature_cache,
            recovery_workers=args.recovery_workers,
        )
        print(
            json.dumps(
                {
                    "model_key": model_key,
                    "db_path": str(args.db_path),
                    "n_candidates": len(prepared.table),
                    "n_trainable": int(prepared.model_input[prepared.target_column].notna().sum()),
                    "class_counts": prepared.label_counts,
                    "n_features": len(prepared.feature_columns),
                    "label_audit": prepared.label_audit,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0
    train_review_model(
        model_key,
        db_path=args.db_path,
        output_dir=args.output_dir,
        top_unreviewed_n=args.top_unreviewed,
        recovery_feature_cache=args.recovery_feature_cache,
        recovery_workers=args.recovery_workers,
    )
    return 0


__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_MODEL_DIRS",
    "MODEL_KEYS",
    "PreparedReviewModel",
    "build_parser",
    "is_usable_model_feature",
    "load_review_population",
    "prepare_review_model",
    "script_main",
    "train_review_model",
]
