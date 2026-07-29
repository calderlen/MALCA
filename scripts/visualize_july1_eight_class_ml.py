#!/usr/bin/env python
# coding: utf-8

"""Write July 1 eight-class ML diagnostics and light-curve figures to results."""

# # July 1 Eight-Class ML Separability
# 
# This standalone script evaluates the populated July 1 `stats_*` features plus a curated native-light-curve block of periodicity, dip, and jump measurements for separating eight human review classes. Retrain with `python scripts/train_july1_eight_class_ml.py` from the repository root.
# 
# - `dipper`: review `event_class` is `dipper` or `mixed_dip_and_burst`
# - `eclipsing_binary_like`: human secondary morphology is eclipsing/binary-like
# - `long_term_variable`: review morphology is `long_term_trend`
# - `long_period_variable`: review morphology is `long_period_variability`
# - `microlensing`: review `event_class` is `microlensing` or the human secondary tag is `possible_microlensing_event`
# - `quasi_periodic`: review `event_class` is `quasi_periodic`
# - `brightening_event`: review `event_class` is `brightening_event`, excluding objects already assigned to microlensing
# - `artifact_or_nonvariable`: review morphology is either `artifact_or_bad_photometry` or `nonvariable_or_low_snr`
# 
# Gaia, VSX, ASAS-SN catalog classes, SIMBAD types, and microlensing catalog matches are excluded from both labels and features. Sampling timestamps, duplicate scalar summaries, and threshold-only pipeline decisions are also excluded. External catalog fields appear only in the final comparison section. The reported class scores are class-balanced ranking scores, not calibrated population probabilities.

# In[1]:


from __future__ import annotations

import itertools
import json
import math
import re
import sqlite3
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from malca.io.lightcurve_io import load_lightcurve_df, stable_camera_color
from malca.meta_analysis.ml.candidate_features import (
    add_next_iteration_context_features,
    add_recovery_bounded_event_features,
)
from malca.meta_analysis.ml.feature_policy import (
    STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
)
from malca.meta_analysis.ml.plotting import (
    FEATURE_IMPORTANCE_TOP_N,
    apply_ml_plot_style,
    plot_fraction_count_heatmap,
)
from malca.meta_analysis.ml.review_lightgbm import (
    ASTROPHYSICAL_CONTEXT_FEATURES,
    TrainingConfig,
    add_astrophysical_context_features,
    fit_classifier_with_inner_early_stopping,
    load_target_model,
    score_target_model,
    transform_features,
)
from malca.review.other_eb_triage import EB_REGEX

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning)

apply_ml_plot_style()
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 120)

_report_index = 0


def Markdown(value: object) -> str:
    return str(value)


def Image(*_args: object, **_kwargs: object) -> None:
    return None


def display(value: object, *_args: object, **_kwargs: object) -> None:
    """Persist notebook-style tabular/text output beside the rendered figures."""

    global _report_index
    _report_index += 1
    report_dir = FIG_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = report_dir / f"report_{_report_index:02d}"
    if isinstance(value, pd.DataFrame):
        value.to_csv(stem.with_suffix(".csv"), index=True)
    elif isinstance(value, pd.Series):
        value.to_frame().to_csv(stem.with_suffix(".csv"), index=True)
    elif isinstance(value, str):
        stem.with_suffix(".md").write_text(value + "\n", encoding="utf-8")


plt.show = lambda *_args, **_kwargs: plt.close("all")


def find_repo_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "malca").is_dir() and (path / "output" / "runs").is_dir():
            return path
    raise FileNotFoundError("Could not locate the MALCA repo root")


REPO_ROOT = find_repo_root(Path.cwd())
RUN_DIR = REPO_ROOT / "output" / "runs" / "dat3-full-extended_2026-07-01-v4"
BASE_DIR = RUN_DIR / "results" / "eight_class_ml_separability"
FEATURE_SET_NAME = "stats_plus_periodicity_dip_jump_context"
MODEL_DIR = BASE_DIR / FEATURE_SET_NAME
TRAINING_DB_SNAPSHOT = MODEL_DIR / "training_review_snapshot.db"
DB_PATH = (
    TRAINING_DB_SNAPSHOT
    if TRAINING_DB_SNAPSHOT.exists()
    else RUN_DIR / "review" / "review.db"
)
FIG_DIR = MODEL_DIR / "figures"
LIGHTCURVE_DIR = RUN_DIR / "bundle_assets" / "lightcurves"

TARGET_COLUMN = "human_eight_class_label"
CLASS_ORDER = (
    "dipper",
    "eclipsing_binary_like",
    "long_term_variable",
    "microlensing",
    "long_period_variable",
    "quasi_periodic",
    "brightening_event",
    "artifact_or_nonvariable",
)
EXTERNAL_HINT_CLASSES = (
    "dipper",
    "eclipsing_binary_like",
    "long_period_variable",
    "microlensing",
)
CLASS_DISPLAY = {
    "dipper": "Dipper",
    "eclipsing_binary_like": "EB",
    "long_term_variable": "LTV",
    "long_period_variable": "LPV",
    "microlensing": "Microlensing",
    "quasi_periodic": "Quasi-periodic",
    "brightening_event": "Brightening",
    "artifact_or_nonvariable": "Artifact / nonvariable",
}
CLASS_COLORS = {
    "dipper": "#d95f02",
    "eclipsing_binary_like": "#1b9e77",
    "long_term_variable": "#7570b3",
    "long_period_variable": "#66a61e",
    "microlensing": "#e7298a",
    "quasi_periodic": "#1f78b4",
    "brightening_event": "#e6ab02",
    "artifact_or_nonvariable": "#666666",
}

REBUILD_GALLERIES = False
RANDOM_STATE = 42
CV_FOLDS = 5
MAX_BOOSTING_ROUNDS = 2500
EARLY_STOPPING_ROUNDS = 100
EARLY_STOPPING_SELECTION_FOLDS = 3
PERMUTATION_REPEATS = 8
TOP_PER_CLASS_N = 100
HARD_EXAMPLE_N = 40
PER_PAGE = 20

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
for stale_png in FIG_DIR.rglob("*.png"):
    stale_png.unlink()
report_dir = FIG_DIR / "reports"
if report_dir.exists():
    for stale_report in report_dir.glob("report_*"):
        if stale_report.is_file() and stale_report.suffix in {".csv", ".md"}:
            stale_report.unlink()
print(f"Review DB: {DB_PATH}")
print(f"Model directory: {MODEL_DIR}")


# ## Load The July 1 Review Population

# In[2]:


if not DB_PATH.exists():
    raise FileNotFoundError(DB_PATH)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def read_table_without_payloads(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    schema = pd.read_sql_query(f"PRAGMA table_info({quote_identifier(table_name)})", conn)
    columns = [
        str(name)
        for name in schema["name"]
        if (not str(name).endswith("_json") or str(name) == "morphology_secondary_json")
        and str(name) not in {"payload_json", "legacy_review_json", "notes"}
    ]
    select_list = ", ".join(quote_identifier(column) for column in columns)
    return pd.read_sql_query(f"SELECT {select_list} FROM {quote_identifier(table_name)}", conn)


with sqlite3.connect(f"file:{DB_PATH.resolve()}?mode=ro", uri=True) as conn:
    candidates = read_table_without_payloads(conn, "candidates")
    reviews = read_table_without_payloads(conn, "reviews")

candidates["candidate_id"] = candidates["candidate_id"].astype(str)
reviews["candidate_id"] = reviews["candidate_id"].astype(str)
table = candidates.merge(reviews, on="candidate_id", how="left", suffixes=("", "_review"))
table = table.loc[:, ~table.columns.duplicated()].copy()
table = add_astrophysical_context_features(table)
table = add_next_iteration_context_features(table)
training_snapshot = json.loads((MODEL_DIR / "training_snapshot.json").read_text())
recovery_feature_cache = Path(training_snapshot["recovery_feature_cache"])
table = add_recovery_bounded_event_features(
    table,
    recovery_feature_cache,
    workers=1,
    compute_missing=False,
)
stats_columns = [column for column in table.columns if column.startswith("stats_")]

display(
    pd.DataFrame(
        {
            "surface": ["all candidates", "review rows", "stats_* columns"],
            "count": [len(candidates), len(reviews), len(stats_columns)],
        }
    )
)
display(reviews["event_class"].fillna("<missing>").value_counts().rename_axis("event_class").reset_index(name="n"))


# ## Build Four Mutually Exclusive Human Labels

# In[3]:


def clean_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def parse_secondary_tags(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return {str(item).strip() for item in parsed if str(item).strip()}
    return {part.strip() for part in text.split("|") if part.strip()}


EB_REVIEW_TAGS = {
    "eclipsing_like",
    "detached_binary_like",
    "semi_detached_binary_like",
    "contact_binary_like",
    "ellipsoidal_like",
    "heartbeat_like",
}

secondary_scalar = clean_text(table, "morphology_secondary")
secondary_sets = table.get(
    "morphology_secondary_json",
    pd.Series("", index=table.index),
).map(parse_secondary_tags)
secondary_sets = pd.Series(
    [tags | ({scalar} if scalar else set()) for tags, scalar in zip(secondary_sets, secondary_scalar)],
    index=table.index,
)
table["human_secondary_tags"] = secondary_sets.map(lambda values: "|".join(sorted(values)))

event_class = clean_text(table, "event_class")
morphology_primary = clean_text(table, "morphology_primary")
physical_primary = clean_text(table, "physical_primary")
status = clean_text(table, "status")
workflow = clean_text(table, "workflow_status")
reviewed = (status.ne("") & status.ne("unreviewed")) | (workflow.ne("") & workflow.ne("unreviewed"))

microlensing_mask = reviewed & (
    event_class.eq("microlensing")
    | secondary_sets.map(lambda values: "possible_microlensing_event" in values)
)

class_masks = {
    "dipper": reviewed & event_class.isin({"dipper", "mixed_dip_and_burst"}),
    "eclipsing_binary_like": reviewed
    & (
        secondary_sets.map(lambda values: bool(values & EB_REVIEW_TAGS))
        | physical_primary.eq("eclipsing_or_geometric_binary")
    ),
    "long_term_variable": reviewed & morphology_primary.eq("long_term_trend"),
    "long_period_variable": reviewed
    & morphology_primary.eq("long_period_variability"),
    "microlensing": microlensing_mask,
    "quasi_periodic": reviewed & event_class.eq("quasi_periodic"),
    "brightening_event": reviewed & event_class.eq("brightening_event") & ~microlensing_mask,
    "artifact_or_nonvariable": reviewed
    & morphology_primary.isin(
        {"artifact_or_bad_photometry", "nonvariable_or_low_snr"}
    ),
}
membership_count = sum(mask.astype(int) for mask in class_masks.values())
overlap_rows = table.loc[membership_count.gt(1), ["candidate_id", "event_class", "morphology_secondary", "physical_primary"]]
if not overlap_rows.empty:
    display(overlap_rows)
    raise ValueError(f"Found {len(overlap_rows)} reviewed candidates in more than one target class")

table[TARGET_COLUMN] = pd.Series(pd.NA, index=table.index, dtype="string")
for label, mask in class_masks.items():
    table.loc[mask, TARGET_COLUMN] = label

table["human_label_source"] = "unlabeled_for_eight_class_model"
table.loc[class_masks["dipper"], "human_label_source"] = "human_event_class_dipper"
table.loc[class_masks["eclipsing_binary_like"], "human_label_source"] = "human_secondary_eb_morphology"
table.loc[class_masks["long_term_variable"], "human_label_source"] = "human_morphology_long_term_trend"
table.loc[class_masks["long_period_variable"], "human_label_source"] = "human_morphology_long_period_variability"
table.loc[class_masks["quasi_periodic"], "human_label_source"] = "human_event_class_quasi_periodic"
table.loc[class_masks["brightening_event"], "human_label_source"] = "human_event_class_brightening_non_microlensing"
rejection_mask = class_masks["artifact_or_nonvariable"]
table.loc[
    rejection_mask & morphology_primary.eq("artifact_or_bad_photometry"),
    "human_label_source",
] = "human_morphology_artifact_or_bad_photometry"
table.loc[
    rejection_mask & morphology_primary.eq("nonvariable_or_low_snr"),
    "human_label_source",
] = "human_morphology_nonvariable_or_low_snr"
micro_event = class_masks["microlensing"] & event_class.eq("microlensing")
micro_tag = class_masks["microlensing"] & secondary_sets.map(lambda values: "possible_microlensing_event" in values)
micro_source = pd.Series(
    np.select(
        [micro_event & micro_tag, micro_event, micro_tag],
        ["human_event_and_morphology_microlensing", "human_event_class_microlensing", "human_morphology_microlensing"],
        default="human_microlensing",
    ),
    index=table.index,
)
table.loc[class_masks["microlensing"], "human_label_source"] = micro_source.loc[class_masks["microlensing"]]

trainable = table[TARGET_COLUMN].notna()
label_counts = table.loc[trainable, TARGET_COLUMN].value_counts().reindex(CLASS_ORDER, fill_value=0)
if int(label_counts.min()) < 5:
    raise ValueError(f"An eight-class target has fewer than 5 examples: {label_counts.to_dict()}")

display(label_counts.rename_axis("human_label").reset_index(name="n"))
display(table.loc[trainable, [TARGET_COLUMN, "human_label_source"]].value_counts().rename("n").reset_index())
print(f"Trainable eight-class reviewed rows: {int(trainable.sum()):,}")
print(f"Cross-class overlaps: {int(membership_count.gt(1).sum()):,}")
print(f"Other reviewed rows excluded: {int((reviewed & ~trainable).sum()):,}")


# ## Build External Catalog Context Without Using It For Training

# In[4]:


def truthy_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.fillna("").astype(str).str.strip().str.lower()
    return numeric.fillna(0).ne(0) | text.isin({"true", "yes", "y", "t"})


catalog_columns = [
    "gaia_var_class",
    "vsx_class",
    "asassn_var_type",
    "ztf_var_type",
    "simbad_otype",
]
catalog_text = pd.Series("", index=table.index, dtype="object")
for column in catalog_columns:
    catalog_text = catalog_text + "|" + clean_text(table, column).str.upper()

external_eb = (
    catalog_text.str.contains(EB_REGEX, na=False)
    | pd.to_numeric(table.get("gaia_eb_period", pd.Series(np.nan, index=table.index)), errors="coerce").notna()
    | clean_text(table, "gaia_eb_morph").ne("")
)
ltv_token_pattern = re.compile(r"(?:^|[^A-Z0-9])(?:LPV|MIRA|M|SR|SRA|SRB|SRC|SRD|L|LB|LC)(?:$|[^A-Z0-9])")
external_lpv = catalog_text.str.contains(ltv_token_pattern, na=False)
external_microlensing = (
    truthy_series(table, "microlens_match")
    | clean_text(table, "microlens_name").ne("")
    | clean_text(table, "microlens_catalog").ne("")
    | catalog_text.str.contains(r"MICROLENS|OGLE[-_ ]?BLG|MOA[-_ ]?BLG", regex=True, na=False)
)
external_dipper = catalog_text.str.contains(r"DIPPER|UXOR|UX[ _-]?ORI", regex=True, na=False)

external_masks = {
    "dipper": external_dipper,
    "eclipsing_binary_like": external_eb,
    "long_period_variable": external_lpv,
    "microlensing": external_microlensing,
}
for label, mask in external_masks.items():
    table[f"external_hint_{label}"] = mask.astype(bool)

external_sets = []
for row_index in table.index:
    labels = [label for label in EXTERNAL_HINT_CLASSES if bool(external_masks[label].loc[row_index])]
    external_sets.append(labels)
table["external_hint_labels"] = ["|".join(labels) for labels in external_sets]
table["external_catalog_context"] = [
    labels[0] if len(labels) == 1 else ("multiple" if len(labels) > 1 else "none")
    for labels in external_sets
]

external_coverage = pd.DataFrame(
    {
        "external_hint": list(EXTERNAL_HINT_CLASSES),
        "all_candidates": [int(external_masks[label].sum()) for label in EXTERNAL_HINT_CLASSES],
        "human_labeled": [int((external_masks[label] & trainable).sum()) for label in EXTERNAL_HINT_CLASSES],
    }
)
display(external_coverage)
display(table.loc[trainable, [TARGET_COLUMN, "external_catalog_context"]].value_counts().rename("n").reset_index())


# ## Select Leakage-Controlled Compute-Stats Features

# In[5]:


MODEL_EXCLUDED_STATS = {
    *STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
    "stats_variability_lomb_scargle_best_period_days",
    "stats_mhps_non_zero",
    "stats_harmonics_period",
    "stats_photometry_band_alignment",
}

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
if len(ADDITIONAL_LC_FEATURES) != len(set(ADDITIONAL_LC_FEATURES)):
    raise ValueError("The curated additional feature list contains duplicates")


def is_usable_model_feature(
    series: pd.Series,
    *,
    min_non_null: int = 20,
    max_cardinality: int = 50,
) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return int(numeric.notna().sum()) >= min_non_null and numeric.nunique(dropna=True) > 1
    values = series.dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    return int(values.size) >= min_non_null and 1 < int(values.nunique()) <= max_cardinality


missing_additional_features = [column for column in ADDITIONAL_LC_FEATURES if column not in table.columns]
if missing_additional_features:
    raise KeyError(f"Requested additional features are missing: {missing_additional_features}")
unusable_additional_features = [
    column
    for column in ADDITIONAL_LC_FEATURES
    if not is_usable_model_feature(table.loc[trainable, column])
]
if unusable_additional_features:
    if not (MODEL_DIR / "model.joblib").exists():
        raise ValueError(f"Requested additional features are unusable: {unusable_additional_features}")
    display(
        Markdown(
            "**Saved-model reuse warning:** these requested features are currently sparse or "
            f"constant: `{unusable_additional_features}`. They remain in the saved model schema, "
            "but a new model should not be trained until their coverage is restored."
        )
    )

stats_feature_columns = [
    column
    for column in stats_columns
    if column not in MODEL_EXCLUDED_STATS
    and is_usable_model_feature(table.loc[trainable, column])
]
stats_feature_columns = sorted(
    stats_feature_columns,
    key=lambda column: (-int(table.loc[trainable, column].notna().sum()), column),
)
unusable_context_features = [
    column
    for column in ASTROPHYSICAL_CONTEXT_FEATURES
    if not is_usable_model_feature(table.loc[trainable, column])
]
if unusable_context_features:
    raise ValueError(f"Requested astrophysical-context features are unusable: {unusable_context_features}")

feature_columns = [
    *stats_feature_columns,
    *ADDITIONAL_LC_FEATURES,
    *ASTROPHYSICAL_CONTEXT_FEATURES,
]
model_input = table[["candidate_id", TARGET_COLUMN, *feature_columns]].copy()

print(f"Selected stats_* features: {len(stats_feature_columns):,}")
print(f"Selected additional native-light-curve features: {len(ADDITIONAL_LC_FEATURES):,}")
print(f"Selected astrophysical-context features: {len(ASTROPHYSICAL_CONTEXT_FEATURES):,}")
print(f"Total model features: {len(feature_columns):,}")
print(f"Explicitly excluded stats fields: {sorted(MODEL_EXCLUDED_STATS)}")
display(
    pd.DataFrame(
        {
            "feature": (*ADDITIONAL_LC_FEATURES, *ASTROPHYSICAL_CONTEXT_FEATURES),
            "feature_group": (
                ["periodicity"] * len(ADDITIONAL_PERIODICITY_FEATURES)
                + ["dip"] * len(ADDITIONAL_DIP_FEATURES)
                + ["jump"] * len(ADDITIONAL_JUMP_FEATURES)
                + ["astrophysical_context"] * len(ASTROPHYSICAL_CONTEXT_FEATURES)
            ),
            "labeled_non_null": [
                int(table.loc[trainable, column].notna().sum())
                for column in (*ADDITIONAL_LC_FEATURES, *ASTROPHYSICAL_CONTEXT_FEATURES)
            ],
            "all_non_null": [
                int(table[column].notna().sum())
                for column in (*ADDITIONAL_LC_FEATURES, *ASTROPHYSICAL_CONTEXT_FEATURES)
            ],
        }
    )
)


# ## Load The Saved Eight-Class LightGBM

# In[6]:


model_path = MODEL_DIR / "model.joblib"
if not model_path.exists():
    raise FileNotFoundError(
        f"No saved model at {model_path}. Run scripts/train_july1_eight_class_ml.py from the repository root."
    )
needs_training = False
print("Using the saved eight-class model. Retrain with scripts/train_july1_eight_class_ml.py.")

bundle = load_target_model(MODEL_DIR)
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
label_classes = list(bundle["label_classes"])
probability_columns = list(bundle["probability_columns"])
if set(label_classes) != set(CLASS_ORDER):
    raise ValueError(f"Saved model classes do not match the intended eight classes: {label_classes}")
saved_feature_columns = list(bundle["feature_columns"])
requested_feature_columns = list(feature_columns)
missing_saved_features = sorted(set(saved_feature_columns) - set(table.columns))
if missing_saved_features:
    raise ValueError(
        "Saved model expects features missing from the current Review table: "
        f"{missing_saved_features[:10]}"
    )
feature_schema_drift = saved_feature_columns != requested_feature_columns
if feature_schema_drift:
    feature_columns = saved_feature_columns
    model_input = table[["candidate_id", TARGET_COLUMN, *feature_columns]].copy()
    display(
        Markdown(
            "**Saved-model feature-schema drift:** visualization uses the saved model's "
            f"{len(saved_feature_columns)} predictors instead of the {len(requested_feature_columns)} "
            "currently selected predictors. Retrain to make the artifact current."
        )
    )

current_counts = {str(key): int(value) for key, value in label_counts.items()}
saved_counts = {str(key): int(value) for key, value in metadata["class_counts"].items()}
if current_counts != saved_counts:
    display(Markdown(f"**Label drift warning:** saved counts are `{saved_counts}` but current counts are `{current_counts}`. Retrain before treating scores as current."))

label_audit = {
    "n_candidates": int(len(table)),
    "n_reviewed": int(reviewed.sum()),
    "n_trainable": int(trainable.sum()),
    "class_counts": current_counts,
    "n_overlap_rows": int(membership_count.gt(1).sum()),
    "n_other_reviewed_excluded": int((reviewed & ~trainable).sum()),
    "saved_feature_count": len(saved_feature_columns),
    "current_requested_feature_count": len(requested_feature_columns),
    "feature_schema_drift": feature_schema_drift,
}
(BASE_DIR / "label_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True) + "\n")
display(pd.DataFrame({"value": [metadata["n_rows"], metadata["n_features"], metadata["class_counts"]]}, index=["training rows", "features", "class counts"]))


# ## Score Every Candidate And Export Per-Class Queues

# In[7]:


prediction_frame = score_target_model(MODEL_DIR, model_input)
prediction_columns = ["candidate_id", "y_pred", "prediction_confidence", *probability_columns]

context_columns = [
    "candidate_id", "asas_sn_id", "lc_path", "status", "workflow_status", "event_class",
    "morphology_primary", "morphology_secondary", "physical_primary", "human_secondary_tags",
    TARGET_COLUMN, "human_label_source", "interest_score", "catalog_match", "catalog_source",
    "gaia_var_class", "gaia_eb_period", "gaia_eb_morph", "vsx_class", "asassn_var_type",
    "ztf_var_type", "simbad_otype", "microlens_match", "microlens_catalog", "microlens_name",
    "microlens_te_days", "external_hint_labels", "external_catalog_context",
    *[f"external_hint_{label}" for label in EXTERNAL_HINT_CLASSES],
]
context_columns = [column for column in context_columns if column in table.columns]
scores = table[context_columns].merge(prediction_frame[prediction_columns], on="candidate_id", how="left")

probability_matrix = scores[probability_columns].to_numpy(dtype=float)
sorted_probability = np.sort(probability_matrix, axis=1)
scores["score_margin"] = sorted_probability[:, -1] - sorted_probability[:, -2]
clipped_probability = np.clip(probability_matrix, 1e-12, 1.0)
scores["prediction_entropy"] = -(clipped_probability * np.log(clipped_probability)).sum(axis=1) / math.log(len(label_classes))

score_status = clean_text(scores, "status")
score_workflow = clean_text(scores, "workflow_status")
score_event = clean_text(scores, "event_class")
unreviewed_mask = (
    score_status.isin(("", "unreviewed"))
    & score_workflow.isin(("", "unreviewed"))
    & score_event.isin(("", "unclassified"))
)
scores["is_human_unreviewed"] = unreviewed_mask
unreviewed_scores = scores.loc[unreviewed_mask].copy()

top_queue_frames = []
for label, probability_column in zip(label_classes, probability_columns):
    queue = unreviewed_scores.sort_values(probability_column, ascending=False).head(TOP_PER_CLASS_N).copy()
    queue["queue_class"] = label
    queue["rank_within_class"] = np.arange(1, len(queue) + 1)
    queue.to_csv(MODEL_DIR / f"top{TOP_PER_CLASS_N}_unreviewed_{label}.csv", index=False)
    top_queue_frames.append(queue)
top_queues = pd.concat(top_queue_frames, ignore_index=True)
top_queues.to_csv(MODEL_DIR / "top_unreviewed_by_class.csv", index=False)

ambiguous_queue = unreviewed_scores.sort_values(
    ["prediction_entropy", "score_margin"],
    ascending=[False, True],
).head(500)
ambiguous_queue.to_csv(MODEL_DIR / "most_ambiguous_unreviewed.csv", index=False)
scores.to_parquet(MODEL_DIR / "all_candidates_eight_class_scores.parquet", index=False)

print(f"Scored candidates: {len(scores):,}")
print(f"Human-unreviewed candidates: {len(unreviewed_scores):,}")
display(unreviewed_scores["y_pred"].value_counts().reindex(CLASS_ORDER, fill_value=0).rename_axis("predicted_class").reset_index(name="n"))
display(top_queues[["queue_class", "rank_within_class", "candidate_id", "y_pred", "prediction_confidence", "score_margin", "external_catalog_context"]].groupby("queue_class", sort=False).head(8))


# ## Honest Held-Out Performance

# In[8]:


test_predictions = pd.read_parquet(MODEL_DIR / "test_predictions.parquet")
heldout_true = test_predictions["y_true"].astype(str)
heldout_pred = test_predictions["y_pred"].astype(str)
heldout_confusion = confusion_matrix(heldout_true, heldout_pred, labels=CLASS_ORDER)

report = classification_report(
    heldout_true,
    heldout_pred,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0,
)
heldout_rows = []
for label in CLASS_ORDER:
    probability_column = probability_columns[label_classes.index(label)]
    binary_true = heldout_true.eq(label).astype(int)
    heldout_rows.append(
        {
            "class": label,
            "support": int(binary_true.sum()),
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "roc_auc_ovr": roc_auc_score(binary_true, test_predictions[probability_column]),
            "average_precision_ovr": average_precision_score(binary_true, test_predictions[probability_column]),
        }
    )
heldout_per_class = pd.DataFrame(heldout_rows)
heldout_per_class.to_csv(MODEL_DIR / "heldout_per_class_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 6.5))
plot_fraction_count_heatmap(
    pd.DataFrame(heldout_confusion, index=CLASS_ORDER, columns=CLASS_ORDER),
    cmap="Blues",
    ax=ax,
)
ax.set(title="Held-out confusion", xlabel="Predicted", ylabel="Human label")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "heldout_confusion.pdf", dpi=180)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for label in CLASS_ORDER:
    probability_column = probability_columns[label_classes.index(label)]
    binary_true = heldout_true.eq(label).astype(int)
    fpr, tpr, _ = roc_curve(binary_true, test_predictions[probability_column])
    precision, recall, _ = precision_recall_curve(binary_true, test_predictions[probability_column])
    auc_value = roc_auc_score(binary_true, test_predictions[probability_column])
    ap_value = average_precision_score(binary_true, test_predictions[probability_column])
    axes[0].plot(fpr, tpr, lw=2, color=CLASS_COLORS[label], label=f"{CLASS_DISPLAY[label]} AUC={auc_value:.3f}")
    axes[1].plot(recall, precision, lw=2, color=CLASS_COLORS[label], label=f"{CLASS_DISPLAY[label]} AP={ap_value:.3f}")
axes[0].plot([0, 1], [0, 1], ls="--", color="0.5")
axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out one-vs-rest ROC")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out one-vs-rest precision-recall")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "heldout_ovr_curves.pdf", dpi=180)
plt.show()

display(heldout_per_class)
display(Markdown(f"**Held-out summary:** accuracy={accuracy_score(heldout_true, heldout_pred):.3f}, balanced accuracy={balanced_accuracy_score(heldout_true, heldout_pred):.3f}, macro F1={f1_score(heldout_true, heldout_pred, average='macro'):.3f}."))


# ## Five-Fold Out-Of-Fold Separability

# In[9]:


labeled_model_input = model_input.loc[trainable].reset_index(drop=True)
X_labeled = transform_features(
    labeled_model_input,
    feature_columns=bundle["feature_columns"],
    categorical_maps=bundle["categorical_maps"],
)
y_labels = labeled_model_input[TARGET_COLUMN].astype(str).to_numpy()
class_to_index = {label: index for index, label in enumerate(label_classes)}
y_encoded = np.array([class_to_index[label] for label in y_labels], dtype=int)

splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_probability = np.zeros((len(labeled_model_input), len(label_classes)), dtype=float)
oof_fold = np.zeros(len(labeled_model_input), dtype=int)
fold_metric_rows = []
fold_iteration_rows = []

for fold, (fit_index, validation_index) in enumerate(splitter.split(X_labeled, y_encoded), start=1):
    fold_model, inner_best_iterations = fit_classifier_with_inner_early_stopping(
        X_labeled.iloc[fit_index],
        y_encoded[fit_index],
        config=TrainingConfig(**metadata["config"]),
        n_classes=len(label_classes),
        random_state=RANDOM_STATE + fold,
    )
    selected_iteration = int(fold_model.n_estimators_)
    fold_iteration_rows.append(
        {
            "fold": fold,
            "selected_iteration": selected_iteration,
            "inner_best_iterations": "|".join(map(str, inner_best_iterations)),
        }
    )
    raw_probability = np.asarray(fold_model.predict_proba(X_labeled.iloc[validation_index]), dtype=float)
    aligned_probability = np.zeros((len(validation_index), len(label_classes)), dtype=float)
    for source_index, encoded_class in enumerate(fold_model.classes_):
        aligned_probability[:, int(encoded_class)] = raw_probability[:, source_index]
    aligned_probability = aligned_probability / np.maximum(aligned_probability.sum(axis=1, keepdims=True), 1e-12)
    predicted = aligned_probability.argmax(axis=1)
    oof_probability[validation_index] = aligned_probability
    oof_fold[validation_index] = fold

    for class_index, label in enumerate(label_classes):
        binary_true = (y_encoded[validation_index] == class_index).astype(int)
        precision_value, recall_value, f1_value, support_value = precision_recall_fscore_support(
            y_encoded[validation_index],
            predicted,
            labels=[class_index],
            average=None,
            zero_division=0,
        )
        fold_metric_rows.append(
            {
                "fold": fold,
                "class": label,
                "selected_iteration": selected_iteration,
                "support": int(support_value[0]),
                "precision": float(precision_value[0]),
                "recall": float(recall_value[0]),
                "f1": float(f1_value[0]),
                "roc_auc_ovr": roc_auc_score(binary_true, aligned_probability[:, class_index]),
                "average_precision_ovr": average_precision_score(binary_true, aligned_probability[:, class_index]),
            }
        )

pd.DataFrame(fold_iteration_rows).to_csv(MODEL_DIR / "oof_early_stopping_iterations.csv", index=False)
oof_predicted_index = oof_probability.argmax(axis=1)
oof_predicted_labels = np.array([label_classes[index] for index in oof_predicted_index])
sorted_oof_probability = np.sort(oof_probability, axis=1)
clipped_oof_probability = np.clip(oof_probability, 1e-12, 1.0)

oof = pd.DataFrame(
    {
        "candidate_id": labeled_model_input["candidate_id"].astype(str),
        "human_label": y_labels,
        "oof_predicted_label": oof_predicted_labels,
        "oof_fold": oof_fold,
        "oof_confidence": oof_probability.max(axis=1),
        "oof_margin": sorted_oof_probability[:, -1] - sorted_oof_probability[:, -2],
        "oof_entropy": -(clipped_oof_probability * np.log(clipped_oof_probability)).sum(axis=1) / math.log(len(label_classes)),
        "is_correct": oof_predicted_labels == y_labels,
    }
)
for class_index, probability_column in enumerate(probability_columns):
    oof[probability_column] = oof_probability[:, class_index]

oof_context_columns = [
    "candidate_id", "asas_sn_id", "lc_path", "event_class", "morphology_primary",
    "morphology_secondary", "human_label_source", "external_hint_labels", "external_catalog_context",
]
oof_context_columns = [column for column in oof_context_columns if column in table.columns]
oof = oof.merge(table[oof_context_columns], on="candidate_id", how="left")
oof.to_parquet(MODEL_DIR / "reviewed_oof_predictions.parquet", index=False)

fold_metrics = pd.DataFrame(fold_metric_rows)
fold_metrics.to_csv(MODEL_DIR / "oof_fold_per_class_metrics.csv", index=False)
fold_summary = fold_metrics.groupby("class").agg(
    support=("support", "sum"),
    recall_mean=("recall", "mean"),
    recall_std=("recall", "std"),
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    roc_auc_mean=("roc_auc_ovr", "mean"),
    roc_auc_std=("roc_auc_ovr", "std"),
    average_precision_mean=("average_precision_ovr", "mean"),
    average_precision_std=("average_precision_ovr", "std"),
).reindex(CLASS_ORDER)
fold_summary.to_csv(MODEL_DIR / "oof_per_class_summary.csv")

oof_summary = pd.DataFrame(
    {
        "metric": ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
        "value": [
            accuracy_score(y_labels, oof_predicted_labels),
            balanced_accuracy_score(y_labels, oof_predicted_labels),
            f1_score(y_labels, oof_predicted_labels, average="macro"),
            f1_score(y_labels, oof_predicted_labels, average="weighted"),
        ],
    }
)
oof_summary.to_csv(MODEL_DIR / "oof_overall_metrics.csv", index=False)

(MODEL_DIR / "oof_original_four_subset_metrics.csv").unlink(missing_ok=True)
display(oof_summary)
display(fold_summary)


# In[10]:


oof_confusion = confusion_matrix(oof["human_label"], oof["oof_predicted_label"], labels=CLASS_ORDER)

confusion_display_order = [CLASS_DISPLAY[label] for label in CLASS_ORDER]

fig, ax = plt.subplots(figsize=(8.2, 6.5))
plot_fraction_count_heatmap(
    pd.DataFrame(
        oof_confusion,
        index=confusion_display_order,
        columns=confusion_display_order,
    ),
    cmap="Blues",
    ax=ax,
)
ax.set_title("OOF confusion")
ax.set(xlabel="Predicted", ylabel="Human label")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "oof_confusion_fraction_count.pdf", dpi=180)
for legacy_name in (
    "oof_confusion_counts.png",
    "oof_confusion_recall_normalized.png",
):
    (FIG_DIR / legacy_name).unlink(missing_ok=True)
plt.show()

pairwise_rows = []
pairwise_auc_matrix = pd.DataFrame(np.eye(len(CLASS_ORDER)), index=CLASS_ORDER, columns=CLASS_ORDER)
pairwise_accuracy_matrix = pd.DataFrame(np.eye(len(CLASS_ORDER)), index=CLASS_ORDER, columns=CLASS_ORDER)
for label_a, label_b in itertools.combinations(CLASS_ORDER, 2):
    pair_mask = oof["human_label"].isin((label_a, label_b))
    column_a = probability_columns[label_classes.index(label_a)]
    column_b = probability_columns[label_classes.index(label_b)]
    denominator = oof.loc[pair_mask, column_a] + oof.loc[pair_mask, column_b]
    score_a = oof.loc[pair_mask, column_a] / np.maximum(denominator, 1e-12)
    true_a = oof.loc[pair_mask, "human_label"].eq(label_a).astype(int)
    pair_prediction = np.where(score_a >= 0.5, label_a, label_b)
    auc_value = roc_auc_score(true_a, score_a)
    balanced_value = balanced_accuracy_score(oof.loc[pair_mask, "human_label"], pair_prediction)
    pairwise_rows.append(
        {
            "class_a": label_a,
            "class_b": label_b,
            "n_a": int(oof.loc[pair_mask, "human_label"].eq(label_a).sum()),
            "n_b": int(oof.loc[pair_mask, "human_label"].eq(label_b).sum()),
            "pairwise_roc_auc": auc_value,
            "pairwise_balanced_accuracy": balanced_value,
        }
    )
    pairwise_auc_matrix.loc[label_a, label_b] = auc_value
    pairwise_auc_matrix.loc[label_b, label_a] = auc_value
    pairwise_accuracy_matrix.loc[label_a, label_b] = balanced_value
    pairwise_accuracy_matrix.loc[label_b, label_a] = balanced_value

expected_pairwise_comparisons = math.comb(len(CLASS_ORDER), 2)
if len(pairwise_rows) != expected_pairwise_comparisons:
    raise RuntimeError(
        "Incomplete all-class performance comparison: "
        f"expected {expected_pairwise_comparisons} class pairs, "
        f"found {len(pairwise_rows)}"
    )
pairwise_metrics = pd.DataFrame(pairwise_rows).sort_values("pairwise_roc_auc")
pairwise_metrics.to_csv(MODEL_DIR / "oof_pairwise_separability.csv", index=False)
pairwise_auc_matrix.to_csv(MODEL_DIR / "oof_pairwise_auc_matrix.csv")
pairwise_accuracy_matrix.to_csv(MODEL_DIR / "oof_pairwise_balanced_accuracy_matrix.csv")

fig, ax = plt.subplots(figsize=(8.5, 7.0))
sns.heatmap(
    pairwise_auc_matrix,
    annot=True,
    fmt=".3f",
    vmin=0.5,
    vmax=1,
    cmap="YlGnBu",
    ax=ax,
)
ax.set_title("OOF pairwise ROC AUC: all classes")
fig.tight_layout()
fig.savefig(FIG_DIR / "oof_all_class_pairwise_roc_auc.pdf", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(8.5, 7.0))
sns.heatmap(
    pairwise_accuracy_matrix,
    annot=True,
    fmt=".3f",
    vmin=0.5,
    vmax=1,
    cmap="YlOrBr",
    ax=ax,
)
ax.set_title("OOF pairwise balanced accuracy: all classes")
fig.tight_layout()
fig.savefig(
    FIG_DIR / "oof_all_class_pairwise_balanced_accuracy.pdf", dpi=180
)
(FIG_DIR / "oof_pairwise_separability.png").unlink(missing_ok=True)
plt.show()

confusion_pairs = (
    oof.loc[~oof["is_correct"]]
    .groupby(["human_label", "oof_predicted_label"])
    .size()
    .rename("n")
    .reset_index()
    .sort_values("n", ascending=False)
)
confusion_pairs.to_csv(MODEL_DIR / "oof_confusion_pairs.csv", index=False)
display(pairwise_metrics)
display(confusion_pairs)


# ## Probability Overlap, Confidence, And Ambiguity

# In[11]:


mean_probability = oof.groupby("human_label")[probability_columns].mean().reindex(CLASS_ORDER)
mean_probability.columns = label_classes

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
sns.heatmap(mean_probability, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis", ax=axes[0])
axes[0].set(title="Mean OOF class score by human label", xlabel="Model score", ylabel="Human label")
sns.boxplot(data=oof, x="human_label", y="oof_confidence", hue="is_correct", order=CLASS_ORDER, ax=axes[1])
axes[1].set(title="OOF confidence for correct and wrong labels", xlabel="Human label", ylabel="Maximum class score")
axes[1].tick_params(axis="x", rotation=25)
axes[1].legend(title="Correct")
fig.tight_layout()
fig.savefig(FIG_DIR / "oof_probability_summary.pdf", dpi=180)
plt.show()

score_plot_columns = 3
score_plot_rows = math.ceil(len(CLASS_ORDER) / score_plot_columns)
fig, axes = plt.subplots(score_plot_rows, score_plot_columns, figsize=(15, 4.5 * score_plot_rows), sharex=True)
bins = np.linspace(0, 1, 31)
for ax, target_label in zip(axes.ravel(), CLASS_ORDER):
    probability_column = probability_columns[label_classes.index(target_label)]
    for human_label in CLASS_ORDER:
        values = oof.loc[oof["human_label"].eq(human_label), probability_column]
        ax.hist(values, bins=bins, histtype="step", lw=2, color=CLASS_COLORS[human_label], label=f"{CLASS_DISPLAY[human_label]} (n={len(values)})")
    ax.set_title(f"OOF score for {CLASS_DISPLAY[target_label]}")
    ax.set_yscale("log")
    ax.set_ylabel("Candidates")
for ax in axes.ravel()[len(CLASS_ORDER):]:
    ax.set_axis_off()
for ax in axes[-1, :]:
    ax.set_xlabel("Class score")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR / "oof_class_score_overlap.pdf", dpi=180)
plt.show()


# ## Unsupervised PCA And Descriptive Supervised LDA

# In[12]:


imputer = SimpleImputer(strategy="median")
imputed = imputer.fit_transform(X_labeled)
lower = np.nanpercentile(imputed, 1, axis=0)
upper = np.nanpercentile(imputed, 99, axis=0)
winsorized = np.clip(imputed, lower, upper)
scaled = StandardScaler().fit_transform(winsorized)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coordinates = pca.fit_transform(scaled)
lda = LinearDiscriminantAnalysis(n_components=2)
lda_coordinates = lda.fit_transform(scaled, y_encoded)

projection = pd.DataFrame(
    {
        "candidate_id": labeled_model_input["candidate_id"].astype(str),
        "human_label": y_labels,
        "pca_1": pca_coordinates[:, 0],
        "pca_2": pca_coordinates[:, 1],
        "lda_1": lda_coordinates[:, 0],
        "lda_2": lda_coordinates[:, 1],
    }
)
projection.to_csv(MODEL_DIR / "feature_space_pca_lda.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for label in CLASS_ORDER:
    mask = projection["human_label"].eq(label)
    axes[0].scatter(projection.loc[mask, "pca_1"], projection.loc[mask, "pca_2"], s=18, alpha=0.55, color=CLASS_COLORS[label], label=f"{CLASS_DISPLAY[label]} (n={int(mask.sum())})")
    axes[1].scatter(projection.loc[mask, "lda_1"], projection.loc[mask, "lda_2"], s=18, alpha=0.55, color=CLASS_COLORS[label], label=CLASS_DISPLAY[label])
axes[0].set(
    xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})".replace("%", r"\%"),
    ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})".replace("%", r"\%"),
    title="Unsupervised PCA of stats features",
)
axes[1].set(xlabel="LD1", ylabel="LD2", title="Descriptive LDA fit on all human labels")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_space_pca_lda.pdf", dpi=180)
plt.show()
display(Markdown("PCA is label-blind. LDA uses all labels and is only a descriptive projection; use OOF metrics for honest performance."))


# ## Global Feature Importance: Gain Versus Held-Out Permutation

# In[13]:


gain_importance = pd.read_csv(MODEL_DIR / "feature_importance_gain.csv")
labeled_by_id = labeled_model_input.set_index("candidate_id", drop=False)
heldout_input = labeled_by_id.loc[test_predictions["candidate_id"].astype(str)].reset_index(drop=True)
X_heldout = transform_features(
    heldout_input,
    feature_columns=bundle["feature_columns"],
    categorical_maps=bundle["categorical_maps"],
)
y_heldout_encoded = np.array([class_to_index[label] for label in test_predictions["y_true"].astype(str)], dtype=int)
permutation = permutation_importance(
    bundle["model"],
    X_heldout,
    y_heldout_encoded,
    scoring="f1_macro",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
permutation_frame = pd.DataFrame(
    {
        "feature": bundle["feature_columns"],
        "permutation_macro_f1_drop_mean": permutation.importances_mean,
        "permutation_macro_f1_drop_std": permutation.importances_std,
    }
).sort_values("permutation_macro_f1_drop_mean", ascending=False, ignore_index=True)
permutation_frame.to_csv(MODEL_DIR / "feature_importance_permutation.csv", index=False)

total_gain = float(gain_importance["gain"].sum())
if not np.isfinite(total_gain) or total_gain <= 0:
    raise ValueError(f"Cannot normalize non-positive total LightGBM gain: {total_gain}")
gain_importance["gain_percent"] = 100.0 * gain_importance["gain"] / total_gain
gain_importance.to_csv(MODEL_DIR / "feature_importance_gain_percent.csv", index=False)
gain_plot = (
    gain_importance.nlargest(FEATURE_IMPORTANCE_TOP_N, "gain_percent")
    .sort_values("gain_percent")
    .reset_index(drop=True)
)
top_permutation = permutation_frame.head(FEATURE_IMPORTANCE_TOP_N).sort_values(
    "permutation_macro_f1_drop_mean"
)
gain_y = np.arange(len(gain_plot))
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(
    gain_plot["gain_percent"],
    gain_y,
    color="#326273",
    s=24,
    zorder=3,
)
ax.set_yticks(gain_y, gain_plot["feature"])
ax.set_xlim(0.0, float(gain_plot["gain_percent"].max()) * 1.08)
ax.set_title(f"Top {FEATURE_IMPORTANCE_TOP_N} LightGBM gain features")
ax.set_xlabel(r"Share of total LightGBM gain (\%)", fontsize=14)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="x", which="both", alpha=0.35)
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_gain.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top_permutation["feature"], top_permutation["permutation_macro_f1_drop_mean"], xerr=top_permutation["permutation_macro_f1_drop_std"], color="#a44200", alpha=0.9)
ax.axvline(0, color="black", lw=1)
ax.set_title(f"Top {FEATURE_IMPORTANCE_TOP_N} held-out permutation features")
ax.set_xlabel("Drop in macro F1 when shuffled")
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_permutation.pdf", dpi=180)
plt.show()
display(permutation_frame.head(30))


# ## All-Class Contrast-Specific Feature Importance
# 
# These are separate binary models using the current curated feature set. One-vs-rest compares every class with all other classes combined; pairwise fits every individual class pair. Importance is the decrease in out-of-fold ROC AUC when a feature is shuffled, aggregated over five validation folds and repeated permutations.

# In[ ]:


CONTRAST_PERMUTATION_REPEATS = 4
CONTRAST_TOP_FEATURES = FEATURE_IMPORTANCE_TOP_N
contrast_bundle = load_target_model(MODEL_DIR)
contrast_metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
contrast_feature_columns = list(feature_columns)
contrast_config = TrainingConfig(**contrast_metadata["config"])
contrast_model_input = table[["candidate_id", TARGET_COLUMN, *contrast_feature_columns]].copy()
contrast_labeled_input = contrast_model_input.loc[trainable].reset_index(drop=True)
X_contrast_labeled = transform_features(
    contrast_labeled_input,
    feature_columns=contrast_feature_columns,
    categorical_maps=contrast_bundle["categorical_maps"],
)
contrast_y_labels = contrast_labeled_input[TARGET_COLUMN].astype(str).to_numpy()


def binary_contrast_importance(
    X_contrast: pd.DataFrame,
    y_binary: np.ndarray,
    *,
    kind: str,
    comparison: str,
    positive_class: str,
    negative_class: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_binary = np.asarray(y_binary, dtype=int)
    if set(np.unique(y_binary)) != {0, 1}:
        raise ValueError(f"{comparison} does not contain both binary classes")

    raw_frames = []
    fold_rows = []
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + seed_offset
    )
    for fold, (fit_index, validation_index) in enumerate(splitter.split(X_contrast, y_binary), start=1):
        fold_model, inner_iterations = fit_classifier_with_inner_early_stopping(
            X_contrast.iloc[fit_index],
            y_binary[fit_index],
            config=contrast_config,
            n_classes=2,
            random_state=RANDOM_STATE + seed_offset * 100 + fold,
        )
        validation_probability = np.asarray(
            fold_model.predict_proba(X_contrast.iloc[validation_index]), dtype=float
        )[:, 1]
        fold_rows.append(
            {
                "kind": kind,
                "comparison": comparison,
                "positive_class": positive_class,
                "negative_class": negative_class,
                "fold": fold,
                "n_validation": int(len(validation_index)),
                "n_validation_positive": int(y_binary[validation_index].sum()),
                "baseline_roc_auc": float(
                    roc_auc_score(y_binary[validation_index], validation_probability)
                ),
                "selected_iteration": int(fold_model.n_estimators_),
                "inner_best_iterations": "|".join(map(str, inner_iterations)),
            }
        )
        result = permutation_importance(
            fold_model,
            X_contrast.iloc[validation_index],
            y_binary[validation_index],
            scoring="roc_auc",
            n_repeats=CONTRAST_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE + seed_offset * 1000 + fold,
            n_jobs=1,
        )
        repeat_frame = pd.DataFrame(
            result.importances.T, columns=contrast_feature_columns
        ).melt(var_name="feature", value_name="roc_auc_drop")
        repeat_frame["fold"] = fold
        raw_frames.append(repeat_frame)

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(
            roc_auc_drop_mean=("roc_auc_drop", "mean"),
            roc_auc_drop_std=("roc_auc_drop", "std"),
            n_evaluations=("roc_auc_drop", "size"),
        )
        .sort_values("roc_auc_drop_mean", ascending=False)
    )
    summary.insert(0, "negative_class", negative_class)
    summary.insert(0, "positive_class", positive_class)
    summary.insert(0, "comparison", comparison)
    summary.insert(0, "kind", kind)
    return summary, pd.DataFrame(fold_rows)


one_vs_rest_frames = []
pairwise_frames = []
fold_frames = []
comparison_display = {}
for contrast_index, label in enumerate(CLASS_ORDER, start=1):
    comparison = f"{label}_vs_rest"
    comparison_display[comparison] = f"{CLASS_DISPLAY[label]} vs rest"
    summary, folds = binary_contrast_importance(
        X_contrast_labeled,
        (contrast_y_labels == label).astype(int),
        kind="one_vs_rest",
        comparison=comparison,
        positive_class=label,
        negative_class="all_other_classes",
        seed_offset=contrast_index,
    )
    one_vs_rest_frames.append(summary)
    fold_frames.append(folds)

for contrast_index, (label_a, label_b) in enumerate(
    itertools.combinations(CLASS_ORDER, 2), start=1 + len(CLASS_ORDER)
):
    pair_mask = np.isin(contrast_y_labels, (label_a, label_b))
    comparison = f"{label_a}_vs_{label_b}"
    comparison_display[comparison] = f"{CLASS_DISPLAY[label_a]} vs {CLASS_DISPLAY[label_b]}"
    summary, folds = binary_contrast_importance(
        X_contrast_labeled.loc[pair_mask].reset_index(drop=True),
        (contrast_y_labels[pair_mask] == label_a).astype(int),
        kind="pairwise",
        comparison=comparison,
        positive_class=label_a,
        negative_class=label_b,
        seed_offset=contrast_index,
    )
    pairwise_frames.append(summary)
    fold_frames.append(folds)

one_vs_rest_importance = pd.concat(one_vs_rest_frames, ignore_index=True)
pairwise_importance = pd.concat(pairwise_frames, ignore_index=True)
contrast_fold_metrics = pd.concat(fold_frames, ignore_index=True)
one_vs_rest_importance.to_csv(
    MODEL_DIR / "feature_importance_all_classes_one_vs_rest.csv", index=False
)
pairwise_importance.to_csv(
    MODEL_DIR / "feature_importance_all_classes_pairwise.csv", index=False
)
contrast_fold_metrics.to_csv(
    MODEL_DIR / "feature_importance_all_classes_contrast_fold_metrics.csv",
    index=False,
)
for legacy_name in (
    "feature_importance_original_four_one_vs_rest.csv",
    "feature_importance_original_four_pairwise.csv",
    "feature_importance_original_four_contrast_fold_metrics.csv",
):
    (MODEL_DIR / legacy_name).unlink(missing_ok=True)


def plot_contrast_importance(
    frame: pd.DataFrame,
    comparisons: list[str],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (*output_dir.glob("*.png"), *output_dir.glob("*.pdf")):
        stale.unlink()
    for comparison in comparisons:
        top = (
            frame.loc[frame["comparison"].eq(comparison)]
            .nlargest(CONTRAST_TOP_FEATURES, "roc_auc_drop_mean")
            .sort_values("roc_auc_drop_mean")
        )
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.barh(
            top["feature"],
            top["roc_auc_drop_mean"],
            xerr=top["roc_auc_drop_std"],
            color="#326273",
            alpha=0.9,
        )
        ax.axvline(0, color="black", lw=1)
        ax.set(title=comparison_display[comparison], xlabel="OOF ROC AUC drop when shuffled")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{comparison}.pdf", dpi=180)
        plt.show()


one_vs_rest_comparisons = [f"{label}_vs_rest" for label in CLASS_ORDER]
plot_contrast_importance(
    one_vs_rest_importance,
    one_vs_rest_comparisons,
    output_dir=FIG_DIR / "feature_importance_all_classes_one_vs_rest",
)
pairwise_comparisons = [
    f"{a}_vs_{b}" for a, b in itertools.combinations(CLASS_ORDER, 2)
]
if len(pairwise_comparisons) != expected_pairwise_comparisons:
    raise RuntimeError(
        "Incomplete all-class feature-importance comparison: "
        f"expected {expected_pairwise_comparisons} class pairs, "
        f"found {len(pairwise_comparisons)}"
    )
plot_contrast_importance(
    pairwise_importance,
    pairwise_comparisons,
    output_dir=FIG_DIR / "feature_importance_all_classes_pairwise",
)
for legacy_name in (
    "feature_importance_original_four_one_vs_rest.png",
    "feature_importance_original_four_pairwise.png",
):
    (FIG_DIR / legacy_name).unlink(missing_ok=True)

display(
    contrast_fold_metrics.groupby(["kind", "comparison"])["baseline_roc_auc"]
    .agg(["mean", "std"])
    .round(3)
)
display(one_vs_rest_importance.groupby("comparison", sort=False).head(10))
display(pairwise_importance.groupby("comparison", sort=False).head(10))


# ## Distributions Of The Most Reproducible Features

# In[14]:


distribution_features = permutation_frame.head(6)["feature"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feature in zip(axes.ravel(), distribution_features):
    frame = table.loc[trainable, [TARGET_COLUMN, feature]].copy()
    frame[feature] = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_values = frame[feature].dropna()
    if finite_values.empty:
        ax.set_axis_off()
        continue
    lower_value, upper_value = finite_values.quantile([0.01, 0.99])
    frame["clipped_value"] = frame[feature].clip(lower_value, upper_value)
    sns.boxplot(
        data=frame,
        x=TARGET_COLUMN,
        y="clipped_value",
        order=CLASS_ORDER,
        palette=[CLASS_COLORS[label] for label in CLASS_ORDER],
        showfliers=False,
        ax=ax,
    )
    ax.set(title=feature, xlabel="", ylabel=r"1\%-99\% clipped value")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "top_feature_distributions_by_class.pdf", dpi=180)
plt.show()


# ## Compare Human And ML Labels With External Catalog Hints

# In[15]:


labeled_scores = scores.loc[scores[TARGET_COLUMN].notna()].copy()
human_catalog_crosstab = pd.crosstab(
    labeled_scores[TARGET_COLUMN],
    labeled_scores["external_catalog_context"],
).reindex(CLASS_ORDER, fill_value=0)
unreviewed_catalog_crosstab = pd.crosstab(
    unreviewed_scores["y_pred"],
    unreviewed_scores["external_catalog_context"],
).reindex(CLASS_ORDER, fill_value=0)
human_catalog_crosstab.to_csv(MODEL_DIR / "human_labels_vs_external_catalog_context.csv")
unreviewed_catalog_crosstab.to_csv(MODEL_DIR / "unreviewed_ml_predictions_vs_external_catalog_context.csv")

external_metric_rows = []
for label in EXTERNAL_HINT_CLASSES:
    human_positive = labeled_scores[TARGET_COLUMN].eq(label)
    external_positive = labeled_scores[f"external_hint_{label}"].fillna(False).astype(bool)
    true_positive = int((human_positive & external_positive).sum())
    false_positive = int((~human_positive & external_positive).sum())
    false_negative = int((human_positive & ~external_positive).sum())
    external_metric_rows.append(
        {
            "class": label,
            "external_positive_human_rows": int(external_positive.sum()),
            "human_positive_rows": int(human_positive.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "external_precision_vs_human": true_positive / max(true_positive + false_positive, 1),
            "external_recall_vs_human": true_positive / max(true_positive + false_negative, 1),
        }
    )
external_metrics = pd.DataFrame(external_metric_rows)
external_metrics.to_csv(MODEL_DIR / "external_catalog_metrics_vs_human_labels.csv", index=False)

fig, ax = plt.subplots(figsize=(8.4, 6.2))
plot_fraction_count_heatmap(
    human_catalog_crosstab,
    cmap="Blues",
    ax=ax,
)
ax.set(
    title="Human labels versus external catalog context",
    xlabel="External context",
    ylabel="Human label",
)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "human_labels_vs_external_catalog_context.pdf", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(8.4, 6.2))
plot_fraction_count_heatmap(
    unreviewed_catalog_crosstab,
    cmap="YlOrBr",
    ax=ax,
)
ax.set(
    title="Unreviewed ML prediction versus external context",
    xlabel="External context",
    ylabel="ML prediction",
)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "unreviewed_ml_prediction_vs_external_context.pdf", dpi=180)
(FIG_DIR / "external_catalog_comparison.png").unlink(missing_ok=True)
plt.show()

display(external_metrics)
display(human_catalog_crosstab)
display(Markdown("External hints are comparison-only. A zero-row hint class means this July 1 DB contains no usable external label for that phenomenon; it does not mean the human review label is wrong."))


# ## Actual Light Curves For The Most Confident OOF Mistakes

# In[16]:


hard_examples = (
    oof.loc[~oof["is_correct"]]
    .sort_values(["oof_confidence", "oof_margin"], ascending=False)
    .head(HARD_EXAMPLE_N)
    .copy()
)
hard_examples["mistake_rank"] = np.arange(1, len(hard_examples) + 1)


def resolve_bundle_lightcurve(row: pd.Series) -> Path | None:
    stored = row.get("lc_path")
    if stored is not None and not pd.isna(stored):
        path = Path(str(stored)).expanduser()
        if path.exists():
            return path
        bundled = LIGHTCURVE_DIR / path.name
        if bundled.exists():
            return bundled
    stems = []
    for column in ("asas_sn_id", "candidate_id"):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            stems.extend([text.removeprefix("stv_"), text])
    for stem in dict.fromkeys(stems):
        for suffix in (".dat3", ".dat2", ".dat"):
            path = LIGHTCURVE_DIR / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


@lru_cache(maxsize=64)
def load_cached_lightcurve(path_text: str) -> pd.DataFrame:
    return load_lightcurve_df(Path(path_text), apply_quality=True)


hard_examples["resolved_lightcurve_path"] = hard_examples.apply(resolve_bundle_lightcurve, axis=1).map(lambda value: str(value) if value else "")
hard_examples.to_csv(MODEL_DIR / "hard_oof_misclassifications_with_lightcurve_paths.csv", index=False)


def plot_mistake_panel(ax: plt.Axes, row: pd.Series) -> None:
    path_text = str(row.get("resolved_lightcurve_path", ""))
    title = f"{int(row['mistake_rank']):02d} {row['candidate_id']}"
    subtitle = (
        f"human={CLASS_DISPLAY[row['human_label']]}, pred={CLASS_DISPLAY[row['oof_predicted_label']]}"
        f", conf={float(row['oof_confidence']):.2f}"
    )
    if not path_text:
        ax.text(0.5, 0.5, "missing light curve", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=7.5)
        ax.set_axis_off()
        return
    try:
        lightcurve = load_cached_lightcurve(path_text)
    except Exception as exc:
        ax.text(0.5, 0.5, f"load failed: {exc}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=7.5)
        ax.set_axis_off()
        return
    jd = pd.to_numeric(lightcurve["jd"], errors="coerce")
    mag = pd.to_numeric(lightcurve["mag"], errors="coerce")
    camera = lightcurve.get("camera_name", lightcurve.get("camera", pd.Series("all", index=lightcurve.index))).fillna("all").astype(str)
    finite = jd.notna() & mag.notna()
    for camera_name, indices in camera.loc[finite].groupby(camera.loc[finite]).groups.items():
        indices = pd.Index(indices)
        ax.scatter(jd.loc[indices] - 2450000.0, mag.loc[indices], s=4.5, alpha=0.67, color=stable_camera_color(str(camera_name)), linewidths=0)
    ax.invert_yaxis()
    ax.set_title(f"{title}\n{subtitle}", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


gallery_dir = FIG_DIR / "hard_oof_misclassifications"
gallery_dir.mkdir(parents=True, exist_ok=True)
expected_pages = math.ceil(len(hard_examples) / PER_PAGE) if len(hard_examples) else 0
gallery_paths = [gallery_dir / f"hard_oof_mistakes_page{page:02d}.pdf" for page in range(1, expected_pages + 1)]
if REBUILD_GALLERIES or needs_training or not all(path.exists() for path in gallery_paths):
    gallery_paths = []
    for page_index, start in enumerate(range(0, len(hard_examples), PER_PAGE), start=1):
        page = hard_examples.iloc[start:start + PER_PAGE]
        fig, axes = plt.subplots(5, 4, figsize=(16, 13))
        axes = axes.ravel()
        for ax, (_, row) in zip(axes, page.iterrows()):
            plot_mistake_panel(ax, row)
        for ax in axes[len(page):]:
            ax.set_axis_off()
        fig.supxlabel("JD - 2450000", fontsize=11)
        fig.supylabel("ASAS-SN magnitude", fontsize=11)
        fig.suptitle(f"Most confident out-of-fold human/model disagreements: page {page_index} of {expected_pages}", y=0.995, fontsize=14)
        fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])
        path = gallery_dir / f"hard_oof_mistakes_page{page_index:02d}.pdf"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        gallery_paths.append(path)
else:
    print(f"Reusing {len(gallery_paths)} hard-example gallery pages")

print(f"OOF mistakes: {int((~oof['is_correct']).sum()):,}")
print(f"Hard examples exported: {len(hard_examples):,}; missing light curves: {int(hard_examples['resolved_lightcurve_path'].eq('').sum()):,}")
if gallery_paths:
    display(Image(filename=str(gallery_paths[0]), width=1100))
display(hard_examples[["mistake_rank", "candidate_id", "human_label", "oof_predicted_label", "oof_confidence", "oof_margin", "external_catalog_context", "resolved_lightcurve_path"]])


# ## Per-Class Review Queues And Interpretation Boundary

# In[17]:


predicted_counts = unreviewed_scores["y_pred"].value_counts().reindex(CLASS_ORDER, fill_value=0)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].bar([CLASS_DISPLAY[label] for label in CLASS_ORDER], predicted_counts.values, color=[CLASS_COLORS[label] for label in CLASS_ORDER])
axes[0].set(title="Winning class among human-unreviewed candidates", ylabel="Candidates")
axes[0].tick_params(axis="x", rotation=25)
for label, probability_column in zip(label_classes, probability_columns):
    axes[1].hist(unreviewed_scores[probability_column], bins=np.linspace(0, 1, 41), histtype="step", lw=2, color=CLASS_COLORS[label], label=CLASS_DISPLAY[label])
axes[1].set(title="Unreviewed class-score distributions", xlabel="Class score", ylabel="Candidates", yscale="log")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "unreviewed_prediction_summary.pdf", dpi=180)
plt.show()

for label in CLASS_ORDER:
    display(Markdown(f"### Top unreviewed {CLASS_DISPLAY[label]} scores"))
    display(
        top_queues.loc[top_queues["queue_class"].eq(label), [
            "rank_within_class", "candidate_id", "y_pred", "prediction_confidence", "score_margin",
            *probability_columns, "external_catalog_context",
        ]].head(20)
    )


# ## How To Read This Experiment
# 
# - Prefer the five-fold out-of-fold confusion matrix, per-class fold spread, and pairwise AUC table when judging separability. Each outer fold chooses its boosting iteration through three inner early-stopping folds, then refits on the full outer-training sample.
# - The helper's held-out test set is useful but contains only a handful of microlensing and LTV examples, so its rare-class metrics will move substantially when labels change.
# - Gain importance is an aggregate tree-usage count across the eight classes. Held-out permutation importance is the stronger check that a feature contributes to macro-F1 on unseen rows.
# - PCA is label-blind. LDA is trained on all labels and is only a descriptive view, not an honest performance estimate.
# - Catalog hints are never model inputs or target labels. Their comparison can reveal agreement, catalog incompleteness, or suspicious disagreements, but it cannot replace review truth.
# - Because training uses balanced class weights, the class scores and predicted population counts are not calibrated prevalence estimates.
# - Before automating review decisions, inspect the hard-example light curves, expand microlensing/LTV labels, and repeat this notebook after adjudicating the largest confusion pairs.
