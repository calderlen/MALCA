#!/usr/bin/env python
# coding: utf-8

"""Write July 1 EB-like ML diagnostics and light-curve figures to results."""

# # July 1 Eclipsing-Binary-Like ML Visualizations
# 
# This standalone script loads the saved stats-plus-context LightGBM ranker, scores every July 1 candidate, exports the highest-scored unreviewed queues, and plots the **actual ASAS-SN measurements** in both raw-time and phase-folded views. Retrain with `python scripts/train_july1_eb_ml.py` from the repository root.
# 
# The training truth comes only from human review: `morphology_secondary` / `morphology_secondary_json` tags such as `eclipsing_like`, `detached_binary_like`, and `contact_binary_like`. Gaia and VSX EB classes are never training labels or model features. They are retained strictly for post-model agreement, disagreement, and missed-candidate comparisons. `prob_eb_like` is a ranking score, not a calibrated population probability.
# 

# In[1]:


from __future__ import annotations

import json
import math
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from malca.core.phase import phase_template, resolve_phase_period
from malca.io.lightcurve_io import load_lightcurve_df, stable_camera_color
from malca.meta_analysis.ml.feature_policy import (
    STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
    restore_legacy_excluded_model_features,
)
from malca.meta_analysis.ml.plotting import (
    FEATURE_IMPORTANCE_TOP_N,
    apply_ml_plot_style,
    plot_fraction_count_heatmap,
)
from malca.meta_analysis.ml.review_lightgbm import (
    ASTROPHYSICAL_CONTEXT_FEATURES,
    add_astrophysical_context_features,
    load_target_model,
    score_target_model,
    transform_features,
)
from malca.review.other_eb_triage import EB_REGEX, PERIODIC_REGEX, compute_eb_triage

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning)

apply_ml_plot_style()
pd.set_option("display.max_columns", 110)
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
DB_PATH = RUN_DIR / "review" / "review.db"
BASE_DIR = RUN_DIR / "results" / "eb_feature_selection"
MODEL_NAME = "stats_plus_astrophysical_context"


def resolve_model_dir(base_dir: Path, preferred_name: str) -> Path:
    """Prefer the current retraining target, then the persisted legacy artifact."""

    for name in (preferred_name, "stats_only"):
        candidate = base_dir / name
        if (candidate / "model.joblib").is_file():
            return candidate
    return base_dir / preferred_name


MODEL_DIR = resolve_model_dir(BASE_DIR, MODEL_NAME)
MODEL_NAME = MODEL_DIR.name
FIG_DIR = MODEL_DIR / "figures"
LIGHTCURVE_DIR = RUN_DIR / "bundle_assets" / "lightcurves"

TARGET_COLUMN = "eb_like_label"
POSITIVE_LABEL = "eb_like"
NEGATIVE_LABEL = "not_eb"
PROB_COL = "prob_eb_like"

TOP_UNREVIEWED_N = 500
PER_PAGE = 20
EXCLUDE_KNOWN_EB_FROM_PLOTTED_QUEUE = False
REBUILD_LIGHTCURVE_PAGES = False
PERMUTATION_REPEATS = 8
RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
for stale_png in FIG_DIR.rglob("*.png"):
    stale_png.unlink()

print(f"Review DB: {DB_PATH}")
print(f"Model directory: {MODEL_DIR}")


# ## Load The July 1 Population
# 
# The review database supplies the flattened `stats_*` compute-stats block, catalog classifications, preferred period solutions, local light-curve paths, and human review taxonomy in one candidate-indexed surface.
# 

# In[2]:


if not DB_PATH.exists():
    raise FileNotFoundError(DB_PATH)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def read_table_without_json(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
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
    candidates = read_table_without_json(conn, "candidates")
    reviews = read_table_without_json(conn, "reviews")

candidates["candidate_id"] = candidates["candidate_id"].astype(str)
reviews["candidate_id"] = reviews["candidate_id"].astype(str)
table = candidates.merge(reviews, on="candidate_id", how="left", suffixes=("", "_review"))
table = table.loc[:, ~table.columns.duplicated()].copy()
table = add_astrophysical_context_features(table)
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


# ## Human Review EB Labels And External-Catalog Context
# 
# The positive class is defined entirely by the review taxonomy's EB-like secondary morphology tags. Clear reviewed event classes without an EB-like tag form the negative class. Ambiguous review classes remain unlabeled. Gaia/VSX EB hints and the existing deterministic `eb_score` are computed only for later comparison.
# 

# In[3]:


def clean_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


EB_REVIEW_TAGS = {
    "eclipsing_like",
    "detached_binary_like",
    "semi_detached_binary_like",
    "contact_binary_like",
    "ellipsoidal_like",
    "heartbeat_like",
}


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


secondary_scalar = clean_text(table, "morphology_secondary")
secondary_tag_sets = table.get(
    "morphology_secondary_json",
    pd.Series("", index=table.index),
).map(parse_secondary_tags)
secondary_tag_sets = pd.Series(
    [tags | ({scalar} if scalar else set()) for tags, scalar in zip(secondary_tag_sets, secondary_scalar)],
    index=table.index,
)
table["human_eb_tags"] = secondary_tag_sets.map(lambda tags: "|".join(sorted(tags & EB_REVIEW_TAGS)))

event_class = clean_text(table, "event_class")
physical_primary = clean_text(table, "physical_primary")
status = clean_text(table, "status")
workflow = clean_text(table, "workflow_status")
reviewed = (status.ne("") & status.ne("unreviewed")) | (workflow.ne("") & workflow.ne("unreviewed"))

human_eb_like = reviewed & (
    table["human_eb_tags"].ne("")
    | physical_primary.eq("eclipsing_or_geometric_binary")
)
CLEAR_TRAINING_EVENT_CLASSES = {
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
human_non_eb = reviewed & event_class.isin(CLEAR_TRAINING_EVENT_CLASSES) & ~human_eb_like

table["human_eb_like"] = human_eb_like.astype(bool)
table[TARGET_COLUMN] = pd.NA
table.loc[human_eb_like, TARGET_COLUMN] = POSITIVE_LABEL
table.loc[human_non_eb, TARGET_COLUMN] = NEGATIVE_LABEL
table["eb_label_source"] = np.select(
    [human_eb_like, human_non_eb],
    ["human_review_eb_like", "human_review_non_eb"],
    default="unlabeled",
)

# External catalog context. These flags never enter model_input.
gaia_class = clean_text(table, "gaia_var_class")
vsx_class = clean_text(table, "vsx_class")
gaia_eb_period = pd.to_numeric(table.get("gaia_eb_period", pd.Series(np.nan, index=table.index)), errors="coerce")
gaia_eb_morph = clean_text(table, "gaia_eb_morph")
table["known_eb_hint"] = (
    gaia_class.str.contains(EB_REGEX, na=False)
    | vsx_class.str.contains(EB_REGEX, na=False)
    | gaia_eb_period.notna()
    | gaia_eb_morph.ne("")
)
table["known_non_eb_periodic_hint"] = (
    gaia_class.str.contains(PERIODIC_REGEX, na=False)
    | vsx_class.str.contains(PERIODIC_REGEX, na=False)
) & ~table["known_eb_hint"]

triage_input_columns = [
    "candidate_id", "gaia_var_class", "gaia_eb_period", "gaia_eb_morph", "vsx_class",
    "catalog_match", "periodic_flag", "stats_variability_lomb_scargle_best_period_days",
    "stats_variability_lomb_scargle_peak_power", "stats_variability_lomb_scargle_fap",
    "dip_run_count", "dip_inter_event_spacing_median", "dip_inter_event_spacing_std",
    "dip_amplitude_consistency", "dip_duration_consistency", "dip_symmetry_score",
    "stats_variability_von_neumann_ratio", "stats_variability_stetson_J", "dipper_score",
]
triage_input_columns = [column for column in triage_input_columns if column in table.columns]
deterministic_triage = compute_eb_triage(table[triage_input_columns].copy())
triage_columns = [
    "candidate_id", "eb_score", "eb_bin", "eb_likely_flag", "eb_score_notes",
    "known_periodic_hint", "possible_missed_eb",
]
triage_columns = [column for column in triage_columns if column in deterministic_triage.columns]
table = table.merge(deterministic_triage[triage_columns], on="candidate_id", how="left")

label_source_audit = table.groupby([TARGET_COLUMN, "eb_label_source"], dropna=False).size().rename("n").reset_index()
display(label_source_audit)
display(
    table.loc[human_eb_like, "human_eb_tags"]
    .str.split("|")
    .explode()
    .value_counts()
    .rename_axis("human_review_eb_tag")
    .reset_index(name="n")
)
print(f"Human-reviewed EB-like positives: {int(human_eb_like.sum()):,}")
print(f"Human-reviewed clear non-EB negatives: {int(human_non_eb.sum()):,}")
print(f"Reviewed ambiguous rows excluded from training: {int((reviewed & ~(human_eb_like | human_non_eb)).sum()):,}")
print(f"External Gaia/VSX EB hints retained for comparison: {int(table['known_eb_hint'].sum()):,}")


# ## Select Compute-Stats Features
# 

# In[4]:


MODEL_EXCLUDED_STATS = {
    *STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
    "stats_variability_lomb_scargle_best_period_days",
}


def is_usable_stats_feature(series: pd.Series, *, min_non_null: int = 30, max_cardinality: int = 50) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return int(numeric.notna().sum()) >= min_non_null and numeric.nunique(dropna=True) > 1
    values = series.dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    return int(values.size) >= min_non_null and 1 < int(values.nunique()) <= max_cardinality


trainable = table[TARGET_COLUMN].notna()
feature_columns = [
    column
    for column in stats_columns
    if column not in MODEL_EXCLUDED_STATS
    and is_usable_stats_feature(table.loc[trainable, column])
]
feature_columns = sorted(
    feature_columns,
    key=lambda column: (-int(table.loc[trainable, column].notna().sum()), column),
)
unusable_context_features = [
    column
    for column in ASTROPHYSICAL_CONTEXT_FEATURES
    if not is_usable_stats_feature(table.loc[trainable, column])
]
if unusable_context_features:
    raise ValueError(f"Requested astrophysical-context features are unusable: {unusable_context_features}")
feature_columns = [*feature_columns, *ASTROPHYSICAL_CONTEXT_FEATURES]
model_input = table[["candidate_id", TARGET_COLUMN, *feature_columns]].copy()

label_counts = table.loc[trainable, TARGET_COLUMN].value_counts()
display(label_counts.rename_axis("target").reset_index(name="n"))
print(f"Trainable rows: {int(trainable.sum()):,}")
print(f"Selected total model features: {len(feature_columns):,}")
print(f"Astrophysical-context features: {list(ASTROPHYSICAL_CONTEXT_FEATURES)}")
print("Excluded potential confounders:", sorted(MODEL_EXCLUDED_STATS))


# ## Load The Saved EB-Like Ranker
# 
# Training is owned by `scripts/train_july1_eb_ml.py`; rerun this notebook afterward to regenerate its visualizations.
# 

# In[5]:


model_path = MODEL_DIR / "model.joblib"
if not model_path.exists():
    raise FileNotFoundError(
        f"No saved model at {model_path}. Run scripts/train_july1_eb_ml.py from the repository root."
    )
needs_training = False
bundle = load_target_model(MODEL_DIR)
model_input, restored_legacy_features = restore_legacy_excluded_model_features(
    model_input,
    table,
    bundle["feature_columns"],
)
missing = sorted(set(bundle["feature_columns"]) - set(model_input.columns))
if missing:
    raise ValueError(f"Saved model expects missing features: {missing[:10]}")
if restored_legacy_features:
    print(f"Restored legacy excluded features for saved-model scoring: {list(restored_legacy_features)}")
print("Using the saved EB-like model. Retrain with scripts/train_july1_eb_ml.py.")

metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
display(pd.DataFrame({"value": [metadata["n_rows"], metadata["n_features"], metadata["class_counts"]]}, index=["training rows", "features", "class counts"]))


# ## Score Every Candidate And Build Review Queues
# 

# In[6]:


prediction_frame = score_target_model(MODEL_DIR, model_input)
prediction_columns = ["candidate_id", "y_pred", "prediction_confidence", PROB_COL]

score_context_columns = [
    "candidate_id", "asas_sn_id", "lc_path", "status", "workflow_status", "event_class",
    "morphology_primary", "morphology_secondary", "physical_primary", "interest_score",
    TARGET_COLUMN, "human_eb_like", "human_eb_tags", "eb_label_source",
    "known_eb_hint", "known_non_eb_periodic_hint", "eb_score", "eb_bin", "eb_likely_flag",
    "eb_score_notes", "gaia_var_class", "gaia_eb_period", "gaia_eb_morph", "vsx_class",
    "asassn_var_type", "ztf_var_type", "catalog_match", "catalog_source", "simbad_otype",
    "periodicity_period", "period_consensus_days", "phase_period_days", "pdm_corrected_period",
    "ce_corrected_period", "lsp_period", "periodicity_score", "periodic_flag",
    "stats_variability_periodic_feature_period_days", "stats_variability_lomb_scargle_peak_power",
    "stats_variability_lomb_scargle_fap", "stats_harmonics_model_amplitude",
    "stats_variability_quasi_periodicity_q", "stats_variability_von_neumann_ratio",
    "stats_variability_stetson_J", "stats_amplitude", "dipper_score",
]
score_context_columns = [column for column in score_context_columns if column in table.columns]
scores = table[score_context_columns].merge(prediction_frame[prediction_columns], on="candidate_id", how="left")
scores.to_parquet(MODEL_DIR / "all_candidates_scores.parquet", index=False)

score_status = clean_text(scores, "status")
score_workflow = clean_text(scores, "workflow_status")
score_event = clean_text(scores, "event_class")
unreviewed_mask = (
    score_status.isin(("", "unreviewed"))
    & score_workflow.isin(("", "unreviewed"))
    & score_event.isin(("", "unclassified"))
)
unreviewed_scores = scores.loc[unreviewed_mask].sort_values(PROB_COL, ascending=False).reset_index(drop=True)
overall_top_unreviewed = unreviewed_scores.head(TOP_UNREVIEWED_N).copy()
overall_top_unreviewed["rank_by_eb_probability"] = np.arange(1, len(overall_top_unreviewed) + 1)
overall_top_unreviewed.to_csv(MODEL_DIR / f"top{TOP_UNREVIEWED_N}_unreviewed_by_eb_probability.csv", index=False)

without_external_hint = unreviewed_scores.loc[~unreviewed_scores["known_eb_hint"].fillna(False).astype(bool)].copy()
without_external_hint = without_external_hint.sort_values(PROB_COL, ascending=False).reset_index(drop=True)
top_without_external_hint = without_external_hint.head(TOP_UNREVIEWED_N).copy()
top_without_external_hint["rank_by_eb_probability"] = np.arange(1, len(top_without_external_hint) + 1)
top_without_external_hint.to_csv(MODEL_DIR / f"top{TOP_UNREVIEWED_N}_unreviewed_without_external_eb_hint.csv", index=False)
without_external_hint.head(1000).to_csv(MODEL_DIR / "high_priority_without_external_eb_hint.csv", index=False)

plotted_queue = top_without_external_hint if EXCLUDE_KNOWN_EB_FROM_PLOTTED_QUEUE else overall_top_unreviewed
plotted_queue_name = "unreviewed without an external EB hint" if EXCLUDE_KNOWN_EB_FROM_PLOTTED_QUEUE else "human-unreviewed EB-like"

label_audit = {
    "n_candidates": int(len(table)),
    "n_reviewed": int(reviewed.sum()),
    "n_trainable": int(trainable.sum()),
    "target_counts": {str(key): int(value) for key, value in label_counts.items()},
    "label_source_counts": {str(key): int(value) for key, value in table["eb_label_source"].value_counts().items()},
    "n_unreviewed": int(len(unreviewed_scores)),
    "n_unreviewed_without_external_eb_hint": int(len(without_external_hint)),
}
(BASE_DIR / "label_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True) + "\n")

print(f"Scored candidates: {len(scores):,}")
print(f"Human-unreviewed candidates: {len(unreviewed_scores):,}")
print(f"Unreviewed without an external EB hint: {len(without_external_hint):,}")
display(plotted_queue.head(30))


# ## Held-Out Human-Label Performance
# 
# These metrics test recovery of withheld human `eclipsing_like` and binary-like review tags. The confusion matrix uses the model's default class decision; queue ranking is better assessed with ROC, precision-recall, and top-K recovery.
# 

# In[7]:


test_predictions = pd.read_parquet(MODEL_DIR / "test_predictions.parquet")
confusion = pd.read_csv(MODEL_DIR / "confusion_matrix.csv")
cv_metrics = pd.read_csv(MODEL_DIR / "cv_metrics.csv")

y_true = test_predictions["y_true"].eq(POSITIVE_LABEL).astype(int)
y_score = pd.to_numeric(test_predictions[PROB_COL], errors="coerce")
fpr, tpr, _ = roc_curve(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)
average_precision = average_precision_score(y_true, y_score)
baseline = float(y_true.mean())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
cm = confusion.set_index("y_true")
plot_fraction_count_heatmap(cm, cmap="Blues", ax=axes[0])
axes[0].set_title("Held-out confusion matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

axes[1].plot(fpr, tpr, color="#216869", lw=2, label=f"AUC={roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], color="0.5", ls="--")
axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC")
axes[1].legend()

axes[2].plot(recall, precision, color="#a44200", lw=2, label=f"AP={average_precision:.3f}")
axes[2].axhline(baseline, color="0.5", ls="--", label=f"baseline={baseline:.3f}")
axes[2].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
axes[2].legend()

fig.tight_layout()
fig.savefig(FIG_DIR / "heldout_performance.pdf", dpi=180)
plt.show()

ranked_test = test_predictions.assign(is_eb=y_true, eb_score_model=y_score).sort_values("eb_score_model", ascending=False)
top_k_rows = []
for top_k in (50, 100, 250, 500, 1000):
    selected = ranked_test.head(top_k)
    positives_found = int(selected["is_eb"].sum())
    top_k_rows.append(
        {
            "top_k": top_k,
            "human_eb_labels_found": positives_found,
            "precision_at_k": positives_found / len(selected),
            "recall_at_k": positives_found / max(int(y_true.sum()), 1),
        }
    )
display(pd.DataFrame(top_k_rows))
display(Markdown(f"**Held-out human EB-label ranking:** ROC AUC={roc_auc:.3f}, average precision={average_precision:.3f}, baseline={baseline:.3f}."))
display(cv_metrics.drop(columns=["pr_auc"], errors="ignore"))
display(Markdown("The shared helper's `pr_auc` column is omitted because binary class ordering does not reliably identify the intended positive class."))


# ## Feature Importance: Gain Versus Held-Out Permutation
# 

# In[8]:


gain_importance = pd.read_csv(MODEL_DIR / "feature_importance_gain.csv")
bundle = load_target_model(MODEL_DIR)
test_ids = set(test_predictions["candidate_id"].astype(str))
test_frame = model_input.loc[model_input["candidate_id"].astype(str).isin(test_ids)].copy()
X_test = transform_features(
    test_frame,
    feature_columns=list(bundle["feature_columns"]),
    categorical_maps=bundle["categorical_maps"],
)
y_test = test_frame[TARGET_COLUMN].eq(POSITIVE_LABEL).astype(int).to_numpy()
positive_index = list(bundle["label_classes"]).index(POSITIVE_LABEL)
model = bundle["model"]
base_probability = model.predict_proba(X_test)[:, positive_index]
base_ap = average_precision_score(y_test, base_probability)
base_auc = roc_auc_score(y_test, base_probability)

rng = np.random.default_rng(RANDOM_STATE)
X_values = X_test.to_numpy(copy=True)
permutation_rows = []
for feature_index, feature in enumerate(X_test.columns):
    original = X_values[:, feature_index].copy()
    ap_drops = []
    auc_drops = []
    for _ in range(PERMUTATION_REPEATS):
        shuffled = original.copy()
        rng.shuffle(shuffled)
        X_values[:, feature_index] = shuffled
        probability = model.predict_proba(X_values)[:, positive_index]
        ap_drops.append(base_ap - average_precision_score(y_test, probability))
        auc_drops.append(base_auc - roc_auc_score(y_test, probability))
    X_values[:, feature_index] = original
    permutation_rows.append(
        {
            "feature": feature,
            "permutation_auc_drop_mean": float(np.mean(auc_drops)),
            "permutation_auc_drop_std": float(np.std(auc_drops, ddof=1)),
            "permutation_ap_drop_mean": float(np.mean(ap_drops)),
            "permutation_ap_drop_std": float(np.std(ap_drops, ddof=1)),
        }
    )

permutation_importance = pd.DataFrame(permutation_rows).sort_values(
    "permutation_auc_drop_mean", ascending=False, ignore_index=True
)
permutation_importance.to_csv(MODEL_DIR / "feature_importance_permutation.csv", index=False)

gain_plot = gain_importance.head(FEATURE_IMPORTANCE_TOP_N).sort_values("gain")
perm_plot = permutation_importance.head(FEATURE_IMPORTANCE_TOP_N).sort_values(
    "permutation_auc_drop_mean"
)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
axes[0].barh(gain_plot["feature"], gain_plot["gain"], color="#326273")
axes[0].set_title(f"Top {FEATURE_IMPORTANCE_TOP_N} LightGBM gain features")
axes[0].set_xlabel("Gain")
axes[1].barh(
    perm_plot["feature"],
    perm_plot["permutation_auc_drop_mean"],
    xerr=perm_plot["permutation_auc_drop_std"],
    color="#a44200",
    alpha=0.9,
)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title(
    f"Top {FEATURE_IMPORTANCE_TOP_N} held-out permutation features"
)
axes[1].set_xlabel("Drop in ROC AUC when shuffled")
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_gain_vs_permutation.pdf", dpi=180)
plt.show()

importance_comparison = gain_importance.merge(permutation_importance, on="feature", how="outer")
display(importance_comparison.sort_values("permutation_auc_drop_mean", ascending=False).head(30))


# ## External Catalog Agreement And Disagreement
# 
# Gaia/VSX EB hints are evaluated only after model training. This section measures their agreement with the human review labels and exports the scientifically useful disagreements: human EB-like objects missing an external hint, external EB hints rejected by review, and high-model-score unreviewed objects without an external hint.
# 

# In[9]:


trainable_scores = scores.loc[scores[TARGET_COLUMN].notna()].copy()
trainable_scores["human_eb_label"] = trainable_scores[TARGET_COLUMN].eq(POSITIVE_LABEL)
trainable_scores["external_eb_label"] = trainable_scores["known_eb_hint"].fillna(False).astype(bool)

catalog_confusion = pd.crosstab(
    trainable_scores["human_eb_label"].map({False: "human non-EB", True: "human EB-like"}),
    trainable_scores["external_eb_label"].map({False: "no external EB hint", True: "external EB hint"}),
)
display(catalog_confusion)

true_positive = int((trainable_scores["human_eb_label"] & trainable_scores["external_eb_label"]).sum())
false_positive = int((~trainable_scores["human_eb_label"] & trainable_scores["external_eb_label"]).sum())
false_negative = int((trainable_scores["human_eb_label"] & ~trainable_scores["external_eb_label"]).sum())
catalog_metrics = pd.DataFrame(
    {
        "metric": ["external precision vs human", "external recall vs human", "external agreement"],
        "value": [
            true_positive / max(true_positive + false_positive, 1),
            true_positive / max(true_positive + false_negative, 1),
            float((trainable_scores["human_eb_label"] == trainable_scores["external_eb_label"]).mean()),
        ],
    }
)
display(catalog_metrics)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_fraction_count_heatmap(catalog_confusion, cmap="Blues", ax=axes[0])
axes[0].set_title("External EB hints versus human review")
axes[0].set_xlabel("Gaia/VSX comparison label")
axes[0].set_ylabel("Human training label")

comparison_groups = np.select(
    [
        scores["human_eb_like"].fillna(False).astype(bool),
        scores[TARGET_COLUMN].eq(NEGATIVE_LABEL),
        unreviewed_mask & scores["known_eb_hint"].fillna(False).astype(bool),
        unreviewed_mask & ~scores["known_eb_hint"].fillna(False).astype(bool),
    ],
    ["human EB-like", "human non-EB", "unreviewed + external EB", "unreviewed + no external EB"],
    default="other",
)
scores["comparison_group"] = comparison_groups
for group, color in [
    ("human EB-like", "#a44200"),
    ("human non-EB", "#777777"),
    ("unreviewed + external EB", "#607744"),
    ("unreviewed + no external EB", "#326273"),
]:
    values = scores.loc[scores["comparison_group"].eq(group), PROB_COL].dropna()
    if values.empty:
        continue
    weights = np.ones(len(values), dtype=float) / len(values)
    axes[1].hist(values, bins=np.linspace(0, 1, 41), weights=weights, histtype="step", lw=2, label=f"{group} (n={len(values):,})", color=color)
axes[1].set(xlabel="Predicted EB-like probability", ylabel="Fraction of group", title="ML scores by human/external context")
axes[1].set_yscale("log")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "external_catalog_comparison.pdf", dpi=180)
plt.show()

human_eb_without_external = scores.loc[
    scores["human_eb_like"].fillna(False).astype(bool)
    & ~scores["known_eb_hint"].fillna(False).astype(bool)
].sort_values(PROB_COL, ascending=False)
external_eb_human_non_eb = scores.loc[
    scores[TARGET_COLUMN].eq(NEGATIVE_LABEL)
    & scores["known_eb_hint"].fillna(False).astype(bool)
].sort_values(PROB_COL, ascending=False)
high_ml_without_external = scores.loc[
    unreviewed_mask
    & ~scores["known_eb_hint"].fillna(False).astype(bool)
].sort_values(PROB_COL, ascending=False)

human_eb_without_external.to_csv(MODEL_DIR / "human_eb_like_without_external_catalog_hint.csv", index=False)
external_eb_human_non_eb.to_csv(MODEL_DIR / "external_eb_hint_human_non_eb.csv", index=False)
high_ml_without_external.head(1000).to_csv(MODEL_DIR / "high_ml_score_without_external_eb_hint.csv", index=False)

print(f"Human EB-like without external EB hint: {len(human_eb_without_external):,}")
print(f"External EB hint but human non-EB: {len(external_eb_human_non_eb):,}")
display(human_eb_without_external[["candidate_id", PROB_COL, "human_eb_tags", "gaia_var_class", "vsx_class"]].head(40))
display(external_eb_human_non_eb[["candidate_id", PROB_COL, "event_class", "morphology_secondary", "gaia_var_class", "vsx_class"]].head(40))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    pd.to_numeric(scores.get("eb_score"), errors="coerce"),
    scores[PROB_COL],
    c=scores["known_eb_hint"].fillna(False).astype(int),
    cmap="coolwarm",
    s=8,
    alpha=0.25,
)
ax.set(xlabel="Existing deterministic eb_score", ylabel="Predicted EB-like probability", title="Human-label ML score versus deterministic triage")
fig.tight_layout()
fig.savefig(FIG_DIR / "model_vs_deterministic_eb_score.pdf", dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(plotted_queue["rank_by_eb_probability"], plotted_queue[PROB_COL], color="#326273", lw=1.5)
ax.scatter(plotted_queue["rank_by_eb_probability"], plotted_queue[PROB_COL], s=12, color="#a44200", alpha=0.7)
ax.set(xlabel="Rank in plotted unreviewed queue", ylabel="Predicted EB-like probability", title=f"Top {len(plotted_queue)} {plotted_queue_name} candidates")
fig.tight_layout()
fig.savefig(FIG_DIR / f"top{TOP_UNREVIEWED_N}_plotted_queue_scores.pdf", dpi=180)
plt.show()

display(plotted_queue[[column for column in ["rank_by_eb_probability", "candidate_id", PROB_COL, "periodicity_period", "eb_score", "known_eb_hint", "gaia_var_class", "vsx_class"] if column in plotted_queue.columns]].head(100))


# ## Raw And Phase-Folded Light-Curve Helpers
# 
# Raw panels preserve the observing timeline and camera structure. Phase panels use the review-native preferred period resolver while excluding the unsafe raw Lomb-Scargle period. Points are shown over two cycles; the black curve is a median phase template. EB periods often require a 2x harmonic check when primary and secondary eclipses are folded together.
# 

# In[10]:


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


@lru_cache(maxsize=96)
def load_cached_lightcurve(path_text: str) -> pd.DataFrame:
    return load_lightcurve_df(Path(path_text), apply_quality=True)


def finite_number(value: object, default: float = np.nan) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) and np.isfinite(number) else default


def period_for_row(row: pd.Series) -> tuple[float | None, str]:
    return resolve_phase_period(row.to_dict(), include_lsp=False, include_periodogram_periods=True)


def load_row_lightcurve(row: pd.Series) -> tuple[pd.DataFrame | None, Path | None, str | None]:
    path = resolve_bundle_lightcurve(row)
    if path is None:
        return None, None, "missing light curve"
    try:
        lightcurve = load_cached_lightcurve(str(path))
    except Exception as exc:
        return None, path, f"load failed: {exc}"
    if lightcurve.empty:
        return None, path, "empty light curve"
    return lightcurve, path, None


def panel_title(row: pd.Series, rank_column: str) -> tuple[str, str]:
    rank = int(row[rank_column])
    probability = finite_number(row.get(PROB_COL))
    period, source = period_for_row(row)
    period_text = f"P={period:.5g} d" if period is not None else "P=missing"
    known = "known EB" if bool(row.get("known_eb_hint", False)) else "no EB hint"
    return f"{rank:03d} {row['candidate_id']}", f"p={probability:.3f}, {period_text}, {known}"


def camera_band_series(lightcurve: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    camera = lightcurve.get("camera_name", lightcurve.get("camera", pd.Series("all", index=lightcurve.index))).fillna("all").astype(str)
    band = lightcurve.get("band", pd.Series("", index=lightcurve.index)).fillna("").astype(str).str.lower()
    return camera, band


def plot_raw_panel(ax: plt.Axes, row: pd.Series, rank_column: str) -> None:
    title, subtitle = panel_title(row, rank_column)
    lightcurve, _, error = load_row_lightcurve(row)
    if error is not None or lightcurve is None:
        ax.text(0.5, 0.5, error or "unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_axis_off()
        return
    jd = pd.to_numeric(lightcurve["jd"], errors="coerce")
    mag = pd.to_numeric(lightcurve["mag"], errors="coerce")
    camera, band = camera_band_series(lightcurve)
    finite = jd.notna() & mag.notna()
    for camera_name, indices in camera.loc[finite].groupby(camera.loc[finite]).groups.items():
        indices = pd.Index(indices)
        marker = "s" if band.loc[indices].eq("v").any() and not band.loc[indices].eq("g").any() else "o"
        ax.scatter(jd.loc[indices] - 2450000.0, mag.loc[indices], marker=marker, s=4.5, alpha=0.67, color=stable_camera_color(str(camera_name)), linewidths=0)
    ax.invert_yaxis()
    ax.set_title(f"{title}\n{subtitle}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def aligned_phase_magnitudes(lightcurve: pd.DataFrame, mag: pd.Series, band: pd.Series) -> pd.Series:
    aligned = mag.copy()
    g_mask = band.eq("g") & mag.notna()
    v_mask = band.eq("v") & mag.notna()
    if int(g_mask.sum()) >= 5 and int(v_mask.sum()) >= 5:
        offset = float(mag.loc[v_mask].median() - mag.loc[g_mask].median())
        aligned.loc[v_mask] = aligned.loc[v_mask] - offset
    return aligned


def plot_phase_panel(ax: plt.Axes, row: pd.Series, rank_column: str) -> None:
    title, subtitle = panel_title(row, rank_column)
    period, source = period_for_row(row)
    lightcurve, _, error = load_row_lightcurve(row)
    if error is not None or lightcurve is None or period is None or period <= 0:
        message = error or "missing review-safe period"
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_axis_off()
        return
    jd = pd.to_numeric(lightcurve["jd"], errors="coerce")
    mag = pd.to_numeric(lightcurve["mag"], errors="coerce")
    camera, band = camera_band_series(lightcurve)
    mag = aligned_phase_magnitudes(lightcurve, mag, band)
    finite = jd.notna() & mag.notna()
    epoch = float(jd.loc[finite].min())
    phase = ((jd - epoch) / float(period)) % 1.0
    for camera_name, indices in camera.loc[finite].groupby(camera.loc[finite]).groups.items():
        indices = pd.Index(indices)
        color = stable_camera_color(str(camera_name))
        ax.scatter(phase.loc[indices], mag.loc[indices], s=4.2, alpha=0.55, color=color, linewidths=0)
        ax.scatter(phase.loc[indices] + 1.0, mag.loc[indices], s=4.2, alpha=0.55, color=color, linewidths=0)
    template, counts = phase_template(phase.loc[finite].to_numpy(), mag.loc[finite].to_numpy(), n_bins=48, min_bin_points=3)
    centers = (np.arange(len(template)) + 0.5) / len(template)
    valid_template = np.isfinite(template) & (counts >= 3)
    if int(valid_template.sum()) >= 3:
        for offset in (0.0, 1.0):
            ax.plot(centers[valid_template] + offset, template[valid_template], color="white", lw=3.0, zorder=4)
            ax.plot(centers[valid_template] + offset, template[valid_template], color="black", lw=1.2, zorder=5)
    ax.set_xlim(0, 2)
    ax.invert_yaxis()
    ax.set_title(f"{title}\n{subtitle}\n{source}", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def render_dual_lightcurve_pages(
    data: pd.DataFrame,
    *,
    rank_column: str,
    output_root: Path,
    file_prefix: str,
    page_title: str,
    rebuild: bool,
) -> tuple[list[Path], list[Path]]:
    raw_dir = output_root / "raw"
    phase_dir = output_root / "phase"
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    expected_pages = math.ceil(len(data) / PER_PAGE)
    raw_paths = [raw_dir / f"{file_prefix}_raw_page{page:02d}.pdf" for page in range(1, expected_pages + 1)]
    phase_paths = [phase_dir / f"{file_prefix}_phase_page{page:02d}.pdf" for page in range(1, expected_pages + 1)]
    if not rebuild and raw_paths and all(path.exists() for path in [*raw_paths, *phase_paths]):
        print(f"Reusing {len(raw_paths)} raw and {len(phase_paths)} phase pages in {output_root}")
        return raw_paths, phase_paths

    raw_paths = []
    phase_paths = []
    for page_index, start in enumerate(range(0, len(data), PER_PAGE), start=1):
        page = data.iloc[start:start + PER_PAGE]
        for mode, plotter, directory, suffix in (
            ("raw time", plot_raw_panel, raw_dir, "raw"),
            ("phase folded", plot_phase_panel, phase_dir, "phase"),
        ):
            fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=False, sharey=False)
            axes = axes.ravel()
            for ax, (_, row) in zip(axes, page.iterrows()):
                plotter(ax, row, rank_column)
            for ax in axes[len(page):]:
                ax.set_axis_off()
            fig.supxlabel("JD - 2450000" if suffix == "raw" else "Phase (two cycles)", fontsize=11)
            fig.supylabel("ASAS-SN magnitude", fontsize=11)
            fig.suptitle(f"{page_title}, {mode}: page {page_index} of {expected_pages}", y=0.995, fontsize=14)
            fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])
            path = directory / f"{file_prefix}_{suffix}_page{page_index:02d}.pdf"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            (raw_paths if suffix == "raw" else phase_paths).append(path)
        if page_index == 1 or page_index % 5 == 0 or page_index == expected_pages:
            print(f"Rendered raw+phase page {page_index}/{expected_pages}")
    return raw_paths, phase_paths


# ## Top 500 Human-Unreviewed EB-Like Light Curves

# In[11]:


plotted_queue = plotted_queue.copy()
plotted_queue["resolved_lightcurve_path"] = plotted_queue.apply(resolve_bundle_lightcurve, axis=1).map(lambda value: str(value) if value else "")
plotted_queue["resolved_phase_period"] = plotted_queue.apply(lambda row: period_for_row(row)[0], axis=1)
plotted_queue["resolved_phase_source"] = plotted_queue.apply(lambda row: period_for_row(row)[1], axis=1)
plotted_queue.to_csv(MODEL_DIR / f"top{TOP_UNREVIEWED_N}_plotted_eb_queue_with_lightcurve_paths.csv", index=False)

top_raw_paths, top_phase_paths = render_dual_lightcurve_pages(
    plotted_queue,
    rank_column="rank_by_eb_probability",
    output_root=FIG_DIR / f"top{TOP_UNREVIEWED_N}_eb_lightcurves",
    file_prefix=f"top{TOP_UNREVIEWED_N}_eb_lightcurves",
    page_title=f"Top {TOP_UNREVIEWED_N} {plotted_queue_name}",
    rebuild=REBUILD_LIGHTCURVE_PAGES or needs_training,
)

missing_lightcurves = int(plotted_queue["resolved_lightcurve_path"].eq("").sum())
missing_periods = int(plotted_queue["resolved_phase_period"].isna().sum())
print(f"Saved {len(top_raw_paths)} raw and {len(top_phase_paths)} phase pages")
print(f"Missing light curves: {missing_lightcurves}; missing review-safe periods: {missing_periods}")
if top_raw_paths:
    display(Image(filename=str(top_raw_paths[0]), width=1100))
if top_phase_paths:
    display(Image(filename=str(top_phase_paths[0]), width=1100))
display(plotted_queue[["rank_by_eb_probability", "candidate_id", PROB_COL, "eb_score", "resolved_phase_period", "resolved_phase_source", "resolved_lightcurve_path"]].head(30))


# ## Every Reviewed EB-Like Candidate With Its Model Score
# 

# In[12]:


reviewed_eb = scores.loc[scores["human_eb_like"].fillna(False).astype(bool)].sort_values(PROB_COL, ascending=False).reset_index(drop=True)
reviewed_eb["rank_by_eb_probability"] = np.arange(1, len(reviewed_eb) + 1)
reviewed_eb["resolved_lightcurve_path"] = reviewed_eb.apply(resolve_bundle_lightcurve, axis=1).map(lambda value: str(value) if value else "")
reviewed_eb["resolved_phase_period"] = reviewed_eb.apply(lambda row: period_for_row(row)[0], axis=1)
reviewed_eb["resolved_phase_source"] = reviewed_eb.apply(lambda row: period_for_row(row)[1], axis=1)
reviewed_eb.to_csv(MODEL_DIR / "all_reviewed_eb_like_candidates_with_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.8))
scatter = ax.scatter(
    reviewed_eb["rank_by_eb_probability"],
    reviewed_eb[PROB_COL],
    c=reviewed_eb["known_eb_hint"].fillna(False).astype(int),
    cmap="coolwarm",
    s=34,
    edgecolor="black",
    linewidth=0.2,
)
ax.plot(reviewed_eb["rank_by_eb_probability"], reviewed_eb[PROB_COL], color="0.5", lw=1)
ax.set(xlabel="Rank within human-reviewed EB-like cohort", ylabel="Predicted EB-like probability", title="All human-reviewed EB-like candidates and model scores")
fig.colorbar(scatter, ax=ax, ticks=[0, 1], label="external Gaia/VSX EB hint")
fig.tight_layout()
fig.savefig(FIG_DIR / "all_reviewed_eb_like_scores.pdf", dpi=180)
plt.show()

reviewed_raw_paths, reviewed_phase_paths = render_dual_lightcurve_pages(
    reviewed_eb,
    rank_column="rank_by_eb_probability",
    output_root=FIG_DIR / "reviewed_eb_like_lightcurves",
    file_prefix="reviewed_eb_like_lightcurves",
    page_title="Human-reviewed EB-like candidates ranked by predicted EB-like probability",
    rebuild=REBUILD_LIGHTCURVE_PAGES or needs_training,
)
print(f"Saved {len(reviewed_raw_paths)} raw and {len(reviewed_phase_paths)} phase pages for {len(reviewed_eb)} human-reviewed EB-like candidates")
if reviewed_raw_paths:
    display(Image(filename=str(reviewed_raw_paths[0]), width=1100))
if reviewed_phase_paths:
    display(Image(filename=str(reviewed_phase_paths[0]), width=1100))
display(reviewed_eb[["rank_by_eb_probability", "candidate_id", PROB_COL, "human_eb_tags", "known_eb_hint", "gaia_var_class", "vsx_class", "resolved_phase_period", "resolved_phase_source", "resolved_lightcurve_path"]])


# ## Deterministic Feature Windows
# 
# For each numeric compute-stats feature, these central windows retain 100%, 95%, 90%, or 80% of the human-reviewed EB-like positives. They are one-feature diagnostics, not automatically combined cuts.
# 

# In[13]:


WINDOW_COVERAGES = (1.0, 0.95, 0.90, 0.80)
target = table[TARGET_COLUMN].fillna("").astype(str)
positive_mask = target.eq(POSITIVE_LABEL)
negative_mask = target.eq(NEGATIVE_LABEL)
window_rows = []

for feature in feature_columns:
    values = pd.to_numeric(table[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    positive_values = values.loc[positive_mask].dropna()
    if positive_values.empty:
        continue
    for coverage in WINDOW_COVERAGES:
        tail = (1.0 - coverage) / 2.0
        lo = float(positive_values.quantile(tail))
        hi = float(positive_values.quantile(1.0 - tail))
        inside = values.between(lo, hi, inclusive="both")
        positive_kept = int((inside & positive_mask).sum())
        negative_kept = int((inside & negative_mask).sum())
        reviewed_kept = positive_kept + negative_kept
        all_kept = int(inside.sum())
        window_rows.append(
            {
                "feature": feature,
                "target_coverage": coverage,
                "lo": lo,
                "hi": hi,
                "positive_kept": positive_kept,
                "positive_total": int(positive_mask.sum()),
                "positive_recall": positive_kept / max(int(positive_mask.sum()), 1),
                "clear_non_eb_kept": negative_kept,
                "human_label_precision": positive_kept / reviewed_kept if reviewed_kept else np.nan,
                "all_candidates_kept": all_kept,
                "all_candidate_fraction_kept": all_kept / len(table),
            }
        )

feature_windows = pd.DataFrame(window_rows).sort_values(
    ["target_coverage", "positive_recall", "all_candidate_fraction_kept", "human_label_precision"],
    ascending=[False, False, True, False],
    ignore_index=True,
)
feature_windows.to_csv(MODEL_DIR / "feature_windows.csv", index=False)

top_gain_features = set(gain_importance.head(30)["feature"])
window_view = feature_windows.loc[
    feature_windows["feature"].isin(top_gain_features)
    & feature_windows["target_coverage"].isin([1.0, 0.90])
].sort_values(["target_coverage", "all_candidate_fraction_kept", "human_label_precision"], ascending=[False, True, False])
display(window_view.head(60))

fig, ax = plt.subplots(figsize=(8, 5.5))
for coverage, color in [(1.0, "#333333"), (0.95, "#607744"), (0.90, "#326273"), (0.80, "#a44200")]:
    subset = feature_windows.loc[feature_windows["target_coverage"].eq(coverage)]
    ax.scatter(subset["all_candidate_fraction_kept"], subset["human_label_precision"], s=24, alpha=0.65, label=f"{coverage:.0%} target")
ax.set(xlabel="Fraction of all candidates inside one-feature window", ylabel="Human EB-label precision inside window", title="One-feature deterministic EB windows")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "deterministic_feature_windows.pdf", dpi=180)
plt.show()


# ## Distributions Of The Most Reproducible Human-Label Features

# In[14]:


top_distribution_features = permutation_importance.loc[
    permutation_importance["permutation_auc_drop_mean"].gt(0), "feature"
].head(8).tolist()
if len(top_distribution_features) < 4:
    top_distribution_features = gain_importance.head(8)["feature"].tolist()

distribution_data = table[[TARGET_COLUMN, *top_distribution_features]].copy()
distribution_data["group"] = distribution_data[TARGET_COLUMN].map({POSITIVE_LABEL: "human EB-like", NEGATIVE_LABEL: "human non-EB"}).fillna("unlabeled")

fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
for ax, feature in zip(axes.ravel(), top_distribution_features):
    plot_data = distribution_data[["group", feature]].copy()
    plot_data[feature] = pd.to_numeric(plot_data[feature], errors="coerce")
    finite_values = plot_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_values.empty:
        lo, hi = finite_values.quantile([0.01, 0.99])
        plot_data[feature] = plot_data[feature].clip(lo, hi)
    sns.boxenplot(
        data=plot_data.loc[plot_data["group"].isin(["human EB-like", "human non-EB"])],
        x="group",
        y=feature,
        order=["human EB-like", "human non-EB"],
        showfliers=False,
        ax=ax,
        palette=["#a44200", "#777777"],
    )
    ax.set_xlabel("")
    ax.set_title(feature, fontsize=9)
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
for ax in axes.ravel()[len(top_distribution_features):]:
    ax.set_axis_off()
fig.suptitle("Top held-out human EB-label feature distributions", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "top_feature_distributions.pdf", dpi=180, bbox_inches="tight")
plt.show()


# ## Interpretation Boundary
# 
# - Training labels come only from human `eclipsing_like` and binary-like review morphology tags.
# - Gaia/VSX labels are comparison-only; inspect both agreement and disagreement exports rather than treating catalogs as truth.
# - Start visual review with `top500_plotted_eb_queue_with_lightcurve_paths.csv`, then compare against `top500_unreviewed_without_external_eb_hint.csv` for potentially uncatalogued systems.
# - Inspect raw and phase-folded panels together. Camera offsets, aliases, half-period solutions, pulsators, and rotational variables can all imitate an EB score.
# - Prefer features supported by both LightGBM gain and held-out permutation importance.
# - Treat `prob_eb_like` as a ranking signal until independent calibration is performed on a held-out review campaign.
# 
