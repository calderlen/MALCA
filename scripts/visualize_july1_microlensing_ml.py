#!/usr/bin/env python
# coding: utf-8

"""Write July 1 microlensing-like ML diagnostics and light-curve figures to results."""

# # July 1 Microlensing ML Visualizations
# 
# This standalone script loads the saved stats-plus-context LightGBM ranker, scores every candidate in the July 1 review database, exports the 500 highest-scored unreviewed candidates, and plots their **actual ASAS-SN light curves**. Retrain with `python scripts/train_july1_microlensing_ml.py` from the repository root.
# 
# The positive class combines reviewed `event_class='microlensing'` rows with reviewed rows tagged `possible_microlensing_event`, matching the definition used by the July 1 nine-class experiment. Ambiguous reviews are excluded from training. External microlensing-catalog matches are retained only for after-the-fact comparison and never enter the model. The resulting `prob_microlensing_like` value is an uncalibrated ranking signal from a small, review-selected positive class, not a population probability or a physical microlensing fit.

# In[ ]:


from __future__ import annotations

import json
import math
import sqlite3
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

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
BASE_DIR = RUN_DIR / "results" / "microlensing_feature_selection"
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

TARGET_COLUMN = "microlensing_like_label"
POSITIVE_LABEL = "microlensing_like"
NEGATIVE_LABEL = "not_microlensing"
PROB_COL = "prob_microlensing_like"

TOP_UNREVIEWED_N = 500
PER_PAGE = 20
REBUILD_LIGHTCURVE_PAGES = False
PERMUTATION_REPEATS = 12
RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
for stale_png in FIG_DIR.rglob("*.png"):
    stale_png.unlink()

print(f"Review DB: {DB_PATH}")
print(f"Model directory: {MODEL_DIR}")


# ## Load The July 1 Review Population
# 
# The model uses only the populated `stats_*` block from `compute_stats()`. Human review fields define labels, while persisted microlensing-catalog columns are loaded strictly as diagnostic context.

# In[ ]:


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
microlens_context_columns = [column for column in table.columns if column.startswith("microlens_")]

display(
    pd.DataFrame(
        {
            "surface": ["all candidates", "review rows", "stats_* columns", "microlens_* context columns"],
            "count": [len(candidates), len(reviews), len(stats_columns), len(microlens_context_columns)],
        }
    )
)
display(reviews["event_class"].fillna("<missing>").value_counts().rename_axis("event_class").reset_index(name="n"))


# ## Define The Training Labels And Leakage-Controlled Features
# 
# Primary microlensing reviews and `possible_microlensing_event` secondary tags are positive. Clear, resolved non-microlensing event classes are negative. Heterogeneous or unresolved classes are left unlabeled, reducing label contamination at the cost of fewer training rows. Catalog crossmatches, review fields, and absolute observing dates are excluded from model features.

# In[ ]:


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


secondary_scalar = clean_text(table, "morphology_secondary")
secondary_tag_sets = table.get(
    "morphology_secondary_json",
    pd.Series("", index=table.index),
).map(parse_secondary_tags)
secondary_tag_sets = pd.Series(
    [tags | ({scalar} if scalar else set()) for tags, scalar in zip(secondary_tag_sets, secondary_scalar)],
    index=table.index,
)

event_class = clean_text(table, "event_class")
status = clean_text(table, "status")
workflow = clean_text(table, "workflow_status")
reviewed = (status.ne("") & status.ne("unreviewed")) | (workflow.ne("") & workflow.ne("unreviewed"))
possible_microlensing_tag = secondary_tag_sets.map(lambda tags: "possible_microlensing_event" in tags)
human_microlensing_like = reviewed & (event_class.eq("microlensing") | possible_microlensing_tag)

CLEAR_NEGATIVE_EVENT_CLASSES = {
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
human_non_microlensing = reviewed & event_class.isin(CLEAR_NEGATIVE_EVENT_CLASSES) & ~human_microlensing_like

table["human_microlensing_like"] = human_microlensing_like.astype(bool)
table[TARGET_COLUMN] = pd.NA
table.loc[human_microlensing_like, TARGET_COLUMN] = POSITIVE_LABEL
table.loc[human_non_microlensing, TARGET_COLUMN] = NEGATIVE_LABEL
table["microlensing_label_source"] = np.select(
    [
        human_microlensing_like & event_class.eq("microlensing") & possible_microlensing_tag,
        human_microlensing_like & event_class.eq("microlensing"),
        human_microlensing_like & possible_microlensing_tag,
        human_non_microlensing,
    ],
    [
        "human_event_and_morphology_microlensing",
        "human_event_class_microlensing",
        "human_morphology_microlensing",
        "human_review_non_microlensing",
    ],
    default="unlabeled",
)

MODEL_EXCLUDED_STATS = {
    *STATS_MODEL_FEATURE_EXCLUSION_COLUMNS,
    "stats_variability_lomb_scargle_best_period_days",
}


def is_usable_stats_feature(series: pd.Series, *, min_non_null: int = 20, max_cardinality: int = 50) -> bool:
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


def drop_exact_duplicate_features(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], dict[str, str]]:
    signature_groups: dict[bytes, list[str]] = {}
    normalized: dict[str, pd.Series] = {}
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        else:
            text = series.fillna("").astype(str).str.strip()
            categories = sorted(value for value in text.unique() if value)
            mapping = {value: index for index, value in enumerate(categories)}
            values = text.map(mapping).fillna(-1).astype("float64")
        normalized[column] = values
        signature = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64").tobytes()
        signature_groups.setdefault(signature, []).append(column)

    aliases: dict[str, str] = {}
    for group in signature_groups.values():
        kept_in_group: list[str] = []
        for column in group:
            duplicate_of = next((kept for kept in kept_in_group if normalized[column].equals(normalized[kept])), None)
            if duplicate_of is None:
                kept_in_group.append(column)
            else:
                aliases[column] = duplicate_of
    return [column for column in columns if column not in aliases], aliases


unusable_context_features = [
    column
    for column in ASTROPHYSICAL_CONTEXT_FEATURES
    if not is_usable_stats_feature(table.loc[trainable, column])
]
if unusable_context_features:
    raise ValueError(f"Requested astrophysical-context features are unusable: {unusable_context_features}")
feature_columns = [*feature_columns, *ASTROPHYSICAL_CONTEXT_FEATURES]

feature_columns, duplicate_feature_aliases = drop_exact_duplicate_features(
    table.loc[trainable],
    feature_columns,
)
model_input = table[["candidate_id", TARGET_COLUMN, *feature_columns]].copy()

label_counts = table.loc[trainable, TARGET_COLUMN].value_counts()
display(label_counts.rename_axis("target").reset_index(name="n"))
display(table.loc[trainable, [TARGET_COLUMN, "microlensing_label_source"]].value_counts().rename("n").reset_index())
print(f"Trainable reviewed rows: {int(trainable.sum()):,}")
print(f"Microlensing-like positives: {int(human_microlensing_like.sum()):,}")
print(f"Selected total model features: {len(feature_columns):,}")
print(f"Astrophysical-context features: {list(ASTROPHYSICAL_CONTEXT_FEATURES)}")
print(f"Dropped exact duplicate feature aliases: {duplicate_feature_aliases}")
print(f"Unresolved reviewed rows excluded from training: {int((reviewed & ~trainable).sum()):,}")


# ## Load The Saved Microlensing-Like Ranker
# 
# Training is owned by `scripts/train_july1_microlensing_ml.py`; rerun this notebook afterward to regenerate its visualizations.

# In[ ]:


model_path = MODEL_DIR / "model.joblib"
if not model_path.exists():
    raise FileNotFoundError(
        f"No saved model at {model_path}. Run scripts/train_july1_microlensing_ml.py from the repository root."
    )
needs_training = False
bundle = load_target_model(MODEL_DIR)
model_input, restored_legacy_features = restore_legacy_excluded_model_features(
    model_input,
    table,
    bundle["feature_columns"],
)
saved_features = list(bundle["feature_columns"])
missing = sorted(set(saved_features) - set(model_input.columns))
if missing:
    raise ValueError(f"Saved model expects missing features: {missing[:10]}")
if restored_legacy_features:
    print(f"Restored legacy excluded features for saved-model scoring: {list(restored_legacy_features)}")
print("Using the saved microlensing-like model. Retrain with scripts/train_july1_microlensing_ml.py.")

metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
display(pd.DataFrame({"value": [metadata["n_rows"], metadata["n_features"], metadata["class_counts"]]}, index=["training rows", "features", "class counts"]))


# ## Score Every Candidate And Build The Unreviewed Queue

# In[ ]:


prediction_frame = score_target_model(MODEL_DIR, model_input)
prediction_columns = ["candidate_id", "y_pred", "prediction_confidence", PROB_COL]

score_context_columns = [
    "candidate_id",
    "asas_sn_id",
    "lc_path",
    "status",
    "workflow_status",
    "event_class",
    "morphology_primary",
    "morphology_secondary",
    "physical_primary",
    "interest_score",
    "human_microlensing_like",
    "microlensing_label_source",
    "microlens_match",
    "microlens_catalog",
    "microlens_name",
    "microlens_alt_name",
    "microlens_te_days",
    "microlens_sep_arcsec",
    "vetting_likely_known",
    "catalog_match",
    "catalog_source",
    "simbad_otype",
    "vsx_class",
    "stats_variability_flux_asymmetry_m",
    "stats_variability_quasi_periodicity_q",
    "stats_amplitude",
    "stats_percent_amplitude",
    "stats_photometry_robust_sigma_mag",
    "stats_variability_reduced_chi2_vs_constant",
    "stats_variability_stetson_J",
    "stats_variability_von_neumann_ratio",
    "stats_sf_ml_gamma",
]
score_context_columns = [column for column in score_context_columns if column in table.columns]
scores = table[score_context_columns].merge(
    prediction_frame[prediction_columns],
    on="candidate_id",
    how="left",
)
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
top_unreviewed = unreviewed_scores.head(TOP_UNREVIEWED_N).copy()
top_unreviewed["rank_by_microlensing_probability"] = np.arange(1, len(top_unreviewed) + 1)
top_unreviewed.to_csv(MODEL_DIR / f"top{TOP_UNREVIEWED_N}_unreviewed_by_microlensing_probability.csv", index=False)
unreviewed_scores.head(1000).to_csv(MODEL_DIR / "high_priority_review_queue.csv", index=False)

label_audit = {
    "n_candidates": int(len(table)),
    "n_reviewed": int(reviewed.sum()),
    "n_trainable": int(trainable.sum()),
    "target_counts": {str(k): int(v) for k, v in label_counts.items()},
    "label_source_counts": {str(k): int(v) for k, v in table.loc[trainable, "microlensing_label_source"].value_counts().items()},
    "event_class_counts": {str(k): int(v) for k, v in event_class.loc[reviewed].value_counts().items()},
    "n_unreviewed_scored": int(len(unreviewed_scores)),
}
(BASE_DIR / "label_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True) + "\n")

print(f"Scored candidates: {len(scores):,}")
print(f"Unreviewed candidates: {len(unreviewed_scores):,}")
display(top_unreviewed.head(25))


# ## Held-Out Performance
# 
# The positive class is small, so each holdout contains only a few microlensing-like examples. Read the precision-recall curve, recall-at-rank, and individual mistakes together; expect substantial metric movement as review labels grow.

# In[ ]:


test_predictions = pd.read_parquet(MODEL_DIR / "test_predictions.parquet")
confusion = pd.read_csv(MODEL_DIR / "confusion_matrix.csv")
cv_metrics = pd.read_csv(MODEL_DIR / "cv_metrics.csv")

y_true = test_predictions["y_true"].eq(POSITIVE_LABEL).astype(int)
y_score = pd.to_numeric(test_predictions[PROB_COL], errors="coerce")
fpr, tpr, _ = roc_curve(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
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

ranked_test = test_predictions.assign(is_microlensing=y_true, microlensing_score=y_score).sort_values("microlensing_score", ascending=False)
top_k_rows = []
for top_k in (5, 10, 20, 50, 100):
    selected = ranked_test.head(top_k)
    positives_found = int(selected["is_microlensing"].sum())
    top_k_rows.append(
        {
            "top_k": top_k,
            "rows_available": len(selected),
            "microlensing_found": positives_found,
            "precision_at_k": positives_found / max(len(selected), 1),
            "recall_at_k": positives_found / max(int(y_true.sum()), 1),
        }
    )
display(pd.DataFrame(top_k_rows))
display(Markdown(f"**Held-out microlensing ranking:** ROC AUC={roc_auc:.3f}, average precision={average_precision:.3f}, random baseline={baseline:.3f}."))
display(cv_metrics.drop(columns=["pr_auc"], errors="ignore"))
display(Markdown("The shared helper's saved `pr_auc` column is omitted because its binary class ordering can evaluate the majority class rather than `microlensing_like`."))


# ## Feature Importance: Gain Versus Held-Out Permutation
# 
# LightGBM gain reports how much a feature improved training-tree splits. Permutation importance asks a stricter question: how much does held-out average precision fall when that feature is scrambled? Features that rank highly in both views are more convincing than features that appear only in gain.

# In[ ]:


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

rng = np.random.default_rng(RANDOM_STATE)
X_permuted = X_test.copy()
permutation_rows = []
for feature_index, feature in enumerate(X_test.columns):
    original = X_permuted.iloc[:, feature_index].to_numpy(copy=True)
    drops = []
    for _ in range(PERMUTATION_REPEATS):
        shuffled = original.copy()
        rng.shuffle(shuffled)
        X_permuted.iloc[:, feature_index] = shuffled
        probability = model.predict_proba(X_permuted)[:, positive_index]
        drops.append(base_ap - average_precision_score(y_test, probability))
    X_permuted.iloc[:, feature_index] = original
    permutation_rows.append(
        {
            "feature": feature,
            "permutation_ap_drop_mean": float(np.mean(drops)),
            "permutation_ap_drop_std": float(np.std(drops, ddof=1)),
        }
    )

permutation_importance = pd.DataFrame(permutation_rows).sort_values(
    "permutation_ap_drop_mean", ascending=False, ignore_index=True
)
permutation_importance.to_csv(MODEL_DIR / "feature_importance_permutation.csv", index=False)

gain_plot = gain_importance.head(FEATURE_IMPORTANCE_TOP_N).sort_values("gain")
perm_plot = permutation_importance.head(FEATURE_IMPORTANCE_TOP_N).sort_values(
    "permutation_ap_drop_mean"
)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
axes[0].barh(gain_plot["feature"], gain_plot["gain"], color="#326273")
axes[0].set_title(f"Top {FEATURE_IMPORTANCE_TOP_N} LightGBM gain features")
axes[0].set_xlabel("Gain")
axes[1].barh(
    perm_plot["feature"],
    perm_plot["permutation_ap_drop_mean"],
    xerr=perm_plot["permutation_ap_drop_std"],
    color="#a44200",
    alpha=0.9,
)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title(
    f"Top {FEATURE_IMPORTANCE_TOP_N} held-out permutation features"
)
axes[1].set_xlabel("Drop in microlensing average precision when shuffled")
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_gain_vs_permutation.pdf", dpi=180)
plt.show()

importance_comparison = gain_importance.merge(permutation_importance, on="feature", how="outer")
display(importance_comparison.sort_values("permutation_ap_drop_mean", ascending=False).head(30))


# ## Score Distribution And Top 500 Unreviewed Candidates

# In[ ]:


score_event = clean_text(scores, "event_class")
score_status = clean_text(scores, "status")
scores["score_group"] = np.select(
    [
        scores["human_microlensing_like"].fillna(False).astype(bool),
        score_status.ne("") & ~scores["human_microlensing_like"].fillna(False).astype(bool),
        unreviewed_mask,
    ],
    ["reviewed microlensing-like", "reviewed non-microlensing", "unreviewed"],
    default="other",
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for group, color in [
    ("unreviewed", "#326273"),
    ("reviewed non-microlensing", "#777777"),
    ("reviewed microlensing-like", "#a44200"),
]:
    values = scores.loc[scores["score_group"].eq(group), PROB_COL].dropna()
    if values.empty:
        continue
    weights = np.ones(len(values), dtype=float) / len(values)
    axes[0].hist(values, bins=np.linspace(0, 1, 41), weights=weights, histtype="step", lw=2, label=f"{group} (n={len(values):,})", color=color)
axes[0].set(xlabel="Predicted microlensing-like probability", ylabel="Fraction of group", title="Model score distribution")
axes[0].set_yscale("log")
axes[0].legend()

axes[1].plot(top_unreviewed["rank_by_microlensing_probability"], top_unreviewed[PROB_COL], color="#326273", lw=1.5)
axes[1].scatter(top_unreviewed["rank_by_microlensing_probability"], top_unreviewed[PROB_COL], s=12, color="#a44200", alpha=0.7)
axes[1].set(
    xlabel="Rank among unreviewed candidates",
    ylabel="Predicted microlensing-like probability",
    title=f"Top {TOP_UNREVIEWED_N} unreviewed candidates",
)
fig.tight_layout()
fig.savefig(FIG_DIR / f"top{TOP_UNREVIEWED_N}_unreviewed_microlensing_scores.pdf", dpi=180)
plt.show()

top_view_columns = [
    "rank_by_microlensing_probability", "candidate_id", "asas_sn_id", PROB_COL,
    "stats_variability_flux_asymmetry_m", "stats_amplitude", "stats_variability_von_neumann_ratio",
    "microlens_match", "microlens_catalog", "microlens_name", "microlens_te_days",
    "vetting_likely_known", "catalog_source", "simbad_otype", "vsx_class",
]
display(top_unreviewed[[column for column in top_view_columns if column in top_unreviewed.columns]].head(100))


# ## External Microlensing-Catalog Context
# 
# Catalog matches did not enter `model_input`. This section measures agreement and highlights disagreements after scoring, so the model cannot learn the answer from a pre-existing microlensing label.

# In[ ]:


def truthy_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.fillna("").astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


scores["external_microlensing_match"] = truthy_series(scores, "microlens_match")
external_summary = scores.groupby("external_microlensing_match").agg(
    n=("candidate_id", "size"),
    mean_model_score=(PROB_COL, "mean"),
    median_model_score=(PROB_COL, "median"),
    reviewed_microlensing_like=("human_microlensing_like", "sum"),
).reset_index()
display(external_summary)

reviewed_context = scores.loc[score_status.ne("")].copy()
display(
    pd.crosstab(
        reviewed_context["human_microlensing_like"].fillna(False).astype(bool).rename("human_microlensing_like"),
        reviewed_context["external_microlensing_match"].rename("external_catalog_match"),
        margins=True,
    )
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for matched, label, color in [(False, "no external match", "#777777"), (True, "external match", "#e7298a")]:
    values = scores.loc[scores["external_microlensing_match"].eq(matched), PROB_COL].dropna()
    if values.empty:
        continue
    axes[0].hist(values, bins=np.linspace(0, 1, 31), histtype="step", lw=2, label=f"{label} (n={len(values):,})", color=color)
axes[0].set(xlabel="Predicted microlensing-like probability", ylabel="Candidates", title="Score versus external catalog match")
axes[0].set_yscale("log")
axes[0].legend()

matched_scores = scores.loc[scores["external_microlensing_match"]].copy()
matched_scores["te_days_num"] = pd.to_numeric(matched_scores.get("microlens_te_days"), errors="coerce")
matched_scores = matched_scores.loc[matched_scores["te_days_num"].gt(0)]
if not matched_scores.empty:
    axes[1].scatter(matched_scores["te_days_num"], matched_scores[PROB_COL], c=matched_scores["human_microlensing_like"].astype(int), cmap="coolwarm", s=42, edgecolor="black", linewidth=0.3)
    axes[1].set_xscale("log")
axes[1].set(xlabel="Catalog event timescale tE [days]", ylabel="Predicted microlensing-like probability", title="Matched-event score versus catalog timescale")
fig.tight_layout()
fig.savefig(FIG_DIR / "external_catalog_agreement.pdf", dpi=180)
plt.show()

catalog_matched_low_score = scores.loc[scores["external_microlensing_match"]].nsmallest(50, PROB_COL)
model_high_no_catalog = unreviewed_scores.loc[~truthy_series(unreviewed_scores, "microlens_match")].head(100)
catalog_matched_low_score.to_csv(MODEL_DIR / "external_matches_low_model_score.csv", index=False)
model_high_no_catalog.to_csv(MODEL_DIR / "top_unreviewed_model_scores_without_external_match.csv", index=False)
display(Markdown("**Catalog-matched rows with the lowest model scores**"))
display(catalog_matched_low_score[[column for column in ["candidate_id", PROB_COL, "event_class", "microlens_catalog", "microlens_name", "microlens_te_days"] if column in catalog_matched_low_score.columns]].head(25))


# ## Actual Light-Curve Plotting Helpers
# 
# Each panel shows quality-filtered ASAS-SN observations colored by camera. The black curve is a short-timescale median track used only as a visual guide to a single broad brightening; it is not a Paczyński fit.

# In[ ]:


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


def finite_number(value: object, default: float = np.nan) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) and np.isfinite(number) else default


def plot_microlensing_lightcurve_panel(ax: plt.Axes, row: pd.Series, rank_column: str) -> None:
    path = resolve_bundle_lightcurve(row)
    rank = int(row[rank_column])
    candidate_id = str(row["candidate_id"])
    probability = finite_number(row.get(PROB_COL))
    asymmetry = finite_number(row.get("stats_variability_flux_asymmetry_m"))
    amplitude = finite_number(row.get("stats_amplitude"))
    title = f"{rank:03d} {candidate_id}"
    subtitle = f"p={probability:.3f}, M={asymmetry:+.2f}, amp={amplitude:.2f} mag"

    if path is None:
        ax.text(0.5, 0.5, "missing light curve", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_axis_off()
        return
    try:
        lightcurve = load_lightcurve_df(path, apply_quality=True)
    except Exception as exc:
        ax.text(0.5, 0.5, f"load failed\n{exc}", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_axis_off()
        return

    jd = pd.to_numeric(lightcurve.get("jd"), errors="coerce")
    mag = pd.to_numeric(lightcurve.get("mag"), errors="coerce")
    camera = lightcurve.get("camera_name", lightcurve.get("camera", pd.Series("all", index=lightcurve.index))).fillna("all").astype(str)
    band = lightcurve.get("band", pd.Series("", index=lightcurve.index)).fillna("").astype(str).str.lower()
    finite = jd.notna() & mag.notna()
    if not bool(finite.any()):
        ax.text(0.5, 0.5, "no finite photometry", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_axis_off()
        return

    for camera_name, indices in camera.loc[finite].groupby(camera.loc[finite]).groups.items():
        indices = pd.Index(indices)
        marker = "s" if band.loc[indices].eq("v").any() and not band.loc[indices].eq("g").any() else "o"
        ax.scatter(
            jd.loc[indices] - 2450000.0,
            mag.loc[indices],
            marker=marker,
            s=5.0,
            alpha=0.68,
            color=stable_camera_color(str(camera_name)),
            linewidths=0,
        )

    track = pd.DataFrame({"jd": jd.loc[finite], "mag": mag.loc[finite]}).sort_values("jd")
    span_days = max(float(track["jd"].max() - track["jd"].min()), 1.0)
    bin_days = float(np.clip(span_days / 120.0, 15.0, 90.0))
    track["bin"] = np.floor((track["jd"] - track["jd"].min()) / bin_days).astype(int)
    binned = track.groupby("bin", as_index=False).agg(jd=("jd", "median"), mag=("mag", "median"), n=("mag", "size"))
    binned = binned.loc[binned["n"].ge(2)]
    if len(binned) >= 2:
        x_track = binned["jd"] - 2450000.0
        ax.plot(x_track, binned["mag"], color="white", lw=3.0, alpha=0.9, zorder=4)
        ax.plot(x_track, binned["mag"], color="black", lw=1.2, alpha=0.9, zorder=5)

    ax.invert_yaxis()
    ax.set_title(f"{title}\n{subtitle}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def render_lightcurve_pages(
    data: pd.DataFrame,
    *,
    rank_column: str,
    output_dir: Path,
    file_prefix: str,
    page_title: str,
    rebuild: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_pages = math.ceil(len(data) / PER_PAGE)
    expected_paths = [output_dir / f"{file_prefix}_page{page:02d}.pdf" for page in range(1, expected_pages + 1)]
    if not rebuild and expected_paths and all(path.exists() for path in expected_paths):
        print(f"Reusing {len(expected_paths)} existing light-curve pages in {output_dir}")
        return expected_paths

    page_paths = []
    for page_index, start in enumerate(range(0, len(data), PER_PAGE), start=1):
        page = data.iloc[start:start + PER_PAGE]
        fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, (_, row) in zip(axes, page.iterrows()):
            plot_microlensing_lightcurve_panel(ax, row, rank_column)
        for ax in axes[len(page):]:
            ax.set_axis_off()
        fig.supxlabel("JD - 2450000", fontsize=11)
        fig.supylabel("ASAS-SN magnitude", fontsize=11)
        fig.suptitle(f"{page_title}: page {page_index} of {expected_pages}", y=0.995, fontsize=14)
        fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])
        page_path = output_dir / f"{file_prefix}_page{page_index:02d}.pdf"
        fig.savefig(page_path, dpi=160)
        plt.close(fig)
        page_paths.append(page_path)
        if page_index == 1 or page_index % 5 == 0 or page_index == expected_pages:
            print(f"Rendered page {page_index}/{expected_pages}")
    return page_paths


# ## Top 500 Probability-Ranked Unreviewed Light Curves

# In[ ]:


top_unreviewed = unreviewed_scores.head(TOP_UNREVIEWED_N).copy()
top_unreviewed["rank_by_microlensing_probability"] = np.arange(1, len(top_unreviewed) + 1)
top_unreviewed["resolved_lightcurve_path"] = top_unreviewed.apply(resolve_bundle_lightcurve, axis=1).map(lambda value: str(value) if value else "")
top_unreviewed.to_csv(
    MODEL_DIR / f"top{TOP_UNREVIEWED_N}_unreviewed_by_microlensing_probability_with_lightcurve_paths.csv",
    index=False,
)

top_page_dir = FIG_DIR / f"top{TOP_UNREVIEWED_N}_probability_lightcurves"
top_page_paths = render_lightcurve_pages(
    top_unreviewed,
    rank_column="rank_by_microlensing_probability",
    output_dir=top_page_dir,
    file_prefix=f"top{TOP_UNREVIEWED_N}_microlensing_probability_lightcurves",
    page_title=f"Top {TOP_UNREVIEWED_N} unreviewed by predicted microlensing-like probability",
    rebuild=REBUILD_LIGHTCURVE_PAGES or needs_training,
)

missing_paths = int(top_unreviewed["resolved_lightcurve_path"].eq("").sum())
print(f"Saved {len(top_page_paths)} pages; missing light curves: {missing_paths}")
if top_page_paths:
    display(Image(filename=str(top_page_paths[0]), width=1100))
display(top_unreviewed[["rank_by_microlensing_probability", "candidate_id", PROB_COL, "stats_variability_flux_asymmetry_m", "stats_amplitude", "resolved_lightcurve_path"]].head(25))


# ## Every Reviewed Microlensing-Like Candidate With Its Model Score

# In[ ]:


reviewed_microlensing = scores.loc[scores["human_microlensing_like"].fillna(False).astype(bool)].sort_values(PROB_COL, ascending=False).reset_index(drop=True)
reviewed_microlensing["rank_by_microlensing_probability"] = np.arange(1, len(reviewed_microlensing) + 1)
reviewed_microlensing["resolved_lightcurve_path"] = reviewed_microlensing.apply(resolve_bundle_lightcurve, axis=1).map(lambda value: str(value) if value else "")
reviewed_microlensing.to_csv(MODEL_DIR / "all_reviewed_microlensing_candidates_with_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.8))
catalog_color = truthy_series(reviewed_microlensing, "microlens_match").astype(int)
scatter = ax.scatter(reviewed_microlensing["rank_by_microlensing_probability"], reviewed_microlensing[PROB_COL], c=catalog_color, cmap="coolwarm", s=52, edgecolor="black", linewidth=0.3)
ax.plot(reviewed_microlensing["rank_by_microlensing_probability"], reviewed_microlensing[PROB_COL], color="0.5", lw=1)
ax.set(xlabel="Rank within reviewed microlensing-like candidates", ylabel="Predicted microlensing-like probability", title="All reviewed microlensing-like candidates and model scores")
colorbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
colorbar.set_label("External microlensing catalog match")
fig.tight_layout()
fig.savefig(FIG_DIR / "all_reviewed_microlensing_scores.pdf", dpi=180)
plt.show()

reviewed_page_paths = render_lightcurve_pages(
    reviewed_microlensing,
    rank_column="rank_by_microlensing_probability",
    output_dir=FIG_DIR / "reviewed_microlensing_lightcurves",
    file_prefix="reviewed_microlensing_lightcurves",
    page_title="Reviewed microlensing-like candidates ranked by model score",
    rebuild=REBUILD_LIGHTCURVE_PAGES or needs_training,
)
print(f"Saved {len(reviewed_page_paths)} reviewed-microlensing light-curve pages for {len(reviewed_microlensing)} candidates")
if reviewed_page_paths:
    display(Image(filename=str(reviewed_page_paths[0]), width=1100))
display(reviewed_microlensing[[column for column in ["rank_by_microlensing_probability", "candidate_id", PROB_COL, "event_class", "morphology_secondary", "microlens_match", "microlens_catalog", "resolved_lightcurve_path"] if column in reviewed_microlensing.columns]])


# ## Deterministic Feature Windows
# 
# For each numeric compute-stats feature, these central windows retain 100%, 95%, 90%, or 80% of reviewed microlensing-like positives with non-missing values. They are one-feature diagnostics, not automatically combined cuts or physical event-selection rules.

# In[ ]:


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
                "reviewed_nonmicrolensing_kept": negative_kept,
                "reviewed_precision": positive_kept / reviewed_kept if reviewed_kept else np.nan,
                "all_candidates_kept": all_kept,
                "all_candidate_fraction_kept": all_kept / len(table),
                "positive_nonmissing": int(positive_values.size),
            }
        )

feature_windows = pd.DataFrame(window_rows).sort_values(
    ["target_coverage", "positive_recall", "all_candidate_fraction_kept", "reviewed_precision"],
    ascending=[False, False, True, False],
    ignore_index=True,
)
feature_windows.to_csv(MODEL_DIR / "feature_windows.csv", index=False)

top_gain_features = set(gain_importance.head(30)["feature"])
window_view = feature_windows.loc[
    feature_windows["feature"].isin(top_gain_features)
    & feature_windows["target_coverage"].isin([1.0, 0.90])
].sort_values(["target_coverage", "all_candidate_fraction_kept", "reviewed_precision"], ascending=[False, True, False])
display(window_view.head(60))

fig, ax = plt.subplots(figsize=(8, 5.5))
for coverage, color in [(1.0, "#333333"), (0.95, "#607744"), (0.90, "#326273"), (0.80, "#a44200")]:
    subset = feature_windows.loc[feature_windows["target_coverage"].eq(coverage)]
    ax.scatter(subset["all_candidate_fraction_kept"], subset["reviewed_precision"], s=24, alpha=0.65, label=f"{coverage:.0%} target")
ax.set(xlabel="Fraction of all July 1 candidates inside one-feature window", ylabel="Reviewed microlensing precision inside window", title="One-feature deterministic microlensing windows")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "deterministic_feature_windows.pdf", dpi=180)
plt.show()


# ## Distributions Of The Most Reproducible Features

# In[ ]:


top_distribution_features = permutation_importance.loc[
    permutation_importance["permutation_ap_drop_mean"].gt(0), "feature"
].head(8).tolist()
if len(top_distribution_features) < 4:
    top_distribution_features = gain_importance.head(8)["feature"].tolist()

distribution_data = table[[TARGET_COLUMN, *top_distribution_features]].copy()
distribution_data["group"] = distribution_data[TARGET_COLUMN].map(
    {POSITIVE_LABEL: "reviewed microlensing-like", NEGATIVE_LABEL: "clear reviewed non-microlensing"}
).fillna("unlabeled/unreviewed")

fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
for ax, feature in zip(axes.ravel(), top_distribution_features):
    plot_data = distribution_data[["group", feature]].copy()
    plot_data[feature] = pd.to_numeric(plot_data[feature], errors="coerce")
    finite_values = plot_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_values.empty:
        lo, hi = finite_values.quantile([0.01, 0.99])
        plot_data[feature] = plot_data[feature].clip(lo, hi)
    sns.boxenplot(
        data=plot_data.loc[plot_data["group"].isin(["reviewed microlensing-like", "clear reviewed non-microlensing"])],
        x="group",
        hue="group",
        y=feature,
        order=["reviewed microlensing-like", "clear reviewed non-microlensing"],
        showfliers=False,
        ax=ax,
        palette={"reviewed microlensing-like": "#a44200", "clear reviewed non-microlensing": "#777777"},
        legend=False,
    )
    ax.set_xlabel("")
    ax.set_title(feature, fontsize=9)
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
for ax in axes.ravel()[len(top_distribution_features):]:
    ax.set_axis_off()
fig.suptitle("Top held-out microlensing feature distributions", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "top_feature_distributions.pdf", dpi=180, bbox_inches="tight")
plt.show()


# ## Interpretation Boundary
# 
# - Start review with `top500_unreviewed_by_microlensing_probability_with_lightcurve_paths.csv` and the corresponding light-curve pages.
# - Prefer features supported by both gain and permutation importance, but remember the held-out set contains very few positives.
# - Treat the model value as a queue-ranking score. A stats-only classifier does not establish a Paczyński profile, achromaticity, uniqueness, or physical lens parameters.
# - Use external-catalog agreement as an independent diagnostic. It was intentionally excluded from training, and catalog non-match is not evidence against a new event.
# - Symmetric stellar outbursts, CV/nova events, camera offsets, sparse sampling, and blends can resemble microlensing in summary statistics; inspect the raw colored points and follow up with the dedicated microlensing fitter.
# - Retrain after reviewing a meaningful tranche from both high- and low-score regions so contamination and recall can be estimated rather than only harvesting obvious positives.
