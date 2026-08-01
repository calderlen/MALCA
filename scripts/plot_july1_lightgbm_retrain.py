"""Render diagnostics for a timestamped July 1 LightGBM retraining run.

The training scripts deliberately write compact tables and model artifacts, not
figures.  This companion script reads those immutable artifacts and writes a
small, self-contained diagnostic set under ``<retrain-dir>/figures``.  It
never retrains a model or changes the Review database.

Example
-------
conda run -n malca python scripts/plot_july1_lightgbm_retrain.py \\
    --retrain-dir output/runs/dat3-full-extended_2026-07-01-v4/results/ \\
        lightgbm_retrain_20260729T201841Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.meta_analysis.ml.plotting import apply_ml_plot_style, plot_fraction_count_heatmap


DEFAULT_RETRAIN_DIR = Path(
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "lightgbm_retrain_20260729T201841Z"
)
BINARY_MODEL_DIR = Path("dipper_binary/stats_plus_periodicity_dip_jump")
EIGHT_CLASS_MODEL_DIR = Path("eight_class")
PHYSICAL_MORPHOLOGY_MODEL_DIR = Path("physical_morphology")
FIVE_CLASS_MODEL_DIR = Path("five_class")
DIMMING_HIERARCHY_MODEL_DIR = Path("dimming_hierarchy")
PNG_DPI = 180
TOP_FEATURES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain-dir",
        type=Path,
        default=DEFAULT_RETRAIN_DIR,
        help="Timestamped lightgbm_retrain_* directory to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for figures; defaults to <retrain-dir>/figures.",
    )
    return parser.parse_args()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required retraining artifact is missing: {path}")
    return path


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=PNG_DPI)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _model_labels(model_dir: Path) -> list[str]:
    metadata = json.loads(_require(model_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = metadata.get("label_classes")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"metadata.json has no usable label_classes: {model_dir}")
    return [str(label) for label in labels]


def _confusion_counts(model_dir: Path, labels: Iterable[str]) -> pd.DataFrame:
    labels = list(labels)
    frame = pd.read_csv(_require(model_dir / "confusion_matrix.csv"))
    if "y_true" not in frame.columns:
        raise ValueError(f"Confusion matrix has no y_true column: {model_dir}")
    counts = frame.set_index("y_true")
    missing = sorted(set(labels).difference(counts.columns))
    if missing:
        raise ValueError(f"Confusion matrix is missing predicted labels {missing}: {model_dir}")
    return counts.reindex(index=labels, columns=labels, fill_value=0).apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)


def plot_confusion_matrix(
    model_dir: Path,
    labels: list[str],
    *,
    title: str,
    output_dir: Path,
    stem: str,
    y_label: str = "Held-out human class",
) -> list[Path]:
    counts = _confusion_counts(model_dir, labels)
    size = max(6.4, 0.97 * len(labels) + 3.2)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    plot_fraction_count_heatmap(
        counts,
        ax=ax,
        cmap="Blues",
        fraction_fontsize=8 if len(labels) > 4 else 10,
        count_fontsize=6.5 if len(labels) > 4 else 8,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel(y_label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    return _save(fig, output_dir, stem)


def _feature_name(value: object) -> str:
    return str(value).replace("_", " ")


def plot_gain_importance(
    model_dir: Path,
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    frame = pd.read_csv(_require(model_dir / "feature_importance_gain.csv"))
    required = {"feature", "gain"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Gain importance is missing {missing}: {model_dir}")
    plot_frame = frame.loc[:, ["feature", "gain"]].copy()
    plot_frame["gain"] = pd.to_numeric(plot_frame["gain"], errors="coerce").fillna(0.0)
    plot_frame = plot_frame.nlargest(TOP_FEATURES, "gain").sort_values("gain")
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.barh(
        [_feature_name(value) for value in plot_frame["feature"]],
        plot_frame["gain"],
        color="#3a78a1",
        edgecolor="none",
    )
    ax.set_title(title)
    ax.set_xlabel("LightGBM gain importance")
    ax.grid(axis="x", alpha=0.28)
    ax.tick_params(axis="y", labelsize=8.5)
    return _save(fig, output_dir, stem)


def _unreviewed_mask(frame: pd.DataFrame) -> pd.Series:
    workflow = frame.get("workflow_status", pd.Series("", index=frame.index))
    event_class = frame.get("event_class", pd.Series("", index=frame.index))
    workflow = workflow.fillna("").astype(str).str.strip()
    event_class = event_class.fillna("").astype(str).str.strip()
    return workflow.isin(("", "unreviewed")) & event_class.isin(("", "unclassified"))


def plot_binary_score_distribution(
    model_dir: Path,
    *,
    output_dir: Path,
) -> list[Path]:
    scores = pd.read_parquet(_require(model_dir / "all_candidates_scores.parquet"))
    probability = pd.to_numeric(scores["prob_dipper_like"], errors="coerce")
    unreviewed = _unreviewed_mask(scores)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(
        probability.loc[~unreviewed].dropna(),
        bins=bins,
        histtype="stepfilled",
        alpha=0.48,
        color="#9a7d0a",
        label=f"reviewed or classified (n={(~unreviewed).sum():,})",
    )
    ax.hist(
        probability.loc[unreviewed].dropna(),
        bins=bins,
        histtype="step",
        linewidth=1.9,
        color="#176d9c",
        label=f"human-unreviewed (n={unreviewed.sum():,})",
    )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Binary dipper ranking score, prob_dipper_like")
    ax.set_ylabel("Candidate count (log scale)")
    ax.set_title("Binary dipper score distribution across all candidates")
    ax.legend(frameon=True, fontsize=9)
    return _save(fig, output_dir, "binary_all_candidate_score_distribution")


def plot_calibration(
    model_dir: Path,
    labels: list[str],
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    frame = pd.read_csv(_require(model_dir / "calibration_by_bin.csv"))
    required = {"class_label", "mean_probability", "observed_rate", "n"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Calibration diagnostics are missing {missing}: {model_dir}")
    columns = min(4, len(labels))
    rows = int(np.ceil(len(labels) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.05 * columns, 3.65 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, label in zip(axes.flat, labels):
        class_frame = frame.loc[frame["class_label"].astype(str).eq(label)].copy()
        class_frame["mean_probability"] = pd.to_numeric(
            class_frame["mean_probability"], errors="coerce"
        )
        class_frame["observed_rate"] = pd.to_numeric(
            class_frame["observed_rate"], errors="coerce"
        )
        class_frame["n"] = pd.to_numeric(class_frame["n"], errors="coerce").fillna(0.0)
        class_frame = class_frame.dropna(subset=["mean_probability", "observed_rate"])
        ax.plot((0, 1), (0, 1), color="0.45", linestyle="--", linewidth=1)
        ax.plot(
            class_frame["mean_probability"],
            class_frame["observed_rate"],
            color="#3a78a1",
            linewidth=1.2,
        )
        ax.scatter(
            class_frame["mean_probability"],
            class_frame["observed_rate"],
            s=np.clip(9 * np.sqrt(class_frame["n"].to_numpy(dtype=float)), 18, 115),
            color="#3a78a1",
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.set_title(_feature_name(label), fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(labels) :]:
        ax.remove()
    fig.supxlabel("Mean held-out score")
    fig.supylabel("Observed class fraction")
    fig.suptitle(title, y=1.02)
    return _save(fig, output_dir, stem)


def plot_eight_class_prediction_counts(
    model_dir: Path,
    labels: list[str],
    *,
    output_dir: Path,
) -> list[Path]:
    scores = pd.read_parquet(_require(model_dir / "all_candidates_eight_class_scores.parquet"))
    unreviewed = scores.get("is_human_unreviewed", _unreviewed_mask(scores)).fillna(False).astype(bool)
    counts = scores.loc[unreviewed, "y_pred"].astype(str).value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    bars = ax.bar(
        [_feature_name(label) for label in counts.index],
        counts.to_numpy(dtype=int),
        color="#3a78a1",
        edgecolor="none",
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylabel("Human-unreviewed candidates")
    ax.set_title("Eight-class predictions for the human-unreviewed queue")
    ax.tick_params(axis="x", rotation=38, labelsize=8.5)
    ax.grid(axis="y", alpha=0.28)
    return _save(fig, output_dir, "eight_class_unreviewed_prediction_counts")


def plot_eight_class_uncertainty(model_dir: Path, *, output_dir: Path) -> list[Path]:
    scores = pd.read_parquet(_require(model_dir / "all_candidates_eight_class_scores.parquet"))
    unreviewed = scores.get("is_human_unreviewed", _unreviewed_mask(scores)).fillna(False).astype(bool)
    frame = scores.loc[unreviewed, ["prediction_entropy", "score_margin", "prediction_confidence"]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": (1.35, 1)})
    scatter = axes[0].scatter(
        frame["score_margin"],
        frame["prediction_entropy"],
        c=frame["prediction_confidence"],
        cmap="viridis",
        s=10,
        alpha=0.52,
        linewidth=0,
        rasterized=True,
    )
    axes[0].set_xlabel("Top-two score margin")
    axes[0].set_ylabel("Normalised prediction entropy")
    axes[0].set_title("Uncertainty structure of unreviewed candidates")
    axes[0].grid(alpha=0.24)
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Top-class ranking score")
    axes[1].hist(frame["prediction_entropy"], bins=40, color="#8565a3", alpha=0.82)
    axes[1].set_xlabel("Normalised prediction entropy")
    axes[1].set_ylabel("Human-unreviewed candidates")
    axes[1].set_title("Entropy distribution")
    axes[1].grid(axis="y", alpha=0.24)
    return _save(fig, output_dir, "eight_class_unreviewed_uncertainty")


def plot_physical_morphology_prediction_counts(
    model_dir: Path,
    labels: list[str],
    *,
    output_dir: Path,
) -> list[Path]:
    scores = pd.read_parquet(
        _require(model_dir / "all_candidates_physical_morphology_scores.parquet")
    )
    unreviewed = scores.get(
        "is_human_unreviewed", _unreviewed_mask(scores)
    ).fillna(False).astype(bool)
    counts = (
        scores.loc[unreviewed, "y_pred"]
        .astype(str)
        .value_counts()
        .reindex(labels, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    bars = ax.bar(
        [_feature_name(label) for label in counts.index],
        counts.to_numpy(dtype=int),
        color="#4b8b57",
        edgecolor="none",
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylabel("Human-unreviewed candidates")
    ax.set_title(
        "Morphology-derived physical model predictions for the unreviewed queue"
    )
    ax.tick_params(axis="x", rotation=42, labelsize=8)
    ax.grid(axis="y", alpha=0.28)
    return _save(
        fig, output_dir, "physical_morphology_unreviewed_prediction_counts"
    )


def plot_five_class_prediction_counts(
    model_dir: Path,
    labels: list[str],
    *,
    output_dir: Path,
) -> list[Path]:
    scores = pd.read_parquet(
        _require(model_dir / "all_candidates_five_class_scores.parquet")
    )
    unreviewed = scores.get(
        "is_human_unreviewed", _unreviewed_mask(scores)
    ).fillna(False).astype(bool)
    counts = (
        scores.loc[unreviewed, "y_pred"]
        .astype(str)
        .value_counts()
        .reindex(labels, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    bars = ax.bar(
        [_feature_name(label) for label in counts.index],
        counts.to_numpy(dtype=int),
        color="#a15c38",
        edgecolor="none",
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylabel("Human-unreviewed candidates")
    ax.set_title("Five-class morphology predictions for the unreviewed queue")
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.grid(axis="y", alpha=0.28)
    return _save(
        fig, output_dir, "five_class_unreviewed_prediction_counts"
    )


def plot_five_class_uncertainty(
    model_dir: Path, *, output_dir: Path
) -> list[Path]:
    scores = pd.read_parquet(
        _require(model_dir / "all_candidates_five_class_scores.parquet")
    )
    unreviewed = scores.get(
        "is_human_unreviewed", _unreviewed_mask(scores)
    ).fillna(False).astype(bool)
    frame = scores.loc[
        unreviewed,
        ["prediction_entropy", "score_margin", "prediction_confidence"],
    ].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.0),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    scatter = axes[0].scatter(
        frame["score_margin"],
        frame["prediction_entropy"],
        c=frame["prediction_confidence"],
        cmap="viridis",
        s=10,
        alpha=0.52,
        linewidth=0,
        rasterized=True,
    )
    axes[0].set_xlabel("Top-two score margin")
    axes[0].set_ylabel("Normalised prediction entropy")
    axes[0].set_title("Five-class uncertainty for unreviewed candidates")
    axes[0].grid(alpha=0.24)
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Top-class ranking score")
    axes[1].hist(
        frame["prediction_entropy"],
        bins=40,
        color="#a15c38",
        alpha=0.82,
    )
    axes[1].set_xlabel("Normalised prediction entropy")
    axes[1].set_ylabel("Human-unreviewed candidates")
    axes[1].set_title("Five-class entropy distribution")
    axes[1].grid(axis="y", alpha=0.24)
    return _save(fig, output_dir, "five_class_unreviewed_uncertainty")


def plot_dimming_hierarchy_prediction_counts(
    model_dir: Path,
    *,
    output_dir: Path,
) -> list[Path]:
    scores = pd.read_parquet(
        _require(
            model_dir / "all_candidates_dimming_hierarchy_scores.parquet"
        )
    )
    unreviewed = scores.get(
        "is_human_unreviewed", _unreviewed_mask(scores)
    ).fillna(False).astype(bool)
    parent_order = ("dimming_event", "eclipsing_binary", "junk", "other")
    leaf_order = (
        "recurrent_dimming_event",
        "non_recurrent_dimming_event",
        "eclipsing_binary",
        "junk",
        "other",
    )
    parent_counts = (
        scores.loc[unreviewed, "predicted_parent_class"]
        .astype(str)
        .value_counts()
        .reindex(parent_order, fill_value=0)
    )
    leaf_counts = (
        scores.loc[unreviewed, "predicted_hierarchical_class"]
        .astype(str)
        .value_counts()
        .reindex(leaf_order, fill_value=0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.0))
    for ax, counts, title, color in (
        (
            axes[0],
            parent_counts,
            "Stage 1: four-class parent predictions",
            "#4f759b",
        ),
        (
            axes[1],
            leaf_counts,
            "Stage 2: dimming branch split by recurrence head",
            "#b0653f",
        ),
    ):
        bars = ax.bar(
            [_feature_name(label) for label in counts.index],
            counts.to_numpy(dtype=int),
            color=color,
            edgecolor="none",
        )
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Human-unreviewed candidates")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.28)
    return _save(
        fig, output_dir, "dimming_hierarchy_unreviewed_prediction_counts"
    )


def plot_dimming_hierarchy_score_structure(
    model_dir: Path,
    *,
    output_dir: Path,
) -> list[Path]:
    scores = pd.read_parquet(
        _require(
            model_dir / "all_candidates_dimming_hierarchy_scores.parquet"
        )
    )
    unreviewed = scores.get(
        "is_human_unreviewed", _unreviewed_mask(scores)
    ).fillna(False).astype(bool)
    frame = scores.loc[
        unreviewed,
        [
            "score_parent_dimming_event",
            "score_recurrent_given_dimming",
            "score_hierarchical_recurrent_dimming_event",
        ],
    ].apply(pd.to_numeric, errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    scatter = axes[0].scatter(
        frame["score_parent_dimming_event"],
        frame["score_recurrent_given_dimming"],
        c=frame["score_hierarchical_recurrent_dimming_event"],
        cmap="viridis",
        s=10,
        alpha=0.52,
        linewidth=0,
        rasterized=True,
    )
    axes[0].set_xlabel("Stage-1 dimming ranking score")
    axes[0].set_ylabel("Stage-2 recurrent | dimming ranking score")
    axes[0].set_title("Parent and conditional-head score structure")
    axes[0].grid(alpha=0.24)
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Gated recurrent-dimming score")
    axes[1].hist(
        frame["score_hierarchical_recurrent_dimming_event"],
        bins=45,
        color="#b0653f",
        alpha=0.82,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Gated recurrent-dimming ranking score")
    axes[1].set_ylabel("Human-unreviewed candidates (log scale)")
    axes[1].set_title("Hierarchical recurrent-dimming score")
    axes[1].grid(axis="y", alpha=0.24)
    return _save(
        fig, output_dir, "dimming_hierarchy_score_structure"
    )


def main() -> int:
    args = parse_args()
    retrain_dir = args.retrain_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else retrain_dir / "figures"
    )
    binary_dir = retrain_dir / BINARY_MODEL_DIR
    eight_dir = retrain_dir / EIGHT_CLASS_MODEL_DIR
    physical_dir = retrain_dir / PHYSICAL_MORPHOLOGY_MODEL_DIR
    five_class_dir = retrain_dir / FIVE_CLASS_MODEL_DIR
    dimming_hierarchy_dir = retrain_dir / DIMMING_HIERARCHY_MODEL_DIR
    binary_labels = _model_labels(binary_dir)
    eight_labels = _model_labels(eight_dir)

    apply_ml_plot_style()
    generated: list[Path] = []
    generated += plot_confusion_matrix(
        binary_dir,
        binary_labels,
        title="Binary dipper model: held-out confusion matrix",
        output_dir=output_dir,
        stem="binary_holdout_confusion_matrix",
    )
    generated += plot_gain_importance(
        binary_dir,
        title=f"Binary dipper model: top {TOP_FEATURES} LightGBM gain features",
        output_dir=output_dir,
        stem="binary_feature_importance_gain",
    )
    generated += plot_binary_score_distribution(binary_dir, output_dir=output_dir)
    generated += plot_calibration(
        binary_dir,
        binary_labels,
        title="Binary dipper model: held-out reliability diagnostic (not calibrated)",
        output_dir=output_dir,
        stem="binary_heldout_reliability",
    )
    generated += plot_confusion_matrix(
        eight_dir,
        eight_labels,
        title="Eight-class model: held-out confusion matrix",
        output_dir=output_dir,
        stem="eight_class_holdout_confusion_matrix",
    )
    generated += plot_gain_importance(
        eight_dir,
        title=f"Eight-class model: top {TOP_FEATURES} LightGBM gain features",
        output_dir=output_dir,
        stem="eight_class_feature_importance_gain",
    )
    generated += plot_eight_class_prediction_counts(eight_dir, eight_labels, output_dir=output_dir)
    generated += plot_eight_class_uncertainty(eight_dir, output_dir=output_dir)
    generated += plot_calibration(
        eight_dir,
        eight_labels,
        title="Eight-class model: held-out reliability diagnostics (not calibrated)",
        output_dir=output_dir,
        stem="eight_class_heldout_reliability",
    )
    if physical_dir.is_dir():
        physical_labels = _model_labels(physical_dir)
        generated += plot_confusion_matrix(
            physical_dir,
            physical_labels,
            title=(
                "Morphology-derived physical model: held-out confusion matrix"
            ),
            output_dir=output_dir,
            stem="physical_morphology_holdout_confusion_matrix",
            y_label="Held-out constructed class",
        )
        generated += plot_gain_importance(
            physical_dir,
            title=(
                f"Morphology-derived physical model: top {TOP_FEATURES} "
                "LightGBM gain features"
            ),
            output_dir=output_dir,
            stem="physical_morphology_feature_importance_gain",
        )
        generated += plot_physical_morphology_prediction_counts(
            physical_dir, physical_labels, output_dir=output_dir
        )
        generated += plot_calibration(
            physical_dir,
            physical_labels,
            title=(
                "Morphology-derived physical model: held-out reliability "
                "diagnostics (not calibrated)"
            ),
            output_dir=output_dir,
            stem="physical_morphology_heldout_reliability",
        )
    if five_class_dir.is_dir():
        five_class_labels = _model_labels(five_class_dir)
        generated += plot_confusion_matrix(
            five_class_dir,
            five_class_labels,
            title="Five-class morphology model: held-out confusion matrix",
            output_dir=output_dir,
            stem="five_class_holdout_confusion_matrix",
        )
        generated += plot_gain_importance(
            five_class_dir,
            title=(
                f"Five-class morphology model: top {TOP_FEATURES} "
                "LightGBM gain features"
            ),
            output_dir=output_dir,
            stem="five_class_feature_importance_gain",
        )
        generated += plot_five_class_prediction_counts(
            five_class_dir, five_class_labels, output_dir=output_dir
        )
        generated += plot_five_class_uncertainty(
            five_class_dir, output_dir=output_dir
        )
        generated += plot_calibration(
            five_class_dir,
            five_class_labels,
            title=(
                "Five-class morphology model: held-out reliability "
                "diagnostics (not calibrated)"
            ),
            output_dir=output_dir,
            stem="five_class_heldout_reliability",
        )
    if dimming_hierarchy_dir.is_dir():
        parent_dir = dimming_hierarchy_dir / "parent_four_class"
        recurrence_dir = dimming_hierarchy_dir / "recurrence_head"
        parent_labels = _model_labels(parent_dir)
        recurrence_labels = _model_labels(recurrence_dir)
        generated += plot_confusion_matrix(
            parent_dir,
            parent_labels,
            title="Four-class parent model: held-out confusion matrix",
            output_dir=output_dir,
            stem="four_class_parent_holdout_confusion_matrix",
        )
        generated += plot_gain_importance(
            parent_dir,
            title=(
                f"Four-class parent model: top {TOP_FEATURES} "
                "LightGBM gain features"
            ),
            output_dir=output_dir,
            stem="four_class_parent_feature_importance_gain",
        )
        generated += plot_calibration(
            parent_dir,
            parent_labels,
            title=(
                "Four-class parent model: held-out reliability "
                "diagnostics (not calibrated)"
            ),
            output_dir=output_dir,
            stem="four_class_parent_heldout_reliability",
        )
        generated += plot_confusion_matrix(
            recurrence_dir,
            recurrence_labels,
            title=(
                "Conditional recurrence head: held-out confusion matrix"
            ),
            output_dir=output_dir,
            stem="recurrence_head_holdout_confusion_matrix",
            y_label="Held-out human dimming subclass",
        )
        generated += plot_gain_importance(
            recurrence_dir,
            title=(
                f"Conditional recurrence head: top {TOP_FEATURES} "
                "LightGBM gain features"
            ),
            output_dir=output_dir,
            stem="recurrence_head_feature_importance_gain",
        )
        generated += plot_calibration(
            recurrence_dir,
            recurrence_labels,
            title=(
                "Conditional recurrence head: held-out reliability "
                "diagnostics (not calibrated)"
            ),
            output_dir=output_dir,
            stem="recurrence_head_heldout_reliability",
        )
        generated += plot_dimming_hierarchy_prediction_counts(
            dimming_hierarchy_dir, output_dir=output_dir
        )
        generated += plot_dimming_hierarchy_score_structure(
            dimming_hierarchy_dir, output_dir=output_dir
        )

    manifest = {
        "retrain_dir": str(retrain_dir),
        "binary_model_dir": str(binary_dir),
        "eight_class_model_dir": str(eight_dir),
        "physical_morphology_model_dir": (
            str(physical_dir) if physical_dir.is_dir() else None
        ),
        "five_class_model_dir": (
            str(five_class_dir) if five_class_dir.is_dir() else None
        ),
        "dimming_hierarchy_model_dir": (
            str(dimming_hierarchy_dir)
            if dimming_hierarchy_dir.is_dir()
            else None
        ),
        "note": "All probability-like values are uncalibrated class-balanced ranking scores.",
        "generated_files": [str(path.relative_to(output_dir)) for path in generated],
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} plot files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
