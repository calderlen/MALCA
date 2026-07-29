"""Generate diagnostics for the saved July 1 hierarchical Review model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from malca.meta_analysis.ml.july1_hierarchical_training import (
    DEFAULT_OUTPUT_DIR,
)
from malca.meta_analysis.ml.plotting import (
    FEATURE_IMPORTANCE_TOP_N,
    apply_ml_plot_style,
    plot_fraction_count_heatmap,
)
from malca.meta_analysis.ml.review_lightgbm import load_target_model


apply_ml_plot_style()


HEAD_ORDER = (
    "gate",
    "primary_morphology",
    "quasi_periodic",
    "microlensing_given_brightening",
    "long_timescale_subtype",
    "dipper_recurrence",
)
HEAD_DISPLAY = {
    "gate": "Rejection gate",
    "primary_morphology": "Primary morphology",
    "quasi_periodic": "Quasi-periodic head",
    "microlensing_given_brightening": "Microlensing | brightening",
    "long_timescale_subtype": "LPV vs LTV | long-timescale",
    "dipper_recurrence": "Recurrence | dipper",
}
CLASS_DISPLAY = {
    "artifact_or_nonvariable": "Reject",
    "usable_astrophysical_variable": "Usable variable",
    "brightening_transient": "Brightening",
    "dipper_dimming": "Dipper",
    "eb_geometric_periodic": "EB / geometric",
    "long_timescale_variable": "Long-timescale",
    "other_structured_variable": "Other structured",
    "not_quasi_periodic": "Not quasi-periodic",
    "quasi_periodic": "Quasi-periodic",
    "microlensing_like": "Microlensing-like",
    "not_microlensing_like": "Not microlensing-like",
    "long_period_variable": "LPV",
    "long_term_variable": "LTV",
    "non_recurrent_given_dipper": "Single dipper",
    "recurrent_given_dipper": "Recurrent",
}

def _plot_confusion(
    frame: pd.DataFrame,
    classes: list[str],
    *,
    title: str,
    output: Path,
) -> None:
    matrix = confusion_matrix(frame["y_true"], frame["y_pred"], labels=classes)
    display_classes = [CLASS_DISPLAY.get(label, label) for label in classes]
    width = max(6.5, 1.15 * len(classes) + 3.0)
    fig, ax = plt.subplots(figsize=(width, 5.2))
    plot_fraction_count_heatmap(
        pd.DataFrame(matrix, index=display_classes, columns=display_classes),
        cmap="Blues",
        ax=ax,
    )
    ax.set(
        title=title,
        xlabel="Predicted",
        ylabel="Human label",
    )
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_ovr_curves(
    frame: pd.DataFrame,
    classes: list[str],
    probability_columns: list[str],
    *,
    title: str,
    output: Path,
) -> list[dict[str, object]]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rows: list[dict[str, object]] = []
    palette = sns.color_palette("tab10", n_colors=len(classes))
    for index, (label, column) in enumerate(
        zip(classes, probability_columns)
    ):
        display_label = CLASS_DISPLAY.get(label, label)
        binary_true = frame["y_true"].eq(label).astype(int)
        score = pd.to_numeric(frame[column], errors="coerce")
        if binary_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_true, score)
        precision, recall, _ = precision_recall_curve(binary_true, score)
        auc_value = float(roc_auc_score(binary_true, score))
        ap_value = float(average_precision_score(binary_true, score))
        axes[0].plot(
            fpr,
            tpr,
            color=palette[index],
            lw=2,
            label=f"{display_label} AUC={auc_value:.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=palette[index],
            lw=2,
            label=f"{display_label} AP={ap_value:.3f}",
        )
        rows.append(
            {
                "class": label,
                "support": int(binary_true.sum()),
                "roc_auc_ovr": auc_value,
                "average_precision_ovr": ap_value,
            }
        )
    axes[0].plot([0, 1], [0, 1], "--", color="0.5")
    axes[0].set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="One-vs-rest ROC",
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="One-vs-rest precision-recall",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return rows


def _plot_importance(head_dir: Path, *, title: str, output: Path) -> None:
    path = head_dir / "feature_importance_gain.csv"
    importance = (
        pd.read_csv(path).head(FEATURE_IMPORTANCE_TOP_N).sort_values("gain")
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(importance["feature"], importance["gain"], color="#2a788e")
    ax.set(xlabel="LightGBM gain", title=title)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    model_dir = DEFAULT_OUTPUT_DIR.resolve()
    figures = model_dir / "figures"
    reports = figures / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for stale in (
        *figures.glob("*.png"),
        *figures.glob("*.pdf"),
        *reports.glob("*.csv"),
    ):
        stale.unlink()

    performance_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    for head in HEAD_ORDER:
        head_dir = model_dir / head
        bundle = load_target_model(head_dir)
        metadata = json.loads((head_dir / "metadata.json").read_text())
        predictions = pd.read_parquet(head_dir / "test_predictions.parquet")
        classes = list(bundle["label_classes"])
        probability_columns = list(bundle["probability_columns"])
        display = HEAD_DISPLAY[head]

        _plot_confusion(
            predictions,
            classes,
            title=display,
            output=figures / f"{head}_heldout_confusion.pdf",
        )
        rows = _plot_ovr_curves(
            predictions,
            classes,
            probability_columns,
            title=display,
            output=figures / f"{head}_heldout_ovr_curves.pdf",
        )
        for row in rows:
            row["head"] = head
            curve_rows.append(row)
        _plot_importance(
            head_dir,
            title=f"{display}: top {FEATURE_IMPORTANCE_TOP_N} gain features",
            output=figures / f"{head}_feature_importance.pdf",
        )

        y_true = predictions["y_true"].astype(str)
        y_pred = predictions["y_pred"].astype(str)
        performance_rows.append(
            {
                "head": head,
                "display": display,
                "n_trainable": int(metadata["n_rows"]),
                "n_holdout": int(len(predictions)),
                "n_features": int(metadata["n_features"]),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
            }
        )
        cv_path = head_dir / "cv_metrics.csv"
        if cv_path.exists():
            cv = pd.read_csv(cv_path)
            cv_rows.append(
                {
                    "head": head,
                    "balanced_accuracy_mean": float(
                        cv["balanced_accuracy"].mean()
                    ),
                    "balanced_accuracy_std": float(
                        cv["balanced_accuracy"].std(ddof=1)
                    ),
                    "macro_f1_mean": float(cv["macro_f1"].mean()),
                    "macro_f1_std": float(cv["macro_f1"].std(ddof=1)),
                }
            )

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(reports / "heldout_head_performance.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        reports / "heldout_class_curves.csv", index=False
    )
    pd.DataFrame(cv_rows).to_csv(
        reports / "cross_validation_summary.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(performance))
    axes[0].bar(
        x - 0.18,
        performance["balanced_accuracy"],
        width=0.36,
        label="Balanced accuracy",
    )
    axes[0].bar(
        x + 0.18,
        performance["macro_f1"],
        width=0.36,
        label="Macro F1",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(performance["display"], rotation=35, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Held-out performance by hierarchy head")
    axes[0].legend()
    axes[1].barh(
        performance["display"],
        performance["n_trainable"],
        color="#7ad151",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Trainable human labels (log scale)")
    axes[1].set_title("Training support")
    fig.tight_layout()
    fig.savefig(figures / "hierarchy_head_summary.pdf")
    plt.close(fig)

    scores = pd.read_parquet(
        model_dir / "all_candidates_hierarchical_scores.parquet"
    )
    unreviewed = scores.loc[scores["is_human_unreviewed"]].copy()
    gate_counts = unreviewed["predicted_hierarchy_gate"].value_counts()
    primary_counts = unreviewed[
        "predicted_primary_morphology"
    ].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(gate_counts.index, gate_counts.values, color="#440154")
    axes[0].set_title("Unreviewed gate predictions")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(primary_counts.index, primary_counts.values, color="#21918c")
    axes[1].set_title("Unreviewed conditional primary predictions")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figures / "unreviewed_hierarchy_summary.pdf")
    plt.close(fig)

    print(f"Model directory: {model_dir}")
    print(f"Diagnostic figures: {figures}")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
