"""Plot diagnostics for the focused four-class -> recurrence hierarchy."""

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

from malca.meta_analysis.ml.plotting import (
    apply_ml_plot_style,
    plot_fraction_count_heatmap,
)


DEFAULT_MODEL_DIR = Path(
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "lightgbm_retrain_20260729T201841Z/"
    "four_class_dimming_hierarchy"
)
HEADS = {
    "parent_four_class": "Four-class parent",
    "dimming_recurrence": "Recurrence given dimming",
}
DISPLAY = {
    "dimming_event": "Dimming",
    "eclipsing_binary": "EB",
    "junk": "Junk",
    "other": "Other",
    "non_recurrent_given_dipper": "Non-recurrent",
    "recurrent_given_dipper": "Recurrent",
    "non_recurrent_dimming_event": "Non-recurrent dimming",
    "recurrent_dimming_event": "Recurrent dimming",
}
TOP_FEATURES = 20
PNG_DPI = 180


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required hierarchy artifact is missing: {path}")
    return path


def _save(
    fig: plt.Figure, output_dir: Path, stem: str
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=PNG_DPI)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _labels(head_dir: Path) -> list[str]:
    metadata = json.loads(
        _require(head_dir / "metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata.get("label_classes")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Missing label_classes in {head_dir / 'metadata.json'}")
    return [str(label) for label in labels]


def _plot_confusion(
    head_dir: Path,
    labels: list[str],
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    frame = pd.read_csv(_require(head_dir / "confusion_matrix.csv"))
    counts = (
        frame.set_index("y_true")
        .reindex(index=labels, columns=labels, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    counts.index = [DISPLAY.get(label, label) for label in labels]
    counts.columns = [DISPLAY.get(label, label) for label in labels]
    size = max(6.2, 1.25 * len(labels) + 2.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    plot_fraction_count_heatmap(counts, ax=ax, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Held-out human class")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=35, ha="right", fontsize=9
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    return _save(fig, output_dir, stem)


def _plot_importance(
    head_dir: Path,
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    frame = pd.read_csv(_require(head_dir / "feature_importance_gain.csv"))
    frame["gain"] = pd.to_numeric(frame["gain"], errors="coerce").fillna(0)
    frame = frame.nlargest(TOP_FEATURES, "gain").sort_values("gain")
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    ax.barh(
        frame["feature"].astype(str).str.replace("_", " "),
        frame["gain"],
        color="#3a78a1",
    )
    ax.set_title(title)
    ax.set_xlabel("LightGBM gain importance")
    ax.grid(axis="x", alpha=0.27)
    ax.tick_params(axis="y", labelsize=8.5)
    return _save(fig, output_dir, stem)


def _plot_reliability(
    head_dir: Path,
    labels: list[str],
    *,
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    frame = pd.read_csv(_require(head_dir / "calibration_by_bin.csv"))
    columns = min(4, len(labels))
    rows = int(np.ceil(len(labels) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.15 * columns, 3.7 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, label in zip(axes.flat, labels):
        subset = frame.loc[frame["class_label"].astype(str).eq(label)].copy()
        for column in ("mean_probability", "observed_rate", "n"):
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna(
            subset=["mean_probability", "observed_rate", "n"]
        )
        ax.plot((0, 1), (0, 1), "--", color="0.5", linewidth=1)
        ax.plot(
            subset["mean_probability"],
            subset["observed_rate"],
            color="#3a78a1",
            linewidth=1.2,
        )
        ax.scatter(
            subset["mean_probability"],
            subset["observed_rate"],
            s=np.clip(9 * np.sqrt(subset["n"]), 18, 115),
            color="#3a78a1",
            edgecolor="white",
            linewidth=0.45,
        )
        ax.set_title(DISPLAY.get(label, label), fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.23)
    for ax in axes.flat[len(labels) :]:
        ax.remove()
    fig.supxlabel("Mean held-out score")
    fig.supylabel("Observed class fraction")
    fig.suptitle(title, y=1.02)
    return _save(fig, output_dir, stem)


def _plot_leaf_counts(
    scores: pd.DataFrame, *, output_dir: Path
) -> list[Path]:
    unreviewed = scores["is_human_unreviewed"].fillna(False).astype(bool)
    order = [
        "recurrent_dimming_event",
        "non_recurrent_dimming_event",
        "eclipsing_binary",
        "junk",
        "other",
    ]
    counts = (
        scores.loc[unreviewed, "predicted_hierarchical_leaf"]
        .astype(str)
        .value_counts()
        .reindex(order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    bars = ax.bar(
        [DISPLAY.get(label, label) for label in counts.index],
        counts.to_numpy(dtype=int),
        color="#9b663a",
    )
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylabel("Human-unreviewed candidates")
    ax.set_title("Hierarchical leaf predictions for the unreviewed queue")
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax.grid(axis="y", alpha=0.27)
    return _save(
        fig, output_dir, "hierarchy_unreviewed_leaf_prediction_counts"
    )


def _plot_branch_structure(
    scores: pd.DataFrame, *, output_dir: Path
) -> list[Path]:
    unreviewed = scores["is_human_unreviewed"].fillna(False).astype(bool)
    frame = scores.loc[
        unreviewed,
        [
            "prob_dimming_event",
            "prob_recurrent_given_dimming",
            "prob_recurrent_dimming_event",
            "hierarchical_leaf_entropy",
        ],
    ].apply(pd.to_numeric, errors="coerce").dropna()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.1),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    scatter = axes[0].scatter(
        frame["prob_dimming_event"],
        frame["prob_recurrent_given_dimming"],
        c=frame["prob_recurrent_dimming_event"],
        cmap="viridis",
        s=10,
        alpha=0.5,
        linewidth=0,
        rasterized=True,
    )
    axes[0].set_xlabel("Parent dimming ranking score")
    axes[0].set_ylabel("Conditional recurrent | dimming score")
    axes[0].set_title("Two-stage dimming branch")
    axes[0].grid(alpha=0.23)
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("Composed recurrent-dimming score")
    axes[1].hist(
        frame["hierarchical_leaf_entropy"],
        bins=40,
        color="#8565a3",
        alpha=0.82,
    )
    axes[1].set_xlabel("Normalised five-leaf entropy")
    axes[1].set_ylabel("Human-unreviewed candidates")
    axes[1].set_title("Composed leaf uncertainty")
    axes[1].grid(axis="y", alpha=0.23)
    return _save(fig, output_dir, "hierarchy_unreviewed_branch_structure")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "figures"
    )

    apply_ml_plot_style()
    generated: list[Path] = []
    for key, display in HEADS.items():
        head_dir = model_dir / key
        labels = _labels(head_dir)
        generated += _plot_confusion(
            head_dir,
            labels,
            title=f"{display}: held-out confusion matrix",
            output_dir=output_dir,
            stem=f"{key}_holdout_confusion_matrix",
        )
        generated += _plot_importance(
            head_dir,
            title=f"{display}: top {TOP_FEATURES} gain features",
            output_dir=output_dir,
            stem=f"{key}_feature_importance_gain",
        )
        generated += _plot_reliability(
            head_dir,
            labels,
            title=f"{display}: held-out reliability (not calibrated)",
            output_dir=output_dir,
            stem=f"{key}_heldout_reliability",
        )

    scores = pd.read_parquet(
        _require(
            model_dir
            / "all_candidates_four_class_hierarchical_scores.parquet"
        )
    )
    generated += _plot_leaf_counts(scores, output_dir=output_dir)
    generated += _plot_branch_structure(scores, output_dir=output_dir)
    manifest = {
        "model_dir": str(model_dir),
        "note": (
            "All probability-like values are class-balanced uncalibrated "
            "ranking scores."
        ),
        "generated_files": [
            str(path.relative_to(output_dir)) for path in generated
        ],
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} plot files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
