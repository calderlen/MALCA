"""Plot the tuned/calibrated July 1 parent -> recurrence hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from malca.meta_analysis.ml.plotting import plot_fraction_count_heatmap
from malca.plotting.lightcurve_publication import FIG_SINGLE_COL_SQUARE
try:
    from scripts.plot_july1_four_class_hierarchical_ml import (
        BINARY_CONFUSION_COUNT_FONTSIZE,
        BINARY_CONFUSION_FRACTION_FONTSIZE,
        CONFUSION_AXIS_FONTSIZE,
        CONFUSION_COUNT_FONTSIZE,
        CONFUSION_FRACTION_FONTSIZE,
        CONFUSION_TICK_FONTSIZE,
        DISPLAY,
        HEADS,
        _apply_plot_style,
        _labels,
        _plot_feature_distributions,
        _plot_importance,
        _require,
        _save,
    )
except ModuleNotFoundError:
    from plot_july1_four_class_hierarchical_ml import (
        BINARY_CONFUSION_COUNT_FONTSIZE,
        BINARY_CONFUSION_FRACTION_FONTSIZE,
        CONFUSION_AXIS_FONTSIZE,
        CONFUSION_COUNT_FONTSIZE,
        CONFUSION_FRACTION_FONTSIZE,
        CONFUSION_TICK_FONTSIZE,
        DISPLAY,
        HEADS,
        _apply_plot_style,
        _labels,
        _plot_feature_distributions,
        _plot_importance,
        _require,
        _save,
    )


DEFAULT_MODEL_DIR = Path(
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "lightgbm_probability_tuned_20260802T232447Z"
)


def _plot_calibrated_confusion(
    head_dir: Path,
    labels: list[str],
    *,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """Plot the locked-test confusion matrix from calibrated predictions."""

    predictions = pd.read_parquet(
        _require(head_dir / "test_predictions.parquet")
    )
    counts = pd.crosstab(
        predictions["y_true"].astype(str),
        predictions["y_pred"].astype(str),
    ).reindex(index=labels, columns=labels, fill_value=0)
    counts.index = [DISPLAY.get(label, label) for label in labels]
    counts.columns = [DISPLAY.get(label, label) for label in labels]

    binary = len(labels) == 2
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    plot_fraction_count_heatmap(
        counts,
        ax=ax,
        cmap="Blues",
        square=True,
        fraction_fontsize=(
            BINARY_CONFUSION_FRACTION_FONTSIZE
            if binary
            else CONFUSION_FRACTION_FONTSIZE
        ),
        count_fontsize=(
            BINARY_CONFUSION_COUNT_FONTSIZE
            if binary
            else CONFUSION_COUNT_FONTSIZE
        ),
    )
    ax.set_xlabel(
        "Predicted class", fontsize=CONFUSION_AXIS_FONTSIZE, labelpad=5
    )
    ax.set_ylabel(
        "Human class", fontsize=CONFUSION_AXIS_FONTSIZE, labelpad=5
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        fontsize=CONFUSION_TICK_FONTSIZE,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=CONFUSION_TICK_FONTSIZE,
    )
    return _save(
        fig,
        output_dir,
        stem,
        tight_layout_pad=0.35,
        save_pad_inches=0.02,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--style", choices=("malca", "smplotlib"), default="smplotlib"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "figures" / args.style
    )
    _apply_plot_style(args.style)

    generated: list[Path] = []
    for head_key in HEADS:
        head_dir = model_dir / head_key
        labels = _labels(head_dir)
        generated += _plot_calibrated_confusion(
            head_dir,
            labels,
            output_dir=output_dir,
            stem=f"{head_key}_holdout_confusion_matrix",
        )
        generated += _plot_importance(
            head_dir,
            top_n=10,
            output_dir=output_dir,
            stem=f"{head_key}_feature_importance_gain_top10",
        )

    scores = pd.read_parquet(
        _require(model_dir / "all_candidates_tuned_calibrated_scores.parquet")
    )
    generated += _plot_feature_distributions(
        model_dir,
        "parent_four_class",
        scores,
        output_dir=output_dir,
        top_n=9,
    )

    manifest = {
        "model_dir": str(model_dir),
        "style": args.style,
        "score_semantics": (
            "Confusion matrices use locked-test sigmoid-calibrated predictions; "
            "feature importance is raw production-LightGBM gain."
        ),
        "generated_files": [
            str(path.relative_to(output_dir)) for path in generated
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} plot files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
