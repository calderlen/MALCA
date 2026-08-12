"""Plot diagnostics for the focused four-class -> recurrence hierarchy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, NullLocator

from malca.meta_analysis.ml.plotting import (
    apply_ml_plot_style,
    plot_fraction_count_heatmap,
    suppress_categorical_tick_marks,
)
from malca.plotting.lightcurve_publication import (
    FIG_SINGLE_COL_SQUARE,
    FIG_SINGLE_COL_WIDTH,
    FIG_TWO_COL_WIDTH,
)


DEFAULT_MODEL_DIR = Path(
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "lightgbm_retrain_20260729T201841Z/"
    "four_class_dimming_hierarchy"
)
HEADS = ("parent_four_class", "dimming_recurrence")
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
FEATURE_IMPORTANCE_TOP_NS = (10, 20)
FEATURE_DISTRIBUTION_TOP_N = 9
PNG_DPI = 180
PLOT_STYLES = ("malca", "smplotlib")
FEATURE_IMPORTANCE_HEIGHT = {10: 3.65, 20: 6.35}
IMPORTANCE_FEATURE_FONTSIZE = 10.0
IMPORTANCE_AXIS_FONTSIZE = 12.0
IMPORTANCE_TICK_FONTSIZE = 10.0
CONFUSION_AXIS_FONTSIZE = 13.0
CONFUSION_TICK_FONTSIZE = 10.5
CONFUSION_FRACTION_FONTSIZE = 12.5
CONFUSION_COUNT_FONTSIZE = 9.5
BINARY_CONFUSION_FRACTION_FONTSIZE = 17.0
BINARY_CONFUSION_COUNT_FONTSIZE = 12.5
FEATURE_DISTRIBUTION_COLS = 3
FEATURE_DISTRIBUTION_ROW_HEIGHT = 1.45
FEATURE_DISTRIBUTION_BINS = 34
FEATURE_DISTRIBUTION_QUANTILES = (0.01, 0.99)
FEATURE_DISTRIBUTION_XLABEL_FONTSIZE = 7.5
FEATURE_DISTRIBUTION_LABEL_FONTSIZE = 8.5
FEATURE_DISTRIBUTION_TICK_FONTSIZE = 7.0
FEATURE_DISTRIBUTION_LEGEND_FONTSIZE = 7.5
FEATURE_DISTRIBUTION_GROUPS = {
    "parent_four_class": (
        ("Reviewed dimming", "reviewed_dimming", "#315f72", 0.50),
        ("Reviewed EB", "reviewed_eb", "#7d5a8e", 0.42),
        ("Reviewed junk", "reviewed_junk", "#b28a3e", 0.42),
        ("Reviewed other", "reviewed_other", "#c15f5f", 0.42),
        ("Unreviewed", "unreviewed", "#9aa0a6", 0.25),
    ),
    "dimming_recurrence": (
        ("Reviewed non-recurrent", "reviewed_non_recurrent", "#c15f5f", 0.40),
        ("Unreviewed parent-dimming", "unreviewed_parent_dimming", "#9aa0a6", 0.25),
        ("Reviewed recurrent", "reviewed_recurrent", "#315f72", 0.55),
    ),
}


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required hierarchy artifact is missing: {path}")
    return path


def _apply_plot_style(style: str) -> None:
    if style == "malca":
        apply_ml_plot_style()
        return
    if style == "smplotlib":
        import smplotlib

        style_path = (
            Path(smplotlib.__file__).resolve().parent / "smplot.mplstyle"
        )
        plt.style.use(style_path)
        smplotlib.set_style()
        plt.rcParams["savefig.bbox"] = "tight"
        return
    raise ValueError(f"Unsupported plot style: {style}")


def _save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    tight_layout_pad: float = 1.08,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
    save_pad_inches: float | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=tight_layout_pad, rect=tight_layout_rect)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    save_kwargs = (
        {"pad_inches": save_pad_inches}
        if save_pad_inches is not None
        else {}
    )
    fig.savefig(paths[0], dpi=PNG_DPI, **save_kwargs)
    fig.savefig(paths[1], **save_kwargs)
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


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _snapshot_db(model_dir: Path) -> Path:
    local_snapshot = model_dir / "training_review_snapshot.db"
    if local_snapshot.is_file():
        return local_snapshot
    snapshot_metadata = json.loads(
        _require(model_dir / "training_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot_path = Path(str(snapshot_metadata.get("snapshot_db", "")))
    return _require(snapshot_path)


def _load_feature_values(
    model_dir: Path, features: Iterable[str]
) -> pd.DataFrame:
    """Load selected model features from the immutable training snapshot."""

    requested = list(dict.fromkeys(str(feature) for feature in features))
    db_path = _snapshot_db(model_dir)
    derived_sources = {
        "wise_w3_error": ("w3_err",),
        "wise_w4_error": ("w4_err",),
        "parallax_snr": ("parallax", "parallax_error"),
    }
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as conn:
        available = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        selected = ["candidate_id"]
        for feature in requested:
            if feature in available:
                selected.append(feature)
            for source in derived_sources.get(feature, ()):
                if source in available:
                    selected.append(source)
        selected = list(dict.fromkeys(selected))
        select_list = ", ".join(
            _quote_identifier(column) for column in selected
        )
        values = pd.read_sql_query(
            f"SELECT {select_list} FROM candidates", conn
        )

    for feature, sources in derived_sources.items():
        if feature not in requested or feature in values.columns:
            continue
        if feature.startswith("wise_"):
            values[feature] = pd.to_numeric(
                values[sources[0]], errors="coerce"
            ).where(lambda series: series.gt(0.0))
        elif feature == "parallax_snr":
            parallax = pd.to_numeric(values[sources[0]], errors="coerce")
            error = pd.to_numeric(values[sources[1]], errors="coerce")
            values[feature] = parallax / error.where(error.gt(0.0))

    missing = [feature for feature in requested if feature not in values]
    if missing:
        snapshot_metadata = json.loads(
            _require(model_dir / "training_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        cache_path = Path(
            str(snapshot_metadata.get("recovery_feature_cache", ""))
        )
        if cache_path.is_file():
            cache_columns = set(pd.read_parquet(cache_path).columns)
            cached = [feature for feature in missing if feature in cache_columns]
            if cached:
                cache = pd.read_parquet(
                    cache_path, columns=["candidate_id", *cached]
                )
                values = values.merge(cache, on="candidate_id", how="left")
        missing = [feature for feature in requested if feature not in values]
    if missing:
        raise KeyError(
            "Top model features are absent from the training snapshot and "
            f"recovery cache: {missing}"
        )

    values["candidate_id"] = values["candidate_id"].astype(str)
    return values[["candidate_id", *requested]]


def _feature_distribution_masks(
    scores: pd.DataFrame, head_key: str
) -> dict[str, pd.Series]:
    unreviewed = scores["is_human_unreviewed"].fillna(False).astype(bool)
    if head_key == "parent_four_class":
        parent = scores["human_four_class_parent_label"].astype("string")
        return {
            "reviewed_dimming": parent.eq("dimming_event").fillna(False),
            "reviewed_eb": parent.eq("eclipsing_binary").fillna(False),
            "reviewed_junk": parent.eq("junk").fillna(False),
            "reviewed_other": parent.eq("other").fillna(False),
            "unreviewed": unreviewed,
        }
    if head_key == "dimming_recurrence":
        recurrence = scores["human_dipper_recurrence_label"].astype("string")
        parent_prediction = scores["predicted_parent_class"].astype("string")
        return {
            "reviewed_non_recurrent": recurrence.eq(
                "non_recurrent_given_dipper"
            ).fillna(False),
            "unreviewed_parent_dimming": (
                unreviewed & parent_prediction.eq("dimming_event").fillna(False)
            ),
            "reviewed_recurrent": recurrence.eq(
                "recurrent_given_dipper"
            ).fillna(False),
        }
    raise ValueError(f"Unsupported hierarchy head: {head_key}")


def _top_gain_features(head_dir: Path, top_n: int) -> list[str]:
    frame = pd.read_csv(_require(head_dir / "feature_importance_gain.csv"))
    frame["gain"] = pd.to_numeric(frame["gain"], errors="coerce").fillna(0)
    return (
        frame.nlargest(top_n, "gain")["feature"].astype(str).tolist()
    )


def _feature_distribution_title(feature: str, max_chars: int = 25) -> str:
    """Wrap a raw feature key at underscores for a narrow three-panel row."""

    tokens = feature.split("_")
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current}_{token}"
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _plot_feature_distributions(
    model_dir: Path,
    head_key: str,
    scores: pd.DataFrame,
    *,
    output_dir: Path,
    top_n: int = FEATURE_DISTRIBUTION_TOP_N,
) -> list[Path]:
    """Plot normalized top-feature histograms for reviewed/unreviewed cohorts."""

    features = _top_gain_features(model_dir / head_key, top_n)
    values = _load_feature_values(model_dir, features)
    frame = scores[["candidate_id"]].copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame = frame.merge(values, on="candidate_id", how="left", validate="1:1")
    masks = _feature_distribution_masks(scores.reset_index(drop=True), head_key)
    group_styles = FEATURE_DISTRIBUTION_GROUPS[head_key]
    rows = int(np.ceil(len(features) / FEATURE_DISTRIBUTION_COLS))
    fig, axes = plt.subplots(
        rows,
        FEATURE_DISTRIBUTION_COLS,
        figsize=(FIG_TWO_COL_WIDTH, FEATURE_DISTRIBUTION_ROW_HEIGHT * rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, feature in zip(axes_flat, features):
        numeric = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        finite = numeric.dropna()
        if finite.empty:
            ax.set_axis_off()
            continue
        lo, hi = finite.quantile(FEATURE_DISTRIBUTION_QUANTILES)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            unique = np.sort(finite.unique())
            if len(unique) < 2:
                ax.set_axis_off()
                continue
            lo, hi = float(unique[0]), float(unique[-1])
        bins = np.linspace(float(lo), float(hi), FEATURE_DISTRIBUTION_BINS + 1)
        for legend_label, group_key, color, alpha in group_styles:
            group_values = numeric.loc[masks[group_key]].dropna()
            group_values = group_values.loc[
                group_values.between(lo, hi, inclusive="both")
            ]
            if group_values.empty:
                continue
            weights = np.ones(len(group_values), dtype=float) / len(group_values)
            ax.hist(
                group_values,
                bins=bins,
                weights=weights,
                alpha=alpha,
                label=legend_label,
                color=color,
            )
        ax.set_xlabel(
            _feature_distribution_title(feature),
            fontsize=FEATURE_DISTRIBUTION_XLABEL_FONTSIZE,
            labelpad=2,
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=FEATURE_DISTRIBUTION_TICK_FONTSIZE,
        )
        ax.grid(alpha=0.23)
    for ax in axes_flat[len(features) :]:
        ax.set_axis_off()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        frameon=False,
        fontsize=FEATURE_DISTRIBUTION_LEGEND_FONTSIZE,
        columnspacing=1.2,
        handlelength=1.8,
    )
    axes[rows // 2, 0].set_ylabel(
        "Fraction of group",
        fontsize=FEATURE_DISTRIBUTION_LABEL_FONTSIZE,
        labelpad=4,
    )
    return _save(
        fig,
        output_dir,
        f"{head_key}_top{top_n}_feature_distributions",
        tight_layout_pad=0.35,
        tight_layout_rect=(0.0, 0.0, 1.0, 0.91),
        save_pad_inches=0.04,
    )


def _feature_display_name(feature: object) -> str:
    """Return a compact plotting label without changing the feature key."""

    label = str(feature)
    wise_colors = {
        "w1_w2": "W1-W2",
        "w1_w3": "W1-W3",
        "w2_w3": "W2-W3",
    }
    if label in wise_colors:
        return wise_colors[label]
    for prefix in ("stats_", "derived_"):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
            break
    replacements = (
        ("wise_", "WISE "),
        ("lafler_kinman_", "Lafler-Kinman "),
        ("variability_", ""),
        ("harmonics_", "harm. "),
        ("n_outliers_removed_robust_3sigma", "robust 3sigma outliers"),
        ("max_run_", "max-run "),
        ("lag1_", "lag-1 "),
        ("pdm_", "PDM "),
        ("lsp_", "LSP "),
        ("sed_", "SED "),
        ("sf_ml_", "SF ML "),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    label = label.replace("_", " ")
    token_replacements = {
        "mrp": "MRP",
        "rcs": "RCS",
        "mhps": "MHPS",
        "snr": "S/N",
        "w1": "W1",
        "w2": "W2",
        "w3": "W3",
        "w4": "W4",
    }
    return " ".join(
        token_replacements.get(token, token) for token in label.split()
    )


def _plot_confusion(
    head_dir: Path,
    labels: list[str],
    *,
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
    binary = len(labels) == 2
    fraction_fontsize = (
        BINARY_CONFUSION_FRACTION_FONTSIZE
        if binary
        else CONFUSION_FRACTION_FONTSIZE
    )
    count_fontsize = (
        BINARY_CONFUSION_COUNT_FONTSIZE
        if binary
        else CONFUSION_COUNT_FONTSIZE
    )
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    plot_fraction_count_heatmap(
        counts,
        ax=ax,
        cmap="Blues",
        square=True,
        fraction_fontsize=fraction_fontsize,
        count_fontsize=count_fontsize,
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


def _plot_importance(
    head_dir: Path,
    *,
    top_n: int,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    if top_n not in FEATURE_IMPORTANCE_HEIGHT:
        raise ValueError(f"Unsupported feature-importance count: {top_n}")
    frame = pd.read_csv(_require(head_dir / "feature_importance_gain.csv"))
    frame["gain"] = pd.to_numeric(frame["gain"], errors="coerce").fillna(0)
    frame = frame.nlargest(top_n, "gain").sort_values("gain")
    fig, ax = plt.subplots(
        figsize=(FIG_SINGLE_COL_WIDTH, FEATURE_IMPORTANCE_HEIGHT[top_n])
    )
    ax.barh(
        frame["feature"].map(_feature_display_name),
        frame["gain"],
        color="#3a78a1",
        height=0.72,
        zorder=2,
    )
    ax.set_xlabel(
        "LightGBM gain importance",
        fontsize=IMPORTANCE_AXIS_FONTSIZE,
        labelpad=12,
    )
    ax.grid(axis="x", alpha=0.27, zorder=0)
    suppress_categorical_tick_marks(ax, y=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_minor_locator(NullLocator())
    if frame["gain"].max() >= 10_000:
        ax.ticklabel_format(
            axis="x", style="sci", scilimits=(0, 0), useOffset=False
        )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=IMPORTANCE_FEATURE_FONTSIZE,
        pad=4,
    )
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        labelsize=IMPORTANCE_TICK_FONTSIZE,
    )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.margins(x=0.025, y=0.025)
    return _save(
        fig,
        output_dir,
        stem,
        tight_layout_pad=0.35,
        save_pad_inches=0.02,
    )


def _plot_reliability(
    head_dir: Path,
    labels: list[str],
    *,
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
            label=DISPLAY.get(label, label),
        )
        ax.scatter(
            subset["mean_probability"],
            subset["observed_rate"],
            s=np.clip(9 * np.sqrt(subset["n"]), 18, 115),
            color="#3a78a1",
            edgecolor="white",
            linewidth=0.45,
        )
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.23)
    for ax in axes.flat[len(labels) :]:
        ax.remove()
    fig.supxlabel("Mean score")
    fig.supylabel("Observed class fraction")
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
    suppress_categorical_tick_marks(ax, x=True)
    ax.tick_params(axis="x", which="major", rotation=30, labelsize=8.5)
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
    axes[1].grid(axis="y", alpha=0.23)
    return _save(fig, output_dir, "hierarchy_unreviewed_branch_structure")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--style", choices=PLOT_STYLES, default="malca")
    args = parser.parse_args(list(argv) if argv is not None else None)
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "figures"
    )

    _apply_plot_style(args.style)
    generated: list[Path] = []
    for key in HEADS:
        head_dir = model_dir / key
        labels = _labels(head_dir)
        generated += _plot_confusion(
            head_dir,
            labels,
            output_dir=output_dir,
            stem=f"{key}_holdout_confusion_matrix",
        )
        for top_n in FEATURE_IMPORTANCE_TOP_NS:
            generated += _plot_importance(
                head_dir,
                top_n=top_n,
                output_dir=output_dir,
                stem=f"{key}_feature_importance_gain_top{top_n}",
            )
        generated += _plot_reliability(
            head_dir,
            labels,
            output_dir=output_dir,
            stem=f"{key}_heldout_reliability",
        )

    scores = pd.read_parquet(
        _require(
            model_dir
            / "all_candidates_four_class_hierarchical_scores.parquet"
        )
    )
    for key in HEADS:
        generated += _plot_feature_distributions(
            model_dir,
            key,
            scores,
            output_dir=output_dir,
        )
    generated += _plot_leaf_counts(scores, output_dir=output_dir)
    generated += _plot_branch_structure(scores, output_dir=output_dir)
    manifest = {
        "model_dir": str(model_dir),
        "style": args.style,
        "feature_distributions": {
            "top_n_by_gain": FEATURE_DISTRIBUTION_TOP_N,
            "quantile_range": list(FEATURE_DISTRIBUTION_QUANTILES),
            "parent_unreviewed_cohort": "all human-unreviewed candidates",
            "recurrence_unreviewed_cohort": (
                "human-unreviewed candidates routed to dimming_event by the "
                "parent head"
            ),
        },
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
