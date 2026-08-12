"""Shared typography and count-matrix plotting for MALCA ML diagnostics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator

from malca.plotting.lightcurve_publication import PUBLICATION_STYLE


ML_PLOT_STYLE = {
    **PUBLICATION_STYLE,
    "savefig.bbox": "tight",
}
FEATURE_IMPORTANCE_TOP_N = 10


def apply_ml_plot_style() -> None:
    """Apply the repository's exact Computer Modern Bright ML plot style."""

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook")
    # Apply this last because seaborn also changes several Matplotlib rcParams.
    plt.rcParams.update(ML_PLOT_STYLE)


def suppress_categorical_tick_marks(
    ax: Axes, *, x: bool = False, y: bool = False
) -> Axes:
    """Hide tick marks on categorical axes while preserving their labels."""

    if x:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
    if y:
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y", which="both", left=False, right=False)
    return ax


def row_fraction_frame(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return row fractions and their count denominators."""

    numeric = counts.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if (numeric.to_numpy() < 0).any():
        raise ValueError("Count heatmaps cannot contain negative values")
    totals = numeric.sum(axis=1)
    fractions = numeric.div(totals.replace(0.0, np.nan), axis=0).fillna(0.0)
    return fractions, totals


def _compact_count(value: float) -> str:
    rounded = round(value)
    if np.isclose(value, rounded):
        return f"{int(rounded)}"
    return f"{value:g}"


def _annotation_color(ax: Axes, fraction: float) -> str:
    mesh = ax.collections[0]
    red, green, blue, _alpha = mesh.cmap(mesh.norm(fraction))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.52 else "0.12"


def plot_fraction_count_heatmap(
    counts: pd.DataFrame,
    *,
    ax: Axes | None = None,
    cmap: str = "Blues",
    fraction_format: str = ".2f",
    fraction_fontsize: float = 9.0,
    count_fontsize: float = 7.0,
    cbar: bool = False,
    **heatmap_kwargs: Any,
) -> Axes:
    """Plot row fractions with ``cell count / row total`` below each value.

    Cell color always represents the row fraction. The larger annotation is
    placed in the upper half of the cell and the smaller raw-count annotation
    is placed below it.
    """

    if not isinstance(counts, pd.DataFrame):
        counts = pd.DataFrame(counts)
    numeric_counts = (
        counts.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    )
    fractions, totals = row_fraction_frame(numeric_counts)
    if ax is None:
        ax = plt.gca()

    sns.heatmap(
        fractions,
        annot=False,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        cbar=cbar,
        ax=ax,
        **heatmap_kwargs,
    )
    for row_index, row_total in enumerate(totals.to_numpy(dtype=float)):
        for column_index, count in enumerate(
            numeric_counts.iloc[row_index].to_numpy(dtype=float)
        ):
            fraction = float(fractions.iloc[row_index, column_index])
            color = _annotation_color(ax, fraction)
            fraction_label = (
                format(fraction, fraction_format) if row_total > 0 else "--"
            )
            count_label = f"{_compact_count(count)}/{_compact_count(row_total)}"
            x_position = column_index + 0.5
            ax.text(
                x_position,
                row_index + 0.36,
                fraction_label,
                ha="center",
                va="center",
                color=color,
                fontsize=fraction_fontsize,
            )
            ax.text(
                x_position,
                row_index + 0.68,
                count_label,
                ha="center",
                va="center",
                color=color,
                fontsize=count_fontsize,
            )
    suppress_categorical_tick_marks(ax, x=True, y=True)
    return ax
