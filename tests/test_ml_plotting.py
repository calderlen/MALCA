from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.meta_analysis.ml.plotting import (
    FEATURE_IMPORTANCE_TOP_N,
    apply_ml_plot_style,
    plot_fraction_count_heatmap,
    row_fraction_frame,
)


def test_row_fraction_frame_preserves_counts_and_handles_empty_rows() -> None:
    counts = pd.DataFrame(
        [[3, 1], [0, 0]],
        index=["positive", "empty"],
        columns=["predicted positive", "predicted negative"],
    )

    fractions, totals = row_fraction_frame(counts)

    np.testing.assert_allclose(fractions.to_numpy(), [[0.75, 0.25], [0.0, 0.0]])
    np.testing.assert_allclose(totals.to_numpy(), [4.0, 0.0])


def test_fraction_count_heatmap_uses_two_line_cell_annotations() -> None:
    counts = pd.DataFrame(
        [[3, 1], [0, 4]],
        index=["positive", "negative"],
        columns=["positive", "negative"],
    )
    fig, ax = plt.subplots()

    returned_ax = plot_fraction_count_heatmap(counts, ax=ax)

    assert returned_ax is ax
    np.testing.assert_allclose(
        ax.collections[0].get_array().reshape(2, 2),
        [[0.75, 0.25], [0.0, 1.0]],
    )
    assert [text.get_text() for text in ax.texts] == [
        "0.75",
        "3/4",
        "0.25",
        "1/4",
        "0.00",
        "0/4",
        "1.00",
        "4/4",
    ]
    assert ax.texts[0].get_position()[1] < ax.texts[1].get_position()[1]
    assert ax.texts[0].get_fontsize() > ax.texts[1].get_fontsize()
    plt.close(fig)


def test_apply_ml_plot_style_uses_computer_modern_bright() -> None:
    with plt.rc_context():
        apply_ml_plot_style()

        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert "CMU Bright" in plt.rcParams["font.sans-serif"]
        assert plt.rcParams["text.usetex"] is True
        assert r"\usepackage{cmbright}" in plt.rcParams["text.latex.preamble"]


def test_feature_importance_plots_are_capped_at_ten_features() -> None:
    assert FEATURE_IMPORTANCE_TOP_N == 10
