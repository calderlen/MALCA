"""Tests for publication figure-size constants and helpers."""

from __future__ import annotations

import pytest

from malca.plotting.lightcurve_publication import (
    CMD_BUCKET_STYLE,
    FIG_GRID_PANEL_WIDTH,
    FIG_GRID_ROW_HEIGHT,
    FIG_HEATMAP_MIN_HEIGHT,
    FIG_HEATMAP_ROW_HEIGHT,
    FIG_LC_SINGLE_COL,
    FIG_LC_TWO_COL,
    FIG_ROC_PR_TWO_COL,
    FIG_SINGLE_COL_HEATMAP,
    FIG_SINGLE_COL_LC_WIDE,
    FIG_SINGLE_COL_SQUARE,
    FIG_SINGLE_COL_WIDTH,
    FIG_TWO_COL_STANDARD,
    FIG_TWO_COL_WIDTH,
    PUBLICATION_STYLE,
    PUBLICATION_TIGHT_LAYOUT_PAD,
    finalize_publication_figure,
    save_publication_figure,
    figsize_feature_grid,
    figsize_from_legacy,
    figsize_heatmap_single_col,
    figsize_heatmap_two_col,
    figsize_scale,
    figsize_two_col_grid,
    scaled_publication_text_sizes,
)


def test_publication_width_constants() -> None:
    assert FIG_SINGLE_COL_WIDTH == 3.5
    assert FIG_TWO_COL_WIDTH == 7.0
    assert FIG_SINGLE_COL_SQUARE == (3.5, 3.5)
    assert FIG_LC_SINGLE_COL == (3.5, 2.0)
    assert FIG_LC_TWO_COL == (7.0, 3.0)
    assert FIG_ROC_PR_TWO_COL == (7.0, 3.0)


def test_cmd_bucket_style_tuned_for_single_column_square() -> None:
    assert CMD_BUCKET_STYLE["Microlensing"]["size"] == 28
    assert CMD_BUCKET_STYLE["Dipper"]["size"] == 16


def test_figsize_scale_single_column_square() -> None:
    scale = figsize_scale((3.5, 2.8))
    assert scale == pytest.approx(3.5 / 10.0, rel=1e-3)


def test_figsize_scale_reference_is_unity() -> None:
    from malca.plotting.lightcurve_publication import FIG_HEATMAP_REFERENCE

    assert figsize_scale(FIG_HEATMAP_REFERENCE) == pytest.approx(1.0)


def test_figsize_heatmap_two_col() -> None:
    assert figsize_heatmap_two_col(40) == (7.0, pytest.approx(11.2))
    assert figsize_heatmap_two_col(1) == (7.0, FIG_HEATMAP_MIN_HEIGHT)


def test_figsize_heatmap_two_col_custom_row_height() -> None:
    assert figsize_heatmap_two_col(20, row_height=0.35) == (7.0, pytest.approx(7.0))
    assert figsize_heatmap_two_col(5, row_height=0.35) == (7.0, FIG_HEATMAP_MIN_HEIGHT)


def test_figsize_feature_grid() -> None:
    assert figsize_feature_grid(3, 2) == (FIG_GRID_PANEL_WIDTH * 3, FIG_GRID_ROW_HEIGHT * 2)
    assert figsize_feature_grid(3, 2) == (10.5, 5.0)


def test_legacy_scaled_figsizes() -> None:
    assert FIG_SINGLE_COL_HEATMAP == (3.5, 2.8)
    assert FIG_TWO_COL_STANDARD == (7.0, 4.0)
    assert figsize_from_legacy(10, 8) == FIG_SINGLE_COL_HEATMAP
    assert figsize_heatmap_single_col() == (FIG_SINGLE_COL_HEATMAP[0], pytest.approx(FIG_SINGLE_COL_HEATMAP[1]))
    assert figsize_two_col_grid(2, 4) == (7.0, FIG_GRID_ROW_HEIGHT * 4)


def test_scaled_publication_text_sizes() -> None:
    sizes = scaled_publication_text_sizes(FIG_SINGLE_COL_HEATMAP)
    assert sizes["label"] < 14.0
    assert sizes["title"] < 16.0


def test_publication_style_no_bbox_tight() -> None:
    assert PUBLICATION_STYLE.get("savefig.bbox") is None


def test_publication_tight_layout_pad_value() -> None:
    assert PUBLICATION_TIGHT_LAYOUT_PAD == 0.3


def test_finalize_publication_figure_runs_on_simple_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    finalize_publication_figure(fig)
    plt.close(fig)


def test_finalize_publication_figure_with_colorbar() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    im = ax.pcolormesh(np.random.rand(5, 5))
    plt.colorbar(im, ax=ax)
    finalize_publication_figure(fig)
    plt.close(fig)


def test_save_publication_figure_writes_file(tmp_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "test_output.png"
    save_publication_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0
