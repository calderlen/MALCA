"""Lightweight helpers for inline figure display in Jupyter notebooks."""
from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt


def show_figure(paths: list[Path], fig, *, close: bool = True) -> list[Path]:
    """Display a figure inline in Jupyter, then optionally close it."""
    try:
        from IPython.display import display

        display(fig)
    except ImportError:
        plt.show()
    if close:
        plt.close(fig)
    return paths


def reload_plotting_modules() -> None:
    """Reload plotting modules so notebook edits are picked up without a kernel restart."""
    import sys

    module_names = (
        "malca.plotting.notebook_display",
        "malca.plotting.lightcurve_publication",
        "malca.plotting.paper_figures",
    )
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name in module_names:
        importlib.import_module(module_name)
