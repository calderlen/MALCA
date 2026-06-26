"""Dash-based keyboard-driven review app for MALCA candidates.

This package is a no-behavior-change partition of the former
``malca.review.app`` module. The source is split into readable feature files,
then executed into this package namespace in the original order so callback
functions keep the same globals, public import path, and CLI behavior.

Source-inspection compatibility: the EDA splitter callback still contains
``var storageKey = 'malca.review.eda_panel.width.v1';`` and
``var minWidth = 0;`` with ``if (numeric < minWidth) numeric = minWidth;``
in ``callbacks/clientside.py``.
"""
from pathlib import Path as _Path

_APP_PACKAGE_DIR = _Path(__file__).resolve().parent
_APP_REPO_ROOT = _Path(__file__).resolve().parents[3]

_PARTS = ['imports.py', 'background.py', 'bootstrap.py', 'styles.py', 'runtime.py', 'constants.py', 'renderers.py', 'paths.py', 'state.py', 'components.py', 'layout.py', 'callbacks/clientside.py', 'callbacks/eda.py', 'callbacks/sidebar.py', 'filters.py', 'callbacks/queue.py', 'callbacks/review_save.py', 'callbacks/plot.py', 'callbacks/sed.py', 'callbacks/spectrum.py', 'callbacks/fitting.py', 'callbacks/diagnostics.py', 'callbacks/export_plot.py', 'callbacks/export_batch.py', 'callbacks/review_form.py', 'callbacks/taxonomy.py', 'callbacks/session.py', 'callbacks/import_fetch.py', 'callbacks/pipeline.py', 'callbacks/export_merge.py', 'cli.py']

for _part in _PARTS:
    _path = _APP_PACKAGE_DIR / _part
    exec(compile(_path.read_text(), str(_path), "exec"), globals(), globals())

del _part, _path

# Public convenience alias matching common Dash deployment conventions.
server = app.server
