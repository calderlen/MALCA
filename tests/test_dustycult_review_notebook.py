from __future__ import annotations

import json
from pathlib import Path


def test_dustycult_review_notebook_uses_shared_display_helpers() -> None:
    notebook_path = Path("malca/notebooks/review/dustycult_reviewed_dippers.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )

    assert "from malca.review.dustycult_display import (" in source
    assert "def display_dustycult_review_panel" in source
    assert "display_dustycult_review_panel(candidate_id, FULL_PLOT_MODE)" in source
    assert "def plot_deterministic_and_full" not in source
    assert "def _select_fit_row" not in source
