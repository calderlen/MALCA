from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("astroquery")
pytest.importorskip("celerite2")

from malca.evaluation import reproduce


def test_main_impl_passes_trigger_mode_and_significance(monkeypatch, tmp_path: Path) -> None:
    parser = reproduce.build_parser()
    args = parser.parse_args([])
    args.trigger_mode = "posterior_prob"
    args.significance_threshold = 97.5

    captured: dict[str, object] = {}

    def fake_resolve_candidates(_spec):
        return []

    def fake_build_reproduction_report(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(reproduce, "resolve_candidates", fake_resolve_candidates)
    monkeypatch.setattr(reproduce, "build_reproduction_report", fake_build_reproduction_report)

    reproduce._main_impl(args, plot_out_dir=tmp_path)

    assert captured["trigger_mode"] == "posterior_prob"
    assert captured["significance_threshold"] == 97.5
