from __future__ import annotations

from pathlib import Path

from malca.review.interactive_plot import build_interactive_lightcurve_figure


def _write_native_lc(path: Path) -> None:
    rows = []
    for band, offset in ((0, 0.0), (1, 0.2)):
        for idx in range(4):
            jd = 2458000.0 + idx * 0.25 + band * 0.05
            mag = 14.0 + offset + idx * 0.01
            rows.append(f"{jd:.5f} {mag:.4f} 0.0300 1 7 {band} 0 cam7/field")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_residual_and_phase_hover_use_camera_band_names(tmp_path: Path) -> None:
    lc_path = tmp_path / "source.dat"
    _write_native_lc(lc_path)

    result = build_interactive_lightcurve_figure(
        {"path": str(lc_path), "period_consensus_days": 1.0},
        plot_dir=None,
        selected_cameras=[],
        filter_bad_cameras=False,
        show_baseline=False,
        show_event_markers=False,
        show_residuals=True,
        show_phase_fold=True,
        show_raw_mag=True,
        override_period=None,
        show_diagnostics=False,
        confidence_colors=False,
        run_params={"baseline_func": "global_median"},
        uirevision_key="test",
        selected_bands=["g", "V"],
    )

    fig = result["figure"]
    residual_traces = [
        trace
        for trace in fig.data
        if trace.mode == "markers"
        and trace.showlegend is False
        and "JD - 2458000" in str(trace.hovertemplate)
        and "φ:" not in str(trace.hovertemplate)
    ]
    phase_traces = [
        trace
        for trace in fig.data
        if trace.mode == "markers"
        and trace.showlegend is False
        and "φ:" in str(trace.hovertemplate)
    ]

    assert {trace.name for trace in residual_traces} == {"cam7 (g)", "cam7 (V)"}
    assert {trace.name for trace in phase_traces} == {"cam7 (g)", "cam7 (V)"}
    assert all("<b>%{fullData.name}</b>" in str(trace.hovertemplate) for trace in residual_traces)
    assert all("<b>%{fullData.name}</b>" in str(trace.hovertemplate) for trace in phase_traces)
    assert all("Phase-folded residual" not in str(trace.hovertemplate) for trace in phase_traces)
