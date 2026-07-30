from __future__ import annotations

from pathlib import Path
import json
import struct
import subprocess
import sys
import threading
import time

import pytest

from malca.review.tui_image_viewer import (
    _load_window_geometry,
    _normalized_window_geometry,
    _save_window_geometry,
)
from malca.review.tui_render import (
    ImageCoordinator,
    RenderRequest,
    _DimmingComplexZoom,
    _apply_tui_axis_theme,
    _decode_cutout_bytes,
    _event_window_polarity,
    _format_tui_coordinate_header,
    _load_display_frame,
    _period_processing_config,
    _register_native_cmu_bright_fonts,
    _render_event_zoom_panel,
    _review_style_for_theme,
    _resolve_best_phase_period,
    _style_tui_sed_legend,
    _survey_cutout,
    _write_render_metadata,
    launch_viewer,
    persistent_viewer_command,
    quicklook_command,
    render_lightcurve_png,
    tui_plot_theme,
    resolve_lightcurve_path,
)


def _request(candidate_id: str, **kwargs) -> RenderRequest:
    return RenderRequest(candidate_id=candidate_id, asas_sn_id=None, **kwargs)


def test_tui_plot_theme_matches_review_gui_modes() -> None:
    assert tui_plot_theme("white").figure == "#ffffff"
    assert tui_plot_theme("black").figure == "#0d0d0d"
    assert tui_plot_theme("gray").name == "white"
    assert tui_plot_theme(None).name == "black"
    assert "rgba(" not in tui_plot_theme("white").grid


def test_tui_uses_native_cmu_bright_without_latex(monkeypatch) -> None:
    monkeypatch.setattr(
        "malca.review.tui_render._register_native_cmu_bright_fonts",
        lambda: True,
    )

    style = _review_style_for_theme(tui_plot_theme("black"))

    assert style["text.usetex"] is False
    assert style["font.sans-serif"] == ["CMU Bright"]
    assert style["mathtext.fontset"] == "custom"
    assert style["mathtext.rm"] == "CMU Bright"
    assert style["mathtext.fallback"] == "cm"
    assert "text.latex.preamble" not in style


def test_tui_preserves_latex_fallback_when_native_cmu_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "malca.review.tui_render._register_native_cmu_bright_fonts",
        lambda: False,
    )

    style = _review_style_for_theme(tui_plot_theme("white"))

    assert style["text.usetex"] is True
    assert "cmbright" in str(style["text.latex.preamble"])


def test_native_cmu_registration_uses_texlive_opentype_faces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import matplotlib.font_manager as font_manager

    regular = tmp_path / "cmunbmr.otf"
    oblique = tmp_path / "cmunbmo.otf"
    regular.write_bytes(b"font")
    oblique.write_bytes(b"font")
    registered: list[str] = []

    class Completed:
        returncode = 0
        stdout = str(regular)

    monkeypatch.setattr(
        "malca.review.tui_render.shutil.which",
        lambda name: "/usr/bin/kpsewhich" if name == "kpsewhich" else None,
    )
    monkeypatch.setattr(
        "malca.review.tui_render.subprocess.run",
        lambda *_args, **_kwargs: Completed(),
    )
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [])
    monkeypatch.setattr(
        font_manager.fontManager,
        "addfont",
        lambda path: registered.append(str(path)),
    )
    _register_native_cmu_bright_fonts.cache_clear()
    try:
        assert _register_native_cmu_bright_fonts() is True
    finally:
        _register_native_cmu_bright_fonts.cache_clear()

    assert registered == [str(oblique), str(regular)]


def test_persistent_viewer_geometry_round_trips_between_sessions(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "viewer_geometry.json"

    assert _load_window_geometry(state_path) is None
    assert _save_window_geometry(state_path, "1440x900-120+45") is True
    assert _load_window_geometry(state_path) == "1440x900-120+45"
    assert _normalized_window_geometry("200x100+8-12") == "420x315+8-12"
    assert _normalized_window_geometry("not geometry") is None


def test_tui_coordinate_header_uses_explicit_equals_signs() -> None:
    assert _format_tui_coordinate_header(251.4871, -43.5028) == (
        "α = 251.4871°, δ = -43.5028°"
    )


def test_event_zoom_panel_uses_atlas_selected_dimming_complex() -> None:
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {
            "jd": [
                2458098.9,
                2458099.8,
                2458100.0,
                2458100.3,
                2458199.5,
                2458200.0,
                2458200.4,
            ],
            "mag": [13.9, 13.7, 14.35, 14.1, 13.8, 13.5, 13.7],
            "mag_err": [0.03] * 7,
        }
    )
    dimming_zoom = _DimmingComplexZoom(
        zoom_start_jd=2458099.0,
        zoom_end_jd=2458101.0,
        event_start_jd=2458099.8,
        event_end_jd=2458100.3,
        peak_jd=2458100.0,
        status="baseline_bounded",
    )
    fig, axis = plt.subplots()
    try:
        _render_event_zoom_panel(
            axis,
            frame,
            dimming_zoom,
            jd_offset=2458000.0,
            jd_xlabel="JD − 2458000 [d]",
            np=np,
            theme=tui_plot_theme("white"),
        )
        fig.canvas.draw()

        assert axis.get_title() == ""
        assert axis.get_xlabel() == "JD − 2458000 [d]"
        assert axis.get_ylabel() == "$m$ [mag]"
        assert axis.yaxis_inverted()
        assert axis.get_xlim() == pytest.approx((99.0, 101.0))
        assert len(axis.patches) == 1
        assert any((collection.get_offsets()[:, 0] == 100.0).any() for collection in axis.collections)
    finally:
        plt.close(fig)


def test_event_window_polarity_uses_review_label_then_event_evidence() -> None:
    assert _event_window_polarity({"morphology_primary": "brightening_event"}) == (
        "brightening"
    )
    assert _event_window_polarity(
        {
            "morphology_primary": "dimming_event",
            "jump_significant": 1,
        }
    ) == "dimming"
    assert _event_window_polarity(
        {
            "dip_significant": 0,
            "jump_significant": 1,
        }
    ) == "brightening"
    assert _event_window_polarity(
        {
            "dip_best_delta_bic": 12.0,
            "jump_best_delta_bic": 120.0,
        }
    ) == "brightening"
    assert _event_window_polarity({}) == "dimming"


def test_apply_tui_axis_theme_uses_matplotlib_grid_colors() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = tui_plot_theme("white")
    fig, ax = plt.subplots()
    try:
        _apply_tui_axis_theme(ax, theme)
        fig.savefig("/tmp/tui_theme_grid_test.png", dpi=72)
    finally:
        plt.close(fig)


def test_tui_sed_legend_is_compact_opaque_and_upper_right() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    fig, ax = plt.subplots()
    try:
        ax.scatter([1.0], [1.0], label="APASS")
        ax.scatter([2.0], [2.0], label="Gaia DR3")
        ax.plot([1.0, 2.0], [2.0, 1.0], label="Castelli/Kurucz fit")

        _style_tui_sed_legend(ax, tui_plot_theme("black"))
        legend = ax.get_legend()

        assert legend is not None
        assert legend._loc == Legend.codes["upper right"]
        assert legend.get_frame().get_alpha() == pytest.approx(1.0)
        assert [text.get_text() for text in legend.get_texts()] == [
            "APASS",
            "Gaia DR3",
        ]
        assert all(
            text.get_fontsize() == pytest.approx(8.0)
            for text in legend.get_texts()
        )
    finally:
        plt.close(fig)


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def test_resolver_prefers_existing_stored_path(tmp_path: Path) -> None:
    stored = tmp_path / "stored.dat3"
    stored.write_text("stored", encoding="ascii")
    run_dir = tmp_path / "run"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "123.dat3").write_text("bundle", encoding="ascii")

    request = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=stored,
        source_path=None,
        db_path=None,
        run_dir=run_dir,
    )

    assert resolve_lightcurve_path(request) == stored.resolve()


def test_resolver_infers_bundle_from_review_db_and_uses_id_extension_order(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    review_dir = run_dir / "review"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    review_dir.mkdir(parents=True)
    bundle_dir.mkdir(parents=True)
    db_path = review_dir / "review.db"
    db_path.write_bytes(b"")

    candidate_dat3 = bundle_dir / "candidate-a.dat3"
    asas_raw2 = bundle_dir / "123.raw2"
    asas_dat2 = bundle_dir / "123.dat2"
    candidate_dat3.write_text("candidate", encoding="ascii")
    asas_raw2.write_text("per-camera statistics, not a light curve", encoding="ascii")
    asas_dat2.write_text("asas", encoding="ascii")

    request = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=asas_raw2,
        source_path="/stale/run",
        db_path=db_path,
    )

    # ASAS-SN ID wins over the candidate ID.  A raw2 statistics sidecar must
    # never win over its actual dat2 light curve.
    assert resolve_lightcurve_path(request) == asas_dat2.resolve()


def test_quicklook_uses_owned_process_without_a_shell(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "current.png"
    calls = []
    process = object()

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return process

    monkeypatch.setattr("malca.review.tui_render.subprocess.Popen", fake_popen)

    assert quicklook_command(image) == [
        "/usr/bin/qlmanage",
        "-p",
        str(image),
    ]
    assert launch_viewer(image, "quicklook") is process

    assert calls[0][0] == quicklook_command(image)
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["stdout"] is subprocess.DEVNULL
    assert calls[0][1]["stderr"] is subprocess.DEVNULL
    manifest = tmp_path / "viewer.json"
    assert launch_viewer(manifest, "window") is process
    assert calls[1][0] == persistent_viewer_command(manifest)
    assert launch_viewer(image, "none") is None
    with pytest.raises(ValueError, match="quicklook, none"):
        launch_viewer(image, "browser")


def test_render_lightcurve_png_smoke_has_fixed_dimensions(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat3"
    lc_path.write_text(
        "9000.0 14.10 0.02 1 4 0 0 ba/F1\n"
        "9001.0 14.25 0.03 1 5 1 0 bb/F2\n"
        "9002.0 18.00 0.04 0 4 0 0 ba/F1\n",
        encoding="ascii",
    )
    destination = tmp_path / "current.png"
    request = RenderRequest(candidate_id="candidate-a", asas_sn_id="123", lc_path=lc_path)

    returned = render_lightcurve_png(request, destination)

    assert returned == destination
    png = destination.read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (2430, 1800)
    assert not list(tmp_path.glob(".current.*.png"))


def test_render_asassn_window_applies_configured_padding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lc_path = tmp_path / "123.dat3"
    lc_path.write_text(
        "9000.0 14.10 0.02 1 4 0 0 ba/F1\n"
        "9001.0 14.25 0.03 1 5 1 0 bb/F2\n",
        encoding="ascii",
    )
    destination = tmp_path / "asassn-window.png"
    observed_xlim = []

    from malca.review import tui_render

    original_style_legend = tui_render._style_tui_legend

    def style_legend_spy(ax, theme):
        observed_xlim.append(ax.get_xlim())
        return original_style_legend(ax, theme)

    monkeypatch.setattr(tui_render, "_style_tui_legend", style_legend_spy)

    render_lightcurve_png(
        RenderRequest(
            candidate_id="candidate-a",
            lc_path=lc_path,
            time_window_mode="asassn",
            asassn_window_padding_days=30.0,
        ),
        destination,
    )

    assert observed_xlim
    assert observed_xlim[-1][1] - observed_xlim[-1][0] == pytest.approx(61.0)
    metadata = json.loads(destination.with_suffix(".json").read_text())
    assert metadata["time_window_mode"] == "asassn"
    assert metadata["asassn_window_padding_days"] == 30.0
    assert metadata["external_cadence_bin_days"] == pytest.approx(1.0)


def test_render_lightcurve_png_adds_sed_to_right_column(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lc_path = tmp_path / "123.dat3"
    lc_path.write_text(
        "9000.0 14.10 0.02 1 4 0 0 ba/F1\n"
        "9001.0 14.25 0.03 1 5 1 0 bb/F2\n",
        encoding="ascii",
    )
    destination = tmp_path / "with-sed.png"
    rendered: list[str] = []

    def fake_sed_panel(ax, request, _np, *, theme) -> None:
        rendered.append(str(request.candidate_id))
        ax.text(0.5, 0.5, "SED", ha="center", va="center")

    monkeypatch.setattr(
        "malca.review.tui_render._render_sed_panel",
        fake_sed_panel,
    )

    render_lightcurve_png(
        RenderRequest(candidate_id="candidate-with-sed", lc_path=lc_path),
        destination,
    )

    assert rendered == ["candidate-with-sed"]
    assert destination.is_file()


def test_render_delegates_lightcurve_and_phase_axes_to_publication_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pandas as pd

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text(
        "9000.0 14.10 0.02 1 4 0 0 ba/F1\n"
        "9001.0 14.25 0.03 1 5 1 0 bb/F2\n",
        encoding="ascii",
    )
    observed = []

    def fake_phase_source(request, resolved_path):
        observed.append(("phase_source", resolved_path))
        return pd.DataFrame(
            {
                "JD": [9000.0, 9000.5, 9001.0, 9001.5],
                "mag": [14.1, 14.2, 14.1, 14.2],
                "error": [0.02, 0.02, 0.02, 0.02],
                "resid": [-0.02, 0.03, -0.02, 0.03],
                "camera_name": ["ba", "bb", "ba", "bb"],
                "v_g_band": [0, 1, 0, 1],
            }
        ), 9000.0

    monkeypatch.setattr(
        "malca.review.tui_render._prepare_review_phase_source",
        fake_phase_source,
    )
    from malca.plotting import lightcurve_publication

    original_lightcurve_panel = lightcurve_publication.plot_lightcurve_panel
    original_phase_panel = lightcurve_publication.plot_phase_panel

    def lightcurve_spy(*args, **kwargs):
        observed.append(("lightcurve", kwargs.get("title")))
        return original_lightcurve_panel(*args, **kwargs)

    def phase_spy(*args, **kwargs):
        observed.append(("phase", kwargs.get("period_days"), kwargs.get("xlim")))
        return original_phase_panel(*args, **kwargs)

    monkeypatch.setattr(lightcurve_publication, "plot_lightcurve_panel", lightcurve_spy)
    monkeypatch.setattr(lightcurve_publication, "plot_phase_panel", phase_spy)
    destination = tmp_path / "phase.png"
    request = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=lc_path,
        stored_phase_period_days=2.5,
        stored_phase_source="period_consensus_days",
    )

    render_lightcurve_png(request, destination)

    assert ("phase_source", lc_path.resolve()) in observed
    assert any(item[0] == "lightcurve" for item in observed)
    assert ("phase", 2.5, (0.0, 2.0)) in observed
    width, height = struct.unpack(">II", destination.read_bytes()[16:24])
    assert (width, height) == (2430, 1800)


def test_render_delegates_missing_period_placeholder_to_publication_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lc_path = tmp_path / "123.dat3"
    lc_path.write_text(
        "9000.0 14.10 0.02 1 4 0 0 ba/F1\n"
        "9001.0 14.25 0.03 1 5 1 0 bb/F2\n",
        encoding="ascii",
    )
    observed = []
    from malca.plotting import lightcurve_publication

    original_placeholder = lightcurve_publication.plot_phase_placeholder

    def placeholder_spy(*args, **kwargs):
        observed.append(args[1])
        return original_placeholder(*args, **kwargs)

    monkeypatch.setattr(
        lightcurve_publication,
        "plot_phase_placeholder",
        placeholder_spy,
    )
    destination = tmp_path / "no-period.png"

    render_lightcurve_png(
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path),
        destination,
    )

    assert observed == ["No candidate payload for automatic period search"]
    assert destination.is_file()


def test_publication_median_fallback_and_phase_plot_generate_two_cycles() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    from malca.plotting.lightcurve_publication import (
        plot_phase_panel,
        prepare_median_centered_phase_source,
    )

    frame = pd.DataFrame(
        {
            "jd": [10.0, 11.0],
            "mag": [14.0, 14.2],
            "mag_err": [0.02, 0.02],
            "_tui_camera": ["ba", "ba"],
            "_tui_band": ["g", "g"],
        }
    )

    source = prepare_median_centered_phase_source(
        frame,
        time_col="jd",
        value_col="mag",
        error_col="mag_err",
        band_col="_tui_band",
        camera_col="_tui_camera",
    )
    fig, ax = plt.subplots()
    try:
        result = plot_phase_panel(
            ax,
            source,
            period_days=4.0,
            epoch_jd=10.0,
            value_mode="resid",
            group_by="band-camera",
            legend="none",
        )
    finally:
        plt.close(fig)

    assert result.frame["time_plot"].tolist() == [0.0, 0.25, 1.0, 1.25]
    assert result.frame["value"].tolist() == pytest.approx(
        [-0.1, 0.1, -0.1, 0.1]
    )


def test_cache_key_changes_with_phase_period_and_source(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    base = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=lc_path,
        stored_phase_period_days=2.5,
        stored_phase_source="period_consensus_days",
    )
    different_period = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=lc_path,
        stored_phase_period_days=1.25,
        stored_phase_source="period_consensus_days",
    )
    different_source = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=lc_path,
        stored_phase_period_days=2.5,
        stored_phase_source="phase_period_days",
    )

    keys = {
        ImageCoordinator._request_key(base),
        ImageCoordinator._request_key(different_period),
        ImageCoordinator._request_key(different_source),
    }

    assert len(keys) == 3


def test_cache_key_tracks_display_controls_survey_and_coordinates(tmp_path: Path) -> None:
    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    base = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"ra": 10.0, "dec": -20.0},
    )
    variants = [
        base,
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, manual_phase_period_days=2.0),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, phase_multiplier=2.0),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, force_period_search=True),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, camera_view="cleaned"),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, show_event_markers=True),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, show_external_lightcurves=False),
        RenderRequest(
            candidate_id="candidate-a",
            lc_path=lc_path,
            external_lightcurve_sources=("ztf",),
        ),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, time_window_mode="asassn"),
        RenderRequest(
            candidate_id="candidate-a",
            lc_path=lc_path,
            time_window_mode="asassn",
            asassn_window_padding_days=30.0,
        ),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, plot_theme="black"),
        RenderRequest(candidate_id="candidate-a", lc_path=lc_path, survey_key="dss2"),
        RenderRequest(
            candidate_id="candidate-a",
            lc_path=lc_path,
            payload={"ra": 11.0, "dec": -20.0},
        ),
        RenderRequest(
            candidate_id="candidate-a",
            lc_path=lc_path,
            payload={"ra": 10.0, "dec": -20.0, "jump_significant": 1},
        ),
    ]

    assert len({ImageCoordinator._request_key(request) for request in variants}) == len(
        variants
    )


def test_cache_key_tracks_run_params_and_renderer_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    lightcurve_dir = run_dir / "bundle_assets" / "lightcurves"
    lightcurve_dir.mkdir(parents=True)
    lc_path = lightcurve_dir / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    run_params = run_dir / "run_params.json"
    run_params.write_text("{}", encoding="ascii")
    request = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=lc_path,
        run_dir=run_dir,
    )

    initial = ImageCoordinator._request_key(request)
    run_params.write_text('{"baseline_func": "global_median"}', encoding="ascii")
    changed_params = ImageCoordinator._request_key(request)
    monkeypatch.setattr("malca.review.tui_render.RENDER_CACHE_VERSION", "next-version")
    changed_version = ImageCoordinator._request_key(request)

    assert len({initial, changed_params, changed_version}) == 3


def test_missing_stored_period_uses_pipeline_consensus_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    calls = []

    def fake_search(payload, **kwargs):
        calls.append((payload, kwargs))
        return {"best_period": 2010.78}, "Pipeline long_ls: P=2010.78 d"

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        fake_search,
    )
    request = RenderRequest(
        candidate_id="stv_1010965",
        asas_sn_id="1010965",
        lc_path=lc_path,
        payload={"periodicity_period": 5.998840350865932},
        phase_search_min_days=0.1,
        phase_search_max_days=2500.0,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == pytest.approx(2010.78)
    assert source == "Pipeline search"
    assert warning == ""
    assert calls[0][0]["path"] == str(lc_path.resolve())
    assert calls[0][1]["min_period"] == 0.1
    assert calls[0][1]["max_period"] == 2500.0


def test_pipeline_period_search_receives_only_cleaning_kwargs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    calls = []

    def fake_search(payload, **kwargs):
        calls.append((payload, kwargs))
        return {"best_period": 52.98}, "Pipeline consensus"

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {
            "filter_bad_cameras": True,
            "scatter_ratio": 0.25,
            "clean_max_error_absolute": 0.2,
            "clean_max_error_sigma": 5.0,
            "baseline_name": "per_camera_gp",
            "baseline_kwargs": {"length_scale": 30.0},
        },
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        fake_search,
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"candidate_id": "candidate-a"},
        force_period_search=True,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == pytest.approx(52.98)
    assert source == "Pipeline search (forced)"
    assert warning == ""
    kwargs = calls[0][1]
    assert kwargs["filter_bad_cameras"] is True
    assert kwargs["scatter_ratio"] == 0.25
    assert kwargs["clean_max_error_absolute"] == 0.2
    assert kwargs["clean_max_error_sigma"] == 5.0
    assert "baseline_name" not in kwargs
    assert "baseline_kwargs" not in kwargs


def test_stored_period_is_harmonic_checked_before_render(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    calls = []

    def fake_check(payload, **kwargs):
        calls.append((payload, kwargs))
        return {"best_period": 1.25}, "Auto harmonic check"

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(period_search, "run_harmonic_check_for_payload", fake_check)
    request = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        lc_path=lc_path,
        payload={"period_consensus_days": 2.5},
        stored_phase_period_days=2.5,
        stored_phase_source="period_consensus_days",
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == 1.25
    assert source == "Auto harmonic check"
    assert warning == ""
    assert calls[0][0]["phase_period_days"] == 2.5
    assert calls[0][0]["phase_source"] == "period_consensus_days"


def test_short_stored_period_is_harmonic_checked_regardless_of_cycle_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    calls = []

    def fake_harmonic(payload, **kwargs):
        calls.append((payload, kwargs))
        return {"best_period": 2.0}, "Auto harmonic check"

    def fail_pipeline(*args, **kwargs):
        raise AssertionError("valid stored periods must not trigger pipeline search")

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        fail_pipeline,
    )
    monkeypatch.setattr(
        period_search,
        "run_harmonic_check_for_payload",
        fake_harmonic,
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={
            "phase_period_days": 2.0,
            "stats_time_span_days": 4000.0,
        },
        stored_phase_period_days=2.0,
        stored_phase_source="phase_period_days",
        phase_search_min_days=0.1,
        phase_search_max_days=2500.0,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == pytest.approx(2.0)
    assert source == "Auto harmonic check"
    assert warning == ""
    assert len(calls) == 1


def test_manual_period_and_multiplier_bypass_automatic_search(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    monkeypatch.setattr(
        period_search,
        "run_period_search_for_payload",
        lambda *args, **kwargs: pytest.fail("manual period must bypass search"),
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"ra": 1.0, "dec": 2.0},
        stored_phase_period_days=9.0,
        manual_phase_period_days=2.4,
        phase_multiplier=0.5,
        force_period_search=True,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == pytest.approx(1.2)
    assert source == "Manual ×0.5"
    assert warning == ""


def test_forced_period_search_ignores_stored_period(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    calls = []

    def fake_search(payload, **kwargs):
        calls.append((payload, kwargs))
        return {"best_period": 0.75}, "Pipeline consensus"

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        fake_search,
    )
    monkeypatch.setattr(
        period_search,
        "run_harmonic_check_for_payload",
        lambda *args, **kwargs: pytest.fail("forced search must skip harmonic check"),
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"period_consensus_days": 2.5},
        stored_phase_period_days=2.5,
        stored_phase_source="period_consensus_days",
        force_period_search=True,
        phase_multiplier=2.0,
        phase_search_min_days=0.1,
        phase_search_max_days=10.0,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == pytest.approx(1.5)
    assert source == "Pipeline search (forced) ×2"
    assert warning == ""
    assert calls[0][1]["min_period"] == 0.1
    assert calls[0][1]["max_period"] == 10.0


def test_weak_review_period_candidate_is_labeled_as_weak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        lambda *args, **kwargs: (
            {
                "best_period": 1.000266,
                "period_method": "pdm_review_candidate",
            },
            "Pipeline PDM candidate",
        ),
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"candidate_id": "candidate-a"},
        force_period_search=True,
        phase_search_min_days=0.1,
        phase_search_max_days=10.0,
    )

    period, source, warning = _resolve_best_phase_period(
        request,
        lc_path.resolve(),
    )

    assert period == pytest.approx(1.000266)
    assert source == "PDM candidate (weak)"
    assert warning == ""


def test_forced_pipeline_search_does_not_fall_back_outside_selected_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        lambda *args, **kwargs: (None, "Pipeline consensus: no valid period"),
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"period_consensus_days": 52.98},
        stored_phase_period_days=52.98,
        stored_phase_source="period_consensus_days",
        force_period_search=True,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period is None
    assert source == "Pipeline search (forced)"
    assert "no valid period" in warning.lower()


def test_forced_pipeline_search_strictly_honors_selected_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    calls = []

    def fake_search(payload, **kwargs):
        calls.append(kwargs)
        return {"best_period": 1000.0}, "Pipeline PDM"

    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        fake_search,
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={
            "long_ls_period_days": 2222.29,
            "long_ls_is_significant": True,
            "stats_time_span_days": 3704.0,
        },
        stored_phase_period_days=2222.29,
        stored_phase_source="long_ls_period_days",
        force_period_search=True,
        phase_search_min_days=0.1,
        phase_search_max_days=1852.0,
    )

    period, source, warning = _resolve_best_phase_period(request, lc_path.resolve())

    assert period == pytest.approx(1000.0)
    assert source == "Pipeline search (forced)"
    assert warning == ""
    assert calls[0]["max_period"] == pytest.approx(1852.0)


def test_forced_pipeline_search_rejects_period_outside_selected_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from malca.review import period_search

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    monkeypatch.setattr(
        "malca.review.tui_render._period_processing_config",
        lambda request, path: {},
    )
    monkeypatch.setattr(
        period_search,
        "run_pipeline_period_search_for_payload",
        lambda *args, **kwargs: (
            {"best_period": 3.4303091769975387e-19},
            "Pipeline event_period",
        ),
    )
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"period_consensus_days": 52.98},
        stored_phase_period_days=52.98,
        stored_phase_source="period_consensus_days",
        force_period_search=True,
        phase_search_min_days=0.1,
        phase_search_max_days=10.0,
    )

    period, source, warning = _resolve_best_phase_period(
        request,
        lc_path.resolve(),
    )

    assert period is None
    assert source == "Pipeline search (forced)"
    assert "outside forced 0.1–10 d window" in warning


def test_camera_view_controls_raw_and_phase_camera_filtering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pandas as pd

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text("light curve", encoding="ascii")
    cleaned = pd.DataFrame(
        {
            "JD": [2459000.0],
            "mag": [14.1],
            "camera_name": ["good-camera"],
            "camera#": [4],
            "v_g_band": [0],
        }
    )
    observed = []

    def fake_cleaned(path, **kwargs):
        observed.append(kwargs["filter_bad_cameras"])
        return cleaned, {9}, {}

    monkeypatch.setattr(
        "malca.review.native_lightcurve._load_cleaned_df",
        fake_cleaned,
    )
    cleaned_request = RenderRequest(
        candidate_id="candidate-a", lc_path=lc_path, camera_view="cleaned"
    )
    all_request = RenderRequest(
        candidate_id="candidate-a", lc_path=lc_path, camera_view="all"
    )

    display = _load_display_frame(
        cleaned_request,
        lc_path,
        lambda *args, **kwargs: pytest.fail("cleaned mode uses cleaned loader"),
    )

    assert display["camera_name"].tolist() == ["good-camera"]
    assert observed == [True]
    assert _period_processing_config(cleaned_request, lc_path)["filter_bad_cameras"] is True
    assert _period_processing_config(all_request, lc_path)["filter_bad_cameras"] is False


def test_decaps_blank_tile_falls_back_to_dss2_without_live_network(monkeypatch) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    calls = []

    def fake_fetch(url, timeout=2.0):
        calls.append(url)
        return b"decaps" if "DECaPS" in url else b"dss2"

    def fake_decode(content, _plt):
        if content == b"decaps":
            return np.ones((32, 32, 3), dtype=float)
        image = np.zeros((32, 32, 3), dtype=float)
        image[::2, ::2] = 1.0
        return image

    monkeypatch.setattr("malca.review.tui_render._fetch_cutout_bytes", fake_fetch)
    monkeypatch.setattr("malca.review.tui_render._decode_cutout_bytes", fake_decode)
    request = RenderRequest(
        candidate_id="candidate-a",
        payload={"ra": 164.499083, "dec": -83.218161},
    )

    result = _survey_cutout(request, plt, np)

    assert result.image is not None
    assert result.label == ""
    assert result.message == ""
    assert len(calls) == 2
    assert "DECaPS" in calls[0]
    assert "DSS2" in calls[1]


def test_cutout_decoder_accepts_jpeg_bytes_without_a_filename_suffix() -> None:
    from io import BytesIO

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    output = BytesIO()
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(
        output,
        format="JPEG",
    )

    decoded = _decode_cutout_bytes(output.getvalue(), plt)

    assert decoded.shape == (8, 8, 3)


def test_missing_cutout_coordinates_is_nonfatal_placeholder(monkeypatch) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    monkeypatch.setattr(
        "malca.review.tui_render._fetch_cutout_bytes",
        lambda *args, **kwargs: pytest.fail("missing coordinates must not fetch"),
    )

    result = _survey_cutout(RenderRequest(candidate_id="candidate-a"), plt, np)

    assert result.image is None
    assert result.label == ""
    assert "No RA/Dec" in result.message


def test_render_event_marker_toggle_reuses_canonical_event_helper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pandas as pd

    from malca.review import native_lightcurve

    lc_path = tmp_path / "123.dat3"
    lc_path.write_text(
        "9000.0 14.10 0.02 1 4 0 0 ba/F1\n"
        "9001.0 14.25 0.03 1 5 1 0 bb/F2\n",
        encoding="ascii",
    )
    observed = []

    def fake_events(payload, jd_offset, run_params, *, lc_median=None):
        observed.append((payload, jd_offset, lc_median))
        return [
            {
                "kind": "dip",
                "t0": 2459000.5,
                "half_width": 0.2,
                "base_color": "#ff6b6b",
            }
        ]

    monkeypatch.setattr(native_lightcurve, "_event_entries", fake_events)
    monkeypatch.setattr(
        "malca.review.tui_render._prepare_review_phase_source",
        lambda *args: (pd.DataFrame(), None),
    )
    destination = tmp_path / "events.png"
    request = RenderRequest(
        candidate_id="candidate-a",
        lc_path=lc_path,
        payload={"dip_best_t0": 9000.5},
        manual_phase_period_days=1.0,
        show_event_markers=True,
    )

    render_lightcurve_png(request, destination)

    assert destination.is_file()
    assert observed[0][0]["dip_best_t0"] == 9000.5
    assert observed[0][1] == 0.0
    assert observed[0][2] == pytest.approx(2459000.5)


def test_failed_render_does_not_replace_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "current.png"
    destination.write_bytes(b"previous-image")
    request = _request("missing", lc_path=tmp_path / "missing.dat3")

    with pytest.raises(FileNotFoundError, match="No local light curve"):
        render_lightcurve_png(request, destination)

    assert destination.read_bytes() == b"previous-image"


def test_coordinator_stale_completion_cannot_replace_new_current(tmp_path: Path) -> None:
    started_a = threading.Event()
    release_a = threading.Event()
    started_b = threading.Event()
    release_b = threading.Event()

    def controlled_renderer(request: RenderRequest, output_path) -> Path:
        if request.candidate_id == "a":
            started_a.set()
            assert release_a.wait(2.0)
        else:
            started_b.set()
            assert release_b.wait(2.0)
        output = Path(output_path)
        output.write_bytes(request.candidate_id.encode("ascii"))
        return output

    coordinator = ImageCoordinator(
        tmp_path / "images", viewer="none", cache_size=2, renderer=controlled_renderer
    )
    try:
        first = coordinator.request_current(_request("a"))
        assert first.state == "rendering"
        assert started_a.wait(2.0)

        second = coordinator.request_current(_request("b"))
        assert second.generation == first.generation + 1
        release_a.set()
        assert started_b.wait(2.0)

        # A has finished, but B is the current generation and is still blocked.
        assert coordinator.poll().state == "rendering"
        assert not coordinator.current_path.exists()

        release_b.set()

        def ready_for_b() -> bool:
            return coordinator.poll().state == "ready"

        _wait_until(ready_for_b)
        assert coordinator.current_path.read_bytes() == b"b"
        assert coordinator.status.candidate_id == "b"
    finally:
        release_a.set()
        release_b.set()
        coordinator.close()


def test_prefetch_is_reused_without_second_render(tmp_path: Path) -> None:
    finished = threading.Event()
    rendered = []

    def fake_renderer(request: RenderRequest, output_path) -> Path:
        rendered.append(request.candidate_id)
        output = Path(output_path)
        output.write_bytes(b"prefetched")
        finished.set()
        return output

    request = _request("next")
    coordinator = ImageCoordinator(tmp_path / "images", viewer="none", renderer=fake_renderer)
    try:
        assert coordinator.prefetch(request) is True
        assert finished.wait(2.0)
        coordinator.poll()

        status = coordinator.request_current(request)

        assert status.state == "ready"
        assert coordinator.current_path.read_bytes() == b"prefetched"
        assert rendered == ["next"]
    finally:
        coordinator.close()


def test_forced_period_search_token_bypasses_a_prior_forced_cache_key(
    tmp_path: Path,
) -> None:
    coordinator = ImageCoordinator(tmp_path / "images", viewer="none")
    try:
        first = RenderRequest(
            candidate_id="candidate-a",
            force_period_search=True,
            force_period_search_token="press-1",
        )
        second = RenderRequest(
            candidate_id="candidate-a",
            force_period_search=True,
            force_period_search_token="press-2",
        )

        assert coordinator._request_key(first) != coordinator._request_key(second)
    finally:
        coordinator.close()


def test_coordinator_opens_immutable_cache_path_not_mutable_current(
    tmp_path: Path, monkeypatch
) -> None:
    opened = []

    def fake_renderer(request: RenderRequest, output_path) -> Path:
        output = Path(output_path)
        output.write_bytes(b"candidate pixels")
        return output

    monkeypatch.setattr(
        "malca.review.tui_render.launch_viewer",
        lambda path, viewer: opened.append((Path(path), viewer)),
    )
    coordinator = ImageCoordinator(
        tmp_path / "images", viewer="quicklook", renderer=fake_renderer
    )
    try:
        coordinator.request_current(_request("candidate"))
        _wait_until(lambda: coordinator.poll().state == "ready")

        assert opened == [(next((tmp_path / "images").glob("render-*.png")), "quicklook")]
        assert opened[0][0] != coordinator.current_path
        assert coordinator.current_path.read_bytes() == b"candidate pixels"
    finally:
        coordinator.close()


def test_coordinator_replaces_owned_quicklook_and_closes_it_on_exit(
    tmp_path: Path, monkeypatch
) -> None:
    launched = []

    class FakeProcess:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.terminated = False
            self.killed = False
            self.wait_timeouts = []

        def poll(self):
            return 0 if self.terminated or self.killed else None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout):
            self.wait_timeouts.append(timeout)
            return 0

    def fake_launch(path, viewer):
        assert viewer == "quicklook"
        process = FakeProcess(Path(path))
        launched.append(process)
        return process

    def fake_renderer(request: RenderRequest, output_path) -> Path:
        output = Path(output_path)
        output.write_bytes(request.candidate_id.encode("ascii"))
        return output

    monkeypatch.setattr("malca.review.tui_render.launch_viewer", fake_launch)
    coordinator = ImageCoordinator(
        tmp_path / "images", viewer="quicklook", renderer=fake_renderer
    )
    try:
        coordinator.request_current(_request("first"))
        _wait_until(lambda: coordinator.poll().state == "ready")
        first = launched[0]
        assert first.terminated is False

        # A different period has a different cache key and replaces the same
        # candidate's currently owned Quick Look process.
        coordinator.request_current(
            _request("first", manual_phase_period_days=2.5)
        )
        _wait_until(lambda: coordinator.poll().state == "ready")
        second = launched[1]

        assert len(launched) == 2
        assert first.terminated is True
        assert first.wait_timeouts
        assert second.terminated is False
    finally:
        coordinator.close()

    assert second.terminated is True
    assert second.wait_timeouts


def test_persistent_viewer_updates_manifest_without_restarting_window(
    tmp_path: Path, monkeypatch
) -> None:
    launched = []

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, *, timeout):
            return 0

    process = FakeProcess()

    def fake_launch(path, viewer):
        launched.append((Path(path), viewer))
        return process

    def fake_renderer(request: RenderRequest, output_path) -> Path:
        output = Path(output_path)
        output.write_bytes(request.candidate_id.encode("ascii"))
        return output

    monkeypatch.setattr("malca.review.tui_render.launch_viewer", fake_launch)
    coordinator = ImageCoordinator(
        tmp_path / "images", viewer="window", renderer=fake_renderer
    )
    try:
        coordinator.request_current(_request("first"))
        _wait_until(lambda: coordinator.poll().state == "ready")
        first_manifest = json.loads(
            coordinator.viewer_manifest_path.read_text(encoding="utf-8")
        )

        coordinator.request_current(_request("second"))
        _wait_until(
            lambda: coordinator.poll().state == "ready"
            and coordinator.status.candidate_id == "second"
        )
        second_manifest = json.loads(
            coordinator.viewer_manifest_path.read_text(encoding="utf-8")
        )

        assert launched == [(coordinator.viewer_manifest_path, "window")]
        assert process.terminated is False
        assert first_manifest["path"] != second_manifest["path"]
        assert second_manifest["title"] == "MALCA Review"
    finally:
        coordinator.close()

    assert process.terminated is True


def test_coordinator_kills_only_owned_quicklook_if_graceful_close_times_out(
    tmp_path: Path, monkeypatch
) -> None:
    class StubbornProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, *, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("qlmanage", timeout)
            return 0

    process = StubbornProcess()
    monkeypatch.setattr(
        "malca.review.tui_render.launch_viewer", lambda path, viewer: process
    )
    coordinator = ImageCoordinator(tmp_path / "images", viewer="quicklook")
    coordinator._show_viewer(tmp_path / "image.png")

    coordinator.close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 2
def test_coordinator_ready_status_exposes_render_metadata(tmp_path: Path) -> None:
    def metadata_renderer(request: RenderRequest, output_path) -> Path:
        output = Path(output_path)
        output.write_bytes(b"candidate pixels")
        _write_render_metadata(
            output,
            {
                "phase_period_days": 0.31213094,
                "phase_source": "Auto PDM",
                "display_title": "ASAS-SN 123  ·  Gaia 456",
            },
        )
        return output

    coordinator = ImageCoordinator(
        tmp_path / "images", viewer="none", renderer=metadata_renderer
    )
    try:
        coordinator.request_current(_request("candidate"))
        _wait_until(lambda: coordinator.poll().state == "ready")

        assert coordinator.status.phase_period_days == pytest.approx(0.31213094)
        assert coordinator.status.phase_source == "Auto PDM"
        assert coordinator.status.survey_label is None
    finally:
        coordinator.close()


def test_rapid_navigation_cancels_superseded_queued_renders(tmp_path: Path) -> None:
    started_first = threading.Event()
    release_first = threading.Event()
    rendered = []

    def controlled_renderer(request: RenderRequest, output_path) -> Path:
        rendered.append(request.candidate_id)
        if request.candidate_id == "first":
            started_first.set()
            assert release_first.wait(2.0)
        output = Path(output_path)
        output.write_bytes(request.candidate_id.encode("ascii"))
        return output

    coordinator = ImageCoordinator(
        tmp_path / "images", viewer="none", renderer=controlled_renderer
    )
    try:
        coordinator.request_current(_request("first"))
        assert started_first.wait(2.0)
        coordinator.prefetch(_request("obsolete-prefetch"))
        coordinator.request_current(_request("obsolete-current"))
        coordinator.request_current(_request("latest"))
        release_first.set()

        def latest_ready() -> bool:
            return (
                coordinator.poll().state == "ready"
                and coordinator.status.candidate_id == "latest"
            )

        _wait_until(latest_ready)
        assert rendered == ["first", "latest"]
        assert coordinator.current_path.read_bytes() == b"latest"
    finally:
        release_first.set()
        coordinator.close()


def test_render_cache_is_reused_and_bounded_across_sessions(tmp_path: Path) -> None:
    rendered = []

    def fake_renderer(request: RenderRequest, output_path) -> Path:
        rendered.append(request.candidate_id)
        output = Path(output_path)
        output.write_bytes(request.candidate_id.encode("ascii"))
        return output

    cache_dir = tmp_path / "images"
    cache_dir.mkdir()
    # Simulate older artifacts from a prior process.  The newly rendered entry
    # below is the most recent one and must survive the next startup prune.
    (cache_dir / "render-old-one.png").write_bytes(b"old")
    (cache_dir / "render-old-two.png").write_bytes(b"old")
    first = ImageCoordinator(
        cache_dir, viewer="none", cache_size=2, renderer=fake_renderer
    )
    try:
        first.request_current(_request("cached"))
        _wait_until(lambda: first.poll().state == "ready")
        assert len(list(cache_dir.glob("render-*.png"))) <= 2
    finally:
        first.close()

    # Startup pruning retains only the configured number of render files.
    second = ImageCoordinator(
        cache_dir, viewer="none", cache_size=2, renderer=fake_renderer
    )
    try:
        status = second.request_current(_request("cached"))
        assert status.state == "ready"
        assert rendered == ["cached"]
        assert len(list(cache_dir.glob("render-*.png"))) <= 2
    finally:
        second.close()


def test_existing_non_lightcurve_source_path_does_not_override_bundle(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    lightcurve = bundle_dir / "123.dat3"
    lightcurve.write_text("light curve", encoding="ascii")
    candidate_table = run_dir / "candidates.parquet"
    candidate_table.write_bytes(b"not a light curve")

    request = RenderRequest(
        candidate_id="candidate-a",
        asas_sn_id="123",
        source_path=candidate_table,
        run_dir=run_dir,
    )

    assert resolve_lightcurve_path(request) == lightcurve.resolve()


def test_coordinator_reports_render_errors_and_rejects_calls_after_close(
    tmp_path: Path,
) -> None:
    def broken_renderer(request: RenderRequest, output_path) -> Path:
        raise ValueError("bad light curve")

    coordinator = ImageCoordinator(tmp_path / "images", viewer="none", renderer=broken_renderer)
    coordinator.request_current(_request("bad"))

    def failed() -> bool:
        return coordinator.poll().state == "error"

    _wait_until(failed)
    assert "bad light curve" in (coordinator.status.error or "")
    assert coordinator.current_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert list((tmp_path / "images").glob("error-*.png"))
    coordinator.close()
    assert coordinator.status.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.prefetch(_request("later"))


def test_native_tui_and_period_search_do_not_load_browser_or_plotly() -> None:
    code = """
import sys
from malca.review import period_search
from malca.review.tui_render import RenderRequest, resolve_lightcurve_path

assert period_search.resolve_stored_review_period({"period_consensus_days": 2.5})[0] == 2.5
assert resolve_lightcurve_path(RenderRequest(candidate_id="missing")) is None
forbidden = sorted(
    name for name in sys.modules
    if name == "plotly" or name.startswith("plotly.")
    or name == "malca.review.interactive_plot"
)
if forbidden:
    raise SystemExit("unexpected browser imports: " + ", ".join(forbidden))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_render_sed_panel_overlays_model_curve(tmp_path: Path) -> None:
    from contextlib import closing

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from malca.enrichment.sed_model import (
        SED_MODEL_CURVE_COLUMNS,
        SED_MODEL_FIT_COLUMNS,
        SED_MODEL_FIT_VERSION,
        upsert_sed_model_results,
    )
    from malca.review.sed import SED_COLUMNS, upsert_sed_rows
    from malca.review.store import db_connect
    from malca.review.tui_render import (
        RenderRequest,
        TUI_TICK_MAJOR_LEN,
        _render_sed_panel,
        tui_plot_theme,
    )

    db_path = tmp_path / "review.db"
    candidate_id = "cand-sed-model"
    sed_row = {col: None for col in SED_COLUMNS}
    sed_row.update(
        {
            "candidate_id": candidate_id,
            "source": "2MASS",
            "band": "J",
            "mag": 10.0,
            "mag_system": "Vega",
            "lambda_eff_angstrom": 12350.0,
        }
    )
    fit = {col: None for col in SED_MODEL_FIT_COLUMNS}
    fit.update(
        {
            "candidate_id": candidate_id,
            "model_family": "Castelli/Kurucz 2004",
            "fit_version": SED_MODEL_FIT_VERSION,
            "teff_k": 4200.0,
            "status": "ok",
            "n_fit_points": 3,
        }
    )
    curves = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "model_family": "Castelli/Kurucz 2004",
                "fit_version": SED_MODEL_FIT_VERSION,
                "wavelength_angstrom": wave,
                "lambda_l_lambda_observed": value,
                "flux_lambda_observed": value * 1.0e-45,
                "teff_k": 4200.0,
            }
            for wave, value in [(4000.0, 5.0e32), (7000.0, 8.0e32), (12000.0, 6.0e32)]
        ],
        columns=SED_MODEL_CURVE_COLUMNS,
    )

    with closing(db_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            (candidate_id, '{"distance_gspphot": 100.0}', "2026-07-23T00:00:00"),
        )
        upsert_sed_rows(conn, pd.DataFrame([sed_row]))
        upsert_sed_model_results(conn, pd.DataFrame([fit]), curves)

    request = RenderRequest(
        candidate_id=candidate_id,
        asas_sn_id=None,
        db_path=db_path,
        payload={"candidate_id": candidate_id, "distance_gspphot": 100.0},
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    _render_sed_panel(ax, request, np, theme=tui_plot_theme("white"))
    fig.canvas.draw()
    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    line_labels = [
        line.get_label()
        for line in ax.get_lines()
        if line.get_label() and not line.get_label().startswith("_")
    ]
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    tick_clearance_px = TUI_TICK_MAJOR_LEN * fig.dpi / 72.0
    assert legend_box.x0 >= axes_box.x0 + tick_clearance_px
    assert legend_box.y0 >= axes_box.y0 + tick_clearance_px
    assert all(text.get_fontsize() == pytest.approx(8.0) for text in legend.get_texts())
    assert legend.get_frame().get_alpha() == pytest.approx(1.0)
    assert ax.yaxis.get_major_formatter()(10.0, 0) == "10"
    assert ax.yaxis.get_minor_formatter()(20.0, 0) == "20"
    plt.close(fig)

    assert not any("Castelli/Kurucz" in label for label in labels)
    assert any("Castelli/Kurucz" in label for label in line_labels)
    assert len(ax.get_lines()) >= 1
    ylabel = ax.yaxis.get_label().get_text()
    assert ylabel
    assert ax.yaxis.get_label_position() == "left"


def test_tui_matplotlib_draws_legacy_model_curve() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    from malca.review.sed import LSUN_ERG_S, _draw_sed_model_matplotlib

    fit_rows = pd.DataFrame(
        [{"fit_version": None, "status": "ok", "teff_k": 5000.0}]
    )
    curve_rows = pd.DataFrame(
        {
            "wavelength_angstrom": [4000.0, 10000.0, 200000.0],
            "lambda_l_lambda": [
                2.0 * LSUN_ERG_S,
                1.0 * LSUN_ERG_S,
                0.01 * LSUN_ERG_S,
            ],
            "teff_k": [5000.0, 5000.0, 5000.0],
        }
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        _draw_sed_model_matplotlib(
            ax,
            model_curve_rows=curve_rows,
            model_fit_rows=fit_rows,
            y_col="lambda_l_lambda",
            mode="observed",
            intrinsic_ratio_complete=False,
            theme="white",
        )
        assert len(ax.get_lines()) == 1
        assert ax.get_lines()[0].get_xdata().tolist() == [4000.0, 10000.0, 200000.0]
    finally:
        plt.close(fig)


def test_tui_uses_corrected_photometry_for_legacy_fit(monkeypatch) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from malca.review.tui_render import (
        RenderRequest,
        _render_sed_panel,
        tui_plot_theme,
    )

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "malca.review.tui_render._load_tui_sed_context",
        lambda _request: (
            {"candidate_id": "legacy"},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([{"fit_version": None, "status": "ok"}]),
            pd.DataFrame(),
        ),
    )

    def fake_render(_ax, _payload, **kwargs) -> None:
        seen.update(kwargs)

    monkeypatch.setattr("malca.review.sed.render_sed_matplotlib", fake_render)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        _render_sed_panel(
            ax,
            RenderRequest(candidate_id="legacy"),
            np,
            theme=tui_plot_theme("white"),
        )
    finally:
        plt.close(fig)

    assert seen["extinction_mode"] == "corrected"


def test_tui_external_sources_add_only_selected_legacy_surveys() -> None:
    from malca.review.tui_render import TUI_EXTERNAL_LC_SOURCES

    assert TUI_EXTERNAL_LC_SOURCES == (
        "atlas",
        "ztf",
        "neowise",
        "asas3",
        "crts",
        "dasch",
    )


def test_tui_external_source_choices_cover_supported_magnitude_surveys_only() -> None:
    from malca.review.tui_photometry import (
        TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES,
    )

    assert {
        "atlas",
        "ztf",
        "gaia_epoch",
        "neowise",
        "allwise_mep",
        "aavso",
        "vvvx_virac",
        "ps1",
        "asas3",
        "crts",
        "dasch",
    } <= set(TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES)
    assert "asassn" not in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES
    assert "tess" not in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES
    assert "kepler" not in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES


def test_tui_external_overlay_requests_only_configured_sources(monkeypatch) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from malca.review.tui_render import (
        RenderRequest,
        _overlay_tui_external_lightcurves,
    )

    requested: list[str] = []

    def fake_discover(
        candidate_id,
        payload,
        lc_dir,
        sources,
        default_results_root=None,
    ):
        requested.extend(sources)
        return {}

    monkeypatch.setattr(
        "malca.review.lightcurve_sources.discover_external_lcs",
        fake_discover,
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        drawn = _overlay_tui_external_lightcurves(
            ax,
            RenderRequest(
                candidate_id="candidate-selected-sources",
                external_lightcurve_sources=("ps1", "unknown", "ztf", "ps1"),
            ),
            Path("dummy.dat"),
            asas_median=12.0,
            jd_offset=0.0,
            np=np,
        )
    finally:
        plt.close(fig)

    assert drawn == []
    assert requested == ["ztf", "ps1"]


def test_asassn_time_window_uses_finite_span_and_padding() -> None:
    import numpy as np

    from malca.review.tui_render import _asassn_jd_window

    assert _asassn_jd_window(
        np.array([np.nan, 2459000.0, 2459100.0]),
        np,
        padding_days=30.0,
    ) == (2458970.0, 2459130.0)
    assert _asassn_jd_window(
        np.array([2459000.0]),
        np,
        padding_days=0.0,
    ) == (2458999.5, 2459000.5)


def test_asassn_cadence_window_ignores_intra_visit_measurements() -> None:
    import numpy as np

    from malca.review.tui_render import _asassn_cadence_window_days

    jd = np.array(
        [2459000.0, 2459000.1, 2459001.0, 2459002.0, 2459004.0, np.nan]
    )

    assert _asassn_cadence_window_days(jd, np) == pytest.approx(1.0)
    assert _asassn_cadence_window_days([2459000.0], np) == pytest.approx(1.0)


def test_external_cadence_bins_use_inverse_variance_propagation() -> None:
    import numpy as np

    from malca.review.tui_render import _combine_external_magnitude_cadence_bins

    times, magnitudes, errors, counts = _combine_external_magnitude_cadence_bins(
        [0.0, 0.1, 0.9, 1.8],
        [10.0, 11.0, 12.0, 13.0],
        [1.0, 2.0, 1.0, 0.5],
        window_days=1.0,
        np=np,
    )

    assert counts.tolist() == [3, 1]
    assert times == pytest.approx([0.925 / 2.25, 1.8])
    assert magnitudes == pytest.approx([24.75 / 2.25, 13.0])
    assert errors == pytest.approx([1.0 / np.sqrt(2.25), 0.5])


def test_external_cadence_bins_do_not_chain_across_multiple_windows() -> None:
    import numpy as np

    from malca.review.tui_render import _combine_external_magnitude_cadence_bins

    _times, _magnitudes, _errors, counts = (
        _combine_external_magnitude_cadence_bins(
            [0.0, 0.9, 1.8],
            [10.0, 10.1, 10.2],
            [0.1, 0.1, 0.1],
            window_days=1.0,
            np=np,
        )
    )

    assert counts.tolist() == [2, 1]


def test_tui_overlays_asas3_crts_and_dasch(monkeypatch) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from malca.review.tui_render import (
        RenderRequest,
        TUI_EXTERNAL_LC_SOURCES,
        _overlay_tui_external_lightcurves,
    )

    frames = {
        "asas3": pd.DataFrame(
            {
                "hjd": [2453000.0, 2453000.2, 2453003.0],
                "mag": [12.0, 12.1, 12.2],
                "mag_err": [0.05, 0.05, 0.06],
            }
        ),
        "crts": pd.DataFrame(
            {
                "mjd": [55000.0, 55000.01],
                "mag": [13.0, 13.1],
                "mag_err": [0.08, 0.08],
            }
        ),
        "dasch": pd.DataFrame(
            {
                "hjd": [2420000.0, 2420100.0],
                "mag": [11.5, 11.7],
                "mag_err": [0.12, 0.15],
            }
        ),
    }
    requested = []

    def fake_discover(
        candidate_id,
        payload,
        lc_dir,
        sources,
        default_results_root=None,
    ):
        requested.extend(sources)
        return {source: Path(f"{source}.parquet") for source in frames}

    def fake_load(source_name, path):
        return frames[source_name]

    from malca.review import lightcurve_sources

    monkeypatch.setattr(
        lightcurve_sources,
        "discover_external_lcs",
        fake_discover,
    )
    monkeypatch.setattr(
        lightcurve_sources,
        "load_external_lc_frame",
        fake_load,
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        drawn = _overlay_tui_external_lightcurves(
            ax,
            RenderRequest(candidate_id="cand-legacy"),
            Path("dummy.dat"),
            asas_median=12.0,
            jd_offset=2400000.0,
            np=np,
            cadence_window_days=1.0,
        )
        plotted_point_counts = {
            collection.get_label(): len(collection.get_offsets())
            for collection in ax.collections
            if collection.get_label() in {"ASAS-3 V", "CRTS CV", "DASCH"}
        }
        full_requested = tuple(requested)
        requested.clear()
        clipped = _overlay_tui_external_lightcurves(
            ax,
            RenderRequest(candidate_id="cand-legacy"),
            Path("dummy.dat"),
            asas_median=12.0,
            jd_offset=2400000.0,
            np=np,
            jd_window=(2452999.0, 2453004.0),
            cadence_window_days=1.0,
        )
    finally:
        plt.close(fig)

    assert full_requested == TUI_EXTERNAL_LC_SOURCES
    assert requested == list(TUI_EXTERNAL_LC_SOURCES)
    assert {label for label, _, _ in drawn} == {
        "ASAS-3 V",
        "CRTS CV",
        "DASCH",
    }
    assert {marker for _, _, marker in drawn} == {"D", "s", "+"}
    assert {label for label, _, _ in clipped} == {"ASAS-3 V"}
    assert plotted_point_counts == {"ASAS-3 V": 2, "CRTS CV": 1, "DASCH": 2}


def test_external_neowise_overlay_does_not_warn_on_x_markers() -> None:
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from malca.review.tui_render import RenderRequest, _overlay_tui_external_lightcurves

    fig, ax = plt.subplots(figsize=(4, 3))
    request = RenderRequest(candidate_id="cand-neowise")
    neowise = pd.DataFrame(
        {
            "mjd": [59000.0, 59010.0],
            "w1mpro": [12.0, 12.1],
            "w1sigmpro": [0.05, 0.05],
            "w2mpro": [11.5, 11.6],
            "w2sigmpro": [0.05, 0.05],
        }
    )

    def fake_discover(candidate_id, payload, lc_dir, sources, default_results_root=None):
        return {"neowise": "dummy.parquet"}

    def fake_load(source_name, path):
        return neowise

    from malca.review import lightcurve_sources

    original_discover = lightcurve_sources.discover_external_lcs
    original_load = lightcurve_sources.load_external_lc_frame
    lightcurve_sources.discover_external_lcs = fake_discover
    lightcurve_sources.load_external_lc_frame = fake_load
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawn = _overlay_tui_external_lightcurves(
                ax,
                request,
                Path("dummy.dat"),
                asas_median=12.0,
                jd_offset=0.0,
                np=np,
            )
        assert len(drawn) == 2
        assert len(ax.containers) >= 2
        assert not any("edgecolor" in str(item.message).lower() for item in caught)
    finally:
        lightcurve_sources.discover_external_lcs = original_discover
        lightcurve_sources.load_external_lc_frame = original_load
        plt.close(fig)
