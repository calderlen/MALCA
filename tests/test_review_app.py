from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.review.filter_schema import (
    SIDEBAR_GROUPS,
    VETTING_KNOWN_BOOL_FILTERS,
    VETTING_KNOWN_SELECT_FILTERS,
)
from malca.review.interactive_plot import _build_stat_rows
from malca.review.session import create_queue_data_dict
from malca.review import app as review_app
from malca.review.store import (
    SQLITE_BUSY_TIMEOUT_MS,
    _BOOL_COLS,
    _COL_NAMES,
    _FLOAT_COLS,
    db_connect,
    import_candidates,
    query_queue,
    save_app_state,
    save_review,
)


def _write_skypatrol_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "JD": [2459000.1, 2459001.1, 2459002.1, 2459003.1],
            "Flux": [1000.0, 1012.0, 990.0, 995.0],
            "Flux Error": [8.0, 8.2, 8.4, 8.1],
            "Mag": [14.0, 13.95, 14.12, 14.08],
            "Mag Error": [0.03, 0.03, 0.04, 0.04],
            "Limit": [99.0, 99.0, 99.0, 99.0],
            "FWHM": [2.5, 2.7, 2.6, 2.8],
            "Filter": ["g", "V", "g", "V"],
            "Quality": ["G", "G", "G", "G"],
            "Camera": ["camA", "camA", "camB", "camB"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_band_df(jd: np.ndarray, true_period: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phase = np.mod((jd - jd.min()) / true_period, 1.0)
    # Asymmetric folded shape to distinguish P from P/2 harmonics.
    resid = 0.20 * np.exp(-0.5 * ((phase - 0.35) / 0.06) ** 2)
    resid += 0.05 * np.exp(-0.5 * ((phase - 0.62) / 0.03) ** 2)
    resid += rng.normal(0.0, 0.01, size=jd.size)
    return pd.DataFrame({"JD": jd, "resid": resid})


def test_plot_url_and_route_work_without_plot_dir(tmp_path: Path, monkeypatch) -> None:
    plot_file = tmp_path / "output" / "runs" / "demo" / "plots" / "candidate.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plot_file.write_bytes(b"png-bytes")

    monkeypatch.setattr(review_app, "PLOT_DIR", None)
    monkeypatch.setattr(review_app, "_project_root", lambda: tmp_path)

    assert review_app._plot_url_for_path(plot_file) == "/plots/output/runs/demo/plots/candidate.png"

    client = review_app.app.server.test_client()
    response = client.get("/plots/output/runs/demo/plots/candidate.png")

    assert response.status_code == 200
    assert response.data == b"png-bytes"


def test_db_connect_configures_busy_timeout_and_wal(tmp_path: Path) -> None:
    conn = db_connect(tmp_path / "review.db")
    try:
        busy_timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()

    assert busy_timeout >= SQLITE_BUSY_TIMEOUT_MS
    assert journal_mode == "wal"


def test_review_gui_state_normalization_roundtrip() -> None:
    state = review_app._normalize_review_gui_state(
        {
            "theme_mode": "white",
            "plot_mode": "png",
            "plot_overlays": ["raw", "markers", "invalid"],
            "baseline_opacity": 0.4,
            "residual_height": 0.33,
            "external_source_view": "atlas",
            "camera_values": ["ba", "bc"],
            "band_values": ["V"],
            "yaxis_mode": "flux",
            "period_method": "ce",
            "pdm_min_period": 0.25,
            "pdm_max_period": 50.0,
            "pdm_manual_period": 12.5,
        }
    )

    assert state is not None
    assert state["theme_mode"] == "white"
    assert state["plot_mode"] == "png"
    assert state["plot_overlays"] == ["raw", "markers"]
    assert state["baseline_opacity"] == 0.4
    assert state["residual_height"] == 0.33
    assert state["external_source_view"] == "atlas"
    assert state["camera_values"] == ["ba", "bc"]
    assert state["band_values"] == ["V"]
    assert state["yaxis_mode"] == "flux"
    assert state["period_method"] == "ce"
    assert state["pdm_min_period"] == 0.25
    assert state["pdm_max_period"] == 50.0
    assert state["pdm_manual_period"] == 12.5


def test_review_db_for_plot_dir_prefers_run_local_db(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    plot_dir = run_dir / "plots"
    db_path = run_dir / "review" / "review.db"
    plot_dir.mkdir(parents=True)
    with db_connect(db_path):
        pass

    assert review_app._review_db_for_plot_dir(str(plot_dir)) == db_path.resolve()
    assert review_app._resolve_run_dir_from_db_path(db_path) == run_dir.resolve()


def test_resolve_run_dir_from_standalone_bundled_db(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "ltv" / "ltv"
    db_path = run_dir / "ltv_candidates.db"
    (run_dir / "bundle_assets" / "lightcurves").mkdir(parents=True)
    with db_connect(db_path):
        pass

    assert review_app._resolve_run_dir_from_db_path(db_path) == run_dir.resolve()


def test_effective_local_lc_path_uses_bundle_next_to_standalone_db(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "output" / "ltv" / "ltv"
    db_path = run_dir / "ltv_candidates.db"
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    bundled_lc = bundle_dir / "123.dat2"
    bundled_lc.write_text("2450000.0 14.0 0.1 1 cam 0 0 cf\n")

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "PLOT_DIR", None)

    resolved = review_app._effective_local_lc_path(
        {"candidate_id": "123", "asas_sn_id": "123"},
        stored_lc_path="/data/cluster/lightcurves/123.dat2",
    )

    assert resolved == str(bundled_lc)


def test_db_plot_mismatch_warning_points_to_populated_run_db(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True)
    run_db = run_dir / "review" / "review.db"
    empty_db = tmp_path / "output" / "review" / "review.db"

    with db_connect(run_db) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "RUN-1", "asas_sn_id": "RUN-1"}]),
            source_path=str(run_dir),
            characterize_before_import=False,
            vet_before_import=False,
        )
    with db_connect(empty_db):
        pass

    warning = review_app._db_plot_mismatch_warning(empty_db, str(plot_dir))

    assert str(empty_db.resolve()) in warning
    assert str(run_db.resolve()) in warning
    assert "1 candidates" in warning


def test_update_queue_source_scope_skips_source_filter_for_run_local_db(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    plot_dir = run_dir / "plots"
    db_path = run_dir / "review" / "review.db"
    plot_dir.mkdir(parents=True)
    with db_connect(db_path):
        pass

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "PLOT_DIR", str(plot_dir))

    assert review_app.update_queue_source_scope(0, None) == ""


def test_queue_plot_render_request_tracks_candidate_id() -> None:
    request = review_app.queue_plot_render_request(
        idx=0,
        current_candidate_id="FETCH-TEST-1",
        plot_mode="native",
        overlay_values=[],
        selected_cameras=[],
        preset="Diagnostics",
        residual_height=review_app.DEFAULT_RESIDUAL_FRACTION,
        theme_mode=review_app.DEFAULT_THEME,
        _queue_size=1,
        _pipeline_progress=0,
        baseline_opacity=0.5,
        selected_bands=["g", "V"],
        round_sigfigs=["yes"],
        link_radius=10.0,
        pdm_result=None,
        pdm_manual_period=None,
        yaxis_mode="mag",
        external_source_view="all",
        existing_request={"nonce": 4, "ts": 0.0, "state": {}},
    )

    assert request["nonce"] == 5
    assert request["state"]["candidate_id"] == "FETCH-TEST-1"


def test_keyboard_refresh_queue_handles_r() -> None:
    assert review_app.keyboard_refresh_queue("r\t1234567890", 2) == 3
    assert review_app.keyboard_refresh_queue("R\t1234567890", 2) == 3

    with pytest.raises(review_app.dash.exceptions.PreventUpdate):
        review_app.keyboard_refresh_queue("Shift+R\t1234567890", 2)


def test_reset_numeric_filters_clears_all_numeric_inputs() -> None:
    result = review_app.reset_numeric_filters(1)

    assert result[-1] == "Reset numeric filters to the current queue bounds."
    assert all(value is None for value in result[:-1])
    assert len(result) == len(review_app._NUM_INPUT_STATES) + 1


def test_normalized_queue_index_clamps_to_visible_queue() -> None:
    queue_data = {"candidate_ids": ["Q-1", "Q-2", "Q-3"], "queue_size": 3}

    assert review_app._normalized_queue_index(queue_data, 99) == 2
    assert review_app._normalized_queue_index(queue_data, -5) == 0
    assert review_app._normalized_queue_index(queue_data, 1) is None


def test_preload_next_candidates_skips_non_native_mode() -> None:
    queue_data = {"candidate_ids": ["Q-1", "Q-2"], "queue_size": 2}

    assert review_app.preload_next_candidates(0, queue_data, "png") is review_app.no_update


def test_auto_period_on_navigate_queues_search_when_no_known_period(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_candidate_context", lambda candidate_id: ({"candidate_id": candidate_id}, None, None))
    monkeypatch.setattr(review_app, "_has_external_period", lambda payload: False)

    result, label, manual_period, cache_update, request = review_app.auto_period_on_navigate(
        "FETCH-TEST-1",
        0.1,
        10.0,
        {},
        {"nonce": 0},
    )

    assert result is None
    assert label == "Auto-searching period..."
    assert manual_period is None
    assert cache_update is review_app.no_update
    assert request["candidate_id"] == "FETCH-TEST-1"
    assert request["method"] == "auto"
    assert request["nonce"] == 1


def test_run_auto_period_search_caches_result(monkeypatch) -> None:
    monkeypatch.setattr(review_app, "_candidate_context", lambda candidate_id: ({"candidate_id": candidate_id}, None, None))
    monkeypatch.setattr(
        review_app,
        "_run_period_search_for_payload",
        lambda payload, min_period, max_period, method: (
            {"best_period": 2.5, "method": "CE"},
            "Auto CE/PDM: P=2.50000 d via CE",
        ),
    )

    result, label, cache = review_app.run_auto_period_search(
        {"nonce": 1, "candidate_id": "FETCH-TEST-1", "min_period": 0.1, "max_period": 10.0, "method": "auto"},
        {},
    )

    assert result["best_period"] == 2.5
    assert result["auto"] is True
    assert label == "Auto CE/PDM: P=2.50000 d via CE"
    assert cache["FETCH-TEST-1"]["result"]["best_period"] == 2.5


def test_vetting_known_filter_preset_targets_only_definite_known_types() -> None:
    select_options = {
        "vsx_class": [{"label": "EA", "value": "EA"}, {"label": "RR", "value": "RR"}],
        "microlens_catalog": [{"label": "OGLE", "value": "OGLE"}],
        "asassn_var_type": [{"label": "Mira", "value": "Mira"}],
        "gaia_var_class": [{"label": "RRAB", "value": "RRAB"}],
        "simbad_otype": [{"label": "V*", "value": "V*"}],
        "ztf_var_type": [{"label": "CEP", "value": "CEP"}],
        "tns_type": [{"label": "SN Ia", "value": "SN Ia"}],
        "alerce_lc_class": [{"label": "LPV", "value": "LPV"}],
        "yso_class": [{"label": "Class II", "value": "Class II"}],
    }

    bool_values, select_values = review_app._vetting_known_filter_preset(select_options)
    select_map = dict(zip(VETTING_KNOWN_SELECT_FILTERS, select_values))
    targeted = set(VETTING_KNOWN_BOOL_FILTERS) | set(VETTING_KNOWN_SELECT_FILTERS)

    assert bool_values == ["False", "False"]
    assert select_map["vsx_class"] == ["EA", "RR"]
    assert select_map["alerce_lc_class"] == ["LPV"]
    assert "yso_class" not in select_map
    assert {"pm_cluster_offset_sigma", "pm_total", "high_pm_flag", "yso_class"}.isdisjoint(targeted)


def test_layout_includes_vetting_known_type_button() -> None:
    def collect_ids(component) -> set[str]:
        if component is None:
            return set()
        ids: set[str] = set()
        comp_id = getattr(component, "id", None)
        if isinstance(comp_id, str):
            ids.add(comp_id)
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                ids |= collect_ids(child)
        elif children is not None:
            ids |= collect_ids(children)
        return ids

    layout = review_app.create_layout()

    assert "vetting-known-types-btn" in collect_ids(layout)


def test_load_review_progress_state_reads_db_counts(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "CAND-1", "asas_sn_id": "ASASSN-1"},
                    {"candidate_id": "CAND-2", "asas_sn_id": "ASASSN-2"},
                ]
            ),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        save_review(
            conn,
            candidate_id="CAND-1",
            interest_score=3,
            event_class="other",
            review_pass=1,
            notes="",
            status="reviewed",
            reviewer="tester",
        )

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    review_app._clear_review_state_caches()

    state = review_app.load_review_progress_state(None, None, None, None)

    assert state == {"reviewed": 1, "total": 2}


def test_review_callbacks_have_no_duplicate_outputs_without_allow_duplicate() -> None:
    plain_counts: dict[str, int] = {}

    for callback in review_app.app._callback_list:
        output_key = str(callback["output"])
        if output_key.startswith("..") and output_key.endswith(".."):
            parts = output_key[2:-2].split("...")
        else:
            parts = [output_key]
        for part in parts:
            base, _, duplicate_hash = part.partition("@")
            if duplicate_hash:
                continue
            plain_counts[base] = plain_counts.get(base, 0) + 1

    offenders = {key: count for key, count in plain_counts.items() if count > 1}
    assert offenders == {}, f"Duplicate non-allow_duplicate callback outputs: {offenders}"


def test_update_display_uses_render_request_candidate_and_skips_static_lookup(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    lc_path = tmp_path / "FETCH-TEST-1.csv"
    _write_skypatrol_csv(lc_path)

    conn = db_connect(db_path)
    try:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "FETCH-TEST-1",
                        "asas_sn_id": "FETCH-TEST-1",
                        "lc_path": str(lc_path),
                        "path": str(lc_path),
                    }
                ]
            ),
            source_path="fetch://skypatrol2/asassn/FETCH-TEST-1",
            characterize_before_import=False,
            vet_before_import=False,
        )
    finally:
        conn.close()

    captured: dict[str, object] = {}

    def fake_build_interactive_lightcurve_figure(payload: dict, **kwargs) -> dict:
        captured["candidate_id"] = payload.get("candidate_id")
        captured["plot_dir"] = kwargs.get("plot_dir")
        return {
            "figure": {"data": [{"type": "scatter", "x": [1.0], "y": [2.0]}], "layout": {}},
            "camera_options": [{"label": "camA", "value": "camA"}],
            "camera_values": ["camA"],
            "stat_rows": [],
            "status": "ok",
            "status_message": "",
            "camera_diagnostics": {},
            "warnings": [],
        }

    def fail_candidate_plot_src(_payload: dict | None) -> str:
        raise AssertionError("native render should not search for static plot files")

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "PLOT_DIR", None)
    monkeypatch.setattr(review_app, "build_interactive_lightcurve_figure", fake_build_interactive_lightcurve_figure)
    monkeypatch.setattr(review_app, "_candidate_plot_src", fail_candidate_plot_src)
    monkeypatch.setattr(review_app, "_index_external_lc_paths_from_root", lambda *_args, **_kwargs: {})

    render_request = {
        "nonce": 7,
        "ts": 0.0,
        "state": {
            "idx": 0,
            "candidate_id": "FETCH-TEST-1",
            "plot_mode": "native",
            "overlay_values": [],
            "selected_cameras": [],
            "preset": "Diagnostics",
            "theme": review_app.DEFAULT_THEME,
            "residual_height": review_app.DEFAULT_RESIDUAL_FRACTION,
            "baseline_opacity": 0.5,
            "external_source_view": "all",
        },
    }

    out = review_app.update_display(render_request, 0, None, 1)

    assert captured["candidate_id"] == "FETCH-TEST-1"
    assert captured["plot_dir"] is None
    assert out[5]["data"]
    assert out[6]["display"] == "block"
    assert out[15] == 7


def test_update_header_key_info_shows_cluster_and_local_lc_paths(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    cluster_lc_path = "/cluster/lightcurves/FETCH-TEST-1.dat2"
    local_lc_path = str(tmp_path / "FETCH-TEST-1.csv")

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "FETCH-TEST-1",
                        "asas_sn_id": "FETCH-TEST-1",
                        "path": cluster_lc_path,
                        "lc_path": local_lc_path,
                    }
                ]
            ),
            source_path="cluster://demo",
            characterize_before_import=False,
            vet_before_import=False,
        )

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    asas_text, gaia_text, bottom_bar = review_app.update_header_key_info(
        "FETCH-TEST-1", 1, "", None, ""
    )

    bottom_items = {
        str(item.children[0].children).rstrip(":"): str(item.children[1].title)
        for item in bottom_bar.children
    }

    assert asas_text == "ASAS-SN ID: FETCH-TEST-1"
    assert gaia_text == "Gaia ID: -"
    assert bottom_items["Cluster LC"] == cluster_lc_path
    assert bottom_items["Local LC"] == local_lc_path
    assert bottom_items["DB"] == str(db_path)


def test_update_header_key_info_treats_unresolved_ltv_lc_path_as_cluster_path(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    raw_lc_path = "/data/cluster/lightcurves/LTV-1.dat2"

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "LTV-1",
                        "asas_sn_id": "LTV-1",
                        "lc_path": raw_lc_path,
                    }
                ]
            ),
            source_path="ltv_candidates.db",
            characterize_before_import=False,
            vet_before_import=False,
        )

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    asas_text, gaia_text, bottom_bar = review_app.update_header_key_info(
        "LTV-1", 1, "", None, ""
    )

    bottom_items = {
        str(item.children[0].children).rstrip(":"): str(item.children[1].title)
        for item in bottom_bar.children
    }

    assert asas_text == "ASAS-SN ID: LTV-1"
    assert gaia_text == "Gaia ID: -"
    assert bottom_items["Cluster LC"] == raw_lc_path
    assert bottom_items["Local LC"] == "-"


def test_baseline_provenance_warning_flags_imported_skypatrol(tmp_path: Path) -> None:
    lc_path = tmp_path / "FETCH-BASELINE.csv"
    _write_skypatrol_csv(lc_path)

    warning = review_app._baseline_provenance_warning(
        {"path": str(lc_path), "lc_path": str(lc_path)},
        plot_dir=None,
        run_params={"baseline_func": "gp"},
        stored_lc_path=str(lc_path),
        source_path="fetch://skypatrol2/asassn/FETCH-BASELINE",
    )

    assert warning is not None
    assert "imported SkyPatrol" in warning


def test_update_diagnostic_plots_renders_candidate_first_without_background(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "DIAG-1", "asas_sn_id": "DIAG-1"}]),
            source_path="fetch://skypatrol2/asassn/DIAG-1",
            characterize_before_import=False,
            vet_before_import=False,
        )

    captured: dict[str, object] = {}

    def fake_render(payload: dict, theme: str, background=None):
        captured["payload"] = dict(payload)
        captured["background"] = background
        return ["candidate-only"]

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", fake_render)

    panels, status = review_app.update_diagnostic_plots(
        True,
        "DIAG-1",
        review_app.DEFAULT_THEME,
        {"signature": "", "ready": False, "cached": False, "token": 0},
    )

    assert panels == ["candidate-only"]
    assert captured["background"] is None
    assert status.startswith("Population background loading")


def test_update_diagnostic_plots_uses_cached_background_when_ready(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "DIAG-2", "asas_sn_id": "DIAG-2"}]),
            source_path="fetch://skypatrol2/asassn/DIAG-2",
            characterize_before_import=False,
            vet_before_import=False,
        )

    captured: dict[str, object] = {}
    cached_background = {"plane_periodicity": [1.0, 2.0]}

    def fake_render(payload: dict, theme: str, background=None):
        captured["background"] = background
        return ["with-background"]

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", fake_render)

    signature = review_app._diagnostic_background_signature(str(db_path))
    review_app._store_cached_diagnostic_background(signature, cached_background)

    panels, status = review_app.update_diagnostic_plots(
        True,
        "DIAG-2",
        review_app.DEFAULT_THEME,
        {"signature": signature, "ready": True, "cached": True, "token": 1},
    )

    assert panels == ["with-background"]
    assert captured["background"] == cached_background
    assert status == ""


def test_update_diagnostic_plots_skips_work_when_panel_closed(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "DIAG-3", "asas_sn_id": "DIAG-3"}]),
            source_path="fetch://skypatrol2/asassn/DIAG-3",
            characterize_before_import=False,
            vet_before_import=False,
        )

    def fail_render(*_args, **_kwargs):
        raise AssertionError("diagnostic plots should not render while the panel is closed")

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "_render_diagnostic_plots", fail_render)

    panels, status = review_app.update_diagnostic_plots(
        False,
        "DIAG-3",
        review_app.DEFAULT_THEME,
        {"signature": "", "ready": False, "cached": False, "token": 0},
    )

    assert panels is review_app.no_update
    assert status == ""


def test_open_existing_candidate_matches_local_lc_stem(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    local_lc_path = tmp_path / "lightcurves" / "LC-SEARCH-1.csv"
    cluster_lc_path = "/cluster/bundle/LC-SEARCH-1.dat2"

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "SEARCH-CAND-1",
                        "asas_sn_id": "ASASSN-SEARCH-1",
                        "path": cluster_lc_path,
                        "lc_path": str(local_lc_path),
                    }
                ]
            ),
            source_path="cluster://search",
            characterize_before_import=False,
            vet_before_import=False,
        )

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    queue_data, current_index, notice = review_app.open_existing_candidate(1, None, "LC-SEARCH-1", {"candidate_ids": []})

    assert queue_data["candidate_ids"] == ["SEARCH-CAND-1"]
    assert current_index == 0
    assert "local LC stem" in notice


def test_restore_startup_candidate_reopens_last_candidate_in_queue(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([
                {"candidate_id": "CAND-1", "asas_sn_id": "CAND-1"},
                {"candidate_id": "CAND-2", "asas_sn_id": "CAND-2"},
            ]),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        save_app_state(conn, "review_last_candidate", "CAND-2")

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "INITIAL_CANDIDATE_QUERY", None)

    queue_data, current_index, notice, applied = review_app.restore_startup_candidate(
        {"candidate_ids": ["CAND-1", "CAND-2"], "queue_size": 2, "filter_hash": "demo"},
        False,
    )

    assert queue_data is review_app.no_update
    assert current_index == 1
    assert "Restored last candidate CAND-2" in notice
    assert applied is True


def test_restore_startup_candidate_waits_for_nonempty_queue(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([
                {"candidate_id": "CAND-1", "asas_sn_id": "CAND-1"},
                {"candidate_id": "CAND-2", "asas_sn_id": "CAND-2"},
            ]),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        save_app_state(conn, "review_last_candidate", "CAND-2")

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "INITIAL_CANDIDATE_QUERY", None)

    queue_data, current_index, notice, applied = review_app.restore_startup_candidate(
        {"candidate_ids": [], "queue_size": 0, "filter_hash": "transient-empty"},
        False,
    )

    assert queue_data is review_app.no_update
    assert current_index is review_app.no_update
    assert notice is review_app.no_update
    assert applied is review_app.no_update

    queue_data, current_index, notice, applied = review_app.restore_startup_candidate(
        {"candidate_ids": ["CAND-1", "CAND-2"], "queue_size": 2, "filter_hash": "ready"},
        False,
    )

    assert queue_data is review_app.no_update
    assert current_index == 1
    assert "Restored last candidate CAND-2" in notice
    assert applied is True


def test_persist_last_candidate_waits_for_startup_restore(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([
                {"candidate_id": "CAND-1", "asas_sn_id": "CAND-1"},
                {"candidate_id": "CAND-2", "asas_sn_id": "CAND-2"},
            ]),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        save_app_state(conn, "review_last_candidate", "CAND-2")

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    with pytest.raises(review_app.dash.exceptions.PreventUpdate):
        review_app.persist_last_candidate("CAND-1", False)

    with db_connect(db_path) as conn:
        saved = conn.execute(
            "select value from app_state where key='review_last_candidate'"
        ).fetchone()[0]
    assert saved == "CAND-2"

    review_app.persist_last_candidate("CAND-1", True)
    with db_connect(db_path) as conn:
        saved = conn.execute(
            "select value from app_state where key='review_last_candidate'"
        ).fetchone()[0]
    assert saved == "CAND-1"


def test_restore_saved_queue_filters_marks_startup_ready_without_saved_state(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path):
        pass

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    result = review_app.restore_saved_queue_filters(str(db_path))

    assert result[-1]["ready"] is True
    assert result[-1]["restored"] is False


def test_merge_unhydrated_saved_queue_filter_ui_state_restores_select_filters() -> None:
    ui_state = review_app._normalize_saved_queue_filter_ui_state(
        {
            "filter_unreviewed": ["yes"],
        }
    )
    assert ui_state is not None

    restore_state = {
        "ready": True,
        "restored": True,
        "saved_ui_state": {
            "filter_unreviewed": ["yes"],
            "exclude_asassn_var_type": ["EA"],
        },
    }
    text_option_values = tuple([{"label": "Any", "value": "Any"}] for _ in review_app._TEXT_STATES)
    select_option_values = tuple([] for _ in review_app._SELECT_STATES)

    merged = review_app._merge_unhydrated_saved_queue_filter_ui_state(
        ui_state,
        restore_state,
        text_option_values,
        select_option_values,
    )

    assert merged["exclude_asassn_var_type"] == ["EA"]


def test_rehydrate_saved_text_select_filter_values_restores_sidebar_dropdowns() -> None:
    restore_state = {
        "ready": True,
        "restored": True,
        "saved_ui_state": {
            "filter_unreviewed": ["yes"],
            "exclude_asassn_var_type": ["EA", "EB"],
        },
    }
    text_option_values = tuple([{"label": "Any", "value": "Any"}] for _ in review_app._TEXT_STATES)
    select_option_values = []
    for _, fkey in review_app._SELECT_STATES:
        if fkey == "exclude_asassn_var_type":
            select_option_values.append(
                [
                    {"label": "EA", "value": "EA"},
                    {"label": "EB", "value": "EB"},
                    {"label": "DSCT", "value": "DSCT"},
                ]
            )
        else:
            select_option_values.append([])
    text_current_values = tuple("Any" for _ in review_app._TEXT_STATES)
    select_current_values = tuple([] for _ in review_app._SELECT_STATES)

    result = review_app.rehydrate_saved_text_select_filter_values(
        restore_state,
        *text_option_values,
        *select_option_values,
        *text_current_values,
        *select_current_values,
    )

    select_offset = len(review_app._TEXT_STATES)
    asassn_idx = next(
        idx for idx, (_, fkey) in enumerate(review_app._SELECT_STATES)
        if fkey == "exclude_asassn_var_type"
    )

    assert result[select_offset + asassn_idx] == ["EA", "EB"]
    for idx, value in enumerate(result):
        if idx == select_offset + asassn_idx:
            continue
        assert value is review_app.no_update


def test_persist_queue_filters_waits_for_restore_ready(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path):
        pass

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    values = review_app._queue_filter_ui_values_from_state(
        {
            "filter_unreviewed": ["yes"],
            "failed_periodicity_mode": "False",
            "sort_cols": ["dipper_score"],
            "sort_desc": ["yes"],
        }
    )
    assert values is not None
    numeric_bounds = {}
    text_option_values = tuple([{"label": "Any", "value": "Any"}] for _ in review_app._TEXT_STATES)
    select_option_values = tuple([] for _ in review_app._SELECT_STATES)

    with pytest.raises(review_app.dash.exceptions.PreventUpdate):
        review_app.persist_queue_filters(*values, {"ready": False}, numeric_bounds, *text_option_values, *select_option_values)

    review_app.persist_queue_filters(
        *values,
        {
            "ready": True,
            "restored": True,
            "saved_ui_state": {
                "filter_unreviewed": ["yes"],
                "failed_periodicity_mode": "False",
                "sort_cols": ["dipper_score"],
                "sort_desc": ["yes"],
                "exclude_asassn_var_type": ["EA"],
            },
        },
        numeric_bounds,
        *text_option_values,
        *select_option_values,
    )

    with db_connect(db_path) as conn:
        saved = conn.execute(
            "select value from app_state where key='dash_queue_filter_ui_state_v1'"
        ).fetchone()[0]

    payload = json.loads(saved)
    assert payload["filter_unreviewed"] == ["yes"]
    assert payload["failed_periodicity_mode"] == "False"
    assert payload["sort_cols"] == ["dipper_score"]
    assert payload["sort_desc"] == ["yes"]
    assert payload["exclude_asassn_var_type"] == ["EA"]


def test_persist_queue_filters_normalizes_numeric_bounds_before_saving(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path):
        pass

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    values = review_app._queue_filter_ui_values_from_state(
        {
            "filter_unreviewed": ["yes"],
            "min_baseline_mag": 9.0,
            "max_baseline_mag": 16.0,
            "sort_cols": ["dipper_score"],
            "sort_desc": ["yes"],
        }
    )
    assert values is not None

    numeric_bounds = {
        "baseline_mag": {"min": 9.0, "max": 16.0},
    }
    text_option_values = tuple([{"label": "Any", "value": "Any"}] for _ in review_app._TEXT_STATES)
    select_option_values = tuple([] for _ in review_app._SELECT_STATES)

    review_app.persist_queue_filters(
        *values,
        {"ready": True, "restored": False, "saved_ui_state": None},
        numeric_bounds,
        *text_option_values,
        *select_option_values,
    )

    with db_connect(db_path) as conn:
        saved = conn.execute(
            "select value from app_state where key='dash_queue_filter_ui_state_v1'"
        ).fetchone()[0]

    payload = json.loads(saved)
    assert payload["min_baseline_mag"] is None
    assert payload["max_baseline_mag"] is None


def test_restore_startup_candidate_views_explicit_candidate_when_filtered_out(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "CAND-9", "asas_sn_id": "CAND-9"}]),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))
    monkeypatch.setattr(review_app, "INITIAL_CANDIDATE_QUERY", "CAND-9")

    queue_data, current_index, notice, applied = review_app.restore_startup_candidate(
        {"candidate_ids": [], "queue_size": 0, "filter_hash": "demo"},
        False,
    )

    assert queue_data["candidate_ids"] == ["CAND-9"]
    assert current_index == 0
    assert "Opened CAND-9" in notice
    assert applied is True


def test_resolve_import_sources_accepts_multi_mag_bin_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    file_a = results_dir / "lc_events_vetted_13_13.5.parquet"
    file_b = results_dir / "lc_events_vetted_13.5_14.parquet"
    file_a.touch()
    file_b.touch()

    sources = review_app._resolve_import_sources(str(run_dir))

    assert sources == [file_a.resolve(), file_b.resolve()]


def test_update_queue_source_scope_tracks_multiple_import_paths(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "bin_a.parquet"
    second = tmp_path / "bin_b.parquet"
    first.touch()
    second.touch()

    monkeypatch.setattr(review_app, "DB_PATH", str(tmp_path / "review.db"))
    
    # We need to simulate Standalone mode, BUT not hit the 'if PLOT_DIR is None: return ""' branch.
    # The actual bug in the code is that if PLOT_DIR is None it returns "" instead of tracking imports.
    # Wait, the code says:
    # if PLOT_DIR is None:
    #     return ''
    # This means standalone mode *intentionally* does not track imports in the queue scope.
    # We should mock PLOT_DIR to something that DOES NOT look like a run dir, so `_resolve_run_dir_from_plot_dir` returns None,
    # and then the function proceeds to `_queue_scope_from_import_text(import_path)`.
    # `_resolve_run_dir_from_plot_dir` returns `p.parent` if `p.name == "plots"`, or `p` if `p/"plots"` exists.
    # Let's set PLOT_DIR to a random path that doesn't match these heuristics.
    
    monkeypatch.setattr(review_app, "PLOT_DIR", str(tmp_path / "some_random_dir"))

    scope = review_app.update_queue_source_scope(0, f"{first}\n{second}")

    assert scope["source_paths"] == [str(first.resolve()), str(second.resolve())]
    assert "bin_a.parquet" in scope["label"]


def test_query_queue_supports_multiple_source_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    source_a = str((tmp_path / "a.parquet").resolve())
    source_b = str((tmp_path / "b.parquet").resolve())

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "A-1"}]),
            source_path=source_a,
            characterize_before_import=False,
            vet_before_import=False,
        )
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "B-1"}]),
            source_path=source_b,
            characterize_before_import=False,
            vet_before_import=False,
        )
        out = query_queue(conn, filters={"source_paths": [source_a, source_b]})

    assert set(out["candidate_id"].tolist()) == {"A-1", "B-1"}


def test_create_queue_data_dict_includes_filter_provenance(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    source_path = str((tmp_path / "queue_scope.parquet").resolve())

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {"candidate_id": "Q-1", "ltv_slope": 0.1, "failed_any": False},
                    {"candidate_id": "Q-2", "ltv_slope": 0.4, "failed_any": False},
                    {"candidate_id": "Q-3", "ltv_slope": 0.6, "failed_any": True},
                    {"candidate_id": "Q-4", "ltv_slope": 0.8, "failed_any": False},
                ]
            ),
            source_path=source_path,
            characterize_before_import=False,
            vet_before_import=False,
        )

        queue_data = create_queue_data_dict(
            conn,
            {
                "source_path": source_path,
                "require_failed_any_false": True,
                "min_ltv_slope": 0.5,
                "sort_cols": ["candidate_id"],
                "sort_desc": False,
            },
        )

    assert queue_data["candidate_ids"] == ["Q-4"]
    assert queue_data["queue_size"] == 1
    assert queue_data["scope_size"] == 4
    assert queue_data["filtered_out_count"] == 3

    provenance = {item["label"]: item for item in queue_data["filter_provenance"]}
    assert provenance["Require failed_any=False"]["filtered_count"] == 1
    assert provenance["Require failed_any=False"]["remaining_count"] == 3
    assert provenance["ltv_slope >= 0.5"]["filtered_count"] == 2
    assert provenance["ltv_slope >= 0.5"]["remaining_count"] == 2


def test_default_numeric_bounds_are_treated_as_unset_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    bounds = {
        "ltv_slope": {"min": 0.2, "max": 0.2},
        "ltv_dispersion": {"min": 0.4, "max": 0.4},
    }
    raw_numeric = {
        "min_ltv_slope": 0.2,
        "max_ltv_slope": 0.2,
        "min_ltv_dispersion": 0.4,
        "max_ltv_dispersion": 0.4,
    }

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "NUM-1", "ltv_slope": 0.2}]),
            source_path="test://num-1",
            characterize_before_import=False,
            vet_before_import=False,
        )
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "NUM-2", "ltv_dispersion": 0.4}]),
            source_path="test://num-2",
            characterize_before_import=False,
            vet_before_import=False,
        )

        raw_queue = query_queue(
            conn,
            filters={
                "only_unreviewed": True,
                "require_failed_any_false": True,
                **raw_numeric,
            },
        )
        normalized_numeric = review_app._normalize_numeric_filter_inputs(bounds, raw_numeric)
        normalized_queue = query_queue(
            conn,
            filters={
                "only_unreviewed": True,
                "require_failed_any_false": True,
                **normalized_numeric,
            },
        )

    assert raw_queue.empty
    assert normalized_numeric == {
        "min_ltv_slope": None,
        "max_ltv_slope": None,
        "min_ltv_dispersion": None,
        "max_ltv_dispersion": None,
    }
    assert set(normalized_queue["candidate_id"].tolist()) == {"NUM-1", "NUM-2"}


def test_normalize_numeric_filter_inputs_preserves_narrowed_ranges() -> None:
    bounds = {"ltv_slope": {"min": -0.5, "max": 0.5}}

    normalized = review_app._normalize_numeric_filter_inputs(
        bounds,
        {
            "min_ltv_slope": -0.1,
            "max_ltv_slope": 0.4,
        },
    )

    assert normalized == {
        "min_ltv_slope": -0.1,
        "max_ltv_slope": 0.4,
    }


def test_review_app_sidebar_groups_match_filter_schema() -> None:
    assert review_app._SIDEBAR_GROUPS == list(SIDEBAR_GROUPS)


def test_all_scalar_stats_and_ltv_columns_are_filterable() -> None:
    covered = {col for _group, items in SIDEBAR_GROUPS for _kind, col in items}
    stats_cols = {col for col in _COL_NAMES if col.startswith("stats_") and col in (_FLOAT_COLS | _BOOL_COLS)}
    ltv_cols = {col for col in _COL_NAMES if col.startswith("ltv_") and col in (_FLOAT_COLS | _BOOL_COLS)}

    assert sorted(stats_cols - covered) == []
    assert sorted(ltv_cols - covered) == []
    assert "stats_mhps_pn_flag" in _BOOL_COLS


def test_build_stat_rows_includes_ltv_scalars() -> None:
    rows = dict(
        _build_stat_rows(
            {
                "stats_amplitude": 0.12,
                "ltv_trend_slope_snr": 4.5,
                "ltv_season_spearman_rho": 0.81,
                "ltv_vsx_name": "VSX J123",
            },
            pd.DataFrame(),
            set(),
        )
    )

    assert rows["stats_amplitude"] == "0.12"
    assert rows["ltv_trend_slope_snr"] == "4.5"
    assert rows["ltv_season_spearman_rho"] == "0.81"
    assert "ltv_vsx_name" not in rows


def test_import_candidates_callback_accepts_multiple_files(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "review.db"
    src_a = tmp_path / "first.csv"
    src_b = tmp_path / "second.csv"
    pd.DataFrame([{"candidate_id": "M-1"}]).to_csv(src_a, index=False)
    pd.DataFrame([{"candidate_id": "M-2"}]).to_csv(src_b, index=False)

    monkeypatch.setattr(review_app, "DB_PATH", str(db_path))

    status, trigger = review_app.import_candidates_callback(
        1,
        f"{src_a}\n{src_b}",
        [],
        None,
        None,
        None,
        [],
        None,
        [],
        [],
        0,
    )

    with db_connect(db_path) as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    assert trigger == 1
    assert count == 2
    assert "from 2 file(s)" in status


def test_merge_review_db_callback_updates_target_with_newer_review(tmp_path: Path, monkeypatch) -> None:
    source_db = tmp_path / "source.db"
    target_db = tmp_path / "target.db"

    with db_connect(source_db) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "C1"}]),
            source_path="subset://demo",
            characterize_before_import=False,
            vet_before_import=False,
        )
        save_review(
            conn,
            candidate_id="C1",
            interest_score=4,
            event_class="dipper",
            review_pass=2,
            notes="subset newer",
            status="reviewed",
            reviewer="subset",
        )

    with db_connect(target_db) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "C1"}]),
            source_path="master://demo",
            characterize_before_import=False,
            vet_before_import=False,
        )
        conn.execute(
            """
            INSERT INTO reviews (candidate_id, interest_score, event_class, review_pass, notes, status, reviewer, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("C1", 1, "other", 1, "older", "reviewed", "master", "2026-03-11T00:00:00+00:00"),
        )
        conn.commit()

    monkeypatch.setattr(review_app, "DB_PATH", str(source_db))

    status = review_app.merge_review_db_callback(1, str(target_db))

    with db_connect(target_db) as conn:
        row = conn.execute(
            "SELECT interest_score, event_class, review_pass, notes, reviewer FROM reviews WHERE candidate_id=?",
            ("C1",),
        ).fetchone()

    assert "Merged into" in status
    assert "updated=1" in status
    assert row == (4, "dipper", 2, "subset newer", "subset")


def test_arbitrate_harmonic_period_prefers_double_when_base_is_half() -> None:
    true_period = 2.8912
    base_period = true_period / 2.0
    rng = np.random.default_rng(456)
    jd_g = np.sort(rng.uniform(0.0, 1200.0, 420))
    jd_v = np.sort(rng.uniform(2600.0, 3800.0, 420))

    band_dfs = {
        0: _make_band_df(jd_g, true_period=true_period, seed=10),
        1: _make_band_df(jd_v, true_period=true_period, seed=20),
    }
    selected_period, factor, diag = review_app._arbitrate_harmonic_period(
        band_dfs,
        base_period,
        min_period=0.1,
        max_period=4.0,
    )

    assert factor == 2.0
    assert abs(selected_period - true_period) < abs(base_period - true_period)
    assert np.isfinite(diag["objective"])


def test_arbitrate_harmonic_period_respects_search_bounds() -> None:
    true_period = 2.8912
    base_period = true_period / 2.0
    rng = np.random.default_rng(789)
    jd_g = np.sort(rng.uniform(0.0, 1200.0, 420))
    jd_v = np.sort(rng.uniform(2600.0, 3800.0, 420))

    band_dfs = {
        0: _make_band_df(jd_g, true_period=true_period, seed=30),
        1: _make_band_df(jd_v, true_period=true_period, seed=40),
    }
    selected_period, factor, _ = review_app._arbitrate_harmonic_period(
        band_dfs,
        base_period,
        min_period=0.1,
        max_period=2.0,
    )

    # 2x harmonic is out of range, so arbitration should keep the base period.
    assert factor == 1.0
    assert abs(selected_period - base_period) < 1e-10
