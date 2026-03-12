from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.review import app as review_app
from malca.review.store import SQLITE_BUSY_TIMEOUT_MS, db_connect, import_candidates, query_queue


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


def test_review_db_for_plot_dir_prefers_run_local_db(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "runs" / "demo"
    plot_dir = run_dir / "plots"
    db_path = run_dir / "review" / "review.db"
    plot_dir.mkdir(parents=True)
    with db_connect(db_path):
        pass

    assert review_app._review_db_for_plot_dir(str(plot_dir)) == db_path.resolve()
    assert review_app._resolve_run_dir_from_db_path(db_path) == run_dir.resolve()


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
    assert status.startswith("Showing candidate diagnostics first")


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
    assert status == "Population background loaded."


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
    monkeypatch.setattr(review_app, "PLOT_DIR", None)

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
