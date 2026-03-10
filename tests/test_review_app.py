from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.review import app as review_app
from malca.review.store import SQLITE_BUSY_TIMEOUT_MS, db_connect, import_candidates


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
