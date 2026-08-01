from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pandas as pd

from scripts import microlensing
from scripts.microlensing import load_review_microlensing_candidate_ids


def test_load_review_microlensing_candidate_ids_uses_review_labels(tmp_path):
    db_path = tmp_path / "review.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY);
            CREATE TABLE reviews (
                candidate_id TEXT,
                event_class TEXT,
                morphology_secondary TEXT,
                morphology_secondary_json TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO candidates(candidate_id) VALUES (?)",
            [
                ("candidate-b",),
                ("candidate-a",),
                ("possible-scalar",),
                ("possible-json",),
                ("not-reviewed",),
            ],
        )
        conn.executemany(
            """
            INSERT INTO reviews(
                candidate_id,
                event_class,
                morphology_secondary,
                morphology_secondary_json
            ) VALUES (?, ?, ?, ?)
            """,
            [
                ("candidate-b", " Microlensing ", None, None),
                ("candidate-a", "microlensing", None, None),
                ("possible-scalar", "brightening_event", "possible_microlensing_event", None),
                (
                    "possible-json",
                    "brightening_event",
                    "single_brightening",
                    '["single_brightening", "possible_microlensing_event"]',
                ),
                ("not-reviewed", "brightening_event", "single_brightening", '["single_brightening"]'),
                ("stale-review-row", "microlensing", None, None),
            ],
        )

    assert load_review_microlensing_candidate_ids(db_path) == [
        "candidate-a",
        "candidate-b",
        "possible-json",
        "possible-scalar",
    ]


def test_fitting_uses_configured_review_connections(monkeypatch, tmp_path):
    db_path = tmp_path / "review.db"
    connection_calls = []

    @contextmanager
    def fake_db_connect(path, *, initialize_if_missing):
        connection_calls.append((path, initialize_if_missing))
        yield object()

    monkeypatch.setattr(microlensing, "db_connect", fake_db_connect)
    monkeypatch.setattr(microlensing, "ensure_review_db_schema", lambda path: None)
    monkeypatch.setattr(microlensing, "infer_plot_dir_from_source", lambda path: tmp_path)
    monkeypatch.setattr(microlensing, "_load_candidate_context", lambda conn, candidate_id, **kwargs: {})
    monkeypatch.setattr(
        microlensing,
        "fit_candidate_context",
        lambda context: {"summary": {"candidate_id": "candidate-a"}},
    )

    microlensing._fit_candidate_context_from_db_task(str(db_path), "candidate-a")
    results_df, _ = microlensing.fit_microlensing_candidates(
        db_path,
        candidate_ids=["candidate-a"],
        show_progress=False,
    )

    assert results_df["candidate_id"].tolist() == ["candidate-a"]
    assert connection_calls == [(db_path.resolve(), False), (db_path.resolve(), False)]


def test_cmd_plot_uses_gaia_photometry_columns(monkeypatch, tmp_path):
    captured = {}

    def capture_figure(fig, *args, **kwargs):
        ax = fig.axes[0]
        captured["texts"] = [text.get_text() for text in ax.texts]
        captured["collections"] = len(ax.collections)
        captured["points"] = len(ax.collections[0].get_offsets()) if ax.collections else 0
        microlensing.plt.close(fig)

    monkeypatch.setattr(microlensing, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(microlensing, "save_publication_figure", capture_figure)
    microlensing._save_microlensing_cmd_plot(
        pd.DataFrame(
            {
                "phot_g_mean_mag": [14.2, 15.1],
                "phot_bp_mean_mag": [14.7, 15.9],
                "phot_rp_mean_mag": [13.8, 14.7],
                "parallax": [2.0, 1.4],
                "mg": [float("nan"), 5.83],
                "quality_tier": ["Gold", "Silver"],
                "quality_score": [0.9, 0.7],
            }
        ),
        tmp_path / "microlensing_cmd.pdf",
    )

    assert not captured["texts"]
    assert captured["collections"] == 1
    assert captured["points"] == 2


def test_grid_uses_the_selected_model_and_event_window(monkeypatch, tmp_path):
    out_path = tmp_path / "microlensing_grid.pdf"
    lightcurve = pd.DataFrame(
        {
            "JD": [2458700.0, 2458900.0, 2458950.0, 2459000.0, 2459050.0, 2459100.0, 2459300.0],
            "mag": [14.0, 14.0, 13.8, 13.5, 13.8, 14.0, 14.0],
            "error": [0.03] * 7,
        }
    )
    monkeypatch.setattr(
        microlensing,
        "_prepare_lightcurve_df",
        lambda path, prefer_g_band: (lightcurve, "g"),
    )
    model_calls = []

    def fake_evaluate_model(model_name, params, jd, t_ref):
        model_calls.append((model_name, float(jd.min()), float(jd.max())))
        return pd.Series([14.0] * len(jd)).to_numpy()

    monkeypatch.setattr(microlensing, "_evaluate_model", fake_evaluate_model)
    candidate_id = "candidate-a"
    microlensing._save_microlensing_candidate_grid_plot(
        pd.DataFrame(
            {
                "candidate_id": [candidate_id],
                "lc_path": ["unused.dat"],
                "band_used": ["g"],
                "quality_tier": ["Suspect"],
                "quality_score": [0.9],
                "best_model": ["fred"],
                "fit_t0_jd": [2459000.0],
                "half_window_days": [80.0],
            }
        ),
        out_path,
        fit_results=[
            {
                "summary": {"candidate_id": candidate_id, "best_model": "fred"},
                "best_seed_result": {
                    "selected_model": "fred",
                    "fits": {
                        "fred": {
                            "success": True,
                            "params": [0.5, 2459000.0, 20.0, 40.0, 14.0, 0.0],
                            "t_ref": 2459000.0,
                        },
                    },
                },
            },
        ],
    )

    assert out_path.is_file()
    assert model_calls == [("fred", 2458880.0, 2459120.0)]
