from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_paper_candidate_tables import (
    build_candidate_table,
    build_source_row,
    export_tables,
    latex_escape,
    load_review_candidate_payloads,
    render_longtable,
    render_single_page_table,
    select_distance,
    select_distance_with_uncertainty,
)


def _review_db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE reviews (
                candidate_id TEXT PRIMARY KEY,
                event_class TEXT,
                workflow_status TEXT,
                status TEXT,
                classification_confidence INTEGER,
                updated_at TEXT,
                disposition TEXT,
                duplicate_of TEXT
            );
            """
        )
        candidates = [
            (
                "dip_1",
                json.dumps(
                    {
                        "external_stats": {
                            "ra": 15.0,
                            "dec": -20.0,
                            "phot_g_mean_mag": 13.0,
                            "bp_rp": 1.0,
                            "distance_gspphot": 1000.0,
                            "A_v_3d": 0.2,
                        },
                        "lc_stats": {"stats_photometry_mean_mag": 13.4},
                    }
                ),
            ),
            ("ltv_1", json.dumps({"ra": 30.0, "dec": 5.0})),
            ("ml_unreviewed", json.dumps({"ra": 45.0, "dec": 10.0})),
            ("eb_1", json.dumps({"ra": 60.0, "dec": 15.0})),
        ]
        conn.executemany("INSERT INTO candidates VALUES (?, ?)", candidates)
        conn.executemany(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("dip_1", "dipper", "reviewed", "reviewed", 4, "2026-01-01", "keep", None),
                ("ltv_1", "ltv", "reviewed", "reviewed", 3, "2026-01-02", "keep", None),
                ("ml_unreviewed", "microlensing", "needs_followup", "reviewed", 2, "2026-01-03", "ambiguous", None),
                ("eb_1", "periodic", "reviewed", "reviewed", 1, "2026-01-04", "keep", None),
            ],
        )
    return path


def test_load_review_candidate_payloads_excludes_eb_and_nonreviewed(tmp_path: Path) -> None:
    payloads = load_review_candidate_payloads(_review_db(tmp_path / "review.db"))

    assert [payload["candidate_id"] for payload in payloads] == ["dip_1", "ltv_1"]
    assert payloads[0]["stats_photometry_mean_mag"] == pytest.approx(13.4)
    assert payloads[0]["distance_gspphot"] == pytest.approx(1000.0)


def test_load_review_candidate_payloads_rejects_eb_class(tmp_path: Path) -> None:
    db_path = _review_db(tmp_path / "review.db")

    with pytest.raises(ValueError, match="intentionally excludes eclipsing binaries"):
        load_review_candidate_payloads(db_path, classes=("periodic",))


def test_select_distance_prefers_existing_bailer_jones_value() -> None:
    distance, source = select_distance(
        {
            "bj_r_med_photogeo": 875.0,
            "distance_gspphot": 900.0,
            "parallax": 2.0,
        }
    )

    assert distance == pytest.approx(875.0)
    assert source == "Bailer-Jones photogeometric"


def test_select_distance_does_not_invert_parallax_without_a_posterior() -> None:
    assert select_distance({"parallax": 0.4, "parallax_error": 0.3}) == (None, None)


def test_select_distance_carries_matching_posterior_interval() -> None:
    result = select_distance_with_uncertainty(
        {
            "bj_r_med_photogeo": 875.0,
            "bj_r_lo_photogeo": 810.0,
            "bj_r_hi_photogeo": 960.0,
        }
    )

    assert result["distance_pc"] == pytest.approx(875.0)
    assert result["distance_lower_pc"] == pytest.approx(810.0)
    assert result["distance_upper_pc"] == pytest.approx(960.0)
    assert result["distance_uncertainty_source"] == "Bailer-Jones photogeometric interval"


def test_build_source_row_computes_publication_quantities() -> None:
    row = build_source_row(
        {
            "candidate_id": "dip_1",
            "event_class": "dipper",
            "ra": 15.0,
            "dec": -20.0,
            "stats_photometry_mean_mag": 13.4,
            "phot_g_mean_mag": 13.0,
            "bp_rp": 1.0,
            "distance_gspphot": 1000.0,
            "A_v_3d": 0.2,
            "rv_amplitude_robust": 12.5,
            "ruwe": 1.05,
        }
    )

    assert row["source"] == "J010000-200000"
    assert row["aligned_asassn_mean_mag"] == pytest.approx(13.4)
    assert row["absolute_g_mag"] == pytest.approx(2.8422)
    assert row["bp_rp_mag"] == pytest.approx(0.9174)
    assert row["distance_source"] == "Gaia GSP-Phot"


def test_build_source_row_does_not_treat_gaia_g_as_asassn_mean() -> None:
    row = build_source_row(
        {
            "candidate_id": "dip_1",
            "event_class": "dipper",
            "ra": 15.0,
            "dec": -20.0,
            "phot_g_mean_mag": 13.0,
        }
    )

    assert row["aligned_asassn_mean_mag"] is None
    assert row["gaia_g_mag"] == pytest.approx(13.0)


def test_render_longtable_has_class_sections_and_escaped_text() -> None:
    table = build_candidate_table(
        [
            {"candidate_id": "dip_1", "event_class": "dipper", "ra": 15.0, "dec": -20.0},
            {"candidate_id": "ltv_1", "event_class": "ltv", "ra": 30.0, "dec": 5.0},
            {"candidate_id": "ml_1", "event_class": "microlensing", "ra": 45.0, "dec": 10.0},
        ],
        search_method="MALCA & visual",
    )
    latex = render_longtable(
        table,
        caption="Candidates & properties",
        label="tab:test candidates",
        include_sections=True,
    )

    assert r"\begin{longtable}{lrrlrrrrrr}" in latex
    assert r"\textit{Dipper candidates}" in latex
    assert r"\textit{Long-term-variable candidates}" in latex
    assert r"\textit{Microlensing candidates}" in latex
    assert r"$\langle m_{\rm ASAS\text{-}SN}\rangle$" in latex
    assert r"Mean $g$" not in latex
    assert r"MALCA \& visual" in latex
    assert r"\label{tab:test-candidates}" in latex
    assert latex_escape("a_b") == r"a\_b"
    assert "periodic" not in latex


def test_render_single_page_table_is_nonbreaking() -> None:
    table = build_candidate_table(
        [
            {"candidate_id": "dip_1", "event_class": "dipper", "ra": 15.0, "dec": -20.0},
            {"candidate_id": "ltv_1", "event_class": "ltv", "ra": 30.0, "dec": 5.0},
        ]
    )
    latex = render_single_page_table(
        table,
        caption="Candidate properties",
        label="tab:candidate-properties",
        include_sections=True,
    )

    assert r"\begin{table*}[p]" in latex
    assert r"\begin{adjustbox}{max totalsize={\textwidth}{0.82\textheight},center}" in latex
    assert r"\begin{tabular}{lrrlrrrrrr}" in latex
    assert r"\begin{longtable}" not in latex
    assert r"\endfirsthead" not in latex
    assert r"\textit{Dipper candidates}" in latex


def test_candidate_table_sorts_within_class_by_ra() -> None:
    table = build_candidate_table(
        [
            {"candidate_id": "late", "event_class": "dipper", "ra": 30.0, "dec": 0.0},
            {"candidate_id": "early", "event_class": "dipper", "ra": 10.0, "dec": 0.0},
        ]
    )

    assert table["candidate_id"].tolist() == ["early", "late"]
    assert pd.to_numeric(table["ra_deg"]).tolist() == [10.0, 30.0]


def test_export_tables_writes_three_separate_single_page_tables(tmp_path: Path) -> None:
    table = build_candidate_table(
        [
            {"candidate_id": "dip_1", "event_class": "dipper", "ra": 15.0, "dec": -20.0},
            {"candidate_id": "ltv_1", "event_class": "ltv", "ra": 30.0, "dec": 5.0},
            {"candidate_id": "ml_1", "event_class": "microlensing", "ra": 45.0, "dec": 10.0},
        ]
    )

    written = export_tables(table, tmp_path)
    written_names = {path.name for path in written}

    assert "dipper_candidates_single_page.tex" in written_names
    assert "ltv_candidates_single_page.tex" in written_names
    assert "microlensing_candidates_single_page.tex" in written_names
    assert "candidate_source_properties_three_tables.tex" in written_names
    master = (tmp_path / "candidate_source_properties_three_tables.tex").read_text()
    assert master.count(r"\begin{table*}[p]") == 3
    assert master.count(r"\end{table*}") == 3
