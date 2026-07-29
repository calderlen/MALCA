from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from malca.enrichment.gaia_binary import (
    build_gaia_binary_evidence,
    normalize_gaia_binary_aliases,
    run_gaia_binary_enrichment,
)
from malca.review.store import init_db, upsert_candidates_frame


def test_normalize_gaia_binary_aliases_preserves_large_ids_and_eb_fields() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A"],
            "source_id_gaia": ["3564313717372918912"],
            "ruwe_gaia": [1.7],
            "pmra_gaia": [2.5],
            "pmdec_gaia": [-1.5],
            "period_gaia_eb_days": [2.25],
            "period_gaia_eb_class": ["TWO_GAUSSIANS"],
        }
    )

    out = normalize_gaia_binary_aliases(frame)

    assert out.loc[0, "gaia_id"] == "3564313717372918912"
    assert out.loc[0, "ruwe"] == 1.7
    assert out.loc[0, "pmra"] == 2.5
    assert out.loc[0, "pmdec"] == -1.5
    assert out.loc[0, "gaia_eb_period"] == 2.25
    assert out.loc[0, "gaia_eb_morph"] == "TWO_GAUSSIANS"


def test_evidence_counts_distinct_families_and_not_copied_nss_eb_twice() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C", "D"],
            "source_id_gaia": [
                "100000000000000001",
                "100000000000000002",
                "100000000000000003",
                "100000000000000004",
            ],
            "period_gaia_eb_days": [2.0, 2.0, pd.NA, pd.NA],
            "period_gaia_eb_match": [True, True, False, False],
            "period_asassn_var_days": [2.0, 3.0, pd.NA, 4.0],
        }
    )
    nss = pd.DataFrame(
        {
            "solution_id": ["1", "2", "3"],
            "source_id": ["100000000000000001", "100000000000000001", "100000000000000002"],
            "nss_solution_type": ["EclipsingBinary", "SB1", "EclipsingBinary"],
            "period": [2.0, 2.0, 2.0],
            "period_error": [0.01, 0.02, 0.01],
            "semi_amplitude_primary": [pd.NA, 35.0, pd.NA],
        }
    )
    gaia_source = pd.DataFrame(
        {
            "gaia_id": [
                "100000000000000001",
                "100000000000000002",
                "100000000000000003",
                "100000000000000004",
            ],
            "ruwe": [1.0, 1.0, 2.0, 1.0],
            "visibility_periods_used": [12, 12, 10, 12],
            "astrometric_excess_noise_sig": [0.0, 0.0, 3.0, 0.0],
        }
    )
    gaia_eb = pd.DataFrame(
        {
            "source_id": [100000000000000001, 100000000000000002],
            "period": [2.0, 2.0],
            "period_error": [0.001, 0.001],
            "var_type": ["TWO_GAUSSIANS", "TWO_GAUSSIANS"],
            "derived_primary_ecl_depth": [0.5, 0.4],
            "derived_secondary_ecl_depth": [0.2, 0.1],
            "derived_primary_ecl_phase": [0.0, 0.0],
            "derived_secondary_ecl_phase": [0.5, 0.5],
        }
    )

    out = build_gaia_binary_evidence(
        candidates,
        nss_solutions=nss,
        gaia_source=gaia_source,
        gaia_eb=gaia_eb,
    ).set_index("candidate_id")

    assert out.loc["A", "gaia_nss_solution_count"] == 2
    assert bool(out.loc["A", "gaia_nss_photometric_duplicate_of_eb"])
    assert out.loc["A", "gaia_binary_evidence_families"] == (
        "gaia_photometric_eb,gaia_spectroscopy,independent_period"
    )
    assert out.loc["A", "gaia_binary_n_evidence_families"] == 3
    assert out.loc["A", "gaia_eb_evidence_level"] == "very_strong"
    assert bool(out.loc["A", "gaia_eb_two_eclipses"])

    # EB catalog + its copied NSS EclipsingBinary solution remain one family.
    assert out.loc["B", "gaia_binary_n_evidence_families"] == 1
    assert bool(out.loc["B", "gaia_binary_period_conflict"])
    assert out.loc["B", "gaia_eb_evidence_level"] == "conflicted"

    # RUWE is supporting evidence only, never an EB/binary catalog confirmation.
    assert bool(out.loc["C", "gaia_astrometric_anomaly_flag"])
    assert out.loc["C", "gaia_binary_evidence_level"] == "supporting"
    assert out.loc["C", "gaia_eb_evidence_level"] == "supporting"

    # An independent period alone is not a conflict without a Gaia/NSS
    # reference period to compare against.
    assert out.loc["D", "gaia_binary_period_n_independent"] == 1
    assert not bool(out.loc["D", "gaia_binary_period_conflict"])
    assert out.loc["D", "gaia_binary_evidence_level"] == "none"


def test_run_gaia_binary_enrichment_writes_run_scoped_sidecars(tmp_path: Path) -> None:
    source_id = "3564313717372918912"
    candidates = pd.DataFrame(
        {
            "candidate_id": ["stv_A"],
            "source_id_gaia": [source_id],
            "period_asassn_var_days": [4.0],
        }
    )
    nss_path = tmp_path / "NssTwoBodyOrbit_1.csv.gz"
    pd.DataFrame(
        {
            "solution_id": ["1"],
            "source_id": [source_id],
            "nss_solution_type": ["SB2"],
            "period": [4.0],
            "period_error": [0.02],
            "semi_amplitude_primary": [25.0],
            "semi_amplitude_secondary": [35.0],
        }
    ).to_csv(nss_path, index=False)
    gaia_source_path = tmp_path / "gaia_dr3_crossmatched.parquet"
    pd.DataFrame(
        {
            "source_id": [source_id],
            "ruwe": [1.1],
            "rv_nb_transits": [12],
            "rv_amplitude_robust": [42.0],
        }
    ).to_parquet(gaia_source_path, index=False)
    evidence_path = tmp_path / "gaia_binary_evidence.parquet"
    nss_output = tmp_path / "gaia_nss_candidate_solutions.parquet"

    evidence, nss_long = run_gaia_binary_enrichment(
        candidates,
        gaia_source_path=gaia_source_path,
        nss_paths=[nss_path],
        eb_cache_dir=tmp_path,
        nss_output=nss_output,
        evidence_output=evidence_path,
        fetch_eb=False,
        show_progress=False,
    )

    assert evidence_path.exists()
    assert nss_output.exists()
    assert evidence.loc[0, "gaia_nss_has_sb2"]
    assert evidence.loc[0, "gaia_binary_evidence_level"] == "strong"
    assert nss_long.loc[0, "candidate_id"] == "stv_A"


def test_run_gaia_binary_enrichment_does_not_scan_nss_for_empty_gaia_cohort(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "gaia_binary_evidence.parquet"
    nss_output = tmp_path / "gaia_nss_candidate_solutions.parquet"

    evidence, nss_long = run_gaia_binary_enrichment(
        pd.DataFrame({"candidate_id": pd.Series(dtype="string"), "gaia_id": pd.Series(dtype="string")}),
        nss_paths=[tmp_path / "missing.csv.gz"],
        nss_output=nss_output,
        evidence_output=evidence_path,
        fetch_eb=False,
        show_progress=False,
    )

    assert evidence.empty
    assert nss_long.empty
    assert evidence_path.exists()
    assert nss_output.exists()


def test_review_ingest_promotes_gaia_pipeline_aliases() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        init_db(conn)
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                {
                    "candidate_id": ["A"],
                    "source_id_gaia": ["3564313717372918912"],
                    "ruwe_gaia": [1.6],
                    "pmra_gaia": [4.0],
                    "pmdec_gaia": [-2.0],
                    "period_gaia_eb_days": [1.25],
                    "period_gaia_eb_class": ["TWO_GAUSSIANS"],
                }
            ),
        )
        row = conn.execute(
            "SELECT gaia_id, ruwe, pmra, pmdec, gaia_eb_period, gaia_eb_morph "
            "FROM candidates WHERE candidate_id = 'A'"
        ).fetchone()

    assert row == ("3564313717372918912", 1.6, 4.0, -2.0, 1.25, "TWO_GAUSSIANS")
