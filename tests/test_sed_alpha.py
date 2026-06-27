from __future__ import annotations

from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from malca.enrichment.sed_alpha import (
    compute_sed_alpha_features,
    fit_sed_alpha_for_candidate,
    upsert_sed_alpha_results,
)
from malca.review.store import db_connect, get_candidate_payload, upsert_candidates_frame


def _sed_rows(
    candidate_id: str = "sed-alpha-cand",
    *,
    alpha: float = -0.7,
    waves_micron: tuple[float, ...] = (2.159, 3.4, 12.0, 22.0),
    use_luminosity: bool = True,
) -> pd.DataFrame:
    rows = []
    for idx, wave_micron in enumerate(waves_micron):
        wave_angstrom = wave_micron * 1.0e4
        y = wave_micron ** alpha
        rows.append(
            {
                "candidate_id": candidate_id,
                "source": "Test",
                "band": f"b{idx}",
                "lambda_eff_angstrom": wave_angstrom,
                "lambda_l_lambda": y if use_luminosity else np.nan,
                "flux_lambda": y / wave_angstrom,
                "is_upper_limit": False,
                "is_synthetic": False,
                "quality_flags": "",
                "av_coeff": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_sed_alpha_fit_recovers_known_spectral_index() -> None:
    result = fit_sed_alpha_for_candidate("sed-alpha-cand", _sed_rows(alpha=-0.7))

    assert result["sed_alpha_status"] == "ok"
    assert result["sed_alpha"] == pytest.approx(-0.7, abs=1.0e-12)
    assert result["sed_alpha_class"] == "Class II"
    assert result["sed_alpha_n_points"] == 4


def test_sed_alpha_falls_back_to_lambda_flux_lambda() -> None:
    result = fit_sed_alpha_for_candidate(
        "sed-alpha-cand",
        _sed_rows(alpha=0.45, use_luminosity=False),
    )

    assert result["sed_alpha_status"] == "ok"
    assert result["sed_alpha"] == pytest.approx(0.45, abs=1.0e-12)
    assert result["sed_alpha_class"] == "Class I"


def test_sed_alpha_uses_consistent_flux_scale_when_luminosity_is_incomplete() -> None:
    rows = _sed_rows(alpha=0.2, use_luminosity=False)
    rows.loc[0, "lambda_l_lambda"] = 1.0e50

    result = fit_sed_alpha_for_candidate("sed-alpha-cand", rows)

    assert result["sed_alpha_status"] == "ok"
    assert result["sed_alpha"] == pytest.approx(0.2, abs=1.0e-12)
    assert result["sed_alpha_class"] == "Flat"


def test_sed_alpha_requires_valid_points_and_wavelength_anchors() -> None:
    too_few = _sed_rows(alpha=-0.7)
    too_few.loc[2, "is_upper_limit"] = True
    too_few.loc[3, "is_synthetic"] = True

    assert fit_sed_alpha_for_candidate("cand", too_few)["sed_alpha_status"] == "insufficient_valid_points"
    assert fit_sed_alpha_for_candidate(
        "cand",
        _sed_rows(waves_micron=(2.2, 3.4, 4.6)),
    )["sed_alpha_status"] == "missing_red_anchor"
    assert fit_sed_alpha_for_candidate(
        "cand",
        _sed_rows(waves_micron=(3.4, 12.0, 22.0)),
    )["sed_alpha_status"] == "missing_blue_anchor"


def test_sed_alpha_upsert_updates_review_columns_and_payload(tmp_path) -> None:
    db_path = tmp_path / "review.db"
    candidates = pd.DataFrame(
        [{"candidate_id": "sed-alpha-cand", "teff50": 4200.0, "teff_gspphot": 4300.0}]
    )
    rows = compute_sed_alpha_features(candidates, _sed_rows(alpha=-0.2))

    with closing(db_connect(db_path)) as conn:
        upsert_candidates_frame(conn, candidates)
        assert upsert_sed_alpha_results(conn, rows) == 1
        sql_row = conn.execute(
            "SELECT sed_alpha, sed_alpha_class, sed_alpha_n_points FROM candidates WHERE candidate_id = ?",
            ("sed-alpha-cand",),
        ).fetchone()
        payload = get_candidate_payload(conn, "sed-alpha-cand")

    assert sql_row[0] == pytest.approx(-0.2, abs=1.0e-12)
    assert sql_row[1] == "Flat"
    assert int(sql_row[2]) == 4
    assert payload["sed_alpha"] == pytest.approx(-0.2, abs=1.0e-12)
    assert payload["sed_alpha_class"] == "Flat"
