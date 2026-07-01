from __future__ import annotations

import numpy as np
import pandas as pd

from malca.review.spectrum_linelist import (
    air_to_vacuum_angstrom,
    crossmatch_line_fits_to_apogee,
    format_apogee_species_id,
    load_apogee_synspec_linelist,
)


def test_load_apogee_synspec_linelist_converts_air_nm_to_vacuum_angstrom(tmp_path) -> None:
    linelist_path = tmp_path / "linelist.txt"
    linelist_path.write_text(
        _apogee_line(lambda_air_nm=1500.0, newgf=-1.25, astgf=-1.05, species_id=26.00)
        + _apogee_line(lambda_air_nm=1701.0, newgf=-2.25, astgf=-2.05, species_id=8.00),
        encoding="utf-8",
    )

    table = load_apogee_synspec_linelist(
        linelist_path,
        min_vacuum_angstrom=14900.0,
        max_vacuum_angstrom=15100.0,
    )

    assert len(table) == 1
    assert table.loc[0, "matched_species"] == "Fe I"
    assert table.loc[0, "species_id"] == 26.00
    assert table.loc[0, "lambda_vac_aa"] > 15000.0
    np.testing.assert_allclose(
        table.loc[0, "lambda_vac_aa"],
        air_to_vacuum_angstrom(np.array([15000.0]))[0],
    )


def test_crossmatch_line_fits_to_apogee_keeps_all_matches_and_best_match() -> None:
    reference = pd.DataFrame(
        {
            "lambda_air_nm": [1500.0, 1500.001, 1600.0],
            "lambda_vac_aa": air_to_vacuum_angstrom(np.array([15000.0, 15000.01, 16000.0])),
            "species_id": [26.00, 12.00, 8.00],
            "matched_species": ["Fe I", "Mg I", "O I"],
            "newgf": [-1.0, -2.0, -3.0],
            "astgf": [-1.1, -2.1, -3.1],
        }
    )
    line_fits = pd.DataFrame(
        {
            "line_center": [reference.loc[0, "lambda_vac_aa"] + 0.002, 15500.0],
            "line_type": ["absorption", "emission"],
        }
    )

    result = crossmatch_line_fits_to_apogee(line_fits, reference, velocity_tolerance_kms=30.0)

    assert len(result.line_fits) == 2
    assert result.line_fits.loc[0, "matched_species"] == "Fe I"
    assert result.line_fits.loc[0, "match_count"] == 2
    assert result.line_fits.loc[0, "match_rank"] == 1
    assert result.line_fits.loc[1, "match_count"] == 0
    matched = result.matches[result.matches["detected_line_index"] == 0]
    assert list(matched["match_rank"]) == [1, 2]
    assert set(matched["matched_species"]) == {"Fe I", "Mg I"}


def test_format_apogee_species_id_formats_atomic_species() -> None:
    assert format_apogee_species_id(1.00) == "H I"
    assert format_apogee_species_id(26.01) == "Fe II"
    assert format_apogee_species_id(608.00) == "APOGEE species 608.00"


def _apogee_line(*, lambda_air_nm: float, newgf: float, astgf: float, species_id: float) -> str:
    chars = list(" " * 181)
    _put(chars, 0, 9, f"{lambda_air_nm:9.4f}")
    _put(chars, 18, 25, f"{newgf:7.3f}")
    _put(chars, 34, 41, f"{astgf:7.3f}")
    _put(chars, 46, 54, f"{species_id:8.2f}")
    return "".join(chars).rstrip() + "\n"


def _put(chars: list[str], start: int, end: int, value: str) -> None:
    chars[start:end] = list(value[: end - start])
