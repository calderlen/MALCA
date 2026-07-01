from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np


APOGEE_DR17_SYNSPEC_LINELIST_URL = (
    "https://data.sdss.org/sas/dr17/apogee/spectro/speclib/linelists/synspec/"
    "linelist.20200921.nlte.txt"
)
DEFAULT_MATCH_TOLERANCE_KMS = 30.0
SPEED_OF_LIGHT_KMS = 299792.458

MATCH_COLUMNS = [
    "matched_species",
    "reference_lambda_vac_aa",
    "delta_lambda",
    "delta_v_kms",
    "astgf",
    "match_count",
    "match_rank",
]

_ATOMIC_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
)
_ROMAN_ION_STAGES = ("I", "II", "III", "IV", "V", "VI")


@dataclass(frozen=True)
class LinelistCrossmatchResult:
    line_fits: Any
    matches: Any


def download_apogee_dr17_synspec_linelist(
    cache_path: str | Path,
    *,
    overwrite: bool = False,
    url: str = APOGEE_DR17_SYNSPEC_LINELIST_URL,
) -> Path:
    """Download the APOGEE DR17 Synspec master linelist if it is not cached."""
    path = Path(cache_path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    urlretrieve(url, tmp_path)
    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty APOGEE linelist from {url}")
    tmp_path.replace(path)
    return path


def load_apogee_synspec_linelist(
    linelist_path: str | Path,
    *,
    min_vacuum_angstrom: float | None = None,
    max_vacuum_angstrom: float | None = None,
) -> Any:
    """Load useful APOGEE master-linelist columns and convert air nm to vacuum Å."""
    import pandas as pd

    table = pd.read_fwf(
        linelist_path,
        colspecs=[(0, 9), (18, 25), (34, 41), (46, 54)],
        names=["lambda_air_nm", "newgf", "astgf", "species_id"],
    )
    for column in ("lambda_air_nm", "newgf", "astgf", "species_id"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[np.isfinite(table["lambda_air_nm"]) & np.isfinite(table["species_id"])].copy()
    table["lambda_air_aa"] = table["lambda_air_nm"] * 10.0
    table["lambda_vac_aa"] = air_to_vacuum_angstrom(table["lambda_air_aa"].to_numpy(dtype=float))
    table = table[np.isfinite(table["lambda_vac_aa"])].copy()
    if min_vacuum_angstrom is not None:
        table = table[table["lambda_vac_aa"] >= float(min_vacuum_angstrom)].copy()
    if max_vacuum_angstrom is not None:
        table = table[table["lambda_vac_aa"] <= float(max_vacuum_angstrom)].copy()
    table["matched_species"] = [format_apogee_species_id(value) for value in table["species_id"]]
    return table.sort_values("lambda_vac_aa", kind="mergesort").reset_index(drop=True)


def air_to_vacuum_angstrom(wavelength_air_angstrom: np.ndarray) -> np.ndarray:
    from astropy import units as u
    from specutils.utils.wcs_utils import air_to_vac

    wavelength = np.asarray(wavelength_air_angstrom, dtype=np.float64) * u.AA
    return np.asarray(air_to_vac(wavelength).to_value(u.AA), dtype=np.float64)


def format_apogee_species_id(species_id: float) -> str:
    """Format APOGEE atomic species IDs like 26.00 as Fe I."""
    value = float(species_id)
    if not np.isfinite(value):
        return ""
    atomic_number = int(np.floor(value + 1e-6))
    ion_code = int(round((value - atomic_number) * 100.0))
    if 0 < atomic_number < len(_ATOMIC_SYMBOLS) and 0 <= ion_code < len(_ROMAN_ION_STAGES):
        return f"{_ATOMIC_SYMBOLS[atomic_number]} {_ROMAN_ION_STAGES[ion_code]}"
    return f"APOGEE species {value:.2f}"


def crossmatch_line_fits_to_apogee(
    line_fits: Any,
    reference_linelist: Any,
    *,
    velocity_tolerance_kms: float = DEFAULT_MATCH_TOLERANCE_KMS,
    line_center_column: str = "line_center",
) -> LinelistCrossmatchResult:
    """Crossmatch fitted line centers to every APOGEE transition within a velocity window."""
    import pandas as pd

    lines = line_fits.reset_index(drop=True).copy()
    reference = reference_linelist.sort_values("lambda_vac_aa", kind="mergesort").reset_index(drop=True)
    reference_wavelength = reference["lambda_vac_aa"].to_numpy(dtype=np.float64)
    tolerance = abs(float(velocity_tolerance_kms))

    match_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    for line_index, line_row in lines.iterrows():
        base = line_row.to_dict()
        base["detected_line_index"] = int(line_index)
        center = float(line_row[line_center_column])
        if not np.isfinite(center) or len(reference_wavelength) == 0:
            unmatched = _unmatched_row(base)
            match_rows.append(unmatched)
            enriched_rows.append(unmatched)
            continue

        delta_lambda_max = center * tolerance / SPEED_OF_LIGHT_KMS
        lo = np.searchsorted(reference_wavelength, center - delta_lambda_max, side="left")
        hi = np.searchsorted(reference_wavelength, center + delta_lambda_max, side="right")
        candidates = reference.iloc[lo:hi].copy()
        if candidates.empty:
            unmatched = _unmatched_row(base)
            match_rows.append(unmatched)
            enriched_rows.append(unmatched)
            continue

        candidates["delta_lambda"] = candidates["lambda_vac_aa"] - center
        candidates["delta_v_kms"] = SPEED_OF_LIGHT_KMS * candidates["delta_lambda"] / center
        candidates = candidates.reindex(
            candidates["delta_v_kms"].abs().sort_values(kind="mergesort").index
        )
        match_count = int(len(candidates))
        first_row: dict[str, Any] | None = None
        for rank, candidate in enumerate(candidates.itertuples(index=False), start=1):
            matched = {
                **base,
                "matched_species": candidate.matched_species,
                "reference_lambda_vac_aa": float(candidate.lambda_vac_aa),
                "delta_lambda": float(candidate.delta_lambda),
                "delta_v_kms": float(candidate.delta_v_kms),
                "astgf": float(candidate.astgf) if np.isfinite(candidate.astgf) else np.nan,
                "match_count": match_count,
                "match_rank": rank,
                "reference_lambda_air_nm": float(candidate.lambda_air_nm),
                "reference_species_id": float(candidate.species_id),
                "reference_newgf": float(candidate.newgf) if np.isfinite(candidate.newgf) else np.nan,
            }
            match_rows.append(matched)
            if first_row is None:
                first_row = matched
        if first_row is not None:
            enriched_rows.append(first_row)

    matches = pd.DataFrame(match_rows)
    enriched = pd.DataFrame(enriched_rows)
    for column in MATCH_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = np.nan
        if column not in matches.columns:
            matches[column] = np.nan
    return LinelistCrossmatchResult(line_fits=enriched, matches=matches)


def _unmatched_row(base: dict[str, Any]) -> dict[str, Any]:
    return {
        **base,
        "matched_species": "",
        "reference_lambda_vac_aa": np.nan,
        "delta_lambda": np.nan,
        "delta_v_kms": np.nan,
        "astgf": np.nan,
        "match_count": 0,
        "match_rank": np.nan,
        "reference_lambda_air_nm": np.nan,
        "reference_species_id": np.nan,
        "reference_newgf": np.nan,
    }
