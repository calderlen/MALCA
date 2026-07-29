"""Versioned native-observable definitions for synthetic photometry.

Filter throughput and catalog calibration are deliberately represented by
different objects.  A physical response curve can therefore be shared by AB,
Vega, and quoted-monochromatic-flux catalog products without duplicating the
curve cache or pretending that ``Jy`` is an SVO magnitude system.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


PHOTOMETRIC_CALIBRATION_VERSION = "sed-native-observable-v1"

OBSERVABLE_AB_MAG = "ab_mag"
OBSERVABLE_VEGA_MAG = "vega_mag"
OBSERVABLE_QUOTED_FNU = "quoted_fnu"

REFERENCE_FLAT_FNU = "flat_fnu"
REFERENCE_NU_FNU_CONSTANT = "nu_fnu_constant"
REFERENCE_BLACKBODY = "blackbody"

CONTRACT_AB_FLAT_FNU_COUNT_RATIO = "ab_flat_fnu_count_ratio"
CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT = "response_matched_vega_zero_point_count_ratio"
CONTRACT_REFERENCE_SPECTRUM_COUNT_RATIO = "reference_spectrum_count_ratio"

_VALID_OBSERVABLES = {
    OBSERVABLE_AB_MAG,
    OBSERVABLE_VEGA_MAG,
    OBSERVABLE_QUOTED_FNU,
}
_VALID_REFERENCE_SPECTRA = {
    REFERENCE_FLAT_FNU,
    REFERENCE_NU_FNU_CONSTANT,
    REFERENCE_BLACKBODY,
}


@dataclass(frozen=True)
class PhotometricCalibration:
    """Describe how an integrated signal is reported by a catalog.

    ``zero_point_jy`` is used by magnitude systems.  Quoted-flux products are
    instead normalized to a one-Jy reference spectrum at
    ``reference_wavelength_angstrom``.
    """

    calibration_id: str
    observable_kind: str
    reference_spectrum: str = REFERENCE_FLAT_FNU
    reference_wavelength_angstrom: float | None = None
    zero_point_jy: float | None = None
    reference_temperature_k: float | None = None
    version: str = PHOTOMETRIC_CALIBRATION_VERSION
    forward_contract: str = ""

    def __post_init__(self) -> None:
        observable = str(self.observable_kind or "").strip().lower()
        reference = str(self.reference_spectrum or "").strip().lower()
        if observable not in _VALID_OBSERVABLES:
            raise ValueError(f"Unsupported native observable: {self.observable_kind!r}")
        if reference not in _VALID_REFERENCE_SPECTRA:
            raise ValueError(f"Unsupported reference spectrum: {self.reference_spectrum!r}")
        object.__setattr__(self, "observable_kind", observable)
        object.__setattr__(self, "reference_spectrum", reference)
        contract = str(self.forward_contract or "").strip()
        if not contract:
            contract = (
                CONTRACT_AB_FLAT_FNU_COUNT_RATIO
                if observable == OBSERVABLE_AB_MAG
                else CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT
                if observable == OBSERVABLE_VEGA_MAG
                else CONTRACT_REFERENCE_SPECTRUM_COUNT_RATIO
            )
        object.__setattr__(self, "forward_contract", contract)

        zero_point = _finite_positive(self.zero_point_jy)
        reference_wave = _finite_positive(self.reference_wavelength_angstrom)
        temperature = _finite_positive(self.reference_temperature_k)
        object.__setattr__(self, "zero_point_jy", zero_point)
        object.__setattr__(self, "reference_wavelength_angstrom", reference_wave)
        object.__setattr__(self, "reference_temperature_k", temperature)

        if observable in {OBSERVABLE_AB_MAG, OBSERVABLE_VEGA_MAG} and zero_point is None:
            raise ValueError(f"{observable} requires a positive zero_point_jy.")
        if observable == OBSERVABLE_QUOTED_FNU and reference_wave is None:
            raise ValueError("quoted_fnu requires a positive reference wavelength.")
        if reference == REFERENCE_BLACKBODY and temperature is None:
            raise ValueError("A blackbody reference requires reference_temperature_k.")

    @property
    def calibration_hash(self) -> str:
        payload = {
            "calibration_id": self.calibration_id,
            "forward_contract": self.forward_contract,
            "observable_kind": self.observable_kind,
            "reference_spectrum": self.reference_spectrum,
            "reference_temperature_k": self.reference_temperature_k,
            "reference_wavelength_angstrom": self.reference_wavelength_angstrom,
            "version": self.version,
            "zero_point_jy": self.zero_point_jy,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ab_calibration(*, calibration_id: str = "AB/3631Jy") -> PhotometricCalibration:
    """Return the exact AB magnitude definition."""
    return PhotometricCalibration(
        calibration_id=calibration_id,
        observable_kind=OBSERVABLE_AB_MAG,
        reference_spectrum=REFERENCE_FLAT_FNU,
        zero_point_jy=3631.0,
        forward_contract=CONTRACT_AB_FLAT_FNU_COUNT_RATIO,
    )


def response_matched_vega_zero_point_calibration(
    zero_point_jy: float,
    *,
    calibration_id: str,
) -> PhotometricCalibration:
    """Return an exact response-matched Vega count-ratio calibration.

    The scalar must obey ``zero_point_jy = C(Vega) / C(flat 1 Jy)`` for the
    same response and detector convention.  MALCA's model equivalent flux is
    ``C(model) / C(flat 1 Jy)``, so dividing by this zero point cancels the
    flat-reference count rate and gives exactly ``C(model) / C(Vega)``.  No
    blackbody or monochromatic approximation to Vega is introduced.
    """
    return PhotometricCalibration(
        calibration_id=calibration_id,
        observable_kind=OBSERVABLE_VEGA_MAG,
        reference_spectrum=REFERENCE_FLAT_FNU,
        zero_point_jy=zero_point_jy,
        forward_contract=CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT,
    )


def vega_zero_point_calibration(
    zero_point_jy: float,
    *,
    calibration_id: str,
) -> PhotometricCalibration:
    """Backward-compatible alias for a response-matched Vega calibration."""
    return response_matched_vega_zero_point_calibration(
        zero_point_jy,
        calibration_id=calibration_id,
    )


def quoted_fnu_calibration(
    calibration_id: str,
    reference_wavelength_angstrom: float,
    *,
    reference_spectrum: str = REFERENCE_NU_FNU_CONSTANT,
    reference_temperature_k: float | None = None,
) -> PhotometricCalibration:
    """Return a mission-style quoted monochromatic-flux calibration."""
    return PhotometricCalibration(
        calibration_id=calibration_id,
        observable_kind=OBSERVABLE_QUOTED_FNU,
        reference_spectrum=reference_spectrum,
        reference_wavelength_angstrom=reference_wavelength_angstrom,
        reference_temperature_k=reference_temperature_k,
        forward_contract=CONTRACT_REFERENCE_SPECTRUM_COUNT_RATIO,
    )


def mission_quoted_fnu_calibration(
    filter_id: str,
    reference_wavelength_angstrom: float,
) -> PhotometricCalibration:
    """Build the documented broad-IR reference convention used by MALCA.

    IRAC, AKARI, IRAS, and PACS quote fluxes relative to a constant
    ``nu * F_nu`` reference spectrum.  MIPS 24 micron uses a 10,000 K
    blackbody convention.  Unknown missions must be configured explicitly so
    that a flat-Fnu assumption is never silently substituted.
    """
    clean_id = str(filter_id or "").strip()
    folded = clean_id.casefold()
    if folded == "spitzer/mips.24mu":
        return quoted_fnu_calibration(
            f"{clean_id}/quoted-fnu/blackbody-10000K",
            reference_wavelength_angstrom,
            reference_spectrum=REFERENCE_BLACKBODY,
            reference_temperature_k=10000.0,
        )
    constant_nufnu_prefixes = (
        "spitzer/irac.",
        "akari/irc.",
        "akari/fis.",
        "iras/iras.",
        "herschel/pacs.",
    )
    if folded.startswith(constant_nufnu_prefixes):
        return quoted_fnu_calibration(
            f"{clean_id}/quoted-fnu/nuFnu-constant",
            reference_wavelength_angstrom,
            reference_spectrum=REFERENCE_NU_FNU_CONSTANT,
        )
    raise ValueError(f"No mission quoted-Fnu convention is registered for {clean_id!r}.")


def reference_fnu_jy(
    wavelength_angstrom: np.ndarray,
    calibration: PhotometricCalibration,
) -> np.ndarray:
    """Evaluate the calibration reference spectrum, normalized to one Jy.

    For quoted-Fnu calibrations the normalization is one Jy at the mission
    reference wavelength.  Magnitude calibration references are flat in Fnu.
    """
    wave = np.asarray(wavelength_angstrom, dtype=float)
    if np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("Reference-spectrum wavelengths must be finite and positive.")
    kind = calibration.reference_spectrum
    reference_wave = calibration.reference_wavelength_angstrom
    if kind == REFERENCE_FLAT_FNU:
        return np.ones_like(wave)
    if reference_wave is None:
        raise ValueError("The selected reference spectrum requires a reference wavelength.")
    if kind == REFERENCE_NU_FNU_CONSTANT:
        # nu Fnu = constant, hence Fnu is proportional to wavelength.
        return wave / reference_wave
    if kind == REFERENCE_BLACKBODY:
        temperature = float(calibration.reference_temperature_k or 0.0)
        # B_nu is proportional to nu^3 / (exp(h nu / kT) - 1).  Constants
        # cancel in the ratio; hc/k is expressed in Angstrom K.
        hc_over_k_angstrom_k = 1.438776877e8
        exponent = hc_over_k_angstrom_k / (wave * temperature)
        reference_exponent = hc_over_k_angstrom_k / (reference_wave * temperature)
        nu_ratio_cubed = np.power(reference_wave / wave, 3)
        return nu_ratio_cubed * np.expm1(reference_exponent) / np.expm1(exponent)
    raise ValueError(f"Unsupported reference spectrum: {kind}")


def _finite_positive(value: float | None) -> float | None:
    try:
        result = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if result is None or not math.isfinite(result) or result <= 0:
        return None
    return result
